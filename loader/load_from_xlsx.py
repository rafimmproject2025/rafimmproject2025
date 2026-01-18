#!/usr/bin/env python3
import os
import hashlib
from datetime import datetime
from typing import Any, Optional

import pandas as pd
import psycopg

# Config
DB_DSN = os.getenv("DB_DSN", "postgresql://postgres:postgres@db:5432/candidate_news")
NEWS_XLSX = os.getenv("NEWS_XLSX", "/data/Prothom Alo 2024 Election News.xlsx")
CODEBOOK_XLSX = os.getenv("CODEBOOK_XLSX", "/data/Bangladesh_Election_Media_Monitoring_Codebook.xlsx")
PORTAL_NAME = os.getenv("PORTAL_NAME", "Prothom Alo")
PORTAL_LANG = os.getenv("PORTAL_LANG", "bn")


# =========================
# Date normalization helpers
# =========================
from datetime import datetime as _dt, timedelta as _td

# Bangla digits and month names mapping for date parsing
BN_DIGITS = str.maketrans({
    "০": "0", "১": "1", "২": "2", "৩": "3", "৪": "4",
    "৫": "5", "৬": "6", "৭": "7", "৮": "8", "৯": "9",
})

BN_MONTH_MAP = {
    # Jan
    "জানুয়ারি": "January", "জানুয়ারি": "January", "জানুয়ারি": "January",
    # Feb
    "ফেব্রুয়ারি": "February", "ফেব্রুয়ারি": "February",
    # Mar
    "মার্চ": "March",
    # Apr
    "এপ্রিল": "April",
    # May
    "মে": "May",
    # Jun
    "জুন": "June",
    # Jul
    "জুলাই": "July",
    # Aug
    "আগস্ট": "August", "আগস্ট": "August",
    # Sep
    "সেপ্টেম্বর": "September", "সেপ্টেম্বর": "September",
    # Oct
    "অক্টোবর": "October", "অক্টোবর": "October",
    # Nov
    "নভেম্বর": "November",
    # Dec
    "ডিসেম্বর": "December",
}


def _bn_date_to_ascii_en(s: str) -> str:
    # replace Bangla digits
    t = s.translate(BN_DIGITS)
    # replace Bangla month names with English
    for bn, en in BN_MONTH_MAP.items():
        if bn in t:
            t = t.replace(bn, en)
    # trim extra spaces
    return " ".join(t.split())


def _excel_serial_to_dt(serial: float) -> Optional[_dt]:
    try:
        base = _dt(1899, 12, 30)  # Excel epoch (accounts for 1900 leap bug)
        return base + _td(days=float(serial))
    except Exception:
        return None


def _safe_dt(dt: Any) -> Optional[_dt]:
    try:
        if isinstance(dt, pd.Timestamp):
            py = dt.to_pydatetime()
        else:
            py = dt
        # Accept only reasonable years
        if isinstance(py, _dt) and 1990 <= py.year <= 2100:
            return py.replace(microsecond=0)
        return None
    except Exception:
        return None


def normalize_date(val: Any) -> Optional[_dt]:
    """Best-effort normalize various date representations to a sane datetime or None."""
    # 1) Try direct pandas parsing
    try:
        dt = pd.to_datetime(val, errors="coerce", utc=False, dayfirst=False)
        ok = _safe_dt(dt)
        if ok is not None:
            return ok
    except Exception:
        pass

    # 2) Try day-first parsing for common DD-MM-YYYY
    try:
        dt = pd.to_datetime(val, errors="coerce", utc=False, dayfirst=True)
        ok = _safe_dt(dt)
        if ok is not None:
            return ok
    except Exception:
        pass

    # 3) Bangla date normalization (digits + month names)
    try:
        if isinstance(val, str):
            s = val.strip()
            # Heuristic: if string contains Bangla numerals or letters, normalize
            if any("\u0980" <= ch <= "\u09ff" for ch in s):
                s2 = _bn_date_to_ascii_en(s)
                dt = pd.to_datetime(s2, errors="coerce", utc=False, dayfirst=True)
                ok = _safe_dt(dt)
                if ok is not None:
                    return ok
    except Exception:
        pass

    # 4) Numeric paths
    try:
        if isinstance(val, (int, float)):
            x = float(val)
        else:
            s = str(val).strip()
            # Pattern like '48113-11-21 ...' -> treat leading 5+ digits as Excel serial
            import re as _re

            m = _re.match(r"^(\d{5,})-\d{1,2}-\d{1,2}", s)
            if m:
                x = float(m.group(1))
            else:
                # plain number string
                x = float(s) if s.replace(".", "", 1).isdigit() else None
        if x is not None:
            # Excel serial days range heuristic
            if 20000 <= x <= 60000:
                ok = _safe_dt(_excel_serial_to_dt(x))
                if ok is not None:
                    return ok
            # Unix epoch seconds
            if 631152000 <= x <= 4102444800:  # 1990-01-01 to 2100-01-01
                ok = _safe_dt(_dt.utcfromtimestamp(x))
                if ok is not None:
                    return ok
            # Unix epoch milliseconds
            if 631152000000 <= x <= 4102444800000:
                ok = _safe_dt(_dt.utcfromtimestamp(x / 1000.0))
                if ok is not None:
                    return ok
    except Exception:
        pass

    # 5) Give up
    return None


# expose under old name for callers
parse_date = normalize_date


# -------------------------------------------------------------------
# Import helpers
# -------------------------------------------------------------------
def _pick_date_field(row: pd.Series) -> Any:
    try:
        data = row.to_dict()
    except Exception:
        data = dict(row)
    norm = {}
    for k, v in (data or {}).items():
        try:
            norm[str(k).strip().lower()] = v
        except Exception:
            continue

    keys = {
        "date", "published", "published date", "published_date",
        "published_at", "pubdate", "pub_date", "pub date",
        "release_date", "release date", "news_date", "news date",
    }
    for kk in keys:
        if kk in norm:
            val = norm.get(kk)
            if val is not None and str(val).strip():
                return val

    # fallback: first value that looks like a date fragment
    for val in norm.values():
        if val is None:
            continue
        s = str(val).strip()
        if not s:
            continue
        if any(ch.isdigit() for ch in s) and ("/" in s or "-" in s or " " in s):
            return val
    return None


def _ensure_portal(con, name: str, lang: str) -> int:
    row = con.execute("SELECT portal_id FROM portal WHERE name=%s", (name,)).fetchone()
    if row:
        return row[0]
    row = con.execute(
        "INSERT INTO portal(name, language) VALUES(%s,%s) ON CONFLICT (name) DO UPDATE SET language=EXCLUDED.language RETURNING portal_id",
        (name, lang),
    ).fetchone()
    return row[0]


def _hash_url(url: str) -> Optional[str]:
    if not url:
        return None
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def _clean_str(val: Any) -> Optional[str]:
    try:
        s = (val or "").strip()
    except Exception:
        try:
            s = str(val).strip()
        except Exception:
            s = ""
    return s or None


def _parse_keywords(val: Any) -> list[str]:
    if val is None:
        return []
    try:
        # Handle NaN from pandas
        if isinstance(val, float) and pd.isna(val):
            return []
    except Exception:
        pass
    try:
        text = str(val)
    except Exception:
        return []
    items = []
    for part in text.replace("\n", ",").split(","):
        p = part.strip()
        if p:
            items.append(p)
    return items


def load_codebook():
    if not CODEBOOK_XLSX or not os.path.exists(CODEBOOK_XLSX):
        print(f"[warn] CODEBOOK_XLSX not found at {CODEBOOK_XLSX}; skipping codebook load.")
        return
    df = pd.read_excel(CODEBOOK_XLSX, sheet_name=0)
    if df.empty:
        print("[info] no rows in CODEBOOK_XLSX")
        return

    con = psycopg.connect(DB_DSN, autocommit=True)
    inserted_cats = 0
    updated_cats = 0
    inserted_kw = 0
    skipped_rows = 0
    with con:
        for _, r in df.iterrows():
            category = _clean_str(r.get("Category"))
            if not category:
                skipped_rows += 1
                continue
            phase = _clean_str(r.get("Phase"))
            definition = _clean_str(r.get("Definition"))
            subcats = _clean_str(r.get("Subcategories"))
            keywords = _parse_keywords(r.get("Keywords"))

            with con.cursor() as cur:
                cur.execute(
                    "SELECT category_id, definition, subcategories FROM codebook_category WHERE LOWER(category)=LOWER(%s) LIMIT 1",
                    (category,),
                )
                row = cur.fetchone()
                if row:
                    cat_id = row[0]
                    cur.execute(
                        """
                        UPDATE codebook_category
                        SET phase = COALESCE(%s, phase),
                            definition = COALESCE(NULLIF(%s,''), definition),
                            subcategories = COALESCE(NULLIF(%s,''), subcategories)
                        WHERE category_id=%s
                        """,
                        (phase, definition, subcats, cat_id),
                    )
                    if cur.rowcount:
                        updated_cats += cur.rowcount
                else:
                    cur.execute(
                        """
                        INSERT INTO codebook_category(phase, category, definition, subcategories, is_auto, review_needed, created_by)
                        VALUES (%s,%s,%s,%s,FALSE,FALSE,%s)
                        RETURNING category_id
                        """,
                        (phase, category, definition, subcats, "loader:codebook"),
                    )
                    cat_id = cur.fetchone()[0]
                    inserted_cats += 1

                for kw in keywords:
                    try:
                        cur.execute(
                            """
                            INSERT INTO codebook_keyword(category_id, term, weight, lang)
                            VALUES (%s,%s,%s,%s)
                            ON CONFLICT (category_id, term, lang) DO NOTHING
                            """,
                            (cat_id, kw, 1.0, "bn"),
                        )
                        inserted_kw += cur.rowcount
                    except Exception as e:
                        print(f"[warn] keyword insert skipped for '{kw}': {e}")

    print(f"[ok] codebook load complete: categories inserted={inserted_cats}, updated={updated_cats}, keywords added={inserted_kw}, skipped_rows={skipped_rows}")


def load_news():
    df = pd.read_excel(NEWS_XLSX, sheet_name=0)
    if df.empty:
        print("[info] no rows in NEWS_XLSX")
        return

    con = psycopg.connect(DB_DSN, autocommit=True)
    with con:
        pid = _ensure_portal(con, PORTAL_NAME, PORTAL_LANG)
        inserted = 0
        skipped = 0
        for _, r in df.iterrows():
            title = (r.get("Headline") or "").strip()
            content = r.get("Content")
            url = (r.get("Link") or "").strip() or None
            if not title and not content:
                skipped += 1
                continue

            published = normalize_date(_pick_date_field(r))
            if published is None:
                published = datetime.now()

            url_hash = _hash_url(url) if url else None

            try:
                con.execute(
                    """
                    INSERT INTO article(portal_id, title, url, url_hash, published_at,
                                        content_raw, content_clean, lang, status)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (url) DO NOTHING
                    """,
                    (pid, title, url, url_hash, published, content, content, PORTAL_LANG, "queued"),
                )
                inserted += 1
            except Exception as e:
                skipped += 1
                print(f"[warn] insert skipped: {e}")
                continue
        # Safety: ensure no NULL published_at remains
        with con.cursor() as cur:
            cur.execute("UPDATE article SET published_at = COALESCE(published_at, created_at, now()) WHERE published_at IS NULL")
        print(f"[ok] import complete: inserted {inserted}, skipped {skipped}")


if __name__ == "__main__":
    load_codebook()
    load_news()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, math, json, httpx, tempfile, asyncio, subprocess, unicodedata, pandas as pd, hashlib, shutil
from time import monotonic
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta, date

import psycopg
from fastapi import FastAPI, HTTPException, Query, Body, Depends, UploadFile, File, Form
from fastapi.responses import FileResponse
from app.auth import (
    authenticate_user,
    issue_token_for_user,
    ensure_default_admin,
    require_user,
    require_admin,
    list_users as auth_list_users,
    create_user as auth_create_user,
    update_user as auth_update_user,
    record_login,
    _public_user,
)


app = FastAPI(title="News Agent Backend")

@app.get("/api/health")
@app.get("/health")
async def health():
    return {"ok": True}

@app.get("/api/_routes")
def _routes():
    return sorted([r.path for r in app.routes])


@app.post("/api/auth/login")
async def api_auth_login(
    username: str = Body(..., embed=True),
    password: str = Body(..., embed=True),
):
    user = await authenticate_user(username, password)
    if not user:
        raise HTTPException(401, "Invalid username or password")
    token = await issue_token_for_user(user)
    await record_login(user["user_id"])
    return {"access_token": token, "token_type": "bearer", "user": _public_user(user)}


@app.post("/api/auth/refresh")
async def api_auth_refresh(current_user=Depends(require_user)):
    token = await issue_token_for_user(current_user)
    return {"access_token": token, "token_type": "bearer", "user": _public_user(current_user)}


@app.get("/api/auth/me")
async def api_auth_me(current_user=Depends(require_user)):
    return {"user": _public_user(current_user)}


@app.get("/api/admin/users")
async def api_admin_users(current_user=Depends(require_admin)):
    return {"items": await auth_list_users()}


@app.post("/api/admin/users")
async def api_admin_create_user(
    username: str = Body(..., embed=True),
    password: str = Body(..., embed=True),
    role: str = Body("user", embed=True),
    display_name: Optional[str] = Body(None, embed=True),
    is_active: bool = Body(True, embed=True),
    current_user=Depends(require_admin),
):
    user = await auth_create_user(username=username, password=password, role=role, display_name=display_name, is_active=is_active)
    return {"user": user}


@app.patch("/api/admin/users/{user_id}")
async def api_admin_update_user(
    user_id: int,
    display_name: Optional[str] = Body(None, embed=True),
    role: Optional[str] = Body(None, embed=True),
    is_active: Optional[bool] = Body(None, embed=True),
    password: Optional[str] = Body(None, embed=True),
    current_user=Depends(require_admin),
):
    user = await auth_update_user(
        user_id=user_id,
        display_name=display_name,
        role=role,
        is_active=is_active,
        password=password,
    )
    return {"user": user}


@app.post("/api/admin/dedupe/urlhash")
async def api_admin_dedupe_urlhash(current_user=Depends(require_admin)):
    """
    Mark duplicate articles by shared url_hash; keeps lowest article_id as canonical.
    """
    res = await _mark_duplicates_by_urlhash()
    return res


@app.post("/api/admin/upload_xlsx")
async def api_admin_upload_xlsx(
    file: UploadFile = File(...),
    portal_name: Optional[str] = Form(None),
    portal_lang: Optional[str] = Form("bn"),
    current_user=Depends(require_admin),
):
    suffix = (os.path.splitext(file.filename or "")[1] or "").lower()
    if suffix not in {".xlsx", ".xls", ".xlsm"}:
        raise HTTPException(400, "Only Excel files (.xlsx/.xlsm/.xls) are supported")
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(400, f"Could not read uploaded file: {e}")
    if not content:
        raise HTTPException(400, "Uploaded file is empty")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".xlsx") as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        summary = await _ingest_xlsx_file(tmp_path, default_portal=portal_name, portal_lang=portal_lang or "bn")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    if summary.get("_error"):
        raise HTTPException(400, summary.get("_error"))
    return summary

DB_DSN = os.getenv("DB_DSN", "postgresql://postgres:postgres@localhost:5432/candidate_news")
MCP_URL = os.getenv("MCP_URL", "http://mcp:5000/mcp")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
_mcp_client: httpx.AsyncClient | None = None
_candidate_ref_cols: Optional[dict] = None
LLM_TAG_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Date helpers for filtering article published_at
def _parse_date(val: Optional[str]) -> Optional[date]:
    if not val:
        return None
    try:
        return datetime.fromisoformat(val).date()
    except Exception:
        return None


def _apply_date_filters(where: list[str], params: list[Any], start_dt: Optional[date], end_dt: Optional[date]):
    if start_dt:
        where.append("a.published_at >= %s")
        params.append(datetime.combine(start_dt, datetime.min.time()))
    if end_dt:
        where.append("a.published_at < %s")
        params.append(datetime.combine(end_dt + timedelta(days=1), datetime.min.time()))


try:
    from openai import OpenAI
    _oa = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception as e:
    _oa = None
    print("[warn] OpenAI client not initialized:", e)

async def _get_conn():
    return await psycopg.AsyncConnection.connect(DB_DSN)


async def _ensure_article_dimensions():
    """
    Add missing article columns used by auto-tagging/stats/dup-detection.
    Safe to run repeatedly.
    """
    async with await _get_conn() as con:
        await con.execute(
            """
            ALTER TABLE article
              ADD COLUMN IF NOT EXISTS party TEXT,
              ADD COLUMN IF NOT EXISTS candidate TEXT,
              ADD COLUMN IF NOT EXISTS region TEXT,
              ADD COLUMN IF NOT EXISTS auto_used BOOLEAN NOT NULL DEFAULT FALSE,
              ADD COLUMN IF NOT EXISTS is_duplicate BOOLEAN NOT NULL DEFAULT FALSE;
            """
        )
        await con.execute("CREATE INDEX IF NOT EXISTS idx_article_party ON article(LOWER(party));")
        await con.execute("CREATE INDEX IF NOT EXISTS idx_article_candidate ON article(LOWER(candidate));")
        await con.execute("CREATE INDEX IF NOT EXISTS idx_article_region ON article(LOWER(region));")
        await con.execute("CREATE INDEX IF NOT EXISTS idx_article_auto_used ON article(auto_used);")
        try:
            await con.execute(
                """
                ALTER TABLE article
                  ADD COLUMN IF NOT EXISTS dup_of BIGINT REFERENCES article(article_id);
                """
            )
        except Exception:
            pass
        await con.execute("CREATE INDEX IF NOT EXISTS idx_article_is_duplicate ON article(is_duplicate);")
        await con.execute("CREATE INDEX IF NOT EXISTS idx_article_dup_of ON article(dup_of);")


async def _ensure_auto_category_tables():
    """
    Ensure auto_category pipeline tables exist so dashboard actions don't fail on fresh DBs.
    Mirrors db/init/007_auto_categories.sql.
    """
    async with await _get_conn() as con:
        await con.execute(
            """
            CREATE TABLE IF NOT EXISTS auto_category (
              id BIGSERIAL PRIMARY KEY,
              label TEXT NOT NULL,
              top_terms TEXT[] DEFAULT ARRAY[]::TEXT[],
              size INT DEFAULT 0,
              model_name TEXT,
              algo TEXT,
              params_json JSONB DEFAULT '{}'::jsonb,
              created_at TIMESTAMPTZ DEFAULT now()
            );
            CREATE TABLE IF NOT EXISTS auto_category_article (
              category_id BIGINT NOT NULL REFERENCES auto_category(id) ON DELETE CASCADE,
              article_id  BIGINT NOT NULL REFERENCES article(article_id) ON DELETE CASCADE,
              score DOUBLE PRECISION,
              rank INT,
              PRIMARY KEY (category_id, article_id)
            );
            CREATE TABLE IF NOT EXISTS auto_category_centroid (
              category_id BIGINT PRIMARY KEY REFERENCES auto_category(id) ON DELETE CASCADE,
              model_name TEXT,
              dim INT,
              vector DOUBLE PRECISION[],
              updated_at TIMESTAMPTZ DEFAULT now()
            );
            CREATE TABLE IF NOT EXISTS article_embedding (
              article_id BIGINT NOT NULL REFERENCES article(article_id) ON DELETE CASCADE,
              model_name TEXT NOT NULL,
              dim INT,
              vector DOUBLE PRECISION[],
              PRIMARY KEY (article_id, model_name)
            );
            CREATE INDEX IF NOT EXISTS idx_auto_category_size ON auto_category(size DESC);
            CREATE INDEX IF NOT EXISTS idx_auto_cat_article_article ON auto_category_article(article_id);
            """
        )

@app.on_event("startup")
async def _startup_bootstrap():
    await _ensure_article_dimensions()
    await _ensure_auto_category_tables()
    await ensure_default_admin()

# --------- minimal helpers (fetch, save labels) ----------
async def _fetch_categories() -> List[Dict[str, Any]]:
    """
    Return categories with optional auto metadata (top_terms/model/algo/is_auto).
    This prefers a case-insensitive match to auto_category.label when present.
    """
    sql = """
        SELECT
          c.category,
          c.is_auto,
          COALESCE(ac.top_terms, ARRAY[]::TEXT[]) AS top_terms,
          ac.model_name,
          ac.algo
        FROM codebook_category c
        LEFT JOIN LATERAL (
          SELECT top_terms, model_name, algo
          FROM auto_category ac
          WHERE LOWER(ac.label) = LOWER(c.category)
          ORDER BY ac.id DESC
          LIMIT 1
        ) ac ON TRUE
        ORDER BY c.category
    """
    items: List[Dict[str, Any]] = []
    async with await _get_conn() as con:
        cur = await con.execute(sql)
        for row in await cur.fetchall():
            items.append({
                "category": row[0],
                "is_auto": bool(row[1]),
                "top_terms": list(row[2] or []),
                "model_name": row[3],
                "algo": row[4],
            })
    return items

async def _fetch_candidates(limit: int = 500) -> List[Dict[str, Any]]:
    async with await _get_conn() as con:
        cur = await con.execute(
            """
            SELECT candidate, COUNT(*) AS total
            FROM article
            WHERE candidate IS NOT NULL AND NULLIF(TRIM(candidate), '') IS NOT NULL
            GROUP BY candidate
            ORDER BY total DESC, candidate
            LIMIT %s
            """,
            (int(limit),),
        )
        rows = await cur.fetchall()
    return [{"candidate": r[0], "total": int(r[1])} for r in rows]

async def _existing_candidates(limit: int = 1000) -> List[str]:
    rows = await _fetch_candidates(limit=limit)
    return [r["candidate"] for r in rows if r.get("candidate")]

async def _candidate_ref_columns() -> dict:
    global _candidate_ref_cols
    if _candidate_ref_cols is not None:
        return _candidate_ref_cols
    cols = set()
    async with await _get_conn() as con:
        cur = await con.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'candidate_ref'
            """
        )
        cols = {r[0].lower() for r in await cur.fetchall()}
    _candidate_ref_cols = {
        "name_bn": "name_bn" in cols,
        "seat": "seat" in cols,
    }
    return _candidate_ref_cols

async def _candidate_refs(limit: int = 2000) -> List[Dict[str, Any]]:
    cols = await _candidate_ref_columns()
    has_bn = cols.get("name_bn", False)
    has_seat = cols.get("seat", False)
    select_fields = "candidate_id, name"
    if has_bn:
        select_fields += ", name_bn"
    if has_seat:
        select_fields += ", seat"
    select_fields += ", party, aliases"
    async with await _get_conn() as con:
        cur = await con.execute(
            """
            SELECT {fields}
            FROM candidate_ref
            WHERE is_active IS TRUE
            ORDER BY COALESCE({order_by}, name)
            LIMIT %s
            """.format(fields=select_fields, order_by=("name_bn" if has_bn else "name")),
            (int(limit),),
        )
        rows = await cur.fetchall()
    items = []
    for r in rows:
        idx = 0
        cid = r[idx]; idx += 1
        name = r[idx]; idx += 1
        name_bn = r[idx] if has_bn else None
        idx += (1 if has_bn else 0)
        seat = r[idx] if has_seat else None
        idx += (1 if has_seat else 0)
        party = r[idx]; idx += 1
        aliases = list(r[idx] or [])
        items.append({
            "candidate_id": cid,
            "name": name,
            "name_bn": name_bn,
            "party": party,
            "seat": seat,
            "aliases": aliases,
        })
    return items

def _norm_name(val: Optional[str]) -> Optional[str]:
    if val is None:
        return None
    return val.strip().lower() or None

async def _candidate_bn_map(limit: int = 5000) -> Dict[str, str]:
    refs = await _candidate_refs(limit=limit)
    m: Dict[str, str] = {}
    for r in refs:
        name = _norm_name(r.get("name"))
        name_bn = r.get("name_bn") or r.get("name")
        if name:
            m[name] = name_bn
        for alias in r.get("aliases") or []:
            alias_norm = _norm_name(alias)
            if alias_norm:
                m[alias_norm] = name_bn
    return m

async def _upsert_candidate_ref(name: str, party: Optional[str], aliases: List[str],
                                name_bn: Optional[str], seat: Optional[str]) -> Dict[str, Any]:
    cols = await _candidate_ref_columns()
    has_bn = cols.get("name_bn", False)
    has_seat = cols.get("seat", False)
    async with await _get_conn() as con:
        if has_bn or has_seat:
            row = await (await con.execute(
                """
                INSERT INTO candidate_ref(name, name_bn, party, seat, aliases)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (name) DO UPDATE
                SET name_bn = COALESCE(EXCLUDED.name_bn, candidate_ref.name_bn),
                    party = COALESCE(EXCLUDED.party, candidate_ref.party),
                    seat = COALESCE(EXCLUDED.seat, candidate_ref.seat),
                    aliases = CASE WHEN array_length(candidate_ref.aliases,1) IS NULL OR array_length(candidate_ref.aliases,1)=0
                                   THEN EXCLUDED.aliases ELSE candidate_ref.aliases END,
                    is_active = TRUE
                RETURNING candidate_id, name, name_bn, party, seat, aliases
                """,
                (name, name_bn, party, seat, aliases or []),
            )).fetchone()
        else:
            row = await (await con.execute(
                """
                INSERT INTO candidate_ref(name, party, aliases)
                VALUES (%s, %s, %s)
                ON CONFLICT (name) DO UPDATE
                SET party = COALESCE(EXCLUDED.party, candidate_ref.party),
                    aliases = CASE WHEN array_length(candidate_ref.aliases,1) IS NULL OR array_length(candidate_ref.aliases,1)=0
                                   THEN EXCLUDED.aliases ELSE candidate_ref.aliases END,
                    is_active = TRUE
                RETURNING candidate_id, name, party, aliases
                """,
                (name, party, aliases or []),
            )).fetchone()
            row = (row[0], row[1], None, row[2], None, row[3])  # align tuple
    return {
        "candidate_id": row[0],
        "name": row[1],
        "name_bn": row[2],
        "party": row[3],
        "seat": row[4],
        "aliases": list(row[5] or []),
    }

async def _candidate_ref_lookup(name: str) -> Optional[Dict[str, Any]]:
    target = (name or "").strip().lower()
    if not target:
        return None
    refs = await _candidate_refs(limit=5000)
    for r in refs:
        candidates = [r.get("name"), r.get("name_bn")] + list(r.get("aliases") or [])
        if any(target == (c or "").strip().lower() for c in candidates if c):
            return r
    return None

async def _fetch_article(aid: int) -> Dict[str, Any]:
    async with await _get_conn() as con:
        cur = await con.execute("""
            SELECT article_id, title, content_clean, content_raw, lang
            FROM article WHERE article_id=%s
        """, (aid,))
        row = await cur.fetchone()
    if not row:
        raise HTTPException(404, f"article_id={aid} not found")
    return {"article_id": row[0], "title": row[1], "content_clean": row[2], "content_raw": row[3], "lang": row[4] or "bn"}

async def _get_article_status(aid: int) -> Optional[str]:
    async with await _get_conn() as con:
        cur = await con.execute("SELECT status FROM article WHERE article_id=%s", (aid,))
        row = await cur.fetchone()
    return (row[0] if row else None)

async def _save_labels(aid: int, labels: List[Dict[str, Any]], model_name: str, source: str):
    source = (source or "ml").lower()
    async with await _get_conn() as con:
        async with con.transaction():
            run_id = (await (await con.execute(
                "INSERT INTO classification_run(model) VALUES(%s) RETURNING run_id", (model_name,)
            )).fetchone())[0]
            for lab in labels:
                await con.execute("""
                    INSERT INTO article_label(article_id, category_id, score, source, run_id, is_primary)
                    SELECT %s, category_id, %s, %s::label_source, %s, %s
                    FROM codebook_category WHERE category=%s
                    ON CONFLICT (article_id, category_id, source) DO UPDATE
                    SET score=EXCLUDED.score, run_id=EXCLUDED.run_id, is_primary=EXCLUDED.is_primary
                """, (aid, float(lab.get("score", 1.0) or 1.0), source, run_id, bool(lab.get("is_primary", False)), lab["category"]))
            await con.execute("UPDATE article SET status='classified' WHERE article_id=%s", (aid,))

def _merge_labels(*label_lists: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    bycat: Dict[str, Dict[str, Any]] = {}
    for labs in label_lists:
        for lab in labs or []:
            cat = lab.get("category")
            if not cat:
                continue
            score = float(lab.get("score", 0) or 0)
            src = (lab.get("source") or lab.get("mode") or "rule").lower()
            cur = bycat.get(cat)
            if cur is None or score > float(cur.get("score", 0) or 0):
                bycat[cat] = {"category": cat, "score": score, "source": src, "is_primary": False}
    if bycat:
        max(bycat.values(), key=lambda x: x.get("score", 0)).update({"is_primary": True})
    return list(bycat.values())


# -------------------------------------------------------------------
# Duplicate detection helpers
# -------------------------------------------------------------------

async def _mark_duplicates_by_urlhash() -> Dict[str, Any]:
    """
    Mark duplicates by shared url_hash; keep lowest article_id as canonical.
    """
    sql = """
        WITH dups AS (
          SELECT url_hash, MIN(article_id) AS keep_id, ARRAY_AGG(article_id) AS ids
          FROM article
          WHERE url_hash IS NOT NULL
          GROUP BY url_hash
          HAVING COUNT(*) > 1
        ),
        todo AS (
          SELECT unnest(array_remove(ids, keep_id)) AS article_id,
                 keep_id
          FROM dups
        ),
        upd AS (
          UPDATE article a
          SET is_duplicate = TRUE,
              dup_of = t.keep_id
          FROM todo t
          WHERE a.article_id = t.article_id
          RETURNING a.article_id, t.keep_id
        )
        SELECT
          (SELECT COUNT(*) FROM dups) AS groups,
          (SELECT COUNT(*) FROM upd) AS updated;
    """
    async with await _get_conn() as con:
        cur = await con.execute(sql)
        row = await cur.fetchone()
    return {"groups": int(row[0] or 0), "updated": int(row[1] or 0)}


# -------------------------------------------------------------------
# XLSX ingest helpers (used by admin upload endpoint)
# -------------------------------------------------------------------

def _clean_str(val: Any) -> Optional[str]:
    try:
        s = (val or "").strip()
    except Exception:
        try:
            s = str(val).strip()
        except Exception:
            return None
    return s or None


def _clean_content(val: Any) -> Optional[str]:
    try:
        if isinstance(val, float) and pd.isna(val):
            return None
    except Exception:
        pass
    if val is None:
        return None
    try:
        text = str(val)
    except Exception:
        return None
    text = text.strip()
    return text or None


def _hash_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def _pick_date_field_dict(data: Dict[str, Any]) -> Any:
    keys = {
        "date", "published", "published date", "published_date",
        "published_at", "pubdate", "pub_date", "pub date",
        "release_date", "release date", "news_date", "news date",
    }
    for k in keys:
        if k in data:
            val = data.get(k)
            if val is not None and str(val).strip():
                return val
    for val in data.values():
        if val is None:
            continue
        s = str(val).strip()
        if not s:
            continue
        if any(ch.isdigit() for ch in s) and ("/" in s or "-" in s or " " in s):
            return val
    return None


def _excel_serial_to_dt(serial: float) -> Optional[datetime]:
    try:
        base = datetime(1899, 12, 30)
        return base + timedelta(days=float(serial))
    except Exception:
        return None


def _safe_dt_upload(dt: Any) -> Optional[datetime]:
    try:
        if isinstance(dt, pd.Timestamp):
            py = dt.to_pydatetime()
        else:
            py = dt
        if isinstance(py, datetime) and 1990 <= py.year <= 2100:
            return py.replace(microsecond=0)
    except Exception:
        pass
    return None


def normalize_date_upload(val: Any) -> Optional[datetime]:
    for dayfirst in (False, True):
        try:
            dt = pd.to_datetime(val, errors="coerce", utc=False, dayfirst=dayfirst)
            ok = _safe_dt_upload(dt)
            if ok is not None:
                return ok
        except Exception:
            pass
    try:
        if isinstance(val, (int, float)):
            x = float(val)
        else:
            s = str(val).strip()
            x = float(s) if s.replace(".", "", 1).isdigit() else None
    except Exception:
        x = None
    if x is not None:
        if 20000 <= x <= 60000:
            ok = _safe_dt_upload(_excel_serial_to_dt(x))
            if ok is not None:
                return ok
        if 631152000 <= x <= 4102444800:
            ok = _safe_dt_upload(datetime.utcfromtimestamp(x))
            if ok is not None:
                return ok
        if 631152000000 <= x <= 4102444800000:
            ok = _safe_dt_upload(datetime.utcfromtimestamp(x / 1000.0))
            if ok is not None:
                return ok
    return None


async def _ensure_portal_async(con, name: Optional[str], lang: Optional[str]) -> Optional[int]:
    if not name:
        return None
    row = await (await con.execute(
        "SELECT portal_id FROM portal WHERE LOWER(name)=LOWER(%s) LIMIT 1",
        (name,),
    )).fetchone()
    if row:
        return row[0]
    row = await (await con.execute(
        """
        INSERT INTO portal(name, language, is_active)
        VALUES (%s, %s, TRUE)
        ON CONFLICT (name) DO UPDATE SET language = COALESCE(EXCLUDED.language, portal.language)
        RETURNING portal_id
        """,
        (name, lang or "bn"),
    )).fetchone()
    return row[0] if row else None


async def _ingest_xlsx_file(path: str, *, default_portal: Optional[str], portal_lang: str = "bn") -> Dict[str, Any]:
    default_portal = _clean_str(default_portal)
    try:
        df = pd.read_excel(path, sheet_name=0)
    except Exception as e:
        return {"_error": f"read_failed: {e}"}
    if df.empty:
        return {"rows": 0, "inserted": 0, "skipped": 0, "warnings": ["empty_sheet"]}

    inserted = 0
    skipped = 0
    warnings: List[str] = []

    async with await _get_conn() as con:
        await con.set_autocommit(True)
        for idx, row in df.iterrows():
            try:
                data: Dict[str, Any] = {}
                for k in df.columns:
                    try:
                        data[str(k).strip().lower()] = row.get(k)
                    except Exception:
                        continue

                title = _clean_str(data.get("headline") or data.get("title"))
                content = _clean_content(data.get("content") or data.get("body") or data.get("text"))
                url = _clean_str(data.get("link") or data.get("url"))
                portal_name = _clean_str(data.get("newspaper") or data.get("portal") or data.get("source") or default_portal)
                lang = _clean_str(data.get("lang")) or portal_lang or "bn"

                if not title and not content:
                    skipped += 1
                    continue

                published_val = _pick_date_field_dict(data)
                published_at = normalize_date_upload(published_val) or datetime.now()
                pid = await _ensure_portal_async(con, portal_name, lang)

                cur = await con.execute(
                    """
                    INSERT INTO article(portal_id, title, url, url_hash, published_at,
                                        content_raw, content_clean, lang, status)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (url) DO NOTHING
                    """,
                    (pid, title, url, _hash_url(url), published_at, content, content, lang or "bn", "queued"),
                )
                if cur.rowcount and cur.rowcount > 0:
                    inserted += int(cur.rowcount)
                else:
                    skipped += 1
            except Exception as e:
                skipped += 1
                warnings.append(f"row {idx}: {e}")
                continue

    return {"rows": int(len(df)), "inserted": inserted, "skipped": skipped, "warnings": warnings}


async def _get_mcp_client() -> httpx.AsyncClient:
    global _mcp_client
    if _mcp_client is None or _mcp_client.is_closed:
        _mcp_client = httpx.AsyncClient()
    return _mcp_client

async def _mcp_call(name: str, arguments: Dict[str, Any] | None = None, timeout: int = 60) -> Dict[str, Any]:
    try:
        payload = {"jsonrpc": "2.0", "id": "1", "method": "tools/call", "params": {"name": name, "arguments": arguments or {}}}
        client = await _get_mcp_client()
        r = await client.post(MCP_URL, json=payload, timeout=timeout)
        r.raise_for_status()
        return (r.json().get("result") or {})
    except Exception as e:
        return {"_error": f"mcp_failed: {e}"}

def _llm_extract_candidate(title: str, content: str) -> Dict[str, Any]:
    """
    Use an LLM to extract the primary candidate mention.
    Returns {"candidate": str|None, "party": str|None, "seat": str|None}
    """
    if _oa is None:
        return {"_error": "llm_not_configured"}
    text = f"Title: {title or ''}\n\nContent: {content or ''}"
    text = text[:8000]  # keep prompt bounded
    prompt = (
        "Identify the main election candidate mentioned in the news. "
        "Return JSON with keys: candidate (Bangla name if present, else as-is), "
        "party (if stated), seat (constituency, area, or district if stated). "
        "If no candidate is clearly mentioned, return an empty JSON object."
    )
    try:
        resp = _oa.chat.completions.create(
            model=LLM_TAG_MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = (resp.choices[0].message.content or "{}").strip()
        data = json.loads(raw)
        return {
            "candidate": (data.get("candidate") or "").strip() or None,
            "party": (data.get("party") or "").strip() or None,
            "seat": (data.get("seat") or "").strip() or None,
        }
    except Exception as e:
        return {"_error": f"llm_extract_failed: {e}"}

# --------- (… your classify endpoints unchanged) ----------
#   -- keeping your previous classify, exports, generated, etc.
#   -- omitted here for brevity; keep the ones you already run successfully
#   -- nothing in those depends on the auto-category logic

# =========================================================
# Auto-Category endpoints (pipeline integration + breakdown)
# =========================================================

AUTO_PIPELINE_SCRIPT = os.getenv("AUTO_PIPELINE_SCRIPT", "auto_category_pipeline.py")
EMB_MODEL = os.getenv("EMB_MODEL", "intfloat/multilingual-e5-base")


try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None
_emb_model = None

def _prefix_for_model(name: str) -> str:
    return "passage: " if ("e5" in (name or "").lower()) else ""

async def _get_emb_model():
    global _emb_model
    if SentenceTransformer is None:
        return None
    if _emb_model is None:
        _emb_model = await asyncio.to_thread(SentenceTransformer, EMB_MODEL)
    return _emb_model

def _norm(s: str) -> str:
    return unicodedata.normalize("NFKC", (s or "")).lower()

_SPACE_RX = re.compile(r"\s+", flags=re.UNICODE)

def _snippet(text: str, term: str, window: int = 60) -> Optional[str]:
    if not text or not term:
        return None
    tlo, slo = term.lower(), text.lower()
    i = slo.find(tlo)
    if i < 0:
        return None
    start = max(0, i - window // 2)
    end = min(len(text), i + len(term) + window // 2)
    s = text[start:end].replace("\n", " ")
    return ("..." if start > 0 else "") + s + ("..." if end < len(text) else "")

def _cosine(a: List[float], b: List[float]) -> float:
    import math as _m
    na = _m.sqrt(sum(x*x for x in a)) or 1e-9
    nb = _m.sqrt(sum(x*x for x in b)) or 1e-9
    return float(sum(x*y for x, y in zip(a, b)) / (na * nb))

# -----------------------------
# Geography helpers
# -----------------------------
_REGION_DIVISION_MAP = [
    # Dhaka division
    ("Dhaka", ["dhaka", "ঢাকা"]),
    ("Dhaka", ["gazipur", "গাজীপুর"]),
    ("Dhaka", ["narayanganj", "নারায়ণগঞ্জ", "নারায়নগঞ্জ"]),
    ("Dhaka", ["narsingdi", "নরসিংদী"]),
    ("Dhaka", ["kishoreganj", "কিশোরগঞ্জ"]),
    ("Dhaka", ["manikganj", "মানিকগঞ্জ"]),
    ("Dhaka", ["munshiganj", "মুন্সীগঞ্জ"]),
    ("Dhaka", ["faridpur", "ফরিদপুর"]),
    ("Dhaka", ["gopalganj", "গোপালগঞ্জ"]),
    ("Dhaka", ["madaripur", "মাদারীপুর"]),
    ("Dhaka", ["rajbari", "রাজবাড়ী", "রাজবাড়ী"]),
    ("Dhaka", ["shariatpur", "শরীয়তপুর", "শরীয়তপুর"]),
    ("Dhaka", ["tangail", "টাঙ্গাইল"]),
    # Mymensingh division
    ("Mymensingh", ["mymensingh", "ময়মনসিংহ", "ময়মনসিংহ"]),
    ("Mymensingh", ["jamalpur", "জামালপুর"]),
    ("Mymensingh", ["netrokona", "নেত্রকোনা"]),
    ("Mymensingh", ["sherpur", "শেরপুর"]),
    # Chattogram division
    ("Chattogram", ["chattogram", "chattagram", "chittagong", "চট্টগ্রাম"]),
    ("Chattogram", ["coxs", "cox", "কক্সবাজার"]),
    ("Chattogram", ["rangamati", "রাঙ্গামাটি", "রাংগামাটি"]),
    ("Chattogram", ["khagrachari", "খাগড়াছড়ি", "খাগড়াছড়ি", "খাগড়াছড়ি"]),
    ("Chattogram", ["bandarban", "বান্দরবান"]),
    ("Chattogram", ["feni", "ফেনী"]),
    ("Chattogram", ["chandpur", "চাঁদপুর"]),
    ("Chattogram", ["noakhali", "নোয়াখালী", "নোয়াখালী"]),
    ("Chattogram", ["lakshmipur", "laxmipur", "লক্ষ্মীপুর", "লক্ষ্মীপুর"]),
    ("Chattogram", ["cumilla", "comilla", "কুমিল্লা"]),
    ("Chattogram", ["brahmanbaria", "ব্রাহ্মণবাড়িয়া", "ব্রাহ্মণবাড়িয়া"]),
    # Rajshahi division
    ("Rajshahi", ["rajshahi", "রাজশাহী"]),
    ("Rajshahi", ["naogaon", "নওগাঁ", "নওগাঁ"]),  # same spelling two forms
    ("Rajshahi", ["natore", "নাটোর"]),
    ("Rajshahi", ["chapai", "nawabganj", "চাঁপাইনবাবগঞ্জ", "নবাবগঞ্জ"]),
    ("Rajshahi", ["bogura", "bogra", "বগুড়া", "বগুড়া"]),
    ("Rajshahi", ["joypurhat", "জয়পুরহাট", "জয়পুরহাট", "জয়পুরহাট"]),
    ("Rajshahi", ["sirajganj", "সিরাজগঞ্জ"]),
    ("Rajshahi", ["pabna", "পাবনা"]),
    # Khulna division
    ("Khulna", ["khulna", "খুলনা"]),
    ("Khulna", ["jashore", "jeshore", "যশোর"]),
    ("Khulna", ["satkhira", "সাতক্ষীরা"]),
    ("Khulna", ["bagerhat", "বাগেরহাট", "বাগেরহাট"]),
    ("Khulna", ["narail", "নড়াইল", "নড়াইল"]),
    ("Khulna", ["magura", "মাগুরা"]),
    ("Khulna", ["jhenaidah", "jhenaidaha", "ঝিনাইদহ", "ঝিনাইদাহ", "ঝিনাইদাহ"]),
    ("Khulna", ["kushtia", "কুষ্টিয়া", "কুষ্টিয়া"]),
    ("Khulna", ["chuadanga", "চুয়াডাঙ্গা", "চুয়াডাঙ্গা", "চুয়াডাঙ্গা"]),
    ("Khulna", ["meherpur", "মেহেরপুর"]),
    # Barishal division
    ("Barishal", ["barishal", "barisal", "বরিশাল"]),
    ("Barishal", ["patuakhali", "পটুয়াখালী", "পটুয়াখালী"]),
    ("Barishal", ["bhola", "ভোলা"]),
    ("Barishal", ["barguna", "বরগুনা"]),
    ("Barishal", ["pirojpur", "পিরোজপুর", "পিরোজপুর"]),
    ("Barishal", ["jhalokathi", "jhalokati", "ঝালকাঠি", "ঝালকাঠি"]),
    # Sylhet division
    ("Sylhet", ["sylhet", "সিলেট"]),
    ("Sylhet", ["moulvibazar", "maulvibazar", "moulavibazar", "মৌলভীবাজার"]),
    ("Sylhet", ["habiganj", "হবিগঞ্জ", "হবিগঞ্জ"]),
    ("Sylhet", ["sunamganj", "সুনামগঞ্জ"]),
    # Rangpur division
    ("Rangpur", ["rangpur", "রংপুর"]),
    ("Rangpur", ["gaibandha", "গাইবান্ধা"]),
    ("Rangpur", ["kurigram", "কুড়িগ্রাম", "কুড়িগ্রাম"]),
    ("Rangpur", ["lalmonirhat", "লালমনিরহাট"]),
    ("Rangpur", ["nilphamari", "নীলফামারী"]),
    ("Rangpur", ["panchagarh", "পঞ্চগড়", "পঞ্চগড়"]),
    ("Rangpur", ["thakurgaon", "ঠাকুরগাঁও"]),
    ("Rangpur", ["dinajpur", "দিনাজপুর"]),
]

_DIVISION_KEYWORDS = [
    ("Dhaka", ["dhaka", "ঢাকা"]),
    ("Chattogram", ["chattogram", "chattagram", "chittagong", "চট্টগ্রাম"]),
    ("Rajshahi", ["rajshahi", "রাজশাহী"]),
    ("Khulna", ["khulna", "খুলনা"]),
    ("Sylhet", ["sylhet", "সিলেট"]),
    ("Barishal", ["barishal", "barisal", "বরিশাল"]),
    ("Mymensingh", ["mymensingh", "ময়মনসিংহ", "ময়মনসিংহ"]),
    ("Rangpur", ["rangpur", "রংপুর"]),
]

def _infer_division(region: str | None) -> str | None:
    """
    Best-effort division inference from region text (Bangla or English).
    Prefer explicit substring mapping; fall back to broad keywords.
    """
    if not region:
        return None
    r = (region or "").lower()
    # explicit map
    for div, keys in _REGION_DIVISION_MAP:
        if any(k in r for k in keys):
            return div
    # fallback broad keywords
    for div, keys in _DIVISION_KEYWORDS:
        if any(k in r for k in keys):
            return div
    return None

async def _tag_candidate_party_for_article(article_id: int) -> None:
    """
    Auto-tag candidate/party/region for an article using LLM extraction and
    candidate_ref lookup. Only fills missing fields; never overwrites user data.
    """
    async with await _get_conn() as con:
        cur = await con.execute(
            """
            SELECT title, content_clean, content_raw, candidate, party, region
            FROM article
            WHERE article_id = %s
            """,
            (article_id,),
        )
        row = await cur.fetchone()
    if not row:
        return

    title, clean, raw, cand_existing, party_existing, region_existing = row
    cand_existing = (cand_existing or "").strip() or None
    party_existing = (party_existing or "").strip() or None
    region_existing = (region_existing or "").strip() or None

    text_body = (clean or raw or "")[:6000]
    try:
        res = await asyncio.to_thread(_llm_extract_candidate, title or "", text_body)
    except Exception:
        res = {}
    if not isinstance(res, dict):
        res = {}

    cand_new = (res.get("candidate") or "").strip() or None
    party_llm = (res.get("party") or "").strip() or None
    seat_llm = (res.get("seat") or "").strip() or None

    lookup_name = cand_existing or cand_new
    cand_ref = await _candidate_ref_lookup(lookup_name) if lookup_name else None

    cand_to_set = None
    if not cand_existing:
        if cand_ref:
            cand_to_set = (cand_ref.get("name_bn") or cand_ref.get("name") or cand_new) or None
        else:
            cand_to_set = cand_new

    party_suggestion = party_llm or ((cand_ref.get("party") or None) if cand_ref else None)
    party_to_set = party_suggestion if not party_existing else None

    region_suggestion = seat_llm or ((cand_ref.get("seat") or None) if cand_ref else None)
    if not region_suggestion:
        region_suggestion = _infer_division((title or "") + " " + text_body)
    region_to_set = region_suggestion if not region_existing else None

    if not any([cand_to_set, party_to_set, region_to_set]):
        return

    async with await _get_conn() as con:
        await con.execute(
            """
            UPDATE article
            SET candidate = COALESCE(%s, candidate),
                party = COALESCE(%s, party),
                region = COALESCE(%s, region),
                updated_at = now()
            WHERE article_id = %s
            """,
            (cand_to_set, party_to_set, region_to_set, article_id),
        )

async def _auto_fetch_category(cat_id: int) -> Optional[dict]:
    async with await psycopg.AsyncConnection.connect(DB_DSN) as con:
        cur = await con.execute("""SELECT id, label, top_terms, size, model_name, algo FROM auto_category WHERE id = %s""",(cat_id,))
        r = await cur.fetchone()
    if not r:
        return None
    return {"id": r[0], "label": r[1], "top_terms": list(r[2] or []), "size": r[3], "model_name": r[4], "algo": r[5]}

async def _auto_best_category_for_article(article_id: int) -> Optional[int]:
    async with await psycopg.AsyncConnection.connect(DB_DSN) as con:
        cur = await con.execute("""
            SELECT category_id
            FROM auto_category_article
            WHERE article_id = %s
            ORDER BY COALESCE(rank, 999999), score DESC NULLS LAST, category_id
            LIMIT 1
        """, (article_id,))
        r = await cur.fetchone()
    return (r[0] if r else None)

async def _auto_load_centroid(cat_id: int) -> Optional[List[float]]:
    async with await psycopg.AsyncConnection.connect(DB_DSN) as con:
        cur = await con.execute("SELECT vector FROM auto_category_centroid WHERE category_id = %s", (cat_id,))
        r = await cur.fetchone()
    return list(r[0]) if r and r[0] else None

# ---------------------------------------------
# Basic endpoints used by the Streamlit UI
# ---------------------------------------------

async def _status_counts(start_dt: Optional[date] = None, end_dt: Optional[date] = None) -> Dict[str, int]:
    """
    Aggregate article status counts. Treat NULL/empty statuses as 'new'
    so existing data without explicit status still shows up in the UI.
    """
    wanted = {"queued", "new", "needs_review", "classified"}
    out = {k: 0 for k in wanted}
    where: List[str] = []
    params: List[Any] = []
    _apply_date_filters(where, params, start_dt, end_dt)
    where_sql = f" WHERE {' AND '.join(where)}" if where else ""
    async with await _get_conn() as con:
        try:
            cur = await con.execute(
                f"""
                SELECT LOWER(
                           NULLIF(TRIM(COALESCE(status, 'new')), '')
                       ) AS st,
                       COUNT(*)
                FROM article a
                {where_sql}
                GROUP BY st
                """,
                tuple(params),
            )
            rows = await cur.fetchall()
            for st, cnt in rows:
                st = (st or "new").strip().lower()
                if st in out:
                    out[st] = int(cnt)
        except Exception:
            pass
        try:
            cur2 = await con.execute(f"SELECT COUNT(*) FROM article a{where_sql}", tuple(params))
            total = (await cur2.fetchone())[0]
            dup_sql = f"SELECT COUNT(*) FROM article a{where_sql} {' AND ' if where_sql else ' WHERE '} is_duplicate IS TRUE"
            dup_cnt = (await (await con.execute(dup_sql, tuple(params))).fetchone())[0]
        except Exception:
            total = sum(out.values())
            dup_cnt = 0
    # Derive unclassified as total minus classified
    out_extra = {
        "total": int(total),
        "unclassified": int(max(0, int(total) - int(out.get("classified", 0)))),
        "duplicates": int(dup_cnt),
    }
    out.update(out_extra)
    return out

@app.get("/api/counts")
async def api_counts(start_date: Optional[str] = None, end_date: Optional[str] = None, current_user=Depends(require_user)):
    start_dt = _parse_date(start_date)
    end_dt = _parse_date(end_date)
    return await _status_counts(start_dt, end_dt)

@app.get("/api/categories")
async def api_categories(current_user=Depends(require_user)):
    cats = await _fetch_categories()
    return {"items": cats}

@app.get("/api/candidates")
async def api_candidates(limit: int = Query(500, ge=1, le=2000), current_user=Depends(require_user)):
    return {"items": await _fetch_candidates(limit=limit)}

@app.get("/api/candidate_refs")
async def api_candidate_refs(limit: int = Query(2000, ge=1, le=5000), current_user=Depends(require_user)):
    return {"items": await _candidate_refs(limit=limit)}

@app.post("/api/candidate_refs")
async def api_candidate_refs_upsert(
    name: str = Body(..., embed=True),
    name_bn: Optional[str] = Body(None, embed=True),
    party: Optional[str] = Body(None, embed=True),
    seat: Optional[str] = Body(None, embed=True),
    aliases: Optional[List[str]] = Body(None, embed=True),
    current_user=Depends(require_user),
):
    nm = (name or "").strip()
    if not nm:
        raise HTTPException(400, "name required")
    aliases_clean = [a.strip() for a in (aliases or []) if a and str(a).strip()]
    rec = await _upsert_candidate_ref(
        nm,
        (party or None),
        aliases_clean,
        (name_bn or "").strip() or None,
        (seat or "").strip() or None,
    )
    return {"item": rec}

@app.get("/api/classified")
async def api_classified(
    limit: int = Query(200, ge=1, le=5000),
    primary_only: bool = True,
    unique: bool = True,
    source: Optional[str] = Query(None, pattern="^(rule|ml|llm|human)$"),
    category: Optional[str] = None,
    group_by: Optional[str] = Query(None, pattern="^(category|candidate|party)$"),
    current_user=Depends(require_user),
):
    where = []
    params: List[Any] = []
    if primary_only:
        where.append("al.is_primary IS TRUE")
    if source:
        where.append("al.source = %s::label_source")
        params.append(source)
    if category:
        where.append("c.category = %s")
        params.append(category)
    where_sql = (" WHERE " + " AND ".join(where)) if where else ""

    if group_by:
        key_sql = {"category": "c.category", "candidate": "a.candidate", "party": "a.party"}[group_by]
        sql = f"""
            SELECT {key_sql} AS key,
                   COUNT(*) AS total,
                   COUNT(*) FILTER (WHERE al.is_primary) AS primary_count
            FROM article a
            JOIN article_label al ON al.article_id = a.article_id
            JOIN codebook_category c ON c.category_id = al.category_id
            {where_sql}
            GROUP BY {key_sql}
            ORDER BY primary_count DESC, total DESC NULLS LAST, key NULLS LAST
            LIMIT %s
        """
        params2 = params + [int(limit)]
        async with await _get_conn() as con:
            cur = await con.execute(sql, tuple(params2))
            rows = await cur.fetchall()
        items = []
        for r in rows:
            items.append({
                group_by: r[0],
                "total": int(r[1]),
                "primary": int(r[2]),
            })
        return {"items": items, "group_by": group_by}

    base_fields = "a.article_id, a.title, a.url, a.published_at, c.category, al.score, al.source, a.party, a.candidate"
    order_unique = "a.article_id, CASE WHEN al.is_primary THEN 0 ELSE 1 END, al.score DESC NULLS LAST, c.category"
    order_all = "a.published_at DESC NULLS LAST, al.score DESC NULLS LAST, a.article_id DESC"

    if unique:
        sql = f"""
            SELECT DISTINCT ON (a.article_id) {base_fields}
            FROM article a
            JOIN article_label al ON al.article_id = a.article_id
            JOIN codebook_category c ON c.category_id = al.category_id
            {where_sql}
            ORDER BY {order_unique}
            LIMIT %s
        """
    else:
        sql = f"""
            SELECT {base_fields}
            FROM article a
            JOIN article_label al ON al.article_id = a.article_id
            JOIN codebook_category c ON c.category_id = al.category_id
            {where_sql}
            ORDER BY {order_all}
            LIMIT %s
        """
    params2 = params + [int(limit)]

    async with await _get_conn() as con:
        cur = await con.execute(sql, tuple(params2))
        rows = await cur.fetchall()

    items = []
    for r in rows:
        items.append({
            "article_id": r[0],
            "title": r[1],
            "url": r[2],
            "published_at": r[3],
            "category": r[4],
            "score": float(r[5]) if r[5] is not None else None,
            "mode": str(r[6]) if r[6] is not None else None,
            "party": r[7],
            "candidate": r[8],
        })
    return {"items": items}


# ---------------------------------------------
# Auto candidate tagging (heuristic)
# ---------------------------------------------

def _count_candidate_hits(text: str, cand: str) -> int:
    if not text or not cand:
        return 0
    t = _norm(text)
    c = _norm(cand)
    if not c or len(c) < 2:
        return 0
    # unicode-aware word boundary-ish match; allow flexible spacing inside names
    if " " in c:
        pat = re.escape(c.strip())
        pat = _SPACE_RX.sub(r"\\s+", pat)
        rx = re.compile(rf"(?<!\\w){pat}(?!\\w)", flags=re.UNICODE)
        return len(rx.findall(t))
    return len(re.findall(rf"(?<!\\w){re.escape(c)}(?!\\w)", t, flags=re.UNICODE))


def _guess_candidates_from_rows(rows: List[tuple], max_terms: int = 50) -> List[str]:
    """
    Fallback heuristic: extract frequent tokens from titles/content as seed candidates.
    """
    stop = {
        "the","a","an","and","or","of","in","on","for","to","from","by","with","as","at",
        "is","are","was","were","be","been","it","this","that","these","those","but","if",
        "not","no","yes","you","we","they","he","she","i","my","our","your","their","its",
        "about","into","after","before","over","under","more","most","less","least","also",
        "new","news","video","photo","photos","pic","pics","via","today","bangladesh","bd",
        "said","says","will","would","could","should","one","two","three",
    }
    freq: Dict[str, int] = {}
    for _aid, title, content in rows:
        text = f"{title} {content}"
        for tok in re.findall(r"\w+", text, flags=re.UNICODE):
            t = tok.strip()
            if len(t) < 3:
                continue
            lo = t.lower()
            if lo in stop:
                continue
            freq[lo] = freq.get(lo, 0) + 1
    sorted_terms = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [t for t, _ in sorted_terms[:max_terms]]


@app.post("/api/auto/tag_candidates")
async def api_auto_tag_candidates(
    limit: int = Query(200, ge=1, le=2000),
    min_hits: int = Query(1, ge=1, le=10),
    seeds: Optional[List[str]] = Body(None, embed=True),
    mode: str = Query("heuristic", pattern="^(heuristic|llm|hybrid)$"),
    current_user=Depends(require_user),
):
    """
    Auto-tagging of candidates.
    - heuristic: keyword/alias matching (fast, uses master list + seeds).
    - llm: use LLM extraction to pick candidate/party/seat from text.
    - hybrid: heuristic first; if no hit and LLM available, fall back to LLM.
    """
    seed_list = [s.strip() for s in (seeds or []) if s and str(s).strip()]

    async with await _get_conn() as con:
        cur = await con.execute(
            """
            SELECT article_id, COALESCE(title,''), COALESCE(content_clean, content_raw, '')
            FROM article
            WHERE candidate IS NULL OR NULLIF(TRIM(candidate), '') IS NULL
            ORDER BY article_id ASC
            LIMIT %s
            """,
            (int(limit),),
        )
        rows = await cur.fetchall()

        if not rows:
            return {"ok": True, "tagged": 0, "scanned": 0, "items": [], "reason": "no_articles"}

        if mode == "llm":
            if _oa is None:
                return {"ok": False, "tagged": 0, "scanned": 0, "items": [], "reason": "llm_not_configured"}
            tagged = 0
            items = []
            for aid, title, content in rows:
                res = await asyncio.to_thread(_llm_extract_candidate, title, content)
                if res.get("_error"):
                    continue
                cand = res.get("candidate")
                if not cand:
                    continue
                party_val = res.get("party")
                seat_val = res.get("seat")
                await con.execute(
                    """
                    UPDATE article
                    SET candidate=%s,
                        party = COALESCE(%s, party),
                        region = COALESCE(%s, region),
                        updated_at=now()
                    WHERE article_id=%s
                    """,
                    (cand, party_val, seat_val, aid),
                )
                tagged += 1
                items.append({
                    "article_id": aid,
                    "candidate": cand,
                    "hits": None,
                    "seat": seat_val,
                    "party": party_val,
                    "mode": "llm",
                })
            return {"ok": True, "tagged": tagged, "scanned": len(rows), "items": items[:50], "mode": mode}

        # heuristic mode (default)
        refs = await _candidate_refs(limit=2000)
        alias_pairs: List[tuple[str, str]] = []
        canon_meta: Dict[str, Dict[str, Optional[str]]] = {}
        seen_alias_keys: set[str] = set()
        for r in refs:
            canon = (r.get("name_bn") or r.get("name") or "").strip()
            if not canon:
                continue
            canon_key = canon.lower()
            canon_meta[canon_key] = {
                "party": (r.get("party") or "").strip() or None,
                "seat": (r.get("seat") or "").strip() or None,
                "name": canon,
            }
            base_aliases = [r.get("name"), r.get("name_bn")] + list(r.get("aliases") or [])
            for alias in base_aliases:
                alias_clean = (alias or "").strip()
                if not alias_clean:
                    continue
                key = f"{canon_key}::{alias_clean.lower()}"
                if key in seen_alias_keys:
                    continue
                alias_pairs.append((canon, alias_clean))
                seen_alias_keys.add(key)
            seat_alias = (r.get("seat") or "").strip()
            if seat_alias:
                key = f"{canon_key}::{seat_alias.lower()}"
                if key not in seen_alias_keys:
                    alias_pairs.append((canon, seat_alias))
                    seen_alias_keys.add(key)
        existing = await _existing_candidates(limit=2000)
        candidates_set = set(seed_list + existing + [m["name"] for m in canon_meta.values()])
        if not candidates_set:
            candidates_set.update(_guess_candidates_from_rows(rows))
        candidates = sorted(candidates_set)

        if not candidates:
            return {"ok": False, "tagged": 0, "scanned": len(rows), "items": [], "reason": "no_candidates"}

        tagged = 0
        items = []
        for aid, title, content in rows:
            text = f"{title}\n{content}"
            best = None
            best_hits = 0
            best_meta = None
            # check aliases first to preserve canonical name
            for canon, alias in alias_pairs:
                hits = _count_candidate_hits(text, alias)
                if hits > best_hits:
                    best_hits, best = hits, canon
                    best_meta = canon_meta.get(canon.lower())
            # then names
            for cand in candidates:
                hits = _count_candidate_hits(text, cand)
                if hits > best_hits:
                    best_hits, best = hits, cand
                    best_meta = canon_meta.get(cand.lower())
            if best and best_hits >= int(min_hits):
                party_val = (best_meta.get("party") if best_meta else None) or None
                seat_val = (best_meta.get("seat") if best_meta else None) or None
                await con.execute(
                    """
                    UPDATE article
                    SET candidate=%s,
                        party = COALESCE(%s, party),
                        region = COALESCE(%s, region),
                        updated_at=now()
                    WHERE article_id=%s
                    """,
                    (best, party_val, seat_val, aid),
                )
                tagged += 1
                items.append({
                    "article_id": aid,
                    "candidate": best,
                    "hits": best_hits,
                    "seat": seat_val,
                    "party": party_val,
                    "mode": "heuristic",
                })

        if mode == "hybrid":
            # heuristic first, then LLM for misses (if configured)
            matched_ids = {it["article_id"] for it in items}
            if _oa is not None:
                for aid, title, content in rows:
                    if aid in matched_ids:
                        continue
                    res = await asyncio.to_thread(_llm_extract_candidate, title, content)
                    if res.get("_error"):
                        continue
                    cand = res.get("candidate")
                    if not cand:
                        continue
                    party_val = res.get("party")
                    seat_val = res.get("seat")
                    await con.execute(
                        """
                        UPDATE article
                        SET candidate=%s,
                            party = COALESCE(%s, party),
                            region = COALESCE(%s, region),
                            updated_at=now()
                        WHERE article_id=%s
                        """,
                        (cand, party_val, seat_val, aid),
                    )
                    tagged += 1
                    items.append({
                        "article_id": aid,
                        "candidate": cand,
                        "hits": None,
                        "seat": seat_val,
                        "party": party_val,
                        "mode": "llm",
                    })

    return {"ok": True, "tagged": tagged, "scanned": len(rows), "items": items[:50], "mode": mode}


# ---------------------------------------------
# Classification and generation endpoints
# ---------------------------------------------

def _count_term(text_norm: str, term: str) -> int:
    if not text_norm or not term:
        return 0
    t = (term or "").strip().lower()
    if not t:
        return 0
    if " " in t:
        return text_norm.count(t)
    # single token: approximate word boundary using unicode-aware \w
    return len(re.findall(rf"(?<!\\w){re.escape(t)}(?!\\w)", text_norm, flags=re.UNICODE))

async def _classify_rule_fallback(aid: int) -> List[Dict[str, Any]]:
    """Rule-based scoring in backend as a fallback when MCP yields no labels."""
    art = await _fetch_article(aid)
    title = _norm(art.get("title") or "")
    content = _norm((art.get("content_clean") or art.get("content_raw") or ""))
    text = f"{title}\n{content}".strip()
    if not text:
        return []
    async with await _get_conn() as con:
        cur = await con.execute(
            """
            SELECT c.category, k.term, COALESCE(k.weight,1.0)
            FROM codebook_keyword k
            JOIN codebook_category c ON c.category_id = k.category_id
            """
        )
        rows = await cur.fetchall()
    by_cat: Dict[str, float] = {}
    for cat, term, weight in rows:
        if not cat or not term:
            continue
        cnt = _count_term(text, str(term))
        if cnt:
            by_cat[cat] = by_cat.get(cat, 0.0) + float(weight) * float(cnt)
    if not by_cat:
        return []
    # normalize to 0..1 by max and mark single primary
    top = max(by_cat.values()) or 1.0
    labels = [{"category": c, "score": float(s)/float(top), "source": "rule", "is_primary": False} for c, s in by_cat.items()]
    labels.sort(key=lambda x: x["score"], reverse=True)
    labels[0]["is_primary"] = True
    return labels[:10]

async def _classify_article_internal(
    article_id: int,
    mode: str = "auto",
    skip_if_classified: bool = True,
) -> Dict[str, Any]:
    if skip_if_classified:
        st = await _get_article_status(article_id)
        if (st or "").strip().lower() == "classified":
            return {"ok": True, "skipped": True, "reason": "already_classified", "mode": mode, "labels": []}

    labels_rule: List[Dict[str, Any]] = []
    labels_ml: List[Dict[str, Any]] = []
    chosen_source = None

    if mode == "rule":
        res = await _mcp_call("classify_text", {"article_id": article_id})
        labels_rule = list(res.get("labels", []) or [])
        for l in labels_rule:
            l["source"] = "rule"
        final = labels_rule
        chosen_source = "rule"

    elif mode == "ml":
        res = await _mcp_call("classify_ml", {"article_id": article_id})
        labels_ml = list(res.get("labels", []) or [])
        for l in labels_ml:
            l["source"] = "ml"
        final = labels_ml
        chosen_source = "ml"

    elif mode == "hybrid":
        rr = await _mcp_call("classify_text", {"article_id": article_id})
        labels_rule = list(rr.get("labels", []) or [])
        for l in labels_rule:
            l["source"] = "rule"
        mr = await _mcp_call("classify_ml", {"article_id": article_id})
        labels_ml = list(mr.get("labels", []) or [])
        for l in labels_ml:
            l["source"] = "ml"
        final = _merge_labels(labels_rule, labels_ml)
        if not final:
            try:
                final = await _classify_rule_fallback(article_id)
            except Exception:
                final = []
        chosen_source = (final[0]["source"] if final else "ml")

    elif mode == "llm":
        try:
            gen = await _generate_from_article_internal(article_id=article_id, mode="llm", save_official=False)
            final = list(gen.get("labels", []) or [])
        except Exception:
            final = []
        for l in final:
            l.setdefault("source", "llm")
        chosen_source = "llm"

    else:  # auto
        rr = await _mcp_call("classify_text", {"article_id": article_id})
        labels_rule = list(rr.get("labels", []) or [])
        for l in labels_rule:
            l["source"] = "rule"
        if labels_rule:
            final = labels_rule
            chosen_source = "rule"
        else:
            mr = await _mcp_call("classify_ml", {"article_id": article_id})
            labels_ml = list(mr.get("labels", []) or [])
            for l in labels_ml:
                l["source"] = "ml"
            if labels_ml:
                final = labels_ml
                chosen_source = "ml"
            else:
                try:
                    gen = await _generate_from_article_internal(article_id=article_id, mode="llm", save_official=False)
                    final = list(gen.get("labels", []) or [])
                except Exception:
                    final = []
                for l in final:
                    l.setdefault("source", "llm")
                chosen_source = "llm"

    if final:
        await _save_labels(article_id, final, model_name=OPENAI_MODEL, source=chosen_source or "ml")
        await _tag_candidate_party_for_article(article_id)
    return {"ok": True, "mode": mode, "used": chosen_source, "labels": final}


async def _fetch_top_labels_for_articles(ids: List[int]) -> List[Dict[str, Any]]:
    if not ids:
        return []
    async with await _get_conn() as con:
        cur = await con.execute(
            """
            WITH labeled AS (
              SELECT
                a.article_id, a.title, a.url, a.published_at, a.status, a.lang,
                a.portal_id, a.party, a.candidate, a.region,
                al.category_id, c.category, al.score, al.is_primary,
                ROW_NUMBER() OVER (
                  PARTITION BY a.article_id
                  ORDER BY COALESCE(al.is_primary, FALSE) DESC, al.score DESC NULLS LAST, al.category_id
                ) AS rn
              FROM article a
              LEFT JOIN article_label al ON al.article_id = a.article_id
              LEFT JOIN codebook_category c ON c.category_id = al.category_id
              WHERE a.article_id = ANY(%s)
            )
            SELECT l.article_id, l.title, l.url, l.published_at, l.status, l.lang,
                   l.portal_id, p.name AS portal_name,
                   l.party, l.candidate, l.region,
                   l.category, l.score, l.is_primary
            FROM labeled l
            LEFT JOIN portal p ON p.portal_id = l.portal_id
            WHERE l.rn = 1
            ORDER BY l.article_id
            """,
            (ids,),
        )
        rows = await cur.fetchall()

    items: List[Dict[str, Any]] = []
    for r in rows:
        items.append({
            "article_id": r[0],
            "title": r[1],
            "url": r[2],
            "published_at": r[3],
            "status": r[4],
            "lang": r[5],
            "portal_id": r[6],
            "portal": r[7],
            "party": r[8],
            "candidate": r[9],
            "region": r[10],
            "category": r[11],
            "score": float(r[12]) if r[12] is not None else None,
            "is_primary": bool(r[13]) if r[13] is not None else None,
        })
    return items


async def _fetch_keywords_for_categories(categories: List[str], limit_terms: int = 8) -> Dict[str, List[str]]:
    cats = [c for c in (categories or []) if c]
    if not cats:
        return {}
    rows: List[tuple] = []
    # First try codebook_category_keyword (if present)
    try:
        async with await _get_conn() as con:
            cur = await con.execute(
                """
                SELECT c.category, k.keyword, k.weight
                FROM codebook_category_keyword k
                JOIN codebook_category c ON c.category_id = k.category_id
                WHERE c.category = ANY(%s)
                ORDER BY c.category, k.weight DESC NULLS LAST, k.keyword ASC
                """,
                (cats,),
            )
            rows = await cur.fetchall()
    except Exception:
        rows = []

    # Fallback to legacy codebook_keyword table if the above is empty/missing
    if not rows:
        try:
            async with await _get_conn() as con:
                cur = await con.execute(
                    """
                    SELECT c.category, k.term AS keyword, k.weight
                    FROM codebook_keyword k
                    JOIN codebook_category c ON c.category_id = k.category_id
                    WHERE c.category = ANY(%s)
                    ORDER BY c.category, k.weight DESC NULLS LAST, k.term ASC
                    """,
                    (cats,),
                )
                rows = await cur.fetchall()
        except Exception:
            rows = []

    kw_map: Dict[str, List[str]] = {}
    for cat, kw, weight in rows:
        if not kw:
            continue
        lst = kw_map.setdefault(cat, [])
        if len(lst) < limit_terms:
            lst.append(str(kw))
    return kw_map


def _match_terms_in_text(text: Optional[str], terms: List[str], limit_terms: int = 12) -> List[str]:
    """
    Return the subset of terms that appear in the given text (case-insensitive).
    Stops after limit_terms matches to keep payload small.
    """
    if not text or not terms:
        return []
    txt = str(text).lower()
    matched: List[str] = []
    for term in terms:
        t = (term or "").strip()
        if not t:
            continue
        if t.lower() in txt:
            matched.append(t)
            if len(matched) >= limit_terms:
                break
    return matched


async def _fetch_articles_text(ids: List[int]) -> Dict[int, str]:
    if not ids:
        return {}
    async with await _get_conn() as con:
        cur = await con.execute(
            """
            SELECT article_id, title, content_clean
            FROM article
            WHERE article_id = ANY(%s)
            """,
            (ids,),
        )
        rows = await cur.fetchall()
    res: Dict[int, str] = {}
    for aid, title, content in rows:
        parts = [title or "", content or ""]
        res[int(aid)] = " ".join(p for p in parts if p)
    return res


@app.post("/api/classify/{article_id}")
async def api_classify_article(
    article_id: int,
    mode: str = Query("auto", pattern="^(rule|ml|hybrid|llm|auto)$"),
    skip_if_classified: bool = Query(True),
    current_user=Depends(require_user),
):
    return await _classify_article_internal(article_id=article_id, mode=mode, skip_if_classified=skip_if_classified)


async def _generate_from_article_internal(
    article_id: int,
    mode: str = "hybrid",
    save_official: bool = False,
) -> Dict[str, Any]:
    await _fetch_article(article_id)  # ensure exists

    labels_rule: List[Dict[str, Any]] = []
    labels_ml: List[Dict[str, Any]] = []

    if mode in ("rule", "hybrid"):
        rr = await _mcp_call("classify_text", {"article_id": article_id})
        labels_rule = list(rr.get("labels", []) or [])
        for l in labels_rule:
            l["source"] = "rule"
    if mode in ("ml", "hybrid"):
        mr = await _mcp_call("classify_ml", {"article_id": article_id})
        labels_ml = list(mr.get("labels", []) or [])
        for l in labels_ml:
            l["source"] = "ml"

    final = _merge_labels(labels_rule, labels_ml)
    if not final and mode in ("rule", "hybrid", "llm"):
        try:
            final = await _classify_rule_fallback(article_id)
        except Exception:
            final = []
    gen_mode = ("llm" if mode == "llm" else (final[0]["source"] if final else None))

    async with await _get_conn() as con:
        async with con.transaction():
            if final:
                best_cat = None
                best_score = -1
                for l in final:
                    if float(l.get("score", 0) or 0) > best_score:
                        best_score = float(l.get("score", 0) or 0)
                        best_cat = l.get("category")

                for l in final:
                    cat = l.get("category")
                    score = float(l.get("score", 0) or 0)
                    src = gen_mode or (l.get("source") or "rule")
                    is_primary = (cat == best_cat)
                    await con.execute(
                        """
                        INSERT INTO codebook_category_generate(article_id, category_id, mode, score, is_primary)
                        SELECT %s, category_id, %s::label_source, %s, %s
                        FROM codebook_category WHERE category=%s
                        ON CONFLICT (article_id, category_id, mode) DO UPDATE
                        SET score=EXCLUDED.score, is_primary=EXCLUDED.is_primary
                        """,
                        (article_id, src, score, is_primary, cat)
                    )

            if save_official and final:
                for l in final:
                    cat = l.get("category")
                    score = float(l.get("score", 0) or 0)
                    src = gen_mode or (l.get("source") or "rule")
                    is_primary = bool(l.get("is_primary", False))
                    await con.execute(
                        """
                        INSERT INTO article_category_label(article_id, category_id, score, mode, is_primary)
                        SELECT %s, category_id, %s, %s::label_source, %s
                        FROM codebook_category WHERE category=%s
                        ON CONFLICT (article_id, category_id, mode) DO UPDATE
                        SET score=EXCLUDED.score, is_primary=EXCLUDED.is_primary
                        """,
                        (article_id, score, src, is_primary, cat)
                    )

            await con.execute(
                "UPDATE article SET gen_cats_at = now(), gen_cats_mode = %s::label_source WHERE article_id=%s",
                (gen_mode if gen_mode in ("rule", "ml", "llm", "human") else None, article_id)
            )

    return {"ok": True, "mode": gen_mode, "labels": final}


@app.post("/api/generate_from_article/{article_id}")
async def api_generate_from_article(
    article_id: int,
    mode: str = Query("hybrid", pattern="^(rule|ml|hybrid|llm)$"),
    save_official: bool = Query(False),
    current_user=Depends(require_user),
):
    return await _generate_from_article_internal(article_id=article_id, mode=mode, save_official=save_official)


@app.get("/api/generated")
async def api_generated(
    limit: int = Query(200, ge=1, le=5000),
    unique: bool = True,
    mode: Optional[str] = Query(None, pattern="^(rule|ml|llm|human)$"),
    category: Optional[str] = None,
    current_user=Depends(require_user),
):
    where = []
    params: List[Any] = []
    if mode:
        where.append("g.mode = %s::label_source")
        params.append(mode)
    if category:
        where.append("c.category = %s")
        params.append(category)
    where_sql = (" WHERE " + " AND ".join(where)) if where else ""

    order_unique = "a.article_id, CASE WHEN g.is_primary THEN 0 ELSE 1 END, g.score DESC NULLS LAST, c.category"
    order_all = "a.published_at DESC NULLS LAST, g.score DESC NULLS LAST, a.article_id DESC"

    if unique:
        sql = f"""
            SELECT DISTINCT ON (a.article_id)
              a.article_id, a.title, a.url, a.published_at, a.lang,
              c.category, g.score, g.is_primary, g.mode, g.created_at
            FROM codebook_category_generate g
            JOIN codebook_category c ON c.category_id = g.category_id
            JOIN article a            ON a.article_id   = g.article_id
            {where_sql}
            ORDER BY {order_unique}
            LIMIT %s
        """
    else:
        sql = f"""
            SELECT 
              a.article_id, a.title, a.url, a.published_at, a.lang,
              c.category, g.score, g.is_primary, g.mode, g.created_at
            FROM codebook_category_generate g
            JOIN codebook_category c ON c.category_id = g.category_id
            JOIN article a            ON a.article_id   = g.article_id
            {where_sql}
            ORDER BY {order_all}
            LIMIT %s
        """

    params2 = params + [int(limit)]
    async with await _get_conn() as con:
        cur = await con.execute(sql, tuple(params2))
        rows = await cur.fetchall()
    items = []
    for r in rows:
        items.append({
            "article_id": r[0],
            "title": r[1],
            "url": r[2],
            "published_at": r[3],
            "lang": r[4],
            "category": r[5],
            "score": float(r[6]) if r[6] is not None else None,
            "is_primary": bool(r[7]),
            "mode": str(r[8]),
            "created_at": r[9],
        })
    return {"items": items}


@app.get("/api/articles")
async def api_articles(
    status: Optional[str] = Query(None, pattern="^(queued|new|needs_review|classified)$"),
    limit: int = Query(50, ge=1, le=5000),
    candidate_missing: bool = False,
    current_user=Depends(require_user),
):
    where = []
    params: List[Any] = []
    if status:
        where.append("a.status = %s")
        params.append(status)
    if candidate_missing:
        where.append("(a.candidate IS NULL OR NULLIF(TRIM(a.candidate), '') IS NULL)")
    where_sql = (" WHERE " + " AND ".join(where)) if where else ""

    sql = f"""
        SELECT a.article_id, a.title, a.url, a.published_at,
               a.content_clean, a.content_raw, a.lang, a.status
        FROM article a
        {where_sql}
        ORDER BY a.published_at DESC NULLS LAST, a.article_id DESC
        LIMIT %s
    """
    params2 = params + [int(limit)]
    async with await _get_conn() as con:
        cur = await con.execute(sql, tuple(params2))
        rows = await cur.fetchall()
    items = []
    for r in rows:
        items.append({
            "article_id": r[0],
            "title": r[1],
            "url": r[2],
            "published_at": r[3],
            "content_clean": r[4],
            "content_raw": r[5],
            "lang": r[6],
            "status": r[7],
        })
    return {"items": items}


@app.post("/api/articles/{article_id}/tag_candidate")
async def api_tag_candidate(
    article_id: int,
    candidate: str = Body(..., embed=True),
    party: Optional[str] = Body(None, embed=True),
    region: Optional[str] = Body(None, embed=True),
    current_user=Depends(require_user),
):
    cand = (candidate or "").strip()
    if not cand:
        raise HTTPException(400, "candidate required")
    cand_ref = await _candidate_ref_lookup(cand)
    canon_name = (cand_ref.get("name_bn") or cand_ref.get("name") or cand) if cand_ref else cand
    party_val = (party or None) or ((cand_ref.get("party") or None) if cand_ref else None)
    region_val = (region or None) or ((cand_ref.get("seat") or None) if cand_ref else None)
    async with await _get_conn() as con:
        async with con.transaction():
            cur = await con.execute(
                """
                UPDATE article
                SET candidate = %s,
                    party = COALESCE(%s, party),
                    region = COALESCE(%s, region),
                    updated_at = now()
                WHERE article_id = %s
                """,
                (canon_name, party_val, region_val, article_id),
            )
            if cur.rowcount == 0:
                raise HTTPException(404, f"article_id={article_id} not found")
    return {"ok": True, "article_id": article_id, "candidate": canon_name, "party": party_val, "region": region_val}


@app.post("/api/generate_from_articles")
async def api_generate_from_articles(
    limit: int = Query(200, ge=1, le=5000),
    mode: str = Query("hybrid", pattern="^(rule|ml|hybrid|llm)$"),
    only_missing: bool = True,
    save_official: bool = False,
    current_user=Depends(require_user),
):
    # Pick candidate article_ids
    async with await _get_conn() as con:
        if only_missing:
            cur = await con.execute(
                """
                SELECT a.article_id
                FROM article a
                WHERE (a.status IN ('new','queued','needs_review') OR a.status IS NULL)
                  AND (a.gen_cats_at IS NULL OR a.updated_at > a.gen_cats_at)
                ORDER BY a.article_id ASC
                LIMIT %s
                """,
                (int(limit),),
            )
        else:
            cur = await con.execute(
                """
                SELECT a.article_id
                FROM article a
                ORDER BY a.article_id ASC
                LIMIT %s
                """,
                (int(limit),),
            )
        ids = [r[0] for r in await cur.fetchall()]

    requested = int(limit)
    processed = 0
    failed_count = 0
    saved_scratch = 0
    saved_generated = 0
    ids_failed: List[int] = []

    for aid in ids:
        try:
            res = await _generate_from_article_internal(article_id=aid, mode=mode, save_official=save_official)
            labs = list(res.get("labels", []) or [])
            processed += 1
            saved_scratch += len(labs)
            if save_official:
                saved_generated += len(labs)
        except Exception:
            failed_count += 1
            ids_failed.append(aid)

    return {
        "requested": requested,
        "picked": len(ids),
        "processed": processed,
        "saved_scratch": saved_scratch,
        "saved_generated": saved_generated,
        "failed_count": failed_count,
        "ids": ids,
        "ids_failed": ids_failed,
        "mode": mode,
        "only_missing": only_missing,
    }

@app.post("/api/batch/classify")
async def api_batch_classify(
    limit: int = Query(200, ge=1, le=5000),
    mode: str = Query("auto", pattern="^(rule|ml|hybrid|llm|auto)$"),
    current_user=Depends(require_user),
):
    """
    Classify a batch of unclassified articles (queued/new/needs_review).
    Reuses the per-article classify endpoint to keep behavior consistent.
    """
    async with await _get_conn() as con:
        cur = await con.execute(
            """
            SELECT article_id
            FROM article
            WHERE status IN ('queued','new','needs_review')
            ORDER BY article_id ASC
            LIMIT %s
            """,
            (int(limit),),
        )
        ids = [r[0] for r in await cur.fetchall()]

    requested = int(limit)
    sem = asyncio.Semaphore(min(8, max(1, len(ids))))  # avoid overloading DB/MCP
    async def _classify_one(aid: int):
        async with sem:
            try:
                res = await _classify_article_internal(article_id=aid, mode=mode, skip_if_classified=True)
                return True, aid, res or {}
            except Exception as e:
                return False, aid, {"error": str(e)}

    results = await asyncio.gather(*(_classify_one(aid) for aid in ids))
    processed = sum(1 for ok, _, _ in results if ok)
    ids_failed = [aid for ok, aid, _ in results if not ok]
    failed = len(ids_failed)
    classified_ids = [aid for ok, aid, detail in results if ok and not (detail or {}).get("skipped")]
    details_by_id = {aid: (detail or {}) for ok, aid, detail in results if ok}

    classified_items = await _fetch_top_labels_for_articles(classified_ids)
    kw_map = await _fetch_keywords_for_categories([it.get("category") for it in classified_items], limit_terms=8)
    article_texts = await _fetch_articles_text(classified_ids)
    enriched: List[Dict[str, Any]] = []
    for item in classified_items:
        cat = item.get("category")
        detail = details_by_id.get(item["article_id"], {})
        terms_all = kw_map.get(cat, [])
        matched_terms = _match_terms_in_text(article_texts.get(item["article_id"]), terms_all, limit_terms=12)
        terms = matched_terms or terms_all
        enriched.append({
            **item,
            "mode": mode,
            "model_used": detail.get("used") or detail.get("mode") or mode,
            "keywords": terms,
            "mode_terms": terms,
            "terms": terms,
        })
    categorized = sum(1 for it in enriched if it.get("category"))

    return {
        "requested": requested,
        "picked": len(ids),
        "processed": processed,
        "failed": failed,
        "ids_failed": ids_failed,
        "mode": mode,
        "categorized": categorized,
        "classified_items": enriched,
    }

@app.get("/api/auto/categories")
async def api_auto_categories(current_user=Depends(require_user)):
    """List auto categories if the table exists; otherwise return empty list.
    Avoids a 500 when the auto-category schema hasn't been applied yet.
    """
    try:
        async with await _get_conn() as con:
            cur = await con.execute(
                """
                SELECT id, label, size, top_terms, model_name, algo
                FROM auto_category
                ORDER BY size DESC, id
                """
            )
            rows = await cur.fetchall()
    except Exception as e:
        # If table is missing or any other read error, return an empty list gracefully
        return {"items": []}
    items = []
    for r in rows:
        terms = list(r[3] or [])
        items.append({
            "id": r[0], "label": r[1], "size": r[2],
            "top_terms": terms, "terms_display": ", ".join(terms[:8]),
            "model_name": r[4], "algo": r[5],
        })
    return {"items": items}

@app.get("/api/auto/categories/{cat_id}")
async def api_auto_category_detail(cat_id: int, top: int = 50, current_user=Depends(require_user)):
    cat = await _auto_fetch_category(cat_id)
    if not cat:
        raise HTTPException(404, "Category not found")
    async with await psycopg.AsyncConnection.connect(DB_DSN) as con:
        cur = await con.execute("""
            SELECT article_id, score, rank
            FROM auto_category_article
            WHERE category_id = %s
            ORDER BY COALESCE(rank, 999999), score DESC NULLS LAST, article_id
            LIMIT %s
        """,(cat_id, top))
        links = [{"article_id": r[0], "score": r[1], "rank": r[2]} for r in await cur.fetchall()]
    return {"category": cat, "articles": links}


@app.post("/api/auto/promote/{cat_id}")
async def api_auto_promote(cat_id: int, current_user=Depends(require_admin)):
    """
    Promote an auto_category into the human codebook:
    - Insert a new row in codebook_category (is_auto=true, review_needed=true) if not existing
    - Insert top_terms as keywords
    - Upsert article_label for articles linked to this auto_category
    """
    # Load auto category
    cat = await _auto_fetch_category(cat_id)
    if not cat:
        raise HTTPException(404, "Auto category not found")

    raw_label = cat.get("label") or f"Auto {cat_id}"
    label_parts = [p.strip() for p in str(raw_label).split("|") if p and p.strip()]
    label = label_parts[0] if label_parts else raw_label
    terms_raw = list(cat.get("top_terms") or [])
    terms = [t.strip() for t in terms_raw if t and str(t).strip()]
    if not terms and label_parts:
        terms = label_parts
    # Keep definition concise: only the terms string
    definition_terms = ", ".join(terms[:6])
    definition = definition_terms or None

    # Use autocommit to avoid a single failed insert aborting the whole promotion.
    import psycopg
    async with await psycopg.AsyncConnection.connect(DB_DSN, autocommit=True) as con:
        try:
            saved = 0
            # Find or create codebook category by case-insensitive match
            row = await (await con.execute(
                "SELECT category_id FROM codebook_category WHERE LOWER(category) = LOWER(%s) LIMIT 1",
                (label,)
            )).fetchone()
            if row:
                code_cid = row[0]
                # Optionally backfill definition if missing
                try:
                    await con.execute(
                        "UPDATE codebook_category SET definition = COALESCE(definition, %s) WHERE category_id=%s AND definition IS NULL",
                        (definition, code_cid),
                    )
                except Exception:
                    pass
            else:
                row = await (await con.execute(
                    """
                    INSERT INTO codebook_category(category, definition, is_auto, review_needed, created_by, phase)
                    VALUES (%s, %s, TRUE, TRUE, %s, %s)
                    RETURNING category_id
                    """,
                    (label, definition, "system:auto_promote", "auto"),
                )).fetchone()
                code_cid = row[0]

            # Insert keywords (generic ON CONFLICT to avoid constraint-name issues)
            for t in terms:
                try:
                    await con.execute(
                        """
                        INSERT INTO codebook_keyword(category_id, term, weight, lang)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (category_id, term, lang) DO NOTHING
                        """,
                        (code_cid, t, 1.0, "bn"),
                    )
                except Exception:
                    pass

            # Fetch articles linked to this auto category
            cur = await con.execute(
                "SELECT article_id, COALESCE(score, 1.0) FROM auto_category_article WHERE category_id = %s",
                (cat_id,),
            )
            links = await cur.fetchall()

            # Insert a new classification_run for provenance
            run_id = (await (await con.execute(
                "INSERT INTO classification_run(model) VALUES(%s) RETURNING run_id",
                (f"auto_promote:{cat.get('model_name') or 'unknown'}",)
            )).fetchone())[0]

            for aid, score in links:
                try:
                    await con.execute(
                        """
                        INSERT INTO article_label(article_id, category_id, score, source, run_id, is_primary)
                        VALUES (%s, %s, %s, %s::label_source, %s, %s)
                        ON CONFLICT (article_id, category_id, source) DO UPDATE
                        SET score=EXCLUDED.score, run_id=EXCLUDED.run_id
                        """,
                        (aid, code_cid, float(score or 1.0), "ml", run_id, False),
                    )
                    await con.execute("UPDATE article SET status='classified' WHERE article_id=%s", (aid,))
                    saved += 1
                except Exception:
                    pass
        except Exception as e:
            raise HTTPException(500, f"auto_promote_failed: {e}")

    return {"ok": True, "promoted_label": label, "codebook_category_id": code_cid, "assigned": saved}

# @app.post("/api/auto/assign")
# async def api_auto_assign(
#     embeddings: bool = True,
#     threshold: float = 0.38,
#     cluster_unassigned: bool = False,
#     min_cluster_size: int = 20,
#     emb_model: Optional[str] = None,
# ):
#     args = [
#         "python", AUTO_PIPELINE_SCRIPT,
#         "--mode", "update",
#         "--dsn", DB_DSN,
#         "--threshold", str(threshold),
#         "--min-cluster-size", str(min_cluster_size),
#     ]
#     if embeddings:
#         args.append("--use-embeddings")
#     if cluster_unassigned:
#         args.append("--cluster-unassigned")
#     if emb_model:
#         args += ["--emb-model", emb_model]

#     env = os.environ.copy()
#     env["PYTHONWARNINGS"] = "ignore::FutureWarning:sklearn"

#     proc = await asyncio.to_thread(
#         subprocess.run, args,
#         capture_output=True, text=True, timeout=1800, env=env
#     )
#     logs = (proc.stdout or "") + "\n" + (proc.stderr or "")
#     return {"ok": (proc.returncode == 0), "logs": logs[-5000:], "args": args}

# @app.post("/api/auto/init")
# async def api_auto_init(
#     embeddings: bool = True,
#     min_cluster_size: int = 20,
#     limit: int | None = None,
#     emb_model: Optional[str] = None,
# ):
#     args = [
#         "python", AUTO_PIPELINE_SCRIPT,
#         "--mode", "init",
#         "--dsn", DB_DSN,
#         "--min-cluster-size", str(min_cluster_size),
#     ]
#     if embeddings:
#         args.append("--use-embeddings")
#     if limit is not None:
#         args += ["--limit", str(int(limit))]
#     if emb_model:
#         args += ["--emb-model", emb_model]

#     env = os.environ.copy()
#     env["PYTHONWARNINGS"] = "ignore::FutureWarning:sklearn"

#     proc = await asyncio.to_thread(
#         subprocess.run, args,
#         capture_output=True, text=True, timeout=3600, env=env
#     )
#     logs = (proc.stdout or "") + "\n" + (proc.stderr or "")
#     return {"ok": (proc.returncode == 0), "logs": logs[-8000:], "args": args}

# --- Rebuild (init) ---
@app.post("/api/auto/init")
async def api_auto_init(
    embeddings: bool = True,
    min_cluster_size: int = 20,
    limit: int | None = None,
    model_name: str | None = None,
    current_user=Depends(require_admin),
):
    args = ["python", AUTO_PIPELINE_SCRIPT, "--mode", "init", "--dsn", DB_DSN, "--min-cluster-size", str(min_cluster_size)]
    if embeddings:
        args.append("--use-embeddings")
    if limit is not None:
        args += ["--limit", str(int(limit))]
    if model_name:
        args += ["--emb-model", model_name]

    # env = os.environ.copy()
    # # silence sklearn FutureWarnings only
    # env["PYTHONWARNINGS"] = "ignore::FutureWarning:sklearn"

    # proc = await asyncio.to_thread(
    #     subprocess.run, args,
    #     capture_output=True, text=True, timeout=3600, env=env
    # )

    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "ignore::FutureWarning:sklearn"
    env["UMAP_PARALLEL"] = "1"                      # let UMAP parallelize
    env["NUMBA_NUM_THREADS"] = str(os.cpu_count())  # UMAP/Numba threads
    env["OMP_NUM_THREADS"] = str(os.cpu_count())    # BLAS/OpenMP
    env["OPENBLAS_NUM_THREADS"] = str(os.cpu_count())
    env["MKL_NUM_THREADS"] = str(os.cpu_count())
    env["TOKENIZERS_PARALLELISM"] = "false"         # quiet HF tokenizers

    proc = await asyncio.to_thread(
        subprocess.run, args,
        capture_output=True, text=True, timeout=3600, env=env
    )

    logs = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return {"ok": (proc.returncode == 0), "logs": logs[-10000:]}

# --- Assign/update ---
@app.post("/api/auto/assign")
async def api_auto_assign(
    embeddings: bool = True,
    threshold: float = 0.38,
    cluster_unassigned: bool = False,
    min_cluster_size: int = 20,
    model_name: str | None = None,
    current_user=Depends(require_admin),
):
    args = ["python", AUTO_PIPELINE_SCRIPT, "--mode", "update", "--dsn", DB_DSN,
            "--threshold", str(threshold), "--min-cluster-size", str(min_cluster_size)]
    if embeddings:
        args.append("--use-embeddings")
    if cluster_unassigned:
        args.append("--cluster-unassigned")
    if model_name:
        args += ["--emb-model", model_name]

    # env = os.environ.copy()
    # env["PYTHONWARNINGS"] = "ignore::FutureWarning:sklearn"

    # proc = await asyncio.to_thread(
    #     subprocess.run, args,
    #     capture_output=True, text=True, timeout=1800, env=env
    # )

    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "ignore::FutureWarning:sklearn"
    env["UMAP_PARALLEL"] = "1"                      # let UMAP parallelize
    env["NUMBA_NUM_THREADS"] = str(os.cpu_count())  # UMAP/Numba threads
    env["OMP_NUM_THREADS"] = str(os.cpu_count())    # BLAS/OpenMP
    env["OPENBLAS_NUM_THREADS"] = str(os.cpu_count())
    env["MKL_NUM_THREADS"] = str(os.cpu_count())
    env["TOKENIZERS_PARALLELISM"] = "false"         # quiet HF tokenizers

    proc = await asyncio.to_thread(
        subprocess.run, args,
        capture_output=True, text=True, timeout=3600, env=env
    )

    
    logs = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return {"ok": (proc.returncode == 0), "logs": logs[-10000:]}

@app.get("/api/auto/breakdown")
async def api_auto_breakdown(article_id: int, category_id: int, current_user=Depends(require_user)):
    cat = await _auto_fetch_category(category_id)
    if not cat:
        raise HTTPException(404, "Category not found")
    art = await _fetch_article(article_id)
    fields_src = {
        "title": art.get("title") or "",
        "content": (art.get("content_clean") or art.get("content_raw") or ""),
    }
    fields_norm = {k: _norm(v) for k, v in fields_src.items()}
    terms = list(cat.get("top_terms") or [])

    # term hits
    field_rows = []
    for fname in ("title", "content"):
        text_src = fields_src[fname]
        text_norm = fields_norm[fname]
        hits = []
        for term in terms:
            if not term:
                continue
            cnt = text_norm.count(term.lower()) if " " in term else len(re.findall(rf"(?<!\w){re.escape(term.lower())}(?!\w)", text_norm, flags=re.UNICODE))
            if cnt > 0:
                hits.append({"term": term, "count": cnt, "snippet": _snippet(text_src, term)})
        field_rows.append({"field": fname, "hits": hits, "cosine": None})

    # cosine per field if we have a centroid + model
    centroid = await _auto_load_centroid(category_id)
    if centroid:
        try:
            from sentence_transformers import SentenceTransformer  # ensure import
            model = await _get_emb_model()
            if model is not None:
                pref = _prefix_for_model(getattr(model, "model_card", None) or getattr(model, "model_name_or_path", ""))
                enc = await asyncio.to_thread(
                    model.encode,
                    [ (pref + fields_src["title"]) if pref else fields_src["title"],
                      (pref + fields_src["content"]) if pref else fields_src["content"] ],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                vecs = [list(map(float, v)) for v in enc]
                field_rows[0]["cosine"] = _cosine(vecs[0], centroid)
                field_rows[1]["cosine"] = _cosine(vecs[1], centroid)
        except Exception as e:
            field_rows.append({"warn":"embed failure", "error":repr(e)})

    return {
        "article_id": article_id,
        "category_id": category_id,
        "category_label": cat["label"],
        "top_terms": terms,
        "fields": field_rows,
    }

@app.get("/api/auto/breakdown/best")
async def api_auto_breakdown_best(article_id: int, current_user=Depends(require_user)):
    cat_id = await _auto_best_category_for_article(article_id)
    if cat_id is None:
        raise HTTPException(404, "No auto category for this article yet")
    return await api_auto_breakdown(article_id=article_id, category_id=cat_id, current_user=current_user)


# =========================================================
# Aggregates + search for dashboard pages
# =========================================================

async def _fetch_category_stats(status: Optional[str], primary_only: bool, limit: int,
                               start_dt: Optional[date], end_dt: Optional[date]) -> List[Dict[str, Any]]:
    where = []
    params: List[Any] = []
    if status:
        where.append("a.status = %s")
        params.append(status)
    if primary_only:
        where.append("al.is_primary IS TRUE")
    _apply_date_filters(where, params, start_dt, end_dt)
    where_sql = (" WHERE " + " AND ".join(where)) if where else ""
    sql = f"""
        SELECT c.category,
               COUNT(*) AS total,
               COUNT(*) FILTER (WHERE al.is_primary IS TRUE) AS primary_count,
               MAX(a.published_at) AS last_published_at
        FROM article_label al
        JOIN codebook_category c ON c.category_id = al.category_id
        JOIN article a ON a.article_id = al.article_id
        {where_sql}
        GROUP BY c.category
        ORDER BY primary_count DESC, total DESC, c.category
        LIMIT %s
    """
    params2 = params + [int(limit)]
    async with await _get_conn() as con:
        cur = await con.execute(sql, tuple(params2))
        rows = await cur.fetchall()
    items = []
    for r in rows:
        items.append({
            "category": r[0],
            "total": int(r[1]),
            "primary": int(r[2]),
            "last_published_at": r[3],
        })
    return items


async def _fetch_portal_stats(status: Optional[str], primary_only: bool,
                              start_dt: Optional[date], end_dt: Optional[date]) -> List[Dict[str, Any]]:
    where = []
    params: List[Any] = []
    if status:
        where.append("a.status = %s")
        params.append(status)
    if primary_only:
        where.append("al.is_primary IS TRUE")
    _apply_date_filters(where, params, start_dt, end_dt)
    where_sql = (" WHERE " + " AND ".join(where)) if where else ""

    # totals per portal
    sql_totals = f"""
        SELECT a.portal_id, COUNT(*) AS total, COUNT(*) FILTER (WHERE al.is_primary) AS primary_count
        FROM article a
        LEFT JOIN article_label al ON al.article_id = a.article_id
        {where_sql}
        GROUP BY a.portal_id
    """
    # category mix per portal (primary labels)
    sql_mix = f"""
        SELECT a.portal_id, c.category, COUNT(*) AS cnt
        FROM article a
        JOIN article_label al ON al.article_id = a.article_id
        JOIN codebook_category c ON c.category_id = al.category_id
        {where_sql + (" AND" if where_sql else " WHERE")} al.is_primary IS TRUE
        GROUP BY a.portal_id, c.category
    """
    async with await _get_conn() as con:
        totals_rows = await (await con.execute(sql_totals, tuple(params))).fetchall()
        mix_rows = await (await con.execute(sql_mix, tuple(params))).fetchall()
        portals_rows = await (await con.execute("SELECT portal_id, name FROM portal")).fetchall()

    mix_map: Dict[Any, Dict[str, int]] = {}
    for pid, cat, cnt in mix_rows:
        if pid not in mix_map:
            mix_map[pid] = {}
        mix_map[pid][cat] = int(cnt)

    name_map = {r[0]: r[1] for r in portals_rows}
    items = []
    for pid, total, primary in totals_rows:
        items.append({
            "portal_id": pid,
            "portal": name_map.get(pid) or f"Portal {pid}",
            "total": int(total),
            "primary": int(primary),
            "by_category": mix_map.get(pid, {}),
        })
    # include portals with zero if desired
    for pid, name in name_map.items():
        if not any(it["portal_id"] == pid for it in items):
            items.append({"portal_id": pid, "portal": name, "total": 0, "primary": 0, "by_category": {}})
    items.sort(key=lambda x: x["total"], reverse=True)
    return items


async def _fetch_text_key_stats(field: str, status: Optional[str], primary_only: bool,
                               category: Optional[str] = None,
                               start_dt: Optional[date] = None, end_dt: Optional[date] = None) -> List[Dict[str, Any]]:
    """
    Generic stats for party / candidate / region fields on article.
    """
    if field not in {"party", "candidate", "region"}:
        return []
    where = [f"a.{field} IS NOT NULL", f"NULLIF(TRIM(a.{field}), '') IS NOT NULL"]
    params: List[Any] = []
    if status:
        where.append("a.status = %s")
        params.append(status)
    if primary_only:
        where.append("al.is_primary IS TRUE")
    _apply_date_filters(where, params, start_dt, end_dt)
    where_sql = " WHERE " + " AND ".join(where)
    sql = f"""
        SELECT a.{field} AS key,
               COUNT(*) AS total,
               COUNT(*) FILTER (WHERE al.is_primary) AS primary_count,
               jsonb_object_agg(c.category, cnt) AS by_category
        FROM (
            SELECT a.{field}, al.category_id, COUNT(*) AS cnt
            FROM article a
            JOIN article_label al ON al.article_id = a.article_id
            {where_sql}
            GROUP BY a.{field}, al.category_id
        ) sub
        JOIN codebook_category c ON c.category_id = sub.category_id
        JOIN article a ON a.{field} = sub.{field}
        JOIN article_label al ON al.article_id = a.article_id AND al.category_id = sub.category_id
        {where_sql}
        GROUP BY a.{field}
        ORDER BY total DESC, key
    """
    async with await _get_conn() as con:
        cur = await con.execute(sql, tuple(params*2))  # params used twice
        rows = await cur.fetchall()
    items = []
    for r in rows:
        items.append({
            field: r[0],
            "total": int(r[1]),
            "primary": int(r[2]),
            "by_category": r[3] or {},
        })
    if category:
        for it in items:
            cat_cnt = int((it.get("by_category") or {}).get(category, 0))
            it["primary"] = cat_cnt
            it["total"] = cat_cnt
    return items


@app.get("/api/stats/categories")
async def api_stats_categories(status: Optional[str] = Query(None, pattern="^(queued|new|needs_review|classified)$"),
                               primary_only: bool = True,
                               limit: int = Query(200, ge=1, le=2000),
                               start_date: Optional[str] = None,
                               end_date: Optional[str] = None,
                               current_user=Depends(require_user)):
    start_dt = _parse_date(start_date)
    end_dt = _parse_date(end_date)
    return {"items": await _fetch_category_stats(status, primary_only, limit, start_dt, end_dt)}


@app.get("/api/stats/media")
async def api_stats_media(status: Optional[str] = Query(None, pattern="^(queued|new|needs_review|classified)$"),
                          primary_only: bool = True,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None,
                          current_user=Depends(require_user)):
    start_dt = _parse_date(start_date)
    end_dt = _parse_date(end_date)
    return {"items": await _fetch_portal_stats(status, primary_only, start_dt, end_dt)}


@app.get("/api/stats/party")
async def api_stats_party(status: Optional[str] = Query(None, pattern="^(queued|new|needs_review|classified)$"),
                          primary_only: bool = True,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None,
                          current_user=Depends(require_user)):
    start_dt = _parse_date(start_date)
    end_dt = _parse_date(end_date)
    return {"items": await _fetch_text_key_stats("party", status, primary_only, None, start_dt, end_dt)}


@app.get("/api/stats/candidate")
async def api_stats_candidate(status: Optional[str] = Query(None, pattern="^(queued|new|needs_review|classified)$"),
                              primary_only: bool = True,
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None,
                              current_user=Depends(require_user)):
    start_dt = _parse_date(start_date)
    end_dt = _parse_date(end_date)
    return {"items": await _fetch_text_key_stats("candidate", status, primary_only, None, start_dt, end_dt)}


@app.get("/api/stats/geography")
async def api_stats_geography(status: Optional[str] = Query(None, pattern="^(queued|new|needs_review|classified)$"),
                              primary_only: bool = True,
                              category: Optional[str] = None,
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None,
                              current_user=Depends(require_user)):
    start_dt = _parse_date(start_date)
    end_dt = _parse_date(end_date)
    items = await _fetch_text_key_stats("region", status, primary_only, category=category, start_dt=start_dt, end_dt=end_dt)
    for it in items:
        it["division"] = _infer_division(it.get("region") or it.get("key"))
    return {"items": items}


@app.get("/api/articles/search")
async def api_articles_search(
    category: Optional[str] = None,
    portal_id: Optional[int] = None,
    party: Optional[str] = None,
    candidate: Optional[str] = None,
    region: Optional[str] = None,
    title_contains: Optional[str] = None,
    status: Optional[str] = Query(None, pattern="^(queued|new|needs_review|classified)$"),
    candidate_missing: bool = False,
    primary_only: bool = True,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = Query(200, ge=1, le=2000),
    offset: int = Query(0, ge=0, le=5000),
    current_user=Depends(require_user),
):
    where = []
    params: List[Any] = []
    start_dt = _parse_date(start_date)
    end_dt = _parse_date(end_date)

    if status:
        where.append("a.status = %s")
        params.append(status)
    if portal_id is not None:
        where.append("a.portal_id = %s")
        params.append(int(portal_id))
    if party:
        where.append("LOWER(a.party) = LOWER(%s)")
        params.append(party.strip())
    if candidate:
        where.append("LOWER(a.candidate) = LOWER(%s)")
        params.append(candidate.strip())
    if region:
        where.append("LOWER(a.region) = LOWER(%s)")
        params.append(region.strip())
    if candidate_missing:
        where.append("(a.candidate IS NULL OR NULLIF(TRIM(a.candidate), '') IS NULL)")
    if title_contains:
        where.append("a.title ILIKE %s")
        params.append(f"%{title_contains}%")
    if category:
        where.append("c.category = %s")
        params.append(category)
    if primary_only:
        where.append("COALESCE(al.is_primary, TRUE) IS TRUE")
    _apply_date_filters(where, params, start_dt, end_dt)

    where_sql = " WHERE " + " AND ".join(where) if where else ""

    sql = f"""
        WITH labeled AS (
          SELECT
            a.article_id, a.title, a.url, a.published_at, a.status, a.lang,
            a.portal_id, a.party, a.candidate, a.region,
            al.category_id, c.category, al.score, al.is_primary,
            ROW_NUMBER() OVER (
              PARTITION BY a.article_id
              ORDER BY COALESCE(al.is_primary, FALSE) DESC, al.score DESC NULLS LAST, al.category_id
            ) AS rn
          FROM article a
          LEFT JOIN article_label al ON al.article_id = a.article_id
          LEFT JOIN codebook_category c ON c.category_id = al.category_id
          {where_sql}
        )
        SELECT l.article_id, l.title, l.url, l.published_at, l.status, l.lang,
               l.portal_id, p.name AS portal_name,
               l.party, l.candidate, l.region,
               l.category, l.score, l.is_primary
        FROM labeled l
        LEFT JOIN portal p ON p.portal_id = l.portal_id
        WHERE l.rn = 1
        ORDER BY l.published_at DESC NULLS LAST, l.article_id DESC
        LIMIT %s OFFSET %s
    """
    params2 = params + [int(limit), int(offset)]
    async with await _get_conn() as con:
        cur = await con.execute(sql, tuple(params2))
        rows = await cur.fetchall()

    items = []
    for r in rows:
        items.append({
            "article_id": r[0],
            "title": r[1],
            "url": r[2],
            "published_at": r[3],
            "status": r[4],
            "lang": r[5],
            "portal_id": r[6],
            "portal": r[7],
            "party": r[8],
            "candidate": r[9],
            "region": r[10],
            "category": r[11],
            "score": float(r[12]) if r[12] is not None else None,
            "is_primary": bool(r[13]) if r[13] is not None else None,
        })
    return {"items": items}


@app.get("/api/articles/classified")
async def api_articles_classified(
    status: Optional[str] = Query("classified", pattern="^(queued|new|needs_review|classified)$"),
    primary_only: bool = True,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = Query(200, ge=1, le=2000),
    offset: int = Query(0, ge=0, le=5000),
    current_user=Depends(require_user),
):
    where = []
    params: List[Any] = []
    start_dt = _parse_date(start_date)
    end_dt = _parse_date(end_date)

    if status:
        where.append("a.status = %s")
        params.append(status)
    if primary_only:
        where.append("COALESCE(al.is_primary, TRUE) IS TRUE")
    _apply_date_filters(where, params, start_dt, end_dt)

    where_sql = " WHERE " + " AND ".join(where) if where else ""

    sql = f"""
        WITH labeled AS (
          SELECT
            a.article_id, a.title, a.url, a.published_at, a.status, a.lang,
            a.portal_id, a.party, a.candidate, a.region,
            al.category_id, c.category, al.score, al.is_primary, al.source,
            ROW_NUMBER() OVER (
              PARTITION BY a.article_id
              ORDER BY COALESCE(al.is_primary, FALSE) DESC, al.score DESC NULLS LAST, al.category_id
            ) AS rn
          FROM article a
          LEFT JOIN article_label al ON al.article_id = a.article_id
          LEFT JOIN codebook_category c ON c.category_id = al.category_id
          {where_sql}
        )
        SELECT l.article_id, l.title, l.url, l.published_at, l.status, l.lang,
               l.portal_id, p.name AS portal_name,
               l.party, l.candidate, l.region, a.content_clean,
               l.category, l.score, l.is_primary, l.source
        FROM labeled l
        LEFT JOIN portal p ON p.portal_id = l.portal_id
        LEFT JOIN article a ON a.article_id = l.article_id
        WHERE l.rn = 1 AND l.category IS NOT NULL
        ORDER BY l.published_at DESC NULLS LAST, l.article_id DESC
        LIMIT %s OFFSET %s
    """
    params2 = params + [int(limit), int(offset)]
    async with await _get_conn() as con:
        cur = await con.execute(sql, tuple(params2))
        rows = await cur.fetchall()

    cats = []
    cand_bn_map = await _candidate_bn_map()
    items = []
    texts: Dict[int, str] = {}
    for r in rows:
        cat_val = r[12]
        cats.append(cat_val)
        cand_val = r[9]
        cand_bn = cand_bn_map.get(_norm_name(cand_val)) if cand_val else None
        mode_val = r[15]
        content_val = r[11] or ""
        title_val = r[1] or ""
        text_for_match = " ".join([title_val, content_val]).strip()
        aid = int(r[0])
        items.append({
            "article_id": aid,
            "title": title_val,
            "url": r[2],
            "published_at": r[3],
            "status": r[4],
            "lang": r[5],
            "portal_id": r[6],
            "portal": r[7],
            "party": r[8],
            "candidate": cand_val,
            "candidate_bn": cand_bn or cand_val,
            "region": r[10],
            "category": cat_val,
            "score": float(r[13]) if r[13] is not None else None,
            "is_primary": bool(r[14]) if r[14] is not None else None,
            "mode": mode_val,
        })
        texts[aid] = text_for_match

    kw_map = await _fetch_keywords_for_categories(cats, limit_terms=50)
    for it in items:
        terms_all = kw_map.get(it.get("category"), [])
        matched = _match_terms_in_text(texts.get(it["article_id"]), terms_all, limit_terms=12)
        it["keywords"] = terms_all
        it["terms"] = matched or terms_all
        it["mode_terms"] = matched or terms_all

    return {"items": items}

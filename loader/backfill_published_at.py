#!/usr/bin/env python3
"""
Backfill article.published_at:
- Try to read the NEWS_XLSX and match rows by URL to set published_at from the source Date column.
- Any remaining NULL published_at rows fall back to created_at.
"""

import os
import sys
from typing import Any, Dict
import pandas as pd
import psycopg

# Ensure loader module is importable when run from repo root
HERE = os.path.abspath(os.path.dirname(__file__))
PARENT = os.path.abspath(os.path.join(HERE, ".."))
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from loader.load_from_xlsx import _pick_date_field, normalize_date  # type: ignore

DB_DSN = os.getenv("DB_DSN", "postgresql://postgres:postgres@db:5432/candidate_news")
NEWS_XLSX = os.getenv("NEWS_XLSX", "/data/Prothom Alo 2024 Election News.xlsx")


def _build_url_date_map() -> Dict[str, Any]:
    if not os.path.exists(NEWS_XLSX):
        print(f"[warn] NEWS_XLSX not found at {NEWS_XLSX}; will only fallback to created_at.")
        return {}
    df = pd.read_excel(NEWS_XLSX, sheet_name=0)
    url_to_date: Dict[str, Any] = {}
    for _, r in df.iterrows():
        url = (r.get("Link") or "").strip()
        if not url:
            continue
        dt = normalize_date(_pick_date_field(r))
        if dt:
            url_to_date[url] = dt
    return url_to_date


def backfill():
    url_dates = _build_url_date_map()
    updated_from_excel = 0
    updated_from_created = 0

    con = psycopg.connect(DB_DSN)
    with con:
        con.execute("BEGIN")
        if url_dates:
            for url, dt in url_dates.items():
                cur = con.execute(
                    """
                    UPDATE article
                    SET published_at = %s
                    WHERE url = %s AND published_at IS NULL
                    """,
                    (dt, url),
                )
                updated_from_excel += cur.rowcount

        # fallback: set remaining NULL to created_at
        cur = con.execute(
            "UPDATE article SET published_at = created_at WHERE published_at IS NULL"
        )
        updated_from_created = cur.rowcount
        con.execute("COMMIT")

    print(f"[ok] backfill done. from_excel={updated_from_excel}, from_created_at={updated_from_created}")


if __name__ == "__main__":
    backfill()

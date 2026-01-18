from fastapi import FastAPI, Request
import os, re, unicodedata
import psycopg

app = FastAPI()

DB_DSN = os.getenv("DB_DSN", "postgresql://postgres:postgres@db:5432/candidate_news")

TOOLS = {"classify_text": {"name": "classify_text"}, "save_labels": {"name": "save_labels"}}

def _norm(s: str) -> str:
    return unicodedata.normalize("NFKC", (s or "")).lower()

_SPACE_RX = re.compile(r"\s+", flags=re.UNICODE)

def _count_term(text: str, term: str) -> int:
    """Count matches for a term using Unicode-aware boundaries.
    - For single-token terms: use word boundaries.
    - For multi-token terms: collapse internal spaces in term to \s+ and match with boundaries.
    """
    if not text or not term:
        return 0
    t = (term or "").strip().lower()
    if not t or len(t) < 2:
        return 0
    if " " in t:
        # Replace spaces inside term with \s+ to match any whitespace
        pat = re.escape(t)
        pat = _SPACE_RX.sub(r"\\s+", pat)
        rx = re.compile(rf"(?<!\w){pat}(?!\w)", flags=re.UNICODE)
        return len(rx.findall(text))
    # whole-token-ish match for single terms
    return len(re.findall(rf"(?<!\w){re.escape(t)}(?!\w)", text, flags=re.UNICODE))

def _fetch_article(conn, aid: int):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT article_id, COALESCE(title,''), COALESCE(content_clean, content_raw, ''), COALESCE(lang,'bn')
            FROM article WHERE article_id=%s
            """,
            (aid,),
        )
        r = cur.fetchone()
    if not r:
        return None
    return {"article_id": r[0], "title": r[1], "content": r[2], "lang": (r[3] or "bn").lower()}

def _fetch_keywords(conn, lang: str | None = None):
    sql = (
        """
        SELECT c.category, k.term, COALESCE(k.weight,1.0)
        FROM codebook_keyword k
        JOIN codebook_category c ON c.category_id = k.category_id
        """
        + (" WHERE LOWER(COALESCE(k.lang,'bn')) = %s" if lang else "")
    )
    with conn.cursor() as cur:
        cur.execute(sql, ((lang.lower(),) if lang else tuple()))
        rows = cur.fetchall()
    by_cat = {}
    for cat, term, weight in rows:
        t = (term or "").strip()
        if not t or len(t) < 2:
            continue
        by_cat.setdefault(cat, []).append((t, float(weight)))
    return by_cat

def _save_labels(conn, article_id: int, labels):
    # Persist into official table article_label using category name lookup
    with conn.cursor() as cur:
        cur.execute("INSERT INTO classification_run(model) VALUES (%s) RETURNING run_id", ("mcp:rule",))
        run_id = cur.fetchone()[0]
        for lab in labels or []:
            cat = lab.get("category"); score = float(lab.get("score", 0) or 0)
            is_primary = bool(lab.get("is_primary", False))
            src = (lab.get("source") or "rule").lower()
            cur.execute(
                """
                INSERT INTO article_label(article_id, category_id, score, source, run_id, is_primary)
                SELECT %s, category_id, %s, %s::label_source, %s, %s
                FROM codebook_category WHERE category=%s
                ON CONFLICT (article_id, category_id, source) DO UPDATE
                SET score=EXCLUDED.score, run_id=EXCLUDED.run_id, is_primary=EXCLUDED.is_primary
                """,
                (article_id, score, src, run_id, is_primary, cat),
            )
        cur.execute("UPDATE article SET status='classified' WHERE article_id=%s", (article_id,))

def _classify_rule(conn, article_id: int):
    art = _fetch_article(conn, article_id)
    if not art:
        return []
    title = _norm(art["title"]) or ""
    content = _norm(art["content"]) or ""
    text = (title + "\n" + content).strip()
    kw = _fetch_keywords(conn, lang=art.get("lang"))
    scores = []
    TITLE_BOOST = 2.0  # weight title hits higher
    for cat, terms in kw.items():
        s = 0.0
        for term, w in terms:
            tnorm = _norm(term)
            if not tnorm or len(tnorm) < 2:
                continue
            cnt_title = _count_term(title, tnorm)
            cnt_body = _count_term(content, tnorm)
            if cnt_title or cnt_body:
                s += float(w) * (TITLE_BOOST * cnt_title + cnt_body)
        if s > 0:
            scores.append({"category": cat, "score": s, "source": "rule"})
    scores.sort(key=lambda x: x["score"], reverse=True)
    if scores:
        # normalize top to 1.0
        top = max(x["score"] for x in scores) or 1.0
        for x in scores:
            x["score"] = float(x["score"]) / float(top)
        scores[0]["is_primary"] = True
    return scores[:10]

@app.get("/health")
async def health():
    return {
        "ok": True,
        "tools": list(TOOLS),
        "provider": os.getenv("MODEL_PROVIDER"),
        "model": os.getenv("MODEL_ID"),
        "has_key": bool(os.getenv("MODEL_API_KEY")),
    }

def register_tools():
    TOOLS["classify_text"] = {"name": "classify_text", "description": "Rule-based (keywords)"}
    if os.getenv("ENABLE_ML", "false").lower() == "true":
        TOOLS["classify_ml"] = {"name": "classify_ml", "description": "ML"}
    TOOLS["save_labels"] = {"name": "save_labels", "description": "Persist labels"}

register_tools()

@app.post("/mcp")
async def mcp_endpoint(request: Request):
    body = await request.json()
    rid = body.get("id"); method = body.get("method")
    try:
        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": rid, "result": {"tools": list(TOOLS.values())}}
        if method == "tools/call":
            params = body.get("params") or {}
            name = params.get("name"); args = params.get("arguments") or {}
            if name not in TOOLS:
                return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"Unknown tool {name}"}}
            # DB connection per call (fast for psycopg 3)
            with psycopg.connect(DB_DSN) as conn:
                if name == "classify_text":
                    aid = int(args.get("article_id"))
                    return {"jsonrpc": "2.0", "id": rid, "result": {"labels": _classify_rule(conn, aid)}}
                if name == "save_labels":
                    aid = int(args.get("article_id")); labels = args.get("labels") or []
                    _save_labels(conn, aid, labels)
                    return {"jsonrpc": "2.0", "id": rid, "result": {"ok": True}}
                if name == "classify_ml":
                    # Not enabled: return empty labels so callers can fall back
                    return {"jsonrpc": "2.0", "id": rid, "result": {"labels": []}}
            return {"jsonrpc": "2.0", "id": rid, "result": {}}
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": "Unknown method"}}
    except Exception as e:
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32000, "message": str(e)}}

import os, time, httpx
import requests, json

import os, time, requests

import psycopg, asyncio

DB_DSN = os.getenv("DB_DSN", "postgresql://postgres:postgres@db:5432/candidate_news")
BACKEND = os.getenv("BACKEND_URL", "http://backend:8000")

MCP_URL = os.getenv("MCP_URL","http://mcp:5000/mcp")
MODE = os.getenv("CLASSIFIER_MODE", "hybrid")  # rule | ml | hybrid
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "stub")  # "stub" | "openai" | "hf"
MODEL_ID = os.getenv("MODEL_ID", "")   
SERVICE_TOKEN = os.getenv("SERVICE_TOKEN")
AUTH_HEADERS = {"Authorization": f"Bearer {SERVICE_TOKEN}"} if SERVICE_TOKEN else {}



RULE_PRIMARY = float(os.getenv("RULE_PRIMARY","0.60"))
ML_PRIMARY   = float(os.getenv("ML_PRIMARY","0.55"))
BATCH = int(os.getenv("ENRICH_BATCH","100"))
SLEEP = float(os.getenv("ENRICH_SLEEP","1.0"))



def classify(aid: int, mode: str = "hybrid"):
    r = requests.post(f"{BACKEND}/api/classify/{aid}", params={"mode": mode}, timeout=30, headers=AUTH_HEADERS)
    r.raise_for_status()
    return r.json()  # {"ok":true,"mode":"...","labels":[...]}

# … in your worker loop …
res = classify(article_id, "hybrid")
print("[enricher] got", res)



def call(name, arguments):
    payload = {"jsonrpc":"2.0","id":"1","method":"tools/call","params":{"name":name,"arguments":arguments}}
    r = httpx.post(MCP_URL, json=payload, timeout=60)
    r.raise_for_status()
    return r.json().get("result",{})

def _mcp_call(name: str, arguments: dict | None = None, timeout: int = 60) -> dict:
    payload = {
        "jsonrpc": "2.0", "id": "1",
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments or {}},
    }
    r = requests.post(MCP_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json().get("result", {}) or {}

def _merge_labels(*label_lists):
    """Union by category; keep the highest score. Then mark the single best as primary."""
    bycat = {}
    for labs in label_lists:
        for lab in labs or []:
            cat = lab["category"]
            cur = bycat.get(cat)
            if cur is None or float(lab.get("score", 0)) > float(cur.get("score", 0)):
                bycat[cat] = {
                    "category": cat,
                    "score": float(lab.get("score", 0)),
                    "source": lab.get("source", "rule"),  # keep original source
                    "is_primary": False,
                }
    if bycat:
        max(bycat.values(), key=lambda x: x["score"])["is_primary"] = True
    return list(bycat.values())

def classify_and_save(aid: int):
    """Run rule / ml / hybrid, save via MCP, return final labels."""
    if MODE == "ml":
        ml = _mcp_call("classify_ml", {"article_id": aid}).get("labels", [])
        final = _merge_labels(ml)
    elif MODE == "hybrid":
        rl = _mcp_call("classify_text", {"article_id": aid}).get("labels", [])
        ml = _mcp_call("classify_ml",  {"article_id": aid}).get("labels", [])
        final = _merge_labels(rl, ml)
    else:  # "rule"
        rl = _mcp_call("classify_text", {"article_id": aid}).get("labels", [])
        final = _merge_labels(rl)

    if final:
        _mcp_call("save_labels", {"article_id": aid, "labels": final})
    return final



def main_loop():
    print("[enricher] starting worker...")
    while True:
        try:
            items = call("fetch_queue", {"limit":25}).get("items",[])
            if not items:
                print("[enricher] idle, no items; sleeping 20s")
                time.sleep(20); continue
            # for it in items:
            #     aid = it["article_id"]
            #     print(f"[enricher] classifying article_id={aid}")
            #     res = call("classify_text", {"article_id": aid})
            #     labels = res.get("labels",[])
            #     call("save_labels", {"article_id": aid, "labels": labels})
            #     print(f"[enricher] saved labels for article_id={aid}: {labels}")
            for it in items:
                aid = it["article_id"]
                print(f"[enricher] classifying article_id={aid} mode={MODE}")
                labels = classify_and_save(aid)
                print(f"[enricher] saved labels for {aid}: {labels}")
        except Exception as e:
            print("[enricher] ERROR:", e)
            time.sleep(10)


# Added for Generate Category
async def _pick_ids(limit=BATCH):
    async with await psycopg.AsyncConnection.connect(DB_DSN) as con:
        cur = await con.execute("""
            SELECT a.article_id
            FROM article a
            WHERE a.status IN ('new','queued','needs_review')
              AND (a.gen_cats_at IS NULL OR a.updated_at > a.gen_cats_at) -- not generated yet or changed
            ORDER BY a.article_id ASC
            LIMIT %s
        """, (limit,))
        return [r[0] for r in await cur.fetchall()]

async def classify_with_existing(client, aid):
    # 1) rule
    r = await client.post(f"{BACKEND}/api/classify/{aid}", params={"mode":"rule"}, timeout=90, headers=AUTH_HEADERS)
    lr = r.json().get("labels",[]) if r.status_code==200 else []
    best = max(lr, key=lambda x:x.get("score",0), default=None)
    if best and best["score"] >= RULE_PRIMARY:
        return lr, "rule"

    # 2) ml
    r = await client.post(f"{BACKEND}/api/classify/{aid}", params={"mode":"ml"}, timeout=120, headers=AUTH_HEADERS)
    lm = r.json().get("labels",[]) if r.status_code==200 else []
    best = max(lm, key=lambda x:x.get("score",0), default=None)
    if best and best["score"] >= ML_PRIMARY:
        return lm, "ml"

    return (lr or []) + (lm or []), None

async def generate_new_category(client, aid):
    # Use backend’s hybrid generator (which merges rule/ml/llm via MCP) and writes scratch + article_category_label
    r = await client.post(f"{BACKEND}/api/generate_from_article/{aid}",
                          params={"mode":"hybrid","save_official":"false"}, timeout=180, headers=AUTH_HEADERS)
    return r.json()

async def mark_done(aid, mode):
    async with await psycopg.AsyncConnection.connect(DB_DSN) as con:
        await con.execute("""
          UPDATE article SET gen_cats_at = now(), gen_cats_mode = %s::label_source
          WHERE article_id = %s
        """, (mode if mode in ("rule","ml","llm","human") else None, aid))

async def loop():
    async with httpx.AsyncClient() as client:
        while True:
            ids = await _pick_ids()
            if not ids:
                await asyncio.sleep(SLEEP); continue
            for aid in ids:
                try:
                    labels, mode = await classify_with_existing(client, aid)
                    if mode:
                        await mark_done(aid, mode)  # classified via existing taxonomy
                        continue
                    # else try to generate category
                    await generate_new_category(client, aid)
                    await mark_done(aid, "llm")   # provenance: generated via LLM (hybrid)
                except Exception as e:
                    # leave for next cycle; optionally set status='needs_review'
                    pass
            await asyncio.sleep(SLEEP)

if __name__ == "__main__":
    main_loop()

    asyncio.run(loop())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Auto Category Pipeline

Modes:
  --mode init
      - Load all (or limited) articles
      - Build embeddings (if requested), cluster with UMAP+HDBSCAN (fallback KMeans)
      - Extract Unicode/Bangla-aware keyphrases for labels
      - Create categories + centroid vectors + memberships

  --mode update
      - Assign new/unassigned articles to nearest existing centroid (cosine)
      - Optionally cluster leftover unassigned into new categories

Tables expected:
  article(article_id, title, content_clean, content_raw, ...)
  auto_category(id, label, top_terms text[], size int, model_name text, algo text, params_json jsonb, created_at timestamptz default now())
  auto_category_article(category_id int, article_id int, score double precision, rank int, unique (category_id, article_id))
  auto_category_centroid(category_id int primary key, model_name text, dim int, vector double precision[], updated_at timestamptz default now())
  article_embedding(article_id int, model_name text, dim int, vector double precision[], primary key (article_id, model_name))

Env:
  DB_DSN       (default: postgresql://postgres:postgres@localhost:5432/candidate_news)
  EMB_MODEL    (default: intfloat/multilingual-e5-base)
"""

import os
import re
import json
import math
import argparse
import unicodedata
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import psycopg

# --- Optional deps ---
try:
    import umap
except Exception:
    umap = None

try:
    import hdbscan
except Exception:
    hdbscan = None

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer

# Embeddings
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

DEFAULT_DSN = os.getenv("DB_DSN", "postgresql://postgres:postgres@localhost:5432/candidate_news")
DEFAULT_MODEL = os.getenv("EMB_MODEL", "intfloat/multilingual-e5-base")  # strong multilingual semantic model
LLM_MODEL = os.getenv("LLM_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
ENABLE_LLM_LABELS = os.getenv("ENABLE_LLM_LABELS", "true").lower() in ("1", "true", "yes")

# Optional: LLM client for nicer cluster labels
try:
    from openai import OpenAI
    _oa_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception:
    _oa_client = None


# -----------------------------------------------------------------------------
# DB
# -----------------------------------------------------------------------------
@contextmanager
def pg_conn(dsn: str):
    conn = psycopg.connect(dsn, autocommit=True)
    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:
            pass


def ensure_auto_used_column(conn):
    """
    Some older DBs may not yet have article.auto_used; add it so the pipeline can run.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1
            FROM information_schema.columns
            WHERE table_name = 'article' AND column_name = 'auto_used'
            LIMIT 1
            """
        )
        if cur.fetchone():
            return
        print("[info] adding article.auto_used column for auto-category pipeline")
        cur.execute("ALTER TABLE article ADD COLUMN IF NOT EXISTS auto_used BOOLEAN NOT NULL DEFAULT FALSE;")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_article_auto_used ON article(auto_used);")


# -----------------------------------------------------------------------------
# Fetch helpers
# -----------------------------------------------------------------------------
def fetch_articles(conn, limit=None, only_unused: bool = True) -> pd.DataFrame:
    """
    Return DataFrame with columns: id, title, content
    id = article.article_id
    """
    with conn.cursor() as cur:
        sql = """
        SELECT
          a.article_id AS id,
          COALESCE(a.title, '') AS title,
          COALESCE(a.content_clean, a.content_raw, '') AS content
        FROM article a
        WHERE (%s IS FALSE OR COALESCE(a.auto_used, FALSE) IS FALSE)
        ORDER BY a.article_id
        """
        if limit:
            sql += " LIMIT %s"
            cur.execute(sql, (only_unused, int(limit)))
        else:
            cur.execute(sql, (only_unused,))
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=["id", "title", "content"])


def fetch_unassigned(conn, limit=None, only_unused: bool = True) -> pd.DataFrame:
    """
    Return unassigned articles (no row in auto_category_article).
    """
    with conn.cursor() as cur:
        sql = """
        SELECT
          a.article_id AS id,
          COALESCE(a.title, '') AS title,
          COALESCE(a.content_clean, a.content_raw, '') AS content
        FROM article a
        LEFT JOIN auto_category_article aca
          ON aca.article_id = a.article_id
        WHERE aca.article_id IS NULL
          AND (%s IS FALSE OR COALESCE(a.auto_used, FALSE) IS FALSE)
        ORDER BY a.article_id
        """
        if limit:
            sql += " LIMIT %s"
            cur.execute(sql, (only_unused, int(limit)))
        else:
            cur.execute(sql, (only_unused,))
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=["id", "title", "content"])


# -----------------------------------------------------------------------------
# Text building
# -----------------------------------------------------------------------------
def build_texts(df: pd.DataFrame, title_w=2.0, content_w=1.0) -> List[str]:
    out = []
    for _, r in df.iterrows():
        t = ((r["title"] + " ") * int(round(title_w))) + ((r["content"] + " ") * int(round(content_w)))
        out.append(t.strip())
    return out


# -----------------------------------------------------------------------------
# Embeddings
# -----------------------------------------------------------------------------
def load_sentence_model(name: str):
    if SentenceTransformer is None:
        return None
    try:
        return SentenceTransformer(name)
    except Exception as e:
        print(f"[warn] sentence model failed to load: {e}")
        return None


def embed_texts(model: "SentenceTransformer", texts: List[str]) -> np.ndarray:
    """
    Encode texts to L2-normalized float32 vectors.
    Compatible across sentence-transformers versions.
    """
    if model is None:
        raise RuntimeError("No embedding model (use --use-embeddings / set EMB_MODEL).")
    if not isinstance(texts, (list, tuple)):
        texts = list(texts)

    try:
        emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        emb = np.asarray(emb, dtype=np.float32)
    except TypeError:
        emb = model.encode(texts, show_progress_bar=True)
        emb = np.asarray(emb, dtype=np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        emb = (emb / norms).astype(np.float32)
    return emb


# -----------------------------------------------------------------------------
# Dimensionality reduction (optional)
# -----------------------------------------------------------------------------
def reduce_with_umap(emb: np.ndarray, n_components: int = 50, metric: str = "cosine", random_state: int = 42) -> Optional[np.ndarray]:
    if umap is None:
        return None
    if emb.shape[0] < 10:
        return None
    try:
        n_comp = min(n_components, emb.shape[1])
        reducer = umap.UMAP(
            n_components=n_comp,
            metric=metric,
            n_neighbors=30,
            min_dist=0.0,
            random_state=random_state,
        )
        X_red = reducer.fit_transform(emb)
        return X_red.astype(np.float32)
    except Exception as e:
        print(f"[warn] UMAP reduction failed: {e}")
        return None


def mark_auto_used(conn, article_ids: List[int] | Set[int]):
    """
    Mark articles as already used for auto-category generation/assignment to prevent reuse.
    """
    ids = list({int(a) for a in (article_ids or [])})
    if not ids:
        return
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE article SET auto_used = TRUE WHERE article_id = ANY(%s)",
            (ids,),
        )


# -----------------------------------------------------------------------------
# Clustering
# -----------------------------------------------------------------------------
def cluster_embeddings(emb: np.ndarray, min_cluster_size=20) -> np.ndarray:
    """
    Try UMAP + HDBSCAN. Fall back to HDBSCAN on emb. Then to KMeans.
    Returns labels (len = n_samples) with -1 for noise.
    """
    # 1) UMAP + HDBSCAN
    if hdbscan is not None:
        X_red = reduce_with_umap(emb, n_components=50, metric="cosine", random_state=42)
        if X_red is not None:
            try:
                cl = hdbscan.HDBSCAN(
                    min_cluster_size=int(min_cluster_size),
                    min_samples=max(1, int(min_cluster_size // 2)),
                    cluster_selection_method="eom",
                    metric="euclidean",
                )
                return cl.fit_predict(X_red)
            except Exception as e:
                print(f"[warn] HDBSCAN on UMAP failed: {e}")

        # 2) Raw HDBSCAN (cosine distance approximated via euclidean in normalized space works OK)
        try:
            cl = hdbscan.HDBSCAN(
                min_cluster_size=int(min_cluster_size),
                min_samples=max(1, int(min_cluster_size // 2)),
                cluster_selection_method="eom",
                metric="euclidean",
            )
            return cl.fit_predict(emb)
        except Exception as e:
            print(f"[warn] HDBSCAN on embeddings failed: {e}")

    # 3) KMeans fallback
    n = emb.shape[0]
    k_guess = max(6, min(30, max(2, n // max(1, min_cluster_size))))
    print(f"[info] Falling back to KMeans(k={k_guess})")
    km = KMeans(n_clusters=k_guess, n_init="auto", random_state=42)
    return km.fit_predict(emb)


# -----------------------------------------------------------------------------
# Bangla/Unicode-aware tokenization & label extraction
# -----------------------------------------------------------------------------
# a very small Bangla + EN stoplist (extend as needed)
BN_EN_STOP = {
    # English
    "the","a","an","and","or","of","in","on","for","to","from","by","with","as","at",
    "is","are","was","were","be","been","it","this","that","these","those","but","if",
    "not","no","yes","you","we","they","he","she","i","my","our","your","their","its",
    "about","into","after","before","over","under","more","most","less","least","also",
    "new","news","video","photo","photos","pic","pics","via","today","bangladesh","bd",

    # Bangla (starter set; safe to expand)
    "এই","ওই","এটা","ওটা","করে","করছে","করেন","করা","হয়","হয়ে","হচ্ছে","হল","ছিল","ছিলেন","এবং","বা","এর","এই","ও","তে","কে","টি",
    "যা","যে","যিনি","তাঁর","তার","তাদের","তাকে","তাদেরকে","একটি","একজন","একটি","একই","যেখানে","যখন","তখন","কারণ",
}

def _norm_text(s: str) -> str:
    import unicodedata
    return unicodedata.normalize("NFKC", (s or "")).lower()

def _unicode_tokenizer(text: str) -> List[str]:
    """
    Unicode-friendly tokenizer: keep letters/numbers (Bangla included), split otherwise.
    """
    text = _norm_text(text)
    # split on anything that's not a letter/number/underscore
    toks = re.split(r"[^\w]+", text, flags=re.UNICODE)
    toks = [t for t in toks if t and not t.isnumeric()]
    return toks

def _tokenizer_ngram(tokens: List[str], n: int) -> List[str]:
    if n == 1:
        return tokens
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def _custom_tokenizer_for_tfidf(s: str) -> List[str]:
    # Build 1-3 gram tokens; filter short bits and stops
    toks = _unicode_tokenizer(s)
    toks = [t for t in toks if len(t) >= 2 and t not in BN_EN_STOP]

    bigrams = _tokenizer_ngram(toks, 2)
    trigrams = _tokenizer_ngram(toks, 3)
    all_toks = toks + bigrams + trigrams
    # Drop 1-char syllables/noise
    all_toks = [t for t in all_toks if len(t.replace(" ", "")) >= 2]
    return all_toks

def _yake_fallback(text: str, top_k=8) -> List[str]:
    try:
        import yake
        kw = yake.KeywordExtractor(lan="bn", n=3, top=top_k)
        pairs = kw.extract_keywords(text[:20000])
        # pairs: list of (keyphrase, score) lower is better; return ordered phrases
        phrases = [p[0] for p in pairs if p and isinstance(p[0], str)]
        # de-dup while preserving order
        seen = set(); out = []
        for p in phrases:
            q = p.strip()
            if not q or q in seen: continue
            seen.add(q)
            out.append(q)
        return out[:top_k]
    except Exception:
        return []

def ctfidf_labels(texts_by_cluster: List[str], n_top=8) -> List[List[str]]:
    """
    Unicode-aware CTFIDF-ish: we just TF-IDF the merged text per cluster with a custom tokenizer.
    Robust to Bangla content; falls back to YAKE if TF-IDF cannot build a vocab.
    """
    if not texts_by_cluster:
        return [[]]

    try:
        vec = TfidfVectorizer(
            analyzer="word",
            tokenizer=_custom_tokenizer_for_tfidf,  # custom
            token_pattern=None,
            lowercase=False,   # tokenizer already lowercases
            min_df=1,
            max_df=1.0,
            max_features=60000,
        )
        X = vec.fit_transform(texts_by_cluster)
        X = normalize(X, axis=1)
        terms = np.array(vec.get_feature_names_out())
        labels = []
        for i in range(X.shape[0]):
            row = X.getrow(i).toarray()[0]
            if row.sum() == 0:
                # YAKE fallback on that cluster text
                labels.append(_yake_fallback(texts_by_cluster[i], n_top) or ["(no-terms)"])
                continue
            # Take top-n by weight; prefer phrases containing a space if possible
            top_idx = np.argsort(row)[::-1][: max(n_top*2, n_top)]
            cands = [terms[j] for j in top_idx]
            # Prefer multi-word first, then single
            multi = [c for c in cands if " " in c]
            single = [c for c in cands if " " not in c]
            ordered = multi + single
            # Remove duplicates / sub-pieces that are fully contained in better phrases (light dedupe)
            final = []
            for t in ordered:
                if any(t != u and t in u for u in final):
                    continue
                if t not in final:
                    final.append(t)
                if len(final) >= n_top:
                    break
            if not final:
                final = _yake_fallback(texts_by_cluster[i], n_top) or ["(no-terms)"]
            labels.append(final[:n_top])
        return labels

    except ValueError as e:
        # empty vocabulary; all stopwords or noise -> fallback to YAKE per cluster
        print(f"[warn] TF-IDF labeling failed: {e}")
        return [(_yake_fallback(t, n_top) or ["(no-terms)"]) for t in texts_by_cluster]
    except Exception as e:
        print(f"[warn] TF-IDF labeling unexpected error: {e}")
        return [(_yake_fallback(t, n_top) or ["(no-terms)"]) for t in texts_by_cluster]


# -----------------------------------------------------------------------------
# Centroids & DB writes
# -----------------------------------------------------------------------------
def compute_centroids(labels: np.ndarray, emb: np.ndarray) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    for cid in sorted(set(labels)):
        if cid == -1:
            continue
        idx = np.where(labels == cid)[0]
        if len(idx) == 0:
            continue
        vec = emb[idx].mean(axis=0)
        vnorm = vec / max(1e-9, np.linalg.norm(vec))
        out[cid] = vnorm.astype(np.float32)
    return out


def insert_categories(conn, cluster_ids: List[int], labels_per_cluster: List[List[str]],
                      sizes: Dict[int, int], algo: str, model_name: str) -> Dict[int, int]:
    """
    Insert rows into auto_category and return mapping {cluster_label -> category_id}
    """
    def _norm_label(s: str) -> str:
        return unicodedata.normalize("NFKC", (s or "")).strip().lower()

    id_map: Dict[int, int] = {}
    with conn.cursor() as cur:
        cur.execute("SELECT id, LOWER(label) FROM auto_category")
        existing = {lbl or "": cid for cid, lbl in cur.fetchall()}
        seen_norm = set(existing.keys())
        for i, cid in enumerate(cluster_ids):
            if cid == -1:
                continue
            terms = labels_per_cluster[i] if i < len(labels_per_cluster) else []
            # build display label: use the first meaningful phrase; keep top_terms for detail
            label_text = (terms[0].strip() if terms else f"Topic {cid}") or f"Topic {cid}"
            label_norm = _norm_label(label_text)
            if label_norm in seen_norm:
                reuse_id = existing.get(label_norm)
                if reuse_id:
                    id_map[cid] = reuse_id
                    # Update size/top_terms if we found more data for an existing label
                    try:
                        cur.execute(
                            """
                            UPDATE auto_category
                            SET size = GREATEST(COALESCE(size,0), %s),
                                top_terms = CASE WHEN array_length(top_terms,1) IS NULL OR array_length(top_terms,1)=0 THEN %s ELSE top_terms END
                            WHERE id = %s
                            """,
                            (int(sizes.get(cid, 0)), terms[:8], reuse_id),
                        )
                    except Exception:
                        pass
                continue
            # store up to 8 terms, trimmed
            terms = [t.strip() for t in terms if t and t.strip()][:8]
            size = int(sizes.get(cid, 0))
            cur.execute(
                """
                INSERT INTO auto_category(label, top_terms, size, model_name, algo, params_json)
                VALUES (%s, %s, %s, %s, %s, %s) RETURNING id
                """,
                (label_text, terms, size, model_name, algo, json.dumps({"created_by": "init"})),
            )
            new_id = cur.fetchone()[0]
            id_map[cid] = new_id
            existing[label_norm] = new_id
            seen_norm.add(label_norm)
    return id_map


def insert_memberships(conn, cat_id_map: Dict[int, int], labels: np.ndarray, article_ids: List[int], scores=None):
    rows = []
    used: Set[int] = set()
    for aid, lab in zip(article_ids, labels):
        if lab == -1:
            continue
        cid = cat_id_map.get(int(lab))
        if cid is None:
            continue
        sc = 1.0
        if isinstance(scores, dict):
            sc = float(scores.get(aid, 1.0))
        elif isinstance(scores, (int, float)):
            sc = float(scores)
        rows.append((int(cid), int(aid), sc, 1))
        used.add(int(aid))
    if not rows:
        return set()
    with conn.cursor() as cur:
        cur.executemany(
        """
        INSERT INTO auto_category_article(category_id, article_id, score, rank)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (category_id, article_id) DO NOTHING
        """,
        rows,
    )
    return used


def upsert_centroids(conn, cat_id_map: Dict[int, int], centroids: Dict[int, np.ndarray], model_name: str):
    rows = []
    for cid, cat_id in cat_id_map.items():
        vec = centroids.get(cid)
        if vec is None:
            continue
        rows.append((cat_id, model_name, int(vec.shape[0]), [float(x) for x in vec.tolist()]))
    if not rows:
        return
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO auto_category_centroid(category_id, model_name, dim, vector)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (category_id) DO UPDATE SET
              model_name = EXCLUDED.model_name,
              dim = EXCLUDED.dim,
              vector = EXCLUDED.vector,
              updated_at = now()
            """,
            rows,
        )


def save_article_embeddings(conn, article_ids: List[int], model_name: str, vectors: np.ndarray):
    """
    Cache per-article embedding vectors (optional).
    Expects article_embedding(article_id, model_name, dim, vector, PRIMARY KEY(article_id, model_name))
    """
    if vectors is None or not len(article_ids):
        return
    try:
        dim = int(vectors.shape[1])
    except Exception:
        dim = len(vectors[0])
    rows = [(int(aid), model_name, dim, [float(x) for x in vec]) for aid, vec in zip(article_ids, vectors)]
    try:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO article_embedding(article_id, model_name, dim, vector)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (article_id, model_name) DO UPDATE
                  SET dim = EXCLUDED.dim,
                      vector = EXCLUDED.vector
                """,
                rows,
            )
    except Exception as e:
        # If the cache table is missing, skip silently and continue the pipeline.
        print(f"[warn] article_embedding write skipped: {e}")


# ----------------------------------------------------------------------------- 
# LLM-assisted label refinement
# -----------------------------------------------------------------------------

def _llm_labels_for_cluster(key_terms: List[str], sample_text: str, max_labels: int = 5) -> List[str]:
    """
    Use an LLM to propose concise Bengali labels for a cluster.
    Falls back to empty list on any error.
    """
    if not ENABLE_LLM_LABELS or _oa_client is None:
        return []
    terms_text = ", ".join(key_terms[:8])
    snippet = (sample_text or "")[:800]
    prompt = (
        "আপনি একটি সংবাদ বিষয়ক ট্যাক্সোনমি সহকারী। নিম্নের শব্দ ও উদাহরণ টেক্সট দেখে "
        "খুব সংক্ষিপ্ত ৩-৫টি বাংলা লেবেল লিখুন। প্রতিটি লেবেল ২-৪ শব্দের হবে, "
        "কোনও নম্বর/বুলেট ছাড়া, প্রতি লাইনে একটি করে। সাধারণ শব্দ এড়িয়ে চলুন।\n\n"
        f"কীওয়ার্ড: {terms_text or '(কিছু নেই)'}\n"
        f"উদাহরণ টেক্সট: {snippet or '(টেক্সট নেই)'}\n\n"
        "আউটপুট:\n"
    )
    try:
        resp = _oa_client.chat.completions.create(
            model=LLM_MODEL or "gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=120,
            timeout=15,
        )
        text = (resp.choices[0].message.content or "").strip()
        labels = []
        for line in text.splitlines():
            t = line.strip(" -•\t\r\n")
            if not t or len(t) < 2:
                continue
            labels.append(t)
            if len(labels) >= max_labels:
                break
        return labels
    except Exception as e:
        print(f"[warn] LLM label generation failed: {e}")
        return []


def refine_labels_with_llm(texts_by_cluster: List[str], labels_by_cluster: List[List[str]]) -> List[List[str]]:
    """
    Post-process cluster labels with an LLM. Keeps originals if LLM is unavailable or returns nothing.
    """
    if not ENABLE_LLM_LABELS or _oa_client is None:
        return labels_by_cluster
    refined: List[List[str]] = []
    for idx, base_labels in enumerate(labels_by_cluster):
        text = texts_by_cluster[idx] if idx < len(texts_by_cluster) else ""
        llm_labels = _llm_labels_for_cluster(base_labels or [], text)
        if llm_labels:
            refined.append(llm_labels)
        else:
            refined.append(base_labels)
    return refined


# -----------------------------------------------------------------------------
# Assignment to existing categories
# -----------------------------------------------------------------------------
def assign_to_existing(conn, model: "SentenceTransformer", df: pd.DataFrame, threshold: float = 0.38):
    if df.empty:
        return [], []
    texts = build_texts(df)
    emb = embed_texts(model, texts)
    # Save cache
    save_article_embeddings(conn, df["id"].tolist(), getattr(model, "model_card", DEFAULT_MODEL), emb)

    # Load centroids
    with conn.cursor() as cur:
        cur.execute("SELECT category_id, model_name, dim, vector FROM auto_category_centroid")
        rows = cur.fetchall()
    if not rows:
        return [], df["id"].tolist()

    # Vectorize similarity to avoid O(n*m) Python loops
    cid_list = []
    centroid_vecs = []
    for cid, _mn, _dim, vec in rows:
        arr = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(arr) or 1e-9
        cid_list.append(cid)
        centroid_vecs.append(arr / norm)

    centroid_mat = np.stack(centroid_vecs) if centroid_vecs else np.zeros((0, emb.shape[1]), dtype=np.float32)
    if centroid_mat.size == 0:
        return [], df["id"].tolist()

    # emb is already normalized in embed_texts, so plain dot is cosine similarity
    sims = np.dot(emb, centroid_mat.T)  # shape: (n_articles, n_centroids)
    best_idx = np.argmax(sims, axis=1)
    best_scores = sims[np.arange(sims.shape[0]), best_idx]

    assignments = []
    unassigned = []
    article_ids = df["id"].tolist()
    for row_idx, aid in enumerate(article_ids):
        score = float(best_scores[row_idx])
        if math.isfinite(score) and score >= threshold:
            assignments.append((aid, cid_list[int(best_idx[row_idx])], score))
        else:
            unassigned.append(aid)
    return assignments, unassigned


def insert_assignments(conn, assignments: List[Tuple[int, int, float]]):
    """
    assignments: list of (article_id, category_id, score)
    """
    if not assignments:
        return set()
    rows = [(cid, aid, float(score), 1) for (aid, cid, score) in assignments]
    used = {int(aid) for (aid, _cid, _s) in assignments}
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO auto_category_article(category_id, article_id, score, rank)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (category_id, article_id) DO NOTHING
            """,
            rows,
        )
    return used


# -----------------------------------------------------------------------------
# INIT / UPDATE
# -----------------------------------------------------------------------------
def init_mode(conn, model: Optional["SentenceTransformer"], limit: Optional[int], min_cluster_size: int):
    df = fetch_articles(conn, limit=limit)
    if df.empty:
        print("[warn] no articles")
        return
    texts = build_texts(df)

    if model is not None:
        # --- Embedding path ---
        print("[info] building embeddings…")
        emb = embed_texts(model, texts)
        save_article_embeddings(conn, df["id"].tolist(), getattr(model, "model_card", DEFAULT_MODEL), emb)

        print("[info] clustering (UMAP+HDBSCAN → fallback KMeans)…")
        labels = cluster_embeddings(emb, min_cluster_size=min_cluster_size)
        cluster_ids = sorted(set(labels))
        kept = [cid for cid in cluster_ids if cid != -1]

        if not kept:
            print("[warn] HDBSCAN produced only noise; retrying with KMeans fallback")
            labels = cluster_embeddings(emb, min_cluster_size=max(5, min_cluster_size // 2))
            cluster_ids = sorted(set(labels))
            kept = [cid for cid in cluster_ids if cid != -1]

        sizes: Dict[int, int] = {cid: int((labels == cid).sum()) for cid in kept}
        texts_by_cluster = []
        for cid in kept:
            idx = np.where(labels == cid)[0]
            texts_by_cluster.append(" ".join([texts[i] for i in idx]) or " ")

        print("[info] extracting keyphrases for labels…")
        top_terms = ctfidf_labels(texts_by_cluster, n_top=8)
        top_terms = refine_labels_with_llm(texts_by_cluster, top_terms)

        algo_str = f"embeddings+{'umap+hdbscan' if hdbscan else 'kmeans'}"
        cat_id_map = insert_categories(conn, kept, top_terms, sizes, algo=algo_str, model_name=DEFAULT_MODEL)
        used_ids = insert_memberships(conn, cat_id_map, labels, df["id"].tolist())
        mark_auto_used(conn, used_ids)

        print("[info] computing centroids…")
        cents = compute_centroids(labels, emb)
        upsert_centroids(conn, cat_id_map, cents, model_name=DEFAULT_MODEL)
        print("[ok] init complete with embeddings")
        return

    # --- TF-IDF fallback path ---
    print("[info] embeddings disabled → TF-IDF + KMeans")
    vec = TfidfVectorizer(
        analyzer="word",
        tokenizer=_custom_tokenizer_for_tfidf,
        token_pattern=None,
        lowercase=False,
        min_df=2,
        max_df=0.6,
        max_features=75000,
    )
    X = vec.fit_transform(texts)
    n = X.shape[0]
    k = max(6, min(30, max(2, n // max(1, min_cluster_size))))
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = km.fit_predict(X)
    kept = sorted(set(labels))
    sizes = {cid: int((labels == cid).sum()) for cid in kept}
    texts_by_cluster = []
    for cid in kept:
        idx = np.where(labels == cid)[0]
        texts_by_cluster.append(" ".join([texts[i] for i in idx]) or " ")
    top_terms = ctfidf_labels(texts_by_cluster, n_top=8)
    top_terms = refine_labels_with_llm(texts_by_cluster, top_terms)
    cat_id_map = insert_categories(conn, kept, top_terms, sizes, algo=f"tfidf+kmeans(k={km.n_clusters})", model_name="tfidf")
    used_ids = insert_memberships(conn, cat_id_map, labels, df["id"].tolist())
    mark_auto_used(conn, used_ids)
    print("[ok] init complete (tfidf+kmeans). Note: centroids not available without embeddings.")


def cluster_unassigned_and_add(conn, model: Optional["SentenceTransformer"], df_un: pd.DataFrame, min_cluster_size: int):
    if df_un.empty:
        print("[info] nothing unassigned to cluster")
        return
    texts = build_texts(df_un)

    if model is not None:
        emb = embed_texts(model, texts)
        save_article_embeddings(conn, df_un["id"].tolist(), getattr(model, "model_card", DEFAULT_MODEL), emb)
        labels = cluster_embeddings(emb, min_cluster_size=min_cluster_size)
        kept = [cid for cid in sorted(set(labels)) if cid != -1]
        if not kept:
            print("[info] unassigned did not form stable clusters")
            return
        sizes = {cid: int((labels == cid).sum()) for cid in kept}
        texts_by_cluster = []
        for cid in kept:
            idx = np.where(labels == cid)[0]
            texts_by_cluster.append(" ".join([texts[i] for i in idx]) or " ")
        top_terms = ctfidf_labels(texts_by_cluster, n_top=8)
        top_terms = refine_labels_with_llm(texts_by_cluster, top_terms)
        algo_str = f"embeddings+{'umap+hdbscan' if hdbscan else 'kmeans'}"
        cat_id_map = insert_categories(conn, kept, top_terms, sizes, algo=algo_str, model_name=DEFAULT_MODEL)
        used_ids = insert_memberships(conn, cat_id_map, labels, df_un["id"].tolist())
        mark_auto_used(conn, used_ids)
        cents = compute_centroids(labels, emb)
        upsert_centroids(conn, cat_id_map, cents, model_name=DEFAULT_MODEL)
        print(f"[ok] created {len(kept)} new categories from unassigned")
        return

    # TF-IDF fallback
    vec = TfidfVectorizer(
        analyzer="word",
        tokenizer=_custom_tokenizer_for_tfidf,
        token_pattern=None,
        lowercase=False,
        min_df=2,
        max_df=0.6,
        max_features=50000,
    )
    X = vec.fit_transform(texts)
    n = X.shape[0]
    k = max(2, min(10, n // max(1, min_cluster_size)))
    if k < 2:
        print("[info] not enough unassigned to form new clusters")
        return
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = km.fit_predict(X)
    kept = sorted(set(labels))
    sizes = {cid: int((labels == cid).sum()) for cid in kept}
    texts_by_cluster = []
    for cid in kept:
        idx = np.where(labels == cid)[0]
        texts_by_cluster.append(" ".join([texts[i] for i in idx]) or " ")
    top_terms = ctfidf_labels(texts_by_cluster, n_top=8)
    cat_id_map = insert_categories(conn, kept, top_terms, sizes, algo=f"tfidf+kmeans(k={km.n_clusters})", model_name="tfidf")
    used_ids = insert_memberships(conn, cat_id_map, labels, df_un["id"].tolist())
    mark_auto_used(conn, used_ids)
    print(f"[ok] created {len(kept)} new tfidf categories from unassigned")


def update_mode(conn, model: Optional["SentenceTransformer"], threshold: float, min_cluster_size: int, limit: Optional[int], also_cluster_unassigned: bool):
    df = fetch_unassigned(conn, limit=limit, only_unused=True)
    if df.empty:
        print("[info] no unassigned (new) articles")
        return

    if model is None:
        # simple lexical fallback using existing category top_terms
        with conn.cursor() as cur:
            cur.execute("SELECT id, top_terms FROM auto_category")
            cats = cur.fetchall()
        termsets = [(cid, set(sum((t.split() for t in (terms or [])), []))) for cid, terms in cats]
        assignments, unassigned_ids = [], []
        for _, r in df.iterrows():
            text = (r["title"] + " " + r["content"]).lower()
            best_cid, best_hits = None, 0
            for cid, terms in termsets:
                hits = sum(text.count(term.lower()) for term in terms if len(term) > 2)
                if hits > best_hits:
                    best_hits, best_cid = hits, cid
            if best_cid is not None and best_hits >= 2:
                assignments.append((r["id"], best_cid, float(best_hits)))
            else:
                unassigned_ids.append(r["id"])
        used_ids = insert_assignments(conn, assignments)
        mark_auto_used(conn, used_ids)
        print(f"[ok] assigned {len(assignments)} by term-match; {len(unassigned_ids)} remain")
        if also_cluster_unassigned and unassigned_ids:
            df_un = df[df["id"].isin(unassigned_ids)].reset_index(drop=True)
            cluster_unassigned_and_add(conn, model, df_un, min_cluster_size=min_cluster_size)
        return

    # Embedding path
    assignments, unassigned_ids = assign_to_existing(conn, model, df, threshold=threshold)
    used_ids = insert_assignments(conn, assignments)
    mark_auto_used(conn, used_ids)
    print(f"[ok] assigned {len(assignments)} to existing categories; {len(unassigned_ids)} unassigned")
    if also_cluster_unassigned and unassigned_ids:
        df_un = df[df["id"].isin(unassigned_ids)].reset_index(drop=True)
        cluster_unassigned_and_add(conn, model, df_un, min_cluster_size=min_cluster_size)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default=DEFAULT_DSN)
    ap.add_argument("--mode", choices=["init", "update"], required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-cluster-size", type=int, default=20)
    ap.add_argument("--threshold", type=float, default=0.38, help="cosine sim threshold")
    ap.add_argument("--cluster-unassigned", action="store_true", help="cluster unassigned into new categories")
    ap.add_argument("--use-embeddings", dest="use_embeddings", action="store_true")
    ap.add_argument("--emb-model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    use_emb = getattr(args, "use_embeddings", False)
    model = load_sentence_model(args.emb_model) if use_emb else None

    # Silence sklearn FutureWarnings in subprocess logs if needed
    os.environ.setdefault("PYTHONWARNINGS", "ignore::FutureWarning:sklearn")

    with pg_conn(args.dsn) as conn:
        ensure_auto_used_column(conn)
        if args.mode == "init":
            init_mode(conn, model, limit=args.limit, min_cluster_size=args.min_cluster_size)
        else:
            update_mode(
                conn,
                model,
                threshold=args.threshold,
                min_cluster_size=args.min_cluster_size,
                limit=args.limit,
                also_cluster_unassigned=args.cluster_unassigned,
            )


if __name__ == "__main__":
    main()

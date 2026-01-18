CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS codebook_category (
  category_id SERIAL PRIMARY KEY,
  phase TEXT,
  category TEXT NOT NULL,
  definition TEXT,
  subcategories TEXT,
  examples TEXT,
  notes TEXT
);

CREATE TABLE IF NOT EXISTS codebook_keyword (
  keyword_id SERIAL PRIMARY KEY,
  category_id INT REFERENCES codebook_category(category_id) ON DELETE CASCADE,
  term TEXT NOT NULL,
  weight NUMERIC NOT NULL DEFAULT 1.0,
  lang TEXT DEFAULT 'bn',
  CONSTRAINT uq_codebook_keyword UNIQUE (category_id, term, lang)
);

CREATE TABLE IF NOT EXISTS portal (
  portal_id SERIAL PRIMARY KEY,
  name TEXT NOT NULL,
  rss_url TEXT,
  base_url TEXT,
  language TEXT DEFAULT 'bn',
  is_active BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS article (
  article_id BIGSERIAL PRIMARY KEY,
  portal_id INT REFERENCES portal(portal_id),
  title TEXT,
  url TEXT UNIQUE,
  published_at TIMESTAMPTZ,
  content_raw TEXT,
  content_clean TEXT,
  lang TEXT DEFAULT 'bn',
  url_hash TEXT UNIQUE,
  status TEXT DEFAULT 'new',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS classification_run (
  run_id BIGSERIAL PRIMARY KEY,
  model TEXT,
  started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  finished_at TIMESTAMPTZ
);

DO $$ BEGIN
  -- CREATE TYPE label_source AS ENUM ('rule','llm','human');
  CREATE TYPE label_source AS ENUM ('rule','ml','llm','human');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

CREATE TABLE IF NOT EXISTS article_label (
  article_id BIGINT REFERENCES article(article_id) ON DELETE CASCADE,
  category_id INT REFERENCES codebook_category(category_id) ON DELETE CASCADE,
  score NUMERIC,
  source label_source NOT NULL,
  run_id BIGINT REFERENCES classification_run(run_id),
  is_primary BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (article_id, category_id, source)
);

CREATE TABLE IF NOT EXISTS correction (
  correction_id BIGSERIAL PRIMARY KEY,
  article_id BIGINT REFERENCES article(article_id) ON DELETE CASCADE,
  category_id INT REFERENCES codebook_category(category_id),
  action TEXT CHECK (action IN ('add','remove')),
  notes TEXT,
  created_by TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Keep portal names unique to support upserts
CREATE UNIQUE INDEX IF NOT EXISTS idx_portal_name ON portal(name);

CREATE INDEX IF NOT EXISTS idx_article_ft ON article USING gin (to_tsvector('simple', coalesce(title,'') || ' ' || coalesce(content_clean,'')));
CREATE INDEX IF NOT EXISTS idx_keyword_term ON codebook_keyword(term);

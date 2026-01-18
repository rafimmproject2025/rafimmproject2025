-- Unified schema for News Agent (merged from init scripts)
-- Idempotent: uses IF NOT EXISTS / CREATE OR REPLACE where feasible.

------------------------------------------------------------------
-- Extensions / Types
------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS pg_trgm;

DO $$ BEGIN
  CREATE TYPE label_source AS ENUM ('rule','ml','llm','human');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'app_role') THEN
    CREATE TYPE app_role AS ENUM ('admin','user');
  END IF;
END $$;

------------------------------------------------------------------
-- Core tables
------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS codebook_category (
  category_id SERIAL PRIMARY KEY,
  phase TEXT,
  category TEXT NOT NULL,
  definition TEXT,
  subcategories TEXT,
  examples TEXT,
  notes TEXT,
  is_auto BOOLEAN NOT NULL DEFAULT FALSE,
  review_needed BOOLEAN NOT NULL DEFAULT FALSE,
  parent_category_id INT NULL REFERENCES codebook_category(category_id),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  created_by TEXT
);

CREATE TABLE IF NOT EXISTS codebook_keyword (
  keyword_id SERIAL PRIMARY KEY,
  category_id INT REFERENCES codebook_category(category_id) ON DELETE CASCADE,
  term TEXT NOT NULL,
  weight NUMERIC NOT NULL DEFAULT 1.0,
  lang TEXT DEFAULT 'bn',
  CONSTRAINT uq_codebook_keyword UNIQUE (category_id, term, lang)
);

CREATE TABLE IF NOT EXISTS candidate_ref (
  candidate_id SERIAL PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  name_bn TEXT,
  party TEXT,
  seat TEXT,
  aliases TEXT[] DEFAULT ARRAY[]::TEXT[],
  is_active BOOLEAN NOT NULL DEFAULT TRUE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
ALTER TABLE candidate_ref
  ADD COLUMN IF NOT EXISTS name_bn TEXT,
  ADD COLUMN IF NOT EXISTS seat TEXT;

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
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  gen_cats_at TIMESTAMPTZ,
  gen_cats_mode label_source,
  party TEXT,
  candidate TEXT,
  region TEXT,
  auto_used BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS classification_run (
  run_id BIGSERIAL PRIMARY KEY,
  model TEXT,
  started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  finished_at TIMESTAMPTZ
);

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

------------------------------------------------------------------
-- Generated labels scratch + derived tables
------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS codebook_category_generate (
  gen_id      BIGSERIAL PRIMARY KEY,
  article_id  BIGINT NOT NULL REFERENCES article(article_id) ON DELETE CASCADE,
  category_id INT    NOT NULL REFERENCES codebook_category(category_id) ON DELETE CASCADE,
  mode        label_source NOT NULL,
  score       NUMERIC,
  is_primary  BOOLEAN DEFAULT FALSE,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (article_id, category_id, mode)
);

CREATE OR REPLACE VIEW v_category_generate AS
SELECT
  g.gen_id, g.article_id, a.title, a.url, a.published_at, a.lang,
  c.category, g.score, g.is_primary, g.mode, g.created_at
FROM codebook_category_generate g
JOIN codebook_category c ON c.category_id = g.category_id
JOIN article a            ON a.article_id   = g.article_id;

-- Generated labels table (parallel to article_label)
CREATE TABLE IF NOT EXISTS article_category_label (
  article_id BIGINT NOT NULL REFERENCES article(article_id) ON DELETE CASCADE,
  category_id INT NOT NULL REFERENCES codebook_category(category_id) ON DELETE CASCADE,
  score NUMERIC,
  mode  label_source NOT NULL,
  is_primary BOOLEAN DEFAULT FALSE,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (article_id, category_id, mode)
);

------------------------------------------------------------------
-- Auto-category pipeline schema
------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS auto_category (
  id          BIGSERIAL PRIMARY KEY,
  label       TEXT NOT NULL,
  top_terms   TEXT[] DEFAULT ARRAY[]::TEXT[],
  size        INT DEFAULT 0,
  model_name  TEXT,
  algo        TEXT,
  params_json JSONB DEFAULT '{}'::jsonb,
  created_at  TIMESTAMPTZ DEFAULT now()
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
  vector DOUBLE PRECISION[] NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now(),
  PRIMARY KEY (article_id, model_name)
);

------------------------------------------------------------------
-- Auth
------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS app_user (
  user_id SERIAL PRIMARY KEY,
  username TEXT NOT NULL UNIQUE,
  password_hash TEXT NOT NULL,
  role app_role NOT NULL DEFAULT 'user',
  display_name TEXT,
  is_active BOOLEAN NOT NULL DEFAULT TRUE,
  last_login TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_app_user_role ON app_user(role);
CREATE INDEX IF NOT EXISTS idx_app_user_active ON app_user(is_active);
CREATE UNIQUE INDEX IF NOT EXISTS uq_app_user_lower ON app_user (LOWER(username));

------------------------------------------------------------------
-- Backfill newer columns for existing deployments
------------------------------------------------------------------
ALTER TABLE article
  ADD COLUMN IF NOT EXISTS party TEXT,
  ADD COLUMN IF NOT EXISTS candidate TEXT,
  ADD COLUMN IF NOT EXISTS region TEXT,
  ADD COLUMN IF NOT EXISTS auto_used BOOLEAN NOT NULL DEFAULT FALSE;

------------------------------------------------------------------
-- Indexes
------------------------------------------------------------------
CREATE UNIQUE INDEX IF NOT EXISTS idx_portal_name ON portal(name);
CREATE INDEX IF NOT EXISTS idx_article_ft ON article USING gin (to_tsvector('simple', coalesce(title,'') || ' ' || coalesce(content_clean,'')));
CREATE INDEX IF NOT EXISTS idx_keyword_term ON codebook_keyword(term);
CREATE INDEX IF NOT EXISTS idx_candidate_ref_name ON candidate_ref(LOWER(name));
CREATE INDEX IF NOT EXISTS idx_candidate_ref_name_bn ON candidate_ref(LOWER(name_bn));
CREATE INDEX IF NOT EXISTS idx_candidate_ref_aliases ON candidate_ref USING gin (aliases);
CREATE INDEX IF NOT EXISTS idx_candidate_ref_seat ON candidate_ref(LOWER(seat));
CREATE INDEX IF NOT EXISTS idx_article_status ON article(status);
CREATE INDEX IF NOT EXISTS idx_article_auto_used ON article(auto_used);
CREATE INDEX IF NOT EXISTS idx_article_gen ON article(gen_cats_at);
CREATE INDEX IF NOT EXISTS idx_al_article ON article_label(article_id);
CREATE INDEX IF NOT EXISTS idx_cg_article ON codebook_category_generate(article_id);
CREATE INDEX IF NOT EXISTS idx_cg_mode    ON codebook_category_generate(mode);
CREATE INDEX IF NOT EXISTS idx_acl_article ON article_category_label(article_id);
CREATE INDEX IF NOT EXISTS idx_acl_mode    ON article_category_label(mode);
CREATE INDEX IF NOT EXISTS idx_auto_category_size ON auto_category(size DESC);
CREATE INDEX IF NOT EXISTS idx_auto_cat_article_article ON auto_category_article(article_id);
CREATE INDEX IF NOT EXISTS idx_auto_cat_article_category ON auto_category_article(category_id);
CREATE INDEX IF NOT EXISTS idx_article_party ON article(LOWER(party));
CREATE INDEX IF NOT EXISTS idx_article_candidate ON article(LOWER(candidate));
CREATE INDEX IF NOT EXISTS idx_article_region ON article(LOWER(region));
CREATE INDEX IF NOT EXISTS idx_article_embedding_article ON article_embedding(article_id);

------------------------------------------------------------------
-- Cleanup existing auto-generated definitions (idempotent)
------------------------------------------------------------------
UPDATE codebook_category
SET definition = NULLIF(
    REGEXP_REPLACE(definition, '^Auto-generated from embeddings \\([^)]*\\) algo=[^;]*;\\s*top_terms:\\s*', '', 'i'),
    ''
)
WHERE definition ILIKE 'Auto-generated from embeddings%';

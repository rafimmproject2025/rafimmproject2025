-- ENUM used by backend (if you don’t already have it)
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'label_source') THEN
    CREATE TYPE label_source AS ENUM ('rule','ml','llm','human');
  END IF;
END $$;

-- Articles must exist (you likely already have this)
-- Minimal example (do NOT apply if your schema already exists):
-- CREATE TABLE IF NOT EXISTS article (
--   article_id BIGSERIAL PRIMARY KEY,
--   title TEXT,
--   url TEXT,
--   content_clean TEXT,
--   content_raw TEXT,
--   lang TEXT,
--   status TEXT DEFAULT 'queued',
--   published_at TIMESTAMPTZ,
--   created_at TIMESTAMPTZ DEFAULT now(),
--   gen_cats_at TIMESTAMPTZ,
--   gen_cats_mode label_source
-- );

-- Auto categories
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

-- Optional: indexes for speed
CREATE INDEX IF NOT EXISTS idx_auto_category_size ON auto_category(size DESC);
CREATE INDEX IF NOT EXISTS idx_auto_cat_article_article ON auto_category_article(article_id);
CREATE INDEX IF NOT EXISTS idx_article_status ON article(status);

-- auto categories core
CREATE TABLE IF NOT EXISTS auto_category (
  id          SERIAL PRIMARY KEY,
  label       TEXT NOT NULL,
  top_terms   TEXT[] NOT NULL,
  size        INTEGER NOT NULL,
  model_name  TEXT,
  algo        TEXT,
  params_json JSONB,
  created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS auto_category_article (
  category_id INTEGER NOT NULL REFERENCES auto_category(id) ON DELETE CASCADE,
  article_id  INTEGER NOT NULL REFERENCES article(article_id) ON DELETE CASCADE,
  score       REAL,
  rank        INTEGER,
  PRIMARY KEY (category_id, article_id)
);

ALTER TABLE auto_category_article
  DROP CONSTRAINT IF EXISTS auto_category_article_article_id_fkey,
  ADD CONSTRAINT auto_category_article_article_id_fkey
    FOREIGN KEY (article_id) REFERENCES article(article_id) ON DELETE CASCADE;

-- optional: centroids for cosine assignment + field-wise cosine in breakdown
CREATE TABLE IF NOT EXISTS auto_category_centroid (
  category_id INTEGER PRIMARY KEY REFERENCES auto_category(id) ON DELETE CASCADE,
  model_name  TEXT NOT NULL,
  dim         INTEGER NOT NULL,
  vector      DOUBLE PRECISION[] NOT NULL,
  updated_at  TIMESTAMPTZ DEFAULT now()
);


CREATE TABLE IF NOT EXISTS article_embedding (
  article_id INTEGER NOT NULL REFERENCES article(article_id) ON DELETE CASCADE,
  model_name TEXT NOT NULL,
  dim       INTEGER NOT NULL,
  vector    DOUBLE PRECISION[] NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now(),
  PRIMARY KEY (article_id, model_name)
);

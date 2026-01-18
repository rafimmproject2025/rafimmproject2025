-- 1) Track generation on article rows
ALTER TABLE article
  ADD COLUMN IF NOT EXISTS gen_cats_at   TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS gen_cats_mode label_source;

-- 2) Scratch table (keep if you already created it)
CREATE TABLE IF NOT EXISTS codebook_category_generate (
  gen_id      BIGSERIAL PRIMARY KEY,
  article_id  BIGINT NOT NULL REFERENCES article(article_id) ON DELETE CASCADE,
  category_id INT    NOT NULL REFERENCES codebook_category(category_id) ON DELETE CASCADE,
  mode        label_source NOT NULL,            -- rule | ml | llm | human
  score       NUMERIC,
  is_primary  BOOLEAN DEFAULT FALSE,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (article_id, category_id, mode)
);

CREATE INDEX IF NOT EXISTS idx_cg_article ON codebook_category_generate(article_id);
CREATE INDEX IF NOT EXISTS idx_cg_mode    ON codebook_category_generate(mode);

CREATE OR REPLACE VIEW v_category_generate AS
SELECT
  g.gen_id, g.article_id, a.title, a.url, a.published_at, a.lang,
  c.category, g.score, g.is_primary, g.mode, g.created_at
FROM codebook_category_generate g
JOIN codebook_category c ON c.category_id = g.category_id
JOIN article a            ON a.article_id   = g.article_id;

-- 3) Generated labels table (parallel to article_label)
CREATE TABLE IF NOT EXISTS article_category_label (
  article_id BIGINT NOT NULL REFERENCES article(article_id) ON DELETE CASCADE,
  category_id INT NOT NULL REFERENCES codebook_category(category_id) ON DELETE CASCADE,
  score NUMERIC,
  mode  label_source NOT NULL,                  -- rule | ml | llm | human
  is_primary BOOLEAN DEFAULT FALSE,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (article_id, category_id, mode)
);

CREATE INDEX IF NOT EXISTS idx_acl_article ON article_category_label(article_id);
CREATE INDEX IF NOT EXISTS idx_acl_mode    ON article_category_label(mode);

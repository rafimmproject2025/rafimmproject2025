-- Scratch space to store generated category candidates by mode
-- Reuses the existing enum: label_source ('rule','ml','llm','human')

CREATE TABLE IF NOT EXISTS codebook_category_generate (
  gen_id      BIGSERIAL PRIMARY KEY,
  article_id  BIGINT NOT NULL REFERENCES article(article_id) ON DELETE CASCADE,
  category_id INT    NOT NULL REFERENCES codebook_category(category_id) ON DELETE CASCADE,
  mode        label_source NOT NULL,            -- which classifier produced it
  score       NUMERIC,                          -- 0..1 or model score
  is_primary  BOOLEAN DEFAULT FALSE,            -- winner per mode if you want
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (article_id, category_id, mode)        -- avoid duplicates per mode
);

-- Handy view to read it with category names
CREATE OR REPLACE VIEW v_category_generate AS
SELECT
  g.gen_id, g.article_id, a.title, a.url, a.published_at, a.lang,
  c.category, g.score, g.is_primary, g.mode, g.created_at
FROM codebook_category_generate g
JOIN codebook_category c ON c.category_id = g.category_id
JOIN article a            ON a.article_id   = g.article_id;

-- Indexes
CREATE INDEX IF NOT EXISTS idx_cg_article ON codebook_category_generate(article_id);
CREATE INDEX IF NOT EXISTS idx_cg_mode    ON codebook_category_generate(mode);

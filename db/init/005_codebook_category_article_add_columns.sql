-- codebook_category
ALTER TABLE codebook_category
  ADD COLUMN IF NOT EXISTS is_auto BOOLEAN NOT NULL DEFAULT FALSE,
  ADD COLUMN IF NOT EXISTS review_needed BOOLEAN NOT NULL DEFAULT FALSE,
  ADD COLUMN IF NOT EXISTS parent_category_id INT NULL REFERENCES codebook_category(category_id),
  ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  ADD COLUMN IF NOT EXISTS created_by TEXT;

-- CREATE UNIQUE INDEX IF NOT EXISTS uq_codebook_category_lower ON codebook_category (LOWER(category));

-- article (already in your code, but ensure present)
ALTER TABLE article
  ADD COLUMN IF NOT EXISTS gen_cats_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS gen_cats_mode label_source;

ALTER TABLE codebook_keyword
  ADD CONSTRAINT IF NOT EXISTS uq_codebook_keyword UNIQUE (category_id, term, lang);

-- indexes that help the loop
CREATE INDEX IF NOT EXISTS idx_article_status ON article(status);
CREATE INDEX IF NOT EXISTS idx_article_gen ON article(gen_cats_at);
CREATE INDEX IF NOT EXISTS idx_al_article ON article_label(article_id);






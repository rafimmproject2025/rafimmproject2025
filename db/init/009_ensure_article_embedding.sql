-- Ensure article_embedding exists (idempotent)
CREATE TABLE IF NOT EXISTS article_embedding (
  article_id BIGINT NOT NULL REFERENCES article(article_id) ON DELETE CASCADE,
  model_name TEXT NOT NULL,
  dim INT,
  vector DOUBLE PRECISION[] NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now(),
  PRIMARY KEY (article_id, model_name)
);

CREATE INDEX IF NOT EXISTS idx_article_embedding_article ON article_embedding(article_id);

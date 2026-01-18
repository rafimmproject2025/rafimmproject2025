-- Additional dimensions for dashboard filters
ALTER TABLE article
  ADD COLUMN IF NOT EXISTS party TEXT,
  ADD COLUMN IF NOT EXISTS candidate TEXT,
  ADD COLUMN IF NOT EXISTS region TEXT,
  ADD COLUMN IF NOT EXISTS auto_used BOOLEAN NOT NULL DEFAULT FALSE;

CREATE INDEX IF NOT EXISTS idx_article_party ON article(LOWER(party));
CREATE INDEX IF NOT EXISTS idx_article_candidate ON article(LOWER(candidate));
CREATE INDEX IF NOT EXISTS idx_article_region ON article(LOWER(region));
CREATE INDEX IF NOT EXISTS idx_auto_cat_article_category ON auto_category_article(category_id);
CREATE INDEX IF NOT EXISTS idx_article_auto_used ON article(auto_used);

-- Candidate reference list
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
CREATE INDEX IF NOT EXISTS idx_candidate_ref_name ON candidate_ref(LOWER(name));
CREATE INDEX IF NOT EXISTS idx_candidate_ref_name_bn ON candidate_ref(LOWER(name_bn));
CREATE INDEX IF NOT EXISTS idx_candidate_ref_aliases ON candidate_ref USING gin (aliases);
CREATE INDEX IF NOT EXISTS idx_candidate_ref_seat ON candidate_ref(LOWER(seat));

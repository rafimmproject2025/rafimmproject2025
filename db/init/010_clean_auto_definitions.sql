-- Strip auto-generation boilerplate from codebook_category.definition
UPDATE codebook_category
SET definition = NULLIF(
    REGEXP_REPLACE(definition, '^Auto-generated from embeddings \\([^)]*\\) algo=[^;]*;\\s*top_terms:\\s*', '', 'i'),
    ''
)
WHERE definition ILIKE 'Auto-generated from embeddings%';

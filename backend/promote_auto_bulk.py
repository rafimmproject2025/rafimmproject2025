#!/usr/bin/env python3
"""
Promote all auto_category rows into codebook_category, seed keywords, and apply labels to articles.
Intended for one-off bulk promotion after regenerating auto categories.
"""

import os
import psycopg

DSN = os.getenv("DB_DSN", "postgresql://postgres:postgres@db:5432/candidate_news")


def main():
    with psycopg.connect(DSN, autocommit=True) as con:
        cur = con.cursor()
        cur.execute(
            "INSERT INTO classification_run(model) VALUES (%s) RETURNING run_id",
            ("auto_promote:bulk",),
        )
        run_id = cur.fetchone()[0]

        cur.execute("SELECT id, label, top_terms FROM auto_category ORDER BY id")
        cats = cur.fetchall()

        promoted = 0
        labels_written = 0

        for cid, raw_label, terms in cats:
            label_parts = [p.strip() for p in (raw_label or "").split("|") if p.strip()]
            label = label_parts[0] if label_parts else (raw_label or f"Auto {cid}")
            terms_list = [t.strip() for t in (terms or []) if isinstance(t, str) and t.strip()]
            if not terms_list and label_parts:
                terms_list = label_parts
            definition = ", ".join(terms_list[:6]) if terms_list else None

            # find or create codebook_category
            cur.execute(
                "SELECT category_id, definition FROM codebook_category WHERE LOWER(category)=LOWER(%s) LIMIT 1",
                (label,),
            )
            row = cur.fetchone()
            if row:
                code_cid = row[0]
                if not row[1] and definition:
                    cur.execute(
                        "UPDATE codebook_category SET definition=%s WHERE category_id=%s",
                        (definition, code_cid),
                    )
            else:
                cur.execute(
                    """
                    INSERT INTO codebook_category(category, definition, is_auto, review_needed, created_by, phase)
                    VALUES (%s, %s, TRUE, TRUE, %s, %s)
                    RETURNING category_id
                    """,
                    (label, definition, "system:auto_promote", "auto"),
                )
                code_cid = cur.fetchone()[0]

            # keywords
            for t in terms_list:
                try:
                    cur.execute(
                        """
                        INSERT INTO codebook_keyword(category_id, term, weight, lang)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (category_id, term, lang) DO NOTHING
                        """,
                        (code_cid, t, 1.0, "bn"),
                    )
                except Exception:
                    pass

            # labels from auto_category_article
            cur.execute(
                "SELECT article_id, COALESCE(score,1.0), COALESCE(rank,1) FROM auto_category_article WHERE category_id=%s",
                (cid,),
            )
            links = cur.fetchall()
            for aid, score, rank in links:
                try:
                    cur.execute(
                        """
                        INSERT INTO article_label(article_id, category_id, score, source, run_id, is_primary)
                        VALUES (%s, %s, %s, %s::label_source, %s, %s)
                        ON CONFLICT (article_id, category_id, source) DO UPDATE
                          SET score=EXCLUDED.score, run_id=EXCLUDED.run_id, is_primary=EXCLUDED.is_primary
                        """,
                        (aid, code_cid, float(score or 1.0), "ml", run_id, bool(rank == 1)),
                    )
                    labels_written += 1
                except Exception:
                    pass

            promoted += 1

        cur.execute(
            "UPDATE article SET status='classified' WHERE article_id IN (SELECT DISTINCT article_id FROM article_label)"
        )

    print(f"Promoted {promoted} auto categories; labels_written={labels_written}; run_id={run_id}")


if __name__ == "__main__":
    main()

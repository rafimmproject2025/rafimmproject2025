import os
import psycopg
import pandas as pd

DB_DSN = os.getenv("DB_DSN","postgresql://postgres:postgres@db:5432/candidate_news")

def export_excel(out_file="classified_news.xlsx"):
    sql = """
    SELECT a.article_id,
           a.title,
           a.status,
           c.category,
           al.score,
           al.is_primary
    FROM article a
    JOIN article_label al ON a.article_id = al.article_id
    JOIN codebook_category c ON c.category_id = al.category_id
    WHERE a.status = 'classified'
    ORDER BY a.article_id;
    """
    with psycopg.connect(DB_DSN) as con:
        df = pd.read_sql(sql, con)
    df.to_excel(out_file, index=False)
    print(f"Exported {len(df)} rows → {out_file}")

if __name__ == "__main__":
    export_excel("/mnt/data/classified_news.xlsx")

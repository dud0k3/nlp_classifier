from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


DOCUMENTS_CSV = "data/processed/documents.csv"
KEYWORDS_CSV = "data/processed/keywords.csv"


def extract_top_keywords_per_doc(top_n: int = 10) -> pd.DataFrame:
    docs_path = Path(DOCUMENTS_CSV)
    if not docs_path.exists():
        raise FileNotFoundError("Run src.extract_texts first to create documents.csv")

    df = pd.read_csv(docs_path)
    vectorizer = TfidfVectorizer(stop_words="english", max_features=3000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df["text"])
    feature_names = vectorizer.get_feature_names_out()

    rows = []
    for row_idx, (_, row) in enumerate(df.iterrows()):
        vec = X[row_idx].toarray().ravel()
        top_idx = vec.argsort()[::-1][:top_n]
        keywords = [feature_names[i] for i in top_idx if vec[i] > 0]
        rows.append(
            {
                "path": row["path"],
                "title": row["title"],
                "keywords": ", ".join(keywords),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(KEYWORDS_CSV, index=False)
    return out


if __name__ == "__main__":
    df = extract_top_keywords_per_doc()
    print(df.to_string(index=False))

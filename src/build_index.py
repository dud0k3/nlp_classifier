from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


DOCUMENTS_CSV = "data/processed/documents.csv"
VECTORIZER_PATH = "data/processed/vectorizer.joblib"
MATRIX_PATH = "data/processed/tfidf.joblib"
CLUSTERS_PATH = "data/processed/clusters.csv"


def build_index() -> None:
    docs_path = Path(DOCUMENTS_CSV)
    if not docs_path.exists():
        raise FileNotFoundError("Run src.extract_texts first to create documents.csv")

    df = pd.read_csv(docs_path)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=3000,
        ngram_range=(1, 2),
        min_df=1,
    )
    X = vectorizer.fit_transform(df["text"])

    n_clusters = min(5, max(2, len(df) // 2))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    cluster_df = df[["path", "title"]].copy()
    cluster_df["cluster"] = labels
    cluster_df.to_csv(CLUSTERS_PATH, index=False)

    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(X, MATRIX_PATH)

    print(f"Saved vectorizer to {VECTORIZER_PATH}")
    print(f"Saved TF-IDF matrix to {MATRIX_PATH}")
    print(f"Saved clusters to {CLUSTERS_PATH}")


if __name__ == "__main__":
    build_index()

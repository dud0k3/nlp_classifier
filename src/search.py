from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


DOCUMENTS_CSV = "data/processed/documents.csv"
VECTORIZER_PATH = "data/processed/vectorizer.joblib"
MATRIX_PATH = "data/processed/tfidf.joblib"


def search(query: str, top_k: int = 5) -> pd.DataFrame:
    if not Path(DOCUMENTS_CSV).exists():
        raise FileNotFoundError("documents.csv not found. Run src.extract_texts first.")

    if not Path(VECTORIZER_PATH).exists() or not Path(MATRIX_PATH).exists():
        raise FileNotFoundError("Index files not found. Run src.build_index first.")

    df = pd.read_csv(DOCUMENTS_CSV)
    vectorizer = joblib.load(VECTORIZER_PATH)
    X = joblib.load(MATRIX_PATH)

    q = vectorizer.transform([query])
    scores = cosine_similarity(q, X).ravel()

    result = df[["path", "title"]].copy()
    result["score"] = scores
    result = result.sort_values("score", ascending=False).head(top_k)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    result = search(args.query, args.top_k)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()

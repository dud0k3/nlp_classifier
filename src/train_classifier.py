from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


DOCUMENTS_CSV = "data/processed/documents.csv"
LABELS_CSV = "data/manual_labels.csv"
MODEL_PATH = "data/processed/classifier.joblib"
PIPELINE_PATH = "data/processed/classifier_pipeline.joblib"
PREDICTIONS_PATH = "data/processed/train_predictions.csv"


def train_classifier() -> pd.DataFrame:
    docs_path = Path(DOCUMENTS_CSV)
    labels_path = Path(LABELS_CSV)

    if not docs_path.exists():
        raise FileNotFoundError("Run src.extract_texts first to create documents.csv")
    if not labels_path.exists():
        raise FileNotFoundError("manual_labels.csv not found")

    docs = pd.read_csv(docs_path)
    labels = pd.read_csv(labels_path)
    df = docs.merge(labels[["path", "label"]], on="path", how="inner")

    if df.empty:
        raise ValueError("No labeled documents matched documents.csv")

    vectorizer = TfidfVectorizer(stop_words="english", max_features=3000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df["text"])
    y = df["label"]

    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(X, y)

    preds = model.predict(X)
    report = classification_report(y, preds, digits=3)
    print("Training classification report:")
    print(report)

    out = df[["path", "title", "label"]].copy()
    out["predicted_label"] = preds
    Path(PREDICTIONS_PATH).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(PREDICTIONS_PATH, index=False)

    joblib.dump(model, MODEL_PATH)
    joblib.dump({"vectorizer": vectorizer, "model": model}, PIPELINE_PATH)

    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved pipeline to {PIPELINE_PATH}")
    print(f"Saved predictions to {PREDICTIONS_PATH}")
    print("Warning: the current labeled set is very small, so these results are only a baseline.")

    return out


if __name__ == "__main__":
    train_classifier()

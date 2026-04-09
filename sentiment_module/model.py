from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


@dataclass
class TrainingResult:
    """Container with simple training metrics."""

    test_size: float
    report: str


class TweetSentimentModule:
    """Tweet sentiment classifier with baseline and advanced pipelines.

    Available model types:
    - ``tfidf_logreg``: classic TF-IDF + Logistic Regression baseline.
    - ``tfidf_sgd_ensemble``: wider TF-IDF space + SGD (log loss) for larger corpora.
    """

    def __init__(self, model_type: str = "tfidf_logreg", random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state
        self.pipeline = self._build_pipeline(model_type)

    def _build_pipeline(self, model_type: str) -> Pipeline:
        if model_type == "tfidf_logreg":
            return Pipeline(
                steps=[
                    (
                        "tfidf",
                        TfidfVectorizer(
                            lowercase=True,
                            ngram_range=(1, 2),
                            min_df=2,
                            max_df=0.95,
                            sublinear_tf=True,
                        ),
                    ),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=2000,
                            C=3.0,
                            class_weight="balanced",
                            random_state=self.random_state,
                        ),
                    ),
                ]
            )

        if model_type == "tfidf_sgd_ensemble":
            return Pipeline(
                steps=[
                    (
                        "tfidf",
                        TfidfVectorizer(
                            lowercase=True,
                            ngram_range=(1, 3),
                            analyzer="word",
                            min_df=2,
                            max_df=0.98,
                            sublinear_tf=True,
                        ),
                    ),
                    (
                        "clf",
                        SGDClassifier(
                            loss="log_loss",
                            alpha=1e-5,
                            penalty="elasticnet",
                            l1_ratio=0.15,
                            max_iter=3000,
                            random_state=self.random_state,
                        ),
                    ),
                ]
            )

        raise ValueError(
            f"Unknown model_type={model_type!r}. "
            "Use 'tfidf_logreg' or 'tfidf_sgd_ensemble'."
        )

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> "TweetSentimentModule":
        self.pipeline.fit(texts, labels)
        return self

    def train_with_validation(
        self,
        texts: Sequence[str],
        labels: Sequence[str],
        test_size: float = 0.2,
    ) -> TrainingResult:
        x_train, x_test, y_train, y_test = train_test_split(
            texts,
            labels,
            test_size=test_size,
            random_state=self.random_state,
            stratify=labels,
        )
        self.pipeline.fit(x_train, y_train)
        predictions = self.pipeline.predict(x_test)
        report = classification_report(y_test, predictions)
        return TrainingResult(test_size=test_size, report=report)

    def predict(self, texts: Iterable[str]) -> list[str]:
        return self.pipeline.predict(list(texts)).tolist()

    def predict_proba(self, texts: Iterable[str]) -> list[list[float]]:
        texts = list(texts)
        clf = self.pipeline.named_steps["clf"]
        if not hasattr(clf, "predict_proba"):
            raise RuntimeError(
                "Current classifier does not expose predict_proba. "
                "Switch to 'tfidf_logreg' for calibrated probabilities."
            )
        probabilities = self.pipeline.predict_proba(texts)
        return probabilities.tolist()

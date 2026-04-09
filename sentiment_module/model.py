from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


@dataclass
class TrainingResult:
    """Результат обучения с коротким текстовым отчётом."""

    validation_share: float
    quality_report: str


class TweetSentimentModule:
    """Классификатор тональности твитов.

    Доступные режимы:
    - ``tfidf_logreg``: классический baseline (TF-IDF + Logistic Regression).
    - ``tfidf_sgd_ensemble``: более ёмкая TF-IDF схема + SGDClassifier.
    """

    def __init__(self, model_type: str = "tfidf_logreg", random_seed: int = 42):
        self.model_type = model_type
        self.random_seed = random_seed
        self.classification_pipeline = self._build_pipeline(model_type)

    def _build_pipeline(self, selected_model_type: str) -> Pipeline:
        if selected_model_type == "tfidf_logreg":
            return Pipeline(
                steps=[
                    (
                        "tfidf_vectorizer",
                        TfidfVectorizer(
                            lowercase=True,
                            ngram_range=(1, 2),
                            min_df=2,
                            max_df=0.95,
                            sublinear_tf=True,
                        ),
                    ),
                    (
                        "sentiment_classifier",
                        LogisticRegression(
                            max_iter=2000,
                            C=3.0,
                            class_weight="balanced",
                            random_state=self.random_seed,
                        ),
                    ),
                ]
            )

        if selected_model_type == "tfidf_sgd_ensemble":
            return Pipeline(
                steps=[
                    (
                        "tfidf_vectorizer",
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
                        "sentiment_classifier",
                        SGDClassifier(
                            loss="log_loss",
                            alpha=1e-5,
                            penalty="elasticnet",
                            l1_ratio=0.15,
                            max_iter=3000,
                            random_state=self.random_seed,
                        ),
                    ),
                ]
            )

        raise ValueError(
            f"Unknown model_type={selected_model_type!r}. "
            "Use 'tfidf_logreg' or 'tfidf_sgd_ensemble'."
        )

    def fit(
        self,
        tweet_texts: Sequence[str],
        sentiment_labels: Sequence[str],
    ) -> "TweetSentimentModule":
        self.classification_pipeline.fit(tweet_texts, sentiment_labels)
        return self

    def train_with_validation(
        self,
        tweet_texts: Sequence[str],
        sentiment_labels: Sequence[str],
        validation_share: float = 0.2,
    ) -> TrainingResult:
        (
            train_tweet_texts,
            validation_tweet_texts,
            train_sentiment_labels,
            validation_sentiment_labels,
        ) = train_test_split(
            tweet_texts,
            sentiment_labels,
            test_size=validation_share,
            random_state=self.random_seed,
            stratify=sentiment_labels,
        )

        self.classification_pipeline.fit(train_tweet_texts, train_sentiment_labels)

        predicted_labels = self.classification_pipeline.predict(validation_tweet_texts)
        quality_report = classification_report(validation_sentiment_labels, predicted_labels)

        return TrainingResult(
            validation_share=validation_share,
            quality_report=quality_report,
        )

    def predict(self, tweet_texts: Iterable[str]) -> list[str]:
        tweet_text_list = list(tweet_texts)
        return self.classification_pipeline.predict(tweet_text_list).tolist()

    def predict_proba(self, tweet_texts: Iterable[str]) -> list[list[float]]:
        tweet_text_list = list(tweet_texts)
        classifier = self.classification_pipeline.named_steps["sentiment_classifier"]
        if not hasattr(classifier, "predict_proba"):
            raise RuntimeError(
                "Current classifier does not expose predict_proba. "
                "Switch to 'tfidf_logreg' for probability scores."
            )

        class_probabilities = self.classification_pipeline.predict_proba(tweet_text_list)
        return class_probabilities.tolist()

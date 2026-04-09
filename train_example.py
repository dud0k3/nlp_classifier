"""Небольшой живой пример обучения модели на твитах."""

from sentiment_module import TweetSentimentModule


def run_demo() -> None:
    training_tweets = [
        "I love this phone, battery life is amazing!",
        "Worst service ever, very disappointed",
        "Not bad, but delivery was slow",
        "Absolutely fantastic quality!",
        "Terrible app, keeps crashing",
        "I am happy with this purchase",
        "I hate this update",
        "Great support and fast response",
        "It is okay, nothing special",
        "Awful experience, never again",
    ]
    training_labels = [
        "positive",
        "negative",
        "neutral",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "neutral",
        "negative",
    ]

    sentiment_model = TweetSentimentModule(model_type="tfidf_logreg")
    training_metrics = sentiment_model.train_with_validation(
        training_tweets,
        training_labels,
        validation_share=0.3,
    )

    print("=== Validation report ===")
    print(training_metrics.quality_report)

    tweets_for_inference = [
        "The camera is brilliant!",
        "This product is useless and bad",
        "Package arrived, all normal",
    ]
    print("=== Predictions ===")
    print(sentiment_model.predict(tweets_for_inference))


if __name__ == "__main__":
    run_demo()

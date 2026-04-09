"""Quick example for tweet sentiment training."""

from sentiment_module import TweetSentimentModule


def demo() -> None:
    tweets = [
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
    labels = [
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

    model = TweetSentimentModule(model_type="tfidf_logreg")
    metrics = model.train_with_validation(tweets, labels, test_size=0.3)

    print("=== Classification Report ===")
    print(metrics.report)

    new_tweets = [
        "The camera is brilliant!",
        "This product is useless and bad",
        "Package arrived, all normal",
    ]
    print("=== Predictions ===")
    print(model.predict(new_tweets))


if __name__ == "__main__":
    demo()

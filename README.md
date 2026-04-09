# nlp_classifier

Небольшой и понятный модуль для определения тональности твитов.
Идея простая: сначала взять рабочий baseline (`TF-IDF + Logistic Regression`), а потом при желании перейти к более «тяжёлой» конфигурации.

## Что уже реализовано

- `TweetSentimentModule` — основной класс.
- 2 режима обучения:
  - `tfidf_logreg` — стабильная классика для быстрого старта;
  - `tfidf_sgd_ensemble` — чуть более мощная конфигурация для больших наборов данных.

## Установка

```bash
pip install scikit-learn
```

## Быстрый старт

```python
from sentiment_module import TweetSentimentModule

training_texts = [
    "I love this product",
    "Worst thing ever",
    "It is okay",
]
training_labels = ["positive", "negative", "neutral"]

sentiment_model = TweetSentimentModule(model_type="tfidf_logreg")
sentiment_model.fit(training_texts, training_labels)

predicted_labels = sentiment_model.predict(["Amazing quality", "Very bad support"])
print(predicted_labels)
```

## Валидация

```python
validation_result = sentiment_model.train_with_validation(
    training_texts,
    training_labels,
    validation_share=0.2,
)
print(validation_result.quality_report)
```

## Запуск примера

```bash
python train_example.py
```

## Куда развивать дальше

1. Добавить предобработку твитов: URL, mentions, hashtags, emoji.
2. Добавить подбор гиперпараметров (`GridSearchCV`/Optuna).
3. Перейти на трансформеры (`DistilBERT`, `RuBERT`) для контекстно-сложных кейсов.
4. Сохранение модели через `joblib` и API на FastAPI.

# nlp_classifier
Модуль для определения тональности твитов.
baseline (`TF-IDF + Logistic Regression`)

## Что реализовал

- `TweetSentimentModule` — основной класс.
- 2 режима обучения:
  - `tfidf_logreg` — tf-idf + logreg.
  - `tfidf_sgd_ensemble` — чуть более мощная конфигурация для больших наборов данных.

## Установка

```bash
pip install scikit-learn
```

## Старт

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

## Пример

```bash
python train_example.py
```

## Че дальше 

1. Добавить предобработку твитов: ссылок , эмодзи , хэштегов
2. Добавить подбор гиперпараметров (`GridSearchCV`).
3. Перейти на трансформеры (`DistilBERT`, `RuBERT`) для контекста.

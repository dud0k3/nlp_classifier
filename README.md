# nlp_classifier

Мини-модуль для определения тональности твитов (sentiment analysis).

## Что внутри

- `TweetSentimentModule` — модуль классификации тональности.
- 2 режима:
  - `tfidf_logreg` — классический `TF-IDF + LogisticRegression` (базовый, стабильный).
  - `tfidf_sgd_ensemble` — более «широкий» TF-IDF + `SGDClassifier` (быстрее на больших данных).

## Установка зависимостей

```bash
pip install scikit-learn
```

## Быстрый старт

```python
from sentiment_module import TweetSentimentModule

texts = [
    "I love this product",
    "Worst thing ever",
    "It is okay",
]
labels = ["positive", "negative", "neutral"]

model = TweetSentimentModule(model_type="tfidf_logreg")
model.fit(texts, labels)

print(model.predict(["Amazing quality", "Very bad support"]))
```

## Валидация

```python
metrics = model.train_with_validation(texts, labels, test_size=0.2)
print(metrics.report)
```

## Пример запуска

```bash
python train_example.py
```

## Идеи для расширения (более сложно и обширно)

1. Добавить очистку твитов: hashtags, mentions, urls, emoji-нормализация.
2. Поддержать `class_weight` по данным и Optuna/grid-search.
3. Перейти на трансформеры (`DistilBERT`, `RuBERT`) для лучшего качества на сложных контекстах.
4. Добавить сохранение модели (`joblib`) и REST API (FastAPI).

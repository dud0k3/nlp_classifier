# Twitter Elections Integrity NLP Project

A complete NLP portfolio project built around the archived static Twitter website about Elections Integrity.

## What this project does

This project turns an archived static website into a small NLP corpus and then builds:

- **text extraction** from HTML pages
- **document metadata table**
- **TF-IDF document vectors**
- **semantic search** over pages with cosine similarity
- **keyword extraction** for each page
- **document clustering** with KMeans
- a simple **CLI** for indexing and search

## Important note about the dataset

The uploaded archive is **not** a standard machine learning dataset with labels and it does **not** contain a trained NLP model.
It is a static website snapshot with HTML pages and assets.

So the project is designed correctly around the real contents of the archive:
we first extract textual content and then apply NLP methods to the extracted documents.

## Recommended repository structure

```text
nlp_classifier/
├── data/
│   └── processed/
├── notebooks/
├── src/
│   ├── extract_texts.py
│   ├── build_index.py
│   ├── search.py
│   ├── keywords.py
│   └── utils.py
├── tests/
│   └── test_utils.py
├── requirements.txt
└── README.md
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How to use

### 1. Put the archive in the project root

Expected filename:

```text
twitter-data-staticsite.zip
```

### 2. Extract documents from HTML

```bash
python -m src.extract_texts --archive twitter-data-staticsite.zip
```

This creates:

- `data/processed/documents.csv`

### 3. Build TF-IDF index and clusters

```bash
python -m src.build_index
```

This creates:

- `data/processed/tfidf.joblib`
- `data/processed/vectorizer.joblib`
- `data/processed/clusters.csv`

### 4. Search the corpus

```bash
python -m src.search --query "political advertising transparency"
```

### 5. Extract top keywords per page

```bash
python -m src.keywords
```

This creates:

- `data/processed/keywords.csv`

## Why this is a good GitHub NLP project

This repo shows that you can:

- inspect an unfamiliar dataset correctly
- convert raw web documents into structured text
- build a real NLP pipeline
- document assumptions honestly
- ship reproducible code

## Future improvements

- add sentence-transformer embeddings
- add topic modeling with BERTopic or LDA
- add a Streamlit app
- add evaluation for retrieval quality
- compare TF-IDF search vs embedding search

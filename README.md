# KGIN Knowledge Graph Book Recommendation System

This project is a graduation-design style book recommender based on:

- Knowledge graph construction in Neo4j
- KGIN model training with BPR ranking loss
- Flask backend APIs + built-in single-page frontend
- Explainable recommendation evidence (author/category/tag/two-hop)

Only **KGIN** is kept as the recommendation model in this repository.

## 1. Data preparation

Put these files under `D:\Book_Recommendation\data\raw`:

- `books.csv`
- `ratings.csv`
- `book_tags.csv`
- `tags.csv`
- `to_read.csv` (recommended)

## 2. Environment

```bat
cd /d D:\Book_Recommendation
py -3 -m venv .venv
.venv\Scripts\activate
pip install -r backend\requirements.txt
```

## 3. End-to-end pipeline (KGIN only)

```bat
cd /d D:\Book_Recommendation
.venv\Scripts\activate

python scripts\01_preprocess.py
python scripts\02_import_neo4j.py
python scripts\07_train_kgin.py
python scripts\05_evaluate.py --eval-mode mixed --model-type kgin
python backend\app.py
```

Open in browser:

- [http://127.0.0.1:5000](http://127.0.0.1:5000)

## 4. Model behavior

`ModelService` loads only KGIN artifacts:

- `artifacts/kgin_book_embeddings.npy`
- `artifacts/kgin_book2idx.json`
- `artifacts/kgin_user_embeddings.npy` (optional for extension)
- `artifacts/kgin_user2idx.json` (optional for extension)

Model status API:

- `GET /api/model/status`

## 5. Main scripts

- `scripts/01_preprocess.py`
  - data cleaning and field alignment
  - noisy tag filtering
  - outputs processed training data and triples
- `scripts/02_import_neo4j.py`
  - import triples into Neo4j
- `scripts/07_train_kgin.py`
  - train KGIN with user-book feedback + KG edges
- `scripts/05_evaluate.py`
  - offline metrics: `HitRate@K`, `NDCG@K`, `MRR`
  - eval mode: `rating | to_read | mixed`

## 6. Main APIs

- `POST /api/auth/login`
- `GET /api/books/search?keyword=...&limit=...`
- `POST /api/rate`
- `GET /api/ratings?user_id=...`
- `GET /api/recommend?user_id=...&top_n=...`
- `GET /api/model/status`
- `GET /api/model/metrics`

## 7. Notes

- This repository is configured for **KGIN-only** experiments and deployment.
- Baseline model comparison scripts were removed.

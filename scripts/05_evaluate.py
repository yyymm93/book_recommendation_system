from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate recommendation by HitRate@K, NDCG@K, MRR."
    )
    parser.add_argument(
        "--ratings-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "user_book_ratings.csv",
        help="Path to processed ratings csv.",
    )
    parser.add_argument(
        "--to-read-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "user_book_to_read.csv",
        help="Path to processed to-read csv.",
    )
    parser.add_argument(
        "--eval-mode",
        type=str,
        default="mixed",
        choices=["rating", "to_read", "mixed"],
        help="Positive sample source: rating / to_read / mixed.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="kgin",
        choices=["kgin"],
        help="Only KGIN artifacts are supported.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "eval_metrics.json",
        help="Path to save evaluation json result.",
    )
    parser.add_argument(
        "--ks",
        type=str,
        default="5,10,20",
        help="Comma-separated K values, e.g. 5,10,20",
    )
    parser.add_argument(
        "--min-rating",
        type=float,
        default=4.0,
        help="Only ratings >= min_rating are treated as positive in rating-based modes.",
    )
    parser.add_argument(
        "--min-history",
        type=int,
        default=3,
        help="Minimum positive books per user before leave-one-out split.",
    )
    parser.add_argument(
        "--neg-sample",
        type=int,
        default=100,
        help="Number of negatives sampled per user for ranking.",
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=0,
        help="Max users to evaluate (0 means all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def _clean_user_book_df(df: pd.DataFrame, user_col: str, book_col: str) -> pd.DataFrame:
    out = df[[user_col, book_col]].copy()
    out[user_col] = pd.to_numeric(out[user_col], errors="coerce")
    out[book_col] = pd.to_numeric(out[book_col], errors="coerce")
    out = out.dropna(subset=[user_col, book_col])
    out[user_col] = out[user_col].astype(int)
    out[book_col] = out[book_col].astype(int).astype(str)
    return out.rename(columns={user_col: "user_id", book_col: "book_id"})


def load_inputs(
    ratings_path: Path,
    to_read_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, dict[str, int], str]:
    if not ratings_path.exists():
        raise FileNotFoundError(f"ratings file not found: {ratings_path}")

    ratings_raw = pd.read_csv(ratings_path)
    for col in ["user_id", "book_id", "rating"]:
        if col not in ratings_raw.columns:
            raise ValueError(f"ratings csv missing column: {col}")
    ratings = ratings_raw[["user_id", "book_id", "rating"]].copy()
    ratings["user_id"] = pd.to_numeric(ratings["user_id"], errors="coerce")
    ratings["book_id"] = pd.to_numeric(ratings["book_id"], errors="coerce")
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    ratings = ratings.dropna(subset=["user_id", "book_id", "rating"])
    ratings["user_id"] = ratings["user_id"].astype(int)
    ratings["book_id"] = ratings["book_id"].astype(int).astype(str)
    ratings["rating"] = ratings["rating"].astype(float)

    if to_read_path.exists():
        to_read_raw = pd.read_csv(to_read_path)
        for col in ["user_id", "book_id"]:
            if col not in to_read_raw.columns:
                raise ValueError(f"to_read csv missing column: {col}")
        to_read = _clean_user_book_df(to_read_raw, "user_id", "book_id")
    else:
        to_read = pd.DataFrame(columns=["user_id", "book_id"])

    artifacts = PROJECT_ROOT / "artifacts"
    emb_path = artifacts / "kgin_book_embeddings.npy"
    book_map_path = artifacts / "kgin_book2idx.json"
    selected_name = "kgin"

    if not emb_path.exists():
        raise FileNotFoundError(f"embedding file not found: {emb_path}")
    if not book_map_path.exists():
        raise FileNotFoundError(f"book map file not found: {book_map_path}")

    entity_emb = np.load(emb_path)
    book_entity_ids: dict[str, int] = json.loads(book_map_path.read_text(encoding="utf-8"))
    return ratings, to_read, entity_emb, book_entity_ids, selected_name


def build_user_histories(
    ratings: pd.DataFrame,
    to_read: pd.DataFrame,
    valid_books: set[str],
    eval_mode: str,
    min_rating: float,
    min_history: int,
) -> tuple[dict[int, list[str]], dict[str, int]]:
    positives = []

    if eval_mode in {"rating", "mixed"}:
        r = ratings[ratings["rating"] >= min_rating].copy()
        r = r[r["book_id"].isin(valid_books)]
        r = r[["user_id", "book_id"]]
        positives.append(r)

    if eval_mode in {"to_read", "mixed"}:
        if not to_read.empty:
            t = to_read[to_read["book_id"].isin(valid_books)].copy()
            t = t[["user_id", "book_id"]]
            positives.append(t)

    if not positives:
        return {}, {"rating_positives": 0, "to_read_positives": 0, "merged_positives": 0}

    rating_count = len(positives[0]) if (eval_mode in {"rating", "mixed"}) else 0
    to_read_count = len(positives[-1]) if (eval_mode in {"to_read", "mixed"} and not to_read.empty) else 0

    merged = pd.concat(positives, ignore_index=True).drop_duplicates(subset=["user_id", "book_id"])
    grouped = merged.groupby("user_id")["book_id"].apply(list).to_dict()
    histories = {int(uid): books for uid, books in grouped.items() if len(books) >= (min_history + 1)}

    stats = {
        "rating_positives": int(rating_count),
        "to_read_positives": int(to_read_count),
        "merged_positives": int(len(merged)),
    }
    return histories, stats


def cosine_scores(
    profile_vec: np.ndarray,
    candidate_book_ids: list[str],
    entity_emb: np.ndarray,
    book_entity_ids: dict[str, int],
) -> np.ndarray:
    idx = [book_entity_ids[bid] for bid in candidate_book_ids]
    cand = entity_emb[idx]
    cand_norm = cand / (np.linalg.norm(cand, axis=1, keepdims=True) + 1e-8)
    p = profile_vec / (np.linalg.norm(profile_vec) + 1e-8)
    return np.dot(cand_norm, p)


def evaluate(
    histories: dict[int, list[str]],
    entity_emb: np.ndarray,
    book_entity_ids: dict[str, int],
    ks: list[int],
    neg_sample: int,
    seed: int,
    max_users: int,
) -> dict:
    rng = np.random.default_rng(seed)
    all_books = np.array(sorted(book_entity_ids.keys()))

    hit_sum = {k: 0.0 for k in ks}
    ndcg_sum = {k: 0.0 for k in ks}
    mrr_sum = 0.0
    evaluated_users = 0

    user_items = list(histories.items())
    rng.shuffle(user_items)
    if max_users > 0:
        user_items = user_items[:max_users]

    for _uid, books in user_items:
        test_book = rng.choice(np.array(books))
        train_books = [b for b in books if b != test_book]
        if not train_books:
            continue

        train_idx = [book_entity_ids[b] for b in train_books if b in book_entity_ids]
        if not train_idx:
            continue

        profile = entity_emb[train_idx].mean(axis=0)

        excluded = set(train_books)
        excluded.add(test_book)
        neg_pool = np.array([b for b in all_books if b not in excluded])
        if neg_pool.size == 0:
            continue

        sample_size = min(int(neg_sample), int(neg_pool.size))
        negatives = rng.choice(neg_pool, size=sample_size, replace=False).tolist()
        candidates = [str(test_book)] + [str(b) for b in negatives]

        scores = cosine_scores(profile, candidates, entity_emb, book_entity_ids)
        order = np.argsort(-scores)
        ranked = [candidates[i] for i in order]
        rank = ranked.index(str(test_book)) + 1

        for k in ks:
            if rank <= k:
                hit_sum[k] += 1.0
                ndcg_sum[k] += 1.0 / math.log2(rank + 1.0)

        mrr_sum += 1.0 / rank
        evaluated_users += 1

    if evaluated_users == 0:
        return {
            "evaluated_users": 0,
            "metrics": {},
        }

    metrics = {f"hit_rate@{k}": round(hit_sum[k] / evaluated_users, 6) for k in ks}
    metrics.update({f"ndcg@{k}": round(ndcg_sum[k] / evaluated_users, 6) for k in ks})
    metrics["mrr"] = round(mrr_sum / evaluated_users, 6)

    return {
        "evaluated_users": evaluated_users,
        "metrics": metrics,
    }


def main() -> None:
    args = parse_args()
    ks = sorted({int(x.strip()) for x in args.ks.split(",") if x.strip()})
    if not ks:
        raise ValueError("ks must contain at least one positive integer")
    if min(ks) <= 0:
        raise ValueError("all K values must be > 0")

    ratings, to_read, entity_emb, book_entity_ids, selected_model = load_inputs(
        ratings_path=args.ratings_path,
        to_read_path=args.to_read_path,
    )
    histories, source_stats = build_user_histories(
        ratings=ratings,
        to_read=to_read,
        valid_books=set(book_entity_ids.keys()),
        eval_mode=args.eval_mode,
        min_rating=float(args.min_rating),
        min_history=int(args.min_history),
    )

    result = evaluate(
        histories=histories,
        entity_emb=entity_emb,
        book_entity_ids=book_entity_ids,
        ks=ks,
        neg_sample=int(args.neg_sample),
        seed=int(args.seed),
        max_users=int(args.max_users),
    )
    result["source_stats"] = source_stats
    result["config"] = {
        "ratings_path": str(args.ratings_path),
        "to_read_path": str(args.to_read_path),
        "eval_mode": args.eval_mode,
        "model_type": "kgin",
        "selected_model": selected_model,
        "emb_path": str(PROJECT_ROOT / "artifacts" / "kgin_book_embeddings.npy"),
        "book_map_path": str(PROJECT_ROOT / "artifacts" / "kgin_book2idx.json"),
        "ks": ks,
        "min_rating": float(args.min_rating),
        "min_history": int(args.min_history),
        "neg_sample": int(args.neg_sample),
        "max_users": int(args.max_users),
        "seed": int(args.seed),
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("=== Evaluation Result ===")
    print(f"embedding_model: {selected_model}")
    print(f"eval_mode: {args.eval_mode}")
    print(f"source_stats: {source_stats}")
    print(f"evaluated_users: {result['evaluated_users']}")
    if result["metrics"]:
        for k, v in result["metrics"].items():
            print(f"{k}: {v}")
    else:
        print("metrics: empty (not enough valid users/interactions)")
    print(f"saved_to: {args.output_path}")


if __name__ == "__main__":
    main()

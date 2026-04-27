from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from backend.kgns.kgin import train_kgin_bpr

PROCESSED = PROJECT_ROOT / "data" / "processed"
ARTIFACTS = PROJECT_ROOT / "artifacts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train KGIN-lite recommender with BPR.")
    parser.add_argument(
        "--feedback-path",
        type=Path,
        default=PROCESSED / "user_book_feedback.csv",
        help="Path to merged feedback csv with columns user_id, book_id, score",
    )
    parser.add_argument(
        "--triples-path",
        type=Path,
        default=PROCESSED / "triples.csv",
        help="Path to KG triples csv with columns head, relation, tail",
    )
    parser.add_argument(
        "--books-path",
        type=Path,
        default=PROCESSED / "books_clean.csv",
        help="Path to cleaned books csv with column book_id",
    )
    parser.add_argument("--min-score", type=float, default=0.6)
    parser.add_argument("--emb-dim", type=int, default=64)
    parser.add_argument("--num-intents", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--reg", type=float, default=1e-4)
    parser.add_argument("--intent-reg", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--samples-per-epoch", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def _parse_book_token(token: str) -> str | None:
    t = str(token)
    if t.startswith("book:"):
        return t.split(":", 1)[1]
    return None


def main() -> None:
    args = parse_args()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    if not args.feedback_path.exists():
        raise FileNotFoundError(f"feedback file not found: {args.feedback_path}")
    if not args.triples_path.exists():
        raise FileNotFoundError(f"triples file not found: {args.triples_path}")
    if not args.books_path.exists():
        raise FileNotFoundError(f"books file not found: {args.books_path}")

    feedback = pd.read_csv(args.feedback_path)
    for col in ["user_id", "book_id", "score"]:
        if col not in feedback.columns:
            raise ValueError(f"{args.feedback_path} missing column: {col}")

    feedback["user_id"] = pd.to_numeric(feedback["user_id"], errors="coerce")
    feedback["book_id"] = pd.to_numeric(feedback["book_id"], errors="coerce")
    feedback["score"] = pd.to_numeric(feedback["score"], errors="coerce")
    feedback = feedback.dropna(subset=["user_id", "book_id", "score"])
    feedback["user_id"] = feedback["user_id"].astype(int)
    feedback["book_id"] = feedback["book_id"].astype(int).astype(str)
    feedback["score"] = feedback["score"].astype(float)
    feedback = feedback[feedback["score"] >= float(args.min_score)].copy()
    feedback = feedback.drop_duplicates(subset=["user_id", "book_id"], keep="first").reset_index(drop=True)
    if feedback.empty:
        raise ValueError("no interactions left after score filtering")

    books = pd.read_csv(args.books_path)
    if "book_id" not in books.columns:
        raise ValueError("books_clean.csv must contain book_id")
    books["book_id"] = pd.to_numeric(books["book_id"], errors="coerce")
    books = books.dropna(subset=["book_id"])
    books["book_id"] = books["book_id"].astype(int).astype(str)
    books = books.drop_duplicates(subset=["book_id"], keep="first")

    # Keep all books in item universe so model can recommend beyond interacted subset.
    book_ids = sorted(books["book_id"].tolist(), key=lambda x: int(x))
    user_ids = sorted(feedback["user_id"].unique().tolist())

    user2idx = {str(uid): i for i, uid in enumerate(user_ids)}
    book2idx = {str(bid): i for i, bid in enumerate(book_ids)}

    feedback = feedback[feedback["book_id"].isin(set(book2idx.keys()))].copy()
    feedback["u_idx"] = feedback["user_id"].map(lambda x: user2idx[str(int(x))])
    feedback["i_idx"] = feedback["book_id"].map(lambda x: book2idx[str(x)])
    interactions = feedback[["u_idx", "i_idx"]].to_numpy(dtype=np.int64)

    triples = pd.read_csv(args.triples_path)
    for col in ["head", "relation", "tail"]:
        if col not in triples.columns:
            raise ValueError(f"{args.triples_path} missing column: {col}")

    rel_values = sorted(triples["relation"].astype(str).unique().tolist())
    relation2idx = {r: i for i, r in enumerate(rel_values)}

    # In KGIN, item->entity edges use non-book tail nodes as entities.
    entity_tokens = sorted(
        {
            str(t)
            for t in triples["tail"].astype(str).tolist()
            if not str(t).startswith("book:")
        }
    )
    entity2idx = {t: i for i, t in enumerate(entity_tokens)}

    kg_edges_list = []
    for row in triples.itertuples(index=False):
        head = str(row.head)
        rel = str(row.relation)
        tail = str(row.tail)

        bid = _parse_book_token(head)
        if bid is None:
            continue
        i_idx = book2idx.get(str(bid))
        if i_idx is None:
            continue
        if tail.startswith("book:"):
            # skip item-item here to keep entity space cleaner
            continue
        e_idx = entity2idx.get(tail)
        r_idx = relation2idx.get(rel)
        if e_idx is None or r_idx is None:
            continue
        kg_edges_list.append((int(i_idx), int(r_idx), int(e_idx)))

    if not kg_edges_list:
        raise ValueError("no valid kg edges built from triples")
    kg_edges = np.array(kg_edges_list, dtype=np.int64)

    result = train_kgin_bpr(
        interactions=interactions,
        kg_edges=kg_edges,
        num_users=len(user2idx),
        num_items=len(book2idx),
        num_entities=len(entity2idx),
        num_relations=len(relation2idx),
        emb_dim=int(args.emb_dim),
        num_intents=int(args.num_intents),
        alpha=float(args.alpha),
        beta=float(args.beta),
        dropout=float(args.dropout),
        epochs=int(args.epochs),
        lr=float(args.lr),
        reg=float(args.reg),
        intent_reg=float(args.intent_reg),
        batch_size=int(args.batch_size),
        samples_per_epoch=int(args.samples_per_epoch),
        seed=int(args.seed),
        device=args.device,
    )

    np.save(ARTIFACTS / "kgin_user_embeddings.npy", result.user_emb)
    np.save(ARTIFACTS / "kgin_book_embeddings.npy", result.item_emb)
    (ARTIFACTS / "kgin_user2idx.json").write_text(
        json.dumps(user2idx, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (ARTIFACTS / "kgin_book2idx.json").write_text(
        json.dumps(book2idx, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (ARTIFACTS / "kgin_entity2idx.json").write_text(
        json.dumps(entity2idx, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (ARTIFACTS / "kgin_relation2idx.json").write_text(
        json.dumps(relation2idx, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (ARTIFACTS / "kgin_meta.json").write_text(
        json.dumps(
            {
                "num_users": len(user2idx),
                "num_books": len(book2idx),
                "num_entities": len(entity2idx),
                "num_relations": len(relation2idx),
                "emb_dim": int(args.emb_dim),
                "num_intents": int(args.num_intents),
                "epochs": int(args.epochs),
                "min_score": float(args.min_score),
                "avg_loss": result.avg_loss,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[ok] interactions: {len(interactions)}")
    print(f"[ok] kg_edges: {len(kg_edges)}")
    print(
        f"[ok] users={len(user2idx)}, books={len(book2idx)}, "
        f"entities={len(entity2idx)}, relations={len(relation2idx)}"
    )
    print(f"[ok] avg_loss: {result.avg_loss:.6f}")
    print(f"[ok] saved: {ARTIFACTS / 'kgin_book_embeddings.npy'}")


if __name__ == "__main__":
    main()

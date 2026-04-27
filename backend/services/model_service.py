from __future__ import annotations

import json

import numpy as np

from config import Settings


class ModelService:
    def __init__(self) -> None:
        self.kgin_book_emb: np.ndarray | None = None
        self.kgin_book2idx: dict[str, int] = {}
        self.kgin_user_emb: np.ndarray | None = None
        self.kgin_user2idx: dict[str, int] = {}

        self.active_model: str = "none"

    def load(self) -> bool:
        if self._load_kgin():
            self.active_model = "kgin"
            return True

        self.active_model = "none"
        return False

    def _load_kgin(self) -> bool:
        book_emb_path = Settings.ARTIFACT_ROOT / "kgin_book_embeddings.npy"
        book_map_path = Settings.ARTIFACT_ROOT / "kgin_book2idx.json"
        user_emb_path = Settings.ARTIFACT_ROOT / "kgin_user_embeddings.npy"
        user_map_path = Settings.ARTIFACT_ROOT / "kgin_user2idx.json"
        if not book_emb_path.exists() or not book_map_path.exists():
            return False

        self.kgin_book_emb = np.load(book_emb_path)
        self.kgin_book2idx = json.loads(book_map_path.read_text(encoding="utf-8"))
        if user_emb_path.exists() and user_map_path.exists():
            self.kgin_user_emb = np.load(user_emb_path)
            self.kgin_user2idx = json.loads(user_map_path.read_text(encoding="utf-8"))
        return True

    def recommend_by_profile(
        self,
        rated_books: list[str],
        top_n: int = 10,
        excluded: set[str] | None = None,
        book_weights: dict[str, float] | None = None,
    ) -> list[dict]:
        excluded = excluded or set()
        candidate_books = [b for b in self.available_book_ids() if b not in excluded]
        score_map = self.score_candidates_by_profile(
            rated_books=rated_books,
            candidate_books=candidate_books,
            excluded=excluded,
            book_weights=book_weights,
        )
        if not score_map:
            return []

        scored = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
        return [{"book_id": bid, "score": round(score, 6)} for bid, score in scored[:top_n]]

    def available_book_ids(self) -> list[str]:
        if self.kgin_book_emb is not None and self.kgin_book2idx:
            return list(self.kgin_book2idx.keys())
        return []

    def score_candidates_by_profile(
        self,
        rated_books: list[str],
        candidate_books: list[str],
        excluded: set[str] | None = None,
        book_weights: dict[str, float] | None = None,
    ) -> dict[str, float]:
        excluded = excluded or set()
        profile = self._build_profile_vector(rated_books=rated_books, book_weights=book_weights)
        if profile is None:
            return {}

        scored: dict[str, float] = {}
        for book_id in candidate_books:
            bid = str(book_id)
            if bid in excluded:
                continue
            emb = self._get_book_embedding(bid)
            if emb is None:
                continue
            e = emb / (np.linalg.norm(emb) + 1e-8)
            score = float(np.dot(profile, e))
            scored[bid] = score
        return scored

    def _build_profile_vector(
        self,
        rated_books: list[str],
        book_weights: dict[str, float] | None = None,
    ) -> np.ndarray | None:
        vectors = []
        weights = []
        book_weights = book_weights or {}

        for b in rated_books:
            bid = str(b)
            emb = self._get_book_embedding(bid)
            if emb is None:
                continue
            vectors.append(emb)
            weights.append(max(0.05, float(book_weights.get(bid, 1.0))))

        if not vectors:
            return None

        vec = np.stack(vectors, axis=0)
        w_arr = np.array(weights, dtype=np.float32)
        w_arr = w_arr / (np.sum(w_arr) + 1e-8)
        profile = np.sum(vec * w_arr[:, None], axis=0)
        profile = profile / (np.linalg.norm(profile) + 1e-8)
        return profile

    def _get_book_embedding(self, book_id: str) -> np.ndarray | None:
        bid = str(book_id)

        if self.kgin_book_emb is not None and self.kgin_book2idx:
            idx = self.kgin_book2idx.get(bid)
            if idx is not None:
                return self.kgin_book_emb[int(idx)]

        return None

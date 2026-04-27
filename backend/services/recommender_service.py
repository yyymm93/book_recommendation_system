from __future__ import annotations

from collections import defaultdict
import math

from services.local_store import list_user_ratings
from services.model_service import ModelService
from services.neo4j_service import Neo4jService


class RecommenderService:
    REASON_WEIGHTS = {
        "AUTHOR": 3.0,
        "CATEGORY": 2.0,
        "TAG": 1.5,
        "TWO_HOP": 1.0,
    }

    REASON_LABELS = {
        "AUTHOR": "\u540c\u4f5c\u8005\u5173\u8054",
        "CATEGORY": "\u540c\u7c7b\u522b\u5173\u8054",
        "TAG": "\u540c\u6807\u7b7e\u5173\u8054",
        "TWO_HOP": "\u4e8c\u8df3\u8def\u5f84\u5173\u8054",
    }

    FUSION_WEIGHTS = {
        "graph": 0.5,
        "embedding": 0.35,
        "quality": 0.15,
    }
    DIVERSITY_AUTHOR_PENALTY = 10.0
    DIVERSITY_CATEGORY_PENALTY = 6.0
    DIVERSITY_NEW_AUTHOR_BONUS = 2.5
    DIVERSITY_NEW_CATEGORY_BONUS = 1.5

    def __init__(self, neo4j_service: Neo4jService, model_service: ModelService) -> None:
        self.neo4j = neo4j_service
        self.model = model_service

    def recommend_for_user(
        self,
        user_id: int,
        top_n: int = 10,
        diversity_strength: float = 1.0,
    ) -> list[dict]:
        ratings = list_user_ratings(user_id)
        if not ratings:
            return []

        seed_scores = {str(r["book_id"]): float(r["rating"]) for r in ratings}
        seed_book_ids = list(seed_scores.keys())
        seed_emb_weights = {
            bid: (0.2 + max(0.0, min(5.0, score)) / 5.0) for bid, score in seed_scores.items()
        }

        graph_links = self.neo4j.graph_candidate_links(
            seed_book_ids=seed_book_ids,
            excluded_book_ids=seed_book_ids,
            per_reason_limit=max(30, top_n * 10),
        )

        graph_agg: dict[str, dict] = {}
        for link in graph_links:
            candidate_id = str(link["candidate_id"])
            seed_book_id = str(link["seed_book_id"])
            reason_type = str(link["reason_type"])
            reason_value = str(link["reason_value"])
            bridge_title = str(link["bridge_title"])
            seed_title = str(link["seed_title"])

            seed_weight = 1.0 + max(0.0, min(5.0, seed_scores.get(seed_book_id, 3.0))) / 5.0
            reason_weight = self.REASON_WEIGHTS.get(reason_type, 0.5)
            score_add = reason_weight * seed_weight

            if candidate_id not in graph_agg:
                graph_agg[candidate_id] = {
                    "graph_score": 0.0,
                    "evidence": [],
                    "seed_titles": set(),
                }

            graph_agg[candidate_id]["graph_score"] += score_add
            graph_agg[candidate_id]["seed_titles"].add(seed_title)
            graph_agg[candidate_id]["evidence"].append(
                {
                    "seed_book_id": seed_book_id,
                    "seed_title": seed_title,
                    "type": reason_type,
                    "value": reason_value,
                    "bridge_title": bridge_title,
                    "weight": round(score_add, 4),
                }
            )

        emb_recs = self.model.recommend_by_profile(
            rated_books=seed_book_ids,
            top_n=max(200, top_n * 20),
            excluded=set(seed_book_ids),
            book_weights=seed_emb_weights,
        )
        emb_recs_ids = [str(r["book_id"]) for r in emb_recs]

        candidate_ids = set(graph_agg.keys()).union(set(emb_recs_ids))
        if not candidate_ids:
            return []

        emb_score_map = self.model.score_candidates_by_profile(
            rated_books=seed_book_ids,
            candidate_books=sorted(candidate_ids),
            excluded=set(seed_book_ids),
            book_weights=seed_emb_weights,
        )

        rows = self.neo4j.books_by_ids(sorted(candidate_ids))
        info_map = {str(r["book_id"]): r for r in rows}

        graph_raw_map = {cid: float(graph_agg.get(cid, {}).get("graph_score", 0.0)) for cid in candidate_ids}
        emb_raw_map = {cid: float(emb_score_map.get(cid, 0.0)) for cid in candidate_ids}
        quality_raw_map = {cid: self._book_quality_score(info_map.get(cid, {})) for cid in candidate_ids}

        graph_norm_map = self._minmax_normalize_map(graph_raw_map)
        emb_norm_map = self._minmax_normalize_map(emb_raw_map)
        quality_norm_map = self._minmax_normalize_map(quality_raw_map)

        merged = []
        for candidate_id in candidate_ids:
            row = info_map.get(candidate_id)
            if not row:
                continue

            graph_score = graph_raw_map.get(candidate_id, 0.0)
            emb_score = emb_raw_map.get(candidate_id, 0.0)
            quality_score = quality_raw_map.get(candidate_id, 0.0)

            graph_norm = graph_norm_map.get(candidate_id, 0.0)
            emb_norm = emb_norm_map.get(candidate_id, 0.0)
            quality_norm = quality_norm_map.get(candidate_id, 0.0)

            final_score_norm = (
                self.FUSION_WEIGHTS["graph"] * graph_norm
                + self.FUSION_WEIGHTS["embedding"] * emb_norm
                + self.FUSION_WEIGHTS["quality"] * quality_norm
            )
            final_score = round(final_score_norm * 100.0, 6)
            evidence = graph_agg.get(candidate_id, {}).get("evidence", [])
            seed_titles = sorted(graph_agg.get(candidate_id, {}).get("seed_titles", set()))

            merged.append(
                {
                    "book_id": candidate_id,
                    "title": row["title"],
                    "author": row["author"],
                    "category": row["category"],
                    "score": final_score,
                    "graph_score": round(graph_score, 6),
                    "embedding_score": round(emb_score, 6),
                    "quality_score": round(quality_score, 6),
                    "graph_norm": round(graph_norm, 6),
                    "embedding_norm": round(emb_norm, 6),
                    "quality_norm": round(quality_norm, 6),
                    "reason": self._build_reason(
                        evidence=evidence,
                        seed_titles=seed_titles,
                        emb_score=emb_score,
                        quality_score=quality_score,
                    ),
                    "evidence": evidence[:8],
                    "seed_titles": seed_titles[:5],
                }
            )

        merged.sort(key=lambda x: x["score"], reverse=True)
        reranked = self._diversify_re_rank(
            merged,
            top_n=top_n,
            diversity_strength=diversity_strength,
        )
        for idx, item in enumerate(reranked, start=1):
            item["rank"] = idx
        return reranked

    def _build_reason(
        self,
        evidence: list[dict],
        seed_titles: list[str],
        emb_score: float,
        quality_score: float,
    ) -> str:
        evidence = evidence or []
        seed_titles = seed_titles or []

        type_counter: dict[str, int] = defaultdict(int)
        for item in evidence:
            type_counter[str(item.get("type", ""))] += 1

        reason_parts = []
        for reason_type, _weight in sorted(
            self.REASON_WEIGHTS.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            if type_counter.get(reason_type, 0) > 0:
                reason_parts.append(self.REASON_LABELS.get(reason_type, reason_type))

        if not reason_parts:
            reason_parts.append("\u5411\u91cf\u76f8\u4f3c\u63a8\u8350")

        seed_part = ""
        if seed_titles:
            seed_titles_text = "\u300b\u300a".join(seed_titles[:2])
            seed_part = f"\uff1b\u4e3b\u8981\u4f9d\u636e\uff1a\u300a{seed_titles_text}\u300b"

        emb_part = ""
        if emb_score > 0:
            emb_part = f"\uff1b\u5d4c\u5165\u76f8\u4f3c\u5ea6 {emb_score:.3f}"
        quality_part = ""
        if quality_score > 0.5:
            quality_part = "\uff1b\u4f18\u8d28\u53e3\u7891\u52a0\u6743"

        return "\u3001".join(reason_parts) + seed_part + emb_part + quality_part

    def _minmax_normalize_map(self, values: dict[str, float]) -> dict[str, float]:
        if not values:
            return {}
        vmin = min(values.values())
        vmax = max(values.values())
        if abs(vmax - vmin) < 1e-9:
            return {k: 0.0 for k in values.keys()}
        return {k: (v - vmin) / (vmax - vmin) for k, v in values.items()}

    def _book_quality_score(self, row: dict | None) -> float:
        row = row or {}
        avg_rating = float(row.get("average_rating") or 0.0)
        ratings_count = float(row.get("ratings_count") or 0.0)
        if avg_rating <= 0:
            return 0.0

        rating_part = max(0.0, min(5.0, avg_rating)) / 5.0
        count_part = min(1.0, math.log1p(max(0.0, ratings_count)) / math.log1p(100000.0))
        return 0.7 * rating_part + 0.3 * count_part

    def _diversify_re_rank(
        self,
        ranked: list[dict],
        top_n: int,
        diversity_strength: float = 1.0,
    ) -> list[dict]:
        if not ranked:
            return []
        diversity_strength = max(0.0, min(2.0, float(diversity_strength)))

        # strength=0 means no diversification, keep original relevance ranking.
        if diversity_strength <= 1e-9:
            return ranked[: int(top_n)]

        remaining = list(ranked)
        selected: list[dict] = []
        author_count: dict[str, int] = defaultdict(int)
        category_count: dict[str, int] = defaultdict(int)

        while remaining and len(selected) < int(top_n):
            best_idx = 0
            best_adjusted = float("-inf")

            for i, item in enumerate(remaining):
                base = float(item.get("score", 0.0))
                author = str(item.get("author") or "").strip().lower()
                category = str(item.get("category") or "").strip().lower()

                a_cnt = author_count.get(author, 0) if author else 0
                c_cnt = category_count.get(category, 0) if category else 0

                penalty = diversity_strength * (
                    self.DIVERSITY_AUTHOR_PENALTY * float(a_cnt)
                    + self.DIVERSITY_CATEGORY_PENALTY * float(c_cnt)
                )
                bonus = 0.0
                if author and a_cnt == 0:
                    bonus += diversity_strength * self.DIVERSITY_NEW_AUTHOR_BONUS
                if category and c_cnt == 0:
                    bonus += diversity_strength * self.DIVERSITY_NEW_CATEGORY_BONUS

                adjusted = base - penalty + bonus
                if adjusted > best_adjusted:
                    best_adjusted = adjusted
                    best_idx = i

            chosen = remaining.pop(best_idx)
            selected.append(chosen)

            chosen_author = str(chosen.get("author") or "").strip().lower()
            chosen_category = str(chosen.get("category") or "").strip().lower()
            if chosen_author:
                author_count[chosen_author] += 1
            if chosen_category:
                category_count[chosen_category] += 1

        return selected

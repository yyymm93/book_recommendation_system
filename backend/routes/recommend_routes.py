from __future__ import annotations

import json
from datetime import datetime

from flask import Blueprint, current_app, request

from config import Settings


recommend_bp = Blueprint("recommend", __name__)


@recommend_bp.route("/api/recommend", methods=["GET"])
def recommend():
    user_id = request.args.get("user_id", type=int)
    top_n = max(1, min(request.args.get("top_n", default=10, type=int), 50))
    diversity_strength = request.args.get("diversity_strength", default=1.0, type=float)
    diversity_strength = max(0.0, min(diversity_strength, 2.0))
    if not user_id:
        return {"code": 400, "message": "user_id is required", "data": []}, 400

    svc = current_app.extensions["recommender_service"]
    rows = svc.recommend_for_user(
        user_id=user_id,
        top_n=top_n,
        diversity_strength=diversity_strength,
    )
    return {"code": 0, "message": "success", "data": rows}


@recommend_bp.route("/api/model/status", methods=["GET"])
def model_status():
    model = current_app.extensions["model_service"]
    loaded = model.active_model != "none"
    if model.kgin_book_emb is not None:
        entity_count = int(model.kgin_book_emb.shape[0])
        book_count = len(model.kgin_book2idx)
    else:
        entity_count = 0
        book_count = 0

    return {
        "code": 0,
        "message": "success",
        "data": {
            "loaded": loaded,
            "model": model.active_model,
            "entity_count": entity_count,
            "book_embedding_count": book_count,
        },
    }


@recommend_bp.route("/api/model/metrics", methods=["GET"])
def model_metrics():
    metrics_path = Settings.ARTIFACT_ROOT / "eval_metrics.json"
    if not metrics_path.exists():
        return {
            "code": 0,
            "message": "metrics file not found",
            "data": {
                "loaded": False,
                "model": None,
                "eval_mode": None,
                "evaluated_users": 0,
                "hit_rate@10": None,
                "ndcg@10": None,
                "mrr": None,
                "metrics": {},
                "source_stats": {},
                "updated_at": None,
            },
        }

    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "code": 0,
            "message": "metrics file parse failed",
            "data": {
                "loaded": False,
                "model": None,
                "eval_mode": None,
                "evaluated_users": 0,
                "hit_rate@10": None,
                "ndcg@10": None,
                "mrr": None,
                "metrics": {},
                "source_stats": {},
                "updated_at": None,
            },
        }

    metrics = payload.get("metrics", {}) or {}
    cfg = payload.get("config", {}) or {}
    updated_at = datetime.fromtimestamp(metrics_path.stat().st_mtime).isoformat(timespec="seconds")
    selected_model = cfg.get("selected_model") or cfg.get("model_type")

    return {
        "code": 0,
        "message": "success",
        "data": {
            "loaded": True,
            "model": selected_model,
            "eval_mode": cfg.get("eval_mode"),
            "evaluated_users": int(payload.get("evaluated_users", 0) or 0),
            "hit_rate@10": metrics.get("hit_rate@10"),
            "ndcg@10": metrics.get("ndcg@10"),
            "mrr": metrics.get("mrr"),
            "metrics": metrics,
            "source_stats": payload.get("source_stats", {}) or {},
            "updated_at": updated_at,
        },
    }

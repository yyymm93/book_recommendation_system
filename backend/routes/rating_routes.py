from __future__ import annotations

from flask import Blueprint, current_app, request

from services.local_store import delete_user_rating, list_user_ratings, upsert_rating


rating_bp = Blueprint("rating", __name__)


@rating_bp.route("/api/rate", methods=["POST"])
def rate_book():
    payload = request.get_json(silent=True) or {}
    user_id = payload.get("user_id")
    book_id = payload.get("book_id")
    rating = payload.get("rating")

    if user_id is None or not book_id or rating is None:
        return {"code": 400, "message": "user_id, book_id, rating are required", "data": None}, 400

    try:
        rating_val = float(rating)
    except ValueError:
        return {"code": 400, "message": "rating must be numeric", "data": None}, 400

    if rating_val < 0.0 or rating_val > 5.0:
        return {"code": 400, "message": "rating must be in [0, 5]", "data": None}, 400

    upsert_rating(user_id=int(user_id), book_id=str(book_id), rating=rating_val)
    return {
        "code": 0,
        "message": "saved",
        "data": {"user_id": int(user_id), "book_id": str(book_id), "rating": rating_val},
    }


@rating_bp.route("/api/ratings", methods=["GET"])
def get_ratings():
    user_id = request.args.get("user_id", type=int)
    if not user_id:
        return {"code": 400, "message": "user_id is required", "data": []}, 400

    rows = list_user_ratings(user_id)
    if not rows:
        return {"code": 0, "message": "success", "data": []}

    neo = current_app.extensions["neo4j_service"]
    detail_rows = neo.books_by_ids([str(r["book_id"]) for r in rows])
    detail_map = {str(r["book_id"]): r for r in detail_rows}

    merged = []
    for row in rows:
        book_id = str(row["book_id"])
        detail = detail_map.get(book_id, {})
        merged.append(
            {
                "id": row["id"],
                "user_id": row["user_id"],
                "book_id": book_id,
                "rating": row["rating"],
                "title": detail.get("title", book_id),
                "author": detail.get("author", "Unknown"),
                "category": detail.get("category", "Unknown"),
            }
        )
    return {"code": 0, "message": "success", "data": merged}


@rating_bp.route("/api/rate", methods=["DELETE"])
def remove_rating():
    user_id = request.args.get("user_id", type=int)
    book_id = (request.args.get("book_id") or "").strip()
    if not user_id or not book_id:
        return {"code": 400, "message": "user_id and book_id are required", "data": None}, 400

    deleted = delete_user_rating(user_id=user_id, book_id=book_id)
    return {"code": 0, "message": "deleted" if deleted else "not found", "data": {"deleted": deleted}}

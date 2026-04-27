from __future__ import annotations

from flask import Blueprint, current_app, request


graph_bp = Blueprint("graph", __name__)


@graph_bp.route("/api/graph/subgraph", methods=["GET"])
def subgraph():
    book_id = (request.args.get("book_id") or "").strip()
    if not book_id:
        return {"code": 400, "message": "book_id is required", "data": []}, 400

    neo = current_app.extensions["neo4j_service"]
    rows = neo.book_subgraph(book_id)
    return {"code": 0, "message": "success", "data": rows}


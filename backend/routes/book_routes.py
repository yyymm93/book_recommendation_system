from __future__ import annotations

from flask import Blueprint, current_app, request


book_bp = Blueprint("book", __name__)


@book_bp.route("/api/books", methods=["GET"])
def list_books():
    limit = max(1, min(request.args.get("limit", default=20, type=int), 100))
    keyword = (request.args.get("keyword") or "").strip()
    neo = current_app.extensions["neo4j_service"]
    rows = neo.list_books(limit=limit, keyword=keyword)
    return {"code": 0, "message": "success", "data": rows}


@book_bp.route("/api/books/search", methods=["GET"])
def search_books():
    limit = max(1, min(request.args.get("limit", default=20, type=int), 100))
    keyword = (request.args.get("keyword") or "").strip()
    neo = current_app.extensions["neo4j_service"]
    rows = neo.list_books(limit=limit, keyword=keyword)
    return {"code": 0, "message": "success", "data": rows}

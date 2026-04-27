from __future__ import annotations

from flask import Blueprint, request

from services.local_store import login_or_register


auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/api/auth/login", methods=["POST"])
def login():
    payload = request.get_json(silent=True) or {}
    username = payload.get("username", "")
    password = payload.get("password", "")
    user = login_or_register(username=username, password=password)
    if not user:
        return {"code": 401, "message": "invalid username/password", "data": None}, 401
    return {"code": 0, "message": "success", "data": user}


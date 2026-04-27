from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from config import Settings


def init_local_db() -> None:
    Settings.DATA_ROOT.mkdir(parents=True, exist_ok=True)
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ratings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                book_id TEXT NOT NULL,
                rating REAL NOT NULL,
                UNIQUE(user_id, book_id)
            )
            """
        )
        conn.commit()


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    db_path: Path = Settings.DB_PATH
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def login_or_register(username: str, password: str) -> dict | None:
    username = (username or "").strip()
    password = (password or "").strip()
    if not username or not password:
        return None

    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, username, password FROM users WHERE username = ?",
            (username,),
        ).fetchone()

        if row:
            if row["password"] != password:
                return None
            return {"user_id": row["id"], "username": row["username"]}

        cur = conn.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (username, password),
        )
        conn.commit()
        return {"user_id": cur.lastrowid, "username": username}


def upsert_rating(user_id: int, book_id: str, rating: float) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO ratings (user_id, book_id, rating)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id, book_id)
            DO UPDATE SET rating = excluded.rating
            """,
            (int(user_id), str(book_id), float(rating)),
        )
        conn.commit()


def get_user_rated_book_ids(user_id: int) -> list[str]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT book_id FROM ratings WHERE user_id = ?",
            (int(user_id),),
        ).fetchall()
    return [str(r["book_id"]) for r in rows]


def list_user_ratings(user_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, user_id, book_id, rating
            FROM ratings
            WHERE user_id = ?
            ORDER BY id DESC
            """,
            (int(user_id),),
        ).fetchall()
    return [dict(r) for r in rows]


def delete_user_rating(user_id: int, book_id: str) -> bool:
    with get_conn() as conn:
        cur = conn.execute(
            "DELETE FROM ratings WHERE user_id = ? AND book_id = ?",
            (int(user_id), str(book_id)),
        )
        conn.commit()
        return cur.rowcount > 0

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"


def parse_entity(token: str) -> tuple[str, str]:
    prefix, value = token.split(":", 1)
    if prefix == "book":
        return "Book", value
    if prefix == "author":
        return "Author", value
    if prefix == "category":
        return "Category", value
    if prefix == "tag":
        return "Tag", value
    return "Entity", value


def _safe_str(v: Any) -> str:
    if pd.isna(v):
        return ""
    return str(v).strip()


def _safe_int(v: Any) -> int:
    if pd.isna(v):
        return 0
    try:
        return int(float(v))
    except Exception:
        return 0


def _safe_float(v: Any) -> float:
    if pd.isna(v):
        return 0.0
    try:
        return float(v)
    except Exception:
        return 0.0


def _book_params(row: pd.Series) -> dict[str, Any]:
    return {
        "book_id": _safe_str(row.get("book_id")),
        "goodreads_book_id": _safe_str(row.get("goodreads_book_id")),
        "best_book_id": _safe_str(row.get("best_book_id")),
        "work_id": _safe_str(row.get("work_id")),
        "title": _safe_str(row.get("title")) or "Unknown",
        "original_title": _safe_str(row.get("original_title")),
        "authors": _safe_str(row.get("authors")) or "Unknown",
        "year": _safe_int(row.get("original_publication_year")),
        "language_code": _safe_str(row.get("language_code")),
        "isbn": _safe_str(row.get("isbn")),
        "isbn13": _safe_str(row.get("isbn13")),
        "books_count": _safe_int(row.get("books_count")),
        "average_rating": _safe_float(row.get("average_rating")),
        "ratings_count": _safe_int(row.get("ratings_count")),
        "work_ratings_count": _safe_int(row.get("work_ratings_count")),
        "work_text_reviews_count": _safe_int(row.get("work_text_reviews_count")),
        "ratings_1": _safe_int(row.get("ratings_1")),
        "ratings_2": _safe_int(row.get("ratings_2")),
        "ratings_3": _safe_int(row.get("ratings_3")),
        "ratings_4": _safe_int(row.get("ratings_4")),
        "ratings_5": _safe_int(row.get("ratings_5")),
        "image_url": _safe_str(row.get("image_url")),
        "small_image_url": _safe_str(row.get("small_image_url")),
    }


def main() -> None:
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "12345678")

    triples = pd.read_csv(PROCESSED / "triples.csv")
    books = pd.read_csv(PROCESSED / "books_clean.csv")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        session.run(
            "CREATE CONSTRAINT book_book_id IF NOT EXISTS FOR (b:Book) REQUIRE b.book_id IS UNIQUE"
        )
        session.run(
            "CREATE CONSTRAINT author_name IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE"
        )
        session.run(
            "CREATE CONSTRAINT category_name IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE"
        )
        session.run(
            "CREATE CONSTRAINT tag_name IF NOT EXISTS FOR (t:Tag) REQUIRE t.name IS UNIQUE"
        )

        for _, row in books.iterrows():
            session.run(
                """
                MERGE (b:Book {book_id: $book_id})
                SET b.title = $title,
                    b.original_title = $original_title,
                    b.authors = $authors,
                    b.year = $year,
                    b.language_code = $language_code,
                    b.isbn = $isbn,
                    b.isbn13 = $isbn13,
                    b.goodreads_book_id = $goodreads_book_id,
                    b.best_book_id = $best_book_id,
                    b.work_id = $work_id,
                    b.books_count = $books_count,
                    b.average_rating = $average_rating,
                    b.ratings_count = $ratings_count,
                    b.work_ratings_count = $work_ratings_count,
                    b.work_text_reviews_count = $work_text_reviews_count,
                    b.ratings_1 = $ratings_1,
                    b.ratings_2 = $ratings_2,
                    b.ratings_3 = $ratings_3,
                    b.ratings_4 = $ratings_4,
                    b.ratings_5 = $ratings_5,
                    b.image_url = $image_url,
                    b.small_image_url = $small_image_url
                """,
                _book_params(row),
            )

        for row in triples.itertuples(index=False):
            h_label, h_value = parse_entity(str(row.head))
            t_label, t_value = parse_entity(str(row.tail))
            rel = str(row.relation)

            if h_label == "Book":
                session.run(
                    f"""
                    MERGE (h:Book {{book_id: $h_book_id}})
                    ON CREATE SET h.title = $h_book_id
                    MERGE (t:{t_label} {{name: $t_name}})
                    MERGE (h)-[:{rel}]->(t)
                    """,
                    {"h_book_id": h_value, "t_name": t_value},
                )
            else:
                session.run(
                    f"""
                    MERGE (h:{h_label} {{name: $h_name}})
                    MERGE (t:{t_label} {{name: $t_name}})
                    MERGE (h)-[:{rel}]->(t)
                    """,
                    {"h_name": h_value, "t_name": t_value},
                )

    driver.close()
    print(f"[ok] Neo4j graph imported. books={len(books)}, triples={len(triples)}")


if __name__ == "__main__":
    main()

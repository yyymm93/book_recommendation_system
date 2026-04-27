from __future__ import annotations

from typing import Any

from neo4j import GraphDatabase

from config import Settings


class Neo4jService:
    def __init__(self) -> None:
        self.driver = GraphDatabase.driver(
            Settings.NEO4J_URI,
            auth=(Settings.NEO4J_USER, Settings.NEO4J_PASSWORD),
        )

    def close(self) -> None:
        self.driver.close()

    def run(self, query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        with self.driver.session() as session:
            result = session.run(query, params or {})
            return [r.data() for r in result]

    def list_books(self, limit: int = 20, keyword: str = "") -> list[dict[str, Any]]:
        rows = self.run(
            """
            MATCH (b:Book)
            OPTIONAL MATCH (b)-[:WRITTEN_BY]->(a:Author)
            OPTIONAL MATCH (b)-[:BELONGS_TO]->(c:Category)
            OPTIONAL MATCH (b)-[:HAS_TAG]->(t:Tag)
            WITH b,
                 collect(DISTINCT a.name) AS authors,
                 collect(DISTINCT c.name) AS categories,
                 collect(DISTINCT t.name) AS tags
            WHERE $keyword = ""
               OR toLower(coalesce(b.title, b.book_name, "")) CONTAINS toLower($keyword)
               OR toLower(coalesce(b.original_title, "")) CONTAINS toLower($keyword)
               OR any(name IN authors WHERE toLower(name) CONTAINS toLower($keyword))
               OR any(name IN tags WHERE toLower(name) CONTAINS toLower($keyword))
            RETURN b.book_id AS book_id,
                   coalesce(b.title, b.book_name) AS title,
                   coalesce(head(authors), 'Unknown') AS author,
                   coalesce(head(categories), head(tags), 'Unknown') AS category,
                   coalesce(b.average_rating, 0.0) AS average_rating,
                   coalesce(b.ratings_count, 0) AS ratings_count,
                   coalesce(b.language_code, '') AS language_code
            ORDER BY title
            LIMIT $limit
            """,
            {"limit": int(limit), "keyword": (keyword or "").strip()},
        )
        return rows

    def books_by_ids(self, book_ids: list[str]) -> list[dict[str, Any]]:
        if not book_ids:
            return []

        rows = self.run(
            """
            UNWIND $book_ids AS bid
            MATCH (b:Book {book_id: bid})
            OPTIONAL MATCH (b)-[:WRITTEN_BY]->(a:Author)
            OPTIONAL MATCH (b)-[:BELONGS_TO]->(c:Category)
            OPTIONAL MATCH (b)-[:HAS_TAG]->(t:Tag)
            WITH b, collect(DISTINCT a.name) AS authors, collect(DISTINCT c.name) AS categories, collect(DISTINCT t.name) AS tags
            RETURN b.book_id AS book_id,
                   coalesce(b.title, b.book_name, b.book_id) AS title,
                   coalesce(head(authors), 'Unknown') AS author,
                   coalesce(head(categories), head(tags), 'Unknown') AS category,
                   coalesce(b.average_rating, 0.0) AS average_rating,
                   coalesce(b.ratings_count, 0) AS ratings_count,
                   coalesce(b.language_code, '') AS language_code
            """,
            {"book_ids": [str(x) for x in book_ids]},
        )
        return rows

    def graph_candidate_links(
        self,
        seed_book_ids: list[str],
        excluded_book_ids: list[str],
        per_reason_limit: int = 50,
    ) -> list[dict[str, Any]]:
        if not seed_book_ids:
            return []

        rows = self.run(
            """
            UNWIND $seed_book_ids AS seed_id
            MATCH (seed:Book {book_id: seed_id})
            WITH DISTINCT seed
            CALL (seed) {
              MATCH (seed)-[:WRITTEN_BY]->(a:Author)<-[:WRITTEN_BY]-(candidate:Book)
              WHERE candidate <> seed
              RETURN candidate.book_id AS candidate_id,
                     'AUTHOR' AS reason_type,
                     a.name AS reason_value,
                     '' AS bridge_title
              LIMIT $per_reason_limit

              UNION

              MATCH (seed)-[:BELONGS_TO]->(c:Category)<-[:BELONGS_TO]-(candidate:Book)
              WHERE candidate <> seed
              RETURN candidate.book_id AS candidate_id,
                     'CATEGORY' AS reason_type,
                     c.name AS reason_value,
                     '' AS bridge_title
              LIMIT $per_reason_limit

              UNION

              MATCH (seed)-[:HAS_TAG]->(t:Tag)<-[:HAS_TAG]-(candidate:Book)
              WHERE candidate <> seed
              RETURN candidate.book_id AS candidate_id,
                     'TAG' AS reason_type,
                     t.name AS reason_value,
                     '' AS bridge_title
              LIMIT $per_reason_limit

              UNION

              MATCH (seed)-[:WRITTEN_BY]->(:Author)<-[:WRITTEN_BY]-(mid:Book)-[:HAS_TAG]->(t:Tag)<-[:HAS_TAG]-(candidate:Book)
              WHERE seed <> mid AND seed <> candidate AND mid <> candidate
              RETURN candidate.book_id AS candidate_id,
                     'TWO_HOP' AS reason_type,
                     t.name AS reason_value,
                     coalesce(mid.title, mid.book_name, mid.book_id) AS bridge_title
              LIMIT $per_reason_limit
            }
            WITH seed, candidate_id, reason_type, reason_value, bridge_title
            WHERE candidate_id IS NOT NULL
              AND candidate_id <> seed.book_id
              AND NOT candidate_id IN $excluded_book_ids
            RETURN DISTINCT seed.book_id AS seed_book_id,
                            coalesce(seed.title, seed.book_name, seed.book_id) AS seed_title,
                            candidate_id,
                            reason_type,
                            coalesce(reason_value, '') AS reason_value,
                            coalesce(bridge_title, '') AS bridge_title
            """,
            {
                "seed_book_ids": [str(x) for x in seed_book_ids],
                "excluded_book_ids": [str(x) for x in excluded_book_ids],
                "per_reason_limit": int(per_reason_limit),
            },
        )
        return rows

    def book_subgraph(self, book_id: str) -> list[dict[str, Any]]:
        return self.run(
            """
            MATCH (b:Book {book_id: $book_id})
            OPTIONAL MATCH (b)-[r1]->(n1)
            OPTIONAL MATCH (n2)-[r2]->(b)
            WITH b,
                 collect(DISTINCT {
                   source_id: b.book_id,
                   source_name: coalesce(b.title, b.book_name, b.book_id),
                   rel: type(r1),
                   target_type: labels(n1)[0],
                   target_name: coalesce(n1.name, n1.title, n1.book_name)
                 }) + collect(DISTINCT {
                   source_id: coalesce(n2.book_id, n2.name, 'node'),
                   source_name: coalesce(n2.title, n2.book_name, n2.name),
                   rel: type(r2),
                   target_type: 'Book',
                   target_name: coalesce(b.title, b.book_name, b.book_id)
                 }) AS links
            UNWIND links AS link
            WITH link
            WHERE link.rel IS NOT NULL
            RETURN link.source_id AS source_id,
                   link.source_name AS source_name,
                   link.rel AS rel,
                   link.target_type AS target_type,
                   link.target_name AS target_name
            """,
            {"book_id": str(book_id)},
        )

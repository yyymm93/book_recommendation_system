from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW = PROJECT_ROOT / "data" / "raw"
OUT = PROJECT_ROOT / "data" / "processed"

BOOK_KEEP_COLUMNS = [
    "id",
    "book_id",
    "goodreads_book_id",
    "best_book_id",
    "work_id",
    "books_count",
    "isbn",
    "isbn13",
    "authors",
    "original_publication_year",
    "original_title",
    "title",
    "language_code",
    "average_rating",
    "ratings_count",
    "work_ratings_count",
    "work_text_reviews_count",
    "ratings_1",
    "ratings_2",
    "ratings_3",
    "ratings_4",
    "ratings_5",
    "image_url",
    "small_image_url",
]

SHELF_TAG_BLACKLIST = {
    "to-read",
    "to read",
    "read",
    "currently-reading",
    "currently reading",
    "owned",
    "favourites",
    "favorites",
    "default",
    "wish-list",
    "wishlist",
    "library",
    "ebooks",
    "ebook",
    "kindle",
    "books-i-own",
    "my-books",
    "my book",
    "i-own",
}


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _to_int_col(df: pd.DataFrame, col: str, default: int = 0) -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df))
    return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(int)


def _to_float_col(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df))
    return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)


def _clean_user_book_df(df: pd.DataFrame, score_col: str | None = None) -> pd.DataFrame:
    out = df.copy()
    out["user_id"] = pd.to_numeric(out["user_id"], errors="coerce")
    out["book_id"] = pd.to_numeric(out["book_id"], errors="coerce")
    if score_col:
        out[score_col] = pd.to_numeric(out[score_col], errors="coerce")
        out = out.dropna(subset=["user_id", "book_id", score_col])
    else:
        out = out.dropna(subset=["user_id", "book_id"])
    out["user_id"] = out["user_id"].astype(int)
    out["book_id"] = out["book_id"].astype(int).astype(str)
    if score_col:
        out[score_col] = out[score_col].astype(float)
    return out


def _normalize_tag_name(tag: object) -> str:
    if pd.isna(tag):
        return ""
    s = str(tag).strip().lower()
    if not s:
        return ""

    s = s.replace("_", " ")
    s = re.sub(r"^[^\w\u4e00-\u9fff]+|[^\w\u4e00-\u9fff]+$", "", s)
    s = re.sub(r"[-]{2,}", "-", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _is_valid_tag_name(tag: str) -> tuple[bool, str]:
    if not tag:
        return False, "empty"

    invalid_literals = {
        "#name?",
        "nan",
        "none",
        "null",
        "n/a",
        "na",
        "unknown",
        "unk",
        "?",
        "-",
    }
    if tag in invalid_literals:
        return False, "invalid_literal"

    normalized_cmp = tag.replace("-", " ").strip()
    if tag in SHELF_TAG_BLACKLIST or normalized_cmp in SHELF_TAG_BLACKLIST:
        return False, "shelf_tag"

    if re.fullmatch(r"\d+", tag):
        return False, "numeric_only"
    if not re.search(r"[a-z\u4e00-\u9fff]", tag):
        return False, "no_alpha_or_cjk"
    if len(tag) < 2:
        return False, "too_short"
    return True, "ok"


def _to_numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce").dropna().astype(int)


def _overlap_ratio(base_set: set[int], sample_series: pd.Series) -> float:
    if sample_series.empty:
        return 0.0
    vals = set(sample_series.dropna().astype(int).tolist())
    if not vals:
        return 0.0
    return len(vals & base_set) / max(1, len(vals))


def _select_book_key_column(
    books: pd.DataFrame,
    ratings_raw: pd.DataFrame,
    to_read_raw: pd.DataFrame,
    book_tags_raw: pd.DataFrame,
) -> tuple[str, dict[str, str]]:
    candidates = [c for c in ["id", "book_id", "goodreads_book_id"] if c in books.columns]
    if not candidates:
        raise ValueError("books.csv must contain at least one of: id, book_id, goodreads_book_id")

    ratings_col = _first_existing_column(ratings_raw, ["book_id", "goodreads_book_id", "id"])
    to_read_col = _first_existing_column(to_read_raw, ["book_id", "goodreads_book_id", "id"]) if not to_read_raw.empty else None
    tags_col = _first_existing_column(book_tags_raw, ["book_id", "goodreads_book_id", "id"])

    source_cols = {
        "ratings": ratings_col or "",
        "to_read": to_read_col or "",
        "book_tags": tags_col or "",
    }

    if not ratings_col:
        raise ValueError("ratings.csv must contain one of: book_id, goodreads_book_id, id")
    if not tags_col:
        raise ValueError("book_tags.csv must contain one of: book_id, goodreads_book_id, id")

    ratings_series = _to_numeric_series(ratings_raw, ratings_col)
    to_read_series = _to_numeric_series(to_read_raw, to_read_col) if to_read_col else pd.Series(dtype=int)
    tags_series = _to_numeric_series(book_tags_raw, tags_col)

    weights = {"ratings": 0.6, "book_tags": 0.3, "to_read": 0.1}
    best_col = candidates[0]
    best_score = -1.0
    for col in candidates:
        base_set = set(_to_numeric_series(books, col).tolist())
        score = 0.0
        score += weights["ratings"] * _overlap_ratio(base_set, ratings_series)
        score += weights["book_tags"] * _overlap_ratio(base_set, tags_series)
        if to_read_col:
            score += weights["to_read"] * _overlap_ratio(base_set, to_read_series)
        if score > best_score:
            best_score = score
            best_col = col

    return best_col, source_cols


def _build_key_maps(books: pd.DataFrame) -> dict[str, dict[int, str]]:
    maps: dict[str, dict[int, str]] = {}
    for col in ["id", "book_id", "goodreads_book_id"]:
        if col not in books.columns:
            continue
        tmp = pd.DataFrame(
            {
                "raw": pd.to_numeric(books[col], errors="coerce"),
                "canonical": books["book_id"],
            }
        ).dropna(subset=["raw"])
        tmp["raw"] = tmp["raw"].astype(int)
        tmp = tmp.drop_duplicates(subset=["raw"], keep="first")
        maps[col] = dict(zip(tmp["raw"].tolist(), tmp["canonical"].tolist()))
    return maps


def _map_external_book_ids(df: pd.DataFrame, src_col: str, key_maps: dict[str, dict[int, str]]) -> pd.Series:
    raw = pd.to_numeric(df[src_col], errors="coerce")
    mapped = pd.Series([None] * len(df), index=df.index, dtype=object)

    if src_col in key_maps:
        mapped = raw.map(key_maps[src_col])

    if mapped.isna().any():
        remain_idx = mapped[mapped.isna()].index
        for col, m in key_maps.items():
            if col == src_col:
                continue
            if len(remain_idx) == 0:
                break
            sub_raw = raw.loc[remain_idx]
            sub_mapped = sub_raw.map(m)
            mapped.loc[remain_idx] = mapped.loc[remain_idx].fillna(sub_mapped)
            remain_idx = mapped[mapped.isna()].index

    return mapped


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    books_raw = pd.read_csv(RAW / "books.csv")
    ratings_raw = pd.read_csv(RAW / "ratings.csv")
    to_read_path = RAW / "to_read.csv"
    to_read_raw = pd.read_csv(to_read_path) if to_read_path.exists() else pd.DataFrame(columns=["user_id", "book_id"])
    book_tags_raw = pd.read_csv(RAW / "book_tags.csv")
    tags_raw = pd.read_csv(RAW / "tags.csv")

    keep_cols = [c for c in BOOK_KEEP_COLUMNS if c in books_raw.columns]
    books = books_raw[keep_cols].copy()

    # In some Goodbooks variants, large goodreads id is stored in book_id while id is 1..10000 key.
    if "goodreads_book_id" not in books.columns and "book_id" in books.columns and "id" in books.columns:
        books["goodreads_book_id"] = books["book_id"]

    book_key_col, source_cols = _select_book_key_column(books, ratings_raw, to_read_raw, book_tags_raw)

    books["book_id"] = pd.to_numeric(books[book_key_col], errors="coerce")
    books = books.dropna(subset=["book_id"]).copy()
    books["book_id"] = books["book_id"].astype(int).astype(str)
    books = books.drop_duplicates(subset=["book_id"], keep="first")

    title_col = _first_existing_column(books, ["title", "original_title"])
    if title_col is None:
        raise ValueError("books.csv must contain title or original_title")

    if "authors" not in books.columns:
        books["authors"] = "Unknown"
    books["title"] = books[title_col].fillna("").astype(str).str.strip()
    if "original_title" not in books.columns:
        books["original_title"] = books["title"]
    books["original_title"] = books["original_title"].fillna("").astype(str).str.strip()
    books.loc[books["title"] == "", "title"] = books["original_title"]
    books.loc[books["title"] == "", "title"] = "Unknown"
    books["authors"] = books["authors"].fillna("").astype(str).str.strip()
    books.loc[books["authors"] == "", "authors"] = "Unknown"

    books["original_publication_year"] = _to_int_col(books, "original_publication_year", 0)
    books["average_rating"] = _to_float_col(books, "average_rating", 0.0)
    books["ratings_count"] = _to_int_col(books, "ratings_count", 0)
    books["work_ratings_count"] = _to_int_col(books, "work_ratings_count", 0)
    books["work_text_reviews_count"] = _to_int_col(books, "work_text_reviews_count", 0)
    books["books_count"] = _to_int_col(books, "books_count", 0)
    books["ratings_1"] = _to_int_col(books, "ratings_1", 0)
    books["ratings_2"] = _to_int_col(books, "ratings_2", 0)
    books["ratings_3"] = _to_int_col(books, "ratings_3", 0)
    books["ratings_4"] = _to_int_col(books, "ratings_4", 0)
    books["ratings_5"] = _to_int_col(books, "ratings_5", 0)

    for col in ["language_code", "isbn", "isbn13", "image_url", "small_image_url", "goodreads_book_id", "best_book_id", "work_id", "id"]:
        if col not in books.columns:
            books[col] = ""
        books[col] = books[col].fillna("").astype(str).str.strip()

    key_maps = _build_key_maps(books)

    # Build triples.
    triples: list[tuple[str, str, str]] = []
    for row in books.itertuples(index=False):
        b = f"book:{row.book_id}"
        first_author = str(row.authors).split(",")[0].strip() or "Unknown"
        triples.append((b, "WRITTEN_BY", f"author:{first_author}"))

    if "tag_id" not in tags_raw.columns:
        raise ValueError("tags.csv must contain column: tag_id")
    tag_name_col = _first_existing_column(tags_raw, ["tag_name", "name"])
    if tag_name_col is None:
        raise ValueError("tags.csv must contain tag_name (or name)")

    tags_clean = tags_raw[["tag_id", tag_name_col]].rename(columns={tag_name_col: "tag_name"}).copy()
    tags_clean["tag_id"] = pd.to_numeric(tags_clean["tag_id"], errors="coerce")
    tags_clean = tags_clean.dropna(subset=["tag_id"])
    tags_clean["tag_id"] = tags_clean["tag_id"].astype(int)
    tags_clean["tag_name_raw"] = tags_clean["tag_name"]
    tags_clean["tag_name"] = tags_clean["tag_name"].map(_normalize_tag_name)
    tags_clean[["is_valid", "invalid_reason"]] = tags_clean["tag_name"].apply(
        lambda x: pd.Series(_is_valid_tag_name(x))
    )
    valid_tags = tags_clean[tags_clean["is_valid"]].copy()
    tag_map = dict(valid_tags[["tag_id", "tag_name"]].values.tolist())

    tags_book_col = source_cols["book_tags"]
    if not tags_book_col:
        raise ValueError("book_tags.csv must contain one of: book_id, goodreads_book_id, id")
    for col in [tags_book_col, "tag_id", "count"]:
        if col not in book_tags_raw.columns:
            raise ValueError(f"book_tags.csv must contain column: {col}")

    merged = book_tags_raw[[tags_book_col, "tag_id", "count"]].copy()
    merged["book_id"] = _map_external_book_ids(merged, tags_book_col, key_maps)
    merged["tag_id"] = pd.to_numeric(merged["tag_id"], errors="coerce")
    merged["count"] = pd.to_numeric(merged["count"], errors="coerce").fillna(0).astype(int)
    merged = merged.dropna(subset=["book_id", "tag_id"])
    merged["tag_id"] = merged["tag_id"].astype(int)
    merged = merged[merged["tag_id"].isin(set(valid_tags["tag_id"].tolist()))]
    merged = merged.sort_values("count", ascending=False)

    top_tags = merged.groupby("book_id").head(5)
    top_category = merged.groupby("book_id").head(1)

    for row in top_tags.itertuples(index=False):
        tag_name = str(tag_map.get(int(row.tag_id), "unknown_tag")).strip() or "unknown_tag"
        triples.append((f"book:{row.book_id}", "HAS_TAG", f"tag:{tag_name}"))

    for row in top_category.itertuples(index=False):
        tag_name = str(tag_map.get(int(row.tag_id), "unknown_category")).strip() or "unknown_category"
        triples.append((f"book:{row.book_id}", "BELONGS_TO", f"category:{tag_name}"))

    triples_df = pd.DataFrame(triples, columns=["head", "relation", "tail"]).drop_duplicates()
    triples_df.to_csv(OUT / "triples.csv", index=False)

    ratings_book_col = source_cols["ratings"]
    for col in ["user_id", ratings_book_col, "rating"]:
        if col not in ratings_raw.columns:
            raise ValueError(f"ratings.csv must contain column: {col}")
    ratings = ratings_raw[["user_id", ratings_book_col, "rating"]].copy().rename(columns={ratings_book_col: "book_id"})
    ratings["book_id"] = _map_external_book_ids(ratings, "book_id", key_maps)
    ratings = ratings.dropna(subset=["book_id"])
    ratings = _clean_user_book_df(ratings[["user_id", "book_id", "rating"]], score_col="rating")
    ratings = ratings[ratings["book_id"].isin(books["book_id"])]
    ratings.to_csv(OUT / "user_book_ratings.csv", index=False)

    to_read = pd.DataFrame(columns=["user_id", "book_id"])
    if not to_read_raw.empty:
        to_read_book_col = source_cols["to_read"] or "book_id"
        for col in ["user_id", to_read_book_col]:
            if col not in to_read_raw.columns:
                raise ValueError(f"to_read.csv must contain column: {col}")
        to_read = to_read_raw[["user_id", to_read_book_col]].copy().rename(columns={to_read_book_col: "book_id"})
        to_read["book_id"] = _map_external_book_ids(to_read, "book_id", key_maps)
        to_read = to_read.dropna(subset=["book_id"])
        to_read = _clean_user_book_df(to_read[["user_id", "book_id"]])
        to_read = to_read[to_read["book_id"].isin(books["book_id"])]
    to_read.to_csv(OUT / "user_book_to_read.csv", index=False)

    explicit = ratings.copy()
    explicit["score"] = (explicit["rating"] / 5.0).clip(lower=0.0, upper=1.0)
    explicit["source"] = "rating"

    implicit = to_read.copy()
    if not implicit.empty:
        implicit["score"] = 0.8
        implicit["source"] = "to_read"
    else:
        implicit["score"] = pd.Series(dtype=float)
        implicit["source"] = pd.Series(dtype=str)

    feedback = pd.concat(
        [
            explicit[["user_id", "book_id", "score", "source"]],
            implicit[["user_id", "book_id", "score", "source"]],
        ],
        ignore_index=True,
    )
    if not feedback.empty:
        feedback = (
            feedback.sort_values("score", ascending=False)
            .drop_duplicates(subset=["user_id", "book_id"], keep="first")
            .reset_index(drop=True)
        )
    feedback.to_csv(OUT / "user_book_feedback.csv", index=False)

    books.to_csv(OUT / "books_clean.csv", index=False)
    valid_tags[["tag_id", "tag_name"]].to_csv(OUT / "tags_clean.csv", index=False)
    tags_clean[~tags_clean["is_valid"]][["tag_id", "tag_name_raw", "tag_name", "invalid_reason"]].to_csv(
        OUT / "tags_filtered_out.csv",
        index=False,
    )

    print(f"[ok] book key selected from books.csv: {book_key_col}")
    print(f"[ok] source cols: {source_cols}")
    print(f"[ok] books_raw: {len(books_raw)} -> books_clean: {len(books)}")
    print(f"[ok] triples: {len(triples_df)}")
    print(f"[ok] ratings_clean: {len(ratings)}")
    print(f"[ok] to_read_clean: {len(to_read)}")
    print(f"[ok] feedback_merged: {len(feedback)}")
    print(f"[ok] tags_valid: {len(valid_tags)} / tags_total: {len(tags_clean)}")
    print(f"[ok] books_clean columns: {list(books.columns)}")


if __name__ == "__main__":
    main()

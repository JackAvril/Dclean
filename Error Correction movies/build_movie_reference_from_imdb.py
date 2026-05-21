"""
Build movie_reference.csv from IMDb official non-commercial datasets.

Purpose
-------
Create an external movie metadata reference file for Movies repair candidate generation,
without using movies_clean.csv.

Input:
    movies_dirty.csv
    IMDb TSV files in imdb_dir:
        title.basics.tsv.gz
        title.ratings.tsv.gz
        title.crew.tsv.gz
        name.basics.tsv.gz

Output:
    movie_reference.csv
    movie_reference_imdb_match_diagnosis.csv

Download IMDb data:
    mkdir -p /mnt/mydata/dq/projects/Splittree/movie_reference_data
    cd /mnt/mydata/dq/projects/Splittree/movie_reference_data
    wget https://datasets.imdbws.com/title.basics.tsv.gz
    wget https://datasets.imdbws.com/title.ratings.tsv.gz
    wget https://datasets.imdbws.com/title.crew.tsv.gz
    wget https://datasets.imdbws.com/name.basics.tsv.gz

Run:
    cd /mnt/mydata/dq/projects/Splittree/test/Error\\ Correction\\ movies

    python build_movie_reference_from_imdb.py \
      --dirty_file /mnt/mydata/dq/projects/Splittree/movies/movies_dirty.csv \
      --imdb_dir /mnt/mydata/dq/projects/Splittree/movie_reference_data \
      --output movie_reference.csv \
      --diagnosis_output movie_reference_imdb_match_diagnosis.csv

Notes
-----
1. This script does NOT read movies_clean.csv.
2. It uses only external IMDb non-commercial datasets plus dirty table keys.
3. Matching is conservative:
   - normalized Name + Year
   - slug extracted from dirty Id + Year
   - optional title-year nearby match when exact year fails
4. If multiple IMDb titles match the same key, it chooses the highest-rated / most-voted candidate,
   but marks ambiguity in diagnosis.
"""

import argparse
import gzip
import os
import re
import unicodedata
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


EMPTY_TOKENS = {"", "empty", "null", "none", "nan", "na", "n/a", "missing", "{null}"}

TARGET_REF_COLUMNS = [
    "ref_key",
    "ref_name",
    "ref_year",
    "ref_release_date",
    "ref_country",
    "ref_imdb_id",
    "ref_description",
    "ref_duration",
    "ref_rating_value",
    "ref_director",
    "ref_creator",
    "ref_genre",
    "ref_source",
    "ref_match_confidence",
    "ref_support_count",
    "ref_total_count",
]


# ============================================================
# Basic helpers
# ============================================================

def norm_text(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()


def canonical_empty_text(x: Any) -> str:
    s = norm_text(x)
    return "empty" if s.lower() in EMPTY_TOKENS else s


def strip_accents(x: Any) -> str:
    s = norm_text(x)
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(ch)
    )


def normalize_title_key(x: Any) -> str:
    """
    Normalize movie title for matching:
    - remove accents
    - & -> and
    - remove punctuation
    - lowercase
    """
    s = strip_accents(x).lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # Some common article normalization: keep both caller-side and index-side exact same.
    return s


def parse_year(x: Any) -> Optional[int]:
    s = canonical_empty_text(x)
    if s.lower() == "empty":
        return None
    m = re.search(r"(18\d{2}|19\d{2}|20\d{2}|21\d{2})", s)
    if not m:
        return None
    y = int(m.group(1))
    return y if 1800 <= y <= 2100 else None


def parse_year_triple_middle(x: Any) -> Optional[int]:
    s = canonical_empty_text(x)
    years = [int(y) for y in re.findall(r"(18\d{2}|19\d{2}|20\d{2}|21\d{2})", s)]
    if len(years) == 3 and years[1] == years[0] + 1 and years[2] == years[1] + 1:
        return years[1]
    return None


def parse_slug_year(x: Any) -> Tuple[str, Optional[int]]:
    """
    Examples:
        immortals_2011 -> ("immortals", 2011)
        the_dark_knight_2008 -> ("the dark knight", 2008)
    """
    s = canonical_empty_text(x).strip()
    if s.lower() == "empty":
        return "", None

    m = re.search(r"(.+?)[_\-\s](18\d{2}|19\d{2}|20\d{2}|21\d{2})$", s)
    if not m:
        return "", None

    name = m.group(1).replace("_", " ").replace("-", " ")
    year = int(m.group(2))
    return normalize_title_key(name), year


def split_imdb_ids(x: Any) -> List[str]:
    s = norm_text(x)
    if not s or s == "\\N":
        return []
    return [p for p in s.split(",") if p and p != "\\N"]


def runtime_to_duration(runtime_minutes: Any) -> str:
    s = norm_text(runtime_minutes)
    if not s or s == "\\N":
        return ""
    try:
        m = int(float(s))
        if 1 <= m <= 1000:
            return f"{m} min"
    except Exception:
        pass
    return ""


def normalize_imdb_genres(x: Any) -> str:
    s = norm_text(x)
    if not s or s == "\\N":
        return ""
    parts = [p.strip() for p in s.split(",") if p.strip() and p.strip() != "\\N"]
    # IMDb uses "Sci-Fi"; keep it.
    return ",".join(parts)


def normalize_rating(x: Any) -> str:
    s = norm_text(x)
    if not s or s == "\\N":
        return ""
    try:
        v = float(s)
        if 0 <= v <= 10:
            return f"{v:.1f}"
    except Exception:
        pass
    return ""


def normalize_people_names(names: List[str]) -> str:
    out = []
    seen = set()
    for n in names:
        n = re.sub(r"\s+", " ", norm_text(n)).strip()
        if not n or n == "\\N":
            continue
        k = n.lower()
        if k not in seen:
            seen.add(k)
            out.append(n)
    return ",".join(out)


# ============================================================
# Load dirty keys
# ============================================================

def load_dirty_movie_keys(dirty_file: str) -> pd.DataFrame:
    df = pd.read_csv(dirty_file, encoding="utf-8-sig")

    required_any = {"Name", "Year"}
    if not required_any.issubset(df.columns):
        raise ValueError(f"Dirty file must contain at least Name and Year columns, got: {list(df.columns)}")

    rows = []
    for rid, r in df.iterrows():
        name = canonical_empty_text(r.get("Name", ""))
        name_key = normalize_title_key(name)
        year = parse_year_triple_middle(r.get("Year", "")) or parse_year(r.get("Year", ""))

        slug_key, slug_year = parse_slug_year(r.get("Id", "")) if "Id" in df.columns else ("", None)

        release_year = parse_year(r.get("Release Date", "")) if "Release Date" in df.columns else None

        keys = []
        if name_key and year:
            keys.append(("name_year", name_key, year))
        if slug_key and slug_year:
            keys.append(("slug_year", slug_key, slug_year))
        if name_key and release_year:
            keys.append(("name_release_year", name_key, release_year))

        rows.append({
            "row_id": int(rid),
            "dirty_name": name,
            "dirty_name_key": name_key,
            "dirty_year": year,
            "dirty_id": canonical_empty_text(r.get("Id", "")) if "Id" in df.columns else "",
            "dirty_slug_key": slug_key,
            "dirty_slug_year": slug_year,
            "dirty_release_year": release_year,
            "match_keys": keys,
        })

    return pd.DataFrame(rows)


# ============================================================
# Load IMDb datasets
# ============================================================

def imdb_path(imdb_dir: str, filename: str) -> str:
    p = os.path.join(imdb_dir, filename)
    if not os.path.exists(p):
        raise FileNotFoundError(f"IMDb file not found: {p}")
    return p


def read_imdb_tsv(path: str, usecols=None, dtype=str) -> pd.DataFrame:
    print(f"[INFO] reading IMDb file: {path}")
    return pd.read_csv(
        path,
        sep="\t",
        compression="gzip",
        dtype=dtype,
        usecols=usecols,
        na_values=["\\N"],
        keep_default_na=False,
        low_memory=False,
    )


def load_imdb_basics(imdb_dir: str) -> pd.DataFrame:
    path = imdb_path(imdb_dir, "title.basics.tsv.gz")
    usecols = [
        "tconst",
        "titleType",
        "primaryTitle",
        "originalTitle",
        "isAdult",
        "startYear",
        "runtimeMinutes",
        "genres",
    ]
    df = read_imdb_tsv(path, usecols=usecols)

    # Keep movie-like titles only.
    df["titleType"] = df["titleType"].astype(str)
    df = df[df["titleType"].isin(["movie", "tvMovie", "video"])].copy()

    if "isAdult" in df.columns:
        df = df[df["isAdult"].astype(str) != "1"].copy()

    df["startYear_int"] = pd.to_numeric(df["startYear"], errors="coerce")
    df = df[df["startYear_int"].notna()].copy()
    df["startYear_int"] = df["startYear_int"].astype(int)

    df["primary_key"] = df["primaryTitle"].map(normalize_title_key)
    df["original_key"] = df["originalTitle"].map(normalize_title_key)
    df["ref_duration"] = df["runtimeMinutes"].map(runtime_to_duration)
    df["ref_genre"] = df["genres"].map(normalize_imdb_genres)

    print(f"[INFO] IMDb basics movie-like rows: {len(df)}")
    return df


def load_imdb_ratings(imdb_dir: str) -> pd.DataFrame:
    path = imdb_path(imdb_dir, "title.ratings.tsv.gz")
    usecols = ["tconst", "averageRating", "numVotes"]
    df = read_imdb_tsv(path, usecols=usecols)
    df["ref_rating_value"] = df["averageRating"].map(normalize_rating)
    df["numVotes_int"] = pd.to_numeric(df["numVotes"], errors="coerce").fillna(0).astype(int)
    print(f"[INFO] IMDb ratings rows: {len(df)}")
    return df[["tconst", "ref_rating_value", "numVotes_int"]]


def load_imdb_crew(imdb_dir: str) -> pd.DataFrame:
    path = imdb_path(imdb_dir, "title.crew.tsv.gz")
    usecols = ["tconst", "directors", "writers"]
    df = read_imdb_tsv(path, usecols=usecols)
    print(f"[INFO] IMDb crew rows: {len(df)}")
    return df


def load_needed_names(imdb_dir: str, needed_nconst: set) -> Dict[str, str]:
    if not needed_nconst:
        return {}

    path = imdb_path(imdb_dir, "name.basics.tsv.gz")
    name_map = {}
    chunksize = 500000
    print(f"[INFO] reading IMDb names in chunks: {path}")
    for chunk in pd.read_csv(
        path,
        sep="\t",
        compression="gzip",
        dtype=str,
        usecols=["nconst", "primaryName"],
        na_values=["\\N"],
        keep_default_na=False,
        chunksize=chunksize,
        low_memory=False,
    ):
        sub = chunk[chunk["nconst"].isin(needed_nconst)]
        for _, r in sub.iterrows():
            name_map[str(r["nconst"])] = norm_text(r["primaryName"])
        if len(name_map) >= len(needed_nconst):
            break

    print(f"[INFO] resolved IMDb person names: {len(name_map)}/{len(needed_nconst)}")
    return name_map


# ============================================================
# Matching
# ============================================================

def build_title_year_index(basics: pd.DataFrame) -> Dict[Tuple[str, int], List[int]]:
    idx = defaultdict(list)

    for i, r in basics.iterrows():
        year = int(r["startYear_int"])
        pk = str(r["primary_key"])
        ok = str(r["original_key"])

        if pk:
            idx[(pk, year)].append(i)
        if ok and ok != pk:
            idx[(ok, year)].append(i)

    return dict(idx)


def candidate_rows_for_key(
    idx: Dict[Tuple[str, int], List[int]],
    title_key: str,
    year: int,
    allow_year_window: int = 1,
) -> List[int]:
    hits = []
    if (title_key, year) in idx:
        hits.extend(idx[(title_key, year)])

    # fallback nearby years for release year mismatch.
    if not hits and allow_year_window > 0:
        for dy in range(1, allow_year_window + 1):
            for y2 in [year - dy, year + dy]:
                if (title_key, y2) in idx:
                    hits.extend(idx[(title_key, y2)])

    return sorted(set(hits))


def choose_best_imdb_match(candidates: pd.DataFrame) -> Tuple[Optional[pd.Series], float, str]:
    """
    Choose best candidate by numVotes when multiple exact title/year matches exist.

    IMPORTANT:
    Keep the original IMDb dataframe index in column __imdb_row_index__.
    The previous version used reset_index(drop=True), so best.name became 0/1/2
    instead of the original basics/base row index. That caused movie_reference.csv
    to contain only one row even when diagnosis showed thousands of matches.
    """
    if candidates.empty:
        return None, 0.0, "no_match"

    c = candidates.copy()
    if "__imdb_row_index__" not in c.columns:
        c["__imdb_row_index__"] = c.index

    if "numVotes_int" not in c.columns:
        c["numVotes_int"] = 0

    c["_rating_sort"] = pd.to_numeric(c.get("ref_rating_value", 0), errors="coerce").fillna(0.0)
    c = c.sort_values(["numVotes_int", "_rating_sort"], ascending=[False, False])

    if len(c) == 1:
        return c.iloc[0], 0.98, "unique_match"

    top_votes = int(c.iloc[0].get("numVotes_int", 0))
    second_votes = int(c.iloc[1].get("numVotes_int", 0))

    # If the top candidate has far more votes, accept it with moderate confidence.
    if top_votes >= 100 and top_votes >= max(second_votes * 3, second_votes + 500):
        return c.iloc[0], 0.90, f"ambiguous_but_top_votes_dominant:{len(c)}"

    # Otherwise still choose top, but mark lower confidence.
    return c.iloc[0], 0.70, f"ambiguous_top_selected:{len(c)}"


def build_reference_from_imdb(dirty_keys: pd.DataFrame, basics: pd.DataFrame, ratings: pd.DataFrame, crew: pd.DataFrame, imdb_dir: str):
    # Merge basics + ratings + crew.
    # Preserve original row index so matched reference rows can be recovered after candidate sorting.
    basics = basics.copy()
    basics["__imdb_row_index__"] = basics.index
    base = basics.merge(ratings, on="tconst", how="left")
    base = base.merge(crew, on="tconst", how="left")
    base["ref_rating_value"] = base["ref_rating_value"].fillna("")
    base["numVotes_int"] = pd.to_numeric(base["numVotes_int"], errors="coerce").fillna(0).astype(int)

    idx = build_title_year_index(base)

    matched_indices = set()
    diagnosis_rows = []

    for _, dk in dirty_keys.iterrows():
        row_hits = []
        row_keys = dk["match_keys"] or []

        for key_type, title_key, year in row_keys:
            if not title_key or not year:
                continue

            candidate_idx = candidate_rows_for_key(idx, title_key, int(year), allow_year_window=1)
            if candidate_idx:
                cand_df = base.loc[candidate_idx].copy()
                best, conf, reason = choose_best_imdb_match(cand_df)
                if best is not None:
                    matched_indices.add(int(best.get('__imdb_row_index__', best.name)))
                    row_hits.append({
                        "key_type": key_type,
                        "title_key": title_key,
                        "year": year,
                        "matched_tconst": best["tconst"],
                        "matched_title": best["primaryTitle"],
                        "matched_year": int(best["startYear_int"]),
                        "matched_count": len(candidate_idx),
                        "confidence": conf,
                        "reason": reason,
                        "numVotes": int(best.get("numVotes_int", 0)),
                    })
                    break

        if row_hits:
            h = row_hits[0]
            diagnosis_rows.append({
                "row_id": int(dk["row_id"]),
                "dirty_name": dk["dirty_name"],
                "dirty_year": dk["dirty_year"],
                "dirty_id": dk["dirty_id"],
                "match_key_type": h["key_type"],
                "match_title_key": h["title_key"],
                "match_year_requested": h["year"],
                "matched_tconst": h["matched_tconst"],
                "matched_title": h["matched_title"],
                "matched_year": h["matched_year"],
                "matched_count": h["matched_count"],
                "match_confidence": h["confidence"],
                "match_reason": h["reason"],
                "matched_numVotes": h["numVotes"],
                "unique_or_accepted_match": 1,
            })
        else:
            diagnosis_rows.append({
                "row_id": int(dk["row_id"]),
                "dirty_name": dk["dirty_name"],
                "dirty_year": dk["dirty_year"],
                "dirty_id": dk["dirty_id"],
                "match_key_type": "",
                "match_title_key": "",
                "match_year_requested": "",
                "matched_tconst": "",
                "matched_title": "",
                "matched_year": "",
                "matched_count": 0,
                "match_confidence": 0.0,
                "match_reason": "no_match",
                "matched_numVotes": 0,
                "unique_or_accepted_match": 0,
            })

    diagnosis = pd.DataFrame(diagnosis_rows)

    if not matched_indices:
        return pd.DataFrame(columns=TARGET_REF_COLUMNS), diagnosis

    matched = base.loc[sorted(matched_indices)].copy()

    # Resolve crew names.
    needed = set()
    for _, r in matched.iterrows():
        needed.update(split_imdb_ids(r.get("directors", "")))
        needed.update(split_imdb_ids(r.get("writers", "")))

    name_map = load_needed_names(imdb_dir, needed)

    rows = []
    for _, r in matched.iterrows():
        directors = [name_map.get(x, "") for x in split_imdb_ids(r.get("directors", ""))]
        writers = [name_map.get(x, "") for x in split_imdb_ids(r.get("writers", ""))]

        ref_name = canonical_empty_text(r.get("primaryTitle", ""))
        ref_year = str(int(r["startYear_int"]))
        ref_key = f"{normalize_title_key(ref_name)}::{ref_year}"

        rows.append({
            "ref_key": ref_key,
            "ref_name": ref_name,
            "ref_year": ref_year,
            "ref_release_date": "",      # IMDb public basics has no release date.
            "ref_country": "",           # IMDb public basics has no country.
            "ref_imdb_id": canonical_empty_text(r.get("tconst", "")),
            "ref_description": "",       # IMDb public TSV has no plot/description.
            "ref_duration": canonical_empty_text(r.get("ref_duration", "")),
            "ref_rating_value": canonical_empty_text(r.get("ref_rating_value", "")),
            "ref_director": normalize_people_names(directors),
            "ref_creator": normalize_people_names(writers),
            "ref_genre": canonical_empty_text(r.get("ref_genre", "")),
            "ref_source": "imdb_noncommercial_datasets",
            "ref_match_confidence": 0.95,
            "ref_support_count": 1,
            "ref_total_count": 1,
        })

    ref = pd.DataFrame(rows)
    ref = ref.drop_duplicates(subset=["ref_imdb_id"], keep="first")

    for c in TARGET_REF_COLUMNS:
        if c not in ref.columns:
            ref[c] = ""

    return ref[TARGET_REF_COLUMNS], diagnosis


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dirty_file", default="/mnt/mydata/dq/projects/Splittree/movies/movies_dirty.csv")
    p.add_argument("--imdb_dir", default="/mnt/mydata/dq/projects/Splittree/movie_reference_data")
    p.add_argument("--output", default="movie_reference.csv")
    p.add_argument("--diagnosis_output", default="movie_reference_imdb_match_diagnosis.csv")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.dirty_file):
        raise FileNotFoundError(f"dirty_file not found: {args.dirty_file}")
    if not os.path.exists(args.imdb_dir):
        raise FileNotFoundError(f"imdb_dir not found: {args.imdb_dir}")

    print("========== Build movie_reference.csv from IMDb ==========")
    print(f"dirty_file: {args.dirty_file}")
    print(f"imdb_dir: {args.imdb_dir}")
    print(f"output: {args.output}")
    print(f"diagnosis_output: {args.diagnosis_output}")

    dirty_keys = load_dirty_movie_keys(args.dirty_file)
    print(f"[INFO] dirty rows: {len(dirty_keys)}")

    basics = load_imdb_basics(args.imdb_dir)
    ratings = load_imdb_ratings(args.imdb_dir)
    crew = load_imdb_crew(args.imdb_dir)

    ref, diag = build_reference_from_imdb(dirty_keys, basics, ratings, crew, args.imdb_dir)

    ref.to_csv(args.output, index=False, encoding="utf-8-sig")
    diag.to_csv(args.diagnosis_output, index=False, encoding="utf-8-sig")

    print(f"[OK] saved reference: {args.output}, rows={len(ref)}")
    print(f"[OK] saved diagnosis: {args.diagnosis_output}, rows={len(diag)}")

    unique = int(diag["unique_or_accepted_match"].sum()) if "unique_or_accepted_match" in diag.columns else 0
    print(f"[INFO] dirty rows matched: {unique}/{len(diag)} = {unique / max(len(diag), 1):.6f}")

    print("\n[INFO] reference non-empty field counts:")
    for c in TARGET_REF_COLUMNS:
        if c in {"ref_key", "ref_source"}:
            continue
        cnt = int(ref[c].astype(str).str.strip().replace("empty", "").ne("").sum()) if c in ref.columns else 0
        print(f"  {c}: {cnt}")

    print("\n[INFO] top match reasons:")
    if "match_reason" in diag.columns:
        print(diag["match_reason"].value_counts().head(20).to_string())


if __name__ == "__main__":
    main()

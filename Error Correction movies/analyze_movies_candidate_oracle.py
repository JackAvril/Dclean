"""
Movies candidate oracle / coverage analysis.

Purpose
-------
Analyze repair_candidate_features_infer_whitelist.csv without sending the huge CSV to ChatGPT.

It answers:
1. For each true error cell, does the candidate set contain the clean value?
2. Candidate oracle recall by column.
3. For hit candidates, which candidate_source produced the clean value?
4. For miss candidates, which cells have no correct candidate at all?
5. Whether low recall is caused by candidate generation or by inference gate.

Default inputs
--------------
Run this script in:
    /mnt/mydata/dq/projects/Splittree/test/Error Correction movies

Default paths:
    CANDIDATE_FEATURES = repair_candidate_features_infer_whitelist.csv
    CLEAN_FILE = /mnt/mydata/dq/projects/Splittree/movies/movies_clean.csv
    DIRTY_FILE = /mnt/mydata/dq/projects/Splittree/movies/movies_dirty.csv

Outputs
-------
candidate_oracle_analysis_movies/
    candidate_oracle_column_summary.csv
    candidate_oracle_true_error_cell_details.csv
    candidate_oracle_hit_candidates.csv
    candidate_oracle_miss_cells.csv
    candidate_oracle_source_summary.csv
    candidate_oracle_source_by_column_summary.csv
    candidate_oracle_field_type_summary.csv
"""

import argparse
import os
import re
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple, List

import pandas as pd


# ============================================================
# 1. Defaults
# ============================================================

DEFAULT_CANDIDATE_FEATURES = "repair_candidate_features_infer_whitelist.csv"
DEFAULT_CLEAN_FILE = "/mnt/mydata/dq/projects/Splittree/movies/movies_clean.csv"
DEFAULT_DIRTY_FILE = "/mnt/mydata/dq/projects/Splittree/movies/movies_dirty.csv"
DEFAULT_OUTPUT_DIR = "candidate_oracle_analysis_movies"


EMPTY_TOKENS = {"", "empty", "null", "none", "nan", "na", "n/a", "missing", "{null}"}

MONTH_NAME_TO_NUM = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

MONTH_NUM_TO_NAME = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}

COUNTRY_CANONICAL_MAP = {
    "usa": "USA", "us": "USA", "u.s.": "USA", "u.s.a.": "USA",
    "united states": "USA", "united states of america": "USA",
    "uk": "UK", "u.k.": "UK", "united kingdom": "UK", "england": "UK",
    "france": "France", "germany": "Germany", "india": "India", "canada": "Canada",
    "australia": "Australia", "italy": "Italy", "spain": "Spain", "japan": "Japan",
    "china": "China", "hong kong": "Hong Kong", "south korea": "South Korea",
    "korea": "South Korea", "russia": "Russia", "soviet union": "Soviet Union",
    "mexico": "Mexico", "brazil": "Brazil", "argentina": "Argentina",
    "sweden": "Sweden", "denmark": "Denmark", "norway": "Norway",
    "netherlands": "Netherlands", "ireland": "Ireland", "kuwait": "Kuwait",
}

LANGUAGE_CANONICAL_MAP = {
    "eng": "English", "en": "English", "english": "English",
    "fre": "French", "fra": "French", "fr": "French", "french": "French",
    "spa": "Spanish", "es": "Spanish", "spanish": "Spanish",
    "ger": "German", "deu": "German", "de": "German", "german": "German",
    "ita": "Italian", "it": "Italian", "italian": "Italian",
    "jpn": "Japanese", "ja": "Japanese", "japanese": "Japanese",
    "chi": "Chinese", "zho": "Chinese", "zh": "Chinese", "chinese": "Chinese",
    "kor": "Korean", "ko": "Korean", "korean": "Korean",
    "rus": "Russian", "ru": "Russian", "russian": "Russian",
    "ara": "Arabic", "ar": "Arabic", "arabic": "Arabic",
    "por": "Portuguese", "pt": "Portuguese", "portuguese": "Portuguese",
    "hin": "Hindi", "hi": "Hindi", "hindi": "Hindi",
}


# ============================================================
# 2. Basic normalization
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


def norm_lower(x: Any) -> str:
    return norm_text(x).lower()


def canonical_empty_text(x: Any) -> str:
    s = norm_text(x)
    if s.lower() in EMPTY_TOKENS:
        return "empty"
    return s


def is_empty_like(x: Any) -> bool:
    return norm_lower(x) in EMPTY_TOKENS


def safe_float(x: Any, default=None):
    try:
        if x is None:
            return default
        s = str(x).strip().replace(",", "").replace("%", "")
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def safe_int(x: Any, default=None):
    try:
        v = safe_float(x, None)
        if v is None:
            return default
        return int(v)
    except Exception:
        return default


# ============================================================
# 3. Movies column normalization
# ============================================================

def split_list_tokens(x: Any) -> List[str]:
    s = canonical_empty_text(x)
    if s.lower() in {"", "empty"}:
        return []
    parts = re.split(r"[,;/|]+", s)
    out = []
    seen = set()
    for p in parts:
        p = re.sub(r"\s+", " ", str(p).strip())
        if not p:
            continue
        k = p.lower()
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out


def normalize_list_value(x: Any, order_insensitive: bool = False) -> str:
    toks = split_list_tokens(x)
    if not toks:
        return "empty"
    if order_insensitive:
        toks = sorted(toks, key=lambda z: z.lower())
    return ",".join(toks)


def normalize_year(x: Any) -> str:
    s = canonical_empty_text(x)
    if s.lower() == "empty":
        return "empty"
    m = re.search(r"(18\d{2}|19\d{2}|20\d{2}|21\d{2})", s)
    return m.group(1) if m else s.lower()


def normalize_country(x: Any) -> str:
    s = canonical_empty_text(x)
    if s.lower() == "empty":
        return "empty"
    k = s.lower()
    if k in COUNTRY_CANONICAL_MAP:
        return COUNTRY_CANONICAL_MAP[k].lower()
    return re.sub(r"\s+", " ", s).strip().lower()


def normalize_language(x: Any) -> str:
    s = canonical_empty_text(x)
    if s.lower() == "empty":
        return "empty"
    toks = split_list_tokens(s)
    fixed = []
    for t in toks:
        k = t.lower()
        fixed.append(LANGUAGE_CANONICAL_MAP.get(k, t.strip().title()))
    return ",".join(fixed).lower() if fixed else s.lower()


def parse_release_date_parts(x: Any) -> Optional[Dict[str, Any]]:
    s = canonical_empty_text(x)
    if s.lower() == "empty":
        return None

    country = None
    cm = re.search(r"\(([^()]+)\)", s)
    if cm:
        country = normalize_country(cm.group(1))

    s2 = re.sub(r"\([^()]+\)", "", s).strip()

    m = re.search(r"\b(\d{1,2})\s+([A-Za-z]+)\s+(18\d{2}|19\d{2}|20\d{2}|21\d{2})\b", s2)
    if m:
        day = int(m.group(1))
        month = MONTH_NAME_TO_NUM.get(m.group(2).lower())
        year = int(m.group(3))
        if month and 1 <= day <= 31:
            return {"year": year, "month": month, "day": day, "country": country}

    m = re.search(r"\b([A-Za-z]+)\s+(\d{1,2}),?\s+(18\d{2}|19\d{2}|20\d{2}|21\d{2})\b", s2)
    if m:
        month = MONTH_NAME_TO_NUM.get(m.group(1).lower())
        day = int(m.group(2))
        year = int(m.group(3))
        if month and 1 <= day <= 31:
            return {"year": year, "month": month, "day": day, "country": country}

    m = re.search(r"\b([A-Za-z]+)\s+(18\d{2}|19\d{2}|20\d{2}|21\d{2})\b", s2)
    if m:
        month = MONTH_NAME_TO_NUM.get(m.group(1).lower())
        year = int(m.group(2))
        if month:
            return {"year": year, "month": month, "day": None, "country": country}

    m = re.search(r"\b(18\d{2}|19\d{2}|20\d{2}|21\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", s2)
    if m:
        year = int(m.group(1))
        month = int(m.group(2))
        day = int(m.group(3))
        if 1 <= month <= 12 and 1 <= day <= 31:
            return {"year": year, "month": month, "day": day, "country": country}

    m = re.search(r"\b(\d{1,2})/(\d{1,2})/(18\d{2}|19\d{2}|20\d{2}|21\d{2})\b", s2)
    if m:
        month = int(m.group(1))
        day = int(m.group(2))
        year = int(m.group(3))
        if 1 <= month <= 12 and 1 <= day <= 31:
            return {"year": year, "month": month, "day": day, "country": country}

    y = normalize_year(s2)
    if re.fullmatch(r"\d{4}", y):
        return {"year": int(y), "month": None, "day": None, "country": country}

    return None


def normalize_release_date(x: Any) -> str:
    p = parse_release_date_parts(x)
    if not p:
        return canonical_empty_text(x).lower()
    vals = [str(p.get("year") or ""), str(p.get("month") or ""), str(p.get("day") or "")]
    if p.get("country"):
        vals.append(str(p["country"]))
    return "|".join(vals)


def parse_duration_minutes(x: Any) -> Optional[int]:
    s = canonical_empty_text(x).lower()
    if s == "empty":
        return None

    h = 0
    m = 0
    mh = re.search(r"(\d+)\s*(?:h|hr|hrs|hour|hours)\.?\b", s)
    mm = re.search(r"(\d+)\s*(?:m|min|mins|minute|minutes)\.?\b", s)
    if mh or mm:
        if mh:
            h = int(mh.group(1))
        if mm:
            m = int(mm.group(1))
        total = h * 60 + m
        return total if 1 <= total <= 500 else None

    mn = re.search(r"\b(\d{1,3})(?:\.0)?\b", s)
    if mn:
        total = int(mn.group(1))
        return total if 1 <= total <= 500 else None

    return None


def normalize_duration(x: Any) -> str:
    m = parse_duration_minutes(x)
    if m is None:
        return canonical_empty_text(x).lower()
    return f"{m} min"


def parse_rating_value(x: Any) -> Optional[float]:
    s = canonical_empty_text(x).replace(",", ".")
    if s.lower() == "empty":
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if not m:
        return None
    v = float(m.group(1))
    if "%" in s and 0 <= v <= 100:
        v = v / 10.0
    return v if 0 <= v <= 10 else None


def normalize_rating_value(x: Any) -> str:
    v = parse_rating_value(x)
    if v is None:
        return canonical_empty_text(x).lower()
    return f"{v:.3f}".rstrip("0").rstrip(".")


def parse_count_value(x: Any) -> Optional[int]:
    s = canonical_empty_text(x)
    if s.lower() == "empty":
        return None
    m = re.search(r"(\d[\d,]*)", s)
    if not m:
        return None
    return int(m.group(1).replace(",", ""))


def normalize_count_value(x: Any) -> str:
    v = parse_count_value(x)
    if v is None:
        return canonical_empty_text(x).lower()
    return str(v)


def normalize_review_count(x: Any) -> str:
    s = canonical_empty_text(x)
    if s.lower() == "empty":
        return "empty"
    nums = [int(n.replace(",", "")) for n in re.findall(r"\d[\d,]*", s)]
    if not nums:
        return s.lower()
    return str(sum(nums))


def normalize_movie_value(column: str, value: Any, order_insensitive_lists: bool = False, case_insensitive_text: bool = False) -> str:
    col = str(column).strip()

    if col == "Year":
        return normalize_year(value)
    if col == "Release Date":
        return normalize_release_date(value)
    if col == "Duration":
        return normalize_duration(value)
    if col == "RatingValue":
        return normalize_rating_value(value)
    if col == "RatingCount":
        return normalize_count_value(value)
    if col == "ReviewCount":
        return normalize_review_count(value)
    if col == "Country":
        return normalize_country(value)
    if col == "Language":
        return normalize_language(value)
    if col in {"Actors", "Cast", "Creator", "Director", "Genre", "Filming Locations"}:
        return normalize_list_value(value, order_insensitive=order_insensitive_lists).lower()

    s = canonical_empty_text(value)
    if case_insensitive_text:
        return s.lower()
    return s


def values_equal(column: str, a: Any, b: Any, order_insensitive_lists: bool = False, case_insensitive_text: bool = False) -> bool:
    return normalize_movie_value(column, a, order_insensitive_lists, case_insensitive_text) == normalize_movie_value(column, b, order_insensitive_lists, case_insensitive_text)


def movie_column_type(column: str) -> str:
    c = str(column).strip()
    if c in {"Year", "Release Date", "Duration"}:
        return "time"
    if c in {"RatingValue", "RatingCount", "ReviewCount"}:
        return "rating"
    if c in {"Actors", "Cast", "Creator", "Director", "Genre", "Filming Locations"}:
        return "list_or_people"
    if c in {"Name", "Description"}:
        return "text"
    if c in {"Country", "Language"}:
        return "metadata"
    return "other"


# ============================================================
# 4. Candidate source parsing
# ============================================================

def split_sources(x: Any) -> List[str]:
    s = norm_text(x)
    if s == "":
        return []
    parts = re.split(r"[|;,]+", s)
    out = []
    seen = set()
    for p in parts:
        p = p.strip()
        if not p:
            continue
        k = p.lower()
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out


def classify_source_strength(source_text: Any) -> str:
    s = norm_lower(source_text)

    strong_tokens = [
        "id_to_",
        "name_year_to_",
        "movie_metadata_consistency",
        "candidate_equals_allowed_consensus_value",
        "expected_values_from_rules",
        "rule_expected_value",
    ]
    weak_tokens = [
        "column_top_value",
        "global_top_value",
        "similar_column_value",
    ]

    has_strong = any(t in s for t in strong_tokens)
    has_weak = any(t in s for t in weak_tokens)

    if has_strong and has_weak:
        return "strong_plus_weak"
    if has_strong:
        return "strong"
    if has_weak:
        return "weak_only"
    return "other"


def best_source_column(row: pd.Series) -> str:
    for c in ["candidate_source", "candidate_source_text", "sources_joined"]:
        if c in row.index:
            v = norm_text(row.get(c))
            if v:
                return v
    return ""


# ============================================================
# 5. Core analysis
# ============================================================

def build_true_error_cells(clean_df: pd.DataFrame, dirty_df: pd.DataFrame, args) -> pd.DataFrame:
    if clean_df.shape != dirty_df.shape:
        raise ValueError(f"clean/dirty shape mismatch: clean={clean_df.shape}, dirty={dirty_df.shape}")

    rows = []
    for rid in range(len(clean_df)):
        for col in clean_df.columns:
            clean_val = clean_df.at[rid, col]
            dirty_val = dirty_df.at[rid, col]
            is_err = not values_equal(
                col, dirty_val, clean_val,
                order_insensitive_lists=args.list_order_insensitive,
                case_insensitive_text=args.text_case_insensitive,
            )
            if is_err:
                rows.append({
                    "row_id": int(rid),
                    "column": str(col),
                    "movies_column_type": movie_column_type(str(col)),
                    "dirty_value": canonical_empty_text(dirty_val),
                    "clean_value": canonical_empty_text(clean_val),
                    "dirty_norm": normalize_movie_value(
                        col, dirty_val,
                        order_insensitive_lists=args.list_order_insensitive,
                        case_insensitive_text=args.text_case_insensitive,
                    ),
                    "clean_norm": normalize_movie_value(
                        col, clean_val,
                        order_insensitive_lists=args.list_order_insensitive,
                        case_insensitive_text=args.text_case_insensitive,
                    ),
                })
    return pd.DataFrame(rows)


def prepare_candidates(cand_df: pd.DataFrame, args) -> pd.DataFrame:
    required = {"row_id", "column", "candidate_value"}
    missing = required - set(cand_df.columns)
    if missing:
        raise ValueError(f"candidate_features missing required columns: {missing}")

    out = cand_df.copy()
    out["row_id"] = pd.to_numeric(out["row_id"], errors="raise").astype(int)
    out["column"] = out["column"].astype(str).str.strip()
    out["candidate_value"] = out["candidate_value"].map(canonical_empty_text)

    if "dirty_value" in out.columns:
        out["dirty_value"] = out["dirty_value"].map(canonical_empty_text)

    source_col = None
    for c in ["candidate_source", "candidate_source_text", "sources_joined"]:
        if c in out.columns:
            source_col = c
            break
    if source_col is None:
        out["candidate_source_for_analysis"] = ""
    else:
        out["candidate_source_for_analysis"] = out[source_col].map(norm_text)

    out["source_strength"] = out["candidate_source_for_analysis"].map(classify_source_strength)
    out["candidate_norm"] = [
        normalize_movie_value(
            col, val,
            order_insensitive_lists=args.list_order_insensitive,
            case_insensitive_text=args.text_case_insensitive,
        )
        for col, val in zip(out["column"], out["candidate_value"])
    ]

    # useful score columns if present
    for c in ["student_score", "final_score_from_candidate_gen", "source_score", "candidate_rank"]:
        if c not in out.columns:
            out[c] = None

    return out


def analyze_oracle(true_err_df: pd.DataFrame, cand_df: pd.DataFrame, clean_df: pd.DataFrame, args):
    # group candidates by cell
    cand_groups: Dict[Tuple[int, str], pd.DataFrame] = {}
    for key, g in cand_df.groupby(["row_id", "column"], sort=False):
        cand_groups[(int(key[0]), str(key[1]))] = g.copy()

    details = []
    hit_rows = []

    for _, r in true_err_df.iterrows():
        rid = int(r["row_id"])
        col = str(r["column"])
        clean_norm = r["clean_norm"]
        key = (rid, col)
        g = cand_groups.get(key)

        candidate_count = 0 if g is None else len(g)
        has_any_candidate = int(candidate_count > 0)

        if g is None or g.empty:
            has_clean_candidate = 0
            hit_count = 0
            best_hit_source = ""
            best_hit_strength = ""
            best_hit_value = ""
            best_hit_rank = None
            best_hit_score = None
        else:
            hits = g[g["candidate_norm"] == clean_norm].copy()
            has_clean_candidate = int(len(hits) > 0)
            hit_count = int(len(hits))
            if len(hits) > 0:
                # choose best hit by available scores/rank
                if "student_score" in hits.columns and hits["student_score"].notna().any():
                    hits["_sort_score"] = pd.to_numeric(hits["student_score"], errors="coerce").fillna(-1e9)
                    hits = hits.sort_values(["_sort_score"], ascending=[False])
                elif "final_score_from_candidate_gen" in hits.columns:
                    hits["_sort_score"] = pd.to_numeric(hits["final_score_from_candidate_gen"], errors="coerce").fillna(-1e9)
                    hits = hits.sort_values(["_sort_score"], ascending=[False])
                elif "candidate_rank" in hits.columns:
                    hits["_sort_rank"] = pd.to_numeric(hits["candidate_rank"], errors="coerce").fillna(999999)
                    hits = hits.sort_values(["_sort_rank"], ascending=[True])

                h0 = hits.iloc[0]
                best_hit_source = h0.get("candidate_source_for_analysis", "")
                best_hit_strength = h0.get("source_strength", "")
                best_hit_value = h0.get("candidate_value", "")
                best_hit_rank = h0.get("candidate_rank", None)
                best_hit_score = h0.get("student_score", None)

                for _, hr in hits.iterrows():
                    hit_rows.append({
                        "row_id": rid,
                        "column": col,
                        "movies_column_type": r["movies_column_type"],
                        "dirty_value": r["dirty_value"],
                        "clean_value": r["clean_value"],
                        "candidate_value": hr.get("candidate_value", ""),
                        "candidate_source": hr.get("candidate_source_for_analysis", ""),
                        "source_strength": hr.get("source_strength", ""),
                        "student_score": hr.get("student_score", None),
                        "final_score_from_candidate_gen": hr.get("final_score_from_candidate_gen", None),
                        "source_score": hr.get("source_score", None),
                        "candidate_rank": hr.get("candidate_rank", None),
                    })
            else:
                best_hit_source = ""
                best_hit_strength = ""
                best_hit_value = ""
                best_hit_rank = None
                best_hit_score = None

        details.append({
            "row_id": rid,
            "column": col,
            "movies_column_type": r["movies_column_type"],
            "dirty_value": r["dirty_value"],
            "clean_value": r["clean_value"],
            "dirty_norm": r["dirty_norm"],
            "clean_norm": clean_norm,
            "candidate_count": candidate_count,
            "has_any_candidate": has_any_candidate,
            "has_clean_candidate": has_clean_candidate,
            "hit_count": hit_count,
            "best_hit_value": best_hit_value,
            "best_hit_source": best_hit_source,
            "best_hit_strength": best_hit_strength,
            "best_hit_rank": best_hit_rank,
            "best_hit_score": best_hit_score,
        })

    detail_df = pd.DataFrame(details)
    hit_df = pd.DataFrame(hit_rows)

    return detail_df, hit_df


def summarize_by_column(detail_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, g in detail_df.groupby("column", sort=False):
        true_errors = len(g)
        any_cand = int(g["has_any_candidate"].sum())
        clean_hit = int(g["has_clean_candidate"].sum())
        no_cand = true_errors - any_cand
        miss_with_candidates = int(((g["has_any_candidate"] == 1) & (g["has_clean_candidate"] == 0)).sum())

        rows.append({
            "column": col,
            "movies_column_type": g["movies_column_type"].iloc[0],
            "true_error_cells": true_errors,
            "true_error_cells_with_any_candidate": any_cand,
            "true_error_cells_with_clean_candidate": clean_hit,
            "true_error_cells_without_any_candidate": no_cand,
            "true_error_cells_with_candidates_but_no_clean": miss_with_candidates,
            "candidate_presence_rate": any_cand / true_errors if true_errors else 0.0,
            "candidate_oracle_recall": clean_hit / true_errors if true_errors else 0.0,
            "conditional_oracle_recall_given_candidate": clean_hit / any_cand if any_cand else 0.0,
            "avg_candidate_count_for_true_errors": float(g["candidate_count"].mean()) if len(g) else 0.0,
            "median_candidate_count_for_true_errors": float(g["candidate_count"].median()) if len(g) else 0.0,
        })

    return pd.DataFrame(rows).sort_values(
        ["true_error_cells", "candidate_oracle_recall"],
        ascending=[False, False],
    )


def summarize_by_field_type(detail_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for typ, g in detail_df.groupby("movies_column_type", sort=False):
        true_errors = len(g)
        any_cand = int(g["has_any_candidate"].sum())
        clean_hit = int(g["has_clean_candidate"].sum())
        rows.append({
            "movies_column_type": typ,
            "true_error_cells": true_errors,
            "true_error_cells_with_any_candidate": any_cand,
            "true_error_cells_with_clean_candidate": clean_hit,
            "candidate_presence_rate": any_cand / true_errors if true_errors else 0.0,
            "candidate_oracle_recall": clean_hit / true_errors if true_errors else 0.0,
            "conditional_oracle_recall_given_candidate": clean_hit / any_cand if any_cand else 0.0,
        })
    return pd.DataFrame(rows).sort_values("true_error_cells", ascending=False)


def summarize_sources(hit_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if hit_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    rows = []
    # explode source tokens
    exploded = []
    for _, r in hit_df.iterrows():
        sources = split_sources(r.get("candidate_source", ""))
        if not sources:
            sources = ["<empty_source>"]
        for s in sources:
            d = r.to_dict()
            d["source_token"] = s
            exploded.append(d)
    ex = pd.DataFrame(exploded)

    for s, g in ex.groupby("source_token", sort=False):
        rows.append({
            "source_token": s,
            "hit_candidate_rows": len(g),
            "unique_hit_cells": g[["row_id", "column"]].drop_duplicates().shape[0],
            "columns": ",".join(sorted(g["column"].astype(str).unique())),
        })
    source_summary = pd.DataFrame(rows).sort_values("unique_hit_cells", ascending=False)

    rows2 = []
    for (col, s), g in ex.groupby(["column", "source_token"], sort=False):
        rows2.append({
            "column": col,
            "movies_column_type": g["movies_column_type"].iloc[0],
            "source_token": s,
            "hit_candidate_rows": len(g),
            "unique_hit_cells": g[["row_id", "column"]].drop_duplicates().shape[0],
        })
    source_by_col = pd.DataFrame(rows2).sort_values(
        ["column", "unique_hit_cells"],
        ascending=[True, False],
    )

    return source_summary, source_by_col


# ============================================================
# 6. Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate_features", default=DEFAULT_CANDIDATE_FEATURES)
    p.add_argument("--clean_file", default=DEFAULT_CLEAN_FILE)
    p.add_argument("--dirty_file", default=DEFAULT_DIRTY_FILE)
    p.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--list_order_insensitive", action="store_true")
    p.add_argument("--text_case_insensitive", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("========== Movies Candidate Oracle Analysis ==========")
    print(f"candidate_features: {args.candidate_features}")
    print(f"clean_file: {args.clean_file}")
    print(f"dirty_file: {args.dirty_file}")
    print(f"output_dir: {args.output_dir}")
    print(f"list_order_insensitive: {int(args.list_order_insensitive)}")
    print(f"text_case_insensitive: {int(args.text_case_insensitive)}")

    clean_df = pd.read_csv(args.clean_file, encoding="utf-8-sig")
    dirty_df = pd.read_csv(args.dirty_file, encoding="utf-8-sig")
    cand_df_raw = pd.read_csv(args.candidate_features, encoding="utf-8-sig")

    print(f"[INFO] clean shape: {clean_df.shape}")
    print(f"[INFO] dirty shape: {dirty_df.shape}")
    print(f"[INFO] candidate rows: {len(cand_df_raw)}")
    print(f"[INFO] candidate unique cells: {cand_df_raw[['row_id', 'column']].drop_duplicates().shape[0]}")

    true_err_df = build_true_error_cells(clean_df, dirty_df, args)
    print(f"[INFO] true error cells: {len(true_err_df)}")

    cand_df = prepare_candidates(cand_df_raw, args)

    detail_df, hit_df = analyze_oracle(true_err_df, cand_df, clean_df, args)

    column_summary = summarize_by_column(detail_df)
    field_type_summary = summarize_by_field_type(detail_df)
    miss_df = detail_df[detail_df["has_clean_candidate"] == 0].copy()
    source_summary, source_by_col = summarize_sources(hit_df)

    # Save
    detail_path = os.path.join(args.output_dir, "candidate_oracle_true_error_cell_details.csv")
    hit_path = os.path.join(args.output_dir, "candidate_oracle_hit_candidates.csv")
    miss_path = os.path.join(args.output_dir, "candidate_oracle_miss_cells.csv")
    col_path = os.path.join(args.output_dir, "candidate_oracle_column_summary.csv")
    field_path = os.path.join(args.output_dir, "candidate_oracle_field_type_summary.csv")
    src_path = os.path.join(args.output_dir, "candidate_oracle_source_summary.csv")
    src_col_path = os.path.join(args.output_dir, "candidate_oracle_source_by_column_summary.csv")

    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    hit_df.to_csv(hit_path, index=False, encoding="utf-8-sig")
    miss_df.to_csv(miss_path, index=False, encoding="utf-8-sig")
    column_summary.to_csv(col_path, index=False, encoding="utf-8-sig")
    field_type_summary.to_csv(field_path, index=False, encoding="utf-8-sig")
    source_summary.to_csv(src_path, index=False, encoding="utf-8-sig")
    source_by_col.to_csv(src_col_path, index=False, encoding="utf-8-sig")

    # Print concise report
    print("\n========== Overall Oracle Coverage ==========")
    total_true = len(detail_df)
    total_any = int(detail_df["has_any_candidate"].sum())
    total_hit = int(detail_df["has_clean_candidate"].sum())
    print(f"true_error_cells: {total_true}")
    print(f"true_error_cells_with_any_candidate: {total_any}")
    print(f"true_error_cells_with_clean_candidate: {total_hit}")
    print(f"candidate_presence_rate: {total_any / total_true if total_true else 0.0:.6f}")
    print(f"candidate_oracle_recall: {total_hit / total_true if total_true else 0.0:.6f}")
    print(f"conditional_oracle_recall_given_candidate: {total_hit / total_any if total_any else 0.0:.6f}")

    print("\n========== Column Summary Top ==========")
    show_cols = [
        "column",
        "movies_column_type",
        "true_error_cells",
        "true_error_cells_with_any_candidate",
        "true_error_cells_with_clean_candidate",
        "true_error_cells_without_any_candidate",
        "true_error_cells_with_candidates_but_no_clean",
        "candidate_oracle_recall",
        "conditional_oracle_recall_given_candidate",
        "avg_candidate_count_for_true_errors",
    ]
    print(column_summary[show_cols].to_string(index=False))

    print("\n========== Field Type Summary ==========")
    print(field_type_summary.to_string(index=False))

    print("\n========== Source Summary Top ==========")
    if not source_summary.empty:
        print(source_summary.head(30).to_string(index=False))
    else:
        print("[EMPTY] No clean-value candidate hits.")

    print("\n[OK] Saved:")
    for pth in [col_path, detail_path, hit_path, miss_path, src_path, src_col_path, field_path]:
        print(f"  - {pth}")


if __name__ == "__main__":
    main()

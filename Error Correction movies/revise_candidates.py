"""
Movies repair candidate generation script, full logic-migrated from the Flights candidate generation pipeline.

This is a complete Movies adaptation, not a short/compact script.
It preserves the same downstream output contract as the Flights version:
- repair_candidates_expanded.csv
- repair_candidates_grouped.csv
- predicted_error_positions_consensus_augmented.csv

It also writes Movies-specific diagnostics:
- movie_metadata_consistency.csv
- movie_consistency_suspicious_cells.csv

The Flights script uses flight/source/time consensus. For Movies, that idea is migrated to
movie metadata consistency: Id/Name/Year/ReleaseDate/Director/Genre/Rating fields are used to
produce high-confidence candidates for typed fields such as Duration, Year, Release Date,
Country, RatingValue, RatingCount and ReviewCount.
"""

import os
import json
import math
import re
import unicodedata
from difflib import SequenceMatcher
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# ============================================================
# 1. 配置区（直接改这里）
# ============================================================

ERROR_PRED_FILE = "/mnt/mydata/dq/projects/Splittree/test/Error Correction movies/predicted_error_positions.csv"
CONTEXT_FILE = "/mnt/mydata/dq/projects/Splittree/test/Error Detection movies/candidate_llm_contexts_movies.jsonl"
FALLBACK_CONTEXT_FLAT_FILE = "/mnt/mydata/dq/projects/Splittree/test/Error Detection movies/candidate_llm_contexts_flat_movies.csv"
DIRTY_TABLE_CSV = "/mnt/mydata/dq/projects/Splittree/movies/movies_dirty.csv"

OUTPUT_CANDIDATES_EXPANDED_CSV = "repair_candidates_expanded.csv"
OUTPUT_CANDIDATES_GROUPED_CSV = "repair_candidates_grouped.csv"
OUTPUT_MOVIE_CONSISTENCY_CSV = "movie_metadata_consistency.csv"
OUTPUT_CONSISTENCY_SUSPICIOUS_CELLS_CSV = "movie_consistency_suspicious_cells.csv"
OUTPUT_AUGMENTED_ALLOWED_CELLS_CSV = "predicted_error_positions_consensus_augmented.csv"

TOP_K_COLUMN_VALUES = 12
TOP_K_SIMILAR_ROW_VALUES = 8
MAX_CANDIDATES_PER_CELL = 28
MIN_STRING_SIM_FOR_COLUMN_VALUE = 0.42
INCLUDE_DIRTY_VALUE = False

ENABLE_COLUMN_SPECIFIC_GENERATION = True
ENABLE_MOVIE_CONSISTENCY = True
ENABLE_MOVIE_AUGMENTED_WHITELIST = True

MIN_CONSISTENCY_SUPPORT_COUNT = 2
MIN_CONSISTENCY_CONFIDENCE = 0.70
MIN_CONSISTENCY_MARGIN = 0.20
MOVIE_CONSISTENCY_ALLOW_COLUMNS = {
    "Year", "Release Date", "Country", "Duration", "RatingValue", "RatingCount", "ReviewCount",
    # Newly added for Movies oracle coverage. These are not blindly repaired;
    # they only enter candidate generation / whitelist when a high-confidence
    # direct or metadata candidate exists.
    "Id", "Description", "Genre", "Creator", "Director", "Name"
}
CONSISTENCY_STRONG_SCORE = 0.96
CONSISTENCY_MEDIUM_SCORE = 0.86


# ============================================================
# 2. 基础工具函数
# ============================================================

EMPTY_TOKENS = {"", "empty", "null", "none", "nan", "na", "n/a", "missing", "{null}"}

MONTH_NAME_TO_NUM = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}

COUNTRY_CANONICAL_MAP = {
    "usa": "USA", "united states": "USA", "united states of america": "USA", "us": "USA",
    "uk": "UK", "united kingdom": "UK", "england": "UK",
    "france": "France", "germany": "Germany", "india": "India", "canada": "Canada",
    "australia": "Australia", "italy": "Italy", "spain": "Spain", "japan": "Japan",
    "china": "China", "hong kong": "Hong Kong", "south korea": "South Korea",
    "russia": "Russia", "soviet union": "Soviet Union", "mexico": "Mexico", "brazil": "Brazil",
    "argentina": "Argentina", "sweden": "Sweden", "denmark": "Denmark", "norway": "Norway",
    "netherlands": "Netherlands", "ireland": "Ireland", "kuwait": "Kuwait",
}

LANGUAGE_CANONICAL_MAP = {
    "english": "English", "french": "French", "spanish": "Spanish", "german": "German",
    "italian": "Italian", "japanese": "Japanese", "chinese": "Chinese", "mandarin": "Mandarin",
    "hindi": "Hindi", "korean": "Korean", "russian": "Russian", "arabic": "Arabic",
    "portuguese": "Portuguese",
}


def norm_text(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def canonical_empty_text(x: Any) -> str:
    if pd.isna(x):
        return "empty"
    s = str(x).strip()
    return "empty" if s.lower() in EMPTY_TOKENS else s


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or pd.isna(x):
            return default
        return float(str(x).strip().replace(",", ""))
    except Exception:
        return default


def safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None or pd.isna(x):
            return default
        return int(float(str(x).strip().replace(",", "")))
    except Exception:
        return default


def path_exists_nonempty(path: Optional[str]) -> bool:
    return path is not None and str(path).strip() != "" and os.path.exists(str(path))


def resolve_output_path(output_path: str) -> str:
    if os.path.isabs(str(output_path)):
        return str(output_path)
    return os.path.abspath(str(output_path))


def string_similarity(a: Any, b: Any) -> float:
    a = norm_text(a)
    b = norm_text(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def clean_key(x: Any) -> str:
    return re.sub(r"\s+", " ", norm_text(x)).strip()


def split_list_tokens(x: Any) -> List[str]:
    s = canonical_empty_text(x)
    if s in {"", "empty"}:
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


def normalize_list_value(x: Any) -> Optional[str]:
    toks = split_list_tokens(x)
    return ",".join(toks) if toks else None


# ============================================================
# 2.1 Movies high-precision direct/metadata expansion helpers
# ============================================================

MONTH_NUM_TO_NAME = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}

MOVIES_GENRE_CONSERVATIVE_MAP = {
    # RottenTomatoes-style labels -> IMDb-style safer labels.
    # Mappings are intentionally conservative to avoid creating clean-value supersets.
    "kids & family": "Family",
    "science fiction & fantasy": "Sci-Fi",
    "mystery & suspense": "Thriller",
    "action & adventure": "Action",
    "sports & fitness": "Sport",
    "musical & performing arts": "Musical",
    "art house & international": "Drama",
    "animation": "Animation",
    "comedy": "Comedy",
    "drama": "Drama",
    "documentary": "Documentary",
    "horror": "Horror",
    "romance": "Romance",
    "western": "Western",
}
MOVIES_GENRE_DROP_TOKENS = {
    "classics", "special interest", "television", "cult movies", "gay & lesbian"
}


def deaccent_text(x: Any) -> str:
    s = canonical_empty_text(x)
    if s in {"", "empty"}:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s


def normalize_movie_name_candidate(x: Any) -> Optional[str]:
    """Small, high-precision Name normalization candidate."""
    s = canonical_empty_text(x)
    if s in {"", "empty"}:
        return None
    y = deaccent_text(s)
    y = y.replace("&", "and")
    y = re.sub(r"[’`´]", "'", y)
    y = re.sub(r"\s+", " ", y).strip()
    if y and norm_text(y) != norm_text(s):
        return y
    return None


def repair_year_triple_to_middle_candidate(x: Any) -> Optional[str]:
    s = canonical_empty_text(x)
    if s in {"", "empty"}:
        return None
    years = re.findall(r"(18\d{2}|19\d{2}|20\d{2}|21\d{2})", s)
    if len(years) != 3:
        return None
    ys = [int(v) for v in years]
    if ys[1] == ys[0] + 1 and ys[2] == ys[1] + 1:
        return str(ys[1])
    return None


def repair_release_date_wide_limited_candidate(x: Any, default_country: str = "USA") -> Optional[str]:
    """Nov 11, 2011 Wide -> 11 November 2011 (USA)."""
    s = canonical_empty_text(x)
    if s in {"", "empty"}:
        return None
    m = re.search(
        r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2}),?\s+(18\d{2}|19\d{2}|20\d{2}|21\d{2})\s+"
        r"(wide|limited|wide release|limited release)\b",
        s,
        flags=re.I,
    )
    if not m:
        return None
    month = MONTH_NAME_TO_NUM.get(m.group(1).lower())
    day = int(m.group(2))
    year = int(m.group(3))
    if not month or not (1 <= day <= 31):
        return None
    country = default_country or "USA"
    return f"{day} {MONTH_NUM_TO_NAME[month]} {year} ({country})"


def repair_genre_conservative_candidate(x: Any) -> Optional[str]:
    s = canonical_empty_text(x)
    if s in {"", "empty"}:
        return None
    parts = re.split(r"[,;/|]+", s)
    out = []
    seen = set()
    changed = False
    for part in parts:
        p = re.sub(r"\s+", " ", part.strip())
        if not p:
            continue
        key = p.lower()
        if key in MOVIES_GENRE_DROP_TOKENS:
            changed = True
            continue
        mapped = MOVIES_GENRE_CONSERVATIVE_MAP.get(key)
        if mapped is None:
            mapped = p
        else:
            changed = True
        for sub in re.split(r"[,/]+", mapped):
            sub = re.sub(r"\s+", " ", sub.strip())
            if not sub:
                continue
            k = sub.lower()
            if k not in seen:
                seen.add(k)
                out.append(sub)
    if not out or not changed:
        return None
    repaired = ",".join(out)
    if norm_text(repaired) == norm_text(s):
        return None
    return repaired


def repair_duration_single_format_candidate(x: Any) -> Optional[str]:
    """Only creates a candidate; the inference gate can still decide whether to use it."""
    s = canonical_empty_text(x)
    if s in {"", "empty"}:
        return None
    # Avoid multi-duration values.
    if "," in s or "/" in s or "|" in s or ";" in s:
        return None
    fixed = normalize_duration_candidate(s)
    if fixed and norm_text(fixed) != norm_text(s):
        return fixed
    return None


def repair_rating_value_single_format_candidate(x: Any) -> Optional[str]:
    """Only creates a candidate; avoid multi-rating strings such as 6.8/10,6.3/10."""
    s = canonical_empty_text(x)
    if s in {"", "empty"}:
        return None
    if "," in s or "|" in s or ";" in s:
        return None
    if "/10" not in s.lower() and "%" not in s and not re.search(r"\d,\d", s):
        return None
    fixed = normalize_rating_value_candidate(s)
    if fixed and norm_text(fixed) != norm_text(s):
        return fixed
    return None


def looks_like_imdb_id(x: Any) -> bool:
    return bool(re.fullmatch(r"tt\d{5,12}", canonical_empty_text(x).strip()))


def looks_like_movie_slug_id(x: Any) -> bool:
    s = canonical_empty_text(x).strip().lower()
    return bool(s and not looks_like_imdb_id(s) and re.search(r"[_-](18\d{2}|19\d{2}|20\d{2}|21\d{2})$", s))




# ============================================================
# 3. Movies 专属规范化
# ============================================================

def normalize_year_candidate(x: Any) -> Optional[str]:
    s = canonical_empty_text(x)
    if s in {"", "empty"}:
        return None
    m = re.search(r"(18\d{2}|19\d{2}|20\d{2}|21\d{2})", s)
    if not m:
        return None
    y = int(m.group(1))
    return str(y) if 1800 <= y <= 2100 else None


def normalize_country_candidate(x: Any) -> Optional[str]:
    s = canonical_empty_text(x)
    if s in {"", "empty"}:
        return None
    k = norm_text(s)
    if k in COUNTRY_CANONICAL_MAP:
        return COUNTRY_CANONICAL_MAP[k]
    if re.fullmatch(r"[A-Za-z][A-Za-z .'-]{1,40}", s.strip()):
        return re.sub(r"\s+", " ", s.strip()).title()
    return None


def normalize_language_candidate(x: Any) -> Optional[str]:
    s = canonical_empty_text(x)
    if s in {"", "empty"}:
        return None
    toks = split_list_tokens(s)
    fixed = []
    for t in toks:
        k = norm_text(t)
        fixed.append(LANGUAGE_CANONICAL_MAP.get(k, t.strip().title()))
    return ",".join(fixed) if fixed else None


def parse_release_date_parts(x: Any) -> Optional[Dict[str, Any]]:
    s = canonical_empty_text(x)
    if s in {"", "empty"}:
        return None
    country = None
    cm = re.search(r"\(([^()]+)\)", s)
    if cm:
        country = normalize_country_candidate(cm.group(1)) or cm.group(1).strip()
    s_no_country = re.sub(r"\([^()]+\)", "", s).strip()

    m = re.search(r"\b(\d{1,2})\s+([A-Za-z]+)\s+(18\d{2}|19\d{2}|20\d{2}|21\d{2})\b", s_no_country)
    if m:
        day = int(m.group(1))
        month = MONTH_NAME_TO_NUM.get(m.group(2).lower())
        year = int(m.group(3))
        if month and 1 <= day <= 31:
            return {"year": year, "month": month, "day": day, "country": country}

    m = re.search(r"\b([A-Za-z]+)\s+(18\d{2}|19\d{2}|20\d{2}|21\d{2})\b", s_no_country)
    if m:
        month = MONTH_NAME_TO_NUM.get(m.group(1).lower())
        year = int(m.group(2))
        if month:
            return {"year": year, "month": month, "day": None, "country": country}

    m = re.search(r"\b(18\d{2}|19\d{2}|20\d{2}|21\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", s_no_country)
    if m:
        year = int(m.group(1)); month = int(m.group(2)); day = int(m.group(3))
        if 1 <= month <= 12 and 1 <= day <= 31:
            return {"year": year, "month": month, "day": day, "country": country}

    m = re.search(r"\b(\d{1,2})/(\d{1,2})/(18\d{2}|19\d{2}|20\d{2}|21\d{2})\b", s_no_country)
    if m:
        month = int(m.group(1)); day = int(m.group(2)); year = int(m.group(3))
        if 1 <= month <= 12 and 1 <= day <= 31:
            return {"year": year, "month": month, "day": day, "country": country}

    y = normalize_year_candidate(s_no_country)
    if y:
        return {"year": int(y), "month": None, "day": None, "country": country}
    return None


def format_release_date_candidate(parts: Optional[Dict[str, Any]]) -> Optional[str]:
    if not parts or not parts.get("year"):
        return None
    month_names = [None, "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    year = int(parts["year"])
    month = parts.get("month")
    day = parts.get("day")
    country = parts.get("country")
    if month and 1 <= int(month) <= 12 and day:
        base = f"{int(day)} {month_names[int(month)]} {year}"
    elif month and 1 <= int(month) <= 12:
        base = f"{month_names[int(month)]} {year}"
    else:
        base = str(year)
    return f"{base} ({country})" if country else base


def normalize_release_date_candidate(x: Any) -> Optional[str]:
    return format_release_date_candidate(parse_release_date_parts(x))


def normalize_duration_candidate(x: Any) -> Optional[str]:
    s = canonical_empty_text(x)
    if s in {"", "empty"}:
        return None
    h = 0; m = 0
    mh = re.search(r"(\d+)\s*(?:h|hr|hrs|hour|hours)\b", s, flags=re.I)
    mm = re.search(r"(\d+)\s*(?:m|min|mins|minute|minutes)\b", s, flags=re.I)
    if mh or mm:
        if mh: h = int(mh.group(1))
        if mm: m = int(mm.group(1))
        total = h * 60 + m
        if 1 <= total <= 500:
            return f"{total} min"
    mnum = re.search(r"\b(\d{1,3})(?:\.0)?\b", s)
    if mnum:
        num = int(mnum.group(1))
        if 1 <= num <= 500:
            return f"{num} min"
    return None


def normalize_rating_value_candidate(x: Any) -> Optional[str]:
    s = canonical_empty_text(x)
    if s in {"", "empty"}:
        return None
    raw = s.strip().replace(",", ".")
    m = re.search(r"(\d+(?:\.\d+)?)", raw)
    if not m:
        return None
    v = float(m.group(1))
    if "%" in raw and 0 <= v <= 100:
        v /= 10.0
    if 0 <= v <= 10:
        return f"{v:.1f}" if abs(v - round(v)) < 1e-9 else f"{v:.1f}".rstrip("0").rstrip(".")
    return None


def normalize_count_candidate(x: Any) -> Optional[str]:
    s = canonical_empty_text(x)
    if s in {"", "empty"}:
        return None
    m = re.search(r"(\d[\d,]*)", s)
    if not m:
        return None
    return str(int(m.group(1).replace(",", "")))


def normalize_review_count_candidate(x: Any) -> Optional[str]:
    s = canonical_empty_text(x)
    if s in {"", "empty"}:
        return None
    nums = [int(x.replace(",", "")) for x in re.findall(r"\d[\d,]*", s)]
    if not nums:
        return None
    lower = norm_text(s)
    if "user" in lower or "critic" in lower:
        parts = []
        um = re.search(r"(\d[\d,]*)\s*user", lower)
        cm = re.search(r"(\d[\d,]*)\s*critic", lower)
        if um:
            u = int(um.group(1).replace(",", "")); parts.append(f"{u} user" + ("" if u == 1 else "s"))
        if cm:
            c = int(cm.group(1).replace(",", "")); parts.append(f"{c} critic" + ("" if c == 1 else "s"))
        if parts:
            return ",".join(parts)
    return str(sum(nums))


def normalize_by_column(column: Any, value: Any) -> Optional[str]:
    cl = str(column).strip().lower()
    if cl == "year": return normalize_year_candidate(value)
    if cl == "release date": return normalize_release_date_candidate(value)
    if cl in {"country", "release country"}: return normalize_country_candidate(value)
    if cl == "language": return normalize_language_candidate(value)
    if cl == "duration": return normalize_duration_candidate(value)
    if cl == "ratingvalue": return normalize_rating_value_candidate(value)
    if cl in {"ratingcount", "id"}: return normalize_count_candidate(value)
    if cl == "reviewcount": return normalize_review_count_candidate(value)
    if cl in {"actors", "cast", "creator", "director", "genre", "filming locations"}: return normalize_list_value(value)
    return None


def value_equal_for_column(column: Any, a: Any, b: Any) -> bool:
    fa = normalize_by_column(column, a)
    fb = normalize_by_column(column, b)
    if fa is not None and fb is not None:
        return norm_text(fa) == norm_text(fb)
    return norm_text(a) == norm_text(b)


def is_movie_time_col(column: Any, semantic_type: Any = None) -> bool:
    return str(column).strip().lower() in {"year", "release date", "duration"} or norm_text(semantic_type) in {"year_like", "date_like", "duration_like"}


def is_movie_rating_col(column: Any, semantic_type: Any = None) -> bool:
    return str(column).strip().lower() in {"ratingvalue", "ratingcount", "reviewcount"} or norm_text(semantic_type) in {"rating_like", "count_like", "review_count_like"}


def is_movie_people_col(column: Any) -> bool:
    return str(column).strip().lower() in {"director", "creator", "actors", "cast"}


def is_movie_list_col(column: Any, semantic_type: Any = None) -> bool:
    return str(column).strip().lower() in {"actors", "cast", "creator", "genre", "filming locations"} or norm_text(semantic_type) == "categorical_list"


# ============================================================
# 4. 读取输入
# ============================================================

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSONL 解析失败: {path}, line={line_no}, err={e}") from e
    return rows


def load_context_records(path: str) -> List[Dict[str, Any]]:
    lower = str(path).lower()
    if lower.endswith(".jsonl"):
        return load_jsonl(path)
    if lower.endswith(".csv") or lower.endswith(".txt"):
        return pd.read_csv(path, encoding="utf-8-sig").to_dict(orient="records")
    try:
        return load_jsonl(path)
    except Exception:
        return pd.read_csv(path, encoding="utf-8-sig").to_dict(orient="records")


def pick_error_positions(df: pd.DataFrame) -> pd.DataFrame:
    if "final_pred_label" in df.columns:
        return df[pd.to_numeric(df["final_pred_label"], errors="coerce").fillna(0).astype(int) == 1].copy()
    if "final_pred_label_name" in df.columns:
        return df[df["final_pred_label_name"].astype(str).str.strip().str.lower() == "error"].copy()
    if "detector_pred_label" in df.columns:
        return df[pd.to_numeric(df["detector_pred_label"], errors="coerce").fillna(0).astype(int) == 1].copy()
    if "detector_pred_label_name" in df.columns:
        return df[df["detector_pred_label_name"].astype(str).str.strip().str.lower() == "error"].copy()
    return df.copy()


# ============================================================
# 5. 全局字典与上下文
# ============================================================

def consume_movie_row(row_vals: Dict[str, Any], global_dict: Dict[str, Any]):
    if not isinstance(row_vals, dict):
        return
    row = {str(k): canonical_empty_text(v) for k, v in row_vals.items()}
    for c, v in row.items():
        if v not in {"", "empty"}:
            fixed = normalize_by_column(c, v)
            global_dict["column_value_counter"][c][fixed if fixed is not None else v] += 1

    movie_id = row.get("Id", "")
    name = row.get("Name", "")
    year = normalize_year_candidate(row.get("Year", ""))
    release_date = normalize_release_date_candidate(row.get("Release Date", ""))
    release_parts = parse_release_date_parts(row.get("Release Date", ""))
    release_year = str(release_parts["year"]) if release_parts and release_parts.get("year") else None
    release_country = release_parts.get("country") if release_parts else None
    director = row.get("Director", "")
    creator = row.get("Creator", "")
    actors = row.get("Actors", "")
    cast = row.get("Cast", "")
    language = normalize_language_candidate(row.get("Language", ""))
    country = normalize_country_candidate(row.get("Country", ""))
    duration = normalize_duration_candidate(row.get("Duration", ""))
    rating_value = normalize_rating_value_candidate(row.get("RatingValue", ""))
    rating_count = normalize_count_candidate(row.get("RatingCount", ""))
    review_count = normalize_review_count_candidate(row.get("ReviewCount", ""))
    genre = row.get("Genre", "")
    locations = row.get("Filming Locations", "")
    description = row.get("Description", "")

    def add_map(map_name, lhs, rhs):
        if lhs and lhs not in {"", "empty"} and rhs and rhs not in {"", "empty"}:
            global_dict[map_name][clean_key(lhs)][rhs] += 1

    def add_map_tuple(map_name, lhs_tuple, rhs):
        if all(x and x not in {"", "empty"} for x in lhs_tuple) and rhs and rhs not in {"", "empty"}:
            global_dict[map_name][tuple(clean_key(x) for x in lhs_tuple)][rhs] += 1

    add_map("id_to_name", movie_id, name)
    add_map("name_to_id", name, movie_id)
    add_map_tuple("name_year_to_id", (name, year), movie_id)
    add_map("id_to_year", movie_id, year)
    add_map("id_to_release_date", movie_id, release_date)
    add_map("id_to_duration", movie_id, duration)
    add_map("id_to_rating_value", movie_id, rating_value)
    add_map("id_to_rating_count", movie_id, rating_count)
    add_map("name_to_year", name, year)
    add_map("name_to_release_date", name, release_date)
    add_map("name_to_duration", name, duration)
    add_map("name_to_country", name, country)
    add_map_tuple("name_year_to_duration", (name, year), duration)
    add_map_tuple("name_year_to_release_date", (name, year), release_date)
    add_map_tuple("name_year_to_country", (name, year), country)
    add_map("year_to_release_year", year, release_year)
    add_map("release_country_to_country", release_country, country)
    add_map("country_to_language", country, language)
    add_map("country_to_release_country", country, release_country)
    add_map_tuple("director_genre_to_duration", (director, genre), duration)
    add_map_tuple("director_location_to_duration", (director, locations), duration)
    add_map_tuple("rating_count_genre_to_duration", (rating_count, genre), duration)
    add_map_tuple("rating_value_count_to_duration", (rating_value, rating_count), duration)
    add_map_tuple("name_year_to_rating_value", (name, year), rating_value)
    add_map_tuple("name_year_to_rating_count", (name, year), rating_count)
    add_map_tuple("rating_count_genre_to_review_count", (rating_count, genre), review_count)
    add_map("name_to_director", name, director)
    add_map("name_to_creator", name, creator)
    add_map("name_to_description", name, description)
    add_map_tuple("name_year_to_director", (name, year), director)
    add_map_tuple("name_year_to_creator", (name, year), creator)
    add_map_tuple("name_year_to_description", (name, year), description)
    add_map("name_to_actors", name, actors)
    add_map("name_to_cast", name, cast)
    add_map_tuple("director_creator_to_actors", (director, creator), actors)
    add_map_tuple("actors_to_cast", (actors,), cast)


def build_global_dictionaries(context_records: List[Dict[str, Any]], dirty_table_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    keys = [
        "column_value_counter", "id_to_name", "id_to_year", "id_to_release_date", "id_to_duration",
        "id_to_rating_value", "id_to_rating_count", "name_to_year", "name_to_release_date",
        "name_to_duration", "name_to_country", "name_year_to_duration", "name_year_to_release_date",
        "name_year_to_country", "year_to_release_year", "release_country_to_country", "country_to_language",
        "country_to_release_country", "director_genre_to_duration", "director_location_to_duration",
        "rating_count_genre_to_duration", "rating_value_count_to_duration", "name_year_to_rating_value",
        "name_year_to_rating_count", "rating_count_genre_to_review_count",
        "name_to_id", "name_year_to_id",
        "name_to_director", "name_to_creator", "name_to_description",
        "name_year_to_director", "name_year_to_creator", "name_year_to_description",
        "name_to_actors", "name_to_cast", "director_creator_to_actors", "actors_to_cast",
    ]
    global_dict = {k: defaultdict(Counter) for k in keys}

    for obj in context_records:
        row_ctx = obj.get("row_context", {})
        if isinstance(row_ctx, dict):
            consume_movie_row(row_ctx.get("row_values", {}), global_dict)
        sim_ctx = obj.get("similarity_context", {})
        if isinstance(sim_ctx, dict):
            for item in sim_ctx.get("topk_similar_rows", []) or []:
                if isinstance(item, dict):
                    consume_movie_row(item.get("row_values", {}), global_dict)
        col = obj.get("column", "")
        val = obj.get("value", None)
        if col and val is not None:
            fixed = normalize_by_column(col, val)
            val2 = fixed if fixed is not None else canonical_empty_text(val)
            if val2 not in {"", "empty"}:
                global_dict["column_value_counter"][str(col)][val2] += 1
        col_ctx = obj.get("column_context", {})
        if isinstance(col_ctx, dict):
            c = str(col_ctx.get("column", col))
            for t in col_ctx.get("top_values_with_counts", []) or []:
                if isinstance(t, dict):
                    v = canonical_empty_text(t.get("value", ""))
                    fixed = normalize_by_column(c, v)
                    v2 = fixed if fixed is not None else v
                    if v2 not in {"", "empty"}:
                        global_dict["column_value_counter"][c][v2] += safe_int(t.get("count", 1), 1)

    if dirty_table_df is not None and not dirty_table_df.empty:
        for _, r in dirty_table_df.iterrows():
            consume_movie_row({c: r[c] for c in dirty_table_df.columns}, global_dict)
    return global_dict


def _counter_best(counter: Counter):
    if not counter:
        return None, 0, 0, 0.0, 0.0
    total = sum(counter.values())
    items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    best, best_cnt = items[0]
    second = items[1][1] if len(items) > 1 else 0
    return best, best_cnt, total, best_cnt / max(total, 1), (best_cnt - second) / max(total, 1)


def build_movie_metadata_consistency(dirty_table_df: Optional[pd.DataFrame]):
    consistency_map = {}
    consistency_rows = []
    suspicious_rows = []
    if dirty_table_df is None or dirty_table_df.empty:
        return consistency_map, pd.DataFrame(), pd.DataFrame()
    work = dirty_table_df.copy()
    work["__row_id__"] = range(len(work))
    relation_specs = [
        {"name": "id_to_year", "lhs_cols": ["Id"], "rhs_col": "Year", "normalizer": normalize_year_candidate, "score": 0.94},
        {"name": "id_to_release_date", "lhs_cols": ["Id"], "rhs_col": "Release Date", "normalizer": normalize_release_date_candidate, "score": 0.92},
        {"name": "id_to_duration", "lhs_cols": ["Id"], "rhs_col": "Duration", "normalizer": normalize_duration_candidate, "score": 0.90},
        {"name": "name_year_to_duration", "lhs_cols": ["Name", "Year"], "rhs_col": "Duration", "normalizer": normalize_duration_candidate, "score": 0.84},
        {"name": "director_genre_to_duration", "lhs_cols": ["Director", "Genre"], "rhs_col": "Duration", "normalizer": normalize_duration_candidate, "score": 0.78},
        {"name": "rating_value_count_to_duration", "lhs_cols": ["RatingValue", "RatingCount"], "rhs_col": "Duration", "normalizer": normalize_duration_candidate, "score": 0.80},
        {"name": "name_year_to_rating_value", "lhs_cols": ["Name", "Year"], "rhs_col": "RatingValue", "normalizer": normalize_rating_value_candidate, "score": 0.88},
        {"name": "name_year_to_rating_count", "lhs_cols": ["Name", "Year"], "rhs_col": "RatingCount", "normalizer": normalize_count_candidate, "score": 0.86},
        {"name": "release_country_to_country", "lhs_cols": ["Release Date"], "rhs_col": "Country", "normalizer": normalize_country_candidate, "lhs_transform": "release_country", "score": 0.82},
        {"name": "country_to_language", "lhs_cols": ["Country"], "rhs_col": "Language", "normalizer": normalize_language_candidate, "score": 0.62},
    ]
    for spec in relation_specs:
        lhs_cols = spec["lhs_cols"]; rhs_col = spec["rhs_col"]
        if rhs_col not in work.columns or any(c not in work.columns for c in lhs_cols):
            continue
        rhs_normalizer = spec.get("normalizer")
        group_counter = defaultdict(Counter)
        for _, r in work.iterrows():
            lhs_parts = []
            skip = False
            for c in lhs_cols:
                raw = canonical_empty_text(r.get(c, ""))
                if raw in {"", "empty"}:
                    skip = True; break
                if spec.get("lhs_transform") == "release_country" and c == "Release Date":
                    parts = parse_release_date_parts(raw)
                    raw = parts.get("country") if parts else None
                    if not raw:
                        skip = True; break
                fixed = normalize_by_column(c, raw)
                lhs_parts.append(clean_key(fixed if fixed is not None else raw))
            if skip:
                continue
            rhs_raw = canonical_empty_text(r.get(rhs_col, ""))
            if rhs_raw in {"", "empty"}:
                continue
            rhs_val = rhs_normalizer(rhs_raw) if rhs_normalizer else rhs_raw
            if not rhs_val or rhs_val in {"", "empty"}:
                continue
            group_counter[tuple(lhs_parts)][rhs_val] += 1
        for lhs_key, counter in group_counter.items():
            best, best_cnt, total, conf, margin = _counter_best(counter)
            if best is None:
                continue
            high_conf = best_cnt >= MIN_CONSISTENCY_SUPPORT_COUNT and conf >= MIN_CONSISTENCY_CONFIDENCE and margin >= MIN_CONSISTENCY_MARGIN
            consistency_rows.append({"consistency_name": spec["name"], "lhs_key": " || ".join(lhs_key), "rhs_column": rhs_col, "consensus_value": best, "support_count": best_cnt, "total_count": total, "confidence": conf, "margin": margin, "is_high_confidence": int(high_conf)})
            if not high_conf:
                continue
            for _, r in work.iterrows():
                lhs_parts2 = []
                skip2 = False
                for c in lhs_cols:
                    raw = canonical_empty_text(r.get(c, ""))
                    if raw in {"", "empty"}:
                        skip2 = True; break
                    if spec.get("lhs_transform") == "release_country" and c == "Release Date":
                        parts = parse_release_date_parts(raw)
                        raw = parts.get("country") if parts else None
                        if not raw:
                            skip2 = True; break
                    fixed = normalize_by_column(c, raw)
                    lhs_parts2.append(clean_key(fixed if fixed is not None else raw))
                if skip2 or tuple(lhs_parts2) != lhs_key:
                    continue
                row_id = int(r["__row_id__"])
                dirty = canonical_empty_text(r.get(rhs_col, ""))
                if value_equal_for_column(rhs_col, dirty, best):
                    continue
                suspicious_rows.append({"row_id": row_id, "column": rhs_col, "value": dirty, "consistency_name": spec["name"], "lhs_key": " || ".join(lhs_key), "consensus_value": best, "consensus_confidence": conf, "consensus_margin": margin, "consensus_support_count": best_cnt, "consensus_total_count": total, "consensus_reason": "movie_metadata_consistency"})
                consistency_map[(row_id, rhs_col)] = {"candidate_value": best, "consistency_name": spec["name"], "lhs_key": " || ".join(lhs_key), "confidence": conf, "margin": margin, "support_count": best_cnt, "total_count": total, "score": spec.get("score", CONSISTENCY_MEDIUM_SCORE)}
    consistency_df = pd.DataFrame(consistency_rows)
    suspicious_df = pd.DataFrame(suspicious_rows)
    if not suspicious_df.empty:
        suspicious_df = suspicious_df.drop_duplicates(subset=["row_id", "column"], keep="first").reset_index(drop=True)
    return consistency_map, consistency_df, suspicious_df




def build_dirty_table_context_map(dirty_table_df: Optional[pd.DataFrame], global_dict: Dict[str, Any]) -> Dict[Tuple[int, str], Dict[str, Any]]:
    """Create fallback contexts for all dirty-table cells so augmented whitelist cells can generate candidates."""
    out = {}
    if dirty_table_df is None or dirty_table_df.empty:
        return out
    for rid, r in dirty_table_df.iterrows():
        row_values = {str(c): canonical_empty_text(r[c]) for c in dirty_table_df.columns}
        for col in dirty_table_df.columns:
            val = canonical_empty_text(r[col])
            counter = global_dict["column_value_counter"].get(str(col), Counter())
            total = max(sum(counter.values()), 1)
            top_values = []
            for v, cnt in counter.most_common(TOP_K_COLUMN_VALUES):
                if v in {"", "empty"}:
                    continue
                top_values.append({"value": v, "raw_value": v, "count": cnt, "ratio": cnt / total})
            out[(int(rid), str(col))] = {
                "row_id": int(rid),
                "column": str(col),
                "value": val,
                "row_values": row_values,
                "movie_metadata_bundle": {},
                "suggested_correct_value": "",
                "semantic_type": None,
                "detected_type": None,
                "violation_count": None,
                "conflict_score": None,
                "value_frequency": None,
                "value_frequency_rank": None,
                "parsed_year": normalize_year_candidate(val) if str(col) == "Year" else None,
                "parsed_release_year": None,
                "parsed_release_country": None,
                "duration_minutes": None,
                "rating_value_float": None,
                "count_value_int": None,
                "review_total": None,
                "list_token_count": len(split_list_tokens(val)) if str(col) in {"Actors", "Cast", "Creator", "Director", "Genre", "Filming Locations"} else None,
                "neighbor_majority_value": "",
                "neighbor_majority_ratio": 0.0,
                "similar_row_target_values": [],
                "column_top_values": top_values,
                "alternative_values_from_column": [],
                "rule_expected_values": [],
                "expected_values_from_rules": [],
                "prior_error_probability": None,
                "posterior_error_probability": None,
                "main_rule_type": None,
                "main_usage_role": None,
                "strong_rule_count": 0,
                "candidate_generation_rule_count": 0,
                "fd_like_count": 0,
                "context_rule_count": 0,
                "global_rule_count": 0,
                "rare_value_count": 0,
                "typo_rule_count": 0,
                "pattern_rule_count": 0,
                "schema_rule_count": 0,
                "movie_typed_rule_count": 0,
                "movie_consistency_rule_count": 0,
            }
    return out


def best_counter_value(counter: Counter, prefer_imdb_id: bool = False) -> Optional[str]:
    if not counter:
        return None
    items = counter.most_common()
    if prefer_imdb_id:
        for v, _ in items:
            if looks_like_imdb_id(v):
                return v
    return items[0][0] if items else None


def generate_metadata_candidate_from_row_values(column: str, row_values: Dict[str, Any], global_dict: Dict[str, Any]) -> Optional[Tuple[str, str, float, str]]:
    """Used both for whitelist augmentation and candidate generation."""
    col = str(column).strip()
    cl = col.lower()
    dirty = canonical_empty_text(row_values.get(col, ""))
    name = canonical_empty_text(row_values.get("Name", ""))
    year = normalize_year_candidate(row_values.get("Year", ""))
    movie_id = canonical_empty_text(row_values.get("Id", ""))

    def lookup(map_name, key, source, score, prefer_imdb=False):
        counter = global_dict.get(map_name, {}).get(key, Counter())
        v = best_counter_value(counter, prefer_imdb_id=prefer_imdb)
        if v and not value_equal_for_column(col, dirty, v):
            return (v, source, score, f"metadata_lookup={map_name}")
        return None

    if cl == "id":
        cand = lookup("name_year_to_id", (clean_key(name), clean_key(year)), "name_year_to_id", 0.92, prefer_imdb=True)
        if cand: return cand
        cand = lookup("name_to_id", clean_key(name), "name_to_id", 0.80, prefer_imdb=True)
        if cand: return cand

    if cl == "description":
        cand = lookup("name_year_to_description", (clean_key(name), clean_key(year)), "name_year_to_description", 0.86)
        if cand: return cand
        cand = lookup("name_to_description", clean_key(name), "name_to_description", 0.72)
        if cand: return cand

    if cl == "director":
        cand = lookup("name_year_to_director", (clean_key(name), clean_key(year)), "name_year_to_director", 0.86)
        if cand: return cand
        cand = lookup("name_to_director", clean_key(name), "name_to_director", 0.76)
        if cand: return cand

    if cl == "creator":
        cand = lookup("name_year_to_creator", (clean_key(name), clean_key(year)), "name_year_to_creator", 0.84)
        if cand: return cand
        cand = lookup("name_to_creator", clean_key(name), "name_to_creator", 0.74)
        if cand: return cand

    if cl == "name":
        fixed = normalize_movie_name_candidate(dirty)
        if fixed and not value_equal_for_column(col, dirty, fixed):
            return (fixed, "canonical_name_restore", 0.72, "deaccent_ampersand_case")
        cand = lookup("id_to_name", clean_key(movie_id), "id_to_name", 0.82)
        if cand: return cand

    return None


def build_movies_direct_and_metadata_suspicious_cells(dirty_table_df: Optional[pd.DataFrame], global_dict: Dict[str, Any]) -> pd.DataFrame:
    """Augment whitelist with high-confidence direct and metadata candidate cells."""
    rows = []
    if dirty_table_df is None or dirty_table_df.empty:
        return pd.DataFrame(rows)

    for rid, r in dirty_table_df.iterrows():
        row_values = {str(c): canonical_empty_text(r[c]) for c in dirty_table_df.columns}

        # Direct high-precision patterns.
        direct_specs = [
            ("Year", repair_year_triple_to_middle_candidate, "movies_direct_year_triple_to_middle"),
            ("Release Date", repair_release_date_wide_limited_candidate, "movies_direct_release_date_wide_limited"),
            ("Genre", repair_genre_conservative_candidate, "movies_direct_genre_conservative_mapping"),
            ("Duration", repair_duration_single_format_candidate, "movies_direct_duration_single_format"),
            ("RatingValue", repair_rating_value_single_format_candidate, "movies_direct_rating_value_single_format"),
            ("Name", normalize_movie_name_candidate, "movies_direct_name_normalize"),
        ]
        for col, fn, reason in direct_specs:
            if col not in dirty_table_df.columns:
                continue
            dirty = canonical_empty_text(r[col])
            cand = fn(dirty)
            if cand and not value_equal_for_column(col, dirty, cand):
                rows.append({"row_id": int(rid), "column": col, "value": dirty, "candidate_value": cand, "candidate_source": reason, "suspicious_reason": reason})

        # Metadata mapping patterns only when a concrete candidate exists.
        for col in ["Id", "Description", "Creator", "Director", "Name"]:
            if col not in dirty_table_df.columns:
                continue
            dirty = canonical_empty_text(r[col])
            cand_info = generate_metadata_candidate_from_row_values(col, row_values, global_dict)
            if cand_info:
                cand, source, score, extra = cand_info
                if cand and not value_equal_for_column(col, dirty, cand):
                    rows.append({"row_id": int(rid), "column": col, "value": dirty, "candidate_value": cand, "candidate_source": source, "suspicious_reason": source})

    if not rows:
        return pd.DataFrame(rows)
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["row_id", "column"], keep="first").reset_index(drop=True)
    return df

def generate_movie_consistency_candidates(row_id, column, dirty_value, consistency_map):
    info = consistency_map.get((int(row_id), str(column)))
    if not info:
        return []
    val = canonical_empty_text(info.get("candidate_value", ""))
    if val in {"", "empty"} or value_equal_for_column(column, dirty_value, val):
        return []
    conf = safe_float(info.get("confidence", 0.0), 0.0)
    support = safe_int(info.get("support_count", 0), 0)
    if support < MIN_CONSISTENCY_SUPPORT_COUNT:
        return []
    score = CONSISTENCY_STRONG_SCORE if conf >= 0.80 else CONSISTENCY_MEDIUM_SCORE
    score = max(score, safe_float(info.get("score", 0.0), 0.0))
    extra = f"consistency_name={info.get('consistency_name', '')}|lhs_key={info.get('lhs_key', '')}|confidence={conf:.4f}|margin={safe_float(info.get('margin', 0.0), 0.0):.4f}|support_count={support}"
    return [(val, "movie_metadata_consistency", score, extra)]


def build_context_map_rich(context_records, global_dict):
    context_map = {}
    for obj in context_records:
        if "row_id" not in obj or "column" not in obj:
            continue
        try:
            row_id = int(obj["row_id"])
        except Exception:
            continue
        column = str(obj["column"])
        value = canonical_empty_text(obj.get("value", ""))
        row_values = {}
        movie_bundle = {}
        rc = obj.get("row_context", {})
        if isinstance(rc, dict):
            rv = rc.get("row_values", {})
            if isinstance(rv, dict):
                row_values = {str(k): canonical_empty_text(v) for k, v in rv.items()}
            mb = rc.get("movie_metadata_bundle", {})
            if isinstance(mb, dict):
                movie_bundle = mb
        conflict_context = obj.get("conflict_context", {})
        rule_expected_values = []
        expected_values_from_rules = []
        alternative_values_from_column = []
        counter_fields = ["strong_rule_count", "candidate_generation_rule_count", "fd_like_count", "context_rule_count", "global_rule_count", "rare_value_count", "typo_rule_count", "pattern_rule_count", "schema_rule_count", "movie_typed_rule_count", "movie_consistency_rule_count"]
        counters = {k: 0 for k in counter_fields}
        if isinstance(conflict_context, dict):
            for vr in conflict_context.get("violated_rules_detailed", []) or []:
                if isinstance(vr, dict):
                    ev = canonical_empty_text(vr.get("expected_value", None))
                    if ev not in {"", "empty"}: rule_expected_values.append(ev)
            for item in conflict_context.get("expected_values_from_rules", []) or []:
                if isinstance(item, dict):
                    ev = canonical_empty_text(item.get("value", None))
                    if ev not in {"", "empty"}: expected_values_from_rules.append(ev)
            for key in ["alternative_values_from_column", "all_alternative_values"]:
                for item in conflict_context.get(key, []) or []:
                    if isinstance(item, dict):
                        val = canonical_empty_text(item.get("value", None))
                        if val not in {"", "empty"}:
                            alternative_values_from_column.append({"value": val, "count": safe_int(item.get("count", 1), 1), "ratio": safe_float(item.get("ratio", 0.0), 0.0)})
            for k in counters:
                counters[k] = safe_int(conflict_context.get(k, obj.get(k, 0)), 0)
        sim_ctx = obj.get("similarity_context", {})
        neighbor_majority_value = ""
        neighbor_majority_ratio = 0.0
        similar_row_target_values = []
        if isinstance(sim_ctx, dict):
            neighbor_majority_value = canonical_empty_text(sim_ctx.get("neighbor_majority_value", ""))
            neighbor_majority_ratio = safe_float(sim_ctx.get("neighbor_majority_ratio", 0.0), 0.0)
            for item in sim_ctx.get("topk_similar_rows", [])[:TOP_K_SIMILAR_ROW_VALUES]:
                if isinstance(item, dict):
                    tv = canonical_empty_text(item.get("target_column_value", ""))
                    if tv not in {"", "empty"}: similar_row_target_values.append(tv)
        col_ctx = obj.get("column_context", {})
        column_top_values = []
        if isinstance(col_ctx, dict):
            for item in col_ctx.get("top_values_with_counts", [])[:TOP_K_COLUMN_VALUES]:
                if isinstance(item, dict):
                    val = canonical_empty_text(item.get("value", ""))
                    if val not in {"", "empty"}:
                        fixed = normalize_by_column(column, val)
                        column_top_values.append({"value": fixed if fixed is not None else val, "raw_value": val, "count": safe_int(item.get("count", 0), 0), "ratio": safe_float(item.get("ratio", 0.0), 0.0)})
        if not column_top_values:
            counter = global_dict["column_value_counter"].get(column, Counter())
            total = max(sum(counter.values()), 1)
            for val, cnt in counter.most_common(TOP_K_COLUMN_VALUES):
                column_top_values.append({"value": val, "raw_value": val, "count": cnt, "ratio": cnt / total})
        bayes_ctx = obj.get("bayesian_context", {})
        prior_error_probability = obj.get("prior_error_probability", 0.0)
        posterior_error_probability = obj.get("posterior_error_probability", 0.0)
        if isinstance(bayes_ctx, dict):
            prior_error_probability = bayes_ctx.get("prior_error_probability", prior_error_probability)
            posterior_error_probability = bayes_ctx.get("posterior_error_probability", posterior_error_probability)
        context_map[(row_id, column)] = {
            "row_id": row_id, "column": column, "value": value, "row_values": row_values, "movie_metadata_bundle": movie_bundle,
            "suggested_correct_value": canonical_empty_text(obj.get("suggested_correct_value", "")),
            "semantic_type": obj.get("semantic_type", col_ctx.get("semantic_type") if isinstance(col_ctx, dict) else None),
            "detected_type": obj.get("detected_type", col_ctx.get("detected_type") if isinstance(col_ctx, dict) else None),
            "violation_count": obj.get("violation_count", None), "conflict_score": obj.get("conflict_score", None),
            "value_frequency": obj.get("value_frequency", None), "value_frequency_rank": obj.get("value_frequency_rank", None),
            "parsed_year": obj.get("parsed_year", col_ctx.get("parsed_year") if isinstance(col_ctx, dict) else None),
            "parsed_release_year": obj.get("parsed_release_year", None), "parsed_release_country": obj.get("parsed_release_country", None),
            "duration_minutes": obj.get("duration_minutes", col_ctx.get("duration_minutes") if isinstance(col_ctx, dict) else None),
            "rating_value_float": obj.get("rating_value_float", col_ctx.get("rating_value_float") if isinstance(col_ctx, dict) else None),
            "count_value_int": obj.get("count_value_int", col_ctx.get("count_value_int") if isinstance(col_ctx, dict) else None),
            "review_total": obj.get("review_total", None), "list_token_count": obj.get("list_token_count", col_ctx.get("list_token_count") if isinstance(col_ctx, dict) else None),
            "neighbor_majority_value": neighbor_majority_value, "neighbor_majority_ratio": neighbor_majority_ratio, "similar_row_target_values": similar_row_target_values,
            "column_top_values": column_top_values, "alternative_values_from_column": alternative_values_from_column,
            "rule_expected_values": rule_expected_values, "expected_values_from_rules": expected_values_from_rules,
            "prior_error_probability": prior_error_probability, "posterior_error_probability": posterior_error_probability,
            "main_rule_type": obj.get("main_rule_type", conflict_context.get("main_rule_type") if isinstance(conflict_context, dict) else None),
            "main_usage_role": obj.get("main_usage_role", conflict_context.get("main_usage_role") if isinstance(conflict_context, dict) else None),
            **counters,
        }
    return context_map


def merge_flat_into_context_map(context_map, flat_records):
    for obj in flat_records:
        if "row_id" not in obj or "column" not in obj:
            continue
        try:
            row_id = int(obj["row_id"])
        except Exception:
            continue
        column = str(obj["column"])
        key = (row_id, column)
        if key not in context_map:
            context_map[key] = {"row_id": row_id, "column": column, "value": canonical_empty_text(obj.get("value", "")), "row_values": {}, "movie_metadata_bundle": {}, "suggested_correct_value": "", "similar_row_target_values": [], "column_top_values": [], "alternative_values_from_column": [], "rule_expected_values": [], "expected_values_from_rules": []}
        ctx = context_map[key]
        for field in ["semantic_type", "detected_type", "violation_count", "conflict_score", "value_frequency", "value_frequency_rank", "parsed_year", "parsed_release_year", "parsed_release_country", "duration_minutes", "rating_value_float", "count_value_int", "review_total", "list_token_count", "neighbor_majority_value", "neighbor_majority_ratio", "prior_error_probability", "posterior_error_probability", "main_rule_type", "main_usage_role", "strong_rule_count", "candidate_generation_rule_count", "fd_like_count", "context_rule_count", "global_rule_count", "rare_value_count", "typo_rule_count", "pattern_rule_count", "schema_rule_count", "movie_typed_rule_count", "movie_consistency_rule_count"]:
            if field in obj and ctx.get(field) in [None, "", [], "empty"]:
                ctx[field] = obj.get(field)


# ============================================================
# 6. 候选生成
# ============================================================

def add_candidate(candidate_dict, value, source, score, extra=None):
    if value is None:
        return
    value = str(value).strip()
    if value == "":
        return
    key = norm_text(value)
    if key == "":
        return
    if key not in candidate_dict:
        candidate_dict[key] = {"candidate_value": value, "candidate_sources": set(), "source_score": float(score), "is_from_rule": 0, "is_from_neighbor": 0, "is_from_column_top": 0, "is_from_similarity": 0, "is_from_hint": 0, "is_from_dictionary": 0, "is_from_pattern_restore": 0, "is_from_movie_consistency": 0, "extra_info": []}
    item = candidate_dict[key]
    item["candidate_sources"].add(source)
    item["source_score"] = max(item["source_score"], float(score))
    if source in {"rule_expected_value", "expected_values_from_rules"}:
        item["is_from_rule"] = 1
    elif source in {"neighbor_majority", "similar_row_target"}:
        item["is_from_neighbor"] = 1
    elif source in {"column_top_value", "alternative_values_from_column"}:
        item["is_from_column_top"] = 1
    elif source == "similar_column_value":
        item["is_from_similarity"] = 1
    elif source == "suggested_correct_value":
        item["is_from_hint"] = 1
    elif source == "movie_metadata_consistency":
        item["is_from_movie_consistency"] = 1; item["is_from_dictionary"] = 1
    elif source.startswith("canonical_") or source.endswith("_format_restore"):
        item["is_from_pattern_restore"] = 1
    else:
        # dictionary-style movie mappings
        item["is_from_dictionary"] = 1 if ("_to_" in source or source in {"cast_prefix_to_actors", "release_date_country_consistency", "release_date_year_consistency"}) else item["is_from_dictionary"]
    if extra is not None:
        item["extra_info"].append(str(extra))


def counter_top(counter, topk=5):
    return counter.most_common(topk) if counter else []


def movie_row_value(ctx, col):
    rv = ctx.get("row_values", {}) or {}
    mb = ctx.get("movie_metadata_bundle", {}) or {}
    if col in rv:
        return canonical_empty_text(rv[col])
    if col in mb:
        return canonical_empty_text(mb[col])
    return ""


def add_counter_candidates(out, counter, source, base_score, column, dirty_value, extra_prefix="", topk=5):
    for v, cnt in counter_top(counter, topk=topk):
        if not v or value_equal_for_column(column, dirty_value, v):
            continue
        out.append((v, source, base_score, f"{extra_prefix}cnt={cnt}"))
    return out


def generate_movie_column_candidates(dirty_value, ctx, global_dict):
    cands = []
    column = str(ctx.get("column", ""))
    cl = column.lower()
    dirty = canonical_empty_text(dirty_value)
    row_id = movie_row_value(ctx, "Id")
    name = movie_row_value(ctx, "Name")
    year = normalize_year_candidate(movie_row_value(ctx, "Year"))
    release_date = movie_row_value(ctx, "Release Date")
    release_parts = parse_release_date_parts(release_date)
    release_country = release_parts.get("country") if release_parts else None
    director = movie_row_value(ctx, "Director")
    creator = movie_row_value(ctx, "Creator")
    actors = movie_row_value(ctx, "Actors")
    genre = movie_row_value(ctx, "Genre")
    locations = movie_row_value(ctx, "Filming Locations")
    rating_value = normalize_rating_value_candidate(movie_row_value(ctx, "RatingValue"))
    rating_count = normalize_count_candidate(movie_row_value(ctx, "RatingCount"))
    country = normalize_country_candidate(movie_row_value(ctx, "Country"))
    fixed = normalize_by_column(column, dirty)
    if fixed and not value_equal_for_column(column, dirty, fixed):
        cands.append((fixed, f"canonical_{cl.replace(' ', '_')}_restore", 0.93, "normalize_from_dirty"))
    if cl == "duration":
        direct_duration = repair_duration_single_format_candidate(dirty)
        if direct_duration and not value_equal_for_column(column, dirty, direct_duration):
            cands.append((direct_duration, "movies_direct_duration_single_format", 0.88, "single_duration_format"))
        if ctx.get("duration_minutes") not in [None, "", "empty"] and not pd.isna(ctx.get("duration_minutes")):
            dur = normalize_duration_candidate(ctx.get("duration_minutes"))
            if dur: cands.append((dur, "canonical_duration_restore", 0.88, "ctx_duration_minutes"))
        cands = add_counter_candidates(cands, global_dict["id_to_duration"].get(clean_key(row_id), Counter()), "id_to_duration", 0.95, column, dirty, "id|")
        cands = add_counter_candidates(cands, global_dict["name_year_to_duration"].get((clean_key(name), clean_key(year)), Counter()), "name_year_to_duration", 0.88, column, dirty, "name_year|")
        cands = add_counter_candidates(cands, global_dict["director_genre_to_duration"].get((clean_key(director), clean_key(genre)), Counter()), "director_genre_to_duration", 0.78, column, dirty, "director_genre|")
        cands = add_counter_candidates(cands, global_dict["director_location_to_duration"].get((clean_key(director), clean_key(locations)), Counter()), "director_location_to_duration", 0.80, column, dirty, "director_location|")
        cands = add_counter_candidates(cands, global_dict["rating_value_count_to_duration"].get((clean_key(rating_value), clean_key(rating_count)), Counter()), "rating_value_count_to_duration", 0.82, column, dirty, "rating_pair|")
        cands = add_counter_candidates(cands, global_dict["rating_count_genre_to_duration"].get((clean_key(rating_count), clean_key(genre)), Counter()), "rating_count_genre_to_duration", 0.76, column, dirty, "rating_genre|")
    elif cl == "year":
        if ctx.get("parsed_year") not in [None, "", "empty"] and not pd.isna(ctx.get("parsed_year")):
            y = normalize_year_candidate(ctx.get("parsed_year"))
            if y: cands.append((y, "canonical_year_restore", 0.92, "ctx_parsed_year"))
        if release_parts and release_parts.get("year"):
            y = str(release_parts["year"])
            if not value_equal_for_column(column, dirty, y): cands.append((y, "release_date_year_consistency", 0.86, "release_date_year"))
        cands = add_counter_candidates(cands, global_dict["id_to_year"].get(clean_key(row_id), Counter()), "id_to_year", 0.96, column, dirty, "id|")
        cands = add_counter_candidates(cands, global_dict["name_to_year"].get(clean_key(name), Counter()), "name_to_year", 0.82, column, dirty, "name|")
    elif cl == "release date":
        fixed_date = normalize_release_date_candidate(dirty)
        if fixed_date and not value_equal_for_column(column, dirty, fixed_date): cands.append((fixed_date, "canonical_release_date_restore", 0.92, "normalize_release_date"))
        cands = add_counter_candidates(cands, global_dict["id_to_release_date"].get(clean_key(row_id), Counter()), "id_to_release_date", 0.95, column, dirty, "id|")
        cands = add_counter_candidates(cands, global_dict["name_year_to_release_date"].get((clean_key(name), clean_key(year)), Counter()), "name_year_to_release_date", 0.88, column, dirty, "name_year|")
        cands = add_counter_candidates(cands, global_dict["name_to_release_date"].get(clean_key(name), Counter()), "name_to_release_date", 0.78, column, dirty, "name|")
    elif cl == "country":
        if release_country and not value_equal_for_column(column, dirty, release_country): cands.append((release_country, "release_date_country_consistency", 0.84, "release_date_country"))
        cands = add_counter_candidates(cands, global_dict["name_year_to_country"].get((clean_key(name), clean_key(year)), Counter()), "name_year_to_country", 0.86, column, dirty, "name_year|")
        cands = add_counter_candidates(cands, global_dict["name_to_country"].get(clean_key(name), Counter()), "name_to_country", 0.76, column, dirty, "name|")
    elif cl == "language":
        cands = add_counter_candidates(cands, global_dict["country_to_language"].get(clean_key(country), Counter()), "country_to_language", 0.62, column, dirty, "country|")
    elif cl == "ratingvalue":
        direct_rating = repair_rating_value_single_format_candidate(dirty)
        if direct_rating and not value_equal_for_column(column, dirty, direct_rating):
            cands.append((direct_rating, "movies_direct_rating_value_single_format", 0.88, "single_rating_format"))
        if ctx.get("rating_value_float") not in [None, "", "empty"] and not pd.isna(ctx.get("rating_value_float")):
            rv = normalize_rating_value_candidate(ctx.get("rating_value_float"))
            if rv: cands.append((rv, "canonical_rating_value_restore", 0.92, "ctx_rating_value_float"))
        cands = add_counter_candidates(cands, global_dict["id_to_rating_value"].get(clean_key(row_id), Counter()), "id_to_rating_value", 0.94, column, dirty, "id|")
        cands = add_counter_candidates(cands, global_dict["name_year_to_rating_value"].get((clean_key(name), clean_key(year)), Counter()), "name_year_to_rating_value", 0.85, column, dirty, "name_year|")
    elif cl == "ratingcount":
        if ctx.get("count_value_int") not in [None, "", "empty"] and not pd.isna(ctx.get("count_value_int")):
            cnt = normalize_count_candidate(ctx.get("count_value_int"))
            if cnt: cands.append((cnt, "canonical_rating_count_restore", 0.90, "ctx_count_value_int"))
        cands = add_counter_candidates(cands, global_dict["id_to_rating_count"].get(clean_key(row_id), Counter()), "id_to_rating_count", 0.94, column, dirty, "id|")
        cands = add_counter_candidates(cands, global_dict["name_year_to_rating_count"].get((clean_key(name), clean_key(year)), Counter()), "name_year_to_rating_count", 0.84, column, dirty, "name_year|")
    elif cl == "reviewcount":
        if ctx.get("review_total") not in [None, "", "empty"] and not pd.isna(ctx.get("review_total")):
            rv = normalize_review_count_candidate(ctx.get("review_total"))
            if rv: cands.append((rv, "canonical_review_count_restore", 0.88, "ctx_review_total"))
        cands = add_counter_candidates(cands, global_dict["rating_count_genre_to_review_count"].get((clean_key(rating_count), clean_key(genre)), Counter()), "rating_count_genre_to_review_count", 0.70, column, dirty, "rating_genre|")
    elif cl == "id":
        meta = generate_metadata_candidate_from_row_values(column, ctx.get("row_values", {}) or {}, global_dict)
        if meta:
            cands.append(meta)
        cands = add_counter_candidates(cands, global_dict["name_year_to_id"].get((clean_key(name), clean_key(year)), Counter()), "name_year_to_id", 0.92, column, dirty, "name_year|")
        cands = add_counter_candidates(cands, global_dict["name_to_id"].get(clean_key(name), Counter()), "name_to_id", 0.80, column, dirty, "name|")
    elif cl == "description":
        meta = generate_metadata_candidate_from_row_values(column, ctx.get("row_values", {}) or {}, global_dict)
        if meta:
            cands.append(meta)
        cands = add_counter_candidates(cands, global_dict["name_year_to_description"].get((clean_key(name), clean_key(year)), Counter()), "name_year_to_description", 0.86, column, dirty, "name_year|", topk=3)
        cands = add_counter_candidates(cands, global_dict["name_to_description"].get(clean_key(name), Counter()), "name_to_description", 0.72, column, dirty, "name|", topk=3)
    elif cl == "name":
        fixed_name = normalize_movie_name_candidate(dirty)
        if fixed_name and not value_equal_for_column(column, dirty, fixed_name):
            cands.append((fixed_name, "canonical_name_restore", 0.72, "deaccent_ampersand_case"))
        meta = generate_metadata_candidate_from_row_values(column, ctx.get("row_values", {}) or {}, global_dict)
        if meta:
            cands.append(meta)
    elif cl == "director":
        cands = add_counter_candidates(cands, global_dict["name_year_to_director"].get((clean_key(name), clean_key(year)), Counter()), "name_year_to_director", 0.86, column, dirty, "name_year|")
        cands = add_counter_candidates(cands, global_dict["name_to_director"].get(clean_key(name), Counter()), "name_to_director", 0.76, column, dirty, "name|")
    elif cl == "creator":
        cands = add_counter_candidates(cands, global_dict["name_year_to_creator"].get((clean_key(name), clean_key(year)), Counter()), "name_year_to_creator", 0.84, column, dirty, "name_year|")
        cands = add_counter_candidates(cands, global_dict["name_to_creator"].get(clean_key(name), Counter()), "name_to_creator", 0.74, column, dirty, "name|")
    elif cl == "actors":
        cands = add_counter_candidates(cands, global_dict["name_to_actors"].get(clean_key(name), Counter()), "name_to_actors", 0.82, column, dirty, "name|")
        cast = movie_row_value(ctx, "Cast")
        cast_tokens = split_list_tokens(cast)
        if len(cast_tokens) >= 3:
            candidate = ",".join(cast_tokens[:3])
            if not value_equal_for_column(column, dirty, candidate): cands.append((candidate, "cast_prefix_to_actors", 0.78, "first_3_cast_tokens"))
    elif cl == "cast":
        cands = add_counter_candidates(cands, global_dict["name_to_cast"].get(clean_key(name), Counter()), "name_to_cast", 0.82, column, dirty, "name|")
        cands = add_counter_candidates(cands, global_dict["actors_to_cast"].get((clean_key(actors),), Counter()), "actors_to_cast", 0.74, column, dirty, "actors|")
    elif cl == "genre":
        genre_direct = repair_genre_conservative_candidate(dirty)
        if genre_direct and not value_equal_for_column(column, dirty, genre_direct):
            cands.append((genre_direct, "movies_direct_genre_conservative_mapping", 0.94, "rt_style_genre_mapping"))
        fixed_list = normalize_list_value(dirty)
        if fixed_list and norm_text(fixed_list) != norm_text(dirty): cands.append((fixed_list, "canonical_genre_list_restore", 0.70, "normalize_list"))
    elif cl == "filming locations":
        fixed_country = normalize_country_candidate(dirty)
        if fixed_country and norm_text(fixed_country) != norm_text(dirty): cands.append((fixed_country, "canonical_location_country_restore", 0.54, "weak_location_country"))
    return cands


# ============================================================
# 7. 主流程
# ============================================================

def main():
    if not os.path.exists(ERROR_PRED_FILE):
        raise FileNotFoundError(f"ERROR_PRED_FILE 不存在: {ERROR_PRED_FILE}")
    if not os.path.exists(CONTEXT_FILE):
        raise FileNotFoundError(f"CONTEXT_FILE 不存在: {CONTEXT_FILE}")
    pred_df = pd.read_csv(ERROR_PRED_FILE, encoding="utf-8-sig")
    if not {"row_id", "column"}.issubset(set(pred_df.columns)):
        raise ValueError("错误位置文件必须至少包含 row_id,column")
    pred_df["row_id"] = pd.to_numeric(pred_df["row_id"], errors="raise").astype(int)
    pred_df["column"] = pred_df["column"].astype(str).str.strip()
    error_df = pick_error_positions(pred_df)
    keep_cols = ["row_id", "column"] + (["value"] if "value" in error_df.columns else [])
    error_df = error_df[keep_cols].drop_duplicates(subset=["row_id", "column"], keep="first").reset_index(drop=True)
    print(f"[INFO] 预测错误单元格数量: {len(error_df)}")
    context_records = load_context_records(CONTEXT_FILE)
    print(f"[INFO] 主上下文记录数: {len(context_records)}")
    dirty_table_df = None
    if path_exists_nonempty(DIRTY_TABLE_CSV):
        dirty_table_df = pd.read_csv(DIRTY_TABLE_CSV, encoding="utf-8-sig")
        print(f"[INFO] 已读取 Movies 原始脏表: {DIRTY_TABLE_CSV}, shape={dirty_table_df.shape}")
    else:
        print(f"[WARN] DIRTY_TABLE_CSV 不存在，已跳过: {DIRTY_TABLE_CSV}")
    global_dict = build_global_dictionaries(context_records, dirty_table_df=dirty_table_df)
    movie_consistency_map = {}
    movie_consistency_df = pd.DataFrame()
    consistency_suspicious_df = pd.DataFrame()
    if ENABLE_MOVIE_CONSISTENCY:
        movie_consistency_map, movie_consistency_df, consistency_suspicious_df = build_movie_metadata_consistency(dirty_table_df)
        print(f"[INFO] movie consistency entries: {len(movie_consistency_df)}")
        print(f"[INFO] movie consistency suspicious cells: {len(consistency_suspicious_df)}")
        if not movie_consistency_df.empty:
            movie_consistency_df.to_csv(resolve_output_path(OUTPUT_MOVIE_CONSISTENCY_CSV), index=False, encoding="utf-8-sig")
        if not consistency_suspicious_df.empty:
            consistency_suspicious_df.to_csv(resolve_output_path(OUTPUT_CONSISTENCY_SUSPICIOUS_CELLS_CSV), index=False, encoding="utf-8-sig")
    context_map = build_context_map_rich(context_records, global_dict)

    # New: fallback contexts from the dirty table for augmented cells that did not appear in LLM contexts.
    # This is critical for Movies because Id/Description/Genre/Release Date often have no detector context.
    dirty_context_map = build_dirty_table_context_map(dirty_table_df, global_dict)
    for k, v in dirty_context_map.items():
        if k not in context_map:
            context_map[k] = v

    if str(FALLBACK_CONTEXT_FLAT_FILE).strip():
        if path_exists_nonempty(FALLBACK_CONTEXT_FLAT_FILE):
            flat_records = load_context_records(FALLBACK_CONTEXT_FLAT_FILE)
            print(f"[INFO] fallback flat 上下文记录数: {len(flat_records)}")
            merge_flat_into_context_map(context_map, flat_records)
        else:
            print(f"[WARN] FALLBACK_CONTEXT_FLAT_FILE 不存在，已跳过: {FALLBACK_CONTEXT_FLAT_FILE}")
    print(f"[INFO] 可索引上下文条数: {len(context_map)}")
    if ENABLE_MOVIE_AUGMENTED_WHITELIST:
        base_allowed = error_df[["row_id", "column"]].copy()
        if "value" in error_df.columns:
            base_allowed["value"] = error_df["value"]

        add_parts = []
        if not consistency_suspicious_df.empty:
            add_parts.append(
                consistency_suspicious_df[
                    consistency_suspicious_df["column"].astype(str).str.strip().isin(MOVIE_CONSISTENCY_ALLOW_COLUMNS)
                ][["row_id", "column", "value"]].copy()
            )

        direct_meta_suspicious_df = build_movies_direct_and_metadata_suspicious_cells(dirty_table_df, global_dict)
        if not direct_meta_suspicious_df.empty:
            direct_meta_suspicious_df.to_csv(resolve_output_path("movie_direct_metadata_suspicious_cells.csv"), index=False, encoding="utf-8-sig")
            add_parts.append(direct_meta_suspicious_df[["row_id", "column", "value"]].copy())
            print(f"[INFO] direct/metadata suspicious cells: {len(direct_meta_suspicious_df)}")
            print("[INFO] direct/metadata suspicious by column:")
            print(direct_meta_suspicious_df.groupby("column").size().sort_values(ascending=False).head(30).to_string())
        else:
            print("[INFO] direct/metadata suspicious cells: 0")

        before_allowed = len(base_allowed.drop_duplicates(subset=["row_id", "column"]))
        if add_parts:
            error_df = pd.concat([base_allowed] + add_parts, ignore_index=True, sort=False)
        else:
            error_df = base_allowed.copy()
        error_df["row_id"] = pd.to_numeric(error_df["row_id"], errors="raise").astype(int)
        error_df["column"] = error_df["column"].astype(str).str.strip()
        error_df = error_df.drop_duplicates(subset=["row_id", "column"], keep="first").reset_index(drop=True)
        after_allowed = len(error_df)
        print(f"[INFO] Movies augmented whitelist: {before_allowed} -> {after_allowed} (+{after_allowed - before_allowed})")
    error_df.to_csv(resolve_output_path(OUTPUT_AUGMENTED_ALLOWED_CELLS_CSV), index=False, encoding="utf-8-sig")
    expanded_rows = []
    grouped_rows = []
    missing_context_count = 0
    for _, row in error_df.iterrows():
        row_id = int(row["row_id"])
        column = str(row["column"]).strip()
        ctx = context_map.get((row_id, column), None)
        if ctx is None:
            missing_context_count += 1
            dirty_value = canonical_empty_text(row.get("value", ""))
            grouped_rows.append({"row_id": row_id, "column": column, "dirty_value": dirty_value, "candidate_count": 0, "candidates_joined": "", "sources_joined": "", "top_candidate": "", "top_candidate_score": None})
            continue
        dirty_value = canonical_empty_text(row.get("value", ctx.get("value", "")))
        if dirty_value in {"", "empty"}: dirty_value = canonical_empty_text(ctx.get("value", ""))
        candidate_dict = {}
        if INCLUDE_DIRTY_VALUE and dirty_value not in {"", "empty"}: add_candidate(candidate_dict, dirty_value, "dirty_value", 0.01)
        expected_counter = Counter()
        for ev in ctx.get("rule_expected_values", []) + ctx.get("expected_values_from_rules", []):
            ev = canonical_empty_text(ev)
            if ev in {"", "empty"} or value_equal_for_column(column, dirty_value, ev): continue
            fixed = normalize_by_column(column, ev)
            expected_counter[fixed if fixed is not None else ev] += 1
        total_votes = sum(expected_counter.values())
        for ev, cnt in expected_counter.items():
            add_candidate(candidate_dict, ev, "expected_values_from_rules", 0.76 + 0.20 * (cnt / max(total_votes, 1)), extra=f"rule_vote_count={cnt}")
        nmv = ctx.get("neighbor_majority_value", "")
        if nmv and nmv not in {"", "empty"} and not value_equal_for_column(column, dirty_value, nmv):
            fixed = normalize_by_column(column, nmv)
            nr = safe_float(ctx.get("neighbor_majority_ratio", 0.0), 0.0)
            add_candidate(candidate_dict, fixed if fixed is not None else nmv, "neighbor_majority", 0.72 + 0.18 * min(nr, 1.0), extra=f"neighbor_ratio={nr:.4f}")
        sim_counter = Counter()
        for v in ctx.get("similar_row_target_values", []):
            v = canonical_empty_text(v)
            if v in {"", "empty"} or value_equal_for_column(column, dirty_value, v): continue
            fixed = normalize_by_column(column, v)
            sim_counter[fixed if fixed is not None else v] += 1
        for v, cnt in sim_counter.items():
            add_candidate(candidate_dict, v, "similar_row_target", 0.66 + 0.14 * min(cnt, TOP_K_SIMILAR_ROW_VALUES) / max(TOP_K_SIMILAR_ROW_VALUES, 1), extra=f"count={cnt}")
        sv = ctx.get("suggested_correct_value", "")
        if sv and sv not in {"", "empty"} and not value_equal_for_column(column, dirty_value, sv):
            fixed = normalize_by_column(column, sv)
            add_candidate(candidate_dict, fixed if fixed is not None else sv, "suggested_correct_value", 0.93)
        alt_items = []
        alt_items.extend(ctx.get("column_top_values", [])[:TOP_K_COLUMN_VALUES])
        alt_items.extend(ctx.get("alternative_values_from_column", [])[:TOP_K_COLUMN_VALUES])
        dedup_alt = {}
        for item in alt_items:
            if not isinstance(item, dict): continue
            val = canonical_empty_text(item.get("value", ""))
            if val in {"", "empty"} or value_equal_for_column(column, dirty_value, val): continue
            fixed = normalize_by_column(column, val)
            val2 = fixed if fixed is not None else val
            k = norm_text(val2)
            prev = dedup_alt.get(k, {"value": val2, "count": 0, "ratio": 0.0})
            prev["count"] = max(prev["count"], safe_int(item.get("count", 0), 0))
            prev["ratio"] = max(prev["ratio"], safe_float(item.get("ratio", 0.0), 0.0))
            dedup_alt[k] = prev
        for item in dedup_alt.values():
            val = item["value"]; ratio = safe_float(item.get("ratio", 0.0), 0.0); sim = string_similarity(dirty_value, val)
            if is_movie_people_col(column) or is_movie_list_col(column, ctx.get("semantic_type")):
                top_score = 0.20 + 0.08 * min(ratio, 1.0); sim_score = 0.32 + 0.14 * sim + 0.04 * min(ratio, 1.0)
            elif str(column).lower() in {"duration", "ratingvalue", "ratingcount", "reviewcount", "year"}:
                top_score = 0.30 + 0.16 * min(ratio, 1.0); sim_score = 0.44 + 0.20 * sim + 0.08 * min(ratio, 1.0)
            else:
                top_score = 0.28 + 0.14 * min(ratio, 1.0); sim_score = 0.42 + 0.18 * sim + 0.08 * min(ratio, 1.0)
            add_candidate(candidate_dict, val, "column_top_value", top_score, extra=f"ratio={ratio:.4f}")
            if sim >= MIN_STRING_SIM_FOR_COLUMN_VALUE:
                add_candidate(candidate_dict, val, "similar_column_value", sim_score, extra=f"ratio={ratio:.4f}|sim={sim:.4f}")
        if ENABLE_MOVIE_CONSISTENCY:
            for val, source, score, extra in generate_movie_consistency_candidates(row_id, column, dirty_value, movie_consistency_map):
                add_candidate(candidate_dict, val, source, score, extra=extra)
        if ENABLE_COLUMN_SPECIFIC_GENERATION:
            for val, source, score, extra in generate_movie_column_candidates(dirty_value, ctx, global_dict):
                add_candidate(candidate_dict, val, source, score, extra=extra)
        for k in list(candidate_dict.keys()):
            if value_equal_for_column(column, candidate_dict[k]["candidate_value"], dirty_value): candidate_dict.pop(k, None)
        final_candidates = []
        for _, item in candidate_dict.items():
            cand = item["candidate_value"]
            sim = string_similarity(dirty_value, cand)
            source_count = len(item["candidate_sources"])
            support_bonus = 0.035 * max(source_count - 1, 0)
            semantic_bonus = 0.0
            if normalize_by_column(column, cand) is not None: semantic_bonus += 0.04
            cl = column.lower()
            if cl == "duration" and normalize_duration_candidate(cand) is not None: semantic_bonus += 0.08
            elif cl == "year" and normalize_year_candidate(cand) is not None: semantic_bonus += 0.08
            elif cl == "release date" and normalize_release_date_candidate(cand) is not None: semantic_bonus += 0.07
            elif cl == "ratingvalue" and normalize_rating_value_candidate(cand) is not None: semantic_bonus += 0.07
            elif cl in {"ratingcount", "reviewcount"} and normalize_count_candidate(cand) is not None: semantic_bonus += 0.05
            elif item["is_from_movie_consistency"]: semantic_bonus += 0.08
            item["edit_similarity_to_dirty"] = round(sim, 6)
            item["final_score"] = round(item["source_score"] + 0.12 * sim + support_bonus + semantic_bonus, 6)
            item["candidate_sources"] = sorted(list(item["candidate_sources"]))
            item["candidate_source_joined"] = "|".join(item["candidate_sources"])
            final_candidates.append(item)
        final_candidates = sorted(final_candidates, key=lambda x: (-x["final_score"], -x["is_from_rule"], -x["is_from_movie_consistency"], -x["is_from_dictionary"], -x["is_from_pattern_restore"], -x["is_from_neighbor"], x["candidate_value"]))[:MAX_CANDIDATES_PER_CELL]
        for rank_idx, item in enumerate(final_candidates, start=1):
            expanded_rows.append({
                "row_id": row_id, "column": column, "dirty_value": dirty_value, "candidate_rank": rank_idx,
                "candidate_value": item["candidate_value"], "candidate_source": item["candidate_source_joined"], "source_score": item["source_score"], "final_score": item["final_score"],
                "edit_similarity_to_dirty": item["edit_similarity_to_dirty"], "is_from_rule": item["is_from_rule"], "is_from_neighbor": item["is_from_neighbor"], "is_from_column_top": item["is_from_column_top"],
                "is_from_similarity": item["is_from_similarity"], "is_from_hint": item["is_from_hint"], "is_from_dictionary": item["is_from_dictionary"], "is_from_pattern_restore": item["is_from_pattern_restore"],
                "is_from_movie_consistency": item["is_from_movie_consistency"], "source_count": len(item["candidate_sources"]), "extra_info": " || ".join(item["extra_info"][:20]),
                "semantic_type": ctx.get("semantic_type", None), "detected_type": ctx.get("detected_type", None), "violation_count": ctx.get("violation_count", None), "conflict_score": ctx.get("conflict_score", None),
                "value_frequency": ctx.get("value_frequency", None), "value_frequency_rank": ctx.get("value_frequency_rank", None), "neighbor_majority_value": ctx.get("neighbor_majority_value", None), "neighbor_majority_ratio": ctx.get("neighbor_majority_ratio", None),
                "prior_error_probability": ctx.get("prior_error_probability", None), "posterior_error_probability": ctx.get("posterior_error_probability", None), "main_rule_type": ctx.get("main_rule_type", None), "main_usage_role": ctx.get("main_usage_role", None),
                "parsed_year": ctx.get("parsed_year", None), "parsed_release_year": ctx.get("parsed_release_year", None), "parsed_release_country": ctx.get("parsed_release_country", None), "duration_minutes": ctx.get("duration_minutes", None),
                "rating_value_float": ctx.get("rating_value_float", None), "count_value_int": ctx.get("count_value_int", None), "review_total": ctx.get("review_total", None), "list_token_count": ctx.get("list_token_count", None),
                "strong_rule_count": ctx.get("strong_rule_count", None), "candidate_generation_rule_count": ctx.get("candidate_generation_rule_count", None), "fd_like_count": ctx.get("fd_like_count", None), "context_rule_count": ctx.get("context_rule_count", None),
                "global_rule_count": ctx.get("global_rule_count", None), "rare_value_count": ctx.get("rare_value_count", None), "typo_rule_count": ctx.get("typo_rule_count", None), "pattern_rule_count": ctx.get("pattern_rule_count", None), "schema_rule_count": ctx.get("schema_rule_count", None),
                "movie_typed_rule_count": ctx.get("movie_typed_rule_count", None), "movie_consistency_rule_count": ctx.get("movie_consistency_rule_count", None),
                "movie_candidate_is_duration": int(normalize_duration_candidate(item["candidate_value"]) is not None), "movie_candidate_is_year": int(normalize_year_candidate(item["candidate_value"]) is not None),
                "movie_candidate_is_release_date": int(normalize_release_date_candidate(item["candidate_value"]) is not None), "movie_candidate_is_country": int(normalize_country_candidate(item["candidate_value"]) is not None),
                "movie_candidate_is_rating_value": int(normalize_rating_value_candidate(item["candidate_value"]) is not None), "movie_candidate_is_count": int(normalize_count_candidate(item["candidate_value"]) is not None),
                "movie_candidate_is_review_count": int(normalize_review_count_candidate(item["candidate_value"]) is not None), "movie_column_is_time": int(is_movie_time_col(column, ctx.get("semantic_type"))),
                "movie_column_is_rating": int(is_movie_rating_col(column, ctx.get("semantic_type"))), "movie_column_is_people": int(is_movie_people_col(column)), "movie_column_is_list": int(is_movie_list_col(column, ctx.get("semantic_type"))),
            })
        grouped_rows.append({"row_id": row_id, "column": column, "dirty_value": dirty_value, "candidate_count": len(final_candidates), "candidates_joined": " || ".join([x["candidate_value"] for x in final_candidates]), "sources_joined": " || ".join([x["candidate_source_joined"] for x in final_candidates]), "top_candidate": final_candidates[0]["candidate_value"] if final_candidates else "", "top_candidate_score": final_candidates[0]["final_score"] if final_candidates else None})
    expanded_df = pd.DataFrame(expanded_rows)
    grouped_df = pd.DataFrame(grouped_rows)
    if not expanded_df.empty: expanded_df = expanded_df.sort_values(["row_id", "column", "candidate_rank"]).reset_index(drop=True)
    if not grouped_df.empty: grouped_df = grouped_df.sort_values(["row_id", "column"]).reset_index(drop=True)
    out_expanded = resolve_output_path(OUTPUT_CANDIDATES_EXPANDED_CSV)
    out_grouped = resolve_output_path(OUTPUT_CANDIDATES_GROUPED_CSV)
    expanded_df.to_csv(out_expanded, index=False, encoding="utf-8-sig")
    grouped_df.to_csv(out_grouped, index=False, encoding="utf-8-sig")
    print(f"[INFO] 缺失上下文的错误 cell 数量: {missing_context_count}")
    print(f"[INFO] 候选展开文件已保存: {out_expanded}")
    print(f"[INFO] 候选汇总文件已保存: {out_grouped}")
    if not grouped_df.empty:
        print("\n[INFO] 候选数量按列统计:")
        print(grouped_df.groupby("column")["candidate_count"].agg(["count", "mean", "max"]).sort_values("count", ascending=False).head(30).to_string())
        print("\n[INFO] 前 20 个错误 cell 的候选示例：")
        print(grouped_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()

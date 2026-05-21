import json
import math
import re
from collections import Counter
from typing import Dict, Any, List, Optional, Set

import numpy as np
import pandas as pd


# ============================================================
# 1. 用户配置区
# ============================================================

INPUT_TABLE_CSV = "/mnt/mydata/dq/projects/Splittree/movies/movies_dirty.csv"
INPUT_SHRUNK_CSV = "candidate_cells_movies.csv"

OUTPUT_JSONL = "candidate_llm_contexts_movies.jsonl"
OUTPUT_CSV = "candidate_llm_contexts_flat_movies.csv"

MISSING_TOKENS = {
    "", "empty", "null", "none", "nan", "na", "n/a", "missing", "unk",
    "unknown", "not available", "not applicable", "tbd", "na."
}

NUMERIC_RATIO_THRESHOLD = 0.90
CATEGORICAL_UNIQUE_COUNT_THRESHOLD = 50
CATEGORICAL_UNIQUE_RATIO_THRESHOLD = 0.08
ID_LIKE_UNIQUE_RATIO_THRESHOLD = 0.98
CODE_LIKE_AVG_LEN_THRESHOLD = 24

TOP_K_COLUMN_VALUES = 10
TOP_K_ALT_VALUES = 10
TOP_K_SIMILAR_ROWS = 5
TOP_K_ROW_FIELDS_FOR_LLM = 16

ID_COLUMNS = {"Id"}
TEXT_LONG_COLUMNS = {"Description", "Filming Locations"}
PEOPLE_COLUMNS = {"Director", "Creator", "Actors", "Cast"}
LIST_COLUMNS = {"Creator", "Actors", "Cast", "Language", "Country", "Genre"}
NUMERIC_LIKE_COLUMNS = {"Year", "Duration", "RatingValue", "RatingCount", "ReviewCount"}
DATE_COLUMNS = {"Release Date"}

MOVIE_YEAR_MIN = 1880
MOVIE_YEAR_MAX = 2035
RATING_MIN = 0.0
RATING_MAX = 10.0
DURATION_MIN_MINUTES = 1
DURATION_MAX_MINUTES = 600

# 贝叶斯启发式权重
W_RULE = 2.0
W_RARITY = 1.0
W_PATTERN = 0.8
W_NEIGHBOR = 1.2
W_MOVIE_TYPED = 1.1


# ============================================================
# 2. 基础函数
# ============================================================

def normalize_value(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.lower() in MISSING_TOKENS:
        return None
    return s


def normalize_series(s: pd.Series) -> pd.Series:
    return s.map(normalize_value)


def maybe_drop_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [c for c in df.columns if not str(c).lower().startswith("unnamed:")]
    return df[keep_cols].copy()


def detect_basic_pattern(value) -> str:
    if value is None:
        return "NULL"
    if not isinstance(value, str):
        value = str(value)

    parts = []
    prev_type = None
    cnt = 0

    for ch in value:
        if ch.isdigit():
            cur = "D"
        elif ch.isalpha():
            cur = "L"
        elif ch in {" ", "-", "_", "/", ".", ",", "(", ")", "%", ":", ";", "&", "'", "+", "[", "]", "{", "}", '"'}:
            cur = "S"
        else:
            cur = "X"

        if cur == prev_type:
            cnt += 1
        else:
            if prev_type is not None:
                parts.append(f"{prev_type}{cnt}")
            prev_type = cur
            cnt = 1

    if prev_type is not None:
        parts.append(f"{prev_type}{cnt}")

    return "".join(parts)


def try_parse_numeric_series(series: pd.Series):
    cleaned = (
        series.dropna()
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
    )
    numeric = pd.to_numeric(cleaned, errors="coerce")
    success_ratio = numeric.notna().mean() if len(cleaned) > 0 else 0.0
    return numeric, success_ratio


def try_parse_one_numeric(value):
    if value is None:
        return None
    s = str(value).replace(",", "").replace("%", "")
    try:
        return float(s)
    except Exception:
        return None


def text_jaccard_similarity(a: str, b: str) -> float:
    if a is None or b is None:
        return 0.0
    sa = set(str(a).lower().split())
    sb = set(str(b).lower().split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def safe_logit(p: float) -> float:
    eps = 1e-6
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def value_frequency_dict(series: pd.Series, topk=20):
    cnt = Counter(series.dropna().tolist())
    total = len(series.dropna())
    out = []
    for k, v in cnt.most_common(topk):
        out.append({
            "value": k,
            "count": v,
            "ratio": round(v / total, 6) if total > 0 else 0.0
        })
    return out


def charset_features(value: str) -> Dict[str, float]:
    if value is None or len(value) == 0:
        return {
            "length": 0,
            "digit_ratio": 0.0,
            "alpha_ratio": 0.0,
            "symbol_ratio": 0.0,
            "upper_ratio": 0.0,
            "lower_ratio": 0.0,
            "space_ratio": 0.0,
        }

    n = len(str(value))
    digits = sum(ch.isdigit() for ch in str(value))
    alphas = sum(ch.isalpha() for ch in str(value))
    uppers = sum(ch.isupper() for ch in str(value))
    lowers = sum(ch.islower() for ch in str(value))
    spaces = sum(ch.isspace() for ch in str(value))
    symbols = n - digits - alphas - spaces

    return {
        "length": n,
        "digit_ratio": round(digits / n, 6),
        "alpha_ratio": round(alphas / n, 6),
        "symbol_ratio": round(symbols / n, 6),
        "upper_ratio": round(uppers / n, 6),
        "lower_ratio": round(lowers / n, 6),
        "space_ratio": round(spaces / n, 6),
    }


def parse_pipe_list(s: Any, sep: str = "|") -> List[str]:
    if pd.isna(s) or s is None:
        return []
    s = str(s).strip()
    if not s:
        return []
    return [x.strip() for x in s.split(sep) if x.strip()]


def parse_reason_list(s: Any) -> List[str]:
    if pd.isna(s) or s is None:
        return []
    s = str(s).strip()
    if not s:
        return []
    return [x.strip() for x in s.split("||") if x.strip()]


def extract_expected_value_from_reason(reason: str):
    if not reason:
        return None

    patterns = [
        ("suggests rhs=", ", observed="),
        ("suggests ", ", observed="),
        ("dominant=", ", observed="),
        ("expected_value=", ","),
        ("expected ", ","),
    ]

    for left, right in patterns:
        if left in reason and right in reason:
            try:
                a = reason.index(left) + len(left)
                b = reason.index(right, a)
                value = reason[a:b].strip()
                if left == "suggests " and "=" in value:
                    value = value.split("=", 1)[1].strip()
                return value if value else None
            except Exception:
                pass

    return None


# ============================================================
# 3. movies 专用解析函数
# ============================================================

def split_comma_list(s: Optional[str]) -> List[str]:
    if s is None:
        return []
    txt = str(s).strip()
    if txt == "":
        return []
    parts = [p.strip().lower() for p in txt.split(",")]
    parts = [p for p in parts if p and p not in MISSING_TOKENS]
    return parts


def parse_year_value(s: Optional[str]) -> Optional[int]:
    if s is None:
        return None
    txt = str(s).strip().lower()
    m = re.search(r"\b(18\d{2}|19\d{2}|20\d{2})\b", txt)
    if not m:
        return None
    year = int(m.group(1))
    if MOVIE_YEAR_MIN <= year <= MOVIE_YEAR_MAX:
        return year
    return None


def parse_release_date(s: Optional[str]) -> Optional[Dict[str, Any]]:
    if s is None:
        return None
    txt = str(s).strip().lower()
    if txt == "":
        return None

    country = None
    m_country = re.search(r"\(([^)]+)\)", txt)
    if m_country:
        country = m_country.group(1).strip().lower()

    year = parse_year_value(txt)
    if year is None:
        return None

    months = {
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

    day = None
    month = None

    m_full = re.search(
        r"\b(\d{1,2})\s+([a-z]+)\s+(18\d{2}|19\d{2}|20\d{2})\b",
        txt
    )
    if m_full:
        day = int(m_full.group(1))
        month = months.get(m_full.group(2))
    else:
        m_month_year = re.search(r"\b([a-z]+)\s+(18\d{2}|19\d{2}|20\d{2})\b", txt)
        if m_month_year:
            month = months.get(m_month_year.group(1))

    if day is not None and not (1 <= day <= 31):
        return None
    if month is not None and not (1 <= month <= 12):
        return None

    return {
        "year": year,
        "month": month,
        "day": day,
        "country": country,
    }


def parse_duration_minutes(s: Optional[str]) -> Optional[int]:
    if s is None:
        return None
    txt = str(s).strip().lower()
    m = re.search(r"\b(\d{1,4})\s*(min|mins|minute|minutes)\b", txt)
    if m:
        mins = int(m.group(1))
    else:
        m2 = re.fullmatch(r"\d{1,4}", txt)
        if not m2:
            return None
        mins = int(txt)

    if DURATION_MIN_MINUTES <= mins <= DURATION_MAX_MINUTES:
        return mins
    return None


def parse_rating_value(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    txt = str(s).strip().lower().replace(",", "")
    try:
        val = float(txt)
    except Exception:
        return None
    if RATING_MIN <= val <= RATING_MAX:
        return val
    return None


def parse_count_value(s: Optional[str]) -> Optional[int]:
    if s is None:
        return None
    txt = str(s).strip().lower().replace(",", "")
    m = re.search(r"\b(\d+)\b", txt)
    if not m:
        return None
    val = int(m.group(1))
    if val >= 0:
        return val
    return None


def parse_review_count(s: Optional[str]) -> Optional[Dict[str, int]]:
    if s is None:
        return None
    txt = str(s).strip().lower()
    if txt == "":
        return None

    user_count = None
    critic_count = None

    m_user = re.search(r"([\d,]+)\s*user", txt)
    if m_user:
        user_count = int(m_user.group(1).replace(",", ""))

    m_critic = re.search(r"([\d,]+)\s*critic", txt)
    if m_critic:
        critic_count = int(m_critic.group(1).replace(",", ""))

    if user_count is None and critic_count is None:
        return None

    return {
        "user": user_count if user_count is not None else 0,
        "critic": critic_count if critic_count is not None else 0,
    }


def total_review_count(review_count_str: Optional[str]) -> Optional[int]:
    parsed = parse_review_count(review_count_str)
    if parsed is None:
        return None
    return int(parsed.get("user", 0)) + int(parsed.get("critic", 0))


def looks_like_person_or_people_list(s: Optional[str]) -> bool:
    if s is None:
        return False
    parts = split_comma_list(s)
    if len(parts) == 0:
        return False
    return any(re.search(r"[a-z]", p) is not None for p in parts)


def looks_like_category_list(s: Optional[str]) -> bool:
    if s is None:
        return False
    parts = split_comma_list(s)
    if len(parts) == 0:
        return False
    return all(re.search(r"[a-z]", p) is not None for p in parts)


def has_mojibake_or_control_chars(s: Optional[str]) -> bool:
    if s is None:
        return False
    txt = str(s)
    if "�" in txt or "\ufffd" in txt or "��" in txt:
        return True
    for ch in txt:
        if ord(ch) < 32 and ch not in {"\t", "\n", "\r"}:
            return True
    return False


def list_token_count_bucket(s: Optional[str]) -> str:
    parts = split_comma_list(s)
    n = len(parts)
    if n == 0:
        return "empty_list"
    if n == 1:
        return "single"
    if n <= 3:
        return "short_list"
    if n <= 8:
        return "medium_list"
    return "long_list"


def text_length_bucket(s: Optional[str]) -> str:
    if s is None:
        return "null"
    n = len(str(s).strip())
    if n == 0:
        return "empty"
    if n <= 20:
        return "very_short"
    if n <= 80:
        return "short"
    if n <= 200:
        return "medium"
    if n <= 500:
        return "long"
    return "very_long"


def list_overlap_ratio(a: Optional[str], b: Optional[str]) -> float:
    a_set = set(split_comma_list(a))
    b_set = set(split_comma_list(b))
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / max(1, len(a_set))


# ============================================================
# 4. 加载数据
# ============================================================

def load_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df = maybe_drop_unnamed_columns(df)
    for col in df.columns:
        df[col] = normalize_series(df[col])
    return df


def load_shrunk_candidates(path: str) -> List[Dict[str, Any]]:
    df = pd.read_csv(path, dtype=str)
    items = []

    for _, row in df.iterrows():
        item = {
            "row_id": int(row["row_id"]),
            "column": row["column"],
            "value": normalize_value(row["value"]),
            "violation_count": int(float(row["violation_count"])) if pd.notna(row["violation_count"]) else 0,
            "conflict_score": float(row["conflict_score"]) if pd.notna(row["conflict_score"]) else 0.0,
            "rule_ids": parse_pipe_list(row.get("rule_ids")),
            "rule_types": parse_pipe_list(row.get("rule_types")),
            "priorities": parse_pipe_list(row.get("priorities")),
            "usage_roles": parse_pipe_list(row.get("usage_roles")),
            "reasons": parse_reason_list(row.get("reasons")),
        }

        violated_rules = []
        max_len = max(
            len(item["rule_ids"]),
            len(item["rule_types"]),
            len(item["priorities"]),
            len(item["usage_roles"]),
            len(item["reasons"]),
            0
        )

        for i in range(max_len):
            reason = item["reasons"][i] if i < len(item["reasons"]) else None
            violated_rules.append({
                "rule_id": item["rule_ids"][i] if i < len(item["rule_ids"]) else None,
                "rule_type": item["rule_types"][i] if i < len(item["rule_types"]) else None,
                "priority": item["priorities"][i] if i < len(item["priorities"]) else None,
                "usage_role": item["usage_roles"][i] if i < len(item["usage_roles"]) else None,
                "reason": reason,
                "expected_value": extract_expected_value_from_reason(reason),
            })

        item["violated_rules"] = violated_rules
        items.append(item)

    return items


# ============================================================
# 5. 列类型与列统计
# ============================================================

def is_id_like_column(col: str, non_null: pd.Series, unique_ratio: float, numeric_ratio: float, avg_len: float) -> bool:
    if col in ID_COLUMNS:
        return True
    if len(non_null) == 0:
        return False
    if unique_ratio >= ID_LIKE_UNIQUE_RATIO_THRESHOLD and avg_len <= CODE_LIKE_AVG_LEN_THRESHOLD:
        return True
    if unique_ratio >= ID_LIKE_UNIQUE_RATIO_THRESHOLD and numeric_ratio >= NUMERIC_RATIO_THRESHOLD:
        return True
    return False


def is_code_like_column(non_null: pd.Series, unique_ratio: float, numeric_ratio: float, avg_len: float,
                        top_patterns: List[Dict[str, Any]]) -> bool:
    if len(non_null) == 0:
        return False
    if avg_len <= CODE_LIKE_AVG_LEN_THRESHOLD and unique_ratio > CATEGORICAL_UNIQUE_RATIO_THRESHOLD:
        if top_patterns and top_patterns[0]["ratio"] >= 0.7:
            return True
    if avg_len <= 15 and numeric_ratio >= 0.95 and unique_ratio < ID_LIKE_UNIQUE_RATIO_THRESHOLD:
        return True
    return False


def build_movie_column_extra_profile(col: str, non_null: pd.Series) -> Dict[str, Any]:
    extra = {}

    if col == "Year":
        parsed = non_null.map(parse_year_value)
        extra["year_parse_ratio"] = round(float(parsed.notna().mean()), 6) if len(parsed) > 0 else 0.0
        extra["year_min"] = int(parsed.dropna().min()) if parsed.notna().any() else None
        extra["year_max"] = int(parsed.dropna().max()) if parsed.notna().any() else None

    if col == "Release Date":
        parsed = non_null.map(parse_release_date)
        extra["release_date_parse_ratio"] = round(float(parsed.notna().mean()), 6) if len(parsed) > 0 else 0.0
        years = [x["year"] for x in parsed.dropna().tolist()]
        countries = [x.get("country") for x in parsed.dropna().tolist() if x.get("country")]
        extra["release_year_counter"] = dict(Counter(years).most_common(30))
        extra["release_country_counter"] = dict(Counter(countries).most_common(30))

    if col == "Duration":
        parsed = non_null.map(parse_duration_minutes)
        extra["duration_parse_ratio"] = round(float(parsed.notna().mean()), 6) if len(parsed) > 0 else 0.0
        extra["duration_min"] = int(parsed.dropna().min()) if parsed.notna().any() else None
        extra["duration_max"] = int(parsed.dropna().max()) if parsed.notna().any() else None

    if col == "RatingValue":
        parsed = non_null.map(parse_rating_value)
        extra["rating_parse_ratio"] = round(float(parsed.notna().mean()), 6) if len(parsed) > 0 else 0.0
        extra["rating_min"] = float(parsed.dropna().min()) if parsed.notna().any() else None
        extra["rating_max"] = float(parsed.dropna().max()) if parsed.notna().any() else None

    if col == "RatingCount":
        parsed = non_null.map(parse_count_value)
        extra["rating_count_parse_ratio"] = round(float(parsed.notna().mean()), 6) if len(parsed) > 0 else 0.0

    if col == "ReviewCount":
        parsed = non_null.map(parse_review_count)
        extra["review_count_parse_ratio"] = round(float(parsed.notna().mean()), 6) if len(parsed) > 0 else 0.0

    if col in LIST_COLUMNS:
        token_counts = non_null.map(lambda x: len(split_comma_list(x)))
        extra["list_parse_ratio"] = round(float((token_counts > 0).mean()), 6) if len(token_counts) > 0 else 0.0
        extra["list_token_count_top"] = dict(Counter(token_counts.tolist()).most_common(10))
        token_counter = Counter()
        for v in non_null.tolist():
            token_counter.update(split_comma_list(v))
        extra["top_tokens"] = [{"token": k, "count": v} for k, v in token_counter.most_common(30)]

    if col in TEXT_LONG_COLUMNS or col in PEOPLE_COLUMNS or col == "Name":
        length_counter = Counter(non_null.map(text_length_bucket).tolist())
        extra["length_bucket_counter"] = dict(length_counter.most_common())
        extra["mojibake_count"] = int(non_null.map(has_mojibake_or_control_chars).sum())

    return extra


def analyze_columns(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    profiles = {}

    for col in df.columns:
        s = df[col]
        non_null = s.dropna()

        missing_rate = 1.0 - (len(non_null) / len(df) if len(df) > 0 else 0.0)
        unique_count = non_null.nunique()
        unique_ratio = unique_count / len(non_null) if len(non_null) > 0 else 0.0

        top_values = value_frequency_dict(non_null, topk=TOP_K_COLUMN_VALUES)
        freq_counter = Counter(non_null.tolist())

        lengths = non_null.map(lambda x: len(str(x)))
        avg_len = float(lengths.mean()) if len(lengths) > 0 else 0.0

        length_stats = {
            "min": int(lengths.min()) if len(lengths) > 0 else None,
            "max": int(lengths.max()) if len(lengths) > 0 else None,
            "mean": round(avg_len, 6) if len(lengths) > 0 else None,
            "std": round(float(lengths.std(ddof=0)), 6) if len(lengths) > 1 else 0.0,
            "q1": round(float(lengths.quantile(0.25)), 6) if len(lengths) > 0 else None,
            "median": round(float(lengths.quantile(0.5)), 6) if len(lengths) > 0 else None,
            "q3": round(float(lengths.quantile(0.75)), 6) if len(lengths) > 0 else None,
        }

        numeric_vals, numeric_ratio = try_parse_numeric_series(non_null)

        patterns = non_null.map(detect_basic_pattern)
        pattern_counter = Counter(patterns.tolist())
        total_patterns = sum(pattern_counter.values())

        pattern_list = [
            {"pattern": p, "count": c, "ratio": round(c / total_patterns, 6)}
            for p, c in pattern_counter.most_common(10)
        ]

        if col in ID_COLUMNS:
            detected_type = "id"
            semantic_type = "id_like"
        elif col == "Year":
            detected_type = "year_like"
            semantic_type = "numeric"
        elif col == "Release Date":
            detected_type = "date_like"
            semantic_type = "date_like"
        elif col == "Duration":
            detected_type = "duration_like"
            semantic_type = "duration_like"
        elif col == "RatingValue":
            detected_type = "rating_like"
            semantic_type = "numeric"
        elif col in {"RatingCount", "ReviewCount"}:
            detected_type = "count_like"
            semantic_type = "numeric"
        elif col in LIST_COLUMNS:
            detected_type = "list_like"
            semantic_type = "categorical_list"
        else:
            if numeric_ratio >= NUMERIC_RATIO_THRESHOLD:
                detected_type = "numeric"
            elif unique_count <= CATEGORICAL_UNIQUE_COUNT_THRESHOLD or unique_ratio <= CATEGORICAL_UNIQUE_RATIO_THRESHOLD:
                detected_type = "categorical"
            else:
                detected_type = "text"

            if is_id_like_column(col, non_null, unique_ratio, numeric_ratio, avg_len):
                semantic_type = "id_like"
            elif is_code_like_column(non_null, unique_ratio, numeric_ratio, avg_len, pattern_list):
                semantic_type = "code_like"
            elif detected_type == "numeric":
                semantic_type = "numeric"
            elif detected_type == "categorical":
                semantic_type = "categorical"
            else:
                semantic_type = "text"

        numeric_stats = None
        if semantic_type == "numeric" or detected_type in {"year_like", "rating_like", "count_like"}:
            nums = numeric_vals.dropna()
            if len(nums) > 0:
                numeric_stats = {
                    "min": float(nums.min()),
                    "max": float(nums.max()),
                    "mean": float(nums.mean()),
                    "std": float(nums.std(ddof=0)) if len(nums) > 1 else 0.0,
                    "q1": float(nums.quantile(0.25)),
                    "median": float(nums.quantile(0.5)),
                    "q3": float(nums.quantile(0.75)),
                }

        profile = {
            "column": col,
            "missing_rate": round(missing_rate, 6),
            "non_null_count": int(len(non_null)),
            "unique_count": int(unique_count),
            "unique_ratio": round(unique_ratio, 6),
            "detected_type": detected_type,
            "semantic_type": semantic_type,
            "top_values": top_values,
            "value_counter": freq_counter,
            "length_stats": length_stats,
            "numeric_ratio": round(float(numeric_ratio), 6),
            "numeric_stats": numeric_stats,
            "patterns": pattern_list,
            "pattern_counter": pattern_counter,
            "dominant_pattern": pattern_list[0]["pattern"] if pattern_list else None,
            "dominant_pattern_ratio": pattern_list[0]["ratio"] if pattern_list else None,
        }

        profile.update(build_movie_column_extra_profile(col, non_null))
        profiles[col] = profile

    return profiles


# ============================================================
# 6. 行上下文
# ============================================================

def build_movie_metadata_bundle(row: Dict[str, Any]) -> Dict[str, Any]:
    year_val = row.get("Year")
    release_val = row.get("Release Date")
    duration_val = row.get("Duration")
    rating_value = row.get("RatingValue")
    rating_count = row.get("RatingCount")
    review_count = row.get("ReviewCount")

    parsed_year = parse_year_value(year_val)
    parsed_release = parse_release_date(release_val)
    parsed_duration = parse_duration_minutes(duration_val)
    parsed_rating_value = parse_rating_value(rating_value)
    parsed_rating_count = parse_count_value(rating_count)
    parsed_review_count = parse_review_count(review_count)

    release_year_gap = None
    if parsed_year is not None and parsed_release is not None:
        release_year_gap = parsed_release["year"] - parsed_year

    review_total = total_review_count(review_count)
    review_to_rating_ratio = None
    if review_total is not None and parsed_rating_count is not None and parsed_rating_count > 0:
        review_to_rating_ratio = round(review_total / parsed_rating_count, 6)

    actors_val = row.get("Actors")
    cast_val = row.get("Cast")
    actors_cast_overlap = None
    if actors_val is not None and cast_val is not None:
        actors_cast_overlap = round(list_overlap_ratio(actors_val, cast_val), 6)

    return {
        "Id": row.get("Id"),
        "Name": row.get("Name"),
        "Year": year_val,
        "parsed_year": parsed_year,
        "Release Date": release_val,
        "parsed_release_date": parsed_release,
        "release_year_gap": release_year_gap,
        "Director": row.get("Director"),
        "Creator": row.get("Creator"),
        "Actors": actors_val,
        "Cast": cast_val,
        "actors_cast_overlap": actors_cast_overlap,
        "Language": row.get("Language"),
        "Country": row.get("Country"),
        "Duration": duration_val,
        "duration_minutes": parsed_duration,
        "RatingValue": rating_value,
        "rating_value_float": parsed_rating_value,
        "RatingCount": rating_count,
        "rating_count_int": parsed_rating_count,
        "ReviewCount": review_count,
        "review_count_parsed": parsed_review_count,
        "review_total": review_total,
        "review_to_rating_ratio": review_to_rating_ratio,
        "Genre": row.get("Genre"),
        "Filming Locations": row.get("Filming Locations"),
    }


def summarize_row_context(df: pd.DataFrame, row_id: int, col_profiles: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    row = df.iloc[row_id].to_dict()

    values = [v for v in row.values() if v is not None]
    missing_count = sum(v is None for v in row.values())
    non_missing_count = len(values)
    lengths = [len(str(v)) for v in values]

    semantic_type_counts = Counter()
    for col, v in row.items():
        if v is None:
            continue
        semantic_type_counts[col_profiles[col]["semantic_type"]] += 1

    row_patterns = {col: detect_basic_pattern(v) if v is not None else "NULL" for col, v in row.items()}
    row_items = list(row.items())
    row_values_for_llm = row_items[:TOP_K_ROW_FIELDS_FOR_LLM]

    movie_metadata_bundle = build_movie_metadata_bundle(row)

    return {
        "row_values": row,
        "row_values_for_llm": row_values_for_llm,
        "row_patterns": row_patterns,
        "missing_count": int(missing_count),
        "non_missing_count": int(non_missing_count),
        "avg_value_length": round(float(np.mean(lengths)), 6) if lengths else None,
        "max_value_length": int(max(lengths)) if lengths else None,
        "min_value_length": int(min(lengths)) if lengths else None,
        "semantic_type_counts": dict(semantic_type_counts),
        "movie_metadata_bundle": movie_metadata_bundle,
    }


# ============================================================
# 7. 列上下文
# ============================================================

def summarize_column_context(col: str, value: Any, col_profiles: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    prof = col_profiles[col]
    counter = prof["value_counter"]

    value_freq = counter.get(value, 0)
    rank = None
    if value is not None:
        sorted_vals = counter.most_common()
        for idx, (v, c) in enumerate(sorted_vals, start=1):
            if v == value:
                rank = idx
                break

    current_pattern = detect_basic_pattern(value) if value is not None else "NULL"

    parsed_year = parse_year_value(value) if col == "Year" else None
    parsed_release_date = parse_release_date(value) if col == "Release Date" else None
    duration_minutes = parse_duration_minutes(value) if col == "Duration" else None
    rating_value_float = parse_rating_value(value) if col == "RatingValue" else None
    count_value_int = parse_count_value(value) if col in {"RatingCount"} else None
    review_count_parsed = parse_review_count(value) if col == "ReviewCount" else None
    list_tokens = split_comma_list(value) if col in LIST_COLUMNS else []
    list_token_count = len(list_tokens) if col in LIST_COLUMNS else None

    return {
        "column": col,
        "detected_type": prof["detected_type"],
        "semantic_type": prof["semantic_type"],
        "missing_rate": prof["missing_rate"],
        "unique_rate": prof["unique_ratio"],
        "unique_count": prof["unique_count"],
        "non_null_count": prof["non_null_count"],
        "value_frequency": int(value_freq),
        "value_frequency_rank": rank,
        "top_values_with_counts": prof["top_values"],
        "length_stats": prof["length_stats"],
        "numeric_ratio": prof["numeric_ratio"],
        "numeric_stats": prof["numeric_stats"],
        "current_value_pattern": current_pattern,
        "dominant_pattern": prof["dominant_pattern"],
        "dominant_pattern_ratio": prof["dominant_pattern_ratio"],
        "top_patterns_with_counts": prof["patterns"],
        "charset_features": charset_features(value),

        "parsed_year": parsed_year,
        "parsed_release_date": parsed_release_date,
        "duration_minutes": duration_minutes,
        "rating_value_float": rating_value_float,
        "count_value_int": count_value_int,
        "review_count_parsed": review_count_parsed,
        "list_tokens": list_tokens[:30],
        "list_token_count": list_token_count,
        "list_token_count_bucket": list_token_count_bucket(value) if col in LIST_COLUMNS else None,
        "text_length_bucket": text_length_bucket(value),
        "has_mojibake_or_control_chars": has_mojibake_or_control_chars(value),
        "movie_extra_profile": {
            k: v for k, v in prof.items()
            if k not in {
                "column", "missing_rate", "non_null_count", "unique_count", "unique_ratio",
                "detected_type", "semantic_type", "top_values", "value_counter", "length_stats",
                "numeric_ratio", "numeric_stats", "patterns", "pattern_counter",
                "dominant_pattern", "dominant_pattern_ratio"
            }
        },
    }


# ============================================================
# 8. 冲突信息
# ============================================================

MOVIE_TYPED_RULE_TYPES = {
    "movie_year_range",
    "release_date_format",
    "release_year_consistency",
    "duration_format",
    "rating_value_range",
    "rating_count_format",
    "review_count_format",
    "comma_list_format",
    "people_list_format",
    "mojibake_or_control_char",
    "actors_cast_overlap",
    "rating_value_count_coexistence",
    "review_rating_count_order",
}

MOVIE_CONSISTENCY_RULE_TYPES = {
    "release_year_consistency",
    "actors_cast_overlap",
    "rating_value_count_coexistence",
    "review_rating_count_order",
}


def summarize_conflict_info(candidate: Dict[str, Any], col_profiles: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    col = candidate["column"]
    current_value = candidate["value"]
    violated_rules = candidate.get("violated_rules", [])

    rule_type_counter = Counter()
    priority_counter = Counter()
    usage_role_counter = Counter()

    extracted_expected_values = []
    seen = set()

    for r in violated_rules:
        rt = r.get("rule_type")
        pr = r.get("priority")
        ur = r.get("usage_role")

        if rt:
            rule_type_counter[rt] += 1
        if pr:
            priority_counter[pr] += 1
        if ur:
            usage_role_counter[ur] += 1

        ev = r.get("expected_value")
        if ev is not None and ev != current_value and ev not in seen:
            extracted_expected_values.append({
                "source": "rule_reason_parse",
                "value": ev,
                "rule_id": r.get("rule_id"),
                "rule_type": rt,
                "priority": pr,
                "usage_role": ur,
                "reason": r.get("reason"),
            })
            seen.add(ev)

    column_alternative_values = []
    for item in col_profiles[col]["top_values"]:
        v = item["value"]
        if v != current_value and v not in seen:
            column_alternative_values.append({
                "source": "column_top_value",
                "value": v,
                "count": item["count"]
            })
            seen.add(v)
        if len(column_alternative_values) >= TOP_K_ALT_VALUES:
            break

    main_rule_type = rule_type_counter.most_common(1)[0][0] if rule_type_counter else "none"
    main_usage_role = usage_role_counter.most_common(1)[0][0] if usage_role_counter else "none"

    strong_rule_count = usage_role_counter.get("strong_rule", 0)
    candidate_generation_rule_count = usage_role_counter.get("candidate_generation_rule", 0)

    fd_like_count = (
        rule_type_counter.get("functional_dependency", 0)
        + rule_type_counter.get("soft_functional_dependency", 0)
    )

    context_rule_count = (
        rule_type_counter.get("dominant_value_by_context", 0)
        + rule_type_counter.get("dominant_value_by_context_pair", 0)
    )
    global_rule_count = rule_type_counter.get("global_dominant_value", 0)
    rare_value_count = rule_type_counter.get("rare_value", 0)
    typo_rule_count = rule_type_counter.get("dominant_value_typo", 0)

    pattern_rule_count = (
        rule_type_counter.get("pattern", 0)
        + rule_type_counter.get("rare_pattern", 0)
        + rule_type_counter.get("comma_list_format", 0)
        + rule_type_counter.get("people_list_format", 0)
    )

    schema_rule_count = (
        rule_type_counter.get("not_null", 0)
        + rule_type_counter.get("enumeration_domain", 0)
        + rule_type_counter.get("numeric_range", 0)
        + rule_type_counter.get("numeric_outlier_iqr", 0)
        + rule_type_counter.get("high_risk_missing", 0)
        + rule_type_counter.get("paired_missing_consistency", 0)
        + pattern_rule_count
    )

    movie_typed_rule_count = sum(rule_type_counter.get(rt, 0) for rt in MOVIE_TYPED_RULE_TYPES)
    movie_consistency_rule_count = sum(rule_type_counter.get(rt, 0) for rt in MOVIE_CONSISTENCY_RULE_TYPES)

    return {
        "has_rule_violation": len(violated_rules) > 0,
        "violation_count": len(violated_rules),
        "violated_rules_detailed": violated_rules,
        "rule_ids": [r.get("rule_id") for r in violated_rules],
        "rule_types": [r.get("rule_type") for r in violated_rules],
        "priorities": [r.get("priority") for r in violated_rules],
        "usage_roles": [r.get("usage_role") for r in violated_rules],

        "rule_type_counter": dict(rule_type_counter),
        "priority_counter": dict(priority_counter),
        "usage_role_counter": dict(usage_role_counter),

        "main_rule_type": main_rule_type,
        "main_usage_role": main_usage_role,
        "strong_rule_count": strong_rule_count,
        "candidate_generation_rule_count": candidate_generation_rule_count,
        "fd_like_count": fd_like_count,
        "context_rule_count": context_rule_count,
        "global_rule_count": global_rule_count,

        "rare_value_count": rare_value_count,
        "typo_rule_count": typo_rule_count,
        "pattern_rule_count": pattern_rule_count,
        "schema_rule_count": schema_rule_count,
        "movie_typed_rule_count": movie_typed_rule_count,
        "movie_consistency_rule_count": movie_consistency_rule_count,

        "expected_values_from_rules": extracted_expected_values,
        "alternative_values_from_column": column_alternative_values,
        "all_alternative_values": extracted_expected_values + column_alternative_values
    }


# ============================================================
# 9. 相似信息
# ============================================================

def single_cell_similarity(a, b, semantic_type: str):
    if a is None and b is None:
        return None
    if a is None or b is None:
        return 0.0

    if semantic_type == "numeric":
        an = try_parse_one_numeric(a)
        bn = try_parse_one_numeric(b)
        if an is None or bn is None:
            return 0.0
        denom = max(abs(an), abs(bn), 1.0)
        return max(0.0, 1.0 - abs(an - bn) / denom)

    if semantic_type == "date_like":
        ad = parse_release_date(a)
        bd = parse_release_date(b)
        if ad is None or bd is None:
            if detect_basic_pattern(a) == detect_basic_pattern(b):
                return 0.2
            return 0.0
        year_diff = abs(ad["year"] - bd["year"])
        year_sim = max(0.0, 1.0 - year_diff / 20.0)
        country_sim = 0.2 if ad.get("country") and ad.get("country") == bd.get("country") else 0.0
        return min(1.0, year_sim + country_sim)

    if semantic_type == "duration_like":
        av = parse_duration_minutes(a)
        bv = parse_duration_minutes(b)
        if av is None or bv is None:
            return 0.0
        denom = max(abs(av), abs(bv), 1.0)
        return max(0.0, 1.0 - abs(av - bv) / denom)

    if semantic_type == "categorical_list":
        a_set = set(split_comma_list(a))
        b_set = set(split_comma_list(b))
        if not a_set and not b_set:
            return 1.0
        if not a_set or not b_set:
            return 0.0
        return len(a_set & b_set) / len(a_set | b_set)

    if semantic_type in {"categorical", "code_like", "id_like"}:
        if a == b:
            return 1.0
        if detect_basic_pattern(a) == detect_basic_pattern(b):
            return 0.2
        return 0.0

    if a == b:
        return 1.0
    jac = text_jaccard_similarity(a, b)
    if detect_basic_pattern(a) == detect_basic_pattern(b):
        jac = max(jac, 0.2)
    return jac


def row_similarity(target_row: pd.Series, other_row: pd.Series, col_profiles: Dict[str, Dict[str, Any]],
                   ignore_col: str = None) -> float:
    score = 0.0
    count = 0

    for c in target_row.index:
        if ignore_col is not None and c == ignore_col:
            continue
        sim = single_cell_similarity(target_row[c], other_row[c], col_profiles[c]["semantic_type"])
        if sim is None:
            continue
        score += sim
        count += 1

    return round(score / count, 6) if count > 0 else 0.0


def summarize_similarity_info(df: pd.DataFrame, row_id: int, target_col: str,
                              col_profiles: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    target_row = df.iloc[row_id]
    sims = []

    for i in range(len(df)):
        if i == row_id:
            continue
        sim = row_similarity(target_row, df.iloc[i], col_profiles, ignore_col=target_col)
        sims.append((i, sim))

    sims.sort(key=lambda x: x[1], reverse=True)
    topk = sims[:TOP_K_SIMILAR_ROWS]

    similar_rows = []
    target_col_values = []

    for idx, sim in topk:
        row_dict = df.iloc[idx].to_dict()
        target_val = row_dict.get(target_col)
        similar_rows.append({
            "row_id": int(idx),
            "similarity": sim,
            "target_column_value": target_val,
            "row_values": row_dict
        })
        target_col_values.append(target_val)

    value_dist = Counter([v for v in target_col_values if v is not None])
    majority_value = value_dist.most_common(1)[0][0] if value_dist else None
    majority_ratio = round(value_dist.most_common(1)[0][1] / len(target_col_values), 6) if value_dist and len(target_col_values) > 0 else None

    return {
        "topk_similar_rows": similar_rows,
        "neighbor_value_distribution": dict(value_dist),
        "neighbor_majority_value": majority_value,
        "neighbor_majority_ratio": majority_ratio
    }


# ============================================================
# 10. 贝叶斯先验/后验
# ============================================================

def build_candidate_stats(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_column = Counter()
    by_row = Counter()
    for c in candidates:
        by_column[c["column"]] += 1
        by_row[c["row_id"]] += 1
    return {
        "by_column": by_column,
        "by_row": by_row,
        "total_candidates": len(candidates)
    }


def summarize_bayesian_info(
    candidate: Dict[str, Any],
    df: pd.DataFrame,
    col_profiles: Dict[str, Dict[str, Any]],
    candidate_stats: Dict[str, Any],
    similarity_info: Dict[str, Any],
    conflict_ctx: Dict[str, Any],
) -> Dict[str, Any]:
    col = candidate["column"]
    value = candidate["value"]
    semantic_type = col_profiles[col]["semantic_type"]

    col_candidate_count = candidate_stats["by_column"].get(col, 0)
    prior = col_candidate_count / len(df) if len(df) > 0 else 0.01
    prior = min(max(prior, 1e-4), 0.95)

    violated_rules = candidate.get("violated_rules", [])
    rule_signal = min(5.0, len(violated_rules) + candidate.get("conflict_score", 0.0) / 100.0)

    col_counter = col_profiles[col]["value_counter"]
    freq = col_counter.get(value, 0)
    non_null_count = col_profiles[col]["non_null_count"]
    rarity = 1.0 - (freq / non_null_count) if non_null_count > 0 else 1.0
    rarity = max(0.0, min(1.0, rarity))

    current_pattern = detect_basic_pattern(value) if value is not None else "NULL"
    dominant_pattern = col_profiles[col]["dominant_pattern"]
    pattern_signal = 1.0 if dominant_pattern is not None and current_pattern != dominant_pattern else 0.0

    if semantic_type == "id_like":
        rarity *= 0.2
    elif semantic_type == "code_like":
        rarity *= 0.6
    elif semantic_type == "categorical":
        rarity *= 1.0
    elif semantic_type == "categorical_list":
        rarity *= 0.8
    elif semantic_type == "text":
        rarity *= 1.2
    elif semantic_type in {"date_like", "duration_like"}:
        rarity *= 0.9

    neighbor_majority_value = similarity_info.get("neighbor_majority_value")
    neighbor_majority_ratio = similarity_info.get("neighbor_majority_ratio")
    if neighbor_majority_value is None or neighbor_majority_ratio is None:
        neighbor_signal = 0.0
    else:
        neighbor_signal = neighbor_majority_ratio if neighbor_majority_value != value else 0.0

    movie_typed_signal = min(3.0, conflict_ctx.get("movie_typed_rule_count", 0) + conflict_ctx.get("movie_consistency_rule_count", 0))

    logit_post = (
        safe_logit(prior)
        + W_RULE * rule_signal
        + W_RARITY * rarity
        + W_PATTERN * pattern_signal
        + W_NEIGHBOR * neighbor_signal
        + W_MOVIE_TYPED * movie_typed_signal
    )
    posterior = sigmoid(logit_post)

    return {
        "prior_error_probability": round(prior, 6),
        "posterior_error_probability": round(posterior, 6),
        "bayes_evidence": {
            "rule_signal": round(float(rule_signal), 6),
            "rarity_signal": round(float(rarity), 6),
            "pattern_signal": round(float(pattern_signal), 6),
            "neighbor_signal": round(float(neighbor_signal), 6),
            "movie_typed_signal": round(float(movie_typed_signal), 6),
        },
        "note": "posterior is a heuristic Bayesian-style approximation"
    }


# ============================================================
# 11. 构造给 LLM 的上下文文本
# ============================================================

def build_llm_context_text(
    row_id: int,
    col: str,
    value: Any,
    row_ctx: Dict[str, Any],
    col_ctx: Dict[str, Any],
    conflict_ctx: Dict[str, Any],
    sim_ctx: Dict[str, Any],
    bayes_ctx: Dict[str, Any]
) -> str:
    lines = []

    lines.append("Candidate cell:")
    lines.append(f"- row_id: {row_id}")
    lines.append(f"- column: {col}")
    lines.append(f"- current value: {value}")

    lines.append("")
    lines.append("Movie metadata row context:")
    lines.append(f"- missing_count: {row_ctx['missing_count']}")
    lines.append(f"- non_missing_count: {row_ctx['non_missing_count']}")
    lines.append(f"- avg_value_length: {row_ctx['avg_value_length']}")
    lines.append(f"- movie_metadata_bundle: {row_ctx['movie_metadata_bundle']}")
    lines.append("- row field values:")
    for k, v in row_ctx["row_values_for_llm"]:
        lines.append(f"  - {k}: {v}")

    lines.append("")
    lines.append("Column context:")
    lines.append(f"- detected_type: {col_ctx['detected_type']}")
    lines.append(f"- semantic_type: {col_ctx['semantic_type']}")
    lines.append(f"- missing_rate: {col_ctx['missing_rate']}")
    lines.append(f"- unique_rate: {col_ctx['unique_rate']}")
    lines.append(f"- value_frequency: {col_ctx['value_frequency']}")
    lines.append(f"- value_frequency_rank: {col_ctx['value_frequency_rank']}")
    lines.append(f"- parsed_year: {col_ctx['parsed_year']}")
    lines.append(f"- parsed_release_date: {col_ctx['parsed_release_date']}")
    lines.append(f"- duration_minutes: {col_ctx['duration_minutes']}")
    lines.append(f"- rating_value_float: {col_ctx['rating_value_float']}")
    lines.append(f"- count_value_int: {col_ctx['count_value_int']}")
    lines.append(f"- review_count_parsed: {col_ctx['review_count_parsed']}")
    lines.append(f"- list_tokens: {col_ctx['list_tokens']}")
    lines.append(f"- list_token_count: {col_ctx['list_token_count']}")
    lines.append(f"- current_pattern: {col_ctx['current_value_pattern']}")
    lines.append(f"- dominant_pattern: {col_ctx['dominant_pattern']}")
    lines.append(f"- charset_features: {col_ctx['charset_features']}")
    lines.append(f"- movie_extra_profile: {col_ctx['movie_extra_profile']}")

    lines.append("- top values with counts:")
    for item in col_ctx["top_values_with_counts"][:TOP_K_COLUMN_VALUES]:
        lines.append(f"  - {item['value']}: {item['count']}")

    lines.append("- top patterns with counts:")
    for item in col_ctx["top_patterns_with_counts"][:5]:
        lines.append(f"  - {item['pattern']}: count={item['count']}, ratio={item['ratio']}")

    lines.append("")
    lines.append("Conflict information:")
    lines.append(f"- has_rule_violation: {conflict_ctx['has_rule_violation']}")
    lines.append(f"- violation_count: {conflict_ctx['violation_count']}")
    lines.append(f"- main_rule_type: {conflict_ctx['main_rule_type']}")
    lines.append(f"- main_usage_role: {conflict_ctx['main_usage_role']}")
    lines.append(f"- strong_rule_count: {conflict_ctx['strong_rule_count']}")
    lines.append(f"- candidate_generation_rule_count: {conflict_ctx['candidate_generation_rule_count']}")
    lines.append(f"- fd_like_count: {conflict_ctx['fd_like_count']}")
    lines.append(f"- context_rule_count: {conflict_ctx['context_rule_count']}")
    lines.append(f"- global_rule_count: {conflict_ctx['global_rule_count']}")
    lines.append(f"- rare_value_count: {conflict_ctx['rare_value_count']}")
    lines.append(f"- typo_rule_count: {conflict_ctx['typo_rule_count']}")
    lines.append(f"- pattern_rule_count: {conflict_ctx['pattern_rule_count']}")
    lines.append(f"- schema_rule_count: {conflict_ctx['schema_rule_count']}")
    lines.append(f"- movie_typed_rule_count: {conflict_ctx['movie_typed_rule_count']}")
    lines.append(f"- movie_consistency_rule_count: {conflict_ctx['movie_consistency_rule_count']}")

    lines.append("- violated rules:")
    for r in conflict_ctx["violated_rules_detailed"][:20]:
        lines.append(
            f"  - rule_id={r.get('rule_id')}, rule_type={r.get('rule_type')}, "
            f"priority={r.get('priority')}, usage_role={r.get('usage_role')}, "
            f"expected_value={r.get('expected_value')}, reason={r.get('reason')}"
        )

    lines.append("- alternative values from rules/column:")
    for alt in conflict_ctx["all_alternative_values"][:TOP_K_ALT_VALUES]:
        lines.append(f"  - {alt}")

    lines.append("")
    lines.append("Similar rows:")
    lines.append(f"- neighbor_majority_value: {sim_ctx['neighbor_majority_value']}")
    lines.append(f"- neighbor_majority_ratio: {sim_ctx['neighbor_majority_ratio']}")
    lines.append("- top similar rows:")
    for sr in sim_ctx["topk_similar_rows"]:
        lines.append(
            f"  - row_id={sr['row_id']}, similarity={sr['similarity']}, "
            f"target_column_value={sr['target_column_value']}, row_values={sr['row_values']}"
        )

    lines.append("")
    lines.append("Bayesian-style probabilities:")
    lines.append(f"- prior_error_probability: {bayes_ctx['prior_error_probability']}")
    lines.append(f"- posterior_error_probability: {bayes_ctx['posterior_error_probability']}")
    lines.append(f"- evidence: {bayes_ctx['bayes_evidence']}")

    return "\n".join(lines)


# ============================================================
# 12. 主流程
# ============================================================

def main():
    df = load_table(INPUT_TABLE_CSV)
    candidates = load_shrunk_candidates(INPUT_SHRUNK_CSV)

    col_profiles = analyze_columns(df)
    candidate_stats = build_candidate_stats(candidates)

    flat_rows = []

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for cand in candidates:
            row_id = cand["row_id"]
            col = cand["column"]
            value = cand["value"]

            row_ctx = summarize_row_context(df, row_id, col_profiles)
            col_ctx = summarize_column_context(col, value, col_profiles)
            conflict_ctx = summarize_conflict_info(cand, col_profiles)
            sim_ctx = summarize_similarity_info(df, row_id, col, col_profiles)
            bayes_ctx = summarize_bayesian_info(cand, df, col_profiles, candidate_stats, sim_ctx, conflict_ctx)

            llm_context_text = build_llm_context_text(
                row_id, col, value, row_ctx, col_ctx, conflict_ctx, sim_ctx, bayes_ctx
            )

            item = {
                "row_id": row_id,
                "column": col,
                "value": value,
                "violation_count": cand.get("violation_count"),
                "conflict_score": cand.get("conflict_score"),
                "row_context": row_ctx,
                "column_context": col_ctx,
                "conflict_context": conflict_ctx,
                "similarity_context": sim_ctx,
                "bayesian_context": bayes_ctx,
                "llm_context_text": llm_context_text
            }

            f.write(json.dumps(item, ensure_ascii=False) + "\n")

            movie_bundle = row_ctx["movie_metadata_bundle"]
            flat_rows.append({
                "row_id": row_id,
                "column": col,
                "value": value,
                "semantic_type": col_ctx["semantic_type"],
                "detected_type": col_ctx["detected_type"],
                "violation_count": cand.get("violation_count"),
                "conflict_score": cand.get("conflict_score"),
                "value_frequency": col_ctx["value_frequency"],
                "value_frequency_rank": col_ctx["value_frequency_rank"],

                "parsed_year": col_ctx["parsed_year"],
                "parsed_release_year": col_ctx["parsed_release_date"]["year"] if col_ctx["parsed_release_date"] else None,
                "parsed_release_country": col_ctx["parsed_release_date"]["country"] if col_ctx["parsed_release_date"] else None,
                "duration_minutes": col_ctx["duration_minutes"],
                "rating_value_float": col_ctx["rating_value_float"],
                "count_value_int": col_ctx["count_value_int"],
                "review_total": total_review_count(value) if col == "ReviewCount" else None,
                "list_token_count": col_ctx["list_token_count"],
                "list_token_count_bucket": col_ctx["list_token_count_bucket"],
                "text_length_bucket": col_ctx["text_length_bucket"],
                "has_mojibake_or_control_chars": int(col_ctx["has_mojibake_or_control_chars"]),

                "movie_row_year": movie_bundle.get("parsed_year"),
                "movie_row_release_year": movie_bundle.get("parsed_release_date", {}).get("year") if movie_bundle.get("parsed_release_date") else None,
                "movie_row_release_country": movie_bundle.get("parsed_release_date", {}).get("country") if movie_bundle.get("parsed_release_date") else None,
                "movie_row_release_year_gap": movie_bundle.get("release_year_gap"),
                "movie_row_duration_minutes": movie_bundle.get("duration_minutes"),
                "movie_row_rating_value_float": movie_bundle.get("rating_value_float"),
                "movie_row_rating_count_int": movie_bundle.get("rating_count_int"),
                "movie_row_review_total": movie_bundle.get("review_total"),
                "movie_row_review_to_rating_ratio": movie_bundle.get("review_to_rating_ratio"),
                "movie_row_actors_cast_overlap": movie_bundle.get("actors_cast_overlap"),

                "neighbor_majority_value": sim_ctx["neighbor_majority_value"],
                "neighbor_majority_ratio": sim_ctx["neighbor_majority_ratio"],
                "prior_error_probability": bayes_ctx["prior_error_probability"],
                "posterior_error_probability": bayes_ctx["posterior_error_probability"],
                "main_rule_type": conflict_ctx["main_rule_type"],
                "main_usage_role": conflict_ctx["main_usage_role"],
                "strong_rule_count": conflict_ctx["strong_rule_count"],
                "candidate_generation_rule_count": conflict_ctx["candidate_generation_rule_count"],
                "fd_like_count": conflict_ctx["fd_like_count"],
                "context_rule_count": conflict_ctx["context_rule_count"],
                "global_rule_count": conflict_ctx["global_rule_count"],
                "rare_value_count": conflict_ctx["rare_value_count"],
                "typo_rule_count": conflict_ctx["typo_rule_count"],
                "pattern_rule_count": conflict_ctx["pattern_rule_count"],
                "schema_rule_count": conflict_ctx["schema_rule_count"],
                "movie_typed_rule_count": conflict_ctx["movie_typed_rule_count"],
                "movie_consistency_rule_count": conflict_ctx["movie_consistency_rule_count"],
            })

    pd.DataFrame(flat_rows).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"[OK] Full LLM contexts saved to: {OUTPUT_JSONL}")
    print(f"[OK] Flat summary saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

import json
import math
import re
from collections import Counter
from itertools import combinations
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd

# ============================================================
# 1. 用户配置区
# ============================================================

INPUT_CSV = "/mnt/mydata/dq/projects/Splittree/movies/movies_dirty.csv"
OUTPUT_JSON = "rule_pool_movies.json"

MISSING_TOKENS = {
    "", "empty", "null", "none", "nan", "na", "n/a", "missing", "unk",
    "unknown", "not available", "not applicable", "tbd", "na."
}

NUMERIC_RATIO_THRESHOLD = 0.90
ENUM_UNIQUE_THRESHOLD = 50
ENUM_RATIO_THRESHOLD = 0.08

ID_LIKE_UNIQUE_RATIO_THRESHOLD = 0.98
CODE_LIKE_AVG_LEN_THRESHOLD = 24
CODE_LIKE_DOMINANT_PATTERN_THRESHOLD = 0.70
DOMINANT_PATTERN_THRESHOLD = 0.70
DOMINANT_TYPO_MIN_RATIO = 0.50

MAX_LHS_SIZE = 2
FD_CONFIDENCE_THRESHOLD = 0.98
MIN_SUPPORT_ROWS = 5

SOFT_FD_CONFIDENCE_THRESHOLD = 0.80
SOFT_FD_MIN_GROUP_SIZE = 3
PAIR_CONTEXT_MIN_RATIO = 0.70
PAIR_CONTEXT_MIN_GROUP_SIZE = 2

HIGH_CARDINALITY_RATIO_FOR_LHS = 0.98
ALLOW_INDEX_COLUMN = False
SKIP_SCHEMA_FOR_ID_LIKE = True

# movies 数据集字段：样例列来自用户提供的 movies 数据格式
ID_COLUMNS = {"Id"}
TEXT_LONG_COLUMNS = {"Description", "Filming Locations"}
PEOPLE_COLUMNS = {"Director", "Creator", "Actors", "Cast"}
LIST_COLUMNS = {"Creator", "Actors", "Cast", "Language", "Country", "Genre"}
NUMERIC_LIKE_COLUMNS = {"Year", "Duration", "RatingValue", "RatingCount", "ReviewCount"}
DATE_COLUMNS = {"Release Date"}

# 原则：尽可能高召回。除 Id 外，其他字段都可以作为候选目标列。
TARGET_CANDIDATE_COLUMNS = {
    "Name", "Year", "Release Date", "Director", "Creator", "Actors", "Cast",
    "Language", "Country", "Duration", "RatingValue", "RatingCount", "ReviewCount",
    "Genre", "Filming Locations", "Description",
}

HIGH_RISK_MISSING_COLUMNS = {
    # Language/Country 在 movies 中常为多值字段且当前数据误报很高，不再作为高风险缺失列。
    "Name", "Year", "Release Date", "Director",
    "Duration", "RatingValue", "RatingCount", "Genre",
}

ENABLE_RARE_VALUE_RULE = True
RARE_VALUE_MAX_FREQ = 2
ENABLE_RARE_PATTERN_RULE = True
RARE_PATTERN_MAX_RATIO = 0.02

# 长文本/长尾列不适合靠 rare_value / rare_pattern 直接进候选，容易制造大量误报。
# Movies 中 Release Date / ReviewCount / Language / Country 也高度长尾或多值枚举，
# 只靠 rare_value/context dominant 很容易把大量正确值打成候选错误。
SKIP_RARE_VALUE_COLUMNS = {
    "Name", "Director", "Creator", "Actors", "Cast", "Filming Locations", "Description",
    "Release Date", "ReviewCount", "Language", "Country", "Genre",
}
SKIP_RARE_PATTERN_COLUMNS = {
    "Name", "Director", "Creator", "Actors", "Cast", "Filming Locations", "Description",
    "Release Date", "ReviewCount", "Language", "Country", "Genre",
}
SKIP_GLOBAL_DOMINANT_COLUMNS = {
    "Name", "Director", "Creator", "Actors", "Cast", "Filming Locations", "Description",
    "Release Date", "ReviewCount", "Language", "Country", "Genre",
}
SKIP_GENERIC_PATTERN_COLUMNS = {
    "Name", "Director", "Creator", "Actors", "Cast", "Filming Locations", "Description",
    "Release Date", "ReviewCount", "Language", "Country", "Genre",
}

# FD/context dominant 对长尾字段非常危险：
# 这些字段保留格式、一致性、缺失等显式规则，但不作为 FD/context dominant 的 RHS。
UNSAFE_FD_RHS_COLUMNS = {
    "Name", "Director", "Creator", "Actors", "Cast", "Filming Locations", "Description",
    "Release Date", "ReviewCount", "Language", "Country", "Genre", "Year",
}
UNSAFE_CONTEXT_RHS_COLUMNS = {
    "Name", "Director", "Creator", "Actors", "Cast", "Filming Locations", "Description",
    "Release Date", "ReviewCount", "Language", "Country", "Genre", "Year", "RatingValue",
}

MOVIE_YEAR_MIN = 1880
MOVIE_YEAR_MAX = 2035
RATING_MIN = 0.0
RATING_MAX = 10.0
DURATION_MIN_MINUTES = 1
DURATION_MAX_MINUTES = 600
MIN_LIST_TOKEN_COUNT = 1


# ============================================================
# 2. 通用预处理函数
# ============================================================

def normalize_value(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.lower() in MISSING_TOKENS:
        return None
    return s.lower()


def normalize_series(s):
    return s.map(normalize_value)


def maybe_drop_unnamed_columns(df):
    keep_cols = [c for c in df.columns if not str(c).lower().startswith("unnamed:")]
    return df[keep_cols].copy()


def detect_basic_pattern(value):
    if value is None:
        return "NULL"
    parts, prev_type, cnt = [], None, 0
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


def try_parse_numeric(series):
    cleaned = series.dropna().astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False)
    numeric = pd.to_numeric(cleaned, errors="coerce")
    success_ratio = numeric.notna().mean() if len(cleaned) > 0 else 0.0
    return numeric, success_ratio


def shannon_entropy(series):
    vals = series.dropna().tolist()
    if not vals:
        return 0.0
    cnt = Counter(vals)
    total = len(vals)
    ent = 0.0
    for c in cnt.values():
        p = c / total
        ent -= p * math.log2(p)
    return ent


def value_frequency_dict(series, topk=20):
    cnt = Counter(series.dropna().tolist())
    total = sum(cnt.values()) if cnt else 0
    out = []
    for k, v in cnt.most_common(topk):
        out.append({"value": k, "count": v, "ratio": round(v / total, 6) if total > 0 else 0.0})
    return out


def split_comma_list(s: Optional[str]) -> List[str]:
    if s is None:
        return []
    txt = str(s).strip()
    if txt == "":
        return []
    parts = [p.strip().lower() for p in txt.split(",")]
    return [p for p in parts if p and p not in MISSING_TOKENS]


def canonicalize_list_value(s: Optional[str]) -> Optional[str]:
    parts = split_comma_list(s)
    if not parts:
        return None
    return ",".join(parts)


# ============================================================
# 3. movies 专用解析函数
# ============================================================

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
        "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
        "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
        "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
        "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
    }

    day = None
    month = None
    m_full = re.search(r"\b(\d{1,2})\s+([a-z]+)\s+(18\d{2}|19\d{2}|20\d{2})\b", txt)
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

    return {"year": year, "month": month, "day": day, "country": country}


def parse_duration_minutes(s: Optional[str]) -> Optional[int]:
    if s is None:
        return None
    txt = str(s).strip().lower()
    m = re.search(r"\b(\d{1,4})\s*(min|mins|minute|minutes)\b", txt)
    if not m:
        m2 = re.fullmatch(r"\d{1,4}", txt)
        if not m2:
            return None
        mins = int(txt)
    else:
        mins = int(m.group(1))
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
    return {"user": user_count if user_count is not None else 0, "critic": critic_count if critic_count is not None else 0}


def looks_like_imdb_id(s: Optional[str]) -> bool:
    if s is None:
        return False
    return bool(re.fullmatch(r"tt\d{7,9}", str(s).strip().lower()))


def looks_like_person_or_people_list(s: Optional[str]) -> bool:
    if s is None:
        return False
    parts = split_comma_list(s)
    if len(parts) < MIN_LIST_TOKEN_COUNT:
        return False
    return any(re.search(r"[a-z]", p) is not None for p in parts)


def looks_like_category_list(s: Optional[str]) -> bool:
    if s is None:
        return False
    parts = split_comma_list(s)
    if len(parts) < MIN_LIST_TOKEN_COUNT:
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


# ============================================================
# 4. 列分析
# ============================================================

def is_index_like_column(series):
    s = series.dropna()
    if len(s) == 0:
        return False
    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.notna().mean() < 0.95:
        return False
    unique_ratio = s.nunique() / len(s)
    return unique_ratio > 0.98


def is_id_like_column(col_name, non_null, unique_ratio, numeric_ratio, avg_len):
    if col_name == "Id":
        return True
    if len(non_null) == 0:
        return False
    if unique_ratio >= ID_LIKE_UNIQUE_RATIO_THRESHOLD and avg_len <= CODE_LIKE_AVG_LEN_THRESHOLD:
        return True
    if unique_ratio >= ID_LIKE_UNIQUE_RATIO_THRESHOLD and numeric_ratio >= NUMERIC_RATIO_THRESHOLD:
        return True
    return False


def is_code_like_column(non_null, unique_ratio, numeric_ratio, avg_len, top_patterns):
    if len(non_null) == 0:
        return False
    dominant_ratio = top_patterns[0]["ratio"] if top_patterns else 0.0
    if avg_len <= CODE_LIKE_AVG_LEN_THRESHOLD and dominant_ratio >= CODE_LIKE_DOMINANT_PATTERN_THRESHOLD:
        if unique_ratio > ENUM_RATIO_THRESHOLD:
            return True
    if numeric_ratio >= 0.95 and avg_len <= 15 and unique_ratio < ID_LIKE_UNIQUE_RATIO_THRESHOLD:
        return True
    return False


def build_movie_column_extra_profile(col: str, non_null: pd.Series) -> Dict[str, Any]:
    extra = {}
    if col == "Id":
        valid = non_null.map(looks_like_imdb_id)
        extra["imdb_id_valid_ratio"] = round(float(valid.mean()), 6) if len(valid) > 0 else 0.0
    if col == "Year":
        parsed = non_null.map(parse_year_value)
        extra["year_parse_ratio"] = round(float(parsed.notna().mean()), 6) if len(parsed) > 0 else 0.0
        extra["year_min"] = int(parsed.dropna().min()) if parsed.notna().any() else None
        extra["year_max"] = int(parsed.dropna().max()) if parsed.notna().any() else None
    if col == "Release Date":
        parsed = non_null.map(parse_release_date)
        extra["release_date_parse_ratio"] = round(float(parsed.notna().mean()), 6) if len(parsed) > 0 else 0.0
        years = [x["year"] for x in parsed.dropna().tolist()]
        extra["release_year_counter"] = dict(Counter(years).most_common(30))
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
        extra["length_bucket_counter"] = dict(Counter(non_null.map(text_length_bucket).tolist()).most_common())
        extra["mojibake_count"] = int(non_null.map(has_mojibake_or_control_chars).sum())
    return extra


def analyze_columns(df):
    profiles = {}
    n = len(df)
    for col in df.columns:
        s = normalize_series(df[col])
        non_null = s.dropna()
        missing_rate = 1.0 - (len(non_null) / n if n > 0 else 0.0)
        unique_count = non_null.nunique()
        unique_ratio = unique_count / len(non_null) if len(non_null) > 0 else 0.0
        numeric_vals, numeric_ratio = try_parse_numeric(non_null)
        patterns = non_null.map(detect_basic_pattern)
        pattern_counter = Counter(patterns.tolist())
        total_patterns = sum(pattern_counter.values())
        top_patterns = [{"pattern": p, "count": c, "ratio": round(c / total_patterns, 6)} for p, c in pattern_counter.most_common(10)]
        lengths = non_null.map(lambda x: len(str(x)))
        avg_len = float(lengths.mean()) if len(lengths) > 0 else 0.0
        profile = {
            "column": col,
            "missing_rate": round(missing_rate, 6),
            "non_null_count": int(len(non_null)),
            "unique_count": int(unique_count),
            "unique_ratio": round(unique_ratio, 6),
            "numeric_parse_ratio": round(float(numeric_ratio), 6),
            "entropy": round(shannon_entropy(non_null), 6),
            "detected_type": None,
            "semantic_type": None,
            "index_like": is_index_like_column(non_null),
            "avg_length": round(avg_len, 6),
            "top_values": value_frequency_dict(non_null, topk=20),
            "top_patterns": top_patterns,
            "value_counter": Counter(non_null.tolist()),
        }
        profile.update(build_movie_column_extra_profile(col, non_null))
        if col in ID_COLUMNS:
            profile["detected_type"] = "id"
            profile["semantic_type"] = "id_like"
        elif col in {"Year", "Duration", "RatingValue", "RatingCount", "ReviewCount"}:
            profile["detected_type"] = "numeric_like"
            profile["semantic_type"] = "numeric"
        elif col in LIST_COLUMNS:
            profile["detected_type"] = "list_like"
            profile["semantic_type"] = "categorical_list"
        elif col in TEXT_LONG_COLUMNS:
            profile["detected_type"] = "text"
            profile["semantic_type"] = "text"
        elif numeric_ratio >= NUMERIC_RATIO_THRESHOLD:
            profile["detected_type"] = "numeric"
            profile["semantic_type"] = "numeric"
        else:
            if unique_count <= ENUM_UNIQUE_THRESHOLD or unique_ratio <= ENUM_RATIO_THRESHOLD:
                profile["detected_type"] = "categorical"
                profile["semantic_type"] = "categorical"
            else:
                profile["detected_type"] = "text"
                profile["semantic_type"] = "text"
        if profile["semantic_type"] not in {"id_like", "numeric", "categorical_list", "text", "categorical"}:
            if profile["index_like"]:
                profile["semantic_type"] = "id_like"
            elif is_id_like_column(col, non_null, unique_ratio, numeric_ratio, avg_len):
                profile["semantic_type"] = "id_like"
            elif is_code_like_column(non_null, unique_ratio, numeric_ratio, avg_len, top_patterns):
                profile["semantic_type"] = "code_like"
        profiles[col] = profile
    return profiles


# ============================================================
# 5. FD / soft FD 工具
# ============================================================

def calc_fd_confidence_and_support(df, lhs_cols, rhs_col):
    sub = df[lhs_cols + [rhs_col]].dropna()
    support = len(sub)
    if support < MIN_SUPPORT_ROWS:
        return 0.0, support
    grouped = sub.groupby(lhs_cols, dropna=True)[rhs_col]
    majority_sum = 0
    for _, grp in grouped:
        cnt = Counter(grp.tolist())
        majority_sum += cnt.most_common(1)[0][1]
    confidence = majority_sum / support if support > 0 else 0.0
    return round(confidence, 6), int(support)


def calc_soft_fd_confidence_and_support(df, lhs_cols, rhs_col):
    sub = df[lhs_cols + [rhs_col]].dropna()
    support = len(sub)
    if support < SOFT_FD_MIN_GROUP_SIZE:
        return 0.0, support
    grouped = sub.groupby(lhs_cols, dropna=True)[rhs_col]
    majority_sum = 0
    for _, grp in grouped:
        if len(grp) < SOFT_FD_MIN_GROUP_SIZE:
            continue
        cnt = Counter(grp.tolist())
        majority_sum += cnt.most_common(1)[0][1]
    confidence = majority_sum / support if support > 0 else 0.0
    return round(confidence, 6), int(support)


def calc_not_null_confidence_and_support(series):
    total = len(series)
    if total == 0:
        return 0.0, 0
    non_null = series.notna().sum()
    return round(non_null / total, 6), int(total)


def calc_pattern_confidence_and_support(series, dominant_pattern):
    non_null = series.dropna()
    support = len(non_null)
    if support == 0:
        return 0.0, 0
    pats = non_null.map(detect_basic_pattern)
    ok = (pats == dominant_pattern).sum()
    return round(ok / support, 6), int(support)


def select_fd_lhs_candidates(profiles):
    candidates = []
    for col, info in profiles.items():
        if (not ALLOW_INDEX_COLUMN) and info.get("index_like", False):
            continue
        if col in ID_COLUMNS and not ALLOW_INDEX_COLUMN:
            continue
        if info["unique_ratio"] >= HIGH_CARDINALITY_RATIO_FOR_LHS:
            continue
        candidates.append(col)
    return candidates


# ============================================================
# 6. schema / typed rules
# ============================================================

def build_schema_rules(df, profiles):
    rules = []
    rule_id = 1
    for col, info in profiles.items():
        if col not in TARGET_CANDIDATE_COLUMNS:
            continue
        series = normalize_series(df[col])
        not_null_threshold = 0.60
        if col in {"Name", "Year", "Release Date", "Director", "Language", "Country", "Duration", "RatingValue", "Genre"}:
            not_null_threshold = 0.80
        if col in {"Filming Locations", "Description", "ReviewCount"}:
            not_null_threshold = 0.40
        if info["missing_rate"] <= not_null_threshold:
            confidence, support = calc_not_null_confidence_and_support(series)
            rules.append({"rule_id": f"SCHEMA_{rule_id}", "rule_type": "not_null", "column": col, "description": f"{col} should not be null in movies metadata", "confidence": confidence, "support": support, "usage_role": "candidate_generation_rule"})
            rule_id += 1
        if len(info["top_patterns"]) > 0 and col not in SKIP_GENERIC_PATTERN_COLUMNS:
            main_pattern = info["top_patterns"][0]
            if main_pattern["ratio"] >= DOMINANT_PATTERN_THRESHOLD:
                confidence, support = calc_pattern_confidence_and_support(series, main_pattern["pattern"])
                rules.append({"rule_id": f"SCHEMA_{rule_id}", "rule_type": "pattern", "column": col, "description": f"{col} should mostly follow the dominant pattern", "pattern": main_pattern["pattern"], "confidence": confidence, "support": support, "usage_role": "strong_rule"})
                rule_id += 1
        if col == "Year":
            rules.append({"rule_id": f"SCHEMA_{rule_id}", "rule_type": "movie_year_range", "column": col, "description": f"Year should be a movie year in [{MOVIE_YEAR_MIN}, {MOVIE_YEAR_MAX}]", "min_year": MOVIE_YEAR_MIN, "max_year": MOVIE_YEAR_MAX, "confidence": 0.95, "support": info["non_null_count"], "usage_role": "candidate_generation_rule"})
            rule_id += 1
        if col == "Release Date":
            rules.append({"rule_id": f"SCHEMA_{rule_id}", "rule_type": "release_date_format", "column": col, "description": "Release Date should be parseable as a release date, usually containing a year and optionally country", "confidence": 0.90, "support": info["non_null_count"], "usage_role": "candidate_generation_rule"})
            rule_id += 1
            rules.append({"rule_id": f"SCHEMA_{rule_id}", "rule_type": "release_year_consistency", "date_column": "Release Date", "year_column": "Year", "description": "Release Date year should usually agree with Year, allowing small differences for delayed country releases", "allowed_year_gap": 2, "confidence": 0.86, "support": info["non_null_count"], "usage_role": "candidate_generation_rule"})
            rule_id += 1
        if col == "Duration":
            rules.append({"rule_id": f"SCHEMA_{rule_id}", "rule_type": "duration_format", "column": col, "description": "Duration should look like '<minutes> min' and fall in a broad movie duration range", "min_minutes": DURATION_MIN_MINUTES, "max_minutes": DURATION_MAX_MINUTES, "confidence": 0.92, "support": info["non_null_count"], "usage_role": "candidate_generation_rule"})
            rule_id += 1
        if col == "RatingValue":
            rules.append({"rule_id": f"SCHEMA_{rule_id}", "rule_type": "rating_value_range", "column": col, "description": "RatingValue should be numeric in [0, 10]", "min_rating": RATING_MIN, "max_rating": RATING_MAX, "confidence": 0.95, "support": info["non_null_count"], "usage_role": "candidate_generation_rule"})
            rule_id += 1
        if col == "RatingCount":
            rules.append({"rule_id": f"SCHEMA_{rule_id}", "rule_type": "rating_count_format", "column": col, "description": "RatingCount should be a non-negative integer, often formatted with commas", "confidence": 0.92, "support": info["non_null_count"], "usage_role": "candidate_generation_rule"})
            rule_id += 1
        if col == "ReviewCount":
            rules.append({"rule_id": f"SCHEMA_{rule_id}", "rule_type": "review_count_format", "column": col, "description": "ReviewCount should usually contain user and/or critic review counts", "confidence": 0.88, "support": info["non_null_count"], "usage_role": "candidate_generation_rule"})
            rule_id += 1
        if col in LIST_COLUMNS:
            rules.append({"rule_id": f"SCHEMA_{rule_id}", "rule_type": "comma_list_format", "column": col, "description": f"{col} should be a comma-separated list of tokens", "confidence": 0.84, "support": info["non_null_count"], "usage_role": "candidate_generation_rule"})
            rule_id += 1
        if col in PEOPLE_COLUMNS:
            rules.append({"rule_id": f"SCHEMA_{rule_id}", "rule_type": "people_list_format", "column": col, "description": f"{col} should contain person names or comma-separated person names", "confidence": 0.82, "support": info["non_null_count"], "usage_role": "candidate_generation_rule"})
            rule_id += 1
        rules.append({"rule_id": f"SCHEMA_{rule_id}", "rule_type": "mojibake_or_control_char", "column": col, "description": f"{col} should not contain mojibake or control characters", "confidence": 0.92, "support": info["non_null_count"], "usage_role": "candidate_generation_rule"})
        rule_id += 1
    return rules


# ============================================================
# 7. FD / context rules
# ============================================================

def build_fd_rules(df, profiles):
    rules = []
    rule_id = 1
    cols = list(df.columns)
    norm_df = pd.DataFrame({c: normalize_series(df[c]) for c in cols})
    lhs_candidates = select_fd_lhs_candidates(profiles)
    rhs_candidates = [c for c in TARGET_CANDIDATE_COLUMNS if c in cols and c not in UNSAFE_FD_RHS_COLUMNS]
    lhs_sets = []
    for k in range(1, MAX_LHS_SIZE + 1):
        lhs_sets.extend(list(combinations(lhs_candidates, k)))
    for lhs in lhs_sets:
        for rhs in rhs_candidates:
            if rhs in lhs:
                continue
            confidence, support = calc_fd_confidence_and_support(norm_df, list(lhs), rhs)
            if support < MIN_SUPPORT_ROWS:
                continue
            if confidence >= FD_CONFIDENCE_THRESHOLD:
                rules.append({"rule_id": f"FD_{rule_id}", "rule_type": "functional_dependency", "lhs": list(lhs), "rhs": rhs, "description": f"{', '.join(lhs)} -> {rhs}", "confidence": confidence, "support": support, "usage_role": "strong_rule"})
                rule_id += 1
    dedup = {(tuple(sorted(r["lhs"])), r["rhs"]): r for r in rules}
    return list(dedup.values())


def build_soft_fd_rules(df, profiles):
    rules = []
    rule_id = 1
    cols = list(df.columns)
    norm_df = pd.DataFrame({c: normalize_series(df[c]) for c in cols})
    lhs_candidates = select_fd_lhs_candidates(profiles)
    rhs_candidates = [c for c in TARGET_CANDIDATE_COLUMNS if c in cols and c not in UNSAFE_FD_RHS_COLUMNS]
    lhs_sets = []
    for k in range(1, MAX_LHS_SIZE + 1):
        lhs_sets.extend(list(combinations(lhs_candidates, k)))
    for lhs in lhs_sets:
        for rhs in rhs_candidates:
            if rhs in lhs:
                continue
            confidence, support = calc_soft_fd_confidence_and_support(norm_df, list(lhs), rhs)
            if support < SOFT_FD_MIN_GROUP_SIZE:
                continue
            if SOFT_FD_CONFIDENCE_THRESHOLD <= confidence < FD_CONFIDENCE_THRESHOLD:
                rules.append({"rule_id": f"CAND_SOFTFD_{rule_id}", "rule_type": "soft_functional_dependency", "lhs": list(lhs), "rhs": rhs, "description": f"soft FD: {', '.join(lhs)} -> {rhs}", "confidence": confidence, "support": support, "usage_role": "candidate_generation_rule"})
                rule_id += 1
    return rules


def build_majority_map(norm_df: pd.DataFrame, lhs: List[str], rhs: str) -> Tuple[Dict[str, Any], int]:
    sub = norm_df[lhs + [rhs]].dropna()
    if len(sub) < PAIR_CONTEXT_MIN_GROUP_SIZE:
        return {}, 0
    grouped = sub.groupby(lhs, dropna=True)[rhs]
    majority_map = {}
    total_support = 0
    for key, grp in grouped:
        if len(grp) < PAIR_CONTEXT_MIN_GROUP_SIZE:
            continue
        cnt = Counter(grp.tolist())
        maj_val, maj_cnt = cnt.most_common(1)[0]
        ratio = maj_cnt / len(grp)
        if ratio >= PAIR_CONTEXT_MIN_RATIO:
            if not isinstance(key, tuple):
                key = (key,)
            majority_map["||".join(map(str, key))] = {"dominant_value": maj_val, "group_size": len(grp), "ratio": round(ratio, 6)}
            total_support += len(grp)
    return majority_map, total_support


def build_pair_context_rules(df):
    rules = []
    rule_id = 1
    norm_df = pd.DataFrame({c: normalize_series(df[c]) for c in df.columns})
    # 只保留相对可靠的 context dominant。删除 Language/Country/Release Date/ReviewCount
    # 等高长尾字段作为 RHS 的规则，避免大规模误报。
    pair_specs = [
        (["Name", "Year"], "Duration"),
        (["RatingValue"], "RatingCount"),
    ]
    for lhs, rhs in pair_specs:
        if any(c not in norm_df.columns for c in lhs + [rhs]):
            continue
        majority_map, total_support = build_majority_map(norm_df, lhs, rhs)
        if not majority_map:
            continue
        rules.append({"rule_id": f"CAND_PAIRCTX_{rule_id}", "rule_type": "dominant_value_by_context_pair" if len(lhs) >= 2 else "dominant_value_by_context", "lhs": lhs, "rhs": rhs, "description": f"context dominant: {', '.join(lhs)} -> {rhs}", "mapping_size": len(majority_map), "majority_map": majority_map, "confidence": 0.84, "support": total_support, "usage_role": "candidate_generation_rule"})
        rule_id += 1
    return rules


def build_movie_consistency_rules(df, profiles):
    rules = []
    rule_id = 1
    norm_df = pd.DataFrame({c: normalize_series(df[c]) for c in df.columns})
    if {"Year", "Release Date"}.issubset(norm_df.columns):
        rules.append({"rule_id": f"CAND_MOVIECONS_{rule_id}", "rule_type": "release_year_consistency", "year_column": "Year", "date_column": "Release Date", "allowed_year_gap": 2, "description": "Release Date year should usually agree with Year within a small gap", "confidence": 0.86, "support": len(norm_df), "usage_role": "candidate_generation_rule"})
        rule_id += 1
    if {"Actors", "Cast"}.issubset(norm_df.columns):
        rules.append({"rule_id": f"CAND_MOVIECONS_{rule_id}", "rule_type": "actors_cast_overlap", "actors_column": "Actors", "cast_column": "Cast", "min_overlap_ratio": 0.50, "description": "Main Actors should usually overlap with Cast", "confidence": 0.85, "support": len(norm_df), "usage_role": "candidate_generation_rule"})
        rule_id += 1
    if {"RatingValue", "RatingCount"}.issubset(norm_df.columns):
        rules.append({"rule_id": f"CAND_MOVIECONS_{rule_id}", "rule_type": "rating_value_count_coexistence", "rating_value_column": "RatingValue", "rating_count_column": "RatingCount", "description": "RatingValue and RatingCount should usually co-exist", "confidence": 0.82, "support": len(norm_df), "usage_role": "candidate_generation_rule"})
        rule_id += 1
    if {"ReviewCount", "RatingCount"}.issubset(norm_df.columns):
        rules.append({"rule_id": f"CAND_MOVIECONS_{rule_id}", "rule_type": "review_rating_count_order", "review_count_column": "ReviewCount", "rating_count_column": "RatingCount", "description": "ReviewCount should usually be much smaller than RatingCount", "max_review_to_rating_ratio": 0.50, "confidence": 0.78, "support": len(norm_df), "usage_role": "candidate_generation_rule"})
        rule_id += 1
    return rules


# ============================================================
# 8. candidate generation rules
# ============================================================

def build_candidate_generation_rules(df, profiles):
    rules = []
    rule_id = 1
    norm_df = pd.DataFrame({c: normalize_series(df[c]) for c in df.columns})
    for col, prof in profiles.items():
        if col not in TARGET_CANDIDATE_COLUMNS or col in SKIP_GLOBAL_DOMINANT_COLUMNS:
            continue
        top_values = prof.get("top_values", [])
        if not top_values:
            continue
        top1 = top_values[0]
        if top1["ratio"] >= DOMINANT_TYPO_MIN_RATIO:
            rules.append({"rule_id": f"CAND_GDOM_{rule_id}", "rule_type": "global_dominant_value", "column": col, "description": f"global dominant value for {col}", "dominant_value": top1["value"], "dominant_count": top1["count"], "dominant_ratio": top1["ratio"], "confidence": round(top1["ratio"], 6), "support": prof["non_null_count"], "usage_role": "candidate_generation_rule"})
            rule_id += 1

    lhs_candidates = select_fd_lhs_candidates(profiles)
    context_rule_id = 1
    for rhs in TARGET_CANDIDATE_COLUMNS:
        if rhs not in df.columns:
            continue
        if rhs in UNSAFE_CONTEXT_RHS_COLUMNS:
            continue
        for lhs in lhs_candidates:
            if lhs == rhs:
                continue
            sub = norm_df[[lhs, rhs]].dropna()
            if len(sub) < SOFT_FD_MIN_GROUP_SIZE:
                continue
            grouped = sub.groupby(lhs)[rhs]
            majority_map = {}
            total_support = 0
            for lhs_val, grp in grouped:
                if len(grp) < SOFT_FD_MIN_GROUP_SIZE:
                    continue
                cnt = Counter(grp.tolist())
                maj_val, maj_cnt = cnt.most_common(1)[0]
                ratio = maj_cnt / len(grp)
                if ratio >= SOFT_FD_CONFIDENCE_THRESHOLD:
                    majority_map[lhs_val] = {"dominant_value": maj_val, "group_size": len(grp), "ratio": round(ratio, 6)}
                    total_support += len(grp)
            if majority_map:
                rules.append({"rule_id": f"CAND_CTXDOM_{context_rule_id}", "rule_type": "dominant_value_by_context", "lhs": [lhs], "rhs": rhs, "description": f"context dominant: {lhs} -> dominant {rhs}", "mapping_size": len(majority_map), "majority_map": majority_map, "confidence": 0.85, "support": total_support, "usage_role": "candidate_generation_rule"})
                context_rule_id += 1

    if ENABLE_RARE_VALUE_RULE:
        for col, prof in profiles.items():
            if col not in TARGET_CANDIDATE_COLUMNS or col in SKIP_RARE_VALUE_COLUMNS:
                continue
            counter = prof["value_counter"]
            rare_values = [v for v, c in counter.items() if c <= RARE_VALUE_MAX_FREQ]
            if rare_values:
                rules.append({"rule_id": f"CAND_RAREVAL_{rule_id}", "rule_type": "rare_value", "column": col, "description": f"rare values in {col} may be suspicious", "rare_values": rare_values[:5000], "confidence": 0.55, "support": sum(counter[v] for v in rare_values), "usage_role": "candidate_generation_rule"})
                rule_id += 1

    if ENABLE_RARE_PATTERN_RULE:
        for col, prof in profiles.items():
            if col not in TARGET_CANDIDATE_COLUMNS or col in SKIP_RARE_PATTERN_COLUMNS:
                continue
            top_patterns = prof.get("top_patterns", [])
            rare_patterns = [x["pattern"] for x in top_patterns if x["ratio"] <= RARE_PATTERN_MAX_RATIO]
            if rare_patterns:
                rules.append({"rule_id": f"CAND_RAREPAT_{rule_id}", "rule_type": "rare_pattern", "column": col, "description": f"rare patterns in {col} may be suspicious", "rare_patterns": rare_patterns, "confidence": 0.55, "support": prof["non_null_count"], "usage_role": "candidate_generation_rule"})
                rule_id += 1

    for col in HIGH_RISK_MISSING_COLUMNS:
        if col not in profiles:
            continue
        prof = profiles[col]
        rules.append({"rule_id": f"CAND_HRMISS_{rule_id}", "rule_type": "high_risk_missing", "column": col, "description": f"{col} missing values are suspicious in movie candidate generation", "confidence": 0.8, "support": prof["non_null_count"], "usage_role": "candidate_generation_rule"})
        rule_id += 1

    paired_cols = [("RatingValue", "RatingCount"), ("Actors", "Cast"), ("Year", "Release Date"), ("Name", "Year")]
    for a, b in paired_cols:
        if a in df.columns and b in df.columns:
            rules.append({"rule_id": f"CAND_PAIR_{rule_id}", "rule_type": "paired_missing_consistency", "columns": [a, b], "description": f"{a} and {b} should usually co-occur", "confidence": 0.8, "support": len(df), "usage_role": "candidate_generation_rule"})
            rule_id += 1

    rules.extend(build_pair_context_rules(df))
    rules.extend(build_movie_consistency_rules(df, profiles))
    return rules


# ============================================================
# 9. 规则优先级
# ============================================================

def assign_rule_priority(rule):
    rtype = rule.get("rule_type")
    if rtype in {"movie_year_range", "release_date_format", "release_year_consistency", "duration_format", "rating_value_range", "rating_count_format", "review_count_format", "comma_list_format", "people_list_format", "mojibake_or_control_char", "actors_cast_overlap", "rating_value_count_coexistence"}:
        return "medium"
    if rule.get("usage_role", "candidate_generation_rule") == "candidate_generation_rule":
        if rule.get("confidence", 0.0) >= 0.85 and rule.get("support", 0) >= 20:
            return "medium"
        return "low"
    return "medium"


def enrich_rules_with_priority(rule_groups):
    new_groups = {}
    for group_name, rules in rule_groups.items():
        enriched = []
        for r in rules:
            r2 = dict(r)
            r2["priority"] = assign_rule_priority(r2)
            enriched.append(r2)
        new_groups[group_name] = enriched
    return new_groups


# ============================================================
# 10. 主流程
# ============================================================

def build_rule_pool(input_csv, output_json):
    df = pd.read_csv(input_csv, dtype=str)
    df = maybe_drop_unnamed_columns(df)
    for col in df.columns:
        df[col] = normalize_series(df[col])

    profiles = analyze_columns(df)
    schema_rules = build_schema_rules(df, profiles)
    fd_rules = build_fd_rules(df, profiles)
    soft_fd_rules = build_soft_fd_rules(df, profiles)
    candidate_rules = build_candidate_generation_rules(df, profiles)

    rule_groups = {"schema_rules": schema_rules, "fd_rules": fd_rules, "soft_fd_rules": soft_fd_rules, "candidate_rules": candidate_rules}
    rule_groups = enrich_rules_with_priority(rule_groups)

    rule_pool = {
        "metadata": {
            "source_file": input_csv,
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "design_goal": "high_recall_candidate_generation_for_movies",
            "target_candidate_columns": sorted(list(TARGET_CANDIDATE_COLUMNS)),
            "notes": [
                "candidate inclusion is determined only by violating rules in this pool",
                "Id is treated as row/context identifier and is not included as a target candidate column by default",
                "movies rules cover year/date/duration/rating/count/list/text/person metadata",
                "rare_value and rare_pattern are disabled for long-tail free-text/person-list fields to reduce false positives",
                "context rules use movie metadata relations such as Name+Year -> Director/Release Date/Duration/Genre and Actors -> Cast",
            ],
        },
        "column_profiles": profiles,
        "rules": rule_groups,
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(rule_pool, f, ensure_ascii=False, indent=2)

    print(f"[OK] Rule pool saved to: {output_json}")
    print(f"Schema rules: {len(schema_rules)}")
    print(f"FD rules: {len(fd_rules)}")
    print(f"Soft FD rules: {len(soft_fd_rules)}")
    print(f"Candidate generation rules: {len(candidate_rules)}")


if __name__ == "__main__":
    build_rule_pool(INPUT_CSV, OUTPUT_JSON)

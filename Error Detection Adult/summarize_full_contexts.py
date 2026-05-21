import json
import math
import re
from collections import Counter
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# 1. 用户配置区
# ============================================================

INPUT_TABLE_CSV = "/mnt/mydata/dq/projects/Splittree/adult/adult_dirty.csv"
INPUT_SHRUNK_CSV = "candidate_cells_adult.csv"

OUTPUT_JSONL = "candidate_llm_contexts_adult.jsonl"
OUTPUT_CSV = "candidate_llm_contexts_flat_adult.csv"

MISSING_TOKENS = {
    "", "empty", "null", "none", "nan", "na", "n/a", "missing", "unk",
    "unknown", "?", "not available", "contact airline"
}

NUMERIC_RATIO_THRESHOLD = 0.90
CATEGORICAL_UNIQUE_COUNT_THRESHOLD = 80
CATEGORICAL_UNIQUE_RATIO_THRESHOLD = 0.10
ID_LIKE_UNIQUE_RATIO_THRESHOLD = 0.98
CODE_LIKE_AVG_LEN_THRESHOLD = 20

TOP_K_COLUMN_VALUES = 10
TOP_K_ALT_VALUES = 10
TOP_K_SIMILAR_ROWS = 5
TOP_K_ROW_FIELDS_FOR_LLM = 11

ADULT_COLUMNS = {
    "age",
    "workclass",
    "education",
    "maritalstatus",
    "occupation",
    "relationship",
    "race",
    "sex",
    "hoursperweek",
    "country",
    "income",
}

ADULT_CATEGORICAL_COLUMNS = {
    "workclass",
    "education",
    "maritalstatus",
    "occupation",
    "relationship",
    "race",
    "sex",
    "country",
    "income",
}

ADULT_PROFILE_COLUMNS = {
    "age",
    "education",
    "maritalstatus",
    "relationship",
    "sex",
    "hoursperweek",
    "workclass",
    "occupation",
    "income",
}

# 贝叶斯启发式权重
W_RULE = 2.0
W_RARITY = 1.0
W_PATTERN = 0.8
W_NEIGHBOR = 1.2
W_ADULT_CONSISTENCY = 1.4


# ============================================================
# 2. Adult domain / order definitions
# ============================================================

VALID_AGE_BUCKETS = {
    "<18", "18-21", "22-25", "26-30", "31-35", "36-40", "41-45",
    "46-50", "51-55", "56-60", "61-65", "66-70", ">70"
}

VALID_WORKCLASS = {
    "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
    "Local-gov", "State-gov", "Without-pay", "Never-worked"
}

VALID_EDUCATION = {
    "Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th",
    "11th", "12th", "HS-grad", "Some-college", "Assoc-voc",
    "Assoc-acdm", "Bachelors", "Masters", "Prof-school", "Doctorate"
}

VALID_MARITALSTATUS = {
    "Married-civ-spouse", "Divorced", "Never-married", "Separated",
    "Widowed", "Married-spouse-absent", "Married-AF-spouse"
}

VALID_OCCUPATION = {
    "Tech-support", "Craft-repair", "Other-service", "Sales",
    "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
    "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
    "Transport-moving", "Priv-house-serv", "Protective-serv",
    "Armed-Forces"
}

VALID_RELATIONSHIP = {
    "Wife", "Own-child", "Husband", "Not-in-family",
    "Other-relative", "Unmarried"
}

VALID_RACE = {
    "White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"
}

VALID_SEX = {"Male", "Female"}

VALID_INCOME = {"LessThan50K", "MoreThan50K", "<=50K", ">50K", "<50K", ">=50K"}

VALID_COUNTRY = {
    "United-States", "Cambodia", "England", "Puerto-Rico", "Canada",
    "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan",
    "Greece", "South", "China", "Cuba", "Iran", "Honduras",
    "Philippines", "Italy", "Poland", "Jamaica", "Vietnam",
    "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic",
    "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary",
    "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia",
    "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"
}

EDUCATION_ORDER = {
    "Preschool": 0,
    "1st-4th": 1,
    "5th-6th": 2,
    "7th-8th": 3,
    "9th": 4,
    "10th": 5,
    "11th": 6,
    "12th": 7,
    "HS-grad": 8,
    "Some-college": 9,
    "Assoc-voc": 10,
    "Assoc-acdm": 10,
    "Bachelors": 11,
    "Masters": 12,
    "Prof-school": 13,
    "Doctorate": 14,
}


# ============================================================
# 3. 基础函数
# ============================================================

def normalize_value(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.lower() in MISSING_TOKENS:
        return None
    return s


def normalize_for_compare(x):
    if pd.isna(x) or x is None:
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
        elif ch in {" ", "-", "_", "/", ".", ",", "(", ")", "%", ":", ";", "&", "'", "+", "<", ">"}:
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


def parse_int_like(value) -> Optional[int]:
    if value is None:
        return None
    s = str(value).strip().replace(",", "")
    if re.fullmatch(r"[+-]?\d+(\.0+)?", s):
        try:
            return int(float(s))
        except Exception:
            return None
    return None


def parse_age_bucket(value: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    if value is None:
        return None, None

    s = str(value).strip()

    m = re.fullmatch(r"<\s*(\d+)", s)
    if m:
        return 0, int(m.group(1)) - 1

    m = re.fullmatch(r">\s*(\d+)", s)
    if m:
        return int(m.group(1)) + 1, None

    m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return min(a, b), max(a, b)

    n = parse_int_like(s)
    if n is not None:
        return n, n

    return None, None


def is_age_bucket_like(value: Optional[str]) -> bool:
    lo, hi = parse_age_bucket(value)
    if lo is None:
        return False
    if lo < 0:
        return False
    if hi is not None and hi < lo:
        return False
    if hi is not None and hi > 120:
        return False
    return True


def parse_hours_value(value: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    if value is None:
        return None, None

    s = str(value).strip()
    m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", s)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return min(a, b), max(a, b)

    n = parse_int_like(s)
    if n is not None:
        return float(n), float(n)

    return None, None


def is_hours_like(value: Optional[str]) -> bool:
    lo, hi = parse_hours_value(value)
    if lo is None:
        return False
    if lo < 0:
        return False
    if hi is not None and hi < lo:
        return False
    if hi is not None and hi > 120:
        return False
    return True


def canonical_income(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if s in {"LessThan50K", "<=50K", "<50K"}:
        return "LessThan50K"
    if s in {"MoreThan50K", ">50K", ">=50K"}:
        return "MoreThan50K"
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
    if value is None or len(str(value)) == 0:
        return {
            "length": 0,
            "digit_ratio": 0.0,
            "alpha_ratio": 0.0,
            "symbol_ratio": 0.0,
            "upper_ratio": 0.0,
            "lower_ratio": 0.0,
            "space_ratio": 0.0,
        }

    value = str(value)
    n = len(value)
    digits = sum(ch.isdigit() for ch in value)
    alphas = sum(ch.isalpha() for ch in value)
    uppers = sum(ch.isupper() for ch in value)
    lowers = sum(ch.islower() for ch in value)
    spaces = sum(ch.isspace() for ch in value)
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
        ("expected_value=", ","),
        ("expected=", ","),
        ("expected ", ","),
        ("suggests rhs=", ", observed="),
        ("suggests ", ", observed="),
        ("dominant=", ", observed="),
        ("canonical=", ", observed="),
    ]

    for left, right in patterns:
        if left in reason:
            try:
                a = reason.index(left) + len(left)
                if right and right in reason[a:]:
                    b = reason.index(right, a)
                    value = reason[a:b].strip()
                else:
                    value = reason[a:].strip()

                if left == "suggests " and "=" in value:
                    value = value.split("=", 1)[1].strip()

                value = value.strip(":;,. ")
                return value if value else None
            except Exception:
                pass

    return None


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
            "row_id": int(float(row["row_id"])),
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

def is_id_like_column(non_null: pd.Series, unique_ratio: float, numeric_ratio: float, avg_len: float) -> bool:
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


def infer_adult_semantic_type(
    col: str,
    detected_type: str,
    non_null: pd.Series,
    unique_ratio: float,
    numeric_ratio: float,
    avg_len: float,
    top_patterns: List[Dict[str, Any]]
) -> str:
    if col == "age":
        return "age_bucket"
    if col == "workclass":
        return "workclass_category"
    if col == "education":
        return "education_category"
    if col == "maritalstatus":
        return "maritalstatus_category"
    if col == "occupation":
        return "occupation_category"
    if col == "relationship":
        return "relationship_category"
    if col == "race":
        return "race_category"
    if col == "sex":
        return "sex_category"
    if col == "hoursperweek":
        return "hours_range"
    if col == "country":
        return "country_category"
    if col == "income":
        return "income_category"

    if is_id_like_column(non_null, unique_ratio, numeric_ratio, avg_len):
        return "id_like"
    if is_code_like_column(non_null, unique_ratio, numeric_ratio, avg_len, top_patterns):
        return "code_like"
    if detected_type == "numeric":
        return "numeric"
    if detected_type == "categorical":
        return "categorical"
    return "text"


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

        if col == "age":
            detected_type = "categorical"
        elif col == "hoursperweek":
            detected_type = "numeric_or_range"
        elif col in ADULT_CATEGORICAL_COLUMNS:
            detected_type = "categorical"
        elif numeric_ratio >= NUMERIC_RATIO_THRESHOLD:
            detected_type = "numeric"
        elif unique_count <= CATEGORICAL_UNIQUE_COUNT_THRESHOLD or unique_ratio <= CATEGORICAL_UNIQUE_RATIO_THRESHOLD:
            detected_type = "categorical"
        else:
            detected_type = "text"

        semantic_type = infer_adult_semantic_type(
            col, detected_type, non_null, unique_ratio, numeric_ratio, avg_len, pattern_list
        )

        numeric_stats = None
        if detected_type == "numeric":
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

        profiles[col] = {
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

    return profiles


# ============================================================
# 6. Adult 专属上下文
# ============================================================

def build_adult_profile_bundle(row: Dict[str, Any]) -> Dict[str, Any]:
    age = row.get("age")
    hours = row.get("hoursperweek")
    education = row.get("education")
    relationship = row.get("relationship")
    maritalstatus = row.get("maritalstatus")
    sex = row.get("sex")
    workclass = row.get("workclass")
    occupation = row.get("occupation")
    income = row.get("income")

    age_lo, age_hi = parse_age_bucket(age)
    hours_lo, hours_hi = parse_hours_value(hours)

    adult_inconsistencies = []

    if age is not None and not is_age_bucket_like(age):
        adult_inconsistencies.append("invalid_age_bucket")

    if hours is not None and not is_hours_like(hours):
        adult_inconsistencies.append("invalid_hoursperweek")

    if income is not None and canonical_income(income) is None:
        adult_inconsistencies.append("invalid_income_format")

    if relationship in {"Husband", "Wife"} and maritalstatus in {"Never-married", "Divorced", "Separated", "Widowed"}:
        adult_inconsistencies.append("spouse_relationship_but_non_married_status")

    if relationship == "Wife" and sex not in {None, "Female"}:
        adult_inconsistencies.append("wife_relationship_but_not_female")

    if relationship == "Husband" and sex not in {None, "Male"}:
        adult_inconsistencies.append("husband_relationship_but_not_male")

    if age_hi is not None and age_hi < 18 and relationship in {"Husband", "Wife"}:
        adult_inconsistencies.append("very_young_spouse_relationship")

    edu_order = EDUCATION_ORDER.get(education)
    if age_hi is not None and age_hi < 18 and edu_order is not None and edu_order >= EDUCATION_ORDER["Bachelors"]:
        adult_inconsistencies.append("very_young_high_education")

    if age_hi is not None and age_hi <= 21 and education in {"Doctorate", "Prof-school"}:
        adult_inconsistencies.append("young_very_high_education")

    if workclass == "Never-worked" and hours_hi is not None and hours_hi > 0:
        adult_inconsistencies.append("never_worked_nonzero_hours")

    if workclass == "Never-worked" and occupation not in {None, "?"}:
        adult_inconsistencies.append("never_worked_has_occupation")

    if occupation == "Armed-Forces" and workclass not in {None, "Federal-gov"}:
        adult_inconsistencies.append("armed_forces_unusual_workclass")

    return {
        "age": age,
        "age_lower": age_lo,
        "age_upper": age_hi,
        "hoursperweek": hours,
        "hours_lower": hours_lo,
        "hours_upper": hours_hi,
        "education": education,
        "education_order": edu_order,
        "maritalstatus": maritalstatus,
        "relationship": relationship,
        "sex": sex,
        "workclass": workclass,
        "occupation": occupation,
        "race": row.get("race"),
        "country": row.get("country"),
        "income": income,
        "income_canonical": canonical_income(income),
        "adult_profile_inconsistencies": adult_inconsistencies,
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
    adult_profile_bundle = build_adult_profile_bundle(row)

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
        "adult_profile_bundle": adult_profile_bundle,
    }


# ============================================================
# 7. 列上下文
# ============================================================

def build_adult_value_features(col: str, value: Any) -> Dict[str, Any]:
    age_lo, age_hi = parse_age_bucket(value) if col == "age" else (None, None)
    hours_lo, hours_hi = parse_hours_value(value) if col == "hoursperweek" else (None, None)

    canonical = None
    domain_valid = None
    education_order = None

    if col == "income":
        canonical = canonical_income(value)
        domain_valid = canonical is not None
    elif col == "age":
        domain_valid = value in VALID_AGE_BUCKETS if value is not None else None
    elif col == "hoursperweek":
        domain_valid = is_hours_like(value) if value is not None else None
    elif col == "workclass":
        domain_valid = value in VALID_WORKCLASS if value is not None else None
    elif col == "education":
        domain_valid = value in VALID_EDUCATION if value is not None else None
        education_order = EDUCATION_ORDER.get(value)
    elif col == "maritalstatus":
        domain_valid = value in VALID_MARITALSTATUS if value is not None else None
    elif col == "occupation":
        domain_valid = value in VALID_OCCUPATION if value is not None else None
    elif col == "relationship":
        domain_valid = value in VALID_RELATIONSHIP if value is not None else None
    elif col == "race":
        domain_valid = value in VALID_RACE if value is not None else None
    elif col == "sex":
        domain_valid = value in VALID_SEX if value is not None else None
    elif col == "country":
        domain_valid = value in VALID_COUNTRY if value is not None else None

    adult_value_bucket = "unknown"
    if col == "age" and age_lo is not None:
        if age_hi is not None and age_hi < 18:
            adult_value_bucket = "minor"
        elif age_lo <= 25:
            adult_value_bucket = "young_adult"
        elif age_lo <= 50:
            adult_value_bucket = "working_age"
        else:
            adult_value_bucket = "older_adult"
    elif col == "hoursperweek" and hours_lo is not None:
        mid = (hours_lo + (hours_hi if hours_hi is not None else hours_lo)) / 2
        if mid < 20:
            adult_value_bucket = "low_hours"
        elif mid <= 40:
            adult_value_bucket = "standard_hours"
        elif mid <= 60:
            adult_value_bucket = "high_hours"
        else:
            adult_value_bucket = "very_high_hours"

    return {
        "age_lower": age_lo,
        "age_upper": age_hi,
        "hours_lower": hours_lo,
        "hours_upper": hours_hi,
        "canonical_income": canonical,
        "domain_valid": domain_valid,
        "education_order": education_order,
        "adult_value_bucket": adult_value_bucket,
    }


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
    adult_value_features = build_adult_value_features(col, value)

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
        "adult_value_features": adult_value_features,
    }


# ============================================================
# 8. 冲突信息
# ============================================================

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
        + rule_type_counter.get("age_bucket_format", 0)
        + rule_type_counter.get("hours_per_week_format_range", 0)
    )

    schema_rule_count = (
        rule_type_counter.get("not_null", 0)
        + rule_type_counter.get("high_risk_missing", 0)
        + rule_type_counter.get("paired_missing_consistency", 0)
        + pattern_rule_count
        + rule_type_counter.get("workclass_domain", 0)
        + rule_type_counter.get("education_domain", 0)
        + rule_type_counter.get("maritalstatus_domain", 0)
        + rule_type_counter.get("occupation_domain", 0)
        + rule_type_counter.get("relationship_domain", 0)
        + rule_type_counter.get("race_domain", 0)
        + rule_type_counter.get("sex_domain", 0)
        + rule_type_counter.get("country_domain", 0)
        + rule_type_counter.get("income_domain", 0)
        + rule_type_counter.get("income_canonical_format", 0)
    )

    adult_domain_rule_count = (
        rule_type_counter.get("workclass_domain", 0)
        + rule_type_counter.get("education_domain", 0)
        + rule_type_counter.get("maritalstatus_domain", 0)
        + rule_type_counter.get("occupation_domain", 0)
        + rule_type_counter.get("relationship_domain", 0)
        + rule_type_counter.get("race_domain", 0)
        + rule_type_counter.get("sex_domain", 0)
        + rule_type_counter.get("country_domain", 0)
        + rule_type_counter.get("income_domain", 0)
    )

    adult_format_rule_count = (
        rule_type_counter.get("age_bucket_format", 0)
        + rule_type_counter.get("hours_per_week_format_range", 0)
        + rule_type_counter.get("income_canonical_format", 0)
    )

    adult_consistency_rule_count = (
        rule_type_counter.get("age_education_consistency", 0)
        + rule_type_counter.get("age_relationship_consistency", 0)
        + rule_type_counter.get("marital_relationship_consistency", 0)
        + rule_type_counter.get("sex_relationship_consistency", 0)
        + rule_type_counter.get("workclass_occupation_consistency", 0)
        + rule_type_counter.get("education_occupation_consistency", 0)
        + rule_type_counter.get("hours_workclass_consistency", 0)
    )

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

        "adult_domain_rule_count": adult_domain_rule_count,
        "adult_format_rule_count": adult_format_rule_count,
        "adult_consistency_rule_count": adult_consistency_rule_count,

        # 兼容旧训练脚本中的 time_window_rule_count 字段，adult 置 0
        "time_window_rule_count": 0,

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

    if semantic_type in {"numeric"}:
        an = try_parse_one_numeric(a)
        bn = try_parse_one_numeric(b)
        if an is None or bn is None:
            return 0.0
        denom = max(abs(an), abs(bn), 1.0)
        return max(0.0, 1.0 - abs(an - bn) / denom)

    if semantic_type == "age_bucket":
        a_lo, a_hi = parse_age_bucket(a)
        b_lo, b_hi = parse_age_bucket(b)
        if a_lo is None or b_lo is None:
            return 0.0
        a_mid = (a_lo + (a_hi if a_hi is not None else a_lo)) / 2
        b_mid = (b_lo + (b_hi if b_hi is not None else b_lo)) / 2
        return max(0.0, 1.0 - abs(a_mid - b_mid) / 80.0)

    if semantic_type == "hours_range":
        a_lo, a_hi = parse_hours_value(a)
        b_lo, b_hi = parse_hours_value(b)
        if a_lo is None or b_lo is None:
            return 0.0
        a_mid = (a_lo + (a_hi if a_hi is not None else a_lo)) / 2
        b_mid = (b_lo + (b_hi if b_hi is not None else b_lo)) / 2
        return max(0.0, 1.0 - abs(a_mid - b_mid) / 120.0)

    if semantic_type in {
        "categorical", "code_like", "id_like",
        "workclass_category", "education_category", "maritalstatus_category",
        "occupation_category", "relationship_category", "race_category",
        "sex_category", "country_category", "income_category",
    }:
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
# 10. Adult consistency evidence
# ============================================================

def summarize_adult_consistency_info(df: pd.DataFrame, row_id: int, col: str, value: Any) -> Dict[str, Any]:
    row = df.iloc[row_id].to_dict()
    profile = build_adult_profile_bundle(row)

    evidence_flags = list(profile.get("adult_profile_inconsistencies", []))

    if col == "age" and value is not None and not is_age_bucket_like(value):
        evidence_flags.append("target_invalid_age_bucket")

    if col == "hoursperweek" and value is not None and not is_hours_like(value):
        evidence_flags.append("target_invalid_hoursperweek")

    if col == "income" and value is not None and canonical_income(value) is None:
        evidence_flags.append("target_invalid_income_format")

    domain_checks = {
        "workclass": VALID_WORKCLASS,
        "education": VALID_EDUCATION,
        "maritalstatus": VALID_MARITALSTATUS,
        "occupation": VALID_OCCUPATION,
        "relationship": VALID_RELATIONSHIP,
        "race": VALID_RACE,
        "sex": VALID_SEX,
        "country": VALID_COUNTRY,
    }
    if col in domain_checks and value is not None and value not in domain_checks[col]:
        evidence_flags.append(f"target_invalid_{col}_domain")

    same_profile_distribution = {}

    # 同年龄-教育-婚姻画像分布
    profile_cols = ["age", "education", "maritalstatus"]
    if all(c in df.columns for c in profile_cols):
        mask = pd.Series([True] * len(df))
        for c in profile_cols:
            if row.get(c) is None:
                mask = None
                break
            mask = mask & (df[c] == row.get(c))
        if mask is not None:
            sub = df[mask]
            if len(sub) > 1:
                for target in ["relationship", "occupation", "income", "hoursperweek"]:
                    if target in sub.columns:
                        cnt = Counter(sub[target].dropna().tolist())
                        same_profile_distribution[target] = {
                            "top_values": [{"value": k, "count": v} for k, v in cnt.most_common(5)],
                            "group_size": int(len(sub)),
                        }

    # 同 workclass/occupation 的 income、hours 分布
    work_profile_distribution = {}
    work_cols = ["workclass", "occupation"]
    if all(c in df.columns for c in work_cols):
        mask = pd.Series([True] * len(df))
        for c in work_cols:
            if row.get(c) is None:
                mask = None
                break
            mask = mask & (df[c] == row.get(c))
        if mask is not None:
            sub = df[mask]
            if len(sub) > 1:
                for target in ["income", "hoursperweek", "education"]:
                    if target in sub.columns:
                        cnt = Counter(sub[target].dropna().tolist())
                        work_profile_distribution[target] = {
                            "top_values": [{"value": k, "count": v} for k, v in cnt.most_common(5)],
                            "group_size": int(len(sub)),
                        }

    adult_consistency_score = 0.0
    if evidence_flags:
        adult_consistency_score += min(2.0, 0.5 * len(set(evidence_flags)))
    if same_profile_distribution and col in ADULT_PROFILE_COLUMNS:
        adult_consistency_score += 0.5
    if work_profile_distribution and col in {"workclass", "occupation", "income", "hoursperweek", "education"}:
        adult_consistency_score += 0.5

    return {
        "adult_profile_bundle": profile,
        "evidence_flags": evidence_flags,
        "same_profile_distribution": same_profile_distribution,
        "work_profile_distribution": work_profile_distribution,
        "adult_consistency_score": round(float(adult_consistency_score), 6),
    }


# ============================================================
# 11. 贝叶斯先验/后验
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
    adult_consistency_info: Dict[str, Any]
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

    # adult 基本都是类别字段，country/workclass/occupation 的长尾合法值较多，降低 rarity 权重。
    if semantic_type in {"country_category", "occupation_category", "workclass_category"}:
        rarity *= 0.4
    elif semantic_type in {"age_bucket", "hours_range", "income_category"}:
        rarity *= 0.8
    elif "category" in semantic_type:
        rarity *= 0.7
    elif semantic_type == "id_like":
        rarity *= 0.2
    elif semantic_type == "code_like":
        rarity *= 0.6
    elif semantic_type == "text":
        rarity *= 1.0

    neighbor_majority_value = similarity_info.get("neighbor_majority_value")
    neighbor_majority_ratio = similarity_info.get("neighbor_majority_ratio")
    if neighbor_majority_value is None or neighbor_majority_ratio is None:
        neighbor_signal = 0.0
    else:
        neighbor_signal = neighbor_majority_ratio if neighbor_majority_value != value else 0.0

    adult_consistency_signal = adult_consistency_info.get("adult_consistency_score", 0.0)

    logit_post = (
        safe_logit(prior)
        + W_RULE * rule_signal
        + W_RARITY * rarity
        + W_PATTERN * pattern_signal
        + W_NEIGHBOR * neighbor_signal
        + W_ADULT_CONSISTENCY * adult_consistency_signal
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
            "adult_consistency_signal": round(float(adult_consistency_signal), 6),
        },
        "note": "posterior is a heuristic Bayesian-style approximation"
    }


# ============================================================
# 12. 构造给 LLM 的上下文文本
# ============================================================

def build_llm_context_text(
    row_id: int,
    col: str,
    value: Any,
    row_ctx: Dict[str, Any],
    col_ctx: Dict[str, Any],
    conflict_ctx: Dict[str, Any],
    sim_ctx: Dict[str, Any],
    adult_ctx: Dict[str, Any],
    bayes_ctx: Dict[str, Any]
) -> str:
    lines = []

    lines.append("Candidate cell:")
    lines.append(f"- row_id: {row_id}")
    lines.append(f"- column: {col}")
    lines.append(f"- current value: {value}")

    lines.append("")
    lines.append("Row context:")
    lines.append(f"- missing_count: {row_ctx['missing_count']}")
    lines.append(f"- non_missing_count: {row_ctx['non_missing_count']}")
    lines.append(f"- avg_value_length: {row_ctx['avg_value_length']}")
    lines.append("- row field values:")
    for k, v in row_ctx["row_values_for_llm"]:
        lines.append(f"  - {k}: {v}")

    lines.append("")
    lines.append("Adult profile bundle:")
    lines.append(f"- {row_ctx['adult_profile_bundle']}")

    lines.append("")
    lines.append("Column context:")
    lines.append(f"- detected_type: {col_ctx['detected_type']}")
    lines.append(f"- semantic_type: {col_ctx['semantic_type']}")
    lines.append(f"- missing_rate: {col_ctx['missing_rate']}")
    lines.append(f"- unique_rate: {col_ctx['unique_rate']}")
    lines.append(f"- value_frequency: {col_ctx['value_frequency']}")
    lines.append(f"- value_frequency_rank: {col_ctx['value_frequency_rank']}")
    lines.append(f"- current_pattern: {col_ctx['current_value_pattern']}")
    lines.append(f"- dominant_pattern: {col_ctx['dominant_pattern']}")
    lines.append(f"- charset_features: {col_ctx['charset_features']}")
    lines.append(f"- adult_value_features: {col_ctx['adult_value_features']}")
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
    lines.append(f"- pattern_rule_count: {conflict_ctx['pattern_rule_count']}")
    lines.append(f"- schema_rule_count: {conflict_ctx['schema_rule_count']}")
    lines.append(f"- adult_domain_rule_count: {conflict_ctx['adult_domain_rule_count']}")
    lines.append(f"- adult_format_rule_count: {conflict_ctx['adult_format_rule_count']}")
    lines.append(f"- adult_consistency_rule_count: {conflict_ctx['adult_consistency_rule_count']}")

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
    lines.append("Adult consistency evidence:")
    lines.append(f"- evidence_flags: {adult_ctx['evidence_flags']}")
    lines.append(f"- same_profile_distribution: {adult_ctx['same_profile_distribution']}")
    lines.append(f"- work_profile_distribution: {adult_ctx['work_profile_distribution']}")
    lines.append(f"- adult_consistency_score: {adult_ctx['adult_consistency_score']}")

    lines.append("")
    lines.append("Bayesian-style probabilities:")
    lines.append(f"- prior_error_probability: {bayes_ctx['prior_error_probability']}")
    lines.append(f"- posterior_error_probability: {bayes_ctx['posterior_error_probability']}")
    lines.append(f"- evidence: {bayes_ctx['bayes_evidence']}")

    return "\n".join(lines)


# ============================================================
# 13. 主流程
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
            adult_ctx = summarize_adult_consistency_info(df, row_id, col, value)
            bayes_ctx = summarize_bayesian_info(cand, df, col_profiles, candidate_stats, sim_ctx, adult_ctx)

            llm_context_text = build_llm_context_text(
                row_id, col, value, row_ctx, col_ctx, conflict_ctx, sim_ctx, adult_ctx, bayes_ctx
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
                "adult_consistency_context": adult_ctx,
                "bayesian_context": bayes_ctx,
                "llm_context_text": llm_context_text
            }

            f.write(json.dumps(item, ensure_ascii=False) + "\n")

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

                "age_lower": col_ctx["adult_value_features"]["age_lower"],
                "age_upper": col_ctx["adult_value_features"]["age_upper"],
                "hours_lower": col_ctx["adult_value_features"]["hours_lower"],
                "hours_upper": col_ctx["adult_value_features"]["hours_upper"],
                "canonical_income": col_ctx["adult_value_features"]["canonical_income"],
                "domain_valid": col_ctx["adult_value_features"]["domain_valid"],
                "education_order": col_ctx["adult_value_features"]["education_order"],
                "adult_value_bucket": col_ctx["adult_value_features"]["adult_value_bucket"],

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

                "adult_domain_rule_count": conflict_ctx["adult_domain_rule_count"],
                "adult_format_rule_count": conflict_ctx["adult_format_rule_count"],
                "adult_consistency_rule_count": conflict_ctx["adult_consistency_rule_count"],

                # 兼容旧训练/采样脚本
                "time_window_rule_count": 0,
                "time_value_minutes": np.nan,
                "time_window_bucket": "adult_not_applicable",
                "time_value_bucket": "adult_not_applicable",

                "row_missing_count": row_ctx["missing_count"],
                "row_non_missing_count": row_ctx["non_missing_count"],

                "row_age_lower": row_ctx["adult_profile_bundle"]["age_lower"],
                "row_age_upper": row_ctx["adult_profile_bundle"]["age_upper"],
                "row_hours_lower": row_ctx["adult_profile_bundle"]["hours_lower"],
                "row_hours_upper": row_ctx["adult_profile_bundle"]["hours_upper"],
                "row_education_order": row_ctx["adult_profile_bundle"]["education_order"],
                "row_income_canonical": row_ctx["adult_profile_bundle"]["income_canonical"],
                "adult_profile_inconsistency_count": len(row_ctx["adult_profile_bundle"]["adult_profile_inconsistencies"]),

                "adult_consistency_score": adult_ctx["adult_consistency_score"],
                "adult_evidence_flags": "|".join(adult_ctx["evidence_flags"]),

                "current_value_pattern": col_ctx["current_value_pattern"],
                "dominant_pattern": col_ctx["dominant_pattern"],
                "dominant_pattern_ratio": col_ctx["dominant_pattern_ratio"],
            })

    pd.DataFrame(flat_rows).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"[OK] Full LLM contexts saved to: {OUTPUT_JSONL}")
    print(f"[OK] Flat summary saved to: {OUTPUT_CSV}")
    print(f"[INFO] Context rows: {len(flat_rows)}")


if __name__ == "__main__":
    main()

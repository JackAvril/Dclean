import json
import math
import os
import re
import time
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

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

TOP_K_COLUMN_VALUES = 10
TOP_K_ALT_VALUES = 10
TOP_K_SIMILAR_ROWS = 5
TOP_K_ROW_FIELDS_FOR_LLM = 11
PROFILE_TOPK_VALUES = 5
PROGRESS_EVERY = 1000

# 多进程加速参数：保留最终 JSONL/CSV 文件名与字段不变
NUM_WORKERS = 16
CHUNK_SIZE = 800
TMP_PART_DIR = "_tmp_candidate_llm_contexts_adult_parts"
KEEP_TMP_PARTS = True
KEEP_OUTPUT_ORDER = True

# 速度/体积开关
INCLUDE_SIMILAR_ROW_VALUES = True   # 如 JSONL 太大或写入太慢，可改 False
SIMILARITY_BATCH_SIZE = 2048        # 行数很大时可调小，如 512/1024

NUMERIC_RATIO_THRESHOLD = 0.90
CATEGORICAL_UNIQUE_COUNT_THRESHOLD = 80
CATEGORICAL_UNIQUE_RATIO_THRESHOLD = 0.10
ID_LIKE_UNIQUE_RATIO_THRESHOLD = 0.98
CODE_LIKE_AVG_LEN_THRESHOLD = 20

# Bayesian-style heuristic weights
W_RULE = 2.0
W_RARITY = 1.0
W_PATTERN = 0.8
W_NEIGHBOR = 1.2
W_ADULT_CONSISTENCY = 1.4

ADULT_CATEGORICAL_COLUMNS = {
    "workclass", "education", "maritalstatus", "occupation", "relationship",
    "race", "sex", "country", "income"
}
ADULT_PROFILE_COLUMNS = {
    "age", "education", "maritalstatus", "relationship", "sex",
    "hoursperweek", "workclass", "occupation", "income"
}

# ============================================================
# Adult v2 candidate attribution features
# ============================================================
PRIMARY_TARGET_COLUMNS = {"sex", "relationship", "education"}

CONTEXT_ONLY_COLUMNS = {
    "age",
    "workclass",
    "maritalstatus",
    "occupation",
    "race",
    "hoursperweek",
    "country",
    "income",
}

RELATIONSHIP_SPOUSE_VALUES = {"Husband", "Wife"}
NON_MARRIED_STATUS_FOR_SPOUSE = {"Never-married", "Divorced", "Separated", "Widowed"}
VALID_SEX_BY_RELATIONSHIP = {"Husband": "Male", "Wife": "Female"}

RELATIONSHIP_V2_RULE_TYPES = {
    "relationship_marital_spouse_conflict",
    "relationship_age_spouse_conflict",
    "relationship_sex_spouse_conflict",
    "relationship_context_dominant",
}

EDUCATION_V2_RULE_TYPES = {
    "education_age_extreme_conflict",
}

VALID_AGE_BUCKETS = {"<18", "18-21", "22-25", "26-30", "31-35", "36-40", "41-45", "46-50", "51-55", "56-60", "61-65", "66-70", ">70"}
VALID_WORKCLASS = {"Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"}
VALID_EDUCATION = {"Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th", "11th", "12th", "HS-grad", "Some-college", "Assoc-voc", "Assoc-acdm", "Bachelors", "Masters", "Prof-school", "Doctorate"}
VALID_MARITALSTATUS = {"Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"}
VALID_OCCUPATION = {"Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"}
VALID_RELATIONSHIP = {"Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"}
VALID_RACE = {"White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"}
VALID_SEX = {"Male", "Female"}
VALID_COUNTRY = {"United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"}

EDUCATION_ORDER = {
    "Preschool": 0, "1st-4th": 1, "5th-6th": 2, "7th-8th": 3,
    "9th": 4, "10th": 5, "11th": 6, "12th": 7,
    "HS-grad": 8, "Some-college": 9, "Assoc-voc": 10,
    "Assoc-acdm": 10, "Bachelors": 11, "Masters": 12,
    "Prof-school": 13, "Doctorate": 14,
}

DOMAIN_MAP = {
    "workclass": VALID_WORKCLASS,
    "education": VALID_EDUCATION,
    "maritalstatus": VALID_MARITALSTATUS,
    "occupation": VALID_OCCUPATION,
    "relationship": VALID_RELATIONSHIP,
    "race": VALID_RACE,
    "sex": VALID_SEX,
    "country": VALID_COUNTRY,
}

_PATTERN_CACHE: Dict[Any, str] = {}
_CHARSET_CACHE: Dict[Any, Dict[str, float]] = {}
_AGE_CACHE: Dict[Any, Tuple[Optional[int], Optional[int]]] = {}
_HOURS_CACHE: Dict[Any, Tuple[Optional[float], Optional[float]]] = {}
_ADULT_VALUE_FEATURE_CACHE: Dict[Tuple[str, Any], Dict[str, Any]] = {}


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
    if value in _PATTERN_CACHE:
        return _PATTERN_CACHE[value]
    s = str(value)
    parts, prev_type, cnt = [], None, 0
    for ch in s:
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
            prev_type, cnt = cur, 1
    if prev_type is not None:
        parts.append(f"{prev_type}{cnt}")
    pat = "".join(parts)
    _PATTERN_CACHE[value] = pat
    return pat


def try_parse_numeric_series(series: pd.Series):
    cleaned = series.dropna().astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False)
    numeric = pd.to_numeric(cleaned, errors="coerce")
    success_ratio = numeric.notna().mean() if len(cleaned) > 0 else 0.0
    return numeric, success_ratio


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


def parse_age_bucket(value) -> Tuple[Optional[int], Optional[int]]:
    if value in _AGE_CACHE:
        return _AGE_CACHE[value]
    if value is None:
        res = (None, None)
    else:
        s = str(value).strip()
        m = re.fullmatch(r"<\s*(\d+)", s)
        if m:
            res = (0, int(m.group(1)) - 1)
        else:
            m = re.fullmatch(r">\s*(\d+)", s)
            if m:
                res = (int(m.group(1)) + 1, None)
            else:
                m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", s)
                if m:
                    a, b = int(m.group(1)), int(m.group(2))
                    res = (min(a, b), max(a, b))
                else:
                    n = parse_int_like(s)
                    res = (n, n) if n is not None else (None, None)
    _AGE_CACHE[value] = res
    return res


def is_age_bucket_like(value) -> bool:
    lo, hi = parse_age_bucket(value)
    return lo is not None and lo >= 0 and (hi is None or (hi >= lo and hi <= 120))


def parse_hours_value(value) -> Tuple[Optional[float], Optional[float]]:
    if value in _HOURS_CACHE:
        return _HOURS_CACHE[value]
    if value is None:
        res = (None, None)
    else:
        s = str(value).strip()
        m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", s)
        if m:
            a, b = float(m.group(1)), float(m.group(2))
            res = (min(a, b), max(a, b))
        else:
            n = parse_int_like(s)
            res = (float(n), float(n)) if n is not None else (None, None)
    _HOURS_CACHE[value] = res
    return res


def is_hours_like(value) -> bool:
    lo, hi = parse_hours_value(value)
    return lo is not None and lo >= 0 and (hi is None or (hi >= lo and hi <= 120))


def canonical_income(value) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if s in {"LessThan50K", "<=50K", "<50K"}:
        return "LessThan50K"
    if s in {"MoreThan50K", ">50K", ">=50K"}:
        return "MoreThan50K"
    return None


def safe_logit(p: float) -> float:
    eps = 1e-6
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    if x >= 40:
        return 1.0
    if x <= -40:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def charset_features(value) -> Dict[str, float]:
    if value in _CHARSET_CACHE:
        return _CHARSET_CACHE[value]
    if value is None or len(str(value)) == 0:
        res = {"length": 0, "digit_ratio": 0.0, "alpha_ratio": 0.0, "symbol_ratio": 0.0, "upper_ratio": 0.0, "lower_ratio": 0.0, "space_ratio": 0.0}
        _CHARSET_CACHE[value] = res
        return res
    s = str(value)
    n = len(s)
    digits = sum(ch.isdigit() for ch in s)
    alphas = sum(ch.isalpha() for ch in s)
    uppers = sum(ch.isupper() for ch in s)
    lowers = sum(ch.islower() for ch in s)
    spaces = sum(ch.isspace() for ch in s)
    symbols = n - digits - alphas - spaces
    res = {
        "length": n,
        "digit_ratio": round(digits / n, 6),
        "alpha_ratio": round(alphas / n, 6),
        "symbol_ratio": round(symbols / n, 6),
        "upper_ratio": round(uppers / n, 6),
        "lower_ratio": round(lowers / n, 6),
        "space_ratio": round(spaces / n, 6),
    }
    _CHARSET_CACHE[value] = res
    return res


def parse_pipe_list(s: Any, sep: str = "|") -> List[str]:
    if pd.isna(s) or s is None:
        return []
    s = str(s).strip()
    return [x.strip() for x in s.split(sep) if x.strip()] if s else []


def parse_reason_list(s: Any) -> List[str]:
    if pd.isna(s) or s is None:
        return []
    s = str(s).strip()
    return [x.strip() for x in s.split("||") if x.strip()] if s else []


def extract_expected_value_from_reason(reason: str):
    if not reason:
        return None
    patterns = [
        ("expected_value=", ","), ("expected=", ","), ("expected ", ","),
        ("suggests rhs=", ", observed="), ("suggests ", ", observed="),
        ("dominant=", ", observed="), ("canonical=", ", observed="),
    ]
    for left, right in patterns:
        if left in reason:
            try:
                a = reason.index(left) + len(left)
                b = reason.index(right, a) if right and right in reason[a:] else len(reason)
                value = reason[a:b].strip()
                if left == "suggests " and "=" in value:
                    value = value.split("=", 1)[1].strip()
                value = value.strip(":;,. ")
                return value if value else None
            except Exception:
                pass
    return None


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return str(obj)


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
            "value": normalize_value(row.get("value")),
            "violation_count": int(float(row["violation_count"])) if pd.notna(row.get("violation_count")) else 0,
            "conflict_score": float(row["conflict_score"]) if pd.notna(row.get("conflict_score")) else 0.0,
            "rule_ids": parse_pipe_list(row.get("rule_ids")),
            "rule_types": parse_pipe_list(row.get("rule_types")),
            "priorities": parse_pipe_list(row.get("priorities")),
            "usage_roles": parse_pipe_list(row.get("usage_roles")),
            "reasons": parse_reason_list(row.get("reasons")),
        }
        violated_rules = []
        max_len = max(len(item["rule_ids"]), len(item["rule_types"]), len(item["priorities"]), len(item["usage_roles"]), len(item["reasons"]), 0)
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
# 5. 列统计
# ============================================================

def value_frequency_dict(series: pd.Series, topk=20):
    cnt = Counter(series.dropna().tolist())
    total = len(series.dropna())
    return [{"value": k, "count": v, "ratio": round(v / total, 6) if total > 0 else 0.0} for k, v in cnt.most_common(topk)]


def build_value_rank_map(counter: Counter) -> Dict[Any, int]:
    return {v: idx for idx, (v, _) in enumerate(counter.most_common(), start=1)}


def is_id_like_column(non_null: pd.Series, unique_ratio: float, numeric_ratio: float, avg_len: float) -> bool:
    return len(non_null) > 0 and ((unique_ratio >= ID_LIKE_UNIQUE_RATIO_THRESHOLD and avg_len <= CODE_LIKE_AVG_LEN_THRESHOLD) or (unique_ratio >= ID_LIKE_UNIQUE_RATIO_THRESHOLD and numeric_ratio >= NUMERIC_RATIO_THRESHOLD))


def is_code_like_column(non_null: pd.Series, unique_ratio: float, numeric_ratio: float, avg_len: float, top_patterns: List[Dict[str, Any]]) -> bool:
    if len(non_null) == 0:
        return False
    if avg_len <= CODE_LIKE_AVG_LEN_THRESHOLD and unique_ratio > CATEGORICAL_UNIQUE_RATIO_THRESHOLD and top_patterns and top_patterns[0]["ratio"] >= 0.7:
        return True
    if avg_len <= 15 and numeric_ratio >= 0.95 and unique_ratio < ID_LIKE_UNIQUE_RATIO_THRESHOLD:
        return True
    return False


def infer_semantic_type(col, detected_type, non_null, unique_ratio, numeric_ratio, avg_len, top_patterns):
    explicit = {
        "age": "age_bucket", "workclass": "workclass_category", "education": "education_category",
        "maritalstatus": "maritalstatus_category", "occupation": "occupation_category",
        "relationship": "relationship_category", "race": "race_category", "sex": "sex_category",
        "hoursperweek": "hours_range", "country": "country_category", "income": "income_category",
    }
    if col in explicit:
        return explicit[col]
    if is_id_like_column(non_null, unique_ratio, numeric_ratio, avg_len):
        return "id_like"
    if is_code_like_column(non_null, unique_ratio, numeric_ratio, avg_len, top_patterns):
        return "code_like"
    return detected_type if detected_type in {"numeric", "categorical"} else "text"


def analyze_columns(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    profiles = {}
    for col in df.columns:
        s = df[col]
        non_null = s.dropna()
        missing_rate = 1.0 - (len(non_null) / len(df) if len(df) > 0 else 0.0)
        unique_count = non_null.nunique()
        unique_ratio = unique_count / len(non_null) if len(non_null) > 0 else 0.0
        top_values = value_frequency_dict(non_null, topk=TOP_K_COLUMN_VALUES)
        counter = Counter(non_null.tolist())
        rank_map = build_value_rank_map(counter)
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
        pattern_counter = Counter(non_null.map(detect_basic_pattern).tolist())
        total_patterns = sum(pattern_counter.values())
        pattern_list = [{"pattern": p, "count": c, "ratio": round(c / total_patterns, 6)} for p, c in pattern_counter.most_common(10)]
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
        semantic_type = infer_semantic_type(col, detected_type, non_null, unique_ratio, numeric_ratio, avg_len, pattern_list)
        numeric_stats = None
        if detected_type == "numeric":
            nums = numeric_vals.dropna()
            if len(nums) > 0:
                numeric_stats = {"min": float(nums.min()), "max": float(nums.max()), "mean": float(nums.mean()), "std": float(nums.std(ddof=0)) if len(nums) > 1 else 0.0, "q1": float(nums.quantile(0.25)), "median": float(nums.quantile(0.5)), "q3": float(nums.quantile(0.75))}
        profiles[col] = {
            "column": col, "missing_rate": round(missing_rate, 6), "non_null_count": int(len(non_null)),
            "unique_count": int(unique_count), "unique_ratio": round(unique_ratio, 6),
            "detected_type": detected_type, "semantic_type": semantic_type, "top_values": top_values,
            "value_counter": counter, "value_rank_map": rank_map, "length_stats": length_stats,
            "numeric_ratio": round(float(numeric_ratio), 6), "numeric_stats": numeric_stats,
            "patterns": pattern_list, "pattern_counter": pattern_counter,
            "dominant_pattern": pattern_list[0]["pattern"] if pattern_list else None,
            "dominant_pattern_ratio": pattern_list[0]["ratio"] if pattern_list else None,
        }
    return profiles


# ============================================================
# 6. 行上下文、Adult consistency、分组分布预计算
# ============================================================

def build_adult_profile_bundle(row: Dict[str, Any]) -> Dict[str, Any]:
    age, hours = row.get("age"), row.get("hoursperweek")
    education, relationship = row.get("education"), row.get("relationship")
    maritalstatus, sex = row.get("maritalstatus"), row.get("sex")
    workclass, occupation, income = row.get("workclass"), row.get("occupation"), row.get("income")
    age_lo, age_hi = parse_age_bucket(age)
    hours_lo, hours_hi = parse_hours_value(hours)
    flags = []
    if age is not None and not is_age_bucket_like(age): flags.append("invalid_age_bucket")
    if hours is not None and not is_hours_like(hours): flags.append("invalid_hoursperweek")
    if income is not None and canonical_income(income) is None: flags.append("invalid_income_format")
    if relationship in {"Husband", "Wife"} and maritalstatus in {"Never-married", "Divorced", "Separated", "Widowed"}: flags.append("spouse_relationship_but_non_married_status")
    if relationship == "Wife" and sex not in {None, "Female"}: flags.append("wife_relationship_but_not_female")
    if relationship == "Husband" and sex not in {None, "Male"}: flags.append("husband_relationship_but_not_male")
    if age_hi is not None and age_hi < 18 and relationship in {"Husband", "Wife"}: flags.append("very_young_spouse_relationship")
    edu_order = EDUCATION_ORDER.get(education)
    if age_hi is not None and age_hi < 18 and edu_order is not None and edu_order >= EDUCATION_ORDER["Bachelors"]: flags.append("very_young_high_education")
    if age_hi is not None and age_hi <= 21 and education in {"Doctorate", "Prof-school"}: flags.append("young_very_high_education")
    if workclass == "Never-worked" and hours_hi is not None and hours_hi > 0: flags.append("never_worked_nonzero_hours")
    if workclass == "Never-worked" and occupation not in {None, "?"}: flags.append("never_worked_has_occupation")
    if occupation == "Armed-Forces" and workclass not in {None, "Federal-gov"}: flags.append("armed_forces_unusual_workclass")
    return {
        "age": age, "age_lower": age_lo, "age_upper": age_hi,
        "hoursperweek": hours, "hours_lower": hours_lo, "hours_upper": hours_hi,
        "education": education, "education_order": edu_order, "maritalstatus": maritalstatus,
        "relationship": relationship, "sex": sex, "workclass": workclass, "occupation": occupation,
        "race": row.get("race"), "country": row.get("country"), "income": income,
        "income_canonical": canonical_income(income), "adult_profile_inconsistencies": flags,
    }


def summarize_row_context(df: pd.DataFrame, row_id: int, col_profiles: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    row = df.iloc[row_id].to_dict()
    values = [v for v in row.values() if v is not None]
    missing_count = sum(v is None for v in row.values())
    lengths = [len(str(v)) for v in values]
    semantic_type_counts = Counter(col_profiles[col]["semantic_type"] for col, v in row.items() if v is not None)
    row_patterns = {col: detect_basic_pattern(v) if v is not None else "NULL" for col, v in row.items()}
    return {
        "row_values": row,
        "row_values_for_llm": list(row.items())[:TOP_K_ROW_FIELDS_FOR_LLM],
        "row_patterns": row_patterns,
        "missing_count": int(missing_count),
        "non_missing_count": int(len(values)),
        "avg_value_length": round(float(np.mean(lengths)), 6) if lengths else None,
        "max_value_length": int(max(lengths)) if lengths else None,
        "min_value_length": int(min(lengths)) if lengths else None,
        "semantic_type_counts": dict(semantic_type_counts),
        "adult_profile_bundle": build_adult_profile_bundle(row),
    }


def precompute_row_contexts(df, col_profiles):
    return [summarize_row_context(df, i, col_profiles) for i in range(len(df))]


def top_values_for_group(sub: pd.DataFrame, target: str) -> Dict[str, Any]:
    cnt = Counter(sub[target].dropna().tolist())
    return {"top_values": [{"value": k, "count": v} for k, v in cnt.most_common(PROFILE_TOPK_VALUES)], "group_size": int(len(sub))}


def precompute_profile_distributions(df: pd.DataFrame):
    same_profile_map, work_profile_map = {}, {}
    if all(c in df.columns for c in ["age", "education", "maritalstatus"]):
        for key, sub in df.groupby(["age", "education", "maritalstatus"], dropna=False, sort=False):
            if len(sub) <= 1:
                continue
            same_profile_map[key] = {target: top_values_for_group(sub, target) for target in ["relationship", "occupation", "income", "hoursperweek"] if target in sub.columns}
    if all(c in df.columns for c in ["workclass", "occupation"]):
        for key, sub in df.groupby(["workclass", "occupation"], dropna=False, sort=False):
            if len(sub) <= 1:
                continue
            work_profile_map[key] = {target: top_values_for_group(sub, target) for target in ["income", "hoursperweek", "education"] if target in sub.columns}
    return same_profile_map, work_profile_map


# ============================================================
# 7. 列上下文、冲突上下文
# ============================================================

def build_adult_value_features(col: str, value: Any) -> Dict[str, Any]:
    key = (col, value)
    if key in _ADULT_VALUE_FEATURE_CACHE:
        return _ADULT_VALUE_FEATURE_CACHE[key]
    age_lo, age_hi = parse_age_bucket(value) if col == "age" else (None, None)
    hours_lo, hours_hi = parse_hours_value(value) if col == "hoursperweek" else (None, None)
    canonical, domain_valid, education_order = None, None, None
    if col == "income":
        canonical = canonical_income(value)
        domain_valid = canonical is not None
    elif col == "age":
        domain_valid = value in VALID_AGE_BUCKETS if value is not None else None
    elif col == "hoursperweek":
        domain_valid = is_hours_like(value) if value is not None else None
    elif col in DOMAIN_MAP:
        domain_valid = value in DOMAIN_MAP[col] if value is not None else None
        if col == "education":
            education_order = EDUCATION_ORDER.get(value)
    bucket = "unknown"
    if col == "age" and age_lo is not None:
        bucket = "minor" if age_hi is not None and age_hi < 18 else "young_adult" if age_lo <= 25 else "working_age" if age_lo <= 50 else "older_adult"
    elif col == "hoursperweek" and hours_lo is not None:
        mid = (hours_lo + (hours_hi if hours_hi is not None else hours_lo)) / 2
        bucket = "low_hours" if mid < 20 else "standard_hours" if mid <= 40 else "high_hours" if mid <= 60 else "very_high_hours"
    out = {"age_lower": age_lo, "age_upper": age_hi, "hours_lower": hours_lo, "hours_upper": hours_hi, "canonical_income": canonical, "domain_valid": domain_valid, "education_order": education_order, "adult_value_bucket": bucket}
    _ADULT_VALUE_FEATURE_CACHE[key] = out
    return out


def summarize_column_context(col, value, col_profiles):
    prof = col_profiles[col]
    counter = prof["value_counter"]
    return {
        "column": col, "detected_type": prof["detected_type"], "semantic_type": prof["semantic_type"],
        "missing_rate": prof["missing_rate"], "unique_rate": prof["unique_ratio"], "unique_count": prof["unique_count"],
        "non_null_count": prof["non_null_count"], "value_frequency": int(counter.get(value, 0)),
        "value_frequency_rank": prof["value_rank_map"].get(value), "top_values_with_counts": prof["top_values"],
        "length_stats": prof["length_stats"], "numeric_ratio": prof["numeric_ratio"], "numeric_stats": prof["numeric_stats"],
        "current_value_pattern": detect_basic_pattern(value) if value is not None else "NULL",
        "dominant_pattern": prof["dominant_pattern"], "dominant_pattern_ratio": prof["dominant_pattern_ratio"],
        "top_patterns_with_counts": prof["patterns"], "charset_features": charset_features(value),
        "adult_value_features": build_adult_value_features(col, value),
    }


def summarize_conflict_info(candidate, col_profiles):
    col, current_value = candidate["column"], candidate["value"]
    violated_rules = candidate.get("violated_rules", [])
    rule_type_counter, priority_counter, usage_role_counter = Counter(), Counter(), Counter()
    expected, seen = [], set()
    for r in violated_rules:
        rt, pr, ur = r.get("rule_type"), r.get("priority"), r.get("usage_role")
        if rt: rule_type_counter[rt] += 1
        if pr: priority_counter[pr] += 1
        if ur: usage_role_counter[ur] += 1
        ev = r.get("expected_value")
        if ev is not None and ev != current_value and ev not in seen:
            expected.append({"source": "rule_reason_parse", "value": ev, "rule_id": r.get("rule_id"), "rule_type": rt, "priority": pr, "usage_role": ur, "reason": r.get("reason")})
            seen.add(ev)
    alt_col = []
    for item in col_profiles[col]["top_values"]:
        v = item["value"]
        if v != current_value and v not in seen:
            alt_col.append({"source": "column_top_value", "value": v, "count": item["count"]})
            seen.add(v)
        if len(alt_col) >= TOP_K_ALT_VALUES:
            break
    main_rule_type = rule_type_counter.most_common(1)[0][0] if rule_type_counter else "none"
    main_usage_role = usage_role_counter.most_common(1)[0][0] if usage_role_counter else "none"
    fd_like = rule_type_counter.get("functional_dependency", 0) + rule_type_counter.get("soft_functional_dependency", 0)
    context_rule = rule_type_counter.get("dominant_value_by_context", 0) + rule_type_counter.get("dominant_value_by_context_pair", 0)
    adult_domain = sum(rule_type_counter.get(x, 0) for x in ["workclass_domain", "education_domain", "maritalstatus_domain", "occupation_domain", "relationship_domain", "race_domain", "sex_domain", "country_domain", "income_domain"])
    adult_format = rule_type_counter.get("age_bucket_format", 0) + rule_type_counter.get("hours_per_week_format_range", 0) + rule_type_counter.get("income_canonical_format", 0)
    adult_cons = sum(rule_type_counter.get(x, 0) for x in [
        "age_education_consistency",
        "age_relationship_consistency",
        "marital_relationship_consistency",
        "sex_relationship_consistency",
        "workclass_occupation_consistency",
        "education_occupation_consistency",
        "hours_workclass_consistency",
        "relationship_marital_spouse_conflict",
        "relationship_age_spouse_conflict",
        "relationship_sex_spouse_conflict",
        "relationship_context_dominant",
        "education_age_extreme_conflict",
    ])
    pattern_rule = rule_type_counter.get("pattern", 0) + rule_type_counter.get("rare_pattern", 0) + rule_type_counter.get("age_bucket_format", 0) + rule_type_counter.get("hours_per_week_format_range", 0)
    schema_rule = rule_type_counter.get("not_null", 0) + rule_type_counter.get("high_risk_missing", 0) + rule_type_counter.get("paired_missing_consistency", 0) + pattern_rule + adult_domain + rule_type_counter.get("income_canonical_format", 0)
    return {
        "has_rule_violation": len(violated_rules) > 0, "violation_count": len(violated_rules),
        "violated_rules_detailed": violated_rules, "rule_ids": [r.get("rule_id") for r in violated_rules],
        "rule_types": [r.get("rule_type") for r in violated_rules], "priorities": [r.get("priority") for r in violated_rules],
        "usage_roles": [r.get("usage_role") for r in violated_rules], "rule_type_counter": dict(rule_type_counter),
        "priority_counter": dict(priority_counter), "usage_role_counter": dict(usage_role_counter),
        "main_rule_type": main_rule_type, "main_usage_role": main_usage_role,
        "strong_rule_count": usage_role_counter.get("strong_rule", 0),
        "candidate_generation_rule_count": usage_role_counter.get("candidate_generation_rule", 0),
        "fd_like_count": fd_like, "context_rule_count": context_rule,
        "global_rule_count": rule_type_counter.get("global_dominant_value", 0),
        "rare_value_count": rule_type_counter.get("rare_value", 0),
        "typo_rule_count": rule_type_counter.get("dominant_value_typo", 0),
        "pattern_rule_count": pattern_rule, "schema_rule_count": schema_rule,
        "adult_domain_rule_count": adult_domain, "adult_format_rule_count": adult_format,
        "adult_consistency_rule_count": adult_cons, "time_window_rule_count": 0,
        "expected_values_from_rules": expected, "alternative_values_from_column": alt_col,
        "all_alternative_values": expected + alt_col,
    }


# ============================================================
# 8. 相似行快速预计算
# ============================================================

def encode_cell_for_similarity(series: pd.Series, semantic_type: str) -> np.ndarray:
    n = len(series)
    if semantic_type == "age_bucket":
        vals = []
        for v in series.tolist():
            lo, hi = parse_age_bucket(v)
            vals.append(0.0 if lo is None else ((lo + (hi if hi is not None else lo)) / 2.0) / 80.0)
        return np.array(vals, dtype=np.float32).reshape(n, 1)
    if semantic_type == "hours_range":
        vals = []
        for v in series.tolist():
            lo, hi = parse_hours_value(v)
            vals.append(0.0 if lo is None else ((lo + (hi if hi is not None else lo)) / 2.0) / 120.0)
        return np.array(vals, dtype=np.float32).reshape(n, 1)
    if semantic_type == "numeric":
        nums = pd.to_numeric(series.astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False), errors="coerce")
        arr = nums.fillna(nums.median() if nums.notna().any() else 0.0).astype(float).values
        std = np.std(arr) or 1.0
        return ((arr - np.mean(arr)) / std).astype(np.float32).reshape(n, 1)
    vals = series.fillna("__MISSING__").astype(str)
    codes, _ = pd.factorize(vals, sort=False)
    max_code = max(int(codes.max()), 1)
    codes_norm = codes.astype(np.float32) / max_code
    hash_mod = np.array([hash(x) % 997 for x in vals], dtype=np.float32) / 997.0
    return np.stack([codes_norm, hash_mod], axis=1).astype(np.float32)


def build_similarity_matrix(df: pd.DataFrame, col_profiles: Dict[str, Dict[str, Any]]) -> np.ndarray:
    blocks = [encode_cell_for_similarity(df[col], col_profiles[col]["semantic_type"]) for col in df.columns]
    X = np.concatenate(blocks, axis=1).astype(np.float32)
    mean, std = X.mean(axis=0, keepdims=True), X.std(axis=0, keepdims=True)
    std[std <= 1e-8] = 1.0
    X = (X - mean) / std
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    norm[norm <= 1e-8] = 1.0
    return (X / norm).astype(np.float32)


def precompute_neighbor_indices(X: np.ndarray, top_k: int) -> Dict[int, List[Tuple[int, float]]]:
    n = X.shape[0]
    result = {}
    k = min(top_k + 1, n)
    for start in range(0, n, SIMILARITY_BATCH_SIZE):
        end = min(start + SIMILARITY_BATCH_SIZE, n)
        sims = X[start:end] @ X.T
        for local_i, row_id in enumerate(range(start, end)):
            sims[local_i, row_id] = -np.inf
        idx_part = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k] if k < n else np.argsort(-sims, axis=1)[:, :k]
        for local_i, row_id in enumerate(range(start, end)):
            idxs = idx_part[local_i]
            scores = sims[local_i, idxs]
            order = np.argsort(-scores)
            pairs = []
            for j in order:
                score = float(scores[j])
                if not np.isfinite(score):
                    continue
                pairs.append((int(idxs[j]), round((score + 1.0) / 2.0, 6)))
                if len(pairs) >= top_k:
                    break
            result[row_id] = pairs
    return result


def summarize_similarity_info_fast(df: pd.DataFrame, row_id: int, target_col: str, neighbor_indices: Dict[int, List[Tuple[int, float]]]):
    pairs = neighbor_indices.get(row_id, [])[:TOP_K_SIMILAR_ROWS]
    similar_rows, target_values = [], []
    for idx, sim in pairs:
        target_val = df.iloc[idx][target_col]
        item = {"row_id": int(idx), "similarity": sim, "target_column_value": target_val}
        if INCLUDE_SIMILAR_ROW_VALUES:
            item["row_values"] = df.iloc[idx].to_dict()
        similar_rows.append(item)
        target_values.append(target_val)
    dist = Counter([v for v in target_values if v is not None])
    majority_value = dist.most_common(1)[0][0] if dist else None
    majority_ratio = round(dist.most_common(1)[0][1] / len(target_values), 6) if dist and target_values else None
    return {"topk_similar_rows": similar_rows, "neighbor_value_distribution": dict(dist), "neighbor_majority_value": majority_value, "neighbor_majority_ratio": majority_ratio}


# ============================================================
# 9. adult consistency / bayes / 文本
# ============================================================

def summarize_adult_consistency_info_fast(df, row_id, col, value, row_contexts, same_profile_map, work_profile_map):
    row = df.iloc[row_id].to_dict()
    profile = row_contexts[row_id]["adult_profile_bundle"]
    flags = list(profile.get("adult_profile_inconsistencies", []))
    if col == "age" and value is not None and not is_age_bucket_like(value): flags.append("target_invalid_age_bucket")
    if col == "hoursperweek" and value is not None and not is_hours_like(value): flags.append("target_invalid_hoursperweek")
    if col == "income" and value is not None and canonical_income(value) is None: flags.append("target_invalid_income_format")
    if col in DOMAIN_MAP and value is not None and value not in DOMAIN_MAP[col]: flags.append(f"target_invalid_{col}_domain")
    same_dist = same_profile_map.get((row.get("age"), row.get("education"), row.get("maritalstatus")), {})
    work_dist = work_profile_map.get((row.get("workclass"), row.get("occupation")), {})
    score = 0.0
    if flags:
        score += min(2.0, 0.5 * len(set(flags)))
    if same_dist and col in ADULT_PROFILE_COLUMNS:
        score += 0.5
    if work_dist and col in {"workclass", "occupation", "income", "hoursperweek", "education"}:
        score += 0.5
    return {"adult_profile_bundle": profile, "evidence_flags": flags, "same_profile_distribution": same_dist, "work_profile_distribution": work_dist, "adult_consistency_score": round(float(score), 6)}


def build_candidate_stats(candidates):
    by_column, by_row = Counter(), Counter()
    for c in candidates:
        by_column[c["column"]] += 1
        by_row[c["row_id"]] += 1
    return {"by_column": by_column, "by_row": by_row, "total_candidates": len(candidates)}


def summarize_bayesian_info(candidate, df, col_profiles, candidate_stats, similarity_info, adult_consistency_info):
    col, value = candidate["column"], candidate["value"]
    semantic_type = col_profiles[col]["semantic_type"]
    prior = min(max(candidate_stats["by_column"].get(col, 0) / len(df), 1e-4), 0.95) if len(df) else 0.01
    rule_signal = min(5.0, len(candidate.get("violated_rules", [])) + candidate.get("conflict_score", 0.0) / 100.0)
    counter = col_profiles[col]["value_counter"]
    freq, non_null_count = counter.get(value, 0), col_profiles[col]["non_null_count"]
    rarity = 1.0 - (freq / non_null_count) if non_null_count > 0 else 1.0
    rarity = max(0.0, min(1.0, rarity))
    current_pattern = detect_basic_pattern(value) if value is not None else "NULL"
    dominant_pattern = col_profiles[col]["dominant_pattern"]
    pattern_signal = 1.0 if dominant_pattern is not None and current_pattern != dominant_pattern else 0.0
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
    neighbor_value = similarity_info.get("neighbor_majority_value")
    neighbor_ratio = similarity_info.get("neighbor_majority_ratio")
    neighbor_signal = 0.0 if neighbor_value is None or neighbor_ratio is None else (neighbor_ratio if neighbor_value != value else 0.0)
    adult_signal = adult_consistency_info.get("adult_consistency_score", 0.0)
    posterior = sigmoid(safe_logit(prior) + W_RULE * rule_signal + W_RARITY * rarity + W_PATTERN * pattern_signal + W_NEIGHBOR * neighbor_signal + W_ADULT_CONSISTENCY * adult_signal)
    return {"prior_error_probability": round(prior, 6), "posterior_error_probability": round(posterior, 6), "bayes_evidence": {"rule_signal": round(float(rule_signal), 6), "rarity_signal": round(float(rarity), 6), "pattern_signal": round(float(pattern_signal), 6), "neighbor_signal": round(float(neighbor_signal), 6), "adult_consistency_signal": round(float(adult_signal), 6)}, "note": "posterior is a heuristic Bayesian-style approximation"}



def build_adult_v2_attribution_features(row_ctx: Dict[str, Any], col: str, value: Any, conflict_ctx: Dict[str, Any]) -> Dict[str, Any]:
    row_values = row_ctx.get("row_values", {}) or {}
    age = row_values.get("age")
    maritalstatus = row_values.get("maritalstatus")
    relationship = row_values.get("relationship")
    sex = row_values.get("sex")
    education = row_values.get("education")

    age_lo, age_hi = parse_age_bucket(age)
    edu_order = EDUCATION_ORDER.get(education)

    rule_types = set(conflict_ctx.get("rule_types", []))

    is_relationship_spouse_value = int(col == "relationship" and value in RELATIONSHIP_SPOUSE_VALUES)

    relationship_marital_conflict = int(
        col == "relationship"
        and value in RELATIONSHIP_SPOUSE_VALUES
        and maritalstatus in NON_MARRIED_STATUS_FOR_SPOUSE
    )

    relationship_age_conflict = int(
        col == "relationship"
        and value in RELATIONSHIP_SPOUSE_VALUES
        and age_hi is not None
        and age_hi < 18
    )

    expected_sex = VALID_SEX_BY_RELATIONSHIP.get(value)
    relationship_sex_conflict = int(
        col == "relationship"
        and expected_sex is not None
        and sex in VALID_SEX
        and sex != expected_sex
    )

    sex_value_invalid = int(col == "sex" and value is not None and value not in VALID_SEX)

    education_age_extreme_conflict = int(
        col == "education"
        and age_hi is not None
        and age_hi < 18
        and edu_order is not None
        and edu_order >= EDUCATION_ORDER["Bachelors"]
    )

    return {
        "target_is_primary_column": int(col in PRIMARY_TARGET_COLUMNS),
        "target_is_context_only_column": int(col in CONTEXT_ONLY_COLUMNS),

        "is_relationship_spouse_value": is_relationship_spouse_value,
        "relationship_marital_conflict": relationship_marital_conflict,
        "relationship_age_conflict": relationship_age_conflict,
        "relationship_sex_conflict": relationship_sex_conflict,

        "sex_value_invalid": sex_value_invalid,
        "education_age_extreme_conflict": education_age_extreme_conflict,

        "has_relationship_v2_rule": int(len(rule_types & RELATIONSHIP_V2_RULE_TYPES) > 0),
        "has_education_v2_rule": int(len(rule_types & EDUCATION_V2_RULE_TYPES) > 0),
    }



def build_llm_context_text(row_id, col, value, row_ctx, col_ctx, conflict_ctx, sim_ctx, adult_ctx, bayes_ctx, v2_features=None) -> str:
    lines = ["Candidate cell:", f"- row_id: {row_id}", f"- column: {col}", f"- current value: {value}", "", "Row context:", f"- missing_count: {row_ctx['missing_count']}", f"- non_missing_count: {row_ctx['non_missing_count']}", f"- avg_value_length: {row_ctx['avg_value_length']}", "- row field values:"]
    for k, v in row_ctx["row_values_for_llm"]:
        lines.append(f"  - {k}: {v}")
    lines += ["", "Adult profile bundle:", f"- {row_ctx['adult_profile_bundle']}", "", "Column context:", f"- detected_type: {col_ctx['detected_type']}", f"- semantic_type: {col_ctx['semantic_type']}", f"- missing_rate: {col_ctx['missing_rate']}", f"- unique_rate: {col_ctx['unique_rate']}", f"- value_frequency: {col_ctx['value_frequency']}", f"- value_frequency_rank: {col_ctx['value_frequency_rank']}", f"- current_pattern: {col_ctx['current_value_pattern']}", f"- dominant_pattern: {col_ctx['dominant_pattern']}", f"- charset_features: {col_ctx['charset_features']}", f"- adult_value_features: {col_ctx['adult_value_features']}", "- top values with counts:"]
    for item in col_ctx["top_values_with_counts"][:TOP_K_COLUMN_VALUES]:
        lines.append(f"  - {item['value']}: {item['count']}")
    lines.append("- top patterns with counts:")
    for item in col_ctx["top_patterns_with_counts"][:5]:
        lines.append(f"  - {item['pattern']}: count={item['count']}, ratio={item['ratio']}")
    lines += ["", "Conflict information:", f"- has_rule_violation: {conflict_ctx['has_rule_violation']}", f"- violation_count: {conflict_ctx['violation_count']}", f"- main_rule_type: {conflict_ctx['main_rule_type']}", f"- main_usage_role: {conflict_ctx['main_usage_role']}", f"- strong_rule_count: {conflict_ctx['strong_rule_count']}", f"- candidate_generation_rule_count: {conflict_ctx['candidate_generation_rule_count']}", f"- fd_like_count: {conflict_ctx['fd_like_count']}", f"- context_rule_count: {conflict_ctx['context_rule_count']}", f"- global_rule_count: {conflict_ctx['global_rule_count']}", f"- rare_value_count: {conflict_ctx['rare_value_count']}", f"- pattern_rule_count: {conflict_ctx['pattern_rule_count']}", f"- schema_rule_count: {conflict_ctx['schema_rule_count']}", f"- adult_domain_rule_count: {conflict_ctx['adult_domain_rule_count']}", f"- adult_format_rule_count: {conflict_ctx['adult_format_rule_count']}", f"- adult_consistency_rule_count: {conflict_ctx['adult_consistency_rule_count']}", "- violated rules:"]
    for r in conflict_ctx["violated_rules_detailed"][:20]:
        lines.append(f"  - rule_id={r.get('rule_id')}, rule_type={r.get('rule_type')}, priority={r.get('priority')}, usage_role={r.get('usage_role')}, expected_value={r.get('expected_value')}, reason={r.get('reason')}")
    lines.append("- alternative values from rules/column:")
    for alt in conflict_ctx["all_alternative_values"][:TOP_K_ALT_VALUES]:
        lines.append(f"  - {alt}")
    if v2_features is not None:
        lines += [
            "",
            "Adult v2 attribution features:",
            f"- target_is_primary_column: {v2_features.get('target_is_primary_column')}",
            f"- target_is_context_only_column: {v2_features.get('target_is_context_only_column')}",
            f"- is_relationship_spouse_value: {v2_features.get('is_relationship_spouse_value')}",
            f"- relationship_marital_conflict: {v2_features.get('relationship_marital_conflict')}",
            f"- relationship_age_conflict: {v2_features.get('relationship_age_conflict')}",
            f"- relationship_sex_conflict: {v2_features.get('relationship_sex_conflict')}",
            f"- sex_value_invalid: {v2_features.get('sex_value_invalid')}",
            f"- education_age_extreme_conflict: {v2_features.get('education_age_extreme_conflict')}",
            f"- has_relationship_v2_rule: {v2_features.get('has_relationship_v2_rule')}",
            f"- has_education_v2_rule: {v2_features.get('has_education_v2_rule')}",
        ]
    lines += ["", "Similar rows:", f"- neighbor_majority_value: {sim_ctx['neighbor_majority_value']}", f"- neighbor_majority_ratio: {sim_ctx['neighbor_majority_ratio']}", "- top similar rows:"]
    for sr in sim_ctx["topk_similar_rows"]:
        if "row_values" in sr:
            lines.append(f"  - row_id={sr['row_id']}, similarity={sr['similarity']}, target_column_value={sr['target_column_value']}, row_values={sr['row_values']}")
        else:
            lines.append(f"  - row_id={sr['row_id']}, similarity={sr['similarity']}, target_column_value={sr['target_column_value']}")
    lines += ["", "Adult consistency evidence:", f"- evidence_flags: {adult_ctx['evidence_flags']}", f"- same_profile_distribution: {adult_ctx['same_profile_distribution']}", f"- work_profile_distribution: {adult_ctx['work_profile_distribution']}", f"- adult_consistency_score: {adult_ctx['adult_consistency_score']}", "", "Bayesian-style probabilities:", f"- prior_error_probability: {bayes_ctx['prior_error_probability']}", f"- posterior_error_probability: {bayes_ctx['posterior_error_probability']}", f"- evidence: {bayes_ctx['bayes_evidence']}"]
    return "\n".join(lines)



# ============================================================
# 10. 并行主流程
# ============================================================

# worker 共享全局对象。Linux fork 下这些对象主要是 copy-on-write，避免每条候选重复传大对象。
G_DF = None
G_COL_PROFILES = None
G_CANDIDATE_STATS = None
G_ROW_CONTEXTS = None
G_SAME_PROFILE_MAP = None
G_WORK_PROFILE_MAP = None
G_NEIGHBOR_INDICES = None


def _ensure_seq_ids(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for i, cand in enumerate(candidates):
        c = dict(cand)
        c["seq_id"] = i
        out.append(c)
    return out


def _split_chunks(candidates: List[Dict[str, Any]], chunk_size: int) -> List[Tuple[int, List[Dict[str, Any]]]]:
    chunks = []
    for start in range(0, len(candidates), chunk_size):
        chunks.append((len(chunks), candidates[start:start + chunk_size]))
    return chunks


def _clean_tmp_dir(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def _build_flat_row(cand, row_ctx, col_ctx, conflict_ctx, sim_ctx, adult_ctx, bayes_ctx):
    row_id, col, value = cand["row_id"], cand["column"], cand["value"]
    avf = col_ctx["adult_value_features"]
    return {
        "seq_id": cand.get("seq_id"),
        "row_id": row_id, "column": col, "value": value,
        "semantic_type": col_ctx["semantic_type"], "detected_type": col_ctx["detected_type"],
        "violation_count": cand.get("violation_count"), "conflict_score": cand.get("conflict_score"),
        "value_frequency": col_ctx["value_frequency"], "value_frequency_rank": col_ctx["value_frequency_rank"],
        "age_lower": avf["age_lower"], "age_upper": avf["age_upper"],
        "hours_lower": avf["hours_lower"], "hours_upper": avf["hours_upper"],
        "canonical_income": avf["canonical_income"], "domain_valid": avf["domain_valid"],
        "education_order": avf["education_order"], "adult_value_bucket": avf["adult_value_bucket"],
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
    }


def _build_one_context(cand: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    row_id, col, value = cand["row_id"], cand["column"], cand["value"]

    row_ctx = G_ROW_CONTEXTS[row_id]
    col_ctx = summarize_column_context(col, value, G_COL_PROFILES)
    conflict_ctx = summarize_conflict_info(cand, G_COL_PROFILES)
    sim_ctx = summarize_similarity_info_fast(G_DF, row_id, col, G_NEIGHBOR_INDICES)
    adult_ctx = summarize_adult_consistency_info_fast(
        G_DF, row_id, col, value, G_ROW_CONTEXTS, G_SAME_PROFILE_MAP, G_WORK_PROFILE_MAP
    )
    bayes_ctx = summarize_bayesian_info(cand, G_DF, G_COL_PROFILES, G_CANDIDATE_STATS, sim_ctx, adult_ctx)
    v2_features = build_adult_v2_attribution_features(row_ctx, col, value, conflict_ctx)
    llm_context_text = build_llm_context_text(row_id, col, value, row_ctx, col_ctx, conflict_ctx, sim_ctx, adult_ctx, bayes_ctx, v2_features)

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
        "adult_v2_attribution_features": v2_features,
        "bayesian_context": bayes_ctx,
        "llm_context_text": llm_context_text,
    }
    flat_row = _build_flat_row(cand, row_ctx, col_ctx, conflict_ctx, sim_ctx, adult_ctx, bayes_ctx)
    flat_row.update(v2_features)
    return item, flat_row


def _process_chunk(args: Tuple[int, List[Dict[str, Any]]]) -> Dict[str, Any]:
    chunk_id, chunk = args
    jsonl_part = os.path.join(TMP_PART_DIR, f"part_{chunk_id:05d}.jsonl")
    csv_part = os.path.join(TMP_PART_DIR, f"part_{chunk_id:05d}.csv")

    flat_rows = []
    with open(jsonl_part, "w", encoding="utf-8") as f:
        for cand in chunk:
            item, flat_row = _build_one_context(cand)
            f.write(json.dumps(item, ensure_ascii=False, default=json_default) + "\n")
            flat_rows.append(flat_row)

    pd.DataFrame(flat_rows).to_csv(csv_part, index=False, encoding="utf-8-sig")
    return {"chunk_id": chunk_id, "n": len(chunk), "jsonl_part": jsonl_part, "csv_part": csv_part}


def _merge_parts(part_infos: List[Dict[str, Any]]):
    part_infos = sorted(part_infos, key=lambda x: x["chunk_id"])

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:
        for info in part_infos:
            with open(info["jsonl_part"], "r", encoding="utf-8") as fin:
                shutil.copyfileobj(fin, fout)

    flat_parts = []
    for info in part_infos:
        flat_parts.append(pd.read_csv(info["csv_part"]))
    flat_df = pd.concat(flat_parts, ignore_index=True) if flat_parts else pd.DataFrame()

    if "seq_id" in flat_df.columns:
        if KEEP_OUTPUT_ORDER:
            flat_df = flat_df.sort_values("seq_id")
        flat_df = flat_df.drop(columns=["seq_id"]).reset_index(drop=True)

    flat_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")


def main():
    global G_DF, G_COL_PROFILES, G_CANDIDATE_STATS, G_ROW_CONTEXTS
    global G_SAME_PROFILE_MAP, G_WORK_PROFILE_MAP, G_NEIGHBOR_INDICES

    t0 = time.time()
    print("[1/8] Loading table and candidates ...")
    df = load_table(INPUT_TABLE_CSV)
    candidates = _ensure_seq_ids(load_shrunk_candidates(INPUT_SHRUNK_CSV))
    print(f"  table shape = {df.shape}, candidates = {len(candidates)}")

    print("[2/8] Analyzing columns ...")
    col_profiles = analyze_columns(df)
    candidate_stats = build_candidate_stats(candidates)

    print("[3/8] Precomputing row contexts ...")
    row_contexts = precompute_row_contexts(df, col_profiles)

    print("[4/8] Precomputing Adult group distributions ...")
    same_profile_map, work_profile_map = precompute_profile_distributions(df)
    print(f"  same_profile groups = {len(same_profile_map)}, work_profile groups = {len(work_profile_map)}")

    print("[5/8] Precomputing similar rows with vectorized matrix product ...")
    X = build_similarity_matrix(df, col_profiles)
    neighbor_indices = precompute_neighbor_indices(X, top_k=TOP_K_SIMILAR_ROWS)
    print(f"  neighbor index rows = {len(neighbor_indices)}")

    # 设置 worker 共享对象。Linux 上 ProcessPoolExecutor 默认 fork，能复用这部分内存。
    G_DF = df
    G_COL_PROFILES = col_profiles
    G_CANDIDATE_STATS = candidate_stats
    G_ROW_CONTEXTS = row_contexts
    G_SAME_PROFILE_MAP = same_profile_map
    G_WORK_PROFILE_MAP = work_profile_map
    G_NEIGHBOR_INDICES = neighbor_indices

    print("[6/8] Splitting chunks ...")
    _clean_tmp_dir(TMP_PART_DIR)
    chunks = _split_chunks(candidates, CHUNK_SIZE)
    print(f"  chunks = {len(chunks)}, NUM_WORKERS = {NUM_WORKERS}, CHUNK_SIZE = {CHUNK_SIZE}")

    print("[7/8] Building contexts in parallel ...")
    part_infos = []
    if NUM_WORKERS <= 1:
        for i, chunk_args in enumerate(chunks, start=1):
            info = _process_chunk(chunk_args)
            part_infos.append(info)
            print(f"  finished chunks {i}/{len(chunks)}, contexts={sum(x['n'] for x in part_infos)}, elapsed={time.time() - t0:.1f}s")
    else:
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = [executor.submit(_process_chunk, chunk_args) for chunk_args in chunks]
            for done_i, fut in enumerate(as_completed(futures), start=1):
                info = fut.result()
                part_infos.append(info)
                if done_i % 1 == 0 or done_i == len(chunks):
                    print(
                        f"  finished chunks {done_i}/{len(chunks)}, "
                        f"contexts={sum(x['n'] for x in part_infos)}/{len(candidates)}, "
                        f"elapsed={time.time() - t0:.1f}s"
                    )

    print("[8/8] Merging part files ...")
    _merge_parts(part_infos)

    print(f"[OK] Full LLM contexts saved to: {OUTPUT_JSONL}")
    print(f"[OK] Flat summary saved to: {OUTPUT_CSV}")
    print(f"[INFO] Context rows: {len(candidates)}")
    print(f"[INFO] Total elapsed seconds: {time.time() - t0:.2f}")
    print(f"[INFO] Temp part dir: {TMP_PART_DIR}")
    if not KEEP_TMP_PARTS:
        shutil.rmtree(TMP_PART_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()

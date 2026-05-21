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

INPUT_CSV = "/mnt/mydata/dq/projects/Splittree/adult/adult_dirty.csv"
OUTPUT_JSON = "rule_pool_adult.json"

MISSING_TOKENS = {
    "", "empty", "null", "none", "nan", "na", "n/a", "missing", "unk",
    "unknown", "?", "not available", "contact airline"
}

NUMERIC_RATIO_THRESHOLD = 0.90
ENUM_UNIQUE_THRESHOLD = 80
ENUM_RATIO_THRESHOLD = 0.10

ID_LIKE_UNIQUE_RATIO_THRESHOLD = 0.98
CODE_LIKE_AVG_LEN_THRESHOLD = 20
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

HIGH_CARDINALITY_RATIO_FOR_LHS = 0.95
ALLOW_INDEX_COLUMN = False
SKIP_SCHEMA_FOR_ID_LIKE = True

# ------------------------------------------------------------
# adult 数据集字段
# ------------------------------------------------------------
ADULT_COLUMNS = [
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
]

TARGET_CANDIDATE_COLUMNS = set(ADULT_COLUMNS)
HIGH_RISK_MISSING_COLUMNS = set(ADULT_COLUMNS)

ENABLE_RARE_VALUE_RULE = True
RARE_VALUE_MAX_FREQ = 2
SKIP_RARE_VALUE_COLUMNS = {
    "country",       # 国家天然长尾
    "occupation",    # 职业分布长尾
    "workclass",     # 工作类型也可能出现少数合法值
}

ENABLE_RARE_PATTERN_RULE = True
RARE_PATTERN_MAX_RATIO = 0.02
SKIP_RARE_PATTERN_COLUMNS = {
    "age",
    "hoursperweek",
    "country",
}

# 不对这些列使用 global dominant，避免 Private / United-States / LessThan50K 这类多数类制造大量 FP
SKIP_GLOBAL_DOMINANT_COLUMNS = {
    "workclass",
    "education",
    "occupation",
    "race",
    "country",
    "income",
}

# 不对这些列加 generic pattern 规则，改用专门 schema/domain/range 规则
SKIP_GENERIC_PATTERN_COLUMNS = set(ADULT_COLUMNS)

FD_RHS_CANDIDATE_COLUMNS = set(TARGET_CANDIDATE_COLUMNS)
EXTRA_SKIP_FD_LHS_COLUMNS = set()

# ------------------------------------------------------------
# Adult 专属 domain
# ------------------------------------------------------------
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
# 2. 基础函数
# ============================================================

def normalize_value(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.lower() in MISSING_TOKENS:
        return None
    return s


def normalize_series(s):
    return s.map(normalize_value)


def maybe_drop_unnamed_columns(df):
    keep_cols = [c for c in df.columns if not str(c).lower().startswith("unnamed:")]
    return df[keep_cols].copy()


def detect_basic_pattern(value):
    if value is None:
        return "NULL"
    parts, prev_type, cnt = [], None, 0
    for ch in str(value):
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


def try_parse_numeric(series):
    cleaned = series.dropna().astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False)
    numeric = pd.to_numeric(cleaned, errors="coerce")
    success_ratio = numeric.notna().mean() if len(cleaned) > 0 else 0.0
    return numeric, success_ratio


def parse_int_like(x: Optional[str]) -> Optional[int]:
    if x is None:
        return None
    s = str(x).strip().replace(",", "")
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


def canonical_income(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if s in {"LessThan50K", "<=50K", "<50K"}:
        return "LessThan50K"
    if s in {"MoreThan50K", ">50K", ">=50K"}:
        return "MoreThan50K"
    return None


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


def is_index_like_column(series):
    s = series.dropna()
    if len(s) == 0:
        return False
    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.notna().mean() < 0.95:
        return False
    unique_ratio = s.nunique() / len(s)
    return unique_ratio > 0.98


def is_id_like_column(non_null, unique_ratio, numeric_ratio, avg_len):
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


# ============================================================
# 3. 规则统计函数
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
    used_support = 0
    for _, grp in grouped:
        if len(grp) < SOFT_FD_MIN_GROUP_SIZE:
            continue
        cnt = Counter(grp.tolist())
        majority_sum += cnt.most_common(1)[0][1]
        used_support += len(grp)
    confidence = majority_sum / used_support if used_support > 0 else 0.0
    return round(confidence, 6), int(used_support)


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


# ============================================================
# 4. 列画像
# ============================================================

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
        top_patterns = [
            {"pattern": p, "count": c, "ratio": round(c / total_patterns, 6)}
            for p, c in pattern_counter.most_common(10)
        ]

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

        if col == "age":
            profile["detected_type"] = "categorical"
            profile["semantic_type"] = "age_bucket"
        elif col == "hoursperweek":
            profile["detected_type"] = "numeric_or_range"
            profile["semantic_type"] = "hours_range"
        elif col in {
            "workclass", "education", "maritalstatus", "occupation", "relationship",
            "race", "sex", "country", "income"
        }:
            profile["detected_type"] = "categorical"
            profile["semantic_type"] = f"{col}_category"
        elif numeric_ratio >= NUMERIC_RATIO_THRESHOLD:
            profile["detected_type"] = "numeric"
            profile["semantic_type"] = "numeric"
        else:
            if unique_count <= ENUM_UNIQUE_THRESHOLD or unique_ratio <= ENUM_RATIO_THRESHOLD:
                profile["detected_type"] = "categorical"
            else:
                profile["detected_type"] = "text"

            if profile["index_like"]:
                profile["semantic_type"] = "id_like"
            elif is_id_like_column(non_null, unique_ratio, numeric_ratio, avg_len):
                profile["semantic_type"] = "id_like"
            elif is_code_like_column(non_null, unique_ratio, numeric_ratio, avg_len, top_patterns):
                profile["semantic_type"] = "code_like"
            elif profile["detected_type"] == "numeric":
                profile["semantic_type"] = "numeric"
            elif profile["detected_type"] == "categorical":
                profile["semantic_type"] = "categorical"
            else:
                profile["semantic_type"] = "text"

        profiles[col] = profile

    return profiles


# ============================================================
# 5. schema/domain rules
# ============================================================

def build_schema_rules(df, profiles):
    rules = []
    rule_id = 1

    domain_specs = {
        "workclass": VALID_WORKCLASS,
        "education": VALID_EDUCATION,
        "maritalstatus": VALID_MARITALSTATUS,
        "occupation": VALID_OCCUPATION,
        "relationship": VALID_RELATIONSHIP,
        "race": VALID_RACE,
        "sex": VALID_SEX,
        "country": VALID_COUNTRY,
        "income": VALID_INCOME,
    }

    for col, info in profiles.items():
        if col not in TARGET_CANDIDATE_COLUMNS:
            continue

        series = normalize_series(df[col])

        not_null_threshold = 0.70
        if col in {"age", "sex", "income", "hoursperweek"}:
            not_null_threshold = 0.90
        elif col in {"workclass", "education", "maritalstatus", "occupation", "relationship", "race", "country"}:
            not_null_threshold = 0.80

        if info["missing_rate"] <= not_null_threshold:
            confidence, support = calc_not_null_confidence_and_support(series)
            rules.append({
                "rule_id": f"SCHEMA_{rule_id}",
                "rule_type": "not_null",
                "column": col,
                "description": f"{col} should not be null",
                "confidence": confidence,
                "support": support,
                "usage_role": "candidate_generation_rule",
            })
            rule_id += 1

        if len(info["top_patterns"]) > 0 and col not in SKIP_GENERIC_PATTERN_COLUMNS:
            main_pattern = info["top_patterns"][0]
            if main_pattern["ratio"] >= DOMINANT_PATTERN_THRESHOLD:
                confidence, support = calc_pattern_confidence_and_support(series, main_pattern["pattern"])
                rules.append({
                    "rule_id": f"SCHEMA_{rule_id}",
                    "rule_type": "pattern",
                    "column": col,
                    "description": f"{col} should mostly follow the dominant pattern",
                    "pattern": main_pattern["pattern"],
                    "confidence": confidence,
                    "support": support,
                    "usage_role": "strong_rule",
                })
                rule_id += 1

        # adult 专属规则
        if col == "age":
            rules.append({
                "rule_id": f"SCHEMA_{rule_id}",
                "rule_type": "age_bucket_format",
                "column": col,
                "description": "age should be a bucket such as <18, 18-21, 22-25, ..., >70",
                "valid_values": sorted(list(VALID_AGE_BUCKETS)),
                "regex": r"^(<\s*\d+|>\s*\d+|\d+\s*-\s*\d+|\d+)$",
                "confidence": 0.92,
                "support": info["non_null_count"],
                "usage_role": "strong_rule",
            })
            rule_id += 1

        if col == "hoursperweek":
            rules.append({
                "rule_id": f"SCHEMA_{rule_id}",
                "rule_type": "hours_per_week_format_range",
                "column": col,
                "description": "hoursperweek should be numeric or a numeric range and broadly within 0..120",
                "lower_bound": 0,
                "upper_bound": 120,
                "regex": r"^(\d+|\d+\s*-\s*\d+)$",
                "confidence": 0.90,
                "support": info["non_null_count"],
                "usage_role": "strong_rule",
            })
            rule_id += 1

        if col in domain_specs:
            rules.append({
                "rule_id": f"SCHEMA_{rule_id}",
                "rule_type": f"{col}_domain",
                "column": col,
                "description": f"{col} should be in a known adult domain",
                "valid_values": sorted(list(domain_specs[col])),
                "confidence": 0.93 if col != "country" else 0.86,
                "support": info["non_null_count"],
                "usage_role": "strong_rule" if col != "country" else "candidate_generation_rule",
            })
            rule_id += 1

        if col == "income":
            rules.append({
                "rule_id": f"SCHEMA_{rule_id}",
                "rule_type": "income_canonical_format",
                "column": col,
                "description": "income should canonicalize to LessThan50K or MoreThan50K",
                "confidence": 0.92,
                "support": info["non_null_count"],
                "usage_role": "strong_rule",
            })
            rule_id += 1

    return rules


# ============================================================
# 6. FD / soft FD
# ============================================================

def select_fd_lhs_candidates(profiles):
    candidates = []
    for col, info in profiles.items():
        if col in EXTRA_SKIP_FD_LHS_COLUMNS:
            continue
        if (not ALLOW_INDEX_COLUMN) and info.get("index_like", False):
            continue
        if info["unique_ratio"] >= HIGH_CARDINALITY_RATIO_FOR_LHS:
            continue
        candidates.append(col)
    return candidates


def build_fd_rules(df, profiles):
    rules = []
    rule_id = 1

    cols = list(df.columns)
    norm_df = pd.DataFrame({c: normalize_series(df[c]) for c in cols})

    lhs_candidates = select_fd_lhs_candidates(profiles)
    rhs_candidates = [c for c in FD_RHS_CANDIDATE_COLUMNS if c in cols]

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
                rules.append({
                    "rule_id": f"FD_{rule_id}",
                    "rule_type": "functional_dependency",
                    "lhs": list(lhs),
                    "rhs": rhs,
                    "description": f"{', '.join(lhs)} -> {rhs}",
                    "confidence": confidence,
                    "support": support,
                    "usage_role": "strong_rule",
                })
                rule_id += 1

    dedup = {(tuple(sorted(r["lhs"])), r["rhs"]): r for r in rules}
    return list(dedup.values())


def build_soft_fd_rules(df, profiles):
    rules = []
    rule_id = 1

    cols = list(df.columns)
    norm_df = pd.DataFrame({c: normalize_series(df[c]) for c in cols})

    lhs_candidates = select_fd_lhs_candidates(profiles)
    rhs_candidates = [c for c in FD_RHS_CANDIDATE_COLUMNS if c in cols]

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
                rules.append({
                    "rule_id": f"CAND_SOFTFD_{rule_id}",
                    "rule_type": "soft_functional_dependency",
                    "lhs": list(lhs),
                    "rhs": rhs,
                    "description": f"soft FD: {', '.join(lhs)} -> {rhs}",
                    "confidence": confidence,
                    "support": support,
                    "usage_role": "candidate_generation_rule",
                })
                rule_id += 1

    return rules


# ============================================================
# 7. adult context / consistency rules
# ============================================================

def build_pair_context_rules(df):
    rules = []
    rule_id = 1
    norm_df = pd.DataFrame({c: normalize_series(df[c]) for c in df.columns})

    pair_specs = [
        (["age"], "education"),
        (["age"], "relationship"),
        (["maritalstatus"], "relationship"),
        (["relationship"], "maritalstatus"),
        (["workclass"], "occupation"),
        (["education"], "occupation"),
        (["occupation"], "workclass"),
        (["education", "occupation"], "income"),
        (["maritalstatus", "relationship"], "sex"),
        (["country"], "race"),
    ]

    for lhs, rhs in pair_specs:
        if any(c not in norm_df.columns for c in lhs + [rhs]):
            continue

        sub = norm_df[lhs + [rhs]].dropna()
        if len(sub) < PAIR_CONTEXT_MIN_GROUP_SIZE:
            continue

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
                majority_map["||".join(map(str, key))] = {
                    "dominant_value": maj_val,
                    "group_size": len(grp),
                    "ratio": round(ratio, 6),
                }
                total_support += len(grp)

        if majority_map:
            rtype = "dominant_value_by_context_pair" if len(lhs) >= 2 else "dominant_value_by_context"
            rules.append({
                "rule_id": f"CAND_PAIRCTX_{rule_id}",
                "rule_type": rtype,
                "lhs": lhs,
                "rhs": rhs,
                "description": f"context dominant: {', '.join(lhs)} -> {rhs}",
                "mapping_size": len(majority_map),
                "majority_map": majority_map,
                "confidence": 0.84,
                "support": total_support,
                "usage_role": "candidate_generation_rule",
            })
            rule_id += 1

    return rules


def build_adult_consistency_rules(df, profiles):
    rules = []
    rule_id = 1

    if {"age", "education"}.issubset(df.columns):
        rules.append({
            "rule_id": f"CAND_ADULTCONS_{rule_id}",
            "rule_type": "age_education_consistency",
            "columns": ["age", "education"],
            "description": "very young age buckets should be consistent with plausible education levels",
            "confidence": 0.86,
            "support": len(df),
            "usage_role": "candidate_generation_rule",
            "education_order": EDUCATION_ORDER,
        })
        rule_id += 1

    if {"age", "relationship"}.issubset(df.columns):
        rules.append({
            "rule_id": f"CAND_ADULTCONS_{rule_id}",
            "rule_type": "age_relationship_consistency",
            "columns": ["age", "relationship"],
            "description": "age bucket should be consistent with relationship role such as Own-child",
            "confidence": 0.82,
            "support": len(df),
            "usage_role": "candidate_generation_rule",
        })
        rule_id += 1

    if {"maritalstatus", "relationship"}.issubset(df.columns):
        rules.append({
            "rule_id": f"CAND_ADULTCONS_{rule_id}",
            "rule_type": "marital_relationship_consistency",
            "columns": ["maritalstatus", "relationship"],
            "description": "maritalstatus should be logically consistent with relationship",
            "confidence": 0.90,
            "support": len(df),
            "usage_role": "strong_rule",
        })
        rule_id += 1

    if {"sex", "relationship"}.issubset(df.columns):
        rules.append({
            "rule_id": f"CAND_ADULTCONS_{rule_id}",
            "rule_type": "sex_relationship_consistency",
            "columns": ["sex", "relationship"],
            "description": "relationship Wife/Husband should be consistent with sex",
            "confidence": 0.88,
            "support": len(df),
            "usage_role": "candidate_generation_rule",
        })
        rule_id += 1

    if {"workclass", "occupation"}.issubset(df.columns):
        rules.append({
            "rule_id": f"CAND_ADULTCONS_{rule_id}",
            "rule_type": "workclass_occupation_consistency",
            "columns": ["workclass", "occupation"],
            "description": "workclass and occupation should be mutually plausible",
            "confidence": 0.82,
            "support": len(df),
            "usage_role": "candidate_generation_rule",
        })
        rule_id += 1

    if {"education", "occupation"}.issubset(df.columns):
        rules.append({
            "rule_id": f"CAND_ADULTCONS_{rule_id}",
            "rule_type": "education_occupation_consistency",
            "columns": ["education", "occupation"],
            "description": "education and occupation should be mutually plausible by dominant evidence",
            "confidence": 0.80,
            "support": len(df),
            "usage_role": "candidate_generation_rule",
        })
        rule_id += 1

    if {"hoursperweek", "workclass"}.issubset(df.columns):
        rules.append({
            "rule_id": f"CAND_ADULTCONS_{rule_id}",
            "rule_type": "hours_workclass_consistency",
            "columns": ["hoursperweek", "workclass"],
            "description": "hoursperweek should be broadly plausible for the workclass",
            "confidence": 0.78,
            "support": len(df),
            "usage_role": "candidate_generation_rule",
        })
        rule_id += 1

    return rules


def build_candidate_generation_rules(df, profiles):
    rules = []
    rule_id = 1
    norm_df = pd.DataFrame({c: normalize_series(df[c]) for c in df.columns})

    # global dominant
    for col, prof in profiles.items():
        if col not in TARGET_CANDIDATE_COLUMNS or col in SKIP_GLOBAL_DOMINANT_COLUMNS:
            continue

        top_values = prof.get("top_values", [])
        if not top_values:
            continue

        top1 = top_values[0]
        if top1["ratio"] >= DOMINANT_TYPO_MIN_RATIO:
            rules.append({
                "rule_id": f"CAND_GDOM_{rule_id}",
                "rule_type": "global_dominant_value",
                "column": col,
                "description": f"global dominant value for {col}",
                "dominant_value": top1["value"],
                "dominant_count": top1["count"],
                "dominant_ratio": top1["ratio"],
                "confidence": round(top1["ratio"], 6),
                "support": prof["non_null_count"],
                "usage_role": "candidate_generation_rule",
            })
            rule_id += 1

    # single-column context dominant
    lhs_candidates = select_fd_lhs_candidates(profiles)
    context_rule_id = 1
    for rhs in TARGET_CANDIDATE_COLUMNS:
        if rhs not in df.columns:
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
                    majority_map[lhs_val] = {
                        "dominant_value": maj_val,
                        "group_size": len(grp),
                        "ratio": round(ratio, 6),
                    }
                    total_support += len(grp)

            if majority_map:
                rules.append({
                    "rule_id": f"CAND_CTXDOM_{context_rule_id}",
                    "rule_type": "dominant_value_by_context",
                    "lhs": [lhs],
                    "rhs": rhs,
                    "description": f"context dominant: {lhs} -> dominant {rhs}",
                    "mapping_size": len(majority_map),
                    "majority_map": majority_map,
                    "confidence": 0.85,
                    "support": total_support,
                    "usage_role": "candidate_generation_rule",
                })
                context_rule_id += 1

    # rare value
    if ENABLE_RARE_VALUE_RULE:
        for col, prof in profiles.items():
            if col not in TARGET_CANDIDATE_COLUMNS or col in SKIP_RARE_VALUE_COLUMNS:
                continue

            counter = prof["value_counter"]
            rare_values = [v for v, c in counter.items() if c <= RARE_VALUE_MAX_FREQ]
            if rare_values:
                rules.append({
                    "rule_id": f"CAND_RAREVAL_{rule_id}",
                    "rule_type": "rare_value",
                    "column": col,
                    "description": f"rare values in {col} may be suspicious",
                    "rare_values": rare_values[:5000],
                    "confidence": 0.6,
                    "support": sum(counter[v] for v in rare_values),
                    "usage_role": "candidate_generation_rule",
                })
                rule_id += 1

    # rare pattern
    if ENABLE_RARE_PATTERN_RULE:
        for col, prof in profiles.items():
            if col not in TARGET_CANDIDATE_COLUMNS or col in SKIP_RARE_PATTERN_COLUMNS:
                continue

            top_patterns = prof.get("top_patterns", [])
            rare_patterns = [x["pattern"] for x in top_patterns if x["ratio"] <= RARE_PATTERN_MAX_RATIO]
            if rare_patterns:
                rules.append({
                    "rule_id": f"CAND_RAREPAT_{rule_id}",
                    "rule_type": "rare_pattern",
                    "column": col,
                    "description": f"rare patterns in {col} may be suspicious",
                    "rare_patterns": rare_patterns,
                    "confidence": 0.6,
                    "support": prof["non_null_count"],
                    "usage_role": "candidate_generation_rule",
                })
                rule_id += 1

    # high risk missing
    for col in HIGH_RISK_MISSING_COLUMNS:
        if col not in profiles:
            continue
        prof = profiles[col]
        rules.append({
            "rule_id": f"CAND_HRMISS_{rule_id}",
            "rule_type": "high_risk_missing",
            "column": col,
            "description": f"{col} missing values are suspicious in candidate generation",
            "confidence": 0.8,
            "support": prof["non_null_count"],
            "usage_role": "candidate_generation_rule",
        })
        rule_id += 1

    # paired missing consistency
    for a, b in [
        ("workclass", "occupation"),
        ("maritalstatus", "relationship"),
        ("sex", "relationship"),
        ("age", "education"),
        ("country", "race"),
        ("education", "occupation"),
    ]:
        if a in df.columns and b in df.columns:
            rules.append({
                "rule_id": f"CAND_PAIR_{rule_id}",
                "rule_type": "paired_missing_consistency",
                "columns": [a, b],
                "description": f"{a} and {b} should usually co-occur",
                "confidence": 0.8,
                "support": len(df),
                "usage_role": "candidate_generation_rule",
            })
            rule_id += 1

    rules.extend(build_pair_context_rules(df))
    rules.extend(build_adult_consistency_rules(df, profiles))
    return rules


# ============================================================
# 8. priority
# ============================================================

def assign_rule_priority(rule):
    rtype = rule.get("rule_type", "")

    if rtype in {
        "age_bucket_format",
        "hours_per_week_format_range",
        "workclass_domain",
        "education_domain",
        "maritalstatus_domain",
        "occupation_domain",
        "relationship_domain",
        "race_domain",
        "sex_domain",
        "income_domain",
        "income_canonical_format",
        "marital_relationship_consistency",
    }:
        return "high"

    if rtype in {
        "country_domain",
        "age_education_consistency",
        "age_relationship_consistency",
        "sex_relationship_consistency",
        "workclass_occupation_consistency",
        "education_occupation_consistency",
        "hours_workclass_consistency",
    }:
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
# 9. 主流程
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

    rule_groups = {
        "schema_rules": schema_rules,
        "fd_rules": fd_rules,
        "soft_fd_rules": soft_fd_rules,
        "candidate_rules": candidate_rules,
    }
    rule_groups = enrich_rules_with_priority(rule_groups)

    rule_pool = {
        "metadata": {
            "source_file": input_csv,
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "design_goal": "high_recall_candidate_generation_for_adult",
            "target_candidate_columns": sorted(list(TARGET_CANDIDATE_COLUMNS)),
            "notes": [
                "candidate inclusion is determined only by violating rules in this pool",
                "adult-specific rules include categorical domains, age/hour range checks, income canonicalization, and demographic/work-profile consistency rules",
                "global dominant rules are disabled for high-frequency majority columns such as workclass, country, and income to avoid excessive false positives",
                "rare_value is disabled for country/workclass/occupation because they naturally contain legitimate minority values",
                "FD and soft FD rules are mined but should be used as evidence rather than final labels",
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

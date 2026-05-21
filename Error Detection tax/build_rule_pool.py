import json
import math
import re
from collections import Counter
from itertools import combinations
from typing import Dict, List, Any, Optional

import pandas as pd

INPUT_CSV = "/mnt/mydata/dq/projects/Splittree/tax/tax_dirty.csv"
OUTPUT_JSON = "rule_pool_tax.json"

MISSING_TOKENS = {
    "", "empty", "null", "none", "nan", "na", "n/a", "missing", "unk",
    "unknown", "not available", "contact airline"
}

NUMERIC_RATIO_THRESHOLD = 0.90
ENUM_UNIQUE_THRESHOLD = 30
ENUM_RATIO_THRESHOLD = 0.05
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

TARGET_CANDIDATE_COLUMNS = {
    "f_name", "l_name", "gender", "area_code", "phone", "city", "state", "zip",
    "marital_status", "has_child", "salary", "rate", "single_exemp", "married_exemp", "child_exemp"
}
HIGH_RISK_MISSING_COLUMNS = set(TARGET_CANDIDATE_COLUMNS)

FD_RHS_CANDIDATE_COLUMNS = {
    "gender", "area_code", "phone", "city", "state", "zip", "marital_status", "has_child",
    "salary", "rate", "single_exemp", "married_exemp", "child_exemp"
}
EXTRA_SKIP_FD_LHS_COLUMNS = {"f_name", "l_name", "phone"}

ENABLE_RARE_VALUE_RULE = True
RARE_VALUE_MAX_FREQ = 2
SKIP_RARE_VALUE_COLUMNS = {"f_name", "l_name", "city", "phone", "zip", "area_code"}
ENABLE_RARE_PATTERN_RULE = True
RARE_PATTERN_MAX_RATIO = 0.02
SKIP_RARE_PATTERN_COLUMNS = {"f_name", "l_name", "city", "phone"}
SKIP_GLOBAL_DOMINANT_COLUMNS = {
    "f_name", "l_name", "city", "phone", "zip", "area_code", "salary",
    "single_exemp", "married_exemp", "child_exemp"
}
SKIP_GENERIC_PATTERN_COLUMNS = {
    "gender", "area_code", "phone", "state", "zip", "marital_status", "has_child",
    "salary", "rate", "single_exemp", "married_exemp", "child_exemp"
}

VALID_GENDERS = {"m", "f"}
VALID_MARITAL_STATUS = {"s", "m"}
VALID_HAS_CHILD = {"y", "n"}
STATE_CODES = {
    "al", "ak", "az", "ar", "ca", "co", "ct", "dc", "de", "fl", "ga", "hi", "ia", "id", "il",
    "in", "ks", "ky", "la", "ma", "md", "me", "mi", "mn", "mo", "ms", "mt", "nc", "nd", "ne",
    "nh", "nj", "nm", "nv", "ny", "oh", "ok", "or", "pa", "ri", "sc", "sd", "tn", "tx", "ut",
    "va", "vt", "wa", "wi", "wv", "wy"
}
SALARY_MIN, SALARY_MAX = 0, 1000000
RATE_MIN, RATE_MAX = 0.0, 20.0
EXEMP_MIN, EXEMP_MAX = 0, 100000


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
    for ch in str(value):
        if ch.isdigit():
            cur = "D"
        elif ch.isalpha():
            cur = "L"
        elif ch in {" ", "-", "_", "/", ".", ",", "(", ")", "%", ":", ";", "&", "'", "+", "#"}:
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
    cleaned = series.dropna().astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False).str.replace("$", "", regex=False)
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


def is_index_like_column(series):
    s = series.dropna()
    if len(s) == 0:
        return False
    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.notna().mean() < 0.95:
        return False
    return s.nunique() / len(s) > 0.98


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


def parse_int_like(x: Optional[str]) -> Optional[int]:
    if x is None:
        return None
    s = str(x).strip().lower().replace(",", "").replace("$", "")
    if re.fullmatch(r"[+-]?\d+(\.0+)?", s):
        try:
            return int(float(s))
        except Exception:
            return None
    return None


def parse_float_like(x: Optional[str]) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip().lower().replace(",", "").replace("$", "").replace("%", "")
    try:
        return float(s)
    except Exception:
        return None


def analyze_columns(df):
    profiles = {}
    n = len(df)
    for col in df.columns:
        s = normalize_series(df[col])
        non_null = s.dropna()
        missing_rate = 1.0 - (len(non_null) / n if n > 0 else 0.0)
        unique_count = non_null.nunique()
        unique_ratio = unique_count / len(non_null) if len(non_null) > 0 else 0.0
        _, numeric_ratio = try_parse_numeric(non_null)
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
        if numeric_ratio >= NUMERIC_RATIO_THRESHOLD:
            profile["detected_type"] = "numeric"
        elif unique_count <= ENUM_UNIQUE_THRESHOLD or unique_ratio <= ENUM_RATIO_THRESHOLD:
            profile["detected_type"] = "categorical"
        else:
            profile["detected_type"] = "text"
        if profile["index_like"]:
            profile["semantic_type"] = "id_like"
        elif col in {"gender", "marital_status", "has_child", "state"}:
            profile["semantic_type"] = "categorical"
        elif col in {"area_code", "zip"}:
            profile["semantic_type"] = "code_like"
        elif col == "phone":
            profile["semantic_type"] = "phone_like"
        elif col in {"salary", "rate", "single_exemp", "married_exemp", "child_exemp"}:
            profile["semantic_type"] = "numeric"
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


def build_schema_rules(df, profiles):
    rules = []
    rule_id = 1
    for col, info in profiles.items():
        if col not in TARGET_CANDIDATE_COLUMNS:
            continue
        series = normalize_series(df[col])
        not_null_threshold = 0.85 if col in {"f_name", "l_name", "gender", "area_code", "phone", "city", "state", "zip", "marital_status", "has_child", "salary", "rate"} else 0.55
        if info["missing_rate"] <= not_null_threshold:
            confidence, support = calc_not_null_confidence_and_support(series)
            rules.append({"rule_id": f"SCHEMA_{rule_id}", "rule_type": "not_null", "column": col, "description": f"{col} should not be null", "confidence": confidence, "support": support, "usage_role": "candidate_generation_rule"})
            rule_id += 1
        if len(info["top_patterns"]) > 0 and col not in SKIP_GENERIC_PATTERN_COLUMNS:
            main_pattern = info["top_patterns"][0]
            if main_pattern["ratio"] >= DOMINANT_PATTERN_THRESHOLD:
                confidence, support = calc_pattern_confidence_and_support(series, main_pattern["pattern"])
                rules.append({"rule_id": f"SCHEMA_{rule_id}", "rule_type": "pattern", "column": col, "description": f"{col} should mostly follow the dominant pattern", "pattern": main_pattern["pattern"], "confidence": confidence, "support": support, "usage_role": "strong_rule"})
                rule_id += 1
        if col in {"f_name", "l_name"}:
            rules.append({"rule_id": f"SCHEMA_{rule_id}", "rule_type": "person_name_format", "column": col, "description": f"{col} should look like a person name", "regex": r"^[a-z][a-z'\-\. ]{0,49}$", "confidence": 0.82, "support": info["non_null_count"], "usage_role": "candidate_generation_rule"}); rule_id += 1
        if col == "gender":
            rules.append({"rule_id": f"SCHEMA_{rule_id}", "rule_type": "gender_domain", "column": col, "description": "gender should be one of M/F", "valid_values": sorted(list(VALID_GENDERS)), "confidence": 0.95, "support": info["non_null_count"], "usage_role": "strong_rule"}); rule_id += 1
        if col == "area_code":
            rules.append({"rule_id": f"SCHEMA_{rule_id}", "rule_type": "area_code_format", "column": col, "description": "area_code should be a 3-digit area code", "regex": r"^\d{3}$", "confidence": 0.92, "support": info["non_null_count"], "usage_role": "strong_rule"}); rule_id += 1
        if col == "phone":
            rules.append({"rule_id": f"SCHEMA_{rule_id}", "rule_type": "phone_format", "column": col, "description": "phone should be a 7-digit local phone number like 744-9007", "regex": r"^\d{3}-\d{4}$", "confidence": 0.92, "support": info["non_null_count"], "usage_role": "strong_rule"}); rule_id += 1
            rules.append({"rule_id": f"SCHEMA_{rule_id}", "rule_type": "phone_canonical_format", "column": col, "description": "phone should canonicalize to xxx-xxxx", "confidence": 0.90, "support": info["non_null_count"], "usage_role": "candidate_generation_rule"}); rule_id += 1
        if col == "city":
            rules.append({"rule_id": f"SCHEMA_{rule_id}", "rule_type": "city_format", "column": col, "description": "city should contain alphabetic city tokens", "regex": r"^[a-z][a-z\.\-'\s]{1,60}$", "confidence": 0.80, "support": info["non_null_count"], "usage_role": "candidate_generation_rule"}); rule_id += 1
        if col == "state":
            rules.append({"rule_id": f"SCHEMA_{rule_id}", "rule_type": "state_code_domain", "column": col, "description": "state should be a valid US state abbreviation", "valid_values": sorted(list(STATE_CODES)), "confidence": 0.95, "support": info["non_null_count"], "usage_role": "strong_rule"}); rule_id += 1
        if col == "zip":
            rules.append({"rule_id": f"SCHEMA_{rule_id}", "rule_type": "zip_format", "column": col, "description": "zip should be a 5-digit US ZIP code", "regex": r"^\d{5}$", "confidence": 0.94, "support": info["non_null_count"], "usage_role": "strong_rule"}); rule_id += 1
            rules.append({"rule_id": f"SCHEMA_{rule_id}", "rule_type": "zip_canonical_format", "column": col, "description": "zip should canonicalize to 5 digits, preserving leading zeros", "confidence": 0.92, "support": info["non_null_count"], "usage_role": "candidate_generation_rule"}); rule_id += 1
        if col == "marital_status":
            rules.append({"rule_id": f"SCHEMA_{rule_id}", "rule_type": "marital_status_domain", "column": col, "description": "marital_status should be one of S/M", "valid_values": sorted(list(VALID_MARITAL_STATUS)), "confidence": 0.95, "support": info["non_null_count"], "usage_role": "strong_rule"}); rule_id += 1
        if col == "has_child":
            rules.append({"rule_id": f"SCHEMA_{rule_id}", "rule_type": "has_child_domain", "column": col, "description": "has_child should be one of Y/N", "valid_values": sorted(list(VALID_HAS_CHILD)), "confidence": 0.95, "support": info["non_null_count"], "usage_role": "strong_rule"}); rule_id += 1
        if col == "salary":
            rules.append({"rule_id": f"SCHEMA_{rule_id}", "rule_type": "salary_numeric_range", "column": col, "description": "salary should be numeric and within a broad non-negative range", "lower_bound": SALARY_MIN, "upper_bound": SALARY_MAX, "confidence": 0.90, "support": info["non_null_count"], "usage_role": "strong_rule"}); rule_id += 1
        if col == "rate":
            rules.append({"rule_id": f"SCHEMA_{rule_id}", "rule_type": "rate_numeric_range", "column": col, "description": "rate should be numeric and within a broad percentage-like range", "lower_bound": RATE_MIN, "upper_bound": RATE_MAX, "confidence": 0.88, "support": info["non_null_count"], "usage_role": "strong_rule"}); rule_id += 1
        if col in {"single_exemp", "married_exemp", "child_exemp"}:
            rules.append({"rule_id": f"SCHEMA_{rule_id}", "rule_type": "exemption_numeric_range", "column": col, "description": f"{col} should be numeric and non-negative", "lower_bound": EXEMP_MIN, "upper_bound": EXEMP_MAX, "confidence": 0.88, "support": info["non_null_count"], "usage_role": "strong_rule"}); rule_id += 1
    return rules


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
    rules, rule_id = [], 1
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
                rules.append({"rule_id": f"FD_{rule_id}", "rule_type": "functional_dependency", "lhs": list(lhs), "rhs": rhs, "description": f"{', '.join(lhs)} -> {rhs}", "confidence": confidence, "support": support, "usage_role": "strong_rule"})
                rule_id += 1
    dedup = {(tuple(sorted(r["lhs"])), r["rhs"]): r for r in rules}
    return list(dedup.values())


def build_soft_fd_rules(df, profiles):
    rules, rule_id = [], 1
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
                rules.append({"rule_id": f"CAND_SOFTFD_{rule_id}", "rule_type": "soft_functional_dependency", "lhs": list(lhs), "rhs": rhs, "description": f"soft FD: {', '.join(lhs)} -> {rhs}", "confidence": confidence, "support": support, "usage_role": "candidate_generation_rule"})
                rule_id += 1
    return rules


def build_pair_context_rules(df):
    rules, rule_id = [], 1
    norm_df = pd.DataFrame({c: normalize_series(df[c]) for c in df.columns})
    pair_specs = [
        (["zip"], "city"), (["zip"], "state"), (["zip"], "area_code"),
        (["city", "state"], "zip"), (["area_code", "state"], "city"),
        (["marital_status", "has_child", "salary"], "rate"),
        (["marital_status", "has_child", "salary"], "single_exemp"),
        (["marital_status", "has_child", "salary"], "married_exemp"),
        (["marital_status", "has_child", "salary"], "child_exemp"),
    ]
    for lhs, rhs in pair_specs:
        if any(c not in norm_df.columns for c in lhs + [rhs]):
            continue
        sub = norm_df[lhs + [rhs]].dropna()
        if len(sub) < PAIR_CONTEXT_MIN_GROUP_SIZE:
            continue
        grouped = sub.groupby(lhs, dropna=True)[rhs]
        majority_map, total_support = {}, 0
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
        if majority_map:
            rtype = "dominant_value_by_context_pair" if len(lhs) >= 2 else "dominant_value_by_context"
            rules.append({"rule_id": f"CAND_PAIRCTX_{rule_id}", "rule_type": rtype, "lhs": lhs, "rhs": rhs, "description": f"context dominant: {', '.join(lhs)} -> {rhs}", "mapping_size": len(majority_map), "majority_map": majority_map, "confidence": 0.84, "support": total_support, "usage_role": "candidate_generation_rule"})
            rule_id += 1
    return rules


def build_tax_consistency_rules(df, profiles):
    rules, rule_id = [], 1
    if {"zip", "state"}.issubset(df.columns):
        rules.append({"rule_id": f"CAND_TAXCONS_{rule_id}", "rule_type": "zip_state_consistency", "lhs": ["zip"], "rhs": "state", "description": "zip should be consistent with state by dominant evidence in the table", "confidence": 0.88, "support": profiles.get("zip", {}).get("non_null_count", len(df)), "usage_role": "candidate_generation_rule"}); rule_id += 1
    if {"zip", "city"}.issubset(df.columns):
        rules.append({"rule_id": f"CAND_TAXCONS_{rule_id}", "rule_type": "zip_city_consistency", "lhs": ["zip"], "rhs": "city", "description": "zip should be consistent with city by dominant evidence in the table", "confidence": 0.86, "support": profiles.get("zip", {}).get("non_null_count", len(df)), "usage_role": "candidate_generation_rule"}); rule_id += 1
    if {"zip", "area_code"}.issubset(df.columns):
        rules.append({"rule_id": f"CAND_TAXCONS_{rule_id}", "rule_type": "zip_area_code_consistency", "lhs": ["zip"], "rhs": "area_code", "description": "zip should be consistent with area_code by dominant evidence in the table", "confidence": 0.82, "support": profiles.get("zip", {}).get("non_null_count", len(df)), "usage_role": "candidate_generation_rule"}); rule_id += 1
    if {"marital_status", "single_exemp", "married_exemp"}.issubset(df.columns):
        rules.append({"rule_id": f"CAND_TAXCONS_{rule_id}", "rule_type": "marital_exemption_consistency", "columns": ["marital_status", "single_exemp", "married_exemp"], "description": "single filers should usually have married_exemp=0 and married filers should usually have single_exemp=0", "confidence": 0.90, "support": len(df), "usage_role": "strong_rule"}); rule_id += 1
    if {"has_child", "child_exemp"}.issubset(df.columns):
        rules.append({"rule_id": f"CAND_TAXCONS_{rule_id}", "rule_type": "child_exemption_consistency", "columns": ["has_child", "child_exemp"], "description": "records with has_child=N should usually have child_exemp=0", "confidence": 0.90, "support": len(df), "usage_role": "strong_rule"}); rule_id += 1
    if {"salary", "rate"}.issubset(df.columns):
        rules.append({"rule_id": f"CAND_TAXCONS_{rule_id}", "rule_type": "salary_rate_context_consistency", "lhs": ["salary"], "rhs": "rate", "description": "rate should be consistent with salary groups by dominant evidence in the table", "confidence": 0.82, "support": profiles.get("salary", {}).get("non_null_count", len(df)), "usage_role": "candidate_generation_rule"}); rule_id += 1
    if {"marital_status", "has_child", "salary", "rate"}.issubset(df.columns):
        rules.append({"rule_id": f"CAND_TAXCONS_{rule_id}", "rule_type": "tax_profile_rate_consistency", "lhs": ["marital_status", "has_child", "salary"], "rhs": "rate", "description": "rate should be consistent with marital_status, has_child and salary profile", "confidence": 0.84, "support": len(df), "usage_role": "candidate_generation_rule"}); rule_id += 1
    return rules


def build_candidate_generation_rules(df, profiles):
    rules, rule_id = [], 1
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
    for rhs in FD_RHS_CANDIDATE_COLUMNS:
        if rhs not in df.columns:
            continue
        for lhs in lhs_candidates:
            if lhs == rhs:
                continue
            sub = norm_df[[lhs, rhs]].dropna()
            if len(sub) < SOFT_FD_MIN_GROUP_SIZE:
                continue
            grouped = sub.groupby(lhs)[rhs]
            majority_map, total_support = {}, 0
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
                rules.append({"rule_id": f"CAND_RAREVAL_{rule_id}", "rule_type": "rare_value", "column": col, "description": f"rare values in {col} may be suspicious", "rare_values": rare_values[:5000], "confidence": 0.6, "support": sum(counter[v] for v in rare_values), "usage_role": "candidate_generation_rule"})
                rule_id += 1
    if ENABLE_RARE_PATTERN_RULE:
        for col, prof in profiles.items():
            if col not in TARGET_CANDIDATE_COLUMNS or col in SKIP_RARE_PATTERN_COLUMNS:
                continue
            rare_patterns = [x["pattern"] for x in prof.get("top_patterns", []) if x["ratio"] <= RARE_PATTERN_MAX_RATIO]
            if rare_patterns:
                rules.append({"rule_id": f"CAND_RAREPAT_{rule_id}", "rule_type": "rare_pattern", "column": col, "description": f"rare patterns in {col} may be suspicious", "rare_patterns": rare_patterns, "confidence": 0.6, "support": prof["non_null_count"], "usage_role": "candidate_generation_rule"})
                rule_id += 1
    for col in HIGH_RISK_MISSING_COLUMNS:
        if col not in profiles:
            continue
        prof = profiles[col]
        rules.append({"rule_id": f"CAND_HRMISS_{rule_id}", "rule_type": "high_risk_missing", "column": col, "description": f"{col} missing values are suspicious in candidate generation", "confidence": 0.8, "support": prof["non_null_count"], "usage_role": "candidate_generation_rule"})
        rule_id += 1
    for a, b in [("area_code", "phone"), ("city", "state"), ("city", "zip"), ("state", "zip"), ("marital_status", "single_exemp"), ("marital_status", "married_exemp"), ("has_child", "child_exemp")]:
        if a in df.columns and b in df.columns:
            rules.append({"rule_id": f"CAND_PAIR_{rule_id}", "rule_type": "paired_missing_consistency", "columns": [a, b], "description": f"{a} and {b} should usually co-occur", "confidence": 0.8, "support": len(df), "usage_role": "candidate_generation_rule"})
            rule_id += 1
    rules.extend(build_pair_context_rules(df))
    rules.extend(build_tax_consistency_rules(df, profiles))
    return rules


def assign_rule_priority(rule):
    rtype = rule.get("rule_type", "")
    if rtype in {"gender_domain", "state_code_domain", "marital_status_domain", "has_child_domain", "area_code_format", "phone_format", "zip_format", "salary_numeric_range", "rate_numeric_range", "exemption_numeric_range", "marital_exemption_consistency", "child_exemption_consistency"}:
        return "high"
    if rtype in {"zip_state_consistency", "zip_city_consistency", "zip_area_code_consistency", "salary_rate_context_consistency", "tax_profile_rate_consistency", "phone_canonical_format", "zip_canonical_format"}:
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
            "design_goal": "high_recall_candidate_generation_for_tax",
            "target_candidate_columns": sorted(list(TARGET_CANDIDATE_COLUMNS)),
            "notes": [
                "candidate inclusion is determined only by violating rules in this pool",
                "index is treated as context and is not a target candidate column",
                "tax-specific rules include domain checks, format checks, ZIP/state/city/area-code consistency, and tax exemption consistency",
                "rare_value is disabled for high-cardinality name/location/contact fields to avoid excessive false positives",
                "FD and soft FD rules are mined but high-cardinality name/phone columns are restricted to avoid accidental dependencies",
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

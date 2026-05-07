import json
import math
from collections import Counter
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd


# ============================================================
# 1. 用户配置区
# ============================================================

INPUT_TABLE_CSV = "/mnt/mydata/dq/projects/Splittree/hospital/hospital_dirty_no_address23.csv"
INPUT_SHRUNK_CSV = "candidate_cells_v2.csv"

OUTPUT_JSONL = "candidate_llm_contexts_v2.jsonl"
OUTPUT_CSV = "candidate_llm_contexts_flat_v2.csv"

MISSING_TOKENS = {"", "empty", "null", "none", "nan", "na", "n/a", "missing", "unk"}

NUMERIC_RATIO_THRESHOLD = 0.90
CATEGORICAL_UNIQUE_COUNT_THRESHOLD = 30
CATEGORICAL_UNIQUE_RATIO_THRESHOLD = 0.05
ID_LIKE_UNIQUE_RATIO_THRESHOLD = 0.98
CODE_LIKE_AVG_LEN_THRESHOLD = 20

TOP_K_COLUMN_VALUES = 10
TOP_K_ALT_VALUES = 10
TOP_K_SIMILAR_ROWS = 5
TOP_K_ROW_FIELDS_FOR_LLM = 12

# 贝叶斯启发式权重
W_RULE = 2.0
W_RARITY = 1.0
W_PATTERN = 0.8
W_NEIGHBOR = 1.2


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
        elif ch in {" ", "-", "_", "/", ".", ",", "(", ")", "%", ":", ";", "&", "'"}:
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
    """
    兼容多种 reason 模板：
    1) suggests rhs=xxx, observed=yyy
    2) suggests Col=xxx, observed=yyy
    3) dominant=xxx, observed=yyy
    """
    if not reason:
        return None

    patterns = [
        ("suggests rhs=", ", observed="),
        ("suggests ", ", observed="),
        ("dominant=", ", observed="),
    ]

    for left, right in patterns:
        if left in reason and right in reason:
            try:
                a = reason.index(left) + len(left)
                b = reason.index(right, a)
                value = reason[a:b].strip()

                # 针对 "suggests HospitalType=acute care hospitals" 这种情况
                if left == "suggests " and "=" in value:
                    value = value.split("=", 1)[1].strip()

                return value if value else None
            except Exception:
                pass

    return None


# ============================================================
# 3. 加载数据
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
# 4. 列类型与列统计
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

        if numeric_ratio >= NUMERIC_RATIO_THRESHOLD:
            detected_type = "numeric"
        elif unique_count <= CATEGORICAL_UNIQUE_COUNT_THRESHOLD or unique_ratio <= CATEGORICAL_UNIQUE_RATIO_THRESHOLD:
            detected_type = "categorical"
        else:
            detected_type = "text"

        if is_id_like_column(non_null, unique_ratio, numeric_ratio, avg_len):
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
# 5. 行上下文
# ============================================================

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
    }


# ============================================================
# 6. 列上下文
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
    }


# ============================================================
# 7. 冲突信息
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

    # 核心统计
    main_rule_type = rule_type_counter.most_common(1)[0][0] if rule_type_counter else "none"
    main_usage_role = usage_role_counter.most_common(1)[0][0] if usage_role_counter else "none"

    strong_rule_count = usage_role_counter.get("strong_rule", 0)
    candidate_generation_rule_count = usage_role_counter.get("candidate_generation_rule", 0)

    fd_like_count = (
        rule_type_counter.get("functional_dependency", 0)
        + rule_type_counter.get("soft_functional_dependency", 0)
    )

    context_rule_count = rule_type_counter.get("dominant_value_by_context", 0)
    global_rule_count = rule_type_counter.get("global_dominant_value", 0)

    # 新增细分统计
    rare_value_count = rule_type_counter.get("rare_value", 0)
    typo_rule_count = rule_type_counter.get("dominant_value_typo", 0)

    pattern_rule_count = (
        rule_type_counter.get("pattern", 0)
        + rule_type_counter.get("rare_pattern", 0)
        + rule_type_counter.get("score_format", 0)
        + rule_type_counter.get("sample_format", 0)
        + rule_type_counter.get("sample_patients_suffix", 0)
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

        "expected_values_from_rules": extracted_expected_values,
        "alternative_values_from_column": column_alternative_values,
        "all_alternative_values": extracted_expected_values + column_alternative_values
    }


# ============================================================
# 8. 相似信息
# ============================================================

def single_cell_similarity(a, b, semantic_type: str) -> float:
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
# 9. 贝叶斯先验/后验
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
    similarity_info: Dict[str, Any]
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
    elif semantic_type == "text":
        rarity *= 1.2

    neighbor_majority_value = similarity_info.get("neighbor_majority_value")
    neighbor_majority_ratio = similarity_info.get("neighbor_majority_ratio")
    if neighbor_majority_value is None or neighbor_majority_ratio is None:
        neighbor_signal = 0.0
    else:
        neighbor_signal = neighbor_majority_ratio if neighbor_majority_value != value else 0.0

    logit_post = (
        safe_logit(prior)
        + W_RULE * rule_signal
        + W_RARITY * rarity
        + W_PATTERN * pattern_signal
        + W_NEIGHBOR * neighbor_signal
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
        },
        "note": "posterior is a heuristic Bayesian-style approximation"
    }


# ============================================================
# 10. 构造给 LLM 的上下文文本
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
    lines.append("Row context:")
    lines.append(f"- missing_count: {row_ctx['missing_count']}")
    lines.append(f"- non_missing_count: {row_ctx['non_missing_count']}")
    lines.append(f"- avg_value_length: {row_ctx['avg_value_length']}")
    lines.append("- row field values:")
    for k, v in row_ctx["row_values_for_llm"]:
        lines.append(f"  - {k}: {v}")

    lines.append("")
    lines.append("Column context:")
    lines.append(f"- semantic_type: {col_ctx['semantic_type']}")
    lines.append(f"- missing_rate: {col_ctx['missing_rate']}")
    lines.append(f"- unique_rate: {col_ctx['unique_rate']}")
    lines.append(f"- value_frequency: {col_ctx['value_frequency']}")
    lines.append(f"- value_frequency_rank: {col_ctx['value_frequency_rank']}")
    lines.append(f"- current_pattern: {col_ctx['current_value_pattern']}")
    lines.append(f"- dominant_pattern: {col_ctx['dominant_pattern']}")
    lines.append(f"- charset_features: {col_ctx['charset_features']}")
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
# 11. 主流程
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
            bayes_ctx = summarize_bayesian_info(cand, df, col_profiles, candidate_stats, sim_ctx)

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

            flat_rows.append({
                "row_id": row_id,
                "column": col,
                "value": value,
                "semantic_type": col_ctx["semantic_type"],
                "violation_count": cand.get("violation_count"),
                "conflict_score": cand.get("conflict_score"),
                "value_frequency": col_ctx["value_frequency"],
                "value_frequency_rank": col_ctx["value_frequency_rank"],
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
            })

    pd.DataFrame(flat_rows).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"[OK] Full LLM contexts saved to: {OUTPUT_JSONL}")
    print(f"[OK] Flat summary saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
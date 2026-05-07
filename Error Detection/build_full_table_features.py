import json
import math
import re
from collections import Counter, defaultdict
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd


# ============================================================
# 1. 用户配置区
# ============================================================

INPUT_TABLE_CSV = "/home/dq/Splittree/hospital/hospital_dirty_no_address23.csv"
INPUT_RULE_JSON = "rule_pool_new.json"
OUTPUT_CSV = "full_table_features_new.csv"

MISSING_TOKENS = {"", "empty", "null", "none", "nan", "na", "n/a", "missing", "unk"}

# 是否跳过空值参与非 not_null 规则检查
SKIP_NULL_FOR_NON_NOTNULL_RULES = True

# FD 映射上限
MAX_FD_MAPPING_SIZE = 200000

# 冲突分数权重
PRIORITY_WEIGHT = {
    "high": 1.5,
    "medium": 1.0,
    "low": 0.6,
}

RULE_TYPE_WEIGHT = {
    "not_null": 1.2,
    "numeric_range": 0.8,
    "enumeration_domain": 0.9,
    "pattern": 0.7,
    "functional_dependency": 1.5,
}

# 列语义判定阈值
NUMERIC_RATIO_THRESHOLD = 0.90
CATEGORICAL_UNIQUE_COUNT_THRESHOLD = 30
CATEGORICAL_UNIQUE_RATIO_THRESHOLD = 0.05
ID_LIKE_UNIQUE_RATIO_THRESHOLD = 0.98
CODE_LIKE_AVG_LEN_THRESHOLD = 20

# 贝叶斯风格分数权重
W_RULE = 2.0
W_RARITY = 1.0
W_PATTERN = 0.8
W_NEIGHBOR = 1.2

# 相似行
TOP_K_SIMILAR_ROWS = 5


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
    return [{"value": k, "count": v} for k, v in cnt.most_common(topk)]


def is_id_like_column(non_null: pd.Series, unique_ratio: float, numeric_ratio: float, avg_len: float) -> bool:
    if len(non_null) == 0:
        return False
    if unique_ratio >= ID_LIKE_UNIQUE_RATIO_THRESHOLD and avg_len <= CODE_LIKE_AVG_LEN_THRESHOLD:
        return True
    if unique_ratio >= ID_LIKE_UNIQUE_RATIO_THRESHOLD and numeric_ratio >= NUMERIC_RATIO_THRESHOLD:
        return True
    return False


def is_code_like_column(non_null: pd.Series, unique_ratio: float, numeric_ratio: float, avg_len: float, top_patterns: List[Dict[str, Any]]) -> bool:
    if len(non_null) == 0:
        return False
    if avg_len <= CODE_LIKE_AVG_LEN_THRESHOLD and unique_ratio > CATEGORICAL_UNIQUE_RATIO_THRESHOLD:
        if top_patterns and top_patterns[0]["ratio"] >= 0.7:
            return True
    if avg_len <= 15 and numeric_ratio >= 0.95 and unique_ratio < ID_LIKE_UNIQUE_RATIO_THRESHOLD:
        return True
    return False


def rule_weight(rule: Dict[str, Any]) -> float:
    conf = float(rule.get("confidence", 0.0))
    supp = int(rule.get("support", 0))
    priority = rule.get("priority", "low")
    rule_type = rule.get("rule_type", "")

    priority_w = PRIORITY_WEIGHT.get(priority, 0.6)
    type_w = RULE_TYPE_WEIGHT.get(rule_type, 1.0)
    support_part = math.log1p(max(supp, 0))
    score = (0.1 + conf) * support_part * priority_w * type_w
    return round(score, 6)


# ============================================================
# 3. 加载规则池
# ============================================================

def load_rule_pool(rule_json_path: str) -> List[Dict[str, Any]]:
    with open(rule_json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    all_rules = []
    rule_groups = obj.get("rules", {})
    for group_name, rules in rule_groups.items():
        for rule in rules:
            r = dict(rule)
            r["_group_name"] = group_name
            all_rules.append(r)
    return all_rules


def build_fd_majority_mapping(df: pd.DataFrame, lhs: List[str], rhs: str) -> Dict[Tuple, Tuple[Any, int, int]]:
    needed_cols = lhs + [rhs]
    sub = df[needed_cols].dropna()

    if len(sub) == 0:
        return {}

    grouped = sub.groupby(lhs, dropna=True)
    mapping = {}

    for group_key, grp in grouped:
        rhs_vals = grp[rhs].tolist()
        cnt = Counter(rhs_vals)
        majority_val, majority_cnt = cnt.most_common(1)[0]
        total_cnt = len(rhs_vals)

        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        mapping[group_key] = (majority_val, majority_cnt, total_cnt)

        if len(mapping) >= MAX_FD_MAPPING_SIZE:
            break

    return mapping


def prepare_fd_rule_runtime(df: pd.DataFrame, rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    fd_rules = []
    for rule in rules:
        if rule.get("rule_type") != "functional_dependency":
            continue
        lhs = rule["lhs"]
        rhs = rule["rhs"]
        mapping = build_fd_majority_mapping(df, lhs, rhs)
        r = dict(rule)
        r["_majority_mapping"] = mapping
        fd_rules.append(r)
    return fd_rules


# ============================================================
# 4. 列统计与语义类型
# ============================================================

def analyze_columns(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    profiles = {}

    for col in df.columns:
        s = df[col]
        non_null = s.dropna()

        missing_rate = 1.0 - (len(non_null) / len(df) if len(df) > 0 else 0.0)
        unique_count = non_null.nunique()
        unique_ratio = unique_count / len(non_null) if len(non_null) > 0 else 0.0

        top_values = value_frequency_dict(non_null, topk=20)
        freq_counter = Counter(non_null.tolist())

        lengths = non_null.map(lambda x: len(str(x)))
        avg_len = float(lengths.mean()) if len(lengths) > 0 else 0.0

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
            "numeric_ratio": round(float(numeric_ratio), 6),
            "pattern_counter": pattern_counter,
            "patterns": pattern_list,
            "dominant_pattern": pattern_list[0]["pattern"] if pattern_list else None,
            "dominant_pattern_ratio": pattern_list[0]["ratio"] if pattern_list else None,
        }

    return profiles


# ============================================================
# 5. 单元格规则检查
# ============================================================

def check_not_null_rule(row_idx: int, row: pd.Series, rule: Dict[str, Any]):
    col = rule["column"]
    if col not in row:
        return None
    val = row[col]
    if val is None:
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": "value is null but rule requires non-null",
        }
    return None


def check_numeric_range_rule(row_idx: int, row: pd.Series, rule: Dict[str, Any]):
    col = rule["column"]
    if col not in row:
        return None

    val = row[col]
    if val is None and SKIP_NULL_FOR_NON_NOTNULL_RULES:
        return None

    num = try_parse_one_numeric(val)
    if num is None:
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": "value cannot be parsed as numeric for numeric_range rule",
        }

    min_val = rule["range"]["min"]
    max_val = rule["range"]["max"]

    if not (min_val <= num <= max_val):
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": f"value {num} is outside range [{min_val}, {max_val}]",
        }
    return None


def check_enumeration_rule(row_idx: int, row: pd.Series, rule: Dict[str, Any]):
    col = rule["column"]
    if col not in row:
        return None

    val = row[col]
    if val is None and SKIP_NULL_FOR_NON_NOTNULL_RULES:
        return None

    allowed = set(rule.get("allowed_values", []))
    if val not in allowed:
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": "value not in allowed domain",
        }
    return None


def check_pattern_rule(row_idx: int, row: pd.Series, rule: Dict[str, Any]):
    col = rule["column"]
    if col not in row:
        return None

    val = row[col]
    if val is None and SKIP_NULL_FOR_NON_NOTNULL_RULES:
        return None

    observed = detect_basic_pattern(val)
    target = rule["pattern"]

    if observed != target:
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": f"pattern mismatch: observed={observed}, expected={target}",
        }
    return None


def check_fd_rule(row_idx: int, row: pd.Series, rule: Dict[str, Any]):
    lhs = rule["lhs"]
    rhs = rule["rhs"]

    for c in lhs + [rhs]:
        if c not in row:
            return None

    lhs_values = []
    for c in lhs:
        v = row[c]
        if v is None:
            return None
        lhs_values.append(v)

    rhs_val = row[rhs]
    if rhs_val is None and SKIP_NULL_FOR_NON_NOTNULL_RULES:
        return None

    lhs_key = tuple(lhs_values)
    mapping = rule.get("_majority_mapping", {})
    if lhs_key not in mapping:
        return None

    expected_rhs, majority_cnt, total_cnt = mapping[lhs_key]

    if rhs_val != expected_rhs:
        return {
            "row_id": row_idx,
            "column": rhs,
            "value": rhs_val,
            "rule": rule,
            "reason": f"FD violation: lhs={lhs_key} suggests rhs={expected_rhs}, observed={rhs_val}",
            "expected_value": expected_rhs,
            "rule_type": "functional_dependency",
        }
    return None


# ============================================================
# 6. 相似行
# ============================================================

def single_cell_similarity(a, b, semantic_type: str) -> Optional[float]:
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


def row_similarity(target_row: pd.Series, other_row: pd.Series, col_profiles: Dict[str, Dict[str, Any]], ignore_col: str = None) -> float:
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


def summarize_similarity_info(df: pd.DataFrame, row_id: int, target_col: str, col_profiles: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    target_row = df.iloc[row_id]
    sims = []

    for i in range(len(df)):
        if i == row_id:
            continue
        sim = row_similarity(target_row, df.iloc[i], col_profiles, ignore_col=target_col)
        sims.append((i, sim))

    sims.sort(key=lambda x: x[1], reverse=True)
    topk = sims[:TOP_K_SIMILAR_ROWS]

    target_col_values = []
    for idx, sim in topk:
        target_col_values.append(df.iloc[idx][target_col])

    value_dist = Counter([v for v in target_col_values if v is not None])
    majority_value = value_dist.most_common(1)[0][0] if value_dist else None
    majority_ratio = round(value_dist.most_common(1)[0][1] / len(target_col_values), 6) if value_dist and len(target_col_values) > 0 else None

    return {
        "neighbor_majority_value": majority_value,
        "neighbor_majority_ratio": majority_ratio,
    }


# ============================================================
# 7. 构造全表特征
# ============================================================

def build_full_table_features():
    # 读表
    df = pd.read_csv(INPUT_TABLE_CSV, dtype=str)
    df = maybe_drop_unnamed_columns(df)
    for col in df.columns:
        df[col] = normalize_series(df[col])

    # 列分析
    col_profiles = analyze_columns(df)

    # 读规则
    rules = load_rule_pool(INPUT_RULE_JSON)
    not_null_rules = [r for r in rules if r.get("rule_type") == "not_null"]
    numeric_range_rules = [r for r in rules if r.get("rule_type") == "numeric_range"]
    enumeration_rules = [r for r in rules if r.get("rule_type") == "enumeration_domain"]
    pattern_rules = [r for r in rules if r.get("rule_type") == "pattern"]
    fd_rules = prepare_fd_rule_runtime(df, rules)

    # 为加速，按列索引单列规则
    rules_by_col = defaultdict(list)
    for r in not_null_rules + numeric_range_rules + enumeration_rules + pattern_rules:
        if "column" in r:
            rules_by_col[r["column"]].append(r)

    # FD 按 rhs 列索引
    fd_rules_by_rhs = defaultdict(list)
    for r in fd_rules:
        fd_rules_by_rhs[r["rhs"]].append(r)

    # 统计每列候选数量，用来做先验
    # 先扫描一遍 violation_count，后面再出最终结果
    preliminary_counts_by_col = Counter()
    preliminary_records = []

    for row_idx, row in df.iterrows():
        row_series = row
        for col in df.columns:
            value = row_series[col]
            violated = []

            # 单列规则
            for rule in rules_by_col.get(col, []):
                rt = rule.get("rule_type")
                if rt == "not_null":
                    v = check_not_null_rule(row_idx, row_series, rule)
                elif rt == "numeric_range":
                    v = check_numeric_range_rule(row_idx, row_series, rule)
                elif rt == "enumeration_domain":
                    v = check_enumeration_rule(row_idx, row_series, rule)
                elif rt == "pattern":
                    v = check_pattern_rule(row_idx, row_series, rule)
                else:
                    v = None
                if v is not None:
                    violated.append(v)

            # FD 规则（只检查 rhs == 当前列）
            for rule in fd_rules_by_rhs.get(col, []):
                v = check_fd_rule(row_idx, row_series, rule)
                if v is not None:
                    violated.append(v)

            violation_count = len(violated)
            if violation_count > 0:
                preliminary_counts_by_col[col] += 1

            preliminary_records.append({
                "row_id": int(row_idx),
                "column": col,
                "value": value,
                "violations": violated,
            })

    # 二次生成完整特征
    output_rows = []

    total_rows = len(df)

    for rec in preliminary_records:
        row_id = rec["row_id"]
        col = rec["column"]
        value = rec["value"]
        violated = rec["violations"]

        # semantic_type
        semantic_type = col_profiles[col]["semantic_type"]

        # violation_count / conflict_score
        violation_count = len(violated)
        conflict_score = round(sum(rule_weight(v["rule"]) for v in violated), 6)

        # value_frequency / rank
        counter = col_profiles[col]["value_counter"]
        value_frequency = int(counter.get(value, 0))
        value_frequency_rank = None
        if value is not None:
            sorted_vals = counter.most_common()
            for idx, (v, c) in enumerate(sorted_vals, start=1):
                if v == value:
                    value_frequency_rank = float(idx)
                    break

        # similar rows
        sim_info = summarize_similarity_info(df, row_id, col, col_profiles)
        neighbor_majority_value = sim_info["neighbor_majority_value"]
        neighbor_majority_ratio = sim_info["neighbor_majority_ratio"]

        # prior
        col_candidate_count = preliminary_counts_by_col.get(col, 0)
        prior = col_candidate_count / total_rows if total_rows > 0 else 0.01
        prior = min(max(prior, 1e-4), 0.95)

        # posterior
        rule_signal = min(5.0, violation_count + conflict_score / 100.0)

        non_null_count = col_profiles[col]["non_null_count"]
        rarity = 1.0 - (value_frequency / non_null_count) if non_null_count > 0 else 1.0
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

        # main_rule_type
        if violation_count > 0:
            rt_counter = Counter(v["rule"].get("rule_type", "unknown") for v in violated)
            main_rule_type = rt_counter.most_common(1)[0][0]
        else:
            main_rule_type = "none"

        # pattern_bucket
        pattern_bucket = "pattern_match" if current_pattern == dominant_pattern else "pattern_mismatch"

        # rarity_bucket
        if value_frequency_rank is not None:
            if value_frequency_rank <= 3:
                rarity_bucket = "very_common"
            elif value_frequency_rank <= 10:
                rarity_bucket = "common"
            else:
                if value_frequency <= 1:
                    rarity_bucket = "singleton"
                elif value_frequency <= 2:
                    rarity_bucket = "very_rare"
                elif value_frequency <= 5:
                    rarity_bucket = "rare"
                else:
                    rarity_bucket = "mid_frequency"
        else:
            rarity_bucket = "unknown"

        # neighbor_bucket
        if neighbor_majority_ratio is None:
            neighbor_bucket = "no_neighbor_signal"
        else:
            same = str(neighbor_majority_value) == str(value)
            if not same:
                if neighbor_majority_ratio >= 0.8:
                    neighbor_bucket = "strong_support_other"
                elif neighbor_majority_ratio >= 0.5:
                    neighbor_bucket = "medium_support_other"
                else:
                    neighbor_bucket = "weak_support_other"
            else:
                if neighbor_majority_ratio >= 0.8:
                    neighbor_bucket = "strong_support_current"
                else:
                    neighbor_bucket = "weak_support_current"

        output_rows.append({
            "row_id": row_id,
            "column": col,
            "value": value,
            "semantic_type": semantic_type,
            "violation_count": int(violation_count),
            "conflict_score": float(conflict_score),
            "value_frequency": int(value_frequency),
            "value_frequency_rank": value_frequency_rank,
            "neighbor_majority_value": neighbor_majority_value,
            "neighbor_majority_ratio": neighbor_majority_ratio,
            "prior_error_probability": round(float(prior), 6),
            "posterior_error_probability": round(float(posterior), 6),
            "main_rule_type": main_rule_type,
            "pattern_bucket": pattern_bucket,
            "rarity_bucket": rarity_bucket,
            "neighbor_bucket": neighbor_bucket,
        })

    out_df = pd.DataFrame(output_rows)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"[OK] full table features saved to: {OUTPUT_CSV}")
    print(f"Rows in original table: {len(df)}")
    print(f"Columns in original table: {len(df.columns)}")
    print(f"Total cells exported: {len(out_df)}")


if __name__ == "__main__":
    build_full_table_features()
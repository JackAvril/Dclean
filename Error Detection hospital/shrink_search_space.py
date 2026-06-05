import json
import math
import re
from collections import Counter
from typing import Dict, Any, List, Tuple

import pandas as pd


# ============================================================
# 1. 用户配置区
# ============================================================

INPUT_CSV = "/mnt/mydata/dq/projects/Splittree/hospital/hospital_dirty_no_address23.csv"
INPUT_RULE_JSON = "rule_pool_high_recall_v2.json"

OUTPUT_JSONL = "candidate_cells_v2.jsonl"
OUTPUT_CSV = "candidate_cells_v2.csv"

MISSING_TOKENS = {"", "empty", "null", "none", "nan", "na", "n/a", "missing", "unk"}

# 是否跳过缺失值不参与某些规则检查
SKIP_NULL_FOR_NON_NOTNULL_RULES = True

# 每条 FD 规则最多缓存多少个 lhs -> majority rhs
MAX_FD_MAPPING_SIZE = 200000
MAX_CONTEXT_MAPPING_SIZE = 200000

# 用于冲突分数的权重
PRIORITY_WEIGHT = {
    "high": 1.5,
    "medium": 1.0,
    "low": 0.6
}

RULE_TYPE_WEIGHT = {
    "not_null": 1.2,
    "numeric_range": 0.8,
    "enumeration_domain": 0.9,
    "pattern": 0.7,
    "functional_dependency": 1.5,
    "soft_functional_dependency": 1.1,
    "numeric_outlier_iqr": 0.9,
    "rare_value": 0.7,
    "rare_pattern": 0.7,
    "global_dominant_value": 0.6,
    "dominant_value_by_context": 0.9,
    "high_risk_missing": 0.9,
    "paired_missing_consistency": 0.8,
    "dominant_value_typo": 0.95,
    "score_format": 0.85,
    "sample_format": 0.85,
    "sample_patients_suffix": 0.85,
}


# ============================================================
# 2. 基础工具函数
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


def detect_basic_pattern(value: str) -> str:
    if pd.isna(value):
        return "NULL"

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


def try_parse_one_numeric(value):
    if value is None:
        return None
    s = str(value).replace(",", "").replace("%", "")
    try:
        return float(s)
    except Exception:
        return None


def load_rule_pool(rule_json_path: str) -> List[Dict[str, Any]]:
    with open(rule_json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    all_rules = []
    rule_groups = obj.get("rules", {})
    for _, rules in rule_groups.items():
        for rule in rules:
            r = dict(rule)
            all_rules.append(r)

    return all_rules


def rule_weight(rule: Dict[str, Any]) -> float:
    conf = float(rule.get("confidence", rule.get("overall_confidence", 0.0)))
    supp = int(rule.get("support", 0))
    priority = rule.get("priority", "low")
    rule_type = rule.get("rule_type", "")

    priority_w = PRIORITY_WEIGHT.get(priority, 0.6)
    type_w = RULE_TYPE_WEIGHT.get(rule_type, 1.0)

    support_part = math.log1p(max(supp, 1))
    score = (0.1 + conf) * support_part * priority_w * type_w
    return round(score, 6)


def normalized_edit_distance_ratio(a: str, b: str) -> float:
    if a is None or b is None:
        return 1.0
    a = str(a)
    b = str(b)

    if a == b:
        return 0.0

    n, m = len(a), len(b)
    if n == 0 and m == 0:
        return 0.0
    if n == 0 or m == 0:
        return 1.0

    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            tmp = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(
                dp[j] + 1,
                dp[j - 1] + 1,
                prev + cost
            )
            prev = tmp

    dist = dp[m]
    return dist / max(n, m)


# ============================================================
# 3. 为 FD / 上下文规则预构建运行时映射
# ============================================================

def build_fd_majority_mapping(df: pd.DataFrame, lhs: List[str], rhs: str) -> Dict[Tuple, Tuple[Any, int, int]]:
    """
    返回:
        mapping[key] = (majority_rhs_value, majority_count, total_count)
    key 是 lhs 的取值元组
    """
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
    """
    给 functional_dependency / soft_functional_dependency 补充运行期字段:
        _majority_mapping
    """
    fd_like_rules = []
    for rule in rules:
        if rule.get("rule_type") not in {"functional_dependency", "soft_functional_dependency"}:
            continue

        lhs = rule["lhs"]
        rhs = rule["rhs"]
        mapping = build_fd_majority_mapping(df, lhs, rhs)

        r = dict(rule)
        r["_majority_mapping"] = mapping
        fd_like_rules.append(r)

    return fd_like_rules


def prepare_context_dominant_rule_runtime(rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    给 dominant_value_by_context 规则补充:
        _context_mapping[context_value] = (dominant_value, support, dominant_ratio)
    """
    ctx_rules = []

    for rule in rules:
        if rule.get("rule_type") != "dominant_value_by_context":
            continue

        mapping = {}
        for item in rule.get("mappings", []):
            ctx_val = item.get("context_value")
            dom_val = item.get("dominant_value")
            support = int(item.get("support", 0))
            dom_ratio = float(item.get("dominant_ratio", 0.0))
            mapping[ctx_val] = (dom_val, support, dom_ratio)

            if len(mapping) >= MAX_CONTEXT_MAPPING_SIZE:
                break

        r = dict(rule)
        r["_context_mapping"] = mapping
        ctx_rules.append(r)

    return ctx_rules


# ============================================================
# 4. 单元格违规/候选检测
# ============================================================

def check_not_null_rule(row_idx: int, row: pd.Series, rule: Dict[str, Any]):
    col = rule["column"]
    if col not in row:
        return None

    val = row[col]
    if pd.isna(val):
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": "value is null but rule requires non-null"
        }
    return None


def check_numeric_range_rule(row_idx: int, row: pd.Series, rule: Dict[str, Any]):
    col = rule["column"]
    if col not in row:
        return None

    val = row[col]
    if pd.isna(val) and SKIP_NULL_FOR_NON_NOTNULL_RULES:
        return None

    num = try_parse_one_numeric(val)
    if num is None:
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": "value cannot be parsed as numeric for numeric_range rule"
        }

    min_val = rule["range"]["min"]
    max_val = rule["range"]["max"]

    if not (min_val <= num <= max_val):
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": f"value {num} is outside range [{min_val}, {max_val}]"
        }

    return None


def check_enumeration_rule(row_idx: int, row: pd.Series, rule: Dict[str, Any]):
    col = rule["column"]
    if col not in row:
        return None

    val = row[col]
    if pd.isna(val) and SKIP_NULL_FOR_NON_NOTNULL_RULES:
        return None

    allowed = set(rule.get("allowed_values", []))
    if val not in allowed:
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": "value not in allowed domain"
        }

    return None


def check_pattern_rule(row_idx: int, row: pd.Series, rule: Dict[str, Any]):
    col = rule["column"]
    if col not in row:
        return None

    val = row[col]
    if pd.isna(val) and SKIP_NULL_FOR_NON_NOTNULL_RULES:
        return None

    observed = detect_basic_pattern(val)
    target = rule["pattern"]

    if observed != target:
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": f"pattern mismatch: observed={observed}, expected={target}"
        }

    return None


def check_fd_like_rule(row_idx: int, row: pd.Series, rule: Dict[str, Any]):
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
            "reason": f"{rule['rule_type']} violation: lhs={lhs_key} suggests rhs={expected_rhs}, observed={rhs_val}",
            "expected_value": expected_rhs,
            "lhs_values": {c: row[c] for c in lhs},
            "fd_group_majority_count": majority_cnt,
            "fd_group_total_count": total_cnt
        }

    return None


def check_numeric_outlier_rule(row_idx: int, row: pd.Series, rule: Dict[str, Any]):
    col = rule["column"]
    if col not in row:
        return None

    val = row[col]
    if pd.isna(val) and SKIP_NULL_FOR_NON_NOTNULL_RULES:
        return None

    num = try_parse_one_numeric(val)
    if num is None:
        return None

    lower = rule.get("lower_bound")
    upper = rule.get("upper_bound")
    if lower is None or upper is None:
        return None

    if num < lower or num > upper:
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": f"value {num} is outside IQR bound [{lower}, {upper}]"
        }

    return None


def check_rare_value_rule(row_idx: int, row: pd.Series, rule: Dict[str, Any]):
    col = rule["column"]
    if col not in row:
        return None

    val = row[col]
    if pd.isna(val):
        return None

    rare_values = set(rule.get("rare_values", []))
    if val in rare_values:
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": f"value is in rare_values set"
        }

    return None


def check_rare_pattern_rule(row_idx: int, row: pd.Series, rule: Dict[str, Any]):
    col = rule["column"]
    if col not in row:
        return None

    val = row[col]
    if pd.isna(val):
        return None

    observed = detect_basic_pattern(val)
    rare_patterns = set(rule.get("rare_patterns", []))
    if observed in rare_patterns:
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": f"pattern {observed} is rare for this column"
        }

    return None


def check_global_dominant_rule(row_idx: int, row: pd.Series, rule: Dict[str, Any]):
    col = rule["column"]
    if col not in row:
        return None

    val = row[col]
    if pd.isna(val):
        return None

    dominant = rule.get("dominant_value")
    if dominant is None:
        return None

    if val != dominant:
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": f"value differs from global dominant value: dominant={dominant}, observed={val}",
            "expected_value": dominant
        }

    return None


def check_context_dominant_rule(row_idx: int, row: pd.Series, rule: Dict[str, Any]):
    ctx_col = rule.get("context_column")
    tgt_col = rule.get("target_column")

    if ctx_col not in row or tgt_col not in row:
        return None

    ctx_val = row[ctx_col]
    tgt_val = row[tgt_col]
    if ctx_val is None or tgt_val is None:
        return None

    mapping = rule.get("_context_mapping", {})
    if ctx_val not in mapping:
        return None

    dominant_val, support, dominant_ratio = mapping[ctx_val]
    if tgt_val != dominant_val:
        return {
            "row_id": row_idx,
            "column": tgt_col,
            "value": tgt_val,
            "rule": rule,
            "reason": f"context dominant mismatch: {ctx_col}={ctx_val} suggests {tgt_col}={dominant_val}, observed={tgt_val}",
            "expected_value": dominant_val,
            "context_value": ctx_val,
            "context_support": support,
            "context_dominant_ratio": dominant_ratio
        }

    return None


def check_high_risk_missing_rule(row_idx: int, row: pd.Series, rule: Dict[str, Any]):
    col = rule.get("column")
    if col not in row:
        return None

    val = row[col]
    if pd.isna(val):
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": "value is null under high_risk_missing rule"
        }
    return None


def check_paired_missing_consistency_rule(row_idx: int, row: pd.Series, rule: Dict[str, Any]):
    col_a = rule.get("column_a")
    col_b = rule.get("column_b")
    if col_a not in row or col_b not in row:
        return None

    a = row[col_a]
    b = row[col_b]

    # 只抓一空一非空
    if (a is None and b is not None) or (a is not None and b is None):
        bad_col = col_a if a is None else col_b
        bad_val = row[bad_col]
        return {
            "row_id": row_idx,
            "column": bad_col,
            "value": bad_val,
            "rule": rule,
            "reason": f"paired missing inconsistency: {col_a}={a}, {col_b}={b}"
        }

    return None


def check_dominant_value_typo_rule(row_idx: int, row: pd.Series, rule: Dict[str, Any]):
    col = rule.get("column")
    if col not in row:
        return None

    val = row[col]
    if pd.isna(val):
        return None

    dominant = rule.get("dominant_value")
    if dominant is None or val == dominant:
        return None

    same_pattern_required = bool(rule.get("require_same_pattern", True))
    if same_pattern_required and detect_basic_pattern(val) != detect_basic_pattern(dominant):
        return None

    max_edit_ratio = float(rule.get("max_edit_ratio", 0.35))
    edit_ratio = normalized_edit_distance_ratio(val, dominant)

    if edit_ratio <= max_edit_ratio:
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": f"value looks like typo of dominant value: dominant={dominant}, observed={val}, edit_ratio={round(edit_ratio, 6)}",
            "expected_value": dominant
        }

    return None


def check_score_format_rule(row_idx: int, row: pd.Series, rule: Dict[str, Any]):
    col = rule.get("column")
    if col not in row:
        return None

    val = row[col]
    if pd.isna(val):
        return None

    # 允许形式：数字%、空、very small text 规则池可自己控制
    pattern = rule.get("regex_pattern", r"^\d+(\.\d+)?%$")
    if re.match(pattern, str(val)) is None:
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": f"value does not match score format regex: {pattern}"
        }

    return None


def check_sample_format_rule(row_idx: int, row: pd.Series, rule: Dict[str, Any]):
    col = rule.get("column")
    if col not in row:
        return None

    val = row[col]
    if pd.isna(val):
        return None

    pattern = rule.get("regex_pattern", r"^\d+\s+patients$")
    if re.match(pattern, str(val)) is None:
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": f"value does not match sample format regex: {pattern}"
        }

    return None


def check_sample_patients_suffix_rule(row_idx: int, row: pd.Series, rule: Dict[str, Any]):
    col = rule.get("column")
    if col not in row:
        return None

    val = row[col]
    if pd.isna(val):
        return None

    suffix = str(rule.get("required_suffix", "patients")).strip().lower()
    sval = str(val).strip().lower()

    if not sval.endswith(suffix):
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": f"value should end with suffix '{suffix}'"
        }

    return None


# ============================================================
# 5. 聚合为候选错误单元格
# ============================================================

def add_violation(cell_map: Dict[Tuple[int, str], Dict[str, Any]], violation: Dict[str, Any]):
    row_id = int(violation["row_id"])
    column = violation["column"]
    key = (row_id, column)

    rule = violation["rule"]

    violated_rule_info = {
        "rule_id": rule.get("rule_id"),
        "rule_type": rule.get("rule_type"),
        "description": rule.get("description"),
        "confidence": rule.get("confidence", rule.get("overall_confidence")),
        "support": rule.get("support"),
        "priority": rule.get("priority"),
        "usage_role": rule.get("usage_role"),
        "reason": violation.get("reason")
    }

    if "expected_value" in violation:
        violated_rule_info["expected_value"] = violation["expected_value"]
    if "lhs_values" in violation:
        violated_rule_info["lhs_values"] = violation["lhs_values"]
    if "fd_group_majority_count" in violation:
        violated_rule_info["fd_group_majority_count"] = violation["fd_group_majority_count"]
    if "fd_group_total_count" in violation:
        violated_rule_info["fd_group_total_count"] = violation["fd_group_total_count"]
    if "context_value" in violation:
        violated_rule_info["context_value"] = violation["context_value"]
    if "context_support" in violation:
        violated_rule_info["context_support"] = violation["context_support"]
    if "context_dominant_ratio" in violation:
        violated_rule_info["context_dominant_ratio"] = violation["context_dominant_ratio"]

    if key not in cell_map:
        cell_map[key] = {
            "row_id": row_id,
            "column": column,
            "value": violation.get("value"),
            "violated_rules": [],
            "conflict_score": 0.0
        }

    cell_map[key]["violated_rules"].append(violated_rule_info)
    cell_map[key]["conflict_score"] += rule_weight(rule)


# ============================================================
# 6. 主流程
# ============================================================

def shrink_search_space(input_csv: str, rule_json: str, output_jsonl: str, output_csv: str):
    df = pd.read_csv(input_csv, dtype=str)
    df = maybe_drop_unnamed_columns(df)

    for col in df.columns:
        df[col] = normalize_series(df[col])

    rules = load_rule_pool(rule_json)

    # 分规则类型
    not_null_rules = [r for r in rules if r.get("rule_type") == "not_null"]
    numeric_range_rules = [r for r in rules if r.get("rule_type") == "numeric_range"]
    enumeration_rules = [r for r in rules if r.get("rule_type") == "enumeration_domain"]
    pattern_rules = [r for r in rules if r.get("rule_type") == "pattern"]

    fd_like_rules = prepare_fd_rule_runtime(df, rules)

    numeric_outlier_rules = [r for r in rules if r.get("rule_type") == "numeric_outlier_iqr"]
    rare_value_rules = [r for r in rules if r.get("rule_type") == "rare_value"]
    rare_pattern_rules = [r for r in rules if r.get("rule_type") == "rare_pattern"]
    global_dominant_rules = [r for r in rules if r.get("rule_type") == "global_dominant_value"]
    context_dominant_rules = prepare_context_dominant_rule_runtime(rules)

    high_risk_missing_rules = [r for r in rules if r.get("rule_type") == "high_risk_missing"]
    paired_missing_rules = [r for r in rules if r.get("rule_type") == "paired_missing_consistency"]
    dominant_typo_rules = [r for r in rules if r.get("rule_type") == "dominant_value_typo"]
    score_format_rules = [r for r in rules if r.get("rule_type") == "score_format"]
    sample_format_rules = [r for r in rules if r.get("rule_type") == "sample_format"]
    sample_suffix_rules = [r for r in rules if r.get("rule_type") == "sample_patients_suffix"]

    cell_map = {}

    # 逐行扫描
    for row_idx, row in df.iterrows():
        # 1) 强规则
        for rule in not_null_rules:
            v = check_not_null_rule(row_idx, row, rule)
            if v is not None:
                add_violation(cell_map, v)

        for rule in numeric_range_rules:
            v = check_numeric_range_rule(row_idx, row, rule)
            if v is not None:
                add_violation(cell_map, v)

        for rule in enumeration_rules:
            v = check_enumeration_rule(row_idx, row, rule)
            if v is not None:
                add_violation(cell_map, v)

        for rule in pattern_rules:
            v = check_pattern_rule(row_idx, row, rule)
            if v is not None:
                add_violation(cell_map, v)

        for rule in fd_like_rules:
            v = check_fd_like_rule(row_idx, row, rule)
            if v is not None:
                add_violation(cell_map, v)

        # 2) 高召回候选规则
        for rule in numeric_outlier_rules:
            v = check_numeric_outlier_rule(row_idx, row, rule)
            if v is not None:
                add_violation(cell_map, v)

        for rule in rare_value_rules:
            v = check_rare_value_rule(row_idx, row, rule)
            if v is not None:
                add_violation(cell_map, v)

        for rule in rare_pattern_rules:
            v = check_rare_pattern_rule(row_idx, row, rule)
            if v is not None:
                add_violation(cell_map, v)

        for rule in global_dominant_rules:
            v = check_global_dominant_rule(row_idx, row, rule)
            if v is not None:
                add_violation(cell_map, v)

        for rule in context_dominant_rules:
            v = check_context_dominant_rule(row_idx, row, rule)
            if v is not None:
                add_violation(cell_map, v)

        # 3) 新增强规则
        for rule in high_risk_missing_rules:
            v = check_high_risk_missing_rule(row_idx, row, rule)
            if v is not None:
                add_violation(cell_map, v)

        for rule in paired_missing_rules:
            v = check_paired_missing_consistency_rule(row_idx, row, rule)
            if v is not None:
                add_violation(cell_map, v)

        for rule in dominant_typo_rules:
            v = check_dominant_value_typo_rule(row_idx, row, rule)
            if v is not None:
                add_violation(cell_map, v)

        for rule in score_format_rules:
            v = check_score_format_rule(row_idx, row, rule)
            if v is not None:
                add_violation(cell_map, v)

        for rule in sample_format_rules:
            v = check_sample_format_rule(row_idx, row, rule)
            if v is not None:
                add_violation(cell_map, v)

        for rule in sample_suffix_rules:
            v = check_sample_patients_suffix_rule(row_idx, row, rule)
            if v is not None:
                add_violation(cell_map, v)

    # 整理输出
    candidates = list(cell_map.values())
    for item in candidates:
        item["conflict_score"] = round(item["conflict_score"], 6)
        item["violation_count"] = len(item["violated_rules"])

    # 按冲突分数和违反规则数排序
    candidates.sort(key=lambda x: (x["conflict_score"], x["violation_count"]), reverse=True)

    # 输出 JSONL
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for item in candidates:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 输出 CSV
    csv_rows = []
    for item in candidates:
        csv_rows.append({
            "row_id": item["row_id"],
            "column": item["column"],
            "value": item["value"],
            "violation_count": item["violation_count"],
            "conflict_score": item["conflict_score"],
            "rule_ids": "|".join([str(r["rule_id"]) for r in item["violated_rules"]]),
            "rule_types": "|".join([str(r["rule_type"]) for r in item["violated_rules"]]),
            "priorities": "|".join([str(r["priority"]) for r in item["violated_rules"]]),
            "usage_roles": "|".join([str(r.get("usage_role")) for r in item["violated_rules"]]),
            "reasons": " || ".join([str(r["reason"]) for r in item["violated_rules"]])
        })

    pd.DataFrame(csv_rows).to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"[OK] Candidate cells saved to: {output_jsonl}")
    print(f"[OK] Candidate cells CSV saved to: {output_csv}")
    print(f"Total candidate cells: {len(candidates)}")

    if len(df) > 0 and len(df.columns) > 0:
        total_cells = len(df) * len(df.columns)
        shrink_ratio = len(candidates) / total_cells
        print(f"Original cells: {total_cells}")
        print(f"Shrink ratio: {shrink_ratio:.6f}")

    print("\nRule counts used:")
    print(f"  not_null: {len(not_null_rules)}")
    print(f"  numeric_range: {len(numeric_range_rules)}")
    print(f"  enumeration_domain: {len(enumeration_rules)}")
    print(f"  pattern: {len(pattern_rules)}")
    print(f"  fd_like(functional_dependency + soft_functional_dependency): {len(fd_like_rules)}")
    print(f"  numeric_outlier_iqr: {len(numeric_outlier_rules)}")
    print(f"  rare_value: {len(rare_value_rules)}")
    print(f"  rare_pattern: {len(rare_pattern_rules)}")
    print(f"  global_dominant_value: {len(global_dominant_rules)}")
    print(f"  dominant_value_by_context: {len(context_dominant_rules)}")
    print(f"  high_risk_missing: {len(high_risk_missing_rules)}")
    print(f"  paired_missing_consistency: {len(paired_missing_rules)}")
    print(f"  dominant_value_typo: {len(dominant_typo_rules)}")
    print(f"  score_format: {len(score_format_rules)}")
    print(f"  sample_format: {len(sample_format_rules)}")
    print(f"  sample_patients_suffix: {len(sample_suffix_rules)}")

    print("\nTop 20 suspicious cells:")
    for item in candidates[:20]:
        print(
            f"row={item['row_id']}, col={item['column']}, value={item['value']}, "
            f"score={item['conflict_score']}, violations={item['violation_count']}"
        )


if __name__ == "__main__":
    shrink_search_space(INPUT_CSV, INPUT_RULE_JSON, OUTPUT_JSONL, OUTPUT_CSV)
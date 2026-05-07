import json
import math
from collections import Counter, defaultdict
from typing import Dict, Any, List, Tuple

import pandas as pd


# ============================================================
# 1. 用户配置区
# ============================================================

INPUT_CSV = "/home/dq/Splittree/hospital/hospital_dirty_no_address23.csv"
INPUT_RULE_JSON = "rule_pool_new.json"

OUTPUT_JSONL = "candidate_cells_new.jsonl"
OUTPUT_CSV = "candidate_cells_new.csv"

MISSING_TOKENS = {"", "empty", "null", "none", "nan", "na", "n/a", "missing", "unk"}

# 是否跳过缺失值不参与某些规则检查
SKIP_NULL_FOR_NON_NOTNULL_RULES = True

# 每条 FD 规则最多缓存多少个 lhs -> majority rhs
MAX_FD_MAPPING_SIZE = 200000

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
    "functional_dependency": 1.5
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
    if value is None:
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
    for group_name, rules in rule_groups.items():
        for rule in rules:
            r = dict(rule)
            r["_group_name"] = group_name
            all_rules.append(r)

    return all_rules


def rule_weight(rule: Dict[str, Any]) -> float:
    conf = float(rule.get("confidence", 0.0))
    supp = int(rule.get("support", 0))
    priority = rule.get("priority", "low")
    rule_type = rule.get("rule_type", "")

    priority_w = PRIORITY_WEIGHT.get(priority, 0.6)
    type_w = RULE_TYPE_WEIGHT.get(rule_type, 1.0)

    # 支持度做对数压缩，避免特别大时过度主导
    support_part = math.log1p(max(supp, 0))

    # 置信度作为主因子
    score = (0.1 + conf) * support_part * priority_w * type_w
    return round(score, 6)


# ============================================================
# 3. 为 FD 规则预构建多数派映射
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
    给 FD 规则补充运行期字段:
        _majority_mapping
    """
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
# 4. 单元格违规检测
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
            "reason": f"value is null but rule requires non-null"
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
            "reason": f"value cannot be parsed as numeric for numeric_range rule"
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
    if val is None and SKIP_NULL_FOR_NON_NOTNULL_RULES:
        return None

    allowed = set(rule.get("allowed_values", []))
    if val not in allowed:
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": f"value not in allowed domain"
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
            "reason": f"pattern mismatch: observed={observed}, expected={target}"
        }

    return None


def check_fd_rule(row_idx: int, row: pd.Series, rule: Dict[str, Any]):
    lhs = rule["lhs"]
    rhs = rule["rhs"]

    # lhs 或 rhs 缺列直接跳过
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
        return None  # 这类 lhs 组合在数据中没形成足够群体，跳过

    expected_rhs, majority_cnt, total_cnt = mapping[lhs_key]

    if rhs_val != expected_rhs:
        return {
            "row_id": row_idx,
            "column": rhs,
            "value": rhs_val,
            "rule": rule,
            "reason": f"FD violation: lhs={lhs_key} suggests rhs={expected_rhs}, observed={rhs_val}",
            "expected_value": expected_rhs,
            "lhs_values": {c: row[c] for c in lhs},
            "fd_group_majority_count": majority_cnt,
            "fd_group_total_count": total_cnt
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
        "confidence": rule.get("confidence"),
        "support": rule.get("support"),
        "priority": rule.get("priority"),
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
    fd_rules = prepare_fd_rule_runtime(df, rules)

    cell_map = {}

    # 逐行扫描
    for row_idx, row in df.iterrows():
        # 1) 非空规则
        for rule in not_null_rules:
            v = check_not_null_rule(row_idx, row, rule)
            if v is not None:
                add_violation(cell_map, v)

        # 2) 数值范围规则
        for rule in numeric_range_rules:
            v = check_numeric_range_rule(row_idx, row, rule)
            if v is not None:
                add_violation(cell_map, v)

        # 3) 枚举域规则
        for rule in enumeration_rules:
            v = check_enumeration_rule(row_idx, row, rule)
            if v is not None:
                add_violation(cell_map, v)

        # 4) pattern 规则
        for rule in pattern_rules:
            v = check_pattern_rule(row_idx, row, rule)
            if v is not None:
                add_violation(cell_map, v)

        # 5) FD 规则
        for rule in fd_rules:
            v = check_fd_rule(row_idx, row, rule)
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

    # 输出 CSV（便于快速查看）
    csv_rows = []
    for item in candidates:
        csv_rows.append({
            "row_id": item["row_id"],
            "column": item["column"],
            "value": item["value"],
            "violation_count": item["violation_count"],
            "conflict_score": item["conflict_score"],
            "rule_ids": "|".join([r["rule_id"] for r in item["violated_rules"]]),
            "rule_types": "|".join([r["rule_type"] for r in item["violated_rules"]]),
            "priorities": "|".join([str(r["priority"]) for r in item["violated_rules"]]),
            "reasons": " || ".join([r["reason"] for r in item["violated_rules"]])
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

    # 打印前 20 个高风险候选
    print("\nTop 20 suspicious cells:")
    for item in candidates[:20]:
        print(
            f"row={item['row_id']}, col={item['column']}, value={item['value']}, "
            f"score={item['conflict_score']}, violations={item['violation_count']}"
        )


if __name__ == "__main__":
    shrink_search_space(INPUT_CSV, INPUT_RULE_JSON, OUTPUT_JSONL, OUTPUT_CSV)
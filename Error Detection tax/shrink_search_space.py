import json
import math
import re
from collections import Counter
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd


# ============================================================
# 1. 用户配置区
# ============================================================

INPUT_CSV = "/mnt/mydata/dq/projects/Splittree/tax/tax_dirty.csv"
INPUT_RULE_JSON = "rule_pool_tax.json"

OUTPUT_JSONL = "candidate_cells_tax.jsonl"
OUTPUT_CSV = "candidate_cells_tax.csv"

MISSING_TOKENS = {
    "", "empty", "null", "none", "nan", "na", "n/a", "missing", "unk",
    "unknown", "not available", "contact airline"
}

SKIP_NULL_FOR_NON_NOTNULL_RULES = True
MAX_FD_MAPPING_SIZE = 200000
MAX_CONTEXT_MAPPING_SIZE = 200000

PRIORITY_WEIGHT = {"high": 1.5, "medium": 1.0, "low": 0.6}

RULE_TYPE_WEIGHT = {
    "not_null": 1.2,
    "pattern": 0.7,
    "functional_dependency": 1.5,
    "soft_functional_dependency": 1.1,
    "rare_value": 0.7,
    "rare_pattern": 0.7,
    "global_dominant_value": 0.6,
    "dominant_value_by_context": 0.9,
    "dominant_value_by_context_pair": 1.0,
    "high_risk_missing": 0.9,
    "paired_missing_consistency": 0.8,
    "person_name_format": 0.75,
    "gender_domain": 1.25,
    "area_code_format": 1.15,
    "phone_format": 1.15,
    "phone_canonical_format": 1.0,
    "city_format": 0.75,
    "state_code_domain": 1.25,
    "zip_format": 1.15,
    "zip_canonical_format": 1.0,
    "marital_status_domain": 1.25,
    "has_child_domain": 1.25,
    "salary_numeric_range": 1.05,
    "rate_numeric_range": 1.05,
    "exemption_numeric_range": 1.05,
    "zip_state_consistency": 1.1,
    "zip_city_consistency": 1.0,
    "zip_area_code_consistency": 0.9,
    "marital_exemption_consistency": 1.25,
    "child_exemption_consistency": 1.25,
    "salary_rate_context_consistency": 0.9,
    "tax_profile_rate_consistency": 1.0,
}

VALID_GENDERS = {"m", "f"}
VALID_MARITAL_STATUS = {"s", "m"}
VALID_HAS_CHILD = {"y", "n"}

STATE_CODES = {
    "al", "ak", "az", "ar", "ca", "co", "ct", "dc", "de", "fl", "ga",
    "hi", "ia", "id", "il", "in", "ks", "ky", "la", "ma", "md", "me",
    "mi", "mn", "mo", "ms", "mt", "nc", "nd", "ne", "nh", "nj", "nm",
    "nv", "ny", "oh", "ok", "or", "pa", "ri", "sc", "sd", "tn", "tx",
    "ut", "va", "vt", "wa", "wi", "wv", "wy"
}

HIGH_CARD_TEXT_COLUMNS = {"f_name", "l_name", "city", "phone"}
WEAK_LONGTAIL_RULES = {"rare_value", "rare_pattern", "global_dominant_value"}
ENABLE_PRECISION_GUARD = True


# ============================================================
# 2. 基础函数
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
    if pd.isna(value) or value is None:
        return "NULL"
    parts, prev_type, cnt = [], None, 0
    for ch in str(value):
        if ch.isdigit():
            cur = "D"
        elif ch.isalpha():
            cur = "L"
        elif ch in {" ", "-", "_", "/", ".", ",", "(", ")", "%", ":", ";", "&", "'", "+", "#", "$"}:
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


def normalize_phone(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    txt = str(s).strip().lower()
    digits = re.sub(r"\D", "", txt)
    if len(digits) == 7:
        return f"{digits[:3]}-{digits[3:]}"
    return None


def normalize_zip(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    txt = str(s).strip()
    m = re.search(r"(\d{5})", txt)
    if m:
        return m.group(1)
    return None


def normalize_area_code(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    txt = str(s).strip()
    m = re.fullmatch(r"\d{3}", txt)
    if m:
        return txt
    m2 = re.search(r"(\d{3})", txt)
    if m2:
        return m2.group(1)
    return None


def load_rule_pool(rule_json_path):
    with open(rule_json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    all_rules = []
    for _, rules in obj.get("rules", {}).items():
        for rule in rules:
            all_rules.append(dict(rule))
    return all_rules


def should_drop_rule_by_precision_guard(rule: Dict[str, Any]) -> bool:
    if not ENABLE_PRECISION_GUARD:
        return False
    rtype = rule.get("rule_type", "")
    if rtype in WEAK_LONGTAIL_RULES:
        col = rule.get("column")
        if col in HIGH_CARD_TEXT_COLUMNS:
            return True
    if rtype in {"functional_dependency", "soft_functional_dependency", "dominant_value_by_context", "dominant_value_by_context_pair"}:
        rhs = rule.get("rhs")
        if rhs in {"f_name", "l_name"}:
            return True
    return False


def filter_rules_with_precision_guard(rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not ENABLE_PRECISION_GUARD:
        return rules
    kept, removed = [], 0
    for r in rules:
        if should_drop_rule_by_precision_guard(r):
            removed += 1
            continue
        kept.append(r)
    print(f"[PrecisionGuard] rules before={len(rules)}, after={len(kept)}, removed={removed}")
    return kept


def rule_weight(rule):
    conf = float(rule.get("confidence", 0.0))
    supp = int(rule.get("support", 0))
    priority = rule.get("priority", "low")
    rule_type = rule.get("rule_type", "")
    priority_w = PRIORITY_WEIGHT.get(priority, 0.6)
    type_w = RULE_TYPE_WEIGHT.get(rule_type, 1.0)
    support_part = math.log1p(max(supp, 1))
    return round((0.1 + conf) * support_part * priority_w * type_w, 6)


# ============================================================
# 3. runtime mapping
# ============================================================

def build_fd_majority_mapping(df, lhs, rhs):
    sub = df[lhs + [rhs]].dropna()
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


def prepare_fd_rule_runtime(df, rules):
    out = []
    for rule in rules:
        if rule.get("rule_type") not in {"functional_dependency", "soft_functional_dependency"}:
            continue
        r = dict(rule)
        r["_majority_mapping"] = build_fd_majority_mapping(df, rule["lhs"], rule["rhs"])
        out.append(r)
    return out


def prepare_context_dominant_rule_runtime(rules):
    out = []
    for rule in rules:
        if rule.get("rule_type") != "dominant_value_by_context":
            continue
        mapping = {}
        for ctx_val, info in rule.get("majority_map", {}).items():
            mapping[ctx_val] = (
                info.get("dominant_value"),
                int(info.get("group_size", 0)),
                float(info.get("ratio", 0.0)),
            )
        r = dict(rule)
        r["_context_mapping"] = mapping
        out.append(r)
    return out


def prepare_pair_context_rule_runtime(rules):
    out = []
    for rule in rules:
        if rule.get("rule_type") != "dominant_value_by_context_pair":
            continue
        mapping = {}
        for key, info in rule.get("majority_map", {}).items():
            mapping[key] = (
                info.get("dominant_value"),
                int(info.get("group_size", 0)),
                float(info.get("ratio", 0.0)),
            )
        r = dict(rule)
        r["_pair_context_mapping"] = mapping
        out.append(r)
    return out


def build_context_majority_mapping_from_df(df, lhs_cols: List[str], rhs_col: str, min_group_size: int = 2, min_ratio: float = 0.70):
    sub = df[lhs_cols + [rhs_col]].dropna()
    if len(sub) == 0:
        return {}
    grouped = sub.groupby(lhs_cols, dropna=True)[rhs_col]
    mapping = {}
    for key, grp in grouped:
        if len(grp) < min_group_size:
            continue
        cnt = Counter(grp.tolist())
        maj_val, maj_cnt = cnt.most_common(1)[0]
        ratio = maj_cnt / len(grp)
        if ratio >= min_ratio:
            if not isinstance(key, tuple):
                key = (key,)
            mapping[tuple(key)] = (maj_val, maj_cnt, len(grp), ratio)
        if len(mapping) >= MAX_CONTEXT_MAPPING_SIZE:
            break
    return mapping


def prepare_tax_consistency_rule_runtime(df, rules):
    out = []
    for rule in rules:
        rtype = rule.get("rule_type")
        if rtype == "zip_state_consistency":
            r = dict(rule)
            r["_majority_mapping"] = build_context_majority_mapping_from_df(df, ["zip"], "state", 2, 0.70)
            out.append(r)
        elif rtype == "zip_city_consistency":
            r = dict(rule)
            r["_majority_mapping"] = build_context_majority_mapping_from_df(df, ["zip"], "city", 2, 0.70)
            out.append(r)
        elif rtype == "zip_area_code_consistency":
            r = dict(rule)
            r["_majority_mapping"] = build_context_majority_mapping_from_df(df, ["zip"], "area_code", 2, 0.70)
            out.append(r)
        elif rtype == "salary_rate_context_consistency":
            r = dict(rule)
            r["_majority_mapping"] = build_context_majority_mapping_from_df(df, ["salary"], "rate", 2, 0.70)
            out.append(r)
        elif rtype == "tax_profile_rate_consistency":
            r = dict(rule)
            r["_majority_mapping"] = build_context_majority_mapping_from_df(
                df, ["marital_status", "has_child", "salary"], "rate", 2, 0.70
            )
            out.append(r)
    return out


# ============================================================
# 4. add violation
# ============================================================

def add_violation(cell_map, violation):
    row_id = int(violation["row_id"])
    column = violation["column"]
    key = (row_id, column)
    rule = violation["rule"]
    info = {
        "rule_id": rule.get("rule_id"),
        "rule_type": rule.get("rule_type"),
        "description": rule.get("description"),
        "confidence": rule.get("confidence"),
        "support": rule.get("support"),
        "priority": rule.get("priority"),
        "usage_role": rule.get("usage_role"),
        "reason": violation.get("reason"),
    }
    if "expected_value" in violation:
        info["expected_value"] = violation["expected_value"]
    if key not in cell_map:
        cell_map[key] = {
            "row_id": row_id,
            "column": column,
            "value": violation.get("value"),
            "violated_rules": [],
            "conflict_score": 0.0,
        }
    cell_map[key]["violated_rules"].append(info)
    cell_map[key]["conflict_score"] += rule_weight(rule)


# ============================================================
# 5. 通用规则检查
# ============================================================

def check_not_null_rule(row_idx, row, rule):
    col = rule["column"]
    val = row.get(col)
    if pd.isna(val) or val is None:
        return {"row_id": row_idx, "column": col, "value": val, "rule": rule, "reason": "value is null but rule requires non-null"}
    return None


def check_pattern_rule(row_idx, row, rule):
    col = rule["column"]
    val = row.get(col)
    if (pd.isna(val) or val is None) and SKIP_NULL_FOR_NON_NOTNULL_RULES:
        return None
    observed = detect_basic_pattern(val)
    target = rule["pattern"]
    if observed != target:
        return {"row_id": row_idx, "column": col, "value": val, "rule": rule, "reason": f"pattern mismatch: observed={observed}, expected={target}"}
    return None


def check_fd_like_rule(row_idx, row, rule):
    lhs, rhs = rule["lhs"], rule["rhs"]
    lhs_values = []
    for c in lhs:
        v = row.get(c)
        if v is None:
            return None
        lhs_values.append(v)
    rhs_val = row.get(rhs)
    if rhs_val is None and SKIP_NULL_FOR_NON_NOTNULL_RULES:
        return None
    lhs_key = tuple(lhs_values)
    mapping = rule.get("_majority_mapping", {})
    if lhs_key not in mapping:
        return None
    expected_rhs, majority_cnt, total_cnt = mapping[lhs_key]
    if rhs_val != expected_rhs:
        return {"row_id": row_idx, "column": rhs, "value": rhs_val, "rule": rule, "reason": f"{rule['rule_type']} violation: lhs={lhs_key} suggests rhs={expected_rhs}, observed={rhs_val}", "expected_value": expected_rhs}
    return None


def check_rare_value_rule(row_idx, row, rule):
    col = rule["column"]
    val = row.get(col)
    if pd.isna(val) or val is None:
        return None
    if val in set(rule.get("rare_values", [])):
        return {"row_id": row_idx, "column": col, "value": val, "rule": rule, "reason": "value is in rare_values set"}
    return None


def check_rare_pattern_rule(row_idx, row, rule):
    col = rule["column"]
    val = row.get(col)
    if pd.isna(val) or val is None:
        return None
    observed = detect_basic_pattern(val)
    if observed in set(rule.get("rare_patterns", [])):
        return {"row_id": row_idx, "column": col, "value": val, "rule": rule, "reason": f"pattern {observed} is rare for this column"}
    return None


def check_global_dominant_rule(row_idx, row, rule):
    col = rule["column"]
    val = row.get(col)
    if pd.isna(val) or val is None:
        return None
    dominant = rule.get("dominant_value")
    if dominant is not None and val != dominant:
        return {"row_id": row_idx, "column": col, "value": val, "rule": rule, "reason": f"value differs from global dominant value: dominant={dominant}, observed={val}", "expected_value": dominant}
    return None


def check_context_dominant_rule(row_idx, row, rule):
    lhs = rule.get("lhs", [])
    rhs = rule.get("rhs")
    if len(lhs) != 1 or rhs is None:
        return None
    ctx_col = lhs[0]
    ctx_val = row.get(ctx_col)
    tgt_val = row.get(rhs)
    if ctx_val is None or tgt_val is None:
        return None
    mapping = rule.get("_context_mapping", {})
    if ctx_val not in mapping:
        return None
    dominant_val, support, dominant_ratio = mapping[ctx_val]
    if tgt_val != dominant_val:
        return {"row_id": row_idx, "column": rhs, "value": tgt_val, "rule": rule, "reason": f"context dominant mismatch: {ctx_col}={ctx_val} suggests {rhs}={dominant_val}, observed={tgt_val}", "expected_value": dominant_val}
    return None


def check_pair_context_rule(row_idx, row, rule):
    lhs = rule.get("lhs", [])
    rhs = rule.get("rhs")
    if rhs is None or len(lhs) < 2:
        return None
    vals = []
    for c in lhs:
        v = row.get(c)
        if v is None:
            return None
        vals.append(v)
    tgt_val = row.get(rhs)
    if tgt_val is None:
        return None
    key = "||".join(map(str, vals))
    mapping = rule.get("_pair_context_mapping", {})
    if key not in mapping:
        return None
    dominant_val, support, dominant_ratio = mapping[key]
    if tgt_val != dominant_val:
        return {"row_id": row_idx, "column": rhs, "value": tgt_val, "rule": rule, "reason": f"context pair dominant mismatch: {lhs}={vals} suggests {rhs}={dominant_val}, observed={tgt_val}", "expected_value": dominant_val}
    return None


def check_high_risk_missing_rule(row_idx, row, rule):
    col = rule["column"]
    val = row.get(col)
    if pd.isna(val) or val is None:
        return {"row_id": row_idx, "column": col, "value": val, "rule": rule, "reason": "value is null under high_risk_missing rule"}
    return None


def check_paired_missing_consistency_rule(row_idx, row, rule):
    cols = rule.get("columns", [])
    if len(cols) != 2:
        return None
    a_col, b_col = cols
    a, b = row.get(a_col), row.get(b_col)
    if (a is None and b is not None) or (a is not None and b is None):
        bad_col = a_col if a is None else b_col
        return {"row_id": row_idx, "column": bad_col, "value": row.get(bad_col), "rule": rule, "reason": f"paired missing inconsistency: {a_col}={a}, {b_col}={b}"}
    return None


def check_regex_rule(row_idx, row, rule):
    col = rule["column"]
    val = row.get(col)
    if pd.isna(val) or val is None:
        return None
    pattern = rule.get("regex")
    if pattern and re.match(pattern, str(val)) is None:
        return {"row_id": row_idx, "column": col, "value": val, "rule": rule, "reason": f"value does not match regex: {pattern}"}
    return None


# ============================================================
# 6. tax schema/domain/format checks
# ============================================================

def check_domain_rule(row_idx, row, rule):
    col = rule["column"]
    val = row.get(col)
    if val is None:
        return None
    valid_values = set(rule.get("valid_values", []))
    if val not in valid_values:
        return {"row_id": row_idx, "column": col, "value": val, "rule": rule, "reason": f"value is outside domain: observed={val}, valid={sorted(list(valid_values))}"}
    return None


def check_phone_canonical_rule(row_idx, row, rule):
    col = rule["column"]
    val = row.get(col)
    if val is None:
        return None
    canonical = normalize_phone(val)
    if canonical is None:
        return {"row_id": row_idx, "column": col, "value": val, "rule": rule, "reason": f"cannot canonicalize phone as xxx-xxxx: {val}"}
    if str(val).strip().lower() != canonical:
        return {"row_id": row_idx, "column": col, "value": val, "rule": rule, "reason": f"phone canonical mismatch: canonical={canonical}, observed={val}", "expected_value": canonical}
    return None


def check_zip_canonical_rule(row_idx, row, rule):
    col = rule["column"]
    val = row.get(col)
    if val is None:
        return None
    canonical = normalize_zip(val)
    if canonical is None:
        return {"row_id": row_idx, "column": col, "value": val, "rule": rule, "reason": f"cannot canonicalize zip to 5 digits: {val}"}
    if str(val).strip().lower() != canonical:
        return {"row_id": row_idx, "column": col, "value": val, "rule": rule, "reason": f"zip canonical mismatch: canonical={canonical}, observed={val}", "expected_value": canonical}
    return None


def check_numeric_range_rule(row_idx, row, rule):
    col = rule["column"]
    val = row.get(col)
    if val is None:
        return None
    num = parse_float_like(val)
    if num is None:
        return {"row_id": row_idx, "column": col, "value": val, "rule": rule, "reason": f"value is not numeric: {val}"}
    lb = rule.get("lower_bound", None)
    ub = rule.get("upper_bound", None)
    if lb is not None and num < float(lb):
        return {"row_id": row_idx, "column": col, "value": val, "rule": rule, "reason": f"numeric value below lower bound: {num} < {lb}"}
    if ub is not None and num > float(ub):
        return {"row_id": row_idx, "column": col, "value": val, "rule": rule, "reason": f"numeric value above upper bound: {num} > {ub}"}
    return None


def check_area_code_format_rule(row_idx, row, rule):
    col = rule["column"]
    val = row.get(col)
    if val is None:
        return None
    canonical = normalize_area_code(val)
    if canonical is None or not re.fullmatch(r"\d{3}", canonical):
        return {"row_id": row_idx, "column": col, "value": val, "rule": rule, "reason": f"area_code is not a 3-digit code: {val}"}
    if str(val).strip() != canonical:
        return {"row_id": row_idx, "column": col, "value": val, "rule": rule, "reason": f"area_code canonical mismatch: canonical={canonical}, observed={val}", "expected_value": canonical}
    return None


# ============================================================
# 7. tax consistency checks
# ============================================================

def check_tax_context_mapping_rule(row_idx, row, rule):
    lhs = rule.get("lhs", [])
    rhs = rule.get("rhs")
    if not lhs or rhs is None:
        return None
    lhs_values = []
    for c in lhs:
        v = row.get(c)
        if v is None:
            return None
        lhs_values.append(v)
    rhs_val = row.get(rhs)
    if rhs_val is None:
        return None
    key = tuple(lhs_values)
    mapping = rule.get("_majority_mapping", {})
    if key not in mapping:
        return None
    expected, maj_cnt, total_cnt, ratio = mapping[key]
    if rhs_val != expected:
        return {"row_id": row_idx, "column": rhs, "value": rhs_val, "rule": rule, "reason": f"{rule['rule_type']} violation: lhs={key} suggests {rhs}={expected}, observed={rhs_val}, ratio={ratio:.3f}", "expected_value": expected}
    return None


def check_marital_exemption_consistency_rule(row_idx, row, rule):
    ms = row.get("marital_status")
    single = row.get("single_exemp")
    married = row.get("married_exemp")
    if ms is None:
        return None
    single_i = parse_int_like(single)
    married_i = parse_int_like(married)
    if single is not None and single_i is None:
        return {"row_id": row_idx, "column": "single_exemp", "value": single, "rule": rule, "reason": f"single_exemp is not numeric-like: {single}"}
    if married is not None and married_i is None:
        return {"row_id": row_idx, "column": "married_exemp", "value": married, "rule": rule, "reason": f"married_exemp is not numeric-like: {married}"}
    if ms == "s" and married_i is not None and married_i != 0:
        return {"row_id": row_idx, "column": "married_exemp", "value": married, "rule": rule, "reason": f"marital_status=S but married_exemp={married_i}, expected 0", "expected_value": "0"}
    if ms == "m" and single_i is not None and single_i != 0:
        return {"row_id": row_idx, "column": "single_exemp", "value": single, "rule": rule, "reason": f"marital_status=M but single_exemp={single_i}, expected 0", "expected_value": "0"}
    return None


def check_child_exemption_consistency_rule(row_idx, row, rule):
    hc = row.get("has_child")
    child = row.get("child_exemp")
    if hc is None or child is None:
        return None
    child_i = parse_int_like(child)
    if child_i is None:
        return {"row_id": row_idx, "column": "child_exemp", "value": child, "rule": rule, "reason": f"child_exemp is not numeric-like: {child}"}
    if hc == "n" and child_i != 0:
        return {"row_id": row_idx, "column": "child_exemp", "value": child, "rule": rule, "reason": f"has_child=N but child_exemp={child_i}, expected 0", "expected_value": "0"}
    return None


# ============================================================
# 8. 主流程
# ============================================================

def shrink_search_space(input_csv, rule_json, output_jsonl, output_csv):
    df = pd.read_csv(input_csv, dtype=str)
    df = maybe_drop_unnamed_columns(df)
    for col in df.columns:
        df[col] = normalize_series(df[col])
    rules = load_rule_pool(rule_json)
    rules = filter_rules_with_precision_guard(rules)

    not_null_rules = [r for r in rules if r.get("rule_type") == "not_null"]
    pattern_rules = [r for r in rules if r.get("rule_type") == "pattern"]
    fd_like_rules = prepare_fd_rule_runtime(df, rules)
    rare_value_rules = [r for r in rules if r.get("rule_type") == "rare_value"]
    rare_pattern_rules = [r for r in rules if r.get("rule_type") == "rare_pattern"]
    global_dominant_rules = [r for r in rules if r.get("rule_type") == "global_dominant_value"]
    context_dominant_rules = prepare_context_dominant_rule_runtime(rules)
    pair_context_rules = prepare_pair_context_rule_runtime(rules)
    high_risk_missing_rules = [r for r in rules if r.get("rule_type") == "high_risk_missing"]
    paired_missing_rules = [r for r in rules if r.get("rule_type") == "paired_missing_consistency"]
    regex_rules = [r for r in rules if r.get("rule_type") in {"person_name_format", "phone_format", "city_format", "zip_format"}]
    domain_rules = [r for r in rules if r.get("rule_type") in {"gender_domain", "state_code_domain", "marital_status_domain", "has_child_domain"}]
    area_code_rules = [r for r in rules if r.get("rule_type") == "area_code_format"]
    phone_canonical_rules = [r for r in rules if r.get("rule_type") == "phone_canonical_format"]
    zip_canonical_rules = [r for r in rules if r.get("rule_type") == "zip_canonical_format"]
    numeric_range_rules = [r for r in rules if r.get("rule_type") in {"salary_numeric_range", "rate_numeric_range", "exemption_numeric_range"}]
    marital_exemption_rules = [r for r in rules if r.get("rule_type") == "marital_exemption_consistency"]
    child_exemption_rules = [r for r in rules if r.get("rule_type") == "child_exemption_consistency"]
    tax_context_rules = prepare_tax_consistency_rule_runtime(df, rules)

    cell_map = {}
    for row_idx, row in df.iterrows():
        for group, fn in [
            (not_null_rules, check_not_null_rule),
            (pattern_rules, check_pattern_rule),
            (fd_like_rules, check_fd_like_rule),
            (rare_value_rules, check_rare_value_rule),
            (rare_pattern_rules, check_rare_pattern_rule),
            (global_dominant_rules, check_global_dominant_rule),
            (context_dominant_rules, check_context_dominant_rule),
            (pair_context_rules, check_pair_context_rule),
            (high_risk_missing_rules, check_high_risk_missing_rule),
            (paired_missing_rules, check_paired_missing_consistency_rule),
            (regex_rules, check_regex_rule),
            (domain_rules, check_domain_rule),
            (area_code_rules, check_area_code_format_rule),
            (phone_canonical_rules, check_phone_canonical_rule),
            (zip_canonical_rules, check_zip_canonical_rule),
            (numeric_range_rules, check_numeric_range_rule),
            (tax_context_rules, check_tax_context_mapping_rule),
            (marital_exemption_rules, check_marital_exemption_consistency_rule),
            (child_exemption_rules, check_child_exemption_consistency_rule),
        ]:
            for rule in group:
                v = fn(row_idx, row, rule)
                if v is not None:
                    add_violation(cell_map, v)

    candidates = list(cell_map.values())
    for item in candidates:
        item["conflict_score"] = round(item["conflict_score"], 6)
        item["violation_count"] = len(item["violated_rules"])
    candidates.sort(key=lambda x: (x["conflict_score"], x["violation_count"]), reverse=True)

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for item in candidates:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

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
            "reasons": " || ".join([str(r["reason"]) for r in item["violated_rules"]]),
        })
    pd.DataFrame(csv_rows).to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"[OK] Candidate cells saved to: {output_jsonl}")
    print(f"[OK] Candidate cells CSV saved to: {output_csv}")
    print(f"Total candidate cells: {len(candidates)}")
    total_cells = len(df) * len(df.columns)
    print(f"Original cells: {total_cells}")
    print(f"Shrink ratio: {len(candidates) / total_cells:.6f}")


if __name__ == "__main__":
    shrink_search_space(INPUT_CSV, INPUT_RULE_JSON, OUTPUT_JSONL, OUTPUT_CSV)

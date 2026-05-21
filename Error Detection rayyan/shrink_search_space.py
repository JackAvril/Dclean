import json
import math
import re
from collections import Counter
from typing import Dict, Any, List, Optional

import pandas as pd

INPUT_CSV = "/mnt/mydata/dq/projects/Splittree/rayyan/rayyan_dirty.csv"
INPUT_RULE_JSON = "rule_pool_rayyan.json"

OUTPUT_JSONL = "candidate_cells_rayyan.jsonl"
OUTPUT_CSV = "candidate_cells_rayyan.csv"

MISSING_TOKENS = {
    "", "empty", "null", "none", "nan", "na", "n/a", "missing", "unk",
    "not available", "contact airline"
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
    "rare_value": 0.35,
    "rare_pattern": 0.45,
    "global_dominant_value": 0.5,
    "dominant_value_by_context": 0.9,
    "dominant_value_by_context_pair": 1.0,
    "high_risk_missing": 0.9,
    "paired_missing_consistency": 0.8,

    "issn_format": 0.95,
    "issn_loose_format": 0.85,
    "issn_canonical_format": 1.10,
    "language_canonical_format": 1.00,
    "date_like_format": 0.95,
    "date_component_permutation_suspicious": 1.35,
    "date_ymd_swap_suspicious": 1.65,
    "volume_issue_numeric_format": 0.90,
    "pagination_format": 0.55,
    "author_list_braced_format": 0.80,
    "author_list_encoding_corruption": 1.40,
}


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


def load_rule_pool(rule_json_path):
    with open(rule_json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    all_rules = []
    for _, rules in obj.get("rules", {}).items():
        for rule in rules:
            all_rules.append(dict(rule))
    return all_rules


def rule_weight(rule):
    conf = float(rule.get("confidence", 0.0))
    supp = int(rule.get("support", 0))
    priority = rule.get("priority", "low")
    rule_type = rule.get("rule_type", "")
    priority_w = PRIORITY_WEIGHT.get(priority, 0.6)
    type_w = RULE_TYPE_WEIGHT.get(rule_type, 1.0)
    support_part = math.log1p(max(supp, 1))
    return round((0.1 + conf) * support_part * priority_w * type_w, 6)


def canonicalize_language(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    txt = str(s).strip().lower()
    mapping = {
        "eng": "eng",
        "english": "eng",
        "english; abstract language:english": "eng",
        "en": "eng",
        "pol": "pol",
        "polish": "pol",
        "fre": "fre",
        "french": "fre",
        "ger": "ger",
        "german": "ger",
        "spa": "spa",
        "spanish": "spa",
    }
    return mapping.get(txt)


def extract_issn_digits(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    m = re.search(r'(\d{4})[- ]?(\d{3}[\dx])', str(s).strip().lower())
    if not m:
        return None
    return f"{m.group(1)}-{m.group(2)}"


def is_normal_journal_issn_variant(s: Optional[str]) -> bool:
    """
    放行一些当前数据里很常见、但之前容易被误伤的正常 ISSN 形式。
    例如：
      0035-9157 (Print) 0035-9157
      1537-6613
      1537-6613 (Print)
    """
    if s is None:
        return False
    txt = str(s).strip()
    txt_low = txt.lower()

    if re.fullmatch(r'\d{4}-\d{3}[\dx]', txt_low):
        return True
    if re.fullmatch(r'\d{4}-\d{3}[\dx]\s*\(print\)', txt_low):
        return True
    if re.fullmatch(r'\d{4}-\d{3}[\dx]\s*\(print\)\s*\d{4}-\d{3}[\dx]', txt_low):
        return True
    return False


def normalize_date_like(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    txt = str(s).strip().lower()
    txt = re.sub(r'\s+', '', txt)

    m = re.match(r'^(\d{1,2})/(\d{1,2})/(\d{2,4})$', txt)
    if m:
        mm, dd, yy = int(m.group(1)), int(m.group(2)), m.group(3)
        yy_i = int(yy)
        if len(yy) == 2:
            yy_i = 1900 + yy_i if yy_i >= 30 else 2000 + yy_i
        if 1 <= mm <= 12 and 1 <= dd <= 31:
            return f"{yy_i:04d}-{mm:02d}-{dd:02d}"

    m = re.match(r'^(\d{4})-(\d{1,2})-(\d{1,2})$', txt)
    if m:
        yy_i, mm, dd = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mm <= 12 and 1 <= dd <= 31:
            return f"{yy_i:04d}-{mm:02d}-{dd:02d}"
    return None


def split_date_components(s: Optional[str]):
    if s is None:
        return None
    txt = str(s).strip().lower()
    txt = re.sub(r'\s+', '', txt)
    m = re.match(r'^(\d{1,2})/(\d{1,2})/(\d{2,4})$', txt)
    if not m:
        return None
    a, b, c = m.group(1), m.group(2), m.group(3)
    return int(a), int(b), int(c)


def generate_date_permutations(s: Optional[str]):
    comp = split_date_components(s)
    if comp is None:
        return []
    a, b, c = comp
    perms = []
    candidates = [
        (1, 3, 2),
        (2, 3, 1),
        (2, 1, 3),
    ]
    arr = [str(a), str(b), str(c)]
    for idxs in candidates:
        mm_raw, dd_raw, yy_raw = arr[idxs[0] - 1], arr[idxs[1] - 1], arr[idxs[2] - 1]
        try:
            mm = int(mm_raw)
            dd = int(dd_raw)
            yy_i = int(yy_raw)
        except Exception:
            continue
        if len(yy_raw) == 2:
            yy_i = 1900 + yy_i if yy_i >= 30 else 2000 + yy_i
        if 1 <= mm <= 12 and 1 <= dd <= 31:
            perms.append(f"{yy_i:04d}-{mm:02d}-{dd:02d}")
    return list(dict.fromkeys(perms))


def generate_ymd_swap_candidate(s: Optional[str]) -> Optional[str]:
    """
    针对最常见的错位模式：
    a/b/c -> b/c/a
    例如：
      4/2/15 -> 2/15/04
      1/1/13 -> 1/13/01
      3/1/14 -> 1/14/03
    """
    comp = split_date_components(s)
    if comp is None:
        return None

    a, b, c = comp
    a_str, b_str, c_str = str(a), str(b), str(c)
    try:
        mm = int(b_str)
        dd = int(c_str)
        yy_i = int(a_str)
    except Exception:
        return None

    if len(a_str) <= 2:
        yy_i = 1900 + yy_i if yy_i >= 30 else 2000 + yy_i

    if 1 <= mm <= 12 and 1 <= dd <= 31:
        return f"{yy_i:04d}-{mm:02d}-{dd:02d}"
    return None


def canonicalize_volume_issue(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    txt = str(s).strip().lower()
    m = re.search(r'(\d+)', txt)
    if not m:
        return None
    return str(int(m.group(1)))


def looks_like_author_list(s: Optional[str]) -> bool:
    if s is None:
        return False
    txt = str(s).strip()
    return txt.startswith("{") and txt.endswith("}") and '"' in txt


def has_author_list_encoding_corruption(s: Optional[str]) -> bool:
    if s is None:
        return False
    txt = str(s)
    return ("�" in txt) or ("\ufffd" in txt) or ("��" in txt)


def is_normal_pagination_value(s: Optional[str]) -> bool:
    """
    放行常见的正常页码形式，只拦明显污染。
    正常：
      1187-9
      43-49
      714-9
      185-91
    可疑：
      714-9 st - [xxx]
      混入明显元数据串接
    """
    if s is None:
        return False
    txt = str(s).strip().lower()
    if re.fullmatch(r'\d+(-\d+)?', txt):
        return True
    return False


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


def check_not_null_rule(row_idx, row, rule):
    col = rule["column"]
    val = row.get(col)
    if pd.isna(val) or val is None:
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": "value is null but rule requires non-null",
        }
    return None


def check_pattern_rule(row_idx, row, rule):
    col = rule["column"]
    val = row.get(col)
    if (pd.isna(val) or val is None) and SKIP_NULL_FOR_NON_NOTNULL_RULES:
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
        return {
            "row_id": row_idx,
            "column": rhs,
            "value": rhs_val,
            "rule": rule,
            "reason": f"{rule['rule_type']} violation: lhs={lhs_key} suggests rhs={expected_rhs}, observed={rhs_val}",
            "expected_value": expected_rhs,
        }
    return None


def check_rare_value_rule(row_idx, row, rule):
    col = rule["column"]
    val = row.get(col)
    if pd.isna(val) or val is None:
        return None
    if val in set(rule.get("rare_values", [])):
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": "value is in rare_values set",
        }
    return None


def check_rare_pattern_rule(row_idx, row, rule):
    col = rule["column"]
    val = row.get(col)
    if pd.isna(val) or val is None:
        return None
    observed = detect_basic_pattern(val)
    if observed in set(rule.get("rare_patterns", [])):
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": f"pattern {observed} is rare for this column",
        }
    return None


def check_global_dominant_rule(row_idx, row, rule):
    col = rule["column"]
    val = row.get(col)
    if pd.isna(val) or val is None:
        return None
    dominant = rule.get("dominant_value")
    if dominant is not None and val != dominant:
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": f"value differs from global dominant value: dominant={dominant}, observed={val}",
            "expected_value": dominant,
        }
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
        return {
            "row_id": row_idx,
            "column": rhs,
            "value": tgt_val,
            "rule": rule,
            "reason": f"context dominant mismatch: {ctx_col}={ctx_val} suggests {rhs}={dominant_val}, observed={tgt_val}",
            "expected_value": dominant_val,
        }
    return None


def check_pair_context_rule(row_idx, row, rule):
    lhs = rule.get("lhs", [])
    rhs = rule.get("rhs")
    if len(lhs) != 2 or rhs is None:
        return None
    vals = [row.get(lhs[0]), row.get(lhs[1])]
    if vals[0] is None or vals[1] is None:
        return None
    tgt_val = row.get(rhs)
    if tgt_val is None:
        return None
    key = "||".join(map(str, vals))
    mapping = rule.get("_pair_context_mapping", {})
    if key not in mapping:
        return None
    dominant_val, support, dominant_ratio = mapping[key]
    if tgt_val != dominant_val:
        return {
            "row_id": row_idx,
            "column": rhs,
            "value": tgt_val,
            "rule": rule,
            "reason": f"context pair dominant mismatch: {lhs}={vals} suggests {rhs}={dominant_val}, observed={tgt_val}",
            "expected_value": dominant_val,
        }
    return None


def check_high_risk_missing_rule(row_idx, row, rule):
    col = rule["column"]
    val = row.get(col)
    if pd.isna(val) or val is None:
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": "value is null under high_risk_missing rule",
        }
    return None


def check_paired_missing_consistency_rule(row_idx, row, rule):
    cols = rule.get("columns", [])
    if len(cols) != 2:
        return None
    a_col, b_col = cols
    a, b = row.get(a_col), row.get(b_col)
    if (a is None and b is not None) or (a is not None and b is None):
        bad_col = a_col if a is None else b_col
        return {
            "row_id": row_idx,
            "column": bad_col,
            "value": row.get(bad_col),
            "rule": rule,
            "reason": f"paired missing inconsistency: {a_col}={a}, {b_col}={b}",
        }
    return None


def check_regex_rule(row_idx, row, rule):
    col = rule["column"]
    val = row.get(col)
    if pd.isna(val) or val is None:
        return None

    # 先做列专属白名单放行，减少误报
    if col == "journal_issn" and is_normal_journal_issn_variant(val):
        return None

    if col == "article_pagination" and is_normal_pagination_value(val):
        return None

    pattern = rule.get("regex")
    if pattern and re.match(pattern, str(val)) is None:
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": f"value does not match regex: {pattern}",
        }
    return None


def check_language_canonical_rule(row_idx, row, rule):
    col = rule["column"]
    val = row.get(col)
    if val is None:
        return None
    canonical = canonicalize_language(val)
    if canonical is None:
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": f"cannot canonicalize language value: {val}",
        }
    if str(val).strip().lower() != canonical:
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": f"language canonical mismatch: canonical={canonical}, observed={val}",
            "expected_value": canonical,
        }
    return None


def check_issn_canonical_rule(row_idx, row, rule):
    col = rule["column"]
    val = row.get(col)
    if val is None:
        return None

    # 正常常见变体直接放行
    if is_normal_journal_issn_variant(val):
        return None

    canonical = extract_issn_digits(val)
    if canonical is None:
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": f"cannot extract canonical ISSN from value: {val}",
        }
    if canonical not in str(val).strip().lower():
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": f"ISSN canonical mismatch: canonical={canonical}, observed={val}",
            "expected_value": canonical,
        }
    return None


def check_date_like_rule(row_idx, row, rule):
    col = rule["column"]
    val = row.get(col)
    if val is None:
        return None
    canonical = normalize_date_like(val)
    if canonical is None:
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": f"date value is not parseable: {val}",
        }
    return None


def check_date_component_permutation_rule(row_idx, row, rule):
    col = rule["column"]
    val = row.get(col)
    if val is None:
        return None

    current_canonical = normalize_date_like(val)
    if current_canonical is None:
        return None

    perms = generate_date_permutations(val)
    if not perms:
        return None

    date_dist = rule.get("date_distribution", {}) or {}
    top_years = date_dist.get("top_years", {}) or {}
    top_md = date_dist.get("top_month_day", {}) or {}

    current_year = current_canonical[:4]
    current_md = current_canonical[5:] if len(current_canonical) >= 10 else None
    current_score = int(top_years.get(current_year, 0)) + int(top_md.get(current_md, 0)) if current_md else 0

    best_perm = None
    best_score = current_score
    for p in perms:
        p_year = p[:4]
        p_md = p[5:] if len(p) >= 10 else None
        p_score = int(top_years.get(p_year, 0)) + int(top_md.get(p_md, 0)) if p_md else 0
        if p != current_canonical and p_score > best_score:
            best_score = p_score
            best_perm = p

    comp = split_date_components(val)
    if comp is not None:
        mm, dd, yy = comp
        suspicious_shape = (dd == 1) or (mm == 1)
    else:
        suspicious_shape = False

    if best_perm is not None and (best_score >= current_score + 2 or suspicious_shape):
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": f"date component permutation suspicious: current={current_canonical}, better_perm={best_perm}",
            "expected_value": best_perm,
        }
    return None


def check_date_ymd_swap_rule(row_idx, row, rule):
    """
    更强的列专属规则：
    专门抓当前 rayyan 中最主要的 article_jcreated_at 错位模式。
    """
    col = rule["column"]
    val = row.get(col)
    if val is None:
        return None

    current_canonical = normalize_date_like(val)
    if current_canonical is None:
        return None

    swap_candidate = generate_ymd_swap_candidate(val)
    if swap_candidate is None or swap_candidate == current_canonical:
        return None

    date_dist = rule.get("date_distribution", {}) or {}
    top_years = date_dist.get("top_years", {}) or {}
    top_md = date_dist.get("top_month_day", {}) or {}

    current_year = current_canonical[:4]
    current_md = current_canonical[5:] if len(current_canonical) >= 10 else None
    current_score = int(top_years.get(current_year, 0)) + int(top_md.get(current_md, 0)) if current_md else 0

    swap_year = swap_candidate[:4]
    swap_md = swap_candidate[5:] if len(swap_candidate) >= 10 else None
    swap_score = int(top_years.get(swap_year, 0)) + int(top_md.get(swap_md, 0)) if swap_md else 0

    comp = split_date_components(val)
    suspicious_shape = False
    if comp is not None:
        a, b, c = comp
        # 当前漏错里高频形态：第二段常是 1，或第一段是很小的年份段
        suspicious_shape = (b == 1) or (a == 1) or (a <= 31 and c <= 31)

    # 这个规则比 permutation 更硬：
    # 只要 swap 后更合理，或者命中强形态，就直接打入候选
    if (swap_score >= current_score + 1) or suspicious_shape:
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": f"date YY/M/D swap suspicious: current={current_canonical}, swap_candidate={swap_candidate}",
            "expected_value": swap_candidate,
        }
    return None


def check_volume_issue_numeric_rule(row_idx, row, rule):
    col = rule["column"]
    val = row.get(col)
    if val is None:
        return None
    canonical = canonicalize_volume_issue(val)
    if canonical is None:
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": f"volume/issue is not numeric-like: {val}",
        }
    return None


def check_author_list_braced_rule(row_idx, row, rule):
    col = rule["column"]
    val = row.get(col)
    if val is None:
        return None
    if not looks_like_author_list(val):
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": f"author_list does not look like braced quoted list: {val}",
        }
    return None


def check_author_list_encoding_corruption_rule(row_idx, row, rule):
    col = rule["column"]
    val = row.get(col)
    if val is None:
        return None
    if has_author_list_encoding_corruption(val):
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": f"author_list contains mojibake / replacement characters: {val}",
        }
    return None


def shrink_search_space(input_csv, rule_json, output_jsonl, output_csv):
    df = pd.read_csv(input_csv, dtype=str)
    df = maybe_drop_unnamed_columns(df)
    for col in df.columns:
        df[col] = normalize_series(df[col])

    rules = load_rule_pool(rule_json)

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

    regex_rules = [r for r in rules if r.get("rule_type") in {
        "issn_format", "issn_loose_format", "pagination_format"
    }]
    language_rules = [r for r in rules if r.get("rule_type") == "language_canonical_format"]
    issn_canonical_rules = [r for r in rules if r.get("rule_type") == "issn_canonical_format"]
    date_like_rules = [r for r in rules if r.get("rule_type") == "date_like_format"]
    date_permutation_rules = [r for r in rules if r.get("rule_type") == "date_component_permutation_suspicious"]
    date_ymd_swap_rules = [r for r in rules if r.get("rule_type") == "date_ymd_swap_suspicious"]
    volume_issue_rules = [r for r in rules if r.get("rule_type") == "volume_issue_numeric_format"]
    author_list_rules = [r for r in rules if r.get("rule_type") == "author_list_braced_format"]
    author_encoding_rules = [r for r in rules if r.get("rule_type") == "author_list_encoding_corruption"]

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
            (language_rules, check_language_canonical_rule),
            (issn_canonical_rules, check_issn_canonical_rule),
            (date_like_rules, check_date_like_rule),
            (date_permutation_rules, check_date_component_permutation_rule),
            (date_ymd_swap_rules, check_date_ymd_swap_rule),
            (volume_issue_rules, check_volume_issue_numeric_rule),
            (author_list_rules, check_author_list_braced_rule),
            (author_encoding_rules, check_author_list_encoding_corruption_rule),
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
    print(f"Shrink ratio: {len(candidates)/total_cells:.6f}")


if __name__ == "__main__":
    shrink_search_space(INPUT_CSV, INPUT_RULE_JSON, OUTPUT_JSONL, OUTPUT_CSV)

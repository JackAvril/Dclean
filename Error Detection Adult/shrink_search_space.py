import json
import math
import re
from collections import Counter
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd


# ============================================================
# 1. 用户配置区
# ============================================================

INPUT_CSV = "/mnt/mydata/dq/projects/Splittree/adult/adult_dirty.csv"
INPUT_RULE_JSON = "rule_pool_adult.json"

OUTPUT_JSONL = "candidate_cells_adult.jsonl"
OUTPUT_CSV = "candidate_cells_adult.csv"

MISSING_TOKENS = {
    "", "empty", "null", "none", "nan", "na", "n/a", "missing", "unk",
    "unknown", "?", "not available", "contact airline"
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
    "rare_value": 0.65,
    "rare_pattern": 0.65,
    "global_dominant_value": 0.55,
    "dominant_value_by_context": 0.9,
    "dominant_value_by_context_pair": 1.0,
    "high_risk_missing": 0.9,
    "paired_missing_consistency": 0.8,

    # adult schema / domain / format
    "age_bucket_format": 1.25,
    "hours_per_week_format_range": 1.20,
    "workclass_domain": 1.15,
    "education_domain": 1.15,
    "maritalstatus_domain": 1.20,
    "occupation_domain": 1.10,
    "relationship_domain": 1.20,
    "race_domain": 1.10,
    "sex_domain": 1.20,
    "country_domain": 0.95,
    "income_domain": 1.20,
    "income_canonical_format": 1.20,

    # adult consistency
    "age_education_consistency": 1.05,
    "age_relationship_consistency": 1.00,
    "marital_relationship_consistency": 1.25,
    "sex_relationship_consistency": 1.10,
    "workclass_occupation_consistency": 0.95,
    "education_occupation_consistency": 0.95,
    "hours_workclass_consistency": 0.85,

    # adult v2 attribution-aware rules
    "relationship_marital_spouse_conflict": 1.45,
    "relationship_age_spouse_conflict": 1.40,
    "relationship_sex_spouse_conflict": 1.25,
    "relationship_context_dominant": 1.15,
    "education_age_extreme_conflict": 1.10,
}

# ------------------------------------------------------------
# Adult domain definitions
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

# Adult 多数都是类别列，但有些弱规则容易过度放大候选。
# 这里做轻度 precision guard，原则是保持高召回，同时避免 majority/rare 造成候选池爆炸。
ENABLE_PRECISION_GUARD = True
GLOBAL_DOMINANT_SKIP_COLUMNS = {"workclass", "country", "income", "race"}
RARE_SKIP_COLUMNS = {"country", "occupation", "workclass"}

# ------------------------------------------------------------
# Adult v2 候选归因策略
# ------------------------------------------------------------
# 当前 Adult 评估显示：真实错误主要集中在 sex / relationship / 少量 education。
# 因此：
# 1) primary target 可以由强规则、domain/format、少量 attribution-aware 规则进入候选；
# 2) context-only 列主要作为证据，不允许由 rare/global/context-dominant/soft-FD 等弱规则大规模进入候选；
# 3) relationship 与 maritalstatus / age / sex 冲突时，优先把候选归因到 relationship。
PRIMARY_TARGET_COLUMNS = {
    "sex",
    "relationship",
    "education",
}

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

WEAK_RULE_TYPES_FOR_CONTEXT_ONLY = {
    "not_null",
    "high_risk_missing",
    "paired_missing_consistency",
    "rare_value",
    "rare_pattern",
    "global_dominant_value",
    "dominant_value_by_context",
    "dominant_value_by_context_pair",
    "functional_dependency",
    "soft_functional_dependency",
    "age_education_consistency",
    "age_relationship_consistency",
    "marital_relationship_consistency",
    "sex_relationship_consistency",
    "workclass_occupation_consistency",
    "education_occupation_consistency",
    "hours_workclass_consistency",
}

ALLOWED_CONTEXT_ONLY_STRONG_RULE_TYPES = {
    "age_bucket_format",
    "hours_per_week_format_range",
    "workclass_domain",
    "maritalstatus_domain",
    "occupation_domain",
    "race_domain",
    "country_domain",
    "income_domain",
    "income_canonical_format",
}

PRIMARY_RELATIONSHIP_RULE_TYPES = {
    "relationship_marital_spouse_conflict",
    "relationship_age_spouse_conflict",
    "relationship_sex_spouse_conflict",
    "relationship_context_dominant",
}

PRIMARY_EDUCATION_RULE_TYPES = {
    "education_age_extreme_conflict",
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


def norm_key(x: Optional[str]) -> Optional[str]:
    if x is None or pd.isna(x):
        return None
    return str(x).strip().lower()


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


def domain_contains(value: Optional[str], valid_values: set) -> bool:
    if value is None:
        return True
    return str(value).strip() in valid_values


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


# ============================================================
# 3. Precision guard
# ============================================================

def get_rule_target_column(rule: Dict[str, Any]) -> Optional[str]:
    if rule.get("target_column"):
        return rule.get("target_column")
    if rule.get("column"):
        return rule.get("column")
    if rule.get("rhs"):
        return rule.get("rhs")
    return None


def should_drop_rule_by_precision_guard(rule: Dict[str, Any]) -> bool:
    """过滤容易制造海量 FP 的弱规则。

    注意：这一步只是过滤规则本身；最终 add_violation 前还会做一次候选级别过滤。
    """
    if not ENABLE_PRECISION_GUARD:
        return False

    rtype = rule.get("rule_type", "")
    target_col = get_rule_target_column(rule)

    # v2 新增的 relationship attribution rules 必须保留。
    if rtype in PRIMARY_RELATIONSHIP_RULE_TYPES or rtype in PRIMARY_EDUCATION_RULE_TYPES:
        return False

    # context-only 列不允许被弱统计规则作为候选目标。
    if target_col in CONTEXT_ONLY_COLUMNS and rtype in WEAK_RULE_TYPES_FOR_CONTEXT_ONLY:
        return True

    # 老版本 rule_pool 中可能仍有这些爆炸规则，这里强制删除。
    if rtype == "global_dominant_value":
        return True

    if rtype in {"rare_value", "rare_pattern"}:
        return True

    # 普通 context dominant 只允许指向 primary target；context-only 目标一律删除。
    if rtype in {"dominant_value_by_context", "dominant_value_by_context_pair"}:
        rhs = rule.get("rhs")
        if rhs not in PRIMARY_TARGET_COLUMNS:
            return True

    # FD/soft-FD 只允许 RHS 是 primary target。
    if rtype in {"functional_dependency", "soft_functional_dependency"}:
        rhs = rule.get("rhs")
        if rhs not in PRIMARY_TARGET_COLUMNS:
            return True

    # high-risk missing / not_null 只允许 primary target，避免 age/income 等缺失被海量打候选。
    if rtype in {"not_null", "high_risk_missing"} and target_col not in PRIMARY_TARGET_COLUMNS:
        return True

    # paired missing 旧规则噪声很大，除非显式指向 primary target，否则删除。
    if rtype == "paired_missing_consistency":
        return True

    return False


def filter_rules_with_precision_guard(rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not ENABLE_PRECISION_GUARD:
        return rules

    kept = []
    removed = 0
    removed_counter = Counter()

    for r in rules:
        if should_drop_rule_by_precision_guard(r):
            removed += 1
            removed_counter[r.get("rule_type", "unknown")] += 1
            continue
        kept.append(r)

    print(f"[PrecisionGuard] rules before={len(rules)}, after={len(kept)}, removed={removed}")
    if removed_counter:
        print("[PrecisionGuard] removed by rule_type:")
        for k, v in removed_counter.most_common(20):
            print(f"  - {k}: {v}")

    return kept


def should_accept_violation(violation: Dict[str, Any]) -> bool:
    """候选级别二次过滤，防止旧规则或归因错误把上下文列打进候选。"""
    col = violation.get("column")
    rule = violation.get("rule", {})
    rtype = rule.get("rule_type", "")

    if col in PRIMARY_TARGET_COLUMNS:
        return True

    # context-only 列只有强 domain/format 非法值规则可以进入候选。
    if col in CONTEXT_ONLY_COLUMNS:
        return rtype in ALLOWED_CONTEXT_ONLY_STRONG_RULE_TYPES

    return True


# ============================================================
# 4. runtime mapping
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


def prepare_relationship_context_rule_runtime(rules):
    out = []
    for rule in rules:
        if rule.get("rule_type") != "relationship_context_dominant":
            continue

        mapping = {}
        for key, info in rule.get("majority_map", {}).items():
            mapping[key] = (
                info.get("dominant_value"),
                int(info.get("group_size", 0)),
                float(info.get("ratio", 0.0)),
            )

        r = dict(rule)
        r["_relationship_context_mapping"] = mapping
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


def prepare_adult_consistency_rule_runtime(df, rules):
    out = []

    # 用表内多数分布给部分 adult consistency rule 提供上下文 mapping。
    # 逻辑规则本身在 check_* 中单独检查。
    for rule in rules:
        rtype = rule.get("rule_type")

        if rtype == "workclass_occupation_consistency":
            r = dict(rule)
            r["_majority_mapping"] = build_context_majority_mapping_from_df(df, ["workclass"], "occupation", 3, 0.60)
            out.append(r)

        elif rtype == "education_occupation_consistency":
            r = dict(rule)
            r["_majority_mapping"] = build_context_majority_mapping_from_df(df, ["education"], "occupation", 3, 0.60)
            out.append(r)

        elif rtype == "hours_workclass_consistency":
            r = dict(rule)
            r["_majority_mapping"] = build_context_majority_mapping_from_df(df, ["workclass"], "hoursperweek", 3, 0.60)
            out.append(r)

    return out


# ============================================================
# 5. violation aggregation
# ============================================================

def add_violation(cell_map, violation):
    if not should_accept_violation(violation):
        return

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
# 6. 通用规则检查
# ============================================================

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


# ============================================================
# 7. Adult schema/domain checks
# ============================================================

def check_age_bucket_format_rule(row_idx, row, rule):
    col = rule["column"]
    val = row.get(col)

    if val is None:
        return None

    if not is_age_bucket_like(val):
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": f"age value is not a valid bucket: {val}",
        }

    valid_values = set(rule.get("valid_values", []))
    if valid_values and str(val).strip() not in valid_values:
        # 纯数字年龄如果格式合法但不是标准桶，也作为候选，便于 LLM 判定是否需要规范化。
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": f"age bucket is parseable but not in canonical bucket set: {val}",
        }

    return None


def check_hours_per_week_rule(row_idx, row, rule):
    col = rule["column"]
    val = row.get(col)

    if val is None:
        return None

    if not is_hours_like(val):
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": f"hoursperweek is not numeric/range or outside broad range: {val}",
        }

    lo, hi = parse_hours_value(val)
    lb = float(rule.get("lower_bound", 0))
    ub = float(rule.get("upper_bound", 120))

    if lo is not None and lo < lb:
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": f"hoursperweek lower value below bound: {lo} < {lb}",
        }

    if hi is not None and hi > ub:
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": f"hoursperweek upper value above bound: {hi} > {ub}",
        }

    return None


def check_domain_rule(row_idx, row, rule):
    col = rule["column"]
    val = row.get(col)

    if val is None:
        return None

    valid_values = set(rule.get("valid_values", []))

    if str(val).strip() not in valid_values:
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": f"value is outside domain: observed={val}, valid={sorted(list(valid_values))[:20]}",
        }

    return None


def check_income_canonical_rule(row_idx, row, rule):
    col = rule["column"]
    val = row.get(col)

    if val is None:
        return None

    canonical = canonical_income(val)

    if canonical is None:
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": f"income cannot be canonicalized to LessThan50K/MoreThan50K: {val}",
        }

    if str(val).strip() != canonical:
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": f"income canonical mismatch: canonical={canonical}, observed={val}",
            "expected_value": canonical,
        }

    return None



# ============================================================
# 8. Adult v2 attribution-aware checks
# ============================================================

def check_relationship_marital_spouse_conflict_rule(row_idx, row, rule):
    marital = row.get("maritalstatus")
    rel = row.get("relationship")

    if marital is None or rel is None:
        return None

    spouse_values = set(rule.get("spouse_values", [])) or RELATIONSHIP_SPOUSE_VALUES
    conflicting_status = set(rule.get("conflicting_maritalstatus", [])) or NON_MARRIED_STATUS_FOR_SPOUSE

    if rel in spouse_values and marital in conflicting_status:
        return {
            "row_id": row_idx,
            "column": "relationship",
            "value": rel,
            "rule": rule,
            "reason": (
                f"relationship={rel} conflicts with maritalstatus={marital}; "
                f"attribute suspicious value to relationship"
            ),
        }

    return None


def check_relationship_age_spouse_conflict_rule(row_idx, row, rule):
    age = row.get("age")
    rel = row.get("relationship")

    if age is None or rel is None:
        return None

    spouse_values = set(rule.get("spouse_values", [])) or RELATIONSHIP_SPOUSE_VALUES
    max_hi = int(rule.get("max_age_upper_for_conflict", 17))

    lo, hi = parse_age_bucket(age)
    if lo is None:
        return None

    if hi is not None and hi <= max_hi and rel in spouse_values:
        return {
            "row_id": row_idx,
            "column": "relationship",
            "value": rel,
            "rule": rule,
            "reason": (
                f"age bucket {age} is very young but relationship={rel}; "
                f"attribute suspicious value to relationship"
            ),
        }

    return None


def check_relationship_sex_spouse_conflict_rule(row_idx, row, rule):
    sex = row.get("sex")
    rel = row.get("relationship")

    if sex is None or rel is None:
        return None

    expected_by_rel = rule.get("expected_by_relationship", {}) or VALID_SEX_BY_RELATIONSHIP
    expected = expected_by_rel.get(rel)

    if expected is None:
        return None

    # 如果 sex 本身是非法值，交给 sex_domain 规则；如果 sex 是合法值但和 Husband/Wife 冲突，
    # Adult 当前错误模式更可能是 relationship 被替换，所以归因到 relationship。
    if sex in VALID_SEX and sex != expected:
        return {
            "row_id": row_idx,
            "column": "relationship",
            "value": rel,
            "rule": rule,
            "reason": (
                f"relationship={rel} expects sex={expected}, but observed valid sex={sex}; "
                f"attribute suspicious value to relationship"
            ),
            "expected_value": f"non-{rel} relationship",
        }

    return None


def check_relationship_context_dominant_rule(row_idx, row, rule):
    lhs = rule.get("lhs", [])
    rhs = rule.get("rhs", "relationship")
    if rhs != "relationship" or not lhs:
        return None

    rel = row.get("relationship")
    if rel is None:
        return None

    trigger_values = set(rule.get("only_trigger_observed_values", []))
    if trigger_values and rel not in trigger_values:
        return None

    vals = []
    for c in lhs:
        v = row.get(c)
        if v is None:
            return None
        vals.append(v)

    key = "||".join(map(str, vals))
    mapping = rule.get("_relationship_context_mapping", {})

    if key not in mapping:
        return None

    dominant_val, support, ratio = mapping[key]

    if rel != dominant_val:
        return {
            "row_id": row_idx,
            "column": "relationship",
            "value": rel,
            "rule": rule,
            "reason": (
                f"relationship context dominant mismatch: lhs={lhs}={vals} "
                f"suggests relationship={dominant_val}, observed={rel}, ratio={ratio:.3f}, support={support}"
            ),
            "expected_value": dominant_val,
        }

    return None


def check_education_age_extreme_conflict_rule(row_idx, row, rule):
    age = row.get("age")
    edu = row.get("education")

    if age is None or edu is None:
        return None

    lo, hi = parse_age_bucket(age)
    edu_order = EDUCATION_ORDER.get(str(edu).strip())

    if lo is None or hi is None or edu_order is None:
        return None

    max_hi = int(rule.get("max_age_upper_for_conflict", 17))
    min_edu_order = int(rule.get("min_extreme_education_order", EDUCATION_ORDER["Bachelors"]))

    if hi <= max_hi and edu_order >= min_edu_order:
        return {
            "row_id": row_idx,
            "column": "education",
            "value": edu,
            "rule": rule,
            "reason": (
                f"age bucket {age} is very young but education={edu} is high-level; "
                f"attribute suspicious value to education"
            ),
        }

    return None


# ============================================================
# 8. Adult consistency checks
# ============================================================

def check_adult_context_mapping_rule(row_idx, row, rule):
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
        return {
            "row_id": row_idx,
            "column": rhs,
            "value": rhs_val,
            "rule": rule,
            "reason": f"{rule['rule_type']} violation: lhs={key} suggests {rhs}={expected}, observed={rhs_val}, ratio={ratio:.3f}",
            "expected_value": expected,
        }

    return None


def check_age_education_consistency_rule(row_idx, row, rule):
    age = row.get("age")
    edu = row.get("education")

    if age is None or edu is None:
        return None

    lo, hi = parse_age_bucket(age)
    edu_ord = EDUCATION_ORDER.get(str(edu).strip())

    if lo is None or edu_ord is None:
        return None

    # 年龄 <18 时，Doctorate/Masters/Prof-school/Bachelors 明显不合理。
    # 这里只作为 candidate，不是最终 label。
    if hi is not None and hi < 18 and edu_ord >= EDUCATION_ORDER["Bachelors"]:
        return {
            "row_id": row_idx,
            "column": "education",
            "value": edu,
            "rule": rule,
            "reason": f"age bucket {age} is very young but education={edu} is high-level",
        }

    # 18-21 出现 Doctorate/Prof-school 也非常可疑。
    if hi is not None and hi <= 21 and edu in {"Doctorate", "Prof-school"}:
        return {
            "row_id": row_idx,
            "column": "education",
            "value": edu,
            "rule": rule,
            "reason": f"age bucket {age} is young but education={edu} is very high-level",
        }

    return None


def check_age_relationship_consistency_rule(row_idx, row, rule):
    age = row.get("age")
    rel = row.get("relationship")

    if age is None or rel is None:
        return None

    lo, hi = parse_age_bucket(age)
    if lo is None:
        return None

    # <18 或 18-21 更常见 Own-child，但不是强制；
    # 只对明显矛盾角色做候选，如 Husband/Wife。
    if hi is not None and hi < 18 and rel in {"Husband", "Wife"}:
        return {
            "row_id": row_idx,
            "column": "relationship",
            "value": rel,
            "rule": rule,
            "reason": f"age bucket {age} is very young but relationship={rel}",
        }

    return None


def check_marital_relationship_consistency_rule(row_idx, row, rule):
    # 兼容旧 rule_type：旧版 marital_relationship_consistency 也按 v2 归因到 relationship。
    marital = row.get("maritalstatus")
    rel = row.get("relationship")

    if marital is None or rel is None:
        return None

    if rel in RELATIONSHIP_SPOUSE_VALUES and marital in NON_MARRIED_STATUS_FOR_SPOUSE:
        return {
            "row_id": row_idx,
            "column": "relationship",
            "value": rel,
            "rule": rule,
            "reason": (
                f"relationship={rel} but maritalstatus={marital} is inconsistent; "
                f"attribute suspicious value to relationship"
            ),
        }

    return None


def check_sex_relationship_consistency_rule(row_idx, row, rule):
    # 兼容旧 rule_type：如果 sex 合法但与 Husband/Wife 冲突，归因到 relationship；
    # 如果 sex 非法，则由 sex_domain 规则处理 sex。
    sex = row.get("sex")
    rel = row.get("relationship")

    if sex is None or rel is None:
        return None

    expected = VALID_SEX_BY_RELATIONSHIP.get(rel)
    if expected is None:
        return None

    if sex in VALID_SEX and sex != expected:
        return {
            "row_id": row_idx,
            "column": "relationship",
            "value": rel,
            "rule": rule,
            "reason": (
                f"relationship={rel} expects sex={expected}, but observed valid sex={sex}; "
                f"attribute suspicious value to relationship"
            ),
            "expected_value": f"non-{rel} relationship",
        }

    return None


def check_workclass_occupation_consistency_rule(row_idx, row, rule):
    workclass = row.get("workclass")
    occupation = row.get("occupation")

    if workclass is None or occupation is None:
        return None

    # Never-worked / Without-pay 与大多数 occupation 组合可疑。
    if workclass == "Never-worked" and occupation not in {None, "?"}:
        return {
            "row_id": row_idx,
            "column": "occupation",
            "value": occupation,
            "rule": rule,
            "reason": f"workclass=Never-worked but occupation={occupation}",
        }

    if occupation == "Armed-Forces" and workclass not in {"Federal-gov"}:
        return {
            "row_id": row_idx,
            "column": "workclass",
            "value": workclass,
            "rule": rule,
            "reason": f"occupation=Armed-Forces but workclass={workclass} is unusual",
        }

    # 如果 runtime mapping 里有强多数证据，也作为候选。
    return check_adult_context_mapping_rule(row_idx, row, rule)


def check_education_occupation_consistency_rule(row_idx, row, rule):
    # 先不写强逻辑，主要依赖表内多数映射；避免把合法低学历职业误打。
    return check_adult_context_mapping_rule(row_idx, row, rule)


def check_hours_workclass_consistency_rule(row_idx, row, rule):
    workclass = row.get("workclass")
    hours = row.get("hoursperweek")

    if workclass is None or hours is None:
        return None

    lo, hi = parse_hours_value(hours)
    if lo is None:
        return {
            "row_id": row_idx,
            "column": "hoursperweek",
            "value": hours,
            "rule": rule,
            "reason": f"cannot parse hoursperweek={hours}",
        }

    # Without-pay 但极高工时、或 Never-worked 但非零工时，都作为候选。
    if workclass == "Never-worked" and hi is not None and hi > 0:
        return {
            "row_id": row_idx,
            "column": "hoursperweek",
            "value": hours,
            "rule": rule,
            "reason": f"workclass=Never-worked but hoursperweek={hours}",
            "expected_value": "0",
        }

    if workclass == "Without-pay" and lo is not None and lo > 80:
        return {
            "row_id": row_idx,
            "column": "hoursperweek",
            "value": hours,
            "rule": rule,
            "reason": f"workclass=Without-pay but hoursperweek={hours} is very high",
        }

    # 其他组合主要依赖宽格式/range，不用多数 mapping 过度限制，避免 FP。
    return None


# ============================================================
# 9. 主流程
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
    relationship_context_rules = prepare_relationship_context_rule_runtime(rules)
    high_risk_missing_rules = [r for r in rules if r.get("rule_type") == "high_risk_missing"]
    paired_missing_rules = [r for r in rules if r.get("rule_type") == "paired_missing_consistency"]

    regex_rules = [r for r in rules if r.get("rule_type") in {
        # 保留兼容：如果 rule_pool 后续新增普通 regex 规则，可直接生效。
        "generic_regex",
    }]

    age_rules = [r for r in rules if r.get("rule_type") == "age_bucket_format"]
    hours_rules = [r for r in rules if r.get("rule_type") == "hours_per_week_format_range"]

    domain_rules = [r for r in rules if r.get("rule_type") in {
        "workclass_domain",
        "education_domain",
        "maritalstatus_domain",
        "occupation_domain",
        "relationship_domain",
        "race_domain",
        "sex_domain",
        "country_domain",
        "income_domain",
    }]

    income_canonical_rules = [r for r in rules if r.get("rule_type") == "income_canonical_format"]

    age_education_rules = [r for r in rules if r.get("rule_type") == "age_education_consistency"]
    age_relationship_rules = [r for r in rules if r.get("rule_type") == "age_relationship_consistency"]
    marital_relationship_rules = [r for r in rules if r.get("rule_type") == "marital_relationship_consistency"]
    sex_relationship_rules = [r for r in rules if r.get("rule_type") == "sex_relationship_consistency"]

    relationship_marital_rules = [r for r in rules if r.get("rule_type") == "relationship_marital_spouse_conflict"]
    relationship_age_rules = [r for r in rules if r.get("rule_type") == "relationship_age_spouse_conflict"]
    relationship_sex_rules = [r for r in rules if r.get("rule_type") == "relationship_sex_spouse_conflict"]
    education_age_extreme_rules = [r for r in rules if r.get("rule_type") == "education_age_extreme_conflict"]

    adult_context_runtime_rules = prepare_adult_consistency_rule_runtime(df, rules)
    workclass_occupation_rules = [r for r in adult_context_runtime_rules if r.get("rule_type") == "workclass_occupation_consistency"]
    education_occupation_rules = [r for r in adult_context_runtime_rules if r.get("rule_type") == "education_occupation_consistency"]
    hours_workclass_rules = [r for r in rules if r.get("rule_type") == "hours_workclass_consistency"]

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
            (relationship_context_rules, check_relationship_context_dominant_rule),
            (high_risk_missing_rules, check_high_risk_missing_rule),
            (paired_missing_rules, check_paired_missing_consistency_rule),

            (regex_rules, check_regex_rule),
            (age_rules, check_age_bucket_format_rule),
            (hours_rules, check_hours_per_week_rule),
            (domain_rules, check_domain_rule),
            (income_canonical_rules, check_income_canonical_rule),

            (age_education_rules, check_age_education_consistency_rule),
            (age_relationship_rules, check_age_relationship_consistency_rule),
            (marital_relationship_rules, check_marital_relationship_consistency_rule),
            (sex_relationship_rules, check_sex_relationship_consistency_rule),

            (relationship_marital_rules, check_relationship_marital_spouse_conflict_rule),
            (relationship_age_rules, check_relationship_age_spouse_conflict_rule),
            (relationship_sex_rules, check_relationship_sex_spouse_conflict_rule),
            (education_age_extreme_rules, check_education_age_extreme_conflict_rule),

            (workclass_occupation_rules, check_workclass_occupation_consistency_rule),
            (education_occupation_rules, check_education_occupation_consistency_rule),
            (hours_workclass_rules, check_hours_workclass_consistency_rule),
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

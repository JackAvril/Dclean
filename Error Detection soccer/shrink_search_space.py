import json
import math
import re
from collections import Counter
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd


# ============================================================
# 1. 用户配置区
# ============================================================

INPUT_CSV = "/mnt/mydata/dq/projects/Splittree/soccer/soccer_dirty.csv"
INPUT_RULE_JSON = "rule_pool_soccer.json"

OUTPUT_JSONL = "candidate_cells_soccer.jsonl"
OUTPUT_CSV = "candidate_cells_soccer.csv"

MISSING_TOKENS = {
    "", "empty", "null", "none", "nan", "na", "n/a", "missing", "unk",
    "unknown", "?", "not available", "contact airline"
}

SKIP_NULL_FOR_NON_NOTNULL_RULES = True
MAX_FD_MAPPING_SIZE = 200000
MAX_CONTEXT_MAPPING_SIZE = 200000

# ============================================================
# Soccer v2 候选控制
# ============================================================
# 关键目标：
# 1) name/surname/birthplace 不再被排除在候选外；
# 2) missing / schema / domain / regex / numeric range 直接进候选；
# 3) soft-FD / team-context-dominant 默认不直接生成候选，只作为 evidence-only；
# 4) functional_dependency 只有极高置信、高纯度 group 才允许生成候选。

ENABLE_PRECISION_GUARD = True

# 如果规则池里已经标记为 evidence_only 或 allow_candidate_generation=False，
# shrink 阶段必须跳过，避免 soft-FD 再次冲爆候选范围。
SKIP_EVIDENCE_ONLY_RULES = True

# 强 FD 作为候选的额外运行时约束。
FD_CANDIDATE_MIN_CONFIDENCE = 0.995
FD_CANDIDATE_MIN_SUPPORT = 50
FD_CANDIDATE_MIN_GROUP_SIZE = 10
FD_CANDIDATE_MIN_DOMINANT_RATIO = 0.98
FD_CANDIDATE_MIN_DOMINANT_MARGIN = 5

# team/context 规则如果未来允许作为候选，也必须非常严格。
CONTEXT_CANDIDATE_MIN_CONFIDENCE = 0.98
CONTEXT_CANDIDATE_MIN_SUPPORT = 50
CONTEXT_CANDIDATE_MIN_GROUP_SIZE = 20
CONTEXT_CANDIDATE_MIN_DOMINANT_RATIO = 0.98
CONTEXT_CANDIDATE_MIN_DOMINANT_MARGIN = 5

# 对明显弱规则直接禁用候选生成。
DISABLED_CANDIDATE_RULE_TYPES = {
    "soft_functional_dependency",
    "rare_value",
    "rare_pattern",
    "global_dominant_value",
    "dominant_value_by_context",
    "dominant_value_by_context_pair",
    "team_context_dominant",
    "team_season_manager_consistency_hint",
    "city_stadium_consistency_hint",
}

PRIORITY_WEIGHT = {"high": 1.5, "medium": 1.0, "low": 0.6}

RULE_TYPE_WEIGHT = {
    "not_null": 1.2,
    "pattern": 0.75,

    # FD 只有强 FD 允许作为候选，soft-FD 默认被 precision guard 跳过。
    "functional_dependency": 1.10,
    "soft_functional_dependency": 0.25,

    "rare_value": 0.20,
    "rare_pattern": 0.20,
    "global_dominant_value": 0.20,
    "dominant_value_by_context": 0.35,
    "dominant_value_by_context_pair": 0.35,
    "high_risk_missing": 1.10,
    "paired_missing_consistency": 0.7,

    # soccer schema / domain / format
    "birthyear_numeric_range": 1.45,
    "season_numeric_range": 1.35,
    "position_domain": 1.45,

    # name/surname/birthplace 在上一轮中大量漏检，所以 missing/pattern 必须进入候选。
    "name_pattern": 1.05,
    "surname_pattern": 1.05,
    "manager_name_pattern": 1.00,
    "team_name_pattern": 0.95,
    "birthplace_location_pattern": 1.05,
    "city_location_pattern": 0.95,
    "stadium_name_pattern": 0.95,
    "generic_regex": 0.80,

    # soccer context / consistency
    "birthyear_season_age_consistency": 1.45,
    "team_context_dominant": 0.35,
    "team_season_manager_consistency_hint": 0.30,
    "city_stadium_consistency_hint": 0.30,
}

# ------------------------------------------------------------
# Soccer domain definitions
# ------------------------------------------------------------

SOCCER_COLUMNS = [
    "name",
    "surname",
    "birthyear",
    "birthplace",
    "position",
    "team",
    "city",
    "stadium",
    "season",
    "manager",
]

VALID_POSITIONS = {
    "Goalkeeper",
    "Defender",
    "Midfield",
    "Midfielder",
    "Forward",
    "Striker",
    "Winger",
}

POSITION_CANONICAL_MAP = {
    "Midfielder": "Midfield",
}

BIRTHYEAR_MIN = 1940
BIRTHYEAR_MAX = 2010
SEASON_MIN = 1900
SEASON_MAX = 2035
PLAYER_MIN_AGE_AT_SEASON = 15
PLAYER_MAX_AGE_AT_SEASON = 50

PERSON_NAME_REGEX = r"^[A-Za-z]+(?:_[A-Za-z]+)*$"
SURNAME_REGEX = r"^[A-Za-z]+(?:[-_'][A-Za-z]+)*$"
TEAM_REGEX = r"^[A-Za-z0-9]+(?:[._'\-][A-Za-z0-9]+)*$"
LOCATION_REGEX = r"^[A-Za-z]+(?:[ _'\-][A-Za-z]+)*$"
STADIUM_REGEX = r"^[A-Za-z0-9]+(?:[ _'\-][A-Za-z0-9]+)*$"

# ------------------------------------------------------------
# Soccer 候选策略
# ------------------------------------------------------------
# v2：所有列都允许作为候选目标。上一轮 name/surname/birthplace 几乎没进候选，
# 导致 5000+ 真实错误被候选阶段直接漏掉。
PRIMARY_TARGET_COLUMNS = set(SOCCER_COLUMNS)
CONTEXT_ONLY_COLUMNS = set()

# 直接可生成候选的强/中强规则。
ALWAYS_ALLOW_RULE_TYPES = {
    "not_null",
    "high_risk_missing",

    "birthyear_numeric_range",
    "season_numeric_range",
    "position_domain",
    "birthyear_season_age_consistency",

    "name_pattern",
    "surname_pattern",
    "manager_name_pattern",
    "team_name_pattern",
    "birthplace_location_pattern",
    "city_location_pattern",
    "stadium_name_pattern",
    "generic_regex",
    "pattern",
}

SOCCER_STRONG_RULE_TYPES = {
    "birthyear_numeric_range",
    "season_numeric_range",
    "position_domain",
    "birthyear_season_age_consistency",
}

SOCCER_FORMAT_RULE_TYPES = {
    "name_pattern",
    "surname_pattern",
    "manager_name_pattern",
    "team_name_pattern",
    "birthplace_location_pattern",
    "city_location_pattern",
    "stadium_name_pattern",
    "pattern",
    "generic_regex",
}

SOCCER_WEAK_CONTEXT_RULE_TYPES = {
    "team_context_dominant",
    "team_season_manager_consistency_hint",
    "city_stadium_consistency_hint",
    "functional_dependency",
    "soft_functional_dependency",
}

GLOBAL_DOMINANT_SKIP_COLUMNS = set(SOCCER_COLUMNS)
RARE_SKIP_COLUMNS = set(SOCCER_COLUMNS)


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


def parse_year(value: Optional[str]) -> Optional[int]:
    n = parse_int_like(value)
    if n is None:
        return None
    return n


def canonical_position(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if s in POSITION_CANONICAL_MAP:
        return POSITION_CANONICAL_MAP[s]
    if s in VALID_POSITIONS:
        return s
    return None


def load_rule_pool(rule_json_path):
    with open(rule_json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    all_rules = []
    for _, rules in obj.get("rules", {}).items():
        for rule in rules:
            all_rules.append(dict(rule))

    return all_rules


def safe_float(x, default=0.0):
    try:
        if x is None or pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x, default=0):
    try:
        if x is None or pd.isna(x):
            return default
        return int(float(x))
    except Exception:
        return default


def rule_weight(rule):
    conf = safe_float(rule.get("confidence", 0.0), 0.0)
    supp = safe_int(rule.get("support", 0), 0)
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


def is_evidence_only_rule(rule: Dict[str, Any]) -> bool:
    if not SKIP_EVIDENCE_ONLY_RULES:
        return False

    usage_role = str(rule.get("usage_role", "")).strip().lower()
    candidate_flag = rule.get("allow_candidate_generation", None)

    if usage_role == "evidence_only":
        return True

    if candidate_flag is False:
        return True

    candidate_trigger = str(rule.get("candidate_trigger", "")).strip().lower()
    if candidate_trigger in {"disabled", "disabled_evidence_only", "evidence_only"}:
        return True

    return False


def is_fd_candidate_rule_allowed(rule: Dict[str, Any]) -> bool:
    """functional_dependency 只有极高可信时才允许生成候选。"""
    if rule.get("rule_type") != "functional_dependency":
        return False

    if is_evidence_only_rule(rule):
        return False

    rhs = rule.get("rhs") or rule.get("target_column")
    if rhs not in PRIMARY_TARGET_COLUMNS:
        return False

    confidence = safe_float(rule.get("confidence", 0.0), 0.0)
    support = safe_int(rule.get("support", 0), 0)

    if confidence < FD_CANDIDATE_MIN_CONFIDENCE:
        return False
    if support < FD_CANDIDATE_MIN_SUPPORT:
        return False

    return True


def is_context_candidate_rule_allowed(rule: Dict[str, Any]) -> bool:
    """team/context dominant 默认禁用；若规则池显式允许，也要满足高阈值。"""
    if rule.get("rule_type") not in {
        "team_context_dominant",
        "dominant_value_by_context",
        "dominant_value_by_context_pair",
    }:
        return False

    if is_evidence_only_rule(rule):
        return False

    if rule.get("allow_candidate_generation", False) is not True:
        return False

    rhs = rule.get("rhs") or rule.get("target_column")
    if rhs not in PRIMARY_TARGET_COLUMNS:
        return False

    confidence = safe_float(rule.get("confidence", 0.0), 0.0)
    support = safe_int(rule.get("support", 0), 0)

    if confidence < CONTEXT_CANDIDATE_MIN_CONFIDENCE:
        return False
    if support < CONTEXT_CANDIDATE_MIN_SUPPORT:
        return False

    return True


def should_drop_rule_by_precision_guard(rule: Dict[str, Any]) -> bool:
    """过滤容易制造海量 FP 的弱规则。"""
    if not ENABLE_PRECISION_GUARD:
        return False

    if is_evidence_only_rule(rule):
        return True

    rtype = rule.get("rule_type", "")
    target_col = get_rule_target_column(rule)

    if rtype in ALWAYS_ALLOW_RULE_TYPES:
        return False

    # soft-FD / rare / global / generic context 默认禁止直接生成候选。
    if rtype in DISABLED_CANDIDATE_RULE_TYPES:
        return True

    if rtype == "functional_dependency":
        return not is_fd_candidate_rule_allowed(rule)

    if rtype in {"team_context_dominant", "dominant_value_by_context", "dominant_value_by_context_pair"}:
        return not is_context_candidate_rule_allowed(rule)

    if target_col is not None and target_col not in PRIMARY_TARGET_COLUMNS:
        return True

    # 未知规则默认保守过滤，避免新规则类型突然冲爆候选。
    return True


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
        for k, v in removed_counter.most_common(30):
            print(f"  - {k}: {v}")

    kept_counter = Counter([r.get("rule_type", "unknown") for r in kept])
    print("[PrecisionGuard] kept by rule_type:")
    for k, v in kept_counter.most_common(30):
        print(f"  - {k}: {v}")

    return kept


def should_accept_violation(violation: Dict[str, Any]) -> bool:
    col = violation.get("column")
    rule = violation.get("rule", {})
    rtype = rule.get("rule_type", "")

    if is_evidence_only_rule(rule):
        return False

    if col not in PRIMARY_TARGET_COLUMNS:
        return False

    if rtype in ALWAYS_ALLOW_RULE_TYPES:
        return True

    if rtype == "functional_dependency":
        return is_fd_candidate_rule_allowed(rule)

    if rtype in {"team_context_dominant", "dominant_value_by_context", "dominant_value_by_context_pair"}:
        return is_context_candidate_rule_allowed(rule)

    return False


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
        most_common = cnt.most_common(2)
        majority_val, majority_cnt = most_common[0]
        second_cnt = most_common[1][1] if len(most_common) > 1 else 0
        total_cnt = len(rhs_vals)
        ratio = majority_cnt / total_cnt if total_cnt > 0 else 0.0
        margin = majority_cnt - second_cnt

        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        mapping[group_key] = {
            "majority_value": majority_val,
            "majority_count": majority_cnt,
            "second_count": second_cnt,
            "total_count": total_cnt,
            "dominant_ratio": ratio,
            "dominant_margin": margin,
        }

        if len(mapping) >= MAX_FD_MAPPING_SIZE:
            break

    return mapping


def prepare_fd_rule_runtime(df, rules):
    out = []
    for rule in rules:
        if rule.get("rule_type") != "functional_dependency":
            continue
        if not is_fd_candidate_rule_allowed(rule):
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
        if not is_context_candidate_rule_allowed(rule):
            continue

        mapping = {}
        for ctx_val, info in rule.get("majority_map", {}).items():
            majority_cnt = safe_int(info.get("dominant_count", 0), 0)
            group_size = safe_int(info.get("group_size", 0), 0)
            second_count = safe_int(info.get("second_count", 0), 0)
            ratio = safe_float(info.get("ratio", 0.0), 0.0)
            margin = majority_cnt - second_count if majority_cnt > 0 else safe_int(info.get("dominant_margin", 0), 0)

            mapping[ctx_val] = {
                "dominant_value": info.get("dominant_value"),
                "group_size": group_size,
                "ratio": ratio,
                "dominant_count": majority_cnt,
                "second_count": second_count,
                "dominant_margin": margin,
            }

        r = dict(rule)
        r["_context_mapping"] = mapping
        out.append(r)

    return out


def prepare_pair_context_rule_runtime(rules):
    out = []
    for rule in rules:
        if rule.get("rule_type") != "dominant_value_by_context_pair":
            continue
        if not is_context_candidate_rule_allowed(rule):
            continue

        mapping = {}
        for key, info in rule.get("majority_map", {}).items():
            majority_cnt = safe_int(info.get("dominant_count", 0), 0)
            group_size = safe_int(info.get("group_size", 0), 0)
            second_count = safe_int(info.get("second_count", 0), 0)
            ratio = safe_float(info.get("ratio", 0.0), 0.0)
            margin = majority_cnt - second_count if majority_cnt > 0 else safe_int(info.get("dominant_margin", 0), 0)

            mapping[key] = {
                "dominant_value": info.get("dominant_value"),
                "group_size": group_size,
                "ratio": ratio,
                "dominant_count": majority_cnt,
                "second_count": second_count,
                "dominant_margin": margin,
            }

        r = dict(rule)
        r["_pair_context_mapping"] = mapping
        out.append(r)

    return out


def prepare_team_context_rule_runtime(rules):
    out = []
    for rule in rules:
        if rule.get("rule_type") != "team_context_dominant":
            continue
        if not is_context_candidate_rule_allowed(rule):
            continue

        mapping = {}
        for key, info in rule.get("majority_map", {}).items():
            majority_cnt = safe_int(info.get("dominant_count", 0), 0)
            group_size = safe_int(info.get("group_size", 0), 0)
            second_count = safe_int(info.get("second_count", 0), 0)
            ratio = safe_float(info.get("ratio", 0.0), 0.0)
            margin = majority_cnt - second_count if majority_cnt > 0 else safe_int(info.get("dominant_margin", 0), 0)

            mapping[key] = {
                "dominant_value": info.get("dominant_value"),
                "group_size": group_size,
                "ratio": ratio,
                "dominant_count": majority_cnt,
                "second_count": second_count,
                "dominant_margin": margin,
            }

        r = dict(rule)
        r["_team_context_mapping"] = mapping
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

    # 额外保留 FD/context group 质量，后续上下文和采样可以用。
    for extra_key in [
        "group_size",
        "dominant_ratio",
        "dominant_margin",
        "dominant_count",
        "second_count",
    ]:
        if extra_key in violation:
            info[extra_key] = violation[extra_key]

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
    target = rule.get("pattern") or rule.get("dominant_pattern")

    if target is None:
        return None

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

    info = mapping[lhs_key]
    expected_rhs = info["majority_value"]
    majority_cnt = safe_int(info.get("majority_count", 0), 0)
    total_cnt = safe_int(info.get("total_count", 0), 0)
    second_cnt = safe_int(info.get("second_count", 0), 0)
    dominant_ratio = safe_float(info.get("dominant_ratio", 0.0), 0.0)
    dominant_margin = safe_int(info.get("dominant_margin", majority_cnt - second_cnt), 0)

    if total_cnt < FD_CANDIDATE_MIN_GROUP_SIZE:
        return None
    if dominant_ratio < FD_CANDIDATE_MIN_DOMINANT_RATIO:
        return None
    if dominant_margin < FD_CANDIDATE_MIN_DOMINANT_MARGIN:
        return None

    if rhs_val != expected_rhs:
        return {
            "row_id": row_idx,
            "column": rhs,
            "value": rhs_val,
            "rule": rule,
            "reason": (
                f"functional_dependency high-purity violation: lhs={lhs_key} suggests rhs={expected_rhs}, "
                f"observed={rhs_val}, group_size={total_cnt}, ratio={dominant_ratio:.6f}, margin={dominant_margin}"
            ),
            "expected_value": expected_rhs,
            "group_size": total_cnt,
            "dominant_ratio": dominant_ratio,
            "dominant_margin": dominant_margin,
            "dominant_count": majority_cnt,
            "second_count": second_cnt,
        }

    return None


def check_rare_value_rule(row_idx, row, rule):
    return None


def check_rare_pattern_rule(row_idx, row, rule):
    return None


def check_global_dominant_rule(row_idx, row, rule):
    return None


def _context_info_passes_candidate_guard(info: Dict[str, Any]) -> bool:
    group_size = safe_int(info.get("group_size", 0), 0)
    ratio = safe_float(info.get("ratio", 0.0), 0.0)
    margin = safe_int(info.get("dominant_margin", 0), 0)
    dominant_count = safe_int(info.get("dominant_count", 0), 0)
    second_count = safe_int(info.get("second_count", 0), 0)

    if margin <= 0 and dominant_count > 0:
        margin = dominant_count - second_count

    if group_size < CONTEXT_CANDIDATE_MIN_GROUP_SIZE:
        return False
    if ratio < CONTEXT_CANDIDATE_MIN_DOMINANT_RATIO:
        return False
    if margin < CONTEXT_CANDIDATE_MIN_DOMINANT_MARGIN:
        return False
    return True


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

    info = mapping[ctx_val]
    if not _context_info_passes_candidate_guard(info):
        return None

    dominant_val = info.get("dominant_value")

    if tgt_val != dominant_val:
        return {
            "row_id": row_idx,
            "column": rhs,
            "value": tgt_val,
            "rule": rule,
            "reason": (
                f"context dominant high-purity mismatch: {ctx_col}={ctx_val} suggests {rhs}={dominant_val}, "
                f"observed={tgt_val}, group_size={info.get('group_size')}, ratio={info.get('ratio')}"
            ),
            "expected_value": dominant_val,
            "group_size": info.get("group_size"),
            "dominant_ratio": info.get("ratio"),
            "dominant_margin": info.get("dominant_margin"),
            "dominant_count": info.get("dominant_count"),
            "second_count": info.get("second_count"),
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

    info = mapping[key]
    if not _context_info_passes_candidate_guard(info):
        return None

    dominant_val = info.get("dominant_value")

    if tgt_val != dominant_val:
        return {
            "row_id": row_idx,
            "column": rhs,
            "value": tgt_val,
            "rule": rule,
            "reason": (
                f"context pair high-purity mismatch: {lhs}={vals} suggests {rhs}={dominant_val}, "
                f"observed={tgt_val}, group_size={info.get('group_size')}, ratio={info.get('ratio')}"
            ),
            "expected_value": dominant_val,
            "group_size": info.get("group_size"),
            "dominant_ratio": info.get("ratio"),
            "dominant_margin": info.get("dominant_margin"),
            "dominant_count": info.get("dominant_count"),
            "second_count": info.get("second_count"),
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

    if pattern and re.fullmatch(pattern, str(val).strip()) is None:
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": f"value does not match regex: {pattern}",
        }

    return None


# ============================================================
# 7. Soccer schema/domain checks
# ============================================================

def check_birthyear_numeric_range_rule(row_idx, row, rule):
    col = rule["column"]
    val = row.get(col)

    if val is None:
        return None

    year = parse_year(val)
    lb = int(rule.get("lower_bound", BIRTHYEAR_MIN))
    ub = int(rule.get("upper_bound", BIRTHYEAR_MAX))
    regex = rule.get("regex")

    if year is None:
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": f"birthyear is not an integer year: {val}",
        }

    if regex and re.fullmatch(regex, str(val).strip()) is None:
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": f"birthyear does not match regex {regex}: {val}",
        }

    if year < lb or year > ub:
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": f"birthyear outside range: {year} not in [{lb}, {ub}]",
        }

    return None


def check_season_numeric_range_rule(row_idx, row, rule):
    col = rule["column"]
    val = row.get(col)

    if val is None:
        return None

    year = parse_year(val)
    lb = int(rule.get("lower_bound", SEASON_MIN))
    ub = int(rule.get("upper_bound", SEASON_MAX))
    regex = rule.get("regex")

    if year is None:
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": f"season is not an integer year: {val}",
        }

    if regex and re.fullmatch(regex, str(val).strip()) is None:
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": f"season does not match regex {regex}: {val}",
        }

    if year < lb or year > ub:
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": f"season outside range: {year} not in [{lb}, {ub}]",
        }

    return None


def check_position_domain_rule(row_idx, row, rule):
    col = rule["column"]
    val = row.get(col)

    if val is None:
        return None

    valid_values = set(rule.get("valid_values", [])) or VALID_POSITIONS
    canonical_map = rule.get("canonical_map", {}) or POSITION_CANONICAL_MAP
    s = str(val).strip()

    if s in canonical_map:
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": f"position should be canonicalized: observed={s}, canonical={canonical_map[s]}",
            "expected_value": canonical_map[s],
        }

    if s not in valid_values:
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": f"position outside known domain: observed={val}, valid={sorted(list(valid_values))}",
        }

    return None


def check_birthyear_season_age_consistency_rule(row_idx, row, rule):
    birthyear = row.get("birthyear")
    season = row.get("season")

    if birthyear is None or season is None:
        return None

    by = parse_year(birthyear)
    sy = parse_year(season)

    if by is None or sy is None:
        return None

    min_age = int(rule.get("min_age", PLAYER_MIN_AGE_AT_SEASON))
    max_age = int(rule.get("max_age", PLAYER_MAX_AGE_AT_SEASON))
    age = sy - by

    if age < min_age or age > max_age:
        target_col = rule.get("target_column", "birthyear")
        return {
            "row_id": row_idx,
            "column": target_col,
            "value": row.get(target_col),
            "rule": rule,
            "reason": (
                f"implausible player age at season: season={sy}, birthyear={by}, "
                f"age={age}, expected_range=[{min_age}, {max_age}]"
            ),
        }

    return None


def check_team_context_dominant_rule(row_idx, row, rule):
    lhs = rule.get("lhs", [])
    rhs = rule.get("rhs")

    if rhs is None or not lhs:
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
    mapping = rule.get("_team_context_mapping", {})

    if key not in mapping:
        return None

    info = mapping[key]
    if not _context_info_passes_candidate_guard(info):
        return None

    dominant_val = info.get("dominant_value")

    if tgt_val != dominant_val:
        return {
            "row_id": row_idx,
            "column": rhs,
            "value": tgt_val,
            "rule": rule,
            "reason": (
                f"team context high-purity mismatch: lhs={lhs}={vals} suggests "
                f"{rhs}={dominant_val}, observed={tgt_val}, "
                f"ratio={safe_float(info.get('ratio'), 0.0):.6f}, support={safe_int(info.get('group_size'), 0)}"
            ),
            "expected_value": dominant_val,
            "group_size": info.get("group_size"),
            "dominant_ratio": info.get("ratio"),
            "dominant_margin": info.get("dominant_margin"),
            "dominant_count": info.get("dominant_count"),
            "second_count": info.get("second_count"),
        }

    return None


# ============================================================
# 8. 主流程
# ============================================================

def shrink_search_space(input_csv, rule_json, output_jsonl, output_csv):
    df = pd.read_csv(input_csv, dtype=str)
    df = maybe_drop_unnamed_columns(df)

    expected_cols = set(SOCCER_COLUMNS)
    actual_cols = set(df.columns)
    missing_cols = sorted(list(expected_cols - actual_cols))
    extra_cols = sorted(list(actual_cols - expected_cols))
    if missing_cols:
        print(f"[WARN] Missing expected soccer columns: {missing_cols}")
    if extra_cols:
        print(f"[WARN] Extra columns in input: {extra_cols}")

    for col in df.columns:
        df[col] = normalize_series(df[col])

    rules = load_rule_pool(rule_json)
    rules = filter_rules_with_precision_guard(rules)

    not_null_rules = [r for r in rules if r.get("rule_type") == "not_null"]
    pattern_rules = [r for r in rules if r.get("rule_type") == "pattern"]

    # v2: 只准备强 functional_dependency；soft-FD 在 precision guard 中被过滤。
    fd_like_rules = prepare_fd_rule_runtime(df, rules)

    # 下列弱规则默认不会进入 rules；这里保留变量是为了兼容旧规则池，但检查函数返回 None。
    rare_value_rules = [r for r in rules if r.get("rule_type") == "rare_value"]
    rare_pattern_rules = [r for r in rules if r.get("rule_type") == "rare_pattern"]
    global_dominant_rules = [r for r in rules if r.get("rule_type") == "global_dominant_value"]

    context_dominant_rules = prepare_context_dominant_rule_runtime(rules)
    pair_context_rules = prepare_pair_context_rule_runtime(rules)
    team_context_rules = prepare_team_context_rule_runtime(rules)

    high_risk_missing_rules = [r for r in rules if r.get("rule_type") == "high_risk_missing"]
    paired_missing_rules = [r for r in rules if r.get("rule_type") == "paired_missing_consistency"]

    regex_rules = [r for r in rules if r.get("rule_type") in {
        "generic_regex",
        "name_pattern",
        "surname_pattern",
        "manager_name_pattern",
        "team_name_pattern",
        "birthplace_location_pattern",
        "city_location_pattern",
        "stadium_name_pattern",
    }]

    birthyear_rules = [r for r in rules if r.get("rule_type") == "birthyear_numeric_range"]
    season_rules = [r for r in rules if r.get("rule_type") == "season_numeric_range"]
    position_rules = [r for r in rules if r.get("rule_type") == "position_domain"]
    birthyear_season_rules = [r for r in rules if r.get("rule_type") == "birthyear_season_age_consistency"]

    print("[RuntimeRules] active rule counts:")
    print(f"  not_null_rules: {len(not_null_rules)}")
    print(f"  high_risk_missing_rules: {len(high_risk_missing_rules)}")
    print(f"  regex_rules: {len(regex_rules)}")
    print(f"  pattern_rules: {len(pattern_rules)}")
    print(f"  birthyear_rules: {len(birthyear_rules)}")
    print(f"  season_rules: {len(season_rules)}")
    print(f"  position_rules: {len(position_rules)}")
    print(f"  birthyear_season_rules: {len(birthyear_season_rules)}")
    print(f"  fd_like_rules(functional only): {len(fd_like_rules)}")
    print(f"  team_context_rules: {len(team_context_rules)}")
    print(f"  context_dominant_rules: {len(context_dominant_rules)}")
    print(f"  pair_context_rules: {len(pair_context_rules)}")

    cell_map = {}

    for row_idx, row in df.iterrows():
        for group, fn in [
            (not_null_rules, check_not_null_rule),
            (high_risk_missing_rules, check_high_risk_missing_rule),

            (regex_rules, check_regex_rule),
            (pattern_rules, check_pattern_rule),

            (birthyear_rules, check_birthyear_numeric_range_rule),
            (season_rules, check_season_numeric_range_rule),
            (position_rules, check_position_domain_rule),
            (birthyear_season_rules, check_birthyear_season_age_consistency_rule),

            # v2: FD/context 放在最后，而且已经被严格过滤。
            (fd_like_rules, check_fd_like_rule),
            (context_dominant_rules, check_context_dominant_rule),
            (pair_context_rules, check_pair_context_rule),
            (team_context_rules, check_team_context_dominant_rule),

            # 兼容旧规则，默认不会产生 violation。
            (rare_value_rules, check_rare_value_rule),
            (rare_pattern_rules, check_rare_pattern_rule),
            (global_dominant_rules, check_global_dominant_rule),
            (paired_missing_rules, check_paired_missing_consistency_rule),
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
    rule_type_counter = Counter()
    column_counter = Counter()

    for item in candidates:
        column_counter[item["column"]] += 1
        for r in item["violated_rules"]:
            rule_type_counter[r.get("rule_type", "unknown")] += 1

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

    print("[Candidate Columns]")
    for k, v in column_counter.most_common(30):
        print(f"  - {k}: {v}")

    print("[Candidate Rule Types]")
    for k, v in rule_type_counter.most_common(30):
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    shrink_search_space(INPUT_CSV, INPUT_RULE_JSON, OUTPUT_JSONL, OUTPUT_CSV)

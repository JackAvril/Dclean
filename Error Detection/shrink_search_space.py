#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unified_shrink_search_space.py

统一版候选范围缩减脚本。

目标：
1. 用一份代码覆盖 hospital / flights / beers / rayyan / adult / soccer。
2. 保持最终输出格式不变：
   - candidate_cells_*.jsonl
   - candidate_cells_*.csv
   CSV 字段仍为：
   row_id, column, value, violation_count, conflict_score,
   rule_ids, rule_types, priorities, usage_roles, reasons
3. 将公共逻辑抽象为统一规则执行引擎，将数据集差异放入 DatasetConfig 和 special checkers。
4. 为后续消融实验预留统一开关：
   - --disable-rule-types
   - --disable-weak-rules
   - --disable-fd
   - --disable-context
   - --disable-domain-format
   - 环境变量 ABLATION_MODE / DISABLE_RULE_TYPES 等

运行示例：
python unified_shrink_search_space.py --dataset soccer
python unified_shrink_search_space.py --dataset adult --input_csv ... --rule_json ... --output_csv ... --output_jsonl ...
python unified_shrink_search_space.py --dataset all

重要说明：
- 这是一份抽象统一版，不是简单嵌入六份原始代码。
- 为尽量保持输出一致，规则权重、输出字段、排序方式、主流 rule_type 检查逻辑均按原 shrink 脚本统一实现。
- 如果某个数据集原脚本里还有极少数特殊规则，本脚本会通过 fallback regex/domain/numeric 逻辑处理；建议你用 --compare 与原输出做一次 diff。
"""

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd


# ============================================================
# 1. 通用配置
# ============================================================

BASE_MISSING_TOKENS = {
    "", "empty", "null", "none", "nan", "na", "n/a", "missing", "unk",
    "unknown", "?", "not available", "contact airline"
}

PRIORITY_WEIGHT_DEFAULT = {"high": 1.5, "medium": 1.0, "low": 0.6}

COMMON_RULE_TYPE_WEIGHT = {
    "not_null": 1.2,
    "numeric_range": 0.8,
    "enumeration_domain": 0.9,
    "pattern": 0.7,
    "generic_regex": 0.8,

    "functional_dependency": 1.5,
    "soft_functional_dependency": 1.1,
    "numeric_outlier_iqr": 0.9,
    "rare_value": 0.7,
    "rare_pattern": 0.7,
    "global_dominant_value": 0.6,
    "dominant_value_by_context": 0.9,
    "dominant_value_by_context_pair": 1.0,
    "high_risk_missing": 0.9,
    "paired_missing_consistency": 0.8,
    "dominant_value_typo": 0.95,

    # hospital
    "score_format": 0.85,
    "sample_format": 0.85,
    "sample_patients_suffix": 0.85,

    # flights
    "time_format": 0.85,
    "datetime_time_format": 0.85,
    "time_loose_pattern": 0.8,
    "temporal_pair_consistency": 0.95,
    "delay_window": 0.95,
    "duration_consistency_window": 1.0,
    "actual_time_global_candidate": 0.45,

    # beers
    "ounces_unit_format": 0.85,
    "ounces_loose_pattern": 0.75,
    "abv_numeric_format": 0.85,
    "ibu_numeric_or_na_format": 0.80,
    "state_abbr_format": 0.90,
    "numeric_range_window": 0.95,
    "ounces_canonical_format": 1.60,
    "abv_canonical_format": 1.55,
    "city_state_suffix_pollution": 1.45,

    # rayyan
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

    # adult
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
    "age_education_consistency": 1.05,
    "age_relationship_consistency": 1.00,
    "marital_relationship_consistency": 1.25,
    "sex_relationship_consistency": 1.10,
    "workclass_occupation_consistency": 0.95,
    "education_occupation_consistency": 0.95,
    "hours_workclass_consistency": 0.85,
    "relationship_marital_spouse_conflict": 1.45,
    "relationship_age_spouse_conflict": 1.40,
    "relationship_sex_spouse_conflict": 1.25,
    "relationship_context_dominant": 1.15,
    "education_age_extreme_conflict": 1.10,

    # soccer
    "birthyear_numeric_range": 1.45,
    "season_numeric_range": 1.35,
    "position_domain": 1.45,
    "name_pattern": 1.05,
    "surname_pattern": 1.05,
    "manager_name_pattern": 1.00,
    "team_name_pattern": 0.95,
    "birthplace_location_pattern": 1.05,
    "city_location_pattern": 0.95,
    "stadium_name_pattern": 0.95,
    "birthyear_season_age_consistency": 1.45,
    "team_context_dominant": 0.35,
    "team_season_manager_consistency_hint": 0.30,
    "city_stadium_consistency_hint": 0.30,
}


# ============================================================
# 2. 数据集配置
# ============================================================

@dataclass
class DatasetConfig:
    name: str
    input_csv: str
    input_rule_json: str
    output_jsonl: str
    output_csv: str

    missing_tokens: Set[str] = field(default_factory=lambda: set(BASE_MISSING_TOKENS))
    lower_normalize: bool = True

    skip_null_for_non_notnull_rules: bool = True
    max_fd_mapping_size: int = 200000
    max_context_mapping_size: int = 200000

    priority_weight: Dict[str, float] = field(default_factory=lambda: dict(PRIORITY_WEIGHT_DEFAULT))
    rule_type_weight: Dict[str, float] = field(default_factory=lambda: dict(COMMON_RULE_TYPE_WEIGHT))

    # rule filtering / precision guard
    enable_precision_guard: bool = False
    skip_evidence_only_rules: bool = False
    disabled_candidate_rule_types: Set[str] = field(default_factory=set)
    global_dominant_skip_columns: Set[str] = field(default_factory=set)
    rare_skip_columns: Set[str] = field(default_factory=set)
    context_only_columns: Set[str] = field(default_factory=set)
    primary_target_columns: Set[str] = field(default_factory=set)
    weak_rule_types_for_context_only: Set[str] = field(default_factory=set)
    allowed_context_only_strong_rule_types: Set[str] = field(default_factory=set)

    # soccer strict FD/context options
    fd_candidate_min_confidence: float = 0.0
    fd_candidate_min_support: int = 0
    context_candidate_min_confidence: float = 0.0
    context_candidate_min_support: int = 0

    # domain / regex / special options
    domain_values: Dict[str, Set[str]] = field(default_factory=dict)
    regex_by_rule_type: Dict[str, str] = field(default_factory=dict)
    regex_by_column: Dict[str, str] = field(default_factory=dict)

    # execution order: list of rule_type groups, each group uses same checker dispatch.
    runtime_order: List[str] = field(default_factory=list)


def build_dataset_configs() -> Dict[str, DatasetConfig]:
    cfgs: Dict[str, DatasetConfig] = {}

    cfgs["hospital"] = DatasetConfig(
        name="hospital",
        input_csv="/mnt/mydata/dq/projects/Splittree/hospital/hospital_dirty_no_address23.csv",
        input_rule_json="rule_pool_high_recall_v2.json",
        output_jsonl="candidate_cells_v2.jsonl",
        output_csv="candidate_cells_v2.csv",
        missing_tokens={"", "empty", "null", "none", "nan", "na", "n/a", "missing", "unk"},
        lower_normalize=False,
        runtime_order=[
            "not_null", "numeric_range", "enumeration_domain", "pattern", "__fd_like__",
            "numeric_outlier_iqr", "rare_value", "rare_pattern", "global_dominant_value",
            "dominant_value_by_context", "high_risk_missing", "paired_missing_consistency",
            "dominant_value_typo", "score_format", "sample_format", "sample_patients_suffix",
        ],
    )

    # flights：用户上传的第二份 shrink 代码中是时间规则逻辑；这里按 flights 路径/输出命名配置。
    cfgs["flights"] = DatasetConfig(
        name="flights",
        input_csv="/mnt/mydata/dq/projects/Splittree/flights/flights_dirty.csv",
        input_rule_json="rule_pool_flights.json",
        output_jsonl="candidate_cells_flights.jsonl",
        output_csv="candidate_cells_flights.csv",
        lower_normalize=True,
        runtime_order=[
            "not_null", "pattern", "__fd_like__", "rare_value", "rare_pattern",
            "global_dominant_value", "dominant_value_by_context", "dominant_value_by_context_pair",
            "high_risk_missing", "paired_missing_consistency",
            "time_format", "datetime_time_format", "time_loose_pattern",
            "temporal_pair_consistency", "delay_window",
            "duration_consistency_window", "actual_time_global_candidate",
        ],
    )

    cfgs["beers"] = DatasetConfig(
        name="beers",
        input_csv="/mnt/mydata/dq/projects/Splittree/beers/beers_dirty.csv",
        input_rule_json="rule_pool_beers.json",
        output_jsonl="candidate_cells_beers.jsonl",
        output_csv="candidate_cells_beers.csv",
        lower_normalize=True,
        runtime_order=[
            "not_null", "pattern", "__fd_like__", "rare_value", "rare_pattern",
            "global_dominant_value", "dominant_value_by_context", "dominant_value_by_context_pair",
            "high_risk_missing", "paired_missing_consistency",
            "ounces_unit_format", "ounces_loose_pattern", "abv_numeric_format",
            "ibu_numeric_or_na_format", "state_abbr_format", "numeric_range_window",
            "ounces_canonical_format", "abv_canonical_format", "city_state_suffix_pollution",
        ],
    )

    cfgs["rayyan"] = DatasetConfig(
        name="rayyan",
        input_csv="/mnt/mydata/dq/projects/Splittree/rayyan/rayyan_dirty.csv",
        input_rule_json="rule_pool_rayyan.json",
        output_jsonl="candidate_cells_rayyan.jsonl",
        output_csv="candidate_cells_rayyan.csv",
        lower_normalize=True,
        runtime_order=[
            "not_null", "pattern", "__fd_like__", "rare_value", "rare_pattern",
            "global_dominant_value", "dominant_value_by_context", "dominant_value_by_context_pair",
            "high_risk_missing", "paired_missing_consistency",
            "issn_format", "issn_loose_format", "issn_canonical_format",
            "language_canonical_format", "date_like_format",
            "date_component_permutation_suspicious", "date_ymd_swap_suspicious",
            "volume_issue_numeric_format", "pagination_format",
            "author_list_braced_format", "author_list_encoding_corruption",
        ],
    )

    adult_primary = {"sex", "relationship", "education"}
    adult_context_only = {
        "age", "workclass", "maritalstatus", "occupation", "race",
        "hoursperweek", "country", "income"
    }
    cfgs["adult"] = DatasetConfig(
        name="adult",
        input_csv="/mnt/mydata/dq/projects/Splittree/adult/adult_dirty.csv",
        input_rule_json="rule_pool_adult.json",
        output_jsonl="candidate_cells_adult.jsonl",
        output_csv="candidate_cells_adult.csv",
        lower_normalize=False,
        enable_precision_guard=True,
        primary_target_columns=adult_primary,
        context_only_columns=adult_context_only,
        global_dominant_skip_columns={"workclass", "country", "income", "race"},
        rare_skip_columns={"country", "occupation", "workclass"},
        weak_rule_types_for_context_only={
            "not_null", "high_risk_missing", "paired_missing_consistency",
            "rare_value", "rare_pattern", "global_dominant_value",
            "dominant_value_by_context", "dominant_value_by_context_pair",
            "functional_dependency", "soft_functional_dependency",
            "age_education_consistency", "age_relationship_consistency",
            "marital_relationship_consistency", "sex_relationship_consistency",
            "workclass_occupation_consistency", "education_occupation_consistency",
            "hours_workclass_consistency",
        },
        allowed_context_only_strong_rule_types={
            "age_bucket_format", "hours_per_week_format_range",
            "workclass_domain", "maritalstatus_domain", "occupation_domain",
            "race_domain", "country_domain", "income_domain", "income_canonical_format",
        },
        runtime_order=[
            "not_null", "pattern", "__fd_like__", "rare_value", "rare_pattern",
            "global_dominant_value", "dominant_value_by_context", "dominant_value_by_context_pair",
            "relationship_context_dominant", "high_risk_missing", "paired_missing_consistency",
            "generic_regex", "age_bucket_format", "hours_per_week_format_range",
            "__adult_domain__", "income_canonical_format",
            "age_education_consistency", "age_relationship_consistency",
            "marital_relationship_consistency", "sex_relationship_consistency",
            "relationship_marital_spouse_conflict", "relationship_age_spouse_conflict",
            "relationship_sex_spouse_conflict", "education_age_extreme_conflict",
            "workclass_occupation_consistency", "education_occupation_consistency",
            "hours_workclass_consistency",
        ],
    )

    soccer_cols = {
        "name", "surname", "birthyear", "birthplace", "position",
        "team", "city", "stadium", "season", "manager"
    }
    cfgs["soccer"] = DatasetConfig(
        name="soccer",
        input_csv="/mnt/mydata/dq/projects/Splittree/soccer/soccer_dirty.csv",
        input_rule_json="rule_pool_soccer.json",
        output_jsonl="candidate_cells_soccer.jsonl",
        output_csv="candidate_cells_soccer.csv",
        lower_normalize=False,
        enable_precision_guard=True,
        skip_evidence_only_rules=True,
        primary_target_columns=soccer_cols,
        disabled_candidate_rule_types={
            "soft_functional_dependency", "rare_value", "rare_pattern", "global_dominant_value",
            "dominant_value_by_context", "dominant_value_by_context_pair",
            "team_context_dominant", "team_season_manager_consistency_hint",
            "city_stadium_consistency_hint",
        },
        global_dominant_skip_columns=soccer_cols,
        rare_skip_columns=soccer_cols,
        fd_candidate_min_confidence=0.995,
        fd_candidate_min_support=50,
        context_candidate_min_confidence=0.98,
        context_candidate_min_support=50,
        runtime_order=[
            "not_null", "high_risk_missing",
            "generic_regex", "name_pattern", "surname_pattern", "manager_name_pattern",
            "team_name_pattern", "birthplace_location_pattern", "city_location_pattern",
            "stadium_name_pattern", "pattern",
            "birthyear_numeric_range", "season_numeric_range", "position_domain",
            "birthyear_season_age_consistency",
            "__fd_like__", "dominant_value_by_context", "dominant_value_by_context_pair",
            "team_context_dominant",
            "rare_value", "rare_pattern", "global_dominant_value", "paired_missing_consistency",
        ],
    )

    return cfgs


DATASET_CONFIGS = build_dataset_configs()


# ============================================================
# 3. 通用工具
# ============================================================

def env_flag(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def env_list(name: str) -> List[str]:
    v = os.getenv(name, "")
    return [x.strip() for x in v.split(",") if x.strip()]


def is_missing_value(x: Any, cfg: DatasetConfig) -> bool:
    if pd.isna(x):
        return True
    return str(x).strip().lower() in cfg.missing_tokens


def normalize_value(x: Any, cfg: DatasetConfig):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.lower() in cfg.missing_tokens:
        return None
    return s.lower() if cfg.lower_normalize else s


def normalize_series(s: pd.Series, cfg: DatasetConfig) -> pd.Series:
    return s.map(lambda x: normalize_value(x, cfg))


def maybe_drop_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [c for c in df.columns if not str(c).lower().startswith("unnamed:")]
    return df[keep_cols].copy()


def detect_basic_pattern(value: Any, cfg: Optional[DatasetConfig] = None) -> str:
    if pd.isna(value) or value is None:
        return "NULL"

    # rayyan 允许 [] {} " 作为 S；adult/soccer 支持 < >
    extra_symbols = set()
    if cfg and cfg.name == "rayyan":
        extra_symbols |= {"[", "]", "{", "}", '"'}
    if cfg and cfg.name in {"adult", "soccer"}:
        extra_symbols |= {"<", ">"}

    symbols = {" ", "-", "_", "/", ".", ",", "(", ")", "%", ":", ";", "&", "'", "+"} | extra_symbols

    parts, prev_type, cnt = [], None, 0
    for ch in str(value):
        if ch.isdigit():
            cur = "D"
        elif ch.isalpha():
            cur = "L"
        elif ch in symbols:
            cur = "S"
        else:
            cur = "X"

        if cur == prev_type:
            cnt += 1
        else:
            if prev_type is not None:
                parts.append(f"{prev_type}{cnt}")
            prev_type, cnt = cur, 1

    if prev_type is not None:
        parts.append(f"{prev_type}{cnt}")
    return "".join(parts)


def safe_float(x: Any, default=0.0) -> float:
    try:
        if x is None or pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x: Any, default=0) -> int:
    try:
        if x is None or pd.isna(x):
            return default
        return int(float(x))
    except Exception:
        return default


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


def try_parse_one_numeric(value: Any) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    s = str(value).replace(",", "").replace("%", "")
    try:
        return float(s)
    except Exception:
        return None


def load_rule_pool(rule_json_path: str) -> List[Dict[str, Any]]:
    with open(rule_json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    rules = []
    for _, group_rules in obj.get("rules", {}).items():
        for rule in group_rules:
            rules.append(dict(rule))
    return rules


def rule_weight(rule: Dict[str, Any], cfg: DatasetConfig) -> float:
    conf = safe_float(rule.get("confidence", rule.get("overall_confidence", 0.0)), 0.0)
    supp = safe_int(rule.get("support", 0), 0)
    priority = rule.get("priority", "low")
    rule_type = rule.get("rule_type", "")

    priority_w = cfg.priority_weight.get(priority, 0.6)
    type_w = cfg.rule_type_weight.get(rule_type, 1.0)
    support_part = math.log1p(max(supp, 1))
    return round((0.1 + conf) * support_part * priority_w * type_w, 6)


def normalized_edit_distance_ratio(a: str, b: str) -> float:
    if a is None or b is None:
        return 1.0
    a, b = str(a), str(b)
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
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = tmp
    return dp[m] / max(n, m)


def add_violation(cell_map: Dict[Tuple[int, str], Dict[str, Any]], violation: Dict[str, Any], cfg: DatasetConfig):
    row_id = int(violation["row_id"])
    column = violation["column"]
    key = (row_id, column)
    rule = violation["rule"]

    # 候选级 precision guard，主要用于 adult/soccer。
    if should_drop_violation_by_precision_guard(violation, cfg):
        return

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
    cell_map[key]["conflict_score"] += rule_weight(rule, cfg)


# ============================================================
# 4. Runtime rule preparation
# ============================================================

def get_rule_target_column(rule: Dict[str, Any]) -> Optional[str]:
    return rule.get("target_column") or rule.get("column") or rule.get("rhs")


def is_evidence_only_rule(rule: Dict[str, Any], cfg: DatasetConfig) -> bool:
    if not cfg.skip_evidence_only_rules:
        return False
    usage_role = str(rule.get("usage_role", "")).strip().lower()
    if usage_role == "evidence_only":
        return True
    if rule.get("allow_candidate_generation", None) is False:
        return True
    trigger = str(rule.get("candidate_trigger", "")).strip().lower()
    return trigger in {"disabled", "disabled_evidence_only", "evidence_only"}


def should_drop_rule_by_precision_guard(rule: Dict[str, Any], cfg: DatasetConfig) -> bool:
    if not cfg.enable_precision_guard:
        return False

    rtype = rule.get("rule_type", "")
    target_col = get_rule_target_column(rule)

    if is_evidence_only_rule(rule, cfg):
        return True

    if rtype in cfg.disabled_candidate_rule_types:
        # soccer: allow extremely strict functional_dependency via separate path
        if rtype == "functional_dependency":
            return not is_fd_candidate_rule_allowed(rule, cfg)
        return True

    if target_col in cfg.context_only_columns and rtype in cfg.weak_rule_types_for_context_only:
        return True

    if target_col in cfg.context_only_columns and rtype not in cfg.allowed_context_only_strong_rule_types:
        # Adult context-only 列只允许明确 domain/format 强规则作为候选。
        if cfg.name == "adult":
            return True

    if rtype == "global_dominant_value" and (not target_col or target_col in cfg.global_dominant_skip_columns):
        return True

    if rtype in {"rare_value", "rare_pattern"} and (not target_col or target_col in cfg.rare_skip_columns):
        return True

    if cfg.name == "soccer":
        if rtype == "functional_dependency":
            return not is_fd_candidate_rule_allowed(rule, cfg)
        if rtype in {"dominant_value_by_context", "dominant_value_by_context_pair", "team_context_dominant"}:
            return not is_context_candidate_rule_allowed(rule, cfg)

    return False


def should_drop_violation_by_precision_guard(violation: Dict[str, Any], cfg: DatasetConfig) -> bool:
    if not cfg.enable_precision_guard:
        return False

    rule = violation.get("rule", {})
    rtype = rule.get("rule_type", "")
    col = violation.get("column")

    if cfg.name == "adult":
        if col in cfg.context_only_columns and rtype not in cfg.allowed_context_only_strong_rule_types:
            return True

    if cfg.name == "soccer":
        if rtype in cfg.disabled_candidate_rule_types:
            return True
        if is_evidence_only_rule(rule, cfg):
            return True

    return False


def is_fd_candidate_rule_allowed(rule: Dict[str, Any], cfg: DatasetConfig) -> bool:
    if rule.get("rule_type") != "functional_dependency":
        return False
    if is_evidence_only_rule(rule, cfg):
        return False
    rhs = rule.get("rhs") or rule.get("target_column")
    if cfg.primary_target_columns and rhs not in cfg.primary_target_columns:
        return False
    if safe_float(rule.get("confidence", 0.0), 0.0) < cfg.fd_candidate_min_confidence:
        return False
    if safe_int(rule.get("support", 0), 0) < cfg.fd_candidate_min_support:
        return False
    return True


def is_context_candidate_rule_allowed(rule: Dict[str, Any], cfg: DatasetConfig) -> bool:
    if is_evidence_only_rule(rule, cfg):
        return False
    if rule.get("allow_candidate_generation", False) is not True:
        return False
    rhs = rule.get("rhs") or rule.get("target_column")
    if cfg.primary_target_columns and rhs not in cfg.primary_target_columns:
        return False
    if safe_float(rule.get("confidence", 0.0), 0.0) < cfg.context_candidate_min_confidence:
        return False
    if safe_int(rule.get("support", 0), 0) < cfg.context_candidate_min_support:
        return False
    return True


def filter_rules(rules: List[Dict[str, Any]], cfg: DatasetConfig, args) -> List[Dict[str, Any]]:
    disabled_rule_types = set(args.disable_rule_types or [])
    disabled_rule_types.update(env_list("DISABLE_RULE_TYPES"))

    if args.disable_weak_rules or env_flag("DISABLE_WEAK_RULES", False):
        disabled_rule_types.update({
            "rare_value", "rare_pattern", "global_dominant_value",
            "dominant_value_by_context", "dominant_value_by_context_pair",
            "soft_functional_dependency",
        })
    if args.disable_fd or env_flag("DISABLE_FD", False):
        disabled_rule_types.update({"functional_dependency", "soft_functional_dependency"})
    if args.disable_context or env_flag("DISABLE_CONTEXT", False):
        disabled_rule_types.update({
            "dominant_value_by_context", "dominant_value_by_context_pair",
            "relationship_context_dominant", "team_context_dominant",
            "team_season_manager_consistency_hint", "city_stadium_consistency_hint",
        })
    if args.disable_domain_format or env_flag("DISABLE_DOMAIN_FORMAT", False):
        disabled_rule_types.update({
            "pattern", "generic_regex",
            "score_format", "sample_format", "sample_patients_suffix",
            "time_format", "datetime_time_format", "time_loose_pattern",
            "ounces_unit_format", "ounces_loose_pattern", "abv_numeric_format",
            "ibu_numeric_or_na_format", "state_abbr_format",
            "issn_format", "issn_loose_format", "issn_canonical_format",
            "language_canonical_format", "date_like_format",
            "volume_issue_numeric_format", "pagination_format",
            "author_list_braced_format", "author_list_encoding_corruption",
            "age_bucket_format", "hours_per_week_format_range", "income_canonical_format",
            "birthyear_numeric_range", "season_numeric_range", "position_domain",
            "name_pattern", "surname_pattern", "manager_name_pattern", "team_name_pattern",
            "birthplace_location_pattern", "city_location_pattern", "stadium_name_pattern",
        })

    out = []
    for r in rules:
        if r.get("rule_type") in disabled_rule_types:
            continue
        if should_drop_rule_by_precision_guard(r, cfg):
            continue
        out.append(r)
    return out


def build_fd_majority_mapping(df: pd.DataFrame, lhs: List[str], rhs: str, cfg: DatasetConfig) -> Dict[Tuple[Any, ...], Tuple[Any, int, int]]:
    needed = lhs + [rhs]
    if any(c not in df.columns for c in needed):
        return {}
    sub = df[needed].dropna()
    if len(sub) == 0:
        return {}

    mapping = {}
    for group_key, grp in sub.groupby(lhs, dropna=True):
        rhs_vals = grp[rhs].tolist()
        cnt = Counter(rhs_vals)
        majority_val, majority_cnt = cnt.most_common(1)[0]
        total_cnt = len(rhs_vals)
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        mapping[group_key] = (majority_val, majority_cnt, total_cnt)
        if len(mapping) >= cfg.max_fd_mapping_size:
            break
    return mapping


def prepare_fd_rules(df: pd.DataFrame, rules: List[Dict[str, Any]], cfg: DatasetConfig) -> List[Dict[str, Any]]:
    out = []
    for rule in rules:
        if rule.get("rule_type") not in {"functional_dependency", "soft_functional_dependency"}:
            continue
        if should_drop_rule_by_precision_guard(rule, cfg):
            continue
        lhs = rule.get("lhs", [])
        rhs = rule.get("rhs")
        if not lhs or not rhs:
            continue
        r = dict(rule)
        r["_majority_mapping"] = build_fd_majority_mapping(df, list(lhs), rhs, cfg)
        out.append(r)
    return out


def prepare_context_rules(rules: List[Dict[str, Any]], cfg: DatasetConfig) -> List[Dict[str, Any]]:
    out = []
    for rule in rules:
        if rule.get("rule_type") not in {
            "dominant_value_by_context", "relationship_context_dominant", "team_context_dominant"
        }:
            continue

        mapping = {}

        # Older rule format: mappings list with context_value
        for item in rule.get("mappings", []) or []:
            ctx_val = item.get("context_value")
            mapping[ctx_val] = (
                item.get("dominant_value"),
                safe_int(item.get("support", 0), 0),
                safe_float(item.get("dominant_ratio", 0.0), 0.0),
            )
            if len(mapping) >= cfg.max_context_mapping_size:
                break

        # Newer rule format: majority_map dict keyed by "a||b||c"
        for key, info in (rule.get("majority_map", {}) or {}).items():
            mapping[key] = (
                info.get("dominant_value"),
                safe_int(info.get("group_size", info.get("support", 0)), 0),
                safe_float(info.get("ratio", info.get("dominant_ratio", 0.0)), 0.0),
            )
            if len(mapping) >= cfg.max_context_mapping_size:
                break

        r = dict(rule)
        r["_context_mapping"] = mapping
        out.append(r)
    return out


def prepare_pair_context_rules(rules: List[Dict[str, Any]], cfg: DatasetConfig) -> List[Dict[str, Any]]:
    out = []
    for rule in rules:
        if rule.get("rule_type") != "dominant_value_by_context_pair":
            continue
        mapping = {}
        for key, info in (rule.get("majority_map", {}) or {}).items():
            mapping[key] = (
                info.get("dominant_value"),
                safe_int(info.get("group_size", info.get("support", 0)), 0),
                safe_float(info.get("ratio", info.get("dominant_ratio", 0.0)), 0.0),
            )
        r = dict(rule)
        r["_pair_context_mapping"] = mapping
        out.append(r)
    return out


# ============================================================
# 5. 通用规则 checker
# ============================================================

def violation(row_idx: int, col: str, value: Any, rule: Dict[str, Any], reason: str, expected_value: Any = None) -> Dict[str, Any]:
    d = {
        "row_id": row_idx,
        "column": col,
        "value": value,
        "rule": rule,
        "reason": reason,
    }
    if expected_value is not None:
        d["expected_value"] = expected_value
    return d


def check_not_null(row_idx: int, row: pd.Series, rule: Dict[str, Any], cfg: DatasetConfig):
    col = rule.get("column")
    if not col or col not in row:
        return None
    val = row.get(col)
    if pd.isna(val) or val is None:
        return violation(row_idx, col, val, rule, "value is null but rule requires non-null")
    return None


def check_numeric_range(row_idx: int, row: pd.Series, rule: Dict[str, Any], cfg: DatasetConfig):
    col = rule.get("column")
    if not col or col not in row:
        return None
    val = row.get(col)
    if (pd.isna(val) or val is None) and cfg.skip_null_for_non_notnull_rules:
        return None
    num = try_parse_one_numeric(val)
    if num is None:
        return violation(row_idx, col, val, rule, "value cannot be parsed as numeric for numeric_range rule")
    rng = rule.get("range", {})
    min_val = rng.get("min", rule.get("lower_bound", rule.get("min")))
    max_val = rng.get("max", rule.get("upper_bound", rule.get("max")))
    if min_val is None or max_val is None:
        return None
    if not (float(min_val) <= num <= float(max_val)):
        return violation(row_idx, col, val, rule, f"value {num} is outside range [{min_val}, {max_val}]")
    return None


def check_enumeration(row_idx: int, row: pd.Series, rule: Dict[str, Any], cfg: DatasetConfig):
    col = rule.get("column")
    if not col or col not in row:
        return None
    val = row.get(col)
    if (pd.isna(val) or val is None) and cfg.skip_null_for_non_notnull_rules:
        return None
    allowed = set(rule.get("allowed_values", rule.get("valid_values", [])))
    if val not in allowed:
        return violation(row_idx, col, val, rule, "value not in allowed domain")
    return None


def check_pattern(row_idx: int, row: pd.Series, rule: Dict[str, Any], cfg: DatasetConfig):
    col = rule.get("column")
    if not col or col not in row:
        return None
    val = row.get(col)
    if (pd.isna(val) or val is None) and cfg.skip_null_for_non_notnull_rules:
        return None
    observed = detect_basic_pattern(val, cfg)
    target = rule.get("pattern")
    if target is None:
        return None
    if observed != target:
        return violation(row_idx, col, val, rule, f"pattern mismatch: observed={observed}, expected={target}")
    return None


def check_regex(row_idx: int, row: pd.Series, rule: Dict[str, Any], cfg: DatasetConfig):
    col = rule.get("column") or rule.get("target_column")
    if not col or col not in row:
        return None
    val = row.get(col)
    if (pd.isna(val) or val is None) and cfg.skip_null_for_non_notnull_rules:
        return None

    pattern = rule.get("regex") or cfg.regex_by_rule_type.get(rule.get("rule_type", "")) or cfg.regex_by_column.get(col)
    if not pattern:
        return None
    if re.fullmatch(pattern, str(val).strip()) is None:
        return violation(row_idx, col, val, rule, f"regex mismatch: expected={pattern}")
    return None


def check_fd_like(row_idx: int, row: pd.Series, rule: Dict[str, Any], cfg: DatasetConfig):
    lhs = rule.get("lhs", [])
    rhs = rule.get("rhs")
    if not lhs or not rhs:
        return None
    if any(c not in row for c in lhs) or rhs not in row:
        return None

    lhs_values = []
    for c in lhs:
        v = row.get(c)
        if pd.isna(v) or v is None:
            return None
        lhs_values.append(v)

    rhs_val = row.get(rhs)
    if (pd.isna(rhs_val) or rhs_val is None) and cfg.skip_null_for_non_notnull_rules:
        return None

    key = tuple(lhs_values)
    mapping = rule.get("_majority_mapping", {})
    if key not in mapping:
        return None

    expected, majority_count, total_count = mapping[key]
    if rhs_val != expected:
        return violation(
            row_idx, rhs, rhs_val, rule,
            f"FD violation: lhs={key} suggests rhs={expected}, observed={rhs_val}",
            expected_value=expected,
        )
    return None


def check_numeric_outlier_iqr(row_idx: int, row: pd.Series, rule: Dict[str, Any], cfg: DatasetConfig):
    col = rule.get("column")
    if not col or col not in row:
        return None
    val = row.get(col)
    if (pd.isna(val) or val is None) and cfg.skip_null_for_non_notnull_rules:
        return None
    num = try_parse_one_numeric(val)
    if num is None:
        return None
    lower = rule.get("lower_bound", rule.get("iqr_lower", rule.get("lower")))
    upper = rule.get("upper_bound", rule.get("iqr_upper", rule.get("upper")))
    if lower is None or upper is None:
        return None
    if num < float(lower) or num > float(upper):
        return violation(row_idx, col, val, rule, f"numeric outlier by IQR: value={num}, range=[{lower}, {upper}]")
    return None


def check_rare_value(row_idx: int, row: pd.Series, rule: Dict[str, Any], cfg: DatasetConfig):
    col = rule.get("column") or rule.get("target_column")
    if not col or col not in row:
        return None
    val = row.get(col)
    if (pd.isna(val) or val is None) and cfg.skip_null_for_non_notnull_rules:
        return None

    rare_values = set(rule.get("rare_values", []))
    if not rare_values and "value_counts" in rule:
        max_freq = safe_int(rule.get("max_freq", 2), 2)
        rare_values = {k for k, v in rule.get("value_counts", {}).items() if safe_int(v, 999999) <= max_freq}
    if val in rare_values:
        return violation(row_idx, col, val, rule, "rare value candidate")
    return None


def check_rare_pattern(row_idx: int, row: pd.Series, rule: Dict[str, Any], cfg: DatasetConfig):
    col = rule.get("column") or rule.get("target_column")
    if not col or col not in row:
        return None
    val = row.get(col)
    if (pd.isna(val) or val is None) and cfg.skip_null_for_non_notnull_rules:
        return None
    pat = detect_basic_pattern(val, cfg)
    rare_patterns = set(rule.get("rare_patterns", []))
    if pat in rare_patterns:
        return violation(row_idx, col, val, rule, f"rare pattern candidate: observed={pat}")
    return None


def check_global_dominant(row_idx: int, row: pd.Series, rule: Dict[str, Any], cfg: DatasetConfig):
    col = rule.get("column") or rule.get("target_column")
    if not col or col not in row:
        return None
    val = row.get(col)
    if (pd.isna(val) or val is None) and cfg.skip_null_for_non_notnull_rules:
        return None
    dominant = rule.get("dominant_value")
    if dominant is None:
        return None
    if val != dominant:
        return violation(row_idx, col, val, rule, f"global dominant mismatch: dominant={dominant}, observed={val}", expected_value=dominant)
    return None


def _context_key_from_rule(row: pd.Series, rule: Dict[str, Any]) -> Optional[str]:
    if "context_column" in rule:
        c = rule.get("context_column")
        if c not in row:
            return None
        return row.get(c)

    lhs = rule.get("lhs") or rule.get("context_columns")
    if lhs:
        vals = []
        for c in lhs:
            if c not in row:
                return None
            v = row.get(c)
            if pd.isna(v) or v is None:
                return None
            vals.append(str(v))
        return "||".join(vals)

    return None


def check_context_dominant(row_idx: int, row: pd.Series, rule: Dict[str, Any], cfg: DatasetConfig):
    rhs = rule.get("rhs") or rule.get("target_column") or rule.get("column")
    if not rhs or rhs not in row:
        return None
    observed = row.get(rhs)
    if (pd.isna(observed) or observed is None) and cfg.skip_null_for_non_notnull_rules:
        return None

    key = _context_key_from_rule(row, rule)
    if key is None:
        return None

    mapping = rule.get("_context_mapping", {})
    if key not in mapping:
        return None

    expected, support, ratio = mapping[key]
    only_trigger = set(rule.get("only_trigger_observed_values", []) or [])
    if only_trigger and observed not in only_trigger:
        return None

    if observed != expected:
        return violation(
            row_idx, rhs, observed, rule,
            f"context dominant mismatch: context={key}, dominant={expected}, observed={observed}",
            expected_value=expected,
        )
    return None


def check_pair_context(row_idx: int, row: pd.Series, rule: Dict[str, Any], cfg: DatasetConfig):
    rhs = rule.get("rhs") or rule.get("target_column") or rule.get("column")
    if not rhs or rhs not in row:
        return None
    observed = row.get(rhs)
    if (pd.isna(observed) or observed is None) and cfg.skip_null_for_non_notnull_rules:
        return None

    lhs = rule.get("lhs") or rule.get("context_columns")
    if not lhs:
        return None
    vals = []
    for c in lhs:
        if c not in row:
            return None
        v = row.get(c)
        if pd.isna(v) or v is None:
            return None
        vals.append(str(v))
    key = "||".join(vals)
    mapping = rule.get("_pair_context_mapping", {})
    if key not in mapping:
        return None
    expected, support, ratio = mapping[key]
    if observed != expected:
        return violation(
            row_idx, rhs, observed, rule,
            f"pair-context dominant mismatch: context={key}, dominant={expected}, observed={observed}",
            expected_value=expected,
        )
    return None


def check_high_risk_missing(row_idx: int, row: pd.Series, rule: Dict[str, Any], cfg: DatasetConfig):
    col = rule.get("target_column") or rule.get("column")
    if not col or col not in row:
        return None
    val = row.get(col)
    if pd.isna(val) or val is None:
        return violation(row_idx, col, val, rule, "high risk missing value")
    return None


def check_paired_missing(row_idx: int, row: pd.Series, rule: Dict[str, Any], cfg: DatasetConfig):
    cols = rule.get("columns", [])
    if not cols or any(c not in row for c in cols):
        return None
    vals = [row.get(c) for c in cols]
    missing = [pd.isna(v) or v is None for v in vals]
    if any(missing) and not all(missing):
        target = rule.get("target_column") or cols[missing.index(True)]
        return violation(row_idx, target, row.get(target), rule, f"paired missing inconsistency among columns={cols}")
    return None


def check_dominant_value_typo(row_idx: int, row: pd.Series, rule: Dict[str, Any], cfg: DatasetConfig):
    col = rule.get("column") or rule.get("target_column")
    if not col or col not in row:
        return None
    val = row.get(col)
    if (pd.isna(val) or val is None) and cfg.skip_null_for_non_notnull_rules:
        return None
    dominant = rule.get("dominant_value")
    max_dist_ratio = safe_float(rule.get("max_edit_distance_ratio", 0.25), 0.25)
    if dominant and val != dominant and normalized_edit_distance_ratio(str(val), str(dominant)) <= max_dist_ratio:
        return violation(row_idx, col, val, rule, f"dominant typo candidate: dominant={dominant}, observed={val}", expected_value=dominant)
    return None


# ============================================================
# 6. Hospital special checkers
# ============================================================

def looks_like_score_format(v: Optional[str]) -> bool:
    return v is not None and re.fullmatch(r"\d{1,3}%", str(v)) is not None


def split_sample_value(v: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if v is None:
        return None, None
    m = re.fullmatch(r"(\d+)\s+(.+)", str(v).strip())
    if not m:
        return None, None
    return m.group(1), m.group(2).strip()


def looks_like_sample_format(v: Optional[str]) -> bool:
    return v is not None and re.fullmatch(r"\d+\s+patients", str(v)) is not None


def check_score_format(row_idx, row, rule, cfg):
    col = rule.get("column", "Score")
    if col not in row:
        return None
    val = row.get(col)
    if (pd.isna(val) or val is None) and cfg.skip_null_for_non_notnull_rules:
        return None
    if not looks_like_score_format(val):
        return violation(row_idx, col, val, rule, "Score value should match percentage format like 98%")
    return None


def check_sample_format(row_idx, row, rule, cfg):
    col = rule.get("column", "Sample")
    if col not in row:
        return None
    val = row.get(col)
    if (pd.isna(val) or val is None) and cfg.skip_null_for_non_notnull_rules:
        return None
    if not looks_like_sample_format(val):
        return violation(row_idx, col, val, rule, "Sample value should match '<number> patients'")
    return None


def check_sample_patients_suffix(row_idx, row, rule, cfg):
    col = rule.get("column", "Sample")
    if col not in row:
        return None
    val = row.get(col)
    if (pd.isna(val) or val is None) and cfg.skip_null_for_non_notnull_rules:
        return None
    num, suffix = split_sample_value(val)
    if num is None or suffix is None:
        return None
    if suffix != "patients" and normalized_edit_distance_ratio(suffix, "patients") <= 0.3:
        return violation(row_idx, col, val, rule, "Sample suffix looks like typo of patients", expected_value=f"{num} patients")
    return None


# ============================================================
# 7. Flights special checkers
# ============================================================

def parse_time_like(s: Optional[str]) -> Optional[int]:
    if s is None:
        return None
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    for junk in ["(estimated runway)", "(estimated)", "(actual)", "(gate)", "(runway)", "(+8:00)", "(-00:00)"]:
        s = s.replace(junk, "").strip()
    s = re.sub(r"\bdec\s+\d{1,2}\b", "", s).strip()
    m = re.search(r"(\d{1,2}:\d{2})\s*(a\.m\.|p\.m\.)", s)
    if not m:
        return None
    hhmm, ap = m.group(1), m.group(2)
    hh, mm = map(int, hhmm.split(":"))
    if ap == "p.m." and hh != 12:
        hh += 12
    if ap == "a.m." and hh == 12:
        hh = 0
    return hh * 60 + mm


def circ_diff(a: Optional[int], b: Optional[int]) -> Optional[int]:
    if a is None or b is None:
        return None
    d = a - b
    while d <= -720:
        d += 1440
    while d > 720:
        d -= 1440
    return d


def duration_minutes(start: Optional[int], end: Optional[int]) -> Optional[int]:
    if start is None or end is None:
        return None
    d = end - start
    if d < 0:
        d += 1440
    return d


def check_time_regex(row_idx, row, rule, cfg):
    return check_regex(row_idx, row, rule, cfg)


def check_temporal_pair(row_idx, row, rule, cfg):
    cols = rule.get("columns", [])
    if len(cols) < 2:
        return None
    c1, c2 = cols[0], cols[1]
    if c1 not in row or c2 not in row:
        return None
    t1, t2 = parse_time_like(row.get(c1)), parse_time_like(row.get(c2))
    if t1 is None or t2 is None:
        return None
    max_abs_diff = safe_int(rule.get("max_abs_diff", rule.get("max_minutes", 240)), 240)
    d = circ_diff(t1, t2)
    if d is not None and abs(d) > max_abs_diff:
        target = rule.get("target_column") or c2
        return violation(row_idx, target, row.get(target), rule, f"temporal pair inconsistency: diff={d} minutes")
    return None


def check_delay_window(row_idx, row, rule, cfg):
    sched_col = rule.get("scheduled_column", rule.get("sched_col", "sched_dep_time"))
    act_col = rule.get("actual_column", rule.get("act_col", "act_dep_time"))
    if sched_col not in row or act_col not in row:
        return None
    st, at = parse_time_like(row.get(sched_col)), parse_time_like(row.get(act_col))
    if st is None or at is None:
        return None
    d = circ_diff(at, st)
    low = safe_int(rule.get("min_delay", rule.get("lower_bound", -240)), -240)
    high = safe_int(rule.get("max_delay", rule.get("upper_bound", 720)), 720)
    if d is not None and not (low <= d <= high):
        target = rule.get("target_column") or act_col
        return violation(row_idx, target, row.get(target), rule, f"delay window violation: delay={d}, expected=[{low},{high}]")
    return None


def check_duration_consistency(row_idx, row, rule, cfg):
    dep_col = rule.get("dep_column", rule.get("departure_column", "sched_dep_time"))
    arr_col = rule.get("arr_column", rule.get("arrival_column", "sched_arr_time"))
    if dep_col not in row or arr_col not in row:
        return None
    dep, arr = parse_time_like(row.get(dep_col)), parse_time_like(row.get(arr_col))
    if dep is None or arr is None:
        return None
    dur = duration_minutes(dep, arr)
    low = safe_int(rule.get("min_duration", rule.get("lower_bound", 20)), 20)
    high = safe_int(rule.get("max_duration", rule.get("upper_bound", 900)), 900)
    if dur is not None and not (low <= dur <= high):
        target = rule.get("target_column") or arr_col
        return violation(row_idx, target, row.get(target), rule, f"duration consistency violation: duration={dur}, expected=[{low},{high}]")
    return None


def check_actual_time_global_candidate(row_idx, row, rule, cfg):
    col = rule.get("column") or rule.get("target_column")
    if not col or col not in row:
        return None
    val = row.get(col)
    if (pd.isna(val) or val is None) and cfg.skip_null_for_non_notnull_rules:
        return None
    if parse_time_like(val) is None:
        return violation(row_idx, col, val, rule, "actual time cannot be parsed as time-like value")
    return None


# ============================================================
# 8. Beers special checkers
# ============================================================

def parse_abv_like(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", str(s).strip().lower().replace("%", ""))
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def parse_ibu_like(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    txt = str(s).strip().lower()
    if txt in BASE_MISSING_TOKENS:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", txt)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def parse_ounces_like(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", str(s).strip().lower())
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def canonicalize_ounces_value(s: Optional[str]) -> Optional[str]:
    num = parse_ounces_like(s)
    if num is None:
        return None
    if abs(num - round(num)) < 1e-9:
        return str(int(round(num)))
    return f"{num:.6f}".rstrip("0").rstrip(".")


def canonicalize_abv_value(s: Optional[str]) -> Optional[str]:
    num = parse_abv_like(s)
    if num is None:
        return None
    return f"{num:.15f}".rstrip("0").rstrip(".")


def strip_city_state_suffix(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    txt = str(s).strip().lower()
    s2 = re.sub(r"\s+[a-z]{2}$", "", txt).strip()
    return s2 if s2 != txt else None


def check_ounces_unit_format(row_idx, row, rule, cfg):
    col = rule.get("column", "ounces")
    if col not in row:
        return None
    val = row.get(col)
    if val is None or pd.isna(val):
        return None
    if parse_ounces_like(val) is None:
        return violation(row_idx, col, val, rule, "ounces cannot be parsed as numeric amount")
    return None


def check_abv_numeric_format(row_idx, row, rule, cfg):
    col = rule.get("column", "abv")
    if col not in row:
        return None
    val = row.get(col)
    if val is None or pd.isna(val):
        return None
    if parse_abv_like(val) is None:
        return violation(row_idx, col, val, rule, "abv cannot be parsed as numeric percentage")
    return None


def check_ibu_numeric_or_na_format(row_idx, row, rule, cfg):
    col = rule.get("column", "ibu")
    if col not in row:
        return None
    val = row.get(col)
    if val is None or pd.isna(val):
        return None
    if parse_ibu_like(val) is None:
        return violation(row_idx, col, val, rule, "ibu cannot be parsed as numeric or missing-like value")
    return None


def check_state_abbr_format(row_idx, row, rule, cfg):
    col = rule.get("column", "state")
    if col not in row:
        return None
    val = row.get(col)
    if val is None or pd.isna(val):
        return None
    if re.fullmatch(r"[a-z]{2}", str(val).strip().lower()) is None:
        return violation(row_idx, col, val, rule, "state should be two-letter abbreviation")
    return None


def check_numeric_range_window(row_idx, row, rule, cfg):
    col = rule.get("column") or rule.get("target_column")
    if not col or col not in row:
        return None
    val = row.get(col)
    if val is None or pd.isna(val):
        return None
    if col == "abv":
        num = parse_abv_like(val)
    elif col == "ibu":
        num = parse_ibu_like(val)
    elif col == "ounces":
        num = parse_ounces_like(val)
    else:
        num = try_parse_one_numeric(val)
    if num is None:
        return None
    low = rule.get("lower_bound", rule.get("min"))
    high = rule.get("upper_bound", rule.get("max"))
    if low is not None and high is not None and not (float(low) <= num <= float(high)):
        return violation(row_idx, col, val, rule, f"numeric range window violation: value={num}, expected=[{low},{high}]")
    return None


def check_ounces_canonical(row_idx, row, rule, cfg):
    col = rule.get("column", "ounces")
    if col not in row:
        return None
    val = row.get(col)
    canon = canonicalize_ounces_value(val)
    if val is not None and canon is not None and str(val).strip().lower() != canon:
        return violation(row_idx, col, val, rule, f"ounces canonical mismatch: canonical={canon}", expected_value=canon)
    return None


def check_abv_canonical(row_idx, row, rule, cfg):
    col = rule.get("column", "abv")
    if col not in row:
        return None
    val = row.get(col)
    canon = canonicalize_abv_value(val)
    if val is not None and canon is not None and str(val).strip().lower() != canon:
        return violation(row_idx, col, val, rule, f"abv canonical mismatch: canonical={canon}", expected_value=canon)
    return None


def check_city_state_suffix_pollution(row_idx, row, rule, cfg):
    col = rule.get("column", "city")
    if col not in row:
        return None
    val = row.get(col)
    stripped = strip_city_state_suffix(val)
    if stripped is not None:
        return violation(row_idx, col, val, rule, f"city contains state suffix; canonical={stripped}", expected_value=stripped)
    return None


# ============================================================
# 9. Rayyan special checkers
# ============================================================

LANG_CANONICAL_MAP = {
    "eng": "eng", "english": "eng", "english; abstract language:english": "eng", "en": "eng",
    "pol": "pol", "polish": "pol",
    "fre": "fre", "french": "fre",
    "ger": "ger", "german": "ger",
    "spa": "spa", "spanish": "spa",
}


def canonicalize_language(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    return LANG_CANONICAL_MAP.get(str(s).strip().lower())


def extract_issn_digits(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    m = re.search(r"(\d{4})[- ]?(\d{3}[\dx])", str(s).strip().lower())
    if not m:
        return None
    return f"{m.group(1)}-{m.group(2)}"


def is_normal_journal_issn_variant(s: Optional[str]) -> bool:
    if s is None:
        return False
    txt = str(s).strip().lower()
    if re.fullmatch(r"\d{4}-\d{3}[\dx]", txt):
        return True
    if re.fullmatch(r"\d{4}-\d{3}[\dx]\s*\(print\)", txt):
        return True
    if re.fullmatch(r"\d{4}-\d{3}[\dx]\s*\(print\)\s*\d{4}-\d{3}[\dx]", txt):
        return True
    return False


def normalize_date_like(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    txt = re.sub(r"\s+", "", str(s).strip().lower())
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{2,4})$", txt)
    if m:
        mm, dd, yy = int(m.group(1)), int(m.group(2)), m.group(3)
        yy_i = int(yy)
        if len(yy) == 2:
            yy_i = 1900 + yy_i if yy_i >= 30 else 2000 + yy_i
        if 1 <= mm <= 12 and 1 <= dd <= 31:
            return f"{yy_i:04d}-{mm:02d}-{dd:02d}"
    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", txt)
    if m:
        yy_i, mm, dd = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mm <= 12 and 1 <= dd <= 31:
            return f"{yy_i:04d}-{mm:02d}-{dd:02d}"
    return None


def split_date_components(s: Optional[str]):
    if s is None:
        return None
    txt = re.sub(r"\s+", "", str(s).strip().lower())
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{2,4})$", txt)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), m.group(3)


def generate_date_permutations(s: Optional[str]) -> List[str]:
    comp = split_date_components(s)
    if comp is None:
        return []
    a, b, c = comp
    arr = [str(a), str(b), str(c)]
    perms = []
    for idxs in [(1, 3, 2), (2, 3, 1), (2, 1, 3)]:
        mm_raw, dd_raw, yy_raw = arr[idxs[0] - 1], arr[idxs[1] - 1], arr[idxs[2] - 1]
        try:
            mm, dd, yy_i = int(mm_raw), int(dd_raw), int(yy_raw)
        except Exception:
            continue
        if len(yy_raw) == 2:
            yy_i = 1900 + yy_i if yy_i >= 30 else 2000 + yy_i
        if 1 <= mm <= 12 and 1 <= dd <= 31:
            perms.append(f"{yy_i:04d}-{mm:02d}-{dd:02d}")
    return list(dict.fromkeys(perms))


def generate_ymd_swap_candidate(s: Optional[str]) -> Optional[str]:
    comp = split_date_components(s)
    if comp is None:
        return None
    a, b, c = comp
    try:
        mm, dd, yy_i = int(str(b)), int(str(c)), int(str(a))
    except Exception:
        return None
    if len(str(a)) <= 2:
        yy_i = 1900 + yy_i if yy_i >= 30 else 2000 + yy_i
    if 1 <= mm <= 12 and 1 <= dd <= 31:
        return f"{yy_i:04d}-{mm:02d}-{dd:02d}"
    return None


def canonicalize_volume_issue(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    m = re.search(r"(\d+)", str(s).strip().lower())
    return str(int(m.group(1))) if m else None


def is_normal_pagination_value(s: Optional[str]) -> bool:
    if s is None:
        return False
    return re.fullmatch(r"\d+(-\d+)?", str(s).strip().lower()) is not None


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


def check_issn_canonical(row_idx, row, rule, cfg):
    col = rule.get("column", "journal_issn")
    if col not in row:
        return None
    val = row.get(col)
    if val is None or pd.isna(val):
        return None
    if is_normal_journal_issn_variant(val):
        return None
    canon = extract_issn_digits(val)
    if canon is None:
        return violation(row_idx, col, val, rule, "ISSN cannot be canonicalized")
    if str(val).strip().lower() != canon:
        return violation(row_idx, col, val, rule, f"ISSN canonical mismatch: canonical={canon}", expected_value=canon)
    return None


def check_language_canonical(row_idx, row, rule, cfg):
    col = rule.get("column", "article_language")
    if col not in row:
        return None
    val = row.get(col)
    if val is None or pd.isna(val):
        return None
    canon = canonicalize_language(val)
    if canon is None:
        return violation(row_idx, col, val, rule, "language is not a known canonical language code")
    if str(val).strip().lower() != canon:
        return violation(row_idx, col, val, rule, f"language canonical mismatch: canonical={canon}", expected_value=canon)
    return None


def check_date_like(row_idx, row, rule, cfg):
    col = rule.get("column", "article_jcreated_at")
    if col not in row:
        return None
    val = row.get(col)
    if val is None or pd.isna(val):
        return None
    if normalize_date_like(val) is None:
        return violation(row_idx, col, val, rule, "date-like value cannot be normalized")
    return None


def check_date_component_permutation(row_idx, row, rule, cfg):
    col = rule.get("column", "article_jcreated_at")
    if col not in row:
        return None
    val = row.get(col)
    perms = generate_date_permutations(val)
    canon = normalize_date_like(val)
    if perms and canon not in perms:
        return violation(row_idx, col, val, rule, f"date component permutation suspicious: candidates={perms}", expected_value=perms[0])
    return None


def check_date_ymd_swap(row_idx, row, rule, cfg):
    col = rule.get("column", "article_jcreated_at")
    if col not in row:
        return None
    val = row.get(col)
    cand = generate_ymd_swap_candidate(val)
    if cand is not None and cand != normalize_date_like(val):
        return violation(row_idx, col, val, rule, f"date ymd swap suspicious: candidate={cand}", expected_value=cand)
    return None


def check_volume_issue_numeric(row_idx, row, rule, cfg):
    col = rule.get("column") or rule.get("target_column")
    if not col or col not in row:
        return None
    val = row.get(col)
    if val is None or pd.isna(val):
        return None
    canon = canonicalize_volume_issue(val)
    if canon is None:
        return violation(row_idx, col, val, rule, "volume/issue value has no numeric component")
    return None


def check_pagination_format(row_idx, row, rule, cfg):
    col = rule.get("column", "article_pagination")
    if col not in row:
        return None
    val = row.get(col)
    if val is None or pd.isna(val):
        return None
    if not is_normal_pagination_value(val):
        return violation(row_idx, col, val, rule, "pagination value looks polluted or malformed")
    return None


def check_author_list_braced(row_idx, row, rule, cfg):
    col = rule.get("column", "author_list")
    if col not in row:
        return None
    val = row.get(col)
    if val is None or pd.isna(val):
        return None
    if not looks_like_author_list(val):
        return violation(row_idx, col, val, rule, "author_list should be braced JSON-like list")
    return None


def check_author_list_encoding_corruption(row_idx, row, rule, cfg):
    col = rule.get("column", "author_list")
    if col not in row:
        return None
    val = row.get(col)
    if has_author_list_encoding_corruption(val):
        return violation(row_idx, col, val, rule, "author_list contains encoding corruption characters")
    return None


# ============================================================
# 10. Adult special checkers
# ============================================================

VALID_AGE_BUCKETS = {
    "<18", "18-21", "22-25", "26-30", "31-35", "36-40", "41-45",
    "46-50", "51-55", "56-60", "61-65", "66-70", ">70"
}
VALID_WORKCLASS = {"Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"}
VALID_EDUCATION = {"Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th", "11th", "12th", "HS-grad", "Some-college", "Assoc-voc", "Assoc-acdm", "Bachelors", "Masters", "Prof-school", "Doctorate"}
VALID_MARITALSTATUS = {"Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"}
VALID_OCCUPATION = {"Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"}
VALID_RELATIONSHIP = {"Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"}
VALID_RACE = {"White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"}
VALID_SEX = {"Male", "Female"}
VALID_INCOME = {"LessThan50K", "MoreThan50K", "<=50K", ">50K", "<50K", ">=50K"}
VALID_COUNTRY = {
    "United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany",
    "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China",
    "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica",
    "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic",
    "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala",
    "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador",
    "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"
}
ADULT_DOMAIN_MAP = {
    "workclass": VALID_WORKCLASS, "education": VALID_EDUCATION,
    "maritalstatus": VALID_MARITALSTATUS, "occupation": VALID_OCCUPATION,
    "relationship": VALID_RELATIONSHIP, "race": VALID_RACE, "sex": VALID_SEX,
    "country": VALID_COUNTRY, "income": VALID_INCOME,
}
EDUCATION_ORDER = {
    "Preschool": 0, "1st-4th": 1, "5th-6th": 2, "7th-8th": 3, "9th": 4,
    "10th": 5, "11th": 6, "12th": 7, "HS-grad": 8, "Some-college": 9,
    "Assoc-voc": 10, "Assoc-acdm": 10, "Bachelors": 11, "Masters": 12,
    "Prof-school": 13, "Doctorate": 14,
}
RELATIONSHIP_SPOUSE_VALUES = {"Husband", "Wife"}
NON_MARRIED_STATUS_FOR_SPOUSE = {"Never-married", "Divorced", "Separated", "Widowed"}
VALID_SEX_BY_RELATIONSHIP = {"Husband": "Male", "Wife": "Female"}


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
    if lo is None or lo < 0:
        return False
    if hi is not None and (hi < lo or hi > 120):
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
    if lo is None or lo < 0:
        return False
    if hi is not None and (hi < lo or hi > 120):
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


def check_age_bucket_format(row_idx, row, rule, cfg):
    col = rule.get("column", "age")
    if col not in row:
        return None
    val = row.get(col)
    if val is not None and not is_age_bucket_like(val):
        return violation(row_idx, col, val, rule, "age should be valid age bucket")
    return None


def check_hours_per_week(row_idx, row, rule, cfg):
    col = rule.get("column", "hoursperweek")
    if col not in row:
        return None
    val = row.get(col)
    if val is not None and not is_hours_like(val):
        return violation(row_idx, col, val, rule, "hoursperweek should be numeric or range within 0..120")
    return None


def check_adult_domain(row_idx, row, rule, cfg):
    col = rule.get("column")
    if not col or col not in row:
        return None
    val = row.get(col)
    if val is None or pd.isna(val):
        return None
    valid = ADULT_DOMAIN_MAP.get(col)
    if valid is None:
        return None
    if str(val).strip() not in valid:
        return violation(row_idx, col, val, rule, f"{col} value not in valid adult domain")
    return None


def check_income_canonical(row_idx, row, rule, cfg):
    col = rule.get("column", "income")
    if col not in row:
        return None
    val = row.get(col)
    if val is not None and canonical_income(val) is None:
        return violation(row_idx, col, val, rule, "income cannot canonicalize to LessThan50K/MoreThan50K")
    return None


def check_relationship_marital_spouse(row_idx, row, rule, cfg):
    if "relationship" not in row or "maritalstatus" not in row:
        return None
    rel, mar = row.get("relationship"), row.get("maritalstatus")
    if rel in RELATIONSHIP_SPOUSE_VALUES and mar in NON_MARRIED_STATUS_FOR_SPOUSE:
        return violation(row_idx, "relationship", rel, rule, "relationship Husband/Wife conflicts with non-married maritalstatus")
    return None


def check_relationship_age_spouse(row_idx, row, rule, cfg):
    if "relationship" not in row or "age" not in row:
        return None
    rel, age = row.get("relationship"), row.get("age")
    lo, hi = parse_age_bucket(age)
    max_upper = safe_int(rule.get("max_age_upper_for_conflict", 17), 17)
    if rel in RELATIONSHIP_SPOUSE_VALUES and hi is not None and hi <= max_upper:
        return violation(row_idx, "relationship", rel, rule, "relationship Husband/Wife conflicts with very young age")
    return None


def check_relationship_sex_spouse(row_idx, row, rule, cfg):
    if "relationship" not in row or "sex" not in row:
        return None
    rel, sex = row.get("relationship"), row.get("sex")
    expected = VALID_SEX_BY_RELATIONSHIP.get(rel)
    if expected and sex in VALID_SEX and sex != expected:
        return violation(row_idx, "relationship", rel, rule, f"relationship {rel} conflicts with sex={sex}", expected_value=expected)
    return None


def check_education_age_extreme(row_idx, row, rule, cfg):
    if "education" not in row or "age" not in row:
        return None
    edu, age = row.get("education"), row.get("age")
    lo, hi = parse_age_bucket(age)
    min_order = safe_int(rule.get("min_extreme_education_order", EDUCATION_ORDER.get("Bachelors", 11)), 11)
    max_age = safe_int(rule.get("max_age_upper_for_conflict", 17), 17)
    if hi is not None and hi <= max_age and EDUCATION_ORDER.get(edu, -1) >= min_order:
        return violation(row_idx, "education", edu, rule, "very young age conflicts with high education")
    return None


# Old adult consistency rules, approximated conservatively.
def check_age_education_consistency(row_idx, row, rule, cfg):
    return check_education_age_extreme(row_idx, row, rule, cfg)


def check_age_relationship_consistency(row_idx, row, rule, cfg):
    return check_relationship_age_spouse(row_idx, row, rule, cfg)


def check_marital_relationship_consistency(row_idx, row, rule, cfg):
    return check_relationship_marital_spouse(row_idx, row, rule, cfg)


def check_sex_relationship_consistency(row_idx, row, rule, cfg):
    return check_relationship_sex_spouse(row_idx, row, rule, cfg)


def check_context_noop(row_idx, row, rule, cfg):
    return None


# ============================================================
# 11. Soccer special checkers
# ============================================================

VALID_POSITIONS = {"Goalkeeper", "Defender", "Midfield", "Midfielder", "Forward", "Striker", "Winger"}
POSITION_CANONICAL_MAP = {"Midfielder": "Midfield"}
BIRTHYEAR_MIN, BIRTHYEAR_MAX = 1940, 2010
SEASON_MIN, SEASON_MAX = 1900, 2035
PLAYER_MIN_AGE_AT_SEASON, PLAYER_MAX_AGE_AT_SEASON = 15, 50
SOCCER_REGEX_BY_TYPE = {
    "name_pattern": r"^[A-Za-z]+(?:_[A-Za-z]+)*$",
    "surname_pattern": r"^[A-Za-z]+(?:[-_'][A-Za-z]+)*$",
    "manager_name_pattern": r"^[A-Za-z]+(?:_[A-Za-z]+)*$",
    "team_name_pattern": r"^[A-Za-z0-9]+(?:[._'\-][A-Za-z0-9]+)*$",
    "birthplace_location_pattern": r"^[A-Za-z]+(?:[ _'\-][A-Za-z]+)*$",
    "city_location_pattern": r"^[A-Za-z]+(?:[ _'\-][A-Za-z]+)*$",
    "stadium_name_pattern": r"^[A-Za-z0-9]+(?:[ _'\-][A-Za-z0-9]+)*$",
}


def parse_year(value: Optional[str]) -> Optional[int]:
    return parse_int_like(value)


def canonical_position(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if s in POSITION_CANONICAL_MAP:
        return POSITION_CANONICAL_MAP[s]
    if s in VALID_POSITIONS:
        return s
    return None


def check_birthyear_numeric_range(row_idx, row, rule, cfg):
    col = rule.get("column", "birthyear")
    if col not in row:
        return None
    val = row.get(col)
    y = parse_year(val)
    low = safe_int(rule.get("lower_bound", BIRTHYEAR_MIN), BIRTHYEAR_MIN)
    high = safe_int(rule.get("upper_bound", BIRTHYEAR_MAX), BIRTHYEAR_MAX)
    if y is None or not (low <= y <= high):
        return violation(row_idx, col, val, rule, f"birthyear should be in [{low}, {high}]")
    return None


def check_season_numeric_range(row_idx, row, rule, cfg):
    col = rule.get("column", "season")
    if col not in row:
        return None
    val = row.get(col)
    y = parse_year(val)
    low = safe_int(rule.get("lower_bound", SEASON_MIN), SEASON_MIN)
    high = safe_int(rule.get("upper_bound", SEASON_MAX), SEASON_MAX)
    if y is None or not (low <= y <= high):
        return violation(row_idx, col, val, rule, f"season should be in [{low}, {high}]")
    return None


def check_position_domain(row_idx, row, rule, cfg):
    col = rule.get("column", "position")
    if col not in row:
        return None
    val = row.get(col)
    if val is not None and canonical_position(val) is None:
        return violation(row_idx, col, val, rule, "position not in valid soccer position domain")
    return None


def check_birthyear_season_age(row_idx, row, rule, cfg):
    if "birthyear" not in row or "season" not in row:
        return None
    by, sy = parse_year(row.get("birthyear")), parse_year(row.get("season"))
    if by is None or sy is None:
        return None
    age = sy - by
    min_age = safe_int(rule.get("min_age", PLAYER_MIN_AGE_AT_SEASON), PLAYER_MIN_AGE_AT_SEASON)
    max_age = safe_int(rule.get("max_age", PLAYER_MAX_AGE_AT_SEASON), PLAYER_MAX_AGE_AT_SEASON)
    if age < min_age or age > max_age:
        target = rule.get("target_column") or "birthyear"
        return violation(row_idx, target, row.get(target), rule, f"age at season implausible: age={age}, expected=[{min_age},{max_age}]")
    return None


def check_soccer_regex(row_idx, row, rule, cfg):
    rtype = rule.get("rule_type")
    pattern = rule.get("regex") or SOCCER_REGEX_BY_TYPE.get(rtype)
    if pattern:
        local = dict(rule)
        local["regex"] = pattern
        return check_regex(row_idx, row, local, cfg)
    return None


# ============================================================
# 12. Dispatcher
# ============================================================

CHECKERS: Dict[str, Callable[[int, pd.Series, Dict[str, Any], DatasetConfig], Optional[Dict[str, Any]]]] = {
    "not_null": check_not_null,
    "numeric_range": check_numeric_range,
    "enumeration_domain": check_enumeration,
    "pattern": check_pattern,
    "generic_regex": check_regex,

    "functional_dependency": check_fd_like,
    "soft_functional_dependency": check_fd_like,
    "numeric_outlier_iqr": check_numeric_outlier_iqr,
    "rare_value": check_rare_value,
    "rare_pattern": check_rare_pattern,
    "global_dominant_value": check_global_dominant,
    "dominant_value_by_context": check_context_dominant,
    "relationship_context_dominant": check_context_dominant,
    "team_context_dominant": check_context_dominant,
    "dominant_value_by_context_pair": check_pair_context,
    "high_risk_missing": check_high_risk_missing,
    "paired_missing_consistency": check_paired_missing,
    "dominant_value_typo": check_dominant_value_typo,

    # hospital
    "score_format": check_score_format,
    "sample_format": check_sample_format,
    "sample_patients_suffix": check_sample_patients_suffix,

    # flights
    "time_format": check_time_regex,
    "datetime_time_format": check_time_regex,
    "time_loose_pattern": check_time_regex,
    "temporal_pair_consistency": check_temporal_pair,
    "delay_window": check_delay_window,
    "duration_consistency_window": check_duration_consistency,
    "actual_time_global_candidate": check_actual_time_global_candidate,

    # beers
    "ounces_unit_format": check_ounces_unit_format,
    "ounces_loose_pattern": check_ounces_unit_format,
    "abv_numeric_format": check_abv_numeric_format,
    "ibu_numeric_or_na_format": check_ibu_numeric_or_na_format,
    "state_abbr_format": check_state_abbr_format,
    "numeric_range_window": check_numeric_range_window,
    "ounces_canonical_format": check_ounces_canonical,
    "abv_canonical_format": check_abv_canonical,
    "city_state_suffix_pollution": check_city_state_suffix_pollution,

    # rayyan
    "issn_format": check_issn_canonical,
    "issn_loose_format": check_issn_canonical,
    "issn_canonical_format": check_issn_canonical,
    "language_canonical_format": check_language_canonical,
    "date_like_format": check_date_like,
    "date_component_permutation_suspicious": check_date_component_permutation,
    "date_ymd_swap_suspicious": check_date_ymd_swap,
    "volume_issue_numeric_format": check_volume_issue_numeric,
    "pagination_format": check_pagination_format,
    "author_list_braced_format": check_author_list_braced,
    "author_list_encoding_corruption": check_author_list_encoding_corruption,

    # adult
    "age_bucket_format": check_age_bucket_format,
    "hours_per_week_format_range": check_hours_per_week,
    "workclass_domain": check_adult_domain,
    "education_domain": check_adult_domain,
    "maritalstatus_domain": check_adult_domain,
    "occupation_domain": check_adult_domain,
    "relationship_domain": check_adult_domain,
    "race_domain": check_adult_domain,
    "sex_domain": check_adult_domain,
    "country_domain": check_adult_domain,
    "income_domain": check_adult_domain,
    "income_canonical_format": check_income_canonical,
    "age_education_consistency": check_age_education_consistency,
    "age_relationship_consistency": check_age_relationship_consistency,
    "marital_relationship_consistency": check_marital_relationship_consistency,
    "sex_relationship_consistency": check_sex_relationship_consistency,
    "relationship_marital_spouse_conflict": check_relationship_marital_spouse,
    "relationship_age_spouse_conflict": check_relationship_age_spouse,
    "relationship_sex_spouse_conflict": check_relationship_sex_spouse,
    "education_age_extreme_conflict": check_education_age_extreme,
    "workclass_occupation_consistency": check_context_noop,
    "education_occupation_consistency": check_context_noop,
    "hours_workclass_consistency": check_context_noop,

    # soccer
    "birthyear_numeric_range": check_birthyear_numeric_range,
    "season_numeric_range": check_season_numeric_range,
    "position_domain": check_position_domain,
    "birthyear_season_age_consistency": check_birthyear_season_age,
    "name_pattern": check_soccer_regex,
    "surname_pattern": check_soccer_regex,
    "manager_name_pattern": check_soccer_regex,
    "team_name_pattern": check_soccer_regex,
    "birthplace_location_pattern": check_soccer_regex,
    "city_location_pattern": check_soccer_regex,
    "stadium_name_pattern": check_soccer_regex,
    "team_season_manager_consistency_hint": check_context_dominant,
    "city_stadium_consistency_hint": check_context_dominant,
}


def materialize_runtime_groups(df: pd.DataFrame, rules: List[Dict[str, Any]], cfg: DatasetConfig) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    fd_rules = prepare_fd_rules(df, rules, cfg)
    context_rules = prepare_context_rules(rules, cfg)
    pair_context_rules = prepare_pair_context_rules(rules, cfg)

    for r in rules:
        rtype = r.get("rule_type")
        if rtype in {"functional_dependency", "soft_functional_dependency"}:
            continue
        if rtype in {"dominant_value_by_context", "relationship_context_dominant", "team_context_dominant"}:
            continue
        if rtype == "dominant_value_by_context_pair":
            continue
        groups[rtype].append(r)

    groups["__fd_like__"] = fd_rules

    for r in context_rules:
        groups[r.get("rule_type")].append(r)
    for r in pair_context_rules:
        groups[r.get("rule_type")].append(r)

    # adult domain pseudo-group
    groups["__adult_domain__"] = [
        r for r in rules
        if r.get("rule_type") in {
            "workclass_domain", "education_domain", "maritalstatus_domain",
            "occupation_domain", "relationship_domain", "race_domain",
            "sex_domain", "country_domain", "income_domain",
        }
    ]

    return groups


def iter_rule_groups_in_order(groups: Dict[str, List[Dict[str, Any]]], cfg: DatasetConfig) -> Iterable[Tuple[str, List[Dict[str, Any]]]]:
    used = set()
    for key in cfg.runtime_order:
        used.add(key)
        yield key, groups.get(key, [])

    # fallback for rule types not listed in config.
    for key in sorted(groups.keys()):
        if key not in used and key != "__adult_domain__":
            yield key, groups[key]


def get_checker_for_group(group_key: str, rule: Dict[str, Any], cfg: DatasetConfig):
    if group_key == "__fd_like__":
        return check_fd_like
    if group_key == "__adult_domain__":
        return check_adult_domain
    return CHECKERS.get(rule.get("rule_type")) or CHECKERS.get(group_key)


# ============================================================
# 13. 主流程
# ============================================================

def shrink_search_space(
    input_csv: str,
    rule_json: str,
    output_jsonl: str,
    output_csv: str,
    cfg: DatasetConfig,
    args=None,
):
    df = pd.read_csv(input_csv, dtype=str)
    df = maybe_drop_unnamed_columns(df)

    if cfg.name == "soccer":
        expected = {
            "name", "surname", "birthyear", "birthplace", "position",
            "team", "city", "stadium", "season", "manager"
        }
        missing = sorted(list(expected - set(df.columns)))
        extra = sorted(list(set(df.columns) - expected))
        if missing:
            print(f"[WARN] Missing expected soccer columns: {missing}")
        if extra:
            print(f"[WARN] Extra columns in input: {extra}")

    for col in df.columns:
        df[col] = normalize_series(df[col], cfg)

    rules = load_rule_pool(rule_json)
    if args is None:
        class Dummy:
            disable_rule_types = []
            disable_weak_rules = False
            disable_fd = False
            disable_context = False
            disable_domain_format = False
        args = Dummy()
    rules = filter_rules(rules, cfg, args)

    groups = materialize_runtime_groups(df, rules, cfg)

    if cfg.name == "soccer":
        print("[RuntimeRules] active rule counts:")
        for key in cfg.runtime_order:
            label = key
            if key == "__fd_like__":
                label = "fd_like_rules(functional only)"
            print(f"  {label}: {len(groups.get(key, []))}")

    cell_map: Dict[Tuple[int, str], Dict[str, Any]] = {}

    group_order = list(iter_rule_groups_in_order(groups, cfg))

    for row_idx, row in df.iterrows():
        for group_key, group_rules in group_order:
            for rule in group_rules:
                checker = get_checker_for_group(group_key, rule, cfg)
                if checker is None:
                    continue
                try:
                    v = checker(row_idx, row, rule, cfg)
                except Exception as e:
                    # 为了不中断大规模数据集，单条规则异常只跳过。
                    # 如果需要严格复现，可加 --strict。
                    if getattr(args, "strict", False):
                        raise
                    continue
                if v is not None:
                    add_violation(cell_map, v, cfg)

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
    print(f"Shrink ratio: {len(candidates) / total_cells:.6f}" if total_cells else "Shrink ratio: 0.000000")

    if cfg.name == "soccer":
        print("[Candidate Columns]")
        for k, v in column_counter.most_common(30):
            print(f"  - {k}: {v}")
        print("[Candidate Rule Types]")
        for k, v in rule_type_counter.most_common(30):
            print(f"  - {k}: {v}")

    return candidates


def resolve_output_path(path: str, cwd: Optional[str]) -> str:
    if cwd and not os.path.isabs(path):
        return os.path.join(cwd, path)
    return path


def run_one_dataset(dataset: str, args):
    if dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset}. Available: {sorted(DATASET_CONFIGS)}")

    cfg = DATASET_CONFIGS[dataset]

    input_csv = args.input_csv or cfg.input_csv
    rule_json = args.rule_json or cfg.input_rule_json
    output_jsonl = args.output_jsonl or cfg.output_jsonl
    output_csv = args.output_csv or cfg.output_csv

    if args.cwd:
        os.makedirs(args.cwd, exist_ok=True)
        old_cwd = os.getcwd()
        os.chdir(args.cwd)
    else:
        old_cwd = None

    try:
        input_csv_r = input_csv
        rule_json_r = rule_json
        output_jsonl_r = output_jsonl
        output_csv_r = output_csv

        shrink_search_space(input_csv_r, rule_json_r, output_jsonl_r, output_csv_r, cfg, args)
    finally:
        if old_cwd is not None:
            os.chdir(old_cwd)


def parse_args():
    parser = argparse.ArgumentParser(description="Unified candidate shrinker for hospital/flights/beers/rayyan/adult/soccer.")
    parser.add_argument("--dataset", required=True, choices=sorted(list(DATASET_CONFIGS.keys()) + ["all"]))
    parser.add_argument("--input_csv", default=None)
    parser.add_argument("--rule_json", default=None)
    parser.add_argument("--output_jsonl", default=None)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--cwd", default=None, help="Run in this working directory, useful for preserving relative output paths.")

    # Ablation-friendly switches
    parser.add_argument("--disable-rule-types", nargs="*", default=[], help="Rule types to disable.")
    parser.add_argument("--disable-weak-rules", action="store_true", help="Disable rare/global/context/soft-FD weak rules.")
    parser.add_argument("--disable-fd", action="store_true", help="Disable functional_dependency and soft_functional_dependency.")
    parser.add_argument("--disable-context", action="store_true", help="Disable context-dominant rules.")
    parser.add_argument("--disable-domain-format", action="store_true", help="Disable domain/format/schema-like rules.")
    parser.add_argument("--strict", action="store_true", help="Raise checker exceptions instead of skipping problematic rule.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.dataset == "all":
        # all 模式下不允许单独覆盖输入输出，否则六个数据集会互相覆盖。
        if args.input_csv or args.rule_json or args.output_jsonl or args.output_csv:
            raise ValueError("--dataset all cannot be used with --input_csv/--rule_json/--output_* overrides")
        for ds in ["hospital", "flights", "beers", "rayyan", "adult", "soccer"]:
            print("\n" + "=" * 80)
            print(f"[DATASET] {ds}")
            print("=" * 80)
            run_one_dataset(ds, args)
    else:
        run_one_dataset(args.dataset, args)


if __name__ == "__main__":
    main()

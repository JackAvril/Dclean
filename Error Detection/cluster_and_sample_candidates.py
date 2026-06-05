#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unified_cluster_and_sample_candidates.py

统一版聚类采样脚本，覆盖：
hospital / flights / beers / rayyan / adult / soccer

目标：
1. 抽象统一逻辑，方便后续做消融实验。
2. 保持各数据集原来的输入/输出文件名风格：
   - candidate_clustered_*.csv
   - candidate_sampled_*.csv
   - candidate_sampled_with_context_*.jsonl
   - bucket_cluster_summary_*.csv
   - adult/soccer 额外 sample summary JSON
3. 保留原有核心输出字段：
   bucket_id, cluster_id, dist_to_center, sample_role, is_sampled
   adult 兼容 is_sampled_raw / signal_priority
   soccer 兼容 sampling_info / soccer_signal_bucket / quota 采样角色
4. 后续消融可通过统一开关关闭部分 bucket、feature、quota 或 budget cap。

运行示例：
python unified_cluster_and_sample_candidates.py --dataset soccer
python unified_cluster_and_sample_candidates.py --dataset adult --cwd "/mnt/.../Error Detection Adult"
python unified_cluster_and_sample_candidates.py --dataset all

消融示例：
python unified_cluster_and_sample_candidates.py --dataset soccer --disable-quota-sampling
python unified_cluster_and_sample_candidates.py --dataset adult --disable-budget-caps
python unified_cluster_and_sample_candidates.py --dataset flights --disable-bucket-fields neighbor_bucket pattern_bucket

说明：
- 这是抽象统一版，不是六份脚本逐行拼接。
- 若要求 bit-level 完全一致，需要用旧输出做 diff 后微调排序/文案。
"""

import argparse
import json
import math
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. 通用工具
# ============================================================

MISSING_TEXT = "missing"


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x: Any, default: int = 0) -> int:
    try:
        if pd.isna(x):
            return default
        return int(float(x))
    except Exception:
        return default


def safe_str(x: Any, default: str = MISSING_TEXT) -> str:
    if x is None:
        return default
    try:
        if pd.isna(x):
            return default
    except Exception:
        pass
    s = str(x).strip()
    return s if s else default


def env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def env_list(name: str) -> List[str]:
    v = os.getenv(name, "")
    return [x.strip() for x in v.split(",") if x.strip()]


def ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def json_safe_value(x: Any):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        if np.isnan(x):
            return None
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    return x


def normalize_boolish(x: Any) -> str:
    if x is None:
        return "unknown"
    try:
        if pd.isna(x):
            return "unknown"
    except Exception:
        pass
    s = str(x).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return "true"
    if s in {"false", "0", "no", "n"}:
        return "false"
    return s if s else "unknown"


def boolish_to_int(x: Any) -> int:
    s = normalize_boolish(x)
    if s == "true":
        return 1
    if s == "false":
        return 0
    try:
        return int(float(s))
    except Exception:
        return 0


def as_int01(x: Any) -> int:
    if pd.isna(x):
        return 0
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y"}:
        return 1
    if s in {"0", "false", "no", "n"}:
        return 0
    try:
        return int(float(x))
    except Exception:
        return 0


# ============================================================
# 2. 数据集配置
# ============================================================

BASE_NUMERIC_FEATURES = [
    "violation_count",
    "conflict_score",
    "value_frequency",
    "value_frequency_rank",
    "neighbor_majority_ratio",
    "prior_error_probability",
    "posterior_error_probability",
    "strong_rule_count",
    "candidate_generation_rule_count",
    "fd_like_count",
    "context_rule_count",
    "global_rule_count",
    "rare_value_count",
    "typo_rule_count",
    "pattern_rule_count",
    "schema_rule_count",
]

BASE_CATEGORICAL_FEATURES = [
    "semantic_type",
    "main_rule_type",
    "main_usage_role",
    "pattern_bucket",
    "rarity_bucket",
    "neighbor_bucket",
]

BASE_BUCKET_KEYS = [
    "semantic_type",
    "main_rule_type",
    "main_usage_role",
    "pattern_bucket",
    "rarity_bucket",
    "neighbor_bucket",
]


@dataclass
class SamplingConfig:
    name: str
    input_summary_csv: str
    input_detail_jsonl: str
    output_clustered_csv: str
    output_sampled_csv: str
    output_sampled_jsonl: str
    output_bucket_summary_csv: str
    output_sample_summary_json: Optional[str] = None

    use_column_in_bucket: bool = False
    use_rule_in_bucket: bool = False
    use_dataset_signal_in_bucket: bool = False

    min_bucket_size_for_cluster: int = 6
    max_clusters_per_bucket: int = 8
    target_cluster_size: int = 12
    small_bucket_full_sample_threshold: int = 5

    sample_center: bool = True
    sample_boundary: bool = True
    sample_outlier: bool = True
    sample_high_posterior: bool = False
    sample_low_posterior: bool = False

    max_samples_per_cluster: Optional[int] = None
    random_state: int = 42
    choose_k_mode: str = "round"  # round / ceil

    feature_numeric: List[str] = field(default_factory=lambda: list(BASE_NUMERIC_FEATURES))
    feature_categorical: List[str] = field(default_factory=lambda: list(BASE_CATEGORICAL_FEATURES))
    bucket_keys: List[str] = field(default_factory=lambda: list(BASE_BUCKET_KEYS))

    budget_mode: str = "none"  # none / adult / soccer
    max_total_samples: Optional[int] = None
    max_samples_per_column: Dict[str, int] = field(default_factory=dict)
    default_max_samples_per_column: Optional[int] = None
    max_samples_per_bucket: Optional[int] = None
    weak_signal_keep_ratio: float = 1.0
    enable_weak_signal_downsample: bool = False
    max_weak_rule_samples: Optional[int] = None

    min_samples_per_primary_column: int = 0
    min_samples_per_context_column: int = 0
    min_sample_per_active_column: int = 0
    min_sample_per_important_rule: int = 0
    high_recall_focus_columns: Set[str] = field(default_factory=set)
    important_rule_types: Set[str] = field(default_factory=set)
    weak_rule_types: Set[str] = field(default_factory=set)
    primary_target_columns: Set[str] = field(default_factory=set)
    context_only_columns: Set[str] = field(default_factory=set)

    keep_column_diversity_in_global_budget: bool = False


def build_configs() -> Dict[str, SamplingConfig]:
    cfgs: Dict[str, SamplingConfig] = {}

    cfgs["hospital"] = SamplingConfig(
        name="hospital",
        input_summary_csv="candidate_llm_contexts_flat_v2.csv",
        input_detail_jsonl="candidate_llm_contexts_v2.jsonl",
        output_clustered_csv="candidate_clustered_v2.csv",
        output_sampled_csv="candidate_sampled_v2.csv",
        output_sampled_jsonl="candidate_sampled_with_context_v2.jsonl",
        output_bucket_summary_csv="bucket_cluster_summary_v2.csv",
        use_column_in_bucket=False,
    )

    cfgs["flights"] = SamplingConfig(
        name="flights",
        input_summary_csv="candidate_llm_contexts_flat_flights.csv",
        input_detail_jsonl="candidate_llm_contexts_flights.jsonl",
        output_clustered_csv="candidate_clustered_flights.csv",
        output_sampled_csv="candidate_sampled_flights.csv",
        output_sampled_jsonl="candidate_sampled_with_context_flights.jsonl",
        output_bucket_summary_csv="bucket_cluster_summary_flights.csv",
        use_column_in_bucket=False,
        feature_numeric=BASE_NUMERIC_FEATURES + ["time_window_rule_count", "time_value_minutes"],
        feature_categorical=BASE_CATEGORICAL_FEATURES + ["time_window_bucket", "time_value_bucket"],
        bucket_keys=BASE_BUCKET_KEYS + ["time_window_bucket", "time_value_bucket"],
    )

    cfgs["beers"] = SamplingConfig(
        name="beers",
        input_summary_csv="candidate_llm_contexts_flat_beers.csv",
        input_detail_jsonl="candidate_llm_contexts_beers.jsonl",
        output_clustered_csv="candidate_clustered_beers.csv",
        output_sampled_csv="candidate_sampled_beers.csv",
        output_sampled_jsonl="candidate_sampled_with_context_beers.jsonl",
        output_bucket_summary_csv="bucket_cluster_summary_beers.csv",
        use_column_in_bucket=False,
        feature_numeric=BASE_NUMERIC_FEATURES + ["numeric_window_rule_count", "numeric_value"],
        feature_categorical=BASE_CATEGORICAL_FEATURES + ["numeric_window_bucket", "numeric_value_bucket"],
        bucket_keys=BASE_BUCKET_KEYS + ["numeric_window_bucket", "numeric_value_bucket"],
    )

    cfgs["rayyan"] = SamplingConfig(
        name="rayyan",
        input_summary_csv="candidate_llm_contexts_flat_rayyan.csv",
        input_detail_jsonl="candidate_llm_contexts_rayyan.jsonl",
        output_clustered_csv="candidate_clustered_rayyan.csv",
        output_sampled_csv="candidate_sampled_rayyan.csv",
        output_sampled_jsonl="candidate_sampled_with_context_rayyan.jsonl",
        output_bucket_summary_csv="bucket_cluster_summary_rayyan.csv",
        use_column_in_bucket=False,
        feature_numeric=BASE_NUMERIC_FEATURES + ["canonical_rule_count", "numeric_value"],
        feature_categorical=BASE_CATEGORICAL_FEATURES + ["canonical_bucket", "date_value_bucket", "language_bucket"],
        bucket_keys=BASE_BUCKET_KEYS + ["canonical_bucket", "date_value_bucket", "language_bucket"],
    )

    adult_primary = {"sex", "relationship", "education"}
    adult_context = {
        "age", "workclass", "maritalstatus", "occupation",
        "race", "hoursperweek", "country", "income"
    }
    adult_max_col = {
        "relationship": 80,
        "sex": 60,
        "education": 30,
        "age": 3,
        "workclass": 3,
        "maritalstatus": 3,
        "occupation": 3,
        "race": 3,
        "hoursperweek": 3,
        "country": 3,
        "income": 3,
    }
    cfgs["adult"] = SamplingConfig(
        name="adult",
        input_summary_csv="candidate_llm_contexts_flat_adult.csv",
        input_detail_jsonl="candidate_llm_contexts_adult.jsonl",
        output_clustered_csv="candidate_clustered_adult.csv",
        output_sampled_csv="candidate_sampled_adult.csv",
        output_sampled_jsonl="candidate_sampled_with_context_adult.jsonl",
        output_bucket_summary_csv="bucket_cluster_summary_adult.csv",
        output_sample_summary_json="sample_budget_summary_adult.json",
        use_column_in_bucket=True,
        min_bucket_size_for_cluster=8,
        max_clusters_per_bucket=5,
        target_cluster_size=18,
        small_bucket_full_sample_threshold=3,
        max_samples_per_cluster=2,
        feature_numeric=BASE_NUMERIC_FEATURES + [
            "adult_domain_rule_count",
            "adult_format_rule_count",
            "adult_consistency_rule_count",
            "age_lower",
            "age_upper",
            "hours_lower",
            "hours_upper",
            "education_order",
            "adult_profile_inconsistency_count",
            "adult_consistency_score",
            "target_is_primary_column",
            "target_is_context_only_column",
            "is_relationship_spouse_value",
            "relationship_marital_conflict",
            "relationship_age_conflict",
            "relationship_sex_conflict",
            "sex_value_invalid",
            "education_age_extreme_conflict",
            "has_relationship_v2_rule",
            "has_education_v2_rule",
        ],
        feature_categorical=[
            "column",
            "semantic_type",
            "main_rule_type",
            "main_usage_role",
            "pattern_bucket",
            "rarity_bucket",
            "neighbor_bucket",
            "domain_bucket",
            "adult_consistency_bucket",
            "age_sampling_bucket",
            "hours_sampling_bucket",
            "adult_value_bucket",
            "adult_v2_attribution_bucket",
        ],
        bucket_keys=[
            "semantic_type",
            "main_rule_type",
            "main_usage_role",
            "pattern_bucket",
            "rarity_bucket",
            "neighbor_bucket",
            "domain_bucket",
            "adult_consistency_bucket",
            "adult_v2_attribution_bucket",
            "age_sampling_bucket",
            "hours_sampling_bucket",
        ],
        budget_mode="adult",
        max_total_samples=180,
        max_samples_per_column=adult_max_col,
        default_max_samples_per_column=3,
        max_samples_per_bucket=4,
        weak_signal_keep_ratio=0.35,
        enable_weak_signal_downsample=True,
        min_samples_per_primary_column=10,
        min_samples_per_context_column=0,
        primary_target_columns=adult_primary,
        context_only_columns=adult_context,
        keep_column_diversity_in_global_budget=True,
    )

    soccer_columns = {
        "name", "surname", "birthyear", "birthplace", "position",
        "team", "city", "stadium", "season", "manager"
    }
    soccer_important_rules = {
        "not_null",
        "high_risk_missing",
        "birthyear_numeric_range",
        "season_numeric_range",
        "birthyear_season_age_consistency",
        "position_domain",
        "name_pattern",
        "surname_pattern",
        "manager_name_pattern",
        "team_name_pattern",
        "birthplace_location_pattern",
        "city_location_pattern",
        "stadium_name_pattern",
        "generic_regex",
        "pattern",
        "functional_dependency",
    }
    soccer_weak_rules = {
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
    cfgs["soccer"] = SamplingConfig(
        name="soccer",
        input_summary_csv="candidate_llm_contexts_flat_soccer.csv",
        input_detail_jsonl="candidate_llm_contexts_soccer.jsonl",
        output_clustered_csv="candidate_clustered_soccer.csv",
        output_sampled_csv="candidate_sampled_soccer.csv",
        output_sampled_jsonl="candidate_sampled_with_context_soccer.jsonl",
        output_bucket_summary_csv="bucket_cluster_summary_soccer.csv",
        output_sample_summary_json="sample_summary_soccer.json",
        use_column_in_bucket=True,
        use_rule_in_bucket=True,
        use_dataset_signal_in_bucket=True,
        min_bucket_size_for_cluster=8,
        max_clusters_per_bucket=8,
        target_cluster_size=12,
        small_bucket_full_sample_threshold=3,
        sample_high_posterior=True,
        sample_low_posterior=True,
        max_samples_per_cluster=4,
        choose_k_mode="ceil",
        feature_numeric=[
            "violation_count", "conflict_score", "value_frequency", "value_frequency_rank",
            "neighbor_majority_ratio", "prior_error_probability", "posterior_error_probability",
            "strong_rule_count", "candidate_generation_rule_count", "fd_like_count", "evidence_only_fd_count",
            "context_rule_count", "global_rule_count", "rare_value_count", "typo_rule_count",
            "pattern_rule_count", "schema_rule_count", "soccer_domain_rule_count",
            "soccer_format_rule_count", "soccer_numeric_rule_count", "soccer_consistency_rule_count",
            "row_missing_count", "row_non_missing_count", "year_value", "row_birthyear_int",
            "row_season_int", "row_age_at_season", "soccer_profile_inconsistency_count",
            "target_is_primary_column", "target_is_context_only_column",
            "birthyear_value_invalid", "season_value_invalid", "position_value_invalid",
            "position_needs_canonicalization", "birthyear_season_age_conflict", "team_context_conflict",
            "format_rule_signal", "fd_like_signal", "adult_profile_inconsistency_count", "dominant_pattern_ratio",
        ],
        feature_categorical=[
            "column", "semantic_type", "detected_type", "main_rule_type", "main_usage_role",
            "rarity_bucket", "pattern_bucket", "neighbor_bucket", "soccer_value_bucket",
            "soccer_signal_bucket", "soccer_attribution_bucket",
            "canonical_position", "row_position_canonical",
        ],
        bucket_keys=[],  # soccer 使用定制 bucket_id
        budget_mode="soccer",
        max_total_samples=350,
        max_samples_per_column={
            "name": 45,
            "surname": 45,
            "birthyear": 45,
            "birthplace": 50,
            "position": 50,
            "team": 45,
            "city": 40,
            "stadium": 45,
            "season": 45,
            "manager": 45,
        },
        default_max_samples_per_column=40,
        max_samples_per_bucket=20,
        min_sample_per_active_column=8,
        min_sample_per_important_rule=5,
        high_recall_focus_columns={"name", "surname", "birthplace"},
        important_rule_types=soccer_important_rules,
        weak_rule_types=soccer_weak_rules,
        max_weak_rule_samples=45,
        primary_target_columns=soccer_columns,
        context_only_columns=set(),
    )

    return cfgs


CONFIGS = build_configs()


# ============================================================
# 3. 读取详细 JSONL 和通用信息提取
# ============================================================

def load_detail_jsonl(path: str) -> Dict[Tuple[int, str], Dict[str, Any]]:
    detail_map: Dict[Tuple[int, str], Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = (int(obj["row_id"]), str(obj["column"]))
            detail_map[key] = obj
    return detail_map


def extract_main_rule_type_from_detail(detail_obj: Dict[str, Any]) -> str:
    try:
        conflict_ctx = detail_obj.get("conflict_context", {})
        rule_types = conflict_ctx.get("rule_types", [])
        if not rule_types:
            detailed = conflict_ctx.get("violated_rules_detailed", [])
            rule_types = [x.get("rule_type") for x in detailed if x.get("rule_type")]
        if not rule_types:
            return "none"
        return Counter(rule_types).most_common(1)[0][0]
    except Exception:
        return "unknown"


def extract_main_usage_role_from_detail(detail_obj: Dict[str, Any]) -> str:
    try:
        conflict_ctx = detail_obj.get("conflict_context", {})
        usage_roles = conflict_ctx.get("usage_roles", [])
        if not usage_roles:
            detailed = conflict_ctx.get("violated_rules_detailed", [])
            usage_roles = [x.get("usage_role") for x in detailed if x.get("usage_role")]
        if not usage_roles:
            return "none"
        return Counter(usage_roles).most_common(1)[0][0]
    except Exception:
        return "unknown"


def build_pattern_bucket_from_detail(detail_obj: Dict[str, Any]) -> str:
    try:
        col_ctx = detail_obj.get("column_context", {})
        cp = str(col_ctx.get("current_value_pattern", "NULL"))
        dp = str(col_ctx.get("dominant_pattern", "NULL"))
        return "pattern_match" if cp == dp else "pattern_mismatch"
    except Exception:
        return "unknown"


# ============================================================
# 4. 通用 bucket 构造
# ============================================================

def build_rarity_bucket(value_frequency: Any, value_frequency_rank: Any) -> str:
    vf = pd.to_numeric(value_frequency, errors="coerce")
    vr = pd.to_numeric(value_frequency_rank, errors="coerce")

    if pd.notna(vr):
        if vr <= 3:
            return "very_common"
        if vr <= 10:
            return "common"

    if pd.notna(vf):
        if vf <= 1:
            return "singleton"
        if vf <= 2:
            return "very_rare"
        if vf <= 5:
            return "rare"

    return "mid_frequency"


def build_neighbor_bucket(neighbor_majority_ratio: Any, neighbor_majority_value: Any, current_value: Any) -> str:
    nmr = pd.to_numeric(neighbor_majority_ratio, errors="coerce")
    if pd.isna(nmr):
        return "no_neighbor_signal"

    same = str(neighbor_majority_value) == str(current_value)

    if not same:
        if nmr >= 0.8:
            return "strong_support_other"
        if nmr >= 0.5:
            return "medium_support_other"
        return "weak_support_other"

    if nmr >= 0.8:
        return "strong_support_current"
    return "weak_support_current"


def build_time_window_bucket_from_detail(detail_obj: Dict[str, Any]) -> str:
    try:
        cnt = int(detail_obj.get("conflict_context", {}).get("time_window_rule_count", 0))
        if cnt <= 0:
            return "no_time_window_signal"
        if cnt == 1:
            return "single_time_window_signal"
        return "multi_time_window_signal"
    except Exception:
        return "unknown"


def build_time_value_bucket(time_value_minutes: Any) -> str:
    tv = pd.to_numeric(time_value_minutes, errors="coerce")
    if pd.isna(tv):
        return "unparsed_time"
    hour = int(tv) // 60
    if 0 <= hour < 6:
        return "late_night"
    if 6 <= hour < 12:
        return "morning"
    if 12 <= hour < 18:
        return "afternoon"
    return "evening"


def build_numeric_window_bucket_from_detail(detail_obj: Dict[str, Any]) -> str:
    try:
        cnt = int(detail_obj.get("conflict_context", {}).get("numeric_window_rule_count", 0))
        if cnt <= 0:
            return "no_numeric_window_signal"
        if cnt == 1:
            return "single_numeric_window_signal"
        return "multi_numeric_window_signal"
    except Exception:
        return "unknown"


def build_numeric_value_bucket(column: Any, numeric_value: Any) -> str:
    col = str(column)
    nv = pd.to_numeric(numeric_value, errors="coerce")
    if pd.isna(nv):
        return "unparsed_numeric"
    if col == "abv":
        if nv < 0.04:
            return "abv_low"
        if nv < 0.07:
            return "abv_mid"
        return "abv_high"
    if col == "ibu":
        if nv < 20:
            return "ibu_low"
        if nv < 50:
            return "ibu_mid"
        return "ibu_high"
    if col == "ounces":
        if nv < 12:
            return "ounces_small"
        if nv <= 12:
            return "ounces_standard"
        return "ounces_large"
    return "generic_numeric"


def build_canonical_bucket_from_detail(detail_obj: Dict[str, Any]) -> str:
    try:
        cnt = int(detail_obj.get("conflict_context", {}).get("canonical_rule_count", 0))
        if cnt <= 0:
            return "no_canonical_signal"
        if cnt == 1:
            return "single_canonical_signal"
        return "multi_canonical_signal"
    except Exception:
        return "unknown"


def build_date_value_bucket(date_value: Any) -> str:
    if pd.isna(date_value) or date_value is None or str(date_value).strip() == "":
        return "unparsed_date"
    try:
        year = int(str(date_value)[:4])
        if year < 1950:
            return "very_old"
        if year < 1980:
            return "old"
        if year < 2000:
            return "mid_old"
        if year < 2020:
            return "modern"
        return "recent"
    except Exception:
        return "unparsed_date"


def build_language_bucket(language_canonical: Any) -> str:
    if pd.isna(language_canonical) or language_canonical is None or str(language_canonical).strip() == "":
        return "unknown_language"
    return f"lang_{str(language_canonical).strip().lower()}"


def build_adult_consistency_bucket_from_detail(detail_obj: Dict[str, Any]) -> str:
    try:
        adult_ctx = detail_obj.get("adult_consistency_context", {})
        flags = adult_ctx.get("evidence_flags", [])
        score = float(adult_ctx.get("adult_consistency_score", 0.0) or 0.0)
        if flags and score >= 1.0:
            return "strong_adult_consistency_signal"
        if flags:
            return "adult_consistency_flag"
        if score > 0:
            return "weak_adult_consistency_signal"
        return "no_adult_consistency_signal"
    except Exception:
        return "unknown"


def build_domain_bucket(row: pd.Series) -> str:
    if "domain_valid" not in row.index:
        return "unknown_domain"
    val = row.get("domain_valid")
    if pd.isna(val):
        return "domain_not_applicable"
    s = str(val).strip().lower()
    if s in {"true", "1", "yes"}:
        return "domain_valid"
    if s in {"false", "0", "no"}:
        return "domain_invalid"
    return "domain_unknown"


def build_age_bucket_for_sampling(age_lower: Any, age_upper: Any) -> str:
    lo = pd.to_numeric(age_lower, errors="coerce")
    hi = pd.to_numeric(age_upper, errors="coerce")
    if pd.isna(lo):
        return "age_unparsed"
    if pd.notna(hi) and hi < 18:
        return "minor"
    if lo <= 25:
        return "young_adult"
    if lo <= 50:
        return "working_age"
    return "older_adult"


def build_hours_bucket_for_sampling(hours_lower: Any, hours_upper: Any) -> str:
    lo = pd.to_numeric(hours_lower, errors="coerce")
    hi = pd.to_numeric(hours_upper, errors="coerce")
    if pd.isna(lo):
        return "hours_unparsed"
    if pd.isna(hi):
        hi = lo
    mid = (lo + hi) / 2.0
    if mid < 20:
        return "low_hours"
    if mid <= 40:
        return "standard_hours"
    if mid <= 60:
        return "high_hours"
    return "very_high_hours"


RELATIONSHIP_V2_RULE_TYPES = {
    "relationship_marital_spouse_conflict",
    "relationship_age_spouse_conflict",
    "relationship_sex_spouse_conflict",
    "relationship_context_dominant",
}

EDUCATION_V2_RULE_TYPES = {
    "education_age_extreme_conflict",
}


def extract_adult_v2_features_from_detail(detail_obj: Dict[str, Any]) -> Dict[str, Any]:
    feats = detail_obj.get("adult_v2_attribution_features", {}) or {}
    return {
        "target_is_primary_column": as_int01(feats.get("target_is_primary_column", 0)),
        "target_is_context_only_column": as_int01(feats.get("target_is_context_only_column", 0)),
        "is_relationship_spouse_value": as_int01(feats.get("is_relationship_spouse_value", 0)),
        "relationship_marital_conflict": as_int01(feats.get("relationship_marital_conflict", 0)),
        "relationship_age_conflict": as_int01(feats.get("relationship_age_conflict", 0)),
        "relationship_sex_conflict": as_int01(feats.get("relationship_sex_conflict", 0)),
        "sex_value_invalid": as_int01(feats.get("sex_value_invalid", 0)),
        "education_age_extreme_conflict": as_int01(feats.get("education_age_extreme_conflict", 0)),
        "has_relationship_v2_rule": as_int01(feats.get("has_relationship_v2_rule", 0)),
        "has_education_v2_rule": as_int01(feats.get("has_education_v2_rule", 0)),
    }


def build_adult_v2_attribution_bucket(row: pd.Series, cfg: SamplingConfig) -> str:
    col = str(row.get("column", ""))
    rule = str(row.get("main_rule_type", ""))
    if as_int01(row.get("relationship_marital_conflict", 0)) == 1:
        return "relationship_marital_conflict"
    if as_int01(row.get("relationship_age_conflict", 0)) == 1:
        return "relationship_age_conflict"
    if as_int01(row.get("relationship_sex_conflict", 0)) == 1:
        return "relationship_sex_conflict"
    if as_int01(row.get("has_relationship_v2_rule", 0)) == 1 or rule in RELATIONSHIP_V2_RULE_TYPES:
        return "relationship_v2_rule"
    if as_int01(row.get("sex_value_invalid", 0)) == 1:
        return "sex_value_invalid"
    if as_int01(row.get("education_age_extreme_conflict", 0)) == 1 or rule in EDUCATION_V2_RULE_TYPES:
        return "education_age_extreme_conflict"
    if col in cfg.primary_target_columns:
        return "primary_target_other"
    if col in cfg.context_only_columns:
        return "context_only_other"
    return "unknown_attribution"


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

SOCCER_CONTEXT_RULE_TYPES = {
    "team_context_dominant",
    "team_season_manager_consistency_hint",
    "city_stadium_consistency_hint",
    "dominant_value_by_context",
    "dominant_value_by_context_pair",
}

SOCCER_FD_RULE_TYPES = {"functional_dependency"}
SOCCER_EVIDENCE_ONLY_FD_RULE_TYPES = {"soft_functional_dependency"}


def is_strong_or_format_rule(rule_type: Any) -> bool:
    r = safe_str(rule_type, "none")
    return r in SOCCER_STRONG_RULE_TYPES or r in SOCCER_FORMAT_RULE_TYPES or r in {"not_null", "high_risk_missing"}


def build_soccer_signal_bucket(row: pd.Series) -> str:
    rule = safe_str(row.get("main_rule_type"), "none")
    if safe_int(row.get("birthyear_value_invalid"), 0) == 1:
        return "birthyear_invalid_signal"
    if safe_int(row.get("season_value_invalid"), 0) == 1:
        return "season_invalid_signal"
    if safe_int(row.get("position_value_invalid"), 0) == 1:
        return "position_invalid_signal"
    if safe_int(row.get("position_needs_canonicalization"), 0) == 1:
        return "position_canonicalization_signal"
    if safe_int(row.get("birthyear_season_age_conflict"), 0) == 1:
        return "birthyear_season_age_signal"
    if safe_int(row.get("format_rule_signal"), 0) == 1:
        return "format_signal"
    if is_strong_or_format_rule(rule):
        return "strong_rule_signal"
    if rule in SOCCER_FD_RULE_TYPES:
        return "fd_signal"
    if rule in SOCCER_EVIDENCE_ONLY_FD_RULE_TYPES:
        return "evidence_only_fd_signal"
    if rule in SOCCER_CONTEXT_RULE_TYPES:
        return "context_signal"
    if rule in CONFIGS["soccer"].weak_rule_types:
        return "weak_rule_signal"
    return "generic_signal"


# ============================================================
# 5. 读取 summary，补齐字段与分桶字段
# ============================================================

def ensure_required_columns(df: pd.DataFrame, cfg: SamplingConfig) -> pd.DataFrame:
    required_defaults = {
        "row_id": 0,
        "column": MISSING_TEXT,
        "value": None,
        "semantic_type": "unknown",
        "detected_type": "unknown",
        "violation_count": 0,
        "conflict_score": 0.0,
        "value_frequency": 0,
        "value_frequency_rank": np.nan,
        "neighbor_majority_value": None,
        "neighbor_majority_ratio": np.nan,
        "prior_error_probability": 0.0,
        "posterior_error_probability": 0.0,
        "main_rule_type": "none",
        "main_usage_role": "none",
        "strong_rule_count": 0,
        "candidate_generation_rule_count": 0,
        "fd_like_count": 0,
        "context_rule_count": 0,
        "global_rule_count": 0,
        "rare_value_count": 0,
        "typo_rule_count": 0,
        "pattern_rule_count": 0,
        "schema_rule_count": 0,
    }

    if cfg.name == "flights":
        required_defaults.update({
            "time_window_rule_count": 0,
            "time_value_minutes": np.nan,
        })

    if cfg.name == "beers":
        required_defaults.update({
            "numeric_window_rule_count": 0,
            "numeric_value": np.nan,
        })

    if cfg.name == "rayyan":
        required_defaults.update({
            "canonical_rule_count": 0,
            "numeric_value": np.nan,
            "date_value": None,
            "language_canonical": None,
        })

    if cfg.name == "adult":
        required_defaults.update({
            "adult_domain_rule_count": 0,
            "adult_format_rule_count": 0,
            "adult_consistency_rule_count": 0,
            "age_lower": np.nan,
            "age_upper": np.nan,
            "hours_lower": np.nan,
            "hours_upper": np.nan,
            "education_order": np.nan,
            "adult_profile_inconsistency_count": 0,
            "adult_consistency_score": 0.0,
            "domain_valid": np.nan,
            "adult_value_bucket": "unknown",
            "target_is_primary_column": 0,
            "target_is_context_only_column": 0,
            "is_relationship_spouse_value": 0,
            "relationship_marital_conflict": 0,
            "relationship_age_conflict": 0,
            "relationship_sex_conflict": 0,
            "sex_value_invalid": 0,
            "education_age_extreme_conflict": 0,
            "has_relationship_v2_rule": 0,
            "has_education_v2_rule": 0,
        })

    if cfg.name == "soccer":
        required_defaults.update({
            "evidence_only_fd_count": 0,
            "soccer_domain_rule_count": 0,
            "soccer_format_rule_count": 0,
            "soccer_numeric_rule_count": 0,
            "soccer_consistency_rule_count": 0,
            "time_window_rule_count": 0,
            "time_value_minutes": np.nan,
            "time_window_bucket": "soccer_not_applicable",
            "time_value_bucket": "soccer_not_applicable",
            "row_missing_count": 0,
            "row_non_missing_count": 0,
            "year_value": np.nan,
            "row_birthyear_int": np.nan,
            "row_season_int": np.nan,
            "row_age_at_season": np.nan,
            "canonical_position": None,
            "row_position_canonical": None,
            "domain_valid": np.nan,
            "soccer_value_bucket": "unknown",
            "soccer_consistency_score": 0.0,
            "soccer_evidence_flags": "",
            "soccer_profile_inconsistency_count": 0,
            "target_is_primary_column": 0,
            "target_is_context_only_column": 0,
            "birthyear_value_invalid": 0,
            "season_value_invalid": 0,
            "position_value_invalid": 0,
            "position_needs_canonicalization": 0,
            "birthyear_season_age_conflict": 0,
            "team_context_conflict": 0,
            "format_rule_signal": 0,
            "fd_like_signal": 0,
            "soccer_attribution_bucket": "unknown_attribution",
            "adult_profile_inconsistency_count": 0,
            "dominant_pattern_ratio": np.nan,
        })

    for col, default in required_defaults.items():
        if col not in df.columns:
            df[col] = default

    df["row_id"] = pd.to_numeric(df["row_id"], errors="coerce").fillna(0).astype(int)
    df["column"] = df["column"].fillna(MISSING_TEXT).astype(str)

    numeric_cols = [c for c in set(cfg.feature_numeric + ["row_id"]) if c in df.columns]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_summary_csv(path: str, detail_map: Dict[Tuple[int, str], Dict[str, Any]], cfg: SamplingConfig) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = ensure_required_columns(df, cfg)

    main_rule_types = []
    main_usage_roles = []
    pattern_buckets = []

    extra_detail_rows = []

    for _, row in df.iterrows():
        key = (int(row["row_id"]), str(row["column"]))
        detail_obj = detail_map.get(key, {})

        current_main_rule_type = row.get("main_rule_type", None)
        if pd.isna(current_main_rule_type) or current_main_rule_type is None or str(current_main_rule_type).strip() == "":
            current_main_rule_type = extract_main_rule_type_from_detail(detail_obj)

        current_main_usage_role = row.get("main_usage_role", None)
        if pd.isna(current_main_usage_role) or current_main_usage_role is None or str(current_main_usage_role).strip() == "":
            current_main_usage_role = extract_main_usage_role_from_detail(detail_obj)

        main_rule_types.append(current_main_rule_type)
        main_usage_roles.append(current_main_usage_role)
        pattern_buckets.append(build_pattern_bucket_from_detail(detail_obj))

        if cfg.name == "adult":
            extra_detail_rows.append({
                "adult_consistency_bucket": build_adult_consistency_bucket_from_detail(detail_obj),
                **extract_adult_v2_features_from_detail(detail_obj),
            })
        elif cfg.name == "flights":
            extra_detail_rows.append({"time_window_bucket": build_time_window_bucket_from_detail(detail_obj)})
        elif cfg.name == "beers":
            extra_detail_rows.append({"numeric_window_bucket": build_numeric_window_bucket_from_detail(detail_obj)})
        elif cfg.name == "rayyan":
            extra_detail_rows.append({"canonical_bucket": build_canonical_bucket_from_detail(detail_obj)})
        else:
            extra_detail_rows.append({})

    df["main_rule_type"] = main_rule_types
    df["main_usage_role"] = main_usage_roles
    df["pattern_bucket"] = pattern_buckets

    if extra_detail_rows:
        extra_df = pd.DataFrame(extra_detail_rows)
        for col in extra_df.columns:
            if col not in df.columns:
                df[col] = extra_df[col].values
            else:
                # adult 的 attribution 字段优先用 flat CSV，缺失再用 JSONL 兜底。
                if cfg.name == "adult" and col != "adult_consistency_bucket":
                    base = pd.to_numeric(df[col], errors="coerce")
                    fallback = pd.to_numeric(extra_df[col], errors="coerce")
                    df[col] = base.fillna(fallback).fillna(0).astype(int)
                else:
                    df[col] = df[col].where(df[col].notna(), extra_df[col])

    df["rarity_bucket"] = df.apply(
        lambda r: build_rarity_bucket(r.get("value_frequency"), r.get("value_frequency_rank")),
        axis=1,
    )
    df["neighbor_bucket"] = df.apply(
        lambda r: build_neighbor_bucket(r.get("neighbor_majority_ratio"), r.get("neighbor_majority_value"), r.get("value")),
        axis=1,
    )

    if cfg.name == "flights":
        df["time_value_bucket"] = df.apply(lambda r: build_time_value_bucket(r.get("time_value_minutes")), axis=1)
        if "time_window_bucket" not in df.columns:
            df["time_window_bucket"] = "no_time_window_signal"

    elif cfg.name == "beers":
        if "numeric_window_bucket" not in df.columns:
            df["numeric_window_bucket"] = "no_numeric_window_signal"
        df["numeric_value_bucket"] = df.apply(lambda r: build_numeric_value_bucket(r.get("column"), r.get("numeric_value")), axis=1)

    elif cfg.name == "rayyan":
        if "canonical_bucket" not in df.columns:
            df["canonical_bucket"] = "no_canonical_signal"
        df["date_value_bucket"] = df.apply(lambda r: build_date_value_bucket(r.get("date_value")), axis=1)
        df["language_bucket"] = df.apply(lambda r: build_language_bucket(r.get("language_canonical")), axis=1)

    elif cfg.name == "adult":
        df["target_is_primary_column"] = df.apply(
            lambda r: 1 if str(r.get("column")) in cfg.primary_target_columns else as_int01(r.get("target_is_primary_column", 0)),
            axis=1,
        )
        df["target_is_context_only_column"] = df.apply(
            lambda r: 1 if str(r.get("column")) in cfg.context_only_columns else as_int01(r.get("target_is_context_only_column", 0)),
            axis=1,
        )
        df["has_relationship_v2_rule"] = df.apply(
            lambda r: 1 if str(r.get("main_rule_type")) in RELATIONSHIP_V2_RULE_TYPES else as_int01(r.get("has_relationship_v2_rule", 0)),
            axis=1,
        )
        df["has_education_v2_rule"] = df.apply(
            lambda r: 1 if str(r.get("main_rule_type")) in EDUCATION_V2_RULE_TYPES else as_int01(r.get("has_education_v2_rule", 0)),
            axis=1,
        )
        if "adult_consistency_bucket" not in df.columns:
            df["adult_consistency_bucket"] = "no_adult_consistency_signal"
        df["domain_bucket"] = df.apply(build_domain_bucket, axis=1)
        df["age_sampling_bucket"] = df.apply(lambda r: build_age_bucket_for_sampling(r.get("age_lower"), r.get("age_upper")), axis=1)
        df["hours_sampling_bucket"] = df.apply(lambda r: build_hours_bucket_for_sampling(r.get("hours_lower"), r.get("hours_upper")), axis=1)
        df["adult_v2_attribution_bucket"] = df.apply(lambda r: build_adult_v2_attribution_bucket(r, cfg), axis=1)

    elif cfg.name == "soccer":
        rule = df["main_rule_type"].astype(str)
        df["target_is_primary_column"] = df["column"].isin(cfg.primary_target_columns).astype(int)
        df["target_is_context_only_column"] = df["column"].isin(cfg.context_only_columns).astype(int)

        df["birthyear_value_invalid"] = np.where(
            rule.eq("birthyear_numeric_range"),
            1,
            pd.to_numeric(df["birthyear_value_invalid"], errors="coerce").fillna(0),
        ).astype(int)
        df["season_value_invalid"] = np.where(
            rule.eq("season_numeric_range"),
            1,
            pd.to_numeric(df["season_value_invalid"], errors="coerce").fillna(0),
        ).astype(int)
        df["position_value_invalid"] = np.where(
            rule.eq("position_domain"),
            1,
            pd.to_numeric(df["position_value_invalid"], errors="coerce").fillna(0),
        ).astype(int)
        df["birthyear_season_age_conflict"] = np.where(
            rule.eq("birthyear_season_age_consistency"),
            1,
            pd.to_numeric(df["birthyear_season_age_conflict"], errors="coerce").fillna(0),
        ).astype(int)
        df["team_context_conflict"] = np.where(
            rule.isin(SOCCER_CONTEXT_RULE_TYPES),
            1,
            pd.to_numeric(df["team_context_conflict"], errors="coerce").fillna(0),
        ).astype(int)
        df["format_rule_signal"] = np.where(
            rule.isin(SOCCER_FORMAT_RULE_TYPES),
            1,
            pd.to_numeric(df["format_rule_signal"], errors="coerce").fillna(0),
        ).astype(int)
        df["fd_like_signal"] = np.where(
            rule.isin(SOCCER_FD_RULE_TYPES),
            1,
            pd.to_numeric(df["fd_like_signal"], errors="coerce").fillna(0),
        ).astype(int)

        soft_fd_mask = rule.isin(SOCCER_EVIDENCE_ONLY_FD_RULE_TYPES)
        if soft_fd_mask.any():
            df.loc[soft_fd_mask, "evidence_only_fd_count"] = (
                pd.to_numeric(df.loc[soft_fd_mask, "evidence_only_fd_count"], errors="coerce").fillna(0) + 1
            )
            df.loc[soft_fd_mask, "fd_like_signal"] = 0

        missing_attr = df["soccer_attribution_bucket"].astype(str).isin(
            ["", "missing", "unknown", "unknown_attribution", "nan", "None"]
        )
        df.loc[missing_attr & (df["birthyear_value_invalid"] == 1), "soccer_attribution_bucket"] = "birthyear_invalid"
        df.loc[missing_attr & (df["season_value_invalid"] == 1), "soccer_attribution_bucket"] = "season_invalid"
        df.loc[missing_attr & (df["position_value_invalid"] == 1), "soccer_attribution_bucket"] = "position_invalid"
        df.loc[missing_attr & (pd.to_numeric(df["position_needs_canonicalization"], errors="coerce").fillna(0).astype(int) == 1), "soccer_attribution_bucket"] = "position_canonicalization"
        df.loc[missing_attr & (df["birthyear_season_age_conflict"] == 1), "soccer_attribution_bucket"] = "birthyear_season_age_conflict"
        df.loc[missing_attr & (df["team_context_conflict"] == 1), "soccer_attribution_bucket"] = "team_context_conflict"
        df.loc[missing_attr & (df["format_rule_signal"] == 1), "soccer_attribution_bucket"] = "format_pattern_conflict"
        df.loc[missing_attr & (df["fd_like_signal"] == 1), "soccer_attribution_bucket"] = "fd_like_conflict"
        df.loc[missing_attr & df["column"].isin(cfg.primary_target_columns), "soccer_attribution_bucket"] = "primary_target_other"
        df["soccer_signal_bucket"] = df.apply(build_soccer_signal_bucket, axis=1)

    return df


# ============================================================
# 6. 分桶和聚类
# ============================================================

def get_bucket_keys(cfg: SamplingConfig, disabled_bucket_fields: Set[str]) -> List[str]:
    if cfg.name == "soccer":
        # soccer 使用定制 bucket_id，不走这个列表。
        return []
    keys = list(cfg.bucket_keys)
    if cfg.use_column_in_bucket:
        keys = ["column"] + keys
    keys = [k for k in keys if k not in disabled_bucket_fields]
    return keys


def build_bucket_id(row: pd.Series, keys: List[str]) -> str:
    parts = []
    for key in keys:
        val = row.get(key, "unknown")
        if pd.isna(val):
            val = "NULL"
        parts.append(f"{key}={val}")
    return " | ".join(parts)


def assign_buckets(df: pd.DataFrame, cfg: SamplingConfig, disabled_bucket_fields: Set[str]) -> pd.DataFrame:
    out = df.copy()

    if cfg.name == "soccer":
        parts_list = []
        for _, row in out.iterrows():
            parts = []
            if cfg.use_column_in_bucket and "column" not in disabled_bucket_fields:
                parts.append(f"col={safe_str(row.get('column'))}")
            if cfg.use_rule_in_bucket and "main_rule_type" not in disabled_bucket_fields:
                parts.append(f"rule={safe_str(row.get('main_rule_type'), 'none')}")
            if cfg.use_dataset_signal_in_bucket and "soccer_signal_bucket" not in disabled_bucket_fields:
                parts.append(f"sig={safe_str(row.get('soccer_signal_bucket'), 'generic_signal')}")
            if "neighbor_bucket" not in disabled_bucket_fields:
                parts.append(f"nb={safe_str(row.get('neighbor_bucket'))}")
            if "pattern_bucket" not in disabled_bucket_fields:
                parts.append(f"pat={safe_str(row.get('pattern_bucket'))}")
            if not parts:
                parts.append("all")
            parts_list.append("||".join(parts))
        out["bucket_id"] = parts_list
        return out

    keys = get_bucket_keys(cfg, disabled_bucket_fields)
    if not keys:
        out["bucket_id"] = "all"
    else:
        out["bucket_id"] = out.apply(lambda r: build_bucket_id(r, keys), axis=1)
    return out


def prepare_features(bucket_df: pd.DataFrame, cfg: SamplingConfig, disabled_feature_columns: Set[str]) -> Tuple[np.ndarray, List[str]]:
    work = bucket_df.copy()

    numeric_cols = [c for c in cfg.feature_numeric if c not in disabled_feature_columns]
    categorical_cols = [c for c in cfg.feature_categorical if c not in disabled_feature_columns]

    for col in numeric_cols:
        if col not in work.columns:
            work[col] = 0.0
        work[col] = pd.to_numeric(work[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    num_df = work[numeric_cols].copy() if numeric_cols else pd.DataFrame(index=work.index)

    cat_frames = []
    for col in categorical_cols:
        if col not in work.columns:
            work[col] = "unknown"
        dummies = pd.get_dummies(work[col].fillna("unknown").astype(str), prefix=col)
        cat_frames.append(dummies)

    if cat_frames:
        feat_df = pd.concat([num_df] + cat_frames, axis=1)
    else:
        feat_df = num_df

    if feat_df.shape[1] == 0:
        feat_df = pd.DataFrame({"constant": np.zeros(len(work))}, index=work.index)

    scaler = StandardScaler()
    X = scaler.fit_transform(feat_df.values.astype(float))
    return X, feat_df.columns.tolist()


def choose_k(n: int, cfg: SamplingConfig) -> int:
    if n < cfg.min_bucket_size_for_cluster:
        return 1
    if cfg.choose_k_mode == "ceil":
        k = int(math.ceil(n / cfg.target_cluster_size))
    else:
        k = round(n / cfg.target_cluster_size)
    k = max(2, int(k))
    k = min(k, cfg.max_clusters_per_bucket, n)
    return int(k)


def cluster_bucket(bucket_df: pd.DataFrame, cfg: SamplingConfig, disabled_feature_columns: Set[str]) -> pd.DataFrame:
    bucket_df = bucket_df.copy()
    n = len(bucket_df)

    if n <= 1:
        bucket_df["cluster_id"] = 0
        bucket_df["dist_to_center"] = 0.0
        bucket_df["cluster_size"] = n
        return bucket_df

    X, _ = prepare_features(bucket_df, cfg, disabled_feature_columns)
    k = choose_k(n, cfg)

    if k <= 1:
        bucket_df["cluster_id"] = 0
        center = X.mean(axis=0, keepdims=True)
        dist = np.linalg.norm(X - center, axis=1)
        bucket_df["dist_to_center"] = dist
        bucket_df["cluster_size"] = n
        return bucket_df

    model = KMeans(n_clusters=k, random_state=cfg.random_state, n_init=10)
    labels = model.fit_predict(X)
    centers = model.cluster_centers_

    dists = np.zeros(len(X), dtype=float)
    for i in range(len(X)):
        dists[i] = np.linalg.norm(X[i] - centers[labels[i]])

    bucket_df["cluster_id"] = labels.astype(int)
    bucket_df["dist_to_center"] = dists.astype(float)
    size_map = Counter(labels.tolist())
    bucket_df["cluster_size"] = bucket_df["cluster_id"].map(size_map).astype(int)
    return bucket_df


def cluster_candidates(df: pd.DataFrame, cfg: SamplingConfig, disabled_feature_columns: Set[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    clustered_parts = []
    bucket_summary_rows = []

    for bucket_id, bucket_df in df.groupby("bucket_id", sort=False, dropna=False):
        bucket_clustered = cluster_bucket(bucket_df, cfg, disabled_feature_columns)

        for cluster_id, cluster_df in bucket_clustered.groupby("cluster_id", sort=False, dropna=False):
            row = {
                "bucket_id": bucket_id,
                "bucket_size": int(len(bucket_df)),
                "cluster_id": int(cluster_id),
                "cluster_size": int(len(cluster_df)),
            }

            for key in [
                "semantic_type", "main_rule_type", "main_usage_role", "pattern_bucket",
                "rarity_bucket", "neighbor_bucket", "time_window_bucket", "time_value_bucket",
                "numeric_window_bucket", "numeric_value_bucket", "canonical_bucket",
                "date_value_bucket", "language_bucket", "domain_bucket",
                "adult_consistency_bucket", "adult_v2_attribution_bucket",
                "age_sampling_bucket", "hours_sampling_bucket", "soccer_signal_bucket",
                "soccer_attribution_bucket",
            ]:
                if key in cluster_df.columns:
                    row[key] = str(cluster_df[key].iloc[0])

            if "column" in cluster_df.columns:
                row["column_top"] = safe_str(cluster_df["column"].mode().iloc[0])
            if "main_rule_type" in cluster_df.columns:
                row["main_rule_type_top"] = safe_str(cluster_df["main_rule_type"].mode().iloc[0])
            if "posterior_error_probability" in cluster_df.columns:
                row["mean_posterior"] = float(pd.to_numeric(cluster_df["posterior_error_probability"], errors="coerce").fillna(0).mean())
            if "conflict_score" in cluster_df.columns:
                row["mean_conflict_score"] = float(pd.to_numeric(cluster_df["conflict_score"], errors="coerce").fillna(0).mean())

            bucket_summary_rows.append(row)

        clustered_parts.append(bucket_clustered)

    clustered_df = pd.concat(clustered_parts, ignore_index=True) if clustered_parts else df.iloc[0:0].copy()
    bucket_summary_df = pd.DataFrame(bucket_summary_rows)
    return clustered_df, bucket_summary_df


# ============================================================
# 7. 初步代表样本采样
# ============================================================

def add_sample(samples: Dict[Tuple[int, str], Dict[str, Any]], row: pd.Series, role: str):
    key = (int(row["row_id"]), str(row["column"]))
    row_dict = row.to_dict()
    old = samples.get(key)
    if old is None:
        row_dict["sample_role"] = role
        samples[key] = row_dict
    else:
        old_role = safe_str(old.get("sample_role"), "")
        if role not in old_role.split("+"):
            old["sample_role"] = old_role + "+" + role if old_role else role
        samples[key] = old


def sample_from_cluster_generic(cluster_df: pd.DataFrame, cfg: SamplingConfig) -> pd.DataFrame:
    cluster_df = cluster_df.copy().sort_values("dist_to_center").reset_index(drop=True)
    n = len(cluster_df)

    raw_col = "is_sampled_raw" if cfg.budget_mode == "adult" else "is_sampled"
    cluster_df["sample_role"] = ""
    cluster_df[raw_col] = 0

    if n <= cfg.small_bucket_full_sample_threshold:
        cluster_df["sample_role"] = "all"
        cluster_df[raw_col] = 1
        return cluster_df

    sampled_indices: List[int] = []
    roles: Dict[int, str] = {}

    def mark(idx: int, role: str):
        if idx < 0 or idx >= n:
            return
        if idx not in sampled_indices:
            sampled_indices.append(idx)
            roles[idx] = role
        else:
            roles[idx] = roles[idx] + "+" + role if role not in roles[idx].split("+") else roles[idx]

    if cfg.sample_center:
        mark(0, "center")
    if cfg.sample_outlier:
        mark(n - 1, "outlier")
    if cfg.sample_boundary:
        mark(max(0, int(round(0.75 * (n - 1)))), "boundary")
    if cfg.sample_high_posterior and "posterior_error_probability" in cluster_df.columns:
        p = pd.to_numeric(cluster_df["posterior_error_probability"], errors="coerce").fillna(0.0)
        mark(int(p.idxmax()), "high_posterior")
    if cfg.sample_low_posterior and "posterior_error_probability" in cluster_df.columns:
        p = pd.to_numeric(cluster_df["posterior_error_probability"], errors="coerce").fillna(0.0)
        mark(int(p.idxmin()), "low_posterior")

    if cfg.max_samples_per_cluster is not None and len(sampled_indices) > cfg.max_samples_per_cluster:
        # 保持角色优先级：center > outlier > boundary > high posterior > low posterior
        role_rank = {"center": 5, "outlier": 4, "boundary": 3, "high_posterior": 2, "low_posterior": 1}
        sampled_indices = sorted(
            sampled_indices,
            key=lambda idx: max(role_rank.get(r, 0) for r in roles[idx].split("+")),
            reverse=True,
        )[:cfg.max_samples_per_cluster]

    for idx in sampled_indices:
        cluster_df.loc[idx, "sample_role"] = roles[idx]
        cluster_df.loc[idx, raw_col] = 1

    return cluster_df


def add_raw_cluster_samples(clustered_df: pd.DataFrame, cfg: SamplingConfig) -> pd.DataFrame:
    sampled_parts = []
    for (_, _), sub in clustered_df.groupby(["bucket_id", "cluster_id"], sort=False, dropna=False):
        sampled_parts.append(sample_from_cluster_generic(sub, cfg))
    return pd.concat(sampled_parts, ignore_index=True) if sampled_parts else clustered_df.iloc[0:0].copy()


# ============================================================
# 8. 预算优先级和 cap
# ============================================================

def compute_adult_signal_priority(row: pd.Series, cfg: SamplingConfig) -> float:
    score = 0.0
    col_name = str(row.get("column", ""))

    if col_name == "relationship":
        score += 6.0
    elif col_name == "sex":
        score += 5.0
    elif col_name == "education":
        score += 3.0
    elif col_name in cfg.context_only_columns:
        score -= 4.0

    score += safe_float(row.get("relationship_marital_conflict"), 0) * 8.0
    score += safe_float(row.get("relationship_age_conflict"), 0) * 7.0
    score += safe_float(row.get("relationship_sex_conflict"), 0) * 6.0
    score += safe_float(row.get("has_relationship_v2_rule"), 0) * 5.0
    score += safe_float(row.get("sex_value_invalid"), 0) * 5.5
    score += safe_float(row.get("education_age_extreme_conflict"), 0) * 4.5
    score += safe_float(row.get("has_education_v2_rule"), 0) * 3.5

    score += safe_float(row.get("strong_rule_count"), 0) * 3.0
    score += safe_float(row.get("adult_domain_rule_count"), 0) * 2.8
    score += safe_float(row.get("adult_format_rule_count"), 0) * 2.5
    score += safe_float(row.get("adult_consistency_rule_count"), 0) * 2.2
    score += safe_float(row.get("schema_rule_count"), 0) * 1.8
    score += safe_float(row.get("fd_like_count"), 0) * 1.2
    score += safe_float(row.get("context_rule_count"), 0) * 1.0
    score += safe_float(row.get("rare_value_count"), 0) * 0.2
    score += safe_float(row.get("pattern_rule_count"), 0) * 0.4

    score += min(safe_float(row.get("conflict_score"), 0) / 20.0, 5.0)
    score += safe_float(row.get("adult_consistency_score"), 0) * 1.5
    score += safe_float(row.get("posterior_error_probability"), 0) * 2.0

    nb = str(row.get("neighbor_bucket", ""))
    if nb == "strong_support_other":
        score += 2.0
    elif nb == "medium_support_other":
        score += 1.0
    elif nb == "strong_support_current":
        score -= 0.5

    db = str(row.get("domain_bucket", ""))
    if db == "domain_invalid":
        score += 2.5

    acb = str(row.get("adult_consistency_bucket", ""))
    if acb == "strong_adult_consistency_signal":
        score += 2.0
    elif acb == "adult_consistency_flag":
        score += 1.2

    return float(score)


def compute_soccer_priority_score(df: pd.DataFrame, cfg: SamplingConfig) -> pd.Series:
    posterior = pd.to_numeric(df.get("posterior_error_probability"), errors="coerce").fillna(0.0)
    conflict = pd.to_numeric(df.get("conflict_score"), errors="coerce").fillna(0.0)
    strong = pd.to_numeric(df.get("strong_rule_count"), errors="coerce").fillna(0.0)
    soccer_score = pd.to_numeric(df.get("soccer_consistency_score"), errors="coerce").fillna(0.0)

    domain = pd.to_numeric(df.get("soccer_domain_rule_count"), errors="coerce").fillna(0.0)
    fmt = pd.to_numeric(df.get("soccer_format_rule_count"), errors="coerce").fillna(0.0)
    numeric = pd.to_numeric(df.get("soccer_numeric_rule_count"), errors="coerce").fillna(0.0)
    cons = pd.to_numeric(df.get("soccer_consistency_rule_count"), errors="coerce").fillna(0.0)

    birthyear_season = (
        pd.to_numeric(df.get("birthyear_value_invalid", 0), errors="coerce").fillna(0.0)
        + pd.to_numeric(df.get("season_value_invalid", 0), errors="coerce").fillna(0.0)
        + pd.to_numeric(df.get("birthyear_season_age_conflict", 0), errors="coerce").fillna(0.0)
    )
    position_signal = (
        pd.to_numeric(df.get("position_value_invalid", 0), errors="coerce").fillna(0.0)
        + pd.to_numeric(df.get("position_needs_canonicalization", 0), errors="coerce").fillna(0.0)
    )
    format_signal = pd.to_numeric(df.get("format_rule_signal", 0), errors="coerce").fillna(0.0)

    rule = df["main_rule_type"].astype(str) if "main_rule_type" in df.columns else pd.Series(["none"] * len(df), index=df.index)
    col = df["column"].astype(str) if "column" in df.columns else pd.Series(["missing"] * len(df), index=df.index)

    important_rule_bonus = rule.isin(cfg.important_rule_types).astype(float) * 2.0
    focus_col_bonus = col.isin(cfg.high_recall_focus_columns).astype(float) * 1.6
    weak_penalty = rule.isin(cfg.weak_rule_types).astype(float) * 2.2
    soft_fd_penalty = rule.isin(SOCCER_EVIDENCE_ONLY_FD_RULE_TYPES).astype(float) * 3.0

    score = (
        posterior * 2.0
        + conflict * 0.08
        + strong * 1.5
        + soccer_score * 1.1
        + domain * 2.0
        + fmt * 2.0
        + numeric * 2.0
        + cons * 1.6
        + birthyear_season * 2.0
        + position_signal * 2.0
        + format_signal * 1.8
        + important_rule_bonus
        + focus_col_bonus
        - weak_penalty
        - soft_fd_penalty
    )
    return score.astype(float)


def compute_priority(df: pd.DataFrame, cfg: SamplingConfig) -> pd.Series:
    if cfg.budget_mode == "adult":
        return df.apply(lambda r: compute_adult_signal_priority(r, cfg), axis=1).astype(float)
    if cfg.budget_mode == "soccer":
        return compute_soccer_priority_score(df, cfg)
    return (
        pd.to_numeric(df.get("posterior_error_probability"), errors="coerce").fillna(0.0) * 2.0
        + pd.to_numeric(df.get("conflict_score"), errors="coerce").fillna(0.0) * 0.05
        + pd.to_numeric(df.get("strong_rule_count"), errors="coerce").fillna(0.0)
    ).astype(float)


def is_adult_weak_signal_row(row: pd.Series) -> bool:
    keys = [
        "strong_rule_count", "adult_domain_rule_count", "adult_format_rule_count",
        "adult_consistency_rule_count", "schema_rule_count", "fd_like_count",
        "context_rule_count", "has_relationship_v2_rule", "has_education_v2_rule",
        "sex_value_invalid",
    ]
    total = 0.0
    for k in keys:
        total += safe_float(row.get(k), 0.0)
    return total <= 0


def downsample_adult_weak_signals(sampled_df: pd.DataFrame, cfg: SamplingConfig) -> pd.DataFrame:
    if not cfg.enable_weak_signal_downsample or len(sampled_df) == 0:
        return sampled_df

    sampled_df = sampled_df.copy()
    sampled_df["_is_weak_signal"] = sampled_df.apply(is_adult_weak_signal_row, axis=1).astype(int)
    strong_df = sampled_df[sampled_df["_is_weak_signal"] == 0].copy()
    weak_df = sampled_df[sampled_df["_is_weak_signal"] == 1].copy()

    if len(weak_df) == 0:
        return sampled_df.drop(columns=["_is_weak_signal"], errors="ignore")

    keep_n = max(1, int(round(len(weak_df) * cfg.weak_signal_keep_ratio)))
    weak_df = weak_df.sort_values(
        by=["signal_priority", "conflict_score", "posterior_error_probability"],
        ascending=[False, False, False],
    ).head(keep_n)

    out = pd.concat([strong_df, weak_df], ignore_index=True)
    return out.drop(columns=["_is_weak_signal"], errors="ignore")


def enforce_bucket_budget(sampled_df: pd.DataFrame, cfg: SamplingConfig) -> pd.DataFrame:
    if len(sampled_df) == 0 or cfg.max_samples_per_bucket is None:
        return sampled_df

    kept = []
    for bucket_id, group in sampled_df.groupby("bucket_id", sort=False, dropna=False):
        if len(group) <= cfg.max_samples_per_bucket:
            kept.append(group)
            continue

        sort_cols = [c for c in ["signal_priority", "sample_role_priority", "conflict_score", "posterior_error_probability", "_priority_score"] if c in group.columns]
        ascending = []
        for c in sort_cols:
            ascending.append(c == "dist_to_center")
        if "dist_to_center" in group.columns and "dist_to_center" not in sort_cols:
            sort_cols.append("dist_to_center")
            ascending.append(True)

        if not sort_cols:
            g = group.head(cfg.max_samples_per_bucket)
        else:
            g = group.sort_values(by=sort_cols, ascending=ascending).head(cfg.max_samples_per_bucket)
        kept.append(g)

    return pd.concat(kept, ignore_index=True) if kept else sampled_df.iloc[0:0].copy()


def enforce_column_budget(sampled_df: pd.DataFrame, cfg: SamplingConfig, preserve_rule_diversity: bool = False) -> pd.DataFrame:
    if len(sampled_df) == 0 or not cfg.max_samples_per_column:
        return sampled_df

    kept = []
    for col, group in sampled_df.groupby("column", sort=False, dropna=False):
        cap = cfg.max_samples_per_column.get(str(col), cfg.default_max_samples_per_column)
        if cap is None or len(group) <= cap:
            kept.append(group)
            continue

        g = group.copy()
        if "signal_priority" not in g.columns:
            g["signal_priority"] = compute_priority(g, cfg)
        if "_priority_score" not in g.columns:
            g["_priority_score"] = g["signal_priority"]

        if preserve_rule_diversity and "main_rule_type" in g.columns:
            keep_idx = []
            for _, sub in g.groupby("main_rule_type", dropna=False):
                rule_name = safe_str(sub["main_rule_type"].iloc[0], "none")
                per_rule_keep = 1 if rule_name in cfg.weak_rule_types else 2
                keep_idx.extend(sub.sort_values("_priority_score", ascending=False).head(per_rule_keep).index.tolist())
            keep_idx = list(dict.fromkeys(keep_idx))
            remain = cap - len(keep_idx)
            if remain > 0:
                extra = g.drop(index=keep_idx, errors="ignore").sort_values("_priority_score", ascending=False).head(remain).index.tolist()
                keep_idx.extend(extra)
            keep_idx = keep_idx[:cap]
            kept.append(g.loc[keep_idx])
        else:
            kept.append(g.sort_values(
                by=["signal_priority", "sample_role_priority", "conflict_score", "posterior_error_probability"],
                ascending=[False, False, False, False],
            ).head(cap))

    return pd.concat(kept, ignore_index=True) if kept else sampled_df.iloc[0:0].copy()


def ensure_min_column_coverage(sampled_df: pd.DataFrame, clustered_df: pd.DataFrame, cfg: SamplingConfig) -> pd.DataFrame:
    if len(clustered_df) == 0:
        return sampled_df

    sampled_df = sampled_df.copy()
    selected_keys = set(zip(sampled_df["row_id"].astype(int), sampled_df["column"].astype(str)))
    add_parts = []

    for col, pool in clustered_df.groupby("column", sort=False, dropna=False):
        if cfg.budget_mode == "adult":
            min_need = cfg.min_samples_per_primary_column if col in cfg.primary_target_columns else cfg.min_samples_per_context_column
        elif cfg.budget_mode == "soccer":
            min_need = cfg.min_sample_per_active_column
            if str(col) in cfg.high_recall_focus_columns:
                min_need = max(min_need, 12)
        else:
            min_need = 0

        if min_need <= 0:
            continue

        current_n = int((sampled_df["column"] == col).sum()) if len(sampled_df) > 0 else 0
        need = max(0, min_need - current_n)
        if need <= 0:
            continue

        cap = cfg.max_samples_per_column.get(str(col), cfg.default_max_samples_per_column) if cfg.max_samples_per_column else None
        if cap is not None:
            need = min(need, cap - current_n)
        if need <= 0:
            continue

        p = pool.copy()
        p["_key"] = list(zip(p["row_id"].astype(int), p["column"].astype(str)))
        p = p[~p["_key"].isin(selected_keys)].copy()
        if len(p) == 0:
            continue

        if "signal_priority" not in p.columns:
            p["signal_priority"] = compute_priority(p, cfg)

        p = p.sort_values(
            by=["signal_priority", "conflict_score", "posterior_error_probability"],
            ascending=[False, False, False],
        ).head(need)
        p = p.drop(columns=["_key"], errors="ignore")
        if "sample_role" in p.columns:
            p["sample_role"] = p["sample_role"].replace("", "column_quota")
            p.loc[p["sample_role"].isna(), "sample_role"] = "column_quota"
        else:
            p["sample_role"] = "column_quota"
        add_parts.append(p)
        for _, r in p.iterrows():
            selected_keys.add((int(r["row_id"]), str(r["column"])))

    if add_parts:
        sampled_df = pd.concat([sampled_df] + add_parts, ignore_index=True)

    return sampled_df


def enforce_total_budget(sampled_df: pd.DataFrame, cfg: SamplingConfig) -> pd.DataFrame:
    if cfg.max_total_samples is None or len(sampled_df) <= cfg.max_total_samples:
        return sampled_df.reset_index(drop=True)

    df = sampled_df.copy()
    if "signal_priority" not in df.columns:
        df["signal_priority"] = compute_priority(df, cfg)
    if "_priority_score" not in df.columns:
        df["_priority_score"] = df["signal_priority"]

    if cfg.keep_column_diversity_in_global_budget:
        groups = {
            col: g.sort_values(
                by=["signal_priority", "sample_role_priority", "conflict_score", "posterior_error_probability"],
                ascending=[False, False, False, False],
            ).reset_index()
            for col, g in df.groupby("column", sort=False, dropna=False)
        }
        selected_indices = []
        round_idx = 0
        while len(selected_indices) < cfg.max_total_samples:
            added = False
            for _, g in groups.items():
                if round_idx < len(g):
                    selected_indices.append(int(g["index"].iloc[round_idx]))
                    added = True
                    if len(selected_indices) >= cfg.max_total_samples:
                        break
            if not added:
                break
            round_idx += 1
        out = df.loc[selected_indices].copy()
    else:
        keep_indices = []
        for col, sub in df.groupby("column", dropna=False):
            n_min = cfg.min_sample_per_active_column
            if str(col) in cfg.high_recall_focus_columns:
                n_min = max(n_min, 12)
            n = min(n_min, len(sub))
            keep_indices.extend(sub.sort_values("_priority_score", ascending=False).head(n).index.tolist())
        keep_indices = list(dict.fromkeys(keep_indices))

        for rule in cfg.important_rule_types:
            sub = df[df["main_rule_type"].astype(str) == rule]
            if len(sub) == 0:
                continue
            n = min(cfg.min_sample_per_important_rule, len(sub))
            extra = sub.drop(index=keep_indices, errors="ignore").sort_values("_priority_score", ascending=False).head(n).index.tolist()
            keep_indices.extend(extra)
            keep_indices = list(dict.fromkeys(keep_indices))

        remain = cfg.max_total_samples - len(keep_indices)
        if remain > 0:
            extra = df.drop(index=keep_indices, errors="ignore").sort_values("_priority_score", ascending=False).head(remain).index.tolist()
            keep_indices.extend(extra)

        keep_indices = keep_indices[:cfg.max_total_samples]
        out = df.loc[keep_indices].copy()

    if len(out) > cfg.max_total_samples:
        out = out.sort_values(
            by=["signal_priority", "sample_role_priority", "conflict_score", "posterior_error_probability"],
            ascending=[False, False, False, False],
        ).head(cfg.max_total_samples).copy()

    return out.reset_index(drop=True)


def enforce_weak_rule_cap(sample_df: pd.DataFrame, cfg: SamplingConfig) -> pd.DataFrame:
    if len(sample_df) == 0 or cfg.max_weak_rule_samples is None or "main_rule_type" not in sample_df.columns:
        return sample_df

    df = sample_df.copy()
    if "_priority_score" not in df.columns:
        df["_priority_score"] = compute_priority(df, cfg)

    weak_mask = df["main_rule_type"].astype(str).isin(cfg.weak_rule_types)
    weak_df = df[weak_mask].copy()
    strong_df = df[~weak_mask].copy()

    if len(weak_df) <= cfg.max_weak_rule_samples:
        return df.drop(columns=["_priority_score"], errors="ignore").reset_index(drop=True)

    weak_keep = weak_df.sort_values("_priority_score", ascending=False).head(cfg.max_weak_rule_samples)
    out = pd.concat([strong_df, weak_keep], ignore_index=True)
    return out.drop(columns=["_priority_score"], errors="ignore").reset_index(drop=True)


def add_quota_samples(clustered_df: pd.DataFrame, samples: Dict[Tuple[int, str], Dict[str, Any]], cfg: SamplingConfig):
    if len(clustered_df) == 0:
        return

    df = clustered_df.copy()
    df["_priority_score"] = compute_priority(df, cfg)

    # 1) 每个有候选的列保底
    for col, sub in df.groupby("column", dropna=False):
        n_min = cfg.min_sample_per_active_column
        if str(col) in cfg.high_recall_focus_columns:
            n_min = max(n_min, 12)
        n = min(n_min, len(sub))
        for _, row in sub.sort_values("_priority_score", ascending=False).head(n).iterrows():
            add_sample(samples, row, "column_quota")

    # 2) 重要规则保底
    for rule in cfg.important_rule_types:
        sub = df[df["main_rule_type"].astype(str) == rule]
        if len(sub) == 0:
            continue
        n = min(cfg.min_sample_per_important_rule, len(sub))
        for _, row in sub.sort_values("_priority_score", ascending=False).head(n).iterrows():
            add_sample(samples, row, "important_rule_quota")

    # 3) attribution bucket 保底
    if cfg.name == "soccer" and "soccer_attribution_bucket" in df.columns:
        important_buckets = {
            "birthyear_invalid",
            "season_invalid",
            "position_invalid",
            "position_canonicalization",
            "birthyear_season_age_conflict",
            "format_pattern_conflict",
            "primary_target_other",
        }
        for bucket in important_buckets:
            sub = df[df["soccer_attribution_bucket"].astype(str) == bucket]
            if len(sub) == 0:
                continue
            n = min(4, len(sub))
            for _, row in sub.sort_values("_priority_score", ascending=False).head(n).iterrows():
                add_sample(samples, row, "attribution_bucket_quota")


def sample_cluster_representatives_with_dict(clustered_df: pd.DataFrame, cfg: SamplingConfig) -> Dict[Tuple[int, str], Dict[str, Any]]:
    samples: Dict[Tuple[int, str], Dict[str, Any]] = {}

    for (_, _), sub in clustered_df.groupby(["bucket_id", "cluster_id"], dropna=False):
        sub = sub.copy().sort_values("dist_to_center").reset_index(drop=True)
        n = len(sub)

        if n <= cfg.small_bucket_full_sample_threshold:
            for _, row in sub.iterrows():
                add_sample(samples, row, "all")
            continue

        def add_by_pos(pos: int, role: str):
            if 0 <= pos < n:
                add_sample(samples, sub.iloc[pos], role)

        if cfg.sample_center:
            add_by_pos(0, "center")
        if cfg.sample_outlier:
            add_by_pos(n - 1, "outlier")
        if cfg.sample_boundary:
            add_by_pos(max(0, int(round(0.75 * (n - 1)))), "boundary")
        if cfg.sample_high_posterior and "posterior_error_probability" in sub.columns:
            p = pd.to_numeric(sub["posterior_error_probability"], errors="coerce").fillna(0.0)
            add_by_pos(int(p.idxmax()), "high_posterior")
        if cfg.sample_low_posterior and "posterior_error_probability" in sub.columns:
            p = pd.to_numeric(sub["posterior_error_probability"], errors="coerce").fillna(0.0)
            add_by_pos(int(p.idxmin()), "low_posterior")

    return samples


def apply_adult_sampling_budget(clustered_df: pd.DataFrame, disable_budget_caps: bool) -> pd.DataFrame:
    cfg = CONFIGS["adult"]
    work = clustered_df.copy()
    work["signal_priority"] = compute_priority(work, cfg)

    role_priority = {"all": 4, "center": 3, "outlier": 2, "boundary": 1, "": 0}
    work["sample_role_priority"] = work["sample_role"].map(lambda x: role_priority.get(str(x), 0))
    sampled = work[work["is_sampled_raw"] == 1].copy()
    sampled["_orig_idx"] = np.arange(len(sampled))

    if not disable_budget_caps:
        sampled = downsample_adult_weak_signals(sampled, cfg)
        sampled = enforce_bucket_budget(sampled, cfg)
        sampled = enforce_column_budget(sampled, cfg, preserve_rule_diversity=False)
        sampled = ensure_min_column_coverage(sampled, work, cfg)
        sampled = enforce_column_budget(sampled, cfg, preserve_rule_diversity=False)

        sampled = sampled.reset_index(drop=True)
        sampled["_orig_idx"] = np.arange(len(sampled))
        sampled = enforce_total_budget(sampled, cfg)

    sampled = sampled.copy()
    sampled["is_sampled"] = 1
    final_keys = set(zip(sampled["row_id"].astype(int), sampled["column"].astype(str)))

    work["is_sampled"] = work.apply(
        lambda r: 1 if (int(r["row_id"]), str(r["column"])) in final_keys else 0,
        axis=1,
    )
    return work


def apply_soccer_sampling(clustered_df: pd.DataFrame, disable_quota_sampling: bool, disable_budget_caps: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cfg = CONFIGS["soccer"]
    work = clustered_df.copy()
    samples = sample_cluster_representatives_with_dict(work, cfg)

    if not disable_quota_sampling:
        add_quota_samples(work, samples, cfg)

    sample_df = pd.DataFrame(list(samples.values()))
    if len(sample_df) == 0:
        work["is_sampled"] = 0
        work["sample_role"] = ""
        return work, sample_df

    sample_df["_priority_score"] = compute_priority(sample_df, cfg)

    if not disable_budget_caps:
        sample_df = enforce_weak_rule_cap(sample_df, cfg)
        sample_df = enforce_bucket_budget(sample_df, cfg)
        sample_df = enforce_column_budget(sample_df, cfg, preserve_rule_diversity=True)
        sample_df = enforce_total_budget(sample_df, cfg)

    sample_df["is_sampled"] = 1
    sample_df = sample_df.reset_index(drop=True)

    sampled_keys = set((int(r["row_id"]), str(r["column"])) for _, r in sample_df.iterrows())
    key_series = list(zip(work["row_id"].astype(int), work["column"].astype(str)))
    work["is_sampled"] = [1 if k in sampled_keys else 0 for k in key_series]
    role_map = {(int(r["row_id"]), str(r["column"])): r.get("sample_role", "sampled") for _, r in sample_df.iterrows()}
    work["sample_role"] = [role_map.get(k, "") for k in key_series]

    return work, sample_df


# ============================================================
# 9. JSONL 输出
# ============================================================

GENERIC_SUMMARY_FIELDS = [
    "semantic_type", "violation_count", "conflict_score", "value_frequency",
    "value_frequency_rank", "neighbor_majority_value", "neighbor_majority_ratio",
    "prior_error_probability", "posterior_error_probability", "main_rule_type",
    "main_usage_role", "strong_rule_count", "candidate_generation_rule_count",
    "fd_like_count", "context_rule_count", "global_rule_count", "rare_value_count",
    "typo_rule_count", "pattern_rule_count", "schema_rule_count",
    "pattern_bucket", "rarity_bucket", "neighbor_bucket",
]

EXTRA_SUMMARY_FIELDS_BY_DATASET = {
    "flights": ["time_window_rule_count", "time_value_minutes", "time_window_bucket", "time_value_bucket"],
    "beers": ["numeric_window_rule_count", "numeric_value", "numeric_window_bucket", "numeric_value_bucket"],
    "rayyan": ["canonical_rule_count", "numeric_value", "canonical_bucket", "date_value_bucket", "language_bucket"],
    "adult": [
        "adult_domain_rule_count", "adult_format_rule_count", "adult_consistency_rule_count",
        "adult_consistency_score", "adult_consistency_bucket", "domain_bucket",
        "adult_v2_attribution_bucket", "age_sampling_bucket", "hours_sampling_bucket",
        "signal_priority", "target_is_primary_column", "target_is_context_only_column",
        "has_relationship_v2_rule", "has_education_v2_rule", "sex_value_invalid",
    ],
    "soccer": [
        "soccer_domain_rule_count", "soccer_format_rule_count", "soccer_numeric_rule_count",
        "soccer_consistency_rule_count", "soccer_consistency_score", "soccer_signal_bucket",
        "soccer_attribution_bucket", "target_is_primary_column", "target_is_context_only_column",
        "birthyear_value_invalid", "season_value_invalid", "position_value_invalid",
        "position_needs_canonicalization", "birthyear_season_age_conflict", "team_context_conflict",
        "format_rule_signal", "fd_like_signal", "evidence_only_fd_count",
    ],
}


def row_to_summary_context(row: pd.Series, cfg: SamplingConfig) -> Dict[str, Any]:
    fields = GENERIC_SUMMARY_FIELDS + EXTRA_SUMMARY_FIELDS_BY_DATASET.get(cfg.name, [])
    out = {}
    for field in fields:
        if field in row.index:
            out[field] = json_safe_value(row.get(field))
    return out


def export_sampled_jsonl(sampled_df: pd.DataFrame, detail_map: Dict[Tuple[int, str], Dict[str, Any]], output_jsonl: str, cfg: SamplingConfig):
    ensure_parent_dir(output_jsonl)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for _, row in sampled_df.iterrows():
            key = (int(row["row_id"]), str(row["column"]))
            detail_obj = detail_map.get(key, {})

            if cfg.name == "soccer":
                obj = dict(detail_obj) if detail_obj else {
                    "row_id": int(row["row_id"]),
                    "column": str(row["column"]),
                    "value": json_safe_value(row.get("value")),
                    "llm_context_text": "",
                }
                obj["sampling_info"] = {
                    "bucket_id": safe_str(row.get("bucket_id")),
                    "cluster_id": safe_int(row.get("cluster_id"), -1),
                    "cluster_size": safe_int(row.get("cluster_size"), 0),
                    "dist_to_center": safe_float(row.get("dist_to_center"), 0.0),
                    "sample_role": safe_str(row.get("sample_role"), "sampled"),
                    "soccer_signal_bucket": safe_str(row.get("soccer_signal_bucket"), "generic_signal"),
                    "soccer_attribution_bucket": safe_str(row.get("soccer_attribution_bucket"), "unknown_attribution"),
                    "rarity_bucket": safe_str(row.get("rarity_bucket"), "unknown"),
                    "pattern_bucket": safe_str(row.get("pattern_bucket"), "unknown"),
                    "neighbor_bucket": safe_str(row.get("neighbor_bucket"), "unknown"),
                    "target_is_primary_column": safe_int(row.get("target_is_primary_column"), 0),
                    "target_is_context_only_column": safe_int(row.get("target_is_context_only_column"), 0),
                    "main_rule_type": safe_str(row.get("main_rule_type"), "none"),
                }
                f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
                continue

            out = {
                "row_id": int(row["row_id"]),
                "column": str(row["column"]),
                "value": json_safe_value(row.get("value")),
                "bucket_id": row.get("bucket_id"),
                "cluster_id": int(row.get("cluster_id", 0)),
                "sample_role": row.get("sample_role", ""),
                "dist_to_center": float(safe_float(row.get("dist_to_center"), 0.0)),
                "summary_context": row_to_summary_context(row, cfg),
                "detail_context": detail_obj,
            }
            if cfg.name == "adult":
                out["signal_priority"] = float(safe_float(row.get("signal_priority"), 0.0))
            f.write(json.dumps(out, ensure_ascii=False, default=json_safe_value) + "\n")


# ============================================================
# 10. Summary 输出
# ============================================================

def build_sample_summary(clustered_df: pd.DataFrame, sample_df: pd.DataFrame, cfg: SamplingConfig) -> Dict[str, Any]:
    summary = {
        "input_candidates": int(len(clustered_df)),
        "sampled_candidates": int(len(sample_df)),
        "sample_ratio": round(len(sample_df) / len(clustered_df), 6) if len(clustered_df) > 0 else 0.0,
        "bucket_count": int(clustered_df["bucket_id"].nunique()) if "bucket_id" in clustered_df.columns and len(clustered_df) else 0,
        "cluster_count": int(clustered_df[["bucket_id", "cluster_id"]].drop_duplicates().shape[0]) if {"bucket_id", "cluster_id"}.issubset(clustered_df.columns) and len(clustered_df) else 0,
        "column_distribution_input": clustered_df["column"].value_counts(dropna=False).to_dict() if "column" in clustered_df.columns else {},
        "column_distribution_sampled": sample_df["column"].value_counts(dropna=False).to_dict() if len(sample_df) and "column" in sample_df.columns else {},
        "rule_distribution_input": clustered_df["main_rule_type"].value_counts(dropna=False).to_dict() if "main_rule_type" in clustered_df.columns else {},
        "rule_distribution_sampled": sample_df["main_rule_type"].value_counts(dropna=False).to_dict() if len(sample_df) and "main_rule_type" in sample_df.columns else {},
    }

    if cfg.name == "adult":
        summary.update({
            "total_candidates": int(len(clustered_df)),
            "raw_sampled_count_before_budget": int(clustered_df["is_sampled_raw"].sum()) if "is_sampled_raw" in clustered_df.columns else 0,
            "final_sampled_count": int(len(sample_df)),
            "max_total_samples": int(cfg.max_total_samples or 0),
            "use_column_in_bucket": bool(cfg.use_column_in_bucket),
            "max_samples_per_bucket": int(cfg.max_samples_per_bucket or 0),
            "max_samples_per_cluster": int(cfg.max_samples_per_cluster or 0),
            "weak_signal_downsample_enabled": bool(cfg.enable_weak_signal_downsample),
            "weak_signal_keep_ratio": float(cfg.weak_signal_keep_ratio),
            "sampled_by_column": sample_df["column"].value_counts(dropna=False).to_dict() if len(sample_df) else {},
            "sample_role_distribution": sample_df["sample_role"].value_counts(dropna=False).to_dict() if len(sample_df) else {},
            "main_rule_type_distribution_top30": sample_df["main_rule_type"].value_counts(dropna=False).head(30).to_dict() if len(sample_df) else {},
            "adult_v2_attribution_bucket_distribution": sample_df["adult_v2_attribution_bucket"].value_counts(dropna=False).to_dict() if len(sample_df) and "adult_v2_attribution_bucket" in sample_df.columns else {},
        })

    if cfg.name == "soccer":
        summary.update({
            "global_sample_budget": int(cfg.max_total_samples or 0),
            "max_weak_rule_samples": int(cfg.max_weak_rule_samples or 0),
            "soccer_signal_distribution_sampled": sample_df["soccer_signal_bucket"].value_counts(dropna=False).to_dict() if len(sample_df) and "soccer_signal_bucket" in sample_df.columns else {},
            "soccer_attribution_distribution_sampled": sample_df["soccer_attribution_bucket"].value_counts(dropna=False).to_dict() if len(sample_df) and "soccer_attribution_bucket" in sample_df.columns else {},
            "weak_rule_sampled_count": int(sample_df["main_rule_type"].astype(str).isin(cfg.weak_rule_types).sum()) if len(sample_df) and "main_rule_type" in sample_df.columns else 0,
        })

    return summary


# ============================================================
# 11. 主流程
# ============================================================

def run_sampling(
    cfg: SamplingConfig,
    input_summary_csv: str,
    input_detail_jsonl: str,
    output_clustered_csv: str,
    output_sampled_csv: str,
    output_sampled_jsonl: str,
    output_bucket_summary_csv: str,
    output_sample_summary_json: Optional[str],
    args,
):
    disabled_bucket_fields = set(args.disable_bucket_fields or []) | set(env_list("DISABLE_BUCKET_FIELDS"))
    disabled_feature_columns = set(args.disable_feature_columns or []) | set(env_list("DISABLE_FEATURE_COLUMNS"))
    disable_quota_sampling = args.disable_quota_sampling or env_bool("DISABLE_QUOTA_SAMPLING", False)
    disable_budget_caps = args.disable_budget_caps or env_bool("DISABLE_BUDGET_CAPS", False)

    if args.max_total_samples is not None:
        cfg.max_total_samples = int(args.max_total_samples)

    print("[1/6] Loading detailed JSONL context map ...")
    detail_map = load_detail_jsonl(input_detail_jsonl)
    print(f"  detail JSONL records = {len(detail_map)}")

    print("[2/6] Loading context summary CSV ...")
    df = load_summary_csv(input_summary_csv, detail_map, cfg)
    print(f"  input candidates = {len(df)}")

    print("[3/6] Assigning buckets ...")
    df = assign_buckets(df, cfg, disabled_bucket_fields)
    print(f"  buckets = {df['bucket_id'].nunique() if len(df) else 0}")

    print("[4/6] Clustering candidates ...")
    clustered_df, bucket_summary_df = cluster_candidates(df, cfg, disabled_feature_columns)
    print(f"  clustered rows = {len(clustered_df)}")

    print("[5/6] Sampling representatives ...")
    if cfg.budget_mode == "soccer":
        clustered_df, sampled_df = apply_soccer_sampling(clustered_df, disable_quota_sampling, disable_budget_caps)
        if len(sampled_df) > 0:
            sampled_df = sampled_df.reset_index(drop=True)
    else:
        clustered_df = add_raw_cluster_samples(clustered_df, cfg)

        if cfg.budget_mode == "adult":
            clustered_df = apply_adult_sampling_budget(clustered_df, disable_budget_caps)
            sampled_df = clustered_df[clustered_df["is_sampled"] == 1].copy()
            sampled_df = sampled_df.sort_values(
                by=["column", "signal_priority", "conflict_score", "row_id"],
                ascending=[True, False, False, True],
            ).reset_index(drop=True)
        else:
            if "is_sampled" not in clustered_df.columns:
                clustered_df["is_sampled"] = 0
            sampled_df = clustered_df[clustered_df["is_sampled"] == 1].copy()

    print(f"  sampled rows = {len(sampled_df)}")

    print("[6/6] Exporting CSV/JSONL/summary ...")
    ensure_parent_dir(output_clustered_csv)
    ensure_parent_dir(output_sampled_csv)
    ensure_parent_dir(output_bucket_summary_csv)

    clustered_df.to_csv(output_clustered_csv, index=False, encoding="utf-8-sig")
    sampled_df.to_csv(output_sampled_csv, index=False, encoding="utf-8-sig")
    bucket_summary_df.to_csv(output_bucket_summary_csv, index=False, encoding="utf-8-sig")
    export_sampled_jsonl(sampled_df, detail_map, output_sampled_jsonl, cfg)

    if output_sample_summary_json:
        ensure_parent_dir(output_sample_summary_json)
        with open(output_sample_summary_json, "w", encoding="utf-8") as f:
            json.dump(build_sample_summary(clustered_df, sampled_df, cfg), f, ensure_ascii=False, indent=2, default=json_safe_value)

    print("\n===== DONE =====")
    print(f"Clustered CSV       : {output_clustered_csv}")
    print(f"Sampled CSV         : {output_sampled_csv}")
    print(f"Sampled JSONL       : {output_sampled_jsonl}")
    print(f"Bucket summary CSV  : {output_bucket_summary_csv}")
    if output_sample_summary_json:
        print(f"Sample summary JSON : {output_sample_summary_json}")

    print("\n===== STATS =====")
    print(f"Input candidates : {len(clustered_df)}")
    print(f"Sampled          : {len(sampled_df)}")
    if len(clustered_df) > 0:
        print(f"Sample ratio     : {len(sampled_df) / len(clustered_df):.6f}")
    if len(sampled_df) > 0:
        print("\nSampled by column:")
        print(sampled_df["column"].value_counts(dropna=False).to_string())
        print("\nSampled by sample_role:")
        print(sampled_df["sample_role"].value_counts(dropna=False).to_string())
        if "main_rule_type" in sampled_df.columns:
            print("\nSampled by main_rule_type:")
            print(sampled_df["main_rule_type"].value_counts(dropna=False).head(30).to_string())


def run_one_dataset(dataset: str, args):
    cfg = CONFIGS[dataset]

    input_summary_csv = args.input_summary_csv or cfg.input_summary_csv
    input_detail_jsonl = args.input_detail_jsonl or cfg.input_detail_jsonl
    output_clustered_csv = args.output_clustered_csv or cfg.output_clustered_csv
    output_sampled_csv = args.output_sampled_csv or cfg.output_sampled_csv
    output_sampled_jsonl = args.output_sampled_jsonl or cfg.output_sampled_jsonl
    output_bucket_summary_csv = args.output_bucket_summary_csv or cfg.output_bucket_summary_csv
    output_sample_summary_json = args.output_sample_summary_json if args.output_sample_summary_json is not None else cfg.output_sample_summary_json

    old_cwd = None
    if args.cwd:
        os.makedirs(args.cwd, exist_ok=True)
        old_cwd = os.getcwd()
        os.chdir(args.cwd)

    try:
        run_sampling(
            cfg,
            input_summary_csv,
            input_detail_jsonl,
            output_clustered_csv,
            output_sampled_csv,
            output_sampled_jsonl,
            output_bucket_summary_csv,
            output_sample_summary_json,
            args,
        )
    finally:
        if old_cwd is not None:
            os.chdir(old_cwd)


def parse_args():
    parser = argparse.ArgumentParser(description="Unified clustering and sampling for detection candidates.")
    parser.add_argument("--dataset", required=True, choices=sorted(list(CONFIGS.keys()) + ["all"]))

    parser.add_argument("--input_summary_csv", default=None)
    parser.add_argument("--input_detail_jsonl", default=None)
    parser.add_argument("--output_clustered_csv", default=None)
    parser.add_argument("--output_sampled_csv", default=None)
    parser.add_argument("--output_sampled_jsonl", default=None)
    parser.add_argument("--output_bucket_summary_csv", default=None)
    parser.add_argument("--output_sample_summary_json", default=None)
    parser.add_argument("--cwd", default=None)

    # 消融相关开关
    parser.add_argument("--disable-bucket-fields", nargs="*", default=[], help="Disable specific bucket fields, e.g. neighbor_bucket pattern_bucket.")
    parser.add_argument("--disable-feature-columns", nargs="*", default=[], help="Disable specific clustering feature columns.")
    parser.add_argument("--disable-quota-sampling", action="store_true", help="Disable dataset quota supplements, mainly for adult/soccer ablation.")
    parser.add_argument("--disable-budget-caps", action="store_true", help="Disable column/bucket/global budget caps.")
    parser.add_argument("--max-total-samples", type=int, default=None, help="Override global/adult max total samples.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dataset == "all":
        if any([
            args.input_summary_csv,
            args.input_detail_jsonl,
            args.output_clustered_csv,
            args.output_sampled_csv,
            args.output_sampled_jsonl,
            args.output_bucket_summary_csv,
            args.output_sample_summary_json,
        ]):
            raise ValueError("--dataset all cannot be used with input/output override arguments.")
        for ds in ["hospital", "flights", "beers", "rayyan", "adult", "soccer"]:
            print("\n" + "=" * 80)
            print(f"[DATASET] {ds}")
            print("=" * 80)
            run_one_dataset(ds, args)
    else:
        run_one_dataset(args.dataset, args)


if __name__ == "__main__":
    main()

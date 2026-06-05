#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unified_label_propagation_for_candidates.py

统一版标签传播脚本，覆盖：
hospital / flights / beers / rayyan / adult / soccer

设计目标：
1. 抽象统一逻辑，方便后续做消融实验。
2. 保持每个数据集原来的默认输入/输出文件名：
   - candidate_clustered_*.csv
   - candidate_sampled_labeled_*.csv
   - propagated_labels_*.jsonl
   - propagated_labels_*.csv
   - adult/soccer 额外 propagation_summary_*.json
3. 保持核心输出字段：
   row_id, column, value, bucket_id, cluster_id, sample_role, is_sampled, dist_to_center,
   label, label_binary, propagation_confidence, label_source, sample_weight,
   semantic_type, main_rule_type, main_usage_role, pattern_bucket, rarity_bucket, neighbor_bucket,
   feature fields, error/reason fields.
4. 将前四个数据集的通用传播逻辑合并，将 adult/soccer 的专属门控作为配置和策略函数实现。
5. 支持消融开关：
   --disable-cluster-propagation
   --disable-knn-propagation
   --disable-center-filter
   --disable-dataset-gate
   --disable-total-cap
   --knn-k / --cluster-threshold / --knn-threshold / --max-total-propagated

运行示例：
python unified_label_propagation_for_candidates.py --dataset soccer
python unified_label_propagation_for_candidates.py --dataset adult --cwd "/mnt/.../Error Detection Adult"
python unified_label_propagation_for_candidates.py --dataset all

说明：
- 该脚本不读取 ground truth，只使用 clustered candidates 和 LLM labeled CSV。
- 若需要 bit-level 完全一致，请先用旧脚本输出做 diff，再微调排序或字段顺序。
"""

import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. 通用工具
# ============================================================

def normalize_label(x: Any) -> Optional[int]:
    if pd.isna(x) or x is None:
        return None
    s = str(x).strip().lower()
    if s == "error":
        return 1
    if s == "correct":
        return 0
    return None


def parse_is_error(x: Any) -> Optional[int]:
    if pd.isna(x) or x is None:
        return None
    s = str(x).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return 1
    if s in {"false", "0", "no", "n"}:
        return 0
    return None


def normalize_label_binary_from_row(row: pd.Series) -> Optional[int]:
    if "label_binary" in row.index and pd.notna(row["label_binary"]):
        try:
            v = int(float(row["label_binary"]))
            if v in {0, 1}:
                return v
        except Exception:
            pass

    if "label" in row.index and pd.notna(row["label"]):
        y = normalize_label(row["label"])
        if y is not None:
            return y

    if "is_error" in row.index and pd.notna(row["is_error"]):
        y = parse_is_error(row["is_error"])
        if y is not None:
            return y

    return None


def label_to_name(y: int) -> str:
    return "error" if int(y) == 1 else "correct"


def build_key(row_id: Any, column: Any) -> Tuple[int, str]:
    return int(float(row_id)), str(column)


def compute_weight(base_weight: float, conf: float) -> float:
    conf = min(max(float(conf), 0.0), 1.0)
    return round(base_weight * conf, 6)


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


def as_int01(x: Any) -> int:
    if x is None:
        return 0
    try:
        if pd.isna(x):
            return 0
    except Exception:
        pass
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y"}:
        return 1
    if s in {"0", "false", "no", "n"}:
        return 0
    try:
        return int(float(x))
    except Exception:
        return 0


def make_json_safe(x: Any):
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


def safe_none_fill(df: pd.DataFrame, idx: int):
    for col in [
        "error_type",
        "reason_short",
        "reason_detailed",
        "suggested_correct_value",
        "needs_human_review",
        "evidence_used",
    ]:
        if col not in df.columns:
            df[col] = None
        if pd.isna(df.at[idx, col]) or df.at[idx, col] is None:
            df.at[idx, col] = None


# ============================================================
# 2. 配置
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
    "dist_to_center",
]

BASE_CATEGORICAL_FEATURES = [
    "column",
    "semantic_type",
    "main_rule_type",
    "main_usage_role",
    "pattern_bucket",
    "rarity_bucket",
    "neighbor_bucket",
    "sample_role",
]

BASE_OUTPUT_FIELDS = [
    "row_id",
    "column",
    "value",
    "bucket_id",
    "cluster_id",
    "sample_role",
    "is_sampled",
    "dist_to_center",
    "label",
    "label_binary",
    "propagation_confidence",
    "label_source",
    "sample_weight",
    "semantic_type",
    "detected_type",
    "main_rule_type",
    "main_usage_role",
    "pattern_bucket",
    "rarity_bucket",
    "neighbor_bucket",
    "violation_count",
    "conflict_score",
    "value_frequency",
    "value_frequency_rank",
    "neighbor_majority_value",
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
    "error_type",
    "reason_short",
    "reason_detailed",
    "suggested_correct_value",
    "needs_human_review",
    "evidence_used",
]


@dataclass
class PropagationConfig:
    name: str
    input_clustered_csv: str
    input_labeled_csv: str
    output_jsonl: str
    output_csv: str
    output_summary_json: Optional[str] = None

    min_labeled_per_cluster: int = 3
    cluster_propagation_threshold: float = 0.85
    enable_cluster_center_filter: bool = True
    cluster_center_dist_quantile: float = 0.70
    max_cluster_propagation_per_cluster: int = 50

    enable_knn_propagation: bool = True
    knn_k: int = 5
    knn_propagation_threshold: float = 0.80
    min_teacher_samples_for_knn: int = 20
    knn_within_same_column_only: bool = True
    knn_min_margin: float = 0.10
    knn_distance_eps: float = 1e-8
    knn_max_avg_distance: float = 4.5

    weight_llm: float = 1.0
    weight_cluster_prop: float = 0.70
    weight_knn_prop: float = 0.50
    min_propagation_confidence: float = 0.65

    # adult/soccer 专属约束
    dataset_gate: str = "generic"  # generic / adult / soccer
    weak_signal_extra_confidence: float = 0.0
    max_cluster_propagation_per_column: Optional[int] = None
    max_knn_propagation_per_column: Optional[int] = None
    max_total_propagated: Optional[int] = None
    random_state: int = 42

    primary_target_columns: Set[str] = field(default_factory=set)
    context_only_columns: Set[str] = field(default_factory=set)
    long_tail_entity_columns: Set[str] = field(default_factory=set)

    cluster_column_budget: Dict[str, int] = field(default_factory=dict)
    knn_column_budget: Dict[str, int] = field(default_factory=dict)

    numeric_features: List[str] = field(default_factory=lambda: list(BASE_NUMERIC_FEATURES))
    categorical_features: List[str] = field(default_factory=lambda: list(BASE_CATEGORICAL_FEATURES))
    extra_defaults: Dict[str, Any] = field(default_factory=dict)
    extra_output_fields: List[str] = field(default_factory=list)


def build_configs() -> Dict[str, PropagationConfig]:
    common_defaults = {
        "semantic_type": "unknown",
        "detected_type": "unknown",
        "violation_count": 0.0,
        "conflict_score": 0.0,
        "value_frequency": 0.0,
        "value_frequency_rank": 0.0,
        "neighbor_majority_value": None,
        "neighbor_majority_ratio": 0.0,
        "prior_error_probability": 0.0,
        "posterior_error_probability": 0.0,
        "main_rule_type": "unknown",
        "main_usage_role": "unknown",
        "strong_rule_count": 0.0,
        "candidate_generation_rule_count": 0.0,
        "fd_like_count": 0.0,
        "context_rule_count": 0.0,
        "global_rule_count": 0.0,
        "rare_value_count": 0.0,
        "typo_rule_count": 0.0,
        "pattern_rule_count": 0.0,
        "schema_rule_count": 0.0,
        "pattern_bucket": "unknown",
        "rarity_bucket": "unknown",
        "neighbor_bucket": "unknown",
        "bucket_id": "unknown",
        "cluster_id": -1,
        "dist_to_center": 0.0,
        "sample_role": "",
        "is_sampled": 0,
    }

    cfgs: Dict[str, PropagationConfig] = {}

    cfgs["hospital"] = PropagationConfig(
        name="hospital",
        input_clustered_csv="candidate_clustered_v2.csv",
        input_labeled_csv="candidate_sampled_labeled_v2.csv",
        output_jsonl="propagated_labels_v2.jsonl",
        output_csv="propagated_labels_v2.csv",
        extra_defaults=dict(common_defaults),
    )

    cfgs["flights"] = PropagationConfig(
        name="flights",
        input_clustered_csv="candidate_clustered_flights.csv",
        input_labeled_csv="candidate_sampled_labeled_flights.csv",
        output_jsonl="propagated_labels_flights.jsonl",
        output_csv="propagated_labels_flights.csv",
        numeric_features=BASE_NUMERIC_FEATURES + ["time_window_rule_count", "time_value_minutes"],
        categorical_features=BASE_CATEGORICAL_FEATURES + ["time_window_bucket", "time_value_bucket"],
        extra_defaults={
            **common_defaults,
            "time_window_rule_count": 0.0,
            "time_value_minutes": 0.0,
            "time_window_bucket": "unknown",
            "time_value_bucket": "unknown",
        },
        extra_output_fields=["time_window_rule_count", "time_value_minutes", "time_window_bucket", "time_value_bucket"],
    )

    cfgs["beers"] = PropagationConfig(
        name="beers",
        input_clustered_csv="candidate_clustered_beers.csv",
        input_labeled_csv="candidate_sampled_labeled_beers.csv",
        output_jsonl="propagated_labels_beers.jsonl",
        output_csv="propagated_labels_beers.csv",
        numeric_features=BASE_NUMERIC_FEATURES + ["numeric_window_rule_count", "numeric_value"],
        categorical_features=BASE_CATEGORICAL_FEATURES + ["numeric_window_bucket", "numeric_value_bucket"],
        extra_defaults={
            **common_defaults,
            "numeric_window_rule_count": 0.0,
            "numeric_value": 0.0,
            "numeric_window_bucket": "unknown",
            "numeric_value_bucket": "unknown",
        },
        extra_output_fields=["numeric_window_rule_count", "numeric_value", "numeric_window_bucket", "numeric_value_bucket"],
    )

    cfgs["rayyan"] = PropagationConfig(
        name="rayyan",
        input_clustered_csv="candidate_clustered_rayyan.csv",
        input_labeled_csv="candidate_sampled_labeled_rayyan.csv",
        output_jsonl="propagated_labels_rayyan.jsonl",
        output_csv="propagated_labels_rayyan.csv",
        numeric_features=BASE_NUMERIC_FEATURES + ["canonical_rule_count", "numeric_value"],
        categorical_features=BASE_CATEGORICAL_FEATURES + ["canonical_bucket", "date_value_bucket", "language_bucket"],
        extra_defaults={
            **common_defaults,
            "numeric_value": 0.0,
            "date_value": None,
            "language_canonical": None,
            "issn_canonical": None,
            "canonical_rule_count": 0.0,
            "canonical_bucket": "unknown",
            "date_value_bucket": "unknown",
            "language_bucket": "unknown",
        },
        extra_output_fields=[
            "canonical_rule_count",
            "numeric_value",
            "date_value",
            "language_canonical",
            "issn_canonical",
            "canonical_bucket",
            "date_value_bucket",
            "language_bucket",
        ],
    )

    adult_primary = {"sex", "relationship", "education"}
    adult_context = {
        "age",
        "workclass",
        "maritalstatus",
        "occupation",
        "race",
        "hoursperweek",
        "country",
        "income",
    }
    adult_cluster_budget = {
        "relationship": 120,
        "sex": 100,
        "education": 40,
        "age": 0,
        "workclass": 0,
        "maritalstatus": 0,
        "occupation": 0,
        "race": 0,
        "hoursperweek": 0,
        "country": 0,
        "income": 0,
    }
    adult_knn_budget = {
        "relationship": 100,
        "sex": 80,
        "education": 30,
        "age": 0,
        "workclass": 0,
        "maritalstatus": 0,
        "occupation": 0,
        "race": 0,
        "hoursperweek": 0,
        "country": 0,
        "income": 0,
    }
    adult_extra = {
        "adult_domain_rule_count": 0.0,
        "adult_format_rule_count": 0.0,
        "adult_consistency_rule_count": 0.0,
        "age_lower": 0.0,
        "age_upper": 0.0,
        "hours_lower": 0.0,
        "hours_upper": 0.0,
        "education_order": 0.0,
        "adult_profile_inconsistency_count": 0.0,
        "adult_consistency_score": 0.0,
        "time_window_rule_count": 0.0,
        "time_value_minutes": 0.0,
        "signal_priority": 0.0,
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
        "domain_bucket": "unknown",
        "adult_consistency_bucket": "unknown",
        "age_sampling_bucket": "unknown",
        "hours_sampling_bucket": "unknown",
        "adult_value_bucket": "unknown",
        "time_window_bucket": "unknown",
        "time_value_bucket": "unknown",
        "adult_v2_attribution_bucket": "unknown",
    }
    cfgs["adult"] = PropagationConfig(
        name="adult",
        input_clustered_csv="candidate_clustered_adult.csv",
        input_labeled_csv="candidate_sampled_labeled_adult.csv",
        output_jsonl="propagated_labels_adult.jsonl",
        output_csv="propagated_labels_adult.csv",
        output_summary_json="propagation_summary_adult.json",
        cluster_propagation_threshold=0.90,
        cluster_center_dist_quantile=0.55,
        max_cluster_propagation_per_cluster=20,
        knn_propagation_threshold=0.88,
        min_teacher_samples_for_knn=25,
        knn_min_margin=0.18,
        knn_max_avg_distance=3.8,
        min_propagation_confidence=0.78,
        weak_signal_extra_confidence=0.08,
        max_cluster_propagation_per_column=80,
        max_knn_propagation_per_column=80,
        max_total_propagated=800,
        dataset_gate="adult",
        primary_target_columns=adult_primary,
        context_only_columns=adult_context,
        cluster_column_budget=adult_cluster_budget,
        knn_column_budget=adult_knn_budget,
        numeric_features=BASE_NUMERIC_FEATURES + [
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
            "time_window_rule_count",
            "time_value_minutes",
            "signal_priority",
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
        categorical_features=BASE_CATEGORICAL_FEATURES + [
            "detected_type",
            "domain_bucket",
            "adult_consistency_bucket",
            "age_sampling_bucket",
            "hours_sampling_bucket",
            "adult_value_bucket",
            "time_window_bucket",
            "time_value_bucket",
            "adult_v2_attribution_bucket",
        ],
        extra_defaults={**common_defaults, **adult_extra},
        extra_output_fields=list(adult_extra.keys()) + ["propagation_priority"],
    )

    soccer_cols = {
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
    }
    soccer_long_tail = {"name", "surname", "birthplace", "team", "city", "stadium", "manager"}
    soccer_cluster_budget = {
        "birthyear": 120,
        "season": 100,
        "position": 100,
        "birthplace": 80,
        "name": 60,
        "surname": 60,
        "team": 50,
        "city": 50,
        "stadium": 50,
        "manager": 50,
    }
    soccer_knn_budget = {
        "birthyear": 100,
        "season": 80,
        "position": 80,
        "birthplace": 60,
        "name": 45,
        "surname": 45,
        "team": 40,
        "city": 40,
        "stadium": 40,
        "manager": 40,
    }
    soccer_extra = {
        "evidence_only_fd_count": 0.0,
        "soccer_domain_rule_count": 0.0,
        "soccer_format_rule_count": 0.0,
        "soccer_numeric_rule_count": 0.0,
        "soccer_consistency_rule_count": 0.0,
        "soccer_profile_inconsistency_count": 0.0,
        "soccer_consistency_score": 0.0,
        "year_value": 0.0,
        "row_birthyear_int": 0.0,
        "row_season_int": 0.0,
        "row_age_at_season": 0.0,
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
        "adult_domain_rule_count": 0.0,
        "adult_format_rule_count": 0.0,
        "adult_consistency_rule_count": 0.0,
        "adult_profile_inconsistency_count": 0.0,
        "adult_consistency_score": 0.0,
        "age_lower": 0.0,
        "age_upper": 0.0,
        "hours_lower": 0.0,
        "hours_upper": 0.0,
        "education_order": 0.0,
        "time_window_rule_count": 0.0,
        "time_value_minutes": 0.0,
        "signal_priority": 0.0,
        "domain_bucket": "unknown",
        "soccer_consistency_bucket": "unknown",
        "soccer_attribution_bucket": "unknown",
        "soccer_value_bucket": "unknown",
        "canonical_position": None,
        "row_position_canonical": None,
        "adult_consistency_bucket": "unknown",
        "adult_value_bucket": "unknown",
        "time_window_bucket": "unknown",
        "time_value_bucket": "unknown",
        "adult_v2_attribution_bucket": "unknown",
    }
    cfgs["soccer"] = PropagationConfig(
        name="soccer",
        input_clustered_csv="candidate_clustered_soccer.csv",
        input_labeled_csv="candidate_sampled_labeled_soccer.csv",
        output_jsonl="propagated_labels_soccer.jsonl",
        output_csv="propagated_labels_soccer.csv",
        output_summary_json="propagation_summary_soccer.json",
        cluster_propagation_threshold=0.90,
        cluster_center_dist_quantile=0.55,
        max_cluster_propagation_per_cluster=20,
        knn_propagation_threshold=0.88,
        min_teacher_samples_for_knn=25,
        knn_min_margin=0.18,
        knn_max_avg_distance=3.8,
        min_propagation_confidence=0.78,
        weak_signal_extra_confidence=0.08,
        max_cluster_propagation_per_column=80,
        max_knn_propagation_per_column=80,
        max_total_propagated=800,
        dataset_gate="soccer",
        primary_target_columns=soccer_cols,
        context_only_columns=set(),
        long_tail_entity_columns=soccer_long_tail,
        cluster_column_budget=soccer_cluster_budget,
        knn_column_budget=soccer_knn_budget,
        numeric_features=BASE_NUMERIC_FEATURES + [
            "evidence_only_fd_count",
            "soccer_domain_rule_count",
            "soccer_format_rule_count",
            "soccer_numeric_rule_count",
            "soccer_consistency_rule_count",
            "soccer_profile_inconsistency_count",
            "soccer_consistency_score",
            "year_value",
            "row_birthyear_int",
            "row_season_int",
            "row_age_at_season",
            "target_is_primary_column",
            "target_is_context_only_column",
            "birthyear_value_invalid",
            "season_value_invalid",
            "position_value_invalid",
            "position_needs_canonicalization",
            "birthyear_season_age_conflict",
            "team_context_conflict",
            "format_rule_signal",
            "fd_like_signal",
            "adult_domain_rule_count",
            "adult_format_rule_count",
            "adult_consistency_rule_count",
            "adult_profile_inconsistency_count",
            "adult_consistency_score",
            "age_lower",
            "age_upper",
            "hours_lower",
            "hours_upper",
            "education_order",
            "time_window_rule_count",
            "time_value_minutes",
            "signal_priority",
        ],
        categorical_features=BASE_CATEGORICAL_FEATURES + [
            "detected_type",
            "domain_bucket",
            "soccer_consistency_bucket",
            "soccer_attribution_bucket",
            "soccer_value_bucket",
            "canonical_position",
            "row_position_canonical",
            "adult_consistency_bucket",
            "adult_value_bucket",
            "time_window_bucket",
            "time_value_bucket",
            "adult_v2_attribution_bucket",
        ],
        extra_defaults={**common_defaults, **soccer_extra},
        extra_output_fields=list(soccer_extra.keys()) + ["propagation_priority"],
    )

    return cfgs


CONFIGS = build_configs()


# ============================================================
# 3. 读取数据
# ============================================================

def load_clustered_csv(path: str, cfg: PropagationConfig) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if "row_id" not in df.columns or "column" not in df.columns:
        raise ValueError(f"{path} must contain row_id and column columns.")

    df["row_id"] = pd.to_numeric(df["row_id"], errors="coerce").fillna(-1).astype(int)
    df["column"] = df["column"].astype(str)

    for col, default_val in cfg.extra_defaults.items():
        if col not in df.columns:
            df[col] = default_val

    numeric_cols = sorted(set(cfg.numeric_features + ["cluster_id", "dist_to_center", "is_sampled"]))
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Soccer/Adult 兜底纠偏，防止上游统一脚本字段为空
    if cfg.name == "soccer":
        rule = df["main_rule_type"].fillna("").astype(str)
        df["target_is_primary_column"] = df["column"].isin(cfg.primary_target_columns).astype(int)
        df["target_is_context_only_column"] = 0
        df["birthyear_value_invalid"] = np.where(rule.eq("birthyear_numeric_range"), 1, df["birthyear_value_invalid"].fillna(0)).astype(int)
        df["season_value_invalid"] = np.where(rule.eq("season_numeric_range"), 1, df["season_value_invalid"].fillna(0)).astype(int)
        df["position_value_invalid"] = np.where(rule.eq("position_domain"), 1, df["position_value_invalid"].fillna(0)).astype(int)
        df["birthyear_season_age_conflict"] = np.where(rule.eq("birthyear_season_age_consistency"), 1, df["birthyear_season_age_conflict"].fillna(0)).astype(int)
        df["team_context_conflict"] = np.where(rule.isin({"team_context_dominant", "team_season_manager_consistency_hint", "city_stadium_consistency_hint", "dominant_value_by_context", "dominant_value_by_context_pair"}), 1, df["team_context_conflict"].fillna(0)).astype(int)
        df["format_rule_signal"] = np.where(rule.isin({"name_pattern", "surname_pattern", "manager_name_pattern", "team_name_pattern", "birthplace_location_pattern", "city_location_pattern", "stadium_name_pattern", "pattern", "generic_regex"}), 1, df["format_rule_signal"].fillna(0)).astype(int)
        df["fd_like_signal"] = np.where(rule.eq("functional_dependency"), 1, df["fd_like_signal"].fillna(0)).astype(int)
        soft_fd = rule.eq("soft_functional_dependency")
        df.loc[soft_fd, "fd_like_signal"] = 0
        df.loc[soft_fd, "evidence_only_fd_count"] = np.maximum(pd.to_numeric(df.loc[soft_fd, "evidence_only_fd_count"], errors="coerce").fillna(0), 1)

    if cfg.name == "adult":
        df["target_is_primary_column"] = df["column"].isin(cfg.primary_target_columns).astype(int)
        df["target_is_context_only_column"] = df["column"].isin(cfg.context_only_columns).astype(int)

    return df


def load_labeled_csv(path: str, cfg: PropagationConfig) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if "row_id" not in df.columns or "column" not in df.columns:
        raise ValueError(f"{path} must contain row_id and column columns.")

    df["row_id"] = pd.to_numeric(df["row_id"], errors="coerce").fillna(-1).astype(int)
    df["column"] = df["column"].astype(str)

    labels = []
    for _, row in df.iterrows():
        labels.append(normalize_label_binary_from_row(row))
    df["label_binary_norm"] = labels
    df = df[pd.notna(df["label_binary_norm"])].copy()
    df["label_binary_norm"] = df["label_binary_norm"].astype(int)

    if "confidence" not in df.columns:
        df["confidence"] = 1.0
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(1.0).clip(0.0, 1.0)

    # 同一个 cell 重复标注时，adult/soccer 旧脚本保留 confidence 更高者；这里统一采用该策略。
    df["_key"] = list(zip(df["row_id"].astype(int), df["column"].astype(str)))
    df = df.sort_values(by=["_key", "confidence"], ascending=[True, False])
    df = df.drop_duplicates(subset=["_key"], keep="first").drop(columns=["_key"]).copy()

    return df


# ============================================================
# 4. 合并 LLM 标签
# ============================================================

def merge_clustered_and_labeled(clustered_df: pd.DataFrame, labeled_df: pd.DataFrame, cfg: PropagationConfig) -> pd.DataFrame:
    df = clustered_df.copy()

    label_map: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for _, row in labeled_df.iterrows():
        key = build_key(row["row_id"], row["column"])
        label_map[key] = {
            "label_binary": int(row["label_binary_norm"]),
            "confidence": float(row["confidence"]) if pd.notna(row["confidence"]) else 1.0,
            "error_type": row["error_type"] if "error_type" in row else None,
            "reason_short": row["reason_short"] if "reason_short" in row else None,
            "reason_detailed": row["reason_detailed"] if "reason_detailed" in row else None,
            "suggested_correct_value": row["suggested_correct_value"] if "suggested_correct_value" in row else None,
            "needs_human_review": row["needs_human_review"] if "needs_human_review" in row else None,
            "evidence_used": row["evidence_used"] if "evidence_used" in row else None,
        }

    labels, confs, errtypes, reasons_short, reasons_detailed = [], [], [], [], []
    suggested_values, needs_review, evidence_useds, is_labeled = [], [], [], []

    for _, row in df.iterrows():
        key = build_key(row["row_id"], row["column"])
        if key in label_map:
            item = label_map[key]
            labels.append(item["label_binary"])
            confs.append(item["confidence"])
            errtypes.append(item["error_type"])
            reasons_short.append(item["reason_short"])
            reasons_detailed.append(item["reason_detailed"])
            suggested_values.append(item["suggested_correct_value"])
            needs_review.append(item["needs_human_review"])
            evidence_useds.append(item["evidence_used"])
            is_labeled.append(1)
        else:
            labels.append(np.nan)
            confs.append(np.nan)
            errtypes.append(None)
            reasons_short.append(None)
            reasons_detailed.append(None)
            suggested_values.append(None)
            needs_review.append(None)
            evidence_useds.append(None)
            is_labeled.append(0)

    df["label_binary"] = labels
    df["label_confidence"] = confs
    df["error_type"] = errtypes
    df["reason_short"] = reasons_short
    df["reason_detailed"] = reasons_detailed
    df["suggested_correct_value"] = suggested_values
    df["needs_human_review"] = needs_review
    df["evidence_used"] = evidence_useds
    df["is_labeled"] = is_labeled

    df["final_label_binary"] = df["label_binary"]
    df["final_label"] = df["label_binary"].map(lambda x: label_to_name(x) if pd.notna(x) else None)
    df["propagation_confidence"] = df["label_confidence"]
    df["label_source"] = df["is_labeled"].map(lambda x: "llm" if x == 1 else None)
    df["sample_weight"] = df["is_labeled"].map(lambda x: cfg.weight_llm if x == 1 else np.nan)
    df["propagation_priority"] = np.nan

    return df


# ============================================================
# 5. 特征矩阵
# ============================================================

def prepare_features(df: pd.DataFrame, cfg: PropagationConfig) -> Tuple[np.ndarray, List[str]]:
    work = df.copy()

    numeric_cols = []
    for col in cfg.numeric_features:
        if col not in work.columns:
            work[col] = 0.0
        work[col] = pd.to_numeric(work[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        numeric_cols.append(col)

    num_df = work[numeric_cols].copy()
    cat_frames = []
    feat_names = list(numeric_cols)

    for col in cfg.categorical_features:
        if col not in work.columns:
            work[col] = "unknown"
        dummies = pd.get_dummies(work[col].fillna("unknown").astype(str), prefix=col)
        cat_frames.append(dummies)
        feat_names.extend(list(dummies.columns))

    feat_df = pd.concat([num_df] + cat_frames, axis=1) if cat_frames else num_df

    if feat_df.shape[1] == 0:
        feat_df = pd.DataFrame({"constant": np.zeros(len(work))}, index=work.index)
        feat_names = ["constant"]

    scaler = StandardScaler()
    X = scaler.fit_transform(feat_df.values.astype(float))

    return X, feat_names


# ============================================================
# 6. Dataset-specific gate
# ============================================================

RELATIONSHIP_V2_RULE_TYPES = {
    "relationship_marital_spouse_conflict",
    "relationship_age_spouse_conflict",
    "relationship_sex_spouse_conflict",
    "relationship_context_dominant",
}

EDUCATION_V2_RULE_TYPES = {"education_age_extreme_conflict"}


def has_relationship_v2_signal(row: pd.Series) -> bool:
    rule = str(row.get("main_rule_type", ""))
    return (
        as_int01(row.get("relationship_marital_conflict", 0)) == 1
        or as_int01(row.get("relationship_age_conflict", 0)) == 1
        or as_int01(row.get("relationship_sex_conflict", 0)) == 1
        or as_int01(row.get("has_relationship_v2_rule", 0)) == 1
        or rule in RELATIONSHIP_V2_RULE_TYPES
    )


def has_sex_invalid_signal(row: pd.Series) -> bool:
    return as_int01(row.get("sex_value_invalid", 0)) == 1


def has_education_v2_signal(row: pd.Series) -> bool:
    rule = str(row.get("main_rule_type", ""))
    return (
        as_int01(row.get("education_age_extreme_conflict", 0)) == 1
        or as_int01(row.get("has_education_v2_rule", 0)) == 1
        or rule in EDUCATION_V2_RULE_TYPES
    )


def adult_is_weak_signal_row(row: pd.Series, cfg: PropagationConfig) -> bool:
    col = str(row.get("column", ""))
    main_rule = str(row.get("main_rule_type", ""))

    if col in cfg.context_only_columns:
        return True

    if col in {"country", "occupation", "workclass"} and main_rule in {"rare_value", "rare_pattern", "global_dominant_value"}:
        return True

    strong_sum = (
        safe_float(row.get("strong_rule_count", 0))
        + safe_float(row.get("adult_domain_rule_count", 0))
        + safe_float(row.get("adult_format_rule_count", 0))
        + safe_float(row.get("adult_consistency_rule_count", 0))
        + safe_float(row.get("schema_rule_count", 0))
        + safe_float(row.get("fd_like_count", 0))
        + safe_float(row.get("context_rule_count", 0))
        + (1 if has_relationship_v2_signal(row) else 0)
        + (1 if has_sex_invalid_signal(row) else 0)
        + (1 if has_education_v2_signal(row) else 0)
    )
    return strong_sum <= 0


def soccer_has_missing_signal(row: pd.Series) -> bool:
    rule = str(row.get("main_rule_type", ""))
    return rule in {"not_null", "high_risk_missing"} or "missing" in str(row.get("soccer_attribution_bucket", "")).lower()


def soccer_has_format_signal(row: pd.Series) -> bool:
    return (
        as_int01(row.get("format_rule_signal", 0)) == 1
        or safe_float(row.get("soccer_format_rule_count", 0)) > 0
        or safe_float(row.get("adult_format_rule_count", 0)) > 0
    )


def soccer_has_birthyear_invalid(row: pd.Series) -> bool:
    return as_int01(row.get("birthyear_value_invalid", 0)) == 1


def soccer_has_season_invalid(row: pd.Series) -> bool:
    return as_int01(row.get("season_value_invalid", 0)) == 1


def soccer_has_position_invalid(row: pd.Series) -> bool:
    return as_int01(row.get("position_value_invalid", 0)) == 1 or as_int01(row.get("position_needs_canonicalization", 0)) == 1


def soccer_has_age_consistency(row: pd.Series) -> bool:
    return as_int01(row.get("birthyear_season_age_conflict", 0)) == 1 or str(row.get("main_rule_type", "")) == "birthyear_season_age_consistency"


def soccer_has_team_context(row: pd.Series) -> bool:
    rule = str(row.get("main_rule_type", ""))
    return as_int01(row.get("team_context_conflict", 0)) == 1 or rule in {
        "team_context_dominant",
        "team_season_manager_consistency_hint",
        "city_stadium_consistency_hint",
        "dominant_value_by_context",
        "dominant_value_by_context_pair",
    }


def soccer_has_fd_like(row: pd.Series) -> bool:
    rule = str(row.get("main_rule_type", ""))
    return as_int01(row.get("fd_like_signal", 0)) == 1 or safe_float(row.get("fd_like_count", 0)) > 0 or rule == "functional_dependency"


def soccer_has_evidence_only_fd(row: pd.Series) -> bool:
    rule = str(row.get("main_rule_type", ""))
    return safe_float(row.get("evidence_only_fd_count", 0)) > 0 or rule == "soft_functional_dependency"


def soccer_has_direct_schema_signal(row: pd.Series) -> bool:
    return (
        soccer_has_missing_signal(row)
        or soccer_has_format_signal(row)
        or soccer_has_birthyear_invalid(row)
        or soccer_has_season_invalid(row)
        or soccer_has_position_invalid(row)
        or soccer_has_age_consistency(row)
        or safe_float(row.get("soccer_domain_rule_count", 0)) > 0
        or safe_float(row.get("soccer_numeric_rule_count", 0)) > 0
        or safe_float(row.get("schema_rule_count", 0)) > 0
    )


def soccer_is_weak_signal_row(row: pd.Series, cfg: PropagationConfig) -> bool:
    if str(row.get("column", "")) in cfg.context_only_columns:
        return True

    strong_sum = (
        safe_float(row.get("strong_rule_count", 0))
        + safe_float(row.get("soccer_domain_rule_count", 0))
        + safe_float(row.get("soccer_format_rule_count", 0))
        + safe_float(row.get("soccer_numeric_rule_count", 0))
        + safe_float(row.get("soccer_consistency_rule_count", 0))
        + safe_float(row.get("schema_rule_count", 0))
        + (1 if soccer_has_birthyear_invalid(row) else 0)
        + (1 if soccer_has_season_invalid(row) else 0)
        + (1 if soccer_has_position_invalid(row) else 0)
        + (1 if soccer_has_age_consistency(row) else 0)
        + (1 if soccer_has_format_signal(row) else 0)
        + (1 if soccer_has_missing_signal(row) else 0)
    )
    return strong_sum <= 0


def is_weak_signal_row(row: pd.Series, cfg: PropagationConfig) -> bool:
    if cfg.dataset_gate == "adult":
        return adult_is_weak_signal_row(row, cfg)
    if cfg.dataset_gate == "soccer":
        return soccer_is_weak_signal_row(row, cfg)
    return False


def required_confidence_for_row(row: pd.Series, cfg: PropagationConfig, base_threshold: float) -> float:
    if is_weak_signal_row(row, cfg):
        return min(0.98, base_threshold + cfg.weak_signal_extra_confidence)
    return base_threshold


def should_allow_propagation_for_row(
    row: pd.Series,
    cfg: PropagationConfig,
    propagated_conf: float,
    source: str,
    disable_dataset_gate: bool = False,
) -> bool:
    if disable_dataset_gate or cfg.dataset_gate == "generic":
        return propagated_conf >= cfg.min_propagation_confidence

    col = str(row.get("column", ""))

    if cfg.dataset_gate == "adult":
        if col in cfg.context_only_columns:
            return False
        if col not in cfg.primary_target_columns and col not in cfg.context_only_columns:
            return False

        if col == "relationship" and has_relationship_v2_signal(row):
            return propagated_conf >= 0.88
        if col == "sex" and has_sex_invalid_signal(row):
            return propagated_conf >= 0.90
        if col == "education" and has_education_v2_signal(row):
            return propagated_conf >= 0.90

        return propagated_conf >= required_confidence_for_row(row, cfg, cfg.min_propagation_confidence)

    if cfg.dataset_gate == "soccer":
        if col not in cfg.primary_target_columns:
            return False
        if col in cfg.context_only_columns:
            return False

        if soccer_has_missing_signal(row):
            return propagated_conf >= 0.88
        if soccer_has_format_signal(row):
            return propagated_conf >= 0.88
        if col == "birthyear" and (soccer_has_birthyear_invalid(row) or soccer_has_age_consistency(row)):
            return propagated_conf >= 0.88
        if col == "season" and soccer_has_season_invalid(row):
            return propagated_conf >= 0.90
        if col == "position" and soccer_has_position_invalid(row):
            return propagated_conf >= 0.90
        if soccer_has_age_consistency(row):
            return propagated_conf >= 0.88

        direct_schema = soccer_has_direct_schema_signal(row)
        if col in cfg.long_tail_entity_columns and not direct_schema:
            if soccer_has_team_context(row) or soccer_has_fd_like(row) or soccer_has_evidence_only_fd(row):
                return propagated_conf >= max(0.90, 0.92)
            return propagated_conf >= 0.90

        if soccer_has_team_context(row):
            return propagated_conf >= 0.92
        if soccer_has_fd_like(row):
            return propagated_conf >= 0.92
        if soccer_has_evidence_only_fd(row):
            return propagated_conf >= 0.95

        return propagated_conf >= required_confidence_for_row(row, cfg, cfg.min_propagation_confidence)

    return propagated_conf >= cfg.min_propagation_confidence


def propagation_priority(row: pd.Series, cfg: PropagationConfig) -> float:
    score = 0.0
    score += safe_float(row.get("propagation_confidence", 0.0)) * 5.0
    score += safe_float(row.get("strong_rule_count", 0.0)) * 2.5
    score += safe_float(row.get("schema_rule_count", 0.0)) * 1.2
    score += safe_float(row.get("fd_like_count", 0.0)) * 0.8
    score += safe_float(row.get("conflict_score", 0.0)) * 0.03
    score += safe_float(row.get("posterior_error_probability", 0.0)) * 1.0

    if cfg.dataset_gate == "adult":
        score += safe_float(row.get("adult_domain_rule_count", 0.0)) * 2.2
        score += safe_float(row.get("adult_format_rule_count", 0.0)) * 2.0
        score += safe_float(row.get("adult_consistency_rule_count", 0.0)) * 1.8
        if has_relationship_v2_signal(row):
            score += 3.0
        if has_sex_invalid_signal(row):
            score += 2.5
        if has_education_v2_signal(row):
            score += 2.0

    elif cfg.dataset_gate == "soccer":
        score += safe_float(row.get("soccer_domain_rule_count", 0.0)) * 2.2
        score += safe_float(row.get("soccer_format_rule_count", 0.0)) * 2.0
        score += safe_float(row.get("soccer_numeric_rule_count", 0.0)) * 2.0
        score += safe_float(row.get("soccer_consistency_rule_count", 0.0)) * 1.5
        if soccer_has_missing_signal(row):
            score += 2.0
        if soccer_has_format_signal(row):
            score += 2.0
        if soccer_has_birthyear_invalid(row) or soccer_has_season_invalid(row) or soccer_has_position_invalid(row):
            score += 2.5
        if soccer_has_age_consistency(row):
            score += 2.0
        if soccer_has_evidence_only_fd(row):
            score -= 1.5

    if is_weak_signal_row(row, cfg):
        score -= 2.0

    return float(score)


def get_column_budget(cfg: PropagationConfig, col: str, mode: str) -> int:
    if mode == "cluster":
        if cfg.cluster_column_budget:
            return int(cfg.cluster_column_budget.get(col, cfg.max_cluster_propagation_per_column or 10**9))
        return int(cfg.max_cluster_propagation_per_column or 10**9)
    if mode == "knn":
        if cfg.knn_column_budget:
            return int(cfg.knn_column_budget.get(col, cfg.max_knn_propagation_per_column or 10**9))
        return int(cfg.max_knn_propagation_per_column or 10**9)
    return 10**9


# ============================================================
# 7. 第一层传播：Cluster
# ============================================================

def cluster_label_propagation(
    df: pd.DataFrame,
    cfg: PropagationConfig,
    disable_center_filter: bool = False,
    disable_dataset_gate: bool = False,
) -> pd.DataFrame:
    df = df.copy()
    column_added_counter: Counter = Counter()

    for (bucket_id, cluster_id), grp in df.groupby(["bucket_id", "cluster_id"], sort=False, dropna=False):
        labeled_grp = grp[pd.notna(grp["final_label_binary"])]

        if len(labeled_grp) < cfg.min_labeled_per_cluster:
            continue

        counts = labeled_grp["final_label_binary"].value_counts(dropna=True).to_dict()
        error_cnt = counts.get(1.0, 0) + counts.get(1, 0)
        correct_cnt = counts.get(0.0, 0) + counts.get(0, 0)
        total = error_cnt + correct_cnt
        if total == 0:
            continue

        error_ratio = error_cnt / total
        correct_ratio = correct_cnt / total

        propagated_label = None
        propagated_conf = None
        if error_ratio >= cfg.cluster_propagation_threshold:
            propagated_label = 1
            propagated_conf = error_ratio
        elif correct_ratio >= cfg.cluster_propagation_threshold:
            propagated_label = 0
            propagated_conf = correct_ratio

        if propagated_label is None:
            continue

        teacher_support_denom = 8.0 if cfg.dataset_gate in {"adult", "soccer"} else 10.0
        teacher_support_factor = min(1.0, total / teacher_support_denom)
        propagated_conf = float(propagated_conf) * teacher_support_factor

        if propagated_conf < cfg.min_propagation_confidence:
            continue

        unlabeled_grp = grp[pd.isna(grp["final_label_binary"])].copy()
        if len(unlabeled_grp) == 0:
            continue

        if cfg.enable_cluster_center_filter and (not disable_center_filter):
            unlabeled_grp["dist_to_center"] = pd.to_numeric(unlabeled_grp["dist_to_center"], errors="coerce").fillna(np.inf)
            dist_threshold = unlabeled_grp["dist_to_center"].quantile(cfg.cluster_center_dist_quantile)
            unlabeled_grp = unlabeled_grp[unlabeled_grp["dist_to_center"] <= dist_threshold].copy()

        if len(unlabeled_grp) == 0:
            continue

        if cfg.dataset_gate in {"adult", "soccer"}:
            allowed = unlabeled_grp.apply(
                lambda r: should_allow_propagation_for_row(r, cfg, propagated_conf, "cluster_propagation", disable_dataset_gate),
                axis=1,
            )
            unlabeled_grp = unlabeled_grp[allowed].copy()
            if len(unlabeled_grp) == 0:
                continue

            unlabeled_grp["_prop_priority"] = unlabeled_grp.apply(lambda r: propagation_priority(r, cfg), axis=1)
            unlabeled_grp = unlabeled_grp.sort_values(
                by=["_prop_priority", "dist_to_center", "conflict_score", "posterior_error_probability"],
                ascending=[False, True, False, False],
            )
        else:
            unlabeled_grp = unlabeled_grp.sort_values(by="dist_to_center", ascending=True, na_position="last")

        updates = []
        for idx, r in unlabeled_grp.iterrows():
            col = str(r.get("column", ""))
            budget = get_column_budget(cfg, col, "cluster")
            if column_added_counter[col] >= budget:
                continue
            updates.append(idx)
            column_added_counter[col] += 1
            if len(updates) >= cfg.max_cluster_propagation_per_cluster:
                break

        for idx in updates:
            df.at[idx, "final_label_binary"] = propagated_label
            df.at[idx, "final_label"] = label_to_name(propagated_label)
            df.at[idx, "propagation_confidence"] = round(float(propagated_conf), 6)
            df.at[idx, "label_source"] = "cluster_propagation"
            df.at[idx, "sample_weight"] = compute_weight(cfg.weight_cluster_prop, propagated_conf)
            df.at[idx, "propagation_priority"] = propagation_priority(df.loc[idx], cfg)
            safe_none_fill(df, idx)

    return df


# ============================================================
# 8. 第二层传播：KNN
# ============================================================

def _weighted_neighbor_vote(
    neighbor_labels: np.ndarray,
    neighbor_distances: np.ndarray,
    cfg: PropagationConfig,
) -> Tuple[Optional[int], Optional[float]]:
    weights = 1.0 / (neighbor_distances + cfg.knn_distance_eps)

    error_score = float(np.sum(weights[neighbor_labels == 1]))
    correct_score = float(np.sum(weights[neighbor_labels == 0]))
    total_score = error_score + correct_score

    if total_score <= 0:
        return None, None

    error_ratio = error_score / total_score
    correct_ratio = correct_score / total_score

    propagated_label = None
    propagated_conf = None

    if error_ratio >= cfg.knn_propagation_threshold and (error_ratio - correct_ratio) >= cfg.knn_min_margin:
        propagated_label = 1
        propagated_conf = error_ratio
    elif correct_ratio >= cfg.knn_propagation_threshold and (correct_ratio - error_ratio) >= cfg.knn_min_margin:
        propagated_label = 0
        propagated_conf = correct_ratio

    return propagated_label, propagated_conf


def _apply_knn_updates(
    df: pd.DataFrame,
    candidate_updates: List[Tuple[int, int, float, float, float]],
    cfg: PropagationConfig,
    column_added_counter: Counter,
) -> pd.DataFrame:
    # idx, label, conf, avg_distance, priority
    candidate_updates.sort(key=lambda x: x[4], reverse=True)
    for idx, propagated_label, propagated_conf, avg_distance, priority in candidate_updates:
        col = str(df.at[idx, "column"])
        budget = get_column_budget(cfg, col, "knn")
        if column_added_counter[col] >= budget:
            continue

        df.at[idx, "final_label_binary"] = propagated_label
        df.at[idx, "final_label"] = label_to_name(propagated_label)
        df.at[idx, "propagation_confidence"] = round(float(propagated_conf), 6)
        df.at[idx, "label_source"] = "knn_propagation"
        df.at[idx, "sample_weight"] = compute_weight(cfg.weight_knn_prop, propagated_conf)
        df.at[idx, "propagation_priority"] = priority
        safe_none_fill(df, idx)
        column_added_counter[col] += 1
    return df


def knn_label_propagation(
    df: pd.DataFrame,
    X: np.ndarray,
    cfg: PropagationConfig,
    disable_dataset_gate: bool = False,
) -> pd.DataFrame:
    df = df.copy()

    teacher_mask = df["label_source"] == "llm"
    teacher_indices = df[teacher_mask].index.tolist()

    if len(teacher_indices) < cfg.min_teacher_samples_for_knn:
        return df

    column_added_counter: Counter = Counter()

    if cfg.knn_within_same_column_only:
        all_columns = df["column"].dropna().astype(str).unique().tolist()

        for col_name in all_columns:
            col_teacher_indices = df[(df["label_source"] == "llm") & (df["column"] == col_name)].index.tolist()
            col_unlabeled_indices = df[(df["final_label_binary"].isna()) & (df["column"] == col_name)].index.tolist()

            min_col_teachers = max(3, min(cfg.knn_k, cfg.min_teacher_samples_for_knn // 4))
            if len(col_teacher_indices) < min_col_teachers:
                continue
            if len(col_unlabeled_indices) == 0:
                continue

            X_teacher = X[col_teacher_indices]
            y_teacher = df.loc[col_teacher_indices, "final_label_binary"].astype(int).values

            nn_model = NearestNeighbors(n_neighbors=min(cfg.knn_k, len(col_teacher_indices)), metric="euclidean")
            nn_model.fit(X_teacher)

            candidate_updates = []
            for idx in col_unlabeled_indices:
                x = X[idx].reshape(1, -1)
                neighbor_distances, neighbor_pos = nn_model.kneighbors(x)
                neighbor_distances = neighbor_distances[0]
                neighbor_pos = neighbor_pos[0]
                neighbor_labels = y_teacher[neighbor_pos]

                avg_distance = float(np.mean(neighbor_distances))
                if avg_distance > cfg.knn_max_avg_distance:
                    continue

                propagated_label, propagated_conf = _weighted_neighbor_vote(neighbor_labels, neighbor_distances, cfg)
                if propagated_label is None or propagated_conf is None:
                    continue

                if cfg.dataset_gate == "generic":
                    if propagated_conf < cfg.min_propagation_confidence:
                        continue
                else:
                    if not should_allow_propagation_for_row(df.loc[idx], cfg, propagated_conf, "knn_propagation", disable_dataset_gate):
                        continue

                priority = propagation_priority(df.loc[idx], cfg) + propagated_conf * 3.0 - avg_distance * 0.2
                candidate_updates.append((idx, propagated_label, propagated_conf, avg_distance, priority))

            df = _apply_knn_updates(df, candidate_updates, cfg, column_added_counter)

    else:
        X_teacher = X[teacher_indices]
        y_teacher = df.loc[teacher_indices, "final_label_binary"].astype(int).values

        nn_model = NearestNeighbors(n_neighbors=min(cfg.knn_k, len(teacher_indices)), metric="euclidean")
        nn_model.fit(X_teacher)

        unlabeled_indices = df[df["final_label_binary"].isna()].index.tolist()
        candidate_updates = []

        for idx in unlabeled_indices:
            x = X[idx].reshape(1, -1)
            neighbor_distances, neighbor_pos = nn_model.kneighbors(x)
            neighbor_distances = neighbor_distances[0]
            neighbor_pos = neighbor_pos[0]
            neighbor_labels = y_teacher[neighbor_pos]

            avg_distance = float(np.mean(neighbor_distances))
            if avg_distance > cfg.knn_max_avg_distance:
                continue

            propagated_label, propagated_conf = _weighted_neighbor_vote(neighbor_labels, neighbor_distances, cfg)
            if propagated_label is None or propagated_conf is None:
                continue

            if cfg.dataset_gate == "generic":
                if propagated_conf < cfg.min_propagation_confidence:
                    continue
            else:
                if not should_allow_propagation_for_row(df.loc[idx], cfg, propagated_conf, "knn_propagation", disable_dataset_gate):
                    continue

            priority = propagation_priority(df.loc[idx], cfg) + propagated_conf * 3.0 - avg_distance * 0.2
            candidate_updates.append((idx, propagated_label, propagated_conf, avg_distance, priority))

        df = _apply_knn_updates(df, candidate_updates, cfg, column_added_counter)

    return df


# ============================================================
# 9. 总量裁剪
# ============================================================

def limit_total_propagated(df: pd.DataFrame, cfg: PropagationConfig, disable_total_cap: bool = False) -> pd.DataFrame:
    if disable_total_cap or cfg.max_total_propagated is None:
        return df

    df = df.copy()
    propagated_mask = df["label_source"].isin(["cluster_propagation", "knn_propagation"])
    propagated_df = df[propagated_mask].copy()

    if len(propagated_df) <= cfg.max_total_propagated:
        return df

    propagated_df["propagation_priority"] = propagated_df.apply(lambda r: propagation_priority(r, cfg), axis=1)
    keep_idx = propagated_df.sort_values(
        by=["propagation_priority", "propagation_confidence", "conflict_score", "posterior_error_probability"],
        ascending=[False, False, False, False],
    ).head(cfg.max_total_propagated).index

    drop_idx = propagated_df.index.difference(keep_idx)

    for idx in drop_idx:
        df.at[idx, "final_label_binary"] = np.nan
        df.at[idx, "final_label"] = None
        df.at[idx, "propagation_confidence"] = np.nan
        df.at[idx, "label_source"] = None
        df.at[idx, "sample_weight"] = np.nan
        df.at[idx, "propagation_priority"] = np.nan

    return df


# ============================================================
# 10. 导出
# ============================================================

def value_for_output(row: pd.Series, field: str):
    if field == "label":
        return row.get("final_label")
    if field == "label_binary":
        v = row.get("final_label_binary")
        return int(v) if pd.notna(v) else None
    if field == "cluster_id":
        v = row.get(field)
        return int(v) if pd.notna(v) else None
    if field == "is_sampled":
        v = row.get(field)
        return int(v) if pd.notna(v) else None
    if field in {"row_id"}:
        return int(row.get(field)) if pd.notna(row.get(field)) else None
    v = row.get(field)
    return make_json_safe(v)


def row_to_obj(row: pd.Series, cfg: PropagationConfig) -> Dict[str, Any]:
    fields = []
    for field in BASE_OUTPUT_FIELDS + cfg.extra_output_fields:
        if field not in fields:
            fields.append(field)

    obj = {}
    for field in fields:
        if field in {"label", "label_binary"}:
            obj[field] = value_for_output(row, field)
        elif field in row.index:
            obj[field] = value_for_output(row, field)
        else:
            obj[field] = None

    # old scripts use final label fields as output label fields only.
    obj["label"] = row.get("final_label")
    obj["label_binary"] = int(row["final_label_binary"]) if pd.notna(row["final_label_binary"]) else None
    obj["propagation_confidence"] = float(row["propagation_confidence"]) if pd.notna(row.get("propagation_confidence")) else None
    obj["sample_weight"] = float(row["sample_weight"]) if pd.notna(row.get("sample_weight")) else None

    # Ensure JSON-safe recursively for scalar object
    return {k: make_json_safe(v) for k, v in obj.items()}


def export_results(
    df: pd.DataFrame,
    cfg: PropagationConfig,
    output_jsonl: str,
    output_csv: str,
    output_summary_json: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    out_rows = []

    labeled_df = df[pd.notna(df["final_label_binary"])].copy()

    if cfg.dataset_gate in {"adult", "soccer"}:
        labeled_df = labeled_df.sort_values(by=["row_id", "column", "label_source"], ascending=[True, True, True])
    else:
        # 原通用脚本保持 df 顺序导出，这里不强制排序。
        pass

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for _, row in labeled_df.iterrows():
            obj = row_to_obj(row, cfg)
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            out_rows.append(obj)

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    summary = {
        "total_clustered_rows": int(len(df)),
        "total_labeled_or_propagated": int(len(labeled_df)),
        "llm_count": int((df["label_source"] == "llm").sum()),
        "cluster_propagation_count": int((df["label_source"] == "cluster_propagation").sum()),
        "knn_propagation_count": int((df["label_source"] == "knn_propagation").sum()),
        "unlabeled_count": int(df["final_label_binary"].isna().sum()),
        "coverage": float(len(labeled_df) / len(df)) if len(df) > 0 else 0.0,
        "label_source_distribution": df["label_source"].value_counts(dropna=False).to_dict(),
        "final_label_distribution": df["final_label"].value_counts(dropna=False).to_dict(),
        "column_distribution": labeled_df["column"].value_counts(dropna=False).to_dict() if len(labeled_df) else {},
        "column_source_distribution": (
            labeled_df.groupby(["column", "label_source"]).size().reset_index(name="count").to_dict(orient="records")
            if len(labeled_df) else []
        ),
        "config": {
            "dataset": cfg.name,
            "MIN_LABELED_PER_CLUSTER": cfg.min_labeled_per_cluster,
            "CLUSTER_PROPAGATION_THRESHOLD": cfg.cluster_propagation_threshold,
            "CLUSTER_CENTER_DIST_QUANTILE": cfg.cluster_center_dist_quantile,
            "MAX_CLUSTER_PROPAGATION_PER_CLUSTER": cfg.max_cluster_propagation_per_cluster,
            "ENABLE_KNN_PROPAGATION": cfg.enable_knn_propagation,
            "KNN_K": cfg.knn_k,
            "KNN_PROPAGATION_THRESHOLD": cfg.knn_propagation_threshold,
            "KNN_MIN_MARGIN": cfg.knn_min_margin,
            "KNN_MAX_AVG_DISTANCE": cfg.knn_max_avg_distance,
            "MIN_PROPAGATION_CONFIDENCE": cfg.min_propagation_confidence,
            "MAX_TOTAL_PROPAGATED": cfg.max_total_propagated,
            "DATASET_GATE": cfg.dataset_gate,
        },
    }

    if cfg.name == "adult":
        summary["adult_v2_attribution_bucket_distribution"] = (
            labeled_df["adult_v2_attribution_bucket"].value_counts(dropna=False).to_dict()
            if len(labeled_df) and "adult_v2_attribution_bucket" in labeled_df.columns else {}
        )
    if cfg.name == "soccer":
        summary["soccer_attribution_bucket_distribution"] = (
            labeled_df["soccer_attribution_bucket"].value_counts(dropna=False).to_dict()
            if len(labeled_df) and "soccer_attribution_bucket" in labeled_df.columns else {}
        )

    if output_summary_json:
        with open(output_summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=make_json_safe)

    return out_df, summary


# ============================================================
# 11. 主流程
# ============================================================

def run_propagation(cfg: PropagationConfig, args):
    if args.cluster_threshold is not None:
        cfg.cluster_propagation_threshold = float(args.cluster_threshold)
    if args.knn_threshold is not None:
        cfg.knn_propagation_threshold = float(args.knn_threshold)
    if args.knn_k is not None:
        cfg.knn_k = int(args.knn_k)
    if args.max_total_propagated is not None:
        cfg.max_total_propagated = int(args.max_total_propagated)

    input_clustered_csv = args.input_clustered_csv or cfg.input_clustered_csv
    input_labeled_csv = args.input_labeled_csv or cfg.input_labeled_csv
    output_jsonl = args.output_jsonl or cfg.output_jsonl
    output_csv = args.output_csv or cfg.output_csv
    output_summary_json = args.output_summary_json if args.output_summary_json is not None else cfg.output_summary_json

    print("[1/5] 合并 clustered + LLM labels ...")
    clustered_df = load_clustered_csv(input_clustered_csv, cfg)
    labeled_df = load_labeled_csv(input_labeled_csv, cfg)
    df = merge_clustered_and_labeled(clustered_df, labeled_df, cfg)

    print("[2/5] 准备特征矩阵 ...")
    X, feature_names = prepare_features(df, cfg)

    if args.disable_cluster_propagation:
        print("[3/5] 簇内标签传播已关闭。")
    else:
        print("[3/5] 簇内标签传播 ...")
        df = cluster_label_propagation(
            df,
            cfg,
            disable_center_filter=args.disable_center_filter,
            disable_dataset_gate=args.disable_dataset_gate,
        )

    if args.disable_knn_propagation:
        print("[4/5] KNN 标签传播已关闭。")
    else:
        print("[4/5] KNN 标签传播 ...")
        if cfg.enable_knn_propagation:
            df = knn_label_propagation(df, X, cfg, disable_dataset_gate=args.disable_dataset_gate)

    print("[4.5/5] 总量裁剪 ...")
    df = limit_total_propagated(df, cfg, disable_total_cap=args.disable_total_cap)

    print("[5/5] 导出结果 ...")
    out_df, summary = export_results(df, cfg, output_jsonl, output_csv, output_summary_json)

    total = len(df)
    llm_count = int((df["label_source"] == "llm").sum())
    cluster_count = int((df["label_source"] == "cluster_propagation").sum())
    knn_count = int((df["label_source"] == "knn_propagation").sum())
    total_labeled = int(pd.notna(df["final_label_binary"]).sum())

    print(f"[OK] 传播结果 JSONL 已保存: {output_jsonl}")
    print(f"[OK] 传播结果 CSV 已保存: {output_csv}")
    if output_summary_json:
        print(f"[OK] 传播摘要 JSON 已保存: {output_summary_json}")

    print("\n===== 统计信息 =====")
    print(f"总样本数: {total}")
    print(f"LLM 原始标注数: {llm_count}")
    print(f"簇内传播数: {cluster_count}")
    print(f"KNN 传播数: {knn_count}")
    print(f"最终可训练样本数: {total_labeled}")

    if total > 0:
        print(f"训练集覆盖率: {total_labeled / total:.6f}")

    print("\n标签来源分布:")
    print(df["label_source"].value_counts(dropna=False).to_string())

    print("\n最终标签分布:")
    print(df["final_label"].value_counts(dropna=False).to_string())

    print("\n按列分布:")
    if len(out_df) > 0 and "column" in out_df.columns:
        print(out_df["column"].value_counts(dropna=False).to_string())
    else:
        print("(empty)")


def run_one_dataset(dataset: str, args):
    cfg = CONFIGS[dataset]

    old_cwd = None
    if args.cwd:
        os.makedirs(args.cwd, exist_ok=True)
        old_cwd = os.getcwd()
        os.chdir(args.cwd)

    try:
        run_propagation(cfg, args)
    finally:
        if old_cwd is not None:
            os.chdir(old_cwd)


def parse_args():
    parser = argparse.ArgumentParser(description="Unified label propagation for data-cleaning candidates.")
    parser.add_argument("--dataset", required=True, choices=sorted(list(CONFIGS.keys()) + ["all"]))

    parser.add_argument("--input_clustered_csv", default=None)
    parser.add_argument("--input_labeled_csv", default=None)
    parser.add_argument("--output_jsonl", default=None)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--output_summary_json", default=None)
    parser.add_argument("--cwd", default=None)

    # 消融开关
    parser.add_argument("--disable-cluster-propagation", action="store_true")
    parser.add_argument("--disable-knn-propagation", action="store_true")
    parser.add_argument("--disable-center-filter", action="store_true")
    parser.add_argument("--disable-dataset-gate", action="store_true")
    parser.add_argument("--disable-total-cap", action="store_true")

    # 参数覆盖
    parser.add_argument("--cluster-threshold", type=float, default=None)
    parser.add_argument("--knn-threshold", type=float, default=None)
    parser.add_argument("--knn-k", type=int, default=None)
    parser.add_argument("--max-total-propagated", type=int, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.dataset == "all":
        if any([
            args.input_clustered_csv,
            args.input_labeled_csv,
            args.output_jsonl,
            args.output_csv,
            args.output_summary_json,
        ]):
            raise ValueError("--dataset all 不能和输入/输出路径覆盖参数一起使用。")
        for ds in ["hospital", "flights", "beers", "rayyan", "adult", "soccer"]:
            print("\n" + "=" * 80)
            print(f"[DATASET] {ds}")
            print("=" * 80)
            run_one_dataset(ds, args)
    else:
        run_one_dataset(args.dataset, args)


if __name__ == "__main__":
    main()

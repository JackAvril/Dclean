#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unified_build_final_training_set.py

统一版最终训练集构建脚本，覆盖：
hospital / flights / beers / rayyan / adult / soccer

核心流程：
1. 读取 dirty table、candidate_clustered、candidate_sampled_labeled、propagated_labels。
2. 构造候选内训练样本：
   - LLM 直接标注样本 label_source=llm
   - 高置信 cluster_propagation / knn_propagation 伪标签
3. 构造候选外 clean 样本：
   - 从不在候选集合中的单元格采样 correct
   - 优先常见值，避免长尾噪声
4. 合并、去重、平衡 correct/error 比例。
5. 导出 CSV / JSONL / summary JSON。

设计目标：
- 抽象统一逻辑，方便后续做消融实验。
- 保持每个数据集默认输入/输出文件名不变。
- adult/soccer 的列预算、强弱证据优先级、outside clean 控制放进配置和策略函数。

运行示例：
python unified_build_final_training_set.py --dataset soccer
python unified_build_final_training_set.py --dataset adult --cwd "/mnt/mydata/dq/projects/Splittree/test/Error Detection Adult"
python unified_build_final_training_set.py --dataset all

消融示例：
python unified_build_final_training_set.py --dataset soccer --disable-outside-clean
python unified_build_final_training_set.py --dataset soccer --disable-propagated-training
python unified_build_final_training_set.py --dataset soccer --disable-balance
python unified_build_final_training_set.py --dataset soccer --cluster-prop-min-conf 0.99 --knn-prop-min-conf 0.99
"""

import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


# ============================================================
# 1. 通用基础函数
# ============================================================

DEFAULT_MISSING_TOKENS = {
    "", "empty", "null", "none", "nan", "na", "n/a", "missing", "unk",
    "unknown", "?", "not available", "contact airline"
}


def normalize_value(x: Any, missing_tokens: Set[str]):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.lower() in missing_tokens:
        return None
    return s


def normalize_series(s: pd.Series, missing_tokens: Set[str]) -> pd.Series:
    return s.map(lambda x: normalize_value(x, missing_tokens))


def maybe_drop_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [c for c in df.columns if not str(c).lower().startswith("unnamed:")]
    return df[keep_cols].copy()


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


def normalize_label_binary_from_row(row: pd.Series) -> Optional[int]:
    if "label_binary" in row.index and pd.notna(row["label_binary"]):
        try:
            v = int(float(row["label_binary"]))
            if v in {0, 1}:
                return v
        except Exception:
            pass

    if "label_binary_norm" in row.index and pd.notna(row["label_binary_norm"]):
        try:
            v = int(float(row["label_binary_norm"]))
            if v in {0, 1}:
                return v
        except Exception:
            pass

    if "label" in row.index and pd.notna(row["label"]):
        s = str(row["label"]).strip().lower()
        if s == "error":
            return 1
        if s == "correct":
            return 0

    if "is_error" in row.index and pd.notna(row["is_error"]):
        s = str(row["is_error"]).strip().lower()
        if s in {"true", "1", "yes", "y", "error"}:
            return 1
        if s in {"false", "0", "no", "n", "correct"}:
            return 0

    return None


def label_name(y: int) -> str:
    return "error" if int(y) == 1 else "correct"


def build_key(row_id: Any, column: Any) -> Tuple[int, str]:
    return int(float(row_id)), str(column)


def compute_value_frequency_rank(counter: Dict[Any, int], value: Any) -> Optional[int]:
    if value is None:
        return None
    items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    for idx, (v, _) in enumerate(items, start=1):
        if v == value:
            return idx
    return None


def make_json_safe(v: Any):
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        if np.isnan(v):
            return None
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


# ============================================================
# 2. 数据集配置
# ============================================================

BASE_DEFAULTS = {
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
    "pattern_bucket": "unknown",
    "rarity_bucket": "unknown",
    "neighbor_bucket": "unknown",
    "bucket_id": "unknown",
    "cluster_id": -1,
    "dist_to_center": 0.0,
    "sample_role": "",
    "is_sampled": 0,
    "error_type": None,
    "reason_short": None,
    "reason_detailed": None,
    "suggested_correct_value": None,
    "needs_human_review": None,
    "evidence_used": None,
}

BASE_PREFERRED_COLS = [
    "row_id", "column", "value",
    "label", "label_binary", "label_source", "sample_weight", "is_outside_candidate",
    "propagation_confidence",

    "semantic_type",
    "violation_count", "conflict_score",
    "value_frequency", "value_frequency_rank",
    "neighbor_majority_value", "neighbor_majority_ratio",
    "prior_error_probability", "posterior_error_probability",

    "main_rule_type", "main_usage_role",
    "strong_rule_count", "candidate_generation_rule_count",
    "fd_like_count", "context_rule_count", "global_rule_count",
    "rare_value_count", "typo_rule_count", "pattern_rule_count", "schema_rule_count",

    "pattern_bucket", "rarity_bucket", "neighbor_bucket",
    "bucket_id", "cluster_id", "dist_to_center", "sample_role", "is_sampled",

    "error_type", "reason_short", "reason_detailed",
    "suggested_correct_value", "needs_human_review", "evidence_used",
]


@dataclass
class FinalTrainConfig:
    name: str
    input_table_csv: str
    input_clustered_csv: str
    input_labeled_csv: str
    input_propagated_csv: str
    output_final_train_csv: str
    output_final_train_jsonl: str
    output_summary_json: str

    missing_tokens: Set[str] = field(default_factory=lambda: set(DEFAULT_MISSING_TOKENS))

    keep_all_llm_labeled: bool = True
    cluster_prop_min_conf: float = 0.90
    knn_prop_min_conf: float = 0.95

    enable_outside_clean: bool = True
    outside_only_from_candidate_columns: bool = True
    outside_clean_per_column: int = 120
    outside_clean_min_per_column: int = 20
    outside_min_value_freq: int = 2
    outside_max_value_freq_rank: int = 20
    skip_null_in_outside_clean: bool = True

    weight_llm: float = 1.0
    weight_cluster_prop: float = 0.8
    weight_knn_prop: float = 0.6
    weight_outside_clean: float = 0.4

    enable_balance: bool = True
    max_correct_to_error_ratio: float = 1.5
    min_correct_keep: Optional[int] = None
    prefer_internal_correct_first: bool = True
    random_state: int = 42

    # adult/soccer 控制
    dataset_gate: str = "generic"  # generic / adult / soccer
    max_propagated_train_rows: Optional[int] = None
    max_propagated_per_column: Optional[int] = None
    propagated_column_limit: Dict[str, int] = field(default_factory=dict)
    default_propagated_column_limit: int = 10**9

    max_outside_clean_total: Optional[int] = None
    max_outside_clean_per_column: Dict[str, int] = field(default_factory=dict)
    default_max_outside_clean_per_column: int = 10**9
    outside_long_tail_columns: Set[str] = field(default_factory=set)
    outside_clean_allowed_columns: Optional[Set[str]] = None

    primary_target_columns: Set[str] = field(default_factory=set)
    context_only_columns: Set[str] = field(default_factory=set)

    extra_defaults: Dict[str, Any] = field(default_factory=dict)
    preferred_cols: List[str] = field(default_factory=lambda: list(BASE_PREFERRED_COLS))


def build_configs() -> Dict[str, FinalTrainConfig]:
    generic_tokens = {
        "", "empty", "null", "none", "nan", "na", "n/a", "missing", "unk",
        "not available", "contact airline"
    }

    cfgs: Dict[str, FinalTrainConfig] = {}

    cfgs["hospital"] = FinalTrainConfig(
        name="hospital",
        input_table_csv="/mnt/mydata/dq/projects/Splittree/hospital/hospital_dirty_no_address23.csv",
        input_clustered_csv="candidate_clustered_v2.csv",
        input_labeled_csv="candidate_sampled_labeled_v2.csv",
        input_propagated_csv="propagated_labels_v2.csv",
        output_final_train_csv="final_train_dataset_simple_v2.csv",
        output_final_train_jsonl="final_train_dataset_simple_v2.jsonl",
        output_summary_json="final_train_summary_simple_v2.json",
        missing_tokens={"", "empty", "null", "none", "nan", "na", "n/a", "missing", "unk"},
        extra_defaults=dict(BASE_DEFAULTS),
        preferred_cols=list(BASE_PREFERRED_COLS),
    )

    flights_defaults = {
        **BASE_DEFAULTS,
        "time_value_minutes": np.nan,
        "time_window_rule_count": 0,
        "time_window_bucket": "unknown",
        "time_value_bucket": "unknown",
    }
    flights_cols = [
        "row_id", "column", "value",
        "label", "label_binary", "label_source", "sample_weight", "is_outside_candidate",
        "propagation_confidence",
        "semantic_type",
        "violation_count", "conflict_score",
        "value_frequency", "value_frequency_rank", "time_value_minutes",
        "neighbor_majority_value", "neighbor_majority_ratio",
        "prior_error_probability", "posterior_error_probability",
        "main_rule_type", "main_usage_role",
        "strong_rule_count", "candidate_generation_rule_count",
        "fd_like_count", "context_rule_count", "global_rule_count",
        "rare_value_count", "typo_rule_count", "pattern_rule_count", "schema_rule_count",
        "time_window_rule_count",
        "pattern_bucket", "rarity_bucket", "neighbor_bucket",
        "time_window_bucket", "time_value_bucket",
        "bucket_id", "cluster_id", "dist_to_center", "sample_role", "is_sampled",
        "error_type", "reason_short", "reason_detailed",
        "suggested_correct_value", "needs_human_review", "evidence_used",
    ]
    cfgs["flights"] = FinalTrainConfig(
        name="flights",
        input_table_csv="/mnt/mydata/dq/projects/Splittree/flights/flights_dirty.csv",
        input_clustered_csv="candidate_clustered_flights.csv",
        input_labeled_csv="candidate_sampled_labeled_flights.csv",
        input_propagated_csv="propagated_labels_flights.csv",
        output_final_train_csv="final_train_dataset_flights.csv",
        output_final_train_jsonl="final_train_dataset_flights.jsonl",
        output_summary_json="final_train_summary_flights.json",
        missing_tokens=set(generic_tokens),
        extra_defaults=flights_defaults,
        preferred_cols=flights_cols,
    )

    beers_defaults = {
        **BASE_DEFAULTS,
        "numeric_value": np.nan,
        "numeric_window_rule_count": 0,
        "numeric_window_bucket": "unknown",
        "numeric_value_bucket": "unknown",
    }
    beers_cols = [
        "row_id", "column", "value",
        "label", "label_binary", "label_source", "sample_weight", "is_outside_candidate",
        "propagation_confidence",
        "semantic_type",
        "violation_count", "conflict_score",
        "value_frequency", "value_frequency_rank", "numeric_value",
        "neighbor_majority_value", "neighbor_majority_ratio",
        "prior_error_probability", "posterior_error_probability",
        "main_rule_type", "main_usage_role",
        "strong_rule_count", "candidate_generation_rule_count",
        "fd_like_count", "context_rule_count", "global_rule_count",
        "rare_value_count", "typo_rule_count", "pattern_rule_count", "schema_rule_count",
        "numeric_window_rule_count",
        "pattern_bucket", "rarity_bucket", "neighbor_bucket",
        "numeric_window_bucket", "numeric_value_bucket",
        "bucket_id", "cluster_id", "dist_to_center", "sample_role", "is_sampled",
        "error_type", "reason_short", "reason_detailed",
        "suggested_correct_value", "needs_human_review", "evidence_used",
    ]
    cfgs["beers"] = FinalTrainConfig(
        name="beers",
        input_table_csv="/mnt/mydata/dq/projects/Splittree/beers/beers_dirty.csv",
        input_clustered_csv="candidate_clustered_beers.csv",
        input_labeled_csv="candidate_sampled_labeled_beers.csv",
        input_propagated_csv="propagated_labels_beers.csv",
        output_final_train_csv="final_train_dataset_beers.csv",
        output_final_train_jsonl="final_train_dataset_beers.jsonl",
        output_summary_json="final_train_summary_beers.json",
        missing_tokens=set(generic_tokens),
        extra_defaults=beers_defaults,
        preferred_cols=beers_cols,
    )

    rayyan_defaults = {
        **BASE_DEFAULTS,
        "numeric_value": np.nan,
        "date_value": None,
        "language_canonical": None,
        "issn_canonical": None,
        "canonical_rule_count": 0,
        "canonical_bucket": "unknown",
        "date_value_bucket": "unknown",
        "language_bucket": "unknown",
    }
    rayyan_cols = [
        "row_id", "column", "value",
        "label", "label_binary", "label_source", "sample_weight", "is_outside_candidate",
        "propagation_confidence",
        "semantic_type",
        "violation_count", "conflict_score",
        "value_frequency", "value_frequency_rank",
        "numeric_value", "date_value", "language_canonical", "issn_canonical",
        "neighbor_majority_value", "neighbor_majority_ratio",
        "prior_error_probability", "posterior_error_probability",
        "main_rule_type", "main_usage_role",
        "strong_rule_count", "candidate_generation_rule_count",
        "fd_like_count", "context_rule_count", "global_rule_count",
        "rare_value_count", "typo_rule_count", "pattern_rule_count", "schema_rule_count",
        "canonical_rule_count",
        "pattern_bucket", "rarity_bucket", "neighbor_bucket",
        "canonical_bucket", "date_value_bucket", "language_bucket",
        "bucket_id", "cluster_id", "dist_to_center", "sample_role", "is_sampled",
        "error_type", "reason_short", "reason_detailed",
        "suggested_correct_value", "needs_human_review", "evidence_used",
    ]
    cfgs["rayyan"] = FinalTrainConfig(
        name="rayyan",
        input_table_csv="/mnt/mydata/dq/projects/Splittree/rayyan/rayyan_dirty.csv",
        input_clustered_csv="candidate_clustered_rayyan.csv",
        input_labeled_csv="candidate_sampled_labeled_rayyan.csv",
        input_propagated_csv="propagated_labels_rayyan.csv",
        output_final_train_csv="final_train_dataset_rayyan.csv",
        output_final_train_jsonl="final_train_dataset_rayyan.jsonl",
        output_summary_json="final_train_summary_rayyan.json",
        missing_tokens=set(generic_tokens),
        extra_defaults=rayyan_defaults,
        preferred_cols=rayyan_cols,
    )

    adult_primary = {"sex", "relationship", "education"}
    adult_context = {
        "age", "workclass", "maritalstatus", "occupation",
        "race", "hoursperweek", "country", "income"
    }
    adult_extra_defaults = {
        **BASE_DEFAULTS,
        "adult_domain_rule_count": 0,
        "adult_format_rule_count": 0,
        "adult_consistency_rule_count": 0,
        "age_lower": np.nan,
        "age_upper": np.nan,
        "hours_lower": np.nan,
        "hours_upper": np.nan,
        "canonical_income": None,
        "domain_valid": None,
        "education_order": np.nan,
        "adult_value_bucket": "unknown",
        "adult_profile_inconsistency_count": 0,
        "adult_consistency_score": 0.0,
        "adult_evidence_flags": "",
        "domain_bucket": "unknown",
        "adult_consistency_bucket": "unknown",
        "age_sampling_bucket": "unknown",
        "hours_sampling_bucket": "unknown",
        "time_window_rule_count": 0,
        "time_value_minutes": np.nan,
        "time_window_bucket": "adult_not_applicable",
        "time_value_bucket": "adult_not_applicable",
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
        "adult_v2_attribution_bucket": "unknown",
    }
    adult_cols = [
        "row_id", "column", "value",
        "label", "label_binary", "label_source", "sample_weight", "is_outside_candidate",
        "propagation_confidence",
        "semantic_type", "detected_type",
        "violation_count", "conflict_score",
        "value_frequency", "value_frequency_rank",
        "neighbor_majority_value", "neighbor_majority_ratio",
        "prior_error_probability", "posterior_error_probability",
        "main_rule_type", "main_usage_role",
        "strong_rule_count", "candidate_generation_rule_count",
        "fd_like_count", "context_rule_count", "global_rule_count",
        "rare_value_count", "typo_rule_count", "pattern_rule_count", "schema_rule_count",
        "adult_domain_rule_count", "adult_format_rule_count", "adult_consistency_rule_count",
        "age_lower", "age_upper", "hours_lower", "hours_upper",
        "canonical_income", "domain_valid", "education_order",
        "adult_value_bucket", "adult_profile_inconsistency_count",
        "adult_consistency_score", "adult_evidence_flags",
        "pattern_bucket", "rarity_bucket", "neighbor_bucket",
        "domain_bucket", "adult_consistency_bucket",
        "age_sampling_bucket", "hours_sampling_bucket",
        "time_window_rule_count", "time_value_minutes",
        "time_window_bucket", "time_value_bucket",
        "bucket_id", "cluster_id", "dist_to_center", "sample_role", "is_sampled", "signal_priority",
        "target_is_primary_column", "target_is_context_only_column",
        "is_relationship_spouse_value",
        "relationship_marital_conflict", "relationship_age_conflict", "relationship_sex_conflict",
        "sex_value_invalid", "education_age_extreme_conflict",
        "has_relationship_v2_rule", "has_education_v2_rule",
        "adult_v2_attribution_bucket",
        "error_type", "reason_short", "reason_detailed",
        "suggested_correct_value", "needs_human_review", "evidence_used",
    ]
    cfgs["adult"] = FinalTrainConfig(
        name="adult",
        input_table_csv="/mnt/mydata/dq/projects/Splittree/adult/adult_dirty.csv",
        input_clustered_csv="candidate_clustered_adult.csv",
        input_labeled_csv="candidate_sampled_labeled_adult.csv",
        input_propagated_csv="propagated_labels_adult.csv",
        output_final_train_csv="final_train_dataset_adult.csv",
        output_final_train_jsonl="final_train_dataset_adult.jsonl",
        output_summary_json="final_train_summary_adult.json",
        missing_tokens=set(DEFAULT_MISSING_TOKENS),
        cluster_prop_min_conf=0.90,
        knn_prop_min_conf=0.92,
        max_propagated_train_rows=700,
        max_propagated_per_column=80,
        outside_clean_per_column=80,
        outside_clean_min_per_column=15,
        outside_max_value_freq_rank=25,
        max_outside_clean_total=500,
        max_outside_clean_per_column={
            "relationship": 220, "sex": 180, "education": 100,
            "age": 0, "workclass": 0, "maritalstatus": 0, "occupation": 0,
            "race": 0, "hoursperweek": 0, "country": 0, "income": 0,
        },
        default_max_outside_clean_per_column=0,
        outside_long_tail_columns={"country", "occupation", "workclass"},
        weight_cluster_prop=0.75,
        weight_knn_prop=0.60,
        weight_outside_clean=0.35,
        max_correct_to_error_ratio=1.2,
        min_correct_keep=80,
        dataset_gate="adult",
        primary_target_columns=adult_primary,
        context_only_columns=adult_context,
        outside_clean_allowed_columns=set(adult_primary),
        propagated_column_limit={
            "relationship": 140, "sex": 120, "education": 50,
            "age": 0, "workclass": 0, "maritalstatus": 0, "occupation": 0,
            "race": 0, "hoursperweek": 0, "country": 0, "income": 0,
        },
        default_propagated_column_limit=0,
        extra_defaults=adult_extra_defaults,
        preferred_cols=adult_cols,
    )

    soccer_cols_set = {
        "name", "surname", "birthyear", "birthplace", "position",
        "team", "city", "stadium", "season", "manager"
    }
    soccer_extra_defaults = {
        **BASE_DEFAULTS,
        "evidence_only_fd_count": 0,
        "soccer_domain_rule_count": 0,
        "soccer_format_rule_count": 0,
        "soccer_numeric_rule_count": 0,
        "soccer_consistency_rule_count": 0,
        "soccer_profile_inconsistency_count": 0,
        "soccer_consistency_score": 0.0,
        "soccer_evidence_flags": "",
        "year_value": np.nan,
        "canonical_position": None,
        "domain_valid": None,
        "soccer_value_bucket": "unknown",
        "row_birthyear_int": np.nan,
        "row_season_int": np.nan,
        "row_age_at_season": np.nan,
        "row_position_canonical": None,
        "domain_bucket": "unknown",
        "soccer_consistency_bucket": "unknown",
        "signal_priority": 0.0,
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
        "soccer_attribution_bucket": "unknown",
        "adult_domain_rule_count": 0,
        "adult_format_rule_count": 0,
        "adult_consistency_rule_count": 0,
        "age_lower": np.nan,
        "age_upper": np.nan,
        "hours_lower": np.nan,
        "hours_upper": np.nan,
        "canonical_income": None,
        "education_order": np.nan,
        "adult_value_bucket": "soccer_not_applicable",
        "adult_profile_inconsistency_count": 0,
        "adult_consistency_score": 0.0,
        "adult_evidence_flags": "",
        "adult_consistency_bucket": "soccer_not_applicable",
        "age_sampling_bucket": "soccer_not_applicable",
        "hours_sampling_bucket": "soccer_not_applicable",
        "time_window_rule_count": 0,
        "time_value_minutes": np.nan,
        "time_window_bucket": "soccer_not_applicable",
        "time_value_bucket": "soccer_not_applicable",
        "adult_v2_attribution_bucket": "soccer_not_applicable",
    }
    soccer_cols = [
        "row_id", "column", "value",
        "label", "label_binary", "label_source", "sample_weight", "is_outside_candidate",
        "propagation_confidence",
        "semantic_type", "detected_type",
        "violation_count", "conflict_score",
        "value_frequency", "value_frequency_rank",
        "neighbor_majority_value", "neighbor_majority_ratio",
        "prior_error_probability", "posterior_error_probability",
        "main_rule_type", "main_usage_role",
        "strong_rule_count", "candidate_generation_rule_count",
        "fd_like_count", "evidence_only_fd_count", "context_rule_count", "global_rule_count",
        "rare_value_count", "typo_rule_count", "pattern_rule_count", "schema_rule_count",
        "soccer_domain_rule_count", "soccer_format_rule_count",
        "soccer_numeric_rule_count", "soccer_consistency_rule_count",
        "soccer_profile_inconsistency_count", "soccer_consistency_score", "soccer_evidence_flags",
        "year_value", "canonical_position", "domain_valid", "soccer_value_bucket",
        "row_birthyear_int", "row_season_int", "row_age_at_season", "row_position_canonical",
        "pattern_bucket", "rarity_bucket", "neighbor_bucket",
        "domain_bucket", "soccer_consistency_bucket",
        "bucket_id", "cluster_id", "dist_to_center", "sample_role", "is_sampled", "signal_priority",
        "target_is_primary_column", "target_is_context_only_column",
        "birthyear_value_invalid", "season_value_invalid",
        "position_value_invalid", "position_needs_canonicalization",
        "birthyear_season_age_conflict", "team_context_conflict",
        "format_rule_signal", "fd_like_signal", "evidence_only_fd_count", "soccer_attribution_bucket",
        "adult_domain_rule_count", "adult_format_rule_count", "adult_consistency_rule_count",
        "age_lower", "age_upper", "hours_lower", "hours_upper",
        "canonical_income", "education_order",
        "adult_value_bucket", "adult_profile_inconsistency_count",
        "adult_consistency_score", "adult_evidence_flags",
        "adult_consistency_bucket", "age_sampling_bucket", "hours_sampling_bucket",
        "time_window_rule_count", "time_value_minutes",
        "time_window_bucket", "time_value_bucket",
        "adult_v2_attribution_bucket",
        "error_type", "reason_short", "reason_detailed",
        "suggested_correct_value", "needs_human_review", "evidence_used",
    ]
    cfgs["soccer"] = FinalTrainConfig(
        name="soccer",
        input_table_csv="/mnt/mydata/dq/projects/Splittree/soccer/soccer_dirty.csv",
        input_clustered_csv="candidate_clustered_soccer.csv",
        input_labeled_csv="candidate_sampled_labeled_soccer.csv",
        input_propagated_csv="propagated_labels_soccer.csv",
        output_final_train_csv="final_train_dataset_soccer.csv",
        output_final_train_jsonl="final_train_dataset_soccer.jsonl",
        output_summary_json="final_train_summary_soccer.json",
        missing_tokens=set(DEFAULT_MISSING_TOKENS),
        cluster_prop_min_conf=0.90,
        knn_prop_min_conf=0.92,
        max_propagated_train_rows=700,
        max_propagated_per_column=80,
        outside_clean_per_column=70,
        outside_clean_min_per_column=10,
        outside_max_value_freq_rank=25,
        max_outside_clean_total=600,
        max_outside_clean_per_column={
            "birthyear": 100, "season": 90, "position": 90,
            "team": 55, "city": 55, "stadium": 55, "manager": 50,
            "birthplace": 60, "name": 40, "surname": 40,
        },
        default_max_outside_clean_per_column=40,
        outside_long_tail_columns={"name", "surname", "team", "city", "stadium", "manager", "birthplace"},
        weight_cluster_prop=0.75,
        weight_knn_prop=0.60,
        weight_outside_clean=0.35,
        max_correct_to_error_ratio=1.2,
        min_correct_keep=80,
        dataset_gate="soccer",
        primary_target_columns=set(soccer_cols_set),
        context_only_columns=set(),
        outside_clean_allowed_columns=set(soccer_cols_set),
        propagated_column_limit={
            "birthyear": 130, "season": 110, "position": 110,
            "team": 70, "city": 70, "stadium": 70, "manager": 60,
            "birthplace": 80, "name": 60, "surname": 60,
        },
        default_propagated_column_limit=50,
        extra_defaults=soccer_extra_defaults,
        preferred_cols=soccer_cols,
    )

    return cfgs


CONFIGS = build_configs()


# ============================================================
# 3. 数据加载
# ============================================================

def fill_missing_feature_defaults(df: pd.DataFrame, cfg: FinalTrainConfig) -> pd.DataFrame:
    if len(df) == 0:
        return df

    df = df.copy()
    for col, default_val in cfg.extra_defaults.items():
        if col not in df.columns:
            df[col] = default_val

    if cfg.name == "adult":
        if "column" in df.columns:
            df["target_is_primary_column"] = df["column"].astype(str).isin(cfg.primary_target_columns).astype(int)
            df["target_is_context_only_column"] = df["column"].astype(str).isin(cfg.context_only_columns).astype(int)

    if cfg.name == "soccer":
        if "column" in df.columns:
            df["target_is_primary_column"] = df["column"].astype(str).isin(cfg.primary_target_columns).astype(int)
            df["target_is_context_only_column"] = 0

        if "main_rule_type" not in df.columns:
            df["main_rule_type"] = "none"
        rule = df["main_rule_type"].fillna("").astype(str)

        if "evidence_only_fd_count" not in df.columns:
            df["evidence_only_fd_count"] = 0

        soft_fd_mask = rule.isin({"soft_functional_dependency"})
        if soft_fd_mask.any():
            df.loc[soft_fd_mask, "evidence_only_fd_count"] = (
                pd.to_numeric(df.loc[soft_fd_mask, "evidence_only_fd_count"], errors="coerce").fillna(0) + 1
            )
            if "fd_like_signal" in df.columns:
                df.loc[soft_fd_mask, "fd_like_signal"] = 0

        # 由主规则兜底补齐 soccer v2 归因字段
        df["birthyear_value_invalid"] = np.where(rule.eq("birthyear_numeric_range"), 1, df.get("birthyear_value_invalid", 0))
        df["season_value_invalid"] = np.where(rule.eq("season_numeric_range"), 1, df.get("season_value_invalid", 0))
        df["position_value_invalid"] = np.where(rule.eq("position_domain"), 1, df.get("position_value_invalid", 0))
        df["birthyear_season_age_conflict"] = np.where(rule.eq("birthyear_season_age_consistency"), 1, df.get("birthyear_season_age_conflict", 0))
        df["team_context_conflict"] = np.where(
            rule.isin({"team_context_dominant", "team_season_manager_consistency_hint", "city_stadium_consistency_hint", "dominant_value_by_context", "dominant_value_by_context_pair"}),
            1,
            df.get("team_context_conflict", 0),
        )
        df["format_rule_signal"] = np.where(
            rule.isin({
                "name_pattern", "surname_pattern", "manager_name_pattern", "team_name_pattern",
                "birthplace_location_pattern", "city_location_pattern", "stadium_name_pattern",
                "pattern", "generic_regex",
            }),
            1,
            df.get("format_rule_signal", 0),
        )
        df["fd_like_signal"] = np.where(rule.eq("functional_dependency"), 1, df.get("fd_like_signal", 0))

    return df


def load_table(path: str, cfg: FinalTrainConfig) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df = maybe_drop_unnamed_columns(df)
    for col in df.columns:
        df[col] = normalize_series(df[col], cfg.missing_tokens)
    return df


def _load_csv_with_defaults(path: str, cfg: FinalTrainConfig) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if "row_id" in df.columns:
        df["row_id"] = pd.to_numeric(df["row_id"], errors="coerce").fillna(-1).astype(int)
    if "column" in df.columns:
        df["column"] = df["column"].astype(str)
    df = fill_missing_feature_defaults(df, cfg)
    return df


def load_clustered_csv(path: str, cfg: FinalTrainConfig) -> pd.DataFrame:
    df = _load_csv_with_defaults(path, cfg)
    if "row_id" not in df.columns or "column" not in df.columns:
        raise ValueError(f"{path} 必须包含 row_id 和 column。")
    return df


def load_labeled_csv(path: str, cfg: FinalTrainConfig) -> pd.DataFrame:
    df = _load_csv_with_defaults(path, cfg)
    if "row_id" not in df.columns or "column" not in df.columns:
        raise ValueError(f"{path} 必须包含 row_id 和 column。")

    label_binary_list = []
    for _, row in df.iterrows():
        label_binary_list.append(normalize_label_binary_from_row(row))

    df["label_binary_norm"] = label_binary_list
    df = df[pd.notna(df["label_binary_norm"])].copy()
    df["label_binary_norm"] = df["label_binary_norm"].astype(int)

    if "confidence" not in df.columns:
        df["confidence"] = 1.0
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(1.0).clip(0.0, 1.0)

    # 同一 cell 多次标注时保留置信度最高者
    df["_key"] = list(zip(df["row_id"].astype(int), df["column"].astype(str)))
    df = df.sort_values(by=["_key", "confidence"], ascending=[True, False])
    df = df.drop_duplicates(subset=["_key"], keep="first").drop(columns=["_key"]).copy()

    return fill_missing_feature_defaults(df, cfg)


def load_propagated_csv(path: str, cfg: FinalTrainConfig) -> pd.DataFrame:
    df = _load_csv_with_defaults(path, cfg)
    if "row_id" not in df.columns or "column" not in df.columns:
        raise ValueError(f"{path} 必须包含 row_id 和 column。")

    if "label_binary" not in df.columns:
        label_binary_list = []
        for _, row in df.iterrows():
            label_binary_list.append(normalize_label_binary_from_row(row))
        df["label_binary"] = label_binary_list

    if "propagation_confidence" not in df.columns:
        if "confidence" in df.columns:
            df["propagation_confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
        else:
            df["propagation_confidence"] = np.nan

    if "label_source" not in df.columns:
        df["label_source"] = "unknown"

    if "sample_weight" not in df.columns:
        df["sample_weight"] = np.nan

    return fill_missing_feature_defaults(df, cfg)


# ============================================================
# 4. adult/soccer 优先级与信号
# ============================================================

RELATIONSHIP_V2_RULE_TYPES = {
    "relationship_marital_spouse_conflict",
    "relationship_age_spouse_conflict",
    "relationship_sex_spouse_conflict",
    "relationship_context_dominant",
}
EDUCATION_V2_RULE_TYPES = {"education_age_extreme_conflict"}

SOCCER_CONTEXT_RULE_TYPES = {
    "team_context_dominant",
    "team_season_manager_consistency_hint",
    "city_stadium_consistency_hint",
    "dominant_value_by_context",
    "dominant_value_by_context_pair",
}
SOCCER_FD_RULE_TYPES = {"functional_dependency"}
SOCCER_EVIDENCE_ONLY_FD_RULE_TYPES = {"soft_functional_dependency"}


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


def adult_training_priority(row: pd.Series) -> float:
    score = 0.0
    col = str(row.get("column", ""))

    if col == "relationship":
        score += 8.0
    elif col == "sex":
        score += 7.0
    elif col == "education":
        score += 4.0
    elif col in {"age", "workclass", "maritalstatus", "occupation", "race", "hoursperweek", "country", "income"}:
        score -= 8.0

    score += safe_float(row.get("propagation_confidence", 0.0), 0.0) * 5.0
    score += safe_float(row.get("sample_weight", 0.0), 0.0) * 3.0
    score += safe_float(row.get("conflict_score", 0.0), 0.0) / 20.0
    score += safe_float(row.get("strong_rule_count", 0.0), 0.0) * 2.0
    score += safe_float(row.get("adult_domain_rule_count", 0.0), 0.0) * 2.0
    score += safe_float(row.get("adult_format_rule_count", 0.0), 0.0) * 1.8
    score += safe_float(row.get("adult_consistency_rule_count", 0.0), 0.0) * 1.5

    if has_relationship_v2_signal(row):
        score += 10.0
    if has_sex_invalid_signal(row):
        score += 9.0
    if has_education_v2_signal(row):
        score += 5.0

    return float(score)


def soccer_has_birthyear_invalid_signal(row: pd.Series) -> bool:
    return as_int01(row.get("birthyear_value_invalid", 0)) == 1


def soccer_has_season_invalid_signal(row: pd.Series) -> bool:
    return as_int01(row.get("season_value_invalid", 0)) == 1


def soccer_has_position_invalid_signal(row: pd.Series) -> bool:
    return (
        as_int01(row.get("position_value_invalid", 0)) == 1
        or as_int01(row.get("position_needs_canonicalization", 0)) == 1
    )


def soccer_has_age_consistency_signal(row: pd.Series) -> bool:
    rule = str(row.get("main_rule_type", ""))
    return as_int01(row.get("birthyear_season_age_conflict", 0)) == 1 or rule == "birthyear_season_age_consistency"


def soccer_has_team_context_signal(row: pd.Series) -> bool:
    rule = str(row.get("main_rule_type", ""))
    return as_int01(row.get("team_context_conflict", 0)) == 1 or rule in SOCCER_CONTEXT_RULE_TYPES


def soccer_has_fd_like_signal(row: pd.Series) -> bool:
    rule = str(row.get("main_rule_type", ""))
    return as_int01(row.get("fd_like_signal", 0)) == 1 or safe_float(row.get("fd_like_count", 0.0), 0.0) > 0 or rule in SOCCER_FD_RULE_TYPES


def soccer_has_evidence_only_fd_signal(row: pd.Series) -> bool:
    rule = str(row.get("main_rule_type", ""))
    return safe_float(row.get("evidence_only_fd_count", 0.0), 0.0) > 0 or rule in SOCCER_EVIDENCE_ONLY_FD_RULE_TYPES


def soccer_has_format_signal(row: pd.Series) -> bool:
    return (
        as_int01(row.get("format_rule_signal", 0)) == 1
        or safe_float(row.get("soccer_format_rule_count", 0.0), 0.0) > 0
        or safe_float(row.get("adult_format_rule_count", 0.0), 0.0) > 0
    )


def soccer_training_priority(row: pd.Series) -> float:
    score = 0.0
    col = str(row.get("column", ""))

    if col == "birthyear":
        score += 8.0
    elif col == "season":
        score += 7.5
    elif col == "position":
        score += 7.0
    elif col in {"team", "city", "stadium", "manager"}:
        score += 4.0
    elif col == "birthplace":
        score += 4.0
    elif col in {"name", "surname"}:
        score += 3.0

    score += safe_float(row.get("propagation_confidence", 0.0), 0.0) * 5.0
    score += safe_float(row.get("sample_weight", 0.0), 0.0) * 3.0
    score += safe_float(row.get("conflict_score", 0.0), 0.0) / 20.0
    score += safe_float(row.get("strong_rule_count", 0.0), 0.0) * 2.0
    score += safe_float(row.get("soccer_domain_rule_count", 0.0), 0.0) * 2.0
    score += safe_float(row.get("soccer_format_rule_count", 0.0), 0.0) * 1.8
    score += safe_float(row.get("soccer_numeric_rule_count", 0.0), 0.0) * 2.0
    score += safe_float(row.get("soccer_consistency_rule_count", 0.0), 0.0) * 1.5
    score += safe_float(row.get("schema_rule_count", 0.0), 0.0) * 1.0

    if soccer_has_birthyear_invalid_signal(row):
        score += 10.0
    if soccer_has_season_invalid_signal(row):
        score += 9.0
    if soccer_has_position_invalid_signal(row):
        score += 9.0
    if soccer_has_age_consistency_signal(row):
        score += 8.0
    if soccer_has_format_signal(row):
        score += 4.5

    if soccer_has_team_context_signal(row):
        score += 1.0
    if soccer_has_fd_like_signal(row):
        score += 0.8
    if soccer_has_evidence_only_fd_signal(row):
        score -= 1.2

    return float(score)


def training_priority(row: pd.Series, cfg: FinalTrainConfig) -> float:
    if cfg.dataset_gate == "adult":
        return adult_training_priority(row)
    if cfg.dataset_gate == "soccer":
        return soccer_training_priority(row)

    score = 0.0
    score += safe_float(row.get("propagation_confidence", 0.0), 0.0) * 5.0
    score += safe_float(row.get("sample_weight", 0.0), 0.0) * 3.0
    score += safe_float(row.get("conflict_score", 0.0), 0.0) / 20.0
    score += safe_float(row.get("strong_rule_count", 0.0), 0.0) * 1.5
    score += safe_float(row.get("schema_rule_count", 0.0), 0.0)
    return float(score)


def outside_clean_priority(col: str, value_freq: int, value_rank: Optional[int], cfg: FinalTrainConfig) -> float:
    rank = value_rank if value_rank is not None and not pd.isna(value_rank) else 999999
    score = float(value_freq) - 0.05 * float(rank)

    if col in cfg.outside_long_tail_columns:
        score -= 1.0

    if cfg.dataset_gate == "soccer" and col in {"birthyear", "season", "position"}:
        score += 2.0

    return float(score)


# ============================================================
# 5. 训练样本构造
# ============================================================

def merge_extra_features_from_row(out: Dict[str, Any], row: pd.Series, cfg: FinalTrainConfig) -> Dict[str, Any]:
    for col in cfg.extra_defaults.keys():
        if col in row.index and pd.notna(row.get(col)):
            out[col] = row.get(col)

    # 保留所有已存在的额外列，避免上游新特征丢失
    for col in row.index:
        if col in {
            "label", "label_binary", "label_binary_norm", "is_error",
            "confidence", "_key",
        }:
            continue
        if col not in out or (pd.isna(out.get(col)) if not isinstance(out.get(col), (list, dict, tuple)) else False):
            out[col] = row.get(col)

    return out


def source_priority_value(src: Any) -> int:
    order = {
        "llm": 5,
        "cluster_propagation": 4,
        "knn_propagation": 3,
        "outside_clean": 1,
    }
    return order.get(str(src), 0)


def sort_and_dedup_by_priority(df: pd.DataFrame, cfg: FinalTrainConfig) -> pd.DataFrame:
    if len(df) == 0:
        return df

    df = df.copy()
    df["_source_priority"] = df["label_source"].map(source_priority_value)
    df["_conf_sort"] = pd.to_numeric(df.get("propagation_confidence", 0.0), errors="coerce").fillna(0.0)
    df["_weight_sort"] = pd.to_numeric(df.get("sample_weight", 0.0), errors="coerce").fillna(0.0)
    df["_train_priority"] = df.apply(lambda r: training_priority(r, cfg), axis=1)

    df = df.sort_values(
        by=["row_id", "column", "_source_priority", "_train_priority", "_conf_sort", "_weight_sort"],
        ascending=[True, True, False, False, False, False],
    )
    df = df.drop_duplicates(subset=["row_id", "column"], keep="first").copy()
    df = df.drop(columns=["_source_priority", "_conf_sort", "_weight_sort", "_train_priority"], errors="ignore")
    return df


def build_candidate_training_from_labels(
    clustered_df: pd.DataFrame,
    labeled_df: pd.DataFrame,
    propagated_df: pd.DataFrame,
    cfg: FinalTrainConfig,
    disable_propagated_training: bool = False,
) -> pd.DataFrame:
    clustered_map: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for _, row in clustered_df.iterrows():
        clustered_map[build_key(row["row_id"], row["column"])] = row.to_dict()

    train_rows: List[Dict[str, Any]] = []

    # 5.1 LLM 直接标注样本
    if cfg.keep_all_llm_labeled:
        for _, row in labeled_df.iterrows():
            key = build_key(row["row_id"], row["column"])

            # generic 旧逻辑要求在 clustered 中；adult/soccer 允许兜底。
            if cfg.dataset_gate == "generic" and key not in clustered_map:
                continue

            base = dict(clustered_map[key]) if key in clustered_map else {}
            label_binary = int(row["label_binary_norm"])
            confidence = safe_float(row.get("confidence", 1.0), 1.0)

            out = dict(base)
            out["row_id"] = key[0]
            out["column"] = key[1]
            out["value"] = row.get("value", base.get("value"))
            out["label_binary"] = label_binary
            out["label"] = label_name(label_binary)
            out["label_source"] = "llm"
            out["propagation_confidence"] = confidence
            out["sample_weight"] = cfg.weight_llm
            out["is_outside_candidate"] = 0

            for extra_col in [
                "error_type", "reason_short", "reason_detailed",
                "suggested_correct_value", "needs_human_review", "evidence_used",
            ]:
                out[extra_col] = row.get(extra_col) if extra_col in row.index else None

            out = merge_extra_features_from_row(out, row, cfg)
            train_rows.append(out)

    if disable_propagated_training:
        df = pd.DataFrame(train_rows)
        return fill_missing_feature_defaults(sort_and_dedup_by_priority(df, cfg), cfg) if len(df) else df

    # 5.2 传播样本：去掉 LLM 直接标注
    llm_keys = set(build_key(r["row_id"], r["column"]) for _, r in labeled_df.iterrows())
    prop_rows: List[Dict[str, Any]] = []

    for _, row in propagated_df.iterrows():
        key = build_key(row["row_id"], row["column"])
        if key in llm_keys:
            continue
        if cfg.dataset_gate == "generic" and key not in clustered_map:
            continue

        label_source = str(row.get("label_source", "")).strip()
        if label_source not in {"cluster_propagation", "knn_propagation"}:
            continue

        label_binary = normalize_label_binary_from_row(row)
        if label_binary is None:
            continue

        prop_conf = safe_float(row.get("propagation_confidence", np.nan), np.nan)
        if pd.isna(prop_conf):
            continue

        keep = False
        sample_weight = None
        if label_source == "cluster_propagation" and prop_conf >= cfg.cluster_prop_min_conf:
            keep = True
            sample_weight = cfg.weight_cluster_prop
        elif label_source == "knn_propagation" and prop_conf >= cfg.knn_prop_min_conf:
            keep = True
            sample_weight = cfg.weight_knn_prop

        if not keep:
            continue

        base = dict(clustered_map[key]) if key in clustered_map else {}
        out = dict(base)
        out["row_id"] = key[0]
        out["column"] = key[1]
        out["value"] = row.get("value", base.get("value"))
        out["label_binary"] = int(label_binary)
        out["label"] = label_name(label_binary)
        out["label_source"] = label_source
        out["propagation_confidence"] = prop_conf
        out["sample_weight"] = sample_weight
        out["is_outside_candidate"] = 0

        for extra_col in [
            "error_type", "reason_short", "reason_detailed",
            "suggested_correct_value", "needs_human_review", "evidence_used",
        ]:
            out[extra_col] = row.get(extra_col) if extra_col in row.index else None

        out = merge_extra_features_from_row(out, row, cfg)
        prop_rows.append(out)

    prop_df = pd.DataFrame(prop_rows)
    if len(prop_df) > 0 and cfg.dataset_gate in {"adult", "soccer"}:
        prop_df = fill_missing_feature_defaults(prop_df, cfg)
        prop_df["_prop_priority"] = prop_df.apply(lambda r: training_priority(r, cfg), axis=1)

        if cfg.max_propagated_per_column is not None:
            kept_parts = []
            for col, g in prop_df.groupby("column", sort=False):
                col_limit = min(
                    int(cfg.max_propagated_per_column),
                    int(cfg.propagated_column_limit.get(str(col), cfg.default_propagated_column_limit)),
                )
                if col_limit <= 0:
                    continue
                if len(g) > col_limit:
                    g = g.sort_values(
                        by=["_prop_priority", "propagation_confidence", "conflict_score"],
                        ascending=[False, False, False],
                    ).head(col_limit)
                kept_parts.append(g)
            prop_df = pd.concat(kept_parts, ignore_index=True) if kept_parts else prop_df.iloc[0:0]

        if cfg.max_propagated_train_rows is not None and len(prop_df) > cfg.max_propagated_train_rows:
            prop_df = prop_df.sort_values(
                by=["_prop_priority", "propagation_confidence", "conflict_score"],
                ascending=[False, False, False],
            ).head(cfg.max_propagated_train_rows)

        prop_df = prop_df.drop(columns=["_prop_priority"], errors="ignore")

    if len(prop_df) > 0:
        train_rows.extend(prop_df.to_dict(orient="records"))

    df = pd.DataFrame(train_rows)
    if len(df) == 0:
        return df

    df = fill_missing_feature_defaults(df, cfg)
    df = sort_and_dedup_by_priority(df, cfg)
    return df


def build_outside_clean_row(row_id: int, col: str, value: Any, value_freq: int, value_rank: Any, cfg: FinalTrainConfig) -> Dict[str, Any]:
    out = {
        "row_id": int(row_id),
        "column": str(col),
        "value": value,
        "semantic_type": "outside_clean_unknown",
        "violation_count": 0,
        "conflict_score": 0.0,
        "value_frequency": safe_int(value_freq, 0),
        "value_frequency_rank": value_rank,
        "neighbor_majority_value": None,
        "neighbor_majority_ratio": np.nan,
        "prior_error_probability": 0.01,
        "posterior_error_probability": 0.01,
        "main_rule_type": "none",
        "main_usage_role": "outside_clean",
        "strong_rule_count": 0,
        "candidate_generation_rule_count": 0,
        "fd_like_count": 0,
        "context_rule_count": 0,
        "global_rule_count": 0,
        "rare_value_count": 0,
        "typo_rule_count": 0,
        "pattern_rule_count": 0,
        "schema_rule_count": 0,
        "pattern_bucket": "outside_clean_unknown",
        "rarity_bucket": "outside_clean_unknown",
        "neighbor_bucket": "outside_clean_unknown",
        "bucket_id": "outside_clean",
        "cluster_id": -1,
        "dist_to_center": 0.0,
        "sample_role": "outside_clean",
        "is_sampled": 0,
        "label_binary": 0,
        "label": "correct",
        "label_source": "outside_clean",
        "propagation_confidence": 0.95,
        "sample_weight": cfg.weight_outside_clean,
        "is_outside_candidate": 1,
        "error_type": None,
        "reason_short": None,
        "reason_detailed": None,
        "suggested_correct_value": None,
        "needs_human_review": False,
        "evidence_used": "outside_candidate_clean_sampling",
    }

    # 数据集专属 outside-clean 默认字段
    if cfg.name == "flights":
        out.update({
            "time_value_minutes": np.nan,
            "time_window_rule_count": 0,
            "time_window_bucket": "outside_clean_unknown",
            "time_value_bucket": "outside_clean_unknown",
        })

    elif cfg.name == "beers":
        out.update({
            "numeric_value": np.nan,
            "numeric_window_rule_count": 0,
            "numeric_window_bucket": "outside_clean_unknown",
            "numeric_value_bucket": "outside_clean_unknown",
        })

    elif cfg.name == "rayyan":
        out.update({
            "numeric_value": np.nan,
            "date_value": None,
            "language_canonical": None,
            "issn_canonical": None,
            "canonical_rule_count": 0,
            "canonical_bucket": "outside_clean_unknown",
            "date_value_bucket": "outside_clean_unknown",
            "language_bucket": "outside_clean_unknown",
        })

    elif cfg.name == "adult":
        out.update({
            "detected_type": "outside_clean_unknown",
            "adult_domain_rule_count": 0,
            "adult_format_rule_count": 0,
            "adult_consistency_rule_count": 0,
            "age_lower": np.nan,
            "age_upper": np.nan,
            "hours_lower": np.nan,
            "hours_upper": np.nan,
            "canonical_income": None,
            "domain_valid": None,
            "education_order": np.nan,
            "adult_value_bucket": "outside_clean_unknown",
            "adult_profile_inconsistency_count": 0,
            "adult_consistency_score": 0.0,
            "adult_evidence_flags": "",
            "domain_bucket": "outside_clean_unknown",
            "adult_consistency_bucket": "outside_clean_unknown",
            "age_sampling_bucket": "outside_clean_unknown",
            "hours_sampling_bucket": "outside_clean_unknown",
            "time_window_rule_count": 0,
            "time_value_minutes": np.nan,
            "time_window_bucket": "adult_not_applicable",
            "time_value_bucket": "adult_not_applicable",
            "signal_priority": 0.0,
            "target_is_primary_column": 1 if col in cfg.primary_target_columns else 0,
            "target_is_context_only_column": 1 if col in cfg.context_only_columns else 0,
            "is_relationship_spouse_value": 0,
            "relationship_marital_conflict": 0,
            "relationship_age_conflict": 0,
            "relationship_sex_conflict": 0,
            "sex_value_invalid": 0,
            "education_age_extreme_conflict": 0,
            "has_relationship_v2_rule": 0,
            "has_education_v2_rule": 0,
            "adult_v2_attribution_bucket": "outside_clean_primary_target" if col in cfg.primary_target_columns else "outside_clean_other",
        })

    elif cfg.name == "soccer":
        out.update({
            "detected_type": "outside_clean_unknown",
            "evidence_only_fd_count": 0,
            "soccer_domain_rule_count": 0,
            "soccer_format_rule_count": 0,
            "soccer_numeric_rule_count": 0,
            "soccer_consistency_rule_count": 0,
            "soccer_profile_inconsistency_count": 0,
            "soccer_consistency_score": 0.0,
            "soccer_evidence_flags": "",
            "year_value": np.nan,
            "canonical_position": None,
            "domain_valid": None,
            "soccer_value_bucket": "outside_clean_unknown",
            "row_birthyear_int": np.nan,
            "row_season_int": np.nan,
            "row_age_at_season": np.nan,
            "row_position_canonical": None,
            "domain_bucket": "outside_clean_unknown",
            "soccer_consistency_bucket": "outside_clean_unknown",
            "signal_priority": 0.0,
            "target_is_primary_column": 1 if col in cfg.primary_target_columns else 0,
            "target_is_context_only_column": 1 if col in cfg.context_only_columns else 0,
            "birthyear_value_invalid": 0,
            "season_value_invalid": 0,
            "position_value_invalid": 0,
            "position_needs_canonicalization": 0,
            "birthyear_season_age_conflict": 0,
            "team_context_conflict": 0,
            "format_rule_signal": 0,
            "fd_like_signal": 0,
            "soccer_attribution_bucket": "outside_clean_primary_target" if col in cfg.primary_target_columns else "outside_clean_other",
            "adult_domain_rule_count": 0,
            "adult_format_rule_count": 0,
            "adult_consistency_rule_count": 0,
            "age_lower": np.nan,
            "age_upper": np.nan,
            "hours_lower": np.nan,
            "hours_upper": np.nan,
            "canonical_income": None,
            "education_order": np.nan,
            "adult_value_bucket": "soccer_not_applicable",
            "adult_profile_inconsistency_count": 0,
            "adult_consistency_score": 0.0,
            "adult_evidence_flags": "",
            "adult_consistency_bucket": "soccer_not_applicable",
            "age_sampling_bucket": "soccer_not_applicable",
            "hours_sampling_bucket": "soccer_not_applicable",
            "time_window_rule_count": 0,
            "time_value_minutes": np.nan,
            "time_window_bucket": "soccer_not_applicable",
            "time_value_bucket": "soccer_not_applicable",
            "adult_v2_attribution_bucket": "soccer_not_applicable",
        })

    return fill_missing_feature_defaults(pd.DataFrame([out]), cfg).iloc[0].to_dict()


def build_outside_clean_training_simple(
    df_table: pd.DataFrame,
    clustered_df: pd.DataFrame,
    candidate_train_df: pd.DataFrame,
    cfg: FinalTrainConfig,
    disable_outside_clean: bool = False,
) -> pd.DataFrame:
    if disable_outside_clean or (not cfg.enable_outside_clean):
        return pd.DataFrame()

    candidate_key_set: Set[Tuple[int, str]] = set(
        build_key(r["row_id"], r["column"]) for _, r in clustered_df.iterrows()
    )
    candidate_columns = set(str(c) for c in clustered_df["column"].unique().tolist())

    outside_rows: List[Dict[str, Any]] = []

    for col in df_table.columns:
        if cfg.outside_clean_allowed_columns is not None and col not in cfg.outside_clean_allowed_columns:
            continue
        if cfg.outside_only_from_candidate_columns and col not in candidate_columns:
            continue

        col_series = df_table[col]
        value_counter = col_series.dropna().value_counts().to_dict()
        pool_rows = []

        for row_id in range(len(df_table)):
            key = (int(row_id), str(col))
            if key in candidate_key_set:
                continue

            value = df_table.iloc[row_id][col]
            if cfg.skip_null_in_outside_clean and value is None:
                continue

            value_freq = value_counter.get(value, 0) if value is not None else 0
            value_rank = compute_value_frequency_rank(value_counter, value) if value is not None else None

            if value is not None:
                if value_freq < cfg.outside_min_value_freq:
                    continue
                if value_rank is not None and value_rank > cfg.outside_max_value_freq_rank:
                    continue

            pool_rows.append({
                "row_id": row_id,
                "column": col,
                "value": value,
                "value_frequency": value_freq,
                "value_frequency_rank": value_rank,
                "_outside_priority": outside_clean_priority(col, value_freq, value_rank, cfg),
            })

        if len(pool_rows) == 0:
            continue

        pool_df = pd.DataFrame(pool_rows)
        if cfg.dataset_gate in {"adult", "soccer"}:
            pool_df = pool_df.sort_values(
                by=["_outside_priority", "value_frequency", "value_frequency_rank", "row_id"],
                ascending=[False, False, True, True],
            )
            col_limit = int(cfg.max_outside_clean_per_column.get(col, cfg.default_max_outside_clean_per_column))
            if col_limit <= 0:
                continue
            target_n = min(cfg.outside_clean_per_column, col_limit, len(pool_df))
            if target_n < cfg.outside_clean_min_per_column and len(pool_df) >= cfg.outside_clean_min_per_column:
                target_n = min(cfg.outside_clean_min_per_column, col_limit, len(pool_df))
        else:
            pool_df = pool_df.sort_values(
                by=["value_frequency", "value_frequency_rank", "row_id"],
                ascending=[False, True, True],
            )
            target_n = max(cfg.outside_clean_min_per_column, min(cfg.outside_clean_per_column, len(pool_df)))

        selected_df = pool_df.head(target_n).copy()

        for _, row in selected_df.iterrows():
            outside_rows.append(
                build_outside_clean_row(
                    row_id=int(row["row_id"]),
                    col=str(row["column"]),
                    value=row["value"],
                    value_freq=safe_int(row.get("value_frequency", 0), 0),
                    value_rank=row.get("value_frequency_rank", np.nan),
                    cfg=cfg,
                )
            )

    outside_df = pd.DataFrame(outside_rows)
    if len(outside_df) == 0:
        return outside_df

    outside_df = fill_missing_feature_defaults(outside_df, cfg)

    if cfg.dataset_gate in {"adult", "soccer"} and cfg.max_outside_clean_total is not None and len(outside_df) > cfg.max_outside_clean_total:
        outside_df["_outside_priority"] = outside_df.apply(
            lambda r: outside_clean_priority(
                str(r.get("column", "")),
                safe_int(r.get("value_frequency", 0), 0),
                r.get("value_frequency_rank", np.nan),
                cfg,
            ),
            axis=1,
        )
        outside_df = outside_df.sort_values(
            by=["_outside_priority", "value_frequency", "value_frequency_rank", "row_id"],
            ascending=[False, False, True, True],
        ).head(cfg.max_outside_clean_total)
        outside_df = outside_df.drop(columns=["_outside_priority"], errors="ignore")

    return outside_df.reset_index(drop=True)


# ============================================================
# 6. 平衡与导出
# ============================================================

def balance_final_training(df: pd.DataFrame, cfg: FinalTrainConfig, disable_balance: bool = False) -> pd.DataFrame:
    if disable_balance or (not cfg.enable_balance) or len(df) == 0:
        return df

    error_df = df[df["label_binary"] == 1].copy()
    correct_df = df[df["label_binary"] == 0].copy()

    if len(error_df) == 0 or len(correct_df) == 0:
        return df

    max_correct = int(len(error_df) * cfg.max_correct_to_error_ratio)
    if cfg.min_correct_keep is not None:
        max_correct = max(int(cfg.min_correct_keep), max_correct)

    if len(correct_df) <= max_correct:
        final_df = df.copy()
        if cfg.dataset_gate in {"adult", "soccer"}:
            return final_df.sample(frac=1.0, random_state=cfg.random_state).reset_index(drop=True)
        return final_df

    if cfg.prefer_internal_correct_first:
        internal_correct = correct_df[correct_df["is_outside_candidate"] == 0].copy()
        outside_correct = correct_df[correct_df["is_outside_candidate"] == 1].copy()

        keep_parts = []
        remain = max_correct

        if len(internal_correct) > 0:
            internal_correct = internal_correct.sort_values(
                by=["sample_weight", "propagation_confidence", "conflict_score"],
                ascending=[False, False, False],
            )
            keep_internal = internal_correct.head(remain)
            keep_parts.append(keep_internal)
            remain -= len(keep_internal)

        if remain > 0 and len(outside_correct) > 0:
            outside_correct = outside_correct.sort_values(
                by=["sample_weight", "propagation_confidence", "value_frequency"],
                ascending=[False, False, False],
            )
            keep_outside = outside_correct.head(remain)
            keep_parts.append(keep_outside)

        correct_keep_df = pd.concat(keep_parts, ignore_index=True) if keep_parts else pd.DataFrame(columns=df.columns)
    else:
        correct_keep_df = correct_df.sample(n=max_correct, random_state=cfg.random_state)

    final_df = pd.concat([error_df, correct_keep_df], ignore_index=True)
    final_df = final_df.sample(frac=1.0, random_state=cfg.random_state).reset_index(drop=True)
    return final_df


def export_final_train(df: pd.DataFrame, cfg: FinalTrainConfig, out_csv: str, out_jsonl: str, out_summary: str):
    df = df.copy()
    df = fill_missing_feature_defaults(df, cfg)

    existing_cols = []
    for c in cfg.preferred_cols:
        if c in df.columns and c not in existing_cols:
            existing_cols.append(c)
    other_cols = [c for c in df.columns if c not in existing_cols]
    df = df[existing_cols + other_cols]

    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            obj = {}
            for col in df.columns:
                obj[col] = make_json_safe(row[col])
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    summary = {
        "dataset": cfg.name,
        "total_rows": int(len(df)),
        "error_count": int((df["label_binary"] == 1).sum()) if "label_binary" in df.columns else 0,
        "correct_count": int((df["label_binary"] == 0).sum()) if "label_binary" in df.columns else 0,
        "error_ratio": float((df["label_binary"] == 1).mean()) if "label_binary" in df.columns and len(df) > 0 else None,
        "label_source_counter": df["label_source"].value_counts(dropna=False).to_dict() if "label_source" in df.columns else {},
        "outside_candidate_count": int((df["is_outside_candidate"] == 1).sum()) if "is_outside_candidate" in df.columns else 0,
        "inside_candidate_count": int((df["is_outside_candidate"] == 0).sum()) if "is_outside_candidate" in df.columns else 0,
        "column_counter_top20": df["column"].value_counts(dropna=False).head(20).to_dict() if "column" in df.columns else {},
        "label_by_source": (
            df.groupby(["label_source", "label"]).size().reset_index(name="count").to_dict(orient="records")
            if "label_source" in df.columns and "label" in df.columns else []
        ),
        "config": {
            "input_table_csv": cfg.input_table_csv,
            "input_clustered_csv": cfg.input_clustered_csv,
            "input_labeled_csv": cfg.input_labeled_csv,
            "input_propagated_csv": cfg.input_propagated_csv,
            "cluster_prop_min_conf": cfg.cluster_prop_min_conf,
            "knn_prop_min_conf": cfg.knn_prop_min_conf,
            "enable_outside_clean": cfg.enable_outside_clean,
            "outside_clean_per_column": cfg.outside_clean_per_column,
            "outside_clean_min_per_column": cfg.outside_clean_min_per_column,
            "enable_balance": cfg.enable_balance,
            "max_correct_to_error_ratio": cfg.max_correct_to_error_ratio,
            "min_correct_keep": cfg.min_correct_keep,
            "dataset_gate": cfg.dataset_gate,
            "max_propagated_train_rows": cfg.max_propagated_train_rows,
            "max_outside_clean_total": cfg.max_outside_clean_total,
        },
    }

    if cfg.name == "adult" and "adult_v2_attribution_bucket" in df.columns:
        summary["adult_v2_attribution_bucket_counter"] = df["adult_v2_attribution_bucket"].value_counts(dropna=False).to_dict()
    if cfg.name == "soccer" and "soccer_attribution_bucket" in df.columns:
        summary["soccer_attribution_bucket_counter"] = df["soccer_attribution_bucket"].value_counts(dropna=False).to_dict()

    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=make_json_safe)


# ============================================================
# 7. 主流程
# ============================================================

def run_build_final_train(cfg: FinalTrainConfig, args):
    if args.cluster_prop_min_conf is not None:
        cfg.cluster_prop_min_conf = float(args.cluster_prop_min_conf)
    if args.knn_prop_min_conf is not None:
        cfg.knn_prop_min_conf = float(args.knn_prop_min_conf)
    if args.max_correct_to_error_ratio is not None:
        cfg.max_correct_to_error_ratio = float(args.max_correct_to_error_ratio)
    if args.max_propagated_train_rows is not None:
        cfg.max_propagated_train_rows = int(args.max_propagated_train_rows)
    if args.max_outside_clean_total is not None:
        cfg.max_outside_clean_total = int(args.max_outside_clean_total)

    input_table_csv = args.input_table_csv or cfg.input_table_csv
    input_clustered_csv = args.input_clustered_csv or cfg.input_clustered_csv
    input_labeled_csv = args.input_labeled_csv or cfg.input_labeled_csv
    input_propagated_csv = args.input_propagated_csv or cfg.input_propagated_csv

    output_csv = args.output_csv or cfg.output_final_train_csv
    output_jsonl = args.output_jsonl or cfg.output_final_train_jsonl
    output_summary = args.output_summary_json or cfg.output_summary_json

    print("[1/6] Loading data ...")
    df_table = load_table(input_table_csv, cfg)
    clustered_df = load_clustered_csv(input_clustered_csv, cfg)
    labeled_df = load_labeled_csv(input_labeled_csv, cfg)
    propagated_df = load_propagated_csv(input_propagated_csv, cfg)

    print("[2/6] Building candidate-side training set ...")
    candidate_train_df = build_candidate_training_from_labels(
        clustered_df=clustered_df,
        labeled_df=labeled_df,
        propagated_df=propagated_df,
        cfg=cfg,
        disable_propagated_training=args.disable_propagated_training,
    )
    print(f"  candidate_train_df size = {len(candidate_train_df)}")

    print("[3/6] Building outside-clean training set ...")
    outside_clean_df = build_outside_clean_training_simple(
        df_table=df_table,
        clustered_df=clustered_df,
        candidate_train_df=candidate_train_df,
        cfg=cfg,
        disable_outside_clean=args.disable_outside_clean,
    )
    print(f"  outside_clean_df size = {len(outside_clean_df)}")

    print("[4/6] Merging ...")
    final_df = pd.concat([candidate_train_df, outside_clean_df], ignore_index=True)

    if len(final_df) == 0:
        raise RuntimeError("最终训练集为空，请检查输入文件和阈值设置。")

    final_df = fill_missing_feature_defaults(final_df, cfg)
    final_df = sort_and_dedup_by_priority(final_df, cfg)

    print(f"  merged size before balance = {len(final_df)}")

    print("[5/6] Balancing final train set ...")
    final_df = balance_final_training(final_df, cfg, disable_balance=args.disable_balance)
    print(f"  final size after balance = {len(final_df)}")

    print("[6/6] Exporting ...")
    export_final_train(
        df=final_df,
        cfg=cfg,
        out_csv=output_csv,
        out_jsonl=output_jsonl,
        out_summary=output_summary,
    )

    print("\n===== DONE =====")
    print(f"Final train CSV   : {output_csv}")
    print(f"Final train JSONL : {output_jsonl}")
    print(f"Summary JSON      : {output_summary}")

    print("\n===== STATS =====")
    print(f"Total rows   : {len(final_df)}")
    print(f"Error rows   : {(final_df['label_binary'] == 1).sum()}")
    print(f"Correct rows : {(final_df['label_binary'] == 0).sum()}")

    print("\nLabel source distribution:")
    print(final_df["label_source"].value_counts(dropna=False).to_string())

    print("\nOutside candidate distribution:")
    print(final_df["is_outside_candidate"].value_counts(dropna=False).to_string())

    print("\nTop 20 columns:")
    print(final_df["column"].value_counts(dropna=False).head(20).to_string())


def run_one_dataset(dataset: str, args):
    cfg = CONFIGS[dataset]

    old_cwd = None
    if args.cwd:
        os.makedirs(args.cwd, exist_ok=True)
        old_cwd = os.getcwd()
        os.chdir(args.cwd)

    try:
        run_build_final_train(cfg, args)
    finally:
        if old_cwd is not None:
            os.chdir(old_cwd)


def parse_args():
    parser = argparse.ArgumentParser(description="Unified final training set builder for data-cleaning candidates.")
    parser.add_argument("--dataset", required=True, choices=sorted(list(CONFIGS.keys()) + ["all"]))

    parser.add_argument("--input_table_csv", default=None)
    parser.add_argument("--input_clustered_csv", default=None)
    parser.add_argument("--input_labeled_csv", default=None)
    parser.add_argument("--input_propagated_csv", default=None)

    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--output_jsonl", default=None)
    parser.add_argument("--output_summary_json", default=None)
    parser.add_argument("--cwd", default=None)

    # 消融开关
    parser.add_argument("--disable-outside-clean", action="store_true")
    parser.add_argument("--disable-propagated-training", action="store_true")
    parser.add_argument("--disable-balance", action="store_true")

    # 参数覆盖
    parser.add_argument("--cluster-prop-min-conf", type=float, default=None)
    parser.add_argument("--knn-prop-min-conf", type=float, default=None)
    parser.add_argument("--max-correct-to-error-ratio", type=float, default=None)
    parser.add_argument("--max-propagated-train-rows", type=int, default=None)
    parser.add_argument("--max-outside-clean-total", type=int, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.dataset == "all":
        if any([
            args.input_table_csv,
            args.input_clustered_csv,
            args.input_labeled_csv,
            args.input_propagated_csv,
            args.output_csv,
            args.output_jsonl,
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

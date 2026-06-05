#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unified_train_mlp.py

统一版训练脚本，覆盖：
hospital / flights / beers / rayyan / adult / soccer

它把六份 train_mlp.py 的公共训练逻辑抽成一套：
1. Stage-1 MLP detector
2. Stage-2 calibrated MLP detector
3. FP verifier / keep-error verifier
4. threshold search
5. full / freeze_reuse 两种训练模式
6. 输出模型文件、preprocessor、metrics、config、final_val_predictions

默认输入输出文件名保持原来风格：
- hospital: final_train_dataset_simple_v2.csv -> two_stage_model_output_v4_with_verifier/
- flights : final_train_dataset_flights.csv   -> two_stage_model_output_flights_v4_with_verifier/
- beers   : final_train_dataset_beers.csv     -> two_stage_model_output_beers_v4_with_verifier/
- rayyan  : final_train_dataset_rayyan.csv    -> two_stage_model_output_rayyan_v4_with_verifier/
- adult   : final_train_dataset_adult.csv     -> two_stage_model_output_adult_v4_with_verifier/
- soccer  : final_train_dataset_soccer.csv    -> two_stage_model_output_soccer_v4_with_verifier/

运行示例：
python unified_train_mlp.py --dataset soccer
python unified_train_mlp.py --dataset adult --mode freeze_reuse
python unified_train_mlp.py --dataset all

消融示例：
python unified_train_mlp.py --dataset soccer --disable-verifier
python unified_train_mlp.py --dataset soccer --mode freeze_reuse --source-dir two_stage_model_output_soccer_v4_with_verifier
python unified_train_mlp.py --dataset soccer --output-dir two_stage_model_output_soccer_ablation_no_verifier
"""

import argparse
import json
import os
import random
import shutil
from dataclasses import dataclass, asdict, field, is_dataclass
from typing import Any, Dict, List, Optional, Tuple, Set

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# 1. 基础工具
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def _json_safe(obj):
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def save_json(obj, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(obj), f, ensure_ascii=False, indent=2)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_existing_columns(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def normalize_text_series_for_compare(s: pd.Series) -> pd.Series:
    return s.fillna("__NULL__").astype(str).str.strip()


def safe_num_series(df: pd.DataFrame, col: str, default=0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)
    return pd.Series(np.full(len(df), default), index=df.index)


def safe_str_series(df: pd.DataFrame, col: str, default="missing") -> pd.Series:
    if col in df.columns:
        return df[col].fillna(default).astype(str)
    return pd.Series([default] * len(df), index=df.index)


def as_int01(x, default=0):
    try:
        if pd.isna(x):
            return default
    except Exception:
        pass
    if x is None:
        return default
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y"}:
        return 1
    if s in {"0", "false", "no", "n"}:
        return 0
    try:
        return int(float(x))
    except Exception:
        return default


def copy_detector_artifacts(src_dir: str, dst_dir: str):
    ensure_dir(dst_dir)
    names = [
        "stage1_base_last.pt",
        "stage1_base_best.pt",
        "stage1_base_preprocessor.joblib",
        "stage1_train_log.csv",
        "stage2_cal_last.pt",
        "stage2_cal_best.pt",
        "stage2_cal_preprocessor.joblib",
        "stage2_train_log.csv",
        "two_stage_config.json",
        "metrics_two_stage.json",
        "stage2_train_analysis.csv",
        "final_val_predictions.csv",
    ]
    for name in names:
        src = os.path.join(src_dir, name)
        dst = os.path.join(dst_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, dst)


# ============================================================
# 2. 配置
# ============================================================

BASE_STAGE1_NUMERIC = [
    "violation_count", "conflict_score", "value_frequency", "value_frequency_rank",
    "neighbor_majority_ratio", "prior_error_probability", "posterior_error_probability",
    "is_equal_to_neighbor_majority", "neighbor_agreement_score",
    "strong_rule_count", "candidate_generation_rule_count", "fd_like_count",
    "context_rule_count", "global_rule_count", "rare_value_count", "typo_rule_count",
    "pattern_rule_count", "schema_rule_count", "candidate_rule_ratio", "rule_total_count",
    "strong_rule_ratio", "context_rule_ratio", "fd_like_ratio", "global_rule_ratio",
    "rule_strength_sum", "has_neighbor_signal", "high_neighbor_consistency",
    "is_rare_value", "neighbor_disagree",
]

BASE_STAGE1_CAT = [
    "column", "semantic_type", "main_rule_type", "main_usage_role",
    "pattern_bucket", "rarity_bucket", "neighbor_bucket", "candidate_rule_dominant",
]

BASE_STAGE2_EXTRA_NUMERIC = [
    "raw_pred_error_prob", "raw_pred_label", "raw_margin_to_threshold",
    "stage1_high_conf_error", "stage1_mid_zone",
]

BASE_STAGE2_TAIL_NUMERIC = [
    "rare_only_signal", "is_weak_only", "is_fp_risk_col",
    "high_prob_but_neighbor_support_current",
]

BASE_VERIFIER_PREFIX_NUMERIC = [
    "raw_pred_error_prob", "raw_pred_label", "raw_margin_to_threshold",
    "stage1_high_conf_error", "stage1_mid_zone",
    "detector_pred_error_prob", "detector_pred_label", "detector_margin_to_threshold",
    "is_stage_disagree",
]


@dataclass
class TrainConfig:
    name: str
    input_csv: str
    output_dir: str
    detector_train_mode: str = "full"
    detector_source_dir: Optional[str] = None
    random_state: int = 42
    label_col: str = "label_binary"
    weight_col: str = "sample_weight"

    final_val_size: float = 0.20
    cal_train_size_in_rest: float = 0.25

    save_epoch_checkpoint: bool = True
    save_every_epochs: int = 5
    auto_find_best_threshold: bool = True

    target_min_recall_stage1: float = 0.97
    target_min_recall_stage2: float = 0.95
    threshold_grid_start: float = 0.10
    threshold_grid_end: float = 0.95
    threshold_grid_step: float = 0.02
    precision_priority_alpha: float = 0.18

    verifier_threshold_grid_start: float = 0.10
    verifier_threshold_grid_end: float = 0.95
    verifier_threshold_grid_step: float = 0.02
    verifier_min_final_recall: float = 0.97

    use_pos_weight: bool = True

    stage2_pos_pred_weight_boost: float = 1.0
    stage2_mid_prob_weight_boost: float = 1.0
    stage2_false_pos_risk_cols: Set[str] = field(default_factory=set)
    stage2_risk_column_weight_boost: float = 1.0
    stage2_support_current_weight_boost: float = 1.0
    stage2_rare_only_weight_boost: float = 1.0
    stage2_domain_signal_weight_boost: float = 1.0
    stage2_consistency_weight_boost: float = 1.0

    mid_prob_low: float = 0.35
    mid_prob_high: float = 0.75

    verifier_n_estimators: int = 300
    verifier_max_depth: int = 8
    verifier_min_samples_leaf: int = 10
    verifier_class_weight: str = "balanced_subsample"

    verifier_fp_base_weight_boost: float = 1.25
    verifier_fd_conflict_fp_boost: float = 2.5
    verifier_rare_only_fp_boost: float = 1.45
    verifier_borderline_fp_boost: float = 1.45
    verifier_risk_column_fp_boost: float = 2.0
    verifier_sample_true_error_boost: float = 1.35
    verifier_rescue_candidate_true_error_boost: float = 2.20
    verifier_rescue_candidate_fp_damp: float = 0.70
    verifier_weak_other_fp_boost: float = 2.2
    verifier_weak_other_rare_fd_fp_boost: float = 2.6
    verifier_measure_col_fp_boost: float = 1.8
    verifier_domain_true_error_boost: float = 1.0
    verifier_format_true_error_boost: float = 1.0
    verifier_consistency_true_error_boost: float = 1.0

    stage1_numeric_features: List[str] = field(default_factory=list)
    stage1_categorical_features: List[str] = field(default_factory=list)
    stage2_numeric_features: List[str] = field(default_factory=list)
    stage2_categorical_features: List[str] = field(default_factory=list)
    verifier_numeric_features: List[str] = field(default_factory=list)
    verifier_categorical_features: List[str] = field(default_factory=list)

    stage1_batch_size: int = 64
    stage1_max_epochs: int = 50
    stage1_lr: float = 1e-3
    stage1_weight_decay: float = 1e-4
    stage1_patience: int = 10
    stage1_min_delta: float = 1e-4
    stage1_hidden_dim_1: int = 128
    stage1_hidden_dim_2: int = 64
    stage1_dropout: float = 0.2

    stage2_batch_size: int = 64
    stage2_max_epochs: int = 50
    stage2_lr: float = 8e-4
    stage2_weight_decay: float = 1e-4
    stage2_patience: int = 10
    stage2_min_delta: float = 1e-4
    stage2_hidden_dim_1: int = 64
    stage2_hidden_dim_2: int = 32
    stage2_dropout: float = 0.2

    dataset_kind: str = "generic"  # generic / flights / beers / rayyan / adult / soccer


def with_stage2_and_verifier(stage1_num: List[str], stage1_cat: List[str]) -> Tuple[List[str], List[str], List[str], List[str]]:
    stage2_num = BASE_STAGE2_EXTRA_NUMERIC + stage1_num + BASE_STAGE2_TAIL_NUMERIC
    stage2_cat = list(stage1_cat)
    verifier_num = BASE_VERIFIER_PREFIX_NUMERIC + stage1_num + BASE_STAGE2_TAIL_NUMERIC
    verifier_cat = list(stage1_cat)
    return stage2_num, stage2_cat, verifier_num, verifier_cat


def build_configs() -> Dict[str, TrainConfig]:
    cfgs: Dict[str, TrainConfig] = {}

    # hospital
    s1n = list(BASE_STAGE1_NUMERIC)
    s1c = list(BASE_STAGE1_CAT)
    s2n, s2c, vn, vc = with_stage2_and_verifier(s1n, s1c)
    cfgs["hospital"] = TrainConfig(
        name="hospital",
        input_csv="final_train_dataset_simple_v2.csv",
        output_dir="two_stage_model_output_v4_with_verifier",
        detector_source_dir="two_stage_model_output_v4_with_verifier",
        stage2_false_pos_risk_cols={"EmergencyService", "Sample", "ZipCode", "HospitalName"},
        stage1_numeric_features=s1n,
        stage1_categorical_features=s1c,
        stage2_numeric_features=s2n,
        stage2_categorical_features=s2c,
        verifier_numeric_features=vn,
        verifier_categorical_features=vc,
        dataset_kind="generic",
    )

    # flights
    flight_extra_num = ["time_window_rule_count", "time_value_minutes"]
    flight_extra_cat = ["time_window_bucket", "time_value_bucket"]
    s1n = [
        "violation_count", "conflict_score", "value_frequency", "value_frequency_rank",
        "neighbor_majority_ratio", "prior_error_probability", "posterior_error_probability",
        "is_equal_to_neighbor_majority", "neighbor_agreement_score",
        "strong_rule_count", "candidate_generation_rule_count", "fd_like_count",
        "context_rule_count", "global_rule_count", "rare_value_count", "typo_rule_count",
        "pattern_rule_count", "schema_rule_count",
    ] + flight_extra_num + [
        "candidate_rule_ratio", "rule_total_count", "strong_rule_ratio",
        "context_rule_ratio", "fd_like_ratio", "global_rule_ratio",
        "rule_strength_sum", "has_neighbor_signal", "high_neighbor_consistency",
        "is_rare_value", "neighbor_disagree",
    ]
    s1c = list(BASE_STAGE1_CAT) + flight_extra_cat
    s2n, s2c, vn, vc = with_stage2_and_verifier(s1n, s1c)
    cfgs["flights"] = TrainConfig(
        name="flights",
        input_csv="final_train_dataset_flights.csv",
        output_dir="two_stage_model_output_flights_v4_with_verifier",
        detector_source_dir="two_stage_model_output_flights_v4_with_verifier",
        stage2_false_pos_risk_cols={"act_dep_time", "act_arr_time", "sched_dep_time", "sched_arr_time"},
        stage1_numeric_features=s1n,
        stage1_categorical_features=s1c,
        stage2_numeric_features=s2n,
        stage2_categorical_features=s2c,
        verifier_numeric_features=vn,
        verifier_categorical_features=vc,
        dataset_kind="flights",
    )

    # beers
    beer_extra_num = ["numeric_window_rule_count", "numeric_value"]
    beer_extra_cat = ["numeric_window_bucket", "numeric_value_bucket"]
    s1n = [
        "violation_count", "conflict_score", "value_frequency", "value_frequency_rank",
        "neighbor_majority_ratio", "prior_error_probability", "posterior_error_probability",
        "is_equal_to_neighbor_majority", "neighbor_agreement_score",
        "strong_rule_count", "candidate_generation_rule_count", "fd_like_count",
        "context_rule_count", "global_rule_count", "rare_value_count", "typo_rule_count",
        "pattern_rule_count", "schema_rule_count",
    ] + beer_extra_num + [
        "candidate_rule_ratio", "rule_total_count", "strong_rule_ratio",
        "context_rule_ratio", "fd_like_ratio", "global_rule_ratio",
        "rule_strength_sum", "has_neighbor_signal", "high_neighbor_consistency",
        "is_rare_value", "neighbor_disagree",
    ]
    s1c = list(BASE_STAGE1_CAT) + beer_extra_cat
    s2n, s2c, vn, vc = with_stage2_and_verifier(s1n, s1c)
    cfgs["beers"] = TrainConfig(
        name="beers",
        input_csv="final_train_dataset_beers.csv",
        output_dir="two_stage_model_output_beers_v4_with_verifier",
        detector_source_dir="two_stage_model_output_beers_v4_with_verifier",
        stage2_false_pos_risk_cols={"abv", "ibu", "ounces", "state", "style"},
        stage1_numeric_features=s1n,
        stage1_categorical_features=s1c,
        stage2_numeric_features=s2n,
        stage2_categorical_features=s2c,
        verifier_numeric_features=vn,
        verifier_categorical_features=vc,
        dataset_kind="beers",
    )

    # rayyan
    ray_extra_num = ["canonical_rule_count", "numeric_value"]
    ray_extra_cat = ["canonical_bucket", "date_value_bucket", "language_bucket"]
    s1n = [
        "violation_count", "conflict_score", "value_frequency", "value_frequency_rank",
        "neighbor_majority_ratio", "prior_error_probability", "posterior_error_probability",
        "is_equal_to_neighbor_majority", "neighbor_agreement_score",
        "strong_rule_count", "candidate_generation_rule_count", "fd_like_count",
        "context_rule_count", "global_rule_count", "rare_value_count", "typo_rule_count",
        "pattern_rule_count", "schema_rule_count",
    ] + ray_extra_num + [
        "candidate_rule_ratio", "rule_total_count", "strong_rule_ratio",
        "context_rule_ratio", "fd_like_ratio", "global_rule_ratio",
        "rule_strength_sum", "has_neighbor_signal", "high_neighbor_consistency",
        "is_rare_value", "neighbor_disagree",
    ]
    s1c = list(BASE_STAGE1_CAT) + ray_extra_cat
    s2n, s2c, vn, vc = with_stage2_and_verifier(s1n, s1c)
    cfgs["rayyan"] = TrainConfig(
        name="rayyan",
        input_csv="final_train_dataset_rayyan.csv",
        output_dir="two_stage_model_output_rayyan_v4_with_verifier",
        detector_source_dir="two_stage_model_output_rayyan_v4_with_verifier",
        stage2_false_pos_risk_cols={"journal_title", "jounral_abbreviation", "journal_issn", "article_language", "article_jcreated_at", "author_list"},
        verifier_sample_true_error_boost=1.20,
        verifier_measure_col_fp_boost=1.40,
        stage1_numeric_features=s1n,
        stage1_categorical_features=s1c,
        stage2_numeric_features=s2n,
        stage2_categorical_features=s2c,
        verifier_numeric_features=vn,
        verifier_categorical_features=vc,
        dataset_kind="rayyan",
    )

    # adult
    adult_v2_numeric = [
        "target_is_primary_column", "target_is_context_only_column",
        "is_relationship_spouse_value", "relationship_marital_conflict",
        "relationship_age_conflict", "relationship_sex_conflict",
        "sex_value_invalid", "education_age_extreme_conflict",
        "has_relationship_v2_rule", "has_education_v2_rule",
        "relationship_v2_conflict_count", "has_relationship_v2_signal",
        "has_education_v2_signal", "is_context_only_weak_signal",
        "adult_v2_signal_strength", "adult_v2_primary_target_signal",
    ]
    adult_v2_cat = ["adult_v2_attribution_bucket"]
    s1n = [
        "violation_count", "conflict_score", "value_frequency", "value_frequency_rank",
        "neighbor_majority_ratio", "prior_error_probability", "posterior_error_probability",
        "is_equal_to_neighbor_majority", "neighbor_agreement_score",
        "strong_rule_count", "candidate_generation_rule_count", "fd_like_count",
        "context_rule_count", "global_rule_count", "rare_value_count", "typo_rule_count",
        "pattern_rule_count", "schema_rule_count",
        "adult_domain_rule_count", "adult_format_rule_count", "adult_consistency_rule_count",
        "age_lower", "age_upper", "hours_lower", "hours_upper", "education_order",
        "adult_profile_inconsistency_count", "adult_consistency_score",
        "time_window_rule_count", "time_value_minutes",
        "candidate_rule_ratio", "rule_total_count",
        "strong_rule_ratio", "context_rule_ratio", "fd_like_ratio", "global_rule_ratio",
        "adult_rule_total_count", "adult_domain_rule_ratio", "adult_format_rule_ratio",
        "adult_consistency_rule_ratio",
        "rule_strength_sum", "has_neighbor_signal", "high_neighbor_consistency",
        "is_rare_value", "neighbor_disagree",
        "domain_invalid_flag", "adult_consistency_flag", "high_adult_consistency_score",
        "has_adult_domain_signal", "has_adult_format_signal", "has_adult_consistency_signal",
        "age_span", "hours_span", "is_minor_age_bucket", "is_high_hours_bucket",
    ] + adult_v2_numeric
    s1c = [
        "column", "semantic_type", "detected_type", "main_rule_type", "main_usage_role",
        "pattern_bucket", "rarity_bucket", "neighbor_bucket", "domain_bucket",
        "adult_consistency_bucket", "age_sampling_bucket", "hours_sampling_bucket",
        "adult_value_bucket", "candidate_rule_dominant",
        "time_window_bucket", "time_value_bucket",
    ] + adult_v2_cat
    s2n, s2c, vn, vc = with_stage2_and_verifier(s1n, s1c)
    cfgs["adult"] = TrainConfig(
        name="adult",
        input_csv="final_train_dataset_adult.csv",
        output_dir="two_stage_model_output_adult_v4_with_verifier",
        detector_source_dir="two_stage_model_output_adult_v4_with_verifier",
        target_min_recall_stage1=0.90,
        target_min_recall_stage2=0.88,
        verifier_min_final_recall=0.88,
        precision_priority_alpha=0.22,
        stage2_pos_pred_weight_boost=1.05,
        stage2_mid_prob_weight_boost=1.20,
        stage2_false_pos_risk_cols={"country", "occupation", "workclass", "education"},
        stage2_risk_column_weight_boost=1.15,
        stage2_support_current_weight_boost=1.10,
        stage2_rare_only_weight_boost=1.20,
        stage2_domain_signal_weight_boost=1.35,
        stage2_consistency_weight_boost=1.30,
        verifier_n_estimators=400,
        verifier_max_depth=10,
        verifier_min_samples_leaf=6,
        verifier_fd_conflict_fp_boost=1.80,
        verifier_rare_only_fp_boost=1.80,
        verifier_risk_column_fp_boost=2.00,
        verifier_rescue_candidate_true_error_boost=1.80,
        verifier_domain_true_error_boost=2.20,
        verifier_format_true_error_boost=2.00,
        verifier_consistency_true_error_boost=1.80,
        stage1_max_epochs=60,
        stage1_patience=12,
        stage1_hidden_dim_1=160,
        stage1_hidden_dim_2=80,
        stage2_max_epochs=60,
        stage2_patience=12,
        stage2_hidden_dim_1=96,
        stage2_hidden_dim_2=48,
        stage1_numeric_features=s1n,
        stage1_categorical_features=s1c,
        stage2_numeric_features=s2n,
        stage2_categorical_features=s2c,
        verifier_numeric_features=vn,
        verifier_categorical_features=vc,
        dataset_kind="adult",
    )

    # soccer
    soccer_numeric = [
        "target_is_primary_column", "target_is_context_only_column",
        "birthyear_value_invalid", "season_value_invalid",
        "position_value_invalid", "position_needs_canonicalization",
        "birthyear_season_age_conflict", "team_context_conflict",
        "format_rule_signal", "fd_like_signal",
        "position_invalid_or_normalization",
        "has_birthyear_or_season_signal", "has_team_context_signal",
        "has_fd_like_signal", "is_context_only_weak_signal",
        "soccer_signal_strength", "soccer_primary_target_signal",
    ]
    soccer_cat = ["soccer_attribution_bucket"]
    s1n = [
        "violation_count", "conflict_score", "value_frequency", "value_frequency_rank",
        "neighbor_majority_ratio", "prior_error_probability", "posterior_error_probability",
        "is_equal_to_neighbor_majority", "neighbor_agreement_score",
        "strong_rule_count", "candidate_generation_rule_count", "fd_like_count", "evidence_only_fd_count",
        "context_rule_count", "global_rule_count", "rare_value_count", "typo_rule_count",
        "pattern_rule_count", "schema_rule_count",
        "soccer_domain_rule_count", "soccer_format_rule_count", "soccer_numeric_rule_count",
        "soccer_consistency_rule_count", "soccer_profile_inconsistency_count", "soccer_consistency_score",
        "year_value", "row_birthyear_int", "row_season_int", "row_age_at_season",
        "birthyear_season_gap", "is_young_player_age", "is_old_player_age",
        "time_window_rule_count", "time_value_minutes",
        "candidate_rule_ratio", "rule_total_count",
        "strong_rule_ratio", "context_rule_ratio", "fd_like_ratio", "global_rule_ratio",
        "soccer_rule_total_count", "soccer_domain_rule_ratio", "soccer_format_rule_ratio",
        "soccer_numeric_rule_ratio", "soccer_consistency_rule_ratio",
        "rule_strength_sum", "has_neighbor_signal", "high_neighbor_consistency",
        "is_rare_value", "neighbor_disagree",
        "domain_invalid_flag", "soccer_consistency_flag", "high_soccer_consistency_score",
        "has_soccer_domain_signal", "has_soccer_format_signal", "has_soccer_numeric_signal",
        "has_soccer_consistency_signal",
        "adult_domain_rule_count", "adult_format_rule_count", "adult_consistency_rule_count",
        "adult_rule_total_count", "adult_domain_rule_ratio", "adult_format_rule_ratio",
        "adult_consistency_rule_ratio",
    ] + soccer_numeric
    s1c = [
        "column", "semantic_type", "detected_type", "main_rule_type", "main_usage_role",
        "pattern_bucket", "rarity_bucket", "neighbor_bucket", "domain_bucket",
        "soccer_consistency_bucket", "soccer_value_bucket", "canonical_position",
        "row_position_canonical", "candidate_rule_dominant",
        "time_window_bucket", "time_value_bucket",
        "adult_consistency_bucket", "age_sampling_bucket", "hours_sampling_bucket",
        "adult_value_bucket", "adult_v2_attribution_bucket",
    ] + soccer_cat
    s2n, s2c, vn, vc = with_stage2_and_verifier(s1n, s1c)
    cfgs["soccer"] = TrainConfig(
        name="soccer",
        input_csv="final_train_dataset_soccer.csv",
        output_dir="two_stage_model_output_soccer_v4_with_verifier",
        detector_source_dir="two_stage_model_output_soccer_v4_with_verifier",
        target_min_recall_stage1=0.90,
        target_min_recall_stage2=0.88,
        verifier_min_final_recall=0.88,
        precision_priority_alpha=0.22,
        stage2_pos_pred_weight_boost=1.05,
        stage2_mid_prob_weight_boost=1.20,
        stage2_false_pos_risk_cols={"name", "surname", "team", "city", "stadium", "manager", "birthplace"},
        stage2_risk_column_weight_boost=1.05,
        stage2_support_current_weight_boost=1.10,
        stage2_rare_only_weight_boost=1.20,
        stage2_domain_signal_weight_boost=1.35,
        stage2_consistency_weight_boost=1.30,
        verifier_n_estimators=400,
        verifier_max_depth=10,
        verifier_min_samples_leaf=6,
        verifier_fd_conflict_fp_boost=1.80,
        verifier_rare_only_fp_boost=1.80,
        verifier_risk_column_fp_boost=1.35,
        verifier_rescue_candidate_true_error_boost=1.80,
        verifier_domain_true_error_boost=2.20,
        verifier_format_true_error_boost=2.00,
        verifier_consistency_true_error_boost=1.80,
        stage1_max_epochs=60,
        stage1_patience=12,
        stage1_hidden_dim_1=160,
        stage1_hidden_dim_2=80,
        stage2_max_epochs=60,
        stage2_patience=12,
        stage2_hidden_dim_1=96,
        stage2_hidden_dim_2=48,
        stage1_numeric_features=s1n,
        stage1_categorical_features=s1c,
        stage2_numeric_features=s2n,
        stage2_categorical_features=s2c,
        verifier_numeric_features=vn,
        verifier_categorical_features=vc,
        dataset_kind="soccer",
    )

    return cfgs


CONFIGS = build_configs()


# ============================================================
# 3. 模型结构
# ============================================================

@dataclass
class StageConfig:
    input_dim: int
    hidden_dim_1: int
    hidden_dim_2: int
    dropout: float
    batch_size: int
    learning_rate: float
    weight_decay: float
    max_epochs: int
    patience: int
    numeric_features: List[str]
    categorical_features: List[str]
    feature_cols: List[str]


@dataclass
class VerifierConfig:
    numeric_features: List[str]
    categorical_features: List[str]
    feature_cols: List[str]
    threshold: float
    metrics: Dict[str, Any]


@dataclass
class TwoStageConfig:
    device: str
    random_state: int
    label_col: str
    weight_col: str
    final_val_size: float
    cal_train_size_in_rest: float
    stage1: StageConfig
    stage2: StageConfig
    verifier: VerifierConfig


class TabularDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, w: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.w = torch.tensor(w, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {"X": self.X[idx], "y": self.y[idx], "w": self.w[idx]}


class MLPDetector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim_1: int, hidden_dim_2: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


# ============================================================
# 4. 派生特征
# ============================================================

ADULT_PRIMARY_TARGET_COLUMNS = {"sex", "relationship", "education"}
ADULT_CONTEXT_ONLY_COLUMNS = {"age", "workclass", "maritalstatus", "occupation", "race", "hoursperweek", "country", "income"}
RELATIONSHIP_V2_RULE_TYPES = {
    "relationship_marital_spouse_conflict",
    "relationship_age_spouse_conflict",
    "relationship_sex_spouse_conflict",
    "relationship_context_dominant",
}
EDUCATION_V2_RULE_TYPES = {"education_age_extreme_conflict"}

SOCCER_COLUMNS = {"name", "surname", "birthyear", "birthplace", "position", "team", "city", "stadium", "season", "manager"}
SOCCER_PRIMARY_TARGET_COLUMNS = set(SOCCER_COLUMNS)
SOCCER_CONTEXT_ONLY_COLUMNS = set()
SOCCER_STRONG_RULE_TYPES = {"birthyear_numeric_range", "season_numeric_range", "position_domain", "birthyear_season_age_consistency"}
SOCCER_FORMAT_RULE_TYPES = {
    "name_pattern", "surname_pattern", "manager_name_pattern", "team_name_pattern",
    "birthplace_location_pattern", "city_location_pattern", "stadium_name_pattern",
    "pattern", "generic_regex",
}
SOCCER_CONTEXT_RULE_TYPES = {
    "team_context_dominant", "team_season_manager_consistency_hint",
    "city_stadium_consistency_hint", "dominant_value_by_context",
    "dominant_value_by_context_pair",
}
SOCCER_FD_RULE_TYPES = {"functional_dependency"}
SOCCER_EVIDENCE_ONLY_FD_RULE_TYPES = {"soft_functional_dependency"}


def add_adult_v2_features(out: pd.DataFrame) -> pd.DataFrame:
    col_series = safe_str_series(out, "column", "missing")
    rule_series = safe_str_series(out, "main_rule_type", "missing")

    for c in [
        "target_is_primary_column", "target_is_context_only_column",
        "is_relationship_spouse_value", "relationship_marital_conflict",
        "relationship_age_conflict", "relationship_sex_conflict",
        "sex_value_invalid", "education_age_extreme_conflict",
        "has_relationship_v2_rule", "has_education_v2_rule",
    ]:
        out[c] = safe_num_series(out, c, 0).astype(int)

    out["target_is_primary_column"] = np.where(col_series.isin(ADULT_PRIMARY_TARGET_COLUMNS), 1, out["target_is_primary_column"]).astype(int)
    out["target_is_context_only_column"] = np.where(col_series.isin(ADULT_CONTEXT_ONLY_COLUMNS), 1, out["target_is_context_only_column"]).astype(int)
    out["has_relationship_v2_rule"] = np.where(rule_series.isin(RELATIONSHIP_V2_RULE_TYPES), 1, out["has_relationship_v2_rule"]).astype(int)
    out["has_education_v2_rule"] = np.where(rule_series.isin(EDUCATION_V2_RULE_TYPES), 1, out["has_education_v2_rule"]).astype(int)

    out["relationship_v2_conflict_count"] = (
        out["relationship_marital_conflict"].astype(int)
        + out["relationship_age_conflict"].astype(int)
        + out["relationship_sex_conflict"].astype(int)
        + out["has_relationship_v2_rule"].astype(int)
    )
    out["has_relationship_v2_signal"] = (out["relationship_v2_conflict_count"] > 0).astype(int)
    out["has_education_v2_signal"] = (
        (out["education_age_extreme_conflict"].astype(int) > 0)
        | (out["has_education_v2_rule"].astype(int) > 0)
    ).astype(int)

    out["is_context_only_weak_signal"] = (
        (out["target_is_context_only_column"].astype(int) == 1)
        & (safe_num_series(out, "adult_domain_rule_count", 0).astype(float) <= 0)
        & (safe_num_series(out, "adult_format_rule_count", 0).astype(float) <= 0)
        & (safe_num_series(out, "adult_consistency_rule_count", 0).astype(float) <= 0)
        & (safe_num_series(out, "schema_rule_count", 0).astype(float) <= 0)
        & (safe_num_series(out, "fd_like_count", 0).astype(float) <= 0)
        & (safe_num_series(out, "context_rule_count", 0).astype(float) <= 0)
    ).astype(int)

    out["adult_v2_signal_strength"] = (
        2.5 * out["has_relationship_v2_signal"].astype(float)
        + 2.0 * out["sex_value_invalid"].astype(float)
        + 1.5 * out["has_education_v2_signal"].astype(float)
        + 1.2 * safe_num_series(out, "has_adult_domain_signal", 0).astype(float)
        + 1.0 * safe_num_series(out, "has_adult_format_signal", 0).astype(float)
        + 1.0 * safe_num_series(out, "has_adult_consistency_signal", 0).astype(float)
        + 0.5 * out["target_is_primary_column"].astype(float)
        - 1.0 * out["is_context_only_weak_signal"].astype(float)
    )
    out["adult_v2_primary_target_signal"] = (
        (out["target_is_primary_column"].astype(int) == 1)
        & (
            (out["has_relationship_v2_signal"].astype(int) == 1)
            | (out["sex_value_invalid"].astype(int) == 1)
            | (out["has_education_v2_signal"].astype(int) == 1)
            | (safe_num_series(out, "has_adult_domain_signal", 0).astype(int) == 1)
            | (safe_num_series(out, "has_adult_format_signal", 0).astype(int) == 1)
            | (safe_num_series(out, "has_adult_consistency_signal", 0).astype(int) == 1)
        )
    ).astype(int)

    if "adult_v2_attribution_bucket" not in out.columns:
        out["adult_v2_attribution_bucket"] = "unknown_attribution"
    out["adult_v2_attribution_bucket"] = out["adult_v2_attribution_bucket"].fillna("unknown_attribution").astype(str)
    return out


def add_soccer_features(out: pd.DataFrame) -> pd.DataFrame:
    col_series = safe_str_series(out, "column", "missing")
    rule_series = safe_str_series(out, "main_rule_type", "missing")

    for c in [
        "target_is_primary_column", "target_is_context_only_column",
        "birthyear_value_invalid", "season_value_invalid", "position_value_invalid",
        "position_needs_canonicalization", "birthyear_season_age_conflict",
        "team_context_conflict", "format_rule_signal", "fd_like_signal",
    ]:
        out[c] = safe_num_series(out, c, 0).astype(int)

    out["target_is_primary_column"] = np.where(col_series.isin(SOCCER_PRIMARY_TARGET_COLUMNS), 1, out["target_is_primary_column"]).astype(int)
    out["target_is_context_only_column"] = np.where(col_series.isin(SOCCER_CONTEXT_ONLY_COLUMNS), 1, 0).astype(int)

    out["birthyear_value_invalid"] = np.where(rule_series.eq("birthyear_numeric_range"), 1, out["birthyear_value_invalid"]).astype(int)
    out["season_value_invalid"] = np.where(rule_series.eq("season_numeric_range"), 1, out["season_value_invalid"]).astype(int)
    out["position_value_invalid"] = np.where(rule_series.eq("position_domain"), 1, out["position_value_invalid"]).astype(int)
    out["birthyear_season_age_conflict"] = np.where(rule_series.eq("birthyear_season_age_consistency"), 1, out["birthyear_season_age_conflict"]).astype(int)
    out["team_context_conflict"] = np.where(rule_series.isin(SOCCER_CONTEXT_RULE_TYPES), 1, out["team_context_conflict"]).astype(int)
    out["format_rule_signal"] = np.where(rule_series.isin(SOCCER_FORMAT_RULE_TYPES), 1, out["format_rule_signal"]).astype(int)
    out["fd_like_signal"] = np.where(rule_series.isin(SOCCER_FD_RULE_TYPES), 1, out["fd_like_signal"]).astype(int)

    if "evidence_only_fd_count" not in out.columns:
        out["evidence_only_fd_count"] = 0
    out["evidence_only_fd_count"] = safe_num_series(out, "evidence_only_fd_count", 0).astype(float)
    soft_fd_mask = rule_series.isin(SOCCER_EVIDENCE_ONLY_FD_RULE_TYPES)
    out.loc[soft_fd_mask, "fd_like_signal"] = 0
    out.loc[soft_fd_mask, "evidence_only_fd_count"] = np.maximum(out.loc[soft_fd_mask, "evidence_only_fd_count"].astype(float), 1.0)

    out["position_invalid_or_normalization"] = (
        (out["position_value_invalid"].astype(int) > 0)
        | (out["position_needs_canonicalization"].astype(int) > 0)
    ).astype(int)

    out["has_birthyear_or_season_signal"] = (
        (out["birthyear_value_invalid"].astype(int) > 0)
        | (out["season_value_invalid"].astype(int) > 0)
        | (out["birthyear_season_age_conflict"].astype(int) > 0)
        | rule_series.isin({"birthyear_numeric_range", "season_numeric_range", "birthyear_season_age_consistency"})
    ).astype(int)

    out["has_team_context_signal"] = (
        (out["team_context_conflict"].astype(int) > 0)
        | rule_series.isin(SOCCER_CONTEXT_RULE_TYPES)
    ).astype(int)

    out["has_fd_like_signal"] = (
        (out["fd_like_signal"].astype(int) > 0)
        | (safe_num_series(out, "fd_like_count", 0.0) > 0)
        | rule_series.isin(SOCCER_FD_RULE_TYPES)
    ).astype(int)

    out["is_context_only_weak_signal"] = (
        (out["target_is_context_only_column"].astype(int) == 1)
        & (safe_num_series(out, "soccer_domain_rule_count", 0.0) <= 0)
        & (safe_num_series(out, "soccer_format_rule_count", 0.0) <= 0)
        & (safe_num_series(out, "soccer_numeric_rule_count", 0.0) <= 0)
        & (safe_num_series(out, "soccer_consistency_rule_count", 0.0) <= 0)
        & (~rule_series.isin(SOCCER_STRONG_RULE_TYPES | SOCCER_FORMAT_RULE_TYPES))
    ).astype(int)

    out["soccer_signal_strength"] = (
        2.5 * out["has_birthyear_or_season_signal"].astype(float)
        + 2.2 * out["position_invalid_or_normalization"].astype(float)
        + 1.5 * safe_num_series(out, "has_soccer_format_signal", 0.0)
        + 1.2 * safe_num_series(out, "has_soccer_consistency_signal", 0.0)
        + 0.8 * out["target_is_primary_column"].astype(float)
        + 0.5 * out["has_team_context_signal"].astype(float)
        + 0.4 * out["has_fd_like_signal"].astype(float)
        - 1.2 * out["is_context_only_weak_signal"].astype(float)
    )

    out["soccer_primary_target_signal"] = (
        (out["target_is_primary_column"].astype(int) == 1)
        & (
            (out["has_birthyear_or_season_signal"].astype(int) == 1)
            | (out["position_invalid_or_normalization"].astype(int) == 1)
            | (safe_num_series(out, "has_soccer_format_signal", 0).astype(int) == 1)
            | (safe_num_series(out, "has_soccer_domain_signal", 0).astype(int) == 1)
            | (safe_num_series(out, "has_soccer_numeric_signal", 0).astype(int) == 1)
            | (safe_num_series(out, "has_soccer_consistency_signal", 0).astype(int) == 1)
        )
    ).astype(int)

    if "soccer_attribution_bucket" not in out.columns:
        out["soccer_attribution_bucket"] = "unknown_attribution"
    out["soccer_attribution_bucket"] = out["soccer_attribution_bucket"].fillna("unknown_attribution").astype(str)
    return out


def add_derived_features(df: pd.DataFrame, cfg: TrainConfig) -> pd.DataFrame:
    out = df.copy()

    if "value" in out.columns and "neighbor_majority_value" in out.columns:
        value_series = normalize_text_series_for_compare(out["value"])
        neighbor_series = normalize_text_series_for_compare(out["neighbor_majority_value"])
        out["is_equal_to_neighbor_majority"] = (value_series == neighbor_series).astype(int)
    else:
        out["is_equal_to_neighbor_majority"] = 0

    ratio = safe_num_series(out, "neighbor_majority_ratio", 0.0)
    out["neighbor_majority_ratio"] = ratio
    out["neighbor_agreement_score"] = np.where(out["is_equal_to_neighbor_majority"] == 1, ratio, -ratio).astype(float)
    out["has_neighbor_signal"] = (ratio > 0).astype(int)
    out["high_neighbor_consistency"] = (ratio >= 0.8).astype(int)
    out["neighbor_disagree"] = ((out["has_neighbor_signal"] == 1) & (out["is_equal_to_neighbor_majority"] == 0)).astype(int)

    for col in [
        "strong_rule_count", "candidate_generation_rule_count", "fd_like_count", "context_rule_count",
        "global_rule_count", "rare_value_count", "typo_rule_count", "pattern_rule_count", "schema_rule_count",
    ]:
        out[col] = safe_num_series(out, col, 0.0)

    out["rule_total_count"] = out["strong_rule_count"] + out["candidate_generation_rule_count"]
    out["candidate_rule_ratio"] = np.where(out["rule_total_count"] > 0, out["candidate_generation_rule_count"] / (out["rule_total_count"] + 1e-8), 0.0)
    out["strong_rule_ratio"] = np.where(out["rule_total_count"] > 0, out["strong_rule_count"] / (out["rule_total_count"] + 1e-8), 0.0)
    out["context_rule_ratio"] = np.where(out["rule_total_count"] > 0, out["context_rule_count"] / (out["rule_total_count"] + 1e-8), 0.0)
    out["fd_like_ratio"] = np.where(out["rule_total_count"] > 0, out["fd_like_count"] / (out["rule_total_count"] + 1e-8), 0.0)
    out["global_rule_ratio"] = np.where(out["rule_total_count"] > 0, out["global_rule_count"] / (out["rule_total_count"] + 1e-8), 0.0)

    out["rule_strength_sum"] = (
        1.00 * out["strong_rule_count"]
        + 0.25 * out["candidate_generation_rule_count"]
        + 0.40 * out["fd_like_count"]
        + 0.35 * out["context_rule_count"]
        + 0.15 * out["global_rule_count"]
        + 0.20 * out["rare_value_count"]
        + 0.35 * out["typo_rule_count"]
        + 0.45 * out["pattern_rule_count"]
        + 0.45 * out["schema_rule_count"]
    )
    if cfg.dataset_kind == "flights":
        out["time_window_rule_count"] = safe_num_series(out, "time_window_rule_count", 0.0)
        out["time_value_minutes"] = safe_num_series(out, "time_value_minutes", 0.0)
        out["rule_strength_sum"] += 0.50 * out["time_window_rule_count"]

    if cfg.dataset_kind == "beers":
        out["numeric_window_rule_count"] = safe_num_series(out, "numeric_window_rule_count", 0.0)
        out["numeric_value"] = safe_num_series(out, "numeric_value", 0.0)
        out["rule_strength_sum"] += 0.50 * out["numeric_window_rule_count"]

    if cfg.dataset_kind == "rayyan":
        out["canonical_rule_count"] = safe_num_series(out, "canonical_rule_count", 0.0)
        out["numeric_value"] = safe_num_series(out, "numeric_value", 0.0)
        out["rule_strength_sum"] += 0.50 * out["canonical_rule_count"]

    value_freq = safe_num_series(out, "value_frequency", 999999)
    value_rank = safe_num_series(out, "value_frequency_rank", 999999)
    long_tail_cols = {"name", "surname", "team", "city", "stadium", "manager", "birthplace", "country", "occupation", "workclass"}
    col_series = safe_str_series(out, "column", "missing")
    out["is_rare_value"] = np.where(
        col_series.isin(long_tail_cols),
        ((value_freq <= 1) | (value_rank > 25)).astype(int),
        ((value_freq <= 5) | (value_rank > 10)).astype(int),
    )

    out["candidate_rule_dominant"] = np.where(out["candidate_generation_rule_count"] > out["strong_rule_count"], "yes", "no")

    out["rare_only_signal"] = (
        (out["rare_value_count"] > 0)
        & (out["strong_rule_count"] <= 0)
        & (out["fd_like_count"] <= 0)
        & (out["context_rule_count"] <= 0)
        & (out["typo_rule_count"] <= 0)
        & (out["schema_rule_count"] <= 0)
    ).astype(int)

    out["is_weak_only"] = (
        (out["rare_only_signal"] == 1)
        | (
            (out["strong_rule_count"] <= 0)
            & (out["fd_like_count"] <= 0)
            & (out["typo_rule_count"] <= 0)
            & (out["schema_rule_count"] <= 0)
            & (out["candidate_generation_rule_count"] > 0)
        )
    ).astype(int)

    out["is_fp_risk_col"] = col_series.isin(cfg.stage2_false_pos_risk_cols).astype(int)
    out["high_prob_but_neighbor_support_current"] = (
        (out["high_neighbor_consistency"] == 1)
        & (out["is_equal_to_neighbor_majority"] == 1)
    ).astype(int)

    if cfg.dataset_kind == "adult":
        for col in [
            "adult_domain_rule_count", "adult_format_rule_count", "adult_consistency_rule_count",
            "age_lower", "age_upper", "hours_lower", "hours_upper", "education_order",
            "adult_profile_inconsistency_count", "adult_consistency_score",
            "time_window_rule_count", "time_value_minutes",
        ]:
            out[col] = safe_num_series(out, col, 0.0)
        out["adult_rule_total_count"] = out["adult_domain_rule_count"] + out["adult_format_rule_count"] + out["adult_consistency_rule_count"]
        out["adult_domain_rule_ratio"] = np.where(out["adult_rule_total_count"] > 0, out["adult_domain_rule_count"] / (out["adult_rule_total_count"] + 1e-8), 0.0)
        out["adult_format_rule_ratio"] = np.where(out["adult_rule_total_count"] > 0, out["adult_format_rule_count"] / (out["adult_rule_total_count"] + 1e-8), 0.0)
        out["adult_consistency_rule_ratio"] = np.where(out["adult_rule_total_count"] > 0, out["adult_consistency_rule_count"] / (out["adult_rule_total_count"] + 1e-8), 0.0)
        out["domain_invalid_flag"] = safe_str_series(out, "domain_bucket", "missing").eq("domain_invalid").astype(int)
        out["adult_consistency_flag"] = safe_str_series(out, "adult_consistency_bucket", "missing").isin([
            "strong_adult_consistency_signal", "adult_consistency_flag", "weak_adult_consistency_signal"
        ]).astype(int)
        out["high_adult_consistency_score"] = (out["adult_consistency_score"] >= 1.0).astype(int)
        out["has_adult_domain_signal"] = (out["adult_domain_rule_count"] > 0).astype(int)
        out["has_adult_format_signal"] = (out["adult_format_rule_count"] > 0).astype(int)
        out["has_adult_consistency_signal"] = (out["adult_consistency_rule_count"] > 0).astype(int)
        out["age_span"] = np.maximum(0.0, out["age_upper"] - out["age_lower"])
        out["hours_span"] = np.maximum(0.0, out["hours_upper"] - out["hours_lower"])
        out["is_minor_age_bucket"] = ((out["age_upper"] > 0) & (out["age_upper"] < 18)).astype(int)
        out["is_high_hours_bucket"] = (out["hours_upper"] > 60).astype(int)
        out["rule_strength_sum"] += (
            1.5 * out["adult_domain_rule_count"]
            + 1.2 * out["adult_format_rule_count"]
            + 1.2 * out["adult_consistency_rule_count"]
        )
        out = add_adult_v2_features(out)

    if cfg.dataset_kind == "soccer":
        for col in [
            "evidence_only_fd_count", "soccer_domain_rule_count", "soccer_format_rule_count",
            "soccer_numeric_rule_count", "soccer_consistency_rule_count",
            "soccer_profile_inconsistency_count", "soccer_consistency_score",
            "year_value", "row_birthyear_int", "row_season_int", "row_age_at_season",
            "time_window_rule_count", "time_value_minutes",
            "adult_domain_rule_count", "adult_format_rule_count", "adult_consistency_rule_count",
        ]:
            out[col] = safe_num_series(out, col, 0.0)

        out["soccer_rule_total_count"] = out["soccer_domain_rule_count"] + out["soccer_format_rule_count"] + out["soccer_numeric_rule_count"] + out["soccer_consistency_rule_count"]
        out["soccer_domain_rule_ratio"] = np.where(out["soccer_rule_total_count"] > 0, out["soccer_domain_rule_count"] / (out["soccer_rule_total_count"] + 1e-8), 0.0)
        out["soccer_format_rule_ratio"] = np.where(out["soccer_rule_total_count"] > 0, out["soccer_format_rule_count"] / (out["soccer_rule_total_count"] + 1e-8), 0.0)
        out["soccer_numeric_rule_ratio"] = np.where(out["soccer_rule_total_count"] > 0, out["soccer_numeric_rule_count"] / (out["soccer_rule_total_count"] + 1e-8), 0.0)
        out["soccer_consistency_rule_ratio"] = np.where(out["soccer_rule_total_count"] > 0, out["soccer_consistency_rule_count"] / (out["soccer_rule_total_count"] + 1e-8), 0.0)

        out["adult_rule_total_count"] = out["adult_domain_rule_count"] + out["adult_format_rule_count"] + out["adult_consistency_rule_count"]
        out["adult_domain_rule_ratio"] = np.where(out["adult_rule_total_count"] > 0, out["adult_domain_rule_count"] / (out["adult_rule_total_count"] + 1e-8), 0.0)
        out["adult_format_rule_ratio"] = np.where(out["adult_rule_total_count"] > 0, out["adult_format_rule_count"] / (out["adult_rule_total_count"] + 1e-8), 0.0)
        out["adult_consistency_rule_ratio"] = np.where(out["adult_rule_total_count"] > 0, out["adult_consistency_rule_count"] / (out["adult_rule_total_count"] + 1e-8), 0.0)

        out["domain_invalid_flag"] = safe_str_series(out, "domain_bucket", "missing").eq("domain_invalid").astype(int)
        out["soccer_consistency_flag"] = safe_str_series(out, "soccer_consistency_bucket", "missing").isin([
            "strong_soccer_consistency_signal", "soccer_consistency_flag", "weak_soccer_consistency_signal"
        ]).astype(int)
        out["high_soccer_consistency_score"] = (out["soccer_consistency_score"] >= 1.0).astype(int)
        out["has_soccer_domain_signal"] = (out["soccer_domain_rule_count"] > 0).astype(int)
        out["has_soccer_format_signal"] = (out["soccer_format_rule_count"] > 0).astype(int)
        out["has_soccer_numeric_signal"] = (out["soccer_numeric_rule_count"] > 0).astype(int)
        out["has_soccer_consistency_signal"] = (out["soccer_consistency_rule_count"] > 0).astype(int)

        out["birthyear_season_gap"] = np.maximum(0.0, out["row_season_int"] - out["row_birthyear_int"])
        out["is_young_player_age"] = ((out["row_age_at_season"] > 0) & (out["row_age_at_season"] < 18)).astype(int)
        out["is_old_player_age"] = (out["row_age_at_season"] > 40).astype(int)

        # adult 兼容字段
        out["adult_consistency_flag"] = out["soccer_consistency_flag"]
        out["high_adult_consistency_score"] = out["high_soccer_consistency_score"]
        out["has_adult_domain_signal"] = out["has_soccer_domain_signal"]
        out["has_adult_format_signal"] = out["has_soccer_format_signal"]
        out["has_adult_consistency_signal"] = out["has_soccer_consistency_signal"]
        for c, v in {
            "age_span": 0.0, "hours_span": 0.0, "is_minor_age_bucket": 0, "is_high_hours_bucket": 0,
            "relationship_v2_conflict_count": 0, "has_relationship_v2_signal": 0,
            "has_education_v2_signal": 0, "adult_v2_signal_strength": 0.0,
            "adult_v2_primary_target_signal": 0,
        }.items():
            if c not in out.columns:
                out[c] = v

        out["rule_strength_sum"] += (
            1.5 * out["soccer_domain_rule_count"]
            + 1.3 * out["soccer_format_rule_count"]
            + 1.5 * out["soccer_numeric_rule_count"]
            + 1.2 * out["soccer_consistency_rule_count"]
        )
        out = add_soccer_features(out)

    return out


# ============================================================
# 5. 训练与预测函数
# ============================================================

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = None
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": None if auc is None else float(auc),
        "confusion_matrix": cm,
    }


def choose_best_threshold(y_true: np.ndarray, y_prob: np.ndarray, min_recall: float, cfg: TrainConfig) -> Tuple[float, Dict[str, Any]]:
    candidates = []
    for t in np.arange(cfg.threshold_grid_start, cfg.threshold_grid_end + 1e-8, cfg.threshold_grid_step):
        m = compute_metrics(y_true, y_prob, threshold=float(t))
        score = m["f1"] + cfg.precision_priority_alpha * m["precision"]
        candidates.append((float(t), m, score))
    valid = [x for x in candidates if x[1]["recall"] >= min_recall]
    if valid:
        valid.sort(key=lambda x: (x[2], x[1]["f1"], x[1]["precision"]), reverse=True)
        return valid[0][0], valid[0][1]
    candidates.sort(key=lambda x: (x[1]["f1"], x[1]["recall"], x[1]["precision"]), reverse=True)
    return candidates[0][0], candidates[0][1]


def compute_final_metrics_with_verifier(y_true, detector_pred_label, verifier_keep_prob, verifier_threshold):
    final_pred = detector_pred_label.copy().astype(int)
    mask = final_pred == 1
    final_pred[mask] = (verifier_keep_prob[mask] >= verifier_threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, final_pred, average="binary", zero_division=0)
    acc = accuracy_score(y_true, final_pred)
    cm = confusion_matrix(y_true, final_pred).tolist()
    return {
        "verifier_threshold": float(verifier_threshold),
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm,
        "detector_positive": int((detector_pred_label == 1).sum()),
        "final_positive": int(final_pred.sum()),
    }


def choose_best_verifier_threshold(y_true, detector_pred_label, verifier_keep_prob, min_recall, cfg: TrainConfig):
    candidates = []
    for t in np.arange(cfg.verifier_threshold_grid_start, cfg.verifier_threshold_grid_end + 1e-8, cfg.verifier_threshold_grid_step):
        m = compute_final_metrics_with_verifier(y_true, detector_pred_label, verifier_keep_prob, float(t))
        score = m["f1"] + cfg.precision_priority_alpha * m["precision"]
        candidates.append((float(t), m, score))
    valid = [x for x in candidates if x[1]["recall"] >= min_recall]
    if valid:
        valid.sort(key=lambda x: (x[2], x[1]["f1"], x[1]["precision"]), reverse=True)
        return valid[0][0], valid[0][1]
    candidates.sort(key=lambda x: (x[1]["f1"], x[1]["recall"], x[1]["precision"]), reverse=True)
    return candidates[0][0], candidates[0][1]


def load_training_data(path: str, cfg: TrainConfig) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if cfg.label_col not in df.columns:
        raise ValueError(f"输入文件缺少标签列: {cfg.label_col}")
    df[cfg.label_col] = pd.to_numeric(df[cfg.label_col], errors="coerce")
    df = df[df[cfg.label_col].isin([0, 1])].copy()
    df[cfg.label_col] = df[cfg.label_col].astype(int)
    if cfg.weight_col not in df.columns:
        df[cfg.weight_col] = 1.0
    df[cfg.weight_col] = pd.to_numeric(df[cfg.weight_col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return df


def build_preprocessor(df: pd.DataFrame, numeric_features: List[str], categorical_features: List[str]):
    num_cols = safe_existing_columns(df, numeric_features)
    cat_cols = safe_existing_columns(df, categorical_features)
    feature_cols = num_cols + cat_cols
    if not feature_cols:
        raise ValueError("没有可用特征列。")

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ])
    return preprocessor, feature_cols, num_cols, cat_cols


def fit_transform_preprocessor(preprocessor, train_df: pd.DataFrame, feature_cols: List[str]):
    X_train = preprocessor.fit_transform(train_df[feature_cols])
    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
    return X_train


def transform_preprocessor(preprocessor, df: pd.DataFrame, feature_cols: List[str]):
    X = preprocessor.transform(df[feature_cols])
    if hasattr(X, "toarray"):
        X = X.toarray()
    return X


def build_pos_weight(y: np.ndarray) -> float:
    pos = float(np.sum(y == 1))
    neg = float(np.sum(y == 0))
    if pos <= 0:
        return 1.0
    return max(1.0, neg / pos)


def weighted_bce_with_logits(logits, y, w, pos_weight):
    loss = nn.functional.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight, reduction="none")
    loss = loss * w
    return loss.mean()


def make_loader(X, y, w, batch_size, shuffle):
    ds = TabularDataset(X=X, y=y, w=w)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(model, loader, optimizer, device, pos_weight):
    model.train()
    total_loss = 0.0
    total_n = 0
    for batch in loader:
        x = batch["X"].to(device)
        y = batch["y"].to(device)
        w = batch["w"].to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = weighted_bce_with_logits(logits, y, w, pos_weight)
        loss.backward()
        optimizer.step()
        n = len(x)
        total_loss += float(loss.item()) * n
        total_n += n
    return {"loss": total_loss / max(total_n, 1)}


def predict_probs(model, loader, device):
    model.eval()
    probs = []
    with torch.no_grad():
        for batch in loader:
            x = batch["X"].to(device)
            logits = model(x)
            p = torch.sigmoid(logits).detach().cpu().numpy()
            probs.append(p)
    return np.concatenate(probs) if probs else np.array([])


def save_checkpoint(path: str, model, optimizer, epoch: int, input_dim: int, hidden_dim_1: int, hidden_dim_2: int, dropout: float, metrics: Dict[str, Any]):
    ensure_dir(os.path.dirname(path))
    torch.save({
        "epoch": int(epoch),
        "input_dim": int(input_dim),
        "hidden_dim_1": int(hidden_dim_1),
        "hidden_dim_2": int(hidden_dim_2),
        "dropout": float(dropout),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "metrics": metrics,
    }, path)


def load_checkpoint(path: str, device: str):
    return torch.load(path, map_location=device)


def run_single_stage_training(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
    model_hparams: Dict[str, Any],
    batch_size: int,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    min_delta: float,
    target_min_recall: float,
    ckpt_last_path: str,
    ckpt_best_path: str,
    preprocessor_path: str,
    train_log_path: str,
    cfg: TrainConfig,
):
    preprocessor, feature_cols, num_cols, cat_cols = build_preprocessor(train_df, numeric_features, categorical_features)
    X_train = fit_transform_preprocessor(preprocessor, train_df, feature_cols)
    X_val = transform_preprocessor(preprocessor, val_df, feature_cols)

    y_train = train_df[cfg.label_col].values.astype(int)
    y_val = val_df[cfg.label_col].values.astype(int)
    w_train = train_df[cfg.weight_col].values.astype(float)
    w_val = val_df[cfg.weight_col].values.astype(float)

    input_dim = X_train.shape[1]
    model = MLPDetector(input_dim=input_dim, **model_hparams).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    pos_weight = build_pos_weight(y_train) if cfg.use_pos_weight else 1.0
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32, device=DEVICE)

    train_loader = make_loader(X_train, y_train, w_train, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, w_val, batch_size=batch_size, shuffle=False)

    best_score = -1e18
    best_epoch = -1
    best_threshold = 0.5
    best_metrics = None
    stale_epochs = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        train_stats = train_one_epoch(model, train_loader, optimizer, DEVICE, pos_weight_tensor)
        val_prob = predict_probs(model, val_loader, DEVICE)

        if cfg.auto_find_best_threshold:
            epoch_threshold, epoch_metrics = choose_best_threshold(y_val, val_prob, min_recall=target_min_recall, cfg=cfg)
        else:
            epoch_threshold = 0.5
            epoch_metrics = compute_metrics(y_val, val_prob, threshold=epoch_threshold)

        score = epoch_metrics["f1"] + cfg.precision_priority_alpha * epoch_metrics["precision"]
        row = {"epoch": epoch, "train_loss": train_stats["loss"], "threshold": epoch_threshold, **epoch_metrics}
        history.append(row)

        save_checkpoint(ckpt_last_path, model, optimizer, epoch, input_dim, model_hparams["hidden_dim_1"], model_hparams["hidden_dim_2"], model_hparams["dropout"], epoch_metrics)

        if cfg.save_epoch_checkpoint and (epoch % cfg.save_every_epochs == 0):
            epoch_path = ckpt_last_path.replace("_last.pt", f"_epoch_{epoch}.pt")
            save_checkpoint(epoch_path, model, optimizer, epoch, input_dim, model_hparams["hidden_dim_1"], model_hparams["hidden_dim_2"], model_hparams["dropout"], epoch_metrics)

        if score > best_score + min_delta:
            best_score = score
            best_epoch = epoch
            best_threshold = epoch_threshold
            best_metrics = epoch_metrics
            stale_epochs = 0
            save_checkpoint(ckpt_best_path, model, optimizer, epoch, input_dim, model_hparams["hidden_dim_1"], model_hparams["hidden_dim_2"], model_hparams["dropout"], epoch_metrics)
            joblib.dump(preprocessor, preprocessor_path)
        else:
            stale_epochs += 1

        print(f"[epoch {epoch:03d}] loss={train_stats['loss']:.6f} "
              f"val_p={epoch_metrics['precision']:.4f} val_r={epoch_metrics['recall']:.4f} "
              f"val_f1={epoch_metrics['f1']:.4f} th={epoch_threshold:.2f}")

        if stale_epochs >= patience:
            print(f"[EARLY STOP] best_epoch={best_epoch}, best_score={best_score:.6f}")
            break

    pd.DataFrame(history).to_csv(train_log_path, index=False, encoding="utf-8-sig")

    best_ckpt = load_checkpoint(ckpt_best_path, DEVICE)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()
    preprocessor = joblib.load(preprocessor_path)

    return {
        "model": model,
        "preprocessor": preprocessor,
        "feature_cols": feature_cols,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "best_epoch": best_epoch,
        "best_threshold": best_threshold,
        "best_metrics": best_metrics,
    }


# ============================================================
# 6. stage2/verifier
# ============================================================

def make_stage2_training_df(df: pd.DataFrame, cfg: TrainConfig) -> pd.DataFrame:
    out = df.copy()
    if cfg.weight_col not in out.columns:
        out[cfg.weight_col] = 1.0
    out[cfg.weight_col] = pd.to_numeric(out[cfg.weight_col], errors="coerce").fillna(1.0)

    raw_prob = safe_num_series(out, "raw_pred_error_prob", 0.0)
    raw_label = safe_num_series(out, "raw_pred_label", 0).astype(int)
    ratio = safe_num_series(out, "neighbor_majority_ratio", 0.0)
    eq_neighbor = safe_num_series(out, "is_equal_to_neighbor_majority", 0).astype(int)
    column_series = safe_str_series(out, "column", "missing")

    out["stage1_high_conf_error"] = (raw_prob >= 0.80).astype(int)
    out["stage1_mid_zone"] = ((raw_prob >= cfg.mid_prob_low) & (raw_prob <= cfg.mid_prob_high)).astype(int)

    support_current_mask = (
        (eq_neighbor == 1)
        & (ratio >= 0.80)
        & (safe_num_series(out, "strong_rule_count", 0.0) <= 0)
    )
    out["high_prob_but_neighbor_support_current"] = ((raw_prob >= 0.80) & support_current_mask).astype(int)
    out["raw_margin_to_threshold"] = raw_prob - 0.5

    weight_multiplier = np.ones(len(out), dtype=float)

    if cfg.stage2_pos_pred_weight_boost != 1.0:
        weight_multiplier *= np.where(raw_label == 1, cfg.stage2_pos_pred_weight_boost, 1.0)
    if cfg.stage2_mid_prob_weight_boost != 1.0:
        weight_multiplier *= np.where(out["stage1_mid_zone"] == 1, cfg.stage2_mid_prob_weight_boost, 1.0)
    if cfg.stage2_risk_column_weight_boost != 1.0:
        weight_multiplier *= np.where(column_series.isin(cfg.stage2_false_pos_risk_cols), cfg.stage2_risk_column_weight_boost, 1.0)
    if cfg.stage2_support_current_weight_boost != 1.0:
        weight_multiplier *= np.where(support_current_mask, cfg.stage2_support_current_weight_boost, 1.0)

    rare_only_signal = safe_num_series(out, "rare_only_signal", 0).astype(int)
    if cfg.stage2_rare_only_weight_boost != 1.0:
        weight_multiplier *= np.where(rare_only_signal == 1, cfg.stage2_rare_only_weight_boost, 1.0)

    if cfg.dataset_kind == "adult":
        strong_domain = safe_num_series(out, "has_adult_domain_signal", 0).astype(int).eq(1)
        strong_cons = safe_num_series(out, "has_adult_consistency_signal", 0).astype(int).eq(1)
        weight_multiplier *= np.where(strong_domain, cfg.stage2_domain_signal_weight_boost, 1.0)
        weight_multiplier *= np.where(strong_cons, cfg.stage2_consistency_weight_boost, 1.0)

    if cfg.dataset_kind == "soccer":
        strong_domain = (
            safe_num_series(out, "has_soccer_domain_signal", 0).astype(int).eq(1)
            | safe_num_series(out, "has_soccer_numeric_signal", 0).astype(int).eq(1)
        )
        strong_cons = safe_num_series(out, "has_soccer_consistency_signal", 0).astype(int).eq(1)
        weight_multiplier *= np.where(strong_domain, cfg.stage2_domain_signal_weight_boost, 1.0)
        weight_multiplier *= np.where(strong_cons, cfg.stage2_consistency_weight_boost, 1.0)

    out[cfg.weight_col] = out[cfg.weight_col] * weight_multiplier
    return out


def attach_stage1_outputs(df, model, preprocessor, feature_cols, threshold, cfg: TrainConfig):
    out = df.copy()
    X = transform_preprocessor(preprocessor, out, feature_cols)
    loader = make_loader(X, np.zeros(len(out)), np.ones(len(out)), batch_size=1024, shuffle=False)
    prob = predict_probs(model, loader, DEVICE)
    out["raw_pred_error_prob"] = prob
    out["raw_pred_label"] = (prob >= threshold).astype(int)
    out["raw_margin_to_threshold"] = prob - threshold
    out["stage1_high_conf_error"] = (prob >= 0.80).astype(int)
    out["stage1_mid_zone"] = ((prob >= cfg.mid_prob_low) & (prob <= cfg.mid_prob_high)).astype(int)
    return out


def attach_stage2_outputs(df, model, preprocessor, feature_cols, threshold, cfg: TrainConfig):
    out = df.copy()
    X = transform_preprocessor(preprocessor, out, feature_cols)
    loader = make_loader(X, np.zeros(len(out)), np.ones(len(out)), batch_size=1024, shuffle=False)
    prob = predict_probs(model, loader, DEVICE)
    out["detector_pred_error_prob"] = prob
    out["detector_pred_label"] = (prob >= threshold).astype(int)
    out["detector_margin_to_threshold"] = prob - threshold
    out["is_stage_disagree"] = (out["raw_pred_label"].astype(int) != out["detector_pred_label"].astype(int)).astype(int)
    return out


def get_verifier_recall_rescue_mask(df: pd.DataFrame, cfg: TrainConfig) -> pd.Series:
    out = df.copy()
    raw_label = safe_num_series(out, "raw_pred_label", 0).astype(int)
    detector_label = safe_num_series(out, "detector_pred_label", 0).astype(int)
    raw_prob = safe_num_series(out, "raw_pred_error_prob", 0.0)
    detector_prob = safe_num_series(out, "detector_pred_error_prob", 0.0)
    posterior = safe_num_series(out, "posterior_error_probability", 0.0)

    common_gate = raw_label.eq(1) & detector_label.eq(0) & (raw_prob >= 0.95) & (posterior >= 0.90) & (detector_prob >= 0.40)

    if cfg.dataset_kind == "adult":
        strong = (
            safe_num_series(out, "adult_v2_primary_target_signal", 0).astype(int).eq(1)
            | safe_num_series(out, "has_adult_domain_signal", 0).astype(int).eq(1)
            | safe_num_series(out, "has_adult_format_signal", 0).astype(int).eq(1)
            | safe_num_series(out, "has_adult_consistency_signal", 0).astype(int).eq(1)
        )
        return (common_gate & strong).astype(bool)

    if cfg.dataset_kind == "soccer":
        strong = (
            safe_num_series(out, "soccer_primary_target_signal", 0).astype(int).eq(1)
            | safe_num_series(out, "has_soccer_domain_signal", 0).astype(int).eq(1)
            | safe_num_series(out, "has_soccer_numeric_signal", 0).astype(int).eq(1)
            | safe_num_series(out, "has_soccer_format_signal", 0).astype(int).eq(1)
            | safe_num_series(out, "has_soccer_consistency_signal", 0).astype(int).eq(1)
        )
        return (common_gate & strong).astype(bool)

    return common_gate.astype(bool)


def build_verifier_dataset(df: pd.DataFrame, cfg: TrainConfig) -> pd.DataFrame:
    out = df.copy()
    rescue_candidate_mask_all = get_verifier_recall_rescue_mask(out, cfg)
    out["is_recall_rescue_candidate"] = rescue_candidate_mask_all.astype(int)

    verifier_pool_mask = safe_num_series(out, "detector_pred_label", 0).astype(int).eq(1) | rescue_candidate_mask_all
    out = out[verifier_pool_mask].copy()
    if len(out) == 0:
        raise ValueError("verifier 训练集为空。")

    out["verifier_label"] = out[cfg.label_col].astype(int)
    if cfg.weight_col not in out.columns:
        out[cfg.weight_col] = 1.0
    out[cfg.weight_col] = pd.to_numeric(out[cfg.weight_col], errors="coerce").fillna(1.0)
    out["verifier_sample_weight"] = out[cfg.weight_col].copy()

    fp_mask = out["verifier_label"] == 0
    true_mask = out["verifier_label"] == 1
    col_series = safe_str_series(out, "column", "missing")
    rule_series = safe_str_series(out, "main_rule_type", "missing")
    nb_series = safe_str_series(out, "neighbor_bucket", "missing")
    rare_only_signal = safe_num_series(out, "rare_only_signal", 0).astype(int)
    detector_prob = safe_num_series(out, "detector_pred_error_prob", 0.0)

    out.loc[fp_mask, "verifier_sample_weight"] *= cfg.verifier_fp_base_weight_boost

    fd_conflict_fp_mask = fp_mask & rule_series.isin(["functional_dependency", "soft_functional_dependency"]) & nb_series.isin(["strong_support_current", "weak_support_current"])
    out.loc[fd_conflict_fp_mask, "verifier_sample_weight"] *= cfg.verifier_fd_conflict_fp_boost

    out.loc[fp_mask & (rare_only_signal == 1), "verifier_sample_weight"] *= cfg.verifier_rare_only_fp_boost
    out.loc[fp_mask & detector_prob.between(0.45, 0.85), "verifier_sample_weight"] *= cfg.verifier_borderline_fp_boost
    out.loc[fp_mask & col_series.isin(cfg.stage2_false_pos_risk_cols), "verifier_sample_weight"] *= cfg.verifier_risk_column_fp_boost
    out.loc[fp_mask & nb_series.eq("weak_support_other"), "verifier_sample_weight"] *= cfg.verifier_weak_other_fp_boost
    out.loc[
        fp_mask & nb_series.eq("weak_support_other") & rule_series.isin(["rare_value", "rare_pattern", "functional_dependency", "soft_functional_dependency"]),
        "verifier_sample_weight"
    ] *= cfg.verifier_weak_other_rare_fd_fp_boost

    # hospital/rayyan generic measure-like columns
    measure_cols = {"MeasureName", "MeasureCode", "Condition", "journal_title", "jounral_abbreviation", "journal_issn"}
    out.loc[fp_mask & col_series.isin(measure_cols), "verifier_sample_weight"] *= cfg.verifier_measure_col_fp_boost

    # true-error signal boosts
    out.loc[true_mask & col_series.eq("Sample"), "verifier_sample_weight"] *= cfg.verifier_sample_true_error_boost

    if cfg.dataset_kind == "adult":
        out.loc[true_mask & safe_num_series(out, "has_adult_domain_signal", 0).astype(int).eq(1), "verifier_sample_weight"] *= cfg.verifier_domain_true_error_boost
        out.loc[true_mask & safe_num_series(out, "has_adult_format_signal", 0).astype(int).eq(1), "verifier_sample_weight"] *= cfg.verifier_format_true_error_boost
        out.loc[true_mask & safe_num_series(out, "has_adult_consistency_signal", 0).astype(int).eq(1), "verifier_sample_weight"] *= cfg.verifier_consistency_true_error_boost

    if cfg.dataset_kind == "soccer":
        out.loc[true_mask & safe_num_series(out, "has_soccer_domain_signal", 0).astype(int).eq(1), "verifier_sample_weight"] *= cfg.verifier_domain_true_error_boost
        out.loc[true_mask & safe_num_series(out, "has_soccer_format_signal", 0).astype(int).eq(1), "verifier_sample_weight"] *= cfg.verifier_format_true_error_boost
        out.loc[true_mask & safe_num_series(out, "has_soccer_consistency_signal", 0).astype(int).eq(1), "verifier_sample_weight"] *= cfg.verifier_consistency_true_error_boost

    rescue_candidate_mask = safe_num_series(out, "is_recall_rescue_candidate", 0).astype(int).eq(1)
    out.loc[rescue_candidate_mask & true_mask, "verifier_sample_weight"] *= cfg.verifier_rescue_candidate_true_error_boost
    out.loc[rescue_candidate_mask & fp_mask, "verifier_sample_weight"] *= cfg.verifier_rescue_candidate_fp_damp

    print("[DEBUG] verifier_label distribution:")
    print(out["verifier_label"].value_counts(dropna=False))
    return out


def train_fp_verifier(train_df, val_df, numeric_features, categorical_features, cfg: TrainConfig, paths: Dict[str, str]):
    ensure_dir(os.path.dirname(paths["verifier_model"]))
    preprocessor, feature_cols, num_cols, cat_cols = build_preprocessor(train_df, numeric_features, categorical_features)
    preprocessor.fit(train_df[feature_cols])

    X_train = preprocessor.transform(train_df[feature_cols])
    X_val = preprocessor.transform(val_df[feature_cols])
    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
    if hasattr(X_val, "toarray"):
        X_val = X_val.toarray()

    y_train = train_df["verifier_label"].values.astype(int)
    y_val = val_df["verifier_label"].values.astype(int)
    w_train = pd.to_numeric(train_df["verifier_sample_weight"], errors="coerce").fillna(1.0).values.astype(float)

    unique_train_classes = np.unique(y_train)
    if len(unique_train_classes) < 2:
        only_class = int(unique_train_classes[0])
        print(f"[WARN] verifier 训练集只有一个类别: {only_class}，使用 DummyClassifier。")
        model = DummyClassifier(strategy="constant", constant=only_class)
        model.fit(X_train, y_train)
        train_prob = model.predict(X_train).astype(float)
        val_prob = model.predict(X_val).astype(float)
        is_single = True
    else:
        model = RandomForestClassifier(
            n_estimators=cfg.verifier_n_estimators,
            max_depth=cfg.verifier_max_depth,
            min_samples_leaf=cfg.verifier_min_samples_leaf,
            class_weight=cfg.verifier_class_weight,
            random_state=cfg.random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train, sample_weight=w_train)
        train_prob = model.predict_proba(X_train)[:, 1]
        val_prob = model.predict_proba(X_val)[:, 1]
        is_single = False

    verifier_train_metrics = compute_metrics(y_train, train_prob, threshold=0.5)
    verifier_val_metrics = compute_metrics(y_val, val_prob, threshold=0.5)

    joblib.dump(model, paths["verifier_model"])
    joblib.dump(preprocessor, paths["verifier_preprocessor"])
    save_json({
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "feature_cols": feature_cols,
        "is_single_class_verifier": bool(is_single),
        "single_class_value": int(unique_train_classes[0]) if len(unique_train_classes) == 1 else None,
    }, paths["verifier_config"])

    return {
        "model": model,
        "preprocessor": preprocessor,
        "feature_cols": feature_cols,
        "verifier_train_metrics": verifier_train_metrics,
        "verifier_val_metrics": verifier_val_metrics,
    }


# ============================================================
# 7. 主训练流程
# ============================================================

def get_paths(output_dir: str) -> Dict[str, str]:
    return {
        "stage1_last": os.path.join(output_dir, "stage1_base_last.pt"),
        "stage1_best": os.path.join(output_dir, "stage1_base_best.pt"),
        "stage1_preprocessor": os.path.join(output_dir, "stage1_base_preprocessor.joblib"),
        "stage1_train_log": os.path.join(output_dir, "stage1_train_log.csv"),
        "stage2_last": os.path.join(output_dir, "stage2_cal_last.pt"),
        "stage2_best": os.path.join(output_dir, "stage2_cal_best.pt"),
        "stage2_preprocessor": os.path.join(output_dir, "stage2_cal_preprocessor.joblib"),
        "stage2_train_log": os.path.join(output_dir, "stage2_train_log.csv"),
        "verifier_model": os.path.join(output_dir, "fp_verifier_model.joblib"),
        "verifier_preprocessor": os.path.join(output_dir, "fp_verifier_preprocessor.joblib"),
        "verifier_config": os.path.join(output_dir, "fp_verifier_config.json"),
        "verifier_metrics": os.path.join(output_dir, "verifier_metrics.json"),
        "verifier_train_csv": os.path.join(output_dir, "verifier_train_data.csv"),
        "verifier_val_csv": os.path.join(output_dir, "verifier_val_data.csv"),
        "config": os.path.join(output_dir, "two_stage_config.json"),
        "metrics": os.path.join(output_dir, "metrics_two_stage.json"),
        "final_val_pred": os.path.join(output_dir, "final_val_predictions.csv"),
        "stage2_train_analysis": os.path.join(output_dir, "stage2_train_analysis.csv"),
    }


def run_full_training(df: pd.DataFrame, cfg: TrainConfig, paths: Dict[str, str]):
    rest_df, final_val_df = train_test_split(
        df, test_size=cfg.final_val_size, random_state=cfg.random_state, stratify=df[cfg.label_col]
    )
    stage1_train_df, cal_train_df = train_test_split(
        rest_df, test_size=cfg.cal_train_size_in_rest, random_state=cfg.random_state, stratify=rest_df[cfg.label_col]
    )

    stage1_result = run_single_stage_training(
        train_df=stage1_train_df,
        val_df=final_val_df,
        numeric_features=cfg.stage1_numeric_features,
        categorical_features=cfg.stage1_categorical_features,
        model_hparams={
            "hidden_dim_1": cfg.stage1_hidden_dim_1,
            "hidden_dim_2": cfg.stage1_hidden_dim_2,
            "dropout": cfg.stage1_dropout,
        },
        batch_size=cfg.stage1_batch_size,
        lr=cfg.stage1_lr,
        weight_decay=cfg.stage1_weight_decay,
        max_epochs=cfg.stage1_max_epochs,
        patience=cfg.stage1_patience,
        min_delta=cfg.stage1_min_delta,
        target_min_recall=cfg.target_min_recall_stage1,
        ckpt_last_path=paths["stage1_last"],
        ckpt_best_path=paths["stage1_best"],
        preprocessor_path=paths["stage1_preprocessor"],
        train_log_path=paths["stage1_train_log"],
        cfg=cfg,
    )

    cal_train_with_stage1 = attach_stage1_outputs(
        cal_train_df, stage1_result["model"], stage1_result["preprocessor"],
        stage1_result["feature_cols"], stage1_result["best_threshold"], cfg
    )
    cal_train_with_stage1 = make_stage2_training_df(cal_train_with_stage1, cfg)

    final_val_with_stage1 = attach_stage1_outputs(
        final_val_df, stage1_result["model"], stage1_result["preprocessor"],
        stage1_result["feature_cols"], stage1_result["best_threshold"], cfg
    )

    stage2_result = run_single_stage_training(
        train_df=cal_train_with_stage1,
        val_df=final_val_with_stage1,
        numeric_features=cfg.stage2_numeric_features,
        categorical_features=cfg.stage2_categorical_features,
        model_hparams={
            "hidden_dim_1": cfg.stage2_hidden_dim_1,
            "hidden_dim_2": cfg.stage2_hidden_dim_2,
            "dropout": cfg.stage2_dropout,
        },
        batch_size=cfg.stage2_batch_size,
        lr=cfg.stage2_lr,
        weight_decay=cfg.stage2_weight_decay,
        max_epochs=cfg.stage2_max_epochs,
        patience=cfg.stage2_patience,
        min_delta=cfg.stage2_min_delta,
        target_min_recall=cfg.target_min_recall_stage2,
        ckpt_last_path=paths["stage2_last"],
        ckpt_best_path=paths["stage2_best"],
        preprocessor_path=paths["stage2_preprocessor"],
        train_log_path=paths["stage2_train_log"],
        cfg=cfg,
    )

    cal_train_with_stage2 = attach_stage2_outputs(
        cal_train_with_stage1, stage2_result["model"], stage2_result["preprocessor"],
        stage2_result["feature_cols"], stage2_result["best_threshold"], cfg
    )
    final_val_with_stage2 = attach_stage2_outputs(
        final_val_with_stage1, stage2_result["model"], stage2_result["preprocessor"],
        stage2_result["feature_cols"], stage2_result["best_threshold"], cfg
    )

    config = {
        "stage1": {
            "input_dim": int(stage1_result["model"].net[0].in_features),
            "hidden_dim_1": cfg.stage1_hidden_dim_1,
            "hidden_dim_2": cfg.stage1_hidden_dim_2,
            "dropout": cfg.stage1_dropout,
            "batch_size": cfg.stage1_batch_size,
            "learning_rate": cfg.stage1_lr,
            "weight_decay": cfg.stage1_weight_decay,
            "max_epochs": cfg.stage1_max_epochs,
            "patience": cfg.stage1_patience,
            "numeric_features": safe_existing_columns(df, cfg.stage1_numeric_features),
            "categorical_features": safe_existing_columns(df, cfg.stage1_categorical_features),
            "feature_cols": stage1_result["feature_cols"],
        },
        "stage2": {
            "input_dim": int(stage2_result["model"].net[0].in_features),
            "hidden_dim_1": cfg.stage2_hidden_dim_1,
            "hidden_dim_2": cfg.stage2_hidden_dim_2,
            "dropout": cfg.stage2_dropout,
            "batch_size": cfg.stage2_batch_size,
            "learning_rate": cfg.stage2_lr,
            "weight_decay": cfg.stage2_weight_decay,
            "max_epochs": cfg.stage2_max_epochs,
            "patience": cfg.stage2_patience,
            "numeric_features": safe_existing_columns(cal_train_with_stage1, cfg.stage2_numeric_features),
            "categorical_features": safe_existing_columns(cal_train_with_stage1, cfg.stage2_categorical_features),
            "feature_cols": stage2_result["feature_cols"],
        },
    }
    metrics = {
        "stage1_best_epoch": stage1_result["best_epoch"],
        "stage1_best_threshold": stage1_result["best_threshold"],
        "stage1_metrics": stage1_result["best_metrics"],
        "stage2_best_epoch": stage2_result["best_epoch"],
        "stage2_best_threshold": stage2_result["best_threshold"],
        "stage2_metrics": stage2_result["best_metrics"],
    }

    save_json(config, paths["config"])
    save_json(metrics, paths["metrics"])

    cal_train_with_stage2.to_csv(paths["stage2_train_analysis"], index=False, encoding="utf-8-sig")
    return stage1_result, stage2_result, cal_train_with_stage2, final_val_with_stage2, config, metrics


def load_detector_models_from_output(paths: Dict[str, str], cfg: TrainConfig):
    config = load_json(paths["config"])
    metrics = load_json(paths["metrics"])

    s1 = config["stage1"]
    stage1_model = MLPDetector(s1["input_dim"], s1["hidden_dim_1"], s1["hidden_dim_2"], s1["dropout"]).to(DEVICE)
    stage1_model.load_state_dict(load_checkpoint(paths["stage1_best"], DEVICE)["model_state_dict"])
    stage1_pre = joblib.load(paths["stage1_preprocessor"])
    stage1_model.eval()

    s2 = config["stage2"]
    stage2_model = MLPDetector(s2["input_dim"], s2["hidden_dim_1"], s2["hidden_dim_2"], s2["dropout"]).to(DEVICE)
    stage2_model.load_state_dict(load_checkpoint(paths["stage2_best"], DEVICE)["model_state_dict"])
    stage2_pre = joblib.load(paths["stage2_preprocessor"])
    stage2_model.eval()

    return stage1_model, stage1_pre, s1["feature_cols"], float(metrics["stage1_best_threshold"]), stage2_model, stage2_pre, s2["feature_cols"], float(metrics["stage2_best_threshold"]), config, metrics


def prepare_verifier_data_with_existing_detector(df: pd.DataFrame, cfg: TrainConfig, paths: Dict[str, str]):
    stage1_model, stage1_pre, stage1_features, stage1_th, stage2_model, stage2_pre, stage2_features, stage2_th, config, metrics = load_detector_models_from_output(paths, cfg)
    rest_df, final_val_df = train_test_split(
        df, test_size=cfg.final_val_size, random_state=cfg.random_state, stratify=df[cfg.label_col]
    )
    _, cal_train_df = train_test_split(
        rest_df, test_size=cfg.cal_train_size_in_rest, random_state=cfg.random_state, stratify=rest_df[cfg.label_col]
    )
    cal_train_with_stage1 = attach_stage1_outputs(cal_train_df, stage1_model, stage1_pre, stage1_features, stage1_th, cfg)
    final_val_with_stage1 = attach_stage1_outputs(final_val_df, stage1_model, stage1_pre, stage1_features, stage1_th, cfg)
    cal_train_with_stage1 = make_stage2_training_df(cal_train_with_stage1, cfg)
    cal_train_with_stage2 = attach_stage2_outputs(cal_train_with_stage1, stage2_model, stage2_pre, stage2_features, stage2_th, cfg)
    final_val_with_stage2 = attach_stage2_outputs(final_val_with_stage1, stage2_model, stage2_pre, stage2_features, stage2_th, cfg)
    return cal_train_with_stage2, final_val_with_stage2, config, metrics


def predict_verifier_keep_prob(verifier_result, df: pd.DataFrame):
    X = verifier_result["preprocessor"].transform(df[verifier_result["feature_cols"]])
    if hasattr(X, "toarray"):
        X = X.toarray()
    model = verifier_result["model"]
    if hasattr(model, "classes_") and len(model.classes_) == 1:
        return np.full(len(df), float(int(model.classes_[0])), dtype=float)
    if hasattr(model, "classes_") and len(model.classes_) == 2:
        class_to_index = {c: i for i, c in enumerate(model.classes_)}
        pos_idx = class_to_index.get(1, 1)
        return model.predict_proba(X)[:, pos_idx]
    return model.predict(X).astype(float)


def finalize_verifier_and_metrics(config, metrics, verifier_result, final_val_with_stage2, cfg: TrainConfig, paths: Dict[str, str], disable_verifier: bool = False):
    y_true_final = final_val_with_stage2[cfg.label_col].values.astype(int)
    detector_pred_label_final = final_val_with_stage2["detector_pred_label"].values.astype(int)

    if disable_verifier:
        all_verifier_keep_prob = np.ones(len(final_val_with_stage2), dtype=float)
        verifier_threshold = 0.5
        verifier_final_metrics = compute_final_metrics_with_verifier(y_true_final, detector_pred_label_final, all_verifier_keep_prob, verifier_threshold)
        verifier_result = {
            "verifier_train_metrics": {},
            "verifier_val_metrics": {},
            "feature_cols": [],
        }
    else:
        all_verifier_keep_prob = predict_verifier_keep_prob(verifier_result, final_val_with_stage2)
        verifier_threshold, verifier_final_metrics = choose_best_verifier_threshold(
            y_true=y_true_final,
            detector_pred_label=detector_pred_label_final,
            verifier_keep_prob=all_verifier_keep_prob,
            min_recall=cfg.verifier_min_final_recall,
            cfg=cfg,
        )

    final_pred_label = detector_pred_label_final.copy()
    detector_positive_mask = detector_pred_label_final == 1
    final_pred_label[detector_positive_mask] = (all_verifier_keep_prob[detector_positive_mask] >= verifier_threshold).astype(int)

    final_out = final_val_with_stage2.copy()
    final_out["true_label"] = final_out[cfg.label_col]
    final_out["verifier_keep_error_prob"] = all_verifier_keep_prob
    final_out["verifier_decision_threshold"] = float(verifier_threshold)
    final_out["verifier_decision"] = np.where(
        final_out["detector_pred_label"].values.astype(int) == 1,
        np.where(all_verifier_keep_prob >= verifier_threshold, "keep_error", "reject_error"),
        "skip",
    )
    final_out["final_pred_label"] = final_pred_label.astype(int)
    final_out["final_pred_label_name"] = final_out["final_pred_label"].map({1: "error", 0: "correct"})
    final_out["final_pred_error_prob"] = final_out["detector_pred_error_prob"]
    final_out.to_csv(paths["final_val_pred"], index=False, encoding="utf-8-sig")

    verifier_metrics_all = {
        "verifier_train_metrics": verifier_result.get("verifier_train_metrics", {}),
        "verifier_val_metrics": verifier_result.get("verifier_val_metrics", {}),
        "best_verifier_threshold_on_final_val": verifier_threshold,
        "final_pipeline_metrics_with_verifier": verifier_final_metrics,
    }
    save_json(verifier_metrics_all, paths["verifier_metrics"])

    two_stage_cfg = TwoStageConfig(
        device=DEVICE,
        random_state=cfg.random_state,
        label_col=cfg.label_col,
        weight_col=cfg.weight_col,
        final_val_size=cfg.final_val_size,
        cal_train_size_in_rest=cfg.cal_train_size_in_rest,
        stage1=StageConfig(**config["stage1"]),
        stage2=StageConfig(**config["stage2"]),
        verifier=VerifierConfig(
            numeric_features=safe_existing_columns(final_val_with_stage2, cfg.verifier_numeric_features),
            categorical_features=safe_existing_columns(final_val_with_stage2, cfg.verifier_categorical_features),
            feature_cols=verifier_result.get("feature_cols", []),
            threshold=float(verifier_threshold),
            metrics=verifier_final_metrics,
        ),
    )

    metrics_all = dict(metrics)
    metrics_all["verifier_train_metrics"] = verifier_result.get("verifier_train_metrics", {})
    metrics_all["verifier_val_metrics"] = verifier_result.get("verifier_val_metrics", {})
    metrics_all["verifier_best_threshold"] = verifier_threshold
    metrics_all["final_metrics_with_verifier"] = verifier_final_metrics
    save_json(metrics_all, paths["metrics"])
    save_json(two_stage_cfg, paths["config"])

    print("\n===== FINAL METRICS WITH VERIFIER =====")
    print(json.dumps(verifier_final_metrics, ensure_ascii=False, indent=2))


def train_verifier_from_stage_outputs(cal_train_with_stage2, final_val_with_stage2, cfg: TrainConfig, paths: Dict[str, str]):
    verifier_train_df = build_verifier_dataset(cal_train_with_stage2, cfg)
    verifier_val_df = build_verifier_dataset(final_val_with_stage2, cfg)

    verifier_train_df.to_csv(paths["verifier_train_csv"], index=False, encoding="utf-8-sig")
    verifier_val_df.to_csv(paths["verifier_val_csv"], index=False, encoding="utf-8-sig")

    return train_fp_verifier(
        train_df=verifier_train_df,
        val_df=verifier_val_df,
        numeric_features=cfg.verifier_numeric_features,
        categorical_features=cfg.verifier_categorical_features,
        cfg=cfg,
        paths=paths,
    )


def run_training(cfg: TrainConfig, disable_verifier: bool = False):
    set_seed(cfg.random_state)
    ensure_dir(cfg.output_dir)
    paths = get_paths(cfg.output_dir)

    df = load_training_data(cfg.input_csv, cfg)
    df = add_derived_features(df, cfg)

    print(f"[INFO] dataset={cfg.name}")
    print(f"[INFO] input={cfg.input_csv}")
    print(f"[INFO] output_dir={cfg.output_dir}")
    print(f"[INFO] rows={len(df)}, pos={int((df[cfg.label_col] == 1).sum())}, neg={int((df[cfg.label_col] == 0).sum())}")
    print(f"[INFO] mode={cfg.detector_train_mode}")

    if cfg.detector_train_mode == "full":
        stage1_result, stage2_result, cal_train_with_stage2, final_val_with_stage2, config, metrics = run_full_training(df, cfg, paths)
    elif cfg.detector_train_mode == "freeze_reuse":
        source_dir = cfg.detector_source_dir or cfg.output_dir
        if not os.path.exists(source_dir):
            raise ValueError(f"DETECTOR_SOURCE_DIR 不存在: {source_dir}")
        if os.path.abspath(source_dir) != os.path.abspath(cfg.output_dir):
            copy_detector_artifacts(source_dir, cfg.output_dir)
        cal_train_with_stage2, final_val_with_stage2, config, metrics = prepare_verifier_data_with_existing_detector(df, cfg, paths)
    else:
        raise ValueError(f"未知 DETECTOR_TRAIN_MODE: {cfg.detector_train_mode}")

    if disable_verifier:
        print("[INFO] verifier disabled; final metrics equal detector filtered by keep_prob=1.")
        finalize_verifier_and_metrics(config, metrics, {}, final_val_with_stage2, cfg, paths, disable_verifier=True)
    else:
        verifier_result = train_verifier_from_stage_outputs(cal_train_with_stage2, final_val_with_stage2, cfg, paths)
        finalize_verifier_and_metrics(config, metrics, verifier_result, final_val_with_stage2, cfg, paths, disable_verifier=False)

    print("\n===== TRAINING DONE =====")
    print(f"Output dir: {cfg.output_dir}")
    print(f"Stage1 best: {paths['stage1_best']}")
    print(f"Stage2 best: {paths['stage2_best']}")
    print(f"Verifier model: {paths['verifier_model']}")
    print(f"Metrics: {paths['metrics']}")


# ============================================================
# 8. CLI
# ============================================================

def clone_config(cfg: TrainConfig) -> TrainConfig:
    # dataclass with sets/lists; deep-copy through json-safe manually
    import copy
    return copy.deepcopy(cfg)


def apply_overrides(cfg: TrainConfig, args) -> TrainConfig:
    cfg = clone_config(cfg)

    if args.input_csv:
        cfg.input_csv = args.input_csv
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.mode:
        cfg.detector_train_mode = args.mode
    if args.source_dir:
        cfg.detector_source_dir = args.source_dir

    if args.target_min_recall_stage1 is not None:
        cfg.target_min_recall_stage1 = float(args.target_min_recall_stage1)
    if args.target_min_recall_stage2 is not None:
        cfg.target_min_recall_stage2 = float(args.target_min_recall_stage2)
    if args.verifier_min_final_recall is not None:
        cfg.verifier_min_final_recall = float(args.verifier_min_final_recall)

    if args.max_epochs is not None:
        cfg.stage1_max_epochs = int(args.max_epochs)
        cfg.stage2_max_epochs = int(args.max_epochs)
    if args.random_state is not None:
        cfg.random_state = int(args.random_state)

    return cfg


def run_one_dataset(dataset: str, args):
    cfg = apply_overrides(CONFIGS[dataset], args)

    old_cwd = None
    if args.cwd:
        old_cwd = os.getcwd()
        os.makedirs(args.cwd, exist_ok=True)
        os.chdir(args.cwd)

    try:
        run_training(cfg, disable_verifier=args.disable_verifier)
    finally:
        if old_cwd is not None:
            os.chdir(old_cwd)


def parse_args():
    parser = argparse.ArgumentParser(description="Unified two-stage MLP + verifier training script.")
    parser.add_argument("--dataset", required=True, choices=sorted(list(CONFIGS.keys()) + ["all"]))

    parser.add_argument("--input_csv", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--cwd", default=None)

    parser.add_argument("--mode", choices=["full", "freeze_reuse"], default=None)
    parser.add_argument("--source_dir", default=None)

    parser.add_argument("--disable-verifier", action="store_true")

    parser.add_argument("--target-min-recall-stage1", type=float, default=None)
    parser.add_argument("--target-min-recall-stage2", type=float, default=None)
    parser.add_argument("--verifier-min-final-recall", type=float, default=None)

    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.dataset == "all":
        if any([args.input_csv, args.output_dir, args.cwd]):
            raise ValueError("--dataset all 不建议和 --input_csv / --output_dir / --cwd 混用。请逐个数据集运行。")
        for ds in ["hospital", "flights", "beers", "rayyan", "adult", "soccer"]:
            print("\n" + "=" * 80)
            print(f"[DATASET] {ds}")
            print("=" * 80)
            run_one_dataset(ds, args)
    else:
        run_one_dataset(args.dataset, args)


if __name__ == "__main__":
    main()

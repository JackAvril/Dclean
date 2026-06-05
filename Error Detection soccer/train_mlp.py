import json
import os
import random
import shutil
from dataclasses import asdict, is_dataclass
from typing import List, Dict, Any, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# 1. 用户配置区
# ============================================================

INPUT_CSV = "final_train_dataset_soccer.csv"
OUTPUT_DIR = "two_stage_model_output_soccer_v4_with_verifier"

# full: 重新训练 stage1/stage2/verifier
# freeze_reuse: 复用 DETECTOR_SOURCE_DIR 中的 detector，仅重新训练 verifier
DETECTOR_TRAIN_MODE = "full"
DETECTOR_SOURCE_DIR = "two_stage_model_output_soccer_v4_with_verifier"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_STATE = 42

LABEL_COL = "label_binary"
WEIGHT_COL = "sample_weight"

FINAL_VAL_SIZE = 0.20
CAL_TRAIN_SIZE_IN_REST = 0.25

SAVE_EPOCH_CHECKPOINT = True
SAVE_EVERY_EPOCHS = 5
AUTO_FIND_BEST_THRESHOLD = True

TARGET_MIN_RECALL_STAGE1 = 0.90
TARGET_MIN_RECALL_STAGE2 = 0.88

THRESHOLD_GRID_START = 0.10
THRESHOLD_GRID_END = 0.95
THRESHOLD_GRID_STEP = 0.02
PRECISION_PRIORITY_ALPHA = 0.22

VERIFIER_THRESHOLD_GRID_START = 0.10
VERIFIER_THRESHOLD_GRID_END = 0.95
VERIFIER_THRESHOLD_GRID_STEP = 0.02
VERIFIER_MIN_FINAL_RECALL = 0.88

USE_POS_WEIGHT = True

# stage2 加权：soccer 中年份/position/domain/format/consistency 信号更关键
STAGE2_POS_PRED_WEIGHT_BOOST = 1.05
STAGE2_MID_PROB_WEIGHT_BOOST = 1.20
STAGE2_FALSE_POS_RISK_COLS = {"name", "surname", "team", "city", "stadium", "manager", "birthplace"}
STAGE2_RISK_COLUMN_WEIGHT_BOOST = 1.05
STAGE2_SUPPORT_CURRENT_WEIGHT_BOOST = 1.10
STAGE2_RARE_ONLY_WEIGHT_BOOST = 1.20
STAGE2_DOMAIN_SIGNAL_WEIGHT_BOOST = 1.35
STAGE2_SOCCER_CONSISTENCY_WEIGHT_BOOST = 1.30

MID_PROB_LOW = 0.35
MID_PROB_HIGH = 0.75

# Verifier 随机森林
VERIFIER_N_ESTIMATORS = 400
VERIFIER_MAX_DEPTH = 10
VERIFIER_MIN_SAMPLES_LEAF = 6
VERIFIER_CLASS_WEIGHT = "balanced_subsample"

VERIFIER_FP_BASE_WEIGHT_BOOST = 1.25
VERIFIER_FD_CONFLICT_FP_BOOST = 1.80
VERIFIER_RARE_ONLY_FP_BOOST = 1.80
VERIFIER_BORDERLINE_FP_BOOST = 1.45
VERIFIER_RISK_COLUMN_FP_BOOST = 1.35
VERIFIER_SAMPLE_TRUE_ERROR_BOOST = 1.35
VERIFIER_RESCUE_CANDIDATE_TRUE_ERROR_BOOST = 1.80
VERIFIER_RESCUE_CANDIDATE_FP_DAMP = 0.75
VERIFIER_WEAK_OTHER_FP_BOOST = 2.20
VERIFIER_DOMAIN_TRUE_ERROR_BOOST = 2.20
VERIFIER_FORMAT_TRUE_ERROR_BOOST = 2.00
VERIFIER_CONSISTENCY_TRUE_ERROR_BOOST = 1.80

STAGE1_LAST_CKPT = os.path.join(OUTPUT_DIR, "stage1_base_last.pt")
STAGE1_BEST_CKPT = os.path.join(OUTPUT_DIR, "stage1_base_best.pt")
STAGE1_PREPROCESSOR = os.path.join(OUTPUT_DIR, "stage1_base_preprocessor.joblib")
STAGE1_TRAIN_LOG = os.path.join(OUTPUT_DIR, "stage1_train_log.csv")

STAGE2_LAST_CKPT = os.path.join(OUTPUT_DIR, "stage2_cal_last.pt")
STAGE2_BEST_CKPT = os.path.join(OUTPUT_DIR, "stage2_cal_best.pt")
STAGE2_PREPROCESSOR = os.path.join(OUTPUT_DIR, "stage2_cal_preprocessor.joblib")
STAGE2_TRAIN_LOG = os.path.join(OUTPUT_DIR, "stage2_train_log.csv")

VERIFIER_MODEL_PATH = os.path.join(OUTPUT_DIR, "fp_verifier_model.joblib")
VERIFIER_PREPROCESSOR_PATH = os.path.join(OUTPUT_DIR, "fp_verifier_preprocessor.joblib")
VERIFIER_CONFIG_PATH = os.path.join(OUTPUT_DIR, "fp_verifier_config.json")
VERIFIER_METRICS_JSON = os.path.join(OUTPUT_DIR, "verifier_metrics.json")
VERIFIER_TRAIN_CSV = os.path.join(OUTPUT_DIR, "verifier_train_data.csv")
VERIFIER_VAL_CSV = os.path.join(OUTPUT_DIR, "verifier_val_data.csv")

CONFIG_PATH = os.path.join(OUTPUT_DIR, "two_stage_config.json")
METRICS_JSON = os.path.join(OUTPUT_DIR, "metrics_two_stage.json")
FINAL_VAL_PRED_CSV = os.path.join(OUTPUT_DIR, "final_val_predictions.csv")
STAGE2_TRAIN_ANALYSIS_CSV = os.path.join(OUTPUT_DIR, "stage2_train_analysis.csv")


# ============================================================
# 2. Soccer 特征列
# ============================================================

SOCCER_COLUMNS = {"name", "surname", "birthyear", "birthplace", "position", "team", "city", "stadium", "season", "manager"}

# Soccer v2：所有列都作为候选目标列。name/surname 不再是 context-only。
PRIMARY_TARGET_COLUMNS = set(SOCCER_COLUMNS)
CONTEXT_ONLY_COLUMNS = set()

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
# Soccer v2：soft-FD 只作为 evidence-only 弱证据，不再作为强 FD-like 信号。
SOCCER_FD_RULE_TYPES = {"functional_dependency"}
SOCCER_EVIDENCE_ONLY_FD_RULE_TYPES = {"soft_functional_dependency"}

SOCCER_NUMERIC_FEATURES = [
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
    "position_invalid_or_normalization",
    "has_birthyear_or_season_signal",
    "has_team_context_signal",
    "has_fd_like_signal",
    "is_context_only_weak_signal",
    "soccer_signal_strength",
    "soccer_primary_target_signal",
]
SOCCER_CATEGORICAL_FEATURES = ["soccer_attribution_bucket"]

STAGE1_NUMERIC_FEATURES = [
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

    # 兼容旧 adult 字段
    "adult_domain_rule_count", "adult_format_rule_count", "adult_consistency_rule_count",
    "adult_rule_total_count", "adult_domain_rule_ratio", "adult_format_rule_ratio",
    "adult_consistency_rule_ratio",
] + SOCCER_NUMERIC_FEATURES

STAGE1_CATEGORICAL_FEATURES = [
    "column", "semantic_type", "detected_type", "main_rule_type", "main_usage_role",
    "pattern_bucket", "rarity_bucket", "neighbor_bucket", "domain_bucket",
    "soccer_consistency_bucket", "soccer_value_bucket", "canonical_position",
    "row_position_canonical", "candidate_rule_dominant",
    "time_window_bucket", "time_value_bucket",

    # 兼容旧字段
    "adult_consistency_bucket", "age_sampling_bucket", "hours_sampling_bucket",
    "adult_value_bucket", "adult_v2_attribution_bucket",
] + SOCCER_CATEGORICAL_FEATURES

STAGE2_NUMERIC_FEATURES = [
    "raw_pred_error_prob", "raw_pred_label", "raw_margin_to_threshold",
    "stage1_high_conf_error", "stage1_mid_zone",
] + STAGE1_NUMERIC_FEATURES + [
    "rare_only_signal", "is_weak_only", "is_fp_risk_col",
    "high_prob_but_neighbor_support_current",
]

STAGE2_CATEGORICAL_FEATURES = STAGE1_CATEGORICAL_FEATURES

VERIFIER_NUMERIC_FEATURES = [
    "raw_pred_error_prob", "raw_pred_label", "raw_margin_to_threshold",
    "stage1_high_conf_error", "stage1_mid_zone",
    "detector_pred_error_prob", "detector_pred_label", "detector_margin_to_threshold",
    "is_stage_disagree",
] + STAGE1_NUMERIC_FEATURES + [
    "rare_only_signal", "is_weak_only", "is_fp_risk_col",
    "high_prob_but_neighbor_support_current",
]

VERIFIER_CATEGORICAL_FEATURES = STAGE1_CATEGORICAL_FEATURES

STAGE1_BATCH_SIZE = 64
STAGE1_MAX_EPOCHS = 60
STAGE1_LR = 1e-3
STAGE1_WEIGHT_DECAY = 1e-4
STAGE1_PATIENCE = 12
STAGE1_MIN_DELTA = 1e-4
STAGE1_HIDDEN_DIM_1 = 160
STAGE1_HIDDEN_DIM_2 = 80
STAGE1_DROPOUT = 0.20

STAGE2_BATCH_SIZE = 64
STAGE2_MAX_EPOCHS = 60
STAGE2_LR = 8e-4
STAGE2_WEIGHT_DECAY = 1e-4
STAGE2_PATIENCE = 12
STAGE2_MIN_DELTA = 1e-4
STAGE2_HIDDEN_DIM_1 = 96
STAGE2_HIDDEN_DIM_2 = 48
STAGE2_DROPOUT = 0.20


# ============================================================
# 3. 通用工具函数
# ============================================================

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
    if isinstance(obj, (list, tuple)):
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
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
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
        "stage1_base_last.pt", "stage1_base_best.pt", "stage1_base_preprocessor.joblib",
        "stage1_train_log.csv", "stage2_cal_last.pt", "stage2_cal_best.pt",
        "stage2_cal_preprocessor.joblib", "stage2_train_log.csv", "two_stage_config.json",
        "metrics_two_stage.json", "stage2_train_analysis.csv", "final_val_predictions.csv",
    ]
    for name in names:
        src = os.path.join(src_dir, name)
        dst = os.path.join(dst_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, dst)


# ============================================================
# 4. 特征构造
# ============================================================

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

    out["target_is_primary_column"] = np.where(
        col_series.isin(PRIMARY_TARGET_COLUMNS), 1, out["target_is_primary_column"]
    ).astype(int)
    out["target_is_context_only_column"] = np.where(
        col_series.isin(CONTEXT_ONLY_COLUMNS), 1, 0
    ).astype(int)

    out["birthyear_value_invalid"] = np.where(
        rule_series.eq("birthyear_numeric_range"), 1, out["birthyear_value_invalid"]
    ).astype(int)
    out["season_value_invalid"] = np.where(
        rule_series.eq("season_numeric_range"), 1, out["season_value_invalid"]
    ).astype(int)
    out["position_value_invalid"] = np.where(
        rule_series.eq("position_domain"), 1, out["position_value_invalid"]
    ).astype(int)
    out["birthyear_season_age_conflict"] = np.where(
        rule_series.eq("birthyear_season_age_consistency"), 1, out["birthyear_season_age_conflict"]
    ).astype(int)
    out["team_context_conflict"] = np.where(
        rule_series.isin(SOCCER_CONTEXT_RULE_TYPES), 1, out["team_context_conflict"]
    ).astype(int)
    out["format_rule_signal"] = np.where(
        rule_series.isin(SOCCER_FORMAT_RULE_TYPES), 1, out["format_rule_signal"]
    ).astype(int)
    out["fd_like_signal"] = np.where(
        rule_series.isin(SOCCER_FD_RULE_TYPES), 1, out["fd_like_signal"]
    ).astype(int)

    if "evidence_only_fd_count" not in out.columns:
        out["evidence_only_fd_count"] = 0
    out["evidence_only_fd_count"] = safe_num_series(out, "evidence_only_fd_count", 0).astype(float)
    soft_fd_mask = rule_series.isin(SOCCER_EVIDENCE_ONLY_FD_RULE_TYPES)
    out.loc[soft_fd_mask, "fd_like_signal"] = 0
    out.loc[soft_fd_mask, "evidence_only_fd_count"] = np.maximum(
        out.loc[soft_fd_mask, "evidence_only_fd_count"].astype(float), 1.0
    )

    out["position_invalid_or_normalization"] = (
        (out["position_value_invalid"].astype(int) > 0)
        | (out["position_needs_canonicalization"].astype(int) > 0)
    ).astype(int)

    out["has_birthyear_or_season_signal"] = (
        (out["birthyear_value_invalid"].astype(int) > 0)
        | (out["season_value_invalid"].astype(int) > 0)
        | (out["birthyear_season_age_conflict"].astype(int) > 0)
        | rule_series.isin({
            "birthyear_numeric_range",
            "season_numeric_range",
            "birthyear_season_age_consistency",
        })
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

    missing_bucket = out["soccer_attribution_bucket"].isin({"", "nan", "None", "missing", "unknown_attribution"})
    out.loc[missing_bucket & (out["birthyear_value_invalid"] == 1), "soccer_attribution_bucket"] = "birthyear_invalid"
    out.loc[missing_bucket & (out["season_value_invalid"] == 1), "soccer_attribution_bucket"] = "season_invalid"
    out.loc[missing_bucket & (out["position_value_invalid"] == 1), "soccer_attribution_bucket"] = "position_invalid"
    out.loc[missing_bucket & (out["position_needs_canonicalization"] == 1), "soccer_attribution_bucket"] = "position_canonicalization"
    out.loc[missing_bucket & (out["birthyear_season_age_conflict"] == 1), "soccer_attribution_bucket"] = "birthyear_season_age_conflict"
    out.loc[missing_bucket & (out["team_context_conflict"] == 1), "soccer_attribution_bucket"] = "team_context_conflict"
    out.loc[missing_bucket & (out["format_rule_signal"] == 1), "soccer_attribution_bucket"] = "format_pattern_conflict"
    out.loc[missing_bucket & (out["fd_like_signal"] == 1), "soccer_attribution_bucket"] = "fd_like_conflict"
    out.loc[missing_bucket & col_series.isin(PRIMARY_TARGET_COLUMNS), "soccer_attribution_bucket"] = "primary_target_other"
    out.loc[missing_bucket & col_series.isin(CONTEXT_ONLY_COLUMNS), "soccer_attribution_bucket"] = "context_only_other"

    return out


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in [
        "strong_rule_count", "candidate_generation_rule_count", "fd_like_count", "evidence_only_fd_count", "context_rule_count",
        "global_rule_count", "rare_value_count", "typo_rule_count", "pattern_rule_count", "schema_rule_count",
        "soccer_domain_rule_count", "soccer_format_rule_count", "soccer_numeric_rule_count",
        "soccer_consistency_rule_count", "soccer_profile_inconsistency_count",
        "adult_domain_rule_count", "adult_format_rule_count", "adult_consistency_rule_count",
        "time_window_rule_count",
    ]:
        out[col] = safe_num_series(out, col, 0.0)

    for col in [
        "conflict_score", "value_frequency", "value_frequency_rank", "neighbor_majority_ratio",
        "prior_error_probability", "posterior_error_probability", "soccer_consistency_score",
        "year_value", "row_birthyear_int", "row_season_int", "row_age_at_season",
        "birthyear_season_gap", "time_value_minutes",
    ]:
        out[col] = safe_num_series(out, col, 0.0)

    if "value" in out.columns and "neighbor_majority_value" in out.columns:
        value_series = normalize_text_series_for_compare(out["value"])
        neighbor_series = normalize_text_series_for_compare(out["neighbor_majority_value"])
        out["is_equal_to_neighbor_majority"] = (value_series == neighbor_series).astype(int)
    else:
        out["is_equal_to_neighbor_majority"] = safe_num_series(out, "is_equal_to_neighbor_majority", 0).astype(int)

    ratio = safe_num_series(out, "neighbor_majority_ratio", 0.0)
    out["neighbor_agreement_score"] = np.where(
        out["is_equal_to_neighbor_majority"] == 1,
        ratio,
        -ratio
    ).astype(float)

    out["has_neighbor_signal"] = (ratio > 0).astype(int)
    out["high_neighbor_consistency"] = (ratio >= 0.8).astype(int)

    out["rule_total_count"] = out["strong_rule_count"] + out["candidate_generation_rule_count"]
    out["candidate_rule_ratio"] = np.where(
        out["rule_total_count"] > 0,
        out["candidate_generation_rule_count"] / (out["rule_total_count"] + 1e-8),
        0.0,
    )
    out["strong_rule_ratio"] = np.where(
        out["rule_total_count"] > 0,
        out["strong_rule_count"] / (out["rule_total_count"] + 1e-8),
        0.0,
    )
    out["context_rule_ratio"] = np.where(
        out["rule_total_count"] > 0,
        out["context_rule_count"] / (out["rule_total_count"] + 1e-8),
        0.0,
    )
    out["fd_like_ratio"] = np.where(
        out["rule_total_count"] > 0,
        out["fd_like_count"] / (out["rule_total_count"] + 1e-8),
        0.0,
    )
    out["global_rule_ratio"] = np.where(
        out["rule_total_count"] > 0,
        out["global_rule_count"] / (out["rule_total_count"] + 1e-8),
        0.0,
    )

    out["soccer_rule_total_count"] = (
        out["soccer_domain_rule_count"] +
        out["soccer_format_rule_count"] +
        out["soccer_numeric_rule_count"] +
        out["soccer_consistency_rule_count"]
    )
    out["soccer_domain_rule_ratio"] = np.where(
        out["soccer_rule_total_count"] > 0,
        out["soccer_domain_rule_count"] / (out["soccer_rule_total_count"] + 1e-8),
        0.0,
    )
    out["soccer_format_rule_ratio"] = np.where(
        out["soccer_rule_total_count"] > 0,
        out["soccer_format_rule_count"] / (out["soccer_rule_total_count"] + 1e-8),
        0.0,
    )
    out["soccer_numeric_rule_ratio"] = np.where(
        out["soccer_rule_total_count"] > 0,
        out["soccer_numeric_rule_count"] / (out["soccer_rule_total_count"] + 1e-8),
        0.0,
    )
    out["soccer_consistency_rule_ratio"] = np.where(
        out["soccer_rule_total_count"] > 0,
        out["soccer_consistency_rule_count"] / (out["soccer_rule_total_count"] + 1e-8),
        0.0,
    )

    out["adult_rule_total_count"] = out["adult_domain_rule_count"] + out["adult_format_rule_count"] + out["adult_consistency_rule_count"]
    out["adult_domain_rule_ratio"] = 0.0
    out["adult_format_rule_ratio"] = 0.0
    out["adult_consistency_rule_ratio"] = 0.0

    long_tail_cols = {"name", "surname", "team", "city", "stadium", "manager", "birthplace"}
    col_series = safe_str_series(out, "column", "missing")
    value_rank = safe_num_series(out, "value_frequency_rank", 999999)
    value_freq = safe_num_series(out, "value_frequency", 999999)
    out["is_rare_value"] = np.where(
        col_series.isin(long_tail_cols),
        ((value_freq <= 1) | (value_rank > 25)).astype(int),
        ((value_freq <= 5) | (value_rank > 10)).astype(int),
    )

    out["neighbor_disagree"] = ((out["has_neighbor_signal"] == 1) & (out["is_equal_to_neighbor_majority"] == 0)).astype(int)

    out["rule_strength_sum"] = (
        1.00 * out["strong_rule_count"] +
        0.25 * out["candidate_generation_rule_count"] +
        0.45 * out["fd_like_count"] +
        0.35 * out["context_rule_count"] +
        0.15 * out["global_rule_count"] +
        0.20 * out["rare_value_count"] +
        0.35 * out["typo_rule_count"] +
        0.45 * out["pattern_rule_count"] +
        0.45 * out["schema_rule_count"] +
        0.95 * out["soccer_domain_rule_count"] +
        0.90 * out["soccer_format_rule_count"] +
        0.95 * out["soccer_numeric_rule_count"] +
        0.80 * out["soccer_consistency_rule_count"]
    )

    out["candidate_rule_dominant"] = np.where(
        out["candidate_generation_rule_count"] > out["strong_rule_count"], "yes", "no"
    )

    out["domain_invalid_flag"] = safe_str_series(out, "domain_bucket", "missing").eq("domain_invalid").astype(int)
    out["soccer_consistency_flag"] = safe_str_series(out, "soccer_consistency_bucket", "missing").isin([
        "strong_soccer_consistency_signal", "soccer_consistency_flag", "weak_soccer_consistency_signal"
    ]).astype(int)
    out["high_soccer_consistency_score"] = (out["soccer_consistency_score"] >= 1.0).astype(int)
    out["has_soccer_domain_signal"] = (out["soccer_domain_rule_count"] > 0).astype(int)
    out["has_soccer_format_signal"] = (out["soccer_format_rule_count"] > 0).astype(int)
    out["has_soccer_numeric_signal"] = (out["soccer_numeric_rule_count"] > 0).astype(int)
    out["has_soccer_consistency_signal"] = (out["soccer_consistency_rule_count"] > 0).astype(int)

    out["birthyear_season_gap"] = np.maximum(
        0.0, safe_num_series(out, "row_season_int", 0.0) - safe_num_series(out, "row_birthyear_int", 0.0)
    )
    out["is_young_player_age"] = ((safe_num_series(out, "row_age_at_season", 0.0) > 0) & (safe_num_series(out, "row_age_at_season", 0.0) < 18)).astype(int)
    out["is_old_player_age"] = (safe_num_series(out, "row_age_at_season", 0.0) > 40).astype(int)

    out = add_soccer_features(out)

    out["rare_only_signal"] = (
        (out["rare_value_count"] > 0) &
        (out["strong_rule_count"] <= 0) &
        (out["fd_like_count"] <= 0) &
        (out["context_rule_count"] <= 0) &
        (out["typo_rule_count"] <= 0) &
        (out["schema_rule_count"] <= 0) &
        (out["soccer_domain_rule_count"] <= 0) &
        (out["soccer_format_rule_count"] <= 0) &
        (out["soccer_numeric_rule_count"] <= 0) &
        (out["soccer_consistency_rule_count"] <= 0)
    ).astype(int)

    out["is_weak_only"] = (
        (out["rare_only_signal"] == 1) |
        (
            (out["strong_rule_count"] <= 0) &
            (out["fd_like_count"] <= 0) &
            (out["typo_rule_count"] <= 0) &
            (out["schema_rule_count"] <= 0) &
            (out["soccer_domain_rule_count"] <= 0) &
            (out["soccer_format_rule_count"] <= 0) &
            (out["soccer_numeric_rule_count"] <= 0) &
            (out["soccer_consistency_rule_count"] <= 0)
        )
    ).astype(int)

    out["is_fp_risk_col"] = safe_str_series(out, "column", "missing").isin(STAGE2_FALSE_POS_RISK_COLS).astype(int)

    out["high_prob_but_neighbor_support_current"] = (
        (out["high_neighbor_consistency"] == 1) &
        (out["is_equal_to_neighbor_majority"] == 1)
    ).astype(int)

    # 兼容旧 adult 字段，避免老 preprocessor 或旧下游读取时报错
    for c, v in {
        "adult_consistency_flag": 0,
        "high_adult_consistency_score": 0,
        "has_adult_domain_signal": 0,
        "has_adult_format_signal": 0,
        "has_adult_consistency_signal": 0,
        "age_span": 0.0,
        "hours_span": 0.0,
        "is_minor_age_bucket": 0,
        "is_high_hours_bucket": 0,
        "relationship_v2_conflict_count": 0,
        "has_relationship_v2_signal": 0,
        "has_education_v2_signal": 0,
        "adult_v2_signal_strength": 0.0,
        "adult_v2_primary_target_signal": 0,
    }.items():
        if c not in out.columns:
            out[c] = v

    return out


# ============================================================
# 5. 模型结构
# ============================================================

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
# 6. 训练工具
# ============================================================

def make_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols),
    ])


def transform_to_tensor(preprocessor, df: pd.DataFrame, device: str):
    x = preprocessor.transform(df)
    return torch.tensor(x, dtype=torch.float32, device=device)


def make_loader(X, y, w, batch_size: int, shuffle: bool):
    ds = torch.utils.data.TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
        torch.tensor(w, dtype=torch.float32),
    )
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def evaluate_probs(y_true, probs, threshold: float) -> Dict[str, Any]:
    pred = (probs >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    acc = accuracy_score(y_true, pred)
    try:
        auc = roc_auc_score(y_true, probs) if len(set(y_true)) > 1 else None
    except Exception:
        auc = None
    return {
        "precision": float(p), "recall": float(r), "f1": float(f1),
        "accuracy": float(acc), "auc": None if auc is None else float(auc),
        "threshold": float(threshold),
        "predicted_positive": int(pred.sum()),
    }


def find_best_threshold(y_true, probs, min_recall: float) -> Dict[str, Any]:
    best = None
    grid = np.arange(THRESHOLD_GRID_START, THRESHOLD_GRID_END + 1e-9, THRESHOLD_GRID_STEP)
    for th in grid:
        m = evaluate_probs(y_true, probs, th)
        if m["recall"] < min_recall:
            continue
        score = m["f1"] + PRECISION_PRIORITY_ALPHA * m["precision"]
        item = dict(m)
        item["score"] = float(score)
        if best is None or score > best["score"]:
            best = item
    if best is None:
        for th in grid:
            m = evaluate_probs(y_true, probs, th)
            score = m["f1"]
            item = dict(m)
            item["score"] = float(score)
            if best is None or score > best["score"]:
                best = item
    return best


def predict_mlp(model, X_np: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    probs = []
    with torch.no_grad():
        for start in range(0, len(X_np), batch_size):
            x = torch.tensor(X_np[start:start + batch_size], dtype=torch.float32, device=DEVICE)
            logits = model(x)
            p = torch.sigmoid(logits).detach().cpu().numpy()
            probs.append(p)
    return np.concatenate(probs) if probs else np.array([])


def train_mlp_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
    hidden1: int,
    hidden2: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    min_delta: float,
    batch_size: int,
    ckpt_last: str,
    ckpt_best: str,
    preprocessor_path: str,
    log_path: str,
    target_min_recall: float,
    model_name: str,
):
    preprocessor = make_preprocessor(num_cols, cat_cols)
    X_train = preprocessor.fit_transform(train_df)
    X_val = preprocessor.transform(val_df)

    joblib.dump(preprocessor, preprocessor_path)

    y_train = train_df[LABEL_COL].astype(int).values
    y_val = val_df[LABEL_COL].astype(int).values

    if WEIGHT_COL in train_df.columns:
        w_train = pd.to_numeric(train_df[WEIGHT_COL], errors="coerce").fillna(1.0).clip(lower=0.05).values
    else:
        w_train = np.ones(len(train_df))

    input_dim = X_train.shape[1]
    model = MLPDetector(input_dim, hidden1, hidden2, dropout).to(DEVICE)

    if USE_POS_WEIGHT:
        pos = max(float((y_train == 1).sum()), 1.0)
        neg = max(float((y_train == 0).sum()), 1.0)
        pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=DEVICE)
    else:
        pos_weight = None

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    loader = make_loader(X_train, y_train, w_train, batch_size, shuffle=True)

    best_score = -1
    best_epoch = -1
    wait = 0
    rows = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0

        for xb, yb, wb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            wb = wb.to(DEVICE)

            optimizer.zero_grad()
            logits = model(xb)
            loss_vec = criterion(logits, yb)
            loss = (loss_vec * wb).mean()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().cpu()) * len(xb)
            total_n += len(xb)

        val_probs = predict_mlp(model, X_val)
        th_info = find_best_threshold(y_val, val_probs, min_recall=target_min_recall)
        val_metrics = evaluate_probs(y_val, val_probs, th_info["threshold"])
        score = val_metrics["f1"] + PRECISION_PRIORITY_ALPHA * val_metrics["precision"]

        row = {
            "epoch": epoch, "loss": total_loss / max(total_n, 1),
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "selected_threshold": th_info["threshold"],
            "score": score,
        }
        rows.append(row)

        torch.save({
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "hidden_dim_1": hidden1,
            "hidden_dim_2": hidden2,
            "dropout": dropout,
            "epoch": epoch,
        }, ckpt_last)

        if SAVE_EPOCH_CHECKPOINT and epoch % SAVE_EVERY_EPOCHS == 0:
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_dim": input_dim,
                "hidden_dim_1": hidden1,
                "hidden_dim_2": hidden2,
                "dropout": dropout,
                "epoch": epoch,
            }, os.path.join(OUTPUT_DIR, f"{model_name}_epoch_{epoch}.pt"))

        if score > best_score + min_delta:
            best_score = score
            best_epoch = epoch
            wait = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_dim": input_dim,
                "hidden_dim_1": hidden1,
                "hidden_dim_2": hidden2,
                "dropout": dropout,
                "epoch": epoch,
                "threshold": th_info["threshold"],
            }, ckpt_best)
        else:
            wait += 1

        print(
            f"[{model_name}] epoch={epoch:03d} loss={row['loss']:.5f} "
            f"P={val_metrics['precision']:.4f} R={val_metrics['recall']:.4f} "
            f"F1={val_metrics['f1']:.4f} th={th_info['threshold']:.3f}"
        )

        if wait >= patience:
            break

    pd.DataFrame(rows).to_csv(log_path, index=False, encoding="utf-8-sig")

    ckpt = torch.load(ckpt_best, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    val_probs = predict_mlp(model, X_val)
    best_th = find_best_threshold(y_val, val_probs, min_recall=target_min_recall)
    final_metrics = evaluate_probs(y_val, val_probs, best_th["threshold"])
    final_metrics["best_epoch"] = int(best_epoch)

    return model, preprocessor, final_metrics


def load_model_from_ckpt(ckpt_path: str) -> MLPDetector:
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model = MLPDetector(
        input_dim=ckpt["input_dim"],
        hidden_dim_1=ckpt["hidden_dim_1"],
        hidden_dim_2=ckpt["hidden_dim_2"],
        dropout=ckpt["dropout"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def add_stage1_features(df: pd.DataFrame, probs: np.ndarray, threshold: float) -> pd.DataFrame:
    out = df.copy()
    out["raw_pred_error_prob"] = probs
    out["raw_pred_label"] = (probs >= threshold).astype(int)
    out["raw_margin_to_threshold"] = probs - threshold
    out["stage1_high_conf_error"] = (probs >= max(threshold + 0.20, 0.75)).astype(int)
    out["stage1_mid_zone"] = ((probs >= MID_PROB_LOW) & (probs <= MID_PROB_HIGH)).astype(int)
    return out


def add_stage2_weights(df: pd.DataFrame) -> pd.Series:
    w = pd.to_numeric(df.get(WEIGHT_COL, 1.0), errors="coerce").fillna(1.0).astype(float)

    w = w * np.where(df["raw_pred_label"].astype(int) == 1, STAGE2_POS_PRED_WEIGHT_BOOST, 1.0)
    w = w * np.where(df["stage1_mid_zone"].astype(int) == 1, STAGE2_MID_PROB_WEIGHT_BOOST, 1.0)
    w = w * np.where(df["is_fp_risk_col"].astype(int) == 1, STAGE2_RISK_COLUMN_WEIGHT_BOOST, 1.0)
    w = w * np.where(df["high_prob_but_neighbor_support_current"].astype(int) == 1, STAGE2_SUPPORT_CURRENT_WEIGHT_BOOST, 1.0)
    w = w * np.where(df["rare_only_signal"].astype(int) == 1, STAGE2_RARE_ONLY_WEIGHT_BOOST, 1.0)
    w = w * np.where(df["has_soccer_domain_signal"].astype(int) == 1, STAGE2_DOMAIN_SIGNAL_WEIGHT_BOOST, 1.0)
    w = w * np.where(df["has_soccer_format_signal"].astype(int) == 1, STAGE2_DOMAIN_SIGNAL_WEIGHT_BOOST, 1.0)
    w = w * np.where(df["has_soccer_numeric_signal"].astype(int) == 1, STAGE2_DOMAIN_SIGNAL_WEIGHT_BOOST, 1.0)
    w = w * np.where(df["has_soccer_consistency_signal"].astype(int) == 1, STAGE2_SOCCER_CONSISTENCY_WEIGHT_BOOST, 1.0)

    return pd.Series(w, index=df.index).clip(lower=0.05, upper=10.0)


def make_detector_predictions(stage1_model, stage1_pre, stage2_model, stage2_pre, df: pd.DataFrame, stage1_th: float, stage2_th: float):
    X1 = stage1_pre.transform(df)
    p1 = predict_mlp(stage1_model, X1)
    stage2_df = add_stage1_features(df, p1, stage1_th)
    X2 = stage2_pre.transform(stage2_df)
    p2 = predict_mlp(stage2_model, X2)
    pred2 = (p2 >= stage2_th).astype(int)
    return stage2_df, p1, p2, pred2


def add_verifier_features(df: pd.DataFrame, detector_probs: np.ndarray, detector_th: float) -> pd.DataFrame:
    out = df.copy()
    out["detector_pred_error_prob"] = detector_probs
    out["detector_pred_label"] = (detector_probs >= detector_th).astype(int)
    out["detector_margin_to_threshold"] = detector_probs - detector_th
    out["is_stage_disagree"] = (out["raw_pred_label"].astype(int) != out["detector_pred_label"].astype(int)).astype(int)
    return out


def train_verifier(train_df: pd.DataFrame, val_df: pd.DataFrame, stage1_model, stage1_pre, stage2_model, stage2_pre, stage1_th: float, stage2_th: float):
    train_stage2_df, p1_train, p2_train, pred_train = make_detector_predictions(
        stage1_model, stage1_pre, stage2_model, stage2_pre, train_df, stage1_th, stage2_th
    )
    val_stage2_df, p1_val, p2_val, pred_val = make_detector_predictions(
        stage1_model, stage1_pre, stage2_model, stage2_pre, val_df, stage1_th, stage2_th
    )

    ver_train = add_verifier_features(train_stage2_df, p2_train, stage2_th)
    ver_val = add_verifier_features(val_stage2_df, p2_val, stage2_th)

    y_train = train_df[LABEL_COL].astype(int).values
    y_val = val_df[LABEL_COL].astype(int).values

    # verifier 的目标是“detector 预测为 error 时是否保留 error”
    keep_train = y_train.copy()
    keep_val = y_val.copy()

    num_cols = safe_existing_columns(ver_train, VERIFIER_NUMERIC_FEATURES)
    cat_cols = safe_existing_columns(ver_train, VERIFIER_CATEGORICAL_FEATURES)

    pre = make_preprocessor(num_cols, cat_cols)
    X_train = pre.fit_transform(ver_train)
    X_val = pre.transform(ver_val)

    sample_weight = np.ones(len(ver_train), dtype=float)
    fp_mask = (pred_train == 1) & (y_train == 0)
    tp_mask = (pred_train == 1) & (y_train == 1)

    sample_weight[fp_mask] *= VERIFIER_FP_BASE_WEIGHT_BOOST
    sample_weight[tp_mask] *= VERIFIER_SAMPLE_TRUE_ERROR_BOOST
    sample_weight *= np.where(ver_train["rare_only_signal"].astype(int).values == 1, VERIFIER_RARE_ONLY_FP_BOOST, 1.0)
    sample_weight *= np.where(ver_train["is_fp_risk_col"].astype(int).values == 1, VERIFIER_RISK_COLUMN_FP_BOOST, 1.0)
    sample_weight *= np.where(ver_train["is_weak_only"].astype(int).values == 1, VERIFIER_WEAK_OTHER_FP_BOOST, 1.0)
    sample_weight *= np.where(ver_train["has_soccer_domain_signal"].astype(int).values == 1, VERIFIER_DOMAIN_TRUE_ERROR_BOOST, 1.0)
    sample_weight *= np.where(ver_train["has_soccer_format_signal"].astype(int).values == 1, VERIFIER_FORMAT_TRUE_ERROR_BOOST, 1.0)
    sample_weight *= np.where(ver_train["has_soccer_numeric_signal"].astype(int).values == 1, VERIFIER_DOMAIN_TRUE_ERROR_BOOST, 1.0)
    sample_weight *= np.where(ver_train["has_soccer_consistency_signal"].astype(int).values == 1, VERIFIER_CONSISTENCY_TRUE_ERROR_BOOST, 1.0)

    if len(set(keep_train)) < 2:
        clf = DummyClassifier(strategy="constant", constant=int(keep_train[0]))
        clf.fit(X_train, keep_train)
    else:
        clf = RandomForestClassifier(
            n_estimators=VERIFIER_N_ESTIMATORS,
            max_depth=VERIFIER_MAX_DEPTH,
            min_samples_leaf=VERIFIER_MIN_SAMPLES_LEAF,
            class_weight=VERIFIER_CLASS_WEIGHT,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        clf.fit(X_train, keep_train, sample_weight=sample_weight)

    if hasattr(clf, "predict_proba") and len(getattr(clf, "classes_", [])) > 1:
        keep_prob_val = clf.predict_proba(X_val)[:, list(clf.classes_).index(1)]
    else:
        keep_prob_val = np.full(len(val_df), float(keep_train.mean()))

    best = None
    for th in np.arange(VERIFIER_THRESHOLD_GRID_START, VERIFIER_THRESHOLD_GRID_END + 1e-9, VERIFIER_THRESHOLD_GRID_STEP):
        final_pred = ((pred_val == 1) & (keep_prob_val >= th)).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_val, final_pred, average="binary", zero_division=0)
        if r < VERIFIER_MIN_FINAL_RECALL:
            continue
        score = f1 + 0.15 * p
        item = {"threshold": float(th), "precision": float(p), "recall": float(r), "f1": float(f1), "score": float(score)}
        if best is None or score > best["score"]:
            best = item

    if best is None:
        best = {"threshold": 0.50, **evaluate_probs(y_val, keep_prob_val, 0.50), "score": 0.0}

    joblib.dump(pre, VERIFIER_PREPROCESSOR_PATH)
    joblib.dump(clf, VERIFIER_MODEL_PATH)

    ver_train.to_csv(VERIFIER_TRAIN_CSV, index=False, encoding="utf-8-sig")
    ver_val.to_csv(VERIFIER_VAL_CSV, index=False, encoding="utf-8-sig")

    save_json({
        "verifier_threshold": best["threshold"],
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
    }, VERIFIER_CONFIG_PATH)
    save_json(best, VERIFIER_METRICS_JSON)

    return clf, pre, best, ver_val, keep_prob_val


# ============================================================
# 7. 主流程
# ============================================================

def main():
    set_seed(RANDOM_STATE)
    ensure_dir(OUTPUT_DIR)

    df = pd.read_csv(INPUT_CSV, low_memory=False)
    df = add_derived_features(df)

    if LABEL_COL not in df.columns:
        raise ValueError(f"训练文件缺少标签列: {LABEL_COL}")

    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce")
    df = df[df[LABEL_COL].notna()].copy()
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    if WEIGHT_COL not in df.columns:
        df[WEIGHT_COL] = 1.0
    df[WEIGHT_COL] = pd.to_numeric(df[WEIGHT_COL], errors="coerce").fillna(1.0).clip(lower=0.05)

    y = df[LABEL_COL].values
    strat = y if len(np.unique(y)) > 1 and min(np.bincount(y)) >= 2 else None

    rest_df, final_val_df = train_test_split(
        df, test_size=FINAL_VAL_SIZE, random_state=RANDOM_STATE, stratify=strat
    )

    y_rest = rest_df[LABEL_COL].values
    strat_rest = y_rest if len(np.unique(y_rest)) > 1 and min(np.bincount(y_rest)) >= 2 else None

    stage1_train_df, cal_train_df = train_test_split(
        rest_df, test_size=CAL_TRAIN_SIZE_IN_REST, random_state=RANDOM_STATE, stratify=strat_rest
    )

    print(f"[INFO] DEVICE={DEVICE}")
    print(f"[INFO] train={len(stage1_train_df)}, cal={len(cal_train_df)}, final_val={len(final_val_df)}")
    print(f"[INFO] label distribution: {df[LABEL_COL].value_counts().to_dict()}")

    stage1_num = safe_existing_columns(stage1_train_df, STAGE1_NUMERIC_FEATURES)
    stage1_cat = safe_existing_columns(stage1_train_df, STAGE1_CATEGORICAL_FEATURES)

    if DETECTOR_TRAIN_MODE == "freeze_reuse":
        copy_detector_artifacts(DETECTOR_SOURCE_DIR, OUTPUT_DIR)
        stage1_model = load_model_from_ckpt(STAGE1_BEST_CKPT)
        stage1_pre = joblib.load(STAGE1_PREPROCESSOR)
        config_old = load_json(CONFIG_PATH)
        stage1_metrics = config_old.get("stage1_metrics", {})
        stage1_th = float(stage1_metrics.get("threshold", 0.5))
    else:
        stage1_model, stage1_pre, stage1_metrics = train_mlp_model(
            train_df=stage1_train_df,
            val_df=cal_train_df,
            num_cols=stage1_num,
            cat_cols=stage1_cat,
            hidden1=STAGE1_HIDDEN_DIM_1,
            hidden2=STAGE1_HIDDEN_DIM_2,
            dropout=STAGE1_DROPOUT,
            lr=STAGE1_LR,
            weight_decay=STAGE1_WEIGHT_DECAY,
            max_epochs=STAGE1_MAX_EPOCHS,
            patience=STAGE1_PATIENCE,
            min_delta=STAGE1_MIN_DELTA,
            batch_size=STAGE1_BATCH_SIZE,
            ckpt_last=STAGE1_LAST_CKPT,
            ckpt_best=STAGE1_BEST_CKPT,
            preprocessor_path=STAGE1_PREPROCESSOR,
            log_path=STAGE1_TRAIN_LOG,
            target_min_recall=TARGET_MIN_RECALL_STAGE1,
            model_name="stage1_base",
        )
        stage1_th = float(stage1_metrics["threshold"])

        # 构造 stage2 训练集
        X_stage1_cal = stage1_pre.transform(cal_train_df)
        p_stage1_cal = predict_mlp(stage1_model, X_stage1_cal)
        cal_stage2_df = add_stage1_features(cal_train_df, p_stage1_cal, stage1_th)
        cal_stage2_df[WEIGHT_COL] = add_stage2_weights(cal_stage2_df)

        X_stage1_val = stage1_pre.transform(final_val_df)
        p_stage1_val = predict_mlp(stage1_model, X_stage1_val)
        final_stage2_df = add_stage1_features(final_val_df, p_stage1_val, stage1_th)

        stage2_num = safe_existing_columns(cal_stage2_df, STAGE2_NUMERIC_FEATURES)
        stage2_cat = safe_existing_columns(cal_stage2_df, STAGE2_CATEGORICAL_FEATURES)

        stage2_model, stage2_pre, stage2_metrics = train_mlp_model(
            train_df=cal_stage2_df,
            val_df=final_stage2_df,
            num_cols=stage2_num,
            cat_cols=stage2_cat,
            hidden1=STAGE2_HIDDEN_DIM_1,
            hidden2=STAGE2_HIDDEN_DIM_2,
            dropout=STAGE2_DROPOUT,
            lr=STAGE2_LR,
            weight_decay=STAGE2_WEIGHT_DECAY,
            max_epochs=STAGE2_MAX_EPOCHS,
            patience=STAGE2_PATIENCE,
            min_delta=STAGE2_MIN_DELTA,
            batch_size=STAGE2_BATCH_SIZE,
            ckpt_last=STAGE2_LAST_CKPT,
            ckpt_best=STAGE2_BEST_CKPT,
            preprocessor_path=STAGE2_PREPROCESSOR,
            log_path=STAGE2_TRAIN_LOG,
            target_min_recall=TARGET_MIN_RECALL_STAGE2,
            model_name="stage2_cal",
        )
        stage2_th = float(stage2_metrics["threshold"])

    if DETECTOR_TRAIN_MODE == "freeze_reuse":
        stage2_model = load_model_from_ckpt(STAGE2_BEST_CKPT)
        stage2_pre = joblib.load(STAGE2_PREPROCESSOR)
        config_old = load_json(CONFIG_PATH)
        stage2_metrics = config_old.get("stage2_metrics", {})
        stage2_th = float(stage2_metrics.get("threshold", 0.5))

    # final validation detector metrics
    final_stage2_df, p1_final, p2_final, pred2_final = make_detector_predictions(
        stage1_model, stage1_pre, stage2_model, stage2_pre, final_val_df, stage1_th, stage2_th
    )
    y_final = final_val_df[LABEL_COL].astype(int).values
    detector_metrics = evaluate_probs(y_final, p2_final, stage2_th)

    # verifier
    verifier_model, verifier_pre, verifier_metrics, ver_val_df, keep_prob_val = train_verifier(
        train_df=cal_train_df,
        val_df=final_val_df,
        stage1_model=stage1_model,
        stage1_pre=stage1_pre,
        stage2_model=stage2_model,
        stage2_pre=stage2_pre,
        stage1_th=stage1_th,
        stage2_th=stage2_th,
    )

    final_pred_with_verifier = ((pred2_final == 1) & (keep_prob_val >= verifier_metrics["threshold"])).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_final, final_pred_with_verifier, average="binary", zero_division=0)
    final_metrics = {"precision": float(p), "recall": float(r), "f1": float(f1), "threshold": float(verifier_metrics["threshold"])}

    final_pred_df = final_val_df.copy()
    final_pred_df["stage1_prob"] = p1_final
    final_pred_df["stage2_prob"] = p2_final
    final_pred_df["detector_pred_label"] = pred2_final
    final_pred_df["verifier_keep_error_prob"] = keep_prob_val
    final_pred_df["final_pred_label"] = final_pred_with_verifier
    final_pred_df.to_csv(FINAL_VAL_PRED_CSV, index=False, encoding="utf-8-sig")

    config = {
        "input_csv": INPUT_CSV,
        "output_dir": OUTPUT_DIR,
        "device": DEVICE,
        "stage1_numeric_features": stage1_num,
        "stage1_categorical_features": stage1_cat,
        "stage2_numeric_features": safe_existing_columns(add_stage1_features(cal_train_df, np.zeros(len(cal_train_df)), stage1_th), STAGE2_NUMERIC_FEATURES),
        "stage2_categorical_features": safe_existing_columns(cal_train_df, STAGE2_CATEGORICAL_FEATURES),
        "stage1_metrics": stage1_metrics,
        "stage2_metrics": stage2_metrics,
        "detector_metrics": detector_metrics,
        "verifier_metrics": verifier_metrics,
        "final_with_verifier_metrics": final_metrics,
        "model_arch": {
            "stage1": {"hidden_dim_1": STAGE1_HIDDEN_DIM_1, "hidden_dim_2": STAGE1_HIDDEN_DIM_2, "dropout": STAGE1_DROPOUT},
            "stage2": {"hidden_dim_1": STAGE2_HIDDEN_DIM_1, "hidden_dim_2": STAGE2_HIDDEN_DIM_2, "dropout": STAGE2_DROPOUT},
        }
    }
    save_json(config, CONFIG_PATH)
    save_json({
        "stage1": stage1_metrics,
        "stage2": stage2_metrics,
        "detector": detector_metrics,
        "verifier": verifier_metrics,
        "final_with_verifier": final_metrics,
    }, METRICS_JSON)

    print("\n===== DONE =====")
    print(f"Model dir: {OUTPUT_DIR}")
    print(f"Stage1: P={stage1_metrics.get('precision', 0):.4f} R={stage1_metrics.get('recall', 0):.4f} F1={stage1_metrics.get('f1', 0):.4f}")
    print(f"Stage2: P={stage2_metrics.get('precision', 0):.4f} R={stage2_metrics.get('recall', 0):.4f} F1={stage2_metrics.get('f1', 0):.4f}")
    print(f"Detector: P={detector_metrics.get('precision', 0):.4f} R={detector_metrics.get('recall', 0):.4f} F1={detector_metrics.get('f1', 0):.4f}")
    print(f"Final+Verifier: P={final_metrics.get('precision', 0):.4f} R={final_metrics.get('recall', 0):.4f} F1={final_metrics.get('f1', 0):.4f}")


if __name__ == "__main__":
    main()

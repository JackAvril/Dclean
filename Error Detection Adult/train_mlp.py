import json
import os
import random
import shutil
from dataclasses import dataclass, asdict, is_dataclass
from typing import List, Dict, Any, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# 1. 用户配置区
# ============================================================

INPUT_CSV = "final_train_dataset_adult.csv"
OUTPUT_DIR = "two_stage_model_output_adult_v4_with_verifier"

# full: 重新训练 stage1/stage2/verifier
# freeze_reuse: 复用 DETECTOR_SOURCE_DIR 中的 detector，仅重新训练 verifier
DETECTOR_TRAIN_MODE = "full"
DETECTOR_SOURCE_DIR = "two_stage_model_output_adult_v4_with_verifier"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_STATE = 42

LABEL_COL = "label_binary"
WEIGHT_COL = "sample_weight"

FINAL_VAL_SIZE = 0.20
CAL_TRAIN_SIZE_IN_REST = 0.25

SAVE_EPOCH_CHECKPOINT = True
SAVE_EVERY_EPOCHS = 5
AUTO_FIND_BEST_THRESHOLD = True

# adult 候选召回通常来自规则池，训练阈值不要一味极低，否则 precision 会掉。
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

# stage2 加权：adult 中 strong/domain/format/consistency 信号更关键
STAGE2_POS_PRED_WEIGHT_BOOST = 1.05
STAGE2_MID_PROB_WEIGHT_BOOST = 1.20
STAGE2_FALSE_POS_RISK_COLS = {"country", "occupation", "workclass", "education"}
STAGE2_RISK_COLUMN_WEIGHT_BOOST = 1.15
STAGE2_SUPPORT_CURRENT_WEIGHT_BOOST = 1.10
STAGE2_RARE_ONLY_WEIGHT_BOOST = 1.20
STAGE2_DOMAIN_SIGNAL_WEIGHT_BOOST = 1.35
STAGE2_ADULT_CONSISTENCY_WEIGHT_BOOST = 1.30

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
VERIFIER_RISK_COLUMN_FP_BOOST = 2.00
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
# 2. Adult 特征列
# ============================================================

STAGE1_NUMERIC_FEATURES = [
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
]

STAGE1_CATEGORICAL_FEATURES = [
    "column", "semantic_type", "detected_type", "main_rule_type", "main_usage_role",
    "pattern_bucket", "rarity_bucket", "neighbor_bucket", "domain_bucket",
    "adult_consistency_bucket", "age_sampling_bucket", "hours_sampling_bucket",
    "adult_value_bucket", "candidate_rule_dominant",
    "time_window_bucket", "time_value_bucket",
]

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

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in [
        "strong_rule_count", "candidate_generation_rule_count", "fd_like_count", "context_rule_count",
        "global_rule_count", "rare_value_count", "typo_rule_count", "pattern_rule_count", "schema_rule_count",
        "adult_domain_rule_count", "adult_format_rule_count", "adult_consistency_rule_count",
        "adult_profile_inconsistency_count", "time_window_rule_count",
    ]:
        out[col] = safe_num_series(out, col, 0.0)

    for col in [
        "conflict_score", "value_frequency", "value_frequency_rank", "neighbor_majority_ratio",
        "prior_error_probability", "posterior_error_probability", "age_lower", "age_upper",
        "hours_lower", "hours_upper", "education_order", "adult_consistency_score",
        "time_value_minutes",
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

    out["adult_rule_total_count"] = (
        out["adult_domain_rule_count"] +
        out["adult_format_rule_count"] +
        out["adult_consistency_rule_count"]
    )
    out["adult_domain_rule_ratio"] = np.where(
        out["adult_rule_total_count"] > 0,
        out["adult_domain_rule_count"] / (out["adult_rule_total_count"] + 1e-8),
        0.0,
    )
    out["adult_format_rule_ratio"] = np.where(
        out["adult_rule_total_count"] > 0,
        out["adult_format_rule_count"] / (out["adult_rule_total_count"] + 1e-8),
        0.0,
    )
    out["adult_consistency_rule_ratio"] = np.where(
        out["adult_rule_total_count"] > 0,
        out["adult_consistency_rule_count"] / (out["adult_rule_total_count"] + 1e-8),
        0.0,
    )

    long_tail_cols = {"country", "occupation", "workclass"}
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
        0.90 * out["adult_domain_rule_count"] +
        0.85 * out["adult_format_rule_count"] +
        0.75 * out["adult_consistency_rule_count"]
    )

    out["candidate_rule_dominant"] = np.where(
        out["candidate_generation_rule_count"] > out["strong_rule_count"],
        "yes",
        "no"
    )

    out["rare_only_signal"] = (
        (out["rare_value_count"] > 0) &
        (out["strong_rule_count"] <= 0) &
        (out["fd_like_count"] <= 0) &
        (out["context_rule_count"] <= 0) &
        (out["typo_rule_count"] <= 0) &
        (out["schema_rule_count"] <= 0) &
        (out["adult_domain_rule_count"] <= 0) &
        (out["adult_format_rule_count"] <= 0) &
        (out["adult_consistency_rule_count"] <= 0)
    ).astype(int)

    out["is_weak_only"] = (
        (out["rare_only_signal"] == 1) |
        (
            (out["strong_rule_count"] <= 0) &
            (out["fd_like_count"] <= 0) &
            (out["typo_rule_count"] <= 0) &
            (out["schema_rule_count"] <= 0) &
            (out["adult_domain_rule_count"] <= 0) &
            (out["adult_format_rule_count"] <= 0) &
            (out["adult_consistency_rule_count"] <= 0) &
            (out["candidate_generation_rule_count"] > 0)
        )
    ).astype(int)

    fp_risk_cols = {"country", "occupation", "workclass", "education", "maritalstatus", "relationship"}
    out["is_fp_risk_col"] = col_series.isin(fp_risk_cols).astype(int)

    out["domain_invalid_flag"] = safe_str_series(out, "domain_bucket", "missing").eq("domain_invalid").astype(int)
    out["adult_consistency_flag"] = safe_str_series(out, "adult_consistency_bucket", "missing").isin([
        "strong_adult_consistency_signal", "adult_consistency_flag", "weak_adult_consistency_signal"
    ]).astype(int)
    out["high_adult_consistency_score"] = (safe_num_series(out, "adult_consistency_score", 0.0) >= 1.0).astype(int)
    out["has_adult_domain_signal"] = (out["adult_domain_rule_count"] > 0).astype(int)
    out["has_adult_format_signal"] = (out["adult_format_rule_count"] > 0).astype(int)
    out["has_adult_consistency_signal"] = (out["adult_consistency_rule_count"] > 0).astype(int)

    out["age_span"] = np.maximum(0.0, out["age_upper"] - out["age_lower"])
    out["hours_span"] = np.maximum(0.0, out["hours_upper"] - out["hours_lower"])
    out["is_minor_age_bucket"] = ((out["age_upper"] > 0) & (out["age_upper"] < 18)).astype(int)
    out["is_high_hours_bucket"] = (((out["hours_lower"] + out["hours_upper"]) / 2.0) > 60).astype(int)

    for col in STAGE1_CATEGORICAL_FEATURES:
        if col not in out.columns:
            out[col] = "missing"
        out[col] = out[col].fillna("missing").astype(str)

    return out


# ============================================================
# 5. 指标与阈值
# ============================================================

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
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


def choose_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_recall: float,
    alpha: float = PRECISION_PRIORITY_ALPHA,
) -> Tuple[float, Dict[str, Any]]:
    candidates = []
    for t in np.arange(THRESHOLD_GRID_START, THRESHOLD_GRID_END + 1e-8, THRESHOLD_GRID_STEP):
        m = compute_metrics(y_true, y_prob, threshold=float(t))
        score = m["f1"] + alpha * m["precision"]
        candidates.append((float(t), m, score))

    valid = [x for x in candidates if x[1]["recall"] >= min_recall]
    if valid:
        valid.sort(key=lambda x: (x[2], x[1]["f1"], x[1]["precision"]), reverse=True)
        return valid[0][0], valid[0][1]

    candidates.sort(key=lambda x: (x[1]["f1"], x[1]["recall"], x[1]["precision"]), reverse=True)
    return candidates[0][0], candidates[0][1]


def compute_final_metrics_with_verifier(
    y_true: np.ndarray,
    detector_pred_label: np.ndarray,
    verifier_keep_prob: np.ndarray,
    threshold: float,
):
    final_pred = detector_pred_label.copy().astype(int)
    mask = detector_pred_label == 1
    final_pred[mask] = (verifier_keep_prob[mask] >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, final_pred, average="binary", zero_division=0
    )
    acc = accuracy_score(y_true, final_pred)
    cm = confusion_matrix(y_true, final_pred).tolist()

    return {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": None,
        "confusion_matrix": cm,
    }


def choose_best_verifier_threshold(
    y_true: np.ndarray,
    detector_pred_label: np.ndarray,
    verifier_keep_prob: np.ndarray,
    min_recall: float = VERIFIER_MIN_FINAL_RECALL,
    alpha: float = PRECISION_PRIORITY_ALPHA,
):
    candidates = []
    for t in np.arange(VERIFIER_THRESHOLD_GRID_START, VERIFIER_THRESHOLD_GRID_END + 1e-8, VERIFIER_THRESHOLD_GRID_STEP):
        m = compute_final_metrics_with_verifier(y_true, detector_pred_label, verifier_keep_prob, float(t))
        score = m["f1"] + alpha * m["precision"]
        candidates.append((float(t), m, score))

    valid = [x for x in candidates if x[1]["recall"] >= min_recall]
    if valid:
        valid.sort(key=lambda x: (x[2], x[1]["f1"], x[1]["precision"]), reverse=True)
        return valid[0][0], valid[0][1]

    candidates.sort(key=lambda x: (x[1]["f1"], x[1]["recall"], x[1]["precision"]), reverse=True)
    return candidates[0][0], candidates[0][1]


# ============================================================
# 6. 数据加载与预处理器
# ============================================================

def load_training_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if LABEL_COL not in df.columns:
        raise ValueError(f"输入文件缺少标签列: {LABEL_COL}")

    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce")
    df = df[df[LABEL_COL].isin([0, 1])].copy()
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    if WEIGHT_COL not in df.columns:
        df[WEIGHT_COL] = 1.0
    df[WEIGHT_COL] = pd.to_numeric(df[WEIGHT_COL], errors="coerce").fillna(1.0)
    df = add_derived_features(df)
    return df


def build_preprocessor(df: pd.DataFrame, numeric_features: List[str], categorical_features: List[str]):
    num_cols = safe_existing_columns(df, numeric_features)
    cat_cols = safe_existing_columns(df, categorical_features)
    feature_cols = num_cols + cat_cols
    if not feature_cols:
        raise ValueError("没有可用特征列。")

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )
    return preprocessor, feature_cols, num_cols, cat_cols


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


def build_pos_weight(y: np.ndarray) -> float:
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    if pos <= 0:
        return 1.0
    return max(1.0, neg / pos)


def weighted_bce_with_logits(logits, targets, sample_weights, pos_weight=None):
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=pos_weight
    )
    return (bce * sample_weights).mean()


def train_one_epoch(model, loader, optimizer, device, pos_weight_tensor):
    model.train()
    total_loss = 0.0
    total_count = 0

    for batch in loader:
        X_batch = batch["X"].to(device)
        y_batch = batch["y"].to(device)
        w_batch = batch["w"].to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = weighted_bce_with_logits(logits, y_batch, w_batch, pos_weight_tensor)
        loss.backward()
        optimizer.step()

        bs = len(X_batch)
        total_loss += float(loss.detach().cpu().item()) * bs
        total_count += bs

    return {"loss": total_loss / max(total_count, 1)}


@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    probs = []
    for batch in loader:
        logits = model(batch["X"].to(device))
        probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs, axis=0) if probs else np.array([])


def save_checkpoint(path, model, optimizer, epoch, best_epoch, best_score, best_threshold, input_dim):
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_threshold": best_threshold,
        "input_dim": input_dim,
    }
    torch.save(ckpt, path)


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


def make_loader(X, y, w, batch_size, shuffle):
    ds = TabularDataset(X=X, y=y, w=w)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


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


# ============================================================
# 7. 模型训练
# ============================================================

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
):
    preprocessor, feature_cols, num_cols, cat_cols = build_preprocessor(train_df, numeric_features, categorical_features)

    X_train = fit_transform_preprocessor(preprocessor, train_df, feature_cols)
    X_val = transform_preprocessor(preprocessor, val_df, feature_cols)

    y_train = train_df[LABEL_COL].values.astype(int)
    y_val = val_df[LABEL_COL].values.astype(int)
    w_train = train_df[WEIGHT_COL].values.astype(float)
    w_val = val_df[WEIGHT_COL].values.astype(float)

    input_dim = X_train.shape[1]

    model = MLPDetector(
        input_dim=input_dim,
        hidden_dim_1=model_hparams["hidden_dim_1"],
        hidden_dim_2=model_hparams["hidden_dim_2"],
        dropout=model_hparams["dropout"],
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    pos_weight = build_pos_weight(y_train) if USE_POS_WEIGHT else 1.0
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

        if AUTO_FIND_BEST_THRESHOLD:
            epoch_best_threshold, epoch_best_metrics = choose_best_threshold(
                y_true=y_val,
                y_prob=val_prob,
                min_recall=target_min_recall,
                alpha=PRECISION_PRIORITY_ALPHA,
            )
        else:
            epoch_best_threshold = 0.5
            epoch_best_metrics = compute_metrics(y_val, val_prob, threshold=0.5)

        epoch_score = epoch_best_metrics["f1"] + PRECISION_PRIORITY_ALPHA * epoch_best_metrics["precision"]

        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "threshold": epoch_best_threshold,
            "val_accuracy": epoch_best_metrics["accuracy"],
            "val_precision": epoch_best_metrics["precision"],
            "val_recall": epoch_best_metrics["recall"],
            "val_f1": epoch_best_metrics["f1"],
            "val_auc": epoch_best_metrics["auc"],
            "score": epoch_score,
        }
        history.append(row)

        save_checkpoint(ckpt_last_path, model, optimizer, epoch, best_epoch, best_score, best_threshold, input_dim)

        if SAVE_EPOCH_CHECKPOINT and (epoch % SAVE_EVERY_EPOCHS == 0):
            epoch_path = ckpt_last_path.replace("_last.pt", f"_epoch_{epoch}.pt")
            save_checkpoint(epoch_path, model, optimizer, epoch, best_epoch, best_score, best_threshold, input_dim)

        improved = epoch_score > best_score + min_delta
        if improved:
            best_score = epoch_score
            best_epoch = epoch
            best_threshold = epoch_best_threshold
            best_metrics = epoch_best_metrics
            stale_epochs = 0
            save_checkpoint(ckpt_best_path, model, optimizer, epoch, best_epoch, best_score, best_threshold, input_dim)
        else:
            stale_epochs += 1

        print(
            f"epoch={epoch:03d} loss={train_stats['loss']:.6f} "
            f"val_p={epoch_best_metrics['precision']:.4f} "
            f"val_r={epoch_best_metrics['recall']:.4f} "
            f"val_f1={epoch_best_metrics['f1']:.4f} th={epoch_best_threshold:.3f}"
        )

        if stale_epochs >= patience:
            print(f"[EarlyStop] epoch={epoch}, best_epoch={best_epoch}")
            break

    pd.DataFrame(history).to_csv(train_log_path, index=False, encoding="utf-8-sig")
    joblib.dump(preprocessor, preprocessor_path)

    # 载入 best 做最终验证输出
    best_ckpt = torch.load(ckpt_best_path, map_location=DEVICE)
    model.load_state_dict(best_ckpt["model_state_dict"])
    val_prob = predict_probs(model, val_loader, DEVICE)

    return {
        "model": model,
        "preprocessor": preprocessor,
        "feature_cols": feature_cols,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "input_dim": input_dim,
        "best_epoch": best_epoch,
        "best_threshold": float(best_threshold),
        "best_metrics": best_metrics,
        "val_prob": val_prob,
        "y_val": y_val,
        "val_index": val_df.index.tolist(),
    }


def make_stage2_training_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if WEIGHT_COL not in out.columns:
        out[WEIGHT_COL] = 1.0
    out[WEIGHT_COL] = pd.to_numeric(out[WEIGHT_COL], errors="coerce").fillna(1.0)

    raw_prob = safe_num_series(out, "raw_pred_error_prob", 0.0)
    raw_label = safe_num_series(out, "raw_pred_label", 0).astype(int)
    ratio = safe_num_series(out, "neighbor_majority_ratio", 0.0)
    eq_neighbor = safe_num_series(out, "is_equal_to_neighbor_majority", 0).astype(int)
    column_series = safe_str_series(out, "column", "missing")

    out["stage1_high_conf_error"] = (raw_prob >= 0.80).astype(int)
    out["stage1_mid_zone"] = ((raw_prob >= MID_PROB_LOW) & (raw_prob <= MID_PROB_HIGH)).astype(int)

    support_current_mask = (
        (eq_neighbor == 1) &
        (ratio >= 0.80) &
        (safe_num_series(out, "strong_rule_count", 0.0) <= 0) &
        (safe_num_series(out, "adult_domain_rule_count", 0.0) <= 0) &
        (safe_num_series(out, "adult_format_rule_count", 0.0) <= 0) &
        (safe_num_series(out, "adult_consistency_rule_count", 0.0) <= 0)
    )
    out["high_prob_but_neighbor_support_current"] = ((raw_prob >= 0.80) & support_current_mask).astype(int)

    out["raw_margin_to_threshold"] = raw_prob - 0.5

    weight_multiplier = np.ones(len(out), dtype=float)
    weight_multiplier *= np.where(raw_label == 1, STAGE2_POS_PRED_WEIGHT_BOOST, 1.0)
    weight_multiplier *= np.where(out["stage1_mid_zone"] == 1, STAGE2_MID_PROB_WEIGHT_BOOST, 1.0)
    weight_multiplier *= np.where(column_series.isin(STAGE2_FALSE_POS_RISK_COLS), STAGE2_RISK_COLUMN_WEIGHT_BOOST, 1.0)
    weight_multiplier *= np.where(support_current_mask, STAGE2_SUPPORT_CURRENT_WEIGHT_BOOST, 1.0)

    rare_only_signal = safe_num_series(out, "rare_only_signal", 0).astype(int)
    weight_multiplier *= np.where(rare_only_signal == 1, STAGE2_RARE_ONLY_WEIGHT_BOOST, 1.0)

    domain_signal = (safe_num_series(out, "adult_domain_rule_count", 0) > 0) | (safe_str_series(out, "domain_bucket", "").eq("domain_invalid"))
    format_signal = safe_num_series(out, "adult_format_rule_count", 0) > 0
    consistency_signal = (safe_num_series(out, "adult_consistency_rule_count", 0) > 0) | (safe_num_series(out, "adult_consistency_score", 0) >= 1.0)

    weight_multiplier *= np.where(domain_signal | format_signal, STAGE2_DOMAIN_SIGNAL_WEIGHT_BOOST, 1.0)
    weight_multiplier *= np.where(consistency_signal, STAGE2_ADULT_CONSISTENCY_WEIGHT_BOOST, 1.0)

    out[WEIGHT_COL] = out[WEIGHT_COL] * weight_multiplier

    return out


def add_stage1_predictions_to_df(stage1_result, train_part: pd.DataFrame, cal_part: pd.DataFrame, final_val_df: pd.DataFrame):
    stage1_model = stage1_result["model"]
    preprocessor = stage1_result["preprocessor"]
    feature_cols = stage1_result["feature_cols"]
    threshold = float(stage1_result["best_threshold"])

    parts = []
    for name, part in [("train", train_part), ("cal", cal_part), ("final_val", final_val_df)]:
        X = transform_preprocessor(preprocessor, part, feature_cols)
        y = part[LABEL_COL].values.astype(int)
        w = part[WEIGHT_COL].values.astype(float)
        loader = make_loader(X, y, w, STAGE1_BATCH_SIZE, shuffle=False)
        prob = predict_probs(stage1_model, loader, DEVICE)

        out = part.copy()
        out["raw_pred_error_prob"] = prob
        out["raw_pred_label"] = (prob >= threshold).astype(int)
        out["raw_margin_to_threshold"] = prob - threshold
        out["stage1_high_conf_error"] = (prob >= 0.80).astype(int)
        out["stage1_mid_zone"] = ((prob >= MID_PROB_LOW) & (prob <= MID_PROB_HIGH)).astype(int)
        out["_split"] = name
        parts.append(out)

    return parts


def load_detector_model(ckpt_path: str, cfg: Dict[str, Any]):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model = MLPDetector(cfg["input_dim"], cfg["hidden_dim_1"], cfg["hidden_dim_2"], cfg["dropout"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def add_stage2_predictions(stage2_result, df: pd.DataFrame, threshold: float):
    model = stage2_result["model"]
    preprocessor = stage2_result["preprocessor"]
    feature_cols = stage2_result["feature_cols"]
    X = transform_preprocessor(preprocessor, df, feature_cols)
    y = df[LABEL_COL].values.astype(int)
    w = df[WEIGHT_COL].values.astype(float)
    loader = make_loader(X, y, w, STAGE2_BATCH_SIZE, shuffle=False)
    prob = predict_probs(model, loader, DEVICE)

    out = df.copy()
    out["detector_pred_error_prob"] = prob
    out["detector_pred_label"] = (prob >= threshold).astype(int)
    out["detector_margin_to_threshold"] = prob - threshold
    out["is_stage_disagree"] = (out["raw_pred_label"].astype(int) != out["detector_pred_label"].astype(int)).astype(int)
    return out


# ============================================================
# 8. verifier
# ============================================================

def build_verifier_train_val(final_val_df: pd.DataFrame, train_aug_df: pd.DataFrame):
    # 用 detector 预测为 error 的样本训练 verifier，目标仍是真实 label_binary。
    # 如果 detector 正样本太少，则用全体候选做后备。
    detector_pos_val = final_val_df[final_val_df["detector_pred_label"] == 1].copy()
    detector_pos_train = train_aug_df[train_aug_df["detector_pred_label"] == 1].copy()

    if len(detector_pos_val) < 20 or detector_pos_val[LABEL_COL].nunique() < 2:
        detector_pos_val = final_val_df.copy()
    if len(detector_pos_train) < 20 or detector_pos_train[LABEL_COL].nunique() < 2:
        detector_pos_train = train_aug_df.copy()

    return detector_pos_train, detector_pos_val


def build_verifier_weights(df: pd.DataFrame) -> np.ndarray:
    y = df[LABEL_COL].astype(int).values
    weights = np.ones(len(df), dtype=float)

    detector_prob = safe_num_series(df, "detector_pred_error_prob", 0.0)
    border = np.abs(detector_prob - 0.5) <= 0.10
    weights *= np.where(border, VERIFIER_BORDERLINE_FP_BOOST, 1.0)

    fd_conflict = safe_num_series(df, "fd_like_count", 0.0) > 0
    rare_only = safe_num_series(df, "rare_only_signal", 0).astype(int) == 1
    risk_col = safe_num_series(df, "is_fp_risk_col", 0).astype(int) == 1
    weak_other = safe_str_series(df, "neighbor_bucket", "").isin({"weak_support_other", "medium_support_other"})

    adult_domain = (safe_num_series(df, "adult_domain_rule_count", 0.0) > 0) | safe_str_series(df, "domain_bucket", "").eq("domain_invalid")
    adult_format = safe_num_series(df, "adult_format_rule_count", 0.0) > 0
    adult_cons = (safe_num_series(df, "adult_consistency_rule_count", 0.0) > 0) | (safe_num_series(df, "adult_consistency_score", 0.0) >= 1.0)

    # y=0 是 verifier 要学会打回的 FP，y=1 是要保留的真错误
    weights *= np.where((y == 0), VERIFIER_FP_BASE_WEIGHT_BOOST, 1.0)
    weights *= np.where((y == 0) & fd_conflict, VERIFIER_FD_CONFLICT_FP_BOOST, 1.0)
    weights *= np.where((y == 0) & rare_only, VERIFIER_RARE_ONLY_FP_BOOST, 1.0)
    weights *= np.where((y == 0) & risk_col, VERIFIER_RISK_COLUMN_FP_BOOST, 1.0)
    weights *= np.where((y == 0) & weak_other, VERIFIER_WEAK_OTHER_FP_BOOST, 1.0)

    weights *= np.where((y == 1), VERIFIER_SAMPLE_TRUE_ERROR_BOOST, 1.0)
    weights *= np.where((y == 1) & adult_domain, VERIFIER_DOMAIN_TRUE_ERROR_BOOST, 1.0)
    weights *= np.where((y == 1) & adult_format, VERIFIER_FORMAT_TRUE_ERROR_BOOST, 1.0)
    weights *= np.where((y == 1) & adult_cons, VERIFIER_CONSISTENCY_TRUE_ERROR_BOOST, 1.0)

    return weights


def train_verifier(verifier_train_df: pd.DataFrame, verifier_val_df: pd.DataFrame):
    preprocessor, feature_cols, num_cols, cat_cols = build_preprocessor(
        verifier_train_df, VERIFIER_NUMERIC_FEATURES, VERIFIER_CATEGORICAL_FEATURES
    )
    X_train = fit_transform_preprocessor(preprocessor, verifier_train_df, feature_cols)
    X_val = transform_preprocessor(preprocessor, verifier_val_df, feature_cols)

    y_train = verifier_train_df[LABEL_COL].astype(int).values
    y_val = verifier_val_df[LABEL_COL].astype(int).values
    weights = build_verifier_weights(verifier_train_df)

    if len(np.unique(y_train)) < 2:
        verifier = DummyClassifier(strategy="constant", constant=int(y_train[0]) if len(y_train) else 1)
        verifier.fit(X_train, y_train)
    else:
        verifier = RandomForestClassifier(
            n_estimators=VERIFIER_N_ESTIMATORS,
            max_depth=VERIFIER_MAX_DEPTH,
            min_samples_leaf=VERIFIER_MIN_SAMPLES_LEAF,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight=VERIFIER_CLASS_WEIGHT,
        )
        verifier.fit(X_train, y_train, sample_weight=weights)

    keep_prob = predict_verifier_keep_prob(verifier, X_val)
    detector_pred = verifier_val_df["detector_pred_label"].astype(int).values

    best_th, best_metrics = choose_best_verifier_threshold(
        y_true=y_val,
        detector_pred_label=detector_pred,
        verifier_keep_prob=keep_prob,
        min_recall=VERIFIER_MIN_FINAL_RECALL,
        alpha=PRECISION_PRIORITY_ALPHA,
    )

    verifier_train_df.to_csv(VERIFIER_TRAIN_CSV, index=False, encoding="utf-8-sig")
    verifier_val_df.to_csv(VERIFIER_VAL_CSV, index=False, encoding="utf-8-sig")

    joblib.dump(verifier, VERIFIER_MODEL_PATH)
    joblib.dump(preprocessor, VERIFIER_PREPROCESSOR_PATH)

    cfg = VerifierConfig(
        numeric_features=num_cols,
        categorical_features=cat_cols,
        feature_cols=feature_cols,
        threshold=float(best_th),
        metrics=best_metrics,
    )
    save_json(cfg, VERIFIER_CONFIG_PATH)
    save_json({
        "best_verifier_threshold_on_final_val": float(best_th),
        "best_metrics": best_metrics,
        "verifier_val_size": int(len(verifier_val_df)),
        "verifier_train_size": int(len(verifier_train_df)),
    }, VERIFIER_METRICS_JSON)

    return {
        "model": verifier,
        "preprocessor": preprocessor,
        "feature_cols": feature_cols,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "best_threshold": float(best_th),
        "best_metrics": best_metrics,
        "val_keep_prob": keep_prob,
    }


def predict_verifier_keep_prob(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "classes_"):
        classes = list(model.classes_)
        if len(classes) == 1:
            return np.full(X.shape[0], float(int(classes[0])), dtype=float)
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X)
            class_to_index = {c: i for i, c in enumerate(classes)}
            if 1 in class_to_index:
                return prob[:, class_to_index[1]]
            return np.zeros(X.shape[0], dtype=float)

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)
        if prob.ndim == 2:
            return prob[:, 1] if prob.shape[1] >= 2 else prob[:, 0]

    pred = model.predict(X)
    return np.asarray(pred, dtype=float)


# ============================================================
# 9. 主训练流程
# ============================================================

def main():
    set_seed(RANDOM_STATE)
    ensure_dir(OUTPUT_DIR)

    df = load_training_data(INPUT_CSV)
    if len(df) < 10:
        raise RuntimeError(f"训练样本太少: {len(df)}")

    print("===== Dataset Stats =====")
    print(f"rows={len(df)}")
    print(df[LABEL_COL].value_counts(dropna=False).to_string())
    print(df["label_source"].value_counts(dropna=False).to_string() if "label_source" in df.columns else "")

    stratify = df[LABEL_COL] if df[LABEL_COL].nunique() == 2 and df[LABEL_COL].value_counts().min() >= 2 else None
    rest_df, final_val_df = train_test_split(
        df,
        test_size=FINAL_VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    stratify_rest = rest_df[LABEL_COL] if rest_df[LABEL_COL].nunique() == 2 and rest_df[LABEL_COL].value_counts().min() >= 2 else None
    train_df, cal_df = train_test_split(
        rest_df,
        test_size=CAL_TRAIN_SIZE_IN_REST,
        random_state=RANDOM_STATE,
        stratify=stratify_rest,
    )

    print("===== Split Stats =====")
    print(f"train={len(train_df)}, cal={len(cal_df)}, final_val={len(final_val_df)}")

    if DETECTOR_TRAIN_MODE == "freeze_reuse":
        print("[INFO] DETECTOR_TRAIN_MODE=freeze_reuse, copying detector artifacts...")
        copy_detector_artifacts(DETECTOR_SOURCE_DIR, OUTPUT_DIR)
        cfg = load_json(CONFIG_PATH)
        metrics = load_json(METRICS_JSON)

        stage1_cfg = cfg["stage1"]
        stage1_preprocessor = joblib.load(STAGE1_PREPROCESSOR)
        stage1_model = load_detector_model(STAGE1_BEST_CKPT, stage1_cfg)
        stage1_result = {
            "model": stage1_model,
            "preprocessor": stage1_preprocessor,
            "feature_cols": stage1_cfg["feature_cols"],
            "best_threshold": metrics["stage1_best_threshold"],
        }
    else:
        print("[INFO] Training stage1...")
        stage1_result = run_single_stage_training(
            train_df=train_df,
            val_df=cal_df,
            numeric_features=STAGE1_NUMERIC_FEATURES,
            categorical_features=STAGE1_CATEGORICAL_FEATURES,
            model_hparams={
                "hidden_dim_1": STAGE1_HIDDEN_DIM_1,
                "hidden_dim_2": STAGE1_HIDDEN_DIM_2,
                "dropout": STAGE1_DROPOUT,
            },
            batch_size=STAGE1_BATCH_SIZE,
            lr=STAGE1_LR,
            weight_decay=STAGE1_WEIGHT_DECAY,
            max_epochs=STAGE1_MAX_EPOCHS,
            patience=STAGE1_PATIENCE,
            min_delta=STAGE1_MIN_DELTA,
            target_min_recall=TARGET_MIN_RECALL_STAGE1,
            ckpt_last_path=STAGE1_LAST_CKPT,
            ckpt_best_path=STAGE1_BEST_CKPT,
            preprocessor_path=STAGE1_PREPROCESSOR,
            train_log_path=STAGE1_TRAIN_LOG,
        )

    train_part, cal_part, final_val_part = add_stage1_predictions_to_df(stage1_result, train_df, cal_df, final_val_df)
    train_stage2_df = make_stage2_training_df(pd.concat([train_part, cal_part], ignore_index=True))
    final_val_stage2_base = make_stage2_training_df(final_val_part)

    if DETECTOR_TRAIN_MODE == "freeze_reuse":
        cfg = load_json(CONFIG_PATH)
        metrics = load_json(METRICS_JSON)
        stage2_cfg = cfg["stage2"]
        stage2_preprocessor = joblib.load(STAGE2_PREPROCESSOR)
        stage2_model = load_detector_model(STAGE2_BEST_CKPT, stage2_cfg)
        stage2_result = {
            "model": stage2_model,
            "preprocessor": stage2_preprocessor,
            "feature_cols": stage2_cfg["feature_cols"],
            "best_threshold": metrics["stage2_best_threshold"],
        }
    else:
        print("[INFO] Training stage2...")
        stage2_result = run_single_stage_training(
            train_df=train_stage2_df,
            val_df=final_val_stage2_base,
            numeric_features=STAGE2_NUMERIC_FEATURES,
            categorical_features=STAGE2_CATEGORICAL_FEATURES,
            model_hparams={
                "hidden_dim_1": STAGE2_HIDDEN_DIM_1,
                "hidden_dim_2": STAGE2_HIDDEN_DIM_2,
                "dropout": STAGE2_DROPOUT,
            },
            batch_size=STAGE2_BATCH_SIZE,
            lr=STAGE2_LR,
            weight_decay=STAGE2_WEIGHT_DECAY,
            max_epochs=STAGE2_MAX_EPOCHS,
            patience=STAGE2_PATIENCE,
            min_delta=STAGE2_MIN_DELTA,
            target_min_recall=TARGET_MIN_RECALL_STAGE2,
            ckpt_last_path=STAGE2_LAST_CKPT,
            ckpt_best_path=STAGE2_BEST_CKPT,
            preprocessor_path=STAGE2_PREPROCESSOR,
            train_log_path=STAGE2_TRAIN_LOG,
        )

    stage2_threshold = float(stage2_result["best_threshold"])
    train_aug_pred = add_stage2_predictions(stage2_result, train_stage2_df, stage2_threshold)
    final_val_pred = add_stage2_predictions(stage2_result, final_val_stage2_base, stage2_threshold)

    train_aug_pred.to_csv(STAGE2_TRAIN_ANALYSIS_CSV, index=False, encoding="utf-8-sig")

    y_val = final_val_pred[LABEL_COL].astype(int).values
    detector_prob = final_val_pred["detector_pred_error_prob"].values
    raw_prob = final_val_pred["raw_pred_error_prob"].values

    stage1_metrics = compute_metrics(y_val, raw_prob, threshold=float(stage1_result["best_threshold"]))
    stage2_metrics = compute_metrics(y_val, detector_prob, threshold=stage2_threshold)

    verifier_train_df, verifier_val_df = build_verifier_train_val(final_val_pred, train_aug_pred)

    print("[INFO] Training verifier...")
    verifier_result = train_verifier(verifier_train_df, verifier_val_df)

    Xv = transform_preprocessor(verifier_result["preprocessor"], final_val_pred, verifier_result["feature_cols"])
    val_keep_prob = predict_verifier_keep_prob(verifier_result["model"], Xv)
    final_metrics_with_verifier = compute_final_metrics_with_verifier(
        y_true=y_val,
        detector_pred_label=final_val_pred["detector_pred_label"].astype(int).values,
        verifier_keep_prob=val_keep_prob,
        threshold=float(verifier_result["best_threshold"]),
    )

    final_val_pred["verifier_keep_error_prob"] = val_keep_prob
    final_val_pred["final_pred_label"] = final_val_pred["detector_pred_label"].astype(int)
    mask = final_val_pred["detector_pred_label"].astype(int) == 1
    final_val_pred.loc[mask, "final_pred_label"] = (
        final_val_pred.loc[mask, "verifier_keep_error_prob"] >= float(verifier_result["best_threshold"])
    ).astype(int)
    final_val_pred.to_csv(FINAL_VAL_PRED_CSV, index=False, encoding="utf-8-sig")

    stage1_config = StageConfig(
        input_dim=int(stage1_result.get("input_dim", load_json(CONFIG_PATH)["stage1"]["input_dim"] if os.path.exists(CONFIG_PATH) else 0)),
        hidden_dim_1=STAGE1_HIDDEN_DIM_1,
        hidden_dim_2=STAGE1_HIDDEN_DIM_2,
        dropout=STAGE1_DROPOUT,
        batch_size=STAGE1_BATCH_SIZE,
        learning_rate=STAGE1_LR,
        weight_decay=STAGE1_WEIGHT_DECAY,
        max_epochs=STAGE1_MAX_EPOCHS,
        patience=STAGE1_PATIENCE,
        numeric_features=safe_existing_columns(df, STAGE1_NUMERIC_FEATURES),
        categorical_features=safe_existing_columns(df, STAGE1_CATEGORICAL_FEATURES),
        feature_cols=stage1_result["feature_cols"],
    )

    stage2_config = StageConfig(
        input_dim=int(stage2_result.get("input_dim", load_json(CONFIG_PATH)["stage2"]["input_dim"] if os.path.exists(CONFIG_PATH) else 0)),
        hidden_dim_1=STAGE2_HIDDEN_DIM_1,
        hidden_dim_2=STAGE2_HIDDEN_DIM_2,
        dropout=STAGE2_DROPOUT,
        batch_size=STAGE2_BATCH_SIZE,
        learning_rate=STAGE2_LR,
        weight_decay=STAGE2_WEIGHT_DECAY,
        max_epochs=STAGE2_MAX_EPOCHS,
        patience=STAGE2_PATIENCE,
        numeric_features=safe_existing_columns(train_stage2_df, STAGE2_NUMERIC_FEATURES),
        categorical_features=safe_existing_columns(train_stage2_df, STAGE2_CATEGORICAL_FEATURES),
        feature_cols=stage2_result["feature_cols"],
    )

    verifier_config = VerifierConfig(
        numeric_features=verifier_result["num_cols"],
        categorical_features=verifier_result["cat_cols"],
        feature_cols=verifier_result["feature_cols"],
        threshold=float(verifier_result["best_threshold"]),
        metrics=verifier_result["best_metrics"],
    )

    config = TwoStageConfig(
        device=DEVICE,
        random_state=RANDOM_STATE,
        label_col=LABEL_COL,
        weight_col=WEIGHT_COL,
        final_val_size=FINAL_VAL_SIZE,
        cal_train_size_in_rest=CAL_TRAIN_SIZE_IN_REST,
        stage1=stage1_config,
        stage2=stage2_config,
        verifier=verifier_config,
    )
    save_json(config, CONFIG_PATH)

    metrics = {
        "stage1_best_threshold": float(stage1_result["best_threshold"]),
        "stage2_best_threshold": float(stage2_threshold),
        "verifier_best_threshold": float(verifier_result["best_threshold"]),
        "stage1_metrics": stage1_metrics,
        "stage2_metrics": stage2_metrics,
        "verifier_metrics": verifier_result["best_metrics"],
        "final_metrics_with_verifier": final_metrics_with_verifier,
        "final_metrics": final_metrics_with_verifier,
        "split_sizes": {
            "train": int(len(train_df)),
            "cal": int(len(cal_df)),
            "final_val": int(len(final_val_df)),
            "verifier_train": int(len(verifier_train_df)),
            "verifier_val": int(len(verifier_val_df)),
        }
    }
    save_json(metrics, METRICS_JSON)

    print("\n===== DONE =====")
    print(f"Model dir: {OUTPUT_DIR}")
    print(f"Stage1: P={stage1_metrics['precision']:.4f} R={stage1_metrics['recall']:.4f} F1={stage1_metrics['f1']:.4f}")
    print(f"Stage2: P={stage2_metrics['precision']:.4f} R={stage2_metrics['recall']:.4f} F1={stage2_metrics['f1']:.4f}")
    print(
        f"Final+Verifier: P={final_metrics_with_verifier['precision']:.4f} "
        f"R={final_metrics_with_verifier['recall']:.4f} "
        f"F1={final_metrics_with_verifier['f1']:.4f}"
    )


if __name__ == "__main__":
    main()

import json
import os
import random
import shutil
from dataclasses import dataclass, asdict, is_dataclass
from typing import List, Dict, Any, Tuple

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

INPUT_CSV = "final_train_dataset_simple_v2.csv"
OUTPUT_DIR = "two_stage_model_output_v4_with_verifier"

# full: 正常训练 detector + verifier
# freeze_reuse: 复用已有 detector，仅重训 verifier
DETECTOR_TRAIN_MODE = "full"

# 当 DETECTOR_TRAIN_MODE = "freeze_reuse" 时使用
DETECTOR_SOURCE_DIR = "two_stage_model_output_v4_with_verifier"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_STATE = 42

LABEL_COL = "label_binary"
WEIGHT_COL = "sample_weight"

FINAL_VAL_SIZE = 0.20
CAL_TRAIN_SIZE_IN_REST = 0.25

SAVE_EPOCH_CHECKPOINT = True
SAVE_EVERY_EPOCHS = 5
AUTO_FIND_BEST_THRESHOLD = True

# detector 阈值偏好
TARGET_MIN_RECALL_STAGE1 = 0.97
TARGET_MIN_RECALL_STAGE2 = 0.95

THRESHOLD_GRID_START = 0.10
THRESHOLD_GRID_END = 0.95
THRESHOLD_GRID_STEP = 0.02
PRECISION_PRIORITY_ALPHA = 0.18

# verifier 阈值搜索
VERIFIER_THRESHOLD_GRID_START = 0.10
VERIFIER_THRESHOLD_GRID_END = 0.95
VERIFIER_THRESHOLD_GRID_STEP = 0.02
VERIFIER_MIN_FINAL_RECALL = 0.97

# detector loss
USE_POS_WEIGHT = True

# Stage2 保持温和，不要把 detector 带偏
STAGE2_POS_PRED_WEIGHT_BOOST = 1.0
STAGE2_MID_PROB_WEIGHT_BOOST = 1.0
STAGE2_FALSE_POS_RISK_COLS = {"EmergencyService", "Sample", "ZipCode", "HospitalName"}
STAGE2_RISK_COLUMN_WEIGHT_BOOST = 1.0
STAGE2_SUPPORT_CURRENT_WEIGHT_BOOST = 1.0
STAGE2_RARE_ONLY_WEIGHT_BOOST = 1.0

MID_PROB_LOW = 0.35
MID_PROB_HIGH = 0.75

# verifier 超参数
VERIFIER_N_ESTIMATORS = 300
VERIFIER_MAX_DEPTH = 8
VERIFIER_MIN_SAMPLES_LEAF = 10
VERIFIER_CLASS_WEIGHT = "balanced_subsample"

# verifier 对高价值 FP 的加权
VERIFIER_FP_BASE_WEIGHT_BOOST = 1.2
VERIFIER_FD_CONFLICT_FP_BOOST = 2.5
VERIFIER_RARE_ONLY_FP_BOOST = 1.4
VERIFIER_BORDERLINE_FP_BOOST = 1.4
VERIFIER_RISK_COLUMN_FP_BOOST = 1.8
VERIFIER_SAMPLE_TRUE_ERROR_BOOST = 1.45

# ============================================================
# 2. 路径
# ============================================================

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
# 3. 特征配置
# ============================================================

STAGE1_NUMERIC_FEATURES = [
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

STAGE1_CATEGORICAL_FEATURES = [
    "column", "semantic_type", "main_rule_type", "main_usage_role",
    "pattern_bucket", "rarity_bucket", "neighbor_bucket", "candidate_rule_dominant",
]

STAGE2_NUMERIC_FEATURES = [
    "raw_pred_error_prob", "raw_pred_label", "raw_margin_to_threshold",
    "stage1_high_conf_error", "stage1_mid_zone",
    "violation_count", "conflict_score", "value_frequency", "value_frequency_rank",
    "neighbor_majority_ratio", "prior_error_probability", "posterior_error_probability",
    "is_equal_to_neighbor_majority", "neighbor_agreement_score",
    "strong_rule_count", "candidate_generation_rule_count", "fd_like_count",
    "context_rule_count", "global_rule_count", "rare_value_count", "typo_rule_count",
    "pattern_rule_count", "schema_rule_count", "candidate_rule_ratio", "rule_total_count",
    "strong_rule_ratio", "context_rule_ratio", "fd_like_ratio", "global_rule_ratio",
    "rule_strength_sum", "has_neighbor_signal", "high_neighbor_consistency",
    "is_rare_value", "neighbor_disagree", "rare_only_signal",
]

STAGE2_CATEGORICAL_FEATURES = [
    "column", "semantic_type", "main_rule_type", "main_usage_role",
    "pattern_bucket", "rarity_bucket", "neighbor_bucket", "candidate_rule_dominant",
]

VERIFIER_NUMERIC_FEATURES = [
    "raw_pred_error_prob", "raw_pred_label", "raw_margin_to_threshold",
    "stage1_high_conf_error", "stage1_mid_zone",
    "detector_pred_error_prob", "detector_pred_label", "detector_margin_to_threshold",
    "is_stage_disagree",
    "violation_count", "conflict_score", "value_frequency", "value_frequency_rank",
    "neighbor_majority_ratio", "prior_error_probability", "posterior_error_probability",
    "is_equal_to_neighbor_majority", "neighbor_agreement_score",
    "strong_rule_count", "candidate_generation_rule_count", "fd_like_count",
    "context_rule_count", "global_rule_count", "rare_value_count", "typo_rule_count",
    "pattern_rule_count", "schema_rule_count", "candidate_rule_ratio", "rule_total_count",
    "strong_rule_ratio", "context_rule_ratio", "fd_like_ratio", "global_rule_ratio",
    "rule_strength_sum", "has_neighbor_signal", "high_neighbor_consistency",
    "is_rare_value", "neighbor_disagree", "rare_only_signal", "is_weak_only", "is_fp_risk_col",
]

VERIFIER_CATEGORICAL_FEATURES = [
    "column", "semantic_type", "main_rule_type", "main_usage_role",
    "pattern_bucket", "rarity_bucket", "neighbor_bucket", "candidate_rule_dominant",
]


# ============================================================
# 4. detector 超参数
# ============================================================

STAGE1_BATCH_SIZE = 64
STAGE1_MAX_EPOCHS = 50
STAGE1_LR = 1e-3
STAGE1_WEIGHT_DECAY = 1e-4
STAGE1_PATIENCE = 10
STAGE1_MIN_DELTA = 1e-4
STAGE1_HIDDEN_DIM_1 = 128
STAGE1_HIDDEN_DIM_2 = 64
STAGE1_DROPOUT = 0.2

STAGE2_BATCH_SIZE = 64
STAGE2_MAX_EPOCHS = 50
STAGE2_LR = 8e-4
STAGE2_WEIGHT_DECAY = 1e-4
STAGE2_PATIENCE = 10
STAGE2_MIN_DELTA = 1e-4
STAGE2_HIDDEN_DIM_1 = 64
STAGE2_HIDDEN_DIM_2 = 32
STAGE2_DROPOUT = 0.2


# ============================================================
# 5. 基础工具
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


def save_json(obj, path: str):
    def convert_obj(o):
        if is_dataclass(o):
            return asdict(o)
        if isinstance(o, dict):
            return {k: convert_obj(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [convert_obj(x) for x in o]
        return o

    with open(path, "w", encoding="utf-8") as f:
        json.dump(convert_obj(obj), f, ensure_ascii=False, indent=2)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_existing_columns(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def normalize_text_series_for_compare(s: pd.Series) -> pd.Series:
    return s.fillna("__NULL__").astype(str).str.strip()


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


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "value" in out.columns and "neighbor_majority_value" in out.columns:
        value_series = normalize_text_series_for_compare(out["value"])
        neighbor_series = normalize_text_series_for_compare(out["neighbor_majority_value"])
        out["is_equal_to_neighbor_majority"] = (value_series == neighbor_series).astype(int)
    else:
        out["is_equal_to_neighbor_majority"] = 0

    ratio = pd.to_numeric(out.get("neighbor_majority_ratio", 0.0), errors="coerce").fillna(0.0)
    out["neighbor_majority_ratio"] = ratio
    out["neighbor_agreement_score"] = np.where(
        out["is_equal_to_neighbor_majority"] == 1,
        ratio,
        -ratio
    ).astype(float)

    for col in [
        "strong_rule_count",
        "candidate_generation_rule_count",
        "fd_like_count",
        "context_rule_count",
        "global_rule_count",
        "rare_value_count",
        "typo_rule_count",
        "pattern_rule_count",
        "schema_rule_count",
    ]:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

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

    out["has_neighbor_signal"] = (ratio > 0).astype(int)
    out["high_neighbor_consistency"] = (ratio >= 0.8).astype(int)
    out["is_rare_value"] = pd.to_numeric(out.get("value_frequency_rank", 999999), errors="coerce").fillna(999999).gt(10).astype(int)
    out["neighbor_disagree"] = ((out["has_neighbor_signal"] == 1) & (out["is_equal_to_neighbor_majority"] == 0)).astype(int)

    out["rule_strength_sum"] = (
        1.0 * out["strong_rule_count"] +
        0.25 * out["candidate_generation_rule_count"] +
        0.4 * out["fd_like_count"] +
        0.25 * out["context_rule_count"] +
        0.15 * out["global_rule_count"] +
        0.5 * out["typo_rule_count"] +
        0.15 * out["pattern_rule_count"] +
        0.25 * out["schema_rule_count"]
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
        (out["schema_rule_count"] <= 0)
    ).astype(int)

    out["is_weak_only"] = (
        (out["rare_only_signal"] == 1) |
        (
            (out["strong_rule_count"] <= 0) &
            (out["fd_like_count"] <= 0) &
            (out["typo_rule_count"] <= 0) &
            (out["schema_rule_count"] <= 0) &
            (out["candidate_generation_rule_count"] > 0)
        )
    ).astype(int)

    risk_cols = {
        "EmergencyService", "Sample", "ZipCode", "HospitalName",
        "ProviderNumber", "PhoneNumber", "Stateavg", "Address1", "Score"
    }
    out["is_fp_risk_col"] = out.get("column", pd.Series(["missing"] * len(out))).astype(str).isin(risk_cols).astype(int)
    return out


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
    for t in np.arange(
        VERIFIER_THRESHOLD_GRID_START,
        VERIFIER_THRESHOLD_GRID_END + 1e-8,
        VERIFIER_THRESHOLD_GRID_STEP,
    ):
        m = compute_final_metrics_with_verifier(
            y_true=y_true,
            detector_pred_label=detector_pred_label,
            verifier_keep_prob=verifier_keep_prob,
            threshold=float(t),
        )
        score = m["f1"] + alpha * m["precision"]
        candidates.append((float(t), m, score))

    valid = [x for x in candidates if x[1]["recall"] >= min_recall]
    if valid:
        valid.sort(key=lambda x: (x[2], x[1]["f1"], x[1]["precision"]), reverse=True)
        return valid[0][0], valid[0][1]

    candidates.sort(key=lambda x: (x[1]["f1"], x[1]["recall"], x[1]["precision"]), reverse=True)
    return candidates[0][0], candidates[0][1]


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
        logits,
        targets,
        reduction="none",
        pos_weight=pos_weight
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
        loss = weighted_bce_with_logits(
            logits=logits,
            targets=y_batch,
            sample_weights=w_batch,
            pos_weight=pos_weight_tensor,
        )
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


def load_checkpoint(path: str, device: str):
    return torch.load(path, map_location=device)


def make_stage2_training_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if WEIGHT_COL not in out.columns:
        out[WEIGHT_COL] = 1.0
    out[WEIGHT_COL] = pd.to_numeric(out[WEIGHT_COL], errors="coerce").fillna(1.0)

    raw_prob = pd.to_numeric(out.get("raw_pred_error_prob", 0.0), errors="coerce").fillna(0.0)
    raw_label = pd.to_numeric(out.get("raw_pred_label", 0), errors="coerce").fillna(0).astype(int)
    ratio = pd.to_numeric(out.get("neighbor_majority_ratio", 0.0), errors="coerce").fillna(0.0)
    eq_neighbor = pd.to_numeric(out.get("is_equal_to_neighbor_majority", 0), errors="coerce").fillna(0).astype(int)
    rare_only = pd.to_numeric(out.get("rare_only_signal", 0), errors="coerce").fillna(0).astype(int)
    col_series = out.get("column", pd.Series(["__UNK__"] * len(out), index=out.index)).astype(str)

    out.loc[raw_label == 1, WEIGHT_COL] *= STAGE2_POS_PRED_WEIGHT_BOOST

    mid_mask = (raw_prob >= MID_PROB_LOW) & (raw_prob <= MID_PROB_HIGH)
    out.loc[mid_mask, WEIGHT_COL] *= STAGE2_MID_PROB_WEIGHT_BOOST

    risk_col_mask = col_series.isin(STAGE2_FALSE_POS_RISK_COLS)
    out.loc[risk_col_mask, WEIGHT_COL] *= STAGE2_RISK_COLUMN_WEIGHT_BOOST

    support_current_mask = (eq_neighbor == 1) & (ratio >= 0.80)
    out.loc[support_current_mask, WEIGHT_COL] *= STAGE2_SUPPORT_CURRENT_WEIGHT_BOOST

    out.loc[rare_only == 1, WEIGHT_COL] *= STAGE2_RARE_ONLY_WEIGHT_BOOST
    return out


def run_stage_training(
    stage_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
    batch_size: int,
    max_epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    min_delta: float,
    hidden_dim_1: int,
    hidden_dim_2: int,
    dropout: float,
    preprocessor_path: str,
    last_ckpt_path: str,
    best_ckpt_path: str,
    train_log_csv: str,
    min_recall_target: float,
):
    ensure_dir(os.path.dirname(preprocessor_path))
    ensure_dir(os.path.dirname(last_ckpt_path))
    ensure_dir(os.path.dirname(best_ckpt_path))
    ensure_dir(os.path.dirname(train_log_csv))

    preprocessor, feature_cols, _, _ = build_preprocessor(
        train_df, numeric_features, categorical_features
    )
    preprocessor.fit(train_df[feature_cols])
    joblib.dump(preprocessor, preprocessor_path)

    X_train = preprocessor.transform(train_df[feature_cols])
    X_val = preprocessor.transform(val_df[feature_cols])
    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
    if hasattr(X_val, "toarray"):
        X_val = X_val.toarray()

    y_train = train_df[LABEL_COL].values.astype(np.float32)
    y_val = val_df[LABEL_COL].values.astype(np.float32)
    w_train = train_df[WEIGHT_COL].values.astype(np.float32)
    w_val = val_df[WEIGHT_COL].values.astype(np.float32)

    train_dataset = TabularDataset(X_train, y_train, w_train)
    val_dataset = TabularDataset(X_val, y_val, w_val)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    input_dim = X_train.shape[1]
    model = MLPDetector(input_dim, hidden_dim_1, hidden_dim_2, dropout).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_score = -1e9
    best_epoch = -1
    best_threshold = 0.5
    best_metrics = None
    patience_count = 0
    best_model_state = None
    train_logs = []

    pos_weight_value = build_pos_weight(y_train) if USE_POS_WEIGHT else 1.0
    pos_weight_tensor = torch.tensor(pos_weight_value, dtype=torch.float32, device=DEVICE)

    for epoch in range(1, max_epochs + 1):
        train_stat = train_one_epoch(model, train_loader, optimizer, DEVICE, pos_weight_tensor)
        val_prob = predict_probs(model, val_loader, DEVICE)

        if AUTO_FIND_BEST_THRESHOLD:
            threshold, metrics = choose_best_threshold(y_val, val_prob, min_recall=min_recall_target)
        else:
            threshold = 0.5
            metrics = compute_metrics(y_val, val_prob, threshold)

        current_score = metrics["f1"] + PRECISION_PRIORITY_ALPHA * metrics["precision"]

        train_logs.append({
            "epoch": epoch,
            "train_loss": train_stat["loss"],
            "val_accuracy": metrics["accuracy"],
            "val_precision": metrics["precision"],
            "val_recall": metrics["recall"],
            "val_f1": metrics["f1"],
            "val_auc": metrics["auc"],
            "threshold": threshold,
            "selection_score": current_score,
            "min_recall_target": min_recall_target,
        })

        print(
            f"[{stage_name}] Epoch {epoch:03d} | "
            f"loss={train_stat['loss']:.6f} | "
            f"precision={metrics['precision']:.4f} | "
            f"recall={metrics['recall']:.4f} | "
            f"f1={metrics['f1']:.4f} | "
            f"threshold={threshold:.3f}"
        )

        save_checkpoint(last_ckpt_path, model, optimizer, epoch, best_epoch, best_score, best_threshold, input_dim)

        if SAVE_EPOCH_CHECKPOINT and (epoch % SAVE_EVERY_EPOCHS == 0):
            epoch_ckpt_path = os.path.join(OUTPUT_DIR, f"{stage_name.lower()}_epoch_{epoch}.pt")
            ensure_dir(os.path.dirname(epoch_ckpt_path))
            save_checkpoint(epoch_ckpt_path, model, optimizer, epoch, best_epoch, best_score, best_threshold, input_dim)

        if current_score > best_score + min_delta:
            best_score = current_score
            best_epoch = epoch
            best_threshold = threshold
            best_metrics = metrics
            patience_count = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            save_checkpoint(best_ckpt_path, model, optimizer, epoch, best_epoch, best_score, best_threshold, input_dim)
        else:
            patience_count += 1
            if patience_count >= patience:
                break

    if best_model_state is None:
        raise RuntimeError(f"[{stage_name}] 训练失败。")

    model.load_state_dict(best_model_state)
    pd.DataFrame(train_logs).to_csv(train_log_csv, index=False, encoding="utf-8-sig")

    return {
        "model": model,
        "preprocessor": preprocessor,
        "feature_cols": feature_cols,
        "best_epoch": best_epoch,
        "best_threshold": best_threshold,
        "best_metrics": best_metrics,
    }


def attach_stage1_outputs(df, stage1_model, stage1_preprocessor, stage1_feature_cols, stage1_threshold: float):
    out = df.copy()
    X = stage1_preprocessor.transform(out[stage1_feature_cols])
    if hasattr(X, "toarray"):
        X = X.toarray()

    dataset = TabularDataset(
        X=X,
        y=np.zeros(len(out), dtype=np.float32),
        w=np.ones(len(out), dtype=np.float32),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=False)

    raw_prob = predict_probs(stage1_model, loader, DEVICE)
    out["raw_pred_error_prob"] = raw_prob
    out["raw_pred_label"] = (out["raw_pred_error_prob"] >= float(stage1_threshold)).astype(int)
    out["raw_pred_label_name"] = out["raw_pred_label"].map({1: "error", 0: "correct"})
    out["stage1_decision_threshold"] = float(stage1_threshold)
    out["raw_margin_to_threshold"] = out["raw_pred_error_prob"] - float(stage1_threshold)
    out["stage1_high_conf_error"] = (out["raw_pred_error_prob"] >= max(float(stage1_threshold), 0.80)).astype(int)
    out["stage1_mid_zone"] = (
        (out["raw_pred_error_prob"] >= MID_PROB_LOW) &
        (out["raw_pred_error_prob"] <= MID_PROB_HIGH)
    ).astype(int)
    return out


def attach_stage2_outputs(df, stage2_model, stage2_preprocessor, stage2_feature_cols, stage2_threshold: float):
    out = df.copy()
    X = stage2_preprocessor.transform(out[stage2_feature_cols])
    if hasattr(X, "toarray"):
        X = X.toarray()

    dataset = TabularDataset(
        X=X,
        y=np.zeros(len(out), dtype=np.float32),
        w=np.ones(len(out), dtype=np.float32),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=False)

    detector_prob = predict_probs(stage2_model, loader, DEVICE)
    out["detector_pred_error_prob"] = detector_prob
    out["detector_pred_label"] = (out["detector_pred_error_prob"] >= float(stage2_threshold)).astype(int)
    out["detector_pred_label_name"] = out["detector_pred_label"].map({1: "error", 0: "correct"})
    out["stage2_decision_threshold"] = float(stage2_threshold)
    out["detector_margin_to_threshold"] = out["detector_pred_error_prob"] - float(stage2_threshold)
    out["is_stage_disagree"] = (out["raw_pred_label"] != out["detector_pred_label"]).astype(int)
    return out


def build_verifier_dataset(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out[out["detector_pred_label"] == 1].copy()
    if len(out) == 0:
        raise ValueError("verifier 训练集为空。")

    out["verifier_label"] = out[LABEL_COL].astype(int)

    if WEIGHT_COL not in out.columns:
        out[WEIGHT_COL] = 1.0
    out[WEIGHT_COL] = pd.to_numeric(out[WEIGHT_COL], errors="coerce").fillna(1.0)

    out["verifier_sample_weight"] = out[WEIGHT_COL].copy()

    fp_mask = out["verifier_label"] == 0
    out.loc[fp_mask, "verifier_sample_weight"] *= VERIFIER_FP_BASE_WEIGHT_BOOST

    fd_conflict_fp_mask = (
        fp_mask &
        out.get("main_rule_type", pd.Series(["missing"] * len(out))).astype(str).isin(
            ["functional_dependency", "soft_functional_dependency"]
        ) &
        out.get("neighbor_bucket", pd.Series(["missing"] * len(out))).astype(str).isin(
            ["strong_support_current", "weak_support_current"]
        )
    )
    out.loc[fd_conflict_fp_mask, "verifier_sample_weight"] *= VERIFIER_FD_CONFLICT_FP_BOOST

    rare_only_fp_mask = fp_mask & (pd.to_numeric(out.get("rare_only_signal", 0), errors="coerce").fillna(0).astype(int) == 1)
    out.loc[rare_only_fp_mask, "verifier_sample_weight"] *= VERIFIER_RARE_ONLY_FP_BOOST

    borderline_fp_mask = fp_mask & pd.to_numeric(out.get("detector_pred_error_prob", 0.0), errors="coerce").fillna(0.0).between(0.45, 0.80)
    out.loc[borderline_fp_mask, "verifier_sample_weight"] *= VERIFIER_BORDERLINE_FP_BOOST

    risk_col_fp_mask = fp_mask & out.get("column", pd.Series(["missing"] * len(out))).astype(str).isin(
        {"ProviderNumber", "PhoneNumber", "Score", "Stateavg", "Address1", "Sample"}
    )
    out.loc[risk_col_fp_mask, "verifier_sample_weight"] *= VERIFIER_RISK_COLUMN_FP_BOOST

    sample_true_error_mask = (out["verifier_label"] == 1) & (out.get("column", pd.Series(["missing"] * len(out))).astype(str) == "Sample")
    out.loc[sample_true_error_mask, "verifier_sample_weight"] *= VERIFIER_SAMPLE_TRUE_ERROR_BOOST

    print("[DEBUG] verifier_label distribution:")
    print(out["verifier_label"].value_counts(dropna=False))

    return out


def train_fp_verifier(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
):
    ensure_dir(os.path.dirname(VERIFIER_MODEL_PATH))
    ensure_dir(os.path.dirname(VERIFIER_PREPROCESSOR_PATH))
    ensure_dir(os.path.dirname(VERIFIER_CONFIG_PATH))

    preprocessor, feature_cols, num_cols, cat_cols = build_preprocessor(
        train_df, numeric_features, categorical_features
    )
    preprocessor.fit(train_df[feature_cols])

    X_train = preprocessor.transform(train_df[feature_cols])
    X_val = preprocessor.transform(val_df[feature_cols])

    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
    if hasattr(X_val, "toarray"):
        X_val = X_val.toarray()

    y_train = train_df["verifier_label"].values.astype(int)
    y_val = val_df["verifier_label"].values.astype(int)
    w_train = pd.to_numeric(
        train_df["verifier_sample_weight"], errors="coerce"
    ).fillna(1.0).values.astype(float)

    unique_train_classes = np.unique(y_train)

    # 单类保护
    if len(unique_train_classes) < 2:
        only_class = int(unique_train_classes[0])
        print(f"[WARN] verifier 训练集只有一个类别: {only_class}，使用 DummyClassifier。")

        model = DummyClassifier(strategy="constant", constant=only_class)
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train).astype(int)
        val_pred = model.predict(X_val).astype(int)

        train_keep_prob = train_pred.astype(float)
        val_keep_prob = val_pred.astype(float)

        verifier_train_metrics = compute_metrics(y_train, train_keep_prob, threshold=0.5)
        verifier_val_metrics = compute_metrics(y_val, val_keep_prob, threshold=0.5)

        joblib.dump(model, VERIFIER_MODEL_PATH)
        joblib.dump(preprocessor, VERIFIER_PREPROCESSOR_PATH)

        verifier_cfg = {
            "numeric_features": num_cols,
            "categorical_features": cat_cols,
            "feature_cols": feature_cols,
            "is_single_class_verifier": True,
            "single_class_value": only_class,
        }
        save_json(verifier_cfg, VERIFIER_CONFIG_PATH)

        return {
            "model": model,
            "preprocessor": preprocessor,
            "feature_cols": feature_cols,
            "verifier_train_metrics": verifier_train_metrics,
            "verifier_val_metrics": verifier_val_metrics,
        }

    model = RandomForestClassifier(
        n_estimators=VERIFIER_N_ESTIMATORS,
        max_depth=VERIFIER_MAX_DEPTH,
        min_samples_leaf=VERIFIER_MIN_SAMPLES_LEAF,
        class_weight=VERIFIER_CLASS_WEIGHT,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, sample_weight=w_train)

    if hasattr(model, "classes_") and len(model.classes_) == 2:
        class_to_index = {c: i for i, c in enumerate(model.classes_)}
        pos_idx = class_to_index.get(1, 1)
        train_keep_prob = model.predict_proba(X_train)[:, pos_idx]
        val_keep_prob = model.predict_proba(X_val)[:, pos_idx]
    else:
        print("[WARN] verifier 模型 classes_ 异常，退化为 predict 输出。")
        train_keep_prob = model.predict(X_train).astype(float)
        val_keep_prob = model.predict(X_val).astype(float)

    verifier_train_metrics = compute_metrics(y_train, train_keep_prob, threshold=0.5)
    verifier_val_metrics = compute_metrics(y_val, val_keep_prob, threshold=0.5)

    joblib.dump(model, VERIFIER_MODEL_PATH)
    joblib.dump(preprocessor, VERIFIER_PREPROCESSOR_PATH)

    verifier_cfg = {
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "feature_cols": feature_cols,
        "is_single_class_verifier": False,
        "single_class_value": None,
    }
    save_json(verifier_cfg, VERIFIER_CONFIG_PATH)

    return {
        "model": model,
        "preprocessor": preprocessor,
        "feature_cols": feature_cols,
        "verifier_train_metrics": verifier_train_metrics,
        "verifier_val_metrics": verifier_val_metrics,
    }


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


def run_full_training(df: pd.DataFrame):
    rest_df, final_val_df = train_test_split(
        df,
        test_size=FINAL_VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[LABEL_COL],
    )
    base_train_df, cal_train_df = train_test_split(
        rest_df,
        test_size=CAL_TRAIN_SIZE_IN_REST,
        random_state=RANDOM_STATE,
        stratify=rest_df[LABEL_COL],
    )

    stage1_result = run_stage_training(
        stage_name="STAGE1_BASE",
        train_df=base_train_df,
        val_df=cal_train_df,
        numeric_features=STAGE1_NUMERIC_FEATURES,
        categorical_features=STAGE1_CATEGORICAL_FEATURES,
        batch_size=STAGE1_BATCH_SIZE,
        max_epochs=STAGE1_MAX_EPOCHS,
        lr=STAGE1_LR,
        weight_decay=STAGE1_WEIGHT_DECAY,
        patience=STAGE1_PATIENCE,
        min_delta=STAGE1_MIN_DELTA,
        hidden_dim_1=STAGE1_HIDDEN_DIM_1,
        hidden_dim_2=STAGE1_HIDDEN_DIM_2,
        dropout=STAGE1_DROPOUT,
        preprocessor_path=STAGE1_PREPROCESSOR,
        last_ckpt_path=STAGE1_LAST_CKPT,
        best_ckpt_path=STAGE1_BEST_CKPT,
        train_log_csv=STAGE1_TRAIN_LOG,
        min_recall_target=TARGET_MIN_RECALL_STAGE1,
    )

    cal_train_with_stage1 = attach_stage1_outputs(
        cal_train_df,
        stage1_result["model"],
        stage1_result["preprocessor"],
        stage1_result["feature_cols"],
        stage1_result["best_threshold"],
    )
    final_val_with_stage1 = attach_stage1_outputs(
        final_val_df,
        stage1_result["model"],
        stage1_result["preprocessor"],
        stage1_result["feature_cols"],
        stage1_result["best_threshold"],
    )

    cal_train_with_stage1 = make_stage2_training_df(cal_train_with_stage1)
    ensure_dir(os.path.dirname(STAGE2_TRAIN_ANALYSIS_CSV))
    cal_train_with_stage1.to_csv(STAGE2_TRAIN_ANALYSIS_CSV, index=False, encoding="utf-8-sig")

    stage2_result = run_stage_training(
        stage_name="STAGE2_CAL",
        train_df=cal_train_with_stage1,
        val_df=final_val_with_stage1,
        numeric_features=STAGE2_NUMERIC_FEATURES,
        categorical_features=STAGE2_CATEGORICAL_FEATURES,
        batch_size=STAGE2_BATCH_SIZE,
        max_epochs=STAGE2_MAX_EPOCHS,
        lr=STAGE2_LR,
        weight_decay=STAGE2_WEIGHT_DECAY,
        patience=STAGE2_PATIENCE,
        min_delta=STAGE2_MIN_DELTA,
        hidden_dim_1=STAGE2_HIDDEN_DIM_1,
        hidden_dim_2=STAGE2_HIDDEN_DIM_2,
        dropout=STAGE2_DROPOUT,
        preprocessor_path=STAGE2_PREPROCESSOR,
        last_ckpt_path=STAGE2_LAST_CKPT,
        best_ckpt_path=STAGE2_BEST_CKPT,
        train_log_csv=STAGE2_TRAIN_LOG,
        min_recall_target=TARGET_MIN_RECALL_STAGE2,
    )

    return stage1_result, stage2_result, final_val_df, cal_train_df


def run_verifier_training_only(df: pd.DataFrame):
    copy_detector_artifacts(DETECTOR_SOURCE_DIR, OUTPUT_DIR)

    config = load_json(os.path.join(DETECTOR_SOURCE_DIR, "two_stage_config.json"))
    metrics = load_json(os.path.join(DETECTOR_SOURCE_DIR, "metrics_two_stage.json"))

    stage1_cfg = config["stage1"]
    stage2_cfg = config["stage2"]

    stage1_preprocessor = joblib.load(os.path.join(DETECTOR_SOURCE_DIR, "stage1_base_preprocessor.joblib"))
    stage2_preprocessor = joblib.load(os.path.join(DETECTOR_SOURCE_DIR, "stage2_cal_preprocessor.joblib"))

    stage1_model = MLPDetector(
        stage1_cfg["input_dim"], stage1_cfg["hidden_dim_1"], stage1_cfg["hidden_dim_2"], stage1_cfg["dropout"]
    ).to(DEVICE)
    stage2_model = MLPDetector(
        stage2_cfg["input_dim"], stage2_cfg["hidden_dim_1"], stage2_cfg["hidden_dim_2"], stage2_cfg["dropout"]
    ).to(DEVICE)

    stage1_ckpt = torch.load(os.path.join(DETECTOR_SOURCE_DIR, "stage1_base_best.pt"), map_location=DEVICE)
    stage2_ckpt = torch.load(os.path.join(DETECTOR_SOURCE_DIR, "stage2_cal_best.pt"), map_location=DEVICE)
    stage1_model.load_state_dict(stage1_ckpt["model_state_dict"])
    stage2_model.load_state_dict(stage2_ckpt["model_state_dict"])
    stage1_model.eval()
    stage2_model.eval()

    stage1_threshold = float(metrics["stage1_best_threshold"])
    stage2_threshold = float(metrics["stage2_best_threshold"])

    rest_df, final_val_df = train_test_split(
        df,
        test_size=FINAL_VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[LABEL_COL],
    )
    _, cal_train_df = train_test_split(
        rest_df,
        test_size=CAL_TRAIN_SIZE_IN_REST,
        random_state=RANDOM_STATE,
        stratify=rest_df[LABEL_COL],
    )

    cal_train_with_stage1 = attach_stage1_outputs(
        cal_train_df, stage1_model, stage1_preprocessor, stage1_cfg["feature_cols"], stage1_threshold
    )
    final_val_with_stage1 = attach_stage1_outputs(
        final_val_df, stage1_model, stage1_preprocessor, stage1_cfg["feature_cols"], stage1_threshold
    )

    cal_train_with_stage1 = make_stage2_training_df(cal_train_with_stage1)

    cal_train_with_stage2 = attach_stage2_outputs(
        cal_train_with_stage1, stage2_model, stage2_preprocessor, stage2_cfg["feature_cols"], stage2_threshold
    )
    final_val_with_stage2 = attach_stage2_outputs(
        final_val_with_stage1, stage2_model, stage2_preprocessor, stage2_cfg["feature_cols"], stage2_threshold
    )

    return config, metrics, cal_train_with_stage2, final_val_with_stage2


def finalize_verifier_and_metrics(config, metrics, verifier_result, final_val_with_stage2):
    verifier_feature_cols = verifier_result["feature_cols"]
    X_verifier_val = verifier_result["preprocessor"].transform(final_val_with_stage2[verifier_feature_cols])
    if hasattr(X_verifier_val, "toarray"):
        X_verifier_val = X_verifier_val.toarray()

    model = verifier_result["model"]

    if hasattr(model, "classes_") and len(model.classes_) == 1:
        only_class = int(model.classes_[0])
        all_verifier_keep_prob = np.full(len(final_val_with_stage2), float(only_class), dtype=float)
    else:
        if hasattr(model, "classes_") and len(model.classes_) == 2:
            class_to_index = {c: i for i, c in enumerate(model.classes_)}
            pos_idx = class_to_index.get(1, 1)
            all_verifier_keep_prob = model.predict_proba(X_verifier_val)[:, pos_idx]
        else:
            all_verifier_keep_prob = model.predict(X_verifier_val).astype(float)

    detector_pred_label_final = final_val_with_stage2["detector_pred_label"].values.astype(int)
    y_true_final = final_val_with_stage2[LABEL_COL].values.astype(int)

    verifier_threshold, verifier_final_metrics = choose_best_verifier_threshold(
        y_true=y_true_final,
        detector_pred_label=detector_pred_label_final,
        verifier_keep_prob=all_verifier_keep_prob,
        min_recall=VERIFIER_MIN_FINAL_RECALL,
        alpha=PRECISION_PRIORITY_ALPHA,
    )

    final_pred_label = detector_pred_label_final.copy()
    detector_positive_mask = detector_pred_label_final == 1
    final_pred_label[detector_positive_mask] = (
        all_verifier_keep_prob[detector_positive_mask] >= verifier_threshold
    ).astype(int)

    final_out = final_val_with_stage2.copy()
    final_out["true_label"] = final_out[LABEL_COL]
    final_out["verifier_keep_error_prob"] = all_verifier_keep_prob
    final_out["verifier_decision_threshold"] = float(verifier_threshold)
    final_out["verifier_decision"] = np.where(
        final_out["detector_pred_label"].values.astype(int) == 1,
        np.where(all_verifier_keep_prob >= verifier_threshold, "keep_error", "reject_error"),
        "skip"
    )
    final_out["final_pred_label"] = final_pred_label.astype(int)
    final_out["final_pred_label_name"] = final_out["final_pred_label"].map({1: "error", 0: "correct"})
    final_out["final_pred_error_prob"] = final_out["detector_pred_error_prob"]
    ensure_dir(os.path.dirname(FINAL_VAL_PRED_CSV))
    final_out.to_csv(FINAL_VAL_PRED_CSV, index=False, encoding="utf-8-sig")

    verifier_metrics_all = {
        "verifier_train_metrics": verifier_result["verifier_train_metrics"],
        "verifier_val_metrics": verifier_result["verifier_val_metrics"],
        "best_verifier_threshold_on_final_val": verifier_threshold,
        "final_pipeline_metrics_with_verifier": verifier_final_metrics,
    }
    save_json(verifier_metrics_all, VERIFIER_METRICS_JSON)

    two_stage_cfg = TwoStageConfig(
        device=DEVICE,
        random_state=RANDOM_STATE,
        label_col=LABEL_COL,
        weight_col=WEIGHT_COL,
        final_val_size=FINAL_VAL_SIZE,
        cal_train_size_in_rest=CAL_TRAIN_SIZE_IN_REST,
        stage1=StageConfig(**config["stage1"]),
        stage2=StageConfig(**config["stage2"]),
        verifier=VerifierConfig(
            numeric_features=safe_existing_columns(final_val_with_stage2, VERIFIER_NUMERIC_FEATURES),
            categorical_features=safe_existing_columns(final_val_with_stage2, VERIFIER_CATEGORICAL_FEATURES),
            feature_cols=verifier_result["feature_cols"],
            threshold=float(verifier_threshold),
            metrics=verifier_final_metrics,
        ),
    )

    metrics_all = dict(metrics)
    metrics_all["verifier_train_metrics"] = verifier_result["verifier_train_metrics"]
    metrics_all["verifier_val_metrics"] = verifier_result["verifier_val_metrics"]
    metrics_all["verifier_best_threshold"] = verifier_threshold
    metrics_all["final_metrics_with_verifier"] = verifier_final_metrics

    save_json(two_stage_cfg, CONFIG_PATH)
    save_json(metrics_all, METRICS_JSON)

    print("\n===== Final pipeline with verifier =====")
    print(json.dumps(verifier_final_metrics, ensure_ascii=False, indent=2))
    print(f"\n[OK] 输出目录: {OUTPUT_DIR}")


def main():
    set_seed(RANDOM_STATE)
    ensure_dir(OUTPUT_DIR)

    df = load_training_data(INPUT_CSV)
    df = add_derived_features(df)

    if DETECTOR_TRAIN_MODE == "full":
        stage1_result, stage2_result, final_val_df, cal_train_df = run_full_training(df)

        cal_train_with_stage1 = attach_stage1_outputs(
            cal_train_df,
            stage1_result["model"],
            stage1_result["preprocessor"],
            stage1_result["feature_cols"],
            stage1_result["best_threshold"],
        )
        final_val_with_stage1 = attach_stage1_outputs(
            final_val_df,
            stage1_result["model"],
            stage1_result["preprocessor"],
            stage1_result["feature_cols"],
            stage1_result["best_threshold"],
        )

        cal_train_with_stage1 = make_stage2_training_df(cal_train_with_stage1)

        cal_train_with_stage2 = attach_stage2_outputs(
            cal_train_with_stage1,
            stage2_result["model"],
            stage2_result["preprocessor"],
            stage2_result["feature_cols"],
            stage2_result["best_threshold"],
        )
        final_val_with_stage2 = attach_stage2_outputs(
            final_val_with_stage1,
            stage2_result["model"],
            stage2_result["preprocessor"],
            stage2_result["feature_cols"],
            stage2_result["best_threshold"],
        )

        config = {
            "stage1": {
                "input_dim": stage1_result["model"].net[0].in_features,
                "hidden_dim_1": STAGE1_HIDDEN_DIM_1,
                "hidden_dim_2": STAGE1_HIDDEN_DIM_2,
                "dropout": STAGE1_DROPOUT,
                "batch_size": STAGE1_BATCH_SIZE,
                "learning_rate": STAGE1_LR,
                "weight_decay": STAGE1_WEIGHT_DECAY,
                "max_epochs": STAGE1_MAX_EPOCHS,
                "patience": STAGE1_PATIENCE,
                "numeric_features": safe_existing_columns(df, STAGE1_NUMERIC_FEATURES),
                "categorical_features": safe_existing_columns(df, STAGE1_CATEGORICAL_FEATURES),
                "feature_cols": stage1_result["feature_cols"],
            },
            "stage2": {
                "input_dim": stage2_result["model"].net[0].in_features,
                "hidden_dim_1": STAGE2_HIDDEN_DIM_1,
                "hidden_dim_2": STAGE2_HIDDEN_DIM_2,
                "dropout": STAGE2_DROPOUT,
                "batch_size": STAGE2_BATCH_SIZE,
                "learning_rate": STAGE2_LR,
                "weight_decay": STAGE2_WEIGHT_DECAY,
                "max_epochs": STAGE2_MAX_EPOCHS,
                "patience": STAGE2_PATIENCE,
                "numeric_features": safe_existing_columns(final_val_with_stage2, STAGE2_NUMERIC_FEATURES),
                "categorical_features": safe_existing_columns(final_val_with_stage2, STAGE2_CATEGORICAL_FEATURES),
                "feature_cols": stage2_result["feature_cols"],
            },
        }
        metrics = {
            "stage1_best_epoch": stage1_result["best_epoch"],
            "stage1_best_threshold": stage1_result["best_threshold"],
            "stage1_best_metrics": stage1_result["best_metrics"],
            "stage2_best_epoch": stage2_result["best_epoch"],
            "stage2_best_threshold": stage2_result["best_threshold"],
            "stage2_best_metrics": stage2_result["best_metrics"],
        }

    elif DETECTOR_TRAIN_MODE == "freeze_reuse":
        config, metrics, cal_train_with_stage2, final_val_with_stage2 = run_verifier_training_only(df)

    else:
        raise ValueError(f"未知 DETECTOR_TRAIN_MODE: {DETECTOR_TRAIN_MODE}")

    verifier_train_df = build_verifier_dataset(cal_train_with_stage2)
    verifier_val_df = build_verifier_dataset(final_val_with_stage2)

    ensure_dir(os.path.dirname(VERIFIER_TRAIN_CSV))
    ensure_dir(os.path.dirname(VERIFIER_VAL_CSV))
    verifier_train_df.to_csv(VERIFIER_TRAIN_CSV, index=False, encoding="utf-8-sig")
    verifier_val_df.to_csv(VERIFIER_VAL_CSV, index=False, encoding="utf-8-sig")

    verifier_result = train_fp_verifier(
        train_df=verifier_train_df,
        val_df=verifier_val_df,
        numeric_features=VERIFIER_NUMERIC_FEATURES,
        categorical_features=VERIFIER_CATEGORICAL_FEATURES,
    )

    finalize_verifier_and_metrics(config, metrics, verifier_result, final_val_with_stage2)


if __name__ == "__main__":
    main()
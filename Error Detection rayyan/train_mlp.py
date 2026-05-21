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


INPUT_CSV = "final_train_dataset_rayyan.csv"
OUTPUT_DIR = "two_stage_model_output_rayyan_v4_with_verifier"

DETECTOR_TRAIN_MODE = "full"
DETECTOR_SOURCE_DIR = "two_stage_model_output_rayyan_v4_with_verifier"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_STATE = 42

LABEL_COL = "label_binary"
WEIGHT_COL = "sample_weight"

FINAL_VAL_SIZE = 0.20
CAL_TRAIN_SIZE_IN_REST = 0.25

SAVE_EPOCH_CHECKPOINT = True
SAVE_EVERY_EPOCHS = 5
AUTO_FIND_BEST_THRESHOLD = True

TARGET_MIN_RECALL_STAGE1 = 0.97
TARGET_MIN_RECALL_STAGE2 = 0.95

THRESHOLD_GRID_START = 0.10
THRESHOLD_GRID_END = 0.95
THRESHOLD_GRID_STEP = 0.02
PRECISION_PRIORITY_ALPHA = 0.18

VERIFIER_THRESHOLD_GRID_START = 0.10
VERIFIER_THRESHOLD_GRID_END = 0.95
VERIFIER_THRESHOLD_GRID_STEP = 0.02
VERIFIER_MIN_FINAL_RECALL = 0.97

USE_POS_WEIGHT = True

STAGE2_POS_PRED_WEIGHT_BOOST = 1.0
STAGE2_MID_PROB_WEIGHT_BOOST = 1.0
STAGE2_FALSE_POS_RISK_COLS = {"journal_title", "jounral_abbreviation", "journal_issn", "article_language", "article_jcreated_at", "author_list"}
STAGE2_RISK_COLUMN_WEIGHT_BOOST = 1.0
STAGE2_SUPPORT_CURRENT_WEIGHT_BOOST = 1.0
STAGE2_RARE_ONLY_WEIGHT_BOOST = 1.0

MID_PROB_LOW = 0.35
MID_PROB_HIGH = 0.75

VERIFIER_N_ESTIMATORS = 300
VERIFIER_MAX_DEPTH = 8
VERIFIER_MIN_SAMPLES_LEAF = 10
VERIFIER_CLASS_WEIGHT = "balanced_subsample"

VERIFIER_FP_BASE_WEIGHT_BOOST = 1.25
VERIFIER_FD_CONFLICT_FP_BOOST = 2.5
VERIFIER_RARE_ONLY_FP_BOOST = 1.45
VERIFIER_BORDERLINE_FP_BOOST = 1.45
VERIFIER_RISK_COLUMN_FP_BOOST = 2.0
VERIFIER_SAMPLE_TRUE_ERROR_BOOST = 1.20
VERIFIER_RESCUE_CANDIDATE_TRUE_ERROR_BOOST = 2.20
VERIFIER_RESCUE_CANDIDATE_FP_DAMP = 0.70
VERIFIER_WEAK_OTHER_FP_BOOST = 2.2
VERIFIER_WEAK_OTHER_RARE_FD_FP_BOOST = 2.6
VERIFIER_MEASURE_COL_FP_BOOST = 1.40

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

STAGE1_NUMERIC_FEATURES = [
    "violation_count", "conflict_score", "value_frequency", "value_frequency_rank",
    "neighbor_majority_ratio", "prior_error_probability", "posterior_error_probability",
    "is_equal_to_neighbor_majority", "neighbor_agreement_score",
    "strong_rule_count", "candidate_generation_rule_count", "fd_like_count",
    "context_rule_count", "global_rule_count", "rare_value_count", "typo_rule_count",
    "pattern_rule_count", "schema_rule_count", "canonical_rule_count", "numeric_value",
    "candidate_rule_ratio", "rule_total_count",
    "strong_rule_ratio", "context_rule_ratio", "fd_like_ratio", "global_rule_ratio",
    "rule_strength_sum", "has_neighbor_signal", "high_neighbor_consistency",
    "is_rare_value", "neighbor_disagree",
]

STAGE1_CATEGORICAL_FEATURES = [
    "column", "semantic_type", "main_rule_type", "main_usage_role",
    "pattern_bucket", "rarity_bucket", "neighbor_bucket", "candidate_rule_dominant",
    "canonical_bucket", "date_value_bucket", "language_bucket",
]

STAGE2_NUMERIC_FEATURES = [
    "raw_pred_error_prob", "raw_pred_label", "raw_margin_to_threshold",
    "stage1_high_conf_error", "stage1_mid_zone",
    "violation_count", "conflict_score", "value_frequency", "value_frequency_rank",
    "neighbor_majority_ratio", "prior_error_probability", "posterior_error_probability",
    "is_equal_to_neighbor_majority", "neighbor_agreement_score",
    "strong_rule_count", "candidate_generation_rule_count", "fd_like_count",
    "context_rule_count", "global_rule_count", "rare_value_count", "typo_rule_count",
    "pattern_rule_count", "schema_rule_count", "canonical_rule_count", "numeric_value",
    "candidate_rule_ratio", "rule_total_count",
    "strong_rule_ratio", "context_rule_ratio", "fd_like_ratio", "global_rule_ratio",
    "rule_strength_sum", "has_neighbor_signal", "high_neighbor_consistency",
    "is_rare_value", "neighbor_disagree", "rare_only_signal",
]

STAGE2_CATEGORICAL_FEATURES = [
    "column", "semantic_type", "main_rule_type", "main_usage_role",
    "pattern_bucket", "rarity_bucket", "neighbor_bucket", "candidate_rule_dominant",
    "canonical_bucket", "date_value_bucket", "language_bucket",
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
    "pattern_rule_count", "schema_rule_count", "canonical_rule_count", "numeric_value",
    "candidate_rule_ratio", "rule_total_count",
    "strong_rule_ratio", "context_rule_ratio", "fd_like_ratio", "global_rule_ratio",
    "rule_strength_sum", "has_neighbor_signal", "high_neighbor_consistency",
    "is_rare_value", "neighbor_disagree", "rare_only_signal", "is_weak_only", "is_fp_risk_col",
]

VERIFIER_CATEGORICAL_FEATURES = [
    "column", "semantic_type", "main_rule_type", "main_usage_role",
    "pattern_bucket", "rarity_bucket", "neighbor_bucket", "candidate_rule_dominant",
    "canonical_bucket", "date_value_bucket", "language_bucket",
]

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
        "journal_title", "jounral_abbreviation", "journal_issn",
        "article_language", "article_jcreated_at", "article_pagination",
        "author_list"
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
    column_series = out.get("column", pd.Series(["missing"] * len(out))).astype(str)

    out["stage1_high_conf_error"] = (raw_prob >= 0.80).astype(int)
    out["stage1_mid_zone"] = ((raw_prob >= MID_PROB_LOW) & (raw_prob <= MID_PROB_HIGH)).astype(int)

    support_current_mask = (
        (eq_neighbor == 1) &
        (ratio >= 0.80) &
        (pd.to_numeric(out.get("strong_rule_count", 0.0), errors="coerce").fillna(0.0) <= 0)
    )
    out["high_prob_but_neighbor_support_current"] = (
        (raw_prob >= 0.80) & support_current_mask
    ).astype(int)

    out["raw_margin_to_threshold"] = raw_prob - 0.5

    weight_multiplier = np.ones(len(out), dtype=float)

    if STAGE2_POS_PRED_WEIGHT_BOOST != 1.0:
        weight_multiplier *= np.where(raw_label == 1, STAGE2_POS_PRED_WEIGHT_BOOST, 1.0)

    if STAGE2_MID_PROB_WEIGHT_BOOST != 1.0:
        weight_multiplier *= np.where(out["stage1_mid_zone"] == 1, STAGE2_MID_PROB_WEIGHT_BOOST, 1.0)

    if STAGE2_RISK_COLUMN_WEIGHT_BOOST != 1.0:
        weight_multiplier *= np.where(column_series.isin(STAGE2_FALSE_POS_RISK_COLS), STAGE2_RISK_COLUMN_WEIGHT_BOOST, 1.0)

    if STAGE2_SUPPORT_CURRENT_WEIGHT_BOOST != 1.0:
        weight_multiplier *= np.where(support_current_mask, STAGE2_SUPPORT_CURRENT_WEIGHT_BOOST, 1.0)

    rare_only_signal = pd.to_numeric(out.get("rare_only_signal", 0), errors="coerce").fillna(0).astype(int)
    if STAGE2_RARE_ONLY_WEIGHT_BOOST != 1.0:
        weight_multiplier *= np.where(rare_only_signal == 1, STAGE2_RARE_ONLY_WEIGHT_BOOST, 1.0)

    out[WEIGHT_COL] = out[WEIGHT_COL] * weight_multiplier

    return out


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
            "input_dim": input_dim,
            "pos_weight": pos_weight,
        }
        history.append(row)

        print(
            f"[Epoch {epoch:03d}] "
            f"loss={train_stats['loss']:.6f} "
            f"thr={epoch_best_threshold:.4f} "
            f"P={epoch_best_metrics['precision']:.4f} "
            f"R={epoch_best_metrics['recall']:.4f} "
            f"F1={epoch_best_metrics['f1']:.4f}"
        )

        save_checkpoint(
            path=ckpt_last_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_epoch=best_epoch,
            best_score=best_score,
            best_threshold=best_threshold,
            input_dim=input_dim,
        )

        if SAVE_EPOCH_CHECKPOINT and (epoch % SAVE_EVERY_EPOCHS == 0):
            epoch_ckpt_path = ckpt_last_path.replace(".pt", f"_epoch{epoch}.pt")
            save_checkpoint(
                path=epoch_ckpt_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_epoch=best_epoch,
                best_score=best_score,
                best_threshold=best_threshold,
                input_dim=input_dim,
            )

        if epoch_score > best_score + min_delta:
            best_score = epoch_score
            best_epoch = epoch
            best_threshold = float(epoch_best_threshold)
            best_metrics = dict(epoch_best_metrics)
            stale_epochs = 0

            save_checkpoint(
                path=ckpt_best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_epoch=best_epoch,
                best_score=best_score,
                best_threshold=best_threshold,
                input_dim=input_dim,
            )
            joblib.dump(preprocessor, preprocessor_path)
        else:
            stale_epochs += 1

        if stale_epochs >= patience:
            print(f"[Early Stop] epoch={epoch}, best_epoch={best_epoch}")
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
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "best_epoch": best_epoch,
        "best_threshold": best_threshold,
        "best_metrics": best_metrics,
    }


def attach_stage1_outputs(
    df: pd.DataFrame,
    model,
    preprocessor,
    feature_cols: List[str],
    threshold: float,
) -> pd.DataFrame:
    out = df.copy()
    X = transform_preprocessor(preprocessor, out, feature_cols)
    loader = make_loader(X, np.zeros(len(out)), np.ones(len(out)), batch_size=1024, shuffle=False)
    prob = predict_probs(model, loader, DEVICE)
    out["raw_pred_error_prob"] = prob
    out["raw_pred_label"] = (prob >= threshold).astype(int)
    out["raw_margin_to_threshold"] = prob - threshold
    out["stage1_high_conf_error"] = (prob >= 0.80).astype(int)
    out["stage1_mid_zone"] = ((prob >= MID_PROB_LOW) & (prob <= MID_PROB_HIGH)).astype(int)
    return out


def attach_stage2_outputs(
    df: pd.DataFrame,
    model,
    preprocessor,
    feature_cols: List[str],
    threshold: float,
) -> pd.DataFrame:
    out = df.copy()
    X = transform_preprocessor(preprocessor, out, feature_cols)
    loader = make_loader(X, np.zeros(len(out)), np.ones(len(out)), batch_size=1024, shuffle=False)
    prob = predict_probs(model, loader, DEVICE)
    out["detector_pred_error_prob"] = prob
    out["detector_pred_label"] = (prob >= threshold).astype(int)
    out["detector_margin_to_threshold"] = prob - threshold
    out["is_stage_disagree"] = (out["raw_pred_label"].astype(int) != out["detector_pred_label"].astype(int)).astype(int)
    return out



def get_verifier_recall_rescue_mask(df: pd.DataFrame) -> pd.Series:
    out = df.copy()

    raw_label = pd.to_numeric(out.get("raw_pred_label", 0), errors="coerce").fillna(0).astype(int)
    detector_label = pd.to_numeric(out.get("detector_pred_label", 0), errors="coerce").fillna(0).astype(int)
    raw_prob = pd.to_numeric(out.get("raw_pred_error_prob", 0.0), errors="coerce").fillna(0.0)
    detector_prob = pd.to_numeric(out.get("detector_pred_error_prob", 0.0), errors="coerce").fillna(0.0)
    posterior = pd.to_numeric(out.get("posterior_error_probability", 0.0), errors="coerce").fillna(0.0)
    conflict = pd.to_numeric(out.get("conflict_score", 0.0), errors="coerce").fillna(0.0)

    col = out.get("column", pd.Series(["missing"] * len(out))).astype(str)
    rule = out.get("main_rule_type", pd.Series(["missing"] * len(out))).astype(str)
    nb = out.get("neighbor_bucket", pd.Series(["missing"] * len(out))).astype(str)

    common_gate = (
        raw_label.eq(1) &
        detector_label.eq(0) &
        (raw_prob >= 0.995) &
        (posterior >= 0.997)
    )

    sample_notnull_fd_mask = (
        col.eq("Sample") &
        rule.isin(["not_null", "functional_dependency", "soft_functional_dependency"]) &
        nb.eq("strong_support_current") &
        (conflict >= 8.0)
    )
    sample_rare_mask = (
        col.eq("Sample") &
        rule.eq("rare_value") &
        nb.eq("weak_support_other") &
        (posterior >= 0.999)
    )
    zipcode_mask = (
        col.eq("ZipCode") &
        rule.eq("soft_functional_dependency") &
        nb.eq("strong_support_other") &
        (conflict >= 40.0)
    )
    emer_mask = (
        col.eq("EmergencyService") &
        rule.eq("soft_functional_dependency") &
        nb.eq("strong_support_other") &
        (conflict >= 40.0)
    )
    hosp_mask = (
        col.eq("HospitalName") &
        rule.eq("soft_functional_dependency") &
        nb.eq("strong_support_other") &
        (conflict >= 40.0)
    )
    cond_mask = (
        col.eq("Condition") &
        rule.eq("soft_functional_dependency") &
        nb.eq("strong_support_other") &
        (conflict >= 8.0)
    )

    pattern_mask = sample_notnull_fd_mask | sample_rare_mask | zipcode_mask | emer_mask | hosp_mask | cond_mask
    near_miss_mask = detector_prob >= 0.60
    return (common_gate & pattern_mask & near_miss_mask).astype(bool)


def build_verifier_dataset(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    rescue_candidate_mask_all = get_verifier_recall_rescue_mask(out)
    out["is_recall_rescue_candidate"] = rescue_candidate_mask_all.astype(int)

    verifier_pool_mask = (
        pd.to_numeric(out.get("detector_pred_label", 0), errors="coerce").fillna(0).astype(int).eq(1) |
        rescue_candidate_mask_all
    )
    out = out[verifier_pool_mask].copy()
    if len(out) == 0:
        raise ValueError("verifier 训练集为空。")

    out["verifier_label"] = out[LABEL_COL].astype(int)

    if WEIGHT_COL not in out.columns:
        out[WEIGHT_COL] = 1.0
    out[WEIGHT_COL] = pd.to_numeric(out[WEIGHT_COL], errors="coerce").fillna(1.0)

    out["verifier_sample_weight"] = out[WEIGHT_COL].copy()

    fp_mask = out["verifier_label"] == 0

    col_series = out.get("column", pd.Series(["missing"] * len(out))).astype(str)
    rule_series = out.get("main_rule_type", pd.Series(["missing"] * len(out))).astype(str)
    nb_series = out.get("neighbor_bucket", pd.Series(["missing"] * len(out))).astype(str)

    rare_only_signal = pd.to_numeric(out.get("rare_only_signal", 0), errors="coerce").fillna(0).astype(int)
    detector_prob = pd.to_numeric(out.get("detector_pred_error_prob", 0.0), errors="coerce").fillna(0.0)

    out.loc[fp_mask, "verifier_sample_weight"] *= VERIFIER_FP_BASE_WEIGHT_BOOST

    fd_conflict_fp_mask = (
        fp_mask &
        rule_series.isin(["functional_dependency", "soft_functional_dependency"]) &
        nb_series.isin(["strong_support_current", "weak_support_current"])
    )
    out.loc[fd_conflict_fp_mask, "verifier_sample_weight"] *= VERIFIER_FD_CONFLICT_FP_BOOST

    rare_only_fp_mask = fp_mask & (rare_only_signal == 1)
    out.loc[rare_only_fp_mask, "verifier_sample_weight"] *= VERIFIER_RARE_ONLY_FP_BOOST

    borderline_fp_mask = fp_mask & detector_prob.between(0.45, 0.85)
    out.loc[borderline_fp_mask, "verifier_sample_weight"] *= VERIFIER_BORDERLINE_FP_BOOST

    risk_col_fp_mask = fp_mask & col_series.isin(
        {"ProviderNumber", "PhoneNumber", "Score", "Stateavg", "Address1", "Sample"}
    )
    out.loc[risk_col_fp_mask, "verifier_sample_weight"] *= VERIFIER_RISK_COLUMN_FP_BOOST

    weak_other_fp_mask = fp_mask & nb_series.eq("weak_support_other")
    out.loc[weak_other_fp_mask, "verifier_sample_weight"] *= VERIFIER_WEAK_OTHER_FP_BOOST

    weak_other_rare_fd_fp_mask = (
        fp_mask &
        nb_series.eq("weak_support_other") &
        rule_series.isin(["rare_value", "rare_pattern", "functional_dependency", "soft_functional_dependency"])
    )
    out.loc[weak_other_rare_fd_fp_mask, "verifier_sample_weight"] *= VERIFIER_WEAK_OTHER_RARE_FD_FP_BOOST

    measure_col_fp_mask = fp_mask & col_series.isin({"MeasureName", "MeasureCode", "Condition"})
    out.loc[measure_col_fp_mask, "verifier_sample_weight"] *= VERIFIER_MEASURE_COL_FP_BOOST

    sample_true_error_mask = (out["verifier_label"] == 1) & col_series.eq("Sample")
    out.loc[sample_true_error_mask, "verifier_sample_weight"] *= VERIFIER_SAMPLE_TRUE_ERROR_BOOST

    rescue_candidate_mask = pd.to_numeric(out.get("is_recall_rescue_candidate", 0), errors="coerce").fillna(0).astype(int).eq(1)
    rescue_true_error_mask = rescue_candidate_mask & (out["verifier_label"] == 1)
    rescue_fp_mask = rescue_candidate_mask & (out["verifier_label"] == 0)

    out.loc[rescue_true_error_mask, "verifier_sample_weight"] *= VERIFIER_RESCUE_CANDIDATE_TRUE_ERROR_BOOST
    out.loc[rescue_fp_mask, "verifier_sample_weight"] *= VERIFIER_RESCUE_CANDIDATE_FP_DAMP

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

    train_prob = model.predict_proba(X_train)[:, 1]
    val_prob = model.predict_proba(X_val)[:, 1]

    verifier_train_metrics = compute_metrics(y_train, train_prob, threshold=0.5)
    verifier_val_metrics = compute_metrics(y_val, val_prob, threshold=0.5)

    joblib.dump(model, VERIFIER_MODEL_PATH)
    joblib.dump(preprocessor, VERIFIER_PREPROCESSOR_PATH)

    verifier_cfg = {
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "feature_cols": feature_cols,
        "is_single_class_verifier": False,
    }
    save_json(verifier_cfg, VERIFIER_CONFIG_PATH)

    return {
        "model": model,
        "preprocessor": preprocessor,
        "feature_cols": feature_cols,
        "verifier_train_metrics": verifier_train_metrics,
        "verifier_val_metrics": verifier_val_metrics,
    }


def run_full_training(df: pd.DataFrame):
    rest_df, final_val_df = train_test_split(
        df,
        test_size=FINAL_VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[LABEL_COL],
    )

    stage1_train_df, cal_train_df = train_test_split(
        rest_df,
        test_size=CAL_TRAIN_SIZE_IN_REST,
        random_state=RANDOM_STATE,
        stratify=rest_df[LABEL_COL],
    )

    stage1_result = run_single_stage_training(
        train_df=stage1_train_df,
        val_df=final_val_df,
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

    cal_train_with_stage1 = attach_stage1_outputs(
        cal_train_df,
        stage1_result["model"],
        stage1_result["preprocessor"],
        stage1_result["feature_cols"],
        stage1_result["best_threshold"],
    )
    cal_train_with_stage1 = make_stage2_training_df(cal_train_with_stage1)

    final_val_with_stage1 = attach_stage1_outputs(
        final_val_df,
        stage1_result["model"],
        stage1_result["preprocessor"],
        stage1_result["feature_cols"],
        stage1_result["best_threshold"],
    )

    stage2_result = run_single_stage_training(
        train_df=cal_train_with_stage1,
        val_df=final_val_with_stage1,
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

    analysis_df = cal_train_with_stage1.copy()
    analysis_df.to_csv(STAGE2_TRAIN_ANALYSIS_CSV, index=False, encoding="utf-8-sig")

    return stage1_result, stage2_result, final_val_df, cal_train_df


def run_verifier_training_only(df: pd.DataFrame):
    if not os.path.exists(DETECTOR_SOURCE_DIR):
        raise ValueError(f"DETECTOR_SOURCE_DIR 不存在: {DETECTOR_SOURCE_DIR}")

    copy_detector_artifacts(DETECTOR_SOURCE_DIR, OUTPUT_DIR)

    config = load_json(CONFIG_PATH)
    metrics = load_json(METRICS_JSON)

    stage1_cfg = config["stage1"]
    stage2_cfg = config["stage2"]

    stage1_model = MLPDetector(
        stage1_cfg["input_dim"],
        stage1_cfg["hidden_dim_1"],
        stage1_cfg["hidden_dim_2"],
        stage1_cfg["dropout"],
    ).to(DEVICE)
    stage1_ckpt = load_checkpoint(STAGE1_BEST_CKPT, DEVICE)
    stage1_model.load_state_dict(stage1_ckpt["model_state_dict"])
    stage1_preprocessor = joblib.load(STAGE1_PREPROCESSOR)
    stage1_model.eval()

    stage2_model = MLPDetector(
        stage2_cfg["input_dim"],
        stage2_cfg["hidden_dim_1"],
        stage2_cfg["hidden_dim_2"],
        stage2_cfg["dropout"],
    ).to(DEVICE)
    stage2_ckpt = load_checkpoint(STAGE2_BEST_CKPT, DEVICE)
    stage2_model.load_state_dict(stage2_ckpt["model_state_dict"])
    stage2_preprocessor = joblib.load(STAGE2_PREPROCESSOR)
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
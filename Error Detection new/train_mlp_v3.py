import json
import os
import random
from dataclasses import dataclass, asdict, is_dataclass
from typing import List, Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
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
OUTPUT_DIR = "two_stage_model_output_v3"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_STATE = 42

LABEL_COL = "label_binary"
WEIGHT_COL = "sample_weight"

# 三段切分：all labeled -> base_train / cal_train / final_val
FINAL_VAL_SIZE = 0.20
CAL_TRAIN_SIZE_IN_REST = 0.25   # 剩余 80% 中切 25%，即全体的 20%

CONTINUE_STAGE1 = False
CONTINUE_STAGE2 = False

SAVE_EPOCH_CHECKPOINT = True
SAVE_EVERY_EPOCHS = 5

AUTO_FIND_BEST_THRESHOLD = True

# ============================================================
# 2. 训练目标偏好：先把模型拉回不过度保守的状态
# ============================================================

TARGET_MIN_RECALL_STAGE1 = 0.95
TARGET_MIN_RECALL_STAGE2 = 0.92

THRESHOLD_GRID_START = 0.10
THRESHOLD_GRID_END = 0.95
THRESHOLD_GRID_STEP = 0.02

# 阈值搜索不再过度偏向 precision
PRECISION_PRIORITY_ALPHA = 0.15

# ============================================================
# 3. loss 配置（简化版）
# ============================================================

# 先不用 focal，恢复最稳定的逐样本加权 BCE
FOCAL_GAMMA = 0.0

# 先关掉 pos_weight，避免和 sample_weight / Stage2 叠加过强
USE_POS_WEIGHT = False

# 先关掉辅助 loss，只保留分类主损失
LAMBDA_NEIGHBOR = 0.0
LAMBDA_RULE = 0.0

MIN_NEIGHBOR_RATIO_FOR_LOSS = 0.55

STRONG_RULE_SCALE = 1.0
CAND_RULE_SCALE = 0.12
FD_LIKE_SCALE = 0.25
CTX_RULE_SCALE = 0.18
GLOBAL_RULE_SCALE = 0.08
TYPO_RULE_SCALE = 0.30
PATTERN_RULE_SCALE = 0.12
SCHEMA_RULE_SCALE = 0.20
RARE_VALUE_RULE_SCALE = 0.0

# ============================================================
# 4. Stage2 样本权重增强（简化版：全部关闭）
# ============================================================

STAGE2_POS_PRED_WEIGHT_BOOST = 1.0
STAGE2_MID_PROB_WEIGHT_BOOST = 1.0
STAGE2_FALSE_POS_RISK_COLS = {"EmergencyService", "Sample", "ZipCode", "HospitalName"}
STAGE2_RISK_COLUMN_WEIGHT_BOOST = 1.0
STAGE2_SUPPORT_CURRENT_WEIGHT_BOOST = 1.0
STAGE2_RARE_ONLY_WEIGHT_BOOST = 1.0

MID_PROB_LOW = 0.35
MID_PROB_HIGH = 0.75

# ============================================================
# 5. 路径
# ============================================================

STAGE1_LAST_CKPT = os.path.join(OUTPUT_DIR, "stage1_base_last.pt")
STAGE1_BEST_CKPT = os.path.join(OUTPUT_DIR, "stage1_base_best.pt")
STAGE1_PREPROCESSOR = os.path.join(OUTPUT_DIR, "stage1_base_preprocessor.joblib")
STAGE1_TRAIN_LOG = os.path.join(OUTPUT_DIR, "stage1_train_log.csv")

STAGE2_LAST_CKPT = os.path.join(OUTPUT_DIR, "stage2_cal_last.pt")
STAGE2_BEST_CKPT = os.path.join(OUTPUT_DIR, "stage2_cal_best.pt")
STAGE2_PREPROCESSOR = os.path.join(OUTPUT_DIR, "stage2_cal_preprocessor.joblib")
STAGE2_TRAIN_LOG = os.path.join(OUTPUT_DIR, "stage2_train_log.csv")

CONFIG_PATH = os.path.join(OUTPUT_DIR, "two_stage_config.json")
METRICS_JSON = os.path.join(OUTPUT_DIR, "metrics_two_stage.json")
FINAL_VAL_PRED_CSV = os.path.join(OUTPUT_DIR, "final_val_predictions.csv")
STAGE2_TRAIN_ANALYSIS_CSV = os.path.join(OUTPUT_DIR, "stage2_train_analysis.csv")

# ============================================================
# 6. 特征配置
# ============================================================

STAGE1_NUMERIC_FEATURES = [
    "violation_count",
    "conflict_score",
    "value_frequency",
    "value_frequency_rank",
    "neighbor_majority_ratio",
    "prior_error_probability",
    "posterior_error_probability",
    "is_equal_to_neighbor_majority",
    "neighbor_agreement_score",
    "strong_rule_count",
    "candidate_generation_rule_count",
    "fd_like_count",
    "context_rule_count",
    "global_rule_count",
    "rare_value_count",
    "typo_rule_count",
    "pattern_rule_count",
    "schema_rule_count",
    "candidate_rule_ratio",
    "rule_total_count",
    "strong_rule_ratio",
    "context_rule_ratio",
    "fd_like_ratio",
    "global_rule_ratio",
    "rule_strength_sum",
    "has_neighbor_signal",
    "high_neighbor_consistency",
    "is_rare_value",
    "neighbor_disagree",
]

STAGE1_CATEGORICAL_FEATURES = [
    "column",
    "semantic_type",
    "main_rule_type",
    "main_usage_role",
    "pattern_bucket",
    "rarity_bucket",
    "neighbor_bucket",
    "candidate_rule_dominant",
]

STAGE2_NUMERIC_FEATURES = [
    "raw_pred_error_prob",
    "raw_pred_label",
    "raw_margin_to_threshold",
    "stage1_high_conf_error",
    "stage1_mid_zone",
    "violation_count",
    "conflict_score",
    "value_frequency",
    "value_frequency_rank",
    "neighbor_majority_ratio",
    "prior_error_probability",
    "posterior_error_probability",
    "is_equal_to_neighbor_majority",
    "neighbor_agreement_score",
    "strong_rule_count",
    "candidate_generation_rule_count",
    "fd_like_count",
    "context_rule_count",
    "global_rule_count",
    "rare_value_count",
    "typo_rule_count",
    "pattern_rule_count",
    "schema_rule_count",
    "candidate_rule_ratio",
    "rule_total_count",
    "strong_rule_ratio",
    "context_rule_ratio",
    "fd_like_ratio",
    "global_rule_ratio",
    "rule_strength_sum",
    "has_neighbor_signal",
    "high_neighbor_consistency",
    "is_rare_value",
    "neighbor_disagree",
    "rare_only_signal",
]

STAGE2_CATEGORICAL_FEATURES = [
    "column",
    "semantic_type",
    "main_rule_type",
    "main_usage_role",
    "pattern_bucket",
    "rarity_bucket",
    "neighbor_bucket",
    "candidate_rule_dominant",
]

# ============================================================
# 7. 超参数
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
# 8. 工具函数
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(obj, path: str):
    def convert_dataclass_to_dict(o):
        if is_dataclass(o):
            return asdict(o)
        elif isinstance(o, dict):
            return {k: convert_dataclass_to_dict(v) for k, v in o.items()}
        elif isinstance(o, (list, tuple)):
            return [convert_dataclass_to_dict(item) for item in o]
        else:
            return o

    obj = convert_dataclass_to_dict(obj)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def safe_existing_columns(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def normalize_text_series_for_compare(s: pd.Series) -> pd.Series:
    return s.fillna("__NULL__").astype(str).str.strip()


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "value" in out.columns and "neighbor_majority_value" in out.columns:
        value_series = normalize_text_series_for_compare(out["value"])
        neighbor_series = normalize_text_series_for_compare(out["neighbor_majority_value"])
        out["is_equal_to_neighbor_majority"] = (value_series == neighbor_series).astype(int)
    else:
        out["is_equal_to_neighbor_majority"] = 0

    if "neighbor_majority_ratio" in out.columns:
        ratio = pd.to_numeric(out["neighbor_majority_ratio"], errors="coerce").fillna(0.0)
    else:
        ratio = pd.Series(np.zeros(len(out)), index=out.index)

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

    out["rule_total_count"] = (
        out["strong_rule_count"] + out["candidate_generation_rule_count"]
    ).astype(float)

    out["candidate_rule_ratio"] = np.where(
        out["rule_total_count"] > 0,
        out["candidate_generation_rule_count"] / (out["rule_total_count"] + 1e-8),
        0.0
    )
    out["strong_rule_ratio"] = np.where(
        out["rule_total_count"] > 0,
        out["strong_rule_count"] / (out["rule_total_count"] + 1e-8),
        0.0
    )
    out["context_rule_ratio"] = np.where(
        out["rule_total_count"] > 0,
        out["context_rule_count"] / (out["rule_total_count"] + 1e-8),
        0.0
    )
    out["fd_like_ratio"] = np.where(
        out["rule_total_count"] > 0,
        out["fd_like_count"] / (out["rule_total_count"] + 1e-8),
        0.0
    )
    out["global_rule_ratio"] = np.where(
        out["rule_total_count"] > 0,
        out["global_rule_count"] / (out["rule_total_count"] + 1e-8),
        0.0
    )

    out["has_neighbor_signal"] = (
        pd.to_numeric(out.get("neighbor_majority_ratio", 0.0), errors="coerce")
        .fillna(0.0)
        .gt(0)
        .astype(int)
    )
    out["high_neighbor_consistency"] = (
        pd.to_numeric(out.get("neighbor_majority_ratio", 0.0), errors="coerce")
        .fillna(0.0)
        .ge(0.8)
        .astype(int)
    )
    out["is_rare_value"] = (
        pd.to_numeric(out.get("value_frequency_rank", np.nan), errors="coerce")
        .fillna(999999)
        .gt(10)
        .astype(int)
    )
    out["neighbor_disagree"] = (
        (out["has_neighbor_signal"] == 1) &
        (out["is_equal_to_neighbor_majority"] == 0)
    ).astype(int)

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
        (pd.to_numeric(out["rare_value_count"], errors="coerce").fillna(0.0) > 0) &
        (pd.to_numeric(out["strong_rule_count"], errors="coerce").fillna(0.0) <= 0) &
        (pd.to_numeric(out["fd_like_count"], errors="coerce").fillna(0.0) <= 0) &
        (pd.to_numeric(out["context_rule_count"], errors="coerce").fillna(0.0) <= 0) &
        (pd.to_numeric(out["typo_rule_count"], errors="coerce").fillna(0.0) <= 0) &
        (pd.to_numeric(out["schema_rule_count"], errors="coerce").fillna(0.0) <= 0)
    ).astype(int)

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
        t, m, _ = valid[0]
        return t, m

    candidates.sort(key=lambda x: (x[1]["f1"], x[1]["recall"], x[1]["precision"]), reverse=True)
    t, m, _ = candidates[0]
    return t, m


# ============================================================
# 9. 数据读取
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
    return df


# ============================================================
# 10. 预处理器
# ============================================================

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


# ============================================================
# 11. Dataset
# ============================================================

class TabularDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, meta: Dict[str, np.ndarray]):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.w = torch.tensor(w, dtype=torch.float32)
        self.meta = {k: torch.tensor(v, dtype=torch.float32) for k, v in meta.items()}

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        item = {"X": self.X[idx], "y": self.y[idx], "w": self.w[idx]}
        for k, v in self.meta.items():
            item[k] = v[idx]
        return item


# ============================================================
# 12. 模型
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
# 13. dataclass
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
class TwoStageConfig:
    device: str
    random_state: int
    label_col: str
    weight_col: str
    final_val_size: float
    cal_train_size_in_rest: float
    stage1: StageConfig
    stage2: StageConfig


# ============================================================
# 14. loss（简化版）
# ============================================================

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
    loss = bce * sample_weights
    return loss.mean()


def build_neighbor_soft_target(is_equal_to_neighbor_majority, neighbor_majority_ratio):
    ratio = torch.clamp(neighbor_majority_ratio, 0.0, 1.0)
    valid_mask = (ratio >= MIN_NEIGHBOR_RATIO_FOR_LOSS).float()
    base = torch.where(is_equal_to_neighbor_majority > 0.5, 1.0 - ratio, ratio)
    soft_target = base
    return soft_target, valid_mask


def build_rule_soft_target(
    strong_rule_count,
    candidate_generation_rule_count,
    fd_like_count,
    context_rule_count,
    global_rule_count,
    rare_value_count,
    typo_rule_count,
    pattern_rule_count,
    schema_rule_count,
):
    score = (
        STRONG_RULE_SCALE * strong_rule_count +
        CAND_RULE_SCALE * candidate_generation_rule_count +
        FD_LIKE_SCALE * fd_like_count +
        CTX_RULE_SCALE * context_rule_count +
        GLOBAL_RULE_SCALE * global_rule_count +
        RARE_VALUE_RULE_SCALE * rare_value_count +
        TYPO_RULE_SCALE * typo_rule_count +
        PATTERN_RULE_SCALE * pattern_rule_count +
        SCHEMA_RULE_SCALE * schema_rule_count
    )
    score = torch.log1p(score)
    soft_target = torch.sigmoid(0.8 * score - 1.0)
    return soft_target


def composite_loss(logits, targets, sample_weights, batch, pos_weight_tensor):
    cls_loss = weighted_bce_with_logits(
        logits=logits,
        targets=targets,
        sample_weights=sample_weights,
        pos_weight=pos_weight_tensor,
    )

    total_loss = cls_loss

    loss_detail = {
        "total_loss": float(total_loss.detach().cpu().item()),
        "cls_loss": float(cls_loss.detach().cpu().item()),
        "neighbor_loss": 0.0,
        "rule_loss": 0.0,
    }
    return total_loss, loss_detail


# ============================================================
# 15. 训练核心
# ============================================================

def train_one_epoch(model, loader, optimizer, device, pos_weight_tensor):
    model.train()
    total_loss = total_cls = total_neighbor = total_rule = 0.0
    total_count = 0

    for batch in loader:
        X_batch = batch["X"].to(device)
        y_batch = batch["y"].to(device)
        w_batch = batch["w"].to(device)

        batch_gpu = {k: v.to(device) for k, v in batch.items() if k not in {"X", "y", "w"}}

        optimizer.zero_grad()
        logits = model(X_batch)
        loss, detail = composite_loss(
            logits=logits,
            targets=y_batch,
            sample_weights=w_batch,
            batch=batch_gpu,
            pos_weight_tensor=pos_weight_tensor,
        )
        loss.backward()
        optimizer.step()

        bs = len(X_batch)
        total_loss += detail["total_loss"] * bs
        total_cls += detail["cls_loss"] * bs
        total_neighbor += detail["neighbor_loss"] * bs
        total_rule += detail["rule_loss"] * bs
        total_count += bs

    return {
        "loss": total_loss / max(total_count, 1),
        "cls_loss": total_cls / max(total_count, 1),
        "neighbor_loss": total_neighbor / max(total_count, 1),
        "rule_loss": total_rule / max(total_count, 1),
    }


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


def build_meta_arrays(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    def get_num_col(name, default=0.0):
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(default).values.astype(np.float32)
        return np.full(len(df), default, dtype=np.float32)

    return {
        "is_equal_to_neighbor_majority": get_num_col("is_equal_to_neighbor_majority", 0.0),
        "neighbor_majority_ratio": get_num_col("neighbor_majority_ratio", 0.0),
        "strong_rule_count": get_num_col("strong_rule_count", 0.0),
        "candidate_generation_rule_count": get_num_col("candidate_generation_rule_count", 0.0),
        "fd_like_count": get_num_col("fd_like_count", 0.0),
        "context_rule_count": get_num_col("context_rule_count", 0.0),
        "global_rule_count": get_num_col("global_rule_count", 0.0),
        "rare_value_count": get_num_col("rare_value_count", 0.0),
        "typo_rule_count": get_num_col("typo_rule_count", 0.0),
        "pattern_rule_count": get_num_col("pattern_rule_count", 0.0),
        "schema_rule_count": get_num_col("schema_rule_count", 0.0),
    }


def make_stage2_training_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if WEIGHT_COL not in out.columns:
        out[WEIGHT_COL] = 1.0
    out[WEIGHT_COL] = pd.to_numeric(out[WEIGHT_COL], errors="coerce").fillna(1.0)
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
    continue_training: bool,
    min_recall_target: float,
):
    if continue_training and os.path.exists(preprocessor_path):
        preprocessor = joblib.load(preprocessor_path)
        num_cols = safe_existing_columns(train_df, numeric_features)
        cat_cols = safe_existing_columns(train_df, categorical_features)
        feature_cols = num_cols + cat_cols
        print(f"[INFO] [{stage_name}] Loaded existing preprocessor: {preprocessor_path}")
    else:
        preprocessor, feature_cols, num_cols, cat_cols = build_preprocessor(train_df, numeric_features, categorical_features)
        preprocessor.fit(train_df[feature_cols])
        joblib.dump(preprocessor, preprocessor_path)
        print(f"[INFO] [{stage_name}] Fitted and saved preprocessor: {preprocessor_path}")

    print(f"\n===== [{stage_name}] 实际使用特征 =====")
    print("Numeric features:", num_cols)
    print("Categorical features:", cat_cols)

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

    meta_train = build_meta_arrays(train_df)
    meta_val = build_meta_arrays(val_df)

    train_dataset = TabularDataset(X_train, y_train, w_train, meta_train)
    val_dataset = TabularDataset(X_val, y_val, w_val, meta_val)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    input_dim = X_train.shape[1]
    model = MLPDetector(input_dim, hidden_dim_1, hidden_dim_2, dropout).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    start_epoch = 1
    best_score = -1e9
    best_epoch = -1
    best_threshold = 0.5
    best_metrics = None
    patience_count = 0
    best_model_state = None
    train_logs = []

    pos_weight_value = build_pos_weight(y_train) if USE_POS_WEIGHT else 1.0
    pos_weight_tensor = torch.tensor(pos_weight_value, dtype=torch.float32, device=DEVICE)

    if continue_training and os.path.exists(last_ckpt_path):
        ckpt = load_checkpoint(last_ckpt_path, DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_epoch = int(ckpt.get("best_epoch", -1))
        best_score = float(ckpt.get("best_score", -1e9))
        best_threshold = float(ckpt.get("best_threshold", 0.5))
        print(f"[INFO] [{stage_name}] Resume from: {last_ckpt_path}, epoch={start_epoch}")
    else:
        print(f"[INFO] [{stage_name}] Start training from scratch.")

    for epoch in range(start_epoch, start_epoch + max_epochs):
        train_stat = train_one_epoch(model, train_loader, optimizer, DEVICE, pos_weight_tensor)
        val_prob = predict_probs(model, val_loader, DEVICE)

        if AUTO_FIND_BEST_THRESHOLD:
            threshold, metrics = choose_best_threshold(y_val, val_prob, min_recall=min_recall_target)
        else:
            threshold = 0.5
            metrics = compute_metrics(y_val, val_prob, threshold)

        current_score = metrics["f1"] + PRECISION_PRIORITY_ALPHA * metrics["precision"]

        log_row = {
            "epoch": epoch,
            "train_loss": train_stat["loss"],
            "train_cls_loss": train_stat["cls_loss"],
            "train_neighbor_loss": train_stat["neighbor_loss"],
            "train_rule_loss": train_stat["rule_loss"],
            "val_accuracy": metrics["accuracy"],
            "val_precision": metrics["precision"],
            "val_recall": metrics["recall"],
            "val_f1": metrics["f1"],
            "val_auc": metrics["auc"],
            "threshold": threshold,
            "selection_score": current_score,
            "min_recall_target": min_recall_target,
        }
        train_logs.append(log_row)

        print(
            f"[{stage_name}] Epoch {epoch:03d} | "
            f"loss={train_stat['loss']:.6f} | "
            f"cls={train_stat['cls_loss']:.6f} | "
            f"neighbor={train_stat['neighbor_loss']:.6f} | "
            f"rule={train_stat['rule_loss']:.6f} | "
            f"precision={metrics['precision']:.4f} | "
            f"recall={metrics['recall']:.4f} | "
            f"f1={metrics['f1']:.4f} | "
            f"score={current_score:.4f} | "
            f"threshold={threshold:.3f}"
        )

        save_checkpoint(last_ckpt_path, model, optimizer, epoch, best_epoch, best_score, best_threshold, input_dim)

        if SAVE_EPOCH_CHECKPOINT and (epoch % SAVE_EVERY_EPOCHS == 0):
            epoch_ckpt_path = os.path.join(OUTPUT_DIR, f"{stage_name.lower()}_epoch_{epoch}.pt")
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
                print(f"[INFO] [{stage_name}] Early stopping at epoch {epoch}.")
                break

    if best_model_state is None:
        raise RuntimeError(f"[{stage_name}] 训练失败，没有得到有效最佳模型。")

    model.load_state_dict(best_model_state)
    pd.DataFrame(train_logs).to_csv(train_log_csv, index=False, encoding="utf-8-sig")

    return {
        "model": model,
        "preprocessor": preprocessor,
        "feature_cols": feature_cols,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_threshold": best_threshold,
        "best_metrics": best_metrics,
        "train_df": train_df,
        "val_df": val_df,
    }


# ============================================================
# 16. 第二阶段数据构造
# ============================================================

def attach_stage1_outputs(df, stage1_model, stage1_preprocessor, stage1_feature_cols, stage1_threshold: float):
    out = df.copy()
    X = stage1_preprocessor.transform(out[stage1_feature_cols])
    if hasattr(X, "toarray"):
        X = X.toarray()

    meta = build_meta_arrays(out)
    dataset = TabularDataset(
        X=X,
        y=np.zeros(len(out), dtype=np.float32),
        w=np.ones(len(out), dtype=np.float32),
        meta=meta,
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


# ============================================================
# 17. 主流程
# ============================================================

def main():
    set_seed(RANDOM_STATE)
    ensure_dir(OUTPUT_DIR)

    df = load_training_data(INPUT_CSV)
    df = add_derived_features(df)

    print("\n===== 总标签分布 =====")
    print(df[LABEL_COL].value_counts(dropna=False).to_string())

    if "label_source" in df.columns:
        print("\n===== 标签来源分布（仅分析，不进模型） =====")
        print(df["label_source"].value_counts(dropna=False).to_string())

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

    print("\n===== 数据切分 =====")
    print(f"base_train size: {len(base_train_df)}")
    print(f"cal_train size : {len(cal_train_df)}")
    print(f"final_val size : {len(final_val_df)}")

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
        continue_training=CONTINUE_STAGE1,
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
        continue_training=CONTINUE_STAGE2,
        min_recall_target=TARGET_MIN_RECALL_STAGE2,
    )

    final_eval_df = final_val_with_stage1.copy()

    X_final = stage2_result["preprocessor"].transform(final_eval_df[stage2_result["feature_cols"]])
    if hasattr(X_final, "toarray"):
        X_final = X_final.toarray()

    meta_final = build_meta_arrays(final_eval_df)
    final_dataset = TabularDataset(
        X=X_final,
        y=final_eval_df[LABEL_COL].values.astype(np.float32),
        w=final_eval_df[WEIGHT_COL].values.astype(np.float32),
        meta=meta_final,
    )
    final_loader = torch.utils.data.DataLoader(final_dataset, batch_size=1024, shuffle=False)

    final_prob = predict_probs(stage2_result["model"], final_loader, DEVICE)
    final_threshold = stage2_result["best_threshold"]
    final_metrics = compute_metrics(
        final_eval_df[LABEL_COL].values.astype(int),
        final_prob,
        threshold=final_threshold
    )

    final_out = final_eval_df.copy()
    final_out["true_label"] = final_out[LABEL_COL]
    final_out["final_pred_error_prob"] = final_prob
    final_out["final_pred_label"] = (final_prob >= final_threshold).astype(int)
    final_out["final_pred_label_name"] = final_out["final_pred_label"].map({1: "error", 0: "correct"})
    final_out["stage2_decision_threshold"] = float(final_threshold)
    final_out.to_csv(FINAL_VAL_PRED_CSV, index=False, encoding="utf-8-sig")

    config = TwoStageConfig(
        device=DEVICE,
        random_state=RANDOM_STATE,
        label_col=LABEL_COL,
        weight_col=WEIGHT_COL,
        final_val_size=FINAL_VAL_SIZE,
        cal_train_size_in_rest=CAL_TRAIN_SIZE_IN_REST,
        stage1=StageConfig(
            input_dim=stage1_result["model"].net[0].in_features,
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
        ),
        stage2=StageConfig(
            input_dim=stage2_result["model"].net[0].in_features,
            hidden_dim_1=STAGE2_HIDDEN_DIM_1,
            hidden_dim_2=STAGE2_HIDDEN_DIM_2,
            dropout=STAGE2_DROPOUT,
            batch_size=STAGE2_BATCH_SIZE,
            learning_rate=STAGE2_LR,
            weight_decay=STAGE2_WEIGHT_DECAY,
            max_epochs=STAGE2_MAX_EPOCHS,
            patience=STAGE2_PATIENCE,
            numeric_features=safe_existing_columns(final_eval_df, STAGE2_NUMERIC_FEATURES),
            categorical_features=safe_existing_columns(final_eval_df, STAGE2_CATEGORICAL_FEATURES),
            feature_cols=stage2_result["feature_cols"],
        ),
    )

    metrics_all = {
        "stage1_best_epoch": stage1_result["best_epoch"],
        "stage1_best_threshold": stage1_result["best_threshold"],
        "stage1_best_metrics": stage1_result["best_metrics"],
        "stage2_best_epoch": stage2_result["best_epoch"],
        "stage2_best_threshold": stage2_result["best_threshold"],
        "stage2_best_metrics": stage2_result["best_metrics"],
        "final_metrics": final_metrics,
    }

    save_json(config, CONFIG_PATH)
    save_json(metrics_all, METRICS_JSON)

    print("\n===== Stage1 best =====")
    print(json.dumps(stage1_result["best_metrics"], ensure_ascii=False, indent=2))
    print("\n===== Stage2 best =====")
    print(json.dumps(stage2_result["best_metrics"], ensure_ascii=False, indent=2))
    print("\n===== Final eval =====")
    print(json.dumps(final_metrics, ensure_ascii=False, indent=2))
    print(f"\n[OK] 输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
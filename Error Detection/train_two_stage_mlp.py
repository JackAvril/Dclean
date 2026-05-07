import json
import os
import random
from dataclasses import dataclass, asdict
from typing import List, Tuple

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

INPUT_CSV = "propagated_labels_new.csv"
OUTPUT_DIR = "two_stage_model_output"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_STATE = 42
LABEL_COL = "label_binary"
WEIGHT_COL = "sample_weight"

# 三段切分
# 全部有标签数据 -> base_train / cal_train / final_val
FINAL_VAL_SIZE = 0.20
CAL_TRAIN_SIZE_IN_REST = 0.25   # 在剩余 80% 中再切 25%，即总数据的 20%

# 是否继续训练
CONTINUE_STAGE1 = False
CONTINUE_STAGE2 = False

# checkpoint 频率
SAVE_EPOCH_CHECKPOINT = True
SAVE_EVERY_EPOCHS = 5

# ============================================================
# 2. 路径
# ============================================================

# stage1
STAGE1_LAST_CKPT = os.path.join(OUTPUT_DIR, "stage1_base_last.pt")
STAGE1_BEST_CKPT = os.path.join(OUTPUT_DIR, "stage1_base_best.pt")
STAGE1_PREPROCESSOR = os.path.join(OUTPUT_DIR, "stage1_base_preprocessor.joblib")
STAGE1_TRAIN_LOG = os.path.join(OUTPUT_DIR, "stage1_train_log.csv")

# stage2
STAGE2_LAST_CKPT = os.path.join(OUTPUT_DIR, "stage2_cal_last.pt")
STAGE2_BEST_CKPT = os.path.join(OUTPUT_DIR, "stage2_cal_best.pt")
STAGE2_PREPROCESSOR = os.path.join(OUTPUT_DIR, "stage2_cal_preprocessor.joblib")
STAGE2_TRAIN_LOG = os.path.join(OUTPUT_DIR, "stage2_train_log.csv")

# common
CONFIG_PATH = os.path.join(OUTPUT_DIR, "two_stage_config.json")
METRICS_JSON = os.path.join(OUTPUT_DIR, "metrics_two_stage.json")
FINAL_VAL_PRED_CSV = os.path.join(OUTPUT_DIR, "final_val_predictions.csv")
UNLABELED_NAN_CSV = os.path.join(OUTPUT_DIR, "hard_cases_nan.csv")

# ============================================================
# 3. 特征配置
# ============================================================

# -------- 第一阶段：基础模型特征 --------
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
]

STAGE1_CATEGORICAL_FEATURES = [
    "column",
    "semantic_type",
    "main_rule_type",
    "pattern_bucket",
    "rarity_bucket",
    "neighbor_bucket",
]

# -------- 第二阶段：校准模型特征 --------
STAGE2_NUMERIC_FEATURES = [
    "raw_pred_error_prob",
    "raw_pred_label",
    "violation_count",
    "conflict_score",
    "value_frequency",
    "value_frequency_rank",
    "neighbor_majority_ratio",
    "prior_error_probability",
    "posterior_error_probability",
    "is_equal_to_neighbor_majority",
    "neighbor_agreement_score",
]

STAGE2_CATEGORICAL_FEATURES = [
    "column",
    "semantic_type",
    "main_rule_type",
    "pattern_bucket",
    "rarity_bucket",
    "neighbor_bucket",
]

# ============================================================
# 4. 第一阶段训练超参数
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

# ============================================================
# 5. 第二阶段训练超参数
# ============================================================

STAGE2_BATCH_SIZE = 64
STAGE2_MAX_EPOCHS = 50
STAGE2_LR = 1e-3
STAGE2_WEIGHT_DECAY = 1e-4
STAGE2_PATIENCE = 10
STAGE2_MIN_DELTA = 1e-4

STAGE2_HIDDEN_DIM_1 = 64
STAGE2_HIDDEN_DIM_2 = 32
STAGE2_DROPOUT = 0.2

AUTO_FIND_BEST_THRESHOLD = True


# ============================================================
# 6. 工具函数
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


def find_best_threshold(y_true, y_prob):
    best_t = 0.5
    best_f1 = -1.0
    best_metrics = None

    for t in np.arange(0.1, 0.91, 0.02):
        m = compute_metrics(y_true, y_prob, threshold=float(t))
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_t = float(t)
            best_metrics = m

    return best_t, best_metrics


# ============================================================
# 7. 数据读取
# ============================================================

def load_training_data(path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)

    if LABEL_COL not in df.columns:
        raise ValueError(f"输入文件缺少标签列: {LABEL_COL}")

    raw_df = df.copy()
    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce")
    nan_df = raw_df[pd.isna(df[LABEL_COL])].copy()

    df = df[df[LABEL_COL].isin([0, 1])].copy()
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    if WEIGHT_COL not in df.columns:
        df[WEIGHT_COL] = 1.0
    df[WEIGHT_COL] = pd.to_numeric(df[WEIGHT_COL], errors="coerce").fillna(1.0)

    return df, nan_df


# ============================================================
# 8. 预处理器
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
# 9. Dataset
# ============================================================

class TabularDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, w: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.w = torch.tensor(w, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.w[idx]


# ============================================================
# 10. 模型
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
# 11. 训练配置 dataclass
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
# 12. 训练核心
# ============================================================

def weighted_bce_loss(logits, targets, sample_weights):
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    losses = loss_fn(logits, targets)
    weighted = losses * sample_weights
    return weighted.mean()


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_count = 0

    for X_batch, y_batch, w_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        w_batch = w_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = weighted_bce_loss(logits, y_batch, w_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(X_batch)
        total_count += len(X_batch)

    return total_loss / max(total_count, 1)


@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    probs = []
    for X_batch, _, _ in loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch)
        batch_prob = torch.sigmoid(logits).cpu().numpy()
        probs.append(batch_prob)

    if not probs:
        return np.array([])
    return np.concatenate(probs, axis=0)


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_epoch: int,
    best_f1: float,
    best_threshold: float,
    input_dim: int,
):
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_f1": best_f1,
        "best_threshold": best_threshold,
        "input_dim": input_dim,
    }
    torch.save(ckpt, path)


def load_checkpoint(path: str, device: str):
    return torch.load(path, map_location=device)


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
):
    if continue_training and os.path.exists(preprocessor_path):
        preprocessor = joblib.load(preprocessor_path)
        num_cols = safe_existing_columns(train_df, numeric_features)
        cat_cols = safe_existing_columns(train_df, categorical_features)
        feature_cols = num_cols + cat_cols
        print(f"[INFO] [{stage_name}] Loaded existing preprocessor: {preprocessor_path}")
    else:
        preprocessor, feature_cols, num_cols, cat_cols = build_preprocessor(
            train_df, numeric_features, categorical_features
        )
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

    train_dataset = TabularDataset(X_train, y_train, w_train)
    val_dataset = TabularDataset(X_val, y_val, w_val)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    input_dim = X_train.shape[1]
    model = MLPDetector(
        input_dim=input_dim,
        hidden_dim_1=hidden_dim_1,
        hidden_dim_2=hidden_dim_2,
        dropout=dropout,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    start_epoch = 1
    best_f1 = -1.0
    best_epoch = -1
    best_threshold = 0.5
    best_metrics = None
    patience_count = 0
    best_model_state = None
    train_logs = []

    if continue_training and os.path.exists(last_ckpt_path):
        ckpt = load_checkpoint(last_ckpt_path, DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_epoch = int(ckpt.get("best_epoch", -1))
        best_f1 = float(ckpt.get("best_f1", -1.0))
        best_threshold = float(ckpt.get("best_threshold", 0.5))
        print(f"[INFO] [{stage_name}] Resume from: {last_ckpt_path}, epoch={start_epoch}")
    else:
        print(f"[INFO] [{stage_name}] Start training from scratch.")

    if continue_training and os.path.exists(best_ckpt_path):
        best_ckpt = load_checkpoint(best_ckpt_path, DEVICE)
        best_model_state = best_ckpt["model_state_dict"]

    for epoch in range(start_epoch, start_epoch + max_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
        val_prob = predict_probs(model, val_loader, DEVICE)

        if AUTO_FIND_BEST_THRESHOLD:
            threshold, metrics = find_best_threshold(y_val, val_prob)
        else:
            threshold = 0.5
            metrics = compute_metrics(y_val, val_prob, threshold)

        current_f1 = metrics["f1"]

        log_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_accuracy": metrics["accuracy"],
            "val_precision": metrics["precision"],
            "val_recall": metrics["recall"],
            "val_f1": metrics["f1"],
            "val_auc": metrics["auc"],
            "threshold": threshold,
        }
        train_logs.append(log_row)

        print(
            f"[{stage_name}] Epoch {epoch:03d} | "
            f"loss={train_loss:.6f} | "
            f"f1={metrics['f1']:.4f} | "
            f"recall={metrics['recall']:.4f} | "
            f"precision={metrics['precision']:.4f} | "
            f"threshold={threshold:.3f}"
        )

        save_checkpoint(
            last_ckpt_path, model, optimizer, epoch,
            best_epoch, best_f1, best_threshold, input_dim
        )

        if SAVE_EPOCH_CHECKPOINT and (epoch % SAVE_EVERY_EPOCHS == 0):
            epoch_ckpt_path = os.path.join(
                OUTPUT_DIR, f"{stage_name.lower()}_epoch_{epoch}.pt"
            )
            save_checkpoint(
                epoch_ckpt_path, model, optimizer, epoch,
                best_epoch, best_f1, best_threshold, input_dim
            )

        if current_f1 > best_f1 + min_delta:
            best_f1 = current_f1
            best_epoch = epoch
            best_threshold = threshold
            best_metrics = metrics
            patience_count = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            save_checkpoint(
                best_ckpt_path, model, optimizer, epoch,
                best_epoch, best_f1, best_threshold, input_dim
            )
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
        "best_f1": best_f1,
        "best_threshold": best_threshold,
        "best_metrics": best_metrics,
        "train_df": train_df,
        "val_df": val_df,
    }


# ============================================================
# 13. 构造第二阶段数据（已修正为与推理一致）
# ============================================================

def attach_stage1_outputs(
    df: pd.DataFrame,
    stage1_model,
    stage1_preprocessor,
    stage1_feature_cols,
    stage1_threshold: float,
):
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

    return out


# ============================================================
# 14. 主流程
# ============================================================

def main():
    set_seed(RANDOM_STATE)
    ensure_dir(OUTPUT_DIR)

    df, nan_df = load_training_data(INPUT_CSV)
    df = add_derived_features(df)

    if len(nan_df) > 0:
        nan_df = add_derived_features(nan_df)
        nan_df.to_csv(UNLABELED_NAN_CSV, index=False, encoding="utf-8-sig")

    print("\n===== 总标签分布 =====")
    print(df[LABEL_COL].value_counts(dropna=False).to_string())

    # ---------- 三段切分 ----------
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

    # ========================================================
    # Stage 1: 训练基础模型
    # ========================================================
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
    )

    # ========================================================
    # Stage 2: 生成 calibration 输入（使用 stage1 best_threshold）
    # ========================================================
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

    # ========================================================
    # Stage 2: 训练校准模型
    # ========================================================
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
    )

    # ========================================================
    # 最终验证：用 stage2 在 final_val 上输出结果
    # ========================================================
    final_eval_df = final_val_with_stage1.copy()

    X_final = stage2_result["preprocessor"].transform(final_eval_df[stage2_result["feature_cols"]])
    if hasattr(X_final, "toarray"):
        X_final = X_final.toarray()

    final_dataset = TabularDataset(
        X=X_final,
        y=final_eval_df[LABEL_COL].values.astype(np.float32),
        w=final_eval_df[WEIGHT_COL].values.astype(np.float32),
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

    # ========================================================
    # 保存统一配置
    # ========================================================
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
    save_json(asdict(config), CONFIG_PATH)

    summary = {
        "device": DEVICE,
        "base_train_size": int(len(base_train_df)),
        "cal_train_size": int(len(cal_train_df)),
        "final_val_size": int(len(final_val_df)),
        "nan_unlabeled_size": int(len(nan_df)),

        "stage1_best_epoch": int(stage1_result["best_epoch"]),
        "stage1_best_f1": float(stage1_result["best_f1"]),
        "stage1_best_threshold": float(stage1_result["best_threshold"]),
        "stage1_metrics": stage1_result["best_metrics"],

        "stage2_best_epoch": int(stage2_result["best_epoch"]),
        "stage2_best_f1": float(stage2_result["best_f1"]),
        "stage2_best_threshold": float(stage2_result["best_threshold"]),
        "stage2_metrics": stage2_result["best_metrics"],

        "final_eval_metrics": final_metrics,
        "continue_stage1": CONTINUE_STAGE1,
        "continue_stage2": CONTINUE_STAGE2,
    }
    save_json(summary, METRICS_JSON)

    print("\n===== 二阶段训练完成 =====")
    print(f"设备: {DEVICE}")
    print(f"base_train: {len(base_train_df)}")
    print(f"cal_train : {len(cal_train_df)}")
    print(f"final_val : {len(final_val_df)}")
    print(f"nan unlabeled: {len(nan_df)}")

    print("\n----- Stage1 -----")
    print(f"best epoch     : {stage1_result['best_epoch']}")
    print(f"best threshold : {stage1_result['best_threshold']:.4f}")
    print(f"best f1        : {stage1_result['best_f1']:.4f}")

    print("\n----- Stage2 -----")
    print(f"best epoch     : {stage2_result['best_epoch']}")
    print(f"best threshold : {stage2_result['best_threshold']:.4f}")
    print(f"best f1        : {stage2_result['best_f1']:.4f}")

    print("\n----- Final Eval (Two-Stage) -----")
    print(f"Accuracy  : {final_metrics['accuracy']:.4f}")
    print(f"Precision : {final_metrics['precision']:.4f}")
    print(f"Recall    : {final_metrics['recall']:.4f}")
    print(f"F1        : {final_metrics['f1']:.4f}")
    print(f"AUC       : {final_metrics['auc']}")

    print("\n工件保存：")
    print(f"Stage1 last ckpt : {STAGE1_LAST_CKPT}")
    print(f"Stage1 best ckpt : {STAGE1_BEST_CKPT}")
    print(f"Stage1 prep      : {STAGE1_PREPROCESSOR}")
    print(f"Stage2 last ckpt : {STAGE2_LAST_CKPT}")
    print(f"Stage2 best ckpt : {STAGE2_BEST_CKPT}")
    print(f"Stage2 prep      : {STAGE2_PREPROCESSOR}")
    print(f"Config           : {CONFIG_PATH}")
    print(f"Metrics          : {METRICS_JSON}")
    print(f"Final val preds  : {FINAL_VAL_PRED_CSV}")


if __name__ == "__main__":
    main()
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

OUTPUT_DIR = "mlp_model_output_enhanced"

# checkpoint 与工件
LAST_CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "mlp_detector_last.pt")
BEST_CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "mlp_detector_best.pt")
PREPROCESSOR_PATH = os.path.join(OUTPUT_DIR, "preprocessor_mlp.joblib")
CONFIG_PATH = os.path.join(OUTPUT_DIR, "mlp_config.json")
METRICS_JSON = os.path.join(OUTPUT_DIR, "metrics_mlp.json")
VAL_PRED_CSV = os.path.join(OUTPUT_DIR, "val_predictions_mlp.csv")
UNLABELED_NAN_CSV = os.path.join(OUTPUT_DIR, "hard_cases_nan_mlp.csv")
TRAIN_LOG_CSV = os.path.join(OUTPUT_DIR, "train_log.csv")

# 是否继续训练
CONTINUE_TRAINING = False

# 是否每隔若干 epoch 保存一个版本快照
SAVE_EPOCH_CHECKPOINT = True
SAVE_EVERY_EPOCHS = 5

RANDOM_STATE = 42
TEST_SIZE = 0.2
LABEL_COL = "label_binary"
WEIGHT_COL = "sample_weight"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 只使用训练和推理都可得到的特征
NUMERIC_FEATURES = [
    "violation_count",
    "conflict_score",
    "value_frequency",
    "value_frequency_rank",
    "neighbor_majority_ratio",
    "prior_error_probability",
    "posterior_error_probability",
    # 新增派生特征
    "is_equal_to_neighbor_majority",
    "neighbor_agreement_score",
]

CATEGORICAL_FEATURES = [
    "column",
    "semantic_type",
    "main_rule_type",
    "pattern_bucket",
    "rarity_bucket",
    "neighbor_bucket",
]

# 训练超参数
BATCH_SIZE = 64
MAX_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 10
MIN_DELTA = 1e-4

# MLP结构
HIDDEN_DIM_1 = 128
HIDDEN_DIM_2 = 64
DROPOUT = 0.2

AUTO_FIND_BEST_THRESHOLD = True


# ============================================================
# 2. 基础工具
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_existing_columns(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def normalize_text_series_for_compare(s: pd.Series) -> pd.Series:
    return s.fillna("__NULL__").astype(str).str.strip()


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于现有列动态生成训练/推理都可用的派生特征。
    """
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


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def print_basic_data_diagnostics(df: pd.DataFrame):
    print("\n===== 标签分布 =====")
    print(df[LABEL_COL].value_counts(dropna=False).to_string())

    if "label_source" in df.columns:
        print("\n===== 标签来源分布 =====")
        print(df["label_source"].value_counts(dropna=False).to_string())

    check_cols = safe_existing_columns(df, NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["value", "neighbor_majority_value"])
    print("\n===== 关键特征缺失率 =====")
    for c in check_cols:
        miss_ratio = df[c].isna().mean()
        print(f"{c}: {miss_ratio:.4f}")


# ============================================================
# 3. 数据读取
# ============================================================

def load_training_data(path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)

    if LABEL_COL not in df.columns:
        raise ValueError(f"输入文件缺少标签列: {LABEL_COL}")

    raw_df = df.copy()
    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce")
    nan_df = raw_df[pd.isna(df[LABEL_COL])].copy()

    # 只保留 0/1 标签用于训练
    df = df[df[LABEL_COL].isin([0, 1])].copy()
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    if WEIGHT_COL not in df.columns:
        df[WEIGHT_COL] = 1.0
    df[WEIGHT_COL] = pd.to_numeric(df[WEIGHT_COL], errors="coerce").fillna(1.0)

    return df, nan_df


# ============================================================
# 4. 预处理
# ============================================================

def build_preprocessor(df: pd.DataFrame):
    num_cols = safe_existing_columns(df, NUMERIC_FEATURES)
    cat_cols = safe_existing_columns(df, CATEGORICAL_FEATURES)

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
# 5. 数据集
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
# 6. 模型
# ============================================================

class MLPDetector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim_1: int = 128, hidden_dim_2: int = 64, dropout: float = 0.2):
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
# 7. 配置
# ============================================================

@dataclass
class TrainConfig:
    input_dim: int
    hidden_dim_1: int
    hidden_dim_2: int
    dropout: float
    batch_size: int
    learning_rate: float
    weight_decay: float
    max_epochs: int
    patience: int
    test_size: float
    numeric_features: List[str]
    categorical_features: List[str]
    feature_cols: List[str]
    label_col: str
    weight_col: str
    device: str


# ============================================================
# 8. 损失/训练/预测
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


# ============================================================
# 9. checkpoint 管理
# ============================================================

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


# ============================================================
# 10. 主训练流程
# ============================================================

def main():
    set_seed(RANDOM_STATE)
    ensure_dir(OUTPUT_DIR)

    df, nan_df = load_training_data(INPUT_CSV)

    # 新增：动态派生特征
    df = add_derived_features(df)
    if len(nan_df) > 0:
        nan_df = add_derived_features(nan_df)
        nan_df.to_csv(UNLABELED_NAN_CSV, index=False, encoding="utf-8-sig")

    print_basic_data_diagnostics(df)

    train_df, val_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[LABEL_COL],
    )

    if "label_source" in train_df.columns:
        print("\n===== 训练集标签来源分布 =====")
        print(train_df["label_source"].value_counts(dropna=False).to_string())

    if "label_source" in val_df.columns:
        print("\n===== 验证集标签来源分布 =====")
        print(val_df["label_source"].value_counts(dropna=False).to_string())

    # ---------- 预处理器 ----------
    if CONTINUE_TRAINING and os.path.exists(PREPROCESSOR_PATH):
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        num_cols = safe_existing_columns(train_df, NUMERIC_FEATURES)
        cat_cols = safe_existing_columns(train_df, CATEGORICAL_FEATURES)
        feature_cols = num_cols + cat_cols
        print(f"[INFO] Loaded existing preprocessor from: {PREPROCESSOR_PATH}")
    else:
        preprocessor, feature_cols, num_cols, cat_cols = build_preprocessor(train_df)
        preprocessor.fit(train_df[feature_cols])
        joblib.dump(preprocessor, PREPROCESSOR_PATH)
        print(f"[INFO] Fitted and saved preprocessor to: {PREPROCESSOR_PATH}")

    print("\n===== 实际使用特征 =====")
    print("Numeric features:", num_cols)
    print("Categorical features:", cat_cols)

    X_train = preprocessor.transform(train_df[feature_cols])
    X_val = preprocessor.transform(val_df[feature_cols])

    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
    if hasattr(X_val, "toarray"):
        X_val = X_val.toarray()

    y_train = train_df[LABEL_COL].values.astype(np.float32)
    w_train = train_df[WEIGHT_COL].values.astype(np.float32)
    y_val = val_df[LABEL_COL].values.astype(np.float32)
    w_val = val_df[WEIGHT_COL].values.astype(np.float32)

    train_dataset = TabularDataset(X_train, y_train, w_train)
    val_dataset = TabularDataset(X_val, y_val, w_val)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    input_dim = X_train.shape[1]

    model = MLPDetector(
        input_dim=input_dim,
        hidden_dim_1=HIDDEN_DIM_1,
        hidden_dim_2=HIDDEN_DIM_2,
        dropout=DROPOUT,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # ---------- 继续训练 / 从头训练 ----------
    start_epoch = 1
    best_f1 = -1.0
    best_epoch = -1
    best_threshold = 0.5
    best_metrics = None
    patience_count = 0
    best_model_state = None
    train_logs = []

    if CONTINUE_TRAINING and os.path.exists(LAST_CHECKPOINT_PATH):
        ckpt = load_checkpoint(LAST_CHECKPOINT_PATH, DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_epoch = int(ckpt.get("best_epoch", -1))
        best_f1 = float(ckpt.get("best_f1", -1.0))
        best_threshold = float(ckpt.get("best_threshold", 0.5))
        print(f"[INFO] Loaded checkpoint from: {LAST_CHECKPOINT_PATH}")
        print(f"[INFO] Resume from epoch {start_epoch}, previous best_f1={best_f1:.4f}")
    else:
        print("[INFO] Start training from scratch.")

    if CONTINUE_TRAINING and os.path.exists(BEST_CHECKPOINT_PATH):
        best_ckpt = load_checkpoint(BEST_CHECKPOINT_PATH, DEVICE)
        best_model_state = best_ckpt["model_state_dict"]

    # ---------- 保存配置 ----------
    config = TrainConfig(
        input_dim=input_dim,
        hidden_dim_1=HIDDEN_DIM_1,
        hidden_dim_2=HIDDEN_DIM_2,
        dropout=DROPOUT,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        test_size=TEST_SIZE,
        numeric_features=safe_existing_columns(df, NUMERIC_FEATURES),
        categorical_features=safe_existing_columns(df, CATEGORICAL_FEATURES),
        feature_cols=feature_cols,
        label_col=LABEL_COL,
        weight_col=WEIGHT_COL,
        device=DEVICE,
    )
    save_json(asdict(config), CONFIG_PATH)

    # ---------- 训练循环 ----------
    for epoch in range(start_epoch, start_epoch + MAX_EPOCHS):
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
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_f1={metrics['f1']:.4f} | "
            f"val_recall={metrics['recall']:.4f} | "
            f"val_precision={metrics['precision']:.4f} | "
            f"threshold={threshold:.3f}"
        )

        save_checkpoint(
            LAST_CHECKPOINT_PATH,
            model,
            optimizer,
            epoch,
            best_epoch,
            best_f1,
            best_threshold,
            input_dim,
        )

        if SAVE_EPOCH_CHECKPOINT and (epoch % SAVE_EVERY_EPOCHS == 0):
            epoch_ckpt_path = os.path.join(OUTPUT_DIR, f"mlp_detector_epoch_{epoch}.pt")
            save_checkpoint(
                epoch_ckpt_path,
                model,
                optimizer,
                epoch,
                best_epoch,
                best_f1,
                best_threshold,
                input_dim,
            )

        if current_f1 > best_f1 + MIN_DELTA:
            best_f1 = current_f1
            best_epoch = epoch
            best_threshold = threshold
            best_metrics = metrics
            patience_count = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            save_checkpoint(
                BEST_CHECKPOINT_PATH,
                model,
                optimizer,
                epoch,
                best_epoch,
                best_f1,
                best_threshold,
                input_dim,
            )
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"[INFO] Early stopping at epoch {epoch}.")
                break

    if best_model_state is None:
        raise RuntimeError("训练失败，没有得到有效最佳模型。")

    # ---------- 恢复最佳模型 ----------
    model.load_state_dict(best_model_state)

    # ---------- 验证集输出 ----------
    val_prob = predict_probs(model, val_loader, DEVICE)
    val_out = val_df.copy()
    val_out["true_label"] = val_out[LABEL_COL]
    val_out["pred_error_prob"] = val_prob
    val_out["pred_label"] = (val_prob >= best_threshold).astype(int)
    val_out["pred_label_name"] = val_out["pred_label"].map({1: "error", 0: "correct"})
    val_out.to_csv(VAL_PRED_CSV, index=False, encoding="utf-8-sig")

    # ---------- 保存训练日志 ----------
    pd.DataFrame(train_logs).to_csv(TRAIN_LOG_CSV, index=False, encoding="utf-8-sig")

    # ---------- 保存最终 summary ----------
    summary = {
        "device": DEVICE,
        "train_size": int(len(train_df)),
        "val_size": int(len(val_df)),
        "nan_unlabeled_size": int(len(nan_df)),
        "start_epoch": int(start_epoch),
        "best_epoch": int(best_epoch),
        "best_f1": float(best_f1),
        "best_threshold": float(best_threshold),
        "metrics": best_metrics,
        "numeric_features_used": safe_existing_columns(df, NUMERIC_FEATURES),
        "categorical_features_used": safe_existing_columns(df, CATEGORICAL_FEATURES),
        "feature_cols_used": feature_cols,
        "continue_training": CONTINUE_TRAINING,
    }
    save_json(summary, METRICS_JSON)

    print("\n===== 训练完成 =====")
    print(f"设备: {DEVICE}")
    print(f"训练集大小: {len(train_df)}")
    print(f"验证集大小: {len(val_df)}")
    print(f"未定难例数(NaN): {len(nan_df)}")
    print(f"起始轮次: {start_epoch}")
    print(f"最佳轮次: {best_epoch}")
    print(f"最佳阈值: {best_threshold:.3f}")
    print(f"Accuracy:  {best_metrics['accuracy']:.4f}")
    print(f"Precision: {best_metrics['precision']:.4f}")
    print(f"Recall:    {best_metrics['recall']:.4f}")
    print(f"F1:        {best_metrics['f1']:.4f}")
    print(f"AUC:       {best_metrics['auc']}")
    print(f"Last checkpoint: {LAST_CHECKPOINT_PATH}")
    print(f"Best checkpoint: {BEST_CHECKPOINT_PATH}")
    print(f"Preprocessor:    {PREPROCESSOR_PATH}")
    print(f"Config:          {CONFIG_PATH}")
    print(f"Train log:       {TRAIN_LOG_CSV}")
    print(f"Val predictions: {VAL_PRED_CSV}")


if __name__ == "__main__":
    main()
import json
import os
from typing import List

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# 1. 用户配置区
# ============================================================

MODEL_DIR = "mlp_model_output_enhanced"

BEST_CHECKPOINT_PATH = os.path.join(MODEL_DIR, "mlp_detector_best.pt")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor_mlp.joblib")
CONFIG_PATH = os.path.join(MODEL_DIR, "mlp_config.json")

# 可替换成:
# 1) candidate_clustered_new.csv
# 2) full_table_features_new.csv
INPUT_FEATURE_CSV = "candidate_clustered_new.csv"

OUTPUT_SCORED_CSV = "candidate_scored_mlp_enhanced_new_somenew.csv"
OUTPUT_HIGH_RISK_CSV = "candidate_high_risk_mlp_enhanced_somenew.csv"
OUTPUT_HARD_CASES_CSV = "candidate_hard_cases_mlp_enhanced_somenew.csv"

# 阈值控制
USE_CHECKPOINT_THRESHOLD = True
MANUAL_ERROR_THRESHOLD = 0.50

HIGH_RISK_THRESHOLD = 0.80
HARD_CASE_LOW = 0.40
HARD_CASE_HIGH = 0.60

BATCH_SIZE = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 2. 模型定义（必须和训练时一致）
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
# 3. 工具函数
# ============================================================

def normalize_text_series_for_compare(s: pd.Series) -> pd.Series:
    return s.fillna("__NULL__").astype(str).str.strip()


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    和训练阶段完全一致的派生特征逻辑。
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


def ensure_required_files():
    required = [
        BEST_CHECKPOINT_PATH,
        PREPROCESSOR_PATH,
        CONFIG_PATH,
        INPUT_FEATURE_CSV,
    ]
    for path in required:
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到文件: {path}")


@torch.no_grad()
def predict_probs(model, X_array: np.ndarray, device: str, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    probs = []

    n = len(X_array)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = torch.tensor(X_array[start:end], dtype=torch.float32).to(device)
        logits = model(batch)
        batch_prob = torch.sigmoid(logits).cpu().numpy()
        probs.append(batch_prob)

    if not probs:
        return np.array([])
    return np.concatenate(probs, axis=0)


def summarize_results(out_df: pd.DataFrame, error_threshold: float):
    print("\n===== 推理统计 =====")
    print(f"设备: {DEVICE}")
    print(f"输入样本数: {len(out_df)}")
    print(f"使用阈值: {error_threshold:.4f}")
    print(f"预测为 error 的数量: {(out_df['pred_label'] == 1).sum()}")
    print(f"预测为 correct 的数量: {(out_df['pred_label'] == 0).sum()}")
    print(f"高风险错误数量 (>= {HIGH_RISK_THRESHOLD}): {(out_df['pred_error_prob'] >= HIGH_RISK_THRESHOLD).sum()}")
    print(
        f"难例数量 ({HARD_CASE_LOW} ~ {HARD_CASE_HIGH}): "
        f"{((out_df['pred_error_prob'] >= HARD_CASE_LOW) & (out_df['pred_error_prob'] <= HARD_CASE_HIGH)).sum()}"
    )

    print("\n错误概率分布概览:")
    print(out_df["pred_error_prob"].describe().to_string())

    show_cols = [c for c in ["row_id", "column", "value", "neighbor_majority_value", "pred_error_prob"] if c in out_df.columns]
    if show_cols:
        print("\nTop 20 高风险样本:")
        print(
            out_df.sort_values("pred_error_prob", ascending=False)[show_cols]
            .head(20)
            .to_string(index=False)
        )


def check_input_output_consistency(df_in: pd.DataFrame, df_out: pd.DataFrame):
    print("\n===== 输入输出一致性检查 =====")
    print(f"输入行数: {len(df_in)}")
    print(f"输出行数: {len(df_out)}")

    if "row_id" in df_out.columns and "column" in df_out.columns:
        dup_count = df_out.duplicated(subset=["row_id", "column"]).sum()
        print(f"输出中重复 (row_id, column) 数量: {dup_count}")

        if dup_count > 0:
            dup_rows = df_out[df_out.duplicated(subset=["row_id", "column"], keep=False)]
            print("\n[WARN] 发现重复 key，前10行如下：")
            print(dup_rows.head(10).to_string(index=False))


# ============================================================
# 4. 主流程
# ============================================================

def main():
    ensure_required_files()

    # ---------- 读取配置 ----------
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    numeric_features = config.get("numeric_features", [])
    categorical_features = config.get("categorical_features", [])
    feature_cols_from_train = config.get("feature_cols", None)

    hidden_dim_1 = config.get("hidden_dim_1", 128)
    hidden_dim_2 = config.get("hidden_dim_2", 64)
    dropout = config.get("dropout", 0.2)

    # ---------- 加载预处理器 ----------
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    # ---------- 加载 checkpoint ----------
    ckpt = torch.load(BEST_CHECKPOINT_PATH, map_location=DEVICE)

    # ---------- 初始化模型 ----------
    input_dim = ckpt["input_dim"]
    model = MLPDetector(
        input_dim=input_dim,
        hidden_dim_1=hidden_dim_1,
        hidden_dim_2=hidden_dim_2,
        dropout=dropout,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    # ---------- 判定阈值 ----------
    checkpoint_threshold = float(ckpt.get("best_threshold", MANUAL_ERROR_THRESHOLD))
    if USE_CHECKPOINT_THRESHOLD:
        error_threshold = checkpoint_threshold
    else:
        error_threshold = float(MANUAL_ERROR_THRESHOLD)

    # ---------- 读取输入特征表 ----------
    df = pd.read_csv(INPUT_FEATURE_CSV)

    # ---------- 动态生成派生特征（和训练一致） ----------
    df = add_derived_features(df)

    # ---------- 特征列检查 ----------
    if feature_cols_from_train is not None:
        expected_feature_cols = feature_cols_from_train
    else:
        expected_feature_cols = numeric_features + categorical_features

    missing_cols = [c for c in expected_feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"输入文件缺少训练时所需特征列: {missing_cols}")

    feature_cols = expected_feature_cols

    print("\n===== 实际用于推理的特征 =====")
    print(feature_cols)

    # ---------- transform ----------
    X = preprocessor.transform(df[feature_cols])
    if hasattr(X, "toarray"):
        X = X.toarray()

    # ---------- 推理 ----------
    prob = predict_probs(model, X, DEVICE, batch_size=BATCH_SIZE)

    # ---------- 输出结果 ----------
    out_df = df.copy()
    out_df["pred_error_prob"] = prob
    out_df["pred_label"] = (out_df["pred_error_prob"] >= error_threshold).astype(int)
    out_df["pred_label_name"] = out_df["pred_label"].map({1: "error", 0: "correct"})
    out_df["decision_threshold"] = error_threshold

    # ---------- 高风险错误 ----------
    high_risk_df = out_df[out_df["pred_error_prob"] >= HIGH_RISK_THRESHOLD].copy()

    # ---------- 难例：最不确定区域 ----------
    hard_cases_df = out_df[
        (out_df["pred_error_prob"] >= HARD_CASE_LOW) &
        (out_df["pred_error_prob"] <= HARD_CASE_HIGH)
    ].copy()

    # ---------- 保存 ----------
    out_df.to_csv(OUTPUT_SCORED_CSV, index=False, encoding="utf-8-sig")
    high_risk_df.to_csv(OUTPUT_HIGH_RISK_CSV, index=False, encoding="utf-8-sig")
    hard_cases_df.to_csv(OUTPUT_HARD_CASES_CSV, index=False, encoding="utf-8-sig")

    print(f"[OK] 全量打分结果已保存: {OUTPUT_SCORED_CSV}")
    print(f"[OK] 高风险错误已保存: {OUTPUT_HIGH_RISK_CSV}")
    print(f"[OK] 难例样本已保存: {OUTPUT_HARD_CASES_CSV}")

    check_input_output_consistency(df, out_df)
    summarize_results(out_df, error_threshold)


if __name__ == "__main__":
    main()
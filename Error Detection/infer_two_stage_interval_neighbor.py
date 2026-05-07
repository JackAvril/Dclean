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

MODEL_DIR = "two_stage_model_output"

# ---------- Stage1 ----------
STAGE1_BEST_CKPT = os.path.join(MODEL_DIR, "stage1_base_best.pt")
STAGE1_PREPROCESSOR = os.path.join(MODEL_DIR, "stage1_base_preprocessor.joblib")

# ---------- Stage2 ----------
STAGE2_BEST_CKPT = os.path.join(MODEL_DIR, "stage2_cal_best.pt")
STAGE2_PREPROCESSOR = os.path.join(MODEL_DIR, "stage2_cal_preprocessor.joblib")

# ---------- Common ----------
CONFIG_PATH = os.path.join(MODEL_DIR, "two_stage_config.json")

# 可替换成:
# 1) candidate_clustered_new.csv
# 2) full_table_features_new.csv
INPUT_FEATURE_CSV = "full_table_features_new.csv"

OUTPUT_SCORED_CSV = "full_table_candidate_scored_two_stage_interval_neighbor.csv"
OUTPUT_HIGH_RISK_CSV = "full_table_candidate_high_risk_two_stage_interval_neighbor.csv"
OUTPUT_HARD_CASES_CSV = "full_table_candidate_hard_cases_two_stage_interval_neighbor.csv"

# 是否使用 stage2 训练得到的最佳阈值
USE_STAGE2_CHECKPOINT_THRESHOLD = True
MANUAL_ERROR_THRESHOLD = 0.75

HIGH_RISK_THRESHOLD = 0.80
HARD_CASE_LOW = 0.40
HARD_CASE_HIGH = 0.60

BATCH_SIZE = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# 2. 区间二阶段配置
# ============================================================

ENABLE_INTERVAL_STAGE2 = True

# raw_pred_error_prob <= LOW：直接保留 Stage1
# raw_pred_error_prob >= HIGH：直接保留 Stage1
# LOW < prob < HIGH：才进入 Stage2
STAGE1_LOW_CONFIDENCE = 0.20
STAGE1_HIGH_CONFIDENCE = 0.90

# ============================================================
# 3. neighbor 轻修正配置
# ============================================================

ENABLE_NEIGHBOR_ADJUSTMENT = True

# 只有 neighbor 强支持当前值，且 value 不太罕见，且最终概率处于中间区间，才做轻修正
NEIGHBOR_RATIO_THRESHOLD = 0.90
NEIGHBOR_MIN_VALUE_FREQUENCY = 2.0

# 只修边界区间，避免误伤高置信真错误
NEIGHBOR_ADJUST_PROB_LOW = 0.3
NEIGHBOR_ADJUST_PROB_HIGH = 0.85

# 轻修正系数，不要太激进
NEIGHBOR_ADJUST_MULTIPLIER = 0.85

MIN_PROB = 0.0
MAX_PROB = 1.0


# ============================================================
# 4. 模型定义（必须和训练时一致）
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
# 5. 工具函数
# ============================================================

def normalize_text_series_for_compare(s: pd.Series) -> pd.Series:
    return s.fillna("__NULL__").astype(str).str.strip()


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    与训练阶段一致，动态生成派生特征。
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
        STAGE1_BEST_CKPT,
        STAGE1_PREPROCESSOR,
        STAGE2_BEST_CKPT,
        STAGE2_PREPROCESSOR,
        CONFIG_PATH,
        INPUT_FEATURE_CSV,
    ]
    for path in required:
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到文件: {path}")


def safe_existing_columns(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


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


def load_stage_model(ckpt_path: str, hidden_dim_1: int, hidden_dim_2: int, dropout: float):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    input_dim = ckpt["input_dim"]

    model = MLPDetector(
        input_dim=input_dim,
        hidden_dim_1=hidden_dim_1,
        hidden_dim_2=hidden_dim_2,
        dropout=dropout,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt


def check_required_feature_cols(df: pd.DataFrame, expected_feature_cols: List[str], stage_name: str):
    missing_cols = [c for c in expected_feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"[{stage_name}] 输入文件缺少训练时所需特征列: {missing_cols}")


def apply_interval_two_stage(
    out_df: pd.DataFrame,
    stage2_model,
    stage2_preprocessor,
    stage2_feature_cols: List[str],
    batch_size: int,
    device: str,
    low_conf: float,
    high_conf: float,
):
    """
    区间二阶段：
    只让中间不确定区间样本进入 Stage2，其余保留 Stage1 概率。
    """
    df = out_df.copy()

    # 默认：最终概率先等于 Stage1
    df["final_pred_error_prob"] = df["raw_pred_error_prob"]
    df["used_stage2"] = 0

    if not ENABLE_INTERVAL_STAGE2:
        # 全部走 Stage2
        X_stage2 = stage2_preprocessor.transform(df[stage2_feature_cols])
        if hasattr(X_stage2, "toarray"):
            X_stage2 = X_stage2.toarray()

        final_prob = predict_probs(stage2_model, X_stage2, device, batch_size=batch_size)
        df["final_pred_error_prob"] = final_prob
        df["used_stage2"] = 1
        return df

    mid_mask = (
        (df["raw_pred_error_prob"] > low_conf) &
        (df["raw_pred_error_prob"] < high_conf)
    )

    mid_idx = df[mid_mask].index.tolist()

    if len(mid_idx) > 0:
        X_stage2 = stage2_preprocessor.transform(df.loc[mid_idx, stage2_feature_cols])
        if hasattr(X_stage2, "toarray"):
            X_stage2 = X_stage2.toarray()

        stage2_prob = predict_probs(stage2_model, X_stage2, device, batch_size=batch_size)

        df.loc[mid_idx, "final_pred_error_prob"] = stage2_prob
        df.loc[mid_idx, "used_stage2"] = 1

    return df


def apply_neighbor_adjustment(out_df: pd.DataFrame):
    """
    对最终概率做轻微 neighbor 修正。
    注意这里只做轻修正，不做硬翻转。
    """
    df = out_df.copy()

    df["neighbor_adjustment_applied"] = 0
    df["neighbor_adjustment_reason"] = ""

    if not ENABLE_NEIGHBOR_ADJUSTMENT:
        return df

    value_freq = pd.to_numeric(df.get("value_frequency", 0), errors="coerce").fillna(0.0)
    neighbor_ratio = pd.to_numeric(df.get("neighbor_majority_ratio", 0), errors="coerce").fillna(0.0)
    equal_flag = pd.to_numeric(df.get("is_equal_to_neighbor_majority", 0), errors="coerce").fillna(0).astype(int)

    mask = (
        (equal_flag == 1) &
        (neighbor_ratio >= NEIGHBOR_RATIO_THRESHOLD) &
        (value_freq >= NEIGHBOR_MIN_VALUE_FREQUENCY) &
        (df["final_pred_error_prob"] >= NEIGHBOR_ADJUST_PROB_LOW) &
        (df["final_pred_error_prob"] <= NEIGHBOR_ADJUST_PROB_HIGH)
    )

    idx = df[mask].index.tolist()

    if len(idx) > 0:
        df.loc[idx, "final_pred_error_prob"] = (
            df.loc[idx, "final_pred_error_prob"] * NEIGHBOR_ADJUST_MULTIPLIER
        )
        df.loc[idx, "neighbor_adjustment_applied"] = 1
        df.loc[idx, "neighbor_adjustment_reason"] = f"neighbor_support*{NEIGHBOR_ADJUST_MULTIPLIER}"

    df["final_pred_error_prob"] = df["final_pred_error_prob"].clip(MIN_PROB, MAX_PROB)
    return df


def summarize_results(out_df: pd.DataFrame, final_threshold: float):
    print("\n===== 二阶段推理统计 =====")
    print(f"设备: {DEVICE}")
    print(f"输入样本数: {len(out_df)}")
    print(f"最终阈值: {final_threshold:.4f}")

    if "raw_pred_label" in out_df.columns:
        print(f"Stage1 预测为 error 的数量: {(out_df['raw_pred_label'] == 1).sum()}")
        print(f"Stage1 预测为 correct 的数量: {(out_df['raw_pred_label'] == 0).sum()}")

    if "used_stage2" in out_df.columns:
        print(f"进入 Stage2 的样本数: {int(out_df['used_stage2'].sum())}")
        print(f"保留 Stage1 的样本数: {int((out_df['used_stage2'] == 0).sum())}")

    if "neighbor_adjustment_applied" in out_df.columns:
        print(f"发生 neighbor 修正的样本数: {int(out_df['neighbor_adjustment_applied'].sum())}")

    if "final_pred_label" in out_df.columns:
        print(f"最终预测为 error 的数量: {(out_df['final_pred_label'] == 1).sum()}")
        print(f"最终预测为 correct 的数量: {(out_df['final_pred_label'] == 0).sum()}")

    print(f"高风险错误数量 (>= {HIGH_RISK_THRESHOLD}): {(out_df['final_pred_error_prob'] >= HIGH_RISK_THRESHOLD).sum()}")
    print(
        f"难例数量 ({HARD_CASE_LOW} ~ {HARD_CASE_HIGH}): "
        f"{((out_df['final_pred_error_prob'] >= HARD_CASE_LOW) & (out_df['final_pred_error_prob'] <= HARD_CASE_HIGH)).sum()}"
    )

    print("\n最终错误概率分布概览:")
    print(out_df["final_pred_error_prob"].describe().to_string())

    show_cols = [
        c for c in [
            "row_id", "column", "value",
            "raw_pred_error_prob", "final_pred_error_prob",
            "raw_pred_label", "final_pred_label",
            "used_stage2", "neighbor_adjustment_applied"
        ] if c in out_df.columns
    ]
    if show_cols:
        print("\nTop 20 高风险样本:")
        print(
            out_df.sort_values("final_pred_error_prob", ascending=False)[show_cols]
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
# 6. 主流程
# ============================================================

def main():
    ensure_required_files()

    # ---------- 读取统一配置 ----------
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    stage1_cfg = config["stage1"]
    stage2_cfg = config["stage2"]

    stage1_feature_cols = stage1_cfg["feature_cols"]
    stage1_hidden_dim_1 = stage1_cfg["hidden_dim_1"]
    stage1_hidden_dim_2 = stage1_cfg["hidden_dim_2"]
    stage1_dropout = stage1_cfg["dropout"]

    stage2_feature_cols = stage2_cfg["feature_cols"]
    stage2_hidden_dim_1 = stage2_cfg["hidden_dim_1"]
    stage2_hidden_dim_2 = stage2_cfg["hidden_dim_2"]
    stage2_dropout = stage2_cfg["dropout"]

    # ---------- 加载预处理器 ----------
    stage1_preprocessor = joblib.load(STAGE1_PREPROCESSOR)
    stage2_preprocessor = joblib.load(STAGE2_PREPROCESSOR)

    # ---------- 加载模型 ----------
    stage1_model, stage1_ckpt = load_stage_model(
        STAGE1_BEST_CKPT,
        hidden_dim_1=stage1_hidden_dim_1,
        hidden_dim_2=stage1_hidden_dim_2,
        dropout=stage1_dropout,
    )

    stage2_model, stage2_ckpt = load_stage_model(
        STAGE2_BEST_CKPT,
        hidden_dim_1=stage2_hidden_dim_1,
        hidden_dim_2=stage2_hidden_dim_2,
        dropout=stage2_dropout,
    )

    # ---------- 阈值 ----------
    stage1_threshold = float(stage1_ckpt.get("best_threshold", 0.5))
    stage2_checkpoint_threshold = float(stage2_ckpt.get("best_threshold", MANUAL_ERROR_THRESHOLD))

    if USE_STAGE2_CHECKPOINT_THRESHOLD:
        final_threshold = stage2_checkpoint_threshold
    else:
        final_threshold = float(MANUAL_ERROR_THRESHOLD)

    # ---------- 读取输入特征表 ----------
    df = pd.read_csv(INPUT_FEATURE_CSV)

    # ---------- 动态生成派生特征（与训练一致） ----------
    df = add_derived_features(df)

    # ========================================================
    # Stage1 推理
    # ========================================================
    check_required_feature_cols(df, stage1_feature_cols, stage_name="STAGE1")

    print("\n===== Stage1 实际用于推理的特征 =====")
    print(stage1_feature_cols)

    X_stage1 = stage1_preprocessor.transform(df[stage1_feature_cols])
    if hasattr(X_stage1, "toarray"):
        X_stage1 = X_stage1.toarray()

    raw_prob = predict_probs(stage1_model, X_stage1, DEVICE, batch_size=BATCH_SIZE)

    out_df = df.copy()
    out_df["raw_pred_error_prob"] = raw_prob
    out_df["raw_pred_label"] = (out_df["raw_pred_error_prob"] >= stage1_threshold).astype(int)
    out_df["raw_pred_label_name"] = out_df["raw_pred_label"].map({1: "error", 0: "correct"})
    out_df["stage1_decision_threshold"] = stage1_threshold

    # ========================================================
    # 区间二阶段
    # ========================================================
    check_required_feature_cols(out_df, stage2_feature_cols, stage_name="STAGE2")

    print("\n===== Stage2 实际用于推理的特征 =====")
    print(stage2_feature_cols)

    out_df = apply_interval_two_stage(
        out_df=out_df,
        stage2_model=stage2_model,
        stage2_preprocessor=stage2_preprocessor,
        stage2_feature_cols=stage2_feature_cols,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        low_conf=STAGE1_LOW_CONFIDENCE,
        high_conf=STAGE1_HIGH_CONFIDENCE,
    )

    # ========================================================
    # neighbor 轻修正
    # ========================================================
    out_df = apply_neighbor_adjustment(out_df)

    # ========================================================
    # 最终输出 label
    # ========================================================
    out_df["final_pred_label"] = (out_df["final_pred_error_prob"] >= final_threshold).astype(int)
    out_df["final_pred_label_name"] = out_df["final_pred_label"].map({1: "error", 0: "correct"})
    out_df["stage2_decision_threshold"] = final_threshold

    # ---------- 为兼容你后续评估代码，再额外给通用列名 ----------
    out_df["pred_error_prob"] = out_df["final_pred_error_prob"]
    out_df["pred_label"] = out_df["final_pred_label"]
    out_df["pred_label_name"] = out_df["final_pred_label_name"]
    out_df["decision_threshold"] = out_df["stage2_decision_threshold"]

    # ---------- 高风险错误 ----------
    high_risk_df = out_df[out_df["final_pred_error_prob"] >= HIGH_RISK_THRESHOLD].copy()

    # ---------- 难例：最不确定区域 ----------
    hard_cases_df = out_df[
        (out_df["final_pred_error_prob"] >= HARD_CASE_LOW) &
        (out_df["final_pred_error_prob"] <= HARD_CASE_HIGH)
    ].copy()

    # ---------- 保存 ----------
    out_df.to_csv(OUTPUT_SCORED_CSV, index=False, encoding="utf-8-sig")
    high_risk_df.to_csv(OUTPUT_HIGH_RISK_CSV, index=False, encoding="utf-8-sig")
    hard_cases_df.to_csv(OUTPUT_HARD_CASES_CSV, index=False, encoding="utf-8-sig")

    print(f"[OK] 全量打分结果已保存: {OUTPUT_SCORED_CSV}")
    print(f"[OK] 高风险错误已保存: {OUTPUT_HIGH_RISK_CSV}")
    print(f"[OK] 难例样本已保存: {OUTPUT_HARD_CASES_CSV}")

    check_input_output_consistency(df, out_df)
    summarize_results(out_df, final_threshold)


if __name__ == "__main__":
    main()
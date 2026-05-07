import json
import os
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ============================================================
# 1. 用户配置区
# ============================================================

INPUT_CSV = "candidate_infer_ready_v2.csv"
MODEL_DIR = "two_stage_model_output_v3"
OUTPUT_DIR = "two_stage_inference_output_v3"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_STAGE2 = True
USE_COLUMN_THRESHOLDS = True
USE_NEIGHBOR_ADJUSTMENT = True

# 保召回前提下尽量提精度：高误报列使用更严格阈值
COLUMN_THRESHOLDS = {
    # 放松：捞回 Sample 列 FN
    "Sample": 0.78,

    # 收紧：压这些列的 FP
    "ProviderNumber": 0.88,
    "PhoneNumber": 0.86,
    "Stateavg": 0.84,
    "Address1": 0.84,

    # 原来误报高发列，继续保守
    "EmergencyService": 0.88,
    "ZipCode": 0.84,
    "HospitalName": 0.84,
}

HARD_CASE_MARGIN = 0.06
HARD_CASE_LOW = 0.35
HARD_CASE_HIGH = 0.65
HIGH_RISK_THRESHOLD = 0.85

NEIGHBOR_SUPPORT_CURRENT_RATIO = 0.95
NEIGHBOR_SUPPORT_CURRENT_SUBTRACT = 0.08
FP_RISK_COLUMNS = {"EmergencyService",
    "Sample",
    "ZipCode",
    "HospitalName",
    "ProviderNumber",
    "PhoneNumber",
    "Stateavg",
    "Address1",}

OUTPUT_ALL_CSV = os.path.join(OUTPUT_DIR, "inference_all_predictions.csv")
OUTPUT_HARD_CASE_CSV = os.path.join(OUTPUT_DIR, "inference_hard_cases.csv")
OUTPUT_HIGH_RISK_CSV = os.path.join(OUTPUT_DIR, "inference_high_risk.csv")
OUTPUT_LLM_JSONL = os.path.join(OUTPUT_DIR, "hard_cases_for_llm.jsonl")

STAGE1_BEST_CKPT = os.path.join(MODEL_DIR, "stage1_base_best.pt")
STAGE1_PREPROCESSOR = os.path.join(MODEL_DIR, "stage1_base_preprocessor.joblib")
STAGE2_BEST_CKPT = os.path.join(MODEL_DIR, "stage2_cal_best.pt")
STAGE2_PREPROCESSOR = os.path.join(MODEL_DIR, "stage2_cal_preprocessor.joblib")
CONFIG_PATH = os.path.join(MODEL_DIR, "two_stage_config.json")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics_two_stage.json")


class MLPDetector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim_1: int, hidden_dim_2: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim_1, hidden_dim_2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim_2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in [
        "strong_rule_count", "candidate_generation_rule_count", "fd_like_count", "context_rule_count",
        "global_rule_count", "rare_value_count", "typo_rule_count", "pattern_rule_count", "schema_rule_count",
    ]:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    if "value" in out.columns and "neighbor_majority_value" in out.columns:
        value_series = normalize_text_series_for_compare(out["value"])
        neighbor_series = normalize_text_series_for_compare(out["neighbor_majority_value"])
        out["is_equal_to_neighbor_majority"] = (value_series == neighbor_series).astype(int)
    else:
        out["is_equal_to_neighbor_majority"] = safe_num_series(out, "is_equal_to_neighbor_majority", 0).astype(int)

    ratio = safe_num_series(out, "neighbor_majority_ratio", 0.0)
    out["neighbor_majority_ratio"] = ratio
    out["neighbor_agreement_score"] = np.where(out["is_equal_to_neighbor_majority"] == 1, ratio, -ratio).astype(float)
    out["has_neighbor_signal"] = (ratio > 0).astype(int)
    out["high_neighbor_consistency"] = (ratio >= 0.8).astype(int)

    out["rule_total_count"] = (out["strong_rule_count"] + out["candidate_generation_rule_count"]).astype(float)
    out["candidate_rule_ratio"] = np.where(out["rule_total_count"] > 0, out["candidate_generation_rule_count"] / (out["rule_total_count"] + 1e-8), 0.0)
    out["strong_rule_ratio"] = np.where(out["rule_total_count"] > 0, out["strong_rule_count"] / (out["rule_total_count"] + 1e-8), 0.0)
    out["context_rule_ratio"] = np.where(out["rule_total_count"] > 0, out["context_rule_count"] / (out["rule_total_count"] + 1e-8), 0.0)
    out["fd_like_ratio"] = np.where(out["rule_total_count"] > 0, out["fd_like_count"] / (out["rule_total_count"] + 1e-8), 0.0)
    out["global_rule_ratio"] = np.where(out["rule_total_count"] > 0, out["global_rule_count"] / (out["rule_total_count"] + 1e-8), 0.0)

    out["is_rare_value"] = safe_num_series(out, "value_frequency_rank", 999999).gt(10).astype(int)
    out["neighbor_disagree"] = ((out["has_neighbor_signal"] == 1) & (out["is_equal_to_neighbor_majority"] == 0)).astype(int)
    out["rule_strength_sum"] = (
        1.0 * out["strong_rule_count"] + 0.25 * out["candidate_generation_rule_count"] +
        0.4 * out["fd_like_count"] + 0.25 * out["context_rule_count"] + 0.15 * out["global_rule_count"] +
        0.5 * out["typo_rule_count"] + 0.15 * out["pattern_rule_count"] + 0.25 * out["schema_rule_count"]
    )
    out["candidate_rule_dominant"] = np.where(out["candidate_generation_rule_count"] > out["strong_rule_count"], "yes", "no")
    out["rare_only_signal"] = (
        (out["rare_value_count"] > 0) & (out["strong_rule_count"] <= 0) & (out["fd_like_count"] <= 0) &
        (out["context_rule_count"] <= 0) & (out["typo_rule_count"] <= 0) & (out["schema_rule_count"] <= 0)
    ).astype(int)

    for col in ["column", "semantic_type", "main_rule_type", "main_usage_role", "pattern_bucket", "rarity_bucket", "neighbor_bucket", "candidate_rule_dominant"]:
        if col not in out.columns:
            out[col] = "missing"
        out[col] = out[col].fillna("missing").astype(str)

    return out


def build_feature_matrix(df: pd.DataFrame, feature_cols, preprocessor):
    X = preprocessor.transform(df[feature_cols])
    if hasattr(X, "toarray"):
        X = X.toarray()
    return X


@torch.no_grad()
def predict_probs(model, X_np: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    probs = []
    for start in range(0, len(X_np), batch_size):
        end = min(start + batch_size, len(X_np))
        xb = torch.tensor(X_np[start:end], dtype=torch.float32, device=DEVICE)
        logits = model(xb)
        probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs, axis=0) if probs else np.array([])


def load_model(ckpt_path: str, cfg: Dict[str, Any]):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model = MLPDetector(cfg["input_dim"], cfg["hidden_dim_1"], cfg["hidden_dim_2"], cfg["dropout"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def apply_column_thresholds(df: pd.DataFrame, default_threshold: float) -> pd.Series:
    thresholds = pd.Series(np.full(len(df), default_threshold), index=df.index, dtype=float)
    if USE_COLUMN_THRESHOLDS:
        col_series = safe_str_series(df, "column", "missing")
        for col_name, t in COLUMN_THRESHOLDS.items():
            thresholds.loc[col_series == col_name] = float(t)
    return thresholds


def apply_neighbor_adjustment(df: pd.DataFrame, prob_col: str) -> pd.Series:
    prob = safe_num_series(df, prob_col, 0.0).copy()
    if not USE_NEIGHBOR_ADJUSTMENT:
        return prob
    support_current_mask = (
        (safe_num_series(df, "is_equal_to_neighbor_majority", 0).astype(int) == 1) &
        (safe_num_series(df, "neighbor_majority_ratio", 0.0) >= NEIGHBOR_SUPPORT_CURRENT_RATIO) &
        (safe_num_series(df, "strong_rule_count", 0.0) <= 0)
    )
    prob.loc[support_current_mask] = np.maximum(0.0, prob.loc[support_current_mask] - NEIGHBOR_SUPPORT_CURRENT_SUBTRACT)
    return prob


def mark_hard_cases(df: pd.DataFrame, final_thresholds: pd.Series) -> pd.DataFrame:
    out = df.copy()
    final_prob = safe_num_series(out, "final_pred_error_prob", 0.0)
    raw_label = safe_num_series(out, "raw_pred_label", 0).astype(int)
    final_label = safe_num_series(out, "final_pred_label", 0).astype(int)
    margin = np.abs(final_prob - final_thresholds)

    out["is_mid_prob"] = ((final_prob >= HARD_CASE_LOW) & (final_prob <= HARD_CASE_HIGH)).astype(int)
    out["is_near_threshold"] = (margin <= HARD_CASE_MARGIN).astype(int)
    out["is_stage_disagree"] = (raw_label != final_label).astype(int)
    out["rule_neighbor_conflict"] = (
        (safe_num_series(out, "strong_rule_count", 0.0) > 0) &
        (safe_num_series(out, "is_equal_to_neighbor_majority", 0).astype(int) == 1) &
        (safe_num_series(out, "neighbor_majority_ratio", 0.0) >= 0.90)
    ).astype(int)
    out["is_weak_only"] = (
        (safe_num_series(out, "rare_only_signal", 0).astype(int) == 1) |
        ((safe_num_series(out, "strong_rule_count", 0.0) <= 0) & (safe_num_series(out, "fd_like_count", 0.0) <= 0) &
         (safe_num_series(out, "typo_rule_count", 0.0) <= 0) & (safe_num_series(out, "schema_rule_count", 0.0) <= 0) &
         (safe_num_series(out, "candidate_generation_rule_count", 0.0) > 0))
    ).astype(int)
    out["is_fp_risk_col"] = safe_str_series(out, "column", "missing").isin(FP_RISK_COLUMNS).astype(int)
    out["high_prob_but_neighbor_support_current"] = (
        (final_prob >= HIGH_RISK_THRESHOLD) &
        (safe_num_series(out, "is_equal_to_neighbor_majority", 0).astype(int) == 1) &
        (safe_num_series(out, "neighbor_majority_ratio", 0.0) >= 0.95)
    ).astype(int)
    out["is_hard_case"] = (
        (out["is_mid_prob"] == 1) |
        (out["is_near_threshold"] == 1) |
        (out["is_stage_disagree"] == 1) |
        (out["rule_neighbor_conflict"] == 1) |
        ((out["is_weak_only"] == 1) & (out["is_fp_risk_col"] == 1)) |
        (out["high_prob_but_neighbor_support_current"] == 1)
    ).astype(int)
    return out


def build_llm_jsonl(df: pd.DataFrame, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            rec = {
                "row_id": row.get("row_id", None),
                "column": row.get("column", None),
                "value": row.get("value", None),
                "semantic_type": row.get("semantic_type", None),
                "raw_pred_error_prob": float(row.get("raw_pred_error_prob", 0.0)),
                "final_pred_error_prob": float(row.get("final_pred_error_prob", 0.0)),
                "raw_pred_label": int(row.get("raw_pred_label", 0)),
                "final_pred_label": int(row.get("final_pred_label", 0)),
                "hard_case_flags": {
                    "is_mid_prob": int(row.get("is_mid_prob", 0)),
                    "is_near_threshold": int(row.get("is_near_threshold", 0)),
                    "is_stage_disagree": int(row.get("is_stage_disagree", 0)),
                    "rule_neighbor_conflict": int(row.get("rule_neighbor_conflict", 0)),
                    "is_weak_only": int(row.get("is_weak_only", 0)),
                    "is_fp_risk_col": int(row.get("is_fp_risk_col", 0)),
                    "high_prob_but_neighbor_support_current": int(row.get("high_prob_but_neighbor_support_current", 0)),
                },
                "rule_context": {
                    "main_rule_type": row.get("main_rule_type", None),
                    "main_usage_role": row.get("main_usage_role", None),
                    "strong_rule_count": float(row.get("strong_rule_count", 0.0)),
                    "candidate_generation_rule_count": float(row.get("candidate_generation_rule_count", 0.0)),
                    "fd_like_count": float(row.get("fd_like_count", 0.0)),
                    "context_rule_count": float(row.get("context_rule_count", 0.0)),
                    "global_rule_count": float(row.get("global_rule_count", 0.0)),
                    "rare_value_count": float(row.get("rare_value_count", 0.0)),
                    "typo_rule_count": float(row.get("typo_rule_count", 0.0)),
                    "pattern_rule_count": float(row.get("pattern_rule_count", 0.0)),
                    "schema_rule_count": float(row.get("schema_rule_count", 0.0)),
                    "conflict_score": float(row.get("conflict_score", 0.0)),
                    "violation_count": float(row.get("violation_count", 0.0)),
                },
                "neighbor_context": {
                    "neighbor_majority_value": row.get("neighbor_majority_value", None),
                    "neighbor_majority_ratio": float(row.get("neighbor_majority_ratio", 0.0)),
                    "is_equal_to_neighbor_majority": int(row.get("is_equal_to_neighbor_majority", 0)),
                    "neighbor_agreement_score": float(row.get("neighbor_agreement_score", 0.0)),
                },
                "column_context": {
                    "value_frequency": float(row.get("value_frequency", 0.0)),
                    "value_frequency_rank": float(row.get("value_frequency_rank", 0.0)),
                    "prior_error_probability": float(row.get("prior_error_probability", 0.0)),
                    "posterior_error_probability": float(row.get("posterior_error_probability", 0.0)),
                    "pattern_bucket": row.get("pattern_bucket", None),
                    "rarity_bucket": row.get("rarity_bucket", None),
                    "neighbor_bucket": row.get("neighbor_bucket", None),
                }
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    ensure_dir(OUTPUT_DIR)
    config = load_json(CONFIG_PATH)
    metrics = load_json(METRICS_PATH)

    stage1_cfg = config["stage1"]
    stage2_cfg = config["stage2"]

    stage1_preprocessor = joblib.load(STAGE1_PREPROCESSOR)
    stage1_model = load_model(STAGE1_BEST_CKPT, stage1_cfg)
    stage1_threshold = float(metrics["stage1_best_threshold"])

    if USE_STAGE2:
        stage2_preprocessor = joblib.load(STAGE2_PREPROCESSOR)
        stage2_model = load_model(STAGE2_BEST_CKPT, stage2_cfg)
        stage2_threshold = float(metrics["stage2_best_threshold"])
    else:
        stage2_preprocessor = None
        stage2_model = None
        stage2_threshold = stage1_threshold

    df = pd.read_csv(INPUT_CSV)
    df = add_derived_features(df)

    X1 = build_feature_matrix(df, stage1_cfg["feature_cols"], stage1_preprocessor)
    raw_prob = predict_probs(stage1_model, X1)

    out_df = df.copy()
    out_df["raw_pred_error_prob"] = raw_prob
    out_df["raw_pred_label"] = (out_df["raw_pred_error_prob"] >= stage1_threshold).astype(int)
    out_df["raw_pred_label_name"] = out_df["raw_pred_label"].map({1: "error", 0: "correct"})
    out_df["stage1_decision_threshold"] = float(stage1_threshold)
    out_df["raw_margin_to_threshold"] = out_df["raw_pred_error_prob"] - float(stage1_threshold)
    out_df["stage1_high_conf_error"] = (out_df["raw_pred_error_prob"] >= max(float(stage1_threshold), 0.80)).astype(int)
    out_df["stage1_mid_zone"] = ((out_df["raw_pred_error_prob"] >= HARD_CASE_LOW) & (out_df["raw_pred_error_prob"] <= HARD_CASE_HIGH)).astype(int)

    if USE_STAGE2:
        X2 = build_feature_matrix(out_df, stage2_cfg["feature_cols"], stage2_preprocessor)
        final_prob = predict_probs(stage2_model, X2)
    else:
        final_prob = raw_prob

    out_df["final_pred_error_prob_before_adjust"] = final_prob
    out_df["final_pred_error_prob"] = apply_neighbor_adjustment(out_df, "final_pred_error_prob_before_adjust")

    final_thresholds = apply_column_thresholds(out_df, stage2_threshold)
    out_df["final_decision_threshold"] = final_thresholds
    out_df["final_pred_label"] = (out_df["final_pred_error_prob"] >= out_df["final_decision_threshold"]).astype(int)
    out_df["final_pred_label_name"] = out_df["final_pred_label"].map({1: "error", 0: "correct"})
    out_df["is_high_risk"] = (out_df["final_pred_error_prob"] >= HIGH_RISK_THRESHOLD).astype(int)

    out_df = mark_hard_cases(out_df, final_thresholds)
    out_df.to_csv(OUTPUT_ALL_CSV, index=False, encoding="utf-8-sig")

    hard_df = out_df[out_df["is_hard_case"] == 1].copy()
    hard_df.to_csv(OUTPUT_HARD_CASE_CSV, index=False, encoding="utf-8-sig")

    high_risk_df = out_df[out_df["is_high_risk"] == 1].copy()
    high_risk_df.to_csv(OUTPUT_HIGH_RISK_CSV, index=False, encoding="utf-8-sig")

    build_llm_jsonl(hard_df, OUTPUT_LLM_JSONL)

    print(f"[OK] 全量推理结果已保存: {OUTPUT_ALL_CSV}")
    print(f"[OK] hard case 已保存: {OUTPUT_HARD_CASE_CSV}")
    print(f"[OK] high risk 已保存: {OUTPUT_HIGH_RISK_CSV}")
    print(f"[OK] LLM 回流输入已保存: {OUTPUT_LLM_JSONL}")
    print(f"[INFO] 总样本数: {len(out_df)}")
    print(f"[INFO] 预测 error 数: {int(out_df['final_pred_label'].sum())}")
    print(f"[INFO] hard case 数: {len(hard_df)}")
    print(f"[INFO] high risk 数: {len(high_risk_df)}")


if __name__ == "__main__":
    main()
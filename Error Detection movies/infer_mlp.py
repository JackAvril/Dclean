import json
import os
from typing import Dict, Any, List

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

INPUT_CSV = "candidate_infer_ready_movies.csv"
MODEL_DIR = "two_stage_model_output_movies_v4_with_verifier"
OUTPUT_DIR = "two_stage_inference_output_movies_v4_with_verifier"

# 安全保护：这个推理脚本只适用于 precision 版候选池。
# 如果误用了 hybrid 版 6万+ 候选池，直接停止，避免再次低精度灾难。
ABORT_ON_OVERSIZED_CANDIDATE_SET = True
MAX_EXPECTED_CANDIDATES_FOR_PRECISION_MODE = 30000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

USE_STAGE2 = True
USE_COLUMN_THRESHOLDS = True
USE_NEIGHBOR_ADJUSTMENT = True

USE_FP_VERIFIER = False  # precision版默认关闭失效 verifier，先用列阈值和后处理控 FP
VERIFIER_ONLY_ON_DETECTOR_ERROR = True

VERIFIER_THRESHOLD_MODE = "manual"
MANUAL_VERIFIER_THRESHOLD = 0.50
MIN_VERIFIER_THRESHOLD = 0.50
DEFAULT_VERIFIER_THRESHOLD = 0.50
PRINT_VERIFIER_DEBUG = True

# 高 precision 基础列阈值保持不变
COLUMN_THRESHOLDS = {
    # 高质量候选列：适度降低阈值，恢复 recall
    "RatingCount": 0.55,
    "ReviewCount": 0.58,
    "Year": 0.62,
    "Filming Locations": 0.55,
    "Creator": 0.58,
    "Actors": 0.60,
    "RatingValue": 0.62,
    "Duration": 0.62,

    # 中等风险列：保持相对保守
    "Release Date": 0.80,
    "Cast": 0.74,
    "Genre": 0.78,

    # 高 FP 长尾/枚举列：继续严格
    "Language": 0.98,
    "Country": 0.98,
    "Description": 0.92,
    "Name": 0.92,
    "Director": 0.90,
}

# 继续小步放松：只动真正高价值 FN 组合
# 相比上一版再小降 0.01~0.02
COND_LOWER_THRESHOLD = {
    # 只放松在 precision 候选池内已经验证较可靠的列
    "RatingCount": 0.52,
    "ReviewCount": 0.52,
    "Year": 0.55,
    "Filming Locations": 0.52,
    "Creator": 0.55,
    "Actors": 0.56,
    "RatingValue": 0.58,
    "Duration": 0.58,
}

GROUP_VERIFIER_THRESHOLD_BY_COLUMN = {
    "Name": 0.60,
    "Year": 0.52,
    "Release Date": 0.52,
    "Duration": 0.54,
    "RatingValue": 0.54,
    "RatingCount": 0.56,
    "ReviewCount": 0.56,
    "Director": 0.58,
    "Creator": 0.58,
    "Actors": 0.58,
    "Cast": 0.58,
    "Language": 0.56,
    "Country": 0.56,
    "Genre": 0.56,
    "Filming Locations": 0.60,
    "Description": 0.62,
}

GROUP_VERIFIER_THRESHOLD_BY_RULE = {
    "rare_value": 0.58,
    "rare_pattern": 0.60,
    "functional_dependency": 0.55,
    "soft_functional_dependency": 0.56,
}

HARD_CASE_MARGIN = 0.05
HARD_CASE_LOW = 0.45
HARD_CASE_HIGH = 0.55
HIGH_RISK_THRESHOLD = 0.85

NEIGHBOR_SUPPORT_CURRENT_RATIO = 0.95
NEIGHBOR_SUPPORT_CURRENT_SUBTRACT = 0.05

FP_RISK_COLUMNS = {"Year", "Release Date", "Duration", "RatingValue", "RatingCount", "ReviewCount", "Actors", "Cast", "Language", "Country", "Genre"}

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

VERIFIER_MODEL_PATH = os.path.join(MODEL_DIR, "fp_verifier_model.joblib")
VERIFIER_PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "fp_verifier_preprocessor.joblib")
VERIFIER_CONFIG_PATH = os.path.join(MODEL_DIR, "fp_verifier_config.json")
VERIFIER_METRICS_PATH = os.path.join(MODEL_DIR, "verifier_metrics.json")


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


def ensure_dir(path: str):
    if path:
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

    if "value" in out.columns and "neighbor_majority_value" in out.columns:
        value_series = normalize_text_series_for_compare(out["value"])
        neighbor_series = normalize_text_series_for_compare(out["neighbor_majority_value"])
        out["is_equal_to_neighbor_majority"] = (value_series == neighbor_series).astype(int)
    else:
        out["is_equal_to_neighbor_majority"] = 0

    ratio = pd.to_numeric(out.get("neighbor_majority_ratio", 0.0), errors="coerce").fillna(0.0)
    out["neighbor_majority_ratio"] = ratio
    out["neighbor_agreement_score"] = np.where(out["is_equal_to_neighbor_majority"] == 1, ratio, -ratio).astype(float)

    numeric_defaults = [
        "strong_rule_count", "candidate_generation_rule_count", "fd_like_count", "context_rule_count",
        "global_rule_count", "rare_value_count", "typo_rule_count", "pattern_rule_count", "schema_rule_count",
        "movie_typed_rule_count", "movie_consistency_rule_count",
        "parsed_year", "parsed_release_year", "duration_minutes", "rating_value_float", "count_value_int",
        "review_total", "list_token_count", "has_mojibake_or_control_chars", "movie_row_year",
        "movie_row_release_year", "movie_row_release_year_gap", "movie_row_duration_minutes",
        "movie_row_rating_value_float", "movie_row_rating_count_int", "movie_row_review_total",
        "movie_row_review_to_rating_ratio", "movie_row_actors_cast_overlap",
    ]
    for col in numeric_defaults:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    out["rule_total_count"] = out["strong_rule_count"] + out["candidate_generation_rule_count"]
    out["candidate_rule_ratio"] = np.where(out["rule_total_count"] > 0, out["candidate_generation_rule_count"] / (out["rule_total_count"] + 1e-8), 0.0)
    out["strong_rule_ratio"] = np.where(out["rule_total_count"] > 0, out["strong_rule_count"] / (out["rule_total_count"] + 1e-8), 0.0)
    out["context_rule_ratio"] = np.where(out["rule_total_count"] > 0, out["context_rule_count"] / (out["rule_total_count"] + 1e-8), 0.0)
    out["fd_like_ratio"] = np.where(out["rule_total_count"] > 0, out["fd_like_count"] / (out["rule_total_count"] + 1e-8), 0.0)
    out["global_rule_ratio"] = np.where(out["rule_total_count"] > 0, out["global_rule_count"] / (out["rule_total_count"] + 1e-8), 0.0)
    out["movie_typed_rule_ratio"] = np.where(out["rule_total_count"] > 0, out["movie_typed_rule_count"] / (out["rule_total_count"] + 1e-8), 0.0)
    out["movie_consistency_rule_ratio"] = np.where(out["rule_total_count"] > 0, out["movie_consistency_rule_count"] / (out["rule_total_count"] + 1e-8), 0.0)
    out["movie_rule_total_count"] = out["movie_typed_rule_count"] + out["movie_consistency_rule_count"]
    out["has_movie_rule_signal"] = (out["movie_rule_total_count"] > 0).astype(int)

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
        0.15 * out["rare_value_count"] +
        0.5 * out["typo_rule_count"] +
        0.15 * out["pattern_rule_count"] +
        0.25 * out["schema_rule_count"] +
        0.45 * out["movie_typed_rule_count"] +
        0.65 * out["movie_consistency_rule_count"]
    )

    out["candidate_rule_dominant"] = np.where(out["candidate_generation_rule_count"] > out["strong_rule_count"], "yes", "no")
    out["rare_only_signal"] = (
        (out["rare_value_count"] > 0) &
        (out["strong_rule_count"] <= 0) & (out["fd_like_count"] <= 0) & (out["context_rule_count"] <= 0) &
        (out["typo_rule_count"] <= 0) & (out["schema_rule_count"] <= 0) &
        (out["movie_typed_rule_count"] <= 0) & (out["movie_consistency_rule_count"] <= 0)
    ).astype(int)
    out["is_weak_only"] = (
        (out["rare_only_signal"] == 1) |
        ((out["strong_rule_count"] <= 0) & (out["fd_like_count"] <= 0) & (out["typo_rule_count"] <= 0) &
         (out["schema_rule_count"] <= 0) & (out["movie_typed_rule_count"] <= 0) &
         (out["movie_consistency_rule_count"] <= 0) & (out["candidate_generation_rule_count"] > 0))
    ).astype(int)

    out["is_movie_time_col"] = out.get("column", pd.Series(["missing"] * len(out))).astype(str).isin(["Year", "Release Date"]).astype(int)
    out["is_movie_rating_col"] = out.get("column", pd.Series(["missing"] * len(out))).astype(str).isin(["RatingValue", "RatingCount", "ReviewCount"]).astype(int)
    out["is_movie_people_col"] = out.get("column", pd.Series(["missing"] * len(out))).astype(str).isin(["Director", "Creator", "Actors", "Cast"]).astype(int)
    out["is_movie_list_col"] = out.get("column", pd.Series(["missing"] * len(out))).astype(str).isin(["Language", "Country", "Genre", "Creator", "Actors", "Cast"]).astype(int)
    out["is_movie_text_col"] = out.get("column", pd.Series(["missing"] * len(out))).astype(str).isin(["Name", "Filming Locations", "Description"]).astype(int)

    risk_cols = {"Release Date", "Year", "Duration", "RatingValue", "RatingCount", "ReviewCount", "Actors", "Cast", "Language", "Country", "Genre"}
    out["is_fp_risk_col"] = out.get("column", pd.Series(["missing"] * len(out))).astype(str).isin(risk_cols).astype(int)

    for col in [
        "column", "semantic_type", "detected_type", "main_rule_type", "main_usage_role", "pattern_bucket", "rarity_bucket",
        "neighbor_bucket", "candidate_rule_dominant", "movie_rule_bucket", "movie_field_bucket", "year_bucket",
        "release_year_gap_bucket", "duration_bucket", "rating_bucket", "count_bucket", "review_ratio_bucket",
        "actors_cast_overlap_bucket", "list_bucket", "text_bucket", "parsed_release_country", "movie_row_release_country",
        "list_token_count_bucket", "text_length_bucket"
    ]:
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


def load_detector_model(ckpt_path: str, cfg: Dict[str, Any]):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model = MLPDetector(cfg["input_dim"], cfg["hidden_dim_1"], cfg["hidden_dim_2"], cfg["dropout"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def apply_base_column_thresholds(df: pd.DataFrame, default_threshold: float) -> pd.Series:
    thresholds = pd.Series(np.full(len(df), default_threshold), index=df.index, dtype=float)
    if USE_COLUMN_THRESHOLDS:
        col_series = safe_str_series(df, "column", "missing")
        for col_name, t in COLUMN_THRESHOLDS.items():
            thresholds.loc[col_series == col_name] = float(t)
    return thresholds


def apply_conditional_detector_thresholds(df: pd.DataFrame, base_thresholds: pd.Series) -> pd.Series:
    thresholds = base_thresholds.copy()
    col = safe_str_series(df, "column", "missing")
    rule = safe_str_series(df, "main_rule_type", "missing")
    nb = safe_str_series(df, "neighbor_bucket", "missing")
    movie_rule_signal = safe_num_series(df, "movie_rule_total_count", 0.0) > 0

    high_value_cols = ["Year", "Release Date", "Duration", "RatingValue", "RatingCount", "ReviewCount", "Actors", "Cast", "Language", "Country", "Genre"]
    for c in high_value_cols:
        if c not in COND_LOWER_THRESHOLD:
            continue
        mask = col.eq(c) & (
            movie_rule_signal |
            rule.isin(["movie_year_range", "release_date_format", "release_year_consistency", "duration_format", "rating_value_range", "rating_count_format", "review_count_format", "actors_cast_overlap", "functional_dependency", "soft_functional_dependency", "dominant_value_by_context", "dominant_value_by_context_pair"])
        ) & nb.isin(["strong_support_other", "medium_support_other", "no_neighbor_signal", "weak_support_other"])
        thresholds.loc[mask] = np.minimum(thresholds.loc[mask], float(COND_LOWER_THRESHOLD[c]))

    support_current_mask = nb.isin(["strong_support_current", "weak_support_current"]) & (safe_num_series(df, "strong_rule_count", 0.0) <= 0)
    thresholds.loc[support_current_mask] = np.maximum(thresholds.loc[support_current_mask], base_thresholds.loc[support_current_mask])
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
    prob.loc[support_current_mask] = np.maximum(
        0.0,
        prob.loc[support_current_mask] - NEIGHBOR_SUPPORT_CURRENT_SUBTRACT
    )
    return prob


def load_verifier_assets():
    if not USE_FP_VERIFIER:
        return None, None, None, DEFAULT_VERIFIER_THRESHOLD, "disabled"

    verifier_model = joblib.load(VERIFIER_MODEL_PATH)
    verifier_preprocessor = joblib.load(VERIFIER_PREPROCESSOR_PATH)
    verifier_config = load_json(VERIFIER_CONFIG_PATH)

    threshold_source = "default"
    metrics_threshold = None
    config_threshold = None

    if os.path.exists(VERIFIER_METRICS_PATH):
        verifier_metrics = load_json(VERIFIER_METRICS_PATH)
        metrics_threshold = verifier_metrics.get("best_verifier_threshold_on_final_val", None)

    if "threshold" in verifier_config:
        config_threshold = verifier_config["threshold"]

    if VERIFIER_THRESHOLD_MODE == "manual":
        verifier_threshold = float(MANUAL_VERIFIER_THRESHOLD)
        threshold_source = "manual"
    elif VERIFIER_THRESHOLD_MODE == "metrics":
        if metrics_threshold is not None:
            verifier_threshold = float(metrics_threshold)
            threshold_source = "metrics"
        elif config_threshold is not None:
            verifier_threshold = float(config_threshold)
            threshold_source = "config"
        else:
            verifier_threshold = float(DEFAULT_VERIFIER_THRESHOLD)
            threshold_source = "default"
    elif VERIFIER_THRESHOLD_MODE == "max_with_min":
        if metrics_threshold is not None:
            candidate = float(metrics_threshold)
            threshold_source = "metrics"
        elif config_threshold is not None:
            candidate = float(config_threshold)
            threshold_source = "config"
        else:
            candidate = float(DEFAULT_VERIFIER_THRESHOLD)
            threshold_source = "default"
        verifier_threshold = max(float(MIN_VERIFIER_THRESHOLD), candidate)
        threshold_source = f"{threshold_source}+min_guard"
    else:
        raise ValueError(f"不支持的 VERIFIER_THRESHOLD_MODE: {VERIFIER_THRESHOLD_MODE}")

    return verifier_model, verifier_preprocessor, verifier_config, verifier_threshold, threshold_source


def build_verifier_feature_matrix(df: pd.DataFrame, feature_cols: List[str], preprocessor):
    X = preprocessor.transform(df[feature_cols])
    if hasattr(X, "toarray"):
        X = X.toarray()
    return X


def predict_verifier_keep_prob(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "classes_"):
        classes = list(model.classes_)
        if len(classes) == 1:
            only_class = int(classes[0])
            return np.full(X.shape[0], float(only_class), dtype=float)

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X)
            class_to_index = {c: i for i, c in enumerate(classes)}
            if 1 in class_to_index:
                return prob[:, class_to_index[1]]
            return np.zeros(X.shape[0], dtype=float)

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)
        if prob.ndim == 2:
            if prob.shape[1] >= 2:
                return prob[:, 1]
            if prob.shape[1] == 1:
                return prob[:, 0]

    if hasattr(model, "decision_function"):
        score = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-score))

    pred = model.predict(X)
    return np.asarray(pred, dtype=float)


def print_verifier_debug_stats(keep_prob: np.ndarray, verifier_threshold: float):
    if len(keep_prob) == 0:
        return

    s = pd.Series(keep_prob)
    print("[DEBUG] verifier_keep_error_prob describe:")
    print(s.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_string())
    print(f"[DEBUG] count keep_prob < threshold({verifier_threshold:.4f}): {int((s < verifier_threshold).sum())}")
    print(f"[DEBUG] count keep_prob >= threshold({verifier_threshold:.4f}): {int((s >= verifier_threshold).sum())}")


def get_group_verifier_threshold(row: pd.Series, default_threshold: float) -> float:
    col = str(row.get("column", "missing"))
    rule = str(row.get("main_rule_type", "missing"))
    nb = str(row.get("neighbor_bucket", "missing"))
    th = default_threshold
    if col in GROUP_VERIFIER_THRESHOLD_BY_COLUMN:
        th = max(th, GROUP_VERIFIER_THRESHOLD_BY_COLUMN[col])
    if rule in GROUP_VERIFIER_THRESHOLD_BY_RULE:
        th = max(th, GROUP_VERIFIER_THRESHOLD_BY_RULE[rule])
    if rule in {"functional_dependency", "soft_functional_dependency"} and nb in {"strong_support_current", "weak_support_current"}:
        th = max(th, 0.60)
    if nb == "weak_support_other" and col in {"Name", "Director", "Creator", "Filming Locations", "Description"}:
        th = max(th, 0.64)
    if col in {"Year", "Release Date", "Duration", "RatingValue", "RatingCount", "ReviewCount"}:
        th = min(th, max(default_threshold, 0.54))
    return float(th)

def apply_fp_verifier(
    df: pd.DataFrame,
    verifier_model,
    verifier_preprocessor,
    verifier_config: Dict[str, Any],
    verifier_threshold: float,
) -> pd.DataFrame:
    out = df.copy()
    out["verifier_keep_error_prob"] = np.nan
    out["verifier_decision"] = "skip"
    out["verifier_threshold_used"] = verifier_threshold
    out["group_verifier_threshold"] = verifier_threshold

    if verifier_model is None:
        out["final_pred_label"] = out["detector_pred_label"].astype(int)
        out["final_pred_label_name"] = out["final_pred_label"].map({1: "error", 0: "correct"})
        return out

    feature_cols = verifier_config["feature_cols"]

    col_series = safe_str_series(out, "column", "missing")
    rule_series = safe_str_series(out, "main_rule_type", "missing")
    neighbor_series = safe_str_series(out, "neighbor_bucket", "missing")

    detector_positive = safe_num_series(out, "detector_pred_label", 0).astype(int) == 1
    fd_mask = rule_series.isin({"functional_dependency", "soft_functional_dependency"})
    risk_col_mask = col_series.isin({"Year", "Release Date", "Duration", "RatingValue", "RatingCount", "ReviewCount", "Actors", "Cast", "Language", "Country", "Genre"})
    neighbor_mask = neighbor_series.isin({"strong_support_current", "weak_support_current", "weak_support_other"})

    verifier_mask = detector_positive & (fd_mask | risk_col_mask | neighbor_mask)

    if verifier_mask.sum() == 0:
        out["final_pred_label"] = out["detector_pred_label"].astype(int)
        out["final_pred_label_name"] = out["final_pred_label"].map({1: "error", 0: "correct"})
        return out

    verifier_df = out.loc[verifier_mask].copy()
    Xv = build_verifier_feature_matrix(verifier_df, feature_cols, verifier_preprocessor)
    keep_prob = predict_verifier_keep_prob(verifier_model, Xv)

    verifier_df["verifier_keep_error_prob"] = keep_prob
    verifier_df["group_verifier_threshold"] = verifier_df.apply(
        lambda row: get_group_verifier_threshold(row, verifier_threshold), axis=1
    )
    verifier_df["verifier_decision"] = np.where(
        verifier_df["verifier_keep_error_prob"] >= verifier_df["group_verifier_threshold"],
        "keep_error",
        "reject_error"
    )

    out.loc[verifier_df.index, "verifier_keep_error_prob"] = verifier_df["verifier_keep_error_prob"]
    out.loc[verifier_df.index, "group_verifier_threshold"] = verifier_df["group_verifier_threshold"]
    out.loc[verifier_df.index, "verifier_decision"] = verifier_df["verifier_decision"]

    out["final_pred_label"] = out["detector_pred_label"].astype(int)
    out.loc[verifier_df.index, "final_pred_label"] = (
        verifier_df["verifier_decision"] == "keep_error"
    ).astype(int)
    out["final_pred_label_name"] = out["final_pred_label"].map({1: "error", 0: "correct"})

    return out


def apply_column_post_rules(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    col = safe_str_series(out, "column", "missing")
    rule = safe_str_series(out, "main_rule_type", "missing")
    nb = safe_str_series(out, "neighbor_bucket", "missing")
    final_label = safe_num_series(out, "final_pred_label", 0).astype(int)
    detector_label = safe_num_series(out, "detector_pred_label", 0).astype(int)
    detector_prob = safe_num_series(out, "detector_pred_error_prob", 0.0)

    movie_rule_cnt = safe_num_series(out, "movie_rule_total_count", 0.0)
    movie_typed_cnt = safe_num_series(out, "movie_typed_rule_count", 0.0)
    movie_cons_cnt = safe_num_series(out, "movie_consistency_rule_count", 0.0)

    strong_rule_count = safe_num_series(out, "strong_rule_count", 0.0)
    fd_like_count = safe_num_series(out, "fd_like_count", 0.0)
    context_rule_count = safe_num_series(out, "context_rule_count", 0.0)
    rare_value_count = safe_num_series(out, "rare_value_count", 0.0)
    pattern_rule_count = safe_num_series(out, "pattern_rule_count", 0.0)
    schema_rule_count = safe_num_series(out, "schema_rule_count", 0.0)

    # ============================================================
    # A. 先压制当前评估中稳定出现的 FP 组合
    #    原则：
    #      1) 不压 RatingCount rare_value，因为它贡献大量 TP；
    #      2) 只压在 fp_details 中明确高 FP、低收益的组合；
    #      3) 压制阶段放在恢复阶段之前，后面只恢复明确高可信规则。
    # ============================================================

    # A1. Language / Country：precision 版里仍然保持严格。
    # 多值枚举列不能被 rare / FD / context dominant 单独判错。
    mask_lang_country_weak = (
        final_label.eq(1) &
        col.isin(["Language", "Country"]) &
        rule.isin([
            "rare_value",
            "rare_pattern",
            "global_dominant_value",
            "dominant_value_by_context",
            "dominant_value_by_context_pair",
            "functional_dependency",
            "soft_functional_dependency",
        ])
    )
    out.loc[mask_lang_country_weak, "final_pred_label"] = 0

    # A2. Cast + actors_cast_overlap：当前 FP 明显集中，先压掉。
    # Cast 的缺失类 not_null 会在后面单独恢复，不和 overlap 混在一起。
    mask_cast_overlap_fp = (
        final_label.eq(1) &
        col.eq("Cast") &
        rule.eq("actors_cast_overlap")
    )
    out.loc[mask_cast_overlap_fp, "final_pred_label"] = 0

    # A3. Release Date + release_year_consistency：当前是稳定 FP 来源。
    # 注意：Release Date + not_null 会在后面恢复。
    mask_release_year_consistency_fp = (
        final_label.eq(1) &
        col.eq("Release Date") &
        rule.eq("release_year_consistency")
    )
    out.loc[mask_release_year_consistency_fp, "final_pred_label"] = 0

    # A4. Duration / RatingValue 的 FD：当前 FP 明显多于 TP。
    # 格式/range/pattern 类规则后面会保留或恢复。
    mask_duration_fd_fp = (
        final_label.eq(1) &
        col.eq("Duration") &
        rule.isin(["functional_dependency", "soft_functional_dependency"])
    )
    out.loc[mask_duration_fd_fp, "final_pred_label"] = 0

    mask_rating_value_fd_fp = (
        final_label.eq(1) &
        col.eq("RatingValue") &
        rule.isin(["functional_dependency", "soft_functional_dependency"])
    )
    out.loc[mask_rating_value_fd_fp, "final_pred_label"] = 0

    # A5. Duration rare_value：当前 FP 略多，先保守压掉；
    # Duration rare_pattern 是高价值 FN 来源，后面恢复。
    mask_duration_rare_value_fp = (
        final_label.eq(1) &
        col.eq("Duration") &
        rule.eq("rare_value")
    )
    out.loc[mask_duration_rare_value_fp, "final_pred_label"] = 0

    # A6. RatingCount rare_pattern：当前主要是 FP；但 rare_value 保留。
    mask_rating_count_rare_pattern_fp = (
        final_label.eq(1) &
        col.eq("RatingCount") &
        rule.eq("rare_pattern")
    )
    out.loc[mask_rating_count_rare_pattern_fp, "final_pred_label"] = 0

    # A7. ReviewCount 的 review_rating_count_order 当前贡献少量 FP，先压掉。
    # ReviewCount 的 not_null / review_count_format 后面恢复。
    mask_review_count_order_fp = (
        final_label.eq(1) &
        col.eq("ReviewCount") &
        rule.eq("review_rating_count_order")
    )
    out.loc[mask_review_count_order_fp, "final_pred_label"] = 0

    # A8. Release Date 的 weak rare/FD/context 继续保守。
    # 但 not_null 不在这里压制，后面会恢复。
    mask_release_date_weak = (
        final_label.eq(1) &
        col.eq("Release Date") &
        rule.isin([
            "rare_value",
            "rare_pattern",
            "functional_dependency",
            "soft_functional_dependency",
            "dominant_value_by_context",
            "dominant_value_by_context_pair",
        ]) &
        (movie_cons_cnt <= 0) &
        (movie_typed_cnt <= 0) &
        (detector_prob < 0.86)
    )
    out.loc[mask_release_date_weak, "final_pred_label"] = 0

    # A9. 长尾文本/人名/列表字段的 weak 规则继续压制。
    # 但明确 not_null 缺失类会在后面按列恢复。
    long_tail_cols = [
        "Name",
        "Director",
        "Creator",
        "Actors",
        "Cast",
        "Filming Locations",
        "Description",
        "Genre",
    ]
    weak_rules = [
        "rare_value",
        "rare_pattern",
        "global_dominant_value",
        "functional_dependency",
        "soft_functional_dependency",
        "dominant_value_by_context",
        "dominant_value_by_context_pair",
    ]
    mask_text_people_weak = (
        final_label.eq(1) &
        col.isin(long_tail_cols) &
        rule.isin(weak_rules) &
        (movie_cons_cnt <= 0) &
        (movie_typed_cnt <= 0) &
        nb.isin([
            "no_neighbor_signal",
            "weak_support_other",
            "weak_support_current",
            "strong_support_current",
        ]) &
        (detector_prob < 0.86)
    )
    out.loc[mask_text_people_weak, "final_pred_label"] = 0

    # A10. 全局 weak-only 低概率压制。
    # 排除当前高价值数值列，避免误杀 RatingCount / ReviewCount / Year / Duration / RatingValue。
    weak_only = (
        (strong_rule_count <= 0) &
        (movie_rule_cnt <= 0) &
        (schema_rule_count <= 0) &
        (fd_like_count + context_rule_count + rare_value_count + pattern_rule_count > 0)
    )
    safe_numeric_cols = ["RatingCount", "ReviewCount", "Year", "Duration", "RatingValue"]
    mask_weak_only_lowprob = (
        final_label.eq(1) &
        weak_only &
        (~col.isin(safe_numeric_cols)) &
        (detector_prob < 0.92)
    )
    out.loc[mask_weak_only_lowprob, "final_pred_label"] = 0

    # ============================================================
    # B. 再恢复当前评估中高可信 FN 组合
    #    原则：
    #      1) 只恢复 precision 候选池内已验证质量较高的列-规则组合；
    #      2) 不恢复 Language/Country；
    #      3) 不大范围恢复 rare/FD/context；
    #      4) RatingCount rare_value 继续保留。
    # ============================================================

    # B1. RatingCount：当前 TP 贡献最大，不能再误杀。
    # 保留 detector=1；如果接近阈值且是 count/rare_value 相关，也恢复。
    mask_rating_count_keep = (
        col.eq("RatingCount") &
        (
            detector_label.eq(1) |
            (
                (detector_prob >= 0.45) &
                rule.isin([
                    "rare_value",
                    "rating_count_format",
                    "not_null",
                    "pattern",
                    "paired_missing_consistency",
                ])
            )
        )
    )
    out.loc[mask_rating_count_keep, "final_pred_label"] = 1

    # B2. ReviewCount：恢复 not_null / review_count_format / pattern，
    # 但不恢复 review_rating_count_order。
    mask_review_count_recover = (
        col.eq("ReviewCount") &
        (detector_prob >= 0.45) &
        (
            detector_label.eq(1) |
            rule.isin(["review_count_format", "not_null", "pattern", "paired_missing_consistency"]) |
            (movie_typed_cnt >= 1)
        ) &
        (~rule.eq("review_rating_count_order"))
    )
    out.loc[mask_review_count_recover, "final_pred_label"] = 1

    # B3. Year：恢复明确年份 / pattern / not_null，不恢复 FD/context/rare。
    mask_year_recover = (
        col.eq("Year") &
        (detector_prob >= 0.45) &
        (
            rule.isin(["movie_year_range", "pattern", "not_null"]) |
            (movie_typed_cnt >= 1)
        )
    )
    out.loc[mask_year_recover, "final_pred_label"] = 1

    # B4. Duration：恢复 duration_format / pattern / not_null / rare_pattern。
    # rare_value 已在上面压掉。
    mask_duration_recover = (
        col.eq("Duration") &
        (detector_prob >= 0.45) &
        (
            rule.isin(["duration_format", "pattern", "not_null", "rare_pattern"]) |
            (movie_typed_cnt >= 1)
        ) &
        (~rule.eq("rare_value"))
    )
    out.loc[mask_duration_recover, "final_pred_label"] = 1

    # B5. RatingValue：恢复 range / pattern / not_null，不恢复 FD。
    mask_rating_value_recover = (
        col.eq("RatingValue") &
        (detector_prob >= 0.45) &
        (
            rule.isin(["rating_value_range", "pattern", "not_null"]) |
            (movie_typed_cnt >= 1)
        ) &
        (~rule.isin(["functional_dependency", "soft_functional_dependency"]))
    )
    out.loc[mask_rating_value_recover, "final_pred_label"] = 1

    # B6. Filming Locations + not_null：当前 FN 大头，且在 precision 候选中质量高。
    # 这里恢复不要求高 detector_prob，因为这些候选通常是缺失类强信号。
    mask_filming_not_null_recover = (
        col.eq("Filming Locations") &
        rule.eq("not_null")
    )
    out.loc[mask_filming_not_null_recover, "final_pred_label"] = 1

    # B7. Release Date + not_null：只恢复缺失类，不恢复 release_year_consistency / rare / FD。
    mask_release_not_null_recover = (
        col.eq("Release Date") &
        rule.eq("not_null")
    )
    out.loc[mask_release_not_null_recover, "final_pred_label"] = 1

    # B8. Cast / Genre / Name 的 not_null：从当前 FN 明细看是小规模高价值恢复。
    mask_cast_not_null_recover = (
        col.eq("Cast") &
        rule.eq("not_null")
    )
    out.loc[mask_cast_not_null_recover, "final_pred_label"] = 1

    mask_genre_not_null_recover = (
        col.eq("Genre") &
        rule.eq("not_null")
    )
    out.loc[mask_genre_not_null_recover, "final_pred_label"] = 1

    mask_name_not_null_recover = (
        col.eq("Name") &
        rule.eq("not_null")
    )
    out.loc[mask_name_not_null_recover, "final_pred_label"] = 1

    # B9. Creator / Actors：只在 not_null 或明确 list/mojibake/paired_missing 规则下恢复。
    # 不恢复 FD / rare / context。
    mask_creator_actors_safe_recover = (
        col.isin(["Creator", "Actors"]) &
        rule.isin([
            "not_null",
            "mojibake_or_control_char",
            "people_list_format",
            "comma_list_format",
            "paired_missing_consistency",
        ])
    )
    out.loc[mask_creator_actors_safe_recover, "final_pred_label"] = 1

    # B10. 显式格式/范围类 movie 规则兜底。
    # 注意：排除前面确认高 FP 的 release_year_consistency、review_rating_count_order、actors_cast_overlap。
    explicit_safe_movie_rules = [
        "movie_year_range",
        "release_date_format",
        "duration_format",
        "rating_value_range",
        "rating_count_format",
        "review_count_format",
        "mojibake_or_control_char",
        "rating_value_count_coexistence",
        "paired_missing_consistency",
    ]
    mask_explicit_safe_recover = (
        rule.isin(explicit_safe_movie_rules) &
        (detector_prob >= 0.45)
    )
    out.loc[mask_explicit_safe_recover, "final_pred_label"] = 1

    out["final_pred_label_name"] = out["final_pred_label"].map({1: "error", 0: "correct"})
    return out



def mark_hard_cases(df: pd.DataFrame, detector_thresholds: pd.Series) -> pd.DataFrame:
    out = df.copy()

    raw_prob = safe_num_series(out, "raw_pred_error_prob", 0.0)
    detector_prob = safe_num_series(out, "detector_pred_error_prob", 0.0)
    verifier_prob = safe_num_series(out, "verifier_keep_error_prob", np.nan)
    raw_label = safe_num_series(out, "raw_pred_label", 0).astype(int)
    detector_label = safe_num_series(out, "detector_pred_label", 0).astype(int)
    final_label = safe_num_series(out, "final_pred_label", 0).astype(int)

    group_th = safe_num_series(out, "group_verifier_threshold", 0.50)
    margin = np.abs(detector_prob - detector_thresholds)

    col = safe_str_series(out, "column", "missing")
    rule = safe_str_series(out, "main_rule_type", "missing")
    nb = safe_str_series(out, "neighbor_bucket", "missing")
    movie_typed_cnt = safe_num_series(out, "movie_typed_rule_count", 0.0)
    movie_cons_cnt = safe_num_series(out, "movie_consistency_rule_count", 0.0)

    out["is_mid_prob"] = ((detector_prob >= HARD_CASE_LOW) & (detector_prob <= HARD_CASE_HIGH)).astype(int)
    out["is_near_threshold"] = (margin <= HARD_CASE_MARGIN).astype(int)
    out["is_stage_disagree"] = ((raw_label != detector_label) | (detector_label != final_label)).astype(int)

    out["rule_neighbor_conflict"] = (
        rule.isin(["functional_dependency", "soft_functional_dependency", "dominant_value_by_context"]) &
        nb.isin(["strong_support_current", "weak_support_current"])
    ).astype(int)

    out["verifier_borderline"] = (
        pd.notna(verifier_prob) &
        (np.abs(verifier_prob - group_th) <= 0.05)
    ).astype(int)

    out["verifier_reject_borderline"] = (
        detector_label.eq(1) &
        final_label.eq(0) &
        pd.notna(verifier_prob) &
        (np.abs(verifier_prob - group_th) <= 0.08)
    ).astype(int)

    # FP 回流：高 FP 风险列 + 弱规则 + 被判 error。
    out["fp_conflict_fd_hardcase"] = (
        final_label.eq(1) &
        col.isin(["Release Date", "Language", "Country", "Cast", "Actors", "Name", "Description", "Genre"]) &
        rule.isin([
            "rare_value", "rare_pattern", "functional_dependency", "soft_functional_dependency",
            "dominant_value_by_context", "dominant_value_by_context_pair"
        ]) &
        nb.isin(["weak_support_other", "strong_support_current", "weak_support_current", "no_neighbor_signal"])
    ).astype(int)

    # FN 回流：precision 候选池中高价值列被挡住的样本。
    out["recall_recovery_hardcase"] = (
        final_label.eq(0) &
        col.isin(["RatingCount", "ReviewCount", "Year", "Duration", "RatingValue", "Filming Locations", "Creator", "Actors"]) &
        (
            detector_prob.between(detector_thresholds - 0.15, detector_thresholds + 1e-9) |
            (raw_prob >= 0.72) |
            (
                rule.isin([
                    "review_count_format", "rating_count_format", "movie_year_range", "duration_format",
                    "rating_value_range", "pattern", "not_null", "mojibake_or_control_char",
                    "people_list_format", "comma_list_format", "actors_cast_overlap"
                ]) &
                (detector_prob >= 0.42)
            ) |
            (
                (movie_typed_cnt + movie_cons_cnt >= 1) &
                (detector_prob >= 0.42)
            )
        )
    ).astype(int)

    out["persistent_fp_high_risk"] = (
        final_label.eq(1) &
        col.isin(["Release Date", "Language", "Country", "Cast", "Actors", "Name", "Description", "Genre"]) &
        rule.isin(["rare_value", "rare_pattern", "functional_dependency", "soft_functional_dependency", "dominant_value_by_context"]) &
        (detector_prob >= 0.45)
    ).astype(int)

    out["is_hard_case"] = (
        (out["is_mid_prob"] == 1) |
        (out["is_near_threshold"] == 1) |
        (out["is_stage_disagree"] == 1) |
        (out["verifier_borderline"] == 1) |
        (out["verifier_reject_borderline"] == 1) |
        (out["fp_conflict_fd_hardcase"] == 1) |
        (out["rule_neighbor_conflict"] == 1) |
        (out["persistent_fp_high_risk"] == 1) |
        (out["recall_recovery_hardcase"] == 1)
    ).astype(int)

    out["is_high_risk"] = (
        (detector_prob >= HIGH_RISK_THRESHOLD) &
        (out["final_pred_label"] == 1)
    ).astype(int)

    return out


def build_llm_jsonl(hard_df: pd.DataFrame, output_jsonl: str):
    ensure_dir(os.path.dirname(output_jsonl))
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for _, row in hard_df.iterrows():
            rec = row.to_dict()
            rec["hard_case_flags"] = {
                "verifier_borderline": int(rec.get("verifier_borderline", 0)),
                "verifier_reject_borderline": int(rec.get("verifier_reject_borderline", 0)),
                "fp_conflict_fd_hardcase": int(rec.get("fp_conflict_fd_hardcase", 0)),
                "rule_neighbor_conflict": int(rec.get("rule_neighbor_conflict", 0)),
                "sample_borderline_hardcase": int(rec.get("sample_borderline_hardcase", 0)),
                "persistent_fp_high_risk": int(rec.get("persistent_fp_high_risk", 0)),
                "recall_recovery_hardcase": int(rec.get("recall_recovery_hardcase", 0)),
                "is_stage_disagree": int(rec.get("is_stage_disagree", 0)),
                "is_near_threshold": int(rec.get("is_near_threshold", 0)),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    ensure_dir(OUTPUT_DIR)

    cfg = load_json(CONFIG_PATH)
    metrics = load_json(METRICS_PATH)

    df = pd.read_csv(INPUT_CSV)

    if ABORT_ON_OVERSIZED_CANDIDATE_SET and len(df) > MAX_EXPECTED_CANDIDATES_FOR_PRECISION_MODE:
        raise RuntimeError(
            f"当前推理候选数={len(df)}，超过 precision 模式上限 "
            f"{MAX_EXPECTED_CANDIDATES_FOR_PRECISION_MODE}。"
            "这通常说明你误用了 hybrid/高召回候选池。"
            "请先重新运行 build_rule_pool_movies_precision.py 和 "
            "shrink_search_space_movies_precision.py，再重新生成 context/infer_ready。"
        )

    df = add_derived_features(df)

    stage1_cfg = cfg["stage1"]
    stage1_preprocessor = joblib.load(STAGE1_PREPROCESSOR)
    stage1_model = load_detector_model(STAGE1_BEST_CKPT, stage1_cfg)

    X1 = build_feature_matrix(df, stage1_cfg["feature_cols"], stage1_preprocessor)
    raw_pred_error_prob = predict_probs(stage1_model, X1)
    raw_threshold = float(metrics["stage1_best_threshold"])
    raw_pred_label = (raw_pred_error_prob >= raw_threshold).astype(int)

    out_df = df.copy()
    out_df["raw_pred_error_prob"] = raw_pred_error_prob
    out_df["raw_pred_label"] = raw_pred_label
    out_df["raw_pred_label_name"] = pd.Series(raw_pred_label).map({1: "error", 0: "correct"})
    out_df["raw_margin_to_threshold"] = raw_pred_error_prob - raw_threshold
    out_df["stage1_high_conf_error"] = (raw_pred_error_prob >= 0.80).astype(int)
    out_df["stage1_mid_zone"] = ((raw_pred_error_prob >= 0.35) & (raw_pred_error_prob <= 0.75)).astype(int)

    detector_threshold = raw_threshold
    detector_pred_error_prob = raw_pred_error_prob

    if USE_STAGE2:
        stage2_cfg = cfg["stage2"]
        stage2_preprocessor = joblib.load(STAGE2_PREPROCESSOR)
        stage2_model = load_detector_model(STAGE2_BEST_CKPT, stage2_cfg)

        X2 = build_feature_matrix(out_df, stage2_cfg["feature_cols"], stage2_preprocessor)
        detector_pred_error_prob = predict_probs(stage2_model, X2)
        detector_threshold = float(metrics["stage2_best_threshold"])

    detector_pred_error_prob_before_adjust = detector_pred_error_prob.copy()
    detector_pred_error_prob = apply_neighbor_adjustment(out_df, "raw_pred_error_prob") if not USE_STAGE2 else detector_pred_error_prob

    base_thresholds = apply_base_column_thresholds(out_df, detector_threshold)
    detector_thresholds = apply_conditional_detector_thresholds(out_df, base_thresholds)

    detector_pred_label = (detector_pred_error_prob >= detector_thresholds.values).astype(int)

    out_df["detector_pred_error_prob_before_adjust"] = detector_pred_error_prob_before_adjust
    out_df["detector_pred_error_prob"] = detector_pred_error_prob
    out_df["detector_pred_label"] = detector_pred_label
    out_df["detector_pred_label_name"] = pd.Series(detector_pred_label).map({1: "error", 0: "correct"})
    out_df["detector_margin_to_threshold"] = detector_pred_error_prob - detector_thresholds.values
    out_df["detector_threshold_used"] = detector_thresholds.values

    out_df["is_stage_disagree"] = (
        out_df["raw_pred_label"].astype(int) != out_df["detector_pred_label"].astype(int)
    ).astype(int)

    verifier_model, verifier_preprocessor, verifier_config, verifier_threshold, threshold_source = load_verifier_assets()

    out_df = apply_fp_verifier(
        out_df,
        verifier_model=verifier_model,
        verifier_preprocessor=verifier_preprocessor,
        verifier_config=verifier_config,
        verifier_threshold=verifier_threshold,
    )

    out_df["final_pred_error_prob_before_adjust"] = out_df["detector_pred_error_prob"]
    out_df["final_pred_error_prob"] = out_df["detector_pred_error_prob"]

    out_df = apply_column_post_rules(out_df)
    out_df = mark_hard_cases(out_df, detector_thresholds)

    ensure_dir(os.path.dirname(OUTPUT_ALL_CSV))
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
    print(f"[INFO] detector 预测 error 数: {int(out_df['detector_pred_label'].sum())}")
    print(f"[INFO] verifier 后最终 error 数: {int(out_df['final_pred_label'].sum())}")
    print(f"[INFO] hard case 数: {len(hard_df)}")
    print(f"[INFO] high risk 数: {len(high_risk_df)}")

    if USE_FP_VERIFIER:
        verifier_used = int(pd.notna(out_df["verifier_keep_error_prob"]).sum())
        verifier_rejected = int((out_df["verifier_decision"] == "reject_error").sum())
        print(f"[INFO] verifier 使用阈值: {verifier_threshold:.4f}")
        print(f"[INFO] verifier 阈值来源: {threshold_source}")
        print(f"[INFO] verifier 参与样本数: {verifier_used}")
        print(f"[INFO] verifier 打回 correct 数: {verifier_rejected}")


if __name__ == "__main__":
    main()
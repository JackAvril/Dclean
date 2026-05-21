import json
import os
from typing import Dict, Any, List

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# 1. 用户配置区
# ============================================================

INPUT_CSV = "candidate_infer_ready_adult.csv"
MODEL_DIR = "two_stage_model_output_adult_v4_with_verifier"
OUTPUT_DIR = "two_stage_inference_output_adult_v4_with_verifier"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

USE_STAGE2 = True
USE_COLUMN_THRESHOLDS = True
USE_NEIGHBOR_ADJUSTMENT = True

USE_FP_VERIFIER = True
VERIFIER_ONLY_ON_DETECTOR_ERROR = True

# manual / metrics / max_with_min
VERIFIER_THRESHOLD_MODE = "metrics"
MANUAL_VERIFIER_THRESHOLD = 0.50
MIN_VERIFIER_THRESHOLD = 0.50
DEFAULT_VERIFIER_THRESHOLD = 0.50
PRINT_VERIFIER_DEBUG = True

# adult 基础列阈值：长尾字段更谨慎，强规则字段适当放松
COLUMN_THRESHOLDS = {
    "country": 0.68,
    "occupation": 0.66,
    "workclass": 0.64,
    "education": 0.62,
    "maritalstatus": 0.60,
    "relationship": 0.58,
    "race": 0.62,
    "sex": 0.56,
    "age": 0.58,
    "hoursperweek": 0.58,
    "income": 0.56,
}

# 条件性降低 detector 阈值：只对强 domain / format / adult consistency 信号放松
COND_LOWER_THRESHOLD = {
    "age": 0.46,
    "hoursperweek": 0.46,
    "income": 0.44,
    "sex": 0.45,
    "relationship": 0.46,
    "maritalstatus": 0.48,
    "education": 0.50,
    "workclass": 0.52,
    "occupation": 0.54,
    "country": 0.56,
    "race": 0.52,
}

GROUP_VERIFIER_THRESHOLD_BY_COLUMN = {
    "country": 0.66,
    "occupation": 0.64,
    "workclass": 0.62,
    "education": 0.60,
    "maritalstatus": 0.58,
    "relationship": 0.56,
    "race": 0.60,
    "sex": 0.54,
    "age": 0.54,
    "hoursperweek": 0.54,
    "income": 0.52,
}

GROUP_VERIFIER_THRESHOLD_BY_RULE = {
    "rare_value": 0.68,
    "rare_pattern": 0.68,
    "global_dominant_value": 0.70,
    "functional_dependency": 0.58,
    "soft_functional_dependency": 0.60,

    "age_bucket_format": 0.50,
    "hours_per_week_format_range": 0.50,
    "income_canonical_format": 0.50,

    "workclass_domain": 0.48,
    "education_domain": 0.48,
    "maritalstatus_domain": 0.48,
    "occupation_domain": 0.50,
    "relationship_domain": 0.48,
    "race_domain": 0.50,
    "sex_domain": 0.46,
    "country_domain": 0.52,
    "income_domain": 0.46,

    "age_education_consistency": 0.50,
    "age_relationship_consistency": 0.50,
    "marital_relationship_consistency": 0.48,
    "sex_relationship_consistency": 0.46,
    "workclass_occupation_consistency": 0.52,
    "education_occupation_consistency": 0.54,
    "hours_workclass_consistency": 0.52,
}

HARD_CASE_MARGIN = 0.05
HARD_CASE_LOW = 0.42
HARD_CASE_HIGH = 0.58
HIGH_RISK_THRESHOLD = 0.85

NEIGHBOR_SUPPORT_CURRENT_RATIO = 0.95
NEIGHBOR_SUPPORT_CURRENT_SUBTRACT = 0.05

FP_RISK_COLUMNS = {
    "country",
    "occupation",
    "workclass",
    "education",
    "maritalstatus",
    "relationship",
}

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


# ============================================================
# 2. 模型结构
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
# 3. 工具函数
# ============================================================

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
    out["neighbor_majority_ratio"] = ratio
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

    col_series = safe_str_series(out, "column", "missing")
    long_tail_cols = {"country", "occupation", "workclass"}
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

    out["is_fp_risk_col"] = col_series.isin(FP_RISK_COLUMNS).astype(int)

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

    support_current_mask = (
        (safe_num_series(out, "is_equal_to_neighbor_majority", 0).astype(int) == 1) &
        (safe_num_series(out, "neighbor_majority_ratio", 0.0) >= 0.80) &
        (safe_num_series(out, "strong_rule_count", 0.0) <= 0) &
        (safe_num_series(out, "adult_domain_rule_count", 0.0) <= 0) &
        (safe_num_series(out, "adult_format_rule_count", 0.0) <= 0) &
        (safe_num_series(out, "adult_consistency_rule_count", 0.0) <= 0)
    )
    raw_prob_placeholder = safe_num_series(out, "raw_pred_error_prob", 0.0)
    out["high_prob_but_neighbor_support_current"] = ((raw_prob_placeholder >= 0.80) & support_current_mask).astype(int)

    for col in [
        "column", "semantic_type", "detected_type", "main_rule_type", "main_usage_role",
        "pattern_bucket", "rarity_bucket", "neighbor_bucket", "domain_bucket",
        "adult_consistency_bucket", "age_sampling_bucket", "hours_sampling_bucket",
        "adult_value_bucket", "candidate_rule_dominant",
        "time_window_bucket", "time_value_bucket",
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


# ============================================================
# 4. 阈值逻辑
# ============================================================

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
    domain_invalid = safe_str_series(df, "domain_bucket", "missing").eq("domain_invalid")
    adult_cons_bucket = safe_str_series(df, "adult_consistency_bucket", "missing")

    domain_signal = (
        (safe_num_series(df, "adult_domain_rule_count", 0.0) > 0) |
        domain_invalid |
        rule.str.contains("_domain", regex=False)
    )
    format_signal = (
        (safe_num_series(df, "adult_format_rule_count", 0.0) > 0) |
        rule.isin(["age_bucket_format", "hours_per_week_format_range", "income_canonical_format"])
    )
    consistency_signal = (
        (safe_num_series(df, "adult_consistency_rule_count", 0.0) > 0) |
        adult_cons_bucket.isin(["strong_adult_consistency_signal", "adult_consistency_flag"]) |
        (safe_num_series(df, "adult_consistency_score", 0.0) >= 1.0)
    )

    strong_neighbor_other = nb.isin(["strong_support_other", "medium_support_other"])
    strong_rule = safe_num_series(df, "strong_rule_count", 0.0) > 0

    for col_name, lower_t in COND_LOWER_THRESHOLD.items():
        mask = (
            col.eq(col_name) &
            (
                domain_signal |
                format_signal |
                consistency_signal |
                (strong_rule & strong_neighbor_other)
            )
        )
        thresholds.loc[mask] = np.minimum(thresholds.loc[mask], float(lower_t))

    # rare-only 长尾字段不放松
    rare_only = safe_num_series(df, "rare_only_signal", 0).astype(int).eq(1)
    long_tail_rare = col.isin({"country", "occupation", "workclass"}) & rare_only
    thresholds.loc[long_tail_rare] = np.maximum(thresholds.loc[long_tail_rare], 0.68)

    # neighbor 明确支持当前值且没有强 adult 信号时，提高阈值，减少 FP
    support_current = nb.isin(["strong_support_current", "weak_support_current"])
    no_adult_signal = ~(domain_signal | format_signal | consistency_signal)
    thresholds.loc[support_current & no_adult_signal] = np.maximum(
        thresholds.loc[support_current & no_adult_signal], 0.70
    )

    return thresholds


def apply_neighbor_adjustment(df: pd.DataFrame, prob_col: str) -> pd.Series:
    prob = safe_num_series(df, prob_col, 0.0).copy()
    if not USE_NEIGHBOR_ADJUSTMENT:
        return prob

    support_current_mask = (
        (safe_num_series(df, "is_equal_to_neighbor_majority", 0).astype(int) == 1) &
        (safe_num_series(df, "neighbor_majority_ratio", 0.0) >= NEIGHBOR_SUPPORT_CURRENT_RATIO) &
        (safe_num_series(df, "strong_rule_count", 0.0) <= 0) &
        (safe_num_series(df, "adult_domain_rule_count", 0.0) <= 0) &
        (safe_num_series(df, "adult_format_rule_count", 0.0) <= 0) &
        (safe_num_series(df, "adult_consistency_rule_count", 0.0) <= 0)
    )
    prob.loc[support_current_mask] = np.maximum(
        0.0,
        prob.loc[support_current_mask] - NEIGHBOR_SUPPORT_CURRENT_SUBTRACT
    )
    return prob


# ============================================================
# 5. verifier
# ============================================================

def load_verifier_assets():
    if not USE_FP_VERIFIER:
        return None, None, None, DEFAULT_VERIFIER_THRESHOLD, "disabled"

    if not (os.path.exists(VERIFIER_MODEL_PATH) and os.path.exists(VERIFIER_PREPROCESSOR_PATH) and os.path.exists(VERIFIER_CONFIG_PATH)):
        print("[WARN] verifier artifacts not found, disable verifier.")
        return None, None, None, DEFAULT_VERIFIER_THRESHOLD, "missing"

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
    domain_bucket = str(row.get("domain_bucket", "missing"))
    adult_bucket = str(row.get("adult_consistency_bucket", "missing"))

    th = default_threshold

    if col in GROUP_VERIFIER_THRESHOLD_BY_COLUMN:
        th = max(th, GROUP_VERIFIER_THRESHOLD_BY_COLUMN[col])

    if rule in GROUP_VERIFIER_THRESHOLD_BY_RULE:
        th = max(th, GROUP_VERIFIER_THRESHOLD_BY_RULE[rule])

    # 强 adult 信号降低 verifier 阈值，避免错杀真错误
    strong_adult_signal = (
        domain_bucket == "domain_invalid" or
        adult_bucket in {"strong_adult_consistency_signal", "adult_consistency_flag"} or
        float(row.get("adult_domain_rule_count", 0) or 0) > 0 or
        float(row.get("adult_format_rule_count", 0) or 0) > 0 or
        float(row.get("adult_consistency_rule_count", 0) or 0) > 0
    )
    if strong_adult_signal:
        th = min(th, 0.52)

    # rare-only 的 country/occupation/workclass 更谨慎
    if col in {"country", "occupation", "workclass"} and rule in {"rare_value", "rare_pattern", "global_dominant_value"}:
        th = max(th, 0.72)

    # neighbor 支持当前值，而且无强 adult 信号，提高阈值
    if nb in {"strong_support_current", "weak_support_current"} and not strong_adult_signal:
        th = max(th, 0.68)

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
    risk_col_mask = col_series.isin(FP_RISK_COLUMNS)
    neighbor_mask = neighbor_series.isin({"strong_support_current", "weak_support_current", "weak_support_other"})
    adult_mask = (
        (safe_num_series(out, "adult_domain_rule_count", 0.0) > 0) |
        (safe_num_series(out, "adult_format_rule_count", 0.0) > 0) |
        (safe_num_series(out, "adult_consistency_rule_count", 0.0) > 0) |
        safe_str_series(out, "domain_bucket", "").eq("domain_invalid")
    )

    verifier_mask = detector_positive & (fd_mask | risk_col_mask | neighbor_mask | adult_mask)

    if verifier_mask.sum() == 0:
        out["final_pred_label"] = out["detector_pred_label"].astype(int)
        out["final_pred_label_name"] = out["final_pred_label"].map({1: "error", 0: "correct"})
        return out

    verifier_df = out.loc[verifier_mask].copy()
    Xv = build_verifier_feature_matrix(verifier_df, feature_cols, verifier_preprocessor)
    keep_prob = predict_verifier_keep_prob(verifier_model, Xv)

    if PRINT_VERIFIER_DEBUG:
        print_verifier_debug_stats(keep_prob, verifier_threshold)

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


# ============================================================
# 6. adult 后处理
# ============================================================

def apply_column_post_rules(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    verifier_prob = safe_num_series(out, "verifier_keep_error_prob", np.nan)
    col = safe_str_series(out, "column", "missing")
    rule = safe_str_series(out, "main_rule_type", "missing")
    nb = safe_str_series(out, "neighbor_bucket", "missing")
    domain_bucket = safe_str_series(out, "domain_bucket", "missing")
    adult_bucket = safe_str_series(out, "adult_consistency_bucket", "missing")
    final_label = safe_num_series(out, "final_pred_label", 0).astype(int)
    detector_label = safe_num_series(out, "detector_pred_label", 0).astype(int)

    adult_signal = (
        (safe_num_series(out, "adult_domain_rule_count", 0.0) > 0) |
        (safe_num_series(out, "adult_format_rule_count", 0.0) > 0) |
        (safe_num_series(out, "adult_consistency_rule_count", 0.0) > 0) |
        domain_bucket.eq("domain_invalid") |
        adult_bucket.isin({"strong_adult_consistency_signal", "adult_consistency_flag"}) |
        (safe_num_series(out, "adult_consistency_score", 0.0) >= 1.0)
    )

    # 1) 强 adult 信号被 verifier 打回，但 detector 分数较高时恢复
    recover_strong_adult = (
        detector_label.eq(1) &
        final_label.eq(0) &
        adult_signal &
        (safe_num_series(out, "detector_pred_error_prob", 0.0) >= 0.45)
    )
    out.loc[recover_strong_adult, "final_pred_label"] = 1

    # 2) rare-only 长尾字段且 neighbor 支持当前值，强制打回
    rare_only = safe_num_series(out, "rare_only_signal", 0).astype(int).eq(1)
    long_tail_fp = (
        final_label.eq(1) &
        col.isin({"country", "occupation", "workclass"}) &
        rare_only &
        nb.isin({"strong_support_current", "weak_support_current", "no_neighbor_signal"}) &
        (~adult_signal)
    )
    out.loc[long_tail_fp, "final_pred_label"] = 0

    # 3) global dominant 不应单独作为 adult 长尾列错误依据
    global_only_long_tail = (
        final_label.eq(1) &
        col.isin({"country", "occupation", "workclass", "education"}) &
        rule.eq("global_dominant_value") &
        (~adult_signal) &
        (safe_num_series(out, "fd_like_count", 0.0) <= 0) &
        (safe_num_series(out, "context_rule_count", 0.0) <= 0)
    )
    out.loc[global_only_long_tail, "final_pred_label"] = 0

    # 4) domain_invalid / format invalid / sex-relationship 强冲突优先保留
    strong_keep = (
        detector_label.eq(1) &
        (
            domain_bucket.eq("domain_invalid") |
            (safe_num_series(out, "adult_format_rule_count", 0.0) > 0) |
            adult_bucket.eq("strong_adult_consistency_signal")
        )
    )
    out.loc[strong_keep, "final_pred_label"] = 1

    out["final_pred_label_name"] = out["final_pred_label"].map({1: "error", 0: "correct"})
    return out


# ============================================================
# 7. hard case 标记
# ============================================================

def mark_hard_cases(df: pd.DataFrame, detector_thresholds: pd.Series) -> pd.DataFrame:
    out = df.copy()

    detector_prob = safe_num_series(out, "detector_pred_error_prob", 0.0)
    raw_prob = safe_num_series(out, "raw_pred_error_prob", 0.0)
    verifier_prob = safe_num_series(out, "verifier_keep_error_prob", np.nan)
    raw_label = safe_num_series(out, "raw_pred_label", 0).astype(int)
    detector_label = safe_num_series(out, "detector_pred_label", 0).astype(int)
    final_label = safe_num_series(out, "final_pred_label", 0).astype(int)

    group_th = safe_num_series(out, "group_verifier_threshold", 0.50)
    margin = np.abs(detector_prob - detector_thresholds)

    col = safe_str_series(out, "column", "missing")
    rule = safe_str_series(out, "main_rule_type", "missing")
    nb = safe_str_series(out, "neighbor_bucket", "missing")
    domain_bucket = safe_str_series(out, "domain_bucket", "missing")
    adult_bucket = safe_str_series(out, "adult_consistency_bucket", "missing")

    adult_signal = (
        (safe_num_series(out, "adult_domain_rule_count", 0.0) > 0) |
        (safe_num_series(out, "adult_format_rule_count", 0.0) > 0) |
        (safe_num_series(out, "adult_consistency_rule_count", 0.0) > 0) |
        domain_bucket.eq("domain_invalid") |
        adult_bucket.isin({"strong_adult_consistency_signal", "adult_consistency_flag"}) |
        (safe_num_series(out, "adult_consistency_score", 0.0) >= 1.0)
    )

    out["is_mid_prob"] = ((detector_prob >= HARD_CASE_LOW) & (detector_prob <= HARD_CASE_HIGH)).astype(int)
    out["is_near_threshold"] = (margin <= HARD_CASE_MARGIN).astype(int)
    out["is_stage_disagree"] = ((raw_label != detector_label) | (detector_label != final_label)).astype(int)

    out["rule_neighbor_conflict"] = (
        rule.isin(["functional_dependency", "soft_functional_dependency"]) &
        nb.isin(["strong_support_current", "weak_support_current"])
    ).astype(int)

    out["adult_signal_hardcase"] = (
        adult_signal &
        detector_prob.between(0.30, 0.80)
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

    out["fp_conflict_fd_hardcase"] = (
        final_label.eq(1) &
        col.isin(["country", "occupation", "workclass", "education"]) &
        rule.isin(["functional_dependency", "soft_functional_dependency", "rare_value", "rare_pattern"]) &
        nb.isin(["strong_support_current", "weak_support_current", "no_neighbor_signal"]) &
        (~adult_signal)
    ).astype(int)

    out["sample_borderline_hardcase"] = (
        col.isin(["age", "hoursperweek", "income", "sex", "relationship", "maritalstatus"]) &
        detector_prob.between(0.30, 0.85)
    ).astype(int)

    out["recall_recovery_hardcase"] = (
        final_label.eq(0) &
        adult_signal &
        (
            detector_prob.between(detector_thresholds - 0.18, detector_thresholds + 1e-9) |
            raw_prob.ge(0.72)
        )
    ).astype(int)

    out["persistent_fp_high_risk"] = (
        final_label.eq(1) &
        col.isin(["country", "occupation", "workclass", "education"]) &
        nb.isin(["weak_support_other", "strong_support_current", "no_neighbor_signal"]) &
        rule.isin(["rare_value", "rare_pattern", "global_dominant_value", "functional_dependency", "soft_functional_dependency"]) &
        (~adult_signal)
    ).astype(int)

    out["is_hard_case"] = (
        (out["is_mid_prob"] == 1) |
        (out["is_near_threshold"] == 1) |
        (out["is_stage_disagree"] == 1) |
        (out["verifier_borderline"] == 1) |
        (out["verifier_reject_borderline"] == 1) |
        (out["fp_conflict_fd_hardcase"] == 1) |
        (out["rule_neighbor_conflict"] == 1) |
        (out["sample_borderline_hardcase"] == 1) |
        (out["persistent_fp_high_risk"] == 1) |
        (out["recall_recovery_hardcase"] == 1) |
        (out["adult_signal_hardcase"] == 1)
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
                "adult_signal_hardcase": int(rec.get("adult_signal_hardcase", 0)),
                "is_stage_disagree": int(rec.get("is_stage_disagree", 0)),
                "is_near_threshold": int(rec.get("is_near_threshold", 0)),
            }
            # pandas/numpy value safe
            clean = {}
            for k, v in rec.items():
                if isinstance(v, (np.integer,)):
                    clean[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    clean[k] = float(v)
                elif pd.isna(v) if not isinstance(v, (dict, list)) else False:
                    clean[k] = None
                else:
                    clean[k] = v
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")


# ============================================================
# 8. 主流程
# ============================================================

def main():
    ensure_dir(OUTPUT_DIR)

    cfg = load_json(CONFIG_PATH)
    metrics = load_json(METRICS_PATH)

    df = pd.read_csv(INPUT_CSV)
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

    out_df = add_derived_features(out_df)

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
    if USE_NEIGHBOR_ADJUSTMENT:
        tmp = out_df.copy()
        tmp["detector_pred_error_prob"] = detector_pred_error_prob
        detector_pred_error_prob = apply_neighbor_adjustment(tmp, "detector_pred_error_prob")

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

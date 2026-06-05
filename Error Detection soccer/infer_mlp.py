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

INPUT_CSV = "candidate_infer_ready_soccer.csv"
MODEL_DIR = "two_stage_model_output_soccer_v4_with_verifier"
OUTPUT_DIR = "two_stage_inference_output_soccer_v4_with_verifier"

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

# Soccer 基础列阈值：年份/position 强规则适当放松，长尾实体字段更谨慎。
COLUMN_THRESHOLDS = {
    "birthyear": 0.48,
    "season": 0.50,
    "position": 0.48,

    "team": 0.62,
    "city": 0.62,
    "stadium": 0.62,
    "manager": 0.64,
    "birthplace": 0.66,

    # context-only 身份列默认非常保守，除非 format 直接非法
    "name": 0.64,
    "surname": 0.64,
}

# 条件性降低 detector 阈值：只对 domain / format / numeric / age-consistency 这种强证据放松
COND_LOWER_THRESHOLD = {
    "birthyear": 0.38,
    "season": 0.40,
    "position": 0.38,

    "team": 0.52,
    "city": 0.52,
    "stadium": 0.52,
    "manager": 0.54,
    "birthplace": 0.56,

    "name": 0.54,
    "surname": 0.54,
}

GROUP_VERIFIER_THRESHOLD_BY_COLUMN = {
    "birthyear": 0.44,
    "season": 0.46,
    "position": 0.44,

    "team": 0.58,
    "city": 0.58,
    "stadium": 0.58,
    "manager": 0.60,
    "birthplace": 0.62,

    "name": 0.60,
    "surname": 0.60,
}

GROUP_VERIFIER_THRESHOLD_BY_RULE = {
    "rare_value": 0.72,
    "rare_pattern": 0.72,
    "global_dominant_value": 0.74,
    "functional_dependency": 0.62,
    "soft_functional_dependency": 0.76,

    "birthyear_numeric_range": 0.42,
    "season_numeric_range": 0.44,
    "position_domain": 0.42,
    "birthyear_season_age_consistency": 0.44,

    "name_pattern": 0.56,
    "surname_pattern": 0.56,
    "manager_name_pattern": 0.56,
    "team_name_pattern": 0.58,
    "birthplace_location_pattern": 0.58,
    "city_location_pattern": 0.58,
    "stadium_name_pattern": 0.58,

    "team_context_dominant": 0.62,
    "team_season_manager_consistency_hint": 0.62,
    "city_stadium_consistency_hint": 0.64,
    "dominant_value_by_context": 0.66,
    "dominant_value_by_context_pair": 0.66,
}

HARD_CASE_MARGIN = 0.05
HARD_CASE_LOW = 0.42
HARD_CASE_HIGH = 0.58
HIGH_RISK_THRESHOLD = 0.85

NEIGHBOR_SUPPORT_CURRENT_RATIO = 0.95
NEIGHBOR_SUPPORT_CURRENT_SUBTRACT = 0.05

FP_RISK_COLUMNS = {
    "name", "surname", "team", "city", "stadium", "manager", "birthplace"
}

# ============================================================
# Soccer v3 precision post-gate
# ============================================================
# 当前 soccer v2 的候选范围已经明显收窄，但推理阶段几乎把所有候选都输出为 error。
# 这个 gate 不读取 clean ground truth，只根据候选自身证据强弱做后处理：
# 1) 对 birthyear / season / position 等强 schema/range/domain 证据保召回；
# 2) 对 team / stadium / city / manager 等长尾实体列，弱证据不直接输出 error；
# 3) team 的 not_null/high_risk_missing 是当前最大 FP 来源，默认不单独作为最终错误。
USE_SOCCER_PRECISION_POST_GATE = True

MISSING_RULE_TYPES = {"not_null", "high_risk_missing"}

ENTITY_COLUMNS = {"team", "city", "stadium", "manager", "birthplace"}
LONG_TAIL_TEXT_COLUMNS = {"name", "surname", "team", "city", "stadium", "manager", "birthplace"}

# v4 纠正：从评估明细看，team 的 not_null 候选主要是真错误；
# 真正制造 6 万 FP 的是 team_name_pattern。不要再打回 team missing。
TEAM_MISSING_ONLY_REJECT = False

# team_name_pattern 目前过宽/过敏，会把大量合法 team 实体打成候选。
# 在没有更可靠 team 关系证据前，推理阶段默认不让 team_name_pattern 直接输出 error。
TEAM_PATTERN_RULE_REJECT = True

# v5: 根据当前评估明细，以下规则最终预测精度极低：
# - stadium_name_pattern: FP=2630, TP≈0
# - team_name_pattern: FP=980, TP≈0
# - birthyear_season_age_consistency: FP=1454, TP≈1
# - surname_pattern: FP=524, TP≈23，低精度但仍有少量 TP
# 因此先在推理阶段进行低可靠规则门控，不改候选集，最大程度保持召回上限。
LOW_PRECISION_RULE_REJECT = True
REJECT_RULE_TYPES_STRICT = {
    "stadium_name_pattern",
    "team_name_pattern",
    "birthyear_season_age_consistency",
}
REJECT_RULE_TYPES_SOFT = {
    "surname_pattern",
}
SURNAME_PATTERN_KEEP_MIN_PROB = 0.995
SURNAME_PATTERN_KEEP_MIN_POSTERIOR = 0.95

# 其他列 missing-only 使用更高阈值，不再沿用普通 detector 阈值。
STADIUM_MISSING_ONLY_MIN_PROB = 0.995
BIRTHYEAR_MISSING_ONLY_MIN_PROB = 0.990
SURNAME_MISSING_ONLY_MIN_PROB = 0.900
ENTITY_WEAK_ONLY_MIN_PROB = 0.950
TEXT_MISSING_ONLY_MIN_PROB = 0.900

# 如果后续你发现 recall 掉太多，可以把对应列阈值稍微降一些；
# 但不要全局降低，否则 team FP 会再次冲爆。

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
# Evidence reliability calibration from LLM-labeled candidates
# ============================================================
# 该模块只读取 LLM 标注样本，不读取 clean ground truth / fp_details / evaluation_summary。
USE_EVIDENCE_RELIABILITY_GATE = True
LLM_LABELED_CSV = "candidate_sampled_labeled_soccer.csv"

EVIDENCE_RELIABILITY_JSON = os.path.join(OUTPUT_DIR, "evidence_reliability_soccer.json")
EVIDENCE_RELIABILITY_CSV = os.path.join(OUTPUT_DIR, "evidence_reliability_soccer.csv")

RELIABILITY_MIN_GROUP_SIZE = 5
RELIABILITY_SMOOTH_ALPHA = 1.0
RELIABILITY_SMOOTH_BETA = 1.0

# 温和阈值校准：低可靠证据提高阈值，高可靠证据轻微降低阈值。
RELIABILITY_LOW_PRECISION = 0.30
RELIABILITY_MID_PRECISION = 0.55
RELIABILITY_HIGH_PRECISION = 0.80

RELIABILITY_DELTA_LOW = 0.20
RELIABILITY_DELTA_MID = 0.08
RELIABILITY_DELTA_HIGH = -0.06
RELIABILITY_DELTA_UNKNOWN_WEAK = 0.12

WEAK_ONLY_MIN_THRESHOLD = 0.68
SOFT_FD_ONLY_MIN_THRESHOLD = 0.70
CONTEXT_DOMINANT_ONLY_MIN_THRESHOLD = 0.72
CONTEXT_ONLY_WEAK_MIN_THRESHOLD = 0.88

WEAK_RULE_TYPES_FOR_GATE = {
    "rare_value",
    "rare_pattern",
    "global_dominant_value",
    "functional_dependency",
    "soft_functional_dependency",
    "dominant_value_by_context",
    "dominant_value_by_context_pair",
    "team_context_dominant",
    "team_season_manager_consistency_hint",
    "city_stadium_consistency_hint",
}

SOCCER_COLUMNS = {"name", "surname", "birthyear", "birthplace", "position", "team", "city", "stadium", "season", "manager"}

# Soccer v2：所有列都作为候选目标列。name/surname 不再是 context-only。
PRIMARY_TARGET_COLUMNS = set(SOCCER_COLUMNS)
CONTEXT_ONLY_COLUMNS = set()

SOCCER_STRONG_RULE_TYPES = {
    "birthyear_numeric_range",
    "season_numeric_range",
    "position_domain",
    "birthyear_season_age_consistency",
}
SOCCER_FORMAT_RULE_TYPES = {
    "name_pattern",
    "surname_pattern",
    "manager_name_pattern",
    "team_name_pattern",
    "birthplace_location_pattern",
    "city_location_pattern",
    "stadium_name_pattern",
    "pattern",
    "generic_regex",
}
SOCCER_CONTEXT_RULE_TYPES = {
    "team_context_dominant",
    "team_season_manager_consistency_hint",
    "city_stadium_consistency_hint",
    "dominant_value_by_context",
    "dominant_value_by_context_pair",
}
# Soccer v2：soft-FD 只作为 evidence-only 弱证据，不再作为强 FD-like 信号。
SOCCER_FD_RULE_TYPES = {"functional_dependency"}
SOCCER_EVIDENCE_ONLY_FD_RULE_TYPES = {"soft_functional_dependency"}


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


def as_int01(x, default=0):
    try:
        if pd.isna(x):
            return default
    except Exception:
        pass
    if x is None:
        return default
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y"}:
        return 1
    if s in {"0", "false", "no", "n"}:
        return 0
    try:
        return int(float(x))
    except Exception:
        return default


def add_soccer_features(out: pd.DataFrame) -> pd.DataFrame:
    col_series = safe_str_series(out, "column", "missing")
    rule_series = safe_str_series(out, "main_rule_type", "missing")

    for c in [
        "target_is_primary_column", "target_is_context_only_column",
        "birthyear_value_invalid", "season_value_invalid", "position_value_invalid",
        "position_needs_canonicalization", "birthyear_season_age_conflict",
        "team_context_conflict", "format_rule_signal", "fd_like_signal",
    ]:
        out[c] = safe_num_series(out, c, 0).astype(int)

    out["target_is_primary_column"] = np.where(
        col_series.isin(PRIMARY_TARGET_COLUMNS), 1, out["target_is_primary_column"]
    ).astype(int)
    out["target_is_context_only_column"] = np.where(
        col_series.isin(CONTEXT_ONLY_COLUMNS), 1, 0
    ).astype(int)

    out["birthyear_value_invalid"] = np.where(
        rule_series.eq("birthyear_numeric_range"), 1, out["birthyear_value_invalid"]
    ).astype(int)
    out["season_value_invalid"] = np.where(
        rule_series.eq("season_numeric_range"), 1, out["season_value_invalid"]
    ).astype(int)
    out["position_value_invalid"] = np.where(
        rule_series.eq("position_domain"), 1, out["position_value_invalid"]
    ).astype(int)
    out["birthyear_season_age_conflict"] = np.where(
        rule_series.eq("birthyear_season_age_consistency"), 1, out["birthyear_season_age_conflict"]
    ).astype(int)
    out["team_context_conflict"] = np.where(
        rule_series.isin(SOCCER_CONTEXT_RULE_TYPES), 1, out["team_context_conflict"]
    ).astype(int)
    out["format_rule_signal"] = np.where(
        rule_series.isin(SOCCER_FORMAT_RULE_TYPES), 1, out["format_rule_signal"]
    ).astype(int)
    out["fd_like_signal"] = np.where(
        rule_series.isin(SOCCER_FD_RULE_TYPES), 1, out["fd_like_signal"]
    ).astype(int)

    if "evidence_only_fd_count" not in out.columns:
        out["evidence_only_fd_count"] = 0
    out["evidence_only_fd_count"] = safe_num_series(out, "evidence_only_fd_count", 0).astype(float)
    soft_fd_mask = rule_series.isin(SOCCER_EVIDENCE_ONLY_FD_RULE_TYPES)
    out.loc[soft_fd_mask, "fd_like_signal"] = 0
    out.loc[soft_fd_mask, "evidence_only_fd_count"] = np.maximum(
        out.loc[soft_fd_mask, "evidence_only_fd_count"].astype(float), 1.0
    )

    out["position_invalid_or_normalization"] = (
        (out["position_value_invalid"].astype(int) > 0)
        | (out["position_needs_canonicalization"].astype(int) > 0)
    ).astype(int)

    out["has_birthyear_or_season_signal"] = (
        (out["birthyear_value_invalid"].astype(int) > 0)
        | (out["season_value_invalid"].astype(int) > 0)
        | (out["birthyear_season_age_conflict"].astype(int) > 0)
        | rule_series.isin({
            "birthyear_numeric_range",
            "season_numeric_range",
            "birthyear_season_age_consistency",
        })
    ).astype(int)

    out["has_team_context_signal"] = (
        (out["team_context_conflict"].astype(int) > 0)
        | rule_series.isin(SOCCER_CONTEXT_RULE_TYPES)
    ).astype(int)

    out["has_fd_like_signal"] = (
        (out["fd_like_signal"].astype(int) > 0)
        | (safe_num_series(out, "fd_like_count", 0.0) > 0)
        | rule_series.isin(SOCCER_FD_RULE_TYPES)
    ).astype(int)

    out["is_context_only_weak_signal"] = (
        (out["target_is_context_only_column"].astype(int) == 1)
        & (safe_num_series(out, "soccer_domain_rule_count", 0.0) <= 0)
        & (safe_num_series(out, "soccer_format_rule_count", 0.0) <= 0)
        & (safe_num_series(out, "soccer_numeric_rule_count", 0.0) <= 0)
        & (safe_num_series(out, "soccer_consistency_rule_count", 0.0) <= 0)
        & (~rule_series.isin(SOCCER_STRONG_RULE_TYPES | SOCCER_FORMAT_RULE_TYPES))
    ).astype(int)

    out["soccer_signal_strength"] = (
        2.5 * out["has_birthyear_or_season_signal"].astype(float)
        + 2.2 * out["position_invalid_or_normalization"].astype(float)
        + 1.5 * safe_num_series(out, "has_soccer_format_signal", 0.0)
        + 1.2 * safe_num_series(out, "has_soccer_consistency_signal", 0.0)
        + 0.8 * out["target_is_primary_column"].astype(float)
        + 0.5 * out["has_team_context_signal"].astype(float)
        + 0.4 * out["has_fd_like_signal"].astype(float)
        - 1.2 * out["is_context_only_weak_signal"].astype(float)
    )

    out["soccer_primary_target_signal"] = (
        (out["target_is_primary_column"].astype(int) == 1)
        & (
            (out["has_birthyear_or_season_signal"].astype(int) == 1)
            | (out["position_invalid_or_normalization"].astype(int) == 1)
            | (safe_num_series(out, "has_soccer_format_signal", 0).astype(int) == 1)
            | (safe_num_series(out, "has_soccer_domain_signal", 0).astype(int) == 1)
            | (safe_num_series(out, "has_soccer_numeric_signal", 0).astype(int) == 1)
            | (safe_num_series(out, "has_soccer_consistency_signal", 0).astype(int) == 1)
        )
    ).astype(int)

    if "soccer_attribution_bucket" not in out.columns:
        out["soccer_attribution_bucket"] = "unknown_attribution"
    out["soccer_attribution_bucket"] = out["soccer_attribution_bucket"].fillna("unknown_attribution").astype(str)
    return out


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in [
        "strong_rule_count", "candidate_generation_rule_count", "fd_like_count", "evidence_only_fd_count", "context_rule_count",
        "global_rule_count", "rare_value_count", "typo_rule_count", "pattern_rule_count", "schema_rule_count",
        "soccer_domain_rule_count", "soccer_format_rule_count", "soccer_numeric_rule_count",
        "soccer_consistency_rule_count", "soccer_profile_inconsistency_count",
        "adult_domain_rule_count", "adult_format_rule_count", "adult_consistency_rule_count",
        "time_window_rule_count",
    ]:
        out[col] = safe_num_series(out, col, 0.0)

    for col in [
        "conflict_score", "value_frequency", "value_frequency_rank", "neighbor_majority_ratio",
        "prior_error_probability", "posterior_error_probability", "soccer_consistency_score",
        "year_value", "row_birthyear_int", "row_season_int", "row_age_at_season",
        "birthyear_season_gap", "time_value_minutes",
    ]:
        out[col] = safe_num_series(out, col, 0.0)

    if "value" in out.columns and "neighbor_majority_value" in out.columns:
        value_series = normalize_text_series_for_compare(out["value"])
        neighbor_series = normalize_text_series_for_compare(out["neighbor_majority_value"])
        out["is_equal_to_neighbor_majority"] = (value_series == neighbor_series).astype(int)
    else:
        out["is_equal_to_neighbor_majority"] = safe_num_series(out, "is_equal_to_neighbor_majority", 0).astype(int)

    ratio = safe_num_series(out, "neighbor_majority_ratio", 0.0)
    out["neighbor_agreement_score"] = np.where(out["is_equal_to_neighbor_majority"] == 1, ratio, -ratio).astype(float)
    out["has_neighbor_signal"] = (ratio > 0).astype(int)
    out["high_neighbor_consistency"] = (ratio >= 0.8).astype(int)

    out["rule_total_count"] = out["strong_rule_count"] + out["candidate_generation_rule_count"]
    out["candidate_rule_ratio"] = np.where(out["rule_total_count"] > 0, out["candidate_generation_rule_count"] / (out["rule_total_count"] + 1e-8), 0.0)
    out["strong_rule_ratio"] = np.where(out["rule_total_count"] > 0, out["strong_rule_count"] / (out["rule_total_count"] + 1e-8), 0.0)
    out["context_rule_ratio"] = np.where(out["rule_total_count"] > 0, out["context_rule_count"] / (out["rule_total_count"] + 1e-8), 0.0)
    out["fd_like_ratio"] = np.where(out["rule_total_count"] > 0, out["fd_like_count"] / (out["rule_total_count"] + 1e-8), 0.0)
    out["global_rule_ratio"] = np.where(out["rule_total_count"] > 0, out["global_rule_count"] / (out["rule_total_count"] + 1e-8), 0.0)

    out["soccer_rule_total_count"] = out["soccer_domain_rule_count"] + out["soccer_format_rule_count"] + out["soccer_numeric_rule_count"] + out["soccer_consistency_rule_count"]
    out["soccer_domain_rule_ratio"] = np.where(out["soccer_rule_total_count"] > 0, out["soccer_domain_rule_count"] / (out["soccer_rule_total_count"] + 1e-8), 0.0)
    out["soccer_format_rule_ratio"] = np.where(out["soccer_rule_total_count"] > 0, out["soccer_format_rule_count"] / (out["soccer_rule_total_count"] + 1e-8), 0.0)
    out["soccer_numeric_rule_ratio"] = np.where(out["soccer_rule_total_count"] > 0, out["soccer_numeric_rule_count"] / (out["soccer_rule_total_count"] + 1e-8), 0.0)
    out["soccer_consistency_rule_ratio"] = np.where(out["soccer_rule_total_count"] > 0, out["soccer_consistency_rule_count"] / (out["soccer_rule_total_count"] + 1e-8), 0.0)

    out["adult_rule_total_count"] = out["adult_domain_rule_count"] + out["adult_format_rule_count"] + out["adult_consistency_rule_count"]
    out["adult_domain_rule_ratio"] = 0.0
    out["adult_format_rule_ratio"] = 0.0
    out["adult_consistency_rule_ratio"] = 0.0

    long_tail_cols = {"name", "surname", "team", "city", "stadium", "manager", "birthplace"}
    col_series = safe_str_series(out, "column", "missing")
    value_rank = safe_num_series(out, "value_frequency_rank", 999999)
    value_freq = safe_num_series(out, "value_frequency", 999999)
    out["is_rare_value"] = np.where(
        col_series.isin(long_tail_cols),
        ((value_freq <= 1) | (value_rank > 25)).astype(int),
        ((value_freq <= 5) | (value_rank > 10)).astype(int),
    )

    out["neighbor_disagree"] = ((out["has_neighbor_signal"] == 1) & (out["is_equal_to_neighbor_majority"] == 0)).astype(int)

    out["rule_strength_sum"] = (
        1.00 * out["strong_rule_count"]
        + 0.25 * out["candidate_generation_rule_count"]
        + 0.25 * out["fd_like_count"]
        + 0.35 * out["context_rule_count"]
        + 0.15 * out["global_rule_count"]
        + 0.20 * out["rare_value_count"]
        + 0.35 * out["typo_rule_count"]
        + 0.45 * out["pattern_rule_count"]
        + 0.45 * out["schema_rule_count"]
        + 0.95 * out["soccer_domain_rule_count"]
        + 0.90 * out["soccer_format_rule_count"]
        + 0.95 * out["soccer_numeric_rule_count"]
        + 0.80 * out["soccer_consistency_rule_count"]
    )

    out["candidate_rule_dominant"] = np.where(out["candidate_generation_rule_count"] > out["strong_rule_count"], "yes", "no")

    out["domain_invalid_flag"] = safe_str_series(out, "domain_bucket", "missing").eq("domain_invalid").astype(int)
    out["soccer_consistency_flag"] = safe_str_series(out, "soccer_consistency_bucket", "missing").isin([
        "strong_soccer_consistency_signal", "soccer_consistency_flag", "weak_soccer_consistency_signal"
    ]).astype(int)
    out["high_soccer_consistency_score"] = (out["soccer_consistency_score"] >= 1.0).astype(int)
    out["has_soccer_domain_signal"] = (out["soccer_domain_rule_count"] > 0).astype(int)
    out["has_soccer_format_signal"] = (out["soccer_format_rule_count"] > 0).astype(int)
    out["has_soccer_numeric_signal"] = (out["soccer_numeric_rule_count"] > 0).astype(int)
    out["has_soccer_consistency_signal"] = (out["soccer_consistency_rule_count"] > 0).astype(int)

    out["birthyear_season_gap"] = np.maximum(0.0, safe_num_series(out, "row_season_int", 0.0) - safe_num_series(out, "row_birthyear_int", 0.0))
    out["is_young_player_age"] = ((safe_num_series(out, "row_age_at_season", 0.0) > 0) & (safe_num_series(out, "row_age_at_season", 0.0) < 18)).astype(int)
    out["is_old_player_age"] = (safe_num_series(out, "row_age_at_season", 0.0) > 40).astype(int)

    out = add_soccer_features(out)

    out["rare_only_signal"] = (
        (out["rare_value_count"] > 0)
        & (out["strong_rule_count"] <= 0)
        & (out["fd_like_count"] <= 0)
        & (out["context_rule_count"] <= 0)
        & (out["typo_rule_count"] <= 0)
        & (out["schema_rule_count"] <= 0)
        & (out["soccer_domain_rule_count"] <= 0)
        & (out["soccer_format_rule_count"] <= 0)
        & (out["soccer_numeric_rule_count"] <= 0)
        & (out["soccer_consistency_rule_count"] <= 0)
    ).astype(int)

    out["is_weak_only"] = (
        (out["rare_only_signal"] == 1)
        | (
            (out["strong_rule_count"] <= 0)
            & (out["fd_like_count"] <= 0)
            & (out["typo_rule_count"] <= 0)
            & (out["schema_rule_count"] <= 0)
            & (out["soccer_domain_rule_count"] <= 0)
            & (out["soccer_format_rule_count"] <= 0)
            & (out["soccer_numeric_rule_count"] <= 0)
            & (out["soccer_consistency_rule_count"] <= 0)
        )
    ).astype(int)

    out["is_fp_risk_col"] = safe_str_series(out, "column", "missing").isin(FP_RISK_COLUMNS).astype(int)
    out["high_prob_but_neighbor_support_current"] = (
        (out["high_neighbor_consistency"] == 1) & (out["is_equal_to_neighbor_majority"] == 1)
    ).astype(int)

    # 兼容旧 adult 字段，避免老 preprocessor 或旧下游读取时报错
    for c, v in {
        "adult_consistency_flag": 0,
        "high_adult_consistency_score": 0,
        "has_adult_domain_signal": 0,
        "has_adult_format_signal": 0,
        "has_adult_consistency_signal": 0,
        "age_span": 0.0,
        "hours_span": 0.0,
        "is_minor_age_bucket": 0,
        "is_high_hours_bucket": 0,
        "relationship_v2_conflict_count": 0,
        "has_relationship_v2_signal": 0,
        "has_education_v2_signal": 0,
        "adult_v2_signal_strength": 0.0,
        "adult_v2_primary_target_signal": 0,
    }.items():
        if c not in out.columns:
            out[c] = v

    return out


def load_model_from_ckpt(ckpt_path: str) -> MLPDetector:
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model = MLPDetector(
        input_dim=ckpt["input_dim"],
        hidden_dim_1=ckpt["hidden_dim_1"],
        hidden_dim_2=ckpt["hidden_dim_2"],
        dropout=ckpt["dropout"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def predict_mlp(model, X_np: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    probs = []
    with torch.no_grad():
        for start in range(0, len(X_np), batch_size):
            x = torch.tensor(X_np[start:start + batch_size], dtype=torch.float32, device=DEVICE)
            logits = model(x)
            p = torch.sigmoid(logits).detach().cpu().numpy()
            probs.append(p)
    return np.concatenate(probs) if probs else np.array([])


def get_stage_threshold(config: Dict[str, Any], stage: str, default=0.5) -> float:
    try:
        return float(config.get(f"{stage}_metrics", {}).get("threshold", default))
    except Exception:
        return default


def build_stage2_df(df: pd.DataFrame, p1: np.ndarray, stage1_th: float) -> pd.DataFrame:
    out = df.copy()
    out["raw_pred_error_prob"] = p1
    out["raw_pred_label"] = (p1 >= stage1_th).astype(int)
    out["raw_margin_to_threshold"] = p1 - stage1_th
    out["stage1_high_conf_error"] = (p1 >= max(stage1_th + 0.20, 0.75)).astype(int)
    out["stage1_mid_zone"] = ((p1 >= 0.35) & (p1 <= 0.75)).astype(int)
    return out


def build_verifier_df(stage2_df: pd.DataFrame, detector_probs: np.ndarray, detector_thresholds: pd.Series) -> pd.DataFrame:
    out = stage2_df.copy()
    out["detector_pred_error_prob"] = detector_probs
    out["detector_pred_label"] = (detector_probs >= detector_thresholds.values).astype(int)
    out["detector_margin_to_threshold"] = detector_probs - detector_thresholds.values
    out["is_stage_disagree"] = (out["raw_pred_label"].astype(int) != out["detector_pred_label"].astype(int)).astype(int)
    return out


def get_base_thresholds(df: pd.DataFrame, default_threshold: float) -> pd.Series:
    col = safe_str_series(df, "column", "missing")
    th = pd.Series(np.full(len(df), default_threshold), index=df.index, dtype=float)

    if USE_COLUMN_THRESHOLDS:
        for c, v in COLUMN_THRESHOLDS.items():
            th.loc[col == c] = float(v)

    strong_signal = (
        (safe_num_series(df, "has_birthyear_or_season_signal", 0).astype(int) == 1)
        | (safe_num_series(df, "position_invalid_or_normalization", 0).astype(int) == 1)
        | (safe_num_series(df, "has_soccer_domain_signal", 0).astype(int) == 1)
        | (safe_num_series(df, "has_soccer_format_signal", 0).astype(int) == 1)
        | (safe_num_series(df, "has_soccer_numeric_signal", 0).astype(int) == 1)
        | (safe_num_series(df, "has_soccer_consistency_signal", 0).astype(int) == 1)
    )

    for c, v in COND_LOWER_THRESHOLD.items():
        mask = (col == c) & strong_signal
        th.loc[mask] = np.minimum(th.loc[mask], float(v))

    if USE_NEIGHBOR_ADJUSTMENT:
        support_current = (
            (safe_num_series(df, "neighbor_majority_ratio", 0.0) >= NEIGHBOR_SUPPORT_CURRENT_RATIO)
            & (safe_num_series(df, "is_equal_to_neighbor_majority", 0).astype(int) == 1)
            & (~strong_signal)
        )
        th.loc[support_current] = np.minimum(0.95, th.loc[support_current] + NEIGHBOR_SUPPORT_CURRENT_SUBTRACT)

    return th.clip(lower=0.05, upper=0.95)


def normalize_label_binary_from_row(row: pd.Series):
    if "label_binary" in row.index and pd.notna(row["label_binary"]):
        v = pd.to_numeric(pd.Series([row["label_binary"]]), errors="coerce").iloc[0]
        if pd.notna(v):
            return 1 if int(float(v)) == 1 else 0
    if "label" in row.index and pd.notna(row["label"]):
        s = str(row["label"]).strip().lower()
        if s == "error":
            return 1
        if s == "correct":
            return 0
    if "is_error" in row.index and pd.notna(row["is_error"]):
        s = str(row["is_error"]).strip().lower()
        if s in {"1", "true", "yes", "error"}:
            return 1
        if s in {"0", "false", "no", "correct"}:
            return 0
    return None


def reliability_key(field: str, value: Any) -> str:
    if value is None or pd.isna(value):
        value = "missing"
    return f"{field}={str(value)}"


def reliability_pair_key(field1: str, value1: Any, field2: str, value2: Any) -> str:
    if value1 is None or pd.isna(value1):
        value1 = "missing"
    if value2 is None or pd.isna(value2):
        value2 = "missing"
    return f"{field1}={str(value1)}||{field2}={str(value2)}"


def estimate_reliability_table(labeled_csv: str) -> Dict[str, Dict[str, Any]]:
    if not USE_EVIDENCE_RELIABILITY_GATE:
        return {}

    if not labeled_csv or not os.path.exists(labeled_csv):
        print(f"[Reliability] labeled csv not found: {labeled_csv}; use generic weak-evidence gate only.")
        return {}

    labeled_df = pd.read_csv(labeled_csv, low_memory=False)
    if len(labeled_df) == 0:
        return {}

    labeled_df = add_derived_features(labeled_df)

    label_binary = []
    for _, row in labeled_df.iterrows():
        label_binary.append(normalize_label_binary_from_row(row))
    labeled_df["_rel_label_binary"] = label_binary
    labeled_df = labeled_df[labeled_df["_rel_label_binary"].notna()].copy()
    if len(labeled_df) == 0:
        print("[Reliability] no valid LLM labels; use generic weak-evidence gate only.")
        return {}

    labeled_df["_rel_label_binary"] = labeled_df["_rel_label_binary"].astype(int)
    if "confidence" in labeled_df.columns:
        labeled_df["_rel_confidence"] = pd.to_numeric(labeled_df["confidence"], errors="coerce").fillna(1.0).clip(0.0, 1.0)
    else:
        labeled_df["_rel_confidence"] = 1.0

    rows = []
    group_specs = [
        ("soccer_attribution_bucket", ["soccer_attribution_bucket"]),
        ("main_rule_type", ["main_rule_type"]),
        ("column__main_rule_type", ["column", "main_rule_type"]),
        ("column__soccer_attribution_bucket", ["column", "soccer_attribution_bucket"]),
    ]

    for group_name, keys in group_specs:
        existing_keys = [k for k in keys if k in labeled_df.columns]
        if len(existing_keys) != len(keys):
            continue

        for group_values, g in labeled_df.groupby(existing_keys, dropna=False):
            if not isinstance(group_values, tuple):
                group_values = (group_values,)

            n = int(len(g))
            error_count = int((g["_rel_label_binary"] == 1).sum())
            correct_count = int((g["_rel_label_binary"] == 0).sum())
            estimated_precision = (
                (error_count + RELIABILITY_SMOOTH_ALPHA)
                / (n + RELIABILITY_SMOOTH_ALPHA + RELIABILITY_SMOOTH_BETA)
            )
            avg_conf = float(g["_rel_confidence"].mean())

            if len(keys) == 1:
                key = reliability_key(keys[0], group_values[0])
            else:
                key = reliability_pair_key(keys[0], group_values[0], keys[1], group_values[1])

            if n < RELIABILITY_MIN_GROUP_SIZE:
                action = "insufficient"
                delta = 0.0
            elif estimated_precision < RELIABILITY_LOW_PRECISION:
                action = "conservative"
                delta = RELIABILITY_DELTA_LOW
            elif estimated_precision < RELIABILITY_MID_PRECISION:
                action = "slightly_conservative"
                delta = RELIABILITY_DELTA_MID
            elif estimated_precision >= RELIABILITY_HIGH_PRECISION:
                action = "trust"
                delta = RELIABILITY_DELTA_HIGH
            else:
                action = "neutral"
                delta = 0.0

            rows.append({
                "group_name": group_name,
                "key": key,
                "n": n,
                "error_count": error_count,
                "correct_count": correct_count,
                "estimated_precision": float(estimated_precision),
                "avg_confidence": avg_conf,
                "threshold_delta": float(delta),
                "action": action,
            })

    reliability = {r["key"]: r for r in rows}

    ensure_dir(os.path.dirname(EVIDENCE_RELIABILITY_JSON))
    with open(EVIDENCE_RELIABILITY_JSON, "w", encoding="utf-8") as f:
        json.dump(reliability, f, ensure_ascii=False, indent=2)
    pd.DataFrame(rows).to_csv(EVIDENCE_RELIABILITY_CSV, index=False, encoding="utf-8-sig")

    print(f"[Reliability] valid LLM labels: {len(labeled_df)}")
    print(f"[Reliability] groups: {len(rows)}")
    print(f"[Reliability] saved: {EVIDENCE_RELIABILITY_JSON}")
    print(f"[Reliability] saved: {EVIDENCE_RELIABILITY_CSV}")

    return reliability


def lookup_reliability_delta_for_row(row: pd.Series, reliability: Dict[str, Dict[str, Any]]) -> float:
    if not reliability:
        return 0.0

    col = row.get("column", "missing")
    rule = row.get("main_rule_type", "missing")
    bucket = row.get("soccer_attribution_bucket", "missing")

    keys = [
        reliability_pair_key("column", col, "soccer_attribution_bucket", bucket),
        reliability_pair_key("column", col, "main_rule_type", rule),
        reliability_key("soccer_attribution_bucket", bucket),
        reliability_key("main_rule_type", rule),
    ]

    for k in keys:
        info = reliability.get(k)
        if info is None:
            continue
        if int(info.get("n", 0)) < RELIABILITY_MIN_GROUP_SIZE:
            continue
        return float(info.get("threshold_delta", 0.0))

    return 0.0


def is_direct_strong_soccer_signal_df(df: pd.DataFrame) -> pd.Series:
    rule = safe_str_series(df, "main_rule_type", "missing")
    domain_invalid = safe_str_series(df, "domain_bucket", "missing").eq("domain_invalid")

    return (
        domain_invalid
        | (safe_num_series(df, "soccer_domain_rule_count", 0.0) > 0)
        | (safe_num_series(df, "soccer_format_rule_count", 0.0) > 0)
        | (safe_num_series(df, "soccer_numeric_rule_count", 0.0) > 0)
        | (safe_num_series(df, "birthyear_value_invalid", 0).astype(int) == 1)
        | (safe_num_series(df, "season_value_invalid", 0).astype(int) == 1)
        | (safe_num_series(df, "position_value_invalid", 0).astype(int) == 1)
        | (safe_num_series(df, "position_needs_canonicalization", 0).astype(int) == 1)
        | (safe_num_series(df, "birthyear_season_age_conflict", 0).astype(int) == 1)
        | rule.isin(SOCCER_STRONG_RULE_TYPES | SOCCER_FORMAT_RULE_TYPES)
    )


def apply_evidence_reliability_gate(df: pd.DataFrame, thresholds: pd.Series, reliability: Dict[str, Dict[str, Any]]) -> pd.Series:
    if not USE_EVIDENCE_RELIABILITY_GATE:
        return thresholds

    out_th = thresholds.copy().astype(float)

    rule = safe_str_series(df, "main_rule_type", "missing")
    direct_strong = is_direct_strong_soccer_signal_df(df)

    weak_rule = rule.isin(WEAK_RULE_TYPES_FOR_GATE)
    context_only_weak = safe_num_series(df, "is_context_only_weak_signal", 0).astype(int).eq(1)

    weak_only = weak_rule & (~direct_strong)
    out_th.loc[weak_only] = np.maximum(out_th.loc[weak_only], WEAK_ONLY_MIN_THRESHOLD)

    soft_fd_only = rule.eq("soft_functional_dependency") & (~direct_strong)
    out_th.loc[soft_fd_only] = np.maximum(out_th.loc[soft_fd_only], SOFT_FD_ONLY_MIN_THRESHOLD)

    context_dom_only = rule.isin(SOCCER_CONTEXT_RULE_TYPES | {"dominant_value_by_context", "dominant_value_by_context_pair"}) & (~direct_strong)
    out_th.loc[context_dom_only] = np.maximum(out_th.loc[context_dom_only], CONTEXT_DOMINANT_ONLY_MIN_THRESHOLD)

    out_th.loc[context_only_weak] = np.maximum(out_th.loc[context_only_weak], CONTEXT_ONLY_WEAK_MIN_THRESHOLD)

    deltas = df.apply(lambda row: lookup_reliability_delta_for_row(row, reliability), axis=1)
    deltas = pd.to_numeric(deltas, errors="coerce").fillna(0.0)
    deltas = pd.Series(deltas, index=df.index)
    deltas.loc[direct_strong & (deltas > 0)] = np.minimum(deltas.loc[direct_strong & (deltas > 0)], 0.06)

    out_th = out_th + deltas

    unknown_weak = weak_only & (deltas.abs() < 1e-12)
    out_th.loc[unknown_weak] = out_th.loc[unknown_weak] + RELIABILITY_DELTA_UNKNOWN_WEAK

    return out_th.clip(lower=0.05, upper=0.95)


def get_verifier_group_thresholds(df: pd.DataFrame, default_threshold: float) -> pd.Series:
    th = pd.Series(np.full(len(df), default_threshold), index=df.index, dtype=float)
    col = safe_str_series(df, "column", "missing")
    rule = safe_str_series(df, "main_rule_type", "missing")

    for c, v in GROUP_VERIFIER_THRESHOLD_BY_COLUMN.items():
        th.loc[col == c] = float(v)
    for r, v in GROUP_VERIFIER_THRESHOLD_BY_RULE.items():
        th.loc[rule == r] = np.minimum(th.loc[rule == r], float(v))

    return th.clip(lower=0.05, upper=0.95)


def load_verifier_threshold() -> float:
    if VERIFIER_THRESHOLD_MODE == "manual":
        return MANUAL_VERIFIER_THRESHOLD

    if os.path.exists(VERIFIER_CONFIG_PATH):
        cfg = load_json(VERIFIER_CONFIG_PATH)
        th = float(cfg.get("verifier_threshold", DEFAULT_VERIFIER_THRESHOLD))
    elif os.path.exists(VERIFIER_METRICS_PATH):
        m = load_json(VERIFIER_METRICS_PATH)
        th = float(m.get("threshold", DEFAULT_VERIFIER_THRESHOLD))
    else:
        th = DEFAULT_VERIFIER_THRESHOLD

    if VERIFIER_THRESHOLD_MODE == "max_with_min":
        th = max(th, MIN_VERIFIER_THRESHOLD)

    return th



def get_non_missing_strong_signal_df(df: pd.DataFrame) -> pd.Series:
    """不把 not_null / high_risk_missing 本身算作强证据。

    说明：
    - is_direct_strong_soccer_signal_df() 会把 schema_rule_count 也算强；
      但 not_null 本身也会贡献 schema_rule_count，这会让 missing-only 样本绕过过滤。
    - 这里专门用于 post-gate，只承认 domain / format / numeric / consistency / position / age 这类更可靠证据。
    """
    rule = safe_str_series(df, "main_rule_type", "missing")
    return (
        (safe_num_series(df, "soccer_domain_rule_count", 0.0) > 0)
        | (safe_num_series(df, "soccer_format_rule_count", 0.0) > 0)
        | (safe_num_series(df, "soccer_numeric_rule_count", 0.0) > 0)
        | (safe_num_series(df, "soccer_consistency_rule_count", 0.0) > 0)
        | (safe_num_series(df, "birthyear_value_invalid", 0).astype(int) == 1)
        | (safe_num_series(df, "season_value_invalid", 0).astype(int) == 1)
        | (safe_num_series(df, "position_value_invalid", 0).astype(int) == 1)
        | (safe_num_series(df, "position_needs_canonicalization", 0).astype(int) == 1)
        | (safe_num_series(df, "birthyear_season_age_conflict", 0).astype(int) == 1)
        | rule.isin(SOCCER_STRONG_RULE_TYPES | SOCCER_FORMAT_RULE_TYPES)
    )


def apply_soccer_precision_post_gate(out_df: pd.DataFrame) -> pd.DataFrame:
    """Soccer v3 推理后处理门控。

    只使用 dirty/candidate/LLM-derived features，不使用 clean ground truth。
    目标是过滤 team/stadium/birthyear/surname 的 missing-only 或 weak-only FP，
    同时尽量保留 birthyear/season/position/format/range/domain/age-consistency 强证据。
    """
    if not USE_SOCCER_PRECISION_POST_GATE:
        return out_df

    out = out_df.copy()

    col = safe_str_series(out, "column", "missing")
    rule = safe_str_series(out, "main_rule_type", "missing")
    prob = safe_num_series(out, "detector_pred_error_prob", 0.0)
    posterior = safe_num_series(out, "posterior_error_probability", 0.0)

    non_missing_strong = get_non_missing_strong_signal_df(out)
    missing_rule = rule.isin(MISSING_RULE_TYPES)

    has_format = (
        (safe_num_series(out, "soccer_format_rule_count", 0.0) > 0)
        | (safe_num_series(out, "format_rule_signal", 0).astype(int) == 1)
        | rule.isin(SOCCER_FORMAT_RULE_TYPES)
    )
    has_domain = (
        (safe_num_series(out, "soccer_domain_rule_count", 0.0) > 0)
        | rule.isin({"position_domain"})
    )
    has_numeric = (
        (safe_num_series(out, "soccer_numeric_rule_count", 0.0) > 0)
        | rule.isin({"birthyear_numeric_range", "season_numeric_range"})
    )
    has_consistency = (
        (safe_num_series(out, "soccer_consistency_rule_count", 0.0) > 0)
        | (safe_num_series(out, "birthyear_season_age_conflict", 0).astype(int) == 1)
        | rule.isin({"birthyear_season_age_consistency"})
    )

    has_fd_or_context = (
        (safe_num_series(out, "has_fd_like_signal", 0).astype(int) == 1)
        | (safe_num_series(out, "has_team_context_signal", 0).astype(int) == 1)
        | rule.isin(SOCCER_CONTEXT_RULE_TYPES | SOCCER_FD_RULE_TYPES)
    )

    missing_only = missing_rule & (~non_missing_strong) & (~has_fd_or_context)
    weak_rule = rule.isin(WEAK_RULE_TYPES_FOR_GATE)
    entity_weak_only = col.isin(ENTITY_COLUMNS) & weak_rule & (~non_missing_strong)

    out["soccer_precision_gate_applied"] = 0
    out["soccer_precision_gate_reason"] = ""

    team_missing_only = col.eq("team") & missing_only
    if TEAM_MISSING_ONLY_REJECT:
        mask = team_missing_only & out["final_pred_label"].astype(int).eq(1)
        out.loc[mask, "final_pred_label"] = 0
        out.loc[mask, "soccer_precision_gate_applied"] = 1
        out.loc[mask, "soccer_precision_gate_reason"] += "|team_missing_only_reject"

    # v4: team 的 major FP 来源是 team_name_pattern，而不是 not_null。
    # 因此保留 team not_null，压制 team_name_pattern。
    team_pattern_only = (
        col.eq("team")
        & rule.eq("team_name_pattern")
        & (~has_domain)
        & (~has_numeric)
        & (~has_consistency)
    )
    if TEAM_PATTERN_RULE_REJECT:
        mask = team_pattern_only & out["final_pred_label"].astype(int).eq(1)
        out.loc[mask, "final_pred_label"] = 0
        out.loc[mask, "soccer_precision_gate_applied"] = 1
        out.loc[mask, "soccer_precision_gate_reason"] += "|team_name_pattern_reject"

    stadium_missing_only = col.eq("stadium") & missing_only
    mask = stadium_missing_only & out["final_pred_label"].astype(int).eq(1) & (
        (prob < STADIUM_MISSING_ONLY_MIN_PROB) | (posterior < 0.90)
    )
    out.loc[mask, "final_pred_label"] = 0
    out.loc[mask, "soccer_precision_gate_applied"] = 1
    out.loc[mask, "soccer_precision_gate_reason"] += "|stadium_missing_only_high_threshold"

    birthyear_missing_only = col.eq("birthyear") & missing_only
    mask = birthyear_missing_only & out["final_pred_label"].astype(int).eq(1) & (
        (prob < BIRTHYEAR_MISSING_ONLY_MIN_PROB) | (posterior < 0.85)
    )
    out.loc[mask, "final_pred_label"] = 0
    out.loc[mask, "soccer_precision_gate_applied"] = 1
    out.loc[mask, "soccer_precision_gate_reason"] += "|birthyear_missing_only_high_threshold"

    text_missing_only = col.isin({"name", "surname", "birthplace", "manager"}) & missing_only
    surname_mask = col.eq("surname") & missing_only & out["final_pred_label"].astype(int).eq(1) & (
        (prob < SURNAME_MISSING_ONLY_MIN_PROB) | (posterior < 0.65)
    )
    out.loc[surname_mask, "final_pred_label"] = 0
    out.loc[surname_mask, "soccer_precision_gate_applied"] = 1
    out.loc[surname_mask, "soccer_precision_gate_reason"] += "|surname_missing_only_threshold"

    other_text_mask = text_missing_only & (~col.eq("surname")) & out["final_pred_label"].astype(int).eq(1) & (
        (prob < TEXT_MISSING_ONLY_MIN_PROB) | (posterior < 0.60)
    )
    out.loc[other_text_mask, "final_pred_label"] = 0
    out.loc[other_text_mask, "soccer_precision_gate_applied"] = 1
    out.loc[other_text_mask, "soccer_precision_gate_reason"] += "|text_missing_only_threshold"

    mask = entity_weak_only & out["final_pred_label"].astype(int).eq(1) & (
        (prob < ENTITY_WEAK_ONLY_MIN_PROB) | (posterior < 0.90)
    )
    out.loc[mask, "final_pred_label"] = 0
    out.loc[mask, "soccer_precision_gate_applied"] = 1
    out.loc[mask, "soccer_precision_gate_reason"] += "|entity_weak_only_threshold"

    # v5: 低可靠规则门控。
    # 这些规则在当前 LLM/候选/评估链路中表现为极低 precision，不适合直接输出 error。
    if LOW_PRECISION_RULE_REJECT:
        strict_low_precision_rule = rule.isin(REJECT_RULE_TYPES_STRICT)
        mask = strict_low_precision_rule & out["final_pred_label"].astype(int).eq(1)
        out.loc[mask, "final_pred_label"] = 0
        out.loc[mask, "soccer_precision_gate_applied"] = 1
        out.loc[mask, "soccer_precision_gate_reason"] += "|low_precision_rule_reject"

        # surname_pattern 有少量 TP，因此不完全一刀切，只保留 detector/posterior 都极高的样本。
        soft_low_precision_rule = rule.isin(REJECT_RULE_TYPES_SOFT)
        mask = soft_low_precision_rule & out["final_pred_label"].astype(int).eq(1) & (
            (prob < SURNAME_PATTERN_KEEP_MIN_PROB) | (posterior < SURNAME_PATTERN_KEEP_MIN_POSTERIOR)
        )
        out.loc[mask, "final_pred_label"] = 0
        out.loc[mask, "soccer_precision_gate_applied"] = 1
        out.loc[mask, "soccer_precision_gate_reason"] += "|surname_pattern_soft_reject"

    # 强证据保护：明确 domain/format/numeric/age/position 信号可以覆盖弱门控。
    # 但 team_name_pattern 在当前 soccer 数据上是低可靠规则，不能被 format 强保护自动恢复。
    protected_format = has_format & (~rule.isin(REJECT_RULE_TYPES_STRICT | REJECT_RULE_TYPES_SOFT))
    strong_keep = (
        (protected_format | has_domain | has_numeric | has_consistency)
        & (
            (prob >= 0.35)
            | (posterior >= 0.50)
            | col.isin({"birthyear", "season", "position"})
        )
    )
    out.loc[strong_keep, "final_pred_label"] = np.maximum(
        out.loc[strong_keep, "final_pred_label"].astype(int),
        out.loc[strong_keep, "detector_pred_label"].astype(int),
    )

    if "verifier_rejected" in out.columns:
        post_rejected = (
            (out["soccer_precision_gate_applied"].astype(int) == 1)
            & (out["detector_pred_label"].astype(int) == 1)
            & (out["final_pred_label"].astype(int) == 0)
        )
        out.loc[post_rejected, "verifier_rejected"] = 1

    return out



def mark_hard_cases(out_df: pd.DataFrame, detector_thresholds: pd.Series) -> pd.DataFrame:
    out = out_df.copy()
    prob = safe_num_series(out, "detector_pred_error_prob", 0.0)
    margin = prob - detector_thresholds

    near_threshold = margin.abs() <= HARD_CASE_MARGIN
    mid_zone = (prob >= HARD_CASE_LOW) & (prob <= HARD_CASE_HIGH)
    stage_disagree = safe_num_series(out, "is_stage_disagree", 0).astype(int).eq(1)
    verifier_borderline = pd.Series(False, index=out.index)
    if "verifier_keep_error_prob" in out.columns:
        verifier_borderline = (safe_num_series(out, "verifier_keep_error_prob", 0.0) - safe_num_series(out, "group_verifier_threshold", DEFAULT_VERIFIER_THRESHOLD)).abs() <= HARD_CASE_MARGIN

    rule_neighbor_conflict = (
        (safe_num_series(out, "neighbor_disagree", 0).astype(int) == 1)
        & (safe_num_series(out, "high_neighbor_consistency", 0).astype(int) == 1)
        & (safe_num_series(out, "detector_pred_label", 0).astype(int) == 1)
    )

    soccer_signal_hardcase = (
        (safe_num_series(out, "soccer_primary_target_signal", 0).astype(int) == 1)
        & (prob.between(0.25, 0.85))
    )

    birthyear_season_hardcase = (
        safe_str_series(out, "column", "missing").isin({"birthyear", "season"})
        & (safe_num_series(out, "has_birthyear_or_season_signal", 0).astype(int) == 1)
        & prob.between(0.25, 0.90)
    )
    position_hardcase = (
        safe_str_series(out, "column", "missing").eq("position")
        & (safe_num_series(out, "position_invalid_or_normalization", 0).astype(int) == 1)
        & prob.between(0.25, 0.90)
    )
    context_fp_hardcase = (
        safe_num_series(out, "is_context_only_weak_signal", 0).astype(int).eq(1)
        & safe_num_series(out, "final_pred_label", 0).astype(int).eq(1)
    )

    out["is_hard_case"] = (
        near_threshold
        | mid_zone
        | stage_disagree
        | verifier_borderline
        | rule_neighbor_conflict
        | soccer_signal_hardcase
        | birthyear_season_hardcase
        | position_hardcase
        | context_fp_hardcase
    ).astype(int)

    out["hard_case_reason"] = ""
    out.loc[near_threshold, "hard_case_reason"] += "|near_threshold"
    out.loc[mid_zone, "hard_case_reason"] += "|mid_zone"
    out.loc[stage_disagree, "hard_case_reason"] += "|stage_disagree"
    out.loc[verifier_borderline, "hard_case_reason"] += "|verifier_borderline"
    out.loc[rule_neighbor_conflict, "hard_case_reason"] += "|rule_neighbor_conflict"
    out.loc[soccer_signal_hardcase, "hard_case_reason"] += "|soccer_signal_hardcase"
    out.loc[birthyear_season_hardcase, "hard_case_reason"] += "|birthyear_season_hardcase"
    out.loc[position_hardcase, "hard_case_reason"] += "|position_hardcase"
    out.loc[context_fp_hardcase, "hard_case_reason"] += "|context_fp_hardcase"

    # v3: high_risk 不再因为 primary_target_signal 而几乎全量置 1。
    direct_strong = get_non_missing_strong_signal_df(out)
    out["high_risk"] = (
        (prob >= HIGH_RISK_THRESHOLD)
        | (safe_num_series(out, "has_birthyear_or_season_signal", 0).astype(int) == 1)
        | (safe_num_series(out, "position_invalid_or_normalization", 0).astype(int) == 1)
        | (
            direct_strong
            & safe_num_series(out, "final_pred_label", 0).astype(int).eq(1)
            & (prob >= 0.60)
        )
    ).astype(int)

    return out


def export_hard_cases_jsonl(df: pd.DataFrame, path: str):
    hard_df = df[df["is_hard_case"] == 1].copy()
    records = []
    for _, row in hard_df.iterrows():
        rec = {}
        for col in hard_df.columns:
            v = row[col]
            if pd.isna(v):
                rec[col] = None
            elif isinstance(v, (np.integer,)):
                rec[col] = int(v)
            elif isinstance(v, (np.floating,)):
                rec[col] = float(v)
            elif isinstance(v, (np.bool_,)):
                rec[col] = bool(v)
            else:
                rec[col] = v
        rec["hard_case_flags"] = {
            "soccer_signal_hardcase": int("soccer_signal_hardcase" in str(row.get("hard_case_reason", ""))),
            "birthyear_season_hardcase": int("birthyear_season_hardcase" in str(row.get("hard_case_reason", ""))),
            "position_hardcase": int("position_hardcase" in str(row.get("hard_case_reason", ""))),
            "context_fp_hardcase": int("context_fp_hardcase" in str(row.get("hard_case_reason", ""))),
            "verifier_borderline": int("verifier_borderline" in str(row.get("hard_case_reason", ""))),
            "rule_neighbor_conflict": int("rule_neighbor_conflict" in str(row.get("hard_case_reason", ""))),
            "is_stage_disagree": int(row.get("is_stage_disagree", 0) or 0),
            "is_near_threshold": int("near_threshold" in str(row.get("hard_case_reason", ""))),
        }
        records.append(rec)

    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ============================================================
# 4. 主流程
# ============================================================

def main():
    ensure_dir(OUTPUT_DIR)

    config = load_json(CONFIG_PATH)
    metrics = load_json(METRICS_PATH) if os.path.exists(METRICS_PATH) else {}

    df = pd.read_csv(INPUT_CSV, low_memory=False)
    df = add_derived_features(df)

    stage1_model = load_model_from_ckpt(STAGE1_BEST_CKPT)
    stage1_pre = joblib.load(STAGE1_PREPROCESSOR)

    stage1_th = get_stage_threshold(config, "stage1", default=0.5)
    stage2_th = get_stage_threshold(config, "stage2", default=0.5)

    X1 = stage1_pre.transform(df)
    p1 = predict_mlp(stage1_model, X1)
    stage2_df = build_stage2_df(df, p1, stage1_th)

    if USE_STAGE2 and os.path.exists(STAGE2_BEST_CKPT) and os.path.exists(STAGE2_PREPROCESSOR):
        stage2_model = load_model_from_ckpt(STAGE2_BEST_CKPT)
        stage2_pre = joblib.load(STAGE2_PREPROCESSOR)
        X2 = stage2_pre.transform(stage2_df)
        detector_prob = predict_mlp(stage2_model, X2)
    else:
        detector_prob = p1
        stage2_th = stage1_th

    detector_thresholds = get_base_thresholds(stage2_df, stage2_th)
    reliability = estimate_reliability_table(LLM_LABELED_CSV)
    detector_thresholds = apply_evidence_reliability_gate(stage2_df, detector_thresholds, reliability)

    detector_pred = (detector_prob >= detector_thresholds.values).astype(int)

    out_df = stage2_df.copy()
    out_df["detector_pred_error_prob"] = detector_prob
    out_df["detector_threshold"] = detector_thresholds.values
    out_df["detector_pred_label"] = detector_pred
    out_df["detector_margin_to_threshold"] = detector_prob - detector_thresholds.values
    out_df["is_stage_disagree"] = (out_df["raw_pred_label"].astype(int) != out_df["detector_pred_label"].astype(int)).astype(int)

    out_df["final_pred_label"] = out_df["detector_pred_label"]
    out_df["final_pred_error_prob"] = out_df["detector_pred_error_prob"]
    out_df["verifier_used"] = 0
    out_df["verifier_keep_error_prob"] = np.nan
    out_df["group_verifier_threshold"] = np.nan
    out_df["verifier_rejected"] = 0

    if USE_FP_VERIFIER and os.path.exists(VERIFIER_MODEL_PATH) and os.path.exists(VERIFIER_PREPROCESSOR_PATH):
        verifier_model = joblib.load(VERIFIER_MODEL_PATH)
        verifier_pre = joblib.load(VERIFIER_PREPROCESSOR_PATH)
        base_verifier_th = load_verifier_threshold()

        verifier_input = build_verifier_df(out_df, detector_prob, detector_thresholds)
        Xv = verifier_pre.transform(verifier_input)

        if hasattr(verifier_model, "predict_proba") and len(getattr(verifier_model, "classes_", [])) > 1:
            classes = list(verifier_model.classes_)
            keep_prob = verifier_model.predict_proba(Xv)[:, classes.index(1)]
        else:
            pred = verifier_model.predict(Xv)
            keep_prob = pred.astype(float)

        group_th = get_verifier_group_thresholds(out_df, base_verifier_th)
        if VERIFIER_THRESHOLD_MODE == "manual":
            group_th[:] = MANUAL_VERIFIER_THRESHOLD
        elif VERIFIER_THRESHOLD_MODE == "max_with_min":
            group_th = np.maximum(group_th, MIN_VERIFIER_THRESHOLD)

        use_mask = out_df["detector_pred_label"].astype(int).eq(1) if VERIFIER_ONLY_ON_DETECTOR_ERROR else pd.Series(True, index=out_df.index)
        keep_mask = keep_prob >= group_th.values

        out_df["verifier_used"] = use_mask.astype(int)
        out_df["verifier_keep_error_prob"] = keep_prob
        out_df["group_verifier_threshold"] = group_th.values
        out_df["verifier_rejected"] = (use_mask & (~keep_mask)).astype(int)

        out_df.loc[use_mask & (~keep_mask), "final_pred_label"] = 0

        if PRINT_VERIFIER_DEBUG:
            print(f"[Verifier] base_th={base_verifier_th:.4f}, used={int(use_mask.sum())}, rejected={int((use_mask & (~keep_mask)).sum())}")
    else:
        print("[Verifier] skipped: verifier model/preprocessor not found or USE_FP_VERIFIER=False")

    out_df = apply_soccer_precision_post_gate(out_df)

    if USE_SOCCER_PRECISION_POST_GATE:
        print(f"[SoccerPostGate] applied={int(safe_num_series(out_df, 'soccer_precision_gate_applied', 0).sum())}, "
              f"final_after_gate={int(out_df['final_pred_label'].sum())}")

    out_df = mark_hard_cases(out_df, detector_thresholds)

    out_df.to_csv(OUTPUT_ALL_CSV, index=False, encoding="utf-8-sig")
    out_df[out_df["is_hard_case"] == 1].to_csv(OUTPUT_HARD_CASE_CSV, index=False, encoding="utf-8-sig")
    out_df[out_df["high_risk"] == 1].to_csv(OUTPUT_HIGH_RISK_CSV, index=False, encoding="utf-8-sig")
    export_hard_cases_jsonl(out_df, OUTPUT_LLM_JSONL)

    print("\n===== INFERENCE DONE =====")
    print(f"Input rows: {len(out_df)}")
    print(f"Detector predicted errors: {int(out_df['detector_pred_label'].sum())}")
    print(f"Final predicted errors: {int(out_df['final_pred_label'].sum())}")
    print(f"Verifier used: {int(out_df['verifier_used'].sum())}")
    print(f"Verifier rejected: {int(out_df['verifier_rejected'].sum())}")
    print(f"Hard cases: {int(out_df['is_hard_case'].sum())}")
    print(f"High risk: {int(out_df['high_risk'].sum())}")
    print(f"Output: {OUTPUT_ALL_CSV}")
    print(f"Hard cases: {OUTPUT_HARD_CASE_CSV}")
    print(f"High risk: {OUTPUT_HIGH_RISK_CSV}")
    print(f"LLM JSONL: {OUTPUT_LLM_JSONL}")


if __name__ == "__main__":
    main()

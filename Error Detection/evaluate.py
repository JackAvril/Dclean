#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unified_evaluate_detection.py

统一版错误检测评估脚本，覆盖：
hospital / flights / beers / rayyan / adult / soccer

设计目标：
1. 公共评估逻辑只保留一份：
   - 读取 clean / dirty
   - 逐单元格构造真实错误集合
   - 读取 inference_all_predictions.csv
   - 解析 detector_pred_label 和 final_pred_label
   - 计算 Candidate Coverage / Candidate-Only Precision
   - 计算 Detector-only P/R/F1
   - 计算 Final P/R/F1
   - 导出 TP / FP / FN / candidate covered / candidate missed 明细

2. 数据集差异放入配置：
   - clean_csv / dirty_csv / infer_csv / output_dir
   - 数据集名称
   - 数据集列
   - 专属明细字段
   - adult v2 / soccer v2 诊断字段

3. 默认输出尽量保持原六份 evaluate.py 一致：
   - evaluation_summary.csv
   - tp_details.csv / fp_details.csv / fn_details.csv
   - detector_tp_details.csv / detector_fp_details.csv / detector_fn_details.csv
   - candidate_covered_true_errors.csv
   - candidate_missed_true_errors.csv
   - adult/soccer 额外输出 group summaries 和 v2 diagnostics

运行示例：
python unified_evaluate_detection.py --dataset soccer
python unified_evaluate_detection.py --dataset soccer --infer_csv two_stage_inference_output_soccer_full/inference_all_predictions.csv --output_dir eval_outputs_soccer_full
python unified_evaluate_detection.py --dataset all
"""

import argparse
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd


# ============================================================
# 1. 数据集配置
# ============================================================

MISSING_TOKENS_DEFAULT = {
    "", "empty", "null", "none", "nan", "na", "n/a", "missing", "unk"
}
MISSING_TOKENS_EXTENDED = {
    "", "empty", "null", "none", "nan", "na", "n/a", "missing", "unk",
    "unknown", "?", "not available", "contact airline"
}

HOSPITAL_COLUMNS: List[str] = []
FLIGHTS_COLUMNS: List[str] = []
BEERS_COLUMNS: List[str] = []
RAYYAN_COLUMNS: List[str] = []
ADULT_COLUMNS = [
    "age", "workclass", "education", "maritalstatus", "occupation",
    "relationship", "race", "sex", "hoursperweek", "country", "income",
]
SOCCER_COLUMNS = [
    "name", "surname", "birthyear", "birthplace", "position",
    "team", "city", "stadium", "season", "manager",
]

ADULT_PRIMARY_TARGET_COLUMNS = {"sex", "relationship", "education"}
ADULT_CONTEXT_ONLY_COLUMNS = {
    "age", "workclass", "maritalstatus", "occupation", "race",
    "hoursperweek", "country", "income",
}
ADULT_RELATIONSHIP_V2_RULE_TYPES = {
    "relationship_marital_spouse_conflict",
    "relationship_age_spouse_conflict",
    "relationship_sex_spouse_conflict",
    "relationship_context_dominant",
}
ADULT_EDUCATION_V2_RULE_TYPES = {"education_age_extreme_conflict"}

SOCCER_PRIMARY_TARGET_COLUMNS = set(SOCCER_COLUMNS)
SOCCER_CONTEXT_ONLY_COLUMNS: Set[str] = set()
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
SOCCER_FD_RULE_TYPES = {"functional_dependency"}
SOCCER_EVIDENCE_ONLY_FD_RULE_TYPES = {"soft_functional_dependency"}


BASE_USEFUL_COLS = [
    "value", "label_source", "semantic_type", "detected_type",

    "violation_count", "conflict_score", "value_frequency", "value_frequency_rank",
    "neighbor_majority_value", "neighbor_majority_ratio",
    "prior_error_probability", "posterior_error_probability",
    "main_rule_type", "main_usage_role",

    "strong_rule_count", "candidate_generation_rule_count", "fd_like_count",
    "context_rule_count", "global_rule_count", "rare_value_count", "typo_rule_count",
    "pattern_rule_count", "schema_rule_count",

    "candidate_rule_ratio", "rule_total_count", "strong_rule_ratio",
    "context_rule_ratio", "fd_like_ratio", "global_rule_ratio", "rule_strength_sum",

    "pattern_bucket", "rarity_bucket", "neighbor_bucket",
    "bucket_id", "cluster_id", "dist_to_center", "sample_role", "is_sampled",

    "error_type", "reason_short", "reason_detailed",
    "llm_label", "llm_is_error", "label", "label_binary", "confidence",
    "propagation_confidence", "sample_weight",

    "raw_pred_error_prob", "raw_pred_label", "raw_pred_label_name",
    "detector_pred_error_prob", "detector_pred_label", "detector_pred_label_name",
    "final_pred_error_prob", "final_pred_label", "final_pred_label_name",
    "verifier_keep_error_prob", "verifier_used", "verifier_rejected",
    "verifier_decision", "group_verifier_threshold", "detector_threshold",

    "is_hard_case", "hard_case_reason", "high_risk", "is_high_risk",
]

FLIGHTS_USEFUL_COLS = [
    "time_value_minutes", "time_window_rule_count", "time_window_bucket", "time_value_bucket",
]
BEERS_USEFUL_COLS = [
    "numeric_value", "numeric_window_rule_count", "numeric_window_bucket", "numeric_value_bucket",
]
RAYYAN_USEFUL_COLS = [
    "numeric_value", "date_value", "language_canonical", "issn_canonical",
    "canonical_rule_count", "canonical_bucket", "date_value_bucket", "language_bucket",
]
ADULT_USEFUL_COLS = [
    "adult_domain_rule_count", "adult_format_rule_count", "adult_consistency_rule_count",
    "adult_rule_total_count", "adult_domain_rule_ratio", "adult_format_rule_ratio", "adult_consistency_rule_ratio",
    "age_lower", "age_upper", "hours_lower", "hours_upper", "canonical_income", "education_order",
    "adult_value_bucket", "adult_consistency_score", "adult_profile_inconsistency_count",
    "adult_consistency_bucket", "domain_bucket", "age_sampling_bucket", "hours_sampling_bucket",

    "target_is_primary_column", "target_is_context_only_column",
    "is_relationship_spouse_value", "relationship_marital_conflict",
    "relationship_age_conflict", "relationship_sex_conflict",
    "sex_value_invalid", "education_age_extreme_conflict",
    "has_relationship_v2_rule", "has_education_v2_rule",
    "relationship_v2_conflict_count", "has_relationship_v2_signal",
    "has_education_v2_signal", "is_context_only_weak_signal",
    "adult_v2_signal_strength", "adult_v2_primary_target_signal",
    "adult_v2_attribution_bucket",
]
SOCCER_USEFUL_COLS = [
    "soccer_domain_rule_count", "soccer_format_rule_count",
    "soccer_numeric_rule_count", "soccer_consistency_rule_count",
    "soccer_rule_total_count", "soccer_domain_rule_ratio", "soccer_format_rule_ratio",
    "soccer_numeric_rule_ratio", "soccer_consistency_rule_ratio",
    "soccer_profile_inconsistency_count", "soccer_consistency_score",
    "soccer_consistency_bucket", "soccer_value_bucket",

    "year_value", "row_birthyear_int", "row_season_int", "row_age_at_season",
    "birthyear_season_gap", "is_young_player_age", "is_old_player_age",
    "canonical_position", "row_position_canonical",

    "target_is_primary_column", "target_is_context_only_column",
    "birthyear_value_invalid", "season_value_invalid",
    "position_value_invalid", "position_needs_canonicalization",
    "birthyear_season_age_conflict", "team_context_conflict",
    "format_rule_signal", "fd_like_signal", "evidence_only_fd_signal", "evidence_only_fd_count",
    "position_invalid_or_normalization", "has_birthyear_or_season_signal",
    "has_team_context_signal", "has_fd_like_signal",
    "is_context_only_weak_signal", "soccer_signal_strength",
    "soccer_primary_target_signal", "soccer_attribution_bucket",

    # soccer 推理脚本兼容 adult 字段
    "adult_domain_rule_count", "adult_format_rule_count", "adult_consistency_rule_count",
    "adult_consistency_bucket", "adult_v2_attribution_bucket",
]


@dataclass
class EvalConfig:
    dataset: str
    display_name: str
    dataset_dir: str
    clean_csv: str
    dirty_csv: str
    infer_csv: str
    output_dir: str
    columns: List[str] = field(default_factory=list)
    missing_tokens: Set[str] = field(default_factory=lambda: set(MISSING_TOKENS_DEFAULT))
    drop_unnamed_columns: bool = True
    special: str = "generic"  # generic / adult / soccer
    export_group_summaries: bool = False
    useful_cols: List[str] = field(default_factory=list)


DATASET_CONFIGS: Dict[str, EvalConfig] = {
    "hospital": EvalConfig(
        dataset="hospital",
        display_name="",
        dataset_dir="Error Detection Hospital",
        clean_csv="/mnt/mydata/dq/projects/Splittree/hospital/hospital_clean_no_address23.csv",
        dirty_csv="/mnt/mydata/dq/projects/Splittree/hospital/hospital_dirty_no_address23.csv",
        infer_csv="active_cleaning_loop_runs_v4/iter_3/infer/inference_all_predictions.csv",
        output_dir="eval_outputs_v4_with_verifier",
        columns=HOSPITAL_COLUMNS,
        useful_cols=BASE_USEFUL_COLS,
    ),
    "flights": EvalConfig(
        dataset="flights",
        display_name="",
        dataset_dir="Error Detection flights",
        clean_csv="/mnt/mydata/dq/projects/Splittree/flights/flights_clean.csv",
        dirty_csv="/mnt/mydata/dq/projects/Splittree/flights/flights_dirty.csv",
        infer_csv="/mnt/mydata/dq/projects/Splittree/test/Error Detection flights/active_cleaning_loop_runs_flights_v4/iter_3/infer/inference_all_predictions.csv",
        output_dir="eval_outputs_flights_v4_with_verifier",
        columns=FLIGHTS_COLUMNS,
        useful_cols=BASE_USEFUL_COLS + FLIGHTS_USEFUL_COLS,
    ),
    "beers": EvalConfig(
        dataset="beers",
        display_name="",
        dataset_dir="Error Detection beers",
        clean_csv="/mnt/mydata/dq/projects/Splittree/beers/beers_clean.csv",
        dirty_csv="/mnt/mydata/dq/projects/Splittree/beers/beers_dirty.csv",
        infer_csv="/mnt/mydata/dq/projects/Splittree/test/Error Detection beers/active_cleaning_loop_runs_beers_v4/iter_3/infer/inference_all_predictions.csv",
        output_dir="eval_outputs_beers_v4_with_verifier",
        columns=BEERS_COLUMNS,
        useful_cols=BASE_USEFUL_COLS + BEERS_USEFUL_COLS,
    ),
    "rayyan": EvalConfig(
        dataset="rayyan",
        display_name="",
        dataset_dir="Error Detection rayyan",
        clean_csv="/mnt/mydata/dq/projects/Splittree/rayyan/rayyan_clean.csv",
        dirty_csv="/mnt/mydata/dq/projects/Splittree/rayyan/rayyan_dirty.csv",
        infer_csv="/mnt/mydata/dq/projects/Splittree/test/Error Detection rayyan/active_cleaning_loop_runs_rayyan_v4/iter_3/infer/inference_all_predictions.csv",
        output_dir="eval_outputs_rayyan_v4_with_verifier",
        columns=RAYYAN_COLUMNS,
        useful_cols=BASE_USEFUL_COLS + RAYYAN_USEFUL_COLS,
    ),
    "adult": EvalConfig(
        dataset="adult",
        display_name="Adult",
        dataset_dir="Error Detection Adult",
        clean_csv="/mnt/mydata/dq/projects/Splittree/adult/adult_clean.csv",
        dirty_csv="/mnt/mydata/dq/projects/Splittree/adult/adult_dirty.csv",
        infer_csv="/mnt/mydata/dq/projects/Splittree/test/Error Detection Adult/active_cleaning_loop_runs_adult_v4/iter_3_infer/inference_all_predictions.csv",
        output_dir="eval_outputs_adult_v4_with_verifier",
        columns=ADULT_COLUMNS,
        missing_tokens=MISSING_TOKENS_EXTENDED,
        special="adult",
        export_group_summaries=True,
        useful_cols=BASE_USEFUL_COLS + ADULT_USEFUL_COLS,
    ),
    "soccer": EvalConfig(
        dataset="soccer",
        display_name="Soccer",
        dataset_dir="Error Detection soccer",
        clean_csv="/mnt/mydata/dq/projects/Splittree/soccer/soccer_clean.csv",
        dirty_csv="/mnt/mydata/dq/projects/Splittree/soccer/soccer_dirty.csv",
        infer_csv="/mnt/mydata/dq/projects/Splittree/test/Error Detection soccer/active_cleaning_loop_runs_soccer_v4/iter_3_infer/inference_all_predictions.csv",
        output_dir="eval_outputs_soccer_v4_with_verifier",
        columns=SOCCER_COLUMNS,
        missing_tokens=MISSING_TOKENS_EXTENDED,
        special="soccer",
        export_group_summaries=True,
        useful_cols=BASE_USEFUL_COLS + SOCCER_USEFUL_COLS,
    ),
}


# ============================================================
# 2. 基础函数
# ============================================================

def maybe_drop_unnamed_columns(df: pd.DataFrame, drop_unnamed: bool = True) -> pd.DataFrame:
    if not drop_unnamed:
        return df.copy()
    keep_cols = [c for c in df.columns if not str(c).lower().startswith("unnamed:")]
    return df[keep_cols].copy()


def normalize_value(x, missing_tokens: Set[str]):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.lower() in missing_tokens:
        return None
    return s


def normalize_df(df: pd.DataFrame, missing_tokens: Set[str]) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        df[col] = df[col].map(lambda x: normalize_value(x, missing_tokens))
    return df


def safe_int(x, default=0):
    try:
        if pd.isna(x):
            return default
        return int(float(x))
    except Exception:
        return default


def safe_float(x, default=None):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def as_int01(v, default=0) -> int:
    try:
        if pd.isna(v) or v is None:
            return default
    except Exception:
        pass
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "error"}:
        return 1
    if s in {"0", "false", "no", "n", "correct"}:
        return 0
    try:
        return int(float(s))
    except Exception:
        return default


def safe_str(v, default=""):
    try:
        if pd.isna(v) or v is None:
            return default
    except Exception:
        pass
    return str(v)


def safe_num_series(df: pd.DataFrame, col: str, default=0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series([default] * len(df), index=df.index)


def safe_str_series(df: pd.DataFrame, col: str, default="") -> pd.Series:
    if col in df.columns:
        return df[col].fillna(default).astype(str)
    return pd.Series([default] * len(df), index=df.index)


def load_table(path: str, cfg: EvalConfig) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df = maybe_drop_unnamed_columns(df, cfg.drop_unnamed_columns)
    df = normalize_df(df, cfg.missing_tokens)
    return df


def make_key(row_id: int, col: str) -> Tuple[int, str]:
    return int(row_id), str(col)


# ============================================================
# 3. adult / soccer 评估补齐字段
# ============================================================

def add_eval_adult_flags_to_infer_df(infer_df: pd.DataFrame) -> pd.DataFrame:
    df = infer_df.copy()

    int_cols = [
        "target_is_primary_column", "target_is_context_only_column",
        "is_relationship_spouse_value", "relationship_marital_conflict",
        "relationship_age_conflict", "relationship_sex_conflict",
        "sex_value_invalid", "education_age_extreme_conflict",
        "has_relationship_v2_rule", "has_education_v2_rule",
        "relationship_v2_conflict_count", "has_relationship_v2_signal",
        "has_education_v2_signal", "is_context_only_weak_signal",
        "adult_v2_primary_target_signal",
    ]
    for c in int_cols:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    if "adult_v2_signal_strength" not in df.columns:
        df["adult_v2_signal_strength"] = 0.0
    df["adult_v2_signal_strength"] = pd.to_numeric(df["adult_v2_signal_strength"], errors="coerce").fillna(0.0)

    if "adult_v2_attribution_bucket" not in df.columns:
        df["adult_v2_attribution_bucket"] = "unknown_attribution"
    df["adult_v2_attribution_bucket"] = df["adult_v2_attribution_bucket"].fillna("unknown_attribution").astype(str)

    col = df["column"].astype(str)
    rule = safe_str_series(df, "main_rule_type", "")

    df["target_is_primary_column"] = ((df["target_is_primary_column"] == 1) | col.isin(ADULT_PRIMARY_TARGET_COLUMNS)).astype(int)
    df["target_is_context_only_column"] = ((df["target_is_context_only_column"] == 1) | col.isin(ADULT_CONTEXT_ONLY_COLUMNS)).astype(int)
    df["has_relationship_v2_rule"] = ((df["has_relationship_v2_rule"] == 1) | rule.isin(ADULT_RELATIONSHIP_V2_RULE_TYPES)).astype(int)
    df["has_education_v2_rule"] = ((df["has_education_v2_rule"] == 1) | rule.isin(ADULT_EDUCATION_V2_RULE_TYPES)).astype(int)

    df["relationship_v2_conflict_count"] = (
        df["relationship_marital_conflict"]
        + df["relationship_age_conflict"]
        + df["relationship_sex_conflict"]
    ).astype(int)

    df["has_relationship_v2_signal"] = (
        (df["has_relationship_v2_signal"] == 1)
        | (df["relationship_v2_conflict_count"] > 0)
        | (df["has_relationship_v2_rule"] == 1)
    ).astype(int)

    df["has_education_v2_signal"] = (
        (df["has_education_v2_signal"] == 1)
        | (df["education_age_extreme_conflict"] == 1)
        | (df["has_education_v2_rule"] == 1)
    ).astype(int)

    adult_domain = safe_num_series(df, "adult_domain_rule_count", 0)
    adult_format = safe_num_series(df, "adult_format_rule_count", 0)
    adult_cons = safe_num_series(df, "adult_consistency_rule_count", 0)

    df["is_context_only_weak_signal"] = (
        (df["target_is_context_only_column"] == 1)
        & (adult_domain <= 0)
        & (adult_format <= 0)
        & (adult_cons <= 0)
        & (df["has_relationship_v2_signal"] <= 0)
        & (df["has_education_v2_signal"] <= 0)
        & (df["sex_value_invalid"] <= 0)
    ).astype(int)

    need_bucket = df["adult_v2_attribution_bucket"].isin(["", "missing", "unknown", "unknown_attribution", "nan", "None"])
    df.loc[need_bucket & (df["relationship_marital_conflict"] == 1), "adult_v2_attribution_bucket"] = "relationship_marital_conflict"
    df.loc[need_bucket & (df["relationship_age_conflict"] == 1), "adult_v2_attribution_bucket"] = "relationship_age_conflict"
    df.loc[need_bucket & (df["relationship_sex_conflict"] == 1), "adult_v2_attribution_bucket"] = "relationship_sex_conflict"
    df.loc[need_bucket & (df["has_relationship_v2_signal"] == 1), "adult_v2_attribution_bucket"] = "relationship_v2_rule"
    df.loc[need_bucket & (df["sex_value_invalid"] == 1), "adult_v2_attribution_bucket"] = "sex_value_invalid"
    df.loc[need_bucket & (df["has_education_v2_signal"] == 1), "adult_v2_attribution_bucket"] = "education_age_extreme_conflict"
    df.loc[need_bucket & col.isin(ADULT_PRIMARY_TARGET_COLUMNS), "adult_v2_attribution_bucket"] = "primary_target_other"
    df.loc[need_bucket & col.isin(ADULT_CONTEXT_ONLY_COLUMNS), "adult_v2_attribution_bucket"] = "context_only_other"

    df["adult_v2_primary_target_signal"] = (
        (df["target_is_primary_column"] == 1)
        & (
            (df["has_relationship_v2_signal"] == 1)
            | (df["sex_value_invalid"] == 1)
            | (df["has_education_v2_signal"] == 1)
            | (adult_domain > 0)
            | (adult_format > 0)
            | (adult_cons > 0)
        )
    ).astype(int)

    return df


def add_eval_soccer_flags_to_infer_df(infer_df: pd.DataFrame) -> pd.DataFrame:
    df = infer_df.copy()

    numeric_flag_cols = [
        "target_is_primary_column", "target_is_context_only_column",
        "birthyear_value_invalid", "season_value_invalid",
        "position_value_invalid", "position_needs_canonicalization",
        "birthyear_season_age_conflict", "team_context_conflict",
        "format_rule_signal", "fd_like_signal", "evidence_only_fd_signal",
        "position_invalid_or_normalization", "has_birthyear_or_season_signal",
        "has_team_context_signal", "has_fd_like_signal",
        "is_context_only_weak_signal", "soccer_primary_target_signal",
    ]
    for c in numeric_flag_cols:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    if "soccer_signal_strength" not in df.columns:
        df["soccer_signal_strength"] = 0.0
    df["soccer_signal_strength"] = pd.to_numeric(df["soccer_signal_strength"], errors="coerce").fillna(0.0)

    if "soccer_attribution_bucket" not in df.columns:
        df["soccer_attribution_bucket"] = "unknown_attribution"
    df["soccer_attribution_bucket"] = df["soccer_attribution_bucket"].fillna("unknown_attribution").astype(str)

    col = df["column"].astype(str)
    rule = safe_str_series(df, "main_rule_type", "")

    # Soccer v2: all soccer columns are candidate target columns.
    df["target_is_primary_column"] = col.isin(SOCCER_PRIMARY_TARGET_COLUMNS).astype(int)
    df["target_is_context_only_column"] = col.isin(SOCCER_CONTEXT_ONLY_COLUMNS).astype(int)

    existing_evidence_only_fd_signal = safe_num_series(df, "evidence_only_fd_signal", 0).astype(int)
    df["evidence_only_fd_signal"] = (
        (existing_evidence_only_fd_signal == 1)
        | rule.isin(SOCCER_EVIDENCE_ONLY_FD_RULE_TYPES)
    ).astype(int)

    if "evidence_only_fd_count" not in df.columns:
        df["evidence_only_fd_count"] = 0
    df["evidence_only_fd_count"] = pd.to_numeric(df["evidence_only_fd_count"], errors="coerce").fillna(0)
    df.loc[rule.isin(SOCCER_EVIDENCE_ONLY_FD_RULE_TYPES), "evidence_only_fd_count"] = (
        df.loc[rule.isin(SOCCER_EVIDENCE_ONLY_FD_RULE_TYPES), "evidence_only_fd_count"].clip(lower=1)
    )

    df["birthyear_value_invalid"] = ((df["birthyear_value_invalid"] == 1) | rule.eq("birthyear_numeric_range")).astype(int)
    df["season_value_invalid"] = ((df["season_value_invalid"] == 1) | rule.eq("season_numeric_range")).astype(int)
    df["position_value_invalid"] = ((df["position_value_invalid"] == 1) | rule.eq("position_domain")).astype(int)
    df["birthyear_season_age_conflict"] = ((df["birthyear_season_age_conflict"] == 1) | rule.eq("birthyear_season_age_consistency")).astype(int)
    df["team_context_conflict"] = ((df["team_context_conflict"] == 1) | rule.isin(SOCCER_CONTEXT_RULE_TYPES)).astype(int)
    df["format_rule_signal"] = ((df["format_rule_signal"] == 1) | rule.isin(SOCCER_FORMAT_RULE_TYPES)).astype(int)
    df["fd_like_signal"] = ((df["fd_like_signal"] == 1) | rule.isin(SOCCER_FD_RULE_TYPES)).astype(int)

    df["position_invalid_or_normalization"] = (
        (df["position_invalid_or_normalization"] == 1)
        | (df["position_value_invalid"] == 1)
        | (df["position_needs_canonicalization"] == 1)
    ).astype(int)

    df["has_birthyear_or_season_signal"] = (
        (df["has_birthyear_or_season_signal"] == 1)
        | (df["birthyear_value_invalid"] == 1)
        | (df["season_value_invalid"] == 1)
        | (df["birthyear_season_age_conflict"] == 1)
    ).astype(int)

    df["has_team_context_signal"] = ((df["has_team_context_signal"] == 1) | (df["team_context_conflict"] == 1)).astype(int)

    fd_like_count = safe_num_series(df, "fd_like_count", 0)
    df["has_fd_like_signal"] = (
        (df["has_fd_like_signal"] == 1)
        | (df["fd_like_signal"] == 1)
        | ((fd_like_count > 0) & (~rule.isin(SOCCER_EVIDENCE_ONLY_FD_RULE_TYPES)))
    ).astype(int)

    # Soft-FD is only evidence-only.
    df.loc[rule.isin(SOCCER_EVIDENCE_ONLY_FD_RULE_TYPES), ["fd_like_signal", "has_fd_like_signal"]] = 0

    soccer_domain = safe_num_series(df, "soccer_domain_rule_count", 0)
    soccer_format = safe_num_series(df, "soccer_format_rule_count", 0)
    soccer_numeric = safe_num_series(df, "soccer_numeric_rule_count", 0)
    soccer_cons = safe_num_series(df, "soccer_consistency_rule_count", 0)

    df["is_context_only_weak_signal"] = (
        (df["target_is_context_only_column"] == 1)
        & (soccer_domain <= 0)
        & (soccer_format <= 0)
        & (soccer_numeric <= 0)
        & (soccer_cons <= 0)
        & (~rule.isin(SOCCER_STRONG_RULE_TYPES | SOCCER_FORMAT_RULE_TYPES))
    ).astype(int)

    df["soccer_primary_target_signal"] = (
        (df["target_is_primary_column"] == 1)
        & (
            (df["has_birthyear_or_season_signal"] == 1)
            | (df["position_invalid_or_normalization"] == 1)
            | (df["format_rule_signal"] == 1)
            | (soccer_domain > 0)
            | (soccer_format > 0)
            | (soccer_numeric > 0)
            | (soccer_cons > 0)
        )
    ).astype(int)

    need_bucket = df["soccer_attribution_bucket"].isin(["", "missing", "unknown", "unknown_attribution", "nan", "None"])
    df.loc[need_bucket & (df["birthyear_value_invalid"] == 1), "soccer_attribution_bucket"] = "birthyear_invalid"
    df.loc[need_bucket & (df["season_value_invalid"] == 1), "soccer_attribution_bucket"] = "season_invalid"
    df.loc[need_bucket & (df["position_value_invalid"] == 1), "soccer_attribution_bucket"] = "position_invalid"
    df.loc[need_bucket & (df["position_needs_canonicalization"] == 1), "soccer_attribution_bucket"] = "position_canonicalization"
    df.loc[need_bucket & (df["birthyear_season_age_conflict"] == 1), "soccer_attribution_bucket"] = "birthyear_season_age_conflict"
    df.loc[need_bucket & (df["team_context_conflict"] == 1), "soccer_attribution_bucket"] = "team_context_conflict"
    df.loc[need_bucket & (df["format_rule_signal"] == 1), "soccer_attribution_bucket"] = "format_pattern_conflict"
    df.loc[need_bucket & (df["fd_like_signal"] == 1), "soccer_attribution_bucket"] = "fd_like_conflict"
    df.loc[need_bucket & col.isin(SOCCER_PRIMARY_TARGET_COLUMNS), "soccer_attribution_bucket"] = "primary_target_other"
    df.loc[need_bucket & col.isin(SOCCER_CONTEXT_ONLY_COLUMNS), "soccer_attribution_bucket"] = "context_only_other"

    return df


# ============================================================
# 4. 构造真实错误集合与解析推理结果
# ============================================================

def build_true_error_cells(
    clean_df: pd.DataFrame,
    dirty_df: pd.DataFrame
) -> Tuple[Set[Tuple[int, str]], Dict[Tuple[int, str], Dict[str, Any]]]:
    if clean_df.shape != dirty_df.shape:
        raise ValueError(f"干净表和脏表形状不一致: clean={clean_df.shape}, dirty={dirty_df.shape}")

    if list(clean_df.columns) != list(dirty_df.columns):
        raise ValueError(
            "干净表和脏表列名或列顺序不一致，请先对齐。\n"
            f"clean columns={list(clean_df.columns)}\n"
            f"dirty columns={list(dirty_df.columns)}"
        )

    true_errors = set()
    true_error_info = {}

    for row_id in range(len(clean_df)):
        for col in clean_df.columns:
            clean_val = clean_df.iat[row_id, clean_df.columns.get_loc(col)]
            dirty_val = dirty_df.iat[row_id, dirty_df.columns.get_loc(col)]
            if clean_val != dirty_val:
                key = (row_id, col)
                true_errors.add(key)
                true_error_info[key] = {"clean_value": clean_val, "dirty_value": dirty_val}

    return true_errors, true_error_info


def parse_label_from_value(v) -> bool:
    try:
        if pd.isna(v):
            return False
    except Exception:
        pass

    val = str(v).strip().lower()
    if val in {"1", "true", "yes", "y", "error"}:
        return True
    if val in {"0", "false", "no", "n", "correct"}:
        return False

    try:
        return int(float(val)) == 1
    except Exception:
        return False


def parse_final_error_flag(row: pd.Series) -> bool:
    for col in ["final_pred_label", "pred_label", "raw_pred_label"]:
        if col in row.index and pd.notna(row[col]):
            return parse_label_from_value(row[col])
    for col in ["final_pred_label_name", "pred_label_name", "raw_pred_label_name"]:
        if col in row.index and pd.notna(row[col]):
            return parse_label_from_value(row[col])
    return False


def parse_detector_error_flag(row: pd.Series) -> bool:
    for col in ["detector_pred_label", "pred_label", "raw_pred_label"]:
        if col in row.index and pd.notna(row[col]):
            return parse_label_from_value(row[col])
    for col in ["detector_pred_label_name", "pred_label_name", "raw_pred_label_name"]:
        if col in row.index and pd.notna(row[col]):
            return parse_label_from_value(row[col])
    return False


def load_infer_result(path: str, cfg: EvalConfig):
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df = maybe_drop_unnamed_columns(df, cfg.drop_unnamed_columns)

    required = ["row_id", "column"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"推理结果文件缺少必要列: {missing}")

    df["row_id"] = df["row_id"].apply(lambda x: safe_int(x, -1))
    df["column"] = df["column"].astype(str)
    df = df[df["row_id"] >= 0].copy()

    if cfg.special == "adult":
        df = add_eval_adult_flags_to_infer_df(df)
    elif cfg.special == "soccer":
        df = add_eval_soccer_flags_to_infer_df(df)

    candidate_cells = set((int(r["row_id"]), str(r["column"])) for _, r in df.iterrows())

    predicted_error_cells = set()
    detector_predicted_error_cells = set()
    infer_row_map = {}

    for _, row in df.iterrows():
        key = (int(row["row_id"]), str(row["column"]))
        infer_row_map[key] = row.to_dict()

        if parse_final_error_flag(row):
            predicted_error_cells.add(key)
        if parse_detector_error_flag(row):
            detector_predicted_error_cells.add(key)

    return df, candidate_cells, predicted_error_cells, detector_predicted_error_cells, infer_row_map


# ============================================================
# 5. 指标计算
# ============================================================

def calc_prf(pred_set: Set[Tuple[int, str]], true_set: Set[Tuple[int, str]]) -> Dict[str, Any]:
    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    precision = tp / len(pred_set) if len(pred_set) > 0 else 0.0
    recall = tp / len(true_set) if len(true_set) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def calc_column_metrics(pred_set: Set[Tuple[int, str]], true_set: Set[Tuple[int, str]], columns: List[str]) -> pd.DataFrame:
    rows = []
    for col in columns:
        true_col = {k for k in true_set if k[1] == col}
        pred_col = {k for k in pred_set if k[1] == col}
        prf = calc_prf(pred_col, true_col)
        rows.append({
            "column": col,
            "true_errors": len(true_col),
            "predicted_errors": len(pred_col),
            "tp": prf["tp"],
            "fp": prf["fp"],
            "fn": prf["fn"],
            "precision": prf["precision"],
            "recall": prf["recall"],
            "f1": prf["f1"],
        })
    return pd.DataFrame(rows)


def calc_candidate_column_metrics(candidate_set: Set[Tuple[int, str]], true_set: Set[Tuple[int, str]], columns: List[str]) -> pd.DataFrame:
    rows = []
    for col in columns:
        true_col = {k for k in true_set if k[1] == col}
        cand_col = {k for k in candidate_set if k[1] == col}
        covered = cand_col & true_col
        missed = true_col - cand_col
        coverage = len(covered) / len(true_col) if len(true_col) > 0 else 0.0
        cand_precision = len(covered) / len(cand_col) if len(cand_col) > 0 else 0.0
        rows.append({
            "column": col,
            "candidate_cells": len(cand_col),
            "true_errors": len(true_col),
            "covered_true_errors": len(covered),
            "missed_true_errors": len(missed),
            "candidate_coverage": coverage,
            "candidate_only_precision": cand_precision,
        })
    return pd.DataFrame(rows)


# ============================================================
# 6. 明细导出
# ============================================================

def get_threshold_value(infer_info: Dict[str, Any]):
    for c in [
        "final_decision_threshold", "verifier_decision_threshold",
        "detector_decision_threshold", "decision_threshold",
        "stage2_decision_threshold", "stage1_decision_threshold",
        "detector_threshold", "group_verifier_threshold",
    ]:
        if c in infer_info:
            return infer_info.get(c)
    return None


def get_clean_dirty_value(df: pd.DataFrame, row_id: int, col: str):
    if row_id < len(df) and col in df.columns:
        return df.iloc[row_id][col]
    return None


def build_detail_rows(
    keys: Set[Tuple[int, str]],
    clean_df: pd.DataFrame,
    dirty_df: pd.DataFrame,
    infer_row_map: Dict[Tuple[int, str], Dict[str, Any]],
    true_error_info: Dict[Tuple[int, str], Dict[str, Any]],
    detail_type: str,
    cfg: EvalConfig,
) -> List[Dict[str, Any]]:
    rows = []

    useful_cols = []
    seen_cols = set()
    for c in cfg.useful_cols:
        if c not in seen_cols:
            useful_cols.append(c)
            seen_cols.add(c)

    for row_id, col in sorted(keys, key=lambda x: (x[0], x[1])):
        infer_info = infer_row_map.get((row_id, col), {})
        dirty_value = get_clean_dirty_value(dirty_df, row_id, col)
        clean_value = get_clean_dirty_value(clean_df, row_id, col)
        row_series = pd.Series(infer_info) if infer_info else pd.Series(dtype=object)

        out = {
            "detail_type": detail_type,
            "row_id": row_id,
            "column": col,
            "dirty_value": dirty_value,
            "clean_value": clean_value,
            "is_true_error": int((row_id, col) in true_error_info),
            "detector_predicted_error": int(parse_detector_error_flag(row_series)) if len(row_series) > 0 else 0,
            "final_predicted_error": int(parse_final_error_flag(row_series)) if len(row_series) > 0 else 0,
        }

        for c in useful_cols:
            if c in infer_info:
                out[c] = infer_info.get(c)

        threshold_value = get_threshold_value(infer_info)
        if threshold_value is not None:
            out["decision_threshold"] = threshold_value

        rows.append(out)

    return rows


def export_detail_files(
    output_dir: str,
    clean_df: pd.DataFrame,
    dirty_df: pd.DataFrame,
    infer_row_map: Dict[Tuple[int, str], Dict[str, Any]],
    true_error_info: Dict[Tuple[int, str], Dict[str, Any]],
    tp_keys: Set[Tuple[int, str]],
    fp_keys: Set[Tuple[int, str]],
    fn_keys: Set[Tuple[int, str]],
    detector_tp_keys: Set[Tuple[int, str]],
    detector_fp_keys: Set[Tuple[int, str]],
    detector_fn_keys: Set[Tuple[int, str]],
    covered_true_errors: Set[Tuple[int, str]],
    candidate_missed_true_errors: Set[Tuple[int, str]],
    cfg: EvalConfig,
):
    detail_specs = [
        ("tp_details.csv", tp_keys, "TP"),
        ("fp_details.csv", fp_keys, "FP"),
        ("fn_details.csv", fn_keys, "FN"),
        ("detector_tp_details.csv", detector_tp_keys, "DETECTOR_TP"),
        ("detector_fp_details.csv", detector_fp_keys, "DETECTOR_FP"),
        ("detector_fn_details.csv", detector_fn_keys, "DETECTOR_FN"),
        ("candidate_covered_true_errors.csv", covered_true_errors, "CANDIDATE_COVERED_TRUE_ERROR"),
        ("candidate_missed_true_errors.csv", candidate_missed_true_errors, "CANDIDATE_MISSED_TRUE_ERROR"),
    ]
    for filename, keys, dtype in detail_specs:
        rows = build_detail_rows(keys, clean_df, dirty_df, infer_row_map, true_error_info, dtype, cfg)
        pd.DataFrame(rows).to_csv(os.path.join(output_dir, filename), index=False, encoding="utf-8-sig")


# ============================================================
# 7. 分组统计与 v2 诊断
# ============================================================

def make_tmp_for_group(infer_df: pd.DataFrame, true_error_cells: Set[Tuple[int, str]]) -> pd.DataFrame:
    tmp = infer_df.copy()
    tmp["_key"] = list(zip(tmp["row_id"].astype(int), tmp["column"].astype(str)))
    tmp["_is_true_error"] = tmp["_key"].isin(true_error_cells).astype(int)
    tmp["_final_pred"] = tmp.apply(parse_final_error_flag, axis=1).astype(int)
    tmp["_detector_pred"] = tmp.apply(parse_detector_error_flag, axis=1).astype(int)
    return tmp


def export_group_summaries(
    infer_df: pd.DataFrame,
    true_error_cells: Set[Tuple[int, str]],
    predicted_error_cells: Set[Tuple[int, str]],
    detector_predicted_error_cells: Set[Tuple[int, str]],
    candidate_cells: Set[Tuple[int, str]],
    output_dir: str,
    cfg: EvalConfig,
):
    if not cfg.export_group_summaries:
        return

    if cfg.columns:
        all_cols = list(cfg.columns)
    else:
        all_cols = sorted(infer_df["column"].dropna().astype(str).unique().tolist())

    candidate_col_df = calc_candidate_column_metrics(candidate_cells, true_error_cells, all_cols)
    detector_col_df = calc_column_metrics(detector_predicted_error_cells, true_error_cells, all_cols)
    final_col_df = calc_column_metrics(predicted_error_cells, true_error_cells, all_cols)

    candidate_col_df.to_csv(os.path.join(output_dir, "candidate_metrics_by_column.csv"), index=False, encoding="utf-8-sig")
    detector_col_df.to_csv(os.path.join(output_dir, "detector_metrics_by_column.csv"), index=False, encoding="utf-8-sig")
    final_col_df.to_csv(os.path.join(output_dir, "final_metrics_by_column.csv"), index=False, encoding="utf-8-sig")

    if cfg.special == "adult":
        group_cols = [
            "column", "main_rule_type", "domain_bucket", "adult_consistency_bucket",
            "adult_v2_attribution_bucket", "target_is_primary_column",
            "target_is_context_only_column", "has_relationship_v2_signal",
            "sex_value_invalid", "has_education_v2_signal", "is_context_only_weak_signal",
            "neighbor_bucket", "rarity_bucket", "pattern_bucket", "verifier_decision",
            "final_pred_label_name", "detector_pred_label_name",
        ]
    elif cfg.special == "soccer":
        group_cols = [
            "column", "main_rule_type", "domain_bucket", "soccer_consistency_bucket",
            "soccer_attribution_bucket", "target_is_primary_column",
            "target_is_context_only_column", "has_birthyear_or_season_signal",
            "position_invalid_or_normalization", "has_team_context_signal",
            "has_fd_like_signal", "evidence_only_fd_signal",
            "is_context_only_weak_signal", "neighbor_bucket", "rarity_bucket",
            "pattern_bucket", "verifier_decision", "final_pred_label_name",
            "detector_pred_label_name",
        ]
    else:
        group_cols = [
            "column", "main_rule_type", "neighbor_bucket", "rarity_bucket",
            "pattern_bucket", "verifier_decision", "final_pred_label_name",
            "detector_pred_label_name",
        ]

    tmp = make_tmp_for_group(infer_df, true_error_cells)
    for gcol in group_cols:
        if gcol not in tmp.columns:
            continue
        summary = tmp.groupby(gcol, dropna=False).agg(
            rows=("_key", "count"),
            true_errors=("_is_true_error", "sum"),
            final_pred_errors=("_final_pred", "sum"),
            detector_pred_errors=("_detector_pred", "sum"),
        ).reset_index()

        summary["true_error_rate_in_group"] = summary["true_errors"] / summary["rows"].clip(lower=1)
        summary["final_pred_rate_in_group"] = summary["final_pred_errors"] / summary["rows"].clip(lower=1)
        summary["detector_pred_rate_in_group"] = summary["detector_pred_errors"] / summary["rows"].clip(lower=1)

        safe_name = gcol.replace("/", "_")
        summary.to_csv(os.path.join(output_dir, f"group_summary_by_{safe_name}.csv"), index=False, encoding="utf-8-sig")


def export_adult_v2_diagnostics(
    infer_df: pd.DataFrame,
    true_error_cells: Set[Tuple[int, str]],
    predicted_error_cells: Set[Tuple[int, str]],
    detector_predicted_error_cells: Set[Tuple[int, str]],
    output_dir: str,
):
    tmp = make_tmp_for_group(infer_df, true_error_cells)

    if "adult_v2_attribution_bucket" in tmp.columns:
        v2_bucket = tmp.groupby("adult_v2_attribution_bucket", dropna=False).agg(
            rows=("_key", "count"),
            true_errors=("_is_true_error", "sum"),
            detector_pred_errors=("_detector_pred", "sum"),
            final_pred_errors=("_final_pred", "sum"),
        ).reset_index()
        v2_bucket["true_error_rate_in_bucket"] = v2_bucket["true_errors"] / v2_bucket["rows"].clip(lower=1)
        v2_bucket["final_pred_rate_in_bucket"] = v2_bucket["final_pred_errors"] / v2_bucket["rows"].clip(lower=1)
        v2_bucket["precision_in_bucket"] = v2_bucket.apply(
            lambda r: r["true_errors"] / r["final_pred_errors"] if r["final_pred_errors"] > 0 else 0.0,
            axis=1,
        )
        v2_bucket.to_csv(os.path.join(output_dir, "adult_v2_bucket_diagnostics.csv"), index=False, encoding="utf-8-sig")

        cross = tmp.groupby(["column", "adult_v2_attribution_bucket"], dropna=False).agg(
            rows=("_key", "count"),
            true_errors=("_is_true_error", "sum"),
            detector_pred_errors=("_detector_pred", "sum"),
            final_pred_errors=("_final_pred", "sum"),
        ).reset_index()
        cross["true_error_rate"] = cross["true_errors"] / cross["rows"].clip(lower=1)
        cross["final_pred_rate"] = cross["final_pred_errors"] / cross["rows"].clip(lower=1)
        cross.to_csv(os.path.join(output_dir, "adult_v2_column_bucket_diagnostics.csv"), index=False, encoding="utf-8-sig")

    for gcol in [
        "target_is_primary_column", "target_is_context_only_column",
        "has_relationship_v2_signal", "sex_value_invalid",
        "has_education_v2_signal", "is_context_only_weak_signal",
    ]:
        if gcol not in tmp.columns:
            continue
        summary = tmp.groupby(gcol, dropna=False).agg(
            rows=("_key", "count"),
            true_errors=("_is_true_error", "sum"),
            detector_pred_errors=("_detector_pred", "sum"),
            final_pred_errors=("_final_pred", "sum"),
        ).reset_index()
        summary["true_error_rate"] = summary["true_errors"] / summary["rows"].clip(lower=1)
        summary["final_pred_rate"] = summary["final_pred_errors"] / summary["rows"].clip(lower=1)
        summary.to_csv(os.path.join(output_dir, f"adult_v2_summary_by_{gcol}.csv"), index=False, encoding="utf-8-sig")


def export_soccer_v2_diagnostics(
    infer_df: pd.DataFrame,
    true_error_cells: Set[Tuple[int, str]],
    predicted_error_cells: Set[Tuple[int, str]],
    detector_predicted_error_cells: Set[Tuple[int, str]],
    output_dir: str,
):
    tmp = make_tmp_for_group(infer_df, true_error_cells)

    if "soccer_attribution_bucket" in tmp.columns:
        v2_bucket = tmp.groupby("soccer_attribution_bucket", dropna=False).agg(
            rows=("_key", "count"),
            true_errors=("_is_true_error", "sum"),
            detector_pred_errors=("_detector_pred", "sum"),
            final_pred_errors=("_final_pred", "sum"),
        ).reset_index()
        v2_bucket["true_error_rate_in_bucket"] = v2_bucket["true_errors"] / v2_bucket["rows"].clip(lower=1)
        v2_bucket["final_pred_rate_in_bucket"] = v2_bucket["final_pred_errors"] / v2_bucket["rows"].clip(lower=1)
        v2_bucket["precision_in_bucket"] = v2_bucket.apply(
            lambda r: r["true_errors"] / r["final_pred_errors"] if r["final_pred_errors"] > 0 else 0.0,
            axis=1,
        )
        v2_bucket.to_csv(os.path.join(output_dir, "soccer_v2_bucket_diagnostics.csv"), index=False, encoding="utf-8-sig")

        cross = tmp.groupby(["column", "soccer_attribution_bucket"], dropna=False).agg(
            rows=("_key", "count"),
            true_errors=("_is_true_error", "sum"),
            detector_pred_errors=("_detector_pred", "sum"),
            final_pred_errors=("_final_pred", "sum"),
        ).reset_index()
        cross["true_error_rate"] = cross["true_errors"] / cross["rows"].clip(lower=1)
        cross["final_pred_rate"] = cross["final_pred_errors"] / cross["rows"].clip(lower=1)
        cross.to_csv(os.path.join(output_dir, "soccer_v2_column_bucket_diagnostics.csv"), index=False, encoding="utf-8-sig")

    for gcol in [
        "target_is_primary_column", "target_is_context_only_column",
        "has_birthyear_or_season_signal", "position_invalid_or_normalization",
        "has_team_context_signal", "has_fd_like_signal", "evidence_only_fd_signal",
        "is_context_only_weak_signal", "soccer_primary_target_signal",
    ]:
        if gcol not in tmp.columns:
            continue
        summary = tmp.groupby(gcol, dropna=False).agg(
            rows=("_key", "count"),
            true_errors=("_is_true_error", "sum"),
            detector_pred_errors=("_detector_pred", "sum"),
            final_pred_errors=("_final_pred", "sum"),
        ).reset_index()
        summary["true_error_rate"] = summary["true_errors"] / summary["rows"].clip(lower=1)
        summary["final_pred_rate"] = summary["final_pred_errors"] / summary["rows"].clip(lower=1)
        summary.to_csv(os.path.join(output_dir, f"soccer_v2_summary_by_{gcol}.csv"), index=False, encoding="utf-8-sig")


# ============================================================
# 8. 主评估流程
# ============================================================

def count_verifier_used_rejected(infer_df: pd.DataFrame) -> Tuple[int, int]:
    n_verifier_used = 0
    n_verifier_rejected = 0

    if "verifier_used" in infer_df.columns:
        n_verifier_used = int(pd.to_numeric(infer_df["verifier_used"], errors="coerce").fillna(0).sum())
    elif "verifier_keep_error_prob" in infer_df.columns:
        n_verifier_used = int(infer_df["verifier_keep_error_prob"].notna().sum())

    if "verifier_rejected" in infer_df.columns:
        n_verifier_rejected = int(pd.to_numeric(infer_df["verifier_rejected"], errors="coerce").fillna(0).sum())
    elif "verifier_decision" in infer_df.columns:
        n_verifier_rejected = int((infer_df["verifier_decision"].astype(str) == "reject_error").sum())

    return n_verifier_used, n_verifier_rejected


def build_summary(
    cfg: EvalConfig,
    clean_csv: str,
    dirty_csv: str,
    infer_csv: str,
    clean_df: pd.DataFrame,
    true_error_cells: Set[Tuple[int, str]],
    candidate_cells: Set[Tuple[int, str]],
    covered_true_errors: Set[Tuple[int, str]],
    candidate_missed_true_errors: Set[Tuple[int, str]],
    detector_predicted_error_cells: Set[Tuple[int, str]],
    predicted_error_cells: Set[Tuple[int, str]],
    detector_prf: Dict[str, Any],
    prf: Dict[str, Any],
    infer_df: pd.DataFrame,
    n_verifier_used: int,
    n_verifier_rejected: int,
) -> Dict[str, Any]:
    summary = {
        "dataset": cfg.dataset,
        "clean_csv": clean_csv,
        "dirty_csv": dirty_csv,
        "infer_csv": infer_csv,

        "n_rows": len(clean_df),
        "n_cols": len(clean_df.columns),
        "n_cells": len(clean_df) * len(clean_df.columns),

        "n_true_errors": len(true_error_cells),

        "n_candidate_cells": len(candidate_cells),
        "n_true_errors_covered_by_candidate": len(covered_true_errors),
        "n_true_errors_missed_by_candidate": len(candidate_missed_true_errors),
        "candidate_coverage": len(covered_true_errors) / len(true_error_cells) if true_error_cells else 0.0,
        "candidate_only_precision": len(covered_true_errors) / len(candidate_cells) if candidate_cells else 0.0,

        "n_detector_predicted_errors": len(detector_predicted_error_cells),
        "detector_tp": detector_prf["tp"],
        "detector_fp": detector_prf["fp"],
        "detector_fn": detector_prf["fn"],
        "detector_precision": detector_prf["precision"],
        "detector_recall": detector_prf["recall"],
        "detector_f1": detector_prf["f1"],

        "n_predicted_errors": len(predicted_error_cells),
        "tp": prf["tp"],
        "fp": prf["fp"],
        "fn": prf["fn"],
        "precision": prf["precision"],
        "recall": prf["recall"],
        "f1": prf["f1"],

        "n_verifier_used": n_verifier_used,
        "n_verifier_rejected": n_verifier_rejected,
    }

    if cfg.special == "adult":
        summary.update({
            "n_primary_target_candidates": int(safe_num_series(infer_df, "target_is_primary_column", 0).sum()),
            "n_context_only_candidates": int(safe_num_series(infer_df, "target_is_context_only_column", 0).sum()),
            "n_relationship_v2_signal": int(safe_num_series(infer_df, "has_relationship_v2_signal", 0).sum()),
            "n_sex_value_invalid_signal": int(safe_num_series(infer_df, "sex_value_invalid", 0).sum()),
            "n_education_v2_signal": int(safe_num_series(infer_df, "has_education_v2_signal", 0).sum()),
            "n_context_only_weak_signal": int(safe_num_series(infer_df, "is_context_only_weak_signal", 0).sum()),
        })

    if cfg.special == "soccer":
        summary.update({
            "n_primary_target_candidates": int(safe_num_series(infer_df, "target_is_primary_column", 0).sum()),
            "n_context_only_candidates": int(safe_num_series(infer_df, "target_is_context_only_column", 0).sum()),
            "n_birthyear_or_season_signal": int(safe_num_series(infer_df, "has_birthyear_or_season_signal", 0).sum()),
            "n_position_signal": int(safe_num_series(infer_df, "position_invalid_or_normalization", 0).sum()),
            "n_team_context_signal": int(safe_num_series(infer_df, "has_team_context_signal", 0).sum()),
            "n_fd_like_signal": int(safe_num_series(infer_df, "has_fd_like_signal", 0).sum()),
            "n_evidence_only_fd_signal": int(safe_num_series(infer_df, "evidence_only_fd_signal", 0).sum()),
            "n_soccer_primary_target_signal": int(safe_num_series(infer_df, "soccer_primary_target_signal", 0).sum()),
            "n_context_only_weak_signal": int(safe_num_series(infer_df, "is_context_only_weak_signal", 0).sum()),
        })

    if "high_risk" in infer_df.columns:
        summary["n_high_risk"] = int(pd.to_numeric(infer_df["high_risk"], errors="coerce").fillna(0).sum())
    elif "is_high_risk" in infer_df.columns:
        summary["n_high_risk"] = int(pd.to_numeric(infer_df["is_high_risk"], errors="coerce").fillna(0).sum())

    if "is_hard_case" in infer_df.columns:
        summary["n_hard_case"] = int(pd.to_numeric(infer_df["is_hard_case"], errors="coerce").fillna(0).sum())

    return summary


def print_summary(summary: Dict[str, Any], output_dir: str, cfg: EvalConfig):
    if cfg.display_name:
        print(f"========== {cfg.display_name} 评估结果 ==========")
    else:
        print("========== 评估结果 ==========")

    print(f"总行数: {summary['n_rows']}")
    print(f"总列数: {summary['n_cols']}")
    print(f"总单元格数: {summary['n_cells']}")
    print()

    print("---- 真实错误统计 ----")
    print(f"真实错误单元格数: {summary['n_true_errors']}")
    print()

    print("---- 候选范围统计 ----")
    print(f"候选范围单元格数: {summary['n_candidate_cells']}")
    print(f"进入候选范围的真实错误数: {summary['n_true_errors_covered_by_candidate']}")
    print(f"未进入候选范围的真实错误数: {summary['n_true_errors_missed_by_candidate']}")
    print(f"Candidate Coverage: {summary['candidate_coverage']:.6f}")
    print(f"Candidate-Only Precision: {summary['candidate_only_precision']:.6f}")
    print()

    print("---- Detector-Only 指标 ----")
    print(f"Detector 预测为错误的单元格数: {summary['n_detector_predicted_errors']}")
    print(f"Detector TP: {summary['detector_tp']}")
    print(f"Detector FP: {summary['detector_fp']}")
    print(f"Detector FN: {summary['detector_fn']}")
    print(f"Detector Precision: {summary['detector_precision']:.6f}")
    print(f"Detector Recall: {summary['detector_recall']:.6f}")
    print(f"Detector F1: {summary['detector_f1']:.6f}")
    print()

    print("---- 最终错误检测指标（Verifier后）----")
    print(f"最终预测为错误的单元格数: {summary['n_predicted_errors']}")
    print(f"TP: {summary['tp']}")
    print(f"FP: {summary['fp']}")
    print(f"FN: {summary['fn']}")
    print(f"Precision: {summary['precision']:.6f}")
    print(f"Recall: {summary['recall']:.6f}")
    print(f"F1: {summary['f1']:.6f}")
    print()

    if cfg.special == "adult":
        print("---- Adult v2 诊断统计 ----")
        print(f"Primary target 候选数: {summary.get('n_primary_target_candidates', 0)}")
        print(f"Context-only 候选数: {summary.get('n_context_only_candidates', 0)}")
        print(f"Relationship v2 signal 数量: {summary.get('n_relationship_v2_signal', 0)}")
        print(f"Sex invalid signal 数量: {summary.get('n_sex_value_invalid_signal', 0)}")
        print(f"Education v2 signal 数量: {summary.get('n_education_v2_signal', 0)}")
        print(f"Context-only weak signal 数量: {summary.get('n_context_only_weak_signal', 0)}")
        print()

    if cfg.special == "soccer":
        print("---- Soccer v2 诊断统计 ----")
        print(f"Primary target 候选数: {summary.get('n_primary_target_candidates', 0)}")
        print(f"Context-only 候选数: {summary.get('n_context_only_candidates', 0)}")
        print(f"Birthyear/Season signal 数量: {summary.get('n_birthyear_or_season_signal', 0)}")
        print(f"Position signal 数量: {summary.get('n_position_signal', 0)}")
        print(f"Team-context signal 数量: {summary.get('n_team_context_signal', 0)}")
        print(f"FD-like signal 数量: {summary.get('n_fd_like_signal', 0)}")
        print(f"Evidence-only soft-FD signal 数量: {summary.get('n_evidence_only_fd_signal', 0)}")
        print(f"Soccer primary target signal 数量: {summary.get('n_soccer_primary_target_signal', 0)}")
        print(f"Context-only weak signal 数量: {summary.get('n_context_only_weak_signal', 0)}")
        print()

    print("---- Verifier 统计 ----")
    print(f"Verifier 参与样本数: {summary['n_verifier_used']}")
    print(f"Verifier 打回 correct 数: {summary['n_verifier_rejected']}")
    if "n_high_risk" in summary:
        print(f"High Risk 数量: {summary['n_high_risk']}")
    if "n_hard_case" in summary:
        print(f"Hard Case 数量: {summary['n_hard_case']}")
    print()

    print("---- 明细文件输出 ----")
    base_files = [
        "evaluation_summary.csv", "tp_details.csv", "fp_details.csv", "fn_details.csv",
        "detector_tp_details.csv", "detector_fp_details.csv", "detector_fn_details.csv",
        "candidate_covered_true_errors.csv", "candidate_missed_true_errors.csv",
    ]
    for name in base_files:
        print(os.path.join(output_dir, name))

    if cfg.export_group_summaries:
        extra_files = [
            "candidate_metrics_by_column.csv",
            "detector_metrics_by_column.csv",
            "final_metrics_by_column.csv",
        ]
        if cfg.special == "adult":
            extra_files += [
                "adult_v2_bucket_diagnostics.csv",
                "adult_v2_column_bucket_diagnostics.csv",
            ]
        if cfg.special == "soccer":
            extra_files += [
                "soccer_v2_bucket_diagnostics.csv",
                "soccer_v2_column_bucket_diagnostics.csv",
            ]
        for name in extra_files:
            print(os.path.join(output_dir, name))


def evaluate_detection(clean_csv: str, dirty_csv: str, infer_csv: str, output_dir: str, cfg: EvalConfig):
    os.makedirs(output_dir, exist_ok=True)

    clean_df = load_table(clean_csv, cfg)
    dirty_df = load_table(dirty_csv, cfg)

    true_error_cells, true_error_info = build_true_error_cells(clean_df, dirty_df)
    infer_df, candidate_cells, predicted_error_cells, detector_predicted_error_cells, infer_row_map = load_infer_result(infer_csv, cfg)

    covered_true_errors = true_error_cells & candidate_cells
    candidate_missed_true_errors = true_error_cells - candidate_cells

    detector_tp_keys = detector_predicted_error_cells & true_error_cells
    detector_fp_keys = detector_predicted_error_cells - true_error_cells
    detector_fn_keys = true_error_cells - detector_predicted_error_cells
    detector_prf = calc_prf(detector_predicted_error_cells, true_error_cells)

    tp_keys = predicted_error_cells & true_error_cells
    fp_keys = predicted_error_cells - true_error_cells
    fn_keys = true_error_cells - predicted_error_cells
    prf = calc_prf(predicted_error_cells, true_error_cells)

    n_verifier_used, n_verifier_rejected = count_verifier_used_rejected(infer_df)

    export_detail_files(
        output_dir=output_dir,
        clean_df=clean_df,
        dirty_df=dirty_df,
        infer_row_map=infer_row_map,
        true_error_info=true_error_info,
        tp_keys=tp_keys,
        fp_keys=fp_keys,
        fn_keys=fn_keys,
        detector_tp_keys=detector_tp_keys,
        detector_fp_keys=detector_fp_keys,
        detector_fn_keys=detector_fn_keys,
        covered_true_errors=covered_true_errors,
        candidate_missed_true_errors=candidate_missed_true_errors,
        cfg=cfg,
    )

    export_group_summaries(
        infer_df=infer_df,
        true_error_cells=true_error_cells,
        predicted_error_cells=predicted_error_cells,
        detector_predicted_error_cells=detector_predicted_error_cells,
        candidate_cells=candidate_cells,
        output_dir=output_dir,
        cfg=cfg,
    )

    if cfg.special == "adult":
        export_adult_v2_diagnostics(
            infer_df=infer_df,
            true_error_cells=true_error_cells,
            predicted_error_cells=predicted_error_cells,
            detector_predicted_error_cells=detector_predicted_error_cells,
            output_dir=output_dir,
        )
    elif cfg.special == "soccer":
        export_soccer_v2_diagnostics(
            infer_df=infer_df,
            true_error_cells=true_error_cells,
            predicted_error_cells=predicted_error_cells,
            detector_predicted_error_cells=detector_predicted_error_cells,
            output_dir=output_dir,
        )

    summary = build_summary(
        cfg=cfg,
        clean_csv=clean_csv,
        dirty_csv=dirty_csv,
        infer_csv=infer_csv,
        clean_df=clean_df,
        true_error_cells=true_error_cells,
        candidate_cells=candidate_cells,
        covered_true_errors=covered_true_errors,
        candidate_missed_true_errors=candidate_missed_true_errors,
        detector_predicted_error_cells=detector_predicted_error_cells,
        predicted_error_cells=predicted_error_cells,
        detector_prf=detector_prf,
        prf=prf,
        infer_df=infer_df,
        n_verifier_used=n_verifier_used,
        n_verifier_rejected=n_verifier_rejected,
    )

    pd.DataFrame([summary]).to_csv(
        os.path.join(output_dir, "evaluation_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    print_summary(summary, output_dir, cfg)
    return summary


# ============================================================
# 9. CLI
# ============================================================

def clone_config(cfg: EvalConfig) -> EvalConfig:
    return EvalConfig(
        dataset=cfg.dataset,
        display_name=cfg.display_name,
        dataset_dir=cfg.dataset_dir,
        clean_csv=cfg.clean_csv,
        dirty_csv=cfg.dirty_csv,
        infer_csv=cfg.infer_csv,
        output_dir=cfg.output_dir,
        columns=list(cfg.columns),
        missing_tokens=set(cfg.missing_tokens),
        drop_unnamed_columns=cfg.drop_unnamed_columns,
        special=cfg.special,
        export_group_summaries=cfg.export_group_summaries,
        useful_cols=list(cfg.useful_cols),
    )


def apply_overrides(cfg: EvalConfig, args) -> EvalConfig:
    cfg = clone_config(cfg)
    if args.clean_csv:
        cfg.clean_csv = args.clean_csv
    if args.dirty_csv:
        cfg.dirty_csv = args.dirty_csv
    if args.infer_csv:
        cfg.infer_csv = args.infer_csv
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.no_group_summaries:
        cfg.export_group_summaries = False
    if args.group_summaries:
        cfg.export_group_summaries = True
    return cfg


def run_one_dataset(dataset: str, args):
    cfg = apply_overrides(DATASET_CONFIGS[dataset], args)

    old_cwd = None
    target_cwd = None
    if getattr(args, "cwd", None):
        target_cwd = args.cwd
    elif getattr(args, "root", None):
        target_cwd = os.path.join(args.root, cfg.dataset_dir)

    if target_cwd:
        old_cwd = os.getcwd()
        os.chdir(target_cwd)

    try:
        return evaluate_detection(
            clean_csv=cfg.clean_csv,
            dirty_csv=cfg.dirty_csv,
            infer_csv=cfg.infer_csv,
            output_dir=cfg.output_dir,
            cfg=cfg,
        )
    finally:
        if old_cwd is not None:
            os.chdir(old_cwd)


def parse_args():
    parser = argparse.ArgumentParser(description="Unified error detection evaluation script.")
    parser.add_argument("--dataset", required=True, choices=sorted(list(DATASET_CONFIGS.keys()) + ["all"]))

    parser.add_argument("--root", default=None, help="Root dir containing Error Detection xxx folders; useful with --dataset all.")
    parser.add_argument("--cwd", default=None, help="Run evaluation after chdir to this dataset directory.")

    parser.add_argument("--clean_csv", default=None)
    parser.add_argument("--dirty_csv", default=None)
    parser.add_argument("--infer_csv", default=None)
    parser.add_argument("--output_dir", default=None)

    parser.add_argument("--group-summaries", action="store_true", help="Force export group summaries.")
    parser.add_argument("--no-group-summaries", action="store_true", help="Disable group summaries.")

    parser.add_argument(
        "--summary_csv",
        default=None,
        help="When --dataset all, optionally save all evaluation summaries into this CSV.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.dataset == "all":
        if args.clean_csv or args.dirty_csv or args.infer_csv or args.output_dir or args.cwd:
            raise ValueError("--dataset all 时不要使用 --cwd / --clean_csv / --dirty_csv / --infer_csv / --output_dir。请用 --root 指定 test 根目录。")

        summaries = []
        for ds in ["hospital", "flights", "beers", "rayyan", "adult", "soccer"]:
            print("\n" + "=" * 80)
            print(f"[DATASET] {ds}")
            print("=" * 80)
            summary = run_one_dataset(ds, args)
            summaries.append(summary)

        if args.summary_csv:
            pd.DataFrame(summaries).to_csv(args.summary_csv, index=False, encoding="utf-8-sig")
    else:
        run_one_dataset(args.dataset, args)


if __name__ == "__main__":
    main()

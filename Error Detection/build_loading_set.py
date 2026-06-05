#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unified_build_infer_candidate_features_from_whitelist.py

统一版构建推理集脚本，覆盖：
hospital / flights / beers / rayyan / adult / soccer

核心流程：
1. 读取 summarize_full_contexts 输出的全候选 flat CSV；
2. 读取 cluster_and_sample_candidates 输出的聚类/采样 CSV；
3. 按 row_id + column 回填 bucket_id / cluster_id / sample_role / bucket 特征；
4. 补齐训练/推理脚本需要的 alignment columns；
5. 构造推理阶段使用的 derived features；
6. 按原数据集风格输出 candidate_infer_ready_*.csv。

设计目标：
- 抽象统一逻辑，便于后续做消融实验。
- 不使用 clean / dirty 对比，不读取 ground truth。
- 默认输入输出文件名保持原脚本不变。
- adult / soccer 的专属 v2 字段与兼容字段在配置和策略函数中处理。
- 支持消融/调试开关：
  --disable-cluster-merge
  --disable-derived-features
  --allow-missing-cluster-file

运行示例：
python unified_build_infer_candidate_features_from_whitelist.py --dataset soccer
python unified_build_infer_candidate_features_from_whitelist.py --dataset adult --cwd "/mnt/.../Error Detection Adult"
python unified_build_infer_candidate_features_from_whitelist.py --dataset all

注意：
- 如果想完全替换原来的 build_loading_set.py 或 build_infer_candidate_features_from_whitelist.py，
  建议先在单个数据集目录里备份旧 candidate_infer_ready_*.csv，然后运行本脚本做字段和行数 diff。
"""

import argparse
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd


# ============================================================
# 1. 通用配置与基础函数
# ============================================================

MISSING_TOKENS = {
    "", "empty", "null", "none", "nan", "na", "n/a", "missing", "unk",
    "unknown", "?", "not available", "contact airline"
}

HIGH_NEIGHBOR_CONSISTENCY_TH = 0.80
RARE_VALUE_FREQ_TH = 5
RARE_VALUE_RANK_TH = 10

DEFAULT_LABEL_SOURCE = "inference"
DEFAULT_SAMPLE_WEIGHT = 1.0
DEFAULT_PROPAGATION_CONFIDENCE = 1.0
DEFAULT_IS_OUTSIDE_CANDIDATE = 0

DEFAULT_PATTERN_BUCKET = "unknown"
DEFAULT_RARITY_BUCKET = "unknown"
DEFAULT_NEIGHBOR_BUCKET = "unknown"
DEFAULT_BUCKET_ID = "full_candidate"
DEFAULT_CLUSTER_ID = -1
DEFAULT_DIST_TO_CENTER = 0.0
DEFAULT_SAMPLE_ROLE = "full_candidate"
DEFAULT_IS_SAMPLED = 0


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(x):
            return default
        v = float(x)
        if np.isfinite(v):
            return v
        return default
    except Exception:
        return default


def safe_int(x: Any, default: int = 0) -> int:
    try:
        if pd.isna(x):
            return default
        return int(float(x))
    except Exception:
        return default


def safe_str(x: Any, default: str = "") -> str:
    try:
        if pd.isna(x):
            return default
    except Exception:
        pass
    return str(x)


def normalize_value(x: Any):
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    s = str(x).strip()
    if s.lower() in MISSING_TOKENS:
        return None
    return s


def normalize_missing_to_nan(x: Any):
    try:
        if pd.isna(x):
            return np.nan
    except Exception:
        pass
    s = str(x).strip()
    if s.lower() in MISSING_TOKENS:
        return np.nan
    return x


def maybe_drop_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [c for c in df.columns if not str(c).lower().startswith("unnamed:")]
    return df[keep_cols].copy()


def normalize_text_series_for_compare(s: pd.Series) -> pd.Series:
    return s.fillna("__NULL__").astype(str).str.strip()


def to_numeric_series(s: Any, default: float = 0.0, index=None) -> pd.Series:
    if isinstance(s, pd.Series):
        out = pd.to_numeric(s, errors="coerce")
        return out.replace([np.inf, -np.inf], np.nan).fillna(default)
    if index is None:
        return pd.Series(dtype=float)
    return pd.Series([default] * len(index), index=index, dtype=float)


def to_int_flag(s: Any, default: int = 0, index=None) -> pd.Series:
    if not isinstance(s, pd.Series):
        if index is None:
            return pd.Series(dtype=int)
        return pd.Series([default] * len(index), index=index, dtype=int)

    if s.dtype == bool:
        return s.astype(int)

    ss = s.astype(str).str.strip().str.lower()
    mapped = ss.map({
        "1": 1, "true": 1, "yes": 1, "y": 1, "t": 1,
        "0": 0, "false": 0, "no": 0, "n": 0, "f": 0,
        "nan": default, "none": default, "": default,
    })
    numeric = pd.to_numeric(s, errors="coerce")
    return mapped.fillna(numeric).fillna(default).astype(int)


def safe_div(a: Any, b: Any, index=None) -> pd.Series:
    aa = to_numeric_series(a, 0.0, index=index)
    bb = to_numeric_series(b, 0.0, index=index)
    out = aa / bb.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def parse_pipe_set(x: Any) -> Set[str]:
    try:
        if pd.isna(x) or x is None:
            return set()
    except Exception:
        pass
    s = str(x).strip()
    if not s:
        return set()
    return {p.strip() for p in re.split(r"[|,;]+", s) if p.strip()}


def parse_int_like(x: Any) -> Optional[int]:
    try:
        if x is None or pd.isna(x):
            return None
    except Exception:
        pass
    s = str(x).strip().replace(",", "")
    if re.fullmatch(r"[+-]?\d+(\.0+)?", s):
        try:
            return int(float(s))
        except Exception:
            return None
    return None


def bucketize_numeric(s: pd.Series, bins: List[float], labels: List[str], default: str) -> pd.Series:
    num = pd.to_numeric(s, errors="coerce")
    out = pd.cut(num, bins=bins, labels=labels, include_lowest=True)
    return out.astype(object).where(out.notna(), default).astype(str)


def first_existing_series(df: pd.DataFrame, col: str, default=None) -> pd.Series:
    """Return a 1-D Series even when duplicate column names exist."""
    matches = [i for i, c in enumerate(df.columns) if c == col]
    if not matches:
        return pd.Series([default] * len(df), index=df.index)

    sub = df.iloc[:, matches]
    if sub.shape[1] == 1:
        return sub.iloc[:, 0]

    cleaned_cols = []
    for j in range(sub.shape[1]):
        s = sub.iloc[:, j]
        s = s.where(~s.isna() & (s.astype(str).str.strip() != ""), np.nan)
        cleaned_cols.append(s.reset_index(drop=True))

    cleaned = pd.concat(cleaned_cols, axis=1)
    cleaned.index = df.index
    return cleaned.bfill(axis=1).iloc[:, 0].fillna(default)


def collapse_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    seen = []
    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        if c in seen:
            continue
        seen.append(c)
        out[c] = first_existing_series(df, c) if list(df.columns).count(c) > 1 else df[c]
    return out


# ============================================================
# 2. 数据集配置
# ============================================================

BASE_REQUIRED_COLS = [
    "row_id", "column", "value", "semantic_type",
    "violation_count", "conflict_score",
    "value_frequency", "value_frequency_rank",
    "neighbor_majority_value", "neighbor_majority_ratio",
    "prior_error_probability", "posterior_error_probability",
    "main_rule_type", "main_usage_role",
    "strong_rule_count", "candidate_generation_rule_count",
    "fd_like_count", "context_rule_count", "global_rule_count",
]

BASE_OPTIONAL_DEFAULTS = {
    "rare_value_count": 0,
    "typo_rule_count": 0,
    "pattern_rule_count": 0,
    "schema_rule_count": 0,
    "pattern_bucket": "",
    "rarity_bucket": "",
    "neighbor_bucket": "",
    "bucket_id": "",
    "cluster_id": np.nan,
    "dist_to_center": np.nan,
    "sample_role": "",
    "is_sampled": np.nan,
}

BASE_CLUSTER_FIELDS = [
    "row_id", "column",
    "pattern_bucket", "rarity_bucket", "neighbor_bucket",
    "bucket_id", "cluster_id", "dist_to_center", "sample_role", "is_sampled",
    "rare_value_count", "typo_rule_count", "pattern_rule_count", "schema_rule_count",
]

BASE_PREFERRED_COLS = [
    "row_id", "column", "value",

    "label", "label_binary", "label_source", "sample_weight",
    "is_outside_candidate", "propagation_confidence",

    "semantic_type",
    "violation_count", "conflict_score",
    "value_frequency", "value_frequency_rank",
    "neighbor_majority_value", "neighbor_majority_ratio",
    "prior_error_probability", "posterior_error_probability",

    "main_rule_type", "main_usage_role",
    "strong_rule_count", "candidate_generation_rule_count",
    "fd_like_count", "context_rule_count", "global_rule_count",
    "rare_value_count", "typo_rule_count", "pattern_rule_count", "schema_rule_count",

    "candidate_rule_ratio", "has_neighbor_signal", "high_neighbor_consistency",
    "is_rare_value", "neighbor_disagree", "is_equal_to_neighbor_majority",
    "neighbor_agreement_score", "rule_total_count", "strong_rule_ratio",
    "context_rule_ratio", "fd_like_ratio", "global_rule_ratio",
    "rule_strength_sum", "candidate_rule_dominant",

    "pattern_bucket", "rarity_bucket", "neighbor_bucket",

    "bucket_id", "cluster_id", "dist_to_center", "sample_role", "is_sampled",

    "error_type", "reason_short", "reason_detailed",
    "suggested_correct_value", "needs_human_review", "evidence_used",
]


@dataclass
class InferConfig:
    name: str
    input_full_candidate_csv: str
    input_clustered_csv: str
    output_infer_ready_csv: str

    required_cols: List[str] = field(default_factory=lambda: list(BASE_REQUIRED_COLS))
    optional_defaults: Dict[str, Any] = field(default_factory=lambda: dict(BASE_OPTIONAL_DEFAULTS))
    cluster_fields: List[str] = field(default_factory=lambda: list(BASE_CLUSTER_FIELDS))
    preferred_cols: List[str] = field(default_factory=lambda: list(BASE_PREFERRED_COLS))

    dataset_kind: str = "generic"  # generic / flights / beers / rayyan / adult / soccer
    allow_missing_cluster_file: bool = False

    category_cols: List[str] = field(default_factory=list)
    numeric_int_cols: List[str] = field(default_factory=list)
    numeric_float_cols: List[str] = field(default_factory=list)


def build_configs() -> Dict[str, InferConfig]:
    cfgs: Dict[str, InferConfig] = {}

    # ---------------- hospital ----------------
    cfgs["hospital"] = InferConfig(
        name="hospital",
        input_full_candidate_csv="candidate_llm_contexts_flat_v2.csv",
        input_clustered_csv="candidate_clustered_v2.csv",
        output_infer_ready_csv="candidate_infer_ready_v2.csv",
        dataset_kind="generic",
    )

    # ---------------- flights ----------------
    flights_optional = {
        **BASE_OPTIONAL_DEFAULTS,
        "time_window_rule_count": 0,
        "time_value_minutes": np.nan,
        "time_window_bucket": "",
        "time_value_bucket": "",
    }
    flights_preferred = [
        "row_id", "column", "value",
        "label", "label_binary", "label_source", "sample_weight",
        "is_outside_candidate", "propagation_confidence",
        "semantic_type",
        "violation_count", "conflict_score",
        "value_frequency", "value_frequency_rank", "time_value_minutes",
        "neighbor_majority_value", "neighbor_majority_ratio",
        "prior_error_probability", "posterior_error_probability",
        "main_rule_type", "main_usage_role",
        "strong_rule_count", "candidate_generation_rule_count",
        "fd_like_count", "context_rule_count", "global_rule_count",
        "rare_value_count", "typo_rule_count", "pattern_rule_count", "schema_rule_count",
        "time_window_rule_count",
        "candidate_rule_ratio", "has_neighbor_signal", "high_neighbor_consistency",
        "is_rare_value", "neighbor_disagree", "is_equal_to_neighbor_majority",
        "neighbor_agreement_score", "rule_total_count", "strong_rule_ratio",
        "context_rule_ratio", "fd_like_ratio", "global_rule_ratio",
        "rule_strength_sum", "candidate_rule_dominant",
        "pattern_bucket", "rarity_bucket", "neighbor_bucket",
        "time_window_bucket", "time_value_bucket",
        "bucket_id", "cluster_id", "dist_to_center", "sample_role", "is_sampled",
        "error_type", "reason_short", "reason_detailed",
        "suggested_correct_value", "needs_human_review", "evidence_used",
    ]
    cfgs["flights"] = InferConfig(
        name="flights",
        input_full_candidate_csv="candidate_llm_contexts_flat_flights.csv",
        input_clustered_csv="candidate_clustered_flights.csv",
        output_infer_ready_csv="candidate_infer_ready_flights.csv",
        required_cols=BASE_REQUIRED_COLS + [
            "rare_value_count", "typo_rule_count", "pattern_rule_count",
            "schema_rule_count", "time_window_rule_count", "time_value_minutes",
        ],
        optional_defaults=flights_optional,
        cluster_fields=BASE_CLUSTER_FIELDS + ["time_window_bucket", "time_value_bucket", "time_window_rule_count", "time_value_minutes"],
        preferred_cols=flights_preferred,
        dataset_kind="flights",
    )

    # ---------------- beers ----------------
    beers_optional = {
        **BASE_OPTIONAL_DEFAULTS,
        "numeric_window_rule_count": 0,
        "numeric_value": np.nan,
        "numeric_window_bucket": "",
        "numeric_value_bucket": "",
    }
    beers_preferred = [
        "row_id", "column", "value",
        "label", "label_binary", "label_source", "sample_weight",
        "is_outside_candidate", "propagation_confidence",
        "semantic_type",
        "violation_count", "conflict_score",
        "value_frequency", "value_frequency_rank", "numeric_value",
        "neighbor_majority_value", "neighbor_majority_ratio",
        "prior_error_probability", "posterior_error_probability",
        "main_rule_type", "main_usage_role",
        "strong_rule_count", "candidate_generation_rule_count",
        "fd_like_count", "context_rule_count", "global_rule_count",
        "rare_value_count", "typo_rule_count", "pattern_rule_count", "schema_rule_count",
        "numeric_window_rule_count",
        "candidate_rule_ratio", "has_neighbor_signal", "high_neighbor_consistency",
        "is_rare_value", "neighbor_disagree", "is_equal_to_neighbor_majority",
        "neighbor_agreement_score", "rule_total_count", "strong_rule_ratio",
        "context_rule_ratio", "fd_like_ratio", "global_rule_ratio",
        "rule_strength_sum", "candidate_rule_dominant",
        "pattern_bucket", "rarity_bucket", "neighbor_bucket",
        "numeric_window_bucket", "numeric_value_bucket",
        "bucket_id", "cluster_id", "dist_to_center", "sample_role", "is_sampled",
        "error_type", "reason_short", "reason_detailed",
        "suggested_correct_value", "needs_human_review", "evidence_used",
    ]
    cfgs["beers"] = InferConfig(
        name="beers",
        input_full_candidate_csv="candidate_llm_contexts_flat_beers.csv",
        input_clustered_csv="candidate_clustered_beers.csv",
        output_infer_ready_csv="candidate_infer_ready_beers.csv",
        required_cols=BASE_REQUIRED_COLS + [
            "rare_value_count", "typo_rule_count", "pattern_rule_count",
            "schema_rule_count", "numeric_window_rule_count", "numeric_value",
        ],
        optional_defaults=beers_optional,
        cluster_fields=BASE_CLUSTER_FIELDS + ["numeric_window_bucket", "numeric_value_bucket", "numeric_window_rule_count", "numeric_value"],
        preferred_cols=beers_preferred,
        dataset_kind="beers",
    )

    # ---------------- rayyan ----------------
    rayyan_optional = {
        **BASE_OPTIONAL_DEFAULTS,
        "canonical_rule_count": 0,
        "numeric_value": np.nan,
        "date_value": "",
        "language_canonical": "",
        "issn_canonical": "",
        "canonical_bucket": "",
        "date_value_bucket": "",
        "language_bucket": "",
    }
    rayyan_preferred = [
        "row_id", "column", "value",
        "label", "label_binary", "label_source", "sample_weight",
        "is_outside_candidate", "propagation_confidence",
        "semantic_type",
        "violation_count", "conflict_score",
        "value_frequency", "value_frequency_rank",
        "numeric_value", "date_value", "language_canonical", "issn_canonical",
        "neighbor_majority_value", "neighbor_majority_ratio",
        "prior_error_probability", "posterior_error_probability",
        "main_rule_type", "main_usage_role",
        "strong_rule_count", "candidate_generation_rule_count",
        "fd_like_count", "context_rule_count", "global_rule_count",
        "rare_value_count", "typo_rule_count", "pattern_rule_count", "schema_rule_count",
        "canonical_rule_count",
        "candidate_rule_ratio", "has_neighbor_signal", "high_neighbor_consistency",
        "is_rare_value", "neighbor_disagree", "is_equal_to_neighbor_majority",
        "neighbor_agreement_score", "rule_total_count", "strong_rule_ratio",
        "context_rule_ratio", "fd_like_ratio", "global_rule_ratio",
        "rule_strength_sum", "candidate_rule_dominant",
        "pattern_bucket", "rarity_bucket", "neighbor_bucket",
        "canonical_bucket", "date_value_bucket", "language_bucket",
        "bucket_id", "cluster_id", "dist_to_center", "sample_role", "is_sampled",
        "error_type", "reason_short", "reason_detailed",
        "suggested_correct_value", "needs_human_review", "evidence_used",
    ]
    cfgs["rayyan"] = InferConfig(
        name="rayyan",
        input_full_candidate_csv="candidate_llm_contexts_flat_rayyan.csv",
        input_clustered_csv="candidate_clustered_rayyan.csv",
        output_infer_ready_csv="candidate_infer_ready_rayyan.csv",
        required_cols=BASE_REQUIRED_COLS + [
            "rare_value_count", "typo_rule_count", "pattern_rule_count",
            "schema_rule_count", "canonical_rule_count", "numeric_value",
        ],
        optional_defaults=rayyan_optional,
        cluster_fields=BASE_CLUSTER_FIELDS + ["canonical_bucket", "date_value_bucket", "language_bucket", "canonical_rule_count", "numeric_value"],
        preferred_cols=rayyan_preferred,
        dataset_kind="rayyan",
    )

    # ---------------- adult ----------------
    adult_optional = {
        **BASE_OPTIONAL_DEFAULTS,
        "detected_type": "unknown",
        "adult_domain_rule_count": 0,
        "adult_format_rule_count": 0,
        "adult_consistency_rule_count": 0,
        "age_lower": np.nan,
        "age_upper": np.nan,
        "hours_lower": np.nan,
        "hours_upper": np.nan,
        "canonical_income": "",
        "domain_valid": "",
        "education_order": np.nan,
        "adult_value_bucket": "unknown",
        "adult_profile_inconsistency_count": 0,
        "adult_consistency_score": 0.0,
        "adult_evidence_flags": "",
        "row_age_lower": np.nan,
        "row_age_upper": np.nan,
        "row_hours_lower": np.nan,
        "row_hours_upper": np.nan,
        "row_education_order": np.nan,
        "row_income_canonical": "",
        "row_missing_count": 0,
        "row_non_missing_count": 0,
        "domain_bucket": "",
        "adult_consistency_bucket": "",
        "age_sampling_bucket": "",
        "hours_sampling_bucket": "",
        "signal_priority": 0.0,
        "target_is_primary_column": 0,
        "target_is_context_only_column": 0,
        "is_relationship_spouse_value": 0,
        "relationship_marital_conflict": 0,
        "relationship_age_conflict": 0,
        "relationship_sex_conflict": 0,
        "sex_value_invalid": 0,
        "education_age_extreme_conflict": 0,
        "has_relationship_v2_rule": 0,
        "has_education_v2_rule": 0,
        "adult_v2_attribution_bucket": "",
        "time_window_rule_count": 0,
        "time_value_minutes": np.nan,
        "time_window_bucket": "",
        "time_value_bucket": "",
    }
    adult_cluster_fields = BASE_CLUSTER_FIELDS + [
        "domain_bucket", "adult_consistency_bucket", "age_sampling_bucket", "hours_sampling_bucket",
        "signal_priority", "adult_domain_rule_count", "adult_format_rule_count",
        "adult_consistency_rule_count", "adult_value_bucket", "domain_valid",
        "adult_consistency_score", "adult_profile_inconsistency_count",
        "target_is_primary_column", "target_is_context_only_column",
        "is_relationship_spouse_value", "relationship_marital_conflict",
        "relationship_age_conflict", "relationship_sex_conflict", "sex_value_invalid",
        "education_age_extreme_conflict", "has_relationship_v2_rule",
        "has_education_v2_rule", "adult_v2_attribution_bucket",
        "time_window_rule_count", "time_value_minutes", "time_window_bucket", "time_value_bucket",
    ]
    adult_preferred = [
        "row_id", "column", "value",
        "label", "label_binary", "label_source", "sample_weight",
        "is_outside_candidate", "propagation_confidence",
        "semantic_type", "detected_type",
        "violation_count", "conflict_score",
        "value_frequency", "value_frequency_rank",
        "neighbor_majority_value", "neighbor_majority_ratio",
        "prior_error_probability", "posterior_error_probability",
        "main_rule_type", "main_usage_role",
        "strong_rule_count", "candidate_generation_rule_count",
        "fd_like_count", "context_rule_count", "global_rule_count",
        "rare_value_count", "typo_rule_count", "pattern_rule_count", "schema_rule_count",
        "adult_domain_rule_count", "adult_format_rule_count", "adult_consistency_rule_count",
        "age_lower", "age_upper", "hours_lower", "hours_upper",
        "canonical_income", "domain_valid", "education_order",
        "adult_value_bucket", "adult_profile_inconsistency_count",
        "adult_consistency_score", "adult_evidence_flags",
        "row_age_lower", "row_age_upper", "row_hours_lower", "row_hours_upper",
        "row_education_order", "row_income_canonical", "row_missing_count", "row_non_missing_count",
        "target_is_primary_column", "target_is_context_only_column",
        "is_relationship_spouse_value", "relationship_marital_conflict",
        "relationship_age_conflict", "relationship_sex_conflict",
        "sex_value_invalid", "education_age_extreme_conflict",
        "has_relationship_v2_rule", "has_education_v2_rule",
        "adult_v2_attribution_bucket",
        "time_window_rule_count", "time_value_minutes",
        "candidate_rule_ratio", "has_neighbor_signal", "high_neighbor_consistency",
        "is_rare_value", "neighbor_disagree", "is_equal_to_neighbor_majority",
        "neighbor_agreement_score", "rule_total_count", "strong_rule_ratio",
        "context_rule_ratio", "fd_like_ratio", "global_rule_ratio",
        "adult_rule_total_count", "adult_domain_rule_ratio",
        "adult_format_rule_ratio", "adult_consistency_rule_ratio",
        "rule_strength_sum", "candidate_rule_dominant",
        "domain_invalid_flag", "adult_consistency_flag", "high_adult_consistency_score",
        "has_adult_domain_signal", "has_adult_format_signal", "has_adult_consistency_signal",
        "age_span", "hours_span", "is_minor_age_bucket", "is_high_hours_bucket",
        "relationship_v2_conflict_count", "has_relationship_v2_signal",
        "has_education_v2_signal", "is_context_only_weak_signal",
        "adult_v2_signal_strength", "adult_v2_primary_target_signal",
        "pattern_bucket", "rarity_bucket", "neighbor_bucket",
        "domain_bucket", "adult_consistency_bucket",
        "age_sampling_bucket", "hours_sampling_bucket",
        "time_window_bucket", "time_value_bucket",
        "bucket_id", "cluster_id", "dist_to_center", "sample_role", "is_sampled", "signal_priority",
        "error_type", "reason_short", "reason_detailed",
        "suggested_correct_value", "needs_human_review", "evidence_used",
    ]
    cfgs["adult"] = InferConfig(
        name="adult",
        input_full_candidate_csv="candidate_llm_contexts_flat_adult.csv",
        input_clustered_csv="candidate_clustered_adult.csv",
        output_infer_ready_csv="candidate_infer_ready_adult.csv",
        required_cols=BASE_REQUIRED_COLS,
        optional_defaults=adult_optional,
        cluster_fields=adult_cluster_fields,
        preferred_cols=adult_preferred,
        dataset_kind="adult",
    )

    # ---------------- soccer ----------------
    soccer_optional = {
        **BASE_OPTIONAL_DEFAULTS,
        "detected_type": "unknown",
        "evidence_only_fd_count": 0,
        "soccer_domain_rule_count": 0,
        "soccer_format_rule_count": 0,
        "soccer_numeric_rule_count": 0,
        "soccer_consistency_rule_count": 0,
        "soccer_profile_inconsistency_count": 0,
        "soccer_consistency_score": 0.0,
        "soccer_evidence_flags": "",
        "year_value": np.nan,
        "canonical_position": "",
        "domain_valid": np.nan,
        "soccer_value_bucket": "unknown",
        "row_birthyear_int": np.nan,
        "row_season_int": np.nan,
        "row_age_at_season": np.nan,
        "row_position_canonical": "",
        "row_missing_count": 0,
        "row_non_missing_count": 0,
        "domain_bucket": "",
        "soccer_consistency_bucket": "",
        "signal_priority": "unknown",
        "target_is_primary_column": 0,
        "target_is_context_only_column": 0,
        "birthyear_value_invalid": 0,
        "season_value_invalid": 0,
        "position_value_invalid": 0,
        "position_needs_canonicalization": 0,
        "birthyear_season_age_conflict": 0,
        "team_context_conflict": 0,
        "format_rule_signal": 0,
        "fd_like_signal": 0,
        "soccer_attribution_bucket": "",
        # adult/time compatibility
        "adult_domain_rule_count": 0,
        "adult_format_rule_count": 0,
        "adult_consistency_rule_count": 0,
        "age_lower": np.nan,
        "age_upper": np.nan,
        "hours_lower": np.nan,
        "hours_upper": np.nan,
        "canonical_income": "",
        "education_order": np.nan,
        "adult_value_bucket": "soccer_not_applicable",
        "adult_profile_inconsistency_count": 0,
        "adult_consistency_score": 0.0,
        "adult_evidence_flags": "",
        "adult_consistency_bucket": "soccer_not_applicable",
        "age_sampling_bucket": "soccer_not_applicable",
        "hours_sampling_bucket": "soccer_not_applicable",
        "adult_v2_attribution_bucket": "soccer_not_applicable",
        "time_window_rule_count": 0,
        "time_value_minutes": np.nan,
        "time_window_bucket": "soccer_not_applicable",
        "time_value_bucket": "soccer_not_applicable",
    }
    soccer_cluster_fields = BASE_CLUSTER_FIELDS + [
        "domain_bucket", "soccer_consistency_bucket", "adult_consistency_bucket",
        "signal_priority",
    ]
    soccer_preferred = [
        "row_id", "column", "value",
        "label", "label_binary", "label_source", "sample_weight",
        "is_outside_candidate", "propagation_confidence",
        "semantic_type", "detected_type", "violation_count", "conflict_score",
        "value_frequency", "value_frequency_rank",
        "neighbor_majority_value", "neighbor_majority_ratio",
        "prior_error_probability", "posterior_error_probability",
        "main_rule_type", "main_usage_role",
        "strong_rule_count", "candidate_generation_rule_count",
        "fd_like_count", "evidence_only_fd_count", "context_rule_count", "global_rule_count",
        "rare_value_count", "typo_rule_count", "pattern_rule_count", "schema_rule_count",
        "soccer_domain_rule_count", "soccer_format_rule_count",
        "soccer_numeric_rule_count", "soccer_consistency_rule_count",
        "soccer_profile_inconsistency_count", "soccer_consistency_score", "soccer_evidence_flags",
        "year_value", "canonical_position", "domain_valid", "soccer_value_bucket",
        "row_birthyear_int", "row_season_int", "row_age_at_season", "row_position_canonical",
        "target_is_primary_column", "target_is_context_only_column",
        "birthyear_value_invalid", "season_value_invalid",
        "position_value_invalid", "position_needs_canonicalization",
        "birthyear_season_age_conflict", "team_context_conflict",
        "format_rule_signal", "fd_like_signal", "soccer_attribution_bucket",
        "candidate_rule_ratio", "has_neighbor_signal", "high_neighbor_consistency",
        "is_rare_value", "neighbor_disagree", "is_equal_to_neighbor_majority",
        "neighbor_agreement_score", "rule_total_count", "strong_rule_ratio",
        "context_rule_ratio", "fd_like_ratio", "global_rule_ratio",
        "soccer_rule_total_count", "soccer_domain_rule_ratio",
        "soccer_format_rule_ratio", "soccer_numeric_rule_ratio",
        "soccer_consistency_rule_ratio", "rule_strength_sum", "candidate_rule_dominant",
        "domain_invalid_flag", "soccer_consistency_flag", "high_soccer_consistency_score",
        "has_soccer_domain_signal", "has_soccer_format_signal",
        "has_soccer_numeric_signal", "has_soccer_consistency_signal",
        "birthyear_season_gap", "is_young_player_age", "is_old_player_age",
        "position_invalid_or_normalization", "has_birthyear_or_season_signal",
        "has_team_context_signal", "has_fd_like_signal",
        "is_context_only_weak_signal", "soccer_signal_strength", "soccer_primary_target_signal",
        "pattern_bucket", "rarity_bucket", "neighbor_bucket",
        "domain_bucket", "soccer_consistency_bucket",
        "adult_domain_rule_count", "adult_format_rule_count", "adult_consistency_rule_count",
        "age_lower", "age_upper", "hours_lower", "hours_upper",
        "canonical_income", "education_order", "adult_value_bucket",
        "adult_profile_inconsistency_count", "adult_consistency_score", "adult_evidence_flags",
        "adult_consistency_bucket", "age_sampling_bucket", "hours_sampling_bucket",
        "adult_v2_attribution_bucket", "time_window_rule_count", "time_value_minutes",
        "time_window_bucket", "time_value_bucket",
        "adult_rule_total_count", "adult_domain_rule_ratio", "adult_format_rule_ratio",
        "adult_consistency_rule_ratio", "adult_consistency_flag", "high_adult_consistency_score",
        "has_adult_domain_signal", "has_adult_format_signal", "has_adult_consistency_signal",
        "age_span", "hours_span", "is_minor_age_bucket", "is_high_hours_bucket",
        "relationship_v2_conflict_count", "has_relationship_v2_signal",
        "has_education_v2_signal", "adult_v2_signal_strength", "adult_v2_primary_target_signal",
        "bucket_id", "cluster_id", "dist_to_center", "sample_role", "is_sampled", "signal_priority",
        "error_type", "reason_short", "reason_detailed",
        "suggested_correct_value", "needs_human_review", "evidence_used",
    ]
    cfgs["soccer"] = InferConfig(
        name="soccer",
        input_full_candidate_csv="candidate_llm_contexts_flat_soccer.csv",
        input_clustered_csv="candidate_clustered_soccer.csv",
        output_infer_ready_csv="candidate_infer_ready_soccer.csv",
        # soccer 原脚本更宽松：只强制 row_id/column，其他字段可兜底补齐
        required_cols=["row_id", "column"],
        optional_defaults=soccer_optional,
        cluster_fields=soccer_cluster_fields,
        preferred_cols=soccer_preferred,
        dataset_kind="soccer",
        allow_missing_cluster_file=True,
    )

    # Common category/numeric settings
    for cfg in cfgs.values():
        cfg.category_cols = [
            "semantic_type", "detected_type",
            "main_rule_type", "main_usage_role",
            "pattern_bucket", "rarity_bucket", "neighbor_bucket",
            "bucket_id", "sample_role",
            "time_window_bucket", "time_value_bucket",
            "numeric_window_bucket", "numeric_value_bucket",
            "canonical_bucket", "date_value_bucket", "language_bucket",
            "domain_bucket", "adult_consistency_bucket",
            "age_sampling_bucket", "hours_sampling_bucket",
            "adult_value_bucket", "canonical_income", "domain_valid",
            "adult_evidence_flags", "row_income_canonical", "adult_v2_attribution_bucket",
            "soccer_consistency_bucket", "soccer_value_bucket", "canonical_position",
            "row_position_canonical", "soccer_evidence_flags", "adult_consistency_bucket",
            "soccer_attribution_bucket", "signal_priority",
        ]
        cfg.numeric_int_cols = [
            "violation_count", "strong_rule_count", "candidate_generation_rule_count",
            "fd_like_count", "context_rule_count", "global_rule_count",
            "rare_value_count", "typo_rule_count", "pattern_rule_count", "schema_rule_count",
            "time_window_rule_count", "numeric_window_rule_count", "canonical_rule_count",
            "adult_domain_rule_count", "adult_format_rule_count", "adult_consistency_rule_count",
            "adult_profile_inconsistency_count", "row_missing_count", "row_non_missing_count",
            "target_is_primary_column", "target_is_context_only_column",
            "is_relationship_spouse_value", "relationship_marital_conflict",
            "relationship_age_conflict", "relationship_sex_conflict",
            "sex_value_invalid", "education_age_extreme_conflict",
            "has_relationship_v2_rule", "has_education_v2_rule",
            "evidence_only_fd_count",
            "soccer_domain_rule_count", "soccer_format_rule_count",
            "soccer_numeric_rule_count", "soccer_consistency_rule_count",
            "soccer_profile_inconsistency_count",
            "birthyear_value_invalid", "season_value_invalid",
            "position_value_invalid", "position_needs_canonicalization",
            "birthyear_season_age_conflict", "team_context_conflict",
            "format_rule_signal", "fd_like_signal",
            "row_missing_count", "row_non_missing_count",
            "cluster_id", "is_sampled",
        ]
        cfg.numeric_float_cols = [
            "conflict_score", "value_frequency", "value_frequency_rank",
            "neighbor_majority_ratio", "prior_error_probability", "posterior_error_probability",
            "time_value_minutes", "numeric_value", "age_lower", "age_upper",
            "hours_lower", "hours_upper", "education_order", "adult_consistency_score",
            "row_age_lower", "row_age_upper", "row_hours_lower", "row_hours_upper",
            "row_education_order", "signal_priority", "dist_to_center",
            "year_value", "row_birthyear_int", "row_season_int",
            "row_age_at_season", "soccer_consistency_score",
        ]

    return cfgs


CONFIGS = build_configs()


# ============================================================
# 3. 读取与合并
# ============================================================

def read_csv_safe(path: str, required: bool = True) -> pd.DataFrame:
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(f"找不到输入文件: {path}")
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, low_memory=False)


def add_missing_defaults(df: pd.DataFrame, cfg: InferConfig) -> pd.DataFrame:
    df = df.copy()
    for col, val in cfg.optional_defaults.items():
        if col not in df.columns:
            df[col] = val

    # 基础字段兜底
    base_defaults = {
        "value": np.nan,
        "semantic_type": "unknown",
        "violation_count": 0,
        "conflict_score": 0.0,
        "value_frequency": 0,
        "value_frequency_rank": 999999,
        "neighbor_majority_value": "",
        "neighbor_majority_ratio": 0.0,
        "prior_error_probability": 0.0,
        "posterior_error_probability": 0.0,
        "main_rule_type": "none",
        "main_usage_role": "none",
        "strong_rule_count": 0,
        "candidate_generation_rule_count": 0,
        "fd_like_count": 0,
        "context_rule_count": 0,
        "global_rule_count": 0,
    }
    for col, val in base_defaults.items():
        if col not in df.columns:
            df[col] = val

    return df


def load_full_candidate_csv(path: str, cfg: InferConfig) -> pd.DataFrame:
    df = read_csv_safe(path, required=True)
    df = maybe_drop_unnamed_columns(collapse_duplicate_columns(df))

    missing = [c for c in cfg.required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"全候选输入文件缺少必要列: {missing}")

    df = add_missing_defaults(df, cfg)

    df["row_id"] = to_numeric_series(df["row_id"], -1).astype(int)
    df["column"] = df["column"].astype(str).str.strip()
    df["value"] = df["value"].map(normalize_missing_to_nan)
    df["neighbor_majority_value"] = df["neighbor_majority_value"].map(normalize_missing_to_nan)

    for c in cfg.category_cols:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)

    for c in cfg.numeric_int_cols:
        if c in df.columns:
            df[c] = to_numeric_series(df[c], 0).astype(int)

    for c in cfg.numeric_float_cols:
        if c in df.columns:
            default = np.nan if c in {"age_lower", "age_upper", "hours_lower", "hours_upper", "education_order", "time_value_minutes", "numeric_value", "year_value", "row_birthyear_int", "row_season_int", "row_age_at_season"} else 0.0
            df[c] = to_numeric_series(df[c], default)

    return df


def load_clustered_csv(path: str, cfg: InferConfig, allow_missing_cluster_file: bool = False) -> pd.DataFrame:
    allow_missing = allow_missing_cluster_file or cfg.allow_missing_cluster_file
    df = read_csv_safe(path, required=not allow_missing)
    if df.empty:
        print(f"[WARN] 未找到或跳过聚类文件 {path}，将只用全候选上下文构建推理集。")
        return pd.DataFrame(columns=["row_id", "column"])

    df = maybe_drop_unnamed_columns(collapse_duplicate_columns(df))
    if "row_id" not in df.columns or "column" not in df.columns:
        raise ValueError("聚类采样文件必须包含 row_id 和 column 字段")

    for col in cfg.cluster_fields:
        if col not in df.columns:
            df[col] = np.nan

    df["row_id"] = to_numeric_series(df["row_id"], -1).astype(int)
    df["column"] = df["column"].astype(str).str.strip()
    keep_cols = []
    for col in cfg.cluster_fields:
        if col not in keep_cols:
            keep_cols.append(col)

    df = df[keep_cols].copy()
    df = df.drop_duplicates(subset=["row_id", "column"], keep="first").copy()
    return df


def merge_cluster_fields(full_df: pd.DataFrame, clustered_df: pd.DataFrame, cfg: InferConfig) -> pd.DataFrame:
    if clustered_df.empty:
        df = full_df.copy()
        for c, default in {
            "bucket_id": DEFAULT_BUCKET_ID,
            "cluster_id": DEFAULT_CLUSTER_ID,
            "dist_to_center": DEFAULT_DIST_TO_CENTER,
            "sample_role": DEFAULT_SAMPLE_ROLE,
            "is_sampled": DEFAULT_IS_SAMPLED,
        }.items():
            if c not in df.columns:
                df[c] = default
            df[c] = df[c].fillna(default) if hasattr(df[c], "fillna") else df[c]
        return df

    df = full_df.merge(clustered_df, on=["row_id", "column"], how="left", suffixes=("", "_from_cluster"))

    fields = [c for c in cfg.cluster_fields if c not in {"row_id", "column"}]
    for col in fields:
        cluster_col = f"{col}_from_cluster"
        if cluster_col in df.columns:
            if col not in df.columns:
                df[col] = df[cluster_col]
            else:
                cur = df[col]
                # full 字段为空、nan、空字符串时才用 cluster 字段回填
                mask = cur.isna() | (cur.astype(str).str.strip() == "")
                df[col] = cur.where(~mask, df[cluster_col])
            df = df.drop(columns=[cluster_col])

    return collapse_duplicate_columns(df)


# ============================================================
# 4. bucket 构造
# ============================================================

def build_rarity_bucket(row: pd.Series) -> str:
    vf = safe_float(row.get("value_frequency"), np.nan)
    vr = safe_float(row.get("value_frequency_rank"), np.nan)

    if not pd.isna(vr):
        if vr <= 3:
            return "very_common"
        if vr <= 10:
            return "common"

    if not pd.isna(vf):
        if vf <= 1:
            return "singleton"
        if vf <= 2:
            return "very_rare"
        if vf <= 5:
            return "rare"

    return "mid_frequency"


def build_neighbor_bucket(row: pd.Series) -> str:
    nmr = safe_float(row.get("neighbor_majority_ratio"), np.nan)
    majority_val = normalize_value(row.get("neighbor_majority_value"))
    current_val = normalize_value(row.get("value"))

    if pd.isna(nmr):
        return "no_neighbor_signal"

    same = majority_val == current_val

    if not same:
        if nmr >= 0.8:
            return "strong_support_other"
        if nmr >= 0.5:
            return "medium_support_other"
        return "weak_support_other"

    if nmr >= 0.8:
        return "strong_support_current"
    return "weak_support_current"


def build_time_window_bucket(row: pd.Series) -> str:
    cnt = safe_int(row.get("time_window_rule_count"), 0)
    if cnt <= 0:
        return "no_time_window_signal"
    if cnt == 1:
        return "single_time_window_signal"
    return "multi_time_window_signal"


def build_time_value_bucket(row: pd.Series) -> str:
    tv = safe_float(row.get("time_value_minutes"), np.nan)
    if pd.isna(tv):
        return "unparsed_time"
    hour = int(tv) // 60
    if 0 <= hour < 6:
        return "late_night"
    if 6 <= hour < 12:
        return "morning"
    if 12 <= hour < 18:
        return "afternoon"
    if 18 <= hour < 24:
        return "evening"
    return "out_of_day_range"


def build_numeric_window_bucket(row: pd.Series) -> str:
    cnt = safe_int(row.get("numeric_window_rule_count"), 0)
    if cnt <= 0:
        return "no_numeric_window_signal"
    if cnt == 1:
        return "single_numeric_window_signal"
    return "multi_numeric_window_signal"


def build_numeric_value_bucket(row: pd.Series) -> str:
    nv = safe_float(row.get("numeric_value"), np.nan)
    col = safe_str(row.get("column"), "")
    if pd.isna(nv):
        return "unparsed_numeric"

    if col == "abv":
        if nv < 0.04:
            return "abv_low"
        if nv < 0.07:
            return "abv_mid"
        return "abv_high"

    if col == "ibu":
        if nv < 20:
            return "ibu_low"
        if nv < 60:
            return "ibu_mid"
        return "ibu_high"

    if nv < 0:
        return "negative_numeric"
    if nv == 0:
        return "zero_numeric"
    if nv < 1:
        return "small_numeric"
    if nv < 100:
        return "medium_numeric"
    return "large_numeric"


def build_canonical_bucket(row: pd.Series) -> str:
    cnt = safe_int(row.get("canonical_rule_count"), 0)
    if cnt <= 0:
        return "no_canonical_signal"
    if cnt == 1:
        return "single_canonical_signal"
    return "multi_canonical_signal"


def build_date_value_bucket(row: pd.Series) -> str:
    dv = safe_str(row.get("date_value"), "")
    if dv == "":
        return "unparsed_date"
    try:
        year = int(dv[:4])
        if year < 1950:
            return "very_old"
        if year < 1980:
            return "old"
        if year < 2000:
            return "mid_old"
        if year < 2025:
            return "recent"
        return "future"
    except Exception:
        return "unparsed_date"


def build_language_bucket(row: pd.Series) -> str:
    lang = safe_str(row.get("language_canonical"), "").strip().lower()
    if not lang:
        return "unparsed_language"
    if lang == "english":
        return "english"
    return "non_english"


PRIMARY_ADULT_COLUMNS = {"sex", "relationship", "education"}
CONTEXT_ONLY_ADULT_COLUMNS = {
    "age", "workclass", "maritalstatus", "occupation", "race", "hoursperweek", "country", "income"
}
RELATIONSHIP_V2_RULE_TYPES = {
    "relationship_marital_spouse_conflict",
    "relationship_age_spouse_conflict",
    "relationship_sex_spouse_conflict",
    "relationship_context_dominant",
}
EDUCATION_V2_RULE_TYPES = {"education_age_extreme_conflict"}


def as_int01_value(x: Any, default: int = 0) -> int:
    try:
        if pd.isna(x) or x is None:
            return default
    except Exception:
        pass
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y"}:
        return 1
    if s in {"0", "false", "no", "n"}:
        return 0
    try:
        return int(float(x))
    except Exception:
        return default


def build_adult_v2_attribution_bucket(row: pd.Series) -> str:
    col = str(row.get("column", ""))
    rule = str(row.get("main_rule_type", ""))

    if as_int01_value(row.get("relationship_marital_conflict", 0)) == 1:
        return "relationship_marital_conflict"
    if as_int01_value(row.get("relationship_age_conflict", 0)) == 1:
        return "relationship_age_conflict"
    if as_int01_value(row.get("relationship_sex_conflict", 0)) == 1:
        return "relationship_sex_conflict"
    if as_int01_value(row.get("has_relationship_v2_rule", 0)) == 1 or rule in RELATIONSHIP_V2_RULE_TYPES:
        return "relationship_v2_rule"
    if as_int01_value(row.get("sex_value_invalid", 0)) == 1:
        return "sex_value_invalid"
    if as_int01_value(row.get("education_age_extreme_conflict", 0)) == 1 or rule in EDUCATION_V2_RULE_TYPES:
        return "education_age_extreme_conflict"
    if col in PRIMARY_ADULT_COLUMNS:
        return "primary_target_other"
    if col in CONTEXT_ONLY_ADULT_COLUMNS:
        return "context_only_other"
    return "unknown_attribution"


def build_domain_bucket(row: pd.Series) -> str:
    val = row.get("domain_valid")
    if val is None:
        return "domain_not_applicable"
    try:
        if pd.isna(val):
            return "domain_not_applicable"
    except Exception:
        pass
    s = str(val).strip().lower()
    if s == "":
        return "domain_not_applicable"
    if s in {"true", "1", "yes"}:
        return "domain_valid"
    if s in {"false", "0", "no"}:
        return "domain_invalid"
    return "domain_unknown"


def build_adult_consistency_bucket(row: pd.Series) -> str:
    flags = str(row.get("adult_evidence_flags", "") or "").strip()
    score = safe_float(row.get("adult_consistency_score"), 0.0)
    if flags and score >= 1.0:
        return "strong_adult_consistency_signal"
    if flags:
        return "adult_consistency_flag"
    if score > 0:
        return "weak_adult_consistency_signal"
    return "no_adult_consistency_signal"


def build_age_sampling_bucket(row: pd.Series) -> str:
    lo = safe_float(row.get("age_lower"), np.nan)
    hi = safe_float(row.get("age_upper"), np.nan)
    if pd.isna(lo):
        return "age_unparsed"
    if not pd.isna(hi) and hi < 18:
        return "minor"
    if lo <= 25:
        return "young_adult"
    if lo <= 50:
        return "working_age"
    return "older_adult"


def build_hours_sampling_bucket(row: pd.Series) -> str:
    lo = safe_float(row.get("hours_lower"), np.nan)
    hi = safe_float(row.get("hours_upper"), np.nan)
    if pd.isna(lo):
        return "hours_unparsed"
    if pd.isna(hi):
        hi = lo
    mid = (lo + hi) / 2.0
    if mid < 20:
        return "low_hours"
    if mid <= 40:
        return "standard_hours"
    if mid <= 60:
        return "high_hours"
    return "very_high_hours"


VALID_POSITIONS = {"Goalkeeper", "Defender", "Midfield", "Midfielder", "Forward", "Striker", "Winger"}
POSITION_CANONICAL_MAP = {"Midfielder": "Midfield"}
PLAYER_MIN_AGE_AT_SEASON = 15
PLAYER_MAX_AGE_AT_SEASON = 50
SOCCER_COLUMNS = {"name", "surname", "birthyear", "birthplace", "position", "team", "city", "stadium", "season", "manager"}
SOCCER_PRIMARY_TARGET_COLUMNS = set(SOCCER_COLUMNS)
SOCCER_CONTEXT_ONLY_COLUMNS = set()


def canonical_position(x: Any) -> Optional[str]:
    try:
        if x is None or pd.isna(x):
            return None
    except Exception:
        pass
    s = str(x).strip()
    if s in POSITION_CANONICAL_MAP:
        return POSITION_CANONICAL_MAP[s]
    if s in VALID_POSITIONS:
        return s
    return None


def build_soccer_consistency_bucket(row: pd.Series) -> str:
    flags = str(row.get("soccer_evidence_flags", "") or "").strip()
    score = safe_float(row.get("soccer_consistency_score"), 0.0)
    if flags and score >= 1.0:
        return "strong_soccer_consistency_signal"
    if flags:
        return "soccer_consistency_flag"
    if score > 0:
        return "weak_soccer_consistency_signal"
    return "no_soccer_consistency_signal"


def fill_one_bucket(df: pd.DataFrame, col: str, builder, default: str) -> pd.DataFrame:
    if col not in df.columns:
        df[col] = ""
    df[col] = df[col].fillna("").astype(str)
    need = df[col].eq("") | df[col].str.lower().isin({"nan", "none"})
    if need.any():
        df.loc[need, col] = df[need].apply(builder, axis=1)
    df[col] = df[col].fillna(default).replace("", default)
    return df


def fill_bucket_fields(df: pd.DataFrame, cfg: InferConfig) -> pd.DataFrame:
    df = df.copy()

    if "pattern_bucket" not in df.columns:
        df["pattern_bucket"] = DEFAULT_PATTERN_BUCKET
    df["pattern_bucket"] = df["pattern_bucket"].fillna(DEFAULT_PATTERN_BUCKET).replace("", DEFAULT_PATTERN_BUCKET)

    df = fill_one_bucket(df, "rarity_bucket", build_rarity_bucket, DEFAULT_RARITY_BUCKET)
    df = fill_one_bucket(df, "neighbor_bucket", build_neighbor_bucket, DEFAULT_NEIGHBOR_BUCKET)

    if cfg.dataset_kind == "flights":
        df = fill_one_bucket(df, "time_window_bucket", build_time_window_bucket, "unknown")
        df = fill_one_bucket(df, "time_value_bucket", build_time_value_bucket, "unknown")

    elif cfg.dataset_kind == "beers":
        df = fill_one_bucket(df, "numeric_window_bucket", build_numeric_window_bucket, "unknown")
        df = fill_one_bucket(df, "numeric_value_bucket", build_numeric_value_bucket, "unknown")

    elif cfg.dataset_kind == "rayyan":
        df = fill_one_bucket(df, "canonical_bucket", build_canonical_bucket, "unknown")
        df = fill_one_bucket(df, "date_value_bucket", build_date_value_bucket, "unknown")
        df = fill_one_bucket(df, "language_bucket", build_language_bucket, "unknown")

    elif cfg.dataset_kind == "adult":
        df = fill_one_bucket(df, "domain_bucket", build_domain_bucket, "unknown")
        df = fill_one_bucket(df, "adult_consistency_bucket", build_adult_consistency_bucket, "unknown")
        df = fill_one_bucket(df, "age_sampling_bucket", build_age_sampling_bucket, "unknown")
        df = fill_one_bucket(df, "hours_sampling_bucket", build_hours_sampling_bucket, "unknown")
        if "time_window_bucket" not in df.columns:
            df["time_window_bucket"] = "adult_not_applicable"
        if "time_value_bucket" not in df.columns:
            df["time_value_bucket"] = "adult_not_applicable"
        df["time_window_bucket"] = df["time_window_bucket"].fillna("").replace("", "adult_not_applicable")
        df["time_value_bucket"] = df["time_value_bucket"].fillna("").replace("", "adult_not_applicable")

        for col in [
            "target_is_primary_column", "target_is_context_only_column",
            "is_relationship_spouse_value", "relationship_marital_conflict",
            "relationship_age_conflict", "relationship_sex_conflict",
            "sex_value_invalid", "education_age_extreme_conflict",
            "has_relationship_v2_rule", "has_education_v2_rule",
        ]:
            if col not in df.columns:
                df[col] = 0
            df[col] = to_int_flag(df[col], 0, index=df.index)

        if "adult_v2_attribution_bucket" not in df.columns:
            df["adult_v2_attribution_bucket"] = ""
        df["adult_v2_attribution_bucket"] = df["adult_v2_attribution_bucket"].fillna("").astype(str)
        need = df["adult_v2_attribution_bucket"].eq("")
        if need.any():
            df.loc[need, "adult_v2_attribution_bucket"] = df[need].apply(build_adult_v2_attribution_bucket, axis=1)

        col_series = df["column"].astype(str)
        df["target_is_primary_column"] = np.where(col_series.isin(PRIMARY_ADULT_COLUMNS), 1, df["target_is_primary_column"].astype(int))
        df["target_is_context_only_column"] = np.where(col_series.isin(CONTEXT_ONLY_ADULT_COLUMNS), 1, df["target_is_context_only_column"].astype(int))
        df["has_relationship_v2_rule"] = np.where(df["main_rule_type"].astype(str).isin(RELATIONSHIP_V2_RULE_TYPES), 1, df["has_relationship_v2_rule"].astype(int))
        df["has_education_v2_rule"] = np.where(df["main_rule_type"].astype(str).isin(EDUCATION_V2_RULE_TYPES), 1, df["has_education_v2_rule"].astype(int))

    elif cfg.dataset_kind == "soccer":
        df = fill_one_bucket(df, "domain_bucket", build_domain_bucket, "unknown")
        df = fill_one_bucket(df, "soccer_consistency_bucket", build_soccer_consistency_bucket, "unknown")
        for c, default in {
            "adult_consistency_bucket": "soccer_not_applicable",
            "age_sampling_bucket": "soccer_not_applicable",
            "hours_sampling_bucket": "soccer_not_applicable",
            "time_window_bucket": "soccer_not_applicable",
            "time_value_bucket": "soccer_not_applicable",
        }.items():
            if c not in df.columns:
                df[c] = default
            df[c] = df[c].fillna("").replace("", default)

    # common sample fields
    for c, default, parser in [
        ("bucket_id", DEFAULT_BUCKET_ID, str),
        ("cluster_id", DEFAULT_CLUSTER_ID, None),
        ("dist_to_center", DEFAULT_DIST_TO_CENTER, None),
        ("sample_role", DEFAULT_SAMPLE_ROLE, str),
        ("is_sampled", DEFAULT_IS_SAMPLED, None),
    ]:
        if c not in df.columns:
            df[c] = default

    df["bucket_id"] = df["bucket_id"].fillna(DEFAULT_BUCKET_ID).replace("", DEFAULT_BUCKET_ID)
    df["cluster_id"] = to_numeric_series(df["cluster_id"], DEFAULT_CLUSTER_ID).astype(int)
    df["dist_to_center"] = to_numeric_series(df["dist_to_center"], DEFAULT_DIST_TO_CENTER)
    df["sample_role"] = df["sample_role"].fillna(DEFAULT_SAMPLE_ROLE).replace("", DEFAULT_SAMPLE_ROLE)
    df["is_sampled"] = to_numeric_series(df["is_sampled"], DEFAULT_IS_SAMPLED).astype(int)

    return df


# ============================================================
# 5. 对齐字段与派生特征
# ============================================================

def add_alignment_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    defaults = {
        "label": "",
        "label_binary": "",
        "label_source": DEFAULT_LABEL_SOURCE,
        "sample_weight": DEFAULT_SAMPLE_WEIGHT,
        "is_outside_candidate": DEFAULT_IS_OUTSIDE_CANDIDATE,
        "propagation_confidence": DEFAULT_PROPAGATION_CONFIDENCE,
        "error_type": "",
        "reason_short": "",
        "reason_detailed": "",
        "suggested_correct_value": "",
        "needs_human_review": False,
        "evidence_used": "",
    }
    for c, default in defaults.items():
        if c not in df.columns:
            df[c] = default
    return df


def infer_soccer_rule_type_sets_from_row(row: pd.Series) -> Dict[str, int]:
    rules = set()
    rules.update(parse_pipe_set(row.get("rule_types")))
    rules.update(parse_pipe_set(row.get("main_rule_type")))
    flags = parse_pipe_set(row.get("soccer_evidence_flags"))
    rules.update(flags)

    domain = {"position_domain"}
    fmt = {
        "name_pattern", "surname_pattern", "manager_name_pattern", "team_name_pattern",
        "birthplace_location_pattern", "city_location_pattern", "stadium_name_pattern",
        "pattern", "generic_regex", "rare_pattern"
    }
    numeric = {"birthyear_numeric_range", "season_numeric_range"}
    consistency = {
        "birthyear_season_age_consistency", "team_context_dominant",
        "team_season_manager_consistency_hint", "city_stadium_consistency_hint",
        "dominant_value_by_context", "dominant_value_by_context_pair"
    }
    fd = {"functional_dependency", "soft_functional_dependency"}
    return {
        "has_domain": int(len(rules & domain) > 0),
        "has_format": int(len(rules & fmt) > 0),
        "has_numeric": int(len(rules & numeric) > 0),
        "has_consistency": int(len(rules & consistency) > 0),
        "has_fd": int(len(rules & fd) > 0),
    }


def add_common_derived_features(df: pd.DataFrame, cfg: InferConfig) -> pd.DataFrame:
    df = df.copy()
    idx = df.index

    numeric_cols = [
        "strong_rule_count", "candidate_generation_rule_count", "fd_like_count",
        "context_rule_count", "global_rule_count", "rare_value_count",
        "typo_rule_count", "pattern_rule_count", "schema_rule_count",
        "conflict_score", "value_frequency", "value_frequency_rank",
        "neighbor_majority_ratio", "prior_error_probability", "posterior_error_probability",
    ]
    for c in numeric_cols:
        if c not in df.columns:
            df[c] = 0
        df[c] = to_numeric_series(df[c], 0.0, index=idx)

    strong_cnt = df["strong_rule_count"].astype(float)
    cand_cnt = df["candidate_generation_rule_count"].astype(float)
    total_cnt = strong_cnt + cand_cnt

    df["candidate_rule_ratio"] = np.where(total_cnt > 0, cand_cnt / (total_cnt + 1e-8), 0.0)
    df["has_neighbor_signal"] = (df["neighbor_majority_ratio"].astype(float) > 0).astype(int)
    df["high_neighbor_consistency"] = (df["neighbor_majority_ratio"].astype(float) >= HIGH_NEIGHBOR_CONSISTENCY_TH).astype(int)

    def _is_rare(row):
        vf = safe_float(row.get("value_frequency"), 0.0)
        vr = safe_float(row.get("value_frequency_rank"), np.nan)
        if cfg.dataset_kind == "adult" and str(row.get("column", "")) in {"country", "occupation", "workclass"}:
            if vf <= 1:
                return 1
            if not pd.isna(vr) and vr > 25:
                return 1
            return 0
        if cfg.dataset_kind == "soccer" and str(row.get("column", "")) in {"name", "surname", "team", "city", "stadium", "manager", "birthplace"}:
            if vf <= 1:
                return 1
            if not pd.isna(vr) and vr > 25:
                return 1
            return 0
        if vf <= RARE_VALUE_FREQ_TH:
            return 1
        if not pd.isna(vr) and vr > RARE_VALUE_RANK_TH:
            return 1
        return 0

    df["is_rare_value"] = df.apply(_is_rare, axis=1)

    value_series = normalize_text_series_for_compare(df["value"])
    neighbor_series = normalize_text_series_for_compare(df["neighbor_majority_value"])
    ratio = to_numeric_series(df["neighbor_majority_ratio"], 0.0, index=idx)

    df["is_equal_to_neighbor_majority"] = ((ratio > 0) & (value_series == neighbor_series)).astype(int)
    df["neighbor_disagree"] = ((ratio > 0) & (value_series != neighbor_series)).astype(int)
    df["neighbor_agreement_score"] = np.where(df["is_equal_to_neighbor_majority"] == 1, ratio, -ratio).astype(float)

    df["rule_total_count"] = df["strong_rule_count"] + df["candidate_generation_rule_count"]
    df["strong_rule_ratio"] = safe_div(df["strong_rule_count"], df["rule_total_count"], index=idx)
    df["context_rule_ratio"] = safe_div(df["context_rule_count"], df["rule_total_count"], index=idx)
    df["fd_like_ratio"] = safe_div(df["fd_like_count"], df["rule_total_count"], index=idx)
    df["global_rule_ratio"] = safe_div(df["global_rule_count"], df["rule_total_count"], index=idx)

    extra_strength_cols = []
    if cfg.dataset_kind == "flights":
        extra_strength_cols = ["time_window_rule_count"]
    elif cfg.dataset_kind == "beers":
        extra_strength_cols = ["numeric_window_rule_count"]
    elif cfg.dataset_kind == "rayyan":
        extra_strength_cols = ["canonical_rule_count"]

    for c in extra_strength_cols:
        if c not in df.columns:
            df[c] = 0
        df[c] = to_numeric_series(df[c], 0.0, index=idx)

    df["rule_strength_sum"] = (
        df["strong_rule_count"].astype(float)
        + df["fd_like_count"].astype(float)
        + df["context_rule_count"].astype(float)
        + df["global_rule_count"].astype(float)
        + df["rare_value_count"].astype(float)
        + df["typo_rule_count"].astype(float)
        + df["pattern_rule_count"].astype(float)
        + df["schema_rule_count"].astype(float)
        + sum(df[c].astype(float) for c in extra_strength_cols)
    )

    df["candidate_rule_dominant"] = np.where(
        df["candidate_generation_rule_count"].astype(float) > df["strong_rule_count"].astype(float),
        "yes",
        "no",
    )

    return df


def add_adult_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    idx = df.index

    adult_num_cols = [
        "adult_domain_rule_count", "adult_format_rule_count", "adult_consistency_rule_count",
        "age_lower", "age_upper", "hours_lower", "hours_upper", "education_order",
        "adult_profile_inconsistency_count", "adult_consistency_score", "time_window_rule_count",
        "target_is_primary_column", "target_is_context_only_column", "is_relationship_spouse_value",
        "relationship_marital_conflict", "relationship_age_conflict", "relationship_sex_conflict",
        "sex_value_invalid", "education_age_extreme_conflict", "has_relationship_v2_rule",
        "has_education_v2_rule",
    ]
    for c in adult_num_cols:
        if c not in df.columns:
            df[c] = 0
        df[c] = to_numeric_series(df[c], 0.0, index=idx)

    df["adult_rule_total_count"] = (
        df["adult_domain_rule_count"] + df["adult_format_rule_count"] + df["adult_consistency_rule_count"]
    )
    df["adult_domain_rule_ratio"] = safe_div(df["adult_domain_rule_count"], df["adult_rule_total_count"], index=idx)
    df["adult_format_rule_ratio"] = safe_div(df["adult_format_rule_count"], df["adult_rule_total_count"], index=idx)
    df["adult_consistency_rule_ratio"] = safe_div(df["adult_consistency_rule_count"], df["adult_rule_total_count"], index=idx)

    df["domain_invalid_flag"] = df["domain_bucket"].astype(str).eq("domain_invalid").astype(int)
    df["adult_consistency_flag"] = df["adult_consistency_bucket"].astype(str).isin([
        "strong_adult_consistency_signal", "adult_consistency_flag", "weak_adult_consistency_signal"
    ]).astype(int)
    df["high_adult_consistency_score"] = (df["adult_consistency_score"].astype(float) >= 1.0).astype(int)
    df["has_adult_domain_signal"] = (df["adult_domain_rule_count"] > 0).astype(int)
    df["has_adult_format_signal"] = (df["adult_format_rule_count"] > 0).astype(int)
    df["has_adult_consistency_signal"] = (df["adult_consistency_rule_count"] > 0).astype(int)

    df["age_span"] = np.maximum(0.0, df["age_upper"] - df["age_lower"])
    df.loc[df["age_upper"].isna() | df["age_lower"].isna(), "age_span"] = 0.0
    df["hours_span"] = np.maximum(0.0, df["hours_upper"] - df["hours_lower"])
    df.loc[df["hours_upper"].isna() | df["hours_lower"].isna(), "hours_span"] = 0.0

    df["is_minor_age_bucket"] = ((df["age_upper"] > 0) & (df["age_upper"] < 18)).astype(int)
    df["is_high_hours_bucket"] = (df["hours_upper"] > 60).astype(int)

    df["relationship_v2_conflict_count"] = (
        df["relationship_marital_conflict"].astype(int)
        + df["relationship_age_conflict"].astype(int)
        + df["relationship_sex_conflict"].astype(int)
        + df["has_relationship_v2_rule"].astype(int)
    )
    df["has_relationship_v2_signal"] = (df["relationship_v2_conflict_count"] > 0).astype(int)
    df["has_education_v2_signal"] = (
        (df["education_age_extreme_conflict"].astype(int) > 0)
        | (df["has_education_v2_rule"].astype(int) > 0)
    ).astype(int)

    df["is_context_only_weak_signal"] = (
        (df["target_is_context_only_column"].astype(int) == 1)
        & (df["has_adult_domain_signal"] == 0)
        & (df["has_adult_format_signal"] == 0)
        & (df["has_adult_consistency_signal"] == 0)
        & (df["schema_rule_count"].astype(float) <= 0)
        & (df["fd_like_count"].astype(float) <= 0)
        & (df["context_rule_count"].astype(float) <= 0)
    ).astype(int)

    df["adult_v2_signal_strength"] = (
        2.5 * df["has_relationship_v2_signal"].astype(float)
        + 2.0 * df["sex_value_invalid"].astype(float)
        + 1.5 * df["has_education_v2_signal"].astype(float)
        + 1.2 * df["has_adult_domain_signal"].astype(float)
        + 1.0 * df["has_adult_format_signal"].astype(float)
        + 1.0 * df["has_adult_consistency_signal"].astype(float)
        + 0.5 * df["target_is_primary_column"].astype(float)
        - 1.0 * df["is_context_only_weak_signal"].astype(float)
    )
    df["adult_v2_primary_target_signal"] = (
        (df["target_is_primary_column"].astype(int) == 1)
        & (
            (df["has_relationship_v2_signal"].astype(int) == 1)
            | (df["sex_value_invalid"].astype(int) == 1)
            | (df["has_education_v2_signal"].astype(int) == 1)
            | (df["has_adult_domain_signal"].astype(int) == 1)
            | (df["has_adult_format_signal"].astype(int) == 1)
            | (df["has_adult_consistency_signal"].astype(int) == 1)
        )
    ).astype(int)

    df["rule_strength_sum"] = (
        df["rule_strength_sum"].astype(float)
        + 1.5 * df["adult_domain_rule_count"].astype(float)
        + 1.2 * df["adult_format_rule_count"].astype(float)
        + 1.2 * df["adult_consistency_rule_count"].astype(float)
        + 2.0 * df["has_relationship_v2_signal"].astype(float)
        + 1.8 * df["sex_value_invalid"].astype(float)
        + 1.5 * df["has_education_v2_signal"].astype(float)
    )

    return df


def add_soccer_value_features_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    col = df["column"].astype(str)

    if "year_value" not in df.columns:
        df["year_value"] = np.nan
    year = to_numeric_series(df["year_value"], np.nan, index=df.index)

    # 补 birthyear / season 的 year_value
    parsed_value = df["value"].map(parse_int_like)
    year = np.where(col.isin({"birthyear", "season"}) & pd.Series(year, index=df.index).isna(), parsed_value, year)
    df["year_value"] = pd.to_numeric(pd.Series(year, index=df.index), errors="coerce")

    if "canonical_position" not in df.columns:
        df["canonical_position"] = ""
    pos_mask = col.eq("position")
    df.loc[pos_mask & (df["canonical_position"].fillna("").astype(str) == ""), "canonical_position"] = (
        df.loc[pos_mask, "value"].map(canonical_position).fillna("")
    )

    return df


def add_soccer_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_soccer_value_features_if_missing(df.copy())
    idx = df.index

    soccer_num_cols = [
        "evidence_only_fd_count", "soccer_domain_rule_count", "soccer_format_rule_count",
        "soccer_numeric_rule_count", "soccer_consistency_rule_count",
        "soccer_profile_inconsistency_count", "soccer_consistency_score",
        "year_value", "row_birthyear_int", "row_season_int", "row_age_at_season",
        "target_is_primary_column", "target_is_context_only_column",
        "birthyear_value_invalid", "season_value_invalid",
        "position_value_invalid", "position_needs_canonicalization",
        "birthyear_season_age_conflict", "team_context_conflict",
        "format_rule_signal", "fd_like_signal",
        "adult_domain_rule_count", "adult_format_rule_count", "adult_consistency_rule_count",
        "adult_profile_inconsistency_count", "adult_consistency_score",
    ]
    for c in soccer_num_cols:
        if c not in df.columns:
            df[c] = 0
        df[c] = to_numeric_series(df[c], 0.0, index=idx)

    col = df["column"].astype(str)
    rule = df["main_rule_type"].astype(str)

    df["target_is_primary_column"] = col.isin(SOCCER_PRIMARY_TARGET_COLUMNS).astype(int)
    df["target_is_context_only_column"] = col.isin(SOCCER_CONTEXT_ONLY_COLUMNS).astype(int)

    # 根据 main_rule_type 兜底补信号
    df["birthyear_value_invalid"] = np.where(rule.eq("birthyear_numeric_range"), 1, df["birthyear_value_invalid"]).astype(int)
    df["season_value_invalid"] = np.where(rule.eq("season_numeric_range"), 1, df["season_value_invalid"]).astype(int)
    df["position_value_invalid"] = np.where(rule.eq("position_domain"), 1, df["position_value_invalid"]).astype(int)
    df["birthyear_season_age_conflict"] = np.where(rule.eq("birthyear_season_age_consistency"), 1, df["birthyear_season_age_conflict"]).astype(int)

    ctx_rules = {
        "team_context_dominant", "team_season_manager_consistency_hint",
        "city_stadium_consistency_hint", "dominant_value_by_context", "dominant_value_by_context_pair"
    }
    format_rules = {
        "name_pattern", "surname_pattern", "manager_name_pattern",
        "team_name_pattern", "birthplace_location_pattern", "city_location_pattern",
        "stadium_name_pattern", "pattern", "generic_regex", "rare_pattern"
    }
    df["team_context_conflict"] = np.where(rule.isin(ctx_rules), 1, df["team_context_conflict"]).astype(int)
    df["format_rule_signal"] = np.where(rule.isin(format_rules), 1, df["format_rule_signal"]).astype(int)

    soft_fd_mask = rule.eq("soft_functional_dependency")
    df.loc[soft_fd_mask, "evidence_only_fd_count"] = np.maximum(
        df.loc[soft_fd_mask, "evidence_only_fd_count"].astype(float), 1.0
    )
    df["fd_like_signal"] = np.where(rule.eq("functional_dependency"), 1, df["fd_like_signal"]).astype(int)
    df.loc[soft_fd_mask, "fd_like_signal"] = 0

    inferred = df.apply(infer_soccer_rule_type_sets_from_row, axis=1, result_type="expand")
    df["has_soccer_domain_signal"] = ((df["soccer_domain_rule_count"] > 0) | inferred["has_domain"].astype(bool)).astype(int)
    df["has_soccer_format_signal"] = ((df["soccer_format_rule_count"] > 0) | inferred["has_format"].astype(bool) | (df["format_rule_signal"] > 0)).astype(int)
    df["has_soccer_numeric_signal"] = ((df["soccer_numeric_rule_count"] > 0) | inferred["has_numeric"].astype(bool) | (df["birthyear_value_invalid"] > 0) | (df["season_value_invalid"] > 0)).astype(int)
    df["has_soccer_consistency_signal"] = ((df["soccer_consistency_rule_count"] > 0) | inferred["has_consistency"].astype(bool) | (df["birthyear_season_age_conflict"] > 0)).astype(int)
    df["has_fd_like_signal"] = ((df["fd_like_count"] > 0) | inferred["has_fd"].astype(bool) | (df["fd_like_signal"] > 0)).astype(int)
    df.loc[soft_fd_mask, "has_fd_like_signal"] = 0

    df["soccer_rule_total_count"] = (
        df["soccer_domain_rule_count"] + df["soccer_format_rule_count"]
        + df["soccer_numeric_rule_count"] + df["soccer_consistency_rule_count"]
    )
    df["soccer_domain_rule_ratio"] = safe_div(df["soccer_domain_rule_count"], df["soccer_rule_total_count"], index=idx)
    df["soccer_format_rule_ratio"] = safe_div(df["soccer_format_rule_count"], df["soccer_rule_total_count"], index=idx)
    df["soccer_numeric_rule_ratio"] = safe_div(df["soccer_numeric_rule_count"], df["soccer_rule_total_count"], index=idx)
    df["soccer_consistency_rule_ratio"] = safe_div(df["soccer_consistency_rule_count"], df["soccer_rule_total_count"], index=idx)

    # 兼容 adult 字段
    df["adult_rule_total_count"] = (
        df["adult_domain_rule_count"] + df["adult_format_rule_count"] + df["adult_consistency_rule_count"]
    )
    df["adult_domain_rule_ratio"] = safe_div(df["adult_domain_rule_count"], df["adult_rule_total_count"], index=idx)
    df["adult_format_rule_ratio"] = safe_div(df["adult_format_rule_count"], df["adult_rule_total_count"], index=idx)
    df["adult_consistency_rule_ratio"] = safe_div(df["adult_consistency_rule_count"], df["adult_rule_total_count"], index=idx)

    domain_txt = df.get("domain_valid", pd.Series([np.nan] * len(df), index=idx)).astype(str).str.lower()
    df["domain_invalid_flag"] = domain_txt.isin(["false", "0"]).astype(int)
    df["soccer_consistency_flag"] = ((df["soccer_consistency_rule_count"] > 0) | (df["soccer_consistency_score"] > 0)).astype(int)
    df["high_soccer_consistency_score"] = (df["soccer_consistency_score"] >= 1.0).astype(int)

    df["has_adult_domain_signal"] = df["has_soccer_domain_signal"]
    df["has_adult_format_signal"] = df["has_soccer_format_signal"]
    df["has_adult_consistency_signal"] = df["has_soccer_consistency_signal"]
    df["adult_consistency_flag"] = df["soccer_consistency_flag"]
    df["high_adult_consistency_score"] = df["high_soccer_consistency_score"]

    df["position_invalid_or_normalization"] = (
        (df["position_value_invalid"].astype(int) == 1)
        | (df["position_needs_canonicalization"].astype(int) == 1)
    ).astype(int)
    df["has_birthyear_or_season_signal"] = (
        (df["birthyear_value_invalid"].astype(int) == 1)
        | (df["season_value_invalid"].astype(int) == 1)
        | (df["birthyear_season_age_conflict"].astype(int) == 1)
        | (df["has_soccer_numeric_signal"].astype(int) == 1)
    ).astype(int)
    df["has_team_context_signal"] = (
        (df["team_context_conflict"].astype(int) == 1)
        | (df["context_rule_count"].astype(float) > 0)
        | rule.isin(ctx_rules)
    ).astype(int)

    age = to_numeric_series(df["row_age_at_season"], np.nan, index=idx)
    df["birthyear_season_gap"] = age
    df["is_young_player_age"] = (age.notna() & (age < PLAYER_MIN_AGE_AT_SEASON)).astype(int)
    df["is_old_player_age"] = (age.notna() & (age > PLAYER_MAX_AGE_AT_SEASON)).astype(int)

    # adult 旧字段兜底
    for c, val in {
        "age_span": np.nan,
        "hours_span": np.nan,
        "is_minor_age_bucket": 0,
        "is_high_hours_bucket": 0,
        "relationship_v2_conflict_count": 0,
        "has_relationship_v2_signal": 0,
        "has_education_v2_signal": 0,
        "adult_v2_signal_strength": 0,
        "adult_v2_primary_target_signal": 0,
    }.items():
        if c not in df.columns:
            df[c] = val

    neigh_ratio = to_numeric_series(df["neighbor_majority_ratio"], 0.0, index=idx)
    df["rule_strength_sum"] = (
        2.0 * df["has_soccer_domain_signal"]
        + 1.6 * df["has_soccer_numeric_signal"]
        + 1.3 * df["has_soccer_format_signal"]
        + 1.4 * df["has_soccer_consistency_signal"]
        + 0.5 * df["has_fd_like_signal"]
        + 0.7 * df["neighbor_disagree"] * neigh_ratio
        + 0.4 * df["is_rare_value"]
    ).round(6)
    df["candidate_rule_dominant"] = np.where(df["candidate_rule_ratio"].astype(float) >= 0.5, 1, 0)

    df["soccer_signal_strength"] = (
        df["rule_strength_sum"].astype(float)
        + to_numeric_series(df["posterior_error_probability"], 0.0, index=idx).clip(0, 1)
        + 0.5 * df["target_is_primary_column"].astype(float)
    ).round(6)

    df["soccer_primary_target_signal"] = (
        (df["target_is_primary_column"].astype(int) == 1)
        & (
            (df["has_soccer_domain_signal"].astype(int) == 1)
            | (df["has_soccer_numeric_signal"].astype(int) == 1)
            | (df["has_soccer_format_signal"].astype(int) == 1)
            | (df["has_soccer_consistency_signal"].astype(int) == 1)
            | (df["neighbor_disagree"].astype(int) == 1)
            | (df["has_fd_like_signal"].astype(int) == 1)
        )
    ).astype(int)

    df["is_context_only_weak_signal"] = (
        (df["target_is_context_only_column"].astype(int) == 1)
        & (df["has_soccer_domain_signal"].astype(int) == 0)
        & (df["has_soccer_numeric_signal"].astype(int) == 0)
        & (df["has_soccer_format_signal"].astype(int) == 0)
        & (df["has_soccer_consistency_signal"].astype(int) == 0)
    ).astype(int)

    if "soccer_attribution_bucket" not in df.columns:
        df["soccer_attribution_bucket"] = "primary_target_other"
    df["soccer_attribution_bucket"] = df["soccer_attribution_bucket"].fillna("").replace("", "primary_target_other")
    df.loc[df["birthyear_value_invalid"].astype(int).eq(1), "soccer_attribution_bucket"] = "birthyear_invalid"
    df.loc[df["season_value_invalid"].astype(int).eq(1), "soccer_attribution_bucket"] = "season_invalid"
    df.loc[df["position_invalid_or_normalization"].astype(int).eq(1), "soccer_attribution_bucket"] = "position_invalid_or_normalization"
    df.loc[
        (df["birthyear_season_age_conflict"].astype(int).eq(1))
        & (~df["soccer_attribution_bucket"].isin(["birthyear_invalid", "season_invalid"])),
        "soccer_attribution_bucket",
    ] = "birthyear_season_age_conflict"
    df.loc[
        (df["has_team_context_signal"].astype(int).eq(1))
        & (df["soccer_attribution_bucket"].eq("primary_target_other")),
        "soccer_attribution_bucket",
    ] = "team_context_conflict"
    df.loc[
        (df["has_fd_like_signal"].astype(int).eq(1))
        & (df["soccer_attribution_bucket"].eq("primary_target_other")),
        "soccer_attribution_bucket",
    ] = "fd_like_conflict"
    df.loc[df["target_is_context_only_column"].astype(int).eq(1), "soccer_attribution_bucket"] = "context_only_other"

    # 兼容 adult 旧字段
    df["adult_profile_inconsistency_count"] = df["soccer_profile_inconsistency_count"]
    df["adult_consistency_score"] = df["soccer_consistency_score"]
    df["adult_consistency_flag"] = df["soccer_consistency_flag"]
    df["adult_v2_attribution_bucket"] = df["soccer_attribution_bucket"]

    return df


def add_derived_features(df: pd.DataFrame, cfg: InferConfig) -> pd.DataFrame:
    df = add_common_derived_features(df, cfg)
    if cfg.dataset_kind == "adult":
        df = add_adult_derived_features(df)
    elif cfg.dataset_kind == "soccer":
        df = add_soccer_derived_features(df)
    return df


def reorder_columns(df: pd.DataFrame, cfg: InferConfig) -> pd.DataFrame:
    preferred = []
    for c in cfg.preferred_cols:
        if c not in preferred:
            preferred.append(c)
    exist = [c for c in preferred if c in df.columns]
    extra = [c for c in df.columns if c not in exist]
    return df[exist + extra].copy()


# ============================================================
# 6. 主流程
# ============================================================

def run_for_dataset(cfg: InferConfig, args):
    input_full = args.input_full_candidate_csv or cfg.input_full_candidate_csv
    input_clustered = args.input_clustered_csv or cfg.input_clustered_csv
    output_csv = args.output_csv or cfg.output_infer_ready_csv

    print("[1/5] 读取全候选上下文文件...")
    full_df = load_full_candidate_csv(input_full, cfg)
    print(f"  full candidate rows = {len(full_df)}, cols = {len(full_df.columns)}")

    print("[2/5] 读取聚类采样结果文件...")
    if args.disable_cluster_merge:
        clustered_df = pd.DataFrame(columns=["row_id", "column"])
        print("  cluster merge disabled.")
    else:
        clustered_df = load_clustered_csv(input_clustered, cfg, allow_missing_cluster_file=args.allow_missing_cluster_file)
        print(f"  clustered rows = {len(clustered_df)}, cols = {len(clustered_df.columns)}")

    print("[3/5] merge 聚类字段...")
    df = merge_cluster_fields(full_df, clustered_df, cfg)

    print("[4/5] 补 bucket / 对齐列 / 派生特征...")
    df = fill_bucket_fields(df, cfg)
    df = add_alignment_columns(df)
    if not args.disable_derived_features:
        df = add_derived_features(df, cfg)

    print("[5/5] 输出推理对齐文件...")
    df = reorder_columns(df, cfg)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"[OK] 推理对齐文件已保存: {output_csv}")
    print(f"[OK] 行数: {len(df)}")
    print(f"[OK] 列数: {len(df.columns)}")

    print("\n===== 预览前5行 =====")
    preview_cols = [c for c in ["row_id", "column", "value", "main_rule_type", "label_source"] if c in df.columns]
    print(df[preview_cols].head().to_string(index=False))

    print("\n===== 列分布 Top 20 =====")
    if "column" in df.columns:
        print(df["column"].value_counts().head(20).to_string())


def run_one_dataset(dataset: str, args):
    cfg = CONFIGS[dataset]
    old_cwd = None
    if args.cwd:
        os.makedirs(args.cwd, exist_ok=True)
        old_cwd = os.getcwd()
        os.chdir(args.cwd)
    try:
        run_for_dataset(cfg, args)
    finally:
        if old_cwd is not None:
            os.chdir(old_cwd)


def parse_args():
    parser = argparse.ArgumentParser(description="Unified infer-ready candidate feature builder.")
    parser.add_argument("--dataset", required=True, choices=sorted(list(CONFIGS.keys()) + ["all"]))

    parser.add_argument("--input_full_candidate_csv", default=None)
    parser.add_argument("--input_clustered_csv", default=None)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--cwd", default=None)

    parser.add_argument("--disable-cluster-merge", action="store_true")
    parser.add_argument("--disable-derived-features", action="store_true")
    parser.add_argument("--allow-missing-cluster-file", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.dataset == "all":
        if any([args.input_full_candidate_csv, args.input_clustered_csv, args.output_csv]):
            raise ValueError("--dataset all 不能和输入/输出路径覆盖参数一起使用。")
        for ds in ["hospital", "flights", "beers", "rayyan", "adult", "soccer"]:
            print("\n" + "=" * 80)
            print(f"[DATASET] {ds}")
            print("=" * 80)
            run_one_dataset(ds, args)
    else:
        run_one_dataset(args.dataset, args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unified_infer_mlp.py

统一版推理脚本，覆盖：
hospital / flights / beers / rayyan / adult / soccer

统一流程：
1. 读取 candidate_infer_ready_*.csv
2. 补齐 derived features
3. 加载 stage1 detector
4. 可选加载 stage2 detector
5. 使用数据集专属 column threshold / conditional threshold
6. 可选使用 evidence reliability gate（adult / soccer）
7. 可选使用 FP verifier
8. 可选使用数据集专属 post-gate / post-rules
9. 输出：
   - inference_all_predictions.csv
   - inference_hard_cases.csv
   - inference_high_risk.csv
   - hard_cases_for_llm.jsonl

默认输入输出保持六份原脚本风格。
脚本不读取 clean.csv / ground truth / evaluation_summary。
"""

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 1. 通用工具
# ============================================================

def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def safe_num_series(df: pd.DataFrame, col: str, default=0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)
    return pd.Series(np.full(len(df), default), index=df.index)


def safe_str_series(df: pd.DataFrame, col: str, default="missing") -> pd.Series:
    if col in df.columns:
        return df[col].fillna(default).astype(str)
    return pd.Series([default] * len(df), index=df.index)


def normalize_text_series_for_compare(s: pd.Series) -> pd.Series:
    return s.fillna("__NULL__").astype(str).str.strip()


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


def is_missing_like_series(s: pd.Series) -> pd.Series:
    return s.isna() | s.astype(str).str.strip().str.lower().isin({
        "", "nan", "none", "null", "na", "n/a", "missing", "__null__"
    })


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return str(obj)


# ============================================================
# 2. 配置
# ============================================================

@dataclass
class InferConfig:
    dataset: str
    input_csv: str
    model_dir: str
    output_dir: str

    use_stage2: bool = True
    use_column_thresholds: bool = True
    use_neighbor_adjustment: bool = True

    use_fp_verifier: bool = True
    verifier_only_on_detector_error: bool = True

    verifier_threshold_mode: str = "manual"   # manual / metrics / max_with_min
    manual_verifier_threshold: float = 0.50
    min_verifier_threshold: float = 0.50
    default_verifier_threshold: float = 0.50
    print_verifier_debug: bool = True

    column_thresholds: Dict[str, float] = field(default_factory=dict)
    cond_lower_threshold: Dict[str, float] = field(default_factory=dict)
    group_verifier_threshold_by_column: Dict[str, float] = field(default_factory=dict)
    group_verifier_threshold_by_rule: Dict[str, float] = field(default_factory=dict)

    hard_case_margin: float = 0.05
    hard_case_low: float = 0.45
    hard_case_high: float = 0.55
    high_risk_threshold: float = 0.85

    neighbor_support_current_ratio: float = 0.95
    neighbor_support_current_subtract: float = 0.05

    fp_risk_columns: Set[str] = field(default_factory=set)

    # evidence reliability gate
    use_evidence_reliability_gate: bool = False
    llm_labeled_csv: Optional[str] = None
    reliability_min_group_size: int = 5
    reliability_smooth_alpha: float = 1.0
    reliability_smooth_beta: float = 1.0
    reliability_low_precision: float = 0.30
    reliability_mid_precision: float = 0.55
    reliability_high_precision: float = 0.80
    reliability_delta_low: float = 0.20
    reliability_delta_mid: float = 0.08
    reliability_delta_high: float = -0.06
    reliability_delta_unknown_weak: float = 0.12
    weak_only_min_threshold: float = 0.68
    soft_fd_only_min_threshold: float = 0.70
    context_dominant_only_min_threshold: float = 0.72
    context_only_weak_min_threshold: float = 0.88
    weak_rule_types_for_gate: Set[str] = field(default_factory=set)
    reliability_group_keys: List[Tuple[str, ...]] = field(default_factory=list)

    # dataset-specific options
    bypass_verifier_rules: Set[Tuple[str, str]] = field(default_factory=set)
    bypass_verifier_date_rules: Set[str] = field(default_factory=set)
    recover_date_rules: Set[str] = field(default_factory=set)

    use_soccer_precision_post_gate: bool = False
    soccer_team_missing_only_reject: bool = False
    soccer_team_pattern_rule_reject: bool = True
    soccer_low_precision_rule_reject: bool = True
    soccer_reject_rule_types_strict: Set[str] = field(default_factory=set)
    soccer_reject_rule_types_soft: Set[str] = field(default_factory=set)
    surname_pattern_keep_min_prob: float = 0.995
    surname_pattern_keep_min_posterior: float = 0.95
    stadium_missing_only_min_prob: float = 0.995
    birthyear_missing_only_min_prob: float = 0.990
    surname_missing_only_min_prob: float = 0.900
    entity_weak_only_min_prob: float = 0.950
    text_missing_only_min_prob: float = 0.900


def build_configs() -> Dict[str, InferConfig]:
    cfgs: Dict[str, InferConfig] = {}

    cfgs["hospital"] = InferConfig(
        dataset="hospital",
        input_csv="candidate_infer_ready_v2.csv",
        model_dir="two_stage_model_output_v4_with_verifier",
        output_dir="two_stage_inference_output_v4_with_verifier",
        verifier_threshold_mode="manual",
        column_thresholds={
            "Sample": 0.78,
            "EmergencyService": 0.88,
            "ZipCode": 0.84,
            "HospitalName": 0.84,
        },
        cond_lower_threshold={
            "Sample": 0.68,
            "ZipCode": 0.74,
            "EmergencyService": 0.77,
            "HospitalName": 0.74,
        },
        group_verifier_threshold_by_column={
            "ProviderNumber": 0.64,
            "PhoneNumber": 0.62,
            "Score": 0.60,
            "Stateavg": 0.60,
            "Address1": 0.60,
            "Sample": 0.46,
            "MeasureName": 0.62,
            "MeasureCode": 0.62,
            "Condition": 0.61,
        },
        group_verifier_threshold_by_rule={
            "rare_value": 0.58,
            "rare_pattern": 0.60,
            "functional_dependency": 0.55,
            "soft_functional_dependency": 0.56,
        },
        fp_risk_columns={
            "EmergencyService", "Sample", "ZipCode", "HospitalName",
            "ProviderNumber", "PhoneNumber", "Score", "Stateavg",
            "Address1", "MeasureName", "MeasureCode", "Condition",
        },
    )

    cfgs["flights"] = InferConfig(
        dataset="flights",
        input_csv="candidate_infer_ready_flights.csv",
        model_dir="two_stage_model_output_flights_v4_with_verifier",
        output_dir="two_stage_inference_output_flights_v4_with_verifier",
        verifier_threshold_mode="manual",
        column_thresholds={
            # 保留原 flights 脚本里的阈值字典；若列不存在则不会生效。
            "Sample": 0.78,
            "EmergencyService": 0.88,
            "ZipCode": 0.84,
            "HospitalName": 0.84,
        },
        cond_lower_threshold={
            "Sample": 0.68,
            "ZipCode": 0.74,
            "EmergencyService": 0.77,
            "HospitalName": 0.74,
        },
        group_verifier_threshold_by_column={
            "ProviderNumber": 0.64,
            "PhoneNumber": 0.62,
            "Score": 0.60,
            "Stateavg": 0.60,
            "Address1": 0.60,
            "Sample": 0.46,
            "MeasureName": 0.62,
            "MeasureCode": 0.62,
            "Condition": 0.61,
        },
        group_verifier_threshold_by_rule={
            "rare_value": 0.58,
            "rare_pattern": 0.60,
            "functional_dependency": 0.55,
            "soft_functional_dependency": 0.56,
        },
        fp_risk_columns={"sched_dep_time", "act_dep_time", "sched_arr_time", "act_arr_time"},
    )

    cfgs["beers"] = InferConfig(
        dataset="beers",
        input_csv="candidate_infer_ready_beers.csv",
        model_dir="two_stage_model_output_beers_v4_with_verifier",
        output_dir="two_stage_inference_output_beers_v4_with_verifier",
        use_fp_verifier=False,
        verifier_threshold_mode="manual",
        column_thresholds={},
        cond_lower_threshold={
            "abv": 0.46,
            "ibu": 0.48,
            "ounces": 0.47,
            "style": 0.50,
            "brewery_name": 0.50,
            "city": 0.50,
            "state": 0.52,
        },
        group_verifier_threshold_by_column={
            "abv": 0.54,
            "ibu": 0.54,
            "ounces": 0.53,
            "style": 0.56,
            "brewery_name": 0.56,
            "city": 0.56,
            "state": 0.58,
        },
        group_verifier_threshold_by_rule={
            "functional_dependency": 0.56,
            "soft_functional_dependency": 0.54,
            "dominant_value_by_context": 0.55,
            "dominant_value_by_context_pair": 0.55,
            "numeric_range_window": 0.54,
            "state_abbr_format": 0.58,
            "abv_numeric_format": 0.56,
            "ibu_numeric_or_na_format": 0.56,
            "ounces_unit_format": 0.55,
            "ounces_loose_pattern": 0.54,
            "ounces_canonical_format": 0.38,
            "abv_canonical_format": 0.50,
            "city_state_suffix_pollution": 0.52,
        },
        fp_risk_columns={"abv", "ibu", "ounces", "state", "style", "brewery_name", "city"},
        bypass_verifier_rules={("ounces", "ounces_canonical_format")},
    )

    cfgs["rayyan"] = InferConfig(
        dataset="rayyan",
        input_csv="candidate_infer_ready_rayyan.csv",
        model_dir="two_stage_model_output_rayyan_v4_with_verifier",
        output_dir="two_stage_inference_output_rayyan_v4_with_verifier",
        verifier_threshold_mode="manual",
        column_thresholds={
            "article_language": 0.72,
            "journal_issn": 0.76,
            "article_jcreated_at": 0.74,
            "author_list": 0.74,
        },
        cond_lower_threshold={
            "article_language": 0.62,
            "journal_issn": 0.68,
            "article_jcreated_at": 0.62,
            "author_list": 0.66,
        },
        group_verifier_threshold_by_column={
            "journal_title": 0.60,
            "jounral_abbreviation": 0.60,
            "journal_issn": 0.58,
            "article_language": 0.52,
            "article_jcreated_at": 0.56,
            "article_pagination": 0.58,
            "author_list": 0.56,
        },
        group_verifier_threshold_by_rule={
            "rare_value": 0.58,
            "rare_pattern": 0.60,
            "functional_dependency": 0.55,
            "soft_functional_dependency": 0.56,
            "language_canonical_format": 0.50,
            "issn_canonical_format": 0.52,
            "date_like_format": 0.54,
            "date_component_permutation_suspicious": 0.46,
            "date_ymd_swap_suspicious": 0.40,
            "volume_issue_numeric_format": 0.54,
        },
        fp_risk_columns={
            "journal_title", "jounral_abbreviation", "journal_issn",
            "article_language", "article_jcreated_at", "author_list",
        },
        bypass_verifier_date_rules={"date_component_permutation_suspicious", "date_ymd_swap_suspicious"},
        recover_date_rules={
            "date_component_permutation_suspicious",
            "date_ymd_swap_suspicious",
            "date_like_format",
            "functional_dependency",
            "soft_functional_dependency",
        },
    )

    adult_weak = {
        "rare_value", "rare_pattern", "global_dominant_value",
        "functional_dependency", "soft_functional_dependency",
        "dominant_value_by_context", "dominant_value_by_context_pair",
        "relationship_context_dominant",
    }
    cfgs["adult"] = InferConfig(
        dataset="adult",
        input_csv="candidate_infer_ready_adult.csv",
        model_dir="two_stage_model_output_adult_v4_with_verifier",
        output_dir="two_stage_inference_output_adult_v4_with_verifier",
        verifier_threshold_mode="metrics",
        column_thresholds={
            "relationship": 0.48,
            "sex": 0.46,
            "education": 0.58,
            "age": 0.80,
            "workclass": 0.82,
            "maritalstatus": 0.78,
            "occupation": 0.82,
            "race": 0.82,
            "hoursperweek": 0.78,
            "country": 0.84,
            "income": 0.78,
        },
        cond_lower_threshold={
            "relationship": 0.38,
            "sex": 0.38,
            "education": 0.48,
            "age": 0.64,
            "hoursperweek": 0.62,
            "income": 0.62,
            "maritalstatus": 0.66,
            "workclass": 0.68,
            "occupation": 0.70,
            "country": 0.72,
            "race": 0.70,
        },
        group_verifier_threshold_by_column={
            "relationship": 0.46,
            "sex": 0.44,
            "education": 0.58,
            "age": 0.72,
            "workclass": 0.74,
            "maritalstatus": 0.70,
            "occupation": 0.74,
            "race": 0.74,
            "hoursperweek": 0.70,
            "country": 0.76,
            "income": 0.70,
        },
        group_verifier_threshold_by_rule={
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
            "relationship_marital_spouse_conflict": 0.42,
            "relationship_age_spouse_conflict": 0.44,
            "relationship_sex_spouse_conflict": 0.44,
            "relationship_context_dominant": 0.48,
            "education_age_extreme_conflict": 0.52,
        },
        hard_case_low=0.42,
        hard_case_high=0.58,
        fp_risk_columns={"country", "occupation", "workclass", "education", "maritalstatus", "relationship"},
        use_evidence_reliability_gate=True,
        llm_labeled_csv="candidate_sampled_labeled_adult.csv",
        weak_rule_types_for_gate=adult_weak,
        reliability_group_keys=[
            ("adult_v2_attribution_bucket",),
            ("main_rule_type",),
            ("column", "main_rule_type"),
            ("column", "adult_v2_attribution_bucket"),
        ],
    )

    soccer_weak = {
        "rare_value", "rare_pattern", "global_dominant_value",
        "functional_dependency", "soft_functional_dependency",
        "dominant_value_by_context", "dominant_value_by_context_pair",
        "team_context_dominant", "team_season_manager_consistency_hint",
        "city_stadium_consistency_hint",
    }
    cfgs["soccer"] = InferConfig(
        dataset="soccer",
        input_csv="candidate_infer_ready_soccer.csv",
        model_dir="two_stage_model_output_soccer_v4_with_verifier",
        output_dir="two_stage_inference_output_soccer_v4_with_verifier",
        verifier_threshold_mode="metrics",
        column_thresholds={
            "birthyear": 0.48,
            "season": 0.50,
            "position": 0.48,
            "team": 0.62,
            "city": 0.62,
            "stadium": 0.62,
            "manager": 0.64,
            "birthplace": 0.66,
            "name": 0.64,
            "surname": 0.64,
        },
        cond_lower_threshold={
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
        },
        group_verifier_threshold_by_column={
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
        },
        group_verifier_threshold_by_rule={
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
        },
        hard_case_low=0.42,
        hard_case_high=0.58,
        fp_risk_columns={"name", "surname", "team", "city", "stadium", "manager", "birthplace"},
        use_evidence_reliability_gate=True,
        llm_labeled_csv="candidate_sampled_labeled_soccer.csv",
        weak_rule_types_for_gate=soccer_weak,
        reliability_group_keys=[
            ("soccer_attribution_bucket",),
            ("main_rule_type",),
            ("column", "main_rule_type"),
            ("column", "soccer_attribution_bucket"),
        ],
        use_soccer_precision_post_gate=True,
        soccer_reject_rule_types_strict={
            "stadium_name_pattern",
            "team_name_pattern",
            "birthyear_season_age_consistency",
        },
        soccer_reject_rule_types_soft={"surname_pattern"},
    )

    return cfgs


CONFIGS = build_configs()


# ============================================================
# 3. 模型结构与预测
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


def load_detector_model(ckpt_path: str, stage_cfg: Optional[Dict[str, Any]] = None) -> MLPDetector:
    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    def pick(name, default=None):
        if name in ckpt:
            return ckpt[name]
        if stage_cfg and name in stage_cfg:
            return stage_cfg[name]
        return default

    input_dim = int(pick("input_dim"))
    h1 = int(pick("hidden_dim_1", 128))
    h2 = int(pick("hidden_dim_2", 64))
    dropout = float(pick("dropout", 0.2))

    model = MLPDetector(input_dim=input_dim, hidden_dim_1=h1, hidden_dim_2=h2, dropout=dropout).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def transform_with_preprocessor(preprocessor, df: pd.DataFrame, feature_cols: Optional[List[str]] = None):
    if feature_cols:
        try:
            X = preprocessor.transform(df[feature_cols])
        except Exception:
            X = preprocessor.transform(df)
    else:
        X = preprocessor.transform(df)
    if hasattr(X, "toarray"):
        X = X.toarray()
    return X


def predict_mlp(model: MLPDetector, X_np: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    probs = []
    with torch.no_grad():
        for start in range(0, len(X_np), batch_size):
            x = torch.tensor(X_np[start:start + batch_size], dtype=torch.float32, device=DEVICE)
            logits = model(x)
            p = torch.sigmoid(logits).detach().cpu().numpy()
            probs.append(p)
    return np.concatenate(probs) if probs else np.array([])


def get_stage_threshold(config: Dict[str, Any], metrics: Dict[str, Any], stage: str, default=0.5) -> float:
    # preferred: metrics_two_stage.json from existing scripts
    for key in [f"{stage}_best_threshold"]:
        if key in metrics:
            return float(metrics[key])
    try:
        if f"{stage}_metrics" in metrics and "threshold" in metrics[f"{stage}_metrics"]:
            return float(metrics[f"{stage}_metrics"]["threshold"])
    except Exception:
        pass
    try:
        if stage in config and "metrics" in config[stage] and "threshold" in config[stage]["metrics"]:
            return float(config[stage]["metrics"]["threshold"])
    except Exception:
        pass
    return float(default)


# ============================================================
# 4. 派生特征
# ============================================================

ADULT_PRIMARY_TARGET_COLUMNS = {"sex", "relationship", "education"}
ADULT_CONTEXT_ONLY_COLUMNS = {
    "age", "workclass", "maritalstatus", "occupation",
    "race", "hoursperweek", "country", "income"
}
RELATIONSHIP_V2_RULE_TYPES = {
    "relationship_marital_spouse_conflict",
    "relationship_age_spouse_conflict",
    "relationship_sex_spouse_conflict",
    "relationship_context_dominant",
}
EDUCATION_V2_RULE_TYPES = {"education_age_extreme_conflict"}

SOCCER_COLUMNS = {"name", "surname", "birthyear", "birthplace", "position", "team", "city", "stadium", "season", "manager"}
SOCCER_PRIMARY_TARGET_COLUMNS = set(SOCCER_COLUMNS)
SOCCER_CONTEXT_ONLY_COLUMNS = set()
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


def add_adult_v2_features(out: pd.DataFrame) -> pd.DataFrame:
    col_series = safe_str_series(out, "column", "missing")
    rule_series = safe_str_series(out, "main_rule_type", "missing")

    for c in [
        "target_is_primary_column", "target_is_context_only_column",
        "is_relationship_spouse_value", "relationship_marital_conflict",
        "relationship_age_conflict", "relationship_sex_conflict",
        "sex_value_invalid", "education_age_extreme_conflict",
        "has_relationship_v2_rule", "has_education_v2_rule",
    ]:
        out[c] = safe_num_series(out, c, 0).astype(int)

    out["target_is_primary_column"] = np.where(col_series.isin(ADULT_PRIMARY_TARGET_COLUMNS), 1, out["target_is_primary_column"]).astype(int)
    out["target_is_context_only_column"] = np.where(col_series.isin(ADULT_CONTEXT_ONLY_COLUMNS), 1, out["target_is_context_only_column"]).astype(int)
    out["has_relationship_v2_rule"] = np.where(rule_series.isin(RELATIONSHIP_V2_RULE_TYPES), 1, out["has_relationship_v2_rule"]).astype(int)
    out["has_education_v2_rule"] = np.where(rule_series.isin(EDUCATION_V2_RULE_TYPES), 1, out["has_education_v2_rule"]).astype(int)

    out["relationship_v2_conflict_count"] = (
        out["relationship_marital_conflict"].astype(int)
        + out["relationship_age_conflict"].astype(int)
        + out["relationship_sex_conflict"].astype(int)
        + out["has_relationship_v2_rule"].astype(int)
    )
    out["has_relationship_v2_signal"] = (out["relationship_v2_conflict_count"] > 0).astype(int)
    out["has_education_v2_signal"] = (
        (out["education_age_extreme_conflict"].astype(int) > 0)
        | (out["has_education_v2_rule"].astype(int) > 0)
    ).astype(int)

    out["is_context_only_weak_signal"] = (
        (out["target_is_context_only_column"].astype(int) == 1)
        & (safe_num_series(out, "adult_domain_rule_count", 0) <= 0)
        & (safe_num_series(out, "adult_format_rule_count", 0) <= 0)
        & (safe_num_series(out, "adult_consistency_rule_count", 0) <= 0)
        & (out["has_relationship_v2_signal"].astype(int) <= 0)
        & (out["has_education_v2_signal"].astype(int) <= 0)
        & (out["sex_value_invalid"].astype(int) <= 0)
    ).astype(int)

    out["adult_v2_signal_strength"] = (
        2.5 * out["has_relationship_v2_signal"].astype(float)
        + 2.0 * out["sex_value_invalid"].astype(float)
        + 1.5 * out["has_education_v2_signal"].astype(float)
        + 1.2 * safe_num_series(out, "has_adult_domain_signal", 0).astype(float)
        + 1.0 * safe_num_series(out, "has_adult_format_signal", 0).astype(float)
        + 1.0 * safe_num_series(out, "has_adult_consistency_signal", 0).astype(float)
        + 0.5 * out["target_is_primary_column"].astype(float)
        - 1.0 * out["is_context_only_weak_signal"].astype(float)
    )
    out["adult_v2_primary_target_signal"] = (
        (out["target_is_primary_column"].astype(int) == 1)
        & (
            (out["has_relationship_v2_signal"].astype(int) == 1)
            | (out["sex_value_invalid"].astype(int) == 1)
            | (out["has_education_v2_signal"].astype(int) == 1)
            | (safe_num_series(out, "has_adult_domain_signal", 0).astype(int) == 1)
            | (safe_num_series(out, "has_adult_format_signal", 0).astype(int) == 1)
            | (safe_num_series(out, "has_adult_consistency_signal", 0).astype(int) == 1)
        )
    ).astype(int)

    if "adult_v2_attribution_bucket" not in out.columns:
        out["adult_v2_attribution_bucket"] = "unknown_attribution"
    out["adult_v2_attribution_bucket"] = out["adult_v2_attribution_bucket"].fillna("unknown_attribution").astype(str)
    return out


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

    out["target_is_primary_column"] = np.where(col_series.isin(SOCCER_PRIMARY_TARGET_COLUMNS), 1, out["target_is_primary_column"]).astype(int)
    out["target_is_context_only_column"] = np.where(col_series.isin(SOCCER_CONTEXT_ONLY_COLUMNS), 1, 0).astype(int)

    out["birthyear_value_invalid"] = np.where(rule_series.eq("birthyear_numeric_range"), 1, out["birthyear_value_invalid"]).astype(int)
    out["season_value_invalid"] = np.where(rule_series.eq("season_numeric_range"), 1, out["season_value_invalid"]).astype(int)
    out["position_value_invalid"] = np.where(rule_series.eq("position_domain"), 1, out["position_value_invalid"]).astype(int)
    out["birthyear_season_age_conflict"] = np.where(rule_series.eq("birthyear_season_age_consistency"), 1, out["birthyear_season_age_conflict"]).astype(int)
    out["team_context_conflict"] = np.where(rule_series.isin(SOCCER_CONTEXT_RULE_TYPES), 1, out["team_context_conflict"]).astype(int)
    out["format_rule_signal"] = np.where(rule_series.isin(SOCCER_FORMAT_RULE_TYPES), 1, out["format_rule_signal"]).astype(int)
    out["fd_like_signal"] = np.where(rule_series.isin(SOCCER_FD_RULE_TYPES), 1, out["fd_like_signal"]).astype(int)

    if "evidence_only_fd_count" not in out.columns:
        out["evidence_only_fd_count"] = 0
    out["evidence_only_fd_count"] = safe_num_series(out, "evidence_only_fd_count", 0).astype(float)
    soft_fd_mask = rule_series.isin(SOCCER_EVIDENCE_ONLY_FD_RULE_TYPES)
    out.loc[soft_fd_mask, "fd_like_signal"] = 0
    out.loc[soft_fd_mask, "evidence_only_fd_count"] = np.maximum(out.loc[soft_fd_mask, "evidence_only_fd_count"].astype(float), 1.0)

    out["position_invalid_or_normalization"] = (
        (out["position_value_invalid"].astype(int) > 0)
        | (out["position_needs_canonicalization"].astype(int) > 0)
    ).astype(int)
    out["has_birthyear_or_season_signal"] = (
        (out["birthyear_value_invalid"].astype(int) > 0)
        | (out["season_value_invalid"].astype(int) > 0)
        | (out["birthyear_season_age_conflict"].astype(int) > 0)
        | rule_series.isin({"birthyear_numeric_range", "season_numeric_range", "birthyear_season_age_consistency"})
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
    out.loc[soft_fd_mask, "has_fd_like_signal"] = 0

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


def add_derived_features(df: pd.DataFrame, cfg: InferConfig) -> pd.DataFrame:
    out = df.copy()

    for col in [
        "strong_rule_count", "candidate_generation_rule_count", "fd_like_count", "context_rule_count",
        "global_rule_count", "rare_value_count", "typo_rule_count", "pattern_rule_count", "schema_rule_count",
        "time_window_rule_count", "numeric_window_rule_count", "canonical_rule_count",
        "adult_domain_rule_count", "adult_format_rule_count", "adult_consistency_rule_count",
        "soccer_domain_rule_count", "soccer_format_rule_count", "soccer_numeric_rule_count",
        "soccer_consistency_rule_count", "evidence_only_fd_count",
    ]:
        out[col] = safe_num_series(out, col, 0.0)

    for col in [
        "conflict_score", "value_frequency", "value_frequency_rank",
        "neighbor_majority_ratio", "prior_error_probability", "posterior_error_probability",
        "time_value_minutes", "numeric_value", "year_value", "row_birthyear_int",
        "row_season_int", "row_age_at_season", "soccer_consistency_score",
        "adult_consistency_score",
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
    out["neighbor_agreement_score"] = np.where(out["is_equal_to_neighbor_majority"] == 1, ratio, -ratio).astype(float)
    out["has_neighbor_signal"] = (ratio > 0).astype(int)
    out["high_neighbor_consistency"] = (ratio >= 0.8).astype(int)
    out["neighbor_disagree"] = ((out["has_neighbor_signal"] == 1) & (out["is_equal_to_neighbor_majority"] == 0)).astype(int)

    out["rule_total_count"] = out["strong_rule_count"] + out["candidate_generation_rule_count"]
    out["candidate_rule_ratio"] = np.where(out["rule_total_count"] > 0, out["candidate_generation_rule_count"] / (out["rule_total_count"] + 1e-8), 0.0)
    out["strong_rule_ratio"] = np.where(out["rule_total_count"] > 0, out["strong_rule_count"] / (out["rule_total_count"] + 1e-8), 0.0)
    out["context_rule_ratio"] = np.where(out["rule_total_count"] > 0, out["context_rule_count"] / (out["rule_total_count"] + 1e-8), 0.0)
    out["fd_like_ratio"] = np.where(out["rule_total_count"] > 0, out["fd_like_count"] / (out["rule_total_count"] + 1e-8), 0.0)
    out["global_rule_ratio"] = np.where(out["rule_total_count"] > 0, out["global_rule_count"] / (out["rule_total_count"] + 1e-8), 0.0)

    col_series = safe_str_series(out, "column", "missing")
    long_tail_cols = {"name", "surname", "team", "city", "stadium", "manager", "birthplace", "country", "occupation", "workclass"}
    value_freq = safe_num_series(out, "value_frequency", 999999)
    value_rank = safe_num_series(out, "value_frequency_rank", 999999)
    out["is_rare_value"] = np.where(
        col_series.isin(long_tail_cols),
        ((value_freq <= 1) | (value_rank > 25)).astype(int),
        ((value_freq <= 5) | (value_rank > 10)).astype(int),
    )

    out["rule_strength_sum"] = (
        1.00 * out["strong_rule_count"]
        + 0.25 * out["candidate_generation_rule_count"]
        + 0.40 * out["fd_like_count"]
        + 0.35 * out["context_rule_count"]
        + 0.15 * out["global_rule_count"]
        + 0.20 * out["rare_value_count"]
        + 0.35 * out["typo_rule_count"]
        + 0.45 * out["pattern_rule_count"]
        + 0.45 * out["schema_rule_count"]
        + 0.50 * out["time_window_rule_count"]
        + 0.50 * out["numeric_window_rule_count"]
        + 0.50 * out["canonical_rule_count"]
    )

    out["candidate_rule_dominant"] = np.where(out["candidate_generation_rule_count"] > out["strong_rule_count"], "yes", "no")
    out["rare_only_signal"] = (
        (out["rare_value_count"] > 0)
        & (out["strong_rule_count"] <= 0)
        & (out["fd_like_count"] <= 0)
        & (out["context_rule_count"] <= 0)
        & (out["typo_rule_count"] <= 0)
        & (out["schema_rule_count"] <= 0)
    ).astype(int)
    out["is_weak_only"] = (
        (out["rare_only_signal"] == 1)
        | (
            (out["strong_rule_count"] <= 0)
            & (out["fd_like_count"] <= 0)
            & (out["typo_rule_count"] <= 0)
            & (out["schema_rule_count"] <= 0)
            & (out["candidate_generation_rule_count"] > 0)
        )
    ).astype(int)

    out["is_fp_risk_col"] = col_series.isin(cfg.fp_risk_columns).astype(int)
    out["high_prob_but_neighbor_support_current"] = (
        (out["high_neighbor_consistency"] == 1)
        & (out["is_equal_to_neighbor_majority"] == 1)
    ).astype(int)

    if cfg.dataset == "adult":
        out["adult_rule_total_count"] = out["adult_domain_rule_count"] + out["adult_format_rule_count"] + out["adult_consistency_rule_count"]
        out["adult_domain_rule_ratio"] = np.where(out["adult_rule_total_count"] > 0, out["adult_domain_rule_count"] / (out["adult_rule_total_count"] + 1e-8), 0.0)
        out["adult_format_rule_ratio"] = np.where(out["adult_rule_total_count"] > 0, out["adult_format_rule_count"] / (out["adult_rule_total_count"] + 1e-8), 0.0)
        out["adult_consistency_rule_ratio"] = np.where(out["adult_rule_total_count"] > 0, out["adult_consistency_rule_count"] / (out["adult_rule_total_count"] + 1e-8), 0.0)
        out["domain_invalid_flag"] = safe_str_series(out, "domain_bucket", "missing").eq("domain_invalid").astype(int)
        out["adult_consistency_flag"] = safe_str_series(out, "adult_consistency_bucket", "missing").isin([
            "strong_adult_consistency_signal", "adult_consistency_flag", "weak_adult_consistency_signal"
        ]).astype(int)
        out["high_adult_consistency_score"] = (safe_num_series(out, "adult_consistency_score", 0.0) >= 1.0).astype(int)
        out["has_adult_domain_signal"] = (out["adult_domain_rule_count"] > 0).astype(int)
        out["has_adult_format_signal"] = (out["adult_format_rule_count"] > 0).astype(int)
        out["has_adult_consistency_signal"] = (out["adult_consistency_rule_count"] > 0).astype(int)
        out["age_span"] = np.maximum(0.0, safe_num_series(out, "age_upper", 0.0) - safe_num_series(out, "age_lower", 0.0))
        out["hours_span"] = np.maximum(0.0, safe_num_series(out, "hours_upper", 0.0) - safe_num_series(out, "hours_lower", 0.0))
        out["is_minor_age_bucket"] = ((safe_num_series(out, "age_upper", 0.0) > 0) & (safe_num_series(out, "age_upper", 0.0) < 18)).astype(int)
        out["is_high_hours_bucket"] = (safe_num_series(out, "hours_upper", 0.0) > 60).astype(int)
        out = add_adult_v2_features(out)

    elif cfg.dataset == "soccer":
        out["soccer_rule_total_count"] = out["soccer_domain_rule_count"] + out["soccer_format_rule_count"] + out["soccer_numeric_rule_count"] + out["soccer_consistency_rule_count"]
        out["soccer_domain_rule_ratio"] = np.where(out["soccer_rule_total_count"] > 0, out["soccer_domain_rule_count"] / (out["soccer_rule_total_count"] + 1e-8), 0.0)
        out["soccer_format_rule_ratio"] = np.where(out["soccer_rule_total_count"] > 0, out["soccer_format_rule_count"] / (out["soccer_rule_total_count"] + 1e-8), 0.0)
        out["soccer_numeric_rule_ratio"] = np.where(out["soccer_rule_total_count"] > 0, out["soccer_numeric_rule_count"] / (out["soccer_rule_total_count"] + 1e-8), 0.0)
        out["soccer_consistency_rule_ratio"] = np.where(out["soccer_rule_total_count"] > 0, out["soccer_consistency_rule_count"] / (out["soccer_rule_total_count"] + 1e-8), 0.0)

        out["adult_rule_total_count"] = out["adult_domain_rule_count"] + out["adult_format_rule_count"] + out["adult_consistency_rule_count"]
        out["adult_domain_rule_ratio"] = np.where(out["adult_rule_total_count"] > 0, out["adult_domain_rule_count"] / (out["adult_rule_total_count"] + 1e-8), 0.0)
        out["adult_format_rule_ratio"] = np.where(out["adult_rule_total_count"] > 0, out["adult_format_rule_count"] / (out["adult_rule_total_count"] + 1e-8), 0.0)
        out["adult_consistency_rule_ratio"] = np.where(out["adult_rule_total_count"] > 0, out["adult_consistency_rule_count"] / (out["adult_rule_total_count"] + 1e-8), 0.0)

        out["domain_invalid_flag"] = safe_str_series(out, "domain_bucket", "missing").eq("domain_invalid").astype(int)
        out["soccer_consistency_flag"] = safe_str_series(out, "soccer_consistency_bucket", "missing").isin([
            "strong_soccer_consistency_signal", "soccer_consistency_flag", "weak_soccer_consistency_signal"
        ]).astype(int)
        out["high_soccer_consistency_score"] = (out["soccer_consistency_score"] >= 1.0).astype(int)
        out["has_soccer_domain_signal"] = (out["soccer_domain_rule_count"] > 0).astype(int)
        out["has_soccer_format_signal"] = (out["soccer_format_rule_count"] > 0).astype(int)
        out["has_soccer_numeric_signal"] = (out["soccer_numeric_rule_count"] > 0).astype(int)
        out["has_soccer_consistency_signal"] = (out["soccer_consistency_rule_count"] > 0).astype(int)

        out["birthyear_season_gap"] = np.maximum(0.0, out["row_season_int"] - out["row_birthyear_int"])
        out["is_young_player_age"] = ((out["row_age_at_season"] > 0) & (out["row_age_at_season"] < 18)).astype(int)
        out["is_old_player_age"] = (out["row_age_at_season"] > 40).astype(int)

        out["adult_consistency_flag"] = out["soccer_consistency_flag"]
        out["high_adult_consistency_score"] = out["high_soccer_consistency_score"]
        out["has_adult_domain_signal"] = out["has_soccer_domain_signal"]
        out["has_adult_format_signal"] = out["has_soccer_format_signal"]
        out["has_adult_consistency_signal"] = out["has_soccer_consistency_signal"]
        for c, v in {
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
        out = add_soccer_features(out)

    return out


# ============================================================
# 5. 阈值与 gate
# ============================================================

def build_stage2_df(df: pd.DataFrame, p1: np.ndarray, stage1_th: float) -> pd.DataFrame:
    out = df.copy()
    out["raw_pred_error_prob"] = p1
    out["raw_pred_label"] = (p1 >= stage1_th).astype(int)
    out["raw_pred_label_name"] = pd.Series(out["raw_pred_label"]).map({1: "error", 0: "correct"})
    out["raw_margin_to_threshold"] = p1 - stage1_th
    out["stage1_high_conf_error"] = (p1 >= max(stage1_th + 0.20, 0.75)).astype(int)
    out["stage1_mid_zone"] = ((p1 >= 0.35) & (p1 <= 0.75)).astype(int)
    return out


def is_direct_strong_signal_df(df: pd.DataFrame, cfg: InferConfig) -> pd.Series:
    rule = safe_str_series(df, "main_rule_type", "missing")
    domain_invalid = safe_str_series(df, "domain_bucket", "missing").eq("domain_invalid")
    if cfg.dataset == "adult":
        return (
            domain_invalid
            | (safe_num_series(df, "adult_domain_rule_count", 0.0) > 0)
            | (safe_num_series(df, "adult_format_rule_count", 0.0) > 0)
            | (safe_num_series(df, "adult_consistency_rule_count", 0.0) > 0)
            | (safe_num_series(df, "sex_value_invalid", 0).astype(int) == 1)
            | (safe_num_series(df, "has_relationship_v2_signal", 0).astype(int) == 1)
            | (safe_num_series(df, "has_education_v2_signal", 0).astype(int) == 1)
            | rule.isin(set(cfg.group_verifier_threshold_by_rule.keys()) - {"rare_value", "rare_pattern", "global_dominant_value", "functional_dependency", "soft_functional_dependency"})
        )
    if cfg.dataset == "soccer":
        return (
            domain_invalid
            | (safe_num_series(df, "soccer_domain_rule_count", 0.0) > 0)
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
    return (
        (safe_num_series(df, "strong_rule_count", 0.0) > 0)
        | (safe_num_series(df, "schema_rule_count", 0.0) > 0)
        | (safe_num_series(df, "typo_rule_count", 0.0) > 0)
        | (safe_num_series(df, "pattern_rule_count", 0.0) > 0)
        | (safe_num_series(df, "time_window_rule_count", 0.0) > 0)
        | (safe_num_series(df, "numeric_window_rule_count", 0.0) > 0)
        | (safe_num_series(df, "canonical_rule_count", 0.0) > 0)
    )


def get_base_thresholds(df: pd.DataFrame, default_threshold: float, cfg: InferConfig) -> pd.Series:
    col = safe_str_series(df, "column", "missing")
    th = pd.Series(np.full(len(df), default_threshold), index=df.index, dtype=float)

    if cfg.use_column_thresholds:
        for c, v in cfg.column_thresholds.items():
            th.loc[col == c] = float(v)

    strong_signal = is_direct_strong_signal_df(df, cfg)
    for c, v in cfg.cond_lower_threshold.items():
        mask = (col == c) & strong_signal
        th.loc[mask] = np.minimum(th.loc[mask], float(v))

    if cfg.use_neighbor_adjustment:
        support_current = (
            (safe_num_series(df, "neighbor_majority_ratio", 0.0) >= cfg.neighbor_support_current_ratio)
            & (safe_num_series(df, "is_equal_to_neighbor_majority", 0).astype(int) == 1)
            & (~strong_signal)
        )
        th.loc[support_current] = np.minimum(0.95, th.loc[support_current] + cfg.neighbor_support_current_subtract)

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


def reliability_tuple_key(keys: Tuple[str, ...], values: Tuple[Any, ...]) -> str:
    parts = []
    for k, v in zip(keys, values):
        if v is None or pd.isna(v):
            v = "missing"
        parts.append(f"{k}={str(v)}")
    return "||".join(parts)


def estimate_reliability_table(cfg: InferConfig, output_dir: str) -> Dict[str, Dict[str, Any]]:
    if not cfg.use_evidence_reliability_gate:
        return {}
    labeled_csv = cfg.llm_labeled_csv
    if not labeled_csv or not os.path.exists(labeled_csv):
        print(f"[Reliability] labeled csv not found: {labeled_csv}; use generic weak-evidence gate only.")
        return {}

    labeled_df = pd.read_csv(labeled_csv, low_memory=False)
    if len(labeled_df) == 0:
        return {}

    labeled_df = add_derived_features(labeled_df, cfg)
    labels = [normalize_label_binary_from_row(row) for _, row in labeled_df.iterrows()]
    labeled_df["_rel_label_binary"] = labels
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
    for keys in cfg.reliability_group_keys:
        existing = [k for k in keys if k in labeled_df.columns]
        if len(existing) != len(keys):
            continue

        group_name = "__".join(keys)
        for group_values, g in labeled_df.groupby(list(keys), dropna=False):
            if not isinstance(group_values, tuple):
                group_values = (group_values,)

            n = int(len(g))
            error_count = int((g["_rel_label_binary"] == 1).sum())
            correct_count = int((g["_rel_label_binary"] == 0).sum())
            estimated_precision = (
                (error_count + cfg.reliability_smooth_alpha)
                / (n + cfg.reliability_smooth_alpha + cfg.reliability_smooth_beta)
            )
            avg_conf = float(g["_rel_confidence"].mean())
            key = reliability_tuple_key(keys, group_values)

            if n < cfg.reliability_min_group_size:
                action, delta = "insufficient", 0.0
            elif estimated_precision < cfg.reliability_low_precision:
                action, delta = "conservative", cfg.reliability_delta_low
            elif estimated_precision < cfg.reliability_mid_precision:
                action, delta = "slightly_conservative", cfg.reliability_delta_mid
            elif estimated_precision >= cfg.reliability_high_precision:
                action, delta = "trust", cfg.reliability_delta_high
            else:
                action, delta = "neutral", 0.0

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
    suffix = cfg.dataset
    json_path = os.path.join(output_dir, f"evidence_reliability_{suffix}.json")
    csv_path = os.path.join(output_dir, f"evidence_reliability_{suffix}.csv")
    ensure_dir(output_dir)
    save_json(reliability, json_path)
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[Reliability] valid LLM labels: {len(labeled_df)}")
    print(f"[Reliability] groups: {len(rows)}")
    print(f"[Reliability] saved: {json_path}")
    print(f"[Reliability] saved: {csv_path}")
    return reliability


def lookup_reliability_delta_for_row(row: pd.Series, reliability: Dict[str, Dict[str, Any]], cfg: InferConfig) -> float:
    if not reliability:
        return 0.0
    for keys in cfg.reliability_group_keys:
        values = tuple(row.get(k, "missing") for k in keys)
        k = reliability_tuple_key(keys, values)
        info = reliability.get(k)
        if info is None:
            continue
        if int(info.get("n", 0)) < cfg.reliability_min_group_size:
            continue
        return float(info.get("threshold_delta", 0.0))
    return 0.0


def apply_evidence_reliability_gate(df: pd.DataFrame, thresholds: pd.Series, reliability: Dict[str, Dict[str, Any]], cfg: InferConfig) -> pd.Series:
    if not cfg.use_evidence_reliability_gate:
        return thresholds

    out_th = thresholds.copy().astype(float)
    rule = safe_str_series(df, "main_rule_type", "missing")
    direct_strong = is_direct_strong_signal_df(df, cfg)

    weak_rule = rule.isin(cfg.weak_rule_types_for_gate)
    context_only_weak = safe_num_series(df, "is_context_only_weak_signal", 0).astype(int).eq(1)

    weak_only = weak_rule & (~direct_strong)
    out_th.loc[weak_only] = np.maximum(out_th.loc[weak_only], cfg.weak_only_min_threshold)

    soft_fd_only = rule.eq("soft_functional_dependency") & (~direct_strong)
    out_th.loc[soft_fd_only] = np.maximum(out_th.loc[soft_fd_only], cfg.soft_fd_only_min_threshold)

    context_rule_names = {
        "dominant_value_by_context", "dominant_value_by_context_pair",
        "relationship_context_dominant", "team_context_dominant",
        "team_season_manager_consistency_hint", "city_stadium_consistency_hint",
    }
    context_dom_only = rule.isin(context_rule_names) & (~direct_strong)
    out_th.loc[context_dom_only] = np.maximum(out_th.loc[context_dom_only], cfg.context_dominant_only_min_threshold)
    out_th.loc[context_only_weak] = np.maximum(out_th.loc[context_only_weak], cfg.context_only_weak_min_threshold)

    if reliability:
        deltas = df.apply(lambda row: lookup_reliability_delta_for_row(row, reliability, cfg), axis=1)
        deltas = pd.to_numeric(deltas, errors="coerce").fillna(0.0)
        deltas = pd.Series(deltas, index=df.index)
        deltas.loc[direct_strong & (deltas > 0)] = np.minimum(deltas.loc[direct_strong & (deltas > 0)], 0.06)
        out_th = out_th + deltas
        unknown_weak = weak_only & (deltas.abs() < 1e-12)
    else:
        unknown_weak = weak_only

    out_th.loc[unknown_weak] = out_th.loc[unknown_weak] + cfg.reliability_delta_unknown_weak
    return out_th.clip(lower=0.05, upper=0.95)


# ============================================================
# 6. verifier 和后处理
# ============================================================

def load_verifier_threshold(cfg: InferConfig, verifier_config_path: str, verifier_metrics_path: str) -> float:
    if cfg.verifier_threshold_mode == "manual":
        return cfg.manual_verifier_threshold

    th = cfg.default_verifier_threshold
    if os.path.exists(verifier_config_path):
        try:
            vc = load_json(verifier_config_path)
            # 兼容 unified_train_mlp 的 two_stage_config 结构或旧 fp_verifier_config
            th = float(vc.get("verifier_threshold", vc.get("threshold", th)))
        except Exception:
            pass

    if os.path.exists(verifier_metrics_path):
        try:
            vm = load_json(verifier_metrics_path)
            if "best_verifier_threshold_on_final_val" in vm:
                th = float(vm["best_verifier_threshold_on_final_val"])
            elif "verifier_threshold" in vm:
                th = float(vm["verifier_threshold"])
            elif "threshold" in vm:
                th = float(vm["threshold"])
        except Exception:
            pass

    if cfg.verifier_threshold_mode == "max_with_min":
        th = max(th, cfg.min_verifier_threshold)
    return float(th)


def get_verifier_group_thresholds(df: pd.DataFrame, base_th: float, cfg: InferConfig) -> pd.Series:
    th = pd.Series(np.full(len(df), base_th), index=df.index, dtype=float)
    col = safe_str_series(df, "column", "missing")
    rule = safe_str_series(df, "main_rule_type", "missing")

    for c, v in cfg.group_verifier_threshold_by_column.items():
        th.loc[col == c] = float(v)
    for r, v in cfg.group_verifier_threshold_by_rule.items():
        th.loc[rule == r] = np.minimum(th.loc[rule == r], float(v))

    # beers: strong structured rules can bypass stricter verifier thresholds
    for c, r in cfg.bypass_verifier_rules:
        mask = col.eq(c) & rule.eq(r)
        th.loc[mask] = np.minimum(th.loc[mask], 0.05)

    # rayyan: date permutation/swap rules are usually strong enough to keep
    if cfg.bypass_verifier_date_rules:
        th.loc[rule.isin(cfg.bypass_verifier_date_rules)] = np.minimum(th.loc[rule.isin(cfg.bypass_verifier_date_rules)], 0.05)

    return th.clip(lower=0.05, upper=0.95)


def build_verifier_df(stage2_df: pd.DataFrame, detector_probs: np.ndarray, detector_thresholds: pd.Series) -> pd.DataFrame:
    out = stage2_df.copy()
    out["detector_pred_error_prob"] = detector_probs
    out["detector_pred_label"] = (detector_probs >= detector_thresholds.values).astype(int)
    out["detector_margin_to_threshold"] = detector_probs - detector_thresholds.values
    out["is_stage_disagree"] = (out["raw_pred_label"].astype(int) != out["detector_pred_label"].astype(int)).astype(int)
    return out


def predict_verifier_keep_prob(model, preprocessor, verifier_input: pd.DataFrame):
    try:
        Xv = preprocessor.transform(verifier_input)
    except Exception:
        # Some preprocessors were fitted on specific feature_cols stored in config.
        Xv = preprocessor.transform(verifier_input.copy())
    if hasattr(Xv, "toarray"):
        Xv = Xv.toarray()

    if hasattr(model, "predict_proba") and len(getattr(model, "classes_", [])) > 1:
        classes = list(model.classes_)
        pos_idx = classes.index(1) if 1 in classes else -1
        return model.predict_proba(Xv)[:, pos_idx]
    pred = model.predict(Xv)
    return pred.astype(float)


def apply_fp_verifier(out_df: pd.DataFrame, detector_prob: np.ndarray, detector_thresholds: pd.Series, cfg: InferConfig, paths: Dict[str, str]) -> pd.DataFrame:
    out_df = out_df.copy()
    out_df["final_pred_label"] = out_df["detector_pred_label"].astype(int)
    out_df["final_pred_label_name"] = out_df["final_pred_label"].map({1: "error", 0: "correct"})
    out_df["final_pred_error_prob"] = out_df["detector_pred_error_prob"]
    out_df["verifier_used"] = 0
    out_df["verifier_keep_error_prob"] = np.nan
    out_df["group_verifier_threshold"] = np.nan
    out_df["verifier_rejected"] = 0
    out_df["verifier_decision"] = "skip"

    if not cfg.use_fp_verifier:
        print("[Verifier] skipped: USE_FP_VERIFIER=False")
        return out_df
    if not (os.path.exists(paths["verifier_model"]) and os.path.exists(paths["verifier_preprocessor"])):
        print("[Verifier] skipped: verifier model/preprocessor not found")
        return out_df

    verifier_model = joblib.load(paths["verifier_model"])
    verifier_pre = joblib.load(paths["verifier_preprocessor"])
    base_th = load_verifier_threshold(cfg, paths["verifier_config"], paths["verifier_metrics"])

    verifier_input = build_verifier_df(out_df, detector_prob, detector_thresholds)
    keep_prob = predict_verifier_keep_prob(verifier_model, verifier_pre, verifier_input)
    group_th = get_verifier_group_thresholds(out_df, base_th, cfg)

    if cfg.verifier_threshold_mode == "manual":
        # Keep dataset-specific group overrides, but manual base is the fallback.
        pass
    elif cfg.verifier_threshold_mode == "max_with_min":
        group_th = np.maximum(group_th, cfg.min_verifier_threshold)

    use_mask = out_df["detector_pred_label"].astype(int).eq(1) if cfg.verifier_only_on_detector_error else pd.Series(True, index=out_df.index)
    keep_mask = keep_prob >= group_th.values

    out_df["verifier_used"] = use_mask.astype(int)
    out_df["verifier_keep_error_prob"] = keep_prob
    out_df["group_verifier_threshold"] = group_th.values
    out_df["verifier_rejected"] = (use_mask & (~keep_mask)).astype(int)
    out_df["verifier_decision"] = np.where(
        use_mask,
        np.where(keep_mask, "keep_error", "reject_error"),
        "skip",
    )
    out_df.loc[use_mask & (~keep_mask), "final_pred_label"] = 0
    out_df["final_pred_label_name"] = out_df["final_pred_label"].map({1: "error", 0: "correct"})

    if cfg.print_verifier_debug:
        print(f"[Verifier] base_th={base_th:.4f}, used={int(use_mask.sum())}, rejected={int((use_mask & (~keep_mask)).sum())}")
    return out_df


def apply_neighbor_adjustment(df: pd.DataFrame, prob_col: str) -> np.ndarray:
    prob = safe_num_series(df, prob_col, 0.0).astype(float).values.copy()
    ratio = safe_num_series(df, "neighbor_majority_ratio", 0.0).astype(float).values
    equal = safe_num_series(df, "is_equal_to_neighbor_majority", 0).astype(int).values
    strong_current = (ratio >= 0.90) & (equal == 1)
    strong_other = (ratio >= 0.90) & (equal == 0)
    prob[strong_current] = np.maximum(0.0, prob[strong_current] - 0.03)
    prob[strong_other] = np.minimum(1.0, prob[strong_other] + 0.03)
    return prob


def apply_column_post_rules(out_df: pd.DataFrame, cfg: InferConfig) -> pd.DataFrame:
    out = out_df.copy()
    col = safe_str_series(out, "column", "missing")
    rule = safe_str_series(out, "main_rule_type", "missing")
    prob = safe_num_series(out, "detector_pred_error_prob", 0.0)
    posterior = safe_num_series(out, "posterior_error_probability", 0.0)

    # beers: keep strong canonical ounces errors if detector predicted them
    if cfg.dataset == "beers" and cfg.bypass_verifier_rules:
        for c, r in cfg.bypass_verifier_rules:
            mask = col.eq(c) & rule.eq(r) & out["detector_pred_label"].astype(int).eq(1)
            out.loc[mask, "final_pred_label"] = 1
            out.loc[mask, "final_pred_label_name"] = "error"
            out.loc[mask, "verifier_decision"] = "bypass_keep_error"

    # rayyan: recover date permutation / swap if detector confidence is high
    if cfg.dataset == "rayyan" and cfg.recover_date_rules:
        mask = (
            rule.isin(cfg.recover_date_rules)
            & out["detector_pred_label"].astype(int).eq(1)
            & ((prob >= 0.50) | (posterior >= 0.90))
        )
        out.loc[mask, "final_pred_label"] = 1
        out.loc[mask, "final_pred_label_name"] = "error"
        out.loc[mask, "verifier_decision"] = np.where(
            out.loc[mask, "verifier_decision"].astype(str).eq("reject_error"),
            "date_rule_recover_keep_error",
            out.loc[mask, "verifier_decision"],
        )

    return out


def get_non_missing_strong_soccer_signal_df(df: pd.DataFrame) -> pd.Series:
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


def apply_soccer_precision_post_gate(out_df: pd.DataFrame, cfg: InferConfig) -> pd.DataFrame:
    if not cfg.use_soccer_precision_post_gate:
        return out_df

    out = out_df.copy()
    out["soccer_precision_gate_applied"] = 0
    out["soccer_precision_gate_reason"] = ""

    col = safe_str_series(out, "column", "missing")
    rule = safe_str_series(out, "main_rule_type", "missing")
    prob = safe_num_series(out, "detector_pred_error_prob", 0.0)
    posterior = safe_num_series(out, "posterior_error_probability", 0.0)
    strong_non_missing = get_non_missing_strong_soccer_signal_df(out)

    value_series = out["value"] if "value" in out.columns else pd.Series([None] * len(out), index=out.index)
    missing_rule = rule.isin({"not_null", "high_risk_missing"})
    missing_only = missing_rule & (~strong_non_missing)

    if cfg.soccer_team_missing_only_reject:
        team_missing_only = col.eq("team") & missing_only
        mask = team_missing_only & out["final_pred_label"].astype(int).eq(1)
        out.loc[mask, "final_pred_label"] = 0
        out.loc[mask, "soccer_precision_gate_applied"] = 1
        out.loc[mask, "soccer_precision_gate_reason"] += "|team_missing_only_reject"

    if cfg.soccer_team_pattern_rule_reject:
        mask = col.eq("team") & rule.eq("team_name_pattern") & out["final_pred_label"].astype(int).eq(1)
        out.loc[mask, "final_pred_label"] = 0
        out.loc[mask, "soccer_precision_gate_applied"] = 1
        out.loc[mask, "soccer_precision_gate_reason"] += "|team_pattern_rule_reject"

    stadium_missing_only = col.eq("stadium") & missing_only
    mask = stadium_missing_only & out["final_pred_label"].astype(int).eq(1) & (
        (prob < cfg.stadium_missing_only_min_prob) | (posterior < 0.90)
    )
    out.loc[mask, "final_pred_label"] = 0
    out.loc[mask, "soccer_precision_gate_applied"] = 1
    out.loc[mask, "soccer_precision_gate_reason"] += "|stadium_missing_only_high_threshold"

    birthyear_missing_only = col.eq("birthyear") & missing_only
    mask = birthyear_missing_only & out["final_pred_label"].astype(int).eq(1) & (
        (prob < cfg.birthyear_missing_only_min_prob) | (posterior < 0.85)
    )
    out.loc[mask, "final_pred_label"] = 0
    out.loc[mask, "soccer_precision_gate_applied"] = 1
    out.loc[mask, "soccer_precision_gate_reason"] += "|birthyear_missing_only_high_threshold"

    text_missing_only = col.isin({"name", "surname", "birthplace", "manager"}) & missing_only
    surname_mask = col.eq("surname") & missing_only & out["final_pred_label"].astype(int).eq(1) & (
        (prob < cfg.surname_missing_only_min_prob) | (posterior < 0.65)
    )
    out.loc[surname_mask, "final_pred_label"] = 0
    out.loc[surname_mask, "soccer_precision_gate_applied"] = 1
    out.loc[surname_mask, "soccer_precision_gate_reason"] += "|surname_missing_only_threshold"

    other_text_mask = text_missing_only & (~col.eq("surname")) & out["final_pred_label"].astype(int).eq(1) & (
        (prob < cfg.text_missing_only_min_prob) | (posterior < 0.60)
    )
    out.loc[other_text_mask, "final_pred_label"] = 0
    out.loc[other_text_mask, "soccer_precision_gate_applied"] = 1
    out.loc[other_text_mask, "soccer_precision_gate_reason"] += "|text_missing_only_threshold"

    entity_weak_only = col.isin({"team", "city", "stadium", "manager", "birthplace"}) & (~strong_non_missing)
    mask = entity_weak_only & out["final_pred_label"].astype(int).eq(1) & (
        (prob < cfg.entity_weak_only_min_prob) | (posterior < 0.90)
    )
    out.loc[mask, "final_pred_label"] = 0
    out.loc[mask, "soccer_precision_gate_applied"] = 1
    out.loc[mask, "soccer_precision_gate_reason"] += "|entity_weak_only_threshold"

    if cfg.soccer_low_precision_rule_reject:
        strict_low_precision_rule = rule.isin(cfg.soccer_reject_rule_types_strict)
        mask = strict_low_precision_rule & out["final_pred_label"].astype(int).eq(1)
        out.loc[mask, "final_pred_label"] = 0
        out.loc[mask, "soccer_precision_gate_applied"] = 1
        out.loc[mask, "soccer_precision_gate_reason"] += "|low_precision_rule_reject"

        soft_low_precision_rule = rule.isin(cfg.soccer_reject_rule_types_soft)
        mask = soft_low_precision_rule & out["final_pred_label"].astype(int).eq(1) & (
            (prob < cfg.surname_pattern_keep_min_prob) | (posterior < cfg.surname_pattern_keep_min_posterior)
        )
        out.loc[mask, "final_pred_label"] = 0
        out.loc[mask, "soccer_precision_gate_applied"] = 1
        out.loc[mask, "soccer_precision_gate_reason"] += "|soft_low_precision_rule_threshold"

    out["final_pred_label_name"] = out["final_pred_label"].map({1: "error", 0: "correct"})
    return out


# ============================================================
# 7. hard cases 与导出
# ============================================================

def mark_hard_cases(out_df: pd.DataFrame, detector_thresholds: pd.Series, cfg: InferConfig) -> pd.DataFrame:
    out = out_df.copy()
    prob = safe_num_series(out, "detector_pred_error_prob", 0.0)
    margin = prob - detector_thresholds

    near_threshold = margin.abs() <= cfg.hard_case_margin
    mid_zone = (prob >= cfg.hard_case_low) & (prob <= cfg.hard_case_high)
    stage_disagree = safe_num_series(out, "is_stage_disagree", 0).astype(int).eq(1)

    verifier_borderline = pd.Series(False, index=out.index)
    if "verifier_keep_error_prob" in out.columns and "group_verifier_threshold" in out.columns:
        verifier_borderline = (
            safe_num_series(out, "verifier_keep_error_prob", 0.0)
            - safe_num_series(out, "group_verifier_threshold", cfg.default_verifier_threshold)
        ).abs() <= cfg.hard_case_margin

    rule_neighbor_conflict = (
        (safe_num_series(out, "neighbor_disagree", 0).astype(int) == 1)
        & (safe_num_series(out, "high_neighbor_consistency", 0).astype(int) == 1)
        & (safe_num_series(out, "detector_pred_label", 0).astype(int) == 1)
    )

    dataset_signal_hardcase = pd.Series(False, index=out.index)
    if cfg.dataset == "adult":
        dataset_signal_hardcase = (
            (safe_num_series(out, "adult_v2_primary_target_signal", 0).astype(int) == 1)
            & prob.between(0.25, 0.85)
        )
    elif cfg.dataset == "soccer":
        dataset_signal_hardcase = (
            (safe_num_series(out, "soccer_primary_target_signal", 0).astype(int) == 1)
            & prob.between(0.25, 0.85)
        )

    out["is_hard_case"] = (
        near_threshold
        | mid_zone
        | stage_disagree
        | verifier_borderline
        | rule_neighbor_conflict
        | dataset_signal_hardcase
    ).astype(int)

    out["hard_case_reason"] = ""
    out.loc[near_threshold, "hard_case_reason"] += "|near_threshold"
    out.loc[mid_zone, "hard_case_reason"] += "|mid_zone"
    out.loc[stage_disagree, "hard_case_reason"] += "|stage_disagree"
    out.loc[verifier_borderline, "hard_case_reason"] += "|verifier_borderline"
    out.loc[rule_neighbor_conflict, "hard_case_reason"] += "|rule_neighbor_conflict"
    out.loc[dataset_signal_hardcase, "hard_case_reason"] += f"|{cfg.dataset}_signal_hardcase"

    high_risk = (
        (safe_num_series(out, "final_pred_label", 0).astype(int) == 1)
        | (prob >= cfg.high_risk_threshold)
    )
    if cfg.dataset == "adult":
        high_risk = high_risk | (safe_num_series(out, "adult_v2_primary_target_signal", 0).astype(int) == 1)
    elif cfg.dataset == "soccer":
        high_risk = high_risk | (safe_num_series(out, "soccer_primary_target_signal", 0).astype(int) == 1)

    out["high_risk"] = high_risk.astype(int)
    out["is_high_risk"] = out["high_risk"]
    return out


def export_hard_cases_jsonl(df: pd.DataFrame, path: str):
    hard_df = df[df["is_hard_case"] == 1].copy()
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
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
            f.write(json.dumps(rec, ensure_ascii=False, default=json_default) + "\n")


# ============================================================
# 8. 主流程
# ============================================================

def get_paths(cfg: InferConfig) -> Dict[str, str]:
    return {
        "output_all_csv": os.path.join(cfg.output_dir, "inference_all_predictions.csv"),
        "output_hard_case_csv": os.path.join(cfg.output_dir, "inference_hard_cases.csv"),
        "output_high_risk_csv": os.path.join(cfg.output_dir, "inference_high_risk.csv"),
        "output_llm_jsonl": os.path.join(cfg.output_dir, "hard_cases_for_llm.jsonl"),
        "stage1_best": os.path.join(cfg.model_dir, "stage1_base_best.pt"),
        "stage1_preprocessor": os.path.join(cfg.model_dir, "stage1_base_preprocessor.joblib"),
        "stage2_best": os.path.join(cfg.model_dir, "stage2_cal_best.pt"),
        "stage2_preprocessor": os.path.join(cfg.model_dir, "stage2_cal_preprocessor.joblib"),
        "config": os.path.join(cfg.model_dir, "two_stage_config.json"),
        "metrics": os.path.join(cfg.model_dir, "metrics_two_stage.json"),
        "verifier_model": os.path.join(cfg.model_dir, "fp_verifier_model.joblib"),
        "verifier_preprocessor": os.path.join(cfg.model_dir, "fp_verifier_preprocessor.joblib"),
        "verifier_config": os.path.join(cfg.model_dir, "fp_verifier_config.json"),
        "verifier_metrics": os.path.join(cfg.model_dir, "verifier_metrics.json"),
    }


def run_inference(cfg: InferConfig):
    ensure_dir(cfg.output_dir)
    paths = get_paths(cfg)

    config = load_json(paths["config"])
    metrics = load_json(paths["metrics"]) if os.path.exists(paths["metrics"]) else {}

    df = pd.read_csv(cfg.input_csv, low_memory=False)
    df = add_derived_features(df, cfg)

    stage1_cfg = config.get("stage1", {})
    stage1_model = load_detector_model(paths["stage1_best"], stage1_cfg)
    stage1_pre = joblib.load(paths["stage1_preprocessor"])
    stage1_th = get_stage_threshold(config, metrics, "stage1", default=0.5)
    stage2_th = get_stage_threshold(config, metrics, "stage2", default=0.5)

    stage1_features = stage1_cfg.get("feature_cols")
    X1 = transform_with_preprocessor(stage1_pre, df, stage1_features)
    p1 = predict_mlp(stage1_model, X1)
    stage2_df = build_stage2_df(df, p1, stage1_th)
    stage2_df = add_derived_features(stage2_df, cfg)

    if cfg.use_stage2 and os.path.exists(paths["stage2_best"]) and os.path.exists(paths["stage2_preprocessor"]):
        stage2_cfg = config.get("stage2", {})
        stage2_model = load_detector_model(paths["stage2_best"], stage2_cfg)
        stage2_pre = joblib.load(paths["stage2_preprocessor"])
        stage2_features = stage2_cfg.get("feature_cols")
        X2 = transform_with_preprocessor(stage2_pre, stage2_df, stage2_features)
        detector_prob = predict_mlp(stage2_model, X2)
    else:
        detector_prob = p1
        stage2_th = stage1_th

    detector_prob_before_adjust = detector_prob.copy()
    if cfg.use_neighbor_adjustment:
        tmp = stage2_df.copy()
        tmp["detector_pred_error_prob"] = detector_prob
        detector_prob = apply_neighbor_adjustment(tmp, "detector_pred_error_prob")

    detector_thresholds = get_base_thresholds(stage2_df, stage2_th, cfg)
    reliability = estimate_reliability_table(cfg, cfg.output_dir)
    detector_thresholds = apply_evidence_reliability_gate(stage2_df, detector_thresholds, reliability, cfg)

    detector_pred = (detector_prob >= detector_thresholds.values).astype(int)

    out_df = stage2_df.copy()
    out_df["detector_pred_error_prob_before_adjust"] = detector_prob_before_adjust
    out_df["detector_pred_error_prob"] = detector_prob
    out_df["detector_threshold"] = detector_thresholds.values
    out_df["detector_threshold_used"] = detector_thresholds.values
    out_df["detector_pred_label"] = detector_pred
    out_df["detector_pred_label_name"] = pd.Series(detector_pred).map({1: "error", 0: "correct"})
    out_df["detector_margin_to_threshold"] = detector_prob - detector_thresholds.values
    out_df["is_stage_disagree"] = (out_df["raw_pred_label"].astype(int) != out_df["detector_pred_label"].astype(int)).astype(int)

    out_df = apply_fp_verifier(out_df, detector_prob, detector_thresholds, cfg, paths)
    out_df["final_pred_error_prob_before_adjust"] = out_df["detector_pred_error_prob"]
    out_df["final_pred_error_prob"] = out_df["detector_pred_error_prob"]

    out_df = apply_column_post_rules(out_df, cfg)
    if cfg.dataset == "soccer":
        out_df = apply_soccer_precision_post_gate(out_df, cfg)
        if cfg.use_soccer_precision_post_gate:
            print(f"[SoccerPostGate] applied={int(safe_num_series(out_df, 'soccer_precision_gate_applied', 0).sum())}, "
                  f"final_after_gate={int(out_df['final_pred_label'].sum())}")

    out_df = mark_hard_cases(out_df, detector_thresholds, cfg)

    ensure_dir(os.path.dirname(paths["output_all_csv"]))
    out_df.to_csv(paths["output_all_csv"], index=False, encoding="utf-8-sig")
    out_df[out_df["is_hard_case"] == 1].to_csv(paths["output_hard_case_csv"], index=False, encoding="utf-8-sig")
    out_df[out_df["high_risk"] == 1].to_csv(paths["output_high_risk_csv"], index=False, encoding="utf-8-sig")
    export_hard_cases_jsonl(out_df, paths["output_llm_jsonl"])

    print("\n===== INFERENCE DONE =====")
    print(f"Dataset: {cfg.dataset}")
    print(f"Input rows: {len(out_df)}")
    print(f"Detector predicted errors: {int(out_df['detector_pred_label'].sum())}")
    print(f"Final predicted errors: {int(out_df['final_pred_label'].sum())}")
    print(f"Verifier used: {int(safe_num_series(out_df, 'verifier_used', 0).sum())}")
    print(f"Verifier rejected: {int(safe_num_series(out_df, 'verifier_rejected', 0).sum())}")
    print(f"Hard cases: {int(out_df['is_hard_case'].sum())}")
    print(f"High risk: {int(out_df['high_risk'].sum())}")
    print(f"Output: {paths['output_all_csv']}")
    print(f"Hard cases: {paths['output_hard_case_csv']}")
    print(f"High risk: {paths['output_high_risk_csv']}")
    print(f"LLM JSONL: {paths['output_llm_jsonl']}")


# ============================================================
# 9. CLI
# ============================================================

def clone_config(cfg: InferConfig) -> InferConfig:
    import copy
    return copy.deepcopy(cfg)


def apply_overrides(cfg: InferConfig, args) -> InferConfig:
    cfg = clone_config(cfg)
    if args.input_csv:
        cfg.input_csv = args.input_csv
    if args.model_dir:
        cfg.model_dir = args.model_dir
    if args.output_dir:
        cfg.output_dir = args.output_dir

    if args.disable_stage2:
        cfg.use_stage2 = False
    if args.disable_verifier:
        cfg.use_fp_verifier = False
    if args.disable_column_thresholds:
        cfg.use_column_thresholds = False
    if args.disable_neighbor_adjustment:
        cfg.use_neighbor_adjustment = False
    if args.disable_reliability_gate:
        cfg.use_evidence_reliability_gate = False
    if args.disable_soccer_post_gate:
        cfg.use_soccer_precision_post_gate = False

    if args.verifier_threshold_mode:
        cfg.verifier_threshold_mode = args.verifier_threshold_mode
    if args.manual_verifier_threshold is not None:
        cfg.manual_verifier_threshold = float(args.manual_verifier_threshold)

    return cfg


def run_one_dataset(dataset: str, args):
    cfg = apply_overrides(CONFIGS[dataset], args)

    old_cwd = None
    if args.cwd:
        old_cwd = os.getcwd()
        os.makedirs(args.cwd, exist_ok=True)
        os.chdir(args.cwd)

    try:
        run_inference(cfg)
    finally:
        if old_cwd is not None:
            os.chdir(old_cwd)


def parse_args():
    parser = argparse.ArgumentParser(description="Unified MLP inference script for six datasets.")
    parser.add_argument("--dataset", required=True, choices=sorted(list(CONFIGS.keys()) + ["all"]))

    parser.add_argument("--input_csv", default=None)
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--cwd", default=None)

    parser.add_argument("--disable-stage2", action="store_true")
    parser.add_argument("--disable-verifier", action="store_true")
    parser.add_argument("--disable-column-thresholds", action="store_true")
    parser.add_argument("--disable-neighbor-adjustment", action="store_true")
    parser.add_argument("--disable-reliability-gate", action="store_true")
    parser.add_argument("--disable-soccer-post-gate", action="store_true")

    parser.add_argument("--verifier-threshold-mode", choices=["manual", "metrics", "max_with_min"], default=None)
    parser.add_argument("--manual-verifier-threshold", type=float, default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    if args.dataset == "all":
        if any([args.input_csv, args.model_dir, args.output_dir, args.cwd]):
            raise ValueError("--dataset all 不建议和 --input_csv / --model_dir / --output_dir / --cwd 混用。请逐个数据集运行。")
        for ds in ["hospital", "flights", "beers", "rayyan", "adult", "soccer"]:
            print("\n" + "=" * 80)
            print(f"[DATASET] {ds}")
            print("=" * 80)
            run_one_dataset(ds, args)
    else:
        run_one_dataset(args.dataset, args)


if __name__ == "__main__":
    main()

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

# ============================================================
# Adult v2 attribution features
# ============================================================
PRIMARY_TARGET_COLUMNS = {"sex", "relationship", "education"}
CONTEXT_ONLY_COLUMNS = {
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

ADULT_V2_NUMERIC_FEATURES = [
    "target_is_primary_column",
    "target_is_context_only_column",
    "is_relationship_spouse_value",
    "relationship_marital_conflict",
    "relationship_age_conflict",
    "relationship_sex_conflict",
    "sex_value_invalid",
    "education_age_extreme_conflict",
    "has_relationship_v2_rule",
    "has_education_v2_rule",
    "relationship_v2_conflict_count",
    "has_relationship_v2_signal",
    "has_education_v2_signal",
    "is_context_only_weak_signal",
    "adult_v2_signal_strength",
    "adult_v2_primary_target_signal",
]
ADULT_V2_CATEGORICAL_FEATURES = [
    "adult_v2_attribution_bucket",
]


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
    "relationship": 0.48,
    "sex": 0.46,
    "education": 0.58,

    # context-only 列默认非常保守，除非 domain/format 直接非法
    "age": 0.80,
    "workclass": 0.82,
    "maritalstatus": 0.78,
    "occupation": 0.82,
    "race": 0.82,
    "hoursperweek": 0.78,
    "country": 0.84,
    "income": 0.78,
}

# 条件性降低 detector 阈值：只对强 domain / format / adult consistency 信号放松
COND_LOWER_THRESHOLD = {
    "relationship": 0.38,
    "sex": 0.38,
    "education": 0.48,

    # context-only 只在强 domain/format 证据下小幅放松
    "age": 0.64,
    "hoursperweek": 0.62,
    "income": 0.62,
    "maritalstatus": 0.66,
    "workclass": 0.68,
    "occupation": 0.70,
    "country": 0.72,
    "race": 0.70,
}

GROUP_VERIFIER_THRESHOLD_BY_COLUMN = {
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

    "relationship_marital_spouse_conflict": 0.42,
    "relationship_age_spouse_conflict": 0.44,
    "relationship_sex_spouse_conflict": 0.44,
    "relationship_context_dominant": 0.48,
    "education_age_extreme_conflict": 0.52,
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
# Evidence reliability calibration from LLM-labeled candidates
# ============================================================
# 该模块只读取 LLM 标注样本，不读取 clean ground truth / fp_details / evaluation_summary。
USE_EVIDENCE_RELIABILITY_GATE = True
LLM_LABELED_CSV = "candidate_sampled_labeled_adult.csv"

EVIDENCE_RELIABILITY_JSON = os.path.join(OUTPUT_DIR, "evidence_reliability_adult.json")
EVIDENCE_RELIABILITY_CSV = os.path.join(OUTPUT_DIR, "evidence_reliability_adult.csv")

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

# weak-only 候选的最低阈值。不是直接删除，只是要求模型更高置信度。
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
    "relationship_context_dominant",
}



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


def add_adult_v2_features(out: pd.DataFrame) -> pd.DataFrame:
    col_series = safe_str_series(out, "column", "missing")
    rule_series = safe_str_series(out, "main_rule_type", "missing")

    # Base v2 fields, with fallback from column/rule_type if upstream old files are used.
    for c in [
        "target_is_primary_column", "target_is_context_only_column",
        "is_relationship_spouse_value", "relationship_marital_conflict",
        "relationship_age_conflict", "relationship_sex_conflict",
        "sex_value_invalid", "education_age_extreme_conflict",
        "has_relationship_v2_rule", "has_education_v2_rule",
    ]:
        out[c] = safe_num_series(out, c, 0).astype(int)

    out["target_is_primary_column"] = np.where(
        col_series.isin(PRIMARY_TARGET_COLUMNS), 1, out["target_is_primary_column"]
    ).astype(int)
    out["target_is_context_only_column"] = np.where(
        col_series.isin(CONTEXT_ONLY_COLUMNS), 1, out["target_is_context_only_column"]
    ).astype(int)

    out["has_relationship_v2_rule"] = np.where(
        rule_series.isin(RELATIONSHIP_V2_RULE_TYPES), 1, out["has_relationship_v2_rule"]
    ).astype(int)
    out["has_education_v2_rule"] = np.where(
        rule_series.isin(EDUCATION_V2_RULE_TYPES), 1, out["has_education_v2_rule"]
    ).astype(int)

    out["relationship_v2_conflict_count"] = (
        out["relationship_marital_conflict"].astype(int)
        + out["relationship_age_conflict"].astype(int)
        + out["relationship_sex_conflict"].astype(int)
    )

    out["has_relationship_v2_signal"] = (
        (out["relationship_v2_conflict_count"] > 0)
        | (out["has_relationship_v2_rule"].astype(int) > 0)
        | rule_series.isin(RELATIONSHIP_V2_RULE_TYPES)
    ).astype(int)

    out["has_education_v2_signal"] = (
        (out["education_age_extreme_conflict"].astype(int) > 0)
        | (out["has_education_v2_rule"].astype(int) > 0)
        | rule_series.isin(EDUCATION_V2_RULE_TYPES)
    ).astype(int)

    out["is_context_only_weak_signal"] = (
        (out["target_is_context_only_column"].astype(int) == 1)
        & (safe_num_series(out, "adult_domain_rule_count", 0.0) <= 0)
        & (safe_num_series(out, "adult_format_rule_count", 0.0) <= 0)
        & (safe_num_series(out, "adult_consistency_rule_count", 0.0) <= 0)
        & (out["has_relationship_v2_signal"].astype(int) <= 0)
        & (out["has_education_v2_signal"].astype(int) <= 0)
        & (out["sex_value_invalid"].astype(int) <= 0)
    ).astype(int)

    out["adult_v2_signal_strength"] = (
        2.5 * out["has_relationship_v2_signal"].astype(float)
        + 2.0 * out["sex_value_invalid"].astype(float)
        + 1.5 * out["has_education_v2_signal"].astype(float)
        + 0.8 * out["target_is_primary_column"].astype(float)
        - 1.2 * out["is_context_only_weak_signal"].astype(float)
    )

    out["adult_v2_primary_target_signal"] = (
        (out["target_is_primary_column"].astype(int) == 1)
        & (
            (out["has_relationship_v2_signal"].astype(int) == 1)
            | (out["sex_value_invalid"].astype(int) == 1)
            | (out["has_education_v2_signal"].astype(int) == 1)
            | (safe_num_series(out, "adult_domain_rule_count", 0.0) > 0)
            | (safe_num_series(out, "adult_format_rule_count", 0.0) > 0)
        )
    ).astype(int)

    if "adult_v2_attribution_bucket" not in out.columns:
        out["adult_v2_attribution_bucket"] = "unknown_attribution"
    out["adult_v2_attribution_bucket"] = out["adult_v2_attribution_bucket"].fillna("unknown_attribution").astype(str)
    missing_bucket = out["adult_v2_attribution_bucket"].isin({"", "nan", "None", "missing"})
    out.loc[missing_bucket & (out["relationship_marital_conflict"] == 1), "adult_v2_attribution_bucket"] = "relationship_marital_conflict"
    out.loc[missing_bucket & (out["relationship_age_conflict"] == 1), "adult_v2_attribution_bucket"] = "relationship_age_conflict"
    out.loc[missing_bucket & (out["relationship_sex_conflict"] == 1), "adult_v2_attribution_bucket"] = "relationship_sex_conflict"
    out.loc[missing_bucket & (out["has_relationship_v2_rule"] == 1), "adult_v2_attribution_bucket"] = "relationship_v2_rule"
    out.loc[missing_bucket & (out["sex_value_invalid"] == 1), "adult_v2_attribution_bucket"] = "sex_value_invalid"
    out.loc[missing_bucket & (out["has_education_v2_rule"] == 1), "adult_v2_attribution_bucket"] = "education_age_extreme_conflict"
    out.loc[missing_bucket & col_series.isin(PRIMARY_TARGET_COLUMNS), "adult_v2_attribution_bucket"] = "primary_target_other"
    out.loc[missing_bucket & col_series.isin(CONTEXT_ONLY_COLUMNS), "adult_v2_attribution_bucket"] = "context_only_other"

    return out



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
    """从 LLM 标注样本估计 evidence reliability。

    注意：只使用 LLM labels，不使用 ground truth。
    """
    if not USE_EVIDENCE_RELIABILITY_GATE:
        return {}

    if not labeled_csv or not os.path.exists(labeled_csv):
        print(f"[Reliability] labeled csv not found: {labeled_csv}; use generic weak-evidence gate only.")
        return {}

    labeled_df = pd.read_csv(labeled_csv, low_memory=False)
    if len(labeled_df) == 0:
        return {}

    # 补齐 v2 字段，兼容旧标注文件。
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
        ("adult_v2_attribution_bucket", ["adult_v2_attribution_bucket"]),
        ("main_rule_type", ["main_rule_type"]),
        ("column__main_rule_type", ["column", "main_rule_type"]),
        ("column__adult_v2_attribution_bucket", ["column", "adult_v2_attribution_bucket"]),
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
    bucket = row.get("adult_v2_attribution_bucket", "missing")

    # 从更细粒度到更粗粒度依次查找。
    keys = [
        reliability_pair_key("column", col, "adult_v2_attribution_bucket", bucket),
        reliability_pair_key("column", col, "main_rule_type", rule),
        reliability_key("adult_v2_attribution_bucket", bucket),
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


def is_direct_strong_v2_signal_df(df: pd.DataFrame) -> pd.Series:
    """真正强的直接证据：domain/format/relationship-marital/relationship-sex/sex-domain 等。

    注意 relationship_context_dominant / soft-FD 不算直接强证据。
    """
    rule = safe_str_series(df, "main_rule_type", "missing")
    domain_invalid = safe_str_series(df, "domain_bucket", "missing").eq("domain_invalid")

    return (
        domain_invalid |
        (safe_num_series(df, "adult_domain_rule_count", 0.0) > 0) |
        (safe_num_series(df, "adult_format_rule_count", 0.0) > 0) |
        (safe_num_series(df, "relationship_marital_conflict", 0).astype(int) == 1) |
        (safe_num_series(df, "relationship_sex_conflict", 0).astype(int) == 1) |
        (safe_num_series(df, "sex_value_invalid", 0).astype(int) == 1) |
        (safe_num_series(df, "education_age_extreme_conflict", 0).astype(int) == 1) |
        rule.str.contains("_domain", regex=False) |
        rule.isin(["age_bucket_format", "hours_per_week_format_range", "income_canonical_format"])
    )


def apply_evidence_reliability_gate(
    df: pd.DataFrame,
    thresholds: pd.Series,
    reliability: Dict[str, Dict[str, Any]],
) -> pd.Series:
    """基于 LLM 标注样本的 reliability + 通用 weak-evidence 原则做温和阈值校准。"""
    if not USE_EVIDENCE_RELIABILITY_GATE:
        return thresholds

    out_th = thresholds.copy().astype(float)

    col = safe_str_series(df, "column", "missing")
    rule = safe_str_series(df, "main_rule_type", "missing")
    direct_strong = is_direct_strong_v2_signal_df(df)

    weak_rule = rule.isin(WEAK_RULE_TYPES_FOR_GATE)
    context_only_weak = safe_num_series(df, "is_context_only_weak_signal", 0).astype(int).eq(1)

    # 1) 通用弱证据门控：弱证据不能仅凭低置信度直接输出。
    weak_only = weak_rule & (~direct_strong)
    out_th.loc[weak_only] = np.maximum(out_th.loc[weak_only], WEAK_ONLY_MIN_THRESHOLD)

    # soft-FD 如果没有 sex/domain/format 等直接强证据，要求更高置信度。
    soft_fd_only = rule.eq("soft_functional_dependency") & (~direct_strong)
    out_th.loc[soft_fd_only] = np.maximum(out_th.loc[soft_fd_only], SOFT_FD_ONLY_MIN_THRESHOLD)

    # context-dominant 如果没有直接冲突证据，只作为弱证据。
    context_dom_only = rule.isin({"relationship_context_dominant", "dominant_value_by_context", "dominant_value_by_context_pair"}) & (~direct_strong)
    out_th.loc[context_dom_only] = np.maximum(out_th.loc[context_dom_only], CONTEXT_DOMINANT_ONLY_MIN_THRESHOLD)

    out_th.loc[context_only_weak] = np.maximum(out_th.loc[context_only_weak], CONTEXT_ONLY_WEAK_MIN_THRESHOLD)

    # 2) LLM reliability calibration：只用 LLM 标注样本，不用 ground truth。
    deltas = df.apply(lambda row: lookup_reliability_delta_for_row(row, reliability), axis=1)
    deltas = pd.to_numeric(deltas, errors="coerce").fillna(0.0)

    # 对强证据最多只允许轻微提高阈值，避免因为 LLM 样本不足错杀召回。
    deltas = pd.Series(deltas, index=df.index)
    deltas.loc[direct_strong & (deltas > 0)] = np.minimum(deltas.loc[direct_strong & (deltas > 0)], 0.06)

    out_th = out_th + deltas

    # 3) 低可靠/未知 weak-only 的额外轻微惩罚。
    unknown_weak = weak_only & (deltas.abs() < 1e-12)
    out_th.loc[unknown_weak] = out_th.loc[unknown_weak] + RELIABILITY_DELTA_UNKNOWN_WEAK

    return out_th.clip(lower=0.05, upper=0.95)



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

    out = add_adult_v2_features(out)

    for col in [
        "column", "semantic_type", "detected_type", "main_rule_type", "main_usage_role",
        "pattern_bucket", "rarity_bucket", "neighbor_bucket", "domain_bucket",
        "adult_consistency_bucket", "age_sampling_bucket", "hours_sampling_bucket",
        "adult_value_bucket", "candidate_rule_dominant",
        "time_window_bucket", "time_value_bucket", "adult_v2_attribution_bucket",
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
    relationship_v2_signal = safe_num_series(df, "has_relationship_v2_signal", 0).astype(int) == 1
    sex_invalid_signal = safe_num_series(df, "sex_value_invalid", 0).astype(int) == 1
    education_v2_signal = safe_num_series(df, "has_education_v2_signal", 0).astype(int) == 1
    # 只有直接强证据才参与降低阈值；relationship_context_dominant/soft-FD 这类弱证据不降低阈值。
    relationship_direct_signal = (
        safe_num_series(df, "relationship_marital_conflict", 0).astype(int).eq(1) |
        safe_num_series(df, "relationship_sex_conflict", 0).astype(int).eq(1)
    )
    primary_v2_signal = relationship_direct_signal | sex_invalid_signal | education_v2_signal

    strong_neighbor_other = nb.isin(["strong_support_other", "medium_support_other"])
    strong_rule = safe_num_series(df, "strong_rule_count", 0.0) > 0

    for col_name, lower_t in COND_LOWER_THRESHOLD.items():
        mask = (
            col.eq(col_name) &
            (
                domain_signal |
                format_signal |
                consistency_signal |
                primary_v2_signal |
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
    no_adult_signal = ~(domain_signal | format_signal | consistency_signal | primary_v2_signal)
    thresholds.loc[support_current & no_adult_signal] = np.maximum(
        thresholds.loc[support_current & no_adult_signal], 0.70
    )

    context_only_weak = safe_num_series(df, "is_context_only_weak_signal", 0).astype(int).eq(1)
    thresholds.loc[context_only_weak] = np.maximum(thresholds.loc[context_only_weak], 0.88)

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
        float(row.get("adult_consistency_rule_count", 0) or 0) > 0 or
        int(float(row.get("relationship_marital_conflict", 0) or 0)) == 1 or
        int(float(row.get("relationship_sex_conflict", 0) or 0)) == 1 or
        int(float(row.get("sex_value_invalid", 0) or 0)) == 1 or
        int(float(row.get("has_education_v2_signal", 0) or 0)) == 1
    )
    if strong_adult_signal:
        th = min(th, 0.50)

    if int(float(row.get("is_context_only_weak_signal", 0) or 0)) == 1:
        th = max(th, 0.82)

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
    v2_mask = (
        (safe_num_series(out, "has_relationship_v2_signal", 0).astype(int) == 1) |
        (safe_num_series(out, "sex_value_invalid", 0).astype(int) == 1) |
        (safe_num_series(out, "has_education_v2_signal", 0).astype(int) == 1) |
        (safe_num_series(out, "is_context_only_weak_signal", 0).astype(int) == 1)
    )

    verifier_mask = detector_positive & (fd_mask | risk_col_mask | neighbor_mask | adult_mask | v2_mask)

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
    relationship_v2_signal = safe_num_series(out, "has_relationship_v2_signal", 0).astype(int).eq(1)
    sex_invalid_signal = safe_num_series(out, "sex_value_invalid", 0).astype(int).eq(1)
    education_v2_signal = safe_num_series(out, "has_education_v2_signal", 0).astype(int).eq(1)
    context_only_weak = safe_num_series(out, "is_context_only_weak_signal", 0).astype(int).eq(1)

    relationship_v2_signal = safe_num_series(out, "has_relationship_v2_signal", 0).astype(int).eq(1)
    sex_invalid_signal = safe_num_series(out, "sex_value_invalid", 0).astype(int).eq(1)
    education_v2_signal = safe_num_series(out, "has_education_v2_signal", 0).astype(int).eq(1)
    context_only_weak = safe_num_series(out, "is_context_only_weak_signal", 0).astype(int).eq(1)
    detector_prob = safe_num_series(out, "detector_pred_error_prob", 0.0)

    # 0) context-only 弱信号强制打回，避免 age/income/race/country 等重新爆 FP
    out.loc[final_label.eq(1) & context_only_weak, "final_pred_label"] = 0
    final_label = safe_num_series(out, "final_pred_label", 0).astype(int)

    # 0.5) primary target 的 v2 强信号召回保护。
    # 注意：relationship_context_dominant / soft-FD 属于弱证据，不在这里无条件恢复；
    # 只恢复 domain/format 或 relationship_marital/relationship_sex/sex_invalid 这类直接强证据。
    relationship_direct_strong = (
        safe_num_series(out, "relationship_marital_conflict", 0).astype(int).eq(1) |
        safe_num_series(out, "relationship_sex_conflict", 0).astype(int).eq(1)
    )
    education_direct_strong = safe_num_series(out, "education_age_extreme_conflict", 0).astype(int).eq(1)

    recover_v2_primary = (
        detector_label.eq(1) &
        final_label.eq(0) &
        (
            (col.eq("relationship") & relationship_direct_strong & (detector_prob >= 0.38)) |
            (col.eq("sex") & sex_invalid_signal & (detector_prob >= 0.35)) |
            (col.eq("education") & education_direct_strong & (detector_prob >= 0.45))
        )
    )
    out.loc[recover_v2_primary, "final_pred_label"] = 1
    final_label = safe_num_series(out, "final_pred_label", 0).astype(int)

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
    final_label = safe_num_series(out, "final_pred_label", 0).astype(int)

    # 5) Adult v2 precision filters:
    # 当前评估显示：
    # - relationship_v2_rule / relationship_context_dominant-only: 3898 FP, 0 TP
    # - relationship_age_conflict: 2 FP, 0 TP
    # - sex primary_target_other + soft FD: 953 FP, 0 TP
    # 因此这些候选在没有直接 domain/format/relationship-marital/relationship-sex/sex-invalid 证据时强制打回。
    v2_bucket = safe_str_series(out, "adult_v2_attribution_bucket", "missing")
    relationship_marital_conflict = safe_num_series(out, "relationship_marital_conflict", 0).astype(int).eq(1)
    relationship_sex_conflict = safe_num_series(out, "relationship_sex_conflict", 0).astype(int).eq(1)
    relationship_age_conflict = safe_num_series(out, "relationship_age_conflict", 0).astype(int).eq(1)

    rel_context_only_fp = (
        final_label.eq(1) &
        col.eq("relationship") &
        (
            v2_bucket.eq("relationship_v2_rule") |
            (
                rule.eq("relationship_context_dominant") &
                (~relationship_marital_conflict) &
                (~relationship_sex_conflict)
            )
        )
    )
    out.loc[rel_context_only_fp, "final_pred_label"] = 0
    final_label = safe_num_series(out, "final_pred_label", 0).astype(int)

    rel_age_only_fp = (
        final_label.eq(1) &
        col.eq("relationship") &
        v2_bucket.eq("relationship_age_conflict") &
        (~relationship_marital_conflict) &
        (~relationship_sex_conflict)
    )
    out.loc[rel_age_only_fp, "final_pred_label"] = 0
    final_label = safe_num_series(out, "final_pred_label", 0).astype(int)

    sex_softfd_fp = (
        final_label.eq(1) &
        col.eq("sex") &
        (~sex_invalid_signal) &
        v2_bucket.eq("primary_target_other") &
        rule.isin({"soft_functional_dependency", "functional_dependency"})
    )
    out.loc[sex_softfd_fp, "final_pred_label"] = 0

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

    # Adult v2 hard-case flags need these local masks.
    # They are derived here as well, instead of relying on variables from other functions.
    relationship_v2_signal = safe_num_series(out, "has_relationship_v2_signal", 0).astype(int).eq(1)
    sex_invalid_signal = safe_num_series(out, "sex_value_invalid", 0).astype(int).eq(1)
    education_v2_signal = safe_num_series(out, "has_education_v2_signal", 0).astype(int).eq(1)
    context_only_weak = safe_num_series(out, "is_context_only_weak_signal", 0).astype(int).eq(1)

    # 把 v2 信号并入 adult_signal，避免后续 hard-case/recovery 漏掉新规则。
    adult_signal = adult_signal | relationship_v2_signal | sex_invalid_signal | education_v2_signal

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

    out["adult_v2_relationship_hardcase"] = (
        col.eq("relationship") & relationship_v2_signal & detector_prob.between(0.25, 0.80)
    ).astype(int)

    out["adult_v2_sex_invalid_hardcase"] = (
        col.eq("sex") & sex_invalid_signal & detector_prob.between(0.25, 0.80)
    ).astype(int)

    out["adult_v2_context_fp_hardcase"] = (
        context_only_weak & final_label.eq(1)
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
                "adult_v2_relationship_hardcase": int(rec.get("adult_v2_relationship_hardcase", 0)),
                "adult_v2_sex_invalid_hardcase": int(rec.get("adult_v2_sex_invalid_hardcase", 0)),
                "adult_v2_context_fp_hardcase": int(rec.get("adult_v2_context_fp_hardcase", 0)),
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

    df = pd.read_csv(INPUT_CSV, low_memory=False)
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

    reliability_map = estimate_reliability_table(LLM_LABELED_CSV)

    base_thresholds = apply_base_column_thresholds(out_df, detector_threshold)
    detector_thresholds = apply_conditional_detector_thresholds(out_df, base_thresholds)
    detector_thresholds = apply_evidence_reliability_gate(out_df, detector_thresholds, reliability_map)

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

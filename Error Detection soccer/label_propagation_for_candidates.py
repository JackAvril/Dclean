import json
from collections import Counter
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. 用户配置区
# ============================================================

INPUT_CLUSTERED_CSV = "candidate_clustered_soccer.csv"
INPUT_LABELED_CSV = "candidate_sampled_labeled_soccer.csv"

OUTPUT_JSONL = "propagated_labels_soccer.jsonl"
OUTPUT_CSV = "propagated_labels_soccer.csv"
OUTPUT_SUMMARY_JSON = "propagation_summary_soccer.json"


# ============================================================
# 传播策略：soccer 版更保守，避免传播太多
# ============================================================

# 簇内传播：要求同一 bucket+cluster 内至少有多少个 LLM 标注样本
MIN_LABELED_PER_CLUSTER = 3

# 同簇内某个标签占比达到多少才传播
CLUSTER_PROPAGATION_THRESHOLD = 0.90

# 是否限制只传播距离簇中心较近的样本
ENABLE_CLUSTER_CENTER_FILTER = True
CLUSTER_CENTER_DIST_QUANTILE = 0.55

# 每个 cluster 最多传播多少个
MAX_CLUSTER_PROPAGATION_PER_CLUSTER = 20

# 每个 column 最多由 cluster propagation 新增多少个伪标签
MAX_CLUSTER_PROPAGATION_PER_COLUMN = 80

# KNN 传播
ENABLE_KNN_PROPAGATION = True
KNN_K = 5
KNN_PROPAGATION_THRESHOLD = 0.88
MIN_TEACHER_SAMPLES_FOR_KNN = 25
KNN_WITHIN_SAME_COLUMN_ONLY = True

KNN_MIN_MARGIN = 0.18
KNN_DISTANCE_EPS = 1e-8
KNN_MAX_AVG_DISTANCE = 3.8

# 每个 column 最多由 KNN propagation 新增多少个伪标签
MAX_KNN_PROPAGATION_PER_COLUMN = 80

# 总传播规模限制：避免传播过多导致伪标签噪声
MAX_TOTAL_PROPAGATED = 800

# 只保留传播置信度不低于该值的伪标签
MIN_PROPAGATION_CONFIDENCE = 0.78

# 样本权重
WEIGHT_LLM = 1.0
WEIGHT_CLUSTER_PROP = 0.70
WEIGHT_KNN_PROP = 0.50

# 如果一个样本是明显弱信号，只允许更高置信度传播
WEAK_SIGNAL_EXTRA_CONFIDENCE = 0.08

RANDOM_STATE = 42


# ============================================================
# Soccer 传播约束
# ============================================================
# Soccer v2：所有列都允许作为候选目标列。
# 上一轮 name/surname/birthplace 漏检严重，因此不再把 name/surname 设为 context-only。
SOCCER_COLUMNS = [
    "name",
    "surname",
    "birthyear",
    "birthplace",
    "position",
    "team",
    "city",
    "stadium",
    "season",
    "manager",
]

PRIMARY_TARGET_COLUMNS = set(SOCCER_COLUMNS)

CONTEXT_ONLY_COLUMNS = set()

# 长尾实体列：可以传播，但只能在 direct missing/format/domain 等强证据下较保守传播。
LONG_TAIL_ENTITY_COLUMNS = {
    "name",
    "surname",
    "birthplace",
    "team",
    "city",
    "stadium",
    "manager",
}

# 传播预算向强约束列倾斜，弱上下文实体列相对保守
MAX_CLUSTER_PROPAGATION_PER_COLUMN_SOCCER = {
    "birthyear": 120,
    "season": 100,
    "position": 100,
    "birthplace": 80,
    "name": 60,
    "surname": 60,

    # 长尾实体关系列更保守，避免 team/city/stadium/manager 被弱上下文伪标签扩散。
    "team": 50,
    "city": 50,
    "stadium": 50,
    "manager": 50,
}

MAX_KNN_PROPAGATION_PER_COLUMN_SOCCER = {
    "birthyear": 100,
    "season": 80,
    "position": 80,
    "birthplace": 60,
    "name": 45,
    "surname": 45,

    "team": 40,
    "city": 40,
    "stadium": 40,
    "manager": 40,
}

# 强规则传播阈值
MIN_BIRTHYEAR_INVALID_PROP_CONF = 0.88
MIN_SEASON_INVALID_PROP_CONF = 0.90
MIN_POSITION_INVALID_PROP_CONF = 0.90
MIN_AGE_CONFLICT_PROP_CONF = 0.88

# team-context / FD 类弱证据传播更保守
MIN_TEAM_CONTEXT_PROP_CONF = 0.92
MIN_FD_LIKE_PROP_CONF = 0.92

# v2 中已经没有 context-only 列。保留该开关只是为了兼容旧代码。
ALLOW_CONTEXT_ONLY_PROPAGATION = False

# name/surname/birthplace/manager 等长尾文本列只有在 missing/format 强证据下传播。
MIN_FORMAT_PROP_CONF = 0.88
MIN_MISSING_PROP_CONF = 0.88
MIN_LONG_TAIL_ENTITY_PROP_CONF = 0.90


# ============================================================
# 2. 特征列配置：soccer 版
# ============================================================

FEATURE_COLUMNS_NUMERIC = [
    "violation_count",
    "conflict_score",
    "value_frequency",
    "value_frequency_rank",
    "neighbor_majority_ratio",
    "prior_error_probability",
    "posterior_error_probability",

    "strong_rule_count",
    "candidate_generation_rule_count",
    "fd_like_count",
    "evidence_only_fd_count",
    "context_rule_count",
    "global_rule_count",
    "rare_value_count",
    "typo_rule_count",
    "pattern_rule_count",
    "schema_rule_count",

    "soccer_domain_rule_count",
    "soccer_format_rule_count",
    "soccer_numeric_rule_count",
    "soccer_consistency_rule_count",
    "soccer_profile_inconsistency_count",
    "soccer_consistency_score",

    "year_value",
    "row_birthyear_int",
    "row_season_int",
    "row_age_at_season",

    "target_is_primary_column",
    "target_is_context_only_column",
    "birthyear_value_invalid",
    "season_value_invalid",
    "position_value_invalid",
    "position_needs_canonicalization",
    "birthyear_season_age_conflict",
    "team_context_conflict",
    "format_rule_signal",
    "fd_like_signal",

    # 兼容旧字段/旧训练脚本
    "adult_domain_rule_count",
    "adult_format_rule_count",
    "adult_consistency_rule_count",
    "adult_profile_inconsistency_count",
    "adult_consistency_score",
    "age_lower",
    "age_upper",
    "hours_lower",
    "hours_upper",
    "education_order",
    "time_window_rule_count",
    "time_value_minutes",

    "signal_priority",
    "dist_to_center",
]

FEATURE_COLUMNS_CATEGORICAL = [
    "column",
    "semantic_type",
    "detected_type",
    "main_rule_type",
    "main_usage_role",
    "pattern_bucket",
    "rarity_bucket",
    "neighbor_bucket",
    "domain_bucket",
    "soccer_consistency_bucket",
    "soccer_attribution_bucket",
    "soccer_value_bucket",
    "canonical_position",
    "row_position_canonical",

    # 兼容旧字段
    "adult_consistency_bucket",
    "adult_value_bucket",
    "time_window_bucket",
    "time_value_bucket",
    "sample_role",
    "adult_v2_attribution_bucket",
]


# ============================================================
# 3. 基础函数
# ============================================================

def normalize_label(x: Any) -> Optional[int]:
    if pd.isna(x) or x is None:
        return None
    s = str(x).strip().lower()
    if s == "error":
        return 1
    if s == "correct":
        return 0
    return None


def parse_is_error(x: Any) -> Optional[int]:
    if pd.isna(x) or x is None:
        return None
    s = str(x).strip().lower()
    if s in {"true", "1", "yes"}:
        return 1
    if s in {"false", "0", "no"}:
        return 0
    return None


def normalize_label_binary_from_row(row: pd.Series) -> Optional[int]:
    if "label_binary" in row.index and pd.notna(row["label_binary"]):
        try:
            return int(float(row["label_binary"]))
        except Exception:
            pass

    if "label" in row.index and pd.notna(row["label"]):
        y = normalize_label(row["label"])
        if y is not None:
            return y

    if "is_error" in row.index and pd.notna(row["is_error"]):
        y = parse_is_error(row["is_error"])
        if y is not None:
            return y

    return None


def label_to_name(y: int) -> str:
    return "error" if int(y) == 1 else "correct"


def build_key(row_id: Any, column: Any) -> Tuple[int, str]:
    return int(row_id), str(column)


def compute_weight(base_weight: float, conf: float) -> float:
    conf = min(max(float(conf), 0.0), 1.0)
    return round(base_weight * conf, 6)


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x: Any, default: int = 0) -> int:
    try:
        if pd.isna(x):
            return default
        return int(float(x))
    except Exception:
        return default


def as_int01(x: Any) -> int:
    if pd.isna(x) or x is None:
        return 0
    s = str(x).strip().lower()
    if s in {"1", "true", "yes"}:
        return 1
    if s in {"0", "false", "no"}:
        return 0
    try:
        return int(float(x))
    except Exception:
        return 0


def get_column_cluster_budget(col_name: str, mode: str) -> int:
    if mode == "cluster":
        return int(MAX_CLUSTER_PROPAGATION_PER_COLUMN_SOCCER.get(col_name, MAX_CLUSTER_PROPAGATION_PER_COLUMN))
    if mode == "knn":
        return int(MAX_KNN_PROPAGATION_PER_COLUMN_SOCCER.get(col_name, MAX_KNN_PROPAGATION_PER_COLUMN))
    return 0


def has_birthyear_invalid_signal(row: pd.Series) -> bool:
    return as_int01(row.get("birthyear_value_invalid", 0)) == 1


def has_season_invalid_signal(row: pd.Series) -> bool:
    return as_int01(row.get("season_value_invalid", 0)) == 1


def has_position_invalid_signal(row: pd.Series) -> bool:
    return (
        as_int01(row.get("position_value_invalid", 0)) == 1
        or as_int01(row.get("position_needs_canonicalization", 0)) == 1
    )


def has_age_consistency_signal(row: pd.Series) -> bool:
    rule = str(row.get("main_rule_type", ""))
    return (
        as_int01(row.get("birthyear_season_age_conflict", 0)) == 1
        or rule == "birthyear_season_age_consistency"
    )


def has_team_context_signal(row: pd.Series) -> bool:
    rule = str(row.get("main_rule_type", ""))
    return (
        as_int01(row.get("team_context_conflict", 0)) == 1
        or rule in {"team_context_dominant", "team_season_manager_consistency_hint", "city_stadium_consistency_hint"}
    )


def has_fd_like_signal(row: pd.Series) -> bool:
    # Soccer v2: soft_functional_dependency 是 evidence-only 弱证据，不再作为强 FD-like 信号。
    rule = str(row.get("main_rule_type", ""))
    return (
        as_int01(row.get("fd_like_signal", 0)) == 1
        or safe_float(row.get("fd_like_count", 0.0), 0.0) > 0
        or rule == "functional_dependency"
    )


def has_evidence_only_fd_signal(row: pd.Series) -> bool:
    rule = str(row.get("main_rule_type", ""))
    return (
        safe_float(row.get("evidence_only_fd_count", 0.0), 0.0) > 0
        or rule == "soft_functional_dependency"
    )


def has_format_signal(row: pd.Series) -> bool:
    return (
        as_int01(row.get("format_rule_signal", 0)) == 1
        or safe_float(row.get("soccer_format_rule_count", 0.0), 0.0) > 0
        or safe_float(row.get("adult_format_rule_count", 0.0), 0.0) > 0
    )


def has_missing_signal(row: pd.Series) -> bool:
    rule = str(row.get("main_rule_type", ""))
    return rule in {"not_null", "high_risk_missing"} or "missing" in str(row.get("soccer_attribution_bucket", "")).lower()


def has_direct_schema_signal(row: pd.Series) -> bool:
    return (
        has_missing_signal(row)
        or has_format_signal(row)
        or has_birthyear_invalid_signal(row)
        or has_season_invalid_signal(row)
        or has_position_invalid_signal(row)
        or has_age_consistency_signal(row)
        or safe_float(row.get("soccer_domain_rule_count", 0.0), 0.0) > 0
        or safe_float(row.get("soccer_numeric_rule_count", 0.0), 0.0) > 0
        or safe_float(row.get("schema_rule_count", 0.0), 0.0) > 0
    )


def has_strong_soccer_signal(row: pd.Series) -> bool:
    return has_direct_schema_signal(row)


def should_allow_propagation_for_row(row: pd.Series, propagated_conf: float, source: str) -> bool:
    """Soccer v2 伪标签传播门控。

    目标：
    1. 所有 soccer 列都可作为 target；
    2. name/surname/birthplace 不再一刀切禁止传播；
    3. 长尾实体列只在 missing/format 等直接强证据下传播；
    4. team-context / FD / soft-FD 这类弱证据要求更高置信，soft-FD 不单独传播 error。
    """
    col = str(row.get("column", ""))

    if col not in PRIMARY_TARGET_COLUMNS:
        return False

    # 兼容旧逻辑：如果未来又设置了 context-only，仍默认不传播。
    if col in CONTEXT_ONLY_COLUMNS and not ALLOW_CONTEXT_ONLY_PROPAGATION:
        return False

    direct_schema = has_direct_schema_signal(row)

    if has_missing_signal(row):
        return propagated_conf >= MIN_MISSING_PROP_CONF

    if has_format_signal(row):
        return propagated_conf >= MIN_FORMAT_PROP_CONF

    if col == "birthyear" and (has_birthyear_invalid_signal(row) or has_age_consistency_signal(row)):
        return propagated_conf >= MIN_BIRTHYEAR_INVALID_PROP_CONF

    if col == "season" and has_season_invalid_signal(row):
        return propagated_conf >= MIN_SEASON_INVALID_PROP_CONF

    if col == "position" and has_position_invalid_signal(row):
        return propagated_conf >= MIN_POSITION_INVALID_PROP_CONF

    if has_age_consistency_signal(row):
        return propagated_conf >= MIN_AGE_CONFLICT_PROP_CONF

    # 长尾实体列没有直接 schema/format/missing 证据时，不允许低置信传播。
    if col in LONG_TAIL_ENTITY_COLUMNS and not direct_schema:
        if has_team_context_signal(row) or has_fd_like_signal(row) or has_evidence_only_fd_signal(row):
            return propagated_conf >= max(MIN_LONG_TAIL_ENTITY_PROP_CONF, MIN_TEAM_CONTEXT_PROP_CONF, MIN_FD_LIKE_PROP_CONF)
        return propagated_conf >= MIN_LONG_TAIL_ENTITY_PROP_CONF

    if has_team_context_signal(row):
        return propagated_conf >= MIN_TEAM_CONTEXT_PROP_CONF

    if has_fd_like_signal(row):
        return propagated_conf >= MIN_FD_LIKE_PROP_CONF

    # soft-FD evidence-only 不单独支持 error 伪标签传播；若传播 correct，可以保守允许。
    if has_evidence_only_fd_signal(row):
        return propagated_conf >= 0.95

    return propagated_conf >= required_confidence_for_row(row, MIN_PROPAGATION_CONFIDENCE)

def safe_none_fill(df: pd.DataFrame, idx: int):
    for col in [
        "error_type",
        "reason_short",
        "reason_detailed",
        "suggested_correct_value",
        "needs_human_review",
        "evidence_used",
    ]:
        if col not in df.columns:
            df[col] = None
        if pd.isna(df.at[idx, col]) or df.at[idx, col] is None:
            df.at[idx, col] = None


def is_weak_signal_row(row: pd.Series) -> bool:
    strong = safe_float(row.get("strong_rule_count", 0.0), 0.0)
    domain = safe_float(row.get("soccer_domain_rule_count", 0.0), 0.0) + safe_float(row.get("adult_domain_rule_count", 0.0), 0.0)
    fmt = safe_float(row.get("soccer_format_rule_count", 0.0), 0.0) + safe_float(row.get("adult_format_rule_count", 0.0), 0.0)
    numeric = safe_float(row.get("soccer_numeric_rule_count", 0.0), 0.0)
    cons = safe_float(row.get("soccer_consistency_rule_count", 0.0), 0.0) + safe_float(row.get("adult_consistency_rule_count", 0.0), 0.0)
    schema = safe_float(row.get("schema_rule_count", 0.0), 0.0)
    fd = safe_float(row.get("fd_like_count", 0.0), 0.0)
    ctx = safe_float(row.get("context_rule_count", 0.0), 0.0)

    col = str(row.get("column", ""))
    main_rule = str(row.get("main_rule_type", ""))

    # 兼容旧逻辑：context-only 列默认弱信号。
    if col in CONTEXT_ONLY_COLUMNS:
        return True

    # 长尾实体字段的 rare/global/context/soft-FD 统计信号较弱。
    if col in LONG_TAIL_ENTITY_COLUMNS and main_rule in {
        "rare_value", "rare_pattern", "global_dominant_value",
        "team_context_dominant", "dominant_value_by_context", "dominant_value_by_context_pair",
        "soft_functional_dependency",
    }:
        return True

    direct = (
        1 if has_birthyear_invalid_signal(row) else 0
    ) + (
        1 if has_season_invalid_signal(row) else 0
    ) + (
        1 if has_position_invalid_signal(row) else 0
    ) + (
        1 if has_age_consistency_signal(row) else 0
    ) + (
        1 if has_format_signal(row) else 0
    )

    return (strong + domain + fmt + numeric + cons + schema + fd + ctx + direct) <= 0


def required_confidence_for_row(row: pd.Series, base_threshold: float) -> float:
    if is_weak_signal_row(row):
        return min(0.98, base_threshold + WEAK_SIGNAL_EXTRA_CONFIDENCE)
    return base_threshold


def propagation_priority(row: pd.Series) -> float:
    """
    后续做总量裁剪时使用。越大越优先保留。
    """
    score = 0.0
    score += safe_float(row.get("propagation_confidence", 0.0), 0.0) * 5.0
    score += safe_float(row.get("strong_rule_count", 0.0), 0.0) * 2.5
    score += safe_float(row.get("soccer_domain_rule_count", 0.0), 0.0) * 2.2
    score += safe_float(row.get("soccer_format_rule_count", 0.0), 0.0) * 2.0
    score += safe_float(row.get("soccer_numeric_rule_count", 0.0), 0.0) * 2.2
    score += safe_float(row.get("soccer_consistency_rule_count", 0.0), 0.0) * 1.8

    # 兼容旧字段
    score += safe_float(row.get("adult_domain_rule_count", 0.0), 0.0) * 1.5
    score += safe_float(row.get("adult_format_rule_count", 0.0), 0.0) * 1.2
    score += safe_float(row.get("adult_consistency_rule_count", 0.0), 0.0) * 1.0

    score += safe_float(row.get("schema_rule_count", 0.0), 0.0) * 1.2
    score += safe_float(row.get("fd_like_count", 0.0), 0.0) * 0.8
    score += safe_float(row.get("context_rule_count", 0.0), 0.0) * 0.7
    score += safe_float(row.get("posterior_error_probability", 0.0), 0.0) * 1.0
    score += safe_float(row.get("signal_priority", 0.0), 0.0) * 0.3

    col = str(row.get("column", ""))
    if col == "birthyear":
        score += 5.0
    elif col == "season":
        score += 4.5
    elif col == "position":
        score += 4.0
    elif col in {"birthplace", "name", "surname"}:
        score += 3.0
    elif col in {"team", "city", "stadium", "manager"}:
        score += 2.0
    elif col in CONTEXT_ONLY_COLUMNS:
        score -= 6.0

    if has_birthyear_invalid_signal(row):
        score += 8.0
    if has_season_invalid_signal(row):
        score += 7.0
    if has_position_invalid_signal(row):
        score += 7.0
    if has_age_consistency_signal(row):
        score += 6.0
    if has_missing_signal(row):
        score += 5.0
    if has_format_signal(row):
        score += 4.0

    # team context 和 FD 是弱证据，优先级不要过高；soft-FD evidence-only 需要降权。
    if has_team_context_signal(row):
        score += 0.8
    if has_fd_like_signal(row):
        score += 0.6
    if has_evidence_only_fd_signal(row):
        score -= 1.5

    nb = str(row.get("neighbor_bucket", ""))
    if nb == "strong_support_other":
        score += 1.2
    elif nb == "strong_support_current":
        score += 0.4

    if str(row.get("domain_bucket", "")) == "domain_invalid":
        score += 1.5

    return float(score)


# ============================================================
# 4. 读取数据
# ============================================================

def load_clustered_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["row_id"] = df["row_id"].astype(int)
    df["column"] = df["column"].astype(str)

    expected_defaults = {
        "value": None,
        "semantic_type": "unknown",
        "detected_type": "unknown",
        "violation_count": 0.0,
        "conflict_score": 0.0,
        "value_frequency": 0.0,
        "value_frequency_rank": 0.0,
        "neighbor_majority_value": None,
        "neighbor_majority_ratio": 0.0,
        "prior_error_probability": 0.0,
        "posterior_error_probability": 0.0,

        "main_rule_type": "unknown",
        "main_usage_role": "unknown",
        "strong_rule_count": 0.0,
        "candidate_generation_rule_count": 0.0,
        "fd_like_count": 0.0,
        "evidence_only_fd_count": 0.0,
        "context_rule_count": 0.0,
        "global_rule_count": 0.0,
        "rare_value_count": 0.0,
        "typo_rule_count": 0.0,
        "pattern_rule_count": 0.0,
        "schema_rule_count": 0.0,

        "soccer_domain_rule_count": 0.0,
        "soccer_format_rule_count": 0.0,
        "soccer_numeric_rule_count": 0.0,
        "soccer_consistency_rule_count": 0.0,
        "soccer_profile_inconsistency_count": 0.0,
        "soccer_consistency_score": 0.0,

        "year_value": 0.0,
        "canonical_position": None,
        "domain_valid": None,
        "soccer_value_bucket": "unknown",
        "row_birthyear_int": 0.0,
        "row_season_int": 0.0,
        "row_age_at_season": 0.0,
        "row_position_canonical": None,

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
        "soccer_attribution_bucket": "unknown",

        "pattern_bucket": "unknown",
        "rarity_bucket": "unknown",
        "neighbor_bucket": "unknown",
        "domain_bucket": "unknown",
        "soccer_consistency_bucket": "unknown",

        # 兼容旧字段
        "adult_domain_rule_count": 0.0,
        "adult_format_rule_count": 0.0,
        "adult_consistency_rule_count": 0.0,
        "age_lower": 0.0,
        "age_upper": 0.0,
        "hours_lower": 0.0,
        "hours_upper": 0.0,
        "canonical_income": None,
        "education_order": 0.0,
        "adult_value_bucket": "soccer_not_applicable",
        "adult_profile_inconsistency_count": 0.0,
        "adult_consistency_score": 0.0,
        "adult_consistency_bucket": "soccer_not_applicable",
        "age_sampling_bucket": "soccer_not_applicable",
        "hours_sampling_bucket": "soccer_not_applicable",
        "time_window_rule_count": 0.0,
        "time_value_minutes": 0.0,
        "time_window_bucket": "soccer_not_applicable",
        "time_value_bucket": "soccer_not_applicable",
        "adult_v2_attribution_bucket": "soccer_not_applicable",

        "bucket_id": "unknown",
        "cluster_id": -1,
        "dist_to_center": 0.0,
        "sample_role": "",
        "is_sampled": 0,
        "signal_priority": 0.0,
    }

    for col, default_val in expected_defaults.items():
        if col not in df.columns:
            df[col] = default_val

    numeric_cols = [
        "violation_count",
        "conflict_score",
        "value_frequency",
        "value_frequency_rank",
        "neighbor_majority_ratio",
        "prior_error_probability",
        "posterior_error_probability",

        "strong_rule_count",
        "candidate_generation_rule_count",
        "fd_like_count",
        "evidence_only_fd_count",
        "context_rule_count",
        "global_rule_count",
        "rare_value_count",
        "typo_rule_count",
        "pattern_rule_count",
        "schema_rule_count",

        "soccer_domain_rule_count",
        "soccer_format_rule_count",
        "soccer_numeric_rule_count",
        "soccer_consistency_rule_count",
        "soccer_profile_inconsistency_count",
        "soccer_consistency_score",

        "year_value",
        "row_birthyear_int",
        "row_season_int",
        "row_age_at_season",

        "target_is_primary_column",
        "target_is_context_only_column",
        "birthyear_value_invalid",
        "season_value_invalid",
        "position_value_invalid",
        "position_needs_canonicalization",
        "birthyear_season_age_conflict",
        "team_context_conflict",
        "format_rule_signal",
        "fd_like_signal",

        "adult_domain_rule_count",
        "adult_format_rule_count",
        "adult_consistency_rule_count",
        "age_lower",
        "age_upper",
        "hours_lower",
        "hours_upper",
        "education_order",
        "adult_profile_inconsistency_count",
        "adult_consistency_score",
        "time_window_rule_count",
        "time_value_minutes",

        "cluster_id",
        "dist_to_center",
        "is_sampled",
        "signal_priority",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Soccer v2：所有 soccer 列都是 primary target；soft-FD 作为 evidence-only。
    df["target_is_primary_column"] = df["column"].isin(PRIMARY_TARGET_COLUMNS).astype(int)
    df["target_is_context_only_column"] = df["column"].isin(CONTEXT_ONLY_COLUMNS).astype(int)

    soft_fd_mask = df["main_rule_type"].astype(str).eq("soft_functional_dependency")
    if soft_fd_mask.any():
        df.loc[soft_fd_mask, "fd_like_signal"] = 0
        df.loc[soft_fd_mask, "evidence_only_fd_count"] = (
            pd.to_numeric(df.loc[soft_fd_mask, "evidence_only_fd_count"], errors="coerce").fillna(0.0) + 1.0
        )

    return df


def load_labeled_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["row_id"] = df["row_id"].astype(int)
    df["column"] = df["column"].astype(str)

    labels = []
    for _, row in df.iterrows():
        labels.append(normalize_label_binary_from_row(row))
    df["label_binary_norm"] = labels

    df = df[pd.notna(df["label_binary_norm"])].copy()
    df["label_binary_norm"] = df["label_binary_norm"].astype(int)

    if "confidence" not in df.columns:
        df["confidence"] = 1.0
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(1.0)

    expected_cols = {
        "error_type": None,
        "reason_short": None,
        "reason_detailed": None,
        "suggested_correct_value": None,
        "needs_human_review": None,
        "evidence_used": None,
    }
    for col, default_val in expected_cols.items():
        if col not in df.columns:
            df[col] = default_val

    return df


# ============================================================
# 5. 合并 clustered + labeled
# ============================================================

def merge_clustered_and_labeled(clustered_df: pd.DataFrame, labeled_df: pd.DataFrame) -> pd.DataFrame:
    df = clustered_df.copy()

    # 如果同一个 cell 被重复标注，保留 confidence 更高的一条
    labeled_df = labeled_df.copy()
    labeled_df["_key"] = list(zip(labeled_df["row_id"].astype(int), labeled_df["column"].astype(str)))
    labeled_df = labeled_df.sort_values(by=["_key", "confidence"], ascending=[True, False])
    labeled_df = labeled_df.drop_duplicates(subset=["_key"], keep="first").copy()

    label_map: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for _, row in labeled_df.iterrows():
        key = build_key(row["row_id"], row["column"])
        label_map[key] = {
            "label_binary": int(row["label_binary_norm"]),
            "confidence": float(row["confidence"]) if pd.notna(row["confidence"]) else 1.0,
            "error_type": row["error_type"] if "error_type" in row else None,
            "reason_short": row["reason_short"] if "reason_short" in row else None,
            "reason_detailed": row["reason_detailed"] if "reason_detailed" in row else None,
            "suggested_correct_value": row["suggested_correct_value"] if "suggested_correct_value" in row else None,
            "needs_human_review": row["needs_human_review"] if "needs_human_review" in row else None,
            "evidence_used": row["evidence_used"] if "evidence_used" in row else None,
        }

    labels = []
    confs = []
    errtypes = []
    reasons_short = []
    reasons_detailed = []
    suggested_values = []
    needs_review = []
    evidence_useds = []
    is_labeled = []

    for _, row in df.iterrows():
        key = build_key(row["row_id"], row["column"])
        if key in label_map:
            item = label_map[key]
            labels.append(item["label_binary"])
            confs.append(item["confidence"])
            errtypes.append(item["error_type"])
            reasons_short.append(item["reason_short"])
            reasons_detailed.append(item["reason_detailed"])
            suggested_values.append(item["suggested_correct_value"])
            needs_review.append(item["needs_human_review"])
            evidence_useds.append(item["evidence_used"])
            is_labeled.append(1)
        else:
            labels.append(np.nan)
            confs.append(np.nan)
            errtypes.append(None)
            reasons_short.append(None)
            reasons_detailed.append(None)
            suggested_values.append(None)
            needs_review.append(None)
            evidence_useds.append(None)
            is_labeled.append(0)

    df["label_binary"] = labels
    df["label_confidence"] = confs
    df["error_type"] = errtypes
    df["reason_short"] = reasons_short
    df["reason_detailed"] = reasons_detailed
    df["suggested_correct_value"] = suggested_values
    df["needs_human_review"] = needs_review
    df["evidence_used"] = evidence_useds
    df["is_labeled"] = is_labeled

    df["final_label_binary"] = df["label_binary"]
    df["final_label"] = df["label_binary"].map(lambda x: label_to_name(x) if pd.notna(x) else None)
    df["propagation_confidence"] = df["label_confidence"]
    df["label_source"] = df["is_labeled"].map(lambda x: "llm" if x == 1 else None)
    df["sample_weight"] = df["is_labeled"].map(lambda x: WEIGHT_LLM if x == 1 else np.nan)
    df["propagation_priority"] = np.nan

    return df


# ============================================================
# 6. 准备特征向量
# ============================================================

def prepare_features(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    work = df.copy()

    for col in FEATURE_COLUMNS_NUMERIC:
        if col not in work.columns:
            work[col] = 0.0
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)

    num_df = work[FEATURE_COLUMNS_NUMERIC].copy()

    cat_frames = []
    feat_names = FEATURE_COLUMNS_NUMERIC.copy()

    for col in FEATURE_COLUMNS_CATEGORICAL:
        if col not in work.columns:
            work[col] = "unknown"
        dummies = pd.get_dummies(work[col].fillna("unknown").astype(str), prefix=col)
        cat_frames.append(dummies)
        feat_names.extend(list(dummies.columns))

    feat_df = pd.concat([num_df] + cat_frames, axis=1) if cat_frames else num_df

    scaler = StandardScaler()
    X = scaler.fit_transform(feat_df.values)

    return X, feat_names


# ============================================================
# 7. 第一层传播：簇内传播
# ============================================================

def cluster_label_propagation(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    column_added_counter = Counter()

    for (bucket_id, cluster_id), grp in df.groupby(["bucket_id", "cluster_id"], sort=False):
        labeled_grp = grp[pd.notna(grp["final_label_binary"])]

        if len(labeled_grp) < MIN_LABELED_PER_CLUSTER:
            continue

        counts = labeled_grp["final_label_binary"].value_counts(dropna=True).to_dict()
        error_cnt = counts.get(1.0, 0) + counts.get(1, 0)
        correct_cnt = counts.get(0.0, 0) + counts.get(0, 0)
        total = error_cnt + correct_cnt

        if total == 0:
            continue

        error_ratio = error_cnt / total
        correct_ratio = correct_cnt / total

        propagated_label = None
        propagated_conf = None

        if error_ratio >= CLUSTER_PROPAGATION_THRESHOLD:
            propagated_label = 1
            propagated_conf = error_ratio
        elif correct_ratio >= CLUSTER_PROPAGATION_THRESHOLD:
            propagated_label = 0
            propagated_conf = correct_ratio

        if propagated_label is None:
            continue

        # teacher 越少，置信度越打折
        teacher_support_factor = min(1.0, total / 8.0)
        propagated_conf = float(propagated_conf) * teacher_support_factor

        if propagated_conf < MIN_PROPAGATION_CONFIDENCE:
            continue

        unlabeled_grp = grp[pd.isna(grp["final_label_binary"])].copy()
        if len(unlabeled_grp) == 0:
            continue

        if ENABLE_CLUSTER_CENTER_FILTER:
            unlabeled_grp["dist_to_center"] = pd.to_numeric(
                unlabeled_grp["dist_to_center"], errors="coerce"
            ).fillna(np.inf)
            dist_threshold = unlabeled_grp["dist_to_center"].quantile(CLUSTER_CENTER_DIST_QUANTILE)
            unlabeled_grp = unlabeled_grp[unlabeled_grp["dist_to_center"] <= dist_threshold].copy()

        if len(unlabeled_grp) == 0:
            continue

        unlabeled_grp["_row_allowed_to_propagate"] = unlabeled_grp.apply(
            lambda r: should_allow_propagation_for_row(r, propagated_conf, "cluster_propagation"), axis=1
        )
        unlabeled_grp = unlabeled_grp[unlabeled_grp["_row_allowed_to_propagate"] == 1].copy()

        if len(unlabeled_grp) == 0:
            continue

        unlabeled_grp["_prop_priority"] = unlabeled_grp.apply(propagation_priority, axis=1)
        unlabeled_grp = unlabeled_grp.sort_values(
            by=["_prop_priority", "dist_to_center", "conflict_score"],
            ascending=[False, True, False],
            na_position="last"
        ).head(MAX_CLUSTER_PROPAGATION_PER_CLUSTER)

        for idx in unlabeled_grp.index.tolist():
            col_name = str(df.at[idx, "column"])
            if column_added_counter[col_name] >= get_column_cluster_budget(col_name, "cluster"):
                continue

            df.at[idx, "final_label_binary"] = propagated_label
            df.at[idx, "final_label"] = label_to_name(propagated_label)
            df.at[idx, "propagation_confidence"] = round(float(propagated_conf), 6)
            df.at[idx, "label_source"] = "cluster_propagation"
            df.at[idx, "sample_weight"] = compute_weight(WEIGHT_CLUSTER_PROP, propagated_conf)
            df.at[idx, "propagation_priority"] = propagation_priority(df.loc[idx])
            safe_none_fill(df, idx)

            column_added_counter[col_name] += 1

    return df


# ============================================================
# 8. 第二层传播：距离加权 KNN
# ============================================================

def _weighted_neighbor_vote(neighbor_labels: np.ndarray, neighbor_distances: np.ndarray) -> Tuple[Optional[int], Optional[float]]:
    weights = 1.0 / (neighbor_distances + KNN_DISTANCE_EPS)

    error_score = float(np.sum(weights[neighbor_labels == 1]))
    correct_score = float(np.sum(weights[neighbor_labels == 0]))
    total_score = error_score + correct_score

    if total_score <= 0:
        return None, None

    error_ratio = error_score / total_score
    correct_ratio = correct_score / total_score

    propagated_label = None
    propagated_conf = None

    if error_ratio >= KNN_PROPAGATION_THRESHOLD and (error_ratio - correct_ratio) >= KNN_MIN_MARGIN:
        propagated_label = 1
        propagated_conf = error_ratio
    elif correct_ratio >= KNN_PROPAGATION_THRESHOLD and (correct_ratio - error_ratio) >= KNN_MIN_MARGIN:
        propagated_label = 0
        propagated_conf = correct_ratio

    return propagated_label, propagated_conf


def knn_label_propagation(df: pd.DataFrame, X: np.ndarray) -> pd.DataFrame:
    df = df.copy()

    teacher_mask = (df["label_source"] == "llm")
    teacher_indices = df[teacher_mask].index.tolist()

    if len(teacher_indices) < MIN_TEACHER_SAMPLES_FOR_KNN:
        return df

    column_added_counter = Counter()

    if KNN_WITHIN_SAME_COLUMN_ONLY:
        all_columns = df["column"].dropna().astype(str).unique().tolist()

        for col_name in all_columns:
            col_teacher_indices = df[
                (df["label_source"] == "llm") & (df["column"] == col_name)
            ].index.tolist()

            col_unlabeled_indices = df[
                (df["final_label_binary"].isna()) & (df["column"] == col_name)
            ].index.tolist()

            min_col_teachers = max(3, min(KNN_K, MIN_TEACHER_SAMPLES_FOR_KNN // 4))
            if len(col_teacher_indices) < min_col_teachers:
                continue
            if len(col_unlabeled_indices) == 0:
                continue

            X_teacher = X[col_teacher_indices]
            y_teacher = df.loc[col_teacher_indices, "final_label_binary"].astype(int).values

            nn_model = NearestNeighbors(
                n_neighbors=min(KNN_K, len(col_teacher_indices)),
                metric="euclidean"
            )
            nn_model.fit(X_teacher)

            candidate_updates = []

            for idx in col_unlabeled_indices:
                x = X[idx].reshape(1, -1)
                neighbor_distances, neighbor_pos = nn_model.kneighbors(x)
                neighbor_distances = neighbor_distances[0]
                neighbor_pos = neighbor_pos[0]
                neighbor_labels = y_teacher[neighbor_pos]

                avg_distance = float(np.mean(neighbor_distances))
                if avg_distance > KNN_MAX_AVG_DISTANCE:
                    continue

                propagated_label, propagated_conf = _weighted_neighbor_vote(
                    neighbor_labels, neighbor_distances
                )

                if propagated_label is None or propagated_conf is None:
                    continue

                if not should_allow_propagation_for_row(df.loc[idx], propagated_conf, "knn_propagation"):
                    continue

                priority = propagation_priority(df.loc[idx]) + propagated_conf * 3.0 - avg_distance * 0.2
                candidate_updates.append((idx, propagated_label, propagated_conf, avg_distance, priority))

            candidate_updates.sort(key=lambda x: (x[4], x[2], -x[3]), reverse=True)

            for idx, propagated_label, propagated_conf, avg_distance, priority in candidate_updates:
                if column_added_counter[col_name] >= get_column_cluster_budget(col_name, "knn"):
                    break

                df.at[idx, "final_label_binary"] = propagated_label
                df.at[idx, "final_label"] = label_to_name(propagated_label)
                df.at[idx, "propagation_confidence"] = round(float(propagated_conf), 6)
                df.at[idx, "label_source"] = "knn_propagation"
                df.at[idx, "sample_weight"] = compute_weight(WEIGHT_KNN_PROP, propagated_conf)
                df.at[idx, "propagation_priority"] = round(float(priority), 6)
                safe_none_fill(df, idx)

                column_added_counter[col_name] += 1

    else:
        X_teacher = X[teacher_indices]
        y_teacher = df.loc[teacher_indices, "final_label_binary"].astype(int).values

        nn_model = NearestNeighbors(
            n_neighbors=min(KNN_K, len(teacher_indices)),
            metric="euclidean"
        )
        nn_model.fit(X_teacher)

        unlabeled_indices = df[df["final_label_binary"].isna()].index.tolist()
        candidate_updates = []

        for idx in unlabeled_indices:
            x = X[idx].reshape(1, -1)
            neighbor_distances, neighbor_pos = nn_model.kneighbors(x)
            neighbor_distances = neighbor_distances[0]
            neighbor_pos = neighbor_pos[0]
            neighbor_labels = y_teacher[neighbor_pos]

            avg_distance = float(np.mean(neighbor_distances))
            if avg_distance > KNN_MAX_AVG_DISTANCE:
                continue

            propagated_label, propagated_conf = _weighted_neighbor_vote(
                neighbor_labels, neighbor_distances
            )

            if propagated_label is None or propagated_conf is None:
                continue

            if not should_allow_propagation_for_row(df.loc[idx], propagated_conf, "knn_propagation"):
                continue

            priority = propagation_priority(df.loc[idx]) + propagated_conf * 3.0 - avg_distance * 0.2
            candidate_updates.append((idx, propagated_label, propagated_conf, avg_distance, priority))

        candidate_updates.sort(key=lambda x: (x[4], x[2], -x[3]), reverse=True)

        for idx, propagated_label, propagated_conf, avg_distance, priority in candidate_updates:
            df.at[idx, "final_label_binary"] = propagated_label
            df.at[idx, "final_label"] = label_to_name(propagated_label)
            df.at[idx, "propagation_confidence"] = round(float(propagated_conf), 6)
            df.at[idx, "label_source"] = "knn_propagation"
            df.at[idx, "sample_weight"] = compute_weight(WEIGHT_KNN_PROP, propagated_conf)
            df.at[idx, "propagation_priority"] = round(float(priority), 6)
            safe_none_fill(df, idx)

    return df


# ============================================================
# 9. 总量裁剪：保留 LLM，全局限制传播样本
# ============================================================

def limit_total_propagated(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    llm_df = df[df["label_source"] == "llm"].copy()
    prop_df = df[df["label_source"].isin(["cluster_propagation", "knn_propagation"])].copy()
    unlabeled_df = df[pd.isna(df["label_source"])].copy()

    if len(prop_df) <= MAX_TOTAL_PROPAGATED:
        return df

    prop_df["_prop_priority_final"] = prop_df.apply(propagation_priority, axis=1)

    # 按列 round-robin 保留，避免某一列传播样本过多
    selected_indices = []
    grouped = {
        col: g.sort_values(
            by=["_prop_priority_final", "propagation_confidence", "conflict_score"],
            ascending=[False, False, False]
        )
        for col, g in prop_df.groupby("column", sort=False)
    }

    round_idx = 0
    while len(selected_indices) < MAX_TOTAL_PROPAGATED:
        added = False
        for col, g in grouped.items():
            if round_idx < len(g):
                selected_indices.append(g.index[round_idx])
                added = True
                if len(selected_indices) >= MAX_TOTAL_PROPAGATED:
                    break
        if not added:
            break
        round_idx += 1

    keep_prop_df = prop_df.loc[selected_indices].copy()

    drop_prop_df = prop_df.drop(index=selected_indices).copy()
    for col in ["final_label_binary", "final_label", "propagation_confidence", "label_source", "sample_weight", "propagation_priority"]:
        drop_prop_df[col] = np.nan if col != "final_label" else None

    out = pd.concat([llm_df, keep_prop_df, drop_prop_df, unlabeled_df], axis=0)
    out = out.sort_index()
    return out


# ============================================================
# 10. 导出结果
# ============================================================

def make_json_safe(v):
    if pd.isna(v):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def row_to_obj(row: pd.Series) -> Dict[str, Any]:
    obj = {
        "row_id": int(row["row_id"]),
        "column": str(row["column"]),
        "value": make_json_safe(row.get("value")),
        "bucket_id": make_json_safe(row.get("bucket_id")),
        "cluster_id": int(row["cluster_id"]) if pd.notna(row["cluster_id"]) else None,
        "sample_role": make_json_safe(row.get("sample_role")),
        "is_sampled": int(row["is_sampled"]) if pd.notna(row["is_sampled"]) else None,
        "dist_to_center": float(row["dist_to_center"]) if pd.notna(row["dist_to_center"]) else None,
        "signal_priority": float(row["signal_priority"]) if pd.notna(row.get("signal_priority")) else None,

        "label": row["final_label"],
        "label_binary": int(row["final_label_binary"]),

        "propagation_confidence": float(row["propagation_confidence"]) if pd.notna(row["propagation_confidence"]) else None,
        "label_source": make_json_safe(row.get("label_source")),
        "sample_weight": float(row["sample_weight"]) if pd.notna(row["sample_weight"]) else None,
        "propagation_priority": float(row["propagation_priority"]) if pd.notna(row.get("propagation_priority")) else None,

        "semantic_type": make_json_safe(row.get("semantic_type")),
        "detected_type": make_json_safe(row.get("detected_type")),
        "main_rule_type": make_json_safe(row.get("main_rule_type")),
        "main_usage_role": make_json_safe(row.get("main_usage_role")),
        "pattern_bucket": make_json_safe(row.get("pattern_bucket")),
        "rarity_bucket": make_json_safe(row.get("rarity_bucket")),
        "neighbor_bucket": make_json_safe(row.get("neighbor_bucket")),
        "domain_bucket": make_json_safe(row.get("domain_bucket")),

        "soccer_consistency_bucket": make_json_safe(row.get("soccer_consistency_bucket")),
        "soccer_attribution_bucket": make_json_safe(row.get("soccer_attribution_bucket")),
        "soccer_value_bucket": make_json_safe(row.get("soccer_value_bucket")),
        "canonical_position": make_json_safe(row.get("canonical_position")),
        "row_position_canonical": make_json_safe(row.get("row_position_canonical")),

        "target_is_primary_column": int(as_int01(row.get("target_is_primary_column", 0))),
        "target_is_context_only_column": int(as_int01(row.get("target_is_context_only_column", 0))),
        "birthyear_value_invalid": int(as_int01(row.get("birthyear_value_invalid", 0))),
        "season_value_invalid": int(as_int01(row.get("season_value_invalid", 0))),
        "position_value_invalid": int(as_int01(row.get("position_value_invalid", 0))),
        "position_needs_canonicalization": int(as_int01(row.get("position_needs_canonicalization", 0))),
        "birthyear_season_age_conflict": int(as_int01(row.get("birthyear_season_age_conflict", 0))),
        "team_context_conflict": int(as_int01(row.get("team_context_conflict", 0))),
        "format_rule_signal": int(as_int01(row.get("format_rule_signal", 0))),
        "fd_like_signal": int(as_int01(row.get("fd_like_signal", 0))),
        "evidence_only_fd_count": float(row.get("evidence_only_fd_count", 0.0)) if pd.notna(row.get("evidence_only_fd_count", 0.0)) else 0.0,

        "violation_count": float(row["violation_count"]) if pd.notna(row["violation_count"]) else None,
        "conflict_score": float(row["conflict_score"]) if pd.notna(row["conflict_score"]) else None,
        "value_frequency": float(row["value_frequency"]) if pd.notna(row["value_frequency"]) else None,
        "value_frequency_rank": float(row["value_frequency_rank"]) if pd.notna(row["value_frequency_rank"]) else None,

        "neighbor_majority_value": make_json_safe(row.get("neighbor_majority_value")),
        "neighbor_majority_ratio": float(row["neighbor_majority_ratio"]) if pd.notna(row["neighbor_majority_ratio"]) else None,
        "prior_error_probability": float(row["prior_error_probability"]) if pd.notna(row["prior_error_probability"]) else None,
        "posterior_error_probability": float(row["posterior_error_probability"]) if pd.notna(row["posterior_error_probability"]) else None,

        "strong_rule_count": float(row["strong_rule_count"]) if pd.notna(row["strong_rule_count"]) else None,
        "candidate_generation_rule_count": float(row["candidate_generation_rule_count"]) if pd.notna(row["candidate_generation_rule_count"]) else None,
        "fd_like_count": float(row["fd_like_count"]) if pd.notna(row["fd_like_count"]) else None,
        "evidence_only_fd_count": float(row.get("evidence_only_fd_count", 0.0)) if pd.notna(row.get("evidence_only_fd_count", 0.0)) else None,
        "context_rule_count": float(row["context_rule_count"]) if pd.notna(row["context_rule_count"]) else None,
        "global_rule_count": float(row["global_rule_count"]) if pd.notna(row["global_rule_count"]) else None,
        "rare_value_count": float(row["rare_value_count"]) if pd.notna(row["rare_value_count"]) else None,
        "typo_rule_count": float(row["typo_rule_count"]) if pd.notna(row["typo_rule_count"]) else None,
        "pattern_rule_count": float(row["pattern_rule_count"]) if pd.notna(row["pattern_rule_count"]) else None,
        "schema_rule_count": float(row["schema_rule_count"]) if pd.notna(row["schema_rule_count"]) else None,

        "soccer_domain_rule_count": float(row["soccer_domain_rule_count"]) if pd.notna(row["soccer_domain_rule_count"]) else None,
        "soccer_format_rule_count": float(row["soccer_format_rule_count"]) if pd.notna(row["soccer_format_rule_count"]) else None,
        "soccer_numeric_rule_count": float(row["soccer_numeric_rule_count"]) if pd.notna(row["soccer_numeric_rule_count"]) else None,
        "soccer_consistency_rule_count": float(row["soccer_consistency_rule_count"]) if pd.notna(row["soccer_consistency_rule_count"]) else None,
        "soccer_profile_inconsistency_count": float(row["soccer_profile_inconsistency_count"]) if pd.notna(row["soccer_profile_inconsistency_count"]) else None,
        "soccer_consistency_score": float(row["soccer_consistency_score"]) if pd.notna(row["soccer_consistency_score"]) else None,

        "year_value": float(row["year_value"]) if pd.notna(row["year_value"]) else None,
        "row_birthyear_int": float(row["row_birthyear_int"]) if pd.notna(row["row_birthyear_int"]) else None,
        "row_season_int": float(row["row_season_int"]) if pd.notna(row["row_season_int"]) else None,
        "row_age_at_season": float(row["row_age_at_season"]) if pd.notna(row["row_age_at_season"]) else None,

        # 兼容旧字段
        "adult_consistency_bucket": make_json_safe(row.get("adult_consistency_bucket")),
        "adult_value_bucket": make_json_safe(row.get("adult_value_bucket")),
        "adult_v2_attribution_bucket": make_json_safe(row.get("adult_v2_attribution_bucket")),
        "time_window_bucket": make_json_safe(row.get("time_window_bucket")),
        "time_value_bucket": make_json_safe(row.get("time_value_bucket")),

        "adult_domain_rule_count": float(row["adult_domain_rule_count"]) if pd.notna(row["adult_domain_rule_count"]) else None,
        "adult_format_rule_count": float(row["adult_format_rule_count"]) if pd.notna(row["adult_format_rule_count"]) else None,
        "adult_consistency_rule_count": float(row["adult_consistency_rule_count"]) if pd.notna(row["adult_consistency_rule_count"]) else None,

        "age_lower": float(row["age_lower"]) if pd.notna(row["age_lower"]) else None,
        "age_upper": float(row["age_upper"]) if pd.notna(row["age_upper"]) else None,
        "hours_lower": float(row["hours_lower"]) if pd.notna(row["hours_lower"]) else None,
        "hours_upper": float(row["hours_upper"]) if pd.notna(row["hours_upper"]) else None,
        "canonical_income": make_json_safe(row.get("canonical_income")),
        "domain_valid": make_json_safe(row.get("domain_valid")),
        "education_order": float(row["education_order"]) if pd.notna(row["education_order"]) else None,
        "adult_profile_inconsistency_count": float(row["adult_profile_inconsistency_count"]) if pd.notna(row["adult_profile_inconsistency_count"]) else None,
        "adult_consistency_score": float(row["adult_consistency_score"]) if pd.notna(row["adult_consistency_score"]) else None,

        "time_value_minutes": float(row["time_value_minutes"]) if pd.notna(row["time_value_minutes"]) else None,
        "time_window_rule_count": float(row["time_window_rule_count"]) if pd.notna(row["time_window_rule_count"]) else None,

        "error_type": make_json_safe(row.get("error_type")),
        "reason_short": make_json_safe(row.get("reason_short")),
        "reason_detailed": make_json_safe(row.get("reason_detailed")),
        "suggested_correct_value": make_json_safe(row.get("suggested_correct_value")),
        "needs_human_review": make_json_safe(row.get("needs_human_review")),
        "evidence_used": make_json_safe(row.get("evidence_used")),
    }

    return obj


def export_results(df: pd.DataFrame, output_jsonl: str, output_csv: str, output_summary_json: str):
    out_rows = []

    labeled_df = df[pd.notna(df["final_label_binary"])].copy()
    labeled_df = labeled_df.sort_values(
        by=["row_id", "column", "label_source"],
        ascending=[True, True, True]
    )

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for _, row in labeled_df.iterrows():
            obj = row_to_obj(row)
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            out_rows.append(obj)

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    summary = {
        "total_clustered_rows": int(len(df)),
        "total_labeled_or_propagated": int(len(labeled_df)),
        "llm_count": int((df["label_source"] == "llm").sum()),
        "cluster_propagation_count": int((df["label_source"] == "cluster_propagation").sum()),
        "knn_propagation_count": int((df["label_source"] == "knn_propagation").sum()),
        "unlabeled_count": int(df["final_label_binary"].isna().sum()),
        "coverage": float(len(labeled_df) / len(df)) if len(df) > 0 else 0.0,
        "label_source_distribution": df["label_source"].value_counts(dropna=False).to_dict(),
        "final_label_distribution": df["final_label"].value_counts(dropna=False).to_dict(),
        "column_distribution": labeled_df["column"].value_counts(dropna=False).to_dict() if len(labeled_df) else {},
        "soccer_attribution_bucket_distribution": (
            labeled_df["soccer_attribution_bucket"].value_counts(dropna=False).to_dict()
            if len(labeled_df) and "soccer_attribution_bucket" in labeled_df.columns else {}
        ),
        "column_source_distribution": (
            labeled_df.groupby(["column", "label_source"]).size().reset_index(name="count").to_dict(orient="records")
            if len(labeled_df) else []
        ),
        "config": {
            "MIN_LABELED_PER_CLUSTER": MIN_LABELED_PER_CLUSTER,
            "CLUSTER_PROPAGATION_THRESHOLD": CLUSTER_PROPAGATION_THRESHOLD,
            "CLUSTER_CENTER_DIST_QUANTILE": CLUSTER_CENTER_DIST_QUANTILE,
            "MAX_CLUSTER_PROPAGATION_PER_CLUSTER": MAX_CLUSTER_PROPAGATION_PER_CLUSTER,
            "MAX_CLUSTER_PROPAGATION_PER_COLUMN": MAX_CLUSTER_PROPAGATION_PER_COLUMN,
            "ENABLE_KNN_PROPAGATION": ENABLE_KNN_PROPAGATION,
            "KNN_K": KNN_K,
            "KNN_PROPAGATION_THRESHOLD": KNN_PROPAGATION_THRESHOLD,
            "KNN_MIN_MARGIN": KNN_MIN_MARGIN,
            "KNN_MAX_AVG_DISTANCE": KNN_MAX_AVG_DISTANCE,
            "MAX_KNN_PROPAGATION_PER_COLUMN": MAX_KNN_PROPAGATION_PER_COLUMN,
            "MAX_TOTAL_PROPAGATED": MAX_TOTAL_PROPAGATED,
            "MIN_PROPAGATION_CONFIDENCE": MIN_PROPAGATION_CONFIDENCE,
            "PRIMARY_TARGET_COLUMNS": sorted(list(PRIMARY_TARGET_COLUMNS)),
            "CONTEXT_ONLY_COLUMNS": sorted(list(CONTEXT_ONLY_COLUMNS)),
            "ALLOW_CONTEXT_ONLY_PROPAGATION": ALLOW_CONTEXT_ONLY_PROPAGATION,
            "MIN_BIRTHYEAR_INVALID_PROP_CONF": MIN_BIRTHYEAR_INVALID_PROP_CONF,
            "MIN_SEASON_INVALID_PROP_CONF": MIN_SEASON_INVALID_PROP_CONF,
            "MIN_POSITION_INVALID_PROP_CONF": MIN_POSITION_INVALID_PROP_CONF,
            "MIN_AGE_CONFLICT_PROP_CONF": MIN_AGE_CONFLICT_PROP_CONF,
            "MIN_TEAM_CONTEXT_PROP_CONF": MIN_TEAM_CONTEXT_PROP_CONF,
            "MIN_FD_LIKE_PROP_CONF": MIN_FD_LIKE_PROP_CONF,
            "MIN_FORMAT_PROP_CONF": MIN_FORMAT_PROP_CONF,
            "MIN_MISSING_PROP_CONF": MIN_MISSING_PROP_CONF,
            "MIN_LONG_TAIL_ENTITY_PROP_CONF": MIN_LONG_TAIL_ENTITY_PROP_CONF,
        }
    }

    with open(output_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return out_df, summary


# ============================================================
# 11. 主流程
# ============================================================

def main():
    clustered_df = load_clustered_csv(INPUT_CLUSTERED_CSV)
    labeled_df = load_labeled_csv(INPUT_LABELED_CSV)

    print("[1/5] 合并 clustered + LLM labels ...")
    df = merge_clustered_and_labeled(clustered_df, labeled_df)

    print("[2/5] 准备特征矩阵 ...")
    X, feature_names = prepare_features(df)

    print("[3/5] 簇内标签传播 ...")
    df = cluster_label_propagation(df)

    print("[4/5] KNN 标签传播 ...")
    if ENABLE_KNN_PROPAGATION:
        df = knn_label_propagation(df, X)

    print("[4.5/5] 总量裁剪 ...")
    df = limit_total_propagated(df)

    print("[5/5] 导出结果 ...")
    out_df, summary = export_results(df, OUTPUT_JSONL, OUTPUT_CSV, OUTPUT_SUMMARY_JSON)

    total = len(df)
    llm_count = int((df["label_source"] == "llm").sum())
    cluster_count = int((df["label_source"] == "cluster_propagation").sum())
    knn_count = int((df["label_source"] == "knn_propagation").sum())
    total_labeled = int(pd.notna(df["final_label_binary"]).sum())

    print(f"[OK] 传播结果 JSONL 已保存: {OUTPUT_JSONL}")
    print(f"[OK] 传播结果 CSV 已保存: {OUTPUT_CSV}")
    print(f"[OK] 传播摘要 JSON 已保存: {OUTPUT_SUMMARY_JSON}")

    print("\n===== 统计信息 =====")
    print(f"总样本数: {total}")
    print(f"LLM 原始标注数: {llm_count}")
    print(f"簇内传播数: {cluster_count}")
    print(f"KNN 传播数: {knn_count}")
    print(f"最终可训练样本数: {total_labeled}")

    if total > 0:
        print(f"训练集覆盖率: {total_labeled / total:.6f}")

    print("\n标签来源分布:")
    print(df["label_source"].value_counts(dropna=False).to_string())

    print("\n最终标签分布:")
    print(df["final_label"].value_counts(dropna=False).to_string())

    print("\n按列分布:")
    if len(out_df) > 0:
        print(out_df["column"].value_counts(dropna=False).to_string())
    else:
        print("(empty)")


if __name__ == "__main__":
    main()

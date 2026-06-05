import json
import math
import os
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. 用户配置区
# ============================================================

# 输入1：soccer 候选上下文扁平表，由 summarize_full_contexts_soccer.py 输出
INPUT_SUMMARY_CSV = "candidate_llm_contexts_flat_soccer.csv"

# 输入2：soccer 候选详细上下文 JSONL，由 summarize_full_contexts_soccer.py 输出
INPUT_DETAIL_JSONL = "candidate_llm_contexts_soccer.jsonl"

# 输出
OUTPUT_CLUSTERED_CSV = "candidate_clustered_soccer.csv"
OUTPUT_SAMPLED_CSV = "candidate_sampled_soccer.csv"
OUTPUT_SAMPLED_JSONL = "candidate_sampled_with_context_soccer.jsonl"
OUTPUT_BUCKET_SUMMARY_CSV = "bucket_cluster_summary_soccer.csv"
OUTPUT_SAMPLE_SUMMARY_JSON = "sample_summary_soccer.json"

# ------------------------------------------------------------
# 采样预算控制：避免后续 LLM token 消耗过大
# ------------------------------------------------------------
# soccer v2 的目标：
# 1) name/surname/birthplace 之前漏检严重，因此必须有采样覆盖；
# 2) soft-FD / rare / global-dominant 等弱信号不能大量占用预算；
# 3) 总预算仍控制在 350 左右，避免 LLM token 成本过高。
GLOBAL_SAMPLE_BUDGET = 350

# 每一列最多采样多少条，防止某一列独占预算。
# v2：name/surname/birthplace 被纳入 primary target，因此配额要足够。
MAX_SAMPLE_PER_COLUMN = {
    "name": 45,
    "surname": 45,
    "birthyear": 45,
    "birthplace": 50,
    "position": 50,
    "team": 45,
    "city": 40,
    "stadium": 45,
    "season": 45,
    "manager": 45,
}
DEFAULT_MAX_SAMPLE_PER_COLUMN = 40

# 每个 bucket 最多采样多少条
MAX_SAMPLE_PER_BUCKET = 20

# 每个 cluster 最多采样多少条
MAX_SAMPLE_PER_CLUSTER = 4

# 小桶是否全采。为防止“几乎全标注”，这里设置得比较保守。
SMALL_BUCKET_FULL_SAMPLE_THRESHOLD = 3

# 每个列至少采样多少条；仅当该列有候选时生效。
# 若候选总量很少，则实际采样数不会超过该列候选数。
MIN_SAMPLE_PER_ACTIVE_COLUMN = 8

# v2 重点补齐列：上一轮这些列候选覆盖不足或漏检严重
HIGH_RECALL_FOCUS_COLUMNS = {"name", "surname", "birthplace"}

# 每个主要规则类型至少采样多少条；仅当该规则类型有候选时生效。
MIN_SAMPLE_PER_IMPORTANT_RULE = 5

# v2：只把强 schema / domain / format / numeric / consistency 规则列为 important。
# 不再把 soft-FD / rare-value / rare-pattern 当作 important rule 强制采样。
IMPORTANT_RULE_TYPES = {
    "not_null",
    "high_risk_missing",

    "birthyear_numeric_range",
    "season_numeric_range",
    "birthyear_season_age_consistency",
    "position_domain",

    "name_pattern",
    "surname_pattern",
    "manager_name_pattern",
    "team_name_pattern",
    "birthplace_location_pattern",
    "city_location_pattern",
    "stadium_name_pattern",
    "generic_regex",
    "pattern",

    # 只有强 FD 允许少量采样，soft-FD 不强制采样
    "functional_dependency",
}

# 低价值弱规则：不禁止采样，但降低优先级，且不强制保底
WEAK_RULE_TYPES = {
    "soft_functional_dependency",
    "rare_value",
    "rare_pattern",
    "global_dominant_value",
    "dominant_value_by_context",
    "dominant_value_by_context_pair",
    "team_context_dominant",
    "team_season_manager_consistency_hint",
    "city_stadium_consistency_hint",
}

# 如果弱规则样本很多，最多保留多少条弱规则样本。
# 这只是最终采样预算内的软限制，避免 LLM 花在 FD/rare 噪声上。
MAX_WEAK_RULE_SAMPLES = 45

# ------------------------------------------------------------
# 聚类参数
# ------------------------------------------------------------
USE_COLUMN_IN_BUCKET = True
USE_RULE_IN_BUCKET = True
USE_SOCCER_SIGNAL_IN_BUCKET = True

MIN_BUCKET_SIZE_FOR_CLUSTER = 8
MAX_CLUSTERS_PER_BUCKET = 8
TARGET_CLUSTER_SIZE = 12

SAMPLE_CENTER = True
SAMPLE_BOUNDARY = True
SAMPLE_OUTLIER = True
SAMPLE_HIGH_POSTERIOR = True
SAMPLE_LOW_POSTERIOR = True

RANDOM_STATE = 42

# ------------------------------------------------------------
# 训练/推理脚本兼容字段
# ------------------------------------------------------------
MISSING_TEXT = "missing"

SOCCER_COLUMNS = [
    "name", "surname", "birthyear", "birthplace", "position",
    "team", "city", "stadium", "season", "manager",
]

# v2：所有 soccer 列都作为候选目标列，不再设置 context-only。
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

# v2：soft-FD 只作为 evidence-only，不再作为高价值采样信号。
SOCCER_FD_RULE_TYPES = {"functional_dependency"}
SOCCER_EVIDENCE_ONLY_FD_RULE_TYPES = {"soft_functional_dependency"}


# ============================================================
# 2. 基础函数
# ============================================================

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


def safe_str(x: Any, default: str = MISSING_TEXT) -> str:
    if x is None:
        return default
    try:
        if pd.isna(x):
            return default
    except Exception:
        pass
    s = str(x).strip()
    return s if s else default


def ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def normalize_boolish(x: Any) -> str:
    if x is None:
        return "unknown"
    try:
        if pd.isna(x):
            return "unknown"
    except Exception:
        pass
    s = str(x).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return "true"
    if s in {"false", "0", "no", "n"}:
        return "false"
    return s if s else "unknown"


def boolish_to_int(x: Any) -> int:
    s = normalize_boolish(x)
    if s == "true":
        return 1
    if s == "false":
        return 0
    try:
        return int(float(s))
    except Exception:
        return 0


def is_weak_rule(rule_type: Any) -> bool:
    return safe_str(rule_type, "none") in WEAK_RULE_TYPES


def is_strong_or_format_rule(rule_type: Any) -> bool:
    r = safe_str(rule_type, "none")
    return r in SOCCER_STRONG_RULE_TYPES or r in SOCCER_FORMAT_RULE_TYPES or r in {"not_null", "high_risk_missing"}


# ============================================================
# 3. 数据加载与字段补齐
# ============================================================

def load_summary_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    # 基础字段
    required_defaults = {
        "row_id": 0,
        "column": MISSING_TEXT,
        "value": None,
        "semantic_type": "unknown",
        "detected_type": "unknown",
        "violation_count": 0,
        "conflict_score": 0.0,
        "value_frequency": 0,
        "value_frequency_rank": np.nan,
        "neighbor_majority_value": None,
        "neighbor_majority_ratio": np.nan,
        "prior_error_probability": 0.0,
        "posterior_error_probability": 0.0,
        "main_rule_type": "none",
        "main_usage_role": "none",
        "strong_rule_count": 0,
        "candidate_generation_rule_count": 0,
        "fd_like_count": 0,
        "evidence_only_fd_count": 0,
        "context_rule_count": 0,
        "global_rule_count": 0,
        "rare_value_count": 0,
        "typo_rule_count": 0,
        "pattern_rule_count": 0,
        "schema_rule_count": 0,
        "soccer_domain_rule_count": 0,
        "soccer_format_rule_count": 0,
        "soccer_numeric_rule_count": 0,
        "soccer_consistency_rule_count": 0,
        "time_window_rule_count": 0,
        "time_value_minutes": np.nan,
        "time_window_bucket": "soccer_not_applicable",
        "time_value_bucket": "soccer_not_applicable",
        "row_missing_count": 0,
        "row_non_missing_count": 0,
        "year_value": np.nan,
        "row_birthyear_int": np.nan,
        "row_season_int": np.nan,
        "row_age_at_season": np.nan,
        "canonical_position": None,
        "row_position_canonical": None,
        "domain_valid": np.nan,
        "soccer_value_bucket": "unknown",
        "soccer_consistency_score": 0.0,
        "soccer_evidence_flags": "",
        "soccer_profile_inconsistency_count": 0,

        # soccer v2 attribution features
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
        "soccer_attribution_bucket": "unknown_attribution",

        # 与 adult 旧脚本兼容
        "adult_domain_rule_count": 0,
        "adult_format_rule_count": 0,
        "adult_consistency_rule_count": 0,
        "adult_consistency_score": 0.0,
        "adult_evidence_flags": "",
        "adult_profile_inconsistency_count": 0,
        "current_value_pattern": "unknown",
        "dominant_pattern": "unknown",
        "dominant_pattern_ratio": np.nan,
    }

    for col, default in required_defaults.items():
        if col not in df.columns:
            df[col] = default

    df["row_id"] = pd.to_numeric(df["row_id"], errors="coerce").fillna(0).astype(int)
    df["column"] = df["column"].fillna(MISSING_TEXT).astype(str)

    numeric_cols = [
        "violation_count", "conflict_score", "value_frequency", "value_frequency_rank",
        "neighbor_majority_ratio", "prior_error_probability", "posterior_error_probability",
        "strong_rule_count", "candidate_generation_rule_count", "fd_like_count", "evidence_only_fd_count",
        "context_rule_count", "global_rule_count", "rare_value_count", "typo_rule_count",
        "pattern_rule_count", "schema_rule_count", "soccer_domain_rule_count",
        "soccer_format_rule_count", "soccer_numeric_rule_count", "soccer_consistency_rule_count",
        "time_window_rule_count", "time_value_minutes", "row_missing_count", "row_non_missing_count",
        "year_value", "row_birthyear_int", "row_season_int", "row_age_at_season",
        "soccer_consistency_score", "soccer_profile_inconsistency_count",
        "target_is_primary_column", "target_is_context_only_column",
        "birthyear_value_invalid", "season_value_invalid",
        "position_value_invalid", "position_needs_canonicalization",
        "birthyear_season_age_conflict", "team_context_conflict",
        "format_rule_signal", "fd_like_signal",
        "adult_domain_rule_count", "adult_format_rule_count", "adult_consistency_rule_count",
        "adult_consistency_score", "adult_profile_inconsistency_count", "dominant_pattern_ratio",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    text_cols = [
        "semantic_type", "detected_type", "main_rule_type", "main_usage_role",
        "soccer_value_bucket", "canonical_position", "row_position_canonical",
        "soccer_evidence_flags", "soccer_attribution_bucket",
        "adult_evidence_flags", "current_value_pattern", "dominant_pattern",
        "time_window_bucket", "time_value_bucket",
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna(MISSING_TEXT).astype(str)

    # v2 纠偏：所有 soccer 列都是 primary target，不再是 context-only。
    df["target_is_primary_column"] = df["column"].isin(PRIMARY_TARGET_COLUMNS).astype(int)
    df["target_is_context_only_column"] = df["column"].isin(CONTEXT_ONLY_COLUMNS).astype(int)

    # v2 纠偏：如果旧上下文里把 soft-FD 计入 fd_like_count，需要尽量降级。
    # 这里只能根据 main_rule_type 做保守修正，完整修正已在 summarize_full_contexts_soccer_v2.py 中完成。
    soft_fd_mask = df["main_rule_type"].astype(str).isin(SOCCER_EVIDENCE_ONLY_FD_RULE_TYPES)
    if soft_fd_mask.any():
        df.loc[soft_fd_mask, "evidence_only_fd_count"] = (
            pd.to_numeric(df.loc[soft_fd_mask, "evidence_only_fd_count"], errors="coerce").fillna(0) + 1
        )
        df.loc[soft_fd_mask, "fd_like_signal"] = 0

    # 如果规则类型能直接推断 attribution feature，则补齐，兼容旧 context 文件。
    rule = df["main_rule_type"].astype(str)

    df["birthyear_value_invalid"] = np.where(
        rule.eq("birthyear_numeric_range"), 1, pd.to_numeric(df["birthyear_value_invalid"], errors="coerce").fillna(0)
    ).astype(int)
    df["season_value_invalid"] = np.where(
        rule.eq("season_numeric_range"), 1, pd.to_numeric(df["season_value_invalid"], errors="coerce").fillna(0)
    ).astype(int)
    df["position_value_invalid"] = np.where(
        rule.eq("position_domain"), 1, pd.to_numeric(df["position_value_invalid"], errors="coerce").fillna(0)
    ).astype(int)
    df["birthyear_season_age_conflict"] = np.where(
        rule.eq("birthyear_season_age_consistency"), 1, pd.to_numeric(df["birthyear_season_age_conflict"], errors="coerce").fillna(0)
    ).astype(int)
    df["team_context_conflict"] = np.where(
        rule.isin(SOCCER_CONTEXT_RULE_TYPES), 1, pd.to_numeric(df["team_context_conflict"], errors="coerce").fillna(0)
    ).astype(int)
    df["format_rule_signal"] = np.where(
        rule.isin(SOCCER_FORMAT_RULE_TYPES), 1, pd.to_numeric(df["format_rule_signal"], errors="coerce").fillna(0)
    ).astype(int)
    df["fd_like_signal"] = np.where(
        rule.isin(SOCCER_FD_RULE_TYPES), 1, pd.to_numeric(df["fd_like_signal"], errors="coerce").fillna(0)
    ).astype(int)

    # 补齐 attribution bucket
    missing_attr = df["soccer_attribution_bucket"].isin(
        ["", "missing", "unknown", "unknown_attribution", "nan", "None"]
    )
    df.loc[missing_attr & (df["birthyear_value_invalid"] == 1), "soccer_attribution_bucket"] = "birthyear_invalid"
    df.loc[missing_attr & (df["season_value_invalid"] == 1), "soccer_attribution_bucket"] = "season_invalid"
    df.loc[missing_attr & (df["position_value_invalid"] == 1), "soccer_attribution_bucket"] = "position_invalid"
    df.loc[missing_attr & (pd.to_numeric(df["position_needs_canonicalization"], errors="coerce").fillna(0).astype(int) == 1), "soccer_attribution_bucket"] = "position_canonicalization"
    df.loc[missing_attr & (df["birthyear_season_age_conflict"] == 1), "soccer_attribution_bucket"] = "birthyear_season_age_conflict"
    df.loc[missing_attr & (df["team_context_conflict"] == 1), "soccer_attribution_bucket"] = "team_context_conflict"
    df.loc[missing_attr & (df["format_rule_signal"] == 1), "soccer_attribution_bucket"] = "format_pattern_conflict"
    df.loc[missing_attr & (df["fd_like_signal"] == 1), "soccer_attribution_bucket"] = "fd_like_conflict"
    df.loc[missing_attr & df["column"].isin(PRIMARY_TARGET_COLUMNS), "soccer_attribution_bucket"] = "primary_target_other"

    # 去重：同一个单元格可能被多个规则重复带出，保留 violation/conflict/posterior 更强的那条。
    # soccer v2 中如果同一 cell 有强 schema 和弱 FD，优先保留强 schema/domain/format/range 规则。
    df["_strong_sampling_signal"] = (
        df["main_rule_type"].isin(IMPORTANT_RULE_TYPES).astype(int) * 10
        + pd.to_numeric(df["strong_rule_count"], errors="coerce").fillna(0).clip(0, 5)
        + pd.to_numeric(df["soccer_domain_rule_count"], errors="coerce").fillna(0).clip(0, 5)
        + pd.to_numeric(df["soccer_format_rule_count"], errors="coerce").fillna(0).clip(0, 5)
        + pd.to_numeric(df["soccer_numeric_rule_count"], errors="coerce").fillna(0).clip(0, 5)
        + pd.to_numeric(df["soccer_consistency_rule_count"], errors="coerce").fillna(0).clip(0, 5)
    )

    df = df.sort_values(
        by=[
            "row_id", "column",
            "_strong_sampling_signal",
            "violation_count", "conflict_score", "posterior_error_probability",
        ],
        ascending=[True, True, False, False, False, False]
    ).drop_duplicates(subset=["row_id", "column"], keep="first").reset_index(drop=True)

    df = df.drop(columns=["_strong_sampling_signal"], errors="ignore")
    return df


def load_detail_jsonl(path: str) -> Dict[Tuple[int, str], Dict[str, Any]]:
    detail_map: Dict[Tuple[int, str], Dict[str, Any]] = {}
    if not os.path.exists(path):
        return detail_map

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                key = (int(obj.get("row_id")), str(obj.get("column")))
                # 若重复，保留第一条，通常与扁平表去重后兼容
                if key not in detail_map:
                    detail_map[key] = obj
            except Exception:
                continue
    return detail_map


# ============================================================
# 4. 派生分桶和聚类特征
# ============================================================

def build_rarity_bucket(row: pd.Series) -> str:
    rank = safe_float(row.get("value_frequency_rank"), math.nan)
    freq = safe_float(row.get("value_frequency"), 0.0)
    col = safe_str(row.get("column"))

    if math.isnan(rank):
        return "rank_unknown"

    # name/surname/birthplace 是长尾实体列，低频不应天然等价于错误。
    if col in {"name", "surname", "birthplace", "team", "stadium", "manager", "city"}:
        if freq <= 1:
            return "entity_singleton"
        if rank <= 10:
            return "entity_top10"
        if rank <= 50:
            return "entity_mid_freq"
        return "entity_long_tail"

    if freq <= 1:
        return "singleton"
    if rank <= 3:
        return "top3"
    if rank <= 10:
        return "top10"
    if rank <= 30:
        return "mid_freq"
    return "long_tail"


def build_pattern_bucket(row: pd.Series) -> str:
    cur = safe_str(row.get("current_value_pattern"), "unknown")
    dom = safe_str(row.get("dominant_pattern"), "unknown")
    ratio = safe_float(row.get("dominant_pattern_ratio"), 0.0)
    rule = safe_str(row.get("main_rule_type"), "none")

    if cur in {"NULL", "missing", "unknown"}:
        return "null_or_unknown_pattern"

    if rule in SOCCER_FORMAT_RULE_TYPES:
        return "explicit_format_rule"

    if dom not in {"missing", "unknown", "None"} and cur == dom:
        return "dominant_pattern_match"
    if ratio >= 0.70:
        return "dominant_pattern_mismatch"
    return "pattern_other"


def build_neighbor_bucket(row: pd.Series) -> str:
    ratio = safe_float(row.get("neighbor_majority_ratio"), math.nan)
    value = safe_str(row.get("value"), "__NULL__")
    neigh = safe_str(row.get("neighbor_majority_value"), "__NULL__")

    if math.isnan(ratio):
        return "no_neighbor_signal"
    if value == neigh:
        if ratio >= 0.80:
            return "strong_support_current"
        return "weak_support_current"
    else:
        if ratio >= 0.80:
            return "strong_support_other"
        return "weak_support_other"


def build_soccer_signal_bucket(row: pd.Series) -> str:
    col = safe_str(row.get("column"))
    rule = safe_str(row.get("main_rule_type"), "none")
    flags = safe_str(row.get("soccer_evidence_flags"), "")
    score = safe_float(row.get("soccer_consistency_score"), 0.0)
    domain_valid = normalize_boolish(row.get("domain_valid"))
    attr = safe_str(row.get("soccer_attribution_bucket"), "unknown_attribution")

    # 优先使用 summarize_full_contexts_soccer_v2.py 已经生成的 attribution bucket
    if attr not in {"", "missing", "unknown", "unknown_attribution", "nan", "None"}:
        return f"attr_{attr}"

    if col in {"name", "surname", "birthplace"} and rule in {"not_null", "high_risk_missing"}:
        return f"{col}_missing_signal"

    if col in {"name", "surname", "birthplace", "manager", "team", "city", "stadium"} and rule in SOCCER_FORMAT_RULE_TYPES:
        return f"{col}_format_signal"

    if col == "position" and (domain_valid == "false" or rule == "position_domain"):
        return "position_invalid_domain"

    if col in {"birthyear", "season"} and rule in {
        "birthyear_numeric_range",
        "season_numeric_range",
        "birthyear_season_age_consistency",
    }:
        return f"{col}_numeric_or_age_signal"

    if col in {"team", "stadium", "city", "manager"} and (
        rule in SOCCER_CONTEXT_RULE_TYPES or score > 0
    ):
        return "club_context_consistency_signal"

    if "invalid" in flags or "conflict" in flags:
        return "soccer_profile_conflict"

    if rule in SOCCER_FD_RULE_TYPES:
        return "fd_like_signal"

    if rule in SOCCER_EVIDENCE_ONLY_FD_RULE_TYPES:
        return "evidence_only_soft_fd_signal"

    if rule in {"rare_value", "rare_pattern", "global_dominant_value"}:
        return "weak_rarity_or_global_signal"

    if safe_float(row.get("strong_rule_count"), 0.0) > 0:
        return "strong_rule_signal"

    return "generic_signal"


def add_sampling_buckets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rarity_bucket"] = out.apply(build_rarity_bucket, axis=1)
    out["pattern_bucket"] = out.apply(build_pattern_bucket, axis=1)
    out["neighbor_bucket"] = out.apply(build_neighbor_bucket, axis=1)
    out["soccer_signal_bucket"] = out.apply(build_soccer_signal_bucket, axis=1)

    bucket_parts = []
    for _, row in out.iterrows():
        parts = []
        if USE_COLUMN_IN_BUCKET:
            parts.append(f"col={safe_str(row.get('column'))}")
        if USE_RULE_IN_BUCKET:
            parts.append(f"rule={safe_str(row.get('main_rule_type'), 'none')}")
        if USE_SOCCER_SIGNAL_IN_BUCKET:
            parts.append(f"sig={safe_str(row.get('soccer_signal_bucket'), 'generic_signal')}")
        parts.append(f"nb={safe_str(row.get('neighbor_bucket'))}")
        parts.append(f"pat={safe_str(row.get('pattern_bucket'))}")
        bucket_parts.append("||".join(parts))

    out["bucket_id"] = bucket_parts
    return out


def build_numeric_feature_columns(df: pd.DataFrame) -> List[str]:
    candidate_cols = [
        "violation_count", "conflict_score", "value_frequency", "value_frequency_rank",
        "neighbor_majority_ratio", "prior_error_probability", "posterior_error_probability",
        "strong_rule_count", "candidate_generation_rule_count", "fd_like_count", "evidence_only_fd_count",
        "context_rule_count", "global_rule_count", "rare_value_count", "typo_rule_count",
        "pattern_rule_count", "schema_rule_count", "soccer_domain_rule_count",
        "soccer_format_rule_count", "soccer_numeric_rule_count", "soccer_consistency_rule_count",
        "row_missing_count", "row_non_missing_count", "year_value", "row_birthyear_int",
        "row_season_int", "row_age_at_season", "soccer_profile_inconsistency_count",
        "target_is_primary_column", "target_is_context_only_column",
        "birthyear_value_invalid", "season_value_invalid", "position_value_invalid",
        "position_needs_canonicalization", "birthyear_season_age_conflict", "team_context_conflict",
        "format_rule_signal", "fd_like_signal",
        "adult_profile_inconsistency_count", "dominant_pattern_ratio",
    ]
    return [c for c in candidate_cols if c in df.columns]


def add_category_codes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cat_cols = [
        "column", "semantic_type", "detected_type", "main_rule_type", "main_usage_role",
        "rarity_bucket", "pattern_bucket", "neighbor_bucket", "soccer_value_bucket",
        "soccer_signal_bucket", "soccer_attribution_bucket",
        "canonical_position", "row_position_canonical",
    ]
    for c in cat_cols:
        if c in out.columns:
            out[f"{c}__code"] = out[c].fillna(MISSING_TEXT).astype(str).astype("category").cat.codes
    return out


def build_feature_matrix_for_bucket(bucket_df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    X = bucket_df[feature_cols].copy()
    for c in feature_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if len(X) == 0:
        return np.empty((0, len(feature_cols)))
    scaler = StandardScaler()
    return scaler.fit_transform(X.values.astype(float))


def choose_cluster_count(n: int) -> int:
    if n < MIN_BUCKET_SIZE_FOR_CLUSTER:
        return 1
    k = int(math.ceil(n / TARGET_CLUSTER_SIZE))
    k = max(2, min(MAX_CLUSTERS_PER_BUCKET, k, n))
    return k


# ============================================================
# 5. 聚类
# ============================================================

def cluster_candidates(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    work = add_sampling_buckets(df)
    work = add_category_codes(work)

    numeric_cols = build_numeric_feature_columns(work)
    code_cols = [c for c in work.columns if c.endswith("__code")]
    feature_cols = numeric_cols + code_cols

    if not feature_cols:
        raise RuntimeError("没有可用聚类特征，请检查 candidate_llm_contexts_flat_soccer.csv 字段。")

    clustered_parts = []
    summary_rows = []

    for bucket_id, bucket_df in work.groupby("bucket_id", dropna=False):
        bucket_df = bucket_df.copy()
        n = len(bucket_df)
        k = choose_cluster_count(n)

        if n < MIN_BUCKET_SIZE_FOR_CLUSTER or k <= 1:
            bucket_df["cluster_id"] = 0
            bucket_df["dist_to_center"] = 0.0
            bucket_df["cluster_size"] = n
        else:
            X = build_feature_matrix_for_bucket(bucket_df, feature_cols)
            km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
            labels = km.fit_predict(X)
            centers = km.cluster_centers_
            dists = np.linalg.norm(X - centers[labels], axis=1)
            bucket_df["cluster_id"] = labels.astype(int)
            bucket_df["dist_to_center"] = dists.astype(float)
            size_map = Counter(labels.tolist())
            bucket_df["cluster_size"] = bucket_df["cluster_id"].map(size_map).astype(int)

        for cid, sub in bucket_df.groupby("cluster_id", dropna=False):
            summary_rows.append({
                "bucket_id": bucket_id,
                "cluster_id": int(cid),
                "bucket_size": int(n),
                "cluster_size": int(len(sub)),
                "column_top": safe_str(sub["column"].mode().iloc[0]) if len(sub) else MISSING_TEXT,
                "main_rule_type_top": safe_str(sub["main_rule_type"].mode().iloc[0]) if len(sub) else MISSING_TEXT,
                "soccer_signal_bucket_top": safe_str(sub["soccer_signal_bucket"].mode().iloc[0]) if len(sub) else MISSING_TEXT,
                "mean_posterior": float(pd.to_numeric(sub["posterior_error_probability"], errors="coerce").fillna(0).mean()),
                "mean_conflict_score": float(pd.to_numeric(sub["conflict_score"], errors="coerce").fillna(0).mean()),
            })

        clustered_parts.append(bucket_df)

    clustered_df = pd.concat(clustered_parts, ignore_index=True) if clustered_parts else work
    summary_df = pd.DataFrame(summary_rows)

    return clustered_df, summary_df


# ============================================================
# 6. 预算采样
# ============================================================

def _add_sample(samples: Dict[Tuple[int, str], Dict[str, Any]], row: pd.Series, role: str):
    key = (int(row["row_id"]), str(row["column"]))
    if key not in samples:
        obj = row.to_dict()
        obj["sample_role"] = role
        obj["is_sampled"] = 1
        samples[key] = obj
    else:
        old_role = safe_str(samples[key].get("sample_role"), "")
        if role not in old_role.split("+"):
            samples[key]["sample_role"] = old_role + "+" + role if old_role else role


def compute_priority_score(df: pd.DataFrame) -> pd.Series:
    """采样优先级：强 schema/domain/format/range/consistency 优先，弱 FD/rare 降权。"""
    posterior = pd.to_numeric(df.get("posterior_error_probability"), errors="coerce").fillna(0.0)
    conflict = pd.to_numeric(df.get("conflict_score"), errors="coerce").fillna(0.0)
    strong = pd.to_numeric(df.get("strong_rule_count"), errors="coerce").fillna(0.0)
    soccer_score = pd.to_numeric(df.get("soccer_consistency_score"), errors="coerce").fillna(0.0)

    domain = pd.to_numeric(df.get("soccer_domain_rule_count"), errors="coerce").fillna(0.0)
    fmt = pd.to_numeric(df.get("soccer_format_rule_count"), errors="coerce").fillna(0.0)
    numeric = pd.to_numeric(df.get("soccer_numeric_rule_count"), errors="coerce").fillna(0.0)
    cons = pd.to_numeric(df.get("soccer_consistency_rule_count"), errors="coerce").fillna(0.0)

    birthyear_season = (
        pd.to_numeric(df.get("birthyear_value_invalid", 0), errors="coerce").fillna(0.0)
        + pd.to_numeric(df.get("season_value_invalid", 0), errors="coerce").fillna(0.0)
        + pd.to_numeric(df.get("birthyear_season_age_conflict", 0), errors="coerce").fillna(0.0)
    )
    position_signal = (
        pd.to_numeric(df.get("position_value_invalid", 0), errors="coerce").fillna(0.0)
        + pd.to_numeric(df.get("position_needs_canonicalization", 0), errors="coerce").fillna(0.0)
    )
    format_signal = pd.to_numeric(df.get("format_rule_signal", 0), errors="coerce").fillna(0.0)

    rule = df["main_rule_type"].astype(str) if "main_rule_type" in df.columns else pd.Series(["none"] * len(df), index=df.index)
    col = df["column"].astype(str) if "column" in df.columns else pd.Series(["missing"] * len(df), index=df.index)

    important_rule_bonus = rule.isin(IMPORTANT_RULE_TYPES).astype(float) * 2.0
    focus_col_bonus = col.isin(HIGH_RECALL_FOCUS_COLUMNS).astype(float) * 1.6
    weak_penalty = rule.isin(WEAK_RULE_TYPES).astype(float) * 2.2
    soft_fd_penalty = rule.isin(SOCCER_EVIDENCE_ONLY_FD_RULE_TYPES).astype(float) * 3.0

    score = (
        posterior * 2.0
        + conflict * 0.08
        + strong * 1.5
        + soccer_score * 1.1
        + domain * 2.0
        + fmt * 2.0
        + numeric * 2.0
        + cons * 1.6
        + birthyear_season * 2.0
        + position_signal * 2.0
        + format_signal * 1.8
        + important_rule_bonus
        + focus_col_bonus
        - weak_penalty
        - soft_fd_penalty
    )
    return score.astype(float)


def _sample_cluster_representatives(sub: pd.DataFrame, samples: Dict[Tuple[int, str], Dict[str, Any]]):
    sub = sub.copy()
    sub["_posterior"] = pd.to_numeric(sub["posterior_error_probability"], errors="coerce").fillna(0.0)
    sub["_dist"] = pd.to_numeric(sub["dist_to_center"], errors="coerce").fillna(0.0)
    sub["_conflict"] = pd.to_numeric(sub["conflict_score"], errors="coerce").fillna(0.0)
    sub["_priority_score"] = compute_priority_score(sub)

    n = len(sub)
    if n <= SMALL_BUCKET_FULL_SAMPLE_THRESHOLD:
        for _, row in sub.iterrows():
            _add_sample(samples, row, "small_cluster_full")
        return

    selected_indices = set()

    if SAMPLE_CENTER:
        idx = sub.sort_values(["_dist", "_priority_score", "_posterior"], ascending=[True, False, False]).index[0]
        selected_indices.add(idx)

    if SAMPLE_OUTLIER and len(selected_indices) < MAX_SAMPLE_PER_CLUSTER:
        idx = sub.sort_values(["_dist", "_priority_score", "_posterior"], ascending=[False, False, False]).index[0]
        selected_indices.add(idx)

    if SAMPLE_HIGH_POSTERIOR and len(selected_indices) < MAX_SAMPLE_PER_CLUSTER:
        idx = sub.sort_values(["_priority_score", "_posterior", "_conflict"], ascending=[False, False, False]).index[0]
        selected_indices.add(idx)

    if SAMPLE_LOW_POSTERIOR and len(selected_indices) < MAX_SAMPLE_PER_CLUSTER:
        idx = sub.sort_values(["_posterior", "_priority_score", "_conflict"], ascending=[True, False, False]).index[0]
        selected_indices.add(idx)

    if SAMPLE_BOUNDARY and len(selected_indices) < MAX_SAMPLE_PER_CLUSTER:
        median_dist = sub["_dist"].median()
        tmp = sub.assign(_boundary=(sub["_dist"] - median_dist).abs())
        idx = tmp.sort_values(["_boundary", "_priority_score", "_posterior"], ascending=[True, False, False]).index[0]
        selected_indices.add(idx)

    selected_indices = list(selected_indices)[:MAX_SAMPLE_PER_CLUSTER]

    center_idx = sub.sort_values(["_dist", "_priority_score", "_posterior"], ascending=[True, False, False]).index[0]
    outlier_idx = sub.sort_values(["_dist", "_priority_score", "_posterior"], ascending=[False, False, False]).index[0]
    high_idx = sub.sort_values(["_priority_score", "_posterior", "_conflict"], ascending=[False, False, False]).index[0]

    for idx in selected_indices:
        row = sub.loc[idx]
        role_parts = []
        if idx == center_idx:
            role_parts.append("center")
        if idx == outlier_idx:
            role_parts.append("outlier")
        if idx == high_idx:
            role_parts.append("high_priority")
        if not role_parts:
            role_parts.append("boundary")
        _add_sample(samples, row, "+".join(role_parts))


def add_quota_samples(clustered_df: pd.DataFrame, samples: Dict[Tuple[int, str], Dict[str, Any]]):
    """先额外加入列/规则保底样本，再统一走 cap，避免强信号列被全局预算挤掉。"""
    df = clustered_df.copy()
    df["_priority_score"] = compute_priority_score(df)

    # 1) 每个有候选的列保底，尤其 name/surname/birthplace。
    for col, sub in df.groupby("column", dropna=False):
        n_min = MIN_SAMPLE_PER_ACTIVE_COLUMN
        if str(col) in HIGH_RECALL_FOCUS_COLUMNS:
            n_min = max(n_min, 12)
        n = min(n_min, len(sub))
        for _, row in sub.sort_values("_priority_score", ascending=False).head(n).iterrows():
            _add_sample(samples, row, "column_quota")

    # 2) 重要规则保底。
    for rule in IMPORTANT_RULE_TYPES:
        sub = df[df["main_rule_type"].astype(str) == rule]
        if len(sub) == 0:
            continue
        n = min(MIN_SAMPLE_PER_IMPORTANT_RULE, len(sub))
        for _, row in sub.sort_values("_priority_score", ascending=False).head(n).iterrows():
            _add_sample(samples, row, "important_rule_quota")

    # 3) soccer attribution bucket 保底，确保每类强信号至少被 LLM 看见。
    if "soccer_attribution_bucket" in df.columns:
        important_buckets = {
            "birthyear_invalid",
            "season_invalid",
            "position_invalid",
            "position_canonicalization",
            "birthyear_season_age_conflict",
            "format_pattern_conflict",
            "primary_target_other",
        }
        for bucket in important_buckets:
            sub = df[df["soccer_attribution_bucket"].astype(str) == bucket]
            if len(sub) == 0:
                continue
            n = min(4, len(sub))
            for _, row in sub.sort_values("_priority_score", ascending=False).head(n).iterrows():
                _add_sample(samples, row, "attribution_bucket_quota")


def enforce_weak_rule_cap(sample_df: pd.DataFrame) -> pd.DataFrame:
    if len(sample_df) == 0 or "main_rule_type" not in sample_df.columns:
        return sample_df

    df = sample_df.copy()
    df["_priority_score"] = compute_priority_score(df)
    weak_mask = df["main_rule_type"].astype(str).isin(WEAK_RULE_TYPES)

    weak_df = df[weak_mask].copy()
    strong_df = df[~weak_mask].copy()

    if len(weak_df) <= MAX_WEAK_RULE_SAMPLES:
        return df.drop(columns=["_priority_score"], errors="ignore").reset_index(drop=True)

    weak_keep = weak_df.sort_values("_priority_score", ascending=False).head(MAX_WEAK_RULE_SAMPLES)
    out = pd.concat([strong_df, weak_keep], ignore_index=True)
    return out.drop(columns=["_priority_score"], errors="ignore").reset_index(drop=True)


def enforce_bucket_cap(sample_df: pd.DataFrame) -> pd.DataFrame:
    if len(sample_df) == 0:
        return sample_df

    parts = []
    for bucket_id, sub in sample_df.groupby("bucket_id", dropna=False):
        if len(sub) <= MAX_SAMPLE_PER_BUCKET:
            parts.append(sub)
            continue
        sub = sub.copy()
        sub["_priority_score"] = compute_priority_score(sub)
        parts.append(sub.sort_values("_priority_score", ascending=False).head(MAX_SAMPLE_PER_BUCKET).drop(columns=["_priority_score"]))
    return pd.concat(parts, ignore_index=True)


def enforce_column_cap(sample_df: pd.DataFrame) -> pd.DataFrame:
    if len(sample_df) == 0:
        return sample_df

    parts = []
    for col, sub in sample_df.groupby("column", dropna=False):
        cap = MAX_SAMPLE_PER_COLUMN.get(str(col), DEFAULT_MAX_SAMPLE_PER_COLUMN)
        if len(sub) <= cap:
            parts.append(sub)
            continue

        sub = sub.copy()
        sub["_priority_score"] = compute_priority_score(sub)

        # 保留多样性：优先每个规则类型留一点，再按分数补满。
        keep_idx = []
        for _, g in sub.groupby("main_rule_type", dropna=False):
            # 弱规则每类最多先保留 1 条，强规则每类最多 2 条
            rule_name = safe_str(g["main_rule_type"].iloc[0], "none")
            per_rule_keep = 1 if rule_name in WEAK_RULE_TYPES else 2
            keep_idx.extend(g.sort_values("_priority_score", ascending=False).head(per_rule_keep).index.tolist())

        keep_idx = list(dict.fromkeys(keep_idx))
        remain = cap - len(keep_idx)
        if remain > 0:
            extra = sub.drop(index=keep_idx, errors="ignore").sort_values("_priority_score", ascending=False).head(remain).index.tolist()
            keep_idx.extend(extra)

        keep_idx = keep_idx[:cap]
        parts.append(sub.loc[keep_idx].drop(columns=["_priority_score"]))

    return pd.concat(parts, ignore_index=True)


def enforce_global_budget(sample_df: pd.DataFrame) -> pd.DataFrame:
    if len(sample_df) <= GLOBAL_SAMPLE_BUDGET:
        return sample_df.reset_index(drop=True)

    df = sample_df.copy()
    df["_priority_score"] = compute_priority_score(df)

    keep_indices = []

    # 1) 每个有候选的列保底
    for col, sub in df.groupby("column", dropna=False):
        n_min = MIN_SAMPLE_PER_ACTIVE_COLUMN
        if str(col) in HIGH_RECALL_FOCUS_COLUMNS:
            n_min = max(n_min, 12)
        n = min(n_min, len(sub))
        keep_indices.extend(sub.sort_values("_priority_score", ascending=False).head(n).index.tolist())

    keep_indices = list(dict.fromkeys(keep_indices))

    # 2) 重要规则类型保底
    for rule in IMPORTANT_RULE_TYPES:
        sub = df[df["main_rule_type"].astype(str) == rule]
        if len(sub) == 0:
            continue
        n = min(MIN_SAMPLE_PER_IMPORTANT_RULE, len(sub))
        extra = sub.drop(index=keep_indices, errors="ignore").sort_values("_priority_score", ascending=False).head(n).index.tolist()
        keep_indices.extend(extra)
        keep_indices = list(dict.fromkeys(keep_indices))

    # 3) 剩余按优先级补齐
    remain = GLOBAL_SAMPLE_BUDGET - len(keep_indices)
    if remain > 0:
        extra = df.drop(index=keep_indices, errors="ignore").sort_values("_priority_score", ascending=False).head(remain).index.tolist()
        keep_indices.extend(extra)

    keep_indices = keep_indices[:GLOBAL_SAMPLE_BUDGET]
    return df.loc[keep_indices].drop(columns=["_priority_score"]).reset_index(drop=True)


def sample_candidates(clustered_df: pd.DataFrame) -> pd.DataFrame:
    samples: Dict[Tuple[int, str], Dict[str, Any]] = {}

    # 先按 bucket 和 cluster 采代表样本
    for (_, _), sub in clustered_df.groupby(["bucket_id", "cluster_id"], dropna=False):
        _sample_cluster_representatives(sub, samples)

    # 再补列/规则/attribution 保底样本
    add_quota_samples(clustered_df, samples)

    sample_df = pd.DataFrame(list(samples.values()))
    if len(sample_df) == 0:
        return sample_df

    # 控制弱规则、单桶、单列、全局预算
    sample_df = enforce_weak_rule_cap(sample_df)
    sample_df = enforce_bucket_cap(sample_df)
    sample_df = enforce_column_cap(sample_df)
    sample_df = enforce_global_budget(sample_df)

    sample_df["is_sampled"] = 1
    return sample_df.reset_index(drop=True)


# ============================================================
# 7. JSONL 输出
# ============================================================

def export_sampled_jsonl(sample_df: pd.DataFrame, detail_map: Dict[Tuple[int, str], Dict[str, Any]], output_jsonl: str):
    ensure_parent_dir(output_jsonl)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for _, row in sample_df.iterrows():
            key = (int(row["row_id"]), str(row["column"]))
            obj = detail_map.get(key)
            if obj is None:
                obj = {
                    "row_id": int(row["row_id"]),
                    "column": str(row["column"]),
                    "value": None if pd.isna(row.get("value")) else row.get("value"),
                    "llm_context_text": "",
                }

            obj = dict(obj)
            obj["sampling_info"] = {
                "bucket_id": safe_str(row.get("bucket_id")),
                "cluster_id": safe_int(row.get("cluster_id"), -1),
                "cluster_size": safe_int(row.get("cluster_size"), 0),
                "dist_to_center": safe_float(row.get("dist_to_center"), 0.0),
                "sample_role": safe_str(row.get("sample_role"), "sampled"),
                "soccer_signal_bucket": safe_str(row.get("soccer_signal_bucket"), "generic_signal"),
                "soccer_attribution_bucket": safe_str(row.get("soccer_attribution_bucket"), "unknown_attribution"),
                "rarity_bucket": safe_str(row.get("rarity_bucket"), "unknown"),
                "pattern_bucket": safe_str(row.get("pattern_bucket"), "unknown"),
                "neighbor_bucket": safe_str(row.get("neighbor_bucket"), "unknown"),
                "target_is_primary_column": safe_int(row.get("target_is_primary_column"), 0),
                "target_is_context_only_column": safe_int(row.get("target_is_context_only_column"), 0),
                "main_rule_type": safe_str(row.get("main_rule_type"), "none"),
            }
            f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")


# ============================================================
# 8. 统计输出
# ============================================================

def build_sample_summary(clustered_df: pd.DataFrame, sample_df: pd.DataFrame) -> Dict[str, Any]:
    summary = {
        "input_candidates": int(len(clustered_df)),
        "sampled_candidates": int(len(sample_df)),
        "sample_ratio": round(len(sample_df) / len(clustered_df), 6) if len(clustered_df) > 0 else 0.0,
        "global_sample_budget": int(GLOBAL_SAMPLE_BUDGET),
        "max_weak_rule_samples": int(MAX_WEAK_RULE_SAMPLES),
        "column_distribution_input": clustered_df["column"].value_counts(dropna=False).to_dict() if "column" in clustered_df.columns else {},
        "column_distribution_sampled": sample_df["column"].value_counts(dropna=False).to_dict() if len(sample_df) and "column" in sample_df.columns else {},
        "rule_distribution_input": clustered_df["main_rule_type"].value_counts(dropna=False).to_dict() if "main_rule_type" in clustered_df.columns else {},
        "rule_distribution_sampled": sample_df["main_rule_type"].value_counts(dropna=False).to_dict() if len(sample_df) and "main_rule_type" in sample_df.columns else {},
        "soccer_signal_distribution_sampled": sample_df["soccer_signal_bucket"].value_counts(dropna=False).to_dict() if len(sample_df) and "soccer_signal_bucket" in sample_df.columns else {},
        "soccer_attribution_distribution_sampled": sample_df["soccer_attribution_bucket"].value_counts(dropna=False).to_dict() if len(sample_df) and "soccer_attribution_bucket" in sample_df.columns else {},
        "bucket_count": int(clustered_df["bucket_id"].nunique()) if "bucket_id" in clustered_df.columns else 0,
        "cluster_count": int(clustered_df[["bucket_id", "cluster_id"]].drop_duplicates().shape[0]) if {"bucket_id", "cluster_id"}.issubset(clustered_df.columns) else 0,
    }

    if len(sample_df) and "main_rule_type" in sample_df.columns:
        summary["weak_rule_sampled_count"] = int(sample_df["main_rule_type"].astype(str).isin(WEAK_RULE_TYPES).sum())
    else:
        summary["weak_rule_sampled_count"] = 0

    return summary


# ============================================================
# 9. 主流程
# ============================================================

def main():
    print("[1/6] Loading soccer context summary ...")
    df = load_summary_csv(INPUT_SUMMARY_CSV)
    print(f"  input candidates after dedup = {len(df)}")

    print("[2/6] Loading detailed JSONL context map ...")
    detail_map = load_detail_jsonl(INPUT_DETAIL_JSONL)
    print(f"  detail JSONL records = {len(detail_map)}")

    print("[3/6] Clustering candidates ...")
    clustered_df, bucket_summary_df = cluster_candidates(df)
    clustered_df["is_sampled"] = 0
    clustered_df["sample_role"] = ""
    print(f"  clustered rows = {len(clustered_df)}")
    print(f"  buckets = {clustered_df['bucket_id'].nunique()}")

    print("[4/6] Budgeted representative sampling ...")
    sample_df = sample_candidates(clustered_df)

    if len(sample_df) > 0:
        sampled_keys = set((int(r["row_id"]), str(r["column"])) for _, r in sample_df.iterrows())
        key_series = list(zip(clustered_df["row_id"].astype(int), clustered_df["column"].astype(str)))
        clustered_df["is_sampled"] = [1 if k in sampled_keys else 0 for k in key_series]
        role_map = {(int(r["row_id"]), str(r["column"])): r.get("sample_role", "sampled") for _, r in sample_df.iterrows()}
        clustered_df["sample_role"] = [role_map.get(k, "") for k in key_series]

    print(f"  sampled rows = {len(sample_df)}")

    print("[5/6] Exporting CSV/JSONL ...")
    clustered_df.to_csv(OUTPUT_CLUSTERED_CSV, index=False, encoding="utf-8-sig")
    sample_df.to_csv(OUTPUT_SAMPLED_CSV, index=False, encoding="utf-8-sig")
    bucket_summary_df.to_csv(OUTPUT_BUCKET_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    export_sampled_jsonl(sample_df, detail_map, OUTPUT_SAMPLED_JSONL)

    print("[6/6] Exporting sampling summary ...")
    summary = build_sample_summary(clustered_df, sample_df)
    with open(OUTPUT_SAMPLE_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n===== DONE =====")
    print(f"Clustered CSV       : {OUTPUT_CLUSTERED_CSV}")
    print(f"Sampled CSV         : {OUTPUT_SAMPLED_CSV}")
    print(f"Sampled JSONL       : {OUTPUT_SAMPLED_JSONL}")
    print(f"Bucket summary CSV  : {OUTPUT_BUCKET_SUMMARY_CSV}")
    print(f"Sample summary JSON : {OUTPUT_SAMPLE_SUMMARY_JSON}")
    print("\n===== STATS =====")
    print(f"Input candidates : {len(clustered_df)}")
    print(f"Sampled          : {len(sample_df)}")
    if len(clustered_df) > 0:
        print(f"Sample ratio     : {len(sample_df) / len(clustered_df):.6f}")
    if len(sample_df) > 0:
        print("\nSampled by column:")
        print(sample_df["column"].value_counts(dropna=False).to_string())
        print("\nSampled by main_rule_type:")
        print(sample_df["main_rule_type"].value_counts(dropna=False).head(30).to_string())
        print("\nSampled by soccer_signal_bucket:")
        print(sample_df["soccer_signal_bucket"].value_counts(dropna=False).to_string())
        if "soccer_attribution_bucket" in sample_df.columns:
            print("\nSampled by soccer_attribution_bucket:")
            print(sample_df["soccer_attribution_bucket"].value_counts(dropna=False).head(30).to_string())


if __name__ == "__main__":
    main()

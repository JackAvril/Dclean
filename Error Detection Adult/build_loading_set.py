import pandas as pd
import numpy as np


# ============================================================
# 1. 用户配置区
# ============================================================

# 输入1：整个候选范围的上下文结果（adult 新版）
INPUT_FULL_CANDIDATE_CSV = "candidate_llm_contexts_flat_adult.csv"

# 输入2：聚类采样结果（字段更完整，用来回填 bucket / cluster 等信息）
INPUT_CLUSTERED_CSV = "candidate_clustered_adult.csv"

# 输出：和训练格式尽量对齐的推理输入文件
OUTPUT_INFER_READY_CSV = "candidate_infer_ready_adult.csv"


# 一些阈值
HIGH_NEIGHBOR_CONSISTENCY_TH = 0.80
RARE_VALUE_FREQ_TH = 5
RARE_VALUE_RANK_TH = 10

# 默认值
DEFAULT_LABEL_SOURCE = "inference"
DEFAULT_SAMPLE_WEIGHT = 1.0
DEFAULT_PROPAGATION_CONFIDENCE = 1.0
DEFAULT_IS_OUTSIDE_CANDIDATE = 0

DEFAULT_PATTERN_BUCKET = "unknown"
DEFAULT_RARITY_BUCKET = "unknown"
DEFAULT_NEIGHBOR_BUCKET = "unknown"
DEFAULT_DOMAIN_BUCKET = "unknown"
DEFAULT_ADULT_CONSISTENCY_BUCKET = "unknown"
DEFAULT_AGE_SAMPLING_BUCKET = "unknown"
DEFAULT_HOURS_SAMPLING_BUCKET = "unknown"

# 兼容旧 flights/time 字段
DEFAULT_TIME_WINDOW_BUCKET = "adult_not_applicable"
DEFAULT_TIME_VALUE_BUCKET = "adult_not_applicable"

DEFAULT_BUCKET_ID = "full_candidate"
DEFAULT_CLUSTER_ID = -1
DEFAULT_DIST_TO_CENTER = 0.0
DEFAULT_SAMPLE_ROLE = "full_candidate"
DEFAULT_IS_SAMPLED = 0

# adult 任务中的字段
ADULT_COLUMNS = {
    "age",
    "workclass",
    "education",
    "maritalstatus",
    "occupation",
    "relationship",
    "race",
    "sex",
    "hoursperweek",
    "country",
    "income",
}

ADULT_LONG_TAIL_COLUMNS = {"country", "occupation", "workclass"}


# ============================================================
# 2. 基础工具函数
# ============================================================

def safe_float(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x, default=0):
    try:
        if pd.isna(x):
            return default
        return int(float(x))
    except Exception:
        return default


def safe_str(x, default=""):
    if pd.isna(x):
        return default
    return str(x)


def normalize_value(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.lower() in {
        "", "empty", "null", "none", "nan", "na", "n/a", "missing", "unk",
        "unknown", "?", "not available", "contact airline"
    }:
        return None
    return s


def maybe_drop_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [c for c in df.columns if not str(c).lower().startswith("unnamed:")]
    return df[keep_cols].copy()


def normalize_text_series_for_compare(s: pd.Series) -> pd.Series:
    return s.fillna("__NULL__").astype(str).str.strip()


def parse_bool_like(x, default=None):
    if pd.isna(x):
        return default
    s = str(x).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    return default


# ============================================================
# 3. 读取与预处理
# ============================================================

def load_full_candidate_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df = maybe_drop_unnamed_columns(df)

    required_cols = [
        "row_id", "column", "value", "semantic_type",
        "violation_count", "conflict_score",
        "value_frequency", "value_frequency_rank",
        "neighbor_majority_value", "neighbor_majority_ratio",
        "prior_error_probability", "posterior_error_probability",
        "main_rule_type", "main_usage_role",
        "strong_rule_count", "candidate_generation_rule_count",
        "fd_like_count", "context_rule_count", "global_rule_count",
        "rare_value_count", "typo_rule_count",
        "pattern_rule_count", "schema_rule_count",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"全候选输入文件缺少必要列: {missing}")

    optional_cols_with_defaults = {
        "detected_type": "unknown",

        # adult 专属字段
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

        # bucket 字段
        "pattern_bucket": "",
        "rarity_bucket": "",
        "neighbor_bucket": "",
        "domain_bucket": "",
        "adult_consistency_bucket": "",
        "age_sampling_bucket": "",
        "hours_sampling_bucket": "",
        "bucket_id": "",
        "cluster_id": np.nan,
        "dist_to_center": np.nan,
        "sample_role": "",
        "is_sampled": np.nan,
        "signal_priority": 0.0,

        # 兼容旧 flights/time 字段
        "time_window_rule_count": 0,
        "time_value_minutes": np.nan,
        "time_window_bucket": "",
        "time_value_bucket": "",
    }
    for col, default_val in optional_cols_with_defaults.items():
        if col not in df.columns:
            df[col] = default_val

    df["row_id"] = df["row_id"].apply(lambda x: safe_int(x, -1))
    df["column"] = df["column"].apply(lambda x: safe_str(x, ""))
    df["value"] = df["value"].map(normalize_value)
    df["neighbor_majority_value"] = df["neighbor_majority_value"].map(normalize_value)

    category_cols = [
        "semantic_type",
        "detected_type",
        "main_rule_type",
        "main_usage_role",
        "pattern_bucket",
        "rarity_bucket",
        "neighbor_bucket",
        "domain_bucket",
        "adult_consistency_bucket",
        "age_sampling_bucket",
        "hours_sampling_bucket",
        "adult_value_bucket",
        "canonical_income",
        "domain_valid",
        "adult_evidence_flags",
        "row_income_canonical",
        "time_window_bucket",
        "time_value_bucket",
        "bucket_id",
        "sample_role",
    ]
    for c in category_cols:
        df[c] = df[c].apply(lambda x: safe_str(x, ""))

    numeric_int_cols = [
        "violation_count",
        "strong_rule_count",
        "candidate_generation_rule_count",
        "fd_like_count",
        "context_rule_count",
        "global_rule_count",
        "rare_value_count",
        "typo_rule_count",
        "pattern_rule_count",
        "schema_rule_count",
        "adult_domain_rule_count",
        "adult_format_rule_count",
        "adult_consistency_rule_count",
        "adult_profile_inconsistency_count",
        "row_missing_count",
        "row_non_missing_count",
        "time_window_rule_count",
        "cluster_id",
        "is_sampled",
    ]
    for c in numeric_int_cols:
        df[c] = df[c].apply(lambda x: safe_int(x, 0))

    numeric_float_cols = [
        "conflict_score",
        "value_frequency",
        "value_frequency_rank",
        "neighbor_majority_ratio",
        "prior_error_probability",
        "posterior_error_probability",
        "age_lower",
        "age_upper",
        "hours_lower",
        "hours_upper",
        "education_order",
        "adult_consistency_score",
        "row_age_lower",
        "row_age_upper",
        "row_hours_lower",
        "row_hours_upper",
        "row_education_order",
        "time_value_minutes",
        "dist_to_center",
        "signal_priority",
    ]
    for c in numeric_float_cols:
        df[c] = df[c].apply(lambda x: safe_float(x, 0.0))

    return df


def load_clustered_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df = maybe_drop_unnamed_columns(df)

    needed_cols = [
        "row_id", "column",

        "pattern_bucket", "rarity_bucket", "neighbor_bucket",
        "domain_bucket", "adult_consistency_bucket",
        "age_sampling_bucket", "hours_sampling_bucket",
        "bucket_id", "cluster_id", "dist_to_center", "sample_role", "is_sampled",
        "signal_priority",

        "rare_value_count", "typo_rule_count", "pattern_rule_count", "schema_rule_count",
        "adult_domain_rule_count", "adult_format_rule_count", "adult_consistency_rule_count",
        "adult_value_bucket", "domain_valid", "adult_consistency_score",
        "adult_profile_inconsistency_count",

        # 兼容旧字段
        "time_window_rule_count", "time_value_minutes",
        "time_window_bucket", "time_value_bucket",
    ]
    for col in needed_cols:
        if col not in df.columns:
            df[col] = np.nan

    df["row_id"] = df["row_id"].apply(lambda x: safe_int(x, -1))
    df["column"] = df["column"].apply(lambda x: safe_str(x, ""))

    df = df[needed_cols].copy()
    df = df.drop_duplicates(subset=["row_id", "column"], keep="first").copy()
    return df


# ============================================================
# 4. merge 聚类字段
# ============================================================

def merge_cluster_fields(full_df: pd.DataFrame, clustered_df: pd.DataFrame) -> pd.DataFrame:
    df = full_df.merge(
        clustered_df,
        on=["row_id", "column"],
        how="left",
        suffixes=("", "_from_cluster")
    )

    merge_cols = [
        "pattern_bucket", "rarity_bucket", "neighbor_bucket",
        "domain_bucket", "adult_consistency_bucket",
        "age_sampling_bucket", "hours_sampling_bucket",
        "bucket_id", "cluster_id", "dist_to_center", "sample_role", "is_sampled",
        "signal_priority",

        "rare_value_count", "typo_rule_count", "pattern_rule_count", "schema_rule_count",
        "adult_domain_rule_count", "adult_format_rule_count", "adult_consistency_rule_count",
        "adult_value_bucket", "domain_valid", "adult_consistency_score",
        "adult_profile_inconsistency_count",

        "time_window_rule_count", "time_value_minutes",
        "time_window_bucket", "time_value_bucket",
    ]

    for col in merge_cols:
        cluster_col = f"{col}_from_cluster"
        if cluster_col in df.columns:
            if col not in df.columns:
                df[col] = df[cluster_col]
            else:
                df[col] = df[col].where(
                    ~df[col].isna() & (df[col].astype(str).str.strip() != ""),
                    df[cluster_col]
                )
            df = df.drop(columns=[cluster_col])

    return df


# ============================================================
# 5. 重建 / 补齐 bucket 字段
# ============================================================

def build_rarity_bucket(row: pd.Series) -> str:
    vf = safe_float(row.get("value_frequency"), np.nan)
    vr = safe_float(row.get("value_frequency_rank"), np.nan)

    if not pd.isna(vr):
        if vr <= 3:
            return "very_common"
        elif vr <= 10:
            return "common"

    if not pd.isna(vf):
        if vf <= 1:
            return "singleton"
        elif vf <= 2:
            return "very_rare"
        elif vf <= 5:
            return "rare"

    return "mid_frequency"


def build_neighbor_bucket(row: pd.Series) -> str:
    nmr = safe_float(row.get("neighbor_majority_ratio"), np.nan)
    majority_val = normalize_value(row.get("neighbor_majority_value"))
    current_val = normalize_value(row.get("value"))

    if pd.isna(nmr):
        return "no_neighbor_signal"

    same = (majority_val == current_val)

    if not same:
        if nmr >= 0.8:
            return "strong_support_other"
        elif nmr >= 0.5:
            return "medium_support_other"
        else:
            return "weak_support_other"
    else:
        if nmr >= 0.8:
            return "strong_support_current"
        else:
            return "weak_support_current"


def build_domain_bucket(row: pd.Series) -> str:
    if "domain_valid" not in row.index:
        return "unknown_domain"

    val = row.get("domain_valid")
    if pd.isna(val) or str(val).strip() == "":
        return "domain_not_applicable"

    s = str(val).strip().lower()
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


def build_time_window_bucket(row: pd.Series) -> str:
    return DEFAULT_TIME_WINDOW_BUCKET


def build_time_value_bucket(row: pd.Series) -> str:
    return DEFAULT_TIME_VALUE_BUCKET


def fill_bucket_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "pattern_bucket" not in df.columns:
        df["pattern_bucket"] = DEFAULT_PATTERN_BUCKET
    df["pattern_bucket"] = df["pattern_bucket"].fillna(DEFAULT_PATTERN_BUCKET).replace("", DEFAULT_PATTERN_BUCKET)

    if "rarity_bucket" not in df.columns:
        df["rarity_bucket"] = np.nan
    df["rarity_bucket"] = df["rarity_bucket"].fillna("")
    need_fill_rarity = df["rarity_bucket"].eq("")
    df.loc[need_fill_rarity, "rarity_bucket"] = df[need_fill_rarity].apply(build_rarity_bucket, axis=1)

    if "neighbor_bucket" not in df.columns:
        df["neighbor_bucket"] = np.nan
    df["neighbor_bucket"] = df["neighbor_bucket"].fillna("")
    need_fill_neighbor = df["neighbor_bucket"].eq("")
    df.loc[need_fill_neighbor, "neighbor_bucket"] = df[need_fill_neighbor].apply(build_neighbor_bucket, axis=1)

    if "domain_bucket" not in df.columns:
        df["domain_bucket"] = np.nan
    df["domain_bucket"] = df["domain_bucket"].fillna("")
    need_fill_domain = df["domain_bucket"].eq("")
    df.loc[need_fill_domain, "domain_bucket"] = df[need_fill_domain].apply(build_domain_bucket, axis=1)

    if "adult_consistency_bucket" not in df.columns:
        df["adult_consistency_bucket"] = np.nan
    df["adult_consistency_bucket"] = df["adult_consistency_bucket"].fillna("")
    need_fill_adult = df["adult_consistency_bucket"].eq("")
    df.loc[need_fill_adult, "adult_consistency_bucket"] = df[need_fill_adult].apply(
        build_adult_consistency_bucket, axis=1
    )

    if "age_sampling_bucket" not in df.columns:
        df["age_sampling_bucket"] = np.nan
    df["age_sampling_bucket"] = df["age_sampling_bucket"].fillna("")
    need_fill_age = df["age_sampling_bucket"].eq("")
    df.loc[need_fill_age, "age_sampling_bucket"] = df[need_fill_age].apply(
        build_age_sampling_bucket, axis=1
    )

    if "hours_sampling_bucket" not in df.columns:
        df["hours_sampling_bucket"] = np.nan
    df["hours_sampling_bucket"] = df["hours_sampling_bucket"].fillna("")
    need_fill_hours = df["hours_sampling_bucket"].eq("")
    df.loc[need_fill_hours, "hours_sampling_bucket"] = df[need_fill_hours].apply(
        build_hours_sampling_bucket, axis=1
    )

    if "time_window_bucket" not in df.columns:
        df["time_window_bucket"] = np.nan
    df["time_window_bucket"] = df["time_window_bucket"].fillna("")
    need_fill_time_window = df["time_window_bucket"].eq("")
    df.loc[need_fill_time_window, "time_window_bucket"] = df[need_fill_time_window].apply(
        build_time_window_bucket, axis=1
    )

    if "time_value_bucket" not in df.columns:
        df["time_value_bucket"] = np.nan
    df["time_value_bucket"] = df["time_value_bucket"].fillna("")
    need_fill_time_value = df["time_value_bucket"].eq("")
    df.loc[need_fill_time_value, "time_value_bucket"] = df[need_fill_time_value].apply(
        build_time_value_bucket, axis=1
    )

    if "bucket_id" not in df.columns:
        df["bucket_id"] = DEFAULT_BUCKET_ID
    df["bucket_id"] = df["bucket_id"].fillna(DEFAULT_BUCKET_ID).replace("", DEFAULT_BUCKET_ID)

    if "cluster_id" not in df.columns:
        df["cluster_id"] = DEFAULT_CLUSTER_ID
    df["cluster_id"] = df["cluster_id"].apply(lambda x: safe_int(x, DEFAULT_CLUSTER_ID))

    if "dist_to_center" not in df.columns:
        df["dist_to_center"] = DEFAULT_DIST_TO_CENTER
    df["dist_to_center"] = df["dist_to_center"].apply(lambda x: safe_float(x, DEFAULT_DIST_TO_CENTER))

    if "sample_role" not in df.columns:
        df["sample_role"] = DEFAULT_SAMPLE_ROLE
    df["sample_role"] = df["sample_role"].fillna(DEFAULT_SAMPLE_ROLE).replace("", DEFAULT_SAMPLE_ROLE)

    if "is_sampled" not in df.columns:
        df["is_sampled"] = DEFAULT_IS_SAMPLED
    df["is_sampled"] = df["is_sampled"].apply(lambda x: safe_int(x, DEFAULT_IS_SAMPLED))

    return df


# ============================================================
# 6. 补训练 / 推理对齐字段
# ============================================================

def add_alignment_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "label" not in df.columns:
        df["label"] = ""
    if "label_binary" not in df.columns:
        df["label_binary"] = ""
    if "label_source" not in df.columns:
        df["label_source"] = DEFAULT_LABEL_SOURCE
    if "sample_weight" not in df.columns:
        df["sample_weight"] = DEFAULT_SAMPLE_WEIGHT
    if "is_outside_candidate" not in df.columns:
        df["is_outside_candidate"] = DEFAULT_IS_OUTSIDE_CANDIDATE
    if "propagation_confidence" not in df.columns:
        df["propagation_confidence"] = DEFAULT_PROPAGATION_CONFIDENCE

    for col, default_val in [
        ("error_type", ""),
        ("reason_short", ""),
        ("reason_detailed", ""),
        ("suggested_correct_value", ""),
        ("needs_human_review", False),
        ("evidence_used", "")
    ]:
        if col not in df.columns:
            df[col] = default_val

    return df


# ============================================================
# 7. 补训练时需要的派生特征
# ============================================================

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 先确保关键计数字段存在并是数值
    numeric_count_cols = [
        "strong_rule_count",
        "candidate_generation_rule_count",
        "fd_like_count",
        "context_rule_count",
        "global_rule_count",
        "rare_value_count",
        "typo_rule_count",
        "pattern_rule_count",
        "schema_rule_count",
        "adult_domain_rule_count",
        "adult_format_rule_count",
        "adult_consistency_rule_count",
        "time_window_rule_count",
    ]
    for c in numeric_count_cols:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    strong_cnt = df["strong_rule_count"].astype(float)
    cand_cnt = df["candidate_generation_rule_count"].astype(float)
    total_cnt = strong_cnt + cand_cnt

    df["candidate_rule_ratio"] = np.where(total_cnt > 0, cand_cnt / (total_cnt + 1e-8), 0.0)

    df["has_neighbor_signal"] = df["neighbor_majority_ratio"].apply(
        lambda x: 0 if pd.isna(x) or safe_float(x, 0.0) <= 0.0 else 1
    )

    df["high_neighbor_consistency"] = df["neighbor_majority_ratio"].apply(
        lambda x: 1 if safe_float(x, 0.0) >= HIGH_NEIGHBOR_CONSISTENCY_TH else 0
    )

    def _is_rare(row):
        vf = safe_float(row["value_frequency"], 0.0)
        vr = safe_float(row["value_frequency_rank"], np.nan)

        # adult 的 country / occupation / workclass 是天然长尾，降低 rare 标志强度；
        # 这里仍保留二值特征，但阈值更保守。
        col = str(row.get("column", ""))
        if col in ADULT_LONG_TAIL_COLUMNS:
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

    def _neighbor_disagree(row):
        mv = normalize_value(row["neighbor_majority_value"])
        cv = normalize_value(row["value"])
        nmr = safe_float(row["neighbor_majority_ratio"], 0.0)
        if mv is None or cv is None or nmr <= 0.0:
            return 0
        return 1 if mv != cv else 0

    df["neighbor_disagree"] = df.apply(_neighbor_disagree, axis=1)

    value_series = normalize_text_series_for_compare(df["value"])
    neighbor_series = normalize_text_series_for_compare(df["neighbor_majority_value"])
    df["is_equal_to_neighbor_majority"] = (value_series == neighbor_series).astype(int)

    ratio = pd.to_numeric(df["neighbor_majority_ratio"], errors="coerce").fillna(0.0)
    df["neighbor_agreement_score"] = np.where(
        df["is_equal_to_neighbor_majority"] == 1,
        ratio,
        -ratio
    ).astype(float)

    df["rule_total_count"] = (
        df["strong_rule_count"].astype(float) +
        df["candidate_generation_rule_count"].astype(float)
    )

    df["strong_rule_ratio"] = np.where(
        df["rule_total_count"] > 0,
        df["strong_rule_count"].astype(float) / (df["rule_total_count"] + 1e-8),
        0.0
    )

    df["context_rule_ratio"] = np.where(
        df["rule_total_count"] > 0,
        df["context_rule_count"].astype(float) / (df["rule_total_count"] + 1e-8),
        0.0
    )

    df["fd_like_ratio"] = np.where(
        df["rule_total_count"] > 0,
        df["fd_like_count"].astype(float) / (df["rule_total_count"] + 1e-8),
        0.0
    )

    df["global_rule_ratio"] = np.where(
        df["rule_total_count"] > 0,
        df["global_rule_count"].astype(float) / (df["rule_total_count"] + 1e-8),
        0.0
    )

    df["adult_rule_total_count"] = (
        df["adult_domain_rule_count"].astype(float) +
        df["adult_format_rule_count"].astype(float) +
        df["adult_consistency_rule_count"].astype(float)
    )

    df["adult_domain_rule_ratio"] = np.where(
        df["adult_rule_total_count"] > 0,
        df["adult_domain_rule_count"].astype(float) / (df["adult_rule_total_count"] + 1e-8),
        0.0
    )

    df["adult_format_rule_ratio"] = np.where(
        df["adult_rule_total_count"] > 0,
        df["adult_format_rule_count"].astype(float) / (df["adult_rule_total_count"] + 1e-8),
        0.0
    )

    df["adult_consistency_rule_ratio"] = np.where(
        df["adult_rule_total_count"] > 0,
        df["adult_consistency_rule_count"].astype(float) / (df["adult_rule_total_count"] + 1e-8),
        0.0
    )

    df["rule_strength_sum"] = (
        1.00 * df["strong_rule_count"].astype(float)
        + 0.25 * df["candidate_generation_rule_count"].astype(float)
        + 0.45 * df["fd_like_count"].astype(float)
        + 0.35 * df["context_rule_count"].astype(float)
        + 0.15 * df["global_rule_count"].astype(float)
        + 0.20 * df["rare_value_count"].astype(float)
        + 0.35 * df["typo_rule_count"].astype(float)
        + 0.45 * df["pattern_rule_count"].astype(float)
        + 0.45 * df["schema_rule_count"].astype(float)
        + 0.90 * df["adult_domain_rule_count"].astype(float)
        + 0.85 * df["adult_format_rule_count"].astype(float)
        + 0.75 * df["adult_consistency_rule_count"].astype(float)
        + 0.00 * df["time_window_rule_count"].astype(float)
    )

    df["candidate_rule_dominant"] = np.where(
        df["candidate_generation_rule_count"].astype(float) >
        df["strong_rule_count"].astype(float),
        "yes",
        "no"
    )

    # adult 专属派生特征
    df["domain_invalid_flag"] = df["domain_bucket"].astype(str).eq("domain_invalid").astype(int)
    df["adult_consistency_flag"] = df["adult_consistency_bucket"].astype(str).isin([
        "strong_adult_consistency_signal",
        "adult_consistency_flag",
        "weak_adult_consistency_signal",
    ]).astype(int)

    df["high_adult_consistency_score"] = (
        pd.to_numeric(df["adult_consistency_score"], errors="coerce").fillna(0.0) >= 1.0
    ).astype(int)

    df["has_adult_domain_signal"] = (df["adult_domain_rule_count"].astype(float) > 0).astype(int)
    df["has_adult_format_signal"] = (df["adult_format_rule_count"].astype(float) > 0).astype(int)
    df["has_adult_consistency_signal"] = (df["adult_consistency_rule_count"].astype(float) > 0).astype(int)

    # 年龄/工时解析相关
    for c in ["age_lower", "age_upper", "hours_lower", "hours_upper", "education_order"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df["age_span"] = np.maximum(0.0, df["age_upper"].astype(float) - df["age_lower"].astype(float))
    df["hours_span"] = np.maximum(0.0, df["hours_upper"].astype(float) - df["hours_lower"].astype(float))

    df["is_minor_age_bucket"] = (
        (df["age_upper"].astype(float) > 0) & (df["age_upper"].astype(float) < 18)
    ).astype(int)

    df["is_high_hours_bucket"] = (
        ((df["hours_lower"].astype(float) + df["hours_upper"].astype(float)) / 2.0) > 60
    ).astype(int)

    return df


# ============================================================
# 8. 重排列顺序
# ============================================================

def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    preferred_cols = [
        "row_id",
        "column",
        "value",

        "label",
        "label_binary",
        "label_source",
        "sample_weight",
        "is_outside_candidate",
        "propagation_confidence",

        "semantic_type",
        "detected_type",
        "violation_count",
        "conflict_score",
        "value_frequency",
        "value_frequency_rank",

        "neighbor_majority_value",
        "neighbor_majority_ratio",
        "prior_error_probability",
        "posterior_error_probability",

        "main_rule_type",
        "main_usage_role",
        "strong_rule_count",
        "candidate_generation_rule_count",
        "fd_like_count",
        "context_rule_count",
        "global_rule_count",
        "rare_value_count",
        "typo_rule_count",
        "pattern_rule_count",
        "schema_rule_count",

        "adult_domain_rule_count",
        "adult_format_rule_count",
        "adult_consistency_rule_count",
        "age_lower",
        "age_upper",
        "hours_lower",
        "hours_upper",
        "canonical_income",
        "domain_valid",
        "education_order",
        "adult_value_bucket",
        "adult_profile_inconsistency_count",
        "adult_consistency_score",
        "adult_evidence_flags",

        "row_age_lower",
        "row_age_upper",
        "row_hours_lower",
        "row_hours_upper",
        "row_education_order",
        "row_income_canonical",
        "row_missing_count",
        "row_non_missing_count",

        # 兼容旧字段
        "time_window_rule_count",
        "time_value_minutes",

        "candidate_rule_ratio",
        "has_neighbor_signal",
        "high_neighbor_consistency",
        "is_rare_value",
        "neighbor_disagree",
        "is_equal_to_neighbor_majority",
        "neighbor_agreement_score",
        "rule_total_count",
        "strong_rule_ratio",
        "context_rule_ratio",
        "fd_like_ratio",
        "global_rule_ratio",
        "adult_rule_total_count",
        "adult_domain_rule_ratio",
        "adult_format_rule_ratio",
        "adult_consistency_rule_ratio",
        "rule_strength_sum",
        "candidate_rule_dominant",

        "domain_invalid_flag",
        "adult_consistency_flag",
        "high_adult_consistency_score",
        "has_adult_domain_signal",
        "has_adult_format_signal",
        "has_adult_consistency_signal",
        "age_span",
        "hours_span",
        "is_minor_age_bucket",
        "is_high_hours_bucket",

        "pattern_bucket",
        "rarity_bucket",
        "neighbor_bucket",
        "domain_bucket",
        "adult_consistency_bucket",
        "age_sampling_bucket",
        "hours_sampling_bucket",

        # 兼容旧字段
        "time_window_bucket",
        "time_value_bucket",

        "bucket_id",
        "cluster_id",
        "dist_to_center",
        "sample_role",
        "is_sampled",
        "signal_priority",

        "error_type",
        "reason_short",
        "reason_detailed",
        "suggested_correct_value",
        "needs_human_review",
        "evidence_used",
    ]

    exist = [c for c in preferred_cols if c in df.columns]
    extra = [c for c in df.columns if c not in exist]
    return df[exist + extra].copy()


# ============================================================
# 9. 主流程
# ============================================================

def main():
    print("[1/5] 读取全候选上下文文件...")
    full_df = load_full_candidate_csv(INPUT_FULL_CANDIDATE_CSV)

    print("[2/5] 读取聚类采样结果文件...")
    clustered_df = load_clustered_csv(INPUT_CLUSTERED_CSV)

    print("[3/5] merge 聚类字段...")
    df = merge_cluster_fields(full_df, clustered_df)

    print("[4/5] 补 bucket / 对齐列 / 派生特征...")
    df = fill_bucket_fields(df)
    df = add_alignment_columns(df)
    df = add_derived_features(df)

    print("[5/5] 输出推理对齐文件...")
    df = reorder_columns(df)
    df.to_csv(OUTPUT_INFER_READY_CSV, index=False, encoding="utf-8-sig")

    print(f"[OK] 推理对齐文件已保存: {OUTPUT_INFER_READY_CSV}")
    print(f"[OK] 行数: {len(df)}")
    print(f"[OK] 列数: {len(df.columns)}")

    print("\n===== 预览前5行 =====")
    preview_cols = [
        "row_id", "column", "value",
        "semantic_type",
        "violation_count", "conflict_score",
        "pattern_bucket", "rarity_bucket", "neighbor_bucket",
        "domain_bucket", "adult_consistency_bucket",
        "age_sampling_bucket", "hours_sampling_bucket",
        "bucket_id", "cluster_id", "dist_to_center",
        "candidate_rule_ratio", "has_neighbor_signal",
        "high_neighbor_consistency", "is_rare_value",
        "is_equal_to_neighbor_majority", "neighbor_agreement_score",
        "rule_total_count", "adult_rule_total_count",
        "rule_strength_sum",
        "domain_invalid_flag", "adult_consistency_flag",
        "label_source", "is_outside_candidate"
    ]
    preview_cols = [c for c in preview_cols if c in df.columns]
    print(df[preview_cols].head().to_string(index=False))


if __name__ == "__main__":
    main()

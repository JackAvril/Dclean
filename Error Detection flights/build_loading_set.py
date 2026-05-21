import pandas as pd
import numpy as np


# ============================================================
# 1. 用户配置区
# ============================================================

# 输入1：整个候选范围的上下文结果（flights 新版）
INPUT_FULL_CANDIDATE_CSV = "candidate_llm_contexts_flat_flights.csv"

# 输入2：聚类采样结果（字段更完整，用来回填 bucket / cluster 等信息）
INPUT_CLUSTERED_CSV = "candidate_clustered_flights.csv"

# 输出：和训练格式尽量对齐的推理输入文件
OUTPUT_INFER_READY_CSV = "candidate_infer_ready_flights.csv"


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
DEFAULT_TIME_WINDOW_BUCKET = "unknown"
DEFAULT_TIME_VALUE_BUCKET = "unknown"
DEFAULT_BUCKET_ID = "full_candidate"
DEFAULT_CLUSTER_ID = -1
DEFAULT_DIST_TO_CENTER = 0.0
DEFAULT_SAMPLE_ROLE = "full_candidate"
DEFAULT_IS_SAMPLED = 0


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
    if s.lower() in {"", "empty", "null", "none", "nan", "na", "n/a", "missing", "unk"}:
        return None
    return s


def maybe_drop_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [c for c in df.columns if not str(c).lower().startswith("unnamed:")]
    return df[keep_cols].copy()


def normalize_text_series_for_compare(s: pd.Series) -> pd.Series:
    return s.fillna("__NULL__").astype(str).str.strip()


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
        "time_window_rule_count", "time_value_minutes",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"全候选输入文件缺少必要列: {missing}")

    optional_cols_with_defaults = {
        "pattern_bucket": "",
        "rarity_bucket": "",
        "neighbor_bucket": "",
        "time_window_bucket": "",
        "time_value_bucket": "",
        "bucket_id": "",
        "cluster_id": np.nan,
        "dist_to_center": np.nan,
        "sample_role": "",
        "is_sampled": np.nan,
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
        "main_rule_type",
        "main_usage_role",
        "pattern_bucket",
        "rarity_bucket",
        "neighbor_bucket",
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
        "time_value_minutes",
        "neighbor_majority_ratio",
        "prior_error_probability",
        "posterior_error_probability",
        "dist_to_center",
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
        "time_window_bucket", "time_value_bucket",
        "bucket_id", "cluster_id", "dist_to_center", "sample_role", "is_sampled",
        "rare_value_count", "typo_rule_count", "pattern_rule_count", "schema_rule_count",
        "time_window_rule_count", "time_value_minutes",
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

    for col in [
        "pattern_bucket",
        "rarity_bucket",
        "neighbor_bucket",
        "time_window_bucket",
        "time_value_bucket",
        "bucket_id",
        "cluster_id",
        "dist_to_center",
        "sample_role",
        "is_sampled",
        "rare_value_count",
        "typo_rule_count",
        "pattern_rule_count",
        "schema_rule_count",
        "time_window_rule_count",
        "time_value_minutes",
    ]:
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


def build_time_window_bucket(row: pd.Series) -> str:
    cnt = safe_int(row.get("time_window_rule_count"), 0)
    if cnt <= 0:
        return "no_time_window_signal"
    elif cnt == 1:
        return "single_time_window_signal"
    else:
        return "multi_time_window_signal"


def build_time_value_bucket(row: pd.Series) -> str:
    tv = safe_float(row.get("time_value_minutes"), np.nan)
    if pd.isna(tv):
        return "unparsed_time"

    hour = int(tv) // 60
    if 0 <= hour < 6:
        return "late_night"
    elif 6 <= hour < 12:
        return "morning"
    elif 12 <= hour < 18:
        return "afternoon"
    else:
        return "evening"


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

    if "time_window_bucket" not in df.columns:
        df["time_window_bucket"] = np.nan
    df["time_window_bucket"] = df["time_window_bucket"].fillna("")
    need_fill_time_window = df["time_window_bucket"].eq("")
    df.loc[need_fill_time_window, "time_window_bucket"] = df[need_fill_time_window].apply(build_time_window_bucket, axis=1)

    if "time_value_bucket" not in df.columns:
        df["time_value_bucket"] = np.nan
    df["time_value_bucket"] = df["time_value_bucket"].fillna("")
    need_fill_time_value = df["time_value_bucket"].eq("")
    df.loc[need_fill_time_value, "time_value_bucket"] = df[need_fill_time_value].apply(build_time_value_bucket, axis=1)

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

    strong_cnt = df["strong_rule_count"].astype(float)
    cand_cnt = df["candidate_generation_rule_count"].astype(float)
    total_cnt = strong_cnt + cand_cnt

    df["candidate_rule_ratio"] = np.where(total_cnt > 0, cand_cnt / total_cnt, 0.0)

    df["has_neighbor_signal"] = df["neighbor_majority_ratio"].apply(
        lambda x: 0 if pd.isna(x) or safe_float(x, 0.0) <= 0.0 else 1
    )

    df["high_neighbor_consistency"] = df["neighbor_majority_ratio"].apply(
        lambda x: 1 if safe_float(x, 0.0) >= HIGH_NEIGHBOR_CONSISTENCY_TH else 0
    )

    def _is_rare(row):
        vf = safe_float(row["value_frequency"], 0.0)
        vr = safe_float(row["value_frequency_rank"], np.nan)
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

    df["rule_strength_sum"] = (
        df["strong_rule_count"].astype(float)
        + df["fd_like_count"].astype(float)
        + df["context_rule_count"].astype(float)
        + df["global_rule_count"].astype(float)
        + df["rare_value_count"].astype(float)
        + df["typo_rule_count"].astype(float)
        + df["pattern_rule_count"].astype(float)
        + df["schema_rule_count"].astype(float)
        + df["time_window_rule_count"].astype(float)
    )

    df["candidate_rule_dominant"] = np.where(
        df["candidate_generation_rule_count"].astype(float) >
        df["strong_rule_count"].astype(float),
        "yes",
        "no"
    )

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
        "violation_count",
        "conflict_score",
        "value_frequency",
        "value_frequency_rank",
        "time_value_minutes",
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
        "time_window_rule_count",

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
        "rule_strength_sum",
        "candidate_rule_dominant",

        "pattern_bucket",
        "rarity_bucket",
        "neighbor_bucket",
        "time_window_bucket",
        "time_value_bucket",

        "bucket_id",
        "cluster_id",
        "dist_to_center",
        "sample_role",
        "is_sampled",

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
        "time_window_bucket", "time_value_bucket",
        "bucket_id", "cluster_id", "dist_to_center",
        "candidate_rule_ratio", "has_neighbor_signal",
        "high_neighbor_consistency", "is_rare_value",
        "is_equal_to_neighbor_majority", "neighbor_agreement_score",
        "rule_total_count", "strong_rule_ratio",
        "label_source", "is_outside_candidate"
    ]
    preview_cols = [c for c in preview_cols if c in df.columns]
    print(df[preview_cols].head().to_string(index=False))


if __name__ == "__main__":
    main()

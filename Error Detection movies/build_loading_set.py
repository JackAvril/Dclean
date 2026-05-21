import pandas as pd
import numpy as np


# ============================================================
# 1. 用户配置区
# ============================================================

# 输入1：整个候选范围的上下文结果（movies 新版）
INPUT_FULL_CANDIDATE_CSV = "candidate_llm_contexts_flat_movies.csv"

# 输入2：聚类采样结果（字段更完整，用来回填 bucket / cluster 等信息）
INPUT_CLUSTERED_CSV = "candidate_clustered_movies.csv"

# 输出：和训练格式尽量对齐的推理输入文件
OUTPUT_INFER_READY_CSV = "candidate_infer_ready_movies.csv"


# ============================================================
# 2. 阈值与默认值
# ============================================================

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

DEFAULT_MOVIE_RULE_BUCKET = "unknown"
DEFAULT_MOVIE_FIELD_BUCKET = "unknown"
DEFAULT_YEAR_BUCKET = "unknown"
DEFAULT_RELEASE_YEAR_GAP_BUCKET = "unknown"
DEFAULT_DURATION_BUCKET = "unknown"
DEFAULT_RATING_BUCKET = "unknown"
DEFAULT_COUNT_BUCKET = "unknown"
DEFAULT_REVIEW_RATIO_BUCKET = "unknown"
DEFAULT_ACTORS_CAST_OVERLAP_BUCKET = "unknown"
DEFAULT_LIST_BUCKET = "unknown"
DEFAULT_TEXT_BUCKET = "unknown"

DEFAULT_BUCKET_ID = "full_candidate"
DEFAULT_CLUSTER_ID = -1
DEFAULT_DIST_TO_CENTER = 0.0
DEFAULT_SAMPLE_ROLE = "full_candidate"
DEFAULT_IS_SAMPLED = 0
DEFAULT_SAMPLING_PRIORITY_SCORE = 0.0


MISSING_TOKENS = {
    "", "empty", "null", "none", "nan", "na", "n/a", "missing", "unk",
    "unknown", "not available", "not applicable", "tbd", "na."
}


# ============================================================
# 3. movies 字段配置
# ============================================================

MOVIES_NUMERIC_FEATURE_COLUMNS = [
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
    "context_rule_count",
    "global_rule_count",
    "rare_value_count",
    "typo_rule_count",
    "pattern_rule_count",
    "schema_rule_count",
    "movie_typed_rule_count",
    "movie_consistency_rule_count",

    "parsed_year",
    "parsed_release_year",
    "duration_minutes",
    "rating_value_float",
    "count_value_int",
    "review_total",
    "list_token_count",
    "has_mojibake_or_control_chars",

    "movie_row_year",
    "movie_row_release_year",
    "movie_row_release_year_gap",
    "movie_row_duration_minutes",
    "movie_row_rating_value_float",
    "movie_row_rating_count_int",
    "movie_row_review_total",
    "movie_row_review_to_rating_ratio",
    "movie_row_actors_cast_overlap",

    "cluster_id",
    "dist_to_center",
    "is_sampled",
    "sampling_priority_score",
]

MOVIES_CATEGORICAL_FEATURE_COLUMNS = [
    "semantic_type",
    "detected_type",
    "main_rule_type",
    "main_usage_role",

    "pattern_bucket",
    "rarity_bucket",
    "neighbor_bucket",
    "movie_rule_bucket",
    "movie_field_bucket",
    "year_bucket",
    "release_year_gap_bucket",
    "duration_bucket",
    "rating_bucket",
    "count_bucket",
    "review_ratio_bucket",
    "actors_cast_overlap_bucket",
    "list_bucket",
    "text_bucket",

    "parsed_release_country",
    "movie_row_release_country",
    "list_token_count_bucket",
    "text_length_bucket",

    "bucket_id",
    "sample_role",
]

MOVIES_CLUSTER_BACKFILL_COLUMNS = [
    "pattern_bucket",
    "rarity_bucket",
    "neighbor_bucket",
    "movie_rule_bucket",
    "movie_field_bucket",
    "year_bucket",
    "release_year_gap_bucket",
    "duration_bucket",
    "rating_bucket",
    "count_bucket",
    "review_ratio_bucket",
    "actors_cast_overlap_bucket",
    "list_bucket",
    "text_bucket",

    "bucket_id",
    "cluster_id",
    "dist_to_center",
    "sample_role",
    "is_sampled",
    "sampling_priority_score",

    "rare_value_count",
    "typo_rule_count",
    "pattern_rule_count",
    "schema_rule_count",
    "movie_typed_rule_count",
    "movie_consistency_rule_count",

    "parsed_year",
    "parsed_release_year",
    "parsed_release_country",
    "duration_minutes",
    "rating_value_float",
    "count_value_int",
    "review_total",
    "list_token_count",
    "list_token_count_bucket",
    "text_length_bucket",
    "has_mojibake_or_control_chars",

    "movie_row_year",
    "movie_row_release_year",
    "movie_row_release_country",
    "movie_row_release_year_gap",
    "movie_row_duration_minutes",
    "movie_row_rating_value_float",
    "movie_row_rating_count_int",
    "movie_row_review_total",
    "movie_row_review_to_rating_ratio",
    "movie_row_actors_cast_overlap",
]


# ============================================================
# 4. 基础工具函数
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
    if s.lower() in MISSING_TOKENS:
        return None
    return s


def maybe_drop_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [c for c in df.columns if not str(c).lower().startswith("unnamed:")]
    return df[keep_cols].copy()


def normalize_text_series_for_compare(s: pd.Series) -> pd.Series:
    return s.fillna("__NULL__").astype(str).str.strip()


def get_default_for_col(col: str):
    numeric_defaults = {
        "violation_count": 0,
        "conflict_score": 0.0,
        "value_frequency": 0,
        "value_frequency_rank": np.nan,
        "neighbor_majority_ratio": np.nan,
        "prior_error_probability": 0.0,
        "posterior_error_probability": 0.0,

        "strong_rule_count": 0,
        "candidate_generation_rule_count": 0,
        "fd_like_count": 0,
        "context_rule_count": 0,
        "global_rule_count": 0,
        "rare_value_count": 0,
        "typo_rule_count": 0,
        "pattern_rule_count": 0,
        "schema_rule_count": 0,
        "movie_typed_rule_count": 0,
        "movie_consistency_rule_count": 0,

        "parsed_year": np.nan,
        "parsed_release_year": np.nan,
        "duration_minutes": np.nan,
        "rating_value_float": np.nan,
        "count_value_int": np.nan,
        "review_total": np.nan,
        "list_token_count": np.nan,
        "has_mojibake_or_control_chars": 0,

        "movie_row_year": np.nan,
        "movie_row_release_year": np.nan,
        "movie_row_release_year_gap": np.nan,
        "movie_row_duration_minutes": np.nan,
        "movie_row_rating_value_float": np.nan,
        "movie_row_rating_count_int": np.nan,
        "movie_row_review_total": np.nan,
        "movie_row_review_to_rating_ratio": np.nan,
        "movie_row_actors_cast_overlap": np.nan,

        "cluster_id": DEFAULT_CLUSTER_ID,
        "dist_to_center": DEFAULT_DIST_TO_CENTER,
        "is_sampled": DEFAULT_IS_SAMPLED,
        "sampling_priority_score": DEFAULT_SAMPLING_PRIORITY_SCORE,
    }

    cat_defaults = {
        "semantic_type": "unknown",
        "detected_type": "unknown",
        "main_rule_type": "none",
        "main_usage_role": "none",

        "pattern_bucket": DEFAULT_PATTERN_BUCKET,
        "rarity_bucket": DEFAULT_RARITY_BUCKET,
        "neighbor_bucket": DEFAULT_NEIGHBOR_BUCKET,
        "movie_rule_bucket": DEFAULT_MOVIE_RULE_BUCKET,
        "movie_field_bucket": DEFAULT_MOVIE_FIELD_BUCKET,
        "year_bucket": DEFAULT_YEAR_BUCKET,
        "release_year_gap_bucket": DEFAULT_RELEASE_YEAR_GAP_BUCKET,
        "duration_bucket": DEFAULT_DURATION_BUCKET,
        "rating_bucket": DEFAULT_RATING_BUCKET,
        "count_bucket": DEFAULT_COUNT_BUCKET,
        "review_ratio_bucket": DEFAULT_REVIEW_RATIO_BUCKET,
        "actors_cast_overlap_bucket": DEFAULT_ACTORS_CAST_OVERLAP_BUCKET,
        "list_bucket": DEFAULT_LIST_BUCKET,
        "text_bucket": DEFAULT_TEXT_BUCKET,

        "parsed_release_country": "unknown",
        "movie_row_release_country": "unknown",
        "list_token_count_bucket": "unknown",
        "text_length_bucket": "unknown",

        "bucket_id": DEFAULT_BUCKET_ID,
        "sample_role": DEFAULT_SAMPLE_ROLE,
    }

    if col in numeric_defaults:
        return numeric_defaults[col]
    if col in cat_defaults:
        return cat_defaults[col]
    return np.nan


def ensure_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            df[col] = get_default_for_col(col)
    return df


# ============================================================
# 5. 读取与预处理
# ============================================================

def load_full_candidate_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df = maybe_drop_unnamed_columns(df)

    required_cols = [
        "row_id",
        "column",
        "value",
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
        "movie_typed_rule_count",
        "movie_consistency_rule_count",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"全候选输入文件缺少必要列: {missing}")

    optional_cols = [
        "parsed_year",
        "parsed_release_year",
        "parsed_release_country",
        "duration_minutes",
        "rating_value_float",
        "count_value_int",
        "review_total",
        "list_token_count",
        "list_token_count_bucket",
        "text_length_bucket",
        "has_mojibake_or_control_chars",

        "movie_row_year",
        "movie_row_release_year",
        "movie_row_release_country",
        "movie_row_release_year_gap",
        "movie_row_duration_minutes",
        "movie_row_rating_value_float",
        "movie_row_rating_count_int",
        "movie_row_review_total",
        "movie_row_review_to_rating_ratio",
        "movie_row_actors_cast_overlap",

        "pattern_bucket",
        "rarity_bucket",
        "neighbor_bucket",
        "movie_rule_bucket",
        "movie_field_bucket",
        "year_bucket",
        "release_year_gap_bucket",
        "duration_bucket",
        "rating_bucket",
        "count_bucket",
        "review_ratio_bucket",
        "actors_cast_overlap_bucket",
        "list_bucket",
        "text_bucket",

        "bucket_id",
        "cluster_id",
        "dist_to_center",
        "sample_role",
        "is_sampled",
        "sampling_priority_score",
    ]

    df = ensure_columns(df, optional_cols)

    df["row_id"] = df["row_id"].apply(lambda x: safe_int(x, -1))
    df["column"] = df["column"].apply(lambda x: safe_str(x, ""))
    df["value"] = df["value"].map(normalize_value)

    if "neighbor_majority_value" not in df.columns:
        df["neighbor_majority_value"] = None
    df["neighbor_majority_value"] = df["neighbor_majority_value"].map(normalize_value)

    category_cols = [
        "semantic_type",
        "detected_type",
        "main_rule_type",
        "main_usage_role",
        "pattern_bucket",
        "rarity_bucket",
        "neighbor_bucket",
        "movie_rule_bucket",
        "movie_field_bucket",
        "year_bucket",
        "release_year_gap_bucket",
        "duration_bucket",
        "rating_bucket",
        "count_bucket",
        "review_ratio_bucket",
        "actors_cast_overlap_bucket",
        "list_bucket",
        "text_bucket",
        "parsed_release_country",
        "movie_row_release_country",
        "list_token_count_bucket",
        "text_length_bucket",
        "bucket_id",
        "sample_role",
    ]

    for c in category_cols:
        if c not in df.columns:
            df[c] = get_default_for_col(c)
        df[c] = df[c].apply(lambda x: safe_str(x, get_default_for_col(c)))

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
        "movie_typed_rule_count",
        "movie_consistency_rule_count",
        "has_mojibake_or_control_chars",
        "cluster_id",
        "is_sampled",
    ]

    for c in numeric_int_cols:
        if c not in df.columns:
            df[c] = get_default_for_col(c)
        df[c] = df[c].apply(lambda x: safe_int(x, get_default_for_col(c)))

    numeric_float_cols = [
        "conflict_score",
        "value_frequency",
        "value_frequency_rank",
        "neighbor_majority_ratio",
        "prior_error_probability",
        "posterior_error_probability",

        "parsed_year",
        "parsed_release_year",
        "duration_minutes",
        "rating_value_float",
        "count_value_int",
        "review_total",
        "list_token_count",

        "movie_row_year",
        "movie_row_release_year",
        "movie_row_release_year_gap",
        "movie_row_duration_minutes",
        "movie_row_rating_value_float",
        "movie_row_rating_count_int",
        "movie_row_review_total",
        "movie_row_review_to_rating_ratio",
        "movie_row_actors_cast_overlap",

        "dist_to_center",
        "sampling_priority_score",
    ]

    for c in numeric_float_cols:
        if c not in df.columns:
            df[c] = get_default_for_col(c)
        df[c] = df[c].apply(lambda x: safe_float(x, get_default_for_col(c)))

    return df


def load_clustered_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df = maybe_drop_unnamed_columns(df)

    needed_cols = ["row_id", "column"] + MOVIES_CLUSTER_BACKFILL_COLUMNS
    for col in needed_cols:
        if col not in df.columns:
            df[col] = get_default_for_col(col)

    df["row_id"] = df["row_id"].apply(lambda x: safe_int(x, -1))
    df["column"] = df["column"].apply(lambda x: safe_str(x, ""))

    df = df[needed_cols].copy()
    df = df.drop_duplicates(subset=["row_id", "column"], keep="first").copy()
    return df


# ============================================================
# 6. merge 聚类字段
# ============================================================

def merge_cluster_fields(full_df: pd.DataFrame, clustered_df: pd.DataFrame) -> pd.DataFrame:
    df = full_df.merge(
        clustered_df,
        on=["row_id", "column"],
        how="left",
        suffixes=("", "_from_cluster")
    )

    for col in MOVIES_CLUSTER_BACKFILL_COLUMNS:
        cluster_col = f"{col}_from_cluster"
        if cluster_col in df.columns:
            if col not in df.columns:
                df[col] = df[cluster_col]
            else:
                left = df[col]
                right = df[cluster_col]

                left_as_str = left.astype(str)
                invalid_left = (
                    left.isna()
                    | left_as_str.str.strip().eq("")
                    | left_as_str.str.lower().isin({"nan", "none", "null"})
                )

                df[col] = df[col].where(~invalid_left, right)

            df = df.drop(columns=[cluster_col])

    return df


# ============================================================
# 7. 重建 / 补齐 bucket 字段
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


def build_movie_rule_bucket(row: pd.Series) -> str:
    typed = safe_int(row.get("movie_typed_rule_count"), 0)
    consistency = safe_int(row.get("movie_consistency_rule_count"), 0)

    if typed <= 0 and consistency <= 0:
        return "no_movie_rule_signal"
    if typed > 0 and consistency > 0:
        return "typed_and_consistency_signal"
    if consistency > 0:
        return "movie_consistency_signal"
    if typed == 1:
        return "single_movie_typed_signal"
    return "multi_movie_typed_signal"


def build_movie_field_bucket(row: pd.Series) -> str:
    col = str(row.get("column", ""))
    if col in {"Year", "Release Date"}:
        return "movie_time_metadata"
    if col in {"Duration"}:
        return "movie_duration_metadata"
    if col in {"RatingValue", "RatingCount", "ReviewCount"}:
        return "movie_rating_metadata"
    if col in {"Director", "Creator", "Actors", "Cast"}:
        return "movie_people_metadata"
    if col in {"Language", "Country", "Genre"}:
        return "movie_categorical_list_metadata"
    if col in {"Name"}:
        return "movie_title_metadata"
    if col in {"Filming Locations", "Description"}:
        return "movie_text_metadata"
    return "movie_other_metadata"


def build_year_bucket(row: pd.Series) -> str:
    y = safe_float(row.get("parsed_year"), np.nan)
    if pd.isna(y):
        y = safe_float(row.get("movie_row_year"), np.nan)

    if pd.isna(y):
        return "unparsed_year"

    y = int(y)
    if y < 1930:
        return "very_old_movie"
    if y < 1970:
        return "classic_movie"
    if y < 2000:
        return "late_20c_movie"
    if y < 2015:
        return "modern_movie"
    return "recent_movie"


def build_release_year_gap_bucket(row: pd.Series) -> str:
    gap = safe_float(row.get("movie_row_release_year_gap"), np.nan)
    if pd.isna(gap):
        return "no_release_year_gap"

    gap = int(gap)
    if gap == 0:
        return "same_release_year"
    if abs(gap) <= 1:
        return "near_release_year"
    if abs(gap) <= 2:
        return "allowed_release_year_gap"
    return "large_release_year_gap"


def build_duration_bucket(row: pd.Series) -> str:
    d = safe_float(row.get("duration_minutes"), np.nan)
    if pd.isna(d):
        d = safe_float(row.get("movie_row_duration_minutes"), np.nan)

    if pd.isna(d):
        return "unparsed_duration"

    d = float(d)
    if d < 40:
        return "short_duration"
    if d < 90:
        return "medium_short_duration"
    if d <= 150:
        return "standard_duration"
    if d <= 240:
        return "long_duration"
    return "extreme_duration"


def build_rating_bucket(row: pd.Series) -> str:
    r = safe_float(row.get("rating_value_float"), np.nan)
    if pd.isna(r):
        r = safe_float(row.get("movie_row_rating_value_float"), np.nan)

    if pd.isna(r):
        return "unparsed_rating"

    if r < 4:
        return "low_rating"
    if r < 6:
        return "mid_low_rating"
    if r < 8:
        return "mid_high_rating"
    return "high_rating"


def build_count_bucket(row: pd.Series) -> str:
    c = safe_float(row.get("count_value_int"), np.nan)
    if pd.isna(c):
        c = safe_float(row.get("movie_row_rating_count_int"), np.nan)

    if pd.isna(c):
        return "unparsed_count"

    if c <= 0:
        return "zero_count"
    if c < 100:
        return "small_count"
    if c < 10000:
        return "medium_count"
    if c < 100000:
        return "large_count"
    return "very_large_count"


def build_review_ratio_bucket(row: pd.Series) -> str:
    r = safe_float(row.get("movie_row_review_to_rating_ratio"), np.nan)
    if pd.isna(r):
        return "no_review_ratio"

    if r <= 0.01:
        return "tiny_review_ratio"
    if r <= 0.05:
        return "small_review_ratio"
    if r <= 0.20:
        return "medium_review_ratio"
    if r <= 0.50:
        return "large_review_ratio"
    return "extreme_review_ratio"


def build_actors_cast_overlap_bucket(row: pd.Series) -> str:
    o = safe_float(row.get("movie_row_actors_cast_overlap"), np.nan)
    if pd.isna(o):
        return "no_overlap_signal"

    if o >= 0.8:
        return "strong_actor_cast_overlap"
    if o >= 0.5:
        return "medium_actor_cast_overlap"
    if o > 0:
        return "weak_actor_cast_overlap"
    return "zero_actor_cast_overlap"


def build_list_bucket(row: pd.Series) -> str:
    existing = safe_str(row.get("list_token_count_bucket"), "")
    if existing and existing.lower() not in {"nan", "none", "null", "unknown"}:
        return existing

    n = safe_float(row.get("list_token_count"), np.nan)
    if pd.isna(n):
        return "not_list_field"

    n = int(n)
    if n == 0:
        return "empty_list"
    if n == 1:
        return "single"
    if n <= 3:
        return "short_list"
    if n <= 8:
        return "medium_list"
    return "long_list"


def build_text_bucket(row: pd.Series) -> str:
    tb = safe_str(row.get("text_length_bucket"), "")
    if tb and tb.lower() not in {"nan", "none", "null", "unknown"}:
        return tb
    return "unknown_text_length"


def fill_one_bucket(df: pd.DataFrame, col: str, default_val: str, builder_fn=None) -> pd.DataFrame:
    if col not in df.columns:
        df[col] = np.nan

    df[col] = df[col].fillna("")
    invalid = df[col].astype(str).str.strip().eq("") | df[col].astype(str).str.lower().isin({"nan", "none", "null"})

    if builder_fn is not None:
        df.loc[invalid, col] = df[invalid].apply(builder_fn, axis=1)
    else:
        df.loc[invalid, col] = default_val

    df[col] = df[col].fillna(default_val).replace("", default_val)
    return df


def fill_bucket_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = fill_one_bucket(df, "pattern_bucket", DEFAULT_PATTERN_BUCKET, None)
    df = fill_one_bucket(df, "rarity_bucket", DEFAULT_RARITY_BUCKET, build_rarity_bucket)
    df = fill_one_bucket(df, "neighbor_bucket", DEFAULT_NEIGHBOR_BUCKET, build_neighbor_bucket)

    df = fill_one_bucket(df, "movie_rule_bucket", DEFAULT_MOVIE_RULE_BUCKET, build_movie_rule_bucket)
    df = fill_one_bucket(df, "movie_field_bucket", DEFAULT_MOVIE_FIELD_BUCKET, build_movie_field_bucket)

    df = fill_one_bucket(df, "year_bucket", DEFAULT_YEAR_BUCKET, build_year_bucket)
    df = fill_one_bucket(df, "release_year_gap_bucket", DEFAULT_RELEASE_YEAR_GAP_BUCKET, build_release_year_gap_bucket)
    df = fill_one_bucket(df, "duration_bucket", DEFAULT_DURATION_BUCKET, build_duration_bucket)
    df = fill_one_bucket(df, "rating_bucket", DEFAULT_RATING_BUCKET, build_rating_bucket)
    df = fill_one_bucket(df, "count_bucket", DEFAULT_COUNT_BUCKET, build_count_bucket)
    df = fill_one_bucket(df, "review_ratio_bucket", DEFAULT_REVIEW_RATIO_BUCKET, build_review_ratio_bucket)
    df = fill_one_bucket(df, "actors_cast_overlap_bucket", DEFAULT_ACTORS_CAST_OVERLAP_BUCKET, build_actors_cast_overlap_bucket)
    df = fill_one_bucket(df, "list_bucket", DEFAULT_LIST_BUCKET, build_list_bucket)
    df = fill_one_bucket(df, "text_bucket", DEFAULT_TEXT_BUCKET, build_text_bucket)

    df = fill_one_bucket(df, "bucket_id", DEFAULT_BUCKET_ID, None)

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

    if "sampling_priority_score" not in df.columns:
        df["sampling_priority_score"] = DEFAULT_SAMPLING_PRIORITY_SCORE
    df["sampling_priority_score"] = df["sampling_priority_score"].apply(
        lambda x: safe_float(x, DEFAULT_SAMPLING_PRIORITY_SCORE)
    )

    return df


# ============================================================
# 8. 补训练 / 推理对齐字段
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
# 9. 补训练 / 推理需要的派生特征
# ============================================================

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 确保基础计数字段存在
    for col in [
        "strong_rule_count",
        "candidate_generation_rule_count",
        "fd_like_count",
        "context_rule_count",
        "global_rule_count",
        "rare_value_count",
        "typo_rule_count",
        "pattern_rule_count",
        "schema_rule_count",
        "movie_typed_rule_count",
        "movie_consistency_rule_count",
    ]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

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

    df["movie_typed_rule_ratio"] = np.where(
        df["rule_total_count"] > 0,
        df["movie_typed_rule_count"].astype(float) / (df["rule_total_count"] + 1e-8),
        0.0
    )

    df["movie_consistency_rule_ratio"] = np.where(
        df["rule_total_count"] > 0,
        df["movie_consistency_rule_count"].astype(float) / (df["rule_total_count"] + 1e-8),
        0.0
    )

    df["movie_rule_total_count"] = (
        df["movie_typed_rule_count"].astype(float) +
        df["movie_consistency_rule_count"].astype(float)
    )

    df["has_movie_rule_signal"] = (df["movie_rule_total_count"] > 0).astype(int)

    df["rule_strength_sum"] = (
        1.0 * df["strong_rule_count"].astype(float)
        + 0.25 * df["candidate_generation_rule_count"].astype(float)
        + 0.4 * df["fd_like_count"].astype(float)
        + 0.25 * df["context_rule_count"].astype(float)
        + 0.15 * df["global_rule_count"].astype(float)
        + 0.15 * df["rare_value_count"].astype(float)
        + 0.5 * df["typo_rule_count"].astype(float)
        + 0.15 * df["pattern_rule_count"].astype(float)
        + 0.25 * df["schema_rule_count"].astype(float)
        + 0.45 * df["movie_typed_rule_count"].astype(float)
        + 0.65 * df["movie_consistency_rule_count"].astype(float)
    )

    df["candidate_rule_dominant"] = np.where(
        df["candidate_generation_rule_count"].astype(float) >
        df["strong_rule_count"].astype(float),
        "yes",
        "no"
    )

    df["rare_only_signal"] = (
        (df["rare_value_count"].astype(float) > 0) &
        (df["strong_rule_count"].astype(float) <= 0) &
        (df["fd_like_count"].astype(float) <= 0) &
        (df["context_rule_count"].astype(float) <= 0) &
        (df["typo_rule_count"].astype(float) <= 0) &
        (df["schema_rule_count"].astype(float) <= 0) &
        (df["movie_typed_rule_count"].astype(float) <= 0) &
        (df["movie_consistency_rule_count"].astype(float) <= 0)
    ).astype(int)

    df["is_weak_only"] = (
        (df["rare_only_signal"] == 1) |
        (
            (df["strong_rule_count"].astype(float) <= 0) &
            (df["fd_like_count"].astype(float) <= 0) &
            (df["context_rule_count"].astype(float) <= 0) &
            (df["typo_rule_count"].astype(float) <= 0) &
            (df["schema_rule_count"].astype(float) <= 0) &
            (df["movie_typed_rule_count"].astype(float) <= 0) &
            (df["movie_consistency_rule_count"].astype(float) <= 0) &
            (df["candidate_generation_rule_count"].astype(float) > 0)
        )
    ).astype(int)

    df["is_movie_time_col"] = df["column"].isin(["Year", "Release Date"]).astype(int)
    df["is_movie_rating_col"] = df["column"].isin(["RatingValue", "RatingCount", "ReviewCount"]).astype(int)
    df["is_movie_people_col"] = df["column"].isin(["Director", "Creator", "Actors", "Cast"]).astype(int)
    df["is_movie_list_col"] = df["column"].isin(["Language", "Country", "Genre", "Creator", "Actors", "Cast"]).astype(int)
    df["is_movie_text_col"] = df["column"].isin(["Name", "Filming Locations", "Description"]).astype(int)

    return df


# ============================================================
# 10. 重排列顺序
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
        "movie_typed_rule_count",
        "movie_consistency_rule_count",

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
        "movie_typed_rule_ratio",
        "movie_consistency_rule_ratio",
        "movie_rule_total_count",
        "has_movie_rule_signal",
        "rule_strength_sum",
        "candidate_rule_dominant",
        "rare_only_signal",
        "is_weak_only",

        "parsed_year",
        "parsed_release_year",
        "parsed_release_country",
        "duration_minutes",
        "rating_value_float",
        "count_value_int",
        "review_total",
        "list_token_count",
        "list_token_count_bucket",
        "text_length_bucket",
        "has_mojibake_or_control_chars",

        "movie_row_year",
        "movie_row_release_year",
        "movie_row_release_country",
        "movie_row_release_year_gap",
        "movie_row_duration_minutes",
        "movie_row_rating_value_float",
        "movie_row_rating_count_int",
        "movie_row_review_total",
        "movie_row_review_to_rating_ratio",
        "movie_row_actors_cast_overlap",

        "is_movie_time_col",
        "is_movie_rating_col",
        "is_movie_people_col",
        "is_movie_list_col",
        "is_movie_text_col",

        "pattern_bucket",
        "rarity_bucket",
        "neighbor_bucket",
        "movie_rule_bucket",
        "movie_field_bucket",
        "year_bucket",
        "release_year_gap_bucket",
        "duration_bucket",
        "rating_bucket",
        "count_bucket",
        "review_ratio_bucket",
        "actors_cast_overlap_bucket",
        "list_bucket",
        "text_bucket",

        "bucket_id",
        "cluster_id",
        "dist_to_center",
        "sample_role",
        "is_sampled",
        "sampling_priority_score",

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
# 11. 主流程
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

    # 稳定排序，便于和评估文件对齐
    df = df.sort_values(by=["row_id", "column"]).reset_index(drop=True)

    df.to_csv(OUTPUT_INFER_READY_CSV, index=False, encoding="utf-8-sig")

    print(f"[OK] 推理对齐文件已保存: {OUTPUT_INFER_READY_CSV}")
    print(f"[OK] 行数: {len(df)}")
    print(f"[OK] 列数: {len(df.columns)}")

    print("\n===== 按列候选数 Top 20 =====")
    print(df["column"].value_counts(dropna=False).head(20).to_string())

    print("\n===== 预览前5行 =====")
    preview_cols = [
        "row_id", "column", "value",
        "semantic_type", "detected_type",
        "violation_count", "conflict_score",
        "pattern_bucket", "rarity_bucket", "neighbor_bucket",
        "movie_rule_bucket", "movie_field_bucket",
        "year_bucket", "duration_bucket", "rating_bucket", "count_bucket",
        "bucket_id", "cluster_id", "dist_to_center",
        "candidate_rule_ratio", "has_neighbor_signal",
        "high_neighbor_consistency", "is_rare_value",
        "is_equal_to_neighbor_majority", "neighbor_agreement_score",
        "rule_total_count", "strong_rule_ratio",
        "movie_rule_total_count", "has_movie_rule_signal",
        "label_source", "is_outside_candidate"
    ]
    preview_cols = [c for c in preview_cols if c in df.columns]
    print(df[preview_cols].head().to_string(index=False))


if __name__ == "__main__":
    main()

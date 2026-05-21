import json
from collections import Counter
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. 用户配置区
# ============================================================

INPUT_SUMMARY_CSV = "candidate_llm_contexts_flat_movies.csv"
INPUT_DETAIL_JSONL = "candidate_llm_contexts_movies.jsonl"

OUTPUT_CLUSTERED_CSV = "candidate_clustered_movies.csv"
OUTPUT_SAMPLED_CSV = "candidate_sampled_movies.csv"
OUTPUT_SAMPLED_JSONL = "candidate_sampled_with_context_movies.jsonl"
OUTPUT_BUCKET_SUMMARY_CSV = "bucket_cluster_summary_movies.csv"

# 这版的核心目标：真正少量采样，而不是近似全标注。
# 关键原则：粗分桶 + 桶内聚类 + 每簇少量代表点 + 全局预算上限。
USE_COLUMN_IN_BUCKET = True

MIN_BUCKET_SIZE_FOR_CLUSTER = 20
MAX_CLUSTERS_PER_BUCKET = 3
TARGET_CLUSTER_SIZE = 80
SMALL_BUCKET_SAMPLE_THRESHOLD = 5

SAMPLE_CENTER = True
SAMPLE_BOUNDARY = False
SAMPLE_OUTLIER = False

# 全局 LLM 标注预算。你可以改成 500 / 800 / 1500。
MAX_TOTAL_SAMPLES = 1000

# 防止某一列，比如 Release Date，把预算吃完。
MAX_SAMPLES_PER_COLUMN = {
    "Release Date": 160,
    "Year": 100,
    "Duration": 100,
    "RatingValue": 100,
    "RatingCount": 100,
    "ReviewCount": 100,
    "Language": 100,
    "Country": 100,
    "Genre": 100,
    "Director": 120,
    "Creator": 120,
    "Actors": 120,
    "Cast": 120,
    "Name": 100,
    "Filming Locations": 80,
    "Description": 80,
}

# 防止某个 bucket 过大。
MAX_SAMPLES_PER_BUCKET = 12

RANDOM_STATE = 42


# ============================================================
# 2. 读取详细 JSONL
# ============================================================

def load_detail_jsonl(path: str) -> Dict[Tuple[int, str], Dict[str, Any]]:
    detail_map = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] 跳过无法解析的 JSONL 行: {path}:{line_no}")
                continue
            key = (int(obj["row_id"]), str(obj["column"]))
            detail_map[key] = obj
    return detail_map


def extract_main_rule_type_from_detail(detail_obj: Dict[str, Any]) -> str:
    try:
        conflict_ctx = detail_obj.get("conflict_context", {})
        rule_types = conflict_ctx.get("rule_types", [])
        if not rule_types:
            detailed = conflict_ctx.get("violated_rules_detailed", [])
            rule_types = [x.get("rule_type") for x in detailed if x.get("rule_type")]
        if not rule_types:
            return "none"
        return Counter(rule_types).most_common(1)[0][0]
    except Exception:
        return "unknown"


def extract_main_usage_role_from_detail(detail_obj: Dict[str, Any]) -> str:
    try:
        conflict_ctx = detail_obj.get("conflict_context", {})
        usage_roles = conflict_ctx.get("usage_roles", [])
        if not usage_roles:
            detailed = conflict_ctx.get("violated_rules_detailed", [])
            usage_roles = [x.get("usage_role") for x in detailed if x.get("usage_role")]
        if not usage_roles:
            return "none"
        return Counter(usage_roles).most_common(1)[0][0]
    except Exception:
        return "unknown"


def build_pattern_bucket(detail_obj: Dict[str, Any]) -> str:
    try:
        col_ctx = detail_obj.get("column_context", {})
        cp = str(col_ctx.get("current_value_pattern", "NULL"))
        dp = str(col_ctx.get("dominant_pattern", "NULL"))
        return "pattern_match" if cp == dp else "pattern_mismatch"
    except Exception:
        return "unknown"


def build_movie_rule_bucket(detail_obj: Dict[str, Any]) -> str:
    try:
        conflict_ctx = detail_obj.get("conflict_context", {})
        typed_cnt = int(conflict_ctx.get("movie_typed_rule_count", 0))
        consistency_cnt = int(conflict_ctx.get("movie_consistency_rule_count", 0))
        if typed_cnt + consistency_cnt <= 0:
            return "no_movie_rule_signal"
        if typed_cnt > 0 and consistency_cnt > 0:
            return "typed_and_consistency_signal"
        if consistency_cnt > 0:
            return "movie_consistency_signal"
        if typed_cnt == 1:
            return "single_movie_typed_signal"
        return "multi_movie_typed_signal"
    except Exception:
        return "unknown"


def build_movie_field_bucket(column: str) -> str:
    col = str(column)
    if col in {"Year", "Release Date"}:
        return "movie_time_metadata"
    if col == "Duration":
        return "movie_duration_metadata"
    if col in {"RatingValue", "RatingCount", "ReviewCount"}:
        return "movie_rating_metadata"
    if col in {"Director", "Creator", "Actors", "Cast"}:
        return "movie_people_metadata"
    if col in {"Language", "Country", "Genre"}:
        return "movie_categorical_list_metadata"
    if col == "Name":
        return "movie_title_metadata"
    if col in {"Filming Locations", "Description"}:
        return "movie_text_metadata"
    return "movie_other_metadata"


# ============================================================
# 3. 分桶辅助字段
# ============================================================

def to_num(x, default=np.nan):
    v = pd.to_numeric(x, errors="coerce")
    return default if pd.isna(v) else v


def build_rarity_bucket(value_frequency, value_frequency_rank) -> str:
    vf = to_num(value_frequency)
    vr = to_num(value_frequency_rank)
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


def build_neighbor_bucket(neighbor_majority_ratio, neighbor_majority_value, current_value) -> str:
    nmr = to_num(neighbor_majority_ratio)
    if pd.isna(nmr):
        return "no_neighbor_signal"
    same = str(neighbor_majority_value) == str(current_value)
    if not same:
        if nmr >= 0.8:
            return "strong_support_other"
        if nmr >= 0.5:
            return "medium_support_other"
        return "weak_support_other"
    if nmr >= 0.8:
        return "strong_support_current"
    return "weak_support_current"


def bucket_from_numeric(x, cuts, names, missing_name):
    v = to_num(x)
    if pd.isna(v):
        return missing_name
    for cut, name in zip(cuts, names):
        if v <= cut:
            return name
    return names[-1]


def build_year_bucket(year_value):
    return bucket_from_numeric(
        year_value,
        [1930, 1970, 2000, 2015, 9999],
        ["very_old_movie", "classic_movie", "late_20c_movie", "modern_movie", "recent_movie"],
        "unparsed_year",
    )


def build_release_year_gap_bucket(gap_value):
    gap = to_num(gap_value)
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


def build_duration_bucket(duration_minutes):
    return bucket_from_numeric(
        duration_minutes,
        [40, 90, 150, 240, 99999],
        ["short_duration", "medium_short_duration", "standard_duration", "long_duration", "extreme_duration"],
        "unparsed_duration",
    )


def build_rating_value_bucket(rating_value):
    return bucket_from_numeric(
        rating_value,
        [4, 6, 8, 999],
        ["low_rating", "mid_low_rating", "mid_high_rating", "high_rating"],
        "unparsed_rating",
    )


def build_count_bucket(count_value):
    return bucket_from_numeric(
        count_value,
        [0, 100, 10000, 100000, 10**18],
        ["zero_count", "small_count", "medium_count", "large_count", "very_large_count"],
        "unparsed_count",
    )


def build_review_ratio_bucket(ratio_value):
    return bucket_from_numeric(
        ratio_value,
        [0.01, 0.05, 0.20, 0.50, 10**18],
        ["tiny_review_ratio", "small_review_ratio", "medium_review_ratio", "large_review_ratio", "extreme_review_ratio"],
        "no_review_ratio",
    )


def build_overlap_bucket(overlap_value):
    o = to_num(overlap_value)
    if pd.isna(o):
        return "no_overlap_signal"
    if o >= 0.8:
        return "strong_actor_cast_overlap"
    if o >= 0.5:
        return "medium_actor_cast_overlap"
    if o > 0:
        return "weak_actor_cast_overlap"
    return "zero_actor_cast_overlap"


def build_list_bucket(list_token_count, list_token_count_bucket):
    if pd.notna(list_token_count_bucket) and str(list_token_count_bucket).strip():
        return str(list_token_count_bucket)
    n = to_num(list_token_count)
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


def build_text_bucket(text_length_bucket):
    if pd.isna(text_length_bucket) or str(text_length_bucket).strip() == "":
        return "unknown_text_length"
    return str(text_length_bucket)


# ============================================================
# 4. 读取 summary 并补充字段
# ============================================================

def load_summary_csv(summary_csv: str, detail_map: Dict[Tuple[int, str], Dict[str, Any]]) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)
    df["row_id"] = df["row_id"].astype(int)
    df["column"] = df["column"].astype(str)

    main_rule_types, main_usage_roles = [], []
    pattern_buckets, movie_rule_buckets, movie_field_buckets = [], [], []

    for _, row in df.iterrows():
        key = (int(row["row_id"]), row["column"])
        detail_obj = detail_map.get(key, {})

        rt = row.get("main_rule_type", None)
        if pd.isna(rt) or rt is None or str(rt).strip() == "":
            rt = extract_main_rule_type_from_detail(detail_obj)
        ur = row.get("main_usage_role", None)
        if pd.isna(ur) or ur is None or str(ur).strip() == "":
            ur = extract_main_usage_role_from_detail(detail_obj)

        main_rule_types.append(str(rt))
        main_usage_roles.append(str(ur))
        pattern_buckets.append(build_pattern_bucket(detail_obj))
        movie_rule_buckets.append(build_movie_rule_bucket(detail_obj))
        movie_field_buckets.append(build_movie_field_bucket(row["column"]))

    df["main_rule_type"] = main_rule_types
    df["main_usage_role"] = main_usage_roles
    df["pattern_bucket"] = pattern_buckets
    df["movie_rule_bucket"] = movie_rule_buckets
    df["movie_field_bucket"] = movie_field_buckets

    df["rarity_bucket"] = df.apply(lambda r: build_rarity_bucket(r.get("value_frequency"), r.get("value_frequency_rank")), axis=1)
    df["neighbor_bucket"] = df.apply(lambda r: build_neighbor_bucket(r.get("neighbor_majority_ratio"), r.get("neighbor_majority_value"), r.get("value")), axis=1)
    df["year_bucket"] = df.apply(lambda r: build_year_bucket(r.get("parsed_year") if pd.notna(r.get("parsed_year", np.nan)) else r.get("movie_row_year")), axis=1)
    df["release_year_gap_bucket"] = df.apply(lambda r: build_release_year_gap_bucket(r.get("movie_row_release_year_gap")), axis=1)
    df["duration_bucket"] = df.apply(lambda r: build_duration_bucket(r.get("duration_minutes") if pd.notna(r.get("duration_minutes", np.nan)) else r.get("movie_row_duration_minutes")), axis=1)
    df["rating_bucket"] = df.apply(lambda r: build_rating_value_bucket(r.get("rating_value_float") if pd.notna(r.get("rating_value_float", np.nan)) else r.get("movie_row_rating_value_float")), axis=1)
    df["count_bucket"] = df.apply(lambda r: build_count_bucket(r.get("count_value_int") if pd.notna(r.get("count_value_int", np.nan)) else r.get("movie_row_rating_count_int")), axis=1)
    df["review_ratio_bucket"] = df.apply(lambda r: build_review_ratio_bucket(r.get("movie_row_review_to_rating_ratio")), axis=1)
    df["actors_cast_overlap_bucket"] = df.apply(lambda r: build_overlap_bucket(r.get("movie_row_actors_cast_overlap")), axis=1)
    df["list_bucket"] = df.apply(lambda r: build_list_bucket(r.get("list_token_count"), r.get("list_token_count_bucket")), axis=1)
    df["text_bucket"] = df.apply(lambda r: build_text_bucket(r.get("text_length_bucket")), axis=1)

    numeric_defaults = [
        "violation_count", "conflict_score", "value_frequency", "value_frequency_rank",
        "neighbor_majority_ratio", "prior_error_probability", "posterior_error_probability",
        "strong_rule_count", "candidate_generation_rule_count", "fd_like_count",
        "context_rule_count", "global_rule_count", "rare_value_count", "typo_rule_count",
        "pattern_rule_count", "schema_rule_count", "movie_typed_rule_count",
        "movie_consistency_rule_count", "parsed_year", "parsed_release_year", "duration_minutes",
        "rating_value_float", "count_value_int", "review_total", "list_token_count",
        "has_mojibake_or_control_chars", "movie_row_year", "movie_row_release_year",
        "movie_row_release_year_gap", "movie_row_duration_minutes", "movie_row_rating_value_float",
        "movie_row_rating_count_int", "movie_row_review_total", "movie_row_review_to_rating_ratio",
        "movie_row_actors_cast_overlap",
    ]
    for col in numeric_defaults:
        if col not in df.columns:
            df[col] = 0
    return df


# ============================================================
# 5. 粗分桶
# ============================================================

def get_bucket_keys() -> List[str]:
    if USE_COLUMN_IN_BUCKET:
        return ["column", "main_rule_type", "movie_field_bucket", "neighbor_bucket"]
    return ["main_rule_type", "movie_field_bucket", "neighbor_bucket"]


def build_bucket_id(row: pd.Series, keys: List[str]) -> str:
    parts = []
    for key in keys:
        val = row.get(key, "unknown")
        if pd.isna(val):
            val = "NULL"
        parts.append(f"{key}={val}")
    return " | ".join(parts)


def assign_buckets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    keys = get_bucket_keys()
    df["bucket_id"] = df.apply(lambda r: build_bucket_id(r, keys), axis=1)
    return df


# ============================================================
# 6. 聚类特征
# ============================================================

FEATURE_COLUMNS_NUMERIC = [
    "violation_count", "conflict_score", "value_frequency", "value_frequency_rank",
    "neighbor_majority_ratio", "prior_error_probability", "posterior_error_probability",
    "strong_rule_count", "candidate_generation_rule_count", "fd_like_count",
    "context_rule_count", "global_rule_count", "rare_value_count", "typo_rule_count",
    "pattern_rule_count", "schema_rule_count", "movie_typed_rule_count",
    "movie_consistency_rule_count", "parsed_year", "parsed_release_year",
    "duration_minutes", "rating_value_float", "count_value_int", "review_total",
    "list_token_count", "has_mojibake_or_control_chars", "movie_row_year",
    "movie_row_release_year", "movie_row_release_year_gap", "movie_row_duration_minutes",
    "movie_row_rating_value_float", "movie_row_rating_count_int", "movie_row_review_total",
    "movie_row_review_to_rating_ratio", "movie_row_actors_cast_overlap",
]

FEATURE_COLUMNS_CATEGORICAL = [
    "semantic_type", "detected_type", "main_rule_type", "main_usage_role",
    "pattern_bucket", "rarity_bucket", "neighbor_bucket", "movie_rule_bucket",
    "movie_field_bucket", "year_bucket", "release_year_gap_bucket", "duration_bucket",
    "rating_bucket", "count_bucket", "review_ratio_bucket", "actors_cast_overlap_bucket",
    "list_bucket", "text_bucket",
]


def prepare_features(bucket_df: pd.DataFrame):
    work = bucket_df.copy()
    for col in FEATURE_COLUMNS_NUMERIC:
        if col not in work.columns:
            work[col] = 0.0
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)
    num_df = work[FEATURE_COLUMNS_NUMERIC].copy()

    cat_frames = []
    for col in FEATURE_COLUMNS_CATEGORICAL:
        if col not in work.columns:
            work[col] = "unknown"
        cat_frames.append(pd.get_dummies(work[col].fillna("unknown").astype(str), prefix=col))

    feat_df = pd.concat([num_df] + cat_frames, axis=1) if cat_frames else num_df
    scaler = StandardScaler()
    X = scaler.fit_transform(feat_df.values)
    return X


def choose_k(n: int) -> int:
    if n < MIN_BUCKET_SIZE_FOR_CLUSTER:
        return 1
    k = max(1, round(n / TARGET_CLUSTER_SIZE))
    k = min(k, MAX_CLUSTERS_PER_BUCKET, n)
    return max(1, k)


def cluster_bucket(bucket_df: pd.DataFrame) -> pd.DataFrame:
    bucket_df = bucket_df.copy()
    n = len(bucket_df)
    if n <= 1:
        bucket_df["cluster_id"] = 0
        bucket_df["dist_to_center"] = 0.0
        return bucket_df

    X = prepare_features(bucket_df)
    k = choose_k(n)
    if k <= 1:
        bucket_df["cluster_id"] = 0
        center = X.mean(axis=0, keepdims=True)
        bucket_df["dist_to_center"] = np.linalg.norm(X - center, axis=1)
        return bucket_df

    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = model.fit_predict(X)
    centers = model.cluster_centers_
    dists = np.array([np.linalg.norm(X[i] - centers[labels[i]]) for i in range(len(X))])
    bucket_df["cluster_id"] = labels
    bucket_df["dist_to_center"] = dists
    return bucket_df


def sample_from_cluster(cluster_df: pd.DataFrame) -> pd.DataFrame:
    cluster_df = cluster_df.copy().sort_values("dist_to_center").reset_index(drop=True)
    n = len(cluster_df)
    cluster_df["sample_role"] = ""
    cluster_df["is_sampled"] = 0

    # 小簇也只抽 1 条代表样本，不全采样。
    if n <= SMALL_BUCKET_SAMPLE_THRESHOLD:
        cluster_df.loc[0, "sample_role"] = "center"
        cluster_df.loc[0, "is_sampled"] = 1
        return cluster_df

    sampled_indices = []
    roles = {}
    if SAMPLE_CENTER:
        sampled_indices.append(0)
        roles[0] = "center"
    if SAMPLE_OUTLIER:
        idx = n - 1
        if idx not in sampled_indices:
            sampled_indices.append(idx)
            roles[idx] = "outlier"
    if SAMPLE_BOUNDARY:
        idx = max(0, int(round(0.75 * (n - 1))))
        if idx not in sampled_indices:
            sampled_indices.append(idx)
            roles[idx] = "boundary"
    if not sampled_indices:
        sampled_indices = [0]
        roles[0] = "center"

    for idx in sampled_indices:
        cluster_df.loc[idx, "sample_role"] = roles[idx]
        cluster_df.loc[idx, "is_sampled"] = 1
    return cluster_df


# ============================================================
# 7. 采样预算控制
# ============================================================

def sampling_priority_score(row: pd.Series) -> float:
    def n(col):
        v = pd.to_numeric(row.get(col, 0), errors="coerce")
        return 0.0 if pd.isna(v) else float(v)
    return (
        0.01 * n("conflict_score")
        + 2.0 * n("posterior_error_probability")
        + 0.3 * n("violation_count")
        + 0.5 * n("movie_typed_rule_count")
        + 0.8 * n("movie_consistency_rule_count")
        + 0.2 * n("neighbor_majority_ratio")
    )


def cap_by_group(df: pd.DataFrame, group_col: str, cap_getter) -> pd.DataFrame:
    if len(df) == 0:
        return df
    parts = []
    for key, g in df.groupby(group_col, sort=False):
        cap = int(cap_getter(key))
        g = g.sort_values(
            ["sampling_priority_score", "conflict_score", "posterior_error_probability"],
            ascending=[False, False, False]
        )
        parts.append(g.head(cap))
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0].copy()


def apply_sampling_budget(sampled_df: pd.DataFrame) -> pd.DataFrame:
    before = len(sampled_df)
    sampled_df = sampled_df.copy()
    sampled_df["sampling_priority_score"] = sampled_df.apply(sampling_priority_score, axis=1)

    sampled_df = cap_by_group(sampled_df, "bucket_id", lambda _: MAX_SAMPLES_PER_BUCKET)
    after_bucket = len(sampled_df)

    sampled_df = cap_by_group(sampled_df, "column", lambda c: MAX_SAMPLES_PER_COLUMN.get(str(c), max(50, MAX_TOTAL_SAMPLES // 10)))
    after_column = len(sampled_df)

    if len(sampled_df) > MAX_TOTAL_SAMPLES:
        sampled_df = sampled_df.sort_values(
            ["sampling_priority_score", "conflict_score", "posterior_error_probability"],
            ascending=[False, False, False]
        ).head(MAX_TOTAL_SAMPLES).copy()
    after_global = len(sampled_df)

    sampled_df["is_sampled"] = 1
    print("\n===== 采样预算控制 =====")
    print(f"预算前采样数: {before}")
    print(f"桶上限后采样数: {after_bucket}")
    print(f"列上限后采样数: {after_column}")
    print(f"全局上限后采样数: {after_global}")
    print(f"MAX_TOTAL_SAMPLES: {MAX_TOTAL_SAMPLES}")
    return sampled_df


def nan_to_none(x):
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    return x


def row_to_summary_context(row: pd.Series) -> Dict[str, Any]:
    fields = [
        "semantic_type", "detected_type", "violation_count", "conflict_score",
        "value_frequency", "value_frequency_rank", "neighbor_majority_value",
        "neighbor_majority_ratio", "prior_error_probability", "posterior_error_probability",
        "main_rule_type", "main_usage_role", "strong_rule_count",
        "candidate_generation_rule_count", "fd_like_count", "context_rule_count",
        "global_rule_count", "rare_value_count", "typo_rule_count", "pattern_rule_count",
        "schema_rule_count", "movie_typed_rule_count", "movie_consistency_rule_count",
        "parsed_year", "parsed_release_year", "parsed_release_country", "duration_minutes",
        "rating_value_float", "count_value_int", "review_total", "list_token_count",
        "list_token_count_bucket", "text_length_bucket", "has_mojibake_or_control_chars",
        "movie_row_year", "movie_row_release_year", "movie_row_release_country",
        "movie_row_release_year_gap", "movie_row_duration_minutes",
        "movie_row_rating_value_float", "movie_row_rating_count_int",
        "movie_row_review_total", "movie_row_review_to_rating_ratio",
        "movie_row_actors_cast_overlap", "pattern_bucket", "rarity_bucket",
        "neighbor_bucket", "movie_rule_bucket", "movie_field_bucket", "year_bucket",
        "release_year_gap_bucket", "duration_bucket", "rating_bucket", "count_bucket",
        "review_ratio_bucket", "actors_cast_overlap_bucket", "list_bucket", "text_bucket",
        "sampling_priority_score",
    ]
    return {f: nan_to_none(row.get(f)) if f in row.index else None for f in fields}


# ============================================================
# 8. 主流程
# ============================================================

def main():
    detail_map = load_detail_jsonl(INPUT_DETAIL_JSONL)
    summary_df = load_summary_csv(INPUT_SUMMARY_CSV, detail_map)
    summary_df = assign_buckets(summary_df)

    clustered_parts = []
    bucket_summary_rows = []

    for bucket_id, bucket_df in summary_df.groupby("bucket_id", sort=False):
        bucket_clustered = cluster_bucket(bucket_df)
        bucket_parts = []
        for cluster_id, cluster_df in bucket_clustered.groupby("cluster_id", sort=False):
            cluster_sampled = sample_from_cluster(cluster_df)
            bucket_parts.append(cluster_sampled)
            bucket_summary_rows.append({
                "bucket_id": bucket_id,
                "bucket_size": len(bucket_df),
                "cluster_id": int(cluster_id),
                "cluster_size": len(cluster_df),
                "sampled_count_before_budget": int(cluster_sampled["is_sampled"].sum()),
                "column": str(cluster_df["column"].iloc[0]) if "column" in cluster_df.columns else "unknown",
                "main_rule_type": str(cluster_df["main_rule_type"].iloc[0]) if "main_rule_type" in cluster_df.columns else "unknown",
                "neighbor_bucket": str(cluster_df["neighbor_bucket"].iloc[0]) if "neighbor_bucket" in cluster_df.columns else "unknown",
                "movie_field_bucket": str(cluster_df["movie_field_bucket"].iloc[0]) if "movie_field_bucket" in cluster_df.columns else "unknown",
            })
        clustered_parts.append(pd.concat(bucket_parts, ignore_index=True))

    clustered_df = pd.concat(clustered_parts, ignore_index=True) if clustered_parts else summary_df.iloc[0:0].copy()
    clustered_df.to_csv(OUTPUT_CLUSTERED_CSV, index=False, encoding="utf-8-sig")

    sampled_df = clustered_df[clustered_df["is_sampled"] == 1].copy()
    sampled_df = apply_sampling_budget(sampled_df)
    sampled_df = sampled_df.sort_values(["column", "main_rule_type", "bucket_id", "cluster_id", "row_id"]).reset_index(drop=True)
    sampled_df.to_csv(OUTPUT_SAMPLED_CSV, index=False, encoding="utf-8-sig")

    with open(OUTPUT_SAMPLED_JSONL, "w", encoding="utf-8") as f:
        for _, row in sampled_df.iterrows():
            key = (int(row["row_id"]), str(row["column"]))
            out = {
                "row_id": int(row["row_id"]),
                "column": str(row["column"]),
                "value": nan_to_none(row.get("value")),
                "bucket_id": row["bucket_id"],
                "cluster_id": int(row["cluster_id"]),
                "sample_role": row["sample_role"],
                "dist_to_center": float(row["dist_to_center"]),
                "summary_context": row_to_summary_context(row),
                "detail_context": detail_map.get(key, {}),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    pd.DataFrame(bucket_summary_rows).to_csv(OUTPUT_BUCKET_SUMMARY_CSV, index=False, encoding="utf-8-sig")

    print(f"[OK] 完整聚类结果已保存: {OUTPUT_CLUSTERED_CSV}")
    print(f"[OK] 采样摘要 CSV 已保存: {OUTPUT_SAMPLED_CSV}")
    print(f"[OK] 采样详细 JSONL 已保存: {OUTPUT_SAMPLED_JSONL}")
    print(f"[OK] 分桶/聚类摘要已保存: {OUTPUT_BUCKET_SUMMARY_CSV}")
    print("\n===== 统计信息 =====")
    print(f"总候选数: {len(clustered_df)}")
    print(f"最终采样数: {len(sampled_df)}")
    print(f"总桶数: {clustered_df['bucket_id'].nunique() if len(clustered_df) > 0 else 0}")
    print(f"总簇数: {clustered_df[['bucket_id', 'cluster_id']].drop_duplicates().shape[0] if len(clustered_df) > 0 else 0}")
    if len(sampled_df) > 0:
        print("\n按列采样分布:")
        print(sampled_df["column"].value_counts(dropna=False).to_string())
        print("\n采样角色分布:")
        print(sampled_df["sample_role"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()

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

INPUT_SUMMARY_CSV = "candidate_llm_contexts_flat_adult.csv"
INPUT_DETAIL_JSONL = "candidate_llm_contexts_adult.jsonl"

OUTPUT_CLUSTERED_CSV = "candidate_clustered_adult.csv"
OUTPUT_SAMPLED_CSV = "candidate_sampled_adult.csv"
OUTPUT_SAMPLED_JSONL = "candidate_sampled_with_context_adult.jsonl"
OUTPUT_BUCKET_SUMMARY_CSV = "bucket_cluster_summary_adult.csv"
OUTPUT_SAMPLE_BUDGET_SUMMARY_JSON = "sample_budget_summary_adult.json"

# 是否把 column 纳入分桶。
# adult 字段语义差异较大，建议 True，让每列内部有代表样本。
USE_COLUMN_IN_BUCKET = True

# 聚类参数
MIN_BUCKET_SIZE_FOR_CLUSTER = 8
MAX_CLUSTERS_PER_BUCKET = 5
TARGET_CLUSTER_SIZE = 18
SMALL_BUCKET_FULL_SAMPLE_THRESHOLD = 3

SAMPLE_CENTER = True
SAMPLE_BOUNDARY = True
SAMPLE_OUTLIER = True

# ============================================================
# 采样预算控制：避免后续 LLM token 消耗过大
# ============================================================

# 总体最多送 LLM 标注多少个候选。
# 如果钱包紧张，可改成 120；如果想更稳，可改成 250。
MAX_TOTAL_SAMPLES = 180

# 每列最多采样多少个。
MAX_SAMPLES_PER_COLUMN = {
    "age": 20,
    "workclass": 16,
    "education": 22,
    "maritalstatus": 20,
    "occupation": 22,
    "relationship": 22,
    "race": 14,
    "sex": 14,
    "hoursperweek": 22,
    "country": 18,
    "income": 18,
}
DEFAULT_MAX_SAMPLES_PER_COLUMN = 18

# 每个 bucket 最多采样多少个，防止某个大桶吞掉预算。
MAX_SAMPLES_PER_BUCKET = 4

# 每个 cluster 最多采样多少个。
# center/boundary/outlier 理论最多 3 个，这里默认最多 2，节省 token。
MAX_SAMPLES_PER_CLUSTER = 2

# 优先保留高价值信号的样本。
# 先覆盖强规则/成人一致性规则，再覆盖弱规则。
PRIORITIZE_STRONG_SIGNAL = True

# 最少覆盖：尽量保证每个列至少有若干样本。
MIN_SAMPLES_PER_COLUMN_IF_AVAILABLE = 5

# 对非常弱且 token 价值低的样本做降采样。
ENABLE_WEAK_SIGNAL_DOWNSAMPLE = True
WEAK_SIGNAL_KEEP_RATIO = 0.35

RANDOM_STATE = 42


# ============================================================
# 2. 读取详细 jsonl，建立索引
# ============================================================

def load_detail_jsonl(path: str) -> Dict[Tuple[int, str], Dict[str, Any]]:
    detail_map = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = (int(obj["row_id"]), obj["column"])
            detail_map[key] = obj
    return detail_map


# ============================================================
# 3. 从详细上下文中提取补充信息
# ============================================================

def extract_main_rule_type_from_detail(detail_obj: Dict[str, Any]) -> str:
    try:
        conflict_ctx = detail_obj.get("conflict_context", {})
        rule_types = conflict_ctx.get("rule_types", [])
        if not rule_types:
            detailed = conflict_ctx.get("violated_rules_detailed", [])
            rule_types = [x.get("rule_type") for x in detailed if x.get("rule_type")]
        if not rule_types:
            return "none"
        cnt = Counter(rule_types)
        return cnt.most_common(1)[0][0]
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
        cnt = Counter(usage_roles)
        return cnt.most_common(1)[0][0]
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


def build_adult_consistency_bucket(detail_obj: Dict[str, Any]) -> str:
    try:
        adult_ctx = detail_obj.get("adult_consistency_context", {})
        flags = adult_ctx.get("evidence_flags", [])
        score = float(adult_ctx.get("adult_consistency_score", 0.0) or 0.0)
        if flags and score >= 1.0:
            return "strong_adult_consistency_signal"
        if flags:
            return "adult_consistency_flag"
        if score > 0:
            return "weak_adult_consistency_signal"
        return "no_adult_consistency_signal"
    except Exception:
        return "unknown"


def build_domain_bucket(row: pd.Series) -> str:
    if "domain_valid" not in row.index:
        return "unknown_domain"
    val = row.get("domain_valid")
    if pd.isna(val):
        return "domain_not_applicable"
    s = str(val).strip().lower()
    if s in {"true", "1", "yes"}:
        return "domain_valid"
    if s in {"false", "0", "no"}:
        return "domain_invalid"
    return "domain_unknown"


# ============================================================
# 4. 通用分桶字段
# ============================================================

def build_rarity_bucket(value_frequency: Any, value_frequency_rank: Any) -> str:
    vf = pd.to_numeric(value_frequency, errors="coerce")
    vr = pd.to_numeric(value_frequency_rank, errors="coerce")

    if pd.notna(vr):
        if vr <= 3:
            return "very_common"
        elif vr <= 10:
            return "common"

    if pd.notna(vf):
        if vf <= 1:
            return "singleton"
        elif vf <= 2:
            return "very_rare"
        elif vf <= 5:
            return "rare"

    return "mid_frequency"


def build_neighbor_bucket(neighbor_majority_ratio: Any, neighbor_majority_value: Any, current_value: Any) -> str:
    nmr = pd.to_numeric(neighbor_majority_ratio, errors="coerce")

    if pd.isna(nmr):
        return "no_neighbor_signal"

    same = str(neighbor_majority_value) == str(current_value)

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


def build_age_bucket_for_sampling(age_lower: Any, age_upper: Any) -> str:
    lo = pd.to_numeric(age_lower, errors="coerce")
    hi = pd.to_numeric(age_upper, errors="coerce")
    if pd.isna(lo):
        return "age_unparsed"
    if pd.notna(hi) and hi < 18:
        return "minor"
    if lo <= 25:
        return "young_adult"
    if lo <= 50:
        return "working_age"
    return "older_adult"


def build_hours_bucket_for_sampling(hours_lower: Any, hours_upper: Any) -> str:
    lo = pd.to_numeric(hours_lower, errors="coerce")
    hi = pd.to_numeric(hours_upper, errors="coerce")
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


# ============================================================
# 5. 读取 summary csv，并补充分桶字段
# ============================================================

def load_summary_csv(summary_csv: str, detail_map: Dict[Tuple[int, str], Dict[str, Any]]) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)

    df["row_id"] = df["row_id"].astype(int)
    df["column"] = df["column"].astype(str)

    main_rule_types = []
    main_usage_roles = []
    pattern_buckets = []
    adult_consistency_buckets = []

    for _, row in df.iterrows():
        key = (int(row["row_id"]), row["column"])
        detail_obj = detail_map.get(key, {})

        current_main_rule_type = row.get("main_rule_type", None)
        if pd.isna(current_main_rule_type) or current_main_rule_type is None or str(current_main_rule_type).strip() == "":
            current_main_rule_type = extract_main_rule_type_from_detail(detail_obj)

        current_main_usage_role = row.get("main_usage_role", None)
        if pd.isna(current_main_usage_role) or current_main_usage_role is None or str(current_main_usage_role).strip() == "":
            current_main_usage_role = extract_main_usage_role_from_detail(detail_obj)

        main_rule_types.append(current_main_rule_type)
        main_usage_roles.append(current_main_usage_role)
        pattern_buckets.append(build_pattern_bucket(detail_obj))
        adult_consistency_buckets.append(build_adult_consistency_bucket(detail_obj))

    df["main_rule_type"] = main_rule_types
    df["main_usage_role"] = main_usage_roles
    df["pattern_bucket"] = pattern_buckets
    df["adult_consistency_bucket"] = adult_consistency_buckets

    df["rarity_bucket"] = df.apply(
        lambda r: build_rarity_bucket(r.get("value_frequency"), r.get("value_frequency_rank")),
        axis=1
    )

    df["neighbor_bucket"] = df.apply(
        lambda r: build_neighbor_bucket(
            r.get("neighbor_majority_ratio"),
            r.get("neighbor_majority_value"),
            r.get("value")
        ),
        axis=1
    )

    df["domain_bucket"] = df.apply(build_domain_bucket, axis=1)

    df["age_sampling_bucket"] = df.apply(
        lambda r: build_age_bucket_for_sampling(r.get("age_lower"), r.get("age_upper")),
        axis=1
    )

    df["hours_sampling_bucket"] = df.apply(
        lambda r: build_hours_bucket_for_sampling(r.get("hours_lower"), r.get("hours_upper")),
        axis=1
    )

    # 兼容旧训练脚本的时间字段，adult 中固定为 not applicable。
    df["time_window_bucket"] = "adult_not_applicable"
    df["time_value_bucket"] = "adult_not_applicable"

    numeric_defaults = [
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
        "time_value_minutes",
        "age_lower",
        "age_upper",
        "hours_lower",
        "hours_upper",
        "education_order",
        "adult_profile_inconsistency_count",
        "adult_consistency_score",
    ]
    for col in numeric_defaults:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


# ============================================================
# 6. 分桶
# ============================================================

def get_bucket_keys() -> List[str]:
    keys = [
        "semantic_type",
        "main_rule_type",
        "main_usage_role",
        "pattern_bucket",
        "rarity_bucket",
        "neighbor_bucket",
        "domain_bucket",
        "adult_consistency_bucket",
        "age_sampling_bucket",
        "hours_sampling_bucket",
    ]
    if USE_COLUMN_IN_BUCKET:
        keys = ["column"] + keys
    return keys


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
# 7. 聚类特征
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
    "education_order",
    "adult_profile_inconsistency_count",
    "adult_consistency_score",
]

FEATURE_COLUMNS_CATEGORICAL = [
    "column",
    "semantic_type",
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
        dummies = pd.get_dummies(work[col].fillna("unknown").astype(str), prefix=col)
        cat_frames.append(dummies)

    feat_df = pd.concat([num_df] + cat_frames, axis=1) if cat_frames else num_df

    scaler = StandardScaler()
    X = scaler.fit_transform(feat_df.values)
    return X, feat_df.columns.tolist()


# ============================================================
# 8. 自动选簇数
# ============================================================

def choose_k(n: int) -> int:
    if n < MIN_BUCKET_SIZE_FOR_CLUSTER:
        return 1
    k = max(2, round(n / TARGET_CLUSTER_SIZE))
    k = min(k, MAX_CLUSTERS_PER_BUCKET)
    k = min(k, n)
    return k


# ============================================================
# 9. 桶内聚类
# ============================================================

def cluster_bucket(bucket_df: pd.DataFrame) -> pd.DataFrame:
    bucket_df = bucket_df.copy()
    n = len(bucket_df)

    if n <= 1:
        bucket_df["cluster_id"] = 0
        bucket_df["dist_to_center"] = 0.0
        return bucket_df

    X, _ = prepare_features(bucket_df)
    k = choose_k(n)

    if k <= 1:
        bucket_df["cluster_id"] = 0
        center = X.mean(axis=0, keepdims=True)
        dist = np.linalg.norm(X - center, axis=1)
        bucket_df["dist_to_center"] = dist
        return bucket_df

    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = model.fit_predict(X)
    centers = model.cluster_centers_

    dists = np.zeros(len(X), dtype=float)
    for i in range(len(X)):
        dists[i] = np.linalg.norm(X[i] - centers[labels[i]])

    bucket_df["cluster_id"] = labels
    bucket_df["dist_to_center"] = dists
    return bucket_df


# ============================================================
# 10. 样本优先级
# ============================================================

def compute_signal_priority(row: pd.Series) -> float:
    """
    分数越高，越优先保留给 LLM。
    """
    score = 0.0

    score += float(pd.to_numeric(row.get("strong_rule_count", 0), errors="coerce") or 0) * 3.0
    score += float(pd.to_numeric(row.get("adult_domain_rule_count", 0), errors="coerce") or 0) * 2.8
    score += float(pd.to_numeric(row.get("adult_format_rule_count", 0), errors="coerce") or 0) * 2.5
    score += float(pd.to_numeric(row.get("adult_consistency_rule_count", 0), errors="coerce") or 0) * 2.2
    score += float(pd.to_numeric(row.get("schema_rule_count", 0), errors="coerce") or 0) * 1.8
    score += float(pd.to_numeric(row.get("fd_like_count", 0), errors="coerce") or 0) * 1.2
    score += float(pd.to_numeric(row.get("context_rule_count", 0), errors="coerce") or 0) * 1.0
    score += float(pd.to_numeric(row.get("rare_value_count", 0), errors="coerce") or 0) * 0.4
    score += float(pd.to_numeric(row.get("rare_value_count", 0), errors="coerce") or 0) * 0.4

    score += min(float(pd.to_numeric(row.get("conflict_score", 0), errors="coerce") or 0) / 20.0, 5.0)
    score += float(pd.to_numeric(row.get("adult_consistency_score", 0), errors="coerce") or 0) * 1.5
    score += float(pd.to_numeric(row.get("posterior_error_probability", 0), errors="coerce") or 0) * 2.0

    nb = str(row.get("neighbor_bucket", ""))
    if nb == "strong_support_other":
        score += 2.0
    elif nb == "medium_support_other":
        score += 1.0
    elif nb == "strong_support_current":
        score -= 0.5

    db = str(row.get("domain_bucket", ""))
    if db == "domain_invalid":
        score += 2.5

    acb = str(row.get("adult_consistency_bucket", ""))
    if acb == "strong_adult_consistency_signal":
        score += 2.0
    elif acb == "adult_consistency_flag":
        score += 1.2

    return float(score)


def is_weak_signal_row(row: pd.Series) -> bool:
    strong = pd.to_numeric(row.get("strong_rule_count", 0), errors="coerce")
    domain = pd.to_numeric(row.get("adult_domain_rule_count", 0), errors="coerce")
    fmt = pd.to_numeric(row.get("adult_format_rule_count", 0), errors="coerce")
    cons = pd.to_numeric(row.get("adult_consistency_rule_count", 0), errors="coerce")
    schema = pd.to_numeric(row.get("schema_rule_count", 0), errors="coerce")
    fd = pd.to_numeric(row.get("fd_like_count", 0), errors="coerce")
    ctx = pd.to_numeric(row.get("context_rule_count", 0), errors="coerce")

    strong = 0 if pd.isna(strong) else strong
    domain = 0 if pd.isna(domain) else domain
    fmt = 0 if pd.isna(fmt) else fmt
    cons = 0 if pd.isna(cons) else cons
    schema = 0 if pd.isna(schema) else schema
    fd = 0 if pd.isna(fd) else fd
    ctx = 0 if pd.isna(ctx) else ctx

    return (strong + domain + fmt + cons + schema + fd + ctx) <= 0


# ============================================================
# 11. 每簇初步采样
# ============================================================

def sample_from_cluster(cluster_df: pd.DataFrame) -> pd.DataFrame:
    cluster_df = cluster_df.copy().sort_values("dist_to_center").reset_index(drop=True)
    n = len(cluster_df)

    cluster_df["sample_role"] = ""
    cluster_df["is_sampled_raw"] = 0

    if n <= SMALL_BUCKET_FULL_SAMPLE_THRESHOLD:
        cluster_df["sample_role"] = "all"
        cluster_df["is_sampled_raw"] = 1
        return cluster_df

    sampled_indices = []
    roles = {}

    if SAMPLE_CENTER:
        center_idx = 0
        sampled_indices.append(center_idx)
        roles[center_idx] = "center"

    if SAMPLE_OUTLIER:
        outlier_idx = n - 1
        if outlier_idx not in sampled_indices:
            sampled_indices.append(outlier_idx)
            roles[outlier_idx] = "outlier"

    if SAMPLE_BOUNDARY:
        boundary_idx = max(0, int(round(0.75 * (n - 1))))
        if boundary_idx not in sampled_indices:
            sampled_indices.append(boundary_idx)
            roles[boundary_idx] = "boundary"

    # token 控制：每簇最多保留 MAX_SAMPLES_PER_CLUSTER 个。
    if len(sampled_indices) > MAX_SAMPLES_PER_CLUSTER:
        # center 优先，然后 outlier，再 boundary。
        role_priority = {"center": 3, "outlier": 2, "boundary": 1, "all": 4}
        sampled_indices = sorted(
            sampled_indices,
            key=lambda idx: role_priority.get(roles.get(idx, ""), 0),
            reverse=True
        )[:MAX_SAMPLES_PER_CLUSTER]

    for idx in sampled_indices:
        cluster_df.loc[idx, "sample_role"] = roles[idx]
        cluster_df.loc[idx, "is_sampled_raw"] = 1

    return cluster_df


# ============================================================
# 12. 预算裁剪
# ============================================================

def downsample_weak_signals(sampled_df: pd.DataFrame) -> pd.DataFrame:
    if not ENABLE_WEAK_SIGNAL_DOWNSAMPLE or len(sampled_df) == 0:
        return sampled_df

    sampled_df = sampled_df.copy()
    sampled_df["_is_weak_signal"] = sampled_df.apply(is_weak_signal_row, axis=1).astype(int)

    strong_df = sampled_df[sampled_df["_is_weak_signal"] == 0].copy()
    weak_df = sampled_df[sampled_df["_is_weak_signal"] == 1].copy()

    if len(weak_df) == 0:
        return sampled_df.drop(columns=["_is_weak_signal"])

    keep_n = max(1, int(round(len(weak_df) * WEAK_SIGNAL_KEEP_RATIO)))
    weak_df = weak_df.sort_values(
        by=["signal_priority", "conflict_score", "posterior_error_probability"],
        ascending=[False, False, False]
    ).head(keep_n)

    out = pd.concat([strong_df, weak_df], ignore_index=True)
    out = out.drop(columns=["_is_weak_signal"])
    return out


def enforce_bucket_budget(sampled_df: pd.DataFrame) -> pd.DataFrame:
    if len(sampled_df) == 0:
        return sampled_df

    kept_parts = []
    for bucket_id, group in sampled_df.groupby("bucket_id", sort=False):
        if len(group) <= MAX_SAMPLES_PER_BUCKET:
            kept_parts.append(group)
            continue
        g = group.sort_values(
            by=["signal_priority", "sample_role_priority", "dist_to_center"],
            ascending=[False, False, True]
        ).head(MAX_SAMPLES_PER_BUCKET)
        kept_parts.append(g)

    return pd.concat(kept_parts, ignore_index=True) if kept_parts else sampled_df.iloc[0:0].copy()


def enforce_column_budget(sampled_df: pd.DataFrame) -> pd.DataFrame:
    if len(sampled_df) == 0:
        return sampled_df

    kept_parts = []
    for col, group in sampled_df.groupby("column", sort=False):
        max_n = MAX_SAMPLES_PER_COLUMN.get(col, DEFAULT_MAX_SAMPLES_PER_COLUMN)
        if len(group) <= max_n:
            kept_parts.append(group)
            continue

        g = group.sort_values(
            by=["signal_priority", "sample_role_priority", "conflict_score", "posterior_error_probability"],
            ascending=[False, False, False, False]
        ).head(max_n)
        kept_parts.append(g)

    return pd.concat(kept_parts, ignore_index=True) if kept_parts else sampled_df.iloc[0:0].copy()


def ensure_min_column_coverage(sampled_df: pd.DataFrame, clustered_df: pd.DataFrame) -> pd.DataFrame:
    """
    在总预算允许的情况下，给每个列补足少量样本，避免某些列完全没有 LLM 标签。
    """
    if len(clustered_df) == 0:
        return sampled_df

    sampled_df = sampled_df.copy()
    selected_keys = set(zip(sampled_df["row_id"].astype(int), sampled_df["column"].astype(str)))

    add_parts = []

    for col, pool in clustered_df.groupby("column", sort=False):
        current_n = int((sampled_df["column"] == col).sum()) if len(sampled_df) > 0 else 0
        need = max(0, MIN_SAMPLES_PER_COLUMN_IF_AVAILABLE - current_n)
        if need <= 0:
            continue

        max_for_col = MAX_SAMPLES_PER_COLUMN.get(col, DEFAULT_MAX_SAMPLES_PER_COLUMN)
        need = min(need, max_for_col - current_n)
        if need <= 0:
            continue

        pool = pool.copy()
        pool["_key"] = list(zip(pool["row_id"].astype(int), pool["column"].astype(str)))
        pool = pool[~pool["_key"].isin(selected_keys)].copy()
        if len(pool) == 0:
            continue

        pool = pool.sort_values(
            by=["signal_priority", "conflict_score", "posterior_error_probability"],
            ascending=[False, False, False]
        ).head(need)
        pool = pool.drop(columns=["_key"])
        add_parts.append(pool)

        for _, r in pool.iterrows():
            selected_keys.add((int(r["row_id"]), str(r["column"])))

    if add_parts:
        sampled_df = pd.concat([sampled_df] + add_parts, ignore_index=True)

    return sampled_df


def enforce_total_budget(sampled_df: pd.DataFrame) -> pd.DataFrame:
    if len(sampled_df) <= MAX_TOTAL_SAMPLES:
        return sampled_df

    sampled_df = sampled_df.copy()

    # 先按列做 round-robin，避免大列吃掉全部预算。
    parts = []
    groups = {
        col: g.sort_values(
            by=["signal_priority", "sample_role_priority", "conflict_score", "posterior_error_probability"],
            ascending=[False, False, False, False]
        ).reset_index(drop=True)
        for col, g in sampled_df.groupby("column", sort=False)
    }

    selected_indices = []
    round_idx = 0
    while len(selected_indices) < MAX_TOTAL_SAMPLES:
        added = False
        for col, g in groups.items():
            if round_idx < len(g):
                selected_indices.append(g.index[round_idx] if "_orig_idx" not in g.columns else g["_orig_idx"].iloc[round_idx])
                added = True
                if len(selected_indices) >= MAX_TOTAL_SAMPLES:
                    break
        if not added:
            break
        round_idx += 1

    # 为了避免 index 混乱，使用预设 _orig_idx。
    if "_orig_idx" in sampled_df.columns:
        out = sampled_df[sampled_df["_orig_idx"].isin(selected_indices)].copy()
    else:
        out = sampled_df.sort_values(
            by=["signal_priority", "sample_role_priority", "conflict_score", "posterior_error_probability"],
            ascending=[False, False, False, False]
        ).head(MAX_TOTAL_SAMPLES).copy()

    if len(out) > MAX_TOTAL_SAMPLES:
        out = out.sort_values(
            by=["signal_priority", "sample_role_priority", "conflict_score", "posterior_error_probability"],
            ascending=[False, False, False, False]
        ).head(MAX_TOTAL_SAMPLES).copy()

    return out


def apply_sampling_budget(clustered_df: pd.DataFrame) -> pd.DataFrame:
    work = clustered_df.copy()

    work["signal_priority"] = work.apply(compute_signal_priority, axis=1)

    role_priority = {
        "all": 4,
        "center": 3,
        "outlier": 2,
        "boundary": 1,
        "": 0,
    }
    work["sample_role_priority"] = work["sample_role"].map(lambda x: role_priority.get(str(x), 0))

    sampled = work[work["is_sampled_raw"] == 1].copy()
    sampled["_orig_idx"] = np.arange(len(sampled))

    sampled = downsample_weak_signals(sampled)
    sampled = enforce_bucket_budget(sampled)
    sampled = enforce_column_budget(sampled)
    sampled = ensure_min_column_coverage(sampled, work)
    sampled = enforce_column_budget(sampled)

    # 重新设置 _orig_idx，方便 total budget。
    sampled = sampled.reset_index(drop=True)
    sampled["_orig_idx"] = np.arange(len(sampled))
    sampled = enforce_total_budget(sampled)

    sampled = sampled.copy()
    sampled["is_sampled"] = 1

    final_keys = set(zip(sampled["row_id"].astype(int), sampled["column"].astype(str)))
    work["is_sampled"] = work.apply(
        lambda r: 1 if (int(r["row_id"]), str(r["column"])) in final_keys else 0,
        axis=1
    )

    # 如果某些 raw sample 被裁掉，则 sample_role 保留原信息，但 is_sampled=0。
    return work


# ============================================================
# 13. 输出 JSON 兼容
# ============================================================

def json_safe_value(v):
    if pd.isna(v):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def row_to_summary_context(row: pd.Series) -> Dict[str, Any]:
    keys = [
        "semantic_type",
        "detected_type",
        "violation_count",
        "conflict_score",
        "value_frequency",
        "value_frequency_rank",
        "age_lower",
        "age_upper",
        "hours_lower",
        "hours_upper",
        "canonical_income",
        "domain_valid",
        "education_order",
        "adult_value_bucket",
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
        "pattern_bucket",
        "rarity_bucket",
        "neighbor_bucket",
        "domain_bucket",
        "adult_consistency_bucket",
        "age_sampling_bucket",
        "hours_sampling_bucket",
        "time_window_rule_count",
        "time_window_bucket",
        "time_value_bucket",
    ]
    return {k: json_safe_value(row.get(k)) for k in keys if k in row.index}


# ============================================================
# 14. 主流程
# ============================================================

def main():
    detail_map = load_detail_jsonl(INPUT_DETAIL_JSONL)
    summary_df = load_summary_csv(INPUT_SUMMARY_CSV, detail_map)
    summary_df = assign_buckets(summary_df)

    clustered_parts = []
    bucket_summary_rows = []

    for bucket_id, bucket_df in summary_df.groupby("bucket_id", sort=False):
        bucket_clustered = cluster_bucket(bucket_df)
        bucket_clustered_parts = []

        for cluster_id, cluster_df in bucket_clustered.groupby("cluster_id", sort=False):
            cluster_sampled = sample_from_cluster(cluster_df)
            bucket_clustered_parts.append(cluster_sampled)

            bucket_summary_rows.append({
                "bucket_id": bucket_id,
                "bucket_size": len(bucket_df),
                "cluster_id": int(cluster_id),
                "cluster_size": len(cluster_df),
                "raw_sampled_count": int(cluster_sampled["is_sampled_raw"].sum()),
                "semantic_type": str(cluster_df["semantic_type"].iloc[0]) if "semantic_type" in cluster_df.columns else "unknown",
                "main_rule_type": str(cluster_df["main_rule_type"].iloc[0]) if "main_rule_type" in cluster_df.columns else "unknown",
                "main_usage_role": str(cluster_df["main_usage_role"].iloc[0]) if "main_usage_role" in cluster_df.columns else "unknown",
                "pattern_bucket": str(cluster_df["pattern_bucket"].iloc[0]) if "pattern_bucket" in cluster_df.columns else "unknown",
                "rarity_bucket": str(cluster_df["rarity_bucket"].iloc[0]) if "rarity_bucket" in cluster_df.columns else "unknown",
                "neighbor_bucket": str(cluster_df["neighbor_bucket"].iloc[0]) if "neighbor_bucket" in cluster_df.columns else "unknown",
                "domain_bucket": str(cluster_df["domain_bucket"].iloc[0]) if "domain_bucket" in cluster_df.columns else "unknown",
                "adult_consistency_bucket": str(cluster_df["adult_consistency_bucket"].iloc[0]) if "adult_consistency_bucket" in cluster_df.columns else "unknown",
                "age_sampling_bucket": str(cluster_df["age_sampling_bucket"].iloc[0]) if "age_sampling_bucket" in cluster_df.columns else "unknown",
                "hours_sampling_bucket": str(cluster_df["hours_sampling_bucket"].iloc[0]) if "hours_sampling_bucket" in cluster_df.columns else "unknown",
            })

        clustered_bucket = pd.concat(bucket_clustered_parts, ignore_index=True)
        clustered_parts.append(clustered_bucket)

    clustered_df = pd.concat(clustered_parts, ignore_index=True) if clustered_parts else pd.DataFrame()
    clustered_df = apply_sampling_budget(clustered_df)

    clustered_df.to_csv(OUTPUT_CLUSTERED_CSV, index=False, encoding="utf-8-sig")

    sampled_df = clustered_df[clustered_df["is_sampled"] == 1].copy()
    sampled_df = sampled_df.sort_values(
        by=["column", "signal_priority", "conflict_score", "row_id"],
        ascending=[True, False, False, True]
    ).reset_index(drop=True)
    sampled_df.to_csv(OUTPUT_SAMPLED_CSV, index=False, encoding="utf-8-sig")

    with open(OUTPUT_SAMPLED_JSONL, "w", encoding="utf-8") as f:
        for _, row in sampled_df.iterrows():
            key = (int(row["row_id"]), row["column"])
            detail_obj = detail_map.get(key, {})

            out = {
                "row_id": int(row["row_id"]),
                "column": row["column"],
                "value": json_safe_value(row.get("value")),
                "bucket_id": row["bucket_id"],
                "cluster_id": int(row["cluster_id"]),
                "sample_role": row["sample_role"],
                "dist_to_center": float(row["dist_to_center"]),
                "signal_priority": float(row.get("signal_priority", 0.0)),
                "summary_context": row_to_summary_context(row),
                "detail_context": detail_obj
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    bucket_summary_df = pd.DataFrame(bucket_summary_rows)
    if len(bucket_summary_df) > 0:
        sampled_count_by_bucket = sampled_df.groupby("bucket_id").size().to_dict()
        bucket_summary_df["sampled_count_after_budget"] = bucket_summary_df["bucket_id"].map(
            lambda x: int(sampled_count_by_bucket.get(x, 0))
        )
    bucket_summary_df.to_csv(OUTPUT_BUCKET_SUMMARY_CSV, index=False, encoding="utf-8-sig")

    summary = {
        "total_candidates": int(len(clustered_df)),
        "raw_sampled_count_before_budget": int(clustered_df["is_sampled_raw"].sum()) if "is_sampled_raw" in clustered_df.columns else 0,
        "final_sampled_count": int(len(sampled_df)),
        "max_total_samples": int(MAX_TOTAL_SAMPLES),
        "use_column_in_bucket": bool(USE_COLUMN_IN_BUCKET),
        "max_samples_per_bucket": int(MAX_SAMPLES_PER_BUCKET),
        "max_samples_per_cluster": int(MAX_SAMPLES_PER_CLUSTER),
        "weak_signal_downsample_enabled": bool(ENABLE_WEAK_SIGNAL_DOWNSAMPLE),
        "weak_signal_keep_ratio": float(WEAK_SIGNAL_KEEP_RATIO),
        "sampled_by_column": sampled_df["column"].value_counts(dropna=False).to_dict() if len(sampled_df) else {},
        "sample_role_distribution": sampled_df["sample_role"].value_counts(dropna=False).to_dict() if len(sampled_df) else {},
        "main_rule_type_distribution_top30": sampled_df["main_rule_type"].value_counts(dropna=False).head(30).to_dict() if len(sampled_df) else {},
    }

    with open(OUTPUT_SAMPLE_BUDGET_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[OK] 完整聚类结果已保存: {OUTPUT_CLUSTERED_CSV}")
    print(f"[OK] 采样摘要 CSV 已保存: {OUTPUT_SAMPLED_CSV}")
    print(f"[OK] 采样详细 JSONL 已保存: {OUTPUT_SAMPLED_JSONL}")
    print(f"[OK] 分桶/聚类摘要已保存: {OUTPUT_BUCKET_SUMMARY_CSV}")
    print(f"[OK] 采样预算摘要已保存: {OUTPUT_SAMPLE_BUDGET_SUMMARY_JSON}")

    print("\n===== 统计信息 =====")
    print(f"总候选数: {len(clustered_df)}")
    print(f"预算前 raw sampled 数: {int(clustered_df['is_sampled_raw'].sum()) if 'is_sampled_raw' in clustered_df.columns else 0}")
    print(f"最终采样数: {len(sampled_df)}")
    print(f"总桶数: {clustered_df['bucket_id'].nunique() if len(clustered_df) else 0}")
    print(f"总簇数: {clustered_df[['bucket_id', 'cluster_id']].drop_duplicates().shape[0] if len(clustered_df) else 0}")

    print("\n按列采样分布:")
    if len(sampled_df):
        print(sampled_df["column"].value_counts(dropna=False).to_string())
    else:
        print("(empty)")

    print("\n采样角色分布:")
    if len(sampled_df):
        print(sampled_df["sample_role"].value_counts(dropna=False).to_string())
    else:
        print("(empty)")


if __name__ == "__main__":
    main()

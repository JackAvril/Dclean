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

# 输入1：错误的大致上下文（用于聚类）
INPUT_SUMMARY_CSV = "candidate_llm_contexts_flat_v2.csv"

# 输入2：错误的详细上下文（用于补全给 LLM）
INPUT_DETAIL_JSONL = "candidate_llm_contexts_v2.jsonl"

# 输出
OUTPUT_CLUSTERED_CSV = "candidate_clustered_v2.csv"
OUTPUT_SAMPLED_CSV = "candidate_sampled_v2.csv"
OUTPUT_SAMPLED_JSONL = "candidate_sampled_with_context_v2.jsonl"
OUTPUT_BUCKET_SUMMARY_CSV = "bucket_cluster_summary_v2.csv"

# 是否把 column 纳入分桶
# True：同一列内部聚类，更保守
# False：跨列通用聚类，更像通用方法
USE_COLUMN_IN_BUCKET = False

# 聚类参数
MIN_BUCKET_SIZE_FOR_CLUSTER = 6
MAX_CLUSTERS_PER_BUCKET = 8
TARGET_CLUSTER_SIZE = 12
SMALL_BUCKET_FULL_SAMPLE_THRESHOLD = 5

# 每簇采样策略
SAMPLE_CENTER = True
SAMPLE_BOUNDARY = True
SAMPLE_OUTLIER = True

RANDOM_STATE = 42


# ============================================================
# 2. 读取详细 jsonl，建立索引
# ============================================================

def load_detail_jsonl(path: str) -> Dict[Tuple[int, str], Dict[str, Any]]:
    """
    建立 (row_id, column) -> 详细上下文对象 的映射
    """
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

    df["main_rule_type"] = main_rule_types
    df["main_usage_role"] = main_usage_roles
    df["pattern_bucket"] = pattern_buckets

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

    # 确保新增数值字段存在
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
    ]
    for col in numeric_defaults:
        if col not in df.columns:
            df[col] = 0

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
]

FEATURE_COLUMNS_CATEGORICAL = [
    "semantic_type",
    "main_rule_type",
    "main_usage_role",
    "pattern_bucket",
    "rarity_bucket",
    "neighbor_bucket",
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
# 10. 每簇采样
# ============================================================

def sample_from_cluster(cluster_df: pd.DataFrame) -> pd.DataFrame:
    cluster_df = cluster_df.copy().sort_values("dist_to_center").reset_index(drop=True)
    n = len(cluster_df)

    if n <= SMALL_BUCKET_FULL_SAMPLE_THRESHOLD:
        cluster_df["sample_role"] = "all"
        cluster_df["is_sampled"] = 1
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

    cluster_df["sample_role"] = ""
    cluster_df["is_sampled"] = 0

    for idx in sampled_indices:
        cluster_df.loc[idx, "sample_role"] = roles[idx]
        cluster_df.loc[idx, "is_sampled"] = 1

    return cluster_df


# ============================================================
# 11. 主流程
# ============================================================

def main():
    # 读取详细上下文
    detail_map = load_detail_jsonl(INPUT_DETAIL_JSONL)

    # 读取摘要上下文并补充字段
    summary_df = load_summary_csv(INPUT_SUMMARY_CSV, detail_map)

    # 分桶
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
                "sampled_count": int(cluster_sampled["is_sampled"].sum()),
                "semantic_type": str(cluster_df["semantic_type"].iloc[0]) if "semantic_type" in cluster_df.columns else "unknown",
                "main_rule_type": str(cluster_df["main_rule_type"].iloc[0]) if "main_rule_type" in cluster_df.columns else "unknown",
                "main_usage_role": str(cluster_df["main_usage_role"].iloc[0]) if "main_usage_role" in cluster_df.columns else "unknown",
                "pattern_bucket": str(cluster_df["pattern_bucket"].iloc[0]) if "pattern_bucket" in cluster_df.columns else "unknown",
                "rarity_bucket": str(cluster_df["rarity_bucket"].iloc[0]) if "rarity_bucket" in cluster_df.columns else "unknown",
                "neighbor_bucket": str(cluster_df["neighbor_bucket"].iloc[0]) if "neighbor_bucket" in cluster_df.columns else "unknown",
            })

        clustered_bucket = pd.concat(bucket_clustered_parts, ignore_index=True)
        clustered_parts.append(clustered_bucket)

    clustered_df = pd.concat(clustered_parts, ignore_index=True)

    # 保存完整聚类结果
    clustered_df.to_csv(OUTPUT_CLUSTERED_CSV, index=False, encoding="utf-8-sig")

    # 抽样结果
    sampled_df = clustered_df[clustered_df["is_sampled"] == 1].copy()
    sampled_df.to_csv(OUTPUT_SAMPLED_CSV, index=False, encoding="utf-8-sig")

    # 导出采样后的完整上下文
    with open(OUTPUT_SAMPLED_JSONL, "w", encoding="utf-8") as f:
        for _, row in sampled_df.iterrows():
            key = (int(row["row_id"]), row["column"])
            detail_obj = detail_map.get(key, {})

            out = {
                "row_id": int(row["row_id"]),
                "column": row["column"],
                "value": row.get("value"),
                "bucket_id": row["bucket_id"],
                "cluster_id": int(row["cluster_id"]),
                "sample_role": row["sample_role"],
                "dist_to_center": float(row["dist_to_center"]),
                "summary_context": {
                    "semantic_type": row.get("semantic_type"),
                    "violation_count": row.get("violation_count"),
                    "conflict_score": row.get("conflict_score"),
                    "value_frequency": row.get("value_frequency"),
                    "value_frequency_rank": row.get("value_frequency_rank"),
                    "neighbor_majority_value": row.get("neighbor_majority_value"),
                    "neighbor_majority_ratio": row.get("neighbor_majority_ratio"),
                    "prior_error_probability": row.get("prior_error_probability"),
                    "posterior_error_probability": row.get("posterior_error_probability"),
                    "main_rule_type": row.get("main_rule_type"),
                    "main_usage_role": row.get("main_usage_role"),
                    "strong_rule_count": row.get("strong_rule_count"),
                    "candidate_generation_rule_count": row.get("candidate_generation_rule_count"),
                    "fd_like_count": row.get("fd_like_count"),
                    "context_rule_count": row.get("context_rule_count"),
                    "global_rule_count": row.get("global_rule_count"),
                    "rare_value_count": row.get("rare_value_count"),
                    "typo_rule_count": row.get("typo_rule_count"),
                    "pattern_rule_count": row.get("pattern_rule_count"),
                    "schema_rule_count": row.get("schema_rule_count"),
                    "pattern_bucket": row.get("pattern_bucket"),
                    "rarity_bucket": row.get("rarity_bucket"),
                    "neighbor_bucket": row.get("neighbor_bucket"),
                },
                "detail_context": detail_obj
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    # 保存桶摘要
    pd.DataFrame(bucket_summary_rows).to_csv(
        OUTPUT_BUCKET_SUMMARY_CSV, index=False, encoding="utf-8-sig"
    )

    print(f"[OK] 完整聚类结果已保存: {OUTPUT_CLUSTERED_CSV}")
    print(f"[OK] 采样摘要 CSV 已保存: {OUTPUT_SAMPLED_CSV}")
    print(f"[OK] 采样详细 JSONL 已保存: {OUTPUT_SAMPLED_JSONL}")
    print(f"[OK] 分桶/聚类摘要已保存: {OUTPUT_BUCKET_SUMMARY_CSV}")

    print("\n===== 统计信息 =====")
    print(f"总候选数: {len(clustered_df)}")
    print(f"总采样数: {len(sampled_df)}")
    print(f"总桶数: {clustered_df['bucket_id'].nunique()}")
    print(f"总簇数: {clustered_df[['bucket_id', 'cluster_id']].drop_duplicates().shape[0]}")

    print("\n采样角色分布:")
    print(sampled_df["sample_role"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
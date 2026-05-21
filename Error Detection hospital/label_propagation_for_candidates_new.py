import json
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. 用户配置区
# ============================================================

# 输入1：完整聚类结果（新版）
INPUT_CLUSTERED_CSV = "candidate_clustered_v2.csv"

# 输入2：LLM 标注结果（新版）
INPUT_LABELED_CSV = "candidate_sampled_labeled_v2.csv"

# 输出
OUTPUT_JSONL = "propagated_labels_v2.jsonl"
OUTPUT_CSV = "propagated_labels_v2.csv"

# -------- Cluster 内传播参数（收紧版）--------
MIN_LABELED_PER_CLUSTER = 3
CLUSTER_PROPAGATION_THRESHOLD = 0.85

# 只给簇内“中心区域”的未标注样本传播
ENABLE_CLUSTER_CENTER_FILTER = True
CLUSTER_CENTER_DIST_QUANTILE = 0.70   # 仅传播给簇内 dist_to_center 最靠近中心的前 50%
MAX_CLUSTER_PROPAGATION_PER_CLUSTER = 50

# -------- KNN 传播参数（收紧版）--------
ENABLE_KNN_PROPAGATION = True
KNN_K = 5
KNN_PROPAGATION_THRESHOLD = 0.80
MIN_TEACHER_SAMPLES_FOR_KNN = 20
KNN_WITHIN_SAME_COLUMN_ONLY = True

# KNN 额外稳健性约束
KNN_MIN_MARGIN = 0.10       # error/correct 两边差距不够大，不传播
KNN_DISTANCE_EPS = 1e-8        # 距离权重数值稳定项
KNN_MAX_AVG_DISTANCE = 4.5     # 平均邻居距离过大，说明局部不可靠，不传播

# -------- 样本权重基值（更保守）--------
WEIGHT_LLM = 1.0
WEIGHT_CLUSTER_PROP = 0.7
WEIGHT_KNN_PROP = 0.5

# -------- 最低传播置信度过滤 --------
MIN_PROPAGATION_CONFIDENCE = 0.65


# ============================================================
# 2. 特征列配置（适配新版）
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
    "dist_to_center",
]

FEATURE_COLUMNS_CATEGORICAL = [
    "column",
    "semantic_type",
    "main_rule_type",
    "main_usage_role",
    "pattern_bucket",
    "rarity_bucket",
    "neighbor_bucket",
    "sample_role",
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


def label_to_name(y: int) -> str:
    return "error" if int(y) == 1 else "correct"


def build_key(row_id: Any, column: Any) -> Tuple[int, str]:
    return int(row_id), str(column)


def compute_weight(base_weight: float, conf: float) -> float:
    conf = min(max(float(conf), 0.0), 1.0)
    return round(base_weight * conf, 6)


def safe_none_fill(df: pd.DataFrame, idx: int):
    """
    保持输出字段结构不变；传播样本没有 LLM 理由时置空。
    """
    if pd.isna(df.at[idx, "error_type"]) or df.at[idx, "error_type"] is None:
        df.at[idx, "error_type"] = None
    if pd.isna(df.at[idx, "reason_short"]) or df.at[idx, "reason_short"] is None:
        df.at[idx, "reason_short"] = None
    if pd.isna(df.at[idx, "reason_detailed"]) or df.at[idx, "reason_detailed"] is None:
        df.at[idx, "reason_detailed"] = None
    if pd.isna(df.at[idx, "suggested_correct_value"]) or df.at[idx, "suggested_correct_value"] is None:
        df.at[idx, "suggested_correct_value"] = None
    if pd.isna(df.at[idx, "needs_human_review"]) or df.at[idx, "needs_human_review"] is None:
        df.at[idx, "needs_human_review"] = None
    if pd.isna(df.at[idx, "evidence_used"]) or df.at[idx, "evidence_used"] is None:
        df.at[idx, "evidence_used"] = None


# ============================================================
# 4. 读取数据
# ============================================================

def load_clustered_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["row_id"] = df["row_id"].astype(int)
    df["column"] = df["column"].astype(str)

    expected_defaults = {
        "semantic_type": "unknown",
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
        "context_rule_count": 0.0,
        "global_rule_count": 0.0,
        "rare_value_count": 0.0,
        "typo_rule_count": 0.0,
        "pattern_rule_count": 0.0,
        "schema_rule_count": 0.0,
        "pattern_bucket": "unknown",
        "rarity_bucket": "unknown",
        "neighbor_bucket": "unknown",
        "bucket_id": "unknown",
        "cluster_id": -1,
        "dist_to_center": 0.0,
        "sample_role": "",
        "is_sampled": 0,
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
        "context_rule_count",
        "global_rule_count",
        "rare_value_count",
        "typo_rule_count",
        "pattern_rule_count",
        "schema_rule_count",
        "cluster_id",
        "dist_to_center",
        "is_sampled",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_labeled_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["row_id"] = df["row_id"].astype(int)
    df["column"] = df["column"].astype(str)

    if "label" in df.columns:
        df["label_binary"] = df["label"].map(normalize_label)
    elif "is_error" in df.columns:
        df["label_binary"] = df["is_error"].map(parse_is_error)
    else:
        raise ValueError("标注文件中必须包含 label 或 is_error 列。")

    if "confidence" not in df.columns:
        df["confidence"] = 1.0
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(1.0)

    df = df[pd.notna(df["label_binary"])].copy()
    df["label_binary"] = df["label_binary"].astype(int)

    return df


# ============================================================
# 5. 合并 clustered + labeled
# ============================================================

def merge_clustered_and_labeled(clustered_df: pd.DataFrame, labeled_df: pd.DataFrame) -> pd.DataFrame:
    df = clustered_df.copy()

    label_map: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for _, row in labeled_df.iterrows():
        key = build_key(row["row_id"], row["column"])
        label_map[key] = {
            "label_binary": int(row["label_binary"]),
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
# 7. 第一层传播：更稳健的簇内传播
# ============================================================

def cluster_label_propagation(df: pd.DataFrame) -> pd.DataFrame:
    """
    改进点：
    1. 提高最小老师数与传播阈值
    2. 不再整簇全传播，只传播中心区域样本
    3. 每个 cluster 最多传播固定数量，避免大面积污染
    4. 传播置信度加入 teacher support 惩罚
    """
    df = df.copy()

    for (bucket_id, cluster_id), grp in df.groupby(["bucket_id", "cluster_id"], sort=False):
        grp_idx = grp.index.tolist()
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

        # 额外按 teacher 数量做置信度折扣，减少小簇误传播
        teacher_support_factor = min(1.0, total / 10.0)
        propagated_conf = float(propagated_conf) * teacher_support_factor

        if propagated_conf < MIN_PROPAGATION_CONFIDENCE:
            continue

        unlabeled_grp = grp[pd.isna(grp["final_label_binary"])].copy()
        if len(unlabeled_grp) == 0:
            continue

        # 只传播给簇中心附近样本，避免簇边界污染
        if ENABLE_CLUSTER_CENTER_FILTER:
            unlabeled_grp["dist_to_center"] = pd.to_numeric(
                unlabeled_grp["dist_to_center"], errors="coerce"
            ).fillna(np.inf)
            dist_threshold = unlabeled_grp["dist_to_center"].quantile(CLUSTER_CENTER_DIST_QUANTILE)
            unlabeled_grp = unlabeled_grp[unlabeled_grp["dist_to_center"] <= dist_threshold].copy()

        if len(unlabeled_grp) == 0:
            continue

        # 限制每个簇最多传播数
        unlabeled_grp = unlabeled_grp.sort_values(
            by="dist_to_center", ascending=True, na_position="last"
        ).head(MAX_CLUSTER_PROPAGATION_PER_CLUSTER)

        target_indices = unlabeled_grp.index.tolist()

        for idx in target_indices:
            df.at[idx, "final_label_binary"] = propagated_label
            df.at[idx, "final_label"] = label_to_name(propagated_label)
            df.at[idx, "propagation_confidence"] = round(float(propagated_conf), 6)
            df.at[idx, "label_source"] = "cluster_propagation"
            df.at[idx, "sample_weight"] = compute_weight(WEIGHT_CLUSTER_PROP, propagated_conf)
            safe_none_fill(df, idx)

    return df


# ============================================================
# 8. 第二层传播：距离加权 KNN 传播
# ============================================================

def _weighted_neighbor_vote(neighbor_labels: np.ndarray, neighbor_distances: np.ndarray) -> Tuple[Optional[int], Optional[float]]:
    """
    距离加权投票：
    - 距离越近，权重越大
    - 同时要求 error/correct 之间有足够 margin
    """
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
    """
    改进点：
    1. 使用距离加权投票代替简单多数比例
    2. 加入 margin 约束，避免边界点被硬传
    3. 邻居平均距离过大则拒绝传播
    4. 仍保持输出格式不变
    """
    df = df.copy()

    teacher_mask = (df["label_source"] == "llm")
    teacher_indices = df[teacher_mask].index.tolist()

    if len(teacher_indices) < MIN_TEACHER_SAMPLES_FOR_KNN:
        return df

    if KNN_WITHIN_SAME_COLUMN_ONLY:
        all_columns = df["column"].dropna().astype(str).unique().tolist()

        for col_name in all_columns:
            col_teacher_indices = df[
                (df["label_source"] == "llm") & (df["column"] == col_name)
            ].index.tolist()

            col_unlabeled_indices = df[
                (df["final_label_binary"].isna()) & (df["column"] == col_name)
            ].index.tolist()

            if len(col_teacher_indices) < max(3, min(KNN_K, MIN_TEACHER_SAMPLES_FOR_KNN // 4)):
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
                if propagated_conf < MIN_PROPAGATION_CONFIDENCE:
                    continue

                df.at[idx, "final_label_binary"] = propagated_label
                df.at[idx, "final_label"] = label_to_name(propagated_label)
                df.at[idx, "propagation_confidence"] = round(float(propagated_conf), 6)
                df.at[idx, "label_source"] = "knn_propagation"
                df.at[idx, "sample_weight"] = compute_weight(WEIGHT_KNN_PROP, propagated_conf)
                safe_none_fill(df, idx)

    else:
        X_teacher = X[teacher_indices]
        y_teacher = df.loc[teacher_indices, "final_label_binary"].astype(int).values

        nn_model = NearestNeighbors(
            n_neighbors=min(KNN_K, len(teacher_indices)),
            metric="euclidean"
        )
        nn_model.fit(X_teacher)

        unlabeled_indices = df[df["final_label_binary"].isna()].index.tolist()

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
            if propagated_conf < MIN_PROPAGATION_CONFIDENCE:
                continue

            df.at[idx, "final_label_binary"] = propagated_label
            df.at[idx, "final_label"] = label_to_name(propagated_label)
            df.at[idx, "propagation_confidence"] = round(float(propagated_conf), 6)
            df.at[idx, "label_source"] = "knn_propagation"
            df.at[idx, "sample_weight"] = compute_weight(WEIGHT_KNN_PROP, propagated_conf)
            safe_none_fill(df, idx)

    return df


# ============================================================
# 9. 导出结果（保持不变）
# ============================================================

def export_results(df: pd.DataFrame, output_jsonl: str, output_csv: str):
    out_rows = []

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            if pd.isna(row["final_label_binary"]):
                continue

            obj = {
                "row_id": int(row["row_id"]),
                "column": str(row["column"]),
                "value": row.get("value"),
                "bucket_id": row.get("bucket_id"),
                "cluster_id": int(row["cluster_id"]) if pd.notna(row["cluster_id"]) else None,
                "sample_role": row.get("sample_role"),
                "is_sampled": int(row["is_sampled"]) if pd.notna(row["is_sampled"]) else None,
                "dist_to_center": float(row["dist_to_center"]) if pd.notna(row["dist_to_center"]) else None,

                "label": row["final_label"],
                "label_binary": int(row["final_label_binary"]),

                "propagation_confidence": float(row["propagation_confidence"]) if pd.notna(row["propagation_confidence"]) else None,
                "label_source": row.get("label_source"),
                "sample_weight": float(row["sample_weight"]) if pd.notna(row["sample_weight"]) else None,

                "semantic_type": row.get("semantic_type"),
                "main_rule_type": row.get("main_rule_type"),
                "main_usage_role": row.get("main_usage_role"),
                "pattern_bucket": row.get("pattern_bucket"),
                "rarity_bucket": row.get("rarity_bucket"),
                "neighbor_bucket": row.get("neighbor_bucket"),

                "violation_count": float(row["violation_count"]) if pd.notna(row["violation_count"]) else None,
                "conflict_score": float(row["conflict_score"]) if pd.notna(row["conflict_score"]) else None,
                "value_frequency": float(row["value_frequency"]) if pd.notna(row["value_frequency"]) else None,
                "value_frequency_rank": float(row["value_frequency_rank"]) if pd.notna(row["value_frequency_rank"]) else None,
                "neighbor_majority_value": row.get("neighbor_majority_value"),
                "neighbor_majority_ratio": float(row["neighbor_majority_ratio"]) if pd.notna(row["neighbor_majority_ratio"]) else None,
                "prior_error_probability": float(row["prior_error_probability"]) if pd.notna(row["prior_error_probability"]) else None,
                "posterior_error_probability": float(row["posterior_error_probability"]) if pd.notna(row["posterior_error_probability"]) else None,

                "strong_rule_count": float(row["strong_rule_count"]) if pd.notna(row["strong_rule_count"]) else None,
                "candidate_generation_rule_count": float(row["candidate_generation_rule_count"]) if pd.notna(row["candidate_generation_rule_count"]) else None,
                "fd_like_count": float(row["fd_like_count"]) if pd.notna(row["fd_like_count"]) else None,
                "context_rule_count": float(row["context_rule_count"]) if pd.notna(row["context_rule_count"]) else None,
                "global_rule_count": float(row["global_rule_count"]) if pd.notna(row["global_rule_count"]) else None,

                "rare_value_count": float(row["rare_value_count"]) if pd.notna(row["rare_value_count"]) else None,
                "typo_rule_count": float(row["typo_rule_count"]) if pd.notna(row["typo_rule_count"]) else None,
                "pattern_rule_count": float(row["pattern_rule_count"]) if pd.notna(row["pattern_rule_count"]) else None,
                "schema_rule_count": float(row["schema_rule_count"]) if pd.notna(row["schema_rule_count"]) else None,

                "error_type": row.get("error_type"),
                "reason_short": row.get("reason_short"),
                "reason_detailed": row.get("reason_detailed"),
                "suggested_correct_value": row.get("suggested_correct_value"),
                "needs_human_review": row.get("needs_human_review"),
                "evidence_used": row.get("evidence_used"),
            }

            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            out_rows.append(obj)

    pd.DataFrame(out_rows).to_csv(output_csv, index=False, encoding="utf-8-sig")


# ============================================================
# 10. 主流程（保持不变）
# ============================================================

def main():
    clustered_df = load_clustered_csv(INPUT_CLUSTERED_CSV)
    labeled_df = load_labeled_csv(INPUT_LABELED_CSV)

    df = merge_clustered_and_labeled(clustered_df, labeled_df)

    X, _ = prepare_features(df)

    df = cluster_label_propagation(df)

    if ENABLE_KNN_PROPAGATION:
        df = knn_label_propagation(df, X)

    export_results(df, OUTPUT_JSONL, OUTPUT_CSV)

    total = len(df)
    llm_count = int((df["label_source"] == "llm").sum())
    cluster_count = int((df["label_source"] == "cluster_propagation").sum())
    knn_count = int((df["label_source"] == "knn_propagation").sum())
    total_labeled = int(pd.notna(df["final_label_binary"]).sum())

    print(f"[OK] 传播结果 JSONL 已保存: {OUTPUT_JSONL}")
    print(f"[OK] 传播结果 CSV 已保存: {OUTPUT_CSV}")

    print("\n===== 统计信息 =====")
    print(f"总样本数: {total}")
    print(f"LLM原始标注数: {llm_count}")
    print(f"簇内传播数: {cluster_count}")
    print(f"KNN传播数: {knn_count}")
    print(f"最终可训练样本数: {total_labeled}")

    if total > 0:
        print(f"训练集覆盖率: {total_labeled / total:.6f}")

    print("\n标签来源分布:")
    print(df["label_source"].value_counts(dropna=False).to_string())

    print("\n最终标签分布:")
    print(df["final_label"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
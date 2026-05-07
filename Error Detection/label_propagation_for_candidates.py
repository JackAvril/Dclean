import json
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. 用户配置区
# ============================================================

# 输入1：完整聚类结果
INPUT_CLUSTERED_CSV = "candidate_clustered_new.csv"

# 输入2：LLM 标注结果
INPUT_LABELED_CSV = "candidate_sampled_labeled_new.csv"

# 输出
OUTPUT_JSONL = "propagated_labels_new.jsonl"
OUTPUT_CSV = "propagated_labels_new.csv"

# -------- Cluster 内传播参数 --------
# 同簇内已标注样本数至少多少，才允许传播
MIN_LABELED_PER_CLUSTER = 2

# 同簇一致度至少多少才传播
CLUSTER_PROPAGATION_THRESHOLD = 0.80

# -------- KNN 传播参数 --------
ENABLE_KNN_PROPAGATION = True
KNN_K = 5

# KNN 邻居一致度阈值
KNN_PROPAGATION_THRESHOLD = 0.80

# 已用于 KNN 的可作为教师的样本，至少要多少条
MIN_TEACHER_SAMPLES_FOR_KNN = 20

# KNN 是否只在同列内找邻居
KNN_WITHIN_SAME_COLUMN_ONLY = True

# -------- 样本权重基值 --------
WEIGHT_LLM = 1.0
WEIGHT_CLUSTER_PROP = 0.8
WEIGHT_KNN_PROP = 0.6

# -------- 最低传播置信度过滤 --------
MIN_PROPAGATION_CONFIDENCE = 0.60


# ============================================================
# 2. 特征列配置
# ============================================================

FEATURE_COLUMNS_NUMERIC = [
    "violation_count",
    "conflict_score",
    "value_frequency",
    "value_frequency_rank",
    "neighbor_majority_ratio",
    "prior_error_probability",
    "posterior_error_probability",
]

FEATURE_COLUMNS_CATEGORICAL = [
    "column",               # 关键：加入列名，避免跨列传播过度
    "semantic_type",
    "main_rule_type",
    "pattern_bucket",
    "rarity_bucket",
    "neighbor_bucket",
]


# ============================================================
# 3. 基础函数
# ============================================================

def normalize_label(x: Any) -> Optional[int]:
    """
    统一 label:
    error -> 1
    correct -> 0
    """
    if pd.isna(x) or x is None:
        return None
    s = str(x).strip().lower()
    if s == "error":
        return 1
    if s == "correct":
        return 0
    return None


def parse_is_error(x: Any) -> Optional[int]:
    """
    安全解析 is_error，避免 bool("False") == True 的问题
    """
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


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def compute_weight(base_weight: float, conf: float) -> float:
    """
    传播标签的训练权重 = 基值 * 传播置信度
    并限制到合理范围
    """
    conf = min(max(float(conf), 0.0), 1.0)
    return round(base_weight * conf, 6)


# ============================================================
# 4. 读取数据
# ============================================================

def load_clustered_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["row_id"] = df["row_id"].astype(int)
    df["column"] = df["column"].astype(str)
    return df


def load_labeled_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["row_id"] = df["row_id"].astype(int)
    df["column"] = df["column"].astype(str)

    # 统一 label
    if "label" in df.columns:
        df["label_binary"] = df["label"].map(normalize_label)
    elif "is_error" in df.columns:
        df["label_binary"] = df["is_error"].map(parse_is_error)
    else:
        raise ValueError("标注文件中必须包含 label 或 is_error 列。")

    # confidence
    if "confidence" not in df.columns:
        df["confidence"] = 1.0
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(1.0)

    # 过滤掉无效标签
    df = df[pd.notna(df["label_binary"])].copy()
    df["label_binary"] = df["label_binary"].astype(int)

    return df


# ============================================================
# 5. 合并 clustered + labeled
# ============================================================

def merge_clustered_and_labeled(clustered_df: pd.DataFrame, labeled_df: pd.DataFrame) -> pd.DataFrame:
    df = clustered_df.copy()

    label_map: Dict[Tuple[int, str], int] = {}
    conf_map: Dict[Tuple[int, str], float] = {}
    errtype_map: Dict[Tuple[int, str], Any] = {}
    reason_map: Dict[Tuple[int, str], Any] = {}

    for _, row in labeled_df.iterrows():
        key = build_key(row["row_id"], row["column"])
        label_map[key] = int(row["label_binary"])
        conf_map[key] = float(row["confidence"]) if pd.notna(row["confidence"]) else 1.0
        errtype_map[key] = row["error_type"] if "error_type" in row else None
        reason_map[key] = row["reason_short"] if "reason_short" in row else None

    labels = []
    confs = []
    errtypes = []
    reasons = []
    is_labeled = []

    for _, row in df.iterrows():
        key = build_key(row["row_id"], row["column"])
        if key in label_map:
            labels.append(label_map[key])
            confs.append(conf_map[key])
            errtypes.append(errtype_map[key])
            reasons.append(reason_map[key])
            is_labeled.append(1)
        else:
            labels.append(np.nan)
            confs.append(np.nan)
            errtypes.append(None)
            reasons.append(None)
            is_labeled.append(0)

    df["label_binary"] = labels
    df["label_confidence"] = confs
    df["error_type"] = errtypes
    df["reason_short"] = reasons
    df["is_labeled"] = is_labeled

    # 初始化传播结果列
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

    # 数值特征
    for col in FEATURE_COLUMNS_NUMERIC:
        if col not in work.columns:
            work[col] = 0.0
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)

    num_df = work[FEATURE_COLUMNS_NUMERIC].copy()

    # 类别特征
    cat_frames = []
    feat_names = FEATURE_COLUMNS_NUMERIC.copy()

    for col in FEATURE_COLUMNS_CATEGORICAL:
        if col not in work.columns:
            work[col] = "unknown"
        dummies = pd.get_dummies(work[col].fillna("unknown"), prefix=col)
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

    for (bucket_id, cluster_id), grp in df.groupby(["bucket_id", "cluster_id"], sort=False):
        grp_idx = grp.index.tolist()

        # 这里用 final_label_binary 作为簇内已知标签来源，允许 LLM 已标注样本做簇内传播
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

        if propagated_conf < MIN_PROPAGATION_CONFIDENCE:
            continue

        # 传播给当前 cluster 内未标注样本
        unlabeled_mask = (df.index.isin(grp_idx)) & (df["final_label_binary"].isna())
        target_indices = df[unlabeled_mask].index.tolist()

        for idx in target_indices:
            df.at[idx, "final_label_binary"] = propagated_label
            df.at[idx, "final_label"] = label_to_name(propagated_label)
            df.at[idx, "propagation_confidence"] = round(float(propagated_conf), 6)
            df.at[idx, "label_source"] = "cluster_propagation"
            df.at[idx, "sample_weight"] = compute_weight(WEIGHT_CLUSTER_PROP, propagated_conf)

    return df


# ============================================================
# 8. 第二层传播：KNN传播
# ============================================================

def knn_label_propagation(df: pd.DataFrame, X: np.ndarray) -> pd.DataFrame:
    df = df.copy()

    # 关键修正1：
    # KNN 教师集只使用 LLM 原始标签，防止传播标签继续扩散放大误差
    teacher_mask = (df["label_source"] == "llm")
    teacher_indices = df[teacher_mask].index.tolist()

    if len(teacher_indices) < MIN_TEACHER_SAMPLES_FOR_KNN:
        return df

    # 按列传播，更稳
    if KNN_WITHIN_SAME_COLUMN_ONLY:
        all_columns = df["column"].dropna().astype(str).unique().tolist()

        for col_name in all_columns:
            col_teacher_indices = df[(df["label_source"] == "llm") & (df["column"] == col_name)].index.tolist()
            col_unlabeled_indices = df[(df["final_label_binary"].isna()) & (df["column"] == col_name)].index.tolist()

            if len(col_teacher_indices) < max(2, min(KNN_K, MIN_TEACHER_SAMPLES_FOR_KNN // 4)):
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
                distances, neighbor_pos = nn_model.kneighbors(x)
                neighbor_pos = neighbor_pos[0]
                distances = distances[0]

                neighbor_labels = y_teacher[neighbor_pos]

                error_ratio = float(np.mean(neighbor_labels == 1))
                correct_ratio = float(np.mean(neighbor_labels == 0))

                propagated_label = None
                propagated_conf = None

                if error_ratio >= KNN_PROPAGATION_THRESHOLD:
                    propagated_label = 1
                    propagated_conf = error_ratio
                elif correct_ratio >= KNN_PROPAGATION_THRESHOLD:
                    propagated_label = 0
                    propagated_conf = correct_ratio

                if propagated_label is None:
                    continue

                if propagated_conf < MIN_PROPAGATION_CONFIDENCE:
                    continue

                df.at[idx, "final_label_binary"] = propagated_label
                df.at[idx, "final_label"] = label_to_name(propagated_label)
                df.at[idx, "propagation_confidence"] = round(float(propagated_conf), 6)
                df.at[idx, "label_source"] = "knn_propagation"
                df.at[idx, "sample_weight"] = compute_weight(WEIGHT_KNN_PROP, propagated_conf)

    else:
        # 跨列传播版本（不推荐，但保留）
        X_teacher = X[teacher_indices]
        y_teacher = df.loc[teacher_indices, "final_label_binary"].astype(int).values

        nn_model = NearestNeighbors(n_neighbors=min(KNN_K, len(teacher_indices)), metric="euclidean")
        nn_model.fit(X_teacher)

        unlabeled_indices = df[df["final_label_binary"].isna()].index.tolist()

        for idx in unlabeled_indices:
            x = X[idx].reshape(1, -1)
            distances, neighbor_pos = nn_model.kneighbors(x)
            neighbor_pos = neighbor_pos[0]
            distances = distances[0]

            neighbor_labels = y_teacher[neighbor_pos]

            error_ratio = float(np.mean(neighbor_labels == 1))
            correct_ratio = float(np.mean(neighbor_labels == 0))

            propagated_label = None
            propagated_conf = None

            if error_ratio >= KNN_PROPAGATION_THRESHOLD:
                propagated_label = 1
                propagated_conf = error_ratio
            elif correct_ratio >= KNN_PROPAGATION_THRESHOLD:
                propagated_label = 0
                propagated_conf = correct_ratio

            if propagated_label is None:
                continue

            if propagated_conf < MIN_PROPAGATION_CONFIDENCE:
                continue

            df.at[idx, "final_label_binary"] = propagated_label
            df.at[idx, "final_label"] = label_to_name(propagated_label)
            df.at[idx, "propagation_confidence"] = round(float(propagated_conf), 6)
            df.at[idx, "label_source"] = "knn_propagation"
            df.at[idx, "sample_weight"] = compute_weight(WEIGHT_KNN_PROP, propagated_conf)

    return df


# ============================================================
# 9. 导出结果
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
                "dist_to_center": float(row["dist_to_center"]) if pd.notna(row["dist_to_center"]) else None,

                # 最终训练标签
                "label": row["final_label"],
                "label_binary": int(row["final_label_binary"]),

                # 标签质量与来源
                "propagation_confidence": float(row["propagation_confidence"]) if pd.notna(row["propagation_confidence"]) else None,
                "label_source": row.get("label_source"),
                "sample_weight": float(row["sample_weight"]) if pd.notna(row["sample_weight"]) else None,

                # 结构化特征
                "semantic_type": row.get("semantic_type"),
                "main_rule_type": row.get("main_rule_type"),
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
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            out_rows.append(obj)

    pd.DataFrame(out_rows).to_csv(output_csv, index=False, encoding="utf-8-sig")


# ============================================================
# 10. 主流程
# ============================================================

def main():
    clustered_df = load_clustered_csv(INPUT_CLUSTERED_CSV)
    labeled_df = load_labeled_csv(INPUT_LABELED_CSV)

    df = merge_clustered_and_labeled(clustered_df, labeled_df)

    # 准备特征
    X, feat_names = prepare_features(df)

    # 第一层：簇内传播
    df = cluster_label_propagation(df)

    # 第二层：KNN传播
    if ENABLE_KNN_PROPAGATION:
        df = knn_label_propagation(df, X)

    # 导出
    export_results(df, OUTPUT_JSONL, OUTPUT_CSV)

    # 统计
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
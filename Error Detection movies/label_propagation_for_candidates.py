import json
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. 用户配置区
# ============================================================

INPUT_CLUSTERED_CSV = "candidate_clustered_movies.csv"
INPUT_LABELED_CSV = "candidate_sampled_labeled_movies.csv"

OUTPUT_JSONL = "propagated_labels_movies.jsonl"
OUTPUT_CSV = "propagated_labels_movies.csv"

# ------------------------------------------------------------
# 传播总原则：
#   1) LLM 原始标签全部保留。
#   2) 只对“高度一致、距离较近、同列/同桶”的样本做少量传播。
#   3) 不追求把所有候选都打满标签，避免伪标签过多污染训练集。
# ------------------------------------------------------------

# 簇内传播：更保守
MIN_LABELED_PER_CLUSTER = 3
CLUSTER_PROPAGATION_THRESHOLD = 0.90

ENABLE_CLUSTER_CENTER_FILTER = True
CLUSTER_CENTER_DIST_QUANTILE = 0.50
MAX_CLUSTER_PROPAGATION_PER_CLUSTER = 8

# KNN 传播：更保守
ENABLE_KNN_PROPAGATION = True
KNN_K = 5
KNN_PROPAGATION_THRESHOLD = 0.88
MIN_TEACHER_SAMPLES_FOR_KNN = 30
KNN_WITHIN_SAME_COLUMN_ONLY = True

KNN_MIN_MARGIN = 0.18
KNN_DISTANCE_EPS = 1e-8
KNN_MAX_AVG_DISTANCE = 3.2

# 传播预算：防止传播太多
ENABLE_PROPAGATION_BUDGET = True

# 传播出来的样本总数上限，不包含 LLM 原始标注。
# 如果你希望更省，可以改成 300；如果希望更多覆盖，可以改成 800。
MAX_TOTAL_PROPAGATED = 500

# 每列最多传播多少个，不包含 LLM 原始标注。
MAX_PROPAGATED_PER_COLUMN = {
    "Release Date": 80,
    "Year": 50,
    "Duration": 50,
    "RatingValue": 50,
    "RatingCount": 50,
    "ReviewCount": 50,
    "Language": 50,
    "Country": 50,
    "Genre": 50,
    "Director": 60,
    "Creator": 60,
    "Actors": 60,
    "Cast": 60,
    "Name": 50,
    "Filming Locations": 40,
    "Description": 40,
}

# 每个 bucket 最多传播多少个，不包含 LLM 原始标注。
MAX_PROPAGATED_PER_BUCKET = 20

# 权重
WEIGHT_LLM = 1.0
WEIGHT_CLUSTER_PROP = 0.65
WEIGHT_KNN_PROP = 0.45

MIN_PROPAGATION_CONFIDENCE = 0.72

# 是否过滤 LLM 低置信度样本作为 teacher。
# 注意：低置信度 LLM 样本仍然会保留进训练集，只是不参与传播。
TEACHER_MIN_CONFIDENCE_FOR_PROPAGATION = 0.75
EXCLUDE_HUMAN_REVIEW_FROM_TEACHER = True

RANDOM_STATE = 42


# ============================================================
# 2. 特征列配置：movies 版
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
    "dist_to_center",
    "sampling_priority_score",
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


def parse_bool_like(x: Any) -> Optional[bool]:
    if pd.isna(x) or x is None:
        return None
    s = str(x).strip().lower()
    if s in {"true", "1", "yes"}:
        return True
    if s in {"false", "0", "no"}:
        return False
    return None


def label_to_name(y: int) -> str:
    return "error" if int(y) == 1 else "correct"


def build_key(row_id: Any, column: Any) -> Tuple[int, str]:
    return int(row_id), str(column)


def compute_weight(base_weight: float, conf: float) -> float:
    conf = min(max(float(conf), 0.0), 1.0)
    return round(base_weight * conf, 6)


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


def safe_numeric(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.full(len(df), default), index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def safe_str(df: pd.DataFrame, col: str, default: str = "unknown") -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index)
    return df[col].fillna(default).astype(str)


# ============================================================
# 4. 读取数据
# ============================================================

def load_clustered_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["row_id"] = df["row_id"].astype(int)
    df["column"] = df["column"].astype(str)

    expected_defaults = {
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
        "context_rule_count": 0.0,
        "global_rule_count": 0.0,
        "rare_value_count": 0.0,
        "typo_rule_count": 0.0,
        "pattern_rule_count": 0.0,
        "schema_rule_count": 0.0,
        "movie_typed_rule_count": 0.0,
        "movie_consistency_rule_count": 0.0,
        "parsed_year": 0.0,
        "parsed_release_year": 0.0,
        "parsed_release_country": "unknown",
        "duration_minutes": 0.0,
        "rating_value_float": 0.0,
        "count_value_int": 0.0,
        "review_total": 0.0,
        "list_token_count": 0.0,
        "list_token_count_bucket": "unknown",
        "text_length_bucket": "unknown",
        "has_mojibake_or_control_chars": 0.0,
        "movie_row_year": 0.0,
        "movie_row_release_year": 0.0,
        "movie_row_release_country": "unknown",
        "movie_row_release_year_gap": 0.0,
        "movie_row_duration_minutes": 0.0,
        "movie_row_rating_value_float": 0.0,
        "movie_row_rating_count_int": 0.0,
        "movie_row_review_total": 0.0,
        "movie_row_review_to_rating_ratio": 0.0,
        "movie_row_actors_cast_overlap": 0.0,
        "pattern_bucket": "unknown",
        "rarity_bucket": "unknown",
        "neighbor_bucket": "unknown",
        "movie_rule_bucket": "unknown",
        "movie_field_bucket": "unknown",
        "year_bucket": "unknown",
        "release_year_gap_bucket": "unknown",
        "duration_bucket": "unknown",
        "rating_bucket": "unknown",
        "count_bucket": "unknown",
        "review_ratio_bucket": "unknown",
        "actors_cast_overlap_bucket": "unknown",
        "list_bucket": "unknown",
        "text_bucket": "unknown",
        "bucket_id": "unknown",
        "cluster_id": -1,
        "dist_to_center": 0.0,
        "sample_role": "",
        "is_sampled": 0,
        "sampling_priority_score": 0.0,
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

    if "needs_human_review" not in df.columns:
        df["needs_human_review"] = False
    df["needs_human_review_bool"] = df["needs_human_review"].map(parse_bool_like).fillna(False)

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
            "needs_human_review_bool": bool(row["needs_human_review_bool"]) if "needs_human_review_bool" in row else False,
            "evidence_used": row["evidence_used"] if "evidence_used" in row else None,
        }

    labels = []
    confs = []
    errtypes = []
    reasons_short = []
    reasons_detailed = []
    suggested_values = []
    needs_review = []
    needs_review_bool = []
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
            needs_review_bool.append(item["needs_human_review_bool"])
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
            needs_review_bool.append(False)
            evidence_useds.append(None)
            is_labeled.append(0)

    df["label_binary"] = labels
    df["label_confidence"] = confs
    df["error_type"] = errtypes
    df["reason_short"] = reasons_short
    df["reason_detailed"] = reasons_detailed
    df["suggested_correct_value"] = suggested_values
    df["needs_human_review"] = needs_review
    df["needs_human_review_bool"] = needs_review_bool
    df["evidence_used"] = evidence_useds
    df["is_labeled"] = is_labeled

    df["final_label_binary"] = df["label_binary"]
    df["final_label"] = df["label_binary"].map(lambda x: label_to_name(x) if pd.notna(x) else None)
    df["propagation_confidence"] = df["label_confidence"]
    df["label_source"] = df["is_labeled"].map(lambda x: "llm" if x == 1 else None)
    df["sample_weight"] = df["is_labeled"].map(lambda x: WEIGHT_LLM if x == 1 else np.nan)

    df["can_be_teacher"] = (
        (df["label_source"] == "llm") &
        (pd.to_numeric(df["label_confidence"], errors="coerce").fillna(0.0) >= TEACHER_MIN_CONFIDENCE_FOR_PROPAGATION)
    ).astype(int)

    if EXCLUDE_HUMAN_REVIEW_FROM_TEACHER:
        df.loc[df["needs_human_review_bool"] == True, "can_be_teacher"] = 0

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
# 7. 传播候选打分与预算控制
# ============================================================

def propagation_priority_score(row: pd.Series) -> float:
    conf = pd.to_numeric(row.get("propagation_confidence", 0.0), errors="coerce")
    conflict_score = pd.to_numeric(row.get("conflict_score", 0.0), errors="coerce")
    posterior = pd.to_numeric(row.get("posterior_error_probability", 0.0), errors="coerce")
    violation_count = pd.to_numeric(row.get("violation_count", 0.0), errors="coerce")
    movie_typed = pd.to_numeric(row.get("movie_typed_rule_count", 0.0), errors="coerce")
    movie_cons = pd.to_numeric(row.get("movie_consistency_rule_count", 0.0), errors="coerce")
    dist = pd.to_numeric(row.get("dist_to_center", 0.0), errors="coerce")

    conf = 0.0 if pd.isna(conf) else float(conf)
    conflict_score = 0.0 if pd.isna(conflict_score) else float(conflict_score)
    posterior = 0.0 if pd.isna(posterior) else float(posterior)
    violation_count = 0.0 if pd.isna(violation_count) else float(violation_count)
    movie_typed = 0.0 if pd.isna(movie_typed) else float(movie_typed)
    movie_cons = 0.0 if pd.isna(movie_cons) else float(movie_cons)
    dist = 0.0 if pd.isna(dist) else float(dist)

    # 距离越小越优先
    dist_bonus = 1.0 / (1.0 + max(dist, 0.0))

    return (
        3.0 * conf
        + 1.2 * posterior
        + 0.01 * conflict_score
        + 0.2 * violation_count
        + 0.4 * movie_typed
        + 0.6 * movie_cons
        + 0.5 * dist_bonus
    )


def apply_propagation_budget(df: pd.DataFrame) -> pd.DataFrame:
    if not ENABLE_PROPAGATION_BUDGET:
        return df

    out = df.copy()

    llm_mask = out["label_source"].eq("llm")
    prop_mask = out["label_source"].isin(["cluster_propagation", "knn_propagation"])

    llm_df = out[llm_mask].copy()
    unlabeled_df = out[out["final_label_binary"].isna()].copy()
    prop_df = out[prop_mask].copy()

    if len(prop_df) == 0:
        return out

    prop_df["propagation_priority_score"] = prop_df.apply(propagation_priority_score, axis=1)

    # 1) 每 bucket 限制
    bucket_kept = []
    for bucket_id, g in prop_df.groupby("bucket_id", sort=False):
        g = g.sort_values(
            ["propagation_priority_score", "propagation_confidence", "conflict_score"],
            ascending=[False, False, False]
        ).head(MAX_PROPAGATED_PER_BUCKET)
        bucket_kept.append(g)
    prop_df = pd.concat(bucket_kept, ignore_index=False) if bucket_kept else prop_df.iloc[0:0].copy()

    # 2) 每列限制
    column_kept = []
    for col, g in prop_df.groupby("column", sort=False):
        cap = int(MAX_PROPAGATED_PER_COLUMN.get(str(col), max(20, MAX_TOTAL_PROPAGATED // 12)))
        g = g.sort_values(
            ["propagation_priority_score", "propagation_confidence", "conflict_score"],
            ascending=[False, False, False]
        ).head(cap)
        column_kept.append(g)
    prop_df = pd.concat(column_kept, ignore_index=False) if column_kept else prop_df.iloc[0:0].copy()

    # 3) 全局限制
    if len(prop_df) > MAX_TOTAL_PROPAGATED:
        prop_df = prop_df.sort_values(
            ["propagation_priority_score", "propagation_confidence", "conflict_score"],
            ascending=[False, False, False]
        ).head(MAX_TOTAL_PROPAGATED)

    kept_prop_indices = set(prop_df.index.tolist())

    # 对被预算裁掉的传播样本，恢复成未标注
    drop_prop_indices = out[prop_mask & (~out.index.isin(kept_prop_indices))].index.tolist()
    for idx in drop_prop_indices:
        out.at[idx, "final_label_binary"] = np.nan
        out.at[idx, "final_label"] = None
        out.at[idx, "propagation_confidence"] = np.nan
        out.at[idx, "label_source"] = None
        out.at[idx, "sample_weight"] = np.nan
        safe_none_fill(out, idx)

    print("\n===== 传播预算控制 =====")
    print(f"预算前传播数: {int(prop_mask.sum())}")
    print(f"预算后传播数: {len(kept_prop_indices)}")
    print(f"MAX_TOTAL_PROPAGATED: {MAX_TOTAL_PROPAGATED}")

    return out


# ============================================================
# 8. 第一层传播：保守簇内传播
# ============================================================

def cluster_label_propagation(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for (bucket_id, cluster_id), grp in df.groupby(["bucket_id", "cluster_id"], sort=False):
        labeled_grp = grp[(grp["can_be_teacher"] == 1)]

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

        unlabeled_grp = unlabeled_grp.sort_values(
            by=["dist_to_center", "conflict_score"],
            ascending=[True, False],
            na_position="last"
        ).head(MAX_CLUSTER_PROPAGATION_PER_CLUSTER)

        for idx in unlabeled_grp.index.tolist():
            df.at[idx, "final_label_binary"] = propagated_label
            df.at[idx, "final_label"] = label_to_name(propagated_label)
            df.at[idx, "propagation_confidence"] = round(float(propagated_conf), 6)
            df.at[idx, "label_source"] = "cluster_propagation"
            df.at[idx, "sample_weight"] = compute_weight(WEIGHT_CLUSTER_PROP, propagated_conf)
            safe_none_fill(df, idx)

    return df


# ============================================================
# 9. 第二层传播：距离加权 KNN 传播
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

    teacher_mask = (df["can_be_teacher"] == 1)
    teacher_indices = df[teacher_mask].index.tolist()

    if len(teacher_indices) < MIN_TEACHER_SAMPLES_FOR_KNN:
        return df

    if KNN_WITHIN_SAME_COLUMN_ONLY:
        all_columns = df["column"].dropna().astype(str).unique().tolist()

        for col_name in all_columns:
            col_teacher_indices = df[
                (df["can_be_teacher"] == 1) & (df["column"] == col_name)
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
# 10. 导出结果
# ============================================================

def to_json_safe(x):
    if pd.isna(x):
        return None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return x


def export_results(df: pd.DataFrame, output_jsonl: str, output_csv: str):
    out_rows = []

    # 稳定排序
    sort_cols = [c for c in ["row_id", "column", "label_source"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).copy()

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            if pd.isna(row["final_label_binary"]):
                continue

            obj = {
                "row_id": int(row["row_id"]),
                "column": str(row["column"]),
                "value": to_json_safe(row.get("value")),
                "bucket_id": to_json_safe(row.get("bucket_id")),
                "cluster_id": int(row["cluster_id"]) if pd.notna(row["cluster_id"]) else None,
                "sample_role": to_json_safe(row.get("sample_role")),
                "is_sampled": int(row["is_sampled"]) if pd.notna(row["is_sampled"]) else None,
                "dist_to_center": float(row["dist_to_center"]) if pd.notna(row["dist_to_center"]) else None,

                "label": row["final_label"],
                "label_binary": int(row["final_label_binary"]),

                "propagation_confidence": float(row["propagation_confidence"]) if pd.notna(row["propagation_confidence"]) else None,
                "label_source": to_json_safe(row.get("label_source")),
                "sample_weight": float(row["sample_weight"]) if pd.notna(row["sample_weight"]) else None,

                "semantic_type": to_json_safe(row.get("semantic_type")),
                "detected_type": to_json_safe(row.get("detected_type")),
                "main_rule_type": to_json_safe(row.get("main_rule_type")),
                "main_usage_role": to_json_safe(row.get("main_usage_role")),
                "pattern_bucket": to_json_safe(row.get("pattern_bucket")),
                "rarity_bucket": to_json_safe(row.get("rarity_bucket")),
                "neighbor_bucket": to_json_safe(row.get("neighbor_bucket")),
                "movie_rule_bucket": to_json_safe(row.get("movie_rule_bucket")),
                "movie_field_bucket": to_json_safe(row.get("movie_field_bucket")),
                "year_bucket": to_json_safe(row.get("year_bucket")),
                "release_year_gap_bucket": to_json_safe(row.get("release_year_gap_bucket")),
                "duration_bucket": to_json_safe(row.get("duration_bucket")),
                "rating_bucket": to_json_safe(row.get("rating_bucket")),
                "count_bucket": to_json_safe(row.get("count_bucket")),
                "review_ratio_bucket": to_json_safe(row.get("review_ratio_bucket")),
                "actors_cast_overlap_bucket": to_json_safe(row.get("actors_cast_overlap_bucket")),
                "list_bucket": to_json_safe(row.get("list_bucket")),
                "text_bucket": to_json_safe(row.get("text_bucket")),

                "violation_count": float(row["violation_count"]) if pd.notna(row["violation_count"]) else None,
                "conflict_score": float(row["conflict_score"]) if pd.notna(row["conflict_score"]) else None,
                "value_frequency": float(row["value_frequency"]) if pd.notna(row["value_frequency"]) else None,
                "value_frequency_rank": float(row["value_frequency_rank"]) if pd.notna(row["value_frequency_rank"]) else None,
                "neighbor_majority_value": to_json_safe(row.get("neighbor_majority_value")),
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
                "movie_typed_rule_count": float(row["movie_typed_rule_count"]) if pd.notna(row["movie_typed_rule_count"]) else None,
                "movie_consistency_rule_count": float(row["movie_consistency_rule_count"]) if pd.notna(row["movie_consistency_rule_count"]) else None,

                "parsed_year": to_json_safe(row.get("parsed_year")),
                "parsed_release_year": to_json_safe(row.get("parsed_release_year")),
                "parsed_release_country": to_json_safe(row.get("parsed_release_country")),
                "duration_minutes": to_json_safe(row.get("duration_minutes")),
                "rating_value_float": to_json_safe(row.get("rating_value_float")),
                "count_value_int": to_json_safe(row.get("count_value_int")),
                "review_total": to_json_safe(row.get("review_total")),
                "list_token_count": to_json_safe(row.get("list_token_count")),
                "has_mojibake_or_control_chars": to_json_safe(row.get("has_mojibake_or_control_chars")),
                "movie_row_release_year_gap": to_json_safe(row.get("movie_row_release_year_gap")),
                "movie_row_review_to_rating_ratio": to_json_safe(row.get("movie_row_review_to_rating_ratio")),
                "movie_row_actors_cast_overlap": to_json_safe(row.get("movie_row_actors_cast_overlap")),

                "error_type": to_json_safe(row.get("error_type")),
                "reason_short": to_json_safe(row.get("reason_short")),
                "reason_detailed": to_json_safe(row.get("reason_detailed")),
                "suggested_correct_value": to_json_safe(row.get("suggested_correct_value")),
                "needs_human_review": to_json_safe(row.get("needs_human_review")),
                "evidence_used": to_json_safe(row.get("evidence_used")),
            }

            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            out_rows.append(obj)

    pd.DataFrame(out_rows).to_csv(output_csv, index=False, encoding="utf-8-sig")


# ============================================================
# 11. 主流程
# ============================================================

def main():
    clustered_df = load_clustered_csv(INPUT_CLUSTERED_CSV)
    labeled_df = load_labeled_csv(INPUT_LABELED_CSV)

    df = merge_clustered_and_labeled(clustered_df, labeled_df)

    X, _ = prepare_features(df)

    before_labeled = int(pd.notna(df["final_label_binary"]).sum())
    before_llm = int((df["label_source"] == "llm").sum())

    df = cluster_label_propagation(df)

    if ENABLE_KNN_PROPAGATION:
        df = knn_label_propagation(df, X)

    df = apply_propagation_budget(df)

    export_results(df, OUTPUT_JSONL, OUTPUT_CSV)

    total = len(df)
    llm_count = int((df["label_source"] == "llm").sum())
    cluster_count = int((df["label_source"] == "cluster_propagation").sum())
    knn_count = int((df["label_source"] == "knn_propagation").sum())
    propagated_count = cluster_count + knn_count
    total_labeled = int(pd.notna(df["final_label_binary"]).sum())

    print(f"[OK] 传播结果 JSONL 已保存: {OUTPUT_JSONL}")
    print(f"[OK] 传播结果 CSV 已保存: {OUTPUT_CSV}")

    print("\n===== 统计信息 =====")
    print(f"总候选样本数: {total}")
    print(f"LLM原始标注数: {llm_count}")
    print(f"簇内传播数: {cluster_count}")
    print(f"KNN传播数: {knn_count}")
    print(f"传播总数: {propagated_count}")
    print(f"最终可训练样本数: {total_labeled}")

    if total > 0:
        print(f"训练集覆盖率: {total_labeled / total:.6f}")

    if before_llm > 0:
        print(f"传播/LLM比例: {propagated_count / before_llm:.6f}")

    print("\n标签来源分布:")
    print(df["label_source"].value_counts(dropna=False).to_string())

    print("\n最终标签分布:")
    print(df["final_label"].value_counts(dropna=False).to_string())

    print("\n按列传播分布:")
    prop_df = df[df["label_source"].isin(["cluster_propagation", "knn_propagation"])]
    if len(prop_df) > 0:
        print(prop_df["column"].value_counts(dropna=False).to_string())
    else:
        print("无传播样本")


if __name__ == "__main__":
    main()

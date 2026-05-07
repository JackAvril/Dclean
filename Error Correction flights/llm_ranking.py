import os
import json
import time
import math
from collections import Counter, defaultdict
from itertools import combinations
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from openai import OpenAI


# ============================================================
# 1. 用户配置区
# ============================================================

# 输入1：候选特征文件
# 推荐使用白名单过滤之后的特征文件，避免把检测阶段未判错的 cell 送入 repair 标注。
INPUT_CANDIDATE_FEATURES_CSV = "/mnt/mydata/dq/projects/Splittree/test/Error Correction flights/repair_candidate_features_infer_whitelist.csv"


# 输入2：详细上下文 rich jsonl
INPUT_DETAIL_JSONL = "/mnt/mydata/dq/projects/Splittree/test/Error Detection flights/candidate_llm_contexts_flights.jsonl"

# 输出：如果是相对路径，会自动输出到 INPUT_CANDIDATE_FEATURES_CSV 所在目录
OUTPUT_CELL_SUMMARY_CSV = "repair_cell_summary_for_sampling_v2.csv"
OUTPUT_CLUSTERED_CSV = "repair_cell_clustered_v2.csv"
OUTPUT_SAMPLED_CSV = "repair_cell_sampled_v2.csv"
OUTPUT_SAMPLED_JSONL = "repair_cell_sampled_with_candidates_v2.jsonl"

OUTPUT_PAIRWISE_REQUESTS_JSONL = "repair_pairwise_requests_v2.jsonl"
OUTPUT_PAIRWISE_RESPONSES_JSONL = "repair_pairwise_responses_v2.jsonl"
OUTPUT_TEACHER_RANKING_CSV = "repair_teacher_pairwise_ranking_v2.csv"
OUTPUT_TEACHER_TOP1_CSV = "repair_teacher_top1_labels_v2.csv"
OUTPUT_BUCKET_SUMMARY_CSV = "repair_bucket_cluster_summary_v2.csv"

# 是否把 column 纳入分桶
# Flights 建议 True，因为 sched_dep_time / act_dep_time / sched_arr_time / act_arr_time 的修复语义不同。
USE_COLUMN_IN_BUCKET = True

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

# 候选控制
DEFAULT_MAX_CANDIDATES_PER_CELL_FOR_LLM = 6
HARD_COLUMN_MAX_CANDIDATES = 8

# Hospital hard columns + Flights time columns
HOSPITAL_HARD_COLUMNS = {"MeasureCode", "Stateavg", "Condition", "MeasureName", "Score", "Sample"}
TIME_COLUMNS = {"sched_dep_time", "act_dep_time", "sched_arr_time", "act_arr_time"}

HARD_COLUMNS = HOSPITAL_HARD_COLUMNS | TIME_COLUMNS

DROP_UNKNOWN_IF_TOO_MANY = False

# Pairwise 策略
PAIRWISE_STRATEGY = "one_vs_top"   # 可选: all_pairs / one_vs_top
UNKNOWN_CANDIDATE_NAME = "Unknown"

# LLM 配置（OpenAI 兼容接口，可接 DeepSeek）
MODEL_NAME = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-fd93f5216ae0488f8bf32d9e2a14a14a")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
TEMPERATURE = 0.0
MAX_RETRIES = 3
RETRY_SLEEP_SECONDS = 2.0

# 若只想先导出采样与 pairwise 请求，不实际调用 LLM，可设为 True
DRY_RUN_ONLY_EXPORT_REQUESTS = False


# ============================================================
# 2. 基础工具
# ============================================================

def write_jsonl_record(fp, obj):
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")


def resolve_output_path(path: str, anchor_file: str) -> str:
    if os.path.isabs(str(path)):
        return str(path)
    base_dir = os.path.dirname(os.path.abspath(anchor_file))
    return os.path.join(base_dir, str(path))


def ensure_output_paths():
    global OUTPUT_CELL_SUMMARY_CSV
    global OUTPUT_CLUSTERED_CSV
    global OUTPUT_SAMPLED_CSV
    global OUTPUT_SAMPLED_JSONL
    global OUTPUT_PAIRWISE_REQUESTS_JSONL
    global OUTPUT_PAIRWISE_RESPONSES_JSONL
    global OUTPUT_TEACHER_RANKING_CSV
    global OUTPUT_TEACHER_TOP1_CSV
    global OUTPUT_BUCKET_SUMMARY_CSV

    OUTPUT_CELL_SUMMARY_CSV = resolve_output_path(OUTPUT_CELL_SUMMARY_CSV, INPUT_CANDIDATE_FEATURES_CSV)
    OUTPUT_CLUSTERED_CSV = resolve_output_path(OUTPUT_CLUSTERED_CSV, INPUT_CANDIDATE_FEATURES_CSV)
    OUTPUT_SAMPLED_CSV = resolve_output_path(OUTPUT_SAMPLED_CSV, INPUT_CANDIDATE_FEATURES_CSV)
    OUTPUT_SAMPLED_JSONL = resolve_output_path(OUTPUT_SAMPLED_JSONL, INPUT_CANDIDATE_FEATURES_CSV)
    OUTPUT_PAIRWISE_REQUESTS_JSONL = resolve_output_path(OUTPUT_PAIRWISE_REQUESTS_JSONL, INPUT_CANDIDATE_FEATURES_CSV)
    OUTPUT_PAIRWISE_RESPONSES_JSONL = resolve_output_path(OUTPUT_PAIRWISE_RESPONSES_JSONL, INPUT_CANDIDATE_FEATURES_CSV)
    OUTPUT_TEACHER_RANKING_CSV = resolve_output_path(OUTPUT_TEACHER_RANKING_CSV, INPUT_CANDIDATE_FEATURES_CSV)
    OUTPUT_TEACHER_TOP1_CSV = resolve_output_path(OUTPUT_TEACHER_TOP1_CSV, INPUT_CANDIDATE_FEATURES_CSV)
    OUTPUT_BUCKET_SUMMARY_CSV = resolve_output_path(OUTPUT_BUCKET_SUMMARY_CSV, INPUT_CANDIDATE_FEATURES_CSV)

    for p in [
        OUTPUT_CELL_SUMMARY_CSV,
        OUTPUT_CLUSTERED_CSV,
        OUTPUT_SAMPLED_CSV,
        OUTPUT_SAMPLED_JSONL,
        OUTPUT_PAIRWISE_REQUESTS_JSONL,
        OUTPUT_PAIRWISE_RESPONSES_JSONL,
        OUTPUT_TEACHER_RANKING_CSV,
        OUTPUT_TEACHER_TOP1_CSV,
        OUTPUT_BUCKET_SUMMARY_CSV,
    ]:
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)


def norm_text(x):
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip().lower()


def canonical_text(x):
    s = norm_text(x)
    if s in {"", "nan", "none", "null", "na", "n/a", "missing", "empty"}:
        return ""
    return str(x).strip()


def safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        if isinstance(x, float) and math.isnan(x):
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x, default=0):
    try:
        if x is None:
            return default
        if isinstance(x, float) and math.isnan(x):
            return default
        return int(float(x))
    except Exception:
        return default


def looks_like_time_column(column: str, semantic_type: Optional[str] = None) -> bool:
    col = norm_text(column)
    sem = norm_text(semantic_type)
    return sem == "time_like" or col in TIME_COLUMNS or col.endswith("_time")


# ============================================================
# 3. 读取详细 jsonl
# ============================================================

def load_detail_jsonl(path: Optional[str]) -> Dict[Tuple[int, str], Dict[str, Any]]:
    detail_map = {}
    if path is None or path == "" or not os.path.exists(path):
        return detail_map

    with open(path, "r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                key = (int(obj["row_id"]), str(obj["column"]))
                detail_map[key] = obj
            except Exception as e:
                print(f"[WARN] 跳过无法解析的 JSONL 行: line={line_no}, err={e}")
                continue
    return detail_map


# ============================================================
# 4. 构建 cell-level 摘要（从 candidate-level 特征聚合）
# ============================================================

def first_non_null(series: pd.Series, default=None):
    for x in series:
        if pd.notna(x):
            return x
    return default


def get_numeric_series(g: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in g.columns:
        return pd.to_numeric(g[col], errors="coerce").fillna(default)
    return pd.Series([default] * len(g), index=g.index)


def build_cell_summary_df(cand_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    grouped = cand_df.groupby(["row_id", "column"], sort=False)

    for (row_id, column), g in grouped:
        g = g.copy()

        if "final_score_from_candidate_gen" in g.columns:
            score_col = "final_score_from_candidate_gen"
        elif "final_score" in g.columns:
            score_col = "final_score"
        else:
            score_col = None

        if score_col:
            if "candidate_rank" in g.columns:
                g = g.sort_values([score_col, "candidate_rank"], ascending=[False, True])
            else:
                g = g.sort_values([score_col], ascending=[False])
        elif "candidate_rank" in g.columns:
            g = g.sort_values(["candidate_rank"], ascending=[True])

        dirty_value = first_non_null(g["dirty_value"], "") if "dirty_value" in g.columns else ""
        semantic_type = first_non_null(g["semantic_type"], "unknown") if "semantic_type" in g.columns else "unknown"
        main_rule_type = first_non_null(g["main_rule_type"], "unknown") if "main_rule_type" in g.columns else "unknown"
        main_usage_role = first_non_null(g["main_usage_role"], "unknown") if "main_usage_role" in g.columns else "unknown"

        violation_count = pd.to_numeric(first_non_null(g["violation_count"], 0), errors="coerce")
        conflict_score = pd.to_numeric(first_non_null(g["conflict_score"], 0), errors="coerce")

        candidate_count = len(g)
        top1_score = float(g[score_col].iloc[0]) if score_col and len(g) >= 1 else 0.0
        top2_score = float(g[score_col].iloc[1]) if score_col and len(g) >= 2 else 0.0
        top_margin = top1_score - top2_score if candidate_count >= 2 else top1_score

        rule_support_s = get_numeric_series(g, "candidate_rule_support_ratio", 0.0)
        bayes_s = get_numeric_series(g, "bayes_mean_cond_prob", 0.0)
        edit_sim_s = get_numeric_series(g, "edit_similarity_to_dirty", 0.0)

        source_count_top1 = safe_int(g["source_count"].iloc[0], 0) if "source_count" in g.columns and len(g) >= 1 else 0

        top1_is_from_rule = safe_int(g["is_from_rule"].iloc[0], 0) if "is_from_rule" in g.columns and len(g) >= 1 else 0
        top1_is_from_neighbor = safe_int(g["is_from_neighbor"].iloc[0], 0) if "is_from_neighbor" in g.columns and len(g) >= 1 else 0
        top1_is_from_similarity = safe_int(g["is_from_similarity"].iloc[0], 0) if "is_from_similarity" in g.columns and len(g) >= 1 else 0
        top1_is_from_hint = safe_int(g["is_from_hint"].iloc[0], 0) if "is_from_hint" in g.columns and len(g) >= 1 else 0
        top1_is_from_dictionary = safe_int(g["is_from_dictionary"].iloc[0], 0) if "is_from_dictionary" in g.columns and len(g) >= 1 else 0
        top1_is_from_pattern_restore = safe_int(g["is_from_pattern_restore"].iloc[0], 0) if "is_from_pattern_restore" in g.columns and len(g) >= 1 else 0

        # Flights time-like features
        is_time_like_column = int(
            looks_like_time_column(str(column), semantic_type)
            or (get_numeric_series(g, "is_time_like_column", 0.0).max() > 0)
        )

        candidate_is_valid_time_max = get_numeric_series(g, "candidate_is_valid_time", 0.0).max()
        avg_time_context_gain = get_numeric_series(g, "candidate_time_context_gain", 0.0).mean()
        max_time_context_gain = get_numeric_series(g, "candidate_time_context_gain", 0.0).max()
        min_time_diff = get_numeric_series(g, "candidate_dirty_time_minute_diff", 999999.0).min()
        time_source_any = get_numeric_series(g, "contains_time_source", 0.0).max()
        top1_time_context_gain = safe_float(g["candidate_time_context_gain"].iloc[0], 0.0) if "candidate_time_context_gain" in g.columns and len(g) >= 1 else 0.0
        top1_candidate_is_valid_time = safe_int(g["candidate_is_valid_time"].iloc[0], 0) if "candidate_is_valid_time" in g.columns and len(g) >= 1 else 0
        top1_candidate_in_time_range = safe_int(g["candidate_in_time_range"].iloc[0], 0) if "candidate_in_time_range" in g.columns and len(g) >= 1 else 0

        # Detection / verifier features if passed from upstream
        final_pred_error_prob = safe_float(first_non_null(g["final_pred_error_prob"], 0.0), 0.0) if "final_pred_error_prob" in g.columns else 0.0
        is_high_risk = safe_int(first_non_null(g["is_high_risk"], 0), 0) if "is_high_risk" in g.columns else 0
        is_hard_case_from_detection = safe_int(first_non_null(g["is_hard_case"], 0), 0) if "is_hard_case" in g.columns else 0

        if top_margin >= 0.25:
            confidence_bucket = "high_margin"
        elif top_margin >= 0.10:
            confidence_bucket = "mid_margin"
        else:
            confidence_bucket = "low_margin"

        if candidate_count <= 2:
            candidate_count_bucket = "small"
        elif candidate_count <= 4:
            candidate_count_bucket = "medium"
        else:
            candidate_count_bucket = "large"

        rows.append({
            "row_id": int(row_id),
            "column": str(column),
            "dirty_value": dirty_value,
            "semantic_type": semantic_type,
            "main_rule_type": main_rule_type,
            "main_usage_role": main_usage_role,
            "violation_count": float(violation_count) if pd.notna(violation_count) else 0.0,
            "conflict_score": float(conflict_score) if pd.notna(conflict_score) else 0.0,
            "candidate_count": int(candidate_count),
            "top1_score": float(top1_score),
            "top2_score": float(top2_score),
            "top_margin": float(top_margin),
            "avg_rule_support": float(rule_support_s.mean()),
            "max_rule_support": float(rule_support_s.max()),
            "avg_bayes": float(bayes_s.mean()),
            "max_bayes": float(bayes_s.max()),
            "avg_edit_sim": float(edit_sim_s.mean()),
            "max_edit_sim": float(edit_sim_s.max()),
            "source_count_top1": int(source_count_top1),
            "top1_is_from_rule": int(top1_is_from_rule),
            "top1_is_from_neighbor": int(top1_is_from_neighbor),
            "top1_is_from_similarity": int(top1_is_from_similarity),
            "top1_is_from_hint": int(top1_is_from_hint),
            "top1_is_from_dictionary": int(top1_is_from_dictionary),
            "top1_is_from_pattern_restore": int(top1_is_from_pattern_restore),

            "is_time_like_column": int(is_time_like_column),
            "candidate_is_valid_time_max": float(candidate_is_valid_time_max),
            "avg_time_context_gain": float(avg_time_context_gain),
            "max_time_context_gain": float(max_time_context_gain),
            "min_candidate_dirty_time_diff": float(min_time_diff if min_time_diff != 999999.0 else 0.0),
            "time_source_any": int(time_source_any),
            "top1_time_context_gain": float(top1_time_context_gain),
            "top1_candidate_is_valid_time": int(top1_candidate_is_valid_time),
            "top1_candidate_in_time_range": int(top1_candidate_in_time_range),

            "final_pred_error_prob": float(final_pred_error_prob),
            "is_high_risk": int(is_high_risk),
            "is_hard_case_from_detection": int(is_hard_case_from_detection),

            "confidence_bucket": confidence_bucket,
            "candidate_count_bucket": candidate_count_bucket,
            "is_hard_column": int(str(column) in HARD_COLUMNS or is_time_like_column == 1),
        })

    return pd.DataFrame(rows)


# ============================================================
# 5. 分桶
# ============================================================

def get_bucket_keys() -> List[str]:
    keys = [
        "semantic_type",
        "main_rule_type",
        "main_usage_role",
        "confidence_bucket",
        "candidate_count_bucket",
        "is_hard_column",
        "is_time_like_column",
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
# 6. 聚类特征
# ============================================================

FEATURE_COLUMNS_NUMERIC = [
    "violation_count",
    "conflict_score",
    "candidate_count",
    "top1_score",
    "top2_score",
    "top_margin",
    "avg_rule_support",
    "max_rule_support",
    "avg_bayes",
    "max_bayes",
    "avg_edit_sim",
    "max_edit_sim",
    "source_count_top1",
    "top1_is_from_rule",
    "top1_is_from_neighbor",
    "top1_is_from_similarity",
    "top1_is_from_hint",
    "top1_is_from_dictionary",
    "top1_is_from_pattern_restore",
    "is_hard_column",

    "is_time_like_column",
    "candidate_is_valid_time_max",
    "avg_time_context_gain",
    "max_time_context_gain",
    "min_candidate_dirty_time_diff",
    "time_source_any",
    "top1_time_context_gain",
    "top1_candidate_is_valid_time",
    "top1_candidate_in_time_range",

    "final_pred_error_prob",
    "is_high_risk",
    "is_hard_case_from_detection",
]

FEATURE_COLUMNS_CATEGORICAL = [
    "semantic_type",
    "main_rule_type",
    "main_usage_role",
    "confidence_bucket",
    "candidate_count_bucket",
]


def prepare_features(bucket_df: pd.DataFrame):
    work = bucket_df.copy()

    for col in FEATURE_COLUMNS_NUMERIC:
        if col not in work.columns:
            work[col] = 0.0
        work[col] = pd.to_numeric(work[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    num_df = work[FEATURE_COLUMNS_NUMERIC].copy()

    cat_frames = []
    for col in FEATURE_COLUMNS_CATEGORICAL:
        if col not in work.columns:
            work[col] = "unknown"
        dummies = pd.get_dummies(work[col].fillna("unknown").astype(str), prefix=col)
        cat_frames.append(dummies)

    feat_df = pd.concat([num_df] + cat_frames, axis=1) if cat_frames else num_df

    # 如果某桶内只有一行，StandardScaler 没必要；但函数外已处理，这里仍保底
    scaler = StandardScaler()
    X = scaler.fit_transform(feat_df.values)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, feat_df.columns.tolist()


# ============================================================
# 7. 自动选簇数
# ============================================================

def choose_k(n: int) -> int:
    if n < MIN_BUCKET_SIZE_FOR_CLUSTER:
        return 1
    k = max(2, round(n / TARGET_CLUSTER_SIZE))
    k = min(k, MAX_CLUSTERS_PER_BUCKET)
    k = min(k, n)
    return k


# ============================================================
# 8. 桶内聚类
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
# 9. 每簇采样
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
# 10. 为 sampled cell 准备候选集与详细上下文
# ============================================================

def safe_jsonable(x):
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass

    if isinstance(x, (np.integer, )):
        return int(x)
    if isinstance(x, (np.floating, )):
        return float(x)
    return x


def get_max_candidates_for_column(column: str) -> int:
    return HARD_COLUMN_MAX_CANDIDATES if str(column) in HARD_COLUMNS else DEFAULT_MAX_CANDIDATES_PER_CELL_FOR_LLM


CANDIDATE_FEATURE_EXPORT_COLS = [
    "candidate_value",
    "candidate_rank",
    "candidate_source",
    "source_count",
    "source_score",
    "is_from_rule",
    "is_from_neighbor",
    "is_from_similarity",
    "is_from_hint",
    "is_from_dictionary",
    "is_from_pattern_restore",
    "is_rule_and_dictionary_supported",
    "candidate_rule_support_ratio",
    "rule_support_gain",
    "candidate_matches_similar_row_target",
    "candidate_match_ratio_in_similar_rows",
    "bayes_mean_cond_prob",
    "bayes_max_cond_prob",
    "edit_similarity_to_dirty",
    "avg_keyboard_distance",
    "phonetic_similarity_to_dirty",
    "candidate_in_column_top_values",

    # Hospital-style
    "candidate_looks_like_measurecode",
    "candidate_looks_like_stateavg",
    "candidate_family_match_dirty",
    "candidate_suffix_match_dirty",
    "candidate_state_prefix_match_row_state",
    "candidate_stateavg_matches_row_measurecode",
    "candidate_condition_matches_measurecode_family",
    "candidate_name_anchor_overlap_with_condition",
    "candidate_name_anchor_overlap_with_measurecode_family",
    "candidate_is_valid_percent",
    "dirty_is_valid_percent",
    "candidate_percent_restoration_gain",
    "candidate_is_valid_sample_pattern",
    "dirty_is_valid_sample_pattern",
    "candidate_sample_restoration_gain",

    # Flights time-style
    "contains_time_source",
    "candidate_is_valid_time",
    "dirty_is_valid_time",
    "candidate_time_format_restoration_gain",
    "candidate_time_minutes",
    "dirty_time_minutes",
    "candidate_dirty_time_minute_diff",
    "candidate_in_time_range",
    "candidate_outside_time_range",
    "candidate_min_diff_to_row_time",
    "candidate_mean_diff_to_row_time",
    "dirty_min_diff_to_row_time",
    "candidate_time_context_gain",
]


def build_candidate_list_for_cell(cell_key: Tuple[int, str], cand_df: pd.DataFrame) -> List[Dict[str, Any]]:
    row_id, column = cell_key
    g = cand_df[(cand_df["row_id"] == row_id) & (cand_df["column"] == column)].copy()

    score_col = None
    if "final_score_from_candidate_gen" in g.columns:
        score_col = "final_score_from_candidate_gen"
    elif "final_score" in g.columns:
        score_col = "final_score"

    if score_col:
        if "candidate_rank" in g.columns:
            g = g.sort_values([score_col, "candidate_rank"], ascending=[False, True])
        else:
            g = g.sort_values([score_col], ascending=[False])
    elif "candidate_rank" in g.columns:
        g = g.sort_values(["candidate_rank"], ascending=[True])

    limit = get_max_candidates_for_column(column)

    if DROP_UNKNOWN_IF_TOO_MANY and len(g) > limit and "candidate_value" in g.columns:
        g = g[g["candidate_value"].astype(str) != UNKNOWN_CANDIDATE_NAME]

    g = g.head(limit).copy()

    candidate_rows = []
    for _, r in g.iterrows():
        item = {}

        for col in CANDIDATE_FEATURE_EXPORT_COLS:
            if col in r.index:
                item[col] = safe_jsonable(r.get(col))

        if score_col:
            item["candidate_score"] = safe_jsonable(r.get(score_col))
        else:
            item["candidate_score"] = None

        candidate_rows.append(item)

    return candidate_rows


def build_sampled_jsonl(sampled_df: pd.DataFrame, cand_df: pd.DataFrame, detail_map: Dict[Tuple[int, str], Dict[str, Any]]):
    with open(OUTPUT_SAMPLED_JSONL, "w", encoding="utf-8") as f:
        for _, row in sampled_df.iterrows():
            key = (int(row["row_id"]), row["column"])
            detail_obj = detail_map.get(key, {})

            out = {
                "row_id": int(row["row_id"]),
                "column": row["column"],
                "dirty_value": row.get("dirty_value"),
                "bucket_id": row["bucket_id"],
                "cluster_id": int(row["cluster_id"]),
                "sample_role": row["sample_role"],
                "dist_to_center": float(row["dist_to_center"]),
                "cell_summary": {
                    "semantic_type": row.get("semantic_type"),
                    "main_rule_type": row.get("main_rule_type"),
                    "main_usage_role": row.get("main_usage_role"),
                    "violation_count": row.get("violation_count"),
                    "conflict_score": row.get("conflict_score"),
                    "candidate_count": row.get("candidate_count"),
                    "top1_score": row.get("top1_score"),
                    "top2_score": row.get("top2_score"),
                    "top_margin": row.get("top_margin"),
                    "confidence_bucket": row.get("confidence_bucket"),
                    "candidate_count_bucket": row.get("candidate_count_bucket"),
                    "is_hard_column": row.get("is_hard_column"),
                    "is_time_like_column": row.get("is_time_like_column"),
                    "avg_time_context_gain": row.get("avg_time_context_gain"),
                    "max_time_context_gain": row.get("max_time_context_gain"),
                    "time_source_any": row.get("time_source_any"),
                    "final_pred_error_prob": row.get("final_pred_error_prob"),
                    "is_high_risk": row.get("is_high_risk"),
                    "is_hard_case_from_detection": row.get("is_hard_case_from_detection"),
                },
                "candidates": build_candidate_list_for_cell(key, cand_df),
                "detail_context": detail_obj
            }
            write_jsonl_record(f, out)


# ============================================================
# 11. 构建 pairwise 请求
# ============================================================

def make_pair_ids(candidates: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    n = len(candidates)
    if n <= 1:
        return []

    if PAIRWISE_STRATEGY == "one_vs_top":
        top_idx = 0
        return [(top_idx, j) for j in range(1, n)]

    return list(combinations(range(n), 2))


def build_context_for_prompt(detail_context: Dict[str, Any], dirty_value: str, column: str) -> Dict[str, Any]:
    out = {
        "column": column,
        "dirty_value": dirty_value,
    }

    if not isinstance(detail_context, dict):
        return out

    row_ctx = detail_context.get("row_context", {})
    if isinstance(row_ctx, dict):
        row_values = row_ctx.get("row_values", {})
        if isinstance(row_values, dict):
            out["row_values"] = row_values

        # Flights rich time bundle
        time_bundle = row_ctx.get("time_bundle", {})
        if isinstance(time_bundle, dict):
            out["time_bundle"] = time_bundle

        row_patterns = row_ctx.get("row_patterns", {})
        if isinstance(row_patterns, dict):
            out["row_patterns"] = row_patterns

    col_ctx = detail_context.get("column_context", {})
    if isinstance(col_ctx, dict):
        out["column_context"] = {
            "semantic_type": col_ctx.get("semantic_type"),
            "detected_type": col_ctx.get("detected_type"),
            "top_values_with_counts": col_ctx.get("top_values_with_counts"),
            "dominant_pattern": col_ctx.get("dominant_pattern"),
            "current_value_pattern": col_ctx.get("current_value_pattern"),
            "numeric_stats": col_ctx.get("numeric_stats"),
            "time_stats": col_ctx.get("time_stats"),
            "time_value_minutes": col_ctx.get("time_value_minutes"),
            "top_patterns_with_counts": col_ctx.get("top_patterns_with_counts"),
        }

    conf_ctx = detail_context.get("conflict_context", {})
    if isinstance(conf_ctx, dict):
        violated = conf_ctx.get("violated_rules_detailed", [])
        out["conflict_context"] = {
            "has_rule_violation": conf_ctx.get("has_rule_violation"),
            "violation_count": conf_ctx.get("violation_count"),
            "strong_rule_count": conf_ctx.get("strong_rule_count"),
            "candidate_generation_rule_count": conf_ctx.get("candidate_generation_rule_count"),
            "fd_like_count": conf_ctx.get("fd_like_count"),
            "time_window_rule_count": conf_ctx.get("time_window_rule_count"),
            "violated_rules_detailed": violated[:10] if isinstance(violated, list) else [],
            "expected_values_from_rules": conf_ctx.get("expected_values_from_rules", []),
            "alternative_values_from_column": conf_ctx.get("alternative_values_from_column", []),
            "all_alternative_values": conf_ctx.get("all_alternative_values", []),
        }

    sim_ctx = detail_context.get("similarity_context", {})
    if isinstance(sim_ctx, dict):
        out["similarity_context"] = {
            "neighbor_majority_value": sim_ctx.get("neighbor_majority_value"),
            "neighbor_majority_ratio": sim_ctx.get("neighbor_majority_ratio"),
            "neighbor_value_distribution": sim_ctx.get("neighbor_value_distribution"),
            "topk_similar_rows": sim_ctx.get("topk_similar_rows", [])[:5],
        }

    bayes_ctx = detail_context.get("bayesian_context", {})
    if isinstance(bayes_ctx, dict):
        out["bayesian_context"] = bayes_ctx

    return out


def build_column_specific_instruction(column: str) -> str:
    column = str(column)

    if column in TIME_COLUMNS or column.endswith("_time"):
        return (
            "本列是航班时间列。优先选择满足以下条件的候选："
            "1) 是合法时间格式，例如 '7:10 a.m.' 或 '4:00 p.m.'；"
            "2) 与 violated rules / expected_values_from_rules 中的建议值一致；"
            "3) 与同行的 flight、act_dep_time、sched_arr_time、act_arr_time 等时间上下文一致；"
            "4) 不要仅因为列内高频就选择候选；"
            "5) 对空值或 nan，规则强支持的具体时间通常比泛化高频时间更可信。"
        )

    if column in {"MeasureCode", "Stateavg"}:
        return (
            "本列是结构化编码列。优先看 family/prefix/suffix 是否一致，"
            "以及它是否与同行的 State / MeasureCode / Stateavg 保持结构一致；"
            "不要只因为字符串相似就选择 sibling code。"
        )

    if column == "Condition":
        return (
            "本列是疾病/主题分类列。优先看它是否与 MeasureCode family 一致："
            "hf->heart failure, ami->heart attack, pn->pneumonia, scip->surgical infection prevention。"
        )

    if column == "MeasureName":
        return (
            "本列是长文本 canonical phrase。优先看它是否与 MeasureCode 和 Condition 一致，"
            "不要只因为词面相似就选择语义不匹配的候选。"
        )

    if column == "Score":
        return (
            "本列是百分比。已经是合法百分号格式的原值要非常谨慎改动；"
            "优先恢复 malformed percent（例如 95x -> 95%），不要无根据把合法分数改成别的合法分数。"
        )

    if column == "Sample":
        return (
            "本列通常应符合 '\\d+ patients' 模式。优先恢复 pattern typo，"
            "不要随意补空值，也不要选择不符合样式的候选。"
        )

    return (
        "优先考虑规则一致性、同行上下文一致性、统计支持和合理的拼写/模式变换。"
    )


def build_pairwise_prompt(context_obj: Dict[str, Any], cand_a: Dict[str, Any], cand_b: Dict[str, Any], column: str) -> str:
    column_instruction = build_column_specific_instruction(column)

    prompt = f"""
你现在在做表格数据清洗中的错误修复排序任务。

目标：
对于一个被检测为错误的单元格，请在两个候选修复值 A 和 B 中判断谁更适合作为最终修复值。
你必须只在 A、B、Unknown 中三选一：
- A：候选 A 更好
- B：候选 B 更好
- Unknown：无法可靠判断，或者两个都不够好

通用判断原则：
1. 优先考虑规则一致性：FD、soft FD、上下文主导值、时间约束等是否支持该候选
2. 再考虑同行上下文一致性：该候选是否与同一行其它属性更匹配
3. 再考虑统计支持：列内高频、共现概率、邻居支持、similar row 支持
4. 再考虑变换合理性：拼写相似、键盘误触、发音相似、pattern/time format 恢复
5. 如果两个候选都明显不合理，输出 Unknown
6. 不要仅因为候选更常见就无条件选择它
7. 对时间列，优先选择规则强支持且符合时间上下文的候选

本列专属判断：
{column_instruction}

请严格输出 JSON，不要输出额外解释文字。
JSON 格式如下：
{{
  "winner": "A" 或 "B" 或 "Unknown",
  "reason_type": "rule_consistency" / "context_match" / "time_consistency" / "cooccurrence_support" / "spelling_similarity" / "pattern_match" / "uncertain" / "other",
  "reason": "一句简短原因"
}}

下面是待判断样本：

上下文：
{json.dumps(context_obj, ensure_ascii=False, indent=2)}

候选 A：
{json.dumps(cand_a, ensure_ascii=False, indent=2)}

候选 B：
{json.dumps(cand_b, ensure_ascii=False, indent=2)}
""".strip()
    return prompt


def build_pairwise_requests(sampled_df: pd.DataFrame, cand_df: pd.DataFrame, detail_map: Dict[Tuple[int, str], Dict[str, Any]]) -> List[Dict[str, Any]]:
    requests = []

    for _, row in sampled_df.iterrows():
        row_id = int(row["row_id"])
        column = str(row["column"])
        dirty_value = row.get("dirty_value")
        key = (row_id, column)

        candidates = build_candidate_list_for_cell(key, cand_df)
        if len(candidates) <= 1:
            continue

        pair_ids = make_pair_ids(candidates)
        detail_obj = detail_map.get(key, {})
        context_obj = build_context_for_prompt(detail_obj, dirty_value, column)

        for pair_idx, (i, j) in enumerate(pair_ids):
            cand_a = candidates[i]
            cand_b = candidates[j]

            req = {
                "request_id": f"{row_id}__{column}__pair{pair_idx}",
                "row_id": row_id,
                "column": column,
                "dirty_value": dirty_value,
                "candidate_a_index": i,
                "candidate_b_index": j,
                "candidate_a_value": cand_a.get("candidate_value"),
                "candidate_b_value": cand_b.get("candidate_value"),
                "pairwise_prompt": build_pairwise_prompt(context_obj, cand_a, cand_b, column),
                "context_obj": context_obj,
                "candidate_a": cand_a,
                "candidate_b": cand_b,
            }
            requests.append(req)

    return requests


# ============================================================
# 12. LLM 调用
# ============================================================

def get_client() -> OpenAI:
    if not API_KEY:
        raise ValueError("未设置 DEEPSEEK_API_KEY，无法调用 LLM。建议先设置 DRY_RUN_ONLY_EXPORT_REQUESTS=True 只导出请求。")
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def call_llm_pairwise(client: OpenAI, prompt: str) -> Dict[str, Any]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=TEMPERATURE,
                messages=[
                    {"role": "system", "content": "你是一个严谨的表格数据清洗排序专家。你必须只输出 JSON。"},
                    {"role": "user", "content": prompt},
                ],
            )
            text = resp.choices[0].message.content.strip()

            try:
                return json.loads(text)
            except Exception:
                start = text.find("{")
                end = text.rfind("}")
                if start != -1 and end != -1 and end > start:
                    return json.loads(text[start:end+1])
                raise ValueError(f"LLM 输出不是合法 JSON: {text}")

        except Exception as e:
            if attempt == MAX_RETRIES:
                return {
                    "winner": "Unknown",
                    "reason_type": "uncertain",
                    "reason": f"LLM 调用失败: {str(e)}"
                }
            time.sleep(RETRY_SLEEP_SECONDS)


# ============================================================
# 13. 聚合 pairwise 输出为 teacher 排序
# ============================================================

def aggregate_pairwise_results(pairwise_results: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    grouped = defaultdict(list)
    for x in pairwise_results:
        grouped[(int(x["row_id"]), str(x["column"]))].append(x)

    final_rows = []
    top1_rows = []

    for (row_id, column), items in grouped.items():
        candidate_counter = {}
        wins = Counter()
        losses = Counter()
        reason_counter = Counter()

        dirty_value = items[0].get("dirty_value", "")

        for x in items:
            a = x.get("candidate_a_value", None)
            b = x.get("candidate_b_value", None)
            candidate_counter[a] = 1
            candidate_counter[b] = 1

            winner = x.get("winner", "Unknown")
            reason_type = x.get("reason_type", "uncertain")
            reason_counter[reason_type] += 1

            if winner == "A":
                wins[a] += 1
                losses[b] += 1
            elif winner == "B":
                wins[b] += 1
                losses[a] += 1

        def safe_candidate_name(x):
            if x is None:
                return ""
            return str(x)

        all_candidates = list(candidate_counter.keys())
        scores = []
        for cand in all_candidates:
            score = wins[cand] - losses[cand]
            scores.append((cand, score, wins[cand], losses[cand]))

        scores = sorted(
            scores,
            key=lambda x: (-x[1], -x[2], x[3], safe_candidate_name(x[0]))
        )

        for rank_idx, (cand, score, win_cnt, loss_cnt) in enumerate(scores, start=1):
            final_rows.append({
                "row_id": row_id,
                "column": column,
                "dirty_value": dirty_value,
                "teacher_rank": rank_idx,
                "candidate_value": cand,
                "pairwise_score": score,
                "win_count": win_cnt,
                "loss_count": loss_cnt,
            })

        if scores:
            best_cand = "" if scores[0][0] is None else scores[0][0]
            dominant_reason = reason_counter.most_common(1)[0][0] if reason_counter else "uncertain"
            top1_rows.append({
                "row_id": row_id,
                "column": column,
                "dirty_value": dirty_value,
                "teacher_best_candidate": best_cand,
                "teacher_reason_type": dominant_reason,
                "candidate_count_in_pairwise": len(all_candidates),
            })

    return final_rows, top1_rows


# ============================================================
# 14. 主流程
# ============================================================

def main():
    ensure_output_paths()

    if not os.path.exists(INPUT_CANDIDATE_FEATURES_CSV):
        raise FileNotFoundError(f"候选特征文件不存在: {INPUT_CANDIDATE_FEATURES_CSV}")

    cand_df = pd.read_csv(INPUT_CANDIDATE_FEATURES_CSV, encoding="utf-8-sig")
    required = {"row_id", "column", "candidate_value"}
    missing = required - set(cand_df.columns)
    if missing:
        raise ValueError(f"候选特征文件缺少必要字段: {missing}")

    cand_df["row_id"] = pd.to_numeric(cand_df["row_id"], errors="raise").astype(int)
    cand_df["column"] = cand_df["column"].astype(str)

    detail_map = load_detail_jsonl(INPUT_DETAIL_JSONL)
    print(f"[INFO] candidate feature rows: {len(cand_df)}")
    print(f"[INFO] detail context cells: {len(detail_map)}")

    # 1) 构造 cell-level 摘要
    cell_summary_df = build_cell_summary_df(cand_df)
    cell_summary_df.to_csv(OUTPUT_CELL_SUMMARY_CSV, index=False, encoding="utf-8-sig")

    # 2) 分桶
    cell_summary_df = assign_buckets(cell_summary_df)

    clustered_parts = []
    bucket_summary_rows = []

    for bucket_id, bucket_df in cell_summary_df.groupby("bucket_id", sort=False):
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
                "confidence_bucket": str(cluster_df["confidence_bucket"].iloc[0]) if "confidence_bucket" in cluster_df.columns else "unknown",
                "candidate_count_bucket": str(cluster_df["candidate_count_bucket"].iloc[0]) if "candidate_count_bucket" in cluster_df.columns else "unknown",
                "is_hard_column": int(cluster_df["is_hard_column"].iloc[0]) if "is_hard_column" in cluster_df.columns else 0,
                "is_time_like_column": int(cluster_df["is_time_like_column"].iloc[0]) if "is_time_like_column" in cluster_df.columns else 0,
            })

        clustered_bucket = pd.concat(bucket_clustered_parts, ignore_index=True)
        clustered_parts.append(clustered_bucket)

    if clustered_parts:
        clustered_df = pd.concat(clustered_parts, ignore_index=True)
    else:
        clustered_df = pd.DataFrame()

    clustered_df.to_csv(OUTPUT_CLUSTERED_CSV, index=False, encoding="utf-8-sig")

    sampled_df = clustered_df[clustered_df["is_sampled"] == 1].copy() if not clustered_df.empty else pd.DataFrame()
    sampled_df.to_csv(OUTPUT_SAMPLED_CSV, index=False, encoding="utf-8-sig")

    if not sampled_df.empty:
        build_sampled_jsonl(sampled_df, cand_df, detail_map)
    else:
        with open(OUTPUT_SAMPLED_JSONL, "w", encoding="utf-8") as f:
            pass

    pd.DataFrame(bucket_summary_rows).to_csv(
        OUTPUT_BUCKET_SUMMARY_CSV, index=False, encoding="utf-8-sig"
    )

    # 3) 生成 pairwise 请求
    requests = build_pairwise_requests(sampled_df, cand_df, detail_map) if not sampled_df.empty else []

    with open(OUTPUT_PAIRWISE_REQUESTS_JSONL, "w", encoding="utf-8") as f:
        for req in requests:
            write_jsonl_record(f, req)

    print(f"[OK] cell-level 摘要已保存: {OUTPUT_CELL_SUMMARY_CSV}")
    print(f"[OK] 完整聚类结果已保存: {OUTPUT_CLUSTERED_CSV}")
    print(f"[OK] 采样摘要 CSV 已保存: {OUTPUT_SAMPLED_CSV}")
    print(f"[OK] 采样详细 JSONL 已保存: {OUTPUT_SAMPLED_JSONL}")
    print(f"[OK] bucket/cluster 摘要已保存: {OUTPUT_BUCKET_SUMMARY_CSV}")
    print(f"[OK] pairwise 请求已保存: {OUTPUT_PAIRWISE_REQUESTS_JSONL}")
    print(f"[INFO] 采样 cell 数: {len(sampled_df)}")
    print(f"[INFO] pairwise 请求数: {len(requests)}")

    if DRY_RUN_ONLY_EXPORT_REQUESTS:
        print("[INFO] 当前为 DRY RUN，仅导出请求，不调用 LLM。")
        return

    # 4) 调用 LLM
    client = get_client()

    responses = []
    with open(OUTPUT_PAIRWISE_RESPONSES_JSONL, "w", encoding="utf-8") as f:
        for idx, req in enumerate(requests, start=1):
            if idx % 20 == 0:
                print(f"[INFO] LLM progress: {idx}/{len(requests)}")

            llm_out = call_llm_pairwise(client, req["pairwise_prompt"])

            item = {
                "request_id": req["request_id"],
                "row_id": req["row_id"],
                "column": req["column"],
                "dirty_value": req["dirty_value"],
                "candidate_a_index": req["candidate_a_index"],
                "candidate_b_index": req["candidate_b_index"],
                "candidate_a_value": req["candidate_a_value"],
                "candidate_b_value": req["candidate_b_value"],
                "winner": llm_out.get("winner", "Unknown"),
                "reason_type": llm_out.get("reason_type", "uncertain"),
                "reason": llm_out.get("reason", ""),
            }
            responses.append(item)
            write_jsonl_record(f, item)

    # 5) 聚合为 teacher 排序结果
    ranking_rows, top1_rows = aggregate_pairwise_results(responses)

    pd.DataFrame(ranking_rows).to_csv(OUTPUT_TEACHER_RANKING_CSV, index=False, encoding="utf-8-sig")
    pd.DataFrame(top1_rows).to_csv(OUTPUT_TEACHER_TOP1_CSV, index=False, encoding="utf-8-sig")

    print(f"[OK] pairwise 响应已保存: {OUTPUT_PAIRWISE_RESPONSES_JSONL}")
    print(f"[OK] teacher 排序结果已保存: {OUTPUT_TEACHER_RANKING_CSV}")
    print(f"[OK] teacher top1 标签已保存: {OUTPUT_TEACHER_TOP1_CSV}")

    print("\n===== 统计信息 =====")
    print(f"总 cell 数: {cell_summary_df.shape[0]}")
    print(f"总采样 cell 数: {sampled_df.shape[0]}")
    print(f"总 pairwise 请求数: {len(requests)}")
    if not clustered_df.empty:
        print(f"总桶数: {clustered_df['bucket_id'].nunique()}")
        print(f"总簇数: {clustered_df[['bucket_id', 'cluster_id']].drop_duplicates().shape[0]}")

    if len(sampled_df) > 0:
        print("\n采样角色分布:")
        print(sampled_df["sample_role"].value_counts(dropna=False).to_string())

        if "is_time_like_column" in sampled_df.columns:
            print("\n时间列采样分布:")
            print(sampled_df["is_time_like_column"].value_counts(dropna=False).to_string())

        print("\n采样列分布:")
        print(sampled_df["column"].value_counts(dropna=False).head(20).to_string())


if __name__ == "__main__":
    main()

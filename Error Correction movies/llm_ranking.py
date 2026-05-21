"""
Movies sampling + LLM pairwise labeling script, full logic-migrated from the Flights sampling/labeling pipeline.

Migration principle
-------------------
Keep the full Flights framework:
1. read candidate-level features after whitelist filtering
2. aggregate into cell-level summaries
3. bucket by semantic/rule/candidate/margin characteristics
4. cluster within each bucket
5. sample representative cells from clusters
6. export sampled cell CSV + JSONL
7. build pairwise comparison requests
8. call LLM concurrently with resume support
9. aggregate pairwise labels into teacher ranking and teacher top1

Convert Flights-specific logic:
- time/flight consensus features
    -> Movies duration/year/release/rating/count/list/metadata consistency features
- Flights time columns
    -> Movies hard columns: Duration, Year, Release Date, RatingValue, RatingCount, ReviewCount,
       Actors, Cast, Country, Language, Filming Locations
- Flights prompt
    -> Movies-specific prompt with typed field consistency and metadata consistency

Sampling policy for Movies
--------------------------
Movies has many rows and wide text/list columns, so this version intentionally avoids over-sampling:
- USE_COLUMN_IN_BUCKET = False by default
- MAX_CLUSTERS_PER_BUCKET = 5
- TARGET_CLUSTER_SIZE = 25
- SMALL_BUCKET_FULL_SAMPLE_THRESHOLD = 3
- SAMPLE_OUTLIER = False
- GLOBAL_MAX_SAMPLED_CELLS = 260
- MAX_SAMPLED_CELLS_PER_COLUMN = 45
- PAIRWISE_STRATEGY = one_vs_top

Outputs
-------
repair_cell_summary_for_sampling_v2.csv
repair_cell_clustered_v2.csv
repair_cell_sampled_v2.csv
repair_cell_sampled_with_candidates_v2.jsonl
repair_pairwise_requests_v2.jsonl
repair_pairwise_responses_v2.jsonl
repair_teacher_pairwise_ranking_v2.csv
repair_teacher_top1_labels_v2.csv
repair_bucket_cluster_summary_v2.csv
"""

import os
import json
import time
import math
import re
from collections import Counter, defaultdict
from itertools import combinations
from typing import Dict, Any, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from openai import OpenAI


# ============================================================
# 1. 用户配置区
# ============================================================

# 输入1：候选特征文件
# 推荐使用白名单过滤后的特征文件，避免把检测阶段未判错的 cell 送入 repair 标注。
INPUT_CANDIDATE_FEATURES_CSV = "repair_candidate_features_infer_whitelist.csv"

# 输入2：详细上下文 rich jsonl
INPUT_DETAIL_JSONL = "/mnt/mydata/dq/projects/Splittree/test/Error Detection movies/candidate_llm_contexts_movies.jsonl"

# 输出：相对路径默认输出到当前运行目录
OUTPUT_CELL_SUMMARY_CSV = "repair_cell_summary_for_sampling_v2.csv"
OUTPUT_CLUSTERED_CSV = "repair_cell_clustered_v2.csv"
OUTPUT_SAMPLED_CSV = "repair_cell_sampled_v2.csv"
OUTPUT_SAMPLED_JSONL = "repair_cell_sampled_with_candidates_v2.jsonl"

OUTPUT_PAIRWISE_REQUESTS_JSONL = "repair_pairwise_requests_v2.jsonl"
OUTPUT_PAIRWISE_RESPONSES_JSONL = "repair_pairwise_responses_v2.jsonl"
OUTPUT_TEACHER_RANKING_CSV = "repair_teacher_pairwise_ranking_v2.csv"
OUTPUT_TEACHER_TOP1_CSV = "repair_teacher_top1_labels_v2.csv"
OUTPUT_BUCKET_SUMMARY_CSV = "repair_bucket_cluster_summary_v2.csv"

# Movies 行数较多，默认不要按 column 过度切桶，否则桶数会过多、采样数量会膨胀。
USE_COLUMN_IN_BUCKET = False

# 聚类参数：Movies 默认比 Flights 更保守
MIN_BUCKET_SIZE_FOR_CLUSTER = 8
MAX_CLUSTERS_PER_BUCKET = 5
TARGET_CLUSTER_SIZE = 25
SMALL_BUCKET_FULL_SAMPLE_THRESHOLD = 3

# 每簇采样策略：Movies 不默认抽 outlier，避免异常文本/list 导致 LLM 请求膨胀。
SAMPLE_CENTER = True
SAMPLE_BOUNDARY = True
SAMPLE_OUTLIER = False

RANDOM_STATE = 42

# 全局采样上限：防止 Movies 数据集过大导致 LLM 请求过多
GLOBAL_MAX_SAMPLED_CELLS = 260
MAX_SAMPLED_CELLS_PER_COLUMN = 45
MAX_SAMPLED_CELLS_PER_BUCKET = 35

# 候选控制
DEFAULT_MAX_CANDIDATES_PER_CELL_FOR_LLM = 5
HARD_COLUMN_MAX_CANDIDATES = 6
TEXT_LIST_COLUMN_MAX_CANDIDATES = 4

# Movies hard columns
MOVIE_TIME_COLUMNS = {"Year", "Release Date", "Duration"}
MOVIE_RATING_COLUMNS = {"RatingValue", "RatingCount", "ReviewCount"}
MOVIE_TEXT_LIST_COLUMNS = {"Actors", "Cast", "Creator", "Director", "Genre", "Filming Locations"}
MOVIE_METADATA_COLUMNS = {"Name", "Country", "Language"}
MOVIE_HARD_COLUMNS = MOVIE_TIME_COLUMNS | MOVIE_RATING_COLUMNS | MOVIE_TEXT_LIST_COLUMNS | MOVIE_METADATA_COLUMNS

DROP_UNKNOWN_IF_TOO_MANY = False

# Pairwise 策略
# one_vs_top 请求数显著少于 all_pairs，Movies 默认使用 one_vs_top。
PAIRWISE_STRATEGY = "one_vs_top"   # 可选: all_pairs / one_vs_top
UNKNOWN_CANDIDATE_NAME = "Unknown"

# LLM 配置（OpenAI 兼容接口，可接 DeepSeek）
MODEL_NAME = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-fd93f5216ae0488f8bf32d9e2a14a14a")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
TEMPERATURE = 0.0
MAX_RETRIES = 3
RETRY_SLEEP_SECONDS = 2.0

# LLM 并发配置：
# Movies 请求量较大，默认 6；如果限流，调小。
LLM_MAX_WORKERS = int(os.getenv("LLM_MAX_WORKERS", "6"))

# 每完成多少条写一次进度
LLM_PROGRESS_EVERY = int(os.getenv("LLM_PROGRESS_EVERY", "20"))

# 是否跳过已有 response，用于断点续跑
RESUME_FROM_EXISTING_RESPONSES = True

# 若只想先导出采样与 pairwise 请求，不实际调用 LLM，可设为 True
DRY_RUN_ONLY_EXPORT_REQUESTS = False


# ============================================================
# 2. 基础工具
# ============================================================

EMPTY_TOKENS = {"", "nan", "none", "null", "na", "n/a", "missing", "empty", "{null}"}


def write_jsonl_record(fp, obj):
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")


def resolve_output_path(path: str, anchor_file: str = None) -> str:
    """
    输出路径：
    - 绝对路径：原样使用；
    - 相对路径：保存到当前运行目录。
    """
    if os.path.isabs(str(path)):
        return str(path)
    return os.path.abspath(str(path))


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
    if s in EMPTY_TOKENS:
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


def get_numeric_series(g: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in g.columns:
        return pd.to_numeric(g[col], errors="coerce").fillna(default)
    return pd.Series([default] * len(g), index=g.index)


def first_non_null(series: pd.Series, default=None):
    for x in series:
        if pd.notna(x):
            return x
    return default


def looks_like_movie_time_column(column: str, semantic_type: Optional[str] = None) -> bool:
    col = str(column).strip()
    sem = norm_text(semantic_type)
    return col in MOVIE_TIME_COLUMNS or sem in {"year_like", "date_like", "duration_like"}


def looks_like_movie_rating_column(column: str, semantic_type: Optional[str] = None) -> bool:
    col = str(column).strip()
    sem = norm_text(semantic_type)
    return col in MOVIE_RATING_COLUMNS or sem in {"rating_like", "count_like", "review_count_like"}


def looks_like_movie_text_list_column(column: str, semantic_type: Optional[str] = None) -> bool:
    col = str(column).strip()
    sem = norm_text(semantic_type)
    return col in MOVIE_TEXT_LIST_COLUMNS or sem in {"categorical_list", "text"}


def looks_like_movie_hard_column(column: str, semantic_type: Optional[str] = None) -> bool:
    col = str(column).strip()
    return (
        col in MOVIE_HARD_COLUMNS
        or looks_like_movie_time_column(column, semantic_type)
        or looks_like_movie_rating_column(column, semantic_type)
        or looks_like_movie_text_list_column(column, semantic_type)
    )


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

def build_cell_summary_df(cand_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = cand_df.groupby(["row_id", "column"], sort=False)

    for (row_id, column), g in grouped:
        g = g.copy()

        if "final_score_from_candidate_gen" in g.columns:
            score_col = "final_score_from_candidate_gen"
        elif "final_score" in g.columns:
            score_col = "final_score"
        elif "source_score" in g.columns:
            score_col = "source_score"
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
        detected_type = first_non_null(g["detected_type"], "unknown") if "detected_type" in g.columns else "unknown"
        main_rule_type = first_non_null(g["main_rule_type"], "unknown") if "main_rule_type" in g.columns else "unknown"
        main_usage_role = first_non_null(g["main_usage_role"], "unknown") if "main_usage_role" in g.columns else "unknown"

        violation_count = safe_float(first_non_null(g["violation_count"], 0.0), 0.0) if "violation_count" in g.columns else 0.0
        conflict_score = safe_float(first_non_null(g["conflict_score"], 0.0), 0.0) if "conflict_score" in g.columns else 0.0

        candidate_count = len(g)
        top1_score = float(g[score_col].iloc[0]) if score_col and len(g) >= 1 else 0.0
        top2_score = float(g[score_col].iloc[1]) if score_col and len(g) >= 2 else 0.0
        top_margin = top1_score - top2_score if candidate_count >= 2 else top1_score

        rule_support_s = get_numeric_series(g, "candidate_rule_support_ratio", 0.0)
        rule_gain_s = get_numeric_series(g, "rule_support_gain", 0.0)
        bayes_s = get_numeric_series(g, "bayes_mean_cond_prob", 0.0)
        bayes_max_s = get_numeric_series(g, "bayes_max_cond_prob", 0.0)
        edit_sim_s = get_numeric_series(g, "edit_similarity_to_dirty", 0.0)
        phonetic_sim_s = get_numeric_series(g, "phonetic_similarity_to_dirty", 0.0)

        source_count_top1 = safe_int(g["source_count"].iloc[0], 0) if "source_count" in g.columns and len(g) >= 1 else 0

        top1_is_from_rule = safe_int(g["is_from_rule"].iloc[0], 0) if "is_from_rule" in g.columns and len(g) >= 1 else 0
        top1_is_from_neighbor = safe_int(g["is_from_neighbor"].iloc[0], 0) if "is_from_neighbor" in g.columns and len(g) >= 1 else 0
        top1_is_from_similarity = safe_int(g["is_from_similarity"].iloc[0], 0) if "is_from_similarity" in g.columns and len(g) >= 1 else 0
        top1_is_from_hint = safe_int(g["is_from_hint"].iloc[0], 0) if "is_from_hint" in g.columns and len(g) >= 1 else 0
        top1_is_from_dictionary = safe_int(g["is_from_dictionary"].iloc[0], 0) if "is_from_dictionary" in g.columns and len(g) >= 1 else 0
        top1_is_from_pattern_restore = safe_int(g["is_from_pattern_restore"].iloc[0], 0) if "is_from_pattern_restore" in g.columns and len(g) >= 1 else 0
        top1_is_from_movie_consistency = safe_int(g["is_from_movie_consistency"].iloc[0], 0) if "is_from_movie_consistency" in g.columns and len(g) >= 1 else 0

        # Movies typed features
        is_movie_time = int(
            looks_like_movie_time_column(str(column), semantic_type)
            or (get_numeric_series(g, "movie_column_is_time", 0.0).max() > 0)
            or (get_numeric_series(g, "allowed_is_movie_time_col", 0.0).max() > 0)
        )
        is_movie_rating = int(
            looks_like_movie_rating_column(str(column), semantic_type)
            or (get_numeric_series(g, "movie_column_is_rating", 0.0).max() > 0)
            or (get_numeric_series(g, "allowed_is_movie_rating_col", 0.0).max() > 0)
        )
        is_movie_people = int(
            str(column) in {"Director", "Creator", "Actors", "Cast"}
            or (get_numeric_series(g, "movie_column_is_people", 0.0).max() > 0)
            or (get_numeric_series(g, "allowed_is_movie_people_col", 0.0).max() > 0)
        )
        is_movie_list = int(
            looks_like_movie_text_list_column(str(column), semantic_type)
            or (get_numeric_series(g, "movie_column_is_list", 0.0).max() > 0)
            or (get_numeric_series(g, "allowed_is_movie_list_col", 0.0).max() > 0)
        )
        is_movie_text = int(
            str(column) in {"Name", "Description"}
            or (get_numeric_series(g, "movie_column_is_text", 0.0).max() > 0)
            or (get_numeric_series(g, "allowed_is_movie_text_col", 0.0).max() > 0)
        )

        movie_candidate_is_duration_max = get_numeric_series(g, "movie_candidate_is_duration", 0.0).max()
        movie_candidate_is_year_max = get_numeric_series(g, "movie_candidate_is_year", 0.0).max()
        movie_candidate_is_release_date_max = get_numeric_series(g, "movie_candidate_is_release_date", 0.0).max()
        movie_candidate_is_rating_value_max = get_numeric_series(g, "movie_candidate_is_rating_value", 0.0).max()
        movie_candidate_is_count_max = get_numeric_series(g, "movie_candidate_is_count", 0.0).max()
        movie_candidate_is_review_count_max = get_numeric_series(g, "movie_candidate_is_review_count", 0.0).max()

        movie_consistency_any = get_numeric_series(g, "contains_movie_consistency_source", 0.0).max()
        movie_consistency_strength_max = get_numeric_series(g, "movie_consistency_strength", 0.0).max()
        movie_consistency_confidence_max = get_numeric_series(g, "movie_consistency_confidence", 0.0).max()
        movie_consistency_support_max = get_numeric_series(g, "movie_consistency_support_count", 0.0).max()
        candidate_equals_allowed_consensus_any = get_numeric_series(g, "candidate_equals_allowed_consensus_value", 0.0).max()
        under_strong_movie_consistency_cell = get_numeric_series(g, "candidate_under_strong_movie_consistency_cell", 0.0).max()

        duration_restoration_any = get_numeric_series(g, "movie_candidate_duration_restoration", 0.0).max()
        year_restoration_any = get_numeric_series(g, "movie_candidate_year_restoration", 0.0).max()
        release_date_restoration_any = get_numeric_series(g, "movie_candidate_release_date_restoration", 0.0).max()
        rating_restoration_any = get_numeric_series(g, "movie_candidate_rating_value_restoration", 0.0).max()
        count_restoration_any = get_numeric_series(g, "movie_candidate_count_restoration", 0.0).max()
        review_count_restoration_any = get_numeric_series(g, "movie_candidate_review_count_restoration", 0.0).max()
        mojibake_repair_gain_any = get_numeric_series(g, "movie_mojibake_repair_gain", 0.0).max()

        # Detection / verifier features if passed from upstream
        final_pred_error_prob = safe_float(first_non_null(g["final_pred_error_prob"], 0.0), 0.0) if "final_pred_error_prob" in g.columns else 0.0
        is_high_risk = safe_int(first_non_null(g["is_high_risk"], 0), 0) if "is_high_risk" in g.columns else 0
        is_hard_case_from_detection = safe_int(first_non_null(g["is_hard_case"], 0), 0) if "is_hard_case" in g.columns else 0

        # allowed metadata if merged by whitelist
        allowed_is_movie_metadata_consistency = int(get_numeric_series(g, "allowed_is_movie_metadata_consistency", 0.0).max())
        allowed_is_strong_movie_consistency = int(get_numeric_series(g, "allowed_is_strong_movie_consistency", 0.0).max())
        allowed_recall_recovery_hardcase = int(get_numeric_series(g, "allowed_recall_recovery_hardcase", 0.0).max())
        allowed_sample_borderline_hardcase = int(get_numeric_series(g, "allowed_sample_borderline_hardcase", 0.0).max())

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

        if is_movie_time:
            movie_field_bucket = "time"
        elif is_movie_rating:
            movie_field_bucket = "rating"
        elif is_movie_people:
            movie_field_bucket = "people"
        elif is_movie_list:
            movie_field_bucket = "list"
        elif is_movie_text:
            movie_field_bucket = "text"
        else:
            movie_field_bucket = "other"

        is_hard_column = int(looks_like_movie_hard_column(str(column), semantic_type))

        rows.append({
            "row_id": int(row_id),
            "column": str(column),
            "dirty_value": dirty_value,
            "semantic_type": semantic_type,
            "detected_type": detected_type,
            "main_rule_type": main_rule_type,
            "main_usage_role": main_usage_role,
            "violation_count": float(violation_count),
            "conflict_score": float(conflict_score),
            "candidate_count": int(candidate_count),
            "top1_score": float(top1_score),
            "top2_score": float(top2_score),
            "top_margin": float(top_margin),
            "avg_rule_support": float(rule_support_s.mean()),
            "max_rule_support": float(rule_support_s.max()),
            "avg_rule_gain": float(rule_gain_s.mean()),
            "max_rule_gain": float(rule_gain_s.max()),
            "avg_bayes": float(bayes_s.mean()),
            "max_bayes": float(bayes_max_s.max()),
            "avg_edit_sim": float(edit_sim_s.mean()),
            "max_edit_sim": float(edit_sim_s.max()),
            "avg_phonetic_sim": float(phonetic_sim_s.mean()),
            "max_phonetic_sim": float(phonetic_sim_s.max()),
            "source_count_top1": int(source_count_top1),
            "top1_is_from_rule": int(top1_is_from_rule),
            "top1_is_from_neighbor": int(top1_is_from_neighbor),
            "top1_is_from_similarity": int(top1_is_from_similarity),
            "top1_is_from_hint": int(top1_is_from_hint),
            "top1_is_from_dictionary": int(top1_is_from_dictionary),
            "top1_is_from_pattern_restore": int(top1_is_from_pattern_restore),
            "top1_is_from_movie_consistency": int(top1_is_from_movie_consistency),

            "is_movie_time_col": int(is_movie_time),
            "is_movie_rating_col": int(is_movie_rating),
            "is_movie_people_col": int(is_movie_people),
            "is_movie_list_col": int(is_movie_list),
            "is_movie_text_col": int(is_movie_text),
            "movie_field_bucket": movie_field_bucket,

            "movie_candidate_is_duration_max": float(movie_candidate_is_duration_max),
            "movie_candidate_is_year_max": float(movie_candidate_is_year_max),
            "movie_candidate_is_release_date_max": float(movie_candidate_is_release_date_max),
            "movie_candidate_is_rating_value_max": float(movie_candidate_is_rating_value_max),
            "movie_candidate_is_count_max": float(movie_candidate_is_count_max),
            "movie_candidate_is_review_count_max": float(movie_candidate_is_review_count_max),

            "movie_consistency_any": int(movie_consistency_any),
            "movie_consistency_strength_max": float(movie_consistency_strength_max),
            "movie_consistency_confidence_max": float(movie_consistency_confidence_max),
            "movie_consistency_support_max": float(movie_consistency_support_max),
            "candidate_equals_allowed_consensus_any": int(candidate_equals_allowed_consensus_any),
            "under_strong_movie_consistency_cell": int(under_strong_movie_consistency_cell),

            "duration_restoration_any": int(duration_restoration_any),
            "year_restoration_any": int(year_restoration_any),
            "release_date_restoration_any": int(release_date_restoration_any),
            "rating_restoration_any": int(rating_restoration_any),
            "count_restoration_any": int(count_restoration_any),
            "review_count_restoration_any": int(review_count_restoration_any),
            "mojibake_repair_gain_any": int(mojibake_repair_gain_any),

            "final_pred_error_prob": float(final_pred_error_prob),
            "is_high_risk": int(is_high_risk),
            "is_hard_case_from_detection": int(is_hard_case_from_detection),
            "allowed_is_movie_metadata_consistency": allowed_is_movie_metadata_consistency,
            "allowed_is_strong_movie_consistency": allowed_is_strong_movie_consistency,
            "allowed_recall_recovery_hardcase": allowed_recall_recovery_hardcase,
            "allowed_sample_borderline_hardcase": allowed_sample_borderline_hardcase,

            "confidence_bucket": confidence_bucket,
            "candidate_count_bucket": candidate_count_bucket,
            "is_hard_column": int(is_hard_column),
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
        "movie_field_bucket",
        "is_hard_column",
        "movie_consistency_any",
        "allowed_is_strong_movie_consistency",
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
    "avg_rule_gain",
    "max_rule_gain",
    "avg_bayes",
    "max_bayes",
    "avg_edit_sim",
    "max_edit_sim",
    "avg_phonetic_sim",
    "max_phonetic_sim",
    "source_count_top1",
    "top1_is_from_rule",
    "top1_is_from_neighbor",
    "top1_is_from_similarity",
    "top1_is_from_hint",
    "top1_is_from_dictionary",
    "top1_is_from_pattern_restore",
    "top1_is_from_movie_consistency",

    "is_movie_time_col",
    "is_movie_rating_col",
    "is_movie_people_col",
    "is_movie_list_col",
    "is_movie_text_col",

    "movie_candidate_is_duration_max",
    "movie_candidate_is_year_max",
    "movie_candidate_is_release_date_max",
    "movie_candidate_is_rating_value_max",
    "movie_candidate_is_count_max",
    "movie_candidate_is_review_count_max",

    "movie_consistency_any",
    "movie_consistency_strength_max",
    "movie_consistency_confidence_max",
    "movie_consistency_support_max",
    "candidate_equals_allowed_consensus_any",
    "under_strong_movie_consistency_cell",

    "duration_restoration_any",
    "year_restoration_any",
    "release_date_restoration_any",
    "rating_restoration_any",
    "count_restoration_any",
    "review_count_restoration_any",
    "mojibake_repair_gain_any",

    "final_pred_error_prob",
    "is_high_risk",
    "is_hard_case_from_detection",
    "allowed_recall_recovery_hardcase",
    "allowed_sample_borderline_hardcase",
    "is_hard_column",
]

FEATURE_COLUMNS_CATEGORICAL = [
    "semantic_type",
    "detected_type",
    "main_rule_type",
    "main_usage_role",
    "confidence_bucket",
    "candidate_count_bucket",
    "movie_field_bucket",
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

    if SAMPLE_BOUNDARY:
        boundary_idx = max(0, int(round(0.70 * (n - 1))))
        if boundary_idx not in sampled_indices:
            sampled_indices.append(boundary_idx)
            roles[boundary_idx] = "boundary"

    if SAMPLE_OUTLIER:
        outlier_idx = n - 1
        if outlier_idx not in sampled_indices:
            sampled_indices.append(outlier_idx)
            roles[outlier_idx] = "outlier"

    cluster_df["sample_role"] = ""
    cluster_df["is_sampled"] = 0

    for idx in sampled_indices:
        cluster_df.loc[idx, "sample_role"] = roles[idx]
        cluster_df.loc[idx, "is_sampled"] = 1

    return cluster_df


def cap_sampled_cells(sampled_df: pd.DataFrame) -> pd.DataFrame:
    """
    Movies 行数较多，聚类初采样后再做全局/按列/按桶 cap，避免 LLM 请求过多。
    保留优先级：
    1. recall recovery / strong movie consistency / movie consistency
    2. hard column
    3. low margin
    4. center > boundary > all > outlier
    5. dist_to_center 小的优先
    """
    if sampled_df.empty:
        return sampled_df

    df = sampled_df.copy()

    role_priority = {
        "center": 0,
        "boundary": 1,
        "all": 2,
        "outlier": 3,
    }
    df["__role_priority__"] = df["sample_role"].map(role_priority).fillna(9).astype(int)
    df["__priority_score__"] = (
        2.0 * pd.to_numeric(df.get("allowed_recall_recovery_hardcase", 0), errors="coerce").fillna(0)
        + 1.5 * pd.to_numeric(df.get("allowed_is_strong_movie_consistency", 0), errors="coerce").fillna(0)
        + 1.2 * pd.to_numeric(df.get("movie_consistency_any", 0), errors="coerce").fillna(0)
        + 1.0 * pd.to_numeric(df.get("is_hard_column", 0), errors="coerce").fillna(0)
        + 0.8 * (1.0 - pd.to_numeric(df.get("top_margin", 0), errors="coerce").fillna(0).clip(lower=0, upper=1))
        + 0.4 * pd.to_numeric(df.get("is_high_risk", 0), errors="coerce").fillna(0)
    )

    # Per bucket cap
    parts = []
    for bucket_id, g in df.groupby("bucket_id", sort=False):
        g = g.sort_values(["__priority_score__", "__role_priority__", "dist_to_center"], ascending=[False, True, True])
        parts.append(g.head(MAX_SAMPLED_CELLS_PER_BUCKET))
    df = pd.concat(parts, ignore_index=True) if parts else df

    # Per column cap
    parts = []
    for col, g in df.groupby("column", sort=False):
        g = g.sort_values(["__priority_score__", "__role_priority__", "dist_to_center"], ascending=[False, True, True])
        parts.append(g.head(MAX_SAMPLED_CELLS_PER_COLUMN))
    df = pd.concat(parts, ignore_index=True) if parts else df

    # Global cap
    df = df.sort_values(["__priority_score__", "__role_priority__", "dist_to_center"], ascending=[False, True, True])
    if GLOBAL_MAX_SAMPLED_CELLS is not None and GLOBAL_MAX_SAMPLED_CELLS > 0:
        df = df.head(GLOBAL_MAX_SAMPLED_CELLS)

    df = df.drop(columns=["__role_priority__", "__priority_score__"], errors="ignore")
    df = df.sort_values(["row_id", "column"]).reset_index(drop=True)
    return df


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
    column = str(column)
    if column in MOVIE_TEXT_LIST_COLUMNS:
        return TEXT_LIST_COLUMN_MAX_CANDIDATES
    if column in MOVIE_HARD_COLUMNS:
        return HARD_COLUMN_MAX_CANDIDATES
    return DEFAULT_MAX_CANDIDATES_PER_CELL_FOR_LLM


CANDIDATE_FEATURE_EXPORT_COLS = [
    "candidate_value",
    "candidate_rank",
    "candidate_source",
    "candidate_source_text",
    "source_count",
    "source_score",
    "final_score_from_candidate_gen",
    "is_from_rule",
    "is_from_neighbor",
    "is_from_similarity",
    "is_from_hint",
    "is_from_dictionary",
    "is_from_pattern_restore",
    "is_rule_and_dictionary_supported",
    "contains_dictionary_source",
    "contains_movie_dictionary_source",
    "contains_canonical_source",
    "contains_movie_canonical_source",
    "contains_movie_consistency_source",
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
    "candidate_column_top_ratio",

    # Movies consistency
    "is_from_movie_consistency",
    "movie_consistency_name",
    "movie_consistency_lhs_key",
    "movie_consistency_confidence",
    "movie_consistency_margin",
    "movie_consistency_support_count",
    "movie_consistency_total_count",
    "candidate_equals_movie_consistency_value",
    "movie_consistency_strength",
    "has_movie_consistency_suspicious_record",

    # Allowed metadata
    "allowed_consensus_value",
    "allowed_consensus_confidence",
    "allowed_consensus_margin",
    "allowed_consensus_support_count",
    "allowed_consensus_total_count",
    "allowed_consensus_reason",
    "allowed_consistency_name",
    "allowed_lhs_key",
    "allowed_is_movie_metadata_consistency",
    "allowed_is_strong_movie_consistency",
    "candidate_equals_allowed_consensus_value",
    "candidate_and_allowed_both_movie_consistency",
    "candidate_under_strong_movie_consistency_cell",

    # Movies typed candidate features
    "movie_candidate_is_duration",
    "movie_dirty_is_duration",
    "movie_candidate_duration_restoration",
    "movie_candidate_duration_minutes",
    "movie_duration_abs_diff",
    "movie_candidate_duration_in_broad_range",
    "movie_candidate_duration_in_common_movie_range",
    "movie_candidate_duration_diff_to_row_duration",
    "movie_candidate_is_year",
    "movie_dirty_is_year",
    "movie_candidate_year_restoration",
    "movie_candidate_year_equals_release_year",
    "movie_candidate_year_release_gap",
    "movie_candidate_is_release_date",
    "movie_dirty_is_release_date",
    "movie_candidate_release_date_restoration",
    "movie_candidate_release_year",
    "movie_candidate_release_country",
    "movie_candidate_release_year_equals_row_year",
    "movie_candidate_is_country",
    "movie_dirty_is_country",
    "movie_candidate_country_restoration",
    "movie_candidate_country_equals_release_country",
    "movie_candidate_is_language",
    "movie_dirty_is_language",
    "movie_candidate_is_rating_value",
    "movie_dirty_is_rating_value",
    "movie_candidate_rating_value_restoration",
    "movie_candidate_rating_value_float",
    "movie_rating_value_abs_diff",
    "movie_candidate_rating_in_range",
    "movie_candidate_rating_diff_to_row_rating",
    "movie_candidate_is_count",
    "movie_dirty_is_count",
    "movie_candidate_count_restoration",
    "movie_candidate_count_int",
    "movie_count_abs_diff",
    "movie_candidate_count_nonnegative",
    "movie_candidate_count_diff_to_row_count",
    "movie_candidate_is_review_count",
    "movie_dirty_is_review_count",
    "movie_candidate_review_count_restoration",
    "movie_candidate_review_total",
    "movie_dirty_review_total",
    "movie_candidate_list_token_count",
    "movie_dirty_list_token_count",
    "movie_list_jaccard_to_dirty",
    "movie_candidate_contains_dirty_list",
    "movie_dirty_contains_candidate_list",
    "movie_mojibake_repair_gain",
    "movie_candidate_actors_overlap_with_cast",
    "movie_candidate_actors_is_cast_prefix",
    "movie_candidate_cast_contains_actors",

    # Movies column flags
    "movie_column_is_time",
    "movie_column_is_rating",
    "movie_column_is_people",
    "movie_column_is_list",
    "movie_column_is_text",
]


def build_candidate_list_for_cell(cell_key: Tuple[int, str], cand_df: pd.DataFrame) -> List[Dict[str, Any]]:
    row_id, column = cell_key
    g = cand_df[(cand_df["row_id"] == row_id) & (cand_df["column"] == column)].copy()

    score_col = None
    if "final_score_from_candidate_gen" in g.columns:
        score_col = "final_score_from_candidate_gen"
    elif "final_score" in g.columns:
        score_col = "final_score"
    elif "source_score" in g.columns:
        score_col = "source_score"

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

        # Always ensure candidate_value exists
        if "candidate_value" not in item and "candidate_value" in r.index:
            item["candidate_value"] = safe_jsonable(r.get("candidate_value"))

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
                    "detected_type": row.get("detected_type"),
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
                    "movie_field_bucket": row.get("movie_field_bucket"),
                    "is_hard_column": row.get("is_hard_column"),
                    "is_movie_time_col": row.get("is_movie_time_col"),
                    "is_movie_rating_col": row.get("is_movie_rating_col"),
                    "is_movie_people_col": row.get("is_movie_people_col"),
                    "is_movie_list_col": row.get("is_movie_list_col"),
                    "is_movie_text_col": row.get("is_movie_text_col"),
                    "movie_consistency_any": row.get("movie_consistency_any"),
                    "movie_consistency_strength_max": row.get("movie_consistency_strength_max"),
                    "movie_consistency_confidence_max": row.get("movie_consistency_confidence_max"),
                    "candidate_equals_allowed_consensus_any": row.get("candidate_equals_allowed_consensus_any"),
                    "under_strong_movie_consistency_cell": row.get("under_strong_movie_consistency_cell"),
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

        movie_bundle = row_ctx.get("movie_metadata_bundle", {})
        if isinstance(movie_bundle, dict):
            out["movie_metadata_bundle"] = movie_bundle

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
            "parsed_year": col_ctx.get("parsed_year"),
            "duration_minutes": col_ctx.get("duration_minutes"),
            "rating_value_float": col_ctx.get("rating_value_float"),
            "count_value_int": col_ctx.get("count_value_int"),
            "list_token_count": col_ctx.get("list_token_count"),
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
            "movie_typed_rule_count": conf_ctx.get("movie_typed_rule_count"),
            "movie_consistency_rule_count": conf_ctx.get("movie_consistency_rule_count"),
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
    column = str(column).strip()

    if column == "Duration":
        return (
            "本列是电影时长列。优先选择合法的时长格式，例如 '90 min'；"
            "如果候选来自 id_to_duration、name_year_to_duration、director_genre_to_duration、"
            "movie_metadata_consistency 或 canonical_duration_restore，要结合同行 Name/Year/Director/Genre/Rating 信息判断；"
            "不要仅因为 '90 min' 高频就选择它，必须有上下文或规则支持。"
        )

    if column == "Year":
        return (
            "本列是电影年份。优先选择 1800-2100 范围内的四位年份；"
            "重点看 Release Date 中的年份、Id/Name 对 Year 的一致性，以及规则 expected value。"
        )

    if column == "Release Date":
        return (
            "本列是上映日期。优先选择合法日期表达，例如 '29 August 2014 (France)'；"
            "要检查日期年份是否与 Year 一致，括号国家是否与 Country 或 Release Country 一致。"
        )

    if column == "Country":
        return (
            "本列是国家。优先选择与 Release Date 中括号国家、Name/Year 组合、电影语言等一致的候选；"
            "不要仅因为国家高频就替换。"
        )

    if column == "Language":
        return (
            "本列是语言。优先选择规范语言值，例如 English/French/Spanish；"
            "语言通常弱依赖 Country，但不能只由 Country 推断，除非有规则或 metadata consistency 支持。"
        )

    if column == "RatingValue":
        return (
            "本列是评分值。优先选择 0-10 范围内的一位小数评分；"
            "候选如果来自 id_to_rating_value、name_year_to_rating_value 或 canonical_rating_value_restore 更可信；"
            "不要把已经合法的评分无根据改成别的合法评分。"
        )

    if column == "RatingCount":
        return (
            "本列是评分人数/评分次数。优先选择非负整数；"
            "重点看 Id、Name+Year、RatingValue 的上下文一致性。"
        )

    if column == "ReviewCount":
        return (
            "本列是评论数量。候选可能是总数，也可能是 '1 user,2 critics' 格式；"
            "优先选择与 RatingCount、Genre、同行上下文一致的候选。"
        )

    if column in {"Actors", "Cast"}:
        return (
            "本列是人物列表。优先选择与 Name、Director、Creator、同行 Actors/Cast 列一致的列表；"
            "Actors 往往是 Cast 的子集或前若干关键演员，不能只根据列内高频选择。"
        )

    if column in {"Director", "Creator"}:
        return (
            "本列是创作者/导演。优先选择与电影 Name、Year、Actors/Cast、Genre 一致的人名；"
            "对空值修复要非常谨慎，需要强规则或 metadata consistency 支持。"
        )

    if column == "Genre":
        return (
            "本列是类型列表。优先保留合理的类别列表格式，判断候选是否与电影 Name/Director/Rating/Duration 上下文一致；"
            "不要仅根据列内高频类别修复。"
        )

    if column == "Filming Locations":
        return (
            "本列是拍摄地点。地点可能是城市/地区/国家组合；"
            "只在候选有明显上下文支持时选择，不要把具体地点简单改成高频国家。"
        )

    if column == "Name":
        return (
            "本列是电影名称。除非有 Id 或强 metadata consistency 支持，否则不要轻易选择高频电影名。"
        )

    return (
        "优先考虑规则一致性、同行电影元数据一致性、列类型合法性、统计/共现支持和合理的格式恢复。"
    )


def build_pairwise_prompt(context_obj: Dict[str, Any], cand_a: Dict[str, Any], cand_b: Dict[str, Any], column: str) -> str:
    column_instruction = build_column_specific_instruction(column)

    prompt = f"""
你现在在做 Movies 表格数据清洗中的错误修复排序任务。

目标：
对于一个被检测为错误的单元格，请在两个候选修复值 A 和 B 中判断谁更适合作为最终修复值。
你必须只在 A、B、Unknown 中三选一：
- A：候选 A 更好
- B：候选 B 更好
- Unknown：无法可靠判断，或者两个都不够好

通用判断原则：
1. 优先考虑规则一致性：FD、soft FD、expected_values_from_rules、movie_metadata_consistency 是否支持该候选
2. 再考虑同行电影元数据一致性：Name、Year、Release Date、Country、Duration、RatingValue、RatingCount、Actors/Cast 等是否互相匹配
3. 再考虑列类型合法性：年份、日期、时长、评分、计数、人物列表、国家/语言格式是否合理
4. 再考虑统计支持：列内高频、共现概率、邻居支持、similar row 支持
5. 再考虑变换合理性：格式恢复、拼写相似、键盘误触、发音相似、列表规范化
6. 如果两个候选都明显不合理，输出 Unknown
7. 不要仅因为候选更常见就无条件选择它；Movies 中高频国家、语言、90 min、USA 等候选容易造成误修

本列专属判断：
{column_instruction}

请严格输出 JSON，不要输出额外解释文字。
JSON 格式如下：
{{
  "winner": "A" 或 "B" 或 "Unknown",
  "reason_type": "rule_consistency" / "movie_metadata_consistency" / "typed_format" / "context_match" / "cooccurrence_support" / "spelling_similarity" / "pattern_match" / "uncertain" / "other",
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
                    {"role": "system", "content": "你是一个严谨的 Movies 表格数据清洗排序专家。你必须只输出 JSON。"},
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


def normalize_response_item(req: Dict[str, Any], llm_out: Dict[str, Any]) -> Dict[str, Any]:
    winner = str(llm_out.get("winner", "Unknown")).strip()
    if winner not in {"A", "B", "Unknown"}:
        winner = "Unknown"

    return {
        "request_id": req["request_id"],
        "row_id": req["row_id"],
        "column": req["column"],
        "dirty_value": req["dirty_value"],
        "candidate_a_index": req["candidate_a_index"],
        "candidate_b_index": req["candidate_b_index"],
        "candidate_a_value": req["candidate_a_value"],
        "candidate_b_value": req["candidate_b_value"],
        "winner": winner,
        "reason_type": llm_out.get("reason_type", "uncertain"),
        "reason": llm_out.get("reason", ""),
    }


def load_existing_responses(path: str) -> Dict[str, Dict[str, Any]]:
    existing = {}
    if not RESUME_FROM_EXISTING_RESPONSES or not os.path.exists(path):
        return existing

    with open(path, "r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                rid = obj.get("request_id", None)
                if rid:
                    existing[str(rid)] = obj
            except Exception as e:
                print(f"[WARN] 跳过已有 response 中无法解析的行: line={line_no}, err={e}")

    if existing:
        print(f"[INFO] loaded existing LLM responses for resume: {len(existing)}")
    return existing


def call_one_request(req: Dict[str, Any]) -> Dict[str, Any]:
    # 每个线程独立 client，避免共享连接对象带来的潜在并发问题。
    client = get_client()
    llm_out = call_llm_pairwise(client, req["pairwise_prompt"])
    return normalize_response_item(req, llm_out)


def run_llm_requests(requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    并发调用 LLM，并支持断点续跑。
    为了后续聚合稳定，最终返回顺序按 requests 原始顺序排列。
    """
    existing = load_existing_responses(OUTPUT_PAIRWISE_RESPONSES_JSONL)

    all_by_id = {}
    pending = []
    for req in requests:
        rid = str(req["request_id"])
        if rid in existing:
            all_by_id[rid] = existing[rid]
        else:
            pending.append(req)

    print(f"[INFO] existing responses reused: {len(all_by_id)}")
    print(f"[INFO] pending LLM requests: {len(pending)}")
    print(f"[INFO] LLM_MAX_WORKERS: {LLM_MAX_WORKERS}")

    if not pending:
        return [all_by_id[str(req["request_id"])] for req in requests if str(req["request_id"]) in all_by_id]

    write_lock = threading.Lock()
    done_count = 0

    # append 模式：已有结果不重写，新增结果边完成边落盘，防止中途断掉全丢。
    with open(OUTPUT_PAIRWISE_RESPONSES_JSONL, "a", encoding="utf-8") as f:
        if LLM_MAX_WORKERS <= 1:
            for req in pending:
                item = call_one_request(req)
                with write_lock:
                    write_jsonl_record(f, item)
                    f.flush()
                all_by_id[str(req["request_id"])] = item
                done_count += 1
                if done_count % LLM_PROGRESS_EVERY == 0:
                    print(f"[INFO] LLM progress: new_done={done_count}/{len(pending)}, total_done={len(all_by_id)}/{len(requests)}")
        else:
            with ThreadPoolExecutor(max_workers=LLM_MAX_WORKERS) as ex:
                future_to_req = {ex.submit(call_one_request, req): req for req in pending}
                for fut in as_completed(future_to_req):
                    req = future_to_req[fut]
                    try:
                        item = fut.result()
                    except Exception as e:
                        item = normalize_response_item(req, {
                            "winner": "Unknown",
                            "reason_type": "uncertain",
                            "reason": f"并发调用异常: {str(e)}"
                        })

                    with write_lock:
                        write_jsonl_record(f, item)
                        f.flush()

                    all_by_id[str(req["request_id"])] = item
                    done_count += 1
                    if done_count % LLM_PROGRESS_EVERY == 0:
                        print(f"[INFO] LLM progress: new_done={done_count}/{len(pending)}, total_done={len(all_by_id)}/{len(requests)}")

    # 返回时恢复 request 原始顺序，避免并发完成顺序影响后续 CSV 顺序。
    return [all_by_id[str(req["request_id"])] for req in requests if str(req["request_id"]) in all_by_id]


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
                "sampled_count_before_global_cap": int(cluster_sampled["is_sampled"].sum()),
                "semantic_type": str(cluster_df["semantic_type"].iloc[0]) if "semantic_type" in cluster_df.columns else "unknown",
                "detected_type": str(cluster_df["detected_type"].iloc[0]) if "detected_type" in cluster_df.columns else "unknown",
                "main_rule_type": str(cluster_df["main_rule_type"].iloc[0]) if "main_rule_type" in cluster_df.columns else "unknown",
                "main_usage_role": str(cluster_df["main_usage_role"].iloc[0]) if "main_usage_role" in cluster_df.columns else "unknown",
                "confidence_bucket": str(cluster_df["confidence_bucket"].iloc[0]) if "confidence_bucket" in cluster_df.columns else "unknown",
                "candidate_count_bucket": str(cluster_df["candidate_count_bucket"].iloc[0]) if "candidate_count_bucket" in cluster_df.columns else "unknown",
                "movie_field_bucket": str(cluster_df["movie_field_bucket"].iloc[0]) if "movie_field_bucket" in cluster_df.columns else "unknown",
                "is_hard_column": int(cluster_df["is_hard_column"].iloc[0]) if "is_hard_column" in cluster_df.columns else 0,
                "movie_consistency_any": int(cluster_df["movie_consistency_any"].iloc[0]) if "movie_consistency_any" in cluster_df.columns else 0,
            })

        clustered_bucket = pd.concat(bucket_clustered_parts, ignore_index=True)
        clustered_parts.append(clustered_bucket)

    if clustered_parts:
        clustered_df = pd.concat(clustered_parts, ignore_index=True)
    else:
        clustered_df = pd.DataFrame()

    clustered_df.to_csv(OUTPUT_CLUSTERED_CSV, index=False, encoding="utf-8-sig")

    sampled_df_initial = clustered_df[clustered_df["is_sampled"] == 1].copy() if not clustered_df.empty else pd.DataFrame()
    sampled_df = cap_sampled_cells(sampled_df_initial)
    sampled_df.to_csv(OUTPUT_SAMPLED_CSV, index=False, encoding="utf-8-sig")

    if not sampled_df.empty:
        build_sampled_jsonl(sampled_df, cand_df, detail_map)
    else:
        with open(OUTPUT_SAMPLED_JSONL, "w", encoding="utf-8") as f:
            pass

    bucket_summary_df = pd.DataFrame(bucket_summary_rows)
    if not bucket_summary_df.empty:
        if not sampled_df.empty:
            sampled_bucket_counts = sampled_df.groupby("bucket_id").size().reset_index(name="sampled_count_after_global_cap")
            bucket_summary_df = bucket_summary_df.merge(sampled_bucket_counts, on="bucket_id", how="left")
            bucket_summary_df["sampled_count_after_global_cap"] = bucket_summary_df["sampled_count_after_global_cap"].fillna(0).astype(int)
        else:
            bucket_summary_df["sampled_count_after_global_cap"] = 0

    bucket_summary_df.to_csv(
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
    print(f"[INFO] 初始采样 cell 数: {len(sampled_df_initial)}")
    print(f"[INFO] cap 后采样 cell 数: {len(sampled_df)}")
    print(f"[INFO] pairwise 请求数: {len(requests)}")
    print(f"[INFO] PAIRWISE_STRATEGY: {PAIRWISE_STRATEGY}")
    print(f"[INFO] GLOBAL_MAX_SAMPLED_CELLS: {GLOBAL_MAX_SAMPLED_CELLS}")
    print(f"[INFO] MAX_SAMPLED_CELLS_PER_COLUMN: {MAX_SAMPLED_CELLS_PER_COLUMN}")
    print(f"[INFO] MAX_SAMPLED_CELLS_PER_BUCKET: {MAX_SAMPLED_CELLS_PER_BUCKET}")

    if DRY_RUN_ONLY_EXPORT_REQUESTS:
        print("[INFO] 当前为 DRY RUN，仅导出请求，不调用 LLM。")
        return

    # 4) 调用 LLM
    responses = run_llm_requests(requests)

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

        print("\nMovies 字段类型采样分布:")
        if "movie_field_bucket" in sampled_df.columns:
            print(sampled_df["movie_field_bucket"].value_counts(dropna=False).to_string())

        print("\nMovies consistency 采样分布:")
        if "movie_consistency_any" in sampled_df.columns:
            print(sampled_df["movie_consistency_any"].value_counts(dropna=False).to_string())

        print("\n采样列分布:")
        print(sampled_df["column"].value_counts(dropna=False).head(30).to_string())


if __name__ == "__main__":
    main()

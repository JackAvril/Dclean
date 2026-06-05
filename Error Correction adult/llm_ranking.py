
"""
Adult LLM pairwise ranking for repair candidates.

Adapted from the Flights LLM ranking script, but:
1. Uses Adult candidate feature file by default.
2. Removes Flights-specific time/source consensus instructions.
3. Adds Adult-specific sampling features:
   relationship consistency, sex/marital conflicts, education-age conflict,
   domain validity, Adult profile distribution, Adult v2 signals.
4. Adds Adult-specific pairwise prompt instructions.
5. Keeps clustering/sampling/pairwise/LLM/resume/teacher aggregation workflow.

Inputs
------
repair_candidate_features_infer_whitelist.csv
candidate_llm_contexts_adult.jsonl

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

Usage
-----
cd /mnt/mydata/dq/projects/Splittree/test/Error\ Correction\ adult

# Only export requests first:
python llm_ranking_adult.py --dry_run_only_export_requests 1

# Actually call DeepSeek/OpenAI-compatible API:
export DEEPSEEK_API_KEY="你的key"
python llm_ranking_adult.py --dry_run_only_export_requests 0 --llm_max_workers 4

Notes
-----
Do not hardcode API keys in this file. Use environment variables:
  DEEPSEEK_API_KEY
  DEEPSEEK_BASE_URL
  DEEPSEEK_MODEL
"""

import argparse
import json
import math
import os
import re
import time
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# ============================================================
# 1. Defaults
# ============================================================

DEFAULT_INPUT_CANDIDATE_FEATURES_CSV = "repair_candidate_features_infer_whitelist.csv"
DEFAULT_INPUT_DETAIL_JSONL = "/mnt/mydata/dq/projects/Splittree/test/Error Detection Adult/candidate_llm_contexts_adult.jsonl"

DEFAULT_OUTPUT_CELL_SUMMARY_CSV = "repair_cell_summary_for_sampling_v2.csv"
DEFAULT_OUTPUT_CLUSTERED_CSV = "repair_cell_clustered_v2.csv"
DEFAULT_OUTPUT_SAMPLED_CSV = "repair_cell_sampled_v2.csv"
DEFAULT_OUTPUT_SAMPLED_JSONL = "repair_cell_sampled_with_candidates_v2.jsonl"

DEFAULT_OUTPUT_PAIRWISE_REQUESTS_JSONL = "repair_pairwise_requests_v2.jsonl"
DEFAULT_OUTPUT_PAIRWISE_RESPONSES_JSONL = "repair_pairwise_responses_v2.jsonl"
DEFAULT_OUTPUT_TEACHER_RANKING_CSV = "repair_teacher_pairwise_ranking_v2.csv"
DEFAULT_OUTPUT_TEACHER_TOP1_CSV = "repair_teacher_top1_labels_v2.csv"
DEFAULT_OUTPUT_BUCKET_SUMMARY_CSV = "repair_bucket_cluster_summary_v2.csv"

# Adult sampling defaults.
DEFAULT_USE_COLUMN_IN_BUCKET = True
DEFAULT_MIN_BUCKET_SIZE_FOR_CLUSTER = 6
DEFAULT_MAX_CLUSTERS_PER_BUCKET = 8
DEFAULT_TARGET_CLUSTER_SIZE = 12
DEFAULT_SMALL_BUCKET_FULL_SAMPLE_THRESHOLD = 5

DEFAULT_SAMPLE_CENTER = True
DEFAULT_SAMPLE_BOUNDARY = True
DEFAULT_SAMPLE_OUTLIER = True
DEFAULT_RANDOM_STATE = 42

DEFAULT_MAX_CANDIDATES_PER_CELL_FOR_LLM = 6
HARD_COLUMN_MAX_CANDIDATES = 8

# Adult hard / semantically risky columns.
ADULT_HARD_COLUMNS = {
    "relationship", "sex", "maritalstatus", "education",
    "age", "hoursperweek", "income", "workclass", "occupation",
}
ADULT_PRIMARY_REPAIR_COLUMNS = {
    "relationship", "sex", "maritalstatus", "education", "age", "hoursperweek", "income",
}

PAIRWISE_STRATEGY = "one_vs_top"   # all_pairs / one_vs_top
UNKNOWN_CANDIDATE_NAME = "Unknown"
DROP_UNKNOWN_IF_TOO_MANY = False

# LLM defaults.
MODEL_NAME = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-fd93f5216ae0488f8bf32d9e2a14a14a")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
TEMPERATURE = 0.0
MAX_RETRIES = 3
RETRY_SLEEP_SECONDS = 2.0
LLM_MAX_WORKERS = int(os.getenv("LLM_MAX_WORKERS", "4"))
LLM_PROGRESS_EVERY = int(os.getenv("LLM_PROGRESS_EVERY", "20"))
RESUME_FROM_EXISTING_RESPONSES = True


# ============================================================
# 2. Basic tools
# ============================================================

def write_jsonl_record(fp, obj):
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")


def resolve_output_path(path: str) -> str:
    if os.path.isabs(str(path)):
        return str(path)
    return os.path.abspath(str(path))


def ensure_parent(path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


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
    if s in {"", "nan", "none", "null", "na", "n/a", "missing", "empty", "?"}:
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


def normalize_col_name(column: Any) -> str:
    c = norm_text(column)
    aliases = {
        "marital-status": "maritalstatus",
        "marital_status": "maritalstatus",
        "marital status": "maritalstatus",
        "hours-per-week": "hoursperweek",
        "hours_per_week": "hoursperweek",
        "hours per week": "hoursperweek",
        "native-country": "country",
        "native_country": "country",
        "capital-gain": "capital_gain",
        "capital-loss": "capital_loss",
        "education-num": "education_num",
        "education_num": "education_num",
    }
    return aliases.get(c, c)


def bool_from_int(x) -> int:
    return 1 if safe_int(x, 0) != 0 else 0


# ============================================================
# 3. Detail jsonl
# ============================================================

def load_detail_jsonl(path: Optional[str]) -> Dict[Tuple[int, str], Dict[str, Any]]:
    detail_map = {}
    if path is None or str(path).strip() == "" or not os.path.exists(path):
        print(f"[WARN] detail jsonl not found: {path}")
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
# 4. Cell-level summary from candidate-level features
# ============================================================

def first_non_null(series: pd.Series, default=None):
    for x in series:
        if pd.notna(x):
            return x
    return default

def first_existing_non_null(g: pd.DataFrame, cols: List[str], default=None):
    """
    Safely read first non-null value from the first existing column.
    Adult whitelist fields are often prefixed as allowed_*.
    """
    for col in cols:
        if col in g.columns:
            return first_non_null(g[col], default)
    return default


def get_numeric_series(g: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in g.columns:
        return pd.to_numeric(g[col], errors="coerce").fillna(default)
    return pd.Series([default] * len(g), index=g.index)


def get_score_col(g: pd.DataFrame) -> Optional[str]:
    for c in [
        "final_score_from_candidate_gen",
        "final_score",
        "source_score",
        "candidate_score",
    ]:
        if c in g.columns:
            return c
    return None


def sort_candidate_group(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    score_col = get_score_col(g)
    if score_col:
        if "candidate_rank" in g.columns:
            return g.sort_values([score_col, "candidate_rank"], ascending=[False, True])
        return g.sort_values([score_col], ascending=[False])
    if "candidate_rank" in g.columns:
        return g.sort_values(["candidate_rank"], ascending=[True])
    return g


def build_cell_summary_df(cand_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = cand_df.groupby(["row_id", "column"], sort=False)

    for (row_id, column), g in grouped:
        g = sort_candidate_group(g)
        score_col = get_score_col(g)

        dirty_value = first_non_null(g["dirty_value"], "") if "dirty_value" in g.columns else ""
        semantic_type = first_non_null(g["semantic_type"], "unknown") if "semantic_type" in g.columns else "unknown"
        main_rule_type = first_non_null(g["main_rule_type"], "unknown") if "main_rule_type" in g.columns else "unknown"
        main_usage_role = first_non_null(g["main_usage_role"], "unknown") if "main_usage_role" in g.columns else "unknown"

        violation_count = safe_float(first_non_null(g["violation_count"], 0), 0)
        conflict_score = safe_float(first_non_null(g["conflict_score"], 0), 0)

        candidate_count = len(g)
        top1_score = float(g[score_col].iloc[0]) if score_col and len(g) >= 1 else 0.0
        top2_score = float(g[score_col].iloc[1]) if score_col and len(g) >= 2 else 0.0
        top_margin = top1_score - top2_score if candidate_count >= 2 else top1_score

        rule_support_s = get_numeric_series(g, "candidate_rule_support_ratio", 0.0)
        rule_gain_s = get_numeric_series(g, "rule_support_gain", 0.0)
        bayes_s = get_numeric_series(g, "bayes_mean_cond_prob", 0.0)
        edit_sim_s = get_numeric_series(g, "edit_similarity_to_dirty", 0.0)

        same_profile_gain_s = get_numeric_series(g, "same_profile_ratio_gain", 0.0)
        work_profile_gain_s = get_numeric_series(g, "work_profile_ratio_gain", 0.0)
        relationship_gain_s = get_numeric_series(g, "relationship_consistency_gain", 0.0)
        domain_gain_s = get_numeric_series(g, "candidate_domain_validity_gain", 0.0)
        education_gain_s = get_numeric_series(g, "education_age_conflict_reduction", 0.0)

        source_count_top1 = safe_int(g["source_count"].iloc[0], 0) if "source_count" in g.columns and len(g) >= 1 else 0

        top1_is_from_rule = safe_int(g["is_from_rule"].iloc[0], 0) if "is_from_rule" in g.columns and len(g) >= 1 else 0
        top1_is_from_neighbor = safe_int(g["is_from_neighbor"].iloc[0], 0) if "is_from_neighbor" in g.columns and len(g) >= 1 else 0
        top1_is_from_similarity = safe_int(g["is_from_similarity"].iloc[0], 0) if "is_from_similarity" in g.columns and len(g) >= 1 else 0
        top1_is_from_hint = safe_int(g["is_from_hint"].iloc[0], 0) if "is_from_hint" in g.columns and len(g) >= 1 else 0
        top1_is_from_dictionary = safe_int(g["is_from_dictionary"].iloc[0], 0) if "is_from_dictionary" in g.columns and len(g) >= 1 else 0
        top1_is_from_adult_profile = safe_int(g["is_from_adult_profile"].iloc[0], 0) if "is_from_adult_profile" in g.columns and len(g) >= 1 else 0
        top1_is_from_adult_direct_rule = safe_int(g["is_from_adult_direct_rule"].iloc[0], 0) if "is_from_adult_direct_rule" in g.columns and len(g) >= 1 else 0
        top1_is_from_domain = safe_int(g["is_from_domain"].iloc[0], 0) if "is_from_domain" in g.columns and len(g) >= 1 else 0

        final_pred_error_prob = safe_float(
            first_existing_non_null(g, ["final_pred_error_prob", "allowed_final_pred_error_prob"], 0.0),
            0.0,
        )
        is_high_risk = safe_int(
            first_existing_non_null(g, ["is_high_risk", "allowed_is_high_risk"], 0),
            0,
        )
        is_hard_case_from_detection = safe_int(
            first_existing_non_null(g, ["is_hard_case", "allowed_is_hard_case"], 0),
            0,
        )

        target_is_primary = safe_int(
            first_existing_non_null(g, ["target_is_primary_column", "allowed_target_is_primary_column"], 0),
            0,
        )
        target_is_context_only = safe_int(
            first_existing_non_null(g, ["target_is_context_only_column", "allowed_target_is_context_only_column"], 0),
            0,
        )

        adult_consistency_score = safe_float(
            first_existing_non_null(g, ["adult_consistency_score", "allowed_adult_consistency_score"], 0.0),
            0.0,
        )
        adult_evidence_flag_count = safe_int(first_non_null(g["adult_evidence_flag_count"], 0), 0) if "adult_evidence_flag_count" in g.columns else 0

        relationship_marital_conflict = max(
            get_numeric_series(g, "relationship_marital_conflict", 0).max(),
            get_numeric_series(g, "allowed_relationship_marital_conflict", 0).max(),
        )
        relationship_age_conflict = max(
            get_numeric_series(g, "relationship_age_conflict", 0).max(),
            get_numeric_series(g, "allowed_relationship_age_conflict", 0).max(),
        )
        relationship_sex_conflict = max(
            get_numeric_series(g, "relationship_sex_conflict", 0).max(),
            get_numeric_series(g, "allowed_relationship_sex_conflict", 0).max(),
        )
        sex_value_invalid = max(
            get_numeric_series(g, "sex_value_invalid", 0).max(),
            get_numeric_series(g, "allowed_sex_value_invalid", 0).max(),
        )
        education_age_extreme_conflict = max(
            get_numeric_series(g, "education_age_extreme_conflict", 0).max(),
            get_numeric_series(g, "allowed_education_age_extreme_conflict", 0).max(),
        )

        col_norm = normalize_col_name(column)
        is_hard_column = int(col_norm in ADULT_HARD_COLUMNS or target_is_primary == 1)

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

        adult_conflict_bucket = []
        if relationship_marital_conflict:
            adult_conflict_bucket.append("rel_marital")
        if relationship_age_conflict:
            adult_conflict_bucket.append("rel_age")
        if relationship_sex_conflict:
            adult_conflict_bucket.append("rel_sex")
        if sex_value_invalid:
            adult_conflict_bucket.append("sex_invalid")
        if education_age_extreme_conflict:
            adult_conflict_bucket.append("edu_age")
        adult_conflict_bucket = "+".join(adult_conflict_bucket) if adult_conflict_bucket else "none"

        rows.append({
            "row_id": int(row_id),
            "column": str(column),
            "dirty_value": dirty_value,
            "semantic_type": semantic_type,
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
            "max_rule_support_gain": float(rule_gain_s.max()),
            "avg_bayes": float(bayes_s.mean()),
            "max_bayes": float(bayes_s.max()),
            "avg_edit_sim": float(edit_sim_s.mean()),
            "max_edit_sim": float(edit_sim_s.max()),

            "avg_same_profile_gain": float(same_profile_gain_s.mean()),
            "max_same_profile_gain": float(same_profile_gain_s.max()),
            "avg_work_profile_gain": float(work_profile_gain_s.mean()),
            "max_work_profile_gain": float(work_profile_gain_s.max()),
            "avg_relationship_gain": float(relationship_gain_s.mean()),
            "max_relationship_gain": float(relationship_gain_s.max()),
            "avg_domain_gain": float(domain_gain_s.mean()),
            "max_domain_gain": float(domain_gain_s.max()),
            "avg_education_age_gain": float(education_gain_s.mean()),
            "max_education_age_gain": float(education_gain_s.max()),

            "source_count_top1": int(source_count_top1),
            "top1_is_from_rule": int(top1_is_from_rule),
            "top1_is_from_neighbor": int(top1_is_from_neighbor),
            "top1_is_from_similarity": int(top1_is_from_similarity),
            "top1_is_from_hint": int(top1_is_from_hint),
            "top1_is_from_dictionary": int(top1_is_from_dictionary),
            "top1_is_from_adult_profile": int(top1_is_from_adult_profile),
            "top1_is_from_adult_direct_rule": int(top1_is_from_adult_direct_rule),
            "top1_is_from_domain": int(top1_is_from_domain),

            "final_pred_error_prob": float(final_pred_error_prob),
            "is_high_risk": int(is_high_risk),
            "is_hard_case_from_detection": int(is_hard_case_from_detection),

            "adult_consistency_score": float(adult_consistency_score),
            "adult_evidence_flag_count": int(adult_evidence_flag_count),
            "target_is_primary_column": int(target_is_primary),
            "target_is_context_only_column": int(target_is_context_only),

            "relationship_marital_conflict": int(relationship_marital_conflict),
            "relationship_age_conflict": int(relationship_age_conflict),
            "relationship_sex_conflict": int(relationship_sex_conflict),
            "sex_value_invalid": int(sex_value_invalid),
            "education_age_extreme_conflict": int(education_age_extreme_conflict),
            "adult_conflict_bucket": adult_conflict_bucket,

            "confidence_bucket": confidence_bucket,
            "candidate_count_bucket": candidate_count_bucket,
            "is_hard_column": int(is_hard_column),
        })

    return pd.DataFrame(rows)


# ============================================================
# 5. Buckets
# ============================================================

def get_bucket_keys(use_column_in_bucket: bool) -> List[str]:
    keys = [
        "semantic_type",
        "main_rule_type",
        "main_usage_role",
        "confidence_bucket",
        "candidate_count_bucket",
        "is_hard_column",
        "adult_conflict_bucket",
        "target_is_primary_column",
    ]
    if use_column_in_bucket:
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


def assign_buckets(df: pd.DataFrame, use_column_in_bucket: bool) -> pd.DataFrame:
    df = df.copy()
    keys = get_bucket_keys(use_column_in_bucket)
    df["bucket_id"] = df.apply(lambda r: build_bucket_id(r, keys), axis=1)
    return df


# ============================================================
# 6. Clustering features
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
    "max_rule_support_gain",
    "avg_bayes",
    "max_bayes",
    "avg_edit_sim",
    "max_edit_sim",
    "avg_same_profile_gain",
    "max_same_profile_gain",
    "avg_work_profile_gain",
    "max_work_profile_gain",
    "avg_relationship_gain",
    "max_relationship_gain",
    "avg_domain_gain",
    "max_domain_gain",
    "avg_education_age_gain",
    "max_education_age_gain",
    "source_count_top1",
    "top1_is_from_rule",
    "top1_is_from_neighbor",
    "top1_is_from_similarity",
    "top1_is_from_hint",
    "top1_is_from_dictionary",
    "top1_is_from_adult_profile",
    "top1_is_from_adult_direct_rule",
    "top1_is_from_domain",
    "final_pred_error_prob",
    "is_high_risk",
    "is_hard_case_from_detection",
    "adult_consistency_score",
    "adult_evidence_flag_count",
    "target_is_primary_column",
    "target_is_context_only_column",
    "relationship_marital_conflict",
    "relationship_age_conflict",
    "relationship_sex_conflict",
    "sex_value_invalid",
    "education_age_extreme_conflict",
    "is_hard_column",
]

FEATURE_COLUMNS_CATEGORICAL = [
    "semantic_type",
    "main_rule_type",
    "main_usage_role",
    "confidence_bucket",
    "candidate_count_bucket",
    "adult_conflict_bucket",
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


def choose_k(n: int, min_bucket_size: int, target_cluster_size: int, max_clusters: int) -> int:
    if n < min_bucket_size:
        return 1
    k = max(2, round(n / target_cluster_size))
    k = min(k, max_clusters)
    k = min(k, n)
    return k


def cluster_bucket(bucket_df: pd.DataFrame, args) -> pd.DataFrame:
    bucket_df = bucket_df.copy()
    n = len(bucket_df)

    if n <= 1:
        bucket_df["cluster_id"] = 0
        bucket_df["dist_to_center"] = 0.0
        return bucket_df

    X, _ = prepare_features(bucket_df)
    k = choose_k(n, args.min_bucket_size_for_cluster, args.target_cluster_size, args.max_clusters_per_bucket)

    if k <= 1:
        bucket_df["cluster_id"] = 0
        center = X.mean(axis=0, keepdims=True)
        dist = np.linalg.norm(X - center, axis=1)
        bucket_df["dist_to_center"] = dist
        return bucket_df

    model = KMeans(n_clusters=k, random_state=args.random_state, n_init=10)
    labels = model.fit_predict(X)
    centers = model.cluster_centers_

    dists = np.zeros(len(X), dtype=float)
    for i in range(len(X)):
        dists[i] = np.linalg.norm(X[i] - centers[labels[i]])

    bucket_df["cluster_id"] = labels
    bucket_df["dist_to_center"] = dists
    return bucket_df


def sample_from_cluster(cluster_df: pd.DataFrame, args) -> pd.DataFrame:
    cluster_df = cluster_df.copy().sort_values("dist_to_center").reset_index(drop=True)
    n = len(cluster_df)

    if n <= args.small_bucket_full_sample_threshold:
        cluster_df["sample_role"] = "all"
        cluster_df["is_sampled"] = 1
        return cluster_df

    sampled_indices = []
    roles = {}

    if args.sample_center:
        center_idx = 0
        sampled_indices.append(center_idx)
        roles[center_idx] = "center"

    if args.sample_outlier:
        outlier_idx = n - 1
        if outlier_idx not in sampled_indices:
            sampled_indices.append(outlier_idx)
            roles[outlier_idx] = "outlier"

    if args.sample_boundary:
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
# 7. Candidate list for LLM
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


def get_max_candidates_for_column(column: str, args) -> int:
    return args.hard_column_max_candidates if normalize_col_name(column) in ADULT_HARD_COLUMNS else args.default_max_candidates_per_cell_for_llm


CANDIDATE_FEATURE_EXPORT_COLS = [
    "candidate_value",
    "candidate_rank",
    "candidate_source",
    "candidate_sources",
    "candidate_source_text",
    "source_count",
    "source_score",
    "is_multi_source_candidate",

    "is_from_rule",
    "is_from_neighbor",
    "is_from_similarity",
    "is_from_hint",
    "is_from_dictionary",
    "is_from_pattern_restore",
    "is_from_adult_profile",
    "is_from_adult_direct_rule",
    "is_from_domain",

    "contains_rule_expected_source",
    "contains_same_profile_source",
    "contains_work_profile_source",
    "contains_adult_profile_source",
    "contains_relationship_source",
    "contains_sex_source",
    "contains_maritalstatus_source",
    "contains_education_source",
    "contains_age_source",
    "contains_hours_source",
    "contains_income_source",
    "contains_domain_source",
    "contains_neighbor_source",
    "contains_column_top_source",
    "contains_hint_source",
    "is_weak_frequency_only_candidate",

    "candidate_rule_support_ratio",
    "rule_support_gain",
    "candidate_matches_any_rule_expected",
    "candidate_matches_any_strong_rule_expected",
    "candidate_matches_any_expected_values_from_rules",
    "candidate_equals_neighbor_majority",
    "candidate_equals_suggested_correct_value",
    "candidate_matches_similar_row_target",
    "candidate_match_ratio_in_similar_rows",

    "bayes_mean_cond_prob",
    "bayes_max_cond_prob",
    "bayes_prod_logprob",
    "edit_similarity_to_dirty",
    "avg_keyboard_distance",
    "phonetic_similarity_to_dirty",
    "candidate_in_column_top_values",
    "candidate_column_top_ratio",

    "candidate_in_adult_domain",
    "dirty_in_adult_domain",
    "candidate_domain_validity_gain",
    "candidate_equals_domain_canonical_dirty",

    "candidate_relationship_consistency_score",
    "dirty_relationship_consistency_score",
    "relationship_consistency_gain",
    "candidate_education_age_conflict",
    "dirty_education_age_conflict",
    "education_age_conflict_reduction",
    "candidate_is_spouse_relationship",
    "dirty_is_spouse_relationship",
    "candidate_spouse_gender_match",
    "dirty_spouse_gender_match",
    "candidate_spouse_marital_match",
    "dirty_spouse_marital_match",

    "same_profile_group_size",
    "candidate_same_profile_count",
    "dirty_same_profile_count",
    "candidate_same_profile_ratio",
    "dirty_same_profile_ratio",
    "candidate_same_profile_rank",
    "dirty_same_profile_rank",
    "same_profile_ratio_gain",
    "work_profile_group_size",
    "candidate_work_profile_count",
    "dirty_work_profile_count",
    "candidate_work_profile_ratio",
    "dirty_work_profile_ratio",
    "candidate_work_profile_rank",
    "dirty_work_profile_rank",
    "work_profile_ratio_gain",

    "adult_consistency_score",
    "adult_evidence_flags",
    "adult_evidence_flag_count",
    "target_is_primary_column",
    "target_is_context_only_column",
    "relationship_marital_conflict",
    "relationship_age_conflict",
    "relationship_sex_conflict",
    "sex_value_invalid",
    "education_age_extreme_conflict",
    "has_relationship_v2_rule",
    "has_education_v2_rule",
]


def build_candidate_list_for_cell(cell_key: Tuple[int, str], cand_df: pd.DataFrame, args) -> List[Dict[str, Any]]:
    row_id, column = cell_key
    g = cand_df[(cand_df["row_id"] == row_id) & (cand_df["column"] == column)].copy()
    g = sort_candidate_group(g)

    limit = get_max_candidates_for_column(column, args)

    if DROP_UNKNOWN_IF_TOO_MANY and len(g) > limit and "candidate_value" in g.columns:
        g = g[g["candidate_value"].astype(str) != UNKNOWN_CANDIDATE_NAME]

    g = g.head(limit).copy()

    score_col = get_score_col(g)
    candidate_rows = []
    for _, r in g.iterrows():
        item = {}
        for col in CANDIDATE_FEATURE_EXPORT_COLS:
            if col in r.index:
                item[col] = safe_jsonable(r.get(col))

        if score_col:
            item["candidate_score"] = safe_jsonable(r.get(score_col))
        else:
            item["candidate_score"] = safe_jsonable(r.get("source_score", None))

        if "candidate_value" not in item and "candidate_value" in r.index:
            item["candidate_value"] = safe_jsonable(r.get("candidate_value"))

        candidate_rows.append(item)

    return candidate_rows


def build_sampled_jsonl(sampled_df: pd.DataFrame, cand_df: pd.DataFrame, detail_map: Dict[Tuple[int, str], Dict[str, Any]], output_path: str, args):
    with open(output_path, "w", encoding="utf-8") as f:
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
                    "adult_conflict_bucket": row.get("adult_conflict_bucket"),
                    "adult_consistency_score": row.get("adult_consistency_score"),
                    "adult_evidence_flag_count": row.get("adult_evidence_flag_count"),
                    "target_is_primary_column": row.get("target_is_primary_column"),
                    "relationship_marital_conflict": row.get("relationship_marital_conflict"),
                    "relationship_age_conflict": row.get("relationship_age_conflict"),
                    "relationship_sex_conflict": row.get("relationship_sex_conflict"),
                    "sex_value_invalid": row.get("sex_value_invalid"),
                    "education_age_extreme_conflict": row.get("education_age_extreme_conflict"),
                    "final_pred_error_prob": row.get("final_pred_error_prob"),
                    "is_high_risk": row.get("is_high_risk"),
                    "is_hard_case_from_detection": row.get("is_hard_case_from_detection"),
                },
                "candidates": build_candidate_list_for_cell(key, cand_df, args),
                "detail_context": detail_obj,
            }
            write_jsonl_record(f, out)


# ============================================================
# 8. Pairwise request construction
# ============================================================

def make_pair_ids(candidates: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    n = len(candidates)
    if n <= 1:
        return []

    if PAIRWISE_STRATEGY == "one_vs_top":
        return [(0, j) for j in range(1, n)]

    return list(combinations(range(n), 2))


def compress_row_values(row_values: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(row_values, dict):
        return {}

    keep = [
        "age", "workclass", "education", "maritalstatus", "marital-status",
        "occupation", "relationship", "race", "sex", "hoursperweek",
        "hours-per-week", "country", "native-country", "income"
    ]
    out = {}
    for k, v in row_values.items():
        if normalize_col_name(k) in {normalize_col_name(x) for x in keep}:
            out[k] = v
    return out


def build_context_for_prompt(detail_context: Dict[str, Any], dirty_value: str, column: str, cell_row: Optional[pd.Series] = None) -> Dict[str, Any]:
    out = {
        "column": column,
        "dirty_value": dirty_value,
    }

    if isinstance(detail_context, dict):
        row_ctx = detail_context.get("row_context", {})
        if isinstance(row_ctx, dict):
            row_values = row_ctx.get("row_values", {})
            if isinstance(row_values, dict):
                out["row_values"] = compress_row_values(row_values)

        col_ctx = detail_context.get("column_context", {})
        if isinstance(col_ctx, dict):
            out["column_context"] = {
                "semantic_type": col_ctx.get("semantic_type"),
                "detected_type": col_ctx.get("detected_type"),
                "top_values_with_counts": col_ctx.get("top_values_with_counts"),
                "dominant_pattern": col_ctx.get("dominant_pattern"),
                "current_value_pattern": col_ctx.get("current_value_pattern"),
                "numeric_stats": col_ctx.get("numeric_stats"),
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
                "adult_domain_rule_count": conf_ctx.get("adult_domain_rule_count"),
                "adult_format_rule_count": conf_ctx.get("adult_format_rule_count"),
                "adult_consistency_rule_count": conf_ctx.get("adult_consistency_rule_count"),
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

        adult_ctx = detail_context.get("adult_consistency_context", {})
        if isinstance(adult_ctx, dict):
            bundle = adult_ctx.get("adult_profile_bundle", {})
            out["adult_consistency_context"] = {
                "adult_profile_bundle": bundle,
                "adult_profile_inconsistencies": bundle.get("adult_profile_inconsistencies", []) if isinstance(bundle, dict) else [],
                "evidence_flags": adult_ctx.get("evidence_flags", []),
                "same_profile_distribution": adult_ctx.get("same_profile_distribution", {}),
                "work_profile_distribution": adult_ctx.get("work_profile_distribution", {}),
                "adult_consistency_score": adult_ctx.get("adult_consistency_score"),
            }

        adult_v2 = detail_context.get("adult_v2_attribution_features", {})
        if isinstance(adult_v2, dict):
            out["adult_v2_attribution_features"] = adult_v2

        bayes_ctx = detail_context.get("bayesian_context", {})
        if isinstance(bayes_ctx, dict):
            out["bayesian_context"] = bayes_ctx

    # Fallback with cell-level summary fields.
    if cell_row is not None:
        out["cell_summary_fallback"] = {
            "semantic_type": cell_row.get("semantic_type"),
            "main_rule_type": cell_row.get("main_rule_type"),
            "main_usage_role": cell_row.get("main_usage_role"),
            "adult_conflict_bucket": cell_row.get("adult_conflict_bucket"),
            "adult_consistency_score": cell_row.get("adult_consistency_score"),
            "relationship_marital_conflict": cell_row.get("relationship_marital_conflict"),
            "relationship_age_conflict": cell_row.get("relationship_age_conflict"),
            "relationship_sex_conflict": cell_row.get("relationship_sex_conflict"),
            "sex_value_invalid": cell_row.get("sex_value_invalid"),
            "education_age_extreme_conflict": cell_row.get("education_age_extreme_conflict"),
        }

    return out


def build_column_specific_instruction(column: str) -> str:
    col = normalize_col_name(column)

    if col == "relationship":
        return (
            "本列是 Adult 数据集的 relationship。优先选择与 sex、maritalstatus、age 一致的候选。"
            "例如 Husband 通常应与 Male 和已婚状态一致；Wife 通常应与 Female 和已婚状态一致；"
            "Own-child 通常更适合年轻或未婚上下文。不要仅因为列内高频选择 Husband。"
        )

    if col == "sex":
        return (
            "本列是 sex。优先选择与 relationship 一致的候选：Husband->Male，Wife->Female。"
            "如果 relationship 不提供强约束，则不要仅凭列内高频修改。"
        )

    if col == "maritalstatus":
        return (
            "本列是 maritalstatus。若 relationship 是 Husband/Wife，通常应选择 Married-civ-spouse 或 Married-AF-spouse；"
            "若 relationship 是 Own-child/Not-in-family/Unmarried，要结合 age 和上下文判断。"
        )

    if col == "education":
        return (
            "本列是 education。优先考虑 education 与 age 的合理性，以及是否属于 Adult 合法教育枚举。"
            "若非常年轻却出现 Masters/Doctorate/Prof-school 等，要谨慎判断是否应修为更合理学历。"
        )

    if col == "age":
        return (
            "本列是 age。优先选择合法年龄或合理年龄桶修复。不要仅根据 relationship 过度推断具体年龄；"
            "除非规则证据或上下文强支持，否则保持 Unknown 更安全。"
        )

    if col == "hoursperweek":
        return (
            "本列是 hoursperweek。优先选择合法数值/范围规范化候选。"
            "不要仅因为 work profile 高频就随意改动工作时长。"
        )

    if col == "income":
        return (
            "本列是 income。优先做规范化：<=50K/LessThan50K，>50K/MoreThan50K。"
            "不要仅凭 occupation/education 高频预测收入，除非规则或上下文强支持。"
        )

    if col in {"workclass", "occupation", "race", "country"}:
        return (
            "本列是 Adult 枚举字段。优先选择合法 domain 值和规则明确支持的候选；"
            "不要仅因列内高频或相似行多数就修改。"
        )

    return (
        "优先考虑规则一致性、Adult profile 一致性、domain 合法性、同行上下文一致性和候选来源可靠性。"
    )


def build_pairwise_prompt(context_obj: Dict[str, Any], cand_a: Dict[str, Any], cand_b: Dict[str, Any], column: str) -> str:
    column_instruction = build_column_specific_instruction(column)

    prompt = f"""
你现在在做 Adult 表格数据清洗中的错误修复排序任务。

目标：
对于一个被检测为错误的单元格，请在两个候选修复值 A 和 B 中判断谁更适合作为最终修复值。
你必须只在 A、B、Unknown 中三选一：
- A：候选 A 更好
- B：候选 B 更好
- Unknown：无法可靠判断，或者两个候选都不够好

Adult 数据集字段包括：
age, workclass, education, maritalstatus, occupation, relationship, race, sex, hoursperweek, country, income。

通用判断原则：
1. 优先考虑强规则证据：expected_values_from_rules、violated_rules_detailed、relationship/sex/maritalstatus/education-age 约束。
2. 优先考虑 Adult 语义一致性：relationship 与 sex/maritalstatus/age 是否一致，education 与 age 是否合理，income/hours 是否只是规范化。
3. 再考虑 domain 合法性：候选是否属于 Adult 合法枚举或合法数值范围。
4. 再考虑 profile 分布：same_profile_distribution / work_profile_distribution 只能作为辅助证据，不能单独压过强规则。
5. 再考虑邻居/相似行/列内高频：这些是弱证据，不要仅因为候选常见就选择它。
6. 如果两个候选都明显不可靠，或只是高频弱候选，输出 Unknown。
7. 对 relationship、sex、maritalstatus 这类互相约束字段，候选应能减少上下文冲突。
8. 对 income、hoursperweek，除非是明显格式/规范化修复，否则不要凭统计猜测。

本列专属判断：
{column_instruction}

请严格输出 JSON，不要输出额外解释文字。
JSON 格式如下：
{{
  "winner": "A" 或 "B" 或 "Unknown",
  "reason_type": "rule_consistency" / "adult_profile_consistency" / "domain_validity" / "relationship_consistency" / "education_age_consistency" / "context_match" / "cooccurrence_support" / "spelling_similarity" / "pattern_match" / "uncertain" / "other",
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


def build_pairwise_requests(sampled_df: pd.DataFrame, cand_df: pd.DataFrame, detail_map: Dict[Tuple[int, str], Dict[str, Any]], args) -> List[Dict[str, Any]]:
    requests = []

    for _, row in sampled_df.iterrows():
        row_id = int(row["row_id"])
        column = str(row["column"])
        dirty_value = row.get("dirty_value")
        key = (row_id, column)

        candidates = build_candidate_list_for_cell(key, cand_df, args)
        if len(candidates) <= 1:
            continue

        pair_ids = make_pair_ids(candidates)
        detail_obj = detail_map.get(key, {})
        context_obj = build_context_for_prompt(detail_obj, dirty_value, column, row)

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
# 9. LLM call
# ============================================================

def get_client(args) -> Any:
    if OpenAI is None:
        raise ImportError("openai package is not installed. Please install openai or use --dry_run_only_export_requests 1.")
    if not args.api_key:
        raise ValueError("未设置 API key。请设置 DEEPSEEK_API_KEY 或传入 --api_key。")
    return OpenAI(api_key=args.api_key, base_url=args.base_url)


def call_llm_pairwise(client: Any, prompt: str, args) -> Dict[str, Any]:
    for attempt in range(1, args.max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=args.model_name,
                temperature=args.temperature,
                messages=[
                    {"role": "system", "content": "你是一个严谨的 Adult 表格数据清洗排序专家。你必须只输出 JSON。"},
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
            if attempt == args.max_retries:
                return {
                    "winner": "Unknown",
                    "reason_type": "uncertain",
                    "reason": f"LLM 调用失败: {str(e)}",
                }
            time.sleep(args.retry_sleep_seconds)


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


def load_existing_responses(path: str, resume: bool) -> Dict[str, Dict[str, Any]]:
    existing = {}
    if not resume or not os.path.exists(path):
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


def call_one_request(req: Dict[str, Any], args) -> Dict[str, Any]:
    client = get_client(args)
    llm_out = call_llm_pairwise(client, req["pairwise_prompt"], args)
    return normalize_response_item(req, llm_out)


def run_llm_requests(requests: List[Dict[str, Any]], output_path: str, args) -> List[Dict[str, Any]]:
    existing = load_existing_responses(output_path, args.resume_from_existing_responses)

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
    print(f"[INFO] LLM max workers: {args.llm_max_workers}")

    if not pending:
        return [all_by_id[str(req["request_id"])] for req in requests if str(req["request_id"]) in all_by_id]

    write_lock = threading.Lock()
    done_count = 0

    with open(output_path, "a", encoding="utf-8") as f:
        if args.llm_max_workers <= 1:
            for req in pending:
                item = call_one_request(req, args)
                with write_lock:
                    write_jsonl_record(f, item)
                    f.flush()
                all_by_id[str(req["request_id"])] = item
                done_count += 1
                if done_count % args.llm_progress_every == 0:
                    print(f"[INFO] LLM progress: new_done={done_count}/{len(pending)}, total_done={len(all_by_id)}/{len(requests)}")
        else:
            with ThreadPoolExecutor(max_workers=args.llm_max_workers) as ex:
                future_to_req = {ex.submit(call_one_request, req, args): req for req in pending}
                for fut in as_completed(future_to_req):
                    req = future_to_req[fut]
                    try:
                        item = fut.result()
                    except Exception as e:
                        item = normalize_response_item(req, {
                            "winner": "Unknown",
                            "reason_type": "uncertain",
                            "reason": f"并发调用异常: {str(e)}",
                        })

                    with write_lock:
                        write_jsonl_record(f, item)
                        f.flush()

                    all_by_id[str(req["request_id"])] = item
                    done_count += 1
                    if done_count % args.llm_progress_every == 0:
                        print(f"[INFO] LLM progress: new_done={done_count}/{len(pending)}, total_done={len(all_by_id)}/{len(requests)}")

    return [all_by_id[str(req["request_id"])] for req in requests if str(req["request_id"]) in all_by_id]


# ============================================================
# 10. Aggregate pairwise outputs
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
            key=lambda x: (-x[1], -x[2], x[3], safe_candidate_name(x[0])),
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
# 11. Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--input_candidate_features_csv", default=DEFAULT_INPUT_CANDIDATE_FEATURES_CSV)
    p.add_argument("--input_detail_jsonl", default=DEFAULT_INPUT_DETAIL_JSONL)

    p.add_argument("--output_cell_summary_csv", default=DEFAULT_OUTPUT_CELL_SUMMARY_CSV)
    p.add_argument("--output_clustered_csv", default=DEFAULT_OUTPUT_CLUSTERED_CSV)
    p.add_argument("--output_sampled_csv", default=DEFAULT_OUTPUT_SAMPLED_CSV)
    p.add_argument("--output_sampled_jsonl", default=DEFAULT_OUTPUT_SAMPLED_JSONL)
    p.add_argument("--output_pairwise_requests_jsonl", default=DEFAULT_OUTPUT_PAIRWISE_REQUESTS_JSONL)
    p.add_argument("--output_pairwise_responses_jsonl", default=DEFAULT_OUTPUT_PAIRWISE_RESPONSES_JSONL)
    p.add_argument("--output_teacher_ranking_csv", default=DEFAULT_OUTPUT_TEACHER_RANKING_CSV)
    p.add_argument("--output_teacher_top1_csv", default=DEFAULT_OUTPUT_TEACHER_TOP1_CSV)
    p.add_argument("--output_bucket_summary_csv", default=DEFAULT_OUTPUT_BUCKET_SUMMARY_CSV)

    p.add_argument("--use_column_in_bucket", type=int, default=int(DEFAULT_USE_COLUMN_IN_BUCKET))
    p.add_argument("--min_bucket_size_for_cluster", type=int, default=DEFAULT_MIN_BUCKET_SIZE_FOR_CLUSTER)
    p.add_argument("--max_clusters_per_bucket", type=int, default=DEFAULT_MAX_CLUSTERS_PER_BUCKET)
    p.add_argument("--target_cluster_size", type=int, default=DEFAULT_TARGET_CLUSTER_SIZE)
    p.add_argument("--small_bucket_full_sample_threshold", type=int, default=DEFAULT_SMALL_BUCKET_FULL_SAMPLE_THRESHOLD)

    p.add_argument("--sample_center", type=int, default=int(DEFAULT_SAMPLE_CENTER))
    p.add_argument("--sample_boundary", type=int, default=int(DEFAULT_SAMPLE_BOUNDARY))
    p.add_argument("--sample_outlier", type=int, default=int(DEFAULT_SAMPLE_OUTLIER))
    p.add_argument("--random_state", type=int, default=DEFAULT_RANDOM_STATE)

    p.add_argument("--default_max_candidates_per_cell_for_llm", type=int, default=DEFAULT_MAX_CANDIDATES_PER_CELL_FOR_LLM)
    p.add_argument("--hard_column_max_candidates", type=int, default=HARD_COLUMN_MAX_CANDIDATES)

    p.add_argument("--model_name", default=MODEL_NAME)
    p.add_argument("--api_key", default=API_KEY)
    p.add_argument("--base_url", default=BASE_URL)
    p.add_argument("--temperature", type=float, default=TEMPERATURE)
    p.add_argument("--max_retries", type=int, default=MAX_RETRIES)
    p.add_argument("--retry_sleep_seconds", type=float, default=RETRY_SLEEP_SECONDS)
    p.add_argument("--llm_max_workers", type=int, default=LLM_MAX_WORKERS)
    p.add_argument("--llm_progress_every", type=int, default=LLM_PROGRESS_EVERY)
    p.add_argument("--resume_from_existing_responses", type=int, default=int(RESUME_FROM_EXISTING_RESPONSES))
    p.add_argument("--dry_run_only_export_requests", type=int, default=0)

    args = p.parse_args()

    args.use_column_in_bucket = bool(args.use_column_in_bucket)
    args.sample_center = bool(args.sample_center)
    args.sample_boundary = bool(args.sample_boundary)
    args.sample_outlier = bool(args.sample_outlier)
    args.resume_from_existing_responses = bool(args.resume_from_existing_responses)
    args.dry_run_only_export_requests = bool(args.dry_run_only_export_requests)

    return args


def main():
    args = parse_args()

    output_paths = [
        args.output_cell_summary_csv,
        args.output_clustered_csv,
        args.output_sampled_csv,
        args.output_sampled_jsonl,
        args.output_pairwise_requests_jsonl,
        args.output_pairwise_responses_jsonl,
        args.output_teacher_ranking_csv,
        args.output_teacher_top1_csv,
        args.output_bucket_summary_csv,
    ]
    for p in output_paths:
        ensure_parent(resolve_output_path(p))

    args.output_cell_summary_csv = resolve_output_path(args.output_cell_summary_csv)
    args.output_clustered_csv = resolve_output_path(args.output_clustered_csv)
    args.output_sampled_csv = resolve_output_path(args.output_sampled_csv)
    args.output_sampled_jsonl = resolve_output_path(args.output_sampled_jsonl)
    args.output_pairwise_requests_jsonl = resolve_output_path(args.output_pairwise_requests_jsonl)
    args.output_pairwise_responses_jsonl = resolve_output_path(args.output_pairwise_responses_jsonl)
    args.output_teacher_ranking_csv = resolve_output_path(args.output_teacher_ranking_csv)
    args.output_teacher_top1_csv = resolve_output_path(args.output_teacher_top1_csv)
    args.output_bucket_summary_csv = resolve_output_path(args.output_bucket_summary_csv)

    if not os.path.exists(args.input_candidate_features_csv):
        raise FileNotFoundError(f"候选特征文件不存在: {args.input_candidate_features_csv}")

    cand_df = pd.read_csv(args.input_candidate_features_csv, encoding="utf-8-sig")
    required = {"row_id", "column", "candidate_value"}
    missing = required - set(cand_df.columns)
    if missing:
        raise ValueError(f"候选特征文件缺少必要字段: {missing}")

    cand_df["row_id"] = pd.to_numeric(cand_df["row_id"], errors="raise").astype(int)
    cand_df["column"] = cand_df["column"].astype(str)

    detail_map = load_detail_jsonl(args.input_detail_jsonl)
    print(f"[INFO] candidate feature rows: {len(cand_df)}")
    print(f"[INFO] detail context cells: {len(detail_map)}")

    # 1) Cell-level summary
    cell_summary_df = build_cell_summary_df(cand_df)
    cell_summary_df.to_csv(args.output_cell_summary_csv, index=False, encoding="utf-8-sig")

    # 2) Bucket + cluster + sample
    cell_summary_df = assign_buckets(cell_summary_df, args.use_column_in_bucket)

    clustered_parts = []
    bucket_summary_rows = []

    for bucket_id, bucket_df in cell_summary_df.groupby("bucket_id", sort=False):
        bucket_clustered = cluster_bucket(bucket_df, args)
        bucket_clustered_parts = []

        for cluster_id, cluster_df in bucket_clustered.groupby("cluster_id", sort=False):
            cluster_sampled = sample_from_cluster(cluster_df, args)
            bucket_clustered_parts.append(cluster_sampled)

            bucket_summary_rows.append({
                "bucket_id": bucket_id,
                "bucket_size": len(bucket_df),
                "cluster_id": int(cluster_id),
                "cluster_size": len(cluster_df),
                "sampled_count": int(cluster_sampled["is_sampled"].sum()),
                "column": str(cluster_df["column"].iloc[0]) if "column" in cluster_df.columns else "unknown",
                "semantic_type": str(cluster_df["semantic_type"].iloc[0]) if "semantic_type" in cluster_df.columns else "unknown",
                "main_rule_type": str(cluster_df["main_rule_type"].iloc[0]) if "main_rule_type" in cluster_df.columns else "unknown",
                "main_usage_role": str(cluster_df["main_usage_role"].iloc[0]) if "main_usage_role" in cluster_df.columns else "unknown",
                "confidence_bucket": str(cluster_df["confidence_bucket"].iloc[0]) if "confidence_bucket" in cluster_df.columns else "unknown",
                "candidate_count_bucket": str(cluster_df["candidate_count_bucket"].iloc[0]) if "candidate_count_bucket" in cluster_df.columns else "unknown",
                "adult_conflict_bucket": str(cluster_df["adult_conflict_bucket"].iloc[0]) if "adult_conflict_bucket" in cluster_df.columns else "unknown",
                "is_hard_column": int(cluster_df["is_hard_column"].iloc[0]) if "is_hard_column" in cluster_df.columns else 0,
            })

        clustered_bucket = pd.concat(bucket_clustered_parts, ignore_index=True)
        clustered_parts.append(clustered_bucket)

    clustered_df = pd.concat(clustered_parts, ignore_index=True) if clustered_parts else pd.DataFrame()
    clustered_df.to_csv(args.output_clustered_csv, index=False, encoding="utf-8-sig")

    sampled_df = clustered_df[clustered_df["is_sampled"] == 1].copy() if not clustered_df.empty else pd.DataFrame()
    sampled_df.to_csv(args.output_sampled_csv, index=False, encoding="utf-8-sig")

    if not sampled_df.empty:
        build_sampled_jsonl(sampled_df, cand_df, detail_map, args.output_sampled_jsonl, args)
    else:
        with open(args.output_sampled_jsonl, "w", encoding="utf-8"):
            pass

    pd.DataFrame(bucket_summary_rows).to_csv(
        args.output_bucket_summary_csv, index=False, encoding="utf-8-sig"
    )

    # 3) Pairwise requests
    requests = build_pairwise_requests(sampled_df, cand_df, detail_map, args) if not sampled_df.empty else []
    with open(args.output_pairwise_requests_jsonl, "w", encoding="utf-8") as f:
        for req in requests:
            write_jsonl_record(f, req)

    print(f"[OK] cell-level 摘要已保存: {args.output_cell_summary_csv}")
    print(f"[OK] 完整聚类结果已保存: {args.output_clustered_csv}")
    print(f"[OK] 采样摘要 CSV 已保存: {args.output_sampled_csv}")
    print(f"[OK] 采样详细 JSONL 已保存: {args.output_sampled_jsonl}")
    print(f"[OK] bucket/cluster 摘要已保存: {args.output_bucket_summary_csv}")
    print(f"[OK] pairwise 请求已保存: {args.output_pairwise_requests_jsonl}")
    print(f"[INFO] 采样 cell 数: {len(sampled_df)}")
    print(f"[INFO] pairwise 请求数: {len(requests)}")

    if args.dry_run_only_export_requests:
        print("[INFO] 当前为 DRY RUN，仅导出请求，不调用 LLM。")
        return

    # 4) LLM call
    responses = run_llm_requests(requests, args.output_pairwise_responses_jsonl, args)

    # 5) Aggregate teacher ranking
    ranking_rows, top1_rows = aggregate_pairwise_results(responses)

    pd.DataFrame(ranking_rows).to_csv(args.output_teacher_ranking_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(top1_rows).to_csv(args.output_teacher_top1_csv, index=False, encoding="utf-8-sig")

    print(f"[OK] pairwise 响应已保存: {args.output_pairwise_responses_jsonl}")
    print(f"[OK] teacher 排序结果已保存: {args.output_teacher_ranking_csv}")
    print(f"[OK] teacher top1 标签已保存: {args.output_teacher_top1_csv}")

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

        print("\nAdult conflict bucket 分布:")
        print(sampled_df["adult_conflict_bucket"].value_counts(dropna=False).head(20).to_string())

        print("\n采样列分布:")
        print(sampled_df["column"].value_counts(dropna=False).head(20).to_string())


if __name__ == "__main__":
    main()

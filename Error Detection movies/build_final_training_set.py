import json
from typing import Dict, Any, Optional, Set, Tuple, List

import numpy as np
import pandas as pd


# ============================================================
# 1. 用户配置区
# ============================================================

INPUT_TABLE_CSV = "/mnt/mydata/dq/projects/Splittree/movies/movies_dirty.csv"
INPUT_CLUSTERED_CSV = "candidate_clustered_movies.csv"
INPUT_LABELED_CSV = "candidate_sampled_labeled_movies.csv"
INPUT_PROPAGATED_CSV = "propagated_labels_movies.csv"

OUTPUT_FINAL_TRAIN_CSV = "final_train_dataset_movies.csv"
OUTPUT_FINAL_TRAIN_JSONL = "final_train_dataset_movies.jsonl"
OUTPUT_SUMMARY_JSON = "final_train_summary_movies.json"

MISSING_TOKENS = {
    "", "empty", "null", "none", "nan", "na", "n/a", "missing", "unk",
    "unknown", "not available", "not applicable", "tbd", "na."
}

# ------------------------------------------------------------
# 候选内样本保留策略
# ------------------------------------------------------------
KEEP_ALL_LLM_LABELED = True

# 注意：这里比传播脚本再做一次筛选。
# 传播脚本已经限制了传播数量，这里继续只收高置信传播样本。
CLUSTER_PROP_MIN_CONF = 0.90
KNN_PROP_MIN_CONF = 0.93

# 是否保留低置信 LLM 样本。
# 默认保留，因为它们是真实 LLM 标签，但权重会根据置信度调整。
KEEP_LOW_CONF_LLM = True
LLM_LOW_CONF_THRESHOLD = 0.60
LLM_HUMAN_REVIEW_WEIGHT_DECAY = 0.70

# ------------------------------------------------------------
# 训练集大小控制
# ------------------------------------------------------------
ENABLE_TRAIN_SIZE_BUDGET = True

# 最终训练集最大规模。
# 如果 LLM 标注约 1000，传播约 300~500，outside clean 适量补充，
# 最终一般控制在 1500~2500 比较合适。
MAX_FINAL_TRAIN_ROWS = 2500

# 候选内传播样本最多保留多少，不包含 LLM 样本。
MAX_PROPAGATED_TRAIN_ROWS = 500

# 每列候选内训练样本最多多少，避免 Release Date 等列过多。
MAX_INTERNAL_TRAIN_PER_COLUMN = {
    "Release Date": 220,
    "Year": 140,
    "Duration": 140,
    "RatingValue": 140,
    "RatingCount": 140,
    "ReviewCount": 140,
    "Language": 140,
    "Country": 140,
    "Genre": 140,
    "Director": 160,
    "Creator": 160,
    "Actors": 160,
    "Cast": 160,
    "Name": 140,
    "Filming Locations": 100,
    "Description": 100,
}

# ------------------------------------------------------------
# 候选外 clean 样本构造
# ------------------------------------------------------------
ENABLE_OUTSIDE_CLEAN = True
OUTSIDE_ONLY_FROM_CANDIDATE_COLUMNS = True

# movies 数据集字段长尾明显，outside clean 不宜太多。
OUTSIDE_CLEAN_PER_COLUMN = 60
OUTSIDE_CLEAN_MIN_PER_COLUMN = 10

# 对候选外 clean 的值频率约束。
# 越大越保守，但可能导致长尾列补不到样本。
OUTSIDE_MIN_VALUE_FREQ = 2
OUTSIDE_MAX_VALUE_FREQ_RANK = 30
SKIP_NULL_IN_OUTSIDE_CLEAN = True

# outside clean 全局上限
MAX_OUTSIDE_CLEAN_TOTAL = 600

# 若错误样本较少，按错误数限制 clean 数量
MAX_OUTSIDE_CLEAN_TO_ERROR_RATIO = 0.8

# ------------------------------------------------------------
# 样本权重
# ------------------------------------------------------------
WEIGHT_LLM = 1.0
WEIGHT_LLM_LOW_CONF = 0.75
WEIGHT_CLUSTER_PROP = 0.65
WEIGHT_KNN_PROP = 0.45
WEIGHT_OUTSIDE_CLEAN = 0.35

# ------------------------------------------------------------
# 类别平衡
# ------------------------------------------------------------
ENABLE_BALANCE = True

# correct 最多是 error 的多少倍。
# 因为候选外 clean 都是 correct，太多会让模型偏向 correct。
MAX_CORRECT_TO_ERROR_RATIO = 1.3
PREFER_INTERNAL_CORRECT_FIRST = True

# 如果 error 太少，至少保留一定数量的 correct，防止训练不稳定。
MIN_CORRECT_KEEP = 100

RANDOM_STATE = 42


# ============================================================
# 2. movies 字段配置
# ============================================================

MOVIES_COLUMNS = [
    "Id",
    "Name",
    "Year",
    "Release Date",
    "Director",
    "Creator",
    "Actors",
    "Cast",
    "Language",
    "Country",
    "Duration",
    "RatingValue",
    "RatingCount",
    "ReviewCount",
    "Genre",
    "Filming Locations",
    "Description",
]

MOVIES_TARGET_COLUMNS = {
    "Name",
    "Year",
    "Release Date",
    "Director",
    "Creator",
    "Actors",
    "Cast",
    "Language",
    "Country",
    "Duration",
    "RatingValue",
    "RatingCount",
    "ReviewCount",
    "Genre",
    "Filming Locations",
    "Description",
}

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
    "dist_to_center",
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
    "bucket_id",
    "sample_role",
]

MOVIES_META_COLUMNS = MOVIES_NUMERIC_FEATURE_COLUMNS + MOVIES_CATEGORICAL_FEATURE_COLUMNS


# ============================================================
# 3. 基础函数
# ============================================================

def normalize_value(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.lower() in MISSING_TOKENS:
        return None
    return s


def normalize_series(s: pd.Series) -> pd.Series:
    return s.map(normalize_value)


def maybe_drop_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [c for c in df.columns if not str(c).lower().startswith("unnamed:")]
    return df[keep_cols].copy()


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


def safe_bool(x, default=False) -> bool:
    if pd.isna(x) or x is None:
        return default
    s = str(x).strip().lower()
    if s in {"true", "1", "yes"}:
        return True
    if s in {"false", "0", "no"}:
        return False
    return default


def normalize_label_binary_from_row(row: pd.Series) -> Optional[int]:
    if "label_binary" in row.index and pd.notna(row["label_binary"]):
        try:
            return int(float(row["label_binary"]))
        except Exception:
            pass

    if "label_binary_norm" in row.index and pd.notna(row["label_binary_norm"]):
        try:
            return int(float(row["label_binary_norm"]))
        except Exception:
            pass

    if "label" in row.index and pd.notna(row["label"]):
        s = str(row["label"]).strip().lower()
        if s == "error":
            return 1
        if s == "correct":
            return 0

    if "is_error" in row.index and pd.notna(row["is_error"]):
        s = str(row["is_error"]).strip().lower()
        if s in {"true", "1", "yes"}:
            return 1
        if s in {"false", "0", "no"}:
            return 0

    return None


def compute_value_frequency_rank(counter: Dict[Any, int], value: Any) -> Optional[int]:
    if value is None:
        return None
    items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    for idx, (v, _) in enumerate(items, start=1):
        if v == value:
            return idx
    return None


def to_json_safe(x):
    if pd.isna(x):
        return None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return x


def get_default_for_feature_col(col: str):
    if col in MOVIES_NUMERIC_FEATURE_COLUMNS:
        return 0.0
    if col in MOVIES_CATEGORICAL_FEATURE_COLUMNS:
        return "unknown"
    return None


def ensure_movies_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    expected_cols = {
        "value": None,
        "semantic_type": "unknown",
        "detected_type": "unknown",
        "main_rule_type": "none",
        "main_usage_role": "none",
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

    for col in MOVIES_NUMERIC_FEATURE_COLUMNS:
        expected_cols.setdefault(col, 0.0)

    for col in MOVIES_CATEGORICAL_FEATURE_COLUMNS:
        expected_cols.setdefault(col, "unknown")

    for col, default_val in expected_cols.items():
        if col not in df.columns:
            df[col] = default_val

    for col in MOVIES_NUMERIC_FEATURE_COLUMNS + ["cluster_id", "dist_to_center", "is_sampled"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ============================================================
# 4. 数据加载
# ============================================================

def load_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df = maybe_drop_unnamed_columns(df)
    for col in df.columns:
        df[col] = normalize_series(df[col])
    return df


def load_clustered_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["row_id"] = df["row_id"].astype(int)
    df["column"] = df["column"].astype(str)
    df = ensure_movies_columns(df)
    return df


def load_labeled_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["row_id"] = df["row_id"].astype(int)
    df["column"] = df["column"].astype(str)

    label_binary_list = []
    for _, row in df.iterrows():
        label_binary_list.append(normalize_label_binary_from_row(row))

    df["label_binary_norm"] = label_binary_list
    df = df[pd.notna(df["label_binary_norm"])].copy()
    df["label_binary_norm"] = df["label_binary_norm"].astype(int)

    if "confidence" not in df.columns:
        df["confidence"] = 1.0
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(1.0)

    expected_cols = {
        "error_type": None,
        "reason_short": None,
        "reason_detailed": None,
        "suggested_correct_value": None,
        "needs_human_review": None,
        "evidence_used": None,
    }

    for col in MOVIES_META_COLUMNS:
        expected_cols.setdefault(col, get_default_for_feature_col(col))

    for col, default_val in expected_cols.items():
        if col not in df.columns:
            df[col] = default_val

    return df


def load_propagated_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["row_id"] = df["row_id"].astype(int)
    df["column"] = df["column"].astype(str)

    if "label_binary" not in df.columns:
        label_binary_list = []
        for _, row in df.iterrows():
            label_binary_list.append(normalize_label_binary_from_row(row))
        df["label_binary"] = label_binary_list

    if "propagation_confidence" not in df.columns:
        if "confidence" in df.columns:
            df["propagation_confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
        else:
            df["propagation_confidence"] = np.nan

    if "label_source" not in df.columns:
        df["label_source"] = "unknown"

    if "sample_weight" not in df.columns:
        df["sample_weight"] = np.nan

    expected_cols = {
        "error_type": None,
        "reason_short": None,
        "reason_detailed": None,
        "suggested_correct_value": None,
        "needs_human_review": None,
        "evidence_used": None,
        "value": None,
        "bucket_id": "unknown",
        "cluster_id": -1,
        "dist_to_center": 0.0,
        "sample_role": "",
        "is_sampled": 0,
    }

    for col in MOVIES_META_COLUMNS:
        expected_cols.setdefault(col, get_default_for_feature_col(col))

    for col, default_val in expected_cols.items():
        if col not in df.columns:
            df[col] = default_val

    df = ensure_movies_columns(df)

    return df


# ============================================================
# 5. 样本质量与优先级
# ============================================================

def compute_llm_sample_weight(confidence: float, needs_human_review: Any) -> float:
    conf = min(max(float(confidence), 0.0), 1.0)
    if conf < LLM_LOW_CONF_THRESHOLD:
        weight = WEIGHT_LLM_LOW_CONF
    else:
        weight = WEIGHT_LLM

    if safe_bool(needs_human_review, False):
        weight *= LLM_HUMAN_REVIEW_WEIGHT_DECAY

    return round(float(weight), 6)


def internal_priority_score(row: pd.Series) -> float:
    label_source = str(row.get("label_source", "unknown"))
    label_binary = normalize_label_binary_from_row(row)

    sample_weight = safe_float(row.get("sample_weight", 0.0), 0.0)
    prop_conf = safe_float(row.get("propagation_confidence", 0.0), 0.0)
    conflict_score = safe_float(row.get("conflict_score", 0.0), 0.0)
    posterior = safe_float(row.get("posterior_error_probability", 0.0), 0.0)
    violation_count = safe_float(row.get("violation_count", 0.0), 0.0)
    movie_typed = safe_float(row.get("movie_typed_rule_count", 0.0), 0.0)
    movie_cons = safe_float(row.get("movie_consistency_rule_count", 0.0), 0.0)

    source_bonus = {
        "llm": 3.0,
        "cluster_propagation": 1.7,
        "knn_propagation": 1.2,
        "outside_clean": 0.2,
    }.get(label_source, 0.0)

    error_bonus = 1.0 if label_binary == 1 else 0.0

    return (
        source_bonus
        + 2.0 * sample_weight
        + 1.5 * prop_conf
        + 1.0 * posterior
        + 0.01 * conflict_score
        + 0.2 * violation_count
        + 0.4 * movie_typed
        + 0.6 * movie_cons
        + error_bonus
    )


# ============================================================
# 6. 候选内训练集构造
# ============================================================

def build_candidate_training_from_labels(
    clustered_df: pd.DataFrame,
    labeled_df: pd.DataFrame,
    propagated_df: pd.DataFrame
) -> pd.DataFrame:
    clustered_map = {}
    for _, row in clustered_df.iterrows():
        clustered_map[(int(row["row_id"]), str(row["column"]))] = row.to_dict()

    train_rows = []

    # 6.1 LLM 直接标注样本
    if KEEP_ALL_LLM_LABELED:
        for _, row in labeled_df.iterrows():
            key = (int(row["row_id"]), str(row["column"]))
            if key not in clustered_map:
                continue

            label_binary = int(row["label_binary_norm"])
            label = "error" if label_binary == 1 else "correct"
            confidence = safe_float(row.get("confidence", 1.0), 1.0)

            if (not KEEP_LOW_CONF_LLM) and confidence < LLM_LOW_CONF_THRESHOLD:
                continue

            base = dict(clustered_map[key])
            out = dict(base)

            out["label_binary"] = label_binary
            out["label"] = label
            out["label_source"] = "llm"
            out["propagation_confidence"] = confidence
            out["sample_weight"] = compute_llm_sample_weight(confidence, row.get("needs_human_review", False))
            out["is_outside_candidate"] = 0

            for extra_col in [
                "error_type",
                "reason_short",
                "reason_detailed",
                "suggested_correct_value",
                "needs_human_review",
                "evidence_used",
            ]:
                out[extra_col] = row.get(extra_col) if extra_col in row else None

            for extra_col in MOVIES_META_COLUMNS:
                if extra_col in row and pd.notna(row.get(extra_col)):
                    out[extra_col] = row.get(extra_col)

            train_rows.append(out)

    # 6.2 传播样本：去掉已被 LLM 直接标注的
    llm_keys = set((int(r["row_id"]), str(r["column"])) for _, r in labeled_df.iterrows())

    prop_rows = []
    for _, row in propagated_df.iterrows():
        key = (int(row["row_id"]), str(row["column"]))
        if key in llm_keys:
            continue
        if key not in clustered_map:
            continue

        label_source = str(row.get("label_source", "")).strip()
        label_binary = normalize_label_binary_from_row(row)
        if label_binary is None:
            continue

        prop_conf = safe_float(row.get("propagation_confidence", np.nan), np.nan)
        if pd.isna(prop_conf):
            continue

        keep = False
        sample_weight = None

        if label_source == "cluster_propagation" and prop_conf >= CLUSTER_PROP_MIN_CONF:
            keep = True
            sample_weight = WEIGHT_CLUSTER_PROP
        elif label_source == "knn_propagation" and prop_conf >= KNN_PROP_MIN_CONF:
            keep = True
            sample_weight = WEIGHT_KNN_PROP

        if not keep:
            continue

        base = dict(clustered_map[key])
        out = dict(base)

        out["label_binary"] = int(label_binary)
        out["label"] = "error" if int(label_binary) == 1 else "correct"
        out["label_source"] = label_source
        out["propagation_confidence"] = prop_conf
        out["sample_weight"] = sample_weight
        out["is_outside_candidate"] = 0

        for extra_col in MOVIES_META_COLUMNS:
            if extra_col in row and pd.notna(row.get(extra_col)):
                out[extra_col] = row.get(extra_col)

        for extra_col in [
            "error_type",
            "reason_short",
            "reason_detailed",
            "suggested_correct_value",
            "needs_human_review",
            "evidence_used",
        ]:
            out[extra_col] = row.get(extra_col) if extra_col in row else None

        prop_rows.append(out)

    # 控制传播样本总数
    if len(prop_rows) > 0:
        prop_df = pd.DataFrame(prop_rows)
        prop_df["_internal_priority_score"] = prop_df.apply(internal_priority_score, axis=1)
        prop_df = prop_df.sort_values(
            by=["_internal_priority_score", "propagation_confidence", "conflict_score"],
            ascending=[False, False, False]
        ).head(MAX_PROPAGATED_TRAIN_ROWS)
        prop_df = prop_df.drop(columns=["_internal_priority_score"], errors="ignore")
        train_rows.extend(prop_df.to_dict("records"))

    df = pd.DataFrame(train_rows)
    if len(df) == 0:
        return df

    df = ensure_movies_columns(df)

    source_priority = {"llm": 4, "cluster_propagation": 2, "knn_propagation": 1}
    df["_source_priority"] = df["label_source"].map(lambda x: source_priority.get(str(x), 0))
    df["_internal_priority_score"] = df.apply(internal_priority_score, axis=1)

    df = df.sort_values(
        by=["row_id", "column", "_source_priority", "_internal_priority_score", "propagation_confidence"],
        ascending=[True, True, False, False, False]
    )
    df = df.drop_duplicates(subset=["row_id", "column"], keep="first").copy()
    df = df.drop(columns=["_source_priority", "_internal_priority_score"], errors="ignore")

    df = apply_internal_column_budget(df)

    return df


def apply_internal_column_budget(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return df

    parts = []
    for col, g in df.groupby("column", sort=False):
        cap = MAX_INTERNAL_TRAIN_PER_COLUMN.get(str(col), max(80, MAX_FINAL_TRAIN_ROWS // 10))
        if len(g) <= cap:
            parts.append(g)
            continue

        # error 优先保留，再保留高质量 correct
        g = g.copy()
        g["_internal_priority_score"] = g.apply(internal_priority_score, axis=1)

        err = g[g["label_binary"] == 1].sort_values(
            ["_internal_priority_score", "sample_weight", "propagation_confidence"],
            ascending=[False, False, False]
        )
        cor = g[g["label_binary"] == 0].sort_values(
            ["_internal_priority_score", "sample_weight", "propagation_confidence"],
            ascending=[False, False, False]
        )

        if len(err) >= cap:
            keep = err.head(cap)
        else:
            remain = cap - len(err)
            keep = pd.concat([err, cor.head(remain)], ignore_index=False)

        keep = keep.drop(columns=["_internal_priority_score"], errors="ignore")
        parts.append(keep)

    return pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0].copy()


# ============================================================
# 7. 候选外 clean 样本构造
# ============================================================

def build_outside_clean_training_simple(
    df_table: pd.DataFrame,
    clustered_df: pd.DataFrame,
    candidate_train_df: pd.DataFrame
) -> pd.DataFrame:
    if not ENABLE_OUTSIDE_CLEAN:
        return pd.DataFrame()

    candidate_key_set: Set[Tuple[int, str]] = set(
        (int(r["row_id"]), str(r["column"])) for _, r in clustered_df.iterrows()
    )

    candidate_columns = set(str(c) for c in clustered_df["column"].unique().tolist())
    candidate_train_error_count = int((candidate_train_df["label_binary"] == 1).sum()) if len(candidate_train_df) > 0 else 0

    outside_rows = []

    for col in df_table.columns:
        if OUTSIDE_ONLY_FROM_CANDIDATE_COLUMNS and col not in candidate_columns:
            continue
        if col not in MOVIES_TARGET_COLUMNS:
            continue

        col_series = df_table[col]
        value_counter = col_series.dropna().value_counts().to_dict()

        pool_rows = []

        for row_id in range(len(df_table)):
            key = (row_id, col)
            if key in candidate_key_set:
                continue

            value = df_table.iloc[row_id][col]

            if SKIP_NULL_IN_OUTSIDE_CLEAN and value is None:
                continue

            value_freq = value_counter.get(value, 0) if value is not None else 0
            value_rank = compute_value_frequency_rank(value_counter, value) if value is not None else None

            if value is not None:
                # 长尾文本列不能太严格，否则完全补不到 clean；但依然不取 singleton
                if col in {"Name", "Director", "Creator", "Actors", "Cast", "Filming Locations", "Description"}:
                    if value_freq < max(1, OUTSIDE_MIN_VALUE_FREQ - 1):
                        continue
                    if value_rank is not None and value_rank > max(OUTSIDE_MAX_VALUE_FREQ_RANK, 50):
                        continue
                else:
                    if value_freq < OUTSIDE_MIN_VALUE_FREQ:
                        continue
                    if value_rank is not None and value_rank > OUTSIDE_MAX_VALUE_FREQ_RANK:
                        continue

            pool_rows.append({
                "row_id": row_id,
                "column": col,
                "value": value,
                "value_frequency": value_freq,
                "value_frequency_rank": value_rank,
            })

        if len(pool_rows) == 0:
            continue

        pool_df = pd.DataFrame(pool_rows)
        pool_df = pool_df.sort_values(
            by=["value_frequency", "value_frequency_rank", "row_id"],
            ascending=[False, True, True]
        ).copy()

        target_n = max(OUTSIDE_CLEAN_MIN_PER_COLUMN, min(OUTSIDE_CLEAN_PER_COLUMN, len(pool_df)))
        selected_df = pool_df.head(target_n).copy()

        for _, row in selected_df.iterrows():
            outside_rows.append(build_one_outside_clean_row(row))

    outside_df = pd.DataFrame(outside_rows)

    if len(outside_df) == 0:
        return outside_df

    # 按错误数量限制 outside clean
    if candidate_train_error_count > 0:
        max_by_error = int(candidate_train_error_count * MAX_OUTSIDE_CLEAN_TO_ERROR_RATIO)
        max_outside = min(MAX_OUTSIDE_CLEAN_TOTAL, max_by_error)
    else:
        max_outside = min(MAX_OUTSIDE_CLEAN_TOTAL, len(outside_df))

    if len(outside_df) > max_outside:
        outside_df = outside_df.sort_values(
            by=["value_frequency", "value_frequency_rank"],
            ascending=[False, True]
        ).head(max_outside).copy()

    return outside_df


def build_one_outside_clean_row(row: pd.Series) -> Dict[str, Any]:
    col = str(row["column"])

    out = {
        "row_id": int(row["row_id"]),
        "column": col,
        "value": row["value"],

        "semantic_type": "outside_clean_unknown",
        "detected_type": "outside_clean_unknown",
        "violation_count": 0,
        "conflict_score": 0.0,
        "value_frequency": safe_int(row.get("value_frequency", 0), 0),
        "value_frequency_rank": row.get("value_frequency_rank", np.nan),
        "neighbor_majority_value": None,
        "neighbor_majority_ratio": np.nan,
        "prior_error_probability": 0.01,
        "posterior_error_probability": 0.01,

        "main_rule_type": "none",
        "main_usage_role": "outside_clean",
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

        "pattern_bucket": "outside_clean_unknown",
        "rarity_bucket": "outside_clean_unknown",
        "neighbor_bucket": "outside_clean_unknown",
        "movie_rule_bucket": "outside_clean_unknown",
        "movie_field_bucket": movies_field_bucket_for_column(col),
        "year_bucket": "outside_clean_unknown",
        "release_year_gap_bucket": "outside_clean_unknown",
        "duration_bucket": "outside_clean_unknown",
        "rating_bucket": "outside_clean_unknown",
        "count_bucket": "outside_clean_unknown",
        "review_ratio_bucket": "outside_clean_unknown",
        "actors_cast_overlap_bucket": "outside_clean_unknown",
        "list_bucket": "outside_clean_unknown",
        "text_bucket": "outside_clean_unknown",

        "bucket_id": "outside_clean",
        "cluster_id": -1,
        "dist_to_center": 0.0,
        "sample_role": "outside_clean",
        "is_sampled": 0,
        "sampling_priority_score": 0.0,

        "label_binary": 0,
        "label": "correct",
        "label_source": "outside_clean",
        "propagation_confidence": 0.95,
        "sample_weight": WEIGHT_OUTSIDE_CLEAN,
        "is_outside_candidate": 1,

        "error_type": "likely_correct",
        "reason_short": "Outside candidate set, used as clean sample.",
        "reason_detailed": (
            "This cell is outside the high-recall candidate set and is sampled "
            "as a likely clean training example for the movies dataset."
        ),
        "suggested_correct_value": row["value"],
        "needs_human_review": False,
        "evidence_used": "outside_candidate_sampling",
    }

    # movies 专用数值列补默认值
    for c in [
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
    ]:
        out[c] = np.nan

    for c in ["parsed_release_country", "movie_row_release_country", "list_token_count_bucket", "text_length_bucket"]:
        out[c] = "outside_clean_unknown"

    return out


def movies_field_bucket_for_column(column: str) -> str:
    col = str(column)
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


# ============================================================
# 8. 平衡训练集
# ============================================================

def balance_final_training(df: pd.DataFrame) -> pd.DataFrame:
    if (not ENABLE_BALANCE) or len(df) == 0:
        return df

    error_df = df[df["label_binary"] == 1].copy()
    correct_df = df[df["label_binary"] == 0].copy()

    if len(error_df) == 0 or len(correct_df) == 0:
        return df

    max_correct = max(MIN_CORRECT_KEEP, int(len(error_df) * MAX_CORRECT_TO_ERROR_RATIO))

    if len(correct_df) <= max_correct:
        return df

    if PREFER_INTERNAL_CORRECT_FIRST:
        internal_correct = correct_df[correct_df["is_outside_candidate"] == 0].copy()
        outside_correct = correct_df[correct_df["is_outside_candidate"] == 1].copy()

        keep_parts = []
        remain = max_correct

        if len(internal_correct) > 0:
            internal_correct["_internal_priority_score"] = internal_correct.apply(internal_priority_score, axis=1)
            internal_correct = internal_correct.sort_values(
                by=["_internal_priority_score", "sample_weight", "propagation_confidence", "conflict_score"],
                ascending=[False, False, False, False]
            )
            keep_internal = internal_correct.head(remain).drop(columns=["_internal_priority_score"], errors="ignore")
            keep_parts.append(keep_internal)
            remain -= len(keep_internal)

        if remain > 0 and len(outside_correct) > 0:
            outside_correct = outside_correct.sort_values(
                by=["sample_weight", "propagation_confidence", "value_frequency"],
                ascending=[False, False, False]
            )
            keep_outside = outside_correct.head(remain)
            keep_parts.append(keep_outside)

        correct_keep_df = pd.concat(keep_parts, ignore_index=True) if keep_parts else pd.DataFrame(columns=df.columns)
    else:
        correct_keep_df = correct_df.sample(n=max_correct, random_state=RANDOM_STATE)

    final_df = pd.concat([error_df, correct_keep_df], ignore_index=True)

    # 最终全局大小限制
    if ENABLE_TRAIN_SIZE_BUDGET and len(final_df) > MAX_FINAL_TRAIN_ROWS:
        final_df = apply_final_size_budget(final_df)

    final_df = final_df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    return final_df


def apply_final_size_budget(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) <= MAX_FINAL_TRAIN_ROWS:
        return df

    df = df.copy()
    df["_internal_priority_score"] = df.apply(internal_priority_score, axis=1)

    error_df = df[df["label_binary"] == 1].copy()
    correct_df = df[df["label_binary"] == 0].copy()

    if len(error_df) >= MAX_FINAL_TRAIN_ROWS:
        keep = error_df.sort_values(
            by=["_internal_priority_score", "sample_weight", "propagation_confidence"],
            ascending=[False, False, False]
        ).head(MAX_FINAL_TRAIN_ROWS)
        return keep.drop(columns=["_internal_priority_score"], errors="ignore")

    remain = MAX_FINAL_TRAIN_ROWS - len(error_df)

    correct_keep = correct_df.sort_values(
        by=["_internal_priority_score", "sample_weight", "propagation_confidence"],
        ascending=[False, False, False]
    ).head(remain)

    keep = pd.concat([error_df, correct_keep], ignore_index=True)
    return keep.drop(columns=["_internal_priority_score"], errors="ignore")


# ============================================================
# 9. 导出
# ============================================================

def export_final_train(df: pd.DataFrame, out_csv: str, out_jsonl: str, out_summary: str):
    df = df.copy()

    preferred_cols = [
        "row_id", "column", "value",
        "label", "label_binary", "label_source", "sample_weight", "is_outside_candidate",
        "propagation_confidence",

        "semantic_type", "detected_type",
        "violation_count", "conflict_score",
        "value_frequency", "value_frequency_rank",
        "neighbor_majority_value", "neighbor_majority_ratio",
        "prior_error_probability", "posterior_error_probability",

        "main_rule_type", "main_usage_role",
        "strong_rule_count", "candidate_generation_rule_count",
        "fd_like_count", "context_rule_count", "global_rule_count",
        "rare_value_count", "typo_rule_count", "pattern_rule_count", "schema_rule_count",
        "movie_typed_rule_count", "movie_consistency_rule_count",

        "parsed_year", "parsed_release_year", "parsed_release_country",
        "duration_minutes", "rating_value_float", "count_value_int",
        "review_total", "list_token_count", "list_token_count_bucket",
        "text_length_bucket", "has_mojibake_or_control_chars",

        "movie_row_year", "movie_row_release_year", "movie_row_release_country",
        "movie_row_release_year_gap", "movie_row_duration_minutes",
        "movie_row_rating_value_float", "movie_row_rating_count_int",
        "movie_row_review_total", "movie_row_review_to_rating_ratio",
        "movie_row_actors_cast_overlap",

        "pattern_bucket", "rarity_bucket", "neighbor_bucket",
        "movie_rule_bucket", "movie_field_bucket",
        "year_bucket", "release_year_gap_bucket", "duration_bucket",
        "rating_bucket", "count_bucket", "review_ratio_bucket",
        "actors_cast_overlap_bucket", "list_bucket", "text_bucket",

        "bucket_id", "cluster_id", "dist_to_center", "sample_role", "is_sampled",
        "sampling_priority_score",

        "error_type", "reason_short", "reason_detailed",
        "suggested_correct_value", "needs_human_review", "evidence_used",
    ]

    existing_cols = [c for c in preferred_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in existing_cols]
    df = df[existing_cols + other_cols]

    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            obj = {}
            for col in df.columns:
                obj[col] = to_json_safe(row[col])
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    summary = {
        "total_rows": int(len(df)),
        "error_count": int((df["label_binary"] == 1).sum()) if "label_binary" in df.columns else 0,
        "correct_count": int((df["label_binary"] == 0).sum()) if "label_binary" in df.columns else 0,
        "label_source_counter": df["label_source"].value_counts(dropna=False).to_dict() if "label_source" in df.columns else {},
        "outside_candidate_count": int((df["is_outside_candidate"] == 1).sum()) if "is_outside_candidate" in df.columns else 0,
        "inside_candidate_count": int((df["is_outside_candidate"] == 0).sum()) if "is_outside_candidate" in df.columns else 0,
        "column_counter_top20": df["column"].value_counts(dropna=False).head(20).to_dict() if "column" in df.columns else {},
        "label_by_source": (
            df.groupby(["label_source", "label"]).size().reset_index(name="count").to_dict("records")
            if "label_source" in df.columns and "label" in df.columns else []
        ),
        "config": {
            "MAX_FINAL_TRAIN_ROWS": MAX_FINAL_TRAIN_ROWS,
            "MAX_PROPAGATED_TRAIN_ROWS": MAX_PROPAGATED_TRAIN_ROWS,
            "MAX_OUTSIDE_CLEAN_TOTAL": MAX_OUTSIDE_CLEAN_TOTAL,
            "MAX_CORRECT_TO_ERROR_RATIO": MAX_CORRECT_TO_ERROR_RATIO,
            "MAX_OUTSIDE_CLEAN_TO_ERROR_RATIO": MAX_OUTSIDE_CLEAN_TO_ERROR_RATIO,
            "CLUSTER_PROP_MIN_CONF": CLUSTER_PROP_MIN_CONF,
            "KNN_PROP_MIN_CONF": KNN_PROP_MIN_CONF,
        }
    }

    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


# ============================================================
# 10. 主流程
# ============================================================

def main():
    print("[1/6] Loading data ...")
    df_table = load_table(INPUT_TABLE_CSV)
    clustered_df = load_clustered_csv(INPUT_CLUSTERED_CSV)
    labeled_df = load_labeled_csv(INPUT_LABELED_CSV)
    propagated_df = load_propagated_csv(INPUT_PROPAGATED_CSV)

    print("[2/6] Building candidate-side training set ...")
    candidate_train_df = build_candidate_training_from_labels(
        clustered_df=clustered_df,
        labeled_df=labeled_df,
        propagated_df=propagated_df
    )
    print(f"  candidate_train_df size = {len(candidate_train_df)}")

    if len(candidate_train_df) > 0:
        print("  candidate label source distribution:")
        print(candidate_train_df["label_source"].value_counts(dropna=False).to_string())
        print("  candidate label distribution:")
        print(candidate_train_df["label"].value_counts(dropna=False).to_string())

    print("[3/6] Building outside-clean training set ...")
    outside_clean_df = build_outside_clean_training_simple(
        df_table=df_table,
        clustered_df=clustered_df,
        candidate_train_df=candidate_train_df
    )
    print(f"  outside_clean_df size = {len(outside_clean_df)}")

    print("[4/6] Merging ...")
    final_df = pd.concat([candidate_train_df, outside_clean_df], ignore_index=True)

    if len(final_df) == 0:
        raise RuntimeError("最终训练集为空，请检查输入文件和阈值设置。")

    final_df = final_df.sort_values(
        by=["row_id", "column", "sample_weight", "propagation_confidence"],
        ascending=[True, True, False, False]
    )
    final_df = final_df.drop_duplicates(subset=["row_id", "column"], keep="first").copy()

    print(f"  merged size before balance = {len(final_df)}")

    print("[5/6] Balancing final train set ...")
    final_df = balance_final_training(final_df)
    print(f"  final size after balance = {len(final_df)}")

    print("[6/6] Exporting ...")
    export_final_train(
        df=final_df,
        out_csv=OUTPUT_FINAL_TRAIN_CSV,
        out_jsonl=OUTPUT_FINAL_TRAIN_JSONL,
        out_summary=OUTPUT_SUMMARY_JSON
    )

    print("\n===== DONE =====")
    print(f"Final train CSV   : {OUTPUT_FINAL_TRAIN_CSV}")
    print(f"Final train JSONL : {OUTPUT_FINAL_TRAIN_JSONL}")
    print(f"Summary JSON      : {OUTPUT_SUMMARY_JSON}")

    print("\n===== STATS =====")
    print(f"Total rows   : {len(final_df)}")
    print(f"Error rows   : {(final_df['label_binary'] == 1).sum()}")
    print(f"Correct rows : {(final_df['label_binary'] == 0).sum()}")

    print("\nLabel source distribution:")
    print(final_df["label_source"].value_counts(dropna=False).to_string())

    print("\nOutside candidate distribution:")
    print(final_df["is_outside_candidate"].value_counts(dropna=False).to_string())

    print("\nTop 20 columns:")
    print(final_df["column"].value_counts(dropna=False).head(20).to_string())


if __name__ == "__main__":
    main()

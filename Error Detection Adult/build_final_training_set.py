import json
from typing import Dict, Any, Optional, Set, Tuple, List

import numpy as np
import pandas as pd


# ============================================================
# 1. 用户配置区
# ============================================================

INPUT_TABLE_CSV = "/mnt/mydata/dq/projects/Splittree/adult/adult_dirty.csv"
INPUT_CLUSTERED_CSV = "candidate_clustered_adult.csv"
INPUT_LABELED_CSV = "candidate_sampled_labeled_adult.csv"
INPUT_PROPAGATED_CSV = "propagated_labels_adult.csv"

OUTPUT_FINAL_TRAIN_CSV = "final_train_dataset_adult.csv"
OUTPUT_FINAL_TRAIN_JSONL = "final_train_dataset_adult.jsonl"
OUTPUT_SUMMARY_JSON = "final_train_summary_adult.json"

MISSING_TOKENS = {
    "", "empty", "null", "none", "nan", "na", "n/a", "missing", "unk",
    "unknown", "?", "not available", "contact airline"
}

# -------------------------
# 候选内样本保留策略
# -------------------------

KEEP_ALL_LLM_LABELED = True

# adult 传播要更保守，和 label_propagation_adult.py 保持一致：
# 只有高置信传播标签才进入训练集。
CLUSTER_PROP_MIN_CONF = 0.90
KNN_PROP_MIN_CONF = 0.92

# 传播样本总量控制。None 表示不额外限制。
MAX_PROPAGATED_TRAIN_ROWS = 700
MAX_PROPAGATED_PER_COLUMN = 80

# -------------------------
# 候选外 clean 样本构造
# -------------------------

ENABLE_OUTSIDE_CLEAN = True
OUTSIDE_ONLY_FROM_CANDIDATE_COLUMNS = True

# adult 类别字段较多，不要过多加入 outside clean，否则模型会过度偏向 correct。
OUTSIDE_CLEAN_PER_COLUMN = 80
OUTSIDE_CLEAN_MIN_PER_COLUMN = 15
OUTSIDE_MIN_VALUE_FREQ = 2
OUTSIDE_MAX_VALUE_FREQ_RANK = 25
SKIP_NULL_IN_OUTSIDE_CLEAN = True

# 候选外 clean 总量限制。
MAX_OUTSIDE_CLEAN_TOTAL = 900
MAX_OUTSIDE_CLEAN_PER_COLUMN = {
    "age": 80,
    "workclass": 70,
    "education": 80,
    "maritalstatus": 70,
    "occupation": 80,
    "relationship": 80,
    "race": 60,
    "sex": 50,
    "hoursperweek": 80,
    "country": 80,
    "income": 60,
}
DEFAULT_MAX_OUTSIDE_CLEAN_PER_COLUMN = 70

# 对长尾列 outside clean 更谨慎。
OUTSIDE_LONG_TAIL_COLUMNS = {"country", "occupation", "workclass"}

# -------------------------
# 样本权重
# -------------------------

WEIGHT_LLM = 1.0
WEIGHT_CLUSTER_PROP = 0.75
WEIGHT_KNN_PROP = 0.60
WEIGHT_OUTSIDE_CLEAN = 0.35

# -------------------------
# 类别平衡
# -------------------------

ENABLE_BALANCE = True

# 训练集中 correct : error 的最大比例。
# adult 任务里候选外 clean 过多会拉低召回，因此默认 1.2。
MAX_CORRECT_TO_ERROR_RATIO = 1.2

# 如果 error 很少，至少保留一定 correct，防止训练单类。
MIN_CORRECT_KEEP = 80

PREFER_INTERNAL_CORRECT_FIRST = True

# -------------------------
# adult 专属字段
# -------------------------

ADULT_COLUMNS = [
    "age",
    "workclass",
    "education",
    "maritalstatus",
    "occupation",
    "relationship",
    "race",
    "sex",
    "hoursperweek",
    "country",
    "income",
]

ADULT_DOMAIN_COLUMNS = {
    "workclass",
    "education",
    "maritalstatus",
    "occupation",
    "relationship",
    "race",
    "sex",
    "country",
    "income",
}

ADULT_PROFILE_COLUMNS = {
    "age",
    "education",
    "maritalstatus",
    "relationship",
    "sex",
    "hoursperweek",
    "workclass",
    "occupation",
    "income",
}

RANDOM_STATE = 42


# ============================================================
# 2. 基础函数
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


def label_name(y: int) -> str:
    return "error" if int(y) == 1 else "correct"


def compute_value_frequency_rank(counter: Dict[Any, int], value: Any) -> Optional[int]:
    if value is None:
        return None
    items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    for idx, (v, _) in enumerate(items, start=1):
        if v == value:
            return idx
    return None


def make_json_safe(v):
    if pd.isna(v):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def build_key(row_id: Any, column: Any) -> Tuple[int, str]:
    return int(row_id), str(column)


def sort_and_dedup_by_priority(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return df

    df = df.copy()

    source_priority = {
        "llm": 5,
        "cluster_propagation": 4,
        "knn_propagation": 3,
        "outside_clean": 1,
    }
    df["_source_priority"] = df["label_source"].map(lambda x: source_priority.get(str(x), 0))
    df["_conf_sort"] = pd.to_numeric(df.get("propagation_confidence", 0.0), errors="coerce").fillna(0.0)
    df["_weight_sort"] = pd.to_numeric(df.get("sample_weight", 0.0), errors="coerce").fillna(0.0)

    df = df.sort_values(
        by=["row_id", "column", "_source_priority", "_conf_sort", "_weight_sort"],
        ascending=[True, True, False, False, False],
    )
    df = df.drop_duplicates(subset=["row_id", "column"], keep="first").copy()
    df = df.drop(columns=["_source_priority", "_conf_sort", "_weight_sort"])
    return df


# ============================================================
# 3. 数据加载
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

    expected_cols = {
        "value": None,
        "semantic_type": "unknown",
        "detected_type": "unknown",
        "violation_count": 0,
        "conflict_score": 0.0,
        "value_frequency": 0,
        "value_frequency_rank": np.nan,

        "neighbor_majority_value": None,
        "neighbor_majority_ratio": np.nan,
        "prior_error_probability": 0.0,
        "posterior_error_probability": 0.0,

        "main_rule_type": "none",
        "main_usage_role": "none",
        "strong_rule_count": 0,
        "candidate_generation_rule_count": 0,
        "fd_like_count": 0,
        "context_rule_count": 0,
        "global_rule_count": 0,
        "rare_value_count": 0,
        "typo_rule_count": 0,
        "pattern_rule_count": 0,
        "schema_rule_count": 0,

        "adult_domain_rule_count": 0,
        "adult_format_rule_count": 0,
        "adult_consistency_rule_count": 0,

        "age_lower": np.nan,
        "age_upper": np.nan,
        "hours_lower": np.nan,
        "hours_upper": np.nan,
        "canonical_income": None,
        "domain_valid": None,
        "education_order": np.nan,
        "adult_value_bucket": "unknown",
        "adult_profile_inconsistency_count": 0,
        "adult_consistency_score": 0.0,
        "adult_evidence_flags": "",

        "pattern_bucket": "unknown",
        "rarity_bucket": "unknown",
        "neighbor_bucket": "unknown",
        "domain_bucket": "unknown",
        "adult_consistency_bucket": "unknown",
        "age_sampling_bucket": "unknown",
        "hours_sampling_bucket": "unknown",

        # 兼容旧脚本字段
        "time_window_rule_count": 0,
        "time_value_minutes": np.nan,
        "time_window_bucket": "adult_not_applicable",
        "time_value_bucket": "adult_not_applicable",

        "bucket_id": "unknown",
        "cluster_id": -1,
        "dist_to_center": 0.0,
        "sample_role": "",
        "is_sampled": 0,
        "signal_priority": 0.0,
    }

    for col, default_val in expected_cols.items():
        if col not in df.columns:
            df[col] = default_val

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

        "semantic_type": "unknown",
        "detected_type": "unknown",
        "main_rule_type": "none",
        "main_usage_role": "none",
        "rare_value_count": 0,
        "typo_rule_count": 0,
        "pattern_rule_count": 0,
        "schema_rule_count": 0,

        "adult_domain_rule_count": 0,
        "adult_format_rule_count": 0,
        "adult_consistency_rule_count": 0,

        "age_lower": np.nan,
        "age_upper": np.nan,
        "hours_lower": np.nan,
        "hours_upper": np.nan,
        "canonical_income": None,
        "domain_valid": None,
        "education_order": np.nan,
        "adult_value_bucket": "unknown",
        "adult_profile_inconsistency_count": 0,
        "adult_consistency_score": 0.0,
        "adult_evidence_flags": "",

        "time_window_rule_count": 0,
        "time_value_minutes": np.nan,
        "time_window_bucket": "adult_not_applicable",
        "time_value_bucket": "adult_not_applicable",
    }
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
        "value": None,
        "error_type": None,
        "reason_short": None,
        "reason_detailed": None,
        "suggested_correct_value": None,
        "needs_human_review": None,
        "evidence_used": None,

        "semantic_type": "unknown",
        "detected_type": "unknown",
        "main_rule_type": "none",
        "main_usage_role": "none",
        "pattern_bucket": "unknown",
        "rarity_bucket": "unknown",
        "neighbor_bucket": "unknown",
        "domain_bucket": "unknown",
        "adult_consistency_bucket": "unknown",
        "age_sampling_bucket": "unknown",
        "hours_sampling_bucket": "unknown",
        "adult_value_bucket": "unknown",

        "violation_count": 0,
        "conflict_score": 0.0,
        "value_frequency": 0,
        "value_frequency_rank": np.nan,
        "neighbor_majority_value": None,
        "neighbor_majority_ratio": np.nan,
        "prior_error_probability": 0.0,
        "posterior_error_probability": 0.0,

        "strong_rule_count": 0,
        "candidate_generation_rule_count": 0,
        "fd_like_count": 0,
        "context_rule_count": 0,
        "global_rule_count": 0,
        "rare_value_count": 0,
        "typo_rule_count": 0,
        "pattern_rule_count": 0,
        "schema_rule_count": 0,

        "adult_domain_rule_count": 0,
        "adult_format_rule_count": 0,
        "adult_consistency_rule_count": 0,

        "age_lower": np.nan,
        "age_upper": np.nan,
        "hours_lower": np.nan,
        "hours_upper": np.nan,
        "canonical_income": None,
        "domain_valid": None,
        "education_order": np.nan,
        "adult_profile_inconsistency_count": 0,
        "adult_consistency_score": 0.0,
        "adult_evidence_flags": "",

        # 兼容旧字段
        "time_window_rule_count": 0,
        "time_value_minutes": np.nan,
        "time_window_bucket": "adult_not_applicable",
        "time_value_bucket": "adult_not_applicable",

        "bucket_id": "unknown",
        "cluster_id": -1,
        "dist_to_center": 0.0,
        "sample_role": "",
        "is_sampled": 0,
        "signal_priority": 0.0,
    }
    for col, default_val in expected_cols.items():
        if col not in df.columns:
            df[col] = default_val

    return df


# ============================================================
# 4. 特征列补齐
# ============================================================

ADULT_FEATURE_COLS = [
    "semantic_type",
    "detected_type",
    "violation_count",
    "conflict_score",
    "value_frequency",
    "value_frequency_rank",
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

    "age_lower",
    "age_upper",
    "hours_lower",
    "hours_upper",
    "canonical_income",
    "domain_valid",
    "education_order",
    "adult_value_bucket",
    "adult_profile_inconsistency_count",
    "adult_consistency_score",
    "adult_evidence_flags",

    "pattern_bucket",
    "rarity_bucket",
    "neighbor_bucket",
    "domain_bucket",
    "adult_consistency_bucket",
    "age_sampling_bucket",
    "hours_sampling_bucket",

    # 兼容旧字段
    "time_window_rule_count",
    "time_value_minutes",
    "time_window_bucket",
    "time_value_bucket",

    "bucket_id",
    "cluster_id",
    "dist_to_center",
    "sample_role",
    "is_sampled",
    "signal_priority",
]


def merge_extra_features_from_row(out: Dict[str, Any], row: pd.Series):
    for col in ADULT_FEATURE_COLS:
        if col in row.index and pd.notna(row.get(col)):
            out[col] = row.get(col)
    return out


def fill_missing_feature_defaults(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return df

    df = df.copy()

    defaults = {
        "semantic_type": "unknown",
        "detected_type": "unknown",
        "violation_count": 0,
        "conflict_score": 0.0,
        "value_frequency": 0,
        "value_frequency_rank": np.nan,
        "neighbor_majority_value": None,
        "neighbor_majority_ratio": np.nan,
        "prior_error_probability": 0.0,
        "posterior_error_probability": 0.0,

        "main_rule_type": "none",
        "main_usage_role": "none",
        "strong_rule_count": 0,
        "candidate_generation_rule_count": 0,
        "fd_like_count": 0,
        "context_rule_count": 0,
        "global_rule_count": 0,
        "rare_value_count": 0,
        "typo_rule_count": 0,
        "pattern_rule_count": 0,
        "schema_rule_count": 0,

        "adult_domain_rule_count": 0,
        "adult_format_rule_count": 0,
        "adult_consistency_rule_count": 0,

        "age_lower": np.nan,
        "age_upper": np.nan,
        "hours_lower": np.nan,
        "hours_upper": np.nan,
        "canonical_income": None,
        "domain_valid": None,
        "education_order": np.nan,
        "adult_value_bucket": "unknown",
        "adult_profile_inconsistency_count": 0,
        "adult_consistency_score": 0.0,
        "adult_evidence_flags": "",

        "pattern_bucket": "unknown",
        "rarity_bucket": "unknown",
        "neighbor_bucket": "unknown",
        "domain_bucket": "unknown",
        "adult_consistency_bucket": "unknown",
        "age_sampling_bucket": "unknown",
        "hours_sampling_bucket": "unknown",

        "time_window_rule_count": 0,
        "time_value_minutes": np.nan,
        "time_window_bucket": "adult_not_applicable",
        "time_value_bucket": "adult_not_applicable",

        "bucket_id": "unknown",
        "cluster_id": -1,
        "dist_to_center": 0.0,
        "sample_role": "",
        "is_sampled": 0,
        "signal_priority": 0.0,
    }

    for col, default_val in defaults.items():
        if col not in df.columns:
            df[col] = default_val

    return df


# ============================================================
# 5. 候选内训练集构造
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

    # 5.1 LLM 直接标注样本
    if KEEP_ALL_LLM_LABELED:
        for _, row in labeled_df.iterrows():
            key = (int(row["row_id"]), str(row["column"]))

            base = dict(clustered_map[key]) if key in clustered_map else {}

            label_binary = int(row["label_binary_norm"])
            label = label_name(label_binary)
            confidence = safe_float(row.get("confidence", 1.0), 1.0)

            out = dict(base)
            out["row_id"] = key[0]
            out["column"] = key[1]
            out["value"] = row.get("value", base.get("value"))
            out["label_binary"] = label_binary
            out["label"] = label
            out["label_source"] = "llm"
            out["propagation_confidence"] = confidence
            out["sample_weight"] = WEIGHT_LLM
            out["is_outside_candidate"] = 0

            for extra_col in [
                "error_type",
                "reason_short",
                "reason_detailed",
                "suggested_correct_value",
                "needs_human_review",
                "evidence_used",
            ]:
                out[extra_col] = row.get(extra_col) if extra_col in row.index else None

            out = merge_extra_features_from_row(out, row)
            train_rows.append(out)

    # 5.2 传播样本：去掉已被 LLM 直接标注的
    llm_keys = set((int(r["row_id"]), str(r["column"])) for _, r in labeled_df.iterrows())

    prop_rows = []

    for _, row in propagated_df.iterrows():
        key = (int(row["row_id"]), str(row["column"]))
        if key in llm_keys:
            continue

        label_source = str(row.get("label_source", "")).strip()
        if label_source not in {"cluster_propagation", "knn_propagation"}:
            continue

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

        base = dict(clustered_map[key]) if key in clustered_map else {}

        out = dict(base)
        out["row_id"] = key[0]
        out["column"] = key[1]
        out["value"] = row.get("value", base.get("value"))
        out["label_binary"] = int(label_binary)
        out["label"] = label_name(label_binary)
        out["label_source"] = label_source
        out["propagation_confidence"] = prop_conf
        out["sample_weight"] = sample_weight
        out["is_outside_candidate"] = 0

        for extra_col in [
            "error_type",
            "reason_short",
            "reason_detailed",
            "suggested_correct_value",
            "needs_human_review",
            "evidence_used",
        ]:
            out[extra_col] = row.get(extra_col) if extra_col in row.index else None

        out = merge_extra_features_from_row(out, row)
        prop_rows.append(out)

    prop_df = pd.DataFrame(prop_rows)
    if len(prop_df) > 0:
        prop_df = prop_df.copy()
        prop_df["_prop_priority"] = (
            pd.to_numeric(prop_df["propagation_confidence"], errors="coerce").fillna(0.0) * 5.0
            + pd.to_numeric(prop_df.get("strong_rule_count", 0), errors="coerce").fillna(0.0) * 2.0
            + pd.to_numeric(prop_df.get("adult_domain_rule_count", 0), errors="coerce").fillna(0.0) * 1.8
            + pd.to_numeric(prop_df.get("adult_format_rule_count", 0), errors="coerce").fillna(0.0) * 1.6
            + pd.to_numeric(prop_df.get("adult_consistency_rule_count", 0), errors="coerce").fillna(0.0) * 1.4
            + pd.to_numeric(prop_df.get("conflict_score", 0), errors="coerce").fillna(0.0) / 20.0
        )

        if MAX_PROPAGATED_PER_COLUMN is not None:
            kept_parts = []
            for col, g in prop_df.groupby("column", sort=False):
                if len(g) > MAX_PROPAGATED_PER_COLUMN:
                    g = g.sort_values(
                        by=["_prop_priority", "propagation_confidence", "conflict_score"],
                        ascending=[False, False, False],
                    ).head(MAX_PROPAGATED_PER_COLUMN)
                kept_parts.append(g)
            prop_df = pd.concat(kept_parts, ignore_index=True) if kept_parts else prop_df.iloc[0:0]

        if MAX_PROPAGATED_TRAIN_ROWS is not None and len(prop_df) > MAX_PROPAGATED_TRAIN_ROWS:
            prop_df = prop_df.sort_values(
                by=["_prop_priority", "propagation_confidence", "conflict_score"],
                ascending=[False, False, False],
            ).head(MAX_PROPAGATED_TRAIN_ROWS)

        prop_df = prop_df.drop(columns=["_prop_priority"], errors="ignore")
        train_rows.extend(prop_df.to_dict(orient="records"))

    df = pd.DataFrame(train_rows)
    if len(df) == 0:
        return df

    df = fill_missing_feature_defaults(df)
    df = sort_and_dedup_by_priority(df)
    return df


# ============================================================
# 6. 候选外 clean 样本构造
# ============================================================

def outside_clean_priority(col: str, value_freq: int, value_rank: Optional[int]) -> float:
    rank = value_rank if value_rank is not None and not pd.isna(value_rank) else 999999
    score = float(value_freq) - 0.05 * float(rank)

    if col in OUTSIDE_LONG_TAIL_COLUMNS:
        score -= 1.0

    return score


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
                "_outside_priority": outside_clean_priority(col, value_freq, value_rank),
            })

        if len(pool_rows) == 0:
            continue

        pool_df = pd.DataFrame(pool_rows)
        pool_df = pool_df.sort_values(
            by=["_outside_priority", "value_frequency", "value_frequency_rank", "row_id"],
            ascending=[False, False, True, True],
        ).copy()

        col_limit = MAX_OUTSIDE_CLEAN_PER_COLUMN.get(col, DEFAULT_MAX_OUTSIDE_CLEAN_PER_COLUMN)
        target_n = max(OUTSIDE_CLEAN_MIN_PER_COLUMN, min(OUTSIDE_CLEAN_PER_COLUMN, col_limit, len(pool_df)))
        selected_df = pool_df.head(target_n).copy()

        for _, row in selected_df.iterrows():
            outside_rows.append({
                "row_id": int(row["row_id"]),
                "column": str(row["column"]),
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

                "adult_domain_rule_count": 0,
                "adult_format_rule_count": 0,
                "adult_consistency_rule_count": 0,

                "age_lower": np.nan,
                "age_upper": np.nan,
                "hours_lower": np.nan,
                "hours_upper": np.nan,
                "canonical_income": None,
                "domain_valid": None,
                "education_order": np.nan,
                "adult_value_bucket": "outside_clean_unknown",
                "adult_profile_inconsistency_count": 0,
                "adult_consistency_score": 0.0,
                "adult_evidence_flags": "",

                "pattern_bucket": "outside_clean_unknown",
                "rarity_bucket": "outside_clean_unknown",
                "neighbor_bucket": "outside_clean_unknown",
                "domain_bucket": "outside_clean_unknown",
                "adult_consistency_bucket": "outside_clean_unknown",
                "age_sampling_bucket": "outside_clean_unknown",
                "hours_sampling_bucket": "outside_clean_unknown",

                "time_window_rule_count": 0,
                "time_value_minutes": np.nan,
                "time_window_bucket": "adult_not_applicable",
                "time_value_bucket": "adult_not_applicable",

                "bucket_id": "outside_clean",
                "cluster_id": -1,
                "dist_to_center": 0.0,
                "sample_role": "outside_clean",
                "is_sampled": 0,
                "signal_priority": 0.0,

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
                    "as a likely clean training example for Adult dataset."
                ),
                "suggested_correct_value": row["value"],
                "needs_human_review": False,
                "evidence_used": "outside_candidate_sampling",
            })

    outside_df = pd.DataFrame(outside_rows)

    if len(outside_df) == 0:
        return outside_df

    # 先按列限制
    kept_parts = []
    for col, g in outside_df.groupby("column", sort=False):
        col_limit = MAX_OUTSIDE_CLEAN_PER_COLUMN.get(col, DEFAULT_MAX_OUTSIDE_CLEAN_PER_COLUMN)
        if len(g) > col_limit:
            g = g.sort_values(
                by=["value_frequency", "value_frequency_rank", "row_id"],
                ascending=[False, True, True],
            ).head(col_limit)
        kept_parts.append(g)
    outside_df = pd.concat(kept_parts, ignore_index=True) if kept_parts else outside_df.iloc[0:0]

    # 再按候选 error 数和总量限制
    if candidate_train_error_count > 0:
        max_by_ratio = int(candidate_train_error_count * MAX_CORRECT_TO_ERROR_RATIO)
        max_outside_clean_total = min(MAX_OUTSIDE_CLEAN_TOTAL, max_by_ratio)
    else:
        max_outside_clean_total = min(MAX_OUTSIDE_CLEAN_TOTAL, len(outside_df))

    if len(outside_df) > max_outside_clean_total:
        outside_df = outside_df.sample(n=max_outside_clean_total, random_state=RANDOM_STATE)

    outside_df = fill_missing_feature_defaults(outside_df)
    return outside_df


# ============================================================
# 7. 平衡训练集
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
        final_df = df.copy()
        return final_df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    if PREFER_INTERNAL_CORRECT_FIRST:
        internal_correct = correct_df[correct_df["is_outside_candidate"] == 0].copy()
        outside_correct = correct_df[correct_df["is_outside_candidate"] == 1].copy()

        keep_parts = []
        remain = max_correct

        if len(internal_correct) > 0:
            internal_correct = internal_correct.sort_values(
                by=["sample_weight", "propagation_confidence", "conflict_score"],
                ascending=[False, False, False],
            )
            keep_internal = internal_correct.head(remain)
            keep_parts.append(keep_internal)
            remain -= len(keep_internal)

        if remain > 0 and len(outside_correct) > 0:
            outside_correct = outside_correct.sort_values(
                by=["sample_weight", "propagation_confidence", "value_frequency"],
                ascending=[False, False, False],
            )
            keep_outside = outside_correct.head(remain)
            keep_parts.append(keep_outside)

        correct_keep_df = pd.concat(keep_parts, ignore_index=True) if keep_parts else pd.DataFrame(columns=df.columns)
    else:
        correct_keep_df = correct_df.sample(n=max_correct, random_state=RANDOM_STATE)

    final_df = pd.concat([error_df, correct_keep_df], ignore_index=True)
    final_df = final_df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    return final_df


# ============================================================
# 8. 导出
# ============================================================

def export_final_train(df: pd.DataFrame, out_csv: str, out_jsonl: str, out_summary: str):
    df = df.copy()
    df = fill_missing_feature_defaults(df)

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

        "adult_domain_rule_count", "adult_format_rule_count", "adult_consistency_rule_count",

        "age_lower", "age_upper", "hours_lower", "hours_upper",
        "canonical_income", "domain_valid", "education_order",
        "adult_value_bucket", "adult_profile_inconsistency_count",
        "adult_consistency_score", "adult_evidence_flags",

        "pattern_bucket", "rarity_bucket", "neighbor_bucket",
        "domain_bucket", "adult_consistency_bucket",
        "age_sampling_bucket", "hours_sampling_bucket",

        # 兼容旧字段
        "time_window_rule_count", "time_value_minutes",
        "time_window_bucket", "time_value_bucket",

        "bucket_id", "cluster_id", "dist_to_center", "sample_role", "is_sampled", "signal_priority",

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
                obj[col] = make_json_safe(row[col])
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    summary = {
        "total_rows": int(len(df)),
        "error_count": int((df["label_binary"] == 1).sum()) if "label_binary" in df.columns else 0,
        "correct_count": int((df["label_binary"] == 0).sum()) if "label_binary" in df.columns else 0,
        "error_ratio": float((df["label_binary"] == 1).mean()) if "label_binary" in df.columns and len(df) > 0 else None,
        "label_source_counter": df["label_source"].value_counts(dropna=False).to_dict() if "label_source" in df.columns else {},
        "outside_candidate_count": int((df["is_outside_candidate"] == 1).sum()) if "is_outside_candidate" in df.columns else 0,
        "inside_candidate_count": int((df["is_outside_candidate"] == 0).sum()) if "is_outside_candidate" in df.columns else 0,
        "column_counter_top20": df["column"].value_counts(dropna=False).head(20).to_dict() if "column" in df.columns else {},
        "column_label_source_distribution": (
            df.groupby(["column", "label_source"]).size().reset_index(name="count").to_dict(orient="records")
            if "column" in df.columns and "label_source" in df.columns else []
        ),
        "column_label_distribution": (
            df.groupby(["column", "label"]).size().reset_index(name="count").to_dict(orient="records")
            if "column" in df.columns and "label" in df.columns else []
        ),
        "config": {
            "CLUSTER_PROP_MIN_CONF": CLUSTER_PROP_MIN_CONF,
            "KNN_PROP_MIN_CONF": KNN_PROP_MIN_CONF,
            "MAX_PROPAGATED_TRAIN_ROWS": MAX_PROPAGATED_TRAIN_ROWS,
            "MAX_PROPAGATED_PER_COLUMN": MAX_PROPAGATED_PER_COLUMN,
            "ENABLE_OUTSIDE_CLEAN": ENABLE_OUTSIDE_CLEAN,
            "OUTSIDE_CLEAN_PER_COLUMN": OUTSIDE_CLEAN_PER_COLUMN,
            "MAX_OUTSIDE_CLEAN_TOTAL": MAX_OUTSIDE_CLEAN_TOTAL,
            "MAX_CORRECT_TO_ERROR_RATIO": MAX_CORRECT_TO_ERROR_RATIO,
            "MIN_CORRECT_KEEP": MIN_CORRECT_KEEP,
        }
    }

    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


# ============================================================
# 9. 主流程
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

    final_df = fill_missing_feature_defaults(final_df)

    final_df = final_df.sort_values(
        by=["row_id", "column", "sample_weight", "propagation_confidence"],
        ascending=[True, True, False, False],
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

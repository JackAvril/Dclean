import json
from typing import Dict, Any, Optional, Set, Tuple

import numpy as np
import pandas as pd


INPUT_TABLE_CSV = "/mnt/mydata/dq/projects/Splittree/rayyan/rayyan_dirty.csv"
INPUT_CLUSTERED_CSV = "candidate_clustered_rayyan.csv"
INPUT_LABELED_CSV = "candidate_sampled_labeled_rayyan.csv"
INPUT_PROPAGATED_CSV = "propagated_labels_rayyan.csv"

OUTPUT_FINAL_TRAIN_CSV = "final_train_dataset_rayyan.csv"
OUTPUT_FINAL_TRAIN_JSONL = "final_train_dataset_rayyan.jsonl"
OUTPUT_SUMMARY_JSON = "final_train_summary_rayyan.json"

MISSING_TOKENS = {
    "", "empty", "null", "none", "nan", "na", "n/a", "missing", "unk",
    "not available", "contact airline"
}

KEEP_ALL_LLM_LABELED = True
CLUSTER_PROP_MIN_CONF = 0.90
KNN_PROP_MIN_CONF = 0.95

ENABLE_OUTSIDE_CLEAN = True
OUTSIDE_ONLY_FROM_CANDIDATE_COLUMNS = True
OUTSIDE_CLEAN_PER_COLUMN = 120
OUTSIDE_CLEAN_MIN_PER_COLUMN = 20
OUTSIDE_MIN_VALUE_FREQ = 2
OUTSIDE_MAX_VALUE_FREQ_RANK = 20
SKIP_NULL_IN_OUTSIDE_CLEAN = True

WEIGHT_LLM = 1.0
WEIGHT_CLUSTER_PROP = 0.8
WEIGHT_KNN_PROP = 0.6
WEIGHT_OUTSIDE_CLEAN = 0.4

ENABLE_BALANCE = True
MAX_CORRECT_TO_ERROR_RATIO = 1.5
PREFER_INTERNAL_CORRECT_FIRST = True

RANDOM_STATE = 42


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
        "violation_count": 0,
        "conflict_score": 0.0,
        "value_frequency": 0,
        "value_frequency_rank": np.nan,
        "numeric_value": np.nan,
        "date_value": None,
        "language_canonical": None,
        "issn_canonical": None,
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
        "canonical_rule_count": 0,
        "pattern_bucket": "unknown",
        "rarity_bucket": "unknown",
        "neighbor_bucket": "unknown",
        "canonical_bucket": "unknown",
        "date_value_bucket": "unknown",
        "language_bucket": "unknown",
        "bucket_id": "unknown",
        "cluster_id": -1,
        "dist_to_center": 0.0,
        "sample_role": "",
        "is_sampled": 0,
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
        "main_usage_role": "none",
        "rare_value_count": 0,
        "typo_rule_count": 0,
        "pattern_rule_count": 0,
        "schema_rule_count": 0,
        "canonical_rule_count": 0,
        "numeric_value": np.nan,
        "date_value": None,
        "language_canonical": None,
        "issn_canonical": None,
        "canonical_bucket": "unknown",
        "date_value_bucket": "unknown",
        "language_bucket": "unknown",
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
        "error_type": None,
        "reason_short": None,
        "reason_detailed": None,
        "suggested_correct_value": None,
        "needs_human_review": None,
        "evidence_used": None,
        "semantic_type": "unknown",
        "main_rule_type": "none",
        "main_usage_role": "none",
        "pattern_bucket": "unknown",
        "rarity_bucket": "unknown",
        "neighbor_bucket": "unknown",
        "canonical_bucket": "unknown",
        "date_value_bucket": "unknown",
        "language_bucket": "unknown",
        "violation_count": 0,
        "conflict_score": 0.0,
        "value_frequency": 0,
        "value_frequency_rank": np.nan,
        "numeric_value": np.nan,
        "date_value": None,
        "language_canonical": None,
        "issn_canonical": None,
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
        "canonical_rule_count": 0,
        "bucket_id": "unknown",
        "cluster_id": -1,
        "dist_to_center": 0.0,
        "sample_role": "",
        "is_sampled": 0,
    }
    for col, default_val in expected_cols.items():
        if col not in df.columns:
            df[col] = default_val
    return df


def build_candidate_training_from_labels(
    clustered_df: pd.DataFrame,
    labeled_df: pd.DataFrame,
    propagated_df: pd.DataFrame
) -> pd.DataFrame:
    clustered_map = {}
    for _, row in clustered_df.iterrows():
        clustered_map[(int(row["row_id"]), str(row["column"]))] = row.to_dict()

    train_rows = []

    if KEEP_ALL_LLM_LABELED:
        for _, row in labeled_df.iterrows():
            key = (int(row["row_id"]), str(row["column"]))
            if key not in clustered_map:
                continue

            base = dict(clustered_map[key])
            label_binary = int(row["label_binary_norm"])
            label = "error" if label_binary == 1 else "correct"
            confidence = safe_float(row.get("confidence", 1.0), 1.0)

            out = dict(base)
            out["label_binary"] = label_binary
            out["label"] = label
            out["label_source"] = "llm"
            out["propagation_confidence"] = confidence
            out["sample_weight"] = WEIGHT_LLM
            out["is_outside_candidate"] = 0

            for extra_col in [
                "error_type", "reason_short", "reason_detailed",
                "suggested_correct_value", "needs_human_review", "evidence_used",
            ]:
                out[extra_col] = row.get(extra_col) if extra_col in row else None

            for extra_col in [
                "main_usage_role", "rare_value_count", "typo_rule_count",
                "pattern_rule_count", "schema_rule_count", "canonical_rule_count",
                "numeric_value", "date_value", "language_canonical",
                "issn_canonical", "canonical_bucket", "date_value_bucket", "language_bucket",
            ]:
                if extra_col in row and pd.notna(row.get(extra_col)):
                    out[extra_col] = row.get(extra_col)

            train_rows.append(out)

    llm_keys = set((int(r["row_id"]), str(r["column"])) for _, r in labeled_df.iterrows())

    for _, row in propagated_df.iterrows():
        key = (int(row["row_id"]), str(row["column"]))
        if key in llm_keys or key not in clustered_map:
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

        for extra_col in [
            "semantic_type", "violation_count", "conflict_score",
            "value_frequency", "value_frequency_rank",
            "numeric_value", "date_value", "language_canonical", "issn_canonical",
            "neighbor_majority_value", "neighbor_majority_ratio",
            "prior_error_probability", "posterior_error_probability",
            "main_rule_type", "main_usage_role", "strong_rule_count",
            "candidate_generation_rule_count", "fd_like_count", "context_rule_count",
            "global_rule_count", "rare_value_count", "typo_rule_count",
            "pattern_rule_count", "schema_rule_count", "canonical_rule_count",
            "pattern_bucket", "rarity_bucket", "neighbor_bucket",
            "canonical_bucket", "date_value_bucket", "language_bucket",
            "bucket_id", "cluster_id", "dist_to_center", "sample_role", "is_sampled",
        ]:
            if extra_col in row and pd.notna(row.get(extra_col)):
                out[extra_col] = row.get(extra_col)

        for extra_col in [
            "error_type", "reason_short", "reason_detailed",
            "suggested_correct_value", "needs_human_review", "evidence_used",
        ]:
            out[extra_col] = row.get(extra_col) if extra_col in row else None

        train_rows.append(out)

    df = pd.DataFrame(train_rows)
    if len(df) == 0:
        return df

    source_priority = {"llm": 3, "cluster_propagation": 2, "knn_propagation": 1}
    df["_source_priority"] = df["label_source"].map(lambda x: source_priority.get(str(x), 0))
    df = df.sort_values(
        by=["row_id", "column", "_source_priority", "propagation_confidence"],
        ascending=[True, True, False, False]
    )
    df = df.drop_duplicates(subset=["row_id", "column"], keep="first").copy()
    df = df.drop(columns=["_source_priority"])
    return df


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
            outside_rows.append({
                "row_id": int(row["row_id"]),
                "column": str(row["column"]),
                "value": row["value"],

                "semantic_type": "outside_clean_unknown",
                "violation_count": 0,
                "conflict_score": 0.0,
                "value_frequency": safe_int(row.get("value_frequency", 0), 0),
                "value_frequency_rank": row.get("value_frequency_rank", np.nan),
                "numeric_value": np.nan,
                "date_value": None,
                "language_canonical": None,
                "issn_canonical": None,
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
                "canonical_rule_count": 0,

                "pattern_bucket": "outside_clean_unknown",
                "rarity_bucket": "outside_clean_unknown",
                "neighbor_bucket": "outside_clean_unknown",
                "canonical_bucket": "outside_clean_unknown",
                "date_value_bucket": "outside_clean_unknown",
                "language_bucket": "outside_clean_unknown",

                "bucket_id": "outside_clean",
                "cluster_id": -1,
                "dist_to_center": 0.0,
                "sample_role": "outside_clean",
                "is_sampled": 0,

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
                    "as a likely clean training example."
                ),
                "suggested_correct_value": row["value"],
                "needs_human_review": False,
                "evidence_used": "outside_candidate_sampling",
            })

    outside_df = pd.DataFrame(outside_rows)

    if len(outside_df) == 0:
        return outside_df

    if candidate_train_error_count > 0:
        max_outside_clean_total = int(candidate_train_error_count * MAX_CORRECT_TO_ERROR_RATIO)
        if len(outside_df) > max_outside_clean_total:
            outside_df = outside_df.sample(n=max_outside_clean_total, random_state=RANDOM_STATE)

    return outside_df


def balance_final_training(df: pd.DataFrame) -> pd.DataFrame:
    if (not ENABLE_BALANCE) or len(df) == 0:
        return df

    error_df = df[df["label_binary"] == 1].copy()
    correct_df = df[df["label_binary"] == 0].copy()

    if len(error_df) == 0 or len(correct_df) == 0:
        return df

    max_correct = int(len(error_df) * MAX_CORRECT_TO_ERROR_RATIO)
    if len(correct_df) <= max_correct:
        return df

    if PREFER_INTERNAL_CORRECT_FIRST:
        internal_correct = correct_df[correct_df["is_outside_candidate"] == 0].copy()
        outside_correct = correct_df[correct_df["is_outside_candidate"] == 1].copy()

        keep_parts = []
        remain = max_correct

        if len(internal_correct) > 0:
            internal_correct = internal_correct.sort_values(
                by=["sample_weight", "propagation_confidence", "conflict_score"],
                ascending=[False, False, False]
            )
            keep_internal = internal_correct.head(remain)
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
    final_df = final_df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    return final_df


def export_final_train(df: pd.DataFrame, out_csv: str, out_jsonl: str, out_summary: str):
    df = df.copy()

    preferred_cols = [
        "row_id", "column", "value",
        "label", "label_binary", "label_source", "sample_weight", "is_outside_candidate",
        "propagation_confidence",

        "semantic_type",
        "violation_count", "conflict_score",
        "value_frequency", "value_frequency_rank",
        "numeric_value", "date_value", "language_canonical", "issn_canonical",
        "neighbor_majority_value", "neighbor_majority_ratio",
        "prior_error_probability", "posterior_error_probability",

        "main_rule_type", "main_usage_role",
        "strong_rule_count", "candidate_generation_rule_count",
        "fd_like_count", "context_rule_count", "global_rule_count",
        "rare_value_count", "typo_rule_count", "pattern_rule_count", "schema_rule_count",
        "canonical_rule_count",

        "pattern_bucket", "rarity_bucket", "neighbor_bucket",
        "canonical_bucket", "date_value_bucket", "language_bucket",
        "bucket_id", "cluster_id", "dist_to_center", "sample_role", "is_sampled",

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
                val = row[col]
                if pd.isna(val):
                    obj[col] = None
                elif isinstance(val, (np.integer,)):
                    obj[col] = int(val)
                elif isinstance(val, (np.floating,)):
                    obj[col] = float(val)
                else:
                    obj[col] = val
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    summary = {
        "total_rows": int(len(df)),
        "error_count": int((df["label_binary"] == 1).sum()) if "label_binary" in df.columns else 0,
        "correct_count": int((df["label_binary"] == 0).sum()) if "label_binary" in df.columns else 0,
        "label_source_counter": df["label_source"].value_counts(dropna=False).to_dict() if "label_source" in df.columns else {},
        "outside_candidate_count": int((df["is_outside_candidate"] == 1).sum()) if "is_outside_candidate" in df.columns else 0,
        "inside_candidate_count": int((df["is_outside_candidate"] == 0).sum()) if "is_outside_candidate" in df.columns else 0,
        "column_counter_top20": df["column"].value_counts(dropna=False).head(20).to_dict() if "column" in df.columns else {},
    }

    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


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

    print("\\n===== DONE =====")
    print(f"Final train CSV   : {OUTPUT_FINAL_TRAIN_CSV}")
    print(f"Final train JSONL : {OUTPUT_FINAL_TRAIN_JSONL}")
    print(f"Summary JSON      : {OUTPUT_SUMMARY_JSON}")

    print("\\n===== STATS =====")
    print(f"Total rows   : {len(final_df)}")
    print(f"Error rows   : {(final_df['label_binary'] == 1).sum()}")
    print(f"Correct rows : {(final_df['label_binary'] == 0).sum()}")

    print("\\nLabel source distribution:")
    print(final_df["label_source"].value_counts(dropna=False).to_string())

    print("\\nOutside candidate distribution:")
    print(final_df["is_outside_candidate"].value_counts(dropna=False).to_string())

    print("\\nTop 20 columns:")
    print(final_df["column"].value_counts(dropna=False).head(20).to_string())


if __name__ == "__main__":
    main()

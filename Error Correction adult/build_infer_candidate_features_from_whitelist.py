
"""
Build Adult inference candidate features from whitelist.

This script is adapted from the Flights whitelist builder, but removes
Flights-specific consensus metadata and keeps Adult-specific diagnostic fields.

Inputs
------
1. Candidate feature file:
   repair_candidate_features_v2.csv

2. Allowed cells / whitelist file. Supported formats:
   A. predicted_error_positions_adult_augmented.csv
      Output of revise_candidate_adult.py. Contains row_id,column and optional Adult fields.
   B. predicted_error_positions.csv
      Contains row_id,column and optional value/final_pred_label fields.
   C. inference_all_predictions.csv
      Contains final_pred_label / final_pred_label_name and will be filtered to error cells.

Output
------
repair_candidate_features_infer_whitelist.csv

Usage
-----
cd /mnt/mydata/dq/projects/Splittree/test/Error\ Correction\ adult

python build_infer_features_whitelist_adult.py \
  --candidate_features repair_candidate_features_v2.csv \
  --allowed_cells predicted_error_positions_adult_augmented.csv \
  --output repair_candidate_features_infer_whitelist.csv
"""

import argparse
import os
from pathlib import Path
from typing import List

import pandas as pd


# ============================================================
# 1. Defaults
# ============================================================

DEFAULT_INPUT_CANDIDATE_FEATURES = "repair_candidate_features_v2.csv"
DEFAULT_INPUT_ALLOWED_CELLS = "predicted_error_positions_adult_augmented.csv"
DEFAULT_OUTPUT_INFER_FEATURES = "repair_candidate_features_infer_whitelist.csv"

OUTPUT_MISSING_ALLOWED_CELLS = True
OUTPUT_FILTERED_OUT_FEATURES = True
OUTPUT_WHITELIST_SUMMARY = True

FALLBACK_ALLOWED_CELLS = [
    "predicted_error_positions_adult_augmented.csv",
    "predicted_error_positions.csv",
    "allowed_cells_from_detection.csv",
    "inference_all_predictions.csv",
    "./infer/inference_all_predictions.csv",
    "/mnt/mydata/dq/projects/Splittree/test/Error Detection Adult/active_cleaning_loop_runs_adult_v4/iter_3_infer/inference_all_predictions.csv",
    "/mnt/mydata/dq/projects/Splittree/test/Error Detection Adult/active_cleaning_loop_runs_adult_v4/iter_3/infer/inference_all_predictions.csv",
    "/mnt/mydata/dq/projects/Splittree/test/Error Detection Adult/inference_all_predictions.csv",
]

FALLBACK_CANDIDATE_FEATURES = [
    "repair_candidate_features_v2.csv",
    "/mnt/mydata/dq/projects/Splittree/test/Error Correction adult/repair_candidate_features_v2.csv",
    "/mnt/mydata/dq/projects/Splittree/test/Error Correction Adult/repair_candidate_features_v2.csv",
]


# ============================================================
# 2. Utilities
# ============================================================

def resolve_output_path(output_path: str) -> Path:
    output_path = str(output_path)
    if os.path.isabs(output_path):
        return Path(output_path)
    return Path(os.path.abspath(output_path))


def resolve_existing_file(path: str, fallback_paths: List[str] = None, file_desc: str = "file") -> str:
    path = str(path).strip() if path is not None else ""
    if path and Path(path).expanduser().exists() and Path(path).expanduser().is_file():
        return path

    if path and Path(path).expanduser().exists() and Path(path).expanduser().is_dir():
        direct = Path(path).expanduser() / "inference_all_predictions.csv"
        if direct.exists() and direct.is_file():
            print(f"[WARN] {file_desc} was a directory. Using: {direct}")
            return str(direct)

    for fb in fallback_paths or []:
        p = Path(str(fb)).expanduser()
        if p.exists() and p.is_file():
            if path:
                print(f"[WARN] {file_desc} not found: {path}")
            print(f"[INFO] fallback {file_desc}: {p}")
            return str(p)

    raise FileNotFoundError(f"{file_desc} not found: {path}")


def normalize_key_columns(df: pd.DataFrame, file_name: str) -> pd.DataFrame:
    required = {"row_id", "column"}
    if not required.issubset(set(df.columns)):
        raise ValueError(
            f"{file_name} must contain columns {required}, "
            f"but got columns: {list(df.columns)}"
        )

    out = df.copy()
    out["row_id"] = pd.to_numeric(out["row_id"], errors="raise").astype(int)
    out["column"] = out["column"].astype(str).str.strip()
    return out


def filter_allowed_cells(allow_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compatible whitelist inputs:
    1. Adult augmented whitelist: no prediction label columns, treat as whitelist.
    2. inference_all_predictions.csv: has final_pred_label / final_pred_label_name, filter errors.
    3. Detector/raw fallback labels.
    """
    df = allow_df.copy()

    if "final_pred_label" in df.columns:
        mask = pd.to_numeric(df["final_pred_label"], errors="coerce").fillna(0).astype(int) == 1
        print("[INFO] allowed cells source: final_pred_label == 1")
        return df.loc[mask].copy()

    if "final_pred_label_name" in df.columns:
        mask = df["final_pred_label_name"].astype(str).str.strip().str.lower() == "error"
        print("[INFO] allowed cells source: final_pred_label_name == error")
        return df.loc[mask].copy()

    if "pred_label" in df.columns:
        mask = pd.to_numeric(df["pred_label"], errors="coerce").fillna(0).astype(int) == 1
        print("[WARN] no final_pred_label found; fallback to pred_label == 1")
        return df.loc[mask].copy()

    if "pred_label_name" in df.columns:
        mask = df["pred_label_name"].astype(str).str.strip().str.lower() == "error"
        print("[WARN] no final_pred_label_name found; fallback to pred_label_name == error")
        return df.loc[mask].copy()

    if "detector_pred_label" in df.columns:
        mask = pd.to_numeric(df["detector_pred_label"], errors="coerce").fillna(0).astype(int) == 1
        print("[WARN] no final_pred_label found; fallback to detector_pred_label == 1")
        return df.loc[mask].copy()

    if "detector_pred_label_name" in df.columns:
        mask = df["detector_pred_label_name"].astype(str).str.strip().str.lower() == "error"
        print("[WARN] no final_pred_label_name found; fallback to detector_pred_label_name == error")
        return df.loc[mask].copy()

    if "raw_pred_label" in df.columns:
        mask = pd.to_numeric(df["raw_pred_label"], errors="coerce").fillna(0).astype(int) == 1
        print("[WARN] no final/detector label found; fallback to raw_pred_label == 1")
        return df.loc[mask].copy()

    if "raw_pred_label_name" in df.columns:
        mask = df["raw_pred_label_name"].astype(str).str.strip().str.lower() == "error"
        print("[WARN] no final/detector label found; fallback to raw_pred_label_name == error")
        return df.loc[mask].copy()

    print("[INFO] no prediction label columns found; treat input file itself as whitelist")
    return df.copy()


def select_allowed_meta_columns(allow: pd.DataFrame) -> List[str]:
    """
    Keep Adult-specific diagnostic columns in the infer feature output.
    These columns help later gate/inference scripts know why a cell was allowed.
    """
    preferred = [
        "row_id", "column",

        # base
        "value",
        "semantic_type",
        "detected_type",

        # prediction labels/probabilities, if whitelist came from inference output
        "final_pred_label",
        "final_pred_label_name",
        "final_pred_error_prob",
        "final_pred_error_prob_before_adjust",
        "detector_pred_label",
        "detector_pred_label_name",
        "detector_pred_error_prob",
        "raw_pred_label",
        "raw_pred_label_name",
        "raw_pred_error_prob",
        "verifier_decision",
        "verifier_keep_error_prob",
        "verifier_threshold_used",

        # general evidence
        "violation_count",
        "conflict_score",
        "main_rule_type",
        "main_usage_role",
        "neighbor_majority_value",
        "neighbor_majority_ratio",
        "prior_error_probability",
        "posterior_error_probability",

        # Adult rule counts
        "adult_domain_rule_count",
        "adult_format_rule_count",
        "adult_consistency_rule_count",
        "adult_rule_total_count",
        "adult_domain_rule_ratio",
        "adult_format_rule_ratio",
        "adult_consistency_rule_ratio",

        # Adult profile / row features
        "age_lower",
        "age_upper",
        "hours_lower",
        "hours_upper",
        "canonical_income",
        "domain_valid",
        "education_order",
        "adult_value_bucket",
        "row_age_lower",
        "row_age_upper",
        "row_hours_lower",
        "row_hours_upper",
        "row_education_order",
        "row_income_canonical",
        "row_missing_count",
        "row_non_missing_count",
        "adult_profile_inconsistency_count",
        "adult_consistency_score",
        "adult_evidence_flags",

        # Adult attribution / v2 flags
        "target_is_primary_column",
        "target_is_context_only_column",
        "is_context_only_weak_signal",
        "is_relationship_spouse_value",
        "relationship_marital_conflict",
        "relationship_age_conflict",
        "relationship_sex_conflict",
        "sex_value_invalid",
        "education_age_extreme_conflict",
        "has_relationship_v2_rule",
        "has_education_v2_rule",
        "adult_v2_attribution_bucket",
        "relationship_v2_conflict_count",
        "has_relationship_v2_signal",
        "has_education_v2_signal",
        "adult_v2_signal_strength",
        "adult_v2_primary_target_signal",
        "adult_v2_relationship_hardcase",
        "adult_v2_sex_invalid_hardcase",
        "adult_v2_context_fp_hardcase",

        # sampling / buckets / diagnostics
        "pattern_bucket",
        "rarity_bucket",
        "neighbor_bucket",
        "domain_bucket",
        "adult_consistency_bucket",
        "age_sampling_bucket",
        "hours_sampling_bucket",
        "is_hard_case",
        "is_high_risk",
        "adult_signal_hardcase",
        "sample_role",
        "is_sampled",
        "signal_priority",
    ]
    return [c for c in preferred if c in allow.columns]


def prefix_allowed_meta_columns(allow_meta: pd.DataFrame, merged_columns: set) -> pd.DataFrame:
    """
    Prefix allowed meta columns if they collide with candidate feature columns.
    Keep row_id,column unchanged.
    """
    rename_map = {}
    for c in allow_meta.columns:
        if c in {"row_id", "column"}:
            continue
        if c in merged_columns:
            rename_map[c] = f"allowed_{c}"
        else:
            rename_map[c] = f"allowed_{c}"
    return allow_meta.rename(columns=rename_map)


def build_column_summary(cand: pd.DataFrame, allow_unique: pd.DataFrame, merged: pd.DataFrame, missing_allowed: pd.DataFrame) -> pd.DataFrame:
    cand_cells = cand[["row_id", "column"]].drop_duplicates()
    kept_cells = merged[["row_id", "column"]].drop_duplicates()

    rows = []
    all_cols = sorted(set(allow_unique["column"].astype(str)) | set(cand_cells["column"].astype(str)))
    for col in all_cols:
        cand_col_cells = cand_cells[cand_cells["column"] == col]
        allow_col_cells = allow_unique[allow_unique["column"] == col]
        kept_col_cells = kept_cells[kept_cells["column"] == col]
        miss_col_cells = missing_allowed[missing_allowed["column"] == col]

        cand_rows = cand[cand["column"] == col]
        kept_rows = merged[merged["column"] == col]

        rows.append({
            "column": col,
            "candidate_unique_cells_before": len(cand_col_cells),
            "allowed_unique_cells": len(allow_col_cells),
            "candidate_unique_cells_after": len(kept_col_cells),
            "missing_allowed_cells": len(miss_col_cells),
            "candidate_rows_before": len(cand_rows),
            "candidate_rows_after": len(kept_rows),
            "avg_candidates_per_kept_cell": (
                len(kept_rows) / max(len(kept_col_cells), 1)
            ),
        })
    return pd.DataFrame(rows)


def build_whitelist_features(candidate_features_file: str, allowed_cells_file: str, output_infer_features: str):
    candidate_features_file = resolve_existing_file(
        candidate_features_file,
        FALLBACK_CANDIDATE_FEATURES,
        file_desc="candidate features file",
    )
    allowed_cells_file = resolve_existing_file(
        allowed_cells_file,
        FALLBACK_ALLOWED_CELLS,
        file_desc="allowed cells file",
    )

    # ------------------------------------------------------------
    # A. Read candidate features
    # ------------------------------------------------------------
    cand = pd.read_csv(candidate_features_file, encoding="utf-8-sig")
    cand = normalize_key_columns(cand, "candidate features file")

    # ------------------------------------------------------------
    # B. Read allowed cells and filter to predicted-error cells if needed
    # ------------------------------------------------------------
    allow_raw = pd.read_csv(allowed_cells_file, encoding="utf-8-sig")
    allow_raw = normalize_key_columns(allow_raw, "allowed cells file")
    allow = filter_allowed_cells(allow_raw)
    allow = normalize_key_columns(allow, "filtered allowed cells")

    allow_unique = allow[["row_id", "column"]].drop_duplicates().copy()

    # Adult metadata from whitelist, prefixed as allowed_* in final output.
    allow_meta_cols = select_allowed_meta_columns(allow)
    allow_meta = allow[allow_meta_cols].drop_duplicates(subset=["row_id", "column"], keep="first").copy()

    if not allow_meta.empty:
        allow_meta = prefix_allowed_meta_columns(allow_meta, set(cand.columns))

    # ------------------------------------------------------------
    # C. Statistics before filtering
    # ------------------------------------------------------------
    cand_unique_before = cand[["row_id", "column"]].drop_duplicates().shape[0]
    cand_rows_before = len(cand)

    # ------------------------------------------------------------
    # D. Inner join by row_id + column
    # ------------------------------------------------------------
    merged = cand.merge(
        allow_unique.assign(__keep__=1),
        on=["row_id", "column"],
        how="inner",
    ).drop(columns=["__keep__"])

    if not allow_meta.empty:
        merged = merged.merge(
            allow_meta,
            on=["row_id", "column"],
            how="left",
        )

    cand_unique_after = merged[["row_id", "column"]].drop_duplicates().shape[0]
    cand_rows_after = len(merged)

    # ------------------------------------------------------------
    # E. Save main output
    # ------------------------------------------------------------
    output_path = resolve_output_path(output_infer_features)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------
    # F. Missing allowed cells: allowed but no candidate feature
    # ------------------------------------------------------------
    missing_allowed = allow_unique.merge(
        merged[["row_id", "column"]].drop_duplicates(),
        on=["row_id", "column"],
        how="left",
        indicator=True,
    )
    missing_allowed = missing_allowed[missing_allowed["_merge"] == "left_only"].drop(columns=["_merge"])

    missing_path = None
    if OUTPUT_MISSING_ALLOWED_CELLS:
        missing_path = output_path.with_name(output_path.stem + "_missing_allowed_cells.csv")
        if len(missing_allowed) > 0:
            missing_with_meta = missing_allowed.merge(
                allow,
                on=["row_id", "column"],
                how="left",
            )
            missing_with_meta.to_csv(missing_path, index=False, encoding="utf-8-sig")
        else:
            missing_allowed.to_csv(missing_path, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------
    # G. Filtered-out candidate features
    # ------------------------------------------------------------
    filtered_out_path = None
    if OUTPUT_FILTERED_OUT_FEATURES:
        filtered_out = cand.merge(
            allow_unique.assign(__keep__=1),
            on=["row_id", "column"],
            how="left",
        )
        filtered_out = filtered_out[filtered_out["__keep__"].isna()].drop(columns=["__keep__"])

        filtered_out_path = output_path.with_name(output_path.stem + "_filtered_out.csv")
        if len(filtered_out) > 0:
            filtered_out.to_csv(filtered_out_path, index=False, encoding="utf-8-sig")
        else:
            filtered_out.to_csv(filtered_out_path, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------
    # H. Column-level summary
    # ------------------------------------------------------------
    summary_path = None
    if OUTPUT_WHITELIST_SUMMARY:
        summary = build_column_summary(cand, allow_unique, merged, missing_allowed)
        summary_path = output_path.with_name(output_path.stem + "_summary_by_column.csv")
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------
    # I. Print statistics
    # ------------------------------------------------------------
    print("\n========== Build Adult infer candidate features from whitelist ==========")
    print(f"input_candidate_features: {candidate_features_file}")
    print(f"input_allowed_cells: {allowed_cells_file}")
    print(f"output_infer_features: {output_path}")
    print("")
    print("---- Cell-level statistics ----")
    print(f"candidate_unique_cells_before: {cand_unique_before}")
    print(f"allowed_unique_cells: {len(allow_unique)}")
    print(f"candidate_unique_cells_after: {cand_unique_after}")
    print(f"missing_allowed_cells: {len(missing_allowed)}")
    print("")
    print("---- Row-level statistics ----")
    print(f"candidate_rows_before: {cand_rows_before}")
    print(f"candidate_rows_after: {cand_rows_after}")
    print(f"filtered_out_rows: {cand_rows_before - cand_rows_after}")

    if len(missing_allowed) > 0:
        print("")
        print("[WARN] Some whitelist cells are not present in candidate features.")
        print(f"[WARN] missing allowed cells saved to: {missing_path}")

    if filtered_out_path is not None:
        print(f"[INFO] filtered-out candidate features saved to: {filtered_out_path}")

    if summary_path is not None:
        print(f"[INFO] summary by column saved to: {summary_path}")

    # Adult-specific stats if columns exist.
    print("")
    print("---- Adult whitelist diagnostics ----")
    for col in [
        "adult_evidence_flags",
        "adult_v2_attribution_bucket",
        "adult_consistency_bucket",
        "main_rule_type",
        "semantic_type",
    ]:
        allowed_col = f"allowed_{col}"
        if allowed_col in merged.columns:
            print(f"\n{allowed_col} Top 20:")
            print(merged[["row_id", "column", allowed_col]].drop_duplicates()[allowed_col].value_counts(dropna=False).head(20).to_string())

    if "column" in merged.columns:
        print("\nKept candidate rows by column:")
        print(merged["column"].value_counts().head(30).to_string())

    print(f"\n[OK] saved to: {output_path.resolve()}")

    return merged


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate_features", default=DEFAULT_INPUT_CANDIDATE_FEATURES)
    p.add_argument("--allowed_cells", default=DEFAULT_INPUT_ALLOWED_CELLS)
    p.add_argument("--output", default=DEFAULT_OUTPUT_INFER_FEATURES)
    return p.parse_args()


def main():
    args = parse_args()
    build_whitelist_features(
        candidate_features_file=args.candidate_features,
        allowed_cells_file=args.allowed_cells,
        output_infer_features=args.output,
    )


if __name__ == "__main__":
    main()

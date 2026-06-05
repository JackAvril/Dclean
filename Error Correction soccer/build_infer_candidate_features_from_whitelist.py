#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Soccer inference candidate features from whitelist.

This script is adapted from the Adult whitelist builder, but removes Adult-specific
relationship/education/profile metadata and keeps Soccer-specific diagnostic fields.

Inputs
------
1. Candidate feature file:
   repair_candidate_features_v2.csv

2. Allowed cells / whitelist file. Supported formats:
   A. predicted_error_positions_soccer_augmented.csv
      Output of revise_candidate_soccer.py. Contains row_id,column and optional Soccer fields.
   B. predicted_error_positions.csv
      Output of wrong_position_soccer.py. Contains row_id,column and optional value/final_pred_label fields.
   C. inference_all_predictions.csv
      Contains final_pred_label / final_pred_label_name and will be filtered to error cells.

Output
------
repair_candidate_features_infer_whitelist.csv

Usage
-----
cd /mnt/mydata/dq/projects/Splittree/test/Error\ Correction\ soccer

python build_infer_features_whitelist_soccer.py \
  --candidate_features repair_candidate_features_v2.csv \
  --allowed_cells predicted_error_positions_soccer_augmented.csv \
  --output repair_candidate_features_infer_whitelist.csv

Notes
-----
This script does not regenerate candidates. It only performs a safe inner join:
    candidate_features[row_id,column] ∩ allowed_cells[row_id,column]

So if many allowed cells are missing in the output, the problem is usually upstream:
    - candidate generation did not produce candidates for those cells, or
    - row_id/column naming is inconsistent.
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
DEFAULT_INPUT_ALLOWED_CELLS = "predicted_error_positions_soccer_augmented.csv"
DEFAULT_OUTPUT_INFER_FEATURES = "repair_candidate_features_infer_whitelist.csv"

OUTPUT_MISSING_ALLOWED_CELLS = True
OUTPUT_FILTERED_OUT_FEATURES = True
OUTPUT_WHITELIST_SUMMARY = True

# 保留为单路径风格，不使用一堆候选路径。
# 如果你的文件名不一致，直接改这里即可。
DEFAULT_FALLBACK_PREDICTED_POSITIONS = "predicted_error_positions.csv"
DEFAULT_FALLBACK_INFERENCE_ALL = "inference_all_predictions.csv"


# ============================================================
# 2. Utilities
# ============================================================

def resolve_output_path(output_path: str) -> Path:
    output_path = str(output_path)
    if os.path.isabs(output_path):
        return Path(output_path)
    return Path(os.path.abspath(output_path))


def resolve_existing_file(path: str, fallback_paths: List[str] = None, file_desc: str = "file") -> str:
    """
    Resolve a file path.

    The script mostly expects explicit paths. Fallbacks are only local default files
    in the current working directory, not long global path lists.
    """
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
    1. Soccer augmented whitelist: no prediction label columns, treat as whitelist.
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
    Keep Soccer-specific diagnostic columns in the infer feature output.
    These columns help later LTR inference / gate scripts know why a cell was allowed.
    """
    preferred = [
        "row_id", "column",

        # base
        "value",
        "dirty_value",
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
        "detector_pred_error_prob_before_adjust",
        "raw_pred_label",
        "raw_pred_label_name",
        "raw_pred_error_prob",
        "verifier_decision",
        "verifier_used",
        "verifier_keep_error_prob",
        "verifier_threshold_used",
        "verifier_rejected",

        # general evidence
        "violation_count",
        "conflict_score",
        "value_frequency",
        "value_frequency_rank",
        "main_rule_type",
        "main_usage_role",
        "neighbor_majority_value",
        "neighbor_majority_ratio",
        "prior_error_probability",
        "posterior_error_probability",

        # general rule counts
        "strong_rule_count",
        "candidate_generation_rule_count",
        "fd_like_count",
        "evidence_only_fd_count",
        "context_rule_count",
        "global_rule_count",
        "rare_value_count",
        "typo_rule_count",
        "pattern_rule_count",
        "schema_rule_count",

        # Soccer rule counts and ratios
        "soccer_domain_rule_count",
        "soccer_format_rule_count",
        "soccer_numeric_rule_count",
        "soccer_consistency_rule_count",
        "soccer_rule_total_count",
        "soccer_domain_rule_ratio",
        "soccer_format_rule_ratio",
        "soccer_numeric_rule_ratio",
        "soccer_consistency_rule_ratio",

        # Soccer core value / row context features
        "year_value",
        "canonical_position",
        "domain_valid",
        "soccer_value_bucket",
        "row_birthyear_int",
        "row_season_int",
        "row_age_at_season",
        "row_position_canonical",
        "birthyear_season_gap",
        "row_team",
        "row_city",
        "row_stadium",
        "row_manager",
        "row_missing_count",
        "row_non_missing_count",

        # Soccer consistency / attribution
        "soccer_profile_inconsistency_count",
        "soccer_consistency_score",
        "soccer_evidence_flags",
        "soccer_attribution_bucket",
        "soccer_consistency_bucket",
        "target_is_primary_column",
        "target_is_context_only_column",
        "is_context_only_weak_signal",
        "birthyear_value_invalid",
        "season_value_invalid",
        "position_value_invalid",
        "position_needs_canonicalization",
        "birthyear_season_age_conflict",
        "team_context_conflict",
        "format_rule_signal",
        "fd_like_signal",
        "domain_invalid_flag",
        "soccer_consistency_flag",
        "high_soccer_consistency_score",
        "has_soccer_domain_signal",
        "has_soccer_format_signal",
        "has_soccer_numeric_signal",
        "has_soccer_consistency_signal",
        "has_birthyear_or_season_signal",
        "has_team_context_signal",
        "has_fd_like_signal",
        "soccer_signal_strength",
        "soccer_primary_target_signal",
        "soccer_precision_gate_applied",
        "soccer_precision_gate_reason",

        # Candidate-generation augmentation from revise_candidate_soccer.py
        "candidate_count",
        "has_rule_candidate",
        "has_soccer_profile_candidate",
        "has_soccer_direct_rule_candidate",
        "has_neighbor_candidate",
        "has_domain_candidate",
        "has_numeric_candidate",

        # New profile-majority candidate-generation diagnostics from revised revise_candidate.py.
        # These may appear in predicted_error_positions_soccer_augmented.csv.
        "has_profile_key_majority_candidate",
        "has_entity_pair_majority_candidate",
        "has_team_season_position_candidate",

        "top_candidate_value",
        "top_candidate_score",
        "top_candidate_sources",

        # buckets / sampling / diagnostics
        "current_value_pattern",
        "dominant_pattern",
        "dominant_pattern_ratio",
        "pattern_bucket",
        "rarity_bucket",
        "neighbor_bucket",
        "domain_bucket",
        "bucket_id",
        "cluster_id",
        "dist_to_center",
        "sample_role",
        "is_sampled",
        "signal_priority",
        "is_hard_case",
        "hard_case_reason",
        "is_high_risk",
        "high_risk",
        "rare_only_signal",
        "is_weak_only",
        "is_fp_risk_col",
        "high_prob_but_neighbor_support_current",
        "is_mid_prob",
        "is_near_threshold",
        "verifier_borderline",
        "verifier_reject_borderline",
        "fp_conflict_fd_hardcase",
        "sample_borderline_hardcase",
        "recall_recovery_hardcase",
        "persistent_fp_high_risk",

        # LLM annotation / explanation fields if present
        "label",
        "label_binary",
        "label_source",
        "sample_weight",
        "error_type",
        "reason_short",
        "reason_detailed",
        "suggested_correct_value",
        "needs_human_review",
        "evidence_used",

        # Compatibility fields that may still exist because the Soccer detector reused parts
        # of Adult feature code. We keep them only to avoid downstream KeyError.
        "adult_domain_rule_count",
        "adult_format_rule_count",
        "adult_consistency_rule_count",
        "time_window_rule_count",
        "time_value_minutes",
        "time_window_bucket",
        "time_value_bucket",
        "adult_profile_inconsistency_count",
        "adult_consistency_score",
        "adult_evidence_flags",
        "adult_value_bucket",
        "age_lower",
        "age_upper",
        "hours_lower",
        "hours_upper",
        "canonical_income",
        "education_order",
        "row_age_lower",
        "row_age_upper",
        "row_hours_lower",
        "row_hours_upper",
        "row_education_order",
        "row_income_canonical",
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
        # Prefix all metadata uniformly to make provenance explicit.
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

        row = {
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
        }

        # New profile-majority diagnostics. These columns are optional; they appear
        # after using revised revise_candidate.py and revised build_features.py.
        for diag_col in [
            "is_from_profile_key_majority",
            "is_from_entity_pair_majority",
            "is_from_team_season_position",
            "candidate_entity_safe_fill",
            "candidate_position_profile_strong",
            "candidate_has_high_conf_profile_majority",
            "candidate_is_pure_weak_global_for_entity",
        ]:
            if diag_col in kept_rows.columns:
                row[f"{diag_col}_rows_after"] = int(
                    pd.to_numeric(kept_rows[diag_col], errors="coerce").fillna(0).sum()
                )
            else:
                row[f"{diag_col}_rows_after"] = 0

        rows.append(row)
    return pd.DataFrame(rows)


def build_whitelist_features(candidate_features_file: str, allowed_cells_file: str, output_infer_features: str):
    candidate_features_file = resolve_existing_file(
        candidate_features_file,
        [DEFAULT_INPUT_CANDIDATE_FEATURES],
        file_desc="candidate features file",
    )
    allowed_cells_file = resolve_existing_file(
        allowed_cells_file,
        [
            DEFAULT_INPUT_ALLOWED_CELLS,
            DEFAULT_FALLBACK_PREDICTED_POSITIONS,
            DEFAULT_FALLBACK_INFERENCE_ALL,
        ],
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

    # Soccer metadata from whitelist, prefixed as allowed_* in final output.
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
    print("\n========== Build Soccer infer candidate features from whitelist ==========")
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
        print("[HINT] If this number is high, inspect candidate generation coverage first.")

    if filtered_out_path is not None:
        print(f"[INFO] filtered-out candidate features saved to: {filtered_out_path}")

    if summary_path is not None:
        print(f"[INFO] summary by column saved to: {summary_path}")

    # Soccer-specific diagnostics if columns exist.
    print("")
    print("---- Soccer whitelist diagnostics ----")
    diagnostic_cols = [
        "soccer_evidence_flags",
        "soccer_attribution_bucket",
        "soccer_consistency_bucket",
        "soccer_precision_gate_reason",
        "main_rule_type",
        "semantic_type",
    ]
    for col in diagnostic_cols:
        allowed_col = f"allowed_{col}"
        if allowed_col in merged.columns:
            print(f"\n{allowed_col} Top 20:")
            print(
                merged[["row_id", "column", allowed_col]]
                .drop_duplicates()[allowed_col]
                .value_counts(dropna=False)
                .head(20)
                .to_string()
            )

    if "column" in merged.columns:
        print("\nKept candidate rows by column:")
        print(merged["column"].value_counts().head(30).to_string())

        print("\nKept unique cells by column:")
        print(merged[["row_id", "column"]].drop_duplicates()["column"].value_counts().head(30).to_string())

    # New profile-majority candidate diagnostics from revised candidate generation / feature building.
    new_candidate_diag_cols = [
        "is_from_profile_key_majority",
        "is_from_entity_pair_majority",
        "is_from_team_season_position",
        "candidate_has_profile_majority_signal",
        "candidate_has_high_conf_profile_majority",
        "candidate_has_very_high_conf_profile_majority",
        "candidate_entity_safe_fill",
        "candidate_position_profile_strong",
        "candidate_is_name_entity_majority",
        "candidate_is_surname_entity_majority",
        "candidate_is_birthplace_entity_majority",
        "candidate_is_position_profile_majority",
        "candidate_is_profile_or_pair_majority",
        "candidate_is_pure_weak_global_for_entity",
    ]
    existing_new_diag_cols = [c for c in new_candidate_diag_cols if c in merged.columns]
    if existing_new_diag_cols:
        print("\n---- New profile-majority candidate feature diagnostics ----")
        for c in existing_new_diag_cols:
            try:
                print(f"{c}: {pd.to_numeric(merged[c], errors='coerce').fillna(0).sum():.0f}")
            except Exception:
                print(f"{c}: non-numeric or unavailable")

        print("\nNew profile-majority feature sums by column:")
        try:
            diag_by_col = merged[["column"] + existing_new_diag_cols].copy()
            for c in existing_new_diag_cols:
                diag_by_col[c] = pd.to_numeric(diag_by_col[c], errors="coerce").fillna(0)
            print(diag_by_col.groupby("column")[existing_new_diag_cols].sum().head(30).to_string())
        except Exception as e:
            print(f"[WARN] failed to print profile-majority by-column diagnostics: {e}")

    allowed_new_diag_cols = [
        "allowed_has_profile_key_majority_candidate",
        "allowed_has_entity_pair_majority_candidate",
        "allowed_has_team_season_position_candidate",
    ]
    existing_allowed_new_diag_cols = [c for c in allowed_new_diag_cols if c in merged.columns]
    if existing_allowed_new_diag_cols:
        print("\n---- Allowed-cell profile-majority diagnostics ----")
        allowed_unique_for_diag = merged[["row_id", "column"] + existing_allowed_new_diag_cols].drop_duplicates()
        for c in existing_allowed_new_diag_cols:
            try:
                print(f"{c}: {pd.to_numeric(allowed_unique_for_diag[c], errors='coerce').fillna(0).sum():.0f}")
            except Exception:
                print(f"{c}: non-numeric or unavailable")

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

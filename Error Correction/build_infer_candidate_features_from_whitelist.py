#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified whitelist builder for repair inference features.

Datasets:
  hospital, flights, beers, rayyan, adult, soccer

Purpose
-------
Filter candidate-level repair features to only those cells allowed by the
detection stage. This is the common logic behind all dataset-specific
build_infer_features_whitelist*.py scripts:

  repair_candidate_features_v2.csv
      ∩ row_id,column from predicted_error_positions*.csv
      -> repair_candidate_features_infer_whitelist.csv

Default output filenames remain:
  repair_candidate_features_infer_whitelist.csv
  repair_candidate_features_infer_whitelist_missing_allowed_cells.csv
  repair_candidate_features_infer_whitelist_filtered_out_features.csv
  repair_candidate_features_infer_whitelist_summary_by_column.csv

This version is readable and configurable. Dataset-specific differences are
handled only by:
  - default input paths
  - preferred whitelist metadata columns
  - whether to filter final_pred_label-like columns when present
"""

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd


# ============================================================
# 1. Dataset configs
# ============================================================

COMMON_META_COLUMNS = [
    "row_id", "column",
    "value", "dirty_value", "semantic_type", "detected_type",
    "final_pred_label", "final_pred_label_name",
    "final_pred_error_prob", "final_pred_error_prob_before_adjust",
    "detector_pred_label", "detector_pred_label_name", "detector_pred_error_prob",
    "raw_pred_label", "raw_pred_label_name", "raw_pred_error_prob",
    "verifier_decision", "verifier_keep_error_prob", "verifier_threshold_used",
    "violation_count", "conflict_score", "value_frequency", "value_frequency_rank",
    "main_rule_type", "main_usage_role",
    "neighbor_majority_value", "neighbor_majority_ratio",
    "prior_error_probability", "posterior_error_probability",
    "strong_rule_count", "candidate_generation_rule_count", "fd_like_count",
    "context_rule_count", "global_rule_count", "rare_value_count",
    "typo_rule_count", "pattern_rule_count", "schema_rule_count",
    "candidate_count", "has_rule_candidate", "has_neighbor_candidate",
    "has_domain_candidate", "has_numeric_candidate",
    "top_candidate_value", "top_candidate_score", "top_candidate_sources",
    "label", "label_binary", "label_source", "sample_role", "is_sampled",
    "error_type", "reason_short", "reason_detailed", "suggested_correct_value",
    "needs_human_review", "evidence_used",
]

FLIGHTS_META_COLUMNS = [
    "time_window_bucket", "time_value_bucket", "time_value_minutes",
    "row_flight", "row_src", "row_sched_dep_time", "row_act_dep_time",
    "row_sched_arr_time", "row_act_arr_time",
    "trusted_source_count", "time_consensus_value", "time_consensus_confidence",
    "time_consensus_margin", "source_consensus_bucket",
]

BEERS_META_COLUMNS = [
    "state_city_consistency", "state_city_conflict", "abv_value", "ibu_value",
    "ounces_value", "beers_domain_rule_count", "beers_numeric_rule_count",
    "beers_profile_rule_count", "beer_style_bucket", "brewery_profile_bucket",
]

RAYYAN_META_COLUMNS = [
    "article_language", "journal_issn", "journal_title", "journal_abbreviation",
    "jounral_abbreviation", "article_jcreated_at", "article_jvolumn",
    "article_jissue", "article_pagination", "rayyan_metadata_rule_count",
    "rayyan_consensus_value", "rayyan_consensus_confidence",
    "rayyan_consensus_margin", "metadata_consensus_bucket",
]

ADULT_META_COLUMNS = [
    "age_lower", "age_upper", "hours_lower", "hours_upper", "canonical_income",
    "education_order", "adult_value_bucket", "row_age_lower", "row_age_upper",
    "row_hours_lower", "row_hours_upper", "row_education_order",
    "row_income_canonical", "adult_profile_inconsistency_count",
    "adult_consistency_score", "adult_evidence_flags",
    "is_relationship_spouse_value", "relationship_marital_conflict",
    "relationship_age_conflict", "relationship_sex_conflict", "sex_value_invalid",
    "education_age_extreme_conflict", "has_relationship_v2_rule",
    "has_education_v2_rule",
]

SOCCER_META_COLUMNS = [
    "year_value", "canonical_position", "domain_valid", "soccer_value_bucket",
    "row_birthyear_int", "row_season_int", "row_age_at_season",
    "row_position_canonical", "row_team", "row_city", "row_stadium",
    "row_manager", "soccer_profile_inconsistency_count",
    "soccer_consistency_score", "soccer_evidence_flags",
    "birthyear_value_invalid", "season_value_invalid", "position_value_invalid",
    "position_needs_canonicalization", "birthyear_season_age_conflict",
    "team_context_conflict", "format_rule_signal", "fd_like_signal",
    "soccer_attribution_bucket",
    "has_profile_key_majority_candidate",
    "has_entity_pair_majority_candidate",
    "has_team_season_position_candidate",
]


@dataclass
class DatasetConfig:
    name: str
    default_candidate_features: str
    default_allowed_cells: str
    default_output: str = "repair_candidate_features_infer_whitelist.csv"
    meta_columns: List[str] = field(default_factory=list)
    filter_prediction_labels_if_present: bool = True


DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "hospital": DatasetConfig(
        name="hospital",
        default_candidate_features="repair_candidate_features_v2.csv",
        default_allowed_cells="predicted_error_positions.csv",
        meta_columns=COMMON_META_COLUMNS,
    ),
    "flights": DatasetConfig(
        name="flights",
        default_candidate_features="repair_candidate_features_v2.csv",
        default_allowed_cells="predicted_error_positions.csv",
        meta_columns=COMMON_META_COLUMNS + FLIGHTS_META_COLUMNS,
    ),
    "beers": DatasetConfig(
        name="beers",
        default_candidate_features="repair_candidate_features_v2.csv",
        default_allowed_cells="predicted_error_positions.csv",
        meta_columns=COMMON_META_COLUMNS + BEERS_META_COLUMNS,
    ),
    "rayyan": DatasetConfig(
        name="rayyan",
        default_candidate_features="repair_candidate_features_v2.csv",
        default_allowed_cells="predicted_error_positions.csv",
        meta_columns=COMMON_META_COLUMNS + RAYYAN_META_COLUMNS,
    ),
    "adult": DatasetConfig(
        name="adult",
        default_candidate_features="repair_candidate_features_v2.csv",
        default_allowed_cells="predicted_error_positions_adult_augmented.csv",
        meta_columns=COMMON_META_COLUMNS + ADULT_META_COLUMNS,
    ),
    "soccer": DatasetConfig(
        name="soccer",
        default_candidate_features="repair_candidate_features_v2.csv",
        default_allowed_cells="predicted_error_positions_soccer_augmented.csv",
        meta_columns=COMMON_META_COLUMNS + SOCCER_META_COLUMNS,
    ),
}


# ============================================================
# 2. Utilities
# ============================================================

def read_csv(path: str, desc: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{desc} not found: {path}")
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def normalize_key_columns(df: pd.DataFrame, desc: str) -> pd.DataFrame:
    required = {"row_id", "column"}
    if not required.issubset(df.columns):
        raise ValueError(f"{desc} must contain {required}, got columns={list(df.columns)}")
    out = df.copy()
    out["row_id"] = pd.to_numeric(out["row_id"], errors="raise").astype(int)
    out["column"] = out["column"].astype(str).str.strip()
    return out


def filter_allowed_cells(df: pd.DataFrame, enabled: bool = True) -> pd.DataFrame:
    if not enabled:
        return df.copy()

    label_specs = [
        ("final_pred_label", "numeric"),
        ("final_pred_label_name", "name"),
        ("pred_label", "numeric"),
        ("pred_label_name", "name"),
        ("detector_pred_label", "numeric"),
        ("detector_pred_label_name", "name"),
        ("raw_pred_label", "numeric"),
        ("raw_pred_label_name", "name"),
    ]

    for col, kind in label_specs:
        if col not in df.columns:
            continue
        if kind == "numeric":
            mask = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int) == 1
            print(f"[INFO] allowed cells filtered by {col} == 1")
        else:
            mask = df[col].astype(str).str.strip().str.lower() == "error"
            print(f"[INFO] allowed cells filtered by {col} == error")
        return df.loc[mask].copy()

    print("[INFO] no prediction label columns found; treat allowed input as whitelist")
    return df.copy()


def select_allowed_meta_columns(allow: pd.DataFrame, cfg: DatasetConfig) -> List[str]:
    cols = []
    for c in cfg.meta_columns:
        if c in allow.columns and c not in cols:
            cols.append(c)
    if "row_id" not in cols:
        cols.insert(0, "row_id")
    if "column" not in cols:
        cols.insert(1, "column")
    return cols


def prefix_allowed_meta_columns(meta: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for c in meta.columns:
        if c not in {"row_id", "column"}:
            rename[c] = f"allowed_{c}"
    return meta.rename(columns=rename)


def output_related_paths(output_path: Path) -> Dict[str, Path]:
    stem = output_path.stem
    return {
        "missing": output_path.with_name(stem + "_missing_allowed_cells.csv"),
        "filtered_out": output_path.with_name(stem + "_filtered_out_features.csv"),
        "summary": output_path.with_name(stem + "_summary_by_column.csv"),
    }


def build_column_summary(
    cand: pd.DataFrame,
    allow_unique: pd.DataFrame,
    merged: pd.DataFrame,
    missing_allowed: pd.DataFrame,
) -> pd.DataFrame:
    cand_cells = cand[["row_id", "column"]].drop_duplicates()
    kept_cells = merged[["row_id", "column"]].drop_duplicates()
    all_cols = sorted(set(cand_cells["column"].astype(str)) | set(allow_unique["column"].astype(str)))

    rows = []
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
            "avg_candidates_per_kept_cell": len(kept_rows) / max(len(kept_col_cells), 1),
        }

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

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# 3. Main
# ============================================================

def build_whitelist_features(
    dataset: str,
    candidate_features: str,
    allowed_cells: str,
    output: str,
    filter_labels: bool = True,
    keep_allowed_metadata: bool = True,
    output_missing: bool = True,
    output_filtered_out: bool = True,
    output_summary: bool = True,
) -> pd.DataFrame:
    cfg = DATASET_CONFIGS[dataset]

    cand = normalize_key_columns(read_csv(candidate_features, "candidate features"), "candidate features")
    allow_raw = normalize_key_columns(read_csv(allowed_cells, "allowed cells"), "allowed cells")
    allow = normalize_key_columns(
        filter_allowed_cells(allow_raw, enabled=filter_labels and cfg.filter_prediction_labels_if_present),
        "filtered allowed cells",
    )

    allow_unique = allow[["row_id", "column"]].drop_duplicates().copy()

    cand_unique_before = cand[["row_id", "column"]].drop_duplicates().shape[0]
    cand_rows_before = len(cand)

    merged = cand.merge(
        allow_unique.assign(__keep__=1),
        on=["row_id", "column"],
        how="inner",
    ).drop(columns=["__keep__"])

    if keep_allowed_metadata:
        meta_cols = select_allowed_meta_columns(allow, cfg)
        allow_meta = allow[meta_cols].drop_duplicates(subset=["row_id", "column"], keep="first").copy()
        allow_meta = prefix_allowed_meta_columns(allow_meta)
        merged = merged.merge(allow_meta, on=["row_id", "column"], how="left")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")

    kept_unique = merged[["row_id", "column"]].drop_duplicates()
    missing_allowed = allow_unique.merge(
        kept_unique,
        on=["row_id", "column"],
        how="left",
        indicator=True,
    )
    missing_allowed = missing_allowed[missing_allowed["_merge"] == "left_only"].drop(columns=["_merge"])

    related = output_related_paths(output_path)

    if output_missing and not missing_allowed.empty:
        missing_allowed.to_csv(related["missing"], index=False, encoding="utf-8-sig")
        print(f"[WARN] missing allowed cells saved: {related['missing']}")

    if output_filtered_out:
        filtered_out = cand.merge(
            kept_unique.assign(__kept__=1),
            on=["row_id", "column"],
            how="left",
        )
        filtered_out = filtered_out[filtered_out["__kept__"].isna()].drop(columns=["__kept__"])
        filtered_out.to_csv(related["filtered_out"], index=False, encoding="utf-8-sig")

    if output_summary:
        summary = build_column_summary(cand, allow_unique, merged, missing_allowed)
        summary.to_csv(related["summary"], index=False, encoding="utf-8-sig")

    cand_unique_after = kept_unique.shape[0]

    print("========== Unified build infer candidate features from whitelist ==========")
    print(f"dataset: {dataset}")
    print(f"input_candidate_features: {candidate_features}")
    print(f"input_allowed_cells: {allowed_cells}")
    print(f"output_infer_features: {output}")
    print(f"candidate_rows_before: {cand_rows_before}")
    print(f"candidate_unique_cells_before: {cand_unique_before}")
    print(f"allowed_unique_cells: {len(allow_unique)}")
    print(f"candidate_unique_cells_after: {cand_unique_after}")
    print(f"output_rows: {len(merged)}")
    print(f"[OK] saved to: {output_path.resolve()}")

    if "column" in merged.columns:
        print("\nKept candidate rows by column:")
        print(merged["column"].value_counts().head(30).to_string())
        print("\nKept unique cells by column:")
        print(merged[["row_id", "column"]].drop_duplicates()["column"].value_counts().head(30).to_string())

    return merged


def main():
    parser = argparse.ArgumentParser(description="Unified whitelist builder for repair inference features.")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_CONFIGS.keys()))
    parser.add_argument("--candidate_features", default=None)
    parser.add_argument("--allowed_cells", default=None)
    parser.add_argument("--output", default=None)

    parser.add_argument("--filter_labels", type=int, default=1)
    parser.add_argument("--keep_allowed_metadata", type=int, default=1)
    parser.add_argument("--output_missing", type=int, default=1)
    parser.add_argument("--output_filtered_out", type=int, default=1)
    parser.add_argument("--output_summary", type=int, default=1)

    args = parser.parse_args()
    cfg = DATASET_CONFIGS[args.dataset]

    candidate_features = args.candidate_features or cfg.default_candidate_features
    allowed_cells = args.allowed_cells or cfg.default_allowed_cells
    output = args.output or cfg.default_output

    build_whitelist_features(
        dataset=args.dataset,
        candidate_features=candidate_features,
        allowed_cells=allowed_cells,
        output=output,
        filter_labels=bool(args.filter_labels),
        keep_allowed_metadata=bool(args.keep_allowed_metadata),
        output_missing=bool(args.output_missing),
        output_filtered_out=bool(args.output_filtered_out),
        output_summary=bool(args.output_summary),
    )


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export predicted error positions for Soccer dataset.

This script is adapted from the Adult predicted-error-position exporter, but
keeps Soccer-specific fields:
  - soccer domain / format / numeric / consistency rule counts
  - birthyear / season / age-at-season signals
  - position canonicalization and domain validity signals
  - team / city / stadium / manager context
  - soccer precision gate outputs
  - verifier / detector / raw prediction outputs
  - sampling / hard case / high risk diagnostics

Input
-----
Usually:
    inference_all_predictions.csv

Required columns:
    row_id, column

Prediction label columns, in priority order:
    final_pred_label
    final_pred_label_name
    detector_pred_label
    detector_pred_label_name
    raw_pred_label
    raw_pred_label_name

Output
------
    predicted_error_positions.csv

Usage
-----
cd /mnt/mydata/dq/projects/Splittree/test/Error\ Correction\ soccer

python wrong_position_soccer.py \
  --input_file "/mnt/mydata/dq/projects/Splittree/test/Error Detection Soccer/active_cleaning_loop_runs_soccer_v4/iter_3_infer/inference_all_predictions.csv" \
  --output_file predicted_error_positions.csv

If --input_file is omitted, the script tries common Soccer paths and ./inference_all_predictions.csv.
"""

import argparse
import os
from pathlib import Path
from typing import List

import pandas as pd


# ============================================================
# 1. Default paths
# ============================================================

COMMON_INPUT_CANDIDATES = [
    "inference_all_predictions.csv",
    "./inference_all_predictions.csv",
    "./infer/inference_all_predictions.csv",
    "./active_cleaning_loop_runs_soccer_v4/iter_3_infer/inference_all_predictions.csv",
    "./active_cleaning_loop_runs_soccer_v4/iter_3/infer/inference_all_predictions.csv",
    "/mnt/mydata/dq/projects/Splittree/test/Error Detection Soccer/active_cleaning_loop_runs_soccer_v4/iter_3_infer/inference_all_predictions.csv",
    "/mnt/mydata/dq/projects/Splittree/test/Error Detection Soccer/active_cleaning_loop_runs_soccer_v4/iter_3/infer/inference_all_predictions.csv",
    "/mnt/mydata/dq/projects/Splittree/test/Error Detection Soccer/active_cleaning_loop_runs_soccer_v4/iter_2_infer/inference_all_predictions.csv",
    "/mnt/mydata/dq/projects/Splittree/test/Error Detection Soccer/active_cleaning_loop_runs_soccer_v4/iter_2/infer/inference_all_predictions.csv",
    "/mnt/mydata/dq/projects/Splittree/test/Error Detection Soccer/active_cleaning_loop_runs_soccer_v4/iter_1_infer/inference_all_predictions.csv",
    "/mnt/mydata/dq/projects/Splittree/test/Error Detection Soccer/active_cleaning_loop_runs_soccer_v4/iter_1/infer/inference_all_predictions.csv",
    "/mnt/mydata/dq/projects/Splittree/test/Error Detection Soccer/inference_all_predictions.csv",
    "/mnt/mydata/dq/projects/Splittree/test/Error Detection soccer/active_cleaning_loop_runs_soccer_v4/iter_3_infer/inference_all_predictions.csv",
    "/mnt/mydata/dq/projects/Splittree/test/Error Detection soccer/active_cleaning_loop_runs_soccer_v4/iter_3/infer/inference_all_predictions.csv",
    "/mnt/mydata/dq/projects/Splittree/test/Error Detection soccer/inference_all_predictions.csv",
]

DEFAULT_OUTPUT_FILE = "predicted_error_positions.csv"


# ============================================================
# 2. Utilities
# ============================================================

def resolve_input_file(input_file: str) -> str:
    """
    Resolve input file robustly.

    If input_file is:
      - a CSV file: use it directly
      - a directory: search inference_all_predictions.csv inside it
      - empty: try COMMON_INPUT_CANDIDATES
    """
    p = str(input_file).strip() if input_file is not None else ""

    if p:
        if os.path.isdir(p):
            direct = os.path.join(p, "inference_all_predictions.csv")
            if os.path.exists(direct) and os.path.isfile(direct):
                return direct

            recursive_hits = list(Path(p).glob("**/inference_all_predictions.csv"))
            recursive_hits = [x for x in recursive_hits if x.is_file()]
            if len(recursive_hits) == 1:
                return str(recursive_hits[0])
            if len(recursive_hits) > 1:
                print("[WARN] 你传入的是目录，且里面找到多个 inference_all_predictions.csv：")
                for x in recursive_hits[:30]:
                    print(f"  - {x}")
                raise ValueError("请用 --input_file 指定其中一个具体 CSV 文件。")

            raise IsADirectoryError(
                f"你传入的 input_file 是目录，不是 CSV 文件: {p}\n"
                f"并且该目录下没有找到 inference_all_predictions.csv。"
            )

        if not os.path.exists(p):
            raise FileNotFoundError(f"指定的 input_file 不存在: {p}")

        if not os.path.isfile(p):
            raise ValueError(f"指定的 input_file 不是普通文件: {p}")

        return p

    for cand in COMMON_INPUT_CANDIDATES:
        if os.path.exists(cand) and os.path.isfile(cand):
            return cand

    msg = [
        "没有找到输入文件。请用 --input_file 指定 inference_all_predictions.csv。",
        "",
        "脚本尝试过以下路径：",
    ]
    for cand in COMMON_INPUT_CANDIDATES:
        msg.append(f"  - {cand}")
    raise FileNotFoundError("\n".join(msg))


def safe_col_list(cols: List[str], df: pd.DataFrame) -> List[str]:
    seen = set()
    out = []
    for c in cols:
        if c in df.columns and c not in seen:
            out.append(c)
            seen.add(c)
    return out


def to_numeric_if_exists(df: pd.DataFrame, cols: List[str]):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def to_int_if_exists(df: pd.DataFrame, cols: List[str]):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)


def build_error_mask(df: pd.DataFrame) -> pd.Series:
    """
    Build predicted-error mask.

    Priority:
      1. final_pred_label
      2. final_pred_label_name
      3. detector_pred_label
      4. detector_pred_label_name
      5. raw_pred_label
      6. raw_pred_label_name
    """
    if "final_pred_label" in df.columns:
        return pd.to_numeric(df["final_pred_label"], errors="coerce").fillna(0).astype(int) == 1

    if "final_pred_label_name" in df.columns:
        return df["final_pred_label_name"].astype(str).str.strip().str.lower().eq("error")

    if "detector_pred_label" in df.columns:
        print("[WARN] 没有 final_pred_label，退回使用 detector_pred_label")
        return pd.to_numeric(df["detector_pred_label"], errors="coerce").fillna(0).astype(int) == 1

    if "detector_pred_label_name" in df.columns:
        print("[WARN] 没有 final_pred_label_name，退回使用 detector_pred_label_name")
        return df["detector_pred_label_name"].astype(str).str.strip().str.lower().eq("error")

    if "raw_pred_label" in df.columns:
        print("[WARN] 没有 final/detector 标签，退回使用 raw_pred_label")
        return pd.to_numeric(df["raw_pred_label"], errors="coerce").fillna(0).astype(int) == 1

    if "raw_pred_label_name" in df.columns:
        print("[WARN] 没有 final/detector 标签，退回使用 raw_pred_label_name")
        return df["raw_pred_label_name"].astype(str).str.strip().str.lower().eq("error")

    raise ValueError(
        "文件中没有找到可用预测标签字段："
        "final_pred_label / final_pred_label_name / detector_pred_label / "
        "detector_pred_label_name / raw_pred_label / raw_pred_label_name"
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_file", default="", help="Soccer inference_all_predictions.csv or a directory containing it.")
    p.add_argument("--output_file", default=DEFAULT_OUTPUT_FILE, help="Output predicted_error_positions.csv")
    p.add_argument(
        "--keep_correct_for_debug",
        action="store_true",
        help="输出全部 cell，并增加 selected_as_error 字段；默认只输出预测为 error 的 cell。",
    )
    return p.parse_args()


# ============================================================
# 3. Main
# ============================================================

def main():
    args = parse_args()
    input_file = resolve_input_file(args.input_file)
    output_file = args.output_file

    df = pd.read_csv(input_file, encoding="utf-8-sig")

    print("读取完成")
    print(f"输入文件: {input_file}")
    print(f"总记录数: {len(df)}")
    print(f"字段数: {len(df.columns)}")

    required = {"row_id", "column"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"输入文件缺少必要定位字段: {missing}")

    error_mask = build_error_mask(df)

    if args.keep_correct_for_debug:
        error_df = df.copy()
        error_df["selected_as_error"] = error_mask.astype(int)
    else:
        error_df = df.loc[error_mask].copy()

    # ------------------------------------------------------------
    # A. Core locating fields
    # ------------------------------------------------------------
    base_cols = [
        "row_id",
        "column",
        "value",
    ]

    # ------------------------------------------------------------
    # B. Label / propagation fields
    # ------------------------------------------------------------
    label_cols = [
        "label",
        "label_binary",
        "label_source",
        "sample_weight",
        "is_outside_candidate",
        "propagation_confidence",
    ]

    # ------------------------------------------------------------
    # C. Prediction outputs
    # ------------------------------------------------------------
    final_pred_cols = [
        "final_pred_label",
        "final_pred_label_name",
        "final_pred_error_prob_before_adjust",
        "final_pred_error_prob",
    ]

    detector_cols = [
        "detector_pred_error_prob_before_adjust",
        "detector_pred_error_prob",
        "detector_pred_label",
        "detector_pred_label_name",
        "detector_margin_to_threshold",
        "detector_threshold",
        "detector_threshold_used",
        "is_stage_disagree",
    ]

    raw_cols = [
        "raw_pred_error_prob",
        "raw_pred_label",
        "raw_pred_label_name",
        "raw_margin_to_threshold",
        "stage1_high_conf_error",
        "stage1_mid_zone",
    ]

    verifier_cols = [
        "verifier_used",
        "verifier_keep_error_prob",
        "verifier_decision",
        "verifier_threshold_used",
        "group_verifier_threshold",
        "verifier_rejected",
    ]

    # ------------------------------------------------------------
    # D. General evidence fields
    # ------------------------------------------------------------
    general_evidence_cols = [
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
        "candidate_rule_ratio",
        "has_neighbor_signal",
        "high_neighbor_consistency",
        "is_rare_value",
        "neighbor_disagree",
        "is_equal_to_neighbor_majority",
        "neighbor_agreement_score",
        "rule_total_count",
        "strong_rule_ratio",
        "context_rule_ratio",
        "fd_like_ratio",
        "global_rule_ratio",
        "rule_strength_sum",
        "candidate_rule_dominant",
        "rule_neighbor_conflict",
        "evidence_only_fd_count",
    ]

    # ------------------------------------------------------------
    # E. Soccer-specific rule counts and profile signals
    # ------------------------------------------------------------
    soccer_rule_cols = [
        "soccer_domain_rule_count",
        "soccer_format_rule_count",
        "soccer_numeric_rule_count",
        "soccer_consistency_rule_count",
        "soccer_rule_total_count",
        "soccer_domain_rule_ratio",
        "soccer_format_rule_ratio",
        "soccer_numeric_rule_ratio",
        "soccer_consistency_rule_ratio",
        "domain_invalid_flag",
        "soccer_consistency_flag",
        "high_soccer_consistency_score",
        "has_soccer_domain_signal",
        "has_soccer_format_signal",
        "has_soccer_numeric_signal",
        "has_soccer_consistency_signal",
        "soccer_signal_strength",
        "soccer_primary_target_signal",
    ]

    soccer_value_cols = [
        "year_value",
        "canonical_position",
        "domain_valid",
        "soccer_value_bucket",
        "row_birthyear_int",
        "row_season_int",
        "row_age_at_season",
        "row_position_canonical",
        "birthyear_season_gap",
        "is_young_player_age",
        "is_old_player_age",
        "position_invalid_or_normalization",
    ]

    soccer_target_cols = [
        "target_is_primary_column",
        "target_is_context_only_column",
        "is_context_only_weak_signal",
    ]

    soccer_consistency_cols = [
        "birthyear_value_invalid",
        "season_value_invalid",
        "position_value_invalid",
        "position_needs_canonicalization",
        "birthyear_season_age_conflict",
        "team_context_conflict",
        "format_rule_signal",
        "fd_like_signal",
        "has_birthyear_or_season_signal",
        "has_team_context_signal",
        "has_fd_like_signal",
        "soccer_profile_inconsistency_count",
        "soccer_consistency_score",
        "soccer_evidence_flags",
        "soccer_attribution_bucket",
        "soccer_consistency_bucket",
        "soccer_precision_gate_applied",
        "soccer_precision_gate_reason",
    ]

    soccer_row_context_cols = [
        "row_team",
        "row_city",
        "row_stadium",
        "row_manager",
    ]

    # ------------------------------------------------------------
    # F. Buckets / sampling
    # ------------------------------------------------------------
    bucket_cols = [
        "pattern_bucket",
        "rarity_bucket",
        "neighbor_bucket",
        "domain_bucket",
        "soccer_consistency_bucket",
        "bucket_id",
        "cluster_id",
        "dist_to_center",
        "sample_role",
        "is_sampled",
        "signal_priority",
    ]

    # ------------------------------------------------------------
    # G. LLM annotation / explanation fields
    # ------------------------------------------------------------
    llm_cols = [
        "error_type",
        "reason_short",
        "reason_detailed",
        "suggested_correct_value",
        "needs_human_review",
        "evidence_used",
    ]

    # ------------------------------------------------------------
    # H. Analysis / hard case fields
    # ------------------------------------------------------------
    analysis_cols = [
        "current_value_pattern",
        "dominant_pattern",
        "dominant_pattern_ratio",
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
        "is_hard_case",
        "hard_case_reason",
        "is_high_risk",
        "high_risk",
    ]

    # ------------------------------------------------------------
    # I. Compatibility fields accidentally retained from Adult pipeline
    #    Soccer input example still contains these columns. Keep them so that
    #    downstream scripts do not break if they were already expecting them.
    # ------------------------------------------------------------
    adult_compat_cols = [
        "adult_domain_rule_count",
        "adult_format_rule_count",
        "adult_consistency_rule_count",
        "age_lower",
        "age_upper",
        "hours_lower",
        "hours_upper",
        "canonical_income",
        "education_order",
        "adult_value_bucket",
        "adult_profile_inconsistency_count",
        "adult_consistency_score",
        "adult_evidence_flags",
        "adult_consistency_bucket",
        "age_sampling_bucket",
        "hours_sampling_bucket",
        "adult_v2_attribution_bucket",
        "time_window_rule_count",
        "time_value_minutes",
        "time_window_bucket",
        "time_value_bucket",
        "adult_rule_total_count",
        "adult_domain_rule_ratio",
        "adult_format_rule_ratio",
        "adult_consistency_rule_ratio",
        "adult_consistency_flag",
        "high_adult_consistency_score",
        "has_adult_domain_signal",
        "has_adult_format_signal",
        "has_adult_consistency_signal",
        "age_span",
        "hours_span",
        "is_minor_age_bucket",
        "is_high_hours_bucket",
        "relationship_v2_conflict_count",
        "has_relationship_v2_signal",
        "has_education_v2_signal",
        "adult_v2_signal_strength",
        "adult_v2_primary_target_signal",
        "row_missing_count",
        "row_non_missing_count",
        "row_age_lower",
        "row_age_upper",
        "row_hours_lower",
        "row_hours_upper",
        "row_education_order",
        "row_income_canonical",
    ]

    debug_cols = ["selected_as_error"] if args.keep_correct_for_debug else []

    wanted_cols = (
        base_cols
        + label_cols
        + final_pred_cols
        + detector_cols
        + raw_cols
        + verifier_cols
        + general_evidence_cols
        + soccer_rule_cols
        + soccer_value_cols
        + soccer_target_cols
        + soccer_consistency_cols
        + soccer_row_context_cols
        + bucket_cols
        + llm_cols
        + analysis_cols
        + adult_compat_cols
        + debug_cols
    )

    keep_cols = safe_col_list(wanted_cols, error_df)
    result_df = error_df[keep_cols].copy()

    # ------------------------------------------------------------
    # J. Type cleanup
    # ------------------------------------------------------------
    if "row_id" in result_df.columns:
        result_df["row_id"] = pd.to_numeric(result_df["row_id"], errors="raise").astype(int)

    numeric_cols = [
        "sample_weight",
        "propagation_confidence",
        "final_pred_error_prob_before_adjust",
        "final_pred_error_prob",
        "detector_pred_error_prob_before_adjust",
        "detector_pred_error_prob",
        "detector_margin_to_threshold",
        "detector_threshold",
        "detector_threshold_used",
        "raw_pred_error_prob",
        "raw_margin_to_threshold",
        "verifier_keep_error_prob",
        "verifier_threshold_used",
        "group_verifier_threshold",
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
        "candidate_rule_ratio",
        "neighbor_agreement_score",
        "rule_total_count",
        "strong_rule_ratio",
        "context_rule_ratio",
        "fd_like_ratio",
        "global_rule_ratio",
        "rule_strength_sum",
        "evidence_only_fd_count",
        "soccer_domain_rule_count",
        "soccer_format_rule_count",
        "soccer_numeric_rule_count",
        "soccer_consistency_rule_count",
        "soccer_rule_total_count",
        "soccer_domain_rule_ratio",
        "soccer_format_rule_ratio",
        "soccer_numeric_rule_ratio",
        "soccer_consistency_rule_ratio",
        "soccer_signal_strength",
        "year_value",
        "row_birthyear_int",
        "row_season_int",
        "row_age_at_season",
        "birthyear_season_gap",
        "soccer_profile_inconsistency_count",
        "soccer_consistency_score",
        "dist_to_center",
        "dominant_pattern_ratio",
        # Adult compatibility numeric fields
        "adult_domain_rule_count",
        "adult_format_rule_count",
        "adult_consistency_rule_count",
        "age_lower",
        "age_upper",
        "hours_lower",
        "hours_upper",
        "education_order",
        "adult_profile_inconsistency_count",
        "adult_consistency_score",
        "time_window_rule_count",
        "time_value_minutes",
        "adult_rule_total_count",
        "adult_domain_rule_ratio",
        "adult_format_rule_ratio",
        "adult_consistency_rule_ratio",
        "age_span",
        "hours_span",
        "relationship_v2_conflict_count",
        "adult_v2_signal_strength",
        "row_missing_count",
        "row_non_missing_count",
        "row_age_lower",
        "row_age_upper",
        "row_hours_lower",
        "row_hours_upper",
        "row_education_order",
    ]
    to_numeric_if_exists(result_df, numeric_cols)

    int_cols = [
        "label_binary",
        "is_outside_candidate",
        "final_pred_label",
        "detector_pred_label",
        "raw_pred_label",
        "stage1_high_conf_error",
        "stage1_mid_zone",
        "is_stage_disagree",
        "verifier_used",
        "verifier_rejected",
        "target_is_primary_column",
        "target_is_context_only_column",
        "birthyear_value_invalid",
        "season_value_invalid",
        "position_value_invalid",
        "position_needs_canonicalization",
        "birthyear_season_age_conflict",
        "team_context_conflict",
        "format_rule_signal",
        "fd_like_signal",
        "has_neighbor_signal",
        "high_neighbor_consistency",
        "is_rare_value",
        "neighbor_disagree",
        "is_equal_to_neighbor_majority",
        "candidate_rule_dominant",
        "domain_invalid_flag",
        "soccer_consistency_flag",
        "high_soccer_consistency_score",
        "has_soccer_domain_signal",
        "has_soccer_format_signal",
        "has_soccer_numeric_signal",
        "has_soccer_consistency_signal",
        "is_young_player_age",
        "is_old_player_age",
        "position_invalid_or_normalization",
        "has_birthyear_or_season_signal",
        "has_team_context_signal",
        "has_fd_like_signal",
        "is_context_only_weak_signal",
        "soccer_primary_target_signal",
        "soccer_precision_gate_applied",
        "is_sampled",
        "rare_only_signal",
        "is_weak_only",
        "is_fp_risk_col",
        "high_prob_but_neighbor_support_current",
        "is_mid_prob",
        "is_near_threshold",
        "rule_neighbor_conflict",
        "verifier_borderline",
        "verifier_reject_borderline",
        "fp_conflict_fd_hardcase",
        "sample_borderline_hardcase",
        "recall_recovery_hardcase",
        "persistent_fp_high_risk",
        "is_hard_case",
        "is_high_risk",
        "high_risk",
        # Adult compatibility int fields
        "adult_consistency_flag",
        "high_adult_consistency_score",
        "has_adult_domain_signal",
        "has_adult_format_signal",
        "has_adult_consistency_signal",
        "is_minor_age_bucket",
        "is_high_hours_bucket",
        "has_relationship_v2_signal",
        "has_education_v2_signal",
        "adult_v2_primary_target_signal",
        "selected_as_error",
    ]
    to_int_if_exists(result_df, int_cols)

    # ------------------------------------------------------------
    # K. Sorting and output
    # ------------------------------------------------------------
    sort_cols = []
    if "row_id" in result_df.columns:
        sort_cols.append("row_id")
    if "column" in result_df.columns:
        sort_cols.append("column")
    if sort_cols:
        result_df = result_df.sort_values(sort_cols).reset_index(drop=True)

    if result_df.empty:
        print("[WARN] 没有筛选出任何预测错误单元格，请检查预测标签字段。")

    result_df.to_csv(output_file, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------
    # L. Console summary
    # ------------------------------------------------------------
    print("\n========== Soccer 预测错误单元格定位结果 ==========")
    print(f"最终预测为错误的单元格数: {len(result_df)}")
    print(f"输出文件: {output_file}")

    if "column" in result_df.columns:
        print("\n---- 各列预测错误数量 Top 30 ----")
        print(result_df["column"].value_counts().head(30).to_string())

    if "semantic_type" in result_df.columns:
        print("\n---- semantic_type 分布 Top 30 ----")
        print(result_df["semantic_type"].value_counts(dropna=False).head(30).to_string())

    if "main_rule_type" in result_df.columns:
        print("\n---- main_rule_type 分布 Top 30 ----")
        print(result_df["main_rule_type"].value_counts(dropna=False).head(30).to_string())

    if "main_usage_role" in result_df.columns:
        print("\n---- main_usage_role 分布 ----")
        print(result_df["main_usage_role"].value_counts(dropna=False).head(30).to_string())

    if "soccer_attribution_bucket" in result_df.columns:
        print("\n---- Soccer attribution bucket 分布 Top 30 ----")
        print(result_df["soccer_attribution_bucket"].value_counts(dropna=False).head(30).to_string())

    if "soccer_consistency_bucket" in result_df.columns:
        print("\n---- Soccer consistency bucket 分布 Top 30 ----")
        print(result_df["soccer_consistency_bucket"].value_counts(dropna=False).head(30).to_string())

    if "soccer_precision_gate_reason" in result_df.columns:
        print("\n---- Soccer precision gate reason 分布 Top 30 ----")
        print(result_df["soccer_precision_gate_reason"].value_counts(dropna=False).head(30).to_string())

    if "soccer_evidence_flags" in result_df.columns:
        print("\n---- Soccer evidence flags Top 30 ----")
        print(result_df["soccer_evidence_flags"].value_counts(dropna=False).head(30).to_string())

    if "is_hard_case" in result_df.columns:
        hard_case_count = pd.to_numeric(result_df["is_hard_case"], errors="coerce").fillna(0).astype(int).sum()
        print(f"\n难例数量 is_hard_case=1: {hard_case_count}")

    high_risk_col = "is_high_risk" if "is_high_risk" in result_df.columns else ("high_risk" if "high_risk" in result_df.columns else None)
    if high_risk_col:
        high_risk_count = pd.to_numeric(result_df[high_risk_col], errors="coerce").fillna(0).astype(int).sum()
        print(f"高风险数量 {high_risk_col}=1: {high_risk_count}")

    if "verifier_rejected" in result_df.columns:
        rejected_count = pd.to_numeric(result_df["verifier_rejected"], errors="coerce").fillna(0).astype(int).sum()
        print(f"Verifier rejected 数量: {rejected_count}")

    if "verifier_decision" in result_df.columns:
        print("\n---- Verifier 决策分布 ----")
        print(result_df["verifier_decision"].value_counts(dropna=False).to_string())

    print("\n---- 前 20 个错误位置 ----")
    if len(result_df) > 0:
        display_cols = safe_col_list(
            [
                "row_id",
                "column",
                "value",
                "final_pred_label_name",
                "final_pred_error_prob",
                "detector_pred_label_name",
                "detector_pred_error_prob",
                "semantic_type",
                "main_rule_type",
                "main_usage_role",
                "soccer_evidence_flags",
                "soccer_attribution_bucket",
                "soccer_precision_gate_reason",
                "is_hard_case",
                "hard_case_reason",
                "is_high_risk",
                "high_risk",
            ],
            result_df,
        )
        print(result_df[display_cols].head(20).to_string(index=False))
    else:
        print("[EMPTY]")


if __name__ == "__main__":
    main()

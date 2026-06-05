
"""
Export predicted error positions for Adult dataset.

Adapted from the Flights exporter, but keeps Adult-specific fields:
relationship consistency, sex/education/age/hours rules, Adult v2 hardcase flags,
domain/format/consistency signals, verifier outputs, and detector outputs.

Usage:
    python export_predicted_error_positions_adult.py \
      --input_file /path/to/inference_all_predictions.csv \
      --output_file predicted_error_positions.csv

If --input_file is omitted, the script tries common Adult paths and ./inference_all_predictions.csv.
"""

import argparse
import os
from typing import List

import pandas as pd


COMMON_INPUT_CANDIDATES = [
    "/mnt/mydata/dq/projects/Splittree/test/Error Detection Adult/active_cleaning_loop_runs_adult_v4/iter_3_infer/inference_all_predictions.csv"
]


DEFAULT_OUTPUT_FILE = "predicted_error_positions.csv"


def resolve_input_file(input_file: str) -> str:
    if input_file and str(input_file).strip():
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"指定的 input_file 不存在: {input_file}")
        return input_file

    for p in COMMON_INPUT_CANDIDATES:
        if os.path.exists(p):
            return p

    raise FileNotFoundError(
        "没有找到输入文件。请用 --input_file 指定 inference_all_predictions.csv。"
    )


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
    p.add_argument("--input_file", default="", help="Adult inference_all_predictions.csv")
    p.add_argument("--output_file", default=DEFAULT_OUTPUT_FILE, help="Output predicted_error_positions.csv")
    p.add_argument(
        "--keep_correct_for_debug",
        action="store_true",
        help="输出全部 cell，并增加 selected_as_error 字段；默认只输出预测为 error 的 cell。",
    )
    return p.parse_args()


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

    base_cols = ["row_id", "column", "value"]

    label_cols = [
        "label", "label_binary", "label_source", "sample_weight",
        "is_outside_candidate", "propagation_confidence",
    ]

    final_pred_cols = [
        "final_pred_label", "final_pred_label_name",
        "final_pred_error_prob_before_adjust", "final_pred_error_prob",
    ]

    detector_cols = [
        "detector_pred_error_prob_before_adjust",
        "detector_pred_error_prob",
        "detector_pred_label",
        "detector_pred_label_name",
        "detector_margin_to_threshold",
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
        "verifier_keep_error_prob",
        "verifier_decision",
        "verifier_threshold_used",
        "group_verifier_threshold",
    ]

    general_evidence_cols = [
        "semantic_type", "detected_type",
        "violation_count", "conflict_score",
        "value_frequency", "value_frequency_rank",
        "neighbor_majority_value", "neighbor_majority_ratio",
        "prior_error_probability", "posterior_error_probability",
        "main_rule_type", "main_usage_role",
        "strong_rule_count", "candidate_generation_rule_count",
        "fd_like_count", "context_rule_count", "global_rule_count",
        "rare_value_count", "typo_rule_count", "pattern_rule_count", "schema_rule_count",
        "time_window_rule_count", "time_value_minutes",
        "candidate_rule_ratio", "has_neighbor_signal", "high_neighbor_consistency",
        "is_rare_value", "neighbor_disagree", "is_equal_to_neighbor_majority",
        "neighbor_agreement_score", "rule_total_count", "strong_rule_ratio",
        "context_rule_ratio", "fd_like_ratio", "global_rule_ratio",
        "rule_strength_sum", "candidate_rule_dominant", "rule_neighbor_conflict",
    ]

    adult_rule_cols = [
        "adult_domain_rule_count", "adult_format_rule_count", "adult_consistency_rule_count",
        "adult_rule_total_count", "adult_domain_rule_ratio", "adult_format_rule_ratio",
        "adult_consistency_rule_ratio", "domain_invalid_flag", "adult_consistency_flag",
        "high_adult_consistency_score", "has_adult_domain_signal", "has_adult_format_signal",
        "has_adult_consistency_signal", "adult_signal_hardcase",
    ]

    adult_numeric_cols = [
        "age_lower", "age_upper", "hours_lower", "hours_upper",
        "canonical_income", "domain_valid", "education_order", "adult_value_bucket",
        "adult_profile_inconsistency_count", "adult_consistency_score", "adult_evidence_flags",
        "row_age_lower", "row_age_upper", "row_hours_lower", "row_hours_upper",
        "row_education_order", "row_income_canonical",
        "row_missing_count", "row_non_missing_count",
        "age_span", "hours_span", "is_minor_age_bucket", "is_high_hours_bucket",
    ]

    adult_target_cols = [
        "target_is_primary_column", "target_is_context_only_column", "is_context_only_weak_signal",
    ]

    adult_relationship_cols = [
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
    ]

    bucket_cols = [
        "pattern_bucket", "rarity_bucket", "neighbor_bucket", "domain_bucket",
        "adult_consistency_bucket", "age_sampling_bucket", "hours_sampling_bucket",
        "time_window_bucket", "time_value_bucket", "bucket_id", "cluster_id",
        "dist_to_center", "sample_role", "is_sampled", "signal_priority",
    ]

    llm_cols = [
        "error_type", "reason_short", "reason_detailed",
        "suggested_correct_value", "needs_human_review", "evidence_used",
    ]

    analysis_cols = [
        "current_value_pattern", "dominant_pattern", "dominant_pattern_ratio",
        "rare_only_signal", "is_weak_only", "is_fp_risk_col",
        "high_prob_but_neighbor_support_current",
        "is_mid_prob", "is_near_threshold",
        "verifier_borderline", "verifier_reject_borderline",
        "fp_conflict_fd_hardcase", "sample_borderline_hardcase",
        "recall_recovery_hardcase", "persistent_fp_high_risk",
        "is_hard_case", "is_high_risk",
    ]

    debug_cols = ["selected_as_error"] if args.keep_correct_for_debug else []

    wanted_cols = (
        base_cols + label_cols + final_pred_cols + detector_cols + raw_cols
        + verifier_cols + general_evidence_cols + adult_rule_cols
        + adult_numeric_cols + adult_target_cols + adult_relationship_cols
        + bucket_cols + llm_cols + analysis_cols + debug_cols
    )

    keep_cols = safe_col_list(wanted_cols, error_df)
    result_df = error_df[keep_cols].copy()

    if "row_id" in result_df.columns:
        result_df["row_id"] = pd.to_numeric(result_df["row_id"], errors="raise").astype(int)

    numeric_cols = [
        "sample_weight", "propagation_confidence",
        "final_pred_error_prob_before_adjust", "final_pred_error_prob",
        "detector_pred_error_prob_before_adjust", "detector_pred_error_prob",
        "detector_margin_to_threshold", "detector_threshold_used",
        "raw_pred_error_prob", "raw_margin_to_threshold",
        "verifier_keep_error_prob", "verifier_threshold_used", "group_verifier_threshold",
        "violation_count", "conflict_score", "value_frequency", "value_frequency_rank",
        "neighbor_majority_ratio", "prior_error_probability", "posterior_error_probability",
        "strong_rule_count", "candidate_generation_rule_count", "fd_like_count",
        "context_rule_count", "global_rule_count", "rare_value_count", "typo_rule_count",
        "pattern_rule_count", "schema_rule_count", "time_window_rule_count", "time_value_minutes",
        "candidate_rule_ratio", "neighbor_agreement_score", "rule_total_count",
        "strong_rule_ratio", "context_rule_ratio", "fd_like_ratio", "global_rule_ratio",
        "rule_strength_sum", "adult_domain_rule_count", "adult_format_rule_count",
        "adult_consistency_rule_count", "adult_rule_total_count", "adult_domain_rule_ratio",
        "adult_format_rule_ratio", "adult_consistency_rule_ratio",
        "age_lower", "age_upper", "hours_lower", "hours_upper",
        "education_order", "adult_profile_inconsistency_count", "adult_consistency_score",
        "row_age_lower", "row_age_upper", "row_hours_lower", "row_hours_upper",
        "row_education_order", "row_missing_count", "row_non_missing_count",
        "age_span", "hours_span", "relationship_v2_conflict_count",
        "adult_v2_signal_strength", "dist_to_center", "dominant_pattern_ratio",
    ]
    to_numeric_if_exists(result_df, numeric_cols)

    int_cols = [
        "label_binary", "is_outside_candidate",
        "final_pred_label", "detector_pred_label", "raw_pred_label",
        "stage1_high_conf_error", "stage1_mid_zone", "is_stage_disagree",
        "domain_invalid_flag", "adult_consistency_flag", "high_adult_consistency_score",
        "has_adult_domain_signal", "has_adult_format_signal", "has_adult_consistency_signal",
        "target_is_primary_column", "target_is_context_only_column",
        "is_relationship_spouse_value", "relationship_marital_conflict",
        "relationship_age_conflict", "relationship_sex_conflict", "sex_value_invalid",
        "education_age_extreme_conflict", "has_relationship_v2_rule", "has_education_v2_rule",
        "is_minor_age_bucket", "is_high_hours_bucket",
        "has_relationship_v2_signal", "has_education_v2_signal",
        "adult_v2_primary_target_signal", "is_context_only_weak_signal",
        "is_sampled", "rare_only_signal", "is_weak_only", "is_fp_risk_col",
        "high_prob_but_neighbor_support_current", "is_mid_prob", "is_near_threshold",
        "rule_neighbor_conflict", "adult_signal_hardcase", "verifier_borderline",
        "verifier_reject_borderline", "fp_conflict_fd_hardcase", "sample_borderline_hardcase",
        "recall_recovery_hardcase", "adult_v2_relationship_hardcase",
        "adult_v2_sex_invalid_hardcase", "adult_v2_context_fp_hardcase",
        "persistent_fp_high_risk", "is_hard_case", "is_high_risk", "selected_as_error",
    ]
    to_int_if_exists(result_df, int_cols)

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

    print("\n========== Adult 预测错误单元格定位结果 ==========")
    print(f"最终预测为错误的单元格数: {len(result_df)}")
    print(f"输出文件: {output_file}")

    if "column" in result_df.columns:
        print("\n---- 各列预测错误数量 Top 20 ----")
        print(result_df["column"].value_counts().head(20).to_string())

    if "semantic_type" in result_df.columns:
        print("\n---- semantic_type 分布 Top 20 ----")
        print(result_df["semantic_type"].value_counts(dropna=False).head(20).to_string())

    if "main_rule_type" in result_df.columns:
        print("\n---- main_rule_type 分布 Top 20 ----")
        print(result_df["main_rule_type"].value_counts(dropna=False).head(20).to_string())

    if "adult_v2_attribution_bucket" in result_df.columns:
        print("\n---- Adult v2 attribution bucket 分布 Top 30 ----")
        print(result_df["adult_v2_attribution_bucket"].value_counts(dropna=False).head(30).to_string())

    if "adult_consistency_bucket" in result_df.columns:
        print("\n---- Adult consistency bucket 分布 Top 30 ----")
        print(result_df["adult_consistency_bucket"].value_counts(dropna=False).head(30).to_string())

    if "is_hard_case" in result_df.columns:
        hard_case_count = pd.to_numeric(result_df["is_hard_case"], errors="coerce").fillna(0).astype(int).sum()
        print(f"\n难例数量 is_hard_case=1: {hard_case_count}")

    if "is_high_risk" in result_df.columns:
        high_risk_count = pd.to_numeric(result_df["is_high_risk"], errors="coerce").fillna(0).astype(int).sum()
        print(f"高风险数量 is_high_risk=1: {high_risk_count}")

    if "adult_signal_hardcase" in result_df.columns:
        adult_hard = pd.to_numeric(result_df["adult_signal_hardcase"], errors="coerce").fillna(0).astype(int).sum()
        print(f"Adult signal hardcase 数量: {adult_hard}")

    if "verifier_decision" in result_df.columns:
        print("\n---- Verifier 决策分布 ----")
        print(result_df["verifier_decision"].value_counts(dropna=False).to_string())

    print("\n---- 前 20 个错误位置 ----")
    if len(result_df) > 0:
        display_cols = safe_col_list(
            [
                "row_id", "column", "value",
                "final_pred_label_name", "final_pred_error_prob",
                "detector_pred_label_name", "detector_pred_error_prob",
                "semantic_type", "main_rule_type", "adult_evidence_flags",
                "is_hard_case", "is_high_risk",
            ],
            result_df,
        )
        print(result_df[display_cols].head(20).to_string(index=False))
    else:
        print("[EMPTY]")


if __name__ == "__main__":
    main()

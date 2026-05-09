import os
import pandas as pd


# ============================================================
# 1. 输入输出路径
# ============================================================

# Beers 错误检测推理结果文件
# 按你的实际路径修改这里即可。
INPUT_FILE = (
    "/mnt/mydata/dq/projects/Splittree/test/Error Detection beers/"
    "active_cleaning_loop_runs_beers_v4/iter_3/infer/inference_all_predictions.csv"
)

# 输出到当前运行目录
OUTPUT_FILE = "predicted_error_positions.csv"


# ============================================================
# 2. 工具函数
# ============================================================

def safe_to_int_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="raise").astype(int)


def safe_to_numeric(df: pd.DataFrame, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def pick_error_mask(df: pd.DataFrame):
    """
    自动兼容不同版本检测输出字段。
    优先使用 verifier 后的 final_pred_label；
    如果没有 final 字段，再退回 detector 字段；
    如果 detector 字段也没有，再退回 raw 字段。
    """
    if "final_pred_label" in df.columns:
        print("[INFO] 使用 final_pred_label == 1 作为最终错误位置")
        return pd.to_numeric(df["final_pred_label"], errors="coerce").fillna(0).astype(int) == 1

    if "final_pred_label_name" in df.columns:
        print("[INFO] 使用 final_pred_label_name == error 作为最终错误位置")
        return (
            df["final_pred_label_name"]
            .astype(str)
            .str.strip()
            .str.lower()
            == "error"
        )

    if "detector_pred_label" in df.columns:
        print("[WARN] 没有 final_pred_label，退回使用 detector_pred_label == 1")
        return pd.to_numeric(df["detector_pred_label"], errors="coerce").fillna(0).astype(int) == 1

    if "detector_pred_label_name" in df.columns:
        print("[WARN] 没有 final_pred_label_name，退回使用 detector_pred_label_name == error")
        return (
            df["detector_pred_label_name"]
            .astype(str)
            .str.strip()
            .str.lower()
            == "error"
        )

    if "raw_pred_label" in df.columns:
        print("[WARN] 没有 final/detector 标签，退回使用 raw_pred_label == 1")
        return pd.to_numeric(df["raw_pred_label"], errors="coerce").fillna(0).astype(int) == 1

    if "raw_pred_label_name" in df.columns:
        print("[WARN] 没有 final/detector 标签，退回使用 raw_pred_label_name == error")
        return (
            df["raw_pred_label_name"]
            .astype(str)
            .str.strip()
            .str.lower()
            == "error"
        )

    raise ValueError(
        "文件中没有找到可用的预测标签字段。支持："
        "final_pred_label / final_pred_label_name / "
        "detector_pred_label / detector_pred_label_name / "
        "raw_pred_label / raw_pred_label_name"
    )


# ============================================================
# 3. 读取文件
# ============================================================

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"输入文件不存在: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

print("读取完成")
print(f"输入文件: {INPUT_FILE}")
print(f"总记录数: {len(df)}")
print(f"字段数: {len(df.columns)}")

required_base = {"row_id", "column"}
if not required_base.issubset(set(df.columns)):
    raise ValueError(f"输入文件必须至少包含字段: {required_base}")


# ============================================================
# 4. 筛选最终预测为 error 的 cell
# ============================================================

error_mask = pick_error_mask(df)
error_df = df.loc[error_mask].copy()


# ============================================================
# 5. 只保留错误定位与分析相关字段
# ============================================================

base_cols = [
    "row_id",
    "column",
    "value",
]

label_cols = [
    "label",
    "label_binary",
    "label_source",
    "sample_weight",
    "is_outside_candidate",
    "propagation_confidence",
    "sample_role",
    "is_sampled",
    "error_type",
    "reason_short",
    "reason_detailed",
    "suggested_correct_value",
    "needs_human_review",
    "evidence_used",
]

evidence_cols = [
    "semantic_type",
    "violation_count",
    "conflict_score",
    "value_frequency",
    "value_frequency_rank",
    "numeric_value",
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
    "numeric_window_rule_count",
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
]

bucket_cols = [
    "pattern_bucket",
    "rarity_bucket",
    "neighbor_bucket",
    "numeric_window_bucket",
    "numeric_value_bucket",
    "bucket_id",
    "cluster_id",
    "dist_to_center",
]

risk_cols = [
    "rare_only_signal",
    "is_weak_only",
    "is_fp_risk_col",
    "rule_neighbor_conflict",
    "numeric_borderline_hardcase",
    "semantic_recovery_hardcase",
    "persistent_fp_high_risk",
    "ounces_canonical_reject_hardcase",
    "is_hard_case",
    "is_high_risk",
]

raw_detector_cols = [
    "raw_pred_error_prob",
    "raw_pred_label",
    "raw_pred_label_name",
    "raw_margin_to_threshold",
    "stage1_high_conf_error",
    "stage1_mid_zone",
    "detector_pred_error_prob_before_adjust",
    "detector_pred_error_prob",
    "detector_pred_label",
    "detector_pred_label_name",
    "detector_margin_to_threshold",
    "detector_threshold_used",
    "is_stage_disagree",
]

verifier_final_cols = [
    "verifier_keep_error_prob",
    "verifier_decision",
    "verifier_threshold_used",
    "group_verifier_threshold",
    "final_pred_label",
    "final_pred_label_name",
    "final_pred_error_prob_before_adjust",
    "final_pred_error_prob",
    "is_mid_prob",
    "is_near_threshold",
    "verifier_borderline",
    "verifier_reject_borderline",
]

wanted_cols = (
    base_cols
    + label_cols
    + evidence_cols
    + bucket_cols
    + risk_cols
    + raw_detector_cols
    + verifier_final_cols
)

keep_cols = [col for col in wanted_cols if col in error_df.columns]
result_df = error_df[keep_cols].copy()


# ============================================================
# 6. 类型整理
# ============================================================

if "row_id" in result_df.columns:
    result_df["row_id"] = safe_to_int_series(result_df["row_id"])

numeric_cols = [
    "label_binary",
    "sample_weight",
    "is_outside_candidate",
    "propagation_confidence",
    "violation_count",
    "conflict_score",
    "value_frequency",
    "value_frequency_rank",
    "numeric_value",
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
    "numeric_window_rule_count",
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
    "bucket_id",
    "cluster_id",
    "dist_to_center",
    "rare_only_signal",
    "is_weak_only",
    "is_fp_risk_col",
    "raw_pred_error_prob",
    "raw_pred_label",
    "raw_margin_to_threshold",
    "stage1_high_conf_error",
    "stage1_mid_zone",
    "detector_pred_error_prob_before_adjust",
    "detector_pred_error_prob",
    "detector_pred_label",
    "detector_margin_to_threshold",
    "detector_threshold_used",
    "is_stage_disagree",
    "verifier_keep_error_prob",
    "verifier_threshold_used",
    "group_verifier_threshold",
    "final_pred_label",
    "final_pred_error_prob_before_adjust",
    "final_pred_error_prob",
    "is_mid_prob",
    "is_near_threshold",
    "verifier_borderline",
    "verifier_reject_borderline",
    "rule_neighbor_conflict",
    "numeric_borderline_hardcase",
    "semantic_recovery_hardcase",
    "persistent_fp_high_risk",
    "ounces_canonical_reject_hardcase",
    "is_hard_case",
    "is_high_risk",
]

result_df = safe_to_numeric(result_df, numeric_cols)


# ============================================================
# 7. 去重与排序
# ============================================================

before_dedup = len(result_df)
if {"row_id", "column"}.issubset(set(result_df.columns)):
    result_df = result_df.drop_duplicates(subset=["row_id", "column"], keep="first")

sort_cols = [c for c in ["row_id", "column"] if c in result_df.columns]
if sort_cols:
    result_df = result_df.sort_values(sort_cols).reset_index(drop=True)


# ============================================================
# 8. 输出
# ============================================================

if result_df.empty:
    print("[WARN] 没有筛选出任何预测错误单元格，请检查 final_pred_label / final_pred_label_name 等预测标签字段。")

result_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")


# ============================================================
# 9. 打印统计信息
# ============================================================

print("\n========== Beers 预测错误单元格定位结果 ==========")
print(f"最终预测为错误的原始记录数: {len(error_df)}")
print(f"去重前记录数: {before_dedup}")
print(f"最终预测为错误的唯一单元格数: {len(result_df)}")
print(f"输出文件: {os.path.abspath(OUTPUT_FILE)}")

if "column" in result_df.columns:
    print("\n---- 各列预测错误数量 Top 30 ----")
    print(result_df["column"].value_counts().head(30).to_string())

if "semantic_type" in result_df.columns:
    print("\n---- semantic_type 分布 ----")
    print(result_df["semantic_type"].value_counts(dropna=False).to_string())

if "main_rule_type" in result_df.columns:
    print("\n---- main_rule_type 分布 ----")
    print(result_df["main_rule_type"].value_counts(dropna=False).head(20).to_string())

if "verifier_decision" in result_df.columns:
    print("\n---- Verifier 决策分布 ----")
    print(result_df["verifier_decision"].value_counts(dropna=False).to_string())

for flag_col in [
    "is_hard_case",
    "is_high_risk",
    "numeric_borderline_hardcase",
    "semantic_recovery_hardcase",
    "persistent_fp_high_risk",
    "ounces_canonical_reject_hardcase",
]:
    if flag_col in result_df.columns:
        cnt = pd.to_numeric(result_df[flag_col], errors="coerce").fillna(0).astype(int).sum()
        print(f"{flag_col}=1 数量: {cnt}")

print("\n---- 前 20 个错误位置 ----")
print(result_df.head(20).to_string(index=False))

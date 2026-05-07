import os
import pandas as pd


# ============================================================
# 1. 输入输出路径
# ============================================================

INPUT_FILE = "/mnt/mydata/dq/projects/Splittree/test/Error Detection flights/active_cleaning_loop_runs_flights_v4/iter_3/infer/inference_all_predictions.csv"

# 建议输出到和输入文件同目录，避免你在别的目录运行时找不到结果
OUTPUT_FILE = "predicted_error_positions.csv"


# ============================================================
# 2. 读取文件
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("读取完成")
print(f"输入文件: {INPUT_FILE}")
print(f"总记录数: {len(df)}")
print(f"字段数: {len(df.columns)}")


# ============================================================
# 3. 兼容不同版本字段：确定最终预测标签字段
# ============================================================

if "final_pred_label" in df.columns:
    # 推荐优先使用这个，最稳定：1 = error, 0 = correct
    error_mask = df["final_pred_label"].fillna(0).astype(int) == 1

elif "final_pred_label_name" in df.columns:
    # 兼容字符串版本：error / correct
    error_mask = (
        df["final_pred_label_name"]
        .astype(str)
        .str.strip()
        .str.lower()
        == "error"
    )

elif "detector_pred_label" in df.columns:
    # 如果没有 verifier 后的最终结果，就退回 detector 结果
    print("警告：没有 final_pred_label，退回使用 detector_pred_label")
    error_mask = df["detector_pred_label"].fillna(0).astype(int) == 1

elif "detector_pred_label_name" in df.columns:
    print("警告：没有 final_pred_label_name，退回使用 detector_pred_label_name")
    error_mask = (
        df["detector_pred_label_name"]
        .astype(str)
        .str.strip()
        .str.lower()
        == "error"
    )

else:
    raise ValueError(
        "文件中没有找到可用的预测标签字段："
        "final_pred_label / final_pred_label_name / "
        "detector_pred_label / detector_pred_label_name"
    )


error_df = df.loc[error_mask].copy()


# ============================================================
# 4. 只保留错误定位与分析相关字段
# ============================================================

# 最核心的定位字段
base_cols = [
    "row_id",
    "column",
    "value",
]

# 最终预测结果字段
final_pred_cols = [
    "final_pred_label",
    "final_pred_label_name",
    "final_pred_error_prob",
    "final_pred_error_prob_before_adjust",
]

# detector 阶段字段，方便看最终结果来自哪里
detector_cols = [
    "detector_pred_error_prob",
    "detector_pred_label",
    "detector_pred_label_name",
    "detector_margin_to_threshold",
    "detector_threshold_used",
]

# raw 阶段字段，方便对比是否经过二阶段/调整
raw_cols = [
    "raw_pred_error_prob",
    "raw_pred_label",
    "raw_pred_label_name",
    "raw_margin_to_threshold",
]

# verifier 字段，方便看是否被 verifier 保留
verifier_cols = [
    "verifier_keep_error_prob",
    "verifier_decision",
    "verifier_threshold_used",
    "group_verifier_threshold",
]

# 难例 / 高风险 / 边界样本字段
analysis_cols = [
    "is_hard_case",
    "is_high_risk",
    "is_mid_prob",
    "is_near_threshold",
    "verifier_borderline",
    "verifier_reject_borderline",
    "fp_conflict_fd_hardcase",
    "sample_borderline_hardcase",
    "recall_recovery_hardcase",
    "persistent_fp_high_risk",
]

# 规则与上下文字段，方便后续分析为什么被判错
evidence_cols = [
    "semantic_type",
    "main_rule_type",
    "main_usage_role",
    "violation_count",
    "conflict_score",
    "neighbor_majority_value",
    "neighbor_majority_ratio",
    "prior_error_probability",
    "posterior_error_probability",
    "rule_neighbor_conflict",
    "pattern_bucket",
    "rarity_bucket",
    "neighbor_bucket",
    "time_window_bucket",
    "time_value_bucket",
]

# 如果有 LLM 标注信息，也保留
llm_cols = [
    "label",
    "label_binary",
    "label_source",
    "sample_role",
    "is_sampled",
    "error_type",
    "reason_short",
    "reason_detailed",
    "suggested_correct_value",
    "needs_human_review",
    "evidence_used",
]

wanted_cols = (
    base_cols
    + final_pred_cols
    + detector_cols
    + raw_cols
    + verifier_cols
    + analysis_cols
    + evidence_cols
    + llm_cols
)

keep_cols = [col for col in wanted_cols if col in error_df.columns]

result_df = error_df[keep_cols].copy()


# ============================================================
# 5. 类型整理
# ============================================================

if "row_id" in result_df.columns:
    result_df["row_id"] = pd.to_numeric(result_df["row_id"], errors="coerce")

prob_cols = [
    "final_pred_error_prob",
    "final_pred_error_prob_before_adjust",
    "detector_pred_error_prob",
    "raw_pred_error_prob",
    "verifier_keep_error_prob",
    "prior_error_probability",
    "posterior_error_probability",
    "neighbor_majority_ratio",
    "conflict_score",
]

for col in prob_cols:
    if col in result_df.columns:
        result_df[col] = pd.to_numeric(result_df[col], errors="coerce")


# ============================================================
# 6. 排序
#    建议按 row_id、column 排序，便于回到原表定位
# ============================================================

sort_cols = []

if "row_id" in result_df.columns:
    sort_cols.append("row_id")

if "column" in result_df.columns:
    sort_cols.append("column")

if sort_cols:
    result_df = result_df.sort_values(sort_cols).reset_index(drop=True)


# ============================================================
# 7. 输出
# ============================================================

result_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")


# ============================================================
# 8. 打印统计信息
# ============================================================

print("\n========== 预测错误单元格定位结果 ==========")
print(f"最终预测为错误的单元格数: {len(result_df)}")
print(f"输出文件: {OUTPUT_FILE}")

if "column" in result_df.columns:
    print("\n---- 各列预测错误数量 Top 20 ----")
    print(result_df["column"].value_counts().head(20).to_string())

if "is_hard_case" in result_df.columns:
    hard_case_count = pd.to_numeric(result_df["is_hard_case"], errors="coerce").fillna(0).astype(int).sum()
    print(f"\n难例数量 is_hard_case=1: {hard_case_count}")

if "is_high_risk" in result_df.columns:
    high_risk_count = pd.to_numeric(result_df["is_high_risk"], errors="coerce").fillna(0).astype(int).sum()
    print(f"高风险数量 is_high_risk=1: {high_risk_count}")

if "verifier_decision" in result_df.columns:
    print("\n---- Verifier 决策分布 ----")
    print(result_df["verifier_decision"].value_counts(dropna=False).to_string())

print("\n---- 前 20 个错误位置 ----")
print(result_df.head(20).to_string(index=False))
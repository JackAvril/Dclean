import os
import pandas as pd


# ============================================================
# 1. 输入输出路径
# ============================================================

# Movies 错误检测推理结果文件
# 如果你的实际路径不同，只需要改这里。
INPUT_FILE = "/mnt/mydata/dq/projects/Splittree/test/Error Detection movies/active_cleaning_loop_runs_movies_v4/iter_3/infer/inference_all_predictions.csv"



# 输出到当前运行目录
OUTPUT_FILE = "predicted_error_positions.csv"

# 额外诊断输出，方便后续候选生成 / 白名单 / 误差分析使用
OUTPUT_COLUMN_SUMMARY = "predicted_error_positions_column_summary.csv"
OUTPUT_BUCKET_SUMMARY = "predicted_error_positions_bucket_summary.csv"
OUTPUT_RISK_SUMMARY = "predicted_error_positions_risk_summary.csv"


# ============================================================
# 2. 工具函数
# ============================================================

def safe_numeric_series(s: pd.Series, fill_value=None):
    out = pd.to_numeric(s, errors="coerce")
    if fill_value is not None:
        out = out.fillna(fill_value)
    return out


def safe_to_numeric_inplace(df: pd.DataFrame, cols):
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
        return safe_numeric_series(df["final_pred_label"], fill_value=0).astype(int) == 1

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
        return safe_numeric_series(df["detector_pred_label"], fill_value=0).astype(int) == 1

    if "detector_pred_label_name" in df.columns:
        print("[WARN] 没有 final_pred_label_name，退回使用 detector_pred_label_name == error")
        return (
            df["detector_pred_label_name"]
            .astype(str)
            .str.strip()
            .str.lower()
            == "error"
        )

    if "pred_label" in df.columns:
        print("[WARN] 没有 final/detector 标签，退回使用 pred_label == 1")
        return safe_numeric_series(df["pred_label"], fill_value=0).astype(int) == 1

    if "pred_label_name" in df.columns:
        print("[WARN] 没有 final/detector 标签，退回使用 pred_label_name == error")
        return (
            df["pred_label_name"]
            .astype(str)
            .str.strip()
            .str.lower()
            == "error"
        )

    if "raw_pred_label" in df.columns:
        print("[WARN] 没有 final/detector/pred 标签，退回使用 raw_pred_label == 1")
        return safe_numeric_series(df["raw_pred_label"], fill_value=0).astype(int) == 1

    if "raw_pred_label_name" in df.columns:
        print("[WARN] 没有 final/detector/pred 标签，退回使用 raw_pred_label_name == error")
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
        "pred_label / pred_label_name / "
        "raw_pred_label / raw_pred_label_name"
    )


def ensure_required_columns(df: pd.DataFrame):
    required_base = {"row_id", "column"}
    if not required_base.issubset(set(df.columns)):
        raise ValueError(
            f"输入文件必须至少包含字段: {required_base}，"
            f"当前字段为: {list(df.columns)}"
        )


def normalize_key_columns(df: pd.DataFrame):
    out = df.copy()

    if "row_id" in out.columns:
        out["row_id"] = pd.to_numeric(out["row_id"], errors="raise").astype(int)

    if "column" in out.columns:
        out["column"] = out["column"].astype(str).str.strip()

    return out


def save_if_not_empty(df: pd.DataFrame, path: str):
    if df is not None and not df.empty:
        df.to_csv(path, index=False, encoding="utf-8-sig")


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

ensure_required_columns(df)


# ============================================================
# 4. 兼容不同版本字段：确定最终预测为 error 的 cell
# ============================================================

error_mask = pick_error_mask(df)
error_df = df.loc[error_mask].copy()


# ============================================================
# 5. 保留错误定位与分析相关字段
# ============================================================

# ------------------------------------------------------------
# A. 最核心定位字段
# ------------------------------------------------------------
base_cols = [
    "row_id",
    "column",
    "value",
]

# ------------------------------------------------------------
# B. LLM 标注 / 标签传播 / 采样字段
# ------------------------------------------------------------
label_cols = [
    "label",
    "label_binary",
    "label_source",
    "sample_weight",
    "is_outside_candidate",
    "propagation_confidence",
    "sample_role",
    "is_sampled",
    "sampling_priority_score",
    "error_type",
    "reason_short",
    "reason_detailed",
    "suggested_correct_value",
    "needs_human_review",
    "evidence_used",
]

# ------------------------------------------------------------
# C. 通用规则与上下文证据字段
# ------------------------------------------------------------
generic_evidence_cols = [
    "semantic_type",
    "detected_type",
    "main_rule_type",
    "main_usage_role",
    "violation_count",
    "conflict_score",
    "value_frequency",
    "value_frequency_rank",
    "neighbor_majority_value",
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
    "rare_only_signal",
    "is_weak_only",
]

# ------------------------------------------------------------
# D. Movies 专属规则字段
#    你给的 movies 格式里包含 movie_typed_rule_count /
#    movie_consistency_rule_count / movie_rule_total_count 等。
# ------------------------------------------------------------
movie_rule_cols = [
    "movie_typed_rule_count",
    "movie_consistency_rule_count",
    "movie_typed_rule_ratio",
    "movie_consistency_rule_ratio",
    "movie_rule_total_count",
    "has_movie_rule_signal",
]

# ------------------------------------------------------------
# E. Movies 专属解析字段
#    用于后续候选生成和误差分析：
#    年份、上映国家、时长、评分、计数、评论、列表长度等。
# ------------------------------------------------------------
movie_parsed_cols = [
    "parsed_year",
    "parsed_release_year",
    "parsed_release_country",
    "duration_minutes",
    "rating_value_float",
    "count_value_int",
    "review_total",
    "list_token_count",
    "list_token_count_bucket",
    "text_length_bucket",
    "has_mojibake_or_control_chars",
]

# ------------------------------------------------------------
# F. Movies 行级上下文字段
#    用于分析当前 cell 与所在电影行的其他字段是否一致。
# ------------------------------------------------------------
movie_row_context_cols = [
    "movie_row_year",
    "movie_row_release_year",
    "movie_row_release_country",
    "movie_row_release_year_gap",
    "movie_row_duration_minutes",
    "movie_row_rating_value_float",
    "movie_row_rating_count_int",
    "movie_row_review_total",
    "movie_row_review_to_rating_ratio",
    "movie_row_actors_cast_overlap",
]

# ------------------------------------------------------------
# G. Movies 字段类型标记
# ------------------------------------------------------------
movie_type_flag_cols = [
    "is_movie_time_col",
    "is_movie_rating_col",
    "is_movie_people_col",
    "is_movie_list_col",
    "is_movie_text_col",
]

# ------------------------------------------------------------
# H. 分桶 / 聚类字段
# ------------------------------------------------------------
bucket_cols = [
    "pattern_bucket",
    "rarity_bucket",
    "neighbor_bucket",

    # Movies 专属 bucket
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

    # 通用聚类
    "bucket_id",
    "cluster_id",
    "dist_to_center",
]

# ------------------------------------------------------------
# I. raw / stage1 / detector 预测字段
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# J. verifier / final 预测字段
# ------------------------------------------------------------
verifier_final_cols = [
    "verifier_keep_error_prob",
    "verifier_decision",
    "verifier_threshold_used",
    "group_verifier_threshold",
    "verifier_bypass",
    "final_pred_label",
    "final_pred_label_name",
    "final_pred_error_prob_before_adjust",
    "final_pred_error_prob",
    "is_mid_prob",
    "is_near_threshold",
    "verifier_borderline",
    "verifier_reject_borderline",
]

# ------------------------------------------------------------
# K. 难例 / 高风险 / 边界样本字段
# ------------------------------------------------------------
risk_cols = [
    "is_fp_risk_col",
    "rule_neighbor_conflict",
    "fp_conflict_fd_hardcase",
    "sample_borderline_hardcase",
    "recall_recovery_hardcase",
    "persistent_fp_high_risk",
    "is_hard_case",
    "is_high_risk",
]

# 汇总所有希望保留的字段
wanted_cols = (
    base_cols
    + label_cols
    + generic_evidence_cols
    + movie_rule_cols
    + movie_parsed_cols
    + movie_row_context_cols
    + movie_type_flag_cols
    + bucket_cols
    + raw_detector_cols
    + verifier_final_cols
    + risk_cols
)

# 只保留真实存在字段，避免不同版本输出字段不一致时报错
keep_cols = [col for col in wanted_cols if col in error_df.columns]

result_df = error_df[keep_cols].copy()


# ============================================================
# 6. 类型整理
# ============================================================

result_df = normalize_key_columns(result_df)

# 数值型字段统一转 numeric，便于后续脚本读取
numeric_cols = [
    # labels / sampling
    "label_binary",
    "sample_weight",
    "is_outside_candidate",
    "propagation_confidence",
    "is_sampled",
    "sampling_priority_score",
    "needs_human_review",

    # generic evidence
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

    # movie rule fields
    "movie_typed_rule_count",
    "movie_consistency_rule_count",
    "movie_typed_rule_ratio",
    "movie_consistency_rule_ratio",
    "movie_rule_total_count",
    "has_movie_rule_signal",

    # movie parsed fields
    "parsed_year",
    "parsed_release_year",
    "duration_minutes",
    "rating_value_float",
    "count_value_int",
    "review_total",
    "list_token_count",
    "has_mojibake_or_control_chars",

    # movie row context
    "movie_row_year",
    "movie_row_release_year",
    "movie_row_release_year_gap",
    "movie_row_duration_minutes",
    "movie_row_rating_value_float",
    "movie_row_rating_count_int",
    "movie_row_review_total",
    "movie_row_review_to_rating_ratio",
    "movie_row_actors_cast_overlap",

    # movie type flags
    "is_movie_time_col",
    "is_movie_rating_col",
    "is_movie_people_col",
    "is_movie_list_col",
    "is_movie_text_col",

    # clustering
    "bucket_id",
    "cluster_id",
    "dist_to_center",

    # model outputs
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
    "verifier_bypass",
    "final_pred_label",
    "final_pred_error_prob_before_adjust",
    "final_pred_error_prob",
    "is_mid_prob",
    "is_near_threshold",

    # risk
    "is_fp_risk_col",
    "rule_neighbor_conflict",
    "verifier_borderline",
    "verifier_reject_borderline",
    "fp_conflict_fd_hardcase",
    "sample_borderline_hardcase",
    "recall_recovery_hardcase",
    "persistent_fp_high_risk",
    "is_hard_case",
    "is_high_risk",
]

result_df = safe_to_numeric_inplace(result_df, numeric_cols)

# 字符串字段整理
for col in [
    "column",
    "value",
    "semantic_type",
    "detected_type",
    "main_rule_type",
    "main_usage_role",
    "parsed_release_country",
    "movie_row_release_country",
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
    "sample_role",
    "error_type",
    "reason_short",
    "reason_detailed",
    "suggested_correct_value",
    "evidence_used",
    "final_pred_label_name",
    "detector_pred_label_name",
    "raw_pred_label_name",
    "verifier_decision",
]:
    if col in result_df.columns:
        result_df[col] = result_df[col].astype(str).str.strip()


# ============================================================
# 7. 去重与排序
# ============================================================

before_dedup = len(result_df)

if {"row_id", "column"}.issubset(set(result_df.columns)):
    result_df = result_df.drop_duplicates(subset=["row_id", "column"], keep="first")

sort_cols = []
for c in ["row_id", "column"]:
    if c in result_df.columns:
        sort_cols.append(c)

if sort_cols:
    result_df = result_df.sort_values(sort_cols).reset_index(drop=True)


# ============================================================
# 8. 输出主文件
# ============================================================

if result_df.empty:
    print("[WARN] 没有筛选出任何预测错误单元格，请检查 final_pred_label / final_pred_label_name 等预测标签字段。")

result_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")


# ============================================================
# 9. 额外诊断输出
# ============================================================

# 9.1 列级错误数量与平均置信度
if not result_df.empty and "column" in result_df.columns:
    agg_dict = {
        "row_id": "count",
    }

    if "final_pred_error_prob" in result_df.columns:
        agg_dict["final_pred_error_prob"] = "mean"
    if "detector_pred_error_prob" in result_df.columns:
        agg_dict["detector_pred_error_prob"] = "mean"
    if "posterior_error_probability" in result_df.columns:
        agg_dict["posterior_error_probability"] = "mean"
    if "conflict_score" in result_df.columns:
        agg_dict["conflict_score"] = "mean"
    if "violation_count" in result_df.columns:
        agg_dict["violation_count"] = "mean"
    if "movie_rule_total_count" in result_df.columns:
        agg_dict["movie_rule_total_count"] = "mean"

    column_summary = result_df.groupby("column", dropna=False).agg(agg_dict).reset_index()
    column_summary = column_summary.rename(columns={"row_id": "predicted_error_cell_count"})
    column_summary = column_summary.sort_values("predicted_error_cell_count", ascending=False).reset_index(drop=True)
    column_summary.to_csv(OUTPUT_COLUMN_SUMMARY, index=False, encoding="utf-8-sig")
else:
    column_summary = pd.DataFrame()

# 9.2 bucket 诊断
bucket_summary_rows = []
for bucket_col in [
    "movie_field_bucket",
    "movie_rule_bucket",
    "year_bucket",
    "release_year_gap_bucket",
    "duration_bucket",
    "rating_bucket",
    "count_bucket",
    "review_ratio_bucket",
    "actors_cast_overlap_bucket",
    "list_bucket",
    "text_bucket",
    "pattern_bucket",
    "rarity_bucket",
    "neighbor_bucket",
]:
    if bucket_col in result_df.columns:
        vc = result_df[bucket_col].value_counts(dropna=False).reset_index()
        vc.columns = [bucket_col, "count"]
        vc.insert(0, "bucket_column", bucket_col)
        vc = vc.rename(columns={bucket_col: "bucket_value"})
        bucket_summary_rows.append(vc)

if bucket_summary_rows:
    bucket_summary = pd.concat(bucket_summary_rows, ignore_index=True)
    bucket_summary.to_csv(OUTPUT_BUCKET_SUMMARY, index=False, encoding="utf-8-sig")
else:
    bucket_summary = pd.DataFrame()

# 9.3 risk 诊断
risk_summary_rows = []
for flag_col in [
    "is_hard_case",
    "is_high_risk",
    "is_fp_risk_col",
    "rule_neighbor_conflict",
    "fp_conflict_fd_hardcase",
    "sample_borderline_hardcase",
    "recall_recovery_hardcase",
    "persistent_fp_high_risk",
]:
    if flag_col in result_df.columns:
        s = pd.to_numeric(result_df[flag_col], errors="coerce").fillna(0).astype(int)
        risk_summary_rows.append({
            "flag": flag_col,
            "count_1": int((s == 1).sum()),
            "count_0": int((s == 0).sum()),
            "ratio_1": float((s == 1).mean()) if len(s) > 0 else 0.0,
        })

if risk_summary_rows:
    risk_summary = pd.DataFrame(risk_summary_rows)
    risk_summary.to_csv(OUTPUT_RISK_SUMMARY, index=False, encoding="utf-8-sig")
else:
    risk_summary = pd.DataFrame()


# ============================================================
# 10. 打印统计信息
# ============================================================

print("\n========== Movies 预测错误单元格定位结果 ==========")
print(f"最终预测为错误的原始记录数: {len(error_df)}")
print(f"去重前记录数: {before_dedup}")
print(f"最终预测为错误的唯一单元格数: {len(result_df)}")
print(f"输出文件: {os.path.abspath(OUTPUT_FILE)}")

if not column_summary.empty:
    print(f"列级统计输出: {os.path.abspath(OUTPUT_COLUMN_SUMMARY)}")

if not bucket_summary.empty:
    print(f"bucket 统计输出: {os.path.abspath(OUTPUT_BUCKET_SUMMARY)}")

if not risk_summary.empty:
    print(f"risk 统计输出: {os.path.abspath(OUTPUT_RISK_SUMMARY)}")

if "column" in result_df.columns:
    print("\n---- 各列预测错误数量 Top 30 ----")
    print(result_df["column"].value_counts().head(30).to_string())

if "semantic_type" in result_df.columns:
    print("\n---- semantic_type 分布 ----")
    print(result_df["semantic_type"].value_counts(dropna=False).head(30).to_string())

if "detected_type" in result_df.columns:
    print("\n---- detected_type 分布 ----")
    print(result_df["detected_type"].value_counts(dropna=False).head(30).to_string())

if "main_rule_type" in result_df.columns:
    print("\n---- main_rule_type 分布 ----")
    print(result_df["main_rule_type"].value_counts(dropna=False).head(30).to_string())

if "main_usage_role" in result_df.columns:
    print("\n---- main_usage_role 分布 ----")
    print(result_df["main_usage_role"].value_counts(dropna=False).head(30).to_string())

for bucket_col in [
    "movie_field_bucket",
    "movie_rule_bucket",
    "year_bucket",
    "release_year_gap_bucket",
    "duration_bucket",
    "rating_bucket",
    "count_bucket",
    "review_ratio_bucket",
    "actors_cast_overlap_bucket",
    "list_bucket",
    "text_bucket",
]:
    if bucket_col in result_df.columns:
        print(f"\n---- {bucket_col} 分布 Top 20 ----")
        print(result_df[bucket_col].value_counts(dropna=False).head(20).to_string())

if "verifier_decision" in result_df.columns:
    print("\n---- Verifier 决策分布 ----")
    print(result_df["verifier_decision"].value_counts(dropna=False).to_string())

if "final_pred_label_name" in result_df.columns:
    print("\n---- final_pred_label_name 分布 ----")
    print(result_df["final_pred_label_name"].value_counts(dropna=False).to_string())

for flag_col in [
    "is_hard_case",
    "is_high_risk",
    "is_fp_risk_col",
    "rule_neighbor_conflict",
    "fp_conflict_fd_hardcase",
    "sample_borderline_hardcase",
    "recall_recovery_hardcase",
    "persistent_fp_high_risk",
]:
    if flag_col in result_df.columns:
        cnt = pd.to_numeric(result_df[flag_col], errors="coerce").fillna(0).astype(int).sum()
        print(f"{flag_col}=1 数量: {int(cnt)}")

if "final_pred_error_prob" in result_df.columns:
    print("\n---- final_pred_error_prob 描述统计 ----")
    print(pd.to_numeric(result_df["final_pred_error_prob"], errors="coerce").describe().to_string())

if "posterior_error_probability" in result_df.columns:
    print("\n---- posterior_error_probability 描述统计 ----")
    print(pd.to_numeric(result_df["posterior_error_probability"], errors="coerce").describe().to_string())

print("\n---- 前 20 个错误位置 ----")
if result_df.empty:
    print("[EMPTY]")
else:
    print(result_df.head(20).to_string(index=False))

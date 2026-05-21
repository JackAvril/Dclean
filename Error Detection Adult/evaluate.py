import os
import pandas as pd
from typing import Set, Tuple, Dict, Any, List


# ============================================================
# 1. 用户配置区
# ============================================================

CLEAN_CSV = "/mnt/mydata/dq/projects/Splittree/adult/adult_clean.csv"
DIRTY_CSV = "/mnt/mydata/dq/projects/Splittree/adult/adult_dirty.csv"

# 默认评估普通推理输出。
# 如果你跑的是主动学习主控脚本，可改成：
# "/mnt/mydata/dq/projects/Splittree/test/Error Detection adult/active_cleaning_loop_runs_adult_v4/iter_3_infer/inference_all_predictions.csv"
# 或 best_model 对应的推理目录输出。
INFER_RESULT_CSV = "/mnt/mydata/dq/projects/Splittree/test/Error Detection adult/two_stage_inference_output_adult_v4_with_verifier/inference_all_predictions.csv"

OUTPUT_DIR = "eval_outputs_adult_v4_with_verifier"

DROP_UNNAMED_COLUMNS = True

MISSING_TOKENS = {
    "", "empty", "null", "none", "nan", "na", "n/a", "missing", "unk",
    "unknown", "?", "not available", "contact airline"
}

# adult 数据集列，主要用于后续按列统计和明细字段排序
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

# 是否额外输出按列统计、按规则统计
EXPORT_GROUP_SUMMARIES = True


# ============================================================
# 2. 基础函数
# ============================================================

def maybe_drop_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not DROP_UNNAMED_COLUMNS:
        return df.copy()
    keep_cols = [c for c in df.columns if not str(c).lower().startswith("unnamed:")]
    return df[keep_cols].copy()


def normalize_value(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.lower() in MISSING_TOKENS:
        return None
    return s


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        df[col] = df[col].map(normalize_value)
    return df


def safe_int(x, default=0):
    try:
        if pd.isna(x):
            return default
        return int(float(x))
    except Exception:
        return default


def safe_float(x, default=None):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def load_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df = maybe_drop_unnamed_columns(df)
    df = normalize_df(df)
    return df


def make_key(row_id: int, col: str) -> Tuple[int, str]:
    return int(row_id), str(col)


def bool_from_any(v, default=False) -> bool:
    if pd.isna(v):
        return default
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "error"}:
        return True
    if s in {"0", "false", "no", "n", "correct"}:
        return False
    try:
        return int(float(s)) == 1
    except Exception:
        return default


# ============================================================
# 3. 构造真实错误集合
# ============================================================

def build_true_error_cells(
    clean_df: pd.DataFrame,
    dirty_df: pd.DataFrame
) -> Tuple[Set[Tuple[int, str]], Dict[Tuple[int, str], Dict[str, Any]]]:
    if clean_df.shape != dirty_df.shape:
        raise ValueError(f"干净表和脏表形状不一致: clean={clean_df.shape}, dirty={dirty_df.shape}")

    if list(clean_df.columns) != list(dirty_df.columns):
        raise ValueError(
            "干净表和脏表列名或列顺序不一致，请先对齐。\n"
            f"clean columns={list(clean_df.columns)}\n"
            f"dirty columns={list(dirty_df.columns)}"
        )

    true_errors = set()
    true_error_info = {}

    for row_id in range(len(clean_df)):
        for col in clean_df.columns:
            clean_val = clean_df.iat[row_id, clean_df.columns.get_loc(col)]
            dirty_val = dirty_df.iat[row_id, dirty_df.columns.get_loc(col)]
            if clean_val != dirty_val:
                key = (row_id, col)
                true_errors.add(key)
                true_error_info[key] = {
                    "clean_value": clean_val,
                    "dirty_value": dirty_val
                }

    return true_errors, true_error_info


# ============================================================
# 4. 解析推理结果
# ============================================================

def parse_label_from_value(v) -> bool:
    if pd.isna(v):
        return False
    val = str(v).strip().lower()

    if val in {"1", "true", "yes", "y"}:
        return True
    if val in {"0", "false", "no", "n"}:
        return False

    if val == "error":
        return True
    if val == "correct":
        return False

    try:
        return int(float(val)) == 1
    except Exception:
        return False


def parse_final_error_flag(row: pd.Series) -> bool:
    # 优先使用最终 verifier 后标签
    for col in ["final_pred_label", "pred_label", "raw_pred_label"]:
        if col in row.index and pd.notna(row[col]):
            return parse_label_from_value(row[col])

    for col in ["final_pred_label_name", "pred_label_name", "raw_pred_label_name"]:
        if col in row.index and pd.notna(row[col]):
            return parse_label_from_value(row[col])

    return False


def parse_detector_error_flag(row: pd.Series) -> bool:
    # Detector-only 优先 detector_pred_label；如果没有则退化到 pred/raw
    for col in ["detector_pred_label", "pred_label", "raw_pred_label"]:
        if col in row.index and pd.notna(row[col]):
            return parse_label_from_value(row[col])

    for col in ["detector_pred_label_name", "pred_label_name", "raw_pred_label_name"]:
        if col in row.index and pd.notna(row[col]):
            return parse_label_from_value(row[col])

    return False


def load_infer_result(path: str):
    df = pd.read_csv(path, dtype=str)
    df = maybe_drop_unnamed_columns(df)

    required = ["row_id", "column"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"推理结果文件缺少必要列: {missing}")

    df["row_id"] = df["row_id"].apply(lambda x: safe_int(x, -1))
    df["column"] = df["column"].astype(str)

    # 去掉异常 row_id
    df = df[df["row_id"] >= 0].copy()

    candidate_cells = set((int(r["row_id"]), str(r["column"])) for _, r in df.iterrows())

    predicted_error_cells = set()
    detector_predicted_error_cells = set()
    infer_row_map = {}

    # 如果同一个 cell 有重复记录，后面的会覆盖前面的。正常情况下不会重复。
    for _, row in df.iterrows():
        key = (int(row["row_id"]), str(row["column"]))
        row_dict = row.to_dict()
        infer_row_map[key] = row_dict

        if parse_final_error_flag(row):
            predicted_error_cells.add(key)

        if parse_detector_error_flag(row):
            detector_predicted_error_cells.add(key)

    return df, candidate_cells, predicted_error_cells, detector_predicted_error_cells, infer_row_map


# ============================================================
# 5. 指标计算
# ============================================================

def calc_prf(pred_set: Set[Tuple[int, str]], true_set: Set[Tuple[int, str]]) -> Dict[str, Any]:
    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    precision = tp / len(pred_set) if len(pred_set) > 0 else 0.0
    recall = tp / len(true_set) if len(true_set) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def calc_column_metrics(
    pred_set: Set[Tuple[int, str]],
    true_set: Set[Tuple[int, str]],
    columns: List[str]
) -> pd.DataFrame:
    rows = []
    for col in columns:
        true_col = set(k for k in true_set if k[1] == col)
        pred_col = set(k for k in pred_set if k[1] == col)
        prf = calc_prf(pred_col, true_col)
        rows.append({
            "column": col,
            "true_errors": len(true_col),
            "predicted_errors": len(pred_col),
            "tp": prf["tp"],
            "fp": prf["fp"],
            "fn": prf["fn"],
            "precision": prf["precision"],
            "recall": prf["recall"],
            "f1": prf["f1"],
        })
    return pd.DataFrame(rows)


def calc_candidate_column_metrics(
    candidate_set: Set[Tuple[int, str]],
    true_set: Set[Tuple[int, str]],
    columns: List[str]
) -> pd.DataFrame:
    rows = []
    for col in columns:
        true_col = set(k for k in true_set if k[1] == col)
        cand_col = set(k for k in candidate_set if k[1] == col)
        covered = cand_col & true_col
        missed = true_col - cand_col
        coverage = len(covered) / len(true_col) if len(true_col) > 0 else 0.0
        cand_precision = len(covered) / len(cand_col) if len(cand_col) > 0 else 0.0
        rows.append({
            "column": col,
            "candidate_cells": len(cand_col),
            "true_errors": len(true_col),
            "covered_true_errors": len(covered),
            "missed_true_errors": len(missed),
            "candidate_coverage": coverage,
            "candidate_only_precision": cand_precision,
        })
    return pd.DataFrame(rows)


# ============================================================
# 6. 明细导出
# ============================================================

def get_threshold_value(infer_info: Dict[str, Any]):
    for c in [
        "final_decision_threshold",
        "verifier_decision_threshold",
        "detector_decision_threshold",
        "decision_threshold",
        "stage2_decision_threshold",
        "stage1_decision_threshold",
        "detector_threshold_used",
        "group_verifier_threshold",
        "verifier_threshold_used",
    ]:
        if c in infer_info:
            return infer_info.get(c)
    return None


def get_clean_dirty_value(
    row_id: int,
    col: str,
    clean_df: pd.DataFrame,
    dirty_df: pd.DataFrame
):
    dirty_value = dirty_df.iloc[row_id][col] if row_id < len(dirty_df) and col in dirty_df.columns else None
    clean_value = clean_df.iloc[row_id][col] if row_id < len(clean_df) and col in clean_df.columns else None
    return clean_value, dirty_value


def build_detail_rows(
    keys: Set[Tuple[int, str]],
    clean_df: pd.DataFrame,
    dirty_df: pd.DataFrame,
    infer_row_map: Dict[Tuple[int, str], Dict[str, Any]],
    true_error_info: Dict[Tuple[int, str], Dict[str, Any]],
    detail_type: str
) -> List[Dict[str, Any]]:
    rows = []

    for row_id, col in sorted(keys, key=lambda x: (x[0], x[1])):
        infer_info = infer_row_map.get((row_id, col), {})

        clean_value, dirty_value = get_clean_dirty_value(row_id, col, clean_df, dirty_df)

        row_series = pd.Series(infer_info) if len(infer_info) > 0 else pd.Series(dtype=object)

        out = {
            "detail_type": detail_type,
            "row_id": row_id,
            "column": col,
            "dirty_value": dirty_value,
            "clean_value": clean_value,
            "is_true_error": int((row_id, col) in true_error_info),
            "detector_predicted_error": int(parse_detector_error_flag(row_series)) if len(row_series) > 0 else 0,
            "final_predicted_error": int(parse_final_error_flag(row_series)) if len(row_series) > 0 else 0,
        }

        useful_cols = [
            # 原始与标签来源
            "value",
            "label_source",
            "semantic_type",
            "detected_type",

            # 基础证据
            "violation_count",
            "conflict_score",
            "value_frequency",
            "value_frequency_rank",
            "neighbor_majority_value",
            "neighbor_majority_ratio",
            "prior_error_probability",
            "posterior_error_probability",

            # 规则统计
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

            # adult 专属证据
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

            # 兼容旧字段
            "time_window_rule_count",
            "time_value_minutes",

            # 派生特征
            "candidate_rule_ratio",
            "rule_total_count",
            "strong_rule_ratio",
            "context_rule_ratio",
            "fd_like_ratio",
            "global_rule_ratio",
            "adult_rule_total_count",
            "adult_domain_rule_ratio",
            "adult_format_rule_ratio",
            "adult_consistency_rule_ratio",
            "rule_strength_sum",
            "domain_invalid_flag",
            "adult_consistency_flag",
            "high_adult_consistency_score",
            "has_adult_domain_signal",
            "has_adult_format_signal",
            "has_adult_consistency_signal",
            "age_span",
            "hours_span",
            "is_minor_age_bucket",
            "is_high_hours_bucket",
            "is_rare_value",
            "neighbor_disagree",
            "is_equal_to_neighbor_majority",
            "neighbor_agreement_score",
            "is_weak_only",
            "rare_only_signal",
            "is_fp_risk_col",

            # bucket 与聚类
            "pattern_bucket",
            "rarity_bucket",
            "neighbor_bucket",
            "domain_bucket",
            "adult_consistency_bucket",
            "age_sampling_bucket",
            "hours_sampling_bucket",
            "adult_value_bucket",
            "time_window_bucket",
            "time_value_bucket",
            "bucket_id",
            "cluster_id",
            "dist_to_center",
            "sample_role",
            "is_sampled",
            "signal_priority",

            # LLM/训练附加信息
            "error_type",
            "reason_short",
            "reason_detailed",
            "suggested_correct_value",
            "needs_human_review",
            "evidence_used",

            # stage1
            "raw_pred_error_prob",
            "raw_pred_label",
            "raw_pred_label_name",
            "stage1_decision_threshold",
            "raw_margin_to_threshold",
            "stage1_high_conf_error",
            "stage1_mid_zone",

            # stage2 detector
            "detector_pred_error_prob_before_adjust",
            "detector_pred_error_prob",
            "detector_pred_label",
            "detector_pred_label_name",
            "detector_decision_threshold",
            "detector_threshold_used",
            "detector_margin_to_threshold",

            # verifier
            "verifier_keep_error_prob",
            "verifier_decision",
            "verifier_decision_threshold",
            "verifier_threshold_used",
            "group_verifier_threshold",

            # final
            "final_pred_error_prob_before_adjust",
            "final_pred_error_prob",
            "final_pred_label",
            "final_pred_label_name",
            "final_decision_threshold",

            # hard/high risk flags
            "is_high_risk",
            "is_hard_case",
            "is_mid_prob",
            "is_near_threshold",
            "is_stage_disagree",
            "rule_neighbor_conflict",
            "adult_signal_hardcase",
            "verifier_borderline",
            "verifier_reject_borderline",
            "fp_conflict_fd_hardcase",
            "sample_borderline_hardcase",
            "recall_recovery_hardcase",
            "persistent_fp_high_risk",
        ]

        for c in useful_cols:
            out[c] = infer_info.get(c)

        out["decision_threshold"] = get_threshold_value(infer_info)
        rows.append(out)

    return rows


# ============================================================
# 7. group summary 导出
# ============================================================

def export_group_summaries(
    infer_df: pd.DataFrame,
    true_error_cells: Set[Tuple[int, str]],
    predicted_error_cells: Set[Tuple[int, str]],
    detector_predicted_error_cells: Set[Tuple[int, str]],
    candidate_cells: Set[Tuple[int, str]],
    output_dir: str,
):
    if not EXPORT_GROUP_SUMMARIES:
        return

    all_cols = sorted(infer_df["column"].dropna().astype(str).unique().tolist())

    candidate_col_df = calc_candidate_column_metrics(candidate_cells, true_error_cells, all_cols)
    detector_col_df = calc_column_metrics(detector_predicted_error_cells, true_error_cells, all_cols)
    final_col_df = calc_column_metrics(predicted_error_cells, true_error_cells, all_cols)

    candidate_col_df.to_csv(os.path.join(output_dir, "candidate_metrics_by_column.csv"), index=False, encoding="utf-8-sig")
    detector_col_df.to_csv(os.path.join(output_dir, "detector_metrics_by_column.csv"), index=False, encoding="utf-8-sig")
    final_col_df.to_csv(os.path.join(output_dir, "final_metrics_by_column.csv"), index=False, encoding="utf-8-sig")

    # 推理结果自身的分布统计
    group_cols = [
        "column",
        "main_rule_type",
        "domain_bucket",
        "adult_consistency_bucket",
        "neighbor_bucket",
        "rarity_bucket",
        "pattern_bucket",
        "verifier_decision",
        "final_pred_label_name",
        "detector_pred_label_name",
    ]

    for gcol in group_cols:
        if gcol not in infer_df.columns:
            continue
        tmp = infer_df.copy()
        tmp["_key"] = list(zip(tmp["row_id"].astype(int), tmp["column"].astype(str)))
        tmp["_is_true_error"] = tmp["_key"].isin(true_error_cells).astype(int)
        tmp["_final_pred"] = tmp.apply(parse_final_error_flag, axis=1).astype(int)
        tmp["_detector_pred"] = tmp.apply(parse_detector_error_flag, axis=1).astype(int)

        summary = tmp.groupby(gcol, dropna=False).agg(
            rows=("_key", "count"),
            true_errors=("_is_true_error", "sum"),
            final_pred_errors=("_final_pred", "sum"),
            detector_pred_errors=("_detector_pred", "sum"),
        ).reset_index()

        summary["true_error_rate_in_group"] = summary["true_errors"] / summary["rows"].clip(lower=1)
        summary["final_pred_rate_in_group"] = summary["final_pred_errors"] / summary["rows"].clip(lower=1)
        summary["detector_pred_rate_in_group"] = summary["detector_pred_errors"] / summary["rows"].clip(lower=1)

        safe_name = gcol.replace("/", "_")
        summary.to_csv(os.path.join(output_dir, f"group_summary_by_{safe_name}.csv"), index=False, encoding="utf-8-sig")


# ============================================================
# 8. 主评估流程
# ============================================================

def evaluate_detection(clean_csv: str, dirty_csv: str, infer_csv: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    clean_df = load_table(clean_csv)
    dirty_df = load_table(dirty_csv)

    true_error_cells, true_error_info = build_true_error_cells(clean_df, dirty_df)

    infer_df, candidate_cells, predicted_error_cells, detector_predicted_error_cells, infer_row_map = load_infer_result(infer_csv)

    covered_true_errors = true_error_cells & candidate_cells
    candidate_missed_true_errors = true_error_cells - candidate_cells

    candidate_coverage = len(covered_true_errors) / len(true_error_cells) if len(true_error_cells) > 0 else 0.0
    candidate_only_precision = len(covered_true_errors) / len(candidate_cells) if len(candidate_cells) > 0 else 0.0

    detector_tp_keys = detector_predicted_error_cells & true_error_cells
    detector_fp_keys = detector_predicted_error_cells - true_error_cells
    detector_fn_keys = true_error_cells - detector_predicted_error_cells
    detector_prf = calc_prf(detector_predicted_error_cells, true_error_cells)

    tp_keys = predicted_error_cells & true_error_cells
    fp_keys = predicted_error_cells - true_error_cells
    fn_keys = true_error_cells - predicted_error_cells
    prf = calc_prf(predicted_error_cells, true_error_cells)

    n_verifier_used = 0
    n_verifier_rejected = 0
    if "verifier_keep_error_prob" in infer_df.columns:
        n_verifier_used = int(infer_df["verifier_keep_error_prob"].notna().sum())
    if "verifier_decision" in infer_df.columns:
        n_verifier_rejected = int((infer_df["verifier_decision"].astype(str) == "reject_error").sum())

    # 明细导出
    tp_rows = build_detail_rows(tp_keys, clean_df, dirty_df, infer_row_map, true_error_info, "TP")
    fp_rows = build_detail_rows(fp_keys, clean_df, dirty_df, infer_row_map, true_error_info, "FP")
    fn_rows = build_detail_rows(fn_keys, clean_df, dirty_df, infer_row_map, true_error_info, "FN")

    detector_tp_rows = build_detail_rows(detector_tp_keys, clean_df, dirty_df, infer_row_map, true_error_info, "DETECTOR_TP")
    detector_fp_rows = build_detail_rows(detector_fp_keys, clean_df, dirty_df, infer_row_map, true_error_info, "DETECTOR_FP")
    detector_fn_rows = build_detail_rows(detector_fn_keys, clean_df, dirty_df, infer_row_map, true_error_info, "DETECTOR_FN")

    covered_rows = build_detail_rows(
        covered_true_errors, clean_df, dirty_df, infer_row_map, true_error_info, "CANDIDATE_COVERED_TRUE_ERROR"
    )
    missed_rows = build_detail_rows(
        candidate_missed_true_errors, clean_df, dirty_df, infer_row_map, true_error_info, "CANDIDATE_MISSED_TRUE_ERROR"
    )

    pd.DataFrame(tp_rows).to_csv(os.path.join(output_dir, "tp_details.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(fp_rows).to_csv(os.path.join(output_dir, "fp_details.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(fn_rows).to_csv(os.path.join(output_dir, "fn_details.csv"), index=False, encoding="utf-8-sig")

    pd.DataFrame(detector_tp_rows).to_csv(os.path.join(output_dir, "detector_tp_details.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(detector_fp_rows).to_csv(os.path.join(output_dir, "detector_fp_details.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(detector_fn_rows).to_csv(os.path.join(output_dir, "detector_fn_details.csv"), index=False, encoding="utf-8-sig")

    pd.DataFrame(covered_rows).to_csv(os.path.join(output_dir, "candidate_covered_true_errors.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(missed_rows).to_csv(os.path.join(output_dir, "candidate_missed_true_errors.csv"), index=False, encoding="utf-8-sig")

    export_group_summaries(
        infer_df=infer_df,
        true_error_cells=true_error_cells,
        predicted_error_cells=predicted_error_cells,
        detector_predicted_error_cells=detector_predicted_error_cells,
        candidate_cells=candidate_cells,
        output_dir=output_dir,
    )

    summary = {
        "dataset": "adult",
        "clean_csv": clean_csv,
        "dirty_csv": dirty_csv,
        "infer_csv": infer_csv,

        "n_rows": len(clean_df),
        "n_cols": len(clean_df.columns),
        "n_cells": len(clean_df) * len(clean_df.columns),

        "n_true_errors": len(true_error_cells),

        "n_candidate_cells": len(candidate_cells),
        "n_true_errors_covered_by_candidate": len(covered_true_errors),
        "n_true_errors_missed_by_candidate": len(candidate_missed_true_errors),
        "candidate_coverage": candidate_coverage,
        "candidate_only_precision": candidate_only_precision,

        "n_detector_predicted_errors": len(detector_predicted_error_cells),
        "detector_tp": detector_prf["tp"],
        "detector_fp": detector_prf["fp"],
        "detector_fn": detector_prf["fn"],
        "detector_precision": detector_prf["precision"],
        "detector_recall": detector_prf["recall"],
        "detector_f1": detector_prf["f1"],

        "n_predicted_errors": len(predicted_error_cells),
        "tp": prf["tp"],
        "fp": prf["fp"],
        "fn": prf["fn"],
        "precision": prf["precision"],
        "recall": prf["recall"],
        "f1": prf["f1"],

        "n_verifier_used": n_verifier_used,
        "n_verifier_rejected": n_verifier_rejected,
    }

    if "is_high_risk" in infer_df.columns:
        summary["n_high_risk"] = int(pd.to_numeric(infer_df["is_high_risk"], errors="coerce").fillna(0).sum())
    if "is_hard_case" in infer_df.columns:
        summary["n_hard_case"] = int(pd.to_numeric(infer_df["is_hard_case"], errors="coerce").fillna(0).sum())

    pd.DataFrame([summary]).to_csv(
        os.path.join(output_dir, "evaluation_summary.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print("========== Adult 评估结果 ==========")
    print(f"总行数: {summary['n_rows']}")
    print(f"总列数: {summary['n_cols']}")
    print(f"总单元格数: {summary['n_cells']}")
    print()

    print("---- 真实错误统计 ----")
    print(f"真实错误单元格数: {summary['n_true_errors']}")
    print()

    print("---- 候选范围统计 ----")
    print(f"候选范围单元格数: {summary['n_candidate_cells']}")
    print(f"进入候选范围的真实错误数: {summary['n_true_errors_covered_by_candidate']}")
    print(f"未进入候选范围的真实错误数: {summary['n_true_errors_missed_by_candidate']}")
    print(f"Candidate Coverage: {summary['candidate_coverage']:.6f}")
    print(f"Candidate-Only Precision: {summary['candidate_only_precision']:.6f}")
    print()

    print("---- Detector-Only 指标 ----")
    print(f"Detector 预测为错误的单元格数: {summary['n_detector_predicted_errors']}")
    print(f"Detector TP: {summary['detector_tp']}")
    print(f"Detector FP: {summary['detector_fp']}")
    print(f"Detector FN: {summary['detector_fn']}")
    print(f"Detector Precision: {summary['detector_precision']:.6f}")
    print(f"Detector Recall: {summary['detector_recall']:.6f}")
    print(f"Detector F1: {summary['detector_f1']:.6f}")
    print()

    print("---- 最终错误检测指标（Verifier后）----")
    print(f"最终预测为错误的单元格数: {summary['n_predicted_errors']}")
    print(f"TP: {summary['tp']}")
    print(f"FP: {summary['fp']}")
    print(f"FN: {summary['fn']}")
    print(f"Precision: {summary['precision']:.6f}")
    print(f"Recall: {summary['recall']:.6f}")
    print(f"F1: {summary['f1']:.6f}")
    print()

    print("---- Verifier 统计 ----")
    print(f"Verifier 参与样本数: {summary['n_verifier_used']}")
    print(f"Verifier 打回 correct 数: {summary['n_verifier_rejected']}")
    if "n_high_risk" in summary:
        print(f"High Risk 数量: {summary['n_high_risk']}")
    if "n_hard_case" in summary:
        print(f"Hard Case 数量: {summary['n_hard_case']}")
    print()

    print("---- 明细文件输出 ----")
    print(os.path.join(output_dir, "evaluation_summary.csv"))
    print(os.path.join(output_dir, "tp_details.csv"))
    print(os.path.join(output_dir, "fp_details.csv"))
    print(os.path.join(output_dir, "fn_details.csv"))
    print(os.path.join(output_dir, "detector_tp_details.csv"))
    print(os.path.join(output_dir, "detector_fp_details.csv"))
    print(os.path.join(output_dir, "detector_fn_details.csv"))
    print(os.path.join(output_dir, "candidate_covered_true_errors.csv"))
    print(os.path.join(output_dir, "candidate_missed_true_errors.csv"))

    if EXPORT_GROUP_SUMMARIES:
        print(os.path.join(output_dir, "candidate_metrics_by_column.csv"))
        print(os.path.join(output_dir, "detector_metrics_by_column.csv"))
        print(os.path.join(output_dir, "final_metrics_by_column.csv"))

    return summary


if __name__ == "__main__":
    evaluate_detection(
        clean_csv=CLEAN_CSV,
        dirty_csv=DIRTY_CSV,
        infer_csv=INFER_RESULT_CSV,
        output_dir=OUTPUT_DIR
    )

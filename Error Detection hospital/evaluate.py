import os
import pandas as pd
from typing import Set, Tuple, Dict, Any, List


# ============================================================
# 1. 用户配置区
# ============================================================

CLEAN_CSV = "/mnt/mydata/dq/projects/Splittree/hospital/hospital_clean_no_address23.csv"
DIRTY_CSV = "/mnt/mydata/dq/projects/Splittree/hospital/hospital_dirty_no_address23.csv"

# 改成你最后一轮推理结果文件（detector + verifier 联合推理输出）
INFER_RESULT_CSV = "active_cleaning_loop_runs_v4/iter_3/infer/inference_all_predictions.csv"

OUTPUT_DIR = "eval_outputs_v4_with_verifier"

DROP_UNNAMED_COLUMNS = True
MISSING_TOKENS = {"", "empty", "null", "none", "nan", "na", "n/a", "missing", "unk"}


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


# ============================================================
# 3. 构造真实错误集合
# ============================================================

def build_true_error_cells(
    clean_df: pd.DataFrame,
    dirty_df: pd.DataFrame
) -> Tuple[Set[Tuple[int, str]], Dict[Tuple[int, str], Dict[str, Any]]]:
    """
    返回:
        true_errors: {(row_id, column), ...}
        true_error_info[(row_id, column)] = {
            "clean_value": ...,
            "dirty_value": ...
        }
    """
    if clean_df.shape != dirty_df.shape:
        raise ValueError(f"干净表和脏表形状不一致: clean={clean_df.shape}, dirty={dirty_df.shape}")

    if list(clean_df.columns) != list(dirty_df.columns):
        raise ValueError("干净表和脏表列名或列顺序不一致，请先对齐。")

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

    if val in {"1", "true"}:
        return True
    if val in {"0", "false"}:
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
    """
    按优先级解析最终是否判错：
    1) final_pred_label
    2) final_pred_label_name
    3) pred_label
    4) pred_label_name
    5) raw_pred_label
    6) raw_pred_label_name
    """
    for col in ["final_pred_label", "pred_label", "raw_pred_label"]:
        if col in row.index and pd.notna(row[col]):
            return parse_label_from_value(row[col])

    for col in ["final_pred_label_name", "pred_label_name", "raw_pred_label_name"]:
        if col in row.index and pd.notna(row[col]):
            return parse_label_from_value(row[col])

    return False


def parse_detector_error_flag(row: pd.Series) -> bool:
    """
    专门解析 detector 的输出标签：
    1) detector_pred_label
    2) detector_pred_label_name
    3) pred_label
    4) pred_label_name
    5) raw_pred_label
    6) raw_pred_label_name
    """
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

    candidate_cells = set((int(r["row_id"]), str(r["column"])) for _, r in df.iterrows())

    predicted_error_cells = set()
    detector_predicted_error_cells = set()
    infer_row_map = {}

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
    ]:
        if c in infer_info:
            return infer_info.get(c)
    return None


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

        dirty_value = dirty_df.iloc[row_id][col] if row_id < len(dirty_df) and col in dirty_df.columns else None
        clean_value = clean_df.iloc[row_id][col] if row_id < len(clean_df) and col in clean_df.columns else None

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
            # 基础
            "value",
            "label_source",
            "semantic_type",

            # 规则 / 上下文
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
            "rule_total_count",
            "strong_rule_ratio",
            "context_rule_ratio",
            "fd_like_ratio",
            "global_rule_ratio",
            "rule_strength_sum",

            # bucket / cluster
            "pattern_bucket",
            "rarity_bucket",
            "neighbor_bucket",
            "bucket_id",
            "cluster_id",
            "dist_to_center",
            "sample_role",
            "is_sampled",

            # 理由
            "error_type",
            "reason_short",
            "reason_detailed",
            "suggested_correct_value",
            "needs_human_review",
            "evidence_used",

            # detector
            "raw_pred_error_prob",
            "raw_pred_label",
            "raw_pred_label_name",
            "stage1_decision_threshold",
            "raw_margin_to_threshold",
            "stage1_high_conf_error",
            "stage1_mid_zone",

            "detector_pred_error_prob_before_adjust",
            "detector_pred_error_prob",
            "detector_pred_label",
            "detector_pred_label_name",
            "detector_decision_threshold",
            "detector_margin_to_threshold",

            # verifier
            "verifier_keep_error_prob",
            "verifier_decision",
            "verifier_decision_threshold",

            # final
            "final_pred_error_prob_before_adjust",
            "final_pred_error_prob",
            "final_pred_label",
            "final_pred_label_name",
            "final_decision_threshold",

            # 风险 / 难例
            "is_high_risk",
            "is_hard_case",
            "is_mid_prob",
            "is_near_threshold",
            "is_stage_disagree",
            "rule_neighbor_conflict",
            "is_weak_only",
            "is_fp_risk_col",
            "high_prob_but_neighbor_support_current",
        ]

        for c in useful_cols:
            out[c] = infer_info.get(c)

        out["decision_threshold"] = get_threshold_value(infer_info)

        rows.append(out)

    return rows


# ============================================================
# 7. 主评估流程
# ============================================================

def evaluate_detection(clean_csv: str, dirty_csv: str, infer_csv: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    clean_df = load_table(clean_csv)
    dirty_df = load_table(dirty_csv)

    true_error_cells, true_error_info = build_true_error_cells(clean_df, dirty_df)

    infer_df, candidate_cells, predicted_error_cells, detector_predicted_error_cells, infer_row_map = load_infer_result(infer_csv)

    # ----------------------
    # 候选覆盖
    # ----------------------
    covered_true_errors = true_error_cells & candidate_cells
    candidate_missed_true_errors = true_error_cells - candidate_cells

    candidate_coverage = len(covered_true_errors) / len(true_error_cells) if len(true_error_cells) > 0 else 0.0
    candidate_only_precision = len(covered_true_errors) / len(candidate_cells) if len(candidate_cells) > 0 else 0.0

    # ----------------------
    # detector-only 指标
    # ----------------------
    detector_tp_keys = detector_predicted_error_cells & true_error_cells
    detector_fp_keys = detector_predicted_error_cells - true_error_cells
    detector_fn_keys = true_error_cells - detector_predicted_error_cells
    detector_prf = calc_prf(detector_predicted_error_cells, true_error_cells)

    # ----------------------
    # final 指标（verifier 后）
    # ----------------------
    tp_keys = predicted_error_cells & true_error_cells
    fp_keys = predicted_error_cells - true_error_cells
    fn_keys = true_error_cells - predicted_error_cells

    prf = calc_prf(predicted_error_cells, true_error_cells)

    # verifier 额外统计
    n_verifier_used = 0
    n_verifier_rejected = 0
    if "verifier_keep_error_prob" in infer_df.columns:
        n_verifier_used = int(infer_df["verifier_keep_error_prob"].notna().sum())
    if "verifier_decision" in infer_df.columns:
        n_verifier_rejected = int((infer_df["verifier_decision"].astype(str) == "reject_error").sum())

    # ----------------------
    # 导出明细
    # ----------------------
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

    # ----------------------
    # summary
    # ----------------------
    summary = {
        "n_rows": len(clean_df),
        "n_cols": len(clean_df.columns),
        "n_cells": len(clean_df) * len(clean_df.columns),

        "n_true_errors": len(true_error_cells),

        "n_candidate_cells": len(candidate_cells),
        "n_true_errors_covered_by_candidate": len(covered_true_errors),
        "n_true_errors_missed_by_candidate": len(candidate_missed_true_errors),
        "candidate_coverage": candidate_coverage,
        "candidate_only_precision": candidate_only_precision,

        # detector-only
        "n_detector_predicted_errors": len(detector_predicted_error_cells),
        "detector_tp": detector_prf["tp"],
        "detector_fp": detector_prf["fp"],
        "detector_fn": detector_prf["fn"],
        "detector_precision": detector_prf["precision"],
        "detector_recall": detector_prf["recall"],
        "detector_f1": detector_prf["f1"],

        # final
        "n_predicted_errors": len(predicted_error_cells),
        "tp": prf["tp"],
        "fp": prf["fp"],
        "fn": prf["fn"],
        "precision": prf["precision"],
        "recall": prf["recall"],
        "f1": prf["f1"],

        # verifier
        "n_verifier_used": n_verifier_used,
        "n_verifier_rejected": n_verifier_rejected,
    }

    # 可选补充统计：高风险和难例比例
    if "is_high_risk" in infer_df.columns:
        summary["n_high_risk"] = int(pd.to_numeric(infer_df["is_high_risk"], errors="coerce").fillna(0).sum())
    if "is_hard_case" in infer_df.columns:
        summary["n_hard_case"] = int(pd.to_numeric(infer_df["is_hard_case"], errors="coerce").fillna(0).sum())

    pd.DataFrame([summary]).to_csv(
        os.path.join(output_dir, "evaluation_summary.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # ----------------------
    # 打印
    # ----------------------
    print("========== 评估结果 ==========")
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

    return summary


if __name__ == "__main__":
    evaluate_detection(
        clean_csv=CLEAN_CSV,
        dirty_csv=DIRTY_CSV,
        infer_csv=INFER_RESULT_CSV,
        output_dir=OUTPUT_DIR
    )
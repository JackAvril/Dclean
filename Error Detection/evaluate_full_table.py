import os
import json
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)


# ============================================================
# 1. 用户配置区
# ============================================================

DIRTY_CSV = "/home/dq/Splittree/hospital/hospital_dirty_no_address23.csv"
CLEAN_CSV = "/home/dq/Splittree/hospital/hospital_clean_no_address23.csv"

# 这里放“全表推理结果”
PREDICTION_CSV = "full_table_candidate_scored_two_stage_interval_neighbor.csv"

OUTPUT_DIR = "evaluation_full_table"
MERGED_DETAIL_CSV = os.path.join(OUTPUT_DIR, "full_table_eval_detail.csv")
ERROR_CASES_CSV = os.path.join(OUTPUT_DIR, "full_table_error_cases.csv")
COLUMN_METRICS_CSV = os.path.join(OUTPUT_DIR, "full_table_column_metrics.csv")
SUMMARY_JSON = os.path.join(OUTPUT_DIR, "full_table_summary_metrics.json")

MISSING_TOKENS = {"", "empty", "null", "none", "nan", "na", "n/a", "missing", "unk"}

# 是否忽略某些列，例如 index
IGNORE_COLUMNS = []

# 若预测文件里出现重复 key，是否严格报错
STRICT_ONE_TO_ONE = True


# ============================================================
# 2. 基础函数
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def normalize_value(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.lower() in MISSING_TOKENS:
        return None
    return s


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = out[c].map(normalize_value)
    return out


def maybe_drop_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [c for c in df.columns if not str(c).lower().startswith("unnamed:")]
    return df[keep_cols].copy()


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": float(acc),
        "precision_error": float(precision),
        "recall_error": float(recall),
        "f1_error": float(f1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "confusion_matrix": cm.tolist(),
    }


# ============================================================
# 3. 构造整表 ground truth
# ============================================================

def build_ground_truth_from_dirty_clean(
    dirty_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    ignore_columns: List[str]
) -> pd.DataFrame:
    if list(dirty_df.columns) != list(clean_df.columns):
        raise ValueError("脏数据集和干净数据集的列名或列顺序不一致。")

    if len(dirty_df) != len(clean_df):
        raise ValueError("脏数据集和干净数据集的行数不一致。")

    records = []

    for row_id in range(len(dirty_df)):
        dirty_row = dirty_df.iloc[row_id]
        clean_row = clean_df.iloc[row_id]

        for col in dirty_df.columns:
            if col in ignore_columns:
                continue

            dirty_val = dirty_row[col]
            clean_val = clean_row[col]

            true_label = 1 if dirty_val != clean_val else 0

            records.append({
                "row_id": int(row_id),
                "column": str(col),
                "dirty_value": dirty_val,
                "clean_value": clean_val,
                "true_label": int(true_label),
                "is_true_error": bool(true_label),
            })

    return pd.DataFrame(records)


# ============================================================
# 4. 读取全表预测结果
# ============================================================

def load_prediction_df(path: str) -> pd.DataFrame:
    pred_df = pd.read_csv(path)

    required_cols = ["row_id", "column", "pred_label"]
    for c in required_cols:
        if c not in pred_df.columns:
            raise ValueError(f"预测文件缺少必要列: {c}")

    pred_df["row_id"] = pd.to_numeric(pred_df["row_id"], errors="coerce").astype(int)
    pred_df["column"] = pred_df["column"].astype(str)
    pred_df["pred_label"] = pd.to_numeric(pred_df["pred_label"], errors="coerce").fillna(0).astype(int)

    if "pred_error_prob" in pred_df.columns:
        pred_df["pred_error_prob"] = pd.to_numeric(pred_df["pred_error_prob"], errors="coerce")
    else:
        pred_df["pred_error_prob"] = np.nan

    if STRICT_ONE_TO_ONE:
        dup_count = pred_df.duplicated(subset=["row_id", "column"]).sum()
        if dup_count > 0:
            dup_rows = pred_df[pred_df.duplicated(subset=["row_id", "column"], keep=False)].copy()
            raise ValueError(
                f"预测文件中存在重复的 (row_id, column)，数量={dup_count}。\n"
                f"请先去重。示例重复行：\n{dup_rows.head(10).to_string(index=False)}"
            )

    return pred_df


# ============================================================
# 5. 主评估逻辑（全表）
# ============================================================

def evaluate_full_table():
    ensure_dir(OUTPUT_DIR)

    # ---------- 读取 dirty / clean ----------
    dirty_df = pd.read_csv(DIRTY_CSV, dtype=str)
    clean_df = pd.read_csv(CLEAN_CSV, dtype=str)

    dirty_df = maybe_drop_unnamed_columns(dirty_df)
    clean_df = maybe_drop_unnamed_columns(clean_df)

    dirty_df = normalize_df(dirty_df)
    clean_df = normalize_df(clean_df)

    # ---------- 构造整表 ground truth ----------
    gt_df = build_ground_truth_from_dirty_clean(
        dirty_df=dirty_df,
        clean_df=clean_df,
        ignore_columns=IGNORE_COLUMNS,
    )

    # ---------- 读取全表预测结果 ----------
    pred_df = load_prediction_df(PREDICTION_CSV)

    # ---------- 合并 ----------
    merged = gt_df.merge(
        pred_df,
        on=["row_id", "column"],
        how="left",
        validate="one_to_one"
    )

    # 没有预测到的默认认为 pred=0
    missing_pred = merged["pred_label"].isna().sum()
    if missing_pred > 0:
        print(f"[WARN] 有 {missing_pred} 个整表单元格没有匹配到预测结果，默认补为 0。")

    merged["pred_label"] = merged["pred_label"].fillna(0).astype(int)
    merged["pred_error_prob"] = merged["pred_error_prob"].fillna(0.0)

    y_true = merged["true_label"].values.astype(int)
    y_pred = merged["pred_label"].values.astype(int)

    # ---------- 总体指标 ----------
    overall_metrics = compute_binary_metrics(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["correct", "error"],
        zero_division=0,
        output_dict=True
    )

    # ---------- TP / FP / FN / TN ----------
    def judge_case(row):
        if row["true_label"] == 1 and row["pred_label"] == 1:
            return "TP"
        elif row["true_label"] == 0 and row["pred_label"] == 1:
            return "FP"
        elif row["true_label"] == 1 and row["pred_label"] == 0:
            return "FN"
        else:
            return "TN"

    merged["case_type"] = merged.apply(judge_case, axis=1)

    # ---------- 每列指标 ----------
    column_metrics = []
    for col, grp in merged.groupby("column"):
        yt = grp["true_label"].values.astype(int)
        yp = grp["pred_label"].values.astype(int)
        m = compute_binary_metrics(yt, yp)

        column_metrics.append({
            "column": col,
            "n_cells": int(len(grp)),
            "n_true_error": int(grp["true_label"].sum()),
            "n_pred_error": int(grp["pred_label"].sum()),
            "accuracy": m["accuracy"],
            "precision_error": m["precision_error"],
            "recall_error": m["recall_error"],
            "f1_error": m["f1_error"],
            "tp": m["tp"],
            "fp": m["fp"],
            "fn": m["fn"],
            "tn": m["tn"],
        })

    column_metrics_df = pd.DataFrame(column_metrics).sort_values(
        ["f1_error", "recall_error", "precision_error"],
        ascending=[False, False, False]
    )

    # ---------- 错判样本 ----------
    error_cases_df = merged[merged["case_type"].isin(["FP", "FN"])].copy()

    # ---------- 保存 ----------
    merged.to_csv(MERGED_DETAIL_CSV, index=False, encoding="utf-8-sig")
    error_cases_df.to_csv(ERROR_CASES_CSV, index=False, encoding="utf-8-sig")
    column_metrics_df.to_csv(COLUMN_METRICS_CSV, index=False, encoding="utf-8-sig")

    summary = {
        "dirty_csv": DIRTY_CSV,
        "clean_csv": CLEAN_CSV,
        "prediction_csv": PREDICTION_CSV,
        "evaluation_scope": "full_table",
        "n_rows_table": int(len(dirty_df)),
        "n_columns_table": int(len(dirty_df.columns)),
        "n_total_cells_evaluated": int(len(merged)),
        "n_true_error_cells": int(merged["true_label"].sum()),
        "n_pred_error_cells": int(merged["pred_label"].sum()),
        "overall_metrics": overall_metrics,
        "classification_report": report,
    }

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ---------- 打印 ----------
    print("===== 全表评估结果 =====")
    print(f"整表单元格总数: {len(merged)}")
    print(f"真实错误单元格数: {merged['true_label'].sum()}")
    print(f"预测错误单元格数: {merged['pred_label'].sum()}")
    print()
    print(f"Accuracy      : {overall_metrics['accuracy']:.6f}")
    print(f"Precision(err): {overall_metrics['precision_error']:.6f}")
    print(f"Recall(err)   : {overall_metrics['recall_error']:.6f}")
    print(f"F1(err)       : {overall_metrics['f1_error']:.6f}")
    print()
    print(f"TP={overall_metrics['tp']}  FP={overall_metrics['fp']}  FN={overall_metrics['fn']}  TN={overall_metrics['tn']}")
    print()
    print("===== error 类别的 P / R / F1 =====")
    print(f"P = {report['error']['precision']:.6f}")
    print(f"R = {report['error']['recall']:.6f}")
    print(f"F1 = {report['error']['f1-score']:.6f}")
    print()
    print("===== 文件已保存 =====")
    print(f"全表逐单元格评估: {MERGED_DETAIL_CSV}")
    print(f"全表错判样本    : {ERROR_CASES_CSV}")
    print(f"全表按列指标    : {COLUMN_METRICS_CSV}")
    print(f"全表总体 JSON   : {SUMMARY_JSON}")


if __name__ == "__main__":
    evaluate_full_table()
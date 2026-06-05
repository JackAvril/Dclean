
"""
Adult candidate/error-position coverage analysis.

用途
----
判断 predicted_error_positions.csv 覆盖了多少真实错误。

它会比较：
    dirty table vs clean table  -> 得到真实错误单元格集合
    predicted_error_positions.csv -> 得到候选/预测错误位置集合

输出：
    adult_coverage_analysis/
        adult_coverage_summary.txt
        adult_coverage_by_column.csv
        covered_true_errors.csv
        missed_true_errors.csv
        predicted_positions_with_truth.csv
        false_positive_positions.csv

运行示例
--------
cd /mnt/mydata/dq/projects/Splittree/test/Error\ Correction\ adult

python analyze_adult_error_position_coverage.py \
  --dirty_file /mnt/mydata/dq/projects/Splittree/adult/adult_dirty.csv \
  --clean_file /mnt/mydata/dq/projects/Splittree/adult/adult_clean.csv \
  --predicted_positions predicted_error_positions.csv \
  --output_dir adult_coverage_analysis

如果你的文件名不同，直接改命令里的路径即可。
"""

import argparse
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


DEFAULT_DIRTY_FILE = "/mnt/mydata/dq/projects/Splittree/adult/adult_dirty.csv"
DEFAULT_CLEAN_FILE = "/mnt/mydata/dq/projects/Splittree/adult/adult_clean.csv"
DEFAULT_PREDICTED_POSITIONS = "predicted_error_positions.csv"
DEFAULT_OUTPUT_DIR = "adult_coverage_analysis"


# ============================================================
# 1. Adult normalization
# ============================================================

EMPTY_TOKENS = {"", "empty", "null", "none", "nan", "na", "n/a", "missing", "?", "{null}"}

ADULT_COLUMNS_ALIASES = {
    "income": {"income", "class", "salary", "label"},
    "hours-per-week": {"hours-per-week", "hours_per_week", "hours", "hours-per"},
    "education-num": {"education-num", "education_num", "education_order"},
}


def norm_text(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()


def canonical_empty(x: Any) -> str:
    s = norm_text(x)
    if s.lower() in EMPTY_TOKENS:
        return ""
    return s


def normalize_income(x: Any) -> str:
    s = canonical_empty(x).strip()
    if not s:
        return ""
    s = s.replace(".", "")
    s = re.sub(r"\s+", "", s)
    low = s.lower()
    if low in {"<=50k", "lessthan50k", "less_than_50k", "0"}:
        return "LessThan50K"
    if low in {">50k", "morethan50k", "more_than_50k", "1"}:
        return "MoreThan50K"
    return s


def normalize_numeric(x: Any) -> str:
    s = canonical_empty(x)
    if not s:
        return ""
    s = s.replace(",", "")
    try:
        v = float(s)
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        return f"{v:.6f}".rstrip("0").rstrip(".")
    except Exception:
        return s.strip()


def normalize_category(x: Any) -> str:
    s = canonical_empty(x)
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s.strip())
    # Adult 常见值中 "-" 有意义，保留；只做大小写宽松。
    return s.lower()


def normalize_value_for_column(column: str, value: Any) -> str:
    col = str(column).strip()
    cl = col.lower()

    if cl in {"age", "fnlwgt", "education-num", "education_num", "capital-gain", "capital_gain",
              "capital-loss", "capital_loss", "hours-per-week", "hours_per_week"}:
        return normalize_numeric(value)

    if cl in {"income", "class", "salary", "label"}:
        return normalize_income(value)

    return normalize_category(value)


def values_equal(column: str, a: Any, b: Any) -> bool:
    return normalize_value_for_column(column, a) == normalize_value_for_column(column, b)


# ============================================================
# 2. Core functions
# ============================================================

def read_csv_any(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def normalize_row_id_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="raise").astype(int)


def build_true_error_cells(dirty: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    if dirty.shape != clean.shape:
        raise ValueError(f"dirty/clean 形状不一致: dirty={dirty.shape}, clean={clean.shape}")
    if list(dirty.columns) != list(clean.columns):
        raise ValueError(
            "dirty/clean 列名不一致。\n"
            f"dirty columns={list(dirty.columns)}\n"
            f"clean columns={list(clean.columns)}"
        )

    rows = []
    for rid in range(len(dirty)):
        for col in dirty.columns:
            dv = dirty.at[rid, col]
            cv = clean.at[rid, col]
            if not values_equal(col, dv, cv):
                rows.append({
                    "row_id": int(rid),
                    "column": str(col),
                    "dirty_value": dv,
                    "clean_value": cv,
                    "dirty_norm": normalize_value_for_column(str(col), dv),
                    "clean_norm": normalize_value_for_column(str(col), cv),
                })
    return pd.DataFrame(rows)


def build_predicted_positions(pred: pd.DataFrame) -> pd.DataFrame:
    required = {"row_id", "column"}
    missing = required - set(pred.columns)
    if missing:
        raise ValueError(f"predicted_positions 缺少必要字段: {missing}")

    out = pred.copy()
    out["row_id"] = normalize_row_id_series(out["row_id"])
    out["column"] = out["column"].astype(str)

    keep_cols = ["row_id", "column"]
    optional_cols = [
        "value",
        "final_pred_label",
        "final_pred_label_name",
        "final_pred_error_prob",
        "detector_pred_error_prob",
        "semantic_type",
        "main_rule_type",
        "main_usage_role",
        "adult_evidence_flags",
        "adult_v2_attribution_bucket",
        "adult_consistency_bucket",
        "is_hard_case",
        "is_high_risk",
    ]
    keep_cols += [c for c in optional_cols if c in out.columns]
    out = out[keep_cols].copy()
    out = out.drop_duplicates(subset=["row_id", "column"], keep="first").reset_index(drop=True)
    return out


def summarize_by_column(true_df: pd.DataFrame, pred_df: pd.DataFrame, merged_pred: pd.DataFrame) -> pd.DataFrame:
    true_count = true_df.groupby("column").size().rename("true_error_cells")
    pred_count = pred_df.groupby("column").size().rename("predicted_error_positions")

    covered = merged_pred[merged_pred["is_true_error"] == 1]
    covered_count = covered.groupby("column").size().rename("covered_true_errors")

    fp = merged_pred[merged_pred["is_true_error"] == 0]
    fp_count = fp.groupby("column").size().rename("false_positive_positions")

    cols = sorted(set(true_count.index) | set(pred_count.index))
    rows = []
    for col in cols:
        t = int(true_count.get(col, 0))
        p = int(pred_count.get(col, 0))
        c = int(covered_count.get(col, 0))
        f = int(fp_count.get(col, 0))
        missed = t - c
        precision = c / p if p else 0.0
        coverage = c / t if t else 0.0
        rows.append({
            "column": col,
            "true_error_cells": t,
            "predicted_error_positions": p,
            "covered_true_errors": c,
            "missed_true_errors": missed,
            "false_positive_positions": f,
            "position_precision": precision,
            "candidate_coverage_recall": coverage,
        })
    return pd.DataFrame(rows).sort_values(
        ["true_error_cells", "candidate_coverage_recall"],
        ascending=[False, True],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirty_file", default=DEFAULT_DIRTY_FILE)
    parser.add_argument("--clean_file", default=DEFAULT_CLEAN_FILE)
    parser.add_argument("--predicted_positions", default=DEFAULT_PREDICTED_POSITIONS)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dirty = read_csv_any(args.dirty_file)
    clean = read_csv_any(args.clean_file)
    pred_raw = read_csv_any(args.predicted_positions)

    print("========== Adult Error Position Coverage ==========")
    print(f"dirty_file: {args.dirty_file}")
    print(f"clean_file: {args.clean_file}")
    print(f"predicted_positions: {args.predicted_positions}")
    print(f"dirty shape: {dirty.shape}")
    print(f"clean shape: {clean.shape}")
    print(f"predicted rows raw: {len(pred_raw)}")

    true_df = build_true_error_cells(dirty, clean)
    pred_df = build_predicted_positions(pred_raw)

    true_keys = true_df[["row_id", "column"]].copy()
    true_keys["is_true_error"] = 1

    merged_pred = pred_df.merge(true_keys, on=["row_id", "column"], how="left")
    merged_pred["is_true_error"] = merged_pred["is_true_error"].fillna(0).astype(int)

    true_with_pred = true_df.merge(
        pred_df[["row_id", "column"]].assign(is_predicted_position=1),
        on=["row_id", "column"],
        how="left",
    )
    true_with_pred["is_predicted_position"] = true_with_pred["is_predicted_position"].fillna(0).astype(int)

    covered_true_errors = true_with_pred[true_with_pred["is_predicted_position"] == 1].copy()
    missed_true_errors = true_with_pred[true_with_pred["is_predicted_position"] == 0].copy()
    false_positive_positions = merged_pred[merged_pred["is_true_error"] == 0].copy()

    total_true = len(true_df)
    total_pred = len(pred_df)
    covered = len(covered_true_errors)
    missed = len(missed_true_errors)
    fp = len(false_positive_positions)

    coverage = covered / total_true if total_true else 0.0
    position_precision = covered / total_pred if total_pred else 0.0

    col_summary = summarize_by_column(true_df, pred_df, merged_pred)

    # Save outputs
    true_df.to_csv(os.path.join(args.output_dir, "true_error_cells.csv"), index=False, encoding="utf-8-sig")
    covered_true_errors.to_csv(os.path.join(args.output_dir, "covered_true_errors.csv"), index=False, encoding="utf-8-sig")
    missed_true_errors.to_csv(os.path.join(args.output_dir, "missed_true_errors.csv"), index=False, encoding="utf-8-sig")
    merged_pred.to_csv(os.path.join(args.output_dir, "predicted_positions_with_truth.csv"), index=False, encoding="utf-8-sig")
    false_positive_positions.to_csv(os.path.join(args.output_dir, "false_positive_positions.csv"), index=False, encoding="utf-8-sig")
    col_summary.to_csv(os.path.join(args.output_dir, "adult_coverage_by_column.csv"), index=False, encoding="utf-8-sig")

    summary_text = f"""========== Adult 覆盖率统计 ==========
total_cells: {dirty.shape[0] * dirty.shape[1]}
total_rows: {dirty.shape[0]}
total_columns: {dirty.shape[1]}

true_error_cells: {total_true}
predicted_error_positions: {total_pred}

covered_true_errors: {covered}
missed_true_errors: {missed}
false_positive_positions: {fp}

Candidate Coverage / Recall: {coverage:.6f}
Position Precision: {position_precision:.6f}
"""
    with open(os.path.join(args.output_dir, "adult_coverage_summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary_text)

    print("\n" + summary_text)

    print("---- 各列覆盖率 Top ----")
    show_cols = [
        "column",
        "true_error_cells",
        "predicted_error_positions",
        "covered_true_errors",
        "missed_true_errors",
        "false_positive_positions",
        "position_precision",
        "candidate_coverage_recall",
    ]
    print(col_summary[show_cols].to_string(index=False))

    print(f"\n[OK] 输出目录: {args.output_dir}")
    print(f"[OK] 漏掉的真实错误: {os.path.join(args.output_dir, 'missed_true_errors.csv')}")
    print(f"[OK] 覆盖到的真实错误: {os.path.join(args.output_dir, 'covered_true_errors.csv')}")
    print(f"[OK] 各列覆盖统计: {os.path.join(args.output_dir, 'adult_coverage_by_column.csv')}")


if __name__ == "__main__":
    main()

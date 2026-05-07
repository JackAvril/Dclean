
import json
from typing import Dict

import pandas as pd


DIRTY_CSV = "/mnt/mydata/dq/projects/Splittree/flights/flights_dirty.csv"
CLEAN_CSV = "/mnt/mydata/dq/projects/Splittree/flights/flights_clean.csv"
CANDIDATE_CSV = "/mnt/mydata/dq/projects/Splittree/test/Error Detection flights/candidate_cells_flights.csv"

OUTPUT_SUMMARY_JSON = "candidate_coverage_summary.json"
OUTPUT_TP_CSV = "candidate_true_errors.csv"
OUTPUT_FP_CSV = "candidate_clean_cells.csv"
OUTPUT_ALL_CSV = "candidate_with_labels.csv"

MISSING_TOKENS = {"", "empty", "null", "none", "nan", "na", "n/a", "missing", "unk"}


def normalize_value(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.lower() in MISSING_TOKENS:
        return None
    return s.lower()


def normalize_series(s: pd.Series) -> pd.Series:
    return s.map(normalize_value)


def safe_get(df: pd.DataFrame, row_id: int, col: str):
    if row_id < 0 or row_id >= len(df):
        return None
    if col not in df.columns:
        return None
    return df.iloc[row_id][col]


def compare_values(dirty_val, clean_val) -> bool:
    return normalize_value(dirty_val) != normalize_value(clean_val)


def evaluate_candidate_coverage(dirty_csv: str, clean_csv: str, candidate_csv: str):
    dirty_df = pd.read_csv(dirty_csv, dtype=str)
    clean_df = pd.read_csv(clean_csv, dtype=str)
    cand_df = pd.read_csv(candidate_csv, dtype=str)

    if len(dirty_df) != len(clean_df):
        raise ValueError(
            f"脏数据集与干净数据集行数不一致: dirty={len(dirty_df)}, clean={len(clean_df)}"
        )

    if list(dirty_df.columns) != list(clean_df.columns):
        raise ValueError("脏数据集与干净数据集列不一致，请先对齐列名和列顺序。")

    if "row_id" not in cand_df.columns or "column" not in cand_df.columns:
        raise ValueError("candidate csv 至少需要包含 row_id 和 column 两列。")

    cand_df["row_id"] = pd.to_numeric(cand_df["row_id"], errors="coerce")
    cand_df = cand_df[cand_df["row_id"].notna()].copy()
    cand_df["row_id"] = cand_df["row_id"].astype(int)

    result_rows = []
    for _, r in cand_df.iterrows():
        row_id = int(r["row_id"])
        col = str(r["column"])

        dirty_val = safe_get(dirty_df, row_id, col)
        clean_val = safe_get(clean_df, row_id, col)
        is_true_error = compare_values(dirty_val, clean_val)

        out = dict(r)
        out["dirty_value_actual"] = dirty_val
        out["clean_value_actual"] = clean_val
        out["is_true_error"] = int(is_true_error)
        result_rows.append(out)

    result_df = pd.DataFrame(result_rows)
    tp_df = result_df[result_df["is_true_error"] == 1].copy()
    fp_df = result_df[result_df["is_true_error"] == 0].copy()

    total_candidate_cells = len(result_df)
    covered_true_errors = len(tp_df)
    candidate_precision = covered_true_errors / total_candidate_cells if total_candidate_cells > 0 else 0.0

    total_true_errors = 0
    per_column_error_counts: Dict[str, int] = {}
    for col in dirty_df.columns:
        dirty_norm = normalize_series(dirty_df[col])
        clean_norm = normalize_series(clean_df[col])
        diff_mask = dirty_norm != clean_norm
        col_errors = int(diff_mask.sum())
        per_column_error_counts[col] = col_errors
        total_true_errors += col_errors

    candidate_coverage = covered_true_errors / total_true_errors if total_true_errors > 0 else 0.0

    summary = {
        "dirty_csv": dirty_csv,
        "clean_csv": clean_csv,
        "candidate_csv": candidate_csv,
        "row_count": int(len(dirty_df)),
        "column_count": int(len(dirty_df.columns)),
        "total_cells": int(len(dirty_df) * len(dirty_df.columns)),
        "candidate_cells": int(total_candidate_cells),
        "covered_true_errors": int(covered_true_errors),
        "candidate_clean_cells": int(len(fp_df)),
        "total_true_errors_in_table": int(total_true_errors),
        "candidate_coverage": round(candidate_coverage, 6),
        "candidate_precision": round(candidate_precision, 6),
        "per_column_true_error_count": per_column_error_counts,
    }

    return summary, result_df, tp_df, fp_df


def main():
    summary, all_df, tp_df, fp_df = evaluate_candidate_coverage(
        dirty_csv=DIRTY_CSV,
        clean_csv=CLEAN_CSV,
        candidate_csv=CANDIDATE_CSV,
    )

    with open(OUTPUT_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    all_df.to_csv(OUTPUT_ALL_CSV, index=False, encoding="utf-8-sig")
    tp_df.to_csv(OUTPUT_TP_CSV, index=False, encoding="utf-8-sig")
    fp_df.to_csv(OUTPUT_FP_CSV, index=False, encoding="utf-8-sig")

    print("========== 候选范围覆盖评估 ==========")
    print(f"总行数: {summary['row_count']}")
    print(f"总列数: {summary['column_count']}")
    print(f"总单元格数: {summary['total_cells']}")
    print()
    print(f"候选单元格数: {summary['candidate_cells']}")
    print(f"候选中真实错误数: {summary['covered_true_errors']}")
    print(f"候选中实际干净单元格数: {summary['candidate_clean_cells']}")
    print(f"全表真实错误总数: {summary['total_true_errors_in_table']}")
    print(f"Candidate Coverage: {summary['candidate_coverage']:.6f}")
    print(f"Candidate Precision: {summary['candidate_precision']:.6f}")
    print()
    print(f"[OK] Summary saved to: {OUTPUT_SUMMARY_JSON}")
    print(f"[OK] All labeled candidates saved to: {OUTPUT_ALL_CSV}")
    print(f"[OK] Candidate true errors saved to: {OUTPUT_TP_CSV}")
    print(f"[OK] Candidate clean cells saved to: {OUTPUT_FP_CSV}")


if __name__ == "__main__":
    main()

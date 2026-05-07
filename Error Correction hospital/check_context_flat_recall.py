import math
from collections import defaultdict
import pandas as pd
from pathlib import Path

# ============================================================
# 1. 配置区：直接改这里
# ============================================================
DIRTY_CSV = "/mnt/mydata/dq/projects/Splittree/hospital/hospital_dirty_no_address23.csv"
CLEAN_CSV = "/mnt/mydata/dq/projects/Splittree/hospital/hospital_clean_no_address23.csv"

# 检测阶段缩减范围后的 flat 上下文
CONTEXT_FLAT_CSV = "/mnt/mydata/dq/projects/Splittree/test/Error Detection new/candidate_llm_contexts_flat_v2.csv"

# 可选：如果你还想顺便看 predicted_error_positions 对比，也可填
PREDICTED_ERROR_FILE = ""   # 例如 "predicted_error_positions.csv"，不需要就留空

OUTPUT_DIR = "context_recall_check_output"


# ============================================================
# 2. 工具函数
# ============================================================
EMPTY_TOKENS = {"", "empty", "null", "none", "nan", "na", "n/a", "missing"}


def norm_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def canonical_text(x):
    s = norm_text(x).lower()
    return "empty" if s in EMPTY_TOKENS else norm_text(x)


def safe_mkdir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. 主逻辑
# ============================================================
def main():
    safe_mkdir(OUTPUT_DIR)

    dirty_df = pd.read_csv(DIRTY_CSV, encoding="utf-8-sig")
    clean_df = pd.read_csv(CLEAN_CSV, encoding="utf-8-sig")
    ctx_df = pd.read_csv(CONTEXT_FLAT_CSV, encoding="utf-8-sig")

    dirty_df = dirty_df.apply(lambda col: col.map(canonical_text))
    clean_df = clean_df.apply(lambda col: col.map(canonical_text))

    # context key
    if "row_id" not in ctx_df.columns or "column" not in ctx_df.columns:
        raise ValueError("CONTEXT_FLAT_CSV 必须至少包含 row_id, column 两列")

    ctx_df["row_id"] = ctx_df["row_id"].astype(int)
    ctx_df["column"] = ctx_df["column"].astype(str)

    context_keys = set((int(r["row_id"]), str(r["column"])) for _, r in ctx_df.iterrows())

    # optional predicted file
    pred_keys = set()
    if str(PREDICTED_ERROR_FILE).strip():
        pred_df = pd.read_csv(PREDICTED_ERROR_FILE, encoding="utf-8-sig")
        if "row_id" not in pred_df.columns or "column" not in pred_df.columns:
            raise ValueError("PREDICTED_ERROR_FILE 必须至少包含 row_id, column")
        pred_df["row_id"] = pred_df["row_id"].astype(int)
        pred_df["column"] = pred_df["column"].astype(str)

        if "final_pred_label" in pred_df.columns:
            pred_df = pred_df[pred_df["final_pred_label"] == 1].copy()
        elif "final_pred_label_name" in pred_df.columns:
            pred_df = pred_df[pred_df["final_pred_label_name"].astype(str).str.lower() == "error"].copy()

        pred_keys = set((int(r["row_id"]), str(r["column"])) for _, r in pred_df.iterrows())

    total_cells = int(dirty_df.shape[0] * dirty_df.shape[1])

    # true errors from dirty vs clean
    true_error_keys = []
    detail_rows = []
    per_col_stat = defaultdict(lambda: {"true_errors": 0, "covered_by_context": 0, "covered_by_pred": 0, "covered_by_both": 0})

    for row_id in range(len(dirty_df)):
        for col in dirty_df.columns:
            dirty_val = dirty_df.iloc[row_id][col]
            clean_val = clean_df.iloc[row_id][col]

            if dirty_val == clean_val:
                continue

            key = (row_id, str(col))
            in_context = int(key in context_keys)
            in_pred = int(key in pred_keys) if pred_keys else 0
            in_both = int(in_context and in_pred) if pred_keys else 0

            true_error_keys.append(key)

            per_col_stat[str(col)]["true_errors"] += 1
            per_col_stat[str(col)]["covered_by_context"] += in_context
            if pred_keys:
                per_col_stat[str(col)]["covered_by_pred"] += in_pred
                per_col_stat[str(col)]["covered_by_both"] += in_both

            detail_rows.append({
                "row_id": row_id,
                "column": str(col),
                "dirty_value": dirty_val,
                "clean_value": clean_val,
                "in_context_flat": in_context,
                "in_predicted_error_file": in_pred,
                "in_both": in_both,
            })

    true_error_keys = set(true_error_keys)

    context_only_not_true = context_keys - true_error_keys
    pred_only_not_true = pred_keys - true_error_keys if pred_keys else set()

    summary = {
        "total_cells": total_cells,
        "true_error_cells": len(true_error_keys),
        "context_rows": len(ctx_df),
        "unique_context_cells": len(context_keys),
        "true_errors_covered_by_context": sum(1 for k in true_error_keys if k in context_keys),
        "context_recall_on_true_errors": (
            sum(1 for k in true_error_keys if k in context_keys) / len(true_error_keys)
            if true_error_keys else 0.0
        ),
        "context_cells_not_true_errors": len(context_only_not_true),
    }

    if pred_keys:
        summary.update({
            "unique_predicted_error_cells": len(pred_keys),
            "true_errors_covered_by_predicted_file": sum(1 for k in true_error_keys if k in pred_keys),
            "predicted_file_recall_on_true_errors": (
                sum(1 for k in true_error_keys if k in pred_keys) / len(true_error_keys)
                if true_error_keys else 0.0
            ),
            "true_errors_covered_by_both_context_and_predicted": sum(1 for k in true_error_keys if (k in context_keys and k in pred_keys)),
            "both_recall_on_true_errors": (
                sum(1 for k in true_error_keys if (k in context_keys and k in pred_keys)) / len(true_error_keys)
                if true_error_keys else 0.0
            ),
            "predicted_cells_not_true_errors": len(pred_only_not_true),
        })

    summary_df = pd.DataFrame([summary])

    per_col_rows = []
    for col, stat in sorted(per_col_stat.items(), key=lambda x: (-x[1]["true_errors"], x[0])):
        row = {
            "column": col,
            "true_errors": stat["true_errors"],
            "covered_by_context": stat["covered_by_context"],
            "context_recall": stat["covered_by_context"] / stat["true_errors"] if stat["true_errors"] else 0.0,
        }
        if pred_keys:
            row.update({
                "covered_by_predicted": stat["covered_by_pred"],
                "predicted_recall": stat["covered_by_pred"] / stat["true_errors"] if stat["true_errors"] else 0.0,
                "covered_by_both": stat["covered_by_both"],
                "both_recall": stat["covered_by_both"] / stat["true_errors"] if stat["true_errors"] else 0.0,
            })
        per_col_rows.append(row)

    detail_df = pd.DataFrame(detail_rows)

    summary_path = Path(OUTPUT_DIR) / "context_recall_summary.csv"
    per_col_path = Path(OUTPUT_DIR) / "context_recall_by_column.csv"
    detail_path = Path(OUTPUT_DIR) / "context_recall_details.csv"

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(per_col_rows).to_csv(per_col_path, index=False, encoding="utf-8-sig")
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")

    print("========== Context Recall Check ==========")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"[OK] 输出目录: {OUTPUT_DIR}")
    print(f"[OK] summary: {summary_path}")
    print(f"[OK] by_column: {per_col_path}")
    print(f"[OK] details: {detail_path}")


if __name__ == "__main__":
    main()

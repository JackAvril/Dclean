import os
import pandas as pd


# ============================================================
# 1. 用户配置区
# ============================================================

CLEAN_FILE = "/mnt/mydata/dq/projects/Splittree/hospital/hospital_clean_no_address23.csv"
DIRTY_FILE = "/mnt/mydata/dq/projects/Splittree/hospital/hospital_dirty_no_address23.csv"
REPAIR_RESULT_FILE = "/mnt/mydata/dq/projects/Splittree/test/Error Correction hospital/repair_iterative_workdir_widetrain_whitelistinfer/model_output/inference_top1_repairs.csv"
OUTPUT_DIR = "repair_eval_output"

OUTPUT_SUMMARY_CSV = os.path.join(OUTPUT_DIR, "full_table_eval_summary.csv")
OUTPUT_CELL_DETAILS_CSV = os.path.join(OUTPUT_DIR, "full_table_eval_all_cells.csv")
OUTPUT_TP_CSV = os.path.join(OUTPUT_DIR, "full_table_tp_details.csv")
OUTPUT_FP_CSV = os.path.join(OUTPUT_DIR, "full_table_fp_details.csv")
OUTPUT_FN_CSV = os.path.join(OUTPUT_DIR, "full_table_fn_details.csv")
OUTPUT_TN_CSV = os.path.join(OUTPUT_DIR, "full_table_tn_details.csv")
OUTPUT_REPAIRED_TABLE_CSV = os.path.join(OUTPUT_DIR, "repaired_table.csv")


# ============================================================
# 2. 工具函数
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def norm_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def read_table(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def validate_same_shape(clean_df: pd.DataFrame, dirty_df: pd.DataFrame):
    if clean_df.shape != dirty_df.shape:
        raise ValueError(
            f"clean 与 dirty 形状不一致: clean={clean_df.shape}, dirty={dirty_df.shape}"
        )
    if list(clean_df.columns) != list(dirty_df.columns):
        raise ValueError("clean 与 dirty 列名不一致，请先对齐列顺序和列名。")


def stringify_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    将整张表统一转成字符串表示，避免 int/float/object 混合回填时报 dtype 错误。
    """
    out = df.copy()
    for col in out.columns:
        out[col] = out[col].map(norm_text).astype(object)
    return out


def load_repair_results(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")

    if "row_id" not in df.columns or "column" not in df.columns:
        raise ValueError("修复结果文件必须包含 row_id 和 column")

    df["row_id"] = df["row_id"].astype(int)
    df["column"] = df["column"].astype(str)

    repair_col = None
    for c in ["student_best_candidate", "candidate_value", "repair_value", "predicted_value"]:
        if c in df.columns:
            repair_col = c
            break

    if repair_col is None:
        raise ValueError(
            "修复结果文件中未找到修复值字段。需要包含 student_best_candidate / candidate_value / repair_value / predicted_value 之一。"
        )

    out = df[["row_id", "column", repair_col]].copy()
    out = out.rename(columns={repair_col: "predicted_repair"})
    out["predicted_repair"] = out["predicted_repair"].map(norm_text)

    out = out.drop_duplicates(subset=["row_id", "column"], keep="first").reset_index(drop=True)
    return out


def apply_repairs_to_dirty(dirty_df: pd.DataFrame, repair_df: pd.DataFrame) -> pd.DataFrame:
    """
    在字符串化后的 dirty 表上应用修复，避免 pandas dtype 冲突。
    """
    repaired_df = stringify_table(dirty_df)
    n_rows = len(repaired_df)
    valid_columns = set(repaired_df.columns)

    for _, r in repair_df.iterrows():
        row_id = int(r["row_id"])
        column = str(r["column"])
        pred_val = norm_text(r["predicted_repair"])

        if row_id < 0 or row_id >= n_rows:
            continue
        if column not in valid_columns:
            continue

        repaired_df.at[row_id, column] = pred_val

    return repaired_df


# ============================================================
# 3. 全表评估逻辑
# ============================================================

def build_full_table_eval(clean_df: pd.DataFrame, dirty_df: pd.DataFrame, repaired_df: pd.DataFrame, repair_df: pd.DataFrame) -> pd.DataFrame:
    repair_pos_set = set((int(r["row_id"]), str(r["column"])) for _, r in repair_df.iterrows())

    rows = []
    for row_id in range(len(clean_df)):
        for column in clean_df.columns:
            clean_val = norm_text(clean_df.at[row_id, column])
            dirty_val = norm_text(dirty_df.at[row_id, column])
            repaired_val = norm_text(repaired_df.at[row_id, column])

            is_true_error = int(dirty_val != clean_val)
            is_touched = int((row_id, str(column)) in repair_pos_set or repaired_val != dirty_val)
            is_final_correct = int(repaired_val == clean_val)

            if is_true_error == 1 and is_final_correct == 1:
                eval_type = "TP"
            elif is_true_error == 0 and repaired_val != clean_val:
                eval_type = "FP"
            elif is_true_error == 1 and is_final_correct == 0:
                eval_type = "FN"
            else:
                eval_type = "TN"

            rows.append({
                "row_id": row_id,
                "column": column,
                "dirty_value": dirty_val,
                "clean_value": clean_val,
                "repaired_value": repaired_val,
                "is_true_error": is_true_error,
                "is_touched": is_touched,
                "is_final_correct": is_final_correct,
                "eval_type": eval_type,
            })

    return pd.DataFrame(rows)


def compute_metrics(eval_df: pd.DataFrame) -> dict:
    tp = int((eval_df["eval_type"] == "TP").sum())
    fp = int((eval_df["eval_type"] == "FP").sum())
    fn = int((eval_df["eval_type"] == "FN").sum())
    tn = int((eval_df["eval_type"] == "TN").sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)

    total_cells = len(eval_df)
    total_true_errors = int(eval_df["is_true_error"].sum())
    touched_cells = int(eval_df["is_touched"].sum())
    remaining_wrong_cells = int((eval_df["repaired_value"] != eval_df["clean_value"]).sum())

    return {
        "total_cells": total_cells,
        "true_error_cells": total_true_errors,
        "touched_cells": touched_cells,
        "remaining_wrong_cells_after_repair": remaining_wrong_cells,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Accuracy": accuracy,
    }


# ============================================================
# 4. main
# ============================================================

def main():
    ensure_dir(OUTPUT_DIR)

    clean_df_raw = read_table(CLEAN_FILE)
    dirty_df_raw = read_table(DIRTY_FILE)
    validate_same_shape(clean_df_raw, dirty_df_raw)

    # 全部转成字符串视图再评估，保证 clean/dirty/repaired 比较口径一致
    clean_df = stringify_table(clean_df_raw)
    dirty_df = stringify_table(dirty_df_raw)

    repair_df = load_repair_results(REPAIR_RESULT_FILE)
    repaired_df = apply_repairs_to_dirty(dirty_df, repair_df)

    repaired_df.to_csv(OUTPUT_REPAIRED_TABLE_CSV, index=False, encoding="utf-8-sig")

    eval_df = build_full_table_eval(clean_df, dirty_df, repaired_df, repair_df)
    metrics = compute_metrics(eval_df)

    eval_df.to_csv(OUTPUT_CELL_DETAILS_CSV, index=False, encoding="utf-8-sig")
    eval_df[eval_df["eval_type"] == "TP"].to_csv(OUTPUT_TP_CSV, index=False, encoding="utf-8-sig")
    eval_df[eval_df["eval_type"] == "FP"].to_csv(OUTPUT_FP_CSV, index=False, encoding="utf-8-sig")
    eval_df[eval_df["eval_type"] == "FN"].to_csv(OUTPUT_FN_CSV, index=False, encoding="utf-8-sig")
    eval_df[eval_df["eval_type"] == "TN"].to_csv(OUTPUT_TN_CSV, index=False, encoding="utf-8-sig")

    summary_df = pd.DataFrame([metrics])
    summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")

    print("========== 全表端到端修复评估结果 ==========")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")

    print(f"\n[OK] 评估摘要已保存: {OUTPUT_SUMMARY_CSV}")
    print(f"[OK] 修复后的整表已保存: {OUTPUT_REPAIRED_TABLE_CSV}")
    print(f"[OK] 全量单元格评估明细已保存: {OUTPUT_CELL_DETAILS_CSV}")
    print(f"[OK] TP 明细已保存: {OUTPUT_TP_CSV}")
    print(f"[OK] FP 明细已保存: {OUTPUT_FP_CSV}")
    print(f"[OK] FN 明细已保存: {OUTPUT_FN_CSV}")
    print(f"[OK] TN 明细已保存: {OUTPUT_TN_CSV}")


if __name__ == "__main__":
    main()

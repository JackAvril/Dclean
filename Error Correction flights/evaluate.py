import os
import pandas as pd


# ============================================================
# 1. 用户配置区
# ============================================================

# Flights clean / dirty 表
# 按你的实际文件名改这里即可
CLEAN_FILE = "/mnt/mydata/dq/projects/Splittree/flights/flights_clean.csv"
DIRTY_FILE = "/mnt/mydata/dq/projects/Splittree/flights/flights_dirty.csv"

# 修复结果文件
# 适配 infer_ltr_flights_full_adapted.py / infer_ltr_flights_adapted.py 输出：
# inference_top1_repairs.csv
REPAIR_RESULT_FILE = "/mnt/mydata/dq/projects/Splittree/test/Error Correction flights/inference_top1_repairs.csv"

OUTPUT_DIR = (
    "/mnt/mydata/dq/projects/Splittree/test/Error Correction flights/"
    "repair_eval_output_flights"
)

OUTPUT_SUMMARY_CSV = os.path.join(OUTPUT_DIR, "full_table_eval_summary.csv")
OUTPUT_CELL_DETAILS_CSV = os.path.join(OUTPUT_DIR, "full_table_eval_all_cells.csv")
OUTPUT_TP_CSV = os.path.join(OUTPUT_DIR, "full_table_tp_details.csv")
OUTPUT_FP_CSV = os.path.join(OUTPUT_DIR, "full_table_fp_details.csv")
OUTPUT_FN_CSV = os.path.join(OUTPUT_DIR, "full_table_fn_details.csv")
OUTPUT_TN_CSV = os.path.join(OUTPUT_DIR, "full_table_tn_details.csv")
OUTPUT_REPAIRED_TABLE_CSV = os.path.join(OUTPUT_DIR, "repaired_table.csv")
OUTPUT_TOUCHED_CSV = os.path.join(OUTPUT_DIR, "touched_cells_details.csv")
OUTPUT_COLUMN_SUMMARY_CSV = os.path.join(OUTPUT_DIR, "column_level_eval_summary.csv")


# ============================================================
# 2. 工具函数
# ============================================================

EMPTY_TOKENS = {"", "empty", "null", "none", "nan", "na", "n/a", "missing"}


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def norm_text(x):
    """
    统一比较口径。
    注意：这里不会把 7:10 a.m. 改写成其它格式，只做基础字符串清洗。
    如果你的 clean/dirty 中时间格式本身不统一，需要在这里额外增加时间标准化。
    """
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in EMPTY_TOKENS:
        return ""
    return s


def read_table(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"表文件不存在: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def validate_same_shape(clean_df: pd.DataFrame, dirty_df: pd.DataFrame):
    if clean_df.shape != dirty_df.shape:
        raise ValueError(
            f"clean 与 dirty 形状不一致: clean={clean_df.shape}, dirty={dirty_df.shape}"
        )

    if list(clean_df.columns) != list(dirty_df.columns):
        raise ValueError(
            "clean 与 dirty 列名不一致，请先对齐列顺序和列名。\n"
            f"clean columns={list(clean_df.columns)}\n"
            f"dirty columns={list(dirty_df.columns)}"
        )


def stringify_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    将整张表统一转成字符串视图，避免 int/float/object 混合导致比较或回填异常。
    """
    out = df.copy()
    for col in out.columns:
        out[col] = out[col].map(norm_text).astype(object)
    return out


def find_repair_value_column(df: pd.DataFrame) -> str:
    """
    兼容不同版本修复结果输出字段。
    优先级说明：
    - predicted_repair_value：新版推理脚本经过 gate 后真正决定写回表的值
    - predicted_repair / repair_value / predicted_value：通用修复值字段
    - student_top_candidate / student_best_candidate / candidate_value：模型 top 候选，不一定经过 gate
    """
    candidates = [
        "predicted_repair_value",
        "predicted_repair",
        "repair_value",
        "predicted_value",
        "student_best_candidate",
        "student_top_candidate",
        "candidate_value",
        "teacher_best_candidate",
    ]

    for c in candidates:
        if c in df.columns:
            return c

    raise ValueError(
        "修复结果文件中未找到修复值字段。支持字段包括："
        "predicted_repair_value / predicted_repair / repair_value / predicted_value / "
        "student_best_candidate / student_top_candidate / candidate_value / teacher_best_candidate"
    )


def load_repair_results(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"修复结果文件不存在: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")

    if "row_id" not in df.columns or "column" not in df.columns:
        raise ValueError("修复结果文件必须包含 row_id 和 column")

    df["row_id"] = pd.to_numeric(df["row_id"], errors="raise").astype(int)
    df["column"] = df["column"].astype(str).str.strip()

    repair_col = find_repair_value_column(df)

    keep_cols = ["row_id", "column", repair_col]

    # 尽量保留推理阶段分析字段，方便评估后追踪
    optional_cols = [
        "dirty_value",
        "student_top_candidate",
        "student_score",
        "student_margin",
        "evidence_strength",
        "gate_passed",
        "gate_reason",
        "student_reason_pred",
        "candidate_count",
    ]
    for c in optional_cols:
        if c in df.columns and c not in keep_cols:
            keep_cols.append(c)

    out = df[keep_cols].copy()
    out = out.rename(columns={repair_col: "predicted_repair"})
    out["predicted_repair"] = out["predicted_repair"].map(norm_text)

    # 同一个 cell 若重复出现，保留第一条，通常 inference_top1 每个 cell 只有一条
    out = out.drop_duplicates(subset=["row_id", "column"], keep="first").reset_index(drop=True)
    return out


def apply_repairs_to_dirty(dirty_df: pd.DataFrame, repair_df: pd.DataFrame) -> pd.DataFrame:
    """
    在字符串化后的 dirty 表上应用修复。
    只回填 row_id / column 合法的记录。
    """
    repaired_df = stringify_table(dirty_df)
    n_rows = len(repaired_df)
    valid_columns = set(repaired_df.columns)

    skipped_bad_row = 0
    skipped_bad_col = 0

    for _, r in repair_df.iterrows():
        row_id = int(r["row_id"])
        column = str(r["column"]).strip()
        pred_val = norm_text(r["predicted_repair"])

        if row_id < 0 or row_id >= n_rows:
            skipped_bad_row += 1
            continue

        if column not in valid_columns:
            skipped_bad_col += 1
            continue

        repaired_df.at[row_id, column] = pred_val

    if skipped_bad_row > 0:
        print(f"[WARN] 跳过越界 row_id 修复记录数: {skipped_bad_row}")
    if skipped_bad_col > 0:
        print(f"[WARN] 跳过不存在 column 修复记录数: {skipped_bad_col}")

    return repaired_df


# ============================================================
# 3. 全表评估逻辑
# ============================================================

def build_full_table_eval(
    clean_df: pd.DataFrame,
    dirty_df: pd.DataFrame,
    repaired_df: pd.DataFrame,
    repair_df: pd.DataFrame,
) -> pd.DataFrame:
    repair_pos_set = set(
        (int(r["row_id"]), str(r["column"]).strip())
        for _, r in repair_df.iterrows()
    )

    repair_info_map = {}
    for _, r in repair_df.iterrows():
        key = (int(r["row_id"]), str(r["column"]).strip())
        repair_info_map[key] = r.to_dict()

    rows = []

    for row_id in range(len(clean_df)):
        for column in clean_df.columns:
            column = str(column)

            clean_val = norm_text(clean_df.at[row_id, column])
            dirty_val = norm_text(dirty_df.at[row_id, column])
            repaired_val = norm_text(repaired_df.at[row_id, column])

            is_true_error = int(dirty_val != clean_val)
            is_touched = int((row_id, column) in repair_pos_set or repaired_val != dirty_val)
            is_final_correct = int(repaired_val == clean_val)

            # 端到端修复评价：
            # TP：原来错，修完对
            # FP：原来不错，但修坏了
            # FN：原来错，修完仍错
            # TN：原来不错，修完仍对
            if is_true_error == 1 and is_final_correct == 1:
                eval_type = "TP"
            elif is_true_error == 0 and repaired_val != clean_val:
                eval_type = "FP"
            elif is_true_error == 1 and is_final_correct == 0:
                eval_type = "FN"
            else:
                eval_type = "TN"

            info = repair_info_map.get((row_id, column), {})

            row = {
                "row_id": row_id,
                "column": column,
                "dirty_value": dirty_val,
                "clean_value": clean_val,
                "repaired_value": repaired_val,
                "is_true_error": is_true_error,
                "is_touched": is_touched,
                "is_final_correct": is_final_correct,
                "eval_type": eval_type,
            }

            # 合并推理阶段信息
            for extra_col in [
                "predicted_repair",
                "student_top_candidate",
                "student_score",
                "student_margin",
                "evidence_strength",
                "gate_passed",
                "gate_reason",
                "student_reason_pred",
                "candidate_count",
            ]:
                if extra_col in info:
                    row[extra_col] = info.get(extra_col)

            rows.append(row)

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

    changed_cells = int((eval_df["repaired_value"] != eval_df["dirty_value"]).sum())
    unchanged_true_errors = int(
        ((eval_df["is_true_error"] == 1) & (eval_df["repaired_value"] == eval_df["dirty_value"])).sum()
    )

    return {
        "total_cells": total_cells,
        "true_error_cells": total_true_errors,
        "touched_cells": touched_cells,
        "changed_cells": changed_cells,
        "remaining_wrong_cells_after_repair": remaining_wrong_cells,
        "unchanged_true_errors": unchanged_true_errors,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Accuracy": accuracy,
    }


def build_column_summary(eval_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for column, g in eval_df.groupby("column", sort=False):
        tp = int((g["eval_type"] == "TP").sum())
        fp = int((g["eval_type"] == "FP").sum())
        fn = int((g["eval_type"] == "FN").sum())
        tn = int((g["eval_type"] == "TN").sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)

        rows.append({
            "column": column,
            "total_cells": len(g),
            "true_error_cells": int(g["is_true_error"].sum()),
            "touched_cells": int(g["is_touched"].sum()),
            "changed_cells": int((g["repaired_value"] != g["dirty_value"]).sum()),
            "remaining_wrong_cells_after_repair": int((g["repaired_value"] != g["clean_value"]).sum()),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Accuracy": accuracy,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["true_error_cells", "F1"], ascending=[False, False]).reset_index(drop=True)
    return out


# ============================================================
# 4. main
# ============================================================

def main():
    ensure_dir(OUTPUT_DIR)

    print("[INFO] Loading clean / dirty tables...")
    clean_df_raw = read_table(CLEAN_FILE)
    dirty_df_raw = read_table(DIRTY_FILE)
    validate_same_shape(clean_df_raw, dirty_df_raw)

    clean_df = stringify_table(clean_df_raw)
    dirty_df = stringify_table(dirty_df_raw)

    print("[INFO] Loading repair results...")
    repair_df = load_repair_results(REPAIR_RESULT_FILE)

    print("[INFO] Applying repairs...")
    repaired_df = apply_repairs_to_dirty(dirty_df, repair_df)
    repaired_df.to_csv(OUTPUT_REPAIRED_TABLE_CSV, index=False, encoding="utf-8-sig")

    print("[INFO] Building full-table evaluation...")
    eval_df = build_full_table_eval(clean_df, dirty_df, repaired_df, repair_df)
    metrics = compute_metrics(eval_df)
    column_summary_df = build_column_summary(eval_df)

    eval_df.to_csv(OUTPUT_CELL_DETAILS_CSV, index=False, encoding="utf-8-sig")
    eval_df[eval_df["eval_type"] == "TP"].to_csv(OUTPUT_TP_CSV, index=False, encoding="utf-8-sig")
    eval_df[eval_df["eval_type"] == "FP"].to_csv(OUTPUT_FP_CSV, index=False, encoding="utf-8-sig")
    eval_df[eval_df["eval_type"] == "FN"].to_csv(OUTPUT_FN_CSV, index=False, encoding="utf-8-sig")
    eval_df[eval_df["eval_type"] == "TN"].to_csv(OUTPUT_TN_CSV, index=False, encoding="utf-8-sig")
    eval_df[eval_df["is_touched"] == 1].to_csv(OUTPUT_TOUCHED_CSV, index=False, encoding="utf-8-sig")

    pd.DataFrame([metrics]).to_csv(OUTPUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    column_summary_df.to_csv(OUTPUT_COLUMN_SUMMARY_CSV, index=False, encoding="utf-8-sig")

    print("\n========== Flights 全表端到端修复评估结果 ==========")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")

    print("\n---- 各列评估摘要 Top ----")
    if not column_summary_df.empty:
        show_cols = [
            "column",
            "true_error_cells",
            "touched_cells",
            "TP",
            "FP",
            "FN",
            "Precision",
            "Recall",
            "F1",
        ]
        print(column_summary_df[show_cols].head(20).to_string(index=False))

    print(f"\n[OK] 评估摘要已保存: {OUTPUT_SUMMARY_CSV}")
    print(f"[OK] 列级评估摘要已保存: {OUTPUT_COLUMN_SUMMARY_CSV}")
    print(f"[OK] 修复后的整表已保存: {OUTPUT_REPAIRED_TABLE_CSV}")
    print(f"[OK] 全量单元格评估明细已保存: {OUTPUT_CELL_DETAILS_CSV}")
    print(f"[OK] touched 明细已保存: {OUTPUT_TOUCHED_CSV}")
    print(f"[OK] TP 明细已保存: {OUTPUT_TP_CSV}")
    print(f"[OK] FP 明细已保存: {OUTPUT_FP_CSV}")
    print(f"[OK] FN 明细已保存: {OUTPUT_FN_CSV}")
    print(f"[OK] TN 明细已保存: {OUTPUT_TN_CSV}")


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Diagnose repair output consistency without repaired_table.csv")
    p.add_argument("--dirty", default="/mnt/mydata/dq/projects/Splittree/hospital/hospital_dirty_no_address23.csv")
    p.add_argument("--clean", default="/mnt/mydata/dq/projects/Splittree/hospital/hospital_clean_no_address23.csv")
    p.add_argument("--repair_results", default="/mnt/mydata/dq/projects/Splittree/test/Error Correction hospital/repair_iterative_workdir_final_gate/model_output/inference_top1_repairs.csv")
    p.add_argument("--output_dir", default="repair_diagnosis_output")
    return p.parse_args()


ARGS = parse_args()
EMPTY_TOKENS = {"", "empty", "null", "none", "nan", "na", "n/a", "missing"}


def safe_mkdir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def norm_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def canonical_text(x):
    s = norm_text(x).lower()
    if s in EMPTY_TOKENS:
        return "empty"
    return norm_text(x)


def numeric_canonical_text(x):
    s = canonical_text(x)
    if s == "empty":
        return s
    try:
        v = float(s)
        if abs(v - int(v)) < 1e-12:
            return str(int(v))
        return str(v)
    except Exception:
        return s


def build_cell_set_from_repair_results(df: pd.DataFrame):
    return set((int(r["row_id"]), str(r["column"])) for _, r in df.iterrows())


def apply_repairs_to_dirty(dirty_df: pd.DataFrame, repair_results: pd.DataFrame):
    # 关键修复：
    # 把整表转成 object，避免 pandas 因原列 dtype=int/float
    # 而拒绝写入字符串修复值（例如 "999" / "empty"）
    repaired = dirty_df.copy().astype(object)

    applied_rows = []
    for _, r in repair_results.iterrows():
        row_id = int(r["row_id"])
        col = str(r["column"])
        pred = r["student_best_candidate"]

        if col not in repaired.columns:
            continue
        if row_id < 0 or row_id >= len(repaired):
            continue

        old_val = repaired.at[row_id, col]
        repaired.at[row_id, col] = pred
        applied_rows.append({
            "row_id": row_id,
            "column": col,
            "old_value": old_val,
            "new_value": pred,
        })

    return repaired, pd.DataFrame(applied_rows)


def main():
    safe_mkdir(ARGS.output_dir)

    dirty = pd.read_csv(ARGS.dirty, encoding="utf-8-sig")
    clean = pd.read_csv(ARGS.clean, encoding="utf-8-sig")
    repair_results = pd.read_csv(ARGS.repair_results, encoding="utf-8-sig")

    required_cols = {"row_id", "column", "student_best_candidate"}
    if not required_cols.issubset(set(repair_results.columns)):
        raise ValueError(f"repair_results must contain columns: {required_cols}")

    if dirty.shape != clean.shape:
        raise ValueError(f"dirty table shape {dirty.shape} != clean table shape {clean.shape}")
    if list(dirty.columns) != list(clean.columns):
        raise ValueError("dirty and clean columns mismatch")

    repair_results["row_id"] = repair_results["row_id"].astype(int)
    repair_results["column"] = repair_results["column"].astype(str)
    repair_results["student_best_candidate"] = repair_results["student_best_candidate"].map(canonical_text)

    repaired, applied_df = apply_repairs_to_dirty(dirty.copy(), repair_results)

    dirty_c = dirty.copy().apply(lambda col: col.map(canonical_text))
    clean_c = clean.copy().apply(lambda col: col.map(canonical_text))
    repaired_c = repaired.copy().apply(lambda col: col.map(canonical_text))

    dirty_numc = dirty.copy().apply(lambda col: col.map(numeric_canonical_text))
    repaired_numc = repaired.copy().apply(lambda col: col.map(numeric_canonical_text))

    repair_cells = build_cell_set_from_repair_results(repair_results)
    repair_cell_counts = (
        repair_results.groupby(["row_id", "column"]).size().reset_index(name="count")
    )
    duplicate_repairs = repair_cell_counts[repair_cell_counts["count"] > 1].copy()

    touched_rows = []
    true_error_touched = 0
    clean_cell_touched = 0
    outside_repair_count = 0
    numeric_repr_same_count = 0
    total_touched = 0
    total_true_errors = 0

    for row_id in range(len(dirty)):
        for col in dirty.columns:
            d = dirty_c.iloc[row_id][col]
            c = clean_c.iloc[row_id][col]
            r = repaired_c.iloc[row_id][col]

            d_num = dirty_numc.iloc[row_id][col]
            r_num = repaired_numc.iloc[row_id][col]

            is_true_error = int(d != c)
            is_touched = int(d != r)
            in_repair_file = int((row_id, str(col)) in repair_cells)

            if is_true_error:
                total_true_errors += 1

            if is_touched:
                total_touched += 1
                if is_true_error:
                    true_error_touched += 1
                else:
                    clean_cell_touched += 1

                if not in_repair_file:
                    outside_repair_count += 1

                same_numeric_repr = int(d_num == r_num and d != r)
                if same_numeric_repr:
                    numeric_repr_same_count += 1

                touched_rows.append({
                    "row_id": row_id,
                    "column": str(col),
                    "dirty_value": d,
                    "clean_value": c,
                    "repaired_value": r,
                    "dirty_numcanonical": d_num,
                    "repaired_numcanonical": r_num,
                    "is_true_error": is_true_error,
                    "in_repair_results_file": in_repair_file,
                    "numeric_repr_only_diff": same_numeric_repr,
                })

    touched_df = pd.DataFrame(touched_rows)

    by_col_rows = []
    for col in dirty.columns:
        col = str(col)
        sub = touched_df[touched_df["column"] == col] if not touched_df.empty else pd.DataFrame()
        by_col_rows.append({
            "column": col,
            "touched_cells": int(len(sub)),
            "touched_true_errors": int(sub["is_true_error"].sum()) if not sub.empty else 0,
            "touched_clean_cells": int((sub["is_true_error"] == 0).sum()) if not sub.empty else 0,
            "outside_repair_file": int((sub["in_repair_results_file"] == 0).sum()) if not sub.empty else 0,
            "numeric_repr_only_diff": int(sub["numeric_repr_only_diff"].sum()) if not sub.empty else 0,
        })
    by_col_df = pd.DataFrame(by_col_rows).sort_values(["touched_cells", "outside_repair_file"], ascending=[False, False])

    summary = {
        "dirty_shape_rows": dirty.shape[0],
        "dirty_shape_cols": dirty.shape[1],
        "repair_results_rows": len(repair_results),
        "repair_results_unique_cells": len(repair_cells),
        "repair_results_duplicate_cells": len(duplicate_repairs),
        "applied_repairs_rows": len(applied_df),
        "reconstructed_repaired_table_touched_cells": total_touched,
        "touched_true_errors": true_error_touched,
        "touched_clean_cells": clean_cell_touched,
        "touched_cells_outside_repair_results": outside_repair_count,
        "numeric_repr_only_diff_count": numeric_repr_same_count,
        "total_true_errors": total_true_errors,
    }
    summary_df = pd.DataFrame([summary])

    summary_path = Path(ARGS.output_dir) / "repair_diagnosis_summary.csv"
    dup_path = Path(ARGS.output_dir) / "repair_result_duplicate_cells.csv"
    touched_path = Path(ARGS.output_dir) / "touched_cells_diagnosis.csv"
    bycol_path = Path(ARGS.output_dir) / "touched_cells_by_column.csv"
    applied_path = Path(ARGS.output_dir) / "applied_repairs_materialized.csv"
    reconstructed_path = Path(ARGS.output_dir) / "reconstructed_repaired_table.csv"

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    duplicate_repairs.to_csv(dup_path, index=False, encoding="utf-8-sig")
    touched_df.to_csv(touched_path, index=False, encoding="utf-8-sig")
    by_col_df.to_csv(bycol_path, index=False, encoding="utf-8-sig")
    applied_df.to_csv(applied_path, index=False, encoding="utf-8-sig")
    repaired.to_csv(reconstructed_path, index=False, encoding="utf-8-sig")

    print("========== Repair Diagnosis (No repaired_table.csv needed) ==========")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"[OK] summary: {summary_path}")
    print(f"[OK] duplicates: {dup_path}")
    print(f"[OK] touched detail: {touched_path}")
    print(f"[OK] by column: {bycol_path}")
    print(f"[OK] applied repairs: {applied_path}")
    print(f"[OK] reconstructed repaired table: {reconstructed_path}")


if __name__ == "__main__":
    main()

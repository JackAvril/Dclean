import pandas as pd
from pathlib import Path


# ============================================================
# 1. 直接在这里改文件路径
# ============================================================

INPUT_FILE = "/mnt/mydata/dq/projects/Splittree/test/Error Detection new/active_cleaning_loop_runs_v4/iter_3/infer/inference_all_predictions.csv"
OUTPUT_FILE = "allowed_cells_from_detection.csv"


# ============================================================
# 2. 主逻辑
# ============================================================

def main():
    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

    required_base = {"row_id", "column"}
    if not required_base.issubset(set(df.columns)):
        raise ValueError(f"输入文件必须至少包含列: {required_base}")

    error_mask = None

    if "final_pred_label" in df.columns:
        error_mask = pd.to_numeric(df["final_pred_label"], errors="coerce").fillna(0).astype(int) == 1
        used_rule = "final_pred_label == 1"
    elif "final_pred_label_name" in df.columns:
        error_mask = df["final_pred_label_name"].astype(str).str.strip().str.lower() == "error"
        used_rule = "final_pred_label_name == error"
    elif "pred_label" in df.columns:
        error_mask = pd.to_numeric(df["pred_label"], errors="coerce").fillna(0).astype(int) == 1
        used_rule = "pred_label == 1"
    elif "pred_label_name" in df.columns:
        error_mask = df["pred_label_name"].astype(str).str.strip().str.lower() == "error"
        used_rule = "pred_label_name == error"
    else:
        raise ValueError(
            "未找到可用于筛选错误位置的标签列。支持的列包括："
            "final_pred_label / final_pred_label_name / pred_label / pred_label_name"
        )

    out = df.loc[error_mask, ["row_id", "column"]].copy()
    out["row_id"] = pd.to_numeric(out["row_id"], errors="raise").astype(int)
    out["column"] = out["column"].astype(str)

    before_dedup = len(out)
    out = out.drop_duplicates(subset=["row_id", "column"]).sort_values(
        ["row_id", "column"], ascending=[True, True]
    ).reset_index(drop=True)

    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("========== Build allowed_cells ==========")
    print(f"input_file: {INPUT_FILE}")
    print(f"output_file: {OUTPUT_FILE}")
    print(f"input_rows: {len(df)}")
    print(f"filter_rule: {used_rule}")
    print(f"error_rows_before_dedup: {before_dedup}")
    print(f"allowed_unique_cells: {len(out)}")
    print(f"[OK] saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()

import json
from typing import Dict, List, Any

import pandas as pd


# ============================================================
# 1. 用户配置区
# ============================================================

CLEAN_CSV = "/home/dq/Splittree/hospital/hospital_clean_no_address23.csv"
DIRTY_CSV = "/home/dq/Splittree/hospital/hospital_dirty_no_address23.csv"

OUTPUT_JSONL = "error_cells_new.jsonl"
OUTPUT_CSV = "error_cells_new.csv"
OUTPUT_SUMMARY_JSON = "error_summary_new.json"

# 这些值视为缺失，并做统一化
MISSING_TOKENS = {"", "empty", "null", "none", "nan", "na", "n/a", "missing", "unk"}

# 是否忽略字符串前后空格
STRIP_WHITESPACE = True

# 是否忽略大小写差异
IGNORE_CASE = False

# 是否删除 pandas 自动生成的 unnamed 列
DROP_UNNAMED_COLUMNS = True


# ============================================================
# 2. 基础工具函数
# ============================================================

def normalize_value(x):
    """
    对单元格做统一化，避免因为 empty / null / 空格 等造成伪差异。
    """
    if pd.isna(x):
        return None

    s = str(x)
    if STRIP_WHITESPACE:
        s = s.strip()

    if IGNORE_CASE:
        s = s.lower()

    if s.lower() in MISSING_TOKENS:
        return None

    return s


def maybe_drop_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not DROP_UNNAMED_COLUMNS:
        return df
    keep_cols = [c for c in df.columns if not str(c).lower().startswith("unnamed:")]
    return df[keep_cols].copy()


def load_and_normalize_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df = maybe_drop_unnamed_columns(df)

    for col in df.columns:
        df[col] = df[col].map(normalize_value)

    return df


# ============================================================
# 3. 比较逻辑
# ============================================================

def compare_clean_dirty(clean_df: pd.DataFrame, dirty_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    逐单元格比较 clean / dirty，输出所有错误位置。
    """
    # 先检查行列是否一致
    if list(clean_df.columns) != list(dirty_df.columns):
        raise ValueError(
            "clean 和 dirty 的列不一致。\n"
            f"clean columns = {list(clean_df.columns)}\n"
            f"dirty columns = {list(dirty_df.columns)}"
        )

    if len(clean_df) != len(dirty_df):
        raise ValueError(
            f"clean 和 dirty 的行数不一致：clean={len(clean_df)}, dirty={len(dirty_df)}"
        )

    errors = []
    columns = list(clean_df.columns)

    # 优先尝试读取 index 列作为业务行标识
    has_index_col = "index" in columns

    for row_idx in range(len(clean_df)):
        clean_row = clean_df.iloc[row_idx]
        dirty_row = dirty_df.iloc[row_idx]

        row_identifier = None
        if has_index_col:
            row_identifier = dirty_row["index"]

        for col in columns:
            clean_val = clean_row[col]
            dirty_val = dirty_row[col]

            if clean_val != dirty_val:
                errors.append({
                    "row_id": row_idx,              # 0-based 行号
                    "index": row_identifier,        # 数据集里的 index 字段值
                    "column": col,
                    "dirty_value": dirty_val,
                    "clean_value": clean_val
                })

    return errors


# ============================================================
# 4. 输出结果
# ============================================================

def save_results(errors: List[Dict[str, Any]]):
    # 保存 jsonl
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for item in errors:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 保存 csv
    error_df = pd.DataFrame(errors)
    error_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    # 汇总统计
    by_column = {}
    by_row = {}

    for e in errors:
        col = e["column"]
        row = str(e["row_id"])

        by_column[col] = by_column.get(col, 0) + 1
        by_row[row] = by_row.get(row, 0) + 1

    summary = {
        "total_errors": len(errors),
        "columns_with_errors": len(by_column),
        "rows_with_errors": len(by_row),
        "error_count_by_column": by_column,
        "error_count_by_row": by_row
    }

    with open(OUTPUT_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[OK] 错误明细 JSONL 已保存: {OUTPUT_JSONL}")
    print(f"[OK] 错误明细 CSV 已保存: {OUTPUT_CSV}")
    print(f"[OK] 错误汇总 JSON 已保存: {OUTPUT_SUMMARY_JSON}")

    print("\n========== 统计结果 ==========")
    print(f"总错误单元格数: {len(errors)}")
    print(f"出现错误的列数: {len(by_column)}")
    print(f"出现错误的行数: {len(by_row)}")

    if len(by_column) > 0:
        print("\n按列统计错误数：")
        sorted_cols = sorted(by_column.items(), key=lambda x: x[1], reverse=True)
        for col, cnt in sorted_cols:
            print(f"  {col}: {cnt}")

    print("\n前 20 处错误示例：")
    for item in errors[:20]:
        print(
            f"row_id={item['row_id']}, index={item['index']}, column={item['column']}, "
            f"dirty={item['dirty_value']}, clean={item['clean_value']}"
        )


# ============================================================
# 5. 主程序
# ============================================================

def main():
    clean_df = load_and_normalize_csv(CLEAN_CSV)
    dirty_df = load_and_normalize_csv(DIRTY_CSV)

    errors = compare_clean_dirty(clean_df, dirty_df)
    save_results(errors)


if __name__ == "__main__":
    main()
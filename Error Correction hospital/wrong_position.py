import pandas as pd

# =========================
# 1. 输入输出路径
# =========================
INPUT_FILE = "/mnt/mydata/dq/projects/Splittree/test/Error Detection new/active_cleaning_loop_runs_v4/iter_3/infer/inference_all_predictions.csv"   # 你的预测结果文件
OUTPUT_FILE = "predicted_error_positions.csv"

# =========================
# 2. 读取文件
# =========================
# 你的这个 txt 实际上是逗号分隔的表，所以按 csv 读
df = pd.read_csv(INPUT_FILE)

# =========================
# 3. 找出最终预测为 error 的记录
# =========================
# 优先使用 final_pred_label / final_pred_label_name
if "final_pred_label" in df.columns:
    error_mask = (df["final_pred_label"] == 1)
elif "final_pred_label_name" in df.columns:
    error_mask = (df["final_pred_label_name"].astype(str).str.lower() == "error")
else:
    raise ValueError("文件中没有找到 final_pred_label 或 final_pred_label_name 字段")

error_df = df.loc[error_mask].copy()

# =========================
# 4. 只保留错误位置相关字段
# =========================
keep_cols = []

for col in ["row_id", "column", "value", "final_pred_error_prob", "final_pred_label_name"]:
    if col in error_df.columns:
        keep_cols.append(col)

result_df = error_df[keep_cols].copy()

# =========================
# 5. 按位置排序（可选）
# =========================
sort_cols = [c for c in ["row_id", "column"] if c in result_df.columns]
if sort_cols:
    result_df = result_df.sort_values(sort_cols).reset_index(drop=True)

# =========================
# 6. 输出到文件
# =========================
result_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

# =========================
# 7. 打印结果
# =========================
print(f"总共预测为错误的单元格数: {len(result_df)}")
print("\n前20个错误位置如下：")
print(result_df.head(20).to_string(index=False))
print(f"\n已保存到: {OUTPUT_FILE}")
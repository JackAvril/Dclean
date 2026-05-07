import pandas as pd


# =========================
# 1. 文件路径配置
# =========================
INPUT_CSV = "/home/dq/Splittree/hospital/hospital_clean.csv"
OUTPUT_CSV = "/home/dq/Splittree/hospital/hospital_clean_no_address23.csv"


# =========================
# 2. 读取数据
# =========================
df = pd.read_csv(INPUT_CSV, dtype=str)


# =========================
# 3. 删除 Address2 和 Address3 两列
# =========================
cols_to_remove = ["Address2", "Address3"]

# 只删除实际存在的列，避免报错
existing_cols_to_remove = [c for c in cols_to_remove if c in df.columns]
df = df.drop(columns=existing_cols_to_remove)


# =========================
# 4. 保存结果
# =========================
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(f"[OK] 已删除列: {existing_cols_to_remove}")
print(f"[OK] 新文件已保存到: {OUTPUT_CSV}")
print(f"[OK] 剩余列数: {len(df.columns)}")
# Raha + Baran baseline runner

这个目录给你一套可以直接跑的 Raha + Baran baseline 脚本。

## 1. 安装

```bash
conda create -n raha_baran python=3.9 -y
conda activate raha_baran
bash install_raha_baran.sh
```

如果 `pip install raha` 失败，可以用官方仓库方式：

```bash
git clone https://github.com/BigDaMa/raha.git
pip install -e raha
```

## 2. 单个数据集运行

```bash
python run_raha_baran_baseline.py \
  --name hospital \
  --dirty /path/to/hospital_dirty.csv \
  --clean /path/to/hospital_clean.csv \
  --out_dir ./baseline_outputs/hospital_raha_baran \
  --labeling_budget 20 \
  --baran_input raha \
  --clean_raha_cache
```

`--baran_input raha` 表示端到端：先 Raha 检测，再 Baran 修复。

如果你想测 Baran 在真实错误位置已知时的修复上限，用：

```bash
python run_raha_baran_baseline.py \
  --name hospital \
  --dirty /path/to/hospital_dirty.csv \
  --clean /path/to/hospital_clean.csv \
  --out_dir ./baseline_outputs/hospital_baran_oracle \
  --labeling_budget 20 \
  --baran_input oracle \
  --clean_raha_cache
```

## 3. 批量运行

先修改 `datasets.example.json` 里的路径，然后：

```bash
python run_many_raha_baran.py \
  --config datasets.example.json \
  --base_out ./baseline_outputs \
  --labeling_budget 20 \
  --baran_input raha \
  --clean_raha_cache
```

## 4. 输出文件

每个数据集输出目录包含：

- `raha_detected_cells.csv`：Raha 检测到的错误单元格
- `raha_detection_metrics.json`：检测指标
- `baran_corrected_cells.csv`：Baran 输出的修复结果
- `baran_repaired.csv`：修复后的完整表
- `baran_repair_metrics_end_to_end.json`：全表端到端修复指标
- `summary.json`：检测 + 修复汇总

## 5. 论文实验建议

主表建议用：

```bash
--baran_input raha
```

这表示完整 Raha+Baran pipeline。

补充实验可以用：

```bash
--baran_input oracle
```

这表示 Baran 修复能力上限，不要和端到端方法直接混为一谈。


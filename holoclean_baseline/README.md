# HoloClean Baseline Runner

这个包用于跑 HoloClean baseline，输出 **HoloClean 修复后的全表端到端指标**。

HoloClean 官方实现说明：HoloClean 是基于 PyTorch 和 PostgreSQL 的统计推理式数据修复系统；官方 README 里说明它测试过 Python 2.7、3.6、3.7，并要求 PostgreSQL 9.4+。因此它比 Raha/Baran 更挑环境。

## 1. 推荐环境

```bash
conda create -n hc37 python=3.7 -y
conda activate hc37
```

安装 HoloClean：

```bash
git clone https://github.com/HoloClean/holoclean.git
cd holoclean
pip install -r requirements.txt
cd ..
```

如果 requirements 安装失败，可以先执行：

```bash
bash install_holoclean_env.sh
```

## 2. PostgreSQL

如果服务器支持 Docker：

```bash
bash scripts/start_postgres_docker.sh
```

如果没有 Docker，需要安装 PostgreSQL，并创建：

```sql
CREATE DATABASE holo;
CREATE USER holocleanuser;
ALTER USER holocleanuser WITH PASSWORD 'abcd1234';
GRANT ALL PRIVILEGES ON DATABASE holo TO holocleanuser;
\c holo
ALTER SCHEMA public OWNER TO holocleanuser;
```

## 3. 单数据集运行

### 方式 A：使用已有 HoloClean DC 文件

```bash
python run_holoclean_baseline.py \
  --name hospital \
  --dirty /path/to/hospital_dirty.csv \
  --clean /path/to/hospital_clean.csv \
  --out_dir ./baseline_outputs/hospital_holoclean \
  --holoclean_home /path/to/holoclean \
  --dc_path /path/to/holoclean_constraints.txt \
  --epochs 10 \
  --threads 1
```

### 方式 B：使用 FD 文件自动转 DC

FD 文件示例：

```text
ProviderNumber->HospitalName
ProviderNumber->City
MeasureCode->MeasureName
```

运行：

```bash
python run_holoclean_baseline.py \
  --name hospital \
  --dirty /path/to/hospital_dirty.csv \
  --clean /path/to/hospital_clean.csv \
  --out_dir ./baseline_outputs/hospital_holoclean \
  --holoclean_home /path/to/holoclean \
  --fd_path configs/hospital_fds.example.txt \
  --epochs 10 \
  --threads 1
```

### 方式 C：从 dirty 自动挖 FD，再转 DC

```bash
python run_holoclean_baseline.py \
  --name hospital \
  --dirty /path/to/hospital_dirty.csv \
  --clean /path/to/hospital_clean.csv \
  --out_dir ./baseline_outputs/hospital_holoclean_auto_fd \
  --holoclean_home /path/to/holoclean \
  --auto_mine_fds \
  --fd_confidence 0.98 \
  --fd_min_support 5 \
  --epochs 10 \
  --threads 1
```

注意：`auto_mine_fds` 只用 dirty，不用 clean，所以不会泄漏 ground truth。

## 4. 输出文件

输出目录里主要看：

```text
holoclean_repaired.csv
holoclean_repair_metrics_end_to_end.json
holoclean_repair_details.csv
holoclean_builtin_report.json
summary.json
holoclean_constraints.txt
holoclean_truth_cells.csv
```

其中最适合和你方法对比的是：

```text
holoclean_repair_metrics_end_to_end.json
```

指标定义：

- TP：原本是错误单元格，且修复后等于 clean；
- FP：发生了修改，但不是正确修复；
- FN：真实错误单元格没有被正确修复；
- Precision = TP / changed_cells；
- Recall = TP / true_error_cells；
- F1 = 2PR/(P+R)。

## 5. 多数据集批量运行

先改 `datasets.example.json`，然后：

```bash
python run_many_holoclean.py \
  --config datasets.example.json \
  --base_out ./baseline_outputs \
  --holoclean_home /path/to/holoclean \
  --epochs 10 \
  --threads 1
```

如果统一使用自动 FD：

```bash
python run_many_holoclean.py \
  --config datasets.example.json \
  --base_out ./baseline_outputs \
  --holoclean_home /path/to/holoclean \
  --auto_mine_fds \
  --fd_confidence 0.98 \
  --fd_min_support 5
```

## 6. 论文实验建议

HoloClean 本质上是 repair baseline，不建议把它放到“错误检测表”里和 Raha/ZeroED 比。它更适合放在：

```text
Error Correction / End-to-End Repair Baselines
```

也就是和你的修复结果比较：

```text
HoloClean vs Raha+Baran vs ZeroEC vs Ours
```

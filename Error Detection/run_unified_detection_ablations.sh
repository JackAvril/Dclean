#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Unified Error Detection Ablation Runner
# ============================================================
# 适配当前目录：
#   /mnt/mydata/dq/projects/Splittree/test/Error Detection/
#     build_rule_pool.py
#     shrink_search_space.py
#     summarize_full_contexts.py
#     cluster_and_sample_candidates.py
#     annotate_with_deepseek.py
#     label_propagation_for_candidates.py
#     build_final_training_set.py
#     build_loading_set.py
#     train_mlp.py
#     infer_mlp.py
#     control_iteration.py
#     evaluate.py
#
# 数据路径：
#   /mnt/mydata/dq/projects/Splittree/hospital/
#   /mnt/mydata/dq/projects/Splittree/flights/
#   /mnt/mydata/dq/projects/Splittree/beers/
#   /mnt/mydata/dq/projects/Splittree/rayyan/
#   /mnt/mydata/dq/projects/Splittree/adult/
#   /mnt/mydata/dq/projects/Splittree/soccer/
#
# 用法：
#   bash run_unified_detection_ablations.sh soccer all
#   bash run_unified_detection_ablations.sh soccer prep
#   bash run_unified_detection_ablations.sh soccer full
#   bash run_unified_detection_ablations.sh soccer ablations
#   bash run_unified_detection_ablations.sh soccer eval
#
#   bash run_unified_detection_ablations.sh all all
#   DATASETS="soccer adult" bash run_unified_detection_ablations.sh custom all
#
# 常用环境变量：
#   export DEEPSEEK_API_KEY="你的key"
#   export MAX_ITER=3
#   export MAX_HARD=60
#   export SKIP_ANNOTATE=1          # 已经标注过时可跳过 annotate_with_deepseek.py
#   export SKIP_PREP_IF_EXISTS=1    # 如果 final_train_dataset_*.csv 和 candidate_infer_ready_*.csv 已存在则跳过 prep
# ============================================================

ROOT="/mnt/mydata/dq/projects/Splittree"
TEST_ROOT="${ROOT}/test"
SCRIPT_DIR="${TEST_ROOT}/Error Detection"
RUN_ROOT="${SCRIPT_DIR}/ablation_work"

PYTHON_BIN="${PYTHON_BIN:-python}"

MAX_ITER="${MAX_ITER:-3}"
MAX_HARD="${MAX_HARD:-60}"
SKIP_ANNOTATE="${SKIP_ANNOTATE:-0}"
SKIP_PREP_IF_EXISTS="${SKIP_PREP_IF_EXISTS:-0}"

TARGET="${1:-soccer}"
ACTION="${2:-all}"

# ------------------------------------------------------------
# 数据集路径配置
# ------------------------------------------------------------

dataset_dirty_csv() {
  case "$1" in
    hospital) echo "${ROOT}/hospital/hospital_dirty_no_address23.csv" ;;
    flights)  echo "${ROOT}/flights/flights_dirty.csv" ;;
    beers)    echo "${ROOT}/beers/beers_dirty.csv" ;;
    rayyan)   echo "${ROOT}/rayyan/rayyan_dirty.csv" ;;
    adult)    echo "${ROOT}/adult/adult_dirty.csv" ;;
    soccer)   echo "${ROOT}/soccer/soccer_dirty.csv" ;;
    *) echo "" ;;
  esac
}

dataset_clean_csv() {
  case "$1" in
    hospital) echo "${ROOT}/hospital/hospital_clean_no_address23.csv" ;;
    flights)  echo "${ROOT}/flights/flights_clean.csv" ;;
    beers)    echo "${ROOT}/beers/beers_clean.csv" ;;
    rayyan)   echo "${ROOT}/rayyan/rayyan_clean.csv" ;;
    adult)    echo "${ROOT}/adult/adult_clean.csv" ;;
    soccer)   echo "${ROOT}/soccer/soccer_clean.csv" ;;
    *) echo "" ;;
  esac
}

# 统一脚本默认输出的训练集 / 推理集文件名
final_train_csv_name() {
  case "$1" in
    hospital) echo "final_train_dataset_simple_v2.csv" ;;
    *)        echo "final_train_dataset_${1}.csv" ;;
  esac
}

infer_ready_csv_name() {
  case "$1" in
    hospital) echo "candidate_infer_ready_v2.csv" ;;
    *)        echo "candidate_infer_ready_${1}.csv" ;;
  esac
}

dataset_list() {
  if [[ "${TARGET}" == "all" ]]; then
    echo "hospital flights beers rayyan adult soccer"
  elif [[ "${TARGET}" == "custom" ]]; then
    echo "${DATASETS:-soccer}"
  else
    echo "${TARGET}"
  fi
}

# ------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------

log() {
  echo
  echo "========== $* =========="
}

need_file() {
  local f="$1"
  if [[ ! -f "$f" ]]; then
    echo "[ERROR] 文件不存在: $f"
    exit 1
  fi
}

need_script() {
  need_file "${SCRIPT_DIR}/$1"
}

check_scripts() {
  need_script build_rule_pool.py
  need_script shrink_search_space.py
  need_script summarize_full_contexts.py
  need_script cluster_and_sample_candidates.py
  need_script annotate_with_deepseek.py
  need_script label_propagation_for_candidates.py
  need_script build_final_training_set.py
  need_script build_loading_set.py
  need_script train_mlp.py
  need_script infer_mlp.py
  need_script control_iteration.py
  need_script evaluate.py
}

check_dataset_paths() {
  local ds="$1"
  local dirty
  local clean
  dirty="$(dataset_dirty_csv "$ds")"
  clean="$(dataset_clean_csv "$ds")"

  if [[ -z "$dirty" || -z "$clean" ]]; then
    echo "[ERROR] 未知数据集: $ds"
    exit 1
  fi
  need_file "$dirty"
  need_file "$clean"
}

work_dir() {
  echo "${RUN_ROOT}/$1"
}

mkdir_work() {
  local ds="$1"
  mkdir -p "$(work_dir "$ds")"
}

run_cmd() {
  local name="$1"
  shift
  echo
  echo "[CMD][$name] $*"
  "$@"
}

run_step() {
  local ds="$1"
  local step="$2"
  local script="$3"
  shift 3
  local wd
  wd="$(work_dir "$ds")"

  run_cmd "${ds}:${step}" \
    "$PYTHON_BIN" "${SCRIPT_DIR}/${script}" \
      --dataset "$ds" \
      --cwd "$wd" \
      "$@"
}

model_dir_full() {
  local ds="$1"
  local wd
  wd="$(work_dir "$ds")"

  if [[ -d "${wd}/active_cleaning_loop_runs_${ds}_full/best_model" ]]; then
    echo "${wd}/active_cleaning_loop_runs_${ds}_full/best_model"
  elif [[ -d "${wd}/active_cleaning_loop_runs_${ds}_full/iter_0_train" ]]; then
    echo "${wd}/active_cleaning_loop_runs_${ds}_full/iter_0_train"
  elif [[ -d "${wd}/two_stage_model_output_${ds}_v4_with_verifier" ]]; then
    echo "${wd}/two_stage_model_output_${ds}_v4_with_verifier"
  elif [[ "$ds" == "hospital" && -d "${wd}/two_stage_model_output_v4_with_verifier" ]]; then
    echo "${wd}/two_stage_model_output_v4_with_verifier"
  else
    echo ""
  fi
}

copy_if_exists() {
  local src="$1"
  local dst="$2"
  if [[ -f "$src" ]]; then
    mkdir -p "$(dirname "$dst")"
    cp "$src" "$dst"
    return 0
  fi
  return 1
}

standardize_full_inference() {
  local ds="$1"
  local wd
  wd="$(work_dir "$ds")"

  mkdir -p "${wd}/two_stage_inference_output_${ds}_full"

  # 按可能路径查找 Full/Ours 的推理输出
  copy_if_exists "${wd}/active_cleaning_loop_runs_${ds}_full/best_infer/inference_all_predictions.csv" \
                 "${wd}/two_stage_inference_output_${ds}_full/inference_all_predictions.csv" && return 0

  copy_if_exists "${wd}/active_cleaning_loop_runs_${ds}_full/iter_${MAX_ITER}_infer/inference_all_predictions.csv" \
                 "${wd}/two_stage_inference_output_${ds}_full/inference_all_predictions.csv" && return 0

  copy_if_exists "${wd}/active_cleaning_loop_runs_${ds}_full/iter_3_infer/inference_all_predictions.csv" \
                 "${wd}/two_stage_inference_output_${ds}_full/inference_all_predictions.csv" && return 0

  copy_if_exists "${wd}/active_cleaning_loop_runs_${ds}_full/iter_2_infer/inference_all_predictions.csv" \
                 "${wd}/two_stage_inference_output_${ds}_full/inference_all_predictions.csv" && return 0

  copy_if_exists "${wd}/active_cleaning_loop_runs_${ds}_full/iter_1_infer/inference_all_predictions.csv" \
                 "${wd}/two_stage_inference_output_${ds}_full/inference_all_predictions.csv" && return 0

  copy_if_exists "${wd}/active_cleaning_loop_runs_${ds}_full/iter_0_infer/inference_all_predictions.csv" \
                 "${wd}/two_stage_inference_output_${ds}_full/inference_all_predictions.csv" && return 0

  echo "[WARN][$ds] 未找到 Full/Ours inference_all_predictions.csv，后续 eval 可能显示缺失。"
  return 1
}

# ------------------------------------------------------------
# 1. Full pipeline prep
# ------------------------------------------------------------

run_prep() {
  local ds="$1"
  local wd
  local train_csv
  local infer_csv

  wd="$(work_dir "$ds")"
  train_csv="${wd}/$(final_train_csv_name "$ds")"
  infer_csv="${wd}/$(infer_ready_csv_name "$ds")"

  log "[$ds] PREP: build evidence / label / train set / infer set"
  check_dataset_paths "$ds"
  mkdir_work "$ds"

  if [[ "$SKIP_PREP_IF_EXISTS" == "1" && -f "$train_csv" && -f "$infer_csv" ]]; then
    echo "[SKIP][$ds] prep outputs already exist:"
    echo "  $train_csv"
    echo "  $infer_csv"
    return 0
  fi

  run_step "$ds" "build_rule_pool" build_rule_pool.py
  run_step "$ds" "shrink_search_space" shrink_search_space.py
  run_step "$ds" "summarize_full_contexts" summarize_full_contexts.py
  run_step "$ds" "cluster_and_sample_candidates" cluster_and_sample_candidates.py

  if [[ "$SKIP_ANNOTATE" == "1" ]]; then
    echo "[SKIP][$ds] annotate_with_deepseek.py because SKIP_ANNOTATE=1"
  else
    run_step "$ds" "annotate_with_deepseek" annotate_with_deepseek.py
  fi

  run_step "$ds" "label_propagation" label_propagation_for_candidates.py
  run_step "$ds" "build_final_training_set" build_final_training_set.py
  run_step "$ds" "build_loading_set" build_loading_set.py
}

# ------------------------------------------------------------
# 2. Full / Ours
# ------------------------------------------------------------

run_full() {
  local ds="$1"
  local wd
  wd="$(work_dir "$ds")"

  log "[$ds] FULL / OURS"
  mkdir_work "$ds"

  run_step "$ds" "control_iteration_full" control_iteration.py \
    --train-script "${SCRIPT_DIR}/train_mlp.py" \
    --infer-script "${SCRIPT_DIR}/infer_mlp.py" \
    --work-dir "${wd}/active_cleaning_loop_runs_${ds}_full" \
    --max-iterations "$MAX_ITER" \
    --max-hard-cases-per-iter "$MAX_HARD"

  standardize_full_inference "$ds" || true
}

# ------------------------------------------------------------
# 3. Ablation: w/o Evidence Reliability Gate
# ------------------------------------------------------------

run_no_reliability_gate() {
  local ds="$1"
  local wd
  local md
  wd="$(work_dir "$ds")"
  md="$(model_dir_full "$ds")"

  log "[$ds] ABLATION: w/o Evidence Reliability Gate"

  if [[ -z "$md" ]]; then
    echo "[ERROR][$ds] 找不到 Full 模型目录，请先运行 full。"
    exit 1
  fi

  run_step "$ds" "infer_no_reliability_gate" infer_mlp.py \
    --model_dir "$md" \
    --output_dir "${wd}/two_stage_inference_output_${ds}_ablation_no_reliability_gate" \
    --disable-reliability-gate
}

# ------------------------------------------------------------
# 4. Ablation: w/o Verifier
# ------------------------------------------------------------

run_no_verifier() {
  local ds="$1"
  local wd
  local md
  wd="$(work_dir "$ds")"
  md="$(model_dir_full "$ds")"

  log "[$ds] ABLATION: w/o Verifier"

  if [[ -z "$md" ]]; then
    echo "[ERROR][$ds] 找不到 Full 模型目录，请先运行 full。"
    exit 1
  fi

  run_step "$ds" "infer_no_verifier" infer_mlp.py \
    --model_dir "$md" \
    --output_dir "${wd}/two_stage_inference_output_${ds}_ablation_no_verifier" \
    --disable-verifier
}

# ------------------------------------------------------------
# 5. Ablation: w/o Label Propagation
# ------------------------------------------------------------

run_no_label_prop() {
  local ds="$1"
  local wd
  wd="$(work_dir "$ds")"

  log "[$ds] ABLATION: w/o Label Propagation"

  run_step "$ds" "build_final_training_no_label_prop" build_final_training_set.py \
    --disable-propagated-training \
    --output_csv "${wd}/final_train_dataset_${ds}_ablation_no_label_prop.csv" \
    --output_jsonl "${wd}/final_train_dataset_${ds}_ablation_no_label_prop.jsonl" \
    --output_summary_json "${wd}/final_train_summary_${ds}_ablation_no_label_prop.json"

  run_step "$ds" "train_no_label_prop" train_mlp.py \
    --input_csv "${wd}/final_train_dataset_${ds}_ablation_no_label_prop.csv" \
    --output_dir "${wd}/two_stage_model_output_${ds}_ablation_no_label_prop"

  run_step "$ds" "infer_no_label_prop" infer_mlp.py \
    --model_dir "${wd}/two_stage_model_output_${ds}_ablation_no_label_prop" \
    --output_dir "${wd}/two_stage_inference_output_${ds}_ablation_no_label_prop"
}

# ------------------------------------------------------------
# 6. Ablation: w/o Hard-case Feedback
# ------------------------------------------------------------

run_no_hard_feedback() {
  local ds="$1"
  local wd
  local ab_work
  local md

  wd="$(work_dir "$ds")"
  ab_work="${wd}/active_cleaning_loop_runs_${ds}_ablation_no_hard_feedback"

  log "[$ds] ABLATION: w/o Hard-case Feedback"

  run_step "$ds" "control_no_hard_feedback" control_iteration.py \
    --train-script "${SCRIPT_DIR}/train_mlp.py" \
    --infer-script "${SCRIPT_DIR}/infer_mlp.py" \
    --work-dir "$ab_work" \
    --disable-hardcase-feedback

  if [[ -d "${ab_work}/best_model" ]]; then
    md="${ab_work}/best_model"
  elif [[ -d "${ab_work}/iter_0_train" ]]; then
    md="${ab_work}/iter_0_train"
  else
    echo "[ERROR][$ds] 找不到 w/o Hard-case Feedback 模型目录。"
    exit 1
  fi

  run_step "$ds" "infer_no_hard_feedback" infer_mlp.py \
    --model_dir "$md" \
    --output_dir "${wd}/two_stage_inference_output_${ds}_ablation_no_hard_feedback"
}

run_ablations() {
  local ds="$1"
  run_no_reliability_gate "$ds"
  run_no_verifier "$ds"
  run_no_label_prop "$ds"
  run_no_hard_feedback "$ds"
}

# ------------------------------------------------------------
# 7. Evaluation
# ------------------------------------------------------------

eval_variant() {
  local ds="$1"
  local variant_key="$2"
  local variant_dir="$3"
  local wd
  local infer_csv
  local out_dir

  wd="$(work_dir "$ds")"
  infer_csv="${wd}/${variant_dir}/inference_all_predictions.csv"
  out_dir="${wd}/eval_outputs_${ds}_${variant_key}"

  if [[ ! -f "$infer_csv" ]]; then
    echo "[MISS][$ds][$variant_key] $infer_csv"
    return 0
  fi

  run_step "$ds" "eval_${variant_key}" evaluate.py \
    --infer_csv "$infer_csv" \
    --output_dir "$out_dir"
}

run_eval() {
  local ds="$1"

  log "[$ds] EVALUATION"

  eval_variant "$ds" "full" "two_stage_inference_output_${ds}_full"
  eval_variant "$ds" "no_reliability_gate" "two_stage_inference_output_${ds}_ablation_no_reliability_gate"
  eval_variant "$ds" "no_label_prop" "two_stage_inference_output_${ds}_ablation_no_label_prop"
  eval_variant "$ds" "no_hard_feedback" "two_stage_inference_output_${ds}_ablation_no_hard_feedback"
  eval_variant "$ds" "no_verifier" "two_stage_inference_output_${ds}_ablation_no_verifier"
}

collect_summary() {
  local summary_csv="${RUN_ROOT}/ablation_summary_all.csv"
  local summary_md="${RUN_ROOT}/ablation_summary_all.md"

  log "COLLECT SUMMARY"

  "$PYTHON_BIN" - <<PY
import os
from pathlib import Path
import pandas as pd

run_root = Path("${RUN_ROOT}")
rows = []
variants = [
    ("full", "Full / Ours"),
    ("no_reliability_gate", "w/o Evidence Reliability Gate"),
    ("no_label_prop", "w/o Label Propagation"),
    ("no_hard_feedback", "w/o Hard-case Feedback"),
    ("no_verifier", "w/o Verifier"),
]

for ds_dir in sorted([p for p in run_root.iterdir() if p.is_dir()]):
    ds = ds_dir.name
    for key, name in variants:
        eval_dir = ds_dir / f"eval_outputs_{ds}_{key}"
        summary = eval_dir / "evaluation_summary.csv"
        if not summary.exists():
            rows.append({
                "dataset": ds, "variant_key": key, "variant": name,
                "status": "missing_eval",
            })
            continue
        try:
            df = pd.read_csv(summary)
            if len(df) == 0:
                rows.append({
                    "dataset": ds, "variant_key": key, "variant": name,
                    "status": "empty_summary",
                })
                continue
            r = df.iloc[0].to_dict()
            rows.append({
                "dataset": ds,
                "variant_key": key,
                "variant": name,
                "status": "success",
                "precision": r.get("precision", r.get("final_precision", "")),
                "recall": r.get("recall", r.get("final_recall", "")),
                "f1": r.get("f1", r.get("final_f1", "")),
                "candidate_coverage": r.get("candidate_coverage", r.get("Candidate Coverage", "")),
                "candidate_only_precision": r.get("candidate_only_precision", r.get("Candidate-Only Precision", "")),
                "tp": r.get("tp", ""),
                "fp": r.get("fp", ""),
                "fn": r.get("fn", ""),
            })
        except Exception as e:
            rows.append({
                "dataset": ds, "variant_key": key, "variant": name,
                "status": f"read_failed: {e}",
            })

out = pd.DataFrame(rows)
out.to_csv("${summary_csv}", index=False, encoding="utf-8-sig")

def fmt(x):
    try:
        if pd.isna(x) or x == "":
            return ""
        return f"{float(x):.4f}"
    except Exception:
        return str(x)

lines = []
lines.append("| Dataset | Variant | Precision | Recall | F1 | Cand.Cov | Cand.Prec | Status |")
lines.append("|---|---|---:|---:|---:|---:|---:|---|")
for _, r in out.iterrows():
    lines.append(
        f"| {r.get('dataset','')} | {r.get('variant','')} | {fmt(r.get('precision',''))} | "
        f"{fmt(r.get('recall',''))} | {fmt(r.get('f1',''))} | "
        f"{fmt(r.get('candidate_coverage',''))} | {fmt(r.get('candidate_only_precision',''))} | "
        f"{r.get('status','')} |"
    )

ok = out[out["status"].eq("success")].copy()
if len(ok) > 0:
    lines.append("")
    lines.append("## Average over successful datasets")
    lines.append("")
    lines.append("| Variant | Avg Precision | Avg Recall | Avg F1 | N |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, g in ok.groupby("variant", sort=False):
        def avg(col):
            v = pd.to_numeric(g[col], errors="coerce").dropna()
            return v.mean() if len(v) else ""
        lines.append(
            f"| {name} | {fmt(avg('precision'))} | {fmt(avg('recall'))} | {fmt(avg('f1'))} | {len(g)} |"
        )

Path("${summary_md}").write_text("\\n".join(lines) + "\\n", encoding="utf-8")

print("[OK] summary csv:", "${summary_csv}")
print("[OK] summary md :", "${summary_md}")
print("\\n".join(lines))
PY
}

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

main() {
  check_scripts
  mkdir -p "$RUN_ROOT"

  if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
    echo "[WARN] DEEPSEEK_API_KEY 没有设置。需要调用 DeepSeek 的步骤会失败。"
    echo "       如果你只做 no_reliability_gate/no_verifier/eval，可能不影响。"
  fi

  local ds_list
  ds_list="$(dataset_list)"

  echo "[INFO] SCRIPT_DIR=${SCRIPT_DIR}"
  echo "[INFO] RUN_ROOT=${RUN_ROOT}"
  echo "[INFO] TARGET=${TARGET}"
  echo "[INFO] ACTION=${ACTION}"
  echo "[INFO] DATASETS=${ds_list}"

  for ds in $ds_list; do
    check_dataset_paths "$ds"

    case "$ACTION" in
      prep)
        run_prep "$ds"
        ;;
      full)
        run_prep "$ds"
        run_full "$ds"
        ;;
      ablations)
        run_ablations "$ds"
        ;;
      eval)
        run_eval "$ds"
        ;;
      all)
        run_prep "$ds"
        run_full "$ds"
        run_ablations "$ds"
        run_eval "$ds"
        ;;
      no_reliability_gate)
        run_no_reliability_gate "$ds"
        ;;
      no_verifier)
        run_no_verifier "$ds"
        ;;
      no_label_prop)
        run_no_label_prop "$ds"
        ;;
      no_hard_feedback)
        run_no_hard_feedback "$ds"
        ;;
      *)
        echo "[ERROR] 未知 ACTION=${ACTION}"
        echo "可选: prep | full | ablations | eval | all | no_reliability_gate | no_verifier | no_label_prop | no_hard_feedback"
        exit 1
        ;;
    esac
  done

  if [[ "$ACTION" == "eval" || "$ACTION" == "all" ]]; then
    collect_summary
  fi

  echo
  echo "[DONE] ACTION=${ACTION}"
}

main "$@"

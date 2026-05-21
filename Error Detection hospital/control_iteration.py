import os
import json
import time
import math
import importlib.util
from typing import Dict, Any, List, Tuple

import pandas as pd
from openai import OpenAI

# ============================================================
# 1. 用户配置区
# ============================================================
BASE_TRAIN_CSV = "final_train_dataset_simple_v2.csv"
BASE_INFERENCE_CSV = "candidate_infer_ready_v2.csv"

TRAIN_SCRIPT_PATH = "train_mlp_v3.py"
INFER_SCRIPT_PATH = "infer_mlp_v3.py"

WORK_DIR = "active_cleaning_loop_runs"
os.makedirs(WORK_DIR, exist_ok=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-fd93f5216ae0488f8bf32d9e2a14a14a")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

LLM_TEMPERATURE = 0.0
LLM_MAX_RETRIES = 3
LLM_SLEEP_SECONDS = 1.0

MAX_ITERATIONS = 3
MAX_HARD_CASES_PER_ITER = 50
PRIORITIZE_HARD_CASES = True
RUN_INITIAL_TRAIN = True

CONVERGENCE_PATIENCE = 2
F1_IMPROVEMENT_EPS = 0.003
HARD_CASE_RATE_CHANGE_EPS = 0.05
LLM_USEFUL_RATE_EPS = 0.10

SYSTEM_PROMPT = """
You are an expert data cleaning annotator.

Task:
Given one candidate cell and structured evidence, determine whether the current value is erroneous.

Important rules:
1. Use only the provided evidence.
2. Be conservative. Do not mark a value as error only because it is rare.
3. If neighbor evidence strongly supports the current value, and no strong rule clearly contradicts it, prefer correct.
4. Return strict JSON only.

Output JSON:
{
  "label_binary": 0 or 1,
  "label": "correct" or "error",
  "confidence": float between 0 and 1,
  "reason_short": "...",
  "reason_detailed": "...",
  "should_change": true or false
}
"""


def build_user_prompt(rec: Dict[str, Any]) -> str:
    return json.dumps(rec, ensure_ascii=False, indent=2)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def save_jsonl(records: List[Dict[str, Any]], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def robust_float(x, default=0.0):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return default
        return float(x)
    except Exception:
        return default


def robust_int(x, default=0):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return default
        return int(x)
    except Exception:
        return default


def make_key(df: pd.DataFrame) -> pd.Series:
    return df["row_id"].astype(str) + "||" + df["column"].astype(str)


def load_module_from_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def create_client():
    if not DEEPSEEK_API_KEY:
        raise ValueError("未设置 DEEPSEEK_API_KEY")
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


def llm_relabel_one(client, rec: Dict[str, Any]) -> Dict[str, Any]:
    user_prompt = build_user_prompt(rec)
    for attempt in range(LLM_MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                temperature=LLM_TEMPERATURE,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            text = resp.choices[0].message.content.strip()
            parsed = json.loads(text)
            return {
                "row_id": rec.get("row_id"),
                "column": rec.get("column"),
                "value": rec.get("value"),
                "label_binary": robust_int(parsed.get("label_binary"), 0),
                "label": parsed.get("label", "correct"),
                "confidence": robust_float(parsed.get("confidence"), 0.5),
                "reason_short": parsed.get("reason_short", ""),
                "reason_detailed": parsed.get("reason_detailed", ""),
                "should_change": bool(parsed.get("should_change", False)),
                "label_source": "llm_relabel",
            }
        except Exception as e:
            if attempt == LLM_MAX_RETRIES - 1:
                return {
                    "row_id": rec.get("row_id"),
                    "column": rec.get("column"),
                    "value": rec.get("value"),
                    "label_binary": None,
                    "label": None,
                    "confidence": 0.0,
                    "reason_short": "llm_failed",
                    "reason_detailed": str(e),
                    "should_change": None,
                    "label_source": "llm_failed",
                }
            time.sleep(LLM_SLEEP_SECONDS)


def select_hard_cases_for_llm(records: List[Dict[str, Any]], max_cases: int) -> List[Dict[str, Any]]:
    if not PRIORITIZE_HARD_CASES:
        return records[:max_cases]
    def score_fn(r: Dict[str, Any]) -> int:
        flags = r.get("hard_case_flags", {})
        score = 0
        score += 3 * int(flags.get("is_stage_disagree", 0))
        score += 3 * int(flags.get("rule_neighbor_conflict", 0))
        score += 2 * int(flags.get("high_prob_but_neighbor_support_current", 0))
        score += 2 * int(flags.get("is_near_threshold", 0))
        score += 1 * int(flags.get("is_mid_prob", 0))
        score += 1 * int(flags.get("is_fp_risk_col", 0))
        return score
    return sorted(records, key=score_fn, reverse=True)[:max_cases]


def relabel_hard_cases(hard_case_jsonl: str, output_jsonl: str, max_cases: int) -> Tuple[List[Dict[str, Any]], int]:
    client = create_client()
    records = load_jsonl(hard_case_jsonl)
    records = select_hard_cases_for_llm(records, max_cases)
    relabeled, changed_count = [], 0
    for rec in records:
        new_rec = llm_relabel_one(client, rec)
        relabeled.append(new_rec)
        final_pred = robust_int(rec.get("final_pred_label", 0), 0)
        if new_rec["label_binary"] is not None and new_rec["label_binary"] != final_pred:
            changed_count += 1
    save_jsonl(relabeled, output_jsonl)
    return relabeled, changed_count


def merge_relabeled_into_trainset(base_train_csv: str, inference_csv: str, relabeled_jsonl: str, output_train_csv: str):
    base_train_df = pd.read_csv(base_train_csv)
    inference_df = pd.read_csv(inference_csv)
    relabeled_df = pd.DataFrame(load_jsonl(relabeled_jsonl))
    if len(relabeled_df) == 0:
        base_train_df.to_csv(output_train_csv, index=False, encoding="utf-8-sig")
        return {"n_relabeled": 0, "n_added": 0, "n_updated": 0}
    relabeled_df = relabeled_df[relabeled_df["label_binary"].notna()].copy()
    if len(relabeled_df) == 0:
        base_train_df.to_csv(output_train_csv, index=False, encoding="utf-8-sig")
        return {"n_relabeled": 0, "n_added": 0, "n_updated": 0}
    relabeled_df["label_binary"] = relabeled_df["label_binary"].astype(int)

    base_train_df["_key"] = make_key(base_train_df)
    inference_df["_key"] = make_key(inference_df)
    relabeled_df["_key"] = relabeled_df["row_id"].astype(str) + "||" + relabeled_df["column"].astype(str)

    relabeled_feature_df = inference_df[inference_df["_key"].isin(relabeled_df["_key"])].copy()
    relabeled_feature_df = relabeled_feature_df.merge(
        relabeled_df[["_key", "label_binary", "label", "confidence", "reason_short", "reason_detailed", "label_source"]],
        on="_key", how="left", suffixes=("", "_llm")
    )
    relabeled_feature_df["sample_weight"] = relabeled_feature_df["confidence"].fillna(0.8).clip(lower=0.5, upper=1.0)
    relabeled_feature_df["propagation_confidence"] = relabeled_feature_df["confidence"].fillna(0.8)
    relabeled_feature_df["label"] = relabeled_feature_df["label"].fillna(relabeled_feature_df["label_binary"].map({1: "error", 0: "correct"}))

    old_keys = set(base_train_df["_key"].tolist())
    new_keys = set(relabeled_feature_df["_key"].tolist())
    n_updated = len(old_keys & new_keys)
    n_added = len(new_keys - old_keys)

    base_train_df = base_train_df[~base_train_df["_key"].isin(new_keys)].copy()
    for col in base_train_df.columns:
        if col not in relabeled_feature_df.columns:
            relabeled_feature_df[col] = None
    for col in relabeled_feature_df.columns:
        if col not in base_train_df.columns:
            base_train_df[col] = None
    relabeled_feature_df = relabeled_feature_df[base_train_df.columns]

    merged_df = pd.concat([base_train_df, relabeled_feature_df], ignore_index=True)
    if "_key" in merged_df.columns:
        merged_df = merged_df.drop(columns=["_key"])
    merged_df.to_csv(output_train_csv, index=False, encoding="utf-8-sig")
    return {"n_relabeled": len(relabeled_df), "n_added": n_added, "n_updated": n_updated}


def run_training_once(train_script_path: str, input_csv: str, output_dir: str) -> Dict[str, Any]:
    train_mod = load_module_from_path("train_mod_dynamic", train_script_path)
    train_mod.INPUT_CSV = input_csv
    train_mod.OUTPUT_DIR = output_dir
    train_mod.STAGE1_LAST_CKPT = os.path.join(output_dir, "stage1_base_last.pt")
    train_mod.STAGE1_BEST_CKPT = os.path.join(output_dir, "stage1_base_best.pt")
    train_mod.STAGE1_PREPROCESSOR = os.path.join(output_dir, "stage1_base_preprocessor.joblib")
    train_mod.STAGE1_TRAIN_LOG = os.path.join(output_dir, "stage1_train_log.csv")
    train_mod.STAGE2_LAST_CKPT = os.path.join(output_dir, "stage2_cal_last.pt")
    train_mod.STAGE2_BEST_CKPT = os.path.join(output_dir, "stage2_cal_best.pt")
    train_mod.STAGE2_PREPROCESSOR = os.path.join(output_dir, "stage2_cal_preprocessor.joblib")
    train_mod.STAGE2_TRAIN_LOG = os.path.join(output_dir, "stage2_train_log.csv")
    train_mod.CONFIG_PATH = os.path.join(output_dir, "two_stage_config.json")
    train_mod.METRICS_JSON = os.path.join(output_dir, "metrics_two_stage.json")
    train_mod.FINAL_VAL_PRED_CSV = os.path.join(output_dir, "final_val_predictions.csv")
    train_mod.STAGE2_TRAIN_ANALYSIS_CSV = os.path.join(output_dir, "stage2_train_analysis.csv")
    ensure_dir(output_dir)
    train_mod.main()
    return load_json(os.path.join(output_dir, "metrics_two_stage.json"))


def run_inference_once(infer_script_path: str, input_csv: str, model_dir: str, output_dir: str) -> Dict[str, str]:
    infer_mod = load_module_from_path("infer_mod_dynamic", infer_script_path)
    infer_mod.INPUT_CSV = input_csv
    infer_mod.MODEL_DIR = model_dir
    infer_mod.OUTPUT_DIR = output_dir
    infer_mod.STAGE1_BEST_CKPT = os.path.join(model_dir, "stage1_base_best.pt")
    infer_mod.STAGE1_PREPROCESSOR = os.path.join(model_dir, "stage1_base_preprocessor.joblib")
    infer_mod.STAGE2_BEST_CKPT = os.path.join(model_dir, "stage2_cal_best.pt")
    infer_mod.STAGE2_PREPROCESSOR = os.path.join(model_dir, "stage2_cal_preprocessor.joblib")
    infer_mod.CONFIG_PATH = os.path.join(model_dir, "two_stage_config.json")
    infer_mod.METRICS_PATH = os.path.join(model_dir, "metrics_two_stage.json")
    infer_mod.OUTPUT_ALL_CSV = os.path.join(output_dir, "inference_all_predictions.csv")
    infer_mod.OUTPUT_HARD_CASE_CSV = os.path.join(output_dir, "inference_hard_cases.csv")
    infer_mod.OUTPUT_HIGH_RISK_CSV = os.path.join(output_dir, "inference_high_risk.csv")
    infer_mod.OUTPUT_LLM_JSONL = os.path.join(output_dir, "hard_cases_for_llm.jsonl")
    ensure_dir(output_dir)
    infer_mod.main()
    return {
        "all_csv": infer_mod.OUTPUT_ALL_CSV,
        "hard_case_csv": infer_mod.OUTPUT_HARD_CASE_CSV,
        "high_risk_csv": infer_mod.OUTPUT_HIGH_RISK_CSV,
        "hard_case_jsonl": infer_mod.OUTPUT_LLM_JSONL,
    }


def has_converged(history: List[Dict[str, Any]]) -> bool:
    if len(history) < CONVERGENCE_PATIENCE + 1:
        return False
    recent = history[-(CONVERGENCE_PATIENCE + 1):]
    small_f1_improve, small_hard_change, low_useful_rate = True, True, True
    for i in range(1, len(recent)):
        prev, curr = recent[i - 1], recent[i]
        if (curr["val_f1"] - prev["val_f1"]) >= F1_IMPROVEMENT_EPS:
            small_f1_improve = False
        if abs(curr["hard_case_rate"] - prev["hard_case_rate"]) >= HARD_CASE_RATE_CHANGE_EPS:
            small_hard_change = False
        if curr["llm_useful_rate"] >= LLM_USEFUL_RATE_EPS:
            low_useful_rate = False
    return sum([small_f1_improve, small_hard_change, low_useful_rate]) >= 2


def main():
    history = []
    current_train_csv = BASE_TRAIN_CSV
    current_model_dir = None
    current_infer_dir = None

    if RUN_INITIAL_TRAIN:
        init_train_dir = os.path.join(WORK_DIR, "iter_0_train")
        run_training_once(TRAIN_SCRIPT_PATH, current_train_csv, init_train_dir)
        current_model_dir = init_train_dir
    else:
        current_model_dir = os.path.join(WORK_DIR, "iter_0_train")

    init_infer_dir = os.path.join(WORK_DIR, "iter_0_infer")
    run_inference_once(INFER_SCRIPT_PATH, BASE_INFERENCE_CSV, current_model_dir, init_infer_dir)
    current_infer_dir = init_infer_dir

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n================ ITERATION {iteration} ================\n")
        iter_dir = os.path.join(WORK_DIR, f"iter_{iteration}")
        ensure_dir(iter_dir)

        hard_case_jsonl = os.path.join(current_infer_dir, "hard_cases_for_llm.jsonl")
        relabeled_jsonl = os.path.join(iter_dir, "hard_cases_relabeled.jsonl")
        merged_train_csv = os.path.join(iter_dir, "train_with_llm_feedback.csv")
        train_out_dir = os.path.join(iter_dir, "train")
        infer_out_dir = os.path.join(iter_dir, "infer")

        relabeled_records, changed_count = relabel_hard_cases(hard_case_jsonl, relabeled_jsonl, MAX_HARD_CASES_PER_ITER)
        merge_stat = merge_relabeled_into_trainset(current_train_csv, BASE_INFERENCE_CSV, relabeled_jsonl, merged_train_csv)
        train_metrics = run_training_once(TRAIN_SCRIPT_PATH, merged_train_csv, train_out_dir)
        infer_outputs = run_inference_once(INFER_SCRIPT_PATH, BASE_INFERENCE_CSV, train_out_dir, infer_out_dir)

        hard_case_df = pd.read_csv(infer_outputs["hard_case_csv"])
        all_pred_df = pd.read_csv(infer_outputs["all_csv"])
        hard_case_rate = 0.0 if len(all_pred_df) == 0 else len(hard_case_df) / len(all_pred_df)
        llm_useful_rate = 0.0 if len(relabeled_records) == 0 else changed_count / len(relabeled_records)

        record = {
            "iteration": iteration,
            "val_f1": train_metrics["final_metrics"]["f1"],
            "val_precision": train_metrics["final_metrics"]["precision"],
            "val_recall": train_metrics["final_metrics"]["recall"],
            "hard_case_count": int(len(hard_case_df)),
            "all_count": int(len(all_pred_df)),
            "hard_case_rate": float(hard_case_rate),
            "llm_relabeled_count": int(len(relabeled_records)),
            "llm_changed_count": int(changed_count),
            "llm_useful_rate": float(llm_useful_rate),
            "merge_stat": merge_stat,
            "train_dir": train_out_dir,
            "infer_dir": infer_out_dir,
        }
        history.append(record)
        save_json({"history": history}, os.path.join(WORK_DIR, "iteration_history.json"))
        print(json.dumps(record, ensure_ascii=False, indent=2))

        current_train_csv = merged_train_csv
        current_model_dir = train_out_dir
        current_infer_dir = infer_out_dir

        if has_converged(history):
            print(f"\n[STOP] 在第 {iteration} 轮达到收敛条件，停止迭代。")
            break

    print("\n[DONE] 全部迭代结束。")
    print(f"[INFO] 迭代历史保存在: {os.path.join(WORK_DIR, 'iteration_history.json')}")


if __name__ == "__main__":
    main()
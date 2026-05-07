import os
import json
import time
import shutil
import importlib.util
from typing import Dict, Any, List, Tuple

import pandas as pd
from openai import OpenAI

BASE_TRAIN_CSV = "final_train_dataset_flights.csv"
BASE_INFERENCE_CSV = "candidate_infer_ready_flights.csv"

TRAIN_SCRIPT_PATH = "train_mlp_flights.py"
INFER_SCRIPT_PATH = "infer_mlp_flights.py"

WORK_DIR = "active_cleaning_loop_runs_flights_v4"
os.makedirs(WORK_DIR, exist_ok=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-fd93f5216ae0488f8bf32d9e2a14a14a")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

LLM_TEMPERATURE = 0.0
LLM_MAX_RETRIES = 3
LLM_SLEEP_SECONDS = 1.0

MAX_ITERATIONS = 3
MAX_HARD_CASES_PER_ITER = 80
PRIORITIZE_HARD_CASES = True
RUN_INITIAL_TRAIN = True

BEST_MODEL_POLICY = "f1_with_recall_floor"
RECALL_FLOOR_FOR_BEST = 0.02

CONVERGENCE_PATIENCE = 5
F1_IMPROVEMENT_EPS = 0.003
HARD_CASE_RATE_CHANGE_EPS = 0.05
LLM_USEFUL_RATE_EPS = 0.10
RECALL_DROP_EPS = 0.01

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
    if path:
        os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: str):
    ensure_dir(os.path.dirname(path))
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
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def robust_float(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def robust_int(x, default=0):
    try:
        if x is None:
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
            err_msg = str(e)
            if "Insufficient Balance" in err_msg or "402" in err_msg:
                raise RuntimeError(err_msg)
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

    def score_fn(r: Dict[str, Any]) -> float:
        flags = r.get("hard_case_flags", {})
        score = 0.0

        score += 4.0 * int(flags.get("verifier_borderline", 0))
        score += 4.0 * int(flags.get("verifier_reject_borderline", 0))
        score += 3.0 * int(flags.get("fp_conflict_fd_hardcase", 0))
        score += 2.5 * int(flags.get("rule_neighbor_conflict", 0))
        score += 2.5 * int(flags.get("sample_borderline_hardcase", 0))
        score += 4.0 * int(flags.get("persistent_fp_high_risk", 0))
        score += 3.0 * int(flags.get("recall_recovery_hardcase", 0))
        score += 1.0 * int(flags.get("is_stage_disagree", 0))
        score += 0.5 * int(flags.get("is_near_threshold", 0))

        verifier_prob = r.get("verifier_keep_error_prob", None)
        group_th = r.get("group_verifier_threshold", 0.50)
        if verifier_prob is not None:
            try:
                verifier_prob = float(verifier_prob)
                group_th = float(group_th)
                score += max(0.0, 0.08 - abs(verifier_prob - group_th)) * 12.0
            except Exception:
                pass

        return score

    return sorted(records, key=score_fn, reverse=True)[:max_cases]


def relabel_hard_cases(hard_case_jsonl: str, output_jsonl: str, max_cases: int) -> Tuple[List[Dict[str, Any]], int]:
    client = create_client()
    records = load_jsonl(hard_case_jsonl)
    records = select_hard_cases_for_llm(records, max_cases)

    relabeled = []
    changed_count = 0

    for rec in records:
        new_rec = llm_relabel_one(client, rec)
        relabeled.append(new_rec)

        final_pred = robust_int(rec.get("final_pred_label", 0), 0)
        if new_rec["label_binary"] is not None and new_rec["label_binary"] != final_pred:
            changed_count += 1

    save_jsonl(relabeled, output_jsonl)
    return relabeled, changed_count


def merge_relabeled_into_trainset(
    base_train_csv: str,
    inference_csv: str,
    relabeled_jsonl: str,
    output_train_csv: str
):
    base_train_df = pd.read_csv(base_train_csv)
    inference_df = pd.read_csv(inference_csv)
    relabeled_df = pd.DataFrame(load_jsonl(relabeled_jsonl))

    if len(relabeled_df) == 0:
        ensure_dir(os.path.dirname(output_train_csv))
        base_train_df.to_csv(output_train_csv, index=False, encoding="utf-8-sig")
        return {"n_relabeled": 0, "n_added": 0, "n_updated": 0}

    relabeled_df = relabeled_df[relabeled_df["label_binary"].notna()].copy()
    if len(relabeled_df) == 0:
        ensure_dir(os.path.dirname(output_train_csv))
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
    relabeled_feature_df["label"] = relabeled_feature_df["label"].fillna(
        relabeled_feature_df["label_binary"].map({1: "error", 0: "correct"})
    )

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

    ensure_dir(os.path.dirname(output_train_csv))
    merged_df.to_csv(output_train_csv, index=False, encoding="utf-8-sig")
    return {"n_relabeled": len(relabeled_df), "n_added": n_added, "n_updated": n_updated}


def extract_final_metrics(train_metrics: Dict[str, Any]) -> Dict[str, Any]:
    if "final_metrics_with_verifier" in train_metrics:
        return train_metrics["final_metrics_with_verifier"]
    if "final_metrics" in train_metrics:
        return train_metrics["final_metrics"]
    return {"f1": 0.0, "precision": 0.0, "recall": 0.0}


def copy_best_dir(src_dir: str, dst_dir: str):
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)


def is_better_result(curr: Dict[str, Any], best: Dict[str, Any]) -> bool:
    curr_recall = robust_float(curr.get("val_recall"), 0.0)
    best_recall = robust_float(best.get("val_recall"), 0.0)
    curr_f1 = robust_float(curr.get("val_f1"), 0.0)
    best_f1 = robust_float(best.get("val_f1"), 0.0)

    if BEST_MODEL_POLICY == "recall_first":
        if curr_recall > best_recall + 1e-9:
            return True
        if abs(curr_recall - best_recall) <= 1e-9 and curr_f1 > best_f1:
            return True
        return False

    if BEST_MODEL_POLICY == "f1_with_recall_floor":
        recall_floor = best_recall - RECALL_FLOOR_FOR_BEST
        if curr_recall >= recall_floor and curr_f1 > best_f1 + 1e-9:
            return True
        if curr_f1 >= best_f1 and curr_recall > best_recall + 1e-9:
            return True
        return False

    return curr_f1 > best_f1


def run_training_once(
    train_script_path: str,
    input_csv: str,
    output_dir: str,
    mode: str,
    detector_source_dir: str = ""
) -> Dict[str, Any]:
    train_mod = load_module_from_path("train_mod_dynamic", train_script_path)

    train_mod.INPUT_CSV = input_csv
    train_mod.OUTPUT_DIR = output_dir
    train_mod.DETECTOR_TRAIN_MODE = mode
    train_mod.DETECTOR_SOURCE_DIR = detector_source_dir if detector_source_dir else output_dir

    train_mod.STAGE1_LAST_CKPT = os.path.join(output_dir, "stage1_base_last.pt")
    train_mod.STAGE1_BEST_CKPT = os.path.join(output_dir, "stage1_base_best.pt")
    train_mod.STAGE1_PREPROCESSOR = os.path.join(output_dir, "stage1_base_preprocessor.joblib")
    train_mod.STAGE1_TRAIN_LOG = os.path.join(output_dir, "stage1_train_log.csv")

    train_mod.STAGE2_LAST_CKPT = os.path.join(output_dir, "stage2_cal_last.pt")
    train_mod.STAGE2_BEST_CKPT = os.path.join(output_dir, "stage2_cal_best.pt")
    train_mod.STAGE2_PREPROCESSOR = os.path.join(output_dir, "stage2_cal_preprocessor.joblib")
    train_mod.STAGE2_TRAIN_LOG = os.path.join(output_dir, "stage2_train_log.csv")

    train_mod.VERIFIER_MODEL_PATH = os.path.join(output_dir, "fp_verifier_model.joblib")
    train_mod.VERIFIER_PREPROCESSOR_PATH = os.path.join(output_dir, "fp_verifier_preprocessor.joblib")
    train_mod.VERIFIER_CONFIG_PATH = os.path.join(output_dir, "fp_verifier_config.json")
    train_mod.VERIFIER_METRICS_JSON = os.path.join(output_dir, "verifier_metrics.json")
    train_mod.VERIFIER_TRAIN_CSV = os.path.join(output_dir, "verifier_train_data.csv")
    train_mod.VERIFIER_VAL_CSV = os.path.join(output_dir, "verifier_val_data.csv")

    train_mod.CONFIG_PATH = os.path.join(output_dir, "two_stage_config.json")
    train_mod.METRICS_JSON = os.path.join(output_dir, "metrics_two_stage.json")
    train_mod.FINAL_VAL_PRED_CSV = os.path.join(output_dir, "final_val_predictions.csv")
    train_mod.STAGE2_TRAIN_ANALYSIS_CSV = os.path.join(output_dir, "stage2_train_analysis.csv")

    ensure_dir(output_dir)
    train_mod.main()
    return load_json(os.path.join(output_dir, "metrics_two_stage.json"))


def run_inference_once(
    infer_script_path: str,
    input_csv: str,
    model_dir: str,
    output_dir: str
) -> Dict[str, str]:
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

    infer_mod.VERIFIER_MODEL_PATH = os.path.join(model_dir, "fp_verifier_model.joblib")
    infer_mod.VERIFIER_PREPROCESSOR_PATH = os.path.join(model_dir, "fp_verifier_preprocessor.joblib")
    infer_mod.VERIFIER_CONFIG_PATH = os.path.join(model_dir, "fp_verifier_config.json")
    infer_mod.VERIFIER_METRICS_PATH = os.path.join(model_dir, "verifier_metrics.json")

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

        prev_f1 = robust_float(prev.get("val_f1"), 0.0)
        curr_f1 = robust_float(curr.get("val_f1"), 0.0)
        if (curr_f1 - prev_f1) >= F1_IMPROVEMENT_EPS:
            small_f1_improve = False

        prev_hard = prev.get("hard_case_rate", None)
        curr_hard = curr.get("hard_case_rate", None)
        if prev_hard is not None and curr_hard is not None:
            if abs(float(curr_hard) - float(prev_hard)) >= HARD_CASE_RATE_CHANGE_EPS:
                small_hard_change = False

        curr_useful = curr.get("llm_useful_rate", None)
        if curr_useful is not None:
            if float(curr_useful) >= LLM_USEFUL_RATE_EPS:
                low_useful_rate = False

    return sum([small_f1_improve, small_hard_change, low_useful_rate]) >= 2


def main():
    history = []

    init_train_dir = os.path.join(WORK_DIR, "iter_0_train")
    init_infer_dir = os.path.join(WORK_DIR, "iter_0_infer")

    if RUN_INITIAL_TRAIN:
        init_metrics = run_training_once(
            TRAIN_SCRIPT_PATH,
            BASE_TRAIN_CSV,
            init_train_dir,
            mode="full",
            detector_source_dir=""
        )
    else:
        init_metrics = load_json(os.path.join(init_train_dir, "metrics_two_stage.json"))

    run_inference_once(INFER_SCRIPT_PATH, BASE_INFERENCE_CSV, init_train_dir, init_infer_dir)

    init_final = extract_final_metrics(init_metrics)
    best_record = {
        "iteration": 0,
        "val_f1": robust_float(init_final.get("f1"), 0.0),
        "val_precision": robust_float(init_final.get("precision"), 0.0),
        "val_recall": robust_float(init_final.get("recall"), 0.0),
        "hard_case_count": 0,
        "all_count": 0,
        "hard_case_rate": 0.0,
        "llm_relabeled_count": 0,
        "llm_changed_count": 0,
        "llm_useful_rate": 0.0,
        "merge_stat": {},
        "train_dir": init_train_dir,
        "infer_dir": init_infer_dir,
    }
    history.append(best_record)

    best_dir = os.path.join(WORK_DIR, "best_model")
    copy_best_dir(init_train_dir, best_dir)

    current_train_csv = BASE_TRAIN_CSV
    frozen_detector_dir = init_train_dir
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

        relabeled_records, changed_count = relabel_hard_cases(
            hard_case_jsonl, relabeled_jsonl, MAX_HARD_CASES_PER_ITER
        )

        merge_stat = merge_relabeled_into_trainset(
            current_train_csv, BASE_INFERENCE_CSV, relabeled_jsonl, merged_train_csv
        )

        train_metrics = run_training_once(
            TRAIN_SCRIPT_PATH,
            merged_train_csv,
            train_out_dir,
            mode="freeze_reuse",
            detector_source_dir=frozen_detector_dir,
        )
        infer_outputs = run_inference_once(INFER_SCRIPT_PATH, BASE_INFERENCE_CSV, train_out_dir, infer_out_dir)

        hard_case_df = pd.read_csv(infer_outputs["hard_case_csv"])
        all_pred_df = pd.read_csv(infer_outputs["all_csv"])

        hard_case_rate = 0.0 if len(all_pred_df) == 0 else len(hard_case_df) / len(all_pred_df)
        llm_useful_rate = 0.0 if len(relabeled_records) == 0 else changed_count / len(relabeled_records)

        final_metrics = extract_final_metrics(train_metrics)

        record = {
            "iteration": iteration,
            "val_f1": robust_float(final_metrics.get("f1"), 0.0),
            "val_precision": robust_float(final_metrics.get("precision"), 0.0),
            "val_recall": robust_float(final_metrics.get("recall"), 0.0),
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

        if is_better_result(record, best_record):
            best_record = record
            copy_best_dir(train_out_dir, best_dir)

        if robust_float(record["val_recall"]) < robust_float(best_record["val_recall"]) - RECALL_DROP_EPS:
            print(f"\n[STOP] Recall 明显下降，停止在第 {iteration} 轮。")
            break

        current_train_csv = merged_train_csv
        current_infer_dir = infer_out_dir

        if has_converged(history):
            print(f"\n[STOP] 在第 {iteration} 轮达到收敛条件，停止迭代。")
            break

    print("\n[DONE] 全部迭代结束。")
    print(f"[INFO] 最优模型目录: {best_dir}")
    print(f"[INFO] 迭代历史保存在: {os.path.join(WORK_DIR, 'iteration_history.json')}")


if __name__ == "__main__":
    main()
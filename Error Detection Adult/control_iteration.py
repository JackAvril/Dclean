import os
import json
import time
import shutil
import importlib.util
from typing import Dict, Any, List, Tuple

import pandas as pd
from openai import OpenAI


# ============================================================
# 1. 用户配置区
# ============================================================

BASE_TRAIN_CSV = "final_train_dataset_adult.csv"
BASE_INFERENCE_CSV = "candidate_infer_ready_adult.csv"
BASE_LLM_LABELED_CSV = "candidate_sampled_labeled_adult.csv"

TRAIN_SCRIPT_PATH = "train_mlp.py"
INFER_SCRIPT_PATH = "infer_mlp.py"

WORK_DIR = "active_cleaning_loop_runs_adult_v4"
os.makedirs(WORK_DIR, exist_ok=True)

# 不要把 key 写死到代码里。运行前执行：
# export DEEPSEEK_API_KEY="你的key"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-fd93f5216ae0488f8bf32d9e2a14a14a")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

LLM_TEMPERATURE = 0.0
LLM_MAX_RETRIES = 3
LLM_SLEEP_SECONDS = 1.0

# adult 默认最多 3 轮，每轮最多 60 个 hard cases，防止 token 消耗过大。
MAX_ITERATIONS = 3
MAX_HARD_CASES_PER_ITER = 60
PRIORITIZE_HARD_CASES = True
RUN_INITIAL_TRAIN = True

# 是否用第一轮 detector 继续训练。adult 初期建议 full，后续迭代 freeze_reuse 加 verifier 更稳定。
ITER_TRAIN_MODE = "freeze_reuse"

BEST_MODEL_POLICY = "f1_with_recall_floor"
RECALL_FLOOR_FOR_BEST = 0.03

CONVERGENCE_PATIENCE = 3
F1_IMPROVEMENT_EPS = 0.003
HARD_CASE_RATE_CHANGE_EPS = 0.05
LLM_USEFUL_RATE_EPS = 0.10
RECALL_DROP_EPS = 0.02

SYSTEM_PROMPT = """
You are an expert data-cleaning annotator for the Adult census-style tabular dataset.

Task:
Given one candidate cell and structured evidence, determine whether the current value is erroneous.

Important rules:
1. Use only the provided evidence.
2. Be conservative. Do not mark a value as error only because it is rare.
3. Country, occupation, and workclass are long-tail fields. Rarity alone is weak evidence.
4. Domain/format/adult-consistency conflicts are stronger evidence.
5. If neighbor evidence strongly supports the current value and no strong adult/domain/format rule contradicts it, prefer correct.
6. Return strict JSON only.

Adult-specific consistency hints:
- age is usually a bucket such as <18, 18-21, 22-25, ..., >70.
- hoursperweek should be a reasonable number or numeric range.
- income should be LessThan50K or MoreThan50K if the dataset uses canonical labels.
- Adult v2 primary targets are sex, relationship, and education.
- relationship=Wife usually implies sex=Female; relationship=Husband usually implies sex=Male.
- If relationship=Husband/Wife conflicts with valid sex/maritalstatus/age context, prefer marking relationship as erroneous.
- If the candidate column is sex, only mark it as error when the sex value itself is outside {Male, Female} or directly corrupted.
- Context-only columns such as age, income, race, country, workclass, occupation, maritalstatus, and hoursperweek should not be marked error unless there is direct domain/format evidence.
- relationship=Husband/Wife conflicts with Never-married, Divorced, Separated, or Widowed.
- very young age conflicts with very high education such as Bachelors, Masters, Doctorate.
- workclass=Never-worked conflicts with nonzero hoursperweek or normal occupation.

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


# ============================================================
# 2. 文件与模块工具
# ============================================================

def build_user_prompt(rec: Dict[str, Any]) -> str:
    # 去掉过长字段，避免 hard case 回流单条 token 太大
    rec2 = dict(rec)
    for k in list(rec2.keys()):
        if isinstance(rec2[k], str) and len(rec2[k]) > 3000:
            rec2[k] = rec2[k][:2200] + "\n...[TRUNCATED]...\n" + rec2[k][-600:]
    return json.dumps(rec2, ensure_ascii=False, indent=2)


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
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
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
        return int(float(x))
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
        raise ValueError("未设置 DEEPSEEK_API_KEY，请先 export DEEPSEEK_API_KEY='你的key'")
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


# ============================================================
# 3. LLM 难例回流
# ============================================================

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
                response_format={"type": "json_object"},
            )
            text = resp.choices[0].message.content.strip()
            parsed = json.loads(text)

            label_binary = robust_int(parsed.get("label_binary"), 0)
            label_binary = 1 if label_binary == 1 else 0

            return {
                "row_id": rec.get("row_id"),
                "column": rec.get("column"),
                "value": rec.get("value"),
                "label_binary": label_binary,
                "label": "error" if label_binary == 1 else "correct",
                "confidence": min(max(robust_float(parsed.get("confidence"), 0.5), 0.0), 1.0),
                "reason_short": parsed.get("reason_short", ""),
                "reason_detailed": parsed.get("reason_detailed", ""),
                "should_change": bool(parsed.get("should_change", label_binary == 1)),
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

        score += 4.0 * int(flags.get("adult_signal_hardcase", 0))
        score += 4.0 * int(flags.get("verifier_borderline", 0))
        score += 4.0 * int(flags.get("verifier_reject_borderline", 0))
        score += 3.0 * int(flags.get("fp_conflict_fd_hardcase", 0))
        score += 2.5 * int(flags.get("rule_neighbor_conflict", 0))
        score += 2.5 * int(flags.get("sample_borderline_hardcase", 0))
        score += 4.0 * int(flags.get("persistent_fp_high_risk", 0))
        score += 3.0 * int(flags.get("recall_recovery_hardcase", 0))
        score += 1.0 * int(flags.get("is_stage_disagree", 0))
        score += 0.5 * int(flags.get("is_near_threshold", 0))

        # adult 强信号优先
        score += 3.0 * robust_float(r.get("adult_domain_rule_count"), 0.0)
        score += 2.5 * robust_float(r.get("adult_format_rule_count"), 0.0)
        score += 2.0 * robust_float(r.get("adult_consistency_rule_count"), 0.0)
        score += 1.0 * robust_float(r.get("adult_consistency_score"), 0.0)

        # Adult v2 hard cases 优先，避免回流 token 被 context-only 弱样本消耗
        score += 6.0 * robust_int(r.get("has_relationship_v2_signal"), 0)
        score += 5.0 * robust_int(r.get("sex_value_invalid"), 0)
        score += 3.5 * robust_int(r.get("has_education_v2_signal"), 0)
        score -= 5.0 * robust_int(r.get("is_context_only_weak_signal"), 0)

        flags = r.get("hard_case_flags", {})
        score += 5.0 * int(flags.get("adult_v2_relationship_hardcase", 0))
        score += 4.0 * int(flags.get("adult_v2_sex_invalid_hardcase", 0))
        score -= 3.0 * int(flags.get("adult_v2_context_fp_hardcase", 0))

        # 长尾 rare-only 不优先，除非有 adult 信号
        col = str(r.get("column", ""))
        rule = str(r.get("main_rule_type", ""))
        if col in {"country", "occupation", "workclass"} and rule in {"rare_value", "rare_pattern", "global_dominant_value"}:
            if robust_float(r.get("adult_domain_rule_count"), 0) <= 0 and robust_float(r.get("adult_format_rule_count"), 0) <= 0:
                score -= 3.0

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

    for i, rec in enumerate(records, start=1):
        new_rec = llm_relabel_one(client, rec)
        relabeled.append(new_rec)

        final_pred = robust_int(rec.get("final_pred_label", 0), 0)
        if new_rec["label_binary"] is not None and new_rec["label_binary"] != final_pred:
            changed_count += 1

        print(
            f"[LLM] {i}/{len(records)} row={new_rec.get('row_id')} col={new_rec.get('column')} "
            f"label={new_rec.get('label')} conf={new_rec.get('confidence')}"
        )

    save_jsonl(relabeled, output_jsonl)
    return relabeled, changed_count


# ============================================================
# 4. 合并回训练集
# ============================================================

def merge_relabeled_into_trainset(
    base_train_csv: str,
    inference_csv: str,
    relabeled_jsonl: str,
    output_train_csv: str
):
    base_train_df = pd.read_csv(base_train_csv, low_memory=False)
    inference_df = pd.read_csv(inference_csv, low_memory=False)
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
    relabeled_feature_df["is_outside_candidate"] = 0
    relabeled_feature_df["error_type"] = "llm_relabel"
    relabeled_feature_df["evidence_used"] = "active_cleaning_loop"

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


# ============================================================
# 5. 训练 / 推理动态调用
# ============================================================

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
    output_dir: str,
    llm_labeled_csv: str = BASE_LLM_LABELED_CSV,
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

    # Evidence reliability gate: keep original output format; only put extra diagnostics under this iter dir.
    if hasattr(infer_mod, "LLM_LABELED_CSV"):
        infer_mod.LLM_LABELED_CSV = llm_labeled_csv
    if hasattr(infer_mod, "EVIDENCE_RELIABILITY_JSON"):
        infer_mod.EVIDENCE_RELIABILITY_JSON = os.path.join(output_dir, "evidence_reliability_adult.json")
    if hasattr(infer_mod, "EVIDENCE_RELIABILITY_CSV"):
        infer_mod.EVIDENCE_RELIABILITY_CSV = os.path.join(output_dir, "evidence_reliability_adult.csv")

    ensure_dir(output_dir)
    infer_mod.main()

    return {
        "output_all_csv": infer_mod.OUTPUT_ALL_CSV,
        "hard_case_csv": infer_mod.OUTPUT_HARD_CASE_CSV,
        "high_risk_csv": infer_mod.OUTPUT_HIGH_RISK_CSV,
        "hard_case_jsonl": infer_mod.OUTPUT_LLM_JSONL,
    }


# ============================================================
# 6. 迭代控制
# ============================================================

def summarize_inference(output_all_csv: str) -> Dict[str, Any]:
    df = pd.read_csv(output_all_csv, low_memory=False)
    total = len(df)
    hard = int(df["is_hard_case"].sum()) if "is_hard_case" in df.columns else 0
    high_risk = int(df["is_high_risk"].sum()) if "is_high_risk" in df.columns else 0
    final_error = int(df["final_pred_label"].sum()) if "final_pred_label" in df.columns else 0
    detector_error = int(df["detector_pred_label"].sum()) if "detector_pred_label" in df.columns else 0
    return {
        "total": total,
        "hard_case_count": hard,
        "hard_case_rate": hard / total if total else 0.0,
        "high_risk_count": high_risk,
        "final_error_count": final_error,
        "detector_error_count": detector_error,
    }


def should_stop(history: List[Dict[str, Any]]) -> bool:
    if len(history) < 2:
        return False
    if len(history) < CONVERGENCE_PATIENCE + 1:
        return False

    recent = history[-(CONVERGENCE_PATIENCE + 1):]
    f1_values = [robust_float(x.get("val_f1"), 0.0) for x in recent]
    hard_rates = [robust_float(x.get("hard_case_rate"), 0.0) for x in recent]
    useful_rates = [robust_float(x.get("llm_useful_rate"), 0.0) for x in recent]
    recall_values = [robust_float(x.get("val_recall"), 0.0) for x in recent]

    f1_improve = max(f1_values) - f1_values[0]
    hard_change = abs(hard_rates[-1] - hard_rates[0])
    useful_recent = useful_rates[-1]
    recall_drop = recall_values[0] - recall_values[-1]

    if f1_improve < F1_IMPROVEMENT_EPS and hard_change < HARD_CASE_RATE_CHANGE_EPS and useful_recent < LLM_USEFUL_RATE_EPS:
        return True
    if recall_drop > RECALL_DROP_EPS and f1_improve < F1_IMPROVEMENT_EPS:
        return True

    return False


def main():
    history = []
    best_summary = {
        "iter": -1,
        "val_precision": 0.0,
        "val_recall": 0.0,
        "val_f1": 0.0,
        "model_dir": "",
        "train_csv": "",
    }

    best_dir = os.path.join(WORK_DIR, "best_model")

    if RUN_INITIAL_TRAIN:
        print("========== Iter 0: initial training ==========")
        init_train_dir = os.path.join(WORK_DIR, "iter_0_train")
        train_metrics = run_training_once(
            TRAIN_SCRIPT_PATH,
            BASE_TRAIN_CSV,
            init_train_dir,
            mode="full",
        )
        final_metrics = extract_final_metrics(train_metrics)

        init_infer_dir = os.path.join(WORK_DIR, "iter_0_infer")
        infer_paths = run_inference_once(INFER_SCRIPT_PATH, BASE_INFERENCE_CSV, init_train_dir, init_infer_dir)
        infer_summary = summarize_inference(infer_paths["output_all_csv"])

        current = {
            "iter": 0,
            "train_csv": BASE_TRAIN_CSV,
            "model_dir": init_train_dir,
            "infer_dir": init_infer_dir,
            "val_precision": robust_float(final_metrics.get("precision"), 0.0),
            "val_recall": robust_float(final_metrics.get("recall"), 0.0),
            "val_f1": robust_float(final_metrics.get("f1"), 0.0),
            **infer_summary,
            "n_relabeled": 0,
            "n_added": 0,
            "n_updated": 0,
            "llm_changed_count": 0,
            "llm_useful_rate": 0.0,
        }
        history.append(current)
        best_summary = current
        copy_best_dir(init_train_dir, best_dir)
    else:
        raise RuntimeError("当前脚本要求 RUN_INITIAL_TRAIN=True，以保证初始模型存在。")

    prev_train_csv = BASE_TRAIN_CSV
    prev_model_dir = best_summary["model_dir"]

    for it in range(1, MAX_ITERATIONS + 1):
        print(f"\n========== Iter {it}: active relabel + retrain ==========")

        prev_infer_dir = history[-1]["infer_dir"]
        hard_jsonl = os.path.join(prev_infer_dir, "hard_cases_for_llm.jsonl")
        relabel_jsonl = os.path.join(WORK_DIR, f"iter_{it}_llm_relabels.jsonl")

        try:
            relabeled, changed_count = relabel_hard_cases(
                hard_jsonl,
                relabel_jsonl,
                max_cases=MAX_HARD_CASES_PER_ITER,
            )
        except RuntimeError as e:
            print(f"[STOP] LLM 调用失败: {e}")
            break

        n_relabeled = len([r for r in relabeled if r.get("label_binary") is not None])
        llm_useful_rate = changed_count / n_relabeled if n_relabeled > 0 else 0.0

        iter_train_csv = os.path.join(WORK_DIR, f"iter_{it}_train_dataset.csv")
        merge_stats = merge_relabeled_into_trainset(
            base_train_csv=prev_train_csv,
            inference_csv=BASE_INFERENCE_CSV,
            relabeled_jsonl=relabel_jsonl,
            output_train_csv=iter_train_csv,
        )

        iter_train_dir = os.path.join(WORK_DIR, f"iter_{it}_train")
        train_metrics = run_training_once(
            TRAIN_SCRIPT_PATH,
            iter_train_csv,
            iter_train_dir,
            mode=ITER_TRAIN_MODE,
            detector_source_dir=prev_model_dir,
        )
        final_metrics = extract_final_metrics(train_metrics)

        iter_infer_dir = os.path.join(WORK_DIR, f"iter_{it}_infer")
        infer_paths = run_inference_once(INFER_SCRIPT_PATH, BASE_INFERENCE_CSV, iter_train_dir, iter_infer_dir)
        infer_summary = summarize_inference(infer_paths["output_all_csv"])

        current = {
            "iter": it,
            "train_csv": iter_train_csv,
            "model_dir": iter_train_dir,
            "infer_dir": iter_infer_dir,
            "val_precision": robust_float(final_metrics.get("precision"), 0.0),
            "val_recall": robust_float(final_metrics.get("recall"), 0.0),
            "val_f1": robust_float(final_metrics.get("f1"), 0.0),
            **infer_summary,
            "n_relabeled": merge_stats.get("n_relabeled", 0),
            "n_added": merge_stats.get("n_added", 0),
            "n_updated": merge_stats.get("n_updated", 0),
            "llm_changed_count": changed_count,
            "llm_useful_rate": llm_useful_rate,
        }
        history.append(current)

        print(
            f"[ITER {it}] P={current['val_precision']:.4f} "
            f"R={current['val_recall']:.4f} F1={current['val_f1']:.4f} "
            f"hard_rate={current['hard_case_rate']:.4f} useful={llm_useful_rate:.4f}"
        )

        if is_better_result(current, best_summary):
            print(f"[BEST] 更新最佳模型: iter={it}")
            best_summary = current
            copy_best_dir(iter_train_dir, best_dir)

        prev_train_csv = iter_train_csv
        prev_model_dir = iter_train_dir

        save_json({"history": history, "best": best_summary}, os.path.join(WORK_DIR, "active_loop_summary.json"))
        pd.DataFrame(history).to_csv(os.path.join(WORK_DIR, "active_loop_history.csv"), index=False, encoding="utf-8-sig")

        if should_stop(history):
            print("[STOP] 达到收敛条件，停止迭代。")
            break

    save_json({"history": history, "best": best_summary}, os.path.join(WORK_DIR, "active_loop_summary.json"))
    pd.DataFrame(history).to_csv(os.path.join(WORK_DIR, "active_loop_history.csv"), index=False, encoding="utf-8-sig")

    print("\n===== ACTIVE LOOP DONE =====")
    print(f"Best iter: {best_summary['iter']}")
    print(f"Best model dir: {best_dir}")
    print(
        f"Best val metrics: P={best_summary['val_precision']:.4f}, "
        f"R={best_summary['val_recall']:.4f}, F1={best_summary['val_f1']:.4f}"
    )
    print(f"History CSV: {os.path.join(WORK_DIR, 'active_loop_history.csv')}")


if __name__ == "__main__":
    main()

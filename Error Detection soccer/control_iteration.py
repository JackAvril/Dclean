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

BASE_TRAIN_CSV = "final_train_dataset_soccer.csv"
BASE_INFERENCE_CSV = "candidate_infer_ready_soccer.csv"
BASE_LLM_LABELED_CSV = "candidate_sampled_labeled_soccer.csv"

TRAIN_SCRIPT_PATH = "train_mlp.py"
INFER_SCRIPT_PATH = "infer_mlp.py"

WORK_DIR = "active_cleaning_loop_runs_soccer_v4"
os.makedirs(WORK_DIR, exist_ok=True)

# 不要把 key 写死到代码里。运行前执行：
# export DEEPSEEK_API_KEY="你的key"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-fd93f5216ae0488f8bf32d9e2a14a14a")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

LLM_TEMPERATURE = 0.0
LLM_MAX_RETRIES = 3
LLM_SLEEP_SECONDS = 1.0

# Soccer 默认最多 3 轮，每轮最多 60 个 hard cases，防止 token 消耗过大。
MAX_ITERATIONS = 3
MAX_HARD_CASES_PER_ITER = 60
PRIORITIZE_HARD_CASES = True
RUN_INITIAL_TRAIN = True

# 是否用第一轮 detector 继续训练。初期建议 full，后续迭代 freeze_reuse 加 verifier 更稳定。
ITER_TRAIN_MODE = "freeze_reuse"

BEST_MODEL_POLICY = "f1_with_recall_floor"
RECALL_FLOOR_FOR_BEST = 0.03

CONVERGENCE_PATIENCE = 3
F1_IMPROVEMENT_EPS = 0.003
HARD_CASE_RATE_CHANGE_EPS = 0.05
LLM_USEFUL_RATE_EPS = 0.10
RECALL_DROP_EPS = 0.02

SYSTEM_PROMPT = """
You are an expert data-cleaning annotator for a soccer player/team tabular dataset.

Task:
Given one candidate cell and structured evidence, determine whether the current value is erroneous.

Important rules:
1. Use only the provided evidence.
2. Be conservative. Do not mark a value as error only because it is rare.
3. Player names, team names, city names, stadium names, and manager names are long-tail entity fields. Rarity alone is weak evidence.
4. Domain/format/numeric-range/soccer-consistency conflicts are stronger evidence.
5. If neighbor evidence strongly supports the current value and no strong domain/format/range/consistency rule contradicts it, prefer correct.
6. Team-context dominant evidence is useful but weak by itself; do not mark a plausible entity value as error solely because of a majority mismatch.
7. Return strict JSON only.

Soccer-specific consistency hints:
- Columns are: name, surname, birthyear, birthplace, position, team, city, stadium, season, manager.
- birthyear should usually be a plausible four-digit player birth year, roughly 1940..2010.
- season should usually be a plausible four-digit season year, roughly 1900..2035.
- age_at_season = season - birthyear should usually be about 15..50 for professional players.
- position should be a known soccer position, such as Goalkeeper, Defender, Midfield, Forward.
- Midfielder may be a normalization variant of Midfield if the dataset canonical form is Midfield.
- name and manager usually look like alphabetic tokens separated by underscores.
- surname usually looks alphabetic, optionally with hyphen, underscore, or apostrophe.
- In soccer v2, name and surname are target columns. Missing values or clear format violations are direct evidence; rarity alone remains weak evidence.
- For implausible age at season, decide whether birthyear or season is the candidate column; attribute the error to the candidate only if its value is the suspicious part.

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

        score += 4.0 * int(flags.get("soccer_signal_hardcase", 0))
        score += 4.0 * int(flags.get("birthyear_season_hardcase", 0))
        score += 4.0 * int(flags.get("position_hardcase", 0))
        score += 4.0 * int(flags.get("verifier_borderline", 0))
        score += 3.0 * int(flags.get("rule_neighbor_conflict", 0))
        score -= 3.0 * int(flags.get("context_fp_hardcase", 0))
        score += 1.0 * int(flags.get("is_stage_disagree", 0))
        score += 0.5 * int(flags.get("is_near_threshold", 0))

        score += 4.0 * robust_int(r.get("has_birthyear_or_season_signal"), 0)
        score += 4.0 * robust_int(r.get("position_invalid_or_normalization"), 0)
        score += 2.0 * robust_int(r.get("has_soccer_domain_signal"), 0)
        score += 2.0 * robust_int(r.get("has_soccer_format_signal"), 0)
        score += 2.0 * robust_int(r.get("has_soccer_numeric_signal"), 0)
        score += 1.5 * robust_int(r.get("has_soccer_consistency_signal"), 0)
        score += 1.0 * robust_float(r.get("soccer_consistency_score"), 0.0)
        score -= 5.0 * robust_int(r.get("is_context_only_weak_signal"), 0)

        col = str(r.get("column", ""))
        rule = str(r.get("main_rule_type", ""))
        if col in {"name", "surname", "team", "city", "stadium", "manager", "birthplace"} and rule in {"rare_value", "rare_pattern", "global_dominant_value"}:
            if robust_float(r.get("soccer_domain_rule_count"), 0) <= 0 and robust_float(r.get("soccer_format_rule_count"), 0) <= 0:
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

    # merge 后如果 pandas 产生 label_binary_llm，则优先用 LLM 的标签。
    if "label_binary_llm" in relabeled_feature_df.columns:
        relabeled_feature_df["label_binary"] = pd.to_numeric(relabeled_feature_df["label_binary_llm"], errors="coerce")
    else:
        relabeled_feature_df["label_binary"] = pd.to_numeric(relabeled_feature_df["label_binary"], errors="coerce")

    relabeled_feature_df = relabeled_feature_df[relabeled_feature_df["label_binary"].notna()].copy()
    relabeled_feature_df["label_binary"] = relabeled_feature_df["label_binary"].astype(int)

    if "label_llm" in relabeled_feature_df.columns:
        relabeled_feature_df["label"] = relabeled_feature_df["label_llm"].fillna(
            relabeled_feature_df["label_binary"].map({1: "error", 0: "correct"})
        )
    else:
        relabeled_feature_df["label"] = relabeled_feature_df["label_binary"].map({1: "error", 0: "correct"})

    relabeled_feature_df["sample_weight"] = pd.to_numeric(relabeled_feature_df["confidence"], errors="coerce").fillna(0.8).clip(lower=0.5, upper=1.0)
    relabeled_feature_df["propagation_confidence"] = pd.to_numeric(relabeled_feature_df["confidence"], errors="coerce").fillna(0.8)
    relabeled_feature_df["is_outside_candidate"] = 0
    relabeled_feature_df["error_type"] = "llm_relabel"
    relabeled_feature_df["evidence_used"] = "active_cleaning_loop"
    relabeled_feature_df["label_source"] = "llm_relabel"

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
# 5. 调用 train / infer
# ============================================================

def run_training_once(train_script: str, train_csv: str, output_dir: str, train_mode: str, detector_source_dir: str = ""):
    train_mod = load_module_from_path("train_mlp_runtime", train_script)

    train_mod.INPUT_CSV = train_csv
    train_mod.OUTPUT_DIR = output_dir
    train_mod.DETECTOR_TRAIN_MODE = train_mode
    if detector_source_dir:
        train_mod.DETECTOR_SOURCE_DIR = detector_source_dir

    # 重新设置所有输出路径
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

    train_mod.main()

    metrics_path = os.path.join(output_dir, "metrics_two_stage.json")
    metrics = load_json(metrics_path) if os.path.exists(metrics_path) else {}
    return metrics


def run_inference_once(infer_script: str, infer_csv: str, model_dir: str, output_dir: str):
    infer_mod = load_module_from_path("infer_mlp_runtime", infer_script)

    infer_mod.INPUT_CSV = infer_csv
    infer_mod.MODEL_DIR = model_dir
    infer_mod.OUTPUT_DIR = output_dir

    infer_mod.OUTPUT_ALL_CSV = os.path.join(output_dir, "inference_all_predictions.csv")
    infer_mod.OUTPUT_HARD_CASE_CSV = os.path.join(output_dir, "inference_hard_cases.csv")
    infer_mod.OUTPUT_HIGH_RISK_CSV = os.path.join(output_dir, "inference_high_risk.csv")
    infer_mod.OUTPUT_LLM_JSONL = os.path.join(output_dir, "hard_cases_for_llm.jsonl")

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

    infer_mod.EVIDENCE_RELIABILITY_JSON = os.path.join(output_dir, "evidence_reliability_soccer.json")
    infer_mod.EVIDENCE_RELIABILITY_CSV = os.path.join(output_dir, "evidence_reliability_soccer.csv")
    infer_mod.LLM_LABELED_CSV = BASE_LLM_LABELED_CSV

    infer_mod.main()

    return {
        "all_csv": infer_mod.OUTPUT_ALL_CSV,
        "hard_case_csv": infer_mod.OUTPUT_HARD_CASE_CSV,
        "high_risk_csv": infer_mod.OUTPUT_HIGH_RISK_CSV,
        "hard_case_jsonl": infer_mod.OUTPUT_LLM_JSONL,
    }


# ============================================================
# 6. 迭代控制
# ============================================================

def extract_loop_metrics(infer_all_csv: str) -> Dict[str, Any]:
    df = pd.read_csv(infer_all_csv, low_memory=False)
    total = len(df)
    final_errors = int(pd.to_numeric(df.get("final_pred_label", 0), errors="coerce").fillna(0).sum())
    hard_cases = int(pd.to_numeric(df.get("is_hard_case", 0), errors="coerce").fillna(0).sum())
    high_risk = int(pd.to_numeric(df.get("high_risk", 0), errors="coerce").fillna(0).sum())
    verifier_rejected = int(pd.to_numeric(df.get("verifier_rejected", 0), errors="coerce").fillna(0).sum())
    return {
        "total": total,
        "final_errors": final_errors,
        "hard_cases": hard_cases,
        "high_risk": high_risk,
        "verifier_rejected": verifier_rejected,
        "hard_case_rate": hard_cases / total if total else 0.0,
    }


def metric_f1_from_training_metrics(metrics: Dict[str, Any]) -> Tuple[float, float]:
    final = metrics.get("final_with_verifier", {}) or metrics.get("final_with_verifier_metrics", {})
    f1 = robust_float(final.get("f1"), 0.0)
    recall = robust_float(final.get("recall"), 0.0)
    return f1, recall


def should_stop(history: List[Dict[str, Any]]) -> bool:
    if len(history) < CONVERGENCE_PATIENCE + 1:
        return False

    recent = history[-(CONVERGENCE_PATIENCE + 1):]
    f1s = [robust_float(x.get("train_f1"), 0.0) for x in recent]
    hard_rates = [robust_float(x.get("hard_case_rate"), 0.0) for x in recent]
    useful_rates = [robust_float(x.get("llm_useful_rate"), 0.0) for x in recent]
    recalls = [robust_float(x.get("train_recall"), 0.0) for x in recent]

    f1_improve = max(f1s[1:]) - f1s[0]
    hard_change = abs(hard_rates[-1] - hard_rates[0])
    useful_rate = useful_rates[-1]
    recall_drop = recalls[0] - recalls[-1]

    if recall_drop > RECALL_DROP_EPS:
        return False

    return (
        f1_improve < F1_IMPROVEMENT_EPS
        and hard_change < HARD_CASE_RATE_CHANGE_EPS
        and useful_rate < LLM_USEFUL_RATE_EPS
    )


def main():
    history = []

    current_train_csv = BASE_TRAIN_CSV
    current_model_dir = os.path.join(WORK_DIR, "iter_0_train")
    current_infer_dir = os.path.join(WORK_DIR, "iter_0_infer")

    if RUN_INITIAL_TRAIN:
        print("\n========== Iter 0: initial training ==========")
        train_metrics = run_training_once(
            TRAIN_SCRIPT_PATH,
            current_train_csv,
            current_model_dir,
            train_mode="full",
        )
    else:
        train_metrics = {}

    print("\n========== Iter 0: initial inference ==========")
    infer_paths = run_inference_once(
        INFER_SCRIPT_PATH,
        BASE_INFERENCE_CSV,
        current_model_dir,
        current_infer_dir,
    )
    infer_metrics = extract_loop_metrics(infer_paths["all_csv"])
    train_f1, train_recall = metric_f1_from_training_metrics(train_metrics)

    history.append({
        "iter": 0,
        "train_csv": current_train_csv,
        "model_dir": current_model_dir,
        "infer_dir": current_infer_dir,
        "train_f1": train_f1,
        "train_recall": train_recall,
        **infer_metrics,
        "llm_relabeled": 0,
        "llm_changed": 0,
        "llm_useful_rate": 0.0,
    })
    save_json({"history": history}, os.path.join(WORK_DIR, "loop_history.json"))

    for it in range(1, MAX_ITERATIONS + 1):
        print(f"\n========== Iter {it}: hard-case relabel ==========")

        prev_infer_jsonl = history[-1]["infer_dir"] + "/hard_cases_for_llm.jsonl"
        relabeled_jsonl = os.path.join(WORK_DIR, f"iter_{it}_relabeled_hard_cases.jsonl")

        relabeled, changed_count = relabel_hard_cases(
            prev_infer_jsonl,
            relabeled_jsonl,
            max_cases=MAX_HARD_CASES_PER_ITER,
        )

        valid_relabeled = [r for r in relabeled if r.get("label_binary") is not None]
        useful_rate = changed_count / len(valid_relabeled) if valid_relabeled else 0.0

        print(f"[Iter {it}] relabeled={len(valid_relabeled)}, changed={changed_count}, useful_rate={useful_rate:.4f}")

        iter_train_csv = os.path.join(WORK_DIR, f"iter_{it}_train_dataset.csv")
        merge_info = merge_relabeled_into_trainset(
            base_train_csv=current_train_csv,
            inference_csv=infer_paths["all_csv"],
            relabeled_jsonl=relabeled_jsonl,
            output_train_csv=iter_train_csv,
        )

        print(f"[Iter {it}] merge_info={merge_info}")

        iter_model_dir = os.path.join(WORK_DIR, f"iter_{it}_train")
        iter_infer_dir = os.path.join(WORK_DIR, f"iter_{it}_infer")

        detector_source = current_model_dir if ITER_TRAIN_MODE == "freeze_reuse" else ""
        train_metrics = run_training_once(
            TRAIN_SCRIPT_PATH,
            iter_train_csv,
            iter_model_dir,
            train_mode=ITER_TRAIN_MODE,
            detector_source_dir=detector_source,
        )

        infer_paths = run_inference_once(
            INFER_SCRIPT_PATH,
            BASE_INFERENCE_CSV,
            iter_model_dir,
            iter_infer_dir,
        )

        infer_metrics = extract_loop_metrics(infer_paths["all_csv"])
        train_f1, train_recall = metric_f1_from_training_metrics(train_metrics)

        history.append({
            "iter": it,
            "train_csv": iter_train_csv,
            "model_dir": iter_model_dir,
            "infer_dir": iter_infer_dir,
            "train_f1": train_f1,
            "train_recall": train_recall,
            **infer_metrics,
            "llm_relabeled": len(valid_relabeled),
            "llm_changed": changed_count,
            "llm_useful_rate": useful_rate,
            "merge_info": merge_info,
        })
        save_json({"history": history}, os.path.join(WORK_DIR, "loop_history.json"))

        current_train_csv = iter_train_csv
        current_model_dir = iter_model_dir

        if should_stop(history):
            print(f"[STOP] convergence reached at iter={it}")
            break

    # 选择 best iter
    best = None
    for item in history:
        f1 = robust_float(item.get("train_f1"), 0.0)
        recall = robust_float(item.get("train_recall"), 0.0)
        score = f1
        if BEST_MODEL_POLICY == "f1_with_recall_floor" and recall < RECALL_FLOOR_FOR_BEST:
            score -= 1.0
        item = dict(item)
        item["_score"] = score
        if best is None or score > best["_score"]:
            best = item

    save_json({"history": history, "best": best}, os.path.join(WORK_DIR, "loop_history_final.json"))

    print("\n========== LOOP DONE ==========")
    print(f"Best iter: {best.get('iter') if best else None}")
    print(f"Best model dir: {best.get('model_dir') if best else None}")
    print(f"Best inference dir: {best.get('infer_dir') if best else None}")
    print(f"History: {os.path.join(WORK_DIR, 'loop_history_final.json')}")


if __name__ == "__main__":
    main()

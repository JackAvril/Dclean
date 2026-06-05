#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unified_control_iteration.py

统一版主动迭代控制脚本，覆盖：
hospital / flights / beers / rayyan / adult / soccer

它把六份 control_iteration.py 的公共逻辑抽象成一份统一调度器：

1. 初始训练
2. 初始推理
3. 从 hard_cases_for_llm.jsonl 中选择难例
4. 调用 DeepSeek / OpenAI-compatible API 重新标注 hard cases
5. 将回流标注合并进训练集
6. 继续训练，默认 freeze_reuse 复用 detector，只更新 verifier
7. 再推理
8. 记录 history
9. 根据 F1 / recall / hard case rate / LLM useful rate 判断是否停止
10. 保存 best_model

默认仍然调用各数据集目录里原来的 train_mlp*.py 和 infer_mlp*.py。
这样不会改变你原来的模型文件名和推理输出文件名。

默认输出结构保持原风格：
- active_cleaning_loop_runs_v4
- active_cleaning_loop_runs_flights_v4
- active_cleaning_loop_runs_beers_v4
- active_cleaning_loop_runs_rayyan_v4
- active_cleaning_loop_runs_adult_v4
- active_cleaning_loop_runs_soccer_v4

核心消融开关：
--disable-hardcase-feedback      不做 hard case 回流，只训练+推理
--disable-llm                    不调用 LLM，直接复用已有 relabel jsonl 或空回流
--train-mode full/freeze_reuse   控制迭代阶段训练方式
--max-iterations N
--max-hard-cases-per-iter N
--dataset all                    顺序跑六个数据集
--parallel-all                   并行跑六个数据集，适合多 GPU / 多 CPU 场景
"""

import argparse
import copy
import importlib.util
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# ============================================================
# 1. Dataset configs
# ============================================================

GENERIC_SYSTEM_PROMPT = """
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

RAYYAN_SYSTEM_PROMPT = """
You are an expert bibliographic metadata cleaning annotator.

Task:
Given one candidate bibliographic metadata cell and structured evidence, determine whether the current value is erroneous.

Important rules:
1. Use only the provided evidence.
2. Be conservative. Do not mark a value as error only because it is rare. For journal metadata, prefer structural consistency signals over weak rarity clues.
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

ADULT_SYSTEM_PROMPT = """
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

SOCCER_SYSTEM_PROMPT = """
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


@dataclass
class LoopConfig:
    dataset: str
    dataset_dir: str
    base_train_csv: str
    base_inference_csv: str
    train_script_path: str
    infer_script_path: str
    work_dir: str

    base_llm_labeled_csv: Optional[str] = None
    system_prompt: str = GENERIC_SYSTEM_PROMPT

    max_iterations: int = 3
    max_hard_cases_per_iter: int = 80
    prioritize_hard_cases: bool = True
    run_initial_train: bool = True

    iter_train_mode: str = "freeze_reuse"
    best_model_policy: str = "f1_with_recall_floor"
    recall_floor_for_best: float = 0.02

    convergence_patience: int = 5
    f1_improvement_eps: float = 0.003
    hard_case_rate_change_eps: float = 0.05
    llm_useful_rate_eps: float = 0.10
    recall_drop_eps: float = 0.01

    # Most original scripts merge relabels from candidate_infer_ready_*.csv.
    # Soccer original uses current inference_all_predictions.csv to preserve post-inference features.
    merge_from_current_inference_output: bool = False

    # old-style output compatibility
    write_iteration_history_json: bool = True
    write_active_loop_summary_json: bool = True
    write_active_loop_history_csv: bool = True
    write_loop_history_json: bool = True
    write_loop_history_final_json: bool = True


def build_configs() -> Dict[str, LoopConfig]:
    return {
        "hospital": LoopConfig(
            dataset="hospital",
            dataset_dir="Error Detection Hospital",
            base_train_csv="final_train_dataset_simple_v2.csv",
            base_inference_csv="candidate_infer_ready_v2.csv",
            train_script_path="train_mlp_v4.py",
            infer_script_path="infer_mlp_v4.py",
            work_dir="active_cleaning_loop_runs_v4",
            system_prompt=GENERIC_SYSTEM_PROMPT,
            max_hard_cases_per_iter=80,
            convergence_patience=2,
            recall_floor_for_best=0.015,
            recall_drop_eps=0.01,
        ),
        "flights": LoopConfig(
            dataset="flights",
            dataset_dir="Error Detection flights",
            base_train_csv="final_train_dataset_flights.csv",
            base_inference_csv="candidate_infer_ready_flights.csv",
            train_script_path="train_mlp_flights.py",
            infer_script_path="infer_mlp_flights.py",
            work_dir="active_cleaning_loop_runs_flights_v4",
            system_prompt=GENERIC_SYSTEM_PROMPT,
            max_hard_cases_per_iter=80,
            convergence_patience=5,
            recall_floor_for_best=0.02,
            recall_drop_eps=0.01,
        ),
        "beers": LoopConfig(
            dataset="beers",
            dataset_dir="Error Detection beers",
            base_train_csv="final_train_dataset_beers.csv",
            base_inference_csv="candidate_infer_ready_beers.csv",
            train_script_path="train_mlp_beers.py",
            infer_script_path="infer_mlp_beers.py",
            work_dir="active_cleaning_loop_runs_beers_v4",
            system_prompt=GENERIC_SYSTEM_PROMPT,
            max_hard_cases_per_iter=80,
            convergence_patience=5,
            recall_floor_for_best=0.02,
            recall_drop_eps=0.01,
        ),
        "rayyan": LoopConfig(
            dataset="rayyan",
            dataset_dir="Error Detection rayyan",
            base_train_csv="final_train_dataset_rayyan.csv",
            base_inference_csv="candidate_infer_ready_rayyan.csv",
            train_script_path="train_mlp.py",
            infer_script_path="infer_mlp.py",
            work_dir="active_cleaning_loop_runs_rayyan_v4",
            system_prompt=RAYYAN_SYSTEM_PROMPT,
            max_hard_cases_per_iter=80,
            convergence_patience=5,
            recall_floor_for_best=0.02,
            recall_drop_eps=0.01,
        ),
        "adult": LoopConfig(
            dataset="adult",
            dataset_dir="Error Detection Adult",
            base_train_csv="final_train_dataset_adult.csv",
            base_inference_csv="candidate_infer_ready_adult.csv",
            base_llm_labeled_csv="candidate_sampled_labeled_adult.csv",
            train_script_path="train_mlp.py",
            infer_script_path="infer_mlp.py",
            work_dir="active_cleaning_loop_runs_adult_v4",
            system_prompt=ADULT_SYSTEM_PROMPT,
            max_hard_cases_per_iter=60,
            convergence_patience=3,
            recall_floor_for_best=0.03,
            recall_drop_eps=0.02,
        ),
        "soccer": LoopConfig(
            dataset="soccer",
            dataset_dir="Error Detection soccer",
            base_train_csv="final_train_dataset_soccer.csv",
            base_inference_csv="candidate_infer_ready_soccer.csv",
            base_llm_labeled_csv="candidate_sampled_labeled_soccer.csv",
            train_script_path="train_mlp.py",
            infer_script_path="infer_mlp.py",
            work_dir="active_cleaning_loop_runs_soccer_v4",
            system_prompt=SOCCER_SYSTEM_PROMPT,
            max_hard_cases_per_iter=60,
            convergence_patience=3,
            recall_floor_for_best=0.03,
            recall_drop_eps=0.02,
            merge_from_current_inference_output=True,
        ),
    }


CONFIGS = build_configs()


# ============================================================
# 2. Basic IO helpers
# ============================================================

def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def robust_float(x, default=0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        if pd.isna(v):
            return default
        return v
    except Exception:
        return default


def robust_int(x, default=0) -> int:
    try:
        if x is None:
            return default
        v = int(float(x))
        return v
    except Exception:
        return default


def load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
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
            if not line:
                continue
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


def make_key(df: pd.DataFrame) -> pd.Series:
    return df["row_id"].astype(str) + "||" + df["column"].astype(str)


def load_module_from_path(module_name: str, file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到脚本: {file_path}")
    unique_name = f"{module_name}_{int(time.time() * 1000)}_{os.getpid()}"
    spec = importlib.util.spec_from_file_location(unique_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"无法加载脚本: {file_path}")
    spec.loader.exec_module(mod)
    return mod


def copy_best_dir(src_dir: str, dst_dir: str):
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)


# ============================================================
# 3. LLM relabeling
# ============================================================

def create_client():
    if OpenAI is None:
        raise RuntimeError("当前环境没有安装 openai 包。请先 pip install openai。")
    api_key = os.getenv("DEEPSEEK_API_KEY", "sk-fd93f5216ae0488f8bf32d9e2a14a14a")
    if not api_key:
        raise ValueError("未设置 DEEPSEEK_API_KEY。请先执行: export DEEPSEEK_API_KEY='你的key'")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    return OpenAI(api_key=api_key, base_url=base_url)


def build_user_prompt(rec: Dict[str, Any], max_str_len: int = 3000) -> str:
    rec2 = dict(rec)
    for k in list(rec2.keys()):
        v = rec2[k]
        if isinstance(v, str) and len(v) > max_str_len:
            rec2[k] = v[:2200] + "\n...[TRUNCATED]...\n" + v[-600:]
    return json.dumps(rec2, ensure_ascii=False, indent=2)


def parse_llm_json(text: str) -> Dict[str, Any]:
    text = str(text).strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except Exception:
        # fallback: extract first json object
        left = text.find("{")
        right = text.rfind("}")
        if left >= 0 and right > left:
            return json.loads(text[left:right + 1])
        raise


def llm_relabel_one(client, rec: Dict[str, Any], cfg: LoopConfig, response_format: bool = True) -> Dict[str, Any]:
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    max_retries = int(os.getenv("LLM_MAX_RETRIES", "3"))
    sleep_seconds = float(os.getenv("LLM_SLEEP_SECONDS", "1.0"))

    user_prompt = build_user_prompt(rec)

    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": model,
                "temperature": temperature,
                "messages": [
                    {"role": "system", "content": cfg.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
            if response_format:
                kwargs["response_format"] = {"type": "json_object"}

            resp = client.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content.strip()
            parsed = parse_llm_json(text)

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
            if attempt == max_retries - 1:
                return {
                    "row_id": rec.get("row_id"),
                    "column": rec.get("column"),
                    "value": rec.get("value"),
                    "label_binary": None,
                    "label": None,
                    "confidence": 0.0,
                    "reason_short": "llm_failed",
                    "reason_detailed": err_msg,
                    "should_change": None,
                    "label_source": "llm_failed",
                }
            time.sleep(sleep_seconds)


def select_hard_cases_for_llm(records: List[Dict[str, Any]], max_cases: int, cfg: LoopConfig) -> List[Dict[str, Any]]:
    if not cfg.prioritize_hard_cases:
        return records[:max_cases]

    def score_generic(r: Dict[str, Any]) -> float:
        flags = r.get("hard_case_flags", {}) or {}
        score = 0.0
        score += 4.0 * int(flags.get("verifier_borderline", 0))
        score += 4.0 * int(flags.get("verifier_reject_borderline", 0))
        score += 4.0 * int(flags.get("persistent_fp_high_risk", 0))
        score += 3.0 * int(flags.get("fp_conflict_fd_hardcase", 0))
        score += 3.0 * int(flags.get("recall_recovery_hardcase", 0))
        score += 2.5 * int(flags.get("rule_neighbor_conflict", 0))
        score += 2.5 * int(flags.get("sample_borderline_hardcase", 0))
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

    def score_adult(r: Dict[str, Any]) -> float:
        score = score_generic(r)
        flags = r.get("hard_case_flags", {}) or {}
        score += 4.0 * int(flags.get("adult_signal_hardcase", 0))
        score += 5.0 * int(flags.get("adult_v2_relationship_hardcase", 0))
        score += 4.0 * int(flags.get("adult_v2_sex_invalid_hardcase", 0))
        score -= 3.0 * int(flags.get("adult_v2_context_fp_hardcase", 0))

        score += 3.0 * robust_float(r.get("adult_domain_rule_count"), 0.0)
        score += 2.5 * robust_float(r.get("adult_format_rule_count"), 0.0)
        score += 2.0 * robust_float(r.get("adult_consistency_rule_count"), 0.0)
        score += 1.0 * robust_float(r.get("adult_consistency_score"), 0.0)
        score += 6.0 * robust_int(r.get("has_relationship_v2_signal"), 0)
        score += 5.0 * robust_int(r.get("sex_value_invalid"), 0)
        score += 3.5 * robust_int(r.get("has_education_v2_signal"), 0)
        score -= 5.0 * robust_int(r.get("is_context_only_weak_signal"), 0)

        col = str(r.get("column", ""))
        rule = str(r.get("main_rule_type", ""))
        if col in {"country", "occupation", "workclass"} and rule in {"rare_value", "rare_pattern", "global_dominant_value"}:
            if robust_float(r.get("adult_domain_rule_count"), 0) <= 0 and robust_float(r.get("adult_format_rule_count"), 0) <= 0:
                score -= 3.0
        return score

    def score_soccer(r: Dict[str, Any]) -> float:
        score = score_generic(r)
        flags = r.get("hard_case_flags", {}) or {}
        score += 4.0 * int(flags.get("soccer_signal_hardcase", 0))
        score += 4.0 * int(flags.get("birthyear_season_hardcase", 0))
        score += 4.0 * int(flags.get("position_hardcase", 0))
        score -= 3.0 * int(flags.get("context_fp_hardcase", 0))

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
        return score

    if cfg.dataset == "adult":
        score_fn = score_adult
    elif cfg.dataset == "soccer":
        score_fn = score_soccer
    else:
        score_fn = score_generic

    return sorted(records, key=score_fn, reverse=True)[:max_cases]


def relabel_hard_cases(
    hard_case_jsonl: str,
    output_jsonl: str,
    max_cases: int,
    cfg: LoopConfig,
    disable_llm: bool = False,
    response_format: bool = True,
) -> Tuple[List[Dict[str, Any]], int]:
    records = load_jsonl(hard_case_jsonl)
    records = select_hard_cases_for_llm(records, max_cases, cfg)

    if disable_llm:
        print("[INFO] --disable-llm enabled. Write empty relabel file.")
        save_jsonl([], output_jsonl)
        return [], 0

    if len(records) == 0:
        save_jsonl([], output_jsonl)
        return [], 0

    client = create_client()
    relabeled = []
    changed_count = 0

    for i, rec in enumerate(records, start=1):
        new_rec = llm_relabel_one(client, rec, cfg=cfg, response_format=response_format)
        relabeled.append(new_rec)

        final_pred = robust_int(rec.get("final_pred_label", 0), 0)
        if new_rec.get("label_binary") is not None and robust_int(new_rec.get("label_binary"), 0) != final_pred:
            changed_count += 1

        print(
            f"[LLM] {i}/{len(records)} "
            f"row={new_rec.get('row_id')} col={new_rec.get('column')} "
            f"label={new_rec.get('label')} conf={new_rec.get('confidence')}"
        )

    save_jsonl(relabeled, output_jsonl)
    return relabeled, changed_count


# ============================================================
# 4. Merge relabeled feedback
# ============================================================

def merge_relabeled_into_trainset(
    base_train_csv: str,
    inference_csv: str,
    relabeled_jsonl: str,
    output_train_csv: str,
) -> Dict[str, int]:
    base_train_df = pd.read_csv(base_train_csv, low_memory=False)
    inference_df = pd.read_csv(inference_csv, low_memory=False)
    relabeled_records = load_jsonl(relabeled_jsonl)
    relabeled_df = pd.DataFrame(relabeled_records)

    if len(relabeled_df) == 0 or "label_binary" not in relabeled_df.columns:
        ensure_dir(os.path.dirname(output_train_csv))
        base_train_df.to_csv(output_train_csv, index=False, encoding="utf-8-sig")
        return {"n_relabeled": 0, "n_added": 0, "n_updated": 0}

    relabeled_df = relabeled_df[relabeled_df["label_binary"].notna()].copy()
    if len(relabeled_df) == 0:
        ensure_dir(os.path.dirname(output_train_csv))
        base_train_df.to_csv(output_train_csv, index=False, encoding="utf-8-sig")
        return {"n_relabeled": 0, "n_added": 0, "n_updated": 0}

    relabeled_df["label_binary"] = relabeled_df["label_binary"].astype(int)
    relabeled_df["label"] = relabeled_df.get("label", relabeled_df["label_binary"].map({1: "error", 0: "correct"}))

    base_train_df["_key"] = make_key(base_train_df)
    inference_df["_key"] = make_key(inference_df)
    relabeled_df["_key"] = relabeled_df["row_id"].astype(str) + "||" + relabeled_df["column"].astype(str)

    relabeled_feature_df = inference_df[inference_df["_key"].isin(relabeled_df["_key"])].copy()
    if len(relabeled_feature_df) == 0:
        ensure_dir(os.path.dirname(output_train_csv))
        base_train_df.drop(columns=["_key"], errors="ignore").to_csv(output_train_csv, index=False, encoding="utf-8-sig")
        return {"n_relabeled": int(len(relabeled_df)), "n_added": 0, "n_updated": 0}

    keep_cols = [
        "_key", "label_binary", "label", "confidence",
        "reason_short", "reason_detailed", "label_source",
        "should_change",
    ]
    keep_cols = [c for c in keep_cols if c in relabeled_df.columns]
    relabeled_feature_df = relabeled_feature_df.merge(
        relabeled_df[keep_cols],
        on="_key",
        how="left",
        suffixes=("", "_llm"),
    )

    if "confidence" not in relabeled_feature_df.columns:
        relabeled_feature_df["confidence"] = 0.8
    relabeled_feature_df["confidence"] = pd.to_numeric(relabeled_feature_df["confidence"], errors="coerce").fillna(0.8).clip(lower=0.5, upper=1.0)

    relabeled_feature_df["sample_weight"] = relabeled_feature_df["confidence"]
    relabeled_feature_df["propagation_confidence"] = relabeled_feature_df["confidence"]
    relabeled_feature_df["label"] = relabeled_feature_df["label"].fillna(
        relabeled_feature_df["label_binary"].map({1: "error", 0: "correct"})
    )
    relabeled_feature_df["label_source"] = relabeled_feature_df.get("label_source", "llm_relabel")
    relabeled_feature_df["label_source"] = relabeled_feature_df["label_source"].fillna("llm_relabel")

    old_keys = set(base_train_df["_key"].astype(str).tolist())
    new_keys = set(relabeled_feature_df["_key"].astype(str).tolist())
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
    merged_df = merged_df.drop(columns=["_key"], errors="ignore")

    ensure_dir(os.path.dirname(output_train_csv))
    merged_df.to_csv(output_train_csv, index=False, encoding="utf-8-sig")

    return {
        "n_relabeled": int(len(relabeled_df)),
        "n_added": int(n_added),
        "n_updated": int(n_updated),
    }


# ============================================================
# 5. Train / infer dynamic invocation
# ============================================================

def _set_if_possible(mod, name: str, value: Any):
    try:
        setattr(mod, name, value)
    except Exception:
        pass


def run_training_once(
    train_script_path: str,
    input_csv: str,
    output_dir: str,
    mode: str,
    detector_source_dir: str = "",
    dataset: Optional[str] = None,
) -> Dict[str, Any]:
    print(f"[TRAIN] script={train_script_path}")
    print(f"[TRAIN] input={input_csv}")
    print(f"[TRAIN] output={output_dir}")
    print(f"[TRAIN] mode={mode}, detector_source={detector_source_dir}")

    train_mod = load_module_from_path("train_mlp_runtime", train_script_path)
    ensure_dir(output_dir)

    # Common old-script variables
    for name in ["INPUT_CSV"]:
        _set_if_possible(train_mod, name, input_csv)
    for name in ["OUTPUT_DIR", "MODEL_DIR"]:
        _set_if_possible(train_mod, name, output_dir)

    _set_if_possible(train_mod, "DETECTOR_TRAIN_MODE", mode)
    _set_if_possible(train_mod, "DETECTOR_SOURCE_DIR", detector_source_dir if detector_source_dir else output_dir)

    path_map = {
        "STAGE1_LAST_CKPT": "stage1_base_last.pt",
        "STAGE1_BEST_CKPT": "stage1_base_best.pt",
        "STAGE1_PREPROCESSOR": "stage1_base_preprocessor.joblib",
        "STAGE1_TRAIN_LOG": "stage1_train_log.csv",
        "STAGE2_LAST_CKPT": "stage2_cal_last.pt",
        "STAGE2_BEST_CKPT": "stage2_cal_best.pt",
        "STAGE2_PREPROCESSOR": "stage2_cal_preprocessor.joblib",
        "STAGE2_TRAIN_LOG": "stage2_train_log.csv",
        "VERIFIER_MODEL_PATH": "fp_verifier_model.joblib",
        "VERIFIER_PREPROCESSOR_PATH": "fp_verifier_preprocessor.joblib",
        "VERIFIER_CONFIG_PATH": "fp_verifier_config.json",
        "VERIFIER_METRICS_JSON": "verifier_metrics.json",
        "VERIFIER_METRICS_PATH": "verifier_metrics.json",
        "VERIFIER_TRAIN_CSV": "verifier_train_data.csv",
        "VERIFIER_VAL_CSV": "verifier_val_data.csv",
        "CONFIG_PATH": "two_stage_config.json",
        "METRICS_JSON": "metrics_two_stage.json",
        "METRICS_PATH": "metrics_two_stage.json",
        "FINAL_VAL_PRED_CSV": "final_val_predictions.csv",
        "STAGE2_TRAIN_ANALYSIS_CSV": "stage2_train_analysis.csv",
    }
    for attr, filename in path_map.items():
        _set_if_possible(train_mod, attr, os.path.join(output_dir, filename))

    # Call train script main() with clean argv; otherwise it will parse outer control_iteration argv.
    if hasattr(train_mod, "main"):
        old_argv = sys.argv[:]
        argv = [
            train_script_path,
            "--dataset", dataset or "soccer",
            "--cwd", os.getcwd(),
            "--input_csv", input_csv,
            "--output_dir", output_dir,
            "--mode", mode,
        ]
        if detector_source_dir:
            argv.extend(["--source_dir", detector_source_dir])
        sys.argv = argv
        try:
            train_mod.main()
        finally:
            sys.argv = old_argv
    else:
        raise RuntimeError(f"训练脚本没有 main(): {train_script_path}")

    metrics_path = os.path.join(output_dir, "metrics_two_stage.json")
    return load_json(metrics_path)


def run_inference_once(
    infer_script_path: str,
    input_csv: str,
    model_dir: str,
    output_dir: str,
    cfg: LoopConfig,
    llm_labeled_csv: Optional[str] = None,
) -> Dict[str, str]:
    print(f"[INFER] script={infer_script_path}")
    print(f"[INFER] input={input_csv}")
    print(f"[INFER] model={model_dir}")
    print(f"[INFER] output={output_dir}")

    infer_mod = load_module_from_path("infer_mlp_runtime", infer_script_path)
    ensure_dir(output_dir)

    _set_if_possible(infer_mod, "INPUT_CSV", input_csv)
    _set_if_possible(infer_mod, "MODEL_DIR", model_dir)
    _set_if_possible(infer_mod, "OUTPUT_DIR", output_dir)

    model_path_map = {
        "STAGE1_BEST_CKPT": "stage1_base_best.pt",
        "STAGE1_PREPROCESSOR": "stage1_base_preprocessor.joblib",
        "STAGE2_BEST_CKPT": "stage2_cal_best.pt",
        "STAGE2_PREPROCESSOR": "stage2_cal_preprocessor.joblib",
        "CONFIG_PATH": "two_stage_config.json",
        "METRICS_PATH": "metrics_two_stage.json",
        "METRICS_JSON": "metrics_two_stage.json",
        "VERIFIER_MODEL_PATH": "fp_verifier_model.joblib",
        "VERIFIER_PREPROCESSOR_PATH": "fp_verifier_preprocessor.joblib",
        "VERIFIER_CONFIG_PATH": "fp_verifier_config.json",
        "VERIFIER_METRICS_PATH": "verifier_metrics.json",
        "VERIFIER_METRICS_JSON": "verifier_metrics.json",
    }
    for attr, filename in model_path_map.items():
        _set_if_possible(infer_mod, attr, os.path.join(model_dir, filename))

    output_path_map = {
        "OUTPUT_ALL_CSV": "inference_all_predictions.csv",
        "OUTPUT_HARD_CASE_CSV": "inference_hard_cases.csv",
        "OUTPUT_HIGH_RISK_CSV": "inference_high_risk.csv",
        "OUTPUT_LLM_JSONL": "hard_cases_for_llm.jsonl",
    }
    for attr, filename in output_path_map.items():
        _set_if_possible(infer_mod, attr, os.path.join(output_dir, filename))

    if llm_labeled_csv:
        _set_if_possible(infer_mod, "LLM_LABELED_CSV", llm_labeled_csv)
    if hasattr(infer_mod, "EVIDENCE_RELIABILITY_JSON"):
        _set_if_possible(infer_mod, "EVIDENCE_RELIABILITY_JSON", os.path.join(output_dir, f"evidence_reliability_{cfg.dataset}.json"))
    if hasattr(infer_mod, "EVIDENCE_RELIABILITY_CSV"):
        _set_if_possible(infer_mod, "EVIDENCE_RELIABILITY_CSV", os.path.join(output_dir, f"evidence_reliability_{cfg.dataset}.csv"))

    # Call infer script main() with clean argv; otherwise it will parse outer control_iteration argv.
    if hasattr(infer_mod, "main"):
        old_argv = sys.argv[:]
        sys.argv = [
            infer_script_path,
            "--dataset", cfg.dataset,
            "--cwd", os.getcwd(),
            "--input_csv", input_csv,
            "--model_dir", model_dir,
            "--output_dir", output_dir,
        ]
        try:
            infer_mod.main()
        finally:
            sys.argv = old_argv
    else:
        raise RuntimeError(f"推理脚本没有 main(): {infer_script_path}")

    return {
        "all_csv": os.path.join(output_dir, "inference_all_predictions.csv"),
        "output_all_csv": os.path.join(output_dir, "inference_all_predictions.csv"),
        "hard_case_csv": os.path.join(output_dir, "inference_hard_cases.csv"),
        "high_risk_csv": os.path.join(output_dir, "inference_high_risk.csv"),
        "hard_case_jsonl": os.path.join(output_dir, "hard_cases_for_llm.jsonl"),
    }


# ============================================================
# 6. Metrics / stopping
# ============================================================

def extract_final_metrics(train_metrics: Dict[str, Any]) -> Dict[str, Any]:
    candidate_keys = [
        "final_metrics_with_verifier",
        "final_with_verifier",
        "final_with_verifier_metrics",
        "final_pipeline_metrics_with_verifier",
        "final_metrics",
    ]
    for key in candidate_keys:
        val = train_metrics.get(key)
        if isinstance(val, dict) and any(k in val for k in ["f1", "precision", "recall"]):
            return val

    # Some scripts put it under verifier_metrics-like nested keys
    verifier = train_metrics.get("verifier_metrics", {})
    if isinstance(verifier, dict):
        for key in ["final_pipeline_metrics_with_verifier", "final_metrics_with_verifier"]:
            val = verifier.get(key)
            if isinstance(val, dict):
                return val

    return {"f1": 0.0, "precision": 0.0, "recall": 0.0}


def summarize_inference(output_all_csv: str) -> Dict[str, Any]:
    if not os.path.exists(output_all_csv):
        return {
            "total": 0,
            "hard_case_count": 0,
            "hard_case_rate": 0.0,
            "high_risk_count": 0,
            "final_error_count": 0,
            "detector_error_count": 0,
            "verifier_rejected": 0,
        }

    df = pd.read_csv(output_all_csv, low_memory=False)
    total = len(df)

    def sum_col(c):
        if c not in df.columns:
            return 0
        return int(pd.to_numeric(df[c], errors="coerce").fillna(0).sum())

    hard = sum_col("is_hard_case")
    high = sum_col("high_risk") if "high_risk" in df.columns else sum_col("is_high_risk")
    final_error = sum_col("final_pred_label")
    detector_error = sum_col("detector_pred_label")
    rejected = sum_col("verifier_rejected")

    return {
        "total": int(total),
        "all_count": int(total),
        "hard_case_count": int(hard),
        "hard_cases": int(hard),
        "hard_case_rate": float(hard / total) if total else 0.0,
        "high_risk_count": int(high),
        "high_risk": int(high),
        "final_error_count": int(final_error),
        "final_errors": int(final_error),
        "detector_error_count": int(detector_error),
        "verifier_rejected": int(rejected),
    }


def is_better_result(curr: Dict[str, Any], best: Dict[str, Any], cfg: LoopConfig) -> bool:
    curr_recall = robust_float(curr.get("val_recall", curr.get("train_recall")), 0.0)
    best_recall = robust_float(best.get("val_recall", best.get("train_recall")), 0.0)
    curr_f1 = robust_float(curr.get("val_f1", curr.get("train_f1")), 0.0)
    best_f1 = robust_float(best.get("val_f1", best.get("train_f1")), 0.0)

    if cfg.best_model_policy == "recall_first":
        if curr_recall > best_recall + 1e-9:
            return True
        if abs(curr_recall - best_recall) <= 1e-9 and curr_f1 > best_f1:
            return True
        return False

    if cfg.best_model_policy == "f1_with_recall_floor":
        # Match original behavior: allow F1 improvement as long as recall is not too far below best.
        recall_floor = best_recall - cfg.recall_floor_for_best
        if curr_recall >= recall_floor and curr_f1 > best_f1 + 1e-9:
            return True
        if curr_f1 >= best_f1 and curr_recall > best_recall + 1e-9:
            return True
        return False

    return curr_f1 > best_f1 + 1e-9


def should_stop(history: List[Dict[str, Any]], cfg: LoopConfig) -> bool:
    if len(history) < cfg.convergence_patience + 1:
        return False

    recent = history[-(cfg.convergence_patience + 1):]
    f1s = [robust_float(x.get("val_f1", x.get("train_f1")), 0.0) for x in recent]
    recalls = [robust_float(x.get("val_recall", x.get("train_recall")), 0.0) for x in recent]
    hard_rates = [robust_float(x.get("hard_case_rate"), 0.0) for x in recent]
    useful_rates = [robust_float(x.get("llm_useful_rate"), 0.0) for x in recent]

    f1_improve = max(f1s[1:]) - f1s[0] if len(f1s) > 1 else 0.0
    hard_change = abs(hard_rates[-1] - hard_rates[0]) if len(hard_rates) > 1 else 0.0
    useful_recent = useful_rates[-1] if useful_rates else 0.0
    recall_drop = recalls[0] - recalls[-1] if len(recalls) > 1 else 0.0

    if recall_drop > cfg.recall_drop_eps and f1_improve < cfg.f1_improvement_eps:
        return True

    return (
        f1_improve < cfg.f1_improvement_eps
        and hard_change < cfg.hard_case_rate_change_eps
        and useful_recent < cfg.llm_useful_rate_eps
    )


# ============================================================
# 7. History output compatibility
# ============================================================

def normalize_record_for_history(record: Dict[str, Any]) -> Dict[str, Any]:
    r = dict(record)
    if "iter" not in r and "iteration" in r:
        r["iter"] = r["iteration"]
    if "iteration" not in r and "iter" in r:
        r["iteration"] = r["iter"]
    if "train_f1" not in r and "val_f1" in r:
        r["train_f1"] = r["val_f1"]
    if "train_recall" not in r and "val_recall" in r:
        r["train_recall"] = r["val_recall"]
    if "val_f1" not in r and "train_f1" in r:
        r["val_f1"] = r["train_f1"]
    if "val_recall" not in r and "train_recall" in r:
        r["val_recall"] = r["train_recall"]
    if "val_precision" not in r and "train_precision" in r:
        r["val_precision"] = r["train_precision"]
    return r


def save_history_outputs(work_dir: str, history: List[Dict[str, Any]], best: Dict[str, Any], cfg: LoopConfig):
    ensure_dir(work_dir)
    norm_history = [normalize_record_for_history(x) for x in history]
    norm_best = normalize_record_for_history(best) if best else {}

    if cfg.write_iteration_history_json:
        save_json({"history": norm_history, "best": norm_best}, os.path.join(work_dir, "iteration_history.json"))
    if cfg.write_active_loop_summary_json:
        save_json({"history": norm_history, "best": norm_best}, os.path.join(work_dir, "active_loop_summary.json"))
    if cfg.write_active_loop_history_csv:
        pd.DataFrame(norm_history).to_csv(os.path.join(work_dir, "active_loop_history.csv"), index=False, encoding="utf-8-sig")
    if cfg.write_loop_history_json:
        save_json({"history": norm_history}, os.path.join(work_dir, "loop_history.json"))
    if cfg.write_loop_history_final_json:
        save_json({"history": norm_history, "best": norm_best}, os.path.join(work_dir, "loop_history_final.json"))


# ============================================================
# 8. Main loop
# ============================================================

def apply_cli_overrides(cfg: LoopConfig, args) -> LoopConfig:
    cfg = copy.deepcopy(cfg)

    if args.work_dir:
        cfg.work_dir = args.work_dir
    if args.base_train_csv:
        cfg.base_train_csv = args.base_train_csv
    if args.base_inference_csv:
        cfg.base_inference_csv = args.base_inference_csv
    if args.base_llm_labeled_csv:
        cfg.base_llm_labeled_csv = args.base_llm_labeled_csv
    if args.train_script:
        cfg.train_script_path = args.train_script
    if args.infer_script:
        cfg.infer_script_path = args.infer_script

    if args.max_iterations is not None:
        cfg.max_iterations = args.max_iterations
    if args.max_hard_cases_per_iter is not None:
        cfg.max_hard_cases_per_iter = args.max_hard_cases_per_iter
    if args.train_mode is not None:
        cfg.iter_train_mode = args.train_mode
    if args.no_prioritize_hard_cases:
        cfg.prioritize_hard_cases = False
    if args.no_initial_train:
        cfg.run_initial_train = False
    if args.recall_floor_for_best is not None:
        cfg.recall_floor_for_best = args.recall_floor_for_best
    if args.convergence_patience is not None:
        cfg.convergence_patience = args.convergence_patience

    return cfg


def run_loop(cfg: LoopConfig, args) -> Dict[str, Any]:
    if args.cwd:
        old_cwd = os.getcwd()
        os.makedirs(args.cwd, exist_ok=True)
        os.chdir(args.cwd)
    else:
        old_cwd = None

    try:
        ensure_dir(cfg.work_dir)
        print("\n" + "=" * 80)
        print(f"[DATASET] {cfg.dataset}")
        print(f"[CWD] {os.getcwd()}")
        print(f"[WORK_DIR] {cfg.work_dir}")
        print("=" * 80)

        train_script = cfg.train_script_path
        infer_script = cfg.infer_script_path
        base_train_csv = cfg.base_train_csv
        base_inference_csv = cfg.base_inference_csv

        if not os.path.exists(base_train_csv):
            raise FileNotFoundError(f"找不到训练集: {base_train_csv}")
        if not os.path.exists(base_inference_csv):
            raise FileNotFoundError(f"找不到推理集: {base_inference_csv}")
        if not os.path.exists(train_script):
            raise FileNotFoundError(f"找不到训练脚本: {train_script}")
        if not os.path.exists(infer_script):
            raise FileNotFoundError(f"找不到推理脚本: {infer_script}")

        history: List[Dict[str, Any]] = []
        best_record: Dict[str, Any] = {}
        best_dir = os.path.join(cfg.work_dir, "best_model")

        # Iter 0
        current_train_csv = base_train_csv
        current_model_dir = os.path.join(cfg.work_dir, "iter_0_train")
        current_infer_dir = os.path.join(cfg.work_dir, "iter_0_infer")

        if cfg.run_initial_train:
            print("\n========== Iter 0: initial training ==========")
            train_metrics = run_training_once(
                train_script,
                current_train_csv,
                current_model_dir,
                mode="full",
                detector_source_dir="",
                dataset=cfg.dataset,
            )
        else:
            print("\n========== Iter 0: skip training and load metrics ==========")
            train_metrics = load_json(os.path.join(current_model_dir, "metrics_two_stage.json"))

        print("\n========== Iter 0: initial inference ==========")
        infer_paths = run_inference_once(
            infer_script,
            base_inference_csv,
            current_model_dir,
            current_infer_dir,
            cfg=cfg,
            llm_labeled_csv=cfg.base_llm_labeled_csv,
        )

        final_metrics = extract_final_metrics(train_metrics)
        infer_summary = summarize_inference(infer_paths["all_csv"])

        record0 = {
            "iter": 0,
            "iteration": 0,
            "train_csv": current_train_csv,
            "model_dir": current_model_dir,
            "train_dir": current_model_dir,
            "infer_dir": current_infer_dir,
            "val_precision": robust_float(final_metrics.get("precision"), 0.0),
            "val_recall": robust_float(final_metrics.get("recall"), 0.0),
            "val_f1": robust_float(final_metrics.get("f1"), 0.0),
            "train_precision": robust_float(final_metrics.get("precision"), 0.0),
            "train_recall": robust_float(final_metrics.get("recall"), 0.0),
            "train_f1": robust_float(final_metrics.get("f1"), 0.0),
            **infer_summary,
            "llm_relabeled_count": 0,
            "llm_relabeled": 0,
            "n_relabeled": 0,
            "n_added": 0,
            "n_updated": 0,
            "llm_changed_count": 0,
            "llm_changed": 0,
            "llm_useful_rate": 0.0,
            "merge_stat": {},
            "merge_info": {},
        }
        history.append(record0)
        best_record = dict(record0)
        copy_best_dir(current_model_dir, best_dir)
        save_history_outputs(cfg.work_dir, history, best_record, cfg)

        if args.disable_hardcase_feedback:
            print("[INFO] --disable-hardcase-feedback enabled. Stop after iter 0.")
            return {"history": history, "best": best_record, "work_dir": cfg.work_dir}

        prev_train_csv = current_train_csv
        prev_model_dir = current_model_dir
        prev_infer_paths = infer_paths

        # Iterations
        for it in range(1, cfg.max_iterations + 1):
            print(f"\n========== Iter {it}: hard-case relabel + retrain ==========")

            iter_dir = os.path.join(cfg.work_dir, f"iter_{it}")
            ensure_dir(iter_dir)

            hard_jsonl = prev_infer_paths["hard_case_jsonl"]
            relabeled_jsonl = os.path.join(iter_dir, "hard_cases_relabeled.jsonl")
            iter_train_csv = os.path.join(iter_dir, "train_with_llm_feedback.csv")
            iter_model_dir = os.path.join(cfg.work_dir, f"iter_{it}_train")
            iter_infer_dir = os.path.join(cfg.work_dir, f"iter_{it}_infer")

            try:
                relabeled, changed_count = relabel_hard_cases(
                    hard_jsonl,
                    relabeled_jsonl,
                    max_cases=cfg.max_hard_cases_per_iter,
                    cfg=cfg,
                    disable_llm=args.disable_llm,
                    response_format=not args.disable_response_format,
                )
            except RuntimeError as e:
                print(f"[STOP] LLM 调用失败: {e}")
                break

            valid_relabeled = [r for r in relabeled if r.get("label_binary") is not None]
            useful_rate = changed_count / len(valid_relabeled) if valid_relabeled else 0.0

            merge_source_csv = prev_infer_paths["all_csv"] if cfg.merge_from_current_inference_output else base_inference_csv
            merge_stats = merge_relabeled_into_trainset(
                base_train_csv=prev_train_csv,
                inference_csv=merge_source_csv,
                relabeled_jsonl=relabeled_jsonl,
                output_train_csv=iter_train_csv,
            )
            print(f"[Iter {it}] merge_stats={merge_stats}")

            detector_source = prev_model_dir if cfg.iter_train_mode == "freeze_reuse" else ""
            train_metrics = run_training_once(
                train_script,
                iter_train_csv,
                iter_model_dir,
                mode=cfg.iter_train_mode,
                detector_source_dir=detector_source,
                dataset=cfg.dataset,
            )

            infer_paths = run_inference_once(
                infer_script,
                base_inference_csv,
                iter_model_dir,
                iter_infer_dir,
                cfg=cfg,
                llm_labeled_csv=cfg.base_llm_labeled_csv,
            )

            final_metrics = extract_final_metrics(train_metrics)
            infer_summary = summarize_inference(infer_paths["all_csv"])

            current = {
                "iter": it,
                "iteration": it,
                "train_csv": iter_train_csv,
                "model_dir": iter_model_dir,
                "train_dir": iter_model_dir,
                "infer_dir": iter_infer_dir,
                "val_precision": robust_float(final_metrics.get("precision"), 0.0),
                "val_recall": robust_float(final_metrics.get("recall"), 0.0),
                "val_f1": robust_float(final_metrics.get("f1"), 0.0),
                "train_precision": robust_float(final_metrics.get("precision"), 0.0),
                "train_recall": robust_float(final_metrics.get("recall"), 0.0),
                "train_f1": robust_float(final_metrics.get("f1"), 0.0),
                **infer_summary,
                "llm_relabeled_count": int(len(valid_relabeled)),
                "llm_relabeled": int(len(valid_relabeled)),
                "n_relabeled": int(merge_stats.get("n_relabeled", 0)),
                "n_added": int(merge_stats.get("n_added", 0)),
                "n_updated": int(merge_stats.get("n_updated", 0)),
                "llm_changed_count": int(changed_count),
                "llm_changed": int(changed_count),
                "llm_useful_rate": float(useful_rate),
                "merge_stat": merge_stats,
                "merge_info": merge_stats,
            }
            history.append(current)

            print(
                f"[ITER {it}] "
                f"P={current['val_precision']:.4f} "
                f"R={current['val_recall']:.4f} "
                f"F1={current['val_f1']:.4f} "
                f"hard_rate={current['hard_case_rate']:.4f} "
                f"useful={useful_rate:.4f}"
            )

            if is_better_result(current, best_record, cfg):
                print(f"[BEST] 更新最佳模型: iter={it}")
                best_record = dict(current)
                copy_best_dir(iter_model_dir, best_dir)

            save_history_outputs(cfg.work_dir, history, best_record, cfg)

            # Stop rules
            if robust_float(current.get("val_recall"), 0.0) < robust_float(best_record.get("val_recall"), 0.0) - cfg.recall_drop_eps:
                print(f"[STOP] Recall 明显低于 best，停止在 iter={it}")
                break

            if should_stop(history, cfg):
                print(f"[STOP] 收敛，停止在 iter={it}")
                break

            prev_train_csv = iter_train_csv
            prev_model_dir = iter_model_dir
            prev_infer_paths = infer_paths

        save_history_outputs(cfg.work_dir, history, best_record, cfg)

        print("\n========== LOOP DONE ==========")
        print(f"Best iter: {best_record.get('iter')}")
        print(f"Best model dir: {best_dir}")
        print(f"Best inference dir: {best_record.get('infer_dir')}")
        print(f"History: {os.path.join(cfg.work_dir, 'active_loop_history.csv')}")
        return {"history": history, "best": best_record, "work_dir": cfg.work_dir}

    finally:
        if old_cwd is not None:
            os.chdir(old_cwd)


# ============================================================
# 9. CLI
# ============================================================

def infer_dataset_dir(root: str, dataset: str) -> str:
    cfg = CONFIGS[dataset]
    return os.path.join(root, cfg.dataset_dir)


def run_dataset_from_args(dataset: str, args) -> Dict[str, Any]:
    cfg = apply_cli_overrides(CONFIGS[dataset], args)

    # If --cwd is not provided and --root is provided, run under default dataset dir.
    local_args = copy.copy(args)
    if args.cwd is None and args.root is not None:
        local_args.cwd = infer_dataset_dir(args.root, dataset)

    return run_loop(cfg, local_args)


def parse_args():
    parser = argparse.ArgumentParser(description="Unified active hard-case feedback controller for error detection.")
    parser.add_argument("--dataset", required=True, choices=sorted(list(CONFIGS.keys()) + ["all"]))

    parser.add_argument("--root", default=None, help="Root dir containing Error Detection xxx folders. Used with --dataset all or when --cwd is omitted.")
    parser.add_argument("--cwd", default=None, help="Run inside this dataset directory.")

    parser.add_argument("--base-train-csv", default=None)
    parser.add_argument("--base-inference-csv", default=None)
    parser.add_argument("--base-llm-labeled-csv", default=None)
    parser.add_argument("--train-script", default=None)
    parser.add_argument("--infer-script", default=None)
    parser.add_argument("--work-dir", default=None)

    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--max-hard-cases-per-iter", type=int, default=None)
    parser.add_argument("--train-mode", choices=["full", "freeze_reuse"], default=None)
    parser.add_argument("--convergence-patience", type=int, default=None)
    parser.add_argument("--recall-floor-for-best", type=float, default=None)

    parser.add_argument("--disable-hardcase-feedback", action="store_true", help="Only run iter 0 train+infer.")
    parser.add_argument("--disable-llm", action="store_true", help="Do not call LLM; write empty relabel file.")
    parser.add_argument("--disable-response-format", action="store_true", help="Disable response_format={'type':'json_object'} for older OpenAI-compatible servers.")
    parser.add_argument("--no-prioritize-hard-cases", action="store_true")
    parser.add_argument("--no-initial-train", action="store_true")

    parser.add_argument("--parallel-all", action="store_true", help="When --dataset all, run datasets in parallel.")
    parser.add_argument("--max-workers", type=int, default=2)

    return parser.parse_args()


def _run_dataset_worker(dataset: str, args_dict: Dict[str, Any]) -> Dict[str, Any]:
    ns = argparse.Namespace(**args_dict)
    return run_dataset_from_args(dataset, ns)


def main():
    args = parse_args()

    if args.dataset == "all":
        datasets = ["hospital", "flights", "beers", "rayyan", "adult", "soccer"]
        if args.cwd is not None:
            raise ValueError("--dataset all 时不要设置 --cwd；请设置 --root。")
        if args.root is None:
            raise ValueError("--dataset all 需要设置 --root，例如 /mnt/mydata/dq/projects/Splittree/test")

        if args.parallel_all:
            print(f"[INFO] Running all datasets in parallel, max_workers={args.max_workers}")
            args_dict = vars(args)
            results = {}
            with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
                futs = {ex.submit(_run_dataset_worker, ds, args_dict): ds for ds in datasets}
                for fut in as_completed(futs):
                    ds = futs[fut]
                    try:
                        results[ds] = fut.result()
                        print(f"[DONE] {ds}")
                    except Exception as e:
                        results[ds] = {"error": str(e)}
                        print(f"[ERROR] {ds}: {e}")
            print(json.dumps(results, ensure_ascii=False, indent=2, default=str))
        else:
            results = {}
            for ds in datasets:
                results[ds] = run_dataset_from_args(ds, args)
            print(json.dumps(results, ensure_ascii=False, indent=2, default=str))
    else:
        run_dataset_from_args(args.dataset, args)


if __name__ == "__main__":
    main()

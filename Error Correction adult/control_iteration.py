#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adult iterative repair controller.

Pipeline:
1. Initialize accumulated pairwise labels.
2. Build inference whitelist features.
3. Train Adult pairwise LTR student.
4. Run Adult inference with conservative gates.
5. Send hard cases back to LLM for pairwise labels.
6. Merge labels and repeat.

This script is adapted from the Flights controller, but defaults to Adult paths
and Adult scripts:
  train_ltr.py
  infer_ltr.py
  build_infer_features_whitelist_adult.py
"""

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


def parse_args():
    p = argparse.ArgumentParser(description="Adult iterative repair controller")
    p.add_argument("--candidate_features", default="repair_candidate_features_v2.csv")
    p.add_argument("--allowed_cells_csv", default="predicted_error_positions_adult_augmented.csv")
    p.add_argument("--initial_pairwise_responses", default="repair_pairwise_responses_v2.jsonl")
    p.add_argument("--train_script", default="train_ltr.py")
    p.add_argument("--infer_script", default="infer_ltr.py")
    p.add_argument("--build_infer_features_script", default="build_infer_candidate_features_from_whitelist.py")
    p.add_argument("--work_dir", default="repair_iterative_workdir_adult")
    p.add_argument("--max_iterations", type=int, default=3)
    p.add_argument("--max_hard_cases_per_iter", type=int, default=200)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--global_min_margin", type=float, default=0.06)
    p.add_argument("--llm_max_per_iter", type=int, default=200)
    return p.parse_args()


ARGS = parse_args()
WORK_DIR = Path(ARGS.work_dir)
MODEL_DIR = WORK_DIR / "model_output"
ACCUM_PAIRWISE = WORK_DIR / "pairwise_responses_accumulated.jsonl"
LLM_NEW_RESPONSES = WORK_DIR / "pairwise_responses_new_from_llm.jsonl"
HARD_CASES_PATH = MODEL_DIR / "hard_cases_for_llm.jsonl"
INFER_FEATURES_PATH = WORK_DIR / "repair_candidate_features_infer_whitelist.csv"

MODEL_NAME = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-fd93f5216ae0488f8bf32d9e2a14a14a")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
TEMPERATURE = 0.0
MAX_RETRIES = 3
RETRY_SLEEP_SECONDS = 2.0


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                print(f"[WARN] Skip bad jsonl line: {path}, line={line_no}, err={e}")
    return rows


def write_jsonl(path: Path, rows):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def dedup_pairwise_rows(rows):
    dedup = {}
    for x in rows:
        key = (
            int(x.get("row_id", -1)),
            str(x.get("column", "")),
            str(x.get("candidate_a_value", "")),
            str(x.get("candidate_b_value", "")),
        )
        dedup[key] = x
    return list(dedup.values())


def bootstrap_pairwise_file():
    ensure_dir(WORK_DIR)
    ensure_dir(MODEL_DIR)
    if not ACCUM_PAIRWISE.exists():
        init_rows = load_jsonl(Path(ARGS.initial_pairwise_responses))
        init_rows = dedup_pairwise_rows(init_rows)
        write_jsonl(ACCUM_PAIRWISE, init_rows)
        print(f"[INFO] Initialized accumulated pairwise file: {ACCUM_PAIRWISE}")
        print(f"[INFO] Initial pairwise count: {len(init_rows)}")


def run_subprocess(cmd, env=None):
    print("[RUN]", " ".join(str(x) for x in cmd))
    subprocess.run([str(x) for x in cmd], check=True, env=env)


def get_client():
    if OpenAI is None:
        raise ImportError("openai package is not installed.")
    if not API_KEY:
        raise ValueError("Missing DEEPSEEK_API_KEY.")
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def normalize_col_name(column: str) -> str:
    c = str(column).strip().lower()
    aliases = {
        "marital-status": "maritalstatus",
        "marital_status": "maritalstatus",
        "hours-per-week": "hoursperweek",
        "hours_per_week": "hoursperweek",
        "native-country": "country",
        "native_country": "country",
    }
    return aliases.get(c, c)


def build_column_specific_instruction(column: str) -> str:
    col = normalize_col_name(column)
    if col == "relationship":
        return "本列是 relationship。优先选择与 sex、maritalstatus、age 一致的候选，例如 Husband->Male 且已婚，Wife->Female 且已婚。"
    if col == "sex":
        return "本列是 sex。relationship=Husband 强支持 Male，relationship=Wife 强支持 Female。"
    if col == "maritalstatus":
        return "本列是 maritalstatus。若 relationship 是 Husband/Wife，通常应为 Married-civ-spouse 或 Married-AF-spouse。"
    if col == "education":
        return "本列是 education。注意 age 与 education 的合理性，年轻人不应无强证据地出现过高学历。"
    if col == "age":
        return "本列是 age。优先合法年龄/年龄桶，不要凭统计高频强行修改。"
    if col == "hoursperweek":
        return "本列是 hoursperweek。优先合法数值或明显格式修复，不要只凭 profile 高频修改。"
    if col == "income":
        return "本列是 income。优先规范化 <=50K/>50K 到 LessThan50K/MoreThan50K，不要凭职业学历猜测收入。"
    return "优先考虑规则一致性、Adult profile 一致性、domain 合法性、上下文一致性和候选来源可靠性。"


def build_pair_prompt(obj):
    column_instruction = build_column_specific_instruction(obj.get("column", ""))
    prompt = f"""
你现在在做 Adult 表格数据清洗中的错误修复排序任务。

目标：
对于一个被检测为错误的单元格，请在两个候选修复值 A 和 B 中判断谁更适合作为最终修复值。
你必须只在 A、B、Unknown 中三选一：
- A：候选 A 更好
- B：候选 B 更好
- Unknown：无法可靠判断，或者两个都不够好

Adult 字段包括：
age, workclass, education, maritalstatus, occupation, relationship, race, sex, hoursperweek, country, income。

判断原则：
1. 优先考虑强规则证据和 expected values。
2. 优先考虑 relationship/sex/maritalstatus/age/education 的语义一致性。
3. domain 合法性是必要条件，但不能单独压过强上下文。
4. profile 分布、邻居多数、列内高频是弱证据。
5. 如果两个候选都不可靠，输出 Unknown。

本列专属判断：
{column_instruction}

请严格输出 JSON，不要输出额外解释文字。
JSON 格式如下：
{{
  "winner": "A" 或 "B" 或 "Unknown",
  "reason_type": "rule_consistency" / "adult_profile_consistency" / "domain_validity" / "relationship_consistency" / "education_age_consistency" / "context_match" / "cooccurrence_support" / "spelling_similarity" / "pattern_match" / "uncertain" / "other",
  "reason": "一句简短原因"
}}

样本：
{json.dumps(obj, ensure_ascii=False, indent=2)}
""".strip()
    return prompt


def call_llm_pairwise(client, prompt: str):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=TEMPERATURE,
                messages=[
                    {"role": "system", "content": "你是一个严谨的 Adult 表格数据清洗排序专家。你必须只输出 JSON。"},
                    {"role": "user", "content": prompt},
                ],
            )
            text = resp.choices[0].message.content.strip()
            try:
                return json.loads(text)
            except Exception:
                start = text.find("{")
                end = text.rfind("}")
                if start != -1 and end != -1 and end > start:
                    return json.loads(text[start:end+1])
                raise ValueError(f"Invalid JSON output: {text}")
        except Exception as e:
            if attempt == MAX_RETRIES:
                return {
                    "winner": "Unknown",
                    "reason_type": "uncertain",
                    "reason": f"LLM call failed: {str(e)}",
                }
            time.sleep(RETRY_SLEEP_SECONDS)


def label_hard_cases_with_llm():
    hard_cases = load_jsonl(HARD_CASES_PATH)
    if not hard_cases:
        print("[INFO] No hard cases found, skip LLM relabeling.")
        return []

    client = get_client()
    new_rows = []
    for obj in hard_cases[:min(ARGS.max_hard_cases_per_iter, ARGS.llm_max_per_iter)]:
        llm_out = call_llm_pairwise(client, build_pair_prompt(obj))
        winner = llm_out.get("winner", "Unknown")
        if winner not in {"A", "B", "Unknown"}:
            winner = "Unknown"
        new_rows.append({
            "row_id": int(obj["row_id"]),
            "column": str(obj["column"]),
            "dirty_value": str(obj.get("dirty_value", "")),
            "candidate_a_value": str(obj.get("candidate_a_value", "")),
            "candidate_b_value": str(obj.get("candidate_b_value", "")),
            "winner": winner,
            "reason_type": llm_out.get("reason_type", "uncertain"),
            "reason": llm_out.get("reason", ""),
            "source": "iterative_hard_case_llm_adult",
        })

    write_jsonl(LLM_NEW_RESPONSES, new_rows)
    print(f"[INFO] New LLM hard-case labels: {len(new_rows)}")
    return new_rows


def merge_new_pairwise_rows(new_rows):
    all_rows = load_jsonl(ACCUM_PAIRWISE)
    all_rows.extend(new_rows)
    all_rows = dedup_pairwise_rows(all_rows)
    write_jsonl(ACCUM_PAIRWISE, all_rows)
    print(f"[INFO] Accumulated pairwise count: {len(all_rows)}")


def build_infer_features(script_path: Path):
    env = os.environ.copy()
    env["INPUT_CANDIDATE_FEATURES"] = str(ARGS.candidate_features)
    env["INPUT_ALLOWED_CELLS"] = str(ARGS.allowed_cells_csv)
    env["OUTPUT_INFER_FEATURES"] = str(INFER_FEATURES_PATH)

    # Prefer explicit CLI because our Adult whitelist builder supports it.
    cmd = [
        "python", str(script_path),
        "--candidate_features", str(ARGS.candidate_features),
        "--allowed_cells", str(ARGS.allowed_cells_csv),
        "--output", str(INFER_FEATURES_PATH),
    ]
    run_subprocess(cmd, env=env)

    if INFER_FEATURES_PATH.exists():
        print(f"[INFO] Found infer whitelist features at expected path: {INFER_FEATURES_PATH}")
        return

    expected_name = INFER_FEATURES_PATH.name
    candidate_paths = [
        Path(ARGS.candidate_features).resolve().parent / expected_name,
        Path.cwd().resolve() / expected_name,
        script_path.resolve().parent / expected_name,
    ]

    for p in candidate_paths:
        if p.exists():
            ensure_dir(INFER_FEATURES_PATH.parent)
            shutil.copy2(p, INFER_FEATURES_PATH)
            print(f"[INFO] copied infer features to controller workdir: {INFER_FEATURES_PATH}")
            return

    searched = "\n".join(str(p) for p in candidate_paths)
    raise FileNotFoundError(
        "Expected infer whitelist features file not found after build step.\n"
        f"Expected: {INFER_FEATURES_PATH}\n"
        f"Also searched:\n{searched}"
    )


def main():
    bootstrap_pairwise_file()
    script_dir = Path(__file__).resolve().parent

    train_script = Path(ARGS.train_script)
    infer_script = Path(ARGS.infer_script)
    build_script = Path(ARGS.build_infer_features_script)

    if not train_script.is_absolute():
        train_script = script_dir / train_script
    if not infer_script.is_absolute():
        infer_script = script_dir / infer_script
    if not build_script.is_absolute():
        build_script = script_dir / build_script

    for name, path in [("Train", train_script), ("Infer", infer_script), ("Build", build_script)]:
        if not path.exists():
            raise FileNotFoundError(f"{name} script not found: {path}")

    resume_ckpt = None

    for it in range(1, ARGS.max_iterations + 1):
        print(f"\n================ ADULT ITERATION {it} ================")

        build_infer_features(build_script)

        train_cmd = [
            "python", str(train_script),
            "--candidate_features", str(ARGS.candidate_features),
            "--pairwise_responses", str(ACCUM_PAIRWISE),
            "--output_dir", str(MODEL_DIR),
            "--epochs", str(ARGS.epochs),
            "--batch_size", str(ARGS.batch_size),
            "--lr", str(ARGS.lr),
        ]
        if resume_ckpt:
            train_cmd.extend(["--resume_checkpoint", str(resume_ckpt)])
        run_subprocess(train_cmd)

        best_ckpt = MODEL_DIR / "best_model.pt"
        if not best_ckpt.exists():
            raise FileNotFoundError(f"Best checkpoint not found after training: {best_ckpt}")

        env = os.environ.copy()
        env["REPAIR_INFER_OUTPUT_DIR"] = str(MODEL_DIR)

        infer_cmd = [
            "python", str(infer_script),
            "--candidate_features", str(INFER_FEATURES_PATH),
            "--allowed_cells_csv", str(ARGS.allowed_cells_csv),
            "--model_dir", str(MODEL_DIR),
            "--checkpoint", str(best_ckpt),
            "--global_min_margin", str(ARGS.global_min_margin),
            "--max_hard_cases", str(ARGS.max_hard_cases_per_iter),
        ]
        run_subprocess(infer_cmd, env=env)

        if it >= ARGS.max_iterations:
            print("[INFO] Reached max iterations.")
            break

        new_rows = label_hard_cases_with_llm()
        if not new_rows:
            print("[INFO] No new hard-case labels; stop.")
            break

        before = len(load_jsonl(ACCUM_PAIRWISE))
        merge_new_pairwise_rows(new_rows)
        after = len(load_jsonl(ACCUM_PAIRWISE))

        if after <= before:
            print("[INFO] No new unique pairwise rows added; stop.")
            break

        resume_ckpt = best_ckpt

    print("\n========== Adult iterative repair finished ==========")
    print(f"work_dir: {WORK_DIR}")
    print(f"model_dir: {MODEL_DIR}")
    print(f"accum_pairwise: {ACCUM_PAIRWISE}")
    print(f"top1_repairs: {MODEL_DIR / 'inference_top1_repairs.csv'}")
    print(f"ranked_candidates: {MODEL_DIR / 'inference_all_candidate_ranked.csv'}")


if __name__ == "__main__":
    main()

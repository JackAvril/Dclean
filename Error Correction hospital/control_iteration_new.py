import os
import json
import time
import subprocess
import argparse
from pathlib import Path

from openai import OpenAI


def parse_args():
    p = argparse.ArgumentParser(description="Iterative repair controller: wide train, build infer whitelist features, then whitelist inference")
    p.add_argument("--candidate_features", default="repair_candidate_features_v2.csv")
    p.add_argument("--allowed_cells_csv", default="allowed_cells_from_detection.csv")
    p.add_argument("--initial_pairwise_responses", default="repair_pairwise_responses_v2.jsonl")
    p.add_argument("--train_script", default="train_ltr_new.py")
    p.add_argument("--infer_script", default="infer_ltr_new.py")
    p.add_argument("--build_infer_features_script", default="build_infer_candidate_features_from_whitelist.py")
    p.add_argument("--work_dir", default="repair_iterative_workdir_widetrain_whitelistinfer")
    p.add_argument("--max_iterations", type=int, default=3)
    p.add_argument("--max_hard_cases_per_iter", type=int, default=200)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--global_min_margin", type=float, default=0.06)
    return p.parse_args()


ARGS = parse_args()
WORK_DIR = Path(ARGS.work_dir)
MODEL_DIR = WORK_DIR / "model_output"
ACCUM_PAIRWISE = WORK_DIR / "pairwise_responses_accumulated.jsonl"
LLM_NEW_RESPONSES = WORK_DIR / "pairwise_responses_new_from_llm.jsonl"
HARD_CASES_PATH = MODEL_DIR / "hard_cases_for_llm.jsonl"
INFER_FEATURES_PATH = Path("repair_candidate_features_infer_whitelist.csv")

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
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows):
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
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def get_client():
    if not API_KEY:
        raise ValueError("Missing DEEPSEEK_API_KEY / API_KEY")
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def build_column_specific_instruction(column: str) -> str:
    column = str(column)
    if column in {"MeasureCode", "Stateavg"}:
        return "本列是结构化编码列。优先看 family/prefix/suffix 是否一致，以及它是否与同行的 State / MeasureCode / Stateavg 保持结构一致；不要只因为字符串相似就选择 sibling code。"
    if column == "Condition":
        return "本列是疾病/主题分类列。优先看它是否与 MeasureCode family 一致：hf->heart failure, ami->heart attack, pn->pneumonia, scip->surgical infection prevention。"
    if column == "MeasureName":
        return "本列是长文本 canonical phrase。优先看它是否与 MeasureCode 和 Condition 一致，不要只因为词面相似就选择语义不匹配的候选。"
    if column == "Score":
        return "本列是百分比。已经是合法百分号格式的原值要非常谨慎改动；优先恢复 malformed percent（例如 95x -> 95%），不要无根据把合法分数改成别的合法分数。"
    if column == "Sample":
        return "本列通常应符合 '\\d+ patients' 模式。优先恢复 pattern typo，不要随意补空值，也不要选择不符合样式的候选。"
    return "优先考虑规则一致性、同行上下文一致性、统计支持和合理的拼写/模式变换。"


def build_pair_prompt(obj):
    column_instruction = build_column_specific_instruction(obj.get("column", ""))
    prompt = f"""
你现在在做表格数据清洗中的错误修复排序任务。

目标：
对于一个被检测为错误的单元格，请在两个候选修复值 A 和 B 中判断谁更适合作为最终修复值。
你必须只在 A、B、Unknown 中三选一：
- A：候选 A 更好
- B：候选 B 更好
- Unknown：无法可靠判断，或者两个都不够好

通用判断原则：
1. 优先考虑规则一致性
2. 再考虑上下文一致性
3. 再考虑统计支持
4. 再考虑字符串变换合理性
5. 如果两个候选都明显不合理，输出 Unknown

本列专属判断：
{column_instruction}

请严格输出 JSON，不要输出额外解释文字。
JSON 格式如下：
{{
  "winner": "A" 或 "B" 或 "Unknown",
  "reason_type": "rule_consistency" / "context_match" / "cooccurrence_support" / "spelling_similarity" / "pattern_match" / "uncertain" / "other",
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
                    {"role": "system", "content": "你是一个严谨的表格数据清洗排序专家。你必须只输出 JSON。"},
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
                    "reason": f"LLM call failed: {str(e)}"
                }
            time.sleep(RETRY_SLEEP_SECONDS)


def label_hard_cases_with_llm():
    hard_cases = load_jsonl(HARD_CASES_PATH)
    if not hard_cases:
        print("[INFO] No hard cases found, skip LLM relabeling.")
        return []
    client = get_client()
    new_rows = []
    for obj in hard_cases[:ARGS.max_hard_cases_per_iter]:
        llm_out = call_llm_pairwise(client, build_pair_prompt(obj))
        new_rows.append({
            "row_id": int(obj["row_id"]),
            "column": str(obj["column"]),
            "dirty_value": str(obj.get("dirty_value", "")),
            "candidate_a_value": str(obj.get("candidate_a_value", "")),
            "candidate_b_value": str(obj.get("candidate_b_value", "")),
            "winner": llm_out.get("winner", "Unknown"),
            "reason_type": llm_out.get("reason_type", "uncertain"),
            "reason": llm_out.get("reason", ""),
            "source": "iterative_hard_case_llm",
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

    # 兼容“路径写死在代码里”的脚本：直接运行；若脚本支持环境变量，可读取环境变量
    # 为了更稳，同时在当前工作目录创建一份临时脚本参数文件也可以，但先不复杂化
    run_subprocess(["python", str(script_path)], env=env)

    if not INFER_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Expected infer whitelist features file not found after build step: {INFER_FEATURES_PATH}"
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

    if not train_script.exists():
        raise FileNotFoundError(f"Train script not found: {train_script}")
    if not infer_script.exists():
        raise FileNotFoundError(f"Infer script not found: {infer_script}")
    if not build_script.exists():
        raise FileNotFoundError(f"Build infer features script not found: {build_script}")

    resume_ckpt = None
    for it in range(1, ARGS.max_iterations + 1):
        print(f"\n================ ITERATION {it} ================")

        train_cmd = [
            "python", str(train_script),
            "--candidate_features", ARGS.candidate_features,
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
        expected_meta = [
            MODEL_DIR / "feature_columns.json",
            MODEL_DIR / "reason_label_mapping.json",
            MODEL_DIR / "column_profiles.json",
        ]
        for meta_path in expected_meta:
            if not meta_path.exists():
                raise FileNotFoundError(f"Training finished but missing required file: {meta_path}")

        # 根本性修复：先构建“白名单预裁后的推理特征文件”
        build_infer_features(build_script)

        infer_cmd = [
            "python", str(infer_script),
            "--candidate_features", str(INFER_FEATURES_PATH),
            "--allowed_cells_csv", ARGS.allowed_cells_csv,
            "--model_dir", str(MODEL_DIR),
            "--checkpoint", str(best_ckpt),
            "--global_min_margin", str(ARGS.global_min_margin),
            "--max_hard_cases", str(ARGS.max_hard_cases_per_iter),
        ]
        run_subprocess(infer_cmd)

        new_rows = label_hard_cases_with_llm()
        if not new_rows:
            print("[INFO] No new hard-case labels, stop iteration.")
            break
        merge_new_pairwise_rows(new_rows)
        resume_ckpt = best_ckpt

    print("\n[OK] Iterative process finished.")
    print("[OK] Final top-1 repairs:", MODEL_DIR / "inference_top1_repairs.csv")
    print("[OK] Final ranked candidates:", MODEL_DIR / "inference_all_candidate_ranked.csv")
    print("[OK] Infer whitelist features:", INFER_FEATURES_PATH)


if __name__ == "__main__":
    main()

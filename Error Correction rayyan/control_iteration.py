"""
Rayyan iterative repair controller, logic-migrated from the Flights controller.

Migration principle:
- Keep the full Flights iterative controller:
  bootstrap accumulated pairwise labels, train, build whitelist inference features,
  run inference, mine hard cases, query LLM, merge pairwise labels, resume training.
- Convert scripts / prompts / workdir:
  Flights time consensus -> Rayyan journal/article metadata + canonical consistency.
"""

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
    p.add_argument("--allowed_cells_csv", default="predicted_error_positions_consensus_augmented.csv")
    p.add_argument("--initial_pairwise_responses", default="repair_pairwise_responses_v2.jsonl")
    p.add_argument("--train_script", default="train_ltr.py")
    p.add_argument("--infer_script", default="infer_ltr.py")
    p.add_argument("--build_infer_features_script", default="build_infer_candidate_features_from_whitelist.py")
    p.add_argument("--work_dir", default="repair_iterative_workdir_rayyan")
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
    column_raw = str(column)
    column = column_raw.strip().lower()

    if column == "article_language":
        return (
            "本列是文章语言列。优先选择 eng/fre/ger/spa 等规范三字母语言值；"
            "重点参考 journal_title、jounral_abbreviation、journal_issn、article_title 对 article_language 的支持；"
            "若候选来自 rayyan_metadata_consensus、journal_title_to_language、issn_to_language 或 canonical_language_restore，应重点考虑。"
        )
    if column == "journal_issn":
        return (
            "本列是期刊 ISSN。优先选择合法 ISSN 格式 ####-#### 或 ####-###X；"
            "若候选通过 ISSN 校验位、来自 journal_title_to_issn / journal_abbr_to_issn / rayyan_metadata_consensus / canonical_issn_restore，应重点考虑。"
        )
    if column == "journal_title":
        return (
            "本列是期刊全名。优先选择与 journal_issn、jounral_abbreviation、article_language 一致的候选；"
            "不要仅因为列内高频或字符串相似就选择。"
        )
    if column in {"jounral_abbreviation", "journal_abbreviation"}:
        return (
            "本列是期刊缩写/别名。优先选择与 journal_issn、journal_title、article_language 一致的候选；"
            "注意数据中字段名可能拼写为 jounral_abbreviation。"
        )
    if column == "article_jcreated_at":
        return (
            "本列是日期列。优先选择可规范化为日期的候选，例如 2005-01-01 -> 1/1/05；"
            "候选来自 canonical_date_restore 或 date_value 时更可信。"
        )
    if column == "article_jvolumn":
        return (
            "本列是期刊卷号。优先选择合法整数卷号，并参考 journal_title + article_jissue 对 article_jvolumn 的一致性。"
        )
    if column == "article_jissue":
        return (
            "本列是期刊期号。优先选择合法整数期号，并参考 journal_title + article_jvolumn 对 article_jissue 的一致性。"
        )
    if column == "article_pagination":
        return (
            "本列是页码范围。优先选择格式规范的页码，例如 1442–6 -> 1442-6；"
            "候选来自 canonical_pagination_restore 时更可信。"
        )
    if column == "article_title":
        return (
            "本列是文章标题。优先选择与 id、journal metadata、language 上下文一致的标题。"
        )
    return "优先考虑规则一致性、同行上下文一致性、Rayyan metadata consensus、统计支持和合理的规范化/拼写变换。"

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
2. 再考虑同行上下文一致性
3. 再考虑 Rayyan journal/article metadata consensus
4. 再考虑 language / ISSN / date / pagination 等 canonical 格式恢复
5. 再考虑统计支持和字符串变换合理性
5. 如果两个候选都明显不合理，输出 Unknown

本列专属判断：
{column_instruction}

请严格输出 JSON，不要输出额外解释文字。
JSON 格式如下：
{{
  "winner": "A" 或 "B" 或 "Unknown",
  "reason_type": "rule_consistency" / "context_match" / "rayyan_metadata_consistency" / "canonical_format" / "cooccurrence_support" / "spelling_similarity" / "pattern_match" / "uncertain" / "other",
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
    """
    构建推理阶段使用的白名单候选特征文件。

    注意：有些 build_infer_features_whitelist 脚本会把输出保存到
    candidate_features 所在目录，而主控脚本希望文件在 WORK_DIR 下。
    因此这里做路径兼容：
    1) 优先使用 WORK_DIR / repair_candidate_features_infer_whitelist.csv；
    2) 如果不存在，就去候选特征文件所在目录、当前目录、脚本目录查找；
    3) 找到后复制一份到 WORK_DIR，保证后续 infer_script 路径稳定。
    """
    env = os.environ.copy()
    env["INPUT_CANDIDATE_FEATURES"] = str(ARGS.candidate_features)
    env["INPUT_ALLOWED_CELLS"] = str(ARGS.allowed_cells_csv)
    env["OUTPUT_INFER_FEATURES"] = str(INFER_FEATURES_PATH)

    run_subprocess(["python", str(script_path)], env=env)

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
            INFER_FEATURES_PATH.write_bytes(p.read_bytes())
            print(f"[INFO] build script saved infer features to: {p}")
            print(f"[INFO] copied infer features to controller workdir: {INFER_FEATURES_PATH}")
            return

    searched = "\n".join(str(p) for p in candidate_paths)
    raise FileNotFoundError(
        "Expected infer whitelist features file not found after build step.\n"
        f"Expected: {INFER_FEATURES_PATH}\n"
        f"Also searched:\n{searched}\n"
        "请检查 build_infer_features_whitelist_rayyan_logic_migrated_current_dir.py 的 OUTPUT_INFER_FEATURES / 保存目录。"
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

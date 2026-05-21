import os
import json
import time
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI


# ============================================================
# 1. 用户配置区
# ============================================================

INPUT_JSONL = "candidate_sampled_with_context_rayyan.jsonl"

OUTPUT_JSONL = "candidate_sampled_labeled_rayyan.jsonl"
OUTPUT_CSV = "candidate_sampled_labeled_rayyan.csv"

MODEL = "deepseek-chat"

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-fd93f5216ae0488f8bf32d9e2a14a14a")
BASE_URL = "https://api.deepseek.com"

TEMPERATURE = 0.0
MAX_TOKENS = 1200

SKIP_ALREADY_DONE = True

# 并行相关
ENABLE_PARALLEL = True
MAX_WORKERS = 6

# 单请求完成后的轻微节流（每个线程内）
SLEEP_BETWEEN_REQUESTS_SEC = 0.1

MAX_RETRIES = 5
RETRY_BACKOFF_SEC = 2.0

LIMIT = None


# ============================================================
# 2. 系统提示词
# ============================================================

SYSTEM_PROMPT = """
You are an expert data-cleaning annotation assistant for bibliographic metadata.

The user will provide a candidate table cell and its structured context.
Your task is to decide whether the candidate cell is erroneous.

You must return valid JSON only.

Required JSON schema:
{
  "is_error": true,
  "label": "error",
  "confidence": 0.97,
  "error_type": "pattern_error",
  "reason_short": "A concise reason.",
  "reason_detailed": "A slightly more detailed explanation using the evidence.",
  "evidence_used": ["fd_conflict", "neighbor_majority", "pattern"],
  "suggested_correct_value": "eng",
  "needs_human_review": false
}

Rules:
1. label must be either "error" or "correct".
2. is_error must be true if label == "error", otherwise false.
3. confidence must be a float between 0 and 1.
4. error_type must be one of:
   ["spelling_variant", "semantic_conflict", "pattern_error", "domain_error",
    "missing_value_error", "likely_correct", "unclear"].
5. evidence_used should be a list selected from:
   ["fd_conflict", "neighbor_majority", "column_frequency", "pattern",
    "row_context", "column_context", "bayesian_signal", "other"].
6. suggested_correct_value can be null if the cell seems correct or if uncertain.
7. needs_human_review should be true when the evidence is conflicting or uncertain.
8. Output must be json only, with no markdown fences.
9. Use the provided evidence conservatively. Prefer strong structural evidence and neighbor consensus over weak frequency clues.
10. For bibliographic metadata, pay attention to journal-title / abbreviation / ISSN consistency, language canonical forms, date plausibility, volume/issue numeric consistency, pagination anomalies, and author-list formatting.
11. When the value is missing but should co-occur with a paired field, consider missing_value_error.
12. When the evidence is weak and the current value still looks plausible, prefer "correct" or "unclear".
"""


# ============================================================
# 3. 工具函数
# ============================================================

def build_summary_prefix(sample_obj: Dict[str, Any]) -> str:
    summary_ctx = sample_obj.get("summary_context", {})
    if not summary_ctx:
        return ""

    lines = []
    lines.append("Summary signals:")
    lines.append(f"- semantic_type: {summary_ctx.get('semantic_type')}")
    lines.append(f"- violation_count: {summary_ctx.get('violation_count')}")
    lines.append(f"- conflict_score: {summary_ctx.get('conflict_score')}")
    lines.append(f"- value_frequency: {summary_ctx.get('value_frequency')}")
    lines.append(f"- value_frequency_rank: {summary_ctx.get('value_frequency_rank')}")
    lines.append(f"- numeric_value: {summary_ctx.get('numeric_value')}")
    lines.append(f"- date_value: {summary_ctx.get('date_value')}")
    lines.append(f"- language_canonical: {summary_ctx.get('language_canonical')}")
    lines.append(f"- issn_canonical: {summary_ctx.get('issn_canonical')}")
    lines.append(f"- neighbor_majority_value: {summary_ctx.get('neighbor_majority_value')}")
    lines.append(f"- neighbor_majority_ratio: {summary_ctx.get('neighbor_majority_ratio')}")
    lines.append(f"- prior_error_probability: {summary_ctx.get('prior_error_probability')}")
    lines.append(f"- posterior_error_probability: {summary_ctx.get('posterior_error_probability')}")
    lines.append(f"- main_rule_type: {summary_ctx.get('main_rule_type')}")
    lines.append(f"- main_usage_role: {summary_ctx.get('main_usage_role')}")
    lines.append(f"- strong_rule_count: {summary_ctx.get('strong_rule_count')}")
    lines.append(f"- candidate_generation_rule_count: {summary_ctx.get('candidate_generation_rule_count')}")
    lines.append(f"- fd_like_count: {summary_ctx.get('fd_like_count')}")
    lines.append(f"- context_rule_count: {summary_ctx.get('context_rule_count')}")
    lines.append(f"- global_rule_count: {summary_ctx.get('global_rule_count')}")
    lines.append(f"- rare_value_count: {summary_ctx.get('rare_value_count')}")
    lines.append(f"- typo_rule_count: {summary_ctx.get('typo_rule_count')}")
    lines.append(f"- pattern_rule_count: {summary_ctx.get('pattern_rule_count')}")
    lines.append(f"- schema_rule_count: {summary_ctx.get('schema_rule_count')}")
    lines.append(f"- canonical_rule_count: {summary_ctx.get('canonical_rule_count')}")
    lines.append(f"- pattern_bucket: {summary_ctx.get('pattern_bucket')}")
    lines.append(f"- rarity_bucket: {summary_ctx.get('rarity_bucket')}")
    lines.append(f"- neighbor_bucket: {summary_ctx.get('neighbor_bucket')}")
    lines.append(f"- canonical_bucket: {summary_ctx.get('canonical_bucket')}")
    lines.append(f"- date_value_bucket: {summary_ctx.get('date_value_bucket')}")
    lines.append(f"- language_bucket: {summary_ctx.get('language_bucket')}")
    return "\n".join(lines)


def build_user_prompt(sample_obj: Dict[str, Any]) -> str:
    detail_ctx = sample_obj.get("detail_context", {})
    llm_text = detail_ctx.get("llm_context_text")
    summary_prefix = build_summary_prefix(sample_obj)

    if llm_text:
        return (
            "Please analyze the following candidate bibliographic metadata cell context and output json only.\n\n"
            + summary_prefix
            + "\n\nDetailed context:\n"
            + llm_text
        )

    return (
        "Please analyze the following candidate bibliographic metadata cell context and output json only.\n\n"
        + summary_prefix
        + "\n\nFull sample json:\n"
        + json.dumps(sample_obj, ensure_ascii=False, indent=2)
    )


def create_client() -> OpenAI:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY 未设置。请先设置环境变量 DEEPSEEK_API_KEY。")
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)


def load_jsonl(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
            if limit is not None and len(items) >= limit:
                break
    return items


def load_done_objects(path: str) -> List[Dict[str, Any]]:
    objs = []
    if not os.path.exists(path):
        return objs

    bad_count = 0
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                objs.append(json.loads(line))
            except Exception as e:
                bad_count += 1
                print(f"[WARN] 跳过损坏行: line={lineno}, err={e}")

    if bad_count > 0:
        print(f"[WARN] 读取已有输出时，共跳过 {bad_count} 行损坏 JSON。")
    return objs


def build_order_map(samples: List[Dict[str, Any]]) -> Dict[Tuple[int, str], int]:
    order = {}
    for idx, sample in enumerate(samples):
        key = (int(sample["row_id"]), str(sample["column"]))
        order[key] = idx
    return order


def safe_parse_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json", "", 1).strip()
    return json.loads(text)


def normalize_label_result(result: Dict[str, Any]) -> Dict[str, Any]:
    label = str(result.get("label", "")).strip().lower()
    is_error = result.get("is_error", None)

    if label not in {"error", "correct"}:
        if isinstance(is_error, bool):
            label = "error" if is_error else "correct"
        else:
            label = "correct"

    if not isinstance(is_error, bool):
        is_error = (label == "error")

    confidence = result.get("confidence", 0.5)
    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.5
    confidence = min(max(confidence, 0.0), 1.0)

    error_type = str(result.get("error_type", "unclear")).strip()
    allowed_error_types = {
        "spelling_variant",
        "semantic_conflict",
        "pattern_error",
        "domain_error",
        "missing_value_error",
        "likely_correct",
        "unclear",
    }
    if error_type not in allowed_error_types:
        error_type = "unclear"

    evidence_used = result.get("evidence_used", [])
    if not isinstance(evidence_used, list):
        evidence_used = ["other"]

    suggested_correct_value = result.get("suggested_correct_value", None)
    if suggested_correct_value == "":
        suggested_correct_value = None

    needs_human_review = result.get("needs_human_review", False)
    if not isinstance(needs_human_review, bool):
        needs_human_review = False

    return {
        "is_error": is_error,
        "label": label,
        "confidence": confidence,
        "error_type": error_type,
        "reason_short": str(result.get("reason_short", "")).strip(),
        "reason_detailed": str(result.get("reason_detailed", "")).strip(),
        "evidence_used": evidence_used,
        "suggested_correct_value": suggested_correct_value,
        "needs_human_review": needs_human_review,
    }


def call_deepseek_with_retry(user_prompt: str) -> Dict[str, Any]:
    last_err = None
    client = create_client()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )

            content = response.choices[0].message.content
            parsed = safe_parse_json(content)
            if SLEEP_BETWEEN_REQUESTS_SEC > 0:
                time.sleep(SLEEP_BETWEEN_REQUESTS_SEC)
            return normalize_label_result(parsed)

        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SEC * attempt)
            else:
                raise last_err


def process_one_sample(sample: Dict[str, Any], input_order: int) -> Dict[str, Any]:
    row_id = int(sample["row_id"])
    column = str(sample["column"])
    value = sample.get("value", None)

    user_prompt = build_user_prompt(sample)
    label_result = call_deepseek_with_retry(user_prompt)

    return {
        "row_id": row_id,
        "column": column,
        "value": value,
        "bucket_id": sample.get("bucket_id"),
        "cluster_id": sample.get("cluster_id"),
        "sample_role": sample.get("sample_role"),
        "dist_to_center": sample.get("dist_to_center"),
        "summary_context": sample.get("summary_context", {}),
        "label_result": label_result,
        "_input_order": input_order,
    }


def sort_output_objects(objs: List[Dict[str, Any]], order_map: Dict[Tuple[int, str], int]) -> List[Dict[str, Any]]:
    def _key(obj: Dict[str, Any]):
        key = (int(obj["row_id"]), str(obj["column"]))
        return order_map.get(key, 10**12)

    dedup = {}
    for obj in objs:
        key = (int(obj["row_id"]), str(obj["column"]))
        dedup[key] = obj

    sorted_objs = sorted(dedup.values(), key=_key)
    for obj in sorted_objs:
        obj.pop("_input_order", None)
    return sorted_objs


def build_final_rows(sorted_objs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    final_rows = []
    for obj in sorted_objs:
        lr = obj["label_result"]
        summary_ctx = obj.get("summary_context", {})

        final_rows.append({
            "row_id": obj["row_id"],
            "column": obj["column"],
            "value": obj.get("value"),

            "label": lr["label"],
            "is_error": lr["is_error"],
            "confidence": lr["confidence"],
            "error_type": lr["error_type"],
            "reason_short": lr["reason_short"],
            "reason_detailed": lr["reason_detailed"],
            "suggested_correct_value": lr["suggested_correct_value"],
            "needs_human_review": lr["needs_human_review"],
            "evidence_used": "|".join(lr["evidence_used"]),

            "bucket_id": obj.get("bucket_id"),
            "cluster_id": obj.get("cluster_id"),
            "sample_role": obj.get("sample_role"),
            "dist_to_center": obj.get("dist_to_center"),

            "semantic_type": summary_ctx.get("semantic_type"),
            "violation_count": summary_ctx.get("violation_count"),
            "conflict_score": summary_ctx.get("conflict_score"),
            "value_frequency": summary_ctx.get("value_frequency"),
            "value_frequency_rank": summary_ctx.get("value_frequency_rank"),
            "numeric_value": summary_ctx.get("numeric_value"),
            "date_value": summary_ctx.get("date_value"),
            "language_canonical": summary_ctx.get("language_canonical"),
            "issn_canonical": summary_ctx.get("issn_canonical"),
            "neighbor_majority_value": summary_ctx.get("neighbor_majority_value"),
            "neighbor_majority_ratio": summary_ctx.get("neighbor_majority_ratio"),
            "prior_error_probability": summary_ctx.get("prior_error_probability"),
            "posterior_error_probability": summary_ctx.get("posterior_error_probability"),

            "main_rule_type": summary_ctx.get("main_rule_type"),
            "main_usage_role": summary_ctx.get("main_usage_role"),
            "strong_rule_count": summary_ctx.get("strong_rule_count"),
            "candidate_generation_rule_count": summary_ctx.get("candidate_generation_rule_count"),
            "fd_like_count": summary_ctx.get("fd_like_count"),
            "context_rule_count": summary_ctx.get("context_rule_count"),
            "global_rule_count": summary_ctx.get("global_rule_count"),

            "rare_value_count": summary_ctx.get("rare_value_count"),
            "typo_rule_count": summary_ctx.get("typo_rule_count"),
            "pattern_rule_count": summary_ctx.get("pattern_rule_count"),
            "schema_rule_count": summary_ctx.get("schema_rule_count"),
            "canonical_rule_count": summary_ctx.get("canonical_rule_count"),

            "pattern_bucket": summary_ctx.get("pattern_bucket"),
            "rarity_bucket": summary_ctx.get("rarity_bucket"),
            "neighbor_bucket": summary_ctx.get("neighbor_bucket"),
            "canonical_bucket": summary_ctx.get("canonical_bucket"),
            "date_value_bucket": summary_ctx.get("date_value_bucket"),
            "language_bucket": summary_ctx.get("language_bucket"),
        })
    return final_rows


# ============================================================
# 4. 主流程
# ============================================================

def main():
    all_samples = load_jsonl(INPUT_JSONL, limit=LIMIT)
    order_map = build_order_map(all_samples)

    existing_objs = load_done_objects(OUTPUT_JSONL) if SKIP_ALREADY_DONE else []
    done_keys = {
        (int(obj["row_id"]), str(obj["column"]))
        for obj in existing_objs
        if "row_id" in obj and "column" in obj
    }

    pending = []
    skipped = 0
    for idx, sample in enumerate(all_samples):
        key = (int(sample["row_id"]), str(sample["column"]))
        if key in done_keys:
            skipped += 1
            continue
        pending.append((idx, sample))

    new_results = []
    processed = 0

    if ENABLE_PARALLEL and len(pending) > 0:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_map = {
                executor.submit(process_one_sample, sample, idx): (idx, sample)
                for idx, sample in pending
            }

            for future in as_completed(future_map):
                idx, sample = future_map[future]
                row_id = int(sample["row_id"])
                column = str(sample["column"])
                try:
                    out_obj = future.result()
                    new_results.append(out_obj)
                    lr = out_obj["label_result"]
                    processed += 1
                    print(f"[OK] row_id={row_id}, column={column}, label={lr['label']}, conf={lr['confidence']:.3f}")
                except Exception as e:
                    print(f"[ERROR] row_id={row_id}, column={column}, err={e}")
    else:
        for idx, sample in pending:
            row_id = int(sample["row_id"])
            column = str(sample["column"])
            try:
                out_obj = process_one_sample(sample, idx)
                new_results.append(out_obj)
                lr = out_obj["label_result"]
                processed += 1
                print(f"[OK] row_id={row_id}, column={column}, label={lr['label']}, conf={lr['confidence']:.3f}")
            except Exception as e:
                print(f"[ERROR] row_id={row_id}, column={column}, err={e}")

    merged_objs = existing_objs + new_results
    sorted_objs = sort_output_objects(merged_objs, order_map)

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for obj in sorted_objs:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    final_rows = build_final_rows(sorted_objs)
    if final_rows:
        pd.DataFrame(final_rows).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\n===== 完成 =====")
    print(f"新处理样本数: {processed}")
    print(f"跳过已完成样本数: {skipped}")
    print(f"JSONL 输出: {OUTPUT_JSONL}")
    print(f"CSV 输出: {OUTPUT_CSV}")
    print(f"并行开关: {ENABLE_PARALLEL}")
    print(f"并行线程数: {MAX_WORKERS}")


if __name__ == "__main__":
    main()

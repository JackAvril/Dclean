import os
import json
import time
from typing import Any, Dict, List, Optional

import pandas as pd
from openai import OpenAI


# ============================================================
# 1. 用户配置区
# ============================================================

INPUT_JSONL = "candidate_sampled_with_context_v2.jsonl"

OUTPUT_JSONL = "candidate_sampled_labeled_v2.jsonl"
OUTPUT_CSV = "candidate_sampled_labeled_v2.csv"

# DeepSeek 模型
MODEL = "deepseek-chat"

# API 配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-fd93f5216ae0488f8bf32d9e2a14a14a")
BASE_URL = "https://api.deepseek.com"

# 生成参数
TEMPERATURE = 0.0
MAX_TOKENS = 1200

# 断点续跑
SKIP_ALREADY_DONE = True

# 调用节流
SLEEP_BETWEEN_REQUESTS_SEC = 0.3

# 重试
MAX_RETRIES = 5
RETRY_BACKOFF_SEC = 2.0

# 如果你想只先跑前 N 条做测试，可设成整数；None 表示全量
LIMIT = None


# ============================================================
# 2. 系统提示词
# ============================================================

SYSTEM_PROMPT = """
You are an expert data-cleaning annotation assistant.

The user will provide a candidate spreadsheet cell and its structured context.
Your task is to decide whether the candidate cell is erroneous.

You must return valid JSON only.

Required JSON schema:
{
  "is_error": true,
  "label": "error",
  "confidence": 0.97,
  "error_type": "spelling_variant",
  "reason_short": "A concise reason.",
  "reason_detailed": "A slightly more detailed explanation using the evidence.",
  "evidence_used": ["fd_conflict", "neighbor_majority", "column_frequency"],
  "suggested_correct_value": "acute care hospitals",
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
10. When the current value is a corrupted variant of a dominant canonical value, prefer error_type="spelling_variant".
"""


# ============================================================
# 3. 工具函数
# ============================================================

def build_summary_prefix(sample_obj: Dict[str, Any]) -> str:
    """
    从 summary_context 中抽出一段简短摘要，放在 llm_context_text 前面，
    让模型优先注意新版聚类/规则统计特征。
    """
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
    lines.append(f"- pattern_bucket: {summary_ctx.get('pattern_bucket')}")
    lines.append(f"- rarity_bucket: {summary_ctx.get('rarity_bucket')}")
    lines.append(f"- neighbor_bucket: {summary_ctx.get('neighbor_bucket')}")

    return "\n".join(lines)


def build_user_prompt(sample_obj: Dict[str, Any]) -> str:
    """
    优先使用 detail_context 里的 llm_context_text。
    同时补一个 summary_context 摘要前缀。
    """
    detail_ctx = sample_obj.get("detail_context", {})
    llm_text = detail_ctx.get("llm_context_text")
    summary_prefix = build_summary_prefix(sample_obj)

    if llm_text:
        return (
            "Please analyze the following candidate cell context and output json only.\n\n"
            + summary_prefix
            + "\n\nDetailed context:\n"
            + llm_text
        )

    return (
        "Please analyze the following candidate cell context and output json only.\n\n"
        + summary_prefix
        + "\n\nFull sample json:\n"
        + json.dumps(sample_obj, ensure_ascii=False, indent=2)
    )


def create_client() -> OpenAI:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError(
            "DEEPSEEK_API_KEY 未设置。请先设置环境变量 DEEPSEEK_API_KEY。"
        )

    return OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=BASE_URL,
    )


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


def load_done_keys(path: str) -> set:
    done = set()
    if not os.path.exists(path):
        return done

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = (int(obj["row_id"]), str(obj["column"]))
            done.add(key)
    return done


def safe_parse_json(text: str) -> Dict[str, Any]:
    """
    尝试解析模型返回 JSON。
    """
    text = text.strip()

    # 处理偶尔出现的 markdown fence
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json", "", 1).strip()

    return json.loads(text)


def normalize_label_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    规整输出，避免模型字段稍有偏差。
    """
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


def call_deepseek_with_retry(client: OpenAI, user_prompt: str) -> Dict[str, Any]:
    last_err = None

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
            return normalize_label_result(parsed)

        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SEC * attempt)
            else:
                raise last_err


# ============================================================
# 4. 主流程
# ============================================================

def main():
    client = create_client()

    samples = load_jsonl(INPUT_JSONL, limit=LIMIT)

    done_keys = load_done_keys(OUTPUT_JSONL) if SKIP_ALREADY_DONE else set()

    out_f = open(OUTPUT_JSONL, "a" if SKIP_ALREADY_DONE else "w", encoding="utf-8")

    processed = 0
    skipped = 0

    try:
        for sample in samples:
            row_id = int(sample["row_id"])
            column = str(sample["column"])
            value = sample.get("value", None)

            key = (row_id, column)
            if key in done_keys:
                skipped += 1
                continue

            user_prompt = build_user_prompt(sample)

            try:
                label_result = call_deepseek_with_retry(client, user_prompt)

                out_obj = {
                    "row_id": row_id,
                    "column": column,
                    "value": value,
                    "bucket_id": sample.get("bucket_id"),
                    "cluster_id": sample.get("cluster_id"),
                    "sample_role": sample.get("sample_role"),
                    "dist_to_center": sample.get("dist_to_center"),
                    "summary_context": sample.get("summary_context", {}),
                    "label_result": label_result,
                }

                out_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                out_f.flush()

                processed += 1
                print(
                    f"[OK] row_id={row_id}, column={column}, "
                    f"label={label_result['label']}, conf={label_result['confidence']:.3f}"
                )

            except Exception as e:
                print(f"[ERROR] row_id={row_id}, column={column}, err={e}")

            time.sleep(SLEEP_BETWEEN_REQUESTS_SEC)

    finally:
        out_f.close()

    # 重新读 JSONL，统一导出 CSV
    final_rows = []
    if os.path.exists(OUTPUT_JSONL):
        with open(OUTPUT_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                obj = json.loads(line)
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

                    "pattern_bucket": summary_ctx.get("pattern_bucket"),
                    "rarity_bucket": summary_ctx.get("rarity_bucket"),
                    "neighbor_bucket": summary_ctx.get("neighbor_bucket"),
                })

    if final_rows:
        pd.DataFrame(final_rows).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\n===== 完成 =====")
    print(f"新处理样本数: {processed}")
    print(f"跳过已完成样本数: {skipped}")
    print(f"JSONL 输出: {OUTPUT_JSONL}")
    print(f"CSV 输出: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
import os
import json
import time
from typing import Any, Dict, List, Optional

import pandas as pd
from openai import OpenAI


# ============================================================
# 1. 用户配置区
# ============================================================

INPUT_JSONL = "candidate_sampled_with_context_new.jsonl"

OUTPUT_JSONL = "candidate_sampled_labeled_new.jsonl"
OUTPUT_CSV = "candidate_sampled_labeled_new.csv"

# DeepSeek 模型
MODEL = "deepseek-chat"

# API 配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-fd93f5216ae0488f8bf32d9e2a14a14a")
BASE_URL = "https://api.deepseek.com"

# 生成参数
TEMPERATURE = 0.0
MAX_TOKENS = 800

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
"""


# ============================================================
# 3. 工具函数
# ============================================================

def build_user_prompt(sample_obj: Dict[str, Any]) -> str:
    """
    优先使用 detail_context 里的 llm_context_text。
    若不存在，则退化为整个 JSON。
    """
    detail_ctx = sample_obj.get("detail_context", {})
    llm_text = detail_ctx.get("llm_context_text")

    if llm_text:
        return (
            "Please analyze the following candidate cell context and output json only.\n\n"
            f"{llm_text}"
        )

    return (
        "Please analyze the following candidate cell context and output json only.\n\n"
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
        for i, line in enumerate(f):
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

    if isinstance(is_error, bool):
        pass
    else:
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

    flat_rows = []

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
                    "label_result": label_result,
                }

                out_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                out_f.flush()

                flat_rows.append({
                    "row_id": row_id,
                    "column": column,
                    "value": value,
                    "label": label_result["label"],
                    "is_error": label_result["is_error"],
                    "confidence": label_result["confidence"],
                    "error_type": label_result["error_type"],
                    "reason_short": label_result["reason_short"],
                    "reason_detailed": label_result["reason_detailed"],
                    "suggested_correct_value": label_result["suggested_correct_value"],
                    "needs_human_review": label_result["needs_human_review"],
                    "evidence_used": "|".join(label_result["evidence_used"]),
                    "bucket_id": sample.get("bucket_id"),
                    "cluster_id": sample.get("cluster_id"),
                    "sample_role": sample.get("sample_role"),
                    "dist_to_center": sample.get("dist_to_center"),
                })

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

    # 如果启用了断点续跑，这里把旧 JSONL 重新读出来，统一导出 CSV
    final_rows = []
    if os.path.exists(OUTPUT_JSONL):
        with open(OUTPUT_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                lr = obj["label_result"]
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
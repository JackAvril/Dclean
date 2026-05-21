import os
import json
import time
import traceback
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from openai import OpenAI


# ============================================================
# 1. 用户配置区
# ============================================================

INPUT_JSONL = "candidate_sampled_with_context_adult.jsonl"

OUTPUT_JSONL = "candidate_sampled_labeled_adult.jsonl"
OUTPUT_CSV = "candidate_sampled_labeled_adult.csv"

# DeepSeek 模型
MODEL = "deepseek-chat"

# API 配置
# 注意：不要把 API key 写死到代码里。
# 运行前执行：
# export DEEPSEEK_API_KEY="你的key"
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

# 控制 prompt 长度，避免单条样本 token 过大
# 如果想给 LLM 更多上下文，可调大；如果钱包紧张，可调小到 8000。
MAX_DETAIL_CONTEXT_CHARS = 12000

# 是否把完整 sample json 放进 prompt 兜底。
# 通常不需要，detail_context.llm_context_text 已经足够。
INCLUDE_FULL_SAMPLE_WHEN_NO_LLM_TEXT = True

# 输出过程中是否打印详细错误栈
PRINT_TRACEBACK_ON_ERROR = False


# ============================================================
# 2. 系统提示词：Adult 数据集专用
# ============================================================

SYSTEM_PROMPT = """
You are an expert data-cleaning annotation assistant for the Adult census-style tabular dataset.

The user will provide a candidate spreadsheet cell and its structured context.
Your task is to decide whether the candidate cell is erroneous.

You must return valid JSON only.

Required JSON schema:
{
  "is_error": true,
  "label": "error",
  "confidence": 0.97,
  "error_type": "domain_error",
  "reason_short": "A concise reason.",
  "reason_detailed": "A slightly more detailed explanation using the evidence.",
  "evidence_used": ["domain_constraint", "adult_consistency", "row_context"],
  "suggested_correct_value": "Male",
  "needs_human_review": false
}

Rules:
1. label must be either "error" or "correct".
2. is_error must be true if label == "error", otherwise false.
3. confidence must be a float between 0 and 1.
4. error_type must be one of:
   [
     "domain_error",
     "format_error",
     "semantic_conflict",
     "missing_value_error",
     "context_conflict",
     "likely_correct",
     "unclear"
   ].
5. evidence_used should be a list selected from:
   [
     "domain_constraint",
     "format_constraint",
     "adult_consistency",
     "fd_conflict",
     "neighbor_majority",
     "column_frequency",
     "pattern",
     "row_context",
     "column_context",
     "bayesian_signal",
     "other"
   ].
6. suggested_correct_value can be null if the cell seems correct or if uncertain.
7. needs_human_review should be true when the evidence is conflicting or uncertain.
8. Output must be JSON only, with no markdown fences.

Adult dataset-specific guidance:
1. age is usually a bucket such as <18, 18-21, 22-25, ..., >70. A parseable but non-canonical age may be an error or normalization issue.
2. hoursperweek may be a number or a numeric range such as 18-21; it should broadly be within 0..120.
3. income should be canonicalized to LessThan50K or MoreThan50K. Values like <=50K or >50K are suspicious if the dataset uses canonical labels.
4. Domain fields include workclass, education, maritalstatus, occupation, relationship, race, sex, country, and income. A value outside the known domain is strong evidence of error.
5. relationship=Wife should usually imply sex=Female. relationship=Husband should usually imply sex=Male.
6. relationship=Husband/Wife conflicts with maritalstatus such as Never-married, Divorced, Separated, or Widowed.
7. Very young age buckets, especially <18, conflict with very high education such as Bachelors, Masters, Prof-school, or Doctorate.
8. workclass=Never-worked conflicts with nonzero hoursperweek or a normal occupation.
9. Treat country and occupation carefully: rare values can still be correct. Do not mark them as errors based only on low frequency.
10. Prefer strong domain/format/consistency evidence over weak frequency or dominant-value clues.
11. If evidence is weak and the current value is plausible, prefer "correct" or "unclear".
"""


# ============================================================
# 3. 工具函数：prompt 构造
# ============================================================

def build_summary_prefix(sample_obj: Dict[str, Any]) -> str:
    """
    从 summary_context 中抽出一段简短摘要，放在 llm_context_text 前面，
    让模型优先注意聚类/规则统计/adult 专属特征。
    """
    summary_ctx = sample_obj.get("summary_context", {})

    if not summary_ctx:
        return ""

    lines = []
    lines.append("Summary signals:")
    lines.append(f"- semantic_type: {summary_ctx.get('semantic_type')}")
    lines.append(f"- detected_type: {summary_ctx.get('detected_type')}")
    lines.append(f"- violation_count: {summary_ctx.get('violation_count')}")
    lines.append(f"- conflict_score: {summary_ctx.get('conflict_score')}")
    lines.append(f"- value_frequency: {summary_ctx.get('value_frequency')}")
    lines.append(f"- value_frequency_rank: {summary_ctx.get('value_frequency_rank')}")

    lines.append(f"- age_lower: {summary_ctx.get('age_lower')}")
    lines.append(f"- age_upper: {summary_ctx.get('age_upper')}")
    lines.append(f"- hours_lower: {summary_ctx.get('hours_lower')}")
    lines.append(f"- hours_upper: {summary_ctx.get('hours_upper')}")
    lines.append(f"- canonical_income: {summary_ctx.get('canonical_income')}")
    lines.append(f"- domain_valid: {summary_ctx.get('domain_valid')}")
    lines.append(f"- education_order: {summary_ctx.get('education_order')}")
    lines.append(f"- adult_value_bucket: {summary_ctx.get('adult_value_bucket')}")

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

    lines.append(f"- adult_domain_rule_count: {summary_ctx.get('adult_domain_rule_count')}")
    lines.append(f"- adult_format_rule_count: {summary_ctx.get('adult_format_rule_count')}")
    lines.append(f"- adult_consistency_rule_count: {summary_ctx.get('adult_consistency_rule_count')}")

    lines.append(f"- pattern_bucket: {summary_ctx.get('pattern_bucket')}")
    lines.append(f"- rarity_bucket: {summary_ctx.get('rarity_bucket')}")
    lines.append(f"- neighbor_bucket: {summary_ctx.get('neighbor_bucket')}")
    lines.append(f"- domain_bucket: {summary_ctx.get('domain_bucket')}")
    lines.append(f"- adult_consistency_bucket: {summary_ctx.get('adult_consistency_bucket')}")
    lines.append(f"- age_sampling_bucket: {summary_ctx.get('age_sampling_bucket')}")
    lines.append(f"- hours_sampling_bucket: {summary_ctx.get('hours_sampling_bucket')}")

    return "\n".join(lines)


def truncate_text(text: str, max_chars: int) -> str:
    if text is None:
        return ""
    text = str(text)
    if max_chars is None or max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    head = text[: int(max_chars * 0.75)]
    tail = text[-int(max_chars * 0.25):]
    return (
        head
        + "\n\n...[TRUNCATED_TO_SAVE_TOKENS]...\n\n"
        + tail
    )


def build_user_prompt(sample_obj: Dict[str, Any]) -> str:
    """
    优先使用 detail_context 里的 llm_context_text。
    同时补一个 summary_context 摘要前缀。
    """
    detail_ctx = sample_obj.get("detail_context", {}) or {}
    llm_text = detail_ctx.get("llm_context_text")
    summary_prefix = build_summary_prefix(sample_obj)

    if llm_text:
        llm_text = truncate_text(llm_text, MAX_DETAIL_CONTEXT_CHARS)
        return (
            "Please analyze the following candidate cell context and output JSON only.\n\n"
            + summary_prefix
            + "\n\nDetailed context:\n"
            + llm_text
        )

    if INCLUDE_FULL_SAMPLE_WHEN_NO_LLM_TEXT:
        sample_text = truncate_text(
            json.dumps(sample_obj, ensure_ascii=False, indent=2),
            MAX_DETAIL_CONTEXT_CHARS
        )
        return (
            "Please analyze the following candidate cell context and output JSON only.\n\n"
            + summary_prefix
            + "\n\nFull sample JSON:\n"
            + sample_text
        )

    return (
        "Please analyze the following candidate cell context and output JSON only.\n\n"
        + summary_prefix
    )


# ============================================================
# 4. API / JSON 工具
# ============================================================

def create_client() -> OpenAI:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError(
            "DEEPSEEK_API_KEY 未设置。请先运行：export DEEPSEEK_API_KEY='你的key'"
        )

    return OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=BASE_URL,
    )


def load_jsonl(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] JSONL 解析失败，跳过 line={line_no}: {e}")
                continue

            if limit is not None and len(items) >= limit:
                break
    return items


def load_done_keys(path: str) -> Set[Tuple[int, str]]:
    """
    断点续跑：鲁棒读取旧输出。
    如果某行损坏，不让整个程序崩掉。
    """
    done = set()
    if not os.path.exists(path):
        return done

    bad_lines = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                key = (int(obj["row_id"]), str(obj["column"]))
                done.add(key)
            except Exception:
                bad_lines += 1
                print(f"[WARN] 跳过损坏的已完成输出行 line={line_no}")

    if bad_lines > 0:
        print(f"[WARN] OUTPUT_JSONL 中有 {bad_lines} 行损坏，已跳过。")

    return done


def safe_parse_json(text: str) -> Dict[str, Any]:
    """
    尝试解析模型返回 JSON。
    """
    text = (text or "").strip()

    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json", "", 1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 兜底：截取第一个 { 到最后一个 }
        l = text.find("{")
        r = text.rfind("}")
        if l >= 0 and r > l:
            return json.loads(text[l:r + 1])
        raise


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
        "domain_error",
        "format_error",
        "semantic_conflict",
        "missing_value_error",
        "context_conflict",
        "likely_correct",
        "unclear",
    }
    if error_type not in allowed_error_types:
        error_type = "unclear"

    evidence_used = result.get("evidence_used", [])
    if not isinstance(evidence_used, list):
        evidence_used = ["other"]

    allowed_evidence = {
        "domain_constraint",
        "format_constraint",
        "adult_consistency",
        "fd_conflict",
        "neighbor_majority",
        "column_frequency",
        "pattern",
        "row_context",
        "column_context",
        "bayesian_signal",
        "other",
    }
    evidence_used = [str(x) for x in evidence_used if str(x) in allowed_evidence]
    if not evidence_used:
        evidence_used = ["other"]

    suggested_correct_value = result.get("suggested_correct_value", None)
    if suggested_correct_value == "":
        suggested_correct_value = None

    needs_human_review = result.get("needs_human_review", False)
    if not isinstance(needs_human_review, bool):
        needs_human_review = str(needs_human_review).strip().lower() in {"true", "1", "yes"}

    reason_short = str(result.get("reason_short", "")).strip()
    reason_detailed = str(result.get("reason_detailed", "")).strip()

    # label=correct 时，error_type 更适合设为 likely_correct，避免后续训练误解。
    if label == "correct" and error_type not in {"likely_correct", "unclear"}:
        error_type = "likely_correct"

    return {
        "is_error": is_error,
        "label": label,
        "confidence": confidence,
        "error_type": error_type,
        "reason_short": reason_short,
        "reason_detailed": reason_detailed,
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
            print(f"[WARN] API/JSON 失败 attempt={attempt}/{MAX_RETRIES}: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SEC * attempt)
            else:
                raise last_err


# ============================================================
# 5. 输出展开
# ============================================================

def flatten_output_row(obj: Dict[str, Any]) -> Dict[str, Any]:
    lr = obj["label_result"]
    summary_ctx = obj.get("summary_context", {}) or {}

    return {
        "row_id": obj["row_id"],
        "column": obj["column"],
        "value": obj.get("value"),

        "label": lr["label"],
        "label_binary": 1 if lr["label"] == "error" else 0,
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
        "signal_priority": obj.get("signal_priority"),

        "semantic_type": summary_ctx.get("semantic_type"),
        "detected_type": summary_ctx.get("detected_type"),
        "violation_count": summary_ctx.get("violation_count"),
        "conflict_score": summary_ctx.get("conflict_score"),
        "value_frequency": summary_ctx.get("value_frequency"),
        "value_frequency_rank": summary_ctx.get("value_frequency_rank"),

        "age_lower": summary_ctx.get("age_lower"),
        "age_upper": summary_ctx.get("age_upper"),
        "hours_lower": summary_ctx.get("hours_lower"),
        "hours_upper": summary_ctx.get("hours_upper"),
        "canonical_income": summary_ctx.get("canonical_income"),
        "domain_valid": summary_ctx.get("domain_valid"),
        "education_order": summary_ctx.get("education_order"),
        "adult_value_bucket": summary_ctx.get("adult_value_bucket"),

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

        "adult_domain_rule_count": summary_ctx.get("adult_domain_rule_count"),
        "adult_format_rule_count": summary_ctx.get("adult_format_rule_count"),
        "adult_consistency_rule_count": summary_ctx.get("adult_consistency_rule_count"),

        # 兼容旧训练/传播脚本
        "time_window_rule_count": summary_ctx.get("time_window_rule_count", 0),
        "time_value_minutes": summary_ctx.get("time_value_minutes"),
        "time_window_bucket": summary_ctx.get("time_window_bucket", "adult_not_applicable"),
        "time_value_bucket": summary_ctx.get("time_value_bucket", "adult_not_applicable"),

        "pattern_bucket": summary_ctx.get("pattern_bucket"),
        "rarity_bucket": summary_ctx.get("rarity_bucket"),
        "neighbor_bucket": summary_ctx.get("neighbor_bucket"),
        "domain_bucket": summary_ctx.get("domain_bucket"),
        "adult_consistency_bucket": summary_ctx.get("adult_consistency_bucket"),
        "age_sampling_bucket": summary_ctx.get("age_sampling_bucket"),
        "hours_sampling_bucket": summary_ctx.get("hours_sampling_bucket"),
    }


def export_csv_from_jsonl(jsonl_path: str, csv_path: str):
    final_rows = []
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    final_rows.append(flatten_output_row(obj))
                except Exception as e:
                    print(f"[WARN] 导出 CSV 时跳过损坏行 line={line_no}: {e}")

    if final_rows:
        df = pd.DataFrame(final_rows)
        df = df.sort_values(by=["row_id", "column"]).reset_index(drop=True)
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(csv_path, index=False, encoding="utf-8-sig")


# ============================================================
# 6. 主流程
# ============================================================

def main():
    client = create_client()
    samples = load_jsonl(INPUT_JSONL, limit=LIMIT)

    done_keys = load_done_keys(OUTPUT_JSONL) if SKIP_ALREADY_DONE else set()

    mode = "a" if SKIP_ALREADY_DONE else "w"
    out_f = open(OUTPUT_JSONL, mode, encoding="utf-8")

    processed = 0
    skipped = 0
    failed = 0

    try:
        for idx, sample in enumerate(samples, start=1):
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
                    "signal_priority": sample.get("signal_priority"),
                    "summary_context": sample.get("summary_context", {}),
                    "label_result": label_result,
                }

                out_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                out_f.flush()

                processed += 1
                print(
                    f"[OK] {idx}/{len(samples)} row_id={row_id}, column={column}, "
                    f"label={label_result['label']}, conf={label_result['confidence']:.3f}, "
                    f"type={label_result['error_type']}"
                )

            except Exception as e:
                failed += 1
                print(f"[ERROR] {idx}/{len(samples)} row_id={row_id}, column={column}, err={e}")
                if PRINT_TRACEBACK_ON_ERROR:
                    traceback.print_exc()

            time.sleep(SLEEP_BETWEEN_REQUESTS_SEC)

    finally:
        out_f.close()

    export_csv_from_jsonl(OUTPUT_JSONL, OUTPUT_CSV)

    print("\n===== 完成 =====")
    print(f"输入样本数: {len(samples)}")
    print(f"新处理样本数: {processed}")
    print(f"跳过已完成样本数: {skipped}")
    print(f"失败样本数: {failed}")
    print(f"JSONL 输出: {OUTPUT_JSONL}")
    print(f"CSV 输出: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

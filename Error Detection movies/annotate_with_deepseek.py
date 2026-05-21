import os
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from openai import OpenAI


# ============================================================
# 1. 用户配置区
# ============================================================

INPUT_JSONL = "candidate_sampled_with_context_movies.jsonl"

OUTPUT_JSONL = "candidate_sampled_labeled_movies.jsonl"
OUTPUT_CSV = "candidate_sampled_labeled_movies.csv"

# 并行时先写临时 JSONL，最后统一排序生成正式 JSONL / CSV
OUTPUT_JSONL_TMP = "candidate_sampled_labeled_movies.tmp.jsonl"

MODEL = "deepseek-chat"

# 不建议在代码里硬编码 API key。运行前执行：
# export DEEPSEEK_API_KEY="你的key"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-fd93f5216ae0488f8bf32d9e2a14a14a")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

TEMPERATURE = 0.0
MAX_TOKENS = 1400

SKIP_ALREADY_DONE = True

# 并行线程数。建议先用 4 或 6；如果 API 限流/余额不足，就调小。
MAX_WORKERS = 6

# 每个任务开始前轻微错峰，减少瞬时请求尖峰
PER_TASK_STAGGER_SEC = 0.05

MAX_RETRIES = 5
RETRY_BACKOFF_SEC = 2.0

# None 表示全量；调试时可以设为 20
LIMIT = None


# ============================================================
# 2. 系统提示词
# ============================================================

SYSTEM_PROMPT = """
You are an expert data-cleaning annotation assistant for movie metadata tables.

The user will provide a candidate spreadsheet cell and its structured context.
Your task is to decide whether the candidate cell is erroneous.

You must return valid JSON only.

Required JSON schema:
{
  "is_error": true,
  "label": "error",
  "confidence": 0.97,
  "error_type": "semantic_conflict",
  "reason_short": "A concise reason.",
  "reason_detailed": "A slightly more detailed explanation using the evidence.",
  "evidence_used": ["movie_metadata_consistency", "neighbor_majority", "column_context"],
  "suggested_correct_value": "1960",
  "needs_human_review": false
}

Rules:
1. label must be either "error" or "correct".
2. is_error must be true if label == "error", otherwise false.
3. confidence must be a float between 0 and 1.
4. error_type must be one of:
   ["spelling_variant", "semantic_conflict", "pattern_error", "domain_error",
    "missing_value_error", "format_error", "list_membership_error",
    "rating_count_error", "movie_year_error", "likely_correct", "unclear"].
5. evidence_used should be a list selected from:
   ["fd_conflict", "neighbor_majority", "column_frequency", "pattern",
    "row_context", "column_context", "bayesian_signal",
    "movie_metadata_consistency", "year_release_consistency",
    "duration_format", "rating_range", "count_format",
    "actors_cast_overlap", "list_format", "mojibake", "other"].
6. suggested_correct_value can be null if the cell seems correct or if uncertain.
7. needs_human_review should be true when the evidence is conflicting or uncertain.
8. Output must be json only, with no markdown fences.
9. Use the provided evidence conservatively. Prefer strong structural evidence, movie metadata consistency, and neighbor consensus over weak frequency clues.
10. For movie metadata:
    - Year should usually be a valid movie year.
    - Release Date should usually contain a parseable year/date and should be broadly consistent with Year.
    - Duration should usually be a plausible minute duration such as "109 min".
    - RatingValue should be numeric in the 0-10 range.
    - RatingCount should be a non-negative integer-like count.
    - ReviewCount should usually contain user and/or critic review counts.
    - Language, Country, Genre, Creator, Actors, and Cast are often comma-separated lists.
    - Actors should usually overlap with Cast when both are present.
    - Mojibake or replacement characters usually indicate an error.
11. Missing values can be errors when the field is expected to co-occur with related fields.
12. Many movie titles, names, people lists, countries, languages, and descriptions are naturally long-tailed.
    Do not mark a value as an error only because it is rare.
13. When evidence is weak and the current value still looks plausible, prefer "correct" or "unclear".
"""


# ============================================================
# 3. Prompt 构造
# ============================================================

def build_summary_prefix(sample_obj: Dict[str, Any]) -> str:
    summary_ctx = sample_obj.get("summary_context", {})
    if not summary_ctx:
        return ""

    lines = []
    lines.append("Summary signals:")
    for k in [
        "semantic_type", "detected_type", "violation_count", "conflict_score",
        "value_frequency", "value_frequency_rank", "neighbor_majority_value",
        "neighbor_majority_ratio", "prior_error_probability",
        "posterior_error_probability", "main_rule_type", "main_usage_role",
        "strong_rule_count", "candidate_generation_rule_count", "fd_like_count",
        "context_rule_count", "global_rule_count", "rare_value_count",
        "typo_rule_count", "pattern_rule_count", "schema_rule_count",
        "movie_typed_rule_count", "movie_consistency_rule_count",
    ]:
        lines.append(f"- {k}: {summary_ctx.get(k)}")

    lines.append("")
    lines.append("Parsed movie-cell signals:")
    for k in [
        "parsed_year", "parsed_release_year", "parsed_release_country",
        "duration_minutes", "rating_value_float", "count_value_int",
        "review_total", "list_token_count", "list_token_count_bucket",
        "text_length_bucket", "has_mojibake_or_control_chars",
    ]:
        lines.append(f"- {k}: {summary_ctx.get(k)}")

    lines.append("")
    lines.append("Row-level movie metadata signals:")
    for k in [
        "movie_row_year", "movie_row_release_year", "movie_row_release_country",
        "movie_row_release_year_gap", "movie_row_duration_minutes",
        "movie_row_rating_value_float", "movie_row_rating_count_int",
        "movie_row_review_total", "movie_row_review_to_rating_ratio",
        "movie_row_actors_cast_overlap",
    ]:
        lines.append(f"- {k}: {summary_ctx.get(k)}")

    lines.append("")
    lines.append("Buckets:")
    for k in [
        "pattern_bucket", "rarity_bucket", "neighbor_bucket", "movie_rule_bucket",
        "movie_field_bucket", "year_bucket", "release_year_gap_bucket",
        "duration_bucket", "rating_bucket", "count_bucket", "review_ratio_bucket",
        "actors_cast_overlap_bucket", "list_bucket", "text_bucket",
    ]:
        lines.append(f"- {k}: {summary_ctx.get(k)}")

    return "\n".join(lines)


def build_user_prompt(sample_obj: Dict[str, Any]) -> str:
    detail_ctx = sample_obj.get("detail_context", {})
    llm_text = detail_ctx.get("llm_context_text")
    summary_prefix = build_summary_prefix(sample_obj)

    if llm_text:
        return (
            "Please analyze the following candidate movie metadata cell context and output json only.\n\n"
            + summary_prefix
            + "\n\nDetailed context:\n"
            + llm_text
        )

    return (
        "Please analyze the following candidate movie metadata cell context and output json only.\n\n"
        + summary_prefix
        + "\n\nFull sample json:\n"
        + json.dumps(sample_obj, ensure_ascii=False, indent=2)
    )


# ============================================================
# 4. API / JSON 工具
# ============================================================

def create_client() -> OpenAI:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError(
            "DEEPSEEK_API_KEY 未设置。请先设置环境变量：export DEEPSEEK_API_KEY='你的key'"
        )
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


def read_jsonl_safely(path: str) -> List[Dict[str, Any]]:
    items = []
    if not os.path.exists(path):
        return items

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[WARN] 跳过无法解析的 JSONL 行: {path}:{line_no}")
    return items


def make_key(obj: Dict[str, Any]) -> Tuple[int, str]:
    return (int(obj["row_id"]), str(obj["column"]))


def load_done_keys(*paths: str) -> set:
    done = set()
    for path in paths:
        for obj in read_jsonl_safely(path):
            try:
                done.add(make_key(obj))
            except Exception:
                continue
    return done


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

    allowed_error_types = {
        "spelling_variant",
        "semantic_conflict",
        "pattern_error",
        "domain_error",
        "missing_value_error",
        "format_error",
        "list_membership_error",
        "rating_count_error",
        "movie_year_error",
        "likely_correct",
        "unclear",
    }
    error_type = str(result.get("error_type", "unclear")).strip()
    if error_type not in allowed_error_types:
        error_type = "unclear"

    evidence_used = result.get("evidence_used", [])
    if not isinstance(evidence_used, list):
        evidence_used = ["other"]

    allowed_evidence = {
        "fd_conflict",
        "neighbor_majority",
        "column_frequency",
        "pattern",
        "row_context",
        "column_context",
        "bayesian_signal",
        "movie_metadata_consistency",
        "year_release_consistency",
        "duration_format",
        "rating_range",
        "count_format",
        "actors_cast_overlap",
        "list_format",
        "mojibake",
        "other",
    }
    evidence_used = [
        str(x) if str(x) in allowed_evidence else "other"
        for x in evidence_used
    ]
    if not evidence_used:
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
# 5. 并行任务
# ============================================================

def build_output_obj(sample: Dict[str, Any], label_result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "row_id": int(sample["row_id"]),
        "column": str(sample["column"]),
        "value": sample.get("value", None),
        "bucket_id": sample.get("bucket_id"),
        "cluster_id": sample.get("cluster_id"),
        "sample_role": sample.get("sample_role"),
        "dist_to_center": sample.get("dist_to_center"),
        "summary_context": sample.get("summary_context", {}),
        "label_result": label_result,
    }


def process_one_sample(sample: Dict[str, Any], worker_id: int = 0) -> Dict[str, Any]:
    # 每个线程单独创建 client，避免共享连接对象带来的线程安全问题。
    client = create_client()

    if PER_TASK_STAGGER_SEC > 0:
        time.sleep(PER_TASK_STAGGER_SEC * (worker_id % max(1, MAX_WORKERS)))

    user_prompt = build_user_prompt(sample)
    label_result = call_deepseek_with_retry(client, user_prompt)
    return build_output_obj(sample, label_result)


# ============================================================
# 6. 排序与 CSV 展平
# ============================================================

def sort_output_objs(objs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        objs,
        key=lambda x: (
            int(x.get("row_id", 10**18)),
            str(x.get("column", "")),
            str(x.get("bucket_id", "")),
            int(x.get("cluster_id", -1)) if x.get("cluster_id") is not None else -1,
        )
    )


def write_sorted_jsonl(objs: List[Dict[str, Any]], path: str):
    objs = sort_output_objs(objs)
    with open(path, "w", encoding="utf-8") as f:
        for obj in objs:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def build_flat_rows_from_objs(objs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    final_rows = []

    for obj in sort_output_objs(objs):
        lr = obj["label_result"]
        summary_ctx = obj.get("summary_context", {})

        row = {
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
        }

        # 保留原 summary_context 的所有字段，避免后续训练/传播脚本缺字段。
        for k, v in summary_ctx.items():
            if k not in row:
                row[k] = v

        final_rows.append(row)

    return final_rows


def rebuild_final_outputs():
    """
    合并 tmp 和正式输出中的历史结果，去重后排序，生成最终 JSONL 和 CSV。
    """
    all_objs = []

    for path in [OUTPUT_JSONL, OUTPUT_JSONL_TMP]:
        all_objs.extend(read_jsonl_safely(path))

    dedup = {}
    for obj in all_objs:
        try:
            dedup[make_key(obj)] = obj
        except Exception:
            continue

    final_objs = sort_output_objs(list(dedup.values()))
    write_sorted_jsonl(final_objs, OUTPUT_JSONL)

    final_rows = build_flat_rows_from_objs(final_objs)
    if final_rows:
        pd.DataFrame(final_rows).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    return final_objs


# ============================================================
# 7. 主流程
# ============================================================

def main():
    # 先检查 API key，避免跑到一半才失败。
    _ = create_client()

    samples = load_jsonl(INPUT_JSONL, limit=LIMIT)
    done_keys = load_done_keys(OUTPUT_JSONL, OUTPUT_JSONL_TMP) if SKIP_ALREADY_DONE else set()

    pending_samples = []
    for sample in samples:
        key = (int(sample["row_id"]), str(sample["column"]))
        if key not in done_keys:
            pending_samples.append(sample)

    print("===== 并行标注配置 =====")
    print(f"输入样本数: {len(samples)}")
    print(f"已完成样本数: {len(done_keys)}")
    print(f"待处理样本数: {len(pending_samples)}")
    print(f"MAX_WORKERS: {MAX_WORKERS}")
    print(f"临时 JSONL 输出: {OUTPUT_JSONL_TMP}")
    print(f"最终 JSONL 输出: {OUTPUT_JSONL}")
    print(f"最终 CSV 输出: {OUTPUT_CSV}")

    if not pending_samples:
        final_objs = rebuild_final_outputs()
        print("\n===== 无需新处理，已重建排序输出 =====")
        print(f"最终样本数: {len(final_objs)}")
        return

    write_lock = threading.Lock()
    processed = 0
    failed = 0

    with open(OUTPUT_JSONL_TMP, "a", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_sample = {}

            for idx, sample in enumerate(pending_samples):
                future = executor.submit(process_one_sample, sample, idx)
                future_to_sample[future] = sample

            for future in as_completed(future_to_sample):
                sample = future_to_sample[future]
                row_id = int(sample["row_id"])
                column = str(sample["column"])

                try:
                    out_obj = future.result()

                    with write_lock:
                        out_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                        out_f.flush()

                    processed += 1
                    lr = out_obj["label_result"]
                    print(
                        f"[OK] row_id={row_id}, column={column}, "
                        f"label={lr['label']}, conf={lr['confidence']:.3f}, "
                        f"processed={processed}/{len(pending_samples)}"
                    )

                except Exception as e:
                    failed += 1
                    print(f"[ERROR] row_id={row_id}, column={column}, err={e}")

    final_objs = rebuild_final_outputs()

    print("\n===== 完成 =====")
    print(f"新处理样本数: {processed}")
    print(f"失败样本数: {failed}")
    print(f"最终样本数: {len(final_objs)}")
    print(f"JSONL 输出: {OUTPUT_JSONL}")
    print(f"CSV 输出: {OUTPUT_CSV}")
    print(f"临时 JSONL 输出: {OUTPUT_JSONL_TMP}")


if __name__ == "__main__":
    main()

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
OUTPUT_JSONL_TMP = "candidate_sampled_labeled_movies.tmp.jsonl"

MODEL = "deepseek-chat"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-fd93f5216ae0488f8bf32d9e2a14a14a")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

TEMPERATURE = 0.0
MAX_TOKENS = 1000

SKIP_ALREADY_DONE = True
MAX_WORKERS = 4
PER_TASK_STAGGER_SEC = 0.05
MAX_RETRIES = 5
RETRY_BACKOFF_SEC = 2.0

# 额外硬上限：即使采样文件异常变成几万条，也不会继续烧钱。
# 正式跑建议 1000；先试流程可以改成 100 或 200。
MAX_LABELS_THIS_RUN = 1000
LIMIT = None

# 如果发现输入采样数超过这个阈值，直接拒绝运行，提醒你先重采样。
# 这是为了避免再次把全候选集交给 LLM。
ABORT_IF_INPUT_SAMPLES_GT = 3000

# 是否使用较短 prompt。True 会省 token。
USE_COMPACT_PROMPT = True


# ============================================================
# 2. 系统提示词
# ============================================================

SYSTEM_PROMPT = """
You are an expert data-cleaning annotation assistant for movie metadata tables.
Decide whether the candidate cell is erroneous using the provided evidence.
Return valid JSON only.

Required JSON schema:
{
  "is_error": true,
  "label": "error",
  "confidence": 0.97,
  "error_type": "semantic_conflict",
  "reason_short": "A concise reason.",
  "reason_detailed": "A short explanation using evidence.",
  "evidence_used": ["movie_metadata_consistency", "neighbor_majority"],
  "suggested_correct_value": "1960",
  "needs_human_review": false
}

Rules:
1. label must be "error" or "correct".
2. is_error must match label.
3. confidence must be between 0 and 1.
4. error_type must be one of:
   ["spelling_variant", "semantic_conflict", "pattern_error", "domain_error",
    "missing_value_error", "format_error", "list_membership_error",
    "rating_count_error", "movie_year_error", "likely_correct", "unclear"].
5. evidence_used should be selected from:
   ["fd_conflict", "neighbor_majority", "column_frequency", "pattern",
    "row_context", "column_context", "bayesian_signal",
    "movie_metadata_consistency", "year_release_consistency",
    "duration_format", "rating_range", "count_format",
    "actors_cast_overlap", "list_format", "mojibake", "other"].
6. Many titles, people names, countries, languages, genres, locations, and descriptions are naturally long-tailed. Do not mark a value as error only because it is rare.
7. Prefer strong structural evidence, movie metadata consistency, neighbor consensus, and explicit format violations.
8. When evidence is weak and the value is plausible, prefer "correct" or "unclear".
"""


# ============================================================
# 3. Prompt 构造
# ============================================================

def pick(obj: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    return {k: obj.get(k) for k in keys if k in obj}


def build_compact_user_prompt(sample_obj: Dict[str, Any]) -> str:
    sc = sample_obj.get("summary_context", {}) or {}
    dc = sample_obj.get("detail_context", {}) or {}

    # 从 detail 中拿少量关键上下文，避免把超长相似行/整行文本全部塞给 LLM。
    row_ctx = dc.get("row_context", {}) or {}
    col_ctx = dc.get("column_context", {}) or {}
    conflict_ctx = dc.get("conflict_context", {}) or {}
    sim_ctx = dc.get("similarity_context", {}) or {}
    bayes_ctx = dc.get("bayesian_context", {}) or {}

    movie_bundle = row_ctx.get("movie_metadata_bundle", {}) or {}

    compact = {
        "candidate": {
            "row_id": sample_obj.get("row_id"),
            "column": sample_obj.get("column"),
            "value": sample_obj.get("value"),
        },
        "summary_signals": pick(sc, [
            "semantic_type", "detected_type", "violation_count", "conflict_score",
            "value_frequency", "value_frequency_rank", "neighbor_majority_value",
            "neighbor_majority_ratio", "prior_error_probability", "posterior_error_probability",
            "main_rule_type", "main_usage_role", "strong_rule_count",
            "candidate_generation_rule_count", "fd_like_count", "context_rule_count",
            "pattern_rule_count", "schema_rule_count", "movie_typed_rule_count",
            "movie_consistency_rule_count", "parsed_year", "parsed_release_year",
            "parsed_release_country", "duration_minutes", "rating_value_float",
            "count_value_int", "review_total", "list_token_count",
            "has_mojibake_or_control_chars", "movie_row_release_year_gap",
            "movie_row_review_to_rating_ratio", "movie_row_actors_cast_overlap",
            "pattern_bucket", "rarity_bucket", "neighbor_bucket", "movie_rule_bucket",
            "movie_field_bucket", "year_bucket", "release_year_gap_bucket",
            "duration_bucket", "rating_bucket", "count_bucket", "review_ratio_bucket",
            "actors_cast_overlap_bucket", "list_bucket", "text_bucket",
        ]),
        "row_movie_metadata": pick(movie_bundle, [
            "Id", "Name", "Year", "parsed_year", "Release Date", "parsed_release_date",
            "release_year_gap", "Director", "Creator", "Actors", "Cast",
            "actors_cast_overlap", "Language", "Country", "Duration", "duration_minutes",
            "RatingValue", "rating_value_float", "RatingCount", "rating_count_int",
            "ReviewCount", "review_total", "review_to_rating_ratio", "Genre",
        ]),
        "column_context": pick(col_ctx, [
            "detected_type", "semantic_type", "missing_rate", "unique_rate",
            "value_frequency", "value_frequency_rank", "current_value_pattern",
            "dominant_pattern", "parsed_year", "parsed_release_date", "duration_minutes",
            "rating_value_float", "count_value_int", "review_count_parsed",
            "list_tokens", "list_token_count", "text_length_bucket",
            "has_mojibake_or_control_chars",
        ]),
        "conflict_context": {
            "main_rule_type": conflict_ctx.get("main_rule_type"),
            "main_usage_role": conflict_ctx.get("main_usage_role"),
            "violation_count": conflict_ctx.get("violation_count"),
            "rule_type_counter": conflict_ctx.get("rule_type_counter"),
            "expected_values_from_rules": conflict_ctx.get("expected_values_from_rules", [])[:5],
            "violated_rules_detailed": conflict_ctx.get("violated_rules_detailed", [])[:8],
        },
        "similarity_context": {
            "neighbor_majority_value": sim_ctx.get("neighbor_majority_value"),
            "neighbor_majority_ratio": sim_ctx.get("neighbor_majority_ratio"),
            "neighbor_value_distribution": sim_ctx.get("neighbor_value_distribution"),
        },
        "bayesian_context": pick(bayes_ctx, [
            "prior_error_probability", "posterior_error_probability", "bayes_evidence",
        ]),
    }

    return (
        "Analyze this candidate movie metadata cell and output JSON only.\n"
        + json.dumps(compact, ensure_ascii=False, indent=2)
    )


def build_full_user_prompt(sample_obj: Dict[str, Any]) -> str:
    detail_ctx = sample_obj.get("detail_context", {})
    llm_text = detail_ctx.get("llm_context_text")
    if llm_text:
        return "Analyze this candidate movie metadata cell and output JSON only.\n\n" + llm_text
    return "Analyze this candidate movie metadata cell and output JSON only.\n\n" + json.dumps(sample_obj, ensure_ascii=False, indent=2)


def build_user_prompt(sample_obj: Dict[str, Any]) -> str:
    if USE_COMPACT_PROMPT:
        return build_compact_user_prompt(sample_obj)
    return build_full_user_prompt(sample_obj)


# ============================================================
# 4. API / JSON 工具
# ============================================================

def create_client() -> OpenAI:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY 未设置。请先执行：export DEEPSEEK_API_KEY='你的key'")
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
        is_error = label == "error"

    try:
        confidence = float(result.get("confidence", 0.5))
    except Exception:
        confidence = 0.5
    confidence = min(max(confidence, 0.0), 1.0)

    allowed_error_types = {
        "spelling_variant", "semantic_conflict", "pattern_error", "domain_error",
        "missing_value_error", "format_error", "list_membership_error",
        "rating_count_error", "movie_year_error", "likely_correct", "unclear",
    }
    error_type = str(result.get("error_type", "unclear")).strip()
    if error_type not in allowed_error_types:
        error_type = "unclear"

    evidence_used = result.get("evidence_used", [])
    if not isinstance(evidence_used, list):
        evidence_used = ["other"]
    allowed_evidence = {
        "fd_conflict", "neighbor_majority", "column_frequency", "pattern",
        "row_context", "column_context", "bayesian_signal", "movie_metadata_consistency",
        "year_release_consistency", "duration_format", "rating_range", "count_format",
        "actors_cast_overlap", "list_format", "mojibake", "other",
    }
    evidence_used = [str(x) if str(x) in allowed_evidence else "other" for x in evidence_used]
    if not evidence_used:
        evidence_used = ["other"]

    suggested_correct_value = result.get("suggested_correct_value", None)
    if suggested_correct_value == "":
        suggested_correct_value = None

    needs_human_review = result.get("needs_human_review", False)
    if not isinstance(needs_human_review, bool):
        needs_human_review = False

    return {
        "is_error": bool(is_error),
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
    client = create_client()
    if PER_TASK_STAGGER_SEC > 0:
        time.sleep(PER_TASK_STAGGER_SEC * (worker_id % max(1, MAX_WORKERS)))
    user_prompt = build_user_prompt(sample)
    label_result = call_deepseek_with_retry(client, user_prompt)
    return build_output_obj(sample, label_result)


# ============================================================
# 6. 排序与 CSV
# ============================================================

def sort_output_objs(objs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        objs,
        key=lambda x: (
            int(x.get("row_id", 10**18)),
            str(x.get("column", "")),
            str(x.get("bucket_id", "")),
            int(x.get("cluster_id", -1)) if x.get("cluster_id") is not None else -1,
        ),
    )


def write_sorted_jsonl(objs: List[Dict[str, Any]], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for obj in sort_output_objs(objs):
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def build_flat_rows_from_objs(objs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for obj in sort_output_objs(objs):
        lr = obj["label_result"]
        sc = obj.get("summary_context", {})
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
        # 保留 summary_context 里所有扁平字段，后续训练/传播方便用。
        for k, v in sc.items():
            if k not in row:
                row[k] = v
        rows.append(row)
    return rows


def rebuild_final_outputs():
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
    rows = build_flat_rows_from_objs(final_objs)
    if rows:
        pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    return final_objs


# ============================================================
# 7. 主流程
# ============================================================

def main():
    _ = create_client()

    samples = load_jsonl(INPUT_JSONL, limit=LIMIT)

    if len(samples) > ABORT_IF_INPUT_SAMPLES_GT:
        raise RuntimeError(
            f"输入采样文件包含 {len(samples)} 条，超过 ABORT_IF_INPUT_SAMPLES_GT={ABORT_IF_INPUT_SAMPLES_GT}。\n"
            f"这说明采样仍然过多，容易接近全标注。请先重新运行 cluster_sample_movies_budgeted.py，"
            f"或调低 MAX_TOTAL_SAMPLES 后再运行。"
        )

    done_keys = load_done_keys(OUTPUT_JSONL, OUTPUT_JSONL_TMP) if SKIP_ALREADY_DONE else set()

    pending_samples = []
    for sample in samples:
        key = (int(sample["row_id"]), str(sample["column"]))
        if key not in done_keys:
            pending_samples.append(sample)

    if MAX_LABELS_THIS_RUN is not None:
        pending_samples = pending_samples[: int(MAX_LABELS_THIS_RUN)]

    print("===== 并行标注配置 =====")
    print(f"输入采样数: {len(samples)}")
    print(f"已完成样本数: {len(done_keys)}")
    print(f"本轮待处理样本数: {len(pending_samples)}")
    print(f"MAX_LABELS_THIS_RUN: {MAX_LABELS_THIS_RUN}")
    print(f"MAX_WORKERS: {MAX_WORKERS}")
    print(f"USE_COMPACT_PROMPT: {USE_COMPACT_PROMPT}")
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

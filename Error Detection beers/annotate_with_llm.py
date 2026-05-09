import os
import json
import time
import threading
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import pandas as pd
from openai import OpenAI


# ============================================================
# 1. 用户配置区
# ============================================================

INPUT_JSONL = "candidate_sampled_with_context_beers.jsonl"

OUTPUT_JSONL = "candidate_sampled_labeled_beers.jsonl"
OUTPUT_CSV = "candidate_sampled_labeled_beers.csv"

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

# 如果你想只先跑前 N 条做测试，可设成整数；None 表示全量
LIMIT = None

# ---------------- 并行参数 ----------------
# 并发线程数。建议先从 4 或 6 开始，太大可能触发限流。
MAX_WORKERS = 6

# 是否保持输出顺序与输入顺序一致
# True: 结果会按输入顺序写入 JSONL，便于复现实验和断点续跑对齐
PRESERVE_INPUT_ORDER = True

# 主线程轮询已完成 future 的时间间隔
MAIN_LOOP_SLEEP_SEC = 0.05

# 单 worker 请求前的轻微抖动，避免瞬时并发过于尖锐
WORKER_STAGGER_SEC = 0.05

# 重试
MAX_RETRIES = 5
RETRY_BACKOFF_SEC = 2.0


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
  "error_type": "pattern_error",
  "reason_short": "A concise reason.",
  "reason_detailed": "A slightly more detailed explanation using the evidence.",
  "evidence_used": ["fd_conflict", "neighbor_majority", "pattern"],
  "suggested_correct_value": "12.0 oz.",
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
10. For beer dataset fields, pay attention to brewery consistency, style consistency, numeric plausibility
    of abv/ibu/ounces, unit-format validity, and dominant context values.
11. When the value is missing but should co-occur with a paired field, consider missing_value_error.
12. For brewery-related fields, cross-check brewery_id / brewery_name / city / state consistency.
13. For beer-related fields, cross-check (beer_name, brewery_id) against style / abv / ibu / ounces.
14. When the evidence is weak and the current value still looks plausible, prefer "correct" or "unclear".
"""


# ============================================================
# 3. 全局状态（并行控制）
# ============================================================

STOP_EVENT = threading.Event()
PRINT_LOCK = threading.Lock()


# ============================================================
# 4. 工具函数
# ============================================================

def safe_print(*args, **kwargs):
    with PRINT_LOCK:
        print(*args, **kwargs)


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
    lines.append(f"- numeric_window_rule_count: {summary_ctx.get('numeric_window_rule_count')}")
    lines.append(f"- pattern_bucket: {summary_ctx.get('pattern_bucket')}")
    lines.append(f"- rarity_bucket: {summary_ctx.get('rarity_bucket')}")
    lines.append(f"- neighbor_bucket: {summary_ctx.get('neighbor_bucket')}")
    lines.append(f"- numeric_window_bucket: {summary_ctx.get('numeric_window_bucket')}")
    lines.append(f"- numeric_value_bucket: {summary_ctx.get('numeric_value_bucket')}")

    return "\n".join(lines)


def build_user_prompt(sample_obj: Dict[str, Any]) -> str:
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
        raise RuntimeError("DEEPSEEK_API_KEY 未设置。请先设置环境变量 DEEPSEEK_API_KEY。")

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


def call_deepseek_with_retry(client: OpenAI, user_prompt: str) -> Dict[str, Any]:
    last_err = None

    for attempt in range(1, MAX_RETRIES + 1):
        if STOP_EVENT.is_set():
            raise RuntimeError("Stopped because a global stop event was triggered.")

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
            err_msg = str(e)

            # 余额不足：全局停止
            if "Insufficient Balance" in err_msg or "402" in err_msg:
                STOP_EVENT.set()
                raise RuntimeError(err_msg)

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SEC * attempt)
            else:
                raise last_err

    raise last_err


def worker_process_one(index: int, sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    返回统一结果结构：
    {
        "index": index,
        "ok": bool,
        "stop": bool,
        "row_id": ...,
        "column": ...,
        "out_obj": ... or None,
        "error": ... or None
    }
    """
    if STOP_EVENT.is_set():
        return {
            "index": index,
            "ok": False,
            "stop": True,
            "row_id": int(sample["row_id"]),
            "column": str(sample["column"]),
            "out_obj": None,
            "error": "Global stop event set before processing.",
        }

    # 轻微错峰
    if WORKER_STAGGER_SEC > 0:
        time.sleep((index % max(MAX_WORKERS, 1)) * WORKER_STAGGER_SEC)

    client = create_client()

    row_id = int(sample["row_id"])
    column = str(sample["column"])
    value = sample.get("value", None)

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

        return {
            "index": index,
            "ok": True,
            "stop": False,
            "row_id": row_id,
            "column": column,
            "out_obj": out_obj,
            "error": None,
        }

    except Exception as e:
        err_msg = str(e)
        stop_now = ("Insufficient Balance" in err_msg or "402" in err_msg or STOP_EVENT.is_set())
        if stop_now:
            STOP_EVENT.set()

        return {
            "index": index,
            "ok": False,
            "stop": stop_now,
            "row_id": row_id,
            "column": column,
            "out_obj": None,
            "error": err_msg,
        }


def flush_ready_results_in_order(
    buffer: Dict[int, Dict[str, Any]],
    next_write_index: int,
    out_f,
) -> Tuple[int, int]:
    """
    将 buffer 中从 next_write_index 开始连续可写的成功结果写入文件。
    返回：
        next_write_index, newly_written_count
    """
    newly_written = 0

    while next_write_index in buffer:
        item = buffer[next_write_index]
        if item["ok"] and item["out_obj"] is not None:
            out_f.write(json.dumps(item["out_obj"], ensure_ascii=False) + "\n")
            out_f.flush()
            newly_written += 1
        # 无论成功失败，这个 index 都算“已处理完成，可推进顺序”
        del buffer[next_write_index]
        next_write_index += 1

    return next_write_index, newly_written


def build_flat_row_from_obj(obj: Dict[str, Any]) -> Dict[str, Any]:
    lr = obj["label_result"]
    summary_ctx = obj.get("summary_context", {})
    return {
        "row_id": int(obj["row_id"]),
        "column": str(obj["column"]),
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
        "numeric_window_rule_count": summary_ctx.get("numeric_window_rule_count"),

        "pattern_bucket": summary_ctx.get("pattern_bucket"),
        "rarity_bucket": summary_ctx.get("rarity_bucket"),
        "neighbor_bucket": summary_ctx.get("neighbor_bucket"),
        "numeric_window_bucket": summary_ctx.get("numeric_window_bucket"),
        "numeric_value_bucket": summary_ctx.get("numeric_value_bucket"),
    }


def sort_labeled_jsonl_and_export(output_jsonl: str, output_csv: str):
    rows = []
    objs = []

    if not os.path.exists(output_jsonl):
        return

    with open(output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            obj["row_id"] = int(obj["row_id"])
            obj["column"] = str(obj["column"])
            objs.append(obj)
            rows.append(build_flat_row_from_obj(obj))

    if not objs:
        return

    objs.sort(key=lambda x: (x["row_id"], x["column"]))
    rows.sort(key=lambda x: (x["row_id"], x["column"]))

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for obj in objs:
            f.write(json.dumps(obj, ensure_ascii=False) + "\\n")

    pd.DataFrame(rows).to_csv(output_csv, index=False, encoding="utf-8-sig")


# ============================================================
# 5. 主流程
# ============================================================

def main():
    samples = load_jsonl(INPUT_JSONL, limit=LIMIT)

    done_keys = load_done_keys(OUTPUT_JSONL) if SKIP_ALREADY_DONE else set()

    # 过滤已完成样本，但保留原始顺序
    pending_samples = []
    skipped = 0
    for sample in samples:
        key = (int(sample["row_id"]), str(sample["column"]))
        if key in done_keys:
            skipped += 1
            continue
        pending_samples.append(sample)

    if not pending_samples:
        safe_print("没有需要新处理的样本。")
        sort_labeled_jsonl_and_export(OUTPUT_JSONL, OUTPUT_CSV)
        safe_print(f"JSONL 输出: {OUTPUT_JSONL}")
        safe_print(f"CSV 输出: {OUTPUT_CSV}")
        return

    processed = 0
    failed = 0
    stop_due_to_balance = False

    out_f = open(OUTPUT_JSONL, "a" if SKIP_ALREADY_DONE else "w", encoding="utf-8")

    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {}
            next_submit_index = 0
            next_write_index = 0
            result_buffer = {}

            # 先提交一批
            initial_n = min(MAX_WORKERS, len(pending_samples))
            for _ in range(initial_n):
                sample = pending_samples[next_submit_index]
                fut = executor.submit(worker_process_one, next_submit_index, sample)
                futures[fut] = next_submit_index
                next_submit_index += 1

            while futures:
                done, _ = wait(list(futures.keys()), timeout=MAIN_LOOP_SLEEP_SEC, return_when=FIRST_COMPLETED)
                if not done:
                    continue

                for fut in done:
                    idx = futures.pop(fut)
                    res = fut.result()

                    if res["ok"]:
                        lr = res["out_obj"]["label_result"]
                        safe_print(
                            f"[OK] row_id={res['row_id']}, column={res['column']}, "
                            f"label={lr['label']}, conf={lr['confidence']:.3f}"
                        )
                        processed += 1
                    else:
                        safe_print(f"[ERROR] row_id={res['row_id']}, column={res['column']}, err={res['error']}")
                        failed += 1
                        if res["stop"]:
                            stop_due_to_balance = True

                    if PRESERVE_INPUT_ORDER:
                        result_buffer[idx] = res
                    else:
                        if res["ok"] and res["out_obj"] is not None:
                            out_f.write(json.dumps(res["out_obj"], ensure_ascii=False) + "\n")
                            out_f.flush()

                    if stop_due_to_balance:
                        STOP_EVENT.set()

                # 顺序写出
                if PRESERVE_INPUT_ORDER:
                    next_write_index, _ = flush_ready_results_in_order(result_buffer, next_write_index, out_f)

                # 继续补提交
                while (not STOP_EVENT.is_set()) and next_submit_index < len(pending_samples) and len(futures) < MAX_WORKERS:
                    sample = pending_samples[next_submit_index]
                    fut = executor.submit(worker_process_one, next_submit_index, sample)
                    futures[fut] = next_submit_index
                    next_submit_index += 1

                # 若触发全局停止，则不再提交新任务；已有任务继续收尾
                if STOP_EVENT.is_set() and not futures:
                    break

            # 最后如果按顺序写，尽量把已经完成的结果继续刷掉
            if PRESERVE_INPUT_ORDER:
                next_write_index, _ = flush_ready_results_in_order(result_buffer, next_write_index, out_f)

    finally:
        out_f.close()

    sort_labeled_jsonl_and_export(OUTPUT_JSONL, OUTPUT_CSV)

    safe_print("\n===== 完成 =====")
    safe_print(f"新处理样本数: {processed}")
    safe_print(f"失败样本数: {failed}")
    safe_print(f"跳过已完成样本数: {skipped}")
    if stop_due_to_balance:
        safe_print("程序提前停止原因: API 余额不足或触发全局停止。")
    safe_print(f"JSONL 输出: {OUTPUT_JSONL}")
    safe_print(f"CSV 输出: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

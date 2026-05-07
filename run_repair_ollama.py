#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_repair_ollama.py

用途：
- 读取 INPUT_JSONL（每行一个 evidence JSON）
- 调用服务器本地 Ollama(/api/chat) 生成修复候选 topk
- 强制用输入 one_obj["cell"] 回填 row_id / col / current（避免模型乱写导致 row_id 错）
- 输出到 OUTPUT_JSONL（jsonl）

运行示例：
  python -u run_repair_ollama.py

也可用环境变量覆盖配置：
  INPUT_JSONL=hospital_evidence1.jsonl OUTPUT_JSONL=out.jsonl OLLAMA_MODEL=qwen2.5:32b python -u run_repair_ollama.py
"""

import os
import json
import time
from typing import Any, Dict, Iterable, Optional, Tuple

import requests


# =========================
# 0) 配置区（优先用环境变量覆盖）
# =========================
INPUT_JSONL = os.getenv("INPUT_JSONL", "hospital_evidence1.jsonl")
OUTPUT_JSONL = os.getenv("OUTPUT_JSONL", "hospital_repairs_top3_qwen.jsonl")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:32b")

TOP_K = int(os.getenv("TOP_K", "3"))

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
RETRY_BACKOFF_SEC = float(os.getenv("RETRY_BACKOFF_SEC", "2.0"))

TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
NUM_PREDICT = int(os.getenv("NUM_PREDICT", "900"))  # 输出长度上限（类似 max_tokens）
TIMEOUT_SEC = int(os.getenv("TIMEOUT_SEC", "300"))

DEBUG_DIR = os.getenv("DEBUG_DIR", "debug_bad_outputs")


# =========================
# 1) Prompt
# =========================
SYSTEM_PROMPT = r"""
You are a data cleaning assistant. You repair one cell value at a time.

Hard rules:
- Use ONLY the provided JSON context as evidence. Do NOT invent facts or external knowledge.
- Return STRICT JSON only (no markdown, no extra keys, no analysis text).
- The output MUST match the required JSON schema exactly.

Task:
Given a target cell, propose TOP_K candidate repaired values (strings).
Each candidate must include evidence: which rule(s) and which row_id examples support it.

Ranking:
- If conflict_context contains other_rows_same_lhs_examples or other_rows_satisfying_cfd_examples,
  treat their RHS values as strongest evidence.
- Prefer candidates that appear in evidence rows.
- If multiple RHS values appear, rank by frequency, then by compatibility with row_context/column_context formats.
- If no evidence rows exist, you may fall back to column top_values and similar_rows_topk, but confidence must be low.

IMPORTANT:
- You MUST output the JSON schema only.
""".strip()


def build_user_prompt(one_obj: Dict[str, Any], top_k: int) -> str:
    return (
        f"TOP_K={top_k}.\n"
        f"Repair this cell based on the following JSON context:\n"
        f"{json.dumps(one_obj, ensure_ascii=False)}"
    )


# =========================
# 2) 期望输出 JSON Schema（用于 Ollama format）
#    说明：我们仍然让 schema 要求 row_id/col/current 存在，
#    但最终会强制用输入 cell 回填这三项，避免模型乱写。
# =========================
def get_output_schema(top_k: int) -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["row_id", "col", "current", "should_change", "topk", "chosen", "confidence", "warnings"],
        "properties": {
            "row_id": {"type": "integer"},
            "col": {"type": "string"},
            "current": {"type": "string"},
            "should_change": {"type": "boolean"},
            "topk": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["value", "score", "evidence"],
                    "properties": {
                        "value": {"type": "string"},
                        "score": {"type": "number"},
                        "evidence": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["type", "rule_id", "rule_text", "supporting_rows", "note"],
                                "properties": {
                                    "type": {"type": "string", "enum": ["FD", "CFD", "COLUMN", "SIMILAR_ROW"]},
                                    "rule_id": {"type": ["string", "null"]},
                                    "rule_text": {"type": ["string", "null"]},
                                    "supporting_rows": {"type": "array", "items": {"type": "integer"}},
                                    "note": {"type": "string"},
                                },
                            },
                        },
                    },
                },
            },
            "chosen": {"type": "string"},
            "confidence": {"type": "number"},
            "warnings": {"type": "array", "items": {"type": "string"}},
        },
    }


# =========================
# 3) IO
# =========================
def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def save_debug(basename: str, content: str) -> str:
    os.makedirs(DEBUG_DIR, exist_ok=True)
    p = os.path.join(DEBUG_DIR, basename)
    with open(p, "w", encoding="utf-8") as f:
        f.write(content)
    return p


# =========================
# 4) Ollama API
# =========================
def ollama_chat(messages, model: str, fmt: Optional[Any]) -> Tuple[str, Dict[str, Any]]:
    """
    调用 Ollama /api/chat
    - stream=False 一次性返回，便于解析
    - fmt:
        - None: 不强制格式
        - "json": 强制返回合法 JSON
        - dict(JSON schema): 强制返回符合 schema 的 JSON
    """
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": NUM_PREDICT,
        },
    }
    if fmt is not None:
        payload["format"] = fmt

    resp = requests.post(url, json=payload, timeout=TIMEOUT_SEC)
    resp.raise_for_status()
    data = resp.json()
    text = (data.get("message", {}) or {}).get("content", "")
    return (text or "").strip(), data


# =========================
# 5) 校验/解析 + 强制回填（修复 row_id 错的关键）
# =========================
REQUIRED_KEYS = {"row_id", "col", "current", "should_change", "topk", "chosen", "confidence", "warnings"}


def validate_minimal_schema(obj: Dict[str, Any]) -> None:
    if not isinstance(obj, dict):
        raise ValueError("Output is not a JSON object.")

    missing = REQUIRED_KEYS - set(obj.keys())
    if missing:
        raise ValueError(f"Missing required keys: {sorted(list(missing))}")

    if not isinstance(obj["topk"], list) or len(obj["topk"]) == 0:
        raise ValueError("topk must be a non-empty list.")

    for item in obj["topk"]:
        if not isinstance(item, dict):
            raise ValueError("Each topk item must be an object.")
        for k in ("value", "score", "evidence"):
            if k not in item:
                raise ValueError(f"topk item missing key: {k}")
        if not isinstance(item["evidence"], list) or len(item["evidence"]) == 0:
            raise ValueError("topk[i].evidence must be a non-empty list.")


def extract_json(text: str) -> Dict[str, Any]:
    if not text:
        raise ValueError("Empty model output.")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        l = text.find("{")
        r = text.rfind("}")
        if l != -1 and r != -1 and r > l:
            return json.loads(text[l : r + 1])
        raise ValueError("Cannot extract JSON object from model output.")


def force_fill_from_input(out: Dict[str, Any], one_obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    强制用输入 one_obj['cell'] 覆盖 row_id/col/current（避免模型乱写导致 row_id 错）
    并清洗 supporting_rows 确保都是 int。
    """
    cell = one_obj.get("cell", {})
    if not isinstance(cell, dict):
        cell = {}

    in_row_id = cell.get("row_id", None)
    in_col = cell.get("col", None)
    in_current = cell.get("current", "")

    # 覆盖三大字段
    if isinstance(in_row_id, int):
        out["row_id"] = in_row_id
    if isinstance(in_col, str) and in_col:
        out["col"] = in_col
    if isinstance(in_current, str):
        out["current"] = in_current

    # 清洗 evidence.supporting_rows
    topk = out.get("topk", [])
    if isinstance(topk, list):
        for item in topk:
            if not isinstance(item, dict):
                continue
            ev = item.get("evidence", [])
            if not isinstance(ev, list):
                continue
            for e in ev:
                if not isinstance(e, dict):
                    continue
                sr = e.get("supporting_rows", [])
                if isinstance(sr, list):
                    cleaned = []
                    for x in sr:
                        if isinstance(x, int):
                            cleaned.append(x)
                        elif isinstance(x, str) and x.isdigit():
                            cleaned.append(int(x))
                    e["supporting_rows"] = cleaned

    return out


# =========================
# 6) 核心：调用 + 重试
# =========================
def call_llm_with_retries(one_obj: Dict[str, Any]) -> Dict[str, Any]:
    user_prompt = build_user_prompt(one_obj, TOP_K)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    last_err: Optional[Exception] = None
    last_raw: str = ""
    last_meta: Dict[str, Any] = {}

    schema = get_output_schema(TOP_K)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw, meta = ollama_chat(messages=messages, model=MODEL, fmt=schema)
            last_raw, last_meta = raw, meta

            obj = extract_json(raw)
            validate_minimal_schema(obj)
            obj = force_fill_from_input(obj, one_obj)  # <<< 关键：回填 row_id/col/current
            return obj

        except Exception as e:
            last_err = e

            # 降级：至少保证合法 JSON，再自己校验字段
            try:
                raw2, meta2 = ollama_chat(messages=messages, model=MODEL, fmt="json")
                last_raw, last_meta = raw2, meta2
                obj2 = extract_json(raw2)
                validate_minimal_schema(obj2)
                obj2 = force_fill_from_input(obj2, one_obj)  # <<< 同样回填
                return obj2
            except Exception as e2:
                last_err = e2

            time.sleep(RETRY_BACKOFF_SEC * attempt)

    # 全失败：落盘 debug
    ts = int(time.time())
    bad_path = save_debug(
        f"bad_{ts}.txt",
        "==== LAST_ERROR ====\n"
        + repr(last_err)
        + "\n\n==== LAST_RAW_TEXT ====\n"
        + (last_raw or "<EMPTY>")
        + "\n\n==== LAST_META_JSON ====\n"
        + json.dumps(last_meta, ensure_ascii=False, indent=2),
    )

    # 失败时也给一个“单层、可用”的 fallback（你说不要两层，这里就 1 层对象）
    # 但注意：fallback 会让 value/chosen 为空，主要用于不中断跑完全量。
    cell = one_obj.get("cell", {}) if isinstance(one_obj.get("cell"), dict) else {}
    fallback = {
        "row_id": cell.get("row_id", None),
        "col": cell.get("col", None),
        "current": cell.get("current", ""),
        "should_change": True,
        "topk": [
            {
                "value": "",
                "score": 0.0,
                "evidence": [
                    {
                        "type": "COLUMN",
                        "rule_id": None,
                        "rule_text": None,
                        "supporting_rows": [],
                        "note": "Fallback: model output could not be coerced to schema",
                    }
                ],
            }
        ],
        "chosen": "",
        "confidence": 0.05,
        "warnings": [f"llm_failed_fallback_saved:{bad_path}"],
    }
    # 如果你希望失败就直接抛错中断，把下面 return fallback 改成 raise RuntimeError(...)
    return fallback


# =========================
# 7) 主流程
# =========================
def main() -> None:
    if not os.path.exists(INPUT_JSONL):
        raise FileNotFoundError(f"INPUT_JSONL not found: {INPUT_JSONL}")

    # ===== 计时开始 =====
    t0 = time.perf_counter()

    # 清空旧输出
    if os.path.exists(OUTPUT_JSONL):
        os.remove(OUTPUT_JSONL)

    total = 0

    for i, one_obj in enumerate(read_jsonl(INPUT_JSONL), start=1):
        out = call_llm_with_retries(one_obj)

        # meta
        out["_meta"] = {
            "input_index": i,
            "model": MODEL,
            "backend": "ollama",
            "base_url": OLLAMA_BASE_URL,
        }

        append_jsonl(OUTPUT_JSONL, out)

        rid = out.get("row_id", None)
        col = out.get("col", None)
        print(f"[OK] {i} row_id={rid} col={col} -> {OUTPUT_JSONL}", flush=True)

        total += 1

    # ===== 计时结束 =====
    t1 = time.perf_counter()
    total_sec = t1 - t0
    avg_sec = total_sec / total if total > 0 else 0.0

    print("\n========== SUMMARY ==========", flush=True)
    print(f"Total items processed : {total}", flush=True)
    print(f"Total time            : {total_sec:.2f} seconds", flush=True)
    print(f"Avg time per item     : {avg_sec:.2f} seconds", flush=True)
    print("Done.", flush=True)

    if __name__ == "__main__": main()

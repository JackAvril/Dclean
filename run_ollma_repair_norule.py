#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_repair_ollama_context_json.py

用途：
- 读取一个“上下文 JSON 文件”（包含 schema/meta/summary/cells 列表）
- 对 cells 里的每个 cell-item 调用本地 Ollama (/api/chat)，生成 topk=3 修复候选 + evidence
- 强制用输入 item["cell"] 回填 row_id/col/current，避免模型乱写
- 输出到 OUTPUT_JSONL（jsonl），一行对应一个 cell 的修复建议

运行示例：
  python -u run_repair_ollama_context_json.py

也可用环境变量覆盖：
  INPUT_JSON=context.json OUTPUT_JSONL=repairs.jsonl TOP_K=3 OLLAMA_MODEL=qwen2.5:32b python -u run_repair_ollama_context_json.py
"""

import os
import json
import time
from typing import Any, Dict, List, Optional, Tuple

import requests


# =========================
# 0) 配置区（优先用环境变量覆盖）
# =========================
INPUT_JSON = os.getenv("INPUT_JSON", "context_model.json")     # 你的“单个 JSON 上下文文件”
OUTPUT_JSONL = os.getenv("OUTPUT_JSONL", "hospital_repairs_top3_qwen_norule.jsonl")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:32b")

TOP_K = int(os.getenv("TOP_K", "3"))

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
RETRY_BACKOFF_SEC = float(os.getenv("RETRY_BACKOFF_SEC", "2.0"))

TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
NUM_PREDICT = int(os.getenv("NUM_PREDICT", "900"))
TIMEOUT_SEC = int(os.getenv("TIMEOUT_SEC", "300"))

DEBUG_DIR = os.getenv("DEBUG_DIR", "debug_bad_outputs")


# =========================
# 1) Prompt（适配“无 FD/CFD、只有 conflict_rows_examples”）
# =========================
SYSTEM_PROMPT = r"""
You are a data cleaning assistant. You repair one cell value at a time.

Hard rules:
- Use ONLY the provided JSON context as evidence. Do NOT invent facts or external knowledge.
- Return STRICT JSON only (no markdown, no analysis, no extra keys).
- The output MUST match the required JSON schema exactly.

Task:
Given a target cell item (including column_context / row_context / conflict_context / bayes),
propose TOP_K candidate repaired values (strings).
Each candidate must include evidence: which rows/examples support it.

Evidence priority & ranking:
1) If conflict_context.conflict_rows_examples exists:
   - Treat their target_value as the strongest evidence (vote by frequency; break ties by higher similarity).
2) Else if row_context.similar_rows_topk exists:
   - Use their target_value as secondary evidence.
3) Else fall back to column_context.top_values (column frequency).
4) Always ensure candidates look compatible with column type/format.

Scoring:
- Provide a score in [0, 1]. You may map vote ratio + similarity intuitively.
- chosen must be one of the proposed values (typically the top-1).
- confidence is overall confidence in [0, 1]; lower if evidence is weak.

Output must be STRICT JSON only.
""".strip()


def build_user_prompt(cell_item: Dict[str, Any], top_k: int) -> str:
    # 注意：这里传入的是 cells[i] 的整个 item（含 cell/contexts/bayes）
    return (
        f"TOP_K={top_k}.\n"
        f"Repair this cell based on the following JSON context (one cell item):\n"
        f"{json.dumps(cell_item, ensure_ascii=False)}"
    )


# =========================
# 2) 输出 JSON Schema（用于 Ollama format）
#    重点：支持 CONFLICT_ROW；rule_id/rule_text 允许 null
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
                "minItems": top_k,
                "maxItems": top_k,
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
                                    "type": {
                                        "type": "string",
                                        "enum": ["FD", "CFD", "COLUMN", "SIMILAR_ROW", "CONFLICT_ROW"],
                                    },
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
def load_context_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
def ollama_chat(messages: List[Dict[str, str]], model: str, fmt: Optional[Any]) -> Tuple[str, Dict[str, Any]]:
    """
    调用 Ollama /api/chat
    - stream=False 一次性返回
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
# 5) 校验/解析 + 强制回填
# =========================
REQUIRED_KEYS = {"row_id", "col", "current", "should_change", "topk", "chosen", "confidence", "warnings"}


def validate_minimal_schema(obj: Dict[str, Any], top_k: int) -> None:
    if not isinstance(obj, dict):
        raise ValueError("Output is not a JSON object.")

    missing = REQUIRED_KEYS - set(obj.keys())
    if missing:
        raise ValueError(f"Missing required keys: {sorted(list(missing))}")

    if not isinstance(obj["topk"], list) or len(obj["topk"]) < 1:
        raise ValueError("topk must be a non-empty list.")

    # 强制 topk 长度至少为 top_k（模型偶尔少给）
    if len(obj["topk"]) < top_k:
        raise ValueError(f"topk length < {top_k}")

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


def force_fill_from_input(out: Dict[str, Any], cell_item: Dict[str, Any], top_k: int) -> Dict[str, Any]:
    """
    强制用输入 cell_item['cell'] 覆盖 row_id/col/current
    并清洗 supporting_rows，最后强制 topk 截断为 top_k
    """
    cell = cell_item.get("cell", {})
    if not isinstance(cell, dict):
        cell = {}

    in_row_id = cell.get("row_id", None)
    in_col = cell.get("col", None)
    in_current = cell.get("current", "")

    if isinstance(in_row_id, int):
        out["row_id"] = in_row_id
    if isinstance(in_col, str) and in_col:
        out["col"] = in_col
    if isinstance(in_current, str):
        out["current"] = in_current

    # supporting_rows: str->int 清洗
    topk_list = out.get("topk", [])
    if isinstance(topk_list, list):
        for item in topk_list:
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

    # 强制 topk 只保留 top_k
    if isinstance(out.get("topk"), list):
        out["topk"] = out["topk"][:top_k]

    # chosen 必须在 topk 里；否则自动用 top1
    values = [it.get("value", "") for it in out.get("topk", []) if isinstance(it, dict)]
    if values:
        if out.get("chosen") not in values:
            out["chosen"] = values[0]

    return out


# =========================
# 6) 核心：调用 + 重试
# =========================
def call_llm_with_retries(cell_item: Dict[str, Any]) -> Dict[str, Any]:
    user_prompt = build_user_prompt(cell_item, TOP_K)
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
            validate_minimal_schema(obj, TOP_K)
            obj = force_fill_from_input(obj, cell_item, TOP_K)
            return obj

        except Exception as e:
            last_err = e

            # 降级：至少拿到合法 JSON
            try:
                raw2, meta2 = ollama_chat(messages=messages, model=MODEL, fmt="json")
                last_raw, last_meta = raw2, meta2
                obj2 = extract_json(raw2)
                validate_minimal_schema(obj2, TOP_K)
                obj2 = force_fill_from_input(obj2, cell_item, TOP_K)
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

    # fallback（不中断跑完全量）
    cell = cell_item.get("cell", {}) if isinstance(cell_item.get("cell"), dict) else {}
    fallback = {
        "row_id": cell.get("row_id", -1) if isinstance(cell.get("row_id", -1), int) else -1,
        "col": cell.get("col", "") if isinstance(cell.get("col", ""), str) else "",
        "current": cell.get("current", "") if isinstance(cell.get("current", ""), str) else "",
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
        ] + [
            {
                "value": "",
                "score": 0.0,
                "evidence": [
                    {
                        "type": "COLUMN",
                        "rule_id": None,
                        "rule_text": None,
                        "supporting_rows": [],
                        "note": "Fallback padding",
                    }
                ],
            }
            for _ in range(max(0, TOP_K - 1))
        ],
        "chosen": "",
        "confidence": 0.05,
        "warnings": [f"llm_failed_fallback_saved:{bad_path}"],
    }
    return fallback


# =========================
# 7) 主流程：读取单个 JSON，遍历 cells，输出 jsonl
# =========================
def main() -> None:
    if not os.path.exists(INPUT_JSON):
        raise FileNotFoundError(f"INPUT_JSON not found: {INPUT_JSON}")

    context = load_context_json(INPUT_JSON)
    cells = context.get("cells", [])
    if not isinstance(cells, list):
        raise ValueError("Input JSON must contain a list field: cells")

    # 清空旧输出
    if os.path.exists(OUTPUT_JSONL):
        os.remove(OUTPUT_JSONL)

    t0 = time.perf_counter()
    total = 0

    for i, cell_item in enumerate(cells, start=1):
        if not isinstance(cell_item, dict):
            continue

        out = call_llm_with_retries(cell_item)

        # 附加 meta 方便回溯（不影响你主字段）
        out["_meta"] = {
            "input_index": i,
            "model": MODEL,
            "backend": "ollama",
            "base_url": OLLAMA_BASE_URL,
            "input_file": INPUT_JSON,
        }

        append_jsonl(OUTPUT_JSONL, out)

        rid = out.get("row_id", None)
        col = out.get("col", None)
        print(f"[OK] {i}/{len(cells)} row_id={rid} col={col} -> {OUTPUT_JSONL}", flush=True)

        total += 1

    t1 = time.perf_counter()
    total_sec = t1 - t0
    avg_sec = total_sec / total if total > 0 else 0.0

    print("\n========== SUMMARY ==========", flush=True)
    print(f"Input file            : {INPUT_JSON}", flush=True)
    print(f"Output file           : {OUTPUT_JSONL}", flush=True)
    print(f"Total items processed : {total}", flush=True)
    print(f"Total time            : {total_sec:.2f} seconds", flush=True)
    print(f"Avg time per item     : {avg_sec:.2f} seconds", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()

import os
import json
import time
from typing import Any, Dict, Iterable, Optional

from openai import OpenAI


# =========================
# 0) 配置区
# =========================
INPUT_JSONL = "hospital_evidence1.jsonl"
OUTPUT_JSONL = "hospital_repairs_top3.jsonl"

# DeepSeek: OpenAI SDK compatible
# Docs: /chat/completions (DeepSeek API) :contentReference[oaicite:1]{index=1}
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")  # 也可用 deepseek-reasoner（若你开通）
TOP_K = 3

# 重试设置
MAX_RETRIES = 5
RETRY_BACKOFF_SEC = 2.0


# =========================
# 1) Prompt
# =========================
SYSTEM_PROMPT = r"""
You are a data cleaning assistant. You repair one cell value at a time.
Use ONLY the provided JSON context as evidence. Do NOT invent facts or external knowledge.
Return STRICT JSON only, no markdown.

Goal:
Given a target cell that violates FD/CFD constraints, propose TOP_K candidate repaired values (strings).
Each candidate must include evidence: which rule(s) and which row_id examples support it.

Rules:
- If conflict_context contains other_rows_same_lhs_examples, treat their RHS values as the strongest evidence.
- If conflict_context contains other_rows_satisfying_cfd_examples, treat them as the strongest evidence.
- Prefer candidates that appear in the evidence rows.
- If there are multiple different RHS values in evidence, rank by frequency, then by compatibility with row_context and column_context formats.
- If no evidence rows exist (should be rare), you may fall back to column top_values and similar_rows_topk, but mark confidence low.

Output JSON schema:
{
  "row_id": int,
  "col": string,
  "current": string,
  "should_change": bool,
  "topk": [
    {
      "value": string,
      "score": float,
      "evidence": [
        {
          "type": "FD"|"CFD"|"COLUMN"|"SIMILAR_ROW",
          "rule_id": string|null,
          "rule_text": string|null,
          "supporting_rows": [int],
          "note": string
        }
      ]
    }
  ],
  "chosen": string,
  "confidence": float,
  "warnings": [string]
}
""".strip()

def build_user_prompt(one_obj: Dict[str, Any], top_k: int) -> str:
    return f"TOP_K={top_k}.\nRepair this cell based on the following JSON context:\n{json.dumps(one_obj, ensure_ascii=False)}"


# =========================
# 2) IO 工具
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


# =========================
# 3) 调用 DeepSeek
# =========================
def call_llm(client: OpenAI, one_obj: Dict[str, Any]) -> Dict[str, Any]:
    user_prompt = build_user_prompt(one_obj, TOP_K)

    last_err: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,   # 清洗/修复任务建议低温
                max_tokens=800,    # 够输出严格JSON
            )

            text = resp.choices[0].message.content.strip()

            # 强制解析 JSON（模型偶尔会多输出字符，这里做一次容错）
            # 1) 尝试直接 loads
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                # 2) 尝试截取第一个 { 到最后一个 }
                l = text.find("{")
                r = text.rfind("}")
                if l != -1 and r != -1 and r > l:
                    return json.loads(text[l:r+1])
                raise

        except Exception as e:
            last_err = e
            sleep_s = RETRY_BACKOFF_SEC * attempt
            time.sleep(sleep_s)

    raise RuntimeError(f"LLM call failed after retries. Last error: {last_err}")


# =========================
# 4) 主流程
# =========================
def main():
    if not DEEPSEEK_API_KEY:
        raise ValueError(
            "Missing DEEPSEEK_API_KEY. "
            "Set it as an environment variable, e.g.:\n"
            "  set DEEPSEEK_API_KEY=xxxxx   (Windows CMD)\n"
            "  $env:DEEPSEEK_API_KEY='xxxxx' (PowerShell)\n"
        )

    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
    )

    # 可选：清空旧输出
    if os.path.exists(OUTPUT_JSONL):
        os.remove(OUTPUT_JSONL)

    for i, one_obj in enumerate(read_jsonl(INPUT_JSONL), start=1):
        out = call_llm(client, one_obj)

        # 额外补上 trace 信息，方便你对齐输入输出
        out["_meta"] = {
            "input_index": i,
            "model": MODEL,
        }
        append_jsonl(OUTPUT_JSONL, out)
        print(f"[OK] {i} -> wrote 1 line to {OUTPUT_JSONL}")

    print("Done.")

if __name__ == "__main__":
    main()

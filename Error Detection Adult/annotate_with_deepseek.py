import os
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from openai import OpenAI


# ============================================================
# 1. 用户配置区
# ============================================================

INPUT_JSONL = "candidate_sampled_with_context_soccer.jsonl"

OUTPUT_JSONL = "candidate_sampled_labeled_soccer.jsonl"
OUTPUT_CSV = "candidate_sampled_labeled_soccer.csv"

# DeepSeek 模型
MODEL = "deepseek-chat"

# API 配置
# 注意：不要把 API key 写死到代码里。
# 运行前执行：
# export DEEPSEEK_API_KEY="你的key"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
BASE_URL = "https://api.deepseek.com"

# 生成参数
TEMPERATURE = 0.0
MAX_TOKENS = 1200

# 断点续跑
SKIP_ALREADY_DONE = True

# 并行标注配置
# Soccer 采样量通常不大，建议先用 4；如果 API 稳定且余额充足，可改成 6 或 8。
ENABLE_PARALLEL = True
MAX_WORKERS = 4

# 调用节流：并行时每个任务结束后也会 sleep，一般 0.1~0.3 均可。
SLEEP_BETWEEN_REQUESTS_SEC = 0.2

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

# Soccer schema-aware 配置：与规则池/缩减候选/上下文/采样脚本保持一致。
SOCCER_COLUMNS = [
    "name",
    "surname",
    "birthyear",
    "birthplace",
    "position",
    "team",
    "city",
    "stadium",
    "season",
    "manager",
]

PRIMARY_TARGET_COLUMNS = {
    "birthyear",
    "birthplace",
    "position",
    "team",
    "city",
    "stadium",
    "season",
    "manager",
}

CONTEXT_ONLY_COLUMNS = {
    "name",
    "surname",
}

VALID_POSITIONS = {
    "Goalkeeper",
    "Defender",
    "Midfield",
    "Midfielder",
    "Forward",
    "Striker",
    "Winger",
}

POSITION_CANONICAL_MAP = {
    "Midfielder": "Midfield",
}

SOCCER_STRONG_RULE_TYPES = {
    "birthyear_numeric_range",
    "season_numeric_range",
    "position_domain",
    "birthyear_season_age_consistency",
}

SOCCER_FORMAT_RULE_TYPES = {
    "name_pattern",
    "surname_pattern",
    "manager_name_pattern",
    "team_name_pattern",
    "birthplace_location_pattern",
    "city_location_pattern",
    "stadium_name_pattern",
    "pattern",
    "generic_regex",
}

SOCCER_CONTEXT_RULE_TYPES = {
    "team_context_dominant",
    "team_season_manager_consistency_hint",
    "city_stadium_consistency_hint",
    "dominant_value_by_context",
    "dominant_value_by_context_pair",
}

SOCCER_FD_RULE_TYPES = {
    "functional_dependency",
    "soft_functional_dependency",
}

BIRTHYEAR_MIN = 1940
BIRTHYEAR_MAX = 2010
SEASON_MIN = 1900
SEASON_MAX = 2035
PLAYER_MIN_AGE_AT_SEASON = 15
PLAYER_MAX_AGE_AT_SEASON = 50


# ============================================================
# 2. 系统提示词：Soccer 数据集专用
# ============================================================

SYSTEM_PROMPT = """
You are an expert data-cleaning annotation assistant for a soccer player/team tabular dataset.

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
  "evidence_used": ["domain_constraint", "soccer_consistency", "row_context"],
  "suggested_correct_value": "Defender",
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
     "normalization_error",
     "likely_correct",
     "unclear"
   ].
5. evidence_used should be a list selected from:
   [
     "domain_constraint",
     "format_constraint",
     "soccer_consistency",
     "fd_conflict",
     "team_context",
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

Soccer dataset-specific guidance:
1. The columns are: name, surname, birthyear, birthplace, position, team, city, stadium, season, manager.
2. birthyear should usually be a four-digit integer within a plausible player birth-year range, roughly 1940..2010.
3. season should usually be a four-digit integer within a plausible season range, roughly 1900..2035.
4. The player's age at season is season - birthyear. It is usually plausible for a professional football player to be about 15..50.
5. position should be a known soccer playing position, e.g., Goalkeeper, Defender, Midfield, Forward. If a value is outside the known domain, that is strong evidence of error.
6. Midfielder may be a normalization variant of Midfield if the dataset canonical label is Midfield. Treat this as a possible normalization_error rather than a severe semantic error.
7. name and manager usually look like alphabetic tokens separated by underscores, for example Brad_Zhaofang or Xiaomin_Saoudi.
8. surname usually looks alphabetic, optionally with separators such as hyphen, underscore, or apostrophe.
9. team, stadium, city, and birthplace can be long-tail entity names. A rare team/city/stadium/person name is not an error by itself.
10. team-context evidence such as team+season -> stadium/city/manager is useful but not absolute. A football team may change manager, stadium, or host city depending on season or data source.
11. FD/soft-FD evidence should be treated as supporting evidence, not as absolute truth.
12. Prefer strong domain/range/format/age-consistency evidence over weak frequency or dominant-value clues.
13. If the candidate is a context-only identity column such as name or surname, mark it as error only if the value itself has direct missing/format evidence.
14. If evidence is weak and the current value is plausible, prefer "correct" or "unclear".
15. Do not mark a plausible entity string as erroneous only because the neighbor majority or column majority suggests a different value.
16. When a value violates a clear numeric range, domain constraint, or regex/pattern constraint, mark it as error with higher confidence.
17. If suggested_correct_value is provided by rule evidence and it is plausible, you may use it; otherwise set suggested_correct_value to null.
"""


# ============================================================
# 3. 工具函数：prompt 构造
# ============================================================

def as_int01(x: Any) -> int:
    if x is None:
        return 0
    try:
        if pd.isna(x):
            return 0
    except Exception:
        pass
    s = str(x).strip().lower()
    if s in {"1", "true", "yes"}:
        return 1
    if s in {"0", "false", "no"}:
        return 0
    try:
        return int(float(x))
    except Exception:
        return 0


def safe_float(x: Any, default: float = 0.0) -> float:
    if x is None:
        return default
    try:
        if pd.isna(x):
            return default
    except Exception:
        pass
    try:
        return float(x)
    except Exception:
        return default


def get_soccer_summary_lines(summary_ctx: Dict[str, Any]) -> List[str]:
    keys = [
        "target_is_primary_column",
        "target_is_context_only_column",
        "birthyear_value_invalid",
        "season_value_invalid",
        "position_value_invalid",
        "position_needs_canonicalization",
        "birthyear_season_age_conflict",
        "team_context_conflict",
        "format_rule_signal",
        "fd_like_signal",
        "row_birthyear_int",
        "row_season_int",
        "row_age_at_season",
        "row_position_canonical",
        "soccer_attribution_bucket",
    ]
    return [f"- {k}: {summary_ctx.get(k)}" for k in keys if k in summary_ctx]


def build_summary_prefix(sample_obj: Dict[str, Any]) -> str:
    """
    从 summary_context 中抽出一段简短摘要，放在 llm_context_text 前面，
    让模型优先注意聚类/规则统计/Soccer 专属特征。
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

    lines.append(f"- year_value: {summary_ctx.get('year_value')}")
    lines.append(f"- canonical_position: {summary_ctx.get('canonical_position')}")
    lines.append(f"- domain_valid: {summary_ctx.get('domain_valid')}")
    lines.append(f"- soccer_value_bucket: {summary_ctx.get('soccer_value_bucket')}")

    # 兼容字段：如果上游仍保留 adult_* 字段，也一并放入，避免信息丢失。
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

    lines.append(f"- soccer_domain_rule_count: {summary_ctx.get('soccer_domain_rule_count')}")
    lines.append(f"- soccer_format_rule_count: {summary_ctx.get('soccer_format_rule_count')}")
    lines.append(f"- soccer_numeric_rule_count: {summary_ctx.get('soccer_numeric_rule_count')}")
    lines.append(f"- soccer_consistency_rule_count: {summary_ctx.get('soccer_consistency_rule_count')}")

    # 兼容旧训练/传播字段
    lines.append(f"- adult_domain_rule_count: {summary_ctx.get('adult_domain_rule_count')}")
    lines.append(f"- adult_format_rule_count: {summary_ctx.get('adult_format_rule_count')}")
    lines.append(f"- adult_consistency_rule_count: {summary_ctx.get('adult_consistency_rule_count')}")

    lines.append(f"- pattern_bucket: {summary_ctx.get('pattern_bucket')}")
    lines.append(f"- rarity_bucket: {summary_ctx.get('rarity_bucket')}")
    lines.append(f"- neighbor_bucket: {summary_ctx.get('neighbor_bucket')}")
    lines.append(f"- domain_bucket: {summary_ctx.get('domain_bucket')}")
    lines.append(f"- soccer_consistency_bucket: {summary_ctx.get('soccer_consistency_bucket')}")
    lines.append(f"- soccer_attribution_bucket: {summary_ctx.get('soccer_attribution_bucket')}")

    soccer_lines = get_soccer_summary_lines(summary_ctx)
    if soccer_lines:
        lines.append("")
        lines.append("Soccer attribution signals:")
        lines.extend(soccer_lines)

    col = str(sample_obj.get("column", ""))
    main_rule = str(summary_ctx.get("main_rule_type", ""))

    if main_rule == "birthyear_season_age_consistency":
        lines.append("")
        lines.append("Important attribution hint:")
        lines.append("- For implausible age at season, inspect whether birthyear or season itself is invalid; if season looks plausible and birthyear causes the implausible age, mark birthyear as suspicious.")
    elif main_rule == "team_context_dominant":
        lines.append("")
        lines.append("Important attribution hint:")
        lines.append("- Team-context dominant evidence is useful but weak; do not mark a plausible city/stadium/manager as error solely because of majority mismatch.")
    elif col in CONTEXT_ONLY_COLUMNS:
        lines.append("")
        lines.append("Important attribution hint:")
        lines.append("- This is a context-only identity column. Mark it as error only with direct missing/format evidence.")

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
        "normalization_error",
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
        "soccer_consistency",
        "fd_conflict",
        "team_context",
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


def apply_soccer_label_sanity(sample_obj: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    """对 LLM 标注结果做轻量一致性校正，不替代 LLM，只修正明显违背 Soccer 强约束的输出。"""
    out = dict(result)
    summary_ctx = sample_obj.get("summary_context", {}) or {}
    col = str(sample_obj.get("column", ""))
    rule = str(summary_ctx.get("main_rule_type", ""))

    birthyear_invalid = as_int01(summary_ctx.get("birthyear_value_invalid", 0)) == 1
    season_invalid = as_int01(summary_ctx.get("season_value_invalid", 0)) == 1
    position_invalid = as_int01(summary_ctx.get("position_value_invalid", 0)) == 1
    position_norm = as_int01(summary_ctx.get("position_needs_canonicalization", 0)) == 1
    age_conflict = as_int01(summary_ctx.get("birthyear_season_age_conflict", 0)) == 1 or rule == "birthyear_season_age_consistency"
    team_context_conflict = as_int01(summary_ctx.get("team_context_conflict", 0)) == 1 or rule == "team_context_dominant"

    # 明确数值/领域强证据：如果 LLM 判 correct，则进行保守纠正。
    if col == "birthyear" and (birthyear_invalid or age_conflict) and out.get("label") == "correct":
        out["label"] = "error"
        out["is_error"] = True
        out["confidence"] = max(float(out.get("confidence", 0.5)), 0.85)
        out["error_type"] = "semantic_conflict" if age_conflict and not birthyear_invalid else "domain_error"
        ev = set(out.get("evidence_used", []))
        ev.update({"soccer_consistency", "row_context"})
        if birthyear_invalid:
            ev.add("domain_constraint")
        out["evidence_used"] = sorted(ev)
        out["reason_short"] = out.get("reason_short") or "Soccer birthyear/range or age-at-season conflict."
        out["reason_detailed"] = (
            (out.get("reason_detailed") or "")
            + " Soccer consistency indicates invalid birthyear or implausible age at season."
        ).strip()

    if col == "season" and season_invalid and out.get("label") == "correct":
        out["label"] = "error"
        out["is_error"] = True
        out["confidence"] = max(float(out.get("confidence", 0.5)), 0.88)
        out["error_type"] = "domain_error"
        ev = set(out.get("evidence_used", []))
        ev.add("domain_constraint")
        out["evidence_used"] = sorted(ev)

    if col == "position" and position_invalid and out.get("label") == "correct":
        out["label"] = "error"
        out["is_error"] = True
        out["confidence"] = max(float(out.get("confidence", 0.5)), 0.88)
        out["error_type"] = "domain_error"
        ev = set(out.get("evidence_used", []))
        ev.add("domain_constraint")
        out["evidence_used"] = sorted(ev)

    if col == "position" and position_norm and out.get("label") == "correct":
        out["label"] = "error"
        out["is_error"] = True
        out["confidence"] = max(float(out.get("confidence", 0.5)), 0.75)
        out["error_type"] = "normalization_error"
        if not out.get("suggested_correct_value"):
            out["suggested_correct_value"] = POSITION_CANONICAL_MAP.get(str(sample_obj.get("value", "")).strip())
        ev = set(out.get("evidence_used", []))
        ev.update({"domain_constraint", "column_context"})
        out["evidence_used"] = sorted(ev)

    # 上下文 dominant 不能被硬当强真值。如果只有 team context / FD 这类弱证据，降低 error 置信度。
    if out.get("label") == "error":
        direct_strong = (
            birthyear_invalid
            or season_invalid
            or position_invalid
            or position_norm
            or age_conflict
            or safe_float(summary_ctx.get("soccer_domain_rule_count", 0), 0) > 0
            or safe_float(summary_ctx.get("soccer_format_rule_count", 0), 0) > 0
            or safe_float(summary_ctx.get("soccer_numeric_rule_count", 0), 0) > 0
            or safe_float(summary_ctx.get("schema_rule_count", 0), 0) > 0
        )

        if col in CONTEXT_ONLY_COLUMNS and not direct_strong:
            out["confidence"] = min(float(out.get("confidence", 0.5)), 0.65)
            out["needs_human_review"] = True
            out["reason_detailed"] = (
                (out.get("reason_detailed") or "")
                + " Context-only identity column without direct domain/format evidence; downgrade confidence."
            ).strip()

        if team_context_conflict and not direct_strong:
            out["confidence"] = min(float(out.get("confidence", 0.5)), 0.75)
            out["needs_human_review"] = True
            out["reason_detailed"] = (
                (out.get("reason_detailed") or "")
                + " Team-context dominant evidence is weak by itself; downgrade confidence unless supported by direct constraints."
            ).strip()

    return normalize_label_result(out)


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

        # Soccer-specific summary fields
        "year_value": summary_ctx.get("year_value"),
        "canonical_position": summary_ctx.get("canonical_position"),
        "domain_valid": summary_ctx.get("domain_valid"),
        "soccer_value_bucket": summary_ctx.get("soccer_value_bucket"),
        "row_birthyear_int": summary_ctx.get("row_birthyear_int"),
        "row_season_int": summary_ctx.get("row_season_int"),
        "row_age_at_season": summary_ctx.get("row_age_at_season"),
        "row_position_canonical": summary_ctx.get("row_position_canonical"),

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

        "soccer_domain_rule_count": summary_ctx.get("soccer_domain_rule_count"),
        "soccer_format_rule_count": summary_ctx.get("soccer_format_rule_count"),
        "soccer_numeric_rule_count": summary_ctx.get("soccer_numeric_rule_count"),
        "soccer_consistency_rule_count": summary_ctx.get("soccer_consistency_rule_count"),

        # 兼容旧训练/传播脚本
        "adult_domain_rule_count": summary_ctx.get("adult_domain_rule_count"),
        "adult_format_rule_count": summary_ctx.get("adult_format_rule_count"),
        "adult_consistency_rule_count": summary_ctx.get("adult_consistency_rule_count"),
        "time_window_rule_count": summary_ctx.get("time_window_rule_count", 0),
        "time_value_minutes": summary_ctx.get("time_value_minutes"),
        "time_window_bucket": summary_ctx.get("time_window_bucket", "soccer_not_applicable"),
        "time_value_bucket": summary_ctx.get("time_value_bucket", "soccer_not_applicable"),

        "pattern_bucket": summary_ctx.get("pattern_bucket"),
        "rarity_bucket": summary_ctx.get("rarity_bucket"),
        "neighbor_bucket": summary_ctx.get("neighbor_bucket"),
        "domain_bucket": summary_ctx.get("domain_bucket"),
        "soccer_consistency_bucket": summary_ctx.get("soccer_consistency_bucket"),
        "soccer_attribution_bucket": summary_ctx.get("soccer_attribution_bucket"),

        # Soccer attribution features
        "target_is_primary_column": summary_ctx.get("target_is_primary_column"),
        "target_is_context_only_column": summary_ctx.get("target_is_context_only_column"),
        "birthyear_value_invalid": summary_ctx.get("birthyear_value_invalid"),
        "season_value_invalid": summary_ctx.get("season_value_invalid"),
        "position_value_invalid": summary_ctx.get("position_value_invalid"),
        "position_needs_canonicalization": summary_ctx.get("position_needs_canonicalization"),
        "birthyear_season_age_conflict": summary_ctx.get("birthyear_season_age_conflict"),
        "team_context_conflict": summary_ctx.get("team_context_conflict"),
        "format_rule_signal": summary_ctx.get("format_rule_signal"),
        "fd_like_signal": summary_ctx.get("fd_like_signal"),

        # 兼容 Adult v2 字段，soccer 中置空/保留已有值
        "adult_v2_attribution_bucket": summary_ctx.get("adult_v2_attribution_bucket"),
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


def build_output_object(sample: Dict[str, Any], label_result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "row_id": int(sample["row_id"]),
        "column": str(sample["column"]),
        "value": sample.get("value", None),
        "bucket_id": sample.get("bucket_id"),
        "cluster_id": sample.get("cluster_id"),
        "sample_role": sample.get("sample_role"),
        "dist_to_center": sample.get("dist_to_center"),
        "signal_priority": sample.get("signal_priority"),
        "summary_context": sample.get("summary_context", {}),
        "label_result": label_result,
    }


def process_one_sample(client: OpenAI, idx: int, total: int, sample: Dict[str, Any]) -> Tuple[int, Tuple[int, str], Optional[Dict[str, Any]], Optional[str]]:
    row_id = int(sample["row_id"])
    column = str(sample["column"])
    key = (row_id, column)

    try:
        user_prompt = build_user_prompt(sample)
        label_result = call_deepseek_with_retry(client, user_prompt)
        label_result = apply_soccer_label_sanity(sample, label_result)
        out_obj = build_output_object(sample, label_result)
        time.sleep(SLEEP_BETWEEN_REQUESTS_SEC)
        return idx, key, out_obj, None
    except Exception as e:
        if PRINT_TRACEBACK_ON_ERROR:
            traceback.print_exc()
        return idx, key, None, str(e)


def load_existing_outputs(path: str) -> Dict[Tuple[int, str], Dict[str, Any]]:
    existing = {}
    if not os.path.exists(path):
        return existing
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                key = (int(obj["row_id"]), str(obj["column"]))
                existing[key] = obj
            except Exception:
                print(f"[WARN] 读取已有输出时跳过损坏行 line={line_no}")
    return existing


def write_outputs_in_sample_order(samples: List[Dict[str, Any]], output_map: Dict[Tuple[int, str], Dict[str, Any]], output_jsonl: str):
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for sample in samples:
            key = (int(sample["row_id"]), str(sample["column"]))
            obj = output_map.get(key)
            if obj is not None:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ============================================================
# 6. 主流程
# ============================================================

def main():
    client = create_client()
    samples = load_jsonl(INPUT_JSONL, limit=LIMIT)

    existing_outputs = load_existing_outputs(OUTPUT_JSONL) if SKIP_ALREADY_DONE else {}
    done_keys = set(existing_outputs.keys()) if SKIP_ALREADY_DONE else set()

    tasks = []
    skipped = 0
    for idx, sample in enumerate(samples, start=1):
        key = (int(sample["row_id"]), str(sample["column"]))
        if key in done_keys:
            skipped += 1
            continue
        tasks.append((idx, sample))

    processed = 0
    failed = 0
    new_outputs = {}

    if ENABLE_PARALLEL and MAX_WORKERS > 1 and len(tasks) > 1:
        print(f"[INFO] 并行标注启用: MAX_WORKERS={MAX_WORKERS}, 待处理={len(tasks)}")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_map = {
                executor.submit(process_one_sample, client, idx, len(samples), sample): (idx, sample)
                for idx, sample in tasks
            }

            for fut in as_completed(future_map):
                idx, sample = future_map[fut]
                row_id = int(sample["row_id"])
                column = str(sample["column"])

                try:
                    _, key, out_obj, err = fut.result()
                    if err is not None:
                        failed += 1
                        print(f"[ERROR] {idx}/{len(samples)} row_id={row_id}, column={column}, err={err}")
                        continue

                    new_outputs[key] = out_obj
                    lr = out_obj["label_result"]
                    processed += 1
                    print(
                        f"[OK] {idx}/{len(samples)} row_id={row_id}, column={column}, "
                        f"label={lr['label']}, conf={lr['confidence']:.3f}, type={lr['error_type']}"
                    )
                except Exception as e:
                    failed += 1
                    print(f"[ERROR] {idx}/{len(samples)} row_id={row_id}, column={column}, err={e}")
                    if PRINT_TRACEBACK_ON_ERROR:
                        traceback.print_exc()
    else:
        print(f"[INFO] 串行标注: 待处理={len(tasks)}")
        for idx, sample in tasks:
            row_id = int(sample["row_id"])
            column = str(sample["column"])
            key = (row_id, column)

            try:
                user_prompt = build_user_prompt(sample)
                label_result = call_deepseek_with_retry(client, user_prompt)
                label_result = apply_soccer_label_sanity(sample, label_result)
                out_obj = build_output_object(sample, label_result)
                new_outputs[key] = out_obj

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

    # 合并旧输出和新输出，并按 input sample 顺序重写 JSONL，避免并行导致顺序乱。
    output_map = {}
    if SKIP_ALREADY_DONE:
        output_map.update(existing_outputs)
    output_map.update(new_outputs)

    write_outputs_in_sample_order(samples, output_map, OUTPUT_JSONL)
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

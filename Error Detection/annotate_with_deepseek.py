#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unified_annotate_with_deepseek.py

统一版 DeepSeek / OpenAI-compatible 大模型标注脚本，覆盖：
hospital / flights / beers / rayyan / adult / soccer

设计目标：
1. 抽象统一逻辑，方便后续做消融实验。
2. 保持每个数据集原来的默认输入/输出文件名：
   - candidate_sampled_with_context_*.jsonl
   - candidate_sampled_labeled_*.jsonl
   - candidate_sampled_labeled_*.csv
3. 输出 JSONL 保持核心结构：
   row_id, column, value, bucket_id, cluster_id, sample_role, dist_to_center,
   summary_context, label_result
   soccer 兼容 sampling_info / 原始 detail object 输入，并自动补齐 summary_context。
4. 输出 CSV 保持原来主要字段：
   label/is_error/confidence/error_type/reason/evidence_used
   + summary_context 中的训练/传播特征字段。
5. 支持断点续跑、并行、顺序写出、prompt 截断、limit 测试。
6. 后续消融可通过参数关闭 summary prefix、关闭 detail context、只保留强证据等。

安全说明：
- 不再把 API key 写死在代码里。
- 请运行前设置：
  export DEEPSEEK_API_KEY="你的key"
"""

import argparse
import json
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from openai import OpenAI


# ============================================================
# 1. Config
# ============================================================

GENERIC_ALLOWED_ERROR_TYPES = {
    "spelling_variant",
    "semantic_conflict",
    "pattern_error",
    "domain_error",
    "missing_value_error",
    "likely_correct",
    "unclear",
}

SOCCER_ALLOWED_ERROR_TYPES = {
    "domain_error",
    "format_error",
    "semantic_conflict",
    "missing_value_error",
    "context_conflict",
    "normalization_error",
    "likely_correct",
    "unclear",
}

GENERIC_EVIDENCE_ALLOWED = {
    "fd_conflict",
    "neighbor_majority",
    "column_frequency",
    "pattern",
    "row_context",
    "column_context",
    "bayesian_signal",
    "other",
}

SOCCER_EVIDENCE_ALLOWED = {
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


@dataclass
class AnnotateConfig:
    name: str
    input_jsonl: str
    output_jsonl: str
    output_csv: str
    system_prompt: str

    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"
    api_key_env: str = "DEEPSEEK_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 1200
    skip_already_done: bool = True
    enable_parallel: bool = False
    max_workers: int = 1
    sleep_between_requests_sec: float = 0.3
    max_retries: int = 5
    retry_backoff_sec: float = 2.0
    preserve_input_order: bool = True

    max_detail_context_chars: int = 12000
    include_full_sample_when_no_llm_text: bool = True
    print_traceback_on_error: bool = False

    allowed_error_types: Set[str] = field(default_factory=lambda: set(GENERIC_ALLOWED_ERROR_TYPES))
    evidence_allowed: Set[str] = field(default_factory=lambda: set(GENERIC_EVIDENCE_ALLOWED))
    default_error_type_for_error: str = "unclear"

    summary_fields: List[str] = field(default_factory=list)
    prompt_intro: str = "Please analyze the following candidate cell context and output JSON only."
    use_dataset_sanity: bool = False


BASE_SUMMARY_FIELDS = [
    "semantic_type",
    "detected_type",
    "violation_count",
    "conflict_score",
    "value_frequency",
    "value_frequency_rank",
    "neighbor_majority_value",
    "neighbor_majority_ratio",
    "prior_error_probability",
    "posterior_error_probability",
    "main_rule_type",
    "main_usage_role",
    "strong_rule_count",
    "candidate_generation_rule_count",
    "fd_like_count",
    "context_rule_count",
    "global_rule_count",
    "rare_value_count",
    "typo_rule_count",
    "pattern_rule_count",
    "schema_rule_count",
    "pattern_bucket",
    "rarity_bucket",
    "neighbor_bucket",
]

FLIGHTS_FIELDS = [
    "time_value_minutes",
    "time_window_rule_count",
    "time_window_bucket",
    "time_value_bucket",
]

BEERS_FIELDS = [
    "numeric_value",
    "numeric_window_rule_count",
    "numeric_window_bucket",
    "numeric_value_bucket",
]

RAYYAN_FIELDS = [
    "numeric_value",
    "date_value",
    "language_canonical",
    "issn_canonical",
    "canonical_rule_count",
    "canonical_bucket",
    "date_value_bucket",
    "language_bucket",
]

ADULT_FIELDS = [
    "age_lower",
    "age_upper",
    "hours_lower",
    "hours_upper",
    "canonical_income",
    "domain_valid",
    "education_order",
    "adult_value_bucket",
    "adult_domain_rule_count",
    "adult_format_rule_count",
    "adult_consistency_rule_count",
    "adult_consistency_score",
    "adult_evidence_flags",
    "domain_bucket",
    "adult_consistency_bucket",
    "adult_v2_attribution_bucket",
    "target_is_primary_column",
    "target_is_context_only_column",
    "relationship_v2_conflict_count",
    "has_relationship_v2_signal",
    "has_education_v2_signal",
    "adult_v2_signal_strength",
    "adult_v2_primary_target_signal",
    "relationship_marital_conflict",
    "relationship_age_conflict",
    "relationship_sex_conflict",
    "sex_value_invalid",
    "education_age_extreme_conflict",
    "has_relationship_v2_rule",
    "has_education_v2_rule",
    "row_age_lower",
    "row_age_upper",
    "row_hours_lower",
    "row_hours_upper",
    "row_education_order",
    "row_income_canonical",
]

SOCCER_FIELDS = [
    "year_value",
    "canonical_position",
    "domain_valid",
    "soccer_value_bucket",
    "row_birthyear_int",
    "row_season_int",
    "row_age_at_season",
    "row_position_canonical",
    "soccer_domain_rule_count",
    "soccer_format_rule_count",
    "soccer_numeric_rule_count",
    "soccer_consistency_rule_count",
    "adult_domain_rule_count",
    "adult_format_rule_count",
    "adult_consistency_rule_count",
    "evidence_only_fd_count",
    "time_window_rule_count",
    "time_value_minutes",
    "time_window_bucket",
    "time_value_bucket",
    "domain_bucket",
    "soccer_consistency_bucket",
    "soccer_attribution_bucket",
    "soccer_signal_bucket",
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
    "soccer_consistency_score",
    "soccer_evidence_flags",
    "current_value_pattern",
    "dominant_pattern",
    "dominant_pattern_ratio",
]

HOSPITAL_PROMPT = """
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
8. Output must be JSON only, with no markdown fences.
9. Use the provided evidence conservatively. Prefer strong structural evidence and neighbor consensus over weak frequency clues.
10. When the current value is a corrupted variant of a dominant canonical value, prefer error_type="spelling_variant".
"""

FLIGHTS_PROMPT = """
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
  "suggested_correct_value": "7:16 a.m.",
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
8. Output must be JSON only, with no markdown fences.
9. Use the provided evidence conservatively. Prefer strong structural evidence and neighbor consensus over weak frequency clues.
10. For flight time fields, pay attention to time-format validity, delay-window consistency,
    duration consistency, context-dominant values, and neighbor majority.
11. When the value is missing but should co-occur with a paired field, consider missing_value_error.
12. When the evidence is weak and the current value still looks plausible, prefer "correct" or "unclear".
"""

BEERS_PROMPT = """
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
8. Output must be JSON only, with no markdown fences.
9. Use the provided evidence conservatively. Prefer strong structural evidence and neighbor consensus over weak frequency clues.
10. For beer dataset fields, pay attention to brewery consistency, style consistency, numeric plausibility
    of abv/ibu/ounces, unit-format validity, and dominant context values.
11. When the value is missing but should co-occur with a paired field, consider missing_value_error.
12. For brewery-related fields, cross-check brewery_id / brewery_name / city / state consistency.
13. For beer-related fields, cross-check (beer_name, brewery_id) against style / abv / ibu / ounces.
14. When the evidence is weak and the current value still looks plausible, prefer "correct" or "unclear".
"""

RAYYAN_PROMPT = """
You are an expert data-cleaning annotation assistant for bibliographic metadata.

The user will provide a candidate table cell and its structured context.
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
  "suggested_correct_value": "eng",
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
8. Output must be JSON only, with no markdown fences.
9. Use the provided evidence conservatively. Prefer strong structural evidence and neighbor consensus over weak frequency clues.
10. For bibliographic metadata, pay attention to journal-title / abbreviation / ISSN consistency,
    language canonical forms, date plausibility, volume/issue numeric consistency, pagination anomalies,
    and author-list formatting.
11. When the value is missing but should co-occur with a paired field, consider missing_value_error.
12. When the evidence is weak and the current value still looks plausible, prefer "correct" or "unclear".
"""

ADULT_PROMPT = """
You are an expert data-cleaning annotation assistant for an Adult-income tabular dataset.

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
  "evidence_used": ["domain_constraint", "row_context"],
  "suggested_correct_value": "Female",
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
8. Output must be JSON only, with no markdown fences.
9. For Adult data, prefer direct domain/format and explicit consistency evidence over weak FD/frequency evidence.
10. relationship conflicts such as Husband/Wife with non-married status, or sex/relationship mismatch, are important but should be attributed carefully.
11. age, workclass, maritalstatus, occupation, race, hoursperweek, country, and income may be context fields; mark them as error only with direct strong evidence.
12. If evidence is weak and the current value is plausible, prefer "correct" or "unclear".
"""

SOCCER_PROMPT = """
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
2. All columns can be candidate target columns in soccer v2. Do not treat name or surname as context-only.
3. name and manager usually look like alphabetic tokens separated by underscores, for example Brad_Zhaofang or Xiaomin_Saoudi. A missing name/manager or clear format violation is strong evidence.
4. surname usually looks alphabetic, optionally with separators such as hyphen, underscore, or apostrophe. A missing surname or clear format violation is strong evidence.
5. birthplace and city are location-like entity strings. Missing values or clear format violations are strong evidence.
6. birthyear should usually be a four-digit integer within a plausible player birth-year range, roughly 1940..2010.
7. season should usually be a four-digit integer within a plausible season range, roughly 1900..2035.
8. The player's age at season is season - birthyear. It is usually plausible for a professional football player to be about 15..50.
9. position should be a known soccer playing position, e.g., Goalkeeper, Defender, Midfield, Forward. If a value is outside the known domain, that is strong evidence of error.
10. Midfielder may be a normalization variant of Midfield if the dataset canonical label is Midfield. Treat this as normalization_error rather than a severe semantic error.
11. team, stadium, city, birthplace, name, surname, and manager can be long-tail entity fields. Rarity alone is not an error.
12. team-context evidence such as team+season -> stadium/city/manager is useful but not absolute.
13. FD evidence should be treated as supporting evidence, not as absolute truth.
14. soft-FD evidence is weaker than functional_dependency and should not be used alone to mark a plausible value as error.
15. Prefer strong domain/range/format/missing/age-consistency evidence over weak frequency, soft-FD, neighbor majority, or dominant-value clues.
16. If evidence is weak and the current value is plausible, prefer "correct" or "unclear".
17. Do not mark a plausible entity string as erroneous only because the neighbor majority or column majority suggests a different value.
18. When a value violates a clear numeric range, domain constraint, missing constraint, or regex/pattern constraint, mark it as error with higher confidence.
19. If suggested_correct_value is provided by rule evidence and it is plausible, you may use it; otherwise set suggested_correct_value to null.
"""


def build_configs() -> Dict[str, AnnotateConfig]:
    return {
        "hospital": AnnotateConfig(
            name="hospital",
            input_jsonl="candidate_sampled_with_context_v2.jsonl",
            output_jsonl="candidate_sampled_labeled_v2.jsonl",
            output_csv="candidate_sampled_labeled_v2.csv",
            system_prompt=HOSPITAL_PROMPT,
            summary_fields=BASE_SUMMARY_FIELDS,
            enable_parallel=False,
            max_workers=1,
            sleep_between_requests_sec=0.3,
        ),
        "flights": AnnotateConfig(
            name="flights",
            input_jsonl="candidate_sampled_with_context_flights.jsonl",
            output_jsonl="candidate_sampled_labeled_flights.jsonl",
            output_csv="candidate_sampled_labeled_flights.csv",
            system_prompt=FLIGHTS_PROMPT,
            summary_fields=BASE_SUMMARY_FIELDS + FLIGHTS_FIELDS,
            enable_parallel=False,
            max_workers=1,
            sleep_between_requests_sec=0.3,
        ),
        "beers": AnnotateConfig(
            name="beers",
            input_jsonl="candidate_sampled_with_context_beers.jsonl",
            output_jsonl="candidate_sampled_labeled_beers.jsonl",
            output_csv="candidate_sampled_labeled_beers.csv",
            system_prompt=BEERS_PROMPT,
            summary_fields=BASE_SUMMARY_FIELDS + BEERS_FIELDS,
            enable_parallel=True,
            max_workers=6,
            sleep_between_requests_sec=0.05,
        ),
        "rayyan": AnnotateConfig(
            name="rayyan",
            input_jsonl="candidate_sampled_with_context_rayyan.jsonl",
            output_jsonl="candidate_sampled_labeled_rayyan.jsonl",
            output_csv="candidate_sampled_labeled_rayyan.csv",
            system_prompt=RAYYAN_PROMPT,
            summary_fields=BASE_SUMMARY_FIELDS + RAYYAN_FIELDS,
            prompt_intro="Please analyze the following candidate bibliographic metadata cell context and output JSON only.",
            enable_parallel=True,
            max_workers=6,
            sleep_between_requests_sec=0.1,
        ),
        "adult": AnnotateConfig(
            name="adult",
            input_jsonl="candidate_sampled_with_context_adult.jsonl",
            output_jsonl="candidate_sampled_labeled_adult.jsonl",
            output_csv="candidate_sampled_labeled_adult.csv",
            system_prompt=ADULT_PROMPT,
            summary_fields=BASE_SUMMARY_FIELDS + ADULT_FIELDS,
            enable_parallel=True,
            max_workers=4,
            sleep_between_requests_sec=0.2,
        ),
        "soccer": AnnotateConfig(
            name="soccer",
            input_jsonl="candidate_sampled_with_context_soccer.jsonl",
            output_jsonl="candidate_sampled_labeled_soccer.jsonl",
            output_csv="candidate_sampled_labeled_soccer.csv",
            system_prompt=SOCCER_PROMPT,
            summary_fields=BASE_SUMMARY_FIELDS + SOCCER_FIELDS,
            allowed_error_types=set(SOCCER_ALLOWED_ERROR_TYPES),
            evidence_allowed=set(SOCCER_EVIDENCE_ALLOWED),
            default_error_type_for_error="unclear",
            enable_parallel=True,
            max_workers=4,
            sleep_between_requests_sec=0.2,
            max_detail_context_chars=12000,
            include_full_sample_when_no_llm_text=True,
            use_dataset_sanity=True,
        ),
    }


CONFIGS = build_configs()


# ============================================================
# 2. Basic IO
# ============================================================

def load_jsonl(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
            if limit is not None and len(items) >= limit:
                break
    return items


def get_key(obj: Dict[str, Any]) -> Tuple[int, str]:
    return (int(obj["row_id"]), str(obj["column"]))


def load_existing_outputs(path: str) -> Dict[Tuple[int, str], Dict[str, Any]]:
    out: Dict[Tuple[int, str], Dict[str, Any]] = {}
    if not os.path.exists(path):
        return out

    bad = 0
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "row_id" in obj and "column" in obj:
                    out[get_key(obj)] = obj
            except Exception as e:
                bad += 1
                print(f"[WARN] skip broken JSONL line={lineno}, err={e}")
    if bad:
        print(f"[WARN] broken output lines skipped: {bad}")
    return out


def write_outputs_in_sample_order(samples: List[Dict[str, Any]], output_map: Dict[Tuple[int, str], Dict[str, Any]], output_jsonl: str):
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for sample in samples:
            key = get_key(sample)
            obj = output_map.get(key)
            if obj is not None:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ============================================================
# 3. Summary context extraction
# ============================================================

def get_nested(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def is_empty_value(v: Any) -> bool:
    if v is None:
        return True
    try:
        if pd.isna(v):
            return True
    except Exception:
        pass
    s = str(v)
    return s == "" or s.lower() == "nan"


def merge_if_missing(dst: Dict[str, Any], src: Dict[str, Any]):
    for k, v in src.items():
        if k not in dst or is_empty_value(dst.get(k)):
            dst[k] = v


def get_sample_detail_context(sample_obj: Dict[str, Any]) -> Dict[str, Any]:
    detail = sample_obj.get("detail_context")
    if isinstance(detail, dict):
        return detail
    # soccer unified sampling may write original detail object directly with sampling_info.
    return sample_obj


def get_sample_summary_context(sample_obj: Dict[str, Any], cfg: AnnotateConfig) -> Dict[str, Any]:
    summary_ctx = dict(sample_obj.get("summary_context", {}) or {})

    # sampling_info is used by soccer unified sampler.
    sampling_info = sample_obj.get("sampling_info", {}) or {}
    if isinstance(sampling_info, dict):
        merge_if_missing(summary_ctx, sampling_info)

    detail = get_sample_detail_context(sample_obj)

    column_ctx = detail.get("column_context", {}) or {}
    conflict_ctx = detail.get("conflict_context", {}) or {}
    sim_ctx = detail.get("similarity_context", {}) or {}
    bayes_ctx = detail.get("bayesian_context", {}) or {}
    adult_ctx = detail.get("adult_consistency_context", {}) or {}
    adult_attr = detail.get("adult_v2_attribution_features", {}) or {}
    soccer_attr = detail.get("soccer_attribution_features", {}) or {}
    soccer_cons = detail.get("soccer_consistency_context", {}) or detail.get("adult_consistency_context", {}) or {}

    base = {
        "semantic_type": column_ctx.get("semantic_type"),
        "detected_type": column_ctx.get("detected_type"),
        "violation_count": detail.get("violation_count"),
        "conflict_score": detail.get("conflict_score"),
        "value_frequency": column_ctx.get("value_frequency"),
        "value_frequency_rank": column_ctx.get("value_frequency_rank"),
        "neighbor_majority_value": sim_ctx.get("neighbor_majority_value"),
        "neighbor_majority_ratio": sim_ctx.get("neighbor_majority_ratio"),
        "prior_error_probability": bayes_ctx.get("prior_error_probability"),
        "posterior_error_probability": bayes_ctx.get("posterior_error_probability"),
        "main_rule_type": conflict_ctx.get("main_rule_type"),
        "main_usage_role": conflict_ctx.get("main_usage_role"),
        "strong_rule_count": conflict_ctx.get("strong_rule_count"),
        "candidate_generation_rule_count": conflict_ctx.get("candidate_generation_rule_count"),
        "fd_like_count": conflict_ctx.get("fd_like_count"),
        "evidence_only_fd_count": conflict_ctx.get("evidence_only_fd_count"),
        "context_rule_count": conflict_ctx.get("context_rule_count"),
        "global_rule_count": conflict_ctx.get("global_rule_count"),
        "rare_value_count": conflict_ctx.get("rare_value_count"),
        "typo_rule_count": conflict_ctx.get("typo_rule_count"),
        "pattern_rule_count": conflict_ctx.get("pattern_rule_count"),
        "schema_rule_count": conflict_ctx.get("schema_rule_count"),
        "current_value_pattern": column_ctx.get("current_value_pattern"),
        "dominant_pattern": column_ctx.get("dominant_pattern"),
        "dominant_pattern_ratio": column_ctx.get("dominant_pattern_ratio"),
    }

    if cfg.name == "flights":
        base.update({
            "time_value_minutes": column_ctx.get("time_value_minutes"),
            "time_window_rule_count": conflict_ctx.get("time_window_rule_count"),
        })

    elif cfg.name == "beers":
        base.update({
            "numeric_value": column_ctx.get("numeric_value"),
            "numeric_window_rule_count": conflict_ctx.get("numeric_window_rule_count"),
        })

    elif cfg.name == "rayyan":
        base.update({
            "numeric_value": column_ctx.get("numeric_value"),
            "date_value": column_ctx.get("date_value"),
            "language_canonical": column_ctx.get("language_canonical"),
            "issn_canonical": column_ctx.get("issn_canonical"),
            "canonical_rule_count": conflict_ctx.get("canonical_rule_count"),
        })

    elif cfg.name == "adult":
        adult_value_features = column_ctx.get("adult_value_features", {}) or {}
        base.update({
            **adult_value_features,
            "adult_domain_rule_count": conflict_ctx.get("adult_domain_rule_count"),
            "adult_format_rule_count": conflict_ctx.get("adult_format_rule_count"),
            "adult_consistency_rule_count": conflict_ctx.get("adult_consistency_rule_count"),
            "adult_consistency_score": adult_ctx.get("adult_consistency_score"),
            "adult_evidence_flags": "|".join(adult_ctx.get("evidence_flags", [])) if isinstance(adult_ctx.get("evidence_flags"), list) else adult_ctx.get("evidence_flags"),
        })
        base.update(adult_attr)

    elif cfg.name == "soccer":
        soccer_value_features = column_ctx.get("soccer_value_features", {}) or {}
        base.update({
            **soccer_value_features,
            "soccer_domain_rule_count": conflict_ctx.get("soccer_domain_rule_count"),
            "soccer_format_rule_count": conflict_ctx.get("soccer_format_rule_count"),
            "soccer_numeric_rule_count": conflict_ctx.get("soccer_numeric_rule_count"),
            "soccer_consistency_rule_count": conflict_ctx.get("soccer_consistency_rule_count"),
            "adult_domain_rule_count": conflict_ctx.get("adult_domain_rule_count"),
            "adult_format_rule_count": conflict_ctx.get("adult_format_rule_count"),
            "adult_consistency_rule_count": conflict_ctx.get("adult_consistency_rule_count"),
            "soccer_consistency_score": soccer_cons.get("soccer_consistency_score"),
            "soccer_evidence_flags": "|".join(soccer_cons.get("evidence_flags", [])) if isinstance(soccer_cons.get("evidence_flags"), list) else soccer_cons.get("evidence_flags"),
            "time_window_rule_count": conflict_ctx.get("time_window_rule_count", 0),
            "time_value_minutes": None,
            "time_window_bucket": "soccer_not_applicable",
            "time_value_bucket": "soccer_not_applicable",
        })
        base.update(soccer_attr)

    merge_if_missing(summary_ctx, base)

    if cfg.name == "soccer":
        col = str(sample_obj.get("column", detail.get("column", "")))
        rule = str(summary_ctx.get("main_rule_type") or "")

        # Soccer v2: all soccer columns are primary target columns.
        soccer_cols = {"name", "surname", "birthyear", "birthplace", "position", "team", "city", "stadium", "season", "manager"}
        summary_ctx["target_is_primary_column"] = 1 if col in soccer_cols else as_int01(summary_ctx.get("target_is_primary_column"))
        summary_ctx["target_is_context_only_column"] = 0

        if is_empty_value(summary_ctx.get("birthyear_value_invalid")):
            summary_ctx["birthyear_value_invalid"] = 1 if rule == "birthyear_numeric_range" else 0
        if is_empty_value(summary_ctx.get("season_value_invalid")):
            summary_ctx["season_value_invalid"] = 1 if rule == "season_numeric_range" else 0
        if is_empty_value(summary_ctx.get("position_value_invalid")):
            summary_ctx["position_value_invalid"] = 1 if rule == "position_domain" else 0
        if is_empty_value(summary_ctx.get("position_needs_canonicalization")):
            summary_ctx["position_needs_canonicalization"] = 0
        if is_empty_value(summary_ctx.get("birthyear_season_age_conflict")):
            summary_ctx["birthyear_season_age_conflict"] = 1 if rule == "birthyear_season_age_consistency" else 0

        soccer_context_rules = {
            "team_context_dominant",
            "team_season_manager_consistency_hint",
            "city_stadium_consistency_hint",
            "dominant_value_by_context",
            "dominant_value_by_context_pair",
        }
        soccer_format_rules = {
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
        soccer_fd_rules = {"functional_dependency"}
        soccer_soft_fd_rules = {"soft_functional_dependency"}

        if is_empty_value(summary_ctx.get("team_context_conflict")):
            summary_ctx["team_context_conflict"] = 1 if rule in soccer_context_rules else 0
        if is_empty_value(summary_ctx.get("format_rule_signal")):
            summary_ctx["format_rule_signal"] = 1 if rule in soccer_format_rules else 0
        if is_empty_value(summary_ctx.get("fd_like_signal")):
            summary_ctx["fd_like_signal"] = 1 if rule in soccer_fd_rules else 0
        if rule in soccer_soft_fd_rules:
            summary_ctx["fd_like_signal"] = 0
            summary_ctx["evidence_only_fd_count"] = max(as_int01(summary_ctx.get("evidence_only_fd_count")), 1)

        attr = summary_ctx.get("soccer_attribution_bucket")
        if is_empty_value(attr) or str(attr) in {"missing", "unknown", "unknown_attribution", "nan", "None"}:
            if as_int01(summary_ctx.get("birthyear_value_invalid")):
                attr = "birthyear_invalid"
            elif as_int01(summary_ctx.get("season_value_invalid")):
                attr = "season_invalid"
            elif as_int01(summary_ctx.get("position_value_invalid")):
                attr = "position_invalid"
            elif as_int01(summary_ctx.get("position_needs_canonicalization")):
                attr = "position_canonicalization"
            elif as_int01(summary_ctx.get("birthyear_season_age_conflict")):
                attr = "birthyear_season_age_conflict"
            elif as_int01(summary_ctx.get("team_context_conflict")):
                attr = "team_context_conflict"
            elif as_int01(summary_ctx.get("format_rule_signal")):
                attr = "format_pattern_conflict"
            elif as_int01(summary_ctx.get("fd_like_signal")):
                attr = "fd_like_conflict"
            elif col in soccer_cols:
                attr = "primary_target_other"
            else:
                attr = "unknown_attribution"
        summary_ctx["soccer_attribution_bucket"] = attr

    return summary_ctx


def as_int01(x: Any) -> int:
    if x is None:
        return 0
    try:
        if pd.isna(x):
            return 0
    except Exception:
        pass
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y"}:
        return 1
    if s in {"0", "false", "no", "n"}:
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


# ============================================================
# 4. Prompt construction
# ============================================================

def build_summary_prefix(sample_obj: Dict[str, Any], cfg: AnnotateConfig, disable_summary_prefix: bool = False) -> str:
    if disable_summary_prefix:
        return ""

    summary_ctx = get_sample_summary_context(sample_obj, cfg)
    if not summary_ctx:
        return ""

    lines = ["Summary signals:"]
    for k in cfg.summary_fields:
        if k in summary_ctx:
            lines.append(f"- {k}: {summary_ctx.get(k)}")

    if cfg.name == "soccer":
        soccer_keys = [
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
            "evidence_only_fd_count",
            "row_birthyear_int",
            "row_season_int",
            "row_age_at_season",
            "row_position_canonical",
            "soccer_attribution_bucket",
        ]
        soccer_lines = [f"- {k}: {summary_ctx.get(k)}" for k in soccer_keys if k in summary_ctx]
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
        elif main_rule == "soft_functional_dependency":
            lines.append("")
            lines.append("Important attribution hint:")
            lines.append("- soft-FD is weak supporting evidence only. Do not mark a plausible value as error solely based on soft-FD.")
        if col in {"name", "surname", "birthplace"}:
            lines.append("")
            lines.append("Important recall hint:")
            lines.append("- This column had previous missed errors; direct missing/format evidence should be taken seriously, but rarity alone is still not an error.")

    elif cfg.name == "adult":
        main_rule = str(summary_ctx.get("main_rule_type", ""))
        if main_rule in {
            "relationship_marital_spouse_conflict",
            "relationship_age_spouse_conflict",
            "relationship_sex_spouse_conflict",
            "relationship_context_dominant",
        }:
            lines.append("")
            lines.append("Important attribution hint:")
            lines.append("- Relationship consistency evidence usually indicates the relationship cell is suspicious, not necessarily all context fields.")

    return "\n".join(lines)


def truncate_text(text: str, max_chars: int) -> str:
    if text is None:
        return ""
    text = str(text)
    if max_chars is None or max_chars <= 0 or len(text) <= max_chars:
        return text
    head = text[: int(max_chars * 0.75)]
    tail = text[-int(max_chars * 0.25):]
    return head + "\n\n...[TRUNCATED_TO_SAVE_TOKENS]...\n\n" + tail


def build_user_prompt(sample_obj: Dict[str, Any], cfg: AnnotateConfig, disable_summary_prefix=False, disable_detail_context=False) -> str:
    detail_ctx = get_sample_detail_context(sample_obj)
    llm_text = detail_ctx.get("llm_context_text") if isinstance(detail_ctx, dict) else None
    summary_prefix = build_summary_prefix(sample_obj, cfg, disable_summary_prefix=disable_summary_prefix)

    chunks = [cfg.prompt_intro, ""]
    if summary_prefix:
        chunks.append(summary_prefix)
        chunks.append("")

    if llm_text and not disable_detail_context:
        chunks.append("Detailed context:")
        chunks.append(truncate_text(llm_text, cfg.max_detail_context_chars))
    else:
        if cfg.include_full_sample_when_no_llm_text:
            chunks.append("Full sample json:")
            chunks.append(truncate_text(json.dumps(sample_obj, ensure_ascii=False, indent=2), cfg.max_detail_context_chars))
        else:
            chunks.append("No detailed context is provided. Use the summary signals only.")

    return "\n".join(chunks)


# ============================================================
# 5. LLM call and label normalization
# ============================================================

def create_client(cfg: AnnotateConfig) -> OpenAI:
    api_key = os.getenv(cfg.api_key_env, "")
    if not api_key:
        raise RuntimeError(
            f"{cfg.api_key_env} 未设置。请先执行：export {cfg.api_key_env}='你的key'"
        )
    return OpenAI(api_key=api_key, base_url=cfg.base_url)


def safe_parse_json(text: str) -> Dict[str, Any]:
    text = str(text).strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json", "", 1).strip()

    # 有时模型会在 JSON 前后输出少量文本，这里尝试截取最外层 JSON。
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start:end + 1]

    return json.loads(text)


def normalize_label_result(result: Dict[str, Any], cfg: AnnotateConfig) -> Dict[str, Any]:
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
    if error_type not in cfg.allowed_error_types:
        if label == "correct":
            error_type = "likely_correct"
        else:
            error_type = cfg.default_error_type_for_error

    evidence_used = result.get("evidence_used", [])
    if not isinstance(evidence_used, list):
        evidence_used = ["other"]
    evidence_used = [str(x).strip() for x in evidence_used if str(x).strip()]
    if not evidence_used:
        evidence_used = ["other"]
    # 不强制删除旧证据名，避免后续分析丢信息；但空值归 other。
    evidence_used = [x if x in cfg.evidence_allowed else x for x in evidence_used]

    suggested_correct_value = result.get("suggested_correct_value", None)
    if suggested_correct_value == "":
        suggested_correct_value = None

    needs_human_review = result.get("needs_human_review", False)
    if not isinstance(needs_human_review, bool):
        needs_human_review = str(needs_human_review).strip().lower() in {"1", "true", "yes", "y"}

    return {
        "is_error": bool(is_error),
        "label": label,
        "confidence": confidence,
        "error_type": error_type,
        "reason_short": str(result.get("reason_short", "")).strip(),
        "reason_detailed": str(result.get("reason_detailed", "")).strip(),
        "evidence_used": evidence_used,
        "suggested_correct_value": suggested_correct_value,
        "needs_human_review": bool(needs_human_review),
    }


def apply_soccer_label_sanity(sample_obj: Dict[str, Any], result: Dict[str, Any], cfg: AnnotateConfig) -> Dict[str, Any]:
    """Soccer v2 的保守修正：降低弱证据误标，增强强证据一致性。
    该函数只基于样本自身上下文，不读取 ground truth。
    """
    if cfg.name != "soccer":
        return result

    summary_ctx = get_sample_summary_context(sample_obj, cfg)
    col = str(sample_obj.get("column", ""))
    rule = str(summary_ctx.get("main_rule_type", ""))

    strong_direct = (
        as_int01(summary_ctx.get("birthyear_value_invalid"))
        or as_int01(summary_ctx.get("season_value_invalid"))
        or as_int01(summary_ctx.get("position_value_invalid"))
        or as_int01(summary_ctx.get("position_needs_canonicalization"))
        or as_int01(summary_ctx.get("birthyear_season_age_conflict"))
        or as_int01(summary_ctx.get("format_rule_signal"))
        or safe_float(summary_ctx.get("soccer_domain_rule_count")) > 0
        or safe_float(summary_ctx.get("soccer_format_rule_count")) > 0
        or safe_float(summary_ctx.get("soccer_numeric_rule_count")) > 0
    )

    weak_only_rules = {
        "soft_functional_dependency",
        "rare_value",
        "rare_pattern",
        "global_dominant_value",
        "dominant_value_by_context",
        "dominant_value_by_context_pair",
        "team_context_dominant",
        "team_season_manager_consistency_hint",
        "city_stadium_consistency_hint",
    }

    if result["label"] == "error" and rule in weak_only_rules and not strong_direct:
        # 弱证据不能单独把长尾实体判错。
        if col in {"name", "surname", "birthplace", "team", "city", "stadium", "manager"}:
            result = dict(result)
            result["label"] = "correct"
            result["is_error"] = False
            result["error_type"] = "likely_correct"
            result["confidence"] = min(float(result.get("confidence", 0.5)), 0.62)
            result["needs_human_review"] = True
            result["reason_short"] = "Weak evidence only; plausible long-tail soccer entity."
            result["reason_detailed"] = (
                "The available signal is weak context/frequency/soft-FD evidence. "
                "For soccer entity fields, rarity or neighbor mismatch alone is not sufficient to mark a plausible value as an error."
            )

    if result["label"] == "correct" and strong_direct:
        # 强直接证据时提高人工复核标志，但不强行改 label，避免过拟合规则。
        result = dict(result)
        result["needs_human_review"] = True
        result["confidence"] = min(float(result.get("confidence", 0.5)), 0.85)

    return result


def call_llm_with_retry(client: OpenAI, user_prompt: str, cfg: AnnotateConfig) -> Dict[str, Any]:
    last_err = None
    for attempt in range(1, cfg.max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=cfg.model,
                messages=[
                    {"role": "system", "content": cfg.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
            )
            text = resp.choices[0].message.content
            result = safe_parse_json(text)
            return normalize_label_result(result, cfg)
        except Exception as e:
            last_err = e
            msg = str(e)
            if "Insufficient Balance" in msg or "402" in msg:
                raise RuntimeError(f"API balance/quota error: {e}")
            if attempt < cfg.max_retries:
                time.sleep(cfg.retry_backoff_sec * attempt)
            else:
                raise RuntimeError(f"DeepSeek 调用失败，已重试 {cfg.max_retries} 次: {last_err}")
    raise RuntimeError(str(last_err))


# ============================================================
# 6. Output object and CSV flattening
# ============================================================

def get_sample_basic_fields(sample: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "row_id": int(sample["row_id"]),
        "column": str(sample["column"]),
        "value": sample.get("value"),
        "bucket_id": sample.get("bucket_id"),
        "cluster_id": sample.get("cluster_id"),
        "sample_role": sample.get("sample_role"),
        "dist_to_center": sample.get("dist_to_center"),
        "signal_priority": sample.get("signal_priority"),
    }

    sampling_info = sample.get("sampling_info", {}) or {}
    if isinstance(sampling_info, dict):
        for k in ["bucket_id", "cluster_id", "sample_role", "dist_to_center", "signal_priority"]:
            if out.get(k) is None and k in sampling_info:
                out[k] = sampling_info.get(k)

    return out


def build_output_object(sample: Dict[str, Any], label_result: Dict[str, Any], cfg: AnnotateConfig) -> Dict[str, Any]:
    basic = get_sample_basic_fields(sample)
    summary_ctx = get_sample_summary_context(sample, cfg)

    out_obj = {
        **basic,
        "summary_context": summary_ctx,
        "label_result": label_result,
    }

    # 保留旧脚本可读信息
    if "sampling_info" in sample:
        out_obj["sampling_info"] = sample.get("sampling_info")

    return out_obj


def flatten_output_row(obj: Dict[str, Any], cfg: AnnotateConfig) -> Dict[str, Any]:
    lr = obj["label_result"]
    summary_ctx = obj.get("summary_context", {}) or {}

    row = {
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
    }

    if "signal_priority" in obj:
        row["signal_priority"] = obj.get("signal_priority")

    # 兼容旧 hospital/flights/beers/rayyan：这些旧 CSV 没有 label_binary，但多一列一般不影响；
    # 若你要完全旧列，可用 --legacy-no-label-binary。
    for field in cfg.summary_fields:
        row[field] = summary_ctx.get(field)

    # 确保 label propagation / training 常用字段总是存在
    for field in BASE_SUMMARY_FIELDS:
        if field not in row:
            row[field] = summary_ctx.get(field)

    return row


def export_csv_from_outputs(samples: List[Dict[str, Any]], output_map: Dict[Tuple[int, str], Dict[str, Any]], output_csv: str, cfg: AnnotateConfig, legacy_no_label_binary=False):
    rows = []
    for sample in samples:
        key = get_key(sample)
        obj = output_map.get(key)
        if obj is None:
            continue
        row = flatten_output_row(obj, cfg)
        if legacy_no_label_binary and cfg.name not in {"adult", "soccer"}:
            row.pop("label_binary", None)
        rows.append(row)

    if rows:
        pd.DataFrame(rows).to_csv(output_csv, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(output_csv, index=False, encoding="utf-8-sig")


# ============================================================
# 7. Processing
# ============================================================

def process_one_sample(idx: int, total: int, sample: Dict[str, Any], cfg: AnnotateConfig, args) -> Tuple[int, Tuple[int, str], Optional[Dict[str, Any]], Optional[str]]:
    client = create_client(cfg)
    key = get_key(sample)
    try:
        user_prompt = build_user_prompt(
            sample,
            cfg,
            disable_summary_prefix=args.disable_summary_prefix,
            disable_detail_context=args.disable_detail_context,
        )
        label_result = call_llm_with_retry(client, user_prompt, cfg)
        label_result = apply_soccer_label_sanity(sample, label_result, cfg) if cfg.use_dataset_sanity else label_result
        out_obj = build_output_object(sample, label_result, cfg)
        time.sleep(cfg.sleep_between_requests_sec)
        return idx, key, out_obj, None
    except Exception as e:
        if cfg.print_traceback_on_error or args.print_traceback:
            traceback.print_exc()
        return idx, key, None, str(e)


def annotate_dataset(cfg: AnnotateConfig, args):
    if args.model:
        cfg.model = args.model
    if args.base_url:
        cfg.base_url = args.base_url
    if args.max_workers is not None:
        cfg.max_workers = int(args.max_workers)
        cfg.enable_parallel = cfg.max_workers > 1
    if args.sequential:
        cfg.enable_parallel = False
        cfg.max_workers = 1
    if args.limit is not None:
        limit = args.limit
    else:
        limit = args.limit_default
    if args.max_detail_context_chars is not None:
        cfg.max_detail_context_chars = int(args.max_detail_context_chars)

    input_jsonl = args.input_jsonl or cfg.input_jsonl
    output_jsonl = args.output_jsonl or cfg.output_jsonl
    output_csv = args.output_csv or cfg.output_csv

    samples = load_jsonl(input_jsonl, limit=limit)
    output_map = load_existing_outputs(output_jsonl) if cfg.skip_already_done and not args.no_skip else {}

    pending: List[Tuple[int, Dict[str, Any]]] = []
    skipped = 0
    for idx, sample in enumerate(samples):
        key = get_key(sample)
        if key in output_map and cfg.skip_already_done and not args.no_skip:
            skipped += 1
            continue
        pending.append((idx, sample))

    print(f"[INFO] dataset={cfg.name}")
    print(f"[INFO] input samples={len(samples)}")
    print(f"[INFO] pending={len(pending)}, skipped_existing={skipped}")
    print(f"[INFO] output_jsonl={output_jsonl}")
    print(f"[INFO] output_csv={output_csv}")

    processed = 0
    failed = 0

    if pending:
        if cfg.enable_parallel and cfg.max_workers > 1:
            print(f"[INFO] 并行标注启用: MAX_WORKERS={cfg.max_workers}, 待处理={len(pending)}")
            with ThreadPoolExecutor(max_workers=cfg.max_workers) as executor:
                futures = {
                    executor.submit(process_one_sample, idx + 1, len(pending), sample, cfg, args): (idx, sample)
                    for idx, sample in pending
                }
                for done_i, fut in enumerate(as_completed(futures), start=1):
                    _, sample = futures[fut]
                    row_id = int(sample["row_id"])
                    column = str(sample["column"])
                    idx, key, out_obj, err = fut.result()
                    if out_obj is not None:
                        output_map[key] = out_obj
                        lr = out_obj["label_result"]
                        processed += 1
                        print(f"[OK] {done_i}/{len(pending)} row_id={row_id}, column={column}, label={lr['label']}, conf={lr['confidence']:.3f}")
                    else:
                        failed += 1
                        print(f"[ERROR] {done_i}/{len(pending)} row_id={row_id}, column={column}, err={err}")
        else:
            client = create_client(cfg)
            for done_i, (idx0, sample) in enumerate(pending, start=1):
                key = get_key(sample)
                row_id, column = key
                try:
                    user_prompt = build_user_prompt(
                        sample,
                        cfg,
                        disable_summary_prefix=args.disable_summary_prefix,
                        disable_detail_context=args.disable_detail_context,
                    )
                    label_result = call_llm_with_retry(client, user_prompt, cfg)
                    label_result = apply_soccer_label_sanity(sample, label_result, cfg) if cfg.use_dataset_sanity else label_result
                    out_obj = build_output_object(sample, label_result, cfg)
                    output_map[key] = out_obj
                    processed += 1
                    print(f"[OK] {done_i}/{len(pending)} row_id={row_id}, column={column}, label={label_result['label']}, conf={label_result['confidence']:.3f}")
                except Exception as e:
                    failed += 1
                    if cfg.print_traceback_on_error or args.print_traceback:
                        traceback.print_exc()
                    print(f"[ERROR] {done_i}/{len(pending)} row_id={row_id}, column={column}, err={e}")
                time.sleep(cfg.sleep_between_requests_sec)

    write_outputs_in_sample_order(samples, output_map, output_jsonl)
    export_csv_from_outputs(samples, output_map, output_csv, cfg, legacy_no_label_binary=args.legacy_no_label_binary)

    print("\n===== 完成 =====")
    print(f"输入样本数: {len(samples)}")
    print(f"新处理样本数: {processed}")
    print(f"跳过已完成样本数: {skipped}")
    print(f"失败样本数: {failed}")
    print(f"JSONL 输出: {output_jsonl}")
    print(f"CSV 输出: {output_csv}")


# ============================================================
# 8. CLI
# ============================================================

def run_one_dataset(dataset: str, args):
    cfg = CONFIGS[dataset]

    old_cwd = None
    if args.cwd:
        os.makedirs(args.cwd, exist_ok=True)
        old_cwd = os.getcwd()
        os.chdir(args.cwd)

    try:
        annotate_dataset(cfg, args)
    finally:
        if old_cwd is not None:
            os.chdir(old_cwd)


def parse_args():
    parser = argparse.ArgumentParser(description="Unified DeepSeek annotation script for data-cleaning candidates.")
    parser.add_argument("--dataset", required=True, choices=sorted(list(CONFIGS.keys()) + ["all"]))

    parser.add_argument("--input_jsonl", default=None)
    parser.add_argument("--output_jsonl", default=None)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--cwd", default=None)

    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--limit-default", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--no-skip", action="store_true")
    parser.add_argument("--max-detail-context-chars", type=int, default=None)
    parser.add_argument("--print-traceback", action="store_true")

    # Ablation controls
    parser.add_argument("--disable-summary-prefix", action="store_true", help="Do not prepend structured summary signals to the prompt.")
    parser.add_argument("--disable-detail-context", action="store_true", help="Do not include detail_context.llm_context_text in the prompt.")
    parser.add_argument("--legacy-no-label-binary", action="store_true", help="Remove label_binary from generic output CSV for closer legacy compatibility.")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.dataset == "all":
        if args.input_jsonl or args.output_jsonl or args.output_csv:
            raise ValueError("--dataset all cannot be used with input/output overrides.")
        for ds in ["hospital", "flights", "beers", "rayyan", "adult", "soccer"]:
            print("\n" + "=" * 80)
            print(f"[DATASET] {ds}")
            print("=" * 80)
            run_one_dataset(ds, args)
    else:
        run_one_dataset(args.dataset, args)


if __name__ == "__main__":
    main()

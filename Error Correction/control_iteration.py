#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified iterative repair controller.

Datasets:
  hospital, flights, beers, rayyan, adult, soccer

Purpose
-------
This script merges the common control_iteration.py logic from six datasets into
one readable, configurable implementation.

It preserves the standard controller outputs:

  <work_dir>/
    pairwise_responses_accumulated.jsonl
    pairwise_responses_new_from_llm.jsonl
    repair_candidate_features_infer_whitelist.csv
    iteration_summary.csv
    controller_run_config.json
    model_output/
      pairwise_train_materialized.csv
      training_log.csv
      feature_columns.json
      reason_label_mapping.json
      column_profiles.json
      best_model.pt
      last_model.pt
      inference_all_candidate_scores.csv
      inference_all_candidate_ranked.csv
      inference_top1_repairs.csv
      hard_cases_for_llm.jsonl
      input_whitelist_violations.csv
      output_whitelist_violations.csv

Common loop
-----------
1. Bootstrap accumulated pairwise labels from repair_pairwise_responses_v2.jsonl.
2. Build inference whitelist features.
3. Train pairwise LTR student using accumulated labels.
4. Run inference with conservative gates.
5. Read hard_cases_for_llm.jsonl.
6. Ask LLM to label hard pairwise comparisons.
7. Merge new labels into accumulated pairwise file.
8. Repeat.

Dataset-specific differences are isolated in DATASET_CONFIGS:
  - default allowed_cells file
  - default work_dir
  - column-specific prompt instructions
  - reason_type names
  - source tag for new LLM rows
  - optional skip_llm flags and max per iter defaults

No hardcoded API key is used. Set:
  export DEEPSEEK_API_KEY="your_key"
Optional:
  export DEEPSEEK_MODEL="deepseek-chat"
  export DEEPSEEK_BASE_URL="https://api.deepseek.com"

Recommended usage
-----------------
Use with the unified scripts generated earlier:

  python unified_control_iteration_full.py \
    --dataset soccer \
    --train_script unified_train_ltr_full.py \
    --infer_script unified_infer_ltr_full.py \
    --build_infer_features_script unified_build_infer_features_whitelist_full.py

To avoid calling LLM and just run train+infer iterations:

  python unified_control_iteration_full.py --dataset soccer --skip_llm_relabel 1

Ablation examples:
  --train_extra_args "--disable_dataset_derived_features 1"
  --infer_extra_args "--disable_gate 1"
  --infer_extra_args "--disable_dataset_gate 1"
"""

import argparse
import json
import os
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

import pandas as pd


# ============================================================
# 1. Dataset configs
# ============================================================

@dataclass
class DatasetConfig:
    name: str
    default_candidate_features: str
    default_allowed_cells_csv: str
    default_initial_pairwise_responses: str
    default_train_script: str
    default_infer_script: str
    default_build_infer_features_script: str
    default_work_dir: str
    default_global_min_margin: float = 0.06
    default_epochs: int = 20
    default_batch_size: int = 256
    default_lr: float = 1e-3
    default_max_iterations: int = 3
    default_max_hard_cases_per_iter: int = 200
    default_llm_max_per_iter: int = 200
    source_tag: str = "iterative_hard_case_llm"
    system_prompt: str = "你是一个严谨的表格数据清洗排序专家。你必须只输出 JSON。"
    dataset_intro: str = "你现在在做表格数据清洗中的错误修复排序任务。"
    reason_types: List[str] = field(default_factory=lambda: [
        "rule_consistency",
        "context_match",
        "cooccurrence_support",
        "spelling_similarity",
        "pattern_match",
        "uncertain",
        "other",
    ])
    aliases: Dict[str, str] = field(default_factory=dict)


COMMON_ALIASES = {
    "jounral_abbreviation": "journal_abbreviation",
    "marital-status": "maritalstatus",
    "marital_status": "maritalstatus",
    "hours-per-week": "hoursperweek",
    "hours_per_week": "hoursperweek",
    "native-country": "country",
    "native_country": "country",
    "birth_year": "birthyear",
    "birth year": "birthyear",
    "birth-year": "birthyear",
    "player_position": "position",
    "club": "team",
    "home_city": "city",
    "arena": "stadium",
    "coach": "manager",
    "season_year": "season",
    "year": "season",
}


DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "hospital": DatasetConfig(
        name="hospital",
        default_candidate_features="repair_candidate_features_v2.csv",
        default_allowed_cells_csv="allowed_cells_from_detection.csv",
        default_initial_pairwise_responses="repair_pairwise_responses_v2.jsonl",
        default_train_script="train_ltr.py",
        default_infer_script="infer_ltr.py",
        default_build_infer_features_script="build_infer_candidate_features_from_whitelist.py",
        default_work_dir="repair_iterative_workdir_widetrain_whitelistinfer",
        source_tag="iterative_hard_case_llm_hospital",
        system_prompt="你是一个严谨的 Hospital 表格数据清洗排序专家。你必须只输出 JSON。",
        dataset_intro="你现在在做 Hospital 表格数据清洗中的错误修复排序任务。",
        reason_types=[
            "rule_consistency", "context_match", "cooccurrence_support",
            "spelling_similarity", "pattern_match", "uncertain", "other",
        ],
    ),
    "flights": DatasetConfig(
        name="flights",
        default_candidate_features="repair_candidate_features_v2.csv",
        default_allowed_cells_csv="predicted_error_positions_consensus_augmented.csv",
        default_initial_pairwise_responses="repair_pairwise_responses_v2.jsonl",
        default_train_script="train_ltr.py",
        default_infer_script="infer_ltr.py",
        default_build_infer_features_script="build_infer_candidate_features_from_whitelist.py",
        default_work_dir="repair_iterative_workdir_flights",
        source_tag="iterative_hard_case_llm_flights",
        system_prompt="你是一个严谨的 Flights 表格数据清洗排序专家。你必须只输出 JSON。",
        dataset_intro="你现在在做 Flights 航班表格数据清洗中的错误修复排序任务。",
    ),
    "beers": DatasetConfig(
        name="beers",
        default_candidate_features="repair_candidate_features_v2.csv",
        default_allowed_cells_csv="predicted_error_positions.csv",
        default_initial_pairwise_responses="repair_pairwise_responses_v2.jsonl",
        default_train_script="train_ltr.py",
        default_infer_script="infer_ltr.py",
        default_build_infer_features_script="build_infer_candidate_features_from_whitelist.py",
        default_work_dir="repair_iterative_workdir_beers",
        source_tag="iterative_hard_case_llm_beers",
        system_prompt="你是一个严谨的 Beers 表格数据清洗排序专家。你必须只输出 JSON。",
        dataset_intro="你现在在做 Beers 表格数据清洗中的错误修复排序任务。",
        reason_types=[
            "rule_consistency", "context_match", "beers_format_match",
            "cooccurrence_support", "spelling_similarity", "pattern_match",
            "uncertain", "other",
        ],
    ),
    "rayyan": DatasetConfig(
        name="rayyan",
        default_candidate_features="repair_candidate_features_v2.csv",
        default_allowed_cells_csv="predicted_error_positions_consensus_augmented.csv",
        default_initial_pairwise_responses="repair_pairwise_responses_v2.jsonl",
        default_train_script="train_ltr.py",
        default_infer_script="infer_ltr.py",
        default_build_infer_features_script="build_infer_candidate_features_from_whitelist.py",
        default_work_dir="repair_iterative_workdir_rayyan",
        source_tag="iterative_hard_case_llm_rayyan",
        system_prompt="你是一个严谨的 Rayyan 文献元数据清洗排序专家。你必须只输出 JSON。",
        dataset_intro="你现在在做 Rayyan biomedical literature metadata 表格数据清洗中的错误修复排序任务。",
        reason_types=[
            "rule_consistency", "context_match", "rayyan_metadata_consistency",
            "canonical_format", "cooccurrence_support", "spelling_similarity",
            "pattern_match", "uncertain", "other",
        ],
    ),
    "adult": DatasetConfig(
        name="adult",
        default_candidate_features="repair_candidate_features_v2.csv",
        default_allowed_cells_csv="predicted_error_positions_adult_augmented.csv",
        default_initial_pairwise_responses="repair_pairwise_responses_v2.jsonl",
        default_train_script="train_ltr.py",
        default_infer_script="infer_ltr.py",
        default_build_infer_features_script="build_infer_candidate_features_from_whitelist.py",
        default_work_dir="repair_iterative_workdir_adult",
        source_tag="iterative_hard_case_llm_adult",
        system_prompt="你是一个严谨的 Adult 表格数据清洗排序专家。你必须只输出 JSON。",
        dataset_intro="你现在在做 Adult census 表格数据清洗中的错误修复排序任务。",
        reason_types=[
            "rule_consistency", "adult_profile_consistency", "domain_validity",
            "relationship_consistency", "education_age_consistency",
            "context_match", "cooccurrence_support", "spelling_similarity",
            "pattern_match", "uncertain", "other",
        ],
    ),
    "soccer": DatasetConfig(
        name="soccer",
        default_candidate_features="repair_candidate_features_v2.csv",
        default_allowed_cells_csv="predicted_error_positions_soccer_augmented.csv",
        default_initial_pairwise_responses="repair_pairwise_responses_v2.jsonl",
        default_train_script="train_ltr.py",
        default_infer_script="infer_ltr.py",
        default_build_infer_features_script="build_infer_candidate_features_from_whitelist.py",
        default_work_dir="repair_iterative_workdir_soccer",
        source_tag="iterative_hard_case_llm_soccer",
        system_prompt="你是一个严谨的 Soccer 表格数据清洗排序专家。你必须只输出 JSON。",
        dataset_intro="你现在在做 Soccer 表格数据清洗中的错误修复排序任务。",
        default_llm_max_per_iter=200,
        reason_types=[
            "rule_consistency", "soccer_profile_consistency", "domain_validity",
            "position_canonicalization", "year_age_consistency",
            "team_context_consistency", "context_match", "cooccurrence_support",
            "spelling_similarity", "pattern_match", "uncertain", "other",
        ],
    ),
}


# ============================================================
# 2. Basic utilities
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Unified iterative repair controller")

    p.add_argument("--dataset", required=True, choices=sorted(DATASET_CONFIGS.keys()))

    p.add_argument("--candidate_features", default=None)
    p.add_argument("--allowed_cells_csv", default=None)
    p.add_argument("--initial_pairwise_responses", default=None)

    p.add_argument("--train_script", default=None)
    p.add_argument("--infer_script", default=None)
    p.add_argument("--build_infer_features_script", default=None)

    p.add_argument("--work_dir", default=None)
    p.add_argument("--max_iterations", type=int, default=None)
    p.add_argument("--max_hard_cases_per_iter", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--global_min_margin", type=float, default=None)

    p.add_argument("--llm_max_per_iter", type=int, default=None)
    p.add_argument("--llm_max_workers", type=int, default=1)
    p.add_argument("--skip_llm_relabel", type=int, default=0)

    # LLM settings.
    p.add_argument("--model_name", default=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))
    p.add_argument("--base_url", default=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_retries", type=int, default=3)
    p.add_argument("--retry_sleep_seconds", type=float, default=2.0)

    # Extra args forwarded to subprocesses.
    p.add_argument("--train_extra_args", default="")
    p.add_argument("--infer_extra_args", default="")
    p.add_argument("--build_extra_args", default="")

    # Use unified script calling convention.
    p.add_argument("--use_unified_scripts", type=int, default=1)

    # Debug / safety.
    p.add_argument("--dry_run", type=int, default=0)
    p.add_argument("--continue_on_llm_error", type=int, default=1)
    p.add_argument("--stop_if_no_new_labels", type=int, default=1)
    p.add_argument("--force_rebuild_infer_features", type=int, default=1)

    return p.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                print(f"[WARN] skip bad jsonl line: {path}, line={line_no}, err={e}")
    return rows


def write_jsonl(path: Path, rows: List[dict]):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def dedup_pairwise_rows(rows: List[dict]) -> List[dict]:
    dedup = {}
    for x in rows:
        try:
            row_id = int(x.get("row_id", -1))
        except Exception:
            row_id = -1
        col = str(x.get("column", ""))
        a = str(x.get("candidate_a_value", x.get("a_value", "")))
        b = str(x.get("candidate_b_value", x.get("b_value", "")))
        # Treat A/B order as directional because winner is attached to A/B.
        key = (row_id, col, a, b)
        dedup[key] = x
    return list(dedup.values())


def run_subprocess(cmd: List[str], env: Optional[dict] = None, dry_run: bool = False):
    print("[RUN]", " ".join(str(x) for x in cmd))
    if dry_run:
        return
    subprocess.run([str(x) for x in cmd], check=True, env=env)


def resolve_script(script: str, base_dir: Path) -> Path:
    p = Path(script)
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    candidate = base_dir / p
    return candidate.resolve()


def parse_extra_args(s: str) -> List[str]:
    if not s:
        return []
    return shlex.split(s)


def copy_if_found_to_workdir(expected_path: Path, search_paths: List[Path]) -> bool:
    if expected_path.exists():
        return True
    for p in search_paths:
        if p.exists():
            ensure_dir(expected_path.parent)
            shutil.copy2(p, expected_path)
            print(f"[INFO] copied {p} -> {expected_path}")
            return True
    return False


# ============================================================
# 3. Prompt construction
# ============================================================

def normalize_col_name(column: Any) -> str:
    c = str(column).strip().lower()
    return COMMON_ALIASES.get(c, c)


def build_column_specific_instruction(dataset: str, column: str) -> str:
    col_raw = str(column)
    col = normalize_col_name(col_raw)

    if dataset == "hospital":
        if col_raw in {"MeasureCode", "Stateavg"} or col in {"measurecode", "stateavg"}:
            return "本列是结构化编码列。优先看 family/prefix/suffix 是否一致，以及它是否与同行的 MeasureCode / Stateavg / Condition 保持结构一致；不要只因为字符串相似就选择 sibling code。"
        if col_raw == "Condition" or col == "condition":
            return "本列是疾病/主题分类列。优先看它是否与 MeasureCode family 一致：hf->heart failure, ami->heart attack, pn->pneumonia, scip->surgical infection prevention。"
        if col_raw == "MeasureName" or col == "measurename":
            return "本列是长文本 canonical phrase。优先看它是否与 MeasureCode 和 Condition 一致，不要只因为词面相似就选择语义不匹配的候选。"
        if col_raw == "Score" or col == "score":
            return "本列是百分比。已经是合法百分号格式的原值要非常谨慎改动；优先恢复 malformed percent（例如 95x -> 95%），不要无根据把合法分数改成别的合法分数。"
        if col_raw == "Sample" or col == "sample":
            return "本列通常应符合 '\\d+ patients' 模式。优先恢复 pattern typo，不要随意补空值，也不要选择不符合样式的候选。"

    if dataset == "flights":
        if col in {"sched_dep_time", "act_dep_time", "sched_arr_time", "act_arr_time"} or col.endswith("_time"):
            return "本列是航班时间列。优先选择合法时间格式；优先相信规则强支持、可信多源 flight consensus 高置信支持、与 flight 和同一行实际/计划起降时间一致的候选；不要仅因为列内高频就选择。"

    if dataset == "beers":
        if col == "state":
            return "本列是美国州缩写列。优先合法州缩写，并参考 city 后缀、brewery_id、brewery_name 对 state 的支持。"
        if col == "city":
            return "本列是城市列。优先考虑 city/state 格式恢复，例如 Ashland OR 应优先恢复为 Ashland，并检查 state/brewery 上下文。"
        if col == "ounces":
            return "本列是容量列。优先 numeric-unit canonicalization，例如 16 oz 应规范为 16.0 oz.，同时参考 beer_name/style。"
        if col == "abv":
            return "本列是 abv，通常为 0 到 1 之间的小数。6.3% 通常应规范为 0.063；0.09% 应修成 0.09。"
        if col == "ibu":
            return "本列是 IBU 数值。优先合法数值和格式规范化，例如 35 ibu -> 35，同时参考 beer_name/style。"
        if col in {"brewery_id", "brewery_name"}:
            return "本列是 brewery 相关属性。优先选择与 brewery_id/brewery_name/city/state 一致的候选。"
        if col in {"beer_name", "style"}:
            return "本列是 beer_name/style。优先选择与 id、style、ounces、abv、ibu 上下文一致的候选。"

    if dataset == "rayyan":
        if col == "article_language":
            return "本列是文章语言列。优先选择 eng/fre/ger/spa 等规范三字母语言值；重点参考 journal metadata 对 language 的支持。"
        if col == "journal_issn":
            return "本列是期刊 ISSN。优先选择合法 ISSN 格式 ####-#### 或 ####-###X；候选通过校验位或来自 journal_title_to_issn 更可信。"
        if col == "journal_title":
            return "本列是期刊全名。优先选择与 journal_issn、journal_abbreviation、article_language 一致的候选；不要仅因为列内高频或字符串相似就选择。"
        if col == "journal_abbreviation":
            return "本列是期刊缩写/别名。优先选择与 journal_issn、journal_title、article_language 一致的候选。"
        if col == "article_jcreated_at":
            return "本列是日期列。优先选择可规范化为日期的候选，例如 2005-01-01 -> 1/1/05；candidate 来自 canonical_date_restore 时更可信。"
        if col == "article_jvolumn":
            return "本列是期刊卷号。优先选择合法整数卷号，并参考 journal_title + article_jissue 的一致性。"
        if col == "article_jissue":
            return "本列是期刊期号。优先选择合法整数期号，并参考 journal_title + article_jvolumn 的一致性。"
        if col == "article_pagination":
            return "本列是页码范围。优先选择格式规范页码，例如 1442–6 -> 1442-6；候选来自 canonical_pagination_restore 时更可信。"
        if col == "article_title":
            return "本列是文章标题。优先选择与 id、journal metadata、language 上下文一致的标题。"

    if dataset == "adult":
        if col == "relationship":
            return "本列是 relationship。优先选择与 sex、maritalstatus、age 一致的候选，例如 Husband->Male 且已婚，Wife->Female 且已婚。"
        if col == "sex":
            return "本列是 sex。relationship=Husband 强支持 Male，relationship=Wife 强支持 Female。"
        if col == "maritalstatus":
            return "本列是 maritalstatus。若 relationship 是 Husband/Wife，通常应为 Married-civ-spouse 或 Married-AF-spouse。"
        if col == "education":
            return "本列是 education。注意 age 与 education 的合理性，年轻人不应无强证据地出现过高学历。"
        if col == "age":
            return "本列是 age。优先合法年龄/年龄桶，不要凭统计高频强行修改。"
        if col == "hoursperweek":
            return "本列是 hoursperweek。优先合法数值或明显格式修复，不要只凭 profile 高频修改。"
        if col == "income":
            return "本列是 income。优先规范化 <=50K/>50K 到 LessThan50K/MoreThan50K，不要凭职业学历猜测收入。"

    if dataset == "soccer":
        if col == "position":
            return "本列是 position。优先选择合法足球位置；如果候选包含 team_season_position 或 position_profile_strong，应优先于单纯 domain 候选。"
        if col == "birthyear":
            return "本列是 birthyear。优先合法出生年份，并检查 season - birthyear 通常在 15 到 48 岁之间。"
        if col == "season":
            return "本列是 season。优先合法赛季年份，并检查 season - birthyear 通常在 15 到 48 岁之间。"
        if col in {"team", "city", "stadium", "manager"}:
            return "本列是球队上下文字段。优先考虑 team-season、team、stadium-city 的一致性，但不要仅凭全局高频修改。"
        if col in {"name", "surname", "birthplace"}:
            return "本列是文本实体字段。优先考虑 entity_pair_majority/profile_key_majority/candidate_entity_safe_fill；不要凭 column_top/global_dictionary 高频候选修复。"

    return "优先考虑规则一致性、同行上下文一致性、统计支持和合理的拼写/模式变换。"


def build_pair_prompt(cfg: DatasetConfig, obj: dict) -> str:
    column_instruction = build_column_specific_instruction(cfg.name, obj.get("column", ""))
    reason_types = " / ".join(cfg.reason_types)

    dataset_extra = ""
    if cfg.name == "adult":
        dataset_extra = (
            "Adult 字段包括：age, workclass, education, maritalstatus, occupation, "
            "relationship, race, sex, hoursperweek, country, income。\n"
            "特别注意 relationship/sex/maritalstatus/age/education 的一致性。"
        )
    elif cfg.name == "soccer":
        dataset_extra = (
            "Soccer 字段包括：name, surname, birthyear, birthplace, position, team, city, stadium, season, manager。\n"
            "position 必须合法；birthyear/season 要满足年龄关系；name/surname/birthplace 不要凭高频猜测。"
        )
    elif cfg.name == "rayyan":
        dataset_extra = (
            "Rayyan 字段是 biomedical literature metadata。特别注意 language / ISSN / date / pagination 的 canonical 格式，"
            "以及 journal/article metadata consistency。"
        )
    elif cfg.name == "beers":
        dataset_extra = (
            "Beers 字段包括 state, city, ounces, abv, ibu, brewery_id, brewery_name, beer_name, style。"
            "特别注意 state-city、brewery profile、ABV/IBU/ounces 格式合法性。"
        )
    elif cfg.name == "flights":
        dataset_extra = (
            "Flights 字段中时间列必须是可解析时间，并与 flight/source consensus 及起降时间关系一致。"
        )
    elif cfg.name == "hospital":
        dataset_extra = (
            "Hospital 字段中 MeasureCode/Stateavg/Condition/MeasureName 需要结构一致，Score/Sample 需要格式一致。"
        )

    prompt = f"""
{cfg.dataset_intro}

目标：
对于一个被检测为错误的单元格，请在两个候选修复值 A 和 B 中判断谁更适合作为最终修复值。
你必须只在 A、B、Unknown 中三选一：
- A：候选 A 更好
- B：候选 B 更好
- Unknown：无法可靠判断，或者两个都不够好

数据集说明：
{dataset_extra}

通用判断原则：
1. 优先考虑强规则证据和 expected values。
2. 再考虑同行上下文一致性、profile/metadata/consensus 支持。
3. 再考虑 domain / format / canonicalization 合法性。
4. 列内高频、全局字典、字符串相似通常是弱证据，不能单独压过上下文。
5. 如果两个候选都明显不可靠，输出 Unknown。

本列专属判断：
{column_instruction}

请严格输出 JSON，不要输出额外解释文字。
JSON 格式如下：
{{
  "winner": "A" 或 "B" 或 "Unknown",
  "reason_type": "{reason_types}",
  "reason": "一句简短原因"
}}

样本：
{json.dumps(obj, ensure_ascii=False, indent=2)}
""".strip()
    return prompt


def parse_llm_json(text: str) -> dict:
    text = str(text).strip()
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise ValueError(f"Invalid JSON output: {text[:500]}")


def get_client(args):
    if OpenAI is None:
        raise ImportError("openai package is not installed.")
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise ValueError("Missing DEEPSEEK_API_KEY.")
    return OpenAI(api_key=api_key, base_url=args.base_url)


def call_llm_pairwise(client, cfg: DatasetConfig, obj: dict, args) -> dict:
    prompt = build_pair_prompt(cfg, obj)
    for attempt in range(1, args.max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=args.model_name,
                temperature=args.temperature,
                messages=[
                    {"role": "system", "content": cfg.system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
            text = resp.choices[0].message.content.strip()
            out = parse_llm_json(text)
            return {
                "winner": normalize_winner(out.get("winner", "Unknown")),
                "reason_type": out.get("reason_type", "uncertain"),
                "reason": out.get("reason", ""),
                "raw_response": text,
            }
        except Exception as e:
            if attempt == args.max_retries:
                if args.continue_on_llm_error:
                    return {
                        "winner": "Unknown",
                        "reason_type": "uncertain",
                        "reason": f"LLM call failed: {str(e)}",
                        "raw_response": "",
                    }
                raise
            time.sleep(args.retry_sleep_seconds)


def normalize_winner(winner: Any) -> str:
    w = str(winner).strip()
    if w in {"A", "B", "Unknown"}:
        return w
    wu = w.upper()
    if wu in {"A", "CANDIDATE_A", "CANDIDATE A"}:
        return "A"
    if wu in {"B", "CANDIDATE_B", "CANDIDATE B"}:
        return "B"
    return "Unknown"


# ============================================================
# 4. Controller steps
# ============================================================

def bootstrap_pairwise_file(initial_pairwise: Path, accum_pairwise: Path, work_dir: Path):
    ensure_dir(work_dir)
    ensure_dir(accum_pairwise.parent)
    if accum_pairwise.exists():
        rows = load_jsonl(accum_pairwise)
        print(f"[INFO] accumulated pairwise already exists: {accum_pairwise}, count={len(rows)}")
        return

    init_rows = load_jsonl(initial_pairwise)
    init_rows = dedup_pairwise_rows(init_rows)
    write_jsonl(accum_pairwise, init_rows)
    print(f"[INFO] Initialized accumulated pairwise file: {accum_pairwise}")
    print(f"[INFO] Initial pairwise count: {len(init_rows)}")


def build_infer_features(
    cfg: DatasetConfig,
    build_script: Path,
    candidate_features: Path,
    allowed_cells_csv: Path,
    infer_features_path: Path,
    args,
):
    ensure_dir(infer_features_path.parent)

    if infer_features_path.exists() and not args.force_rebuild_infer_features:
        print(f"[INFO] infer features already exist, skip rebuild: {infer_features_path}")
        return

    env = os.environ.copy()
    env["INPUT_CANDIDATE_FEATURES"] = str(candidate_features)
    env["INPUT_ALLOWED_CELLS"] = str(allowed_cells_csv)
    env["OUTPUT_INFER_FEATURES"] = str(infer_features_path)

    if args.use_unified_scripts:
        cmd = [
            "python", str(build_script),
            "--dataset", cfg.name,
            "--candidate_features", str(candidate_features),
            "--allowed_cells", str(allowed_cells_csv),
            "--output", str(infer_features_path),
        ]
    else:
        cmd = [
            "python", str(build_script),
            "--candidate_features", str(candidate_features),
            "--allowed_cells", str(allowed_cells_csv),
            "--output", str(infer_features_path),
        ]

    cmd += parse_extra_args(args.build_extra_args)
    run_subprocess(cmd, env=env, dry_run=bool(args.dry_run))

    if args.dry_run:
        return

    if infer_features_path.exists():
        print(f"[INFO] Found infer whitelist features at expected path: {infer_features_path}")
        return

    expected_name = infer_features_path.name
    candidate_paths = [
        candidate_features.resolve().parent / expected_name,
        Path.cwd().resolve() / expected_name,
        build_script.resolve().parent / expected_name,
    ]

    if copy_if_found_to_workdir(infer_features_path, candidate_paths):
        return

    searched = "\n".join(str(p) for p in candidate_paths)
    raise FileNotFoundError(
        "Expected infer whitelist features file not found after build step.\n"
        f"Expected: {infer_features_path}\n"
        f"Also searched:\n{searched}"
    )


def train_model(
    cfg: DatasetConfig,
    train_script: Path,
    infer_features_path: Path,
    accum_pairwise: Path,
    model_dir: Path,
    args,
):
    ensure_dir(model_dir)

    if args.use_unified_scripts:
        cmd = [
            "python", str(train_script),
            "--dataset", cfg.name,
            "--candidate_features", str(infer_features_path),
            "--pairwise_responses", str(accum_pairwise),
            "--output_dir", str(model_dir),
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--lr", str(args.lr),
        ]
    else:
        cmd = [
            "python", str(train_script),
            "--candidate_features", str(infer_features_path),
            "--pairwise_responses", str(accum_pairwise),
            "--output_dir", str(model_dir),
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--lr", str(args.lr),
        ]

    cmd += parse_extra_args(args.train_extra_args)
    run_subprocess(cmd, dry_run=bool(args.dry_run))


def run_inference(
    cfg: DatasetConfig,
    infer_script: Path,
    infer_features_path: Path,
    allowed_cells_csv: Path,
    model_dir: Path,
    args,
):
    ensure_dir(model_dir)

    env = os.environ.copy()
    # Keep all outputs in model_dir, matching original iterative controllers.
    env["REPAIR_INFER_OUTPUT_DIR"] = str(model_dir)

    if args.use_unified_scripts:
        cmd = [
            "python", str(infer_script),
            "--dataset", cfg.name,
            "--candidate_features", str(infer_features_path),
            "--allowed_cells_csv", str(allowed_cells_csv),
            "--model_dir", str(model_dir),
            "--global_min_margin", str(args.global_min_margin),
            "--max_hard_cases", str(args.max_hard_cases_per_iter),
        ]
    else:
        cmd = [
            "python", str(infer_script),
            "--candidate_features", str(infer_features_path),
            "--allowed_cells_csv", str(allowed_cells_csv),
            "--model_dir", str(model_dir),
            "--global_min_margin", str(args.global_min_margin),
            "--max_hard_cases", str(args.max_hard_cases_per_iter),
        ]

    cmd += parse_extra_args(args.infer_extra_args)
    run_subprocess(cmd, env=env, dry_run=bool(args.dry_run))


def label_hard_cases_with_llm(
    cfg: DatasetConfig,
    hard_cases_path: Path,
    llm_new_responses: Path,
    args,
) -> List[dict]:
    if args.skip_llm_relabel:
        print("[INFO] skip_llm_relabel=1, skip hard-case LLM relabeling.")
        write_jsonl(llm_new_responses, [])
        return []

    hard_cases = load_jsonl(hard_cases_path)
    if not hard_cases:
        print("[INFO] No hard cases found, skip LLM relabeling.")
        write_jsonl(llm_new_responses, [])
        return []

    limit = min(
        int(args.max_hard_cases_per_iter),
        int(args.llm_max_per_iter),
        len(hard_cases),
    )
    selected = hard_cases[:limit]
    print(f"[INFO] Hard cases found: {len(hard_cases)}, selected for LLM: {len(selected)}")

    client = get_client(args)

    new_rows = []
    for i, obj in enumerate(selected, start=1):
        llm_out = call_llm_pairwise(client, cfg, obj, args)
        winner = normalize_winner(llm_out.get("winner", "Unknown"))

        row = {
            "row_id": int(obj.get("row_id", -1)),
            "column": str(obj.get("column", "")),
            "dirty_value": str(obj.get("dirty_value", "")),
            "candidate_a_value": str(obj.get("candidate_a_value", "")),
            "candidate_b_value": str(obj.get("candidate_b_value", "")),
            "winner": winner,
            "reason_type": llm_out.get("reason_type", "uncertain"),
            "reason": llm_out.get("reason", ""),
            "source": cfg.source_tag,
        }
        if "raw_response" in llm_out:
            row["raw_response"] = llm_out["raw_response"]
        new_rows.append(row)

        if i % 20 == 0 or i == len(selected):
            print(f"[INFO] LLM hard-case progress: {i}/{len(selected)}")

    write_jsonl(llm_new_responses, new_rows)
    print(f"[INFO] New LLM hard-case labels saved: {llm_new_responses}, count={len(new_rows)}")
    return new_rows


def merge_new_pairwise_rows(accum_pairwise: Path, new_rows: List[dict]):
    old_rows = load_jsonl(accum_pairwise)
    merged = dedup_pairwise_rows(old_rows + new_rows)
    write_jsonl(accum_pairwise, merged)
    print(f"[INFO] Accumulated pairwise count: {len(merged)}")


def write_iteration_summary(summary_path: Path, rows: List[dict]):
    ensure_dir(summary_path.parent)
    pd.DataFrame(rows).to_csv(summary_path, index=False, encoding="utf-8-sig")


# ============================================================
# 5. Main loop
# ============================================================

def main():
    args = parse_args()
    cfg = DATASET_CONFIGS[args.dataset]

    # Fill defaults from config.
    candidate_features = Path(args.candidate_features or cfg.default_candidate_features)
    allowed_cells_csv = Path(args.allowed_cells_csv or cfg.default_allowed_cells_csv)
    initial_pairwise = Path(args.initial_pairwise_responses or cfg.default_initial_pairwise_responses)

    work_dir = Path(args.work_dir or cfg.default_work_dir)
    model_dir = work_dir / "model_output"
    accum_pairwise = work_dir / "pairwise_responses_accumulated.jsonl"
    llm_new_responses = work_dir / "pairwise_responses_new_from_llm.jsonl"
    hard_cases_path = model_dir / "hard_cases_for_llm.jsonl"
    infer_features_path = work_dir / "repair_candidate_features_infer_whitelist.csv"
    summary_path = work_dir / "iteration_summary.csv"
    config_path = work_dir / "controller_run_config.json"

    args.max_iterations = args.max_iterations if args.max_iterations is not None else cfg.default_max_iterations
    args.max_hard_cases_per_iter = args.max_hard_cases_per_iter if args.max_hard_cases_per_iter is not None else cfg.default_max_hard_cases_per_iter
    args.llm_max_per_iter = args.llm_max_per_iter if args.llm_max_per_iter is not None else cfg.default_llm_max_per_iter
    args.epochs = args.epochs if args.epochs is not None else cfg.default_epochs
    args.batch_size = args.batch_size if args.batch_size is not None else cfg.default_batch_size
    args.lr = args.lr if args.lr is not None else cfg.default_lr
    args.global_min_margin = args.global_min_margin if args.global_min_margin is not None else cfg.default_global_min_margin

    script_base = Path(__file__).resolve().parent
    train_script = resolve_script(args.train_script or cfg.default_train_script, script_base)
    infer_script = resolve_script(args.infer_script or cfg.default_infer_script, script_base)
    build_script = resolve_script(args.build_infer_features_script or cfg.default_build_infer_features_script, script_base)

    ensure_dir(work_dir)
    ensure_dir(model_dir)

    run_config = {
        "dataset": cfg.name,
        "candidate_features": str(candidate_features),
        "allowed_cells_csv": str(allowed_cells_csv),
        "initial_pairwise_responses": str(initial_pairwise),
        "train_script": str(train_script),
        "infer_script": str(infer_script),
        "build_infer_features_script": str(build_script),
        "work_dir": str(work_dir),
        "model_dir": str(model_dir),
        "accum_pairwise": str(accum_pairwise),
        "infer_features_path": str(infer_features_path),
        "max_iterations": args.max_iterations,
        "max_hard_cases_per_iter": args.max_hard_cases_per_iter,
        "llm_max_per_iter": args.llm_max_per_iter,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "global_min_margin": args.global_min_margin,
        "skip_llm_relabel": args.skip_llm_relabel,
        "use_unified_scripts": args.use_unified_scripts,
        "train_extra_args": args.train_extra_args,
        "infer_extra_args": args.infer_extra_args,
        "build_extra_args": args.build_extra_args,
    }
    write_json(config_path, run_config)

    print("========== Unified Iterative Repair Controller ==========")
    for k, v in run_config.items():
        print(f"{k}: {v}")

    # Check scripts exist unless dry-run.
    if not args.dry_run:
        for desc, p in [("train_script", train_script), ("infer_script", infer_script), ("build_infer_features_script", build_script)]:
            if not p.exists():
                raise FileNotFoundError(f"{desc} not found: {p}")
        if not candidate_features.exists():
            raise FileNotFoundError(f"candidate_features not found: {candidate_features}")
        if not allowed_cells_csv.exists():
            raise FileNotFoundError(f"allowed_cells_csv not found: {allowed_cells_csv}")
        if not initial_pairwise.exists() and not accum_pairwise.exists():
            raise FileNotFoundError(f"initial_pairwise_responses not found: {initial_pairwise}")

    bootstrap_pairwise_file(initial_pairwise, accum_pairwise, work_dir)

    iteration_rows = []

    for it in range(1, int(args.max_iterations) + 1):
        print(f"\n========== Iteration {it}/{args.max_iterations} ==========")
        iter_start = time.time()

        print("\n[Step 1] Build whitelist inference features")
        build_infer_features(
            cfg=cfg,
            build_script=build_script,
            candidate_features=candidate_features,
            allowed_cells_csv=allowed_cells_csv,
            infer_features_path=infer_features_path,
            args=args,
        )

        print("\n[Step 2] Train pairwise LTR student")
        train_model(
            cfg=cfg,
            train_script=train_script,
            infer_features_path=infer_features_path,
            accum_pairwise=accum_pairwise,
            model_dir=model_dir,
            args=args,
        )

        print("\n[Step 3] Run inference")
        run_inference(
            cfg=cfg,
            infer_script=infer_script,
            infer_features_path=infer_features_path,
            allowed_cells_csv=allowed_cells_csv,
            model_dir=model_dir,
            args=args,
        )

        print("\n[Step 4] Label hard cases with LLM")
        new_rows = []
        if not args.dry_run:
            new_rows = label_hard_cases_with_llm(
                cfg=cfg,
                hard_cases_path=hard_cases_path,
                llm_new_responses=llm_new_responses,
                args=args,
            )
        else:
            print("[DRY RUN] skip LLM labeling")

        print("\n[Step 5] Merge new labels")
        if not args.dry_run:
            merge_new_pairwise_rows(accum_pairwise, new_rows)

        hard_case_count = len(load_jsonl(hard_cases_path)) if hard_cases_path.exists() else 0
        accum_count = len(load_jsonl(accum_pairwise)) if accum_pairwise.exists() else 0
        elapsed = time.time() - iter_start

        iter_row = {
            "iteration": it,
            "hard_cases_before_llm": hard_case_count,
            "new_llm_labels": len(new_rows),
            "accum_pairwise_count": accum_count,
            "elapsed_seconds": elapsed,
            "model_dir": str(model_dir),
            "infer_features_path": str(infer_features_path),
        }
        iteration_rows.append(iter_row)
        write_iteration_summary(summary_path, iteration_rows)
        print(f"[INFO] iteration summary saved: {summary_path}")
        print(f"[INFO] iteration elapsed seconds: {elapsed:.2f}")

        if int(args.stop_if_no_new_labels) == 1 and len(new_rows) == 0:
            print("[INFO] No new labels generated; stop iterations.")
            break

    print("\n========== Done ==========")
    print(f"work_dir: {work_dir}")
    print(f"model_dir: {model_dir}")
    print(f"accum_pairwise: {accum_pairwise}")
    print(f"infer_features: {infer_features_path}")
    print(f"top1 repairs: {model_dir / 'inference_top1_repairs.csv'}")
    print(f"iteration_summary: {summary_path}")


if __name__ == "__main__":
    main()

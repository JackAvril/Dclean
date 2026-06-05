#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Soccer repair candidate generation.

This script is adapted from the Adult repair candidate generator, but replaces
Adult-specific logic with Soccer-specific candidate generation based on:

  - expected values from rule/conflict evidence
  - Soccer profile consistency evidence
  - position domain / canonicalization
  - birthyear / season numeric-year repair and age-at-season constraints
  - team / season / stadium / city / manager distribution evidence
  - neighbor and similar-row evidence
  - column top values with conservative scoring
  - fallback dictionaries from soccer_dirty.csv

Inputs
------
1. predicted_error_positions.csv
   At least contains row_id,column,value.
   It may also contain Soccer feature columns such as:
     year_value, canonical_position, domain_valid, soccer_value_bucket,
     row_birthyear_int, row_season_int, row_age_at_season,
     row_team, row_city, row_stadium, row_manager,
     soccer_evidence_flags, soccer_attribution_bucket, etc.

2. candidate_llm_contexts_soccer.jsonl
   Rich context records. Each record may contain:
     row_context
     column_context
     conflict_context
     similarity_context
     soccer_consistency_context
     soccer_attribution_features
     bayesian_context

3. Optional fallback flat CSV.

4. Optional soccer_dirty.csv for column dictionaries and row fallback contexts.

Outputs
-------
repair_candidates_expanded.csv
repair_candidates_grouped.csv
predicted_error_positions_soccer_augmented.csv
soccer_candidate_generation_summary.csv

Usage
-----
cd /mnt/mydata/dq/projects/Splittree/test/Error\ Correction\ soccer

python revise_candidate_soccer.py \
  --error_pred_file predicted_error_positions.csv \
  --context_file "/mnt/mydata/dq/projects/Splittree/test/Error Detection Soccer/candidate_llm_contexts_soccer.jsonl" \
  --dirty_table_csv /mnt/mydata/dq/projects/Splittree/soccer/soccer_dirty.csv

If context path differs, use:
find /mnt/mydata/dq/projects/Splittree/test -path "*Soccer*" -name "*contexts*soccer*.jsonl"
"""

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd


# ============================================================
# 1. Defaults and parameters
# ============================================================

DEFAULT_ERROR_PRED_FILE = "predicted_error_positions.csv"
DEFAULT_CONTEXT_FILE = "/mnt/mydata/dq/projects/Splittree/test/Error Detection soccer/candidate_llm_contexts_soccer.jsonl"
DEFAULT_FALLBACK_CONTEXT_FLAT_FILE = "/mnt/mydata/dq/projects/Splittree/test/Error Detection soccer/candidate_llm_contexts_flat_soccer.csv"
DEFAULT_DIRTY_TABLE_CSV = "/mnt/mydata/dq/projects/Splittree/soccer/soccer_dirty.csv"

OUTPUT_CANDIDATES_EXPANDED_CSV = "repair_candidates_expanded.csv"
OUTPUT_CANDIDATES_GROUPED_CSV = "repair_candidates_grouped.csv"
OUTPUT_AUGMENTED_ALLOWED_CELLS_CSV = "predicted_error_positions_soccer_augmented.csv"
OUTPUT_SUMMARY_CSV = "soccer_candidate_generation_summary.csv"

TOP_K_COLUMN_VALUES = 12
TOP_K_SIMILAR_ROW_VALUES = 8
TOP_K_DISTRIBUTION_VALUES = 6
MAX_CANDIDATES_PER_CELL = 28
INCLUDE_DIRTY_VALUE = False

MIN_STRING_SIM_FOR_COLUMN_VALUE = 0.45
MIN_STRING_SIM_FOR_DOMAIN_VALUE = 0.34
MIN_STRING_SIM_FOR_CONTEXT_VALUE = 0.50

SCORE_DIRTY_VALUE = 0.05
SCORE_COLUMN_TOP = 0.22
SCORE_SIMILAR_ROW = 0.40
SCORE_NEIGHBOR_MAJORITY = 0.48
SCORE_RULE_EXPECTED_HIGH = 0.96
SCORE_RULE_EXPECTED_MED = 0.86
SCORE_SOCCER_PROFILE_TOP = 0.90
SCORE_SOCCER_PROFILE_SECOND = 0.74
SCORE_SOCCER_DIRECT_RULE = 0.92
SCORE_DOMAIN_NORMALIZE = 0.82
SCORE_YEAR_NORMALIZE = 0.84
SCORE_CONTEXT_DISTRIBUTION = 0.72
SCORE_CONTEXT_DISTRIBUTION_TOP = 0.88
SCORE_PROFILE_KEY_MAJORITY_TOP = 0.94
SCORE_PROFILE_KEY_MAJORITY_SECOND = 0.78
SCORE_ENTITY_PAIR_MAJORITY_TOP = 0.93
SCORE_ENTITY_PAIR_MAJORITY_SECOND = 0.76
SCORE_TEAM_SEASON_POSITION_TOP = 0.92

# Profile-key majority candidate generation thresholds.
# These are conservative because name/surname/birthplace are entity fields.
PROFILE_KEY_MIN_GROUP_SIZE = 2
PROFILE_KEY_MIN_TOP_COUNT = 2
PROFILE_KEY_MIN_RATIO = 0.70

ENTITY_PAIR_MIN_GROUP_SIZE = 2
ENTITY_PAIR_MIN_TOP_COUNT = 2
ENTITY_PAIR_MIN_RATIO = 0.75

POSITION_PROFILE_MIN_GROUP_SIZE = 2
POSITION_PROFILE_MIN_RATIO = 0.50

TOP_K_PROFILE_MAJORITY_VALUES = 4

EMPTY_TOKENS = {"", "empty", "null", "none", "nan", "na", "n/a", "missing", "?", "{null}", "none/null"}

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

POSITION_DOMAIN_VALUES = [
    "Defender",
    "Forward",
    "Goalkeeper",
    "Midfield",
    "Midfielder",
    "Striker",
    "Winger",
]

POSITION_CANONICAL_MAP = {
    "defender": "Defender",
    "defenders": "Defender",
    "defenders": "Defender",
    "defenderss": "Defender",
    "defenders.": "Defender",
    "defenders,": "Defender",
    "defenders ": "Defender",
    "defenders_": "Defender",
    "defenders-": "Defender",
    "defenders/": "Defender",
    "defenders\\": "Defender",
    "defenders'": "Defender",
    "defenders\"": "Defender",
    "defenders)": "Defender",
    "defenders(": "Defender",
    "defenders]": "Defender",
    "defenders[": "Defender",
    "defenders}": "Defender",
    "defenders{": "Defender",
    "defender": "Defender",
    "defenders": "Defender",
    "defenders": "Defender",
    "defenders": "Defender",
    "defeders": "Defender",
    "defeder": "Defender",
    "dfender": "Defender",
    "defnder": "Defender",
    "defenderes": "Defender",
    "defenders": "Defender",
    "defenders": "Defender",

    "forward": "Forward",
    "forwards": "Forward",
    "forwards": "Forward",
    "forwarder": "Forward",
    "forwad": "Forward",
    "foward": "Forward",

    "goalkeeper": "Goalkeeper",
    "goalkeepers": "Goalkeeper",
    "goal keeper": "Goalkeeper",
    "goal_keeper": "Goalkeeper",
    "gk": "Goalkeeper",

    "midfield": "Midfield",
    "midfields": "Midfield",
    "midfielder": "Midfield",
    "midfielders": "Midfield",
    "midfied": "Midfield",
    "midfiled": "Midfield",

    "striker": "Striker",
    "strikers": "Striker",
    "winger": "Winger",
    "wingers": "Winger",
}

PRIMARY_REPAIR_COLUMNS = {"birthyear", "season", "position"}
CONTEXT_REPAIR_COLUMNS = {"team", "city", "stadium", "manager", "birthplace"}
TEXT_ID_COLUMNS = {"name", "surname", "team", "city", "stadium", "manager", "birthplace"}
YEAR_COLUMNS = {"birthyear", "season"}


# ============================================================
# 2. Basic helpers
# ============================================================

def norm_text(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip().lower()


def canonical_empty_text(x: Any) -> str:
    s = norm_text(x)
    if s in EMPTY_TOKENS:
        return "empty"
    if x is None:
        return "empty"
    try:
        if pd.isna(x):
            return "empty"
    except Exception:
        pass
    return str(x).strip()


def is_empty_like_value(x: Any) -> bool:
    return norm_text(x) in EMPTY_TOKENS


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, float) and math.isnan(x):
            return default
        s = str(x).strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(float(str(x).strip()))
    except Exception:
        return default


def path_exists_nonempty(path: str) -> bool:
    return path is not None and str(path).strip() != "" and os.path.exists(str(path))


def resolve_output_path(output_path: str) -> str:
    output_path = str(output_path)
    if os.path.isabs(output_path):
        return output_path
    return os.path.abspath(output_path)


def resolve_file_path(path: str, desc: str, required: bool = False) -> str:
    """
    Use a single explicit file path.
    No path candidate list is used in this Soccer version.
    """
    path = str(path or "").strip()
    if not path:
        if required:
            raise FileNotFoundError(f"{desc} 路径为空，请在配置区或命令行参数中指定。")
        return ""

    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(f"{desc} 不存在: {path}")
        print(f"[WARN] {desc} 不存在，将跳过: {path}")
        return ""

    if os.path.isdir(path):
        raise IsADirectoryError(f"{desc} 应该是文件路径，不应该是目录: {path}")

    return path


def string_similarity(a: Any, b: Any) -> float:
    a = norm_text(a)
    b = norm_text(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def normalize_col_name(column: Any) -> str:
    c = norm_text(column)
    aliases = {
        "birth_year": "birthyear",
        "birth year": "birthyear",
        "birth-year": "birthyear",
        "birth_year_int": "birthyear",
        "year": "season",
        "season_year": "season",
        "player_position": "position",
        "club": "team",
        "football_club": "team",
        "home_city": "city",
        "arena": "stadium",
        "coach": "manager",
    }
    return aliases.get(c, c)


def canonical_position(value: Any) -> str:
    raw = canonical_empty_text(value)
    if raw == "empty":
        return "empty"

    s = norm_text(raw)
    s2 = re.sub(r"[^a-z]", "", s)
    if s in POSITION_CANONICAL_MAP:
        return POSITION_CANONICAL_MAP[s]
    if s2 in POSITION_CANONICAL_MAP:
        return POSITION_CANONICAL_MAP[s2]

    for v in POSITION_DOMAIN_VALUES:
        if norm_text(v) == s:
            return v

    # Fuzzy canonicalization for close typos only.
    best = None
    best_sim = 0.0
    for v in ["Defender", "Forward", "Goalkeeper", "Midfield", "Striker", "Winger"]:
        sim = string_similarity(s, v)
        if sim > best_sim:
            best_sim = sim
            best = v
    if best is not None and best_sim >= 0.72:
        return best

    return raw


def parse_year_like(value: Any) -> Optional[int]:
    s = canonical_empty_text(value)
    if s == "empty":
        return None

    # Common OCR/typo replacements in years.
    normalized = (
        s.strip()
        .replace("O", "0")
        .replace("o", "0")
        .replace("I", "1")
        .replace("l", "1")
        .replace("|", "1")
        .replace("S", "5")
        .replace("s", "5")
        .replace("B", "8")
    )

    m = re.search(r"(\d{4})", normalized)
    if m:
        return safe_int(m.group(1), None)

    # Cases like 201. or 201_ are too ambiguous. Do not guess.
    return None


def canonical_year(value: Any) -> str:
    y = parse_year_like(value)
    if y is None:
        return canonical_empty_text(value)
    return str(y)


def is_valid_birthyear(year: Optional[int]) -> bool:
    return year is not None and 1940 <= year <= 2010


def is_valid_season(year: Optional[int]) -> bool:
    return year is not None and 1990 <= year <= 2030


def is_valid_age_at_season(birthyear: Optional[int], season: Optional[int]) -> bool:
    if birthyear is None or season is None:
        return True
    age = season - birthyear
    return 15 <= age <= 48


def parse_pattern(value: Any) -> str:
    s = canonical_empty_text(value)
    if s == "empty":
        return "NULL"
    out = []
    for ch in s:
        if ch.isalpha():
            out.append("L")
        elif ch.isdigit():
            out.append("D")
        else:
            out.append("S")
    # Compress runs.
    comp = []
    prev = None
    cnt = 0
    for token in out:
        if token == prev:
            cnt += 1
        else:
            if prev is not None:
                comp.append(f"{prev}{cnt}")
            prev = token
            cnt = 1
    if prev is not None:
        comp.append(f"{prev}{cnt}")
    return "".join(comp)


def get_row_val(row_values: Dict[str, Any], *names: str) -> str:
    if not isinstance(row_values, dict):
        return "empty"
    for n in names:
        if n in row_values:
            return canonical_empty_text(row_values.get(n))
        nn = normalize_col_name(n)
        for k, v in row_values.items():
            if normalize_col_name(k) == nn:
                return canonical_empty_text(v)
    return "empty"


def get_int_from_row(row_values: Dict[str, Any], *names: str) -> Optional[int]:
    v = get_row_val(row_values, *names)
    y = parse_year_like(v)
    return y


def top_values_from_distribution(dist: Dict[str, Any], target_col: str) -> List[Dict[str, Any]]:
    if not isinstance(dist, dict):
        return []
    col_dist = dist.get(target_col) or dist.get(normalize_col_name(target_col))
    if not isinstance(col_dist, dict):
        return []
    tops = col_dist.get("top_values", []) or []
    return [x for x in tops if isinstance(x, dict)]


# ============================================================
# 3. Candidate handling
# ============================================================

def add_candidate(candidate_dict: Dict[str, Dict[str, Any]], value: Any, source: str, score: float, extra: Any = None):
    if value is None:
        return

    value = canonical_empty_text(value)
    if value in {"", "empty"}:
        return

    key = norm_text(value)
    if not key:
        return

    if key not in candidate_dict:
        candidate_dict[key] = {
            "candidate_value": value,
            "candidate_sources": set(),
            "source_score": float(score),
            "is_from_rule": 0,
            "is_from_neighbor": 0,
            "is_from_column_top": 0,
            "is_from_similarity": 0,
            "is_from_hint": 0,
            "is_from_dictionary": 0,
            "is_from_pattern_restore": 0,
            "is_from_soccer_profile": 0,
            "is_from_soccer_direct_rule": 0,
            "is_from_domain": 0,
            "is_from_numeric_rule": 0,
            "is_from_profile_key_majority": 0,
            "is_from_entity_pair_majority": 0,
            "is_from_team_season_position": 0,
            "profile_key_group_size": 0,
            "profile_key_top_count": 0,
            "profile_key_majority_ratio": 0.0,
            "entity_pair_group_size": 0,
            "entity_pair_top_count": 0,
            "entity_pair_majority_ratio": 0.0,
            "team_season_position_group_size": 0,
            "team_season_position_top_count": 0,
            "team_season_position_ratio": 0.0,
            "extra_info": [],
        }

    item = candidate_dict[key]
    item["candidate_sources"].add(str(source))
    item["source_score"] = max(float(item["source_score"]), float(score))

    if source in {"rule_expected_value", "expected_values_from_rules", "rule_reason_parse"}:
        item["is_from_rule"] = 1
    elif source in {"neighbor_majority", "similar_row_target"}:
        item["is_from_neighbor"] = 1
    elif source in {"column_top_value", "alternative_values_from_column"}:
        item["is_from_column_top"] = 1
    elif source in {"similar_column_value", "similar_context_value"}:
        item["is_from_similarity"] = 1
    elif source in {"suggested_correct_value"}:
        item["is_from_hint"] = 1
    elif source.startswith("soccer_profile") or source in {
        "team_season_distribution",
        "team_distribution",
        "stadium_city_distribution",
        "context_distribution",
    }:
        item["is_from_soccer_profile"] = 1
    elif source.startswith("soccer_direct") or source.startswith("position_") or source.startswith("year_") or source.startswith("team_context_"):
        item["is_from_soccer_direct_rule"] = 1
        item["is_from_rule"] = 1
    elif source.startswith("domain_"):
        item["is_from_domain"] = 1
        item["is_from_dictionary"] = 1
    elif source.startswith("numeric_"):
        item["is_from_numeric_rule"] = 1
        item["is_from_rule"] = 1
    elif source.startswith("profile_key_majority"):
        item["is_from_profile_key_majority"] = 1
        item["is_from_soccer_profile"] = 1
    elif source.startswith("entity_pair_majority"):
        item["is_from_entity_pair_majority"] = 1
        item["is_from_soccer_profile"] = 1
    elif source.startswith("team_season_position"):
        item["is_from_team_season_position"] = 1
        item["is_from_soccer_profile"] = 1
    elif "dictionary" in source:
        item["is_from_dictionary"] = 1

    if isinstance(extra, dict):
        for metric_key in [
            "profile_key_group_size",
            "profile_key_top_count",
            "profile_key_majority_ratio",
            "entity_pair_group_size",
            "entity_pair_top_count",
            "entity_pair_majority_ratio",
            "team_season_position_group_size",
            "team_season_position_top_count",
            "team_season_position_ratio",
        ]:
            if metric_key in extra:
                if metric_key.endswith("_ratio"):
                    item[metric_key] = max(float(item.get(metric_key, 0.0)), safe_float(extra.get(metric_key, 0.0), 0.0))
                else:
                    item[metric_key] = max(safe_int(item.get(metric_key, 0), 0), safe_int(extra.get(metric_key, 0), 0))
        item["extra_info"].append(json.dumps(extra, ensure_ascii=False))
    elif extra is not None:
        item["extra_info"].append(str(extra))


# ============================================================
# 4. Input loading
# ============================================================

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    decoder = json.JSONDecoder()
    with open(path, "r", encoding="utf-8-sig") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
                elif isinstance(obj, list):
                    rows.extend([x for x in obj if isinstance(x, dict)])
                continue
            except json.JSONDecodeError:
                pass

            # Robust recovery for non-standard NaN or concatenated JSON objects.
            idx = 0
            parsed_any = False
            while idx < len(line):
                while idx < len(line) and line[idx] not in "{[":
                    idx += 1
                if idx >= len(line):
                    break
                try:
                    obj, end = decoder.raw_decode(line, idx)
                    parsed_any = True
                    if isinstance(obj, dict):
                        rows.append(obj)
                    elif isinstance(obj, list):
                        rows.extend([x for x in obj if isinstance(x, dict)])
                    idx = end
                except json.JSONDecodeError:
                    idx += 1
            if not parsed_any:
                raise ValueError(f"JSONL 解析失败: {path}, line={line_no}")
    return rows


def load_context_records(path: str) -> List[Dict[str, Any]]:
    lower = path.lower()
    if lower.endswith(".jsonl"):
        return load_jsonl(path)
    df = pd.read_csv(path, encoding="utf-8-sig")
    return df.to_dict(orient="records")


def load_error_positions(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"ERROR_PRED_FILE 不存在: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")

    if "row_id" not in df.columns or "column" not in df.columns:
        raise ValueError("错误位置文件必须至少包含 row_id,column 字段。")

    # If full inference file is passed, filter predicted errors.
    if "final_pred_label" in df.columns:
        mask = pd.to_numeric(df["final_pred_label"], errors="coerce").fillna(0).astype(int) == 1
        df = df.loc[mask].copy()
    elif "final_pred_label_name" in df.columns:
        mask = df["final_pred_label_name"].astype(str).str.strip().str.lower().eq("error")
        df = df.loc[mask].copy()
    elif "detector_pred_label" in df.columns and "value" not in df.columns:
        mask = pd.to_numeric(df["detector_pred_label"], errors="coerce").fillna(0).astype(int) == 1
        df = df.loc[mask].copy()

    df["row_id"] = pd.to_numeric(df["row_id"], errors="raise").astype(int)
    df["column"] = df["column"].astype(str)

    if "value" not in df.columns:
        df["value"] = ""

    return df.drop_duplicates(subset=["row_id", "column"], keep="first").reset_index(drop=True)


def build_context_map(records: List[Dict[str, Any]]) -> Dict[Tuple[int, str], Dict[str, Any]]:
    cmap = {}
    for obj in records:
        if "row_id" not in obj or "column" not in obj:
            continue
        try:
            rid = int(obj["row_id"])
        except Exception:
            continue
        col = str(obj["column"])
        cmap[(rid, col)] = obj
    return cmap


def load_dirty_table(path: str) -> Optional[pd.DataFrame]:
    if path_exists_nonempty(path):
        return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    return None


def build_fallback_context(row_id: int, column: str, error_row: pd.Series, dirty_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    value = canonical_empty_text(error_row.get("value", ""))
    row_values = {}

    if dirty_df is not None and 0 <= int(row_id) < len(dirty_df):
        row_values = {str(c): canonical_empty_text(dirty_df.at[int(row_id), c]) for c in dirty_df.columns}
        if value in {"", "empty"} and column in dirty_df.columns:
            value = canonical_empty_text(dirty_df.at[int(row_id), column])

    profile_bundle = {
        "name": get_row_val(row_values, "name"),
        "surname": get_row_val(row_values, "surname"),
        "birthyear": get_row_val(row_values, "birthyear"),
        "birthyear_int": parse_year_like(get_row_val(row_values, "birthyear")),
        "birthplace": get_row_val(row_values, "birthplace"),
        "position": get_row_val(row_values, "position"),
        "position_canonical": canonical_position(get_row_val(row_values, "position")),
        "team": get_row_val(row_values, "team"),
        "city": get_row_val(row_values, "city"),
        "stadium": get_row_val(row_values, "stadium"),
        "season": get_row_val(row_values, "season"),
        "season_int": parse_year_like(get_row_val(row_values, "season")),
        "manager": get_row_val(row_values, "manager"),
    }
    by = profile_bundle["birthyear_int"]
    sy = profile_bundle["season_int"]
    profile_bundle["age_at_season"] = sy - by if by is not None and sy is not None else None

    ctx = {
        "row_id": int(row_id),
        "column": str(column),
        "value": value,
        "row_context": {
            "row_values": row_values,
            "soccer_profile_bundle": profile_bundle,
        },
        "column_context": {"column": column},
        "conflict_context": {},
        "similarity_context": {},
        "soccer_consistency_context": {
            "soccer_profile_bundle": profile_bundle,
            "evidence_flags": str(error_row.get("soccer_evidence_flags", "")).split("|") if "soccer_evidence_flags" in error_row.index else [],
        },
        "soccer_attribution_features": {},
    }

    # Copy known flat features from error row.
    for k in [
        "semantic_type",
        "detected_type",
        "violation_count",
        "conflict_score",
        "neighbor_majority_value",
        "neighbor_majority_ratio",
        "main_rule_type",
        "main_usage_role",
        "soccer_evidence_flags",
        "soccer_consistency_score",
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
        "row_team",
        "row_city",
        "row_stadium",
        "row_manager",
        "soccer_attribution_bucket",
    ]:
        if k in error_row.index:
            ctx[k] = error_row.get(k)

    return ctx


# ============================================================
# 5. Global dictionaries
# ============================================================


def is_reliable_profile_value(value: Any) -> bool:
    v = canonical_empty_text(value)
    if v in {"", "empty"}:
        return False
    # Avoid adding obvious placeholders into profile dictionaries.
    low = norm_text(v)
    if low in {"unknown", "none", "null", "missing", "na", "n/a", "?"}:
        return False
    return True


def make_profile_key(parts: List[Tuple[str, Any]]) -> Optional[Tuple[Tuple[str, str], ...]]:
    clean_parts = []
    for name, value in parts:
        val = canonical_empty_text(value)
        if val in {"", "empty"}:
            return None
        clean_parts.append((str(name), val))
    return tuple(clean_parts)


def add_profile_distribution(global_dict: Dict[str, Any], key_name: str, key: Optional[Tuple[Tuple[str, str], ...]], target_col: str, target_val: Any):
    if key is None:
        return
    val = canonical_value_for_column(target_col, target_val)
    if not is_reliable_profile_value(val):
        return
    global_dict["profile_key_distribution"][(key_name, key)][normalize_col_name(target_col)][val] += 1


def add_entity_pair_distribution(global_dict: Dict[str, Any], key_name: str, key_val: Any, target_col: str, target_val: Any):
    key_val = canonical_empty_text(key_val)
    val = canonical_value_for_column(target_col, target_val)
    if not is_reliable_profile_value(key_val) or not is_reliable_profile_value(val):
        return
    global_dict["entity_pair_distribution"][(key_name, key_val)][normalize_col_name(target_col)][val] += 1


def update_soccer_profile_majority_dicts(row_vals: Dict[str, Any], global_dict: Dict[str, Any]):
    """
    Build conservative co-occurrence dictionaries for entity columns.

    Important:
    - This uses dirty/context rows only, not clean ground truth.
    - It targets the current oracle bottleneck columns: name, surname, birthplace.
    - Keys are chosen to be specific enough to avoid pure global-frequency repairs.
    """
    if not isinstance(row_vals, dict):
        return

    name = get_row_val(row_vals, "name")
    surname = get_row_val(row_vals, "surname")
    birthyear = canonical_year(get_row_val(row_vals, "birthyear"))
    birthplace = get_row_val(row_vals, "birthplace")
    position = canonical_position(get_row_val(row_vals, "position"))
    team = get_row_val(row_vals, "team")
    city = get_row_val(row_vals, "city")
    stadium = get_row_val(row_vals, "stadium")
    season = canonical_year(get_row_val(row_vals, "season"))
    manager = get_row_val(row_vals, "manager")

    # Pair mappings: most useful for name/surname and birthplace.
    add_entity_pair_distribution(global_dict, "surname_to_name", surname, "name", name)
    add_entity_pair_distribution(global_dict, "name_to_surname", name, "surname", surname)
    add_entity_pair_distribution(global_dict, "name_surname_to_birthplace", f"{name}||{surname}", "birthplace", birthplace)
    add_entity_pair_distribution(global_dict, "surname_birthyear_to_name", f"{surname}||{birthyear}", "name", name)
    add_entity_pair_distribution(global_dict, "name_birthyear_to_surname", f"{name}||{birthyear}", "surname", surname)
    add_entity_pair_distribution(global_dict, "surname_birthyear_to_birthplace", f"{surname}||{birthyear}", "birthplace", birthplace)
    add_entity_pair_distribution(global_dict, "name_birthyear_to_birthplace", f"{name}||{birthyear}", "birthplace", birthplace)

    # Profile keys. These are sparse but high precision when they repeat.
    keys = [
        ("team_season_birthyear_position", make_profile_key([
            ("team", team), ("season", season), ("birthyear", birthyear), ("position", position)
        ])),
        ("team_birthyear_position", make_profile_key([
            ("team", team), ("birthyear", birthyear), ("position", position)
        ])),
        ("surname_team_birthyear", make_profile_key([
            ("surname", surname), ("team", team), ("birthyear", birthyear)
        ])),
        ("name_team_birthyear", make_profile_key([
            ("name", name), ("team", team), ("birthyear", birthyear)
        ])),
        ("stadium_city_team_season", make_profile_key([
            ("stadium", stadium), ("city", city), ("team", team), ("season", season)
        ])),
        ("manager_team_season", make_profile_key([
            ("manager", manager), ("team", team), ("season", season)
        ])),
        ("birthplace_birthyear_team", make_profile_key([
            ("birthplace", birthplace), ("birthyear", birthyear), ("team", team)
        ])),
    ]

    for key_name, key in keys:
        for target_col, target_val in [
            ("name", name),
            ("surname", surname),
            ("birthplace", birthplace),
            ("position", position),
            ("team", team),
            ("city", city),
            ("stadium", stadium),
            ("manager", manager),
            ("season", season),
            ("birthyear", birthyear),
        ]:
            add_profile_distribution(global_dict, key_name, key, target_col, target_val)

def consume_row_values(row_vals: Dict[str, Any], global_dict: Dict[str, Any]):
    if not isinstance(row_vals, dict):
        return

    update_soccer_profile_majority_dicts(row_vals, global_dict)

    for c, v in row_vals.items():
        col = normalize_col_name(c)
        val = canonical_empty_text(v)
        if val and val != "empty":
            global_dict["column_value_counter"][str(c)][val] += 1
            global_dict["column_value_counter"][col][canonical_value_for_column(col, val)] += 1

    team = get_row_val(row_vals, "team")
    city = get_row_val(row_vals, "city")
    stadium = get_row_val(row_vals, "stadium")
    manager = get_row_val(row_vals, "manager")
    position = canonical_position(get_row_val(row_vals, "position"))
    season = canonical_year(get_row_val(row_vals, "season"))
    birthyear = canonical_year(get_row_val(row_vals, "birthyear"))

    if team != "empty":
        if city != "empty":
            global_dict["team_distribution"][team]["city"][city] += 1
        if stadium != "empty":
            global_dict["team_distribution"][team]["stadium"][stadium] += 1
        if manager != "empty":
            global_dict["team_distribution"][team]["manager"][manager] += 1
        if position != "empty":
            global_dict["team_distribution"][team]["position"][position] += 1
        if season != "empty":
            global_dict["team_distribution"][team]["season"][season] += 1

    if team != "empty" and season != "empty":
        key = (team, season)
        if city != "empty":
            global_dict["team_season_distribution"][key]["city"][city] += 1
        if stadium != "empty":
            global_dict["team_season_distribution"][key]["stadium"][stadium] += 1
        if manager != "empty":
            global_dict["team_season_distribution"][key]["manager"][manager] += 1
        if position != "empty":
            global_dict["team_season_distribution"][key]["position"][position] += 1

    if stadium != "empty" and city != "empty":
        key2 = (stadium, city)
        if team != "empty":
            global_dict["stadium_city_distribution"][key2]["team"][team] += 1
        if manager != "empty":
            global_dict["stadium_city_distribution"][key2]["manager"][manager] += 1
        if season != "empty":
            global_dict["stadium_city_distribution"][key2]["season"][season] += 1

    by = parse_year_like(birthyear)
    sy = parse_year_like(season)
    if by is not None and sy is not None:
        age = sy - by
        global_dict["age_at_season_counter"][age] += 1


def canonical_value_for_column(column: str, value: Any) -> str:
    col = normalize_col_name(column)
    if col == "position":
        return canonical_position(value)
    if col in YEAR_COLUMNS:
        return canonical_year(value)
    return canonical_empty_text(value)


def build_global_dictionaries(context_records: List[Dict[str, Any]], dirty_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    global_dict = {
        "column_value_counter": defaultdict(Counter),
        "team_distribution": defaultdict(lambda: defaultdict(Counter)),
        "team_season_distribution": defaultdict(lambda: defaultdict(Counter)),
        "stadium_city_distribution": defaultdict(lambda: defaultdict(Counter)),
        "age_at_season_counter": Counter(),

        # Entity/profile majority dictionaries for Soccer repair.
        # These are built only from dirty/context data, not from clean labels.
        "profile_key_distribution": defaultdict(lambda: defaultdict(Counter)),
        "entity_pair_distribution": defaultdict(lambda: defaultdict(Counter)),
    }
    for obj in context_records:
        row_context = obj.get("row_context", {})
        if isinstance(row_context, dict):
            consume_row_values(row_context.get("row_values", {}), global_dict)

        sim_ctx = obj.get("similarity_context", {})
        if isinstance(sim_ctx, dict):
            for item in sim_ctx.get("topk_similar_rows", []) or []:
                if isinstance(item, dict):
                    consume_row_values(item.get("row_values", {}), global_dict)

        col_ctx = obj.get("column_context", {})
        if isinstance(col_ctx, dict):
            col = str(col_ctx.get("column", obj.get("column", "")))
            for t in col_ctx.get("top_values_with_counts", []) or []:
                if isinstance(t, dict):
                    val = canonical_empty_text(t.get("value", ""))
                    cnt = safe_int(t.get("count", 1), 1)
                    if val and val != "empty":
                        global_dict["column_value_counter"][col][val] += cnt
                        global_dict["column_value_counter"][normalize_col_name(col)][canonical_value_for_column(col, val)] += cnt

    if dirty_df is not None:
        for _, r in dirty_df.iterrows():
            consume_row_values({str(c): r[c] for c in dirty_df.columns}, global_dict)

    return global_dict


# ============================================================
# 6. Soccer-specific candidate generation
# ============================================================

def add_position_domain_candidates(dirty_value: str, candidate_dict: Dict[str, Dict[str, Any]], global_dict: Dict[str, Any]):
    canonical = canonical_position(dirty_value)
    if canonical != "empty" and norm_text(canonical) != norm_text(dirty_value) and canonical in POSITION_DOMAIN_VALUES:
        add_candidate(candidate_dict, canonical, "domain_position_canonicalize", SCORE_DOMAIN_NORMALIZE, "canonical position")

    for v in POSITION_DOMAIN_VALUES:
        if norm_text(v) == norm_text(dirty_value):
            continue
        sim = string_similarity(dirty_value, v)
        if sim >= MIN_STRING_SIM_FOR_DOMAIN_VALUE:
            add_candidate(candidate_dict, v, "domain_position_similar_value", 0.25 + 0.35 * sim, f"sim={sim:.4f}")

    # If current value is missing/invalid, add canonical four main roles, but keep weaker score.
    if is_empty_like_value(dirty_value) or canonical_position(dirty_value) not in POSITION_DOMAIN_VALUES:
        for v in ["Defender", "Forward", "Goalkeeper", "Midfield"]:
            cnt = global_dict["column_value_counter"].get("position", Counter()).get(v, 0)
            add_candidate(candidate_dict, v, "domain_position_core_value", 0.35 + min(cnt / 100000.0, 0.20), f"global_count={cnt}")


def add_year_candidates(column: str, dirty_value: str, ctx: Dict[str, Any], candidate_dict: Dict[str, Dict[str, Any]]):
    col = normalize_col_name(column)
    if col not in YEAR_COLUMNS:
        return

    row_values = (ctx.get("row_context", {}) or {}).get("row_values", {}) or {}

    dirty_year = parse_year_like(dirty_value)
    if dirty_year is not None:
        if str(dirty_year) != canonical_empty_text(dirty_value):
            add_candidate(candidate_dict, str(dirty_year), "year_canonicalize", SCORE_YEAR_NORMALIZE, "year typo canonicalization")

    birthyear = get_int_from_row(row_values, "birthyear")
    season = get_int_from_row(row_values, "season")

    if col == "birthyear":
        # If current birthyear invalid but season known, generate plausible birthyears by normal player age.
        if season is None:
            season = safe_int(ctx.get("row_season_int", None), None)
        if season is not None:
            for age, score_delta in [(18, 0.02), (21, 0.05), (24, 0.08), (27, 0.08), (30, 0.05), (33, 0.02)]:
                by = season - age
                if is_valid_birthyear(by):
                    add_candidate(candidate_dict, str(by), "numeric_birthyear_age_window", 0.36 + score_delta, f"season={season}|age={age}")

        # Row attribution field may already contain canonical row birthyear.
        row_by = safe_int(ctx.get("row_birthyear_int", None), None)
        if row_by is not None and is_valid_birthyear(row_by):
            add_candidate(candidate_dict, str(row_by), "numeric_row_birthyear", 0.54, "row_birthyear_int")

    elif col == "season":
        if birthyear is None:
            birthyear = safe_int(ctx.get("row_birthyear_int", None), None)
        if birthyear is not None:
            for age, score_delta in [(18, 0.02), (21, 0.05), (24, 0.08), (27, 0.08), (30, 0.05), (33, 0.02)]:
                sy = birthyear + age
                if is_valid_season(sy):
                    add_candidate(candidate_dict, str(sy), "numeric_season_age_window", 0.36 + score_delta, f"birthyear={birthyear}|age={age}")

        row_sy = safe_int(ctx.get("row_season_int", None), None)
        if row_sy is not None and is_valid_season(row_sy):
            add_candidate(candidate_dict, str(row_sy), "numeric_row_season", 0.54, "row_season_int")


def add_rule_candidates(ctx: Dict[str, Any], column: str, dirty_value: str, candidate_dict: Dict[str, Dict[str, Any]]):
    conflict = ctx.get("conflict_context", {}) or {}
    if not isinstance(conflict, dict):
        return

    # Direct expected values from violated rules.
    for vr in conflict.get("violated_rules_detailed", []) or []:
        if not isinstance(vr, dict):
            continue
        ev = vr.get("expected_value", None)
        if not is_empty_like_value(ev):
            add_candidate(candidate_dict, canonical_value_for_column(column, ev), "rule_expected_value", SCORE_RULE_EXPECTED_HIGH, vr.get("rule_id", ""))

        reason = str(vr.get("reason", ""))
        # Parse valid=[...] from position domain rule.
        if normalize_col_name(column) == "position":
            m = re.search(r"valid=\[([^\]]+)\]", reason)
            if m:
                vals = re.findall(r"'([^']+)'|\"([^\"]+)\"", m.group(1))
                for a, b in vals:
                    val = a or b
                    if val:
                        add_candidate(candidate_dict, canonical_position(val), "rule_reason_parse", SCORE_RULE_EXPECTED_MED, "valid position list")

    # expected_values_from_rules
    for item in conflict.get("expected_values_from_rules", []) or []:
        if isinstance(item, dict):
            val = item.get("value", item.get("expected_value", ""))
            if not is_empty_like_value(val):
                score = safe_float(item.get("score", SCORE_RULE_EXPECTED_MED), SCORE_RULE_EXPECTED_MED)
                add_candidate(candidate_dict, canonical_value_for_column(column, val), "expected_values_from_rules", score, item)
        elif not is_empty_like_value(item):
            add_candidate(candidate_dict, canonical_value_for_column(column, item), "expected_values_from_rules", SCORE_RULE_EXPECTED_MED, item)

    # alternative values from rules/column are useful but weaker.
    for source_key in ["alternative_values_from_column", "all_alternative_values"]:
        for item in conflict.get(source_key, []) or []:
            if not isinstance(item, dict):
                continue
            val = item.get("value", "")
            if is_empty_like_value(val):
                continue
            cnt = safe_int(item.get("count", 0), 0)
            score = SCORE_COLUMN_TOP + min(math.log1p(max(cnt, 0)) / 20.0, 0.30)
            add_candidate(candidate_dict, canonical_value_for_column(column, val), "alternative_values_from_column", score, f"count={cnt}")


def add_column_context_candidates(ctx: Dict[str, Any], column: str, dirty_value: str, candidate_dict: Dict[str, Dict[str, Any]]):
    col_ctx = ctx.get("column_context", {}) or {}
    if not isinstance(col_ctx, dict):
        return

    for idx, item in enumerate(col_ctx.get("top_values_with_counts", []) or []):
        if not isinstance(item, dict):
            continue
        val = item.get("value", "")
        if is_empty_like_value(val):
            continue

        canon_val = canonical_value_for_column(column, val)
        if is_empty_like_value(canon_val) or norm_text(canon_val) == norm_text(dirty_value):
            continue

        cnt = safe_int(item.get("count", 0), 0)
        ratio = safe_float(item.get("ratio", 0.0), 0.0)
        score = SCORE_COLUMN_TOP + min(ratio, 0.25) + min(math.log1p(max(cnt, 0)) / 30.0, 0.18)
        if idx == 0:
            score += 0.05

        add_candidate(candidate_dict, canon_val, "column_top_value", score, f"count={cnt}|ratio={ratio:.6f}")

    # If candidate differs only by typo from a known column value.
    if not is_empty_like_value(dirty_value):
        for item in col_ctx.get("top_values_with_counts", []) or []:
            if not isinstance(item, dict):
                continue
            val = item.get("value", "")
            if is_empty_like_value(val):
                continue
            sim = string_similarity(dirty_value, val)
            if sim >= MIN_STRING_SIM_FOR_COLUMN_VALUE and norm_text(val) != norm_text(dirty_value):
                add_candidate(
                    candidate_dict,
                    canonical_value_for_column(column, val),
                    "similar_column_value",
                    0.35 + 0.35 * sim,
                    f"sim={sim:.4f}",
                )


def add_similarity_candidates(ctx: Dict[str, Any], column: str, dirty_value: str, candidate_dict: Dict[str, Dict[str, Any]]):
    sim_ctx = ctx.get("similarity_context", {}) or {}
    if not isinstance(sim_ctx, dict):
        return

    neighbor_value = sim_ctx.get("neighbor_majority_value", ctx.get("neighbor_majority_value", None))
    neighbor_ratio = safe_float(sim_ctx.get("neighbor_majority_ratio", ctx.get("neighbor_majority_ratio", 0.0)), 0.0)
    if not is_empty_like_value(neighbor_value) and norm_text(neighbor_value) != norm_text(dirty_value):
        add_candidate(
            candidate_dict,
            canonical_value_for_column(column, neighbor_value),
            "neighbor_majority",
            SCORE_NEIGHBOR_MAJORITY + min(neighbor_ratio, 1.0) * 0.20,
            f"ratio={neighbor_ratio:.4f}",
        )

    for item in sim_ctx.get("topk_similar_rows", [])[:TOP_K_SIMILAR_ROW_VALUES] or []:
        if not isinstance(item, dict):
            continue
        val = item.get("target_column_value", "")
        sim = safe_float(item.get("similarity", 0.0), 0.0)
        if is_empty_like_value(val) or norm_text(val) == norm_text(dirty_value):
            continue
        add_candidate(
            candidate_dict,
            canonical_value_for_column(column, val),
            "similar_row_target",
            SCORE_SIMILAR_ROW + min(sim, 1.0) * 0.20,
            f"similarity={sim:.4f}|row_id={item.get('row_id', '')}",
        )



def get_profile_majority_candidates(counter: Counter, min_group_size: int, min_top_count: int, min_ratio: float, top_k: int):
    total = sum(counter.values())
    if total < min_group_size:
        return []
    out = []
    for idx, (val, cnt) in enumerate(counter.most_common(top_k)):
        ratio = cnt / max(total, 1)
        if cnt < min_top_count:
            continue
        if ratio < min_ratio:
            continue
        out.append((val, cnt, total, ratio, idx))
    return out


def get_context_row_values(ctx: Dict[str, Any]) -> Dict[str, Any]:
    row_context = ctx.get("row_context", {}) or {}
    if isinstance(row_context, dict):
        rv = row_context.get("row_values", {}) or {}
        if isinstance(rv, dict):
            return rv
    return {}


def add_profile_key_majority_candidates(ctx: Dict[str, Any], column: str, dirty_value: str, candidate_dict: Dict[str, Dict[str, Any]], global_dict: Dict[str, Any]):
    """
    Add high-precision co-occurrence candidates for Soccer entity fields.

    This is the key fix for low oracle coverage:
      name / surname / birthplace clean values were often missing from candidates.
    """
    col = normalize_col_name(column)
    if col not in {"name", "surname", "birthplace", "position"}:
        return

    row_values = get_context_row_values(ctx)
    if not row_values:
        return

    name = get_row_val(row_values, "name")
    surname = get_row_val(row_values, "surname")
    birthyear = canonical_year(get_row_val(row_values, "birthyear"))
    birthplace = get_row_val(row_values, "birthplace")
    position = canonical_position(get_row_val(row_values, "position"))
    team = get_row_val(row_values, "team")
    city = get_row_val(row_values, "city")
    stadium = get_row_val(row_values, "stadium")
    season = canonical_year(get_row_val(row_values, "season"))
    manager = get_row_val(row_values, "manager")

    # ------------------------------------------------------------
    # A. Entity pair majority candidates.
    # ------------------------------------------------------------
    pair_keys = []
    if col == "name":
        pair_keys.extend([
            ("surname_to_name", surname),
            ("surname_birthyear_to_name", f"{surname}||{birthyear}"),
        ])
    elif col == "surname":
        pair_keys.extend([
            ("name_to_surname", name),
            ("name_birthyear_to_surname", f"{name}||{birthyear}"),
        ])
    elif col == "birthplace":
        pair_keys.extend([
            ("name_surname_to_birthplace", f"{name}||{surname}"),
            ("surname_birthyear_to_birthplace", f"{surname}||{birthyear}"),
            ("name_birthyear_to_birthplace", f"{name}||{birthyear}"),
        ])

    for key_name, key_val in pair_keys:
        key_val = canonical_empty_text(key_val)
        if key_val in {"", "empty"} or "empty" in str(key_val).split("||"):
            continue
        counter = global_dict["entity_pair_distribution"].get((key_name, key_val), {}).get(col, Counter())
        candidates = get_profile_majority_candidates(
            counter,
            ENTITY_PAIR_MIN_GROUP_SIZE,
            ENTITY_PAIR_MIN_TOP_COUNT,
            ENTITY_PAIR_MIN_RATIO,
            TOP_K_PROFILE_MAJORITY_VALUES,
        )
        for val, cnt, total, ratio, idx in candidates:
            if norm_text(val) == norm_text(dirty_value):
                continue
            score = SCORE_ENTITY_PAIR_MAJORITY_TOP if idx == 0 else SCORE_ENTITY_PAIR_MAJORITY_SECOND
            score += min(ratio, 0.20)
            add_candidate(
                candidate_dict,
                canonical_value_for_column(col, val),
                f"entity_pair_majority_{key_name}",
                score,
                {
                    "key_name": key_name,
                    "key_value": key_val,
                    "entity_pair_group_size": total,
                    "entity_pair_top_count": cnt,
                    "entity_pair_majority_ratio": ratio,
                },
            )

    # ------------------------------------------------------------
    # B. Profile-key majority candidates.
    # ------------------------------------------------------------
    profile_keys = [
        ("team_season_birthyear_position", make_profile_key([
            ("team", team), ("season", season), ("birthyear", birthyear), ("position", position)
        ])),
        ("team_birthyear_position", make_profile_key([
            ("team", team), ("birthyear", birthyear), ("position", position)
        ])),
        ("surname_team_birthyear", make_profile_key([
            ("surname", surname), ("team", team), ("birthyear", birthyear)
        ])),
        ("name_team_birthyear", make_profile_key([
            ("name", name), ("team", team), ("birthyear", birthyear)
        ])),
        ("stadium_city_team_season", make_profile_key([
            ("stadium", stadium), ("city", city), ("team", team), ("season", season)
        ])),
        ("manager_team_season", make_profile_key([
            ("manager", manager), ("team", team), ("season", season)
        ])),
        ("birthplace_birthyear_team", make_profile_key([
            ("birthplace", birthplace), ("birthyear", birthyear), ("team", team)
        ])),
    ]

    for key_name, key in profile_keys:
        if key is None:
            continue
        counter = global_dict["profile_key_distribution"].get((key_name, key), {}).get(col, Counter())

        # For position, allow a slightly lower ratio because legal position domain is small.
        min_ratio = POSITION_PROFILE_MIN_RATIO if col == "position" else PROFILE_KEY_MIN_RATIO
        min_group = POSITION_PROFILE_MIN_GROUP_SIZE if col == "position" else PROFILE_KEY_MIN_GROUP_SIZE

        candidates = get_profile_majority_candidates(
            counter,
            min_group,
            PROFILE_KEY_MIN_TOP_COUNT,
            min_ratio,
            TOP_K_PROFILE_MAJORITY_VALUES,
        )
        for val, cnt, total, ratio, idx in candidates:
            if norm_text(val) == norm_text(dirty_value):
                continue
            source = f"profile_key_majority_{key_name}"
            if col == "position" and key_name in {"team_season_birthyear_position", "team_birthyear_position"}:
                source = f"team_season_position_{key_name}"
            score = SCORE_PROFILE_KEY_MAJORITY_TOP if idx == 0 else SCORE_PROFILE_KEY_MAJORITY_SECOND
            if col == "position":
                score = SCORE_TEAM_SEASON_POSITION_TOP if idx == 0 else SCORE_PROFILE_KEY_MAJORITY_SECOND
            score += min(ratio, 0.20)
            extra = {
                "key_name": key_name,
                "profile_key_group_size": total,
                "profile_key_top_count": cnt,
                "profile_key_majority_ratio": ratio,
            }
            if source.startswith("team_season_position"):
                extra.update({
                    "team_season_position_group_size": total,
                    "team_season_position_top_count": cnt,
                    "team_season_position_ratio": ratio,
                })
            add_candidate(
                candidate_dict,
                canonical_value_for_column(col, val),
                source,
                score,
                extra,
            )


def add_soccer_distribution_candidates(ctx: Dict[str, Any], column: str, dirty_value: str, candidate_dict: Dict[str, Dict[str, Any]]):
    soccer_ctx = ctx.get("soccer_consistency_context", {}) or {}
    if not isinstance(soccer_ctx, dict):
        # Some older pipelines store the same block under adult_consistency_context.
        soccer_ctx = ctx.get("adult_consistency_context", {}) or {}
        if not isinstance(soccer_ctx, dict):
            return

    distribution_sources = [
        ("team_season_distribution", SCORE_CONTEXT_DISTRIBUTION_TOP),
        ("team_distribution", SCORE_CONTEXT_DISTRIBUTION),
        ("stadium_city_distribution", SCORE_CONTEXT_DISTRIBUTION),
    ]

    for source_name, base_score in distribution_sources:
        dist = soccer_ctx.get(source_name, {}) or {}
        tops = top_values_from_distribution(dist, normalize_col_name(column))
        if not tops:
            tops = top_values_from_distribution(dist, column)

        group_size = 0
        col_dist = None
        if isinstance(dist, dict):
            col_dist = dist.get(column) or dist.get(normalize_col_name(column))
        if isinstance(col_dist, dict):
            group_size = safe_int(col_dist.get("group_size", 0), 0)

        for idx, item in enumerate(tops[:TOP_K_DISTRIBUTION_VALUES]):
            val = item.get("value", "")
            if is_empty_like_value(val) or norm_text(val) == norm_text(dirty_value):
                continue
            cnt = safe_int(item.get("count", 0), 0)
            ratio = cnt / max(group_size, 1) if group_size > 0 else 0.0
            score = base_score
            if idx == 0:
                score += 0.06
            score += min(ratio, 0.25)
            add_candidate(
                candidate_dict,
                canonical_value_for_column(column, val),
                source_name,
                score,
                f"group_size={group_size}|count={cnt}|ratio={ratio:.4f}",
            )


def add_soccer_global_context_candidates(ctx: Dict[str, Any], column: str, dirty_value: str, candidate_dict: Dict[str, Dict[str, Any]], global_dict: Dict[str, Any]):
    col = normalize_col_name(column)
    row_values = (ctx.get("row_context", {}) or {}).get("row_values", {}) or {}

    team = get_row_val(row_values, "team")
    season = canonical_year(get_row_val(row_values, "season"))
    stadium = get_row_val(row_values, "stadium")
    city = get_row_val(row_values, "city")

    # From team + season group.
    if team != "empty" and season != "empty":
        key = (team, season)
        counter = global_dict["team_season_distribution"].get(key, {}).get(col, Counter())
        for idx, (val, cnt) in enumerate(counter.most_common(TOP_K_DISTRIBUTION_VALUES)):
            if is_empty_like_value(val) or norm_text(val) == norm_text(dirty_value):
                continue
            score = 0.78 if idx == 0 else 0.60
            add_candidate(candidate_dict, canonical_value_for_column(col, val), "soccer_profile_team_season_global", score, f"team={team}|season={season}|count={cnt}")

    # From team group.
    if team != "empty":
        counter = global_dict["team_distribution"].get(team, {}).get(col, Counter())
        for idx, (val, cnt) in enumerate(counter.most_common(TOP_K_DISTRIBUTION_VALUES)):
            if is_empty_like_value(val) or norm_text(val) == norm_text(dirty_value):
                continue
            score = 0.70 if idx == 0 else 0.54
            add_candidate(candidate_dict, canonical_value_for_column(col, val), "soccer_profile_team_global", score, f"team={team}|count={cnt}")

    # From stadium + city group.
    if stadium != "empty" and city != "empty":
        key2 = (stadium, city)
        counter = global_dict["stadium_city_distribution"].get(key2, {}).get(col, Counter())
        for idx, (val, cnt) in enumerate(counter.most_common(TOP_K_DISTRIBUTION_VALUES)):
            if is_empty_like_value(val) or norm_text(val) == norm_text(dirty_value):
                continue
            score = 0.66 if idx == 0 else 0.50
            add_candidate(candidate_dict, canonical_value_for_column(col, val), "soccer_profile_stadium_city_global", score, f"stadium={stadium}|city={city}|count={cnt}")


def add_direct_soccer_candidates(ctx: Dict[str, Any], column: str, dirty_value: str, candidate_dict: Dict[str, Dict[str, Any]], global_dict: Dict[str, Any]):
    col = normalize_col_name(column)

    if col == "position":
        add_position_domain_candidates(dirty_value, candidate_dict, global_dict)

        # Use row_position_canonical if present.
        row_position = ctx.get("row_position_canonical", None)
        if not is_empty_like_value(row_position):
            add_candidate(candidate_dict, canonical_position(row_position), "position_row_canonical", SCORE_SOCCER_DIRECT_RULE, "row_position_canonical")

    if col in YEAR_COLUMNS:
        add_year_candidates(col, dirty_value, ctx, candidate_dict)

    # Flat canonical fields from predicted_error_positions.
    if col == "position":
        canon = ctx.get("canonical_position", None)
        if not is_empty_like_value(canon):
            add_candidate(candidate_dict, canonical_position(canon), "position_flat_canonical", SCORE_SOCCER_DIRECT_RULE, "canonical_position")

    if col in YEAR_COLUMNS:
        y = ctx.get("year_value", None)
        if not is_empty_like_value(y):
            yy = parse_year_like(y)
            if yy is not None:
                add_candidate(candidate_dict, str(yy), "year_flat_value", SCORE_YEAR_NORMALIZE, "year_value")


def add_global_dictionary_candidates(column: str, dirty_value: str, candidate_dict: Dict[str, Dict[str, Any]], global_dict: Dict[str, Any]):
    col = normalize_col_name(column)
    counter = global_dict["column_value_counter"].get(col, Counter())
    if not counter:
        return

    # Add top values for context/text columns and for missing values.
    for idx, (val, cnt) in enumerate(counter.most_common(TOP_K_COLUMN_VALUES)):
        if is_empty_like_value(val) or norm_text(val) == norm_text(dirty_value):
            continue

        if is_empty_like_value(dirty_value):
            score = 0.20 + min(math.log1p(cnt) / 40.0, 0.20)
        else:
            sim = string_similarity(dirty_value, val)
            if col in TEXT_ID_COLUMNS and sim < MIN_STRING_SIM_FOR_CONTEXT_VALUE:
                continue
            score = 0.25 + 0.35 * sim

        add_candidate(candidate_dict, canonical_value_for_column(col, val), "global_column_dictionary", score, f"rank={idx+1}|count={cnt}")


def add_llm_hint_candidate(ctx: Dict[str, Any], column: str, dirty_value: str, candidate_dict: Dict[str, Dict[str, Any]]):
    for key in ["suggested_correct_value", "llm_suggested_value", "suggested_value"]:
        val = ctx.get(key, None)
        if not is_empty_like_value(val) and norm_text(val) != norm_text(dirty_value):
            add_candidate(candidate_dict, canonical_value_for_column(column, val), "suggested_correct_value", 0.82, key)


def candidate_relevance_boost(item: Dict[str, Any], column: str, dirty_value: str, ctx: Dict[str, Any]) -> float:
    val = item.get("candidate_value", "")
    col = normalize_col_name(column)
    boost = 0.0

    if col == "position":
        canon = canonical_position(val)
        if canon in POSITION_DOMAIN_VALUES:
            boost += 0.18
        if canonical_position(dirty_value) == canon and norm_text(dirty_value) != norm_text(val):
            boost += 0.15

    if col == "birthyear":
        y = parse_year_like(val)
        row_season = safe_int(ctx.get("row_season_int", None), None)
        if y is not None and is_valid_birthyear(y):
            boost += 0.10
        if y is not None and row_season is not None and is_valid_age_at_season(y, row_season):
            boost += 0.16

    if col == "season":
        y = parse_year_like(val)
        row_birthyear = safe_int(ctx.get("row_birthyear_int", None), None)
        if y is not None and is_valid_season(y):
            boost += 0.10
        if y is not None and row_birthyear is not None and is_valid_age_at_season(row_birthyear, y):
            boost += 0.16

    if col in {"city", "stadium", "manager", "team", "birthplace", "name", "surname"}:
        if item.get("is_from_profile_key_majority", 0):
            boost += 0.22
        if item.get("is_from_entity_pair_majority", 0):
            boost += 0.24
        if item.get("is_from_soccer_profile", 0):
            boost += 0.14
        if item.get("is_from_rule", 0):
            boost += 0.10

    if item.get("is_from_rule", 0):
        boost += 0.08
    if item.get("is_from_soccer_direct_rule", 0):
        boost += 0.10
    if item.get("is_from_soccer_profile", 0):
        boost += 0.06
    if item.get("is_from_team_season_position", 0):
        boost += 0.20
    if item.get("is_from_profile_key_majority", 0):
        boost += 0.12
    if item.get("is_from_entity_pair_majority", 0):
        boost += 0.12

    return boost


def finalize_candidates(candidate_dict: Dict[str, Dict[str, Any]], column: str, dirty_value: str, ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for item in candidate_dict.values():
        val = item["candidate_value"]
        if not INCLUDE_DIRTY_VALUE and norm_text(val) == norm_text(dirty_value):
            continue

        col = normalize_col_name(column)
        if col == "position":
            val = canonical_position(val)
            if val == "empty":
                continue
            item["candidate_value"] = val
        elif col in YEAR_COLUMNS:
            y = parse_year_like(val)
            if y is not None:
                item["candidate_value"] = str(y)
            # Filter impossible years unless very explicitly from rule.
            if col == "birthyear" and y is not None and not is_valid_birthyear(y):
                if not item.get("is_from_rule", 0):
                    continue
            if col == "season" and y is not None and not is_valid_season(y):
                if not item.get("is_from_rule", 0):
                    continue

        item["candidate_sources"] = "|".join(sorted(item["candidate_sources"]))
        item["extra_info"] = "|".join(item["extra_info"][:8])
        item["source_count"] = len(str(item["candidate_sources"]).split("|")) if item["candidate_sources"] else 0
        item["source_score"] = float(item["source_score"]) + candidate_relevance_boost(item, column, dirty_value, ctx)
        out.append(item)

    # Deduplicate after canonicalization.
    dedup = {}
    for item in out:
        key = norm_text(item["candidate_value"])
        if key not in dedup or item["source_score"] > dedup[key]["source_score"]:
            dedup[key] = item
        else:
            old = dedup[key]
            old_sources = set(str(old.get("candidate_sources", "")).split("|")) | set(str(item.get("candidate_sources", "")).split("|"))
            old["candidate_sources"] = "|".join(sorted(s for s in old_sources if s))
            old["source_count"] = len(old["candidate_sources"].split("|")) if old["candidate_sources"] else 0
            for flag in [
                "is_from_rule", "is_from_neighbor", "is_from_column_top", "is_from_similarity",
                "is_from_hint", "is_from_dictionary", "is_from_pattern_restore",
                "is_from_soccer_profile", "is_from_soccer_direct_rule", "is_from_domain", "is_from_numeric_rule",
                "is_from_profile_key_majority", "is_from_entity_pair_majority", "is_from_team_season_position",
            ]:
                old[flag] = max(safe_int(old.get(flag, 0), 0), safe_int(item.get(flag, 0), 0))
            for metric_key in [
                "profile_key_group_size", "profile_key_top_count",
                "entity_pair_group_size", "entity_pair_top_count",
                "team_season_position_group_size", "team_season_position_top_count",
            ]:
                old[metric_key] = max(safe_int(old.get(metric_key, 0), 0), safe_int(item.get(metric_key, 0), 0))
            for ratio_key in [
                "profile_key_majority_ratio", "entity_pair_majority_ratio", "team_season_position_ratio",
            ]:
                old[ratio_key] = max(safe_float(old.get(ratio_key, 0.0), 0.0), safe_float(item.get(ratio_key, 0.0), 0.0))

    out = list(dedup.values())
    out.sort(key=lambda x: (-float(x.get("source_score", 0.0)), -safe_int(x.get("source_count", 0), 0), str(x.get("candidate_value", ""))))
    return out[:MAX_CANDIDATES_PER_CELL]


def generate_candidates_for_one_cell(row_id: int, column: str, dirty_value: str, ctx: Dict[str, Any], global_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    candidate_dict = {}

    if INCLUDE_DIRTY_VALUE and not is_empty_like_value(dirty_value):
        add_candidate(candidate_dict, dirty_value, "dirty_value", SCORE_DIRTY_VALUE, "original dirty value")

    add_rule_candidates(ctx, column, dirty_value, candidate_dict)
    add_column_context_candidates(ctx, column, dirty_value, candidate_dict)
    add_similarity_candidates(ctx, column, dirty_value, candidate_dict)
    add_profile_key_majority_candidates(ctx, column, dirty_value, candidate_dict, global_dict)
    add_soccer_distribution_candidates(ctx, column, dirty_value, candidate_dict)
    add_soccer_global_context_candidates(ctx, column, dirty_value, candidate_dict, global_dict)
    add_direct_soccer_candidates(ctx, column, dirty_value, candidate_dict, global_dict)
    add_global_dictionary_candidates(column, dirty_value, candidate_dict, global_dict)
    add_llm_hint_candidate(ctx, column, dirty_value, candidate_dict)

    return finalize_candidates(candidate_dict, column, dirty_value, ctx)


# ============================================================
# 7. Context merge / augmentation
# ============================================================

def merge_flat_context_into_map(cmap: Dict[Tuple[int, str], Dict[str, Any]], flat_records: List[Dict[str, Any]]):
    for obj in flat_records:
        if "row_id" not in obj or "column" not in obj:
            continue
        try:
            rid = int(obj["row_id"])
        except Exception:
            continue
        col = str(obj["column"])
        key = (rid, col)
        if key not in cmap:
            cmap[key] = {
                "row_id": rid,
                "column": col,
                "value": obj.get("value", ""),
                "row_context": {"row_values": {}},
                "column_context": {"column": col},
                "conflict_context": {},
                "similarity_context": {},
                "soccer_consistency_context": {},
                "soccer_attribution_features": {},
            }

        ctx = cmap[key]
        for k, v in obj.items():
            if k not in ctx or ctx.get(k) in {None, "", 0, 0.0}:
                ctx[k] = v

        # Mirror flat features into soccer_attribution_features for convenience.
        attr = ctx.setdefault("soccer_attribution_features", {})
        if isinstance(attr, dict):
            for k in [
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
                "row_team",
                "row_city",
                "row_stadium",
                "row_manager",
                "soccer_attribution_bucket",
            ]:
                if k in obj and k not in attr:
                    attr[k] = obj.get(k)


def get_ctx_value(ctx: Dict[str, Any], key: str, default: Any = None) -> Any:
    if key in ctx:
        return ctx.get(key, default)
    attr = ctx.get("soccer_attribution_features", {})
    if isinstance(attr, dict) and key in attr:
        return attr.get(key, default)
    soccer_ctx = ctx.get("soccer_consistency_context", {})
    if isinstance(soccer_ctx, dict) and key in soccer_ctx:
        return soccer_ctx.get(key, default)
    return default


def augment_error_row(error_row: pd.Series, ctx: Dict[str, Any], candidate_count: int, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = error_row.to_dict()

    row_context = ctx.get("row_context", {}) if isinstance(ctx.get("row_context", {}), dict) else {}
    row_values = row_context.get("row_values", {}) if isinstance(row_context.get("row_values", {}), dict) else {}

    soccer_ctx = ctx.get("soccer_consistency_context", {}) if isinstance(ctx.get("soccer_consistency_context", {}), dict) else {}
    if not soccer_ctx:
        soccer_ctx = ctx.get("adult_consistency_context", {}) if isinstance(ctx.get("adult_consistency_context", {}), dict) else {}

    bundle = soccer_ctx.get("soccer_profile_bundle", {}) if isinstance(soccer_ctx.get("soccer_profile_bundle", {}), dict) else {}
    if not bundle:
        bundle = row_context.get("soccer_profile_bundle", {}) if isinstance(row_context.get("soccer_profile_bundle", {}), dict) else {}

    attr = ctx.get("soccer_attribution_features", {}) if isinstance(ctx.get("soccer_attribution_features", {}), dict) else {}

    out["candidate_count"] = candidate_count
    out["has_rule_candidate"] = int(any(c.get("is_from_rule", 0) for c in candidates))
    out["has_soccer_profile_candidate"] = int(any(c.get("is_from_soccer_profile", 0) for c in candidates))
    out["has_soccer_direct_rule_candidate"] = int(any(c.get("is_from_soccer_direct_rule", 0) for c in candidates))
    out["has_neighbor_candidate"] = int(any(c.get("is_from_neighbor", 0) for c in candidates))
    out["has_domain_candidate"] = int(any(c.get("is_from_domain", 0) for c in candidates))
    out["has_numeric_candidate"] = int(any(c.get("is_from_numeric_rule", 0) for c in candidates))
    out["has_profile_key_majority_candidate"] = int(any(c.get("is_from_profile_key_majority", 0) for c in candidates))
    out["has_entity_pair_majority_candidate"] = int(any(c.get("is_from_entity_pair_majority", 0) for c in candidates))
    out["has_team_season_position_candidate"] = int(any(c.get("is_from_team_season_position", 0) for c in candidates))
    out["top_candidate_value"] = candidates[0]["candidate_value"] if candidates else ""
    out["top_candidate_score"] = candidates[0]["source_score"] if candidates else 0.0
    out["top_candidate_sources"] = candidates[0]["candidate_sources"] if candidates else ""

    # Soccer profile fields.
    out["row_birthyear_int"] = out.get("row_birthyear_int", attr.get("row_birthyear_int", bundle.get("birthyear_int", parse_year_like(get_row_val(row_values, "birthyear")))))
    out["row_season_int"] = out.get("row_season_int", attr.get("row_season_int", bundle.get("season_int", parse_year_like(get_row_val(row_values, "season")))))
    out["row_age_at_season"] = out.get("row_age_at_season", attr.get("row_age_at_season", bundle.get("age_at_season", None)))
    out["row_position_canonical"] = out.get("row_position_canonical", attr.get("row_position_canonical", bundle.get("position_canonical", canonical_position(get_row_val(row_values, "position")))))
    out["row_team"] = out.get("row_team", attr.get("row_team", bundle.get("team", get_row_val(row_values, "team"))))
    out["row_city"] = out.get("row_city", attr.get("row_city", bundle.get("city", get_row_val(row_values, "city"))))
    out["row_stadium"] = out.get("row_stadium", attr.get("row_stadium", bundle.get("stadium", get_row_val(row_values, "stadium"))))
    out["row_manager"] = out.get("row_manager", attr.get("row_manager", bundle.get("manager", get_row_val(row_values, "manager"))))

    evidence_flags = []
    if isinstance(soccer_ctx, dict):
        evidence_flags.extend(soccer_ctx.get("evidence_flags", []) or [])
        b = soccer_ctx.get("soccer_profile_bundle", {}) or {}
        if isinstance(b, dict):
            evidence_flags.extend(b.get("soccer_profile_inconsistencies", []) or [])
    if out.get("soccer_evidence_flags", ""):
        evidence_flags.extend(str(out.get("soccer_evidence_flags")).split("|"))
    evidence_flags = [str(x) for x in evidence_flags if str(x).strip()]
    out["soccer_evidence_flags"] = "|".join(list(dict.fromkeys(evidence_flags)))

    out["soccer_consistency_score"] = out.get("soccer_consistency_score", soccer_ctx.get("soccer_consistency_score", ctx.get("soccer_consistency_score", 0.0)))
    out["soccer_attribution_bucket"] = out.get("soccer_attribution_bucket", attr.get("soccer_attribution_bucket", ctx.get("soccer_attribution_bucket", "")))
    out["target_is_primary_column"] = out.get("target_is_primary_column", attr.get("target_is_primary_column", ctx.get("target_is_primary_column", 0)))
    out["target_is_context_only_column"] = out.get("target_is_context_only_column", attr.get("target_is_context_only_column", ctx.get("target_is_context_only_column", 0)))

    for k in [
        "birthyear_value_invalid",
        "season_value_invalid",
        "position_value_invalid",
        "position_needs_canonicalization",
        "birthyear_season_age_conflict",
        "team_context_conflict",
        "format_rule_signal",
        "fd_like_signal",
    ]:
        out[k] = out.get(k, attr.get(k, ctx.get(k, 0)))

    return out


# ============================================================
# 8. Args / main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--error_pred_file", default=DEFAULT_ERROR_PRED_FILE)
    p.add_argument("--context_file", default=DEFAULT_CONTEXT_FILE)
    p.add_argument("--fallback_context_flat_file", default=DEFAULT_FALLBACK_CONTEXT_FLAT_FILE)
    p.add_argument("--dirty_table_csv", default=DEFAULT_DIRTY_TABLE_CSV)
    p.add_argument("--output_candidates_expanded_csv", default=OUTPUT_CANDIDATES_EXPANDED_CSV)
    p.add_argument("--output_candidates_grouped_csv", default=OUTPUT_CANDIDATES_GROUPED_CSV)
    p.add_argument("--output_augmented_allowed_cells_csv", default=OUTPUT_AUGMENTED_ALLOWED_CELLS_CSV)
    p.add_argument("--output_summary_csv", default=OUTPUT_SUMMARY_CSV)
    return p.parse_args()


def main():
    args = parse_args()

    error_pred_file = args.error_pred_file
    context_file = resolve_file_path(args.context_file, "Soccer context file", required=False)
    fallback_context_file = resolve_file_path(args.fallback_context_flat_file, "Soccer fallback flat context", required=False)
    dirty_table_csv = resolve_file_path(args.dirty_table_csv, "soccer dirty table", required=False)

    expanded_path = resolve_output_path(args.output_candidates_expanded_csv)
    grouped_path = resolve_output_path(args.output_candidates_grouped_csv)
    augmented_path = resolve_output_path(args.output_augmented_allowed_cells_csv)
    summary_path = resolve_output_path(args.output_summary_csv)

    error_df = load_error_positions(error_pred_file)
    print(f"[INFO] error positions: {len(error_df)}")
    print(f"[INFO] error_pred_file: {error_pred_file}")

    context_records = []
    if context_file:
        context_records = load_context_records(context_file)
        print(f"[INFO] loaded rich context records: {len(context_records)} from {context_file}")
    else:
        print("[WARN] no rich context file found; fallback context will be used.")

    context_map = build_context_map(context_records)
    print(f"[INFO] indexed rich contexts: {len(context_map)}")

    if fallback_context_file:
        flat_records = load_context_records(fallback_context_file)
        merge_flat_context_into_map(context_map, flat_records)
        print(f"[INFO] merged fallback flat contexts: {len(flat_records)} from {fallback_context_file}")
        print(f"[INFO] indexed contexts after flat merge: {len(context_map)}")
    else:
        print("[INFO] no fallback flat context file loaded.")

    dirty_df = load_dirty_table(dirty_table_csv)
    if dirty_df is not None:
        print(f"[INFO] loaded dirty table: {dirty_table_csv}, shape={dirty_df.shape}")
    else:
        print("[WARN] dirty table not loaded; dictionary-based candidates will be weaker.")

    global_dict = build_global_dictionaries(context_records, dirty_df)
    print("[INFO] built Soccer global dictionaries:")
    print(f"  column dictionaries: {len(global_dict['column_value_counter'])}")
    print(f"  team groups: {len(global_dict['team_distribution'])}")
    print(f"  team-season groups: {len(global_dict['team_season_distribution'])}")
    print(f"  stadium-city groups: {len(global_dict['stadium_city_distribution'])}")
    print(f"  profile-key groups: {len(global_dict['profile_key_distribution'])}")
    print(f"  entity-pair groups: {len(global_dict['entity_pair_distribution'])}")

    expanded_rows = []
    grouped_rows = []
    augmented_rows = []
    missing_context_count = 0

    for _, error_row in error_df.iterrows():
        row_id = int(error_row["row_id"])
        column = str(error_row["column"])
        dirty_value = canonical_empty_text(error_row.get("value", ""))

        ctx = context_map.get((row_id, column))
        if ctx is None:
            missing_context_count += 1
            ctx = build_fallback_context(row_id, column, error_row, dirty_df)

        if dirty_value in {"", "empty"}:
            dirty_value = canonical_empty_text(ctx.get("value", dirty_value))
            if dirty_value in {"", "empty"}:
                rv = (ctx.get("row_context", {}) or {}).get("row_values", {}) if isinstance(ctx.get("row_context", {}), dict) else {}
                dirty_value = get_row_val(rv, column)

        candidates = generate_candidates_for_one_cell(row_id, column, dirty_value, ctx, global_dict)

        grouped_row = {
            "row_id": row_id,
            "column": column,
            "dirty_value": dirty_value,
            "candidate_count": len(candidates),
            "candidate_values": "|".join([c["candidate_value"] for c in candidates]),
            "candidate_sources": " || ".join([c["candidate_sources"] for c in candidates]),
            "candidate_scores": "|".join([f"{float(c['source_score']):.6f}" for c in candidates]),
            "has_rule_candidate": int(any(c.get("is_from_rule", 0) for c in candidates)),
            "has_soccer_profile_candidate": int(any(c.get("is_from_soccer_profile", 0) for c in candidates)),
            "has_soccer_direct_rule_candidate": int(any(c.get("is_from_soccer_direct_rule", 0) for c in candidates)),
            "has_neighbor_candidate": int(any(c.get("is_from_neighbor", 0) for c in candidates)),
            "has_domain_candidate": int(any(c.get("is_from_domain", 0) for c in candidates)),
            "has_numeric_candidate": int(any(c.get("is_from_numeric_rule", 0) for c in candidates)),
            "has_profile_key_majority_candidate": int(any(c.get("is_from_profile_key_majority", 0) for c in candidates)),
            "has_entity_pair_majority_candidate": int(any(c.get("is_from_entity_pair_majority", 0) for c in candidates)),
            "has_team_season_position_candidate": int(any(c.get("is_from_team_season_position", 0) for c in candidates)),
        }
        grouped_rows.append(grouped_row)

        augmented_rows.append(augment_error_row(error_row, ctx, len(candidates), candidates))

        for rank, c in enumerate(candidates, start=1):
            out = error_row.to_dict()
            out.update({
                "row_id": row_id,
                "column": column,
                "dirty_value": dirty_value,
                "candidate_value": c["candidate_value"],
                "candidate_rank": rank,
                "candidate_sources": c["candidate_sources"],
                "candidate_source_text": c["candidate_sources"],
                "source_count": c["source_count"],
                "source_score": c["source_score"],
                "is_from_rule": c["is_from_rule"],
                "is_from_neighbor": c["is_from_neighbor"],
                "is_from_column_top": c["is_from_column_top"],
                "is_from_similarity": c["is_from_similarity"],
                "is_from_hint": c["is_from_hint"],
                "is_from_dictionary": c["is_from_dictionary"],
                "is_from_pattern_restore": c["is_from_pattern_restore"],
                "is_from_soccer_profile": c["is_from_soccer_profile"],
                "is_from_soccer_direct_rule": c["is_from_soccer_direct_rule"],
                "is_from_domain": c["is_from_domain"],
                "is_from_numeric_rule": c["is_from_numeric_rule"],
                "is_from_profile_key_majority": c.get("is_from_profile_key_majority", 0),
                "is_from_entity_pair_majority": c.get("is_from_entity_pair_majority", 0),
                "is_from_team_season_position": c.get("is_from_team_season_position", 0),
                "profile_key_group_size": c.get("profile_key_group_size", 0),
                "profile_key_top_count": c.get("profile_key_top_count", 0),
                "profile_key_majority_ratio": c.get("profile_key_majority_ratio", 0.0),
                "entity_pair_group_size": c.get("entity_pair_group_size", 0),
                "entity_pair_top_count": c.get("entity_pair_top_count", 0),
                "entity_pair_majority_ratio": c.get("entity_pair_majority_ratio", 0.0),
                "team_season_position_group_size": c.get("team_season_position_group_size", 0),
                "team_season_position_top_count": c.get("team_season_position_top_count", 0),
                "team_season_position_ratio": c.get("team_season_position_ratio", 0.0),
                "extra_info": c["extra_info"],
            })
            expanded_rows.append(out)

    expanded_df = pd.DataFrame(expanded_rows)
    grouped_df = pd.DataFrame(grouped_rows)
    augmented_df = pd.DataFrame(augmented_rows)

    expanded_df.to_csv(expanded_path, index=False, encoding="utf-8-sig")
    grouped_df.to_csv(grouped_path, index=False, encoding="utf-8-sig")
    augmented_df.to_csv(augmented_path, index=False, encoding="utf-8-sig")

    summary_rows = []
    if not grouped_df.empty:
        for col, g in grouped_df.groupby("column", sort=False):
            summary_rows.append({
                "column": col,
                "cells": len(g),
                "cells_with_candidates": int((g["candidate_count"] > 0).sum()),
                "avg_candidate_count": float(g["candidate_count"].mean()),
                "max_candidate_count": int(g["candidate_count"].max()),
                "has_rule_candidate_cells": int(g["has_rule_candidate"].sum()),
                "has_soccer_profile_candidate_cells": int(g["has_soccer_profile_candidate"].sum()),
                "has_soccer_direct_rule_candidate_cells": int(g["has_soccer_direct_rule_candidate"].sum()),
                "has_neighbor_candidate_cells": int(g["has_neighbor_candidate"].sum()),
                "has_domain_candidate_cells": int(g["has_domain_candidate"].sum()),
                "has_numeric_candidate_cells": int(g["has_numeric_candidate"].sum()),
                "has_profile_key_majority_candidate_cells": int(g["has_profile_key_majority_candidate"].sum()) if "has_profile_key_majority_candidate" in g.columns else 0,
                "has_entity_pair_majority_candidate_cells": int(g["has_entity_pair_majority_candidate"].sum()) if "has_entity_pair_majority_candidate" in g.columns else 0,
                "has_team_season_position_candidate_cells": int(g["has_team_season_position_candidate"].sum()) if "has_team_season_position_candidate" in g.columns else 0,
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n========== Soccer Candidate Generation Summary ==========")
    print(f"input error cells: {len(error_df)}")
    print(f"missing context fallback used: {missing_context_count}")
    print(f"expanded candidate rows: {len(expanded_df)}")
    print(f"grouped candidate cells: {len(grouped_df)}")

    if not grouped_df.empty:
        print(f"cells with candidates: {(grouped_df['candidate_count'] > 0).sum()}")
        print(f"avg candidates per cell: {grouped_df['candidate_count'].mean():.4f}")

    print(f"\n[OK] expanded candidates: {expanded_path}")
    print(f"[OK] grouped candidates: {grouped_path}")
    print(f"[OK] augmented allowed cells: {augmented_path}")
    print(f"[OK] summary: {summary_path}")

    if not summary_df.empty:
        print("\n---- Column summary ----")
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

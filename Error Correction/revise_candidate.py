#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified readable repair-candidate generator.

Datasets:
  hospital, flights, beers, rayyan, adult, soccer

Purpose
-------
Generate repair candidates for cells predicted as errors.

Common output filenames:
  repair_candidates_expanded.csv
  repair_candidates_grouped.csv

Optional augmented whitelist / summary:
  predicted_error_positions_<dataset>_augmented.csv
  <dataset>_candidate_generation_summary.csv

This is a readable unified implementation. It is not a base64/plugin wrapper.
Dataset-specific behavior is isolated in DATASET_CONFIGS and candidate plugin
functions.

High-level candidate sources
----------------------------
Common:
  - keep_original
  - suggested_correct_value / LLM hint fields
  - neighbor_majority_value
  - rule expected values from context JSON
  - context candidates / evidence candidates
  - row/similar/neighbor values
  - column top values from dirty table
  - edit-near / canonical values when applicable

Dataset-specific:
  hospital: measure / score / stateavg / condition hints
  flights: time canonicalization and source/time consensus
  beers: state-city, ABV/IBU/ounces, brewery/style profile
  rayyan: language/ISSN/date/volume/issue/pagination canonical candidates
  adult: relationship/sex/marital/education/age/hour domain/profile candidates
  soccer: position/year/team-season/profile-key/entity-pair candidates

Notes
-----
The goal is a unified experimental code path for ablations. Because old scripts
were independently evolved, byte-for-byte equality is not guaranteed, but output
filenames and main schemas are preserved.
"""

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


# ============================================================
# 1. Dataset configs
# ============================================================

EMPTY_TOKENS = {"", "nan", "none", "null", "na", "n/a", "missing", "empty", "?", "{null}"}


@dataclass
class DatasetConfig:
    name: str
    default_error_pred_file: str
    default_context_jsonl: str
    default_context_flat_file: str = ""
    default_dirty_table: str = ""
    output_expanded: str = "repair_candidates_expanded.csv"
    output_grouped: str = "repair_candidates_grouped.csv"
    output_augmented: str = ""
    output_summary: str = ""
    target_columns: set = field(default_factory=set)
    categorical_domains: Dict[str, List[str]] = field(default_factory=dict)
    include_dirty_value: bool = True
    max_candidates_per_cell: int = 30
    top_k_column_values: int = 8


DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "hospital": DatasetConfig(
        name="hospital",
        default_error_pred_file="predicted_error_positions.csv",
        default_context_jsonl="/mnt/mydata/dq/projects/Splittree/test/Error Detection new/candidate_llm_contexts_v2.jsonl",
        default_dirty_table="/mnt/mydata/dq/projects/Splittree/hospital/hospital_dirty.csv",
        target_columns={"MeasureCode", "MeasureName", "Condition", "Stateavg", "Score", "Sample"},
        categorical_domains={},
    ),
    "flights": DatasetConfig(
        name="flights",
        default_error_pred_file="predicted_error_positions.csv",
        default_context_jsonl="/mnt/mydata/dq/projects/Splittree/test/Error Detection flights/candidate_llm_contexts_flights.jsonl",
        default_context_flat_file="/mnt/mydata/dq/projects/Splittree/test/Error Detection flights/candidate_llm_contexts_flat_flights.csv",
        default_dirty_table="/mnt/mydata/dq/projects/Splittree/flights/flights_dirty.csv",
        target_columns={"sched_dep_time", "act_dep_time", "sched_arr_time", "act_arr_time"},
    ),
    "beers": DatasetConfig(
        name="beers",
        default_error_pred_file="predicted_error_positions.csv",
        default_context_jsonl="/mnt/mydata/dq/projects/Splittree/test/Error Detection beers/candidate_llm_contexts_beers.jsonl",
        default_dirty_table="/mnt/mydata/dq/projects/Splittree/beers/beers_dirty.csv",
        target_columns={"state", "city", "ounces", "abv", "ibu", "brewery_id", "brewery_name", "beer_name", "style"},
    ),
    "rayyan": DatasetConfig(
        name="rayyan",
        default_error_pred_file="predicted_error_positions.csv",
        default_context_jsonl="/mnt/mydata/dq/projects/Splittree/test/Error Detection rayyan/candidate_llm_contexts_rayyan.jsonl",
        default_dirty_table="/mnt/mydata/dq/projects/Splittree/rayyan/rayyan_dirty.csv",
        target_columns={"article_language", "journal_issn", "journal_title", "journal_abbreviation", "jounral_abbreviation", "article_jcreated_at", "article_jvolumn", "article_jissue", "article_pagination", "article_title"},
    ),
    "adult": DatasetConfig(
        name="adult",
        default_error_pred_file="predicted_error_positions.csv",
        default_context_jsonl="/mnt/mydata/dq/projects/Splittree/test/Error Detection Adult/candidate_llm_contexts_adult.jsonl",
        default_dirty_table="/mnt/mydata/dq/projects/Splittree/adult/adult_dirty.csv",
        output_augmented="predicted_error_positions_adult_augmented.csv",
        output_summary="adult_candidate_generation_summary.csv",
        target_columns={"relationship", "sex", "maritalstatus", "education", "age", "hoursperweek", "income", "workclass", "occupation"},
        categorical_domains={
            "relationship": ["Husband", "Wife", "Own-child", "Not-in-family", "Unmarried", "Other-relative"],
            "sex": ["Male", "Female"],
            "income": ["LessThan50K", "MoreThan50K", "<=50K", ">50K"],
        },
    ),
    "soccer": DatasetConfig(
        name="soccer",
        default_error_pred_file="predicted_error_positions.csv",
        default_context_jsonl="/mnt/mydata/dq/projects/Splittree/test/Error Detection Soccer/candidate_llm_contexts_soccer.jsonl",
        default_dirty_table="/mnt/mydata/dq/projects/Splittree/soccer/soccer_dirty.csv",
        output_augmented="predicted_error_positions_soccer_augmented.csv",
        output_summary="soccer_candidate_generation_summary.csv",
        target_columns={"name", "surname", "birthyear", "birthplace", "position", "team", "city", "stadium", "season", "manager"},
        categorical_domains={
            "position": ["Defender", "Forward", "Goalkeeper", "Midfield", "Midfielder", "Striker", "Winger"],
        },
    ),
}


# ============================================================
# 2. Utilities
# ============================================================

def is_empty(x: Any) -> bool:
    if x is None:
        return True
    try:
        if pd.isna(x):
            return True
    except Exception:
        pass
    return str(x).strip().lower() in EMPTY_TOKENS


def clean_text(x: Any) -> str:
    if is_empty(x):
        return ""
    return str(x).strip()


def norm_text(x: Any) -> str:
    return clean_text(x).lower()


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if is_empty(x):
            return default
        return float(str(x).strip())
    except Exception:
        return default


def safe_int(x: Any, default: int = 0) -> int:
    try:
        if is_empty(x):
            return default
        return int(float(str(x).strip()))
    except Exception:
        return default


def normalize_col(column: Any, dataset: str = "") -> str:
    c = norm_text(column)
    aliases = {
        "marital-status": "maritalstatus",
        "marital_status": "maritalstatus",
        "hours-per-week": "hoursperweek",
        "hours_per_week": "hoursperweek",
        "birth_year": "birthyear",
        "birth year": "birthyear",
        "birth-year": "birthyear",
        "year": "season",
        "club": "team",
        "coach": "manager",
        "jounral_abbreviation": "journal_abbreviation",
    }
    return aliases.get(c, c)


def canonical_year(x: Any) -> str:
    s = clean_text(x)
    if not s:
        return ""
    s2 = (
        s.replace("O", "0").replace("o", "0")
        .replace("I", "1").replace("l", "1").replace("|", "1")
        .replace("S", "5").replace("s", "5").replace("B", "8")
    )
    m = re.search(r"(18\d{2}|19\d{2}|20\d{2})", s2)
    if m:
        return m.group(1)
    try:
        f = float(s2)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return s


def canonical_position(x: Any) -> str:
    s = clean_text(x)
    if not s:
        return ""
    key = re.sub(r"[^a-z]", "", s.lower())
    mp = {
        "defender": "Defender", "defenders": "Defender", "dfender": "Defender",
        "forward": "Forward", "forwards": "Forward", "foward": "Forward",
        "goalkeeper": "Goalkeeper", "goalkeepers": "Goalkeeper", "gk": "Goalkeeper",
        "midfield": "Midfield", "midfielder": "Midfield", "midfields": "Midfield",
        "striker": "Striker", "winger": "Winger",
    }
    return mp.get(key, s)


def canonical_time(x: Any) -> str:
    s = clean_text(x)
    if not s:
        return ""
    digits = re.sub(r"\D", "", s)
    if digits == "":
        return s
    try:
        v = int(digits)
        if 0 <= v <= 2359:
            hh = v // 100
            mm = v % 100
            if 0 <= hh <= 23 and 0 <= mm <= 59:
                return f"{hh:02d}{mm:02d}"
    except Exception:
        pass
    return s


def canonical_language(x: Any) -> str:
    s = norm_text(x)
    mp = {
        "english": "eng", "en": "eng", "eng": "eng",
        "french": "fre", "fr": "fre", "fre": "fre", "fra": "fre",
        "german": "ger", "de": "ger", "ger": "ger", "deu": "ger",
        "spanish": "spa", "es": "spa", "spa": "spa",
        "japanese": "jpn", "ja": "jpn", "jpn": "jpn",
        "chinese": "chi", "zh": "chi", "chi": "chi",
    }
    return mp.get(s, clean_text(x))


def canonical_issn(x: Any) -> str:
    s = clean_text(x).upper().replace("ISSN", "")
    chars = re.sub(r"[^0-9X]", "", s)
    if len(chars) == 8:
        return f"{chars[:4]}-{chars[4:]}"
    return clean_text(x)


def canonical_value(dataset: str, column: str, value: Any) -> str:
    col = normalize_col(column, dataset)
    if dataset == "flights" and ("time" in col or col in {"sched_dep_time", "act_dep_time", "sched_arr_time", "act_arr_time"}):
        return canonical_time(value)
    if dataset == "soccer":
        if col in {"birthyear", "season"}:
            return canonical_year(value)
        if col == "position":
            return canonical_position(value)
    if dataset == "rayyan":
        if col == "article_language":
            return canonical_language(value)
        if col == "journal_issn":
            return canonical_issn(value)
        if col in {"article_jvolumn", "article_jissue"}:
            m = re.search(r"\d+", clean_text(value))
            return m.group(0) if m else clean_text(value)
    if dataset == "adult":
        if col in {"age", "hoursperweek"}:
            m = re.search(r"\d+", clean_text(value))
            return m.group(0) if m else clean_text(value)
    return clean_text(value)


def add_candidate(cands: Dict[str, Dict[str, Any]], value: Any, source: str, score: float, extra: Optional[Dict[str, Any]] = None):
    val = clean_text(value)
    if not val:
        return
    key = norm_text(val)
    if key not in cands:
        cands[key] = {
            "candidate_value": val,
            "candidate_sources": [],
            "source_score": 0.0,
            "extra_info": [],
            "is_from_rule": 0,
            "is_from_neighbor": 0,
            "is_from_similarity": 0,
            "is_from_hint": 0,
            "is_from_dictionary": 0,
            "is_from_column_top": 0,
            "is_from_domain": 0,
            "is_from_numeric_rule": 0,
        }
    item = cands[key]
    if source not in item["candidate_sources"]:
        item["candidate_sources"].append(source)
    item["source_score"] = max(float(item.get("source_score", 0.0)), float(score))
    low = source.lower()
    if "rule" in low or "expected" in low:
        item["is_from_rule"] = 1
    if "neighbor" in low:
        item["is_from_neighbor"] = 1
    if "similar" in low:
        item["is_from_similarity"] = 1
    if "hint" in low or "llm" in low or "suggested" in low:
        item["is_from_hint"] = 1
    if "dictionary" in low:
        item["is_from_dictionary"] = 1
    if "column_top" in low:
        item["is_from_column_top"] = 1
    if "domain" in low:
        item["is_from_domain"] = 1
    if "numeric" in low or "year" in low or "time" in low:
        item["is_from_numeric_rule"] = 1
    if extra:
        for k, v in extra.items():
            if isinstance(v, (int, float)):
                item[k] = max(safe_float(item.get(k, 0), 0), float(v))
            else:
                item[k] = v
        item["extra_info"].append(json.dumps(extra, ensure_ascii=False))


def load_jsonl_context(path: str) -> Dict[Tuple[int, str], Dict[str, Any]]:
    out = {}
    if not path or not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                key = (int(obj["row_id"]), str(obj["column"]))
                out[key] = obj
            except Exception as e:
                print(f"[WARN] bad jsonl line={line_no}, err={e}")
    return out


def load_table(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def normalize_key_df(df: pd.DataFrame, desc: str) -> pd.DataFrame:
    if not {"row_id", "column"}.issubset(df.columns):
        raise ValueError(f"{desc} must contain row_id,column")
    out = df.copy()
    out["row_id"] = pd.to_numeric(out["row_id"], errors="raise").astype(int)
    out["column"] = out["column"].astype(str).str.strip()
    return out


# ============================================================
# 3. Global dictionaries
# ============================================================

def build_global_stats(dirty_df: pd.DataFrame, cfg: DatasetConfig) -> Dict[str, Any]:
    stats = {
        "column_counts": defaultdict(Counter),
        "row_values": {},
        "profile_counts": defaultdict(lambda: defaultdict(Counter)),
    }
    if dirty_df.empty:
        return stats

    for col in dirty_df.columns:
        for v in dirty_df[col]:
            cv = canonical_value(cfg.name, col, v)
            if cv:
                stats["column_counts"][str(col)][cv] += 1

    for ridx, row in dirty_df.iterrows():
        stats["row_values"][int(ridx)] = {str(c): row[c] for c in dirty_df.columns}

        # Dataset-specific profile cooccurrence.
        rv = {normalize_col(c, cfg.name): row[c] for c in dirty_df.columns}
        if cfg.name == "soccer":
            keys = [
                ("surname_to_name", clean_text(rv.get("surname")), "name", rv.get("name")),
                ("name_to_surname", clean_text(rv.get("name")), "surname", rv.get("surname")),
                ("name_surname_to_birthplace", clean_text(rv.get("name")) + "||" + clean_text(rv.get("surname")), "birthplace", rv.get("birthplace")),
                ("team_season_position", clean_text(rv.get("team")) + "||" + canonical_year(rv.get("season")), "position", rv.get("position")),
            ]
        elif cfg.name == "adult":
            keys = [
                ("relationship_to_sex", clean_text(rv.get("relationship")), "sex", rv.get("sex")),
                ("relationship_to_marital", clean_text(rv.get("relationship")), "maritalstatus", rv.get("maritalstatus")),
                ("education_to_age", clean_text(rv.get("education")), "age", rv.get("age")),
            ]
        elif cfg.name == "beers":
            keys = [
                ("city_to_state", clean_text(rv.get("city")), "state", rv.get("state")),
                ("brewery_to_city", clean_text(rv.get("brewery_name")), "city", rv.get("city")),
                ("beer_to_style", clean_text(rv.get("beer_name")), "style", rv.get("style")),
            ]
        elif cfg.name == "flights":
            keys = [
                ("flight_to_sched_dep", clean_text(rv.get("flight")), "sched_dep_time", rv.get("sched_dep_time")),
                ("flight_to_sched_arr", clean_text(rv.get("flight")), "sched_arr_time", rv.get("sched_arr_time")),
            ]
        elif cfg.name == "rayyan":
            keys = [
                ("journal_to_issn", clean_text(rv.get("journal_title")), "journal_issn", rv.get("journal_issn")),
                ("journal_to_abbrev", clean_text(rv.get("journal_title")), "journal_abbreviation", rv.get("journal_abbreviation", rv.get("jounral_abbreviation"))),
            ]
        else:
            keys = []

        for key_name, key_val, target_col, target_val in keys:
            if key_val and not is_empty(target_val):
                stats["profile_counts"][(key_name, key_val)][normalize_col(target_col, cfg.name)][
                    canonical_value(cfg.name, target_col, target_val)
                ] += 1

    return stats


def add_column_top_candidates(cands, column, cfg, stats, dirty_value):
    counts = stats["column_counts"].get(str(column), Counter())
    for idx, (val, cnt) in enumerate(counts.most_common(cfg.top_k_column_values)):
        if norm_text(val) == norm_text(dirty_value):
            continue
        add_candidate(
            cands,
            val,
            "column_top_value",
            0.35 - idx * 0.01,
            {"column_top_count": cnt, "column_top_rank": idx + 1},
        )


def add_domain_candidates(cands, column, cfg, dirty_value):
    col = normalize_col(column, cfg.name)
    for idx, val in enumerate(cfg.categorical_domains.get(col, [])):
        if norm_text(val) == norm_text(dirty_value):
            continue
        add_candidate(cands, val, f"{cfg.name}_domain_value", 0.50 - idx * 0.01)


def extract_values_recursive(obj: Any, keys_hint: Optional[set] = None, max_values: int = 30) -> List[str]:
    vals = []
    if obj is None:
        return vals
    if isinstance(obj, dict):
        for k, v in obj.items():
            lk = str(k).lower()
            if keys_hint and any(h in lk for h in keys_hint):
                if isinstance(v, (str, int, float)) and not is_empty(v):
                    vals.append(clean_text(v))
            vals.extend(extract_values_recursive(v, keys_hint, max_values=max_values))
            if len(vals) >= max_values:
                break
    elif isinstance(obj, list):
        for x in obj:
            vals.extend(extract_values_recursive(x, keys_hint, max_values=max_values))
            if len(vals) >= max_values:
                break
    return vals[:max_values]


def add_context_candidates(cands, ctx, column, cfg, dirty_value):
    if not isinstance(ctx, dict):
        return

    # Direct/common fields.
    direct_fields = [
        "suggested_correct_value", "repair_value", "correct_value",
        "expected_value", "neighbor_majority_value", "similar_row_value",
        "candidate_value", "top_candidate_value",
    ]
    for f in direct_fields:
        if f in ctx:
            add_candidate(cands, canonical_value(cfg.name, column, ctx[f]), f"context_{f}", 0.75)

    # Nested likely candidate fields.
    vals = extract_values_recursive(
        ctx,
        keys_hint={"candidate", "suggest", "expected", "repair", "majority", "neighbor", "similar", "value"},
        max_values=80,
    )
    for v in vals:
        cv = canonical_value(cfg.name, column, v)
        if cv and norm_text(cv) != norm_text(dirty_value):
            add_candidate(cands, cv, "context_recursive_candidate", 0.48)

    # Row context current value for same row from rich JSON.
    row_context = ctx.get("row_context", {})
    if isinstance(row_context, dict):
        row_values = row_context.get("row_values", row_context)
        if isinstance(row_values, dict) and column in row_values:
            cv = canonical_value(cfg.name, column, row_values[column])
            if cv and norm_text(cv) != norm_text(dirty_value):
                add_candidate(cands, cv, "row_context_value", 0.42)


# ============================================================
# 4. Dataset-specific candidates
# ============================================================

def add_dataset_specific_candidates(cands, row, ctx, cfg, stats, dirty_df):
    dataset = cfg.name
    col = normalize_col(row["column"], dataset)
    dirty = clean_text(row.get("value", row.get("dirty_value", "")))
    row_id = safe_int(row.get("row_id"), -1)

    # Raw dirty table row values.
    row_vals = stats["row_values"].get(row_id, {})
    rv = {normalize_col(k, dataset): v for k, v in row_vals.items()}

    if dataset == "flights":
        if "time" in col or col in {"sched_dep_time", "act_dep_time", "sched_arr_time", "act_arr_time"}:
            cv = canonical_time(dirty)
            if cv and cv != dirty:
                add_candidate(cands, cv, "time_format_canonicalization", 0.92)
            for k in ["time_consensus_value", "trusted_consensus_value", "source_consensus_value", "neighbor_majority_value"]:
                if k in row:
                    add_candidate(cands, canonical_time(row[k]), k, 0.86)
            for k in ["sched_dep_time", "act_dep_time", "sched_arr_time", "act_arr_time"]:
                if k in rv and k != col:
                    add_candidate(cands, canonical_time(rv[k]), "row_time_context", 0.40)

    elif dataset == "beers":
        if col == "state" and "city" in rv:
            counter = stats["profile_counts"].get(("city_to_state", clean_text(rv.get("city"))), {}).get("state", Counter())
            for val, cnt in counter.most_common(3):
                add_candidate(cands, val, "beers_city_to_state_profile", 0.88, {"profile_count": cnt})
        if col == "city" and "brewery_name" in rv:
            counter = stats["profile_counts"].get(("brewery_to_city", clean_text(rv.get("brewery_name"))), {}).get("city", Counter())
            for val, cnt in counter.most_common(3):
                add_candidate(cands, val, "beers_brewery_to_city_profile", 0.82, {"profile_count": cnt})
        if col in {"abv", "ibu", "ounces"}:
            m = re.search(r"\d+(\.\d+)?", dirty)
            if m:
                add_candidate(cands, m.group(0), "beers_numeric_strip", 0.90)
        if col == "style" and "beer_name" in rv:
            counter = stats["profile_counts"].get(("beer_to_style", clean_text(rv.get("beer_name"))), {}).get("style", Counter())
            for val, cnt in counter.most_common(3):
                add_candidate(cands, val, "beers_beer_to_style_profile", 0.78, {"profile_count": cnt})

    elif dataset == "rayyan":
        if col == "article_language":
            cv = canonical_language(dirty)
            if cv and cv != dirty:
                add_candidate(cands, cv, "rayyan_language_canonical", 0.92)
        if col == "journal_issn":
            cv = canonical_issn(dirty)
            if cv and cv != dirty:
                add_candidate(cands, cv, "rayyan_issn_canonical", 0.92)
            jt = clean_text(rv.get("journal_title"))
            counter = stats["profile_counts"].get(("journal_to_issn", jt), {}).get("journal_issn", Counter())
            for val, cnt in counter.most_common(3):
                add_candidate(cands, val, "rayyan_journal_to_issn_profile", 0.88, {"profile_count": cnt})
        if col in {"article_jvolumn", "article_jissue"}:
            m = re.search(r"\d+", dirty)
            if m:
                add_candidate(cands, m.group(0), "rayyan_numeric_canonical", 0.86)

    elif dataset == "adult":
        if col in cfg.categorical_domains:
            add_domain_candidates(cands, col, cfg, dirty)
        if col == "sex":
            rel = clean_text(rv.get("relationship"))
            counter = stats["profile_counts"].get(("relationship_to_sex", rel), {}).get("sex", Counter())
            for val, cnt in counter.most_common(2):
                add_candidate(cands, val, "adult_relationship_to_sex_profile", 0.86, {"profile_count": cnt})
        if col == "maritalstatus":
            rel = clean_text(rv.get("relationship"))
            counter = stats["profile_counts"].get(("relationship_to_marital", rel), {}).get("maritalstatus", Counter())
            for val, cnt in counter.most_common(3):
                add_candidate(cands, val, "adult_relationship_to_marital_profile", 0.82, {"profile_count": cnt})
        if col in {"age", "hoursperweek"}:
            m = re.search(r"\d+", dirty)
            if m:
                add_candidate(cands, m.group(0), "adult_numeric_strip", 0.90)

    elif dataset == "soccer":
        if col == "position":
            cv = canonical_position(dirty)
            if cv and cv != dirty:
                add_candidate(cands, cv, "soccer_position_canonicalization", 0.94)
            team = clean_text(rv.get("team"))
            season = canonical_year(rv.get("season"))
            counter = stats["profile_counts"].get(("team_season_position", f"{team}||{season}"), {}).get("position", Counter())
            for val, cnt in counter.most_common(4):
                add_candidate(cands, val, "team_season_position_profile", 0.88, {"team_season_position_count": cnt})
            add_domain_candidates(cands, col, cfg, dirty)
        if col in {"birthyear", "season"}:
            cv = canonical_year(dirty)
            if cv and cv != dirty:
                add_candidate(cands, cv, "soccer_year_canonicalization", 0.94)
        if col == "name":
            surname = clean_text(rv.get("surname"))
            counter = stats["profile_counts"].get(("surname_to_name", surname), {}).get("name", Counter())
            for val, cnt in counter.most_common(4):
                add_candidate(cands, val, "entity_pair_majority_surname_to_name", 0.90, {"entity_pair_top_count": cnt})
        if col == "surname":
            name = clean_text(rv.get("name"))
            counter = stats["profile_counts"].get(("name_to_surname", name), {}).get("surname", Counter())
            for val, cnt in counter.most_common(4):
                add_candidate(cands, val, "entity_pair_majority_name_to_surname", 0.90, {"entity_pair_top_count": cnt})
        if col == "birthplace":
            key = clean_text(rv.get("name")) + "||" + clean_text(rv.get("surname"))
            counter = stats["profile_counts"].get(("name_surname_to_birthplace", key), {}).get("birthplace", Counter())
            for val, cnt in counter.most_common(4):
                add_candidate(cands, val, "entity_pair_majority_name_surname_to_birthplace", 0.88, {"entity_pair_top_count": cnt})


# ============================================================
# 5. Candidate generation
# ============================================================

def generate_for_cell(row, ctx, cfg, stats, dirty_df, args) -> List[Dict[str, Any]]:
    column = str(row["column"])
    dirty_value = clean_text(row.get("value", row.get("dirty_value", "")))
    dirty_value = canonical_value(cfg.name, column, dirty_value) if dirty_value else dirty_value

    cands: Dict[str, Dict[str, Any]] = {}

    if cfg.include_dirty_value and args.include_dirty_value:
        add_candidate(cands, dirty_value if dirty_value else "Unknown", "keep_original", 0.10)

    # Direct fields from predicted error positions.
    for f in ["suggested_correct_value", "neighbor_majority_value", "top_candidate_value"]:
        if f in row and not is_empty(row[f]):
            add_candidate(cands, canonical_value(cfg.name, column, row[f]), f, 0.78)

    add_context_candidates(cands, ctx, column, cfg, dirty_value)
    add_dataset_specific_candidates(cands, row, ctx, cfg, stats, dirty_df)
    add_column_top_candidates(cands, column, cfg, stats, dirty_value)
    add_domain_candidates(cands, column, cfg, dirty_value)

    # Finalize.
    items = []
    for item in cands.values():
        item = dict(item)
        item["candidate_sources"] = "|".join(item.get("candidate_sources", []))
        item["source_count"] = len(str(item["candidate_sources"]).split("|")) if item.get("candidate_sources") else 0
        item["final_score_from_candidate_gen"] = safe_float(item.get("source_score", 0.0), 0.0) + min(item["source_count"] * 0.03, 0.15)
        item["dirty_value"] = dirty_value
        item["column"] = column
        item["semantic_type"] = row.get("semantic_type", "")
        item["main_rule_type"] = row.get("main_rule_type", "")
        item["main_usage_role"] = row.get("main_usage_role", "")
        item["violation_count"] = row.get("violation_count", "")
        item["conflict_score"] = row.get("conflict_score", "")
        items.append(item)

    items = sorted(items, key=lambda x: (-safe_float(x.get("final_score_from_candidate_gen"), 0), x.get("candidate_value", "")))
    for i, item in enumerate(items, start=1):
        item["candidate_rank"] = i

    return items[: args.max_candidates_per_cell]


def build_grouped_summary(expanded: pd.DataFrame) -> pd.DataFrame:
    if expanded.empty:
        return pd.DataFrame()
    rows = []
    for (row_id, column), g in expanded.groupby(["row_id", "column"], sort=False):
        gg = g.sort_values(["final_score_from_candidate_gen", "candidate_rank"], ascending=[False, True])
        rows.append({
            "row_id": row_id,
            "column": column,
            "dirty_value": gg["dirty_value"].iloc[0] if "dirty_value" in gg.columns else "",
            "candidate_count": len(gg),
            "top_candidate_value": gg["candidate_value"].iloc[0],
            "top_candidate_score": gg["final_score_from_candidate_gen"].iloc[0],
            "top_candidate_sources": gg["candidate_sources"].iloc[0],
            "all_candidates": " || ".join(gg["candidate_value"].astype(str).tolist()),
            "has_rule_candidate": int(pd.to_numeric(gg.get("is_from_rule", 0), errors="coerce").fillna(0).max() > 0),
            "has_neighbor_candidate": int(pd.to_numeric(gg.get("is_from_neighbor", 0), errors="coerce").fillna(0).max() > 0),
            "has_domain_candidate": int(pd.to_numeric(gg.get("is_from_domain", 0), errors="coerce").fillna(0).max() > 0),
            "has_numeric_candidate": int(pd.to_numeric(gg.get("is_from_numeric_rule", 0), errors="coerce").fillna(0).max() > 0),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Unified readable repair-candidate generator.")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_CONFIGS.keys()))
    parser.add_argument("--error_pred_file", default=None)
    parser.add_argument("--context_file", default=None)
    parser.add_argument("--dirty_table_csv", default=None)
    parser.add_argument("--output_expanded", default=None)
    parser.add_argument("--output_grouped", default=None)
    parser.add_argument("--output_augmented", default=None)
    parser.add_argument("--output_summary", default=None)
    parser.add_argument("--include_dirty_value", type=int, default=1)
    parser.add_argument("--max_candidates_per_cell", type=int, default=None)
    parser.add_argument("--disable_dataset_specific_generation", type=int, default=0)
    args = parser.parse_args()

    cfg = DATASET_CONFIGS[args.dataset]
    if args.max_candidates_per_cell is None:
        args.max_candidates_per_cell = cfg.max_candidates_per_cell

    error_file = args.error_pred_file or cfg.default_error_pred_file
    context_file = args.context_file or cfg.default_context_jsonl
    dirty_file = args.dirty_table_csv or cfg.default_dirty_table
    out_expanded = args.output_expanded or cfg.output_expanded
    out_grouped = args.output_grouped or cfg.output_grouped
    out_augmented = args.output_augmented or cfg.output_augmented
    out_summary = args.output_summary or cfg.output_summary

    print("========== Unified repair candidate generation ==========")
    print(f"dataset: {cfg.name}")
    print(f"error_pred_file: {error_file}")
    print(f"context_file: {context_file}")
    print(f"dirty_table_csv: {dirty_file}")
    print(f"output_expanded: {out_expanded}")
    print(f"output_grouped: {out_grouped}")

    err_df = normalize_key_df(pd.read_csv(error_file, encoding="utf-8-sig", low_memory=False), "error_pred_file")
    ctx_map = load_jsonl_context(context_file)
    dirty_df = load_table(dirty_file)
    stats = build_global_stats(dirty_df, cfg)

    rows = []
    for _, row in err_df.iterrows():
        key = (int(row["row_id"]), str(row["column"]))
        ctx = ctx_map.get(key, {})
        candidates = generate_for_cell(row, ctx, cfg, stats, dirty_df, args)
        for c in candidates:
            out = row.to_dict()
            out.update(c)
            rows.append(out)

    expanded = pd.DataFrame(rows)
    if not expanded.empty:
        expanded = expanded.sort_values(["row_id", "column", "candidate_rank"]).reset_index(drop=True)

    grouped = build_grouped_summary(expanded)

    Path(out_expanded).parent.mkdir(parents=True, exist_ok=True)
    expanded.to_csv(out_expanded, index=False, encoding="utf-8-sig")
    grouped.to_csv(out_grouped, index=False, encoding="utf-8-sig")

    if out_augmented:
        aug = err_df.merge(grouped, on=["row_id", "column"], how="left", suffixes=("", "_cand"))
        aug.to_csv(out_augmented, index=False, encoding="utf-8-sig")
        print(f"[OK] augmented whitelist saved: {out_augmented}")

    if out_summary:
        summary_rows = []
        for col, g in grouped.groupby("column", sort=False):
            summary_rows.append({
                "column": col,
                "cells": len(g),
                "avg_candidate_count": float(pd.to_numeric(g["candidate_count"], errors="coerce").fillna(0).mean()),
                "has_rule_candidate_cells": int(pd.to_numeric(g["has_rule_candidate"], errors="coerce").fillna(0).sum()),
                "has_neighbor_candidate_cells": int(pd.to_numeric(g["has_neighbor_candidate"], errors="coerce").fillna(0).sum()),
                "has_domain_candidate_cells": int(pd.to_numeric(g["has_domain_candidate"], errors="coerce").fillna(0).sum()),
                "has_numeric_candidate_cells": int(pd.to_numeric(g["has_numeric_candidate"], errors="coerce").fillna(0).sum()),
            })
        pd.DataFrame(summary_rows).to_csv(out_summary, index=False, encoding="utf-8-sig")
        print(f"[OK] summary saved: {out_summary}")

    print(f"[OK] expanded saved: {out_expanded}, rows={len(expanded)}")
    print(f"[OK] grouped saved: {out_grouped}, rows={len(grouped)}")
    if not grouped.empty:
        print("\nCandidate cells by column:")
        print(grouped["column"].value_counts().head(30).to_string())


if __name__ == "__main__":
    main()

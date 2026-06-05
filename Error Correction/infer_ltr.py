#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified LTR inference + conservative gate + hard-case mining.

Datasets:
  hospital, flights, beers, rayyan, adult, soccer

Purpose
-------
This script merges the common inference logic of the six dataset-specific
infer_ltr.py scripts into one readable implementation.

It preserves the standard output files used by the original scripts:

  inference_all_candidate_scores.csv
  inference_all_candidate_ranked.csv
  inference_top1_repairs.csv
  hard_cases_for_llm.jsonl
  input_whitelist_violations.csv
  output_whitelist_violations.csv

For Soccer, if --debug_export_rejected=1, it additionally writes:
  inference_rejected_candidates_debug.csv

Common inference logic
----------------------
1. Load candidate feature CSV, usually repair_candidate_features_infer_whitelist.csv.
2. Reconstruct derived features and align with feature_columns.json.
3. Load best_model.pt / checkpoint.
4. Score every candidate with the student ranker.
5. Rank candidates per (row_id, column).
6. Apply dataset-aware conservative gate.
7. Write top1 repair decisions and hard cases.

Compatibility
-------------
This script can load two checkpoint styles:

A. Original dataset-specific inference/training scripts:
   state_dict keys look like:
     backbone.0.weight, score_head.weight, reason_head.weight

B. Unified training script generated earlier:
   state_dict keys look like:
     encoder.0.weight, scorer.weight, reason_head.weight

The script detects the checkpoint style automatically.

Ablation-friendly switches
--------------------------
--disable_gate 1
    Directly accept top1 candidate. Useful for w/o Conservative Gate.

--disable_dataset_derived_features 1
    Reconstruct only generic derived features.

--disable_dataset_gate 1
    Keep the score margin gate but disable dataset-specific gate rules.

--score_only 1
    Only write all scores/ranked files; still writes top1 with gate fields.

--output_dir
    If set, output files go there. Otherwise use REPAIR_INFER_OUTPUT_DIR or cwd,
    except hospital default keeps original behavior of writing into model_dir unless
    --output_dir is explicitly set.
"""

import argparse
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# 1. Dataset configs
# ============================================================

EMPTY_TOKENS = {"", "empty", "null", "none", "nan", "na", "n/a", "missing", "?", "{null}", "none/null"}

US_STATE_CODES = {
    "al","ak","az","ar","ca","co","ct","de","fl","ga","hi","ia","id","il","in","ks","ky","la","ma","md","me",
    "mi","mn","mo","ms","mt","nc","nd","ne","nh","nj","nm","nv","ny","oh","ok","or","pa","ri","sc","sd","tn",
    "tx","ut","va","vt","wa","wi","wv","wy","dc"
}

ADULT_DOMAIN_VALUES = {
    "workclass": [
        "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
        "Local-gov", "State-gov", "Without-pay", "Never-worked"
    ],
    "education": [
        "Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th",
        "11th", "12th", "HS-grad", "Some-college", "Assoc-voc",
        "Assoc-acdm", "Bachelors", "Masters", "Prof-school", "Doctorate"
    ],
    "maritalstatus": [
        "Married-civ-spouse", "Divorced", "Never-married", "Separated",
        "Widowed", "Married-spouse-absent", "Married-AF-spouse"
    ],
    "occupation": [
        "Tech-support", "Craft-repair", "Other-service", "Sales",
        "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
        "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
        "Transport-moving", "Priv-house-serv", "Protective-serv",
        "Armed-Forces"
    ],
    "relationship": ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"],
    "race": ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"],
    "sex": ["Female", "Male"],
    "income": ["LessThan50K", "MoreThan50K", "<=50K", ">50K"],
}
EDUCATION_ORDER = {
    "Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4,
    "9th": 5, "10th": 6, "11th": 7, "12th": 8,
    "HS-grad": 9, "Some-college": 10, "Assoc-voc": 11, "Assoc-acdm": 12,
    "Bachelors": 13, "Masters": 14, "Prof-school": 15, "Doctorate": 16,
}
MARRIED_STATUSES = {"Married-civ-spouse", "Married-AF-spouse"}
SPOUSE_RELATIONSHIPS = {"Husband", "Wife"}

SOCCER_POSITION_VALUES = {"Goalkeeper", "Defender", "Midfield", "Midfielder", "Forward", "Striker", "Winger"}
POSITION_CANONICAL_MAP = {
    "goalkeeper": "Goalkeeper", "goalkeepers": "Goalkeeper", "goal keeper": "Goalkeeper",
    "goal_keeper": "Goalkeeper", "gk": "Goalkeeper",
    "defender": "Defender", "defenders": "Defender", "dfender": "Defender",
    "defnder": "Defender", "defeder": "Defender", "defense": "Defender", "defence": "Defender",
    "midfield": "Midfield", "midfielder": "Midfield", "midfielders": "Midfield",
    "midfields": "Midfield", "midfied": "Midfield", "midfiled": "Midfield",
    "forward": "Forward", "forwards": "Forward", "foward": "Forward", "forwad": "Forward",
    "striker": "Striker", "strikers": "Striker", "winger": "Winger", "wingers": "Winger",
}

LANGUAGE_CANONICAL_MAP = {
    "eng": "eng", "english": "eng", "en": "eng", "engl": "eng", "anglais": "eng",
    "fre": "fre", "french": "fre", "fr": "fre", "fra": "fre",
    "ger": "ger", "german": "ger", "de": "ger", "deu": "ger",
    "spa": "spa", "spanish": "spa", "es": "spa",
    "jpn": "jpn", "japanese": "jpn", "ja": "jpn",
    "chi": "chi", "chinese": "chi", "zh": "chi",
    "ita": "ita", "italian": "ita", "it": "ita",
    "dut": "dut", "dutch": "dut", "nl": "dut",
    "pol": "pol", "polish": "pol",
    "por": "por", "portuguese": "por",
    "rus": "rus", "russian": "rus",
}


@dataclass
class DatasetConfig:
    name: str
    default_model_dir: str
    default_allowed_cells_csv: str
    column_margin_thresholds: Dict[str, float]
    default_margin_threshold: float
    protected_nonempty_columns: set = field(default_factory=set)
    primary_repair_columns: set = field(default_factory=set)
    risky_context_columns: set = field(default_factory=set)
    domain_values: Dict[str, List[str]] = field(default_factory=dict)
    output_to_model_dir_by_default: bool = False
    output_rejected_debug: bool = False


DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "hospital": DatasetConfig(
        name="hospital",
        default_model_dir="pairwise_ltr_student_output_final_gate",
        default_allowed_cells_csv="allowed_cells_from_detection.csv",
        column_margin_thresholds={
            "MeasureCode": 0.20,
            "Stateavg": 0.20,
            "Condition": 0.15,
            "MeasureName": 0.12,
            "Score": 0.10,
            "Sample": 0.05,
        },
        default_margin_threshold=0.08,
        protected_nonempty_columns={"MeasureCode", "Condition", "MeasureName", "Stateavg"},
        primary_repair_columns={"MeasureCode", "Condition", "MeasureName", "Stateavg", "Score", "Sample"},
        output_to_model_dir_by_default=True,
    ),
    "flights": DatasetConfig(
        name="flights",
        default_model_dir="pairwise_ltr_student_output_flights",
        default_allowed_cells_csv="predicted_error_positions_consensus_augmented.csv",
        column_margin_thresholds={
            "sched_dep_time": 0.03,
            "act_dep_time": 0.03,
            "sched_arr_time": 0.03,
            "act_arr_time": 0.03,
        },
        default_margin_threshold=0.05,
        primary_repair_columns={"sched_dep_time", "act_dep_time", "sched_arr_time", "act_arr_time"},
    ),
    "beers": DatasetConfig(
        name="beers",
        default_model_dir="pairwise_ltr_student_output_beers",
        default_allowed_cells_csv="predicted_error_positions.csv",
        column_margin_thresholds={
            "state": 0.03,
            "city": 0.04,
            "ounces": 0.03,
            "abv": 0.04,
            "ibu": 0.04,
            "brewery_id": 0.05,
            "brewery_name": 0.05,
            "beer_name": 0.06,
            "style": 0.05,
        },
        default_margin_threshold=0.05,
        primary_repair_columns={"state", "city", "ounces", "abv", "ibu", "brewery_id", "brewery_name", "beer_name", "style"},
    ),
    "rayyan": DatasetConfig(
        name="rayyan",
        default_model_dir="pairwise_ltr_student_output_rayyan",
        default_allowed_cells_csv="predicted_error_positions_consensus_augmented.csv",
        column_margin_thresholds={
            "article_language": 0.02,
            "journal_issn": 0.025,
            "article_jcreated_at": 0.03,
            "article_jvolumn": 0.035,
            "article_jissue": 0.035,
            "article_pagination": 0.04,
            "journal_title": 0.05,
            "jounral_abbreviation": 0.05,
            "journal_abbreviation": 0.05,
            "article_title": 0.06,
        },
        default_margin_threshold=0.05,
        primary_repair_columns={"article_jcreated_at", "article_jvolumn", "article_jissue"},
        risky_context_columns={
            "article_language", "journal_title", "jounral_abbreviation", "journal_abbreviation",
            "journal_issn", "article_pagination", "author_list", "article_title",
        },
    ),
    "adult": DatasetConfig(
        name="adult",
        default_model_dir="pairwise_ltr_student_output_adult",
        default_allowed_cells_csv="predicted_error_positions_adult_augmented.csv",
        column_margin_thresholds={
            "relationship": 0.03,
            "sex": 0.03,
            "maritalstatus": 0.04,
            "education": 0.05,
            "age": 0.05,
            "hoursperweek": 0.05,
            "income": 0.04,
            "workclass": 0.08,
            "occupation": 0.08,
            "country": 0.10,
            "race": 0.10,
        },
        default_margin_threshold=0.06,
        primary_repair_columns={"relationship", "sex", "maritalstatus", "education", "age", "hoursperweek", "income"},
        risky_context_columns={"workclass", "occupation", "country", "race"},
        domain_values=ADULT_DOMAIN_VALUES,
    ),
    "soccer": DatasetConfig(
        name="soccer",
        default_model_dir="pairwise_ltr_student_output_soccer",
        default_allowed_cells_csv="predicted_error_positions_soccer_augmented.csv",
        column_margin_thresholds={
            "position": 0.025,
            "birthyear": 0.040,
            "season": 0.040,
            "team": 0.090,
            "city": 0.090,
            "stadium": 0.090,
            "manager": 0.100,
            "birthplace": 0.110,
            "name": 0.140,
            "surname": 0.140,
        },
        default_margin_threshold=0.070,
        primary_repair_columns={"position", "birthyear", "season"},
        risky_context_columns={"team", "city", "stadium", "manager", "birthplace", "name", "surname"},
        domain_values={"position": sorted(SOCCER_POSITION_VALUES)},
        output_rejected_debug=True,
    ),
}


# ============================================================
# 2. Helpers and canonicalizers
# ============================================================

def norm_text(x: Any) -> str:
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()


def norm_lower(x: Any) -> str:
    return norm_text(x).lower()


def canonical_empty_text(x: Any) -> str:
    s = norm_lower(x)
    return "empty" if s in EMPTY_TOKENS else norm_text(x)


def is_empty_like(x: Any) -> bool:
    return norm_lower(x) in EMPTY_TOKENS


def safe_float(x: Any, default=0.0) -> float:
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


def safe_int(x: Any, default=0) -> int:
    try:
        if x is None:
            return default
        if isinstance(x, float) and math.isnan(x):
            return default
        s = str(x).strip()
        if s == "":
            return default
        return int(float(s))
    except Exception:
        return default


def load_json(path: str, default=None):
    if default is None:
        default = {}
    if not os.path.exists(path):
        print(f"[WARN] json not found: {path}")
        return default
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_jsonl_record(fp, obj):
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")


def normalize_col_name(column: Any) -> str:
    c = norm_lower(column)
    aliases = {
        "marital-status": "maritalstatus",
        "marital_status": "maritalstatus",
        "marital status": "maritalstatus",
        "hours-per-week": "hoursperweek",
        "hours_per_week": "hoursperweek",
        "hours per week": "hoursperweek",
        "native-country": "country",
        "native_country": "country",
        "capital-gain": "capital_gain",
        "capital-loss": "capital_loss",
        "education-num": "education_num",
        "education_num": "education_num",
        "birth_year": "birthyear",
        "birth year": "birthyear",
        "birth-year": "birthyear",
        "player_position": "position",
        "club": "team",
        "football_club": "team",
        "home_city": "city",
        "arena": "stadium",
        "coach": "manager",
        "season_year": "season",
        "year": "season",
        "jounral_abbreviation": "journal_abbreviation",
    }
    return aliases.get(c, c)


def source_text(row: Optional[pd.Series]) -> str:
    if row is None:
        return ""
    return str(
        row.get(
            "candidate_source",
            row.get("candidate_source_text", row.get("candidate_sources", ""))
        )
    ).lower()


def feature_value(row: Optional[pd.Series], key: str, default=0.0) -> float:
    if row is None:
        return default
    return safe_float(row.get(key, default), default)


def looks_like_code(x: Any) -> int:
    s = norm_text(x)
    return int(bool(s) and ((bool(re.search(r"[_\-]", s)) or bool(re.search(r"\d", s))) and len(s) <= 80))


def prefix_token(x: Any) -> str:
    parts = re.split(r"[_\-]", norm_text(x))
    return parts[0] if parts and parts[0] else ""


def family_token(x: Any) -> str:
    parts = re.split(r"[_\-]", norm_text(x))
    return parts[1] if len(parts) >= 2 else (parts[0] if parts else "")


def suffix_token(x: Any) -> str:
    parts = re.split(r"[_\-]", norm_text(x))
    return parts[-1] if parts else ""


def edit_ratio(a: Any, b: Any) -> float:
    import difflib
    a = norm_lower(a)
    b = norm_lower(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return float(difflib.SequenceMatcher(None, a, b).ratio())


# ---------------- Time / flights ----------------

def extract_time_candidate_strings(x):
    raw = canonical_empty_text(x)
    s = raw.lower().strip()
    if s in {"", "empty"}:
        return []

    s_norm = s
    s_norm = s_norm.replace("estimated", " ")
    s_norm = s_norm.replace("(estimated)", " ")
    s_norm = s_norm.replace("est.", " ")
    s_norm = s_norm.replace("est", " ")
    s_norm = re.sub(r"\s+", " ", s_norm).strip()

    candidates = []
    patterns = [
        r"(?P<h>\d{1,2})\s*:\s*(?P<m>\d{2})\s*(?P<ampm>a\.?m\.?|p\.?m\.?|am|pm|a|p)\b",
        r"(?P<h>\d{1,2})\s*:\s*(?P<m>\d{2})\s*(?P<ampm>a|p)(?=[a-z])",
        r"(?P<h>\d{1,2})\s*:\s*(?P<m>\d{2})",
    ]
    for pat in patterns:
        for m in re.finditer(pat, s_norm):
            h = m.group("h")
            mm = m.group("m")
            ampm = m.groupdict().get("ampm", "") or ""
            ampm = ampm.replace(".", "")
            if ampm == "a":
                ampm = "am"
            elif ampm == "p":
                ampm = "pm"
            candidates.append(f"{h}:{mm} {ampm}".strip())

    compact = re.sub(r"\D", "", s_norm)
    if len(compact) in {3, 4} and not re.search(r"[/-]", s_norm):
        candidates.append(compact)
    candidates.append(raw)

    out, seen = [], set()
    for c in candidates:
        c = str(c).strip()
        k = c.lower()
        if c and k not in seen:
            seen.add(k)
            out.append(c)
    return out


def _parse_simple_time_piece(x):
    s = canonical_empty_text(x).lower()
    if s in {"", "empty"}:
        return None
    s = s.replace(".", "")
    s = s.replace("estimated", " ")
    s = s.replace("est", " ")
    s = re.sub(r"\s+", " ", s).strip()

    ampm = None
    if re.search(r"\b(am)\b", s) or re.search(r"\d\s*a\b", s):
        ampm = "am"
    elif re.search(r"\b(pm)\b", s) or re.search(r"\d\s*p\b", s):
        ampm = "pm"
    elif re.search(r"\d:\d{2}a(?=[a-z]|\b)", s):
        ampm = "am"
    elif re.search(r"\d:\d{2}p(?=[a-z]|\b)", s):
        ampm = "pm"

    s2 = re.sub(r"\b(am|pm)\b", "", s)
    s2 = re.sub(r"(?<=\d)\s*[ap](?=[a-z]|\b)", "", s2).strip()

    hour = minute = None
    m = re.search(r"(\d{1,2})\s*:\s*(\d{2})", s2)
    if m:
        hour, minute = int(m.group(1)), int(m.group(2))
    else:
        digits = re.sub(r"\D", "", s2)
        if len(digits) in {3, 4}:
            hour, minute = int(digits[:-2]), int(digits[-2:])

    if hour is None or minute is None or minute < 0 or minute > 59:
        return None

    if ampm == "am":
        if hour == 12:
            hour = 0
        elif not (1 <= hour <= 11):
            return None
    elif ampm == "pm":
        if hour == 12:
            pass
        elif 1 <= hour <= 11:
            hour += 12
        else:
            return None
    else:
        if not (0 <= hour <= 23):
            return None

    return hour * 60 + minute


def parse_time_to_minutes_simple(x):
    raw = canonical_empty_text(x)
    if raw.lower() in {"", "empty"}:
        return None
    for cand in extract_time_candidate_strings(raw):
        minutes = _parse_simple_time_piece(cand)
        if minutes is not None:
            return minutes
    return None


def is_valid_time_simple(x) -> int:
    return int(parse_time_to_minutes_simple(x) is not None)


# ---------------- Hospital ----------------

def is_percent_like(s: Any) -> bool:
    return bool(re.fullmatch(r"\d{1,3}%", norm_text(s)))


def patients_like(s: Any) -> bool:
    return bool(re.fullmatch(r"\d+\s+patients", canonical_empty_text(s).lower()))


def likely_measure_code_candidate(cand: Any) -> bool:
    s = canonical_empty_text(cand).lower()
    return s != "empty" and (s.startswith(("ami", "hf", "pn", "scip")) or ("ami" in s) or ("hf" in s) or ("pn" in s) or ("scip" in s))


def likely_stateavg_candidate(cand: Any) -> bool:
    s = canonical_empty_text(cand).lower()
    return s != "empty" and "_" in s and likely_measure_code_candidate(s)


def normalize_measurecode_like(s: Any) -> str:
    raw = canonical_empty_text(s).lower()
    if "_" in raw:
        raw = raw.split("_", 1)[1]
    return raw.replace("_", "-")


def measure_family_name(code: Any) -> str:
    c = normalize_measurecode_like(code)
    if c.startswith("hf"):
        return "heart failure"
    if c.startswith("ami"):
        return "heart attack"
    if c.startswith("pn"):
        return "pneumonia"
    if c.startswith("scip"):
        return "surgical infection prevention"
    return ""


# ---------------- Beers ----------------

def normalize_state_code(x: Any) -> str:
    s = norm_text(x).lower()
    if s in EMPTY_TOKENS:
        return ""
    if s in US_STATE_CODES:
        return s.upper()
    m = re.search(r"\b([a-z]{2})\b$", s)
    if m and m.group(1) in US_STATE_CODES:
        return m.group(1).upper()
    return ""


def strip_state_suffix_city(x: Any) -> str:
    s = norm_text(x)
    parts = s.split()
    if len(parts) >= 2 and parts[-1].lower().strip(".,;:()[]{}") in US_STATE_CODES:
        return " ".join(parts[:-1]).strip()
    return s


def normalize_ounces(x: Any) -> str:
    s = norm_text(x).lower()
    if s in EMPTY_TOKENS:
        return ""
    s = s.replace("ounces", "ounce").replace("ozs", "oz").replace("0z", "oz").replace("o.z.", "oz")
    m = re.search(r"(-?\d+(?:\.\d+)?)", s)
    if not m:
        return ""
    v = safe_float(m.group(1), None)
    if v is None or v <= 0 or v > 100:
        return ""
    return f"{v:.1f} oz."


def normalize_abv(x: Any) -> str:
    s = norm_text(x).lower().replace("abv", "").replace(" ", "")
    if s in EMPTY_TOKENS:
        return ""
    m = re.search(r"(-?\d+(?:\.\d+)?)", s)
    if not m:
        return ""
    v = safe_float(m.group(1), None)
    if v is None:
        return ""
    if "%" in s:
        if v > 1.0:
            v = v / 100.0
    elif v > 1.0:
        v = v / 100.0
    if v < 0 or v > 1:
        return ""
    return f"{v:.3f}".rstrip("0").rstrip(".")


def abv_percent_direct_repair(x: Any) -> str:
    raw = norm_text(x)
    if "%" not in raw:
        return ""
    return normalize_abv(raw)


def normalize_ibu(x: Any) -> str:
    s = norm_text(x).lower().replace("ibu", "")
    if s in EMPTY_TOKENS:
        return ""
    m = re.search(r"(-?\d+(?:\.\d+)?)", s)
    if not m:
        return ""
    v = safe_float(m.group(1), None)
    if v is None or v < 0 or v > 1000:
        return ""
    return str(int(round(v))) if abs(v - round(v)) < 1e-9 else f"{v:.1f}".rstrip("0").rstrip(".")


def normalize_beers_id(x: Any) -> str:
    s = norm_text(x)
    m = re.search(r"\d+", s)
    return str(int(m.group(0))) if m else ""


def normalize_for_beers_column(column: Any, value: Any) -> str:
    c = normalize_col_name(column)
    if c == "state":
        return normalize_state_code(value)
    if c == "city":
        return strip_state_suffix_city(value)
    if c == "ounces":
        return normalize_ounces(value)
    if c == "abv":
        return normalize_abv(value)
    if c == "ibu":
        return normalize_ibu(value)
    if c in {"index", "id", "brewery_id"}:
        return normalize_beers_id(value)
    return norm_text(value)


def is_valid_beers_candidate(column: Any, value: Any) -> bool:
    c = normalize_col_name(column)
    if c == "state":
        return bool(normalize_state_code(value))
    if c == "city":
        return bool(strip_state_suffix_city(value))
    if c == "ounces":
        return bool(normalize_ounces(value))
    if c == "abv":
        return bool(normalize_abv(value))
    if c == "ibu":
        return bool(normalize_ibu(value))
    if c in {"index", "id", "brewery_id"}:
        return bool(normalize_beers_id(value))
    return canonical_empty_text(value) != "empty"


# ---------------- Rayyan ----------------

def normalize_language_candidate(x: Any):
    s = canonical_empty_text(x)
    if s in {"", "empty"}:
        return None
    lower = norm_text(s).lower()
    if "abstract language" in lower:
        parts = re.split(r"abstract language\s*:\s*", lower)
        if len(parts) >= 2:
            lower = parts[-1].strip(";: ,. ")
    lower = lower.strip(";: ,. ")
    if lower in LANGUAGE_CANONICAL_MAP:
        return LANGUAGE_CANONICAL_MAP[lower]
    token = re.sub(r"[^a-z]", "", lower)
    if token in LANGUAGE_CANONICAL_MAP:
        return LANGUAGE_CANONICAL_MAP[token]
    if len(token) == 3 and token.isalpha():
        return token
    return None


def normalize_issn_candidate(x: Any):
    s = canonical_empty_text(x)
    if s in {"", "empty"}:
        return None
    raw = s.upper().replace("ISSN", "").replace(":", " ").strip()
    chars = re.sub(r"[^0-9X]", "", raw)
    if len(chars) != 8:
        return None
    return f"{chars[:4]}-{chars[4:]}"


def is_valid_issn(x: Any):
    fixed = normalize_issn_candidate(x)
    if not fixed:
        return False
    digits = fixed.replace("-", "")
    total = 0
    try:
        for i, ch in enumerate(digits):
            val = 10 if ch == "X" else int(ch)
            total += val * (8 - i)
        return total % 11 == 0
    except Exception:
        return False


def normalize_date_candidate(x: Any):
    s = canonical_empty_text(x)
    if s in {"", "empty"}:
        return None
    s = s.strip()
    m = re.fullmatch(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{mo}/{d}/{str(y)[-2:]}"
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{2}|\d{4})", s)
    if m:
        mo, d, y = int(m.group(1)), int(m.group(2)), m.group(3)
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{mo}/{d}/{y[-2:]}"
    return None


def normalize_numeric_int_candidate(x: Any, min_v=None, max_v=None):
    s = canonical_empty_text(x)
    if s in {"", "empty"}:
        return None
    m = re.search(r"-?\d+", s)
    if not m:
        return None
    v = int(m.group(0))
    if min_v is not None and v < min_v:
        return None
    if max_v is not None and v > max_v:
        return None
    return str(v)


def normalize_pagination_candidate(x: Any):
    s = canonical_empty_text(x)
    if s in {"", "empty"}:
        return None
    s2 = s.replace("–", "-").replace("—", "-").replace("−", "-")
    s2 = re.sub(r"\s*-\s*", "-", s2.strip())
    return s2 if re.search(r"\d", s2) else None


def normalize_by_rayyan_column(column: Any, value: Any):
    col = normalize_col_name(column)
    if col == "article_language":
        return normalize_language_candidate(value)
    if col == "journal_issn":
        return normalize_issn_candidate(value)
    if col == "article_jcreated_at":
        return normalize_date_candidate(value)
    if col == "article_jvolumn":
        return normalize_numeric_int_candidate(value, min_v=0, max_v=10000)
    if col == "article_jissue":
        return normalize_numeric_int_candidate(value, min_v=0, max_v=1000)
    if col == "article_pagination":
        return normalize_pagination_candidate(value)
    if col in {"index", "id"}:
        return normalize_numeric_int_candidate(value, min_v=0)
    return None


# ---------------- Adult ----------------

def canonical_adult_value(column: Any, value: Any) -> str:
    col = normalize_col_name(column)
    raw = canonical_empty_text(value)
    if raw == "empty":
        return "empty"

    if col == "income":
        s = raw.replace(".", "").replace(" ", "")
        low = s.lower()
        if low in {"<=50k", "<=50", "lessthan50k", "less_than_50k", "less50k", "0"}:
            return "LessThan50K"
        if low in {">50k", ">50", "morethan50k", "more_than_50k", "more50k", "1"}:
            return "MoreThan50K"
        return raw

    domain = ADULT_DOMAIN_VALUES.get(col)
    if domain:
        for v in domain:
            if norm_lower(v) == norm_lower(raw):
                return v
        raw2 = raw.replace("_", "-")
        for v in domain:
            if norm_lower(v) == norm_lower(raw2):
                return v
        raw3 = raw.replace("_", " ")
        for v in domain:
            if norm_lower(v).replace("-", " ") == norm_lower(raw3).replace("-", " "):
                return v
    return raw


def parse_numeric_range(x: Any) -> Tuple[Optional[float], Optional[float]]:
    s = canonical_empty_text(x)
    if s == "empty":
        return None, None
    low = s.lower().replace(" ", "")
    if low.startswith("<"):
        v = safe_float(low[1:], None)
        if v is not None:
            return 0.0, v - 1
    if low.startswith(">"):
        v = safe_float(low[1:], None)
        if v is not None:
            return v + 1, 120.0
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", s)
    if m:
        return float(m.group(1)), float(m.group(2))
    v = safe_float(s, None)
    if v is not None:
        return v, v
    return None, None


def adult_value_valid(column: Any, value: Any) -> bool:
    col = normalize_col_name(column)
    if col in ADULT_DOMAIN_VALUES:
        return canonical_adult_value(col, value) in ADULT_DOMAIN_VALUES[col]
    if col == "age":
        lo, hi = parse_numeric_range(value)
        return lo is not None and hi is not None and 0 <= lo <= hi <= 120
    if col == "hoursperweek":
        lo, hi = parse_numeric_range(value)
        return lo is not None and hi is not None and 0 <= lo <= hi <= 120
    return canonical_empty_text(value) != "empty"


# ---------------- Soccer ----------------

def parse_year_like(x: Any) -> Optional[int]:
    s = canonical_empty_text(x)
    if s == "empty":
        return None
    s = (
        s.replace("O", "0").replace("o", "0")
        .replace("I", "1").replace("l", "1").replace("|", "1")
        .replace("S", "5").replace("s", "5").replace("B", "8")
    )
    m = re.search(r"(\d{4})", s)
    if not m:
        return None
    return safe_int(m.group(1), None)


def canonical_year(x: Any) -> str:
    y = parse_year_like(x)
    if y is None:
        return canonical_empty_text(x)
    return str(y)


def is_valid_birthyear_value(x: Any) -> bool:
    y = parse_year_like(x)
    return y is not None and 1940 <= y <= 2010


def is_valid_season_value(x: Any) -> bool:
    y = parse_year_like(x)
    return y is not None and 1990 <= y <= 2030


def canonical_position(x: Any) -> str:
    raw = canonical_empty_text(x)
    if raw == "empty":
        return "empty"
    s = norm_lower(raw)
    compact = re.sub(r"[^a-z]", "", s)
    if s in POSITION_CANONICAL_MAP:
        return POSITION_CANONICAL_MAP[s]
    if compact in POSITION_CANONICAL_MAP:
        return POSITION_CANONICAL_MAP[compact]
    for v in SOCCER_POSITION_VALUES:
        if norm_lower(v) == s:
            return v
    return raw


def position_domain_valid(x: Any) -> bool:
    return canonical_position(x) in SOCCER_POSITION_VALUES


def age_at_season_valid(birthyear: Any, season: Any) -> bool:
    by = parse_year_like(birthyear)
    sy = parse_year_like(season)
    if by is None or sy is None:
        return True
    return 15 <= sy - by <= 48


# ============================================================
# 3. Model classes and checkpoint loading
# ============================================================

class SharedScorerMLP(nn.Module):
    """Compatible with original inference checkpoints using backbone/score_head."""

    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float, num_reason_classes: int):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.score_head = nn.Linear(prev, 1)
        self.reason_head = nn.Linear(prev, max(num_reason_classes, 1))

    def forward_single(self, x):
        h = self.backbone(x)
        return self.score_head(h).squeeze(-1), self.reason_head(h), h


class UnifiedPairwiseLTRModel(nn.Module):
    """Compatible with unified_train_ltr_full.py checkpoints using encoder/scorer."""

    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float, num_reason_classes: int):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        self.encoder = nn.Sequential(*layers)
        self.scorer = nn.Linear(prev, 1)
        self.reason_head = nn.Linear(prev, max(num_reason_classes, 1))

    def forward_single(self, x):
        h = self.encoder(x)
        return self.scorer(h).squeeze(-1), self.reason_head(h), h


def infer_hidden_dims_from_state_dict(state: Dict[str, torch.Tensor], style: str) -> Tuple[int, List[int]]:
    if style == "original":
        prefix = "backbone"
        score_key = "score_head.weight"
    else:
        prefix = "encoder"
        score_key = "scorer.weight"

    linear_layers = []
    pattern = re.compile(rf"^{prefix}\.(\d+)\.weight$")
    for k, v in state.items():
        m = pattern.match(k)
        if m and getattr(v, "ndim", 0) == 2:
            linear_layers.append((int(m.group(1)), int(v.shape[1]), int(v.shape[0])))
    linear_layers.sort()

    if linear_layers:
        input_dim = linear_layers[0][1]
        hidden_dims = [out_dim for _, _, out_dim in linear_layers]
    else:
        # No hidden layer fallback.
        input_dim = int(state[score_key].shape[1]) if score_key in state else 1
        hidden_dims = []

    return input_dim, hidden_dims


def get_checkpoint_state(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for k in ["model_state_dict", "state_dict", "model"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        # Sometimes torch.save(model.state_dict()) directly.
        if all(isinstance(k, str) for k in ckpt.keys()):
            return ckpt
    raise ValueError("Unsupported checkpoint format")


def load_model(checkpoint_path: str, feature_cols: List[str], reason_mapping: Dict[str, int], device: str):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = get_checkpoint_state(ckpt)

    if any(k.startswith("backbone.") or k.startswith("score_head.") for k in state.keys()):
        style = "original"
        input_dim, inferred_hidden = infer_hidden_dims_from_state_dict(state, style)
        hidden_dims = ckpt.get("hidden_dims", inferred_hidden) if isinstance(ckpt, dict) else inferred_hidden
        dropout = ckpt.get("dropout", 0.15) if isinstance(ckpt, dict) else 0.15
        model = SharedScorerMLP(input_dim, hidden_dims, dropout, max(len(reason_mapping), 1)).to(device)
    elif any(k.startswith("encoder.") or k.startswith("scorer.") for k in state.keys()):
        style = "unified"
        input_dim, inferred_hidden = infer_hidden_dims_from_state_dict(state, style)
        hidden_dims = ckpt.get("hidden_dims", inferred_hidden) if isinstance(ckpt, dict) else inferred_hidden
        dropout = ckpt.get("dropout", 0.15) if isinstance(ckpt, dict) else 0.15
        model = UnifiedPairwiseLTRModel(input_dim, hidden_dims, dropout, max(len(reason_mapping), 1)).to(device)
    else:
        raise ValueError("Cannot detect checkpoint model style. Expected backbone/score_head or encoder/scorer keys.")

    if input_dim != len(feature_cols):
        print(
            f"[WARN] checkpoint input_dim={input_dim}, feature_columns={len(feature_cols)}. "
            "The model input dimension follows checkpoint; feature matrix will be aligned/truncated/padded."
        )

    model.load_state_dict(state, strict=False)
    model.eval()
    return model, style, input_dim


# ============================================================
# 4. Feature reconstruction
# ============================================================

NON_FEATURE_COLUMNS = {
    "row_id", "column", "dirty_value", "candidate_value",
    "candidate_rank", "candidate_source", "candidate_sources", "candidate_source_text",
    "extra_info", "neighbor_majority_value", "suggested_correct_value",
    "main_rule_type", "main_usage_role", "semantic_type", "detected_type",
    "candidate_canonical_value", "dirty_canonical_value",
    "all_candidates", "top_candidate_value", "top_candidate_sources",
    "candidate_obj_json", "raw_response", "reason", "reason_label", "winner",
    "candidate_a_value", "candidate_b_value",
    "soccer_evidence_flags", "allowed_soccer_evidence_flags",
    "row_team", "row_city", "row_stadium", "row_manager", "row_position_canonical",
    "allowed_row_team", "allowed_row_city", "allowed_row_stadium",
    "allowed_row_manager", "allowed_row_position_canonical",
}


def get_candidate_value_col(df: pd.DataFrame) -> str:
    for c in ["candidate_value", "repair_candidate", "candidate", "value_candidate", "predicted_repair"]:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot find candidate value column. Actual columns={list(df.columns)}")


def normalize_candidate_features(df: pd.DataFrame) -> pd.DataFrame:
    required = {"row_id", "column"}
    if not required.issubset(df.columns):
        raise ValueError(f"candidate features must contain {required}")
    out = df.copy()
    out["row_id"] = pd.to_numeric(out["row_id"], errors="raise").astype(int)
    out["column"] = out["column"].astype(str).str.strip()

    cand_col = get_candidate_value_col(out)
    if cand_col != "candidate_value":
        out = out.rename(columns={cand_col: "candidate_value"})

    if "dirty_value" not in out.columns:
        if "value" in out.columns:
            out["dirty_value"] = out["value"]
        elif "allowed_value" in out.columns:
            out["dirty_value"] = out["allowed_value"]
        else:
            out["dirty_value"] = ""

    out["candidate_value"] = out["candidate_value"].map(canonical_empty_text)
    out["dirty_value"] = out["dirty_value"].map(canonical_empty_text)
    return out


def add_derived_features(df: pd.DataFrame, cfg: DatasetConfig, disable_dataset_derived: bool = False) -> pd.DataFrame:
    out = df.copy()
    out["column_norm"] = out["column"].map(normalize_col_name)
    out["original_is_empty"] = out["dirty_value"].map(lambda x: int(norm_lower(x) in EMPTY_TOKENS))
    out["candidate_is_empty"] = out["candidate_value"].map(lambda x: int(norm_lower(x) in EMPTY_TOKENS))
    out["candidate_equals_dirty"] = (
        out["candidate_value"].map(norm_lower) == out["dirty_value"].map(norm_lower)
    ).astype(int)

    out["edit_similarity_to_dirty_auto"] = [
        edit_ratio(a, b) for a, b in zip(out["dirty_value"], out["candidate_value"])
    ]
    out["dirty_len"] = out["dirty_value"].map(lambda x: len(norm_text(x)))
    out["candidate_len"] = out["candidate_value"].map(lambda x: len(norm_text(x)))
    out["len_delta_abs"] = (out["candidate_len"] - out["dirty_len"]).abs()
    out["dirty_looks_like_code"] = out["dirty_value"].map(looks_like_code)
    out["candidate_looks_like_code"] = out["candidate_value"].map(looks_like_code)
    out["same_prefix_token"] = [
        int(prefix_token(d).lower() == prefix_token(c).lower() and prefix_token(d) != "")
        for d, c in zip(out["dirty_value"], out["candidate_value"])
    ]
    out["same_family_token"] = [
        int(family_token(d).lower() == family_token(c).lower() and family_token(d) != "")
        for d, c in zip(out["dirty_value"], out["candidate_value"])
    ]

    srcs = out.apply(source_text, axis=1)
    out["contains_rule_source_auto"] = srcs.map(lambda s: int("rule" in s or "expected" in s))
    out["contains_neighbor_source_auto"] = srcs.map(lambda s: int("neighbor" in s))
    out["contains_similarity_source_auto"] = srcs.map(lambda s: int("similar" in s))
    out["contains_profile_source_auto"] = srcs.map(lambda s: int("profile" in s or "consensus" in s or "majority" in s))
    out["contains_frequency_source_auto"] = srcs.map(lambda s: int("frequency" in s or "column_top" in s or "global" in s))
    out["contains_domain_source_auto"] = srcs.map(lambda s: int("domain" in s))
    out["contains_format_source_auto"] = srcs.map(lambda s: int("format" in s or "canonical" in s or "normalize" in s))

    if disable_dataset_derived:
        return out

    if cfg.name == "flights":
        time_cols = {"sched_dep_time", "act_dep_time", "sched_arr_time", "act_arr_time"}
        out["is_time_column"] = out["column_norm"].map(lambda c: int(c in time_cols or c.endswith("_time")))
        out["dirty_time_valid"] = out["dirty_value"].map(is_valid_time_simple)
        out["candidate_time_valid"] = out["candidate_value"].map(is_valid_time_simple)
        out["time_format_gain_auto"] = out["candidate_time_valid"] - out["dirty_time_valid"]

    elif cfg.name == "beers":
        out["candidate_state_code_valid_auto"] = [
            int(c == "state" and bool(normalize_state_code(v)))
            for c, v in zip(out["column_norm"], out["candidate_value"])
        ]
        out["candidate_numeric_beer_valid_auto"] = [
            int(c in {"abv", "ibu", "ounces"} and bool(re.search(r"\d", norm_text(v))))
            for c, v in zip(out["column_norm"], out["candidate_value"])
        ]

    elif cfg.name == "rayyan":
        out["candidate_language_valid_auto"] = [
            int(c == "article_language" and normalize_language_candidate(v) is not None)
            for c, v in zip(out["column_norm"], out["candidate_value"])
        ]
        out["candidate_issn_valid_auto"] = [
            int(c == "journal_issn" and normalize_issn_candidate(v) is not None)
            for c, v in zip(out["column_norm"], out["candidate_value"])
        ]
        out["candidate_numeric_metadata_valid_auto"] = [
            int(c in {"article_jvolumn", "article_jissue"} and bool(re.search(r"\d", norm_text(v))))
            for c, v in zip(out["column_norm"], out["candidate_value"])
        ]

    elif cfg.name == "adult":
        out["candidate_in_adult_domain_auto"] = [
            int(c in ADULT_DOMAIN_VALUES and canonical_adult_value(c, v) in ADULT_DOMAIN_VALUES[c])
            for c, v in zip(out["column_norm"], out["candidate_value"])
        ]
        out["dirty_in_adult_domain_auto"] = [
            int(c in ADULT_DOMAIN_VALUES and canonical_adult_value(c, v) in ADULT_DOMAIN_VALUES[c])
            for c, v in zip(out["column_norm"], out["dirty_value"])
        ]
        out["adult_domain_gain_auto"] = out["candidate_in_adult_domain_auto"] - out["dirty_in_adult_domain_auto"]
        out["candidate_numeric_adult_valid_auto"] = [
            int(c in {"age", "hoursperweek"} and bool(re.search(r"\d", norm_text(v))))
            for c, v in zip(out["column_norm"], out["candidate_value"])
        ]

    elif cfg.name == "soccer":
        out["candidate_position_valid_auto"] = [
            int(c == "position" and position_domain_valid(v))
            for c, v in zip(out["column_norm"], out["candidate_value"])
        ]
        out["dirty_position_valid_auto"] = [
            int(c == "position" and position_domain_valid(v))
            for c, v in zip(out["column_norm"], out["dirty_value"])
        ]
        out["soccer_position_gain_auto"] = out["candidate_position_valid_auto"] - out["dirty_position_valid_auto"]
        out["candidate_year_valid_auto"] = [
            int(c in {"birthyear", "season"} and parse_year_like(v) is not None)
            for c, v in zip(out["column_norm"], out["candidate_value"])
        ]
        out["profile_majority_strength_auto"] = 0.0
        for c in [
            "profile_key_majority_ratio", "entity_pair_majority_ratio", "team_season_position_ratio",
            "candidate_has_high_conf_profile_majority", "candidate_entity_safe_fill",
            "candidate_position_profile_strong",
        ]:
            if c in out.columns:
                out["profile_majority_strength_auto"] += pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    return out


def align_feature_matrix(df: pd.DataFrame, feature_cols: List[str], input_dim: int) -> np.ndarray:
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
    x = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float32)

    if x.shape[1] < input_dim:
        pad = np.zeros((x.shape[0], input_dim - x.shape[1]), dtype=np.float32)
        x = np.concatenate([x, pad], axis=1)
    elif x.shape[1] > input_dim:
        x = x[:, :input_dim]
    return x


def score_candidates(df: pd.DataFrame, model, feature_cols: List[str], input_dim: int, device: str, batch_size: int) -> pd.DataFrame:
    out = df.copy()
    x = align_feature_matrix(out, feature_cols, input_dim)
    scores = []
    reason_ids = []

    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start:start + batch_size]).to(device)
            s, reason_logits, _ = model.forward_single(xb)
            scores.extend(s.detach().cpu().numpy().tolist())
            if reason_logits is not None:
                reason_ids.extend(torch.argmax(reason_logits, dim=-1).detach().cpu().numpy().tolist())
            else:
                reason_ids.extend([0] * len(xb))

    out["student_score"] = scores
    out["student_reason_id"] = reason_ids
    return out


# ============================================================
# 5. Gate logic
# ============================================================

def margin_threshold_for_column(column: Any, cfg: DatasetConfig, global_min_margin: float) -> float:
    c = str(column)
    cn = normalize_col_name(c)
    return max(global_min_margin, cfg.column_margin_thresholds.get(c, cfg.column_margin_thresholds.get(cn, cfg.default_margin_threshold)))


def model_margin_gate(top: pd.Series, second: Optional[pd.Series], cfg: DatasetConfig, global_min_margin: float) -> Tuple[bool, str, float]:
    score1 = safe_float(top.get("student_score"), 0.0)
    score2 = safe_float(second.get("student_score"), score1) if second is not None else score1 - 999.0
    margin = score1 - score2 if second is not None else 999.0
    thr = margin_threshold_for_column(top.get("column"), cfg, global_min_margin)

    if margin < thr:
        return False, f"low_margin<{thr:.4f}", margin
    return True, "margin_pass", margin


def candidate_value_for_output(row: pd.Series, cfg: DatasetConfig) -> str:
    col = normalize_col_name(row.get("column"))
    val = row.get("candidate_value", "")

    if cfg.name == "beers":
        fixed = normalize_for_beers_column(col, val)
        return fixed if fixed else canonical_empty_text(val)
    if cfg.name == "rayyan":
        fixed = normalize_by_rayyan_column(col, val)
        return fixed if fixed else canonical_empty_text(val)
    if cfg.name == "adult":
        return canonical_adult_value(col, val)
    if cfg.name == "soccer":
        if col == "position":
            return canonical_position(val)
        if col in {"birthyear", "season"}:
            return canonical_year(val)
        return canonical_empty_text(val)
    return canonical_empty_text(val)


def get_row_context_from_group(g: pd.DataFrame) -> Dict[str, Any]:
    # Use allowed_/row_ context fields if present.
    ctx = {}
    for c in g.columns:
        if c.startswith("row_") or c.startswith("allowed_row_"):
            base = c.replace("allowed_row_", "").replace("row_", "")
            val = g[c].dropna().iloc[0] if g[c].notna().any() else ""
            if not is_empty_like(val):
                ctx[normalize_col_name(base)] = val
    # Also use dirty values for this candidate group.
    if "dirty_value" in g.columns:
        ctx["dirty_value"] = g["dirty_value"].iloc[0]
    return ctx


def dataset_specific_gate(top: pd.Series, g: pd.DataFrame, cfg: DatasetConfig, args) -> Tuple[bool, str, str]:
    col = normalize_col_name(top.get("column"))
    dirty = canonical_empty_text(top.get("dirty_value", ""))
    cand_raw = canonical_empty_text(top.get("candidate_value", ""))
    pred = candidate_value_for_output(top, cfg)

    # Universal no-op.
    if norm_lower(pred) == norm_lower(dirty):
        return False, "keep_original_or_same_value", dirty

    if pred == "empty" and col in cfg.protected_nonempty_columns:
        return False, "reject_empty_for_protected_column", dirty

    if cfg.name == "hospital":
        if col == "MeasureCode":
            if not likely_measure_code_candidate(pred):
                return False, "reject_invalid_measurecode_candidate", dirty
        if col == "Stateavg":
            if not likely_stateavg_candidate(pred):
                return False, "reject_invalid_stateavg_candidate", dirty
        if col == "Condition":
            fam = measure_family_name(g["candidate_value"].iloc[0] if not g.empty else pred)
            if pred.lower() not in {
                "heart attack", "heart failure", "pneumonia",
                "surgical infection prevention", "children s asthma care",
            } and fam == "":
                return False, "reject_invalid_condition_candidate", dirty
        if col == "Score":
            if not (is_percent_like(pred) or pred.lower() in {"yes", "no", "empty"}):
                return False, "reject_invalid_score_candidate", dirty
        if col == "Sample":
            if not (patients_like(pred) or pred.lower() in {"yes", "no", "empty"}):
                return False, "reject_invalid_sample_candidate", dirty
        return True, "hospital_gate_pass", pred

    if cfg.name == "flights":
        if col in {"sched_dep_time", "act_dep_time", "sched_arr_time", "act_arr_time"}:
            if not is_valid_time_simple(pred):
                return False, "reject_invalid_time", dirty
            if safe_float(top.get("candidate_is_trusted_consensus", 0), 0) > 0:
                return True, "trusted_time_consensus_pass", pred
            if safe_float(top.get("is_from_time_consensus", 0), 0) > 0:
                return True, "time_consensus_pass", pred
            if safe_float(top.get("time_format_gain_auto", top.get("candidate_time_valid", 0)), 0) > 0:
                return True, "time_format_or_validity_pass", pred
            return True, "time_model_margin_pass", pred
        return True, "flights_gate_pass", pred

    if cfg.name == "beers":
        # High precision direct rule for ABV percent formatting.
        if col == "abv":
            direct = abv_percent_direct_repair(dirty)
            if direct and norm_lower(direct) == norm_lower(pred):
                return True, "abv_percent_strip_direct", direct
        if not is_valid_beers_candidate(col, pred):
            return False, "reject_invalid_beers_candidate", dirty
        if col in {"state", "city"}:
            if safe_float(top.get("candidate_equals_state_city_consensus", 0), 0) > 0 or safe_float(top.get("state_city_ratio_gain", 0), 0) > 0:
                return True, "state_city_consensus_pass", pred
        if col in {"abv", "ibu", "ounces"}:
            if safe_float(top.get("beers_numeric_consistency_gain", top.get("candidate_numeric_beer_valid_auto", 0)), 0) > 0:
                return True, "beers_numeric_gate_pass", pred
        if col in {"brewery_name", "beer_name", "style"}:
            # Avoid aggressive text repairs unless profile/context evidence exists.
            profile = max(
                safe_float(top.get("is_from_beers_profile", 0), 0),
                safe_float(top.get("contains_profile_source_auto", 0), 0),
                safe_float(top.get("contains_neighbor_source_auto", 0), 0),
            )
            if profile <= 0 and not args.disable_dataset_gate:
                return False, "reject_weak_text_beers_candidate", dirty
        return True, "beers_gate_pass", pred

    if cfg.name == "rayyan":
        if col in {"article_jcreated_at", "article_jvolumn", "article_jissue"}:
            fixed = normalize_by_rayyan_column(col, cand_raw)
            if fixed:
                return True, "rayyan_direct_canonical_rule", fixed
        if col in cfg.risky_context_columns and not args.disable_dataset_gate:
            return False, "rayyan_blocked_risky_metadata_column", dirty
        fixed = normalize_by_rayyan_column(col, pred)
        if col in {"article_language", "journal_issn", "article_pagination"} and not fixed:
            return False, "reject_invalid_rayyan_canonical", dirty
        return True, "rayyan_gate_pass", fixed if fixed else pred

    if cfg.name == "adult":
        if not adult_value_valid(col, pred):
            return False, "reject_invalid_adult_domain_or_numeric", dirty
        if col in cfg.risky_context_columns and not args.disable_dataset_gate:
            support = max(
                safe_float(top.get("is_from_adult_profile", 0), 0),
                safe_float(top.get("contains_profile_source_auto", 0), 0),
                safe_float(top.get("contains_neighbor_source_auto", 0), 0),
            )
            if support <= 0:
                return False, "reject_risky_context_without_profile_support", dirty
        if col in {"relationship", "sex", "maritalstatus"}:
            support = max(
                safe_float(top.get("relationship_consistency_gain", 0), 0),
                safe_float(top.get("is_from_adult_profile", 0), 0),
                safe_float(top.get("contains_profile_source_auto", 0), 0),
            )
            if support <= 0 and not args.disable_dataset_gate:
                return False, "reject_weak_adult_consistency_support", dirty
        return True, "adult_gate_pass", pred

    if cfg.name == "soccer":
        if col == "position":
            if not position_domain_valid(pred):
                return False, "reject_invalid_position", dirty
            if (
                safe_float(top.get("candidate_position_profile_strong", 0), 0) > 0
                or safe_float(top.get("is_from_team_season_position", 0), 0) > 0
                or safe_float(top.get("position_consistency_gain", top.get("soccer_position_gain_auto", 0)), 0) > 0
            ):
                return True, "position_profile_or_domain_pass", pred
            return True, "position_model_margin_pass", pred

        if col == "birthyear":
            if not is_valid_birthyear_value(pred):
                return False, "reject_invalid_birthyear", dirty
            return True, "birthyear_gate_pass", pred

        if col == "season":
            if not is_valid_season_value(pred):
                return False, "reject_invalid_season", dirty
            return True, "season_gate_pass", pred

        if col in {"name", "surname", "birthplace"}:
            profile = max(
                safe_float(top.get("candidate_entity_safe_fill", 0), 0),
                safe_float(top.get("is_from_entity_pair_majority", 0), 0),
                safe_float(top.get("is_from_profile_key_majority", 0), 0),
                safe_float(top.get("candidate_has_high_conf_profile_majority", 0), 0),
                safe_float(top.get("profile_majority_strength_auto", 0), 0),
            )
            weak_global = safe_float(top.get("candidate_is_pure_weak_global_for_entity", 0), 0) > 0
            if weak_global and not args.disable_dataset_gate:
                return False, "reject_pure_weak_global_entity_candidate", dirty
            if profile <= 0 and not args.disable_dataset_gate:
                return False, "reject_entity_without_profile_majority", dirty
            return True, "entity_profile_majority_pass", pred

        if col in {"team", "city", "stadium", "manager"}:
            support = max(
                safe_float(top.get("team_context_gain", 0), 0),
                safe_float(top.get("contains_profile_source_auto", 0), 0),
                safe_float(top.get("contains_neighbor_source_auto", 0), 0),
            )
            if support <= 0 and not args.allow_profile_only_context_repairs and not args.disable_dataset_gate:
                return False, "reject_weak_soccer_context_candidate", dirty
            return True, "soccer_context_gate_pass", pred

        return True, "soccer_gate_pass", pred

    return True, "generic_gate_pass", pred


def build_top1_decisions(ranked: pd.DataFrame, cfg: DatasetConfig, args) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    rejected_rows = []

    for (row_id, column), g in ranked.groupby(["row_id", "column"], sort=False):
        g = g.sort_values(["student_score", "candidate_rank_for_output"], ascending=[False, True]).copy()
        top = g.iloc[0]
        second = g.iloc[1] if len(g) >= 2 else None

        if args.disable_gate:
            margin = safe_float(top.get("student_score"), 0.0) - (safe_float(second.get("student_score"), 0.0) if second is not None else -999.0)
            pred = candidate_value_for_output(top, cfg)
            gate_passed = int(norm_lower(pred) != norm_lower(top.get("dirty_value", "")))
            gate_reason = "disable_gate_accept_top1" if gate_passed else "disable_gate_same_as_dirty"
        else:
            margin_ok, margin_reason, margin = model_margin_gate(top, second, cfg, args.global_min_margin)
            if not margin_ok:
                gate_passed = 0
                gate_reason = margin_reason
                pred = canonical_empty_text(top.get("dirty_value", ""))
            else:
                ok, ds_reason, pred = dataset_specific_gate(top, g, cfg, args)
                gate_passed = int(ok)
                gate_reason = ds_reason

        dirty_value = canonical_empty_text(top.get("dirty_value", ""))
        row = top.to_dict()
        row.update({
            "student_top_candidate": top.get("candidate_value", ""),
            "student_score": safe_float(top.get("student_score"), 0.0),
            "student_margin": float(margin),
            "student_second_candidate": second.get("candidate_value", "") if second is not None else "",
            "student_second_score": safe_float(second.get("student_score"), 0.0) if second is not None else "",
            "candidate_count": len(g),
            "gate_passed": gate_passed,
            "gate_pass": gate_passed,
            "gate_reason": gate_reason,
            "predicted_repair_value": pred if gate_passed else dirty_value,
            "predicted_repair": pred if gate_passed else dirty_value,
            "repair_value": pred if gate_passed else dirty_value,
            "dirty_value": dirty_value,
        })
        rows.append(row)

        if not gate_passed:
            rej = row.copy()
            rejected_rows.append(rej)

    top1 = pd.DataFrame(rows)
    rejected = pd.DataFrame(rejected_rows)
    return top1, rejected


# ============================================================
# 6. Whitelist validation and hard cases
# ============================================================

def load_allowed_cells(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        print(f"[WARN] allowed_cells_csv not found: {path}")
        return pd.DataFrame(columns=["row_id", "column"])
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if not {"row_id", "column"}.issubset(df.columns):
        print(f"[WARN] allowed_cells_csv lacks row_id,column: {path}")
        return pd.DataFrame(columns=["row_id", "column"])
    df = df.copy()
    df["row_id"] = pd.to_numeric(df["row_id"], errors="coerce").fillna(-1).astype(int)
    df["column"] = df["column"].astype(str).str.strip()
    return df[["row_id", "column"]].drop_duplicates()


def validate_whitelist(cand_df: pd.DataFrame, allowed_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cand_cells = cand_df[["row_id", "column"]].drop_duplicates().copy()
    if allowed_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    input_violation = cand_cells.merge(
        allowed_df.assign(__allowed__=1),
        on=["row_id", "column"],
        how="left",
    )
    input_violation = input_violation[input_violation["__allowed__"].isna()].drop(columns=["__allowed__"])

    missing_allowed = allowed_df.merge(
        cand_cells.assign(__cand__=1),
        on=["row_id", "column"],
        how="left",
    )
    missing_allowed = missing_allowed[missing_allowed["__cand__"].isna()].drop(columns=["__cand__"])
    return input_violation, missing_allowed


def export_hard_cases(top1: pd.DataFrame, ranked: pd.DataFrame, output_jsonl: str, max_hard_cases: int):
    # Hard cases = rejected, low margin, many candidates, or top score near second.
    if top1.empty:
        open(output_jsonl, "w", encoding="utf-8").close()
        return

    tmp = top1.copy()
    tmp["hard_case_score"] = 0.0
    tmp["hard_case_score"] += (1 - pd.to_numeric(tmp.get("gate_passed", 0), errors="coerce").fillna(0)) * 2.0
    tmp["hard_case_score"] += (pd.to_numeric(tmp.get("student_margin", 0), errors="coerce").fillna(0) < 0.10).astype(int) * 1.0
    tmp["hard_case_score"] += (pd.to_numeric(tmp.get("candidate_count", 0), errors="coerce").fillna(0) >= 5).astype(int) * 0.5
    tmp = tmp.sort_values(["hard_case_score", "student_margin"], ascending=[False, True]).head(max_hard_cases)

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for _, r in tmp.iterrows():
            cell_ranked = ranked[(ranked["row_id"] == r["row_id"]) & (ranked["column"] == r["column"])]
            cands = []
            for _, cr in cell_ranked.head(10).iterrows():
                cands.append({
                    "candidate_value": cr.get("candidate_value", ""),
                    "student_score": safe_float(cr.get("student_score"), 0.0),
                    "rank": safe_int(cr.get("student_rank"), 0),
                    "candidate_sources": cr.get("candidate_sources", cr.get("candidate_source", "")),
                })
            obj = {
                "row_id": int(r["row_id"]),
                "column": str(r["column"]),
                "dirty_value": r.get("dirty_value", ""),
                "student_top_candidate": r.get("student_top_candidate", ""),
                "predicted_repair_value": r.get("predicted_repair_value", ""),
                "student_margin": safe_float(r.get("student_margin"), 0.0),
                "gate_passed": safe_int(r.get("gate_passed"), 0),
                "gate_reason": r.get("gate_reason", ""),
                "candidates": cands,
            }
            write_jsonl_record(f, obj)


# ============================================================
# 7. Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Unified LTR inference + conservative gate + hard-case mining")
    p.add_argument("--dataset", required=True, choices=sorted(DATASET_CONFIGS.keys()))
    p.add_argument("--candidate_features", default="repair_candidate_features_infer_whitelist.csv")
    p.add_argument("--allowed_cells_csv", default=None)
    p.add_argument("--model_dir", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--output_dir", default=None)

    p.add_argument("--global_min_margin", type=float, default=None)
    p.add_argument("--joint_topk", type=int, default=4)
    p.add_argument("--max_hard_cases", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=4096)

    # Soccer-compatible options.
    p.add_argument("--allow_profile_only_context_repairs", type=int, default=0)
    p.add_argument("--debug_export_rejected", type=int, default=1)

    # Ablation switches.
    p.add_argument("--disable_gate", type=int, default=0)
    p.add_argument("--disable_dataset_gate", type=int, default=0)
    p.add_argument("--disable_dataset_derived_features", type=int, default=0)
    p.add_argument("--score_only", type=int, default=0)

    return p.parse_args()


def build_output_paths(cfg: DatasetConfig, model_dir: str, output_dir_arg: Optional[str]) -> Dict[str, str]:
    if output_dir_arg:
        outdir = os.path.abspath(output_dir_arg)
    elif os.getenv("REPAIR_INFER_OUTPUT_DIR"):
        outdir = os.path.abspath(os.getenv("REPAIR_INFER_OUTPUT_DIR", os.getcwd()))
    elif cfg.output_to_model_dir_by_default:
        outdir = os.path.abspath(model_dir)
    else:
        outdir = os.path.abspath(os.getcwd())

    os.makedirs(outdir, exist_ok=True)
    return {
        "output_dir": outdir,
        "all_scores": os.path.join(outdir, "inference_all_candidate_scores.csv"),
        "ranked": os.path.join(outdir, "inference_all_candidate_ranked.csv"),
        "top1": os.path.join(outdir, "inference_top1_repairs.csv"),
        "hard_cases": os.path.join(outdir, "hard_cases_for_llm.jsonl"),
        "input_whitelist_violations": os.path.join(outdir, "input_whitelist_violations.csv"),
        "output_whitelist_violations": os.path.join(outdir, "output_whitelist_violations.csv"),
        "rejected_debug": os.path.join(outdir, "inference_rejected_candidates_debug.csv"),
    }


def main():
    args = parse_args()
    cfg = DATASET_CONFIGS[args.dataset]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_dir = args.model_dir or cfg.default_model_dir
    checkpoint_path = args.checkpoint or os.path.join(model_dir, "best_model.pt")
    feature_columns_json = os.path.join(model_dir, "feature_columns.json")
    reason_labels_json = os.path.join(model_dir, "reason_label_mapping.json")
    column_profiles_json = os.path.join(model_dir, "column_profiles.json")
    allowed_cells_csv = args.allowed_cells_csv or cfg.default_allowed_cells_csv
    global_min_margin = args.global_min_margin if args.global_min_margin is not None else cfg.default_margin_threshold

    outputs = build_output_paths(cfg, model_dir, args.output_dir)

    print("========== Unified LTR Inference ==========")
    print(f"dataset: {cfg.name}")
    print(f"candidate_features: {args.candidate_features}")
    print(f"allowed_cells_csv: {allowed_cells_csv}")
    print(f"model_dir: {model_dir}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"feature_columns_json: {feature_columns_json}")
    print(f"output_dir: {outputs['output_dir']}")
    print(f"device: {device}")

    feature_cols = load_json(feature_columns_json, default=[])
    if not isinstance(feature_cols, list) or not feature_cols:
        raise ValueError(f"feature_columns.json invalid or empty: {feature_columns_json}")

    reason_mapping = load_json(reason_labels_json, default={})
    if not isinstance(reason_mapping, dict):
        reason_mapping = {}

    # column_profiles are loaded for compatibility/debugging. Some old scripts used them for extra gates.
    _column_profiles = load_json(column_profiles_json, default={})

    print("\n[1/6] Loading candidate features...")
    cand = pd.read_csv(args.candidate_features, encoding="utf-8-sig", low_memory=False)
    cand = normalize_candidate_features(cand)
    cand = add_derived_features(
        cand,
        cfg,
        disable_dataset_derived=bool(args.disable_dataset_derived_features),
    )
    print(f"[INFO] candidate rows: {len(cand)}")
    print(f"[INFO] unique cells: {cand[['row_id','column']].drop_duplicates().shape[0]}")

    print("\n[2/6] Loading model...")
    model, model_style, input_dim = load_model(checkpoint_path, feature_cols, reason_mapping, device)
    print(f"[INFO] model_style: {model_style}")
    print(f"[INFO] model_input_dim: {input_dim}")
    print(f"[INFO] feature_columns: {len(feature_cols)}")

    print("\n[3/6] Scoring candidates...")
    scored = score_candidates(cand, model, feature_cols, input_dim, device, args.batch_size)

    scored["student_score"] = pd.to_numeric(scored["student_score"], errors="coerce").fillna(0.0)
    scored["candidate_rank_for_output"] = pd.to_numeric(scored.get("candidate_rank", 999999), errors="coerce").fillna(999999).astype(int)

    scored = scored.sort_values(
        ["row_id", "column", "student_score", "candidate_rank_for_output"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)

    scored.to_csv(outputs["all_scores"], index=False, encoding="utf-8-sig")
    print(f"[OK] all candidate scores: {outputs['all_scores']}")

    print("\n[4/6] Ranking candidates per cell...")
    ranked = scored.copy()
    ranked["student_rank"] = ranked.groupby(["row_id", "column"]).cumcount() + 1
    ranked.to_csv(outputs["ranked"], index=False, encoding="utf-8-sig")
    print(f"[OK] ranked candidates: {outputs['ranked']}")

    print("\n[5/6] Whitelist validation...")
    allowed = load_allowed_cells(allowed_cells_csv)
    input_viol, missing_allowed = validate_whitelist(cand, allowed)
    input_viol.to_csv(outputs["input_whitelist_violations"], index=False, encoding="utf-8-sig")
    missing_allowed.to_csv(outputs["output_whitelist_violations"], index=False, encoding="utf-8-sig")
    print(f"[INFO] input whitelist violations: {len(input_viol)}")
    print(f"[INFO] allowed cells missing candidates: {len(missing_allowed)}")

    print("\n[6/6] Building top1 repairs and hard cases...")
    top1, rejected = build_top1_decisions(ranked, cfg, args)
    top1.to_csv(outputs["top1"], index=False, encoding="utf-8-sig")
    export_hard_cases(top1, ranked, outputs["hard_cases"], args.max_hard_cases)

    if cfg.output_rejected_debug and args.debug_export_rejected:
        rejected.to_csv(outputs["rejected_debug"], index=False, encoding="utf-8-sig")
        print(f"[OK] rejected debug: {outputs['rejected_debug']}")

    print(f"[OK] top1 repairs: {outputs['top1']}")
    print(f"[OK] hard cases: {outputs['hard_cases']}")

    print("\n========== Summary ==========")
    print(f"candidate rows: {len(cand)}")
    print(f"unique cells: {cand[['row_id','column']].drop_duplicates().shape[0]}")
    print(f"top1 rows: {len(top1)}")
    if not top1.empty:
        print(f"gate_passed: {int(pd.to_numeric(top1['gate_passed'], errors='coerce').fillna(0).sum())}")
        print("\nGate reason distribution:")
        print(top1["gate_reason"].value_counts(dropna=False).head(30).to_string())
        print("\nTop1 column distribution:")
        print(top1["column"].value_counts(dropna=False).head(30).to_string())


if __name__ == "__main__":
    main()

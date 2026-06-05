#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified full-table end-to-end repair evaluation.

Datasets:
  hospital, flights, beers, rayyan, adult, soccer

Purpose
-------
This script merges the full-table evaluation logic of six dataset-specific
evaluate.py scripts into one readable implementation.

It preserves the standard output files used by the original evaluators:

  full_table_eval_summary.csv
  full_table_eval_all_cells.csv
  full_table_tp_details.csv
  full_table_fp_details.csv
  full_table_fn_details.csv
  full_table_tn_details.csv
  repaired_table.csv
  touched_cells_details.csv
  column_level_eval_summary.csv

Dataset-specific optional outputs:
  gate_reason_eval_summary.csv
  candidate_source_eval_summary.csv
  canonical_eval_summary.csv
  rayyan_canonical_eval_summary.csv
  rayyan_source_eval_summary.csv
  soccer_domain_eval_summary.csv
  adult_domain_eval_summary.csv
  beers_normalized_eval_summary.csv

Common logic:
  clean table + dirty table + inference_top1_repairs.csv
      -> apply repairs to dirty table
      -> compare clean / dirty / repaired over the full table
      -> TP / FP / FN / TN details
      -> summary metrics

Dataset-specific comparison:
  hospital: strict string comparison
  flights : strict string comparison by default
  beers   : normalized state / city / ounces / ABV / IBU / id
  rayyan  : normalized language / ISSN / date / volume / issue / pagination
  adult   : normalized income / categorical domains / numeric ranges
  soccer  : normalized position / birthyear / season / entity text

Usage examples
--------------
Soccer:
  python unified_evaluate_full.py --dataset soccer

Adult:
  python unified_evaluate_full.py --dataset adult

Explicit paths:
  python unified_evaluate_full.py \
    --dataset soccer \
    --clean_file /mnt/mydata/dq/projects/Splittree/soccer/soccer_clean.csv \
    --dirty_file /mnt/mydata/dq/projects/Splittree/soccer/soccer_dirty.csv \
    --repair_result_file /mnt/mydata/dq/projects/Splittree/test/Error\\ Correction\\ soccer/repair_iterative_workdir_soccer/model_output/inference_top1_repairs.csv \
    --output_dir repair_eval_output_soccer

Ablation output:
  python unified_evaluate_full.py \
    --dataset soccer \
    --repair_result_file ablation_no_gate_soccer/model_output/inference_top1_repairs.csv \
    --output_dir ablation_no_gate_soccer/eval
"""

import argparse
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# ============================================================
# 1. Dataset config
# ============================================================

EMPTY_TOKENS = {
    "", "empty", "null", "none", "nan", "na", "n/a", "missing",
    "?", "{null}", "not_available", "not available", "unknown", "unk", "-", "--",
}

US_STATE_CODES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC",
}

STATE_NAME_TO_CODE = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN",
    "mississippi": "MS", "missouri": "MO", "montana": "MT", "nebraska": "NE",
    "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ",
    "new mexico": "NM", "new york": "NY", "north carolina": "NC",
    "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR",
    "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
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

ADULT_DOMAIN_VALUES = {
    "workclass": [
        "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
        "Local-gov", "State-gov", "Without-pay", "Never-worked",
    ],
    "education": [
        "Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th",
        "11th", "12th", "HS-grad", "Some-college", "Assoc-voc",
        "Assoc-acdm", "Bachelors", "Masters", "Prof-school", "Doctorate",
    ],
    "maritalstatus": [
        "Married-civ-spouse", "Divorced", "Never-married", "Separated",
        "Widowed", "Married-spouse-absent", "Married-AF-spouse",
    ],
    "occupation": [
        "Tech-support", "Craft-repair", "Other-service", "Sales",
        "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
        "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
        "Transport-moving", "Priv-house-serv", "Protective-serv",
        "Armed-Forces",
    ],
    "relationship": [
        "Wife", "Own-child", "Husband", "Not-in-family",
        "Other-relative", "Unmarried",
    ],
    "race": ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"],
    "sex": ["Female", "Male"],
    "income": ["LessThan50K", "MoreThan50K"],
}

NUMERIC_OR_RANGE_COLUMNS = {
    "age", "hoursperweek", "hours-per-week", "hours_per_week",
    "education_num", "education-num", "fnlwgt",
    "capital_gain", "capital-gain", "capital_loss", "capital-loss",
}

SOCCER_POSITION_VALUES = ["Goalkeeper", "Defender", "Midfield", "Forward"]
POSITION_ALIASES = {
    "goal keeper": "Goalkeeper", "goal-keeper": "Goalkeeper", "goal_keeper": "Goalkeeper",
    "goalkeeper": "Goalkeeper", "keeper": "Goalkeeper", "gk": "Goalkeeper",
    "defender": "Defender", "defence": "Defender", "defense": "Defender", "df": "Defender", "back": "Defender",
    "midfield": "Midfield", "midfielder": "Midfield", "middle": "Midfield", "mf": "Midfield",
    "forward": "Forward", "striker": "Forward", "attacker": "Forward", "winger": "Forward", "fw": "Forward", "cf": "Forward",
}
SOCCER_TEXT_COLUMNS = {"name", "surname", "birthplace", "team", "city", "stadium", "manager"}

RAYYAN_CANONICAL_COLUMNS = {
    "article_language", "journal_issn", "article_jcreated_at",
    "article_jvolumn", "article_jissue", "article_pagination",
}
RAYYAN_TEXT_METADATA_COLUMNS = {
    "journal_title", "jounral_abbreviation", "journal_abbreviation",
    "article_title", "author_list",
}


@dataclass
class DatasetConfig:
    name: str
    clean_file: str
    dirty_file: str
    repair_result_file: str
    output_dir: str
    normalized_comparison: bool = False
    case_insensitive_text: bool = False
    strict_mode: bool = False
    aliases: Dict[str, str] = field(default_factory=dict)


COMMON_ALIASES = {
    "marital-status": "maritalstatus",
    "marital_status": "maritalstatus",
    "marital status": "maritalstatus",
    "hours-per-week": "hoursperweek",
    "hours_per_week": "hoursperweek",
    "hours per week": "hoursperweek",
    "native-country": "country",
    "native_country": "country",
    "capital-gain": "capital_gain",
    "capital_loss": "capital_loss",
    "capital-loss": "capital_loss",
    "education-num": "education_num",
    "birth_year": "birthyear",
    "birth year": "birthyear",
    "birth-year": "birthyear",
    "year_of_birth": "birthyear",
    "birth_place": "birthplace",
    "birth place": "birthplace",
    "pos": "position",
    "club": "team",
    "stadium_name": "stadium",
    "season_year": "season",
    "coach": "manager",
    "jounral_abbreviation": "journal_abbreviation",
}


DATASET_CONFIGS = {
    "hospital": DatasetConfig(
        name="hospital",
        clean_file="/mnt/mydata/dq/projects/Splittree/hospital/hospital_clean_no_address23.csv",
        dirty_file="/mnt/mydata/dq/projects/Splittree/hospital/hospital_dirty_no_address23.csv",
        repair_result_file="/mnt/mydata/dq/projects/Splittree/test/Error Correction hospital/repair_iterative_workdir_widetrain_whitelistinfer/model_output/inference_top1_repairs.csv",
        output_dir="repair_eval_output",
        normalized_comparison=False,
        case_insensitive_text=False,
    ),
    "flights": DatasetConfig(
        name="flights",
        clean_file="/mnt/mydata/dq/projects/Splittree/flights/flights_clean.csv",
        dirty_file="/mnt/mydata/dq/projects/Splittree/flights/flights_dirty.csv",
        repair_result_file="/mnt/mydata/dq/projects/Splittree/test/Error Correction flights/inference_top1_repairs.csv",
        output_dir="/mnt/mydata/dq/projects/Splittree/test/Error Correction flights/repair_eval_output_flights",
        normalized_comparison=False,
        case_insensitive_text=False,
    ),
    "beers": DatasetConfig(
        name="beers",
        clean_file="/mnt/mydata/dq/projects/Splittree/beers/beers_clean.csv",
        dirty_file="/mnt/mydata/dq/projects/Splittree/beers/beers_dirty.csv",
        repair_result_file="/mnt/mydata/dq/projects/Splittree/test/Error Correction beers/inference_top1_repairs.csv",
        output_dir="/mnt/mydata/dq/projects/Splittree/test/Error Correction beers/repair_eval_output_beers",
        normalized_comparison=True,
        case_insensitive_text=False,
    ),
    "rayyan": DatasetConfig(
        name="rayyan",
        clean_file="/mnt/mydata/dq/projects/Splittree/rayyan/rayyan_clean.csv",
        dirty_file="/mnt/mydata/dq/projects/Splittree/rayyan/rayyan_dirty.csv",
        repair_result_file="/mnt/mydata/dq/projects/Splittree/test/Error Correction rayyan/inference_top1_repairs.csv",
        output_dir="/mnt/mydata/dq/projects/Splittree/test/Error Correction rayyan/repair_eval_output_rayyan",
        normalized_comparison=True,
        case_insensitive_text=False,
    ),
    "adult": DatasetConfig(
        name="adult",
        clean_file="/mnt/mydata/dq/projects/Splittree/adult/adult_clean.csv",
        dirty_file="/mnt/mydata/dq/projects/Splittree/adult/adult_dirty.csv",
        repair_result_file="/mnt/mydata/dq/projects/Splittree/test/Error Correction adult/repair_iterative_workdir_adult/model_output/inference_top1_repairs.csv",
        output_dir="/mnt/mydata/dq/projects/Splittree/test/Error Correction adult/repair_eval_output_adult",
        normalized_comparison=True,
        case_insensitive_text=True,
    ),
    "soccer": DatasetConfig(
        name="soccer",
        clean_file="/mnt/mydata/dq/projects/Splittree/soccer/soccer_clean.csv",
        dirty_file="/mnt/mydata/dq/projects/Splittree/soccer/soccer_dirty.csv",
        repair_result_file="/mnt/mydata/dq/projects/Splittree/test/Error Correction soccer/repair_iterative_workdir_soccer/model_output/inference_top1_repairs.csv",
        output_dir="/mnt/mydata/dq/projects/Splittree/test/Error Correction soccer/repair_eval_output_soccer",
        normalized_comparison=True,
        case_insensitive_text=True,
    ),
}


# ============================================================
# 2. IO and basic utilities
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Unified full-table end-to-end repair evaluation.")
    p.add_argument("--dataset", required=True, choices=sorted(DATASET_CONFIGS.keys()))
    p.add_argument("--clean_file", default="")
    p.add_argument("--dirty_file", default="")
    p.add_argument("--repair_result_file", default="")
    p.add_argument("--output_dir", default="")

    p.add_argument("--normalized_comparison", type=int, default=-1)
    p.add_argument("--case_insensitive_text", type=int, default=-1)
    p.add_argument("--strict_mode", type=int, default=0)
    p.add_argument("--output_extra_summaries", type=int, default=1)
    return p.parse_args()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def norm_raw(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()


def normalize_empty(x: Any) -> str:
    s = norm_raw(x)
    if s.lower() in EMPTY_TOKENS:
        return ""
    return s


def safe_float(x: Any, default=None):
    try:
        if x is None or pd.isna(x):
            return default
        return float(str(x).strip().replace(",", ""))
    except Exception:
        return default


def safe_int(x: Any, default=0):
    try:
        if x is None or pd.isna(x):
            return default
        return int(float(str(x).strip().replace(",", "")))
    except Exception:
        return default


def read_table(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"表文件不存在: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def validate_same_shape(clean_df: pd.DataFrame, dirty_df: pd.DataFrame):
    if clean_df.shape != dirty_df.shape:
        raise ValueError(f"clean 与 dirty 形状不一致: clean={clean_df.shape}, dirty={dirty_df.shape}")
    if list(clean_df.columns) != list(dirty_df.columns):
        raise ValueError(
            "clean 与 dirty 列名不一致，请先对齐列顺序和列名。\n"
            f"clean columns={list(clean_df.columns)}\n"
            f"dirty columns={list(dirty_df.columns)}"
        )


def stringify_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        out[col] = out[col].map(norm_raw).astype(object)
    return out


def normalize_col_name(column: Any) -> str:
    c = norm_raw(column).lower().strip()
    return COMMON_ALIASES.get(c, c)


# ============================================================
# 3. Dataset-specific normalization
# ============================================================

def normalize_state_candidate(x: Any) -> str:
    s = normalize_empty(x)
    if not s:
        return ""
    upper = s.upper()
    if upper in US_STATE_CODES:
        return upper
    lower = s.lower()
    if lower in STATE_NAME_TO_CODE:
        return STATE_NAME_TO_CODE[lower]
    m = re.search(r"\b([A-Za-z]{2})\b$", s.strip())
    if m:
        cand = m.group(1).upper()
        if cand in US_STATE_CODES:
            return cand
    return s


def strip_state_suffix_from_city(city: Any) -> str:
    s = normalize_empty(city)
    if not s:
        return ""
    parts = s.strip().split()
    if len(parts) >= 2:
        last = parts[-1].upper().strip(".,;:()[]{}")
        if last in US_STATE_CODES:
            return " ".join(parts[:-1]).strip()
    return s


def normalize_ounces_candidate(x: Any) -> str:
    s = normalize_empty(x).lower()
    if not s:
        return ""
    s = s.replace("ounces", "ounce")
    s = s.replace("ozs", "oz")
    s = s.replace("0z", "oz")
    s = s.replace("o.z.", "oz")
    s = re.sub(r"\s+", " ", s)
    m = re.search(r"(-?\d+(?:\.\d+)?)", s)
    if not m:
        return normalize_empty(x)
    val = safe_float(m.group(1), None)
    if val is None or val <= 0 or val > 100:
        return normalize_empty(x)
    return f"{val:.1f} oz."


def normalize_abv_candidate(x: Any) -> str:
    s_raw = normalize_empty(x)
    s = s_raw.lower()
    if not s:
        return ""
    s_compact = s.replace("abv", "").replace(" ", "")
    m = re.search(r"(-?\d+(?:\.\d+)?)", s_compact)
    if not m:
        return normalize_empty(x)
    val = safe_float(m.group(1), None)
    if val is None:
        return normalize_empty(x)
    has_percent = "%" in s_compact
    if has_percent:
        if val > 1.0:
            val = val / 100.0
        if val < 0 or val > 1:
            return normalize_empty(x)
        out = f"{val:.3f}".rstrip("0").rstrip(".")
        return "PERCENT_FORMAT_ERROR:" + out
    if val > 1.0:
        val = val / 100.0
    if val < 0 or val > 1:
        return normalize_empty(x)
    out = f"{val:.3f}"
    return out.rstrip("0").rstrip(".") if "." in out else out


def normalize_ibu_candidate(x: Any) -> str:
    s = normalize_empty(x).lower()
    if not s:
        return ""
    s = s.replace("ibu", "")
    m = re.search(r"(-?\d+(?:\.\d+)?)", s)
    if not m:
        return normalize_empty(x)
    val = safe_float(m.group(1), None)
    if val is None or val < 0 or val > 1000:
        return normalize_empty(x)
    if abs(val - round(val)) < 1e-9:
        return str(int(round(val)))
    return f"{val:.1f}".rstrip("0").rstrip(".")


def normalize_numeric_id(x: Any) -> str:
    s = normalize_empty(x)
    if not s:
        return ""
    m = re.search(r"\d+", s)
    if not m:
        return s
    return str(int(m.group(0)))


def normalize_language_candidate(x: Any) -> str:
    s = normalize_empty(x)
    if not s:
        return ""
    lower = s.lower()
    if "abstract language" in lower:
        parts = re.split(r"abstract language\s*:\s*", lower)
        if len(parts) >= 2:
            lower = parts[-1].strip(";: ,.")
    lower = lower.strip(";: ,.")
    if lower in LANGUAGE_CANONICAL_MAP:
        return LANGUAGE_CANONICAL_MAP[lower]
    token = re.sub(r"[^a-z]", "", lower)
    if token in LANGUAGE_CANONICAL_MAP:
        return LANGUAGE_CANONICAL_MAP[token]
    if len(token) == 3 and token.isalpha():
        return token
    return lower


def normalize_issn_candidate(x: Any) -> str:
    s = normalize_empty(x)
    if not s:
        return ""
    raw = s.upper().strip()
    raw = raw.replace("ISSN", "").replace(":", " ").strip()
    chars = re.sub(r"[^0-9X]", "", raw)
    if len(chars) != 8:
        return s
    return f"{chars[:4]}-{chars[4:]}"


def normalize_date_candidate(x: Any) -> str:
    s = normalize_empty(x)
    if not s:
        return ""
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
    return s


def normalize_numeric_int_candidate(x: Any, min_v=None, max_v=None) -> str:
    s = normalize_empty(x)
    if not s:
        return ""
    m = re.search(r"-?\d+", s)
    if not m:
        return s
    v = int(m.group(0))
    if min_v is not None and v < min_v:
        return s
    if max_v is not None and v > max_v:
        return s
    return str(v)


def normalize_pagination_candidate(x: Any) -> str:
    s = normalize_empty(x)
    if not s:
        return ""
    s2 = s.strip()
    s2 = s2.replace("–", "-").replace("—", "-").replace("−", "-")
    s2 = re.sub(r"\s*-\s*", "-", s2)
    s2 = re.sub(r"\s+", " ", s2).strip()
    return s2


def canonical_income(x: Any) -> str:
    s = normalize_empty(x)
    if not s:
        return ""
    t = s.replace(".", "").replace(" ", "")
    low = t.lower()
    if low in {"<=50k", "<=50", "less_than_50k", "lessthan50k", "less50k", "0", "false"}:
        return "LessThan50K"
    if low in {">50k", ">50", "more_than_50k", "morethan50k", "more50k", "1", "true"}:
        return "MoreThan50K"
    return s


def canonical_adult_domain_value(column: Any, value: Any) -> str:
    col = normalize_col_name(column)
    s = normalize_empty(value)
    if not s:
        return ""
    if col == "income":
        return canonical_income(s)
    domain = ADULT_DOMAIN_VALUES.get(col)
    if domain:
        for v in domain:
            if v.lower() == s.lower():
                return v
        s2 = s.replace("_", "-")
        for v in domain:
            if v.lower() == s2.lower():
                return v
        s3 = s.replace("_", " ")
        for v in domain:
            if v.lower().replace("-", " ") == s3.lower().replace("-", " "):
                return v
    return s


def parse_numeric_or_range(x: Any) -> Tuple[Optional[float], Optional[float]]:
    s = normalize_empty(x)
    if not s:
        return None, None
    s = s.replace(",", "").strip()
    low = s.lower().replace(" ", "")
    if low.startswith("<"):
        try:
            v = float(low[1:])
            return 0.0, v - 1
        except Exception:
            return None, None
    if low.startswith(">"):
        try:
            v = float(low[1:])
            return v + 1, 120.0
        except Exception:
            return None, None
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", s)
    if m:
        return float(m.group(1)), float(m.group(2))
    try:
        v = float(s)
        return v, v
    except Exception:
        return None, None


def canonical_numeric_or_range(x: Any) -> str:
    lo, hi = parse_numeric_or_range(x)
    if lo is None or hi is None:
        return normalize_empty(x)

    def fmt(v: float) -> str:
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        return f"{v:.6f}".rstrip("0").rstrip(".")
    if abs(lo - hi) < 1e-9:
        return fmt(lo)
    return f"{fmt(lo)}-{fmt(hi)}"


def fix_year_ocr_like(s: str) -> str:
    trans = str.maketrans({
        "O": "0", "o": "0", "Q": "0",
        "I": "1", "l": "1", "L": "1", "|": "1",
        "S": "5", "s": "5",
        "B": "8",
        "Z": "2", "z": "2",
    })
    return s.translate(trans)


def parse_year_like(x: Any) -> Optional[int]:
    s = normalize_empty(x)
    if not s:
        return None
    s = fix_year_ocr_like(s)
    m_float = re.fullmatch(r"([12]\d{3})\.0+", s)
    if m_float:
        return int(m_float.group(1))
    m = re.search(r"(?<!\d)([12]\d{3})(?!\d)", s)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def canonical_year(column: Any, value: Any) -> str:
    col = normalize_col_name(column)
    y = parse_year_like(value)
    if y is None:
        return normalize_empty(value)
    if col == "birthyear":
        if 1900 <= y <= 2025:
            return str(y)
    if col == "season":
        if 1900 <= y <= 2035:
            return str(y)
    return str(y)


def canonical_position(x: Any) -> str:
    s = normalize_empty(x)
    if not s:
        return ""
    key = re.sub(r"[\s_\-]+", " ", s.strip().lower())
    key_compact = key.replace(" ", "")
    if key in POSITION_ALIASES:
        return POSITION_ALIASES[key]
    if key_compact in POSITION_ALIASES:
        return POSITION_ALIASES[key_compact]
    for v in SOCCER_POSITION_VALUES:
        if v.lower() == key or v.lower() == key_compact:
            return v
    return s


def text_key(s: Any, case_insensitive: bool = True) -> str:
    s = normalize_empty(s)
    if not s:
        return ""
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\s_]+", "_", s.strip())
    return s.lower() if case_insensitive else s


def normalize_for_compare(dataset: str, column: str, value: Any, normalized: bool, case_insensitive_text: bool) -> str:
    s = normalize_empty(value)
    if not s:
        return ""

    col = normalize_col_name(column)

    if not normalized:
        s = re.sub(r"\s+", " ", s.strip())
        return s.lower() if case_insensitive_text else s

    if dataset == "beers":
        if col == "state":
            return normalize_state_candidate(s)
        if col == "city":
            return strip_state_suffix_from_city(s)
        if col == "ounces":
            return normalize_ounces_candidate(s)
        if col == "abv":
            return normalize_abv_candidate(s)
        if col == "ibu":
            return normalize_ibu_candidate(s)
        if col in {"index", "id", "brewery_id"}:
            return normalize_numeric_id(s)
        return s

    if dataset == "rayyan":
        if col == "article_language":
            return normalize_language_candidate(s)
        if col == "journal_issn":
            return normalize_issn_candidate(s)
        if col == "article_jcreated_at":
            return normalize_date_candidate(s)
        if col == "article_jvolumn":
            return normalize_numeric_int_candidate(s, min_v=0, max_v=10000)
        if col == "article_jissue":
            return normalize_numeric_int_candidate(s, min_v=0, max_v=1000)
        if col == "article_pagination":
            return normalize_pagination_candidate(s)
        if col in {"index", "id"}:
            return normalize_numeric_int_candidate(s, min_v=0)
        s2 = re.sub(r"\s+", " ", s.strip())
        if case_insensitive_text and col in RAYYAN_TEXT_METADATA_COLUMNS:
            return s2.lower()
        return s2

    if dataset == "adult":
        numeric_cols = {normalize_col_name(c) for c in NUMERIC_OR_RANGE_COLUMNS}
        if col == "income":
            return canonical_income(s)
        if col in ADULT_DOMAIN_VALUES:
            return canonical_adult_domain_value(col, s)
        if col in numeric_cols:
            return canonical_numeric_or_range(s)
        s2 = re.sub(r"\s+", " ", s.strip())
        return s2.lower() if case_insensitive_text else s2

    if dataset == "soccer":
        if col in {"birthyear", "season"}:
            return canonical_year(col, s)
        if col == "position":
            v = canonical_position(s)
            return v.lower() if case_insensitive_text else v
        if col in SOCCER_TEXT_COLUMNS:
            return text_key(s, case_insensitive=case_insensitive_text)
        s2 = re.sub(r"\s+", " ", s.strip())
        return s2.lower() if case_insensitive_text else s2

    # hospital / flights default.
    s = re.sub(r"\s+", " ", s.strip())
    return s.lower() if case_insensitive_text else s


# ============================================================
# 4. Repair result loading and application
# ============================================================

def find_repair_value_column(df: pd.DataFrame) -> str:
    candidates = [
        "repaired_value",
        "predicted_repair_value",
        "predicted_repair",
        "repair_value",
        "predicted_value",
        "chosen_candidate_value",
        "student_best_candidate",
        "student_top_candidate",
        "top_candidate_value",
        "candidate_value",
        "teacher_best_candidate",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        "修复结果文件中未找到修复值字段。支持字段包括："
        "repaired_value / predicted_repair_value / predicted_repair / repair_value / predicted_value / "
        "chosen_candidate_value / student_best_candidate / student_top_candidate / "
        "top_candidate_value / candidate_value / teacher_best_candidate"
    )


OPTIONAL_INFER_COLUMNS = [
    "dirty_value",
    "chosen_candidate_value",
    "student_top_candidate",
    "student_best_candidate",
    "top_candidate_value",
    "student_score",
    "student_margin",
    "margin",
    "top2_score",
    "evidence_strength",
    "rule_support",
    "neighbor_majority_ratio",
    "gate_passed",
    "gate_pass",
    "pass_gate",
    "gate_reason",
    "student_reason_pred",
    "candidate_count",
    "candidate_source",
    "candidate_source_text",
    "top_candidate_sources",
    "candidate_rule_support_ratio",
    "adult_gain",
    "adult_profile_support",
    "relationship_consistency_gain",
    "education_age_conflict_reduction",
    "candidate_domain_validity_gain",
    "candidate_in_adult_domain",
    "soccer_gain",
    "soccer_profile_support",
    "soccer_consistency_gain",
    "candidate_in_soccer_domain",
    "position_consistency_gain",
    "birthyear_season_age_gain",
    "team_context_gain",
    "row_team",
    "row_city",
    "row_stadium",
    "row_manager",
    "rayyan_consensus_strength",
    "rayyan_consensus_confidence",
    "rayyan_consensus_support_count",
    "is_rayyan_consensus_candidate",
    "contains_rayyan_metadata_source",
    "contains_journal_metadata_source",
    "contains_article_metadata_source",
    "contains_canonical_source",
    "contains_rayyan_consensus_source",
    "rayyan_consensus_candidate",
    "is_from_rayyan_consensus",
    "rayyan_consensus_table_match",
    "rayyan_consensus_table_max_confidence",
    "rayyan_consensus_table_max_margin",
    "rayyan_consensus_table_max_support_count",
    "rayyan_consensus_table_high_conf_match",
    "rayyan_candidate_is_language",
    "rayyan_candidate_is_issn",
    "rayyan_candidate_is_valid_issn",
    "rayyan_candidate_is_date",
    "rayyan_candidate_is_volume",
    "rayyan_candidate_is_issue",
    "rayyan_candidate_is_pagination",
    "rayyan_candidate_is_format_restoration",
    "candidate_equals_allowed_consensus_value",
    "candidate_and_allowed_both_rayyan_consensus",
    "candidate_under_strong_rayyan_consensus_cell",
    "allowed_consensus_name",
    "allowed_lhs_key",
    "allowed_consensus_value",
    "allowed_consensus_confidence",
    "allowed_consensus_margin",
    "allowed_consensus_support_count",
    "allowed_consensus_total_count",
    "allowed_consensus_reason",
    "allowed_is_rayyan_metadata_consensus",
    "allowed_is_strong_rayyan_consensus",
    "is_hard_case",
]


def resolve_repair_result_file(path: str) -> str:
    path = str(path or "").strip()
    if not path:
        raise FileNotFoundError("repair_result_file is empty")
    if not os.path.exists(path):
        raise FileNotFoundError(f"修复结果文件不存在: {path}")
    if os.path.isdir(path):
        direct = os.path.join(path, "inference_top1_repairs.csv")
        if os.path.exists(direct):
            return direct
        raise IsADirectoryError(f"repair_result_file 是目录且没有 inference_top1_repairs.csv: {path}")
    return path


def load_repair_results(path: str) -> pd.DataFrame:
    path = resolve_repair_result_file(path)
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)

    if "row_id" not in df.columns or "column" not in df.columns:
        raise ValueError("修复结果文件必须包含 row_id 和 column")

    df["row_id"] = pd.to_numeric(df["row_id"], errors="raise").astype(int)
    df["column"] = df["column"].astype(str).str.strip()

    repair_col = find_repair_value_column(df)

    keep_cols = ["row_id", "column", repair_col]
    for c in OPTIONAL_INFER_COLUMNS:
        if c in df.columns and c not in keep_cols:
            keep_cols.append(c)

    out = df[keep_cols].copy()
    out = out.rename(columns={repair_col: "predicted_repair"})
    out["predicted_repair"] = out["predicted_repair"].map(norm_raw)

    out = out.drop_duplicates(subset=["row_id", "column"], keep="first").reset_index(drop=True)
    return out


def apply_repairs_to_dirty(dirty_df: pd.DataFrame, repair_df: pd.DataFrame) -> Tuple[pd.DataFrame, int, int]:
    repaired_df = stringify_table(dirty_df)
    n_rows = len(repaired_df)
    valid_columns = set(repaired_df.columns)
    skipped_bad_row = 0
    skipped_bad_col = 0

    for _, r in repair_df.iterrows():
        row_id = int(r["row_id"])
        column = str(r["column"]).strip()
        pred_val = norm_raw(r["predicted_repair"])

        if row_id < 0 or row_id >= n_rows:
            skipped_bad_row += 1
            continue
        if column not in valid_columns:
            skipped_bad_col += 1
            continue

        repaired_df.at[row_id, column] = pred_val

    if skipped_bad_row > 0:
        print(f"[WARN] 跳过越界 row_id 修复记录数: {skipped_bad_row}")
    if skipped_bad_col > 0:
        print(f"[WARN] 跳过不存在 column 修复记录数: {skipped_bad_col}")

    return repaired_df, skipped_bad_row, skipped_bad_col


# ============================================================
# 5. Full table evaluation
# ============================================================

def build_full_table_eval(
    dataset: str,
    clean_df: pd.DataFrame,
    dirty_df: pd.DataFrame,
    repaired_df: pd.DataFrame,
    repair_df: pd.DataFrame,
    normalized_comparison: bool,
    case_insensitive_text: bool,
    strict_mode: bool,
) -> pd.DataFrame:
    repair_pos_set = set((int(r["row_id"]), str(r["column"]).strip()) for _, r in repair_df.iterrows())
    repair_info_map = {}
    for _, r in repair_df.iterrows():
        repair_info_map[(int(r["row_id"]), str(r["column"]).strip())] = r.to_dict()

    rows = []
    for row_id in range(len(clean_df)):
        for column in clean_df.columns:
            column = str(column)

            clean_val = norm_raw(clean_df.at[row_id, column])
            dirty_val = norm_raw(dirty_df.at[row_id, column])
            repaired_val = norm_raw(repaired_df.at[row_id, column])

            dirty_norm = normalize_for_compare(dataset, column, dirty_val, normalized_comparison, case_insensitive_text)
            clean_norm = normalize_for_compare(dataset, column, clean_val, normalized_comparison, case_insensitive_text)
            repaired_norm = normalize_for_compare(dataset, column, repaired_val, normalized_comparison, case_insensitive_text)

            if strict_mode:
                dirty_cmp = normalize_for_compare(dataset, column, dirty_val, False, False)
                clean_cmp = normalize_for_compare(dataset, column, clean_val, False, False)
                repaired_cmp = normalize_for_compare(dataset, column, repaired_val, False, False)
            else:
                dirty_cmp, clean_cmp, repaired_cmp = dirty_norm, clean_norm, repaired_norm

            is_true_error = int(dirty_cmp != clean_cmp)
            is_touched_by_result = int((row_id, column) in repair_pos_set)
            is_changed = int(repaired_cmp != dirty_cmp)
            is_touched = int(is_touched_by_result or is_changed)
            is_final_correct = int(repaired_cmp == clean_cmp)

            if is_true_error == 1 and is_final_correct == 1:
                eval_type = "TP"
            elif is_true_error == 0 and is_final_correct == 0:
                eval_type = "FP"
            elif is_true_error == 1 and is_final_correct == 0:
                eval_type = "FN"
            else:
                eval_type = "TN"

            info = repair_info_map.get((row_id, column), {})

            row = {
                "row_id": row_id,
                "column": column,
                "dirty_value": dirty_val,
                "clean_value": clean_val,
                "repaired_value": repaired_val,
                "dirty_value_norm": dirty_norm,
                "clean_value_norm": clean_norm,
                "repaired_value_norm": repaired_norm,
                "is_true_error": is_true_error,
                "is_touched_by_result": is_touched_by_result,
                "is_changed": is_changed,
                "is_touched": is_touched,
                "is_final_correct": is_final_correct,
                "eval_type": eval_type,
            }

            for c in ["predicted_repair"] + OPTIONAL_INFER_COLUMNS:
                if c in info:
                    row[c] = info.get(c)

            rows.append(row)

    return pd.DataFrame(rows)


def compute_metrics(eval_df: pd.DataFrame) -> Dict[str, Any]:
    tp = int((eval_df["eval_type"] == "TP").sum())
    fp = int((eval_df["eval_type"] == "FP").sum())
    fn = int((eval_df["eval_type"] == "FN").sum())
    tn = int((eval_df["eval_type"] == "TN").sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)

    total_cells = len(eval_df)
    true_error_cells = int(eval_df["is_true_error"].sum())
    touched_cells = int(eval_df["is_touched"].sum())
    changed_cells = int(eval_df["is_changed"].sum())
    touched_by_result_cells = int(eval_df["is_touched_by_result"].sum())
    remaining_wrong_cells = int((eval_df["repaired_value_norm"] != eval_df["clean_value_norm"]).sum())
    unchanged_true_errors = int(
        ((eval_df["is_true_error"] == 1) & (eval_df["is_changed"] == 0)).sum()
    )

    return {
        "total_cells": total_cells,
        "true_error_cells": true_error_cells,
        "touched_cells": touched_cells,
        "touched_by_result_cells": touched_by_result_cells,
        "changed_cells": changed_cells,
        "remaining_wrong_cells_after_repair": remaining_wrong_cells,
        "unchanged_true_errors": unchanged_true_errors,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Accuracy": accuracy,
    }


def build_column_summary(eval_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column, g in eval_df.groupby("column", sort=False):
        tp = int((g["eval_type"] == "TP").sum())
        fp = int((g["eval_type"] == "FP").sum())
        fn = int((g["eval_type"] == "FN").sum())
        tn = int((g["eval_type"] == "TN").sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
        rows.append({
            "column": column,
            "total_cells": len(g),
            "true_error_cells": int(g["is_true_error"].sum()),
            "touched_cells": int(g["is_touched"].sum()),
            "touched_by_result_cells": int(g["is_touched_by_result"].sum()),
            "changed_cells": int(g["is_changed"].sum()),
            "remaining_wrong_cells_after_repair": int((g["repaired_value_norm"] != g["clean_value_norm"]).sum()),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Accuracy": accuracy,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["true_error_cells", "F1"], ascending=[False, False]).reset_index(drop=True)
    return out


def build_group_summary(eval_df: pd.DataFrame, group_col: str, name_col: str) -> pd.DataFrame:
    if group_col not in eval_df.columns:
        return pd.DataFrame()
    rows = []
    for val, g in eval_df[eval_df["is_touched_by_result"] == 1].groupby(group_col, dropna=False):
        tp = int((g["eval_type"] == "TP").sum())
        fp = int((g["eval_type"] == "FP").sum())
        fn = int((g["eval_type"] == "FN").sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall_on_touched_errors = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        rows.append({
            name_col: val,
            "rows": len(g),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Precision": precision,
            "Recall_on_group_errors": recall_on_touched_errors,
        })
    return pd.DataFrame(rows).sort_values(["rows", "Precision"], ascending=[False, False]).reset_index(drop=True) if rows else pd.DataFrame()


def build_dataset_extra_summaries(dataset: str, eval_df: pd.DataFrame, output_dir: str) -> List[str]:
    saved = []

    # Gate reason summary is useful for all datasets.
    gate_col = None
    for c in ["gate_reason", "gate_passed", "gate_pass", "pass_gate"]:
        if c in eval_df.columns:
            gate_col = c
            break
    if gate_col:
        path = os.path.join(output_dir, "gate_reason_eval_summary.csv")
        build_group_summary(eval_df, gate_col, gate_col).to_csv(path, index=False, encoding="utf-8-sig")
        saved.append(path)

    # Candidate/source summary is useful for all datasets if source columns exist.
    for src_col in ["candidate_source", "candidate_source_text", "top_candidate_sources"]:
        if src_col in eval_df.columns:
            path = os.path.join(output_dir, "candidate_source_eval_summary.csv")
            build_group_summary(eval_df, src_col, src_col).to_csv(path, index=False, encoding="utf-8-sig")
            saved.append(path)
            break

    if dataset == "beers":
        cols = ["state", "city", "ounces", "abv", "ibu", "brewery_id"]
        sub = eval_df[eval_df["column"].astype(str).str.lower().isin(cols)].copy()
        if not sub.empty:
            path = os.path.join(output_dir, "beers_normalized_eval_summary.csv")
            build_column_summary(sub).to_csv(path, index=False, encoding="utf-8-sig")
            saved.append(path)

    elif dataset == "rayyan":
        sub = eval_df[eval_df["column"].astype(str).str.lower().isin(RAYYAN_CANONICAL_COLUMNS)].copy()
        if not sub.empty:
            path = os.path.join(output_dir, "rayyan_canonical_eval_summary.csv")
            build_column_summary(sub).to_csv(path, index=False, encoding="utf-8-sig")
            saved.append(path)

    elif dataset == "adult":
        domain_cols = set(ADULT_DOMAIN_VALUES.keys())
        sub = eval_df[eval_df["column"].map(normalize_col_name).isin(domain_cols)].copy()
        if not sub.empty:
            path = os.path.join(output_dir, "adult_domain_eval_summary.csv")
            build_column_summary(sub).to_csv(path, index=False, encoding="utf-8-sig")
            saved.append(path)

    elif dataset == "soccer":
        domain_cols = {"position", "birthyear", "season", "name", "surname", "birthplace"}
        sub = eval_df[eval_df["column"].map(normalize_col_name).isin(domain_cols)].copy()
        if not sub.empty:
            path = os.path.join(output_dir, "soccer_domain_eval_summary.csv")
            build_column_summary(sub).to_csv(path, index=False, encoding="utf-8-sig")
            saved.append(path)

    return saved


# ============================================================
# 6. Main
# ============================================================

def main():
    args = parse_args()
    cfg = DATASET_CONFIGS[args.dataset]

    clean_file = args.clean_file or cfg.clean_file
    dirty_file = args.dirty_file or cfg.dirty_file
    repair_result_file = args.repair_result_file or cfg.repair_result_file
    output_dir = args.output_dir or cfg.output_dir
    normalized_comparison = cfg.normalized_comparison if args.normalized_comparison == -1 else bool(args.normalized_comparison)
    case_insensitive_text = cfg.case_insensitive_text if args.case_insensitive_text == -1 else bool(args.case_insensitive_text)
    strict_mode = bool(args.strict_mode)

    ensure_dir(output_dir)

    output_summary_csv = os.path.join(output_dir, "full_table_eval_summary.csv")
    output_cell_details_csv = os.path.join(output_dir, "full_table_eval_all_cells.csv")
    output_tp_csv = os.path.join(output_dir, "full_table_tp_details.csv")
    output_fp_csv = os.path.join(output_dir, "full_table_fp_details.csv")
    output_fn_csv = os.path.join(output_dir, "full_table_fn_details.csv")
    output_tn_csv = os.path.join(output_dir, "full_table_tn_details.csv")
    output_repaired_table_csv = os.path.join(output_dir, "repaired_table.csv")
    output_touched_csv = os.path.join(output_dir, "touched_cells_details.csv")
    output_column_summary_csv = os.path.join(output_dir, "column_level_eval_summary.csv")

    print("========== Unified full-table repair evaluation ==========")
    print(f"dataset: {args.dataset}")
    print(f"clean_file: {clean_file}")
    print(f"dirty_file: {dirty_file}")
    print(f"repair_result_file: {repair_result_file}")
    print(f"output_dir: {output_dir}")
    print(f"normalized_comparison: {normalized_comparison}")
    print(f"case_insensitive_text: {case_insensitive_text}")
    print(f"strict_mode: {strict_mode}")

    clean_df_raw = read_table(clean_file)
    dirty_df_raw = read_table(dirty_file)
    validate_same_shape(clean_df_raw, dirty_df_raw)

    clean_df = stringify_table(clean_df_raw)
    dirty_df = stringify_table(dirty_df_raw)

    repair_df = load_repair_results(repair_result_file)
    repaired_df, skipped_bad_row, skipped_bad_col = apply_repairs_to_dirty(dirty_df, repair_df)
    repaired_df.to_csv(output_repaired_table_csv, index=False, encoding="utf-8-sig")

    eval_df = build_full_table_eval(
        dataset=args.dataset,
        clean_df=clean_df,
        dirty_df=dirty_df,
        repaired_df=repaired_df,
        repair_df=repair_df,
        normalized_comparison=normalized_comparison,
        case_insensitive_text=case_insensitive_text,
        strict_mode=strict_mode,
    )

    metrics = compute_metrics(eval_df)
    metrics["dataset"] = args.dataset
    metrics["clean_file"] = clean_file
    metrics["dirty_file"] = dirty_file
    metrics["repair_result_file"] = resolve_repair_result_file(repair_result_file)
    metrics["normalized_comparison"] = int(normalized_comparison)
    metrics["case_insensitive_text"] = int(case_insensitive_text)
    metrics["strict_mode"] = int(strict_mode)
    metrics["skipped_bad_row_repairs"] = skipped_bad_row
    metrics["skipped_bad_col_repairs"] = skipped_bad_col

    column_summary_df = build_column_summary(eval_df)

    eval_df.to_csv(output_cell_details_csv, index=False, encoding="utf-8-sig")
    eval_df[eval_df["eval_type"] == "TP"].to_csv(output_tp_csv, index=False, encoding="utf-8-sig")
    eval_df[eval_df["eval_type"] == "FP"].to_csv(output_fp_csv, index=False, encoding="utf-8-sig")
    eval_df[eval_df["eval_type"] == "FN"].to_csv(output_fn_csv, index=False, encoding="utf-8-sig")
    eval_df[eval_df["eval_type"] == "TN"].to_csv(output_tn_csv, index=False, encoding="utf-8-sig")
    eval_df[eval_df["is_touched"] == 1].to_csv(output_touched_csv, index=False, encoding="utf-8-sig")
    column_summary_df.to_csv(output_column_summary_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame([metrics]).to_csv(output_summary_csv, index=False, encoding="utf-8-sig")

    extra_paths = []
    if int(args.output_extra_summaries) == 1:
        extra_paths = build_dataset_extra_summaries(args.dataset, eval_df, output_dir)

    print(f"\n========== {args.dataset} 全表端到端修复评估结果 ==========")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")

    print("\n---- 各列评估摘要 Top ----")
    if not column_summary_df.empty:
        show_cols = [
            "column", "true_error_cells", "touched_cells", "changed_cells",
            "TP", "FP", "FN", "Precision", "Recall", "F1",
        ]
        print(column_summary_df[show_cols].head(30).to_string(index=False))

    print(f"\n[OK] 评估摘要已保存: {output_summary_csv}")
    print(f"[OK] 修复后的整表已保存: {output_repaired_table_csv}")
    print(f"[OK] 全量单元格评估明细已保存: {output_cell_details_csv}")
    print(f"[OK] touched 明细已保存: {output_touched_csv}")
    print(f"[OK] 列级评估摘要已保存: {output_column_summary_csv}")
    print(f"[OK] TP 明细已保存: {output_tp_csv}")
    print(f"[OK] FP 明细已保存: {output_fp_csv}")
    print(f"[OK] FN 明细已保存: {output_fn_csv}")
    print(f"[OK] TN 明细已保存: {output_tn_csv}")
    for p in extra_paths:
        print(f"[OK] 额外摘要已保存: {p}")


if __name__ == "__main__":
    main()

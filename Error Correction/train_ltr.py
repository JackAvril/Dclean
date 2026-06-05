#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified pairwise LTR student training script.

Datasets:
  hospital, flights, beers, rayyan, adult, soccer

Purpose
-------
This script merges the common training logic of the six dataset-specific
train_ltr.py scripts into one readable implementation.

It preserves the standard output files used by the original scripts:

  pairwise_train_materialized.csv
  training_log.csv
  feature_columns.json
  reason_label_mapping.json
  column_profiles.json
  best_model.pt
  last_model.pt

Common training logic
---------------------
1. Read candidate feature CSV.
2. Read LLM pairwise responses JSONL.
3. Materialize pairwise training rows:
      (feature(candidate_a), feature(candidate_b), winner, reason label, weight)
4. Infer numeric feature columns and column profiles.
5. Train a shared MLP scorer s_theta(c, v).
6. Use pairwise margin loss:
      - log sigmoid(s(winner) - s(loser))
7. Optionally train an auxiliary reason classifier.
8. Optionally apply light rule/profile regularization.
9. Save best/last checkpoints and logs.

Dataset-specific differences are in DATASET_CONFIGS:
  - output_dir default
  - column weights
  - domain/canonicalization helpers
  - feature boosts / derived features
  - regularization strength

This file is designed for ablation experiments. It is not a base64/plugin wrapper.
"""

import argparse
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. Dataset configs
# ============================================================

EMPTY_TOKENS = {"", "empty", "null", "none", "nan", "na", "n/a", "missing", "?", "{null}"}


@dataclass
class DatasetConfig:
    name: str
    default_output_dir: str
    column_weight_map: Dict[str, float]
    default_sample_weight: float = 1.0
    rule_reg_strength: float = 0.05
    reason_loss_weight: float = 0.2
    pairwise_loss_weight: float = 1.0
    high_value_columns: set = field(default_factory=set)
    domain_values: Dict[str, List[str]] = field(default_factory=dict)
    aliases: Dict[str, str] = field(default_factory=dict)


COMMON_ALIASES = {
    "jounral_abbreviation": "journal_abbreviation",
    "marital-status": "maritalstatus",
    "marital_status": "maritalstatus",
    "marital status": "maritalstatus",
    "hours-per-week": "hoursperweek",
    "hours_per_week": "hoursperweek",
    "hours per week": "hoursperweek",
    "native-country": "country",
    "native_country": "country",
    "birth_year": "birthyear",
    "birth year": "birthyear",
    "birth-year": "birthyear",
    "season_year": "season",
    "year": "season",
    "club": "team",
    "football_club": "team",
    "coach": "manager",
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
    "relationship": ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"],
    "race": ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"],
    "sex": ["Female", "Male"],
    "income": ["LessThan50K", "MoreThan50K", "<=50K", ">50K"],
}

SOCCER_POSITION_VALUES = ["Goalkeeper", "Defender", "Midfield", "Midfielder", "Forward", "Striker", "Winger"]

DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "hospital": DatasetConfig(
        name="hospital",
        default_output_dir="pairwise_ltr_student_output_iter",
        column_weight_map={
            "Score": 2.5,
            "Sample": 2.2,
            "Stateavg": 2.5,
            "MeasureCode": 2.5,
            "MeasureName": 2.0,
            "Condition": 1.5,
        },
        high_value_columns={"Score", "Sample", "Stateavg", "MeasureCode", "MeasureName"},
    ),
    "flights": DatasetConfig(
        name="flights",
        default_output_dir="pairwise_ltr_student_output_flights",
        column_weight_map={
            "sched_dep_time": 2.3,
            "act_dep_time": 2.0,
            "sched_arr_time": 2.3,
            "act_arr_time": 2.0,
        },
        high_value_columns={"sched_dep_time", "act_dep_time", "sched_arr_time", "act_arr_time"},
    ),
    "beers": DatasetConfig(
        name="beers",
        default_output_dir="pairwise_ltr_student_output_beers",
        column_weight_map={
            "state": 2.4,
            "city": 2.0,
            "ounces": 2.2,
            "abv": 2.1,
            "ibu": 2.1,
            "brewery_id": 1.9,
            "brewery_name": 1.9,
            "beer_name": 1.7,
            "style": 1.8,
        },
        high_value_columns={"state", "city", "ounces", "abv", "ibu", "brewery_id", "brewery_name", "beer_name", "style"},
    ),
    "rayyan": DatasetConfig(
        name="rayyan",
        default_output_dir="pairwise_ltr_student_output_rayyan",
        column_weight_map={
            "article_language": 2.4,
            "journal_issn": 2.3,
            "journal_title": 1.8,
            "jounral_abbreviation": 1.8,
            "journal_abbreviation": 1.8,
            "article_jcreated_at": 2.1,
            "article_jvolumn": 1.9,
            "article_jissue": 1.9,
            "article_pagination": 1.7,
            "article_title": 1.5,
        },
        high_value_columns={
            "article_language", "journal_issn", "article_jcreated_at",
            "article_jvolumn", "article_jissue", "article_pagination",
        },
    ),
    "adult": DatasetConfig(
        name="adult",
        default_output_dir="pairwise_ltr_student_output_adult",
        column_weight_map={
            "relationship": 2.4,
            "sex": 2.2,
            "maritalstatus": 2.1,
            "education": 1.8,
            "age": 1.6,
            "hoursperweek": 1.5,
            "income": 1.5,
            "workclass": 1.25,
            "occupation": 1.25,
            "country": 1.15,
            "race": 1.10,
        },
        high_value_columns={"relationship", "sex", "maritalstatus", "education", "age", "hoursperweek", "income"},
        domain_values=ADULT_DOMAIN_VALUES,
        rule_reg_strength=0.04,
    ),
    "soccer": DatasetConfig(
        name="soccer",
        default_output_dir="pairwise_ltr_student_output_soccer",
        column_weight_map={
            "position": 2.8,
            "birthyear": 2.2,
            "season": 2.2,
            "team": 1.45,
            "city": 1.40,
            "stadium": 1.40,
            "manager": 1.35,
            "birthplace": 1.80,
            "name": 2.30,
            "surname": 2.30,
        },
        high_value_columns={"position", "birthyear", "season", "name", "surname", "birthplace"},
        domain_values={"position": SOCCER_POSITION_VALUES},
    ),
}


# ============================================================
# 2. Basic helpers
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Unified train / resume pairwise LTR student")
    p.add_argument("--dataset", required=True, choices=sorted(DATASET_CONFIGS.keys()))
    p.add_argument("--candidate_features", default="repair_candidate_features_v2.csv")
    p.add_argument("--pairwise_responses", default="repair_pairwise_responses_v2.jsonl")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--resume_checkpoint", default=None)

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hidden_dims", default="128,64")
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--early_stopping_patience", type=int, default=6)
    p.add_argument("--grad_clip_norm", type=float, default=5.0)

    # Ablation switches.
    p.add_argument("--disable_dataset_derived_features", type=int, default=0)
    p.add_argument("--disable_rule_regularization", type=int, default=0)
    p.add_argument("--disable_reason_head", type=int, default=0)
    p.add_argument("--disable_column_weights", type=int, default=0)

    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


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


def normalize_col_name(column: Any) -> str:
    c = norm_lower(column)
    return COMMON_ALIASES.get(c, c)


def is_empty_token(x: Any) -> int:
    return int(norm_lower(x) in EMPTY_TOKENS)


def looks_like_code(x: Any) -> int:
    s = norm_text(x)
    if not s:
        return 0
    return int(((bool(re.search(r"[_\-]", s)) or bool(re.search(r"\d", s))) and len(s) <= 80))


def prefix_token(x: Any) -> str:
    s = norm_text(x)
    parts = re.split(r"[_\-]", s)
    return parts[0] if parts and parts[0] else ""


def family_token(x: Any) -> str:
    s = norm_text(x)
    parts = re.split(r"[_\-]", s)
    return parts[1] if len(parts) >= 2 else (parts[0] if parts else "")


def edit_ratio(a: Any, b: Any) -> float:
    import difflib
    a = norm_lower(a)
    b = norm_lower(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return float(difflib.SequenceMatcher(None, a, b).ratio())


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
    return safe_int(m.group(1), None) if m else None


def parse_time_to_minutes_simple(x: Any) -> Optional[int]:
    s = canonical_empty_text(x).lower()
    if s in {"", "empty"}:
        return None
    s = s.replace(".", "")
    s = re.sub(r"\s+", " ", s).strip()
    ampm = None
    if "am" in s:
        ampm = "am"
    elif "pm" in s:
        ampm = "pm"
    s2 = s.replace("am", "").replace("pm", "").strip()
    hour = minute = None
    m = re.fullmatch(r"(\d{1,2})\s*:\s*(\d{2})", s2)
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


def is_percent_like(x: Any) -> int:
    return int(bool(re.fullmatch(r"\d{1,3}%", canonical_empty_text(x))))


def patients_like(x: Any) -> int:
    return int("patient" in norm_lower(x) or "patients" in norm_lower(x))


def canonical_position(x: Any) -> str:
    raw = canonical_empty_text(x)
    if raw == "empty":
        return "empty"
    compact = re.sub(r"[^a-z]", "", raw.lower())
    mp = {
        "goalkeeper": "Goalkeeper", "goalkeepers": "Goalkeeper", "gk": "Goalkeeper",
        "defender": "Defender", "defenders": "Defender", "dfender": "Defender",
        "midfield": "Midfield", "midfielder": "Midfield", "midfielders": "Midfield",
        "forward": "Forward", "forwards": "Forward", "foward": "Forward",
        "striker": "Striker", "winger": "Winger",
    }
    return mp.get(compact, raw)


def canonical_adult_value(column: Any, value: Any) -> str:
    col = normalize_col_name(column)
    raw = canonical_empty_text(value)
    if raw == "empty":
        return "empty"

    if col == "income":
        s = raw.replace(".", "").replace(" ", "").lower()
        if s in {"<=50k", "<=50", "lessthan50k", "less_than_50k", "less50k", "0"}:
            return "LessThan50K"
        if s in {">50k", ">50", "morethan50k", "more_than_50k", "more50k", "1"}:
            return "MoreThan50K"

    domains = ADULT_DOMAIN_VALUES.get(col)
    if domains:
        raw2 = raw.replace("_", "-")
        raw3 = raw.replace("_", " ")
        for v in domains:
            if norm_lower(v) == norm_lower(raw) or norm_lower(v) == norm_lower(raw2):
                return v
            if norm_lower(v).replace("-", " ") == norm_lower(raw3).replace("-", " "):
                return v
    return raw


def normalize_language_candidate(x: Any) -> str:
    s = norm_lower(x)
    mp = {
        "eng": "eng", "english": "eng", "en": "eng",
        "fre": "fre", "french": "fre", "fr": "fre", "fra": "fre",
        "ger": "ger", "german": "ger", "de": "ger", "deu": "ger",
        "spa": "spa", "spanish": "spa", "es": "spa",
        "jpn": "jpn", "japanese": "jpn", "ja": "jpn",
        "chi": "chi", "chinese": "chi", "zh": "chi",
    }
    token = re.sub(r"[^a-z]", "", s)
    return mp.get(s, mp.get(token, norm_text(x)))


def normalize_issn_candidate(x: Any) -> str:
    s = canonical_empty_text(x).upper().replace("ISSN", "").replace(":", " ")
    chars = re.sub(r"[^0-9X]", "", s)
    if len(chars) == 8:
        return f"{chars[:4]}-{chars[4:]}"
    return canonical_empty_text(x)


# ============================================================
# 3. JSONL and pairwise parsing
# ============================================================

def load_jsonl(path: str) -> List[dict]:
    rows = []
    decoder = json.JSONDecoder()
    if not os.path.exists(path):
        raise FileNotFoundError(f"pairwise response file not found: {path}")

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
                print(f"[WARN] skip unparsable line: {path}, line={line_no}")

    return rows


def maybe_parse_obj(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return {}
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def get_candidate_value_from_obj(obj: Dict[str, Any]) -> str:
    for k in ["value", "candidate_value", "repair_value", "predicted_repair"]:
        if k in obj and not pd.isna(obj[k]):
            return canonical_empty_text(obj[k])
    return ""


def normalize_winner(x: Any) -> str:
    s = str(x).strip().upper()
    if s in {"A", "CANDIDATE_A", "CANDIDATE A"}:
        return "A"
    if s in {"B", "CANDIDATE_B", "CANDIDATE B"}:
        return "B"
    return "Unknown"


def normalize_reason(reason: Any, winner: Any = None) -> str:
    s = norm_lower(reason)
    if not s:
        return "unknown"
    # Coarse reason labels shared by scripts.
    if any(t in s for t in ["rule", "constraint", "domain", "valid"]):
        return "rule_or_domain"
    if any(t in s for t in ["neighbor", "similar", "context", "row"]):
        return "context_or_neighbor"
    if any(t in s for t in ["profile", "consensus", "cooccur", "co-occurrence"]):
        return "profile_or_consensus"
    if any(t in s for t in ["format", "canonical", "normalize", "numeric", "time", "date", "issn", "language"]):
        return "format_or_canonical"
    if any(t in s for t in ["frequency", "common", "global"]):
        return "frequency"
    if normalize_winner(winner) == "Unknown":
        return "unknown_or_tie"
    return "other"


def build_response_records(pairwise_rows: List[dict]) -> List[dict]:
    records = []
    for obj in pairwise_rows:
        row_id = safe_int(obj.get("row_id"), None)
        column = obj.get("column")
        if row_id is None or column is None:
            continue

        cand_a_obj = maybe_parse_obj(obj.get("candidate_a", obj.get("candidate_a_obj", {})))
        cand_b_obj = maybe_parse_obj(obj.get("candidate_b", obj.get("candidate_b_obj", {})))

        a_val = (
            obj.get("candidate_a_value")
            or obj.get("candidate_A_value")
            or obj.get("a_value")
            or get_candidate_value_from_obj(cand_a_obj)
        )
        b_val = (
            obj.get("candidate_b_value")
            or obj.get("candidate_B_value")
            or obj.get("b_value")
            or get_candidate_value_from_obj(cand_b_obj)
        )

        winner = normalize_winner(obj.get("winner", obj.get("choice", obj.get("answer", "Unknown"))))
        if winner == "Unknown":
            continue
        if not a_val or not b_val:
            continue

        records.append({
            "row_id": int(row_id),
            "column": str(column),
            "candidate_a_value": canonical_empty_text(a_val),
            "candidate_b_value": canonical_empty_text(b_val),
            "winner": winner,
            "reason": obj.get("reason", obj.get("explanation", "")),
            "reason_label": normalize_reason(obj.get("reason", obj.get("explanation", "")), winner),
            "raw_response": obj.get("raw_response", ""),
        })

    return records


# ============================================================
# 4. Feature preparation
# ============================================================

NON_FEATURE_COLUMNS = {
    "row_id", "column", "dirty_value", "candidate_value",
    "candidate_rank", "candidate_source", "candidate_sources", "candidate_source_text",
    "extra_info", "neighbor_majority_value", "suggested_correct_value",
    "main_rule_type", "main_usage_role", "semantic_type", "detected_type",
    "candidate_canonical_value", "dirty_canonical_value",
    "all_candidates", "top_candidate_value", "top_candidate_sources",
    "candidate_obj_json",
    "raw_response", "reason", "reason_label", "winner",
    "candidate_a_value", "candidate_b_value",
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
        else:
            out["dirty_value"] = ""

    out["candidate_value"] = out["candidate_value"].map(canonical_empty_text)
    out["dirty_value"] = out["dirty_value"].map(canonical_empty_text)
    return out


def source_text(row: pd.Series) -> str:
    parts = []
    for c in ["candidate_sources", "candidate_source", "source", "sources", "extra_info"]:
        if c in row.index and not pd.isna(row[c]):
            parts.append(str(row[c]))
    return "|".join(parts).lower()


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

    # Source-derived generic flags.
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

    # Dataset-specific derived features.
    if cfg.name == "flights":
        time_cols = {"sched_dep_time", "act_dep_time", "sched_arr_time", "act_arr_time"}
        out["is_time_column"] = out["column_norm"].map(lambda c: int(c in time_cols or c.endswith("_time")))
        out["dirty_time_valid"] = out["dirty_value"].map(lambda x: int(parse_time_to_minutes_simple(x) is not None))
        out["candidate_time_valid"] = out["candidate_value"].map(lambda x: int(parse_time_to_minutes_simple(x) is not None))
        out["time_format_gain_auto"] = out["candidate_time_valid"] - out["dirty_time_valid"]

    elif cfg.name == "beers":
        state_codes = {
            "al","ak","az","ar","ca","co","ct","de","fl","ga","hi","ia","id","il","in","ks","ky","la","ma","md","me",
            "mi","mn","mo","ms","mt","nc","nd","ne","nh","nj","nm","nv","ny","oh","ok","or","pa","ri","sc","sd","tn",
            "tx","ut","va","vt","wa","wi","wv","wy","dc"
        }
        out["candidate_state_code_valid_auto"] = [
            int(c == "state" and norm_lower(v) in state_codes)
            for c, v in zip(out["column_norm"], out["candidate_value"])
        ]
        out["candidate_numeric_beer_valid_auto"] = [
            int(c in {"abv", "ibu", "ounces"} and bool(re.search(r"\d", norm_text(v))))
            for c, v in zip(out["column_norm"], out["candidate_value"])
        ]

    elif cfg.name == "rayyan":
        out["candidate_language_valid_auto"] = [
            int(c == "article_language" and len(normalize_language_candidate(v)) == 3)
            for c, v in zip(out["column_norm"], out["candidate_value"])
        ]
        out["candidate_issn_valid_auto"] = [
            int(c == "journal_issn" and bool(re.fullmatch(r"\d{4}-[\dX]{4}", normalize_issn_candidate(v).upper())))
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
            int(c == "position" and canonical_position(v) in SOCCER_POSITION_VALUES)
            for c, v in zip(out["column_norm"], out["candidate_value"])
        ]
        out["dirty_position_valid_auto"] = [
            int(c == "position" and canonical_position(v) in SOCCER_POSITION_VALUES)
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
            "candidate_has_high_conf_profile_majority", "candidate_entity_safe_fill", "candidate_position_profile_strong",
        ]:
            if c in out.columns:
                out["profile_majority_strength_auto"] += pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    return out


def infer_column_profiles(df: pd.DataFrame) -> Dict[str, Any]:
    profiles = {}
    for col, g in df.groupby("column"):
        vals = pd.concat([g["dirty_value"], g["candidate_value"]], ignore_index=True).dropna().astype(str)
        vals = vals.map(canonical_empty_text)
        vals = vals[vals != "empty"]

        if len(vals) == 0:
            profiles[str(col)] = {
                "archetype": "mixed",
                "avg_len": 0.0,
                "digits_ratio": 0.0,
                "percent_ratio": 0.0,
                "code_ratio": 0.0,
                "alpha_ratio": 0.0,
                "unique_count": 0,
                "sample_values": [],
            }
            continue

        avg_len = float(vals.map(lambda x: len(norm_text(x))).mean())
        digits_ratio = float(vals.map(lambda x: int(bool(re.fullmatch(r"\d+", norm_text(x))))).mean())
        percent_ratio = float(vals.map(lambda x: int(bool(re.fullmatch(r"\d{1,3}%", norm_text(x))))).mean())
        code_ratio = float(vals.map(looks_like_code).mean())
        alpha_ratio = float(vals.map(lambda x: int(bool(re.fullmatch(r"[A-Za-z\s\-\./'_]+", norm_text(x))))).mean())
        unique_count = int(vals.nunique())

        archetype = "mixed"
        if percent_ratio >= 0.55:
            archetype = "percent_pattern"
        elif digits_ratio >= 0.75 and avg_len >= 6:
            archetype = "long_digit_id"
        elif digits_ratio >= 0.75:
            archetype = "short_digit_id"
        elif code_ratio >= 0.55:
            archetype = "structured_code"
        elif unique_count <= 8 and alpha_ratio >= 0.8:
            archetype = "small_enum"
        elif alpha_ratio >= 0.8 and avg_len >= 30:
            archetype = "long_text"
        elif alpha_ratio >= 0.8:
            archetype = "short_text"

        profiles[str(col)] = {
            "archetype": archetype,
            "avg_len": avg_len,
            "digits_ratio": digits_ratio,
            "percent_ratio": percent_ratio,
            "code_ratio": code_ratio,
            "alpha_ratio": alpha_ratio,
            "unique_count": unique_count,
            "sample_values": vals.drop_duplicates().head(10).tolist(),
        }
    return profiles


def choose_feature_columns(df: pd.DataFrame) -> List[str]:
    feature_cols = []
    for c in df.columns:
        if c in NON_FEATURE_COLUMNS:
            continue
        if c.endswith("_json"):
            continue
        if df[c].dtype == "O":
            # Try numeric coercion.
            vals = pd.to_numeric(df[c], errors="coerce")
            if vals.notna().mean() < 0.50:
                continue
        vals = pd.to_numeric(df[c], errors="coerce")
        if vals.notna().any():
            feature_cols.append(c)
    return feature_cols


def get_feature_vector(row: pd.Series, feature_cols: List[str]) -> np.ndarray:
    vals = []
    for c in feature_cols:
        vals.append(safe_float(row.get(c), 0.0))
    return np.array(vals, dtype=np.float32)


def build_candidate_lookup(df: pd.DataFrame) -> Dict[Tuple[int, str, str], int]:
    lookup = {}
    for idx, r in df.iterrows():
        key = (int(r["row_id"]), str(r["column"]), norm_lower(r["candidate_value"]))
        if key not in lookup:
            lookup[key] = idx
    return lookup


def materialize_pairwise(
    cand_df: pd.DataFrame,
    responses: List[dict],
    cfg: DatasetConfig,
    feature_cols: List[str],
    disable_column_weights: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    lookup = build_candidate_lookup(cand_df)
    response_records = build_response_records(responses)

    reason_labels = sorted(set(r["reason_label"] for r in response_records) | {"unknown"})
    reason_to_id = {r: i for i, r in enumerate(reason_labels)}

    rows = []
    skipped = 0
    for r in response_records:
        row_id = r["row_id"]
        col = str(r["column"])
        col_norm = normalize_col_name(col)
        a_val = norm_lower(r["candidate_a_value"])
        b_val = norm_lower(r["candidate_b_value"])

        idx_a = lookup.get((row_id, col, a_val))
        idx_b = lookup.get((row_id, col, b_val))

        # Try normalized column if exact original column not found.
        if idx_a is None:
            for key, idx in lookup.items():
                if key[0] == row_id and normalize_col_name(key[1]) == col_norm and key[2] == a_val:
                    idx_a = idx
                    break
        if idx_b is None:
            for key, idx in lookup.items():
                if key[0] == row_id and normalize_col_name(key[1]) == col_norm and key[2] == b_val:
                    idx_b = idx
                    break

        if idx_a is None or idx_b is None:
            skipped += 1
            continue

        winner = r["winner"]
        if winner == "A":
            pos_idx, neg_idx = idx_a, idx_b
        elif winner == "B":
            pos_idx, neg_idx = idx_b, idx_a
        else:
            continue

        pos = cand_df.loc[pos_idx]
        neg = cand_df.loc[neg_idx]

        weight = cfg.default_sample_weight
        if not disable_column_weights:
            weight *= cfg.column_weight_map.get(col, cfg.column_weight_map.get(col_norm, 1.0))

        # Extra reliability weighting from high-confidence profile / consensus features.
        for boost_col in [
            "candidate_is_trusted_consensus", "candidate_is_rayyan_metadata_consensus",
            "candidate_entity_safe_fill", "candidate_position_profile_strong",
            "candidate_has_high_conf_profile_majority", "is_from_time_consensus",
            "is_from_entity_pair_majority", "is_from_profile_key_majority",
        ]:
            if boost_col in cand_df.columns:
                if safe_float(pos.get(boost_col), 0.0) > safe_float(neg.get(boost_col), 0.0):
                    weight *= 1.15

        row = {
            "row_id": row_id,
            "column": col,
            "candidate_pos_value": pos["candidate_value"],
            "candidate_neg_value": neg["candidate_value"],
            "winner": winner,
            "reason_label": r["reason_label"],
            "reason_id": reason_to_id[r["reason_label"]],
            "sample_weight": float(weight),
            "pos_index": int(pos_idx),
            "neg_index": int(neg_idx),
        }

        # Store feature values for inspection. Prefix pos_/neg_.
        for c in feature_cols:
            row[f"pos_{c}"] = safe_float(pos.get(c), 0.0)
            row[f"neg_{c}"] = safe_float(neg.get(c), 0.0)
        rows.append(row)

    if skipped:
        print(f"[WARN] skipped pairwise rows because candidates were not found: {skipped}")

    return pd.DataFrame(rows), reason_to_id


# ============================================================
# 5. Torch dataset and model
# ============================================================

class PairwiseDataset(Dataset):
    def __init__(self, pair_df: pd.DataFrame, feature_cols: List[str]):
        self.feature_cols = feature_cols
        self.x_pos = pair_df[[f"pos_{c}" for c in feature_cols]].astype(float).values.astype(np.float32)
        self.x_neg = pair_df[[f"neg_{c}" for c in feature_cols]].astype(float).values.astype(np.float32)
        self.reason = pair_df["reason_id"].astype(int).values.astype(np.int64)
        self.weight = pair_df["sample_weight"].astype(float).values.astype(np.float32)

    def __len__(self):
        return len(self.x_pos)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.x_pos[idx]),
            torch.from_numpy(self.x_neg[idx]),
            torch.tensor(self.reason[idx], dtype=torch.long),
            torch.tensor(self.weight[idx], dtype=torch.float32),
        )


class PairwiseLTRModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float, num_reasons: int):
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
        self.reason_head = nn.Linear(prev, max(num_reasons, 1))

    def encode(self, x):
        return self.encoder(x)

    def score(self, x):
        return self.scorer(self.encode(x)).squeeze(-1)

    def forward(self, x_pos, x_neg):
        h_pos = self.encode(x_pos)
        h_neg = self.encode(x_neg)
        s_pos = self.scorer(h_pos).squeeze(-1)
        s_neg = self.scorer(h_neg).squeeze(-1)
        reason_logits = self.reason_head((h_pos + h_neg) / 2.0)
        return s_pos, s_neg, reason_logits


def split_train_val(df: pd.DataFrame, val_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df, df
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_val = max(1, int(len(df) * val_ratio)) if len(df) >= 10 else max(0, int(len(df) * val_ratio))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    if len(train_idx) == 0:
        train_idx = idx
        val_idx = idx[:0]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


def compute_rule_regularization(model: PairwiseLTRModel, batch_x: torch.Tensor) -> torch.Tensor:
    # Small L2-style score magnitude regularization to avoid overconfident scores.
    scores = model.score(batch_x)
    return torch.mean(scores ** 2)


def run_epoch(
    model: PairwiseLTRModel,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: str,
    reason_loss_weight: float,
    pairwise_loss_weight: float,
    rule_reg_strength: float,
    use_rule_regularization: bool,
    grad_clip_norm: float,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_pair = 0.0
    total_reason = 0.0
    total_acc = 0.0
    total_n = 0

    ce = nn.CrossEntropyLoss(reduction="none")

    for x_pos, x_neg, reason, weight in loader:
        x_pos = x_pos.to(device)
        x_neg = x_neg.to(device)
        reason = reason.to(device)
        weight = weight.to(device)

        if is_train:
            optimizer.zero_grad()

        s_pos, s_neg, reason_logits = model(x_pos, x_neg)
        margin = s_pos - s_neg
        pair_loss = -torch.log(torch.sigmoid(margin) + 1e-8)
        pair_loss = pair_loss * weight

        reason_loss = ce(reason_logits, reason) * weight

        loss = pairwise_loss_weight * pair_loss.mean()
        if reason_loss_weight > 0:
            loss = loss + reason_loss_weight * reason_loss.mean()

        if use_rule_regularization and rule_reg_strength > 0:
            reg = 0.5 * (compute_rule_regularization(model, x_pos) + compute_rule_regularization(model, x_neg))
            loss = loss + rule_reg_strength * reg

        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        n = len(weight)
        total_loss += float(loss.item()) * n
        total_pair += float(pair_loss.mean().item()) * n
        total_reason += float(reason_loss.mean().item()) * n
        total_acc += float((margin.detach() > 0).float().mean().item()) * n
        total_n += n

    if total_n == 0:
        return {"loss": 0.0, "pair_loss": 0.0, "reason_loss": 0.0, "pair_acc": 0.0}

    return {
        "loss": total_loss / total_n,
        "pair_loss": total_pair / total_n,
        "reason_loss": total_reason / total_n,
        "pair_acc": total_acc / total_n,
    }


def save_checkpoint(
    path: str,
    model: PairwiseLTRModel,
    optimizer: torch.optim.Optimizer,
    feature_cols: List[str],
    reason_to_id: Dict[str, int],
    cfg: DatasetConfig,
    args: argparse.Namespace,
    epoch: int,
    val_metrics: Dict[str, float],
):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "feature_columns": feature_cols,
        "reason_to_id": reason_to_id,
        "dataset": cfg.name,
        "hidden_dims": [int(x) for x in args.hidden_dims.split(",") if x.strip()],
        "dropout": args.dropout,
        "epoch": epoch,
        "val_metrics": val_metrics,
    }, path)


# ============================================================
# 6. Main
# ============================================================

def main():
    args = parse_args()
    cfg = DATASET_CONFIGS[args.dataset]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = args.output_dir or cfg.default_output_dir
    ensure_dir(output_dir)

    output_pairwise_train_csv = os.path.join(output_dir, "pairwise_train_materialized.csv")
    output_train_log_csv = os.path.join(output_dir, "training_log.csv")
    output_feature_columns_json = os.path.join(output_dir, "feature_columns.json")
    output_reason_labels_json = os.path.join(output_dir, "reason_label_mapping.json")
    output_column_profiles_json = os.path.join(output_dir, "column_profiles.json")
    output_best_model_path = os.path.join(output_dir, "best_model.pt")
    output_last_model_path = os.path.join(output_dir, "last_model.pt")

    set_seed(args.seed)

    print("========== Unified Pairwise LTR Training ==========")
    print(f"dataset: {cfg.name}")
    print(f"candidate_features: {args.candidate_features}")
    print(f"pairwise_responses: {args.pairwise_responses}")
    print(f"output_dir: {output_dir}")
    print(f"device: {device}")

    cand_df = pd.read_csv(args.candidate_features, encoding="utf-8-sig", low_memory=False)
    cand_df = normalize_candidate_features(cand_df)
    cand_df = add_derived_features(
        cand_df,
        cfg,
        disable_dataset_derived=bool(args.disable_dataset_derived_features),
    )

    profiles = infer_column_profiles(cand_df)
    with open(output_column_profiles_json, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)

    feature_cols = choose_feature_columns(cand_df)
    if not feature_cols:
        raise ValueError("No numeric feature columns found for training.")

    responses = load_jsonl(args.pairwise_responses)
    pair_df, reason_to_id = materialize_pairwise(
        cand_df,
        responses,
        cfg,
        feature_cols,
        disable_column_weights=bool(args.disable_column_weights),
    )

    if pair_df.empty:
        raise ValueError(
            "No pairwise training rows were materialized. "
            "Check pairwise_responses candidate values and candidate_features."
        )

    pair_df.to_csv(output_pairwise_train_csv, index=False, encoding="utf-8-sig")
    with open(output_feature_columns_json, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)
    with open(output_reason_labels_json, "w", encoding="utf-8") as f:
        json.dump(reason_to_id, f, ensure_ascii=False, indent=2)

    train_df, val_df = split_train_val(pair_df, args.val_ratio, args.seed)
    train_ds = PairwiseDataset(train_df, feature_cols)
    val_ds = PairwiseDataset(val_df, feature_cols) if not val_df.empty else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False) if val_ds is not None else None

    hidden_dims = [int(x) for x in str(args.hidden_dims).split(",") if str(x).strip()]
    model = PairwiseLTRModel(
        input_dim=len(feature_cols),
        hidden_dims=hidden_dims,
        dropout=args.dropout,
        num_reasons=len(reason_to_id),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 1
    if args.resume_checkpoint:
        ckpt = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = safe_int(ckpt.get("epoch", 0), 0) + 1
        print(f"[INFO] resumed from {args.resume_checkpoint}, start_epoch={start_epoch}")

    reason_loss_weight = 0.0 if args.disable_reason_head else cfg.reason_loss_weight
    use_rule_reg = not bool(args.disable_rule_regularization)
    rule_reg_strength = cfg.rule_reg_strength

    best_val_loss = float("inf")
    best_epoch = -1
    bad_epochs = 0
    logs = []

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            reason_loss_weight=reason_loss_weight,
            pairwise_loss_weight=cfg.pairwise_loss_weight,
            rule_reg_strength=rule_reg_strength,
            use_rule_regularization=use_rule_reg,
            grad_clip_norm=args.grad_clip_norm,
        )

        if val_loader is not None:
            with torch.no_grad():
                val_metrics = run_epoch(
                    model=model,
                    loader=val_loader,
                    optimizer=None,
                    device=device,
                    reason_loss_weight=reason_loss_weight,
                    pairwise_loss_weight=cfg.pairwise_loss_weight,
                    rule_reg_strength=rule_reg_strength,
                    use_rule_regularization=use_rule_reg,
                    grad_clip_norm=args.grad_clip_norm,
                )
        else:
            val_metrics = dict(train_metrics)

        log_row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "num_train_pairs": len(train_df),
            "num_val_pairs": len(val_df),
            "num_features": len(feature_cols),
        }
        logs.append(log_row)
        pd.DataFrame(logs).to_csv(output_train_log_csv, index=False, encoding="utf-8-sig")

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_metrics['loss']:.6f} "
            f"train_acc={train_metrics['pair_acc']:.4f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_acc={val_metrics['pair_acc']:.4f}"
        )

        save_checkpoint(
            output_last_model_path,
            model,
            optimizer,
            feature_cols,
            reason_to_id,
            cfg,
            args,
            epoch,
            val_metrics,
        )

        if val_metrics["loss"] < best_val_loss - 1e-8:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            bad_epochs = 0
            save_checkpoint(
                output_best_model_path,
                model,
                optimizer,
                feature_cols,
                reason_to_id,
                cfg,
                args,
                epoch,
                val_metrics,
            )
        else:
            bad_epochs += 1

        if bad_epochs >= args.early_stopping_patience:
            print(f"[INFO] early stopping at epoch={epoch}, best_epoch={best_epoch}")
            break

    print("\n========== Training done ==========")
    print(f"best_epoch: {best_epoch}")
    print(f"best_val_loss: {best_val_loss:.6f}")
    print(f"[OK] materialized pairwise: {output_pairwise_train_csv}")
    print(f"[OK] training log: {output_train_log_csv}")
    print(f"[OK] feature columns: {output_feature_columns_json}")
    print(f"[OK] reason mapping: {output_reason_labels_json}")
    print(f"[OK] column profiles: {output_column_profiles_json}")
    print(f"[OK] best model: {output_best_model_path}")
    print(f"[OK] last model: {output_last_model_path}")


if __name__ == "__main__":
    main()

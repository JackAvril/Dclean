#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Soccer pairwise LTR student training.

Complete Soccer version adapted from Adult/Flights pairwise LTR training.

Inputs
------
--candidate_features
    repair_candidate_features_v2.csv or repair_candidate_features_infer_whitelist.csv
    Required columns:
        row_id, column, dirty_value, candidate_value

--pairwise_responses
    repair_pairwise_responses_v2.jsonl or pairwise_responses_accumulated.jsonl
    Required fields:
        row_id, column, candidate_a_value, candidate_b_value, winner

Outputs
-------
model_output/
    best_model.pt
    last_model.pt
    feature_columns.json
    reason_label_mapping.json
    column_profiles.json
    pairwise_train_materialized.csv
    training_log.csv

Usage
-----
python train_ltr.py \
  --candidate_features repair_candidate_features_v2.csv \
  --pairwise_responses repair_pairwise_responses_v2.jsonl \
  --output_dir repair_iterative_workdir_soccer/model_output \
  --epochs 25 \
  --batch_size 256 \
  --lr 1e-3
"""

import argparse
import json
import math
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. Arguments / constants
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Train Soccer pairwise LTR student")
    p.add_argument("--candidate_features", default="repair_candidate_features_v2.csv")
    p.add_argument("--pairwise_responses", default="repair_pairwise_responses_v2.jsonl")
    p.add_argument("--output_dir", default="pairwise_ltr_student_output_soccer")
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
    return p.parse_args()


ARGS = parse_args()

RANDOM_SEED = ARGS.seed
VAL_RATIO = ARGS.val_ratio
BATCH_SIZE = ARGS.batch_size
NUM_EPOCHS = ARGS.epochs
LEARNING_RATE = ARGS.lr
WEIGHT_DECAY = ARGS.weight_decay
GRAD_CLIP_NORM = 5.0
EARLY_STOPPING_PATIENCE = ARGS.early_stopping_patience

HIDDEN_DIMS = [int(x) for x in str(ARGS.hidden_dims).split(",") if str(x).strip()]
DROPOUT = ARGS.dropout

PAIRWISE_LOSS_WEIGHT = 1.0
REASON_LOSS_WEIGHT = 0.20
RULE_REG_STRENGTH = 0.05

COLUMN_WEIGHT_MAP = {
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
}
DEFAULT_SAMPLE_WEIGHT = 1.0

# Revised profile-majority supervision weights. These features are produced by
# revise_candidate.py + build_features.py and are essential for lifting Soccer
# oracle coverage on name/surname/birthplace/position.
PROFILE_MAJOR_COLUMNS = {"name", "surname", "birthplace", "position"}
HIGH_CONF_PROFILE_RATIO = 0.75
VERY_HIGH_CONF_PROFILE_RATIO = 0.85

OUTPUT_DIR = ARGS.output_dir
OUTPUT_PAIRWISE_TRAIN_CSV = os.path.join(OUTPUT_DIR, "pairwise_train_materialized.csv")
OUTPUT_TRAIN_LOG_CSV = os.path.join(OUTPUT_DIR, "training_log.csv")
OUTPUT_FEATURE_COLUMNS_JSON = os.path.join(OUTPUT_DIR, "feature_columns.json")
OUTPUT_REASON_LABELS_JSON = os.path.join(OUTPUT_DIR, "reason_label_mapping.json")
OUTPUT_COLUMN_PROFILES_JSON = os.path.join(OUTPUT_DIR, "column_profiles.json")
OUTPUT_BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pt")
OUTPUT_LAST_MODEL_PATH = os.path.join(OUTPUT_DIR, "last_model.pt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMPTY_TOKENS = {"", "empty", "null", "none", "nan", "na", "n/a", "missing", "?", "{null}"}
POSITION_DOMAIN_VALUES = {"Goalkeeper", "Defender", "Midfield", "Midfielder", "Forward", "Striker", "Winger"}
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
YEAR_COLUMNS = {"birthyear", "season"}
SOCCER_PRIMARY_COLUMNS = {"position", "birthyear", "season"}
SOCCER_CONTEXT_COLUMNS = {"team", "city", "stadium", "manager", "birthplace"}
SOCCER_TEXT_COLUMNS = {"name", "surname", "team", "city", "stadium", "manager", "birthplace"}


# ============================================================
# 2. Basic helpers
# ============================================================

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
    aliases = {
        "birth_year": "birthyear", "birth year": "birthyear", "birth-year": "birthyear",
        "player_position": "position", "club": "team", "football_club": "team",
        "home_city": "city", "arena": "stadium", "coach": "manager",
        "season_year": "season", "year": "season",
    }
    return aliases.get(c, c)


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


def is_valid_birthyear(x: Any) -> int:
    y = parse_year_like(x)
    return int(y is not None and 1940 <= y <= 2010)


def is_valid_season(x: Any) -> int:
    y = parse_year_like(x)
    return int(y is not None and 1990 <= y <= 2030)


def is_valid_age_at_season(birthyear: Any, season: Any) -> int:
    by = parse_year_like(birthyear)
    sy = parse_year_like(season)
    if by is None or sy is None:
        return 0
    return int(15 <= sy - by <= 48)


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
    for v in POSITION_DOMAIN_VALUES:
        if norm_lower(v) == s:
            return v
    return raw


def is_valid_position(x: Any) -> int:
    return int(canonical_position(x) in POSITION_DOMAIN_VALUES)


def looks_like_code(x: str) -> int:
    s = norm_text(x)
    return int(bool(s) and ((bool(re.search(r"[_\\-]", s)) or bool(re.search(r"\\d", s))) and len(s) <= 80))


def prefix_token(x: str) -> str:
    s = norm_text(x)
    parts = re.split(r"[_\\-]", s)
    return parts[0] if parts and parts[0] else ""


def family_token(x: str) -> str:
    s = norm_text(x)
    parts = re.split(r"[_\\-]", s)
    return parts[1] if len(parts) >= 2 else (parts[0] if parts else "")


def load_jsonl(path: str) -> List[dict]:
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
                print(f"[WARN] Skip unparsable line: {path}, line={line_no}")
    return rows


# ============================================================
# 3. Feature preparation
# ============================================================

NON_FEATURE_COLUMNS = {
    "row_id", "column", "dirty_value", "candidate_value",
    "candidate_rank", "candidate_source", "candidate_sources", "candidate_source_text",
    "extra_info", "neighbor_majority_value", "suggested_correct_value",
    "main_rule_type", "main_usage_role", "semantic_type", "detected_type",
    "candidate_canonical_value", "dirty_canonical_value",
    "soccer_evidence_flags", "allowed_soccer_evidence_flags",
    "row_team", "row_city", "row_stadium", "row_manager", "row_position_canonical",
    "allowed_row_team", "allowed_row_city", "allowed_row_stadium",
    "allowed_row_manager", "allowed_row_position_canonical",
}


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dirty_value"] = out["dirty_value"].map(canonical_empty_text)
    out["candidate_value"] = out["candidate_value"].map(canonical_empty_text)
    out["original_is_empty"] = out["dirty_value"].map(lambda x: int(norm_lower(x) in EMPTY_TOKENS))
    out["candidate_is_empty"] = out["candidate_value"].map(lambda x: int(norm_lower(x) in EMPTY_TOKENS))
    out["candidate_equals_dirty"] = (
        out["candidate_value"].map(norm_lower) == out["dirty_value"].map(norm_lower)
    ).astype(int)

    out["dirty_is_code_like"] = out["dirty_value"].map(looks_like_code)
    out["candidate_is_code_like"] = out["candidate_value"].map(looks_like_code)
    out["prefix_match"] = (out["dirty_value"].map(prefix_token) == out["candidate_value"].map(prefix_token)).astype(int)
    out["family_match"] = (out["dirty_value"].map(family_token) == out["candidate_value"].map(family_token)).astype(int)
    out["empty_to_nonempty"] = ((out["original_is_empty"] == 1) & (out["candidate_is_empty"] == 0)).astype(int)

    out["column_norm"] = out["column"].map(normalize_col_name)
    for col_name in [
        "name", "surname", "birthyear", "birthplace", "position",
        "team", "city", "stadium", "season", "manager"
    ]:
        out[f"is_col_{col_name}"] = (out["column_norm"] == col_name).astype(int)

    out["dirty_year_parsed"] = out["dirty_value"].map(lambda x: parse_year_like(x) if parse_year_like(x) is not None else 0)
    out["candidate_year_parsed"] = out["candidate_value"].map(lambda x: parse_year_like(x) if parse_year_like(x) is not None else 0)

    out["dirty_position_valid_derived"] = out["dirty_value"].map(is_valid_position)
    out["candidate_position_valid_derived"] = out["candidate_value"].map(is_valid_position)
    out["candidate_position_canonical_equals_raw"] = [
        int(norm_lower(canonical_position(v)) == norm_lower(v)) for v in out["candidate_value"]
    ]
    out["dirty_position_canonical_equals_raw"] = [
        int(norm_lower(canonical_position(v)) == norm_lower(v)) for v in out["dirty_value"]
    ]

    out["candidate_birthyear_valid_derived"] = out["candidate_value"].map(is_valid_birthyear)
    out["dirty_birthyear_valid_derived"] = out["dirty_value"].map(is_valid_birthyear)
    out["candidate_season_valid_derived"] = out["candidate_value"].map(is_valid_season)
    out["dirty_season_valid_derived"] = out["dirty_value"].map(is_valid_season)

    if "candidate_in_soccer_domain" not in out.columns:
        out["candidate_in_soccer_domain"] = [
            int(
                (col == "position" and is_valid_position(cand))
                or (col == "birthyear" and is_valid_birthyear(cand))
                or (col == "season" and is_valid_season(cand))
                or (col not in {"position", "birthyear", "season"} and not norm_lower(cand) in EMPTY_TOKENS)
            )
            for col, cand in zip(out["column_norm"], out["candidate_value"])
        ]

    if "dirty_in_soccer_domain" not in out.columns:
        out["dirty_in_soccer_domain"] = [
            int(
                (col == "position" and is_valid_position(dirty))
                or (col == "birthyear" and is_valid_birthyear(dirty))
                or (col == "season" and is_valid_season(dirty))
                or (col not in {"position", "birthyear", "season"} and not norm_lower(dirty) in EMPTY_TOKENS)
            )
            for col, dirty in zip(out["column_norm"], out["dirty_value"])
        ]

    if "candidate_domain_validity_gain" not in out.columns:
        out["candidate_domain_validity_gain"] = out["candidate_in_soccer_domain"] - out["dirty_in_soccer_domain"]

    if "position_canonicalization_gain" not in out.columns:
        out["position_canonicalization_gain"] = [
            int(col == "position" and is_valid_position(cand) and not is_valid_position(dirty))
            for col, cand, dirty in zip(out["column_norm"], out["candidate_value"], out["dirty_value"])
        ]

    if "year_age_consistency_gain" not in out.columns:
        out["year_age_consistency_gain"] = 0.0

    if "soccer_profile_ratio_gain" not in out.columns:
        gains = []
        for _, r in out.iterrows():
            gains.append(max(
                safe_float(r.get("team_season_ratio_gain", 0), 0),
                safe_float(r.get("team_ratio_gain", 0), 0),
                safe_float(r.get("stadium_city_ratio_gain", 0), 0),
            ))
        out["soccer_profile_ratio_gain"] = gains

    return out


def load_candidate_features(path: str) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"row_id", "column", "dirty_value", "candidate_value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"candidate_features 缺少必要字段: {missing}")

    df["row_id"] = pd.to_numeric(df["row_id"], errors="raise").astype(int)
    df["column"] = df["column"].astype(str).str.strip()
    df["candidate_value"] = df["candidate_value"].map(norm_text)
    df["dirty_value"] = df["dirty_value"].map(norm_text)
    df = add_derived_features(df)

    categorical_cols = [
        c for c in [
            "semantic_type", "main_rule_type", "main_usage_role",
            "candidate_source", "candidate_source_text",
            "soccer_conflict_bucket", "confidence_bucket", "candidate_count_bucket",
            "column_norm", "soccer_attribution_bucket", "allowed_soccer_attribution_bucket",
            "allowed_soccer_precision_gate_reason",
        ]
        if c in df.columns
    ]

    df_cat = pd.get_dummies(
        df[categorical_cols].fillna("unknown").astype(str),
        prefix=categorical_cols,
    ) if categorical_cols else pd.DataFrame(index=df.index)

    numeric_feature_df = pd.DataFrame(index=df.index)
    for col in df.columns:
        if col in NON_FEATURE_COLUMNS or col in categorical_cols:
            continue
        numeric_feature_df[col] = pd.to_numeric(df[col], errors="coerce")
    numeric_feature_df = numeric_feature_df.fillna(0.0)

    full_feat_df = pd.concat([numeric_feature_df, df_cat], axis=1)
    for col in full_feat_df.columns:
        full_feat_df[col] = pd.to_numeric(full_feat_df[col], errors="coerce").fillna(0.0)

    feature_columns = full_feat_df.columns.tolist()
    df_out = df.copy()
    for col in feature_columns:
        df_out[col] = full_feat_df[col].values

    return df_out, feature_columns


def profile_majority_strength_from_row(row: pd.Series) -> float:
    """
    Unified strength for newly added Soccer profile-majority candidates.
    It combines candidate-generation flags and confidence ratios so training
    can emphasize LLM preferences involving these candidates.
    """
    return max(
        safe_float(row.get("candidate_has_profile_majority_signal", 0.0), 0.0),
        safe_float(row.get("candidate_has_high_conf_profile_majority", 0.0), 0.0),
        safe_float(row.get("candidate_has_very_high_conf_profile_majority", 0.0), 0.0),
        safe_float(row.get("candidate_entity_safe_fill", 0.0), 0.0),
        safe_float(row.get("candidate_position_profile_strong", 0.0), 0.0),
        safe_float(row.get("candidate_is_profile_or_pair_majority", 0.0), 0.0),
        safe_float(row.get("is_from_profile_key_majority", 0.0), 0.0),
        safe_float(row.get("is_from_entity_pair_majority", 0.0), 0.0),
        safe_float(row.get("is_from_team_season_position", 0.0), 0.0),
        safe_float(row.get("profile_key_majority_ratio", 0.0), 0.0),
        safe_float(row.get("entity_pair_majority_ratio", 0.0), 0.0),
        safe_float(row.get("team_season_position_ratio", 0.0), 0.0),
    )


def soccer_gain_from_row(row: pd.Series) -> float:
    return max(
        safe_float(row.get("position_canonicalization_gain", 0.0), 0.0),
        safe_float(row.get("year_age_consistency_gain", 0.0), 0.0),
        safe_float(row.get("candidate_domain_validity_gain", 0.0), 0.0),
        safe_float(row.get("team_season_ratio_gain", 0.0), 0.0),
        safe_float(row.get("team_ratio_gain", 0.0), 0.0),
        safe_float(row.get("stadium_city_ratio_gain", 0.0), 0.0),
        safe_float(row.get("soccer_profile_ratio_gain", 0.0), 0.0),
        profile_majority_strength_from_row(row),
    )


def build_candidate_index(df: pd.DataFrame, feature_columns: List[str]) -> Dict[Tuple[int, str, str], dict]:
    idx = {}
    for _, row in df.iterrows():
        key = (int(row["row_id"]), str(row["column"]), norm_text(row["candidate_value"]))
        column = normalize_col_name(row["column"])
        sample_weight = float(COLUMN_WEIGHT_MAP.get(column, DEFAULT_SAMPLE_WEIGHT))

        for k, boost in [
            ("is_from_rule", 0.30),
            ("is_from_soccer_direct_rule", 0.45),
            ("is_from_soccer_profile", 0.25),
            ("is_from_domain", 0.20),
            ("is_from_numeric_rule", 0.25),
            ("contains_rule_expected_source", 0.25),
            ("contains_position_source", 0.20),
            ("contains_birthyear_source", 0.20),
            ("contains_season_source", 0.20),
            ("contains_team_season_source", 0.20),

            # New candidate sources from profile-majority candidate generation.
            ("is_from_profile_key_majority", 0.70),
            ("is_from_entity_pair_majority", 0.80),
            ("is_from_team_season_position", 0.75),
            ("contains_profile_key_majority_source", 0.65),
            ("contains_entity_pair_majority_source", 0.75),
            ("contains_team_season_position_source", 0.70),
            ("candidate_entity_safe_fill", 0.80),
            ("candidate_position_profile_strong", 0.80),
            ("candidate_has_high_conf_profile_majority", 0.70),
            ("candidate_has_very_high_conf_profile_majority", 0.90),
            ("candidate_is_name_entity_majority", 0.60),
            ("candidate_is_surname_entity_majority", 0.60),
            ("candidate_is_birthplace_entity_majority", 0.60),
            ("candidate_is_position_profile_majority", 0.60),
        ]:
            sample_weight *= (1.0 + boost * safe_float(row.get(k, 0.0), 0.0))

        profile_strength = profile_majority_strength_from_row(row)
        if column in PROFILE_MAJOR_COLUMNS and profile_strength >= HIGH_CONF_PROFILE_RATIO:
            sample_weight *= 1.60
        if column in PROFILE_MAJOR_COLUMNS and profile_strength >= VERY_HIGH_CONF_PROFILE_RATIO:
            sample_weight *= 1.35

        if safe_float(row.get("is_weak_frequency_only_candidate", 0.0), 0.0) > 0:
            sample_weight *= 0.70
        if safe_float(row.get("candidate_is_pure_weak_global_for_entity", 0.0), 0.0) > 0:
            sample_weight *= 0.45

        idx[key] = {
            "features": row[feature_columns].astype(float).values.astype(np.float32),
            "rule_support": safe_float(row.get("candidate_rule_support_ratio", 0.0), 0.0),
            "soccer_gain": soccer_gain_from_row(row),
            "sample_weight": float(sample_weight),
        }
    return idx


# ============================================================
# 4. Pairwise data
# ============================================================

REASON_LABEL_DEFAULT = "uncertain"


def normalize_winner(x: str) -> str:
    x = str(x).strip()
    return x if x in {"A", "B", "Unknown"} else "Unknown"


def load_pairwise_responses(path: str) -> List[dict]:
    rows = load_jsonl(path)
    cleaned = []
    skipped = 0
    for x in rows:
        if not isinstance(x, dict) or "row_id" not in x or "column" not in x:
            skipped += 1
            continue
        try:
            cleaned.append({
                "row_id": int(x["row_id"]),
                "column": str(x["column"]),
                "dirty_value": norm_text(x.get("dirty_value", "")),
                "candidate_a_value": norm_text(x.get("candidate_a_value", "")),
                "candidate_b_value": norm_text(x.get("candidate_b_value", "")),
                "winner": normalize_winner(x.get("winner", "Unknown")),
                "reason_type": str(x.get("reason_type", REASON_LABEL_DEFAULT)).strip() or REASON_LABEL_DEFAULT,
            })
        except Exception:
            skipped += 1
    print(f"[INFO] Loaded pairwise responses: {len(cleaned)}")
    print(f"[INFO] Skipped bad responses: {skipped}")
    return cleaned


def build_reason_label_map(pairwise_rows: List[dict]) -> Dict[str, int]:
    labels = sorted(set(x["reason_type"] for x in pairwise_rows) | {REASON_LABEL_DEFAULT})
    return {lab: i for i, lab in enumerate(labels)}


def materialize_pairwise_training_table(pairwise_rows: List[dict], cand_index: Dict[Tuple[int, str, str], dict], reason_map: Dict[str, int]) -> pd.DataFrame:
    rows = []
    skipped = 0
    unknown_skipped = 0

    for x in pairwise_rows:
        if x["winner"] == "Unknown":
            unknown_skipped += 1
            continue

        key_a = (x["row_id"], x["column"], norm_text(x["candidate_a_value"]))
        key_b = (x["row_id"], x["column"], norm_text(x["candidate_b_value"]))

        if key_a not in cand_index or key_b not in cand_index:
            skipped += 1
            continue

        rows.append({
            "row_id": x["row_id"],
            "column": x["column"],
            "dirty_value": x["dirty_value"],
            "candidate_a_value": x["candidate_a_value"],
            "candidate_b_value": x["candidate_b_value"],
            "label_a_better": 1 if x["winner"] == "A" else 0,
            "reason_type": x["reason_type"],
            "reason_label": reason_map.get(x["reason_type"], reason_map[REASON_LABEL_DEFAULT]),
            "sample_weight": 0.5 * (cand_index[key_a]["sample_weight"] + cand_index[key_b]["sample_weight"]),
        })

    print(f"[INFO] Materialized pairwise training rows: {len(rows)}")
    print(f"[INFO] Skipped Unknown winner rows: {unknown_skipped}")
    print(f"[INFO] Skipped missing feature rows: {skipped}")
    return pd.DataFrame(rows)


@dataclass
class PairwiseExample:
    xa: np.ndarray
    xb: np.ndarray
    y: int
    reason_label: int
    sample_weight: float
    rule_support_a: float
    rule_support_b: float
    soccer_gain_a: float
    soccer_gain_b: float


class PairwiseDataset(Dataset):
    def __init__(self, materialized_df: pd.DataFrame, cand_index: Dict[Tuple[int, str, str], dict]):
        self.examples = []
        for _, r in materialized_df.iterrows():
            key_a = (int(r["row_id"]), str(r["column"]), norm_text(r["candidate_a_value"]))
            key_b = (int(r["row_id"]), str(r["column"]), norm_text(r["candidate_b_value"]))
            ia = cand_index[key_a]
            ib = cand_index[key_b]
            self.examples.append(PairwiseExample(
                xa=ia["features"],
                xb=ib["features"],
                y=int(r["label_a_better"]),
                reason_label=int(r["reason_label"]),
                sample_weight=float(r["sample_weight"]),
                rule_support_a=float(ia["rule_support"]),
                rule_support_b=float(ib["rule_support"]),
                soccer_gain_a=float(ia["soccer_gain"]),
                soccer_gain_b=float(ib["soccer_gain"]),
            ))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return (
            torch.tensor(ex.xa, dtype=torch.float32),
            torch.tensor(ex.xb, dtype=torch.float32),
            torch.tensor(ex.y, dtype=torch.float32),
            torch.tensor(ex.reason_label, dtype=torch.long),
            torch.tensor(ex.sample_weight, dtype=torch.float32),
            torch.tensor(ex.rule_support_a, dtype=torch.float32),
            torch.tensor(ex.rule_support_b, dtype=torch.float32),
            torch.tensor(ex.soccer_gain_a, dtype=torch.float32),
            torch.tensor(ex.soccer_gain_b, dtype=torch.float32),
        )


class SharedScorerMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float, num_reason_classes: int):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.score_head = nn.Linear(prev, 1)
        self.reason_head = nn.Linear(prev, num_reason_classes)

    def forward_single(self, x):
        h = self.backbone(x)
        return self.score_head(h).squeeze(-1), self.reason_head(h), h

    def forward_pair(self, xa, xb):
        sa, reason_a, _ = self.forward_single(xa)
        sb, reason_b, _ = self.forward_single(xb)
        return sa, sb, reason_a, reason_b


def split_train_val(materialized_df: pd.DataFrame, val_ratio: float, seed: int):
    keys = materialized_df[["row_id", "column"]].drop_duplicates()
    keys = keys.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    n_val = max(1, int(len(keys) * val_ratio)) if len(keys) > 1 else 0
    val_keys = set((int(r["row_id"]), str(r["column"])) for _, r in keys.iloc[:n_val].iterrows())

    is_val = materialized_df.apply(lambda r: (int(r["row_id"]), str(r["column"])) in val_keys, axis=1)
    train_df = materialized_df.loc[~is_val].reset_index(drop=True)
    val_df = materialized_df.loc[is_val].reset_index(drop=True)

    if len(train_df) == 0 and len(materialized_df) > 0:
        train_df = materialized_df.copy()
        val_df = materialized_df.iloc[0:0].copy()

    return train_df, val_df


def collate_batch(batch):
    xa, xb, y, reason, weight, rsa, rsb, sga, sgb = zip(*batch)
    return (
        torch.stack(xa),
        torch.stack(xb),
        torch.stack(y),
        torch.stack(reason),
        torch.stack(weight),
        torch.stack(rsa),
        torch.stack(rsb),
        torch.stack(sga),
        torch.stack(sgb),
    )


def evaluate(model, loader, criterion_bce, criterion_reason):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for xa, xb, y, reason, weight, rsa, rsb, sga, sgb in loader:
            xa, xb, y, reason, weight = xa.to(DEVICE), xb.to(DEVICE), y.to(DEVICE), reason.to(DEVICE), weight.to(DEVICE)
            rsa, rsb, sga, sgb = rsa.to(DEVICE), rsb.to(DEVICE), sga.to(DEVICE), sgb.to(DEVICE)

            sa, sb, reason_a, reason_b = model.forward_pair(xa, xb)
            logits = sa - sb

            bce = criterion_bce(logits, y)
            bce = (bce * weight).mean()

            reason_logits = torch.where(y.unsqueeze(1) >= 0.5, reason_a, reason_b)
            reason_loss = criterion_reason(reason_logits, reason)

            evidence_gap = (rsa - rsb) + 0.7 * (sga - sgb)
            reg = torch.relu(-logits * evidence_gap.sign()) * torch.clamp(evidence_gap.abs(), max=1.0)
            reg_loss = reg.mean()

            loss = PAIRWISE_LOSS_WEIGHT * bce + REASON_LOSS_WEIGHT * reason_loss + RULE_REG_STRENGTH * reg_loss

            pred = (logits >= 0).float()
            correct += int((pred == y).sum().item())
            total += int(y.numel())
            total_loss += float(loss.item()) * int(y.numel())

    return {
        "loss": total_loss / max(total, 1),
        "pairwise_acc": correct / max(total, 1),
        "n": total,
    }


def save_checkpoint(model, optimizer, epoch: int, best_val_loss: float, path: str):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "hidden_dims": HIDDEN_DIMS,
        "dropout": DROPOUT,
    }, path)


def load_checkpoint_if_needed(model, optimizer, path: Optional[str]):
    if not path:
        return 0, float("inf")
    if not os.path.exists(path):
        raise FileNotFoundError(f"resume_checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
    print(f"[INFO] Resumed checkpoint: {path}, start_epoch={start_epoch}, best_val_loss={best_val_loss}")
    return start_epoch, best_val_loss


def infer_column_profiles(df: pd.DataFrame) -> Dict[str, dict]:
    profiles = {}
    for col, g in df.groupby("column"):
        vals = pd.concat([g["dirty_value"], g["candidate_value"]], ignore_index=True).dropna().astype(str)
        vals = vals.map(canonical_empty_text)
        vals = vals[vals != "empty"]

        if len(vals) == 0:
            profiles[str(col)] = {"archetype": "mixed", "avg_len": 0.0, "numeric_ratio": 0.0, "unique_count": 0}
            continue

        col_norm = normalize_col_name(col)
        numeric_ratio = float(vals.map(lambda x: int(parse_year_like(x) is not None)).mean())
        avg_len = float(vals.map(lambda x: len(norm_text(x))).mean())
        unique_count = int(vals.nunique())

        if col_norm in YEAR_COLUMNS or numeric_ratio >= 0.7:
            archetype = "year_numeric"
        elif col_norm == "position":
            archetype = "position_domain"
        elif col_norm in SOCCER_CONTEXT_COLUMNS:
            archetype = "soccer_context_enum"
        elif col_norm in SOCCER_TEXT_COLUMNS:
            archetype = "soccer_text"
        else:
            archetype = "mixed"

        profiles[str(col)] = {
            "archetype": archetype,
            "avg_len": avg_len,
            "numeric_ratio": numeric_ratio,
            "unique_count": unique_count,
            "sample_values": list(vals.head(20)),
        }
    return profiles


def train_model():
    set_seed(RANDOM_SEED)
    ensure_dir(OUTPUT_DIR)

    cand_df, feature_columns = load_candidate_features(ARGS.candidate_features)
    cand_index = build_candidate_index(cand_df, feature_columns)

    pairwise_rows = load_pairwise_responses(ARGS.pairwise_responses)
    reason_map = build_reason_label_map(pairwise_rows)

    materialized_df = materialize_pairwise_training_table(pairwise_rows, cand_index, reason_map)
    if len(materialized_df) == 0:
        raise ValueError("No usable pairwise training rows. Check pairwise_responses and candidate_features candidate values.")

    materialized_df.to_csv(OUTPUT_PAIRWISE_TRAIN_CSV, index=False, encoding="utf-8-sig")
    train_df, val_df = split_train_val(materialized_df, VAL_RATIO, RANDOM_SEED)
    print(f"[INFO] train rows={len(train_df)}, val rows={len(val_df)}")

    train_ds = PairwiseDataset(train_df, cand_index)
    val_ds = PairwiseDataset(val_df, cand_index) if len(val_df) > 0 else None

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch) if val_ds else None

    input_dim = len(feature_columns)
    num_reason_classes = len(reason_map)
    model = SharedScorerMLP(input_dim, HIDDEN_DIMS, DROPOUT, num_reason_classes).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion_bce = nn.BCEWithLogitsLoss(reduction="none")
    criterion_reason = nn.CrossEntropyLoss()

    start_epoch, best_val_loss = load_checkpoint_if_needed(model, optimizer, ARGS.resume_checkpoint)

    with open(OUTPUT_FEATURE_COLUMNS_JSON, "w", encoding="utf-8-sig") as f:
        json.dump(feature_columns, f, ensure_ascii=False, indent=2)
    with open(OUTPUT_REASON_LABELS_JSON, "w", encoding="utf-8-sig") as f:
        json.dump(reason_map, f, ensure_ascii=False, indent=2)
    with open(OUTPUT_COLUMN_PROFILES_JSON, "w", encoding="utf-8-sig") as f:
        json.dump(infer_column_profiles(cand_df), f, ensure_ascii=False, indent=2)

    logs = []
    no_improve = 0

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0

        for xa, xb, y, reason, weight, rsa, rsb, sga, sgb in train_loader:
            xa, xb, y, reason, weight = xa.to(DEVICE), xb.to(DEVICE), y.to(DEVICE), reason.to(DEVICE), weight.to(DEVICE)
            rsa, rsb, sga, sgb = rsa.to(DEVICE), rsb.to(DEVICE), sga.to(DEVICE), sgb.to(DEVICE)

            optimizer.zero_grad()
            sa, sb, reason_a, reason_b = model.forward_pair(xa, xb)
            logits = sa - sb

            bce = criterion_bce(logits, y)
            bce = (bce * weight).mean()

            reason_logits = torch.where(y.unsqueeze(1) >= 0.5, reason_a, reason_b)
            reason_loss = criterion_reason(reason_logits, reason)

            evidence_gap = (rsa - rsb) + 0.7 * (sga - sgb)
            reg = torch.relu(-logits * evidence_gap.sign()) * torch.clamp(evidence_gap.abs(), max=1.0)
            reg_loss = reg.mean()

            loss = PAIRWISE_LOSS_WEIGHT * bce + REASON_LOSS_WEIGHT * reason_loss + RULE_REG_STRENGTH * reg_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

            pred = (logits.detach() >= 0).float()
            correct += int((pred == y).sum().item())
            total += int(y.numel())
            total_loss += float(loss.item()) * int(y.numel())

        train_metrics = {"loss": total_loss / max(total, 1), "pairwise_acc": correct / max(total, 1), "n": total}
        val_metrics = evaluate(model, val_loader, criterion_bce, criterion_reason) if val_loader else train_metrics

        log_row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_pairwise_acc": train_metrics["pairwise_acc"],
            "val_loss": val_metrics["loss"],
            "val_pairwise_acc": val_metrics["pairwise_acc"],
            "train_n": train_metrics["n"],
            "val_n": val_metrics["n"],
        }
        logs.append(log_row)
        pd.DataFrame(logs).to_csv(OUTPUT_TRAIN_LOG_CSV, index=False, encoding="utf-8-sig")

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_metrics['loss']:.6f}, train_acc={train_metrics['pairwise_acc']:.4f}, "
            f"val_loss={val_metrics['loss']:.6f}, val_acc={val_metrics['pairwise_acc']:.4f}"
        )

        save_checkpoint(model, optimizer, epoch, best_val_loss, OUTPUT_LAST_MODEL_PATH)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(model, optimizer, epoch, best_val_loss, OUTPUT_BEST_MODEL_PATH)
            no_improve = 0
            print(f"[INFO] New best model saved: {OUTPUT_BEST_MODEL_PATH}")
        else:
            no_improve += 1
            if no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"[INFO] Early stopping at epoch {epoch}")
                break

    print("\n========== Soccer LTR Training Done ==========")
    print(f"feature_dim: {len(feature_columns)}")
    print(f"reason_classes: {len(reason_map)}")
    print(f"best_model: {OUTPUT_BEST_MODEL_PATH}")


if __name__ == "__main__":
    train_model()

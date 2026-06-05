#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adult pairwise LTR student training.

Adapted from the Flights train_ltr.py workflow, but removes flight-time specific
sample weighting and adds Adult-aware derived features.

Inputs
------
--candidate_features
    repair_candidate_features_v2.csv or repair_candidate_features_infer_whitelist.csv.
    Must contain row_id, column, dirty_value, candidate_value.

--pairwise_responses
    repair_pairwise_responses_v2.jsonl or accumulated pairwise jsonl.
    Must contain row_id, column, candidate_a_value, candidate_b_value, winner.

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
  --output_dir pairwise_ltr_student_output_adult \
  --epochs 30 \
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
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. Arguments and constants
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Train / resume Adult pairwise LTR student")
    p.add_argument("--candidate_features", default="repair_candidate_features_v2.csv")
    p.add_argument("--pairwise_responses", default="repair_pairwise_responses_v2.jsonl")
    p.add_argument("--output_dir", default="pairwise_ltr_student_output_adult")
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
REASON_LOSS_WEIGHT = 0.2

USE_RULE_REGULARIZATION = True
RULE_REG_STRENGTH = 0.04

COLUMN_WEIGHT_MAP = {
    # Adult primary rule-sensitive columns
    "relationship": 2.4,
    "sex": 2.2,
    "maritalstatus": 2.1,
    "education": 1.8,
    "age": 1.6,
    "hoursperweek": 1.5,
    "income": 1.5,
    # medium-risk categorical columns
    "workclass": 1.25,
    "occupation": 1.25,
    "country": 1.15,
    "race": 1.10,
}
DEFAULT_SAMPLE_WEIGHT = 1.0

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
    "income": ["LessThan50K", "MoreThan50K"],
}

EDUCATION_ORDER = {
    "Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4,
    "9th": 5, "10th": 6, "11th": 7, "12th": 8,
    "HS-grad": 9, "Some-college": 10, "Assoc-voc": 11, "Assoc-acdm": 12,
    "Bachelors": 13, "Masters": 14, "Prof-school": 15, "Doctorate": 16,
}
MARRIED_STATUSES = {"Married-civ-spouse", "Married-AF-spouse"}
SPOUSE_RELATIONSHIPS = {"Husband", "Wife"}


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

            # Robust recovery if multiple JSON objects were concatenated.
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
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x: Any, default=0) -> int:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return default
        return int(float(x))
    except Exception:
        return default


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
    }
    return aliases.get(c, c)


def canonical_domain_value(column: Any, value: Any) -> str:
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

    domains = ADULT_DOMAIN_VALUES.get(col)
    if domains:
        for v in domains:
            if norm_lower(v) == norm_lower(raw):
                return v
        raw2 = raw.replace("_", "-")
        for v in domains:
            if norm_lower(v) == norm_lower(raw2):
                return v
        raw3 = raw.replace("_", " ")
        for v in domains:
            if norm_lower(v).replace("-", " ") == norm_lower(raw3).replace("-", " "):
                return v
    return raw


def is_empty_token(x: str) -> int:
    return int(norm_lower(x) in EMPTY_TOKENS)


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


def education_order_value(x: Any) -> Optional[int]:
    v = canonical_domain_value("education", x)
    return EDUCATION_ORDER.get(v)


def relationship_consistency_score(rel: str, marital: str, sex: str, age_low=None, age_high=None) -> float:
    rel = canonical_domain_value("relationship", rel)
    marital = canonical_domain_value("maritalstatus", marital)
    sex = canonical_domain_value("sex", sex)
    score = 1.0

    if rel == "Husband":
        if sex not in {"empty", "", "Male"}:
            score -= 0.45
        if marital not in {"empty", ""} and marital not in MARRIED_STATUSES:
            score -= 0.35
    elif rel == "Wife":
        if sex not in {"empty", "", "Female"}:
            score -= 0.45
        if marital not in {"empty", ""} and marital not in MARRIED_STATUSES:
            score -= 0.35
    elif rel == "Own-child":
        if marital in MARRIED_STATUSES:
            score -= 0.25

    if rel in SPOUSE_RELATIONSHIPS and age_high is not None and age_high <= 17:
        score -= 0.35

    return max(0.0, min(1.0, score))


def looks_like_code(x: str) -> int:
    s = norm_text(x)
    return int(bool(s) and ((bool(re.search(r"[_\\-]", s)) or bool(re.search(r"\\d", s))) and len(s) <= 32))


def prefix_token(x: str) -> str:
    s = norm_text(x)
    parts = re.split(r"[_\\-]", s)
    return parts[0] if parts and parts[0] else ""


def family_token(x: str) -> str:
    s = norm_text(x)
    parts = re.split(r"[_\\-]", s)
    return parts[1] if len(parts) >= 2 else (parts[0] if parts else "")


def infer_column_profiles(df: pd.DataFrame) -> Dict[str, dict]:
    profiles = {}
    for col, g in df.groupby("column"):
        vals = pd.concat([g["dirty_value"], g["candidate_value"]], ignore_index=True).dropna().astype(str)
        vals = vals.map(canonical_empty_text)
        vals = vals[vals != "empty"]

        if len(vals) == 0:
            profiles[str(col)] = {
                "archetype": "mixed",
                "avg_len": 0.0,
                "numeric_ratio": 0.0,
                "domain_ratio": 0.0,
                "unique_count": 0,
                "sample_values": [],
            }
            continue

        col_norm = normalize_col_name(col)
        domain = ADULT_DOMAIN_VALUES.get(col_norm, [])
        numeric_ratio = float(vals.map(lambda x: int(parse_numeric_range(x)[0] is not None)).mean())
        domain_ratio = float(vals.map(lambda x: int(canonical_domain_value(col_norm, x) in domain)).mean()) if domain else 0.0
        avg_len = float(vals.map(lambda x: len(norm_text(x))).mean())
        unique_count = int(vals.nunique())

        archetype = "mixed"
        if col_norm in {"age", "hoursperweek", "capital_gain", "capital_loss", "education_num"} or numeric_ratio >= 0.7:
            archetype = "numeric_or_range"
        elif col_norm in ADULT_DOMAIN_VALUES or domain_ratio >= 0.5:
            archetype = "adult_domain_enum"
        elif avg_len >= 25:
            archetype = "long_text"
        elif unique_count <= 12:
            archetype = "small_enum"

        profiles[str(col)] = {
            "archetype": archetype,
            "avg_len": avg_len,
            "numeric_ratio": numeric_ratio,
            "domain_ratio": domain_ratio,
            "unique_count": unique_count,
            "sample_values": list(vals.head(20)),
        }
    return profiles


# ============================================================
# 3. Feature preparation
# ============================================================

NON_FEATURE_COLUMNS = {
    "row_id", "column", "dirty_value", "candidate_value",
    "candidate_rank", "candidate_source", "candidate_sources", "candidate_source_text",
    "extra_info", "neighbor_majority_value", "suggested_correct_value",
    "main_rule_type", "main_usage_role", "semantic_type", "detected_type",
    "candidate_canonical_value", "dirty_canonical_value", "row_income_canonical",
    "adult_evidence_flags", "allowed_adult_evidence_flags",
}


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dirty_value"] = out["dirty_value"].map(canonical_empty_text)
    out["candidate_value"] = out["candidate_value"].map(canonical_empty_text)

    out["original_is_empty"] = out["dirty_value"].map(is_empty_token)
    out["candidate_is_empty"] = out["candidate_value"].map(is_empty_token)
    out["candidate_equals_dirty"] = (
        out["candidate_value"].map(norm_lower) == out["dirty_value"].map(norm_lower)
    ).astype(int)

    out["dirty_is_code_like"] = out["dirty_value"].map(looks_like_code)
    out["candidate_is_code_like"] = out["candidate_value"].map(looks_like_code)
    out["prefix_match"] = (out["dirty_value"].map(prefix_token) == out["candidate_value"].map(prefix_token)).astype(int)
    out["family_match"] = (out["dirty_value"].map(family_token) == out["candidate_value"].map(family_token)).astype(int)
    out["empty_to_nonempty"] = ((out["original_is_empty"] == 1) & (out["candidate_is_empty"] == 0)).astype(int)

    out["column_norm"] = out["column"].map(normalize_col_name)
    for col_name in ["relationship", "sex", "maritalstatus", "education", "age", "hoursperweek", "income", "workclass", "occupation", "country", "race"]:
        out[f"is_col_{col_name}"] = (out["column_norm"] == col_name).astype(int)

    out["candidate_domain_canonical_equals_raw"] = [
        int(norm_lower(canonical_domain_value(col, cand)) == norm_lower(cand))
        for col, cand in zip(out["column"], out["candidate_value"])
    ]
    out["dirty_domain_canonical_equals_raw"] = [
        int(norm_lower(canonical_domain_value(col, dirty)) == norm_lower(dirty))
        for col, dirty in zip(out["column"], out["dirty_value"])
    ]

    # Fill Adult consistency features if missing.
    if "candidate_in_adult_domain" not in out.columns:
        out["candidate_in_adult_domain"] = [
            int(canonical_domain_value(col, cand) in ADULT_DOMAIN_VALUES.get(normalize_col_name(col), []))
            for col, cand in zip(out["column"], out["candidate_value"])
        ]
    if "dirty_in_adult_domain" not in out.columns:
        out["dirty_in_adult_domain"] = [
            int(canonical_domain_value(col, dirty) in ADULT_DOMAIN_VALUES.get(normalize_col_name(col), []))
            for col, dirty in zip(out["column"], out["dirty_value"])
        ]
    if "candidate_domain_validity_gain" not in out.columns:
        out["candidate_domain_validity_gain"] = out["candidate_in_adult_domain"] - out["dirty_in_adult_domain"]

    return out


def load_candidate_features(path: str) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"row_id", "column", "dirty_value", "candidate_value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"candidate_features 缺少必要字段: {missing}")

    df["row_id"] = pd.to_numeric(df["row_id"], errors="raise").astype(int)
    df["column"] = df["column"].astype(str)
    df["candidate_value"] = df["candidate_value"].map(norm_text)
    df["dirty_value"] = df["dirty_value"].map(norm_text)
    df = add_derived_features(df)

    categorical_cols = [
        c for c in [
            "semantic_type", "main_rule_type", "main_usage_role",
            "candidate_source", "candidate_source_text",
            "adult_conflict_bucket", "confidence_bucket", "candidate_count_bucket",
            "column_norm",
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


def build_candidate_index(df: pd.DataFrame, feature_columns: List[str]) -> Dict[Tuple[int, str, str], dict]:
    idx = {}
    for _, row in df.iterrows():
        key = (int(row["row_id"]), str(row["column"]), norm_text(row["candidate_value"]))

        column = normalize_col_name(row["column"])
        sample_weight = float(COLUMN_WEIGHT_MAP.get(column, DEFAULT_SAMPLE_WEIGHT))

        # Strong Adult evidence gets more learning weight.
        for k, boost in [
            ("is_from_rule", 0.30),
            ("is_from_adult_direct_rule", 0.40),
            ("is_from_adult_profile", 0.25),
            ("is_from_domain", 0.15),
            ("contains_rule_expected_source", 0.25),
            ("contains_relationship_source", 0.25),
            ("contains_sex_source", 0.20),
            ("contains_maritalstatus_source", 0.20),
            ("contains_education_source", 0.20),
        ]:
            sample_weight *= (1.0 + boost * safe_float(row.get(k, 0.0), 0.0))

        # Downweight weak frequency-only candidate pairs.
        if safe_float(row.get("is_weak_frequency_only_candidate", 0.0), 0.0) > 0:
            sample_weight *= 0.75

        idx[key] = {
            "features": row[feature_columns].astype(float).values.astype(np.float32),
            "rule_support": safe_float(row.get("candidate_rule_support_ratio", 0.0), 0.0),
            "adult_gain": max(
                safe_float(row.get("relationship_consistency_gain", 0.0), 0.0),
                safe_float(row.get("education_age_conflict_reduction", 0.0), 0.0),
                safe_float(row.get("candidate_domain_validity_gain", 0.0), 0.0),
                safe_float(row.get("same_profile_ratio_gain", 0.0), 0.0),
                safe_float(row.get("work_profile_ratio_gain", 0.0), 0.0),
            ),
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

    for x in pairwise_rows:
        if x["winner"] == "Unknown":
            continue

        key_a = (x["row_id"], x["column"], x["candidate_a_value"])
        key_b = (x["row_id"], x["column"], x["candidate_b_value"])

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
    print(f"[INFO] Skipped pairwise rows due to missing candidate features: {skipped}")
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
    adult_gain_a: float
    adult_gain_b: float


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
                adult_gain_a=float(ia["adult_gain"]),
                adult_gain_b=float(ib["adult_gain"]),
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
            torch.tensor(ex.adult_gain_a, dtype=torch.float32),
            torch.tensor(ex.adult_gain_b, dtype=torch.float32),
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
    rng = np.random.default_rng(seed)
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
    xa, xb, y, reason, weight, rsa, rsb, aga, agb = zip(*batch)
    return (
        torch.stack(xa),
        torch.stack(xb),
        torch.stack(y),
        torch.stack(reason),
        torch.stack(weight),
        torch.stack(rsa),
        torch.stack(rsb),
        torch.stack(aga),
        torch.stack(agb),
    )


def evaluate(model, loader, criterion_bce, criterion_reason):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for xa, xb, y, reason, weight, rsa, rsb, aga, agb in loader:
            xa, xb, y, reason, weight = xa.to(DEVICE), xb.to(DEVICE), y.to(DEVICE), reason.to(DEVICE), weight.to(DEVICE)
            rsa, rsb, aga, agb = rsa.to(DEVICE), rsb.to(DEVICE), aga.to(DEVICE), agb.to(DEVICE)

            sa, sb, reason_a, reason_b = model.forward_pair(xa, xb)
            logits = sa - sb

            bce = criterion_bce(logits, y)
            bce = (bce * weight).mean()

            reason_logits = torch.where(y.unsqueeze(1) >= 0.5, reason_a, reason_b)
            reason_loss = criterion_reason(reason_logits, reason)

            rule_gap = (rsa - rsb) + 0.6 * (aga - agb)
            reg = torch.relu(-logits * rule_gap.sign()) * torch.clamp(rule_gap.abs(), max=1.0)
            reg_loss = reg.mean()

            loss = PAIRWISE_LOSS_WEIGHT * bce + REASON_LOSS_WEIGHT * reason_loss
            if USE_RULE_REGULARIZATION:
                loss = loss + RULE_REG_STRENGTH * reg_loss

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
        "input_dim": model.score_head.in_features if isinstance(model.backbone, nn.Identity) else None,
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

        for xa, xb, y, reason, weight, rsa, rsb, aga, agb in train_loader:
            xa, xb, y, reason, weight = xa.to(DEVICE), xb.to(DEVICE), y.to(DEVICE), reason.to(DEVICE), weight.to(DEVICE)
            rsa, rsb, aga, agb = rsa.to(DEVICE), rsb.to(DEVICE), aga.to(DEVICE), agb.to(DEVICE)

            optimizer.zero_grad()

            sa, sb, reason_a, reason_b = model.forward_pair(xa, xb)
            logits = sa - sb

            bce = criterion_bce(logits, y)
            bce = (bce * weight).mean()

            reason_logits = torch.where(y.unsqueeze(1) >= 0.5, reason_a, reason_b)
            reason_loss = criterion_reason(reason_logits, reason)

            rule_gap = (rsa - rsb) + 0.6 * (aga - agb)
            reg = torch.relu(-logits * rule_gap.sign()) * torch.clamp(rule_gap.abs(), max=1.0)
            reg_loss = reg.mean()

            loss = PAIRWISE_LOSS_WEIGHT * bce + REASON_LOSS_WEIGHT * reason_loss
            if USE_RULE_REGULARIZATION:
                loss = loss + RULE_REG_STRENGTH * reg_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

            pred = (logits.detach() >= 0).float()
            correct += int((pred == y).sum().item())
            total += int(y.numel())
            total_loss += float(loss.item()) * int(y.numel())

        train_metrics = {
            "loss": total_loss / max(total, 1),
            "pairwise_acc": correct / max(total, 1),
            "n": total,
        }

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

        monitor_loss = val_metrics["loss"]
        if monitor_loss < best_val_loss:
            best_val_loss = monitor_loss
            save_checkpoint(model, optimizer, epoch, best_val_loss, OUTPUT_BEST_MODEL_PATH)
            no_improve = 0
            print(f"[INFO] New best model saved: {OUTPUT_BEST_MODEL_PATH}")
        else:
            no_improve += 1
            if no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"[INFO] Early stopping at epoch {epoch}")
                break

    print("\n========== Adult LTR Training Done ==========")
    print(f"feature_dim: {len(feature_columns)}")
    print(f"reason_classes: {len(reason_map)}")
    print(f"best_model: {OUTPUT_BEST_MODEL_PATH}")


if __name__ == "__main__":
    train_model()

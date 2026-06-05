#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adult LTR inference + conservative gate + hard-case mining.

Adapted from the Flights infer_ltr.py workflow, but removes time/flight/hospital
gates and adds Adult-specific gates for:
  - relationship / sex / maritalstatus consistency
  - education-age consistency
  - domain validity
  - age / hours / income normalization
  - profile support with conservative acceptance

Inputs
------
--candidate_features
    repair_candidate_features_infer_whitelist.csv
--allowed_cells_csv
    predicted_error_positions_adult_augmented.csv or predicted_error_positions.csv
--model_dir
    pairwise_ltr_student_output_adult or repair_iterative_workdir_adult/model_output
--checkpoint
    optional model checkpoint

Outputs
-------
inference_all_candidate_scores.csv
inference_all_candidate_ranked.csv
inference_top1_repairs.csv
hard_cases_for_llm.jsonl
"""

import argparse
import json
import os
import re
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# 1. Constants / args
# ============================================================

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

COLUMN_MARGIN_THRESHOLDS = {
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
}
DEFAULT_MARGIN_THRESHOLD = 0.06

PRIMARY_REPAIR_COLUMNS = {
    "relationship", "sex", "maritalstatus", "education", "age", "hoursperweek", "income"
}
RISKY_CONTEXT_ONLY_COLUMNS = {"workclass", "occupation", "country", "race"}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    p = argparse.ArgumentParser(description="Adult LTR inference + hard case mining")
    p.add_argument("--candidate_features", default="repair_candidate_features_infer_whitelist.csv")
    p.add_argument("--allowed_cells_csv", default="predicted_error_positions_adult_augmented.csv")
    p.add_argument("--model_dir", default="pairwise_ltr_student_output_adult")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--global_min_margin", type=float, default=0.06)
    p.add_argument("--joint_topk", type=int, default=4)
    p.add_argument("--max_hard_cases", type=int, default=300)
    return p.parse_args()


ARGS = parse_args()
MODEL_DIR = ARGS.model_dir
CHECKPOINT_PATH = ARGS.checkpoint or os.path.join(MODEL_DIR, "best_model.pt")
FEATURE_COLUMNS_JSON = os.path.join(MODEL_DIR, "feature_columns.json")
REASON_LABELS_JSON = os.path.join(MODEL_DIR, "reason_label_mapping.json")
COLUMN_PROFILES_JSON = os.path.join(MODEL_DIR, "column_profiles.json")

OUTPUT_DIR = os.path.abspath(os.getenv("REPAIR_INFER_OUTPUT_DIR", os.getcwd()))
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_ALL_SCORES_CSV = os.path.join(OUTPUT_DIR, "inference_all_candidate_scores.csv")
OUTPUT_ALL_RANKED_CSV = os.path.join(OUTPUT_DIR, "inference_all_candidate_ranked.csv")
OUTPUT_TOP1_CSV = os.path.join(OUTPUT_DIR, "inference_top1_repairs.csv")
OUTPUT_HARD_CASES_JSONL = os.path.join(OUTPUT_DIR, "hard_cases_for_llm.jsonl")
INPUT_WHITELIST_VIOLATIONS_CSV = os.path.join(OUTPUT_DIR, "input_whitelist_violations.csv")
OUTPUT_WHITELIST_VIOLATIONS_CSV = os.path.join(OUTPUT_DIR, "output_whitelist_violations.csv")


# ============================================================
# 2. Helpers
# ============================================================

def norm_text(x):
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()


def norm_lower(x):
    return norm_text(x).lower()


def canonical_empty_text(x):
    s = norm_lower(x)
    return "empty" if s in EMPTY_TOKENS else norm_text(x)


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def safe_int(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default


def load_json(path: str):
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


def is_empty_token(x):
    return int(norm_lower(x) in EMPTY_TOKENS)


def parse_numeric_range(x) -> Tuple[Optional[float], Optional[float]]:
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
    try:
        v = float(s)
        return v, v
    except Exception:
        return None, None


def education_order_value(x) -> Optional[int]:
    return EDUCATION_ORDER.get(canonical_domain_value("education", x))


def looks_like_code(x: str) -> int:
    s = norm_text(x)
    return int(bool(s) and ((bool(re.search(r"[_\\-]", s)) or bool(re.search(r"\\d", s))) and len(s) <= 32))


def prefix_token(x: str) -> str:
    parts = re.split(r"[_\\-]", norm_text(x))
    return parts[0] if parts and parts[0] else ""


def family_token(x: str) -> str:
    parts = re.split(r"[_\\-]", norm_text(x))
    return parts[1] if len(parts) >= 2 else (parts[0] if parts else "")


def domain_valid(column, value) -> int:
    col = normalize_col_name(column)
    domain = ADULT_DOMAIN_VALUES.get(col)
    if not domain:
        lo, hi = parse_numeric_range(value)
        if col == "age":
            return int(lo is not None and hi is not None and 0 <= lo <= hi <= 120)
        if col == "hoursperweek":
            return int(lo is not None and hi is not None and 0 <= lo <= hi <= 100)
        return 1
    return int(canonical_domain_value(col, value) in domain)


def relationship_consistency_score(rel, marital, sex, age_low=None, age_high=None) -> float:
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


# ============================================================
# 3. Model + features
# ============================================================

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
    out["candidate_equals_dirty"] = (out["candidate_value"].map(norm_lower) == out["dirty_value"].map(norm_lower)).astype(int)
    out["dirty_is_code_like"] = out["dirty_value"].map(looks_like_code)
    out["candidate_is_code_like"] = out["candidate_value"].map(looks_like_code)
    out["prefix_match"] = (out["dirty_value"].map(prefix_token) == out["candidate_value"].map(prefix_token)).astype(int)
    out["family_match"] = (out["dirty_value"].map(family_token) == out["candidate_value"].map(family_token)).astype(int)
    out["empty_to_nonempty"] = ((out["original_is_empty"] == 1) & (out["candidate_is_empty"] == 0)).astype(int)
    out["column_norm"] = out["column"].map(normalize_col_name)
    for col_name in ["relationship", "sex", "maritalstatus", "education", "age", "hoursperweek", "income", "workclass", "occupation", "country", "race"]:
        out[f"is_col_{col_name}"] = (out["column_norm"] == col_name).astype(int)
    if "candidate_in_adult_domain" not in out.columns:
        out["candidate_in_adult_domain"] = [domain_valid(c, v) for c, v in zip(out["column"], out["candidate_value"])]
    if "dirty_in_adult_domain" not in out.columns:
        out["dirty_in_adult_domain"] = [domain_valid(c, v) for c, v in zip(out["column"], out["dirty_value"])]
    if "candidate_domain_validity_gain" not in out.columns:
        out["candidate_domain_validity_gain"] = out["candidate_in_adult_domain"] - out["dirty_in_adult_domain"]
    return out


def validate_prefiltered_candidate_features(df: pd.DataFrame, allowed_cells):
    unique_cells = df[["row_id", "column"]].drop_duplicates().copy()
    if allowed_cells is None:
        print(f"[INFO] inference candidate unique cells (no whitelist validation) = {len(unique_cells)}")
        return

    bad_mask = unique_cells.apply(lambda r: (int(r["row_id"]), str(r["column"]).strip()) not in allowed_cells, axis=1)
    bad_count = int(bad_mask.sum())
    print(f"[INFO] prefiltered input whitelist violations = {bad_count}")
    if bad_count > 0:
        unique_cells.loc[bad_mask].to_csv(INPUT_WHITELIST_VIOLATIONS_CSV, index=False, encoding="utf-8-sig")
        raise ValueError(
            "candidate_features is not prefiltered to whitelist. "
            f"Violations saved to: {INPUT_WHITELIST_VIOLATIONS_CSV}"
        )


def load_allowed_cells(path: str):
    if path is None or str(path).strip() == "" or not os.path.exists(path):
        print(f"[WARN] allowed_cells_csv not found or empty: {path}")
        return None
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"row_id", "column"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"allowed_cells_csv must contain columns: {required}")
    df["row_id"] = pd.to_numeric(df["row_id"], errors="raise").astype(int)
    df["column"] = df["column"].astype(str).str.strip()
    allowed = set((int(r["row_id"]), str(r["column"]).strip()) for _, r in df[["row_id", "column"]].drop_duplicates().iterrows())
    print(f"[INFO] allowed whitelist unique cells = {len(allowed)}")
    return allowed


def load_candidate_features_for_inference(path: str, trained_feature_columns: List[str], allowed_cells=None) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["row_id"] = pd.to_numeric(df["row_id"], errors="raise").astype(int)
    df["column"] = df["column"].astype(str).str.strip()
    print(f"[INFO] Candidate rows for inference: {len(df)}")
    print(f"[INFO] inference input unique candidate cells = {df[['row_id', 'column']].drop_duplicates().shape[0]}")

    validate_prefiltered_candidate_features(df, allowed_cells)

    df["candidate_value"] = df["candidate_value"].map(canonical_empty_text)
    df["dirty_value"] = df["dirty_value"].map(canonical_empty_text)
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

    feat_df = pd.concat([numeric_feature_df, df_cat], axis=1)
    for col in feat_df.columns:
        feat_df[col] = pd.to_numeric(feat_df[col], errors="coerce").fillna(0.0)

    for col in trained_feature_columns:
        if col not in feat_df.columns:
            feat_df[col] = 0.0

    feat_df = feat_df[trained_feature_columns].copy()
    for col in trained_feature_columns:
        df[col] = feat_df[col].values
    return df


@torch.no_grad()
def score_all_candidates(model, df: pd.DataFrame, feature_columns: List[str], reason_id_to_name: dict) -> pd.DataFrame:
    model.eval()
    feats = torch.tensor(df[feature_columns].astype(float).values, dtype=torch.float32, device=DEVICE)
    score, reason_logits, _ = model.forward_single(feats)
    reason_pred = reason_logits.argmax(dim=1).detach().cpu().numpy()
    out = df.copy()
    out["student_score"] = score.detach().cpu().numpy()
    out["student_reason_pred_id"] = reason_pred
    out["student_reason_pred"] = [reason_id_to_name.get(int(x), "unknown") for x in reason_pred]
    return out


def rank_candidates(scored_df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = ["row_id", "column", "student_score"]
    ascending = [True, True, False]
    if "candidate_rank" in scored_df.columns:
        sort_cols.append("candidate_rank")
        ascending.append(True)
    ranked = scored_df.sort_values(sort_cols, ascending=ascending).copy()
    ranked["student_rank"] = ranked.groupby(["row_id", "column"]).cumcount() + 1
    return ranked


# ============================================================
# 4. Adult gate / decoding
# ============================================================

def source_text(row: pd.Series) -> str:
    return str(row.get("candidate_source", row.get("candidate_source_text", row.get("candidate_sources", ""))) or "").lower()


def has_source(row: pd.Series, text: str) -> int:
    return int(text.lower() in source_text(row))


def feature_value(row: pd.Series, key: str, default=0.0) -> float:
    return safe_float(row.get(key, default), default)


def rule_support(row: pd.Series) -> float:
    return feature_value(row, "candidate_rule_support_ratio", 0.0)


def adult_profile_support(row: pd.Series) -> float:
    return max(
        feature_value(row, "candidate_same_profile_ratio", 0.0),
        feature_value(row, "candidate_work_profile_ratio", 0.0),
    )


def adult_gain(row: pd.Series) -> float:
    return max(
        feature_value(row, "relationship_consistency_gain", 0.0),
        feature_value(row, "education_age_conflict_reduction", 0.0),
        feature_value(row, "candidate_domain_validity_gain", 0.0),
        feature_value(row, "same_profile_ratio_gain", 0.0),
        feature_value(row, "work_profile_ratio_gain", 0.0),
    )


def candidate_pool(ranked_df: pd.DataFrame, row_id: int, column: str, topk: int) -> pd.DataFrame:
    g = ranked_df[(ranked_df["row_id"] == row_id) & (ranked_df["column"] == column)].copy()
    if g.empty:
        return g
    g = g.sort_values(["student_rank"], ascending=[True]).reset_index(drop=True)
    return g.head(min(topk, len(g))).copy()


def domain_gate(row: pd.Series) -> bool:
    col = normalize_col_name(row.get("column", ""))
    cand = row.get("candidate_value", "")
    return domain_valid(col, cand) == 1


def direct_rule_strength(row: pd.Series) -> float:
    return max(
        feature_value(row, "is_from_adult_direct_rule", 0.0),
        feature_value(row, "contains_rule_expected_source", 0.0),
        feature_value(row, "candidate_matches_any_rule_expected", 0.0),
        feature_value(row, "candidate_matches_any_expected_values_from_rules", 0.0),
        feature_value(row, "candidate_matches_any_strong_rule_expected", 0.0),
    )


def choose_adult_candidate(pool: pd.DataFrame, dirty: str, column: str) -> Optional[pd.Series]:
    if pool.empty:
        return None

    col = normalize_col_name(column)
    legal = []

    for _, row in pool.iterrows():
        cand = canonical_empty_text(row.get("candidate_value", ""))
        if cand == "empty" or norm_lower(cand) == norm_lower(dirty):
            continue

        if not domain_gate(row):
            # Numeric columns can pass if parseable.
            if col not in {"age", "hoursperweek"}:
                continue

        rule = rule_support(row)
        profile = adult_profile_support(row)
        gain = adult_gain(row)
        direct = direct_rule_strength(row)
        source_score = feature_value(row, "source_score", 0.0)
        score = feature_value(row, "student_score", 0.0)
        rank = safe_int(row.get("student_rank", row.get("candidate_rank", 9999)), 9999)

        # Column-specific priorities.
        rel_gain = feature_value(row, "relationship_consistency_gain", 0.0)
        edu_gain = feature_value(row, "education_age_conflict_reduction", 0.0)
        domain_gain = feature_value(row, "candidate_domain_validity_gain", 0.0)

        if col in {"relationship", "sex", "maritalstatus"}:
            key = (
                int(rel_gain > 0),
                rel_gain,
                direct,
                rule,
                profile,
                source_score,
                score,
                -rank,
            )
        elif col == "education":
            key = (
                int(edu_gain > 0),
                edu_gain,
                direct,
                rule,
                profile,
                source_score,
                score,
                -rank,
            )
        elif col in {"age", "hoursperweek", "income"}:
            key = (
                int(domain_gain > 0),
                domain_gain,
                direct,
                rule,
                profile,
                source_score,
                score,
                -rank,
            )
        else:
            key = (
                direct,
                rule,
                int(domain_gain > 0),
                domain_gain,
                profile,
                source_score,
                score,
                -rank,
            )

        legal.append((row, key))

    if not legal:
        return None

    legal.sort(key=lambda x: x[1], reverse=True)
    return legal[0][0]


def pass_adult_gate(column: str, dirty: str, chosen: pd.Series, second: Optional[pd.Series]) -> Tuple[bool, str]:
    col = normalize_col_name(column)
    cand = canonical_empty_text(chosen.get("candidate_value", ""))
    if cand == "empty" or norm_lower(cand) == norm_lower(dirty):
        return False, "adult_keep_original_empty_or_same"

    if not domain_gate(chosen):
        return False, "adult_keep_original_invalid_domain"

    score = feature_value(chosen, "student_score", 0.0)
    second_score = feature_value(second, "student_score", score) if second is not None else score - 999.0
    margin = score - second_score
    min_margin = max(ARGS.global_min_margin, COLUMN_MARGIN_THRESHOLDS.get(col, DEFAULT_MARGIN_THRESHOLD))

    direct = direct_rule_strength(chosen)
    rule = rule_support(chosen)
    gain = adult_gain(chosen)
    profile = adult_profile_support(chosen)
    domain_gain = feature_value(chosen, "candidate_domain_validity_gain", 0.0)
    weak_freq_only = feature_value(chosen, "is_weak_frequency_only_candidate", 0.0)

    # Strong direct rule / expected value.
    if direct > 0 or rule >= 0.50:
        if col in RISKY_CONTEXT_ONLY_COLUMNS and margin < min_margin:
            return False, "adult_keep_original_risky_col_low_margin"
        return True, "adult_rule_or_direct_pass"

    # Relationship consistency fixes.
    if col in {"relationship", "sex", "maritalstatus"}:
        if feature_value(chosen, "relationship_consistency_gain", 0.0) > 0 and margin >= min_margin * 0.6:
            return True, "adult_relationship_consistency_gain_pass"

    # Education-age fixes.
    if col == "education":
        if feature_value(chosen, "education_age_conflict_reduction", 0.0) > 0 and margin >= min_margin * 0.6:
            return True, "adult_education_age_reduction_pass"

    # Domain canonicalization / income normalization.
    if col in {"income", "age", "hoursperweek"}:
        if domain_gain > 0 and margin >= min_margin * 0.5:
            return True, "adult_domain_or_numeric_normalization_pass"

    # Profile-only evidence is weak. Allow only if strong support and margin.
    if profile >= 0.70 and gain > 0 and margin >= max(min_margin, 0.10) and weak_freq_only <= 0:
        return True, "adult_profile_strong_margin_pass"

    # General conservative pass for primary repair columns.
    if col in PRIMARY_REPAIR_COLUMNS:
        if margin >= max(min_margin, 0.12) and gain > 0 and weak_freq_only <= 0:
            return True, "adult_primary_gain_margin_pass"
        return False, "adult_keep_original_primary_low_evidence"

    # Context-only / risky columns stay conservative.
    if col in RISKY_CONTEXT_ONLY_COLUMNS:
        return False, "adult_keep_original_context_only_conservative"

    if margin >= max(min_margin, 0.15) and (rule > 0 or gain > 0.15) and weak_freq_only <= 0:
        return True, "adult_general_margin_pass"

    return False, "adult_keep_original_low_evidence"


def build_top1_with_gate_and_joint_decoding(ranked_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (row_id, column), g in ranked_df.groupby(["row_id", "column"], sort=False):
        g = g.sort_values(["student_rank"]).reset_index(drop=True)
        dirty = canonical_empty_text(g.iloc[0]["dirty_value"])

        pool = g.head(max(ARGS.joint_topk, 1)).copy()
        chosen = choose_adult_candidate(pool, dirty, str(column))

        if chosen is None:
            rows.append({
                "row_id": int(row_id),
                "column": str(column),
                "dirty_value": dirty,
                "repaired_value": dirty,
                "changed": 0,
                "student_score": None,
                "student_margin": 0.0,
                "student_reason_pred": "",
                "gate_pass": 0,
                "gate_reason": "adult_keep_original_no_valid_candidate",
                "candidate_count": len(g),
            })
            continue

        second = None
        if len(g) >= 2:
            # second by student order, excluding chosen candidate value.
            for _, r in g.iterrows():
                if norm_lower(r["candidate_value"]) != norm_lower(chosen["candidate_value"]):
                    second = r
                    break

        passed, gate_reason = pass_adult_gate(str(column), dirty, chosen, second)
        repaired = canonical_empty_text(chosen["candidate_value"]) if passed else dirty
        score = feature_value(chosen, "student_score", 0.0)
        second_score = feature_value(second, "student_score", score) if second is not None else score
        margin = score - second_score if second is not None else score

        rows.append({
            "row_id": int(row_id),
            "column": str(column),
            "dirty_value": dirty,
            "repaired_value": repaired,
            "changed": int(norm_lower(repaired) != norm_lower(dirty)),
            "student_score": score,
            "student_margin": margin,
            "student_reason_pred": chosen.get("student_reason_pred", ""),
            "gate_pass": int(passed),
            "gate_reason": gate_reason,
            "candidate_count": len(g),
            "top_candidate_value": canonical_empty_text(chosen["candidate_value"]),
            "top_candidate_sources": chosen.get("candidate_source_text", chosen.get("candidate_sources", chosen.get("candidate_source", ""))),
            "candidate_rule_support_ratio": rule_support(chosen),
            "adult_gain": adult_gain(chosen),
            "adult_profile_support": adult_profile_support(chosen),
            "relationship_consistency_gain": feature_value(chosen, "relationship_consistency_gain", 0.0),
            "education_age_conflict_reduction": feature_value(chosen, "education_age_conflict_reduction", 0.0),
            "candidate_domain_validity_gain": feature_value(chosen, "candidate_domain_validity_gain", 0.0),
            "candidate_in_adult_domain": feature_value(chosen, "candidate_in_adult_domain", 0.0),
        })

    top1 = pd.DataFrame(rows)

    allowed = load_allowed_cells(ARGS.allowed_cells_csv)
    if allowed is not None:
        bad = top1.apply(lambda r: (int(r["row_id"]), str(r["column"])) not in allowed, axis=1)
        if int(bad.sum()) > 0:
            top1.loc[bad].to_csv(OUTPUT_WHITELIST_VIOLATIONS_CSV, index=False, encoding="utf-8-sig")
            raise ValueError(f"Output repairs include cells outside whitelist: {OUTPUT_WHITELIST_VIOLATIONS_CSV}")

    return top1


def select_hard_cases(ranked_df: pd.DataFrame, top1_df: pd.DataFrame, max_hard_cases: int) -> List[dict]:
    hard = []
    top1_index = {(int(r["row_id"]), str(r["column"])): r for _, r in top1_df.iterrows()}

    for (row_id, column), g in ranked_df.groupby(["row_id", "column"], sort=False):
        g = g.sort_values(["student_rank"]).reset_index(drop=True)
        if len(g) < 2:
            continue

        top = g.iloc[0]
        second = g.iloc[1]
        margin = feature_value(top, "student_score", 0.0) - feature_value(second, "student_score", 0.0)
        top1_row = top1_index.get((int(row_id), str(column)))

        gate_pass = int(top1_row.get("gate_pass", 0)) if top1_row is not None else 0
        gate_reason = str(top1_row.get("gate_reason", "")) if top1_row is not None else ""

        is_hard = (
            margin < max(ARGS.global_min_margin, COLUMN_MARGIN_THRESHOLDS.get(normalize_col_name(column), DEFAULT_MARGIN_THRESHOLD)) * 1.5
            or gate_pass == 0
            or "low_evidence" in gate_reason
        )

        if not is_hard:
            continue

        hard.append({
            "row_id": int(row_id),
            "column": str(column),
            "dirty_value": canonical_empty_text(top["dirty_value"]),
            "candidate_a_value": canonical_empty_text(top["candidate_value"]),
            "candidate_b_value": canonical_empty_text(second["candidate_value"]),
            "candidate_a_score": feature_value(top, "student_score", 0.0),
            "candidate_b_score": feature_value(second, "student_score", 0.0),
            "student_margin": margin,
            "gate_reason": gate_reason,
            "reason": "adult_low_margin_or_gate_rejected",
            "candidate_a_features": {
                "rule_support": rule_support(top),
                "adult_gain": adult_gain(top),
                "profile_support": adult_profile_support(top),
                "source": source_text(top),
            },
            "candidate_b_features": {
                "rule_support": rule_support(second),
                "adult_gain": adult_gain(second),
                "profile_support": adult_profile_support(second),
                "source": source_text(second),
            },
        })

    hard = sorted(hard, key=lambda x: (x["student_margin"], x["row_id"], x["column"]))
    return hard[:max_hard_cases]


def load_model(input_dim: int, num_reason_classes: int) -> SharedScorerMLP:
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    # Hidden dims are stored indirectly. If absent, default to [128, 64].
    hidden_dims = ckpt.get("hidden_dims", [128, 64])
    dropout = float(ckpt.get("dropout", 0.15))
    model = SharedScorerMLP(input_dim, hidden_dims, dropout, num_reason_classes).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def main():
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"checkpoint not found: {CHECKPOINT_PATH}")
    if not os.path.exists(FEATURE_COLUMNS_JSON):
        raise FileNotFoundError(f"feature_columns.json not found: {FEATURE_COLUMNS_JSON}")
    if not os.path.exists(REASON_LABELS_JSON):
        raise FileNotFoundError(f"reason_label_mapping.json not found: {REASON_LABELS_JSON}")

    feature_columns = load_json(FEATURE_COLUMNS_JSON)
    reason_map = load_json(REASON_LABELS_JSON)
    reason_id_to_name = {int(v): k for k, v in reason_map.items()}

    allowed_cells = load_allowed_cells(ARGS.allowed_cells_csv)
    cand_df = load_candidate_features_for_inference(ARGS.candidate_features, feature_columns, allowed_cells=allowed_cells)

    model = load_model(len(feature_columns), len(reason_map))

    scored_df = score_all_candidates(model, cand_df, feature_columns, reason_id_to_name)
    ranked_df = rank_candidates(scored_df)

    ranked_df.to_csv(OUTPUT_ALL_RANKED_CSV, index=False, encoding="utf-8-sig")
    scored_df.to_csv(OUTPUT_ALL_SCORES_CSV, index=False, encoding="utf-8-sig")

    top1_df = build_top1_with_gate_and_joint_decoding(ranked_df)
    top1_df.to_csv(OUTPUT_TOP1_CSV, index=False, encoding="utf-8-sig")

    hard_cases = select_hard_cases(ranked_df, top1_df, ARGS.max_hard_cases)
    with open(OUTPUT_HARD_CASES_JSONL, "w", encoding="utf-8") as f:
        for obj in hard_cases:
            write_jsonl_record(f, obj)

    print("\n========== Adult Inference Done ==========")
    print(f"[OK] all scores: {OUTPUT_ALL_SCORES_CSV}")
    print(f"[OK] all ranked: {OUTPUT_ALL_RANKED_CSV}")
    print(f"[OK] top1 repairs: {OUTPUT_TOP1_CSV}")
    print(f"[OK] hard cases: {OUTPUT_HARD_CASES_JSONL}")
    print(f"candidate cells: {ranked_df[['row_id','column']].drop_duplicates().shape[0]}")
    print(f"changed cells: {int(top1_df['changed'].sum())}")
    print(f"hard cases: {len(hard_cases)}")

    if "gate_reason" in top1_df.columns:
        print("\n---- Gate reason Top 30 ----")
        print(top1_df["gate_reason"].value_counts().head(30).to_string())

    if "column" in top1_df.columns:
        print("\n---- Changed by column ----")
        changed = top1_df[top1_df["changed"] == 1]
        print(changed["column"].value_counts().head(30).to_string())


if __name__ == "__main__":
    main()

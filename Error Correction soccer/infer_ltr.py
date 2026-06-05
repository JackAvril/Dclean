#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Soccer LTR inference + conservative gate + hard-case mining.

Complete Soccer version adapted from Adult infer_ltr.py.

Outputs:
  inference_all_candidate_scores.csv
  inference_all_candidate_ranked.csv
  inference_top1_repairs.csv
  hard_cases_for_llm.jsonl
"""

import argparse
import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# 1. Constants / args
# ============================================================

EMPTY_TOKENS = {"", "empty", "null", "none", "nan", "na", "n/a", "missing", "?", "{null}", "none/null"}

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
PRIMARY_REPAIR_COLUMNS = {"position", "birthyear", "season"}
CONTEXT_REPAIR_COLUMNS = {"team", "city", "stadium", "manager", "birthplace"}
NAME_COLUMNS = {"name", "surname"}

COLUMN_MARGIN_THRESHOLDS = {
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
}
DEFAULT_MARGIN_THRESHOLD = 0.070

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    p = argparse.ArgumentParser(description="Soccer LTR inference + hard case mining")
    p.add_argument("--candidate_features", default="repair_candidate_features_infer_whitelist.csv")
    p.add_argument("--allowed_cells_csv", default="predicted_error_positions_soccer_augmented.csv")
    p.add_argument("--model_dir", default="pairwise_ltr_student_output_soccer")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--global_min_margin", type=float, default=0.06)
    p.add_argument("--joint_topk", type=int, default=6)
    p.add_argument("--max_hard_cases", type=int, default=300)
    p.add_argument("--allow_profile_only_context_repairs", type=int, default=0)
    p.add_argument("--debug_export_rejected", type=int, default=1)
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
OUTPUT_REJECTED_CSV = os.path.join(OUTPUT_DIR, "inference_rejected_candidates_debug.csv")
INPUT_WHITELIST_VIOLATIONS_CSV = os.path.join(OUTPUT_DIR, "input_whitelist_violations.csv")
OUTPUT_WHITELIST_VIOLATIONS_CSV = os.path.join(OUTPUT_DIR, "output_whitelist_violations.csv")


# ============================================================
# 2. Helpers
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


def load_json(path: str):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_jsonl_record(fp, obj):
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")


def normalize_col_name(column: Any) -> str:
    c = norm_lower(column)
    aliases = {
        "birth_year": "birthyear", "birth year": "birthyear", "birth-year": "birthyear",
        "player_position": "position", "club": "team", "football_club": "team",
        "home_city": "city", "arena": "stadium", "coach": "manager",
        "season_year": "season", "year": "season",
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
    for v in POSITION_DOMAIN_VALUES:
        if norm_lower(v) == s:
            return v
    return raw


def position_domain_valid(x: Any) -> bool:
    return canonical_position(x) in POSITION_DOMAIN_VALUES


def age_at_season_valid(birthyear: Any, season: Any) -> bool:
    by = parse_year_like(birthyear)
    sy = parse_year_like(season)
    if by is None or sy is None:
        return True
    return 15 <= sy - by <= 48


def looks_like_code(x: str) -> int:
    s = norm_text(x)
    return int(bool(s) and ((bool(re.search(r"[_\\-]", s)) or bool(re.search(r"\\d", s))) and len(s) <= 80))


def prefix_token(x: str) -> str:
    parts = re.split(r"[_\\-]", norm_text(x))
    return parts[0] if parts and parts[0] else ""


def family_token(x: str) -> str:
    parts = re.split(r"[_\\-]", norm_text(x))
    return parts[1] if len(parts) >= 2 else (parts[0] if parts else "")


# ============================================================
# 3. Model + feature reconstruction
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
    out["candidate_equals_dirty"] = (out["candidate_value"].map(norm_lower) == out["dirty_value"].map(norm_lower)).astype(int)

    out["dirty_is_code_like"] = out["dirty_value"].map(looks_like_code)
    out["candidate_is_code_like"] = out["candidate_value"].map(looks_like_code)
    out["prefix_match"] = (out["dirty_value"].map(prefix_token) == out["candidate_value"].map(prefix_token)).astype(int)
    out["family_match"] = (out["dirty_value"].map(family_token) == out["candidate_value"].map(family_token)).astype(int)
    out["empty_to_nonempty"] = ((out["original_is_empty"] == 1) & (out["candidate_is_empty"] == 0)).astype(int)

    out["column_norm"] = out["column"].map(normalize_col_name)
    for col_name in ["name", "surname", "birthyear", "birthplace", "position", "team", "city", "stadium", "season", "manager"]:
        out[f"is_col_{col_name}"] = (out["column_norm"] == col_name).astype(int)

    out["dirty_year_parsed"] = out["dirty_value"].map(lambda x: parse_year_like(x) if parse_year_like(x) is not None else 0)
    out["candidate_year_parsed"] = out["candidate_value"].map(lambda x: parse_year_like(x) if parse_year_like(x) is not None else 0)

    out["dirty_position_valid_derived"] = out["dirty_value"].map(lambda x: int(position_domain_valid(x)))
    out["candidate_position_valid_derived"] = out["candidate_value"].map(lambda x: int(position_domain_valid(x)))

    out["candidate_birthyear_valid_derived"] = out["candidate_value"].map(lambda x: int(is_valid_birthyear_value(x)))
    out["dirty_birthyear_valid_derived"] = out["dirty_value"].map(lambda x: int(is_valid_birthyear_value(x)))
    out["candidate_season_valid_derived"] = out["candidate_value"].map(lambda x: int(is_valid_season_value(x)))
    out["dirty_season_valid_derived"] = out["dirty_value"].map(lambda x: int(is_valid_season_value(x)))

    if "candidate_in_soccer_domain" not in out.columns:
        vals = []
        for col, cand in zip(out["column_norm"], out["candidate_value"]):
            vals.append(int(
                (col == "position" and position_domain_valid(cand))
                or (col == "birthyear" and is_valid_birthyear_value(cand))
                or (col == "season" and is_valid_season_value(cand))
                or (col not in {"position", "birthyear", "season"} and not is_empty_like(cand))
            ))
        out["candidate_in_soccer_domain"] = vals

    if "dirty_in_soccer_domain" not in out.columns:
        vals = []
        for col, dirty in zip(out["column_norm"], out["dirty_value"]):
            vals.append(int(
                (col == "position" and position_domain_valid(dirty))
                or (col == "birthyear" and is_valid_birthyear_value(dirty))
                or (col == "season" and is_valid_season_value(dirty))
                or (col not in {"position", "birthyear", "season"} and not is_empty_like(dirty))
            ))
        out["dirty_in_soccer_domain"] = vals

    if "candidate_domain_validity_gain" not in out.columns:
        out["candidate_domain_validity_gain"] = out["candidate_in_soccer_domain"] - out["dirty_in_soccer_domain"]

    if "position_canonicalization_gain" not in out.columns:
        out["position_canonicalization_gain"] = [
            int(col == "position" and position_domain_valid(cand) and not position_domain_valid(dirty))
            for col, cand, dirty in zip(out["column_norm"], out["candidate_value"], out["dirty_value"])
        ]

    if "year_age_consistency_gain" not in out.columns:
        out["year_age_consistency_gain"] = 0.0

    if "soccer_profile_ratio_gain" not in out.columns:
        out["soccer_profile_ratio_gain"] = [
            max(
                safe_float(r.get("team_season_ratio_gain", 0), 0),
                safe_float(r.get("team_ratio_gain", 0), 0),
                safe_float(r.get("stadium_city_ratio_gain", 0), 0),
            )
            for _, r in out.iterrows()
        ]

    return out


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
    allowed = set((int(r["row_id"]), str(r["column"])) for _, r in df[["row_id", "column"]].drop_duplicates().iterrows())
    print(f"[INFO] allowed whitelist unique cells = {len(allowed)}")
    return allowed


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


def load_candidate_features_for_inference(path: str, trained_feature_columns: List[str], allowed_cells=None) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"row_id", "column", "dirty_value", "candidate_value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"candidate_features 缺少必要字段: {missing}")

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
            "soccer_conflict_bucket", "confidence_bucket", "candidate_count_bucket",
            "column_norm", "soccer_attribution_bucket", "allowed_soccer_attribution_bucket",
            "allowed_soccer_precision_gate_reason",
        ]
        if c in df.columns
    ]

    df_cat = pd.get_dummies(df[categorical_cols].fillna("unknown").astype(str), prefix=categorical_cols) if categorical_cols else pd.DataFrame(index=df.index)

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


def load_model(input_dim: int, num_reason_classes: int) -> SharedScorerMLP:
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    hidden_dims = ckpt.get("hidden_dims", [128, 64])
    dropout = float(ckpt.get("dropout", 0.15))
    model = SharedScorerMLP(input_dim, hidden_dims, dropout, num_reason_classes).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


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


# ============================================================
# 4. Soccer candidate choice and gate
# ============================================================

def rule_support(row: Optional[pd.Series]) -> float:
    return feature_value(row, "candidate_rule_support_ratio", 0.0)


def direct_rule_strength(row: Optional[pd.Series]) -> float:
    if row is None:
        return 0.0
    return max(
        feature_value(row, "is_from_soccer_direct_rule", 0.0),
        feature_value(row, "is_from_rule", 0.0),
        feature_value(row, "contains_rule_expected_source", 0.0),
        feature_value(row, "candidate_matches_any_rule_expected", 0.0),
        feature_value(row, "candidate_matches_any_expected_values_from_rules", 0.0),
        feature_value(row, "candidate_matches_any_strong_rule_expected", 0.0),
        rule_support(row),
    )


def soccer_profile_support(row: Optional[pd.Series]) -> float:
    if row is None:
        return 0.0
    return max(
        feature_value(row, "candidate_team_season_ratio", 0.0),
        feature_value(row, "candidate_team_ratio", 0.0),
        feature_value(row, "candidate_stadium_city_ratio", 0.0),
    )


def soccer_gain(row: Optional[pd.Series]) -> float:
    if row is None:
        return 0.0
    return max(
        feature_value(row, "position_canonicalization_gain", 0.0),
        feature_value(row, "year_age_consistency_gain", 0.0),
        feature_value(row, "candidate_domain_validity_gain", 0.0),
        feature_value(row, "team_season_ratio_gain", 0.0),
        feature_value(row, "team_ratio_gain", 0.0),
        feature_value(row, "stadium_city_ratio_gain", 0.0),
        feature_value(row, "soccer_profile_ratio_gain", 0.0),
    )


def domain_gate(column: str, candidate_value: Any) -> bool:
    col = normalize_col_name(column)
    if is_empty_like(candidate_value):
        return False
    if col == "position":
        return position_domain_valid(candidate_value)
    if col == "birthyear":
        return is_valid_birthyear_value(candidate_value)
    if col == "season":
        return is_valid_season_value(candidate_value)
    return True


def year_age_gate(column: str, candidate_value: Any, row: Optional[pd.Series]) -> bool:
    col = normalize_col_name(column)
    if row is None:
        return True

    if col == "birthyear":
        season = row.get("row_season_int", row.get("allowed_row_season_int", row.get("row_season", None)))
        if season is None or str(season).strip() == "":
            return True
        return age_at_season_valid(candidate_value, season)

    if col == "season":
        birthyear = row.get("row_birthyear_int", row.get("allowed_row_birthyear_int", row.get("row_birthyear", None)))
        if birthyear is None or str(birthyear).strip() == "":
            return True
        return age_at_season_valid(birthyear, candidate_value)

    return True


def source_evidence_strength(row: Optional[pd.Series]) -> float:
    if row is None:
        return 0.0
    return max(
        direct_rule_strength(row),
        soccer_profile_support(row),
        soccer_gain(row),
        feature_value(row, "source_score", 0.0),
        feature_value(row, "candidate_match_ratio_in_similar_rows", 0.0),
        feature_value(row, "candidate_column_top_ratio", 0.0),
        profile_majority_ratio(row),
        feature_value(row, "candidate_entity_safe_fill", 0.0),
        feature_value(row, "candidate_position_profile_strong", 0.0),
    )


def source_has_any(row: Optional[pd.Series], tokens: List[str]) -> bool:
    """
    Check candidate source text robustly.
    Feature builder versions may store source in candidate_source,
    candidate_source_text, candidate_sources, or top_candidate_sources.
    """
    if row is None:
        return False
    src_parts = []
    for key in ["candidate_source", "candidate_source_text", "candidate_sources", "top_candidate_sources"]:
        if key in row.index:
            src_parts.append(str(row.get(key, "")))
    src = "|".join(src_parts).lower()
    return any(str(t).lower() in src for t in tokens)


def is_frequency_only_source(row: Optional[pd.Series]) -> bool:
    if row is None:
        return False
    if feature_value(row, "is_weak_frequency_only_candidate", 0.0) > 0:
        return True

    src_parts = []
    for key in ["candidate_source", "candidate_source_text", "candidate_sources", "top_candidate_sources"]:
        if key in row.index:
            src_parts.append(str(row.get(key, "")))
    src = "|".join(src_parts).lower()
    if not src:
        return False

    weak_tokens = [
        "column_top_value",
        "global_column_dictionary",
        "alternative_values_from_column",
    ]
    strong_tokens = [
        "rule_expected",
        "expected_values_from_rules",
        "neighbor_majority",
        "similar_row_target",
        "team_season_distribution",
        "stadium_city_distribution",
        "soccer_profile_team_season_global",
        "soccer_profile_stadium_city_global",
    ]

    has_weak = any(t in src for t in weak_tokens)
    has_strong = any(t in src for t in strong_tokens)
    return bool(has_weak and not has_strong)


def has_strong_context_source(row: Optional[pd.Series]) -> bool:
    """
    Strong evidence for filling Soccer context/text columns.
    This is deliberately source-based because many current numeric features
    such as soccer_profile_support are 0 even when the source text contains
    neighbor/similar/team-season evidence.
    """
    return source_has_any(row, [
        "rule_expected_value",
        "expected_values_from_rules",
        "neighbor_majority",
        "similar_row_target",
        "team_season_distribution",
        "stadium_city_distribution",
        "soccer_profile_team_season_global",
        "soccer_profile_stadium_city_global",
        "soccer_profile_team_global",
        "team_distribution",
        "profile_key_majority",
        "entity_pair_majority",
        "team_season_position",
    ])


def has_profile_source(row: Optional[pd.Series]) -> bool:
    return source_has_any(row, [
        "team_season_distribution",
        "stadium_city_distribution",
        "soccer_profile_team_season_global",
        "soccer_profile_stadium_city_global",
        "soccer_profile_team_global",
        "team_distribution",
        "profile_key_majority",
        "entity_pair_majority",
        "team_season_position",
    ])


def has_neighbor_or_similar_source(row: Optional[pd.Series]) -> bool:
    return source_has_any(row, ["neighbor_majority", "similar_row_target"])


def has_strict_rule_source(row: Optional[pd.Series]) -> bool:
    return source_has_any(row, [
        "rule_expected_value",
        "expected_values_from_rules",
        "rule_reason_parse",
        "strong_rule",
    ]) or direct_rule_strength(row) > 0 or rule_support(row) > 0


def is_dirty_year_directly_normalizable(column: str, dirty: Any) -> bool:
    col = normalize_col_name(column)
    if col not in YEAR_COLUMNS or is_empty_like(dirty):
        return False
    y = parse_year_like(dirty)
    if y is None:
        return False
    if col == "birthyear" and not is_valid_birthyear_value(str(y)):
        return False
    if col == "season" and not is_valid_season_value(str(y)):
        return False
    return canonical_year(dirty) != canonical_empty_text(dirty)


def is_dirty_position_directly_normalizable(dirty: Any) -> bool:
    if is_empty_like(dirty):
        return False
    canon = canonical_position(dirty)
    if not position_domain_valid(canon):
        return False
    return canonical_empty_text(dirty) != canon


def should_directly_normalize_dirty(column: str, dirty: Any) -> Tuple[bool, str, str]:
    """
    Very low-risk direct normalization.
    This prevents the model from changing already recoverable dirty values
    such as 1984X -> 1984 or MidfielD -> Midfield into unrelated candidates.
    """
    col = normalize_col_name(column)
    if col in YEAR_COLUMNS and is_dirty_year_directly_normalizable(col, dirty):
        return True, canonical_year(dirty), "soccer_direct_dirty_year_normalization"
    if col == "position" and is_dirty_position_directly_normalizable(dirty):
        return True, canonical_position(dirty), "soccer_direct_dirty_position_normalization"
    return False, canonical_empty_text(dirty), ""


def context_empty_fill_pass(column: str, dirty: Any, chosen: pd.Series, margin: float, min_margin: float) -> Tuple[bool, str]:
    """
    High-precision recall recovery for missing context fields.
    Current evaluation shows these passes have almost zero FP risk when source is strong.
    """
    col = normalize_col_name(column)
    if not is_empty_like(dirty):
        return False, ""

    if col in {"team", "city", "stadium", "manager"}:
        if has_profile_source(chosen) or has_neighbor_or_similar_source(chosen) or has_strict_rule_source(chosen):
            if not is_frequency_only_source(chosen):
                return True, "soccer_context_empty_strong_profile_or_neighbor_pass"

    if col == "birthplace":
        # birthplace is less structurally constrained than team/stadium/manager,
        # so require neighbor/similar/rule evidence, not pure profile/global frequency.
        if has_neighbor_or_similar_source(chosen) or has_strict_rule_source(chosen):
            if not is_frequency_only_source(chosen):
                return True, "soccer_birthplace_empty_neighbor_or_rule_pass"

    return False, ""


def year_empty_fill_pass(column: str, dirty: Any, chosen: pd.Series, margin: float, min_margin: float) -> Tuple[bool, str]:
    """
    Recall recovery for season missing values.
    Birthyear remains conservative because previous FP mostly came from birthyear.
    """
    col = normalize_col_name(column)
    cand = canonical_empty_text(chosen.get("candidate_value", ""))
    if not is_empty_like(dirty):
        return False, ""

    if col == "season":
        if is_valid_season_value(cand) and year_age_gate(col, cand, chosen):
            if has_profile_source(chosen) or has_neighbor_or_similar_source(chosen) or has_strict_rule_source(chosen):
                if not is_frequency_only_source(chosen):
                    return True, "soccer_season_empty_profile_or_neighbor_pass"

    # Do not add a symmetric birthyear empty-profile pass here.
    # Birthyear is more prone to plausible-but-wrong age-window guesses.
    return False, ""





def profile_majority_ratio(row: Optional[pd.Series]) -> float:
    if row is None:
        return 0.0
    return max(
        feature_value(row, "profile_key_majority_ratio", 0.0),
        feature_value(row, "entity_pair_majority_ratio", 0.0),
        feature_value(row, "team_season_position_ratio", 0.0),
        feature_value(row, "max_profile_key_majority_ratio", 0.0),
        feature_value(row, "max_entity_pair_majority_ratio", 0.0),
        feature_value(row, "max_team_season_position_ratio", 0.0),
    )


def profile_majority_group_size(row: Optional[pd.Series]) -> float:
    if row is None:
        return 0.0
    return max(
        feature_value(row, "profile_key_group_size", 0.0),
        feature_value(row, "entity_pair_group_size", 0.0),
        feature_value(row, "team_season_position_group_size", 0.0),
        feature_value(row, "max_profile_key_group_size", 0.0),
        feature_value(row, "max_entity_pair_group_size", 0.0),
        feature_value(row, "max_team_season_position_group_size", 0.0),
    )


def has_profile_majority_signal(row: Optional[pd.Series]) -> bool:
    if row is None:
        return False
    if max(
        feature_value(row, "is_from_profile_key_majority", 0.0),
        feature_value(row, "is_from_entity_pair_majority", 0.0),
        feature_value(row, "is_from_team_season_position", 0.0),
        feature_value(row, "candidate_has_profile_majority_signal", 0.0),
        feature_value(row, "candidate_is_profile_or_pair_majority", 0.0),
        feature_value(row, "candidate_entity_safe_fill", 0.0),
        feature_value(row, "candidate_position_profile_strong", 0.0),
    ) > 0:
        return True
    return source_has_any(row, [
        "profile_key_majority",
        "entity_pair_majority",
        "team_season_position",
    ])


def is_high_conf_profile_majority(row: Optional[pd.Series], min_ratio: float = 0.70, min_group_size: int = 2) -> bool:
    if row is None:
        return False
    if feature_value(row, "candidate_has_high_conf_profile_majority", 0.0) > 0:
        return True
    if feature_value(row, "candidate_has_very_high_conf_profile_majority", 0.0) > 0:
        return True
    if not has_profile_majority_signal(row):
        return False
    return profile_majority_ratio(row) >= min_ratio and profile_majority_group_size(row) >= min_group_size


def is_entity_safe_fill(row: Optional[pd.Series]) -> bool:
    if row is None:
        return False
    return bool(
        feature_value(row, "candidate_entity_safe_fill", 0.0) > 0
        or feature_value(row, "candidate_is_name_entity_majority", 0.0) > 0
        or feature_value(row, "candidate_is_surname_entity_majority", 0.0) > 0
        or feature_value(row, "candidate_is_birthplace_entity_majority", 0.0) > 0
        or (has_profile_majority_signal(row) and profile_majority_ratio(row) >= 0.75)
    )


def is_position_profile_strong(row: Optional[pd.Series]) -> bool:
    if row is None:
        return False
    return bool(
        feature_value(row, "candidate_position_profile_strong", 0.0) > 0
        or feature_value(row, "candidate_is_position_profile_majority", 0.0) > 0
        or feature_value(row, "is_from_team_season_position", 0.0) > 0
        or (source_has_any(row, ["team_season_position"]) and profile_majority_ratio(row) >= 0.50)
    )

def choose_soccer_candidate(pool: pd.DataFrame, dirty: str, column: str) -> Optional[pd.Series]:
    if pool.empty:
        return None

    col = normalize_col_name(column)
    legal = []

    for _, row in pool.iterrows():
        cand = canonical_empty_text(row.get("candidate_value", ""))
        if is_empty_like(cand) or norm_lower(cand) == norm_lower(dirty):
            continue
        if not domain_gate(col, cand):
            continue
        if not year_age_gate(col, cand, row):
            continue

        direct = direct_rule_strength(row)
        profile = soccer_profile_support(row)
        gain = soccer_gain(row)
        source_score = feature_value(row, "source_score", 0.0)
        score = feature_value(row, "student_score", 0.0)
        rank = safe_int(row.get("student_rank", row.get("candidate_rank", 9999)), 9999)

        position_gain = feature_value(row, "position_canonicalization_gain", 0.0)
        year_gain = feature_value(row, "year_age_consistency_gain", 0.0)
        domain_gain = feature_value(row, "candidate_domain_validity_gain", 0.0)
        team_season_gain = feature_value(row, "team_season_ratio_gain", 0.0)
        team_gain = feature_value(row, "team_ratio_gain", 0.0)
        stadium_city_gain = feature_value(row, "stadium_city_ratio_gain", 0.0)
        pm_ratio = profile_majority_ratio(row)
        pm_group = profile_majority_group_size(row)
        high_conf_pm = int(is_high_conf_profile_majority(row, 0.70, 2))
        entity_safe = int(is_entity_safe_fill(row))
        position_profile = int(is_position_profile_strong(row))
        weak_global_entity = feature_value(row, "candidate_is_pure_weak_global_for_entity", 0.0)

        if col == "position":
            key = (
                position_profile,
                pm_ratio,
                pm_group,
                int(position_gain > 0),
                position_gain,
                direct,
                domain_gain,
                int(has_neighbor_or_similar_source(row)),
                profile,
                source_score,
                score,
                -rank,
            )
        elif col in {"name", "surname", "birthplace"}:
            key = (
                entity_safe,
                high_conf_pm,
                pm_ratio,
                pm_group,
                direct,
                int(has_neighbor_or_similar_source(row)),
                feature_value(row, "edit_similarity_to_dirty", 0.0),
                feature_value(row, "candidate_match_ratio_in_similar_rows", 0.0),
                -weak_global_entity,
                source_score,
                score,
                -rank,
            )
        elif col in {"birthyear", "season"}:
            key = (int(year_gain > 0), year_gain, direct, domain_gain, profile, source_score, score, -rank)
        elif col in CONTEXT_REPAIR_COLUMNS:
            key = (
                int(high_conf_pm),
                pm_ratio,
                direct,
                int(team_season_gain > 0),
                team_season_gain,
                int(stadium_city_gain > 0),
                stadium_city_gain,
                team_gain,
                profile,
                source_score,
                score,
                -rank,
            )
        else:
            key = (direct, gain, profile, source_score, score, -rank)

        legal.append((row, key))

    if not legal:
        return None
    legal.sort(key=lambda x: x[1], reverse=True)
    return legal[0][0]


def pass_soccer_gate(column: str, dirty: str, chosen: pd.Series, second: Optional[pd.Series]) -> Tuple[bool, str]:
    col = normalize_col_name(column)
    cand = canonical_empty_text(chosen.get("candidate_value", ""))

    if is_empty_like(cand) or norm_lower(cand) == norm_lower(dirty):
        return False, "soccer_keep_original_empty_or_same"

    if not domain_gate(col, cand):
        return False, "soccer_keep_original_invalid_domain"

    if not year_age_gate(col, cand, chosen):
        return False, "soccer_keep_original_invalid_age_at_season"

    score = feature_value(chosen, "student_score", 0.0)
    second_score = feature_value(second, "student_score", score) if second is not None else score - 999.0
    margin = score - second_score if second is not None else score
    min_margin = max(ARGS.global_min_margin, COLUMN_MARGIN_THRESHOLDS.get(col, DEFAULT_MARGIN_THRESHOLD))

    direct = direct_rule_strength(chosen)
    profile = soccer_profile_support(chosen)
    gain = soccer_gain(chosen)
    source_score = feature_value(chosen, "source_score", 0.0)
    weak_freq_only = max(
        feature_value(chosen, "is_weak_frequency_only_candidate", 0.0),
        feature_value(chosen, "candidate_is_pure_weak_global_for_entity", 0.0),
    )
    pm_ratio = profile_majority_ratio(chosen)
    pm_group = profile_majority_group_size(chosen)

    # Do not apply a global rule/direct pass before column-specific checks.
    # Year and context columns need stricter, safer conditions.

    if col == "position":
        if is_position_profile_strong(chosen) and pm_ratio >= 0.50 and pm_group >= 2 and not is_frequency_only_source(chosen):
            return True, "soccer_position_profile_majority_pass"
        if has_strict_rule_source(chosen) and not is_frequency_only_source(chosen):
            return True, "soccer_position_strict_rule_pass"
        if feature_value(chosen, "position_canonicalization_gain", 0.0) > 0 and margin >= min_margin * 0.5:
            return True, "soccer_position_canonicalization_pass"
        if feature_value(chosen, "candidate_domain_validity_gain", 0.0) > 0 and has_neighbor_or_similar_source(chosen):
            return True, "soccer_position_domain_neighbor_pass"
        if margin >= max(min_margin, 0.12) and source_score >= 0.75 and weak_freq_only <= 0 and has_strong_context_source(chosen):
            return True, "soccer_position_margin_strong_source_pass"
        return False, "soccer_keep_original_position_low_evidence"

    if col in {"birthyear", "season"}:
        yp, yr = year_empty_fill_pass(col, dirty, chosen, margin, min_margin)
        if yp:
            return True, yr

        dirty_has_valid_year = (
            (col == "birthyear" and is_valid_birthyear_value(dirty))
            or (col == "season" and is_valid_season_value(dirty))
        )

        if dirty_has_valid_year and not is_dirty_year_directly_normalizable(col, dirty):
            # If the dirty year is already valid, do not change it unless there is strict evidence.
            # This protects precision, especially for birthyear.
            if has_strict_rule_source(chosen) and has_neighbor_or_similar_source(chosen) and margin >= min_margin:
                return True, "soccer_year_valid_dirty_strict_rule_neighbor_pass"
            return False, "soccer_keep_original_year_valid_dirty_conservative"

        # Dirty is empty/invalid. Birthyear remains conservative; season can use profile evidence.
        if col == "birthyear":
            if has_strict_rule_source(chosen) and has_neighbor_or_similar_source(chosen):
                return True, "soccer_birthyear_invalid_strict_rule_neighbor_pass"
            if has_neighbor_or_similar_source(chosen) and source_score >= 0.78 and margin >= min_margin:
                return True, "soccer_birthyear_invalid_neighbor_high_score_pass"
            return False, "soccer_keep_original_birthyear_dirty_invalid_but_weak_candidate"

        if col == "season":
            if has_strict_rule_source(chosen) or has_profile_source(chosen) or has_neighbor_or_similar_source(chosen):
                if not is_frequency_only_source(chosen):
                    return True, "soccer_season_invalid_profile_or_neighbor_pass"
            return False, "soccer_keep_original_season_dirty_invalid_but_weak_candidate"

    if col in CONTEXT_REPAIR_COLUMNS:
        cp, cr = context_empty_fill_pass(col, dirty, chosen, margin, min_margin)
        if cp:
            return True, cr
        if is_empty_like(dirty) and is_high_conf_profile_majority(chosen, 0.70, 2) and not is_frequency_only_source(chosen):
            return True, "soccer_context_empty_profile_key_majority_pass"
        if has_strict_rule_source(chosen) and margin >= min_margin * 0.5:
            return True, "soccer_context_rule_pass"
        if feature_value(chosen, "team_season_ratio_gain", 0.0) > 0 and profile >= 0.50 and margin >= min_margin:
            return True, "soccer_team_season_profile_pass"
        if feature_value(chosen, "stadium_city_ratio_gain", 0.0) > 0 and profile >= 0.55 and margin >= min_margin:
            return True, "soccer_stadium_city_profile_pass"
        if int(ARGS.allow_profile_only_context_repairs) == 1 and profile >= 0.80 and margin >= max(min_margin, 0.12):
            return True, "soccer_context_profile_only_pass"
        return False, "soccer_keep_original_context_conservative"

    if col in NAME_COLUMNS:
        if is_empty_like(dirty) and is_entity_safe_fill(chosen) and not is_frequency_only_source(chosen):
            return True, "soccer_name_empty_entity_pair_or_profile_majority_pass"
        if is_empty_like(dirty) and (has_neighbor_or_similar_source(chosen) or has_strict_rule_source(chosen)) and not is_frequency_only_source(chosen):
            return True, "soccer_name_empty_neighbor_or_rule_pass"
        if has_strict_rule_source(chosen) and margin >= min_margin:
            return True, "soccer_name_rule_pass"
        if feature_value(chosen, "edit_similarity_to_dirty", 0.0) >= 0.88 and margin >= min_margin and has_neighbor_or_similar_source(chosen):
            return True, "soccer_name_spelling_similarity_pass"
        return False, "soccer_keep_original_name_conservative"

    if margin >= max(min_margin, 0.12) and gain > 0 and weak_freq_only <= 0:
        return True, "soccer_general_gain_margin_pass"

    return False, "soccer_keep_original_low_evidence"


def build_top1_with_gate_and_joint_decoding(ranked_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    rejected_debug_rows = []

    for (row_id, column), g in ranked_df.groupby(["row_id", "column"], sort=False):
        g = g.sort_values(["student_rank"]).reset_index(drop=True)
        dirty = canonical_empty_text(g.iloc[0]["dirty_value"])
        pool = g.head(max(ARGS.joint_topk, 1)).copy()

        direct_norm_ok, direct_norm_value, direct_norm_reason = should_directly_normalize_dirty(str(column), dirty)
        if direct_norm_ok:
            rows.append({
                "row_id": int(row_id), "column": str(column), "dirty_value": dirty,
                "repaired_value": direct_norm_value, "changed": int(norm_lower(direct_norm_value) != norm_lower(dirty)),
                "student_score": None, "student_margin": 999.0,
                "student_reason_pred": "direct_normalization", "gate_pass": 1,
                "gate_reason": direct_norm_reason,
                "candidate_count": len(g),
                "top_candidate_value": direct_norm_value,
                "top_candidate_sources": "direct_dirty_normalization",
                "candidate_rule_support_ratio": 1.0,
                "soccer_gain": 1.0,
                "soccer_profile_support": 0.0,
                "position_canonicalization_gain": int(normalize_col_name(str(column)) == "position"),
                "year_age_consistency_gain": int(normalize_col_name(str(column)) in YEAR_COLUMNS),
                "candidate_domain_validity_gain": 1.0,
                "team_season_ratio_gain": 0.0,
                "team_ratio_gain": 0.0,
                "stadium_city_ratio_gain": 0.0,
                "source_score": 1.0,
            })
            continue

        chosen = choose_soccer_candidate(pool, dirty, str(column))

        if chosen is None:
            rows.append({
                "row_id": int(row_id), "column": str(column), "dirty_value": dirty,
                "repaired_value": dirty, "changed": 0,
                "student_score": None, "student_margin": 0.0,
                "student_reason_pred": "", "gate_pass": 0,
                "gate_reason": "soccer_keep_original_no_valid_candidate",
                "candidate_count": len(g),
            })
            if int(ARGS.debug_export_rejected) == 1:
                for _, rr in pool.iterrows():
                    tmp = rr.to_dict()
                    tmp["reject_stage"] = "no_valid_candidate"
                    rejected_debug_rows.append(tmp)
            continue

        second = None
        for _, r in g.iterrows():
            if norm_lower(r["candidate_value"]) != norm_lower(chosen["candidate_value"]):
                second = r
                break

        passed, gate_reason = pass_soccer_gate(str(column), dirty, chosen, second)
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
            "soccer_gain": soccer_gain(chosen),
            "soccer_profile_support": soccer_profile_support(chosen),
            "position_canonicalization_gain": feature_value(chosen, "position_canonicalization_gain", 0.0),
            "year_age_consistency_gain": feature_value(chosen, "year_age_consistency_gain", 0.0),
            "candidate_domain_validity_gain": feature_value(chosen, "candidate_domain_validity_gain", 0.0),
            "team_season_ratio_gain": feature_value(chosen, "team_season_ratio_gain", 0.0),
            "team_ratio_gain": feature_value(chosen, "team_ratio_gain", 0.0),
            "stadium_city_ratio_gain": feature_value(chosen, "stadium_city_ratio_gain", 0.0),
            "source_score": feature_value(chosen, "source_score", 0.0),
        })

        if (not passed) and int(ARGS.debug_export_rejected) == 1:
            for _, rr in pool.iterrows():
                tmp = rr.to_dict()
                tmp["reject_stage"] = gate_reason
                tmp["chosen_candidate_value"] = canonical_empty_text(chosen["candidate_value"])
                rejected_debug_rows.append(tmp)

    top1 = pd.DataFrame(rows)
    rejected_df = pd.DataFrame(rejected_debug_rows)

    allowed = load_allowed_cells(ARGS.allowed_cells_csv)
    if allowed is not None:
        bad = top1.apply(lambda r: (int(r["row_id"]), str(r["column"])) not in allowed, axis=1)
        if int(bad.sum()) > 0:
            top1.loc[bad].to_csv(OUTPUT_WHITELIST_VIOLATIONS_CSV, index=False, encoding="utf-8-sig")
            raise ValueError(f"Output repairs include cells outside whitelist: {OUTPUT_WHITELIST_VIOLATIONS_CSV}")

    return top1, rejected_df


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

        threshold = max(ARGS.global_min_margin, COLUMN_MARGIN_THRESHOLDS.get(normalize_col_name(column), DEFAULT_MARGIN_THRESHOLD))
        is_hard = margin < threshold * 1.5 or gate_pass == 0 or "low_evidence" in gate_reason or "conservative" in gate_reason

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
            "reason": "soccer_low_margin_or_gate_rejected",
            "candidate_a_features": {
                "rule_support": rule_support(top),
                "soccer_gain": soccer_gain(top),
                "profile_support": soccer_profile_support(top),
                "source": source_text(top),
                "profile_majority_ratio": profile_majority_ratio(top),
                "profile_majority_group_size": profile_majority_group_size(top),
                "entity_safe_fill": int(is_entity_safe_fill(top)),
                "position_profile_strong": int(is_position_profile_strong(top)),
            },
            "candidate_b_features": {
                "rule_support": rule_support(second),
                "soccer_gain": soccer_gain(second),
                "profile_support": soccer_profile_support(second),
                "source": source_text(second),
                "profile_majority_ratio": profile_majority_ratio(second),
                "profile_majority_group_size": profile_majority_group_size(second),
                "entity_safe_fill": int(is_entity_safe_fill(second)),
                "position_profile_strong": int(is_position_profile_strong(second)),
            },
        })

    hard = sorted(hard, key=lambda x: (x["student_margin"], x["row_id"], x["column"]))
    return hard[:max_hard_cases]


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

    scored_df.to_csv(OUTPUT_ALL_SCORES_CSV, index=False, encoding="utf-8-sig")
    ranked_df.to_csv(OUTPUT_ALL_RANKED_CSV, index=False, encoding="utf-8-sig")

    top1_df, rejected_df = build_top1_with_gate_and_joint_decoding(ranked_df)
    top1_df.to_csv(OUTPUT_TOP1_CSV, index=False, encoding="utf-8-sig")
    if int(ARGS.debug_export_rejected) == 1:
        rejected_df.to_csv(OUTPUT_REJECTED_CSV, index=False, encoding="utf-8-sig")

    hard_cases = select_hard_cases(ranked_df, top1_df, ARGS.max_hard_cases)
    with open(OUTPUT_HARD_CASES_JSONL, "w", encoding="utf-8") as f:
        for obj in hard_cases:
            write_jsonl_record(f, obj)

    print("\n========== Soccer Inference Done ==========")
    print(f"[OK] all scores: {OUTPUT_ALL_SCORES_CSV}")
    print(f"[OK] all ranked: {OUTPUT_ALL_RANKED_CSV}")
    print(f"[OK] top1 repairs: {OUTPUT_TOP1_CSV}")
    print(f"[OK] hard cases: {OUTPUT_HARD_CASES_JSONL}")
    if int(ARGS.debug_export_rejected) == 1:
        print(f"[OK] rejected debug: {OUTPUT_REJECTED_CSV}")
    print(f"candidate cells: {ranked_df[['row_id','column']].drop_duplicates().shape[0]}")
    print(f"changed cells: {int(top1_df['changed'].sum())}")
    print(f"hard cases: {len(hard_cases)}")

    if "gate_reason" in top1_df.columns:
        print("\n---- Gate reason Top 30 ----")
        print(top1_df["gate_reason"].value_counts().head(30).to_string())

    changed = top1_df[top1_df["changed"] == 1]
    if not changed.empty:
        print("\n---- Changed by column ----")
        print(changed["column"].value_counts().head(30).to_string())


if __name__ == "__main__":
    main()

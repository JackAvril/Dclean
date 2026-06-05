#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Soccer LLM pairwise ranking for repair candidates.

Adapted from the Adult LLM pairwise ranking script, but:
1. Uses Soccer candidate feature file by default.
2. Removes Adult-specific relationship/sex/maritalstatus/education-age logic.
3. Adds Soccer-specific sampling features:
   position canonicalization/domain validity,
   birthyear/season age-at-season consistency,
   team/season/stadium/city/manager profile evidence,
   Soccer rule / format / numeric / consistency signals.
4. Adds Soccer-specific pairwise prompt instructions.
5. Keeps clustering/sampling/pairwise/LLM/resume/teacher aggregation workflow.

Inputs
------
repair_candidate_features_infer_whitelist.csv
candidate_llm_contexts_soccer.jsonl

Outputs
-------
repair_cell_summary_for_sampling_v2.csv
repair_cell_clustered_v2.csv
repair_cell_sampled_v2.csv
repair_cell_sampled_with_candidates_v2.jsonl
repair_pairwise_requests_v2.jsonl
repair_pairwise_responses_v2.jsonl
repair_teacher_pairwise_ranking_v2.csv
repair_teacher_top1_labels_v2.csv
repair_bucket_cluster_summary_v2.csv

Usage
-----
cd /mnt/mydata/dq/projects/Splittree/test/Error\ Correction\ soccer

# Only export requests first:
python llm_ranking_soccer.py --dry_run_only_export_requests 1

# Actually call DeepSeek/OpenAI-compatible API:
export DEEPSEEK_API_KEY="你的key"
python llm_ranking_soccer.py --dry_run_only_export_requests 0 --llm_max_workers 4

Notes
-----
Do not hardcode API keys in this file. Use environment variables:
  DEEPSEEK_API_KEY
  DEEPSEEK_BASE_URL
  DEEPSEEK_MODEL
"""

import argparse
import json
import math
import os
import re
import time
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# ============================================================
# 1. Defaults
# ============================================================

DEFAULT_INPUT_CANDIDATE_FEATURES_CSV = "repair_candidate_features_infer_whitelist.csv"
DEFAULT_INPUT_DETAIL_JSONL = "/mnt/mydata/dq/projects/Splittree/test/Error Detection Soccer/candidate_llm_contexts_soccer.jsonl"

DEFAULT_OUTPUT_CELL_SUMMARY_CSV = "repair_cell_summary_for_sampling_v2.csv"
DEFAULT_OUTPUT_CLUSTERED_CSV = "repair_cell_clustered_v2.csv"
DEFAULT_OUTPUT_SAMPLED_CSV = "repair_cell_sampled_v2.csv"
DEFAULT_OUTPUT_SAMPLED_JSONL = "repair_cell_sampled_with_candidates_v2.jsonl"

DEFAULT_OUTPUT_PAIRWISE_REQUESTS_JSONL = "repair_pairwise_requests_v2.jsonl"
DEFAULT_OUTPUT_PAIRWISE_RESPONSES_JSONL = "repair_pairwise_responses_v2.jsonl"
DEFAULT_OUTPUT_TEACHER_RANKING_CSV = "repair_teacher_pairwise_ranking_v2.csv"
DEFAULT_OUTPUT_TEACHER_TOP1_CSV = "repair_teacher_top1_labels_v2.csv"
DEFAULT_OUTPUT_BUCKET_SUMMARY_CSV = "repair_bucket_cluster_summary_v2.csv"

# Soccer sampling defaults.
DEFAULT_USE_COLUMN_IN_BUCKET = True
DEFAULT_MIN_BUCKET_SIZE_FOR_CLUSTER = 6
DEFAULT_MAX_CLUSTERS_PER_BUCKET = 8
DEFAULT_TARGET_CLUSTER_SIZE = 12
DEFAULT_SMALL_BUCKET_FULL_SAMPLE_THRESHOLD = 5

DEFAULT_SAMPLE_CENTER = True
DEFAULT_SAMPLE_BOUNDARY = True
DEFAULT_SAMPLE_OUTLIER = True
DEFAULT_RANDOM_STATE = 42

DEFAULT_MAX_CANDIDATES_PER_CELL_FOR_LLM = 6
HARD_COLUMN_MAX_CANDIDATES = 10

# Soccer hard / semantically risky columns.
SOCCER_HARD_COLUMNS = {
    "name", "surname", "position", "birthyear", "season",
    "team", "city", "stadium", "manager", "birthplace"
}
SOCCER_PRIMARY_REPAIR_COLUMNS = {"position", "birthyear", "season"}
SOCCER_CONTEXT_REPAIR_COLUMNS = {"team", "city", "stadium", "manager", "birthplace"}
SOCCER_TEXT_ID_COLUMNS = {"name", "surname", "team", "city", "stadium", "manager", "birthplace"}

PAIRWISE_STRATEGY = "one_vs_top"   # all_pairs / one_vs_top
UNKNOWN_CANDIDATE_NAME = "Unknown"
DROP_UNKNOWN_IF_TOO_MANY = False

# LLM defaults. No hardcoded key.
MODEL_NAME = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-fd93f5216ae0488f8bf32d9e2a14a14a")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
TEMPERATURE = 0.0
MAX_RETRIES = 3
RETRY_SLEEP_SECONDS = 2.0
LLM_MAX_WORKERS = int(os.getenv("LLM_MAX_WORKERS", "4"))
LLM_PROGRESS_EVERY = int(os.getenv("LLM_PROGRESS_EVERY", "20"))
RESUME_FROM_EXISTING_RESPONSES = True

POSITION_DOMAIN_VALUES = {"Defender", "Forward", "Goalkeeper", "Midfield", "Midfielder", "Striker", "Winger"}


# ============================================================
# 2. Basic tools
# ============================================================

def write_jsonl_record(fp, obj):
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")


def resolve_output_path(path: str) -> str:
    if os.path.isabs(str(path)):
        return str(path)
    return os.path.abspath(str(path))


def ensure_parent(path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def norm_text(x):
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip().lower()


def canonical_text(x):
    s = norm_text(x)
    if s in {"", "nan", "none", "null", "na", "n/a", "missing", "empty", "?", "{null}"}:
        return ""
    return str(x).strip()


def safe_float(x, default=0.0):
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


def safe_int(x, default=0):
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
    c = norm_text(column)
    aliases = {
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
    }
    return aliases.get(c, c)


def bool_from_int(x) -> int:
    return 1 if safe_int(x, 0) != 0 else 0


def parse_year_like(x: Any) -> Optional[int]:
    s = canonical_text(x)
    if not s:
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


def is_valid_birthyear(y: Any) -> int:
    yy = parse_year_like(y)
    return int(yy is not None and 1940 <= yy <= 2010)


def is_valid_season(y: Any) -> int:
    yy = parse_year_like(y)
    return int(yy is not None and 1990 <= yy <= 2030)


def is_valid_age_at_season(birthyear: Any, season: Any) -> int:
    by = parse_year_like(birthyear)
    sy = parse_year_like(season)
    if by is None or sy is None:
        return 0
    age = sy - by
    return int(15 <= age <= 48)


def canonical_position(x: Any) -> str:
    raw = canonical_text(x)
    if not raw:
        return ""
    s = norm_text(raw)
    compact = re.sub(r"[^a-z]", "", s)
    mp = {
        "defender": "Defender",
        "defenders": "Defender",
        "dfender": "Defender",
        "defnder": "Defender",
        "defeder": "Defender",
        "forward": "Forward",
        "forwards": "Forward",
        "foward": "Forward",
        "forwad": "Forward",
        "goalkeeper": "Goalkeeper",
        "goalkeepers": "Goalkeeper",
        "goalkeeper": "Goalkeeper",
        "goalkeeper": "Goalkeeper",
        "goalkeeper": "Goalkeeper",
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
    if s in mp:
        return mp[s]
    if compact in mp:
        return mp[compact]
    for v in POSITION_DOMAIN_VALUES:
        if norm_text(v) == s:
            return v
    return raw


# ============================================================
# 3. Detail jsonl
# ============================================================

def load_detail_jsonl(path: Optional[str]) -> Dict[Tuple[int, str], Dict[str, Any]]:
    detail_map = {}
    if path is None or str(path).strip() == "" or not os.path.exists(path):
        print(f"[WARN] detail jsonl not found: {path}")
        return detail_map

    with open(path, "r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                key = (int(obj["row_id"]), str(obj["column"]))
                detail_map[key] = obj
            except Exception as e:
                print(f"[WARN] 跳过无法解析的 JSONL 行: line={line_no}, err={e}")
                continue
    return detail_map


# ============================================================
# 4. Cell-level summary from candidate-level features
# ============================================================

def first_non_null(series: pd.Series, default=None):
    for x in series:
        if pd.notna(x):
            return x
    return default


def first_existing_non_null(g: pd.DataFrame, cols: List[str], default=None):
    for col in cols:
        if col in g.columns:
            return first_non_null(g[col], default)
    return default


def get_numeric_series(g: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in g.columns:
        return pd.to_numeric(g[col], errors="coerce").fillna(default)
    return pd.Series([default] * len(g), index=g.index)


def get_score_col(g: pd.DataFrame) -> Optional[str]:
    for c in [
        "final_score_from_candidate_gen",
        "final_score",
        "source_score",
        "candidate_score",
    ]:
        if c in g.columns:
            return c
    return None


def sort_candidate_group(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    score_col = get_score_col(g)
    if score_col:
        if "candidate_rank" in g.columns:
            return g.sort_values([score_col, "candidate_rank"], ascending=[False, True])
        return g.sort_values([score_col], ascending=[False])
    if "candidate_rank" in g.columns:
        return g.sort_values(["candidate_rank"], ascending=[True])
    return g


def build_soccer_conflict_bucket(g: pd.DataFrame) -> str:
    parts = []
    checks = [
        ("birthyear_invalid", ["birthyear_value_invalid", "allowed_birthyear_value_invalid"]),
        ("season_invalid", ["season_value_invalid", "allowed_season_value_invalid"]),
        ("position_invalid", ["position_value_invalid", "allowed_position_value_invalid"]),
        ("position_norm", ["position_needs_canonicalization", "allowed_position_needs_canonicalization"]),
        ("age_conflict", ["birthyear_season_age_conflict", "allowed_birthyear_season_age_conflict"]),
        ("team_context", ["team_context_conflict", "allowed_team_context_conflict"]),
        ("format", ["format_rule_signal", "allowed_format_rule_signal"]),
        ("fd_like", ["fd_like_signal", "allowed_fd_like_signal"]),
        ("profile_key", ["is_from_profile_key_majority", "candidate_has_profile_majority_signal"]),
        ("entity_pair", ["is_from_entity_pair_majority", "candidate_entity_safe_fill"]),
        ("team_season_position", ["is_from_team_season_position", "candidate_position_profile_strong"]),
    ]
    for name, cols in checks:
        mx = 0
        for c in cols:
            if c in g.columns:
                mx = max(mx, int(get_numeric_series(g, c, 0).max()))
        if mx:
            parts.append(name)
    return "+".join(parts) if parts else "none"


def build_cell_summary_df(cand_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = cand_df.groupby(["row_id", "column"], sort=False)

    for (row_id, column), g in grouped:
        g = sort_candidate_group(g)
        score_col = get_score_col(g)

        dirty_value = first_non_null(g["dirty_value"], "") if "dirty_value" in g.columns else ""
        semantic_type = first_non_null(g["semantic_type"], "unknown") if "semantic_type" in g.columns else "unknown"
        main_rule_type = first_non_null(g["main_rule_type"], "unknown") if "main_rule_type" in g.columns else "unknown"
        main_usage_role = first_non_null(g["main_usage_role"], "unknown") if "main_usage_role" in g.columns else "unknown"

        violation_count = safe_float(first_non_null(g["violation_count"], 0), 0)
        conflict_score = safe_float(first_non_null(g["conflict_score"], 0), 0)

        candidate_count = len(g)
        top1_score = float(g[score_col].iloc[0]) if score_col and len(g) >= 1 else 0.0
        top2_score = float(g[score_col].iloc[1]) if score_col and len(g) >= 2 else 0.0
        top_margin = top1_score - top2_score if candidate_count >= 2 else top1_score

        rule_support_s = get_numeric_series(g, "candidate_rule_support_ratio", 0.0)
        rule_gain_s = get_numeric_series(g, "rule_support_gain", 0.0)
        bayes_s = get_numeric_series(g, "bayes_mean_cond_prob", 0.0)
        edit_sim_s = get_numeric_series(g, "edit_similarity_to_dirty", 0.0)

        team_season_gain_s = get_numeric_series(g, "team_season_ratio_gain", 0.0)
        team_gain_s = get_numeric_series(g, "team_ratio_gain", 0.0)
        stadium_city_gain_s = get_numeric_series(g, "stadium_city_ratio_gain", 0.0)
        soccer_profile_gain_s = get_numeric_series(g, "soccer_profile_ratio_gain", 0.0)
        domain_gain_s = get_numeric_series(g, "candidate_domain_validity_gain", 0.0)
        position_gain_s = get_numeric_series(g, "position_canonicalization_gain", 0.0)
        year_age_gain_s = get_numeric_series(g, "year_age_consistency_gain", 0.0)

        # New profile-majority signals from revised revise_candidate.py / build_features.py.
        profile_ratio_s = get_numeric_series(g, "profile_key_majority_ratio", 0.0)
        entity_pair_ratio_s = get_numeric_series(g, "entity_pair_majority_ratio", 0.0)
        team_season_position_ratio_s = get_numeric_series(g, "team_season_position_ratio", 0.0)
        profile_group_s = get_numeric_series(g, "profile_key_group_size", 0.0)
        entity_pair_group_s = get_numeric_series(g, "entity_pair_group_size", 0.0)
        team_season_position_group_s = get_numeric_series(g, "team_season_position_group_size", 0.0)

        safe_fill_s = get_numeric_series(g, "candidate_entity_safe_fill", 0.0)
        position_profile_strong_s = get_numeric_series(g, "candidate_position_profile_strong", 0.0)
        high_conf_profile_s = get_numeric_series(g, "candidate_has_high_conf_profile_majority", 0.0)
        very_high_conf_profile_s = get_numeric_series(g, "candidate_has_very_high_conf_profile_majority", 0.0)
        pure_weak_global_entity_s = get_numeric_series(g, "candidate_is_pure_weak_global_for_entity", 0.0)

        source_count_top1 = safe_int(g["source_count"].iloc[0], 0) if "source_count" in g.columns and len(g) >= 1 else 0

        top1_is_from_rule = safe_int(g["is_from_rule"].iloc[0], 0) if "is_from_rule" in g.columns and len(g) >= 1 else 0
        top1_is_from_neighbor = safe_int(g["is_from_neighbor"].iloc[0], 0) if "is_from_neighbor" in g.columns and len(g) >= 1 else 0
        top1_is_from_similarity = safe_int(g["is_from_similarity"].iloc[0], 0) if "is_from_similarity" in g.columns and len(g) >= 1 else 0
        top1_is_from_hint = safe_int(g["is_from_hint"].iloc[0], 0) if "is_from_hint" in g.columns and len(g) >= 1 else 0
        top1_is_from_dictionary = safe_int(g["is_from_dictionary"].iloc[0], 0) if "is_from_dictionary" in g.columns and len(g) >= 1 else 0
        top1_is_from_soccer_profile = safe_int(g["is_from_soccer_profile"].iloc[0], 0) if "is_from_soccer_profile" in g.columns and len(g) >= 1 else 0
        top1_is_from_soccer_direct_rule = safe_int(g["is_from_soccer_direct_rule"].iloc[0], 0) if "is_from_soccer_direct_rule" in g.columns and len(g) >= 1 else 0
        top1_is_from_domain = safe_int(g["is_from_domain"].iloc[0], 0) if "is_from_domain" in g.columns and len(g) >= 1 else 0
        top1_is_from_numeric_rule = safe_int(g["is_from_numeric_rule"].iloc[0], 0) if "is_from_numeric_rule" in g.columns and len(g) >= 1 else 0
        top1_is_from_column_top = safe_int(g["is_from_column_top"].iloc[0], 0) if "is_from_column_top" in g.columns and len(g) >= 1 else 0

        top1_is_from_profile_key_majority = safe_int(g["is_from_profile_key_majority"].iloc[0], 0) if "is_from_profile_key_majority" in g.columns and len(g) >= 1 else 0
        top1_is_from_entity_pair_majority = safe_int(g["is_from_entity_pair_majority"].iloc[0], 0) if "is_from_entity_pair_majority" in g.columns and len(g) >= 1 else 0
        top1_is_from_team_season_position = safe_int(g["is_from_team_season_position"].iloc[0], 0) if "is_from_team_season_position" in g.columns and len(g) >= 1 else 0

        final_pred_error_prob = safe_float(
            first_existing_non_null(g, ["final_pred_error_prob", "allowed_final_pred_error_prob"], 0.0),
            0.0,
        )
        is_high_risk = safe_int(
            first_existing_non_null(g, ["is_high_risk", "high_risk", "allowed_is_high_risk", "allowed_high_risk"], 0),
            0,
        )
        is_hard_case_from_detection = safe_int(
            first_existing_non_null(g, ["is_hard_case", "allowed_is_hard_case"], 0),
            0,
        )

        target_is_primary = safe_int(
            first_existing_non_null(g, ["target_is_primary_column", "allowed_target_is_primary_column"], 0),
            0,
        )
        target_is_context_only = safe_int(
            first_existing_non_null(g, ["target_is_context_only_column", "allowed_target_is_context_only_column"], 0),
            0,
        )

        soccer_consistency_score = safe_float(
            first_existing_non_null(g, ["soccer_consistency_score", "allowed_soccer_consistency_score"], 0.0),
            0.0,
        )
        soccer_evidence_flags = str(first_existing_non_null(g, ["soccer_evidence_flags", "allowed_soccer_evidence_flags"], ""))
        soccer_evidence_flag_count = len([x for x in soccer_evidence_flags.split("|") if str(x).strip()])

        soccer_conflict_bucket = build_soccer_conflict_bucket(g)

        col_norm = normalize_col_name(column)
        is_hard_column = int(col_norm in SOCCER_HARD_COLUMNS or target_is_primary == 1)

        if top_margin >= 0.25:
            confidence_bucket = "high_margin"
        elif top_margin >= 0.10:
            confidence_bucket = "mid_margin"
        else:
            confidence_bucket = "low_margin"

        if candidate_count <= 2:
            candidate_count_bucket = "small"
        elif candidate_count <= 4:
            candidate_count_bucket = "medium"
        else:
            candidate_count_bucket = "large"

        rows.append({
            "row_id": int(row_id),
            "column": str(column),
            "dirty_value": dirty_value,
            "semantic_type": semantic_type,
            "main_rule_type": main_rule_type,
            "main_usage_role": main_usage_role,
            "violation_count": float(violation_count),
            "conflict_score": float(conflict_score),
            "candidate_count": int(candidate_count),
            "top1_score": float(top1_score),
            "top2_score": float(top2_score),
            "top_margin": float(top_margin),

            "avg_rule_support": float(rule_support_s.mean()),
            "max_rule_support": float(rule_support_s.max()),
            "max_rule_support_gain": float(rule_gain_s.max()),
            "avg_bayes": float(bayes_s.mean()),
            "max_bayes": float(bayes_s.max()),
            "avg_edit_sim": float(edit_sim_s.mean()),
            "max_edit_sim": float(edit_sim_s.max()),

            "avg_team_season_gain": float(team_season_gain_s.mean()),
            "max_team_season_gain": float(team_season_gain_s.max()),
            "avg_team_gain": float(team_gain_s.mean()),
            "max_team_gain": float(team_gain_s.max()),
            "avg_stadium_city_gain": float(stadium_city_gain_s.mean()),
            "max_stadium_city_gain": float(stadium_city_gain_s.max()),
            "avg_soccer_profile_gain": float(soccer_profile_gain_s.mean()),
            "max_soccer_profile_gain": float(soccer_profile_gain_s.max()),
            "avg_domain_gain": float(domain_gain_s.mean()),
            "max_domain_gain": float(domain_gain_s.max()),
            "avg_position_gain": float(position_gain_s.mean()),
            "max_position_gain": float(position_gain_s.max()),
            "avg_year_age_gain": float(year_age_gain_s.mean()),
            "max_year_age_gain": float(year_age_gain_s.max()),

            "max_profile_key_majority_ratio": float(profile_ratio_s.max()),
            "max_entity_pair_majority_ratio": float(entity_pair_ratio_s.max()),
            "max_team_season_position_ratio": float(team_season_position_ratio_s.max()),
            "max_profile_key_group_size": float(profile_group_s.max()),
            "max_entity_pair_group_size": float(entity_pair_group_s.max()),
            "max_team_season_position_group_size": float(team_season_position_group_s.max()),
            "has_profile_key_majority_candidate": int(get_numeric_series(g, "is_from_profile_key_majority", 0.0).max() > 0),
            "has_entity_pair_majority_candidate": int(get_numeric_series(g, "is_from_entity_pair_majority", 0.0).max() > 0),
            "has_team_season_position_candidate": int(get_numeric_series(g, "is_from_team_season_position", 0.0).max() > 0),
            "has_entity_safe_fill_candidate": int(safe_fill_s.max() > 0),
            "has_position_profile_strong_candidate": int(position_profile_strong_s.max() > 0),
            "has_high_conf_profile_majority_candidate": int(high_conf_profile_s.max() > 0),
            "has_very_high_conf_profile_majority_candidate": int(very_high_conf_profile_s.max() > 0),
            "has_pure_weak_global_entity_candidate": int(pure_weak_global_entity_s.max() > 0),

            "source_count_top1": int(source_count_top1),
            "top1_is_from_rule": int(top1_is_from_rule),
            "top1_is_from_neighbor": int(top1_is_from_neighbor),
            "top1_is_from_similarity": int(top1_is_from_similarity),
            "top1_is_from_hint": int(top1_is_from_hint),
            "top1_is_from_dictionary": int(top1_is_from_dictionary),
            "top1_is_from_soccer_profile": int(top1_is_from_soccer_profile),
            "top1_is_from_soccer_direct_rule": int(top1_is_from_soccer_direct_rule),
            "top1_is_from_domain": int(top1_is_from_domain),
            "top1_is_from_numeric_rule": int(top1_is_from_numeric_rule),
            "top1_is_from_column_top": int(top1_is_from_column_top),
            "top1_is_from_profile_key_majority": int(top1_is_from_profile_key_majority),
            "top1_is_from_entity_pair_majority": int(top1_is_from_entity_pair_majority),
            "top1_is_from_team_season_position": int(top1_is_from_team_season_position),

            "final_pred_error_prob": float(final_pred_error_prob),
            "is_high_risk": int(is_high_risk),
            "is_hard_case_from_detection": int(is_hard_case_from_detection),

            "soccer_consistency_score": float(soccer_consistency_score),
            "soccer_evidence_flag_count": int(soccer_evidence_flag_count),
            "target_is_primary_column": int(target_is_primary),
            "target_is_context_only_column": int(target_is_context_only),

            "birthyear_value_invalid": int(max(get_numeric_series(g, "birthyear_value_invalid", 0).max(), get_numeric_series(g, "allowed_birthyear_value_invalid", 0).max())),
            "season_value_invalid": int(max(get_numeric_series(g, "season_value_invalid", 0).max(), get_numeric_series(g, "allowed_season_value_invalid", 0).max())),
            "position_value_invalid": int(max(get_numeric_series(g, "position_value_invalid", 0).max(), get_numeric_series(g, "allowed_position_value_invalid", 0).max())),
            "position_needs_canonicalization": int(max(get_numeric_series(g, "position_needs_canonicalization", 0).max(), get_numeric_series(g, "allowed_position_needs_canonicalization", 0).max())),
            "birthyear_season_age_conflict": int(max(get_numeric_series(g, "birthyear_season_age_conflict", 0).max(), get_numeric_series(g, "allowed_birthyear_season_age_conflict", 0).max())),
            "team_context_conflict": int(max(get_numeric_series(g, "team_context_conflict", 0).max(), get_numeric_series(g, "allowed_team_context_conflict", 0).max())),

            "soccer_conflict_bucket": soccer_conflict_bucket,
            "confidence_bucket": confidence_bucket,
            "candidate_count_bucket": candidate_count_bucket,
            "is_hard_column": int(is_hard_column),
        })

    return pd.DataFrame(rows)


# ============================================================
# 5. Buckets
# ============================================================

def get_bucket_keys(use_column_in_bucket: bool) -> List[str]:
    keys = [
        "semantic_type",
        "main_rule_type",
        "main_usage_role",
        "confidence_bucket",
        "candidate_count_bucket",
        "is_hard_column",
        "soccer_conflict_bucket",
        "target_is_primary_column",
    ]
    if use_column_in_bucket:
        keys = ["column"] + keys
    return keys


def build_bucket_id(row: pd.Series, keys: List[str]) -> str:
    parts = []
    for key in keys:
        val = row.get(key, "unknown")
        if pd.isna(val):
            val = "NULL"
        parts.append(f"{key}={val}")
    return " | ".join(parts)


def assign_buckets(df: pd.DataFrame, use_column_in_bucket: bool) -> pd.DataFrame:
    df = df.copy()
    keys = get_bucket_keys(use_column_in_bucket)
    df["bucket_id"] = df.apply(lambda r: build_bucket_id(r, keys), axis=1)
    return df


# ============================================================
# 6. Clustering features
# ============================================================

FEATURE_COLUMNS_NUMERIC = [
    "violation_count",
    "conflict_score",
    "candidate_count",
    "top1_score",
    "top2_score",
    "top_margin",
    "avg_rule_support",
    "max_rule_support",
    "max_rule_support_gain",
    "avg_bayes",
    "max_bayes",
    "avg_edit_sim",
    "max_edit_sim",
    "avg_team_season_gain",
    "max_team_season_gain",
    "avg_team_gain",
    "max_team_gain",
    "avg_stadium_city_gain",
    "max_stadium_city_gain",
    "avg_soccer_profile_gain",
    "max_soccer_profile_gain",
    "avg_domain_gain",
    "max_domain_gain",
    "avg_position_gain",
    "max_position_gain",
    "avg_year_age_gain",
    "max_year_age_gain",

    "max_profile_key_majority_ratio",
    "max_entity_pair_majority_ratio",
    "max_team_season_position_ratio",
    "max_profile_key_group_size",
    "max_entity_pair_group_size",
    "max_team_season_position_group_size",
    "has_profile_key_majority_candidate",
    "has_entity_pair_majority_candidate",
    "has_team_season_position_candidate",
    "has_entity_safe_fill_candidate",
    "has_position_profile_strong_candidate",
    "has_high_conf_profile_majority_candidate",
    "has_very_high_conf_profile_majority_candidate",
    "has_pure_weak_global_entity_candidate",

    "source_count_top1",
    "top1_is_from_rule",
    "top1_is_from_neighbor",
    "top1_is_from_similarity",
    "top1_is_from_hint",
    "top1_is_from_dictionary",
    "top1_is_from_soccer_profile",
    "top1_is_from_soccer_direct_rule",
    "top1_is_from_domain",
    "top1_is_from_numeric_rule",
    "top1_is_from_column_top",
    "top1_is_from_profile_key_majority",
    "top1_is_from_entity_pair_majority",
    "top1_is_from_team_season_position",
    "final_pred_error_prob",
    "is_high_risk",
    "is_hard_case_from_detection",
    "soccer_consistency_score",
    "soccer_evidence_flag_count",
    "target_is_primary_column",
    "target_is_context_only_column",
    "birthyear_value_invalid",
    "season_value_invalid",
    "position_value_invalid",
    "position_needs_canonicalization",
    "birthyear_season_age_conflict",
    "team_context_conflict",
    "is_hard_column",
]

FEATURE_COLUMNS_CATEGORICAL = [
    "semantic_type",
    "main_rule_type",
    "main_usage_role",
    "confidence_bucket",
    "candidate_count_bucket",
    "soccer_conflict_bucket",
]


def prepare_features(bucket_df: pd.DataFrame):
    work = bucket_df.copy()

    for col in FEATURE_COLUMNS_NUMERIC:
        if col not in work.columns:
            work[col] = 0.0
        work[col] = pd.to_numeric(work[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    num_df = work[FEATURE_COLUMNS_NUMERIC].copy()
    cat_frames = []
    for col in FEATURE_COLUMNS_CATEGORICAL:
        if col not in work.columns:
            work[col] = "unknown"
        dummies = pd.get_dummies(work[col].fillna("unknown").astype(str), prefix=col)
        cat_frames.append(dummies)

    feat_df = pd.concat([num_df] + cat_frames, axis=1) if cat_frames else num_df

    scaler = StandardScaler()
    X = scaler.fit_transform(feat_df.values)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, feat_df.columns.tolist()


def choose_k(n: int, min_bucket_size: int, target_cluster_size: int, max_clusters: int) -> int:
    if n < min_bucket_size:
        return 1
    k = max(2, round(n / target_cluster_size))
    k = min(k, max_clusters)
    k = min(k, n)
    return k


def cluster_bucket(bucket_df: pd.DataFrame, args) -> pd.DataFrame:
    bucket_df = bucket_df.copy()
    n = len(bucket_df)

    if n <= 1:
        bucket_df["cluster_id"] = 0
        bucket_df["dist_to_center"] = 0.0
        return bucket_df

    X, _ = prepare_features(bucket_df)
    k = choose_k(n, args.min_bucket_size_for_cluster, args.target_cluster_size, args.max_clusters_per_bucket)

    if k <= 1:
        bucket_df["cluster_id"] = 0
        center = X.mean(axis=0, keepdims=True)
        dist = np.linalg.norm(X - center, axis=1)
        bucket_df["dist_to_center"] = dist
        return bucket_df

    model = KMeans(n_clusters=k, random_state=args.random_state, n_init=10)
    labels = model.fit_predict(X)
    centers = model.cluster_centers_

    dists = np.zeros(len(X), dtype=float)
    for i in range(len(X)):
        dists[i] = np.linalg.norm(X[i] - centers[labels[i]])

    bucket_df["cluster_id"] = labels
    bucket_df["dist_to_center"] = dists
    return bucket_df


def sample_from_cluster(cluster_df: pd.DataFrame, args) -> pd.DataFrame:
    cluster_df = cluster_df.copy().sort_values("dist_to_center").reset_index(drop=True)
    n = len(cluster_df)

    if n <= args.small_bucket_full_sample_threshold:
        cluster_df["sample_role"] = "all"
        cluster_df["is_sampled"] = 1
        return cluster_df

    sampled_indices = []
    roles = {}

    if args.sample_center:
        center_idx = 0
        sampled_indices.append(center_idx)
        roles[center_idx] = "center"

    if args.sample_outlier:
        outlier_idx = n - 1
        if outlier_idx not in sampled_indices:
            sampled_indices.append(outlier_idx)
            roles[outlier_idx] = "outlier"

    if args.sample_boundary:
        boundary_idx = max(0, int(round(0.75 * (n - 1))))
        if boundary_idx not in sampled_indices:
            sampled_indices.append(boundary_idx)
            roles[boundary_idx] = "boundary"

    cluster_df["sample_role"] = ""
    cluster_df["is_sampled"] = 0

    for idx in sampled_indices:
        cluster_df.loc[idx, "sample_role"] = roles[idx]
        cluster_df.loc[idx, "is_sampled"] = 1

    return cluster_df


# ============================================================
# 7. Candidate list for LLM
# ============================================================

def safe_jsonable(x):
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass

    if isinstance(x, (np.integer, )):
        return int(x)
    if isinstance(x, (np.floating, )):
        return float(x)
    return x


def get_max_candidates_for_column(column: str, args) -> int:
    return args.hard_column_max_candidates if normalize_col_name(column) in SOCCER_HARD_COLUMNS else args.default_max_candidates_per_cell_for_llm


CANDIDATE_FEATURE_EXPORT_COLS = [
    "candidate_value",
    "candidate_rank",
    "candidate_source",
    "candidate_sources",
    "candidate_source_text",
    "source_count",
    "source_score",
    "is_multi_source_candidate",

    "is_from_rule",
    "is_from_neighbor",
    "is_from_similarity",
    "is_from_hint",
    "is_from_dictionary",
    "is_from_pattern_restore",
    "is_from_column_top",
    "is_from_soccer_profile",
    "is_from_soccer_direct_rule",
    "is_from_domain",
    "is_from_numeric_rule",
    "is_from_profile_key_majority",
    "is_from_entity_pair_majority",
    "is_from_team_season_position",

    "profile_key_group_size",
    "profile_key_top_count",
    "profile_key_majority_ratio",
    "entity_pair_group_size",
    "entity_pair_top_count",
    "entity_pair_majority_ratio",
    "team_season_position_group_size",
    "team_season_position_top_count",
    "team_season_position_ratio",

    "contains_rule_expected_source",
    "contains_team_season_source",
    "contains_team_profile_source",
    "contains_stadium_city_source",
    "contains_soccer_profile_source",
    "contains_position_source",
    "contains_birthyear_source",
    "contains_season_source",
    "contains_team_source",
    "contains_city_source",
    "contains_stadium_source",
    "contains_manager_source",
    "contains_domain_source",
    "contains_neighbor_source",
    "contains_column_top_source",
    "contains_hint_source",
    "contains_profile_key_majority_source",
    "contains_entity_pair_majority_source",
    "contains_team_season_position_source",
    "contains_name_source",
    "contains_surname_source",
    "contains_birthplace_source",
    "is_weak_frequency_only_candidate",
    "candidate_is_pure_weak_global_for_entity",

    "candidate_rule_support_ratio",
    "rule_support_gain",
    "candidate_matches_any_rule_expected",
    "candidate_matches_any_strong_rule_expected",
    "candidate_matches_any_expected_values_from_rules",
    "candidate_equals_neighbor_majority",
    "candidate_equals_suggested_correct_value",
    "candidate_matches_similar_row_target",
    "candidate_match_ratio_in_similar_rows",

    "bayes_mean_cond_prob",
    "bayes_max_cond_prob",
    "bayes_prod_logprob",
    "edit_similarity_to_dirty",
    "avg_keyboard_distance",
    "phonetic_similarity_to_dirty",
    "candidate_in_column_top_values",
    "candidate_column_top_ratio",

    "candidate_has_profile_majority_signal",
    "candidate_has_high_conf_profile_majority",
    "candidate_has_very_high_conf_profile_majority",
    "candidate_entity_safe_fill",
    "candidate_position_profile_strong",
    "candidate_is_name_entity_majority",
    "candidate_is_surname_entity_majority",
    "candidate_is_birthplace_entity_majority",
    "candidate_is_position_profile_majority",
    "candidate_is_profile_or_pair_majority",
    "candidate_entity_majority_ratio_gain_over_dirty",

    "candidate_in_soccer_domain",
    "dirty_in_soccer_domain",
    "candidate_domain_validity_gain",
    "candidate_equals_domain_canonical_dirty",

    "candidate_position_canonical",
    "dirty_position_canonical",
    "position_canonicalization_gain",
    "candidate_birthyear_valid",
    "dirty_birthyear_valid",
    "candidate_season_valid",
    "dirty_season_valid",
    "candidate_age_at_season_valid",
    "dirty_age_at_season_valid",
    "year_age_consistency_gain",

    "team_season_group_size",
    "candidate_team_season_count",
    "dirty_team_season_count",
    "candidate_team_season_ratio",
    "dirty_team_season_ratio",
    "candidate_team_season_rank",
    "dirty_team_season_rank",
    "team_season_ratio_gain",

    "team_group_size",
    "candidate_team_count",
    "dirty_team_count",
    "candidate_team_ratio",
    "dirty_team_ratio",
    "candidate_team_rank",
    "dirty_team_rank",
    "team_ratio_gain",

    "stadium_city_group_size",
    "candidate_stadium_city_count",
    "dirty_stadium_city_count",
    "candidate_stadium_city_ratio",
    "dirty_stadium_city_ratio",
    "candidate_stadium_city_rank",
    "dirty_stadium_city_rank",
    "stadium_city_ratio_gain",

    "soccer_profile_ratio_gain",
    "soccer_consistency_score",
    "soccer_evidence_flags",
    "soccer_evidence_flag_count",
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
    "row_team",
    "row_city",
    "row_stadium",
    "row_manager",
    "row_birthyear_int",
    "row_season_int",
    "row_age_at_season",
    "row_position_canonical",
]


def build_candidate_list_for_cell(cell_key: Tuple[int, str], cand_df: pd.DataFrame, args) -> List[Dict[str, Any]]:
    row_id, column = cell_key
    g = cand_df[(cand_df["row_id"] == row_id) & (cand_df["column"] == column)].copy()
    g = sort_candidate_group(g)

    limit = get_max_candidates_for_column(column, args)

    if DROP_UNKNOWN_IF_TOO_MANY and len(g) > limit and "candidate_value" in g.columns:
        g = g[g["candidate_value"].astype(str) != UNKNOWN_CANDIDATE_NAME]

    g = g.head(limit).copy()

    score_col = get_score_col(g)
    candidate_rows = []
    for _, r in g.iterrows():
        item = {}
        for col in CANDIDATE_FEATURE_EXPORT_COLS:
            if col in r.index:
                item[col] = safe_jsonable(r.get(col))

        if score_col:
            item["candidate_score"] = safe_jsonable(r.get(score_col))
        else:
            item["candidate_score"] = safe_jsonable(r.get("source_score", None))

        if "candidate_value" not in item and "candidate_value" in r.index:
            item["candidate_value"] = safe_jsonable(r.get("candidate_value"))

        candidate_rows.append(item)

    return candidate_rows


def build_sampled_jsonl(sampled_df: pd.DataFrame, cand_df: pd.DataFrame, detail_map: Dict[Tuple[int, str], Dict[str, Any]], output_path: str, args):
    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in sampled_df.iterrows():
            key = (int(row["row_id"]), row["column"])
            detail_obj = detail_map.get(key, {})

            out = {
                "row_id": int(row["row_id"]),
                "column": row["column"],
                "dirty_value": row.get("dirty_value"),
                "bucket_id": row["bucket_id"],
                "cluster_id": int(row["cluster_id"]),
                "sample_role": row["sample_role"],
                "dist_to_center": float(row["dist_to_center"]),
                "cell_summary": {
                    "semantic_type": row.get("semantic_type"),
                    "main_rule_type": row.get("main_rule_type"),
                    "main_usage_role": row.get("main_usage_role"),
                    "violation_count": row.get("violation_count"),
                    "conflict_score": row.get("conflict_score"),
                    "candidate_count": row.get("candidate_count"),
                    "top1_score": row.get("top1_score"),
                    "top2_score": row.get("top2_score"),
                    "top_margin": row.get("top_margin"),
                    "confidence_bucket": row.get("confidence_bucket"),
                    "candidate_count_bucket": row.get("candidate_count_bucket"),
                    "is_hard_column": row.get("is_hard_column"),
                    "soccer_conflict_bucket": row.get("soccer_conflict_bucket"),
                    "soccer_consistency_score": row.get("soccer_consistency_score"),
                    "soccer_evidence_flag_count": row.get("soccer_evidence_flag_count"),
                    "target_is_primary_column": row.get("target_is_primary_column"),
                    "target_is_context_only_column": row.get("target_is_context_only_column"),
                    "birthyear_value_invalid": row.get("birthyear_value_invalid"),
                    "season_value_invalid": row.get("season_value_invalid"),
                    "position_value_invalid": row.get("position_value_invalid"),
                    "position_needs_canonicalization": row.get("position_needs_canonicalization"),
                    "birthyear_season_age_conflict": row.get("birthyear_season_age_conflict"),
                    "team_context_conflict": row.get("team_context_conflict"),
                    "final_pred_error_prob": row.get("final_pred_error_prob"),
                    "is_high_risk": row.get("is_high_risk"),
                    "is_hard_case_from_detection": row.get("is_hard_case_from_detection"),
                },
                "candidates": build_candidate_list_for_cell(key, cand_df, args),
                "detail_context": detail_obj,
            }
            write_jsonl_record(f, out)


# ============================================================
# 8. Pairwise request construction
# ============================================================

def make_pair_ids(candidates: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    n = len(candidates)
    if n <= 1:
        return []

    if PAIRWISE_STRATEGY == "one_vs_top":
        return [(0, j) for j in range(1, n)]

    return list(combinations(range(n), 2))


def compress_row_values(row_values: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(row_values, dict):
        return {}

    keep_norm = {
        "name", "surname", "birthyear", "birthplace", "position",
        "team", "city", "stadium", "season", "manager"
    }
    out = {}
    for k, v in row_values.items():
        if normalize_col_name(k) in keep_norm:
            out[k] = v
    return out


def build_context_for_prompt(detail_context: Dict[str, Any], dirty_value: str, column: str, cell_row: Optional[pd.Series] = None) -> Dict[str, Any]:
    out = {
        "column": column,
        "dirty_value": dirty_value,
    }

    if isinstance(detail_context, dict):
        row_ctx = detail_context.get("row_context", {})
        if isinstance(row_ctx, dict):
            row_values = row_ctx.get("row_values", {})
            if isinstance(row_values, dict):
                out["row_values"] = compress_row_values(row_values)

            soccer_bundle = row_ctx.get("soccer_profile_bundle", {})
            if isinstance(soccer_bundle, dict):
                out["row_soccer_profile_bundle"] = soccer_bundle

        col_ctx = detail_context.get("column_context", {})
        if isinstance(col_ctx, dict):
            out["column_context"] = {
                "semantic_type": col_ctx.get("semantic_type"),
                "detected_type": col_ctx.get("detected_type"),
                "top_values_with_counts": col_ctx.get("top_values_with_counts"),
                "dominant_pattern": col_ctx.get("dominant_pattern"),
                "current_value_pattern": col_ctx.get("current_value_pattern"),
                "numeric_stats": col_ctx.get("numeric_stats"),
                "top_patterns_with_counts": col_ctx.get("top_patterns_with_counts"),
            }

        conf_ctx = detail_context.get("conflict_context", {})
        if isinstance(conf_ctx, dict):
            violated = conf_ctx.get("violated_rules_detailed", [])
            out["conflict_context"] = {
                "has_rule_violation": conf_ctx.get("has_rule_violation"),
                "violation_count": conf_ctx.get("violation_count"),
                "strong_rule_count": conf_ctx.get("strong_rule_count"),
                "candidate_generation_rule_count": conf_ctx.get("candidate_generation_rule_count"),
                "fd_like_count": conf_ctx.get("fd_like_count"),
                "soccer_domain_rule_count": conf_ctx.get("soccer_domain_rule_count"),
                "soccer_format_rule_count": conf_ctx.get("soccer_format_rule_count"),
                "soccer_numeric_rule_count": conf_ctx.get("soccer_numeric_rule_count"),
                "soccer_consistency_rule_count": conf_ctx.get("soccer_consistency_rule_count"),
                "violated_rules_detailed": violated[:10] if isinstance(violated, list) else [],
                "expected_values_from_rules": conf_ctx.get("expected_values_from_rules", []),
                "alternative_values_from_column": conf_ctx.get("alternative_values_from_column", []),
                "all_alternative_values": conf_ctx.get("all_alternative_values", []),
            }

        sim_ctx = detail_context.get("similarity_context", {})
        if isinstance(sim_ctx, dict):
            out["similarity_context"] = {
                "neighbor_majority_value": sim_ctx.get("neighbor_majority_value"),
                "neighbor_majority_ratio": sim_ctx.get("neighbor_majority_ratio"),
                "neighbor_value_distribution": sim_ctx.get("neighbor_value_distribution"),
                "topk_similar_rows": sim_ctx.get("topk_similar_rows", [])[:5],
            }

        soccer_ctx = detail_context.get("soccer_consistency_context", {})
        if not isinstance(soccer_ctx, dict) or not soccer_ctx:
            soccer_ctx = detail_context.get("adult_consistency_context", {})
        if isinstance(soccer_ctx, dict):
            bundle = soccer_ctx.get("soccer_profile_bundle", soccer_ctx.get("adult_profile_bundle", {}))
            out["soccer_consistency_context"] = {
                "soccer_profile_bundle": bundle,
                "soccer_profile_inconsistencies": (
                    bundle.get("soccer_profile_inconsistencies", [])
                    if isinstance(bundle, dict) else []
                ),
                "evidence_flags": soccer_ctx.get("evidence_flags", []),
                "team_season_distribution": soccer_ctx.get("team_season_distribution", {}),
                "team_distribution": soccer_ctx.get("team_distribution", {}),
                "stadium_city_distribution": soccer_ctx.get("stadium_city_distribution", {}),
                "soccer_consistency_score": soccer_ctx.get("soccer_consistency_score"),
            }

        soccer_attr = detail_context.get("soccer_attribution_features", {})
        if isinstance(soccer_attr, dict):
            out["soccer_attribution_features"] = soccer_attr

        bayes_ctx = detail_context.get("bayesian_context", {})
        if isinstance(bayes_ctx, dict):
            out["bayesian_context"] = bayes_ctx

    # Fallback with cell-level summary fields.
    if cell_row is not None:
        out["cell_summary_fallback"] = {
            "semantic_type": cell_row.get("semantic_type"),
            "main_rule_type": cell_row.get("main_rule_type"),
            "main_usage_role": cell_row.get("main_usage_role"),
            "soccer_conflict_bucket": cell_row.get("soccer_conflict_bucket"),
            "soccer_consistency_score": cell_row.get("soccer_consistency_score"),
            "birthyear_value_invalid": cell_row.get("birthyear_value_invalid"),
            "season_value_invalid": cell_row.get("season_value_invalid"),
            "position_value_invalid": cell_row.get("position_value_invalid"),
            "position_needs_canonicalization": cell_row.get("position_needs_canonicalization"),
            "birthyear_season_age_conflict": cell_row.get("birthyear_season_age_conflict"),
            "team_context_conflict": cell_row.get("team_context_conflict"),
        }

    return out


def build_column_specific_instruction(column: str) -> str:
    col = normalize_col_name(column)

    if col == "position":
        return (
            "本列是 Soccer 的 position。优先选择合法且规范的球员位置值，例如 Defender、Forward、Goalkeeper、Midfield。"
            "如果候选带有 is_from_team_season_position、candidate_position_profile_strong 或 team_season_position_ratio 较高，说明它来自 team/season/birthyear/position 的上下文多数值，应优先考虑。"
            "单纯 domain_position_core_value 只是弱候选，不能压过 team-season-position、neighbor/similar、规则证据。"
        )

    if col == "birthyear":
        return (
            "本列是 birthyear。优先选择合法出生年份，并检查与 season 的年龄关系：season - birthyear 通常应在 15 到 48 岁之间。"
            "如果只是 198O/198l/199S 这类字符错误，应选择对应规范年份。"
            "不要仅根据列内高频年份猜测。"
        )

    if col == "season":
        return (
            "本列是 season。优先选择合法赛季年份，并检查与 birthyear 的年龄关系：season - birthyear 通常应在 15 到 48 岁之间。"
            "如果是 2O11/201l 这类格式错误，应选择规范年份。"
            "team-season 的上下文分布可以作为辅助证据。"
        )

    if col in {"team", "city", "stadium", "manager"}:
        return (
            f"本列是 {col}。优先考虑 team/season/stadium/city/manager 之间的上下文一致性。"
            "team_season_distribution、team_distribution、stadium_city_distribution 是重要但仍需谨慎的证据。"
            "如果只是列内高频或相似行弱证据，且没有规则或上下文支持，应输出 Unknown。"
        )

    if col in {"name", "surname", "birthplace"}:
        return (
            f"本列是 {col}。这是文本实体字段，不能凭全局高频猜测。"
            "如果候选带有 entity_pair_majority 或 profile_key_majority，尤其是 candidate_entity_safe_fill=1、candidate_has_high_conf_profile_majority=1、entity_pair_majority_ratio/profile_key_majority_ratio 较高，应优先考虑。"
            "如果候选只是 column_top_value/global_dictionary/domain 候选且没有 profile/neighbor/similar/rule 支持，应倾向 Unknown。"
        )

    return (
        "优先考虑规则一致性、Soccer profile 一致性、domain 合法性、相似行/邻居证据和候选来源可靠性。"
    )


def build_pairwise_prompt(context_obj: Dict[str, Any], cand_a: Dict[str, Any], cand_b: Dict[str, Any], column: str) -> str:
    column_instruction = build_column_specific_instruction(column)

    prompt = f"""
你现在在做 Soccer 表格数据清洗中的错误修复排序任务。

目标：
对于一个被检测为错误的单元格，请在两个候选修复值 A 和 B 中判断谁更适合作为最终修复值。
你必须只在 A、B、Unknown 中三选一：
- A：候选 A 更好
- B：候选 B 更好
- Unknown：无法可靠判断，或者两个候选都不够好

Soccer 数据集字段包括：
name, surname, birthyear, birthplace, position, team, city, stadium, season, manager。

通用判断原则：
1. 优先考虑强规则证据：expected_values_from_rules、violated_rules_detailed、domain / numeric / consistency rules。
2. 对 position，优先选择合法足球位置和值规范化；不要只因为列内高频选择。
3. 对 birthyear 和 season，必须检查年份合法性，以及 season - birthyear 是否通常在 15 到 48 岁之间。
4. 对 team/city/stadium/manager，优先考虑 team-season、team、stadium-city 的上下文分布，但这些证据不能单独压过强规则。
5. 对 name/surname/birthplace，优先考虑 profile_key_majority 和 entity_pair_majority 候选；这些候选来自 name-surname、birthyear-team-position、manager-team-season 等重复上下文的多数值。
6. 如果候选带有 candidate_entity_safe_fill=1、candidate_has_high_conf_profile_majority=1、profile_key_majority_ratio/entity_pair_majority_ratio 较高，且不是纯全局高频候选，应优先于 column_top/global_dictionary。
7. 对 position，如果候选带有 team_season_position 或 candidate_position_profile_strong，应优先于单纯 domain_position_core_value。
8. 邻居、相似行、列内 top value 是辅助证据；如果没有强证据，宁可输出 Unknown。
9. 如果两个候选都明显不可靠，或者只是频率候选，输出 Unknown。
10. 不要为了让表面更常见而过度修复。

本列专属判断：
{column_instruction}

请严格输出 JSON，不要输出额外解释文字。
JSON 格式如下：
{{
  "winner": "A" 或 "B" 或 "Unknown",
  "reason_type": "rule_consistency" / "soccer_profile_consistency" / "domain_validity" / "position_canonicalization" / "year_age_consistency" / "team_context_consistency" / "context_match" / "cooccurrence_support" / "spelling_similarity" / "pattern_match" / "uncertain" / "other",
  "reason": "一句简短原因"
}}

下面是待判断样本：

上下文：
{json.dumps(context_obj, ensure_ascii=False, indent=2)}

候选 A：
{json.dumps(cand_a, ensure_ascii=False, indent=2)}

候选 B：
{json.dumps(cand_b, ensure_ascii=False, indent=2)}
""".strip()
    return prompt


def build_pairwise_requests(sampled_df: pd.DataFrame, cand_df: pd.DataFrame, detail_map: Dict[Tuple[int, str], Dict[str, Any]], args) -> List[Dict[str, Any]]:
    requests = []

    for _, row in sampled_df.iterrows():
        row_id = int(row["row_id"])
        column = str(row["column"])
        dirty_value = row.get("dirty_value")
        key = (row_id, column)

        candidates = build_candidate_list_for_cell(key, cand_df, args)
        if len(candidates) <= 1:
            continue

        pair_ids = make_pair_ids(candidates)
        detail_obj = detail_map.get(key, {})
        context_obj = build_context_for_prompt(detail_obj, dirty_value, column, row)

        for pair_idx, (i, j) in enumerate(pair_ids):
            cand_a = candidates[i]
            cand_b = candidates[j]

            req = {
                "request_id": f"{row_id}__{column}__pair{pair_idx}",
                "row_id": row_id,
                "column": column,
                "dirty_value": dirty_value,
                "candidate_a_index": i,
                "candidate_b_index": j,
                "candidate_a_value": cand_a.get("candidate_value"),
                "candidate_b_value": cand_b.get("candidate_value"),
                "pairwise_prompt": build_pairwise_prompt(context_obj, cand_a, cand_b, column),
                "context_obj": context_obj,
                "candidate_a": cand_a,
                "candidate_b": cand_b,
            }
            requests.append(req)

    return requests


# ============================================================
# 9. LLM call
# ============================================================

def get_client(args) -> Any:
    if OpenAI is None:
        raise ImportError("openai package is not installed. Please install openai or use --dry_run_only_export_requests 1.")
    if not args.api_key:
        raise ValueError("未设置 API key。请设置 DEEPSEEK_API_KEY 或传入 --api_key。")
    return OpenAI(api_key=args.api_key, base_url=args.base_url)


def call_llm_pairwise(client: Any, prompt: str, args) -> Dict[str, Any]:
    for attempt in range(1, args.max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=args.model_name,
                temperature=args.temperature,
                messages=[
                    {"role": "system", "content": "你是一个严谨的 Soccer 表格数据清洗排序专家。你必须只输出 JSON。"},
                    {"role": "user", "content": prompt},
                ],
            )
            text = resp.choices[0].message.content.strip()

            try:
                return json.loads(text)
            except Exception:
                start = text.find("{")
                end = text.rfind("}")
                if start != -1 and end != -1 and end > start:
                    return json.loads(text[start:end+1])
                raise ValueError(f"LLM 输出不是合法 JSON: {text}")

        except Exception as e:
            if attempt == args.max_retries:
                return {
                    "winner": "Unknown",
                    "reason_type": "uncertain",
                    "reason": f"LLM 调用失败: {str(e)}",
                }
            time.sleep(args.retry_sleep_seconds)


def normalize_response_item(req: Dict[str, Any], llm_out: Dict[str, Any]) -> Dict[str, Any]:
    winner = str(llm_out.get("winner", "Unknown")).strip()
    if winner not in {"A", "B", "Unknown"}:
        winner = "Unknown"

    return {
        "request_id": req["request_id"],
        "row_id": req["row_id"],
        "column": req["column"],
        "dirty_value": req["dirty_value"],
        "candidate_a_index": req["candidate_a_index"],
        "candidate_b_index": req["candidate_b_index"],
        "candidate_a_value": req["candidate_a_value"],
        "candidate_b_value": req["candidate_b_value"],
        "winner": winner,
        "reason_type": llm_out.get("reason_type", "uncertain"),
        "reason": llm_out.get("reason", ""),
    }


def load_existing_responses(path: str, resume: bool) -> Dict[str, Dict[str, Any]]:
    existing = {}
    if not resume or not os.path.exists(path):
        return existing

    with open(path, "r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                rid = obj.get("request_id", None)
                if rid:
                    existing[str(rid)] = obj
            except Exception as e:
                print(f"[WARN] 跳过已有 response 中无法解析的行: line={line_no}, err={e}")

    if existing:
        print(f"[INFO] loaded existing LLM responses for resume: {len(existing)}")
    return existing


def call_one_request(req: Dict[str, Any], args) -> Dict[str, Any]:
    client = get_client(args)
    llm_out = call_llm_pairwise(client, req["pairwise_prompt"], args)
    return normalize_response_item(req, llm_out)


def run_llm_requests(requests: List[Dict[str, Any]], output_path: str, args) -> List[Dict[str, Any]]:
    existing = load_existing_responses(output_path, args.resume_from_existing_responses)

    all_by_id = {}
    pending = []
    for req in requests:
        rid = str(req["request_id"])
        if rid in existing:
            all_by_id[rid] = existing[rid]
        else:
            pending.append(req)

    print(f"[INFO] existing responses reused: {len(all_by_id)}")
    print(f"[INFO] pending LLM requests: {len(pending)}")
    print(f"[INFO] LLM max workers: {args.llm_max_workers}")

    if not pending:
        return [all_by_id[str(req["request_id"])] for req in requests if str(req["request_id"]) in all_by_id]

    write_lock = threading.Lock()
    done_count = 0

    with open(output_path, "a", encoding="utf-8") as f:
        if args.llm_max_workers <= 1:
            for req in pending:
                item = call_one_request(req, args)
                with write_lock:
                    write_jsonl_record(f, item)
                    f.flush()
                all_by_id[str(req["request_id"])] = item
                done_count += 1
                if done_count % args.llm_progress_every == 0:
                    print(f"[INFO] LLM progress: new_done={done_count}/{len(pending)}, total_done={len(all_by_id)}/{len(requests)}")
        else:
            with ThreadPoolExecutor(max_workers=args.llm_max_workers) as ex:
                future_to_req = {ex.submit(call_one_request, req, args): req for req in pending}
                for fut in as_completed(future_to_req):
                    req = future_to_req[fut]
                    try:
                        item = fut.result()
                    except Exception as e:
                        item = normalize_response_item(req, {
                            "winner": "Unknown",
                            "reason_type": "uncertain",
                            "reason": f"并发调用异常: {str(e)}",
                        })

                    with write_lock:
                        write_jsonl_record(f, item)
                        f.flush()

                    all_by_id[str(req["request_id"])] = item
                    done_count += 1
                    if done_count % args.llm_progress_every == 0:
                        print(f"[INFO] LLM progress: new_done={done_count}/{len(pending)}, total_done={len(all_by_id)}/{len(requests)}")

    return [all_by_id[str(req["request_id"])] for req in requests if str(req["request_id"]) in all_by_id]


# ============================================================
# 10. Aggregate pairwise outputs
# ============================================================

def aggregate_pairwise_results(pairwise_results: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    grouped = defaultdict(list)
    for x in pairwise_results:
        grouped[(int(x["row_id"]), str(x["column"]))].append(x)

    final_rows = []
    top1_rows = []

    for (row_id, column), items in grouped.items():
        candidate_counter = {}
        wins = Counter()
        losses = Counter()
        reason_counter = Counter()
        dirty_value = items[0].get("dirty_value", "")

        for x in items:
            a = x.get("candidate_a_value", None)
            b = x.get("candidate_b_value", None)
            candidate_counter[a] = 1
            candidate_counter[b] = 1

            winner = x.get("winner", "Unknown")
            reason_type = x.get("reason_type", "uncertain")
            reason_counter[reason_type] += 1

            if winner == "A":
                wins[a] += 1
                losses[b] += 1
            elif winner == "B":
                wins[b] += 1
                losses[a] += 1

        def safe_candidate_name(x):
            if x is None:
                return ""
            return str(x)

        all_candidates = list(candidate_counter.keys())
        scores = []
        for cand in all_candidates:
            score = wins[cand] - losses[cand]
            scores.append((cand, score, wins[cand], losses[cand]))

        scores = sorted(
            scores,
            key=lambda x: (-x[1], -x[2], x[3], safe_candidate_name(x[0])),
        )

        for rank_idx, (cand, score, win_cnt, loss_cnt) in enumerate(scores, start=1):
            final_rows.append({
                "row_id": row_id,
                "column": column,
                "dirty_value": dirty_value,
                "teacher_rank": rank_idx,
                "candidate_value": cand,
                "pairwise_score": score,
                "win_count": win_cnt,
                "loss_count": loss_cnt,
            })

        if scores:
            best_cand = "" if scores[0][0] is None else scores[0][0]
            dominant_reason = reason_counter.most_common(1)[0][0] if reason_counter else "uncertain"
            top1_rows.append({
                "row_id": row_id,
                "column": column,
                "dirty_value": dirty_value,
                "teacher_best_candidate": best_cand,
                "teacher_reason_type": dominant_reason,
                "candidate_count_in_pairwise": len(all_candidates),
            })

    return final_rows, top1_rows


# ============================================================
# 11. Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--input_candidate_features_csv", default=DEFAULT_INPUT_CANDIDATE_FEATURES_CSV)
    p.add_argument("--input_detail_jsonl", default=DEFAULT_INPUT_DETAIL_JSONL)

    p.add_argument("--output_cell_summary_csv", default=DEFAULT_OUTPUT_CELL_SUMMARY_CSV)
    p.add_argument("--output_clustered_csv", default=DEFAULT_OUTPUT_CLUSTERED_CSV)
    p.add_argument("--output_sampled_csv", default=DEFAULT_OUTPUT_SAMPLED_CSV)
    p.add_argument("--output_sampled_jsonl", default=DEFAULT_OUTPUT_SAMPLED_JSONL)
    p.add_argument("--output_pairwise_requests_jsonl", default=DEFAULT_OUTPUT_PAIRWISE_REQUESTS_JSONL)
    p.add_argument("--output_pairwise_responses_jsonl", default=DEFAULT_OUTPUT_PAIRWISE_RESPONSES_JSONL)
    p.add_argument("--output_teacher_ranking_csv", default=DEFAULT_OUTPUT_TEACHER_RANKING_CSV)
    p.add_argument("--output_teacher_top1_csv", default=DEFAULT_OUTPUT_TEACHER_TOP1_CSV)
    p.add_argument("--output_bucket_summary_csv", default=DEFAULT_OUTPUT_BUCKET_SUMMARY_CSV)

    p.add_argument("--use_column_in_bucket", type=int, default=int(DEFAULT_USE_COLUMN_IN_BUCKET))
    p.add_argument("--min_bucket_size_for_cluster", type=int, default=DEFAULT_MIN_BUCKET_SIZE_FOR_CLUSTER)
    p.add_argument("--max_clusters_per_bucket", type=int, default=DEFAULT_MAX_CLUSTERS_PER_BUCKET)
    p.add_argument("--target_cluster_size", type=int, default=DEFAULT_TARGET_CLUSTER_SIZE)
    p.add_argument("--small_bucket_full_sample_threshold", type=int, default=DEFAULT_SMALL_BUCKET_FULL_SAMPLE_THRESHOLD)

    p.add_argument("--sample_center", type=int, default=int(DEFAULT_SAMPLE_CENTER))
    p.add_argument("--sample_boundary", type=int, default=int(DEFAULT_SAMPLE_BOUNDARY))
    p.add_argument("--sample_outlier", type=int, default=int(DEFAULT_SAMPLE_OUTLIER))
    p.add_argument("--random_state", type=int, default=DEFAULT_RANDOM_STATE)

    p.add_argument("--default_max_candidates_per_cell_for_llm", type=int, default=DEFAULT_MAX_CANDIDATES_PER_CELL_FOR_LLM)
    p.add_argument("--hard_column_max_candidates", type=int, default=HARD_COLUMN_MAX_CANDIDATES)

    p.add_argument("--model_name", default=MODEL_NAME)
    p.add_argument("--api_key", default=API_KEY)
    p.add_argument("--base_url", default=BASE_URL)
    p.add_argument("--temperature", type=float, default=TEMPERATURE)
    p.add_argument("--max_retries", type=int, default=MAX_RETRIES)
    p.add_argument("--retry_sleep_seconds", type=float, default=RETRY_SLEEP_SECONDS)
    p.add_argument("--llm_max_workers", type=int, default=LLM_MAX_WORKERS)
    p.add_argument("--llm_progress_every", type=int, default=LLM_PROGRESS_EVERY)
    p.add_argument("--resume_from_existing_responses", type=int, default=int(RESUME_FROM_EXISTING_RESPONSES))
    p.add_argument("--dry_run_only_export_requests", type=int, default=0)

    args = p.parse_args()

    args.use_column_in_bucket = bool(args.use_column_in_bucket)
    args.sample_center = bool(args.sample_center)
    args.sample_boundary = bool(args.sample_boundary)
    args.sample_outlier = bool(args.sample_outlier)
    args.resume_from_existing_responses = bool(args.resume_from_existing_responses)
    args.dry_run_only_export_requests = bool(args.dry_run_only_export_requests)

    return args


def main():
    args = parse_args()

    output_paths = [
        args.output_cell_summary_csv,
        args.output_clustered_csv,
        args.output_sampled_csv,
        args.output_sampled_jsonl,
        args.output_pairwise_requests_jsonl,
        args.output_pairwise_responses_jsonl,
        args.output_teacher_ranking_csv,
        args.output_teacher_top1_csv,
        args.output_bucket_summary_csv,
    ]
    for p in output_paths:
        ensure_parent(resolve_output_path(p))

    args.output_cell_summary_csv = resolve_output_path(args.output_cell_summary_csv)
    args.output_clustered_csv = resolve_output_path(args.output_clustered_csv)
    args.output_sampled_csv = resolve_output_path(args.output_sampled_csv)
    args.output_sampled_jsonl = resolve_output_path(args.output_sampled_jsonl)
    args.output_pairwise_requests_jsonl = resolve_output_path(args.output_pairwise_requests_jsonl)
    args.output_pairwise_responses_jsonl = resolve_output_path(args.output_pairwise_responses_jsonl)
    args.output_teacher_ranking_csv = resolve_output_path(args.output_teacher_ranking_csv)
    args.output_teacher_top1_csv = resolve_output_path(args.output_teacher_top1_csv)
    args.output_bucket_summary_csv = resolve_output_path(args.output_bucket_summary_csv)

    if not os.path.exists(args.input_candidate_features_csv):
        raise FileNotFoundError(f"候选特征文件不存在: {args.input_candidate_features_csv}")

    cand_df = pd.read_csv(args.input_candidate_features_csv, encoding="utf-8-sig")
    required = {"row_id", "column", "candidate_value"}
    missing = required - set(cand_df.columns)
    if missing:
        raise ValueError(f"候选特征文件缺少必要字段: {missing}")

    cand_df["row_id"] = pd.to_numeric(cand_df["row_id"], errors="raise").astype(int)
    cand_df["column"] = cand_df["column"].astype(str)

    detail_map = load_detail_jsonl(args.input_detail_jsonl)
    print(f"[INFO] candidate feature rows: {len(cand_df)}")
    print(f"[INFO] detail context cells: {len(detail_map)}")

    # 1) Cell-level summary
    cell_summary_df = build_cell_summary_df(cand_df)
    cell_summary_df.to_csv(args.output_cell_summary_csv, index=False, encoding="utf-8-sig")

    # 2) Bucket + cluster + sample
    cell_summary_df = assign_buckets(cell_summary_df, args.use_column_in_bucket)

    clustered_parts = []
    bucket_summary_rows = []

    for bucket_id, bucket_df in cell_summary_df.groupby("bucket_id", sort=False):
        bucket_clustered = cluster_bucket(bucket_df, args)
        bucket_clustered_parts = []

        for cluster_id, cluster_df in bucket_clustered.groupby("cluster_id", sort=False):
            cluster_sampled = sample_from_cluster(cluster_df, args)
            bucket_clustered_parts.append(cluster_sampled)

            bucket_summary_rows.append({
                "bucket_id": bucket_id,
                "bucket_size": len(bucket_df),
                "cluster_id": int(cluster_id),
                "cluster_size": len(cluster_df),
                "sampled_count": int(cluster_sampled["is_sampled"].sum()),
                "column": str(cluster_df["column"].iloc[0]) if "column" in cluster_df.columns else "unknown",
                "semantic_type": str(cluster_df["semantic_type"].iloc[0]) if "semantic_type" in cluster_df.columns else "unknown",
                "main_rule_type": str(cluster_df["main_rule_type"].iloc[0]) if "main_rule_type" in cluster_df.columns else "unknown",
                "main_usage_role": str(cluster_df["main_usage_role"].iloc[0]) if "main_usage_role" in cluster_df.columns else "unknown",
                "confidence_bucket": str(cluster_df["confidence_bucket"].iloc[0]) if "confidence_bucket" in cluster_df.columns else "unknown",
                "candidate_count_bucket": str(cluster_df["candidate_count_bucket"].iloc[0]) if "candidate_count_bucket" in cluster_df.columns else "unknown",
                "soccer_conflict_bucket": str(cluster_df["soccer_conflict_bucket"].iloc[0]) if "soccer_conflict_bucket" in cluster_df.columns else "unknown",
                "is_hard_column": int(cluster_df["is_hard_column"].iloc[0]) if "is_hard_column" in cluster_df.columns else 0,
                "has_profile_key_majority_candidate": int(cluster_df["has_profile_key_majority_candidate"].max()) if "has_profile_key_majority_candidate" in cluster_df.columns else 0,
                "has_entity_pair_majority_candidate": int(cluster_df["has_entity_pair_majority_candidate"].max()) if "has_entity_pair_majority_candidate" in cluster_df.columns else 0,
                "has_team_season_position_candidate": int(cluster_df["has_team_season_position_candidate"].max()) if "has_team_season_position_candidate" in cluster_df.columns else 0,
            })

        clustered_bucket = pd.concat(bucket_clustered_parts, ignore_index=True)
        clustered_parts.append(clustered_bucket)

    clustered_df = pd.concat(clustered_parts, ignore_index=True) if clustered_parts else pd.DataFrame()
    clustered_df.to_csv(args.output_clustered_csv, index=False, encoding="utf-8-sig")

    sampled_df = clustered_df[clustered_df["is_sampled"] == 1].copy() if not clustered_df.empty else pd.DataFrame()
    sampled_df.to_csv(args.output_sampled_csv, index=False, encoding="utf-8-sig")

    if not sampled_df.empty:
        build_sampled_jsonl(sampled_df, cand_df, detail_map, args.output_sampled_jsonl, args)
    else:
        with open(args.output_sampled_jsonl, "w", encoding="utf-8"):
            pass

    pd.DataFrame(bucket_summary_rows).to_csv(
        args.output_bucket_summary_csv, index=False, encoding="utf-8-sig"
    )

    # 3) Pairwise requests
    requests = build_pairwise_requests(sampled_df, cand_df, detail_map, args) if not sampled_df.empty else []
    with open(args.output_pairwise_requests_jsonl, "w", encoding="utf-8") as f:
        for req in requests:
            write_jsonl_record(f, req)

    print(f"[OK] cell-level 摘要已保存: {args.output_cell_summary_csv}")
    print(f"[OK] 完整聚类结果已保存: {args.output_clustered_csv}")
    print(f"[OK] 采样摘要 CSV 已保存: {args.output_sampled_csv}")
    print(f"[OK] 采样详细 JSONL 已保存: {args.output_sampled_jsonl}")
    print(f"[OK] bucket/cluster 摘要已保存: {args.output_bucket_summary_csv}")
    print(f"[OK] pairwise 请求已保存: {args.output_pairwise_requests_jsonl}")
    print(f"[INFO] 采样 cell 数: {len(sampled_df)}")
    print(f"[INFO] pairwise 请求数: {len(requests)}")

    if args.dry_run_only_export_requests:
        print("[INFO] 当前为 DRY RUN，仅导出请求，不调用 LLM。")
        return

    # 4) LLM call
    responses = run_llm_requests(requests, args.output_pairwise_responses_jsonl, args)

    # 5) Aggregate teacher ranking
    ranking_rows, top1_rows = aggregate_pairwise_results(responses)

    pd.DataFrame(ranking_rows).to_csv(args.output_teacher_ranking_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(top1_rows).to_csv(args.output_teacher_top1_csv, index=False, encoding="utf-8-sig")

    print(f"[OK] pairwise 响应已保存: {args.output_pairwise_responses_jsonl}")
    print(f"[OK] teacher 排序结果已保存: {args.output_teacher_ranking_csv}")
    print(f"[OK] teacher top1 标签已保存: {args.output_teacher_top1_csv}")

    print("\n===== 统计信息 =====")
    print(f"总 cell 数: {cell_summary_df.shape[0]}")
    print(f"总采样 cell 数: {sampled_df.shape[0]}")
    print(f"总 pairwise 请求数: {len(requests)}")
    if not clustered_df.empty:
        print(f"总桶数: {clustered_df['bucket_id'].nunique()}")
        print(f"总簇数: {clustered_df[['bucket_id', 'cluster_id']].drop_duplicates().shape[0]}")

    if len(sampled_df) > 0:
        print("\n采样角色分布:")
        print(sampled_df["sample_role"].value_counts(dropna=False).to_string())

        print("\nSoccer conflict bucket 分布:")
        print(sampled_df["soccer_conflict_bucket"].value_counts(dropna=False).head(20).to_string())

        print("\n采样列分布:")
        print(sampled_df["column"].value_counts(dropna=False).head(20).to_string())

        for c in [
            "has_profile_key_majority_candidate",
            "has_entity_pair_majority_candidate",
            "has_team_season_position_candidate",
            "has_entity_safe_fill_candidate",
            "has_position_profile_strong_candidate",
            "has_high_conf_profile_majority_candidate",
        ]:
            if c in sampled_df.columns:
                print(f"\n{c} sampled cell count:")
                print(int(pd.to_numeric(sampled_df[c], errors="coerce").fillna(0).sum()))
                print(sampled_df.groupby("column")[c].sum().sort_values(ascending=False).head(20).to_string())


if __name__ == "__main__":
    main()

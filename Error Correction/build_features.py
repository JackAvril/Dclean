#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified readable repair-candidate feature builder.

Datasets:
  hospital, flights, beers, rayyan, adult, soccer

Purpose
-------
Build candidate-level features from repair_candidates_expanded.csv.

Common outputs:
  repair_candidate_features_v2.csv
  built_cooccurrence_dict_v2.json

This is a readable unified implementation. It replaces the dataset-specific
build_features.py family with one configurable script. Dataset-specific feature
logic is isolated in DATASET_CONFIGS and feature helper functions.

The output keeps the common columns expected by downstream scripts:
  row_id, column, dirty_value, candidate_value, candidate_rank,
  final_score_from_candidate_gen, candidate_sources, source_count,
  candidate_rule_support_ratio, edit_similarity_to_dirty,
  bayes_mean_cond_prob, source flags, and dataset-specific gain features.
"""

import argparse
import difflib
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


EMPTY_TOKENS = {"", "nan", "none", "null", "na", "n/a", "missing", "empty", "?", "{null}"}


@dataclass
class DatasetConfig:
    name: str
    default_candidates_file: str = "repair_candidates_expanded.csv"
    default_context_file: str = ""
    default_raw_data_file: str = ""
    output_features_csv: str = "repair_candidate_features_v2.csv"
    output_cooccurrence_json: str = "built_cooccurrence_dict_v2.json"
    domain_values: Dict[str, List[str]] = field(default_factory=dict)


DATASET_CONFIGS = {
    "hospital": DatasetConfig(
        name="hospital",
        default_context_file="/mnt/mydata/dq/projects/Splittree/test/Error Detection new/candidate_llm_contexts_v2.jsonl",
        default_raw_data_file="/mnt/mydata/dq/projects/Splittree/hospital/hospital_dirty.csv",
    ),
    "flights": DatasetConfig(
        name="flights",
        default_context_file="/mnt/mydata/dq/projects/Splittree/test/Error Detection flights/candidate_llm_contexts_flights.jsonl",
        default_raw_data_file="/mnt/mydata/dq/projects/Splittree/flights/flights_dirty.csv",
    ),
    "beers": DatasetConfig(
        name="beers",
        default_context_file="/mnt/mydata/dq/projects/Splittree/test/Error Detection beers/candidate_llm_contexts_beers.jsonl",
        default_raw_data_file="/mnt/mydata/dq/projects/Splittree/beers/beers_dirty.csv",
    ),
    "rayyan": DatasetConfig(
        name="rayyan",
        default_context_file="/mnt/mydata/dq/projects/Splittree/test/Error Detection rayyan/candidate_llm_contexts_rayyan.jsonl",
        default_raw_data_file="/mnt/mydata/dq/projects/Splittree/rayyan/rayyan_dirty.csv",
    ),
    "adult": DatasetConfig(
        name="adult",
        default_context_file="/mnt/mydata/dq/projects/Splittree/test/Error Detection Adult/candidate_llm_contexts_adult.jsonl",
        default_raw_data_file="/mnt/mydata/dq/projects/Splittree/adult/adult_dirty.csv",
        domain_values={
            "relationship": ["Husband", "Wife", "Own-child", "Not-in-family", "Unmarried", "Other-relative"],
            "sex": ["Male", "Female"],
            "income": ["LessThan50K", "MoreThan50K", "<=50K", ">50K"],
        },
    ),
    "soccer": DatasetConfig(
        name="soccer",
        default_context_file="/mnt/mydata/dq/projects/Splittree/test/Error Detection Soccer/candidate_llm_contexts_soccer.jsonl",
        default_raw_data_file="/mnt/mydata/dq/projects/Splittree/soccer/soccer_dirty.csv",
        domain_values={
            "position": ["Defender", "Forward", "Goalkeeper", "Midfield", "Midfielder", "Striker", "Winger"],
        },
    ),
}


# ============================================================
# 1. Utilities
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


def safe_float(x: Any, default=0.0) -> float:
    try:
        if is_empty(x):
            return default
        return float(str(x).strip())
    except Exception:
        return default


def safe_int(x: Any, default=0) -> int:
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


def edit_similarity(a: Any, b: Any) -> float:
    a = norm_text(a)
    b = norm_text(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return float(difflib.SequenceMatcher(None, a, b).ratio())


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
    return m.group(1) if m else s


def parse_year(x: Any) -> int:
    y = canonical_year(x)
    return safe_int(y, -1)


def canonical_time(x: Any) -> str:
    s = clean_text(x)
    digits = re.sub(r"\D", "", s)
    if not digits:
        return s
    v = safe_int(digits, -1)
    if 0 <= v <= 2359:
        hh = v // 100
        mm = v % 100
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            return f"{hh:02d}{mm:02d}"
    return s


def canonical_position(x: Any) -> str:
    s = clean_text(x)
    key = re.sub(r"[^a-z]", "", s.lower())
    mp = {
        "defender": "Defender", "defenders": "Defender", "dfender": "Defender",
        "forward": "Forward", "forwards": "Forward", "foward": "Forward",
        "goalkeeper": "Goalkeeper", "goalkeepers": "Goalkeeper", "gk": "Goalkeeper",
        "midfield": "Midfield", "midfielder": "Midfield", "midfields": "Midfield",
        "striker": "Striker", "winger": "Winger",
    }
    return mp.get(key, s)


def canonical_value(dataset: str, column: str, value: Any) -> str:
    col = normalize_col(column, dataset)
    if dataset == "flights" and ("time" in col or col in {"sched_dep_time", "act_dep_time", "sched_arr_time", "act_arr_time"}):
        return canonical_time(value)
    if dataset == "soccer":
        if col in {"birthyear", "season"}:
            return canonical_year(value)
        if col == "position":
            return canonical_position(value)
    if dataset in {"adult"} and col in {"age", "hoursperweek"}:
        m = re.search(r"\d+", clean_text(value))
        return m.group(0) if m else clean_text(value)
    return clean_text(value)


def normalize_key_df(df: pd.DataFrame, desc: str) -> pd.DataFrame:
    required = {"row_id", "column"}
    if not required.issubset(df.columns):
        raise ValueError(f"{desc} must contain row_id,column")
    out = df.copy()
    out["row_id"] = pd.to_numeric(out["row_id"], errors="raise").astype(int)
    out["column"] = out["column"].astype(str).str.strip()
    return out


# ============================================================
# 2. Cooccurrence stats
# ============================================================

def load_raw_data(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def build_cooccurrence(raw_df: pd.DataFrame, cfg: DatasetConfig) -> Dict[str, Any]:
    stats = {
        "column_counts": defaultdict(Counter),
        "pair_counts": defaultdict(Counter),
        "row_values": {},
    }
    if raw_df.empty:
        return stats

    for ridx, row in raw_df.iterrows():
        rv = {}
        for c in raw_df.columns:
            cv = canonical_value(cfg.name, str(c), row[c])
            rv[str(c)] = cv
            if cv:
                stats["column_counts"][str(c)][cv] += 1
        stats["row_values"][int(ridx)] = rv

        # Pair/cooccurrence maps for dataset-specific features.
        norm = {normalize_col(k, cfg.name): v for k, v in rv.items()}
        if cfg.name == "flights":
            keys = [
                ("flight", norm.get("flight")),
                ("src_flight", f"{norm.get('src','')}||{norm.get('flight','')}"),
            ]
        elif cfg.name == "beers":
            keys = [
                ("city", norm.get("city")),
                ("brewery_name", norm.get("brewery_name")),
                ("beer_name", norm.get("beer_name")),
            ]
        elif cfg.name == "rayyan":
            keys = [
                ("journal_title", norm.get("journal_title")),
                ("journal_issn", norm.get("journal_issn")),
            ]
        elif cfg.name == "adult":
            keys = [
                ("relationship", norm.get("relationship")),
                ("education", norm.get("education")),
                ("workclass_occupation", f"{norm.get('workclass','')}||{norm.get('occupation','')}"),
            ]
        elif cfg.name == "soccer":
            keys = [
                ("team_season", f"{norm.get('team','')}||{norm.get('season','')}"),
                ("name_surname", f"{norm.get('name','')}||{norm.get('surname','')}"),
                ("surname", norm.get("surname")),
                ("name", norm.get("name")),
            ]
        else:
            keys = []

        for key_name, key_val in keys:
            if not key_val:
                continue
            for target_col, target_val in norm.items():
                if target_val:
                    stats["pair_counts"][(key_name, key_val, target_col)][target_val] += 1

    return stats


def save_cooccurrence_summary(stats: Dict[str, Any], path: str):
    summary = {
        "column_counts_size": {k: len(v) for k, v in stats["column_counts"].items()},
        "pair_counts_size": {str(k): len(v) for k, v in list(stats["pair_counts"].items())[:5000]},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


# ============================================================
# 3. Feature helpers
# ============================================================

def source_text(row: pd.Series) -> str:
    parts = []
    for c in ["candidate_sources", "candidate_source", "source", "sources", "extra_info"]:
        if c in row.index and not is_empty(row[c]):
            parts.append(str(row[c]))
    return "|".join(parts).lower()


def source_flag(src: str, patterns: List[str]) -> int:
    return int(any(p in src for p in patterns))


def get_candidate_value_col(df: pd.DataFrame) -> str:
    for c in ["candidate_value", "repair_candidate", "candidate", "value_candidate"]:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot find candidate value column. columns={list(df.columns)}")


def get_row_context(stats: Dict[str, Any], row_id: int) -> Dict[str, Any]:
    return stats.get("row_values", {}).get(int(row_id), {})


def profile_ratio(stats: Dict[str, Any], key_name: str, key_value: str, target_col: str, candidate: str) -> Tuple[float, int, int]:
    counter = stats["pair_counts"].get((key_name, key_value, target_col), Counter())
    total = sum(counter.values())
    cnt = counter.get(candidate, 0)
    ratio = cnt / total if total else 0.0
    return ratio, cnt, total


def compute_dataset_features(row: pd.Series, cfg: DatasetConfig, stats: Dict[str, Any]) -> Dict[str, Any]:
    dataset = cfg.name
    row_id = safe_int(row.get("row_id"), -1)
    col = normalize_col(row.get("column"), dataset)
    cand = canonical_value(dataset, col, row.get("candidate_value", ""))
    dirty = canonical_value(dataset, col, row.get("dirty_value", row.get("value", "")))
    rv_raw = get_row_context(stats, row_id)
    rv = {normalize_col(k, dataset): v for k, v in rv_raw.items()}

    out: Dict[str, Any] = {}

    # Common domain validity.
    domain = cfg.domain_values.get(col, [])
    out["candidate_in_dataset_domain"] = int(cand in domain) if domain else 0
    out["candidate_domain_validity_gain"] = out["candidate_in_dataset_domain"] - (int(dirty in domain) if domain else 0)

    if dataset == "flights":
        if "time" in col or col in {"sched_dep_time", "act_dep_time", "sched_arr_time", "act_arr_time"}:
            cand_time = canonical_time(cand)
            dirty_time = canonical_time(dirty)
            out["candidate_time_canonical"] = int(cand == cand_time and cand != "")
            out["dirty_time_canonical"] = int(dirty == dirty_time and dirty != "")
            out["candidate_time_format_gain"] = out["candidate_time_canonical"] - out["dirty_time_canonical"]
            flight = rv.get("flight", "")
            ratio, cnt, total = profile_ratio(stats, "flight", flight, col, cand)
            out["time_consensus_confidence"] = ratio
            out["time_consensus_count"] = cnt
            out["time_consensus_total"] = total
            out["is_from_time_consensus"] = int(ratio > 0.5 and cnt >= 2)

    elif dataset == "beers":
        if col == "state":
            city = rv.get("city", "")
            ratio, cnt, total = profile_ratio(stats, "city", city, "state", cand)
            out["state_city_ratio"] = ratio
            out["state_city_ratio_gain"] = ratio
            out["candidate_equals_state_city_consensus"] = int(ratio > 0.5 and cnt >= 2)
        if col in {"abv", "ibu", "ounces"}:
            out["candidate_numeric_valid"] = int(bool(re.search(r"\d", cand)))
            out["dirty_numeric_valid"] = int(bool(re.search(r"\d", dirty)))
            out["beers_numeric_consistency_gain"] = out["candidate_numeric_valid"] - out["dirty_numeric_valid"]
        if col in {"city", "style"}:
            key_name = "brewery_name" if col == "city" else "beer_name"
            key_val = rv.get(key_name, "")
            ratio, cnt, total = profile_ratio(stats, key_name, key_val, col, cand)
            out[f"{key_name}_{col}_ratio"] = ratio
            out["is_from_beers_profile"] = int(ratio > 0.5 and cnt >= 2)

    elif dataset == "rayyan":
        if col == "article_language":
            out["candidate_language_valid"] = int(len(cand) == 3 and cand.isalpha())
        if col == "journal_issn":
            out["candidate_issn_valid"] = int(bool(re.fullmatch(r"\d{4}-[\dX]{4}", cand.upper())))
            jt = rv.get("journal_title", "")
            ratio, cnt, total = profile_ratio(stats, "journal_title", jt, "journal_issn", cand)
            out["rayyan_consensus_confidence"] = ratio
            out["candidate_is_rayyan_metadata_consensus"] = int(ratio > 0.5 and cnt >= 2)
        if col in {"article_jvolumn", "article_jissue"}:
            out["candidate_numeric_valid"] = int(cand.isdigit())

    elif dataset == "adult":
        if col == "sex":
            rel = rv.get("relationship", "")
            ratio, cnt, total = profile_ratio(stats, "relationship", rel, "sex", cand)
            out["relationship_consistency_gain"] = ratio
            out["is_from_adult_profile"] = int(ratio > 0.5 and cnt >= 2)
        if col == "maritalstatus":
            rel = rv.get("relationship", "")
            ratio, cnt, total = profile_ratio(stats, "relationship", rel, "maritalstatus", cand)
            out["relationship_marital_ratio"] = ratio
            out["is_from_adult_profile"] = int(ratio > 0.5 and cnt >= 2)
        if col in {"age", "hoursperweek"}:
            out["candidate_numeric_valid"] = int(cand.isdigit())
        if col == "education":
            age = safe_int(rv.get("age"), -1)
            out["education_age_conflict_reduction"] = int(age >= 16) if age >= 0 else 0

    elif dataset == "soccer":
        if col == "position":
            cand_can = canonical_position(cand)
            dirty_can = canonical_position(dirty)
            out["candidate_position_valid"] = int(cand_can in cfg.domain_values.get("position", []))
            out["dirty_position_valid"] = int(dirty_can in cfg.domain_values.get("position", []))
            out["position_consistency_gain"] = out["candidate_position_valid"] - out["dirty_position_valid"]
            key = f"{rv.get('team','')}||{rv.get('season','')}"
            ratio, cnt, total = profile_ratio(stats, "team_season", key, "position", cand_can)
            out["team_season_position_ratio"] = ratio
            out["is_from_team_season_position"] = int(ratio > 0.5 and cnt >= 2)
            out["candidate_position_profile_strong"] = out["is_from_team_season_position"]
        if col in {"birthyear", "season"}:
            by = parse_year(cand if col == "birthyear" else rv.get("birthyear"))
            sy = parse_year(cand if col == "season" else rv.get("season"))
            out["candidate_year_valid"] = int(1900 <= parse_year(cand) <= 2030)
            out["birthyear_season_age_gain"] = int(by > 0 and sy > 0 and 15 <= sy - by <= 48)
        if col == "name":
            surname = rv.get("surname", "")
            ratio, cnt, total = profile_ratio(stats, "surname", surname, "name", cand)
            out["entity_pair_majority_ratio"] = ratio
            out["is_from_entity_pair_majority"] = int(ratio > 0.5 and cnt >= 2)
            out["candidate_entity_safe_fill"] = out["is_from_entity_pair_majority"]
        if col == "surname":
            name = rv.get("name", "")
            ratio, cnt, total = profile_ratio(stats, "name", name, "surname", cand)
            out["entity_pair_majority_ratio"] = ratio
            out["is_from_entity_pair_majority"] = int(ratio > 0.5 and cnt >= 2)
            out["candidate_entity_safe_fill"] = out["is_from_entity_pair_majority"]
        if col == "birthplace":
            key = f"{rv.get('name','')}||{rv.get('surname','')}"
            ratio, cnt, total = profile_ratio(stats, "name_surname", key, "birthplace", cand)
            out["entity_pair_majority_ratio"] = ratio
            out["is_from_entity_pair_majority"] = int(ratio > 0.5 and cnt >= 2)
            out["candidate_entity_safe_fill"] = out["is_from_entity_pair_majority"]

    return out


# ============================================================
# 4. Main feature building
# ============================================================

def build_features(candidates: pd.DataFrame, cfg: DatasetConfig, stats: Dict[str, Any]) -> pd.DataFrame:
    candidates = normalize_key_df(candidates, "candidate file").copy()
    cand_col = get_candidate_value_col(candidates)
    candidates = candidates.rename(columns={cand_col: "candidate_value"}) if cand_col != "candidate_value" else candidates

    rows = []
    for _, r in candidates.iterrows():
        row = r.to_dict()
        src = source_text(r)

        dirty = row.get("dirty_value", row.get("value", ""))
        cand = row.get("candidate_value", "")

        row["dirty_value"] = clean_text(dirty)
        row["candidate_value"] = clean_text(cand)
        row["candidate_sources"] = row.get("candidate_sources", row.get("candidate_source", ""))
        row["source_count"] = len([p for p in str(row.get("candidate_sources", "")).split("|") if p.strip()])

        row["is_from_rule"] = max(safe_int(row.get("is_from_rule"), 0), source_flag(src, ["rule", "expected"]))
        row["is_from_neighbor"] = max(safe_int(row.get("is_from_neighbor"), 0), source_flag(src, ["neighbor"]))
        row["is_from_similarity"] = max(safe_int(row.get("is_from_similarity"), 0), source_flag(src, ["similar"]))
        row["is_from_hint"] = max(safe_int(row.get("is_from_hint"), 0), source_flag(src, ["hint", "suggested", "llm"]))
        row["is_from_dictionary"] = max(safe_int(row.get("is_from_dictionary"), 0), source_flag(src, ["dictionary"]))
        row["is_from_column_top"] = max(safe_int(row.get("is_from_column_top"), 0), source_flag(src, ["column_top"]))
        row["is_from_domain"] = max(safe_int(row.get("is_from_domain"), 0), source_flag(src, ["domain"]))
        row["is_from_numeric_rule"] = max(safe_int(row.get("is_from_numeric_rule"), 0), source_flag(src, ["numeric", "year", "time"]))

        row["edit_similarity_to_dirty"] = edit_similarity(row["candidate_value"], row["dirty_value"])
        row["candidate_equals_dirty"] = int(norm_text(row["candidate_value"]) == norm_text(row["dirty_value"]))
        row["candidate_is_empty"] = int(is_empty(row["candidate_value"]))

        support_flags = [
            row["is_from_rule"], row["is_from_neighbor"], row["is_from_similarity"],
            row["is_from_hint"], row["is_from_dictionary"], row["is_from_domain"],
            row["is_from_numeric_rule"],
        ]
        row["candidate_rule_support_ratio"] = sum(support_flags[:4]) / 4.0
        row["bayes_mean_cond_prob"] = safe_float(row.get("bayes_mean_cond_prob"), 0.0)

        # Dataset-specific features.
        row.update(compute_dataset_features(pd.Series(row), cfg, stats))

        # Final candidate-generation feature score.
        base = safe_float(row.get("source_score"), 0.0)
        base += 0.15 * row["is_from_rule"]
        base += 0.12 * row["is_from_neighbor"]
        base += 0.10 * row["is_from_similarity"]
        base += 0.10 * row["is_from_hint"]
        base += 0.08 * row["is_from_domain"]
        base += 0.06 * row["is_from_numeric_rule"]
        base += 0.04 * row["source_count"]
        base += 0.08 * safe_float(row.get("candidate_domain_validity_gain"), 0.0)
        base += 0.08 * safe_float(row.get("relationship_consistency_gain"), 0.0)
        base += 0.08 * safe_float(row.get("state_city_ratio_gain"), 0.0)
        base += 0.08 * safe_float(row.get("time_consensus_confidence"), 0.0)
        base += 0.08 * safe_float(row.get("entity_pair_majority_ratio"), 0.0)
        base += 0.08 * safe_float(row.get("team_season_position_ratio"), 0.0)
        base -= 0.12 * row["candidate_equals_dirty"]
        row["final_score_from_candidate_gen"] = safe_float(row.get("final_score_from_candidate_gen"), base)

        rows.append(row)

    out = pd.DataFrame(rows)

    if not out.empty:
        out = out.sort_values(
            ["row_id", "column", "final_score_from_candidate_gen", "candidate_value"],
            ascending=[True, True, False, True],
        ).reset_index(drop=True)
        out["candidate_rank"] = out.groupby(["row_id", "column"]).cumcount() + 1

    return out


def main():
    parser = argparse.ArgumentParser(description="Unified readable repair-candidate feature builder.")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_CONFIGS.keys()))
    parser.add_argument("--candidates_file", default=None)
    parser.add_argument("--raw_data_file", default=None)
    parser.add_argument("--output_features_csv", default=None)
    parser.add_argument("--output_cooccurrence_json", default=None)
    args = parser.parse_args()

    cfg = DATASET_CONFIGS[args.dataset]
    candidates_file = args.candidates_file or cfg.default_candidates_file
    raw_data_file = args.raw_data_file or cfg.default_raw_data_file
    output_features = args.output_features_csv or cfg.output_features_csv
    output_cooc = args.output_cooccurrence_json or cfg.output_cooccurrence_json

    print("========== Unified repair candidate feature builder ==========")
    print(f"dataset: {cfg.name}")
    print(f"candidates_file: {candidates_file}")
    print(f"raw_data_file: {raw_data_file}")
    print(f"output_features_csv: {output_features}")
    print(f"output_cooccurrence_json: {output_cooc}")

    cand = pd.read_csv(candidates_file, encoding="utf-8-sig", low_memory=False)
    raw = load_raw_data(raw_data_file)
    stats = build_cooccurrence(raw, cfg)
    save_cooccurrence_summary(stats, output_cooc)

    features = build_features(cand, cfg, stats)
    Path(output_features).parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_features, index=False, encoding="utf-8-sig")

    print(f"[OK] saved features: {output_features}, rows={len(features)}")
    print(f"[OK] saved cooccurrence summary: {output_cooc}")
    if "column" in features.columns:
        print("\nRows by column:")
        print(features["column"].value_counts().head(30).to_string())


if __name__ == "__main__":
    main()

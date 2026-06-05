#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_infer_candidate_features_from_whitelist_soccer_v2.py

作用：
1) 读取 summarize_full_contexts_soccer_v2.py 输出的全候选 flat CSV；
2) 读取 cluster_and_sample_candidates_soccer_v2.py 输出的聚类/采样 CSV；
3) 合并 bucket_id / cluster_id / sample_role 等聚类字段；
4) 为 soccer 数据集补齐训练/推理脚本需要的派生特征；
5) 输出 candidate_infer_ready_soccer.csv，供 infer_mlp_soccer.py / control_iteration_soccer.py 使用。

注意：
- 不使用 clean / dirty 对比结果，不使用 ground truth。
- 尽量保持原有输入输出格式与字段名；新增字段只作为兼容和稳定推理的辅助特征。
"""

import os
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ============================================================
# 1. 用户配置区
# ============================================================

INPUT_FULL_CANDIDATE_CSV = "candidate_llm_contexts_flat_soccer.csv"
INPUT_CLUSTERED_CSV = "candidate_clustered_soccer.csv"
OUTPUT_INFER_READY_CSV = "candidate_infer_ready_soccer.csv"

# 如果聚类文件不存在，是否允许只用 full candidate 构建推理集。
ALLOW_MISSING_CLUSTER_FILE = True

MISSING_TOKENS = {
    "", "empty", "null", "none", "nan", "na", "n/a", "missing", "unk",
    "unknown", "?", "not available", "contact airline"
}

SOCCER_COLUMNS = [
    "name", "surname", "birthyear", "birthplace", "position",
    "team", "city", "stadium", "season", "manager",
]

PRIMARY_TARGET_COLUMNS = {
    "birthyear", "birthplace", "position", "team", "city", "stadium", "season", "manager"
}
CONTEXT_ONLY_COLUMNS = {"name", "surname"}

VALID_POSITIONS = {"Goalkeeper", "Defender", "Midfield", "Midfielder", "Forward", "Striker", "Winger"}
POSITION_CANONICAL_MAP = {"Midfielder": "Midfield"}

BIRTHYEAR_MIN = 1940
BIRTHYEAR_MAX = 2010
SEASON_MIN = 1900
SEASON_MAX = 2035
PLAYER_MIN_AGE_AT_SEASON = 15
PLAYER_MAX_AGE_AT_SEASON = 50

SOCCER_DOMAIN_RULE_TYPES = {"position_domain"}
SOCCER_FORMAT_RULE_TYPES = {
    "name_pattern", "surname_pattern", "manager_name_pattern",
    "team_name_pattern", "birthplace_location_pattern", "city_location_pattern",
    "stadium_name_pattern", "pattern", "generic_regex", "rare_pattern"
}
SOCCER_NUMERIC_RULE_TYPES = {"birthyear_numeric_range", "season_numeric_range"}
SOCCER_CONSISTENCY_RULE_TYPES = {
    "birthyear_season_age_consistency",
    "team_context_dominant",
    "team_season_manager_consistency_hint",
    "city_stadium_consistency_hint",
}
SOCCER_FD_RULE_TYPES = {"functional_dependency", "soft_functional_dependency"}


# ============================================================
# 2. 通用工具函数
# ============================================================

def normalize_missing(x: Any) -> Any:
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s.lower() in MISSING_TOKENS:
        return np.nan
    return x


def read_csv_safe(path: str, required: bool = True) -> pd.DataFrame:
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(f"找不到输入文件: {path}")
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, low_memory=False)


def to_numeric_series(s: pd.Series, default: float = 0.0) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan).fillna(default)


def to_float_value(x: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(x):
            return default
        v = float(x)
        if np.isfinite(v):
            return v
        return default
    except Exception:
        return default


def to_int_flag(s: pd.Series, default: int = 0) -> pd.Series:
    if s is None:
        return pd.Series(dtype=int)
    if s.dtype == bool:
        return s.astype(int)
    ss = s.astype(str).str.strip().str.lower()
    return ss.map({
        "1": 1, "true": 1, "yes": 1, "y": 1, "t": 1,
        "0": 0, "false": 0, "no": 0, "n": 0, "f": 0,
        "nan": default, "none": default, "": default,
    }).fillna(pd.to_numeric(s, errors="coerce").fillna(default)).astype(int)


def parse_pipe_set(x: Any) -> set:
    if pd.isna(x) or x is None:
        return set()
    s = str(x).strip()
    if not s:
        return set()
    return {p.strip() for p in re.split(r"[|,;]+", s) if p.strip()}


def parse_int_like(x: Any) -> Optional[int]:
    if x is None or pd.isna(x):
        return None
    s = str(x).strip().replace(",", "")
    if re.fullmatch(r"[+-]?\d+(\.0+)?", s):
        try:
            return int(float(s))
        except Exception:
            return None
    return None


def canonical_position(x: Any) -> Optional[str]:
    if x is None or pd.isna(x):
        return None
    s = str(x).strip()
    if s in POSITION_CANONICAL_MAP:
        return POSITION_CANONICAL_MAP[s]
    if s in VALID_POSITIONS:
        return s
    return None


def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    aa = to_numeric_series(a, 0.0)
    bb = to_numeric_series(b, 0.0)
    out = aa / bb.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def bucketize_numeric(s: pd.Series, bins: List[float], labels: List[str], default: str) -> pd.Series:
    num = pd.to_numeric(s, errors="coerce")
    out = pd.cut(num, bins=bins, labels=labels, include_lowest=True)
    return out.astype(object).where(out.notna(), default).astype(str)


# ============================================================
# 3. 加载与清洗输入
# ============================================================

def load_full_candidate_csv(path: str) -> pd.DataFrame:
    df = read_csv_safe(path, required=True)
    if "row_id" not in df.columns or "column" not in df.columns:
        raise ValueError("全候选上下文文件必须包含 row_id 和 column 字段")
    if "value" not in df.columns:
        df["value"] = np.nan

    df["row_id"] = to_numeric_series(df["row_id"], -1).astype(int)
    df["column"] = df["column"].astype(str).str.strip()
    df["value"] = df["value"].map(normalize_missing)

    # 基础数值字段兜底
    numeric_defaults = {
        "violation_count": 0,
        "conflict_score": 0.0,
        "value_frequency": 0,
        "value_frequency_rank": 999999,
        "neighbor_majority_ratio": 0.0,
        "prior_error_probability": 0.0,
        "posterior_error_probability": 0.0,
        "strong_rule_count": 0,
        "candidate_generation_rule_count": 0,
        "fd_like_count": 0,
        "context_rule_count": 0,
        "global_rule_count": 0,
        "rare_value_count": 0,
        "typo_rule_count": 0,
        "pattern_rule_count": 0,
        "schema_rule_count": 0,
        "soccer_domain_rule_count": 0,
        "soccer_format_rule_count": 0,
        "soccer_numeric_rule_count": 0,
        "soccer_consistency_rule_count": 0,
        "soccer_profile_inconsistency_count": 0,
        "soccer_consistency_score": 0.0,
        "year_value": np.nan,
        "row_birthyear_int": np.nan,
        "row_season_int": np.nan,
        "row_age_at_season": np.nan,
        "row_missing_count": 0,
        "row_non_missing_count": 0,
    }
    for c, default in numeric_defaults.items():
        if c not in df.columns:
            df[c] = default
        df[c] = to_numeric_series(df[c], default if not pd.isna(default) else np.nan)

    # 兼容 soccer context 脚本中为了适配旧 adult 训练链保留的字段
    compat_defaults = {
        "adult_domain_rule_count": "soccer_domain_rule_count",
        "adult_format_rule_count": "soccer_format_rule_count",
        "adult_consistency_rule_count": "soccer_consistency_rule_count",
        "adult_profile_inconsistency_count": "soccer_profile_inconsistency_count",
        "adult_consistency_score": "soccer_consistency_score",
        "adult_evidence_flags": "soccer_evidence_flags",
    }
    for dst, src in compat_defaults.items():
        if dst not in df.columns:
            df[dst] = df[src] if src in df.columns else 0

    text_defaults = {
        "semantic_type": "unknown",
        "detected_type": "unknown",
        "main_rule_type": "none",
        "main_usage_role": "none",
        "neighbor_majority_value": "",
        "current_value_pattern": "NULL",
        "dominant_pattern": "NULL",
        "soccer_value_bucket": "unknown",
        "canonical_position": "",
        "row_position_canonical": "",
        "soccer_evidence_flags": "",
        "adult_evidence_flags": "",
    }
    for c, default in text_defaults.items():
        if c not in df.columns:
            df[c] = default
        df[c] = df[c].fillna(default)

    if "domain_valid" not in df.columns:
        df["domain_valid"] = np.nan

    return df


def load_clustered_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        if ALLOW_MISSING_CLUSTER_FILE:
            print(f"[WARN] 未找到聚类文件 {path}，将只用全候选上下文构建推理集。")
            return pd.DataFrame(columns=["row_id", "column"])
        raise FileNotFoundError(path)

    df = read_csv_safe(path, required=True)
    if "row_id" not in df.columns or "column" not in df.columns:
        raise ValueError("聚类采样文件必须包含 row_id 和 column 字段")

    df["row_id"] = to_numeric_series(df["row_id"], -1).astype(int)
    df["column"] = df["column"].astype(str).str.strip()
    return collapse_duplicate_columns(df)


# ============================================================
# 4. 合并聚类字段
# ============================================================

def first_non_empty_series(input_df: pd.DataFrame, col: str) -> pd.Series:
    matched = [c for c in input_df.columns if c == col]
    if len(matched) == 1:
        return input_df[col]

    # pandas 在重复列名时 input_df[col] 会返回 DataFrame。
    sub = input_df.loc[:, matched]
    out = pd.Series([np.nan] * len(input_df), index=input_df.index, dtype=object)
    for c in sub.columns:
        s = sub[c]
        if isinstance(s, pd.DataFrame):
            for j in range(s.shape[1]):
                sj = s.iloc[:, j]
                mask = out.isna() | (out.astype(str).str.strip().isin(["", "nan", "None"]))
                valid = ~sj.isna() & (sj.astype(str).str.strip() != "")
                out.loc[mask & valid] = sj.loc[mask & valid]
        else:
            mask = out.isna() | (out.astype(str).str.strip().isin(["", "nan", "None"]))
            valid = ~s.isna() & (s.astype(str).str.strip() != "")
            out.loc[mask & valid] = s.loc[mask & valid]
    return out


def collapse_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    seen = []
    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        if c in seen:
            continue
        seen.append(c)
        out[c] = first_non_empty_series(df, c) if list(df.columns).count(c) > 1 else df[c]
    return out


def merge_cluster_fields(full_df: pd.DataFrame, clustered_df: pd.DataFrame) -> pd.DataFrame:
    if clustered_df.empty:
        df = full_df.copy()
        for c, default in {
            "bucket_id": "unknown_bucket",
            "cluster_id": -1,
            "dist_to_center": 0.0,
            "sample_role": "not_sampled",
            "is_sampled": 0,
            "signal_priority": "unknown",
        }.items():
            if c not in df.columns:
                df[c] = default
        return df

    keep_cols = [
        "row_id", "column", "bucket_id", "cluster_id", "dist_to_center",
        "sample_role", "is_sampled", "signal_priority",
        "pattern_bucket", "rarity_bucket", "neighbor_bucket",
        "domain_bucket", "soccer_consistency_bucket", "adult_consistency_bucket",
    ]
    keep_cols = [c for c in keep_cols if c in clustered_df.columns]
    small = clustered_df[keep_cols].copy()
    small = small.drop_duplicates(subset=["row_id", "column"], keep="first")

    df = full_df.merge(small, on=["row_id", "column"], how="left", suffixes=("", "_cluster"))

    for base in ["pattern_bucket", "rarity_bucket", "neighbor_bucket", "domain_bucket", "soccer_consistency_bucket", "adult_consistency_bucket"]:
        c2 = base + "_cluster"
        if c2 in df.columns:
            if base in df.columns:
                df[base] = df[base].where(df[base].notna() & (df[base].astype(str).str.strip() != ""), df[c2])
            else:
                df[base] = df[c2]
            df = df.drop(columns=[c2])

    defaults = {
        "bucket_id": "unknown_bucket",
        "cluster_id": -1,
        "dist_to_center": 0.0,
        "sample_role": "not_sampled",
        "is_sampled": 0,
        "signal_priority": "unknown",
    }
    for c, default in defaults.items():
        if c not in df.columns:
            df[c] = default
        df[c] = df[c].fillna(default)

    return df


# ============================================================
# 5. bucket 与对齐字段
# ============================================================

def fill_bucket_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "pattern_bucket" not in df.columns:
        cur = df.get("current_value_pattern", pd.Series(["NULL"] * len(df)))
        dom = df.get("dominant_pattern", pd.Series(["NULL"] * len(df)))
        df["pattern_bucket"] = np.where(cur.astype(str) == dom.astype(str), "pattern_match", "pattern_mismatch")
    df["pattern_bucket"] = df["pattern_bucket"].fillna("unknown_pattern")

    if "rarity_bucket" not in df.columns:
        freq = to_numeric_series(df.get("value_frequency", pd.Series([0] * len(df))), 0)
        df["rarity_bucket"] = np.select(
            [freq <= 1, freq <= 3, freq <= 10],
            ["rare_1", "rare_2_3", "rare_4_10"],
            default="common"
        )
    df["rarity_bucket"] = df["rarity_bucket"].fillna("unknown_rarity")

    if "neighbor_bucket" not in df.columns:
        ratio = to_numeric_series(df.get("neighbor_majority_ratio", pd.Series([0] * len(df))), 0)
        neigh_val = df.get("neighbor_majority_value", pd.Series([""] * len(df))).astype(str)
        val = df.get("value", pd.Series([""] * len(df))).astype(str)
        df["neighbor_bucket"] = np.select(
            [(ratio >= 0.8) & (neigh_val != val), (ratio >= 0.6) & (neigh_val != val), ratio > 0],
            ["high_neighbor_disagree", "medium_neighbor_disagree", "weak_neighbor_signal"],
            default="no_neighbor_signal"
        )
    df["neighbor_bucket"] = df["neighbor_bucket"].fillna("no_neighbor_signal")

    if "domain_bucket" not in df.columns:
        dom_valid = df.get("domain_valid", pd.Series([np.nan] * len(df)))
        dom_txt = dom_valid.astype(str).str.lower()
        df["domain_bucket"] = np.where(dom_txt.isin(["false", "0"]), "domain_invalid", "domain_valid_or_unknown")
    df["domain_bucket"] = df["domain_bucket"].fillna("domain_valid_or_unknown")

    if "soccer_consistency_bucket" not in df.columns:
        score = to_numeric_series(df.get("soccer_consistency_score", pd.Series([0] * len(df))), 0)
        df["soccer_consistency_bucket"] = np.select(
            [score >= 1.5, score >= 0.8, score > 0],
            ["high_soccer_consistency", "medium_soccer_consistency", "weak_soccer_consistency"],
            default="no_soccer_consistency"
        )
    df["soccer_consistency_bucket"] = df["soccer_consistency_bucket"].fillna("no_soccer_consistency")

    # 旧 adult 字段兼容
    if "adult_consistency_bucket" not in df.columns:
        df["adult_consistency_bucket"] = df["soccer_consistency_bucket"]
    if "age_sampling_bucket" not in df.columns:
        df["age_sampling_bucket"] = "soccer_not_applicable"
    if "hours_sampling_bucket" not in df.columns:
        df["hours_sampling_bucket"] = "soccer_not_applicable"
    if "adult_v2_attribution_bucket" not in df.columns:
        df["adult_v2_attribution_bucket"] = df.get("soccer_attribution_bucket", "soccer_not_applicable")

    return df


def add_alignment_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 推理集没有标签，但保留字段，避免训练/推理脚本读取时报错。
    if "label" not in df.columns:
        df["label"] = "unknown"
    if "label_binary" not in df.columns:
        df["label_binary"] = np.nan
    if "label_source" not in df.columns:
        df["label_source"] = "unlabeled_inference"
    if "sample_weight" not in df.columns:
        df["sample_weight"] = 1.0
    if "is_outside_candidate" not in df.columns:
        df["is_outside_candidate"] = 0
    if "propagation_confidence" not in df.columns:
        df["propagation_confidence"] = np.nan

    # 错误类型/理由字段兼容 LLM label 文件格式。
    for c in ["error_type", "reason_short", "reason_detailed", "suggested_correct_value", "needs_human_review", "evidence_used"]:
        if c not in df.columns:
            df[c] = ""

    # adult/time 旧链字段兼容
    compat_defaults = {
        "age_lower": np.nan, "age_upper": np.nan, "hours_lower": np.nan, "hours_upper": np.nan,
        "canonical_income": "", "education_order": np.nan, "adult_value_bucket": "soccer_not_applicable",
        "time_window_rule_count": 0, "time_value_minutes": np.nan,
        "time_window_bucket": "soccer_not_applicable", "time_value_bucket": "soccer_not_applicable",
    }
    for c, default in compat_defaults.items():
        if c not in df.columns:
            df[c] = default

    return df


# ============================================================
# 6. Soccer 派生特征
# ============================================================

def infer_rule_type_sets_from_row(row: pd.Series) -> Dict[str, int]:
    rtypes = set()
    for c in ["rule_types", "main_rule_type", "soccer_evidence_flags", "adult_evidence_flags"]:
        if c in row.index:
            rtypes |= parse_pipe_set(row.get(c))

    return {
        "has_domain": int(bool(rtypes & SOCCER_DOMAIN_RULE_TYPES) or to_float_value(row.get("soccer_domain_rule_count"), 0) > 0),
        "has_format": int(bool(rtypes & SOCCER_FORMAT_RULE_TYPES) or to_float_value(row.get("soccer_format_rule_count"), 0) > 0),
        "has_numeric": int(bool(rtypes & SOCCER_NUMERIC_RULE_TYPES) or to_float_value(row.get("soccer_numeric_rule_count"), 0) > 0),
        "has_consistency": int(bool(rtypes & SOCCER_CONSISTENCY_RULE_TYPES) or to_float_value(row.get("soccer_consistency_rule_count"), 0) > 0),
        "has_fd": int(bool(rtypes & SOCCER_FD_RULE_TYPES) or to_float_value(row.get("fd_like_count"), 0) > 0),
    }


def add_soccer_value_features_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "year_value" not in df.columns or df["year_value"].isna().all():
        df["year_value"] = [parse_int_like(v) if c in {"birthyear", "season"} else np.nan for c, v in zip(df["column"], df["value"])]
    else:
        df["year_value"] = to_numeric_series(df["year_value"], np.nan)

    if "canonical_position" not in df.columns:
        df["canonical_position"] = [canonical_position(v) if c == "position" else "" for c, v in zip(df["column"], df["value"])]
    df["canonical_position"] = df["canonical_position"].fillna("")

    if "domain_valid" not in df.columns:
        df["domain_valid"] = np.nan

    # 若上下文没给 attribution 特征，则在这里尽量从当前值推断。
    birthyear_num = to_numeric_series(df["year_value"], np.nan)
    col = df["column"].astype(str)

    if "birthyear_value_invalid" not in df.columns:
        df["birthyear_value_invalid"] = ((col == "birthyear") & (birthyear_num.notna()) & ((birthyear_num < BIRTHYEAR_MIN) | (birthyear_num > BIRTHYEAR_MAX))).astype(int)
    if "season_value_invalid" not in df.columns:
        df["season_value_invalid"] = ((col == "season") & (birthyear_num.notna()) & ((birthyear_num < SEASON_MIN) | (birthyear_num > SEASON_MAX))).astype(int)
    if "position_value_invalid" not in df.columns:
        df["position_value_invalid"] = ((col == "position") & df["canonical_position"].astype(str).eq("")).astype(int)
    if "position_needs_canonicalization" not in df.columns:
        df["position_needs_canonicalization"] = ((col == "position") & df["value"].astype(str).isin(POSITION_CANONICAL_MAP.keys())).astype(int)

    if "row_birthyear_int" not in df.columns:
        df["row_birthyear_int"] = np.nan
    if "row_season_int" not in df.columns:
        df["row_season_int"] = np.nan
    if "row_age_at_season" not in df.columns:
        df["row_age_at_season"] = to_numeric_series(df["row_season_int"], np.nan) - to_numeric_series(df["row_birthyear_int"], np.nan)

    df["row_birthyear_int"] = to_numeric_series(df["row_birthyear_int"], np.nan)
    df["row_season_int"] = to_numeric_series(df["row_season_int"], np.nan)
    df["row_age_at_season"] = to_numeric_series(df["row_age_at_season"], np.nan)

    if "birthyear_season_age_conflict" not in df.columns:
        age = df["row_age_at_season"]
        df["birthyear_season_age_conflict"] = (age.notna() & ((age < PLAYER_MIN_AGE_AT_SEASON) | (age > PLAYER_MAX_AGE_AT_SEASON))).astype(int)

    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = add_soccer_value_features_if_missing(df)

    numeric_cols = [
        "violation_count", "conflict_score", "value_frequency", "value_frequency_rank",
        "neighbor_majority_ratio", "prior_error_probability", "posterior_error_probability",
        "strong_rule_count", "candidate_generation_rule_count", "fd_like_count",
        "context_rule_count", "global_rule_count", "rare_value_count", "typo_rule_count",
        "pattern_rule_count", "schema_rule_count", "soccer_domain_rule_count",
        "soccer_format_rule_count", "soccer_numeric_rule_count", "soccer_consistency_rule_count",
        "soccer_profile_inconsistency_count", "soccer_consistency_score",
        "adult_domain_rule_count", "adult_format_rule_count", "adult_consistency_rule_count",
        "adult_profile_inconsistency_count", "adult_consistency_score",
    ]
    for c in numeric_cols:
        if c not in df.columns:
            df[c] = 0
        df[c] = to_numeric_series(df[c], 0.0)

    col = df["column"].astype(str)
    value = df["value"].astype(str)
    neigh_value = df.get("neighbor_majority_value", pd.Series([""] * len(df))).astype(str)
    neigh_ratio = to_numeric_series(df.get("neighbor_majority_ratio", pd.Series([0] * len(df))), 0.0)

    df["target_is_primary_column"] = col.isin(PRIMARY_TARGET_COLUMNS).astype(int)
    df["target_is_context_only_column"] = col.isin(CONTEXT_ONLY_COLUMNS).astype(int)

    df["candidate_rule_ratio"] = safe_div(df["candidate_generation_rule_count"], df["violation_count"])
    df["rule_total_count"] = df[["strong_rule_count", "candidate_generation_rule_count", "fd_like_count", "context_rule_count", "schema_rule_count"]].sum(axis=1)
    df["strong_rule_ratio"] = safe_div(df["strong_rule_count"], df["rule_total_count"])
    df["context_rule_ratio"] = safe_div(df["context_rule_count"], df["rule_total_count"])
    df["fd_like_ratio"] = safe_div(df["fd_like_count"], df["rule_total_count"])
    df["global_rule_ratio"] = safe_div(df["global_rule_count"], df["rule_total_count"])

    df["soccer_rule_total_count"] = df[[
        "soccer_domain_rule_count", "soccer_format_rule_count",
        "soccer_numeric_rule_count", "soccer_consistency_rule_count"
    ]].sum(axis=1)
    df["soccer_domain_rule_ratio"] = safe_div(df["soccer_domain_rule_count"], df["soccer_rule_total_count"])
    df["soccer_format_rule_ratio"] = safe_div(df["soccer_format_rule_count"], df["soccer_rule_total_count"])
    df["soccer_numeric_rule_ratio"] = safe_div(df["soccer_numeric_rule_count"], df["soccer_rule_total_count"])
    df["soccer_consistency_rule_ratio"] = safe_div(df["soccer_consistency_rule_count"], df["soccer_rule_total_count"])

    # 兼容 adult 旧字段
    df["adult_rule_total_count"] = df[["adult_domain_rule_count", "adult_format_rule_count", "adult_consistency_rule_count"]].sum(axis=1)
    df["adult_domain_rule_ratio"] = safe_div(df["adult_domain_rule_count"], df["adult_rule_total_count"])
    df["adult_format_rule_ratio"] = safe_div(df["adult_format_rule_count"], df["adult_rule_total_count"])
    df["adult_consistency_rule_ratio"] = safe_div(df["adult_consistency_rule_count"], df["adult_rule_total_count"])

    df["has_neighbor_signal"] = (neigh_ratio > 0).astype(int)
    df["high_neighbor_consistency"] = (neigh_ratio >= 0.8).astype(int)
    df["neighbor_disagree"] = ((neigh_ratio > 0) & (neigh_value != value)).astype(int)
    df["is_equal_to_neighbor_majority"] = ((neigh_ratio > 0) & (neigh_value == value)).astype(int)
    df["neighbor_agreement_score"] = np.where(df["is_equal_to_neighbor_majority"].eq(1), neigh_ratio, 0.0)

    freq = to_numeric_series(df["value_frequency"], 0)
    df["is_rare_value"] = (freq <= 2).astype(int)

    dom_txt = df.get("domain_valid", pd.Series([np.nan] * len(df))).astype(str).str.lower()
    df["domain_invalid_flag"] = dom_txt.isin(["false", "0"]).astype(int)
    df["soccer_consistency_flag"] = ((df["soccer_consistency_rule_count"] > 0) | (df["soccer_consistency_score"] > 0)).astype(int)
    df["adult_consistency_flag"] = df["soccer_consistency_flag"]
    df["high_soccer_consistency_score"] = (df["soccer_consistency_score"] >= 1.0).astype(int)
    df["high_adult_consistency_score"] = df["high_soccer_consistency_score"]

    # 如果 rule count 被上游填好，用它；否则从 rule_types/main_rule_type/evidence_flags 中补推断。
    inferred = df.apply(infer_rule_type_sets_from_row, axis=1, result_type="expand")
    df["has_soccer_domain_signal"] = ((df["soccer_domain_rule_count"] > 0) | inferred["has_domain"].astype(bool)).astype(int)
    df["has_soccer_format_signal"] = ((df["soccer_format_rule_count"] > 0) | inferred["has_format"].astype(bool)).astype(int)
    df["has_soccer_numeric_signal"] = ((df["soccer_numeric_rule_count"] > 0) | inferred["has_numeric"].astype(bool)).astype(int)
    df["has_soccer_consistency_signal"] = ((df["soccer_consistency_rule_count"] > 0) | inferred["has_consistency"].astype(bool)).astype(int)
    df["has_fd_like_signal"] = ((df["fd_like_count"] > 0) | inferred["has_fd"].astype(bool)).astype(int)

    df["has_adult_domain_signal"] = df["has_soccer_domain_signal"]
    df["has_adult_format_signal"] = df["has_soccer_format_signal"]
    df["has_adult_consistency_signal"] = df["has_soccer_consistency_signal"]

    df["birthyear_value_invalid"] = to_int_flag(df["birthyear_value_invalid"])
    df["season_value_invalid"] = to_int_flag(df["season_value_invalid"])
    df["position_value_invalid"] = to_int_flag(df["position_value_invalid"])
    df["position_needs_canonicalization"] = to_int_flag(df["position_needs_canonicalization"])
    df["birthyear_season_age_conflict"] = to_int_flag(df["birthyear_season_age_conflict"])

    if "team_context_conflict" not in df.columns:
        df["team_context_conflict"] = ((df["context_rule_count"] > 0) | df.get("main_rule_type", "").astype(str).eq("team_context_dominant")).astype(int)
    else:
        df["team_context_conflict"] = to_int_flag(df["team_context_conflict"])

    if "format_rule_signal" not in df.columns:
        df["format_rule_signal"] = df["has_soccer_format_signal"]
    else:
        df["format_rule_signal"] = to_int_flag(df["format_rule_signal"])

    if "fd_like_signal" not in df.columns:
        df["fd_like_signal"] = df["has_fd_like_signal"]
    else:
        df["fd_like_signal"] = to_int_flag(df["fd_like_signal"])

    df["position_invalid_or_normalization"] = ((df["position_value_invalid"] == 1) | (df["position_needs_canonicalization"] == 1)).astype(int)
    df["has_birthyear_or_season_signal"] = ((df["birthyear_value_invalid"] == 1) | (df["season_value_invalid"] == 1) | (df["birthyear_season_age_conflict"] == 1) | (df["has_soccer_numeric_signal"] == 1)).astype(int)
    df["has_team_context_signal"] = ((df["team_context_conflict"] == 1) | (df["context_rule_count"] > 0)).astype(int)

    age = to_numeric_series(df["row_age_at_season"], np.nan)
    df["birthyear_season_gap"] = age
    df["is_young_player_age"] = (age.notna() & (age < PLAYER_MIN_AGE_AT_SEASON)).astype(int)
    df["is_old_player_age"] = (age.notna() & (age > PLAYER_MAX_AGE_AT_SEASON)).astype(int)

    # adult 旧数值字段兜底
    df["age_span"] = np.nan
    df["hours_span"] = np.nan
    df["is_minor_age_bucket"] = 0
    df["is_high_hours_bucket"] = 0
    df["relationship_v2_conflict_count"] = 0
    df["has_relationship_v2_signal"] = 0
    df["has_education_v2_signal"] = 0
    df["adult_v2_signal_strength"] = 0
    df["adult_v2_primary_target_signal"] = 0

    # rule_strength_sum 是不依赖 GT 的启发式证据强度，用于后续推理 gate / verifier。
    df["rule_strength_sum"] = (
        2.0 * df["has_soccer_domain_signal"]
        + 1.6 * df["has_soccer_numeric_signal"]
        + 1.3 * df["has_soccer_format_signal"]
        + 1.4 * df["has_soccer_consistency_signal"]
        + 0.5 * df["has_fd_like_signal"]
        + 0.7 * df["neighbor_disagree"] * neigh_ratio
        + 0.4 * df["is_rare_value"]
    ).round(6)

    df["candidate_rule_dominant"] = (df["candidate_rule_ratio"] >= 0.5).astype(int)

    df["soccer_signal_strength"] = (
        df["rule_strength_sum"]
        + df["posterior_error_probability"].clip(0, 1)
        + 0.5 * df["target_is_primary_column"]
    ).round(6)

    # primary_target_signal 表示“证据指向主检测列”，不是最终预测标签。
    df["soccer_primary_target_signal"] = (
        (df["target_is_primary_column"] == 1)
        & (
            (df["has_soccer_domain_signal"] == 1)
            | (df["has_soccer_numeric_signal"] == 1)
            | (df["has_soccer_format_signal"] == 1)
            | (df["has_soccer_consistency_signal"] == 1)
            | (df["neighbor_disagree"] == 1)
            | (df["has_fd_like_signal"] == 1)
        )
    ).astype(int)

    # context-only weak signal 主要用于推理阶段过滤 name/surname 的弱统计噪声。
    df["is_context_only_weak_signal"] = (
        (df["target_is_context_only_column"] == 1)
        & (df["has_soccer_domain_signal"] == 0)
        & (df["has_soccer_numeric_signal"] == 0)
        & (df["has_soccer_format_signal"] == 0)
        & (df["has_soccer_consistency_signal"] == 0)
    ).astype(int)

    if "soccer_attribution_bucket" not in df.columns:
        df["soccer_attribution_bucket"] = "primary_target_other"
    df["soccer_attribution_bucket"] = df["soccer_attribution_bucket"].fillna("primary_target_other")
    df.loc[df["birthyear_value_invalid"].eq(1), "soccer_attribution_bucket"] = "birthyear_invalid"
    df.loc[df["season_value_invalid"].eq(1), "soccer_attribution_bucket"] = "season_invalid"
    df.loc[df["position_invalid_or_normalization"].eq(1), "soccer_attribution_bucket"] = "position_invalid_or_normalization"
    df.loc[(df["birthyear_season_age_conflict"].eq(1)) & (~df["soccer_attribution_bucket"].isin(["birthyear_invalid", "season_invalid"])), "soccer_attribution_bucket"] = "birthyear_season_age_conflict"
    df.loc[(df["has_team_context_signal"].eq(1)) & (df["soccer_attribution_bucket"].eq("primary_target_other")), "soccer_attribution_bucket"] = "team_context_conflict"
    df.loc[(df["has_fd_like_signal"].eq(1)) & (df["soccer_attribution_bucket"].eq("primary_target_other")), "soccer_attribution_bucket"] = "fd_like_conflict"
    df.loc[df["target_is_context_only_column"].eq(1), "soccer_attribution_bucket"] = "context_only_other"

    # 再次同步兼容字段
    df["adult_profile_inconsistency_count"] = df["soccer_profile_inconsistency_count"]
    df["adult_consistency_score"] = df["soccer_consistency_score"]
    df["adult_consistency_flag"] = df["soccer_consistency_flag"]
    df["adult_v2_attribution_bucket"] = df["soccer_attribution_bucket"]

    return df


# ============================================================
# 7. 输出列顺序
# ============================================================

def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    preferred_cols = [
        "row_id", "column", "value",

        "label", "label_binary", "label_source", "sample_weight",
        "is_outside_candidate", "propagation_confidence",

        "semantic_type", "detected_type", "violation_count", "conflict_score",
        "value_frequency", "value_frequency_rank",

        "neighbor_majority_value", "neighbor_majority_ratio",
        "prior_error_probability", "posterior_error_probability",

        "main_rule_type", "main_usage_role",
        "strong_rule_count", "candidate_generation_rule_count",
        "fd_like_count", "context_rule_count", "global_rule_count",
        "rare_value_count", "typo_rule_count", "pattern_rule_count", "schema_rule_count",

        "soccer_domain_rule_count", "soccer_format_rule_count",
        "soccer_numeric_rule_count", "soccer_consistency_rule_count",
        "soccer_profile_inconsistency_count", "soccer_consistency_score", "soccer_evidence_flags",

        "year_value", "canonical_position", "domain_valid", "soccer_value_bucket",
        "row_birthyear_int", "row_season_int", "row_age_at_season", "row_position_canonical",

        "target_is_primary_column", "target_is_context_only_column",
        "birthyear_value_invalid", "season_value_invalid",
        "position_value_invalid", "position_needs_canonicalization",
        "birthyear_season_age_conflict", "team_context_conflict",
        "format_rule_signal", "fd_like_signal", "soccer_attribution_bucket",

        "candidate_rule_ratio", "has_neighbor_signal", "high_neighbor_consistency",
        "is_rare_value", "neighbor_disagree", "is_equal_to_neighbor_majority",
        "neighbor_agreement_score", "rule_total_count", "strong_rule_ratio",
        "context_rule_ratio", "fd_like_ratio", "global_rule_ratio",

        "soccer_rule_total_count", "soccer_domain_rule_ratio",
        "soccer_format_rule_ratio", "soccer_numeric_rule_ratio",
        "soccer_consistency_rule_ratio", "rule_strength_sum", "candidate_rule_dominant",

        "domain_invalid_flag", "soccer_consistency_flag", "high_soccer_consistency_score",
        "has_soccer_domain_signal", "has_soccer_format_signal",
        "has_soccer_numeric_signal", "has_soccer_consistency_signal",
        "birthyear_season_gap", "is_young_player_age", "is_old_player_age",

        "position_invalid_or_normalization", "has_birthyear_or_season_signal",
        "has_team_context_signal", "has_fd_like_signal",
        "is_context_only_weak_signal", "soccer_signal_strength", "soccer_primary_target_signal",

        "pattern_bucket", "rarity_bucket", "neighbor_bucket", "domain_bucket", "soccer_consistency_bucket",

        # 兼容旧 adult/time 字段
        "adult_domain_rule_count", "adult_format_rule_count", "adult_consistency_rule_count",
        "age_lower", "age_upper", "hours_lower", "hours_upper",
        "canonical_income", "education_order", "adult_value_bucket",
        "adult_profile_inconsistency_count", "adult_consistency_score", "adult_evidence_flags",
        "adult_consistency_bucket", "age_sampling_bucket", "hours_sampling_bucket",
        "adult_v2_attribution_bucket", "time_window_rule_count", "time_value_minutes",
        "time_window_bucket", "time_value_bucket",
        "adult_rule_total_count", "adult_domain_rule_ratio", "adult_format_rule_ratio",
        "adult_consistency_rule_ratio", "adult_consistency_flag", "high_adult_consistency_score",
        "has_adult_domain_signal", "has_adult_format_signal", "has_adult_consistency_signal",
        "age_span", "hours_span", "is_minor_age_bucket", "is_high_hours_bucket",
        "relationship_v2_conflict_count", "has_relationship_v2_signal", "has_education_v2_signal",
        "adult_v2_signal_strength", "adult_v2_primary_target_signal",

        "bucket_id", "cluster_id", "dist_to_center", "sample_role", "is_sampled", "signal_priority",

        "error_type", "reason_short", "reason_detailed", "suggested_correct_value",
        "needs_human_review", "evidence_used",
    ]

    exist = [c for c in preferred_cols if c in df.columns]
    extra = [c for c in df.columns if c not in exist]
    return df[exist + extra].copy()


# ============================================================
# 8. 主流程
# ============================================================

def main():
    print("[1/5] 读取全候选上下文文件...")
    full_df = load_full_candidate_csv(INPUT_FULL_CANDIDATE_CSV)
    print(f"  full candidate rows = {len(full_df)}, cols = {len(full_df.columns)}")

    print("[2/5] 读取聚类采样结果文件...")
    clustered_df = load_clustered_csv(INPUT_CLUSTERED_CSV)
    print(f"  clustered rows = {len(clustered_df)}, cols = {len(clustered_df.columns)}")

    print("[3/5] merge 聚类字段...")
    df = merge_cluster_fields(full_df, clustered_df)

    print("[4/5] 补 bucket / 对齐列 / 派生 soccer v2 特征...")
    df = fill_bucket_fields(df)
    df = add_alignment_columns(df)
    df = add_derived_features(df)

    print("[5/5] 输出推理对齐文件...")
    df = reorder_columns(df)
    df.to_csv(OUTPUT_INFER_READY_CSV, index=False, encoding="utf-8-sig")

    print(f"[OK] 推理对齐文件已保存: {OUTPUT_INFER_READY_CSV}")
    print(f"[OK] 行数: {len(df)}")
    print(f"[OK] 列数: {len(df.columns)}")

    diag_cols = [
        "target_is_primary_column", "target_is_context_only_column",
        "has_birthyear_or_season_signal", "position_invalid_or_normalization",
        "has_team_context_signal", "has_fd_like_signal",
        "is_context_only_weak_signal", "soccer_primary_target_signal",
    ]
    print("\n===== Soccer v2 推理集诊断 =====")
    for c in diag_cols:
        if c in df.columns:
            print(f"{c}: {int(pd.to_numeric(df[c], errors='coerce').fillna(0).sum())}")

    print("\n===== 预览前5行 =====")
    preview_cols = [
        "row_id", "column", "value", "semantic_type", "violation_count", "conflict_score",
        "pattern_bucket", "rarity_bucket", "neighbor_bucket", "domain_bucket", "soccer_consistency_bucket",
        "bucket_id", "cluster_id", "dist_to_center", "candidate_rule_ratio", "has_neighbor_signal",
        "high_neighbor_consistency", "is_rare_value", "is_equal_to_neighbor_majority", "neighbor_agreement_score",
        "rule_total_count", "soccer_rule_total_count", "rule_strength_sum", "domain_invalid_flag",
        "soccer_consistency_flag", "target_is_primary_column", "target_is_context_only_column",
        "soccer_attribution_bucket", "has_birthyear_or_season_signal", "position_invalid_or_normalization",
        "has_team_context_signal", "soccer_signal_strength", "label_source", "is_outside_candidate",
    ]
    preview_cols = [c for c in preview_cols if c in df.columns]
    print(df[preview_cols].head().to_string(index=False))


if __name__ == "__main__":
    main()

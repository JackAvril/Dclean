#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unified_summarize_full_contexts.py

统一版上下文生成脚本，覆盖：
hospital / flights / beers / rayyan / adult / soccer

设计目标：
1. 抽象统一逻辑，方便后续做消融实验。
2. 保持最终输出文件名和主要输出结构不变：
   - candidate_llm_contexts_*.jsonl
   - candidate_llm_contexts_flat_*.csv
3. JSONL 保持核心结构：
   row_id, column, value, violation_count, conflict_score,
   row_context, column_context, conflict_context,
   similarity_context, bayesian_context, llm_context_text
   adult/soccer 额外保留 adult_consistency_context / soccer_consistency_context
   和 attribution features。
4. CSV 保持各数据集原有主要字段集合，adult/soccer 保留兼容字段。
5. 后续消融可以通过环境变量或命令行关闭部分证据：
   --disable-neighbor-evidence
   --disable-bayesian-evidence
   --disable-dataset-consistency
   --disable-context-text-details

注意：
- 这是抽象统一逻辑版，不是六份原始脚本的逐行拼接。
- 输出文件名、核心字段、后续 build/label/train 可用字段保持一致。
- 如果你要求 bit-level 完全一致，需要再用旧输出做 diff 微调具体 llm_context_text 文案。
"""

import argparse
import json
import math
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd


# ============================================================
# 1. Dataset config
# ============================================================

BASE_MISSING_TOKENS = {
    "", "empty", "null", "none", "nan", "na", "n/a", "missing", "unk",
    "unknown", "?", "not available", "contact airline"
}

@dataclass
class DatasetConfig:
    name: str
    input_table_csv: str
    input_shrunk_csv: str
    output_jsonl: str
    output_csv: str

    missing_tokens: Set[str] = field(default_factory=lambda: set(BASE_MISSING_TOKENS))
    numeric_ratio_threshold: float = 0.90
    categorical_unique_count_threshold: int = 50
    categorical_unique_ratio_threshold: float = 0.08
    id_like_unique_ratio_threshold: float = 0.98
    code_like_avg_len_threshold: float = 20.0

    top_k_column_values: int = 10
    top_k_alt_values: int = 10
    top_k_similar_rows: int = 5
    top_k_row_fields_for_llm: int = 12
    profile_topk_values: int = 5

    w_rule: float = 2.0
    w_rarity: float = 1.0
    w_pattern: float = 0.8
    w_neighbor: float = 1.2
    w_dataset_consistency: float = 0.0

    explicit_semantic_types: Dict[str, str] = field(default_factory=dict)
    categorical_columns: Set[str] = field(default_factory=set)
    numeric_semantic_columns: Set[str] = field(default_factory=set)
    time_columns: Set[str] = field(default_factory=set)
    date_columns: Set[str] = field(default_factory=set)
    numeric_metadata_columns: Set[str] = field(default_factory=set)

    primary_target_columns: Set[str] = field(default_factory=set)
    context_only_columns: Set[str] = field(default_factory=set)

    tmp_part_dir: str = ""
    include_similar_row_values: bool = True
    similarity_batch_size: int = 2048


def build_configs() -> Dict[str, DatasetConfig]:
    cfgs: Dict[str, DatasetConfig] = {}

    cfgs["hospital"] = DatasetConfig(
        name="hospital",
        input_table_csv="/mnt/mydata/dq/projects/Splittree/hospital/hospital_dirty_no_address23.csv",
        input_shrunk_csv="candidate_cells_v2.csv",
        output_jsonl="candidate_llm_contexts_v2.jsonl",
        output_csv="candidate_llm_contexts_flat_v2.csv",
        missing_tokens={"", "empty", "null", "none", "nan", "na", "n/a", "missing", "unk"},
        categorical_unique_count_threshold=30,
        categorical_unique_ratio_threshold=0.05,
        code_like_avg_len_threshold=20,
    )

    cfgs["flights"] = DatasetConfig(
        name="flights",
        input_table_csv="/mnt/mydata/dq/projects/Splittree/flights/flights_dirty.csv",
        input_shrunk_csv="candidate_cells_flights.csv",
        output_jsonl="candidate_llm_contexts_flights.jsonl",
        output_csv="candidate_llm_contexts_flat_flights.csv",
        missing_tokens={"", "empty", "null", "none", "nan", "na", "n/a", "missing", "unk", "not available", "contact airline"},
        categorical_unique_count_threshold=30,
        categorical_unique_ratio_threshold=0.05,
        code_like_avg_len_threshold=20,
        time_columns={"sched_dep_time", "act_dep_time", "sched_arr_time", "act_arr_time"},
        explicit_semantic_types={
            "sched_dep_time": "time",
            "act_dep_time": "time",
            "sched_arr_time": "time",
            "act_arr_time": "time",
        },
    )

    cfgs["beers"] = DatasetConfig(
        name="beers",
        input_table_csv="/mnt/mydata/dq/projects/Splittree/beers/beers_dirty.csv",
        input_shrunk_csv="candidate_cells_beers.csv",
        output_jsonl="candidate_llm_contexts_beers.jsonl",
        output_csv="candidate_llm_contexts_flat_beers.csv",
        missing_tokens={"", "empty", "null", "none", "nan", "na", "n/a", "missing", "unk", "not available", "contact airline"},
        categorical_unique_count_threshold=40,
        categorical_unique_ratio_threshold=0.08,
        code_like_avg_len_threshold=20,
        numeric_semantic_columns={"abv", "ibu", "ounces"},
        explicit_semantic_types={
            "abv": "numeric",
            "ibu": "numeric",
            "ounces": "numeric",
            "state": "state_code",
            "city": "city",
            "name": "beer_name",
            "brewery_name": "brewery_name",
            "style": "beer_style",
        },
    )

    cfgs["rayyan"] = DatasetConfig(
        name="rayyan",
        input_table_csv="/mnt/mydata/dq/projects/Splittree/rayyan/rayyan_dirty.csv",
        input_shrunk_csv="candidate_cells_rayyan.csv",
        output_jsonl="candidate_llm_contexts_rayyan.jsonl",
        output_csv="candidate_llm_contexts_flat_rayyan.csv",
        missing_tokens={"", "empty", "null", "none", "nan", "na", "n/a", "missing", "unk", "not available", "contact airline"},
        categorical_unique_count_threshold=50,
        categorical_unique_ratio_threshold=0.08,
        code_like_avg_len_threshold=20,
        date_columns={"article_jcreated_at"},
        numeric_metadata_columns={"article_jvolumn", "article_jissue"},
        explicit_semantic_types={
            "article_jcreated_at": "date",
            "article_jvolumn": "numeric_metadata",
            "article_jissue": "numeric_metadata",
            "article_language": "language",
            "journal_issn": "issn",
            "article_pagination": "pagination",
            "author_list": "author_list",
        },
    )

    adult_primary = {"sex", "relationship", "education"}
    adult_context_only = {
        "age", "workclass", "maritalstatus", "occupation", "race",
        "hoursperweek", "country", "income"
    }
    cfgs["adult"] = DatasetConfig(
        name="adult",
        input_table_csv="/mnt/mydata/dq/projects/Splittree/adult/adult_dirty.csv",
        input_shrunk_csv="candidate_cells_adult.csv",
        output_jsonl="candidate_llm_contexts_adult.jsonl",
        output_csv="candidate_llm_contexts_flat_adult.csv",
        missing_tokens=set(BASE_MISSING_TOKENS),
        categorical_unique_count_threshold=80,
        categorical_unique_ratio_threshold=0.10,
        code_like_avg_len_threshold=20,
        top_k_row_fields_for_llm=11,
        w_dataset_consistency=1.4,
        categorical_columns={
            "workclass", "education", "maritalstatus", "occupation", "relationship",
            "race", "sex", "country", "income"
        },
        primary_target_columns=adult_primary,
        context_only_columns=adult_context_only,
        explicit_semantic_types={
            "age": "age_bucket",
            "hoursperweek": "hours_range",
            "workclass": "workclass_category",
            "education": "education_category",
            "maritalstatus": "maritalstatus_category",
            "occupation": "occupation_category",
            "relationship": "relationship_category",
            "race": "race_category",
            "sex": "sex_category",
            "country": "country_category",
            "income": "income_category",
        },
        tmp_part_dir="_tmp_candidate_llm_contexts_adult_parts",
    )

    soccer_cols = [
        "name", "surname", "birthyear", "birthplace", "position",
        "team", "city", "stadium", "season", "manager"
    ]
    cfgs["soccer"] = DatasetConfig(
        name="soccer",
        input_table_csv="/mnt/mydata/dq/projects/Splittree/soccer/soccer_dirty.csv",
        input_shrunk_csv="candidate_cells_soccer.csv",
        output_jsonl="candidate_llm_contexts_soccer.jsonl",
        output_csv="candidate_llm_contexts_flat_soccer.csv",
        missing_tokens=set(BASE_MISSING_TOKENS),
        categorical_unique_count_threshold=120,
        categorical_unique_ratio_threshold=0.12,
        code_like_avg_len_threshold=35,
        top_k_row_fields_for_llm=10,
        w_rarity=0.8,
        w_dataset_consistency=1.4,
        categorical_columns={"position", "team", "city", "stadium"},
        primary_target_columns=set(soccer_cols),
        context_only_columns=set(),
        explicit_semantic_types={
            "name": "player_given_name",
            "surname": "player_surname",
            "birthyear": "birth_year",
            "birthplace": "birthplace_location",
            "position": "position_category",
            "team": "team_entity",
            "city": "city_location",
            "stadium": "stadium_entity",
            "season": "season_year",
            "manager": "manager_name",
        },
        tmp_part_dir="_tmp_candidate_llm_contexts_soccer_parts",
    )

    return cfgs


CONFIGS = build_configs()


# ============================================================
# 2. Common utils
# ============================================================

_PATTERN_CACHE: Dict[Tuple[str, Any], str] = {}
_CHARSET_CACHE: Dict[Any, Dict[str, float]] = {}
_VALUE_FEATURE_CACHE: Dict[Tuple[str, str, Any], Dict[str, Any]] = {}


def env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def normalize_value(x, cfg: DatasetConfig):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.lower() in cfg.missing_tokens:
        return None
    return s


def normalize_series(s: pd.Series, cfg: DatasetConfig) -> pd.Series:
    return s.map(lambda x: normalize_value(x, cfg))


def maybe_drop_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [c for c in df.columns if not str(c).lower().startswith("unnamed:")]
    return df[keep_cols].copy()


def detect_basic_pattern(value, cfg: DatasetConfig) -> str:
    if value is None:
        return "NULL"
    key = (cfg.name, value)
    if key in _PATTERN_CACHE:
        return _PATTERN_CACHE[key]

    extra = set()
    if cfg.name in {"adult", "soccer"}:
        extra.update({"<", ">"})
    if cfg.name == "rayyan":
        extra.update({"[", "]", "{", "}", '"'})

    symbols = {" ", "-", "_", "/", ".", ",", "(", ")", "%", ":", ";", "&", "'", "+"} | extra

    parts, prev_type, cnt = [], None, 0
    for ch in str(value):
        if ch.isdigit():
            cur = "D"
        elif ch.isalpha():
            cur = "L"
        elif ch in symbols:
            cur = "S"
        else:
            cur = "X"

        if cur == prev_type:
            cnt += 1
        else:
            if prev_type is not None:
                parts.append(f"{prev_type}{cnt}")
            prev_type, cnt = cur, 1

    if prev_type is not None:
        parts.append(f"{prev_type}{cnt}")
    pat = "".join(parts)
    _PATTERN_CACHE[key] = pat
    return pat


def try_parse_numeric_series(series: pd.Series):
    cleaned = (
        series.dropna().astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
    )
    numeric = pd.to_numeric(cleaned, errors="coerce")
    success_ratio = numeric.notna().mean() if len(cleaned) > 0 else 0.0
    return numeric, success_ratio


def try_parse_one_numeric(value):
    if value is None:
        return None
    s = str(value).replace(",", "").replace("%", "")
    try:
        return float(s)
    except Exception:
        return None


def safe_logit(p: float) -> float:
    eps = 1e-6
    p = min(max(float(p), eps), 1 - eps)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    if x >= 40:
        return 1.0
    if x <= -40:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def charset_features(value) -> Dict[str, float]:
    if value in _CHARSET_CACHE:
        return _CHARSET_CACHE[value]
    if value is None or len(str(value)) == 0:
        res = {
            "length": 0,
            "digit_ratio": 0.0,
            "alpha_ratio": 0.0,
            "symbol_ratio": 0.0,
            "upper_ratio": 0.0,
            "lower_ratio": 0.0,
            "space_ratio": 0.0,
        }
        _CHARSET_CACHE[value] = res
        return res

    s = str(value)
    n = len(s)
    digits = sum(ch.isdigit() for ch in s)
    alphas = sum(ch.isalpha() for ch in s)
    uppers = sum(ch.isupper() for ch in s)
    lowers = sum(ch.islower() for ch in s)
    spaces = sum(ch.isspace() for ch in s)
    symbols = n - digits - alphas - spaces
    res = {
        "length": n,
        "digit_ratio": round(digits / n, 6),
        "alpha_ratio": round(alphas / n, 6),
        "symbol_ratio": round(symbols / n, 6),
        "upper_ratio": round(uppers / n, 6),
        "lower_ratio": round(lowers / n, 6),
        "space_ratio": round(spaces / n, 6),
    }
    _CHARSET_CACHE[value] = res
    return res


def value_frequency_dict(series: pd.Series, topk=20):
    cnt = Counter(series.dropna().tolist())
    total = len(series.dropna())
    return [
        {"value": k, "count": v, "ratio": round(v / total, 6) if total > 0 else 0.0}
        for k, v in cnt.most_common(topk)
    ]


def build_value_rank_map(counter: Counter) -> Dict[Any, int]:
    return {v: idx for idx, (v, _) in enumerate(counter.most_common(), start=1)}


def parse_pipe_list(s: Any, sep: str = "|") -> List[str]:
    if pd.isna(s) or s is None:
        return []
    s = str(s).strip()
    return [x.strip() for x in s.split(sep) if x.strip()] if s else []


def parse_reason_list(s: Any) -> List[str]:
    if pd.isna(s) or s is None:
        return []
    s = str(s).strip()
    return [x.strip() for x in s.split("||") if x.strip()] if s else []


def extract_expected_value_from_reason(reason: str):
    if not reason:
        return None
    patterns = [
        ("expected_value=", ","), ("expected=", ","), ("expected ", ","),
        ("suggests rhs=", ", observed="), ("suggests ", ", observed="),
        ("dominant=", ", observed="), ("canonical=", ", observed="),
        ("canonical=", ","), ("canonical=", " "),
    ]
    for left, right in patterns:
        if left in reason:
            try:
                a = reason.index(left) + len(left)
                b = reason.index(right, a) if right and right in reason[a:] else len(reason)
                value = reason[a:b].strip()
                if left == "suggests " and "=" in value:
                    value = value.split("=", 1)[1].strip()
                value = value.strip(":;,. ")
                return value if value else None
            except Exception:
                pass
    return None


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return str(obj)


# ============================================================
# 3. Dataset-specific parsers
# ============================================================

def parse_int_like(value) -> Optional[int]:
    if value is None:
        return None
    s = str(value).strip().replace(",", "")
    if re.fullmatch(r"[+-]?\d+(\.0+)?", s):
        try:
            return int(float(s))
        except Exception:
            return None
    return None


# flights
def parse_time_like(value) -> Optional[int]:
    if value is None:
        return None
    s = str(value).strip().lower()
    s = re.sub(r"\s+", " ", s)
    for junk in ["(estimated runway)", "(estimated)", "(actual)", "(gate)", "(runway)", "(+8:00)", "(-00:00)"]:
        s = s.replace(junk, "").strip()
    s = re.sub(r"\bdec\s+\d{1,2}\b", "", s).strip()
    m = re.search(r"(\d{1,2}:\d{2})\s*(a\.m\.|p\.m\.)", s)
    if not m:
        return None
    hh, mm = map(int, m.group(1).split(":"))
    ap = m.group(2)
    if ap == "p.m." and hh != 12:
        hh += 12
    if ap == "a.m." and hh == 12:
        hh = 0
    return hh * 60 + mm


# beers
def parse_abv_like(value) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip().lower().replace("%", "")
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def parse_ibu_like(value) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in BASE_MISSING_TOKENS:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def parse_ounces_like(value) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip().lower()
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def parse_numeric_by_column(col: str, value):
    if col == "abv":
        return parse_abv_like(value)
    if col == "ibu":
        return parse_ibu_like(value)
    if col == "ounces":
        return parse_ounces_like(value)
    return try_parse_one_numeric(value)


# rayyan
def normalize_date_like(value) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip().lower()
    s = re.sub(r"\s+", "", s)
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{2,4})$", s)
    if m:
        mm = int(m.group(1))
        dd = int(m.group(2))
        yy_raw = m.group(3)
        yy = int(yy_raw)
        if len(yy_raw) == 2:
            yy = 1900 + yy if yy >= 30 else 2000 + yy
        if 1 <= mm <= 12 and 1 <= dd <= 31:
            return f"{yy:04d}-{mm:02d}-{dd:02d}"
    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", s)
    if m:
        yy = int(m.group(1))
        mm = int(m.group(2))
        dd = int(m.group(3))
        if 1 <= mm <= 12 and 1 <= dd <= 31:
            return f"{yy:04d}-{mm:02d}-{dd:02d}"
    return None


def canonicalize_language(value) -> Optional[str]:
    if value is None:
        return None
    v = str(value).strip().lower()
    mapping = {
        "eng": "eng", "english": "eng", "english; abstract language:english": "eng", "en": "eng",
        "pol": "pol", "polish": "pol",
        "fre": "fre", "french": "fre",
        "ger": "ger", "german": "ger",
        "spa": "spa", "spanish": "spa",
    }
    return mapping.get(v)


def extract_issn_digits(value) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip().lower()
    m = re.search(r"(\d{4})[- ]?(\d{3}[\dx])", s)
    if not m:
        return None
    return f"{m.group(1)}-{m.group(2)}"


def canonicalize_volume_issue(value) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip().lower()
    m = re.search(r"(\d+)", s)
    if not m:
        return None
    return str(int(m.group(1)))


# adult
VALID_AGE_BUCKETS = {"<18", "18-21", "22-25", "26-30", "31-35", "36-40", "41-45", "46-50", "51-55", "56-60", "61-65", "66-70", ">70"}
VALID_WORKCLASS = {"Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"}
VALID_EDUCATION = {"Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th", "11th", "12th", "HS-grad", "Some-college", "Assoc-voc", "Assoc-acdm", "Bachelors", "Masters", "Prof-school", "Doctorate"}
VALID_MARITALSTATUS = {"Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"}
VALID_OCCUPATION = {"Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"}
VALID_RELATIONSHIP = {"Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"}
VALID_RACE = {"White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"}
VALID_SEX = {"Male", "Female"}
VALID_COUNTRY = {
    "United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany",
    "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China",
    "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica",
    "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic",
    "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala",
    "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador",
    "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"
}
DOMAIN_MAP = {
    "workclass": VALID_WORKCLASS, "education": VALID_EDUCATION, "maritalstatus": VALID_MARITALSTATUS,
    "occupation": VALID_OCCUPATION, "relationship": VALID_RELATIONSHIP, "race": VALID_RACE,
    "sex": VALID_SEX, "country": VALID_COUNTRY,
}
EDUCATION_ORDER = {
    "Preschool": 0, "1st-4th": 1, "5th-6th": 2, "7th-8th": 3, "9th": 4, "10th": 5,
    "11th": 6, "12th": 7, "HS-grad": 8, "Some-college": 9, "Assoc-voc": 10,
    "Assoc-acdm": 10, "Bachelors": 11, "Masters": 12, "Prof-school": 13, "Doctorate": 14,
}
RELATIONSHIP_SPOUSE_VALUES = {"Husband", "Wife"}
NON_MARRIED_STATUS_FOR_SPOUSE = {"Never-married", "Divorced", "Separated", "Widowed"}
VALID_SEX_BY_RELATIONSHIP = {"Husband": "Male", "Wife": "Female"}


def parse_age_bucket(value) -> Tuple[Optional[int], Optional[int]]:
    if value is None:
        return None, None
    s = str(value).strip()
    m = re.fullmatch(r"<\s*(\d+)", s)
    if m:
        return 0, int(m.group(1)) - 1
    m = re.fullmatch(r">\s*(\d+)", s)
    if m:
        return int(m.group(1)) + 1, None
    m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return min(a, b), max(a, b)
    n = parse_int_like(s)
    if n is not None:
        return n, n
    return None, None


def parse_hours_value(value) -> Tuple[Optional[float], Optional[float]]:
    if value is None:
        return None, None
    s = str(value).strip()
    m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", s)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return min(a, b), max(a, b)
    n = parse_int_like(s)
    if n is not None:
        return float(n), float(n)
    return None, None


def canonical_income(value) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if s in {"LessThan50K", "<=50K", "<50K"}:
        return "LessThan50K"
    if s in {"MoreThan50K", ">50K", ">=50K"}:
        return "MoreThan50K"
    return None


# soccer
VALID_POSITIONS = {"Goalkeeper", "Defender", "Midfield", "Midfielder", "Forward", "Striker", "Winger"}
POSITION_CANONICAL_MAP = {"Midfielder": "Midfield"}
BIRTHYEAR_MIN = 1940
BIRTHYEAR_MAX = 2010
SEASON_MIN = 1900
SEASON_MAX = 2035
PLAYER_MIN_AGE_AT_SEASON = 15
PLAYER_MAX_AGE_AT_SEASON = 50
PERSON_NAME_REGEX = r"^[A-Za-z]+(?:_[A-Za-z]+)*$"
SURNAME_REGEX = r"^[A-Za-z]+(?:[-_'][A-Za-z]+)*$"
TEAM_REGEX = r"^[A-Za-z0-9]+(?:[._'\-][A-Za-z0-9]+)*$"
LOCATION_REGEX = r"^[A-Za-z]+(?:[ _'\-][A-Za-z]+)*$"
STADIUM_REGEX = r"^[A-Za-z0-9]+(?:[ _'\-][A-Za-z0-9]+)*$"

SOCCER_FORMAT_RULE_TYPES = {
    "name_pattern", "surname_pattern", "manager_name_pattern", "team_name_pattern",
    "birthplace_location_pattern", "city_location_pattern", "stadium_name_pattern",
    "pattern", "generic_regex",
}
SOCCER_CONTEXT_RULE_TYPES = {
    "team_context_dominant", "team_season_manager_consistency_hint",
    "city_stadium_consistency_hint", "dominant_value_by_context", "dominant_value_by_context_pair",
}
SOCCER_FD_RULE_TYPES = {"functional_dependency"}
SOCCER_EVIDENCE_ONLY_FD_RULE_TYPES = {"soft_functional_dependency"}


def parse_year(value) -> Optional[int]:
    return parse_int_like(value)


def canonical_position(value) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if s in POSITION_CANONICAL_MAP:
        return POSITION_CANONICAL_MAP[s]
    if s in VALID_POSITIONS:
        return s
    return None


def regex_fullmatch(value, pattern: str) -> bool:
    if value is None:
        return True
    return re.fullmatch(pattern, str(value).strip()) is not None


def is_birthyear_like(value) -> bool:
    y = parse_year(value)
    return y is not None and BIRTHYEAR_MIN <= y <= BIRTHYEAR_MAX


def is_season_like(value) -> bool:
    y = parse_year(value)
    return y is not None and SEASON_MIN <= y <= SEASON_MAX


def age_at_season(birthyear, season) -> Optional[int]:
    by, sy = parse_year(birthyear), parse_year(season)
    if by is None or sy is None:
        return None
    return sy - by


# ============================================================
# 4. Loaders and profiling
# ============================================================

def load_table(path: str, cfg: DatasetConfig) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df = maybe_drop_unnamed_columns(df)
    for col in df.columns:
        df[col] = normalize_series(df[col], cfg)
    return df


def load_shrunk_candidates(path: str, cfg: DatasetConfig) -> List[Dict[str, Any]]:
    df = pd.read_csv(path, dtype=str)
    items = []
    for _, row in df.iterrows():
        row_id_raw = row.get("row_id")
        try:
            row_id = int(float(row_id_raw))
        except Exception:
            row_id = int(row_id_raw)

        item = {
            "row_id": row_id,
            "column": row["column"],
            "value": normalize_value(row.get("value"), cfg),
            "violation_count": int(float(row["violation_count"])) if pd.notna(row.get("violation_count")) else 0,
            "conflict_score": float(row["conflict_score"]) if pd.notna(row.get("conflict_score")) else 0.0,
            "rule_ids": parse_pipe_list(row.get("rule_ids")),
            "rule_types": parse_pipe_list(row.get("rule_types")),
            "priorities": parse_pipe_list(row.get("priorities")),
            "usage_roles": parse_pipe_list(row.get("usage_roles")),
            "reasons": parse_reason_list(row.get("reasons")),
        }
        violated_rules = []
        max_len = max(
            len(item["rule_ids"]), len(item["rule_types"]), len(item["priorities"]),
            len(item["usage_roles"]), len(item["reasons"]), 0
        )
        for i in range(max_len):
            reason = item["reasons"][i] if i < len(item["reasons"]) else None
            violated_rules.append({
                "rule_id": item["rule_ids"][i] if i < len(item["rule_ids"]) else None,
                "rule_type": item["rule_types"][i] if i < len(item["rule_types"]) else None,
                "priority": item["priorities"][i] if i < len(item["priorities"]) else None,
                "usage_role": item["usage_roles"][i] if i < len(item["usage_roles"]) else None,
                "reason": reason,
                "expected_value": extract_expected_value_from_reason(reason),
            })
        item["violated_rules"] = violated_rules
        items.append(item)
    return items


def is_id_like_column(non_null: pd.Series, unique_ratio: float, numeric_ratio: float, avg_len: float, cfg: DatasetConfig) -> bool:
    if len(non_null) == 0:
        return False
    if unique_ratio >= cfg.id_like_unique_ratio_threshold and avg_len <= cfg.code_like_avg_len_threshold:
        return True
    if unique_ratio >= cfg.id_like_unique_ratio_threshold and numeric_ratio >= cfg.numeric_ratio_threshold:
        return True
    return False


def is_code_like_column(non_null: pd.Series, unique_ratio: float, numeric_ratio: float, avg_len: float,
                        top_patterns: List[Dict[str, Any]], cfg: DatasetConfig) -> bool:
    if len(non_null) == 0:
        return False
    if avg_len <= cfg.code_like_avg_len_threshold and unique_ratio > cfg.categorical_unique_ratio_threshold:
        if top_patterns and top_patterns[0]["ratio"] >= 0.7:
            return True
    if avg_len <= 15 and numeric_ratio >= 0.95 and unique_ratio < cfg.id_like_unique_ratio_threshold:
        return True
    return False


def infer_semantic_type(col: str, detected_type: str, non_null: pd.Series, unique_ratio: float,
                        numeric_ratio: float, avg_len: float, top_patterns: List[Dict[str, Any]],
                        cfg: DatasetConfig) -> str:
    if col in cfg.explicit_semantic_types:
        return cfg.explicit_semantic_types[col]
    if is_id_like_column(non_null, unique_ratio, numeric_ratio, avg_len, cfg):
        return "id_like"
    if is_code_like_column(non_null, unique_ratio, numeric_ratio, avg_len, top_patterns, cfg):
        return "code_like"
    if detected_type in {"numeric", "categorical"}:
        return detected_type
    return "text"


def analyze_columns(df: pd.DataFrame, cfg: DatasetConfig) -> Dict[str, Dict[str, Any]]:
    profiles = {}
    for col in df.columns:
        s = df[col]
        non_null = s.dropna()
        missing_rate = 1.0 - (len(non_null) / len(df) if len(df) else 0.0)
        unique_count = non_null.nunique()
        unique_ratio = unique_count / len(non_null) if len(non_null) > 0 else 0.0
        counter = Counter(non_null.tolist())
        rank_map = build_value_rank_map(counter)
        top_values = value_frequency_dict(non_null, topk=cfg.top_k_column_values)
        lengths = non_null.map(lambda x: len(str(x)))
        avg_len = float(lengths.mean()) if len(lengths) > 0 else 0.0
        length_stats = {
            "min": int(lengths.min()) if len(lengths) > 0 else None,
            "max": int(lengths.max()) if len(lengths) > 0 else None,
            "mean": round(avg_len, 6) if len(lengths) > 0 else None,
            "std": round(float(lengths.std(ddof=0)), 6) if len(lengths) > 1 else 0.0,
            "q1": round(float(lengths.quantile(0.25)), 6) if len(lengths) > 0 else None,
            "median": round(float(lengths.quantile(0.5)), 6) if len(lengths) > 0 else None,
            "q3": round(float(lengths.quantile(0.75)), 6) if len(lengths) > 0 else None,
        }
        numeric_vals, numeric_ratio = try_parse_numeric_series(non_null)
        pattern_counter = Counter(non_null.map(lambda x: detect_basic_pattern(x, cfg)).tolist())
        total_patterns = sum(pattern_counter.values())
        pattern_list = [
            {"pattern": p, "count": c, "ratio": round(c / total_patterns, 6)}
            for p, c in pattern_counter.most_common(10)
        ]

        if col in cfg.time_columns:
            detected_type = "time"
        elif col in cfg.date_columns:
            detected_type = "date"
        elif col in cfg.numeric_semantic_columns or col in cfg.numeric_metadata_columns:
            detected_type = "numeric"
        elif col in {"birthyear", "season"} and cfg.name == "soccer":
            detected_type = "numeric"
        elif col in cfg.categorical_columns:
            detected_type = "categorical"
        elif numeric_ratio >= cfg.numeric_ratio_threshold:
            detected_type = "numeric"
        elif unique_count <= cfg.categorical_unique_count_threshold or unique_ratio <= cfg.categorical_unique_ratio_threshold:
            detected_type = "categorical"
        else:
            detected_type = "text"

        semantic_type = infer_semantic_type(col, detected_type, non_null, unique_ratio, numeric_ratio, avg_len, pattern_list, cfg)

        numeric_stats = None
        if detected_type == "numeric":
            nums = numeric_vals.dropna()
            if len(nums) > 0:
                numeric_stats = {
                    "min": float(nums.min()),
                    "max": float(nums.max()),
                    "mean": float(nums.mean()),
                    "std": float(nums.std(ddof=0)) if len(nums) > 1 else 0.0,
                    "q1": float(nums.quantile(0.25)),
                    "median": float(nums.quantile(0.5)),
                    "q3": float(nums.quantile(0.75)),
                }

        time_stats = None
        if cfg.name == "flights" and col in cfg.time_columns:
            vals = non_null.map(parse_time_like).dropna()
            if len(vals) > 0:
                time_stats = {
                    "min_minutes": int(vals.min()),
                    "max_minutes": int(vals.max()),
                    "mean_minutes": round(float(vals.mean()), 6),
                }

        date_stats = None
        if cfg.name == "rayyan" and col in cfg.date_columns:
            vals = non_null.map(normalize_date_like).dropna()
            if len(vals) > 0:
                date_stats = {
                    "non_null_normalized_count": int(len(vals)),
                    "top_normalized_dates": value_frequency_dict(vals, topk=cfg.profile_topk_values),
                }

        profiles[col] = {
            "column": col,
            "missing_rate": round(missing_rate, 6),
            "non_null_count": int(len(non_null)),
            "unique_count": int(unique_count),
            "unique_ratio": round(unique_ratio, 6),
            "detected_type": detected_type,
            "semantic_type": semantic_type,
            "top_values": top_values,
            "value_counter": counter,
            "value_rank_map": rank_map,
            "length_stats": length_stats,
            "numeric_ratio": round(float(numeric_ratio), 6),
            "numeric_stats": numeric_stats,
            "time_stats": time_stats,
            "date_stats": date_stats,
            "patterns": pattern_list,
            "pattern_counter": pattern_counter,
            "dominant_pattern": pattern_list[0]["pattern"] if pattern_list else None,
            "dominant_pattern_ratio": pattern_list[0]["ratio"] if pattern_list else None,
        }
    return profiles


# ============================================================
# 5. Row / column / conflict contexts
# ============================================================

def build_adult_profile_bundle(row: Dict[str, Any]) -> Dict[str, Any]:
    age = row.get("age")
    hours = row.get("hoursperweek")
    education = row.get("education")
    relationship = row.get("relationship")
    marital = row.get("maritalstatus")
    sex = row.get("sex")
    income = row.get("income")
    workclass = row.get("workclass")
    occupation = row.get("occupation")

    age_lower, age_upper = parse_age_bucket(age)
    hours_lower, hours_upper = parse_hours_value(hours)
    edu_order = EDUCATION_ORDER.get(education)
    income_canon = canonical_income(income)

    flags = []
    if age is not None and age_lower is None:
        flags.append("invalid_age_bucket")
    if hours is not None and hours_lower is None:
        flags.append("invalid_hours_range")
    if income is not None and income_canon is None:
        flags.append("invalid_income_format")
    if relationship in RELATIONSHIP_SPOUSE_VALUES and marital in NON_MARRIED_STATUS_FOR_SPOUSE:
        flags.append("spouse_relationship_with_non_married_status")
    if relationship in VALID_SEX_BY_RELATIONSHIP and sex in VALID_SEX and sex != VALID_SEX_BY_RELATIONSHIP[relationship]:
        flags.append("relationship_sex_conflict")
    if age_upper is not None and age_upper <= 17 and relationship in RELATIONSHIP_SPOUSE_VALUES:
        flags.append("minor_spouse_relationship_conflict")
    if age_upper is not None and age_upper <= 17 and edu_order is not None and edu_order >= EDUCATION_ORDER["Bachelors"]:
        flags.append("minor_high_education_conflict")
    if workclass == "Never-worked" and occupation not in {None, "Other-service"}:
        flags.append("never_worked_with_occupation")

    return {
        "age": age,
        "age_lower": age_lower,
        "age_upper": age_upper,
        "education": education,
        "education_order": edu_order,
        "maritalstatus": marital,
        "relationship": relationship,
        "sex": sex,
        "hoursperweek": hours,
        "hours_lower": hours_lower,
        "hours_upper": hours_upper,
        "workclass": workclass,
        "occupation": occupation,
        "income": income,
        "income_canonical": income_canon,
        "adult_profile_inconsistencies": flags,
    }


def build_soccer_profile_bundle(row: Dict[str, Any]) -> Dict[str, Any]:
    name, surname, birthyear, birthplace = row.get("name"), row.get("surname"), row.get("birthyear"), row.get("birthplace")
    position, team, city, stadium = row.get("position"), row.get("team"), row.get("city"), row.get("stadium")
    season, manager = row.get("season"), row.get("manager")
    by, sy = parse_year(birthyear), parse_year(season)
    age = age_at_season(birthyear, season)
    flags = []
    if birthyear is not None and not is_birthyear_like(birthyear):
        flags.append("invalid_birthyear")
    if season is not None and not is_season_like(season):
        flags.append("invalid_season")
    if age is not None and (age < PLAYER_MIN_AGE_AT_SEASON or age > PLAYER_MAX_AGE_AT_SEASON):
        flags.append("implausible_age_at_season")
    if position is not None and canonical_position(position) is None:
        flags.append("invalid_position_domain")
    if position in POSITION_CANONICAL_MAP:
        flags.append("position_needs_canonicalization")
    if name is not None and not regex_fullmatch(name, PERSON_NAME_REGEX):
        flags.append("invalid_name_pattern")
    if surname is not None and not regex_fullmatch(surname, SURNAME_REGEX):
        flags.append("invalid_surname_pattern")
    if manager is not None and not regex_fullmatch(manager, PERSON_NAME_REGEX):
        flags.append("invalid_manager_pattern")
    if team is not None and not regex_fullmatch(team, TEAM_REGEX):
        flags.append("invalid_team_pattern")
    if birthplace is not None and not regex_fullmatch(birthplace, LOCATION_REGEX):
        flags.append("invalid_birthplace_pattern")
    if city is not None and not regex_fullmatch(city, LOCATION_REGEX):
        flags.append("invalid_city_pattern")
    if stadium is not None and not regex_fullmatch(stadium, STADIUM_REGEX):
        flags.append("invalid_stadium_pattern")

    return {
        "name": name,
        "surname": surname,
        "birthyear": birthyear,
        "birthyear_int": by,
        "birthplace": birthplace,
        "position": position,
        "position_canonical": canonical_position(position),
        "team": team,
        "city": city,
        "stadium": stadium,
        "season": season,
        "season_int": sy,
        "manager": manager,
        "age_at_season": age,
        "soccer_profile_inconsistencies": flags,
    }


def summarize_row_context(df: pd.DataFrame, row_id: int, col_profiles: Dict[str, Dict[str, Any]], cfg: DatasetConfig) -> Dict[str, Any]:
    row = df.iloc[row_id].to_dict()
    values = [v for v in row.values() if v is not None]
    missing_count = sum(v is None for v in row.values())
    lengths = [len(str(v)) for v in values]
    semantic_type_counts = Counter()
    for col, v in row.items():
        if v is None:
            continue
        semantic_type_counts[col_profiles[col]["semantic_type"]] += 1

    out = {
        "row_values": row,
        "row_values_for_llm": list(row.items())[:cfg.top_k_row_fields_for_llm],
        "row_patterns": {col: detect_basic_pattern(v, cfg) if v is not None else "NULL" for col, v in row.items()},
        "missing_count": int(missing_count),
        "non_missing_count": int(len(values)),
        "avg_value_length": round(float(np.mean(lengths)), 6) if lengths else None,
        "max_value_length": int(max(lengths)) if lengths else None,
        "min_value_length": int(min(lengths)) if lengths else None,
        "semantic_type_counts": dict(semantic_type_counts),
    }

    if cfg.name == "flights":
        time_bundle = {}
        for c in cfg.time_columns:
            if c in row:
                time_bundle[c] = row.get(c)
        if all(c in row for c in ["sched_dep_time", "sched_arr_time"]):
            sd, sa = parse_time_like(row.get("sched_dep_time")), parse_time_like(row.get("sched_arr_time"))
            time_bundle["sched_duration_minutes"] = ((sa - sd) % 1440) if sd is not None and sa is not None else None
        if all(c in row for c in ["act_dep_time", "act_arr_time"]):
            ad, aa = parse_time_like(row.get("act_dep_time")), parse_time_like(row.get("act_arr_time"))
            time_bundle["act_duration_minutes"] = ((aa - ad) % 1440) if ad is not None and aa is not None else None
        out["time_bundle"] = time_bundle

    elif cfg.name == "beers":
        numeric_snapshot = {}
        for col, v in row.items():
            if col in cfg.numeric_semantic_columns:
                numeric_snapshot[col] = parse_numeric_by_column(col, v)
        out["numeric_snapshot"] = numeric_snapshot

    elif cfg.name == "rayyan":
        metadata_bundle = {}
        for c in ["journal_title", "jounral_abbreviation", "journal_issn", "article_jvolumn", "article_jissue", "article_jcreated_at"]:
            if c in row:
                metadata_bundle[c] = row.get(c)
        out["metadata_bundle"] = metadata_bundle

    elif cfg.name == "adult":
        out["adult_profile_bundle"] = build_adult_profile_bundle(row)

    elif cfg.name == "soccer":
        out["soccer_profile_bundle"] = build_soccer_profile_bundle(row)

    return out


def build_adult_value_features(col: str, value: Any) -> Dict[str, Any]:
    key = ("adult", col, value)
    if key in _VALUE_FEATURE_CACHE:
        return _VALUE_FEATURE_CACHE[key]

    age_lower, age_upper = parse_age_bucket(value) if col == "age" else (None, None)
    hours_lower, hours_upper = parse_hours_value(value) if col == "hoursperweek" else (None, None)
    income_canon = canonical_income(value) if col == "income" else None
    edu_order = EDUCATION_ORDER.get(value) if col == "education" else None
    domain_valid = None
    if col in DOMAIN_MAP and value is not None:
        domain_valid = value in DOMAIN_MAP[col]
    elif col == "age" and value is not None:
        domain_valid = age_lower is not None
    elif col == "hoursperweek" and value is not None:
        domain_valid = hours_lower is not None
    elif col == "income" and value is not None:
        domain_valid = income_canon is not None

    bucket = "unknown"
    if col == "age":
        if age_upper is not None and age_upper < 18:
            bucket = "minor_age"
        elif age_lower is not None:
            bucket = "adult_age"
        else:
            bucket = "invalid_age"
    elif col == "hoursperweek":
        if hours_upper is not None and hours_upper >= 60:
            bucket = "high_hours"
        elif hours_lower is not None:
            bucket = "normal_hours"
        else:
            bucket = "invalid_hours"
    elif col == "education":
        bucket = "known_education" if edu_order is not None else "unknown_education"
    elif col == "income":
        bucket = "known_income" if income_canon is not None else "unknown_income"
    elif col in DOMAIN_MAP:
        bucket = f"known_{col}" if domain_valid else f"unknown_{col}"

    out = {
        "age_lower": age_lower,
        "age_upper": age_upper,
        "hours_lower": hours_lower,
        "hours_upper": hours_upper,
        "canonical_income": income_canon,
        "domain_valid": domain_valid,
        "education_order": edu_order,
        "adult_value_bucket": bucket,
    }
    _VALUE_FEATURE_CACHE[key] = out
    return out


def build_soccer_value_features(col: str, value: Any) -> Dict[str, Any]:
    key = ("soccer", col, value)
    if key in _VALUE_FEATURE_CACHE:
        return _VALUE_FEATURE_CACHE[key]

    year_value = parse_year(value) if col in {"birthyear", "season"} else None
    domain_valid, canonical = None, None
    if col == "birthyear":
        domain_valid = is_birthyear_like(value) if value is not None else None
    elif col == "season":
        domain_valid = is_season_like(value) if value is not None else None
    elif col == "position":
        canonical = canonical_position(value)
        domain_valid = canonical is not None if value is not None else None
    elif col == "name":
        domain_valid = regex_fullmatch(value, PERSON_NAME_REGEX) if value is not None else None
    elif col == "surname":
        domain_valid = regex_fullmatch(value, SURNAME_REGEX) if value is not None else None
    elif col == "manager":
        domain_valid = regex_fullmatch(value, PERSON_NAME_REGEX) if value is not None else None
    elif col == "team":
        domain_valid = regex_fullmatch(value, TEAM_REGEX) if value is not None else None
    elif col in {"birthplace", "city"}:
        domain_valid = regex_fullmatch(value, LOCATION_REGEX) if value is not None else None
    elif col == "stadium":
        domain_valid = regex_fullmatch(value, STADIUM_REGEX) if value is not None else None

    bucket = "unknown"
    if col == "birthyear" and year_value is not None:
        bucket = "older_generation" if year_value < 1970 else "typical_player_birthyear" if year_value <= 1995 else "young_generation"
    elif col == "season" and year_value is not None:
        bucket = "old_season" if year_value < 2000 else "modern_season" if year_value <= 2025 else "future_season"
    elif col == "position":
        bucket = "known_position" if canonical is not None else "unknown_position"
    elif col in {"team", "stadium", "city", "birthplace", "manager", "name", "surname"}:
        bucket = f"{col}_entity"

    out = {
        "year_value": year_value,
        "canonical_position": canonical,
        "domain_valid": domain_valid,
        "soccer_value_bucket": bucket,
    }
    _VALUE_FEATURE_CACHE[key] = out
    return out


def summarize_column_context(col: str, value: Any, col_profiles: Dict[str, Dict[str, Any]], cfg: DatasetConfig) -> Dict[str, Any]:
    prof = col_profiles[col]
    counter = prof["value_counter"]
    value_freq = int(counter.get(value, 0))
    rank = prof["value_rank_map"].get(value)
    current_pattern = detect_basic_pattern(value, cfg) if value is not None else "NULL"

    out = {
        "column": col,
        "detected_type": prof["detected_type"],
        "semantic_type": prof["semantic_type"],
        "missing_rate": prof["missing_rate"],
        "unique_rate": prof["unique_ratio"],
        "unique_count": prof["unique_count"],
        "non_null_count": prof["non_null_count"],
        "value_frequency": value_freq,
        "value_frequency_rank": rank,
        "top_values_with_counts": prof["top_values"],
        "length_stats": prof["length_stats"],
        "numeric_ratio": prof["numeric_ratio"],
        "numeric_stats": prof["numeric_stats"],
        "current_value_pattern": current_pattern,
        "dominant_pattern": prof["dominant_pattern"],
        "dominant_pattern_ratio": prof["dominant_pattern_ratio"],
        "top_patterns_with_counts": prof["patterns"],
        "charset_features": charset_features(value),
    }

    if cfg.name == "flights":
        out["time_stats"] = prof["time_stats"]
        out["time_value_minutes"] = parse_time_like(value) if col in cfg.time_columns else None
    elif cfg.name == "beers":
        out["numeric_value"] = parse_numeric_by_column(col, value) if col in cfg.numeric_semantic_columns else try_parse_one_numeric(value)
    elif cfg.name == "rayyan":
        out["numeric_value"] = canonicalize_volume_issue(value) if col in cfg.numeric_metadata_columns else None
        out["date_value"] = normalize_date_like(value) if col in cfg.date_columns else None
        out["language_canonical"] = canonicalize_language(value) if col == "article_language" else None
        out["issn_canonical"] = extract_issn_digits(value) if col == "journal_issn" else None
        out["date_stats"] = prof["date_stats"]
    elif cfg.name == "adult":
        out["adult_value_features"] = build_adult_value_features(col, value)
    elif cfg.name == "soccer":
        out["soccer_value_features"] = build_soccer_value_features(col, value)

    return out


def summarize_conflict_info(candidate: Dict[str, Any], cfg: DatasetConfig) -> Dict[str, Any]:
    col, current_value = candidate["column"], candidate["value"]
    violated_rules = candidate.get("violated_rules", [])
    rule_type_counter, priority_counter, usage_role_counter = Counter(), Counter(), Counter()
    expected, seen = [], set()

    for r in violated_rules:
        rt, pr, ur = r.get("rule_type"), r.get("priority"), r.get("usage_role")
        if rt:
            rule_type_counter[rt] += 1
        if pr:
            priority_counter[pr] += 1
        if ur:
            usage_role_counter[ur] += 1
        ev = r.get("expected_value")
        if ev is not None and ev != current_value and ev not in seen:
            expected.append({
                "source": "rule_reason_parse",
                "value": ev,
                "rule_id": r.get("rule_id"),
                "rule_type": rt,
                "priority": pr,
                "usage_role": ur,
                "reason": r.get("reason"),
            })
            seen.add(ev)

    main_rule_type = rule_type_counter.most_common(1)[0][0] if rule_type_counter else "none"
    main_usage_role = usage_role_counter.most_common(1)[0][0] if usage_role_counter else "none"

    fd_like = rule_type_counter.get("functional_dependency", 0) + rule_type_counter.get("soft_functional_dependency", 0)
    context_rule = (
        rule_type_counter.get("dominant_value_by_context", 0)
        + rule_type_counter.get("dominant_value_by_context_pair", 0)
        + rule_type_counter.get("relationship_context_dominant", 0)
        + rule_type_counter.get("team_context_dominant", 0)
        + rule_type_counter.get("team_season_manager_consistency_hint", 0)
        + rule_type_counter.get("city_stadium_consistency_hint", 0)
    )
    pattern_rule = (
        rule_type_counter.get("pattern", 0)
        + rule_type_counter.get("rare_pattern", 0)
        + sum(rule_type_counter.get(x, 0) for x in SOCCER_FORMAT_RULE_TYPES)
        + rule_type_counter.get("score_format", 0)
        + rule_type_counter.get("sample_format", 0)
        + rule_type_counter.get("sample_patients_suffix", 0)
    )
    schema_rule = (
        rule_type_counter.get("not_null", 0)
        + rule_type_counter.get("high_risk_missing", 0)
        + pattern_rule
        + rule_type_counter.get("numeric_range", 0)
        + rule_type_counter.get("enumeration_domain", 0)
    )

    out = {
        "has_rule_violation": len(violated_rules) > 0,
        "violation_count": len(violated_rules),
        "violated_rules_detailed": violated_rules,
        "rule_ids": [r.get("rule_id") for r in violated_rules],
        "rule_types": [r.get("rule_type") for r in violated_rules],
        "priorities": [r.get("priority") for r in violated_rules],
        "usage_roles": [r.get("usage_role") for r in violated_rules],
        "rule_type_counter": dict(rule_type_counter),
        "priority_counter": dict(priority_counter),
        "usage_role_counter": dict(usage_role_counter),
        "main_rule_type": main_rule_type,
        "main_usage_role": main_usage_role,
        "strong_rule_count": usage_role_counter.get("strong_rule", 0),
        "candidate_generation_rule_count": usage_role_counter.get("candidate_generation_rule", 0),
        "fd_like_count": fd_like,
        "context_rule_count": context_rule,
        "global_rule_count": rule_type_counter.get("global_dominant_value", 0),
        "rare_value_count": rule_type_counter.get("rare_value", 0),
        "typo_rule_count": rule_type_counter.get("dominant_value_typo", 0),
        "pattern_rule_count": pattern_rule,
        "schema_rule_count": schema_rule,
        "expected_values_from_rules": expected,
        "alternative_values_from_column": [],
        "all_alternative_values": expected,
    }

    if cfg.name == "flights":
        out["time_window_rule_count"] = (
            rule_type_counter.get("temporal_pair_consistency", 0)
            + rule_type_counter.get("delay_window", 0)
            + rule_type_counter.get("duration_consistency_window", 0)
            + rule_type_counter.get("actual_time_global_candidate", 0)
            + rule_type_counter.get("time_format", 0)
            + rule_type_counter.get("datetime_time_format", 0)
            + rule_type_counter.get("time_loose_pattern", 0)
        )

    elif cfg.name == "beers":
        out["numeric_window_rule_count"] = (
            rule_type_counter.get("numeric_range_window", 0)
            + rule_type_counter.get("ounces_canonical_format", 0)
            + rule_type_counter.get("abv_canonical_format", 0)
            + rule_type_counter.get("abv_numeric_format", 0)
            + rule_type_counter.get("ibu_numeric_or_na_format", 0)
            + rule_type_counter.get("ounces_unit_format", 0)
        )

    elif cfg.name == "rayyan":
        out["canonical_rule_count"] = (
            rule_type_counter.get("issn_canonical_format", 0)
            + rule_type_counter.get("language_canonical_format", 0)
            + rule_type_counter.get("date_component_permutation_suspicious", 0)
            + rule_type_counter.get("date_ymd_swap_suspicious", 0)
            + rule_type_counter.get("author_list_encoding_corruption", 0)
        )

    elif cfg.name == "adult":
        adult_domain = sum(rule_type_counter.get(x, 0) for x in [
            "workclass_domain", "education_domain", "maritalstatus_domain", "occupation_domain",
            "relationship_domain", "race_domain", "sex_domain", "country_domain", "income_domain",
        ])
        adult_format = (
            rule_type_counter.get("age_bucket_format", 0)
            + rule_type_counter.get("hours_per_week_format_range", 0)
            + rule_type_counter.get("income_canonical_format", 0)
        )
        adult_consistency = sum(rule_type_counter.get(x, 0) for x in [
            "age_education_consistency", "age_relationship_consistency", "marital_relationship_consistency",
            "sex_relationship_consistency", "workclass_occupation_consistency", "education_occupation_consistency",
            "hours_workclass_consistency", "relationship_marital_spouse_conflict", "relationship_age_spouse_conflict",
            "relationship_sex_spouse_conflict", "relationship_context_dominant", "education_age_extreme_conflict",
        ])
        out["adult_domain_rule_count"] = adult_domain
        out["adult_format_rule_count"] = adult_format
        out["adult_consistency_rule_count"] = adult_consistency
        out["time_window_rule_count"] = 0

    elif cfg.name == "soccer":
        soccer_domain = rule_type_counter.get("position_domain", 0)
        soccer_format = sum(rule_type_counter.get(x, 0) for x in SOCCER_FORMAT_RULE_TYPES)
        soccer_numeric = rule_type_counter.get("birthyear_numeric_range", 0) + rule_type_counter.get("season_numeric_range", 0)
        soccer_consistency = (
            rule_type_counter.get("birthyear_season_age_consistency", 0)
            + rule_type_counter.get("team_context_dominant", 0)
            + rule_type_counter.get("team_season_manager_consistency_hint", 0)
            + rule_type_counter.get("city_stadium_consistency_hint", 0)
        )
        out["soccer_domain_rule_count"] = soccer_domain
        out["soccer_format_rule_count"] = soccer_format
        out["soccer_numeric_rule_count"] = soccer_numeric
        out["soccer_consistency_rule_count"] = soccer_consistency
        out["adult_domain_rule_count"] = soccer_domain
        out["adult_format_rule_count"] = soccer_format
        out["adult_consistency_rule_count"] = soccer_consistency
        out["evidence_only_fd_count"] = rule_type_counter.get("soft_functional_dependency", 0)
        out["fd_like_count"] = rule_type_counter.get("functional_dependency", 0)
        out["time_window_rule_count"] = 0

    return out


def add_alternative_values(conflict_ctx: Dict[str, Any], col_ctx: Dict[str, Any], cfg: DatasetConfig):
    seen = {x["value"] for x in conflict_ctx.get("expected_values_from_rules", [])}
    current_value = None
    alt_col = []
    for item in col_ctx.get("top_values_with_counts", []):
        v = item.get("value")
        if v != current_value and v not in seen:
            alt_col.append({"source": "column_top_value", "value": v, "count": item.get("count")})
            seen.add(v)
        if len(alt_col) >= cfg.top_k_alt_values:
            break
    conflict_ctx["alternative_values_from_column"] = alt_col
    conflict_ctx["all_alternative_values"] = conflict_ctx.get("expected_values_from_rules", []) + alt_col


# ============================================================
# 6. Similarity
# ============================================================

def encode_cell_for_similarity(series: pd.Series, semantic_type: str) -> np.ndarray:
    n = len(series)
    if semantic_type in {"numeric", "birth_year", "season_year", "numeric_metadata", "time", "date"}:
        if semantic_type == "time":
            nums = series.map(parse_time_like)
        elif semantic_type == "date":
            vals = series.map(normalize_date_like)
            nums = pd.to_datetime(vals, errors="coerce").map(lambda x: x.value if pd.notna(x) else np.nan)
        else:
            nums = pd.to_numeric(series.astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False), errors="coerce")
        nums = pd.to_numeric(nums, errors="coerce")
        fill = nums.median() if nums.notna().any() else 0.0
        arr = nums.fillna(fill).astype(float).values
        std = np.std(arr) or 1.0
        return ((arr - np.mean(arr)) / std).astype(np.float32).reshape(n, 1)

    vals = series.fillna("__MISSING__").astype(str)
    codes, _ = pd.factorize(vals, sort=False)
    max_code = max(int(codes.max()), 1)
    codes_norm = codes.astype(np.float32) / max_code
    hash_mod = np.array([hash(x) % 997 for x in vals], dtype=np.float32) / 997.0
    return np.stack([codes_norm, hash_mod], axis=1).astype(np.float32)


def build_similarity_matrix(df: pd.DataFrame, col_profiles: Dict[str, Dict[str, Any]]) -> np.ndarray:
    blocks = [encode_cell_for_similarity(df[col], col_profiles[col]["semantic_type"]) for col in df.columns]
    X = np.concatenate(blocks, axis=1).astype(np.float32)
    mean, std = X.mean(axis=0, keepdims=True), X.std(axis=0, keepdims=True)
    std[std <= 1e-8] = 1.0
    X = (X - mean) / std
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    norm[norm <= 1e-8] = 1.0
    return (X / norm).astype(np.float32)


def precompute_neighbor_indices_for_candidates(
    X: np.ndarray,
    candidate_row_ids: List[int],
    top_k: int,
    batch_size: int,
) -> Dict[int, List[Tuple[int, float]]]:
    result: Dict[int, List[Tuple[int, float]]] = {}
    n = X.shape[0]
    if n == 0:
        return result
    k = min(top_k + 1, n)
    unique_rows = sorted(set(candidate_row_ids))

    for start in range(0, len(unique_rows), batch_size):
        rows = unique_rows[start:start + batch_size]
        sims = X[rows] @ X.T
        for local_i, row_id in enumerate(rows):
            sims[local_i, row_id] = -np.inf

        if k < n:
            idx_part = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        else:
            idx_part = np.argsort(-sims, axis=1)[:, :k]

        for local_i, row_id in enumerate(rows):
            idxs = idx_part[local_i]
            scores = sims[local_i, idxs]
            order = np.argsort(-scores)
            pairs = []
            for j in order:
                score = float(scores[j])
                if not np.isfinite(score):
                    continue
                pairs.append((int(idxs[j]), round((score + 1.0) / 2.0, 6)))
                if len(pairs) >= top_k:
                    break
            result[int(row_id)] = pairs

    return result


def summarize_similarity_info_fast(
    df: pd.DataFrame,
    row_id: int,
    target_col: str,
    neighbor_indices: Dict[int, List[Tuple[int, float]]],
    cfg: DatasetConfig,
):
    pairs = neighbor_indices.get(row_id, [])[:cfg.top_k_similar_rows]
    similar_rows, target_values = [], []
    for idx, sim in pairs:
        target_val = df.iloc[idx][target_col]
        item = {"row_id": int(idx), "similarity": sim, "target_column_value": target_val}
        if cfg.include_similar_row_values:
            item["row_values"] = df.iloc[idx].to_dict()
        similar_rows.append(item)
        target_values.append(target_val)
    dist = Counter([v for v in target_values if v is not None])
    majority_value = dist.most_common(1)[0][0] if dist else None
    majority_ratio = round(dist.most_common(1)[0][1] / len(target_values), 6) if dist and target_values else None
    return {
        "topk_similar_rows": similar_rows,
        "neighbor_value_distribution": dict(dist),
        "neighbor_majority_value": majority_value,
        "neighbor_majority_ratio": majority_ratio,
    }


# ============================================================
# 7. Dataset consistency
# ============================================================

def summarize_adult_consistency_info(row_ctx: Dict[str, Any], col: str, value: Any, conflict_ctx: Dict[str, Any]):
    bundle = row_ctx.get("adult_profile_bundle", {})
    flags = list(bundle.get("adult_profile_inconsistencies", []))
    rule_types = set(conflict_ctx.get("rule_types", []))

    if col == "age" and value is not None and build_adult_value_features(col, value)["age_lower"] is None:
        flags.append("target_invalid_age")
    if col == "hoursperweek" and value is not None and build_adult_value_features(col, value)["hours_lower"] is None:
        flags.append("target_invalid_hours")
    if col == "income" and value is not None and canonical_income(value) is None:
        flags.append("target_invalid_income")
    if col in DOMAIN_MAP and value is not None and value not in DOMAIN_MAP[col]:
        flags.append(f"target_invalid_{col}_domain")

    relationship_rules = {
        "relationship_marital_spouse_conflict",
        "relationship_age_spouse_conflict",
        "relationship_sex_spouse_conflict",
        "relationship_context_dominant",
    }
    education_rules = {"education_age_extreme_conflict"}
    if rule_types & relationship_rules:
        flags.append("relationship_v2_rule_signal")
    if rule_types & education_rules:
        flags.append("education_v2_rule_signal")

    score = 0.0
    if flags:
        score += min(2.0, 0.5 * len(set(flags)))

    return {
        "adult_profile_bundle": bundle,
        "evidence_flags": flags,
        "adult_consistency_score": round(float(score), 6),
    }


def precompute_soccer_group_distributions(df: pd.DataFrame, cfg: DatasetConfig):
    def top_values_for_group(sub: pd.DataFrame, target: str) -> Dict[str, Any]:
        cnt = Counter(sub[target].dropna().tolist())
        return {
            "top_values": [{"value": k, "count": v} for k, v in cnt.most_common(cfg.profile_topk_values)],
            "group_size": int(len(sub)),
        }

    team_season_map, team_map, stadium_city_map = {}, {}, {}
    if all(c in df.columns for c in ["team", "season"]):
        for key, sub in df.groupby(["team", "season"], dropna=False, sort=False):
            if len(sub) <= 1:
                continue
            team_season_map[key] = {
                target: top_values_for_group(sub, target)
                for target in ["city", "stadium", "manager", "position"]
                if target in sub.columns
            }
    if "team" in df.columns:
        for key, sub in df.groupby(["team"], dropna=False, sort=False):
            if len(sub) <= 1:
                continue
            if not isinstance(key, tuple):
                key = (key,)
            team_map[key] = {
                target: top_values_for_group(sub, target)
                for target in ["city", "stadium", "manager", "position", "season"]
                if target in sub.columns
            }
    if all(c in df.columns for c in ["city", "stadium"]):
        for key, sub in df.groupby(["city", "stadium"], dropna=False, sort=False):
            if len(sub) <= 1:
                continue
            stadium_city_map[key] = {
                target: top_values_for_group(sub, target)
                for target in ["team", "manager", "season"]
                if target in sub.columns
            }
    return team_season_map, team_map, stadium_city_map


def summarize_soccer_consistency_info(
    df: pd.DataFrame,
    row_id: int,
    col: str,
    value: Any,
    row_ctx: Dict[str, Any],
    group_maps: Tuple[Dict, Dict, Dict],
):
    row = df.iloc[row_id].to_dict()
    profile = row_ctx.get("soccer_profile_bundle", {})
    flags = list(profile.get("soccer_profile_inconsistencies", []))
    if col == "birthyear" and value is not None and not is_birthyear_like(value):
        flags.append("target_invalid_birthyear")
    if col == "season" and value is not None and not is_season_like(value):
        flags.append("target_invalid_season")
    if col == "position" and value is not None and canonical_position(value) is None:
        flags.append("target_invalid_position")
    if col == "name" and value is not None and not regex_fullmatch(value, PERSON_NAME_REGEX):
        flags.append("target_invalid_name_pattern")
    if col == "surname" and value is not None and not regex_fullmatch(value, SURNAME_REGEX):
        flags.append("target_invalid_surname_pattern")
    if col == "manager" and value is not None and not regex_fullmatch(value, PERSON_NAME_REGEX):
        flags.append("target_invalid_manager_pattern")
    if col == "team" and value is not None and not regex_fullmatch(value, TEAM_REGEX):
        flags.append("target_invalid_team_pattern")
    if col in {"birthplace", "city"} and value is not None and not regex_fullmatch(value, LOCATION_REGEX):
        flags.append(f"target_invalid_{col}_pattern")
    if col == "stadium" and value is not None and not regex_fullmatch(value, STADIUM_REGEX):
        flags.append("target_invalid_stadium_pattern")

    team_season_map, team_map, stadium_city_map = group_maps
    team_season_distribution = team_season_map.get((row.get("team"), row.get("season")), {})
    team_distribution = team_map.get((row.get("team"),), {})
    stadium_city_distribution = stadium_city_map.get((row.get("city"), row.get("stadium")), {})

    score = 0.0
    if flags:
        score += min(2.0, 0.5 * len(set(flags)))
    if team_season_distribution and col in {"city", "stadium", "manager", "position"}:
        score += 0.5
    if team_distribution and col in {"city", "stadium", "manager", "position", "season"}:
        score += 0.3
    if stadium_city_distribution and col in {"team", "manager", "season"}:
        score += 0.3

    return {
        "soccer_profile_bundle": profile,
        "evidence_flags": flags,
        "team_season_distribution": team_season_distribution,
        "team_distribution": team_distribution,
        "stadium_city_distribution": stadium_city_distribution,
        "soccer_consistency_score": round(float(score), 6),
    }


def build_adult_v2_attribution_features(row_ctx: Dict[str, Any], col: str, value: Any, conflict_ctx: Dict[str, Any]):
    row_values = row_ctx.get("row_values", {}) or {}
    rule_types = set(conflict_ctx.get("rule_types", []))
    spb = row_ctx.get("adult_profile_bundle", {})
    relationship_rules = {
        "relationship_marital_spouse_conflict",
        "relationship_age_spouse_conflict",
        "relationship_sex_spouse_conflict",
        "relationship_context_dominant",
    }
    education_rules = {"education_age_extreme_conflict"}

    rel_signal = int(len(rule_types & relationship_rules) > 0)
    edu_signal = int(len(rule_types & education_rules) > 0)
    bucket = "unknown"
    if col == "relationship" and rel_signal:
        bucket = "relationship_v2_conflict"
    elif col == "education" and edu_signal:
        bucket = "education_v2_conflict"
    elif col in {"sex", "relationship", "education"}:
        bucket = "primary_target_other"
    elif col in CONFIGS["adult"].context_only_columns:
        bucket = "context_only_other"

    return {
        "target_is_primary_column": int(col in CONFIGS["adult"].primary_target_columns),
        "target_is_context_only_column": int(col in CONFIGS["adult"].context_only_columns),
        "relationship_v2_conflict_count": rel_signal,
        "has_relationship_v2_signal": rel_signal,
        "has_education_v2_signal": edu_signal,
        "adult_v2_signal_strength": 2.0 * rel_signal + 1.5 * edu_signal + int(col in CONFIGS["adult"].primary_target_columns),
        "adult_v2_primary_target_signal": int(col in CONFIGS["adult"].primary_target_columns and (rel_signal or edu_signal or conflict_ctx.get("adult_domain_rule_count", 0) > 0)),
        "row_age_lower": spb.get("age_lower"),
        "row_age_upper": spb.get("age_upper"),
        "row_hours_lower": spb.get("hours_lower"),
        "row_hours_upper": spb.get("hours_upper"),
        "row_education_order": spb.get("education_order"),
        "row_income_canonical": spb.get("income_canonical"),
        "adult_attribution_bucket": bucket,
    }


def build_soccer_attribution_features(row_ctx: Dict[str, Any], col: str, value: Any, conflict_ctx: Dict[str, Any]):
    row_values = row_ctx.get("row_values", {}) or {}
    birthyear, season, position = row_values.get("birthyear"), row_values.get("season"), row_values.get("position")
    by, sy = parse_year(birthyear), parse_year(season)
    age = age_at_season(birthyear, season)
    rule_types = set(conflict_ctx.get("rule_types", []))
    birthyear_invalid = int(col == "birthyear" and value is not None and not is_birthyear_like(value))
    season_invalid = int(col == "season" and value is not None and not is_season_like(value))
    position_invalid = int(col == "position" and value is not None and canonical_position(value) is None)
    position_canon = int(col == "position" and value in POSITION_CANONICAL_MAP)
    age_conflict = int("birthyear_season_age_consistency" in rule_types or (age is not None and (age < PLAYER_MIN_AGE_AT_SEASON or age > PLAYER_MAX_AGE_AT_SEASON)))
    team_context_conflict = int("team_context_dominant" in rule_types)
    format_signal = int(len(rule_types & SOCCER_FORMAT_RULE_TYPES) > 0)
    fd_signal = int(len(rule_types & SOCCER_FD_RULE_TYPES) > 0)

    bucket = "unknown"
    if birthyear_invalid:
        bucket = "birthyear_invalid"
    elif season_invalid:
        bucket = "season_invalid"
    elif position_invalid:
        bucket = "position_invalid"
    elif position_canon:
        bucket = "position_canonicalization"
    elif age_conflict:
        bucket = "birthyear_season_age_conflict"
    elif team_context_conflict:
        bucket = "team_context_conflict"
    elif format_signal:
        bucket = "format_pattern_conflict"
    elif fd_signal:
        bucket = "fd_like_conflict"
    elif col in CONFIGS["soccer"].primary_target_columns:
        bucket = "primary_target_other"

    return {
        "target_is_primary_column": int(col in CONFIGS["soccer"].primary_target_columns),
        "target_is_context_only_column": int(col in CONFIGS["soccer"].context_only_columns),
        "birthyear_value_invalid": birthyear_invalid,
        "season_value_invalid": season_invalid,
        "position_value_invalid": position_invalid,
        "position_needs_canonicalization": position_canon,
        "birthyear_season_age_conflict": age_conflict,
        "team_context_conflict": team_context_conflict,
        "format_rule_signal": format_signal,
        "fd_like_signal": fd_signal,
        "row_birthyear_int": by,
        "row_season_int": sy,
        "row_age_at_season": age,
        "row_position_canonical": canonical_position(position),
        "row_team": row_values.get("team"),
        "row_city": row_values.get("city"),
        "row_stadium": row_values.get("stadium"),
        "row_manager": row_values.get("manager"),
        "soccer_attribution_bucket": bucket,
    }


# ============================================================
# 8. Bayesian and text
# ============================================================

def build_candidate_stats(candidates: List[Dict[str, Any]]):
    by_column, by_row = Counter(), Counter()
    for c in candidates:
        by_column[c["column"]] += 1
        by_row[c["row_id"]] += 1
    return {"by_column": by_column, "by_row": by_row, "total_candidates": len(candidates)}


def summarize_bayesian_info(
    candidate: Dict[str, Any],
    df: pd.DataFrame,
    col_profiles: Dict[str, Dict[str, Any]],
    candidate_stats,
    similarity_info,
    dataset_consistency_score: float,
    cfg: DatasetConfig,
):
    col, value = candidate["column"], candidate["value"]
    semantic_type = col_profiles[col]["semantic_type"]
    prior = min(max(candidate_stats["by_column"].get(col, 0) / len(df), 1e-4), 0.95) if len(df) else 0.01
    rule_signal = min(5.0, len(candidate.get("violated_rules", [])) + candidate.get("conflict_score", 0.0) / 100.0)

    counter = col_profiles[col]["value_counter"]
    freq, non_null_count = counter.get(value, 0), col_profiles[col]["non_null_count"]
    rarity = 1.0 - (freq / non_null_count) if non_null_count > 0 else 1.0
    rarity = max(0.0, min(1.0, rarity))

    current_pattern = detect_basic_pattern(value, cfg) if value is not None else "NULL"
    dominant_pattern = col_profiles[col]["dominant_pattern"]
    pattern_signal = 1.0 if dominant_pattern is not None and current_pattern != dominant_pattern else 0.0

    if semantic_type in {"id_like", "code_like"}:
        rarity *= 0.2
    if cfg.name == "soccer":
        if semantic_type in {"player_given_name", "player_surname", "manager_name", "team_entity", "stadium_entity", "city_location", "birthplace_location"}:
            rarity *= 0.35
        elif semantic_type == "position_category":
            rarity *= 0.8
        elif semantic_type in {"birth_year", "season_year"}:
            rarity *= 0.6

    neighbor_value = similarity_info.get("neighbor_majority_value")
    neighbor_ratio = similarity_info.get("neighbor_majority_ratio")
    neighbor_signal = 0.0
    if neighbor_value is not None and neighbor_ratio is not None:
        neighbor_signal = neighbor_ratio if neighbor_value != value else 0.0

    posterior = sigmoid(
        safe_logit(prior)
        + cfg.w_rule * rule_signal
        + cfg.w_rarity * rarity
        + cfg.w_pattern * pattern_signal
        + cfg.w_neighbor * neighbor_signal
        + cfg.w_dataset_consistency * dataset_consistency_score
    )

    return {
        "prior_error_probability": round(prior, 6),
        "posterior_error_probability": round(posterior, 6),
        "bayes_evidence": {
            "rule_signal": round(float(rule_signal), 6),
            "rarity_signal": round(float(rarity), 6),
            "pattern_signal": round(float(pattern_signal), 6),
            "neighbor_signal": round(float(neighbor_signal), 6),
            "dataset_consistency_signal": round(float(dataset_consistency_score), 6),
        },
        "note": "posterior is a heuristic Bayesian-style approximation",
    }


def build_llm_context_text(
    row_id, col, value,
    row_ctx, col_ctx, conflict_ctx, sim_ctx, bayes_ctx,
    cfg: DatasetConfig,
    dataset_ctx: Optional[Dict[str, Any]] = None,
    attribution_features: Optional[Dict[str, Any]] = None,
    include_details: bool = True,
) -> str:
    lines = [
        "Candidate cell:",
        f"- row_id: {row_id}",
        f"- column: {col}",
        f"- current value: {value}",
        "",
        "Row context:",
        f"- missing_count: {row_ctx['missing_count']}",
        f"- non_missing_count: {row_ctx['non_missing_count']}",
        f"- avg_value_length: {row_ctx['avg_value_length']}",
        "- row field values:",
    ]
    for k, v in row_ctx["row_values_for_llm"]:
        lines.append(f"  - {k}: {v}")

    if cfg.name == "flights":
        lines += ["", "Time bundle:", f"- {row_ctx.get('time_bundle', {})}"]
    elif cfg.name == "beers":
        lines += ["", "Numeric snapshot:", f"- {row_ctx.get('numeric_snapshot', {})}"]
    elif cfg.name == "rayyan":
        lines += ["", "Metadata bundle:", f"- {row_ctx.get('metadata_bundle', {})}"]
    elif cfg.name == "adult":
        lines += ["", "Adult profile bundle:", f"- {row_ctx.get('adult_profile_bundle', {})}"]
    elif cfg.name == "soccer":
        lines += ["", "Soccer profile bundle:", f"- {row_ctx.get('soccer_profile_bundle', {})}"]

    lines += [
        "",
        "Column context:",
        f"- detected_type: {col_ctx['detected_type']}",
        f"- semantic_type: {col_ctx['semantic_type']}",
        f"- missing_rate: {col_ctx['missing_rate']}",
        f"- unique_rate: {col_ctx['unique_rate']}",
        f"- value_frequency: {col_ctx['value_frequency']}",
        f"- value_frequency_rank: {col_ctx['value_frequency_rank']}",
        f"- current_pattern: {col_ctx['current_value_pattern']}",
        f"- dominant_pattern: {col_ctx['dominant_pattern']}",
        f"- charset_features: {col_ctx['charset_features']}",
    ]

    if cfg.name == "flights":
        lines += [f"- time_value_minutes: {col_ctx.get('time_value_minutes')}", f"- time_stats: {col_ctx.get('time_stats')}"]
    elif cfg.name == "beers":
        lines += [f"- numeric_value: {col_ctx.get('numeric_value')}"]
    elif cfg.name == "rayyan":
        lines += [
            f"- numeric_value: {col_ctx.get('numeric_value')}",
            f"- date_value: {col_ctx.get('date_value')}",
            f"- language_canonical: {col_ctx.get('language_canonical')}",
            f"- issn_canonical: {col_ctx.get('issn_canonical')}",
        ]
    elif cfg.name == "adult":
        lines += [f"- adult_value_features: {col_ctx.get('adult_value_features')}"]
    elif cfg.name == "soccer":
        lines += [f"- soccer_value_features: {col_ctx.get('soccer_value_features')}"]

    if include_details:
        lines.append("- top values with counts:")
        for item in col_ctx["top_values_with_counts"][:cfg.top_k_column_values]:
            lines.append(f"  - {item['value']}: {item['count']}")
        lines.append("- top patterns with counts:")
        for item in col_ctx["top_patterns_with_counts"][:5]:
            lines.append(f"  - {item['pattern']}: count={item['count']}, ratio={item['ratio']}")

    lines += [
        "",
        "Conflict information:",
        f"- has_rule_violation: {conflict_ctx['has_rule_violation']}",
        f"- violation_count: {conflict_ctx['violation_count']}",
        f"- main_rule_type: {conflict_ctx['main_rule_type']}",
        f"- main_usage_role: {conflict_ctx['main_usage_role']}",
        f"- strong_rule_count: {conflict_ctx['strong_rule_count']}",
        f"- candidate_generation_rule_count: {conflict_ctx['candidate_generation_rule_count']}",
        f"- fd_like_count: {conflict_ctx['fd_like_count']}",
        f"- context_rule_count: {conflict_ctx['context_rule_count']}",
        f"- pattern_rule_count: {conflict_ctx['pattern_rule_count']}",
        f"- schema_rule_count: {conflict_ctx['schema_rule_count']}",
    ]

    if cfg.name == "flights":
        lines.append(f"- time_window_rule_count: {conflict_ctx.get('time_window_rule_count')}")
    elif cfg.name == "beers":
        lines.append(f"- numeric_window_rule_count: {conflict_ctx.get('numeric_window_rule_count')}")
    elif cfg.name == "rayyan":
        lines.append(f"- canonical_rule_count: {conflict_ctx.get('canonical_rule_count')}")
    elif cfg.name == "adult":
        lines += [
            f"- adult_domain_rule_count: {conflict_ctx.get('adult_domain_rule_count')}",
            f"- adult_format_rule_count: {conflict_ctx.get('adult_format_rule_count')}",
            f"- adult_consistency_rule_count: {conflict_ctx.get('adult_consistency_rule_count')}",
        ]
    elif cfg.name == "soccer":
        lines += [
            f"- soccer_domain_rule_count: {conflict_ctx.get('soccer_domain_rule_count')}",
            f"- soccer_format_rule_count: {conflict_ctx.get('soccer_format_rule_count')}",
            f"- soccer_numeric_rule_count: {conflict_ctx.get('soccer_numeric_rule_count')}",
            f"- soccer_consistency_rule_count: {conflict_ctx.get('soccer_consistency_rule_count')}",
        ]

    lines.append("- violated rules:")
    for r in conflict_ctx["violated_rules_detailed"][:20]:
        lines.append(
            f"  - rule_id={r.get('rule_id')}, rule_type={r.get('rule_type')}, "
            f"priority={r.get('priority')}, usage_role={r.get('usage_role')}, "
            f"expected_value={r.get('expected_value')}, reason={r.get('reason')}"
        )

    lines.append("- alternative values from rules/column:")
    for alt in conflict_ctx["all_alternative_values"][:cfg.top_k_alt_values]:
        lines.append(f"  - {alt}")

    if attribution_features is not None:
        title = "Adult attribution features" if cfg.name == "adult" else "Soccer attribution features"
        lines += ["", f"{title}:", f"- {attribution_features}"]

    lines += [
        "",
        "Similar rows:",
        f"- neighbor_majority_value: {sim_ctx['neighbor_majority_value']}",
        f"- neighbor_majority_ratio: {sim_ctx['neighbor_majority_ratio']}",
        "- top similar rows:",
    ]
    for sr in sim_ctx["topk_similar_rows"]:
        if "row_values" in sr:
            lines.append(
                f"  - row_id={sr['row_id']}, similarity={sr['similarity']}, "
                f"target_column_value={sr['target_column_value']}, row_values={sr['row_values']}"
            )
        else:
            lines.append(f"  - row_id={sr['row_id']}, similarity={sr['similarity']}, target_column_value={sr['target_column_value']}")

    if dataset_ctx is not None:
        label = "Adult consistency evidence" if cfg.name == "adult" else "Soccer consistency evidence"
        lines += ["", f"{label}:"]
        for k, v in dataset_ctx.items():
            lines.append(f"- {k}: {v}")

    lines += [
        "",
        "Bayesian-style probabilities:",
        f"- prior_error_probability: {bayes_ctx['prior_error_probability']}",
        f"- posterior_error_probability: {bayes_ctx['posterior_error_probability']}",
        f"- evidence: {bayes_ctx['bayes_evidence']}",
    ]
    return "\n".join(lines)


# ============================================================
# 9. Flat rows
# ============================================================

def base_flat_row(cand, row_id, col, value, col_ctx, conflict_ctx, sim_ctx, bayes_ctx):
    return {
        "row_id": row_id,
        "column": col,
        "value": value,
        "semantic_type": col_ctx["semantic_type"],
        "violation_count": cand.get("violation_count"),
        "conflict_score": cand.get("conflict_score"),
        "value_frequency": col_ctx["value_frequency"],
        "value_frequency_rank": col_ctx["value_frequency_rank"],
        "neighbor_majority_value": sim_ctx["neighbor_majority_value"],
        "neighbor_majority_ratio": sim_ctx["neighbor_majority_ratio"],
        "prior_error_probability": bayes_ctx["prior_error_probability"],
        "posterior_error_probability": bayes_ctx["posterior_error_probability"],
        "main_rule_type": conflict_ctx["main_rule_type"],
        "main_usage_role": conflict_ctx["main_usage_role"],
        "strong_rule_count": conflict_ctx["strong_rule_count"],
        "candidate_generation_rule_count": conflict_ctx["candidate_generation_rule_count"],
        "fd_like_count": conflict_ctx["fd_like_count"],
        "context_rule_count": conflict_ctx["context_rule_count"],
        "global_rule_count": conflict_ctx["global_rule_count"],
        "rare_value_count": conflict_ctx["rare_value_count"],
        "typo_rule_count": conflict_ctx["typo_rule_count"],
        "pattern_rule_count": conflict_ctx["pattern_rule_count"],
        "schema_rule_count": conflict_ctx["schema_rule_count"],
    }


def build_flat_row(cand, row_ctx, col_ctx, conflict_ctx, sim_ctx, dataset_ctx, bayes_ctx, attribution_features, cfg: DatasetConfig):
    row_id, col, value = cand["row_id"], cand["column"], cand["value"]
    row = base_flat_row(cand, row_id, col, value, col_ctx, conflict_ctx, sim_ctx, bayes_ctx)

    if cfg.name == "flights":
        row["time_value_minutes"] = col_ctx.get("time_value_minutes")
        row["time_window_rule_count"] = conflict_ctx.get("time_window_rule_count", 0)

    elif cfg.name == "beers":
        row["numeric_value"] = col_ctx.get("numeric_value")
        row["numeric_window_rule_count"] = conflict_ctx.get("numeric_window_rule_count", 0)

    elif cfg.name == "rayyan":
        row["numeric_value"] = col_ctx.get("numeric_value")
        row["date_value"] = col_ctx.get("date_value")
        row["language_canonical"] = col_ctx.get("language_canonical")
        row["issn_canonical"] = col_ctx.get("issn_canonical")
        row["canonical_rule_count"] = conflict_ctx.get("canonical_rule_count", 0)

    elif cfg.name == "adult":
        avf = col_ctx["adult_value_features"]
        apb = row_ctx["adult_profile_bundle"]
        row.update({
            "detected_type": col_ctx["detected_type"],
            "age_lower": avf["age_lower"],
            "age_upper": avf["age_upper"],
            "hours_lower": avf["hours_lower"],
            "hours_upper": avf["hours_upper"],
            "canonical_income": avf["canonical_income"],
            "domain_valid": avf["domain_valid"],
            "education_order": avf["education_order"],
            "adult_value_bucket": avf["adult_value_bucket"],
            "adult_domain_rule_count": conflict_ctx.get("adult_domain_rule_count", 0),
            "adult_format_rule_count": conflict_ctx.get("adult_format_rule_count", 0),
            "adult_consistency_rule_count": conflict_ctx.get("adult_consistency_rule_count", 0),
            "time_window_rule_count": 0,
            "time_value_minutes": np.nan,
            "time_window_bucket": "adult_not_applicable",
            "time_value_bucket": "adult_not_applicable",
            "row_missing_count": row_ctx["missing_count"],
            "row_non_missing_count": row_ctx["non_missing_count"],
            "row_age_lower": apb.get("age_lower"),
            "row_age_upper": apb.get("age_upper"),
            "row_hours_lower": apb.get("hours_lower"),
            "row_hours_upper": apb.get("hours_upper"),
            "row_education_order": apb.get("education_order"),
            "row_income_canonical": apb.get("income_canonical"),
            "adult_profile_inconsistency_count": len(apb.get("adult_profile_inconsistencies", [])),
            "adult_consistency_score": dataset_ctx.get("adult_consistency_score", 0.0),
            "adult_evidence_flags": "|".join(dataset_ctx.get("evidence_flags", [])),
            "current_value_pattern": col_ctx["current_value_pattern"],
            "dominant_pattern": col_ctx["dominant_pattern"],
            "dominant_pattern_ratio": col_ctx["dominant_pattern_ratio"],
        })
        row.update(attribution_features or {})

    elif cfg.name == "soccer":
        svf = col_ctx["soccer_value_features"]
        spb = row_ctx["soccer_profile_bundle"]
        row.update({
            "detected_type": col_ctx["detected_type"],
            "year_value": svf["year_value"],
            "canonical_position": svf["canonical_position"],
            "domain_valid": svf["domain_valid"],
            "soccer_value_bucket": svf["soccer_value_bucket"],
            "age_lower": np.nan,
            "age_upper": np.nan,
            "hours_lower": np.nan,
            "hours_upper": np.nan,
            "canonical_income": None,
            "education_order": np.nan,
            "adult_value_bucket": "soccer_not_applicable",
            "evidence_only_fd_count": conflict_ctx.get("evidence_only_fd_count", 0),
            "soccer_domain_rule_count": conflict_ctx.get("soccer_domain_rule_count", 0),
            "soccer_format_rule_count": conflict_ctx.get("soccer_format_rule_count", 0),
            "soccer_numeric_rule_count": conflict_ctx.get("soccer_numeric_rule_count", 0),
            "soccer_consistency_rule_count": conflict_ctx.get("soccer_consistency_rule_count", 0),
            "adult_domain_rule_count": conflict_ctx.get("adult_domain_rule_count", 0),
            "adult_format_rule_count": conflict_ctx.get("adult_format_rule_count", 0),
            "adult_consistency_rule_count": conflict_ctx.get("adult_consistency_rule_count", 0),
            "time_window_rule_count": 0,
            "time_value_minutes": np.nan,
            "time_window_bucket": "soccer_not_applicable",
            "time_value_bucket": "soccer_not_applicable",
            "row_missing_count": row_ctx["missing_count"],
            "row_non_missing_count": row_ctx["non_missing_count"],
            "row_birthyear_int": spb.get("birthyear_int"),
            "row_season_int": spb.get("season_int"),
            "row_age_at_season": spb.get("age_at_season"),
            "row_position_canonical": spb.get("position_canonical"),
            "row_age_lower": np.nan,
            "row_age_upper": np.nan,
            "row_hours_lower": np.nan,
            "row_hours_upper": np.nan,
            "row_education_order": np.nan,
            "row_income_canonical": None,
            "adult_profile_inconsistency_count": len(spb.get("soccer_profile_inconsistencies", [])),
            "soccer_profile_inconsistency_count": len(spb.get("soccer_profile_inconsistencies", [])),
            "soccer_consistency_score": dataset_ctx.get("soccer_consistency_score", 0.0),
            "soccer_evidence_flags": "|".join(dataset_ctx.get("evidence_flags", [])),
            "adult_consistency_score": dataset_ctx.get("soccer_consistency_score", 0.0),
            "adult_evidence_flags": "|".join(dataset_ctx.get("evidence_flags", [])),
            "current_value_pattern": col_ctx["current_value_pattern"],
            "dominant_pattern": col_ctx["dominant_pattern"],
            "dominant_pattern_ratio": col_ctx["dominant_pattern_ratio"],
        })
        row.update(attribution_features or {})

    return row


# ============================================================
# 10. Main processing
# ============================================================

def process_one_candidate(
    cand: Dict[str, Any],
    df: pd.DataFrame,
    col_profiles: Dict[str, Dict[str, Any]],
    candidate_stats: Dict[str, Any],
    row_contexts: List[Dict[str, Any]],
    neighbor_indices: Dict[int, List[Tuple[int, float]]],
    cfg: DatasetConfig,
    group_maps=None,
    disable_bayesian=False,
    disable_dataset_consistency=False,
    disable_context_text_details=False,
):
    row_id, col, value = cand["row_id"], cand["column"], cand["value"]
    row_ctx = row_contexts[row_id]
    col_ctx = summarize_column_context(col, value, col_profiles, cfg)
    conflict_ctx = summarize_conflict_info(cand, cfg)
    add_alternative_values(conflict_ctx, col_ctx, cfg)
    sim_ctx = summarize_similarity_info_fast(df, row_id, col, neighbor_indices, cfg)

    dataset_ctx = None
    attribution_features = None
    consistency_score = 0.0

    if cfg.name == "adult":
        dataset_ctx = summarize_adult_consistency_info(row_ctx, col, value, conflict_ctx)
        attribution_features = build_adult_v2_attribution_features(row_ctx, col, value, conflict_ctx)
        consistency_score = 0.0 if disable_dataset_consistency else dataset_ctx.get("adult_consistency_score", 0.0)

    elif cfg.name == "soccer":
        dataset_ctx = summarize_soccer_consistency_info(df, row_id, col, value, row_ctx, group_maps or ({}, {}, {}))
        attribution_features = build_soccer_attribution_features(row_ctx, col, value, conflict_ctx)
        consistency_score = 0.0 if disable_dataset_consistency else dataset_ctx.get("soccer_consistency_score", 0.0)

    if disable_bayesian:
        bayes_ctx = {
            "prior_error_probability": 0.0,
            "posterior_error_probability": 0.0,
            "bayes_evidence": {
                "rule_signal": 0.0,
                "rarity_signal": 0.0,
                "pattern_signal": 0.0,
                "neighbor_signal": 0.0,
                "dataset_consistency_signal": 0.0,
            },
            "note": "disabled by ablation",
        }
    else:
        bayes_ctx = summarize_bayesian_info(cand, df, col_profiles, candidate_stats, sim_ctx, consistency_score, cfg)

    llm_context_text = build_llm_context_text(
        row_id, col, value, row_ctx, col_ctx, conflict_ctx, sim_ctx, bayes_ctx,
        cfg, dataset_ctx=dataset_ctx, attribution_features=attribution_features,
        include_details=not disable_context_text_details,
    )

    item = {
        "row_id": row_id,
        "column": col,
        "value": value,
        "violation_count": cand.get("violation_count"),
        "conflict_score": cand.get("conflict_score"),
        "row_context": row_ctx,
        "column_context": col_ctx,
        "conflict_context": conflict_ctx,
        "similarity_context": sim_ctx,
        "bayesian_context": bayes_ctx,
        "llm_context_text": llm_context_text,
    }

    if cfg.name == "adult":
        item["adult_consistency_context"] = dataset_ctx
        item["adult_v2_attribution_features"] = attribution_features
    elif cfg.name == "soccer":
        item["soccer_consistency_context"] = dataset_ctx
        item["soccer_attribution_features"] = attribution_features
        # 兼容旧 adult 字段
        item["adult_consistency_context"] = dataset_ctx

    flat_row = build_flat_row(cand, row_ctx, col_ctx, conflict_ctx, sim_ctx, dataset_ctx or {}, bayes_ctx, attribution_features, cfg)
    return item, flat_row


def summarize_contexts(
    input_table_csv: str,
    input_shrunk_csv: str,
    output_jsonl: str,
    output_csv: str,
    cfg: DatasetConfig,
    args,
):
    t0 = time.time()
    print("[1/7] Loading table and candidates ...")
    df = load_table(input_table_csv, cfg)
    candidates = load_shrunk_candidates(input_shrunk_csv, cfg)
    print(f"  table shape = {df.shape}, candidates = {len(candidates)}")

    print("[2/7] Analyzing columns ...")
    col_profiles = analyze_columns(df, cfg)
    candidate_stats = build_candidate_stats(candidates)

    print("[3/7] Precomputing row contexts ...")
    row_contexts = [summarize_row_context(df, i, col_profiles, cfg) for i in range(len(df))]

    print("[4/7] Precomputing dataset-specific group distributions ...")
    group_maps = None
    if cfg.name == "soccer":
        group_maps = precompute_soccer_group_distributions(df, cfg)
        print(f"  team-season groups = {len(group_maps[0])}, team groups = {len(group_maps[1])}, stadium-city groups = {len(group_maps[2])}")
    else:
        group_maps = ({}, {}, {})

    print("[5/7] Precomputing similar rows for candidate rows ...")
    if args.disable_neighbor_evidence or env_bool("DISABLE_NEIGHBOR_EVIDENCE", False):
        neighbor_indices = {c["row_id"]: [] for c in candidates}
    else:
        X = build_similarity_matrix(df, col_profiles)
        neighbor_indices = precompute_neighbor_indices_for_candidates(
            X,
            [c["row_id"] for c in candidates],
            top_k=cfg.top_k_similar_rows,
            batch_size=args.similarity_batch_size or cfg.similarity_batch_size,
        )
    print(f"  neighbor index rows = {len(neighbor_indices)}")

    print("[6/7] Building contexts ...")
    disable_bayesian = args.disable_bayesian_evidence or env_bool("DISABLE_BAYESIAN_EVIDENCE", False)
    disable_dataset_consistency = args.disable_dataset_consistency or env_bool("DISABLE_DATASET_CONSISTENCY", False)
    disable_context_text_details = args.disable_context_text_details or env_bool("DISABLE_CONTEXT_TEXT_DETAILS", False)

    flat_rows = []
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for i, cand in enumerate(candidates, start=1):
            item, flat_row = process_one_candidate(
                cand, df, col_profiles, candidate_stats, row_contexts, neighbor_indices,
                cfg, group_maps=group_maps,
                disable_bayesian=disable_bayesian,
                disable_dataset_consistency=disable_dataset_consistency,
                disable_context_text_details=disable_context_text_details,
            )
            f.write(json.dumps(item, ensure_ascii=False, default=json_default) + "\n")
            flat_rows.append(flat_row)
            if args.progress_every and i % args.progress_every == 0:
                print(f"  processed {i}/{len(candidates)}, elapsed={time.time() - t0:.1f}s")

    print("[7/7] Saving flat CSV ...")
    pd.DataFrame(flat_rows).to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"[OK] Full LLM contexts saved to: {output_jsonl}")
    print(f"[OK] Flat summary saved to: {output_csv}")
    print(f"[INFO] Context rows: {len(candidates)}")
    print(f"[INFO] Total elapsed seconds: {time.time() - t0:.2f}")


# ============================================================
# 11. CLI
# ============================================================

def run_one_dataset(dataset: str, args):
    if dataset not in CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset}. Available: {sorted(CONFIGS)}")
    cfg = CONFIGS[dataset]

    cfg.include_similar_row_values = not args.no_similar_row_values

    input_table_csv = args.input_table_csv or cfg.input_table_csv
    input_shrunk_csv = args.input_shrunk_csv or cfg.input_shrunk_csv
    output_jsonl = args.output_jsonl or cfg.output_jsonl
    output_csv = args.output_csv or cfg.output_csv

    old_cwd = None
    if args.cwd:
        os.makedirs(args.cwd, exist_ok=True)
        old_cwd = os.getcwd()
        os.chdir(args.cwd)

    try:
        summarize_contexts(input_table_csv, input_shrunk_csv, output_jsonl, output_csv, cfg, args)
    finally:
        if old_cwd is not None:
            os.chdir(old_cwd)


def parse_args():
    parser = argparse.ArgumentParser(description="Unified full-context generator for error detection datasets.")
    parser.add_argument("--dataset", required=True, choices=sorted(list(CONFIGS.keys()) + ["all"]))
    parser.add_argument("--input_table_csv", default=None)
    parser.add_argument("--input_shrunk_csv", default=None)
    parser.add_argument("--output_jsonl", default=None)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--cwd", default=None)

    # Ablation-friendly evidence switches
    parser.add_argument("--disable-neighbor-evidence", action="store_true")
    parser.add_argument("--disable-bayesian-evidence", action="store_true")
    parser.add_argument("--disable-dataset-consistency", action="store_true")
    parser.add_argument("--disable-context-text-details", action="store_true")

    # Runtime options
    parser.add_argument("--no-similar-row-values", action="store_true", help="Do not include full row_values in similar rows.")
    parser.add_argument("--similarity-batch-size", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dataset == "all":
        if args.input_table_csv or args.input_shrunk_csv or args.output_jsonl or args.output_csv:
            raise ValueError("--dataset all cannot be used with input/output overrides")
        for ds in ["hospital", "flights", "beers", "rayyan", "adult", "soccer"]:
            print("\n" + "=" * 80)
            print(f"[DATASET] {ds}")
            print("=" * 80)
            run_one_dataset(ds, args)
    else:
        run_one_dataset(args.dataset, args)


if __name__ == "__main__":
    main()

"""
Soccer repair candidate feature builder.

This script is adapted from the Flights feature builder, but removes Flights-
specific time/source-consensus features and adds Soccer-specific features for:
  - Soccer entity profile-key majority evidence
  - name/surname/birthplace pair-majority evidence
  - team-season-position distribution evidence
  - birthyear/season age consistency
  - domain validity and domain similarity
  - rule / neighbor / LLM hint / co-occurrence features

Inputs
------
1. repair_candidates_expanded.csv
   Output of revise_candidate_soccer.py. Required columns:
     row_id,column,dirty_value,candidate_value
   Optional columns:
     candidate_sources, source_score, candidate_rank,
     is_from_rule, is_from_neighbor, is_from_column_top,
     is_from_similarity, is_from_hint, is_from_dictionary,
     is_from_adult_profile, is_from_adult_direct_rule, is_from_domain

2. candidate_llm_contexts_soccer.jsonl
   Rich Soccer context file.

3. Optional candidate_llm_contexts_flat_soccer.csv

4. Optional soccer_dirty.csv
   Used for fallback context and empirical co-occurrence dictionary.

Outputs
-------
repair_candidate_features_v2.csv
built_cooccurrence_dict_v2.json

Usage
-----
cd /mnt/mydata/dq/projects/Splittree/test/Error\ Correction\ soccer

python build_features_soccer.py \
  --candidates_file repair_candidates_expanded.csv \
  --context_file "/mnt/mydata/dq/projects/Splittree/test/Error Detection Soccer/candidate_llm_contexts_soccer.jsonl" \
  --fallback_context_flat_file "/mnt/mydata/dq/projects/Splittree/test/Error Detection Soccer/candidate_llm_contexts_flat_soccer.csv" \
  --raw_data_file /mnt/mydata/dq/projects/Splittree/soccer/soccer_dirty.csv \
  --output_features_csv repair_candidate_features_v2.csv
"""

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# ============================================================
# 1. Defaults
# ============================================================

DEFAULT_CANDIDATES_FILE = "repair_candidates_expanded.csv"
DEFAULT_CONTEXT_FILE = "/mnt/mydata/dq/projects/Splittree/test/Error Detection Soccer/candidate_llm_contexts_soccer.jsonl"
DEFAULT_FALLBACK_CONTEXT_FLAT_FILE = "/mnt/mydata/dq/projects/Splittree/test/Error Detection Soccer/candidate_llm_contexts_flat_soccer.csv"
DEFAULT_RAW_DATA_FILE = "/mnt/mydata/dq/projects/Splittree/soccer/soccer_dirty.csv"
DEFAULT_COOCCURRENCE_DICT_FILE = ""
DEFAULT_OUTPUT_FEATURES_CSV = "repair_candidate_features_v2.csv"
DEFAULT_OUTPUT_BUILT_COOCCUR_JSON = "built_cooccurrence_dict_v2.json"

MAX_CONTEXT_PAIRS = 12
SMOOTHING_ALPHA = 1.0
MAX_DISTINCT_VALUES_PER_COLUMN = 2000
TOP_K_SIMILAR_ROW_VALUES = 8

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
    "relationship": [
        "Wife", "Own-child", "Husband", "Not-in-family",
        "Other-relative", "Unmarried"
    ],
    "race": [
        "White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"
    ],
    "sex": ["Female", "Male"],
    "country": [
        "United-States", "Cambodia", "England", "Puerto-Rico", "Canada",
        "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan",
        "Greece", "South", "China", "Cuba", "Iran", "Honduras",
        "Philippines", "Italy", "Poland", "Jamaica", "Vietnam",
        "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic",
        "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary",
        "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia",
        "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"
    ],
    "income": ["LessThan50K", "MoreThan50K"],
}

EDUCATION_ORDER = {
    "Preschool": 1,
    "1st-4th": 2,
    "5th-6th": 3,
    "7th-8th": 4,
    "9th": 5,
    "10th": 6,
    "11th": 7,
    "12th": 8,
    "HS-grad": 9,
    "Some-college": 10,
    "Assoc-voc": 11,
    "Assoc-acdm": 12,
    "Bachelors": 13,
    "Masters": 14,
    "Prof-school": 15,
    "Doctorate": 16,
}

SPOUSE_RELATIONSHIPS = {"Husband", "Wife"}
MARRIED_STATUSES = {"Married-civ-spouse", "Married-AF-spouse"}


SOCCER_DOMAIN_VALUES = {
    "position": ["Goalkeeper", "Defender", "Midfield", "Midfielder", "Forward", "Striker", "Winger"],
}
POSITION_ALIASES = {
    "gk": "Goalkeeper", "goalkeeper": "Goalkeeper", "keeper": "Goalkeeper",
    "df": "Defender", "def": "Defender", "defender": "Defender", "back": "Defender",
    "mf": "Midfield", "mid": "Midfield", "midfield": "Midfield", "midfielder": "Midfield",
    "fw": "Forward", "fwd": "Forward", "forward": "Forward", "striker": "Forward",
}
SOCCER_PRIMARY_COLUMNS = {"name", "surname", "birthyear", "birthplace", "position", "team", "city", "stadium", "season", "manager"}
SOCCER_CONTEXT_COLUMNS = {"team", "city", "stadium", "manager", "season", "birthyear"}



# ============================================================
# 2. Basic helpers
# ============================================================

def norm_text(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip().lower()


def canonical_empty_text(x: Any) -> str:
    s = norm_text(x)
    if s in EMPTY_TOKENS:
        return "empty"
    if x is None:
        return "empty"
    try:
        if pd.isna(x):
            return "empty"
    except Exception:
        pass
    return str(x).strip()


def is_empty_like_value(x: Any) -> bool:
    return norm_text(x) in EMPTY_TOKENS


def safe_float(x: Any, default: float = 0.0) -> float:
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


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(float(str(x).strip()))
    except Exception:
        return default


def path_exists_nonempty(path: str) -> bool:
    return path is not None and str(path).strip() != "" and os.path.exists(str(path))


def resolve_output_path(path: str) -> str:
    path = str(path)
    if os.path.isabs(path):
        return path
    return os.path.abspath(path)


def string_similarity(a: Any, b: Any) -> float:
    a = norm_text(a)
    b = norm_text(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def try_parse_float(x: Any) -> Optional[float]:
    s = canonical_empty_text(x)
    if s == "empty":
        return None
    s = s.replace(",", "").replace("%", "").strip()
    try:
        return float(s)
    except Exception:
        return None


def normalize_col_name(column: Any) -> str:
    c = norm_text(column)
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
        "capital_gain": "capital_gain",
        "capital-loss": "capital_loss",
        "capital_loss": "capital_loss",
        "education-num": "education_num",
        "education_num": "education_num",

        # Soccer aliases
        "birth_year": "birthyear",
        "birth year": "birthyear",
        "birth-year": "birthyear",
        "player_position": "position",
        "player position": "position",
        "club": "team",
        "football_club": "team",
        "football club": "team",
        "home_city": "city",
        "home city": "city",
        "arena": "stadium",
        "coach": "manager",
        "season_year": "season",
        "season year": "season",
        "year": "season",
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
            if norm_text(v) == norm_text(raw):
                return v
        raw2 = raw.replace("_", "-")
        for v in domains:
            if norm_text(v) == norm_text(raw2):
                return v
        raw3 = raw.replace("_", " ")
        for v in domains:
            if norm_text(v).replace("-", " ") == norm_text(raw3).replace("-", " "):
                return v

    return raw


def parse_numeric_range(x: Any) -> Tuple[Optional[float], Optional[float]]:
    s = canonical_empty_text(x)
    if s == "empty":
        return None, None
    low = s.lower().replace(" ", "")

    if low.startswith("<"):
        v = try_parse_float(low[1:])
        if v is not None:
            return 0.0, v - 1
    if low.startswith(">"):
        v = try_parse_float(low[1:])
        if v is not None:
            return v + 1, 120.0

    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", s)
    if m:
        return float(m.group(1)), float(m.group(2))

    v = try_parse_float(s)
    if v is not None:
        return v, v

    return None, None


def canonical_age_bucket(x: Any) -> str:
    lo, hi = parse_numeric_range(x)
    if lo is None or hi is None:
        return canonical_empty_text(x)

    if hi <= 17:
        return "<18"
    if lo <= 21 and hi <= 21:
        return "18-21"
    if lo <= 30 and hi <= 30:
        return "22-30"
    if lo <= 50 and hi <= 50:
        return "31-50"
    return "51+"


def canonical_hours_value(x: Any) -> str:
    lo, hi = parse_numeric_range(x)
    if lo is None or hi is None:
        return canonical_empty_text(x)

    if lo == hi:
        return str(int(lo)) if abs(lo - int(lo)) < 1e-9 else str(lo)
    return f"{int(lo)}-{int(hi)}"


def education_order_value(x: Any) -> Optional[int]:
    v = canonical_domain_value("education", x)
    return EDUCATION_ORDER.get(v)


def get_row_val(row_values: Dict[str, Any], *names: str) -> str:
    if not isinstance(row_values, dict):
        return "empty"
    for n in names:
        if n in row_values:
            return canonical_empty_text(row_values.get(n))
        nn = normalize_col_name(n)
        for k, v in row_values.items():
            if normalize_col_name(k) == nn:
                return canonical_empty_text(v)
    return "empty"


# ============================================================
# 3. Keyboard and phonetic features
# ============================================================

KEYBOARD_ROWS = [
    "1234567890",
    "qwertyuiop",
    "asdfghjkl",
    "zxcvbnm",
]
KEYBOARD_POS = {}
for r, row in enumerate(KEYBOARD_ROWS):
    for c, ch in enumerate(row):
        KEYBOARD_POS[ch] = (r, c)


def char_keyboard_distance(a: Any, b: Any) -> Optional[float]:
    a = norm_text(a)
    b = norm_text(b)
    if len(a) != 1 or len(b) != 1:
        return None
    if a not in KEYBOARD_POS or b not in KEYBOARD_POS:
        return None
    ra, ca = KEYBOARD_POS[a]
    rb, cb = KEYBOARD_POS[b]
    return float(abs(ra - rb) + abs(ca - cb))


def avg_keyboard_distance_for_same_length(a: Any, b: Any) -> Optional[float]:
    a = norm_text(a)
    b = norm_text(b)
    if len(a) != len(b) or len(a) == 0:
        return None
    dists = []
    for x, y in zip(a, b):
        if x == y:
            dists.append(0.0)
        else:
            d = char_keyboard_distance(x, y)
            dists.append(5.0 if d is None else d)
    return sum(dists) / len(dists)


def soundex_token(token: Any) -> str:
    token = norm_text(token)
    if not token:
        return ""

    mapping = {
        **{c: "1" for c in "bfpv"},
        **{c: "2" for c in "cgjkqsxz"},
        **{c: "3" for c in "dt"},
        **{c: "4" for c in "l"},
        **{c: "5" for c in "mn"},
        **{c: "6" for c in "r"},
    }
    first = token[0].upper()
    codes = []
    prev = mapping.get(token[0], "")

    for ch in token[1:]:
        code = mapping.get(ch, "")
        if code != prev and code:
            codes.append(code)
        prev = code

    return (first + "".join(codes) + "000")[:4]


def phrase_soundex(s: Any) -> str:
    s = norm_text(s)
    toks = [t for t in re.split(r"[-/\s]+", s) if t]
    return " ".join(soundex_token(t) for t in toks)


def phonetic_similarity(a: Any, b: Any) -> float:
    return string_similarity(phrase_soundex(a), phrase_soundex(b))


# ============================================================
# 4. Context loading
# ============================================================

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSONL 解析失败: {path}, line={line_no}, err={e}") from e
    return rows


def load_context_records(path: str) -> List[Dict[str, Any]]:
    lower = path.lower()
    if lower.endswith(".jsonl"):
        return load_jsonl(path)
    df = pd.read_csv(path, encoding="utf-8-sig")
    return df.to_dict(orient="records")


def load_dirty_table(path: str) -> Optional[pd.DataFrame]:
    if path_exists_nonempty(path):
        return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    return None


def extract_context_from_rich(obj: Dict[str, Any]) -> Dict[str, Any]:
    row_id = safe_int(obj.get("row_id", -1), -1)
    column = str(obj.get("column", ""))
    value = canonical_empty_text(obj.get("value", ""))

    row_values = {}
    row_context = obj.get("row_context", {})
    if isinstance(row_context, dict):
        rv = row_context.get("row_values", {})
        if isinstance(rv, dict):
            row_values = {str(k): canonical_empty_text(v) for k, v in rv.items()}

    column_context = obj.get("column_context", {}) if isinstance(obj.get("column_context", {}), dict) else {}
    conflict_context = obj.get("conflict_context", {}) if isinstance(obj.get("conflict_context", {}), dict) else {}
    similarity_context = obj.get("similarity_context", {}) if isinstance(obj.get("similarity_context", {}), dict) else {}
    soccer_consistency_context = obj.get("soccer_consistency_context", {}) if isinstance(obj.get("soccer_consistency_context", {}), dict) else {}
    adult_consistency_context = obj.get("adult_consistency_context", {}) if isinstance(obj.get("adult_consistency_context", {}), dict) else {}
    # Older scripts sometimes used adult_v2_attribution_features; Soccer scripts usually use soccer_attribution_features.
    soccer_attr = obj.get("soccer_attribution_features", {}) if isinstance(obj.get("soccer_attribution_features", {}), dict) else {}
    adult_v2 = obj.get("adult_v2_attribution_features", {}) if isinstance(obj.get("adult_v2_attribution_features", {}), dict) else {}
    if soccer_attr:
        adult_v2 = {**adult_v2, **soccer_attr}
    bayes = obj.get("bayesian_context", {}) if isinstance(obj.get("bayesian_context", {}), dict) else {}

    rule_expected_values = []
    expected_values_from_rules = []
    alternative_values_from_column = []

    for vr in conflict_context.get("violated_rules_detailed", []) or []:
        if isinstance(vr, dict):
            ev = canonical_empty_text(vr.get("expected_value", ""))
            if ev not in {"", "empty"}:
                rule_expected_values.append(ev)

    for item in conflict_context.get("expected_values_from_rules", []) or []:
        if isinstance(item, dict):
            ev = canonical_empty_text(item.get("value", item.get("expected_value", "")))
            if ev not in {"", "empty"}:
                expected_values_from_rules.append(ev)

    for item in conflict_context.get("alternative_values_from_column", []) or []:
        if isinstance(item, dict):
            val = canonical_empty_text(item.get("value", ""))
            if val not in {"", "empty"}:
                alternative_values_from_column.append({
                    "value": val,
                    "count": safe_int(item.get("count", 0), 0),
                    "ratio": safe_float(item.get("ratio", 0.0), 0.0),
                })

    column_top_values = []
    for item in column_context.get("top_values_with_counts", []) or []:
        if isinstance(item, dict):
            val = canonical_empty_text(item.get("value", ""))
            if val not in {"", "empty"}:
                column_top_values.append({
                    "value": val,
                    "count": safe_int(item.get("count", 0), 0),
                    "ratio": safe_float(item.get("ratio", 0.0), 0.0),
                })

    similar_row_target_values = []
    for item in similarity_context.get("topk_similar_rows", []) or []:
        if isinstance(item, dict):
            tv = canonical_empty_text(item.get("target_column_value", ""))
            if tv not in {"", "empty"}:
                similar_row_target_values.append(tv)

    adult_profile_bundle = {}
    soccer_profile_bundle = {}
    evidence_flags = []
    same_profile_distribution = {}
    work_profile_distribution = {}
    soccer_distribution_features = {}
    adult_consistency_score = 0.0
    soccer_consistency_score = 0.0

    if isinstance(soccer_consistency_context, dict):
        soccer_profile_bundle = soccer_consistency_context.get("soccer_profile_bundle", {}) or {}
        if not isinstance(soccer_profile_bundle, dict):
            soccer_profile_bundle = {}
        evidence_flags = soccer_consistency_context.get("evidence_flags", []) or []
        if isinstance(soccer_profile_bundle, dict):
            evidence_flags += soccer_profile_bundle.get("soccer_profile_inconsistencies", []) or []
        soccer_distribution_features = {
            "team_season_distribution": soccer_consistency_context.get("team_season_distribution", {}) or {},
            "team_distribution": soccer_consistency_context.get("team_distribution", {}) or {},
            "stadium_city_distribution": soccer_consistency_context.get("stadium_city_distribution", {}) or {},
        }
        soccer_consistency_score = safe_float(soccer_consistency_context.get("soccer_consistency_score", obj.get("soccer_consistency_score", 0.0)), 0.0)

    if isinstance(adult_consistency_context, dict):
        adult_profile_bundle = adult_consistency_context.get("adult_profile_bundle", {}) or {}
        if not isinstance(adult_profile_bundle, dict):
            adult_profile_bundle = {}
        if not evidence_flags:
            evidence_flags = adult_consistency_context.get("evidence_flags", []) or []
        if isinstance(adult_profile_bundle, dict):
            evidence_flags += adult_profile_bundle.get("adult_profile_inconsistencies", []) or []
        same_profile_distribution = adult_consistency_context.get("same_profile_distribution", {}) or {}
        work_profile_distribution = adult_consistency_context.get("work_profile_distribution", {}) or {}
        adult_consistency_score = safe_float(adult_consistency_context.get("adult_consistency_score", 0.0), 0.0)

    be = bayes.get("bayes_evidence", {}) if isinstance(bayes, dict) else {}
    if not isinstance(be, dict):
        be = {}

    return {
        "row_id": row_id,
        "column": column,
        "value": value,
        "semantic_type": obj.get("semantic_type", column_context.get("semantic_type", "")),
        "detected_type": obj.get("detected_type", column_context.get("detected_type", "")),
        "violation_count": obj.get("violation_count", conflict_context.get("violation_count", 0)),
        "conflict_score": obj.get("conflict_score", 0.0),
        "main_rule_type": obj.get("main_rule_type", conflict_context.get("main_rule_type", "")),
        "main_usage_role": obj.get("main_usage_role", conflict_context.get("main_usage_role", "")),
        "row_values": row_values,
        "column_top_values": column_top_values,
        "rule_expected_values": rule_expected_values,
        "expected_values_from_rules": expected_values_from_rules,
        "alternative_values_from_column": alternative_values_from_column,
        "neighbor_majority_value": similarity_context.get("neighbor_majority_value", obj.get("neighbor_majority_value", "")),
        "neighbor_majority_ratio": safe_float(similarity_context.get("neighbor_majority_ratio", obj.get("neighbor_majority_ratio", 0.0)), 0.0),
        "similar_row_target_values": similar_row_target_values[:TOP_K_SIMILAR_ROW_VALUES],
        "suggested_correct_value": obj.get("suggested_correct_value", ""),
        "prior_error_probability": safe_float(bayes.get("prior_error_probability", obj.get("prior_error_probability", 0.0)), 0.0),
        "posterior_error_probability": safe_float(bayes.get("posterior_error_probability", obj.get("posterior_error_probability", 0.0)), 0.0),
        "bayes_rule_signal": safe_float(be.get("rule_signal", 0.0), 0.0),
        "bayes_rarity_signal": safe_float(be.get("rarity_signal", 0.0), 0.0),
        "bayes_pattern_signal": safe_float(be.get("pattern_signal", 0.0), 0.0),
        "bayes_neighbor_signal": safe_float(be.get("neighbor_signal", 0.0), 0.0),
        "bayes_adult_consistency_signal": safe_float(be.get("adult_consistency_signal", 0.0), 0.0),
        "bayes_soccer_consistency_signal": safe_float(be.get("soccer_consistency_signal", 0.0), 0.0),
        "strong_rule_count": safe_int(conflict_context.get("strong_rule_count", obj.get("strong_rule_count", 0)), 0),
        "candidate_generation_rule_count": safe_int(conflict_context.get("candidate_generation_rule_count", obj.get("candidate_generation_rule_count", 0)), 0),
        "fd_like_count": safe_int(conflict_context.get("fd_like_count", obj.get("fd_like_count", 0)), 0),
        "context_rule_count": safe_int(conflict_context.get("context_rule_count", obj.get("context_rule_count", 0)), 0),
        "global_rule_count": safe_int(conflict_context.get("global_rule_count", obj.get("global_rule_count", 0)), 0),
        "rare_value_count": safe_int(conflict_context.get("rare_value_count", obj.get("rare_value_count", 0)), 0),
        "typo_rule_count": safe_int(conflict_context.get("typo_rule_count", obj.get("typo_rule_count", 0)), 0),
        "pattern_rule_count": safe_int(conflict_context.get("pattern_rule_count", obj.get("pattern_rule_count", 0)), 0),
        "schema_rule_count": safe_int(conflict_context.get("schema_rule_count", obj.get("schema_rule_count", 0)), 0),
        "adult_domain_rule_count": safe_int(conflict_context.get("adult_domain_rule_count", obj.get("adult_domain_rule_count", 0)), 0),
        "adult_format_rule_count": safe_int(conflict_context.get("adult_format_rule_count", obj.get("adult_format_rule_count", 0)), 0),
        "adult_consistency_rule_count": safe_int(conflict_context.get("adult_consistency_rule_count", obj.get("adult_consistency_rule_count", 0)), 0),
        "soccer_domain_rule_count": safe_int(conflict_context.get("soccer_domain_rule_count", obj.get("soccer_domain_rule_count", 0)), 0),
        "soccer_format_rule_count": safe_int(conflict_context.get("soccer_format_rule_count", obj.get("soccer_format_rule_count", 0)), 0),
        "soccer_numeric_rule_count": safe_int(conflict_context.get("soccer_numeric_rule_count", obj.get("soccer_numeric_rule_count", 0)), 0),
        "soccer_consistency_rule_count": safe_int(conflict_context.get("soccer_consistency_rule_count", obj.get("soccer_consistency_rule_count", 0)), 0),
        "adult_profile_bundle": adult_profile_bundle,
        "soccer_profile_bundle": soccer_profile_bundle,
        "soccer_distribution_features": soccer_distribution_features,
        "adult_evidence_flags": list(dict.fromkeys([str(x) for x in evidence_flags if str(x).strip()])),
        "soccer_evidence_flags": list(dict.fromkeys([str(x) for x in (obj.get("soccer_evidence_flags", evidence_flags) or []) if str(x).strip()])) if not isinstance(obj.get("soccer_evidence_flags", evidence_flags), str) else str(obj.get("soccer_evidence_flags", "")).split("|"),
        "same_profile_distribution": same_profile_distribution if isinstance(same_profile_distribution, dict) else {},
        "work_profile_distribution": work_profile_distribution if isinstance(work_profile_distribution, dict) else {},
        "adult_consistency_score": adult_consistency_score,
        "soccer_consistency_score": soccer_consistency_score if soccer_consistency_score != 0.0 else safe_float(obj.get("soccer_consistency_score", conflict_context.get("soccer_consistency_score", 0.0)), 0.0),
        "soccer_profile_inconsistency_count": safe_int(obj.get("soccer_profile_inconsistency_count", conflict_context.get("soccer_profile_inconsistency_count", 0)), 0),
        "adult_v2_attribution_features": adult_v2,
        "target_is_primary_column": safe_int(adult_v2.get("target_is_primary_column", obj.get("target_is_primary_column", 0)), 0),
        "target_is_context_only_column": safe_int(adult_v2.get("target_is_context_only_column", obj.get("target_is_context_only_column", 0)), 0),
        "is_relationship_spouse_value": safe_int(adult_v2.get("is_relationship_spouse_value", obj.get("is_relationship_spouse_value", 0)), 0),
        "relationship_marital_conflict": safe_int(adult_v2.get("relationship_marital_conflict", obj.get("relationship_marital_conflict", 0)), 0),
        "relationship_age_conflict": safe_int(adult_v2.get("relationship_age_conflict", obj.get("relationship_age_conflict", 0)), 0),
        "relationship_sex_conflict": safe_int(adult_v2.get("relationship_sex_conflict", obj.get("relationship_sex_conflict", 0)), 0),
        "sex_value_invalid": safe_int(adult_v2.get("sex_value_invalid", obj.get("sex_value_invalid", 0)), 0),
        "education_age_extreme_conflict": safe_int(adult_v2.get("education_age_extreme_conflict", obj.get("education_age_extreme_conflict", 0)), 0),
        "has_relationship_v2_rule": safe_int(adult_v2.get("has_relationship_v2_rule", obj.get("has_relationship_v2_rule", 0)), 0),
        "has_education_v2_rule": safe_int(adult_v2.get("has_education_v2_rule", obj.get("has_education_v2_rule", 0)), 0),
    }


def build_context_map(context_records: List[Dict[str, Any]]) -> Dict[Tuple[int, str], Dict[str, Any]]:
    cmap = {}
    for obj in context_records:
        if "row_id" not in obj or "column" not in obj:
            continue
        ctx = extract_context_from_rich(obj)
        if ctx["row_id"] >= 0 and ctx["column"]:
            cmap[(ctx["row_id"], ctx["column"])] = ctx
    return cmap


def merge_flat_into_context_map(cmap: Dict[Tuple[int, str], Dict[str, Any]], flat_records: List[Dict[str, Any]]):
    for obj in flat_records:
        if "row_id" not in obj or "column" not in obj:
            continue
        rid = safe_int(obj.get("row_id", -1), -1)
        col = str(obj.get("column", ""))
        if rid < 0 or not col:
            continue

        key = (rid, col)
        if key not in cmap:
            cmap[key] = {
                "row_id": rid,
                "column": col,
                "value": canonical_empty_text(obj.get("value", "")),
                "semantic_type": obj.get("semantic_type", ""),
                "detected_type": obj.get("detected_type", ""),
                "violation_count": obj.get("violation_count", 0),
                "conflict_score": obj.get("conflict_score", 0.0),
                "main_rule_type": obj.get("main_rule_type", ""),
                "main_usage_role": obj.get("main_usage_role", ""),
                "row_values": {},
                "column_top_values": [],
                "rule_expected_values": [],
                "expected_values_from_rules": [],
                "alternative_values_from_column": [],
                "neighbor_majority_value": obj.get("neighbor_majority_value", ""),
                "neighbor_majority_ratio": safe_float(obj.get("neighbor_majority_ratio", 0.0), 0.0),
                "similar_row_target_values": [],
                "suggested_correct_value": obj.get("suggested_correct_value", ""),
                "prior_error_probability": safe_float(obj.get("prior_error_probability", 0.0), 0.0),
                "posterior_error_probability": safe_float(obj.get("posterior_error_probability", 0.0), 0.0),
                "bayes_rule_signal": 0.0,
                "bayes_rarity_signal": 0.0,
                "bayes_pattern_signal": 0.0,
                "bayes_neighbor_signal": 0.0,
                "bayes_adult_consistency_signal": 0.0,
                "strong_rule_count": safe_int(obj.get("strong_rule_count", 0), 0),
                "candidate_generation_rule_count": safe_int(obj.get("candidate_generation_rule_count", 0), 0),
                "fd_like_count": safe_int(obj.get("fd_like_count", 0), 0),
                "context_rule_count": safe_int(obj.get("context_rule_count", 0), 0),
                "global_rule_count": safe_int(obj.get("global_rule_count", 0), 0),
                "rare_value_count": safe_int(obj.get("rare_value_count", 0), 0),
                "typo_rule_count": safe_int(obj.get("typo_rule_count", 0), 0),
                "pattern_rule_count": safe_int(obj.get("pattern_rule_count", 0), 0),
                "schema_rule_count": safe_int(obj.get("schema_rule_count", 0), 0),
                "adult_domain_rule_count": safe_int(obj.get("adult_domain_rule_count", 0), 0),
                "adult_format_rule_count": safe_int(obj.get("adult_format_rule_count", 0), 0),
                "adult_consistency_rule_count": safe_int(obj.get("adult_consistency_rule_count", 0), 0),
                "soccer_domain_rule_count": safe_int(obj.get("soccer_domain_rule_count", 0), 0),
                "soccer_format_rule_count": safe_int(obj.get("soccer_format_rule_count", 0), 0),
                "soccer_numeric_rule_count": safe_int(obj.get("soccer_numeric_rule_count", 0), 0),
                "soccer_consistency_rule_count": safe_int(obj.get("soccer_consistency_rule_count", 0), 0),
                "adult_profile_bundle": {},
                "adult_evidence_flags": str(obj.get("adult_evidence_flags", "")).split("|") if obj.get("adult_evidence_flags", "") else [],
                "same_profile_distribution": {},
                "work_profile_distribution": {},
                "adult_consistency_score": safe_float(obj.get("adult_consistency_score", 0.0), 0.0),
                "soccer_consistency_score": safe_float(obj.get("soccer_consistency_score", 0.0), 0.0),
                "soccer_profile_inconsistency_count": safe_int(obj.get("soccer_profile_inconsistency_count", 0), 0),
                "adult_v2_attribution_features": {},
                "target_is_primary_column": safe_int(obj.get("target_is_primary_column", 0), 0),
                "target_is_context_only_column": safe_int(obj.get("target_is_context_only_column", 0), 0),
                "is_relationship_spouse_value": safe_int(obj.get("is_relationship_spouse_value", 0), 0),
                "relationship_marital_conflict": safe_int(obj.get("relationship_marital_conflict", 0), 0),
                "relationship_age_conflict": safe_int(obj.get("relationship_age_conflict", 0), 0),
                "relationship_sex_conflict": safe_int(obj.get("relationship_sex_conflict", 0), 0),
                "sex_value_invalid": safe_int(obj.get("sex_value_invalid", 0), 0),
                "education_age_extreme_conflict": safe_int(obj.get("education_age_extreme_conflict", 0), 0),
                "has_relationship_v2_rule": safe_int(obj.get("has_relationship_v2_rule", 0), 0),
                "has_education_v2_rule": safe_int(obj.get("has_education_v2_rule", 0), 0),
            }

        ctx = cmap[key]
        for field in [
            "semantic_type", "detected_type", "violation_count", "conflict_score",
            "main_rule_type", "main_usage_role", "neighbor_majority_value",
            "neighbor_majority_ratio", "prior_error_probability",
            "posterior_error_probability", "adult_consistency_score",
            "target_is_primary_column", "target_is_context_only_column",
            "is_relationship_spouse_value", "relationship_marital_conflict",
            "relationship_age_conflict", "relationship_sex_conflict",
            "sex_value_invalid", "education_age_extreme_conflict",
            "has_relationship_v2_rule", "has_education_v2_rule",
            "adult_domain_rule_count", "adult_format_rule_count",
            "adult_consistency_rule_count",
        ]:
            if field in obj and (ctx.get(field, "") in {"", None, 0, 0.0}):
                ctx[field] = obj.get(field)


def build_fallback_context(row: pd.Series, dirty_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    rid = safe_int(row.get("row_id", -1), -1)
    col = str(row.get("column", ""))
    dirty_value = canonical_empty_text(row.get("dirty_value", row.get("value", "")))
    row_values = {}

    if dirty_df is not None and 0 <= rid < len(dirty_df):
        row_values = {str(c): canonical_empty_text(dirty_df.at[rid, c]) for c in dirty_df.columns}
        if dirty_value in {"", "empty"} and col in dirty_df.columns:
            dirty_value = canonical_empty_text(dirty_df.at[rid, col])

    return {
        "row_id": rid,
        "column": col,
        "value": dirty_value,
        "semantic_type": row.get("semantic_type", ""),
        "detected_type": row.get("detected_type", ""),
        "violation_count": row.get("violation_count", 0),
        "conflict_score": row.get("conflict_score", 0.0),
        "main_rule_type": row.get("main_rule_type", ""),
        "main_usage_role": row.get("main_usage_role", ""),
        "row_values": row_values,
        "column_top_values": [],
        "rule_expected_values": [],
        "expected_values_from_rules": [],
        "alternative_values_from_column": [],
        "neighbor_majority_value": row.get("neighbor_majority_value", ""),
        "neighbor_majority_ratio": safe_float(row.get("neighbor_majority_ratio", 0.0), 0.0),
        "similar_row_target_values": [],
        "suggested_correct_value": row.get("suggested_correct_value", ""),
        "prior_error_probability": safe_float(row.get("prior_error_probability", 0.0), 0.0),
        "posterior_error_probability": safe_float(row.get("posterior_error_probability", 0.0), 0.0),
        "bayes_rule_signal": 0.0,
        "bayes_rarity_signal": 0.0,
        "bayes_pattern_signal": 0.0,
        "bayes_neighbor_signal": 0.0,
        "bayes_adult_consistency_signal": 0.0,
        "strong_rule_count": safe_int(row.get("strong_rule_count", 0), 0),
        "candidate_generation_rule_count": safe_int(row.get("candidate_generation_rule_count", 0), 0),
        "fd_like_count": safe_int(row.get("fd_like_count", 0), 0),
        "context_rule_count": safe_int(row.get("context_rule_count", 0), 0),
        "global_rule_count": safe_int(row.get("global_rule_count", 0), 0),
        "rare_value_count": safe_int(row.get("rare_value_count", 0), 0),
        "typo_rule_count": safe_int(row.get("typo_rule_count", 0), 0),
        "pattern_rule_count": safe_int(row.get("pattern_rule_count", 0), 0),
        "schema_rule_count": safe_int(row.get("schema_rule_count", 0), 0),
        "adult_domain_rule_count": safe_int(row.get("adult_domain_rule_count", 0), 0),
        "adult_format_rule_count": safe_int(row.get("adult_format_rule_count", 0), 0),
        "adult_consistency_rule_count": safe_int(row.get("adult_consistency_rule_count", 0), 0),
        "soccer_domain_rule_count": safe_int(row.get("soccer_domain_rule_count", 0), 0),
        "soccer_format_rule_count": safe_int(row.get("soccer_format_rule_count", 0), 0),
        "soccer_numeric_rule_count": safe_int(row.get("soccer_numeric_rule_count", 0), 0),
        "soccer_consistency_rule_count": safe_int(row.get("soccer_consistency_rule_count", 0), 0),
        "adult_profile_bundle": {},
        "adult_evidence_flags": str(row.get("adult_evidence_flags", "")).split("|") if row.get("adult_evidence_flags", "") else [],
        "same_profile_distribution": {},
        "work_profile_distribution": {},
        "adult_consistency_score": safe_float(row.get("adult_consistency_score", 0.0), 0.0),
        "soccer_consistency_score": safe_float(row.get("soccer_consistency_score", 0.0), 0.0),
        "soccer_profile_inconsistency_count": safe_int(row.get("soccer_profile_inconsistency_count", 0), 0),
        "adult_v2_attribution_features": {},
        "target_is_primary_column": safe_int(row.get("target_is_primary_column", 0), 0),
        "target_is_context_only_column": safe_int(row.get("target_is_context_only_column", 0), 0),
        "is_relationship_spouse_value": safe_int(row.get("is_relationship_spouse_value", 0), 0),
        "relationship_marital_conflict": safe_int(row.get("relationship_marital_conflict", 0), 0),
        "relationship_age_conflict": safe_int(row.get("relationship_age_conflict", 0), 0),
        "relationship_sex_conflict": safe_int(row.get("relationship_sex_conflict", 0), 0),
        "sex_value_invalid": safe_int(row.get("sex_value_invalid", 0), 0),
        "education_age_extreme_conflict": safe_int(row.get("education_age_extreme_conflict", 0), 0),
        "has_relationship_v2_rule": safe_int(row.get("has_relationship_v2_rule", 0), 0),
        "has_education_v2_rule": safe_int(row.get("has_education_v2_rule", 0), 0),
    }


# ============================================================
# 5. Co-occurrence dictionary
# ============================================================

def build_cooccurrence_from_raw_data(raw_df: pd.DataFrame) -> Dict[str, Any]:
    cooccur = {}
    usable_columns = []
    for col in raw_df.columns:
        distinct = raw_df[col].astype(str).fillna("").map(lambda x: str(x).strip()).nunique()
        if distinct <= MAX_DISTINCT_VALUES_PER_COLUMN:
            usable_columns.append(col)

    print(f"[INFO] 原始表列数: {len(raw_df.columns)}")
    print(f"[INFO] 参与共现建模的列数: {len(usable_columns)}")

    norm_df = raw_df.copy()
    for col in norm_df.columns:
        norm_df[col] = norm_df[col].fillna("").astype(str).map(canonical_empty_text)

    for target_col in usable_columns:
        cooccur[target_col] = {}
        target_series = norm_df[target_col]
        target_domain = sorted(set(v for v in target_series.tolist() if v not in {"", "empty"}))
        domain_size = max(len(target_domain), 1)

        for ctx_col in usable_columns:
            if ctx_col == target_col:
                continue
            pair_counter = Counter()
            ctx_counter = Counter()

            for _, r in norm_df[[target_col, ctx_col]].iterrows():
                v = r[target_col]
                c = r[ctx_col]
                if v in {"", "empty"} or c in {"", "empty"}:
                    continue
                pair_counter[(v, c)] += 1
                ctx_counter[c] += 1

            if not ctx_counter:
                continue

            for ctx_val, den in ctx_counter.items():
                for cand_val in target_domain:
                    num = pair_counter.get((cand_val, ctx_val), 0)
                    prob = (num + SMOOTHING_ALPHA) / (den + SMOOTHING_ALPHA * domain_size)
                    cooccur[target_col].setdefault(cand_val, {})[f"{ctx_col}={ctx_val}"] = prob

    return cooccur


def load_or_build_cooccurrence_dict(cooccur_file: str, raw_data_file: str, output_json: str) -> Optional[Dict[str, Any]]:
    if cooccur_file and os.path.exists(cooccur_file):
        print(f"[INFO] 使用现成共现字典: {cooccur_file}")
        with open(cooccur_file, "r", encoding="utf-8-sig") as f:
            return json.load(f)

    if not raw_data_file or not os.path.exists(raw_data_file):
        print("[WARN] 没有现成共现字典，也没有可用原始数据文件，贝叶斯共现特征将退化为 0。")
        return None

    print(f"[INFO] 从原始数据构建共现字典: {raw_data_file}")
    raw_df = pd.read_csv(raw_data_file, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    cooccur = build_cooccurrence_from_raw_data(raw_df)

    output_path = resolve_output_path(output_json)
    with open(output_path, "w", encoding="utf-8-sig") as f:
        json.dump(cooccur, f, ensure_ascii=False)

    print(f"[INFO] 已保存共现字典: {output_path}")
    return cooccur


# ============================================================
# 6. Feature computations
# ============================================================

def build_rule_vote_counter(ctx: Dict[str, Any]) -> Counter:
    counter = Counter()
    for ev in ctx.get("rule_expected_values", []) or []:
        ev = canonical_empty_text(ev)
        if ev not in {"", "empty"}:
            counter[ev] += 1
    for ev in ctx.get("expected_values_from_rules", []) or []:
        ev = canonical_empty_text(ev)
        if ev not in {"", "empty"}:
            counter[ev] += 2
    return counter


def compute_rule_features(candidate_value: Any, dirty_value: Any, ctx: Dict[str, Any]) -> Dict[str, Any]:
    vote_counter = build_rule_vote_counter(ctx)
    total_votes = sum(vote_counter.values())

    cand = canonical_empty_text(candidate_value)
    dirty = canonical_empty_text(dirty_value)

    cand_vote = vote_counter.get(cand, 0)
    dirty_vote = vote_counter.get(dirty, 0)
    cand_ratio = cand_vote / total_votes if total_votes > 0 else 0.0
    dirty_ratio = dirty_vote / total_votes if total_votes > 0 else 0.0

    alt_values = {
        canonical_empty_text(x.get("value", ""))
        for x in ctx.get("alternative_values_from_column", []) or []
        if isinstance(x, dict)
    }
    sim_row_values = [canonical_empty_text(v) for v in ctx.get("similar_row_target_values", []) or []]

    return {
        "rule_total_votes": total_votes,
        "candidate_rule_vote_count": cand_vote,
        "dirty_rule_vote_count": dirty_vote,
        "candidate_rule_support_ratio": cand_ratio,
        "dirty_rule_support_ratio": dirty_ratio,
        "rule_support_gain": cand_ratio - dirty_ratio,
        "candidate_matches_any_rule_expected": int(cand_vote > 0),
        "candidate_matches_any_strong_rule_expected": int(cand in {canonical_empty_text(v) for v in ctx.get("rule_expected_values", []) or []}),
        "candidate_matches_any_expected_values_from_rules": int(cand in {canonical_empty_text(v) for v in ctx.get("expected_values_from_rules", []) or []}),
        "candidate_in_alternative_values_from_column": int(cand in alt_values),
        "candidate_equals_neighbor_majority": int(
            not is_empty_like_value(ctx.get("neighbor_majority_value", "")) and
            norm_text(candidate_value) == norm_text(ctx.get("neighbor_majority_value", ""))
        ),
        "candidate_equals_suggested_correct_value": int(
            not is_empty_like_value(ctx.get("suggested_correct_value", "")) and
            norm_text(candidate_value) == norm_text(ctx.get("suggested_correct_value", ""))
        ),
        "candidate_matches_similar_row_target": int(cand in sim_row_values),
        "candidate_match_count_in_similar_rows": sum(1 for v in sim_row_values if v == cand),
        "candidate_match_ratio_in_similar_rows": (
            sum(1 for v in sim_row_values if v == cand) / len(sim_row_values)
            if sim_row_values else 0.0
        ),
        "strong_rule_count": safe_int(ctx.get("strong_rule_count", 0), 0),
        "candidate_generation_rule_count": safe_int(ctx.get("candidate_generation_rule_count", 0), 0),
        "fd_like_count": safe_int(ctx.get("fd_like_count", 0), 0),
        "context_rule_count": safe_int(ctx.get("context_rule_count", 0), 0),
        "global_rule_count": safe_int(ctx.get("global_rule_count", 0), 0),
        "rare_value_count": safe_int(ctx.get("rare_value_count", 0), 0),
        "typo_rule_count": safe_int(ctx.get("typo_rule_count", 0), 0),
        "pattern_rule_count": safe_int(ctx.get("pattern_rule_count", 0), 0),
        "schema_rule_count": safe_int(ctx.get("schema_rule_count", 0), 0),
        "adult_domain_rule_count": safe_int(ctx.get("adult_domain_rule_count", 0), 0),
        "adult_format_rule_count": safe_int(ctx.get("adult_format_rule_count", 0), 0),
        "adult_consistency_rule_count": safe_int(ctx.get("adult_consistency_rule_count", 0), 0),
        "soccer_domain_rule_count": safe_int(ctx.get("soccer_domain_rule_count", 0), 0),
        "soccer_format_rule_count": safe_int(ctx.get("soccer_format_rule_count", 0), 0),
        "soccer_numeric_rule_count": safe_int(ctx.get("soccer_numeric_rule_count", 0), 0),
        "soccer_consistency_rule_count": safe_int(ctx.get("soccer_consistency_rule_count", 0), 0),
    }


def extract_context_pairs_from_ctx(ctx: Dict[str, Any]) -> List[Tuple[str, str]]:
    pairs = []
    target_column = normalize_col_name(ctx.get("column", ""))
    row_values = ctx.get("row_values", {}) or {}

    for k, v in row_values.items():
        kk = normalize_col_name(k)
        if kk == target_column:
            continue
        vv = canonical_empty_text(v)
        if vv in {"", "empty"}:
            continue
        pairs.append((str(k), vv))

    return pairs[:MAX_CONTEXT_PAIRS]


def compute_bayes_features(candidate_value: Any, ctx: Dict[str, Any], external_cooccur: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    context_pairs = extract_context_pairs_from_ctx(ctx)

    base = {
        "prior_error_probability": safe_float(ctx.get("prior_error_probability", 0.0), 0.0),
        "posterior_error_probability": safe_float(ctx.get("posterior_error_probability", 0.0), 0.0),
        "bayes_rule_signal": safe_float(ctx.get("bayes_rule_signal", 0.0), 0.0),
        "bayes_rarity_signal": safe_float(ctx.get("bayes_rarity_signal", 0.0), 0.0),
        "bayes_pattern_signal": safe_float(ctx.get("bayes_pattern_signal", 0.0), 0.0),
        "bayes_neighbor_signal": safe_float(ctx.get("bayes_neighbor_signal", 0.0), 0.0),
        "bayes_adult_consistency_signal": safe_float(ctx.get("bayes_adult_consistency_signal", 0.0), 0.0),
        "bayes_soccer_consistency_signal": safe_float(ctx.get("bayes_soccer_consistency_signal", 0.0), 0.0),
    }

    if not context_pairs or external_cooccur is None:
        base.update({
            "context_pair_count_used": 0,
            "bayes_max_cond_prob": 0.0,
            "bayes_min_cond_prob": 0.0,
            "bayes_mean_cond_prob": 0.0,
            "bayes_prod_logprob": 0.0,
            "bayes_nonzero_pair_count": 0,
        })
        return base

    target_column = ctx.get("column", "")
    cand = canonical_empty_text(candidate_value)
    col_map = external_cooccur.get(target_column, {}) or external_cooccur.get(normalize_col_name(target_column), {})
    cand_map = col_map.get(cand, {})

    probs = []
    for ctx_col, ctx_val in context_pairs:
        p = safe_float(cand_map.get(f"{ctx_col}={ctx_val}", 0.0), 0.0)
        probs.append(p)

    nonzero = [p for p in probs if p > 0]
    base.update({
        "context_pair_count_used": len(probs),
        "bayes_max_cond_prob": max(probs) if probs else 0.0,
        "bayes_min_cond_prob": min(probs) if probs else 0.0,
        "bayes_mean_cond_prob": sum(probs) / len(probs) if probs else 0.0,
        "bayes_prod_logprob": sum(math.log(max(p, 1e-12)) for p in probs) if probs else 0.0,
        "bayes_nonzero_pair_count": len(nonzero),
    })
    return base


def compute_transform_features(candidate_value: Any, dirty_value: Any) -> Dict[str, Any]:
    sim = string_similarity(candidate_value, dirty_value)
    phon = phonetic_similarity(candidate_value, dirty_value)
    key_dist = avg_keyboard_distance_for_same_length(candidate_value, dirty_value)

    cand_s = canonical_empty_text(candidate_value)
    dirty_s = canonical_empty_text(dirty_value)

    num_c = try_parse_float(cand_s)
    num_d = try_parse_float(dirty_s)
    both_numeric = int(num_c is not None and num_d is not None)

    if both_numeric:
        abs_diff = abs(num_c - num_d)
        rel_diff = abs_diff / max(abs(num_d), 1e-9)
    else:
        abs_diff = None
        rel_diff = None

    return {
        "edit_similarity_to_dirty": sim,
        "edit_distance_proxy": 1.0 - sim,
        "phonetic_similarity_to_dirty": phon,
        "phonetic_distance_proxy": 1.0 - phon,
        "avg_keyboard_distance": key_dist,
        "length_diff": abs(len(cand_s) - len(dirty_s)),
        "both_numeric": both_numeric,
        "numeric_abs_diff": abs_diff,
        "numeric_rel_diff": rel_diff,
    }


def get_adult_bundle_value(ctx: Dict[str, Any], name: str) -> str:
    bundle = ctx.get("adult_profile_bundle", {}) or {}
    if isinstance(bundle, dict) and name in bundle:
        return canonical_empty_text(bundle.get(name))
    return get_row_val(ctx.get("row_values", {}) or {}, name)


def adult_profile_basics(ctx: Dict[str, Any]) -> Dict[str, Any]:
    row_values = ctx.get("row_values", {}) or {}

    age_raw = get_adult_bundle_value(ctx, "age")
    if age_raw == "empty":
        age_raw = get_row_val(row_values, "age")
    age_low, age_high = parse_numeric_range(age_raw)

    hours_raw = get_adult_bundle_value(ctx, "hoursperweek")
    if hours_raw == "empty":
        hours_raw = get_row_val(row_values, "hoursperweek", "hours-per-week")
    hours_low, hours_high = parse_numeric_range(hours_raw)

    education = canonical_domain_value("education", get_adult_bundle_value(ctx, "education"))
    marital = canonical_domain_value("maritalstatus", get_adult_bundle_value(ctx, "maritalstatus"))
    relationship = canonical_domain_value("relationship", get_adult_bundle_value(ctx, "relationship"))
    sex = canonical_domain_value("sex", get_adult_bundle_value(ctx, "sex"))
    income = canonical_domain_value("income", get_adult_bundle_value(ctx, "income"))

    return {
        "row_age_lower": age_low,
        "row_age_upper": age_high,
        "row_hours_lower": hours_low,
        "row_hours_upper": hours_high,
        "row_education": education,
        "row_education_order": education_order_value(education),
        "row_maritalstatus": marital,
        "row_relationship": relationship,
        "row_sex": sex,
        "row_income_canonical": income,
    }


def compute_column_domain_features(candidate_value: Any, dirty_value: Any, ctx: Dict[str, Any]) -> Dict[str, Any]:
    col = normalize_col_name(ctx.get("column", ""))
    cand = canonical_domain_value(col, candidate_value)
    dirty = canonical_domain_value(col, dirty_value)

    domain = ADULT_DOMAIN_VALUES.get(col, [])
    domain_norm = {norm_text(v): v for v in domain}

    candidate_in_domain = int(norm_text(cand) in domain_norm) if domain else 0
    dirty_in_domain = int(norm_text(dirty) in domain_norm) if domain else 0

    # Column top features
    top_counter = {canonical_empty_text(x.get("value", "")): x for x in ctx.get("column_top_values", []) or [] if isinstance(x, dict)}
    candidate_in_top_values = int(canonical_empty_text(candidate_value) in top_counter or cand in top_counter)
    candidate_top_ratio = safe_float(top_counter.get(canonical_empty_text(candidate_value), top_counter.get(cand, {})).get("ratio", 0.0), 0.0)

    return {
        "candidate_canonical_value": cand,
        "dirty_canonical_value": dirty,
        "candidate_in_adult_domain": candidate_in_domain,
        "dirty_in_adult_domain": dirty_in_domain,
        "candidate_domain_validity_gain": candidate_in_domain - dirty_in_domain,
        "candidate_equals_domain_canonical_dirty": int(norm_text(cand) == norm_text(dirty) and norm_text(candidate_value) != norm_text(dirty_value)),
        "candidate_in_column_top_values": candidate_in_top_values,
        "candidate_column_top_ratio": candidate_top_ratio,
    }


def relationship_consistency_score(rel: str, marital: str, sex: str, age_low: Optional[float], age_high: Optional[float]) -> float:
    score = 1.0
    rel = canonical_domain_value("relationship", rel)
    marital = canonical_domain_value("maritalstatus", marital)
    sex = canonical_domain_value("sex", sex)

    if rel == "Husband":
        if sex and sex != "empty" and sex != "Male":
            score -= 0.45
        if marital and marital != "empty" and marital not in MARRIED_STATUSES:
            score -= 0.35
    elif rel == "Wife":
        if sex and sex != "empty" and sex != "Female":
            score -= 0.45
        if marital and marital != "empty" and marital not in MARRIED_STATUSES:
            score -= 0.35
    elif rel == "Own-child":
        if marital in MARRIED_STATUSES:
            score -= 0.25

    if rel in SPOUSE_RELATIONSHIPS and age_high is not None and age_high <= 17:
        score -= 0.35

    return max(0.0, min(1.0, score))


def compute_adult_consistency_features(candidate_value: Any, dirty_value: Any, ctx: Dict[str, Any]) -> Dict[str, Any]:
    col = normalize_col_name(ctx.get("column", ""))
    prof = adult_profile_basics(ctx)

    marital = prof["row_maritalstatus"]
    sex = prof["row_sex"]
    relationship = prof["row_relationship"]
    age_low = prof["row_age_lower"]
    age_high = prof["row_age_upper"]
    education = prof["row_education"]
    edu_order = prof["row_education_order"]

    cand = canonical_domain_value(col, candidate_value)
    dirty = canonical_domain_value(col, dirty_value)

    # If target candidate changes one field, recompute with candidate plugged in.
    cand_rel = cand if col == "relationship" else relationship
    dirty_rel = dirty if col == "relationship" else relationship

    cand_sex = cand if col == "sex" else sex
    dirty_sex = dirty if col == "sex" else sex

    cand_marital = cand if col == "maritalstatus" else marital
    dirty_marital = dirty if col == "maritalstatus" else marital

    cand_rel_score = relationship_consistency_score(cand_rel, cand_marital, cand_sex, age_low, age_high)
    dirty_rel_score = relationship_consistency_score(dirty_rel, dirty_marital, dirty_sex, age_low, age_high)

    cand_edu_order = education_order_value(cand) if col == "education" else edu_order
    dirty_edu_order = education_order_value(dirty) if col == "education" else edu_order

    cand_education_age_conflict = 0
    dirty_education_age_conflict = 0
    if age_high is not None:
        if cand_edu_order is not None and age_high <= 17 and cand_edu_order >= EDUCATION_ORDER["Bachelors"]:
            cand_education_age_conflict = 1
        if dirty_edu_order is not None and age_high <= 17 and dirty_edu_order >= EDUCATION_ORDER["Bachelors"]:
            dirty_education_age_conflict = 1
        if cand_edu_order is not None and age_high <= 21 and cand_edu_order >= EDUCATION_ORDER["Masters"]:
            cand_education_age_conflict = 1
        if dirty_edu_order is not None and age_high <= 21 and dirty_edu_order >= EDUCATION_ORDER["Masters"]:
            dirty_education_age_conflict = 1

    return {
        "adult_consistency_score": safe_float(ctx.get("adult_consistency_score", 0.0), 0.0),
        "candidate_relationship_consistency_score": cand_rel_score,
        "dirty_relationship_consistency_score": dirty_rel_score,
        "relationship_consistency_gain": cand_rel_score - dirty_rel_score,
        "candidate_education_age_conflict": cand_education_age_conflict,
        "dirty_education_age_conflict": dirty_education_age_conflict,
        "education_age_conflict_reduction": dirty_education_age_conflict - cand_education_age_conflict,
        "candidate_is_spouse_relationship": int(cand_rel in SPOUSE_RELATIONSHIPS),
        "dirty_is_spouse_relationship": int(dirty_rel in SPOUSE_RELATIONSHIPS),
        "candidate_spouse_gender_match": int(
            (cand_rel == "Husband" and cand_sex == "Male") or
            (cand_rel == "Wife" and cand_sex == "Female") or
            cand_rel not in SPOUSE_RELATIONSHIPS
        ),
        "dirty_spouse_gender_match": int(
            (dirty_rel == "Husband" and dirty_sex == "Male") or
            (dirty_rel == "Wife" and dirty_sex == "Female") or
            dirty_rel not in SPOUSE_RELATIONSHIPS
        ),
        "candidate_spouse_marital_match": int(cand_rel not in SPOUSE_RELATIONSHIPS or cand_marital in MARRIED_STATUSES),
        "dirty_spouse_marital_match": int(dirty_rel not in SPOUSE_RELATIONSHIPS or dirty_marital in MARRIED_STATUSES),
        "target_is_primary_column": safe_int(ctx.get("target_is_primary_column", 0), 0),
        "target_is_context_only_column": safe_int(ctx.get("target_is_context_only_column", 0), 0),
        "is_relationship_spouse_value": safe_int(ctx.get("is_relationship_spouse_value", 0), 0),
        "relationship_marital_conflict": safe_int(ctx.get("relationship_marital_conflict", 0), 0),
        "relationship_age_conflict": safe_int(ctx.get("relationship_age_conflict", 0), 0),
        "relationship_sex_conflict": safe_int(ctx.get("relationship_sex_conflict", 0), 0),
        "sex_value_invalid": safe_int(ctx.get("sex_value_invalid", 0), 0),
        "education_age_extreme_conflict": safe_int(ctx.get("education_age_extreme_conflict", 0), 0),
        "has_relationship_v2_rule": safe_int(ctx.get("has_relationship_v2_rule", 0), 0),
        "has_education_v2_rule": safe_int(ctx.get("has_education_v2_rule", 0), 0),
        "row_age_lower": age_low,
        "row_age_upper": age_high,
        "row_hours_lower": prof["row_hours_lower"],
        "row_hours_upper": prof["row_hours_upper"],
        "row_education_order": prof["row_education_order"],
        "row_income_canonical": prof["row_income_canonical"],
    }


def top_distribution_features(dist: Dict[str, Any], candidate_value: Any, dirty_value: Any) -> Dict[str, Any]:
    if not isinstance(dist, dict):
        return {
            "profile_group_size": 0,
            "candidate_profile_count": 0,
            "dirty_profile_count": 0,
            "candidate_profile_ratio": 0.0,
            "dirty_profile_ratio": 0.0,
            "candidate_profile_rank": 0,
            "dirty_profile_rank": 0,
            "profile_ratio_gain": 0.0,
        }

    tops = dist.get("top_values", []) or []
    group_size = safe_int(dist.get("group_size", 0), 0)

    cand = canonical_empty_text(candidate_value)
    dirty = canonical_empty_text(dirty_value)

    cand_count = 0
    dirty_count = 0
    cand_rank = 0
    dirty_rank = 0

    for i, item in enumerate(tops, start=1):
        if not isinstance(item, dict):
            continue
        val = canonical_empty_text(item.get("value", ""))
        cnt = safe_int(item.get("count", 0), 0)
        if norm_text(val) == norm_text(cand):
            cand_count = cnt
            cand_rank = i
        if norm_text(val) == norm_text(dirty):
            dirty_count = cnt
            dirty_rank = i

    cand_ratio = cand_count / max(group_size, 1)
    dirty_ratio = dirty_count / max(group_size, 1)

    return {
        "profile_group_size": group_size,
        "candidate_profile_count": cand_count,
        "dirty_profile_count": dirty_count,
        "candidate_profile_ratio": cand_ratio,
        "dirty_profile_ratio": dirty_ratio,
        "candidate_profile_rank": cand_rank,
        "dirty_profile_rank": dirty_rank,
        "profile_ratio_gain": cand_ratio - dirty_ratio,
    }


def compute_profile_distribution_features(candidate_value: Any, dirty_value: Any, ctx: Dict[str, Any]) -> Dict[str, Any]:
    col = normalize_col_name(ctx.get("column", ""))

    same_profile_distribution = ctx.get("same_profile_distribution", {}) or {}
    work_profile_distribution = ctx.get("work_profile_distribution", {}) or {}

    same_dist = same_profile_distribution.get(col) or same_profile_distribution.get(ctx.get("column", "")) or {}
    work_dist = work_profile_distribution.get(col) or work_profile_distribution.get(ctx.get("column", "")) or {}

    same_feats = top_distribution_features(same_dist, candidate_value, dirty_value)
    work_feats = top_distribution_features(work_dist, candidate_value, dirty_value)

    return {
        "same_profile_group_size": same_feats["profile_group_size"],
        "candidate_same_profile_count": same_feats["candidate_profile_count"],
        "dirty_same_profile_count": same_feats["dirty_profile_count"],
        "candidate_same_profile_ratio": same_feats["candidate_profile_ratio"],
        "dirty_same_profile_ratio": same_feats["dirty_profile_ratio"],
        "candidate_same_profile_rank": same_feats["candidate_profile_rank"],
        "dirty_same_profile_rank": same_feats["dirty_profile_rank"],
        "same_profile_ratio_gain": same_feats["profile_ratio_gain"],
        "work_profile_group_size": work_feats["profile_group_size"],
        "candidate_work_profile_count": work_feats["candidate_profile_count"],
        "dirty_work_profile_count": work_feats["dirty_profile_count"],
        "candidate_work_profile_ratio": work_feats["candidate_profile_ratio"],
        "dirty_work_profile_ratio": work_feats["dirty_profile_ratio"],
        "candidate_work_profile_rank": work_feats["candidate_profile_rank"],
        "dirty_work_profile_rank": work_feats["dirty_profile_rank"],
        "work_profile_ratio_gain": work_feats["profile_ratio_gain"],
    }


def parse_sources(row: pd.Series) -> set:
    text = ""
    for c in ["candidate_sources", "candidate_source", "candidate_source_text"]:
        if c in row.index and str(row.get(c, "")).strip():
            text += "|" + str(row.get(c, ""))
    return {s.strip() for s in text.split("|") if s.strip()}


def compute_source_features(row: pd.Series) -> Dict[str, Any]:
    sources = parse_sources(row)
    lower_sources = {s.lower() for s in sources}

    def contains(substr: str) -> int:
        return int(any(substr in s for s in lower_sources))

    return {
        "candidate_source_text": "|".join(sorted(sources)),
        "source_count": len(sources) if sources else safe_int(row.get("source_count", 0), 0),
        "is_multi_source_candidate": int((len(sources) if sources else safe_int(row.get("source_count", 0), 0)) >= 2),
        "source_score": safe_float(row.get("source_score", 0.0), 0.0),
        "candidate_rank": safe_int(row.get("candidate_rank", 9999), 9999),
        "is_from_rule": safe_int(row.get("is_from_rule", 0), 0),
        "is_from_neighbor": safe_int(row.get("is_from_neighbor", 0), 0),
        "is_from_column_top": safe_int(row.get("is_from_column_top", 0), 0),
        "is_from_similarity": safe_int(row.get("is_from_similarity", 0), 0),
        "is_from_hint": safe_int(row.get("is_from_hint", 0), 0),
        "is_from_dictionary": safe_int(row.get("is_from_dictionary", 0), 0),
        "is_from_pattern_restore": safe_int(row.get("is_from_pattern_restore", 0), 0),
        "is_from_adult_profile": safe_int(row.get("is_from_adult_profile", 0), 0),
        "is_from_adult_direct_rule": safe_int(row.get("is_from_adult_direct_rule", 0), 0),
        "is_from_soccer_profile": safe_int(row.get("is_from_soccer_profile", 0), 0),
        "is_from_soccer_direct_rule": safe_int(row.get("is_from_soccer_direct_rule", 0), 0),
        "is_from_numeric_rule": safe_int(row.get("is_from_numeric_rule", 0), 0),
        "is_from_domain": safe_int(row.get("is_from_domain", 0), 0),
        "is_from_profile_key_majority": safe_int(row.get("is_from_profile_key_majority", 0), 0),
        "is_from_entity_pair_majority": safe_int(row.get("is_from_entity_pair_majority", 0), 0),
        "is_from_team_season_position": safe_int(row.get("is_from_team_season_position", 0), 0),
        "profile_key_group_size": safe_int(row.get("profile_key_group_size", 0), 0),
        "profile_key_top_count": safe_int(row.get("profile_key_top_count", 0), 0),
        "profile_key_majority_ratio": safe_float(row.get("profile_key_majority_ratio", 0.0), 0.0),
        "entity_pair_group_size": safe_int(row.get("entity_pair_group_size", 0), 0),
        "entity_pair_top_count": safe_int(row.get("entity_pair_top_count", 0), 0),
        "entity_pair_majority_ratio": safe_float(row.get("entity_pair_majority_ratio", 0.0), 0.0),
        "team_season_position_group_size": safe_int(row.get("team_season_position_group_size", 0), 0),
        "team_season_position_top_count": safe_int(row.get("team_season_position_top_count", 0), 0),
        "team_season_position_ratio": safe_float(row.get("team_season_position_ratio", 0.0), 0.0),
        "contains_rule_expected_source": int("rule_expected_value" in lower_sources or "expected_values_from_rules" in lower_sources or "rule_reason_parse" in lower_sources),
        "contains_same_profile_source": contains("same_profile"),
        "contains_work_profile_source": contains("work_profile"),
        "contains_adult_profile_source": contains("adult_profile"),
        "contains_soccer_profile_source_general": contains("soccer_profile") or contains("team_season_distribution") or contains("team_distribution") or contains("stadium_city_distribution"),
        "contains_profile_key_majority_source": contains("profile_key_majority"),
        "contains_entity_pair_majority_source": contains("entity_pair_majority"),
        "contains_team_season_position_source": contains("team_season_position"),
        "contains_name_source": contains("name"),
        "contains_surname_source": contains("surname"),
        "contains_birthplace_source": contains("birthplace"),
        "contains_relationship_source": contains("relationship"),
        "contains_sex_source": contains("sex_") or int(any(s.startswith("sex") for s in lower_sources)),
        "contains_maritalstatus_source": contains("maritalstatus") or contains("marital"),
        "contains_education_source": contains("education"),
        "contains_age_source": contains("age"),
        "contains_hours_source": contains("hours"),
        "contains_income_source": contains("income"),
        "contains_domain_source": contains("domain_"),
        "contains_neighbor_source": int("neighbor_majority" in lower_sources or "similar_row_target" in lower_sources),
        "contains_column_top_source": int("column_top_value" in lower_sources or "alternative_values_from_column" in lower_sources),
        "contains_hint_source": int("suggested_correct_value" in lower_sources),
        "is_weak_frequency_only_candidate": int(
            (contains("column_top") or "global_column_dictionary" in lower_sources)
            and not (
                contains("rule") or contains("profile") or contains("entity_pair_majority")
                or contains("profile_key_majority") or contains("team_season_position")
                or contains("neighbor") or contains("similar")
                or contains("relationship") or contains("sex_")
                or contains("marital") or contains("education") or contains("domain_")
            )
        ),
    }




def canonical_soccer_position(x: Any) -> str:
    s = canonical_empty_text(x)
    if s in {"", "empty"}:
        return "empty"
    key = re.sub(r"[^a-z]+", "", s.lower())
    return POSITION_ALIASES.get(key, s)


def parse_year_int(x: Any) -> Optional[int]:
    s = canonical_empty_text(x)
    m = re.search(r"(18\d{2}|19\d{2}|20\d{2}|21\d{2})", s)
    if not m:
        v = try_parse_float(s)
        if v is None:
            return None
        y = int(round(v))
    else:
        y = int(m.group(1))
    if 1800 <= y <= 2200:
        return y
    return None


def soccer_age_at_season(birthyear: Any, season: Any) -> Optional[int]:
    b = parse_year_int(birthyear)
    s0 = parse_year_int(season)
    if b is None or s0 is None:
        return None
    age = s0 - b
    if -5 <= age <= 100:
        return age
    return None


def get_soccer_row_value(ctx: Dict[str, Any], *names: str) -> str:
    return get_row_val(ctx.get("row_values", {}) or {}, *names)


def soccer_profile_basics(ctx: Dict[str, Any]) -> Dict[str, Any]:
    row_values = ctx.get("row_values", {}) or {}
    bundle = ctx.get("soccer_profile_bundle", {}) or {}
    if not isinstance(bundle, dict):
        bundle = {}
    birthyear = get_row_val(row_values, "birthyear")
    season = get_row_val(row_values, "season")
    position = canonical_soccer_position(get_row_val(row_values, "position"))
    team = get_row_val(row_values, "team")
    city = get_row_val(row_values, "city")
    stadium = get_row_val(row_values, "stadium")
    manager = get_row_val(row_values, "manager")

    # Rich Soccer contexts may already provide a profile bundle.
    if birthyear == "empty":
        birthyear = canonical_empty_text(bundle.get("birthyear", bundle.get("birthyear_int", birthyear)))
    if season == "empty":
        season = canonical_empty_text(bundle.get("season", bundle.get("season_int", season)))
    if position == "empty":
        position = canonical_soccer_position(bundle.get("position", bundle.get("position_canonical", position)))
    if team == "empty":
        team = canonical_empty_text(bundle.get("team", team))
    if city == "empty":
        city = canonical_empty_text(bundle.get("city", city))
    if stadium == "empty":
        stadium = canonical_empty_text(bundle.get("stadium", stadium))
    if manager == "empty":
        manager = canonical_empty_text(bundle.get("manager", manager))
    age = soccer_age_at_season(birthyear, season)
    return {
        "row_birthyear_int": parse_year_int(birthyear),
        "row_season_int": parse_year_int(season),
        "row_age_at_season": age,
        "row_position_canonical": position,
        "row_team": team,
        "row_city": city,
        "row_stadium": stadium,
        "row_manager": manager,
    }


def compute_soccer_domain_features(candidate_value: Any, dirty_value: Any, ctx: Dict[str, Any]) -> Dict[str, Any]:
    col = normalize_col_name(ctx.get("column", ""))
    cand_raw = canonical_empty_text(candidate_value)
    dirty_raw = canonical_empty_text(dirty_value)
    cand_pos = canonical_soccer_position(candidate_value) if col == "position" else cand_raw
    dirty_pos = canonical_soccer_position(dirty_value) if col == "position" else dirty_raw

    cand_year = parse_year_int(candidate_value) if col in {"birthyear", "season"} else None
    dirty_year = parse_year_int(dirty_value) if col in {"birthyear", "season"} else None

    candidate_position_valid = int(col == "position" and cand_pos in SOCCER_DOMAIN_VALUES["position"])
    dirty_position_valid = int(col == "position" and dirty_pos in SOCCER_DOMAIN_VALUES["position"])
    candidate_year_valid = 0
    dirty_year_valid = 0
    if col == "birthyear":
        candidate_year_valid = int(cand_year is not None and 1950 <= cand_year <= 2010)
        dirty_year_valid = int(dirty_year is not None and 1950 <= dirty_year <= 2010)
    elif col == "season":
        candidate_year_valid = int(cand_year is not None and 1990 <= cand_year <= 2035)
        dirty_year_valid = int(dirty_year is not None and 1990 <= dirty_year <= 2035)

    top_counter = {canonical_empty_text(x.get("value", "")): x for x in ctx.get("column_top_values", []) or [] if isinstance(x, dict)}
    candidate_in_top_values = int(cand_raw in top_counter or cand_pos in top_counter)
    candidate_top_ratio = safe_float(top_counter.get(cand_raw, top_counter.get(cand_pos, {})).get("ratio", 0.0), 0.0)

    return {
        "candidate_soccer_canonical_value": cand_pos if col == "position" else (cand_year if cand_year is not None else cand_raw),
        "dirty_soccer_canonical_value": dirty_pos if col == "position" else (dirty_year if dirty_year is not None else dirty_raw),
        "candidate_position_valid": candidate_position_valid,
        "dirty_position_valid": dirty_position_valid,
        "candidate_position_validity_gain": candidate_position_valid - dirty_position_valid,
        "candidate_year_valid": candidate_year_valid,
        "dirty_year_valid": dirty_year_valid,
        "candidate_year_validity_gain": candidate_year_valid - dirty_year_valid,
        "candidate_position_needs_canonicalization": int(col == "position" and cand_pos != cand_raw and cand_pos in SOCCER_DOMAIN_VALUES["position"]),
        "dirty_position_needs_canonicalization": int(col == "position" and dirty_pos != dirty_raw and dirty_pos in SOCCER_DOMAIN_VALUES["position"]),
        "candidate_in_column_top_values_soccer": candidate_in_top_values,
        "candidate_column_top_ratio_soccer": candidate_top_ratio,
    }


def compute_soccer_consistency_features(candidate_value: Any, dirty_value: Any, ctx: Dict[str, Any]) -> Dict[str, Any]:
    col = normalize_col_name(ctx.get("column", ""))
    basics = soccer_profile_basics(ctx)
    row_birthyear = basics.get("row_birthyear_int")
    row_season = basics.get("row_season_int")
    row_age = basics.get("row_age_at_season")

    cand_birthyear = row_birthyear
    cand_season = row_season
    dirty_birthyear = row_birthyear
    dirty_season = row_season
    if col == "birthyear":
        cand_birthyear = parse_year_int(candidate_value)
        dirty_birthyear = parse_year_int(dirty_value)
    elif col == "season":
        cand_season = parse_year_int(candidate_value)
        dirty_season = parse_year_int(dirty_value)

    cand_age = soccer_age_at_season(cand_birthyear, cand_season)
    dirty_age = soccer_age_at_season(dirty_birthyear, dirty_season)

    def age_score(age: Optional[int]) -> float:
        if age is None:
            return 0.0
        if 16 <= age <= 45:
            return 1.0
        if 12 <= age <= 55:
            return 0.5
        return -1.0

    cand_age_score = age_score(cand_age)
    dirty_age_score = age_score(dirty_age)
    cand_pos = canonical_soccer_position(candidate_value) if col == "position" else basics.get("row_position_canonical", "")
    dirty_pos = canonical_soccer_position(dirty_value) if col == "position" else basics.get("row_position_canonical", "")
    cand_pos_score = 1.0 if cand_pos in SOCCER_DOMAIN_VALUES["position"] else (-1.0 if col == "position" else 0.0)
    dirty_pos_score = 1.0 if dirty_pos in SOCCER_DOMAIN_VALUES["position"] else (-1.0 if col == "position" else 0.0)

    evidence_flags = ctx.get("soccer_evidence_flags", []) or ctx.get("adult_evidence_flags", []) or []
    if isinstance(evidence_flags, str):
        evidence_flags = [x for x in evidence_flags.split("|") if x]
    evidence_text_norm = "|".join(norm_text(x) for x in evidence_flags)

    return {
        **basics,
        "candidate_age_at_season": cand_age,
        "dirty_age_at_season": dirty_age,
        "candidate_age_validity_score": cand_age_score,
        "dirty_age_validity_score": dirty_age_score,
        "candidate_age_validity_gain": cand_age_score - dirty_age_score,
        "candidate_position_consistency_score": cand_pos_score,
        "dirty_position_consistency_score": dirty_pos_score,
        "candidate_position_consistency_gain": cand_pos_score - dirty_pos_score,
        "soccer_birthyear_season_age_conflict_reduction": int(dirty_age_score < 0 and cand_age_score >= 0),
        "soccer_position_invalid_reduction": int(dirty_pos_score < 0 and cand_pos_score > dirty_pos_score),
        "soccer_consistency_score": safe_float(ctx.get("soccer_consistency_score", 0.0), 0.0),
        "soccer_profile_inconsistency_count": safe_int(ctx.get("soccer_profile_inconsistency_count", 0), 0),
        "has_invalid_position_flag": int("invalid_position" in evidence_text_norm or "position_value_invalid" in evidence_text_norm),
        "has_age_conflict_flag": int("age_conflict" in evidence_text_norm or "birthyear_season_age" in evidence_text_norm),
        "has_team_context_conflict_flag": int("team_context" in evidence_text_norm),
    }


def compute_soccer_source_features(row: pd.Series) -> Dict[str, Any]:
    sources = parse_sources(row)
    lower_sources = {s.lower() for s in sources}

    def contains(substr: str) -> int:
        return int(any(substr in s for s in lower_sources))

    return {
        "is_from_soccer_profile": safe_int(row.get("is_from_soccer_profile", 0), 0),
        "is_from_soccer_direct_rule": safe_int(row.get("is_from_soccer_direct_rule", 0), 0),
        "is_from_soccer_domain": safe_int(row.get("is_from_soccer_domain", row.get("is_from_domain", 0)), 0),
        "is_from_soccer_consensus": safe_int(row.get("is_from_soccer_consensus", 0), 0),
        "is_from_profile_key_majority": safe_int(row.get("is_from_profile_key_majority", 0), 0),
        "is_from_entity_pair_majority": safe_int(row.get("is_from_entity_pair_majority", 0), 0),
        "is_from_team_season_position": safe_int(row.get("is_from_team_season_position", 0), 0),
        "contains_soccer_profile_source": contains("soccer_profile") or contains("team_distribution") or contains("team_season_distribution") or contains("stadium_city_distribution"),
        "contains_soccer_direct_rule_source": contains("soccer_direct") or contains("direct_rule"),
        "contains_profile_key_majority_source": contains("profile_key_majority"),
        "contains_entity_pair_majority_source": contains("entity_pair_majority"),
        "contains_team_season_position_source": contains("team_season_position"),
        "contains_position_source": contains("position"),
        "contains_birthyear_source": contains("birthyear"),
        "contains_season_source": contains("season"),
        "contains_team_source": contains("team"),
        "contains_city_source": contains("city"),
        "contains_stadium_source": contains("stadium"),
        "contains_manager_source": contains("manager"),
        "contains_name_source": contains("name"),
        "contains_surname_source": contains("surname"),
        "contains_birthplace_source": contains("birthplace"),
    }



def compute_profile_majority_candidate_features(row: pd.Series, candidate_value: Any, dirty_value: Any, ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Features for the new candidates produced by revise_candidate_soccer_profile_majority.py.

    These fields are critical for pushing name/surname/birthplace oracle coverage into
    usable ranking signals. Without them, the LTR model sees the new candidates as
    generic dictionary/profile candidates and may not rank them high.
    """
    col = normalize_col_name(ctx.get("column", row.get("column", "")))
    cand = canonical_empty_text(candidate_value)
    dirty = canonical_empty_text(dirty_value)

    profile_group_size = safe_int(row.get("profile_key_group_size", 0), 0)
    profile_top_count = safe_int(row.get("profile_key_top_count", 0), 0)
    profile_ratio = safe_float(row.get("profile_key_majority_ratio", 0.0), 0.0)

    pair_group_size = safe_int(row.get("entity_pair_group_size", 0), 0)
    pair_top_count = safe_int(row.get("entity_pair_top_count", 0), 0)
    pair_ratio = safe_float(row.get("entity_pair_majority_ratio", 0.0), 0.0)

    tsp_group_size = safe_int(row.get("team_season_position_group_size", 0), 0)
    tsp_top_count = safe_int(row.get("team_season_position_top_count", 0), 0)
    tsp_ratio = safe_float(row.get("team_season_position_ratio", 0.0), 0.0)

    is_profile = safe_int(row.get("is_from_profile_key_majority", 0), 0)
    is_pair = safe_int(row.get("is_from_entity_pair_majority", 0), 0)
    is_tsp = safe_int(row.get("is_from_team_season_position", 0), 0)

    max_ratio = max(profile_ratio, pair_ratio, tsp_ratio)
    max_group_size = max(profile_group_size, pair_group_size, tsp_group_size)
    max_top_count = max(profile_top_count, pair_top_count, tsp_top_count)

    is_entity_col = int(col in {"name", "surname", "birthplace"})
    is_text_entity_col = int(col in {"name", "surname", "birthplace", "team", "city", "stadium", "manager"})
    is_position_col = int(col == "position")
    is_year_col = int(col in {"birthyear", "season"})

    dirty_empty = int(is_empty_like_value(dirty))
    cand_empty = int(is_empty_like_value(cand))

    pair_or_profile = int(is_profile or is_pair or is_tsp)
    high_conf_majority = int(pair_or_profile and max_group_size >= 2 and max_top_count >= 2 and max_ratio >= 0.70)
    very_high_conf_majority = int(pair_or_profile and max_group_size >= 3 and max_top_count >= 2 and max_ratio >= 0.80)

    # Conservative "safe fill" signal: useful for infer gate.
    entity_safe_fill = int(
        is_entity_col
        and dirty_empty == 1
        and cand_empty == 0
        and (
            (is_pair and pair_group_size >= 2 and pair_top_count >= 2 and pair_ratio >= 0.75)
            or (is_profile and profile_group_size >= 2 and profile_top_count >= 2 and profile_ratio >= 0.70)
        )
    )

    position_profile_strong = int(
        is_position_col
        and is_tsp
        and tsp_group_size >= 2
        and tsp_top_count >= 2
        and tsp_ratio >= 0.50
    )

    # Penalize pure weak-global candidates for entity columns.
    sources = parse_sources(row)
    lower_sources = {s.lower() for s in sources}
    has_global = int(any(("global_column_dictionary" in s or "column_top_value" in s or "domain_position_core_value" in s) for s in lower_sources))
    has_strong_profile = int(is_profile or is_pair or is_tsp or any(("neighbor" in s or "similar_row" in s or "expected_values_from_rules" in s or "rule_expected" in s) for s in lower_sources))
    pure_weak_global_for_entity = int(is_text_entity_col and has_global and not has_strong_profile)

    return {
        "profile_key_group_size": profile_group_size,
        "profile_key_top_count": profile_top_count,
        "profile_key_majority_ratio": profile_ratio,
        "entity_pair_group_size": pair_group_size,
        "entity_pair_top_count": pair_top_count,
        "entity_pair_majority_ratio": pair_ratio,
        "team_season_position_group_size": tsp_group_size,
        "team_season_position_top_count": tsp_top_count,
        "team_season_position_ratio": tsp_ratio,
        "max_profile_majority_ratio": max_ratio,
        "max_profile_group_size": max_group_size,
        "max_profile_top_count": max_top_count,
        "candidate_has_profile_majority_signal": pair_or_profile,
        "candidate_has_high_conf_profile_majority": high_conf_majority,
        "candidate_has_very_high_conf_profile_majority": very_high_conf_majority,
        "candidate_entity_safe_fill": entity_safe_fill,
        "candidate_position_profile_strong": position_profile_strong,
        "candidate_is_name_entity_majority": int(col == "name" and (is_profile or is_pair)),
        "candidate_is_surname_entity_majority": int(col == "surname" and (is_profile or is_pair)),
        "candidate_is_birthplace_entity_majority": int(col == "birthplace" and (is_profile or is_pair)),
        "candidate_is_position_profile_majority": int(col == "position" and (is_profile or is_tsp)),
        "candidate_is_profile_or_pair_majority": int(is_profile or is_pair),
        "candidate_is_pure_weak_global_for_entity": pure_weak_global_for_entity,
        "candidate_entity_majority_ratio_gain_over_dirty": max_ratio if dirty_empty else max_ratio * 0.5,
        "is_entity_column": is_entity_col,
        "is_text_entity_column": is_text_entity_col,
        "is_position_column": is_position_col,
        "is_year_column": is_year_col,
    }


# ============================================================
# 7. Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--candidates_file", default=DEFAULT_CANDIDATES_FILE)
    p.add_argument("--context_file", default=DEFAULT_CONTEXT_FILE)
    p.add_argument("--fallback_context_flat_file", default=DEFAULT_FALLBACK_CONTEXT_FLAT_FILE)
    p.add_argument("--raw_data_file", default=DEFAULT_RAW_DATA_FILE)
    p.add_argument("--cooccurrence_dict_file", default=DEFAULT_COOCCURRENCE_DICT_FILE)
    p.add_argument("--output_features_csv", default=DEFAULT_OUTPUT_FEATURES_CSV)
    p.add_argument("--output_built_cooccur_json", default=DEFAULT_OUTPUT_BUILT_COOCCUR_JSON)
    return p.parse_args()


def main():
    args = parse_args()

    if not path_exists_nonempty(args.candidates_file):
        raise FileNotFoundError(f"候选展开文件不存在: {args.candidates_file}")

    candidates_df = pd.read_csv(args.candidates_file, encoding="utf-8-sig")
    required = {"row_id", "column", "dirty_value", "candidate_value"}
    missing = required - set(candidates_df.columns)
    if missing:
        raise ValueError(f"候选文件缺少必要字段: {missing}")

    candidates_df["row_id"] = pd.to_numeric(candidates_df["row_id"], errors="raise").astype(int)

    context_records = []
    if path_exists_nonempty(args.context_file):
        rich = load_context_records(args.context_file)
        context_records.extend(rich)
        print(f"[INFO] rich context records: {len(rich)}")
    else:
        print(f"[WARN] context_file 不存在，后续使用 fallback/dirty context: {args.context_file}")

    context_map = build_context_map(context_records)
    print(f"[INFO] indexed rich contexts: {len(context_map)}")

    if args.fallback_context_flat_file and path_exists_nonempty(args.fallback_context_flat_file):
        flat = load_context_records(args.fallback_context_flat_file)
        merge_flat_into_context_map(context_map, flat)
        print(f"[INFO] fallback flat records: {len(flat)}")
        print(f"[INFO] indexed contexts after flat merge: {len(context_map)}")
    else:
        print(f"[INFO] no fallback flat context loaded.")

    dirty_df = load_dirty_table(args.raw_data_file)
    if dirty_df is not None:
        print(f"[INFO] dirty table loaded: shape={dirty_df.shape}")
    else:
        print("[WARN] dirty table not loaded; fallback context and co-occurrence may be weak.")

    cooccur = load_or_build_cooccurrence_dict(
        args.cooccurrence_dict_file,
        args.raw_data_file,
        args.output_built_cooccur_json,
    )

    feature_rows = []
    missing_context_count = 0

    for _, row in candidates_df.iterrows():
        row_id = int(row["row_id"])
        column = str(row["column"])
        dirty_value = canonical_empty_text(row["dirty_value"])
        candidate_value = canonical_empty_text(row["candidate_value"])

        ctx = context_map.get((row_id, column))
        if ctx is None:
            missing_context_count += 1
            ctx = build_fallback_context(row, dirty_df)

        rule_feats = compute_rule_features(candidate_value, dirty_value, ctx)
        bayes_feats = compute_bayes_features(candidate_value, ctx, cooccur)
        trans_feats = compute_transform_features(candidate_value, dirty_value)
        domain_feats = compute_column_domain_features(candidate_value, dirty_value, ctx)
        soccer_domain_feats = compute_soccer_domain_features(candidate_value, dirty_value, ctx)
        adult_consistency_feats = compute_adult_consistency_features(candidate_value, dirty_value, ctx)
        soccer_consistency_feats = compute_soccer_consistency_features(candidate_value, dirty_value, ctx)
        profile_feats = compute_profile_distribution_features(candidate_value, dirty_value, ctx)
        profile_majority_feats = compute_profile_majority_candidate_features(row, candidate_value, dirty_value, ctx)
        source_feats = compute_source_features(row)
        soccer_source_feats = compute_soccer_source_features(row)

        out = row.to_dict()

        evidence_flags = ctx.get("soccer_evidence_flags", []) or ctx.get("adult_evidence_flags", []) or []
        if isinstance(evidence_flags, str):
            evidence_flags = [x for x in evidence_flags.split("|") if x]
        evidence_text = "|".join([str(x) for x in evidence_flags if str(x).strip()])

        out.update({
            "semantic_type": ctx.get("semantic_type", ""),
            "detected_type": ctx.get("detected_type", ""),
            "violation_count": ctx.get("violation_count", 0),
            "conflict_score": ctx.get("conflict_score", 0.0),
            "main_rule_type": ctx.get("main_rule_type", ""),
            "main_usage_role": ctx.get("main_usage_role", ""),
            "has_row_context": int(bool(ctx.get("row_values", {}))),
            "neighbor_majority_value": ctx.get("neighbor_majority_value", ""),
            "neighbor_majority_ratio": safe_float(ctx.get("neighbor_majority_ratio", 0.0), 0.0),
            "suggested_correct_value": ctx.get("suggested_correct_value", ""),
            "soccer_evidence_flags": evidence_text,
            "soccer_evidence_flag_count": len(evidence_flags),
            "adult_evidence_flags": evidence_text,
            "adult_evidence_flag_count": len(evidence_flags),
            "has_spouse_relationship_conflict_flag": int(any("spouse" in norm_text(x) for x in evidence_flags)),
            "has_husband_not_male_flag": int(any("husband" in norm_text(x) and "not_male" in norm_text(x) for x in evidence_flags)),
            "has_wife_not_female_flag": int(any("wife" in norm_text(x) and "not_female" in norm_text(x) for x in evidence_flags)),
            "has_young_spouse_flag": int(any("young" in norm_text(x) or "very_young" in norm_text(x) for x in evidence_flags)),
            "has_education_age_flag": int(any("education" in norm_text(x) and "age" in norm_text(x) for x in evidence_flags)),
        })

        out.update(source_feats)
        out.update(soccer_source_feats)
        out.update(rule_feats)
        out.update(bayes_feats)
        out.update(trans_feats)
        out.update(domain_feats)
        out.update(soccer_domain_feats)
        out.update(adult_consistency_feats)
        out.update(soccer_consistency_feats)
        out.update(profile_feats)
        out.update(profile_majority_feats)

        feature_rows.append(out)

    features_df = pd.DataFrame(feature_rows)
    output_path = resolve_output_path(args.output_features_csv)
    features_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\n========== Soccer Feature Builder Summary ==========")
    print(f"[INFO] candidate rows: {len(candidates_df)}")
    print(f"[INFO] feature rows: {len(features_df)}")
    print(f"[INFO] fallback contexts used: {missing_context_count}")
    print(f"[OK] output features: {output_path}")

    if not features_df.empty:
        show_cols = [
            c for c in [
                "row_id", "column", "dirty_value", "candidate_value",
                "candidate_rank", "source_score", "candidate_source_text",
                "candidate_rule_support_ratio", "rule_support_gain",
                "candidate_same_profile_ratio", "same_profile_ratio_gain",
                "candidate_age_validity_gain", "candidate_position_consistency_gain",
                "soccer_birthyear_season_age_conflict_reduction", "soccer_position_invalid_reduction",
                "relationship_consistency_gain",
                "education_age_conflict_reduction",
                "candidate_position_valid", "candidate_position_validity_gain", "candidate_year_validity_gain",
                "candidate_in_adult_domain", "candidate_domain_validity_gain",
                "edit_similarity_to_dirty", "bayes_mean_cond_prob",
                "contains_rule_expected_source", "contains_soccer_profile_source", "contains_position_source", "contains_team_source",
                "contains_profile_key_majority_source", "contains_entity_pair_majority_source", "contains_team_season_position_source",
                "is_from_profile_key_majority", "is_from_entity_pair_majority", "is_from_team_season_position",
                "profile_key_majority_ratio", "entity_pair_majority_ratio", "team_season_position_ratio",
                "candidate_entity_safe_fill", "candidate_position_profile_strong",
                "candidate_has_high_conf_profile_majority",
                "contains_adult_profile_source",
                "contains_relationship_source", "contains_domain_source",
                "is_weak_frequency_only_candidate", "candidate_is_pure_weak_global_for_entity",
            ] if c in features_df.columns
        ]
        print("\n[INFO] 前 20 行特征预览：")
        print(features_df[show_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()


"""
Adult repair candidate generation.

This script is adapted from the Flights candidate generator, but removes
Flights-specific source-aware time consensus and replaces it with Adult-specific
candidate generation based on:
  - expected values from rule/conflict evidence
  - Adult profile consistency evidence
  - relationship / sex / maritalstatus constraints
  - education-age consistency
  - domain/category values
  - neighbor and similar-row evidence
  - column top values with conservative scoring

Inputs
------
1. predicted_error_positions.csv
   At least contains row_id,column,value. It may also contain Adult feature columns.

2. candidate_llm_contexts_adult.jsonl
   Rich context records. Each record may contain:
     row_context
     column_context
     conflict_context
     similarity_context
     adult_consistency_context
     adult_v2_attribution_features
     bayesian_context

3. Optional fallback flat CSV.

4. Optional adult_dirty.csv for column dictionaries and row fallback contexts.

Outputs
-------
repair_candidates_expanded.csv
repair_candidates_grouped.csv
predicted_error_positions_adult_augmented.csv
adult_candidate_generation_summary.csv

Usage
-----
cd /mnt/mydata/dq/projects/Splittree/test/Error\ Correction\ adult

python revise_candidate_adult.py \
  --error_pred_file predicted_error_positions.csv \
  --context_file "/mnt/mydata/dq/projects/Splittree/test/Error Detection Adult/candidate_llm_contexts_adult.jsonl" \
  --dirty_table_csv /mnt/mydata/dq/projects/Splittree/adult/adult_dirty.csv
"""

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd


# ============================================================
# 1. Defaults and parameters
# ============================================================

DEFAULT_ERROR_PRED_FILE = "predicted_error_positions.csv"
DEFAULT_CONTEXT_FILE = "/mnt/mydata/dq/projects/Splittree/test/Error Detection Adult/candidate_llm_contexts_adult.jsonl"
DEFAULT_FALLBACK_CONTEXT_FLAT_FILE = "/mnt/mydata/dq/projects/Splittree/test/Error Detection Adult/candidate_llm_contexts_flat_adult.csv"
DEFAULT_DIRTY_TABLE_CSV = "/mnt/mydata/dq/projects/Splittree/adult/adult_dirty.csv"

OUTPUT_CANDIDATES_EXPANDED_CSV = "repair_candidates_expanded.csv"
OUTPUT_CANDIDATES_GROUPED_CSV = "repair_candidates_grouped.csv"
OUTPUT_AUGMENTED_ALLOWED_CELLS_CSV = "predicted_error_positions_adult_augmented.csv"
OUTPUT_SUMMARY_CSV = "adult_candidate_generation_summary.csv"

TOP_K_COLUMN_VALUES = 12
TOP_K_SIMILAR_ROW_VALUES = 8
MAX_CANDIDATES_PER_CELL = 24
INCLUDE_DIRTY_VALUE = False

MIN_STRING_SIM_FOR_COLUMN_VALUE = 0.45
MIN_STRING_SIM_FOR_DOMAIN_VALUE = 0.35

# Weak source scores
SCORE_DIRTY_VALUE = 0.05
SCORE_COLUMN_TOP = 0.22
SCORE_SIMILAR_ROW = 0.40
SCORE_NEIGHBOR_MAJORITY = 0.48

# Stronger source scores
SCORE_RULE_EXPECTED_HIGH = 0.95
SCORE_RULE_EXPECTED_MED = 0.86
SCORE_ADULT_PROFILE_TOP = 0.88
SCORE_ADULT_PROFILE_SECOND = 0.72
SCORE_ADULT_DIRECT_RULE = 0.90
SCORE_DOMAIN_NORMALIZE = 0.78

EMPTY_TOKENS = {"", "empty", "null", "none", "nan", "na", "n/a", "missing", "?", "{null}"}

ADULT_COLUMNS = [
    "age",
    "workclass",
    "education",
    "maritalstatus",
    "occupation",
    "relationship",
    "race",
    "sex",
    "hoursperweek",
    "country",
    "income",
]

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

EDUCATION_BY_ORDER = {v: k for k, v in EDUCATION_ORDER.items()}


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


def resolve_output_path(output_path: str) -> str:
    output_path = str(output_path)
    if os.path.isabs(output_path):
        return output_path
    return os.path.abspath(output_path)


def string_similarity(a: Any, b: Any) -> float:
    a = norm_text(a)
    b = norm_text(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def normalize_col_name(column: Any) -> str:
    c = norm_text(column)
    aliases = {
        "marital-status": "maritalstatus",
        "marital_status": "maritalstatus",
        "hours-per-week": "hoursperweek",
        "hours_per_week": "hoursperweek",
        "education-num": "education_num",
        "education_num": "education_num",
        "native-country": "country",
        "native_country": "country",
        "capital-gain": "capital_gain",
        "capital_loss": "capital_loss",
        "capital-loss": "capital_loss",
    }
    return aliases.get(c, c)


def canonical_domain_value(column: str, value: Any) -> str:
    col = normalize_col_name(column)
    raw = canonical_empty_text(value)
    if raw == "empty":
        return "empty"

    if col == "income":
        s = raw.replace(".", "").replace(" ", "")
        low = s.lower()
        if low in {"<=50k", "lessthan50k", "less_than_50k", "0"}:
            return "LessThan50K"
        if low in {">50k", "morethan50k", "more_than_50k", "1"}:
            return "MoreThan50K"
        return raw

    domains = ADULT_DOMAIN_VALUES.get(col)
    if domains:
        for v in domains:
            if norm_text(v) == norm_text(raw):
                return v
        # tolerate underscores
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


def canonical_age_bucket(x: Any) -> str:
    lo, hi = parse_numeric_range(x)
    if lo is None:
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
    if lo is None:
        return canonical_empty_text(x)
    if lo == hi:
        return str(int(lo)) if abs(lo - int(lo)) < 1e-9 else str(lo)
    return f"{int(lo)}-{int(hi)}"


def education_order(value: Any) -> Optional[int]:
    val = canonical_domain_value("education", value)
    return EDUCATION_ORDER.get(val)


def age_lower_upper_from_row(row_values: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    for key in ["age", "Age"]:
        if key in row_values:
            return parse_numeric_range(row_values[key])
    return None, None


def get_row_val(row_values: Dict[str, Any], *names: str) -> str:
    if not isinstance(row_values, dict):
        return ""
    for n in names:
        if n in row_values:
            return canonical_empty_text(row_values.get(n))
        # case-insensitive fallback
        for k, v in row_values.items():
            if norm_text(k) == norm_text(n):
                return canonical_empty_text(v)
    return ""


# ============================================================
# 3. Candidate handling
# ============================================================

def add_candidate(candidate_dict: Dict[str, Dict[str, Any]], value: Any, source: str, score: float, extra: Any = None):
    if value is None:
        return
    value = canonical_empty_text(value)
    if value == "" or value == "empty":
        return

    key = norm_text(value)
    if not key:
        return

    if key not in candidate_dict:
        candidate_dict[key] = {
            "candidate_value": value,
            "candidate_sources": set(),
            "source_score": float(score),
            "is_from_rule": 0,
            "is_from_neighbor": 0,
            "is_from_column_top": 0,
            "is_from_similarity": 0,
            "is_from_hint": 0,
            "is_from_dictionary": 0,
            "is_from_pattern_restore": 0,
            "is_from_adult_profile": 0,
            "is_from_adult_direct_rule": 0,
            "is_from_domain": 0,
            "extra_info": [],
        }

    item = candidate_dict[key]
    item["candidate_sources"].add(str(source))
    item["source_score"] = max(float(item["source_score"]), float(score))

    if source in {"rule_expected_value", "expected_values_from_rules", "rule_reason_parse"}:
        item["is_from_rule"] = 1
    elif source in {"neighbor_majority", "similar_row_target"}:
        item["is_from_neighbor"] = 1
    elif source in {"column_top_value", "alternative_values_from_column"}:
        item["is_from_column_top"] = 1
    elif source in {"similar_column_value"}:
        item["is_from_similarity"] = 1
    elif source in {"suggested_correct_value"}:
        item["is_from_hint"] = 1
    elif source.startswith("adult_profile") or source in {"same_profile_distribution", "work_profile_distribution"}:
        item["is_from_adult_profile"] = 1
    elif source.startswith("adult_direct") or source.startswith("relationship_") or source.startswith("sex_") or source.startswith("maritalstatus_") or source.startswith("education_"):
        item["is_from_adult_direct_rule"] = 1
        item["is_from_rule"] = 1
    elif source.startswith("domain_"):
        item["is_from_domain"] = 1
        item["is_from_dictionary"] = 1
    elif "dictionary" in source:
        item["is_from_dictionary"] = 1

    if extra is not None:
        item["extra_info"].append(str(extra))


# ============================================================
# 4. Input loading
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


def load_error_positions(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"ERROR_PRED_FILE 不存在: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")

    if "row_id" not in df.columns or "column" not in df.columns:
        raise ValueError("错误位置文件必须至少包含 row_id,column 字段。")

    # If full inference file is passed, filter predicted errors.
    if "final_pred_label" in df.columns:
        mask = pd.to_numeric(df["final_pred_label"], errors="coerce").fillna(0).astype(int) == 1
        df = df.loc[mask].copy()
    elif "final_pred_label_name" in df.columns:
        mask = df["final_pred_label_name"].astype(str).str.strip().str.lower().eq("error")
        df = df.loc[mask].copy()
    elif "detector_pred_label" in df.columns and "value" not in df.columns:
        mask = pd.to_numeric(df["detector_pred_label"], errors="coerce").fillna(0).astype(int) == 1
        df = df.loc[mask].copy()

    df["row_id"] = pd.to_numeric(df["row_id"], errors="raise").astype(int)
    df["column"] = df["column"].astype(str)

    if "value" not in df.columns:
        df["value"] = ""

    return df.drop_duplicates(subset=["row_id", "column"], keep="first").reset_index(drop=True)


def build_context_map(records: List[Dict[str, Any]]) -> Dict[Tuple[int, str], Dict[str, Any]]:
    cmap = {}
    for obj in records:
        if "row_id" not in obj or "column" not in obj:
            continue
        try:
            rid = int(obj["row_id"])
        except Exception:
            continue
        col = str(obj["column"])
        cmap[(rid, col)] = obj
    return cmap


def load_dirty_table(path: str) -> Optional[pd.DataFrame]:
    if path_exists_nonempty(path):
        return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    return None


def build_fallback_context(row_id: int, column: str, error_row: pd.Series, dirty_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    value = canonical_empty_text(error_row.get("value", ""))
    row_values = {}

    if dirty_df is not None and 0 <= int(row_id) < len(dirty_df):
        row_values = {str(c): canonical_empty_text(dirty_df.at[int(row_id), c]) for c in dirty_df.columns}
        if value in {"", "empty"} and column in dirty_df.columns:
            value = canonical_empty_text(dirty_df.at[int(row_id), column])

    ctx = {
        "row_id": int(row_id),
        "column": str(column),
        "value": value,
        "row_context": {"row_values": row_values},
        "column_context": {"column": column},
        "conflict_context": {},
        "similarity_context": {},
        "adult_consistency_context": {},
        "adult_v2_attribution_features": {},
    }

    # Copy known flat features from error row.
    for k in [
        "semantic_type", "detected_type", "violation_count", "conflict_score",
        "neighbor_majority_value", "neighbor_majority_ratio",
        "main_rule_type", "main_usage_role",
        "adult_evidence_flags", "adult_consistency_score",
        "target_is_primary_column", "target_is_context_only_column",
        "relationship_marital_conflict", "relationship_age_conflict",
        "relationship_sex_conflict", "sex_value_invalid", "education_age_extreme_conflict",
    ]:
        if k in error_row.index:
            ctx[k] = error_row.get(k)

    return ctx


# ============================================================
# 5. Global dictionaries
# ============================================================

def consume_row_values(row_vals: Dict[str, Any], global_dict: Dict[str, Any]):
    if not isinstance(row_vals, dict):
        return

    for c, v in row_vals.items():
        val = canonical_empty_text(v)
        if val and val != "empty":
            global_dict["column_value_counter"][str(c)][val] += 1
            global_dict["column_value_counter"][normalize_col_name(c)][canonical_domain_value(c, val)] += 1

    age = canonical_age_bucket(get_row_val(row_vals, "age"))
    marital = canonical_domain_value("maritalstatus", get_row_val(row_vals, "maritalstatus", "marital-status"))
    sex = canonical_domain_value("sex", get_row_val(row_vals, "sex"))
    relationship = canonical_domain_value("relationship", get_row_val(row_vals, "relationship"))
    education = canonical_domain_value("education", get_row_val(row_vals, "education"))
    hours = canonical_hours_value(get_row_val(row_vals, "hoursperweek", "hours-per-week"))
    income = canonical_domain_value("income", get_row_val(row_vals, "income"))

    profile_key = (age, marital, sex)
    if relationship and relationship != "empty":
        global_dict["relationship_by_age_marital_sex"][profile_key][relationship] += 1

    rel_key = norm_text(relationship)
    if rel_key:
        if sex and sex != "empty":
            global_dict["sex_by_relationship"][rel_key][sex] += 1
        if marital and marital != "empty":
            global_dict["marital_by_relationship"][rel_key][marital] += 1

    work_key = (
        canonical_domain_value("workclass", get_row_val(row_vals, "workclass")),
        canonical_domain_value("occupation", get_row_val(row_vals, "occupation")),
        education,
    )
    if income and income != "empty":
        global_dict["income_by_work_profile"][work_key][income] += 1
    if hours and hours != "empty":
        global_dict["hours_by_work_profile"][work_key][hours] += 1


def build_global_dictionaries(context_records: List[Dict[str, Any]], dirty_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    global_dict = {
        "column_value_counter": defaultdict(Counter),
        "relationship_by_age_marital_sex": defaultdict(Counter),
        "sex_by_relationship": defaultdict(Counter),
        "marital_by_relationship": defaultdict(Counter),
        "income_by_work_profile": defaultdict(Counter),
        "hours_by_work_profile": defaultdict(Counter),
    }

    for obj in context_records:
        row_context = obj.get("row_context", {})
        if isinstance(row_context, dict):
            consume_row_values(row_context.get("row_values", {}), global_dict)

        sim_ctx = obj.get("similarity_context", {})
        if isinstance(sim_ctx, dict):
            for item in sim_ctx.get("topk_similar_rows", []) or []:
                if isinstance(item, dict):
                    consume_row_values(item.get("row_values", {}), global_dict)

        col_ctx = obj.get("column_context", {})
        if isinstance(col_ctx, dict):
            col = str(col_ctx.get("column", obj.get("column", "")))
            for t in col_ctx.get("top_values_with_counts", []) or []:
                if isinstance(t, dict):
                    val = canonical_empty_text(t.get("value", ""))
                    cnt = safe_int(t.get("count", 1), 1)
                    if val and val != "empty":
                        global_dict["column_value_counter"][col][val] += cnt
                        global_dict["column_value_counter"][normalize_col_name(col)][canonical_domain_value(col, val)] += cnt

    if dirty_df is not None:
        for _, r in dirty_df.iterrows():
            consume_row_values({str(c): r[c] for c in dirty_df.columns}, global_dict)

    return global_dict


# ============================================================
# 6. Adult-specific candidate generation
# ============================================================

def add_domain_candidates(column: str, dirty_value: str, candidate_dict: Dict[str, Dict[str, Any]], global_dict: Dict[str, Any]):
    col = normalize_col_name(column)
    domains = ADULT_DOMAIN_VALUES.get(col)
    if not domains:
        return

    canonical = canonical_domain_value(col, dirty_value)
    if canonical != dirty_value and canonical != "empty" and norm_text(canonical) != norm_text(dirty_value):
        add_candidate(candidate_dict, canonical, "domain_canonicalize", SCORE_DOMAIN_NORMALIZE, "canonical domain value")

    # Similar domain values. Keep as weak candidates.
    for v in domains:
        if norm_text(v) == norm_text(dirty_value):
            continue
        sim = string_similarity(dirty_value, v)
        if sim >= MIN_STRING_SIM_FOR_DOMAIN_VALUE:
            add_candidate(candidate_dict, v, "domain_similar_value", 0.28 + 0.25 * sim, f"sim={sim:.4f}")


def add_adult_profile_distribution_candidates(ctx: Dict[str, Any], column: str, dirty_value: str, candidate_dict: Dict[str, Dict[str, Any]]):
    adult_ctx = ctx.get("adult_consistency_context", {})
    if not isinstance(adult_ctx, dict):
        return

    # same_profile_distribution: target column distribution under same profile
    same_dist = adult_ctx.get("same_profile_distribution", {}) or {}
    col_dist = same_dist.get(column) or same_dist.get(normalize_col_name(column))
    if isinstance(col_dist, dict):
        tops = col_dist.get("top_values", []) or []
        group_size = safe_int(col_dist.get("group_size", 0), 0)
        for idx, item in enumerate(tops[:5]):
            if not isinstance(item, dict):
                continue
            val = item.get("value", "")
            cnt = safe_int(item.get("count", 0), 0)
            if is_empty_like_value(val) or norm_text(val) == norm_text(dirty_value):
                continue
            ratio = cnt / max(group_size, 1)
            if idx == 0:
                score = SCORE_ADULT_PROFILE_TOP if ratio >= 0.5 else 0.68
            else:
                score = SCORE_ADULT_PROFILE_SECOND if ratio >= 0.25 else 0.45
            add_candidate(
                candidate_dict,
                canonical_domain_value(column, val),
                "same_profile_distribution",
                score,
                f"group_size={group_size}|count={cnt}|ratio={ratio:.4f}",
            )

    # work_profile_distribution for income/hours/education.
    work_dist = adult_ctx.get("work_profile_distribution", {}) or {}
    col_dist2 = work_dist.get(column) or work_dist.get(normalize_col_name(column))
    if isinstance(col_dist2, dict):
        tops = col_dist2.get("top_values", []) or []
        group_size = safe_int(col_dist2.get("group_size", 0), 0)
        for idx, item in enumerate(tops[:5]):
            if not isinstance(item, dict):
                continue
            val = item.get("value", "")
            cnt = safe_int(item.get("count", 0), 0)
            if is_empty_like_value(val) or norm_text(val) == norm_text(dirty_value):
                continue
            ratio = cnt / max(group_size, 1)
            score = 0.65 if idx == 0 and ratio >= 0.4 else 0.38
            add_candidate(
                candidate_dict,
                canonical_domain_value(column, val),
                "work_profile_distribution",
                score,
                f"group_size={group_size}|count={cnt}|ratio={ratio:.4f}",
            )


def add_relationship_candidates(ctx: Dict[str, Any], dirty_value: str, candidate_dict: Dict[str, Dict[str, Any]], global_dict: Dict[str, Any]):
    row_values = (ctx.get("row_context", {}) or {}).get("row_values", {}) or {}
    age = canonical_age_bucket(get_row_val(row_values, "age"))
    marital = canonical_domain_value("maritalstatus", get_row_val(row_values, "maritalstatus", "marital-status"))
    sex = canonical_domain_value("sex", get_row_val(row_values, "sex"))
    rel = canonical_domain_value("relationship", dirty_value)

    evidence_flags = []
    adult_ctx = ctx.get("adult_consistency_context", {}) or {}
    if isinstance(adult_ctx, dict):
        evidence_flags.extend(adult_ctx.get("evidence_flags", []) or [])
        bundle = adult_ctx.get("adult_profile_bundle", {}) or {}
        if isinstance(bundle, dict):
            evidence_flags.extend(bundle.get("adult_profile_inconsistencies", []) or [])

    flat_flags = ctx.get("adult_evidence_flags", "")
    if flat_flags:
        evidence_flags.extend(str(flat_flags).split("|"))

    # Rule: spouse relationship with non-married status is usually not-in-family/own-child depending age.
    if rel in {"Husband", "Wife"}:
        if marital in {"Never-married"}:
            if age in {"<18", "18-21"}:
                add_candidate(candidate_dict, "Own-child", "relationship_nonmarried_young_to_own_child", 0.92, "spouse but never-married and young")
            add_candidate(candidate_dict, "Not-in-family", "relationship_nonmarried_spouse_to_not_in_family", 0.78, "spouse but never-married")
        elif marital in {"Divorced", "Separated", "Widowed", "Married-spouse-absent"}:
            add_candidate(candidate_dict, "Not-in-family", "relationship_spouse_marital_conflict", 0.72, f"marital={marital}")

    # Rule: Wife/Husband sex conflict.
    if rel == "Wife" and sex == "Male":
        add_candidate(candidate_dict, "Husband", "relationship_wife_but_male", 0.70, "relationship/sex mismatch")
    if rel == "Husband" and sex == "Female":
        add_candidate(candidate_dict, "Wife", "relationship_husband_but_female", 0.70, "relationship/sex mismatch")

    # Profile distribution learned from table/context.
    key = (age, marital, sex)
    counter = global_dict["relationship_by_age_marital_sex"].get(key, Counter())
    total = sum(counter.values())
    for idx, (val, cnt) in enumerate(counter.most_common(5)):
        if norm_text(val) == norm_text(dirty_value):
            continue
        ratio = cnt / max(total, 1)
        score = 0.84 if idx == 0 and cnt >= 3 and ratio >= 0.55 else 0.50
        add_candidate(candidate_dict, val, "adult_profile_relationship_by_age_marital_sex", score, f"count={cnt}|ratio={ratio:.4f}|key={key}")


def add_sex_candidates(ctx: Dict[str, Any], dirty_value: str, candidate_dict: Dict[str, Dict[str, Any]], global_dict: Dict[str, Any]):
    row_values = (ctx.get("row_context", {}) or {}).get("row_values", {}) or {}
    rel = canonical_domain_value("relationship", get_row_val(row_values, "relationship"))

    if rel == "Husband" and norm_text(dirty_value) != "male":
        add_candidate(candidate_dict, "Male", "sex_from_husband_relationship", 0.88, "relationship=Husband")
    if rel == "Wife" and norm_text(dirty_value) != "female":
        add_candidate(candidate_dict, "Female", "sex_from_wife_relationship", 0.88, "relationship=Wife")

    counter = global_dict["sex_by_relationship"].get(norm_text(rel), Counter())
    total = sum(counter.values())
    for val, cnt in counter.most_common(2):
        if norm_text(val) == norm_text(dirty_value):
            continue
        ratio = cnt / max(total, 1)
        if cnt >= 5 and ratio >= 0.75:
            add_candidate(candidate_dict, val, "adult_profile_sex_by_relationship", 0.72, f"count={cnt}|ratio={ratio:.4f}")


def add_maritalstatus_candidates(ctx: Dict[str, Any], dirty_value: str, candidate_dict: Dict[str, Dict[str, Any]], global_dict: Dict[str, Any]):
    row_values = (ctx.get("row_context", {}) or {}).get("row_values", {}) or {}
    rel = canonical_domain_value("relationship", get_row_val(row_values, "relationship"))
    marital = canonical_domain_value("maritalstatus", dirty_value)

    if rel in {"Husband", "Wife"} and marital not in {"Married-civ-spouse", "Married-AF-spouse"}:
        add_candidate(candidate_dict, "Married-civ-spouse", "maritalstatus_from_spouse_relationship", 0.82, f"relationship={rel}")

    if rel in {"Own-child"} and marital == "Married-civ-spouse":
        add_candidate(candidate_dict, "Never-married", "maritalstatus_own_child_conflict", 0.68, "relationship=Own-child")

    counter = global_dict["marital_by_relationship"].get(norm_text(rel), Counter())
    total = sum(counter.values())
    for val, cnt in counter.most_common(3):
        if norm_text(val) == norm_text(dirty_value):
            continue
        ratio = cnt / max(total, 1)
        if cnt >= 5 and ratio >= 0.6:
            add_candidate(candidate_dict, val, "adult_profile_marital_by_relationship", 0.58, f"count={cnt}|ratio={ratio:.4f}")


def add_education_candidates(ctx: Dict[str, Any], dirty_value: str, candidate_dict: Dict[str, Dict[str, Any]]):
    row_values = (ctx.get("row_context", {}) or {}).get("row_values", {}) or {}
    lo, hi = age_lower_upper_from_row(row_values)
    cur_order = education_order(dirty_value)

    if lo is not None and hi is not None and hi <= 17 and cur_order is not None:
        # Minors with very advanced degree is suspicious; only propose common lower values.
        if cur_order >= EDUCATION_ORDER["Bachelors"]:
            add_candidate(candidate_dict, "HS-grad", "education_minor_advanced_degree_conflict", 0.72, "minor with advanced degree")
            add_candidate(candidate_dict, "Some-college", "education_minor_advanced_degree_conflict", 0.62, "minor with advanced degree")

    if lo is not None and hi is not None and hi <= 21 and cur_order is not None:
        if cur_order >= EDUCATION_ORDER["Masters"]:
            add_candidate(candidate_dict, "Some-college", "education_young_extreme_degree_conflict", 0.68, "young with very advanced degree")
            add_candidate(candidate_dict, "Bachelors", "education_young_extreme_degree_conflict", 0.55, "young with very advanced degree")


def add_age_candidates(ctx: Dict[str, Any], dirty_value: str, candidate_dict: Dict[str, Dict[str, Any]]):
    fixed = canonical_age_bucket(dirty_value)
    if fixed and norm_text(fixed) != norm_text(dirty_value):
        add_candidate(candidate_dict, fixed, "adult_direct_age_bucket_normalize", 0.82, "normalize age bucket")

    # Do not aggressively infer age from relationship; too risky.
    row_values = (ctx.get("row_context", {}) or {}).get("row_values", {}) or {}
    rel = canonical_domain_value("relationship", get_row_val(row_values, "relationship"))
    if rel in {"Husband", "Wife"} and fixed == "<18":
        add_candidate(candidate_dict, "22-30", "adult_direct_age_spouse_very_young_conflict", 0.45, "weak candidate only")


def add_hours_candidates(ctx: Dict[str, Any], dirty_value: str, candidate_dict: Dict[str, Dict[str, Any]], global_dict: Dict[str, Any]):
    fixed = canonical_hours_value(dirty_value)
    if fixed and norm_text(fixed) != norm_text(dirty_value):
        add_candidate(candidate_dict, fixed, "adult_direct_hours_normalize", 0.80, "normalize hours value")

    row_values = (ctx.get("row_context", {}) or {}).get("row_values", {}) or {}
    work_key = (
        canonical_domain_value("workclass", get_row_val(row_values, "workclass")),
        canonical_domain_value("occupation", get_row_val(row_values, "occupation")),
        canonical_domain_value("education", get_row_val(row_values, "education")),
    )
    counter = global_dict["hours_by_work_profile"].get(work_key, Counter())
    total = sum(counter.values())
    for idx, (val, cnt) in enumerate(counter.most_common(5)):
        if norm_text(val) == norm_text(dirty_value):
            continue
        ratio = cnt / max(total, 1)
        if idx == 0 and cnt >= 10 and ratio >= 0.35:
            add_candidate(candidate_dict, val, "adult_profile_hours_by_work_profile", 0.52, f"count={cnt}|ratio={ratio:.4f}")


def add_income_candidates(ctx: Dict[str, Any], dirty_value: str, candidate_dict: Dict[str, Dict[str, Any]], global_dict: Dict[str, Any]):
    fixed = canonical_domain_value("income", dirty_value)
    if fixed and fixed != "empty" and norm_text(fixed) != norm_text(dirty_value):
        add_candidate(candidate_dict, fixed, "domain_income_canonicalize", 0.86, "canonical income")

    row_values = (ctx.get("row_context", {}) or {}).get("row_values", {}) or {}
    work_key = (
        canonical_domain_value("workclass", get_row_val(row_values, "workclass")),
        canonical_domain_value("occupation", get_row_val(row_values, "occupation")),
        canonical_domain_value("education", get_row_val(row_values, "education")),
    )
    counter = global_dict["income_by_work_profile"].get(work_key, Counter())
    total = sum(counter.values())
    for val, cnt in counter.most_common(2):
        if norm_text(val) == norm_text(dirty_value):
            continue
        ratio = cnt / max(total, 1)
        if cnt >= 50 and ratio >= 0.70:
            add_candidate(candidate_dict, val, "adult_profile_income_by_work_profile", 0.56, f"count={cnt}|ratio={ratio:.4f}")


def add_adult_column_specific_candidates(ctx: Dict[str, Any], column: str, dirty_value: str, candidate_dict: Dict[str, Dict[str, Any]], global_dict: Dict[str, Any]):
    col = normalize_col_name(column)

    if col == "relationship":
        add_relationship_candidates(ctx, dirty_value, candidate_dict, global_dict)
    elif col == "sex":
        add_sex_candidates(ctx, dirty_value, candidate_dict, global_dict)
    elif col == "maritalstatus":
        add_maritalstatus_candidates(ctx, dirty_value, candidate_dict, global_dict)
    elif col == "education":
        add_education_candidates(ctx, dirty_value, candidate_dict)
    elif col == "age":
        add_age_candidates(ctx, dirty_value, candidate_dict)
    elif col == "hoursperweek":
        add_hours_candidates(ctx, dirty_value, candidate_dict, global_dict)
    elif col == "income":
        add_income_candidates(ctx, dirty_value, candidate_dict, global_dict)

    add_domain_candidates(col, dirty_value, candidate_dict, global_dict)


# ============================================================
# 7. Generic evidence candidate generation
# ============================================================

def add_conflict_candidates(ctx: Dict[str, Any], dirty_value: str, candidate_dict: Dict[str, Dict[str, Any]]):
    conflict = ctx.get("conflict_context", {})
    if not isinstance(conflict, dict):
        return

    expected = conflict.get("expected_values_from_rules", []) or []
    for item in expected:
        if isinstance(item, dict):
            val = item.get("value", item.get("expected_value", ""))
            priority = norm_text(item.get("priority", ""))
            usage = norm_text(item.get("usage_role", ""))
            rule_type = str(item.get("rule_type", ""))
            if is_empty_like_value(val) or norm_text(val) == norm_text(dirty_value):
                continue
            score = SCORE_RULE_EXPECTED_HIGH if priority == "high" or usage == "strong_rule" else SCORE_RULE_EXPECTED_MED
            add_candidate(candidate_dict, val, "rule_expected_value", score, f"rule_type={rule_type}|priority={priority}|usage={usage}")

    # violated_rules_detailed may have expected_value directly
    for item in conflict.get("violated_rules_detailed", []) or []:
        if not isinstance(item, dict):
            continue
        val = item.get("expected_value", None)
        if is_empty_like_value(val) or norm_text(val) == norm_text(dirty_value):
            continue
        priority = norm_text(item.get("priority", ""))
        usage = norm_text(item.get("usage_role", ""))
        rule_type = str(item.get("rule_type", ""))
        score = SCORE_RULE_EXPECTED_HIGH if priority == "high" or usage == "strong_rule" else SCORE_RULE_EXPECTED_MED
        add_candidate(candidate_dict, val, "expected_values_from_rules", score, f"rule_type={rule_type}|priority={priority}|usage={usage}")

    # all_alternative_values / alternative_values_from_column
    for field, source, base_score in [
        ("alternative_values_from_column", "alternative_values_from_column", SCORE_COLUMN_TOP),
        ("all_alternative_values", "all_alternative_values", 0.40),
    ]:
        for item in conflict.get(field, []) or []:
            if not isinstance(item, dict):
                continue
            val = item.get("value", "")
            src = item.get("source", source)
            if is_empty_like_value(val) or norm_text(val) == norm_text(dirty_value):
                continue
            if norm_text(src) in {"rule_reason_parse", "rule_expected_value"}:
                score = SCORE_RULE_EXPECTED_MED
                src_name = "rule_reason_parse"
            elif norm_text(src) == "column_top_value":
                score = SCORE_COLUMN_TOP
                src_name = "column_top_value"
            else:
                score = base_score
                src_name = str(src)
            add_candidate(candidate_dict, val, src_name, score, f"field={field}")


def add_similarity_candidates(ctx: Dict[str, Any], dirty_value: str, candidate_dict: Dict[str, Dict[str, Any]]):
    sim_ctx = ctx.get("similarity_context", {})
    if not isinstance(sim_ctx, dict):
        return

    maj = sim_ctx.get("neighbor_majority_value", "")
    maj_ratio = safe_float(sim_ctx.get("neighbor_majority_ratio", 0.0), 0.0)
    if not is_empty_like_value(maj) and norm_text(maj) != norm_text(dirty_value):
        add_candidate(candidate_dict, maj, "neighbor_majority", SCORE_NEIGHBOR_MAJORITY + 0.25 * maj_ratio, f"ratio={maj_ratio:.4f}")

    counter = Counter()
    for item in sim_ctx.get("topk_similar_rows", []) or []:
        if not isinstance(item, dict):
            continue
        val = item.get("target_column_value", "")
        if is_empty_like_value(val) or norm_text(val) == norm_text(dirty_value):
            continue
        weight = safe_float(item.get("similarity", 1.0), 1.0)
        counter[canonical_empty_text(val)] += weight

    total = sum(counter.values())
    for idx, (val, weight) in enumerate(counter.most_common(TOP_K_SIMILAR_ROW_VALUES)):
        ratio = weight / max(total, 1e-9)
        score = SCORE_SIMILAR_ROW + 0.30 * ratio
        add_candidate(candidate_dict, val, "similar_row_target", score, f"weighted_count={weight:.4f}|ratio={ratio:.4f}")


def add_column_context_candidates(ctx: Dict[str, Any], column: str, dirty_value: str, candidate_dict: Dict[str, Dict[str, Any]]):
    col_ctx = ctx.get("column_context", {})
    if not isinstance(col_ctx, dict):
        return

    for item in col_ctx.get("top_values_with_counts", []) or []:
        if not isinstance(item, dict):
            continue
        val = item.get("value", "")
        cnt = safe_int(item.get("count", 1), 1)
        ratio = safe_float(item.get("ratio", 0.0), 0.0)
        if is_empty_like_value(val) or norm_text(val) == norm_text(dirty_value):
            continue
        sim = string_similarity(dirty_value, val)
        # category top values are weak; include but low score.
        score = SCORE_COLUMN_TOP + min(0.18, ratio)
        add_candidate(candidate_dict, val, "column_top_value", score, f"count={cnt}|ratio={ratio:.4f}|sim={sim:.4f}")


def add_global_column_candidates(column: str, dirty_value: str, candidate_dict: Dict[str, Dict[str, Any]], global_dict: Dict[str, Any]):
    col = normalize_col_name(column)
    counter = global_dict["column_value_counter"].get(column, Counter()) + global_dict["column_value_counter"].get(col, Counter())
    for val, cnt in counter.most_common(TOP_K_COLUMN_VALUES):
        if is_empty_like_value(val) or norm_text(val) == norm_text(dirty_value):
            continue
        sim = string_similarity(dirty_value, val)
        if sim >= MIN_STRING_SIM_FOR_COLUMN_VALUE or col in ADULT_DOMAIN_VALUES:
            add_candidate(candidate_dict, val, "global_column_dictionary", 0.18 + min(0.15, sim * 0.15), f"freq={cnt}|sim={sim:.4f}")


def add_llm_hint_candidates(ctx: Dict[str, Any], dirty_value: str, candidate_dict: Dict[str, Dict[str, Any]]):
    for field in ["suggested_correct_value", "llm_suggested_correct_value"]:
        val = ctx.get(field, "")
        if not is_empty_like_value(val) and norm_text(val) != norm_text(dirty_value):
            add_candidate(candidate_dict, val, "suggested_correct_value", 0.82, field)

    # Flat error-position row may carry suggested_correct_value in ctx itself.
    if "reason_detailed" in ctx and "suggested_correct_value" in ctx:
        val = ctx.get("suggested_correct_value", "")
        if not is_empty_like_value(val) and norm_text(val) != norm_text(dirty_value):
            add_candidate(candidate_dict, val, "suggested_correct_value", 0.82, "flat_suggested_correct_value")


# ============================================================
# 8. Candidate building
# ============================================================

def candidate_to_rows(row_id: int, column: str, dirty_value: str, ctx: Dict[str, Any], candidate_dict: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    items = list(candidate_dict.values())

    # Remove dirty value if configured.
    filtered = []
    for it in items:
        if not INCLUDE_DIRTY_VALUE and norm_text(it["candidate_value"]) == norm_text(dirty_value):
            continue
        filtered.append(it)

    # Rank by score and strength.
    filtered = sorted(
        filtered,
        key=lambda x: (
            x.get("source_score", 0.0),
            x.get("is_from_rule", 0),
            x.get("is_from_adult_profile", 0),
            x.get("is_from_adult_direct_rule", 0),
            x.get("is_from_neighbor", 0),
        ),
        reverse=True,
    )[:MAX_CANDIDATES_PER_CELL]

    for rank, it in enumerate(filtered, start=1):
        rows.append({
            "row_id": int(row_id),
            "column": str(column),
            "dirty_value": dirty_value,
            "candidate_value": it["candidate_value"],
            "candidate_rank": rank,
            "source_score": float(it.get("source_score", 0.0)),
            "candidate_sources": "|".join(sorted(it.get("candidate_sources", []))),
            "is_from_rule": int(it.get("is_from_rule", 0)),
            "is_from_neighbor": int(it.get("is_from_neighbor", 0)),
            "is_from_column_top": int(it.get("is_from_column_top", 0)),
            "is_from_similarity": int(it.get("is_from_similarity", 0)),
            "is_from_hint": int(it.get("is_from_hint", 0)),
            "is_from_dictionary": int(it.get("is_from_dictionary", 0)),
            "is_from_pattern_restore": int(it.get("is_from_pattern_restore", 0)),
            "is_from_adult_profile": int(it.get("is_from_adult_profile", 0)),
            "is_from_adult_direct_rule": int(it.get("is_from_adult_direct_rule", 0)),
            "is_from_domain": int(it.get("is_from_domain", 0)),
            "extra_info": " || ".join(it.get("extra_info", [])[:10]),
            "semantic_type": ctx.get("semantic_type", (ctx.get("column_context", {}) or {}).get("semantic_type", "")),
            "main_rule_type": ctx.get("main_rule_type", (ctx.get("conflict_context", {}) or {}).get("main_rule_type", "")),
            "main_usage_role": ctx.get("main_usage_role", (ctx.get("conflict_context", {}) or {}).get("main_usage_role", "")),
            "violation_count": ctx.get("violation_count", (ctx.get("conflict_context", {}) or {}).get("violation_count", "")),
            "conflict_score": ctx.get("conflict_score", ""),
            "neighbor_majority_value": (ctx.get("similarity_context", {}) or {}).get("neighbor_majority_value", ctx.get("neighbor_majority_value", "")),
            "neighbor_majority_ratio": (ctx.get("similarity_context", {}) or {}).get("neighbor_majority_ratio", ctx.get("neighbor_majority_ratio", "")),
            "adult_evidence_flags": ctx.get("adult_evidence_flags", "|".join((ctx.get("adult_consistency_context", {}) or {}).get("evidence_flags", []) or [])),
        })

    return rows


def build_candidates_for_cell(row_id: int, column: str, error_row: pd.Series, ctx: Dict[str, Any], global_dict: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    dirty_value = canonical_empty_text(error_row.get("value", ctx.get("value", "")))
    if dirty_value in {"", "empty"}:
        # fallback from row_context if possible
        row_values = (ctx.get("row_context", {}) or {}).get("row_values", {}) or {}
        dirty_value = get_row_val(row_values, column)

    candidate_dict = {}

    if INCLUDE_DIRTY_VALUE:
        add_candidate(candidate_dict, dirty_value, "dirty_value", SCORE_DIRTY_VALUE, "original value")

    # Generic evidence.
    add_conflict_candidates(ctx, dirty_value, candidate_dict)
    add_similarity_candidates(ctx, dirty_value, candidate_dict)
    add_column_context_candidates(ctx, column, dirty_value, candidate_dict)
    add_global_column_candidates(column, dirty_value, candidate_dict, global_dict)
    add_llm_hint_candidates(ctx, dirty_value, candidate_dict)

    # Adult-specific evidence.
    add_adult_profile_distribution_candidates(ctx, normalize_col_name(column), dirty_value, candidate_dict)
    add_adult_column_specific_candidates(ctx, column, dirty_value, candidate_dict, global_dict)

    rows = candidate_to_rows(row_id, column, dirty_value, ctx, candidate_dict)

    summary = {
        "row_id": int(row_id),
        "column": str(column),
        "dirty_value": dirty_value,
        "candidate_count": len(rows),
        "top_candidate": rows[0]["candidate_value"] if rows else "",
        "top_candidate_score": rows[0]["source_score"] if rows else 0.0,
        "top_candidate_sources": rows[0]["candidate_sources"] if rows else "",
        "has_rule_candidate": int(any(r["is_from_rule"] == 1 for r in rows)),
        "has_adult_profile_candidate": int(any(r["is_from_adult_profile"] == 1 for r in rows)),
        "has_adult_direct_rule_candidate": int(any(r["is_from_adult_direct_rule"] == 1 for r in rows)),
        "has_neighbor_candidate": int(any(r["is_from_neighbor"] == 1 for r in rows)),
        "has_domain_candidate": int(any(r["is_from_domain"] == 1 for r in rows)),
    }

    return rows, summary


# ============================================================
# 9. Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--error_pred_file", default=DEFAULT_ERROR_PRED_FILE)
    p.add_argument("--context_file", default=DEFAULT_CONTEXT_FILE)
    p.add_argument("--fallback_context_flat_file", default=DEFAULT_FALLBACK_CONTEXT_FLAT_FILE)
    p.add_argument("--dirty_table_csv", default=DEFAULT_DIRTY_TABLE_CSV)
    p.add_argument("--output_candidates_expanded_csv", default=OUTPUT_CANDIDATES_EXPANDED_CSV)
    p.add_argument("--output_candidates_grouped_csv", default=OUTPUT_CANDIDATES_GROUPED_CSV)
    p.add_argument("--output_augmented_allowed_cells_csv", default=OUTPUT_AUGMENTED_ALLOWED_CELLS_CSV)
    p.add_argument("--output_summary_csv", default=OUTPUT_SUMMARY_CSV)
    return p.parse_args()


def main():
    args = parse_args()

    print("========== Adult Repair Candidate Generation ==========")
    print(f"ERROR_PRED_FILE: {args.error_pred_file}")
    print(f"CONTEXT_FILE: {args.context_file}")
    print(f"FALLBACK_CONTEXT_FLAT_FILE: {args.fallback_context_flat_file}")
    print(f"DIRTY_TABLE_CSV: {args.dirty_table_csv}")

    error_df = load_error_positions(args.error_pred_file)
    print(f"[INFO] loaded error positions: {len(error_df)}")

    context_records = []
    if path_exists_nonempty(args.context_file):
        context_records.extend(load_context_records(args.context_file))
        print(f"[INFO] loaded rich contexts: {len(context_records)}")
    else:
        print(f"[WARN] context_file not found: {args.context_file}")

    if path_exists_nonempty(args.fallback_context_flat_file):
        flat_records = load_context_records(args.fallback_context_flat_file)
        context_records.extend(flat_records)
        print(f"[INFO] loaded fallback flat contexts: {len(flat_records)}")
    else:
        print(f"[INFO] no fallback flat context loaded.")

    dirty_df = load_dirty_table(args.dirty_table_csv)
    if dirty_df is not None:
        print(f"[INFO] loaded dirty table: shape={dirty_df.shape}")
    else:
        print("[INFO] no dirty table loaded.")

    context_map = build_context_map(context_records)
    global_dict = build_global_dictionaries(context_records, dirty_df)

    expanded_rows = []
    grouped_rows = []
    missing_context_count = 0

    for _, erow in error_df.iterrows():
        row_id = int(erow["row_id"])
        column = str(erow["column"])

        ctx = context_map.get((row_id, column))
        if ctx is None:
            missing_context_count += 1
            ctx = build_fallback_context(row_id, column, erow, dirty_df)

        cand_rows, summary = build_candidates_for_cell(row_id, column, erow, ctx, global_dict)
        expanded_rows.extend(cand_rows)
        grouped_rows.append(summary)

    expanded_df = pd.DataFrame(expanded_rows)
    grouped_df = pd.DataFrame(grouped_rows)

    # Augmented allowed cells: for Adult we keep predicted positions as the candidate cell universe.
    # Unlike Flights, we do not add extra consensus cells here.
    augmented_df = error_df.copy().drop_duplicates(subset=["row_id", "column"], keep="first").reset_index(drop=True)

    expanded_path = resolve_output_path(args.output_candidates_expanded_csv)
    grouped_path = resolve_output_path(args.output_candidates_grouped_csv)
    augmented_path = resolve_output_path(args.output_augmented_allowed_cells_csv)
    summary_path = resolve_output_path(args.output_summary_csv)

    expanded_df.to_csv(expanded_path, index=False, encoding="utf-8-sig")
    grouped_df.to_csv(grouped_path, index=False, encoding="utf-8-sig")
    augmented_df.to_csv(augmented_path, index=False, encoding="utf-8-sig")

    # Summary by column
    summary_rows = []
    if not grouped_df.empty:
        for col, g in grouped_df.groupby("column", sort=False):
            summary_rows.append({
                "column": col,
                "cells": len(g),
                "cells_with_candidates": int((g["candidate_count"] > 0).sum()),
                "avg_candidate_count": float(g["candidate_count"].mean()),
                "max_candidate_count": int(g["candidate_count"].max()),
                "has_rule_candidate_cells": int(g["has_rule_candidate"].sum()),
                "has_adult_profile_candidate_cells": int(g["has_adult_profile_candidate"].sum()),
                "has_adult_direct_rule_candidate_cells": int(g["has_adult_direct_rule_candidate"].sum()),
                "has_neighbor_candidate_cells": int(g["has_neighbor_candidate"].sum()),
                "has_domain_candidate_cells": int(g["has_domain_candidate"].sum()),
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n========== Candidate Generation Summary ==========")
    print(f"input error cells: {len(error_df)}")
    print(f"missing context fallback used: {missing_context_count}")
    print(f"expanded candidate rows: {len(expanded_df)}")
    print(f"grouped candidate cells: {len(grouped_df)}")

    if not grouped_df.empty:
        print(f"cells with candidates: {(grouped_df['candidate_count'] > 0).sum()}")
        print(f"avg candidates per cell: {grouped_df['candidate_count'].mean():.4f}")

    print(f"\n[OK] expanded candidates: {expanded_path}")
    print(f"[OK] grouped candidates: {grouped_path}")
    print(f"[OK] augmented allowed cells: {augmented_path}")
    print(f"[OK] summary: {summary_path}")

    if not summary_df.empty:
        print("\n---- Column summary ----")
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

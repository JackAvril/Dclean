import json
import math
import re
from collections import Counter
from typing import Dict, Any, List, Optional

import pandas as pd


# ============================================================
# 1. 用户配置区
# ============================================================

INPUT_CSV = "/mnt/mydata/dq/projects/Splittree/movies/movies_dirty.csv"
INPUT_RULE_JSON = "rule_pool_movies.json"

OUTPUT_JSONL = "candidate_cells_movies.jsonl"
OUTPUT_CSV = "candidate_cells_movies.csv"

MISSING_TOKENS = {
    "", "empty", "null", "none", "nan", "na", "n/a", "missing", "unk",
    "unknown", "not available", "not applicable", "tbd", "na."
}

SKIP_NULL_FOR_NON_NOTNULL_RULES = True
MAX_FD_MAPPING_SIZE = 200000
MAX_CONTEXT_MAPPING_SIZE = 200000

PRIORITY_WEIGHT = {"high": 1.5, "medium": 1.0, "low": 0.6}

RULE_TYPE_WEIGHT = {
    "not_null": 1.2,
    "pattern": 0.7,
    "functional_dependency": 1.05,
    "soft_functional_dependency": 0.75,
    "rare_value": 0.20,
    "rare_pattern": 0.25,
    "global_dominant_value": 0.35,
    "dominant_value_by_context": 0.45,
    "dominant_value_by_context_pair": 0.55,
    "high_risk_missing": 0.9,
    "paired_missing_consistency": 0.8,

    "movie_year_range": 1.15,
    "release_date_format": 1.10,
    "release_year_consistency": 1.25,
    "duration_format": 1.10,
    "rating_value_range": 1.15,
    "rating_count_format": 1.00,
    "review_count_format": 0.95,
    "comma_list_format": 0.85,
    "people_list_format": 0.90,
    "mojibake_or_control_char": 1.35,

    "actors_cast_overlap": 1.15,
    "rating_value_count_coexistence": 0.90,
    "review_rating_count_order": 0.85,
}

MOVIE_YEAR_MIN = 1880
MOVIE_YEAR_MAX = 2035
RATING_MIN = 0.0
RATING_MAX = 10.0
DURATION_MIN_MINUTES = 1
DURATION_MAX_MINUTES = 600
MIN_LIST_TOKEN_COUNT = 1


# ============================================================
# Precision guard：movies 长尾字段规则过滤
# ============================================================
# 这些列不再用 rare_value / rare_pattern 触发候选。
UNSAFE_RARE_COLUMNS = {
    "Name", "Director", "Creator", "Actors", "Cast", "Filming Locations", "Description",
    "Release Date", "ReviewCount", "Language", "Country", "Genre",
}

# 这些列不再用 FD / soft FD 作为候选触发依据；保留格式、缺失和电影一致性规则。
UNSAFE_FD_RHS_COLUMNS = {
    "Name", "Director", "Creator", "Actors", "Cast", "Filming Locations", "Description",
    "Release Date", "ReviewCount", "Language", "Country", "Genre", "Year",
}

# 这些列不再用 context dominant 触发候选。
UNSAFE_CONTEXT_RHS_COLUMNS = {
    "Name", "Director", "Creator", "Actors", "Cast", "Filming Locations", "Description",
    "Release Date", "ReviewCount", "Language", "Country", "Genre", "Year", "RatingValue",
}

SAFE_MOVIE_RULE_TYPES = {
    "not_null", "movie_year_range", "release_date_format", "release_year_consistency",
    "duration_format", "rating_value_range", "rating_count_format", "review_count_format",
    "comma_list_format", "people_list_format", "mojibake_or_control_char",
    "actors_cast_overlap", "rating_value_count_coexistence", "review_rating_count_order",
    "paired_missing_consistency",
}


def precision_guard_keep_rule(rule: Dict[str, Any]) -> bool:
    """过滤掉 movies 中高误报的弱规则。"""
    rtype = rule.get("rule_type")
    col = rule.get("column")
    rhs = rule.get("rhs")

    if rtype in {"rare_value", "rare_pattern"} and col in UNSAFE_RARE_COLUMNS:
        return False

    if rtype in {"functional_dependency", "soft_functional_dependency"} and rhs in UNSAFE_FD_RHS_COLUMNS:
        return False

    if rtype in {"dominant_value_by_context", "dominant_value_by_context_pair"} and rhs in UNSAFE_CONTEXT_RHS_COLUMNS:
        return False

    if rtype == "global_dominant_value" and col in UNSAFE_RARE_COLUMNS:
        return False

    return True


def filter_rules_with_precision_guard(rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    before = len(rules)
    filtered = [r for r in rules if precision_guard_keep_rule(r)]
    removed = before - len(filtered)
    print(f"[PrecisionGuard] rules before={before}, after={len(filtered)}, removed={removed}")
    return filtered


# ============================================================
# 2. 基础工具函数
# ============================================================

def normalize_value(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.lower() in MISSING_TOKENS:
        return None
    return s.lower()


def normalize_series(s):
    return s.map(normalize_value)


def maybe_drop_unnamed_columns(df):
    keep_cols = [c for c in df.columns if not str(c).lower().startswith("unnamed:")]
    return df[keep_cols].copy()


def detect_basic_pattern(value):
    if pd.isna(value) or value is None:
        return "NULL"

    parts, prev_type, cnt = [], None, 0
    for ch in value:
        if ch.isdigit():
            cur = "D"
        elif ch.isalpha():
            cur = "L"
        elif ch in {" ", "-", "_", "/", ".", ",", "(", ")", "%", ":", ";", "&", "'", "+", "[", "]", "{", "}", '"'}:
            cur = "S"
        else:
            cur = "X"

        if cur == prev_type:
            cnt += 1
        else:
            if prev_type is not None:
                parts.append(f"{prev_type}{cnt}")
            prev_type = cur
            cnt = 1

    if prev_type is not None:
        parts.append(f"{prev_type}{cnt}")

    return "".join(parts)


def load_rule_pool(rule_json_path):
    with open(rule_json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    all_rules = []
    for _, rules in obj.get("rules", {}).items():
        for rule in rules:
            all_rules.append(dict(rule))
    return all_rules


def rule_weight(rule):
    conf = float(rule.get("confidence", 0.0))
    supp = int(rule.get("support", 0))
    priority = rule.get("priority", "low")
    rule_type = rule.get("rule_type", "")

    priority_w = PRIORITY_WEIGHT.get(priority, 0.6)
    type_w = RULE_TYPE_WEIGHT.get(rule_type, 1.0)
    support_part = math.log1p(max(supp, 1))

    return round((0.1 + conf) * support_part * priority_w * type_w, 6)


# ============================================================
# 3. movies 专用解析函数
# ============================================================

def split_comma_list(s: Optional[str]) -> List[str]:
    if s is None:
        return []
    txt = str(s).strip()
    if txt == "":
        return []
    parts = [p.strip().lower() for p in txt.split(",")]
    parts = [p for p in parts if p and p not in MISSING_TOKENS]
    return parts


def parse_year_value(s: Optional[str]) -> Optional[int]:
    if s is None:
        return None

    txt = str(s).strip().lower()
    m = re.search(r"\b(18\d{2}|19\d{2}|20\d{2})\b", txt)
    if not m:
        return None

    year = int(m.group(1))
    if MOVIE_YEAR_MIN <= year <= MOVIE_YEAR_MAX:
        return year
    return None


def parse_release_date(s: Optional[str]) -> Optional[Dict[str, Any]]:
    if s is None:
        return None

    txt = str(s).strip().lower()
    if txt == "":
        return None

    country = None
    m_country = re.search(r"\(([^)]+)\)", txt)
    if m_country:
        country = m_country.group(1).strip().lower()

    year = parse_year_value(txt)
    if year is None:
        return None

    months = {
        "january": 1, "jan": 1,
        "february": 2, "feb": 2,
        "march": 3, "mar": 3,
        "april": 4, "apr": 4,
        "may": 5,
        "june": 6, "jun": 6,
        "july": 7, "jul": 7,
        "august": 8, "aug": 8,
        "september": 9, "sep": 9, "sept": 9,
        "october": 10, "oct": 10,
        "november": 11, "nov": 11,
        "december": 12, "dec": 12,
    }

    day = None
    month = None

    m_full = re.search(
        r"\b(\d{1,2})\s+([a-z]+)\s+(18\d{2}|19\d{2}|20\d{2})\b",
        txt
    )
    if m_full:
        day = int(m_full.group(1))
        month = months.get(m_full.group(2))
    else:
        m_month_year = re.search(r"\b([a-z]+)\s+(18\d{2}|19\d{2}|20\d{2})\b", txt)
        if m_month_year:
            month = months.get(m_month_year.group(1))

    if day is not None and not (1 <= day <= 31):
        return None
    if month is not None and not (1 <= month <= 12):
        return None

    return {
        "year": year,
        "month": month,
        "day": day,
        "country": country,
    }


def parse_duration_minutes(s: Optional[str]) -> Optional[int]:
    if s is None:
        return None

    txt = str(s).strip().lower()
    m = re.search(r"\b(\d{1,4})\s*(min|mins|minute|minutes)\b", txt)
    if m:
        mins = int(m.group(1))
    else:
        m2 = re.fullmatch(r"\d{1,4}", txt)
        if not m2:
            return None
        mins = int(txt)

    if DURATION_MIN_MINUTES <= mins <= DURATION_MAX_MINUTES:
        return mins
    return None


def parse_rating_value(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None

    txt = str(s).strip().lower().replace(",", "")
    try:
        val = float(txt)
    except Exception:
        return None

    if RATING_MIN <= val <= RATING_MAX:
        return val
    return None


def parse_count_value(s: Optional[str]) -> Optional[int]:
    if s is None:
        return None

    txt = str(s).strip().lower().replace(",", "")
    m = re.search(r"\b(\d+)\b", txt)
    if not m:
        return None

    val = int(m.group(1))
    if val >= 0:
        return val
    return None


def parse_review_count(s: Optional[str]) -> Optional[Dict[str, int]]:
    if s is None:
        return None

    txt = str(s).strip().lower()
    if txt == "":
        return None

    user_count = None
    critic_count = None

    m_user = re.search(r"([\d,]+)\s*user", txt)
    if m_user:
        user_count = int(m_user.group(1).replace(",", ""))

    m_critic = re.search(r"([\d,]+)\s*critic", txt)
    if m_critic:
        critic_count = int(m_critic.group(1).replace(",", ""))

    if user_count is None and critic_count is None:
        return None

    return {
        "user": user_count if user_count is not None else 0,
        "critic": critic_count if critic_count is not None else 0,
    }


def looks_like_person_or_people_list(s: Optional[str]) -> bool:
    if s is None:
        return False

    parts = split_comma_list(s)
    if len(parts) < MIN_LIST_TOKEN_COUNT:
        return False

    return any(re.search(r"[a-z]", p) is not None for p in parts)


def looks_like_category_list(s: Optional[str]) -> bool:
    if s is None:
        return False

    parts = split_comma_list(s)
    if len(parts) < MIN_LIST_TOKEN_COUNT:
        return False

    return all(re.search(r"[a-z]", p) is not None for p in parts)


def has_mojibake_or_control_chars(s: Optional[str]) -> bool:
    if s is None:
        return False

    txt = str(s)
    if "�" in txt or "\ufffd" in txt or "��" in txt:
        return True

    for ch in txt:
        if ord(ch) < 32 and ch not in {"\t", "\n", "\r"}:
            return True

    return False


def total_review_count(review_count_str: Optional[str]) -> Optional[int]:
    parsed = parse_review_count(review_count_str)
    if parsed is None:
        return None
    return int(parsed.get("user", 0)) + int(parsed.get("critic", 0))


# ============================================================
# 4. runtime mapping 准备
# ============================================================

def build_fd_majority_mapping(df, lhs, rhs):
    sub = df[lhs + [rhs]].dropna()
    if len(sub) == 0:
        return {}

    grouped = sub.groupby(lhs, dropna=True)
    mapping = {}

    for group_key, grp in grouped:
        rhs_vals = grp[rhs].tolist()
        cnt = Counter(rhs_vals)
        majority_val, majority_cnt = cnt.most_common(1)[0]
        total_cnt = len(rhs_vals)

        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        mapping[group_key] = (majority_val, majority_cnt, total_cnt)

        if len(mapping) >= MAX_FD_MAPPING_SIZE:
            break

    return mapping


def prepare_fd_rule_runtime(df, rules):
    out = []
    for rule in rules:
        if rule.get("rule_type") not in {"functional_dependency", "soft_functional_dependency"}:
            continue

        lhs = rule.get("lhs", [])
        rhs = rule.get("rhs")

        if not rhs or any(c not in df.columns for c in lhs + [rhs]):
            continue

        r = dict(rule)
        r["_majority_mapping"] = build_fd_majority_mapping(df, lhs, rhs)
        out.append(r)

    return out


def prepare_context_dominant_rule_runtime(rules):
    out = []
    for rule in rules:
        if rule.get("rule_type") != "dominant_value_by_context":
            continue

        mapping = {}
        for ctx_val, info in rule.get("majority_map", {}).items():
            mapping[ctx_val] = (
                info.get("dominant_value"),
                int(info.get("group_size", 0)),
                float(info.get("ratio", 0.0)),
            )

        r = dict(rule)
        r["_context_mapping"] = mapping
        out.append(r)

    return out


def prepare_pair_context_rule_runtime(rules):
    out = []
    for rule in rules:
        if rule.get("rule_type") != "dominant_value_by_context_pair":
            continue

        mapping = {}
        for key, info in rule.get("majority_map", {}).items():
            mapping[key] = (
                info.get("dominant_value"),
                int(info.get("group_size", 0)),
                float(info.get("ratio", 0.0)),
            )

        r = dict(rule)
        r["_pair_context_mapping"] = mapping
        out.append(r)

    return out


# ============================================================
# 5. 添加候选 violation
# ============================================================

def add_violation(cell_map, violation):
    row_id = int(violation["row_id"])
    column = violation["column"]
    key = (row_id, column)
    rule = violation["rule"]

    info = {
        "rule_id": rule.get("rule_id"),
        "rule_type": rule.get("rule_type"),
        "description": rule.get("description"),
        "confidence": rule.get("confidence"),
        "support": rule.get("support"),
        "priority": rule.get("priority"),
        "usage_role": rule.get("usage_role"),
        "reason": violation.get("reason"),
    }

    if "expected_value" in violation:
        info["expected_value"] = violation["expected_value"]

    if key not in cell_map:
        cell_map[key] = {
            "row_id": row_id,
            "column": column,
            "value": violation.get("value"),
            "violated_rules": [],
            "conflict_score": 0.0,
        }

    cell_map[key]["violated_rules"].append(info)
    cell_map[key]["conflict_score"] += rule_weight(rule)


# ============================================================
# 6. 通用规则检查
# ============================================================

def check_not_null_rule(row_idx, row, rule):
    col = rule["column"]
    val = row.get(col)

    if pd.isna(val) or val is None:
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": "value is null but rule requires non-null",
        }

    return None


def check_pattern_rule(row_idx, row, rule):
    col = rule["column"]
    val = row.get(col)

    if (pd.isna(val) or val is None) and SKIP_NULL_FOR_NON_NOTNULL_RULES:
        return None

    observed = detect_basic_pattern(val)
    target = rule["pattern"]

    if observed != target:
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": f"pattern mismatch: observed={observed}, expected={target}",
        }

    return None


def check_fd_like_rule(row_idx, row, rule):
    lhs = rule["lhs"]
    rhs = rule["rhs"]

    lhs_values = []
    for c in lhs:
        v = row.get(c)
        if v is None:
            return None
        lhs_values.append(v)

    rhs_val = row.get(rhs)

    if rhs_val is None and SKIP_NULL_FOR_NON_NOTNULL_RULES:
        return None

    lhs_key = tuple(lhs_values)
    mapping = rule.get("_majority_mapping", {})

    if lhs_key not in mapping:
        return None

    expected_rhs, majority_cnt, total_cnt = mapping[lhs_key]

    if rhs_val != expected_rhs:
        return {
            "row_id": row_idx,
            "column": rhs,
            "value": rhs_val,
            "rule": rule,
            "reason": f"{rule['rule_type']} violation: lhs={lhs_key} suggests rhs={expected_rhs}, observed={rhs_val}",
            "expected_value": expected_rhs,
        }

    return None


def check_rare_value_rule(row_idx, row, rule):
    col = rule["column"]
    val = row.get(col)

    if pd.isna(val) or val is None:
        return None

    if val in set(rule.get("rare_values", [])):
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": "value is in rare_values set",
        }

    return None


def check_rare_pattern_rule(row_idx, row, rule):
    col = rule["column"]
    val = row.get(col)

    if pd.isna(val) or val is None:
        return None

    observed = detect_basic_pattern(val)

    if observed in set(rule.get("rare_patterns", [])):
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": f"pattern {observed} is rare for this column",
        }

    return None


def check_global_dominant_rule(row_idx, row, rule):
    col = rule["column"]
    val = row.get(col)

    if pd.isna(val) or val is None:
        return None

    dominant = rule.get("dominant_value")

    if dominant is not None and val != dominant:
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": f"value differs from global dominant value: dominant={dominant}, observed={val}",
            "expected_value": dominant,
        }

    return None


def check_context_dominant_rule(row_idx, row, rule):
    lhs = rule.get("lhs", [])
    rhs = rule.get("rhs")

    if len(lhs) != 1 or rhs is None:
        return None

    ctx_col = lhs[0]
    ctx_val = row.get(ctx_col)
    tgt_val = row.get(rhs)

    if ctx_val is None or tgt_val is None:
        return None

    mapping = rule.get("_context_mapping", {})

    if ctx_val not in mapping:
        return None

    dominant_val, support, dominant_ratio = mapping[ctx_val]

    if tgt_val != dominant_val:
        return {
            "row_id": row_idx,
            "column": rhs,
            "value": tgt_val,
            "rule": rule,
            "reason": f"context dominant mismatch: {ctx_col}={ctx_val} suggests {rhs}={dominant_val}, observed={tgt_val}",
            "expected_value": dominant_val,
        }

    return None


def check_pair_context_rule(row_idx, row, rule):
    lhs = rule.get("lhs", [])
    rhs = rule.get("rhs")

    if len(lhs) < 2 or rhs is None:
        return None

    vals = []
    for c in lhs:
        v = row.get(c)
        if v is None:
            return None
        vals.append(v)

    tgt_val = row.get(rhs)
    if tgt_val is None:
        return None

    key = "||".join(map(str, vals))
    mapping = rule.get("_pair_context_mapping", {})

    if key not in mapping:
        return None

    dominant_val, support, dominant_ratio = mapping[key]

    if tgt_val != dominant_val:
        return {
            "row_id": row_idx,
            "column": rhs,
            "value": tgt_val,
            "rule": rule,
            "reason": f"context pair dominant mismatch: {lhs}={vals} suggests {rhs}={dominant_val}, observed={tgt_val}",
            "expected_value": dominant_val,
        }

    return None


def check_high_risk_missing_rule(row_idx, row, rule):
    col = rule["column"]
    val = row.get(col)

    if pd.isna(val) or val is None:
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": "value is null under high_risk_missing rule",
        }

    return None


def check_paired_missing_consistency_rule(row_idx, row, rule):
    cols = rule.get("columns", [])

    if len(cols) != 2:
        return None

    a_col, b_col = cols
    a = row.get(a_col)
    b = row.get(b_col)

    if (a is None and b is not None) or (a is not None and b is None):
        bad_col = a_col if a is None else b_col

        return {
            "row_id": row_idx,
            "column": bad_col,
            "value": row.get(bad_col),
            "rule": rule,
            "reason": f"paired missing inconsistency: {a_col}={a}, {b_col}={b}",
        }

    return None


# ============================================================
# 7. movies typed rules 检查
# ============================================================

def check_movie_year_range_rule(row_idx, row, rule):
    col = rule["column"]
    val = row.get(col)

    if val is None:
        return None

    year = parse_year_value(val)

    if year is None:
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": f"Year is not parseable or outside range [{rule.get('min_year')}, {rule.get('max_year')}]: {val}",
        }

    min_year = int(rule.get("min_year", MOVIE_YEAR_MIN))
    max_year = int(rule.get("max_year", MOVIE_YEAR_MAX))

    if year < min_year or year > max_year:
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": f"Year out of range: {year}, expected in [{min_year}, {max_year}]",
        }

    return None


def check_release_date_format_rule(row_idx, row, rule):
    col = rule["column"]
    val = row.get(col)

    if val is None:
        return None

    parsed = parse_release_date(val)

    if parsed is None:
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": f"Release Date is not parseable as movie release date: {val}",
        }

    return None


def check_release_year_consistency_rule(row_idx, row, rule):
    year_col = rule.get("year_column", "Year")
    date_col = rule.get("date_column", "Release Date")

    if year_col not in row.index or date_col not in row.index:
        return None

    year_val = row.get(year_col)
    date_val = row.get(date_col)

    if year_val is None or date_val is None:
        return None

    year = parse_year_value(year_val)
    parsed_date = parse_release_date(date_val)

    if year is None or parsed_date is None:
        return None

    release_year = parsed_date.get("year")
    allowed_gap = int(rule.get("allowed_year_gap", 2))

    if abs(int(release_year) - int(year)) > allowed_gap:
        return {
            "row_id": row_idx,
            "column": date_col,
            "value": date_val,
            "rule": rule,
            "reason": f"release-year inconsistency: Year={year}, Release Date year={release_year}, allowed_gap={allowed_gap}",
            "expected_value": str(year),
        }

    return None


def check_duration_format_rule(row_idx, row, rule):
    col = rule["column"]
    val = row.get(col)

    if val is None:
        return None

    mins = parse_duration_minutes(val)

    if mins is None:
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": f"Duration is not parseable as '<minutes> min' or outside broad range: {val}",
        }

    min_minutes = int(rule.get("min_minutes", DURATION_MIN_MINUTES))
    max_minutes = int(rule.get("max_minutes", DURATION_MAX_MINUTES))

    if mins < min_minutes or mins > max_minutes:
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": f"Duration out of range: {mins}, expected in [{min_minutes}, {max_minutes}]",
        }

    return None


def check_rating_value_range_rule(row_idx, row, rule):
    col = rule["column"]
    val = row.get(col)

    if val is None:
        return None

    rating = parse_rating_value(val)

    if rating is None:
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": f"RatingValue is not numeric in [0, 10]: {val}",
        }

    return None


def check_rating_count_format_rule(row_idx, row, rule):
    col = rule["column"]
    val = row.get(col)

    if val is None:
        return None

    count = parse_count_value(val)

    if count is None:
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": f"RatingCount is not parseable as non-negative integer: {val}",
        }

    return None


def check_review_count_format_rule(row_idx, row, rule):
    col = rule["column"]
    val = row.get(col)

    if val is None:
        return None

    parsed = parse_review_count(val)

    if parsed is None:
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": f"ReviewCount is not parseable as user/critic review counts: {val}",
        }

    return None


def check_comma_list_format_rule(row_idx, row, rule):
    col = rule["column"]
    val = row.get(col)

    if val is None:
        return None

    if not looks_like_category_list(val):
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": f"{col} is not parseable as comma-separated list: {val}",
        }

    return None


def check_people_list_format_rule(row_idx, row, rule):
    col = rule["column"]
    val = row.get(col)

    if val is None:
        return None

    if not looks_like_person_or_people_list(val):
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": f"{col} is not parseable as people/person list: {val}",
        }

    return None


def check_mojibake_or_control_char_rule(row_idx, row, rule):
    col = rule["column"]
    val = row.get(col)

    if val is None:
        return None

    if has_mojibake_or_control_chars(val):
        return {
            "row_id": row_idx,
            "column": col,
            "value": val,
            "rule": rule,
            "reason": f"{col} contains mojibake or control characters",
        }

    return None


# ============================================================
# 8. movies consistency rules 检查
# ============================================================

def check_actors_cast_overlap_rule(row_idx, row, rule):
    actors_col = rule.get("actors_column", "Actors")
    cast_col = rule.get("cast_column", "Cast")

    actors_val = row.get(actors_col)
    cast_val = row.get(cast_col)

    if actors_val is None or cast_val is None:
        return None

    actors = set(split_comma_list(actors_val))
    cast = set(split_comma_list(cast_val))

    if not actors or not cast:
        return None

    ratio = len(actors & cast) / max(1, len(actors))
    min_ratio = float(rule.get("min_overlap_ratio", 0.5))

    if ratio < min_ratio:
        return {
            "row_id": row_idx,
            "column": cast_col,
            "value": cast_val,
            "rule": rule,
            "reason": f"Actors-Cast overlap too low: overlap_ratio={ratio:.4f}, expected >= {min_ratio}",
            "expected_value": actors_val,
        }

    return None


def check_rating_value_count_coexistence_rule(row_idx, row, rule):
    rating_col = rule.get("rating_value_column", "RatingValue")
    count_col = rule.get("rating_count_column", "RatingCount")

    rv = row.get(rating_col)
    rc = row.get(count_col)

    rv_missing = rv is None
    rc_missing = rc is None

    if rv_missing and not rc_missing:
        return {
            "row_id": row_idx,
            "column": rating_col,
            "value": rv,
            "rule": rule,
            "reason": f"RatingValue missing but RatingCount exists: RatingCount={rc}",
        }

    if rc_missing and not rv_missing:
        return {
            "row_id": row_idx,
            "column": count_col,
            "value": rc,
            "rule": rule,
            "reason": f"RatingCount missing but RatingValue exists: RatingValue={rv}",
        }

    return None


def check_review_rating_count_order_rule(row_idx, row, rule):
    review_col = rule.get("review_count_column", "ReviewCount")
    rating_col = rule.get("rating_count_column", "RatingCount")

    review_val = row.get(review_col)
    rating_val = row.get(rating_col)

    if review_val is None or rating_val is None:
        return None

    review_total = total_review_count(review_val)
    rating_count = parse_count_value(rating_val)

    if review_total is None or rating_count is None or rating_count <= 0:
        return None

    max_ratio = float(rule.get("max_review_to_rating_ratio", 0.50))
    ratio = review_total / rating_count

    if ratio > max_ratio:
        return {
            "row_id": row_idx,
            "column": review_col,
            "value": review_val,
            "rule": rule,
            "reason": f"ReviewCount too large relative to RatingCount: review_total={review_total}, rating_count={rating_count}, ratio={ratio:.4f}",
            "expected_value": f"review_total <= {max_ratio} * rating_count",
        }

    return None


# ============================================================
# 9. 主流程
# ============================================================

def shrink_search_space(input_csv, rule_json, output_jsonl, output_csv):
    df = pd.read_csv(input_csv, dtype=str)
    df = maybe_drop_unnamed_columns(df)

    for col in df.columns:
        df[col] = normalize_series(df[col])

    rules = load_rule_pool(rule_json)
    rules = filter_rules_with_precision_guard(rules)

    not_null_rules = [r for r in rules if r.get("rule_type") == "not_null"]
    pattern_rules = [r for r in rules if r.get("rule_type") == "pattern"]

    fd_like_rules = prepare_fd_rule_runtime(df, rules)

    rare_value_rules = [r for r in rules if r.get("rule_type") == "rare_value"]
    rare_pattern_rules = [r for r in rules if r.get("rule_type") == "rare_pattern"]
    global_dominant_rules = [r for r in rules if r.get("rule_type") == "global_dominant_value"]

    context_dominant_rules = prepare_context_dominant_rule_runtime(rules)
    pair_context_rules = prepare_pair_context_rule_runtime(rules)

    high_risk_missing_rules = [r for r in rules if r.get("rule_type") == "high_risk_missing"]
    paired_missing_rules = [r for r in rules if r.get("rule_type") == "paired_missing_consistency"]

    movie_year_rules = [r for r in rules if r.get("rule_type") == "movie_year_range"]
    release_date_rules = [r for r in rules if r.get("rule_type") == "release_date_format"]
    release_year_consistency_rules = [r for r in rules if r.get("rule_type") == "release_year_consistency"]

    duration_rules = [r for r in rules if r.get("rule_type") == "duration_format"]
    rating_value_rules = [r for r in rules if r.get("rule_type") == "rating_value_range"]
    rating_count_rules = [r for r in rules if r.get("rule_type") == "rating_count_format"]
    review_count_rules = [r for r in rules if r.get("rule_type") == "review_count_format"]

    comma_list_rules = [r for r in rules if r.get("rule_type") == "comma_list_format"]
    people_list_rules = [r for r in rules if r.get("rule_type") == "people_list_format"]
    mojibake_rules = [r for r in rules if r.get("rule_type") == "mojibake_or_control_char"]

    actors_cast_rules = [r for r in rules if r.get("rule_type") == "actors_cast_overlap"]
    rating_coexist_rules = [r for r in rules if r.get("rule_type") == "rating_value_count_coexistence"]
    review_rating_order_rules = [r for r in rules if r.get("rule_type") == "review_rating_count_order"]

    cell_map = {}

    check_groups = [
        (not_null_rules, check_not_null_rule),
        (pattern_rules, check_pattern_rule),

        (fd_like_rules, check_fd_like_rule),
        (rare_value_rules, check_rare_value_rule),
        (rare_pattern_rules, check_rare_pattern_rule),
        (global_dominant_rules, check_global_dominant_rule),
        (context_dominant_rules, check_context_dominant_rule),
        (pair_context_rules, check_pair_context_rule),
        (high_risk_missing_rules, check_high_risk_missing_rule),
        (paired_missing_rules, check_paired_missing_consistency_rule),

        (movie_year_rules, check_movie_year_range_rule),
        (release_date_rules, check_release_date_format_rule),
        (release_year_consistency_rules, check_release_year_consistency_rule),

        (duration_rules, check_duration_format_rule),
        (rating_value_rules, check_rating_value_range_rule),
        (rating_count_rules, check_rating_count_format_rule),
        (review_count_rules, check_review_count_format_rule),

        (comma_list_rules, check_comma_list_format_rule),
        (people_list_rules, check_people_list_format_rule),
        (mojibake_rules, check_mojibake_or_control_char_rule),

        (actors_cast_rules, check_actors_cast_overlap_rule),
        (rating_coexist_rules, check_rating_value_count_coexistence_rule),
        (review_rating_order_rules, check_review_rating_count_order_rule),
    ]

    for row_idx, row in df.iterrows():
        for group, fn in check_groups:
            for rule in group:
                v = fn(row_idx, row, rule)
                if v is not None:
                    add_violation(cell_map, v)

    candidates = list(cell_map.values())

    for item in candidates:
        item["conflict_score"] = round(item["conflict_score"], 6)
        item["violation_count"] = len(item["violated_rules"])

    candidates.sort(key=lambda x: (x["conflict_score"], x["violation_count"]), reverse=True)

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for item in candidates:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    csv_rows = []
    for item in candidates:
        csv_rows.append({
            "row_id": item["row_id"],
            "column": item["column"],
            "value": item["value"],
            "violation_count": item["violation_count"],
            "conflict_score": item["conflict_score"],
            "rule_ids": "|".join([str(r["rule_id"]) for r in item["violated_rules"]]),
            "rule_types": "|".join([str(r["rule_type"]) for r in item["violated_rules"]]),
            "priorities": "|".join([str(r["priority"]) for r in item["violated_rules"]]),
            "usage_roles": "|".join([str(r.get("usage_role")) for r in item["violated_rules"]]),
            "reasons": " || ".join([str(r["reason"]) for r in item["violated_rules"]]),
        })

    pd.DataFrame(csv_rows).to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"[OK] Candidate cells saved to: {output_jsonl}")
    print(f"[OK] Candidate cells CSV saved to: {output_csv}")
    print(f"Total candidate cells: {len(candidates)}")
    total_cells = len(df) * len(df.columns)
    print(f"Original cells: {total_cells}")
    print(f"Shrink ratio: {len(candidates) / total_cells:.6f}")


if __name__ == "__main__":
    shrink_search_space(INPUT_CSV, INPUT_RULE_JSON, OUTPUT_JSONL, OUTPUT_CSV)

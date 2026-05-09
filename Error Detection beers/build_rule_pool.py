import json
import math
import re
from collections import Counter
from itertools import combinations
from typing import Dict, List, Any, Optional

import pandas as pd

INPUT_CSV = "/mnt/mydata/dq/projects/Splittree/beers/beers_dirty.csv"
OUTPUT_JSON = "rule_pool_beers.json"

MISSING_TOKENS = {
    "", "empty", "null", "none", "nan", "na", "n/a", "missing", "unk",
    "not available", "contact airline"
}

NUMERIC_RATIO_THRESHOLD = 0.90
ENUM_UNIQUE_THRESHOLD = 40
ENUM_RATIO_THRESHOLD = 0.08

ID_LIKE_UNIQUE_RATIO_THRESHOLD = 0.98
CODE_LIKE_AVG_LEN_THRESHOLD = 20
CODE_LIKE_DOMINANT_PATTERN_THRESHOLD = 0.70
DOMINANT_PATTERN_THRESHOLD = 0.70
DOMINANT_TYPO_MIN_RATIO = 0.50

MAX_LHS_SIZE = 2
FD_CONFIDENCE_THRESHOLD = 0.98
MIN_SUPPORT_ROWS = 5

SOFT_FD_CONFIDENCE_THRESHOLD = 0.80
SOFT_FD_MIN_GROUP_SIZE = 3
PAIR_CONTEXT_MIN_RATIO = 0.70
PAIR_CONTEXT_MIN_GROUP_SIZE = 2

HIGH_CARDINALITY_RATIO_FOR_LHS = 0.95
ALLOW_INDEX_COLUMN = False

TARGET_CANDIDATE_COLUMNS = {
    "beer_name", "style", "ounces", "abv", "ibu", "brewery_name", "city", "state"
}
HIGH_RISK_MISSING_COLUMNS = {"abv", "ibu", "style", "ounces", "brewery_name", "city", "state"}

ENABLE_RARE_VALUE_RULE = True
RARE_VALUE_MAX_FREQ = 2
ENABLE_RARE_PATTERN_RULE = True
RARE_PATTERN_MAX_RATIO = 0.02

# 这些列天然长尾，不能再用 rare/global-dominant 这类规则硬打候选
SKIP_RARE_VALUE_COLUMNS = {"beer_name", "brewery_name"}
SKIP_RARE_PATTERN_COLUMNS = {"beer_name"}
SKIP_GLOBAL_DOMINANT_COLUMNS = {"beer_name", "brewery_name", "style"}


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
    if value is None:
        return "NULL"
    parts, prev_type, cnt = [], None, 0
    for ch in value:
        if ch.isdigit():
            cur = "D"
        elif ch.isalpha():
            cur = "L"
        elif ch in {" ", "-", "_", "/", ".", ",", "(", ")", "%", ":", ";", "&", "'", "+"}:
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


def try_parse_numeric(series):
    cleaned = (
        series.dropna().astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
    )
    numeric = pd.to_numeric(cleaned, errors="coerce")
    success_ratio = numeric.notna().mean() if len(cleaned) > 0 else 0.0
    return numeric, success_ratio


def shannon_entropy(series):
    vals = series.dropna().tolist()
    if not vals:
        return 0.0
    cnt = Counter(vals)
    total = len(vals)
    ent = 0.0
    for c in cnt.values():
        p = c / total
        ent -= p * math.log2(p)
    return ent


def value_frequency_dict(series, topk=20):
    cnt = Counter(series.dropna().tolist())
    total = sum(cnt.values()) if cnt else 0
    out = []
    for k, v in cnt.most_common(topk):
        out.append({"value": k, "count": v, "ratio": round(v / total, 6) if total > 0 else 0.0})
    return out


def parse_abv_like(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    s = str(s).strip().lower().replace("%", "")
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def parse_ibu_like(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    s = str(s).strip().lower()
    if s in MISSING_TOKENS:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def parse_ounces_like(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    s = str(s).strip().lower()
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def canonicalize_ounces_value(s: Optional[str]) -> Optional[str]:
    """
    beers_clean 中 ounces 是纯数字（如 12），而 dirty 里常是 12.0 oz / 12.0 ounce 等。
    这里统一规范成 clean 风格的字符串。
    """
    num = parse_ounces_like(s)
    if num is None:
        return None
    if abs(num - round(num)) < 1e-9:
        return str(int(round(num)))
    # 很少用到，但保底
    txt = f"{num:.6f}".rstrip("0").rstrip(".")
    return txt


def canonicalize_abv_value(s: Optional[str]) -> Optional[str]:
    """
    beers_clean 中 abv 通常是纯数字字符串，不带百分号。
    """
    num = parse_abv_like(s)
    if num is None:
        return None
    txt = f"{num:.15f}".rstrip("0").rstrip(".")
    return txt


def looks_like_city_with_state_suffix(s: Optional[str]) -> bool:
    if s is None:
        return False
    s = str(s).strip().lower()
    # 如 "san diego ca" / "portland or"
    return re.search(r"\b[a-z]{2}$", s) is not None and len(s.split()) >= 2


def strip_city_state_suffix(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = str(s).strip().lower()
    s2 = re.sub(r"\s+[a-z]{2}$", "", s).strip()
    return s2 if s2 != s else None


def is_index_like_column(series):
    s = series.dropna()
    if len(s) == 0:
        return False
    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.notna().mean() < 0.95:
        return False
    unique_ratio = s.nunique() / len(s)
    return unique_ratio > 0.98


def is_id_like_column(non_null, unique_ratio, numeric_ratio, avg_len):
    if len(non_null) == 0:
        return False
    if unique_ratio >= ID_LIKE_UNIQUE_RATIO_THRESHOLD and avg_len <= CODE_LIKE_AVG_LEN_THRESHOLD:
        return True
    if unique_ratio >= ID_LIKE_UNIQUE_RATIO_THRESHOLD and numeric_ratio >= NUMERIC_RATIO_THRESHOLD:
        return True
    return False


def is_code_like_column(non_null, unique_ratio, numeric_ratio, avg_len, top_patterns):
    if len(non_null) == 0:
        return False
    dominant_ratio = top_patterns[0]["ratio"] if top_patterns else 0.0
    if avg_len <= CODE_LIKE_AVG_LEN_THRESHOLD and dominant_ratio >= CODE_LIKE_DOMINANT_PATTERN_THRESHOLD:
        if unique_ratio > ENUM_RATIO_THRESHOLD:
            return True
    if numeric_ratio >= 0.95 and avg_len <= 15 and unique_ratio < ID_LIKE_UNIQUE_RATIO_THRESHOLD:
        return True
    return False


def calc_fd_confidence_and_support(df, lhs_cols, rhs_col):
    sub = df[lhs_cols + [rhs_col]].dropna()
    support = len(sub)
    if support < MIN_SUPPORT_ROWS:
        return 0.0, support
    grouped = sub.groupby(lhs_cols, dropna=True)[rhs_col]
    majority_sum = 0
    for _, grp in grouped:
        cnt = Counter(grp.tolist())
        majority_sum += cnt.most_common(1)[0][1]
    confidence = majority_sum / support if support > 0 else 0.0
    return round(confidence, 6), int(support)


def calc_soft_fd_confidence_and_support(df, lhs_cols, rhs_col):
    sub = df[lhs_cols + [rhs_col]].dropna()
    support = len(sub)
    if support < SOFT_FD_MIN_GROUP_SIZE:
        return 0.0, support
    grouped = sub.groupby(lhs_cols, dropna=True)[rhs_col]
    majority_sum = 0
    for _, grp in grouped:
        if len(grp) < SOFT_FD_MIN_GROUP_SIZE:
            continue
        cnt = Counter(grp.tolist())
        majority_sum += cnt.most_common(1)[0][1]
    confidence = majority_sum / support if support > 0 else 0.0
    return round(confidence, 6), int(support)


def calc_not_null_confidence_and_support(series):
    total = len(series)
    if total == 0:
        return 0.0, 0
    non_null = series.notna().sum()
    return round(non_null / total, 6), int(total)


def calc_pattern_confidence_and_support(series, dominant_pattern):
    non_null = series.dropna()
    support = len(non_null)
    if support == 0:
        return 0.0, 0
    pats = non_null.map(detect_basic_pattern)
    ok = (pats == dominant_pattern).sum()
    return round(ok / support, 6), int(support)


def analyze_columns(df):
    profiles = {}
    n = len(df)
    for col in df.columns:
        s = normalize_series(df[col])
        non_null = s.dropna()
        missing_rate = 1.0 - (len(non_null) / n if n > 0 else 0.0)
        unique_count = non_null.nunique()
        unique_ratio = unique_count / len(non_null) if len(non_null) > 0 else 0.0
        numeric_vals, numeric_ratio = try_parse_numeric(non_null)
        patterns = non_null.map(detect_basic_pattern)
        pattern_counter = Counter(patterns.tolist())
        total_patterns = sum(pattern_counter.values())
        top_patterns = [{"pattern": p, "count": c, "ratio": round(c / total_patterns, 6)} for p, c in pattern_counter.most_common(10)]
        lengths = non_null.map(lambda x: len(str(x)))
        avg_len = float(lengths.mean()) if len(lengths) > 0 else 0.0
        profile = {
            "column": col,
            "missing_rate": round(missing_rate, 6),
            "non_null_count": int(len(non_null)),
            "unique_count": int(unique_count),
            "unique_ratio": round(unique_ratio, 6),
            "numeric_parse_ratio": round(float(numeric_ratio), 6),
            "entropy": round(shannon_entropy(non_null), 6),
            "detected_type": None,
            "semantic_type": None,
            "index_like": is_index_like_column(non_null),
            "avg_length": round(avg_len, 6),
            "top_values": value_frequency_dict(non_null, topk=20),
            "top_patterns": top_patterns,
            "value_counter": Counter(non_null.tolist()),
        }
        if numeric_ratio >= NUMERIC_RATIO_THRESHOLD:
            profile["detected_type"] = "numeric"
        else:
            if unique_count <= ENUM_UNIQUE_THRESHOLD or unique_ratio <= ENUM_RATIO_THRESHOLD:
                profile["detected_type"] = "categorical"
            else:
                profile["detected_type"] = "text"
        if profile["index_like"]:
            profile["semantic_type"] = "id_like"
        elif is_id_like_column(non_null, unique_ratio, numeric_ratio, avg_len):
            profile["semantic_type"] = "id_like"
        elif is_code_like_column(non_null, unique_ratio, numeric_ratio, avg_len, top_patterns):
            profile["semantic_type"] = "code_like"
        elif profile["detected_type"] == "numeric":
            profile["semantic_type"] = "numeric"
        elif profile["detected_type"] == "categorical":
            profile["semantic_type"] = "categorical"
        else:
            profile["semantic_type"] = "text"
        profiles[col] = profile
    return profiles


def build_schema_rules(df, profiles):
    rules = []
    rule_id = 1
    for col, info in profiles.items():
        if col not in TARGET_CANDIDATE_COLUMNS:
            continue

        series = normalize_series(df[col])

        not_null_threshold = 0.75 if col in {"abv", "style", "ounces", "brewery_name", "city", "state"} else 0.55
        if info["missing_rate"] <= not_null_threshold:
            confidence, support = calc_not_null_confidence_and_support(series)
            rules.append({
                "rule_id": f"SCHEMA_{rule_id}",
                "rule_type": "not_null",
                "column": col,
                "description": f"{col} should not be null",
                "confidence": confidence,
                "support": support,
                "usage_role": "candidate_generation_rule",
            })
            rule_id += 1

        if len(info["top_patterns"]) > 0:
            main_pattern = info["top_patterns"][0]
            if main_pattern["ratio"] >= DOMINANT_PATTERN_THRESHOLD:
                confidence, support = calc_pattern_confidence_and_support(series, main_pattern["pattern"])
                rules.append({
                    "rule_id": f"SCHEMA_{rule_id}",
                    "rule_type": "pattern",
                    "column": col,
                    "description": f"{col} should mostly follow the dominant pattern",
                    "pattern": main_pattern["pattern"],
                    "confidence": confidence,
                    "support": support,
                    "usage_role": "strong_rule",
                })
                rule_id += 1

        if col == "ounces":
            # 保留宽格式规则，但更重要的是规范表示规则
            for rtype, regex, conf in [
                ("ounces_unit_format", r"^\d+(?:\.\d+)?\s*(oz|oz\.|ounce|ounces|oz\. alumi-tek|alumi-tek)$", 0.80),
                ("ounces_loose_pattern", r".*\d+(?:\.\d+)?\s*(oz|ounce).*", 0.75),
            ]:
                rules.append({
                    "rule_id": f"SCHEMA_{rule_id}",
                    "rule_type": rtype,
                    "column": col,
                    "description": f"{col} regex check {rtype}",
                    "regex": regex,
                    "confidence": conf,
                    "support": info["non_null_count"],
                    "usage_role": "candidate_generation_rule",
                })
                rule_id += 1

            rules.append({
                "rule_id": f"SCHEMA_{rule_id}",
                "rule_type": "ounces_canonical_format",
                "column": col,
                "description": "ounces should be stored in canonical clean style (e.g., 12 instead of 12.0 oz.)",
                "confidence": 0.95,
                "support": info["non_null_count"],
                "usage_role": "candidate_generation_rule",
            })
            rule_id += 1

        if col == "abv":
            rules.append({
                "rule_id": f"SCHEMA_{rule_id}",
                "rule_type": "abv_numeric_format",
                "column": col,
                "description": "abv should usually be numeric or numeric percent",
                "regex": r"^\d+(?:\.\d+)?%?$",
                "confidence": 0.82,
                "support": info["non_null_count"],
                "usage_role": "candidate_generation_rule",
            })
            rule_id += 1

            rules.append({
                "rule_id": f"SCHEMA_{rule_id}",
                "rule_type": "abv_canonical_format",
                "column": col,
                "description": "abv should be stored in canonical clean style without redundant percent suffix",
                "confidence": 0.95,
                "support": info["non_null_count"],
                "usage_role": "candidate_generation_rule",
            })
            rule_id += 1

        if col == "ibu":
            rules.append({
                "rule_id": f"SCHEMA_{rule_id}",
                "rule_type": "ibu_numeric_or_na_format",
                "column": col,
                "description": "ibu should usually be numeric or missing token",
                "regex": r"^(\d+(?:\.\d+)?|n/a|na|none|missing)$",
                "confidence": 0.82,
                "support": info["non_null_count"],
                "usage_role": "candidate_generation_rule",
            })
            rule_id += 1

        if col == "state":
            rules.append({
                "rule_id": f"SCHEMA_{rule_id}",
                "rule_type": "state_abbr_format",
                "column": col,
                "description": "state should usually be a two-letter abbreviation",
                "regex": r"^[a-z]{2}$",
                "confidence": 0.85,
                "support": info["non_null_count"],
                "usage_role": "candidate_generation_rule",
            })
            rule_id += 1

        if col == "city":
            rules.append({
                "rule_id": f"SCHEMA_{rule_id}",
                "rule_type": "city_state_suffix_pollution",
                "column": col,
                "description": "city should not end with a trailing state abbreviation like 'san diego ca'",
                "confidence": 0.92,
                "support": info["non_null_count"],
                "usage_role": "candidate_generation_rule",
            })
            rule_id += 1

    return rules


def select_fd_lhs_candidates(profiles):
    candidates = []
    for col, info in profiles.items():
        if (not ALLOW_INDEX_COLUMN) and info.get("index_like", False):
            continue
        if info["unique_ratio"] >= HIGH_CARDINALITY_RATIO_FOR_LHS:
            continue
        candidates.append(col)
    return candidates


def build_fd_rules(df, profiles):
    rules = []
    rule_id = 1
    cols = list(df.columns)
    norm_df = pd.DataFrame({c: normalize_series(df[c]) for c in cols})
    lhs_candidates = select_fd_lhs_candidates(profiles)
    rhs_candidates = [c for c in TARGET_CANDIDATE_COLUMNS if c in cols]
    lhs_sets = []
    for k in range(1, MAX_LHS_SIZE + 1):
        lhs_sets.extend(list(combinations(lhs_candidates, k)))

    for lhs in lhs_sets:
        for rhs in rhs_candidates:
            if rhs in lhs:
                continue
            confidence, support = calc_fd_confidence_and_support(norm_df, list(lhs), rhs)
            if support < MIN_SUPPORT_ROWS:
                continue
            if confidence >= FD_CONFIDENCE_THRESHOLD:
                rules.append({
                    "rule_id": f"FD_{rule_id}",
                    "rule_type": "functional_dependency",
                    "lhs": list(lhs),
                    "rhs": rhs,
                    "description": f"{', '.join(lhs)} -> {rhs}",
                    "confidence": confidence,
                    "support": support,
                    "usage_role": "strong_rule",
                })
                rule_id += 1

    dedup = {(tuple(sorted(r["lhs"])), r["rhs"]): r for r in rules}
    return list(dedup.values())


def build_soft_fd_rules(df, profiles):
    rules = []
    rule_id = 1
    cols = list(df.columns)
    norm_df = pd.DataFrame({c: normalize_series(df[c]) for c in cols})
    lhs_candidates = select_fd_lhs_candidates(profiles)
    rhs_candidates = [c for c in TARGET_CANDIDATE_COLUMNS if c in cols]
    lhs_sets = []
    for k in range(1, MAX_LHS_SIZE + 1):
        lhs_sets.extend(list(combinations(lhs_candidates, k)))

    for lhs in lhs_sets:
        for rhs in rhs_candidates:
            if rhs in lhs:
                continue
            confidence, support = calc_soft_fd_confidence_and_support(norm_df, list(lhs), rhs)
            if support < SOFT_FD_MIN_GROUP_SIZE:
                continue
            if SOFT_FD_CONFIDENCE_THRESHOLD <= confidence < FD_CONFIDENCE_THRESHOLD:
                rules.append({
                    "rule_id": f"CAND_SOFTFD_{rule_id}",
                    "rule_type": "soft_functional_dependency",
                    "lhs": list(lhs),
                    "rhs": rhs,
                    "description": f"soft FD: {', '.join(lhs)} -> {rhs}",
                    "confidence": confidence,
                    "support": support,
                    "usage_role": "candidate_generation_rule",
                })
                rule_id += 1

    return rules


def build_pair_context_rules(df):
    rules = []
    rule_id = 1
    norm_df = pd.DataFrame({c: normalize_series(df[c]) for c in df.columns})
    pair_specs = [
        (["brewery_id"], "brewery_name"),
        (["brewery_id"], "city"),
        (["brewery_id"], "state"),
        (["brewery_name"], "city"),
        (["brewery_name"], "state"),
        (["city"], "state"),
        (["beer_name", "brewery_id"], "style"),
        (["beer_name", "brewery_id"], "abv"),
        (["beer_name", "brewery_id"], "ibu"),
        (["beer_name", "brewery_id"], "ounces"),
    ]
    for lhs, rhs in pair_specs:
        if any(c not in norm_df.columns for c in lhs + [rhs]):
            continue
        sub = norm_df[lhs + [rhs]].dropna()
        if len(sub) < PAIR_CONTEXT_MIN_GROUP_SIZE:
            continue
        grouped = sub.groupby(lhs, dropna=True)[rhs]
        majority_map = {}
        total_support = 0
        for key, grp in grouped:
            if len(grp) < PAIR_CONTEXT_MIN_GROUP_SIZE:
                continue
            cnt = Counter(grp.tolist())
            maj_val, maj_cnt = cnt.most_common(1)[0]
            ratio = maj_cnt / len(grp)
            if ratio >= PAIR_CONTEXT_MIN_RATIO:
                if not isinstance(key, tuple):
                    key = (key,)
                majority_map["||".join(map(str, key))] = {
                    "dominant_value": maj_val,
                    "group_size": len(grp),
                    "ratio": round(ratio, 6),
                }
                total_support += len(grp)
        if majority_map:
            rules.append({
                "rule_id": f"CAND_PAIRCTX_{rule_id}",
                "rule_type": "dominant_value_by_context_pair" if len(lhs) == 2 else "dominant_value_by_context",
                "lhs": lhs,
                "rhs": rhs,
                "description": f"context dominant: {', '.join(lhs)} -> {rhs}",
                "mapping_size": len(majority_map),
                "majority_map": majority_map,
                "confidence": 0.84,
                "support": total_support,
                "usage_role": "candidate_generation_rule",
            })
            rule_id += 1
    return rules


def build_numeric_window_rules(df):
    rules = []
    norm_df = pd.DataFrame({c: normalize_series(df[c]) for c in df.columns})

    if "abv" in norm_df.columns:
        abv_vals = norm_df["abv"].map(parse_abv_like).dropna()
        if len(abv_vals) >= MIN_SUPPORT_ROWS:
            ds = pd.Series(abv_vals)
            q1 = float(ds.quantile(0.001))
            q99 = float(ds.quantile(0.999))
            lower = max(0.0, q1 - 0.01)
            upper = q99 + 0.01
            rules.append({
                "rule_id": "CAND_NUMWIN_1",
                "rule_type": "numeric_range_window",
                "column": "abv",
                "lower_bound": round(lower, 6),
                "upper_bound": round(upper, 6),
                "description": "abv should usually fall within learned numeric window",
                "confidence": 0.82,
                "support": len(abv_vals),
                "usage_role": "candidate_generation_rule",
            })

    if "ibu" in norm_df.columns:
        ibu_vals = norm_df["ibu"].map(parse_ibu_like).dropna()
        if len(ibu_vals) >= MIN_SUPPORT_ROWS:
            ds = pd.Series(ibu_vals)
            q1 = float(ds.quantile(0.001))
            q99 = float(ds.quantile(0.999))
            lower = max(0.0, q1 - 5.0)
            upper = q99 + 5.0
            rules.append({
                "rule_id": "CAND_NUMWIN_2",
                "rule_type": "numeric_range_window",
                "column": "ibu",
                "lower_bound": round(lower, 6),
                "upper_bound": round(upper, 6),
                "description": "ibu should usually fall within learned numeric window",
                "confidence": 0.82,
                "support": len(ibu_vals),
                "usage_role": "candidate_generation_rule",
            })

    if "ounces" in norm_df.columns:
        oz_vals = norm_df["ounces"].map(parse_ounces_like).dropna()
        if len(oz_vals) >= MIN_SUPPORT_ROWS:
            ds = pd.Series(oz_vals)
            q1 = float(ds.quantile(0.001))
            q99 = float(ds.quantile(0.999))
            lower = max(0.0, q1 - 1.0)
            upper = q99 + 1.0
            rules.append({
                "rule_id": "CAND_NUMWIN_3",
                "rule_type": "numeric_range_window",
                "column": "ounces",
                "lower_bound": round(lower, 6),
                "upper_bound": round(upper, 6),
                "description": "ounces should usually fall within learned numeric window",
                "confidence": 0.82,
                "support": len(oz_vals),
                "usage_role": "candidate_generation_rule",
            })

    return rules


def build_candidate_generation_rules(df, profiles):
    rules = []
    rule_id = 1
    norm_df = pd.DataFrame({c: normalize_series(df[c]) for c in df.columns})

    for col, prof in profiles.items():
        if col not in TARGET_CANDIDATE_COLUMNS:
            continue
        if col in SKIP_GLOBAL_DOMINANT_COLUMNS:
            continue
        top_values = prof.get("top_values", [])
        if not top_values:
            continue
        top1 = top_values[0]
        if top1["ratio"] >= DOMINANT_TYPO_MIN_RATIO:
            rules.append({
                "rule_id": f"CAND_GDOM_{rule_id}",
                "rule_type": "global_dominant_value",
                "column": col,
                "description": f"global dominant value for {col}",
                "dominant_value": top1["value"],
                "dominant_count": top1["count"],
                "dominant_ratio": top1["ratio"],
                "confidence": round(top1["ratio"], 6),
                "support": prof["non_null_count"],
                "usage_role": "candidate_generation_rule",
            })
            rule_id += 1

    lhs_candidates = select_fd_lhs_candidates(profiles)
    context_rule_id = 1
    for rhs in TARGET_CANDIDATE_COLUMNS:
        if rhs not in df.columns:
            continue
        for lhs in lhs_candidates:
            if lhs == rhs:
                continue
            sub = norm_df[[lhs, rhs]].dropna()
            if len(sub) < SOFT_FD_MIN_GROUP_SIZE:
                continue
            grouped = sub.groupby(lhs)[rhs]
            majority_map = {}
            total_support = 0
            for lhs_val, grp in grouped:
                if len(grp) < SOFT_FD_MIN_GROUP_SIZE:
                    continue
                cnt = Counter(grp.tolist())
                maj_val, maj_cnt = cnt.most_common(1)[0]
                ratio = maj_cnt / len(grp)
                if ratio >= SOFT_FD_CONFIDENCE_THRESHOLD:
                    majority_map[lhs_val] = {"dominant_value": maj_val, "group_size": len(grp), "ratio": round(ratio, 6)}
                    total_support += len(grp)
            if majority_map:
                rules.append({
                    "rule_id": f"CAND_CTXDOM_{context_rule_id}",
                    "rule_type": "dominant_value_by_context",
                    "lhs": [lhs],
                    "rhs": rhs,
                    "description": f"context dominant: {lhs} -> dominant {rhs}",
                    "mapping_size": len(majority_map),
                    "majority_map": majority_map,
                    "confidence": 0.85,
                    "support": total_support,
                    "usage_role": "candidate_generation_rule",
                })
                context_rule_id += 1

    if ENABLE_RARE_VALUE_RULE:
        for col, prof in profiles.items():
            if col not in TARGET_CANDIDATE_COLUMNS or col in SKIP_RARE_VALUE_COLUMNS:
                continue
            counter = prof["value_counter"]
            rare_values = [v for v, c in counter.items() if c <= RARE_VALUE_MAX_FREQ]
            if rare_values:
                rules.append({
                    "rule_id": f"CAND_RAREVAL_{rule_id}",
                    "rule_type": "rare_value",
                    "column": col,
                    "description": f"rare values in {col} may be suspicious",
                    "rare_values": rare_values[:5000],
                    "confidence": 0.6,
                    "support": sum(counter[v] for v in rare_values),
                    "usage_role": "candidate_generation_rule",
                })
                rule_id += 1

    if ENABLE_RARE_PATTERN_RULE:
        for col, prof in profiles.items():
            if col not in TARGET_CANDIDATE_COLUMNS or col in SKIP_RARE_PATTERN_COLUMNS:
                continue
            top_patterns = prof.get("top_patterns", [])
            rare_patterns = [x["pattern"] for x in top_patterns if x["ratio"] <= RARE_PATTERN_MAX_RATIO]
            if rare_patterns:
                rules.append({
                    "rule_id": f"CAND_RAREPAT_{rule_id}",
                    "rule_type": "rare_pattern",
                    "column": col,
                    "description": f"rare patterns in {col} may be suspicious",
                    "rare_patterns": rare_patterns,
                    "confidence": 0.6,
                    "support": prof["non_null_count"],
                    "usage_role": "candidate_generation_rule",
                })
                rule_id += 1

    for col in HIGH_RISK_MISSING_COLUMNS:
        if col not in profiles:
            continue
        prof = profiles[col]
        rules.append({
            "rule_id": f"CAND_HRMISS_{rule_id}",
            "rule_type": "high_risk_missing",
            "column": col,
            "description": f"{col} missing values are suspicious in candidate generation",
            "confidence": 0.8,
            "support": prof["non_null_count"],
            "usage_role": "candidate_generation_rule",
        })
        rule_id += 1

    if {"brewery_name", "city"}.issubset(df.columns):
        rules.append({
            "rule_id": f"CAND_PAIR_{rule_id}",
            "rule_type": "paired_missing_consistency",
            "columns": ["brewery_name", "city"],
            "description": "brewery_name and city should usually co-occur",
            "confidence": 0.8,
            "support": len(df),
            "usage_role": "candidate_generation_rule",
        })
        rule_id += 1

    if {"brewery_name", "state"}.issubset(df.columns):
        rules.append({
            "rule_id": f"CAND_PAIR_{rule_id}",
            "rule_type": "paired_missing_consistency",
            "columns": ["brewery_name", "state"],
            "description": "brewery_name and state should usually co-occur",
            "confidence": 0.8,
            "support": len(df),
            "usage_role": "candidate_generation_rule",
        })
        rule_id += 1

    rules.extend(build_pair_context_rules(df))
    rules.extend(build_numeric_window_rules(df))
    return rules


def assign_rule_priority(rule):
    if rule.get("usage_role", "candidate_generation_rule") == "candidate_generation_rule":
        if rule.get("confidence", 0.0) >= 0.85 and rule.get("support", 0) >= 20:
            return "medium"
        return "low"
    return "medium"


def enrich_rules_with_priority(rule_groups):
    new_groups = {}
    for group_name, rules in rule_groups.items():
        enriched = []
        for r in rules:
            r2 = dict(r)
            r2["priority"] = assign_rule_priority(r2)
            enriched.append(r2)
        new_groups[group_name] = enriched
    return new_groups


def build_rule_pool(input_csv, output_json):
    df = pd.read_csv(input_csv, dtype=str)
    df = maybe_drop_unnamed_columns(df)
    for col in df.columns:
        df[col] = normalize_series(df[col])

    profiles = analyze_columns(df)
    schema_rules = build_schema_rules(df, profiles)
    fd_rules = build_fd_rules(df, profiles)
    soft_fd_rules = build_soft_fd_rules(df, profiles)
    candidate_rules = build_candidate_generation_rules(df, profiles)

    rule_groups = {
        "schema_rules": schema_rules,
        "fd_rules": fd_rules,
        "soft_fd_rules": soft_fd_rules,
        "candidate_rules": candidate_rules,
    }
    rule_groups = enrich_rules_with_priority(rule_groups)

    rule_pool = {
        "metadata": {
            "source_file": input_csv,
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "design_goal": "high_recall_candidate_generation",
            "notes": [
                "candidate inclusion is determined only by violating rules in this pool",
                "candidate targets focus on non-id beer attributes; index/id/brewery_id mainly serve as context",
                "recall is boosted mainly by brewery consistency rules and canonical-format rules on ounces/abv/city",
                "beer_name and brewery_name skip rare/global-dominant heuristics to reduce false positives",
            ],
        },
        "column_profiles": profiles,
        "rules": rule_groups,
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(rule_pool, f, ensure_ascii=False, indent=2)

    print(f"[OK] Rule pool saved to: {output_json}")
    print(f"Schema rules: {len(schema_rules)}")
    print(f"FD rules: {len(fd_rules)}")
    print(f"Soft FD rules: {len(soft_fd_rules)}")
    print(f"Candidate generation rules: {len(candidate_rules)}")


if __name__ == "__main__":
    build_rule_pool(INPUT_CSV, OUTPUT_JSON)

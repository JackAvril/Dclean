import json
import math
import re
from collections import Counter
from itertools import combinations
from typing import Dict, List, Any, Optional

import pandas as pd

INPUT_CSV = "/mnt/mydata/dq/projects/Splittree/flights/flights_dirty.csv"
OUTPUT_JSON = "rule_pool_flights.json"

MISSING_TOKENS = {
    "", "empty", "null", "none", "nan", "na", "n/a", "missing", "unk",
    "not available", "contact airline"
}

NUMERIC_RATIO_THRESHOLD = 0.90
ENUM_UNIQUE_THRESHOLD = 30
ENUM_RATIO_THRESHOLD = 0.05

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
SKIP_SCHEMA_FOR_ID_LIKE = True

TIME_COLUMNS = {"sched_dep_time", "act_dep_time", "sched_arr_time", "act_arr_time"}
TARGET_CANDIDATE_COLUMNS = set(TIME_COLUMNS)
HIGH_RISK_MISSING_COLUMNS = set(TIME_COLUMNS)

ENABLE_RARE_VALUE_RULE = True
RARE_VALUE_MAX_FREQ = 2
ENABLE_RARE_PATTERN_RULE = True
RARE_PATTERN_MAX_RATIO = 0.02


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
    cleaned = series.dropna().astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False)
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


def parse_time_like(s: Optional[str]) -> Optional[int]:
    if s is None:
        return None
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    for junk in ["(estimated runway)", "(estimated)", "(actual)", "(gate)", "(runway)", "(+8:00)", "(-00:00)"]:
        s = s.replace(junk, "").strip()
    s = re.sub(r"\bdec\s+\d{1,2}\b", "", s).strip()
    m = re.search(r"(\d{1,2}:\d{2})\s*(a\.m\.|p\.m\.)", s)
    if not m:
        return None
    hhmm = m.group(1); ap = m.group(2)
    hh, mm = map(int, hhmm.split(":"))
    if ap == "p.m." and hh != 12:
        hh += 12
    if ap == "a.m." and hh == 12:
        hh = 0
    return hh * 60 + mm


def circ_diff(a: Optional[int], b: Optional[int]) -> Optional[int]:
    if a is None or b is None:
        return None
    d = a - b
    while d <= -720:
        d += 1440
    while d > 720:
        d -= 1440
    return d


def duration(start: Optional[int], end: Optional[int]) -> Optional[int]:
    if start is None or end is None:
        return None
    d = end - start
    if d < 0:
        d += 1440
    return d


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
        not_null_threshold = 0.70 if col in {"act_dep_time", "act_arr_time"} else 0.55
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
        for rtype, regex, conf in [
            ("time_format", r"^\d{1,2}:\d{2}\s*(a\.m\.|p\.m\.)$", 0.85),
            ("datetime_time_format", r"^(\d{1,2}/\d{1,2}/\d{4}\s+)?\d{1,2}:\d{2}\s*(a\.m\.|p\.m\.)(\s+\w+\s+\d{1,2})?.*$", 0.8),
            ("time_loose_pattern", r".*\d{1,2}:\d{2}\s*(a\.m\.|p\.m\.).*", 0.8),
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
        (["flight", "sched_dep_time"], "act_dep_time"),
        (["src", "sched_dep_time"], "act_dep_time"),
        (["flight", "sched_arr_time"], "act_arr_time"),
        (["src", "sched_arr_time"], "act_arr_time"),
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
                majority_map["||".join(map(str, key))] = {"dominant_value": maj_val, "group_size": len(grp), "ratio": round(ratio, 6)}
                total_support += len(grp)
        if majority_map:
            rules.append({
                "rule_id": f"CAND_PAIRCTX_{rule_id}",
                "rule_type": "dominant_value_by_context_pair",
                "lhs": lhs,
                "rhs": rhs,
                "description": f"context pair dominant: {', '.join(lhs)} -> dominant {rhs}",
                "mapping_size": len(majority_map),
                "majority_map": majority_map,
                "confidence": 0.84,
                "support": total_support,
                "usage_role": "candidate_generation_rule",
            })
            rule_id += 1
    return rules


def build_time_window_rules(df):
    rules = []
    norm_df = pd.DataFrame({c: normalize_series(df[c]) for c in df.columns})

    for col in ["act_dep_time", "act_arr_time"]:
        if col in norm_df.columns:
            rules.append({
                "rule_id": f"CAND_ACTFULL_{1 if col=='act_dep_time' else 2}",
                "rule_type": "actual_time_global_candidate",
                "column": col,
                "description": f"{col} is globally included for high recall on flights",
                "confidence": 0.7,
                "support": len(norm_df),
                "usage_role": "candidate_generation_rule",
            })

    if {"sched_dep_time", "act_dep_time"}.issubset(norm_df.columns):
        sched = norm_df["sched_dep_time"].map(parse_time_like)
        act = norm_df["act_dep_time"].map(parse_time_like)
        diffs = [circ_diff(a, s) for a, s in zip(act, sched) if a is not None and s is not None]
        if diffs:
            ds = pd.Series(diffs)
            lower = int(math.floor(ds.quantile(0.0005) - 15))
            upper = int(math.ceil(ds.quantile(0.9995) + 30))
            rules.append({
                "rule_id": "CAND_DELAYWIN_1",
                "rule_type": "delay_window",
                "reference_column": "sched_dep_time",
                "target_column": "act_dep_time",
                "lower_bound": lower,
                "upper_bound": upper,
                "description": "act_dep_time should usually be within learned delay window of sched_dep_time",
                "confidence": 0.82,
                "support": len(diffs),
                "usage_role": "candidate_generation_rule",
            })

    if {"sched_arr_time", "act_arr_time"}.issubset(norm_df.columns):
        sched = norm_df["sched_arr_time"].map(parse_time_like)
        act = norm_df["act_arr_time"].map(parse_time_like)
        diffs = [circ_diff(a, s) for a, s in zip(act, sched) if a is not None and s is not None]
        if diffs:
            ds = pd.Series(diffs)
            lower = int(math.floor(ds.quantile(0.0005) - 15))
            upper = int(math.ceil(ds.quantile(0.9995) + 30))
            rules.append({
                "rule_id": "CAND_DELAYWIN_2",
                "rule_type": "delay_window",
                "reference_column": "sched_arr_time",
                "target_column": "act_arr_time",
                "lower_bound": lower,
                "upper_bound": upper,
                "description": "act_arr_time should usually be within learned delay window of sched_arr_time",
                "confidence": 0.82,
                "support": len(diffs),
                "usage_role": "candidate_generation_rule",
            })

    needed = {"sched_dep_time", "sched_arr_time", "act_dep_time", "act_arr_time"}
    if needed.issubset(norm_df.columns):
        sdep = norm_df["sched_dep_time"].map(parse_time_like)
        sarr = norm_df["sched_arr_time"].map(parse_time_like)
        adep = norm_df["act_dep_time"].map(parse_time_like)
        aarr = norm_df["act_arr_time"].map(parse_time_like)
        diffs = []
        for sd, sa, ad, aa in zip(sdep, sarr, adep, aarr):
            sched_dur = duration(sd, sa)
            act_dur = duration(ad, aa)
            if sched_dur is None or act_dur is None:
                continue
            diffs.append(act_dur - sched_dur)
        if diffs:
            ds = pd.Series(diffs)
            lower = int(math.floor(ds.quantile(0.001) - 20))
            upper = int(math.ceil(ds.quantile(0.995) + 30))
            rules.append({
                "rule_id": "CAND_DURWIN_1",
                "rule_type": "duration_consistency_window",
                "sched_start_column": "sched_dep_time",
                "sched_end_column": "sched_arr_time",
                "act_start_column": "act_dep_time",
                "act_end_column": "act_arr_time",
                "lower_bound": lower,
                "upper_bound": upper,
                "description": "actual flight duration should usually stay within learned window around scheduled duration",
                "confidence": 0.8,
                "support": len(diffs),
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
            if col not in TARGET_CANDIDATE_COLUMNS:
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
            if col not in TARGET_CANDIDATE_COLUMNS:
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

    for a, b in [("sched_dep_time", "act_dep_time"), ("sched_arr_time", "act_arr_time")]:
        if a in df.columns and b in df.columns:
            rules.append({
                "rule_id": f"CAND_PAIR_{rule_id}",
                "rule_type": "paired_missing_consistency",
                "columns": [a, b],
                "description": f"{a} and {b} should usually co-occur",
                "confidence": 0.8,
                "support": len(df),
                "usage_role": "candidate_generation_rule",
            })
            rule_id += 1

    for start_col, end_col in [("sched_dep_time", "sched_arr_time"), ("act_dep_time", "act_arr_time")]:
        if start_col in df.columns and end_col in df.columns:
            rules.append({
                "rule_id": f"CAND_TIMEPAIR_{rule_id}",
                "rule_type": "temporal_pair_consistency",
                "columns": [start_col, end_col],
                "description": f"{end_col} should usually not be earlier than {start_col}",
                "confidence": 0.8,
                "support": len(df),
                "usage_role": "candidate_generation_rule",
            })
            rule_id += 1

    rules.extend(build_pair_context_rules(df))
    rules.extend(build_time_window_rules(df))
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
    rule_groups = {"schema_rules": schema_rules, "fd_rules": fd_rules, "soft_fd_rules": soft_fd_rules, "candidate_rules": candidate_rules}
    rule_groups = enrich_rules_with_priority(rule_groups)
    rule_pool = {
        "metadata": {
            "source_file": input_csv,
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "design_goal": "high_recall_candidate_generation",
            "notes": [
                "candidate inclusion is determined only by violating rules in this pool",
                "candidate target columns are restricted to flight time attributes; src/flight/index serve as context only",
                "recall is boosted mainly by rules on act_dep_time and act_arr_time",
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

import json
import math
import re
from collections import Counter
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd

INPUT_CSV = "/mnt/mydata/dq/projects/Splittree/flights/flights_dirty.csv"
INPUT_RULE_JSON = "rule_pool_flights.json"

OUTPUT_JSONL = "candidate_cells_flights.jsonl"
OUTPUT_CSV = "candidate_cells_flights.csv"

MISSING_TOKENS = {
    "", "empty", "null", "none", "nan", "na", "n/a", "missing", "unk",
    "not available", "contact airline"
}
SKIP_NULL_FOR_NON_NOTNULL_RULES = True
MAX_FD_MAPPING_SIZE = 200000
MAX_CONTEXT_MAPPING_SIZE = 200000

PRIORITY_WEIGHT = {"high": 1.5, "medium": 1.0, "low": 0.6}
RULE_TYPE_WEIGHT = {
    "not_null": 1.2,
    "pattern": 0.7,
    "functional_dependency": 1.5,
    "soft_functional_dependency": 1.1,
    "rare_value": 0.7,
    "rare_pattern": 0.7,
    "global_dominant_value": 0.6,
    "dominant_value_by_context": 0.9,
    "dominant_value_by_context_pair": 1.0,
    "high_risk_missing": 0.9,
    "paired_missing_consistency": 0.8,
    "time_format": 0.85,
    "datetime_time_format": 0.85,
    "time_loose_pattern": 0.8,
    "temporal_pair_consistency": 0.95,
    "delay_window": 0.95,
    "duration_consistency_window": 1.0,
    "actual_time_global_candidate": 0.45,
}


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
        r = dict(rule)
        r["_majority_mapping"] = build_fd_majority_mapping(df, rule["lhs"], rule["rhs"])
        out.append(r)
    return out


def prepare_context_dominant_rule_runtime(rules):
    out = []
    for rule in rules:
        if rule.get("rule_type") != "dominant_value_by_context":
            continue
        mapping = {}
        for ctx_val, info in rule.get("majority_map", {}).items():
            mapping[ctx_val] = (info.get("dominant_value"), int(info.get("group_size", 0)), float(info.get("ratio", 0.0)))
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
            mapping[key] = (info.get("dominant_value"), int(info.get("group_size", 0)), float(info.get("ratio", 0.0)))
        r = dict(rule)
        r["_pair_context_mapping"] = mapping
        out.append(r)
    return out


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
        cell_map[key] = {"row_id": row_id, "column": column, "value": violation.get("value"), "violated_rules": [], "conflict_score": 0.0}
    cell_map[key]["violated_rules"].append(info)
    cell_map[key]["conflict_score"] += rule_weight(rule)


def check_not_null_rule(row_idx, row, rule):
    col = rule["column"]; val = row.get(col)
    if pd.isna(val) or val is None:
        return {"row_id": row_idx, "column": col, "value": val, "rule": rule, "reason": "value is null but rule requires non-null"}
    return None


def check_pattern_rule(row_idx, row, rule):
    col = rule["column"]; val = row.get(col)
    if (pd.isna(val) or val is None) and SKIP_NULL_FOR_NON_NOTNULL_RULES:
        return None
    observed = detect_basic_pattern(val)
    target = rule["pattern"]
    if observed != target:
        return {"row_id": row_idx, "column": col, "value": val, "rule": rule, "reason": f"pattern mismatch: observed={observed}, expected={target}"}
    return None


def check_fd_like_rule(row_idx, row, rule):
    lhs, rhs = rule["lhs"], rule["rhs"]
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
        return {"row_id": row_idx, "column": rhs, "value": rhs_val, "rule": rule, "reason": f"{rule['rule_type']} violation: lhs={lhs_key} suggests rhs={expected_rhs}, observed={rhs_val}", "expected_value": expected_rhs}
    return None


def check_rare_value_rule(row_idx, row, rule):
    col = rule["column"]; val = row.get(col)
    if pd.isna(val) or val is None:
        return None
    if val in set(rule.get("rare_values", [])):
        return {"row_id": row_idx, "column": col, "value": val, "rule": rule, "reason": "value is in rare_values set"}
    return None


def check_rare_pattern_rule(row_idx, row, rule):
    col = rule["column"]; val = row.get(col)
    if pd.isna(val) or val is None:
        return None
    observed = detect_basic_pattern(val)
    if observed in set(rule.get("rare_patterns", [])):
        return {"row_id": row_idx, "column": col, "value": val, "rule": rule, "reason": f"pattern {observed} is rare for this column"}
    return None


def check_global_dominant_rule(row_idx, row, rule):
    col = rule["column"]; val = row.get(col)
    if pd.isna(val) or val is None:
        return None
    dominant = rule.get("dominant_value")
    if dominant is not None and val != dominant:
        return {"row_id": row_idx, "column": col, "value": val, "rule": rule, "reason": f"value differs from global dominant value: dominant={dominant}, observed={val}", "expected_value": dominant}
    return None


def check_context_dominant_rule(row_idx, row, rule):
    lhs = rule.get("lhs", []); rhs = rule.get("rhs")
    if len(lhs) != 1 or rhs is None:
        return None
    ctx_col = lhs[0]
    ctx_val = row.get(ctx_col); tgt_val = row.get(rhs)
    if ctx_val is None or tgt_val is None:
        return None
    mapping = rule.get("_context_mapping", {})
    if ctx_val not in mapping:
        return None
    dominant_val, support, dominant_ratio = mapping[ctx_val]
    if tgt_val != dominant_val:
        return {"row_id": row_idx, "column": rhs, "value": tgt_val, "rule": rule, "reason": f"context dominant mismatch: {ctx_col}={ctx_val} suggests {rhs}={dominant_val}, observed={tgt_val}", "expected_value": dominant_val}
    return None


def check_pair_context_rule(row_idx, row, rule):
    lhs = rule.get("lhs", []); rhs = rule.get("rhs")
    if len(lhs) != 2 or rhs is None:
        return None
    vals = [row.get(lhs[0]), row.get(lhs[1])]
    if vals[0] is None or vals[1] is None:
        return None
    tgt_val = row.get(rhs)
    if tgt_val is None:
        return None
    key = "||".join(map(str, vals))
    mapping = rule.get("_pair_context_mapping", {})
    if key not in mapping:
        return None
    dominant_val, support, dominant_ratio = mapping[key]
    if tgt_val != dominant_val:
        return {"row_id": row_idx, "column": rhs, "value": tgt_val, "rule": rule, "reason": f"context pair dominant mismatch: {lhs}={vals} suggests {rhs}={dominant_val}, observed={tgt_val}", "expected_value": dominant_val}
    return None


def check_high_risk_missing_rule(row_idx, row, rule):
    col = rule["column"]; val = row.get(col)
    if pd.isna(val) or val is None:
        return {"row_id": row_idx, "column": col, "value": val, "rule": rule, "reason": "value is null under high_risk_missing rule"}
    return None


def check_paired_missing_consistency_rule(row_idx, row, rule):
    cols = rule.get("columns", [])
    if len(cols) != 2:
        return None
    a_col, b_col = cols
    a, b = row.get(a_col), row.get(b_col)
    if (a is None and b is not None) or (a is not None and b is None):
        bad_col = a_col if a is None else b_col
        return {"row_id": row_idx, "column": bad_col, "value": row.get(bad_col), "rule": rule, "reason": f"paired missing inconsistency: {a_col}={a}, {b_col}={b}"}
    return None


def check_regex_rule(row_idx, row, rule):
    col = rule["column"]; val = row.get(col)
    if pd.isna(val) or val is None:
        return None
    pattern = rule.get("regex")
    if pattern and re.match(pattern, str(val)) is None:
        return {"row_id": row_idx, "column": col, "value": val, "rule": rule, "reason": f"value does not match regex: {pattern}"}
    return None


def check_temporal_pair_rule(row_idx, row, rule):
    cols = rule.get("columns", [])
    if len(cols) != 2:
        return None
    start_col, end_col = cols
    start_v = row.get(start_col); end_v = row.get(end_col)
    if start_v is None or end_v is None:
        return None
    s = parse_time_like(start_v); e = parse_time_like(end_v)
    if s is None:
        return {"row_id": row_idx, "column": start_col, "value": start_v, "rule": rule, "reason": f"cannot parse start time for temporal consistency: {start_v}"}
    if e is None:
        return {"row_id": row_idx, "column": end_col, "value": end_v, "rule": rule, "reason": f"cannot parse end time for temporal consistency: {end_v}"}
    if e <= s:
        return {"row_id": row_idx, "column": end_col, "value": end_v, "rule": rule, "reason": f"temporal inconsistency: {end_col}={end_v} is not later than {start_col}={start_v}"}
    return None


def check_delay_window_rule(row_idx, row, rule):
    ref_col = rule["reference_column"]; tgt_col = rule["target_column"]
    ref_v = row.get(ref_col); tgt_v = row.get(tgt_col)
    if ref_v is None or tgt_v is None:
        return None
    ref_t = parse_time_like(ref_v); tgt_t = parse_time_like(tgt_v)
    if ref_t is None or tgt_t is None:
        return None
    d = circ_diff(tgt_t, ref_t)
    if d is None:
        return None
    if d < rule["lower_bound"] or d > rule["upper_bound"]:
        return {"row_id": row_idx, "column": tgt_col, "value": tgt_v, "rule": rule, "reason": f"delay window violation: diff={d}, expected in [{rule['lower_bound']}, {rule['upper_bound']}]"}
    return None


def check_duration_consistency_rule(row_idx, row, rule):
    sd = parse_time_like(row.get(rule["sched_start_column"]))
    sa = parse_time_like(row.get(rule["sched_end_column"]))
    ad = parse_time_like(row.get(rule["act_start_column"]))
    aa = parse_time_like(row.get(rule["act_end_column"]))
    if sd is None or sa is None or ad is None or aa is None:
        return None
    sched_dur = duration(sd, sa)
    act_dur = duration(ad, aa)
    if sched_dur is None or act_dur is None:
        return None
    diff = act_dur - sched_dur
    if diff < rule["lower_bound"] or diff > rule["upper_bound"]:
        return {"row_id": row_idx, "column": rule["act_end_column"], "value": row.get(rule["act_end_column"]), "rule": rule, "reason": f"duration consistency violation: diff={diff}, expected in [{rule['lower_bound']}, {rule['upper_bound']}]"}
    return None


def check_actual_time_global_candidate_rule(row_idx, row, rule):
    col = rule["column"]
    return {"row_id": row_idx, "column": col, "value": row.get(col), "rule": rule, "reason": "actual time global candidate rule for high recall"}


def shrink_search_space(input_csv, rule_json, output_jsonl, output_csv):
    df = pd.read_csv(input_csv, dtype=str)
    df = maybe_drop_unnamed_columns(df)
    for col in df.columns:
        df[col] = normalize_series(df[col])
    rules = load_rule_pool(rule_json)

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
    regex_rules = [r for r in rules if r.get("rule_type") in {"time_format", "datetime_time_format", "time_loose_pattern"}]
    temporal_pair_rules = [r for r in rules if r.get("rule_type") == "temporal_pair_consistency"]
    delay_window_rules = [r for r in rules if r.get("rule_type") == "delay_window"]
    duration_rules = [r for r in rules if r.get("rule_type") == "duration_consistency_window"]
    act_global_rules = [r for r in rules if r.get("rule_type") == "actual_time_global_candidate"]

    cell_map = {}
    for row_idx, row in df.iterrows():
        for group, fn in [
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
            (regex_rules, check_regex_rule),
            (temporal_pair_rules, check_temporal_pair_rule),
            (delay_window_rules, check_delay_window_rule),
            (duration_rules, check_duration_consistency_rule),
            (act_global_rules, check_actual_time_global_candidate_rule),
        ]:
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
    print(f"Shrink ratio: {len(candidates)/total_cells:.6f}")


if __name__ == "__main__":
    shrink_search_space(INPUT_CSV, INPUT_RULE_JSON, OUTPUT_JSONL, OUTPUT_CSV)

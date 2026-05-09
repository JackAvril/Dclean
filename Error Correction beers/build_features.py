import os
import json
import math
import re
from difflib import SequenceMatcher
from collections import Counter, defaultdict

import pandas as pd


# ============================================================
# 1. 配置区（直接改这里）
# ============================================================

CANDIDATES_FILE = "/mnt/mydata/dq/projects/Splittree/test/Error Correction beers/repair_candidates_expanded.csv"
CONTEXT_FILE = "/mnt/mydata/dq/projects/Splittree/test/Error Detection beers/candidate_llm_contexts_beers.jsonl"
FALLBACK_CONTEXT_FLAT_FILE = "/mnt/mydata/dq/projects/Splittree/test/Error Detection beers/candidate_llm_contexts_flat_beers.csv"
RAW_DATA_FILE = "/mnt/mydata/dq/projects/Splittree/beers/beers_dirty.csv"
COOCCURRENCE_DICT_FILE = None

OUTPUT_FEATURES_CSV = "repair_candidate_features_v2.csv"
OUTPUT_BUILT_COOCCUR_JSON = "built_cooccurrence_dict_v2.json"

MAX_CONTEXT_PAIRS = 12
ENABLE_NUMERIC_CANDIDATE_CHECK = True
SMOOTHING_ALPHA = 1.0
MAX_DISTINCT_VALUES_PER_COLUMN = 2000
TOP_K_SIMILAR_ROW_VALUES = 8

EMPTY_TOKENS = {"", "empty", "null", "none", "nan", "na", "n/a", "missing"}


def norm_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def canonical_empty_text(x):
    s = norm_text(x)
    return "empty" if s in EMPTY_TOKENS else ("" if pd.isna(x) else str(x).strip())


def safe_float(x, default=0.0):
    try:
        if x is None or pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def path_exists_nonempty(path):
    return path is not None and str(path).strip() != "" and os.path.exists(str(path))


def resolve_output_path(output_path, anchor_input_path=None):
    output_path = str(output_path)
    if os.path.isabs(output_path):
        return output_path
    return os.path.abspath(output_path)


def string_similarity(a, b):
    a = norm_text(a)
    b = norm_text(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def try_parse_float(x):
    try:
        return float(str(x).replace(",", "").strip().replace("%", ""))
    except Exception:
        return None


def load_jsonl(path):
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


def load_context_records(path):
    lower = path.lower()
    if lower.endswith(".jsonl"):
        return load_jsonl(path)
    if lower.endswith(".csv") or lower.endswith(".txt"):
        df = pd.read_csv(path, encoding="utf-8-sig")
        return df.to_dict(orient="records")
    try:
        return load_jsonl(path)
    except Exception:
        df = pd.read_csv(path, encoding="utf-8-sig")
        return df.to_dict(orient="records")


KEYBOARD_ROWS = ["1234567890", "qwertyuiop", "asdfghjkl", "zxcvbnm"]
KEYBOARD_POS = {}
for r, row in enumerate(KEYBOARD_ROWS):
    for c, ch in enumerate(row):
        KEYBOARD_POS[ch] = (r, c)


def char_keyboard_distance(a, b):
    a = norm_text(a)
    b = norm_text(b)
    if len(a) != 1 or len(b) != 1:
        return None
    if a not in KEYBOARD_POS or b not in KEYBOARD_POS:
        return None
    ra, ca = KEYBOARD_POS[a]
    rb, cb = KEYBOARD_POS[b]
    return abs(ra - rb) + abs(ca - cb)


def avg_keyboard_distance_for_same_length(a, b):
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
            dists.append(float(d if d is not None else 5.0))
    return sum(dists) / len(dists)


def soundex_token(token):
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
        if code != prev and code != "":
            codes.append(code)
        prev = code
    return (first + "".join(codes) + "000")[:4]


def phrase_soundex(s):
    s = norm_text(s)
    if not s:
        return ""
    toks = [t for t in s.replace("-", " ").replace("/", " ").split() if t]
    return " ".join(soundex_token(t) for t in toks)


def phonetic_similarity(a, b):
    return string_similarity(phrase_soundex(a), phrase_soundex(b))


US_STATE_CODES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC"
}
STATE_NAME_TO_CODE = {
    "california": "CA", "colorado": "CO", "michigan": "MI", "indiana": "IN",
    "texas": "TX", "oregon": "OR", "pennsylvania": "PA", "illinois": "IL",
    "wisconsin": "WI", "massachusetts": "MA", "oklahoma": "OK", "virginia": "VA",
    "new york": "NY", "north carolina": "NC", "florida": "FL", "washington": "WA",
}


def normalize_state_candidate(x):
    s = canonical_empty_text(x)
    if s in {"", "empty"}:
        return None
    upper = s.strip().upper()
    if upper in US_STATE_CODES:
        return upper
    lower = norm_text(s)
    if lower in STATE_NAME_TO_CODE:
        return STATE_NAME_TO_CODE[lower]
    m = re.search(r"\b([A-Za-z]{2})\b$", s.strip())
    if m:
        cand = m.group(1).upper()
        if cand in US_STATE_CODES:
            return cand
    return None


def strip_state_suffix_from_city(city):
    s = canonical_empty_text(city)
    if s in {"", "empty"}:
        return None
    parts = s.strip().split()
    if len(parts) >= 2:
        last = parts[-1].upper().strip(".,;:()[]{}")
        if last in US_STATE_CODES:
            city_only = " ".join(parts[:-1]).strip()
            if city_only:
                return city_only
    return None


def normalize_city_candidate(x):
    s = canonical_empty_text(x)
    if s in {"", "empty"}:
        return None
    return strip_state_suffix_from_city(s) or s.strip()


def normalize_ounces_candidate(x):
    s = canonical_empty_text(x).lower().strip()
    if s in {"", "empty"}:
        return None
    s = s.replace("ounces", "ounce").replace("ozs", "oz").replace("0z", "oz").replace("o.z.", "oz")
    m = re.search(r"(-?\d+(?:\.\d+)?)", s)
    if not m:
        return None
    val = safe_float(m.group(1), None)
    if val is None or val <= 0 or val > 100:
        return None
    return f"{val:.1f} oz."


def normalize_abv_candidate(x):
    s = canonical_empty_text(x).lower().strip()
    if s in {"", "empty"}:
        return None
    s = s.replace("abv", "").replace(" ", "")
    m = re.search(r"(-?\d+(?:\.\d+)?)", s)
    if not m:
        return None
    val = safe_float(m.group(1), None)
    if val is None:
        return None
    if "%" in s or val > 1.0:
        val = val / 100.0
    if val < 0 or val > 1:
        return None
    out = f"{val:.3f}"
    return out.rstrip("0").rstrip(".") if "." in out else out


def normalize_ibu_candidate(x):
    s = canonical_empty_text(x).lower().strip()
    if s in {"", "empty"}:
        return None
    s = s.replace("ibu", "")
    m = re.search(r"(-?\d+(?:\.\d+)?)", s)
    if not m:
        return None
    val = safe_float(m.group(1), None)
    if val is None or val < 0 or val > 1000:
        return None
    if abs(val - round(val)) < 1e-9:
        return str(int(round(val)))
    return f"{val:.1f}".rstrip("0").rstrip(".")


def normalize_id_candidate(x):
    s = canonical_empty_text(x)
    if s in {"", "empty"}:
        return None
    m = re.search(r"\d+", s)
    return str(int(m.group(0))) if m else None


def normalize_by_column(column, value):
    col = norm_text(column)
    if col == "state":
        return normalize_state_candidate(value)
    if col == "city":
        return normalize_city_candidate(value)
    if col == "ounces":
        return normalize_ounces_candidate(value)
    if col == "abv":
        return normalize_abv_candidate(value)
    if col == "ibu":
        return normalize_ibu_candidate(value)
    if col in {"index", "id", "brewery_id"}:
        return normalize_id_candidate(value)
    return None


def value_equal_for_column(column, a, b):
    fa = normalize_by_column(column, a)
    fb = normalize_by_column(column, b)
    if fa is not None and fb is not None:
        return norm_text(fa) == norm_text(fb)
    return norm_text(a) == norm_text(b)

def build_context_map_rich(context_records):
    context_map = {}
    column_value_counter = defaultdict(Counter)

    for obj in context_records:
        if "row_id" not in obj or "column" not in obj:
            continue
        val = canonical_empty_text(obj.get("value", ""))
        if val not in {"", "empty"}:
            column_value_counter[str(obj["column"])][val] += 1

    for obj in context_records:
        if "row_id" not in obj or "column" not in obj:
            continue
        try:
            row_id = int(obj["row_id"])
        except Exception:
            continue
        column = str(obj["column"])
        value = canonical_empty_text(obj.get("value", ""))

        row_values = {}
        rc = obj.get("row_context", {})
        if isinstance(rc, dict):
            rv = rc.get("row_values", {})
            if isinstance(rv, dict):
                row_values = {str(k): canonical_empty_text(v) for k, v in rv.items()}

        conflict_context = obj.get("conflict_context", {})
        rule_expected_values, expected_values_from_rules = [], []
        alternative_values_from_column = []
        counters = {
            "strong_rule_count": 0, "candidate_generation_rule_count": 0, "fd_like_count": 0,
            "global_rule_count": 0, "typo_rule_count": 0, "pattern_rule_count": 0,
            "schema_rule_count": 0, "numeric_window_rule_count": 0,
        }

        if isinstance(conflict_context, dict):
            for vr in conflict_context.get("violated_rules_detailed", []) or []:
                if isinstance(vr, dict):
                    ev = canonical_empty_text(vr.get("expected_value", None))
                    if ev not in {"", "empty"}:
                        rule_expected_values.append(ev)

            for item in conflict_context.get("expected_values_from_rules", []) or []:
                if isinstance(item, dict):
                    ev = canonical_empty_text(item.get("value", None))
                    if ev not in {"", "empty"}:
                        expected_values_from_rules.append(ev)

            for key in ["alternative_values_from_column", "all_alternative_values"]:
                for item in conflict_context.get(key, []) or []:
                    if isinstance(item, dict):
                        v = canonical_empty_text(item.get("value", None))
                        if v not in {"", "empty"}:
                            alternative_values_from_column.append({
                                "value": v,
                                "count": int(item.get("count", 0) or 0),
                                "ratio": safe_float(item.get("ratio", 0.0), 0.0),
                            })

            for k in counters:
                counters[k] = int(conflict_context.get(k, 0) or 0)

        sim_ctx = obj.get("similarity_context", {})
        neighbor_majority_value = None
        neighbor_majority_ratio = 0.0
        similar_row_target_values = []
        if isinstance(sim_ctx, dict):
            nmv = canonical_empty_text(sim_ctx.get("neighbor_majority_value", ""))
            neighbor_majority_value = None if nmv in {"", "empty"} else nmv
            neighbor_majority_ratio = safe_float(sim_ctx.get("neighbor_majority_ratio", 0.0), 0.0)
            for item in (sim_ctx.get("topk_similar_rows", []) or [])[:TOP_K_SIMILAR_ROW_VALUES]:
                if isinstance(item, dict):
                    tv = canonical_empty_text(item.get("target_column_value", ""))
                    if tv not in {"", "empty"}:
                        similar_row_target_values.append(tv)

        column_top_values = []
        col_ctx = obj.get("column_context", {})
        column_detected_type = obj.get("detected_type", None)
        column_numeric_stats = None
        if isinstance(col_ctx, dict):
            for item in (col_ctx.get("top_values_with_counts", []) or []):
                if isinstance(item, dict) and item.get("value", None) is not None:
                    val = canonical_empty_text(item["value"])
                    if val not in {"", "empty"}:
                        column_top_values.append({
                            "value": val,
                            "count": int(item.get("count", 0) or 0),
                            "ratio": safe_float(item.get("ratio", 0.0), 0.0),
                        })
            column_detected_type = col_ctx.get("detected_type", column_detected_type)
            column_numeric_stats = col_ctx.get("numeric_stats", None)

        if not column_top_values:
            counter = column_value_counter[column]
            total = max(sum(counter.values()), 1)
            for val, cnt in counter.most_common(10):
                column_top_values.append({"value": val, "count": cnt, "ratio": cnt / total})

        bayes = obj.get("bayesian_context", {})
        prior_error_probability = 0.0
        posterior_error_probability = 0.0
        bayes_rule_signal = bayes_rarity_signal = bayes_pattern_signal = bayes_neighbor_signal = 0.0
        if isinstance(bayes, dict):
            prior_error_probability = safe_float(bayes.get("prior_error_probability", 0.0), 0.0)
            posterior_error_probability = safe_float(bayes.get("posterior_error_probability", 0.0), 0.0)
            be = bayes.get("bayes_evidence", {})
            if isinstance(be, dict):
                bayes_rule_signal = safe_float(be.get("rule_signal", 0.0), 0.0)
                bayes_rarity_signal = safe_float(be.get("rarity_signal", 0.0), 0.0)
                bayes_pattern_signal = safe_float(be.get("pattern_signal", 0.0), 0.0)
                bayes_neighbor_signal = safe_float(be.get("neighbor_signal", 0.0), 0.0)

        context_map[(row_id, column)] = {
            "row_id": row_id, "column": column, "value": value,
            "semantic_type": obj.get("semantic_type", None),
            "violation_count": obj.get("violation_count", None),
            "conflict_score": obj.get("conflict_score", None),
            "main_rule_type": obj.get("main_rule_type", None),
            "main_usage_role": obj.get("main_usage_role", None),
            "numeric_value": obj.get("numeric_value", None),
            "row_values": row_values,
            "suggested_correct_value": None if canonical_empty_text(obj.get("suggested_correct_value", "")) == "empty" else canonical_empty_text(obj.get("suggested_correct_value", "")),
            "neighbor_majority_value": neighbor_majority_value,
            "neighbor_majority_ratio": neighbor_majority_ratio,
            "similar_row_target_values": similar_row_target_values,
            "column_top_values": column_top_values,
            "alternative_values_from_column": alternative_values_from_column,
            "rule_expected_values": rule_expected_values,
            "expected_values_from_rules": expected_values_from_rules,
            "column_numeric_stats": column_numeric_stats,
            "column_detected_type": column_detected_type,
            "prior_error_probability": prior_error_probability,
            "posterior_error_probability": posterior_error_probability,
            "bayes_rule_signal": bayes_rule_signal,
            "bayes_rarity_signal": bayes_rarity_signal,
            "bayes_pattern_signal": bayes_pattern_signal,
            "bayes_neighbor_signal": bayes_neighbor_signal,
            **counters,
            "index_value": canonical_empty_text(row_values.get("index", "")),
            "id_value": canonical_empty_text(row_values.get("id", "")),
            "beer_name_value": canonical_empty_text(row_values.get("beer_name", "")),
            "style_value": canonical_empty_text(row_values.get("style", "")),
            "ounces_value": canonical_empty_text(row_values.get("ounces", "")),
            "abv_value": canonical_empty_text(row_values.get("abv", "")),
            "ibu_value": canonical_empty_text(row_values.get("ibu", "")),
            "brewery_id_value": canonical_empty_text(row_values.get("brewery_id", "")),
            "brewery_name_value": canonical_empty_text(row_values.get("brewery_name", "")),
            "city_value": canonical_empty_text(row_values.get("city", "")),
            "state_value": canonical_empty_text(row_values.get("state", "")),
        }

    return context_map


def merge_flat_into_context_map(context_map, flat_records):
    for obj in flat_records:
        if "row_id" not in obj or "column" not in obj:
            continue
        try:
            row_id = int(obj["row_id"])
        except Exception:
            continue
        column = str(obj["column"])
        key = (row_id, column)
        if key not in context_map:
            context_map[key] = {
                "row_id": row_id, "column": column, "value": canonical_empty_text(obj.get("value", "")),
                "semantic_type": obj.get("semantic_type", None),
                "violation_count": obj.get("violation_count", None),
                "conflict_score": obj.get("conflict_score", None),
                "main_rule_type": obj.get("main_rule_type", None),
                "main_usage_role": obj.get("main_usage_role", None),
                "numeric_value": obj.get("numeric_value", None),
                "row_values": {}, "suggested_correct_value": None,
                "neighbor_majority_value": canonical_empty_text(obj.get("neighbor_majority_value", "")),
                "neighbor_majority_ratio": safe_float(obj.get("neighbor_majority_ratio", 0.0), 0.0),
                "similar_row_target_values": [], "column_top_values": [],
                "alternative_values_from_column": [], "rule_expected_values": [], "expected_values_from_rules": [],
                "column_numeric_stats": None, "column_detected_type": obj.get("detected_type", None),
                "prior_error_probability": safe_float(obj.get("prior_error_probability", 0.0), 0.0),
                "posterior_error_probability": safe_float(obj.get("posterior_error_probability", 0.0), 0.0),
                "bayes_rule_signal": 0.0, "bayes_rarity_signal": 0.0,
                "bayes_pattern_signal": 0.0, "bayes_neighbor_signal": 0.0,
                "strong_rule_count": 0, "candidate_generation_rule_count": 0, "fd_like_count": 0,
                "global_rule_count": 0, "typo_rule_count": 0, "pattern_rule_count": 0,
                "schema_rule_count": 0, "numeric_window_rule_count": obj.get("numeric_window_rule_count", 0),
                "index_value": "", "id_value": "", "beer_name_value": "", "style_value": "",
                "ounces_value": "", "abv_value": "", "ibu_value": "",
                "brewery_id_value": "", "brewery_name_value": "", "city_value": "", "state_value": "",
            }


def build_rule_vote_counter(ctx):
    counter = Counter()
    for ev in ctx.get("rule_expected_values", []):
        ev = canonical_empty_text(ev)
        if ev not in {"", "empty"}:
            counter[ev] += 1
    for ev in ctx.get("expected_values_from_rules", []):
        ev = canonical_empty_text(ev)
        if ev not in {"", "empty"}:
            counter[ev] += 2
    return counter


def compute_rule_features(candidate_value, dirty_value, ctx):
    vote_counter = build_rule_vote_counter(ctx)
    total_votes = sum(vote_counter.values())
    cand_vote = vote_counter.get(canonical_empty_text(candidate_value), 0)
    dirty_vote = vote_counter.get(canonical_empty_text(dirty_value), 0)

    cand_fixed = normalize_by_column(ctx.get("column", ""), candidate_value)
    dirty_fixed = normalize_by_column(ctx.get("column", ""), dirty_value)
    if cand_fixed is not None:
        cand_vote = max(cand_vote, vote_counter.get(canonical_empty_text(cand_fixed), 0), vote_counter.get(norm_text(cand_fixed), 0))
    if dirty_fixed is not None:
        dirty_vote = max(dirty_vote, vote_counter.get(canonical_empty_text(dirty_fixed), 0), vote_counter.get(norm_text(dirty_fixed), 0))

    candidate_rule_support_ratio = cand_vote / total_votes if total_votes > 0 else 0.0
    dirty_rule_support_ratio = dirty_vote / total_votes if total_votes > 0 else 0.0

    alt_values = {canonical_empty_text(x.get("value", "")) for x in ctx.get("alternative_values_from_column", []) if isinstance(x, dict)}
    sim_row_values = [canonical_empty_text(v) for v in ctx.get("similar_row_target_values", [])]

    return {
        "rule_total_votes": total_votes,
        "candidate_rule_vote_count": cand_vote,
        "dirty_rule_vote_count": dirty_vote,
        "candidate_rule_support_ratio": candidate_rule_support_ratio,
        "dirty_rule_support_ratio": dirty_rule_support_ratio,
        "rule_support_gain": candidate_rule_support_ratio - dirty_rule_support_ratio,
        "candidate_matches_any_rule_expected": int(cand_vote > 0),
        "candidate_matches_any_strong_rule_expected": int(any(value_equal_for_column(ctx.get("column", ""), candidate_value, x) for x in ctx.get("rule_expected_values", []))),
        "candidate_matches_any_expected_values_from_rules": int(any(value_equal_for_column(ctx.get("column", ""), candidate_value, x) for x in ctx.get("expected_values_from_rules", []))),
        "candidate_in_alternative_values_from_column": int(any(value_equal_for_column(ctx.get("column", ""), candidate_value, x) for x in alt_values)),
        "candidate_equals_neighbor_majority": int(ctx.get("neighbor_majority_value", None) is not None and value_equal_for_column(ctx.get("column", ""), candidate_value, ctx.get("neighbor_majority_value"))),
        "candidate_equals_suggested_correct_value": int(ctx.get("suggested_correct_value", None) is not None and value_equal_for_column(ctx.get("column", ""), candidate_value, ctx.get("suggested_correct_value"))),
        "candidate_matches_similar_row_target": int(any(value_equal_for_column(ctx.get("column", ""), candidate_value, v) for v in sim_row_values)),
        "candidate_match_count_in_similar_rows": sum(1 for v in sim_row_values if value_equal_for_column(ctx.get("column", ""), candidate_value, v)),
        "candidate_match_ratio_in_similar_rows": (sum(1 for v in sim_row_values if value_equal_for_column(ctx.get("column", ""), candidate_value, v)) / len(sim_row_values) if sim_row_values else 0.0),
        "strong_rule_count": int(ctx.get("strong_rule_count", 0) or 0),
        "candidate_generation_rule_count": int(ctx.get("candidate_generation_rule_count", 0) or 0),
        "fd_like_count": int(ctx.get("fd_like_count", 0) or 0),
        "global_rule_count": int(ctx.get("global_rule_count", 0) or 0),
        "typo_rule_count": int(ctx.get("typo_rule_count", 0) or 0),
        "pattern_rule_count": int(ctx.get("pattern_rule_count", 0) or 0),
        "schema_rule_count": int(ctx.get("schema_rule_count", 0) or 0),
        "numeric_window_rule_count": int(ctx.get("numeric_window_rule_count", 0) or 0),
    }

def build_cooccurrence_from_raw_data(raw_df):
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
        norm_df[col] = norm_df[col].fillna("").astype(str).map(lambda x: str(x).strip())

    for target_col in usable_columns:
        cooccur[target_col] = {}
        target_series = norm_df[target_col]
        target_domain = sorted(set(v for v in target_series.tolist() if v != ""))
        domain_size = max(len(target_domain), 1)
        for ctx_col in usable_columns:
            if ctx_col == target_col:
                continue
            pair_counter = Counter()
            ctx_counter = Counter()
            for _, r in norm_df[[target_col, ctx_col]].iterrows():
                v, c = r[target_col], r[ctx_col]
                if v == "" or c == "":
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


def load_or_build_cooccurrence_dict():
    if COOCCURRENCE_DICT_FILE is not None and os.path.exists(COOCCURRENCE_DICT_FILE):
        print(f"[INFO] 使用现成共现字典: {COOCCURRENCE_DICT_FILE}")
        with open(COOCCURRENCE_DICT_FILE, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    if RAW_DATA_FILE is None or RAW_DATA_FILE == "" or not os.path.exists(RAW_DATA_FILE):
        print("[WARN] 没有现成共现字典，也没有可用原始数据文件，贝叶斯共现特征将退化为 0。")
        return None
    print(f"[INFO] 从原始数据集构建经验共现字典: {RAW_DATA_FILE}")
    raw_df = pd.read_csv(RAW_DATA_FILE, encoding="utf-8-sig")
    cooccur = build_cooccurrence_from_raw_data(raw_df)
    output_cooccur_path = resolve_output_path(OUTPUT_BUILT_COOCCUR_JSON, CANDIDATES_FILE)
    with open(output_cooccur_path, "w", encoding="utf-8-sig") as f:
        json.dump(cooccur, f, ensure_ascii=False)
    print(f"[INFO] 已保存构建出的共现字典: {output_cooccur_path}")
    return cooccur


def extract_context_pairs_from_ctx(ctx):
    row_values = ctx.get("row_values", {})
    target_column = ctx.get("column")
    pairs = []
    if isinstance(row_values, dict) and row_values:
        for k, v in row_values.items():
            if str(k) == str(target_column):
                continue
            vv = canonical_empty_text(v)
            if vv in {"", "empty"}:
                continue
            pairs.append((str(k), vv))
    return pairs[:MAX_CONTEXT_PAIRS]


def compute_bayes_features(candidate_value, ctx, external_cooccur):
    context_pairs = extract_context_pairs_from_ctx(ctx)
    base = {
        "prior_error_probability": safe_float(ctx.get("prior_error_probability", 0.0), 0.0),
        "posterior_error_probability": safe_float(ctx.get("posterior_error_probability", 0.0), 0.0),
        "bayes_rule_signal": safe_float(ctx.get("bayes_rule_signal", 0.0), 0.0),
        "bayes_rarity_signal": safe_float(ctx.get("bayes_rarity_signal", 0.0), 0.0),
        "bayes_pattern_signal": safe_float(ctx.get("bayes_pattern_signal", 0.0), 0.0),
        "bayes_neighbor_signal": safe_float(ctx.get("bayes_neighbor_signal", 0.0), 0.0),
    }
    if not context_pairs or external_cooccur is None:
        return {
            "context_pair_count_used": 0, "bayes_max_cond_prob": 0.0, "bayes_min_cond_prob": 0.0,
            "bayes_mean_cond_prob": 0.0, "bayes_prod_logprob": 0.0, "bayes_nonzero_pair_count": 0,
            **base,
        }
    target_column = ctx.get("column")
    col_map = external_cooccur.get(target_column, {})
    cand_map = col_map.get(candidate_value, {})
    fixed = normalize_by_column(target_column, candidate_value)
    if not cand_map and fixed is not None:
        cand_map = col_map.get(fixed, {})
    probs = []
    for ctx_col, ctx_val in context_pairs:
        probs.append(safe_float(cand_map.get(f"{ctx_col}={ctx_val}", 0.0), 0.0))
    nonzero = [p for p in probs if p > 0]
    return {
        "context_pair_count_used": len(probs),
        "bayes_max_cond_prob": max(probs) if probs else 0.0,
        "bayes_min_cond_prob": min(probs) if probs else 0.0,
        "bayes_mean_cond_prob": sum(probs) / len(probs) if probs else 0.0,
        "bayes_prod_logprob": sum(math.log(max(p, 1e-12)) for p in probs) if probs else 0.0,
        "bayes_nonzero_pair_count": len(nonzero),
        **base,
    }


def compute_transform_features(candidate_value, dirty_value):
    sim = string_similarity(candidate_value, dirty_value)
    phon_sim = phonetic_similarity(candidate_value, dirty_value)
    key_dist = avg_keyboard_distance_for_same_length(candidate_value, dirty_value)
    len_a, len_b = len(str(candidate_value)), len(str(dirty_value))
    num_a, num_b = try_parse_float(candidate_value), try_parse_float(dirty_value)
    if ENABLE_NUMERIC_CANDIDATE_CHECK and num_a is not None and num_b is not None:
        both_numeric = 1
        numeric_abs_diff = abs(num_a - num_b)
        numeric_rel_diff = abs(num_a - num_b) / max(abs(num_b), 1e-9)
    else:
        both_numeric = 0
        numeric_abs_diff = None
        numeric_rel_diff = None
    return {
        "edit_similarity_to_dirty": sim, "edit_distance_proxy": 1.0 - sim,
        "phonetic_similarity_to_dirty": phon_sim, "phonetic_distance_proxy": 1.0 - phon_sim,
        "avg_keyboard_distance": key_dist, "length_diff": abs(len_a - len_b),
        "both_numeric": both_numeric, "numeric_abs_diff": numeric_abs_diff, "numeric_rel_diff": numeric_rel_diff,
    }


def compute_beers_plausibility_features(candidate_value, dirty_value, ctx):
    column = norm_text(ctx.get("column", ""))
    row_values = ctx.get("row_values", {}) or {}
    fixed_candidate = normalize_by_column(column, candidate_value)
    fixed_dirty = normalize_by_column(column, dirty_value)
    city_value = canonical_empty_text(row_values.get("city", ctx.get("city_value", "")))
    state_value = canonical_empty_text(row_values.get("state", ctx.get("state_value", "")))
    state_from_city = normalize_state_candidate(city_value)
    city_without_state = normalize_city_candidate(city_value)
    candidate_state = normalize_state_candidate(candidate_value)
    dirty_state = normalize_state_candidate(dirty_value)
    candidate_city = normalize_city_candidate(candidate_value)
    dirty_city = normalize_city_candidate(dirty_value)
    candidate_ounces = normalize_ounces_candidate(candidate_value)
    dirty_ounces = normalize_ounces_candidate(dirty_value)
    candidate_abv = normalize_abv_candidate(candidate_value)
    dirty_abv = normalize_abv_candidate(dirty_value)
    candidate_ibu = normalize_ibu_candidate(candidate_value)
    dirty_ibu = normalize_ibu_candidate(dirty_value)
    candidate_id = normalize_id_candidate(candidate_value)
    dirty_id = normalize_id_candidate(dirty_value)

    cand_abv_float = try_parse_float(candidate_abv) if candidate_abv is not None else None
    dirty_abv_float = try_parse_float(dirty_abv) if dirty_abv is not None else None
    cand_ibu_float = try_parse_float(candidate_ibu) if candidate_ibu is not None else None
    dirty_ibu_float = try_parse_float(dirty_ibu) if dirty_ibu is not None else None

    return {
        "beers_column_is_state": int(column == "state"),
        "beers_column_is_city": int(column == "city"),
        "beers_column_is_ounces": int(column == "ounces"),
        "beers_column_is_abv": int(column == "abv"),
        "beers_column_is_ibu": int(column == "ibu"),
        "beers_column_is_brewery_id": int(column == "brewery_id"),
        "beers_column_is_brewery_name": int(column == "brewery_name"),
        "beers_column_is_beer_name": int(column == "beer_name"),
        "beers_column_is_style": int(column == "style"),
        "beers_candidate_is_valid_state": int(candidate_state is not None),
        "beers_dirty_is_valid_state": int(dirty_state is not None),
        "beers_candidate_is_valid_city": int(candidate_city is not None),
        "beers_dirty_is_valid_city": int(dirty_city is not None),
        "beers_candidate_is_valid_ounces": int(candidate_ounces is not None),
        "beers_dirty_is_valid_ounces": int(dirty_ounces is not None),
        "beers_candidate_is_valid_abv": int(candidate_abv is not None),
        "beers_dirty_is_valid_abv": int(dirty_abv is not None),
        "beers_candidate_is_valid_ibu": int(candidate_ibu is not None),
        "beers_dirty_is_valid_ibu": int(dirty_ibu is not None),
        "beers_candidate_is_valid_id": int(candidate_id is not None),
        "beers_dirty_is_valid_id": int(dirty_id is not None),
        "beers_candidate_state_matches_city_suffix": int(column == "state" and candidate_state is not None and state_from_city is not None and candidate_state == state_from_city),
        "beers_dirty_state_matches_city_suffix": int(column == "state" and dirty_state is not None and state_from_city is not None and dirty_state == state_from_city),
        "beers_state_city_suffix_gain": int(column == "state" and candidate_state is not None and state_from_city is not None and candidate_state == state_from_city) - int(column == "state" and dirty_state is not None and state_from_city is not None and dirty_state == state_from_city),
        "beers_candidate_city_equals_city_without_state": int(column == "city" and city_without_state is not None and norm_text(candidate_value) == norm_text(city_without_state)),
        "beers_dirty_city_has_state_suffix": int(column == "city" and strip_state_suffix_from_city(dirty_value) is not None),
        "beers_candidate_normalized_equals_dirty": int(fixed_candidate is not None and fixed_dirty is not None and norm_text(fixed_candidate) == norm_text(fixed_dirty)),
        "beers_candidate_is_format_restoration": int(fixed_candidate is not None and norm_text(fixed_candidate) != norm_text(candidate_value)),
        "beers_abv_abs_diff_after_norm": abs(cand_abv_float - dirty_abv_float) if cand_abv_float is not None and dirty_abv_float is not None else None,
        "beers_ibu_abs_diff_after_norm": abs(cand_ibu_float - dirty_ibu_float) if cand_ibu_float is not None and dirty_ibu_float is not None else None,
        "beers_candidate_matches_row_state": int(candidate_state is not None and normalize_state_candidate(state_value) == candidate_state),
        "beers_candidate_matches_row_city": int(candidate_city is not None and norm_text(candidate_city) == norm_text(normalize_city_candidate(city_value))),
    }


def compute_column_plausibility_features(candidate_value, dirty_value, ctx):
    column_top_values = ctx.get("column_top_values", [])
    top_counter = {canonical_empty_text(x["value"]): x for x in column_top_values if isinstance(x, dict)}
    candidate_norm = canonical_empty_text(candidate_value)
    candidate_in_top_values = int(candidate_norm in top_counter)
    candidate_top_ratio = safe_float(top_counter.get(candidate_norm, {}).get("ratio", 0.0), 0.0)
    detected_type = str(ctx.get("column_detected_type", "") or "").lower()
    num_stats = ctx.get("column_numeric_stats", None)
    candidate_in_numeric_range = None
    candidate_outside_numeric_range = None
    if detected_type == "numeric" and isinstance(num_stats, dict):
        cand_num = try_parse_float(candidate_value)
        if cand_num is not None:
            min_v, max_v = try_parse_float(num_stats.get("min")), try_parse_float(num_stats.get("max"))
            if min_v is not None and max_v is not None:
                candidate_in_numeric_range = int(min_v <= cand_num <= max_v)
                candidate_outside_numeric_range = int(not (min_v <= cand_num <= max_v))
    out = {
        "candidate_in_column_top_values": candidate_in_top_values,
        "candidate_column_top_ratio": candidate_top_ratio,
        "candidate_in_numeric_range": candidate_in_numeric_range,
        "candidate_outside_numeric_range": candidate_outside_numeric_range,
    }
    out.update(compute_beers_plausibility_features(candidate_value, dirty_value, ctx))
    return out

def main():
    if not path_exists_nonempty(CANDIDATES_FILE):
        raise FileNotFoundError(f"候选展开文件不存在: {CANDIDATES_FILE}")
    if not path_exists_nonempty(CONTEXT_FILE):
        raise FileNotFoundError(f"上下文文件不存在: {CONTEXT_FILE}")

    candidates_df = pd.read_csv(CANDIDATES_FILE, encoding="utf-8-sig")
    if not {"row_id", "column", "dirty_value", "candidate_value"}.issubset(set(candidates_df.columns)):
        raise ValueError("候选文件必须至少包含 row_id,column,dirty_value,candidate_value")
    candidates_df["row_id"] = candidates_df["row_id"].astype(int)

    context_records = load_context_records(CONTEXT_FILE)
    context_map = build_context_map_rich(context_records)
    print(f"[INFO] 上下文记录数: {len(context_records)}")
    print(f"[INFO] 可索引上下文条数: {len(context_map)}")

    if str(FALLBACK_CONTEXT_FLAT_FILE).strip():
        if path_exists_nonempty(FALLBACK_CONTEXT_FLAT_FILE):
            flat_records = load_context_records(FALLBACK_CONTEXT_FLAT_FILE)
            merge_flat_into_context_map(context_map, flat_records)
            print(f"[INFO] fallback flat 上下文记录数: {len(flat_records)}")
        else:
            print(f"[WARN] FALLBACK_CONTEXT_FLAT_FILE 不存在，已跳过: {FALLBACK_CONTEXT_FLAT_FILE}")

    cooccur_dict = load_or_build_cooccurrence_dict()
    feature_rows, missing_context_count = [], 0

    for _, row in candidates_df.iterrows():
        row_id, column = int(row["row_id"]), str(row["column"])
        dirty_value = canonical_empty_text(row["dirty_value"])
        candidate_value = canonical_empty_text(row["candidate_value"])
        ctx = context_map.get((row_id, column), None)
        if ctx is None:
            missing_context_count += 1
            continue

        candidate_source_text = str(row.get("candidate_source", "") or "")
        sources = set([s.strip() for s in candidate_source_text.split("|") if s.strip()])
        ctx_for_feature = dict(ctx)
        ctx_for_feature["candidate_source_text"] = candidate_source_text

        rule_feats = compute_rule_features(candidate_value, dirty_value, ctx_for_feature)
        bayes_feats = compute_bayes_features(candidate_value, ctx_for_feature, cooccur_dict)
        trans_feats = compute_transform_features(candidate_value, dirty_value)
        plaus_feats = compute_column_plausibility_features(candidate_value, dirty_value, ctx_for_feature)

        out = row.to_dict()
        out.update({
            "semantic_type": ctx.get("semantic_type", None),
            "violation_count": ctx.get("violation_count", None),
            "conflict_score": ctx.get("conflict_score", None),
            "main_rule_type": ctx.get("main_rule_type", None),
            "main_usage_role": ctx.get("main_usage_role", None),
            "numeric_value": ctx.get("numeric_value", None),
            "numeric_window_rule_count": ctx.get("numeric_window_rule_count", None),
            "has_row_context": int(bool(ctx.get("row_values", {}))),
            "neighbor_majority_value": ctx.get("neighbor_majority_value", None),
            "neighbor_majority_ratio": safe_float(ctx.get("neighbor_majority_ratio", 0.0), 0.0),
            "suggested_correct_value": ctx.get("suggested_correct_value", None),
            "source_score": safe_float(row.get("source_score", 0.0), 0.0),
            "final_score_from_candidate_gen": safe_float(row.get("final_score", 0.0), 0.0),
            "source_count": int(row.get("source_count", 0) or 0),
            "is_multi_source_candidate": int(int(row.get("source_count", 0) or 0) >= 2),
            "candidate_source_text": candidate_source_text,
            "is_rule_and_dictionary_supported": int(int(row.get("is_from_rule", 0) or 0) == 1 and int(row.get("is_from_dictionary", 0) or 0) == 1),
            "is_neighbor_and_rule_agree": int(int(row.get("is_from_neighbor", 0) or 0) == 1 and int(row.get("is_from_rule", 0) or 0) == 1),
            "is_pattern_restore_candidate": int(int(row.get("is_from_pattern_restore", 0) or 0) == 1),
            "contains_expected_values_from_rules_source": int("expected_values_from_rules" in sources),
            "contains_similar_row_target_source": int("similar_row_target" in sources),
            "contains_beers_dictionary_source": int(any(s in sources for s in {
                "brewery_id_to_state", "brewery_name_to_state", "city_to_state",
                "brewery_id_to_city", "brewery_name_to_city", "brewery_id_to_name",
                "brewery_name_to_id", "beer_id_to_name", "beer_name_to_style",
                "beer_name_to_ounces", "style_to_ounces", "beer_name_to_abv",
                "style_to_abv", "beer_name_to_ibu", "style_to_ibu",
            })),
            "contains_beers_pattern_restore_source": int(any(s in sources for s in {
                "state_from_city_suffix", "pattern_restore_city_strip_state",
                "pattern_restore_ounces", "pattern_restore_abv", "pattern_restore_ibu", "pattern_restore_id",
            })),
            "contains_brewery_source": int(any(s in sources for s in {
                "brewery_id_to_state", "brewery_name_to_state", "brewery_id_to_city",
                "brewery_name_to_city", "brewery_id_to_name", "brewery_name_to_id",
            })),
            "contains_beer_source": int(any(s in sources for s in {
                "beer_id_to_name", "beer_name_to_style", "beer_name_to_ounces",
                "beer_name_to_abv", "beer_name_to_ibu",
            })),
            "contains_style_source": int(any(s in sources for s in {"style_to_ounces", "style_to_abv", "style_to_ibu"})),
            "contains_state_city_source": int(any(s in sources for s in {"state_from_city_suffix", "city_to_state", "pattern_restore_city_strip_state"})),
        })

        for col in [
            "beers_is_state_candidate", "beers_is_city_candidate", "beers_is_ounces_candidate",
            "beers_is_abv_candidate", "beers_is_ibu_candidate",
        ]:
            if col in row.index:
                out[col] = row.get(col)

        out.update(rule_feats)
        out.update(bayes_feats)
        out.update(trans_feats)
        out.update(plaus_feats)
        feature_rows.append(out)

    features_df = pd.DataFrame(feature_rows)
    output_path = resolve_output_path(OUTPUT_FEATURES_CSV, CANDIDATES_FILE)
    features_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"[INFO] 候选记录数: {len(candidates_df)}")
    print(f"[INFO] 成功构造特征的记录数: {len(features_df)}")
    print(f"[INFO] 缺失上下文的候选记录数: {missing_context_count}")
    print(f"[INFO] 输出文件: {output_path}")

    if not features_df.empty:
        show_cols = [c for c in [
            "row_id", "column", "dirty_value", "candidate_value",
            "candidate_rule_support_ratio", "rule_support_gain",
            "bayes_mean_cond_prob", "bayes_max_cond_prob",
            "edit_similarity_to_dirty", "avg_keyboard_distance", "phonetic_similarity_to_dirty",
            "candidate_in_column_top_values", "is_rule_and_dictionary_supported",
            "candidate_matches_similar_row_target", "contains_beers_dictionary_source",
            "contains_beers_pattern_restore_source", "contains_brewery_source",
            "contains_beer_source", "contains_style_source", "contains_state_city_source",
            "beers_candidate_is_valid_state", "beers_candidate_state_matches_city_suffix",
            "beers_state_city_suffix_gain", "beers_candidate_is_valid_ounces",
            "beers_candidate_is_valid_abv", "beers_candidate_is_valid_ibu",
            "beers_candidate_is_format_restoration", "beers_abv_abs_diff_after_norm",
            "beers_ibu_abs_diff_after_norm",
        ] if c in features_df.columns]
        print("\n[INFO] 前 20 行特征预览：")
        print(features_df[show_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()

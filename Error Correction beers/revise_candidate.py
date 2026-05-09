
import os
import json
import math
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher

import pandas as pd


# ============================================================
# 1. 配置区
# ============================================================

ERROR_PRED_FILE = "/mnt/mydata/dq/projects/Splittree/test/Error Correction beers/predicted_error_positions.csv"
CONTEXT_FILE = "/mnt/mydata/dq/projects/Splittree/test/Error Detection beers/candidate_llm_contexts_beers.jsonl"
FALLBACK_CONTEXT_FLAT_FILE = "/mnt/mydata/dq/projects/Splittree/test/Error Detection beers/candidate_llm_contexts_flat_beers.csv"
DIRTY_TABLE_CSV = "/mnt/mydata/dq/projects/Splittree/beers/beers_dirty.csv"

OUTPUT_CANDIDATES_EXPANDED_CSV = "repair_candidates_expanded.csv"
OUTPUT_CANDIDATES_GROUPED_CSV = "repair_candidates_grouped.csv"

TOP_K_COLUMN_VALUES = 12
TOP_K_SIMILAR_ROW_VALUES = 8
MAX_CANDIDATES_PER_CELL = 25
MIN_STRING_SIM_FOR_COLUMN_VALUE = 0.45
INCLUDE_DIRTY_VALUE = False

EMPTY_TOKENS = {"", "empty", "null", "none", "nan", "na", "n/a", "missing"}


# ============================================================
# 2. 通用工具
# ============================================================

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


def resolve_output_path(path):
    if os.path.isabs(str(path)):
        return str(path)
    return os.path.abspath(str(path))


def string_similarity(a, b):
    a = norm_text(a)
    b = norm_text(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


# ============================================================
# 3. Beers 专属规范化
# ============================================================

US_STATE_CODES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC"
}

STATE_NAME_TO_CODE = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN",
    "mississippi": "MS", "missouri": "MO", "montana": "MT", "nebraska": "NE",
    "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ",
    "new mexico": "NM", "new york": "NY", "north carolina": "NC",
    "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR",
    "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
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

    # city 字段常见：Oklahoma City OK / Ashland OR
    m = re.search(r"\b([A-Za-z]{2})\b$", s.strip())
    if m:
        cand = m.group(1).upper()
        if cand in US_STATE_CODES:
            return cand

    return None


def strip_state_suffix_from_city(city):
    """
    Beers 中 city 字段有时会混入州缩写：
    - Oklahoma City OK -> Oklahoma City
    - Ashland OR -> Ashland
    """
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

    stripped = strip_state_suffix_from_city(s)
    if stripped:
        return stripped

    return s.strip()


def normalize_ounces_candidate(x):
    s = canonical_empty_text(x).lower().strip()
    if s in {"", "empty"}:
        return None

    s = s.replace("ounces", "ounce")
    s = s.replace("ozs", "oz")
    s = s.replace("0z", "oz")
    s = s.replace("o.z.", "oz")
    s = re.sub(r"\s+", " ", s)

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
    if not m:
        return None
    return str(int(m.group(0)))


def normalize_by_column(column, value):
    col = norm_text(column)
    if col == "state":
        return normalize_state_candidate(value)
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


# ============================================================
# 4. 输入读取
# ============================================================

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
    lower = str(path).lower()
    if lower.endswith(".jsonl"):
        return load_jsonl(path)
    if lower.endswith(".csv") or lower.endswith(".txt"):
        return pd.read_csv(path, encoding="utf-8-sig").to_dict(orient="records")

    try:
        return load_jsonl(path)
    except Exception:
        return pd.read_csv(path, encoding="utf-8-sig").to_dict(orient="records")


# ============================================================
# 5. 构建全局字典
# ============================================================

def build_global_dictionaries(context_records, dirty_table_df=None):
    column_value_counter = defaultdict(Counter)

    brewery_id_to_state = defaultdict(Counter)
    brewery_name_to_state = defaultdict(Counter)
    city_to_state = defaultdict(Counter)

    brewery_id_to_city = defaultdict(Counter)
    brewery_name_to_city = defaultdict(Counter)
    brewery_id_to_name = defaultdict(Counter)
    brewery_name_to_id = defaultdict(Counter)

    beer_id_to_name = defaultdict(Counter)
    beer_name_to_style = defaultdict(Counter)
    beer_name_to_ounces = defaultdict(Counter)
    style_to_ounces = defaultdict(Counter)

    # Beers 数值属性映射：用于补 abv / ibu 候选
    beer_name_to_abv = defaultdict(Counter)
    style_to_abv = defaultdict(Counter)
    beer_name_to_ibu = defaultdict(Counter)
    style_to_ibu = defaultdict(Counter)

    def consume(row_vals):
        if not isinstance(row_vals, dict):
            return

        row = {str(k): canonical_empty_text(v) for k, v in row_vals.items()}

        for c, v in row.items():
            if v != "":
                column_value_counter[str(c)][v] += 1

        brewery_id = row.get("brewery_id", "")
        brewery_name = row.get("brewery_name", "")
        city = row.get("city", "")
        state = row.get("state", "")
        beer_id = row.get("id", "")
        beer_name = row.get("beer_name", "")
        style = row.get("style", "")
        ounces = row.get("ounces", "")
        abv = row.get("abv", "")
        ibu = row.get("ibu", "")

        state_norm = normalize_state_candidate(state) or normalize_state_candidate(city)
        if brewery_id and state_norm:
            brewery_id_to_state[brewery_id][state_norm] += 1
        if brewery_name and state_norm:
            brewery_name_to_state[norm_text(brewery_name)][state_norm] += 1
        if city and state_norm:
            city_to_state[norm_text(city)][state_norm] += 1

        city_norm = normalize_city_candidate(city)
        if brewery_id and city_norm:
            brewery_id_to_city[brewery_id][city_norm] += 1
        if brewery_name and city_norm:
            brewery_name_to_city[norm_text(brewery_name)][city_norm] += 1
        if brewery_id and brewery_name:
            brewery_id_to_name[brewery_id][brewery_name] += 1
            brewery_name_to_id[norm_text(brewery_name)][brewery_id] += 1

        if beer_id and beer_name:
            beer_id_to_name[beer_id][beer_name] += 1
        if beer_name and style:
            beer_name_to_style[norm_text(beer_name)][style] += 1

        oz = normalize_ounces_candidate(ounces)
        if oz and beer_name:
            beer_name_to_ounces[norm_text(beer_name)][oz] += 1
        if oz and style:
            style_to_ounces[norm_text(style)][oz] += 1

        abv_norm = normalize_abv_candidate(abv)
        if abv_norm and beer_name:
            beer_name_to_abv[norm_text(beer_name)][abv_norm] += 1
        if abv_norm and style:
            style_to_abv[norm_text(style)][abv_norm] += 1

        ibu_norm = normalize_ibu_candidate(ibu)
        if ibu_norm and beer_name:
            beer_name_to_ibu[norm_text(beer_name)][ibu_norm] += 1
        if ibu_norm and style:
            style_to_ibu[norm_text(style)][ibu_norm] += 1

    for obj in context_records:
        row_ctx = obj.get("row_context", {})
        if isinstance(row_ctx, dict):
            consume(row_ctx.get("row_values", {}))

        sim_ctx = obj.get("similarity_context", {})
        if isinstance(sim_ctx, dict):
            for item in sim_ctx.get("topk_similar_rows", []) or []:
                if isinstance(item, dict):
                    consume(item.get("row_values", {}))

        col_ctx = obj.get("column_context", {})
        if isinstance(col_ctx, dict):
            col = str(col_ctx.get("column", obj.get("column", "")))
            for t in col_ctx.get("top_values_with_counts", []) or []:
                if isinstance(t, dict) and t.get("value", None) is not None:
                    val = canonical_empty_text(t.get("value"))
                    cnt = int(t.get("count", 1) or 1)
                    if col and val:
                        column_value_counter[col][val] += cnt

        if "column" in obj and "value" in obj:
            col = str(obj.get("column"))
            val = canonical_empty_text(obj.get("value"))
            if col and val:
                column_value_counter[col][val] += 1

    if dirty_table_df is not None:
        for _, r in dirty_table_df.iterrows():
            consume({c: r[c] for c in dirty_table_df.columns})

    return {
        "column_value_counter": column_value_counter,
        "brewery_id_to_state": brewery_id_to_state,
        "brewery_name_to_state": brewery_name_to_state,
        "city_to_state": city_to_state,
        "brewery_id_to_city": brewery_id_to_city,
        "brewery_name_to_city": brewery_name_to_city,
        "brewery_id_to_name": brewery_id_to_name,
        "brewery_name_to_id": brewery_name_to_id,
        "beer_id_to_name": beer_id_to_name,
        "beer_name_to_style": beer_name_to_style,
        "beer_name_to_ounces": beer_name_to_ounces,
        "style_to_ounces": style_to_ounces,
        "beer_name_to_abv": beer_name_to_abv,
        "style_to_abv": style_to_abv,
        "beer_name_to_ibu": beer_name_to_ibu,
        "style_to_ibu": style_to_ibu,
    }


# ============================================================
# 6. 构建 context_map
# ============================================================

def build_context_map_rich(context_records, global_dict):
    context_map = {}
    column_value_counter = global_dict["column_value_counter"]

    for obj in context_records:
        if "row_id" not in obj or "column" not in obj:
            continue

        try:
            row_id = int(obj["row_id"])
        except Exception:
            continue

        column = str(obj["column"])
        value_str = canonical_empty_text(obj.get("value", ""))

        rule_expected_values = []
        alternative_values_from_column = []
        similar_row_target_values = []

        conflict_context = obj.get("conflict_context", {})
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
                        rule_expected_values.append(ev)

            for item in (conflict_context.get("alternative_values_from_column", []) or []):
                if isinstance(item, dict):
                    val = canonical_empty_text(item.get("value", None))
                    if val not in {"", "empty"}:
                        alternative_values_from_column.append({
                            "value": val,
                            "count": int(item.get("count", 1) or 1),
                            "ratio": safe_float(item.get("ratio", 0.0), 0.0),
                        })

            for item in (conflict_context.get("all_alternative_values", []) or []):
                if isinstance(item, dict):
                    val = canonical_empty_text(item.get("value", None))
                    if val not in {"", "empty"}:
                        alternative_values_from_column.append({
                            "value": val,
                            "count": int(item.get("count", 1) or 1),
                            "ratio": safe_float(item.get("ratio", 0.0), 0.0),
                        })

        neighbor_majority_value = ""
        neighbor_majority_ratio = 0.0
        sim_ctx = obj.get("similarity_context", {})
        if isinstance(sim_ctx, dict):
            neighbor_majority_value = canonical_empty_text(sim_ctx.get("neighbor_majority_value", ""))
            neighbor_majority_ratio = safe_float(sim_ctx.get("neighbor_majority_ratio", 0.0), 0.0)
            for item in sim_ctx.get("topk_similar_rows", [])[:TOP_K_SIMILAR_ROW_VALUES]:
                if isinstance(item, dict):
                    tv = canonical_empty_text(item.get("target_column_value", ""))
                    if tv not in {"", "empty"}:
                        similar_row_target_values.append(tv)

        column_top_values = []
        col_ctx = obj.get("column_context", {})
        if isinstance(col_ctx, dict):
            for item in col_ctx.get("top_values_with_counts", [])[:TOP_K_COLUMN_VALUES]:
                if isinstance(item, dict) and item.get("value", None) is not None:
                    val = canonical_empty_text(item.get("value"))
                    if val not in {"", "empty"}:
                        column_top_values.append({
                            "value": val,
                            "count": int(item.get("count", 0) or 0),
                            "ratio": safe_float(item.get("ratio", 0.0), 0.0),
                        })

        if not column_top_values:
            counter = column_value_counter.get(column, Counter())
            total = max(sum(counter.values()), 1)
            for val, cnt in counter.most_common(TOP_K_COLUMN_VALUES):
                column_top_values.append({"value": val, "count": cnt, "ratio": cnt / total})

        row_values = {}
        row_ctx = obj.get("row_context", {})
        if isinstance(row_ctx, dict):
            rv = row_ctx.get("row_values", {})
            if isinstance(rv, dict):
                row_values = {str(k): canonical_empty_text(v) for k, v in rv.items()}

        context_map[(row_id, column)] = {
            "row_id": row_id,
            "column": column,
            "value": value_str,
            "suggested_correct_value": canonical_empty_text(obj.get("suggested_correct_value", "")),
            "neighbor_majority_value": neighbor_majority_value,
            "neighbor_majority_ratio": neighbor_majority_ratio,
            "value_frequency": obj.get("value_frequency", None),
            "value_frequency_rank": obj.get("value_frequency_rank", None),
            "numeric_value": obj.get("numeric_value", None),
            "semantic_type": obj.get("semantic_type", None),
            "main_rule_type": obj.get("main_rule_type", None),
            "main_usage_role": obj.get("main_usage_role", None),
            "violation_count": obj.get("violation_count", None),
            "conflict_score": obj.get("conflict_score", None),
            "numeric_window_rule_count": obj.get("numeric_window_rule_count", None),
            "rule_expected_values": rule_expected_values,
            "column_top_values": column_top_values,
            "alternative_values_from_column": alternative_values_from_column,
            "similar_row_target_values": similar_row_target_values,
            "row_values": row_values,
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
                "row_id": row_id,
                "column": column,
                "value": canonical_empty_text(obj.get("value", "")),
                "suggested_correct_value": canonical_empty_text(obj.get("suggested_correct_value", "")),
                "neighbor_majority_value": canonical_empty_text(obj.get("neighbor_majority_value", "")),
                "neighbor_majority_ratio": safe_float(obj.get("neighbor_majority_ratio", 0.0), 0.0),
                "value_frequency": obj.get("value_frequency", None),
                "value_frequency_rank": obj.get("value_frequency_rank", None),
                "numeric_value": obj.get("numeric_value", None),
                "semantic_type": obj.get("semantic_type", None),
                "main_rule_type": obj.get("main_rule_type", None),
                "main_usage_role": obj.get("main_usage_role", None),
                "violation_count": obj.get("violation_count", None),
                "conflict_score": obj.get("conflict_score", None),
                "numeric_window_rule_count": obj.get("numeric_window_rule_count", None),
                "rule_expected_values": [],
                "column_top_values": [],
                "alternative_values_from_column": [],
                "similar_row_target_values": [],
                "row_values": {},
            }


# ============================================================
# 7. 候选生成
# ============================================================

def add_candidate(candidate_dict, value, source, score, extra=None):
    if value is None:
        return
    value = str(value).strip()
    if value == "":
        return

    key = norm_text(value)
    if key == "":
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
            "extra_info": [],
        }

    item = candidate_dict[key]
    item["candidate_sources"].add(source)
    item["source_score"] = max(item["source_score"], float(score))

    if source in {"rule_expected_value", "expected_values_from_rules"}:
        item["is_from_rule"] = 1
    elif source in {"neighbor_majority", "similar_row_target"}:
        item["is_from_neighbor"] = 1
    elif source in {"column_top_value", "alternative_values_from_column"}:
        item["is_from_column_top"] = 1
    elif source in {"similar_column_value"}:
        item["is_from_similarity"] = 1
    elif source in {"suggested_correct_value"}:
        item["is_from_hint"] = 1
    elif source in {
        "brewery_id_to_state", "brewery_name_to_state", "city_to_state",
        "brewery_id_to_city", "brewery_name_to_city",
        "brewery_id_to_name", "brewery_name_to_id",
        "beer_id_to_name", "beer_name_to_style",
        "beer_name_to_ounces", "style_to_ounces",
        "beer_name_to_abv", "style_to_abv",
        "beer_name_to_ibu", "style_to_ibu",
    }:
        item["is_from_dictionary"] = 1
    elif source in {
        "state_from_city_suffix", "pattern_restore_city_strip_state",
        "pattern_restore_ounces",
        "pattern_restore_abv", "pattern_restore_ibu", "pattern_restore_id",
    }:
        item["is_from_pattern_restore"] = 1

    if extra is not None:
        item["extra_info"].append(str(extra))


def counter_top(counter, topk=5):
    if not counter:
        return []
    return counter.most_common(topk)


def generate_beers_context_candidates(column, dirty_value, ctx, global_dict):
    cands = []
    col = norm_text(column)
    row_values = ctx.get("row_values", {}) or {}

    brewery_id = canonical_empty_text(row_values.get("brewery_id", ""))
    brewery_name = canonical_empty_text(row_values.get("brewery_name", ""))
    city = canonical_empty_text(row_values.get("city", ""))
    beer_id = canonical_empty_text(row_values.get("id", ""))
    beer_name = canonical_empty_text(row_values.get("beer_name", ""))
    style = canonical_empty_text(row_values.get("style", ""))

    if col == "state":
        state_from_city = normalize_state_candidate(city)
        if state_from_city and not value_equal_for_column(column, dirty_value, state_from_city):
            cands.append((state_from_city, "state_from_city_suffix", 0.96, f"city={city}"))

        for v, cnt in counter_top(global_dict["brewery_id_to_state"].get(brewery_id, Counter())):
            if not value_equal_for_column(column, dirty_value, v):
                cands.append((v, "brewery_id_to_state", 0.92, f"brewery_id={brewery_id}|cnt={cnt}"))

        for v, cnt in counter_top(global_dict["brewery_name_to_state"].get(norm_text(brewery_name), Counter())):
            if not value_equal_for_column(column, dirty_value, v):
                cands.append((v, "brewery_name_to_state", 0.90, f"brewery_name={brewery_name}|cnt={cnt}"))

        for v, cnt in counter_top(global_dict["city_to_state"].get(norm_text(city), Counter())):
            if not value_equal_for_column(column, dirty_value, v):
                cands.append((v, "city_to_state", 0.88, f"city={city}|cnt={cnt}"))

    elif col == "city":
        # city 字段本身可能带州缩写，先给格式恢复候选
        city_stripped = strip_state_suffix_from_city(dirty_value)
        if city_stripped and not value_equal_for_column(column, dirty_value, city_stripped):
            cands.append((city_stripped, "pattern_restore_city_strip_state", 0.94, "strip_state_suffix_from_city"))

        for v, cnt in counter_top(global_dict["brewery_id_to_city"].get(brewery_id, Counter())):
            if not value_equal_for_column(column, dirty_value, v):
                cands.append((v, "brewery_id_to_city", 0.88, f"brewery_id={brewery_id}|cnt={cnt}"))

        for v, cnt in counter_top(global_dict["brewery_name_to_city"].get(norm_text(brewery_name), Counter())):
            if not value_equal_for_column(column, dirty_value, v):
                cands.append((v, "brewery_name_to_city", 0.86, f"brewery_name={brewery_name}|cnt={cnt}"))

    elif col == "brewery_name":
        for v, cnt in counter_top(global_dict["brewery_id_to_name"].get(brewery_id, Counter())):
            if not value_equal_for_column(column, dirty_value, v):
                cands.append((v, "brewery_id_to_name", 0.90, f"brewery_id={brewery_id}|cnt={cnt}"))

    elif col == "brewery_id":
        for v, cnt in counter_top(global_dict["brewery_name_to_id"].get(norm_text(brewery_name), Counter())):
            if not value_equal_for_column(column, dirty_value, v):
                cands.append((v, "brewery_name_to_id", 0.90, f"brewery_name={brewery_name}|cnt={cnt}"))

    elif col == "beer_name":
        for v, cnt in counter_top(global_dict["beer_id_to_name"].get(beer_id, Counter())):
            if not value_equal_for_column(column, dirty_value, v):
                cands.append((v, "beer_id_to_name", 0.90, f"id={beer_id}|cnt={cnt}"))

    elif col == "style":
        for v, cnt in counter_top(global_dict["beer_name_to_style"].get(norm_text(beer_name), Counter())):
            if not value_equal_for_column(column, dirty_value, v):
                cands.append((v, "beer_name_to_style", 0.86, f"beer_name={beer_name}|cnt={cnt}"))

    elif col == "ounces":
        fixed = normalize_ounces_candidate(dirty_value)
        if fixed and not value_equal_for_column(column, dirty_value, fixed):
            cands.append((fixed, "pattern_restore_ounces", 0.95, "normalize_ounces"))

        for v, cnt in counter_top(global_dict["beer_name_to_ounces"].get(norm_text(beer_name), Counter())):
            if not value_equal_for_column(column, dirty_value, v):
                cands.append((v, "beer_name_to_ounces", 0.88, f"beer_name={beer_name}|cnt={cnt}"))

        for v, cnt in counter_top(global_dict["style_to_ounces"].get(norm_text(style), Counter())):
            if not value_equal_for_column(column, dirty_value, v):
                cands.append((v, "style_to_ounces", 0.74, f"style={style}|cnt={cnt}"))

    elif col == "abv":
        fixed = normalize_abv_candidate(dirty_value)
        if fixed and not value_equal_for_column(column, dirty_value, fixed):
            cands.append((fixed, "pattern_restore_abv", 0.92, "normalize_abv"))

        for v, cnt in counter_top(global_dict["beer_name_to_abv"].get(norm_text(beer_name), Counter())):
            if not value_equal_for_column(column, dirty_value, v):
                cands.append((v, "beer_name_to_abv", 0.86, f"beer_name={beer_name}|cnt={cnt}"))

        for v, cnt in counter_top(global_dict["style_to_abv"].get(norm_text(style), Counter())):
            if not value_equal_for_column(column, dirty_value, v):
                cands.append((v, "style_to_abv", 0.70, f"style={style}|cnt={cnt}"))

    elif col == "ibu":
        fixed = normalize_ibu_candidate(dirty_value)
        if fixed and not value_equal_for_column(column, dirty_value, fixed):
            cands.append((fixed, "pattern_restore_ibu", 0.92, "normalize_ibu"))

        for v, cnt in counter_top(global_dict["beer_name_to_ibu"].get(norm_text(beer_name), Counter())):
            if not value_equal_for_column(column, dirty_value, v):
                cands.append((v, "beer_name_to_ibu", 0.86, f"beer_name={beer_name}|cnt={cnt}"))

        for v, cnt in counter_top(global_dict["style_to_ibu"].get(norm_text(style), Counter())):
            if not value_equal_for_column(column, dirty_value, v):
                cands.append((v, "style_to_ibu", 0.68, f"style={style}|cnt={cnt}"))

    elif col in {"index", "id"}:
        fixed = normalize_id_candidate(dirty_value)
        if fixed and not value_equal_for_column(column, dirty_value, fixed):
            cands.append((fixed, "pattern_restore_id", 0.90, "normalize_id"))

    return cands


# ============================================================
# 8. 主流程
# ============================================================

def main():
    pred_df = pd.read_csv(ERROR_PRED_FILE, encoding="utf-8-sig")

    required_base = {"row_id", "column"}
    if not required_base.issubset(set(pred_df.columns)):
        raise ValueError(f"错误位置文件必须至少包含 {required_base}")

    pred_df["row_id"] = pd.to_numeric(pred_df["row_id"], errors="raise").astype(int)

    if "final_pred_label" in pred_df.columns:
        error_df = pred_df[pd.to_numeric(pred_df["final_pred_label"], errors="coerce").fillna(0).astype(int) == 1].copy()
    elif "final_pred_label_name" in pred_df.columns:
        error_df = pred_df[pred_df["final_pred_label_name"].astype(str).str.strip().str.lower() == "error"].copy()
    else:
        error_df = pred_df.copy()

    keep_cols = ["row_id", "column"]
    if "value" in error_df.columns:
        keep_cols.append("value")

    error_df = error_df[keep_cols].drop_duplicates().copy()
    print(f"[INFO] 预测错误单元格数量: {len(error_df)}")

    context_records = load_context_records(CONTEXT_FILE)
    print(f"[INFO] 主上下文记录数: {len(context_records)}")

    dirty_table_df = None
    if str(DIRTY_TABLE_CSV).strip():
        if path_exists_nonempty(DIRTY_TABLE_CSV):
            dirty_table_df = pd.read_csv(DIRTY_TABLE_CSV, encoding="utf-8-sig")
            dirty_table_df = dirty_table_df.apply(lambda col: col.map(canonical_empty_text))
            print(f"[INFO] 已读取 Beers 原始脏表: {DIRTY_TABLE_CSV}, shape={dirty_table_df.shape}")
        else:
            print(f"[WARN] DIRTY_TABLE_CSV 不存在，已跳过: {DIRTY_TABLE_CSV}")

    global_dict = build_global_dictionaries(context_records, dirty_table_df=dirty_table_df)
    context_map = build_context_map_rich(context_records, global_dict)

    if str(FALLBACK_CONTEXT_FLAT_FILE).strip():
        if path_exists_nonempty(FALLBACK_CONTEXT_FLAT_FILE):
            flat_records = load_context_records(FALLBACK_CONTEXT_FLAT_FILE)
            print(f"[INFO] fallback flat 上下文记录数: {len(flat_records)}")
            merge_flat_into_context_map(context_map, flat_records)
        else:
            print(f"[WARN] FALLBACK_CONTEXT_FLAT_FILE 不存在，已跳过: {FALLBACK_CONTEXT_FLAT_FILE}")

    print(f"[INFO] 可索引上下文条数: {len(context_map)}")

    expanded_rows = []
    grouped_rows = []
    missing_context_count = 0

    for _, row in error_df.iterrows():
        row_id = int(row["row_id"])
        column = str(row["column"])
        ctx = context_map.get((row_id, column), None)

        if ctx is None:
            missing_context_count += 1
            dirty_value = "" if "value" not in row or pd.isna(row["value"]) else str(row["value"]).strip()
            grouped_rows.append({
                "row_id": row_id,
                "column": column,
                "dirty_value": dirty_value,
                "candidate_count": 0,
                "candidates_joined": "",
                "sources_joined": "",
                "top_candidate": "",
                "top_candidate_score": None,
            })
            continue

        dirty_value = ctx.get("value", "")
        if ("value" in error_df.columns) and (pd.notna(row.get("value", None))) and str(row.get("value")).strip() != "":
            dirty_value = canonical_empty_text(row["value"])

        candidate_dict = {}

        if INCLUDE_DIRTY_VALUE and dirty_value != "":
            add_candidate(candidate_dict, dirty_value, "dirty_value", 0.01)

        # A. rule expected values
        expected_counter = Counter()
        for ev in ctx.get("rule_expected_values", []):
            ev = canonical_empty_text(ev)
            if ev in {"", "empty"} or value_equal_for_column(column, dirty_value, ev):
                continue
            fixed = normalize_by_column(column, ev)
            if fixed is not None:
                ev = fixed
            expected_counter[ev] += 1

        total_votes = sum(expected_counter.values())
        for ev, cnt in expected_counter.items():
            score = 0.76 + 0.22 * (cnt / max(total_votes, 1))
            add_candidate(candidate_dict, ev, "expected_values_from_rules", score, extra=f"rule_vote_count={cnt}")

        # B. neighbor majority
        neighbor_majority_value = ctx.get("neighbor_majority_value", None)
        if neighbor_majority_value and not value_equal_for_column(column, dirty_value, neighbor_majority_value):
            fixed = normalize_by_column(column, neighbor_majority_value)
            if fixed is not None:
                neighbor_majority_value = fixed
            nr = safe_float(ctx.get("neighbor_majority_ratio", 0.0), 0.0)
            add_candidate(candidate_dict, neighbor_majority_value, "neighbor_majority", 0.78 + 0.16 * min(nr, 1.0), extra=f"neighbor_ratio={nr:.4f}")

        # C. similar row target values
        sim_counter = Counter()
        for v in ctx.get("similar_row_target_values", []):
            v = canonical_empty_text(v)
            if v and not value_equal_for_column(column, dirty_value, v):
                fixed = normalize_by_column(column, v)
                if fixed is not None:
                    v = fixed
                sim_counter[v] += 1

        for v, cnt in sim_counter.items():
            score = 0.72 + 0.12 * min(cnt, TOP_K_SIMILAR_ROW_VALUES) / max(TOP_K_SIMILAR_ROW_VALUES, 1)
            add_candidate(candidate_dict, v, "similar_row_target", score, extra=f"count={cnt}")

        # D. suggested_correct_value
        suggested_correct_value = ctx.get("suggested_correct_value", None)
        if suggested_correct_value and not value_equal_for_column(column, dirty_value, suggested_correct_value):
            fixed = normalize_by_column(column, suggested_correct_value)
            if fixed is not None:
                suggested_correct_value = fixed
            add_candidate(candidate_dict, suggested_correct_value, "suggested_correct_value", 0.93)

        # E. column top / alternative values
        alt_items = []
        alt_items.extend(ctx.get("column_top_values", [])[:TOP_K_COLUMN_VALUES])
        alt_items.extend(ctx.get("alternative_values_from_column", [])[:TOP_K_COLUMN_VALUES])

        dedup_alt = {}
        for item in alt_items:
            if not isinstance(item, dict):
                continue

            val = canonical_empty_text(item.get("value", ""))
            if val in {"", "empty"} or value_equal_for_column(column, dirty_value, val):
                continue

            fixed = normalize_by_column(column, val)
            if fixed is not None:
                val = fixed

            k = norm_text(val)
            prev = dedup_alt.get(k, {"value": val, "count": 0, "ratio": 0.0})
            prev["count"] = max(prev["count"], int(item.get("count", 0) or 0))
            prev["ratio"] = max(prev["ratio"], safe_float(item.get("ratio", 0.0), 0.0))
            dedup_alt[k] = prev

        for item in dedup_alt.values():
            val = item["value"]
            ratio = safe_float(item.get("ratio", 0.0), 0.0)
            sim = string_similarity(dirty_value, val)

            top_score = 0.32 + 0.18 * min(ratio, 1.0)
            add_candidate(candidate_dict, val, "column_top_value", top_score, extra=f"ratio={ratio:.4f}")

            if sim >= MIN_STRING_SIM_FOR_COLUMN_VALUE:
                sim_score = 0.46 + 0.22 * sim + 0.10 * min(ratio, 1.0)
                add_candidate(candidate_dict, val, "similar_column_value", sim_score, extra=f"ratio={ratio:.4f}|sim={sim:.4f}")

        # F. Beers 专属候选
        for val, source, score, extra in generate_beers_context_candidates(column, dirty_value, ctx, global_dict):
            add_candidate(candidate_dict, val, source, score, extra=extra)

        # G. 删除与 dirty 等价的候选
        for k in list(candidate_dict.keys()):
            if value_equal_for_column(column, candidate_dict[k]["candidate_value"], dirty_value):
                candidate_dict.pop(k, None)

        # H. 排序
        final_candidates = []
        for _, item in candidate_dict.items():
            cand = item["candidate_value"]
            sim = string_similarity(dirty_value, cand)
            source_count = len(item["candidate_sources"])
            support_bonus = 0.04 * max(source_count - 1, 0)

            semantic_bonus = 0.0
            fixed_cand = normalize_by_column(column, cand)
            if fixed_cand is not None and norm_text(fixed_cand) == norm_text(cand):
                semantic_bonus += 0.06

            if norm_text(column) == "state" and normalize_state_candidate(cand) is not None:
                semantic_bonus += 0.08
            if norm_text(column) == "city" and normalize_city_candidate(cand) is not None:
                semantic_bonus += 0.04
            if norm_text(column) == "ounces" and normalize_ounces_candidate(cand) is not None:
                semantic_bonus += 0.08
            if norm_text(column) == "abv" and normalize_abv_candidate(cand) is not None:
                semantic_bonus += 0.07
            if norm_text(column) == "ibu" and normalize_ibu_candidate(cand) is not None:
                semantic_bonus += 0.07

            final_score = item["source_score"] + 0.16 * sim + support_bonus + semantic_bonus

            item["edit_similarity_to_dirty"] = round(sim, 6)
            item["final_score"] = round(final_score, 6)
            item["candidate_sources"] = sorted(list(item["candidate_sources"]))
            item["candidate_source_joined"] = "|".join(item["candidate_sources"])
            final_candidates.append(item)

        final_candidates = sorted(
            final_candidates,
            key=lambda x: (
                -x["final_score"],
                -x["is_from_rule"],
                -x["is_from_dictionary"],
                -x["is_from_pattern_restore"],
                -x["is_from_neighbor"],
                x["candidate_value"],
            ),
        )[:MAX_CANDIDATES_PER_CELL]

        for rank_idx, item in enumerate(final_candidates, start=1):
            expanded_rows.append({
                "row_id": row_id,
                "column": column,
                "dirty_value": dirty_value,
                "candidate_rank": rank_idx,
                "candidate_value": item["candidate_value"],
                "candidate_source": item["candidate_source_joined"],
                "source_score": item["source_score"],
                "final_score": item["final_score"],
                "edit_similarity_to_dirty": item["edit_similarity_to_dirty"],
                "is_from_rule": item["is_from_rule"],
                "is_from_neighbor": item["is_from_neighbor"],
                "is_from_column_top": item["is_from_column_top"],
                "is_from_similarity": item["is_from_similarity"],
                "is_from_hint": item["is_from_hint"],
                "is_from_dictionary": item["is_from_dictionary"],
                "is_from_pattern_restore": item["is_from_pattern_restore"],
                "source_count": len(item["candidate_sources"]),
                "extra_info": " || ".join(item["extra_info"][:15]),
                "violation_count": ctx.get("violation_count", None),
                "conflict_score": ctx.get("conflict_score", None),
                "semantic_type": ctx.get("semantic_type", None),
                "main_rule_type": ctx.get("main_rule_type", None),
                "main_usage_role": ctx.get("main_usage_role", None),
                "numeric_value": ctx.get("numeric_value", None),
                "numeric_window_rule_count": ctx.get("numeric_window_rule_count", None),
                "beers_is_state_candidate": int(norm_text(column) == "state" and normalize_state_candidate(item["candidate_value"]) is not None),
                "beers_is_city_candidate": int(norm_text(column) == "city" and normalize_city_candidate(item["candidate_value"]) is not None),
                "beers_is_ounces_candidate": int(norm_text(column) == "ounces" and normalize_ounces_candidate(item["candidate_value"]) is not None),
                "beers_is_abv_candidate": int(norm_text(column) == "abv" and normalize_abv_candidate(item["candidate_value"]) is not None),
                "beers_is_ibu_candidate": int(norm_text(column) == "ibu" and normalize_ibu_candidate(item["candidate_value"]) is not None),
            })

        grouped_rows.append({
            "row_id": row_id,
            "column": column,
            "dirty_value": dirty_value,
            "candidate_count": len(final_candidates),
            "candidates_joined": " || ".join([x["candidate_value"] for x in final_candidates]),
            "sources_joined": " || ".join([x["candidate_source_joined"] for x in final_candidates]),
            "top_candidate": final_candidates[0]["candidate_value"] if final_candidates else "",
            "top_candidate_score": final_candidates[0]["final_score"] if final_candidates else None,
        })

    expanded_df = pd.DataFrame(expanded_rows)
    grouped_df = pd.DataFrame(grouped_rows)

    if not expanded_df.empty:
        expanded_df = expanded_df.sort_values(["row_id", "column", "candidate_rank"]).reset_index(drop=True)

    if not grouped_df.empty:
        grouped_df = grouped_df.sort_values(["row_id", "column"]).reset_index(drop=True)

    out_expanded = resolve_output_path(OUTPUT_CANDIDATES_EXPANDED_CSV)
    out_grouped = resolve_output_path(OUTPUT_CANDIDATES_GROUPED_CSV)

    expanded_df.to_csv(out_expanded, index=False, encoding="utf-8-sig")
    grouped_df.to_csv(out_grouped, index=False, encoding="utf-8-sig")

    print(f"[INFO] 缺失上下文的错误 cell 数量: {missing_context_count}")
    print(f"[INFO] 候选展开文件已保存: {out_expanded}")
    print(f"[INFO] 候选汇总文件已保存: {out_grouped}")

    if not grouped_df.empty:
        print("\n[INFO] 前 20 个错误 cell 的候选示例：")
        print(grouped_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()

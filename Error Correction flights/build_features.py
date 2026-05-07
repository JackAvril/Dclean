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

# 上一步新版候选展开文件
# 建议使用刚才新版候选生成脚本输出的 repair_candidates_expanded_v2.csv。
CANDIDATES_FILE = "/mnt/mydata/dq/projects/Splittree/test/Error Correction flights/repair_candidates_expanded.csv"

# rich context jsonl（推荐）
# 新格式应包含 row_context / column_context / conflict_context / similarity_context / bayesian_context。
CONTEXT_FILE = "/mnt/mydata/dq/projects/Splittree/test/Error Detection flights/candidate_llm_contexts_flights.jsonl"

# 可选：flat csv fallback，没有就留空。
FALLBACK_CONTEXT_FLAT_FILE = "/mnt/mydata/dq/projects/Splittree/test/Error Detection flights/candidate_llm_contexts_flat_flights.csv"

# 原始脏数据表：用于构建经验共现字典。
# 注意：Flights 实验不要再默认读取 Hospital 脏表。
# 如果你有 Flights 原始脏表，可以填：
# RAW_DATA_FILE = "/mnt/mydata/dq/projects/Splittree/flights/flights_dirty.csv"
RAW_DATA_FILE = "/mnt/mydata/dq/projects/Splittree/flights/flights_dirty.csv"

# 可选：现成共现字典。没有就保持 None。
COOCCURRENCE_DICT_FILE = None

# 输出；如果是相对路径，会自动输出到 CANDIDATES_FILE 所在目录。
OUTPUT_FEATURES_CSV = "repair_candidate_features_v2.csv"
OUTPUT_BUILT_COOCCUR_JSON = "built_cooccurrence_dict_v2.json"

# 参数
MAX_CONTEXT_PAIRS = 12
ENABLE_NUMERIC_CANDIDATE_CHECK = True
SMOOTHING_ALPHA = 1.0
MAX_DISTINCT_VALUES_PER_COLUMN = 2000
TOP_K_SIMILAR_ROW_VALUES = 6


# ============================================================
# 2. 基础工具函数
# ============================================================

EMPTY_TOKENS = {"", "empty", "null", "none", "nan", "na", "n/a", "missing"}
CONDITION_CANONICAL = {
    "heart attack",
    "heart failure",
    "pneumonia",
    "surgical infection prevention",
}
MEASURE_ANCHOR_TOKENS = {
    "heart", "attack", "failure", "pneumonia", "surgery", "antibiotic", "infection",
    "beta", "blockers", "children", "asthma", "blood", "clot", "discharge", "arrival"
}


def norm_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def canonical_empty_text(x):
    s = norm_text(x)
    return "empty" if s in EMPTY_TOKENS else ("" if pd.isna(x) else str(x).strip())


def safe_float(x, default=0.0):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return default
        return float(x)
    except Exception:
        return default


def path_exists_nonempty(path):
    return path is not None and str(path).strip() != "" and os.path.exists(str(path))


def resolve_output_path(output_path, anchor_input_path):
    """
    如果 output_path 是相对路径，则默认输出到 anchor_input_path 所在目录。
    这样特征文件会和候选文件放在同一个实验目录里。
    """
    output_path = str(output_path)
    if os.path.isabs(output_path):
        return output_path
    base_dir = os.path.dirname(os.path.abspath(str(anchor_input_path)))
    return os.path.join(base_dir, output_path)


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


# ============================================================
# 3. 键盘距离
# ============================================================

KEYBOARD_ROWS = [
    "1234567890",
    "qwertyuiop",
    "asdfghjkl",
    "zxcvbnm"
]

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
            if d is None:
                d = 5.0
            dists.append(float(d))
    return sum(dists) / len(dists)


# ============================================================
# 4. 发音编码（简化 Soundex）
# ============================================================

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
        if code != prev:
            if code != "":
                codes.append(code)
        prev = code

    result = first + "".join(codes)
    result = (result + "000")[:4]
    return result


def phrase_soundex(s):
    s = norm_text(s)
    if not s:
        return ""
    toks = [t for t in s.replace("-", " ").replace("/", " ").split() if t]
    return " ".join(soundex_token(t) for t in toks)


def phonetic_similarity(a, b):
    sa = phrase_soundex(a)
    sb = phrase_soundex(b)
    return string_similarity(sa, sb)


# ============================================================
# 5. 结构/语义辅助
# ============================================================

def looks_like_measure_code(x: str) -> int:
    s = norm_text(x)
    if not s:
        return 0
    return int(
        s.startswith("ami") or s.startswith("hf") or s.startswith("pn") or s.startswith("scip")
        or ("ami" in s) or ("hf" in s) or ("pn" in s) or ("scip" in s)
    )


def looks_like_stateavg(x: str) -> int:
    s = norm_text(x)
    return int("_" in s and looks_like_measure_code(s) == 1)


def normalize_measurecode_like(x: str) -> str:
    s = canonical_empty_text(x).lower().strip()
    if "_" in s:
        s = s.split("_", 1)[1]
    s = s.replace("_", "-")
    s = re.sub(r"-+", "-", s)
    return s


def prefix_token(x: str) -> str:
    s = canonical_empty_text(x)
    if "_" in s:
        return s.split("_", 1)[0].strip().lower()
    return ""


def family_token(x: str) -> str:
    s = normalize_measurecode_like(x)
    parts = re.split(r"[-_]", s)
    return parts[0] if parts else ""


def suffix_token(x: str) -> str:
    s = normalize_measurecode_like(x)
    parts = re.split(r"[-_]", s)
    return parts[-1] if parts else ""


def measure_family_name(code: str) -> str:
    c = normalize_measurecode_like(code)
    if c.startswith("hf"):
        return "heart failure"
    if c.startswith("ami"):
        return "heart attack"
    if c.startswith("pn"):
        return "pneumonia"
    if c.startswith("scip"):
        return "surgical infection prevention"
    return ""


def extract_anchor_tokens(s: str):
    toks = set(re.findall(r"[a-z]+", norm_text(s)))
    return toks & MEASURE_ANCHOR_TOKENS


def is_valid_percent(s: str) -> int:
    return int(bool(re.fullmatch(r"\d{1,3}%", canonical_empty_text(s))))


def normalize_sample_candidate(x: str):
    s = canonical_empty_text(x).lower().strip()
    if s in {"", "empty"}:
        return None
    digits = "".join(ch for ch in s if ch.isdigit())
    if digits == "":
        return None
    if "patient" in s or "patien" in s or "paxien" in s or "pxtient" in s or "patixnt" in s:
        return f"{int(digits)} patients"
    return None


def is_valid_sample_pattern(s: str) -> int:
    return int(bool(re.fullmatch(r"\d+\s+patients", canonical_empty_text(s).lower())))


TIME_COLUMNS_HINT = {
    "sched_dep_time",
    "act_dep_time",
    "sched_arr_time",
    "act_arr_time",
    "dep_time",
    "arr_time",
    "scheduled_departure",
    "scheduled_arrival",
    "actual_departure",
    "actual_arrival",
}


def looks_like_time_column(column, semantic_type=None):
    col = norm_text(column)
    sem = norm_text(semantic_type)
    return sem == "time_like" or col in TIME_COLUMNS_HINT or col.endswith("_time")


def parse_time_to_minutes(x):
    """
    解析常见航班时间格式：
    - 7:10 a.m.
    - 7:10 am
    - 07:10
    - 710 / 0710
    返回 [0, 1439] 分钟；失败返回 None。
    """
    s = canonical_empty_text(x).lower()
    if s in {"", "empty"}:
        return None

    s = s.replace(".", "")
    s = re.sub(r"\s+", " ", s).strip()

    ampm = None
    if "am" in s:
        ampm = "am"
    elif "pm" in s:
        ampm = "pm"

    s_clean = s.replace("am", "").replace("pm", "").strip()

    hour = None
    minute = None

    m = re.fullmatch(r"(\d{1,2})\s*:\s*(\d{2})", s_clean)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2))
    else:
        digits = re.sub(r"\D", "", s_clean)
        if len(digits) in {3, 4}:
            hour = int(digits[:-2])
            minute = int(digits[-2:])

    if hour is None or minute is None:
        return None
    if minute < 0 or minute > 59:
        return None

    if ampm == "am":
        if hour == 12:
            hour = 0
        elif 1 <= hour <= 11:
            pass
        else:
            return None
    elif ampm == "pm":
        if hour == 12:
            pass
        elif 1 <= hour <= 11:
            hour += 12
        else:
            return None
    else:
        if not (0 <= hour <= 23):
            return None

    return hour * 60 + minute


def circular_minute_diff(a, b):
    """
    跨午夜时取较小差值，例如 23:55 与 00:05 差 10 分钟。
    """
    if a is None or b is None:
        return None
    diff = abs(int(a) - int(b))
    return min(diff, 1440 - diff)


def is_valid_time_value(x):
    return int(parse_time_to_minutes(x) is not None)


def time_format_restoration_gain(candidate_value, dirty_value):
    return int(is_valid_time_value(candidate_value) == 1 and is_valid_time_value(dirty_value) == 0)


# ============================================================
# 6. 统一整理上下文（rich jsonl + fallback flat）
# ============================================================

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
        rule_expected_values = []
        expected_values_from_rules = []
        alternative_values_from_column = []
        strong_rule_count = 0
        candidate_generation_rule_count = 0
        fd_like_count = 0
        global_rule_count = 0
        typo_rule_count = 0
        pattern_rule_count = 0
        schema_rule_count = 0

        if isinstance(conflict_context, dict):
            violated = conflict_context.get("violated_rules_detailed", []) or []
            for vr in violated:
                if isinstance(vr, dict):
                    ev = canonical_empty_text(vr.get("expected_value", None))
                    if ev not in {"", "empty"}:
                        rule_expected_values.append(ev)

            for item in conflict_context.get("expected_values_from_rules", []) or []:
                if isinstance(item, dict):
                    ev = canonical_empty_text(item.get("value", None))
                    if ev not in {"", "empty"}:
                        expected_values_from_rules.append(ev)

            for item in conflict_context.get("alternative_values_from_column", []) or []:
                if isinstance(item, dict):
                    v = canonical_empty_text(item.get("value", None))
                    if v not in {"", "empty"}:
                        alternative_values_from_column.append({
                            "value": v,
                            "count": int(item.get("count", 0) or 0),
                            "ratio": 0.0
                        })

            strong_rule_count = int(conflict_context.get("strong_rule_count", 0) or 0)
            candidate_generation_rule_count = int(conflict_context.get("candidate_generation_rule_count", 0) or 0)
            fd_like_count = int(conflict_context.get("fd_like_count", 0) or 0)
            global_rule_count = int(conflict_context.get("global_rule_count", 0) or 0)
            typo_rule_count = int(conflict_context.get("typo_rule_count", 0) or 0)
            pattern_rule_count = int(conflict_context.get("pattern_rule_count", 0) or 0)
            schema_rule_count = int(conflict_context.get("schema_rule_count", 0) or 0)

        suggested_correct_value = canonical_empty_text(obj.get("suggested_correct_value", ""))
        if suggested_correct_value == "empty":
            suggested_correct_value = None

        similarity_context = obj.get("similarity_context", {})
        neighbor_majority_value = None
        neighbor_majority_ratio = 0.0
        similar_row_target_values = []
        if isinstance(similarity_context, dict):
            nmv = canonical_empty_text(similarity_context.get("neighbor_majority_value", ""))
            neighbor_majority_value = None if nmv in {"", "empty"} else nmv
            neighbor_majority_ratio = safe_float(similarity_context.get("neighbor_majority_ratio", 0.0), 0.0)

            topk = similarity_context.get("topk_similar_rows", []) or []
            for item in topk[:TOP_K_SIMILAR_ROW_VALUES]:
                if isinstance(item, dict):
                    tv = canonical_empty_text(item.get("target_column_value", ""))
                    if tv not in {"", "empty"}:
                        similar_row_target_values.append(tv)

        column_top_values = []
        column_detected_type = obj.get("detected_type", None)
        column_numeric_stats = None
        column_time_stats = None
        column_time_value_minutes = None
        col_ctx = obj.get("column_context", {})
        if isinstance(col_ctx, dict):
            top_vals = col_ctx.get("top_values_with_counts", []) or []
            for item in top_vals:
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
            column_time_stats = col_ctx.get("time_stats", None)
            column_time_value_minutes = col_ctx.get("time_value_minutes", None)

        if not column_top_values:
            counter = column_value_counter[column]
            total = max(sum(counter.values()), 1)
            for val, cnt in counter.most_common(8):
                column_top_values.append({
                    "value": val,
                    "count": cnt,
                    "ratio": cnt / total
                })

        bayesian_context = obj.get("bayesian_context", {})
        prior_error_probability = 0.0
        posterior_error_probability = 0.0
        bayes_rule_signal = 0.0
        bayes_rarity_signal = 0.0
        bayes_pattern_signal = 0.0
        bayes_neighbor_signal = 0.0
        if isinstance(bayesian_context, dict):
            prior_error_probability = safe_float(bayesian_context.get("prior_error_probability", 0.0), 0.0)
            posterior_error_probability = safe_float(bayesian_context.get("posterior_error_probability", 0.0), 0.0)
            be = bayesian_context.get("bayes_evidence", {})
            if isinstance(be, dict):
                bayes_rule_signal = safe_float(be.get("rule_signal", 0.0), 0.0)
                bayes_rarity_signal = safe_float(be.get("rarity_signal", 0.0), 0.0)
                bayes_pattern_signal = safe_float(be.get("pattern_signal", 0.0), 0.0)
                bayes_neighbor_signal = safe_float(be.get("neighbor_signal", 0.0), 0.0)

        row_time_bundle = {}
        if isinstance(rc, dict):
            tb = rc.get("time_bundle", {})
            if isinstance(tb, dict):
                row_time_bundle = {str(k): canonical_empty_text(v) for k, v in tb.items()}

        context_map[(row_id, column)] = {
            "row_id": row_id,
            "column": column,
            "value": value,
            "semantic_type": obj.get("semantic_type", None),
            "violation_count": obj.get("violation_count", None),
            "conflict_score": obj.get("conflict_score", None),
            "main_rule_type": obj.get("main_rule_type", None),
            "main_usage_role": obj.get("main_usage_role", None),
            "row_values": row_values,
            "suggested_correct_value": suggested_correct_value,
            "neighbor_majority_value": neighbor_majority_value,
            "neighbor_majority_ratio": neighbor_majority_ratio,
            "similar_row_target_values": similar_row_target_values,
            "column_top_values": column_top_values,
            "alternative_values_from_column": alternative_values_from_column,
            "rule_expected_values": rule_expected_values,
            "expected_values_from_rules": expected_values_from_rules,
            "column_numeric_stats": column_numeric_stats,
            "column_time_stats": column_time_stats,
            "column_time_value_minutes": column_time_value_minutes,
            "column_detected_type": column_detected_type,
            "row_time_bundle": row_time_bundle,
            "prior_error_probability": prior_error_probability,
            "posterior_error_probability": posterior_error_probability,
            "bayes_rule_signal": bayes_rule_signal,
            "bayes_rarity_signal": bayes_rarity_signal,
            "bayes_pattern_signal": bayes_pattern_signal,
            "bayes_neighbor_signal": bayes_neighbor_signal,
            "strong_rule_count": strong_rule_count,
            "candidate_generation_rule_count": candidate_generation_rule_count,
            "fd_like_count": fd_like_count,
            "global_rule_count": global_rule_count,
            "typo_rule_count": typo_rule_count,
            "pattern_rule_count": pattern_rule_count,
            "schema_rule_count": schema_rule_count,
            "state_value": canonical_empty_text(row_values.get("State", "")),
            "measure_code_value": canonical_empty_text(row_values.get("MeasureCode", "")),
            "condition_value": canonical_empty_text(row_values.get("Condition", "")),
            "measure_name_value": canonical_empty_text(row_values.get("MeasureName", "")),
            "stateavg_value": canonical_empty_text(row_values.get("Stateavg", "")),
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
                "semantic_type": obj.get("semantic_type", None),
                "violation_count": obj.get("violation_count", None),
                "conflict_score": obj.get("conflict_score", None),
                "main_rule_type": obj.get("main_rule_type", None),
                "main_usage_role": obj.get("main_usage_role", None),
                "row_values": {},
                "suggested_correct_value": None,
                "neighbor_majority_value": None,
                "neighbor_majority_ratio": 0.0,
                "similar_row_target_values": [],
                "column_top_values": [],
                "alternative_values_from_column": [],
                "rule_expected_values": [],
                "expected_values_from_rules": [],
                "column_numeric_stats": None,
                "column_time_stats": None,
                "column_time_value_minutes": None,
                "column_detected_type": obj.get("detected_type", None),
                "row_time_bundle": {},
                "prior_error_probability": 0.0,
                "posterior_error_probability": 0.0,
                "bayes_rule_signal": 0.0,
                "bayes_rarity_signal": 0.0,
                "bayes_pattern_signal": 0.0,
                "bayes_neighbor_signal": 0.0,
                "strong_rule_count": 0,
                "candidate_generation_rule_count": 0,
                "fd_like_count": 0,
                "global_rule_count": 0,
                "typo_rule_count": 0,
                "pattern_rule_count": 0,
                "schema_rule_count": 0,
                "state_value": "",
                "measure_code_value": "",
                "condition_value": "",
                "measure_name_value": "",
                "stateavg_value": "",
            }

        ctx = context_map[key]
        for field in [
            "semantic_type", "violation_count", "conflict_score", "main_rule_type", "main_usage_role"
        ]:
            if ctx.get(field, None) in [None, "", []] and field in obj:
                ctx[field] = obj.get(field)

        if not ctx.get("neighbor_majority_value", None):
            nmv = canonical_empty_text(obj.get("neighbor_majority_value", ""))
            ctx["neighbor_majority_value"] = None if nmv in {"", "empty"} else nmv
        if safe_float(ctx.get("neighbor_majority_ratio", 0.0), 0.0) == 0.0 and "neighbor_majority_ratio" in obj:
            ctx["neighbor_majority_ratio"] = safe_float(obj.get("neighbor_majority_ratio", 0.0), 0.0)
        if ctx.get("prior_error_probability", 0.0) == 0.0 and "prior_error_probability" in obj:
            ctx["prior_error_probability"] = safe_float(obj.get("prior_error_probability", 0.0), 0.0)
        if ctx.get("posterior_error_probability", 0.0) == 0.0 and "posterior_error_probability" in obj:
            ctx["posterior_error_probability"] = safe_float(obj.get("posterior_error_probability", 0.0), 0.0)


# ============================================================
# 7. 规则满足性特征
# ============================================================

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
        "candidate_matches_any_strong_rule_expected": int(canonical_empty_text(candidate_value) in {canonical_empty_text(v) for v in ctx.get("rule_expected_values", [])}),
        "candidate_matches_any_expected_values_from_rules": int(canonical_empty_text(candidate_value) in {canonical_empty_text(v) for v in ctx.get("expected_values_from_rules", [])}),
        "candidate_in_alternative_values_from_column": int(canonical_empty_text(candidate_value) in alt_values),
        "candidate_equals_neighbor_majority": int(
            ctx.get("neighbor_majority_value", None) is not None and
            norm_text(candidate_value) == norm_text(ctx.get("neighbor_majority_value"))
        ),
        "candidate_equals_suggested_correct_value": int(
            ctx.get("suggested_correct_value", None) is not None and
            norm_text(candidate_value) == norm_text(ctx.get("suggested_correct_value"))
        ),
        "candidate_matches_similar_row_target": int(canonical_empty_text(candidate_value) in sim_row_values),
        "candidate_match_count_in_similar_rows": sum(1 for v in sim_row_values if canonical_empty_text(candidate_value) == v),
        "candidate_match_ratio_in_similar_rows": (
            sum(1 for v in sim_row_values if canonical_empty_text(candidate_value) == v) / len(sim_row_values)
            if sim_row_values else 0.0
        ),
        "strong_rule_count": int(ctx.get("strong_rule_count", 0) or 0),
        "candidate_generation_rule_count": int(ctx.get("candidate_generation_rule_count", 0) or 0),
        "fd_like_count": int(ctx.get("fd_like_count", 0) or 0),
        "global_rule_count": int(ctx.get("global_rule_count", 0) or 0),
        "typo_rule_count": int(ctx.get("typo_rule_count", 0) or 0),
        "pattern_rule_count": int(ctx.get("pattern_rule_count", 0) or 0),
        "schema_rule_count": int(ctx.get("schema_rule_count", 0) or 0),
    }


# ============================================================
# 8. 从原始数据集构建经验共现字典
# ============================================================

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
                v = r[target_col]
                c = r[ctx_col]
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

                    if cand_val not in cooccur[target_col]:
                        cooccur[target_col][cand_val] = {}
                    cooccur[target_col][cand_val][f"{ctx_col}={ctx_val}"] = prob

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


# ============================================================
# 9. 贝叶斯共现概率 / P(v | context)
# ============================================================

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

    if not context_pairs or external_cooccur is None:
        return {
            "context_pair_count_used": 0,
            "bayes_max_cond_prob": 0.0,
            "bayes_min_cond_prob": 0.0,
            "bayes_mean_cond_prob": 0.0,
            "bayes_prod_logprob": 0.0,
            "bayes_nonzero_pair_count": 0,
            "prior_error_probability": safe_float(ctx.get("prior_error_probability", 0.0), 0.0),
            "posterior_error_probability": safe_float(ctx.get("posterior_error_probability", 0.0), 0.0),
            "bayes_rule_signal": safe_float(ctx.get("bayes_rule_signal", 0.0), 0.0),
            "bayes_rarity_signal": safe_float(ctx.get("bayes_rarity_signal", 0.0), 0.0),
            "bayes_pattern_signal": safe_float(ctx.get("bayes_pattern_signal", 0.0), 0.0),
            "bayes_neighbor_signal": safe_float(ctx.get("bayes_neighbor_signal", 0.0), 0.0),
        }

    target_column = ctx.get("column")
    col_map = external_cooccur.get(target_column, {})
    cand_map = col_map.get(candidate_value, {})

    probs = []
    for ctx_col, ctx_val in context_pairs:
        key = f"{ctx_col}={ctx_val}"
        p = safe_float(cand_map.get(key, 0.0), 0.0)
        probs.append(p)

    nonzero = [p for p in probs if p > 0]
    mean_prob = sum(probs) / len(probs) if probs else 0.0
    max_prob = max(probs) if probs else 0.0
    min_prob = min(probs) if probs else 0.0
    logprod = sum(math.log(max(p, 1e-12)) for p in probs) if probs else 0.0

    return {
        "context_pair_count_used": len(probs),
        "bayes_max_cond_prob": max_prob,
        "bayes_min_cond_prob": min_prob,
        "bayes_mean_cond_prob": mean_prob,
        "bayes_prod_logprob": logprod,
        "bayes_nonzero_pair_count": len(nonzero),
        "prior_error_probability": safe_float(ctx.get("prior_error_probability", 0.0), 0.0),
        "posterior_error_probability": safe_float(ctx.get("posterior_error_probability", 0.0), 0.0),
        "bayes_rule_signal": safe_float(ctx.get("bayes_rule_signal", 0.0), 0.0),
        "bayes_rarity_signal": safe_float(ctx.get("bayes_rarity_signal", 0.0), 0.0),
        "bayes_pattern_signal": safe_float(ctx.get("bayes_pattern_signal", 0.0), 0.0),
        "bayes_neighbor_signal": safe_float(ctx.get("bayes_neighbor_signal", 0.0), 0.0),
    }


# ============================================================
# 10. 变换特征
# ============================================================

def compute_transform_features(candidate_value, dirty_value):
    sim = string_similarity(candidate_value, dirty_value)
    phon_sim = phonetic_similarity(candidate_value, dirty_value)
    key_dist = avg_keyboard_distance_for_same_length(candidate_value, dirty_value)

    len_a = len(str(candidate_value))
    len_b = len(str(dirty_value))
    length_diff = abs(len_a - len_b)

    num_a = try_parse_float(candidate_value)
    num_b = try_parse_float(dirty_value)
    if ENABLE_NUMERIC_CANDIDATE_CHECK and num_a is not None and num_b is not None:
        numeric_abs_diff = abs(num_a - num_b)
        numeric_rel_diff = abs(num_a - num_b) / max(abs(num_b), 1e-9)
        both_numeric = 1
    else:
        numeric_abs_diff = None
        numeric_rel_diff = None
        both_numeric = 0

    return {
        "edit_similarity_to_dirty": sim,
        "edit_distance_proxy": 1.0 - sim,
        "phonetic_similarity_to_dirty": phon_sim,
        "phonetic_distance_proxy": 1.0 - phon_sim,
        "avg_keyboard_distance": key_dist,
        "length_diff": length_diff,
        "both_numeric": both_numeric,
        "numeric_abs_diff": numeric_abs_diff,
        "numeric_rel_diff": numeric_rel_diff,
    }


# ============================================================
# 11. 额外一致性 / plausibility 特征
# ============================================================

def compute_column_plausibility_features(candidate_value, dirty_value, ctx):
    column_top_values = ctx.get("column_top_values", [])
    top_counter = {canonical_empty_text(x["value"]): x for x in column_top_values if isinstance(x, dict)}

    candidate_norm = canonical_empty_text(candidate_value)
    dirty_norm = canonical_empty_text(dirty_value)

    candidate_in_top_values = int(candidate_norm in top_counter)
    candidate_top_ratio = safe_float(top_counter.get(candidate_norm, {}).get("ratio", 0.0), 0.0)

    detected_type = str(ctx.get("column_detected_type", "") or "").lower()
    num_stats = ctx.get("column_numeric_stats", None)

    candidate_in_numeric_range = None
    candidate_outside_numeric_range = None

    if detected_type == "numeric" and isinstance(num_stats, dict):
        cand_num = try_parse_float(candidate_value)
        if cand_num is not None:
            min_v = try_parse_float(num_stats.get("min"))
            max_v = try_parse_float(num_stats.get("max"))
            if min_v is not None and max_v is not None:
                candidate_in_numeric_range = int(min_v <= cand_num <= max_v)
                candidate_outside_numeric_range = int(not (min_v <= cand_num <= max_v))

    row_state = canonical_empty_text(ctx.get("state_value", ""))
    row_measure_code = canonical_empty_text(ctx.get("measure_code_value", ""))
    row_condition = canonical_empty_text(ctx.get("condition_value", ""))
    row_measure_name = canonical_empty_text(ctx.get("measure_name_value", ""))

    candidate_minutes = parse_time_to_minutes(candidate_value)
    dirty_minutes = parse_time_to_minutes(dirty_value)
    is_time_col = looks_like_time_column(ctx.get("column", ""), ctx.get("semantic_type", None))

    candidate_in_time_range = None
    candidate_outside_time_range = None
    time_stats = ctx.get("column_time_stats", None)
    if is_time_col and isinstance(time_stats, dict) and candidate_minutes is not None:
        min_minutes = safe_float(time_stats.get("min_minutes", None), None)
        max_minutes = safe_float(time_stats.get("max_minutes", None), None)
        if min_minutes is not None and max_minutes is not None:
            candidate_in_time_range = int(min_minutes <= candidate_minutes <= max_minutes)
            candidate_outside_time_range = int(not (min_minutes <= candidate_minutes <= max_minutes))

    row_time_values = []
    row_values = ctx.get("row_values", {}) or {}
    for time_col in ["sched_dep_time", "act_dep_time", "sched_arr_time", "act_arr_time"]:
        if time_col == ctx.get("column"):
            continue
        if time_col in row_values:
            mv = parse_time_to_minutes(row_values.get(time_col))
            if mv is not None:
                row_time_values.append(mv)

    if candidate_minutes is not None and row_time_values:
        diffs = [circular_minute_diff(candidate_minutes, mv) for mv in row_time_values]
        diffs = [d for d in diffs if d is not None]
        candidate_min_diff_to_row_time = min(diffs) if diffs else None
        candidate_mean_diff_to_row_time = sum(diffs) / len(diffs) if diffs else None
    else:
        candidate_min_diff_to_row_time = None
        candidate_mean_diff_to_row_time = None

    if dirty_minutes is not None and row_time_values:
        diffs_dirty = [circular_minute_diff(dirty_minutes, mv) for mv in row_time_values]
        diffs_dirty = [d for d in diffs_dirty if d is not None]
        dirty_min_diff_to_row_time = min(diffs_dirty) if diffs_dirty else None
    else:
        dirty_min_diff_to_row_time = None

    if candidate_min_diff_to_row_time is not None and dirty_min_diff_to_row_time is not None:
        candidate_time_context_gain = dirty_min_diff_to_row_time - candidate_min_diff_to_row_time
    else:
        candidate_time_context_gain = None

    return {
        "candidate_in_column_top_values": candidate_in_top_values,
        "candidate_column_top_ratio": candidate_top_ratio,
        "candidate_in_numeric_range": candidate_in_numeric_range,
        "candidate_outside_numeric_range": candidate_outside_numeric_range,
        "candidate_looks_like_measurecode": looks_like_measure_code(candidate_value),
        "dirty_looks_like_measurecode": looks_like_measure_code(dirty_value),
        "candidate_looks_like_stateavg": looks_like_stateavg(candidate_value),
        "dirty_looks_like_stateavg": looks_like_stateavg(dirty_value),
        "candidate_family_match_dirty": int(family_token(candidate_value) == family_token(dirty_value) and family_token(dirty_value) != ""),
        "candidate_suffix_match_dirty": int(suffix_token(candidate_value) == suffix_token(dirty_value) and suffix_token(dirty_value) != ""),
        "candidate_state_prefix_match_row_state": int(prefix_token(candidate_value) == row_state and row_state != ""),
        "candidate_stateavg_matches_row_measurecode": int(
            looks_like_stateavg(candidate_value) == 1 and row_measure_code != "" and normalize_measurecode_like(candidate_value) == normalize_measurecode_like(row_measure_code)
        ),
        "candidate_condition_matches_measurecode_family": int(
            candidate_norm in CONDITION_CANONICAL and row_measure_code != "" and candidate_norm == measure_family_name(row_measure_code)
        ),
        "candidate_name_anchor_overlap_with_measurecode_family": len(extract_anchor_tokens(candidate_value) & extract_anchor_tokens(measure_family_name(row_measure_code))),
        "candidate_name_anchor_overlap_with_condition": len(extract_anchor_tokens(candidate_value) & extract_anchor_tokens(row_condition)),
        "candidate_name_anchor_overlap_with_dirty": len(extract_anchor_tokens(candidate_value) & extract_anchor_tokens(dirty_value)),
        "candidate_name_anchor_overlap_with_row_name": len(extract_anchor_tokens(candidate_value) & extract_anchor_tokens(row_measure_name)),
        "candidate_is_valid_percent": is_valid_percent(candidate_value),
        "dirty_is_valid_percent": is_valid_percent(dirty_value),
        "candidate_percent_restoration_gain": int(is_valid_percent(candidate_value) == 1 and is_valid_percent(dirty_value) == 0),
        "candidate_is_valid_sample_pattern": is_valid_sample_pattern(candidate_value),
        "dirty_is_valid_sample_pattern": is_valid_sample_pattern(dirty_value),
        "candidate_sample_restoration_gain": int(is_valid_sample_pattern(candidate_value) == 1 and is_valid_sample_pattern(dirty_value) == 0),

        # Flights / 通用 time-like 特征
        "is_time_like_column": int(is_time_col),
        "candidate_is_valid_time": is_valid_time_value(candidate_value),
        "dirty_is_valid_time": is_valid_time_value(dirty_value),
        "candidate_time_format_restoration_gain": time_format_restoration_gain(candidate_value, dirty_value),
        "candidate_time_minutes": candidate_minutes,
        "dirty_time_minutes": dirty_minutes,
        "candidate_dirty_time_minute_diff": circular_minute_diff(candidate_minutes, dirty_minutes),
        "candidate_in_time_range": candidate_in_time_range,
        "candidate_outside_time_range": candidate_outside_time_range,
        "candidate_min_diff_to_row_time": candidate_min_diff_to_row_time,
        "candidate_mean_diff_to_row_time": candidate_mean_diff_to_row_time,
        "dirty_min_diff_to_row_time": dirty_min_diff_to_row_time,
        "candidate_time_context_gain": candidate_time_context_gain,
    }


# ============================================================
# 12. 主流程
# ============================================================

candidates_df = pd.read_csv(CANDIDATES_FILE, encoding="utf-8-sig")
if not {"row_id", "column", "dirty_value", "candidate_value"}.issubset(set(candidates_df.columns)):
    raise ValueError("候选文件必须至少包含 row_id,column,dirty_value,candidate_value")

candidates_df["row_id"] = candidates_df["row_id"].astype(int)

if not path_exists_nonempty(CANDIDATES_FILE):
    raise FileNotFoundError(f"候选展开文件不存在: {CANDIDATES_FILE}")
if not path_exists_nonempty(CONTEXT_FILE):
    raise FileNotFoundError(f"上下文文件不存在: {CONTEXT_FILE}")

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

feature_rows = []
missing_context_count = 0

for _, row in candidates_df.iterrows():
    row_id = int(row["row_id"])
    column = str(row["column"])
    dirty_value = canonical_empty_text(row["dirty_value"])
    candidate_value = canonical_empty_text(row["candidate_value"])

    ctx = context_map.get((row_id, column), None)
    if ctx is None:
        missing_context_count += 1
        continue

    rule_feats = compute_rule_features(candidate_value, dirty_value, ctx)
    bayes_feats = compute_bayes_features(candidate_value, ctx, cooccur_dict)
    trans_feats = compute_transform_features(candidate_value, dirty_value)
    plaus_feats = compute_column_plausibility_features(candidate_value, dirty_value, ctx)

    out = row.to_dict()

    candidate_source_text = str(row.get("candidate_source", "") or "")
    sources = set([s.strip() for s in candidate_source_text.split("|") if s.strip()])

    out.update({
        "semantic_type": ctx.get("semantic_type", None),
        "violation_count": ctx.get("violation_count", None),
        "conflict_score": ctx.get("conflict_score", None),
        "main_rule_type": ctx.get("main_rule_type", None),
        "main_usage_role": ctx.get("main_usage_role", None),
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
        "contains_dictionary_source": int(any(s in sources for s in {
            "measurecode_dictionary",
            "stateavg_from_measurecode",
            "condition_from_measurecode",
            "measurename_from_measurecode",
            "measurename_from_condition",
            "time_column_dictionary",
            "time_row_context",
        })),
        "contains_time_source": int(any(s in sources for s in {
            "time_pattern_restore",
            "time_column_dictionary",
            "time_row_context",
        })),
    })

    out.update(rule_feats)
    out.update(bayes_feats)
    out.update(trans_feats)
    out.update(plaus_feats)

    feature_rows.append(out)

features_df = pd.DataFrame(feature_rows)
OUTPUT_FEATURES_CSV = resolve_output_path(OUTPUT_FEATURES_CSV, CANDIDATES_FILE)
features_df.to_csv(OUTPUT_FEATURES_CSV, index=False, encoding="utf-8-sig")

print(f"[INFO] 候选记录数: {len(candidates_df)}")
print(f"[INFO] 成功构造特征的记录数: {len(features_df)}")
print(f"[INFO] 缺失上下文的候选记录数: {missing_context_count}")
print(f"[INFO] 输出文件: {OUTPUT_FEATURES_CSV}")

if not features_df.empty:
    show_cols = [c for c in [
        "row_id", "column", "dirty_value", "candidate_value",
        "candidate_rule_support_ratio", "rule_support_gain",
        "bayes_mean_cond_prob", "bayes_max_cond_prob",
        "edit_similarity_to_dirty", "avg_keyboard_distance",
        "phonetic_similarity_to_dirty", "candidate_in_column_top_values",
        "is_rule_and_dictionary_supported", "candidate_matches_similar_row_target",
        "candidate_condition_matches_measurecode_family",
        "candidate_name_anchor_overlap_with_condition",
        "candidate_percent_restoration_gain", "candidate_sample_restoration_gain",
        "is_time_like_column", "candidate_is_valid_time",
        "candidate_dirty_time_minute_diff", "candidate_time_context_gain",
        "contains_time_source"
    ] if c in features_df.columns]
    print("\n[INFO] 前 20 行特征预览：")
    print(features_df[show_cols].head(20).to_string(index=False))

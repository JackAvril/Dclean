"""
Rayyan repair candidate feature builder, logic-migrated from the Flights feature pipeline.

Migration principle:
- Keep the full Flights-style feature construction framework:
  rich context parsing, fallback flat context, rule support, neighbor/similar-row features,
  Bayesian co-occurrence features, edit/phonetic/keyboard transformation features,
  source-type features, candidate-generation score features, and consensus features.
- Convert Flights-specific features:
  flight/time consensus -> Rayyan journal/article metadata consensus
  time canonicalization -> language / ISSN / date / volume / issue / pagination canonicalization
  time-source flags -> Rayyan canonical / metadata / journal / article source flags
- Preserve downstream compatibility:
  input:  repair_candidates_expanded.csv
  output: repair_candidate_features_v2.csv
"""

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

# 上一步 Rayyan 逻辑迁移版候选展开文件
CANDIDATES_FILE = "/mnt/mydata/dq/projects/Splittree/test/Error Correction rayyan/repair_candidates_expanded.csv"

# Rayyan rich context jsonl
CONTEXT_FILE = "/mnt/mydata/dq/projects/Splittree/test/Error Detection rayyan/candidate_llm_contexts_rayyan.jsonl"

# 可选：flat csv fallback，没有就留空
FALLBACK_CONTEXT_FLAT_FILE = "/mnt/mydata/dq/projects/Splittree/test/Error Detection rayyan/candidate_llm_contexts_flat_rayyan.csv"

# Rayyan 原始脏数据表：用于构建经验共现字典
RAW_DATA_FILE = "/mnt/mydata/dq/projects/Splittree/rayyan/rayyan_dirty.csv"

# 可选：现成共现字典。没有就保持 None
COOCCURRENCE_DICT_FILE = None

# 可选：Rayyan metadata consensus 文件
# 如果候选生成脚本输出了 rayyan_metadata_consensus.csv，这里会读取并构造额外 consensus feature。
RAYYAN_METADATA_CONSENSUS_FILE = "/mnt/mydata/dq/projects/Splittree/test/Error Correction rayyan/rayyan_metadata_consensus.csv"

# 输出；如果是相对路径，会输出到当前运行目录
OUTPUT_FEATURES_CSV = "repair_candidate_features_v2.csv"
OUTPUT_BUILT_COOCCUR_JSON = "built_cooccurrence_dict_v2.json"

# 参数
MAX_CONTEXT_PAIRS = 12
ENABLE_NUMERIC_CANDIDATE_CHECK = True
SMOOTHING_ALPHA = 1.0
MAX_DISTINCT_VALUES_PER_COLUMN = 3000
TOP_K_SIMILAR_ROW_VALUES = 8


# ============================================================
# 2. 基础工具函数
# ============================================================

EMPTY_TOKENS = {"", "empty", "null", "none", "nan", "na", "n/a", "missing"}

LANGUAGE_CANONICAL_MAP = {
    "eng": "eng", "english": "eng", "en": "eng", "engl": "eng", "anglais": "eng",
    "fre": "fre", "french": "fre", "fr": "fre", "fra": "fre",
    "ger": "ger", "german": "ger", "de": "ger", "deu": "ger",
    "spa": "spa", "spanish": "spa", "es": "spa",
    "jpn": "jpn", "japanese": "jpn", "ja": "jpn",
    "chi": "chi", "chinese": "chi", "zh": "chi",
    "ita": "ita", "italian": "ita", "it": "ita",
    "dut": "dut", "dutch": "dut", "nl": "dut",
    "pol": "pol", "polish": "pol",
    "por": "por", "portuguese": "por",
    "rus": "rus", "russian": "rus",
}


def norm_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def canonical_empty_text(x):
    if pd.isna(x):
        return "empty"
    s = str(x).strip()
    return "empty" if s.lower() in EMPTY_TOKENS else s


def safe_float(x, default=0.0):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x, default=0):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return default
        return int(float(x))
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


def clean_key(x):
    return re.sub(r"\s+", " ", norm_text(x)).strip()


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
# 3. 键盘距离 / 发音特征
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
# 4. Rayyan 专属规范化 / 合法性判断
# ============================================================

def normalize_language_candidate(x):
    s = canonical_empty_text(x)
    if s in {"", "empty"}:
        return None

    lower = norm_text(s)

    if "abstract language" in lower:
        parts = re.split(r"abstract language\s*:\s*", lower)
        if len(parts) >= 2:
            lower = parts[-1].strip(";: ,.")

    lower = lower.strip(";: ,.")
    if lower in LANGUAGE_CANONICAL_MAP:
        return LANGUAGE_CANONICAL_MAP[lower]

    token = re.sub(r"[^a-z]", "", lower)
    if token in LANGUAGE_CANONICAL_MAP:
        return LANGUAGE_CANONICAL_MAP[token]

    if len(token) == 3 and token.isalpha():
        return token

    return None


def normalize_issn_candidate(x):
    s = canonical_empty_text(x)
    if s in {"", "empty"}:
        return None

    raw = s.upper().strip()
    raw = raw.replace("ISSN", "").replace(":", " ").strip()
    chars = re.sub(r"[^0-9X]", "", raw)
    if len(chars) != 8:
        return None

    return f"{chars[:4]}-{chars[4:]}"


def is_valid_issn(x):
    fixed = normalize_issn_candidate(x)
    if not fixed:
        return False

    digits = fixed.replace("-", "")
    total = 0
    try:
        for i, ch in enumerate(digits):
            val = 10 if ch == "X" else int(ch)
            total += val * (8 - i)
        return total % 11 == 0
    except Exception:
        return False


def normalize_date_candidate(x):
    s = canonical_empty_text(x)
    if s in {"", "empty"}:
        return None

    s = s.strip()

    # yyyy-mm-dd or yyyy/mm/dd
    m = re.fullmatch(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{mo}/{d}/{str(y)[-2:]}"

    # m/d/yy or m/d/yyyy
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{2}|\d{4})", s)
    if m:
        mo, d, y = int(m.group(1)), int(m.group(2)), m.group(3)
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{mo}/{d}/{y[-2:]}"

    return None


def normalize_numeric_int_candidate(x, min_v=None, max_v=None):
    s = canonical_empty_text(x)
    if s in {"", "empty"}:
        return None

    m = re.search(r"-?\d+", s)
    if not m:
        return None

    v = int(m.group(0))
    if min_v is not None and v < min_v:
        return None
    if max_v is not None and v > max_v:
        return None
    return str(v)


def normalize_volume_candidate(x):
    return normalize_numeric_int_candidate(x, min_v=0, max_v=10000)


def normalize_issue_candidate(x):
    return normalize_numeric_int_candidate(x, min_v=0, max_v=1000)


def normalize_pagination_candidate(x):
    s = canonical_empty_text(x)
    if s in {"", "empty"}:
        return None
    s2 = s.strip()
    s2 = s2.replace("–", "-").replace("—", "-").replace("−", "-")
    s2 = re.sub(r"\s*-\s*", "-", s2)
    if re.search(r"\d", s2):
        return s2
    return None


def normalize_by_column(column, value):
    col = norm_text(column)
    if col == "article_language":
        return normalize_language_candidate(value)
    if col == "journal_issn":
        return normalize_issn_candidate(value)
    if col == "article_jcreated_at":
        return normalize_date_candidate(value)
    if col == "article_jvolumn":
        return normalize_volume_candidate(value)
    if col == "article_jissue":
        return normalize_issue_candidate(value)
    if col == "article_pagination":
        return normalize_pagination_candidate(value)
    if col in {"index", "id"}:
        return normalize_numeric_int_candidate(value, min_v=0)
    return None


def value_equal_for_column(column, a, b):
    fa = normalize_by_column(column, a)
    fb = normalize_by_column(column, b)
    if fa is not None and fb is not None:
        return norm_text(fa) == norm_text(fb)
    return norm_text(a) == norm_text(b)


# ============================================================
# 5. 统一整理上下文（rich jsonl + fallback flat）
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
        context_rule_count = 0
        global_rule_count = 0
        rare_value_count = 0
        typo_rule_count = 0
        pattern_rule_count = 0
        schema_rule_count = 0
        canonical_rule_count = 0

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

            strong_rule_count = int(conflict_context.get("strong_rule_count", 0) or 0)
            candidate_generation_rule_count = int(conflict_context.get("candidate_generation_rule_count", 0) or 0)
            fd_like_count = int(conflict_context.get("fd_like_count", 0) or 0)
            context_rule_count = int(conflict_context.get("context_rule_count", 0) or 0)
            global_rule_count = int(conflict_context.get("global_rule_count", 0) or 0)
            rare_value_count = int(conflict_context.get("rare_value_count", 0) or 0)
            typo_rule_count = int(conflict_context.get("typo_rule_count", 0) or 0)
            pattern_rule_count = int(conflict_context.get("pattern_rule_count", 0) or 0)
            schema_rule_count = int(conflict_context.get("schema_rule_count", 0) or 0)
            canonical_rule_count = int(conflict_context.get("canonical_rule_count", obj.get("canonical_rule_count", 0)) or 0)

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
                        fixed = normalize_by_column(column, tv)
                        similar_row_target_values.append(fixed if fixed is not None else tv)

        column_top_values = []
        column_detected_type = obj.get("detected_type", None)
        column_numeric_stats = None
        column_date_stats = None
        column_current_pattern = None
        column_dominant_pattern = None
        col_ctx = obj.get("column_context", {})
        if isinstance(col_ctx, dict):
            top_vals = col_ctx.get("top_values_with_counts", []) or []
            for item in top_vals:
                if isinstance(item, dict) and item.get("value", None) is not None:
                    val = canonical_empty_text(item["value"])
                    if val not in {"", "empty"}:
                        fixed = normalize_by_column(column, val)
                        column_top_values.append({
                            "value": fixed if fixed is not None else val,
                            "raw_value": val,
                            "count": int(item.get("count", 0) or 0),
                            "ratio": safe_float(item.get("ratio", 0.0), 0.0),
                        })
            column_detected_type = col_ctx.get("detected_type", column_detected_type)
            column_numeric_stats = col_ctx.get("numeric_stats", None)
            column_date_stats = col_ctx.get("date_stats", None)
            column_current_pattern = col_ctx.get("current_value_pattern", None)
            column_dominant_pattern = col_ctx.get("dominant_pattern", None)

        if not column_top_values:
            counter = column_value_counter[column]
            total = max(sum(counter.values()), 1)
            for val, cnt in counter.most_common(10):
                fixed = normalize_by_column(column, val)
                column_top_values.append({
                    "value": fixed if fixed is not None else val,
                    "raw_value": val,
                    "count": cnt,
                    "ratio": cnt / total,
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

        context_map[(row_id, column)] = {
            "row_id": row_id,
            "column": column,
            "value": value,
            "semantic_type": obj.get("semantic_type", None),
            "violation_count": obj.get("violation_count", None),
            "conflict_score": obj.get("conflict_score", None),
            "main_rule_type": obj.get("main_rule_type", None),
            "main_usage_role": obj.get("main_usage_role", None),
            "numeric_value": obj.get("numeric_value", None),
            "date_value": obj.get("date_value", None),
            "language_canonical": obj.get("language_canonical", None),
            "issn_canonical": obj.get("issn_canonical", None),
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
            "column_date_stats": column_date_stats,
            "column_current_pattern": column_current_pattern,
            "column_dominant_pattern": column_dominant_pattern,
            "column_detected_type": column_detected_type,
            "prior_error_probability": prior_error_probability,
            "posterior_error_probability": posterior_error_probability,
            "bayes_rule_signal": bayes_rule_signal,
            "bayes_rarity_signal": bayes_rarity_signal,
            "bayes_pattern_signal": bayes_pattern_signal,
            "bayes_neighbor_signal": bayes_neighbor_signal,
            "strong_rule_count": strong_rule_count,
            "candidate_generation_rule_count": candidate_generation_rule_count,
            "fd_like_count": fd_like_count,
            "context_rule_count": context_rule_count,
            "global_rule_count": global_rule_count,
            "rare_value_count": rare_value_count,
            "typo_rule_count": typo_rule_count,
            "pattern_rule_count": pattern_rule_count,
            "schema_rule_count": schema_rule_count,
            "canonical_rule_count": canonical_rule_count,
            "journal_title_value": canonical_empty_text(row_values.get("journal_title", "")),
            "journal_abbreviation_value": canonical_empty_text(row_values.get("jounral_abbreviation", row_values.get("journal_abbreviation", ""))),
            "journal_issn_value": canonical_empty_text(row_values.get("journal_issn", "")),
            "article_language_value": canonical_empty_text(row_values.get("article_language", "")),
            "article_jvolumn_value": canonical_empty_text(row_values.get("article_jvolumn", "")),
            "article_jissue_value": canonical_empty_text(row_values.get("article_jissue", "")),
            "article_jcreated_at_value": canonical_empty_text(row_values.get("article_jcreated_at", "")),
            "article_pagination_value": canonical_empty_text(row_values.get("article_pagination", "")),
            "article_title_value": canonical_empty_text(row_values.get("article_title", "")),
            "id_value": canonical_empty_text(row_values.get("id", "")),
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
                "numeric_value": obj.get("numeric_value", None),
                "date_value": obj.get("date_value", None),
                "language_canonical": obj.get("language_canonical", None),
                "issn_canonical": obj.get("issn_canonical", None),
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
                "column_date_stats": None,
                "column_current_pattern": None,
                "column_dominant_pattern": None,
                "column_detected_type": obj.get("detected_type", None),
                "prior_error_probability": 0.0,
                "posterior_error_probability": 0.0,
                "bayes_rule_signal": 0.0,
                "bayes_rarity_signal": 0.0,
                "bayes_pattern_signal": 0.0,
                "bayes_neighbor_signal": 0.0,
                "strong_rule_count": 0,
                "candidate_generation_rule_count": 0,
                "fd_like_count": 0,
                "context_rule_count": 0,
                "global_rule_count": 0,
                "rare_value_count": 0,
                "typo_rule_count": 0,
                "pattern_rule_count": 0,
                "schema_rule_count": 0,
                "canonical_rule_count": obj.get("canonical_rule_count", 0),
                "journal_title_value": "",
                "journal_abbreviation_value": "",
                "journal_issn_value": "",
                "article_language_value": "",
                "article_jvolumn_value": "",
                "article_jissue_value": "",
                "article_jcreated_at_value": "",
                "article_pagination_value": "",
                "article_title_value": "",
                "id_value": "",
            }

        ctx = context_map[key]
        for field in [
            "semantic_type", "violation_count", "conflict_score", "main_rule_type",
            "main_usage_role", "numeric_value", "date_value", "language_canonical",
            "issn_canonical", "canonical_rule_count"
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
# 6. 规则满足性特征
# ============================================================

def build_rule_vote_counter(ctx):
    counter = Counter()
    for ev in ctx.get("rule_expected_values", []):
        ev = canonical_empty_text(ev)
        if ev not in {"", "empty"}:
            fixed = normalize_by_column(ctx.get("column", ""), ev)
            counter[fixed if fixed is not None else ev] += 1

    for ev in ctx.get("expected_values_from_rules", []):
        ev = canonical_empty_text(ev)
        if ev not in {"", "empty"}:
            fixed = normalize_by_column(ctx.get("column", ""), ev)
            counter[fixed if fixed is not None else ev] += 2

    return counter


def compute_rule_features(candidate_value, dirty_value, ctx):
    vote_counter = build_rule_vote_counter(ctx)
    total_votes = sum(vote_counter.values())

    column = ctx.get("column", "")
    cand_norms = {canonical_empty_text(candidate_value)}
    dirty_norms = {canonical_empty_text(dirty_value)}
    fixed_cand = normalize_by_column(column, candidate_value)
    fixed_dirty = normalize_by_column(column, dirty_value)
    if fixed_cand is not None:
        cand_norms.add(canonical_empty_text(fixed_cand))
    if fixed_dirty is not None:
        dirty_norms.add(canonical_empty_text(fixed_dirty))

    cand_vote = max([vote_counter.get(v, 0) for v in cand_norms] + [0])
    dirty_vote = max([vote_counter.get(v, 0) for v in dirty_norms] + [0])

    candidate_rule_support_ratio = cand_vote / total_votes if total_votes > 0 else 0.0
    dirty_rule_support_ratio = dirty_vote / total_votes if total_votes > 0 else 0.0

    alt_values = set()
    for x in ctx.get("alternative_values_from_column", []):
        if isinstance(x, dict):
            v = canonical_empty_text(x.get("value", ""))
            if v not in {"", "empty"}:
                alt_values.add(v)
                fixed = normalize_by_column(column, v)
                if fixed is not None:
                    alt_values.add(fixed)

    sim_row_values = []
    for v in ctx.get("similar_row_target_values", []):
        vv = canonical_empty_text(v)
        if vv not in {"", "empty"}:
            sim_row_values.append(vv)
            fixed = normalize_by_column(column, vv)
            if fixed is not None:
                sim_row_values.append(fixed)

    return {
        "rule_total_votes": total_votes,
        "candidate_rule_vote_count": cand_vote,
        "dirty_rule_vote_count": dirty_vote,
        "candidate_rule_support_ratio": candidate_rule_support_ratio,
        "dirty_rule_support_ratio": dirty_rule_support_ratio,
        "rule_support_gain": candidate_rule_support_ratio - dirty_rule_support_ratio,
        "candidate_matches_any_rule_expected": int(cand_vote > 0),
        "candidate_matches_any_strong_rule_expected": int(cand_vote > 0 and int(ctx.get("strong_rule_count", 0) or 0) > 0),
        "candidate_matches_any_expected_values_from_rules": int(any(v in cand_norms for v in vote_counter.keys())),
        "candidate_in_alternative_values_from_column": int(any(v in alt_values for v in cand_norms)),
        "candidate_equals_neighbor_majority": int(
            ctx.get("neighbor_majority_value", None) is not None and
            value_equal_for_column(column, candidate_value, ctx.get("neighbor_majority_value"))
        ),
        "candidate_equals_suggested_correct_value": int(
            ctx.get("suggested_correct_value", None) is not None and
            value_equal_for_column(column, candidate_value, ctx.get("suggested_correct_value"))
        ),
        "candidate_matches_similar_row_target": int(any(value_equal_for_column(column, candidate_value, v) for v in sim_row_values)),
        "candidate_match_count_in_similar_rows": sum(1 for v in sim_row_values if value_equal_for_column(column, candidate_value, v)),
        "candidate_match_ratio_in_similar_rows": (
            sum(1 for v in sim_row_values if value_equal_for_column(column, candidate_value, v)) / len(sim_row_values)
            if sim_row_values else 0.0
        ),
        "strong_rule_count": int(ctx.get("strong_rule_count", 0) or 0),
        "candidate_generation_rule_count": int(ctx.get("candidate_generation_rule_count", 0) or 0),
        "fd_like_count": int(ctx.get("fd_like_count", 0) or 0),
        "context_rule_count": int(ctx.get("context_rule_count", 0) or 0),
        "global_rule_count": int(ctx.get("global_rule_count", 0) or 0),
        "rare_value_count": int(ctx.get("rare_value_count", 0) or 0),
        "typo_rule_count": int(ctx.get("typo_rule_count", 0) or 0),
        "pattern_rule_count": int(ctx.get("pattern_rule_count", 0) or 0),
        "schema_rule_count": int(ctx.get("schema_rule_count", 0) or 0),
        "canonical_rule_count": int(ctx.get("canonical_rule_count", 0) or 0),
    }


# ============================================================
# 7. 从原始数据集构建经验共现字典
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

                # 对 Rayyan 特殊列同步登记规范化值
                v_fixed = normalize_by_column(target_col, v)
                c_fixed = normalize_by_column(ctx_col, c)
                vv = v_fixed if v_fixed is not None else v
                cc = c_fixed if c_fixed is not None else c

                pair_counter[(vv, cc)] += 1
                ctx_counter[cc] += 1

            if not ctx_counter:
                continue

            for ctx_val, den in ctx_counter.items():
                for cand_val in target_domain:
                    cand_fixed = normalize_by_column(target_col, cand_val)
                    cand_key = cand_fixed if cand_fixed is not None else cand_val
                    num = pair_counter.get((cand_key, ctx_val), 0)
                    prob = (num + SMOOTHING_ALPHA) / (den + SMOOTHING_ALPHA * domain_size)

                    if cand_key not in cooccur[target_col]:
                        cooccur[target_col][cand_key] = {}
                    cooccur[target_col][cand_key][f"{ctx_col}={ctx_val}"] = prob

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
# 8. 贝叶斯共现概率 / P(v | context)
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

            fixed = normalize_by_column(k, vv)
            pairs.append((str(k), fixed if fixed is not None else vv))

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
    cand_fixed = normalize_by_column(target_column, candidate_value)
    cand_key = cand_fixed if cand_fixed is not None else candidate_value
    cand_map = col_map.get(cand_key, {})

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
# 9. 变换特征
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

    cand_fixed = normalize_by_column("", candidate_value)
    dirty_fixed = normalize_by_column("", dirty_value)

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
# 10. Rayyan metadata consensus 文件特征
# ============================================================

def load_rayyan_consensus_table(path):
    if not path_exists_nonempty(path):
        return {}

    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception as e:
        print(f"[WARN] 无法读取 Rayyan consensus 文件: {path}, err={e}")
        return {}

    required = {"rhs_column", "consensus_value", "consensus_name", "confidence", "margin", "support_count"}
    if not required.issubset(set(df.columns)):
        print(f"[WARN] Rayyan consensus 文件字段不完整，跳过: {path}")
        return {}

    # 这里不是按 row_id，而是按 rhs_column + consensus_value 聚合，辅助候选来源特征。
    out = defaultdict(list)
    for _, r in df.iterrows():
        col = str(r.get("rhs_column", ""))
        val = canonical_empty_text(r.get("consensus_value", ""))
        if col and val not in {"", "empty"}:
            fixed = normalize_by_column(col, val)
            key_val = fixed if fixed is not None else val
            out[(col, norm_text(key_val))].append({
                "consensus_name": r.get("consensus_name", ""),
                "confidence": safe_float(r.get("confidence", 0.0), 0.0),
                "margin": safe_float(r.get("margin", 0.0), 0.0),
                "support_count": safe_int(r.get("support_count", 0), 0),
                "total_count": safe_int(r.get("total_count", 0), 0),
                "is_high_confidence": safe_int(r.get("is_high_confidence", 0), 0),
            })

    return out


def lookup_consensus_features(consensus_index, column, candidate_value):
    fixed = normalize_by_column(column, candidate_value)
    key_val = fixed if fixed is not None else candidate_value
    items = consensus_index.get((str(column), norm_text(key_val)), [])

    if not items:
        return {
            "rayyan_consensus_table_match": 0,
            "rayyan_consensus_table_max_confidence": 0.0,
            "rayyan_consensus_table_max_margin": 0.0,
            "rayyan_consensus_table_max_support_count": 0,
            "rayyan_consensus_table_high_conf_match": 0,
            "rayyan_consensus_table_match_count": 0,
        }

    return {
        "rayyan_consensus_table_match": 1,
        "rayyan_consensus_table_max_confidence": max(safe_float(x.get("confidence", 0.0), 0.0) for x in items),
        "rayyan_consensus_table_max_margin": max(safe_float(x.get("margin", 0.0), 0.0) for x in items),
        "rayyan_consensus_table_max_support_count": max(safe_int(x.get("support_count", 0), 0) for x in items),
        "rayyan_consensus_table_high_conf_match": int(any(safe_int(x.get("is_high_confidence", 0), 0) == 1 for x in items)),
        "rayyan_consensus_table_match_count": len(items),
    }


# ============================================================
# 11. Rayyan plausibility 特征
# ============================================================

def compute_rayyan_plausibility_features(candidate_value, dirty_value, ctx):
    column = str(ctx.get("column", ""))
    col = norm_text(column)
    row_values = ctx.get("row_values", {}) or {}

    candidate_fixed = normalize_by_column(column, candidate_value)
    dirty_fixed = normalize_by_column(column, dirty_value)

    journal_title = canonical_empty_text(row_values.get("journal_title", ctx.get("journal_title_value", "")))
    journal_abbr = canonical_empty_text(row_values.get("jounral_abbreviation", row_values.get("journal_abbreviation", ctx.get("journal_abbreviation_value", ""))))
    journal_issn = canonical_empty_text(row_values.get("journal_issn", ctx.get("journal_issn_value", "")))
    article_language = canonical_empty_text(row_values.get("article_language", ctx.get("article_language_value", "")))
    article_jvolumn = canonical_empty_text(row_values.get("article_jvolumn", ctx.get("article_jvolumn_value", "")))
    article_jissue = canonical_empty_text(row_values.get("article_jissue", ctx.get("article_jissue_value", "")))
    article_jcreated_at = canonical_empty_text(row_values.get("article_jcreated_at", ctx.get("article_jcreated_at_value", "")))
    article_pagination = canonical_empty_text(row_values.get("article_pagination", ctx.get("article_pagination_value", "")))
    article_title = canonical_empty_text(row_values.get("article_title", ctx.get("article_title_value", "")))
    article_id = canonical_empty_text(row_values.get("id", ctx.get("id_value", "")))

    cand_lang = normalize_language_candidate(candidate_value)
    dirty_lang = normalize_language_candidate(dirty_value)
    row_lang = normalize_language_candidate(article_language)

    cand_issn = normalize_issn_candidate(candidate_value)
    dirty_issn = normalize_issn_candidate(dirty_value)
    row_issn = normalize_issn_candidate(journal_issn)

    cand_date = normalize_date_candidate(candidate_value)
    dirty_date = normalize_date_candidate(dirty_value)
    row_date = normalize_date_candidate(article_jcreated_at)

    cand_vol = normalize_volume_candidate(candidate_value)
    dirty_vol = normalize_volume_candidate(dirty_value)
    row_vol = normalize_volume_candidate(article_jvolumn)

    cand_issue = normalize_issue_candidate(candidate_value)
    dirty_issue = normalize_issue_candidate(dirty_value)
    row_issue = normalize_issue_candidate(article_jissue)

    cand_pagination = normalize_pagination_candidate(candidate_value)
    dirty_pagination = normalize_pagination_candidate(dirty_value)

    return {
        "rayyan_column_is_article_language": int(col == "article_language"),
        "rayyan_column_is_journal_issn": int(col == "journal_issn"),
        "rayyan_column_is_journal_title": int(col == "journal_title"),
        "rayyan_column_is_journal_abbreviation": int(col in {"jounral_abbreviation", "journal_abbreviation"}),
        "rayyan_column_is_article_jcreated_at": int(col == "article_jcreated_at"),
        "rayyan_column_is_article_jvolumn": int(col == "article_jvolumn"),
        "rayyan_column_is_article_jissue": int(col == "article_jissue"),
        "rayyan_column_is_article_pagination": int(col == "article_pagination"),
        "rayyan_column_is_article_title": int(col == "article_title"),

        "rayyan_candidate_is_language": int(cand_lang is not None),
        "rayyan_dirty_is_language": int(dirty_lang is not None),
        "rayyan_candidate_language_equals_row_language": int(cand_lang is not None and row_lang is not None and cand_lang == row_lang),
        "rayyan_candidate_language_restoration": int(cand_lang is not None and dirty_lang is None),

        "rayyan_candidate_is_issn": int(cand_issn is not None),
        "rayyan_dirty_is_issn": int(dirty_issn is not None),
        "rayyan_candidate_is_valid_issn": int(cand_issn is not None and is_valid_issn(cand_issn)),
        "rayyan_dirty_is_valid_issn": int(dirty_issn is not None and is_valid_issn(dirty_issn)),
        "rayyan_candidate_issn_equals_row_issn": int(cand_issn is not None and row_issn is not None and cand_issn == row_issn),
        "rayyan_candidate_issn_restoration": int(cand_issn is not None and dirty_issn is None),

        "rayyan_candidate_is_date": int(cand_date is not None),
        "rayyan_dirty_is_date": int(dirty_date is not None),
        "rayyan_candidate_date_equals_row_date": int(cand_date is not None and row_date is not None and cand_date == row_date),
        "rayyan_candidate_date_restoration": int(cand_date is not None and dirty_date is None),

        "rayyan_candidate_is_volume": int(cand_vol is not None),
        "rayyan_dirty_is_volume": int(dirty_vol is not None),
        "rayyan_candidate_volume_equals_row_volume": int(cand_vol is not None and row_vol is not None and cand_vol == row_vol),

        "rayyan_candidate_is_issue": int(cand_issue is not None),
        "rayyan_dirty_is_issue": int(dirty_issue is not None),
        "rayyan_candidate_issue_equals_row_issue": int(cand_issue is not None and row_issue is not None and cand_issue == row_issue),

        "rayyan_candidate_is_pagination": int(cand_pagination is not None),
        "rayyan_dirty_is_pagination": int(dirty_pagination is not None),
        "rayyan_candidate_pagination_restoration": int(cand_pagination is not None and dirty_pagination is None),

        "rayyan_candidate_normalized_equals_dirty": int(candidate_fixed is not None and dirty_fixed is not None and norm_text(candidate_fixed) == norm_text(dirty_fixed)),
        "rayyan_candidate_is_format_restoration": int(candidate_fixed is not None and norm_text(candidate_fixed) != norm_text(candidate_value)),
        "rayyan_candidate_exact_matches_journal_title": int(col == "journal_title" and norm_text(candidate_value) == norm_text(journal_title)),
        "rayyan_candidate_exact_matches_journal_abbr": int(col in {"jounral_abbreviation", "journal_abbreviation"} and norm_text(candidate_value) == norm_text(journal_abbr)),
        "rayyan_candidate_exact_matches_article_title": int(col == "article_title" and norm_text(candidate_value) == norm_text(article_title)),
        "rayyan_row_has_journal_title": int(journal_title not in {"", "empty"}),
        "rayyan_row_has_journal_abbr": int(journal_abbr not in {"", "empty"}),
        "rayyan_row_has_issn": int(row_issn is not None),
        "rayyan_row_has_language": int(row_lang is not None),
        "rayyan_row_has_volume": int(row_vol is not None),
        "rayyan_row_has_issue": int(row_issue is not None),
        "rayyan_row_has_date": int(row_date is not None),
        "rayyan_title_length": len(article_title) if article_title not in {"", "empty"} else 0,
    }


def compute_column_plausibility_features(candidate_value, dirty_value, ctx):
    column_top_values = ctx.get("column_top_values", [])
    top_values = []
    for x in column_top_values:
        if not isinstance(x, dict):
            continue
        v = canonical_empty_text(x.get("value", ""))
        if v not in {"", "empty"}:
            top_values.append((v, x))

    candidate_in_top_values = int(any(value_equal_for_column(ctx.get("column", ""), candidate_value, v) for v, _ in top_values))
    candidate_top_ratio = 0.0
    for v, item in top_values:
        if value_equal_for_column(ctx.get("column", ""), candidate_value, v):
            candidate_top_ratio = max(candidate_top_ratio, safe_float(item.get("ratio", 0.0), 0.0))

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

    out = {
        "candidate_in_column_top_values": candidate_in_top_values,
        "candidate_column_top_ratio": candidate_top_ratio,
        "candidate_in_numeric_range": candidate_in_numeric_range,
        "candidate_outside_numeric_range": candidate_outside_numeric_range,
    }
    out.update(compute_rayyan_plausibility_features(candidate_value, dirty_value, ctx))
    return out


# ============================================================
# 12. 主流程
# ============================================================

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
    consensus_index = load_rayyan_consensus_table(RAYYAN_METADATA_CONSENSUS_FILE)
    if consensus_index:
        print(f"[INFO] Rayyan consensus feature index entries: {len(consensus_index)}")

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
        consensus_feats = lookup_consensus_features(consensus_index, column, candidate_value)

        out = row.to_dict()

        candidate_source_text = str(row.get("candidate_source", "") or row.get("candidate_source_text", "") or "")
        sources = set([s.strip() for s in candidate_source_text.split("|") if s.strip()])

        is_rayyan_consensus_source = int("rayyan_metadata_consensus" in sources)
        is_canonical_language_source = int("canonical_language_restore" in sources)
        is_canonical_issn_source = int("canonical_issn_restore" in sources)
        is_canonical_date_source = int("canonical_date_restore" in sources)
        is_canonical_volume_source = int("canonical_volume_restore" in sources)
        is_canonical_issue_source = int("canonical_issue_restore" in sources)
        is_canonical_pagination_source = int("canonical_pagination_restore" in sources)

        contains_journal_metadata_source = int(any(s in sources for s in {
            "journal_title_to_language",
            "journal_abbr_to_language",
            "issn_to_language",
            "journal_title_to_issn",
            "journal_abbr_to_issn",
            "issn_to_journal_title",
            "issn_to_journal_abbr",
            "journal_issue_to_volume",
            "journal_volume_to_issue",
            "journal_to_volume",
            "journal_to_issue",
        }))

        contains_article_metadata_source = int(any(s in sources for s in {
            "article_id_to_title",
            "title_to_language",
        }))

        contains_canonical_source = int(any(s in sources for s in {
            "canonical_language_restore",
            "canonical_issn_restore",
            "canonical_date_restore",
            "canonical_volume_restore",
            "canonical_issue_restore",
            "canonical_pagination_restore",
        }))

        out.update({
            "semantic_type": ctx.get("semantic_type", None),
            "violation_count": ctx.get("violation_count", None),
            "conflict_score": ctx.get("conflict_score", None),
            "main_rule_type": ctx.get("main_rule_type", None),
            "main_usage_role": ctx.get("main_usage_role", None),
            "numeric_value": ctx.get("numeric_value", None),
            "date_value": ctx.get("date_value", None),
            "language_canonical": ctx.get("language_canonical", None),
            "issn_canonical": ctx.get("issn_canonical", None),
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
            "contains_dictionary_source": int(contains_journal_metadata_source or contains_article_metadata_source or is_rayyan_consensus_source),
            "contains_rayyan_metadata_source": int(contains_journal_metadata_source or contains_article_metadata_source),
            "contains_journal_metadata_source": contains_journal_metadata_source,
            "contains_article_metadata_source": contains_article_metadata_source,
            "contains_canonical_source": contains_canonical_source,
            "contains_rayyan_consensus_source": is_rayyan_consensus_source,

            # Rayyan source-aware consensus 特征
            "is_from_rayyan_consensus": int(row.get("rayyan_consensus_candidate", 0) or is_rayyan_consensus_source),
            "rayyan_consensus_candidate": int(row.get("rayyan_consensus_candidate", 0) or is_rayyan_consensus_source),
            "rayyan_consensus_info": row.get("rayyan_consensus_info", ""),
            "rayyan_consensus_confidence_from_info": parse_consensus_info_number(row.get("rayyan_consensus_info", ""), "confidence"),
            "rayyan_consensus_margin_from_info": parse_consensus_info_number(row.get("rayyan_consensus_info", ""), "margin"),
            "rayyan_consensus_support_count_from_info": parse_consensus_info_number(row.get("rayyan_consensus_info", ""), "support_count"),

            # Rayyan canonical source 细粒度
            "contains_canonical_language_source": is_canonical_language_source,
            "contains_canonical_issn_source": is_canonical_issn_source,
            "contains_canonical_date_source": is_canonical_date_source,
            "contains_canonical_volume_source": is_canonical_volume_source,
            "contains_canonical_issue_source": is_canonical_issue_source,
            "contains_canonical_pagination_source": is_canonical_pagination_source,

            # candidate-generation 阶段给出的 Rayyan flags
            "rayyan_is_language_candidate": int(row.get("rayyan_is_language_candidate", 0) or 0),
            "rayyan_is_issn_candidate": int(row.get("rayyan_is_issn_candidate", 0) or 0),
            "rayyan_is_valid_issn_candidate": int(row.get("rayyan_is_valid_issn_candidate", 0) or 0),
            "rayyan_is_date_candidate": int(row.get("rayyan_is_date_candidate", 0) or 0),
            "rayyan_is_volume_candidate": int(row.get("rayyan_is_volume_candidate", 0) or 0),
            "rayyan_is_issue_candidate": int(row.get("rayyan_is_issue_candidate", 0) or 0),
            "rayyan_is_pagination_candidate": int(row.get("rayyan_is_pagination_candidate", 0) or 0),
        })

        out.update(rule_feats)
        out.update(bayes_feats)
        out.update(trans_feats)
        out.update(plaus_feats)
        out.update(consensus_feats)

        feature_rows.append(out)

    features_df = pd.DataFrame(feature_rows)
    output_features_path = resolve_output_path(OUTPUT_FEATURES_CSV, CANDIDATES_FILE)
    features_df.to_csv(output_features_path, index=False, encoding="utf-8-sig")

    print(f"[INFO] 候选记录数: {len(candidates_df)}")
    print(f"[INFO] 成功构造特征的记录数: {len(features_df)}")
    print(f"[INFO] 缺失上下文的候选记录数: {missing_context_count}")
    print(f"[INFO] 输出文件: {output_features_path}")

    if not features_df.empty:
        show_cols = [c for c in [
            "row_id", "column", "dirty_value", "candidate_value",
            "candidate_rule_support_ratio", "rule_support_gain",
            "bayes_mean_cond_prob", "bayes_max_cond_prob",
            "edit_similarity_to_dirty", "avg_keyboard_distance",
            "phonetic_similarity_to_dirty", "candidate_in_column_top_values",
            "is_rule_and_dictionary_supported", "candidate_matches_similar_row_target",
            "contains_rayyan_metadata_source", "contains_rayyan_consensus_source",
            "contains_canonical_source",
            "rayyan_candidate_is_language", "rayyan_candidate_language_restoration",
            "rayyan_candidate_is_issn", "rayyan_candidate_is_valid_issn",
            "rayyan_candidate_is_date", "rayyan_candidate_is_volume",
            "rayyan_candidate_is_issue", "rayyan_candidate_is_pagination",
            "rayyan_consensus_table_match", "rayyan_consensus_table_max_confidence",
            "rayyan_consensus_table_max_support_count",
        ] if c in features_df.columns]
        print("\n[INFO] 前 20 行特征预览：")
        print(features_df[show_cols].head(20).to_string(index=False))


def parse_consensus_info_number(info, key):
    if info is None or pd.isna(info):
        return 0.0
    s = str(info)
    m = re.search(rf"{re.escape(key)}=([0-9.]+)", s)
    if not m:
        return 0.0
    return safe_float(m.group(1), 0.0)


if __name__ == "__main__":
    main()

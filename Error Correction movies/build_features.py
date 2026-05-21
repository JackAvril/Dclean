"""
Movies repair candidate feature builder.

完整迁移目标：
- 保留 Flights 版特征构建的主框架：候选展开文件、rich context、flat fallback、
  经验共现字典、规则投票、Bayes 共现、编辑/键盘/发音距离、候选来源特征、
  列合法性和 consistency 特征。
- 将 Flights 中的 time / flight-consensus 逻辑迁移为 Movies 的：
  duration / year / release date / country / rating / count / review / people-list /
  movie metadata consistency 特征。

输入：
    repair_candidates_expanded.csv
    candidate_llm_contexts_movies.jsonl
    candidate_llm_contexts_flat_movies.csv 可选
    movies_dirty.csv 可选，用于构建共现字典

输出：
    repair_candidate_features_v2.csv
    built_cooccurrence_dict_v2.json
"""

import os
import json
import math
import re
from difflib import SequenceMatcher
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# ============================================================
# 1. 配置区
# ============================================================

CANDIDATES_FILE = "/mnt/mydata/dq/projects/Splittree/test/Error Correction movies/repair_candidates_expanded.csv"
CONTEXT_FILE = "/mnt/mydata/dq/projects/Splittree/test/Error Detection movies/candidate_llm_contexts_movies.jsonl"
FALLBACK_CONTEXT_FLAT_FILE = "/mnt/mydata/dq/projects/Splittree/test/Error Detection movies/candidate_llm_contexts_flat_movies.csv"
RAW_DATA_FILE = "/mnt/mydata/dq/projects/Splittree/movies/movies_dirty.csv"

COOCCURRENCE_DICT_FILE = None

MOVIE_CONSISTENCY_SUSPICIOUS_CELLS_FILE = "movie_consistency_suspicious_cells.csv"
# 新增：候选生成阶段输出的 direct / metadata suspicious cells。
# 用于给候选特征补充 allowed direct value、direct reason、metadata source 等强证据。
MOVIE_DIRECT_METADATA_SUSPICIOUS_CELLS_FILE = "movie_direct_metadata_suspicious_cells.csv"

OUTPUT_FEATURES_CSV = "repair_candidate_features_v2.csv"
OUTPUT_BUILT_COOCCUR_JSON = "built_cooccurrence_dict_v2.json"

MAX_CONTEXT_PAIRS = 14
SMOOTHING_ALPHA = 1.0
MAX_DISTINCT_VALUES_PER_COLUMN = 2500
TOP_K_SIMILAR_ROW_VALUES = 8
ENABLE_NUMERIC_CANDIDATE_CHECK = True


# ============================================================
# 2. 基础函数
# ============================================================

EMPTY_TOKENS = {"", "empty", "null", "none", "nan", "na", "n/a", "missing", "{null}"}

MONTH_NAME_TO_NUM = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}

MONTH_NUM_TO_NAME = {
    1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
    7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December",
}

COUNTRY_CANONICAL_MAP = {
    "usa": "USA", "us": "USA", "u.s.": "USA", "u.s.a.": "USA",
    "united states": "USA", "united states of america": "USA",
    "uk": "UK", "u.k.": "UK", "united kingdom": "UK", "england": "UK",
    "france": "France", "germany": "Germany", "india": "India", "canada": "Canada",
    "australia": "Australia", "italy": "Italy", "spain": "Spain", "japan": "Japan",
    "china": "China", "hong kong": "Hong Kong", "south korea": "South Korea",
    "korea": "South Korea", "russia": "Russia", "soviet union": "Soviet Union",
    "mexico": "Mexico", "brazil": "Brazil", "argentina": "Argentina",
    "sweden": "Sweden", "denmark": "Denmark", "norway": "Norway",
    "netherlands": "Netherlands", "ireland": "Ireland", "kuwait": "Kuwait",
}

LANGUAGE_CANONICAL_MAP = {
    "english": "English", "french": "French", "spanish": "Spanish", "german": "German",
    "italian": "Italian", "japanese": "Japanese", "chinese": "Chinese",
    "mandarin": "Mandarin", "hindi": "Hindi", "korean": "Korean",
    "russian": "Russian", "arabic": "Arabic", "portuguese": "Portuguese",
}


def norm_text(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def canonical_empty_text(x: Any) -> str:
    if pd.isna(x):
        return "empty"
    s = str(x).strip()
    return "empty" if s.lower() in EMPTY_TOKENS else s


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or pd.isna(x):
            return default
        return float(str(x).replace(",", "").replace("%", "").strip())
    except Exception:
        return default


def safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None or pd.isna(x):
            return default
        return int(float(str(x).replace(",", "").strip()))
    except Exception:
        return default


def try_parse_float(x: Any) -> Optional[float]:
    try:
        if x is None or pd.isna(x):
            return None
        s = str(x).replace(",", "").replace("%", "").strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


def path_exists_nonempty(path: Any) -> bool:
    return path is not None and str(path).strip() != "" and os.path.exists(str(path))


def resolve_output_path(output_path: Any, anchor_input_path: Any = None) -> str:
    if os.path.isabs(str(output_path)):
        return str(output_path)
    return os.path.abspath(str(output_path))


def string_similarity(a: Any, b: Any) -> float:
    a = norm_text(a)
    b = norm_text(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def clean_key(x: Any) -> str:
    return re.sub(r"\s+", " ", norm_text(x)).strip()


def load_jsonl(path: str) -> List[dict]:
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


def load_context_records(path: str) -> List[dict]:
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
# 3. 键盘距离和发音距离
# ============================================================

KEYBOARD_ROWS = ["1234567890", "qwertyuiop", "asdfghjkl", "zxcvbnm"]
KEYBOARD_POS = {}
for r, row in enumerate(KEYBOARD_ROWS):
    for c, ch in enumerate(row):
        KEYBOARD_POS[ch] = (r, c)


def char_keyboard_distance(a: str, b: str) -> Optional[float]:
    a = norm_text(a)
    b = norm_text(b)
    if len(a) != 1 or len(b) != 1:
        return None
    if a not in KEYBOARD_POS or b not in KEYBOARD_POS:
        return None
    ra, ca = KEYBOARD_POS[a]
    rb, cb = KEYBOARD_POS[b]
    return abs(ra - rb) + abs(ca - cb)


def avg_keyboard_distance_for_same_length(a: Any, b: Any) -> Optional[float]:
    a = norm_text(a)
    b = norm_text(b)
    if len(a) != len(b) or len(a) == 0:
        return None
    vals = []
    for x, y in zip(a, b):
        if x == y:
            vals.append(0.0)
        else:
            d = char_keyboard_distance(x, y)
            vals.append(float(d if d is not None else 5.0))
    return sum(vals) / len(vals)


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
        if code != prev:
            if code:
                codes.append(code)
        prev = code
    return (first + "".join(codes) + "000")[:4]


def phrase_soundex(s: Any) -> str:
    toks = [t for t in re.split(r"[\s,\-/|;:]+", norm_text(s)) if t]
    return " ".join(soundex_token(t) for t in toks)


def phonetic_similarity(a: Any, b: Any) -> float:
    return string_similarity(phrase_soundex(a), phrase_soundex(b))


# ============================================================
# 4. Movies 字段规范化
# ============================================================

def split_list_tokens(x: Any) -> List[str]:
    s = canonical_empty_text(x)
    if s in {"", "empty"}:
        return []
    parts = re.split(r"[,;/|]+", s)
    out, seen = [], set()
    for p in parts:
        p = re.sub(r"\s+", " ", str(p).strip())
        if not p:
            continue
        k = p.lower()
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out


def normalize_list_value(x: Any) -> Optional[str]:
    toks = split_list_tokens(x)
    return ",".join(toks) if toks else None


def list_jaccard(a: Any, b: Any) -> float:
    ta = {norm_text(x) for x in split_list_tokens(a)}
    tb = {norm_text(x) for x in split_list_tokens(b)}
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def list_overlap_ratio(a: Any, b: Any) -> float:
    ta = {norm_text(x) for x in split_list_tokens(a)}
    tb = {norm_text(x) for x in split_list_tokens(b)}
    if not ta:
        return 0.0
    return len(ta & tb) / len(ta)


def normalize_year_candidate(x: Any) -> Optional[str]:
    s = canonical_empty_text(x)
    if s in {"", "empty"}:
        return None
    m = re.search(r"(18\d{2}|19\d{2}|20\d{2}|21\d{2})", s)
    if not m:
        return None
    y = int(m.group(1))
    return str(y) if 1800 <= y <= 2100 else None


def normalize_country_candidate(x: Any) -> Optional[str]:
    s = canonical_empty_text(x)
    if s in {"", "empty"}:
        return None
    k = norm_text(s)
    if k in COUNTRY_CANONICAL_MAP:
        return COUNTRY_CANONICAL_MAP[k]
    if re.fullmatch(r"[A-Za-z][A-Za-z .'-]{1,50}", s.strip()):
        return re.sub(r"\s+", " ", s.strip()).title()
    return None


def normalize_language_candidate(x: Any) -> Optional[str]:
    s = canonical_empty_text(x)
    if s in {"", "empty"}:
        return None
    toks = split_list_tokens(s)
    fixed = []
    for t in toks:
        fixed.append(LANGUAGE_CANONICAL_MAP.get(norm_text(t), t.strip().title()))
    return ",".join(fixed) if fixed else None


def parse_release_date_parts(x: Any) -> Optional[Dict[str, Any]]:
    s = canonical_empty_text(x)
    if s in {"", "empty"}:
        return None
    country = None
    cm = re.search(r"\(([^()]+)\)", s)
    if cm:
        country = normalize_country_candidate(cm.group(1)) or cm.group(1).strip()

    body = re.sub(r"\([^()]+\)", "", s).strip()

    m = re.search(r"\b(\d{1,2})\s+([A-Za-z]+)\s+(18\d{2}|19\d{2}|20\d{2}|21\d{2})\b", body)
    if m:
        day = int(m.group(1))
        month = MONTH_NAME_TO_NUM.get(m.group(2).lower())
        year = int(m.group(3))
        if month and 1 <= day <= 31:
            return {"year": year, "month": month, "day": day, "country": country}

    m = re.search(r"\b([A-Za-z]+)\s+(18\d{2}|19\d{2}|20\d{2}|21\d{2})\b", body)
    if m:
        month = MONTH_NAME_TO_NUM.get(m.group(1).lower())
        year = int(m.group(2))
        if month:
            return {"year": year, "month": month, "day": None, "country": country}

    m = re.search(r"\b(18\d{2}|19\d{2}|20\d{2}|21\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", body)
    if m:
        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= month <= 12 and 1 <= day <= 31:
            return {"year": year, "month": month, "day": day, "country": country}

    m = re.search(r"\b(\d{1,2})/(\d{1,2})/(18\d{2}|19\d{2}|20\d{2}|21\d{2})\b", body)
    if m:
        month, day, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= month <= 12 and 1 <= day <= 31:
            return {"year": year, "month": month, "day": day, "country": country}

    y = normalize_year_candidate(body)
    if y:
        return {"year": int(y), "month": None, "day": None, "country": country}

    return None


def normalize_release_date_candidate(x: Any) -> Optional[str]:
    parts = parse_release_date_parts(x)
    if not parts or not parts.get("year"):
        return None
    year, month, day, country = parts.get("year"), parts.get("month"), parts.get("day"), parts.get("country")
    if month and day:
        base = f"{int(day)} {MONTH_NUM_TO_NAME[int(month)]} {int(year)}"
    elif month:
        base = f"{MONTH_NUM_TO_NAME[int(month)]} {int(year)}"
    else:
        base = str(int(year))
    if country:
        base = f"{base} ({country})"
    return base


def normalize_duration_candidate(x: Any) -> Optional[str]:
    s = canonical_empty_text(x)
    if s in {"", "empty"}:
        return None
    h = 0
    m = 0
    mh = re.search(r"(\d+)\s*(?:h|hr|hrs|hour|hours)\b", s, flags=re.I)
    mm = re.search(r"(\d+)\s*(?:m|min|mins|minute|minutes)\b", s, flags=re.I)
    if mh or mm:
        if mh:
            h = int(mh.group(1))
        if mm:
            m = int(mm.group(1))
        total = h * 60 + m
        if 1 <= total <= 500:
            return f"{total} min"
    mnum = re.search(r"\b(\d{1,3})(?:\.0)?\b", s)
    if mnum:
        num = int(mnum.group(1))
        if 1 <= num <= 500:
            return f"{num} min"
    return None


def duration_minutes(x: Any) -> Optional[int]:
    fixed = normalize_duration_candidate(x)
    if not fixed:
        return None
    m = re.search(r"(\d+)", fixed)
    return int(m.group(1)) if m else None


def normalize_rating_value_candidate(x: Any) -> Optional[str]:
    s = canonical_empty_text(x)
    if s in {"", "empty"}:
        return None
    raw = s.strip().replace(",", ".")
    m = re.search(r"(\d+(?:\.\d+)?)", raw)
    if not m:
        return None
    v = float(m.group(1))
    if "%" in raw and 0 <= v <= 100:
        v = v / 10.0
    if 0 <= v <= 10:
        return f"{int(round(v))}.0" if abs(v - round(v)) < 1e-9 else f"{v:.1f}".rstrip("0").rstrip(".")
    return None


def normalize_count_candidate(x: Any) -> Optional[str]:
    s = canonical_empty_text(x)
    if s in {"", "empty"}:
        return None
    m = re.search(r"(\d[\d,]*)", s)
    if not m:
        return None
    v = int(m.group(1).replace(",", ""))
    return str(v) if v >= 0 else None


def normalize_review_count_candidate(x: Any) -> Optional[str]:
    s = canonical_empty_text(x)
    if s in {"", "empty"}:
        return None
    nums = [int(x.replace(",", "")) for x in re.findall(r"\d[\d,]*", s)]
    if not nums:
        return None
    lower = norm_text(s)
    if "user" in lower or "critic" in lower:
        parts = []
        um = re.search(r"(\d[\d,]*)\s*user", lower)
        cm = re.search(r"(\d[\d,]*)\s*critic", lower)
        if um:
            u = int(um.group(1).replace(",", ""))
            parts.append(f"{u} user" + ("" if u == 1 else "s"))
        if cm:
            c = int(cm.group(1).replace(",", ""))
            parts.append(f"{c} critic" + ("" if c == 1 else "s"))
        if parts:
            return ",".join(parts)
    return str(sum(nums))


def review_total(x: Any) -> Optional[int]:
    fixed = normalize_review_count_candidate(x)
    if not fixed:
        return None
    nums = [int(n) for n in re.findall(r"\d+", fixed)]
    return sum(nums) if nums else None


def normalize_by_movie_column(column: Any, value: Any) -> Optional[str]:
    cl = str(column).strip().lower()
    if cl == "year":
        return normalize_year_candidate(value)
    if cl == "release date":
        return normalize_release_date_candidate(value)
    if cl in {"country", "release country"}:
        return normalize_country_candidate(value)
    if cl == "language":
        return normalize_language_candidate(value)
    if cl == "duration":
        return normalize_duration_candidate(value)
    if cl == "ratingvalue":
        return normalize_rating_value_candidate(value)
    if cl in {"ratingcount", "id"}:
        return normalize_count_candidate(value)
    if cl == "reviewcount":
        return normalize_review_count_candidate(value)
    if cl in {"actors", "cast", "creator", "director", "genre", "filming locations"}:
        return normalize_list_value(value)
    return None


def value_equal_for_column(column: Any, a: Any, b: Any) -> bool:
    fa = normalize_by_movie_column(column, a)
    fb = normalize_by_movie_column(column, b)
    if fa is not None and fb is not None:
        return norm_text(fa) == norm_text(fb)
    return norm_text(a) == norm_text(b)


def is_movie_time_col(column: Any, semantic_type: Any = None) -> int:
    cl = str(column).strip().lower()
    sem = norm_text(semantic_type)
    return int(cl in {"year", "release date", "duration"} or sem in {"year_like", "date_like", "duration_like"})


def is_movie_rating_col(column: Any, semantic_type: Any = None) -> int:
    cl = str(column).strip().lower()
    sem = norm_text(semantic_type)
    return int(cl in {"ratingvalue", "ratingcount", "reviewcount"} or sem in {"rating_like", "count_like", "review_count_like"})


def is_movie_people_col(column: Any) -> int:
    return int(str(column).strip().lower() in {"director", "creator", "actors", "cast"})


def is_movie_list_col(column: Any, semantic_type: Any = None) -> int:
    cl = str(column).strip().lower()
    sem = norm_text(semantic_type)
    return int(cl in {"actors", "cast", "creator", "genre", "filming locations"} or sem == "categorical_list")


def is_movie_text_col(column: Any, semantic_type: Any = None) -> int:
    cl = str(column).strip().lower()
    sem = norm_text(semantic_type)
    return int(cl in {"name", "description"} or sem == "text")


def has_mojibake_or_control_chars(x: Any) -> int:
    s = canonical_empty_text(x)
    if s in {"", "empty"}:
        return 0
    if any(ord(ch) < 32 and ch not in "\t\n\r" for ch in s):
        return 1
    if "�" in s or "Ã" in s or "Â" in s:
        return 1
    return 0


# ============================================================
# 5. 上下文整理
# ============================================================

def build_context_map_rich(context_records: List[dict]) -> Dict[Tuple[int, str], dict]:
    context_map = {}
    column_value_counter = defaultdict(Counter)

    for obj in context_records:
        if "row_id" not in obj or "column" not in obj:
            continue
        val = canonical_empty_text(obj.get("value", ""))
        if val not in {"", "empty"}:
            fixed = normalize_by_movie_column(obj.get("column", ""), val)
            column_value_counter[str(obj["column"])][fixed if fixed else val] += 1

    for obj in context_records:
        if "row_id" not in obj or "column" not in obj:
            continue
        try:
            row_id = int(obj["row_id"])
        except Exception:
            continue
        column = str(obj["column"])
        value = canonical_empty_text(obj.get("value", ""))

        rc = obj.get("row_context", {})
        row_values = {}
        movie_bundle = {}
        if isinstance(rc, dict):
            rv = rc.get("row_values", {})
            if isinstance(rv, dict):
                row_values = {str(k): canonical_empty_text(v) for k, v in rv.items()}
            mb = rc.get("movie_metadata_bundle", {})
            if isinstance(mb, dict):
                movie_bundle = mb

        conflict_context = obj.get("conflict_context", {})
        rule_expected_values = []
        expected_values_from_rules = []
        alternative_values_from_column = []
        counters = {
            "strong_rule_count": 0,
            "candidate_generation_rule_count": 0,
            "fd_like_count": 0,
            "context_rule_count": 0,
            "global_rule_count": 0,
            "rare_value_count": 0,
            "typo_rule_count": 0,
            "pattern_rule_count": 0,
            "schema_rule_count": 0,
            "movie_typed_rule_count": 0,
            "movie_consistency_rule_count": 0,
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
                            fixed = normalize_by_movie_column(column, v)
                            alternative_values_from_column.append({
                                "value": fixed if fixed else v,
                                "count": safe_int(item.get("count", 0), 0),
                                "ratio": safe_float(item.get("ratio", 0.0), 0.0),
                                "source": item.get("source", ""),
                            })
            for k in counters:
                counters[k] = safe_int(conflict_context.get(k, obj.get(k, 0)), 0)

        similarity_context = obj.get("similarity_context", {})
        neighbor_majority_value = None
        neighbor_majority_ratio = 0.0
        similar_row_target_values = []
        if isinstance(similarity_context, dict):
            nmv = canonical_empty_text(similarity_context.get("neighbor_majority_value", ""))
            if nmv not in {"", "empty"}:
                fixed = normalize_by_movie_column(column, nmv)
                neighbor_majority_value = fixed if fixed else nmv
            neighbor_majority_ratio = safe_float(similarity_context.get("neighbor_majority_ratio", 0.0), 0.0)
            for item in (similarity_context.get("topk_similar_rows", []) or [])[:TOP_K_SIMILAR_ROW_VALUES]:
                if isinstance(item, dict):
                    tv = canonical_empty_text(item.get("target_column_value", ""))
                    if tv not in {"", "empty"}:
                        fixed = normalize_by_movie_column(column, tv)
                        similar_row_target_values.append(fixed if fixed else tv)

        col_ctx = obj.get("column_context", {})
        column_top_values = []
        column_detected_type = obj.get("detected_type", None)
        column_numeric_stats = None
        current_value_pattern = None
        dominant_pattern = None
        parsed_year = obj.get("parsed_year", None)
        parsed_release_year = obj.get("parsed_release_year", None)
        parsed_release_country = obj.get("parsed_release_country", None)
        duration_minutes_value = obj.get("duration_minutes", None)
        rating_value_float = obj.get("rating_value_float", None)
        count_value_int = obj.get("count_value_int", None)
        review_total_value = obj.get("review_total", None)
        list_token_count = obj.get("list_token_count", None)
        list_token_count_bucket = obj.get("list_token_count_bucket", None)
        text_length_bucket = obj.get("text_length_bucket", None)

        if isinstance(col_ctx, dict):
            for item in col_ctx.get("top_values_with_counts", []) or []:
                if isinstance(item, dict) and item.get("value", None) is not None:
                    val = canonical_empty_text(item["value"])
                    if val not in {"", "empty"}:
                        fixed = normalize_by_movie_column(column, val)
                        column_top_values.append({
                            "value": fixed if fixed else val,
                            "raw_value": val,
                            "count": safe_int(item.get("count", 0), 0),
                            "ratio": safe_float(item.get("ratio", 0.0), 0.0),
                        })
            column_detected_type = col_ctx.get("detected_type", column_detected_type)
            column_numeric_stats = col_ctx.get("numeric_stats", None)
            current_value_pattern = col_ctx.get("current_value_pattern", None)
            dominant_pattern = col_ctx.get("dominant_pattern", None)
            parsed_year = col_ctx.get("parsed_year", parsed_year)
            duration_minutes_value = col_ctx.get("duration_minutes", duration_minutes_value)
            rating_value_float = col_ctx.get("rating_value_float", rating_value_float)
            count_value_int = col_ctx.get("count_value_int", count_value_int)
            list_token_count = col_ctx.get("list_token_count", list_token_count)
            list_token_count_bucket = col_ctx.get("list_token_count_bucket", list_token_count_bucket)
            text_length_bucket = col_ctx.get("text_length_bucket", text_length_bucket)

        if movie_bundle:
            parsed_year = movie_bundle.get("parsed_year", parsed_year)
            parsed_release_date = movie_bundle.get("parsed_release_date", None)
            if isinstance(parsed_release_date, dict):
                parsed_release_year = parsed_release_date.get("year", parsed_release_year)
                parsed_release_country = parsed_release_date.get("country", parsed_release_country)
            duration_minutes_value = movie_bundle.get("duration_minutes", duration_minutes_value)
            rating_value_float = movie_bundle.get("rating_value_float", rating_value_float)
            count_value_int = movie_bundle.get("rating_count_int", count_value_int)
            review_total_value = movie_bundle.get("review_total", review_total_value)

        if not column_top_values:
            counter = column_value_counter[column]
            total = max(sum(counter.values()), 1)
            for val, cnt in counter.most_common(10):
                column_top_values.append({"value": val, "raw_value": val, "count": cnt, "ratio": cnt / total})

        bayesian_context = obj.get("bayesian_context", {})
        prior_error_probability = safe_float(obj.get("prior_error_probability", 0.0), 0.0)
        posterior_error_probability = safe_float(obj.get("posterior_error_probability", 0.0), 0.0)
        bayes_rule_signal = 0.0
        bayes_rarity_signal = 0.0
        bayes_pattern_signal = 0.0
        bayes_neighbor_signal = 0.0
        bayes_movie_typed_signal = 0.0

        if isinstance(bayesian_context, dict):
            prior_error_probability = safe_float(bayesian_context.get("prior_error_probability", prior_error_probability), 0.0)
            posterior_error_probability = safe_float(bayesian_context.get("posterior_error_probability", posterior_error_probability), 0.0)
            be = bayesian_context.get("bayes_evidence", {})
            if isinstance(be, dict):
                bayes_rule_signal = safe_float(be.get("rule_signal", 0.0), 0.0)
                bayes_rarity_signal = safe_float(be.get("rarity_signal", 0.0), 0.0)
                bayes_pattern_signal = safe_float(be.get("pattern_signal", 0.0), 0.0)
                bayes_neighbor_signal = safe_float(be.get("neighbor_signal", 0.0), 0.0)
                bayes_movie_typed_signal = safe_float(be.get("movie_typed_signal", 0.0), 0.0)

        context_map[(row_id, column)] = {
            "row_id": row_id,
            "column": column,
            "value": value,
            "semantic_type": obj.get("semantic_type", None),
            "detected_type": obj.get("detected_type", column_detected_type),
            "violation_count": obj.get("violation_count", None),
            "conflict_score": obj.get("conflict_score", None),
            "main_rule_type": obj.get("main_rule_type", conflict_context.get("main_rule_type") if isinstance(conflict_context, dict) else None),
            "main_usage_role": obj.get("main_usage_role", conflict_context.get("main_usage_role") if isinstance(conflict_context, dict) else None),
            "row_values": row_values,
            "movie_metadata_bundle": movie_bundle,
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
            "bayes_movie_typed_signal": bayes_movie_typed_signal,
            "parsed_year": parsed_year,
            "parsed_release_year": parsed_release_year,
            "parsed_release_country": parsed_release_country,
            "duration_minutes": duration_minutes_value,
            "rating_value_float": rating_value_float,
            "count_value_int": count_value_int,
            "review_total": review_total_value,
            "list_token_count": list_token_count,
            "list_token_count_bucket": list_token_count_bucket,
            "text_length_bucket": text_length_bucket,
            "current_value_pattern": current_value_pattern,
            "dominant_pattern": dominant_pattern,
            **counters,
        }

    return context_map


def merge_flat_into_context_map(context_map: Dict[Tuple[int, str], dict], flat_records: List[dict]):
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
            context_map[key] = {"row_id": row_id, "column": column, "value": canonical_empty_text(obj.get("value", "")), "row_values": {}, "movie_metadata_bundle": {}}
        ctx = context_map[key]
        for field in [
            "semantic_type", "detected_type", "violation_count", "conflict_score",
            "main_rule_type", "main_usage_role", "parsed_year", "parsed_release_year",
            "parsed_release_country", "duration_minutes", "rating_value_float",
            "count_value_int", "review_total", "list_token_count", "list_token_count_bucket",
            "text_length_bucket", "current_value_pattern", "dominant_pattern",
            "strong_rule_count", "candidate_generation_rule_count", "fd_like_count",
            "context_rule_count", "global_rule_count", "rare_value_count",
            "typo_rule_count", "pattern_rule_count", "schema_rule_count",
            "movie_typed_rule_count", "movie_consistency_rule_count",
            "prior_error_probability", "posterior_error_probability",
        ]:
            if field in obj and ctx.get(field, None) in [None, "", [], "empty"]:
                ctx[field] = obj.get(field)
        nmv = canonical_empty_text(obj.get("neighbor_majority_value", ""))
        if not ctx.get("neighbor_majority_value", None) and nmv not in {"", "empty"}:
            ctx["neighbor_majority_value"] = nmv
        if safe_float(ctx.get("neighbor_majority_ratio", 0.0), 0.0) == 0.0 and "neighbor_majority_ratio" in obj:
            ctx["neighbor_majority_ratio"] = safe_float(obj.get("neighbor_majority_ratio", 0.0), 0.0)


# ============================================================
# 6. 规则投票 / Bayes 共现 / 变换特征
# ============================================================

def build_rule_vote_counter(ctx: dict, column: str) -> Counter:
    counter = Counter()
    for ev in ctx.get("rule_expected_values", []):
        ev = canonical_empty_text(ev)
        if ev not in {"", "empty"}:
            fixed = normalize_by_movie_column(column, ev)
            counter[fixed if fixed else ev] += 1
    for ev in ctx.get("expected_values_from_rules", []):
        ev = canonical_empty_text(ev)
        if ev not in {"", "empty"}:
            fixed = normalize_by_movie_column(column, ev)
            counter[fixed if fixed else ev] += 2
    return counter


def compute_rule_features(candidate_value: Any, dirty_value: Any, ctx: dict) -> Dict[str, Any]:
    column = str(ctx.get("column", ""))
    vote_counter = build_rule_vote_counter(ctx, column)
    total_votes = sum(vote_counter.values())
    cand_fixed = normalize_by_movie_column(column, candidate_value)
    dirty_fixed = normalize_by_movie_column(column, dirty_value)
    cand_key = canonical_empty_text(cand_fixed if cand_fixed else candidate_value)
    dirty_key = canonical_empty_text(dirty_fixed if dirty_fixed else dirty_value)

    cand_vote = vote_counter.get(cand_key, 0)
    dirty_vote = vote_counter.get(dirty_key, 0)

    alt_values = set()
    for x in ctx.get("alternative_values_from_column", []):
        if isinstance(x, dict):
            v = canonical_empty_text(x.get("value", ""))
            fixed = normalize_by_movie_column(column, v)
            alt_values.add(canonical_empty_text(fixed if fixed else v))

    sim_row_values = []
    for v in ctx.get("similar_row_target_values", []):
        fixed = normalize_by_movie_column(column, v)
        sim_row_values.append(canonical_empty_text(fixed if fixed else v))

    neighbor = ctx.get("neighbor_majority_value", None)
    neighbor_fixed = normalize_by_movie_column(column, neighbor)
    neighbor_key = canonical_empty_text(neighbor_fixed if neighbor_fixed else neighbor)

    suggested = ctx.get("suggested_correct_value", None)
    suggested_fixed = normalize_by_movie_column(column, suggested)
    suggested_key = canonical_empty_text(suggested_fixed if suggested_fixed else suggested)

    return {
        "rule_total_votes": total_votes,
        "candidate_rule_vote_count": cand_vote,
        "dirty_rule_vote_count": dirty_vote,
        "candidate_rule_support_ratio": cand_vote / total_votes if total_votes > 0 else 0.0,
        "dirty_rule_support_ratio": dirty_vote / total_votes if total_votes > 0 else 0.0,
        "rule_support_gain": (cand_vote - dirty_vote) / total_votes if total_votes > 0 else 0.0,
        "candidate_matches_any_rule_expected": int(cand_vote > 0),
        "candidate_matches_any_strong_rule_expected": int(cand_key in {canonical_empty_text(normalize_by_movie_column(column, v) or v) for v in ctx.get("rule_expected_values", [])}),
        "candidate_matches_any_expected_values_from_rules": int(cand_key in {canonical_empty_text(normalize_by_movie_column(column, v) or v) for v in ctx.get("expected_values_from_rules", [])}),
        "candidate_in_alternative_values_from_column": int(cand_key in alt_values),
        "candidate_equals_neighbor_majority": int(neighbor_key not in {"", "empty"} and cand_key == neighbor_key),
        "candidate_equals_suggested_correct_value": int(suggested_key not in {"", "empty"} and cand_key == suggested_key),
        "candidate_matches_similar_row_target": int(cand_key in sim_row_values),
        "candidate_match_count_in_similar_rows": sum(1 for v in sim_row_values if cand_key == v),
        "candidate_match_ratio_in_similar_rows": sum(1 for v in sim_row_values if cand_key == v) / len(sim_row_values) if sim_row_values else 0.0,
        "strong_rule_count": safe_int(ctx.get("strong_rule_count", 0), 0),
        "candidate_generation_rule_count": safe_int(ctx.get("candidate_generation_rule_count", 0), 0),
        "fd_like_count": safe_int(ctx.get("fd_like_count", 0), 0),
        "context_rule_count": safe_int(ctx.get("context_rule_count", 0), 0),
        "global_rule_count": safe_int(ctx.get("global_rule_count", 0), 0),
        "rare_value_count": safe_int(ctx.get("rare_value_count", 0), 0),
        "typo_rule_count": safe_int(ctx.get("typo_rule_count", 0), 0),
        "pattern_rule_count": safe_int(ctx.get("pattern_rule_count", 0), 0),
        "schema_rule_count": safe_int(ctx.get("schema_rule_count", 0), 0),
        "movie_typed_rule_count": safe_int(ctx.get("movie_typed_rule_count", 0), 0),
        "movie_consistency_rule_count": safe_int(ctx.get("movie_consistency_rule_count", 0), 0),
    }


def normalize_raw_value_for_cooccur(column: str, value: Any) -> str:
    fixed = normalize_by_movie_column(column, value)
    return canonical_empty_text(fixed if fixed else value)


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
        norm_df[col] = norm_df[col].map(lambda x, c=col: normalize_raw_value_for_cooccur(c, x))

    for target_col in usable_columns:
        cooccur[target_col] = {}
        target_domain = sorted(set(v for v in norm_df[target_col].tolist() if v not in {"", "empty"}))
        domain_size = max(len(target_domain), 1)

        for ctx_col in usable_columns:
            if ctx_col == target_col:
                continue
            pair_counter = Counter()
            ctx_counter = Counter()
            for _, r in norm_df[[target_col, ctx_col]].iterrows():
                v, c = r[target_col], r[ctx_col]
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


def load_or_build_cooccurrence_dict() -> Optional[Dict[str, Any]]:
    if COOCCURRENCE_DICT_FILE is not None and os.path.exists(COOCCURRENCE_DICT_FILE):
        print(f"[INFO] 使用现成共现字典: {COOCCURRENCE_DICT_FILE}")
        with open(COOCCURRENCE_DICT_FILE, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    if not path_exists_nonempty(RAW_DATA_FILE):
        print("[WARN] 没有现成共现字典，也没有可用原始数据文件，贝叶斯共现特征将退化为 0。")
        return None
    print(f"[INFO] 从原始数据集构建经验共现字典: {RAW_DATA_FILE}")
    raw_df = pd.read_csv(RAW_DATA_FILE, encoding="utf-8-sig")
    cooccur = build_cooccurrence_from_raw_data(raw_df)
    output_path = resolve_output_path(OUTPUT_BUILT_COOCCUR_JSON, CANDIDATES_FILE)
    with open(output_path, "w", encoding="utf-8-sig") as f:
        json.dump(cooccur, f, ensure_ascii=False)
    print(f"[INFO] 已保存构建出的共现字典: {output_path}")
    return cooccur


def extract_context_pairs_from_ctx(ctx: dict) -> List[Tuple[str, str]]:
    row_values = ctx.get("row_values", {})
    target_column = ctx.get("column")
    pairs = []
    if isinstance(row_values, dict) and row_values:
        for k, v in row_values.items():
            if str(k) == str(target_column):
                continue
            vv = normalize_raw_value_for_cooccur(str(k), v)
            if vv in {"", "empty"}:
                continue
            pairs.append((str(k), vv))
    priority = {
        "Id": 0, "Name": 1, "Year": 2, "Release Date": 3, "Director": 4,
        "Creator": 5, "Actors": 6, "Genre": 7, "Country": 8, "RatingValue": 9, "RatingCount": 10,
    }
    return sorted(pairs, key=lambda x: priority.get(x[0], 100))[:MAX_CONTEXT_PAIRS]


def compute_bayes_features(candidate_value: Any, ctx: dict, external_cooccur: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    context_pairs = extract_context_pairs_from_ctx(ctx)
    target_column = str(ctx.get("column", ""))
    candidate_key = normalize_raw_value_for_cooccur(target_column, candidate_value)

    base = {
        "prior_error_probability": safe_float(ctx.get("prior_error_probability", 0.0), 0.0),
        "posterior_error_probability": safe_float(ctx.get("posterior_error_probability", 0.0), 0.0),
        "bayes_rule_signal": safe_float(ctx.get("bayes_rule_signal", 0.0), 0.0),
        "bayes_rarity_signal": safe_float(ctx.get("bayes_rarity_signal", 0.0), 0.0),
        "bayes_pattern_signal": safe_float(ctx.get("bayes_pattern_signal", 0.0), 0.0),
        "bayes_neighbor_signal": safe_float(ctx.get("bayes_neighbor_signal", 0.0), 0.0),
        "bayes_movie_typed_signal": safe_float(ctx.get("bayes_movie_typed_signal", 0.0), 0.0),
    }

    if not context_pairs or external_cooccur is None:
        return {
            "context_pair_count_used": 0,
            "bayes_max_cond_prob": 0.0,
            "bayes_min_cond_prob": 0.0,
            "bayes_mean_cond_prob": 0.0,
            "bayes_prod_logprob": 0.0,
            "bayes_nonzero_pair_count": 0,
            **base,
        }

    cand_map = external_cooccur.get(target_column, {}).get(candidate_key, {})
    probs = [safe_float(cand_map.get(f"{ctx_col}={ctx_val}", 0.0), 0.0) for ctx_col, ctx_val in context_pairs]
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


def compute_transform_features(candidate_value: Any, dirty_value: Any, column: Any = None) -> Dict[str, Any]:
    sim = string_similarity(candidate_value, dirty_value)
    phon_sim = phonetic_similarity(candidate_value, dirty_value)
    key_dist = avg_keyboard_distance_for_same_length(candidate_value, dirty_value)
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

    fixed_candidate = normalize_by_movie_column(column, candidate_value) if column is not None else None
    fixed_dirty = normalize_by_movie_column(column, dirty_value) if column is not None else None

    return {
        "edit_similarity_to_dirty": sim,
        "edit_distance_proxy": 1.0 - sim,
        "phonetic_similarity_to_dirty": phon_sim,
        "phonetic_distance_proxy": 1.0 - phon_sim,
        "avg_keyboard_distance": key_dist,
        "length_diff": abs(len(str(candidate_value)) - len(str(dirty_value))),
        "both_numeric": both_numeric,
        "numeric_abs_diff": numeric_abs_diff,
        "numeric_rel_diff": numeric_rel_diff,
        "candidate_normalized_value": fixed_candidate,
        "dirty_normalized_value": fixed_dirty,
        "candidate_normalized_equals_dirty": int(fixed_candidate is not None and fixed_dirty is not None and norm_text(fixed_candidate) == norm_text(fixed_dirty)),
        "candidate_is_canonical_restoration": int(fixed_candidate is not None and norm_text(fixed_candidate) == norm_text(candidate_value) and fixed_dirty is None),
    }


# ============================================================
# 7. Movies plausibility / consistency / source 特征
# ============================================================

def get_row_value(ctx: dict, col: str) -> str:
    rv = ctx.get("row_values", {}) or {}
    if col in rv:
        return canonical_empty_text(rv[col])
    mb = ctx.get("movie_metadata_bundle", {}) or {}
    if col in mb:
        return canonical_empty_text(mb[col])
    return ""


def compute_movie_specific_features(candidate_value: Any, dirty_value: Any, ctx: dict) -> Dict[str, Any]:
    column = str(ctx.get("column", ""))
    cl = column.lower()
    semantic_type = ctx.get("semantic_type", None)

    cand_year = normalize_year_candidate(candidate_value)
    dirty_year = normalize_year_candidate(dirty_value)
    cand_duration = normalize_duration_candidate(candidate_value)
    dirty_duration = normalize_duration_candidate(dirty_value)
    cand_duration_min = duration_minutes(candidate_value)
    dirty_duration_min = duration_minutes(dirty_value)
    cand_release_parts = parse_release_date_parts(candidate_value)
    dirty_release_parts = parse_release_date_parts(dirty_value)
    cand_rating = normalize_rating_value_candidate(candidate_value)
    dirty_rating = normalize_rating_value_candidate(dirty_value)
    cand_count = normalize_count_candidate(candidate_value)
    dirty_count = normalize_count_candidate(dirty_value)
    cand_review = normalize_review_count_candidate(candidate_value)
    dirty_review = normalize_review_count_candidate(dirty_value)
    cand_country = normalize_country_candidate(candidate_value)
    dirty_country = normalize_country_candidate(dirty_value)
    cand_lang = normalize_language_candidate(candidate_value)
    dirty_lang = normalize_language_candidate(dirty_value)

    row_year = normalize_year_candidate(get_row_value(ctx, "Year"))
    row_release_date = get_row_value(ctx, "Release Date")
    row_release_parts = parse_release_date_parts(row_release_date)
    row_release_year = str(row_release_parts["year"]) if row_release_parts and row_release_parts.get("year") else None
    row_release_country = row_release_parts.get("country") if row_release_parts else None
    row_country = normalize_country_candidate(get_row_value(ctx, "Country"))
    row_duration_min = duration_minutes(get_row_value(ctx, "Duration"))
    row_rating_value = normalize_rating_value_candidate(get_row_value(ctx, "RatingValue"))
    row_rating_count = normalize_count_candidate(get_row_value(ctx, "RatingCount"))
    row_review_total = review_total(get_row_value(ctx, "ReviewCount"))
    row_actors = get_row_value(ctx, "Actors")
    row_cast = get_row_value(ctx, "Cast")

    out = {
        "movie_column_is_time": is_movie_time_col(column, semantic_type),
        "movie_column_is_rating": is_movie_rating_col(column, semantic_type),
        "movie_column_is_people": is_movie_people_col(column),
        "movie_column_is_list": is_movie_list_col(column, semantic_type),
        "movie_column_is_text": is_movie_text_col(column, semantic_type),

        "movie_candidate_is_year": int(cand_year is not None),
        "movie_dirty_is_year": int(dirty_year is not None),
        "movie_candidate_year_restoration": int(cand_year is not None and dirty_year is None),

        "movie_candidate_is_duration": int(cand_duration is not None),
        "movie_dirty_is_duration": int(dirty_duration is not None),
        "movie_candidate_duration_restoration": int(cand_duration is not None and dirty_duration is None),
        "movie_candidate_duration_minutes": cand_duration_min,
        "movie_dirty_duration_minutes": dirty_duration_min,
        "movie_duration_abs_diff": abs(cand_duration_min - dirty_duration_min) if cand_duration_min is not None and dirty_duration_min is not None else None,

        "movie_candidate_is_release_date": int(cand_release_parts is not None),
        "movie_dirty_is_release_date": int(dirty_release_parts is not None),
        "movie_candidate_release_date_restoration": int(cand_release_parts is not None and dirty_release_parts is None),

        "movie_candidate_is_country": int(cand_country is not None),
        "movie_dirty_is_country": int(dirty_country is not None),
        "movie_candidate_country_restoration": int(cand_country is not None and dirty_country is None),

        "movie_candidate_is_language": int(cand_lang is not None),
        "movie_dirty_is_language": int(dirty_lang is not None),

        "movie_candidate_is_rating_value": int(cand_rating is not None),
        "movie_dirty_is_rating_value": int(dirty_rating is not None),
        "movie_candidate_rating_value_restoration": int(cand_rating is not None and dirty_rating is None),
        "movie_candidate_rating_value_float": safe_float(cand_rating, None) if cand_rating is not None else None,
        "movie_dirty_rating_value_float": safe_float(dirty_rating, None) if dirty_rating is not None else None,

        "movie_candidate_is_count": int(cand_count is not None),
        "movie_dirty_is_count": int(dirty_count is not None),
        "movie_candidate_count_restoration": int(cand_count is not None and dirty_count is None),
        "movie_candidate_count_int": safe_int(cand_count, None) if cand_count is not None else None,
        "movie_dirty_count_int": safe_int(dirty_count, None) if dirty_count is not None else None,

        "movie_candidate_is_review_count": int(cand_review is not None),
        "movie_dirty_is_review_count": int(dirty_review is not None),
        "movie_candidate_review_count_restoration": int(cand_review is not None and dirty_review is None),
        "movie_candidate_review_total": review_total(candidate_value),
        "movie_dirty_review_total": review_total(dirty_value),

        "movie_candidate_list_token_count": len(split_list_tokens(candidate_value)),
        "movie_dirty_list_token_count": len(split_list_tokens(dirty_value)),
        "movie_list_jaccard_to_dirty": list_jaccard(candidate_value, dirty_value),
        "movie_candidate_contains_dirty_list": int(list_overlap_ratio(dirty_value, candidate_value) >= 0.99 and len(split_list_tokens(dirty_value)) > 0),
        "movie_dirty_contains_candidate_list": int(list_overlap_ratio(candidate_value, dirty_value) >= 0.99 and len(split_list_tokens(candidate_value)) > 0),

        "movie_candidate_has_mojibake_or_control": has_mojibake_or_control_chars(candidate_value),
        "movie_dirty_has_mojibake_or_control": has_mojibake_or_control_chars(dirty_value),
        "movie_mojibake_repair_gain": int(has_mojibake_or_control_chars(dirty_value) == 1 and has_mojibake_or_control_chars(candidate_value) == 0),

        "movie_row_year": row_year,
        "movie_row_release_year": row_release_year,
        "movie_row_release_country": row_release_country,
        "movie_row_country": row_country,
        "movie_row_duration_minutes": row_duration_min,
        "movie_row_rating_value_float": safe_float(row_rating_value, None) if row_rating_value is not None else None,
        "movie_row_rating_count_int": safe_int(row_rating_count, None) if row_rating_count is not None else None,
        "movie_row_review_total": row_review_total,
        "movie_row_actors_cast_overlap": list_overlap_ratio(row_actors, row_cast),
    }

    if cl == "year" and cand_year is not None and row_release_year is not None:
        out["movie_candidate_year_equals_release_year"] = int(str(cand_year) == str(row_release_year))
        out["movie_candidate_year_release_gap"] = int(row_release_year) - int(cand_year)
    else:
        out["movie_candidate_year_equals_release_year"] = 0
        out["movie_candidate_year_release_gap"] = None

    if cl == "release date" and cand_release_parts:
        out["movie_candidate_release_year"] = cand_release_parts.get("year")
        out["movie_candidate_release_country"] = cand_release_parts.get("country")
        out["movie_candidate_release_year_equals_row_year"] = int(row_year is not None and cand_release_parts.get("year") is not None and str(cand_release_parts.get("year")) == str(row_year))
    else:
        out["movie_candidate_release_year"] = None
        out["movie_candidate_release_country"] = None
        out["movie_candidate_release_year_equals_row_year"] = 0

    if cl == "country" and row_release_country:
        out["movie_candidate_country_equals_release_country"] = int(cand_country is not None and norm_text(cand_country) == norm_text(row_release_country))
    else:
        out["movie_candidate_country_equals_release_country"] = 0

    if cl == "duration":
        out["movie_candidate_duration_in_broad_range"] = int(cand_duration_min is not None and 1 <= cand_duration_min <= 500)
        out["movie_candidate_duration_in_common_movie_range"] = int(cand_duration_min is not None and 40 <= cand_duration_min <= 240)
        out["movie_candidate_duration_diff_to_row_duration"] = abs(cand_duration_min - row_duration_min) if cand_duration_min is not None and row_duration_min is not None else None
    else:
        out["movie_candidate_duration_in_broad_range"] = 0
        out["movie_candidate_duration_in_common_movie_range"] = 0
        out["movie_candidate_duration_diff_to_row_duration"] = None

    if cl == "ratingvalue":
        rv = safe_float(cand_rating, None) if cand_rating is not None else None
        out["movie_candidate_rating_in_range"] = int(rv is not None and 0.0 <= rv <= 10.0)
        out["movie_candidate_rating_diff_to_row_rating"] = abs(rv - safe_float(row_rating_value, 0.0)) if rv is not None and row_rating_value is not None else None
    else:
        out["movie_candidate_rating_in_range"] = 0
        out["movie_candidate_rating_diff_to_row_rating"] = None

    if cl == "ratingcount":
        ci = safe_int(cand_count, None) if cand_count is not None else None
        out["movie_candidate_count_nonnegative"] = int(ci is not None and ci >= 0)
        out["movie_candidate_count_diff_to_row_count"] = abs(ci - safe_int(row_rating_count, 0)) if ci is not None and row_rating_count is not None else None
    else:
        out["movie_candidate_count_nonnegative"] = 0
        out["movie_candidate_count_diff_to_row_count"] = None

    if cl == "actors":
        out["movie_candidate_actors_overlap_with_cast"] = list_overlap_ratio(candidate_value, row_cast)
        out["movie_candidate_actors_is_cast_prefix"] = int(norm_text(",".join(split_list_tokens(row_cast)[:len(split_list_tokens(candidate_value))])) == norm_text(normalize_list_value(candidate_value) or "") and len(split_list_tokens(candidate_value)) > 0)
    else:
        out["movie_candidate_actors_overlap_with_cast"] = 0.0
        out["movie_candidate_actors_is_cast_prefix"] = 0

    if cl == "cast":
        out["movie_candidate_cast_contains_actors"] = int(list_overlap_ratio(row_actors, candidate_value) >= 0.99 and len(split_list_tokens(row_actors)) > 0)
    else:
        out["movie_candidate_cast_contains_actors"] = 0

    return out


def compute_column_plausibility_features(candidate_value: Any, dirty_value: Any, ctx: dict) -> Dict[str, Any]:
    column = ctx.get("column", "")
    top_counter = {
        canonical_empty_text(x["value"]): x
        for x in ctx.get("column_top_values", [])
        if isinstance(x, dict)
    }

    fixed_candidate = normalize_by_movie_column(column, candidate_value)
    fixed_dirty = normalize_by_movie_column(column, dirty_value)
    candidate_norm = canonical_empty_text(fixed_candidate if fixed_candidate else candidate_value)
    dirty_norm = canonical_empty_text(fixed_dirty if fixed_dirty else dirty_value)

    candidate_in_top_values = int(candidate_norm in top_counter)
    candidate_top_ratio = safe_float(top_counter.get(candidate_norm, {}).get("ratio", 0.0), 0.0)
    candidate_top_count = safe_int(top_counter.get(candidate_norm, {}).get("count", 0), 0)

    detected_type = str(ctx.get("column_detected_type", "") or ctx.get("detected_type", "") or "").lower()
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
        "candidate_column_top_count": candidate_top_count,
        "candidate_in_numeric_range": candidate_in_numeric_range,
        "candidate_outside_numeric_range": candidate_outside_numeric_range,
        "candidate_normalized_equals_dirty_for_column": int(candidate_norm == dirty_norm and candidate_norm not in {"", "empty"}),
    }
    out.update(compute_movie_specific_features(candidate_value, dirty_value, ctx))
    return out


def parse_extra_info_as_dict(extra_info: Any) -> Dict[str, str]:
    s = str(extra_info or "")
    out = {}
    for part in re.split(r"[|;]", s):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def load_movie_consistency_suspicious_map(path: str) -> Dict[Tuple[int, str], dict]:
    if not path_exists_nonempty(path):
        return {}
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return {}
    if not {"row_id", "column"}.issubset(set(df.columns)):
        return {}
    out = {}
    for _, r in df.iterrows():
        try:
            row_id = int(r["row_id"])
            column = str(r["column"]).strip()
        except Exception:
            continue
        out[(row_id, column)] = r.to_dict()
    return out


def load_movie_direct_metadata_suspicious_map(path: str) -> Dict[Tuple[int, str], dict]:
    """
    读取 movie_direct_metadata_suspicious_cells.csv。
    该文件由增强版候选生成脚本输出，常见字段包括：
    row_id,column,value,direct_candidate_value,direct_reason,direct_confidence,
    metadata_source,direct_source,direct_rule_name 等。
    """
    if not path_exists_nonempty(path):
        return {}
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception as e:
        print(f"[WARN] failed to read movie direct metadata suspicious file: {path}, err={e}")
        return {}
    if not {"row_id", "column"}.issubset(set(df.columns)):
        print(f"[WARN] movie direct metadata suspicious file missing row_id/column: {path}")
        return {}
    out = {}
    for _, r in df.iterrows():
        try:
            row_id = int(r["row_id"])
            column = str(r["column"]).strip()
        except Exception:
            continue
        out[(row_id, column)] = r.to_dict()
    return out


def build_fallback_contexts_from_dirty_table(raw_data_file: str) -> Dict[Tuple[int, str], dict]:
    """
    为没有 LLM rich context 的候选构造 fallback context。
    这是 Movies 数据集很关键的一步：Id / Description / Genre / Name 等 direct/metadata
    候选可能来自全表规则，不一定在 candidate_llm_contexts_movies.jsonl 中。
    如果仍然 ctx is None 就跳过，会导致新增候选在特征构建阶段消失。
    """
    if not path_exists_nonempty(raw_data_file):
        return {}
    try:
        raw_df = pd.read_csv(raw_data_file, encoding="utf-8-sig")
    except Exception as e:
        print(f"[WARN] failed to build fallback context from dirty table: {raw_data_file}, err={e}")
        return {}

    out = {}
    for rid, r in raw_df.iterrows():
        row_values = {str(c): canonical_empty_text(r[c]) for c in raw_df.columns}
        for col in raw_df.columns:
            col = str(col)
            value = canonical_empty_text(r[col])
            release_parts = parse_release_date_parts(row_values.get("Release Date", ""))
            out[(int(rid), col)] = {
                "row_id": int(rid),
                "column": col,
                "value": value,
                "semantic_type": None,
                "detected_type": None,
                "violation_count": 0,
                "conflict_score": 0.0,
                "main_rule_type": "dirty_table_fallback",
                "main_usage_role": "fallback_context",
                "row_values": row_values,
                "movie_metadata_bundle": {},
                "suggested_correct_value": None,
                "neighbor_majority_value": None,
                "neighbor_majority_ratio": 0.0,
                "similar_row_target_values": [],
                "column_top_values": [],
                "alternative_values_from_column": [],
                "rule_expected_values": [],
                "expected_values_from_rules": [],
                "column_numeric_stats": None,
                "column_detected_type": None,
                "prior_error_probability": 0.0,
                "posterior_error_probability": 0.0,
                "bayes_rule_signal": 0.0,
                "bayes_rarity_signal": 0.0,
                "bayes_pattern_signal": 0.0,
                "bayes_neighbor_signal": 0.0,
                "bayes_movie_typed_signal": 0.0,
                "parsed_year": normalize_year_candidate(value) if col == "Year" else None,
                "parsed_release_year": release_parts.get("year") if release_parts else None,
                "parsed_release_country": release_parts.get("country") if release_parts else None,
                "duration_minutes": duration_minutes(value) if col == "Duration" else None,
                "rating_value_float": safe_float(normalize_rating_value_candidate(value), None) if col == "RatingValue" else None,
                "count_value_int": safe_int(normalize_count_candidate(value), None) if col == "RatingCount" else None,
                "review_total": review_total(value) if col == "ReviewCount" else None,
                "list_token_count": len(split_list_tokens(value)) if col in {"Actors", "Cast", "Creator", "Director", "Genre", "Filming Locations"} else None,
                "list_token_count_bucket": None,
                "text_length_bucket": None,
                "current_value_pattern": None,
                "dominant_pattern": None,
                "strong_rule_count": 0,
                "candidate_generation_rule_count": 0,
                "fd_like_count": 0,
                "context_rule_count": 0,
                "global_rule_count": 0,
                "rare_value_count": 0,
                "typo_rule_count": 0,
                "pattern_rule_count": 0,
                "schema_rule_count": 0,
                "movie_typed_rule_count": 0,
                "movie_consistency_rule_count": 0,
            }
    return out


MOVIE_DIRECT_METADATA_SUSPICIOUS_MAP = {}


def source_tokens_from_row(row: pd.Series) -> set:
    source_text = str(row.get("candidate_source", row.get("candidate_source_text", "")) or "")
    return {x.strip() for x in source_text.split("|") if x.strip()}


MOVIE_DIRECT_SOURCES = {
    "direct_year_triple",
    "direct_release_date_wide_limited",
    "direct_genre_conservative_mapping",
    "direct_duration_single_format",
    "direct_rating_value_single_format",
    "movies_direct_year_triple_to_middle",
    "movies_direct_release_date_wide_limited",
    "movies_direct_genre_conservative_mapping",
}

MOVIE_METADATA_STRONG_SOURCE_PREFIXES = (
    "id_to_",
    "name_year_to_",
    "name_to_",
)

MOVIE_METADATA_STRONG_SOURCES = {
    "movie_metadata_consistency",
    "expected_values_from_rules",
    "rule_expected_value",
    "candidate_equals_allowed_consensus_value",
    "release_date_year_consistency",
    "release_date_country_consistency",
    "cast_prefix_to_actors",
    "actors_to_cast",
    "name_to_id",
    "name_year_to_id",
    "name_to_description",
    "name_year_to_description",
    "name_year_to_director",
    "name_year_to_creator",
    "name_to_director",
    "name_to_creator",
}

MOVIE_WEAK_FREQUENCY_SOURCES = {
    "column_top_value",
    "global_top_value",
    "similar_column_value",
    "alternative_values_from_column",
}


def compute_movie_direct_metadata_features(row: pd.Series, candidate_value: Any, dirty_value: Any, ctx: dict) -> Dict[str, Any]:
    """
    给新增 direct / metadata 候选补强特征。
    这些特征用于后续训练/推理 gate 识别：
    - 候选是否来自高置信 direct rule
    - 候选是否等于 allowed direct candidate value
    - 候选是否只来自弱频率 source
    """
    row_id = int(row.get("row_id"))
    column = str(row.get("column")).strip()
    cl = column.lower()
    sources = source_tokens_from_row(row)
    source_text = str(row.get("candidate_source", row.get("candidate_source_text", "")) or "")

    direct_meta = MOVIE_DIRECT_METADATA_SUSPICIOUS_MAP.get((row_id, column), {})
    allowed_direct_value = canonical_empty_text(
        direct_meta.get("direct_candidate_value",
        direct_meta.get("candidate_value",
        direct_meta.get("consensus_value", "")))
    )
    allowed_direct_reason = str(
        direct_meta.get("direct_reason",
        direct_meta.get("direct_rule_name",
        direct_meta.get("metadata_source", "")))
        or ""
    )
    allowed_direct_confidence = safe_float(
        direct_meta.get("direct_confidence",
        direct_meta.get("confidence",
        direct_meta.get("consensus_confidence", 0.0))),
        0.0
    )
    allowed_metadata_source = str(
        direct_meta.get("metadata_source",
        direct_meta.get("direct_source",
        direct_meta.get("source", "")))
        or ""
    )

    candidate_equals_allowed_direct = int(
        allowed_direct_value not in {"", "empty"}
        and value_equal_for_column(column, candidate_value, allowed_direct_value)
    )

    has_direct_source = int(
        any(s in MOVIE_DIRECT_SOURCES for s in sources)
        or any(s.startswith("direct_") for s in sources)
        or "direct_" in source_text
    )

    has_metadata_source = int(
        any(s in MOVIE_METADATA_STRONG_SOURCES for s in sources)
        or any(s.startswith(MOVIE_METADATA_STRONG_SOURCE_PREFIXES) for s in sources)
        or "movie_metadata_consistency" in source_text
    )

    has_id_metadata_source = int(any(s.startswith("id_to_") or s in {"name_to_id", "name_year_to_id"} for s in sources))
    has_name_year_metadata_source = int(any(s.startswith("name_year_to_") for s in sources))
    has_name_metadata_source = int(any(s.startswith("name_to_") for s in sources))
    has_description_source = int(any("description" in s for s in sources))
    has_people_metadata_source = int(any(s in {
        "name_to_director", "name_to_creator", "name_to_actors", "name_to_cast",
        "name_year_to_director", "name_year_to_creator", "director_creator_to_actors",
        "actors_to_cast", "cast_prefix_to_actors"
    } for s in sources))

    has_weak_frequency_source = int(any(s in MOVIE_WEAK_FREQUENCY_SOURCES for s in sources))
    weak_only = int(has_weak_frequency_source == 1 and has_direct_source == 0 and has_metadata_source == 0)

    # 列级强证据，后续 infer gate 可以直接使用。
    id_strong = int(cl == "id" and (has_id_metadata_source or has_name_year_metadata_source or candidate_equals_allowed_direct))
    desc_strong = int(cl == "description" and (has_description_source or has_id_metadata_source or has_name_year_metadata_source or candidate_equals_allowed_direct))
    rating_strong = int(cl == "ratingvalue" and (has_id_metadata_source or has_name_year_metadata_source or candidate_equals_allowed_direct or "id_to_rating_value" in sources or "name_year_to_rating_value" in sources))
    duration_strong = int(cl == "duration" and (has_id_metadata_source or has_name_year_metadata_source or candidate_equals_allowed_direct or "id_to_duration" in sources or "name_year_to_duration" in sources))
    people_strong = int(cl in {"creator", "director", "actors", "cast"} and (has_people_metadata_source or has_name_year_metadata_source or candidate_equals_allowed_direct))
    text_strong = int(cl in {"name", "description"} and (has_name_metadata_source or has_id_metadata_source or candidate_equals_allowed_direct))
    genre_direct = int(cl == "genre" and (has_direct_source or candidate_equals_allowed_direct or "direct_genre_conservative_mapping" in sources))

    return {
        "allowed_direct_candidate_value": allowed_direct_value if allowed_direct_value not in {"", "empty"} else None,
        "allowed_direct_reason": allowed_direct_reason,
        "allowed_direct_confidence": allowed_direct_confidence,
        "allowed_metadata_source": allowed_metadata_source,

        "candidate_equals_allowed_direct_value": candidate_equals_allowed_direct,
        "candidate_under_movie_direct_metadata_cell": int(bool(direct_meta)),
        "candidate_under_movie_direct_rule_cell": int(bool(direct_meta) and ("direct" in allowed_direct_reason.lower() or "direct" in allowed_metadata_source.lower())),
        "candidate_under_movie_metadata_candidate_cell": int(bool(direct_meta) and ("metadata" in allowed_direct_reason.lower() or "metadata" in allowed_metadata_source.lower())),

        "contains_movie_direct_source": has_direct_source,
        "contains_movie_direct_year_source": int(any("year" in s and ("direct" in s or "canonical" in s) for s in sources)),
        "contains_movie_direct_release_date_source": int(any(("release_date" in s or "release date" in s) and ("direct" in s or "canonical" in s) for s in sources)),
        "contains_movie_direct_genre_source": int(any("genre" in s and "direct" in s for s in sources)),
        "contains_movie_direct_duration_source": int(any("duration" in s and ("direct" in s or "canonical" in s) for s in sources)),
        "contains_movie_direct_rating_source": int(any("rating" in s and ("direct" in s or "canonical" in s) for s in sources)),

        "contains_strong_movie_metadata_source": has_metadata_source,
        "contains_name_year_metadata_source": has_name_year_metadata_source,
        "contains_name_metadata_source": has_name_metadata_source,
        "contains_id_metadata_source": has_id_metadata_source,
        "contains_description_source": has_description_source,
        "contains_people_metadata_source": has_people_metadata_source,
        "contains_genre_direct_mapping_source": genre_direct,

        "is_weak_frequency_only_candidate": weak_only,
        "is_column_top_only_candidate": int(sources == {"column_top_value"}),
        "is_neighbor_only_candidate": int(sources == {"neighbor_majority"} or sources == {"similar_row_target"}),
        "is_similarity_only_candidate": int(sources == {"similar_column_value"}),
        "has_strong_source_and_weak_source": int((has_direct_source or has_metadata_source or candidate_equals_allowed_direct) and has_weak_frequency_source),

        "movie_id_candidate_has_strong_source": id_strong,
        "movie_description_candidate_has_strong_source": desc_strong,
        "movie_rating_candidate_has_strong_source": rating_strong,
        "movie_duration_candidate_has_strong_source": duration_strong,
        "movie_people_candidate_has_strong_source": people_strong,
        "movie_text_candidate_has_strong_source": text_strong,
        "movie_genre_candidate_has_direct_mapping": genre_direct,

        # 一个综合分，方便推理时不用重新解析 source。
        "movie_strong_source_score": (
            1.0 * has_direct_source
            + 1.2 * has_metadata_source
            + 1.4 * candidate_equals_allowed_direct
            + 0.5 * min(allowed_direct_confidence, 1.0)
            - 0.6 * weak_only
        ),
    }



MOVIE_CONSISTENCY_SUSPICIOUS_MAP = {}


def compute_movie_consistency_features(row: pd.Series, candidate_value: Any, dirty_value: Any, ctx: dict) -> Dict[str, Any]:
    row_id = int(row.get("row_id"))
    column = str(row.get("column"))
    source_text = str(row.get("candidate_source", row.get("candidate_source_text", "")) or "")
    sources = {x.strip() for x in source_text.split("|") if x.strip()}
    extra = parse_extra_info_as_dict(row.get("extra_info", ""))

    is_consistency_source = int("movie_metadata_consistency" in sources or "movie_metadata_consistency" in source_text)
    suspicious = MOVIE_CONSISTENCY_SUSPICIOUS_MAP.get((row_id, column), {})

    conf = safe_float(extra.get("confidence", suspicious.get("consensus_confidence", 0.0)), 0.0)
    margin = safe_float(extra.get("margin", suspicious.get("consensus_margin", 0.0)), 0.0)
    support_count = safe_int(extra.get("support_count", suspicious.get("consensus_support_count", 0)), 0)
    total_count = safe_int(extra.get("total_count", suspicious.get("consensus_total_count", 0)), 0)
    consistency_name = extra.get("consistency_name", suspicious.get("consistency_name", ""))
    lhs_key = extra.get("lhs_key", suspicious.get("lhs_key", ""))

    suspicious_value = canonical_empty_text(suspicious.get("consensus_value", "")) if suspicious else ""
    candidate_equals_consistency = int(suspicious_value not in {"", "empty"} and value_equal_for_column(column, candidate_value, suspicious_value))

    strength = 0.0
    if is_consistency_source:
        strength += 1.0
    strength += conf
    strength += 0.5 * margin
    strength += min(support_count, 10) * 0.04
    if candidate_equals_consistency:
        strength += 0.5

    return {
        "is_from_movie_consistency": int(safe_int(row.get("is_from_movie_consistency", 0), 0) or is_consistency_source),
        "contains_movie_consistency_source": is_consistency_source,
        "movie_consistency_name": consistency_name,
        "movie_consistency_lhs_key": lhs_key,
        "movie_consistency_confidence": conf,
        "movie_consistency_margin": margin,
        "movie_consistency_support_count": support_count,
        "movie_consistency_total_count": total_count,
        "candidate_equals_movie_consistency_value": candidate_equals_consistency,
        "movie_consistency_strength": strength,
        "has_movie_consistency_suspicious_record": int(bool(suspicious)),
    }


MOVIE_DICTIONARY_SOURCES = {
    "id_to_name", "id_to_year", "id_to_release_date", "id_to_duration",
    "id_to_rating_value", "id_to_rating_count", "name_to_year", "name_to_release_date",
    "name_to_duration", "name_to_country", "name_year_to_duration",
    "name_year_to_release_date", "name_year_to_country", "director_genre_to_duration",
    "director_location_to_duration", "rating_count_genre_to_duration",
    "rating_value_count_to_duration", "name_year_to_rating_value", "name_year_to_rating_count",
    "rating_count_genre_to_review_count", "name_to_director", "name_to_creator",
    "name_to_actors", "name_to_cast", "director_creator_to_actors", "actors_to_cast",
    "release_date_year_consistency", "release_date_country_consistency", "country_to_language",
    "cast_prefix_to_actors",
    "name_to_id", "name_year_to_id", "name_to_description", "name_year_to_description",
    "name_year_to_director", "name_year_to_creator",
}
MOVIE_CANONICAL_SOURCES = {
    "canonical_year_restore", "canonical_release_date_restore", "canonical_duration_restore",
    "canonical_rating_value_restore", "canonical_rating_count_restore",
    "canonical_review_count_restore", "canonical_genre_list_restore",
    "canonical_location_country_restore",
    "direct_year_triple", "direct_release_date_wide_limited",
    "direct_genre_conservative_mapping", "direct_duration_single_format",
    "direct_rating_value_single_format",
}


def compute_candidate_source_features(row: pd.Series) -> Dict[str, Any]:
    source_text = str(row.get("candidate_source", row.get("candidate_source_text", "")) or "")
    sources = {x.strip() for x in source_text.split("|") if x.strip()}

    contains_dictionary_source = int(
        any(s in MOVIE_DICTIONARY_SOURCES for s in sources)
        or any(s.startswith(MOVIE_METADATA_STRONG_SOURCE_PREFIXES) for s in sources)
    )
    contains_canonical_source = int(
        any(s in MOVIE_CANONICAL_SOURCES or s.startswith("canonical_") or s.startswith("direct_") for s in sources)
    )
    contains_movie_consistency_source = int("movie_metadata_consistency" in sources or "movie_metadata_consistency" in source_text)
    contains_rule_source = int(any(s in {"expected_values_from_rules", "rule_expected_value"} for s in sources))
    contains_neighbor_source = int(any(s in {"neighbor_majority", "similar_row_target"} for s in sources))
    contains_column_source = int(any(s in {"column_top_value", "alternative_values_from_column", "similar_column_value"} for s in sources))
    contains_hint_source = int("suggested_correct_value" in sources)
    contains_direct_source = int(any(s.startswith("direct_") for s in sources) or any(s in MOVIE_DIRECT_SOURCES for s in sources))
    contains_strong_metadata_source = int(
        contains_movie_consistency_source == 1
        or any(s.startswith(MOVIE_METADATA_STRONG_SOURCE_PREFIXES) for s in sources)
        or any(s in MOVIE_METADATA_STRONG_SOURCES for s in sources)
    )

    source_count = safe_int(row.get("source_count", len(sources)), len(sources))

    weak_frequency_sources = {"column_top_value", "alternative_values_from_column", "similar_column_value", "global_top_value"}
    contains_weak_frequency_source = int(any(s in weak_frequency_sources for s in sources))
    is_weak_only = int(contains_weak_frequency_source == 1 and contains_direct_source == 0 and contains_strong_metadata_source == 0 and contains_rule_source == 0)

    return {
        "candidate_source_text": source_text,
        "source_count": source_count,
        "is_multi_source_candidate": int(source_count >= 2),
        "contains_dictionary_source": contains_dictionary_source,
        "contains_movie_dictionary_source": contains_dictionary_source,
        "contains_canonical_source": contains_canonical_source,
        "contains_movie_canonical_source": contains_canonical_source,
        "contains_movie_consistency_source": contains_movie_consistency_source,
        "contains_rule_source": contains_rule_source,
        "contains_neighbor_source": contains_neighbor_source,
        "contains_column_top_or_similarity_source": contains_column_source,
        "contains_hint_source": contains_hint_source,
        "contains_movie_direct_source": contains_direct_source,
        "contains_strong_movie_metadata_source": contains_strong_metadata_source,
        "contains_weak_frequency_source": contains_weak_frequency_source,
        "is_weak_frequency_only_candidate": is_weak_only,
        "contains_duration_source": int(any("duration" in s for s in sources)),
        "contains_year_source": int(any("year" in s for s in sources)),
        "contains_release_date_source": int(any("release_date" in s or "release date" in s for s in sources)),
        "contains_rating_source": int(any("rating" in s for s in sources)),
        "contains_review_source": int(any("review" in s for s in sources)),
        "contains_people_source": int(any(s in {
            "name_to_director", "name_to_creator", "name_to_actors", "name_to_cast",
            "name_year_to_director", "name_year_to_creator",
            "actors_to_cast", "cast_prefix_to_actors"
        } for s in sources)),
        "contains_description_source": int(any("description" in s for s in sources)),
        "contains_id_metadata_source": int(any(s.startswith("id_to_") or s in {"name_to_id", "name_year_to_id"} for s in sources)),
        "contains_name_year_metadata_source": int(any(s.startswith("name_year_to_") for s in sources)),
        "contains_name_metadata_source": int(any(s.startswith("name_to_") for s in sources)),
        "is_rule_and_dictionary_supported": int(contains_rule_source == 1 and contains_dictionary_source == 1),
        "is_neighbor_and_rule_agree": int(contains_neighbor_source == 1 and contains_rule_source == 1),
        "is_pattern_restore_candidate": int(safe_int(row.get("is_from_pattern_restore", 0), 0) == 1 or contains_canonical_source == 1),
    }


# ============================================================
# 8. 主流程
# ============================================================

def main():
    global MOVIE_CONSISTENCY_SUSPICIOUS_MAP, MOVIE_DIRECT_METADATA_SUSPICIOUS_MAP

    if not path_exists_nonempty(CANDIDATES_FILE):
        raise FileNotFoundError(f"候选展开文件不存在: {CANDIDATES_FILE}")
    if not path_exists_nonempty(CONTEXT_FILE):
        raise FileNotFoundError(f"上下文文件不存在: {CONTEXT_FILE}")

    candidates_df = pd.read_csv(CANDIDATES_FILE, encoding="utf-8-sig")
    required_cols = {"row_id", "column", "dirty_value", "candidate_value"}
    if not required_cols.issubset(set(candidates_df.columns)):
        raise ValueError(f"候选文件必须至少包含 {required_cols}，当前字段为: {list(candidates_df.columns)}")

    candidates_df["row_id"] = pd.to_numeric(candidates_df["row_id"], errors="raise").astype(int)
    candidates_df["column"] = candidates_df["column"].astype(str).str.strip()

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

    MOVIE_CONSISTENCY_SUSPICIOUS_MAP = load_movie_consistency_suspicious_map(MOVIE_CONSISTENCY_SUSPICIOUS_CELLS_FILE)
    print(f"[INFO] movie consistency suspicious map: {len(MOVIE_CONSISTENCY_SUSPICIOUS_MAP)}")

    MOVIE_DIRECT_METADATA_SUSPICIOUS_MAP = load_movie_direct_metadata_suspicious_map(MOVIE_DIRECT_METADATA_SUSPICIOUS_CELLS_FILE)
    print(f"[INFO] movie direct metadata suspicious map: {len(MOVIE_DIRECT_METADATA_SUSPICIOUS_MAP)}")

    # 将 dirty table fallback context 补进 context_map，避免新增 direct/metadata 候选因缺 ctx 被跳过。
    fallback_context_map = build_fallback_contexts_from_dirty_table(RAW_DATA_FILE)
    added_fallback_context = 0
    for k, v in fallback_context_map.items():
        if k not in context_map:
            context_map[k] = v
            added_fallback_context += 1
    if fallback_context_map:
        print(f"[INFO] dirty-table fallback contexts: {len(fallback_context_map)}, added={added_fallback_context}")

    cooccur_dict = load_or_build_cooccurrence_dict()

    feature_rows = []
    missing_context_count = 0

    for _, row in candidates_df.iterrows():
        row_id = int(row["row_id"])
        column = str(row["column"]).strip()
        dirty_value = canonical_empty_text(row["dirty_value"])
        candidate_value = canonical_empty_text(row["candidate_value"])

        ctx = context_map.get((row_id, column), None)
        if ctx is None:
            missing_context_count += 1
            continue

        rule_feats = compute_rule_features(candidate_value, dirty_value, ctx)
        bayes_feats = compute_bayes_features(candidate_value, ctx, cooccur_dict)
        trans_feats = compute_transform_features(candidate_value, dirty_value, column)
        plaus_feats = compute_column_plausibility_features(candidate_value, dirty_value, ctx)
        source_feats = compute_candidate_source_features(row)
        consistency_feats = compute_movie_consistency_features(row, candidate_value, dirty_value, ctx)
        direct_metadata_feats = compute_movie_direct_metadata_features(row, candidate_value, dirty_value, ctx)

        out = row.to_dict()
        out.update({
            "semantic_type": ctx.get("semantic_type", None),
            "detected_type": ctx.get("detected_type", None),
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
            "parsed_year": ctx.get("parsed_year", None),
            "parsed_release_year": ctx.get("parsed_release_year", None),
            "parsed_release_country": ctx.get("parsed_release_country", None),
            "duration_minutes": ctx.get("duration_minutes", None),
            "rating_value_float": ctx.get("rating_value_float", None),
            "count_value_int": ctx.get("count_value_int", None),
            "review_total": ctx.get("review_total", None),
            "list_token_count": ctx.get("list_token_count", None),
            "list_token_count_bucket": ctx.get("list_token_count_bucket", None),
            "text_length_bucket": ctx.get("text_length_bucket", None),
            "current_value_pattern": ctx.get("current_value_pattern", None),
            "dominant_pattern": ctx.get("dominant_pattern", None),
        })
        out.update(rule_feats)
        out.update(bayes_feats)
        out.update(trans_feats)
        out.update(plaus_feats)
        out.update(source_feats)
        out.update(consistency_feats)
        out.update(direct_metadata_feats)

        feature_rows.append(out)

    features_df = pd.DataFrame(feature_rows)
    output_features_csv = resolve_output_path(OUTPUT_FEATURES_CSV, CANDIDATES_FILE)
    features_df.to_csv(output_features_csv, index=False, encoding="utf-8-sig")

    print(f"[INFO] 候选记录数: {len(candidates_df)}")
    print(f"[INFO] 成功构造特征的记录数: {len(features_df)}")
    print(f"[INFO] 缺失上下文的候选记录数: {missing_context_count}")
    print(f"[INFO] 输出文件: {output_features_csv}")

    if not features_df.empty:
        show_cols = [c for c in [
            "row_id", "column", "dirty_value", "candidate_value",
            "candidate_rule_support_ratio", "rule_support_gain",
            "bayes_mean_cond_prob", "bayes_max_cond_prob",
            "edit_similarity_to_dirty", "avg_keyboard_distance",
            "phonetic_similarity_to_dirty", "candidate_in_column_top_values",
            "contains_dictionary_source", "contains_movie_canonical_source",
            "contains_movie_consistency_source", "movie_consistency_confidence",
            "movie_consistency_strength", "candidate_matches_similar_row_target",
            "movie_candidate_is_duration", "movie_candidate_duration_restoration",
            "movie_candidate_is_year", "movie_candidate_is_release_date",
            "movie_candidate_is_rating_value", "movie_candidate_is_count",
            "movie_candidate_is_review_count", "movie_candidate_list_token_count",
            "movie_column_is_time", "movie_column_is_rating",
            "movie_column_is_people", "movie_column_is_list",
            "candidate_equals_allowed_direct_value", "contains_movie_direct_source",
            "contains_strong_movie_metadata_source", "is_weak_frequency_only_candidate",
            "movie_strong_source_score",
        ] if c in features_df.columns]
        print("\n[INFO] 前 20 行特征预览：")
        print(features_df[show_cols].head(20).to_string(index=False))
        print("\n[INFO] 按列候选特征数量统计：")
        print(features_df.groupby("column")["candidate_value"].count().sort_values(ascending=False).head(30).to_string())


if __name__ == "__main__":
    main()

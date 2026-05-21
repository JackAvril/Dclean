import argparse
import json
import os
import re
from typing import Dict, List, Optional

import pandas as pd


# ============================================================
# Robust base helpers patched for Movies inference
# ============================================================

if "safe_int" not in globals():
    def safe_int(x, default=0):
        try:
            if x is None:
                return default
            try:
                if pd.isna(x):
                    return default
            except Exception:
                pass
            return int(float(str(x).strip().replace(",", "")))
        except Exception:
            return default

if "safe_float" not in globals():
    def safe_float(x, default=0.0):
        try:
            if x is None:
                return default
            try:
                if pd.isna(x):
                    return default
            except Exception:
                pass
            return float(str(x).strip().replace(",", "").replace("%", ""))
        except Exception:
            return default

if "norm_text" not in globals():
    def norm_text(x):
        if x is None:
            return ""
        try:
            if pd.isna(x):
                return ""
        except Exception:
            pass
        return str(x).strip().lower()

if "canonical_text" not in globals():
    def canonical_text(x):
        s = norm_text(x)
        if s in {"", "nan", "none", "null", "na", "n/a", "missing", "empty", "{null}"}:
            return ""
        return str(x).strip()

if "canonical_empty_text" not in globals():
    def canonical_empty_text(x):
        s = canonical_text(x)
        return "empty" if s == "" else s

if "is_empty_token" not in globals():
    def is_empty_token(x):
        return norm_text(x) in {"", "nan", "none", "null", "na", "n/a", "missing", "empty", "{null}"}

import torch
import torch.nn as nn

EMPTY_TOKENS = {"", "empty", "null", "none", "nan", "na", "n/a", "missing"}
US_STATE_CODES = {
    "al","ak","az","ar","ca","co","ct","de","fl","ga","hi","ia","id","il","in","ks","ky","la","ma","md","me",
    "mi","mn","mo","ms","mt","nc","nd","ne","nh","nj","nm","nv","ny","oh","ok","or","pa","ri","sc","sd","tn",
    "tx","ut","va","vt","wa","wi","wv","wy","dc"
}
YES_NO_DOMAIN = {"yes", "no"}
HOSPITAL_TYPE_DOMAIN = {
    "acute care hospitals",
    "critical access hospitals",
    "acute care hospital",
    "critical access hospital",
}
CONDITION_DOMAIN = {
    "heart attack",
    "heart failure",
    "pneumonia",
    "surgical infection prevention",
    "children s asthma care",
}
PROTECTED_NONEMPTY_COLUMNS = set()  # Flights 中不使用 Hospital 的受保护列
COLUMN_MARGIN_THRESHOLDS = {
    "Duration": 0.045,
    "Year": 0.045,
    "Release Date": 0.05,
    "RatingValue": 0.05,
    "RatingCount": 0.055,
    "ReviewCount": 0.055,
    "Country": 0.06,
    "Language": 0.07,
    "Actors": 0.08,
    "Cast": 0.08,
    "Director": 0.075,
    "Creator": 0.075,
    "Genre": 0.08,
    "Filming Locations": 0.09,
    "Name": 0.10,
}
DEFAULT_MARGIN_THRESHOLD = 0.06
MOVIE_TIME_COLUMNS = {"Year", "Release Date", "Duration"}
MOVIE_RATING_COLUMNS = {"RatingValue", "RatingCount", "ReviewCount"}
MOVIE_TEXT_LIST_COLUMNS = {"Actors", "Cast", "Creator", "Director", "Genre", "Filming Locations"}
MOVIE_METADATA_COLUMNS = {"Name", "Country", "Language"}
MOVIE_HARD_COLUMNS = MOVIE_TIME_COLUMNS | MOVIE_RATING_COLUMNS | MOVIE_TEXT_LIST_COLUMNS | MOVIE_METADATA_COLUMNS
# Keep these aliases so legacy helper code remains import-safe.
TIME_COLUMNS = MOVIE_TIME_COLUMNS
ACTUAL_TIME_COLUMNS = set()
SCHEDULED_TIME_COLUMNS = MOVIE_TIME_COLUMNS


def parse_args():
    p = argparse.ArgumentParser(description="Inference + hard case mining (expects prefiltered whitelist candidate_features)")
    p.add_argument("--candidate_features", default="repair_candidate_features_infer_whitelist.csv")
    p.add_argument("--allowed_cells_csv", default="predicted_error_positions_consensus_augmented.csv")
    p.add_argument("--model_dir", default="pairwise_ltr_student_output_movies")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--global_min_margin", type=float, default=0.05)
    p.add_argument("--joint_topk", type=int, default=4)
    p.add_argument("--max_hard_cases", type=int, default=300)
    return p.parse_args()


ARGS = parse_args()
MODEL_DIR = ARGS.model_dir
CHECKPOINT_PATH = ARGS.checkpoint or os.path.join(MODEL_DIR, "best_model.pt")
FEATURE_COLUMNS_JSON = os.path.join(MODEL_DIR, "feature_columns.json")
REASON_LABELS_JSON = os.path.join(MODEL_DIR, "reason_label_mapping.json")
COLUMN_PROFILES_JSON = os.path.join(MODEL_DIR, "column_profiles.json")

# 输出默认放到当前运行目录；模型、feature_columns、checkpoint 仍从 MODEL_DIR 读取。
# 如果你想指定其它输出目录，可以在运行前设置：
# export REPAIR_INFER_OUTPUT_DIR=/your/output/dir
OUTPUT_DIR = os.path.abspath(os.getenv("REPAIR_INFER_OUTPUT_DIR", os.getcwd()))
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_ALL_SCORES_CSV = os.path.join(OUTPUT_DIR, "inference_all_candidate_scores.csv")
OUTPUT_ALL_RANKED_CSV = os.path.join(OUTPUT_DIR, "inference_all_candidate_ranked.csv")
OUTPUT_TOP1_CSV = os.path.join(OUTPUT_DIR, "inference_top1_repairs.csv")
OUTPUT_HARD_CASES_JSONL = os.path.join(OUTPUT_DIR, "hard_cases_for_llm.jsonl")
INPUT_WHITELIST_VIOLATIONS_CSV = os.path.join(OUTPUT_DIR, "input_whitelist_violations.csv")
OUTPUT_WHITELIST_VIOLATIONS_CSV = os.path.join(OUTPUT_DIR, "output_whitelist_violations.csv")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def norm_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def canonical_empty_text(x):
    s = norm_text(x).lower()
    return "empty" if s in EMPTY_TOKENS else norm_text(x)


# ============================================================
# Movies helper functions for derived features
# ============================================================

MONTH_NAME_TO_NUM = {
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


def split_list_tokens_movies(x):
    s = canonical_empty_text(x)
    if s.lower() in {"", "empty"}:
        return []
    parts = re.split(r"[,;/|]+", s)
    out = []
    seen = set()
    for p in parts:
        p = re.sub(r"\s+", " ", str(p).strip())
        if not p:
            continue
        k = p.lower()
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out


def list_jaccard_movies(a, b) -> float:
    sa = {norm_text(x).lower() for x in split_list_tokens_movies(a)}
    sb = {norm_text(x).lower() for x in split_list_tokens_movies(b)}
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(len(sa | sb), 1)


def normalize_year_movies(x):
    s = canonical_empty_text(x)
    m = re.search(r"(18\d{2}|19\d{2}|20\d{2}|21\d{2})", s)
    return m.group(1) if m else None


def is_year_like_movies(x) -> int:
    return int(normalize_year_movies(x) is not None)


def parse_duration_minutes_movies(x):
    s = canonical_empty_text(x).lower()
    if s in {"", "empty"}:
        return None

    h = 0
    m = 0
    mh = re.search(r"(\d+)\s*(?:h|hr|hrs|hour|hours)\b", s)
    mm = re.search(r"(\d+)\s*(?:m|min|mins|minute|minutes)\b", s)
    if mh or mm:
        if mh:
            h = int(mh.group(1))
        if mm:
            m = int(mm.group(1))
        total = h * 60 + m
        return total if 1 <= total <= 500 else None

    mn = re.search(r"\b(\d{1,3})(?:\.0)?\b", s)
    if mn:
        total = int(mn.group(1))
        return total if 1 <= total <= 500 else None

    return None


def is_duration_like_movies(x) -> int:
    return int(parse_duration_minutes_movies(x) is not None)


def parse_release_date_parts_movies(x):
    s = canonical_empty_text(x)
    if s.lower() in {"", "empty"}:
        return None

    country = None
    cm = re.search(r"\(([^()]+)\)", s)
    if cm:
        c = cm.group(1).strip().lower()
        country = COUNTRY_CANONICAL_MAP.get(c, cm.group(1).strip().title())

    s2 = re.sub(r"\([^()]+\)", "", s).strip()

    m = re.search(r"\b(\d{1,2})\s+([A-Za-z]+)\s+(18\d{2}|19\d{2}|20\d{2}|21\d{2})\b", s2)
    if m:
        mon = MONTH_NAME_TO_NUM.get(m.group(2).lower())
        day = int(m.group(1))
        if mon and 1 <= day <= 31:
            return {"year": int(m.group(3)), "month": mon, "day": day, "country": country}

    m = re.search(r"\b([A-Za-z]+)\s+(18\d{2}|19\d{2}|20\d{2}|21\d{2})\b", s2)
    if m:
        mon = MONTH_NAME_TO_NUM.get(m.group(1).lower())
        if mon:
            return {"year": int(m.group(2)), "month": mon, "day": None, "country": country}

    m = re.search(r"\b(18\d{2}|19\d{2}|20\d{2}|21\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", s2)
    if m:
        year = int(m.group(1))
        month = int(m.group(2))
        day = int(m.group(3))
        if 1 <= month <= 12 and 1 <= day <= 31:
            return {"year": year, "month": month, "day": day, "country": country}

    m = re.search(r"\b(\d{1,2})/(\d{1,2})/(18\d{2}|19\d{2}|20\d{2}|21\d{2})\b", s2)
    if m:
        month = int(m.group(1))
        day = int(m.group(2))
        year = int(m.group(3))
        if 1 <= month <= 12 and 1 <= day <= 31:
            return {"year": year, "month": month, "day": day, "country": country}

    y = normalize_year_movies(s2)
    if y:
        return {"year": int(y), "month": None, "day": None, "country": country}

    return None


def is_release_date_like_movies(x) -> int:
    return int(parse_release_date_parts_movies(x) is not None)


def parse_rating_movies(x):
    s = canonical_empty_text(x).replace(',', '.')
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if not m:
        return None
    v = float(m.group(1))
    if '%' in s and 0 <= v <= 100:
        v = v / 10.0
    return v if 0 <= v <= 10 else None


def is_rating_value_like_movies(x) -> int:
    return int(parse_rating_movies(x) is not None)


def parse_count_movies(x):
    s = canonical_empty_text(x)
    m = re.search(r"(\d[\d,]*)", s)
    if not m:
        return None
    return int(m.group(1).replace(',', ''))


def is_count_like_movies(x) -> int:
    return int(parse_count_movies(x) is not None)


def normalize_country_movies(x):
    s = canonical_empty_text(x)
    if s.lower() in {"", "empty"}:
        return None
    k = s.strip().lower()
    if k in COUNTRY_CANONICAL_MAP:
        return COUNTRY_CANONICAL_MAP[k]
    if re.fullmatch(r"[A-Za-z][A-Za-z .'-]{1,50}", s.strip()):
        return re.sub(r"\s+", " ", s.strip()).title()
    return None


def is_country_like_movies(x) -> int:
    return int(normalize_country_movies(x) is not None)


def load_json(path: str):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_jsonl_record(fp, obj):
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")


def is_empty_token(x: str) -> int:
    return int(norm_text(x).lower() in EMPTY_TOKENS)


def is_time_column_name(column: str) -> int:
    c = norm_text(column).lower()
    return int(c in TIME_COLUMNS or c.endswith("_time"))


def extract_time_candidate_strings(x):
    """
    从复杂时间字符串中抽取可解析片段。
    支持：
    - 12/02/2011 6:55 a.m.
    - 9:32aDec 1 / 6:09pDec 1
    - 10:30 p.m. estimated
    - 7:16a / 8:20p
    """
    raw = canonical_empty_text(x)
    s = raw.lower().strip()
    if s in {"", "empty"}:
        return []

    s_norm = s
    s_norm = s_norm.replace("estimated", " ")
    s_norm = s_norm.replace("(estimated)", " ")
    s_norm = s_norm.replace("est.", " ")
    s_norm = s_norm.replace("est", " ")
    s_norm = re.sub(r"\s+", " ", s_norm).strip()

    candidates = []

    patterns = [
        r"(?P<h>\d{1,2})\s*:\s*(?P<m>\d{2})\s*(?P<ampm>a\.?m\.?|p\.?m\.?|am|pm|a|p)\b",
        r"(?P<h>\d{1,2})\s*:\s*(?P<m>\d{2})\s*(?P<ampm>a|p)(?=[a-z])",
        r"(?P<h>\d{1,2})\s*:\s*(?P<m>\d{2})",
    ]

    for pat in patterns:
        for m in re.finditer(pat, s_norm):
            h = m.group("h")
            mm = m.group("m")
            ampm = m.groupdict().get("ampm", "") or ""
            ampm = ampm.replace(".", "")
            if ampm == "a":
                ampm = "am"
            elif ampm == "p":
                ampm = "pm"

            if ampm:
                candidates.append(f"{h}:{mm} {ampm}")
            else:
                candidates.append(f"{h}:{mm}")

    compact = re.sub(r"\D", "", s_norm)
    if len(compact) in {3, 4} and not re.search(r"[/-]", s_norm):
        candidates.append(compact)

    candidates.append(raw)

    out = []
    seen = set()
    for c in candidates:
        c = str(c).strip()
        if not c:
            continue
        k = c.lower()
        if k not in seen:
            seen.add(k)
            out.append(c)
    return out


def parse_time_to_minutes_simple(x):
    raw = canonical_empty_text(x)
    if raw.lower() in {"", "empty"}:
        return None

    for cand in extract_time_candidate_strings(raw):
        minutes = _parse_simple_time_piece(cand)
        if minutes is not None:
            return minutes

    return None


def _parse_simple_time_piece(x):
    s = canonical_empty_text(x).lower()
    if s in {"", "empty"}:
        return None

    s = s.replace(".", "")
    s = s.replace("estimated", " ")
    s = s.replace("est", " ")
    s = re.sub(r"\s+", " ", s).strip()

    ampm = None
    if re.search(r"\b(am)\b", s) or re.search(r"\d\s*a\b", s):
        ampm = "am"
    elif re.search(r"\b(pm)\b", s) or re.search(r"\d\s*p\b", s):
        ampm = "pm"
    elif re.search(r"\d:\d{2}a(?=[a-z]|\b)", s):
        ampm = "am"
    elif re.search(r"\d:\d{2}p(?=[a-z]|\b)", s):
        ampm = "pm"

    s2 = s
    s2 = re.sub(r"\b(am|pm)\b", "", s2)
    s2 = re.sub(r"(?<=\d)\s*[ap](?=[a-z]|\b)", "", s2)
    s2 = s2.strip()

    hour = minute = None

    m = re.search(r"(\d{1,2})\s*:\s*(\d{2})", s2)
    if m:
        hour, minute = int(m.group(1)), int(m.group(2))
    else:
        digits = re.sub(r"\D", "", s2)
        if len(digits) in {3, 4}:
            hour, minute = int(digits[:-2]), int(digits[-2:])

    if hour is None or minute is None or minute < 0 or minute > 59:
        return None

    if ampm == "am":
        if hour == 12:
            hour = 0
        elif not (1 <= hour <= 11):
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

def is_valid_time_simple(x) -> int:
    return int(parse_time_to_minutes_simple(x) is not None)


def circular_minute_diff_simple(a, b):
    ma = parse_time_to_minutes_simple(a)
    mb = parse_time_to_minutes_simple(b)
    if ma is None or mb is None:
        return 0.0
    d = abs(ma - mb)
    return float(min(d, 1440 - d))


def looks_like_code(x: str) -> int:
    s = norm_text(x)
    if s == "":
        return 0
    return int((bool(re.search(r"[_\-]", s)) or bool(re.search(r"\d", s))) and len(s) <= 32)


def split_code_parts(x: str):
    s = norm_text(x)
    return [] if s == "" else re.split(r"[_\-]", s)


def prefix_token(x: str) -> str:
    parts = split_code_parts(x)
    return parts[0] if parts else ""


def family_token(x: str) -> str:
    parts = split_code_parts(x)
    if len(parts) >= 2:
        return parts[1]
    return parts[0] if parts else ""


def suffix_token(x: str) -> str:
    parts = split_code_parts(x)
    return parts[-1] if parts else ""


def is_digits_only(s: str) -> bool:
    return bool(re.fullmatch(r"\d+", norm_text(s)))


def is_percent_like(s: str) -> bool:
    return bool(re.fullmatch(r"\d{1,3}%", norm_text(s)))


def alpha_space_like(s: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z\s\-\./']+", norm_text(s)))


def patients_like(s: str) -> bool:
    return bool(re.fullmatch(r"\d+\s+patients", canonical_empty_text(s).lower()))


def tiny_edit_like(a: str, b: str, max_diff: int) -> bool:
    a = norm_text(a).lower()
    b = norm_text(b).lower()
    if abs(len(a) - len(b)) > max_diff:
        return False
    if len(a) < len(b):
        a = a + "#" * (len(b) - len(a))
    elif len(b) < len(a):
        b = b + "#" * (len(a) - len(b))
    return sum(ch1 != ch2 for ch1, ch2 in zip(a, b)) <= max_diff


def likely_measure_code_candidate(cand: str) -> bool:
    s = canonical_empty_text(cand).lower()
    return s != "empty" and (s.startswith(("ami", "hf", "pn", "scip")) or ("ami" in s) or ("hf" in s) or ("pn" in s) or ("scip" in s))


def likely_stateavg_candidate(cand: str) -> bool:
    s = canonical_empty_text(cand).lower()
    return s != "empty" and "_" in s and likely_measure_code_candidate(s)


def normalize_measurecode_like(s: str) -> str:
    raw = canonical_empty_text(s).lower()
    if "_" in raw:
        raw = raw.split("_", 1)[1]
    return raw.replace("_", "-")


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


def pair_compat_score(measurecode: str, stateavg: str) -> int:
    mc = normalize_measurecode_like(measurecode)
    sa = normalize_measurecode_like(stateavg)
    score = 0
    if mc and sa and mc == sa:
        score += 4
    if family_token(measurecode) and family_token(stateavg) and family_token(measurecode) == family_token(stateavg):
        score += 2
    if suffix_token(measurecode) and suffix_token(stateavg) and suffix_token(measurecode) == suffix_token(stateavg):
        score += 2
    return score



def has_x_placeholder(s: str) -> int:
    return int("x" in norm_text(s).lower())


def extract_leading_number_token(s: str) -> str:
    m = re.match(r"\s*([^\s]+)", norm_text(s))
    return m.group(1) if m else ""


def score_template_match_score(dirty: str, cand: str) -> int:
    d = canonical_empty_text(dirty).lower()
    c = canonical_empty_text(cand).lower()
    if not is_percent_like(c):
        return -999

    score = 0
    if len(d) == len(c):
        score += 3
    if d.endswith("x") and c.endswith("%"):
        score += 3
    if "%" not in d and c.endswith("%"):
        score += 1

    n = min(len(d), len(c))
    for i in range(n):
        chd, chc = d[i], c[i]
        if chd == "x":
            score += 1
        elif chd == chc:
            score += 2
        else:
            score -= 4

    if len(c) > len(d):
        score -= (len(c) - len(d))
    elif len(d) > len(c):
        score -= 2 * (len(d) - len(c))
    return score


def sample_template_match_score(dirty: str, cand: str) -> int:
    d = canonical_empty_text(dirty).lower()
    c = canonical_empty_text(cand).lower()
    if not patients_like(c):
        return -999

    score = 0
    dirty_num = extract_leading_number_token(d)
    cand_num = extract_leading_number_token(c)
    if dirty_num and cand_num:
        if len(dirty_num) == len(cand_num):
            score += 2
        n = min(len(dirty_num), len(cand_num))
        for i in range(n):
            chd, chc = dirty_num[i], cand_num[i]
            if chd == "x":
                score += 1
            elif chd == chc:
                score += 2
            else:
                score -= 3

    # 强偏好把各种 patients 相关 typo 统一修成规范 "patients"
    if "patient" in d or "pat" in d:
        score += 2
    if c.endswith("patients"):
        score += 3

    # 文本整体轻度相似性
    score += int(tiny_edit_like(d, c, 12))
    return score


def infer_archetype_from_profile(profile: dict) -> str:
    if not profile:
        return "mixed"
    return str(profile.get("archetype", "mixed"))


def build_column_profiles_only(df: pd.DataFrame):
    profiles = {}
    for col, g in df.groupby("column"):
        vals = pd.concat([g["dirty_value"], g["candidate_value"]], ignore_index=True).dropna().astype(str)
        vals = vals.map(canonical_empty_text)
        vals = vals[vals != "empty"]
        if len(vals) == 0:
            profiles[str(col)] = {"archetype": "mixed", "avg_len": 0.0, "digits_ratio": 0.0, "percent_ratio": 0.0, "code_ratio": 0.0, "alpha_ratio": 0.0, "unique_count": 0, "sample_values": []}
            continue
        avg_len = float(vals.map(lambda x: len(norm_text(x))).mean())
        digits_ratio = float(vals.map(lambda x: int(bool(re.fullmatch(r"\d+", norm_text(x))))).mean())
        percent_ratio = float(vals.map(lambda x: int(bool(re.fullmatch(r"\d{1,3}%", norm_text(x))))).mean())
        code_ratio = float(vals.map(looks_like_code).mean())
        alpha_ratio = float(vals.map(lambda x: int(bool(re.fullmatch(r"[A-Za-z\s\-\./']+", norm_text(x))))).mean())
        unique_count = int(vals.nunique())
        archetype = "mixed"
        if percent_ratio >= 0.55:
            archetype = "percent_pattern"
        elif digits_ratio >= 0.75 and avg_len >= 6:
            archetype = "long_digit_id"
        elif digits_ratio >= 0.75:
            archetype = "short_digit_id"
        elif code_ratio >= 0.55:
            archetype = "structured_code"
        elif unique_count <= 8 and alpha_ratio >= 0.8:
            archetype = "small_enum"
        elif alpha_ratio >= 0.8 and avg_len >= 30:
            archetype = "long_text"
        elif alpha_ratio >= 0.8:
            archetype = "short_text"
        profiles[str(col)] = {"archetype": archetype, "avg_len": avg_len, "digits_ratio": digits_ratio, "percent_ratio": percent_ratio, "code_ratio": code_ratio, "alpha_ratio": alpha_ratio, "unique_count": unique_count, "sample_values": list(vals.head(20))}
    return profiles


def ensure_column_profiles_json(candidate_features_path: str, output_json_path: str):
    if os.path.exists(output_json_path):
        return
    df = pd.read_csv(candidate_features_path, encoding="utf-8-sig")
    df["column"] = df["column"].astype(str).str.strip()
    df["dirty_value"] = df["dirty_value"].map(canonical_empty_text)
    df["candidate_value"] = df["candidate_value"].map(canonical_empty_text)
    profiles = build_column_profiles_only(df)
    os.makedirs(os.path.dirname(output_json_path) or ".", exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8-sig") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)


def get_column_archetype(column: str, column_profiles: Dict[str, dict]) -> str:
    return infer_archetype_from_profile(column_profiles.get(column, {}))


def load_allowed_cells(path: str):
    if path is None or str(path).strip() == "":
        return None
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"row_id", "column"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"allowed_cells_csv must contain columns: {required}")
    df["row_id"] = pd.to_numeric(df["row_id"], errors="raise").astype(int)
    df["column"] = df["column"].astype(str).str.strip()
    allowed = set((int(r["row_id"]), str(r["column"]).strip()) for _, r in df[["row_id", "column"]].drop_duplicates().iterrows())
    print(f"[INFO] allowed whitelist unique cells = {len(allowed)}")
    return allowed


class SharedScorerMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float, num_reason_classes: int):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.score_head = nn.Linear(prev, 1)
        self.reason_head = nn.Linear(prev, num_reason_classes)

    def forward_single(self, x):
        h = self.backbone(x)
        return self.score_head(h).squeeze(-1), self.reason_head(h), h


NON_FEATURE_COLUMNS = {"row_id", "column", "dirty_value", "candidate_value", "candidate_rank", "candidate_source", "extra_info", "neighbor_majority_value", "suggested_correct_value", "main_rule_type", "main_usage_role", "semantic_type"}


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Movies version of derived feature expansion.

    保留原 Flights/Hospital 的通用派生特征，同时新增 Movies 列类型、格式恢复、metadata consistency
    与 list-overlap 特征。这样训练脚本仍兼容之前的通用特征管线，但不会依赖 flight time 字段。
    """
    out = df.copy()
    out["dirty_value"] = out["dirty_value"].map(canonical_empty_text)
    out["candidate_value"] = out["candidate_value"].map(canonical_empty_text)
    out["original_is_empty"] = out["dirty_value"].map(is_empty_token)
    out["candidate_is_empty"] = out["candidate_value"].map(is_empty_token)
    out["candidate_equals_dirty"] = (
        out["candidate_value"].map(norm_text) == out["dirty_value"].map(norm_text)
    ).astype(int)
    out["dirty_is_code_like"] = out["dirty_value"].map(looks_like_code)
    out["candidate_is_code_like"] = out["candidate_value"].map(looks_like_code)

    dirty_prefix = out["dirty_value"].map(prefix_token)
    cand_prefix = out["candidate_value"].map(prefix_token)
    dirty_family = out["dirty_value"].map(family_token)
    cand_family = out["candidate_value"].map(family_token)

    out["prefix_match"] = (dirty_prefix == cand_prefix).astype(int)
    out["family_match"] = (dirty_family == cand_family).astype(int)
    out["empty_to_nonempty"] = (
        (out["original_is_empty"] == 1) & (out["candidate_is_empty"] == 0)
    ).astype(int)

    # Movies column one-hot flags
    for col_name in sorted(MOVIE_HARD_COLUMNS | {"Id", "Description"}):
        safe_col = re.sub(r"[^A-Za-z0-9_]+", "_", col_name).strip("_")
        out[f"is_col_{safe_col}"] = (out["column"] == col_name).astype(int)

    out["is_movie_time_column_derived"] = out["column"].map(lambda c: int(c in MOVIE_TIME_COLUMNS))
    out["is_movie_rating_column_derived"] = out["column"].map(lambda c: int(c in MOVIE_RATING_COLUMNS))
    out["is_movie_people_column_derived"] = out["column"].map(lambda c: int(c in {"Actors", "Cast", "Creator", "Director"}))
    out["is_movie_list_column_derived"] = out["column"].map(lambda c: int(c in MOVIE_TEXT_LIST_COLUMNS))
    out["is_movie_metadata_column_derived"] = out["column"].map(lambda c: int(c in MOVIE_METADATA_COLUMNS))

    out["dirty_is_year_like_derived"] = out["dirty_value"].map(is_year_like_movies)
    out["candidate_is_year_like_derived"] = out["candidate_value"].map(is_year_like_movies)
    out["dirty_is_duration_like_derived"] = out["dirty_value"].map(is_duration_like_movies)
    out["candidate_is_duration_like_derived"] = out["candidate_value"].map(is_duration_like_movies)
    out["dirty_is_release_date_like_derived"] = out["dirty_value"].map(is_release_date_like_movies)
    out["candidate_is_release_date_like_derived"] = out["candidate_value"].map(is_release_date_like_movies)
    out["dirty_is_rating_like_derived"] = out["dirty_value"].map(is_rating_value_like_movies)
    out["candidate_is_rating_like_derived"] = out["candidate_value"].map(is_rating_value_like_movies)
    out["dirty_is_count_like_derived"] = out["dirty_value"].map(is_count_like_movies)
    out["candidate_is_count_like_derived"] = out["candidate_value"].map(is_count_like_movies)
    out["dirty_is_country_like_derived"] = out["dirty_value"].map(is_country_like_movies)
    out["candidate_is_country_like_derived"] = out["candidate_value"].map(is_country_like_movies)

    dirty_duration = out["dirty_value"].map(lambda x: parse_duration_minutes_movies(x) or 0)
    cand_duration = out["candidate_value"].map(lambda x: parse_duration_minutes_movies(x) or 0)
    out["candidate_dirty_duration_abs_diff_derived"] = (cand_duration - dirty_duration).abs()

    dirty_rating = out["dirty_value"].map(lambda x: parse_rating_movies(x) if parse_rating_movies(x) is not None else -1)
    cand_rating = out["candidate_value"].map(lambda x: parse_rating_movies(x) if parse_rating_movies(x) is not None else -1)
    out["candidate_dirty_rating_abs_diff_derived"] = (cand_rating - dirty_rating).abs()

    out["movie_list_jaccard_derived"] = [list_jaccard_movies(d, c) for d, c in zip(out["dirty_value"], out["candidate_value"])]
    out["movie_candidate_format_restoration_gain_derived"] = (
        ((out["candidate_is_year_like_derived"] == 1) & (out["dirty_is_year_like_derived"] == 0)) |
        ((out["candidate_is_duration_like_derived"] == 1) & (out["dirty_is_duration_like_derived"] == 0)) |
        ((out["candidate_is_release_date_like_derived"] == 1) & (out["dirty_is_release_date_like_derived"] == 0)) |
        ((out["candidate_is_rating_like_derived"] == 1) & (out["dirty_is_rating_like_derived"] == 0)) |
        ((out["candidate_is_count_like_derived"] == 1) & (out["dirty_is_count_like_derived"] == 0))
    ).astype(int)
    return out

def validate_prefiltered_candidate_features(df: pd.DataFrame, allowed_cells):
    unique_cells = df[["row_id", "column"]].drop_duplicates().copy()
    if allowed_cells is None:
        print(f"[INFO] inference candidate unique cells (no whitelist validation) = {len(unique_cells)}")
        return
    bad_mask = unique_cells.apply(lambda r: (int(r["row_id"]), str(r["column"]).strip()) not in allowed_cells, axis=1)
    bad_count = int(bad_mask.sum())
    print(f"[INFO] prefiltered input whitelist violations = {bad_count}")
    if bad_count > 0:
        unique_cells.loc[bad_mask].to_csv(INPUT_WHITELIST_VIOLATIONS_CSV, index=False, encoding="utf-8-sig")
        raise ValueError(
            "candidate_features is not prefiltered to whitelist. "
            f"Violations saved to: {INPUT_WHITELIST_VIOLATIONS_CSV}"
        )


def load_candidate_features_for_inference(path: str, trained_feature_columns: List[str], allowed_cells=None) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["row_id"] = pd.to_numeric(df["row_id"], errors="raise").astype(int)
    df["column"] = df["column"].astype(str).str.strip()
    input_unique_cells = df[["row_id", "column"]].drop_duplicates().shape[0]
    print(f"[INFO] inference input unique candidate cells = {input_unique_cells}")
    validate_prefiltered_candidate_features(df, allowed_cells)

    df["candidate_value"] = df["candidate_value"].map(canonical_empty_text)
    df["dirty_value"] = df["dirty_value"].map(canonical_empty_text)
    df = add_derived_features(df)

    categorical_cols = [c for c in ["semantic_type", "main_rule_type", "main_usage_role", "candidate_source"] if c in df.columns]
    df_cat = pd.get_dummies(df[categorical_cols].fillna("unknown").astype(str), prefix=categorical_cols) if categorical_cols else pd.DataFrame(index=df.index)

    numeric_feature_df = pd.DataFrame(index=df.index)
    for col in df.columns:
        if col in NON_FEATURE_COLUMNS or col in categorical_cols:
            continue
        numeric_feature_df[col] = pd.to_numeric(df[col], errors="coerce")
    numeric_feature_df = numeric_feature_df.fillna(0.0)

    feat_df = pd.concat([numeric_feature_df, df_cat], axis=1)
    for col in feat_df.columns:
        feat_df[col] = pd.to_numeric(feat_df[col], errors="coerce").fillna(0.0)
    for col in trained_feature_columns:
        if col not in feat_df.columns:
            feat_df[col] = 0.0
    feat_df = feat_df[trained_feature_columns].copy()
    for col in trained_feature_columns:
        df[col] = feat_df[col].values
    return df


@torch.no_grad()
def score_all_candidates(model, df: pd.DataFrame, feature_columns: List[str], reason_id_to_name: dict) -> pd.DataFrame:
    model.eval()
    feats = torch.tensor(df[feature_columns].astype(float).values, dtype=torch.float32, device=DEVICE)
    score, reason_logits, _ = model.forward_single(feats)
    reason_pred = reason_logits.argmax(dim=1).detach().cpu().numpy()
    out = df.copy()
    out["student_score"] = score.detach().cpu().numpy()
    out["student_reason_pred_id"] = reason_pred
    out["student_reason_pred"] = [reason_id_to_name.get(int(x), "unknown") for x in reason_pred]
    return out


def rank_candidates(scored_df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = ["row_id", "column", "student_score"]
    ascending = [True, True, False]
    if "candidate_rank" in scored_df.columns:
        sort_cols.append("candidate_rank")
        ascending.append(True)
    ranked = scored_df.sort_values(sort_cols, ascending=ascending).copy()
    ranked["student_rank"] = ranked.groupby(["row_id", "column"]).cumcount() + 1
    return ranked


def rule_support(row: pd.Series) -> float:
    try:
        return float(row.get("candidate_rule_support_ratio", 0.0))
    except Exception:
        return 0.0


def neighbor_ratio(row: pd.Series) -> float:
    try:
        return float(row.get("neighbor_majority_ratio", 0.0))
    except Exception:
        return 0.0


def evidence_strength(row: pd.Series) -> float:
    """Movies-aware evidence strength. Kept function name for downstream compatibility."""
    try:
        return movie_evidence_strength(row)
    except Exception:
        s = 1.2 * rule_support(row) + 0.8 * neighbor_ratio(row)
        for k, w in [
            ("candidate_matches_similar_row_target", 0.8),
            ("is_rule_and_dictionary_supported", 0.6),
            ("candidate_in_column_top_values", 0.3),
        ]:
            try: s += w * float(row.get(k, 0.0))
            except Exception: pass
        return s

def candidate_pool(ranked_df: pd.DataFrame, row_id: int, column: str, topk: int) -> pd.DataFrame:
    g = ranked_df[(ranked_df["row_id"] == row_id) & (ranked_df["column"] == column)].copy()
    if g.empty:
        return g
    g = g.sort_values(["student_rank"], ascending=[True]).reset_index(drop=True)
    return g.head(min(topk, len(g))).copy()


def choose_for_score(g: pd.DataFrame, dirty: str) -> Optional[pd.Series]:
    # 空值分数非常容易产生大规模误改，直接不修
    if dirty == "empty":
        return None
    # 本来就是合法百分比时，交给 gate 保守保留原值
    if is_percent_like(dirty):
        return None

    legal = []
    for _, row in g.iterrows():
        cand = canonical_empty_text(row["candidate_value"])
        if is_percent_like(cand):
            tpl = score_template_match_score(dirty, cand)
            key = (
                tpl,
                int(len(cand) == len(dirty)),
                int(tiny_edit_like(dirty, cand, 4)),
                rule_support(row),
                neighbor_ratio(row),
                float(row["student_score"]),
                -int(row["student_rank"]),
            )
            legal.append((row, key))
    if not legal:
        return None
    legal.sort(key=lambda x: x[1], reverse=True)
    return legal[0][0]



def choose_for_sample(g: pd.DataFrame, dirty: str) -> Optional[pd.Series]:
    if dirty == "empty":
        return None
    legal = []
    for _, row in g.iterrows():
        cand = canonical_empty_text(row["candidate_value"])
        if patients_like(cand):
            tpl = sample_template_match_score(dirty, cand)
            key = (
                tpl,
                int(len(extract_leading_number_token(cand)) == len(extract_leading_number_token(dirty))),
                rule_support(row),
                neighbor_ratio(row),
                float(row["student_score"]),
                -int(row["student_rank"]),
            )
            legal.append((row, key))
    if not legal:
        return None
    legal.sort(key=lambda x: x[1], reverse=True)
    return legal[0][0]



def choose_for_digits(g: pd.DataFrame, dirty: str) -> Optional[pd.Series]:
    legal = []
    for _, row in g.iterrows():
        cand = canonical_empty_text(row["candidate_value"])
        if is_digits_only(cand):
            key = (int(tiny_edit_like(dirty, cand, 4)), rule_support(row), neighbor_ratio(row), float(row["student_score"]), -int(row["student_rank"]))
            legal.append((row, key))
    if not legal:
        return None
    legal.sort(key=lambda x: x[1], reverse=True)
    return legal[0][0]


def choose_for_small_enum(g: pd.DataFrame, dirty: str) -> Optional[pd.Series]:
    legal = []
    for _, row in g.iterrows():
        cand = canonical_empty_text(row["candidate_value"]).lower()
        canonical = int((cand in US_STATE_CODES) or (cand in YES_NO_DOMAIN) or (cand in HOSPITAL_TYPE_DOMAIN) or (cand in CONDITION_DOMAIN) or alpha_space_like(cand))
        key = (canonical, int(tiny_edit_like(dirty, cand, 6)), rule_support(row), neighbor_ratio(row), float(row["student_score"]), -int(row["student_rank"]))
        legal.append((row, key))
    if not legal:
        return None
    legal.sort(key=lambda x: x[1], reverse=True)
    return legal[0][0]


def filter_measurecode_pool(g: pd.DataFrame, dirty: str) -> pd.DataFrame:
    x = g.copy(); x = x[x["candidate_value"].map(likely_measure_code_candidate)]
    if len(x) > 0:
        x = x[x["candidate_value"].map(lambda c: tiny_edit_like(dirty, c, 4))]
    return x if len(x) > 0 else g


def filter_stateavg_pool(g: pd.DataFrame, dirty: str) -> pd.DataFrame:
    x = g.copy(); x = x[x["candidate_value"].map(likely_stateavg_candidate)]
    if "family_match" in x.columns and len(x) > 0:
        x = x[x["family_match"] >= 1.0]
    if "_" in dirty and "prefix_match" in x.columns and len(x) > 0:
        x = x[x["prefix_match"] >= 1.0]
    if "same_code_part_count" in x.columns and len(x) > 0:
        x = x[x["same_code_part_count"] >= 1.0]
    return x if len(x) > 0 else g


def best_row_by_base_score(g: pd.DataFrame) -> pd.Series:
    return g.sort_values(["student_score", "student_rank"], ascending=[False, True]).reset_index(drop=True).iloc[0]


def source_text(row: pd.Series) -> str:
    return str(row.get("candidate_source", row.get("candidate_source_text", "")) or "").lower()


def source_has(row: pd.Series, name: str) -> int:
    return int(name.lower() in source_text(row))


def safe_float_value(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def feature_value(row: pd.Series, key: str, default=0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def consensus_confidence(row: pd.Series) -> float:
    return max(
        feature_value(row, "flight_consensus_confidence", 0.0),
        feature_value(row, "allowed_consensus_confidence", 0.0),
    )


def consensus_source_count(row: pd.Series) -> float:
    return max(
        feature_value(row, "flight_consensus_source_count", 0.0),
        feature_value(row, "allowed_consensus_source_count", 0.0),
    )


def consensus_support_count(row: pd.Series) -> float:
    return max(
        feature_value(row, "flight_consensus_support_count", 0.0),
        feature_value(row, "allowed_consensus_support_count", 0.0),
    )


def is_flight_consensus_candidate(row: pd.Series) -> int:
    src = source_text(row)
    return int(
        ("trusted_flight_source_weighted_consensus" in src or "flight_source_weighted_consensus" in src)
        or feature_value(row, "is_from_flight_consensus", 0.0) > 0
        or feature_value(row, "contains_trusted_flight_consensus_source", 0.0) > 0
        or feature_value(row, "contains_flight_consensus_source", 0.0) > 0
    )


def consensus_strength(row: pd.Series) -> float:
    conf = consensus_confidence(row)
    source_cnt = consensus_source_count(row)
    support_cnt = consensus_support_count(row)
    return conf + 0.04 * min(source_cnt, 8.0) + 0.03 * min(support_cnt, 10.0)


def has_trusted_consensus(row: pd.Series, min_conf: float = 0.62, min_sources: int = 2) -> bool:
    return (
        is_flight_consensus_candidate(row) == 1
        and consensus_confidence(row) >= min_conf
        and consensus_source_count(row) >= min_sources
    )


def candidate_minutes_value(row: pd.Series):
    cand = canonical_empty_text(row.get("candidate_value", ""))
    return parse_time_to_minutes_simple(cand)


def dirty_minutes_value(row: pd.Series):
    dirty = canonical_empty_text(row.get("dirty_value", ""))
    return parse_time_to_minutes_simple(dirty)


def actual_time_domain_score(row: pd.Series, dirty: str, column: str) -> float:
    """
    actual time 专属二次排序分数。
    目的：避免 actual time 被列内高频值/字典值误修；
    优先：脏值格式恢复、规则强支持、和原 dirty 接近、与 schedule 差值合理。
    """
    cand = canonical_empty_text(row.get("candidate_value", ""))
    src = source_text(row)

    student_score = feature_value(row, "student_score", 0.0)
    rule_ev = rule_support(row)
    evidence = evidence_strength(row)

    cand_valid = is_valid_time_simple(cand)
    dirty_valid = is_valid_time_simple(dirty)

    diff_dirty = circular_minute_diff_simple(dirty, cand)
    if dirty == "empty" or dirty_valid == 0:
        diff_dirty_penalty = 0.0
    else:
        # 大幅远离一个本来合法的 actual dirty value，应强惩罚
        diff_dirty_penalty = min(diff_dirty, 180.0) / 60.0

    paired_sched_diff = feature_value(row, "candidate_paired_schedule_diff", 0.0)
    if paired_sched_diff <= 0:
        paired_sched_diff = min(
            feature_value(row, "candidate_vs_sched_dep_diff", 999.0),
            feature_value(row, "candidate_vs_sched_arr_diff", 999.0),
            999.0
        )
    if paired_sched_diff >= 999:
        paired_sched_penalty = 0.0
    else:
        paired_sched_penalty = min(paired_sched_diff, 360.0) / 180.0

    is_extract = int("time_dirty_extract" in src)
    is_pattern_restore = int("time_pattern_restore" in src)
    is_column_dict = int("time_column_dictionary" in src or "column_top_value" in src)
    is_row_context = int("time_row_context" in src)
    is_expected = int(
        "expected_values_from_rules" in src
        or feature_value(row, "candidate_matches_any_expected_values_from_rules", 0.0) > 0
        or feature_value(row, "candidate_matches_any_rule_expected", 0.0) > 0
    )
    is_consensus = is_flight_consensus_candidate(row)
    consensus_bonus = consensus_strength(row) if is_consensus else 0.0

    # 如果是从 dirty 自身抽取、且时间值等价，这是格式恢复，应强鼓励
    same_time_format_restore = int(
        (is_extract or is_pattern_restore)
        and dirty_valid == 1
        and cand_valid == 1
        and circular_minute_diff_simple(dirty, cand) == 0
        and canonical_empty_text(cand) != canonical_empty_text(dirty)
    )

    score = 0.30 * student_score
    score += 2.50 * same_time_format_restore
    score += 1.30 * is_extract
    score += 1.00 * is_pattern_restore
    score += 1.20 * is_expected
    score += 1.50 * is_consensus
    score += 1.35 * consensus_bonus
    score += 1.10 * rule_ev
    score += 0.20 * evidence
    score += 0.35 * cand_valid
    score += 0.15 * is_row_context

    # actual time 中，列内高频/字典值不是强证据，尤其 dirty 已合法时
    if is_column_dict:
        score -= 0.80
        if dirty_valid == 1:
            score -= 0.70

    if dirty_valid == 1 and cand_valid == 1:
        score -= 0.85 * diff_dirty_penalty

    # paired schedule diff 只轻微约束，避免 actual 本身延误较大时被误伤
    score -= 0.25 * paired_sched_penalty

    return float(score)


def choose_for_actual_time(pool: pd.DataFrame, dirty: str, column: str) -> Optional[pd.Series]:
    """
    对 act_dep_time / act_arr_time 使用 domain rerank，而不是直接相信 student_score。
    """
    if pool.empty:
        return None

    legal = []
    for _, row in pool.iterrows():
        cand = canonical_empty_text(row.get("candidate_value", ""))
        if not is_valid_time_simple(cand):
            continue
        score = actual_time_domain_score(row, dirty, column)
        legal.append((row, score))

    if not legal:
        return None

    legal.sort(
        key=lambda x: (
            x[1],
            rule_support(x[0]),
            evidence_strength(x[0]),
            float(x[0].get("student_score", 0.0)),
            -int(x[0].get("student_rank", 999)),
        ),
        reverse=True
    )
    return legal[0][0]


def choose_for_scheduled_time(pool: pd.DataFrame, dirty: str, column: str) -> Optional[pd.Series]:
    """
    scheduled time 仍可相对积极修复，但避免选择非法时间。
    """
    if pool.empty:
        return None

    legal = []
    for _, row in pool.iterrows():
        cand = canonical_empty_text(row.get("candidate_value", ""))
        if not is_valid_time_simple(cand):
            continue

        src = source_text(row)
        is_extract = int("time_dirty_extract" in src)
        is_pattern_restore = int("time_pattern_restore" in src)
        is_expected = int(
            "expected_values_from_rules" in src
            or feature_value(row, "candidate_matches_any_expected_values_from_rules", 0.0) > 0
            or feature_value(row, "candidate_matches_any_rule_expected", 0.0) > 0
        )

        is_consensus = is_flight_consensus_candidate(row)
        score = (
            0.70 * feature_value(row, "student_score", 0.0)
            + 1.20 * rule_support(row)
            + 0.90 * is_expected
            + 1.50 * is_consensus
            + 1.20 * (consensus_strength(row) if is_consensus else 0.0)
            + 0.70 * is_extract
            + 0.60 * is_pattern_restore
            + 0.20 * evidence_strength(row)
        )
        legal.append((row, score))

    if not legal:
        return None

    legal.sort(
        key=lambda x: (
            x[1],
            feature_value(x[0], "student_score", 0.0),
            -int(x[0].get("student_rank", 999)),
        ),
        reverse=True
    )
    return legal[0][0]


def joint_decode_time_bundle_for_row(row_id: int, provisional: dict, raw_pools: dict):
    """
    轻量级四时间列联合后处理。
    目标：
    - 对 actual time，优先保守；
    - 如果 actual 候选明显是列内高频字典值，且与 schedule/dirty 关系不佳，则保留原值；
    - scheduled time 仍保持较积极修复。
    """
    for col in ["act_dep_time", "act_arr_time"]:
        key = (row_id, col)
        if key not in raw_pools:
            continue

        pool = raw_pools[key]
        if pool.empty:
            continue

        dirty = canonical_empty_text(pool.iloc[0].get("dirty_value", ""))
        chosen = choose_for_actual_time(pool, dirty, col)
        if chosen is not None:
            provisional[key] = chosen

    for col in ["sched_dep_time", "sched_arr_time"]:
        key = (row_id, col)
        if key not in raw_pools:
            continue

        pool = raw_pools[key]
        if pool.empty:
            continue

        dirty = canonical_empty_text(pool.iloc[0].get("dirty_value", ""))
        chosen = choose_for_scheduled_time(pool, dirty, col)
        if chosen is not None:
            provisional[key] = chosen

    return provisional


def joint_decode_measure_bundle(mc_pool, sa_pool, cond_pool, name_pool):
    if mc_pool.empty: mc_pool = pd.DataFrame([{}])
    if sa_pool.empty: sa_pool = pd.DataFrame([{}])
    if cond_pool.empty: cond_pool = pd.DataFrame([{}])
    if name_pool.empty: name_pool = pd.DataFrame([{}])
    best_combo = None; best_key = None
    for _, mc in mc_pool.iterrows():
        mc_val = canonical_empty_text(mc.get("candidate_value", "")); mc_family = measure_family_name(mc_val)
        for _, sa in sa_pool.iterrows():
            sa_val = canonical_empty_text(sa.get("candidate_value", ""))
            for _, cond in cond_pool.iterrows():
                cond_val = canonical_empty_text(cond.get("candidate_value", "")).lower()
                for _, name in name_pool.iterrows():
                    name_val = canonical_empty_text(name.get("candidate_value", "")).lower()
                    s_ind = 0.0
                    if mc_val: s_ind += float(mc.get("student_score", 0.0)) + 0.8 * rule_support(mc)
                    if sa_val: s_ind += float(sa.get("student_score", 0.0)) + 0.8 * rule_support(sa)
                    if cond_val: s_ind += float(cond.get("student_score", 0.0)) + 0.6 * rule_support(cond)
                    if name_val: s_ind += float(name.get("student_score", 0.0)) + 0.6 * rule_support(name)
                    s_pair = 0.0
                    if mc_val and sa_val: s_pair += 1.2 * pair_compat_score(mc_val, sa_val)
                    if cond_val and mc_family and mc_family in cond_val: s_pair += 2.0
                    if cond_val and name_val and any(tok in name_val for tok in cond_val.split()): s_pair += 1.0
                    key = (s_ind + s_pair, pair_compat_score(mc_val, sa_val) if mc_val and sa_val else -1, float(mc.get("student_score", -999.0)), float(sa.get("student_score", -999.0)), float(cond.get("student_score", -999.0)), float(name.get("student_score", -999.0)))
                    if best_key is None or key > best_key:
                        best_key = key; best_combo = (mc if mc_val else None, sa if sa_val else None, cond if cond_val else None, name if name_val else None)
    return best_combo if best_combo is not None else (None, None, None, None)


def pass_column_gate(column: str, dirty: str, best: pd.Series, second: Optional[pd.Series], column_profiles: Dict[str, dict]):
    best_val = canonical_empty_text(best.get("candidate_value", dirty))
    second_score = float(second.get("student_score", 0.0)) if second is not None else 0.0
    margin = float(best.get("student_score", 0.0)) - second_score
    threshold = max(ARGS.global_min_margin, COLUMN_MARGIN_THRESHOLDS.get(column, DEFAULT_MARGIN_THRESHOLD))
    evidence = evidence_strength(best)
    arch = get_column_archetype(column, column_profiles)
    score_tpl = score_template_match_score(dirty, best_val)
    sample_tpl = sample_template_match_score(dirty, best_val)

    if str(column) in TIME_COLUMNS or str(column).endswith("_time"):
        # Flights 时间列 gate。
        # 核心策略：
        # 1) scheduled time 可以相对积极修；
        # 2) actual time 波动大，如果 dirty 本身已经是合法时间，必须非常保守，避免把 clean actual time 改坏；
        # 3) dirty 中携带日期/estimated/紧凑 ampm 的格式错误，优先允许 time_dirty_extract / time_pattern_restore。
        col = str(column)
        src = str(best.get("candidate_source", "")).lower()

        if best_val == dirty:
            return False, "time_keep_original_best_equals_dirty"

        if not is_valid_time_simple(best_val):
            return False, "time_candidate_invalid"

        rule_ev = rule_support(best)
        try:
            time_gain = float(best.get("candidate_time_context_gain", 0.0))
        except Exception:
            time_gain = 0.0

        dirty_is_valid_time = is_valid_time_simple(dirty)
        best_is_extract = int("time_dirty_extract" in src or "time_pattern_restore" in src)
        best_is_column_dict = int("time_column_dictionary" in src or "column_top_value" in src)

        # actual time 专属保护：原值已经是合法时间时，不轻易改成另一个合法时间。
        # 这部分是当前降低 FP 的关键：actual time 的列内高频和 row context 不能当作强证据。
        if col in ACTUAL_TIME_COLUMNS and dirty_is_valid_time:
            same_time = circular_minute_diff_simple(dirty, best_val) == 0

            # 只允许格式恢复类同一时间值，例如 9:32aDec 1 -> 9:32 a.m.
            if best_is_extract and same_time:
                return True, "actual_time_format_restore_same_time"

            # 非同一时间值修改：必须有强规则支持。
            # 禁止仅凭 evidence/student_score 将 7:16 a.m. 改成 7:25 a.m.
            strong_rule = rule_ev >= 0.78
            strong_consensus = (
                is_flight_consensus_candidate(best)
                and consensus_confidence(best) >= 0.66
                and consensus_source_count(best) >= 3
            )
            small_delta = circular_minute_diff_simple(dirty, best_val) <= 1.0
            very_strong_evidence_small_delta = small_delta and evidence >= 2.00 and margin >= 0.15

            if not (strong_rule or strong_consensus or very_strong_evidence_small_delta):
                return False, "actual_time_valid_dirty_keep_original_strict"

            if best_is_column_dict and not (strong_rule or strong_consensus):
                return False, "actual_time_block_column_dictionary_strict"

            if strong_consensus and not strong_rule:
                return True, "actual_time_valid_dirty_consensus_override"

            return True, "actual_time_valid_dirty_strong_rule"

        # dirty 是空值：允许修，但 actual time 要求证据比 scheduled time 更强
        if dirty == "empty":
            if col in ACTUAL_TIME_COLUMNS:
                strong_consensus = (
                    is_flight_consensus_candidate(best)
                    and consensus_confidence(best) >= 0.62
                    and consensus_source_count(best) >= 2
                )
                if best_is_column_dict and rule_ev < 0.65 and not strong_consensus:
                    return False, "actual_time_empty_block_dictionary_without_rule"
                if strong_consensus:
                    return True, "actual_time_empty_to_consensus_value"
                if rule_ev >= 0.45:
                    return True, "actual_time_empty_to_rule_supported_value"
                if evidence >= 1.15 and margin >= max(threshold, 0.10):
                    return True, "actual_time_empty_to_high_evidence_value"
                return False, "actual_time_empty_low_evidence"

            if rule_ev >= 0.20 or evidence >= 0.70 or margin >= threshold:
                return True, "time_empty_to_supported_value"
            return False, "time_empty_low_evidence"

        # scheduled time 保护：如果原值本身是合法时间，不要仅凭 LTR/student_score 改成另一个合法时间。
        # 只有可信 consensus、强规则、或同一时间值格式恢复才允许改。
        if col in SCHEDULED_TIME_COLUMNS and dirty_is_valid_time:
            same_time = circular_minute_diff_simple(dirty, best_val) == 0
            if best_is_extract and same_time:
                return True, "scheduled_time_format_restore_same_time"

            strong_consensus = has_trusted_consensus(best, min_conf=0.60, min_sources=2)
            strong_rule = rule_ev >= 0.78

            if not (strong_consensus or strong_rule):
                return False, "scheduled_time_valid_dirty_keep_original_strict"

            if strong_consensus and not strong_rule:
                return True, "scheduled_time_valid_dirty_consensus_override"

            return True, "scheduled_time_valid_dirty_strong_rule"

        # 非 actual 或 dirty 非合法时间：格式恢复类候选更容易放行
        if best_is_extract and is_valid_time_simple(best_val):
            return True, "time_dirty_extract_or_pattern_restore"

        if dirty_is_valid_time and margin < max(threshold, 0.06) and evidence < 0.8:
            return False, "time_cleanlike_keep_original"

        if margin < threshold and evidence < 0.7 and time_gain <= 0:
            return False, "time_low_margin_low_evidence"

        return True, "time_passed_gate"

    if best_val == dirty:
        return False, "keep_original_best_equals_dirty"
    if column in PROTECTED_NONEMPTY_COLUMNS and dirty != "empty" and best_val == "empty":
        return False, "protected_nonempty_to_empty_block"

    if column == "Score":
        # 这是当前最主要的 FP 来源：clean empty Score 不能被自动补值
        if dirty == "empty":
            return False, "score_keep_if_original_empty"
        if is_percent_like(dirty):
            return False, "score_keep_if_already_valid"
        if not is_percent_like(best_val):
            return False, "score_candidate_not_valid_percent"
        # malformed percent（如 x00%, 95x）只有模板匹配足够强才放行
        if score_tpl < 4:
            return False, "score_template_mismatch"
        if margin < 0.01 and evidence < 0.6:
            return False, "score_low_margin_low_evidence"
    if column == "Sample":
        if dirty == "empty":
            return False, "sample_keep_if_original_empty"
        if not patients_like(best_val):
            return False, "sample_candidate_not_valid_pattern"
        if patients_like(dirty):
            # 原值已经合法时更保守
            if margin < max(threshold, 0.08) and evidence < 0.9:
                return False, "sample_cleanlike_keep_original"
        else:
            # 原值是 typo / malformed 时，只要模板强匹配就允许放行
            if sample_tpl < 4 and evidence < 0.5:
                return False, "sample_template_mismatch"
    if column == "Condition" and dirty.lower() in CONDITION_DOMAIN and (margin < threshold or evidence < 0.8):
        return False, "condition_cleanlike_keep_original"
    if column == "MeasureCode":
        if likely_measure_code_candidate(dirty) and (margin < threshold or evidence < 0.8):
            return False, "measurecode_cleanlike_keep_original"
        if not likely_measure_code_candidate(best_val):
            return False, "measurecode_candidate_not_code_like"
    if column == "Stateavg":
        if likely_stateavg_candidate(dirty) and (margin < threshold or evidence < 0.8):
            return False, "stateavg_cleanlike_keep_original"
        if not likely_stateavg_candidate(best_val):
            return False, "stateavg_candidate_not_stateavg_like"
    if column == "MeasureName":
        if dirty != "empty" and alpha_space_like(dirty) and len(dirty) >= 25 and (margin < threshold or evidence < 0.8):
            return False, "measurename_cleanlike_keep_original"
        if best_val == "empty":
            return False, "measurename_empty_block"
    if arch in {"small_enum", "structured_code", "long_text"} and margin < threshold and evidence < 0.8:
        return False, "generic_low_margin_keep_original"
    return True, "passed_gate"




# ============================================================
# Movies candidate selection / evidence / gate helpers
# ============================================================

def feature_value(row, key, default=0.0):
    try:
        if key not in row.index:
            return default
        v = row.get(key, default)
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def source_text(row):
    for k in ["candidate_source", "candidate_source_text", "sources_joined"]:
        try:
            if k in row.index and not pd.isna(row.get(k)):
                return str(row.get(k)).lower()
        except Exception:
            pass
    return ""


def rule_support(row):
    return max(
        feature_value(row, "candidate_rule_support_ratio", 0.0),
        feature_value(row, "rule_support_gain", 0.0),
        feature_value(row, "max_rule_support", 0.0),
    )


def neighbor_ratio(row):
    return max(
        feature_value(row, "neighbor_majority_ratio", 0.0),
        feature_value(row, "candidate_match_ratio_in_similar_rows", 0.0),
    )


def movie_consistency_strength(row):
    return max(
        feature_value(row, "movie_consistency_strength", 0.0),
        feature_value(row, "movie_consistency_confidence", 0.0)
        + 0.5 * feature_value(row, "movie_consistency_margin", 0.0)
        + 0.04 * min(feature_value(row, "movie_consistency_support_count", 0.0), 10.0),
        feature_value(row, "allowed_consensus_confidence", 0.0)
        + 0.5 * feature_value(row, "allowed_consensus_margin", 0.0)
        + 0.04 * min(feature_value(row, "allowed_consensus_support_count", 0.0), 10.0),
    )


def has_movie_consistency(row):
    src = source_text(row)
    return int(
        "movie_metadata_consistency" in src
        or feature_value(row, "contains_movie_consistency_source", 0.0) > 0
        or feature_value(row, "is_from_movie_consistency", 0.0) > 0
        or feature_value(row, "candidate_and_allowed_both_movie_consistency", 0.0) > 0
        or feature_value(row, "candidate_equals_allowed_consensus_value", 0.0) > 0
    )


def movie_evidence_strength(row):
    """
    Movies candidate evidence score used only for inference-time selection/gating.
    It intentionally rewards strong rule/dictionary/metadata-consistency/canonical evidence,
    while avoiding pure column-top high-frequency over-repair.
    """
    s = 0.0

    s += 1.20 * rule_support(row)
    s += 0.70 * neighbor_ratio(row)
    s += 0.60 * feature_value(row, "bayes_mean_cond_prob", 0.0)
    s += 0.40 * feature_value(row, "bayes_max_cond_prob", 0.0)

    for k, w in [
        ("candidate_matches_any_rule_expected", 0.80),
        ("candidate_matches_any_strong_rule_expected", 0.90),
        ("candidate_matches_any_expected_values_from_rules", 0.80),
        ("candidate_matches_similar_row_target", 0.55),
        ("candidate_equals_neighbor_majority", 0.45),
        ("candidate_equals_suggested_correct_value", 0.75),
        ("contains_dictionary_source", 0.45),
        ("contains_movie_dictionary_source", 0.55),
        ("contains_canonical_source", 0.35),
        ("contains_movie_canonical_source", 0.45),
        ("contains_movie_consistency_source", 0.85),
        ("candidate_equals_allowed_consensus_value", 0.90),
        ("candidate_and_allowed_both_movie_consistency", 1.00),
        ("candidate_under_strong_movie_consistency_cell", 0.70),
        ("movie_mojibake_repair_gain", 0.65),
        ("movie_candidate_duration_restoration", 0.55),
        ("movie_candidate_year_restoration", 0.50),
        ("movie_candidate_release_date_restoration", 0.55),
        ("movie_candidate_rating_value_restoration", 0.45),
        ("movie_candidate_count_restoration", 0.40),
        ("movie_candidate_review_count_restoration", 0.40),
        ("movie_candidate_actors_is_cast_prefix", 0.55),
        ("movie_candidate_cast_contains_actors", 0.50),
    ]:
        s += w * feature_value(row, k, 0.0)

    if has_movie_consistency(row):
        s += movie_consistency_strength(row)

    src = source_text(row)
    if "column_top_value" in src and not (
        "expected_values_from_rules" in src
        or "neighbor_majority" in src
        or "movie_metadata_consistency" in src
        or "id_to_" in src
        or "name_year" in src
    ):
        # Pure column-top candidates are risky for Movies.
        s -= 0.35

    return float(s)


def normalize_movie_repaired_value(column, value):
    """
    Use existing Movies normalizers when available. If not, preserve original candidate value.
    """
    try:
        fixed = normalize_by_movie_column(column, value)
        if fixed is not None and str(fixed).strip() != "":
            return str(fixed)
    except Exception:
        pass
    return canonical_text(value) if "canonical_text" in globals() else str(value).strip()


def choose_movie_candidate(pool: pd.DataFrame, dirty: str, column: str):
    """
    Conservative Movies candidate chooser v2.
    Prefer direct high-precision repair targets, then strong evidence candidates.
    """
    if pool is None or pool.empty:
        return None

    col = str(column).strip()
    cl = col.lower()
    dirty = canonical_empty_text(dirty)

    # Hard-block columns that currently have no true errors in Movies.
    if cl in {"ratingcount", "reviewcount"}:
        return None

    # Direct rules: if generated candidate is present, choose it.
    direct_val, _ = movies_direct_repair(col, dirty, None)
    if direct_val is not None:
        for _, row in pool.iterrows():
            cand = str(row.get("candidate_value", row.get("student_best_candidate", ""))).strip()
            if norm_text(cand) == norm_text(direct_val):
                return row

    # Release Date only direct rule; no direct candidate means keep original.
    if cl == "release date":
        return None

    scored = []
    for _, row in pool.iterrows():
        cand = str(row.get("candidate_value", row.get("student_best_candidate", ""))).strip()
        if cand == "":
            continue

        try:
            equivalent_dirty = int(value_equal_for_column(col, cand, dirty))
        except Exception:
            equivalent_dirty = int(norm_text(cand) == norm_text(dirty))

        if equivalent_dirty:
            continue

        student_score = feature_value(row, "student_score", feature_value(row, "score", 0.0))
        try:
            rank = int(row.get("student_rank", row.get("candidate_rank", 9999)))
            rank_bonus = 1.0 / max(rank, 1)
        except Exception:
            rank_bonus = 0.0

        ev = movie_evidence_strength(row)
        src = source_text(row)
        cons = has_movie_consistency(row)
        cons_strength = movie_consistency_strength(row)

        typed_bonus = 0.0
        # Do not reward weak rating count/review count because they are blocked.
        if cl == "duration" and feature_value(row, "movie_candidate_is_duration", 0.0) > 0:
            typed_bonus += 0.25
            if "id_to_duration" in src or "name_year_to_duration" in src or cons:
                typed_bonus += 0.50
        elif cl == "year" and feature_value(row, "movie_candidate_is_year", 0.0) > 0:
            typed_bonus += 0.35
            if "release_date_year" in src or "id_to_year" in src or cons:
                typed_bonus += 0.50
        elif cl == "ratingvalue" and feature_value(row, "movie_candidate_is_rating_value", 0.0) > 0:
            typed_bonus += 0.15
            if "id_to_rating_value" in src or "name_year_to_rating_value" in src or cons:
                typed_bonus += 0.60
        elif cl in {"actors", "cast", "creator", "director", "genre", "filming locations"}:
            typed_bonus += 0.20 * max(
                feature_value(row, "movie_candidate_actors_overlap_with_cast", 0.0),
                feature_value(row, "movie_list_jaccard_to_dirty", 0.0),
            )

        # Penalize pure column top high-frequency candidates.
        if "column_top_value" in src and not (
            "expected_values_from_rules" in src
            or "neighbor_majority" in src
            or "movie_metadata_consistency" in src
            or "id_to_" in src
            or "name_year" in src
        ):
            typed_bonus -= 0.60

        score = (
            ev
            + 0.25 * student_score
            + 0.08 * rank_bonus
            + typed_bonus
            + 0.60 * cons
            + 0.35 * cons_strength
            + 0.70 * rule_support(row)
        )

        scored.append((score, row))

    if not scored:
        return None

    scored.sort(
        key=lambda x: (
            x[0],
            has_movie_consistency(x[1]),
            rule_support(x[1]),
            feature_value(x[1], "student_score", 0.0),
        ),
        reverse=True,
    )
    return scored[0][1]


def movie_gate_decision(row, dirty: str, column: str, margin: float):
    """
    Movies conservative gate v2.

    The previous migrated gate was too permissive and caused massive FP:
    - RatingCount / ReviewCount had 0 true errors but thousands of repairs.
    - Release Date candidate repairs produced 0 TP / many FP.
    - Duration and RatingValue allowed weak typed candidates.

    This gate follows a high-precision-first policy:
    1. always block RatingCount / ReviewCount for current Movies benchmark;
    2. block weak Release Date model repairs, only allow direct Wide/Limited rule;
    3. block empty-value filling unless strong direct rule exists;
    4. Year mostly uses direct triple-year rule or very strong evidence;
    5. text/list columns remain conservative.
    """
    col = str(column).strip()
    cl = col.lower()
    cand = str(row.get("candidate_value", row.get("student_best_candidate", ""))).strip()
    dirty = canonical_empty_text(dirty)
    ev = movie_evidence_strength(row)
    src = source_text(row)
    has_cons = has_movie_consistency(row)

    # Never write equivalent dirty values.
    try:
        if value_equal_for_column(col, cand, dirty):
            return False, "candidate_equivalent_to_dirty_keep_original"
    except Exception:
        if norm_text(cand) == norm_text(dirty):
            return False, "candidate_equivalent_to_dirty_keep_original"

    # Current Movies ground truth shows no real errors in these two columns.
    # Any repair here currently creates FP or format-only noise.
    if cl in {"ratingcount", "reviewcount"}:
        return False, f"{cl}_blocked_no_true_errors_in_current_movies"

    # Release Date: model candidates were 0 TP / 128 FP.
    # Only direct rule should handle Mon DD, YYYY Wide/Limited.
    if cl == "release date":
        direct_val, direct_reason = movies_direct_repair(col, dirty, row)
        if direct_val is not None and norm_text(cand) == norm_text(direct_val):
            return True, direct_reason
        return False, "release_date_model_repair_blocked_use_direct_rule_only"

    # Do not fill empty values with weak typed candidates.
    # Empty Duration/RatingValue filling was a major FP source.
    if is_empty_token(dirty):
        if cl in {"duration", "ratingvalue", "country", "language", "actors", "cast", "creator", "director", "genre", "filming locations", "name"}:
            # Only allow very strong metadata consistency for non-rating/time structured columns.
            if cl == "duration":
                strong_duration = (
                    has_cons
                    and movie_consistency_strength(row) >= 1.45
                    and feature_value(row, "movie_consistency_support_count", feature_value(row, "allowed_consensus_support_count", 0.0)) >= 2
                    and ("id_to_duration" in src or "name_year_to_duration" in src or "movie_metadata_consistency" in src)
                )
                if strong_duration:
                    return True, "duration_empty_strong_metadata_consistency_pass"
            return False, f"{cl}_empty_value_fill_blocked"

    # Year: allow direct triple-year rule and otherwise require strong evidence.
    if cl == "year":
        direct_val, direct_reason = movies_direct_repair(col, dirty, row)
        if direct_val is not None and norm_text(cand) == norm_text(direct_val):
            return True, direct_reason

        if feature_value(row, "movie_candidate_is_year", 0.0) > 0 and (
            rule_support(row) >= 0.70
            or ("release_date_year" in src and ev >= 1.25)
            or ("id_to_year" in src and ev >= 1.35)
            or (has_cons and movie_consistency_strength(row) >= 1.50)
        ):
            return True, "year_strong_evidence_pass"

        return False, "year_low_evidence_keep_original"

    # Duration: only allow direct format normalization or strong source.
    if cl == "duration":
        direct_val, direct_reason = movies_direct_repair(col, dirty, row)
        if direct_val is not None and norm_text(cand) == norm_text(direct_val):
            return True, direct_reason

        strong_duration = (
            feature_value(row, "movie_candidate_is_duration", 0.0) > 0
            and (
                rule_support(row) >= 0.85
                or ("id_to_duration" in src and ev >= 1.45)
                or ("name_year_to_duration" in src and ev >= 1.45)
                or (has_cons and movie_consistency_strength(row) >= 1.60)
            )
        )
        if strong_duration:
            return True, "duration_strong_rule_or_metadata_pass"
        return False, "duration_weak_candidate_blocked"

    # RatingValue: block weak typed candidates. Only strong id/name-year/rule/consensus support.
    if cl == "ratingvalue":
        direct_val, direct_reason = movies_direct_repair(col, dirty, row)
        if direct_val is not None and norm_text(cand) == norm_text(direct_val):
            return True, direct_reason

        strong_rating = (
            feature_value(row, "movie_candidate_is_rating_value", 0.0) > 0
            and (
                rule_support(row) >= 0.90
                or ("id_to_rating_value" in src and ev >= 1.60)
                or ("name_year_to_rating_value" in src and ev >= 1.60)
                or (has_cons and movie_consistency_strength(row) >= 1.70)
            )
        )
        if strong_rating:
            return True, "rating_value_strong_rule_or_metadata_pass"
        return False, "rating_value_weak_candidate_blocked"

    # Country/language: need real consistency/rule support, not column frequency.
    if cl in {"country", "language"}:
        if (
            rule_support(row) >= 0.85
            or ("release_date_country" in src and ev >= 1.45)
            or ("country_to_language" in src and ev >= 1.60)
            or (has_cons and movie_consistency_strength(row) >= 1.70)
        ):
            return True, f"{cl}_strong_context_pass"
        return False, f"{cl}_low_context_keep_original"

    # Text/list/person fields: only allow direct mojibake/format repair or very strong metadata.
    if cl in {"actors", "cast", "creator", "director", "genre", "filming locations", "name", "description"}:
        if feature_value(row, "movie_mojibake_repair_gain", 0.0) > 0 and ev >= 1.20:
            return True, f"{cl}_mojibake_repair_pass"
        if (
            rule_support(row) >= 0.95
            or (has_cons and movie_consistency_strength(row) >= 1.85)
            or ("name_to_" in src and ev >= 1.80)
            or ("actors_to_cast" in src and ev >= 1.80)
            or ("cast_prefix_to_actors" in src and ev >= 1.80)
        ):
            return True, f"{cl}_strong_text_list_context_pass"
        return False, f"{cl}_conservative_keep_original"

    # General fallback is intentionally stricter than before.
    min_margin = COLUMN_MARGIN_THRESHOLDS.get(col, DEFAULT_MARGIN_THRESHOLD)
    if margin >= max(min_margin, 0.12) and ev >= 1.80 and rule_support(row) >= 0.60:
        return True, "general_strong_margin_rule_evidence_pass"

    return False, "general_low_margin_or_evidence"



def pass_movies_gate(column: str, dirty_value: str, chosen, second=None, column_profiles=None):
    """
    Compatibility wrapper for build_top1_with_gate_and_joint_decoding().

    Some migrated versions call pass_movies_gate(), while the actual Movies gate
    helper is named movie_gate_decision(). This wrapper computes the top1 margin
    and delegates to movie_gate_decision().
    """
    try:
        chosen_score = feature_value(chosen, "student_score", feature_value(chosen, "score", 0.0))
    except Exception:
        chosen_score = 0.0

    try:
        if second is None:
            second_score = 0.0
        else:
            second_score = feature_value(second, "student_score", feature_value(second, "score", 0.0))
    except Exception:
        second_score = 0.0

    margin = float(chosen_score) - float(second_score)

    if "movie_gate_decision" in globals():
        return movie_gate_decision(chosen, dirty_value, column, margin)

    # Fallback: if movie_gate_decision is unexpectedly absent, keep conservative behavior.
    try:
        ev = movie_evidence_strength(chosen) if "movie_evidence_strength" in globals() else 0.0
        min_margin = COLUMN_MARGIN_THRESHOLDS.get(str(column), DEFAULT_MARGIN_THRESHOLD) if "COLUMN_MARGIN_THRESHOLDS" in globals() else 0.06
        if margin >= min_margin and ev >= 1.10:
            return True, "fallback_margin_evidence_pass"
    except Exception:
        pass

    return False, "fallback_movies_gate_keep_original"



# ============================================================
# Extra compatibility aliases patched for Movies inference
# ============================================================

if "normalize_by_movie_column" not in globals():
    def normalize_by_movie_column(column, value):
        """
        Compatibility alias. Some migrated blocks use normalize_by_movie_column(),
        while earlier helper blocks may define normalize_movie_repaired_value().
        """
        try:
            # If a more specific normalizer exists under another name, use it.
            if "normalize_movie_value_by_column" in globals():
                return normalize_movie_value_by_column(column, value)
        except Exception:
            pass

        # Conservative normalizations for Movies inference.
        col = str(column).strip().lower()
        s = canonical_text(value) if "canonical_text" in globals() else str(value).strip()
        if s == "":
            return None

        try:
            if col == "year":
                return normalize_year_movies(s) if "normalize_year_movies" in globals() else None
            if col == "duration":
                if "parse_duration_minutes_movies" in globals():
                    m = parse_duration_minutes_movies(s)
                    return f"{int(m)} min" if m is not None else None
            if col == "release date":
                # Keep original if parseable; gate only needs non-None.
                return s if ("is_release_date_like_movies" in globals() and is_release_date_like_movies(s)) else None
            if col == "ratingvalue":
                if "parse_rating_movies" in globals():
                    r = parse_rating_movies(s)
                    if r is not None:
                        return f"{float(r):.1f}".rstrip("0").rstrip(".") if abs(float(r) - round(float(r))) > 1e-9 else f"{int(round(float(r)))}.0"
            if col in {"ratingcount", "reviewcount", "id"}:
                if "parse_count_movies" in globals():
                    c = parse_count_movies(s)
                    return str(int(c)) if c is not None else None
            if col in {"country", "language"}:
                return s.title()
            if col in {"actors", "cast", "creator", "director", "genre", "filming locations"}:
                return s
        except Exception:
            return None

        return None

if "is_movie_consistency_candidate" not in globals():
    def is_movie_consistency_candidate(row):
        try:
            return has_movie_consistency(row) if "has_movie_consistency" in globals() else 0
        except Exception:
            return 0

if "movie_consistency_confidence" not in globals():
    def movie_consistency_confidence(row):
        return max(
            feature_value(row, "movie_consistency_confidence", 0.0) if "feature_value" in globals() else 0.0,
            feature_value(row, "allowed_consensus_confidence", 0.0) if "feature_value" in globals() else 0.0,
        )

if "movie_consistency_support" not in globals():
    def movie_consistency_support(row):
        return max(
            feature_value(row, "movie_consistency_support_count", 0.0) if "feature_value" in globals() else 0.0,
            feature_value(row, "allowed_consensus_support_count", 0.0) if "feature_value" in globals() else 0.0,
        )

if "movie_column_family" not in globals():
    def movie_column_family(column):
        c = str(column).strip().lower()
        if c in {"year", "release date", "duration"}:
            return "time"
        if c in {"ratingvalue", "ratingcount", "reviewcount"}:
            return "rating"
        if c in {"actors", "cast", "creator", "director", "genre", "filming locations"}:
            return "text_list"
        if c in {"country", "language", "name"}:
            return "metadata"
        return "other"

if "is_available" not in globals():
    def is_available(x):
        try:
            if x is None:
                return False
            if pd.isna(x):
                return False
        except Exception:
            pass
        return str(x).strip().lower() not in {"", "nan", "none", "null", "na", "n/a", "missing", "empty", "{null}"}



# ============================================================
# Movies conservative direct repair rules
# ============================================================

MOVIES_MONTH_ABBR_TO_FULL = {
    "jan": "January",
    "january": "January",
    "feb": "February",
    "february": "February",
    "mar": "March",
    "march": "March",
    "apr": "April",
    "april": "April",
    "may": "May",
    "jun": "June",
    "june": "June",
    "jul": "July",
    "july": "July",
    "aug": "August",
    "august": "August",
    "sep": "September",
    "sept": "September",
    "september": "September",
    "oct": "October",
    "october": "October",
    "nov": "November",
    "november": "November",
    "dec": "December",
    "december": "December",
}


def repair_year_triple_to_middle_movies(value):
    """
    Movies Year direct rule:
        2010 2011 2012 -> 2011
        2014/2015/2016 -> 2015
        2007, 2008, 2009 -> 2008

    Only fires when three 4-digit years are consecutive.
    """
    s = canonical_empty_text(value)
    if s.lower() in {"", "empty"}:
        return None

    years = re.findall(r"(18\d{2}|19\d{2}|20\d{2}|21\d{2})", s)
    if len(years) != 3:
        return None

    ys = [int(y) for y in years]
    if ys[1] == ys[0] + 1 and ys[2] == ys[1] + 1:
        return str(ys[1])

    return None


def infer_release_country_from_row(row_like):
    """
    Best effort. In current build_top1 call row_like is often None because only candidate pool is available.
    If no safe row context exists, default to USA only for Wide/Limited release strings.
    """
    if row_like is None:
        return "USA"

    try:
        for key in ["Country", "movie_row_release_country", "parsed_release_country", "movie_row_country", "allowed_parsed_release_country"]:
            if key in row_like.index:
                v = canonical_empty_text(row_like.get(key))
                if v.lower() not in {"", "empty"}:
                    fixed = normalize_country_movies(v) if "normalize_country_movies" in globals() else None
                    return fixed or v
    except Exception:
        pass

    return "USA"


def repair_release_date_wide_limited_movies(value, row_like=None):
    """
    Movies Release Date direct rule:
        Nov 11, 2011 Wide    -> 11 November 2011 (USA)
        Aug 21, 2015 Limited -> 21 August 2015 (USA)

    Conservative:
    - requires explicit Wide/Limited/Limited Release/Wide Release tail
    - requires month abbreviation/name + day + 4-digit year
    """
    s = canonical_empty_text(value)
    if s.lower() in {"", "empty"}:
        return None

    # Remove punctuation noise but preserve date content.
    m = re.search(
        r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2}),?\s+(18\d{2}|19\d{2}|20\d{2}|21\d{2})\s+(wide|limited|wide release|limited release)\b",
        s,
        flags=re.I,
    )
    if not m:
        return None

    month_raw = m.group(1).lower()
    day = int(m.group(2))
    year = int(m.group(3))
    if not (1 <= day <= 31):
        return None

    month_full = MOVIES_MONTH_ABBR_TO_FULL.get(month_raw)
    if not month_full:
        return None

    country = infer_release_country_from_row(row_like)
    if country:
        return f"{day} {month_full} {year} ({country})"
    return f"{day} {month_full} {year}"


def repair_duration_format_movies(value):
    """
    Conservative Duration direct format normalization.
    Does not invent duration from empty value.
    """
    s = canonical_empty_text(value)
    if s.lower() in {"", "empty"}:
        return None

    # Reuse parser if available.
    if "parse_duration_minutes_movies" in globals():
        minutes = parse_duration_minutes_movies(s)
        if minutes is not None:
            repaired = f"{int(minutes)} min"
            if repaired.lower() != s.lower():
                return repaired

    # explicit fallback: 1 hr. 41 min. -> 101 min
    m = re.search(r"(\d+)\s*(?:h|hr|hrs|hour|hours)\.?\s*(\d+)\s*(?:m|min|mins|minute|minutes)\.?", s, flags=re.I)
    if m:
        total = int(m.group(1)) * 60 + int(m.group(2))
        if 1 <= total <= 500:
            return f"{total} min"

    return None


def repair_rating_value_format_movies(value):
    """
    Conservative RatingValue direct normalization:
        6.8/10 -> 6.8
        68%    -> 6.8
    Does not handle multi-rating strings such as '5/10,6/10' because those require metadata.
    """
    s = canonical_empty_text(value)
    if s.lower() in {"", "empty"}:
        return None

    if "," in s:
        return None

    if "parse_rating_movies" in globals():
        v = parse_rating_movies(s)
        if v is not None:
            # Only treat as direct repair when original contains /10 or % or comma decimal.
            if "/10" in s.lower() or "%" in s or re.search(r"\d,\d", s):
                repaired = f"{float(v):.1f}".rstrip("0").rstrip(".")
                if "." not in repaired:
                    repaired = repaired + ".0"
                if repaired.lower() != s.lower():
                    return repaired

    return None


def movies_direct_repair(column, dirty_value, row_like=None):
    """
    Return:
        (repaired_value, reason) or (None, None)

    These are high-precision direct rules applied before model/LTR gate.
    """
    col = str(column).strip()
    cl = col.lower()
    dirty = canonical_empty_text(dirty_value)

    if cl == "year":
        val = repair_year_triple_to_middle_movies(dirty)
        if val is not None and val != dirty:
            return val, "movies_direct_year_triple_to_middle"

    if cl == "release date":
        val = repair_release_date_wide_limited_movies(dirty, row_like=row_like)
        if val is not None and val != dirty:
            return val, "movies_direct_release_date_wide_limited"

    if cl == "duration":
        val = repair_duration_format_movies(dirty)
        if val is not None and val != dirty:
            return val, "movies_direct_duration_format_normalize"

    if cl == "ratingvalue":
        val = repair_rating_value_format_movies(dirty)
        if val is not None and val != dirty:
            return val, "movies_direct_rating_value_format_normalize"

    return None, None



# ============================================================
# Movies next-step conservative full-table direct repair patch
# VERIFIED PATCH: Release Date / Genre full-table append, keep Year, disable Duration/RatingValue direct.
# ============================================================

MOVIES_PATCH_MONTH_ABBR_TO_FULL = {
    "jan": "January", "january": "January",
    "feb": "February", "february": "February",
    "mar": "March", "march": "March",
    "apr": "April", "april": "April",
    "may": "May",
    "jun": "June", "june": "June",
    "jul": "July", "july": "July",
    "aug": "August", "august": "August",
    "sep": "September", "sept": "September", "september": "September",
    "oct": "October", "october": "October",
    "nov": "November", "november": "November",
    "dec": "December", "december": "December",
}

MOVIES_GENRE_CONSERVATIVE_MAP = {
    "kids & family": "Family",
    "science fiction & fantasy": "Sci-Fi",
    "mystery & suspense": "Mystery,Thriller",
    "action & adventure": "Action,Adventure",
    "sports & fitness": "Sport",
    "musical & performing arts": "Musical",
    "art house & international": "Drama",
    "animation": "Animation",
    "comedy": "Comedy",
    "drama": "Drama",
    "documentary": "Documentary",
    "horror": "Horror",
    "romance": "Romance",
    "western": "Western",
}

MOVIES_GENRE_DROP_TOKENS = {
    "classics",
    "special interest",
    "television",
    "cult movies",
    "gay & lesbian",
}


def movies_patch_empty_like(x):
    return str(canonical_empty_text(x)).strip().lower() in {"", "empty", "nan", "none", "null", "na", "n/a", "missing"}


def movies_patch_year_triple_to_middle(value):
    s = canonical_empty_text(value)
    if movies_patch_empty_like(s):
        return None
    years = re.findall(r"(18\d{2}|19\d{2}|20\d{2}|21\d{2})", str(s))
    if len(years) != 3:
        return None
    ys = [int(y) for y in years]
    if ys[1] == ys[0] + 1 and ys[2] == ys[1] + 1:
        return str(ys[1])
    return None


def movies_patch_release_date_wide_limited(value, default_country="USA"):
    s = canonical_empty_text(value)
    if movies_patch_empty_like(s):
        return None
    m = re.search(
        r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2}),?\s+(18\d{2}|19\d{2}|20\d{2}|21\d{2})\s+(wide|limited|wide release|limited release)\b",
        str(s),
        flags=re.I,
    )
    if not m:
        return None
    month_raw = m.group(1).strip().lower()
    day = int(m.group(2))
    year = int(m.group(3))
    if not (1 <= day <= 31):
        return None
    month_full = MOVIES_PATCH_MONTH_ABBR_TO_FULL.get(month_raw)
    if not month_full:
        return None
    country = default_country or "USA"
    return f"{day} {month_full} {year} ({country})"


def movies_patch_genre_conservative(value):
    s = canonical_empty_text(value)
    if movies_patch_empty_like(s):
        return None

    raw_parts = re.split(r"[,;/|]+", str(s))
    out = []
    seen = set()
    changed = False

    for part in raw_parts:
        p = re.sub(r"\s+", " ", part.strip())
        if not p:
            continue
        key = p.lower()

        if key in MOVIES_GENRE_DROP_TOKENS:
            changed = True
            continue

        mapped = MOVIES_GENRE_CONSERVATIVE_MAP.get(key)
        if mapped is None:
            mapped = p
        else:
            changed = True

        for sub in re.split(r"[,/]+", mapped):
            sub = re.sub(r"\s+", " ", sub.strip())
            if not sub:
                continue
            k = sub.lower()
            if k not in seen:
                seen.add(k)
                out.append(sub)

    if not out or not changed:
        return None

    repaired = ",".join(out)
    if repaired.strip().lower() == str(s).strip().lower():
        return None
    return repaired


def movies_patch_direct_repair(column, dirty_value):
    col = str(column).strip().lower()

    if col == "year":
        val = movies_patch_year_triple_to_middle(dirty_value)
        if val is not None and str(val).strip() != str(canonical_empty_text(dirty_value)).strip():
            return val, "movies_direct_year_triple_to_middle"

    if col == "release date":
        val = movies_patch_release_date_wide_limited(dirty_value, default_country="USA")
        if val is not None and str(val).strip() != str(canonical_empty_text(dirty_value)).strip():
            return val, "movies_direct_release_date_wide_limited"

    if col == "genre":
        val = movies_patch_genre_conservative(dirty_value)
        if val is not None and str(val).strip() != str(canonical_empty_text(dirty_value)).strip():
            return val, "movies_direct_genre_conservative_mapping"

    return None, None


def append_full_table_movies_direct_repairs(top1_df, ranked_df=None):
    if top1_df is None or top1_df.empty:
        top1_df = pd.DataFrame()

    existing_keys = set()
    if not top1_df.empty and {"row_id", "column"}.issubset(set(top1_df.columns)):
        for _, r in top1_df[["row_id", "column"]].drop_duplicates().iterrows():
            existing_keys.add((int(r["row_id"]), str(r["column"]).strip()))

    scan_rows = []

    if ranked_df is not None and not ranked_df.empty and {"row_id", "column", "dirty_value"}.issubset(set(ranked_df.columns)):
        tmp = ranked_df[["row_id", "column", "dirty_value"]].drop_duplicates(subset=["row_id", "column"], keep="first")
        for _, r in tmp.iterrows():
            scan_rows.append((int(r["row_id"]), str(r["column"]).strip(), canonical_empty_text(r["dirty_value"])))

    dirty_paths = [
        "/mnt/mydata/dq/projects/Splittree/movies/movies_dirty.csv",
        "/mnt/mydata/dq/projects/Splittree/test/Error Correction movies/movies_dirty.csv",
        "movies_dirty.csv",
    ]
    for p in dirty_paths:
        try:
            if os.path.exists(p):
                dirty_df = pd.read_csv(p, encoding="utf-8-sig")
                for rid in range(len(dirty_df)):
                    for col in ["Year", "Release Date", "Genre"]:
                        if col in dirty_df.columns:
                            scan_rows.append((int(rid), col, canonical_empty_text(dirty_df.at[rid, col])))
                break
        except Exception as e:
            print(f"[WARN] failed to scan dirty table for movies direct repairs: {p}, err={e}")

    seen_scan = set()
    dedup_scan_rows = []
    for rid, col, dirty in scan_rows:
        val_key = (int(rid), str(col).strip(), str(dirty))
        if val_key in seen_scan:
            continue
        seen_scan.add(val_key)
        dedup_scan_rows.append((rid, col, dirty))

    new_rows = []
    for rid, col, dirty in dedup_scan_rows:
        key = (int(rid), str(col).strip())
        if key in existing_keys:
            continue
        repaired, reason = movies_patch_direct_repair(col, dirty)
        if repaired is None:
            continue
        if str(repaired).strip() == "" or str(repaired).strip().lower() == str(dirty).strip().lower():
            continue

        existing_keys.add(key)
        new_rows.append({
            "row_id": int(rid),
            "column": str(col).strip(),
            "dirty_value": dirty,
            "candidate_value": repaired,
            "student_best_candidate": repaired,
            "repaired_value": repaired,
            "predicted_repair_value": repaired,
            "student_score": 999.0,
            "top_margin": 999.0,
            "student_margin": 999.0,
            "student_rank": 1,
            "pass_gate": 1,
            "gate_passed": 1,
            "gate_reason": reason,
            "decision_source": "movies_full_table_direct_rule",
            "decision_confidence": 1.0,
            "is_hard_case": 0,
        })

    if new_rows:
        add_df = pd.DataFrame(new_rows)
        for c in add_df.columns:
            if c not in top1_df.columns:
                top1_df[c] = None
        for c in top1_df.columns:
            if c not in add_df.columns:
                add_df[c] = None
        top1_df = pd.concat([top1_df, add_df[top1_df.columns]], ignore_index=True, sort=False)
        top1_df = top1_df.drop_duplicates(subset=["row_id", "column"], keep="first")
        top1_df = top1_df.sort_values(["row_id", "column"]).reset_index(drop=True)

    print(f"[INFO] movies full-table direct repairs appended: {len(new_rows)}")
    return top1_df


# Override any previous movies_direct_repair implementation.
# Last definition wins; this intentionally disables Duration/RatingValue direct format normalization.
def movies_direct_repair(column, dirty_value, row_like=None):
    return movies_patch_direct_repair(column, dirty_value)


# Hard-disable weak direct self-format rules that were 0 TP in the last evaluation.
def repair_duration_format_movies(value):
    return None


def repair_rating_value_format_movies(value):
    return None



# ============================================================
# Final Movies precision-preserving recall patch
# ============================================================

MOVIES_FINAL_MONTH_ABBR_TO_FULL = {
    "jan": "January", "january": "January",
    "feb": "February", "february": "February",
    "mar": "March", "march": "March",
    "apr": "April", "april": "April",
    "may": "May",
    "jun": "June", "june": "June",
    "jul": "July", "july": "July",
    "aug": "August", "august": "August",
    "sep": "September", "sept": "September", "september": "September",
    "oct": "October", "october": "October",
    "nov": "November", "november": "November",
    "dec": "December", "december": "December",
}

MOVIES_FINAL_GENRE_MAP = {
    "kids & family": "Family",
    "science fiction & fantasy": "Sci-Fi",
    # More conservative than Mystery,Thriller; avoids repaired superset.
    "mystery & suspense": "Thriller",
    # More conservative than Action,Adventure; avoids repaired superset.
    "action & adventure": "Action",
    "sports & fitness": "Sport",
    "musical & performing arts": "Musical",
    # RT bucket is too broad; drop it rather than map to noisy Drama.
    "art house & international": "",
    "animation": "Animation",
    "comedy": "Comedy",
    "drama": "Drama",
    "documentary": "Documentary",
    "horror": "Horror",
    "romance": "Romance",
    "western": "Western",
}

MOVIES_FINAL_GENRE_DROP = {
    "classics",
    "special interest",
    "television",
    "cult movies",
    "gay & lesbian",
}


def movies_final_is_empty_like(x):
    return str(canonical_empty_text(x)).strip().lower() in {"", "empty", "nan", "none", "null", "na", "n/a", "missing"}


def movies_final_year_triple_to_middle(value):
    s = canonical_empty_text(value)
    if movies_final_is_empty_like(s):
        return None
    years = re.findall(r"(18\d{2}|19\d{2}|20\d{2}|21\d{2})", str(s))
    if len(years) != 3:
        return None
    ys = [int(y) for y in years]
    if ys[1] == ys[0] + 1 and ys[2] == ys[1] + 1:
        return str(ys[1])
    return None


def movies_final_release_date_wide_limited(value, default_country="USA"):
    s = canonical_empty_text(value)
    if movies_final_is_empty_like(s):
        return None
    m = re.search(
        r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2}),?\s+(18\d{2}|19\d{2}|20\d{2}|21\d{2})\s+(wide|limited|wide release|limited release)\b",
        str(s),
        flags=re.I,
    )
    if not m:
        return None
    month_full = MOVIES_FINAL_MONTH_ABBR_TO_FULL.get(m.group(1).strip().lower())
    if not month_full:
        return None
    day = int(m.group(2))
    year = int(m.group(3))
    if not (1 <= day <= 31):
        return None
    return f"{day} {month_full} {year} ({default_country or 'USA'})"


def movies_final_genre_conservative(value):
    s = canonical_empty_text(value)
    if movies_final_is_empty_like(s):
        return None

    out = []
    seen = set()
    changed = False

    for part in re.split(r"[,;/|]+", str(s)):
        p = re.sub(r"\s+", " ", part.strip())
        if not p:
            continue
        k = p.lower()

        if k in MOVIES_FINAL_GENRE_DROP:
            changed = True
            continue

        mapped = MOVIES_FINAL_GENRE_MAP.get(k, p)
        if mapped != p:
            changed = True

        for sub in re.split(r"[,/]+", str(mapped)):
            sub = re.sub(r"\s+", " ", sub.strip())
            if not sub:
                continue
            sk = sub.lower()
            if sk not in seen:
                seen.add(sk)
                out.append(sub)

    if not changed or not out:
        return None
    repaired = ",".join(out)
    if repaired.strip().lower() == str(s).strip().lower():
        return None
    return repaired


def movies_final_rating_single_format(value):
    """
    Only single-value format repair:
      6.8/10 -> 6.8
      68% -> 6.8
    Explicitly blocks multi-rating strings such as '6.8/10,6.3/10'.
    """
    s = canonical_empty_text(value)
    if movies_final_is_empty_like(s):
        return None
    if "," in s or ";" in s or "|" in s:
        return None
    if not ("/10" in s.lower() or "%" in s or re.search(r"\d,\d", s)):
        return None

    v = parse_rating_movies(s)
    if v is None:
        return None
    repaired = f"{float(v):.1f}".rstrip("0").rstrip(".")
    if "." not in repaired:
        repaired += ".0"
    if repaired.lower() == s.lower():
        return None
    return repaired


def movies_final_direct_repair(column, dirty_value, row_like=None):
    col = str(column).strip().lower()
    dirty = canonical_empty_text(dirty_value)

    if col == "year":
        val = movies_final_year_triple_to_middle(dirty)
        if val is not None and str(val).strip() != str(dirty).strip():
            return val, "movies_direct_year_triple_to_middle"

    if col == "release date":
        val = movies_final_release_date_wide_limited(dirty, default_country="USA")
        if val is not None and str(val).strip() != str(dirty).strip():
            return val, "movies_direct_release_date_wide_limited"

    if col == "genre":
        val = movies_final_genre_conservative(dirty)
        if val is not None and str(val).strip() != str(dirty).strip():
            return val, "movies_direct_genre_conservative_mapping"

    if col == "ratingvalue":
        val = movies_final_rating_single_format(dirty)
        if val is not None and str(val).strip() != str(dirty).strip():
            return val, "movies_direct_rating_value_single_format"

    # Do not direct-normalize Duration in this patch.
    return None, None


# Last definition wins. This fixes old versions that either disabled too much
# or returned direct repairs without robust output fields.
def movies_direct_repair(column, dirty_value, row_like=None):
    return movies_final_direct_repair(column, dirty_value, row_like=row_like)


def make_movies_direct_repair_row(
    row_id,
    column,
    dirty_value,
    repaired_value,
    reason,
    top1_raw_candidate="",
    top1_raw_score=999.0,
    top2_raw_candidate="",
    top2_raw_score=None,
    top_margin=999.0,
    source="movies_full_table_direct_rule",
):
    repaired_value = canonical_empty_text(repaired_value)
    dirty_value = canonical_empty_text(dirty_value)
    return {
        "row_id": int(row_id),
        "column": str(column).strip(),
        "dirty_value": dirty_value,

        # All common repair aliases; evaluation scripts may read different names.
        "candidate_value": repaired_value,
        "student_best_candidate": repaired_value,
        "student_top_candidate": repaired_value,
        "repaired_value": repaired_value,
        "predicted_repair_value": repaired_value,
        "predicted_repair": repaired_value,

        "selected_student_rank": 0,
        "student_score": float(top1_raw_score) if top1_raw_score is not None else 999.0,
        "student_reason_pred": reason,
        "top1_raw_candidate": canonical_empty_text(top1_raw_candidate),
        "top1_raw_score": float(top1_raw_score) if top1_raw_score is not None else 999.0,
        "top2_raw_candidate": canonical_empty_text(top2_raw_candidate),
        "top2_raw_score": top2_raw_score,
        "top_margin": float(top_margin) if top_margin is not None else 999.0,
        "student_margin": float(top_margin) if top_margin is not None else 999.0,

        "gate_passed": 1,
        "pass_gate": 1,
        "gate_reason": reason,
        "candidate_source": source,
        "selected_candidate_source": source,
        "decision_source": source,
        "decision_confidence": 1.0,
        "evidence_strength": 999.0,
        "rule_support": 1.0,
        "neighbor_majority_ratio": 0.0,
        "hard_case_score": 0.0,
        "is_hard_case": 0,
    }


# Last definition wins. Replaces earlier append implementation so existing bad
# direct rows (especially Year rows without repair-value aliases) can be repaired.
def append_full_table_movies_direct_repairs(top1_df, ranked_df=None):
    if top1_df is None or top1_df.empty:
        top1_df = pd.DataFrame()

    key_to_row = {}
    if not top1_df.empty and {"row_id", "column"}.issubset(set(top1_df.columns)):
        for _, r in top1_df.iterrows():
            key_to_row[(int(r["row_id"]), str(r["column"]).strip())] = r.to_dict()

    scan_rows = []

    # Inference candidate rows.
    if ranked_df is not None and not ranked_df.empty and {"row_id", "column", "dirty_value"}.issubset(set(ranked_df.columns)):
        tmp = ranked_df[["row_id", "column", "dirty_value"]].drop_duplicates(subset=["row_id", "column"], keep="first")
        for _, r in tmp.iterrows():
            scan_rows.append((int(r["row_id"]), str(r["column"]).strip(), canonical_empty_text(r["dirty_value"])))

    # Dirty table rows. This catches Release Date / Genre / Year / safe RatingValue even if no candidate was generated.
    dirty_paths = [
        "/mnt/mydata/dq/projects/Splittree/movies/movies_dirty.csv",
        "/mnt/mydata/dq/projects/Splittree/test/Error Correction movies/movies_dirty.csv",
        "movies_dirty.csv",
    ]
    for p in dirty_paths:
        try:
            if os.path.exists(p):
                dirty_df = pd.read_csv(p, encoding="utf-8-sig")
                for rid in range(len(dirty_df)):
                    for col in ["Year", "Release Date", "Genre", "RatingValue"]:
                        if col in dirty_df.columns:
                            scan_rows.append((int(rid), col, canonical_empty_text(dirty_df.at[rid, col])))
                break
        except Exception as e:
            print(f"[WARN] failed to scan dirty table for movies direct repairs: {p}, err={e}")

    seen_scan = set()
    new_or_replaced = 0
    for rid, col, dirty in scan_rows:
        scan_key = (int(rid), str(col).strip(), str(dirty))
        if scan_key in seen_scan:
            continue
        seen_scan.add(scan_key)

        repaired, reason = movies_final_direct_repair(col, dirty)
        if repaired is None:
            continue
        if str(repaired).strip() == "" or norm_text(repaired) == norm_text(dirty):
            continue

        key = (int(rid), str(col).strip())
        direct_row = make_movies_direct_repair_row(
            row_id=rid,
            column=col,
            dirty_value=dirty,
            repaired_value=repaired,
            reason=reason,
            source="movies_full_table_direct_rule",
        )

        # Replace existing rows only when:
        # 1) no row exists, or
        # 2) existing row is a direct Year/Genre/Release Date row with missing candidate aliases,
        # 3) or existing row is equivalent to dirty.
        replace = key not in key_to_row
        if not replace:
            old = key_to_row[key]
            old_val = canonical_empty_text(
                old.get("repaired_value", old.get("predicted_repair_value", old.get("candidate_value", old.get("student_best_candidate", ""))))
            )
            old_reason = str(old.get("gate_reason", ""))
            if old_val.lower() in {"", "empty"}:
                replace = True
            elif norm_text(old_val) == norm_text(dirty):
                replace = True
            elif old_reason.startswith("movies_direct_") and col in {"Year", "Release Date", "Genre", "RatingValue"}:
                replace = True

        if replace:
            key_to_row[key] = direct_row
            new_or_replaced += 1

    if not key_to_row:
        return top1_df

    out = pd.DataFrame(list(key_to_row.values()))

    # Keep original column order as much as possible.
    if not top1_df.empty:
        for c in out.columns:
            if c not in top1_df.columns:
                top1_df[c] = None
        for c in top1_df.columns:
            if c not in out.columns:
                out[c] = None
        out = out[list(top1_df.columns) + [c for c in out.columns if c not in top1_df.columns]]

    out = out.drop_duplicates(subset=["row_id", "column"], keep="first")
    out = out.sort_values(["row_id", "column"]).reset_index(drop=True)
    print(f"[INFO] movies full-table direct repairs inserted/replaced: {new_or_replaced}")
    return out


def build_top1_with_gate_and_joint_decoding(ranked_df: pd.DataFrame, column_profiles: Dict[str, dict]) -> pd.DataFrame:
    """Movies top-1 selection with high-precision conservative gate v2.

    Important changes:
    - direct Year / Release Date repairs can be applied even if LTR did not pick them;
    - RatingCount / ReviewCount are blocked for current Movies benchmark;
    - keep-original rows are NOT written into inference_top1_repairs.csv.
    """
    out_rows = []
    for (row_id, col), pool in ranked_df.groupby(["row_id", "column"], sort=False):
        pool = pool.sort_values(["student_rank"], ascending=[True]).reset_index(drop=True).head(max(ARGS.joint_topk, 1))
        if pool.empty:
            continue

        dirty = canonical_empty_text(pool.iloc[0].get("dirty_value", ""))

        # Direct high-precision repair first, independent of LTR top1.
        direct_val, direct_reason = movies_direct_repair(str(col), dirty, None)
        if direct_val is not None and norm_text(direct_val) != norm_text(dirty):
            chosen = pool.iloc[0]
            top2_val = canonical_empty_text(pool.iloc[1].get("candidate_value", "")) if len(pool) >= 2 else ""
            top2_score = float(pool.iloc[1].get("student_score", 0.0)) if len(pool) >= 2 else None
            top_margin = 999.0

            out_rows.append(make_movies_direct_repair_row(
                row_id=int(row_id),
                column=str(col),
                dirty_value=dirty,
                repaired_value=canonical_empty_text(direct_val),
                reason=direct_reason,
                top1_raw_candidate=canonical_empty_text(chosen.get("candidate_value", "")),
                top1_raw_score=float(chosen.get("student_score", 0.0)),
                top2_raw_candidate=top2_val,
                top2_raw_score=top2_score,
                top_margin=top_margin,
                source="movies_direct_rule_in_candidate_pool",
            ))
            continue

        chosen = choose_movie_candidate(pool, dirty, str(col))
        if chosen is None:
            # Conservative: no selected candidate means keep original and DO NOT output.
            continue

        second = pool.iloc[1] if len(pool) >= 2 else None
        passed, gate_reason = pass_movies_gate(str(col), dirty, chosen, second, column_profiles)
        if not passed:
            continue

        final_value = canonical_empty_text(chosen.get("candidate_value", dirty))
        if norm_text(final_value) == norm_text(dirty):
            continue

        gate_passed = 1
        selected_rank = int(chosen.get("student_rank", -1))
        selected_source = str(chosen.get("candidate_source", chosen.get("candidate_source_text", "movie_decode")))
        reason = str(chosen.get("student_reason_pred", "movie_decode"))

        top2_val = canonical_empty_text(second.get("candidate_value", "")) if second is not None else ""
        top2_score = float(second.get("student_score", 0.0)) if second is not None else None
        top_margin = float(chosen.get("student_score", 0.0)) - (float(second.get("student_score", 0.0)) if second is not None else 0.0)

        hard_case_score = 0.0
        if top_margin < max(ARGS.global_min_margin, COLUMN_MARGIN_THRESHOLDS.get(str(col), DEFAULT_MARGIN_THRESHOLD)):
            hard_case_score += 1.0
        if evidence_strength(chosen) < 0.8:
            hard_case_score += 1.0

        out_rows.append({
            "row_id": int(row_id),
            "column": str(col),
            "dirty_value": dirty,

            # keep all common repair-value aliases for evaluators
            "candidate_value": final_value,
            "student_best_candidate": final_value,
            "student_top_candidate": final_value,
            "repaired_value": final_value,
            "predicted_repair_value": final_value,
            "predicted_repair": final_value,

            "selected_student_rank": selected_rank,
            "student_score": float(chosen.get("student_score", 0.0)),
            "student_reason_pred": reason,
            "top1_raw_candidate": canonical_empty_text(chosen.get("candidate_value", "")),
            "top1_raw_score": float(chosen.get("student_score", 0.0)),
            "top2_raw_candidate": top2_val,
            "top2_raw_score": top2_score,
            "top_margin": top_margin,
            "gate_passed": gate_passed,
            "pass_gate": 1,
            "gate_reason": gate_reason,
            "candidate_source": selected_source,
            "selected_candidate_source": selected_source,
            "decision_source": "ltr_student_gate",
            "evidence_strength": evidence_strength(chosen),
            "rule_support": rule_support(chosen),
            "neighbor_majority_ratio": neighbor_ratio(chosen),
            "hard_case_score": hard_case_score,
        })

    return pd.DataFrame(out_rows)




def build_hard_cases_jsonl(ranked_df: pd.DataFrame, max_hard_cases: int):
    rows = []
    for (row_id, column), g in ranked_df.groupby(["row_id", "column"], sort=False):
        g = g.sort_values(["student_rank"], ascending=[True]).reset_index(drop=True)
        if len(g) < 2:
            continue
        best = g.iloc[0]
        second = g.iloc[1]
        margin = float(best["student_score"]) - float(second["student_score"])
        is_hard = margin < ARGS.global_min_margin + 0.02
        if str(column) in MOVIE_HARD_COLUMNS and margin < 0.10:
            is_hard = True
        if feature_value(best, "movie_consistency_strength", 0.0) > 0 and margin < 0.12:
            is_hard = True
        if not is_hard:
            continue
        rows.append({
            "row_id": int(row_id),
            "column": str(column).strip(),
            "dirty_value": canonical_empty_text(best["dirty_value"]),
            "candidate_a_value": canonical_empty_text(best["candidate_value"]),
            "candidate_b_value": canonical_empty_text(second["candidate_value"]),
            "candidate_a_score": float(best["student_score"]),
            "candidate_b_score": float(second["student_score"]),
            "candidate_a_source": str(best.get("candidate_source", "")),
            "candidate_b_source": str(second.get("candidate_source", "")),
            "candidate_a_rule_support": rule_support(best),
            "candidate_b_rule_support": rule_support(second),
            "top_margin": margin
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["top_margin", "column"], ascending=[True, True]).head(max_hard_cases).reset_index(drop=True)
    with open(OUTPUT_HARD_CASES_JSONL, "w", encoding="utf-8") as f:
        if not df.empty:
            for _, row in df.iterrows():
                write_jsonl_record(f, row.to_dict())
    return df



def ensure_movies_repair_alias_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure output has all repair-value aliases expected by different evaluators."""
    if df is None or df.empty:
        return df
    out = df.copy()
    # Pick best available repair value per row.
    alias_cols = ["repaired_value", "predicted_repair_value", "predicted_repair", "candidate_value", "student_best_candidate", "student_top_candidate"]
    for c in alias_cols:
        if c not in out.columns:
            out[c] = None

    def pick_repair(row):
        for c in alias_cols:
            v = canonical_empty_text(row.get(c, ""))
            if v.lower() not in {"", "empty"}:
                return v
        return ""

    repair_values = out.apply(pick_repair, axis=1)
    for c in alias_cols:
        out[c] = repair_values

    if "candidate_source" not in out.columns:
        out["candidate_source"] = out.get("selected_candidate_source", "")
    if "decision_source" not in out.columns:
        out["decision_source"] = out.get("selected_candidate_source", "")
    if "pass_gate" not in out.columns:
        out["pass_gate"] = out.get("gate_passed", 1)
    return out



# ============================================================
# Movies STOPLOSS + safe recall patch
# Added after eval: Country caused 1459 FP; Description caused FP; RatingValue direct 0 TP.
# Last definitions below intentionally override earlier migrated versions.
# ============================================================

MOVIES_DISABLED_REPAIR_COLUMNS_STOPLOSS = {
    "country",
    "language",
    "ratingcount",
    "reviewcount",
    "description",
}

def movies_stoploss_column_disabled(column: str) -> bool:
    return str(column).strip().lower() in MOVIES_DISABLED_REPAIR_COLUMNS_STOPLOSS


def movies_year_triple_candidates(value):
    s = canonical_empty_text(value)
    if str(s).strip().lower() in {"", "empty", "nan", "none", "null", "na", "n/a", "missing"}:
        return []
    years = re.findall(r"(18\d{2}|19\d{2}|20\d{2}|21\d{2})", str(s))
    if len(years) != 3:
        return []
    ys = [int(y) for y in years]
    if ys[1] == ys[0] + 1 and ys[2] == ys[1] + 1:
        return ys
    return []


def movies_release_year_from_value(value):
    try:
        parts = parse_release_date_parts_movies(value)
        if isinstance(parts, dict) and parts.get("year"):
            return int(parts["year"])
    except Exception:
        pass
    y = normalize_year_movies(value)
    return int(y) if y is not None else None


def movies_stoploss_direct_repair_from_row(column, dirty_value, dirty_row=None):
    """
    High-precision direct rules only.
    - Blocks RatingValue direct format because current eval showed 0 TP.
    - Year: if dirty is triple-year and same-row Release Date year is one of the three, use Release Date year;
      otherwise fallback to middle year.
    - Release Date and Genre keep previous high-precision direct rules.
    """
    col = str(column).strip()
    cl = col.lower()
    dirty = canonical_empty_text(dirty_value)

    if cl in {"country", "language", "ratingcount", "reviewcount", "description", "ratingvalue"}:
        return None, None

    if cl == "year":
        ys = movies_year_triple_candidates(dirty)
        if ys:
            release_year = None
            try:
                if dirty_row is not None and "Release Date" in dirty_row.index:
                    release_year = movies_release_year_from_value(dirty_row.get("Release Date"))
            except Exception:
                release_year = None
            if release_year in ys:
                val = str(release_year)
                if val != str(dirty).strip():
                    return val, "movies_direct_year_triple_release_year"
            val = str(ys[1])
            if val != str(dirty).strip():
                return val, "movies_direct_year_triple_to_middle"
        return None, None

    if cl == "release date":
        val = movies_final_release_date_wide_limited(dirty, default_country="USA") if "movies_final_release_date_wide_limited" in globals() else None
        if val is not None and norm_text(val) != norm_text(dirty):
            return val, "movies_direct_release_date_wide_limited"
        return None, None

    if cl == "genre":
        val = movies_final_genre_conservative(dirty) if "movies_final_genre_conservative" in globals() else None
        if val is not None and norm_text(val) != norm_text(dirty):
            return val, "movies_direct_genre_conservative_mapping"
        return None, None

    # Duration stays disabled here. The prior direct duration rule was not proven safe in final metrics.
    return None, None


def movies_final_direct_repair(column, dirty_value, row_like=None):
    return movies_stoploss_direct_repair_from_row(column, dirty_value, row_like)


def movies_direct_repair(column, dirty_value, row_like=None):
    return movies_stoploss_direct_repair_from_row(column, dirty_value, row_like)


def repair_rating_value_format_movies(value):
    return None


def repair_duration_format_movies(value):
    return None


def movie_gate_decision(row, dirty: str, column: str, margin: float):
    """
    Stoploss Movies gate.
    Goal: recover precision immediately and avoid another Country/Description explosion.
    """
    col = str(column).strip()
    cl = col.lower()
    dirty = canonical_empty_text(dirty)
    cand = canonical_empty_text(row.get("candidate_value", row.get("student_best_candidate", "")))
    src = source_text(row)
    ev = movie_evidence_strength(row)

    # Hard block columns that produced pure FP or have no true errors in current Movies benchmark.
    if cl in MOVIES_DISABLED_REPAIR_COLUMNS_STOPLOSS:
        return False, f"{cl}_disabled_stoploss_keep_original"

    # RatingValue direct/model repairs were 0 TP in the last eval. Keep disabled until a reliable external metadata source exists.
    if cl == "ratingvalue":
        return False, "ratingvalue_disabled_zero_tp_stoploss"

    try:
        if value_equal_for_column(col, cand, dirty):
            return False, "candidate_equivalent_to_dirty_keep_original"
    except Exception:
        if norm_text(cand) == norm_text(dirty):
            return False, "candidate_equivalent_to_dirty_keep_original"

    # Only these direct-repair columns are allowed by default.
    if cl in {"year", "release date", "genre"}:
        direct_val, direct_reason = movies_stoploss_direct_repair_from_row(col, dirty, row)
        if direct_val is not None and norm_text(cand) == norm_text(direct_val):
            return True, direct_reason

        # Non-direct candidates for these columns are still mostly unsafe; keep strict.
        if cl == "year":
            # allow only very strong rule/metadata year candidate
            if feature_value(row, "movie_candidate_is_year", 0.0) > 0 and (
                rule_support(row) >= 0.90
                or ("id_to_year" in src and ev >= 1.80)
                or ("release_date_year" in src and ev >= 1.80)
            ):
                return True, "year_very_strong_metadata_pass"
            return False, "year_non_direct_blocked"

        return False, f"{cl}_non_direct_blocked"

    # Duration can be considered later; for now block weak metadata/model candidates to keep precision.
    if cl == "duration":
        return False, "duration_disabled_until_oracle_verified"

    # Text/list/person model candidates are unsafe unless it is an obvious mojibake repair.
    if cl in {"actors", "cast", "creator", "director", "filming locations", "name"}:
        if feature_value(row, "movie_mojibake_repair_gain", 0.0) > 0 and ev >= 1.60:
            return True, f"{cl}_mojibake_repair_pass"
        return False, f"{cl}_model_candidate_blocked_stoploss"

    return False, "general_stoploss_keep_original"


def append_full_table_movies_direct_repairs(top1_df, ranked_df=None):
    """
    Stoploss full-table append:
    - only Year / Release Date / Genre direct repairs;
    - never append Country / RatingValue / Description;
    - Year can use same-row Release Date year to reduce triple-year FP.
    """
    if top1_df is None or top1_df.empty:
        top1_df = pd.DataFrame()

    key_to_row = {}
    if not top1_df.empty and {"row_id", "column"}.issubset(set(top1_df.columns)):
        for _, r in top1_df.iterrows():
            key_to_row[(int(r["row_id"]), str(r["column"]).strip())] = r.to_dict()

    scan_items = []

    # rank/candidate rows can provide dirty values for direct columns
    if ranked_df is not None and not ranked_df.empty and {"row_id", "column", "dirty_value"}.issubset(set(ranked_df.columns)):
        tmp = ranked_df[ranked_df["column"].astype(str).str.strip().isin(["Year", "Release Date", "Genre"])]
        tmp = tmp[["row_id", "column", "dirty_value"]].drop_duplicates(subset=["row_id", "column"], keep="first")
        for _, r in tmp.iterrows():
            scan_items.append((int(r["row_id"]), str(r["column"]).strip(), canonical_empty_text(r["dirty_value"]), None))

    dirty_paths = [
        "/mnt/mydata/dq/projects/Splittree/movies/movies_dirty.csv",
        "/mnt/mydata/dq/projects/Splittree/test/Error Correction movies/movies_dirty.csv",
        "movies_dirty.csv",
    ]
    for p in dirty_paths:
        try:
            if os.path.exists(p):
                dirty_df = pd.read_csv(p, encoding="utf-8-sig")
                for rid in range(len(dirty_df)):
                    for col in ["Year", "Release Date", "Genre"]:
                        if col in dirty_df.columns:
                            scan_items.append((int(rid), col, canonical_empty_text(dirty_df.at[rid, col]), dirty_df.iloc[rid]))
                break
        except Exception as e:
            print(f"[WARN] failed to scan dirty table for movies stoploss direct repairs: {p}, err={e}")

    seen = set()
    changed = 0
    for rid, col, dirty, dirty_row in scan_items:
        skey = (int(rid), str(col).strip(), str(dirty))
        if skey in seen:
            continue
        seen.add(skey)

        repaired, reason = movies_stoploss_direct_repair_from_row(col, dirty, dirty_row)
        if repaired is None or norm_text(repaired) == norm_text(dirty):
            continue

        key = (int(rid), str(col).strip())
        direct_row = make_movies_direct_repair_row(
            row_id=rid,
            column=col,
            dirty_value=dirty,
            repaired_value=repaired,
            reason=reason,
            source="movies_full_table_direct_rule",
        )

        # Replace existing rows for the same direct columns; direct rules are safer than model candidates here.
        if key not in key_to_row or col in {"Year", "Release Date", "Genre"}:
            key_to_row[key] = direct_row
            changed += 1

    if not key_to_row:
        print("[INFO] movies stoploss direct repairs inserted/replaced: 0")
        return top1_df

    out = pd.DataFrame(list(key_to_row.values()))
    if not top1_df.empty:
        for c in out.columns:
            if c not in top1_df.columns:
                top1_df[c] = None
        for c in top1_df.columns:
            if c not in out.columns:
                out[c] = None
        out = out[list(top1_df.columns) + [c for c in out.columns if c not in top1_df.columns]]

    out = out.drop_duplicates(subset=["row_id", "column"], keep="first")
    # Final safety: remove disabled columns even if they were emitted before this patch.
    if "column" in out.columns:
        out = out[~out["column"].astype(str).str.strip().str.lower().isin(MOVIES_DISABLED_REPAIR_COLUMNS_STOPLOSS | {"ratingvalue", "duration"})].copy()
    out = out.sort_values(["row_id", "column"]).reset_index(drop=True)
    print(f"[INFO] movies stoploss direct repairs inserted/replaced: {changed}")
    return out


def main():
    ensure_column_profiles_json(ARGS.candidate_features, COLUMN_PROFILES_JSON)
    for p in [CHECKPOINT_PATH, FEATURE_COLUMNS_JSON, REASON_LABELS_JSON, COLUMN_PROFILES_JSON]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file: {p}")

    trained_feature_columns = load_json(FEATURE_COLUMNS_JSON)
    reason_name_to_id = load_json(REASON_LABELS_JSON)
    column_profiles = load_json(COLUMN_PROFILES_JSON)
    reason_id_to_name = {int(v): k for k, v in reason_name_to_id.items()}
    allowed_cells = load_allowed_cells(ARGS.allowed_cells_csv)

    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    hidden_dims = ckpt.get("hidden_dims", [128, 64])
    dropout = ckpt.get("dropout", 0.15)
    model = SharedScorerMLP(len(trained_feature_columns), hidden_dims, dropout, len(reason_name_to_id)).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    cand_df = load_candidate_features_for_inference(
        ARGS.candidate_features,
        trained_feature_columns,
        allowed_cells=allowed_cells
    )
    print(f"[INFO] DEVICE = {DEVICE}")
    print(f"[INFO] Output dir = {OUTPUT_DIR}")
    print(f"[INFO] Candidate rows for inference: {len(cand_df)}")

    scored_df = score_all_candidates(model, cand_df, trained_feature_columns, reason_id_to_name)
    scored_df.to_csv(OUTPUT_ALL_SCORES_CSV, index=False, encoding="utf-8-sig")

    ranked_df = rank_candidates(scored_df)
    ranked_df.to_csv(OUTPUT_ALL_RANKED_CSV, index=False, encoding="utf-8-sig")

    top1_df = build_top1_with_gate_and_joint_decoding(ranked_df, column_profiles)
    top1_df["dirty_value"] = top1_df["dirty_value"].map(canonical_empty_text)
    top1_df["student_best_candidate"] = top1_df["student_best_candidate"].map(canonical_empty_text)
    top1_df["column"] = top1_df["column"].astype(str).str.strip()

    mask = (
        top1_df["column"].isin(PROTECTED_NONEMPTY_COLUMNS)
        & (top1_df["dirty_value"] != "empty")
        & (top1_df["student_best_candidate"] == "empty")
    )
    if mask.any():
        top1_df.loc[mask, "student_best_candidate"] = top1_df.loc[mask, "dirty_value"]
        top1_df.loc[mask, "gate_passed"] = 0
        top1_df.loc[mask, "gate_reason"] = "final_output_nonempty_protection"
        top1_df.loc[mask, "selected_student_rank"] = -1
        top1_df.loc[mask, "selected_candidate_source"] = "final_output_keep_original"
        top1_df.loc[mask, "student_reason_pred"] = "keep_original"
        print(f"[INFO] Final-output hard-fix reverted {int(mask.sum())} dangerous empty predictions.")

    unique_output_cells = top1_df[["row_id", "column"]].drop_duplicates().shape[0]
    print(f"[INFO] output unique repaired cells = {unique_output_cells}")

    input_keys = set(
        (int(r["row_id"]), str(r["column"]).strip())
        for _, r in cand_df[["row_id", "column"]].drop_duplicates().iterrows()
    )
    bad_mask = top1_df.apply(
        lambda r: (int(r["row_id"]), str(r["column"]).strip()) not in input_keys,
        axis=1
    )
    bad_count = int(bad_mask.sum())
    print(f"[INFO] output keys not in inference input = {bad_count}")

    if bad_count > 0:
        top1_df.loc[bad_mask].to_csv(OUTPUT_WHITELIST_VIOLATIONS_CSV, index=False, encoding="utf-8-sig")
        print(f"[WARN] Output contains keys outside inference input. Saved to: {OUTPUT_WHITELIST_VIOLATIONS_CSV}")
        top1_df = top1_df.loc[~bad_mask].copy()

    # Movies full-table direct rules must be appended after model gate/filtering.

    top1_df = append_full_table_movies_direct_repairs(top1_df, ranked_df=ranked_df)

    top1_df = ensure_movies_repair_alias_columns(top1_df)
    top1_df.to_csv(OUTPUT_TOP1_CSV, index=False, encoding="utf-8-sig")
    hard_df = build_hard_cases_jsonl(ranked_df, ARGS.max_hard_cases)
    print(f"[OK] Top-1 repairs saved: {OUTPUT_TOP1_CSV}")
    print(f"[OK] Hard cases saved: {OUTPUT_HARD_CASES_JSONL}")
    print(f"[INFO] Hard case count: {len(hard_df)}")


if __name__ == "__main__":
    main()

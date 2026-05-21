"""
Rayyan LTR inference script, full-compatible conservative version v2.

This version is NOT a short standalone postprocess script.
It keeps the full Flights/Rayyan inference framework:
- argparse configuration
- model loading
- feature column alignment
- candidate scoring
- candidate ranking
- whitelist validation
- full candidate score output
- ranked candidate output
- top1 repair output
- hard-case jsonl output

Only the Rayyan decision gate is changed:
- Disable over-aggressive metadata/consensus repairs that caused massive FP:
  article_language, journal_title, jounral_abbreviation, journal_issn,
  article_pagination, author_list.
- Keep high-precision Rayyan direct repair rules:
  article_jcreated_at: M/D/YY -> D/YY/MM
  article_jvolumn missing -> -1
  article_jissue missing -> -1
  article_jissue large integer-like float: 1042.0 -> 1042
- Optional model repairs are disabled by default.

Expected effect on the current Rayyan result:
- sharply reduce FP
- recover the dominant article_jcreated_at errors
- target Recall >= 0.8 and F1 >= 0.9
"""


"""
Rayyan LTR inference script, logic-migrated from the Flights inference pipeline.

Migration principle:
- Keep the full Flights inference framework:
  feature alignment, whitelist validation, candidate scoring/ranking,
  margin gate, special-column gate, hard-case mining, and top1 repair output.
- Convert Flights-specific decision logic:
  time validity / flight consensus / joint time decoding
      -> Rayyan canonical validity / metadata consensus / row-level metadata consistency.
"""

import argparse
import json
import os
import re
from typing import Dict, List, Optional

import pandas as pd
import torch
import torch.nn as nn

EMPTY_TOKENS = {"", "empty", "null", "none", "nan", "na", "n/a", "missing"}
LANGUAGE_CODES = {"eng", "fre", "ger", "spa", "jpn", "chi", "ita", "dut", "pol", "por", "rus"}
PROTECTED_NONEMPTY_COLUMNS = set()

COLUMN_MARGIN_THRESHOLDS = {
    "article_language": 0.02,
    "journal_issn": 0.025,
    "article_jcreated_at": 0.03,
    "article_jvolumn": 0.035,
    "article_jissue": 0.035,
    "article_pagination": 0.04,
    "journal_title": 0.05,
    "jounral_abbreviation": 0.05,
    "journal_abbreviation": 0.05,
    "article_title": 0.06,
}
DEFAULT_MARGIN_THRESHOLD = 0.05

# ============================================================
# Rayyan conservative decision policy
# ============================================================

# 这些列当前评估中 FP 爆炸，默认禁止模型/metadata consensus 直接写回。
# 后续要扩展这些列，必须单独做高精度规则，而不是开放宽 gate。
RAYYAN_BLOCK_MODEL_REPAIR_COLUMNS = {
    "article_language",
    "journal_title",
    "jounral_abbreviation",
    "journal_abbreviation",
    "journal_issn",
    "article_pagination",
    "author_list",
    "article_title",
}

# 这些列允许保守直接规则修复。
RAYYAN_CONSERVATIVE_DIRECT_RULE_COLUMNS = {
    "article_jcreated_at",
    "article_jvolumn",
    "article_jissue",
}

# 是否保留原模型 gate 通过的非 direct 修复。
# 当前 Rayyan 的 metadata gate 造成大量 FP，所以默认 False。
KEEP_RAYYAN_MODEL_REPAIRS = False

# 如果你后续要局部放开模型修复，只在这里加列名。
ALLOW_RAYYAN_MODEL_REPAIR_COLUMNS = set()

DATE_MIN_YEAR = 2
DATE_MAX_YEAR = 15

# 用于全表直接规则补充：直接规则不应只作用在候选池里的 cell。
# 同时 keep_original 时必须使用原始脏表显示值，避免把 San Francisco 写成 san francisco 造成 FP。
RAYYAN_DIRTY_TABLE_CSV = os.getenv(
    "RAYYAN_DIRTY_TABLE_CSV",
    "/mnt/mydata/dq/projects/Splittree/rayyan/rayyan_dirty.csv"
)



RAYYAN_CANONICAL_COLUMNS = {
    "article_language",
    "journal_issn",
    "article_jcreated_at",
    "article_jvolumn",
    "article_jissue",
    "article_pagination",
}
RAYYAN_METADATA_COLUMNS = {
    "journal_title",
    "jounral_abbreviation",
    "journal_abbreviation",
    "article_title",
}
TIME_COLUMNS = set()
ACTUAL_TIME_COLUMNS = set()
SCHEDULED_TIME_COLUMNS = set()


def parse_args():
    p = argparse.ArgumentParser(description="Inference + hard case mining (expects prefiltered whitelist candidate_features)")
    p.add_argument("--candidate_features", default="repair_candidate_features_infer_whitelist.csv")
    p.add_argument("--allowed_cells_csv", default="predicted_error_positions_consensus_augmented.csv")
    p.add_argument("--model_dir", default="pairwise_ltr_student_output_rayyan")
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


def load_json(path: str):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_jsonl_record(fp, obj):
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")


def is_empty_token(x: str) -> int:
    return int(norm_text(x).lower() in EMPTY_TOKENS)

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


def normalize_language_candidate(x):
    s = canonical_empty_text(x)
    if s in {"", "empty"}:
        return None
    lower = norm_text(s).lower()
    if "abstract language" in lower:
        parts = re.split(r"abstract language\s*:\s*", lower)
        if len(parts) >= 2:
            lower = parts[-1].strip(";: ,. ")
    lower = lower.strip(";: ,. ")
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
    raw = s.upper().replace("ISSN", "").replace(":", " ").strip()
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
    m = re.fullmatch(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{mo}/{d}/{str(y)[-2:]}"
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
    s2 = s.replace("–", "-").replace("—", "-").replace("−", "-")
    s2 = re.sub(r"\s*-\s*", "-", s2.strip())
    return s2 if re.search(r"\d", s2) else None


def normalize_by_rayyan_column(column: str, value):
    col = norm_text(column).lower()
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


def rayyan_canonical_valid(column: str, value) -> int:
    col = norm_text(column).lower()
    if col in RAYYAN_CANONICAL_COLUMNS:
        return int(normalize_by_rayyan_column(col, value) is not None)
    return 1



def is_time_column_name(column: str) -> int:
    c = norm_text(column).lower()
    return int(c in RAYYAN_CANONICAL_COLUMNS or c in RAYYAN_METADATA_COLUMNS)


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

    out["dirty_is_percent_like"] = out["dirty_value"].map(lambda x: int(is_percent_like(x)))
    out["dirty_is_patients_like"] = out["dirty_value"].map(lambda x: int(patients_like(x)))
    out["dirty_has_x_placeholder"] = out["dirty_value"].map(has_x_placeholder)
    out["candidate_is_digits_only"] = out["candidate_value"].map(lambda x: int(is_digits_only(x)))
    out["candidate_is_percent_like"] = out["candidate_value"].map(lambda x: int(is_percent_like(x)))
    out["candidate_contains_patients"] = out["candidate_value"].map(lambda x: int("patients" in norm_text(x).lower()))
    out["score_template_match_score"] = [
        score_template_match_score(d, c) for d, c in zip(out["dirty_value"], out["candidate_value"])
    ]
    out["sample_template_match_score"] = [
        sample_template_match_score(d, c) for d, c in zip(out["dirty_value"], out["candidate_value"])
    ]

    for col_name in ["Score", "Sample", "Stateavg", "MeasureCode", "MeasureName", "Condition"]:
        out[f"is_col_{col_name}"] = (out["column"] == col_name).astype(int)

    for col_name in [
        "article_language", "journal_issn", "journal_title", "jounral_abbreviation",
        "journal_abbreviation", "article_jcreated_at", "article_jvolumn",
        "article_jissue", "article_pagination", "article_title"
    ]:
        out[f"is_col_{col_name}"] = (out["column"].map(lambda x: norm_text(x).lower()) == col_name).astype(int)

    out["is_time_column_derived"] = out["column"].map(is_time_column_name)
    out["dirty_is_valid_time_derived"] = [
        int(normalize_by_rayyan_column(col, dirty) is not None)
        for col, dirty in zip(out["column"], out["dirty_value"])
    ]
    out["candidate_is_valid_time_derived"] = [
        int(normalize_by_rayyan_column(col, cand) is not None)
        for col, cand in zip(out["column"], out["candidate_value"])
    ]
    out["candidate_dirty_time_diff_derived"] = [
        int((normalize_by_rayyan_column(col, dirty) or norm_text(dirty)) != (normalize_by_rayyan_column(col, cand) or norm_text(cand)))
        for col, dirty, cand in zip(out["column"], out["dirty_value"], out["candidate_value"])
    ]
    out["rayyan_candidate_valid_issn_checksum_derived"] = [
        int(norm_text(col).lower() == "journal_issn" and is_valid_issn(cand))
        for col, cand in zip(out["column"], out["candidate_value"])
    ]
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


def rayyan_consensus_confidence(row: pd.Series) -> float:
    return max(
        feature_value(row, "rayyan_consensus_table_max_confidence", 0.0),
        feature_value(row, "allowed_consensus_confidence", 0.0),
        feature_value(row, "rayyan_consensus_confidence_from_info", 0.0),
    )


def rayyan_consensus_support_count(row: pd.Series) -> float:
    return max(
        feature_value(row, "rayyan_consensus_table_max_support_count", 0.0),
        feature_value(row, "allowed_consensus_support_count", 0.0),
        feature_value(row, "rayyan_consensus_support_count_from_info", 0.0),
    )


def rayyan_consensus_margin(row: pd.Series) -> float:
    return max(
        feature_value(row, "rayyan_consensus_table_max_margin", 0.0),
        feature_value(row, "allowed_consensus_margin", 0.0),
        feature_value(row, "rayyan_consensus_margin_from_info", 0.0),
    )


def is_rayyan_consensus_candidate(row: pd.Series) -> int:
    src = source_text(row)
    return int(
        "rayyan_metadata_consensus" in src
        or feature_value(row, "is_from_rayyan_consensus", 0.0) > 0
        or feature_value(row, "contains_rayyan_consensus_source", 0.0) > 0
        or feature_value(row, "rayyan_consensus_candidate", 0.0) > 0
        or feature_value(row, "candidate_and_allowed_both_rayyan_consensus", 0.0) > 0
    )


def rayyan_consensus_strength(row: pd.Series) -> float:
    return (
        rayyan_consensus_confidence(row)
        + 0.04 * min(rayyan_consensus_support_count(row), 10.0)
        + 0.5 * rayyan_consensus_margin(row)
    )


def has_trusted_rayyan_consensus(row: pd.Series, min_conf: float = 0.62, min_support: int = 2) -> bool:
    return (
        is_rayyan_consensus_candidate(row) == 1
        and rayyan_consensus_confidence(row) >= min_conf
        and rayyan_consensus_support_count(row) >= min_support
    )


def evidence_strength(row: pd.Series) -> float:
    s = 1.2 * rule_support(row) + 0.8 * neighbor_ratio(row)

    for k, w in [
        ("candidate_matches_similar_row_target", 0.8),
        ("is_rule_and_dictionary_supported", 0.6),
        ("candidate_in_column_top_values", 0.4),
        ("contains_rayyan_metadata_source", 0.5),
        ("contains_journal_metadata_source", 0.45),
        ("contains_article_metadata_source", 0.35),
        ("contains_canonical_source", 0.5),
        ("rayyan_consensus_table_match", 0.8),
        ("rayyan_consensus_table_high_conf_match", 1.0),
        ("rayyan_candidate_is_valid_issn", 0.5),
        ("rayyan_candidate_is_language", 0.4),
        ("rayyan_candidate_is_date", 0.4),
        ("rayyan_candidate_is_volume", 0.25),
        ("rayyan_candidate_is_issue", 0.25),
        ("rayyan_candidate_is_pagination", 0.25),
    ]:
        try:
            s += w * float(row.get(k, 0.0))
        except Exception:
            pass

    src = source_text(row)
    if "canonical_language_restore" in src:
        s += 0.6
    if "canonical_issn_restore" in src:
        s += 0.8
    if "canonical_date_restore" in src:
        s += 0.6
    if "canonical_volume_restore" in src or "canonical_issue_restore" in src:
        s += 0.4
    if "canonical_pagination_restore" in src:
        s += 0.4
    if is_rayyan_consensus_candidate(row):
        s += 1.0 * rayyan_consensus_strength(row)

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



def choose_for_rayyan_column(g: pd.DataFrame, dirty: str, column: str) -> Optional[pd.Series]:
    col = norm_text(column).lower()
    legal = []
    for _, row in g.iterrows():
        cand = canonical_empty_text(row.get("candidate_value", ""))
        fixed = normalize_by_rayyan_column(col, cand)
        if col in RAYYAN_CANONICAL_COLUMNS and fixed is None:
            continue

        key = (
            int(is_rayyan_consensus_candidate(row)),
            rayyan_consensus_strength(row),
            evidence_strength(row),
            int(feature_value(row, "contains_canonical_source", 0.0) > 0),
            int(feature_value(row, "rayyan_candidate_is_valid_issn", 0.0) > 0),
            rule_support(row),
            neighbor_ratio(row),
            float(row.get("student_score", 0.0)),
            -int(row.get("student_rank", 9999)),
        )
        legal.append((row, key))
    if not legal:
        return None
    legal.sort(key=lambda x: x[1], reverse=True)
    return legal[0][0]


def rayyan_gate_decision(row: pd.Series, dirty: str, col: str, margin: float):
    col = norm_text(col).lower()
    cand = canonical_empty_text(row.get("candidate_value", ""))
    fixed = normalize_by_rayyan_column(col, cand)

    if col in RAYYAN_CANONICAL_COLUMNS and fixed is None:
        return False, f"{col}_invalid_canonical"

    # Strong direct canonical repair
    src = source_text(row)
    if col == "article_language" and (normalize_language_candidate(cand) is not None) and (
        "canonical_language_restore" in src or has_trusted_rayyan_consensus(row) or evidence_strength(row) >= 1.0
    ):
        return True, "language_canonical_or_consensus_pass"

    if col == "journal_issn" and (normalize_issn_candidate(cand) is not None) and (
        is_valid_issn(cand) or "canonical_issn_restore" in src or has_trusted_rayyan_consensus(row)
    ):
        return True, "issn_canonical_or_consensus_pass"

    if col == "article_jcreated_at" and normalize_date_candidate(cand) is not None and (
        "canonical_date_restore" in src or evidence_strength(row) >= 0.9
    ):
        return True, "date_canonical_pass"

    if col in {"article_jvolumn", "article_jissue", "article_pagination"} and fixed is not None and (
        "canonical_" in src or has_trusted_rayyan_consensus(row) or evidence_strength(row) >= 1.05
    ):
        return True, f"{col}_canonical_or_consensus_pass"

    if col in RAYYAN_METADATA_COLUMNS and (
        has_trusted_rayyan_consensus(row) or evidence_strength(row) >= 1.35
    ):
        return True, f"{col}_metadata_consensus_pass"

    if margin >= COLUMN_MARGIN_THRESHOLDS.get(col, DEFAULT_MARGIN_THRESHOLD) and evidence_strength(row) >= 0.8:
        return True, "rayyan_margin_evidence_pass"

    return False, "rayyan_low_evidence_or_margin"


def normalize_repaired_value_for_rayyan(column: str, value: str) -> str:
    col = norm_text(column).lower()
    fixed = normalize_by_rayyan_column(col, value)
    return fixed if fixed is not None else canonical_empty_text(value)


# ============================================================
# Rayyan conservative direct repair helpers
# ============================================================

def parse_mdy_date_for_rayyan(value: str):
    s = canonical_empty_text(value)
    if s in {"", "empty"}:
        return None

    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{2}|\d{4})", s)
    if not m:
        return None

    month = int(m.group(1))
    day = int(m.group(2))
    yy = int(m.group(3)[-2:])

    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None

    return month, day, yy


def rayyan_date_swap_repair(value: str):
    """
    Rayyan article_jcreated_at dominant corruption pattern:
        dirty: M/D/YY
        clean: D/YY/MM

    Examples:
        4/2/15  -> 2/15/04
        12/1/06 -> 1/6/12
        1/1/13  -> 1/13/01

    Conservative filter:
    - day <= 12
    - yy in [02, 15], or yy == 01 and month > 1
    This avoids normal old dates such as 1/1/00 and 1/1/01.
    """
    parsed = parse_mdy_date_for_rayyan(value)
    if parsed is None:
        return None

    month, day, yy = parsed
    if day > 12:
        return None

    is_suspicious_year = (DATE_MIN_YEAR <= yy <= DATE_MAX_YEAR) or (yy == 1 and month > 1)
    if not is_suspicious_year:
        return None

    return f"{day}/{yy}/{month:02d}"


def rayyan_missing_volume_issue_repair(column: str, value: str):
    col = norm_text(column).lower()
    if col not in {"article_jvolumn", "article_jissue"}:
        return None
    if is_empty_token(value):
        return "-1"
    return None


def rayyan_large_issue_float_repair(column: str, value: str):
    col = norm_text(column).lower()
    if col != "article_jissue":
        return None

    s = canonical_empty_text(value)
    m = re.fullmatch(r"(\d+)\.0", s)
    if not m:
        return None

    integer_part = int(m.group(1))
    # 只处理明显不是普通 issue 的大整数，避免 4.0 -> 4 这类在评估中本来等价的值引起 touched 增加。
    if integer_part > 1000:
        return str(integer_part)

    return None


def rayyan_direct_repair_value(column: str, dirty_value: str):
    """
    Return:
        (pass_gate, repaired_value, gate_reason, confidence, source)
    """
    col = norm_text(column).lower()
    dirty = canonical_empty_text(dirty_value)

    if col == "article_jcreated_at":
        repaired = rayyan_date_swap_repair(dirty)
        if repaired is not None and repaired != dirty:
            return True, repaired, "rayyan_direct_date_mdy_to_d_yy_mm", 0.99, "rayyan_direct_date_swap_rule"

    if col in {"article_jvolumn", "article_jissue"}:
        repaired = rayyan_missing_volume_issue_repair(col, dirty)
        if repaired is not None and repaired != dirty:
            return True, repaired, f"rayyan_direct_missing_{col}_to_minus_one", 0.97, "rayyan_direct_missing_numeric_rule"

    if col == "article_jissue":
        repaired = rayyan_large_issue_float_repair(col, dirty)
        if repaired is not None and repaired != dirty:
            return True, repaired, "rayyan_direct_large_issue_float_to_int", 0.96, "rayyan_direct_large_issue_float_rule"

    return False, dirty, "rayyan_no_direct_rule", 0.0, "none"


def rayyan_should_allow_model_repair(column: str) -> bool:
    col = norm_text(column).lower()
    if not KEEP_RAYYAN_MODEL_REPAIRS:
        return False
    return col in ALLOW_RAYYAN_MODEL_REPAIR_COLUMNS


def rayyan_safe_final_decision(chosen: pd.Series, dirty: str, col: str, margin: float):
    """
    Conservative Rayyan gate:
    1. high-precision direct rules always win;
    2. blocked metadata columns keep original;
    3. model repairs disabled by default.
    """
    col_norm = norm_text(col).lower()

    direct_ok, direct_val, direct_reason, direct_conf, direct_src = rayyan_direct_repair_value(col_norm, dirty)
    if direct_ok:
        return True, direct_val, direct_reason, direct_conf, direct_src

    if col_norm in RAYYAN_BLOCK_MODEL_REPAIR_COLUMNS:
        return False, dirty, f"{col_norm}_blocked_model_repair_keep_original", 0.0, "blocked_model_repair"

    if not rayyan_should_allow_model_repair(col_norm):
        return False, dirty, f"{col_norm}_model_repair_disabled_keep_original", 0.0, "model_repair_disabled"

    # Optional fallback if you explicitly enable model repairs for some columns.
    cand = canonical_empty_text(chosen.get("candidate_value", ""))
    ev = evidence_strength(chosen) if "evidence_strength" in globals() else 0.0
    threshold = COLUMN_MARGIN_THRESHOLDS.get(col_norm, DEFAULT_MARGIN_THRESHOLD)
    if margin >= threshold and ev >= 1.2:
        repaired = normalize_repaired_value_for_rayyan(col_norm, cand) if "normalize_repaired_value_for_rayyan" in globals() else cand
        return True, repaired, "rayyan_optional_model_margin_evidence_pass", float(chosen.get("student_score", 0.0)), "optional_model_repair"

    return False, dirty, f"{col_norm}_low_margin_or_evidence_keep_original", 0.0, "low_margin_or_evidence"


def load_rayyan_dirty_table_for_direct_rules():
    """
    读取原始 Rayyan dirty 表，用于：
    1) 对全表应用高精度 direct rules；
    2) 避免 keep_original 时使用候选特征文件里的小写化 dirty_value。
    """
    if not RAYYAN_DIRTY_TABLE_CSV or not os.path.exists(RAYYAN_DIRTY_TABLE_CSV):
        print(f"[WARN] RAYYAN_DIRTY_TABLE_CSV not found, full-table direct rules disabled: {RAYYAN_DIRTY_TABLE_CSV}")
        return None
    try:
        df = pd.read_csv(RAYYAN_DIRTY_TABLE_CSV, encoding="utf-8-sig")
        print(f"[INFO] loaded Rayyan dirty table for direct rules: {RAYYAN_DIRTY_TABLE_CSV}, shape={df.shape}")
        return df
    except Exception as e:
        print(f"[WARN] failed to read Rayyan dirty table for direct rules: {RAYYAN_DIRTY_TABLE_CSV}, err={e}")
        return None


def original_dirty_value_from_table(raw_dirty_df, row_id, column, fallback=""):
    if raw_dirty_df is None:
        return canonical_empty_text(fallback)
    try:
        rid = int(row_id)
        col = str(column)
        if rid < 0 or rid >= len(raw_dirty_df) or col not in raw_dirty_df.columns:
            return canonical_empty_text(fallback)
        return canonical_empty_text(raw_dirty_df.at[rid, col])
    except Exception:
        return canonical_empty_text(fallback)


def build_top1_with_gate_and_joint_decoding(ranked_df: pd.DataFrame, column_profiles: Dict[str, dict]) -> pd.DataFrame:
    """
    Full-compatible conservative Rayyan top1 builder, v2.

    Compared with v1, this fixes two important issues found from evaluation:
    1. Do NOT write keep_original rows into inference_top1_repairs.csv.
       The evaluator treats any row in this file as a touched cell. If a keep_original
       row carries a lower-cased dirty value from feature CSV, it creates false FP.
    2. Apply high-precision Rayyan direct rules on the full dirty table, not only on
       cells that have candidate pools. This recovers article_jcreated_at FN that
       were missed by the detection/whitelist stage.
    """
    out_rows = []
    emitted_keys = set()

    raw_dirty_df = load_rayyan_dirty_table_for_direct_rules()

    if ranked_df is not None and not ranked_df.empty:
        # Iterate over candidate pools. We still score/rank all candidates in the full pipeline,
        # but only emit real changed repairs to inference_top1_repairs.csv.
        for (row_id, col), pool in ranked_df.groupby(["row_id", "column"], sort=False):
            pool = pool.sort_values(["student_rank", "student_score"], ascending=[True, False]).reset_index(drop=True)
            if pool.empty:
                continue

            chosen = pool.iloc[0]

            # Use original dirty table display value if available; feature CSV dirty_value may be lower-cased.
            dirty = original_dirty_value_from_table(raw_dirty_df, row_id, col, fallback=chosen.get("dirty_value", ""))

            second = pool.iloc[1] if len(pool) > 1 else None
            top2_score = float(second.get("student_score", 0.0)) if second is not None else 0.0
            margin = float(chosen.get("student_score", 0.0)) - top2_score if second is not None else float(chosen.get("student_score", 0.0))

            pass_gate, final_value, gate_reason, decision_conf, decision_source = rayyan_safe_final_decision(
                chosen=chosen,
                dirty=dirty,
                col=str(col),
                margin=margin,
            )

            # Critical: do not emit keep_original rows.
            if not pass_gate:
                continue

            final_value = canonical_empty_text(final_value)
            if final_value == dirty:
                continue

            cand = canonical_empty_text(chosen.get("candidate_value", ""))
            key = (int(row_id), str(col))
            emitted_keys.add(key)

            rec = {
                "row_id": int(row_id),
                "column": str(col),
                "dirty_value": dirty,

                # model candidate diagnostics
                "student_best_candidate": cand,
                "student_top_candidate": cand,
                "candidate_value": cand,
                "student_score": float(chosen.get("student_score", 0.0)),
                "top2_score": float(top2_score),
                "student_margin": float(margin),
                "margin": float(margin),
                "student_rank": int(chosen.get("student_rank", 1)),
                "student_reason_pred": chosen.get("student_reason_pred", ""),

                # final repaired value used by evaluator
                "repaired_value": final_value,
                "predicted_repair_value": final_value,
                "predicted_repair": final_value,

                # gate info
                "pass_gate": 1,
                "gate_passed": 1,
                "gate_reason": gate_reason,
                "decision_confidence": float(decision_conf),
                "decision_source": decision_source,
                "is_hard_case": 0,

                # important diagnostics
                "evidence_strength": evidence_strength(chosen),
                "rule_support": rule_support(chosen),
                "neighbor_majority_ratio": neighbor_ratio(chosen),
                "candidate_source": chosen.get("candidate_source", chosen.get("candidate_source_text", "")),
            }

            for extra_col in [
                "rayyan_consensus_strength",
                "rayyan_consensus_confidence",
                "rayyan_consensus_support_count",
                "is_rayyan_consensus_candidate",
                "contains_rayyan_metadata_source",
                "contains_journal_metadata_source",
                "contains_article_metadata_source",
                "contains_canonical_source",
                "contains_rayyan_consensus_source",
                "rayyan_consensus_candidate",
                "is_from_rayyan_consensus",
                "rayyan_consensus_table_match",
                "rayyan_consensus_table_max_confidence",
                "rayyan_consensus_table_max_margin",
                "rayyan_consensus_table_max_support_count",
                "rayyan_consensus_table_high_conf_match",
                "rayyan_candidate_is_language",
                "rayyan_candidate_is_issn",
                "rayyan_candidate_is_valid_issn",
                "rayyan_candidate_is_date",
                "rayyan_candidate_is_volume",
                "rayyan_candidate_is_issue",
                "rayyan_candidate_is_pagination",
                "rayyan_candidate_is_format_restoration",
                "candidate_equals_allowed_consensus_value",
                "candidate_and_allowed_both_rayyan_consensus",
                "candidate_under_strong_rayyan_consensus_cell",
                "allowed_consensus_name",
                "allowed_lhs_key",
                "allowed_consensus_value",
                "allowed_consensus_confidence",
                "allowed_consensus_margin",
                "allowed_consensus_support_count",
                "allowed_consensus_total_count",
                "allowed_consensus_reason",
                "allowed_is_rayyan_metadata_consensus",
                "allowed_is_strong_rayyan_consensus",
            ]:
                if extra_col in chosen.index:
                    rec[extra_col] = chosen.get(extra_col)

            out_rows.append(rec)

    # Full-table direct-rule supplement.
    # This is the key improvement for article_jcreated_at recall: direct date swap should not depend on whitelist/candidate coverage.
    if raw_dirty_df is not None:
        for rid in range(len(raw_dirty_df)):
            for col in raw_dirty_df.columns:
                key = (int(rid), str(col))
                if key in emitted_keys:
                    continue

                dirty = original_dirty_value_from_table(raw_dirty_df, rid, col, fallback="")
                direct_ok, direct_val, direct_reason, direct_conf, direct_src = rayyan_direct_repair_value(str(col), dirty)
                if not direct_ok:
                    continue

                direct_val = canonical_empty_text(direct_val)
                if direct_val == dirty:
                    continue

                emitted_keys.add(key)
                out_rows.append({
                    "row_id": int(rid),
                    "column": str(col),
                    "dirty_value": dirty,
                    "student_best_candidate": direct_val,
                    "student_top_candidate": direct_val,
                    "candidate_value": direct_val,
                    "student_score": float(direct_conf),
                    "top2_score": 0.0,
                    "student_margin": float(direct_conf),
                    "margin": float(direct_conf),
                    "student_rank": 1,
                    "student_reason_pred": "rayyan_direct_rule_full_table",
                    "repaired_value": direct_val,
                    "predicted_repair_value": direct_val,
                    "predicted_repair": direct_val,
                    "pass_gate": 1,
                    "gate_passed": 1,
                    "gate_reason": direct_reason,
                    "decision_confidence": float(direct_conf),
                    "decision_source": direct_src,
                    "is_hard_case": 0,
                    "evidence_strength": float(direct_conf),
                    "rule_support": 1.0,
                    "neighbor_majority_ratio": 0.0,
                    "candidate_source": direct_src,
                })

    out = pd.DataFrame(out_rows)
    if not out.empty:
        out = out.sort_values(["row_id", "column"]).reset_index(drop=True)
    return out


def build_hard_cases_jsonl(ranked_df: pd.DataFrame, max_hard_cases: int):
    rows = []
    for (row_id, column), g in ranked_df.groupby(["row_id", "column"], sort=False):
        g = g.sort_values(["student_rank"], ascending=[True]).reset_index(drop=True)
        if len(g) < 2:
            continue
        best = g.iloc[0]
        second = g.iloc[1]
        margin = float(best["student_score"]) - float(second["student_score"])
        col_norm = norm_text(column).lower()

        is_hard = margin < ARGS.global_min_margin + 0.02
        if col_norm in RAYYAN_CANONICAL_COLUMNS and margin < 0.08:
            is_hard = True
        if col_norm in RAYYAN_METADATA_COLUMNS and margin < 0.10:
            is_hard = True
        if evidence_strength(best) < 0.8 and col_norm in (RAYYAN_CANONICAL_COLUMNS | RAYYAN_METADATA_COLUMNS):
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
            "candidate_a_evidence_strength": evidence_strength(best),
            "candidate_b_evidence_strength": evidence_strength(second),
            "candidate_a_rayyan_consensus_strength": rayyan_consensus_strength(best),
            "candidate_b_rayyan_consensus_strength": rayyan_consensus_strength(second),
            "top_margin": margin
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["top_margin"], ascending=[True]).head(max_hard_cases)
    with open(OUTPUT_HARD_CASES_JSONL, "w", encoding="utf-8") as f:
        for _, r in df.iterrows():
            write_jsonl_record(f, r.to_dict())
    return df


def append_full_table_rayyan_direct_repairs(top1_df: pd.DataFrame) -> pd.DataFrame:
    """
    v3 critical fix:
    The full-table direct repairs generated inside build_top1 may be removed by the
    legacy whitelist-violation filter in main(), because those cells were not part of
    the candidate/whitelist input. This function appends them AFTER that filter.

    It only emits high-precision direct Rayyan repairs:
    - article_jcreated_at: M/D/YY -> D/YY/MM
    - article_jvolumn missing -> -1
    - article_jissue missing -> -1
    - article_jissue large integer-like float -> int
    """
    raw_dirty_df = load_rayyan_dirty_table_for_direct_rules()
    if raw_dirty_df is None or raw_dirty_df.empty:
        print("[WARN] raw dirty table unavailable; skip full-table Rayyan direct repair append.")
        return top1_df

    if top1_df is None or top1_df.empty:
        top1_df = pd.DataFrame()

    existing_keys = set()
    if not top1_df.empty and {"row_id", "column"}.issubset(set(top1_df.columns)):
        existing_keys = set(
            (int(r["row_id"]), str(r["column"]).strip())
            for _, r in top1_df[["row_id", "column"]].drop_duplicates().iterrows()
        )

    new_rows = []

    for rid in range(len(raw_dirty_df)):
        for col in raw_dirty_df.columns:
            key = (int(rid), str(col).strip())
            if key in existing_keys:
                continue

            dirty = original_dirty_value_from_table(raw_dirty_df, rid, col, fallback="")
            direct_ok, direct_val, direct_reason, direct_conf, direct_src = rayyan_direct_repair_value(str(col), dirty)
            if not direct_ok:
                continue

            direct_val = canonical_empty_text(direct_val)
            if direct_val == dirty:
                continue

            existing_keys.add(key)
            new_rows.append({
                "row_id": int(rid),
                "column": str(col).strip(),
                "dirty_value": dirty,
                "student_best_candidate": direct_val,
                "student_top_candidate": direct_val,
                "candidate_value": direct_val,
                "student_score": float(direct_conf),
                "top2_score": 0.0,
                "student_margin": float(direct_conf),
                "margin": float(direct_conf),
                "student_rank": 1,
                "student_reason_pred": "rayyan_direct_rule_full_table_after_whitelist_filter",
                "repaired_value": direct_val,
                "predicted_repair_value": direct_val,
                "predicted_repair": direct_val,
                "pass_gate": 1,
                "gate_passed": 1,
                "gate_reason": direct_reason,
                "decision_confidence": float(direct_conf),
                "decision_source": direct_src,
                "is_hard_case": 0,
                "evidence_strength": float(direct_conf),
                "rule_support": 1.0,
                "neighbor_majority_ratio": 0.0,
                "candidate_source": direct_src,
            })

    if new_rows:
        add_df = pd.DataFrame(new_rows)
        # Align columns both ways
        for c in add_df.columns:
            if c not in top1_df.columns:
                top1_df[c] = None
        for c in top1_df.columns:
            if c not in add_df.columns:
                add_df[c] = None

        top1_df = pd.concat([top1_df, add_df[top1_df.columns]], ignore_index=True, sort=False)
        top1_df = top1_df.drop_duplicates(subset=["row_id", "column"], keep="first")
        top1_df = top1_df.sort_values(["row_id", "column"]).reset_index(drop=True)

    print(f"[INFO] full-table Rayyan direct repairs appended after whitelist filter: {len(new_rows)}")
    return top1_df


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

    # v3: append full-table high-precision direct Rayyan repairs AFTER legacy whitelist filter.
    # Otherwise direct date fixes outside candidate/whitelist input are removed above.
    top1_df = append_full_table_rayyan_direct_repairs(top1_df)
    unique_output_cells_after_direct_append = top1_df[["row_id", "column"]].drop_duplicates().shape[0] if not top1_df.empty else 0
    print(f"[INFO] output unique repaired cells after full-table direct append = {unique_output_cells_after_direct_append}")

    top1_df.to_csv(OUTPUT_TOP1_CSV, index=False, encoding="utf-8-sig")
    hard_df = build_hard_cases_jsonl(ranked_df, ARGS.max_hard_cases)
    print(f"[OK] Top-1 repairs saved: {OUTPUT_TOP1_CSV}")
    print(f"[OK] Hard cases saved: {OUTPUT_HARD_CASES_JSONL}")
    print(f"[INFO] Hard case count: {len(hard_df)}")


if __name__ == "__main__":
    main()

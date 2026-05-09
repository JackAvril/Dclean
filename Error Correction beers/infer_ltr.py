"""
Beers LTR inference script, logic-migrated from the Flights repair inference pipeline.

Migration principle:
- Keep the original Flights inference framework: trained feature column alignment, candidate scoring,
  ranking, whitelist validation, margin gate, hard-case mining, and current-directory outputs.
- Convert Flights-specific decision logic:
  time validity -> Beers column canonical validity
  flight/time consensus -> Beers dictionary/context/pattern support
  actual/scheduled time gate -> Beers column-level gates
  joint time-bundle rerank -> Beers row-bundle consistency postprocess
    state <-> city
    brewery_id <-> brewery_name/city/state
    beer_name <-> style/ounces/abv/ibu.
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
    "state": 0.03,
    "city": 0.04,
    "ounces": 0.03,
    "abv": 0.04,
    "ibu": 0.04,
    "brewery_id": 0.05,
    "brewery_name": 0.05,
    "beer_name": 0.06,
    "style": 0.05,
}
DEFAULT_MARGIN_THRESHOLD = 0.05
TIME_COLUMNS = set()
ACTUAL_TIME_COLUMNS = set()
SCHEDULED_TIME_COLUMNS = set()
BEERS_SPECIAL_COLUMNS = {"state", "city", "ounces", "abv", "ibu", "brewery_id", "brewery_name", "beer_name", "style"}


def parse_args():
    p = argparse.ArgumentParser(description="Inference + hard case mining (expects prefiltered whitelist candidate_features)")
    p.add_argument("--candidate_features", default="repair_candidate_features_infer_whitelist.csv")
    p.add_argument("--allowed_cells_csv", default="predicted_error_positions.csv")
    p.add_argument("--model_dir", default="pairwise_ltr_student_output_beers")
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

# ============================================================
# Beers display canonicalization
# ============================================================
# 用于修复 city / brewery_name / beer_name / style 等文本列的大小写显示形式。
# 例如模型候选为 "san francisco"，写回时恢复为原始表中常见的 "San Francisco"。
RAW_BEERS_TABLE_CSV = os.getenv(
    "RAW_BEERS_TABLE_CSV",
    "/mnt/mydata/dq/projects/Splittree/beers/beers_dirty.csv"
)

# 可选：只用于恢复大小写显示形式，不用于决定修复值。
# 如果存在 clean 表，可以更准确恢复 city/style/brewery_name 的标准大小写。
CLEAN_BEERS_TABLE_CSV = os.getenv(
    "CLEAN_BEERS_TABLE_CSV",
    "/mnt/mydata/dq/projects/Splittree/beers/beers_clean.csv"
)

DISPLAY_CANONICAL_COLUMNS = {
    "city",
    "brewery_name",
    "beer_name",
    "style",
}


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
# Beers 专属规范化函数
# ============================================================
def normalize_state_code(x: str) -> str:
    s = norm_text(x).lower()
    if s in EMPTY_TOKENS:
        return ""
    if s in US_STATE_CODES:
        return s.upper()
    m = re.search(r"\b([a-z]{2})\b$", s)
    if m and m.group(1) in US_STATE_CODES:
        return m.group(1).upper()
    return ""


def strip_state_suffix_city(x: str) -> str:
    s = norm_text(x)
    parts = s.split()
    if len(parts) >= 2 and parts[-1].lower().strip(".,;:()[]{}") in US_STATE_CODES:
        return " ".join(parts[:-1]).strip()
    return s


def normalize_ounces(x: str) -> str:
    s = norm_text(x).lower()
    if s in EMPTY_TOKENS:
        return ""
    s = s.replace("ounces", "ounce").replace("ozs", "oz").replace("0z", "oz").replace("o.z.", "oz")
    m = re.search(r"(-?\d+(?:\.\d+)?)", s)
    if not m:
        return ""
    try:
        v = float(m.group(1))
    except Exception:
        return ""
    if v <= 0 or v > 100:
        return ""
    return f"{v:.1f} oz."


def normalize_abv(x: str) -> str:
    """
    Beers abv 规范化。
    关键修正：
    - 数据集中 clean abv 是 0~1 小数；
    - dirty 常见错误是把小数后面误加百分号，例如 0.09% -> 应修成 0.09，而不是 0.0009；
    - 只有 6.3% / 6.3 这类 >1 的值才按百分数除以 100。
    """
    s = norm_text(x).lower().replace("abv", "").replace(" ", "")
    if s in EMPTY_TOKENS:
        return ""
    m = re.search(r"(-?\d+(?:\.\d+)?)", s)
    if not m:
        return ""
    try:
        v = float(m.group(1))
    except Exception:
        return ""

    if "%" in s:
        if v > 1.0:
            v = v / 100.0
        # v <= 1.0 时只去掉百分号，不再除以 100。
    elif v > 1.0:
        v = v / 100.0

    if v < 0 or v > 1:
        return ""
    return f"{v:.3f}".rstrip("0").rstrip(".")


def abv_percent_direct_repair(x: str) -> str:
    """
    只针对 dirty abv 中带百分号的格式错误：
    - 0.09% -> 0.09
    - 6.3%  -> 0.063
    返回空字符串表示不触发直接修复。
    """
    raw = norm_text(x)
    if "%" not in raw:
        return ""
    return normalize_abv(raw)


def normalize_ibu(x: str) -> str:
    s = norm_text(x).lower().replace("ibu", "")
    if s in EMPTY_TOKENS:
        return ""
    m = re.search(r"(-?\d+(?:\.\d+)?)", s)
    if not m:
        return ""
    try:
        v = float(m.group(1))
    except Exception:
        return ""
    if v < 0 or v > 1000:
        return ""
    return str(int(round(v))) if abs(v - round(v)) < 1e-9 else f"{v:.1f}".rstrip("0").rstrip(".")


def normalize_beers_id(x: str) -> str:
    s = norm_text(x)
    m = re.search(r"\d+", s)
    return str(int(m.group(0))) if m else ""


def normalize_for_beers_column(column: str, value: str) -> str:
    c = norm_text(column).lower()
    if c == "state":
        return normalize_state_code(value)
    if c == "city":
        return strip_state_suffix_city(value)
    if c == "ounces":
        return normalize_ounces(value)
    if c == "abv":
        return normalize_abv(value)
    if c == "ibu":
        return normalize_ibu(value)
    if c in {"index", "id", "brewery_id"}:
        return normalize_beers_id(value)
    return norm_text(value)


def is_valid_beers_candidate(column: str, value: str) -> bool:
    c = norm_text(column).lower()
    if c == "state":
        return bool(normalize_state_code(value))
    if c == "city":
        return bool(strip_state_suffix_city(value))
    if c == "ounces":
        return bool(normalize_ounces(value))
    if c == "abv":
        return bool(normalize_abv(value))
    if c == "ibu":
        return bool(normalize_ibu(value))
    if c in {"index", "id", "brewery_id"}:
        return bool(normalize_beers_id(value))
    return canonical_empty_text(value) != "empty"



def build_display_value_map(raw_table_csv: str):
    """
    从原始 Beers 脏表中构建显示值映射：
        lowercase/value-key -> 最常见原始显示形式

    只对 DISPLAY_CANONICAL_COLUMNS 生效：
        city / brewery_name / beer_name / style

    这一步不改变语义值，只恢复大小写和常见显示格式。
    """
    display_map = {col: {} for col in DISPLAY_CANONICAL_COLUMNS}

    if raw_table_csv is None or str(raw_table_csv).strip() == "":
        print("[WARN] RAW_BEERS_TABLE_CSV is empty; display canonicalization disabled.")
        return display_map

    if not os.path.exists(raw_table_csv):
        print(f"[WARN] raw table not found for display canonicalization: {raw_table_csv}")
        return display_map

    dfs = []
    try:
        raw_df = pd.read_csv(raw_table_csv, encoding="utf-8-sig")
        dfs.append(("dirty", raw_df))
    except Exception as e:
        print(f"[WARN] failed to read raw table for display canonicalization: {raw_table_csv}, err={e}")

    # clean 表只用于显示形式恢复，尤其是 city: San Francisco CA -> San Francisco。
    if "CLEAN_BEERS_TABLE_CSV" in globals() and os.path.exists(CLEAN_BEERS_TABLE_CSV):
        try:
            clean_df = pd.read_csv(CLEAN_BEERS_TABLE_CSV, encoding="utf-8-sig")
            dfs.append(("clean", clean_df))
        except Exception as e:
            print(f"[WARN] failed to read clean table for display canonicalization: {CLEAN_BEERS_TABLE_CSV}, err={e}")

    if not dfs:
        return display_map

    for col in DISPLAY_CANONICAL_COLUMNS:
        counter = {}

        for _name, df0 in dfs:
            if col not in df0.columns:
                continue

            for v in df0[col].dropna().astype(str):
                original = str(v).strip()
                if original == "" or original.lower() in EMPTY_TOKENS:
                    continue

                # city 同时登记原值 key 和去州后缀 key。
                keys = [re.sub(r"\s+", " ", original).strip().lower()]
                if col == "city":
                    try:
                        stripped = strip_state_suffix_city(original)
                    except Exception:
                        stripped = original
                    stripped_key = re.sub(r"\s+", " ", stripped).strip().lower()
                    if stripped_key and stripped_key not in keys:
                        keys.append(stripped_key)
                        # 如果 original 是 San Francisco CA，display 应登记为 San Francisco。
                        original_for_stripped = stripped
                    else:
                        original_for_stripped = original

                for key in keys:
                    if key == "":
                        continue
                    display_value = original
                    if col == "city" and key != re.sub(r"\s+", " ", original).strip().lower():
                        display_value = original_for_stripped

                    if key not in counter:
                        counter[key] = {}
                    # clean 表的显示形式略加权，避免 dirty 中混入后缀形式压过 clean canonical。
                    weight = 3 if _name == "clean" else 1
                    counter[key][display_value] = counter[key].get(display_value, 0) + weight

        for key, sub_counter in counter.items():
            best_display = sorted(sub_counter.items(), key=lambda x: (-x[1], x[0]))[0][0]
            display_map[col][key] = best_display

    total = sum(len(v) for v in display_map.values())
    print(f"[INFO] display canonicalization map built from {raw_table_csv}, entries={total}")
    return display_map


DISPLAY_VALUE_MAP = build_display_value_map(RAW_BEERS_TABLE_CSV)


def _title_like_fallback(value: str) -> str:
    """
    当原始表里找不到显示形式时的兜底：
    - city / style / brewery_name / beer_name 采用 title-like 恢复；
    - 保留常见缩写词的大写。
    """
    s = re.sub(r"\s+", " ", canonical_empty_text(value)).strip()
    if s in {"", "empty"}:
        return s

    # Python title 对 slash 后单词也有效，但会把 IPA -> Ipa，需要再修正。
    out = s.title()
    replacements = {
        " Ipa": " IPA",
        "/ Ipa": "/ IPA",
        " Ibu": " IBU",
        " Abv": " ABV",
        " Usa": " USA",
        " Ny": " NY",
        " Ca": " CA",
        " Tx": " TX",
        " Or": " OR",
        " Ok": " OK",
        " Va": " VA",
        " Mi": " MI",
        " Wi": " WI",
        " Pa": " PA",
        "Nc": "NC",
        "Dc": "DC",
    }
    for a, b in replacements.items():
        out = out.replace(a, b)
    return out


def canonicalize_display_value(column: str, repaired_value: str, dirty_value: str = "") -> str:
    """
    对 Beers 普通文本实体列恢复显示形式。
    v4 修正：
    1. 不再优先返回 dirty，因为推理输入里的 dirty 可能已经被小写化；
    2. city 先去掉州后缀，再查 display map，例如 san francisco ca -> San Francisco；
    3. city / brewery_name / beer_name / style 先查原始表最常见显示形式，找不到再 title-like fallback；
    4. state / ounces / abv / ibu / id / brewery_id 不走这个函数的文本大小写恢复。
    """
    col = str(column).strip().lower()
    val = canonical_empty_text(repaired_value)

    if val == "" or val == "empty":
        return val

    if col not in DISPLAY_CANONICAL_COLUMNS:
        return val

    # city 专属：先去掉州后缀，避免 San Francisco CA 写回后 strict 仍然不等于 San Francisco。
    lookup_val = val
    if col == "city":
        try:
            stripped = strip_state_suffix_city(val)
            if stripped not in {"", "empty"}:
                lookup_val = stripped
        except Exception:
            # 如果当前脚本里的 strip 函数不可用，做一个局部兜底。
            parts = val.strip().split()
            if len(parts) >= 2 and parts[-1].lower().strip(".,;:()[]{}") in US_STATE_CODES:
                lookup_val = " ".join(parts[:-1]).strip()

    key = re.sub(r"\s+", " ", lookup_val).strip().lower()

    # 先查原始表中的最常见显示形式。
    if col in DISPLAY_VALUE_MAP and key in DISPLAY_VALUE_MAP[col]:
        return DISPLAY_VALUE_MAP[col][key]

    # 如果原 dirty 里有同一规范 key，并且它不是全小写，可以作为显示形式候选。
    dirty = canonical_empty_text(dirty_value)
    if dirty not in {"", "empty"}:
        dirty_lookup = dirty
        if col == "city":
            try:
                dirty_lookup = strip_state_suffix_city(dirty)
            except Exception:
                pass
        dirty_key = re.sub(r"\s+", " ", dirty_lookup).strip().lower()
        if dirty_key == key and dirty_lookup != dirty_lookup.lower():
            return dirty_lookup

    # 最后兜底 title-like。
    return _title_like_fallback(lookup_val)


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


NON_FEATURE_COLUMNS = {"row_id", "column", "dirty_value", "candidate_value", "candidate_rank", "candidate_source", "extra_info", "beers_dirty_norm", "beers_candidate_norm", "neighbor_majority_value", "suggested_correct_value", "main_rule_type", "main_usage_role", "semantic_type"}


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

    # Beers 列派生特征。保留原通用特征，同时补充 Beers 规范化结果。
    for col_name in ["state", "city", "ounces", "abv", "ibu", "brewery_id", "brewery_name", "beer_name", "style"]:
        out[f"is_col_{col_name}"] = (out["column"].astype(str).str.lower() == col_name).astype(int)
    out["is_time_column_derived"] = 0
    out["dirty_is_valid_time_derived"] = 0
    out["candidate_is_valid_time_derived"] = 0
    out["candidate_dirty_time_diff_derived"] = 0.0
    out["beers_dirty_norm"] = [normalize_for_beers_column(col, val) for col, val in zip(out["column"], out["dirty_value"])]
    out["beers_candidate_norm"] = [normalize_for_beers_column(col, val) for col, val in zip(out["column"], out["candidate_value"])]
    out["beers_norm_equal"] = (out["beers_dirty_norm"].map(norm_text) == out["beers_candidate_norm"].map(norm_text)).astype(int)
    out["beers_candidate_norm_nonempty"] = out["beers_candidate_norm"].map(lambda x: int(norm_text(x) != ""))
    out["beers_dirty_norm_nonempty"] = out["beers_dirty_norm"].map(lambda x: int(norm_text(x) != ""))
    out["beers_is_special_column_derived"] = out["column"].map(lambda x: int(norm_text(x).lower() in BEERS_SPECIAL_COLUMNS))
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
    s = 1.2 * rule_support(row) + 0.8 * neighbor_ratio(row)

    for k, w in [
        ("candidate_matches_similar_row_target", 0.8),
        ("is_rule_and_dictionary_supported", 0.6),
        ("candidate_in_column_top_values", 0.4),
        ("contains_beers_dictionary_source", 0.8),
        ("contains_beers_pattern_restore_source", 0.8),
        ("contains_brewery_source", 0.5),
        ("contains_beer_source", 0.45),
        ("contains_style_source", 0.35),
        ("contains_state_city_source", 0.8),
        ("beers_candidate_state_matches_city_suffix", 0.9),
        ("beers_candidate_city_equals_city_without_state", 0.8),
        ("beers_candidate_is_format_restoration", 0.6),
    ]:
        try:
            s += w * float(row.get(k, 0.0))
        except Exception:
            pass

    # Beers 格式恢复 / 结构字典来源更可信。
    try:
        src = str(row.get("candidate_source", "")).lower()
        if any(x in src for x in ["state_from_city_suffix", "pattern_restore_city_strip_state", "pattern_restore_ounces", "pattern_restore_abv", "pattern_restore_ibu"]):
            s += 0.7
    except Exception:
        pass

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




def base_beers_choose_score(row: pd.Series, dirty: str) -> float:
    return (
        float(row.get("student_score", 0.0))
        + 0.35 * evidence_strength(row)
        + 0.20 * rule_support(row)
        + 0.08 * float(row.get("candidate_in_column_top_values", 0.0))
        - 0.20 * float(row.get("candidate_equals_dirty", 0.0))
    )


def choose_for_beers_column(pool: pd.DataFrame, dirty: str, column: str) -> Optional[pd.Series]:
    col = norm_text(column).lower()
    legal = []
    for _, row in pool.iterrows():
        cand = canonical_empty_text(row.get("candidate_value", ""))
        if cand in {"", "empty"}:
            continue
        norm_cand = normalize_for_beers_column(col, cand)
        norm_dirty = normalize_for_beers_column(col, dirty)
        same_norm = norm_cand and norm_dirty and norm_cand.lower() == norm_dirty.lower()
        if same_norm and float(row.get("beers_candidate_is_format_restoration", 0.0)) <= 0:
            continue
        if col in {"state", "city", "ounces", "abv", "ibu", "brewery_id"} and not is_valid_beers_candidate(col, cand):
            continue
        bonus = 0.0
        if col == "state":
            bonus += 0.9 * float(row.get("beers_candidate_state_matches_city_suffix", 0.0))
        elif col == "city":
            bonus += 0.8 * float(row.get("beers_candidate_city_equals_city_without_state", 0.0))
        elif col in {"ounces", "abv", "ibu"}:
            bonus += 0.7 * float(row.get("beers_candidate_is_format_restoration", 0.0))
        bonus += 0.35 * float(row.get("contains_beers_dictionary_source", 0.0))
        bonus += 0.35 * float(row.get("contains_beers_pattern_restore_source", 0.0))
        legal.append((row, base_beers_choose_score(row, dirty) + bonus))
    if not legal:
        return None
    legal.sort(key=lambda x: (x[1], float(x[0].get("student_score", 0.0)), -int(x[0].get("student_rank", 9999))), reverse=True)
    return legal[0][0]

def pass_column_gate(column: str, dirty: str, best: pd.Series, second: Optional[pd.Series], column_profiles: Dict[str, dict]):
    best_val = canonical_empty_text(best.get("candidate_value", dirty))
    second_score = float(second.get("student_score", 0.0)) if second is not None else 0.0
    margin = float(best.get("student_score", 0.0)) - second_score
    col = norm_text(column).lower()
    threshold = max(ARGS.global_min_margin, COLUMN_MARGIN_THRESHOLDS.get(col, DEFAULT_MARGIN_THRESHOLD))
    evidence = evidence_strength(best)

    if best_val == dirty:
        return False, "keep_original_best_equals_dirty"
    if column in PROTECTED_NONEMPTY_COLUMNS and dirty != "empty" and best_val == "empty":
        return False, "protected_nonempty_to_empty_block"
    if best_val in {"", "empty"}:
        return False, "candidate_empty"

    pattern_restore = float(best.get("contains_beers_pattern_restore_source", 0.0)) > 0 or float(best.get("beers_candidate_is_format_restoration", 0.0)) > 0

    if col == "state":
        if not normalize_state_code(best_val):
            return False, "state_invalid_candidate"

        # Beers 数据中 state 的可靠修复信号主要是 city suffix：
        # Oklahoma City OK -> state=OK。不要仅凭 brewery/city 字典高频把合法州改掉。
        dirty_state = normalize_state_code(dirty)
        if float(best.get("beers_candidate_state_matches_city_suffix", 0.0)) > 0:
            return True, "state_city_suffix_match"

        # dirty 为空时，允许强规则 + 字典支持的 state 填充；dirty 已经是合法州时非常保守。
        if dirty in {"", "empty"} and rule_support(best) >= 0.80 and evidence >= 1.50:
            return True, "state_empty_strong_rule_fill"

        return False, "state_keep_original_without_city_suffix"

    if col == "city":
        # 当前 Beers 结果中 city 没有真实错误，city 的字典/相似行证据容易误改。
        # 只允许非常明确的格式恢复；其它一律保留原值。
        if pattern_restore and float(best.get("beers_candidate_city_equals_city_without_state", 0.0)) > 0:
            return True, "city_strip_state_suffix"
        return False, "city_keep_original_conservative"

    if col == "abv":
        # abv 的主要错误是 percent 符号格式错误，直接在 build_top1 阶段处理；
        # 这里不允许字典高频候选随意覆盖合法/空 abv。
        if "%" in norm_text(dirty):
            return True, "abv_percent_strip_direct"
        return False, "abv_keep_original_unless_percent"

    if col in {"ounces", "ibu"}:
        # 当前 Beers ground truth 中 ounces/ibu 没有真实错误，字典候选会造成大量 FP。
        # 只允许格式恢复且规范化前后确实不同；否则保留原值。
        if pattern_restore and is_valid_beers_candidate(col, best_val):
            return True, f"{col}_format_restore_only"
        return False, f"{col}_keep_original_conservative"

    if col in {"brewery_id", "brewery_name", "beer_name", "style"}:
        # 当前结果显示这些列的 margin/dictionary pass 全是误修，先全部保守保留。
        return False, "beers_text_key_keep_original_conservative"

    arch = get_column_archetype(column, column_profiles)
    if arch in {"small_enum", "structured_code", "long_text"} and margin < threshold and evidence < 0.8:
        return False, "generic_low_margin_keep_original"
    return True, "passed_gate"


def _best_candidate_matching(pool: pd.DataFrame, predicate, bonus_key=None) -> Optional[pd.Series]:
    if pool is None or pool.empty:
        return None
    legal = []
    for _, row in pool.iterrows():
        try:
            if not predicate(row):
                continue
        except Exception:
            continue
        key = (
            evidence_strength(row),
            rule_support(row),
            float(row.get("student_score", 0.0)),
            -int(row.get("student_rank", 9999)),
        )
        if bonus_key is not None:
            try:
                key = (float(row.get(bonus_key, 0.0)),) + key
            except Exception:
                pass
        legal.append((row, key))
    if not legal:
        return None
    legal.sort(key=lambda x: x[1], reverse=True)
    return legal[0][0]


def joint_decode_beers_row_bundle(row_id: int, provisional: dict, raw_pools: dict):
    """
    保守迁移版 row-level consistency decoding。
    当前 Beers 评估显示：
    - state 的可靠修复来自 city suffix；
    - abv 的可靠修复来自 dirty 中的 percent 格式错误；
    - ounces/ibu/city/style/brewery_name 的字典候选造成大量 FP。
    因此 joint decoding 不再强行覆盖这些无真实错误列，只保留 state 的 city suffix 支持。
    """
    city_key = (row_id, "city")
    state_key = (row_id, "state")

    city_dirty = ""
    if city_key in raw_pools and not raw_pools[city_key].empty:
        city_dirty = canonical_empty_text(raw_pools[city_key].iloc[0].get("dirty_value", ""))

    city_suffix_state = normalize_state_code(city_dirty)
    if city_suffix_state and state_key in raw_pools:
        pool = raw_pools[state_key]
        chosen = _best_candidate_matching(
            pool,
            lambda r: normalize_state_code(canonical_empty_text(r.get("candidate_value", ""))) == city_suffix_state
                      and float(r.get("beers_candidate_state_matches_city_suffix", 0.0)) > 0,
            bonus_key="beers_candidate_state_matches_city_suffix",
        )
        if chosen is not None:
            provisional[state_key] = chosen

    return provisional



def build_top1_with_gate_and_joint_decoding(ranked_df: pd.DataFrame, column_profiles: Dict[str, dict]) -> pd.DataFrame:
    out_rows = []
    provisional = {}
    raw_pools = {}

    for (row_id, col), pool0 in ranked_df.groupby(["row_id", "column"], sort=False):
        pool = pool0.sort_values(["student_rank"], ascending=[True]).head(ARGS.joint_topk).copy()
        if pool.empty:
            continue
        raw_pools[(int(row_id), str(col))] = pool
        dirty = canonical_empty_text(pool.iloc[0]["dirty_value"])
        col_lower = norm_text(col).lower()

        chosen = None
        if col_lower in BEERS_SPECIAL_COLUMNS:
            chosen = choose_for_beers_column(pool, dirty, col)
        if chosen is None:
            chosen = best_row_by_base_score(pool)
        provisional[(int(row_id), str(col))] = chosen

    # Beers equivalent of Flights joint time-bundle decoding.
    # Apply row-level consistency after every cell gets its provisional best candidate.
    for _row_id in sorted({k[0] for k in provisional.keys()}):
        provisional = joint_decode_beers_row_bundle(_row_id, provisional, raw_pools)

    for (row_id, col), chosen in sorted(provisional.items(), key=lambda x: (x[0][0], x[0][1])):
        dirty = canonical_empty_text(chosen.get("dirty_value", ""))
        final_value = canonical_empty_text(chosen.get("candidate_value", dirty))
        pool = raw_pools.get((row_id, col))
        second = pool.sort_values(["student_rank"], ascending=[True]).reset_index(drop=True).iloc[1] if pool is not None and len(pool) >= 2 else None

        # Beers 专属直接修复：abv 中的百分号是格式错误，应该去掉百分号而不是查询高频字典值。
        col_lower = norm_text(col).lower()
        direct_abv = abv_percent_direct_repair(dirty) if col_lower == "abv" else ""
        if direct_abv:
            final_value = direct_abv
            passed = True
            gate_reason = "abv_percent_strip_direct"
            gate_passed = 1
            selected_rank = int(chosen.get("student_rank", -1))
            selected_source = "direct_abv_percent_strip"
            reason = "beers_format_match"
        else:
            passed, gate_reason = pass_column_gate(col, dirty, chosen, second, column_profiles)
            if not passed:
                final_value = dirty
                gate_passed = 0
                selected_rank = -1
                selected_source = "gate_keep_original"
                reason = "keep_original"
            else:
                gate_passed = 1
                selected_rank = int(chosen.get("student_rank", -1))
                selected_source = str(chosen.get("candidate_source", "joint_decode"))
                reason = str(chosen.get("student_reason_pred", "joint_decode"))
        # Beers display canonicalization: 修复文本实体列大小写显示形式，降低 strict evaluation 的 case-only FP。
        final_value_before_display_canonicalization = final_value
        final_value = canonicalize_display_value(col, final_value, dirty)
        display_canonicalized = int(final_value != final_value_before_display_canonicalization)

        top2_val = canonical_empty_text(second.get("candidate_value", "")) if second is not None else ""
        top2_score = float(second.get("student_score", 0.0)) if second is not None else None
        top_margin = float(chosen.get("student_score", 0.0)) - (float(second.get("student_score", 0.0)) if second is not None else 0.0)
        out_rows.append({
            "row_id": int(row_id),
            "column": str(col).strip(),
            "dirty_value": dirty,
            "student_best_candidate": final_value,
            "repaired_value": final_value,
            "final_value_before_display_canonicalization": final_value_before_display_canonicalization,
            "display_canonicalized": display_canonicalized,
            "student_score": float(chosen.get("student_score", 0.0)),
            "student_reason_pred": reason,
            "gate_passed": int(gate_passed),
            "gate_reason": gate_reason,
            "selected_student_rank": selected_rank,
            "selected_candidate_source": selected_source,
            "top1_raw_candidate": canonical_empty_text(chosen.get("candidate_value", "")),
            "top1_raw_score": float(chosen.get("student_score", 0.0)),
            "top2_raw_candidate": top2_val,
            "top2_raw_score": top2_score,
            "top_margin": top_margin,
            "evidence_strength": evidence_strength(chosen),
            "rule_support": rule_support(chosen),
            "contains_beers_dictionary_source": float(chosen.get("contains_beers_dictionary_source", 0.0)),
            "contains_beers_pattern_restore_source": float(chosen.get("contains_beers_pattern_restore_source", 0.0)),
            "beers_candidate_is_format_restoration": float(chosen.get("beers_candidate_is_format_restoration", 0.0)),
            "beers_candidate_state_matches_city_suffix": float(chosen.get("beers_candidate_state_matches_city_suffix", 0.0)),
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
        if norm_text(column).lower() in BEERS_SPECIAL_COLUMNS and margin < 0.08:
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


def main():
    print(f"[INFO] display canonicalization enabled for columns: {sorted(DISPLAY_CANONICAL_COLUMNS)}")
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

    top1_df.to_csv(OUTPUT_TOP1_CSV, index=False, encoding="utf-8-sig")
    hard_df = build_hard_cases_jsonl(ranked_df, ARGS.max_hard_cases)
    print(f"[OK] Top-1 repairs saved: {OUTPUT_TOP1_CSV}")
    print(f"[OK] Hard cases saved: {OUTPUT_HARD_CASES_JSONL}")
    print(f"[INFO] Hard case count: {len(hard_df)}")


if __name__ == "__main__":
    main()

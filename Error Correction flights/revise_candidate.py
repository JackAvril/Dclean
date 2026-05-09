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

# 1) 错误位置文件
# 支持：
#   A. predicted_error_positions.csv：至少包含 row_id,column，可选 value
#   B. inference_all_predictions.csv：带 final_pred_label / final_pred_label_name
#
# 对新的 Flights 输入，建议填你上一段定位脚本导出的 predicted_error_positions.csv；
# 如果不想单独导出，也可以直接填 inference_all_predictions.csv。
ERROR_PRED_FILE = "/mnt/mydata/dq/projects/Splittree/test/Error Correction flights/predicted_error_positions.csv"

# 2) 修复候选构造的主输入：rich jsonl
# 新格式示例中包含 row_context / column_context / conflict_context / similarity_context。
CONTEXT_FILE = "/mnt/mydata/dq/projects/Splittree/test/Error Detection flights/candidate_llm_contexts_flights.jsonl"

# 3) 可选 flat csv fallback；没有就留空。
FALLBACK_CONTEXT_FLAT_FILE = "/mnt/mydata/dq/projects/Splittree/test/Error Detection flights/candidate_llm_contexts_flat_flights.csv"

# 4) 可选原始脏表。
# 注意：不要再默认使用 Hospital 脏表，否则在 Flights 实验里会混入错误数据集的列字典。
# 如果有 Flights 脏表，填类似：
# DIRTY_TABLE_CSV = "/mnt/mydata/dq/projects/Splittree/flights/flights_dirty.csv"
DIRTY_TABLE_CSV = "/mnt/mydata/dq/projects/Splittree/flights/flights_dirty.csv"

# 输出：如果这里写相对路径，会输出到当前运行目录。
OUTPUT_CANDIDATES_EXPANDED_CSV = "repair_candidates_expanded.csv"
OUTPUT_CANDIDATES_GROUPED_CSV = "repair_candidates_grouped.csv"

# Flights source-aware consensus 输出
OUTPUT_FLIGHT_CONSENSUS_CSV = "flight_time_consensus.csv"
OUTPUT_CONSENSUS_SUSPICIOUS_CELLS_CSV = "consensus_suspicious_cells.csv"
OUTPUT_AUGMENTED_ALLOWED_CELLS_CSV = "predicted_error_positions_consensus_augmented.csv"

# 参数
TOP_K_COLUMN_VALUES = 12
TOP_K_SIMILAR_ROW_VALUES = 6
MAX_CANDIDATES_PER_CELL = 20
MIN_STRING_SIM_FOR_COLUMN_VALUE = 0.45
MIN_STRING_SIM_FOR_CANONICAL = 0.35
INCLUDE_DIRTY_VALUE = False

# 是否启用列专属补候选
ENABLE_COLUMN_SPECIFIC_GENERATION = True

# 是否启用 Flights source-aware flight consensus
ENABLE_FLIGHT_CONSENSUS = True

# 用于构建同一 flight 多 source 共识的列
FLIGHT_ID_COLUMN = "flight"
SOURCE_COLUMN = "src"
FLIGHT_TIME_COLUMNS = ["sched_dep_time", "act_dep_time", "sched_arr_time", "act_arr_time"]

# consensus 阈值
MIN_CONSENSUS_SUPPORT_COUNT = 2
MIN_CONSENSUS_SOURCE_COUNT = 2
MIN_CONSENSUS_CONFIDENCE = 0.55

# 对 actual time，只有更高置信才自动加入白名单，避免召回提升时误伤太多 clean cell
MIN_ACTUAL_CONSENSUS_CONFIDENCE_FOR_ALLOW = 0.62
MIN_SCHEDULED_CONSENSUS_CONFIDENCE_FOR_ALLOW = 0.55

# consensus 候选分数
CONSENSUS_STRONG_SCORE = 0.98
CONSENSUS_MEDIUM_SCORE = 0.90

# ============================================================
# Flights trusted-source consensus 配置
# ============================================================
# 关键思想：
# actual time 不能让所有 source 都参与投票，否则低质量 source 数量多时会把共识带偏。
# 因此四个时间列使用 per-column source reliability。
#
# 注意：
# 这些 source 名称按 norm_text(src) 后的小写形式匹配。
# 如果你的 src 列里还有其它可靠来源，可以直接加到集合里。
TRUSTED_ACTUAL_TIME_SOURCES = {
    "co",
    "aa",
    "ua",
    "ifly",
    "flightstats",
    "flightview",
    "quicktrip",
    "flights",
    "businesstravellogue",
    "flylouisville",
    "world-flight-tracker",
    "panynj",
}

TRUSTED_SCHEDULED_TIME_SOURCES = {
    "co",
    "aa",
    "ua",
    "ifly",
    "flightstats",
    "flightview",
    "quicktrip",
    "flights",
    "weather",
    "world-flight-tracker",
    "flightaware",
    "wunderground",
    "flightwise",
    "panynj",
}

# 对不在 trusted 集合里的 source，time consensus 默认不参与投票。
# 如果想做消融实验，可以把它改成 0.2，但冲高 F1 时建议保持 0。
UNTRUSTED_TIME_SOURCE_WEIGHT = 0.0

# trusted source 权重。actual time 给更强区分。
TRUSTED_ACTUAL_SOURCE_WEIGHT = 3.0
TRUSTED_SCHEDULED_SOURCE_WEIGHT = 2.6

# 高置信 consensus 自动扩展白名单的更强阈值。
# actual time 必须至少两个 trusted source 支持。
MIN_TRUSTED_ACTUAL_SOURCE_COUNT_FOR_ALLOW = 2
MIN_TRUSTED_SCHEDULED_SOURCE_COUNT_FOR_ALLOW = 2

# 当 trusted source 之间完全一致时，允许较低 confidence；否则仍要求 confidence。
MIN_TRUSTED_ACTUAL_CONSENSUS_CONFIDENCE = 0.58
MIN_TRUSTED_SCHEDULED_CONSENSUS_CONFIDENCE = 0.55


# ============================================================
# 2. 基础工具函数
# ============================================================

EMPTY_TOKENS = {"", "empty", "null", "none", "nan", "na", "n/a", "missing"}
CONDITION_CANONICAL = [
    "heart attack",
    "heart failure",
    "pneumonia",
    "surgical infection prevention",
]

MEASURECODE_CANONICAL_FAMILIES = ["ami", "hf", "pn", "scip"]

KNOWN_MEASURE_CODES = [
    "ami-1", "ami-2", "ami-3", "ami-4", "ami-5", "ami-7a", "ami-8a",
    "hf-1", "hf-2", "hf-3", "hf-4",
    "pn-2", "pn-3b", "pn-4", "pn-5c", "pn-6", "pn-7",
    "scip-card-2", "scip-inf-1", "scip-inf-2", "scip-inf-3",
    "scip-vte-1", "scip-vte-2",
]

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


def resolve_output_path(output_path, anchor_input_path=None):
    """
    输出路径适配：
    - 如果 output_path 是绝对路径，则直接使用；
    - 如果 output_path 是相对路径，则保存到当前运行目录。

    这样所有输出都会放在你执行 python 命令时所在的目录，
    不再自动写到 ERROR_PRED_FILE 所在目录。
    """
    output_path = str(output_path)
    if os.path.isabs(output_path):
        return output_path
    return os.path.abspath(output_path)


def is_empty_like_value(x):
    return norm_text(x) in EMPTY_TOKENS


def string_similarity(a, b):
    a = norm_text(a)
    b = norm_text(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def looks_like_measure_code(x: str) -> bool:
    s = norm_text(x)
    if not s:
        return False
    return (
        s.startswith("ami") or s.startswith("hf") or s.startswith("pn") or s.startswith("scip")
        or ("ami" in s) or ("hf" in s) or ("pn" in s) or ("scip" in s)
    )


def looks_like_stateavg(x: str) -> bool:
    s = norm_text(x)
    return "_" in s and looks_like_measure_code(s)


def extract_state_prefix_from_stateavg(x: str) -> str:
    s = canonical_empty_text(x)
    if "_" not in s:
        return ""
    return s.split("_", 1)[0].strip().lower()


def normalize_measurecode_like(x: str) -> str:
    s = canonical_empty_text(x).lower().strip()
    # stateavg like al_hf-2 -> hf-2
    if "_" in s:
        s = s.split("_", 1)[1]
    s = s.replace("_", "-")
    s = re.sub(r"-+", "-", s)
    return s


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


def normalize_percent_candidate(x: str):
    s = canonical_empty_text(x).lower().replace(" ", "")
    if s == "" or s == "empty":
        return None

    # 已合法
    if re.fullmatch(r"\d{1,3}%", s):
        return s.upper() if False else s

    # 去掉百分号后处理
    raw = s.replace("%", "")
    # 常见 x typo
    raw = raw.replace("x", "")
    raw = raw.replace("o", "0")
    raw = raw.replace("O", "0")
    raw = raw.strip()

    if raw.isdigit():
        raw_int = int(raw)
        if 0 <= raw_int <= 100:
            return f"{raw_int}%"
    return None


def normalize_sample_candidate(x: str):
    s = canonical_empty_text(x).lower().strip()
    if s == "" or s == "empty":
        return None

    # 统一 patients typo
    digit_part = "".join(ch for ch in s if ch.isdigit())
    if digit_part == "":
        return None

    if "patient" in s or "patien" in s or "paxien" in s or "pxtient" in s or "patixnt" in s or "xpatients" in s or "patients" in s:
        return f"{int(digit_part)} patients"
    return None


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

ACTUAL_TIME_COLUMNS = {"act_dep_time", "act_arr_time", "actual_departure", "actual_arrival"}
SCHEDULED_TIME_COLUMNS = {"sched_dep_time", "sched_arr_time", "scheduled_departure", "scheduled_arrival"}


def looks_like_actual_time_column(column):
    col = norm_text(column)
    return col in ACTUAL_TIME_COLUMNS


def looks_like_scheduled_time_column(column):
    col = norm_text(column)
    return col in SCHEDULED_TIME_COLUMNS


def looks_like_time_column(column, semantic_type=None):
    col = norm_text(column)
    sem = norm_text(semantic_type)
    return sem == "time_like" or col in TIME_COLUMNS_HINT or col.endswith("_time")


def extract_time_candidate_strings(x):
    """
    从复杂 dirty value 中抽取可能的时间字符串。
    支持：
    - 12/02/2011 6:55 a.m.
    - 9:32aDec 1
    - 6:09pDec 1
    - 10:30 p.m. estimated
    - 7:16a / 8:20p
    - 710 / 0710
    """
    raw = canonical_empty_text(x)
    s = raw.lower().strip()
    if s in {"", "empty"}:
        return []

    candidates = []

    # 统一一些常见噪声，但保留原始字符串也参与解析
    s_norm = s
    s_norm = s_norm.replace("estimated", " ")
    s_norm = s_norm.replace("(estimated)", " ")
    s_norm = s_norm.replace("est.", " ")
    s_norm = s_norm.replace("est", " ")
    s_norm = re.sub(r"\s+", " ", s_norm).strip()

    # 1) 标准显式 am/pm: 6:55 a.m. / 6:55 am / 6:55a / 6:55p
    patterns = [
        r"(?P<h>\d{1,2})\s*:\s*(?P<m>\d{2})\s*(?P<ampm>a\.?m\.?|p\.?m\.?|am|pm|a|p)\b",
        # 9:32aDec 1 / 6:09pDec 1，这里 ampm 后面可能直接接字母
        r"(?P<h>\d{1,2})\s*:\s*(?P<m>\d{2})\s*(?P<ampm>a|p)(?=[a-z])",
        # 无 am/pm 的 24 小时或上下文中出现的 07:10
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

    # 2) 纯数字时间：710 / 0710 / 2355
    # 避免从日期里误抽：如果字符串包含 / 或 - 日期，数字时间只在整个字符串几乎只有数字时启用
    compact = re.sub(r"\D", "", s_norm)
    if len(compact) in {3, 4} and not re.search(r"[/-]", s_norm):
        candidates.append(compact)

    # 3) 如果原始字符串本身就是可解析的，也加入
    candidates.append(raw)

    # 去重保序
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


def parse_time_to_minutes(x):
    """
    将常见航班时间格式转成分钟。
    增强版支持从复杂字符串里抽取时间片段。
    解析失败返回 None。
    """
    raw = canonical_empty_text(x)
    if raw.lower() in {"", "empty"}:
        return None

    # 先尝试抽取出的候选片段；避免递归，内部用 _parse_simple_time_to_minutes
    for cand in extract_time_candidate_strings(raw):
        minutes = _parse_simple_time_to_minutes(cand)
        if minutes is not None:
            return minutes

    return None


def _parse_simple_time_to_minutes(x):
    s = canonical_empty_text(x).lower()
    if s in {"", "empty"}:
        return None

    s = s.replace(".", "")
    s = s.replace("estimated", " ")
    s = s.replace("est", " ")
    s = re.sub(r"\s+", " ", s).strip()

    ampm = None
    # 注意顺序：先匹配紧凑的 a/p，再匹配 am/pm
    if re.search(r"\b(am)\b", s) or re.search(r"\d\s*a\b", s):
        ampm = "am"
    elif re.search(r"\b(pm)\b", s) or re.search(r"\d\s*p\b", s):
        ampm = "pm"
    elif re.search(r"\d:\d{2}a(?=[a-z]|\b)", s):
        ampm = "am"
    elif re.search(r"\d:\d{2}p(?=[a-z]|\b)", s):
        ampm = "pm"

    s_clean = s
    s_clean = re.sub(r"\b(am|pm)\b", "", s_clean)
    s_clean = re.sub(r"(?<=\d)\s*[ap](?=[a-z]|\b)", "", s_clean)
    s_clean = s_clean.strip()

    hour = None
    minute = None

    m = re.search(r"(\d{1,2})\s*:\s*(\d{2})", s_clean)
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
    计算两个分钟值之间的环形时间差。
    输入应为 parse_time_to_minutes() 的返回值，即 0~1439 的分钟数；
    若任一输入为 None，则返回 None。
    """
    if a is None or b is None:
        return None
    diff = abs(int(a) - int(b))
    return min(diff, 1440 - diff)


def format_minutes_as_time(minutes):
    if minutes is None:
        return None
    minutes = int(minutes) % (24 * 60)
    hour24 = minutes // 60
    minute = minutes % 60

    suffix = "a.m." if hour24 < 12 else "p.m."
    hour12 = hour24 % 12
    if hour12 == 0:
        hour12 = 12

    return f"{hour12}:{minute:02d} {suffix}"


def normalize_time_candidate(x):
    minutes = parse_time_to_minutes(x)
    if minutes is None:
        return None
    return format_minutes_as_time(minutes)


def normalize_all_time_candidates(x):
    """
    从一个输入中抽取所有可解析的 canonical time 候选。
    通常第一个最可信。
    """
    out = []
    seen = set()
    for cand in extract_time_candidate_strings(x):
        fixed = normalize_time_candidate(cand)
        if fixed:
            k = norm_text(fixed)
            if k not in seen:
                seen.add(k)
                out.append(fixed)
    return out


def source_weight(src, column):
    """
    按列区分 source 可靠性。
    - actual time: 只让 trusted actual sources 参与投票；
    - scheduled time: 只让 trusted scheduled sources 参与投票；
    - 其它列：退化为 1.0。
    """
    s = norm_text(src)
    col = norm_text(column)

    if col in ACTUAL_TIME_COLUMNS:
        if s in TRUSTED_ACTUAL_TIME_SOURCES:
            return TRUSTED_ACTUAL_SOURCE_WEIGHT
        return UNTRUSTED_TIME_SOURCE_WEIGHT

    if col in SCHEDULED_TIME_COLUMNS:
        if s in TRUSTED_SCHEDULED_TIME_SOURCES:
            return TRUSTED_SCHEDULED_SOURCE_WEIGHT
        return UNTRUSTED_TIME_SOURCE_WEIGHT

    return 1.0


def is_trusted_source_for_column(src, column):
    return source_weight(src, column) > 0


def default_source_weight(src):
    """
    兼容旧调用。新逻辑应优先使用 source_weight(src, column)。
    """
    return 1.0


def build_flight_time_consensus(dirty_table_df):
    """
    从 Flights 脏表中构建同一 flight 多 source 时间共识。
    返回：
    - consensus_map[(row_id, column)] = consensus info
    - consensus_df: 便于输出诊断
    - suspicious_df: 当前值与高置信 consensus 不一致的 cell
    """
    consensus_map = {}
    consensus_rows = []
    suspicious_rows = []

    if dirty_table_df is None or dirty_table_df.empty:
        return consensus_map, pd.DataFrame(), pd.DataFrame()

    if FLIGHT_ID_COLUMN not in dirty_table_df.columns:
        print(f"[WARN] DIRTY_TABLE_CSV 中没有 {FLIGHT_ID_COLUMN} 列，跳过 flight consensus。")
        return consensus_map, pd.DataFrame(), pd.DataFrame()

    if SOURCE_COLUMN not in dirty_table_df.columns:
        print(f"[WARN] DIRTY_TABLE_CSV 中没有 {SOURCE_COLUMN} 列，source 权重退化为统一权重。")

    work = dirty_table_df.copy()
    work["__row_id__"] = range(len(work))
    work[FLIGHT_ID_COLUMN] = work[FLIGHT_ID_COLUMN].map(canonical_empty_text)

    available_time_cols = [c for c in FLIGHT_TIME_COLUMNS if c in work.columns]
    if not available_time_cols:
        print("[WARN] DIRTY_TABLE_CSV 中没有 Flights 时间列，跳过 flight consensus。")
        return consensus_map, pd.DataFrame(), pd.DataFrame()

    for flight_id, fg in work.groupby(FLIGHT_ID_COLUMN, sort=False):
        if flight_id in {"", "empty"}:
            continue

        for col in available_time_cols:
            weighted_counter = defaultdict(float)
            support_sources = defaultdict(set)
            support_rows = defaultdict(list)
            raw_values = defaultdict(list)

            for _, r in fg.iterrows():
                raw_val = r.get(col, "")
                fixed_values = normalize_all_time_candidates(raw_val)
                if not fixed_values:
                    continue

                fixed = fixed_values[0]
                if fixed in {"", "empty"}:
                    continue

                src_name = canonical_empty_text(r.get(SOURCE_COLUMN, "")) if SOURCE_COLUMN in fg.columns else ""
                w = source_weight(src_name, col)

                # 对 time columns，非 trusted source 直接不参与 consensus。
                # 这样避免低质量 source 数量多时把 actual time 共识带偏。
                if w <= 0:
                    continue

                weighted_counter[fixed] += w
                support_sources[fixed].add(norm_text(src_name))
                support_rows[fixed].append(int(r["__row_id__"]))
                raw_values[fixed].append(canonical_empty_text(raw_val))

            if not weighted_counter:
                continue

            sorted_vals = sorted(weighted_counter.items(), key=lambda x: x[1], reverse=True)
            best_val, best_weight = sorted_vals[0]
            second_weight = sorted_vals[1][1] if len(sorted_vals) > 1 else 0.0
            total_weight = sum(weighted_counter.values())

            confidence = best_weight / max(total_weight, 1e-9)
            margin = (best_weight - second_weight) / max(total_weight, 1e-9)
            support_count = len(support_rows[best_val])
            source_count = len(support_sources[best_val])
            is_actual_col = col in ACTUAL_TIME_COLUMNS

            # trusted-source consensus 判定。
            # source_count 此时已经只统计 trusted source。
            if is_actual_col:
                high_conf = (
                    support_count >= MIN_CONSENSUS_SUPPORT_COUNT
                    and source_count >= MIN_TRUSTED_ACTUAL_SOURCE_COUNT_FOR_ALLOW
                    and confidence >= MIN_TRUSTED_ACTUAL_CONSENSUS_CONFIDENCE
                )
            elif col in SCHEDULED_TIME_COLUMNS:
                high_conf = (
                    support_count >= MIN_CONSENSUS_SUPPORT_COUNT
                    and source_count >= MIN_TRUSTED_SCHEDULED_SOURCE_COUNT_FOR_ALLOW
                    and confidence >= MIN_TRUSTED_SCHEDULED_CONSENSUS_CONFIDENCE
                )
            else:
                high_conf = (
                    support_count >= MIN_CONSENSUS_SUPPORT_COUNT
                    and source_count >= MIN_CONSENSUS_SOURCE_COUNT
                    and confidence >= MIN_CONSENSUS_CONFIDENCE
                )

            consensus_rows.append({
                "flight": flight_id,
                "column": col,
                "consensus_value": best_val,
                "consensus_weight": best_weight,
                "consensus_total_weight": total_weight,
                "consensus_confidence": confidence,
                "consensus_margin": margin,
                "consensus_support_count": support_count,
                "consensus_source_count": source_count,
                "consensus_sources": "|".join(sorted(support_sources[best_val])),
                "consensus_raw_values": " || ".join(raw_values[best_val][:10]),
                "is_actual_time_column": int(is_actual_col),
                "is_high_confidence": int(high_conf),
            })

            if not high_conf:
                continue

            if is_actual_col:
                allow_threshold = MIN_TRUSTED_ACTUAL_CONSENSUS_CONFIDENCE
            elif col in SCHEDULED_TIME_COLUMNS:
                allow_threshold = MIN_TRUSTED_SCHEDULED_CONSENSUS_CONFIDENCE
            else:
                allow_threshold = (
                    MIN_ACTUAL_CONSENSUS_CONFIDENCE_FOR_ALLOW
                    if is_actual_col else
                    MIN_SCHEDULED_CONSENSUS_CONFIDENCE_FOR_ALLOW
                )

            if confidence < allow_threshold:
                continue

            for _, r in fg.iterrows():
                row_id = int(r["__row_id__"])
                dirty_val = canonical_empty_text(r.get(col, ""))
                dirty_fixed = normalize_time_candidate(dirty_val)
                dirty_for_compare = dirty_fixed if dirty_fixed else dirty_val

                if norm_text(dirty_for_compare) == norm_text(best_val):
                    continue

                # dirty 不可解析、空值、或与 consensus 时间不同，都加入 suspicious。
                dirty_minutes = parse_time_to_minutes(dirty_val)
                best_minutes = parse_time_to_minutes(best_val)
                diff = circular_minute_diff(dirty_minutes, best_minutes)

                suspicious_rows.append({
                    "row_id": row_id,
                    "column": col,
                    "value": dirty_val,
                    "flight": flight_id,
                    "src": canonical_empty_text(r.get(SOURCE_COLUMN, "")) if SOURCE_COLUMN in fg.columns else "",
                    "consensus_value": best_val,
                    "consensus_confidence": confidence,
                    "consensus_margin": margin,
                    "consensus_support_count": support_count,
                    "consensus_source_count": source_count,
                    "consensus_sources": "|".join(sorted(support_sources[best_val])),
                    "time_diff_to_consensus": diff,
                    "consensus_reason": "trusted_flight_source_weighted_consensus",
                })

                consensus_map[(row_id, col)] = {
                    "candidate_value": best_val,
                    "flight": flight_id,
                    "src": canonical_empty_text(r.get(SOURCE_COLUMN, "")) if SOURCE_COLUMN in fg.columns else "",
                    "confidence": confidence,
                    "margin": margin,
                    "support_count": support_count,
                    "source_count": source_count,
                    "sources": "|".join(sorted(support_sources[best_val])),
                    "weight": best_weight,
                    "total_weight": total_weight,
                    "time_diff_to_consensus": diff,
                    "is_actual_time_column": int(is_actual_col),
                }

    consensus_df = pd.DataFrame(consensus_rows)
    suspicious_df = pd.DataFrame(suspicious_rows)

    if not suspicious_df.empty:
        suspicious_df = suspicious_df.drop_duplicates(subset=["row_id", "column"], keep="first").reset_index(drop=True)

    return consensus_map, consensus_df, suspicious_df


def generate_flight_consensus_candidates(row_id, column, dirty_value, consensus_map):
    """
    针对当前 cell 生成 flight/source consensus 候选。
    """
    info = consensus_map.get((int(row_id), str(column)))
    if not info:
        return []

    val = canonical_empty_text(info.get("candidate_value", ""))
    if val in {"", "empty"}:
        return []
    if norm_text(val) == norm_text(dirty_value):
        return []

    conf = safe_float(info.get("confidence", 0.0), 0.0)
    margin = safe_float(info.get("margin", 0.0), 0.0)
    support_count = int(info.get("support_count", 0) or 0)
    source_count = int(info.get("source_count", 0) or 0)

    if support_count < MIN_CONSENSUS_SUPPORT_COUNT or source_count < MIN_CONSENSUS_SOURCE_COUNT:
        return []

    score = CONSENSUS_STRONG_SCORE if conf >= 0.70 else CONSENSUS_MEDIUM_SCORE
    extra = (
        f"confidence={conf:.4f}|margin={margin:.4f}|"
        f"support_count={support_count}|source_count={source_count}|"
        f"sources={info.get('sources', '')}"
    )
    return [(val, "trusted_flight_source_weighted_consensus", score, extra)]


def generate_time_candidates(dirty_value, ctx, global_dict):
    """
    针对 Flights time-like 列补候选。
    改进点：
    1) 从 dirty value 中直接抽取 canonical time，作为强候选；
    2) 规则 expected value 仍由主流程加入，这里不重复；
    3) 同行时间上下文只作为中等候选；
    4) actual time 列降低 column dictionary 权重，避免把正确实际时间误改成高频时间。
    """
    cands = []
    dirty = canonical_empty_text(dirty_value)
    column = ctx.get("column", "")
    sem_type = ctx.get("semantic_type", "")

    if not looks_like_time_column(column, sem_type):
        return cands

    is_actual_col = looks_like_actual_time_column(column)

    # A. 从 dirty value 中抽取 canonical time：最强候选之一
    # 例如 12/02/2011 6:55 a.m. -> 6:55 a.m.; 9:32aDec 1 -> 9:32 a.m.
    dirty_fixed_list = normalize_all_time_candidates(dirty)
    for fixed in dirty_fixed_list:
        if fixed and norm_text(fixed) != norm_text(dirty):
            cands.append((fixed, "time_dirty_extract", 0.95, "extract_canonical_time_from_dirty"))

    # B. 如果 dirty 本身可解析但格式不统一，保留原有 pattern restore 语义
    fixed = normalize_time_candidate(dirty)
    if fixed and norm_text(fixed) != norm_text(dirty):
        cands.append((fixed, "time_pattern_restore", 0.92, "normalize_time_format"))

    # C. 从同行时间字段补候选。
    # 对 actual time，这类候选不宜过强；对 schedule time 可略强。
    row_values = ctx.get("row_values", {}) or {}
    for related_col in ["sched_dep_time", "act_dep_time", "sched_arr_time", "act_arr_time"]:
        if related_col == column:
            continue
        v = row_values.get(related_col, "")
        fixed_v_list = normalize_all_time_candidates(v)
        for fixed_v in fixed_v_list:
            if fixed_v and norm_text(fixed_v) != norm_text(dirty):
                if is_actual_col:
                    score = 0.30
                else:
                    score = 0.42
                cands.append((fixed_v, "time_row_context", score, f"from={related_col}"))

    # D. 当前列全局高频合法时间。
    # actual time 波动大，列内高频容易造成 FP，因此显著降权。
    col_counter = global_dict["column_value_counter"].get(column, Counter())
    for val, cnt in col_counter.most_common(20):
        fixed_val_list = normalize_all_time_candidates(val)
        for fixed_val in fixed_val_list:
            if fixed_val and norm_text(fixed_val) != norm_text(dirty):
                score = 0.20 if is_actual_col else 0.34
                cands.append((fixed_val, "time_column_dictionary", score, f"freq={cnt}"))

    return cands


def is_time_like_candidate(x):
    return normalize_time_candidate(x) is not None


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
            "extra_info": []
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
    elif source in {"measurecode_dictionary", "stateavg_from_measurecode", "condition_from_measurecode",
                    "measurename_from_measurecode", "measurename_from_condition"}:
        item["is_from_dictionary"] = 1
    elif source in {"pattern_restore_score", "pattern_restore_sample", "time_pattern_restore", "time_dirty_extract"}:
        item["is_from_pattern_restore"] = 1
    elif source in {"time_row_context", "time_column_dictionary"}:
        item["is_from_dictionary"] = 1
    elif source in {"trusted_flight_source_weighted_consensus", "flight_source_weighted_consensus", "flight_source_consensus"}:
        item["is_from_rule"] = 1
        item["is_from_dictionary"] = 1

    if extra is not None:
        item["extra_info"].append(str(extra))


# ============================================================
# 3. 输入读取
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
# 4. 构造字典（全局候选补全用）
# ============================================================

def build_global_dictionaries(context_records, dirty_table_df=None):
    """
    从 rich context 中尽量抽取全局 canonical dictionaries：
    - 列 top values
    - row_context.row_values
    - similarity_context.topk_similar_rows[*].row_values
    """
    column_value_counter = defaultdict(Counter)
    measurecode_to_names = defaultdict(Counter)
    condition_to_names = defaultdict(Counter)
    state_prefix_to_measurecodes = defaultdict(Counter)
    hospital_stateavg_from_code = defaultdict(Counter)

    def consume_row_values(row_vals):
        if not isinstance(row_vals, dict):
            return

        for c, v in row_vals.items():
            v = canonical_empty_text(v)
            if v != "":
                column_value_counter[str(c)][v] += 1

        mc = canonical_empty_text(row_vals.get("MeasureCode", ""))
        mn = canonical_empty_text(row_vals.get("MeasureName", ""))
        cond = canonical_empty_text(row_vals.get("Condition", ""))
        sa = canonical_empty_text(row_vals.get("Stateavg", ""))
        state = canonical_empty_text(row_vals.get("State", ""))
        hosp = canonical_empty_text(row_vals.get("HospitalName", ""))

        if mc and mn:
            measurecode_to_names[mc][mn] += 1
        if cond and mn:
            condition_to_names[cond][mn] += 1
        if state and mc:
            state_prefix_to_measurecodes[state.lower()][mc] += 1
        if hosp and mc and sa:
            hospital_stateavg_from_code[(hosp.lower(), mc)][sa] += 1

    for obj in context_records:
        # rich jsonl current row
        row_context = obj.get("row_context", {})
        if isinstance(row_context, dict):
            consume_row_values(row_context.get("row_values", {}))

        # similar rows
        sim_ctx = obj.get("similarity_context", {})
        if isinstance(sim_ctx, dict):
            topk = sim_ctx.get("topk_similar_rows", []) or []
            for item in topk:
                if isinstance(item, dict):
                    consume_row_values(item.get("row_values", {}))

        # column top values
        col_ctx = obj.get("column_context", {})
        if isinstance(col_ctx, dict):
            col = str(col_ctx.get("column", obj.get("column", "")))
            tops = col_ctx.get("top_values_with_counts", []) or []
            for t in tops:
                if isinstance(t, dict) and t.get("value", None) is not None:
                    val = canonical_empty_text(t.get("value"))
                    cnt = int(t.get("count", 1) or 1)
                    if col and val:
                        column_value_counter[col][val] += cnt

        # flat csv fallback
        if "column" in obj and "value" in obj:
            col = str(obj["column"])
            val = canonical_empty_text(obj["value"])
            if col and val:
                column_value_counter[col][val] += 1

    # optional whole dirty table
    if dirty_table_df is not None:
        for col in dirty_table_df.columns:
            for val in dirty_table_df[col].map(canonical_empty_text).tolist():
                if val != "":
                    column_value_counter[str(col)][val] += 1

        # structured dictionaries from dirty table rows
        for _, r in dirty_table_df.iterrows():
            row_vals = {c: canonical_empty_text(r[c]) for c in dirty_table_df.columns}
            consume_row_values(row_vals)

    global_dict = {
        "column_value_counter": column_value_counter,
        "measurecode_to_names": measurecode_to_names,
        "condition_to_names": condition_to_names,
        "state_prefix_to_measurecodes": state_prefix_to_measurecodes,
        "hospital_stateavg_from_code": hospital_stateavg_from_code,
    }
    return global_dict


# ============================================================
# 5. context 标准化
# ============================================================

def merge_flat_into_context_map(context_map, flat_records):
    """
    如果 rich jsonl 没有某些字段，可以用 flat csv 补摘要字段
    """
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
            value_str = canonical_empty_text(obj.get("value", ""))
            context_map[key] = {
                "row_id": row_id,
                "column": column,
                "value": value_str,
                "suggested_correct_value": canonical_empty_text(obj.get("suggested_correct_value", "")),
                "neighbor_majority_value": canonical_empty_text(obj.get("neighbor_majority_value", "")),
                "value_frequency": obj.get("value_frequency", None),
                "value_frequency_rank": obj.get("value_frequency_rank", None),
                "semantic_type": obj.get("semantic_type", None),
                "main_rule_type": obj.get("main_rule_type", None),
                "main_usage_role": obj.get("main_usage_role", None),
                "violation_count": obj.get("violation_count", None),
                "conflict_score": obj.get("conflict_score", None),
                "rule_expected_values": [],
                "column_top_values": [],
                "alternative_values_from_column": [],
                "similar_row_target_values": [],
                "row_values": {},
                "state_value": "",
                "hospital_name": "",
                "measure_code_value": "",
                "condition_value": "",
                "measure_name_value": "",
                "stateavg_value": "",
            }
        ctx = context_map[key]

        for field in [
            "suggested_correct_value", "neighbor_majority_value", "value_frequency",
            "value_frequency_rank", "semantic_type", "main_rule_type",
            "main_usage_role", "violation_count", "conflict_score"
        ]:
            if (ctx.get(field, None) in [None, "", []]) and (field in obj):
                val = obj[field]
                if isinstance(val, str):
                    val = val.strip()
                ctx[field] = val


def build_context_map_rich(context_records, global_dict):
    """
    richer jsonl -> unified context_map
    """
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

        # A. conflict_context.violated_rules_detailed[*].expected_value
        conflict_context = obj.get("conflict_context", {})
        if isinstance(conflict_context, dict):
            violated = conflict_context.get("violated_rules_detailed", []) or []
            for vr in violated:
                if isinstance(vr, dict):
                    ev = canonical_empty_text(vr.get("expected_value", None))
                    if ev not in ["", "empty"]:
                        rule_expected_values.append(ev)

            # expected_values_from_rules
            evs = conflict_context.get("expected_values_from_rules", []) or []
            for item in evs:
                if isinstance(item, dict):
                    ev = canonical_empty_text(item.get("value", None))
                    if ev not in ["", "empty"]:
                        rule_expected_values.append(ev)

            # alternative_values_from_column
            avs = conflict_context.get("alternative_values_from_column", []) or []
            for item in avs:
                if isinstance(item, dict):
                    val = canonical_empty_text(item.get("value", None))
                    if val not in ["", "empty"]:
                        alternative_values_from_column.append({
                            "value": val,
                            "count": int(item.get("count", 1) or 1),
                            "ratio": 0.0
                        })

            # all_alternative_values 里也可能混有 column_top / expected value
            all_avs = conflict_context.get("all_alternative_values", []) or []
            for item in all_avs:
                if isinstance(item, dict):
                    val = canonical_empty_text(item.get("value", None))
                    if val not in ["", "empty"]:
                        alternative_values_from_column.append({
                            "value": val,
                            "count": int(item.get("count", 1) or 1),
                            "ratio": 0.0
                        })

        # B. suggested_correct_value
        suggested_correct_value = canonical_empty_text(obj.get("suggested_correct_value", ""))
        if suggested_correct_value == "empty":
            suggested_correct_value = ""

        # C. neighbor majority
        neighbor_majority_value = ""
        sim_ctx = obj.get("similarity_context", {})
        if isinstance(sim_ctx, dict):
            neighbor_majority_value = canonical_empty_text(sim_ctx.get("neighbor_majority_value", ""))
            topk_similar = sim_ctx.get("topk_similar_rows", []) or []
            for item in topk_similar[:TOP_K_SIMILAR_ROW_VALUES]:
                if isinstance(item, dict):
                    tv = canonical_empty_text(item.get("target_column_value", ""))
                    if tv not in ["", "empty"]:
                        similar_row_target_values.append(tv)

        # D. column top values
        column_top_values = []
        col_ctx = obj.get("column_context", {})
        if isinstance(col_ctx, dict):
            top_vals = col_ctx.get("top_values_with_counts", []) or []
            for item in top_vals[:TOP_K_COLUMN_VALUES]:
                if isinstance(item, dict) and item.get("value", None) is not None:
                    val = canonical_empty_text(item["value"])
                    if val not in ["", "empty"]:
                        column_top_values.append({
                            "value": val,
                            "count": int(item.get("count", 0) or 0),
                            "ratio": safe_float(item.get("ratio", 0.0), 0.0)
                        })

        # fallback: if no top vals, use global counter
        if not column_top_values:
            counter = column_value_counter.get(column, Counter())
            total = max(sum(counter.values()), 1)
            for val, cnt in counter.most_common(TOP_K_COLUMN_VALUES):
                column_top_values.append({"value": val, "count": cnt, "ratio": cnt / total})

        # row_context.row_values
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
            "suggested_correct_value": suggested_correct_value,
            "neighbor_majority_value": neighbor_majority_value,
            "value_frequency": obj.get("value_frequency", None),
            "value_frequency_rank": obj.get("value_frequency_rank", None),
            "semantic_type": obj.get("semantic_type", None),
            "main_rule_type": obj.get("main_rule_type", None),
            "main_usage_role": obj.get("main_usage_role", None),
            "violation_count": obj.get("violation_count", None),
            "conflict_score": obj.get("conflict_score", None),
            "rule_expected_values": rule_expected_values,
            "column_top_values": column_top_values,
            "alternative_values_from_column": alternative_values_from_column,
            "similar_row_target_values": similar_row_target_values,
            "row_values": row_values,
            "state_value": canonical_empty_text(row_values.get("State", "")),
            "hospital_name": canonical_empty_text(row_values.get("HospitalName", "")),
            "measure_code_value": canonical_empty_text(row_values.get("MeasureCode", "")),
            "condition_value": canonical_empty_text(row_values.get("Condition", "")),
            "measure_name_value": canonical_empty_text(row_values.get("MeasureName", "")),
            "stateavg_value": canonical_empty_text(row_values.get("Stateavg", "")),
        }

    return context_map


# ============================================================
# 6. 读取错误位置
# ============================================================

pred_df = pd.read_csv(ERROR_PRED_FILE, encoding="utf-8-sig")

required_base = {"row_id", "column"}
if not required_base.issubset(set(pred_df.columns)):
    raise ValueError(f"错误位置文件必须至少包含 {required_base}")

pred_df["row_id"] = pred_df["row_id"].astype(int)

if "final_pred_label" in pred_df.columns:
    error_df = pred_df[pred_df["final_pred_label"] == 1].copy()
elif "final_pred_label_name" in pred_df.columns:
    error_df = pred_df[pred_df["final_pred_label_name"].astype(str).str.lower() == "error"].copy()
else:
    error_df = pred_df.copy()

keep_cols = ["row_id", "column"]
if "value" in error_df.columns:
    keep_cols.append("value")

error_df = error_df[keep_cols].drop_duplicates().copy()
print(f"[INFO] 预测错误单元格数量: {len(error_df)}")


# ============================================================
# 7. 读取上下文 + 构造 context_map
# ============================================================

context_records = load_context_records(CONTEXT_FILE)
print(f"[INFO] 主上下文记录数: {len(context_records)}")

dirty_table_df = None
if str(DIRTY_TABLE_CSV).strip():
    if path_exists_nonempty(DIRTY_TABLE_CSV):
        dirty_table_df = pd.read_csv(DIRTY_TABLE_CSV, encoding="utf-8-sig")
        dirty_table_df = dirty_table_df.apply(lambda col: col.map(canonical_empty_text))
        print(f"[INFO] 已读取原始脏表: {DIRTY_TABLE_CSV}, shape={dirty_table_df.shape}")
    else:
        print(f"[WARN] DIRTY_TABLE_CSV 不存在，已跳过: {DIRTY_TABLE_CSV}")

global_dict = build_global_dictionaries(context_records, dirty_table_df=dirty_table_df)

flight_consensus_map = {}
flight_consensus_df = pd.DataFrame()
consensus_suspicious_df = pd.DataFrame()
if ENABLE_FLIGHT_CONSENSUS:
    flight_consensus_map, flight_consensus_df, consensus_suspicious_df = build_flight_time_consensus(dirty_table_df)
    print(f"[INFO] flight consensus entries: {len(flight_consensus_df)}")
    print(f"[INFO] consensus suspicious cells: {len(consensus_suspicious_df)}")
    if not flight_consensus_df.empty:
        print("[INFO] trusted consensus by column:")
        print(
            flight_consensus_df.groupby("column")["is_high_confidence"]
            .agg(["count", "sum"])
            .to_string()
        )
    if not consensus_suspicious_df.empty:
        print("[INFO] trusted consensus suspicious by column:")
        print(consensus_suspicious_df["column"].value_counts().to_string())

context_map = build_context_map_rich(context_records, global_dict)

if str(FALLBACK_CONTEXT_FLAT_FILE).strip():
    if path_exists_nonempty(FALLBACK_CONTEXT_FLAT_FILE):
        flat_records = load_context_records(FALLBACK_CONTEXT_FLAT_FILE)
        print(f"[INFO] fallback flat 上下文记录数: {len(flat_records)}")
        merge_flat_into_context_map(context_map, flat_records)
    else:
        print(f"[WARN] FALLBACK_CONTEXT_FLAT_FILE 不存在，已跳过: {FALLBACK_CONTEXT_FLAT_FILE}")

print(f"[INFO] 可索引上下文条数: {len(context_map)}")

# ============================================================
# 7.5 用 high-confidence flight consensus 扩展允许修复范围
# ============================================================

if ENABLE_FLIGHT_CONSENSUS and not consensus_suspicious_df.empty:
    base_allowed = error_df[["row_id", "column"]].copy()
    if "value" in error_df.columns:
        base_allowed["value"] = error_df["value"]

    add_allowed = consensus_suspicious_df[["row_id", "column", "value"]].copy()
    before_allowed = len(base_allowed.drop_duplicates(subset=["row_id", "column"]))

    error_df = pd.concat([base_allowed, add_allowed], ignore_index=True)
    error_df["row_id"] = pd.to_numeric(error_df["row_id"], errors="raise").astype(int)
    error_df["column"] = error_df["column"].astype(str).str.strip()
    error_df = error_df.drop_duplicates(subset=["row_id", "column"], keep="first").reset_index(drop=True)

    after_allowed = len(error_df)
    print(f"[INFO] consensus 扩展允许修复范围: {before_allowed} -> {after_allowed} (+{after_allowed - before_allowed})")


# ============================================================
# 8. 列专属补候选
# ============================================================

def generate_measurecode_candidates(dirty_value, ctx, global_dict):
    cands = []
    dirty = canonical_empty_text(dirty_value)

    # 1) known dictionary
    for code in KNOWN_MEASURE_CODES:
        sim = string_similarity(dirty, code)
        if sim >= MIN_STRING_SIM_FOR_CANONICAL or normalize_measurecode_like(dirty) == normalize_measurecode_like(code):
            cands.append((code, "measurecode_dictionary", 0.82, f"sim={sim:.4f}"))

    # 2) column frequent values
    col_counter = global_dict["column_value_counter"].get("MeasureCode", Counter())
    for val, cnt in col_counter.most_common(20):
        if looks_like_measure_code(val):
            sim = string_similarity(dirty, val)
            if sim >= MIN_STRING_SIM_FOR_CANONICAL:
                cands.append((val, "measurecode_dictionary", 0.78, f"freq={cnt}|sim={sim:.4f}"))

    return cands


def generate_stateavg_candidates(dirty_value, ctx, global_dict, existing_candidate_dict):
    cands = []
    dirty = canonical_empty_text(dirty_value)
    state_prefix = ctx.get("state_value", "").lower()

    if not state_prefix:
        # 尝试从 dirty 本身抽 prefix
        state_prefix = extract_state_prefix_from_stateavg(dirty)

    measurecode_candidates = []
    # 从已有候选里拿 measurecode
    if ctx.get("measure_code_value", "") and looks_like_measure_code(ctx["measure_code_value"]):
        measurecode_candidates.append(ctx["measure_code_value"])
    for item in existing_candidate_dict.values():
        v = canonical_empty_text(item["candidate_value"])
        if looks_like_measure_code(v):
            measurecode_candidates.append(v)

    # 从全局字典拿
    for v, _ in global_dict["state_prefix_to_measurecodes"].get(state_prefix, Counter()).most_common(10):
        measurecode_candidates.append(v)

    measurecode_candidates = list(dict.fromkeys([canonical_empty_text(x) for x in measurecode_candidates if x]))

    for mc in measurecode_candidates:
        if state_prefix:
            sa = f"{state_prefix}_{canonical_empty_text(mc)}"
            sim = string_similarity(dirty, sa)
            if sim >= 0.25 or dirty == "" or dirty == "empty":
                cands.append((sa, "stateavg_from_measurecode", 0.84, f"state={state_prefix}|mc={mc}|sim={sim:.4f}"))

    return cands


def generate_condition_candidates(dirty_value, ctx):
    cands = []
    dirty = canonical_empty_text(dirty_value)
    mc = ctx.get("measure_code_value", "")
    fam = measure_family_name(mc)

    if fam:
        sim = string_similarity(dirty, fam)
        cands.append((fam, "condition_from_measurecode", 0.86, f"mc={mc}|sim={sim:.4f}"))

    for cond in CONDITION_CANONICAL:
        sim = string_similarity(dirty, cond)
        if sim >= MIN_STRING_SIM_FOR_CANONICAL or fam == cond:
            cands.append((cond, "condition_from_measurecode", 0.76, f"sim={sim:.4f}"))

    return cands


def generate_measurename_candidates(dirty_value, ctx, global_dict, existing_candidate_dict):
    cands = []
    dirty = canonical_empty_text(dirty_value)

    mc = canonical_empty_text(ctx.get("measure_code_value", ""))
    cond = canonical_empty_text(ctx.get("condition_value", ""))

    # 1) from measurecode dictionary
    if mc:
        for name, cnt in global_dict["measurecode_to_names"].get(mc, Counter()).most_common(10):
            sim = string_similarity(dirty, name)
            cands.append((name, "measurename_from_measurecode", 0.88, f"mc={mc}|cnt={cnt}|sim={sim:.4f}"))

    # 2) from condition dictionary
    if cond:
        for name, cnt in global_dict["condition_to_names"].get(cond, Counter()).most_common(10):
            sim = string_similarity(dirty, name)
            cands.append((name, "measurename_from_condition", 0.82, f"cond={cond}|cnt={cnt}|sim={sim:.4f}"))

    # 3) from measurecode candidates already generated
    for item in existing_candidate_dict.values():
        v = canonical_empty_text(item["candidate_value"])
        if looks_like_measure_code(v):
            for name, cnt in global_dict["measurecode_to_names"].get(v, Counter()).most_common(6):
                sim = string_similarity(dirty, name)
                cands.append((name, "measurename_from_measurecode", 0.80, f"mc_candidate={v}|cnt={cnt}|sim={sim:.4f}"))

    return cands


def generate_score_candidates(dirty_value):
    cands = []
    dirty = canonical_empty_text(dirty_value)
    fixed = normalize_percent_candidate(dirty)
    if fixed and fixed != dirty:
        cands.append((fixed, "pattern_restore_score", 0.90, "pattern_restore"))
    return cands


def generate_sample_candidates(dirty_value):
    cands = []
    dirty = canonical_empty_text(dirty_value)
    fixed = normalize_sample_candidate(dirty)
    if fixed and fixed != dirty:
        cands.append((fixed, "pattern_restore_sample", 0.90, "pattern_restore"))
    return cands


# ============================================================
# 9. 为每个错误 cell 构造候选修复值
# ============================================================

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
            "top_candidate_score": None
        })
        continue

    dirty_value = ctx.get("value", "")
    if ("value" in error_df.columns) and (pd.notna(row.get("value", None))) and str(row.get("value")).strip() != "":
        dirty_value = canonical_empty_text(row["value"])

    candidate_dict = {}

    if INCLUDE_DIRTY_VALUE and dirty_value != "":
        add_candidate(candidate_dict, dirty_value, "dirty_value", 0.01)

    # -------------------------------------------------
    # A0. Flights 同一 flight 多 source 共识候选
    # -------------------------------------------------
    if ENABLE_FLIGHT_CONSENSUS and looks_like_time_column(column, ctx.get("semantic_type", None)):
        for val, source, score, extra in generate_flight_consensus_candidates(row_id, column, dirty_value, flight_consensus_map):
            add_candidate(candidate_dict, val, source, score, extra=extra)

    # -------------------------------------------------
    # A. 规则 expected values（rich jsonl 主力）
    # -------------------------------------------------
    expected_counter = Counter()
    for ev in ctx.get("rule_expected_values", []):
        ev = canonical_empty_text(ev)
        if ev == "" or norm_text(ev) == norm_text(dirty_value):
            continue
        expected_counter[ev] += 1

    total_votes = sum(expected_counter.values())
    for ev, cnt in expected_counter.items():
        score = 0.74 + 0.22 * (cnt / max(total_votes, 1))
        add_candidate(candidate_dict, ev, "expected_values_from_rules", score, extra=f"rule_vote_count={cnt}")

    # -------------------------------------------------
    # B. 邻居 / 相似行 target values
    # -------------------------------------------------
    neighbor_majority_value = ctx.get("neighbor_majority_value", None)
    if neighbor_majority_value and norm_text(neighbor_majority_value) != norm_text(dirty_value):
        add_candidate(candidate_dict, neighbor_majority_value, "neighbor_majority", 0.88)

    sim_counter = Counter()
    for v in ctx.get("similar_row_target_values", []):
        v = canonical_empty_text(v)
        if v and norm_text(v) != norm_text(dirty_value):
            sim_counter[v] += 1
    for v, cnt in sim_counter.items():
        score = 0.76 + 0.10 * min(cnt, TOP_K_SIMILAR_ROW_VALUES) / max(TOP_K_SIMILAR_ROW_VALUES, 1)
        add_candidate(candidate_dict, v, "similar_row_target", score, extra=f"count={cnt}")

    # -------------------------------------------------
    # C. suggested_correct_value
    # -------------------------------------------------
    suggested_correct_value = ctx.get("suggested_correct_value", None)
    if suggested_correct_value and norm_text(suggested_correct_value) != norm_text(dirty_value):
        add_candidate(candidate_dict, suggested_correct_value, "suggested_correct_value", 0.93)

    # -------------------------------------------------
    # D. 列内 top values / 替代值
    # -------------------------------------------------
    alt_items = []
    alt_items.extend(ctx.get("column_top_values", [])[:TOP_K_COLUMN_VALUES])
    alt_items.extend(ctx.get("alternative_values_from_column", [])[:TOP_K_COLUMN_VALUES])

    dedup_alt = {}
    for item in alt_items:
        if not isinstance(item, dict):
            continue
        val = canonical_empty_text(item.get("value", ""))
        if val == "" or norm_text(val) == norm_text(dirty_value):
            continue
        k = norm_text(val)
        prev = dedup_alt.get(k, {"value": val, "count": 0, "ratio": 0.0})
        prev["count"] = max(prev["count"], int(item.get("count", 0) or 0))
        prev["ratio"] = max(prev["ratio"], safe_float(item.get("ratio", 0.0), 0.0))
        dedup_alt[k] = prev

    for item in dedup_alt.values():
        val = item["value"]
        ratio = safe_float(item.get("ratio", 0.0), 0.0)
        sim = string_similarity(dirty_value, val)

        # actual time 列波动大，列内高频值容易造成误修，降低其权重
        if looks_like_actual_time_column(column):
            top_score = 0.18 + 0.08 * min(ratio, 1.0)
        else:
            top_score = 0.34 + 0.18 * min(ratio, 1.0)
        add_candidate(candidate_dict, val, "column_top_value", top_score, extra=f"ratio={ratio:.4f}")

        if sim >= MIN_STRING_SIM_FOR_COLUMN_VALUE:
            if looks_like_actual_time_column(column):
                sim_score = 0.34 + 0.14 * sim + 0.06 * min(ratio, 1.0)
            else:
                sim_score = 0.48 + 0.22 * sim + 0.12 * min(ratio, 1.0)
            add_candidate(candidate_dict, val, "similar_column_value", sim_score, extra=f"ratio={ratio:.4f}|sim={sim:.4f}")

    # -------------------------------------------------
    # E. 列专属补候选
    # -------------------------------------------------
    if ENABLE_COLUMN_SPECIFIC_GENERATION:
        if column == "MeasureCode":
            for val, source, score, extra in generate_measurecode_candidates(dirty_value, ctx, global_dict):
                add_candidate(candidate_dict, val, source, score, extra=extra)

        elif column == "Stateavg":
            for val, source, score, extra in generate_stateavg_candidates(dirty_value, ctx, global_dict, candidate_dict):
                add_candidate(candidate_dict, val, source, score, extra=extra)

        elif column == "Condition":
            for val, source, score, extra in generate_condition_candidates(dirty_value, ctx):
                add_candidate(candidate_dict, val, source, score, extra=extra)

        elif column == "MeasureName":
            for val, source, score, extra in generate_measurename_candidates(dirty_value, ctx, global_dict, candidate_dict):
                add_candidate(candidate_dict, val, source, score, extra=extra)

        elif column == "Score":
            for val, source, score, extra in generate_score_candidates(dirty_value):
                add_candidate(candidate_dict, val, source, score, extra=extra)

        elif column == "Sample":
            for val, source, score, extra in generate_sample_candidates(dirty_value):
                add_candidate(candidate_dict, val, source, score, extra=extra)

        # Flights 等数据集的通用 time-like 列候选补全
        if looks_like_time_column(column, ctx.get("semantic_type", None)):
            for val, source, score, extra in generate_time_candidates(dirty_value, ctx, global_dict):
                add_candidate(candidate_dict, val, source, score, extra=extra)

    # -------------------------------------------------
    # F. 去掉与脏值完全相同的候选
    # -------------------------------------------------
    remove_keys = []
    for k, item in candidate_dict.items():
        if norm_text(item["candidate_value"]) == norm_text(dirty_value):
            remove_keys.append(k)
    for k in remove_keys:
        candidate_dict.pop(k, None)

    # -------------------------------------------------
    # G. 最终打分与排序
    # -------------------------------------------------
    final_candidates = []
    for _, item in candidate_dict.items():
        cand = item["candidate_value"]
        sim = string_similarity(dirty_value, cand)
        source_count = len(item["candidate_sources"])
        support_bonus = 0.04 * max(source_count - 1, 0)

        # structured semantic bonus
        semantic_bonus = 0.0
        if column == "MeasureCode" and looks_like_measure_code(cand):
            semantic_bonus += 0.08
        if column == "Stateavg" and looks_like_stateavg(cand):
            semantic_bonus += 0.08
        if column == "Condition" and cand.lower() in CONDITION_CANONICAL:
            semantic_bonus += 0.07
        if column == "Score" and normalize_percent_candidate(cand) == cand:
            semantic_bonus += 0.07
        if column == "Sample" and normalize_sample_candidate(cand) == cand:
            semantic_bonus += 0.07
        if column == "MeasureName":
            mc = ctx.get("measure_code_value", "")
            fam = measure_family_name(mc)
            if fam and fam in cand.lower():
                semantic_bonus += 0.05
        if looks_like_time_column(column, ctx.get("semantic_type", None)) and is_time_like_candidate(cand):
            semantic_bonus += 0.06
            if ("trusted_flight_source_weighted_consensus" in item["candidate_sources"]) or ("flight_source_weighted_consensus" in item["candidate_sources"]):
                semantic_bonus += 0.20
            if "time_dirty_extract" in item["candidate_sources"]:
                semantic_bonus += 0.10
            if looks_like_actual_time_column(column) and "time_column_dictionary" in item["candidate_sources"]:
                semantic_bonus -= 0.08

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
            x["candidate_value"]
        )
    )[:MAX_CANDIDATES_PER_CELL]

    # -------------------------------------------------
    # H. 展开输出
    # -------------------------------------------------
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
            "is_from_flight_consensus": int(("trusted_flight_source_weighted_consensus" in item["candidate_sources"]) or ("flight_source_weighted_consensus" in item["candidate_sources"])),
            "flight_consensus_value": flight_consensus_map.get((row_id, column), {}).get("candidate_value", ""),
            "flight_consensus_confidence": flight_consensus_map.get((row_id, column), {}).get("confidence", 0.0),
            "flight_consensus_margin": flight_consensus_map.get((row_id, column), {}).get("margin", 0.0),
            "flight_consensus_support_count": flight_consensus_map.get((row_id, column), {}).get("support_count", 0),
            "flight_consensus_source_count": flight_consensus_map.get((row_id, column), {}).get("source_count", 0),
            "flight_consensus_sources": flight_consensus_map.get((row_id, column), {}).get("sources", ""),
            "violation_count": ctx.get("violation_count", None),
            "conflict_score": ctx.get("conflict_score", None),
            "semantic_type": ctx.get("semantic_type", None),
            "main_rule_type": ctx.get("main_rule_type", None),
            "main_usage_role": ctx.get("main_usage_role", None),
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


# ============================================================
# 10. 保存结果
# ============================================================

expanded_df = pd.DataFrame(expanded_rows)
grouped_df = pd.DataFrame(grouped_rows)

if not expanded_df.empty:
    expanded_df = expanded_df.sort_values(["row_id", "column", "candidate_rank"]).reset_index(drop=True)

if not grouped_df.empty:
    grouped_df = grouped_df.sort_values(["row_id", "column"]).reset_index(drop=True)

OUTPUT_CANDIDATES_EXPANDED_CSV = resolve_output_path(OUTPUT_CANDIDATES_EXPANDED_CSV, ERROR_PRED_FILE)
OUTPUT_CANDIDATES_GROUPED_CSV = resolve_output_path(OUTPUT_CANDIDATES_GROUPED_CSV, ERROR_PRED_FILE)
OUTPUT_FLIGHT_CONSENSUS_CSV = resolve_output_path(OUTPUT_FLIGHT_CONSENSUS_CSV, ERROR_PRED_FILE)
OUTPUT_CONSENSUS_SUSPICIOUS_CELLS_CSV = resolve_output_path(OUTPUT_CONSENSUS_SUSPICIOUS_CELLS_CSV, ERROR_PRED_FILE)
OUTPUT_AUGMENTED_ALLOWED_CELLS_CSV = resolve_output_path(OUTPUT_AUGMENTED_ALLOWED_CELLS_CSV, ERROR_PRED_FILE)

expanded_df.to_csv(OUTPUT_CANDIDATES_EXPANDED_CSV, index=False, encoding="utf-8-sig")
grouped_df.to_csv(OUTPUT_CANDIDATES_GROUPED_CSV, index=False, encoding="utf-8-sig")

if ENABLE_FLIGHT_CONSENSUS:
    flight_consensus_df.to_csv(OUTPUT_FLIGHT_CONSENSUS_CSV, index=False, encoding="utf-8-sig")
    consensus_suspicious_df.to_csv(OUTPUT_CONSENSUS_SUSPICIOUS_CELLS_CSV, index=False, encoding="utf-8-sig")
    error_df.to_csv(OUTPUT_AUGMENTED_ALLOWED_CELLS_CSV, index=False, encoding="utf-8-sig")

print(f"[INFO] 缺失上下文的错误 cell 数量: {missing_context_count}")
print(f"[INFO] 候选展开文件已保存: {OUTPUT_CANDIDATES_EXPANDED_CSV}")
print(f"[INFO] 候选汇总文件已保存: {OUTPUT_CANDIDATES_GROUPED_CSV}")
if ENABLE_FLIGHT_CONSENSUS:
    print(f"[INFO] flight consensus 文件已保存: {OUTPUT_FLIGHT_CONSENSUS_CSV}")
    print(f"[INFO] consensus suspicious cells 已保存: {OUTPUT_CONSENSUS_SUSPICIOUS_CELLS_CSV}")
    print(f"[INFO] consensus augmented allowed cells 已保存: {OUTPUT_AUGMENTED_ALLOWED_CELLS_CSV}")

if not grouped_df.empty:
    print("\n[INFO] 前 20 个错误 cell 的候选示例：")
    print(grouped_df.head(20).to_string(index=False))

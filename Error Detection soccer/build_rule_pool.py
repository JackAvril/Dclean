import json
import math
import re
from collections import Counter
from itertools import combinations
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd


# ============================================================
# 1. 用户配置区
# ============================================================

INPUT_CSV = "/mnt/mydata/dq/projects/Splittree/soccer/soccer_dirty.csv"
OUTPUT_JSON = "rule_pool_soccer.json"

MISSING_TOKENS = {
    "", "empty", "null", "none", "nan", "na", "n/a", "missing", "unk",
    "unknown", "?", "not available", "contact airline"
}

NUMERIC_RATIO_THRESHOLD = 0.90
ENUM_UNIQUE_THRESHOLD = 120
ENUM_RATIO_THRESHOLD = 0.12

ID_LIKE_UNIQUE_RATIO_THRESHOLD = 0.98
CODE_LIKE_AVG_LEN_THRESHOLD = 35
CODE_LIKE_DOMINANT_PATTERN_THRESHOLD = 0.70
DOMINANT_PATTERN_THRESHOLD = 0.70
DOMINANT_TYPO_MIN_RATIO = 0.50

MAX_LHS_SIZE = 2

# FD/soft-FD 是本轮结果中 FP 激增的主要来源。
# 因此强 FD 只保留极高置信、较大支持的关系；soft-FD 只输出为 evidence_only，
# 不允许直接作为候选生成规则。
FD_CONFIDENCE_THRESHOLD = 0.995
MIN_SUPPORT_ROWS = 50
FD_MIN_GROUP_SIZE = 10
FD_MIN_DOMINANT_RATIO = 0.98
FD_MIN_DOMINANT_MARGIN = 5

SOFT_FD_CONFIDENCE_THRESHOLD = 0.90
SOFT_FD_MIN_GROUP_SIZE = 20
SOFT_FD_MIN_SUPPORT_ROWS = 200
SOFT_FD_USAGE_ROLE = "evidence_only"

PAIR_CONTEXT_MIN_RATIO = 0.98
PAIR_CONTEXT_MIN_GROUP_SIZE = 20

HIGH_CARDINALITY_RATIO_FOR_LHS = 0.95
ALLOW_INDEX_COLUMN = False
SKIP_SCHEMA_FOR_ID_LIKE = True

# ------------------------------------------------------------
# soccer 数据集字段
# ------------------------------------------------------------
SOCCER_COLUMNS = [
    "name",
    "surname",
    "birthyear",
    "birthplace",
    "position",
    "team",
    "city",
    "stadium",
    "season",
    "manager",
]

# ------------------------------------------------------------
# Soccer 候选生成策略
# ------------------------------------------------------------
# Soccer 数据集中，name/surname/manager/team/stadium 等文本字段可能存在格式噪声，
# birthyear/season 存在数值范围与年龄一致性约束，position 是强 domain 字段。
# 这里采用 schema-aware 但尽量不过度数据集后验调参的设计：
# 1) PRIMARY_TARGET_COLUMNS 是主要检测列；
# 2) CONTEXT_ONLY_COLUMNS 主要用于上下文证据，不使用 rare/global dominant 大规模造候选；
# 3) team-season/team/stadium/city/manager 等关系用 mined context/FD 作为候选证据。
PRIMARY_TARGET_COLUMNS = set(SOCCER_COLUMNS)

# Soccer 本轮修正：不再把 name/surname 排除在候选目标外。
# 上一版评估中 name/surname/birthplace 几乎没有进入候选，导致候选召回上限被锁死。
# 这里所有列都允许由 schema/missing/format 强规则进入候选；
# 但 FD / context / rare / global-dominant 仍不能把 name/surname 这类长尾身份列大规模拉入候选。
CONTEXT_ONLY_COLUMNS = set()

SCHEMA_RULE_COLUMNS = set(SOCCER_COLUMNS)
TARGET_CANDIDATE_COLUMNS = set(PRIMARY_TARGET_COLUMNS)
HIGH_RISK_MISSING_COLUMNS = set(SOCCER_COLUMNS)

# Soccer domain
VALID_POSITIONS = {
    "Goalkeeper",
    "Defender",
    "Midfield",
    "Midfielder",
    "Forward",
    "Striker",
    "Winger",
}

POSITION_CANONICAL_MAP = {
    "Midfielder": "Midfield",
}

# 年龄/赛季合理范围。Soccer 中 season 是比赛/赛季年份，birthyear 是出生年份。
BIRTHYEAR_MIN = 1940
BIRTHYEAR_MAX = 2010
SEASON_MIN = 1900
SEASON_MAX = 2035
PLAYER_MIN_AGE_AT_SEASON = 15
PLAYER_MAX_AGE_AT_SEASON = 50

# pattern 规则：下划线是该数据集中常见分隔符。
PERSON_NAME_REGEX = r"^[A-Za-z]+(?:_[A-Za-z]+)*$"
SURNAME_REGEX = r"^[A-Za-z]+(?:[-_'][A-Za-z]+)*$"
TEAM_REGEX = r"^[A-Za-z0-9]+(?:[._'\-][A-Za-z0-9]+)*$"
LOCATION_REGEX = r"^[A-Za-z]+(?:[ _'\-][A-Za-z]+)*$"
STADIUM_REGEX = r"^[A-Za-z0-9]+(?:[ _'\-][A-Za-z0-9]+)*$"

# 弱规则控制
ENABLE_RARE_VALUE_RULE = False
RARE_VALUE_MAX_FREQ = 2
SKIP_RARE_VALUE_COLUMNS = set(SOCCER_COLUMNS)

ENABLE_RARE_PATTERN_RULE = False
RARE_PATTERN_MAX_RATIO = 0.02
SKIP_RARE_PATTERN_COLUMNS = set(SOCCER_COLUMNS)

# global dominant 容易把多数队伍/城市/球场误当规则，默认关闭。
SKIP_GLOBAL_DOMINANT_COLUMNS = set(SOCCER_COLUMNS)

# Generic pattern 规则只用于明显格式字段。数值/position 用专门规则。
SKIP_GENERIC_PATTERN_COLUMNS = {
    "birthyear",
    "position",
    "season",
}

# FD/soft-FD 的 RHS 只允许指向相对结构化的非身份列。
# name/surname 的召回依靠 missing/pattern，不依靠统计相关性。
FD_RHS_CANDIDATE_COLUMNS = {
    "birthplace",
    "position",
    "team",
    "city",
    "stadium",
    "season",
    "manager",
}
EXTRA_SKIP_FD_LHS_COLUMNS = {"name", "surname", "manager"}

# team-season 上下文 dominant 规则。通常同一 team + season 的 city/stadium/manager 更稳定。
TEAM_SEASON_CONTEXT_LHS_SPECS = [
    ["team", "season"],
    ["team"],
    ["stadium"],
    ["city", "stadium"],
]
TEAM_CONTEXT_TARGETS = ["city", "stadium", "manager"]
# context dominant 同样不能过宽。上一版 team-context 虽然比 soft-FD 少，
# 但仍应当作为弱证据而不是最终判断。
TEAM_CONTEXT_MIN_GROUP_SIZE = 20
TEAM_CONTEXT_MIN_RATIO = 0.98
TEAM_CONTEXT_MIN_DOMINANT_MARGIN = 5
TEAM_CONTEXT_USAGE_ROLE = "evidence_only"

# birthyear/season 年龄一致性
ENABLE_BIRTHYEAR_SEASON_AGE_RULE = True


# ============================================================
# 2. 基础函数
# ============================================================

def normalize_value(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.lower() in MISSING_TOKENS:
        return None
    return s


def normalize_series(s):
    return s.map(normalize_value)


def maybe_drop_unnamed_columns(df):
    keep_cols = [c for c in df.columns if not str(c).lower().startswith("unnamed:")]
    return df[keep_cols].copy()


def detect_basic_pattern(value):
    if value is None:
        return "NULL"
    parts, prev_type, cnt = [], None, 0
    for ch in str(value):
        if ch.isdigit():
            cur = "D"
        elif ch.isalpha():
            cur = "L"
        elif ch in {" ", "-", "_", "/", ".", ",", "(", ")", "%", ":", ";", "&", "'", "+", "<", ">"}:
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


def try_parse_numeric(series):
    cleaned = series.dropna().astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False)
    numeric = pd.to_numeric(cleaned, errors="coerce")
    success_ratio = numeric.notna().mean() if len(cleaned) > 0 else 0.0
    return numeric, success_ratio


def parse_int_like(x: Optional[str]) -> Optional[int]:
    if x is None:
        return None
    s = str(x).strip().replace(",", "")
    if re.fullmatch(r"[+-]?\d+(\.0+)?", s):
        try:
            return int(float(s))
        except Exception:
            return None
    return None


def parse_year(value: Optional[str]) -> Optional[int]:
    n = parse_int_like(value)
    if n is None:
        return None
    return n


def canonical_position(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if s in POSITION_CANONICAL_MAP:
        return POSITION_CANONICAL_MAP[s]
    if s in VALID_POSITIONS:
        return s
    return None


def is_regex_match(value: Optional[str], pattern: str) -> bool:
    if value is None:
        return True
    return bool(re.fullmatch(pattern, str(value).strip()))


def shannon_entropy(series):
    vals = series.dropna().tolist()
    if not vals:
        return 0.0
    cnt = Counter(vals)
    total = len(vals)
    ent = 0.0
    for c in cnt.values():
        p = c / total
        ent -= p * math.log2(p)
    return ent


def value_frequency_dict(series, topk=20):
    cnt = Counter(series.dropna().tolist())
    total = sum(cnt.values()) if cnt else 0
    out = []
    for k, v in cnt.most_common(topk):
        out.append({"value": k, "count": v, "ratio": round(v / total, 6) if total > 0 else 0.0})
    return out


def is_index_like_column(series):
    s = series.dropna()
    if len(s) == 0:
        return False
    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.notna().mean() < 0.95:
        return False
    unique_ratio = s.nunique() / len(s)
    return unique_ratio > 0.98


def is_id_like_column(non_null, unique_ratio, numeric_ratio, avg_len):
    if len(non_null) == 0:
        return False
    if unique_ratio >= ID_LIKE_UNIQUE_RATIO_THRESHOLD and avg_len <= CODE_LIKE_AVG_LEN_THRESHOLD:
        return True
    if unique_ratio >= ID_LIKE_UNIQUE_RATIO_THRESHOLD and numeric_ratio >= NUMERIC_RATIO_THRESHOLD:
        return True
    return False


def is_code_like_column(non_null, unique_ratio, numeric_ratio, avg_len, top_patterns):
    if len(non_null) == 0:
        return False
    dominant_ratio = top_patterns[0]["ratio"] if top_patterns else 0.0
    if avg_len <= CODE_LIKE_AVG_LEN_THRESHOLD and dominant_ratio >= CODE_LIKE_DOMINANT_PATTERN_THRESHOLD:
        if unique_ratio > ENUM_RATIO_THRESHOLD:
            return True
    if numeric_ratio >= 0.95 and avg_len <= 15 and unique_ratio < ID_LIKE_UNIQUE_RATIO_THRESHOLD:
        return True
    return False


# ============================================================
# 3. 规则统计函数
# ============================================================

def calc_fd_confidence_and_support(df, lhs_cols, rhs_col):
    sub = df[lhs_cols + [rhs_col]].dropna()
    support = len(sub)
    if support < MIN_SUPPORT_ROWS:
        return 0.0, support
    grouped = sub.groupby(lhs_cols, dropna=True)[rhs_col]
    majority_sum = 0
    for _, grp in grouped:
        cnt = Counter(grp.tolist())
        majority_sum += cnt.most_common(1)[0][1]
    confidence = majority_sum / support if support > 0 else 0.0
    return round(confidence, 6), int(support)


def calc_soft_fd_confidence_and_support(df, lhs_cols, rhs_col):
    sub = df[lhs_cols + [rhs_col]].dropna()
    support = len(sub)
    if support < SOFT_FD_MIN_GROUP_SIZE:
        return 0.0, support
    grouped = sub.groupby(lhs_cols, dropna=True)[rhs_col]
    majority_sum = 0
    used_support = 0
    for _, grp in grouped:
        if len(grp) < SOFT_FD_MIN_GROUP_SIZE:
            continue
        cnt = Counter(grp.tolist())
        majority_sum += cnt.most_common(1)[0][1]
        used_support += len(grp)
    confidence = majority_sum / used_support if used_support > 0 else 0.0
    return round(confidence, 6), int(used_support)


def calc_fd_quality_profile(df, lhs_cols, rhs_col, min_group_size):
    """Return stricter FD quality signals used to avoid soft-FD candidate explosion.

    confidence alone is insufficient on long-tail entity data. This profile requires:
    - enough non-null support;
    - enough usable groups;
    - sufficiently pure majority value inside groups;
    - sufficient majority margin over the second value.
    """
    sub = df[lhs_cols + [rhs_col]].dropna()
    support = len(sub)
    if support == 0:
        return {
            "support": 0,
            "used_support": 0,
            "confidence": 0.0,
            "usable_group_count": 0,
            "avg_group_size": 0.0,
            "min_group_size": 0,
            "qualified_group_count": 0,
            "qualified_support": 0,
            "qualified_ratio": 0.0,
        }

    grouped = sub.groupby(lhs_cols, dropna=True)[rhs_col]
    majority_sum = 0
    used_support = 0
    usable_group_count = 0
    qualified_group_count = 0
    qualified_support = 0
    group_sizes = []

    for _, grp in grouped:
        gsize = len(grp)
        if gsize < min_group_size:
            continue
        cnt = Counter(grp.tolist())
        most = cnt.most_common(2)
        maj_val, maj_cnt = most[0]
        second_cnt = most[1][1] if len(most) > 1 else 0
        ratio = maj_cnt / gsize if gsize > 0 else 0.0
        margin = maj_cnt - second_cnt

        usable_group_count += 1
        used_support += gsize
        majority_sum += maj_cnt
        group_sizes.append(gsize)

        if ratio >= FD_MIN_DOMINANT_RATIO and margin >= FD_MIN_DOMINANT_MARGIN:
            qualified_group_count += 1
            qualified_support += gsize

    confidence = majority_sum / used_support if used_support > 0 else 0.0
    qualified_ratio = qualified_support / used_support if used_support > 0 else 0.0

    return {
        "support": int(support),
        "used_support": int(used_support),
        "confidence": round(float(confidence), 6),
        "usable_group_count": int(usable_group_count),
        "avg_group_size": round(float(sum(group_sizes) / len(group_sizes)), 6) if group_sizes else 0.0,
        "min_group_size": int(min(group_sizes)) if group_sizes else 0,
        "qualified_group_count": int(qualified_group_count),
        "qualified_support": int(qualified_support),
        "qualified_ratio": round(float(qualified_ratio), 6),
    }


def calc_not_null_confidence_and_support(series):
    total = len(series)
    if total == 0:
        return 0.0, 0
    non_null = series.notna().sum()
    return round(non_null / total, 6), int(total)


def calc_pattern_confidence_and_support(series, dominant_pattern):
    non_null = series.dropna()
    support = len(non_null)
    if support == 0:
        return 0.0, 0
    pats = non_null.map(detect_basic_pattern)
    ok = (pats == dominant_pattern).sum()
    return round(ok / support, 6), int(support)


# ============================================================
# 4. 列画像
# ============================================================

def analyze_columns(df):
    profiles = {}
    n = len(df)

    for col in df.columns:
        s = normalize_series(df[col])
        non_null = s.dropna()
        missing_rate = 1.0 - (len(non_null) / n if n > 0 else 0.0)
        unique_count = non_null.nunique()
        unique_ratio = unique_count / len(non_null) if len(non_null) > 0 else 0.0
        numeric_vals, numeric_ratio = try_parse_numeric(non_null)

        patterns = non_null.map(detect_basic_pattern)
        pattern_counter = Counter(patterns.tolist())
        total_patterns = sum(pattern_counter.values())
        top_patterns = [
            {"pattern": p, "count": c, "ratio": round(c / total_patterns, 6)}
            for p, c in pattern_counter.most_common(10)
        ]

        lengths = non_null.map(lambda x: len(str(x)))
        avg_len = float(lengths.mean()) if len(lengths) > 0 else 0.0

        profile = {
            "column": col,
            "missing_rate": round(missing_rate, 6),
            "non_null_count": int(len(non_null)),
            "unique_count": int(unique_count),
            "unique_ratio": round(unique_ratio, 6),
            "numeric_parse_ratio": round(float(numeric_ratio), 6),
            "entropy": round(shannon_entropy(non_null), 6),
            "detected_type": None,
            "semantic_type": None,
            "index_like": is_index_like_column(non_null),
            "avg_length": round(avg_len, 6),
            "top_values": value_frequency_dict(non_null, topk=20),
            "top_patterns": top_patterns,
            "value_counter": Counter(non_null.tolist()),
        }

        if col == "birthyear":
            profile["detected_type"] = "numeric"
            profile["semantic_type"] = "birth_year"
        elif col == "season":
            profile["detected_type"] = "numeric"
            profile["semantic_type"] = "season_year"
        elif col == "position":
            profile["detected_type"] = "categorical"
            profile["semantic_type"] = "position_category"
        elif col in {"team", "stadium"}:
            profile["detected_type"] = "categorical" if unique_count <= ENUM_UNIQUE_THRESHOLD or unique_ratio <= 0.25 else "text"
            profile["semantic_type"] = f"{col}_entity"
        elif col in {"name", "surname", "manager"}:
            profile["detected_type"] = "text"
            profile["semantic_type"] = "person_name"
        elif col in {"birthplace", "city"}:
            profile["detected_type"] = "categorical" if unique_count <= ENUM_UNIQUE_THRESHOLD or unique_ratio <= 0.25 else "text"
            profile["semantic_type"] = "location"
        elif numeric_ratio >= NUMERIC_RATIO_THRESHOLD:
            profile["detected_type"] = "numeric"
            profile["semantic_type"] = "numeric"
        else:
            if unique_count <= ENUM_UNIQUE_THRESHOLD or unique_ratio <= ENUM_RATIO_THRESHOLD:
                profile["detected_type"] = "categorical"
            else:
                profile["detected_type"] = "text"

            if profile["index_like"]:
                profile["semantic_type"] = "id_like"
            elif is_id_like_column(non_null, unique_ratio, numeric_ratio, avg_len):
                profile["semantic_type"] = "id_like"
            elif is_code_like_column(non_null, unique_ratio, numeric_ratio, avg_len, top_patterns):
                profile["semantic_type"] = "code_like"
            elif profile["detected_type"] == "numeric":
                profile["semantic_type"] = "numeric"
            elif profile["detected_type"] == "categorical":
                profile["semantic_type"] = "categorical"
            else:
                profile["semantic_type"] = "text"

        profiles[col] = profile

    return profiles


# ============================================================
# 5. schema/domain rules
# ============================================================

def build_schema_rules(df, profiles):
    rules = []
    rule_id = 1

    for col, info in profiles.items():
        if col not in SCHEMA_RULE_COLUMNS:
            continue

        series = normalize_series(df[col])

        if col in HIGH_RISK_MISSING_COLUMNS:
            # Soccer 修正：name/surname/birthplace 的 missing 是上一轮漏召回的主要来源，因此统一纳入 missing guard。
            not_null_threshold = 0.95
            if col in {"team", "position", "season"}:
                not_null_threshold = 0.98
            if info["missing_rate"] <= (1.0 - not_null_threshold):
                confidence, support = calc_not_null_confidence_and_support(series)
                rules.append({
                    "rule_id": f"SCHEMA_{rule_id}",
                    "rule_type": "not_null",
                    "column": col,
                    "target_column": col,
                    "description": f"{col} should not be null in soccer records",
                    "confidence": confidence,
                    "support": support,
                    "usage_role": "strong_rule" if col in PRIMARY_TARGET_COLUMNS else "candidate_generation_rule",
                    "target_scope": "primary_target" if col in PRIMARY_TARGET_COLUMNS else "context_guard",
                })
                rule_id += 1

        if col == "birthyear":
            rules.append({
                "rule_id": f"SCHEMA_{rule_id}",
                "rule_type": "birthyear_numeric_range",
                "column": col,
                "target_column": col,
                "description": f"birthyear should be an integer year within {BIRTHYEAR_MIN}..{BIRTHYEAR_MAX}",
                "lower_bound": BIRTHYEAR_MIN,
                "upper_bound": BIRTHYEAR_MAX,
                "regex": r"^\d{4}$",
                "confidence": 0.95,
                "support": info["non_null_count"],
                "usage_role": "strong_rule",
                "target_scope": "primary_target_numeric_guard",
            })
            rule_id += 1

        if col == "season":
            rules.append({
                "rule_id": f"SCHEMA_{rule_id}",
                "rule_type": "season_numeric_range",
                "column": col,
                "target_column": col,
                "description": f"season should be an integer year within {SEASON_MIN}..{SEASON_MAX}",
                "lower_bound": SEASON_MIN,
                "upper_bound": SEASON_MAX,
                "regex": r"^\d{4}$",
                "confidence": 0.95,
                "support": info["non_null_count"],
                "usage_role": "strong_rule",
                "target_scope": "primary_target_numeric_guard",
            })
            rule_id += 1

        if col == "position":
            rules.append({
                "rule_id": f"SCHEMA_{rule_id}",
                "rule_type": "position_domain",
                "column": col,
                "target_column": col,
                "description": "position should be a known soccer playing position",
                "valid_values": sorted(list(VALID_POSITIONS)),
                "canonical_map": POSITION_CANONICAL_MAP,
                "confidence": 0.96,
                "support": info["non_null_count"],
                "usage_role": "strong_rule",
                "target_scope": "primary_target_domain_guard",
            })
            rule_id += 1

        if col == "name":
            rules.append({
                "rule_id": f"SCHEMA_{rule_id}",
                "rule_type": "name_pattern",
                "column": col,
                "target_column": col,
                "description": "name should contain alphabetic tokens separated by underscores",
                "regex": PERSON_NAME_REGEX,
                "confidence": 0.88,
                "support": info["non_null_count"],
                "usage_role": "candidate_generation_rule",
                "target_scope": "identity_format_guard",
            })
            rule_id += 1

        if col == "surname":
            rules.append({
                "rule_id": f"SCHEMA_{rule_id}",
                "rule_type": "surname_pattern",
                "column": col,
                "target_column": col,
                "description": "surname should be alphabetic, optionally with hyphen/underscore/apostrophe separators",
                "regex": SURNAME_REGEX,
                "confidence": 0.88,
                "support": info["non_null_count"],
                "usage_role": "candidate_generation_rule",
                "target_scope": "identity_format_guard",
            })
            rule_id += 1

        if col == "manager":
            rules.append({
                "rule_id": f"SCHEMA_{rule_id}",
                "rule_type": "manager_name_pattern",
                "column": col,
                "target_column": col,
                "description": "manager should look like a person name, often with underscore-separated tokens",
                "regex": PERSON_NAME_REGEX,
                "confidence": 0.88,
                "support": info["non_null_count"],
                "usage_role": "candidate_generation_rule",
                "target_scope": "primary_target_format_guard",
            })
            rule_id += 1

        if col == "team":
            rules.append({
                "rule_id": f"SCHEMA_{rule_id}",
                "rule_type": "team_name_pattern",
                "column": col,
                "target_column": col,
                "description": "team should look like a club/team entity string",
                "regex": TEAM_REGEX,
                "confidence": 0.86,
                "support": info["non_null_count"],
                "usage_role": "candidate_generation_rule",
                "target_scope": "primary_target_format_guard",
            })
            rule_id += 1

        if col in {"birthplace", "city"}:
            rules.append({
                "rule_id": f"SCHEMA_{rule_id}",
                "rule_type": f"{col}_location_pattern",
                "column": col,
                "target_column": col,
                "description": f"{col} should look like a location name",
                "regex": LOCATION_REGEX,
                "confidence": 0.86,
                "support": info["non_null_count"],
                "usage_role": "candidate_generation_rule",
                "target_scope": "primary_target_format_guard",
            })
            rule_id += 1

        if col == "stadium":
            rules.append({
                "rule_id": f"SCHEMA_{rule_id}",
                "rule_type": "stadium_name_pattern",
                "column": col,
                "target_column": col,
                "description": "stadium should look like a stadium/entity string",
                "regex": STADIUM_REGEX,
                "confidence": 0.86,
                "support": info["non_null_count"],
                "usage_role": "candidate_generation_rule",
                "target_scope": "primary_target_format_guard",
            })
            rule_id += 1

        # Generic dominant pattern rule: only for selected text/entity columns, not for numeric/domain fields.
        if col not in SKIP_GENERIC_PATTERN_COLUMNS:
            top_patterns = info.get("top_patterns", [])
            if top_patterns:
                dominant = top_patterns[0]
                if dominant["ratio"] >= DOMINANT_PATTERN_THRESHOLD:
                    confidence, support = calc_pattern_confidence_and_support(series, dominant["pattern"])
                    rules.append({
                        "rule_id": f"SCHEMA_{rule_id}",
                        "rule_type": "pattern",
                        "column": col,
                        "target_column": col,
                        "description": f"{col} usually follows pattern {dominant['pattern']}",
                        "dominant_pattern": dominant["pattern"],
                        "dominant_pattern_ratio": dominant["ratio"],
                        "confidence": confidence,
                        "support": support,
                        "usage_role": "candidate_generation_rule",
                        "target_scope": "format_guard",
                    })
                    rule_id += 1

    return rules


# ============================================================
# 6. FD / soft FD
# ============================================================

def select_fd_lhs_candidates(profiles):
    candidates = []
    for col, info in profiles.items():
        if col in EXTRA_SKIP_FD_LHS_COLUMNS:
            continue
        if (not ALLOW_INDEX_COLUMN) and info.get("index_like", False):
            continue
        if info["unique_ratio"] >= HIGH_CARDINALITY_RATIO_FOR_LHS:
            continue
        candidates.append(col)
    return candidates


def build_fd_rules(df, profiles):
    rules = []
    rule_id = 1

    cols = list(df.columns)
    norm_df = pd.DataFrame({c: normalize_series(df[c]) for c in cols})

    lhs_candidates = select_fd_lhs_candidates(profiles)
    rhs_candidates = [c for c in FD_RHS_CANDIDATE_COLUMNS if c in cols]

    lhs_sets = []
    for k in range(1, MAX_LHS_SIZE + 1):
        lhs_sets.extend(list(combinations(lhs_candidates, k)))

    for lhs in lhs_sets:
        for rhs in rhs_candidates:
            if rhs in lhs:
                continue

            profile = calc_fd_quality_profile(norm_df, list(lhs), rhs, min_group_size=FD_MIN_GROUP_SIZE)
            confidence = profile["confidence"]
            support = profile["used_support"]
            if support < MIN_SUPPORT_ROWS:
                continue

            # Strong FD must be both highly confident and dominated by qualified groups.
            if (
                confidence >= FD_CONFIDENCE_THRESHOLD
                and profile["qualified_group_count"] > 0
                and profile["qualified_ratio"] >= 0.80
            ):
                rules.append({
                    "rule_id": f"FD_{rule_id}",
                    "rule_type": "functional_dependency",
                    "lhs": list(lhs),
                    "rhs": rhs,
                    "target_column": rhs,
                    "description": f"{', '.join(lhs)} -> {rhs}",
                    "confidence": confidence,
                    "support": support,
                    "usage_role": "strong_rule",
                    "priority": "low",
                    "candidate_trigger": "strict_fd_only",
                    "fd_quality_profile": profile,
                    "candidate_trigger_conditions": {
                        "min_group_size": FD_MIN_GROUP_SIZE,
                        "min_dominant_ratio": FD_MIN_DOMINANT_RATIO,
                        "min_dominant_margin": FD_MIN_DOMINANT_MARGIN,
                        "min_qualified_ratio": 0.80,
                    },
                })
                rule_id += 1

    dedup = {(tuple(sorted(r["lhs"])), r["rhs"]): r for r in rules}
    return list(dedup.values())


def build_soft_fd_rules(df, profiles):
    rules = []
    rule_id = 1

    cols = list(df.columns)
    norm_df = pd.DataFrame({c: normalize_series(df[c]) for c in cols})

    lhs_candidates = select_fd_lhs_candidates(profiles)
    rhs_candidates = [c for c in FD_RHS_CANDIDATE_COLUMNS if c in cols]

    lhs_sets = []
    for k in range(1, MAX_LHS_SIZE + 1):
        lhs_sets.extend(list(combinations(lhs_candidates, k)))

    for lhs in lhs_sets:
        for rhs in rhs_candidates:
            if rhs in lhs:
                continue

            profile = calc_fd_quality_profile(norm_df, list(lhs), rhs, min_group_size=SOFT_FD_MIN_GROUP_SIZE)
            confidence = profile["confidence"]
            support = profile["used_support"]
            if support < SOFT_FD_MIN_SUPPORT_ROWS:
                continue

            if SOFT_FD_CONFIDENCE_THRESHOLD <= confidence < FD_CONFIDENCE_THRESHOLD:
                rules.append({
                    "rule_id": f"SOFTFD_EVID_{rule_id}",
                    "rule_type": "soft_functional_dependency",
                    "lhs": list(lhs),
                    "rhs": rhs,
                    "target_column": rhs,
                    "description": f"soft FD evidence only: {', '.join(lhs)} -> {rhs}",
                    "confidence": confidence,
                    "support": support,
                    "usage_role": SOFT_FD_USAGE_ROLE,
                    "priority": "low",
                    "candidate_trigger": "disabled_evidence_only",
                    "allow_candidate_generation": False,
                    "fd_quality_profile": profile,
                    "note": "soft FD is retained only as evidence; shrink_search_space should not add candidates solely from this rule",
                })
                rule_id += 1

    return rules


# ============================================================
# 7. Soccer context / consistency rules
# ============================================================

def build_team_context_rules(df):
    """Build soccer-specific context dominant rules.

    与 Adult 的 relationship_context_dominant 类似，但这里不使用测试集后验归因。
    只从 dirty table 中挖掘 team/season/stadium/city 等上下文下的多数值分布，
    用作候选生成或证据。弱上下文规则后续应通过 LLM 标注 reliability gate 校准。
    """
    rules = []
    rule_id = 1

    norm_df = pd.DataFrame({c: normalize_series(df[c]) for c in df.columns})

    for rhs in TEAM_CONTEXT_TARGETS:
        if rhs not in norm_df.columns:
            continue

        for lhs in TEAM_SEASON_CONTEXT_LHS_SPECS:
            if rhs in lhs:
                continue
            if any(c not in norm_df.columns for c in lhs + [rhs]):
                continue

            sub = norm_df[lhs + [rhs]].dropna()
            if len(sub) < TEAM_CONTEXT_MIN_GROUP_SIZE:
                continue

            grouped = sub.groupby(lhs, dropna=True)[rhs]
            majority_map = {}
            total_support = 0

            for key, grp in grouped:
                if len(grp) < TEAM_CONTEXT_MIN_GROUP_SIZE:
                    continue

                cnt = Counter(grp.tolist())
                most = cnt.most_common(2)
                maj_val, maj_cnt = most[0]
                second_cnt = most[1][1] if len(most) > 1 else 0
                ratio = maj_cnt / len(grp)
                margin = maj_cnt - second_cnt

                if ratio >= TEAM_CONTEXT_MIN_RATIO and margin >= TEAM_CONTEXT_MIN_DOMINANT_MARGIN:
                    if not isinstance(key, tuple):
                        key = (key,)
                    majority_map["||".join(map(str, key))] = {
                        "dominant_value": maj_val,
                        "group_size": len(grp),
                        "ratio": round(ratio, 6),
                        "dominant_margin": int(margin),
                    }
                    total_support += len(grp)

            if majority_map:
                rules.append({
                    "rule_id": f"CAND_TEAMCTX_{rule_id}",
                    "rule_type": "team_context_dominant",
                    "lhs": lhs,
                    "rhs": rhs,
                    "target_column": rhs,
                    "description": f"soccer context dominant: {', '.join(lhs)} -> {rhs}",
                    "mapping_size": len(majority_map),
                    "majority_map": majority_map,
                    "confidence": 0.84,
                    "support": total_support,
                    "usage_role": TEAM_CONTEXT_USAGE_ROLE,
                    "priority": "low",
                    "candidate_trigger": "disabled_evidence_only",
                    "allow_candidate_generation": False,
                    "min_group_size": TEAM_CONTEXT_MIN_GROUP_SIZE,
                    "min_ratio": TEAM_CONTEXT_MIN_RATIO,
                    "min_dominant_margin": TEAM_CONTEXT_MIN_DOMINANT_MARGIN,
                    "note": "team context is retained only as evidence; shrink_search_space should not add candidates solely from this rule",
                })
                rule_id += 1

    return rules


def build_soccer_consistency_rules(df, profiles):
    rules = []
    rule_id = 1

    if ENABLE_BIRTHYEAR_SEASON_AGE_RULE and {"birthyear", "season"}.issubset(df.columns):
        rules.append({
            "rule_id": f"CAND_SOCCERCONS_{rule_id}",
            "rule_type": "birthyear_season_age_consistency",
            "columns": ["birthyear", "season"],
            "target_column": "birthyear",
            "description": (
                "A player's age at the given season should usually be within a plausible professional football range; "
                "if season-birthyear is outside the configured range, birthyear is suspicious."
            ),
            "min_age": PLAYER_MIN_AGE_AT_SEASON,
            "max_age": PLAYER_MAX_AGE_AT_SEASON,
            "birthyear_lower_bound": BIRTHYEAR_MIN,
            "birthyear_upper_bound": BIRTHYEAR_MAX,
            "season_lower_bound": SEASON_MIN,
            "season_upper_bound": SEASON_MAX,
            "confidence": 0.90,
            "support": len(df),
            "usage_role": "strong_rule",
            "priority": "high",
        })
        rule_id += 1

    if {"team", "manager", "season"}.issubset(df.columns):
        rules.append({
            "rule_id": f"CAND_SOCCERCONS_{rule_id}",
            "rule_type": "team_season_manager_consistency_hint",
            "columns": ["team", "season", "manager"],
            "target_column": "manager",
            "description": (
                "Within a team-season group, manager is expected to be relatively stable; "
                "the actual majority mapping is provided by team_context_dominant rules."
            ),
            "confidence": 0.82,
            "support": len(df),
            "usage_role": "evidence_only",
            "priority": "low",
            "candidate_trigger": "disabled_evidence_only",
            "allow_candidate_generation": False,
        })
        rule_id += 1

    if {"city", "stadium"}.issubset(df.columns):
        rules.append({
            "rule_id": f"CAND_SOCCERCONS_{rule_id}",
            "rule_type": "city_stadium_consistency_hint",
            "columns": ["city", "stadium"],
            "target_column": "stadium",
            "description": (
                "A stadium and city pair should be contextually consistent; "
                "mined city/stadium majority mappings should be used as weak evidence."
            ),
            "confidence": 0.80,
            "support": len(df),
            "usage_role": "evidence_only",
            "priority": "low",
            "candidate_trigger": "disabled_evidence_only",
            "allow_candidate_generation": False,
        })
        rule_id += 1

    return rules


def build_candidate_generation_rules(df, profiles):
    rules = []
    rule_id = 1

    # Soccer: 不启用 rare/global dominant 作为候选生成规则，避免长尾球员、球队、城市被误报。
    # high risk missing: 对关键字段缺失进行候选生成。
    for col in HIGH_RISK_MISSING_COLUMNS:
        if col not in profiles:
            continue
        prof = profiles[col]
        rules.append({
            "rule_id": f"CAND_HRMISS_{rule_id}",
            "rule_type": "high_risk_missing",
            "column": col,
            "target_column": col,
            "description": f"{col} missing values are suspicious in soccer candidate generation",
            "confidence": 0.82,
            "support": prof["non_null_count"],
            "usage_role": "candidate_generation_rule",
            "priority": "medium",
            "target_scope": "schema_missing_guard",
        })
        rule_id += 1

    rules.extend(build_team_context_rules(df))
    rules.extend(build_soccer_consistency_rules(df, profiles))

    return rules


# ============================================================
# 8. priority
# ============================================================

def assign_rule_priority(rule):
    rtype = rule.get("rule_type", "")

    if rtype in {
        "position_domain",
        "birthyear_numeric_range",
        "season_numeric_range",
        "birthyear_season_age_consistency",
    }:
        return "high"

    if rtype in {
        "name_pattern",
        "surname_pattern",
        "manager_name_pattern",
        "team_name_pattern",
        "birthplace_location_pattern",
        "city_location_pattern",
        "stadium_name_pattern",
        "pattern",
        "high_risk_missing",
    }:
        return "medium"

    if rtype in {
        "team_context_dominant",
        "team_season_manager_consistency_hint",
        "city_stadium_consistency_hint",
        "functional_dependency",
        "soft_functional_dependency",
    }:
        return "low"

    if rule.get("usage_role", "candidate_generation_rule") == "candidate_generation_rule":
        if rule.get("confidence", 0.0) >= 0.90 and rule.get("support", 0) >= 20:
            return "medium"
        return "low"

    return "medium"


def enrich_rules_with_priority(rule_groups):
    new_groups = {}
    for group_name, rules in rule_groups.items():
        enriched = []
        for r in rules:
            r2 = dict(r)
            r2["priority"] = assign_rule_priority(r2)
            enriched.append(r2)
        new_groups[group_name] = enriched
    return new_groups


# ============================================================
# 9. 主流程
# ============================================================

def build_rule_pool(input_csv, output_json):
    df = pd.read_csv(input_csv, dtype=str)
    df = maybe_drop_unnamed_columns(df)

    expected_cols = set(SOCCER_COLUMNS)
    actual_cols = set(df.columns)
    missing_cols = sorted(list(expected_cols - actual_cols))
    extra_cols = sorted(list(actual_cols - expected_cols))
    if missing_cols:
        print(f"[WARN] Missing expected soccer columns: {missing_cols}")
    if extra_cols:
        print(f"[WARN] Extra columns in input: {extra_cols}")

    for col in df.columns:
        df[col] = normalize_series(df[col])

    profiles = analyze_columns(df)
    schema_rules = build_schema_rules(df, profiles)
    fd_rules = build_fd_rules(df, profiles)
    soft_fd_rules = build_soft_fd_rules(df, profiles)
    candidate_rules = build_candidate_generation_rules(df, profiles)

    rule_groups = {
        "schema_rules": schema_rules,
        "fd_rules": fd_rules,
        "soft_fd_rules": soft_fd_rules,
        "candidate_rules": candidate_rules,
    }
    rule_groups = enrich_rules_with_priority(rule_groups)

    rule_pool = {
        "metadata": {
            "source_file": input_csv,
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "design_goal": "high_recall_candidate_generation_for_soccer",
            "dataset": "soccer",
            "expected_columns": SOCCER_COLUMNS,
            "target_candidate_columns": sorted(list(TARGET_CANDIDATE_COLUMNS)),
            "primary_target_columns": sorted(list(PRIMARY_TARGET_COLUMNS)),
            "context_only_columns": sorted(list(CONTEXT_ONLY_COLUMNS)),
            "notes": [
                "candidate inclusion is determined only by violating rules in this pool",
                "soccer rule pool uses schema/domain/range/pattern rules plus mined FD/soft-FD/context evidence",
                "rare value, rare pattern, and global dominant rules are disabled to avoid excessive false positives on long-tail soccer entities",
                "birthyear-season age consistency captures implausible player ages at a season",
                "name, surname, and birthplace are now primary candidate columns because the previous run missed many true errors in these fields",
                "not-null, numeric range, domain, and explicit format rules are intended to be direct candidate-generation rules",
                "team/stadium/city/manager context dominant rules are evidence-only by default to avoid large false-positive expansion",
                "soft FD rules are evidence-only and should not be used by shrink_search_space to add candidates directly",
                "strong FD rules are mined only under strict confidence/support/group-purity conditions",
            ],
        },
        "column_profiles": profiles,
        "rules": rule_groups,
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(rule_pool, f, ensure_ascii=False, indent=2)

    print(f"[OK] Rule pool saved to: {output_json}")
    print(f"Schema rules: {len(schema_rules)}")
    print(f"FD rules: {len(fd_rules)}")
    print(f"Soft FD rules: {len(soft_fd_rules)}")
    print(f"Candidate generation rules: {len(candidate_rules)}")


if __name__ == "__main__":
    build_rule_pool(INPUT_CSV, OUTPUT_JSON)

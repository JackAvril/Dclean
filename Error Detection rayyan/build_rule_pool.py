import json
import math
import re
from collections import Counter
from itertools import combinations
from typing import Dict, List, Any, Optional

import pandas as pd

INPUT_CSV = "/mnt/mydata/dq/projects/Splittree/rayyan/rayyan_dirty.csv"
OUTPUT_JSON = "rule_pool_rayyan.json"

MISSING_TOKENS = {
    "", "empty", "null", "none", "nan", "na", "n/a", "missing", "unk",
    "not available", "contact airline"
}

NUMERIC_RATIO_THRESHOLD = 0.90
ENUM_UNIQUE_THRESHOLD = 50
ENUM_RATIO_THRESHOLD = 0.08

ID_LIKE_UNIQUE_RATIO_THRESHOLD = 0.98
CODE_LIKE_AVG_LEN_THRESHOLD = 20
CODE_LIKE_DOMINANT_PATTERN_THRESHOLD = 0.70
DOMINANT_PATTERN_THRESHOLD = 0.70
DOMINANT_TYPO_MIN_RATIO = 0.50

MAX_LHS_SIZE = 2
FD_CONFIDENCE_THRESHOLD = 0.98
MIN_SUPPORT_ROWS = 5

SOFT_FD_CONFIDENCE_THRESHOLD = 0.80
SOFT_FD_MIN_GROUP_SIZE = 3
PAIR_CONTEXT_MIN_RATIO = 0.70
PAIR_CONTEXT_MIN_GROUP_SIZE = 2

HIGH_CARDINALITY_RATIO_FOR_LHS = 0.95
ALLOW_INDEX_COLUMN = False
SKIP_SCHEMA_FOR_ID_LIKE = True

TARGET_CANDIDATE_COLUMNS = {
    "article_language",
    "journal_title",
    "jounral_abbreviation",
    "journal_issn",
    "article_jvolumn",
    "article_jissue",
    "article_jcreated_at",
    "article_pagination",
    "author_list",
}
HIGH_RISK_MISSING_COLUMNS = {
    "article_language",
    "journal_title",
    "jounral_abbreviation",
    "journal_issn",
    "article_jcreated_at",
    "author_list",
}

ENABLE_RARE_VALUE_RULE = True
RARE_VALUE_MAX_FREQ = 2
ENABLE_RARE_PATTERN_RULE = True
RARE_PATTERN_MAX_RATIO = 0.02

# 更激进地限制容易制造误报的列
SKIP_RARE_VALUE_COLUMNS = {
    "article_title", "author_list", "journal_title",
    "jounral_abbreviation", "article_pagination",
    "article_jvolumn", "article_jissue", "article_language",
    # 关键：article_jcreated_at 不再靠 rare_value 进候选，
    # 这个列改由专门的日期错位规则负责召回
    "article_jcreated_at",
}
SKIP_RARE_PATTERN_COLUMNS = {
    "article_title", "author_list", "article_language", "journal_issn"
}
SKIP_GLOBAL_DOMINANT_COLUMNS = {
    "article_title", "author_list", "journal_title",
    "jounral_abbreviation", "article_pagination"
}
# 不对这些列加 generic dominant-pattern 规则，避免误报
SKIP_GENERIC_PATTERN_COLUMNS = {"article_language", "journal_issn", "article_jissue", "article_jvolumn"}

LANG_CANONICAL_MAP = {
    "eng": "eng",
    "english": "eng",
    "english; abstract language:english": "eng",
    "en": "eng",
    "pol": "pol",
    "polish": "pol",
    "fre": "fre",
    "french": "fre",
    "ger": "ger",
    "german": "ger",
    "spa": "spa",
    "spanish": "spa",
}

MOJIBAKE_CHARS = {"�", "��", "\ufffd"}


def normalize_value(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.lower() in MISSING_TOKENS:
        return None
    return s.lower()


def normalize_series(s):
    return s.map(normalize_value)


def maybe_drop_unnamed_columns(df):
    keep_cols = [c for c in df.columns if not str(c).lower().startswith("unnamed:")]
    return df[keep_cols].copy()


def detect_basic_pattern(value):
    if value is None:
        return "NULL"
    parts, prev_type, cnt = [], None, 0
    for ch in value:
        if ch.isdigit():
            cur = "D"
        elif ch.isalpha():
            cur = "L"
        elif ch in {" ", "-", "_", "/", ".", ",", "(", ")", "%", ":", ";", "&", "'", "+", "[", "]", "{", "}", '"'}:
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


def canonicalize_language(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    return LANG_CANONICAL_MAP.get(str(s).strip().lower())


def extract_issn_digits(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    m = re.search(r'(\d{4})[- ]?(\d{3}[\dx])', str(s).strip().lower())
    if not m:
        return None
    return f"{m.group(1)}-{m.group(2)}"


def canonicalize_volume_issue(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    txt = str(s).strip().lower()
    m = re.search(r'(\d+)', txt)
    if not m:
        return None
    return str(int(m.group(1)))


def normalize_date_like(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    txt = str(s).strip().lower()
    txt = re.sub(r'\s+', '', txt)

    m = re.match(r'^(\d{1,2})/(\d{1,2})/(\d{2,4})$', txt)
    if m:
        mm, dd, yy = int(m.group(1)), int(m.group(2)), m.group(3)
        yy_i = int(yy)
        if len(yy) == 2:
            yy_i = 1900 + yy_i if yy_i >= 30 else 2000 + yy_i
        if 1 <= mm <= 12 and 1 <= dd <= 31:
            return f"{yy_i:04d}-{mm:02d}-{dd:02d}"

    m = re.match(r'^(\d{4})-(\d{1,2})-(\d{1,2})$', txt)
    if m:
        yy_i, mm, dd = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mm <= 12 and 1 <= dd <= 31:
            return f"{yy_i:04d}-{mm:02d}-{dd:02d}"
    return None


def split_date_components(s: Optional[str]):
    if s is None:
        return None
    txt = str(s).strip().lower()
    txt = re.sub(r'\s+', '', txt)
    m = re.match(r'^(\d{1,2})/(\d{1,2})/(\d{2,4})$', txt)
    if not m:
        return None
    a, b, c = m.group(1), m.group(2), m.group(3)
    return int(a), int(b), c


def generate_date_permutations(s: Optional[str]):
    """
    保留原先较宽松的候选生成。
    """
    comp = split_date_components(s)
    if comp is None:
        return []
    a, b, c = comp
    perms = []
    candidates = [
        (1, 3, 2),  # m/y/d 近似映射后再规整
        (2, 3, 1),  # d/y/m
        (2, 1, 3),  # d/m/y
    ]
    arr = [str(a), str(b), str(c)]
    for idxs in candidates:
        mm_raw, dd_raw, yy_raw = arr[idxs[0] - 1], arr[idxs[1] - 1], arr[idxs[2] - 1]
        try:
            mm = int(mm_raw)
            dd = int(dd_raw)
            yy_i = int(yy_raw)
        except Exception:
            continue
        if len(yy_raw) == 2:
            yy_i = 1900 + yy_i if yy_i >= 30 else 2000 + yy_i
        if 1 <= mm <= 12 and 1 <= dd <= 31:
            perms.append(f"{yy_i:04d}-{mm:02d}-{dd:02d}")
    return list(dict.fromkeys(perms))


def generate_ymd_swap_candidate(s: Optional[str]) -> Optional[str]:
    """
    专门针对当前 rayyan 中最常见的错位模式：
    dirty 常表现为 YY/M/D 或 D/M/YY 混写成 a/b/c，
    最终想恢复成更像 M/D/YY 的 clean 形式。

    经验上，很多漏错像：
    4/2/15 -> 2/15/04
    1/1/13 -> 1/13/01
    3/1/14 -> 1/14/03

    因此这里重点尝试：
      a/b/c  ->  b/c/a
    然后再规范化成 yyyy-mm-dd。
    """
    comp = split_date_components(s)
    if comp is None:
        return None

    a, b, c = comp
    a_str = str(a)
    b_str = str(b)
    c_str = str(c)

    try:
        mm = int(b_str)
        dd = int(c_str)
        yy_i = int(a_str)
    except Exception:
        return None

    if len(a_str) <= 2:
        yy_i = 1900 + yy_i if yy_i >= 30 else 2000 + yy_i

    if 1 <= mm <= 12 and 1 <= dd <= 31:
        return f"{yy_i:04d}-{mm:02d}-{dd:02d}"
    return None


def canonicalize_pagination(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    txt = str(s).strip().lower()
    m = re.search(r'(\d+)\s*-\s*(\d+)', txt)
    if m:
        return f"{int(m.group(1))}-{int(m.group(2))}"
    m = re.search(r'(\d+)\s*-\s*$', txt)
    if m:
        return str(int(m.group(1)))
    return None


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
    for _, grp in grouped:
        if len(grp) < SOFT_FD_MIN_GROUP_SIZE:
            continue
        cnt = Counter(grp.tolist())
        majority_sum += cnt.most_common(1)[0][1]
    confidence = majority_sum / support if support > 0 else 0.0
    return round(confidence, 6), int(support)


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


def build_date_value_distribution(series):
    canon = series.dropna().map(normalize_date_like).dropna()
    year_counter = Counter([x[:4] for x in canon if len(x) >= 4])
    md_counter = Counter([x[5:] for x in canon if len(x) >= 10])
    return {
        "canonical_count": int(len(canon)),
        "top_years": dict(year_counter.most_common(20)),
        "top_month_day": dict(md_counter.most_common(50)),
    }


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
        top_patterns = [{"pattern": p, "count": c, "ratio": round(c / total_patterns, 6)} for p, c in pattern_counter.most_common(10)]
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
        if col == "article_jcreated_at":
            profile["date_distribution"] = build_date_value_distribution(non_null)
        if numeric_ratio >= NUMERIC_RATIO_THRESHOLD:
            profile["detected_type"] = "numeric"
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


def build_schema_rules(df, profiles):
    rules = []
    rule_id = 1
    for col, info in profiles.items():
        if col not in TARGET_CANDIDATE_COLUMNS:
            continue

        series = normalize_series(df[col])

        not_null_threshold = 0.55
        if col in {"journal_title", "article_language", "article_jcreated_at", "author_list"}:
            not_null_threshold = 0.70

        if info["missing_rate"] <= not_null_threshold:
            confidence, support = calc_not_null_confidence_and_support(series)
            rules.append({
                "rule_id": f"SCHEMA_{rule_id}",
                "rule_type": "not_null",
                "column": col,
                "description": f"{col} should not be null",
                "confidence": confidence,
                "support": support,
                "usage_role": "candidate_generation_rule",
            })
            rule_id += 1

        if len(info["top_patterns"]) > 0 and col not in SKIP_GENERIC_PATTERN_COLUMNS:
            main_pattern = info["top_patterns"][0]
            if main_pattern["ratio"] >= DOMINANT_PATTERN_THRESHOLD:
                confidence, support = calc_pattern_confidence_and_support(series, main_pattern["pattern"])
                rules.append({
                    "rule_id": f"SCHEMA_{rule_id}",
                    "rule_type": "pattern",
                    "column": col,
                    "description": f"{col} should mostly follow the dominant pattern",
                    "pattern": main_pattern["pattern"],
                    "confidence": confidence,
                    "support": support,
                    "usage_role": "strong_rule",
                })
                rule_id += 1

        if col == "journal_issn":
            for rtype, regex, conf in [
                ("issn_format", r"^\d{4}-\d{3}[\dx](\s*\(print\))?$", 0.82),
                ("issn_loose_format", r".*\d{4}[- ]?\d{3}[\dx].*", 0.78),
            ]:
                rules.append({
                    "rule_id": f"SCHEMA_{rule_id}",
                    "rule_type": rtype,
                    "column": col,
                    "description": f"{col} regex check {rtype}",
                    "regex": regex,
                    "confidence": conf,
                    "support": info["non_null_count"],
                    "usage_role": "candidate_generation_rule",
                })
                rule_id += 1

            rules.append({
                "rule_id": f"SCHEMA_{rule_id}",
                "rule_type": "issn_canonical_format",
                "column": col,
                "description": "journal_issn should usually use canonical xxxx-xxxx form",
                "confidence": 0.90,
                "support": info["non_null_count"],
                "usage_role": "candidate_generation_rule",
            })
            rule_id += 1

        if col == "article_language":
            rules.append({
                "rule_id": f"SCHEMA_{rule_id}",
                "rule_type": "language_canonical_format",
                "column": col,
                "description": "article_language should use canonical short code (e.g., eng)",
                "confidence": 0.88,
                "support": info["non_null_count"],
                "usage_role": "candidate_generation_rule",
            })
            rule_id += 1

        if col == "article_jcreated_at":
            rules.append({
                "rule_id": f"SCHEMA_{rule_id}",
                "rule_type": "date_like_format",
                "column": col,
                "description": "article_jcreated_at should usually be parseable as a date",
                "confidence": 0.82,
                "support": info["non_null_count"],
                "usage_role": "candidate_generation_rule",
            })
            rule_id += 1

            # 保留原先较宽松的重排可疑规则
            rules.append({
                "rule_id": f"SCHEMA_{rule_id}",
                "rule_type": "date_component_permutation_suspicious",
                "column": col,
                "description": "article_jcreated_at may have month/day/year component permutation",
                "confidence": 0.86,
                "support": info.get("date_distribution", {}).get("canonical_count", 0),
                "usage_role": "candidate_generation_rule",
                "date_distribution": info.get("date_distribution", {}),
            })
            rule_id += 1

            # 新增：专门针对当前数据中最常见的 YY/M/D -> M/D/YY 错位模式
            rules.append({
                "rule_id": f"SCHEMA_{rule_id}",
                "rule_type": "date_ymd_swap_suspicious",
                "column": col,
                "description": "article_jcreated_at may follow a strong YY/M/D -> M/D/YY swap pattern",
                "confidence": 0.94,
                "support": info.get("date_distribution", {}).get("canonical_count", 0),
                "usage_role": "candidate_generation_rule",
                "date_distribution": info.get("date_distribution", {}),
            })
            rule_id += 1

        if col in {"article_jvolumn", "article_jissue"}:
            rules.append({
                "rule_id": f"SCHEMA_{rule_id}",
                "rule_type": "volume_issue_numeric_format",
                "column": col,
                "description": f"{col} should usually be numeric-like",
                "regex": r"^\d+$",
                "confidence": 0.82,
                "support": info["non_null_count"],
                "usage_role": "candidate_generation_rule",
            })
            rule_id += 1

        if col == "article_pagination":
            rules.append({
                "rule_id": f"SCHEMA_{rule_id}",
                "rule_type": "pagination_format",
                "column": col,
                "description": "article_pagination should usually look like page ranges such as 1187-9 or 43-49",
                "regex": r"^(\d+(-\d+)?)(\s+st\s+-.*)?$",
                "confidence": 0.75,
                "support": info["non_null_count"],
                "usage_role": "candidate_generation_rule",
            })
            rule_id += 1

        if col == "author_list":
            rules.append({
                "rule_id": f"SCHEMA_{rule_id}",
                "rule_type": "author_list_braced_format",
                "column": col,
                "description": "author_list should usually be a braced quoted list",
                "regex": r'^\{.*\}$',
                "confidence": 0.78,
                "support": info["non_null_count"],
                "usage_role": "candidate_generation_rule",
            })
            rule_id += 1
            rules.append({
                "rule_id": f"SCHEMA_{rule_id}",
                "rule_type": "author_list_encoding_corruption",
                "column": col,
                "description": "author_list should not contain mojibake / replacement characters",
                "confidence": 0.92,
                "support": info["non_null_count"],
                "usage_role": "candidate_generation_rule",
            })
            rule_id += 1

    return rules


def select_fd_lhs_candidates(profiles):
    candidates = []
    for col, info in profiles.items():
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
    rhs_candidates = [c for c in TARGET_CANDIDATE_COLUMNS if c in cols]

    lhs_sets = []
    for k in range(1, MAX_LHS_SIZE + 1):
        lhs_sets.extend(list(combinations(lhs_candidates, k)))

    for lhs in lhs_sets:
        for rhs in rhs_candidates:
            if rhs in lhs:
                continue
            confidence, support = calc_fd_confidence_and_support(norm_df, list(lhs), rhs)
            if support < MIN_SUPPORT_ROWS:
                continue
            if confidence >= FD_CONFIDENCE_THRESHOLD:
                rules.append({
                    "rule_id": f"FD_{rule_id}",
                    "rule_type": "functional_dependency",
                    "lhs": list(lhs),
                    "rhs": rhs,
                    "description": f"{', '.join(lhs)} -> {rhs}",
                    "confidence": confidence,
                    "support": support,
                    "usage_role": "strong_rule",
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
    rhs_candidates = [c for c in TARGET_CANDIDATE_COLUMNS if c in cols]

    lhs_sets = []
    for k in range(1, MAX_LHS_SIZE + 1):
        lhs_sets.extend(list(combinations(lhs_candidates, k)))

    for lhs in lhs_sets:
        for rhs in rhs_candidates:
            if rhs in lhs:
                continue
            confidence, support = calc_soft_fd_confidence_and_support(norm_df, list(lhs), rhs)
            if support < SOFT_FD_MIN_GROUP_SIZE:
                continue
            if SOFT_FD_CONFIDENCE_THRESHOLD <= confidence < FD_CONFIDENCE_THRESHOLD:
                rules.append({
                    "rule_id": f"CAND_SOFTFD_{rule_id}",
                    "rule_type": "soft_functional_dependency",
                    "lhs": list(lhs),
                    "rhs": rhs,
                    "description": f"soft FD: {', '.join(lhs)} -> {rhs}",
                    "confidence": confidence,
                    "support": support,
                    "usage_role": "candidate_generation_rule",
                })
                rule_id += 1
    return rules


def build_pair_context_rules(df):
    rules = []
    rule_id = 1
    norm_df = pd.DataFrame({c: normalize_series(df[c]) for c in df.columns})

    pair_specs = [
        (["journal_title"], "jounral_abbreviation"),
        (["jounral_abbreviation"], "journal_title"),
        (["journal_title"], "journal_issn"),
        (["journal_issn"], "journal_title"),
        (["journal_issn"], "jounral_abbreviation"),
        (["journal_title", "journal_issn"], "jounral_abbreviation"),
        (["journal_title", "jounral_abbreviation"], "journal_issn"),
    ]

    for lhs, rhs in pair_specs:
        if any(c not in norm_df.columns for c in lhs + [rhs]):
            continue
        sub = norm_df[lhs + [rhs]].dropna()
        if len(sub) < PAIR_CONTEXT_MIN_GROUP_SIZE:
            continue
        grouped = sub.groupby(lhs, dropna=True)[rhs]
        majority_map = {}
        total_support = 0
        for key, grp in grouped:
            if len(grp) < PAIR_CONTEXT_MIN_GROUP_SIZE:
                continue
            cnt = Counter(grp.tolist())
            maj_val, maj_cnt = cnt.most_common(1)[0]
            ratio = maj_cnt / len(grp)
            if ratio >= PAIR_CONTEXT_MIN_RATIO:
                if not isinstance(key, tuple):
                    key = (key,)
                majority_map["||".join(map(str, key))] = {
                    "dominant_value": maj_val,
                    "group_size": len(grp),
                    "ratio": round(ratio, 6)
                }
                total_support += len(grp)
        if majority_map:
            rules.append({
                "rule_id": f"CAND_PAIRCTX_{rule_id}",
                "rule_type": "dominant_value_by_context_pair" if len(lhs) == 2 else "dominant_value_by_context",
                "lhs": lhs,
                "rhs": rhs,
                "description": f"context dominant: {', '.join(lhs)} -> {rhs}",
                "mapping_size": len(majority_map),
                "majority_map": majority_map,
                "confidence": 0.84,
                "support": total_support,
                "usage_role": "candidate_generation_rule",
            })
            rule_id += 1
    return rules


def build_candidate_generation_rules(df, profiles):
    rules = []
    rule_id = 1
    norm_df = pd.DataFrame({c: normalize_series(df[c]) for c in df.columns})

    for col, prof in profiles.items():
        if col not in TARGET_CANDIDATE_COLUMNS or col in SKIP_GLOBAL_DOMINANT_COLUMNS:
            continue
        top_values = prof.get("top_values", [])
        if not top_values:
            continue
        top1 = top_values[0]
        if top1["ratio"] >= DOMINANT_TYPO_MIN_RATIO:
            rules.append({
                "rule_id": f"CAND_GDOM_{rule_id}",
                "rule_type": "global_dominant_value",
                "column": col,
                "description": f"global dominant value for {col}",
                "dominant_value": top1["value"],
                "dominant_count": top1["count"],
                "dominant_ratio": top1["ratio"],
                "confidence": round(top1["ratio"], 6),
                "support": prof["non_null_count"],
                "usage_role": "candidate_generation_rule",
            })
            rule_id += 1

    lhs_candidates = select_fd_lhs_candidates(profiles)
    context_rule_id = 1
    for rhs in TARGET_CANDIDATE_COLUMNS:
        if rhs not in df.columns:
            continue
        for lhs in lhs_candidates:
            if lhs == rhs:
                continue
            sub = norm_df[[lhs, rhs]].dropna()
            if len(sub) < SOFT_FD_MIN_GROUP_SIZE:
                continue
            grouped = sub.groupby(lhs)[rhs]
            majority_map = {}
            total_support = 0
            for lhs_val, grp in grouped:
                if len(grp) < SOFT_FD_MIN_GROUP_SIZE:
                    continue
                cnt = Counter(grp.tolist())
                maj_val, maj_cnt = cnt.most_common(1)[0]
                ratio = maj_cnt / len(grp)
                if ratio >= SOFT_FD_CONFIDENCE_THRESHOLD:
                    majority_map[lhs_val] = {
                        "dominant_value": maj_val,
                        "group_size": len(grp),
                        "ratio": round(ratio, 6)
                    }
                    total_support += len(grp)
            if majority_map:
                rules.append({
                    "rule_id": f"CAND_CTXDOM_{context_rule_id}",
                    "rule_type": "dominant_value_by_context",
                    "lhs": [lhs],
                    "rhs": rhs,
                    "description": f"context dominant: {lhs} -> dominant {rhs}",
                    "mapping_size": len(majority_map),
                    "majority_map": majority_map,
                    "confidence": 0.85,
                    "support": total_support,
                    "usage_role": "candidate_generation_rule",
                })
                context_rule_id += 1

    if ENABLE_RARE_VALUE_RULE:
        for col, prof in profiles.items():
            if col not in TARGET_CANDIDATE_COLUMNS or col in SKIP_RARE_VALUE_COLUMNS:
                continue
            counter = prof["value_counter"]
            rare_values = [v for v, c in counter.items() if c <= RARE_VALUE_MAX_FREQ]
            if rare_values:
                rules.append({
                    "rule_id": f"CAND_RAREVAL_{rule_id}",
                    "rule_type": "rare_value",
                    "column": col,
                    "description": f"rare values in {col} may be suspicious",
                    "rare_values": rare_values[:5000],
                    "confidence": 0.6,
                    "support": sum(counter[v] for v in rare_values),
                    "usage_role": "candidate_generation_rule",
                })
                rule_id += 1

    if ENABLE_RARE_PATTERN_RULE:
        for col, prof in profiles.items():
            if col not in TARGET_CANDIDATE_COLUMNS or col in SKIP_RARE_PATTERN_COLUMNS:
                continue
            top_patterns = prof.get("top_patterns", [])
            rare_patterns = [x["pattern"] for x in top_patterns if x["ratio"] <= RARE_PATTERN_MAX_RATIO]
            if rare_patterns:
                rules.append({
                    "rule_id": f"CAND_RAREPAT_{rule_id}",
                    "rule_type": "rare_pattern",
                    "column": col,
                    "description": f"rare patterns in {col} may be suspicious",
                    "rare_patterns": rare_patterns,
                    "confidence": 0.6,
                    "support": prof["non_null_count"],
                    "usage_role": "candidate_generation_rule",
                })
                rule_id += 1

    for col in HIGH_RISK_MISSING_COLUMNS:
        if col not in profiles:
            continue
        prof = profiles[col]
        rules.append({
            "rule_id": f"CAND_HRMISS_{rule_id}",
            "rule_type": "high_risk_missing",
            "column": col,
            "description": f"{col} missing values are suspicious in candidate generation",
            "confidence": 0.8,
            "support": prof["non_null_count"],
            "usage_role": "candidate_generation_rule",
        })
        rule_id += 1

    for a, b in [
        ("journal_title", "jounral_abbreviation"),
        ("journal_title", "journal_issn"),
        ("article_jvolumn", "article_jissue"),
    ]:
        if a in df.columns and b in df.columns:
            rules.append({
                "rule_id": f"CAND_PAIR_{rule_id}",
                "rule_type": "paired_missing_consistency",
                "columns": [a, b],
                "description": f"{a} and {b} should usually co-occur",
                "confidence": 0.8,
                "support": len(df),
                "usage_role": "candidate_generation_rule",
            })
            rule_id += 1

    rules.extend(build_pair_context_rules(df))
    return rules


def assign_rule_priority(rule):
    rtype = rule.get("rule_type")
    if rtype in {
        "date_component_permutation_suspicious",
        "date_ymd_swap_suspicious",
        "author_list_encoding_corruption",
    }:
        return "medium"

    if rule.get("usage_role", "candidate_generation_rule") == "candidate_generation_rule":
        if rule.get("confidence", 0.0) >= 0.85 and rule.get("support", 0) >= 20:
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


def build_rule_pool(input_csv, output_json):
    df = pd.read_csv(input_csv, dtype=str)
    df = maybe_drop_unnamed_columns(df)
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
        "candidate_rules": candidate_rules
    }
    rule_groups = enrich_rules_with_priority(rule_groups)

    rule_pool = {
        "metadata": {
            "source_file": input_csv,
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "design_goal": "high_recall_candidate_generation",
            "notes": [
                "candidate inclusion is determined only by violating rules in this pool",
                "candidate targets focus on bibliographic metadata fields; index/id mainly serve as context",
                "recall is boosted mainly by journal consistency rules, date-permutation suspicion, date YY/M/D swap suspicion, author-list encoding corruption, and paired metadata rules",
                "article_jcreated_at no longer relies on rare_value; dedicated date swap rules are used instead",
                "rare_value and generic pattern are weakened on long-tail bibliographic fields to reduce false positives",
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

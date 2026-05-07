import json
import math
import re
from collections import Counter
from itertools import combinations
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd


# ============================================================
# 1. 用户配置区
# ============================================================

INPUT_CSV = "/mnt/mydata/dq/projects/Splittree/hospital/hospital_dirty_no_address23.csv"
OUTPUT_JSON = "rule_pool_high_recall_v2.json"

# 缺失值标记
MISSING_TOKENS = {"", "empty", "null", "none", "nan", "na", "n/a", "missing", "unk"}

# 列类型判定
NUMERIC_RATIO_THRESHOLD = 0.90
ENUM_UNIQUE_THRESHOLD = 30
ENUM_RATIO_THRESHOLD = 0.05

# semantic type 判定
ID_LIKE_UNIQUE_RATIO_THRESHOLD = 0.98
CODE_LIKE_AVG_LEN_THRESHOLD = 20
CODE_LIKE_DOMINANT_PATTERN_THRESHOLD = 0.70

# 主模式规则阈值
DOMINANT_PATTERN_THRESHOLD = 0.70

# FD 挖掘参数（强规则）
MAX_LHS_SIZE = 2
FD_CONFIDENCE_THRESHOLD = 0.98
MIN_SUPPORT_ROWS = 5

# 软 FD 参数（高召回候选规则）
SOFT_FD_CONFIDENCE_THRESHOLD = 0.80
SOFT_FD_MIN_GROUP_SIZE = 3

# 过滤超高基数字段作为 FD 左部候选
HIGH_CARDINALITY_RATIO_FOR_LHS = 0.95

# 是否保留 index 型列参与规则发现
ALLOW_INDEX_COLUMN = False

# 默认不对 id_like 生成太多 schema 规则
SKIP_SCHEMA_FOR_ID_LIKE = True
SKIP_NUMERIC_RANGE_FOR_CODE_LIKE = True

# 高风险列：对缺失值更激进地纳入候选
HIGH_RISK_MISSING_COLUMNS = {"Score", "Sample"}

# 列专属格式规则
ENABLE_SPECIAL_COLUMN_RULES = True

# typo-aware 主值候选规则
ENABLE_DOMINANT_TYPO_RULE = True
DOMINANT_TYPO_MAX_EDIT_DISTANCE = 2
DOMINANT_TYPO_MIN_COUNT = 20
DOMINANT_TYPO_MIN_RATIO = 0.50

# 稀有值弱候选
ENABLE_RARE_VALUE_RULE = True
RARE_VALUE_MAX_FREQ = 2

# 稀有 pattern 弱候选
ENABLE_RARE_PATTERN_RULE = True
RARE_PATTERN_MAX_RATIO = 0.02


# ============================================================
# 2. 基础工具函数
# ============================================================

def normalize_value(x):
    """归一化单元格值。"""
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.lower() in MISSING_TOKENS:
        return None
    return s.lower()


def normalize_series(s: pd.Series) -> pd.Series:
    return s.map(normalize_value)


def maybe_drop_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [c for c in df.columns if not str(c).lower().startswith("unnamed:")]
    return df[keep_cols].copy()


def is_index_like_column(series: pd.Series) -> bool:
    s = series.dropna()
    if len(s) == 0:
        return False

    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.notna().mean() < 0.95:
        return False

    unique_ratio = s.nunique() / len(s)
    return unique_ratio > 0.98


def detect_basic_pattern(value: str) -> str:
    """
    抽取粗粒度模式。
    例:
      2053258100 -> D10
      scip-card-2 -> L4S1L4S1D1
      al_ami-1 -> L2S1L3S1D1
    """
    if value is None:
        return "NULL"

    parts = []
    prev_type = None
    cnt = 0

    for ch in value:
        if ch.isdigit():
            cur = "D"
        elif ch.isalpha():
            cur = "L"
        elif ch in {" ", "-", "_", "/", ".", ",", "(", ")", "%", ":", ";", "&", "'"}:
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


def try_parse_numeric(series: pd.Series):
    cleaned = (
        series.dropna()
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
    )
    numeric = pd.to_numeric(cleaned, errors="coerce")
    success_ratio = numeric.notna().mean() if len(cleaned) > 0 else 0.0
    return numeric, success_ratio


def shannon_entropy(series: pd.Series) -> float:
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


def value_frequency_dict(series: pd.Series, topk=20):
    cnt = Counter(series.dropna().tolist())
    total = sum(cnt.values()) if cnt else 0
    out = []
    for k, v in cnt.most_common(topk):
        out.append({
            "value": k,
            "count": v,
            "ratio": round(v / total, 6) if total > 0 else 0.0
        })
    return out


def levenshtein_distance(a: str, b: str) -> int:
    """简单编辑距离。"""
    if a is None or b is None:
        return 999999
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            rep = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, dele, rep))
        prev = cur
    return prev[-1]


def looks_like_score_format(v: Optional[str]) -> bool:
    """Score 列常见格式：98%、100%、0% 等。"""
    if v is None:
        return False
    return re.fullmatch(r"\d{1,3}%", v) is not None


def split_sample_value(v: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Sample 列预期类似: '149 patients'
    返回 (number_part, suffix_part)
    """
    if v is None:
        return None, None
    m = re.fullmatch(r"(\d+)\s+(.+)", v.strip())
    if not m:
        return None, None
    return m.group(1), m.group(2).strip()


def looks_like_sample_format(v: Optional[str]) -> bool:
    if v is None:
        return False
    return re.fullmatch(r"\d+\s+patients", v) is not None


def is_typo_like_patients(v: Optional[str]) -> bool:
    """
    抓 '619 paxienxs' / '0 patixnts' / '82 patientx' 这类。
    """
    if v is None:
        return False
    num, suffix = split_sample_value(v)
    if num is None or suffix is None:
        return False
    if suffix == "patients":
        return False
    return levenshtein_distance(suffix, "patients") <= 2


# ============================================================
# 3. semantic type 判定
# ============================================================

def is_id_like_column(non_null: pd.Series, unique_ratio: float, numeric_ratio: float, avg_len: float) -> bool:
    if len(non_null) == 0:
        return False

    if unique_ratio >= ID_LIKE_UNIQUE_RATIO_THRESHOLD and avg_len <= CODE_LIKE_AVG_LEN_THRESHOLD:
        return True

    if unique_ratio >= ID_LIKE_UNIQUE_RATIO_THRESHOLD and numeric_ratio >= NUMERIC_RATIO_THRESHOLD:
        return True

    return False


def is_code_like_column(
    non_null: pd.Series,
    unique_ratio: float,
    numeric_ratio: float,
    avg_len: float,
    top_patterns: List[Dict[str, Any]]
) -> bool:
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
# 4. 规则质量评估
# ============================================================

def calc_fd_confidence_and_support(df: pd.DataFrame, lhs_cols: List[str], rhs_col: str):
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


def calc_soft_fd_confidence_and_support(df: pd.DataFrame, lhs_cols: List[str], rhs_col: str):
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


def calc_not_null_confidence_and_support(series: pd.Series):
    total = len(series)
    if total == 0:
        return 0.0, 0
    non_null = series.notna().sum()
    return round(non_null / total, 6), int(total)


def calc_enum_confidence_and_support(series: pd.Series, allowed_values: set):
    non_null = series.dropna()
    support = len(non_null)
    if support == 0:
        return 0.0, 0
    ok = non_null.isin(allowed_values).sum()
    return round(ok / support, 6), int(support)


def calc_pattern_confidence_and_support(series: pd.Series, dominant_pattern: str):
    non_null = series.dropna()
    support = len(non_null)
    if support == 0:
        return 0.0, 0
    pats = non_null.map(detect_basic_pattern)
    ok = (pats == dominant_pattern).sum()
    return round(ok / support, 6), int(support)


def calc_numeric_range_confidence_and_support(series: pd.Series, min_val: float, max_val: float):
    numeric, _ = try_parse_numeric(series)
    nums = numeric.dropna()
    support = len(nums)
    if support == 0:
        return 0.0, 0
    ok = ((nums >= min_val) & (nums <= max_val)).sum()
    return round(ok / support, 6), int(support)


# ============================================================
# 5. 列分析
# ============================================================

def analyze_columns(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
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
            {
                "pattern": p,
                "count": c,
                "ratio": round(c / total_patterns, 6)
            }
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
            "value_counter": Counter(non_null.tolist())
        }

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

        if profile["detected_type"] == "numeric":
            nums = numeric_vals.dropna()
            if len(nums) > 0:
                q1 = float(nums.quantile(0.25))
                q3 = float(nums.quantile(0.75))
                iqr = q3 - q1
                profile["numeric_stats"] = {
                    "min": float(nums.min()),
                    "max": float(nums.max()),
                    "mean": float(nums.mean()),
                    "std": float(nums.std(ddof=0)) if len(nums) > 1 else 0.0,
                    "q1": q1,
                    "median": float(nums.quantile(0.5)),
                    "q3": q3,
                    "iqr": iqr,
                    "iqr_lower": q1 - 1.5 * iqr,
                    "iqr_upper": q3 + 1.5 * iqr,
                }

        profiles[col] = profile

    return profiles


# ============================================================
# 6. 构建 Schema / Domain / Pattern 规则
# ============================================================

def build_schema_rules(df: pd.DataFrame, profiles: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rules = []
    rule_id = 1

    for col, info in profiles.items():
        series = normalize_series(df[col])

        if (not ALLOW_INDEX_COLUMN) and info.get("index_like", False):
            continue

        if SKIP_SCHEMA_FOR_ID_LIKE and info.get("semantic_type") == "id_like":
            continue

        # 6.1 非空规则：对高风险列放宽阈值
        not_null_threshold = 0.30 if col in HIGH_RISK_MISSING_COLUMNS else 0.01
        if info["missing_rate"] <= not_null_threshold:
            confidence, support = calc_not_null_confidence_and_support(series)
            rules.append({
                "rule_id": f"SCHEMA_{rule_id}",
                "rule_type": "not_null",
                "column": col,
                "description": f"{col} should not be null",
                "confidence": confidence,
                "support": support,
                "usage_role": "strong_rule" if col not in HIGH_RISK_MISSING_COLUMNS else "candidate_generation_rule"
            })
            rule_id += 1

        # 6.2 数值范围规则
        if info["detected_type"] == "numeric" and "numeric_stats" in info:
            if not (SKIP_NUMERIC_RANGE_FOR_CODE_LIKE and info.get("semantic_type") in {"code_like", "id_like"}):
                stats = info["numeric_stats"]
                confidence, support = calc_numeric_range_confidence_and_support(
                    series, stats["min"], stats["max"]
                )
                rules.append({
                    "rule_id": f"SCHEMA_{rule_id}",
                    "rule_type": "numeric_range",
                    "column": col,
                    "description": f"{col} should be within observed numeric range",
                    "range": {
                        "min": stats["min"],
                        "max": stats["max"]
                    },
                    "confidence": confidence,
                    "support": support,
                    "usage_role": "strong_rule"
                })
                rule_id += 1

        # 6.3 枚举域规则
        if info["semantic_type"] == "categorical":
            allowed_values = [x["value"] for x in info["top_values"]]
            confidence, support = calc_enum_confidence_and_support(series, set(allowed_values))
            rules.append({
                "rule_id": f"SCHEMA_{rule_id}",
                "rule_type": "enumeration_domain",
                "column": col,
                "description": f"{col} should belong to a limited domain",
                "allowed_values": allowed_values,
                "confidence": confidence,
                "support": support,
                "usage_role": "strong_rule"
            })
            rule_id += 1

        # 6.4 主模式规则
        if len(info["top_patterns"]) > 0:
            main_pattern = info["top_patterns"][0]
            if main_pattern["ratio"] >= DOMINANT_PATTERN_THRESHOLD:
                confidence, support = calc_pattern_confidence_and_support(
                    series, main_pattern["pattern"]
                )
                rules.append({
                    "rule_id": f"SCHEMA_{rule_id}",
                    "rule_type": "pattern",
                    "column": col,
                    "description": f"{col} should mostly follow the dominant pattern",
                    "pattern": main_pattern["pattern"],
                    "confidence": confidence,
                    "support": support,
                    "usage_role": "strong_rule"
                })
                rule_id += 1

        # 6.5 列专属规则：Score
        if ENABLE_SPECIAL_COLUMN_RULES and col == "Score":
            rules.append({
                "rule_id": f"SCHEMA_{rule_id}",
                "rule_type": "score_format",
                "column": col,
                "description": "Score should usually look like a percentage, e.g. 98%",
                "regex": r"^\d{1,3}%$",
                "confidence": 0.9,
                "support": info["non_null_count"],
                "usage_role": "candidate_generation_rule"
            })
            rule_id += 1

        # 6.6 列专属规则：Sample
        if ENABLE_SPECIAL_COLUMN_RULES and col == "Sample":
            rules.append({
                "rule_id": f"SCHEMA_{rule_id}",
                "rule_type": "sample_format",
                "column": col,
                "description": "Sample should usually look like '<number> patients'",
                "regex": r"^\d+\s+patients$",
                "confidence": 0.9,
                "support": info["non_null_count"],
                "usage_role": "candidate_generation_rule"
            })
            rule_id += 1

            rules.append({
                "rule_id": f"SCHEMA_{rule_id}",
                "rule_type": "sample_patients_suffix",
                "column": col,
                "description": "Sample suffix should usually be canonical token 'patients'",
                "canonical_suffix": "patients",
                "max_edit_distance": 2,
                "confidence": 0.9,
                "support": info["non_null_count"],
                "usage_role": "candidate_generation_rule"
            })
            rule_id += 1

    return rules


# ============================================================
# 7. 自动挖掘 FD 规则
# ============================================================

def select_fd_lhs_candidates(profiles: Dict[str, Dict[str, Any]]) -> List[str]:
    candidates = []

    for col, info in profiles.items():
        if (not ALLOW_INDEX_COLUMN) and info.get("index_like", False):
            continue

        if info["unique_ratio"] >= HIGH_CARDINALITY_RATIO_FOR_LHS:
            continue

        candidates.append(col)

    return candidates


def build_fd_rules(df: pd.DataFrame, profiles: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rules = []
    rule_id = 1

    cols = list(df.columns)
    norm_df = pd.DataFrame({c: normalize_series(df[c]) for c in cols})

    lhs_candidates = select_fd_lhs_candidates(profiles)

    rhs_candidates = []
    for c in cols:
        if (not ALLOW_INDEX_COLUMN) and profiles[c].get("index_like", False):
            continue
        rhs_candidates.append(c)

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
                    "usage_role": "strong_rule"
                })
                rule_id += 1

    # 去重 + 去冗余
    dedup = {}
    for r in rules:
        key = (tuple(sorted(r["lhs"])), r["rhs"])
        dedup[key] = r

    filtered = list(dedup.values())

    final_rules = []
    for r in filtered:
        lhs_set = set(r["lhs"])
        rhs = r["rhs"]
        dominated = False

        for r2 in filtered:
            if r2["rhs"] != rhs:
                continue
            lhs2_set = set(r2["lhs"])
            if lhs2_set < lhs_set and r2["confidence"] >= r["confidence"]:
                dominated = True
                break

        if not dominated:
            final_rules.append(r)

    return final_rules


# ============================================================
# 8. 软 FD / 候选生成规则
# ============================================================

def build_soft_fd_rules(df: pd.DataFrame, profiles: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rules = []
    rule_id = 1

    cols = list(df.columns)
    norm_df = pd.DataFrame({c: normalize_series(df[c]) for c in cols})

    lhs_candidates = select_fd_lhs_candidates(profiles)

    rhs_candidates = []
    for c in cols:
        if (not ALLOW_INDEX_COLUMN) and profiles[c].get("index_like", False):
            continue
        rhs_candidates.append(c)

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
                    "usage_role": "candidate_generation_rule"
                })
                rule_id += 1

    dedup = {}
    for r in rules:
        key = (tuple(sorted(r["lhs"])), r["rhs"])
        # 保留更高置信度的
        if key not in dedup or r["confidence"] > dedup[key]["confidence"]:
            dedup[key] = r

    return list(dedup.values())


# ============================================================
# 9. 主值 / 上下文主值 / 稀有值 / typo 候选规则
# ============================================================

def build_candidate_generation_rules(df: pd.DataFrame, profiles: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rules = []
    rule_id = 1

    norm_df = pd.DataFrame({c: normalize_series(df[c]) for c in df.columns})

    # 9.1 global dominant value
    for col, prof in profiles.items():
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
                "usage_role": "candidate_generation_rule"
            })
            rule_id += 1

    # 9.2 dominant value by context
    context_rule_id = 1
    lhs_candidates = select_fd_lhs_candidates(profiles)

    for rhs in df.columns:
        if (not ALLOW_INDEX_COLUMN) and profiles[rhs].get("index_like", False):
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
                    "usage_role": "candidate_generation_rule"
                })
                context_rule_id += 1

    # 9.3 稀有值规则
    if ENABLE_RARE_VALUE_RULE:
        for col, prof in profiles.items():
            if prof["semantic_type"] not in {"categorical", "text", "code_like"}:
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
                    "usage_role": "candidate_generation_rule"
                })
                rule_id += 1

    # 9.4 稀有 pattern 规则
    if ENABLE_RARE_PATTERN_RULE:
        for col, prof in profiles.items():
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
                    "usage_role": "candidate_generation_rule"
                })
                rule_id += 1

    # 9.5 主值 typo 候选
    if ENABLE_DOMINANT_TYPO_RULE:
        for col, prof in profiles.items():
            if prof["semantic_type"] not in {"categorical", "text"}:
                continue

            top_values = prof.get("top_values", [])
            if not top_values:
                continue

            dominant = top_values[0]["value"]
            dominant_count = top_values[0]["count"]
            dominant_ratio = top_values[0]["ratio"]

            if dominant_count < DOMINANT_TYPO_MIN_COUNT or dominant_ratio < DOMINANT_TYPO_MIN_RATIO:
                continue

            suspicious_values = []
            for item in top_values[1:]:
                v = item["value"]
                if abs(len(v) - len(dominant)) > DOMINANT_TYPO_MAX_EDIT_DISTANCE:
                    continue
                if levenshtein_distance(v, dominant) <= DOMINANT_TYPO_MAX_EDIT_DISTANCE:
                    suspicious_values.append(v)

            if suspicious_values:
                rules.append({
                    "rule_id": f"CAND_DTYPO_{rule_id}",
                    "rule_type": "dominant_value_typo",
                    "column": col,
                    "description": f"values close to dominant value in {col} may be typos",
                    "dominant_value": dominant,
                    "suspicious_values": suspicious_values,
                    "confidence": 0.7,
                    "support": dominant_count,
                    "usage_role": "candidate_generation_rule"
                })
                rule_id += 1

    # 9.6 高风险缺失规则
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
            "usage_role": "candidate_generation_rule"
        })
        rule_id += 1

    # 9.7 Score-Sample 联动
    if "Score" in df.columns and "Sample" in df.columns:
        rules.append({
            "rule_id": f"CAND_PAIR_{rule_id}",
            "rule_type": "paired_missing_consistency",
            "columns": ["Score", "Sample"],
            "description": "Score and Sample should usually co-occur",
            "confidence": 0.8,
            "support": len(df),
            "usage_role": "candidate_generation_rule"
        })
        rule_id += 1

    return rules


# ============================================================
# 10. 规则优先级
# ============================================================

def assign_rule_priority(rule: Dict[str, Any]) -> str:
    rule_type = rule["rule_type"]
    conf = rule.get("confidence", 0.0)
    supp = rule.get("support", 0)
    usage_role = rule.get("usage_role", "candidate_generation_rule")

    if usage_role == "candidate_generation_rule":
        if conf >= 0.85 and supp >= 20:
            return "medium"
        return "low"

    if rule_type == "functional_dependency":
        if conf >= 0.995 and supp >= 20:
            return "high"
        return "medium"

    if rule_type == "not_null":
        return "high" if conf >= 0.99 else "medium"

    if rule_type == "enumeration_domain":
        return "medium"

    if rule_type == "pattern":
        return "medium" if conf >= 0.9 else "low"

    if rule_type == "numeric_range":
        return "medium"

    return "low"


def enrich_rules_with_priority(rule_groups: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
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
# 11. 主流程
# ============================================================

def build_rule_pool(input_csv: str, output_json: str):
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
                "schema_rules and fd_rules are relatively strong rules",
                "soft_fd_rules and candidate_rules are weak signals for high-recall candidate generation",
                "candidate set is allowed to include clean tuples",
                "global_dominant_value / context dominant / typo-aware rules should not be treated as hard error rules"
            ]
        },
        "column_profiles": profiles,
        "rules": rule_groups
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(rule_pool, f, ensure_ascii=False, indent=2)

    print(f"[OK] Rule pool saved to: {output_json}")
    print(f"Schema rules: {len(schema_rules)}")
    print(f"FD rules: {len(fd_rules)}")
    print(f"Soft FD rules: {len(soft_fd_rules)}")
    print(f"Candidate generation rules: {len(candidate_rules)}")

    print("\nDetected semantic types:")
    for col, prof in profiles.items():
        print(
            f"  {col}: detected_type={prof['detected_type']}, "
            f"semantic_type={prof['semantic_type']}, "
            f"unique_ratio={prof['unique_ratio']}, "
            f"numeric_parse_ratio={prof['numeric_parse_ratio']}"
        )

    print("\nSpecial notes:")
    if "Score" in profiles:
        print("  - Added Score-specific format and high-risk missing rules")
    if "Sample" in profiles:
        print("  - Added Sample-specific regex / suffix typo-aware / high-risk missing rules")
    if "Score" in profiles and "Sample" in profiles:
        print("  - Added Score-Sample paired consistency rule")


if __name__ == "__main__":
    build_rule_pool(INPUT_CSV, OUTPUT_JSON)
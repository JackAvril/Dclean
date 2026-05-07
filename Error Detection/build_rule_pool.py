import json
import math
from collections import Counter
from itertools import combinations
from typing import Dict, List, Tuple, Any

import pandas as pd


# ============================================================
# 1. 用户配置区
# ============================================================

INPUT_CSV = "/home/dq/Splittree/hospital/hospital_dirty_no_address23.csv"
OUTPUT_JSON = "rule_pool_new.json"

# 缺失值标记
MISSING_TOKENS = {"", "empty", "null", "none", "nan", "na", "n/a", "missing", "unk"}

# 列类型判定
NUMERIC_RATIO_THRESHOLD = 0.90
ENUM_UNIQUE_THRESHOLD = 30
ENUM_RATIO_THRESHOLD = 0.05

# 新增：更细的 semantic type 判定
ID_LIKE_UNIQUE_RATIO_THRESHOLD = 0.98
CODE_LIKE_AVG_LEN_THRESHOLD = 20
CODE_LIKE_DOMINANT_PATTERN_THRESHOLD = 0.70

# 主模式规则阈值
DOMINANT_PATTERN_THRESHOLD = 0.70

# FD 挖掘参数
MAX_LHS_SIZE = 2
FD_CONFIDENCE_THRESHOLD = 0.98
MIN_SUPPORT_ROWS = 5

# 过滤超高基数字段作为 FD 左部候选
HIGH_CARDINALITY_RATIO_FOR_LHS = 0.95

# 是否保留 index 型列参与规则发现
ALLOW_INDEX_COLUMN = False

# 对这些 semantic_type 默认不生成某些 schema 规则
SKIP_SCHEMA_FOR_ID_LIKE = True
SKIP_NUMERIC_RANGE_FOR_CODE_LIKE = True


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
    return s


def normalize_series(s: pd.Series) -> pd.Series:
    return s.map(normalize_value)


def maybe_drop_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """删除 pandas 读 CSV 时常见的 unnamed 列。"""
    keep_cols = [c for c in df.columns if not str(c).lower().startswith("unnamed:")]
    return df[keep_cols].copy()


def is_index_like_column(series: pd.Series) -> bool:
    """
    判断某列是否像自增索引列：
    - 非空
    - 去重后数量接近总行数
    - 可以解析为整数
    """
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
    """
    尝试解析成数值。
    会自动去掉逗号和百分号。
    """
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
    return [{"value": k, "count": v} for k, v in cnt.most_common(topk)]


# ============================================================
# 3. semantic type 判定
# ============================================================

def is_id_like_column(non_null: pd.Series, unique_ratio: float, numeric_ratio: float, avg_len: float) -> bool:
    """
    判断是否像 ID 列：
    - 唯一率极高
    - 通常长度不长
    - 可能是数字或编码
    """
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
    """
    判断是否像编码列：
    - 不是超高唯一 ID
    - 长度通常较短
    - pattern 比较稳定
    - 可能像 ZipCode / MeasureCode / PhoneNumber / Stateavg
    """
    if len(non_null) == 0:
        return False

    dominant_ratio = top_patterns[0]["ratio"] if top_patterns else 0.0

    # 情况1：长度较短 + pattern 稳定 + 不是明显低基数枚举
    if avg_len <= CODE_LIKE_AVG_LEN_THRESHOLD and dominant_ratio >= CODE_LIKE_DOMINANT_PATTERN_THRESHOLD:
        if unique_ratio > ENUM_RATIO_THRESHOLD:
            return True

    # 情况2：几乎都是数字，但不是典型连续数值，而更像编码
    if numeric_ratio >= 0.95 and avg_len <= 15 and unique_ratio < ID_LIKE_UNIQUE_RATIO_THRESHOLD:
        return True

    return False


# ============================================================
# 4. 规则质量评估
# ============================================================

def calc_fd_confidence_and_support(df: pd.DataFrame, lhs_cols: List[str], rhs_col: str):
    """
    近似 FD 质量：
    lhs -> rhs

    support:
        lhs 和 rhs 均非空的有效记录数
    confidence:
        分组后 rhs 多数派占总有效记录的比例
    """
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
    """
    自动分析列属性，为后续规则生成提供基础。
    """
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
            "top_values": value_frequency_dict(non_null, topk=15),
            "top_patterns": top_patterns
        }

        # 原始 detected_type
        if numeric_ratio >= NUMERIC_RATIO_THRESHOLD:
            profile["detected_type"] = "numeric"
        else:
            if unique_count <= ENUM_UNIQUE_THRESHOLD or unique_ratio <= ENUM_RATIO_THRESHOLD:
                profile["detected_type"] = "categorical"
            else:
                profile["detected_type"] = "text"

        # 新增 semantic_type
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

        # 数值统计：只对真正 numeric 语义列更有意义
        if profile["detected_type"] == "numeric":
            nums = numeric_vals.dropna()
            if len(nums) > 0:
                profile["numeric_stats"] = {
                    "min": float(nums.min()),
                    "max": float(nums.max()),
                    "mean": float(nums.mean()),
                    "std": float(nums.std(ddof=0)) if len(nums) > 1 else 0.0,
                    "q1": float(nums.quantile(0.25)),
                    "median": float(nums.quantile(0.5)),
                    "q3": float(nums.quantile(0.75)),
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

        # 可选忽略 index-like 列
        if (not ALLOW_INDEX_COLUMN) and info.get("index_like", False):
            continue

        # id_like 列默认不生成 schema 规则（可减少噪声）
        if SKIP_SCHEMA_FOR_ID_LIKE and info.get("semantic_type") == "id_like":
            continue

        # 6.1 非空规则：缺失率足够低才生成
        if info["missing_rate"] <= 0.01:
            confidence, support = calc_not_null_confidence_and_support(series)
            rules.append({
                "rule_id": f"SCHEMA_{rule_id}",
                "rule_type": "not_null",
                "column": col,
                "description": f"{col} should not be null",
                "confidence": confidence,
                "support": support
            })
            rule_id += 1

        # 6.2 数值范围规则：只对真正 numeric 语义列生成
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
                    "support": support
                })
                rule_id += 1

        # 6.3 枚举域规则：只对 categorical 更合理
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
                "support": support
            })
            rule_id += 1

        # 6.4 主模式规则：对 code_like / text / categorical 都可有帮助
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
                    "support": support
                })
                rule_id += 1

    return rules


# ============================================================
# 7. 自动挖掘 FD 规则
# ============================================================

def select_fd_lhs_candidates(profiles: Dict[str, Dict[str, Any]]) -> List[str]:
    """
    选择可作为 FD 左部的候选列。
    原则：
    - 默认去掉 index-like 列
    - 去掉超高基数字段
    """
    candidates = []

    for col, info in profiles.items():
        if (not ALLOW_INDEX_COLUMN) and info.get("index_like", False):
            continue

        # 超高唯一率列不适合作为通用 LHS 候选
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
                    "support": support
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

            # 若更短左部已经能解释同样 rhs，则移除更长左部
            if lhs2_set < lhs_set and r2["confidence"] >= r["confidence"]:
                dominated = True
                break

        if not dominated:
            final_rules.append(r)

    return final_rules


# ============================================================
# 8. 规则优先级（通用）
# ============================================================

def assign_rule_priority(rule: Dict[str, Any]) -> str:
    """
    给规则一个通用优先级，后面候选错误筛选时可用。
    """
    rule_type = rule["rule_type"]
    conf = rule.get("confidence", 0.0)
    supp = rule.get("support", 0)

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
# 9. 主流程
# ============================================================

def build_rule_pool(input_csv: str, output_json: str):
    df = pd.read_csv(input_csv, dtype=str)
    df = maybe_drop_unnamed_columns(df)

    for col in df.columns:
        df[col] = normalize_series(df[col])

    profiles = analyze_columns(df)

    schema_rules = build_schema_rules(df, profiles)
    fd_rules = build_fd_rules(df, profiles)

    rule_groups = {
        "schema_rules": schema_rules,
        "fd_rules": fd_rules
    }
    rule_groups = enrich_rules_with_priority(rule_groups)

    rule_pool = {
        "metadata": {
            "source_file": input_csv,
            "row_count": int(len(df)),
            "column_count": int(len(df.columns))
        },
        "column_profiles": profiles,
        "rules": rule_groups
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(rule_pool, f, ensure_ascii=False, indent=2)

    print(f"[OK] Rule pool saved to: {output_json}")
    print(f"Schema rules: {len(schema_rules)}")
    print(f"FD rules: {len(fd_rules)}")

    print("\nDetected semantic types:")
    for col, prof in profiles.items():
        print(
            f"  {col}: detected_type={prof['detected_type']}, "
            f"semantic_type={prof['semantic_type']}, "
            f"unique_ratio={prof['unique_ratio']}, "
            f"numeric_parse_ratio={prof['numeric_parse_ratio']}"
        )


if __name__ == "__main__":
    build_rule_pool(INPUT_CSV, OUTPUT_JSON)
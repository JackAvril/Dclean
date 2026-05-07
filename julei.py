from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd


# -----------------------------
# 0) 常用工具
# -----------------------------
NULL_LIKE = {
    "", " ", "na", "n/a", "null", "none", "nan", "-", "--", "unknown", "unk", "?", "nil"
}

def _to_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()

def is_null_like(x) -> bool:
    s = _to_str(x).lower()
    return (s in NULL_LIKE) or (s == "")

def try_parse_float(x) -> Tuple[bool, Optional[float]]:
    if pd.isna(x):
        return False, None
    if isinstance(x, (int, float, np.integer, np.floating)) and not pd.isna(x):
        return True, float(x)

    s = _to_str(x)
    if s == "":
        return False, None

    s2 = s.replace(",", "").replace(" ", "")
    try:
        v = float(s2)
        if np.isfinite(v):
            return True, v
    except Exception:
        pass
    return False, None


# -----------------------------
# 1) 类型判定：只输出 numeric / categorical / id_like / other
# -----------------------------
@dataclass
class TypeDetectConfig:
    sample_size: int = 2000
    numeric_parse_ratio: float = 0.85
    id_unique_ratio: float = 0.98          # 近似唯一 -> id_like
    categorical_unique_ratio: float = 0.20  # 唯一值占比低 -> categorical
    max_len_for_id_like: int = 40           # 太长一般不是 ID
    max_len_for_categorical: int = 60       # 太长一般不是 categorical

@dataclass
class ColumnTypeInfo:
    col: str
    inferred_type: str  # numeric / categorical / id_like / other
    reason: Dict[str, Any]


def infer_column_type_3way(s: pd.Series, cfg: TypeDetectConfig) -> ColumnTypeInfo:
    col = s.name if s.name is not None else "<col>"
    s0 = s.copy()

    # 抽样
    if len(s0) > cfg.sample_size:
        s0 = s0.sample(cfg.sample_size, random_state=42)

    ss = s0.map(_to_str)
    non_null_mask = ~ss.map(is_null_like)
    ss_n = ss[non_null_mask]
    n = len(ss_n)

    if n == 0:
        return ColumnTypeInfo(col, "other", {"note": "all null-like"})

    uniq = ss_n.nunique(dropna=True)
    unique_ratio = float(uniq / max(1, n))
    avg_len = float(ss_n.map(len).mean())

    # numeric：解析成功比例
    ok_flags = []
    vals = []
    for x in ss_n:
        ok, v = try_parse_float(x)
        ok_flags.append(ok)
        vals.append(v if ok else np.nan)
    numeric_ratio = float(np.mean(ok_flags))

    # ✅ 修复：数值型 ID（几乎都是数字 + 几乎唯一）优先判为 id_like
    if (
        numeric_ratio >= cfg.numeric_parse_ratio
        and unique_ratio >= cfg.id_unique_ratio
        and avg_len <= cfg.max_len_for_id_like
    ):
        return ColumnTypeInfo(
            col,
            "id_like",
            {
                "numeric_ratio": numeric_ratio,
                "unique_ratio": unique_ratio,
                "avg_len": avg_len,
                "note": "numeric but near-unique => id_like",
            },
        )

    if numeric_ratio >= cfg.numeric_parse_ratio:
        v = np.array([vv for vv, ok in zip(vals, ok_flags) if ok], dtype=float)
        is_int_ratio = float(np.mean(np.isclose(v, np.round(v))))
        return ColumnTypeInfo(
            col,
            "numeric",
            {
                "numeric_ratio": numeric_ratio,
                "is_int_ratio": is_int_ratio,
                "unique_ratio": unique_ratio,
                "avg_len": avg_len,
            },
        )

    # id_like：近似唯一 + 字符串不太长
    if unique_ratio >= cfg.id_unique_ratio and avg_len <= cfg.max_len_for_id_like:
        return ColumnTypeInfo(
            col,
            "id_like",
            {"unique_ratio": unique_ratio, "avg_len": avg_len},
        )

    # categorical：唯一值占比低 + 字符串不太长
    if unique_ratio <= cfg.categorical_unique_ratio and avg_len <= cfg.max_len_for_categorical:
        return ColumnTypeInfo(
            col,
            "categorical",
            {"unique_ratio": unique_ratio, "avg_len": avg_len},
        )

    # 其余：先不处理（SKIP）
    return ColumnTypeInfo(
        col,
        "other",
        {"numeric_ratio": numeric_ratio, "unique_ratio": unique_ratio, "avg_len": avg_len},
    )


# -----------------------------
# 2) 三类列的分桶/聚类
# -----------------------------
@dataclass
class BucketingConfig:
    # numeric
    numeric_qbins: int = 3               # 第一版建议 3：low/mid/high
    numeric_auto_extreme: bool = True    # ✅ 方案A：自动 extreme
    numeric_extreme_k: float = 10.0      # MAD 鲁棒z阈值，建议 8~12
    numeric_extreme_q: float = 0.005     # MAD退化时回退：0.5%/99.5%

    # categorical
    cat_topk: int = 5
    cat_min_count: int = 20
    cat_min_ratio: float = 0.005


def bucket_numeric(s: pd.Series, cfg: BucketingConfig) -> pd.Series:
    """
    输出桶：
      - NON_NUMERIC：不可解析或 NULL-like
      - NUMERIC__EXTREME：MAD 鲁棒统计判定的极端值
      - NUMERIC__QBIN_i：对剩余数值做 qcut 分箱（i从0开始）
    """
    out = pd.Series(index=s.index, dtype="object")

    ok = pd.Series(False, index=s.index)
    val = pd.Series(np.nan, index=s.index, dtype=float)

    for idx, x in s.items():
        if is_null_like(x):
            out.at[idx] = "NON_NUMERIC"
            continue
        flag, v = try_parse_float(x)
        ok.at[idx] = flag
        val.at[idx] = v if flag else np.nan
        out.at[idx] = "__PENDING__"

    out.loc[~ok] = "NON_NUMERIC"
    numeric_idx = ok[ok].index
    if len(numeric_idx) == 0:
        return out.replace("__PENDING__", "NON_NUMERIC")

    v = val.loc[numeric_idx]

    # ========= 方案A：MAD 自动 EXTREME =========
    if cfg.numeric_auto_extreme:
        eps = 1e-9
        arr = v.values.astype(float)

        med = float(np.nanmedian(arr))
        mad = float(np.nanmedian(np.abs(arr - med)))

        if mad < eps:
            # 退化：列几乎常数，用分位数尾部做 extreme
            q = cfg.numeric_extreme_q
            lo = float(np.nanquantile(arr, q))
            hi = float(np.nanquantile(arr, 1 - q))
            extreme_mask = (v < lo) | (v > hi)
            extreme_method = f"quantile_tail(q={q})"
        else:
            z = np.abs(v - med) / (mad + eps)
            extreme_mask = z > cfg.numeric_extreme_k
            extreme_method = f"MAD(k={cfg.numeric_extreme_k})"

        out.loc[extreme_mask[extreme_mask].index] = "NUMERIC__EXTREME"
        v2 = v.loc[~extreme_mask]
    else:
        extreme_method = "none"
        v2 = v
    # ========================================

    # qcut 分成 low/mid/high（只对非 extreme 部分）
    if len(v2) > 0:
        try:
            qlabels = pd.qcut(v2, q=cfg.numeric_qbins, duplicates="drop")
            cats = qlabels.cat.categories
            cat_to_id = {cat: i for i, cat in enumerate(cats)}
            for idx, cat in qlabels.items():
                out.at[idx] = f"NUMERIC__QBIN_{cat_to_id[cat]}"
        except Exception:
            out.loc[v2.index] = "NUMERIC__QBIN_0"

    out = out.replace("__PENDING__", "NON_NUMERIC")
    # 你如果需要调试 extreme 方法，可以临时 print(extreme_method)
    return out


def bucket_categorical(s: pd.Series, cfg: BucketingConfig) -> pd.Series:
    out = pd.Series(index=s.index, dtype="object")
    ss = s.map(_to_str)
    null_mask = ss.map(is_null_like)
    out.loc[null_mask] = "NULL_LIKE"

    ss2 = ss.loc[~null_mask]
    vc = ss2.value_counts(dropna=False)
    n = len(ss2)

    rare_vals = (vc < cfg.cat_min_count) | (vc / max(1, n) < cfg.cat_min_ratio)
    top_vals = vc.index[: cfg.cat_topk].tolist()

    def assign(v: str) -> str:
        if v in top_vals and not rare_vals.get(v, False):
            return f"TOPK__{v}"
        return "RARE"

    out.loc[~null_mask] = ss2.map(assign)
    return out


def bucket_id_like(s: pd.Series) -> pd.Series:
    out = pd.Series(index=s.index, dtype="object")
    ss = s.map(_to_str)
    null_mask = ss.map(is_null_like)
    out.loc[null_mask] = "NULL_LIKE"

    def bid(v: str) -> str:
        v = v.strip()
        if v == "":
            return "NULL_LIKE"
        L = len(v)
        if v.isdigit():
            return f"ID__DIGITS__LEN_{L}"
        prefix = v[:2].lower() if len(v) >= 2 else v.lower()
        return f"ID__PFX_{prefix}__LEN_{L}"

    out.loc[~null_mask] = ss.loc[~null_mask].map(bid)
    return out


# -----------------------------
# 3) 总入口：只对三类列分桶，其它列 SKIP
# -----------------------------
@dataclass
class AutoBucketerConfig:
    type_cfg: TypeDetectConfig = field(default_factory=TypeDetectConfig)
    bucket_cfg: BucketingConfig = field(default_factory=BucketingConfig)
    output_skip_columns: bool = True  # True: 其他列输出 SKIP；False: 不输出该列 bucket


@dataclass
class ColumnBucketMeta:
    col: str
    inferred_type: str
    bucket_method: str
    stats: Dict[str, Any]
    type_reason: Dict[str, Any]


def auto_bucket_dataframe_3types(
    df: pd.DataFrame,
    cfg: Optional[AutoBucketerConfig] = None,
) -> Tuple[pd.DataFrame, Dict[str, ColumnBucketMeta]]:
    if cfg is None:
        cfg = AutoBucketerConfig()

    bucket_cols: Dict[str, pd.Series] = {}
    meta: Dict[str, ColumnBucketMeta] = {}

    for col in df.columns:
        s = df[col]
        ti = infer_column_type_3way(s, cfg.type_cfg)
        t = ti.inferred_type

        if t == "numeric":
            buckets = bucket_numeric(s, cfg.bucket_cfg)
            method = f"numeric: type-bucket + MAD-extreme + qcut(q={cfg.bucket_cfg.numeric_qbins})"
        elif t == "categorical":
            buckets = bucket_categorical(s, cfg.bucket_cfg)
            method = "categorical: topk+rare"
        elif t == "id_like":
            buckets = bucket_id_like(s)
            method = "id_like: prefix+length"
        else:
            if not cfg.output_skip_columns:
                meta[col] = ColumnBucketMeta(
                    col=col,
                    inferred_type=t,
                    bucket_method="SKIP",
                    stats={"num_buckets": 0, "top_buckets": {}},
                    type_reason=ti.reason,
                )
                continue
            buckets = pd.Series(["SKIP"] * len(s), index=s.index, dtype="object")
            method = "SKIP"

        bucket_col_name = f"{col}__bucket"
        bucket_cols[bucket_col_name] = buckets

        vc = buckets.value_counts(dropna=False)
        stats = {"num_buckets": int(vc.shape[0]), "top_buckets": vc.head(10).to_dict()}

        meta[col] = ColumnBucketMeta(
            col=col,
            inferred_type=t,
            bucket_method=method,
            stats=stats,
            type_reason=ti.reason,
        )

    bucket_df = pd.DataFrame(bucket_cols, index=df.index)
    return bucket_df, meta


# -----------------------------
# 4) main：运行示例
# -----------------------------
if __name__ == "__main__":
    # 读取 CSV（若编码问题，把 encoding 改成 'gbk'）
    df = pd.read_csv("hospital data analysis.csv", encoding="utf-8", engine="python")

    cfg = AutoBucketerConfig(
        bucket_cfg=BucketingConfig(
            numeric_qbins=3,
            numeric_auto_extreme=True,     # ✅ 方案A开启
            numeric_extreme_k=10.0,        # 可调：8~12
            numeric_extreme_q=0.005,       # MAD退化回退分位数
            cat_topk=5,
            cat_min_count=20,
            cat_min_ratio=0.005,
        ),
        output_skip_columns=True
    )

    bucket_df, meta = auto_bucket_dataframe_3types(df, cfg)

    print(bucket_df.head())

    print("\n--- Meta ---")
    for col, m in meta.items():
        print(col, "=>", m.inferred_type, "|", m.bucket_method, "|", m.stats["num_buckets"])

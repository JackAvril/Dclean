"""
Movies whitelist feature builder, full logic-migrated from the Flights whitelist pipeline.

This script keeps the full Flights whitelist-building framework:
- candidate feature fallback
- allowed-cell fallback
- automatic prediction-label filtering
- row_id + column whitelist merge
- allowed metadata merge
- missing allowed-cell export
- filtered-out candidate feature export
- column-level diagnostics
- Movies metadata-consistency diagnostics

Flights-specific metadata such as flight/src/time consensus is migrated to Movies:
- movie_metadata_consistency
- duration/year/release/rating/count/list typed fields
- parsed_year / parsed_release_year / parsed_release_country
- movie_row_* context fields
- movie rule / risk / bucket fields

Inputs:
    repair_candidate_features_v2.csv
    predicted_error_positions_consensus_augmented.csv

Outputs:
    repair_candidate_features_infer_whitelist.csv
    repair_candidate_features_infer_whitelist_missing_allowed_cells.csv
    repair_candidate_features_infer_whitelist_filtered_out.csv
    repair_candidate_features_infer_whitelist_column_summary.csv
    repair_candidate_features_infer_whitelist_allowed_meta_summary.csv
"""

import os
from pathlib import Path
from typing import List, Optional

import pandas as pd


# ============================================================
# 1. 配置区：直接在这里改路径
# ============================================================

# 上一步 Movies 特征构造脚本输出的文件
INPUT_CANDIDATE_FEATURES = "repair_candidate_features_v2.csv"

# 白名单来源，支持三种格式：
# A. predicted_error_positions_consensus_augmented.csv：Movies consistency 扩展后的白名单
# B. predicted_error_positions.csv：至少包含 row_id,column，可选 value/final_pred_label
# C. inference_all_predictions.csv：包含 final_pred_label / final_pred_label_name，会自动筛选最终预测为 error 的 cell
INPUT_ALLOWED_CELLS = "predicted_error_positions_consensus_augmented.csv"

# 输出：如果是相对路径，会输出到当前运行目录
OUTPUT_INFER_FEATURES = "repair_candidate_features_infer_whitelist.csv"

# 是否额外输出缺失白名单 cell
OUTPUT_MISSING_ALLOWED_CELLS = True

# 是否额外输出被过滤掉的候选特征
OUTPUT_FILTERED_OUT_FEATURES = True

# 是否输出列级统计
OUTPUT_COLUMN_SUMMARY = True

# 是否输出 allowed metadata 摘要
OUTPUT_ALLOWED_META_SUMMARY = True

# 如果 INPUT_ALLOWED_CELLS 不存在，按顺序尝试这些 fallback。
FALLBACK_ALLOWED_CELLS = [
    "predicted_error_positions.csv",
    "/mnt/mydata/dq/projects/Splittree/test/Error Detection movies/active_cleaning_loop_runs_movies_v4/iter_3/infer/inference_all_predictions.csv",
]

# 如果 INPUT_CANDIDATE_FEATURES 不存在，按顺序尝试这些 fallback。
FALLBACK_CANDIDATE_FEATURES = [
    "/mnt/mydata/dq/projects/Splittree/test/Error Correction movies/repair_candidate_features_v2.csv",
]

# Movies 候选生成阶段输出的 consistency suspicious 文件，可选。
# 如果存在，会用于补充 allowed metadata，但不会单独扩大白名单；
# 扩大白名单应由 predicted_error_positions_consensus_augmented.csv 控制。
MOVIE_CONSISTENCY_SUSPICIOUS_CELLS = "movie_consistency_suspicious_cells.csv"

# Movies 新版候选生成脚本输出的 direct / metadata suspicious 文件，可选。
# 如果存在：
# 1) 会补充 allowed metadata；
# 2) 会作为额外 whitelist cell 合并进 allowed cells，避免新增候选被白名单阶段过滤掉。
MOVIE_DIRECT_METADATA_SUSPICIOUS_CELLS = "movie_direct_metadata_suspicious_cells.csv"

# Movies consistency summary，可选。仅用于诊断统计，不参与 merge。
MOVIE_METADATA_CONSISTENCY_FILE = "movie_metadata_consistency.csv"


# ============================================================
# 2. 工具函数
# ============================================================

def resolve_output_path(output_path, anchor_input_path=None):
    """
    输出路径适配：
    - 如果 output_path 是绝对路径，则直接使用；
    - 如果 output_path 是相对路径，则保存到当前运行目录。
    """
    output_path = str(output_path)
    if os.path.isabs(output_path):
        return Path(output_path)
    return Path(os.path.abspath(output_path))


def resolve_existing_file(path, fallback_paths=None, file_desc="file"):
    """
    如果 path 存在则返回 path；否则按 fallback_paths 依次查找。
    """
    path = str(path)
    if path and Path(path).expanduser().exists():
        return path

    for fb in fallback_paths or []:
        if fb and Path(str(fb)).expanduser().exists():
            print(f"[WARN] {file_desc} not found: {path}")
            print(f"[INFO] fallback {file_desc}: {fb}")
            return str(fb)

    raise FileNotFoundError(f"{file_desc} not found: {path}")


def normalize_key_columns(df, file_name):
    """
    统一 row_id / column 类型。
    """
    required = {"row_id", "column"}
    if not required.issubset(set(df.columns)):
        raise ValueError(
            f"{file_name} must contain columns {required}, "
            f"but got columns: {list(df.columns)}"
        )

    out = df.copy()
    out["row_id"] = pd.to_numeric(out["row_id"], errors="raise").astype(int)
    out["column"] = out["column"].astype(str).str.strip()
    return out


def safe_numeric_series(s, fill_value=0):
    return pd.to_numeric(s, errors="coerce").fillna(fill_value)


def filter_allowed_cells(allow_df):
    """
    自动兼容不同白名单输入：
    1. Movies consistency augmented whitelist：没有预测标签列，直接作为白名单；
    2. predicted_error_positions.csv：通常也没有预测标签列，直接作为白名单；
    3. inference_all_predictions.csv：有 final_pred_label / final_pred_label_name，自动筛 error；
    4. 兼容 pred_label / detector_pred_label / raw_pred_label 等字段。
    """
    df = allow_df.copy()

    if "final_pred_label" in df.columns:
        mask = safe_numeric_series(df["final_pred_label"], 0).astype(int) == 1
        print("[INFO] allowed cells source: final_pred_label == 1")
        return df.loc[mask].copy()

    if "final_pred_label_name" in df.columns:
        mask = df["final_pred_label_name"].astype(str).str.strip().str.lower() == "error"
        print("[INFO] allowed cells source: final_pred_label_name == error")
        return df.loc[mask].copy()

    if "pred_label" in df.columns:
        mask = safe_numeric_series(df["pred_label"], 0).astype(int) == 1
        print("[WARN] no final_pred_label found; fallback to pred_label == 1")
        return df.loc[mask].copy()

    if "pred_label_name" in df.columns:
        mask = df["pred_label_name"].astype(str).str.strip().str.lower() == "error"
        print("[WARN] no final_pred_label_name found; fallback to pred_label_name == error")
        return df.loc[mask].copy()

    if "detector_pred_label" in df.columns:
        mask = safe_numeric_series(df["detector_pred_label"], 0).astype(int) == 1
        print("[WARN] no final_pred_label found; fallback to detector_pred_label == 1")
        return df.loc[mask].copy()

    if "detector_pred_label_name" in df.columns:
        mask = df["detector_pred_label_name"].astype(str).str.strip().str.lower() == "error"
        print("[WARN] no final_pred_label_name found; fallback to detector_pred_label_name == error")
        return df.loc[mask].copy()

    if "raw_pred_label" in df.columns:
        mask = safe_numeric_series(df["raw_pred_label"], 0).astype(int) == 1
        print("[WARN] no final/detector label found; fallback to raw_pred_label == 1")
        return df.loc[mask].copy()

    if "raw_pred_label_name" in df.columns:
        mask = df["raw_pred_label_name"].astype(str).str.strip().str.lower() == "error"
        print("[WARN] no final/detector label found; fallback to raw_pred_label_name == error")
        return df.loc[mask].copy()

    print("[INFO] no prediction label columns found; treat input file itself as whitelist")
    return df.copy()


def maybe_read_csv(path, desc):
    if path is None or str(path).strip() == "":
        return pd.DataFrame()

    p = Path(str(path))
    if not p.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(p, encoding="utf-8-sig")
        print(f"[INFO] loaded {desc}: {p}, rows={len(df)}")
        return df
    except Exception as e:
        print(f"[WARN] failed to read {desc}: {p}, err={e}")
        return pd.DataFrame()


def add_prefix_to_meta_columns(meta_df, prefix="allowed_"):
    """
    将 allowed metadata 里非 key 字段加 allowed_ 前缀。
    如果字段已经存在 allowed_ 前缀，则不重复添加。
    """
    if meta_df.empty:
        return meta_df

    rename = {}
    for c in meta_df.columns:
        if c in {"row_id", "column"}:
            continue
        if str(c).startswith(prefix):
            continue
        rename[c] = f"{prefix}{c}"

    return meta_df.rename(columns=rename)


def numeric_col(df, col, default=0):
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


# ============================================================
# 3. Movies allowed metadata 字段
# ============================================================

MOVIES_ALLOWED_META_COLS = [
    # key/display
    "row_id", "column", "value",

    # Movies consistency augmented whitelist fields
    "consistency_name", "lhs_key", "consensus_value",
    "consensus_confidence", "consensus_margin",
    "consensus_support_count", "consensus_total_count", "consensus_reason",

    # Movies direct / metadata suspicious whitelist fields
    # 来自 movie_direct_metadata_suspicious_cells.csv 或 augmented whitelist。
    "direct_candidate_value", "direct_reason", "direct_confidence",
    "metadata_source", "direct_source", "direct_rule_name",
    "is_movie_direct_metadata_suspicious", "is_movie_direct_rule_suspicious",
    "is_movie_metadata_candidate_suspicious",

    # Movies detection / context fields
    "semantic_type", "detected_type", "violation_count", "conflict_score",
    "value_frequency", "value_frequency_rank",

    # Movies parsed fields
    "parsed_year", "parsed_release_year", "parsed_release_country",
    "duration_minutes", "rating_value_float", "count_value_int",
    "review_total", "list_token_count", "list_token_count_bucket",
    "text_length_bucket", "has_mojibake_or_control_chars",

    # Movies row-level context fields
    "movie_row_year", "movie_row_release_year", "movie_row_release_country",
    "movie_row_release_year_gap", "movie_row_duration_minutes",
    "movie_row_rating_value_float", "movie_row_rating_count_int",
    "movie_row_review_total", "movie_row_review_to_rating_ratio",
    "movie_row_actors_cast_overlap",

    # Neighbor / Bayesian fields
    "neighbor_majority_value", "neighbor_majority_ratio",
    "prior_error_probability", "posterior_error_probability",

    # Rule fields
    "main_rule_type", "main_usage_role", "strong_rule_count",
    "candidate_generation_rule_count", "fd_like_count", "context_rule_count",
    "global_rule_count", "rare_value_count", "typo_rule_count",
    "pattern_rule_count", "schema_rule_count", "movie_typed_rule_count",
    "movie_consistency_rule_count", "movie_typed_rule_ratio",
    "movie_consistency_rule_ratio", "movie_rule_total_count", "has_movie_rule_signal",

    # Movies column flags
    "is_movie_time_col", "is_movie_rating_col", "is_movie_people_col",
    "is_movie_list_col", "is_movie_text_col",

    # Buckets
    "pattern_bucket", "rarity_bucket", "neighbor_bucket",
    "movie_rule_bucket", "movie_field_bucket", "year_bucket",
    "release_year_gap_bucket", "duration_bucket", "rating_bucket",
    "count_bucket", "review_ratio_bucket", "actors_cast_overlap_bucket",
    "list_bucket", "text_bucket", "bucket_id", "cluster_id", "dist_to_center",

    # Raw / detector / final model outputs
    "raw_pred_error_prob", "raw_pred_label", "raw_pred_label_name",
    "raw_margin_to_threshold", "stage1_high_conf_error", "stage1_mid_zone",
    "detector_pred_error_prob_before_adjust", "detector_pred_error_prob",
    "detector_pred_label", "detector_pred_label_name",
    "detector_margin_to_threshold", "detector_threshold_used", "is_stage_disagree",
    "verifier_keep_error_prob", "verifier_decision", "verifier_threshold_used",
    "group_verifier_threshold", "verifier_bypass",
    "final_pred_label", "final_pred_label_name",
    "final_pred_error_prob_before_adjust", "final_pred_error_prob",
    "is_mid_prob", "is_near_threshold", "verifier_borderline",
    "verifier_reject_borderline",

    # Risk / hard-case fields
    "is_fp_risk_col", "rule_neighbor_conflict", "fp_conflict_fd_hardcase",
    "sample_borderline_hardcase", "recall_recovery_hardcase",
    "persistent_fp_high_risk", "is_hard_case", "is_high_risk",
]


def maybe_load_movie_consistency_suspicious(path):
    """
    可选读取 movie_consistency_suspicious_cells.csv。
    该文件由 Movies 候选生成脚本生成，字段一般包括：
    row_id,column,value,consistency_name,lhs_key,consensus_value,
    consensus_confidence,consensus_margin,consensus_support_count,
    consensus_total_count,consensus_reason
    """
    df = maybe_read_csv(path, "movie consistency suspicious cells")
    if df.empty:
        return pd.DataFrame()

    if not {"row_id", "column"}.issubset(set(df.columns)):
        print(f"[WARN] movie consistency suspicious file missing row_id/column, skipped: {path}")
        return pd.DataFrame()

    return normalize_key_columns(df, "movie consistency suspicious file")




def maybe_load_movie_direct_metadata_suspicious(path):
    """
    可选读取 movie_direct_metadata_suspicious_cells.csv。

    新版 Movies 候选生成脚本会输出该文件，用于把 direct-rule / metadata-rule
    suspicious cells 加入 whitelist，并保留候选来源证据。

    兼容字段示例：
    row_id,column,value,direct_candidate_value,direct_reason,direct_confidence,metadata_source
    """
    df = maybe_read_csv(path, "movie direct metadata suspicious cells")
    if df.empty:
        return pd.DataFrame()

    if not {"row_id", "column"}.issubset(set(df.columns)):
        print(f"[WARN] movie direct metadata suspicious file missing row_id/column, skipped: {path}")
        return pd.DataFrame()

    out = normalize_key_columns(df, "movie direct metadata suspicious file")

    # 标准化关键字段，保证后续 meta merge 和推理 gate 能稳定读取。
    if "direct_candidate_value" not in out.columns:
        for alt in ["candidate_value", "consensus_value", "repair_value", "repaired_value", "predicted_repair_value"]:
            if alt in out.columns:
                out["direct_candidate_value"] = out[alt]
                break
    if "direct_candidate_value" not in out.columns:
        out["direct_candidate_value"] = ""

    if "direct_reason" not in out.columns:
        for alt in ["reason", "gate_reason", "suspicious_reason", "rule_name", "source"]:
            if alt in out.columns:
                out["direct_reason"] = out[alt]
                break
    if "direct_reason" not in out.columns:
        out["direct_reason"] = "movie_direct_metadata_suspicious"

    if "direct_confidence" not in out.columns:
        for alt in ["confidence", "score", "source_score", "final_score"]:
            if alt in out.columns:
                out["direct_confidence"] = pd.to_numeric(out[alt], errors="coerce").fillna(1.0)
                break
    if "direct_confidence" not in out.columns:
        out["direct_confidence"] = 1.0

    if "metadata_source" not in out.columns:
        for alt in ["candidate_source", "source", "direct_source", "rule_source"]:
            if alt in out.columns:
                out["metadata_source"] = out[alt]
                break
    if "metadata_source" not in out.columns:
        out["metadata_source"] = out["direct_reason"]

    out["is_movie_direct_metadata_suspicious"] = 1
    out["is_movie_direct_rule_suspicious"] = (
        out["direct_reason"].astype(str).str.contains("direct|genre|release|year|format", case=False, na=False)
    ).astype(int)
    out["is_movie_metadata_candidate_suspicious"] = (
        out["metadata_source"].astype(str).str.contains("metadata|name_year|id_to|name_to|movie", case=False, na=False)
        | out["direct_reason"].astype(str).str.contains("metadata|name_year|id_to|name_to|movie", case=False, na=False)
    ).astype(int)

    return out


def merge_suspicious_meta(base_meta, suspicious_df, source_name="suspicious"):
    """
    用 row_id,column 把可选 suspicious 文件补充到 allow_meta。
    不要求字段完全一致；只要有 row_id,column 就能补。
    """
    if suspicious_df is None or suspicious_df.empty:
        return base_meta

    meta_cols = [c for c in MOVIES_ALLOWED_META_COLS if c in suspicious_df.columns]
    # direct 文件可能有额外诊断字段，也保留一部分常见字段。
    for c in [
        "direct_candidate_value", "direct_reason", "direct_confidence", "metadata_source",
        "direct_source", "direct_rule_name", "candidate_source", "candidate_value", "source_score",
        "final_score", "suspicious_reason", "rule_name",
    ]:
        if c in suspicious_df.columns and c not in meta_cols:
            meta_cols.append(c)

    if "row_id" not in meta_cols:
        meta_cols.insert(0, "row_id")
    if "column" not in meta_cols:
        meta_cols.insert(1, "column")

    suspicious_meta = suspicious_df[meta_cols].drop_duplicates(subset=["row_id", "column"], keep="first").copy()

    if base_meta is None or base_meta.empty:
        return suspicious_meta

    out = base_meta.merge(
        suspicious_meta,
        on=["row_id", "column"],
        how="left",
        suffixes=("", f"_from_{source_name}"),
    )

    for c in suspicious_meta.columns:
        if c in {"row_id", "column"}:
            continue
        cf = f"{c}_from_{source_name}"
        if cf in out.columns:
            if c not in out.columns:
                out[c] = out[cf]
            else:
                out[c] = out[c].where(out[c].notna(), out[cf])
            out = out.drop(columns=[cf])

    return out

def build_allow_meta(allow, consistency_suspicious_df=None, direct_suspicious_df=None):
    """
    构造 Movies allowed metadata。
    完整迁移 Flights 的 allow_meta merge 思路：
    - 保留 detection / verifier / bucket / risk 诊断字段；
    - 保留 Movies consistency / parsed fields / typed-rule 诊断字段；
    - 如果 movie_consistency_suspicious_cells.csv 存在，用 row_id,column 补充 consistency 信息。
    """
    allow_meta_cols = [c for c in MOVIES_ALLOWED_META_COLS if c in allow.columns]
    allow_meta = allow[allow_meta_cols].drop_duplicates(subset=["row_id", "column"], keep="first").copy()

    # 用 movie_consistency_suspicious_cells.csv 补充 consistency metadata。
    if consistency_suspicious_df is not None and not consistency_suspicious_df.empty:
        suspicious_cols = [
            c for c in [
                "row_id", "column", "value", "consistency_name", "lhs_key",
                "consensus_value", "consensus_confidence", "consensus_margin",
                "consensus_support_count", "consensus_total_count", "consensus_reason",
            ]
            if c in consistency_suspicious_df.columns
        ]
        suspicious_meta = consistency_suspicious_df[suspicious_cols].drop_duplicates(
            subset=["row_id", "column"], keep="first"
        ).copy()

        if allow_meta.empty:
            allow_meta = suspicious_meta
        else:
            allow_meta = allow_meta.merge(
                suspicious_meta,
                on=["row_id", "column"],
                how="left",
                suffixes=("", "_from_consistency_file"),
            )

            for c in [
                "value", "consistency_name", "lhs_key", "consensus_value",
                "consensus_confidence", "consensus_margin", "consensus_support_count",
                "consensus_total_count", "consensus_reason",
            ]:
                cf = f"{c}_from_consistency_file"
                if cf in allow_meta.columns:
                    if c not in allow_meta.columns:
                        allow_meta[c] = allow_meta[cf]
                    else:
                        allow_meta[c] = allow_meta[c].where(allow_meta[c].notna(), allow_meta[cf])
                    allow_meta = allow_meta.drop(columns=[cf])

    # 用 movie_direct_metadata_suspicious_cells.csv 补充 direct / metadata suspicious 字段。
    allow_meta = merge_suspicious_meta(allow_meta, direct_suspicious_df, source_name="direct_metadata")

    if allow_meta.empty:
        return allow_meta

    # Movies direct metadata suspicious 标记。
    if "is_movie_direct_metadata_suspicious" not in allow_meta.columns:
        if "direct_reason" in allow_meta.columns or "direct_candidate_value" in allow_meta.columns:
            allow_meta["is_movie_direct_metadata_suspicious"] = (
                allow_meta.get("direct_reason", pd.Series([None] * len(allow_meta), index=allow_meta.index)).notna()
                | allow_meta.get("direct_candidate_value", pd.Series([None] * len(allow_meta), index=allow_meta.index)).notna()
            ).astype(int)
        else:
            allow_meta["is_movie_direct_metadata_suspicious"] = 0

    if "is_movie_direct_rule_suspicious" not in allow_meta.columns:
        if "direct_reason" in allow_meta.columns:
            allow_meta["is_movie_direct_rule_suspicious"] = (
                allow_meta["direct_reason"].astype(str).str.contains("direct|genre|release|year|format", case=False, na=False)
            ).astype(int)
        else:
            allow_meta["is_movie_direct_rule_suspicious"] = 0

    if "is_movie_metadata_candidate_suspicious" not in allow_meta.columns:
        if "metadata_source" in allow_meta.columns or "direct_reason" in allow_meta.columns:
            metadata_text = (
                allow_meta.get("metadata_source", pd.Series([""] * len(allow_meta), index=allow_meta.index)).astype(str)
                + "|"
                + allow_meta.get("direct_reason", pd.Series([""] * len(allow_meta), index=allow_meta.index)).astype(str)
            )
            allow_meta["is_movie_metadata_candidate_suspicious"] = (
                metadata_text.str.contains("metadata|name_year|id_to|name_to|movie", case=False, na=False)
            ).astype(int)
        else:
            allow_meta["is_movie_metadata_candidate_suspicious"] = 0

    # Movies consistency 标记。
    if "consensus_reason" in allow_meta.columns:
        allow_meta["is_movie_metadata_consistency"] = (
            allow_meta["consensus_reason"].astype(str)
            .str.contains("movie_metadata_consistency", case=False, na=False)
            .astype(int)
        )
    elif "consistency_name" in allow_meta.columns:
        allow_meta["is_movie_metadata_consistency"] = allow_meta["consistency_name"].notna().astype(int)
    else:
        allow_meta["is_movie_metadata_consistency"] = 0

    conf = pd.to_numeric(allow_meta.get("consensus_confidence", 0), errors="coerce").fillna(0)
    margin = pd.to_numeric(allow_meta.get("consensus_margin", 0), errors="coerce").fillna(0)
    support = pd.to_numeric(allow_meta.get("consensus_support_count", 0), errors="coerce").fillna(0)

    allow_meta["is_strong_movie_consistency"] = (
        (allow_meta["is_movie_metadata_consistency"].astype(int) == 1)
        & (conf >= 0.70)
        & (margin >= 0.20)
        & (support >= 2)
    ).astype(int)

    # Movies typed field flags，若原 allowed 文件没有这些字段，则根据列名补齐。
    column_lower = allow_meta["column"].astype(str).str.strip().str.lower()
    if "is_movie_time_col" not in allow_meta.columns:
        allow_meta["is_movie_time_col"] = column_lower.isin(["year", "release date", "duration"]).astype(int)
    if "is_movie_rating_col" not in allow_meta.columns:
        allow_meta["is_movie_rating_col"] = column_lower.isin(["ratingvalue", "ratingcount", "reviewcount"]).astype(int)
    if "is_movie_people_col" not in allow_meta.columns:
        allow_meta["is_movie_people_col"] = column_lower.isin(["director", "creator", "actors", "cast"]).astype(int)
    if "is_movie_list_col" not in allow_meta.columns:
        allow_meta["is_movie_list_col"] = column_lower.isin(["actors", "cast", "creator", "genre", "filming locations"]).astype(int)
    if "is_movie_text_col" not in allow_meta.columns:
        allow_meta["is_movie_text_col"] = column_lower.isin(["name", "description"]).astype(int)

    # 如果风险字段缺失，补 0，保持下游特征稳定。
    for col in [
        "fp_conflict_fd_hardcase", "sample_borderline_hardcase",
        "recall_recovery_hardcase", "persistent_fp_high_risk",
        "is_hard_case", "is_high_risk",
    ]:
        if col not in allow_meta.columns:
            allow_meta[col] = 0

    return allow_meta


# ============================================================
# 4. 诊断统计
# ============================================================

def build_column_summary(cand, merged, allow_unique):
    cand_cells_before = (
        cand[["row_id", "column"]]
        .drop_duplicates()
        .groupby("column")
        .size()
        .reset_index(name="candidate_cell_count_before")
    )

    cand_rows_before = cand.groupby("column").size().reset_index(name="candidate_row_count_before")

    allow_cells = allow_unique.groupby("column").size().reset_index(name="allowed_cell_count")

    if merged.empty:
        cand_cells_after = pd.DataFrame(columns=["column", "candidate_cell_count_after"])
        cand_rows_after = pd.DataFrame(columns=["column", "candidate_row_count_after"])
    else:
        cand_cells_after = (
            merged[["row_id", "column"]]
            .drop_duplicates()
            .groupby("column")
            .size()
            .reset_index(name="candidate_cell_count_after")
        )
        cand_rows_after = merged.groupby("column").size().reset_index(name="candidate_row_count_after")

    out = allow_cells.merge(cand_cells_before, on="column", how="outer")
    out = out.merge(cand_rows_before, on="column", how="outer")
    out = out.merge(cand_cells_after, on="column", how="outer")
    out = out.merge(cand_rows_after, on="column", how="outer")
    out = out.fillna(0)

    for c in [
        "allowed_cell_count", "candidate_cell_count_before", "candidate_row_count_before",
        "candidate_cell_count_after", "candidate_row_count_after",
    ]:
        out[c] = out[c].astype(int)

    out["missing_allowed_cell_count"] = (
        out["allowed_cell_count"] - out["candidate_cell_count_after"]
    ).clip(lower=0)

    out["candidate_cell_keep_ratio"] = (
        out["candidate_cell_count_after"] / out["candidate_cell_count_before"].replace(0, pd.NA)
    ).fillna(0.0)

    out["candidate_row_keep_ratio"] = (
        out["candidate_row_count_after"] / out["candidate_row_count_before"].replace(0, pd.NA)
    ).fillna(0.0)

    out = out.sort_values(["allowed_cell_count", "candidate_cell_count_after"], ascending=[False, False])
    return out.reset_index(drop=True)


def build_allowed_meta_summary(merged):
    rows = []
    if merged.empty:
        return pd.DataFrame(rows)

    def count_cell_where(col, value=1):
        if col not in merged.columns:
            return 0
        s = pd.to_numeric(merged[col], errors="coerce").fillna(0)
        return merged[s == value][["row_id", "column"]].drop_duplicates().shape[0]

    for col in [
        "allowed_is_movie_metadata_consistency", "allowed_is_strong_movie_consistency",
        "allowed_is_movie_direct_metadata_suspicious", "allowed_is_movie_direct_rule_suspicious",
        "allowed_is_movie_metadata_candidate_suspicious",
        "allowed_is_movie_time_col", "allowed_is_movie_rating_col",
        "allowed_is_movie_people_col", "allowed_is_movie_list_col",
        "allowed_is_movie_text_col", "allowed_is_hard_case", "allowed_is_high_risk",
        "allowed_recall_recovery_hardcase", "allowed_sample_borderline_hardcase",
        "allowed_fp_conflict_fd_hardcase", "allowed_persistent_fp_high_risk",
    ]:
        if col in merged.columns:
            rows.append({
                "metric": col,
                "unique_cell_count": count_cell_where(col, 1),
                "candidate_row_count": int((pd.to_numeric(merged[col], errors="coerce").fillna(0) == 1).sum()),
            })

    if "allowed_consistency_name" in merged.columns:
        vc = (
            merged[["row_id", "column", "allowed_consistency_name"]]
            .drop_duplicates()
            .groupby("allowed_consistency_name", dropna=False)
            .size()
            .reset_index(name="unique_cell_count")
            .sort_values("unique_cell_count", ascending=False)
        )
        for _, r in vc.iterrows():
            rows.append({
                "metric": f"allowed_consistency_name={r['allowed_consistency_name']}",
                "unique_cell_count": int(r["unique_cell_count"]),
                "candidate_row_count": int(
                    (merged["allowed_consistency_name"].astype(str) == str(r["allowed_consistency_name"])).sum()
                ),
            })

    return pd.DataFrame(rows)


# ============================================================
# 5. 主逻辑
# ============================================================

def build_whitelist_features(candidate_features_file, allowed_cells_file, output_infer_features):
    candidate_features_file = resolve_existing_file(
        candidate_features_file,
        FALLBACK_CANDIDATE_FEATURES,
        file_desc="candidate features file",
    )
    allowed_cells_file = resolve_existing_file(
        allowed_cells_file,
        FALLBACK_ALLOWED_CELLS,
        file_desc="allowed cells file",
    )

    # ------------------------------------------------------------
    # A. 读取 candidate features
    # ------------------------------------------------------------
    cand = pd.read_csv(candidate_features_file, encoding="utf-8-sig")
    cand = normalize_key_columns(cand, "candidate features file")

    # ------------------------------------------------------------
    # B. 读取 allowed cells，并自动筛选最终 error cell
    # ------------------------------------------------------------
    allow_raw = pd.read_csv(allowed_cells_file, encoding="utf-8-sig")
    allow_raw = normalize_key_columns(allow_raw, "allowed cells file")
    allow = filter_allowed_cells(allow_raw)

    # Movies consistency suspicious metadata，可选补充
    consistency_suspicious_df = maybe_load_movie_consistency_suspicious(MOVIE_CONSISTENCY_SUSPICIOUS_CELLS)

    # Movies direct / metadata suspicious metadata，可选补充。
    # 新版候选生成会输出该文件；这里同时把它追加进 whitelist，避免被过滤。
    direct_suspicious_df = maybe_load_movie_direct_metadata_suspicious(MOVIE_DIRECT_METADATA_SUSPICIOUS_CELLS)
    if not direct_suspicious_df.empty:
        before_allow = allow[["row_id", "column"]].drop_duplicates().shape[0]
        append_cols = [c for c in direct_suspicious_df.columns if c in set(allow.columns) or c in set(MOVIES_ALLOWED_META_COLS)]
        if "row_id" not in append_cols:
            append_cols.insert(0, "row_id")
        if "column" not in append_cols:
            append_cols.insert(1, "column")
        direct_allowed = direct_suspicious_df[append_cols].copy()
        allow = pd.concat([allow, direct_allowed], ignore_index=True, sort=False)
        allow = normalize_key_columns(allow, "allowed cells + direct suspicious cells")
        allow = allow.drop_duplicates(subset=["row_id", "column"], keep="first").reset_index(drop=True)
        after_allow = allow[["row_id", "column"]].drop_duplicates().shape[0]
        print(f"[INFO] direct metadata suspicious expanded whitelist: {before_allow} -> {after_allow} (+{after_allow - before_allow})")

    allow_unique = allow[["row_id", "column"]].drop_duplicates().copy()

    # Movies consistency summary，仅诊断读取
    movie_consistency_summary_df = maybe_read_csv(MOVIE_METADATA_CONSISTENCY_FILE, "movie metadata consistency summary")

    allow_meta = build_allow_meta(allow, consistency_suspicious_df, direct_suspicious_df)
    allow_meta = add_prefix_to_meta_columns(allow_meta, prefix="allowed_")

    # ------------------------------------------------------------
    # C. 统计过滤前信息
    # ------------------------------------------------------------
    cand_unique_before = cand[["row_id", "column"]].drop_duplicates().shape[0]
    cand_rows_before = len(cand)
    allow_unique_count = len(allow_unique)

    # ------------------------------------------------------------
    # D. 按 row_id + column 做白名单过滤
    # ------------------------------------------------------------
    merged = cand.merge(
        allow_unique.assign(__keep__=1),
        on=["row_id", "column"],
        how="inner",
    ).drop(columns=["__keep__"])

    # ------------------------------------------------------------
    # D2. 合并 allowed metadata
    # ------------------------------------------------------------
    if not allow_meta.empty:
        # 避免与 cand 原字段冲突：allow_meta 已统一带 allowed_ 前缀
        merged = merged.merge(allow_meta, on=["row_id", "column"], how="left")

    # ------------------------------------------------------------
    # D3. Movies 白名单诊断特征补充
    # ------------------------------------------------------------
    if not merged.empty:
        if "allowed_consensus_value" in merged.columns:
            merged["candidate_equals_allowed_consensus_value"] = (
                merged["candidate_value"].astype(str).str.strip().str.lower()
                == merged["allowed_consensus_value"].astype(str).str.strip().str.lower()
            ).astype(int)
        else:
            merged["candidate_equals_allowed_consensus_value"] = 0

        if "allowed_direct_candidate_value" in merged.columns:
            merged["candidate_equals_allowed_direct_value"] = (
                merged["candidate_value"].astype(str).str.strip().str.lower()
                == merged["allowed_direct_candidate_value"].astype(str).str.strip().str.lower()
            ).astype(int)
        else:
            merged["candidate_equals_allowed_direct_value"] = 0

        merged["candidate_under_movie_direct_metadata_cell"] = numeric_col(
            merged, "allowed_is_movie_direct_metadata_suspicious", 0
        ).astype(int)
        merged["candidate_under_movie_direct_rule_cell"] = numeric_col(
            merged, "allowed_is_movie_direct_rule_suspicious", 0
        ).astype(int)
        merged["candidate_under_movie_metadata_candidate_cell"] = numeric_col(
            merged, "allowed_is_movie_metadata_candidate_suspicious", 0
        ).astype(int)

        if "allowed_direct_reason" in merged.columns:
            merged["candidate_allowed_direct_reason_match_source"] = (
                merged.get("candidate_source", pd.Series([""] * len(merged), index=merged.index)).astype(str).str.lower()
                .str.contains("direct|genre|release|year|rating|duration|metadata|name_year|id_to|name_to", regex=True, na=False)
                & merged["allowed_direct_reason"].notna()
            ).astype(int)
        else:
            merged["candidate_allowed_direct_reason_match_source"] = 0

        if "contains_movie_consistency_source" in merged.columns:
            src_flag = pd.to_numeric(merged["contains_movie_consistency_source"], errors="coerce").fillna(0).astype(int)
        elif "is_from_movie_consistency" in merged.columns:
            src_flag = pd.to_numeric(merged["is_from_movie_consistency"], errors="coerce").fillna(0).astype(int)
        else:
            src_flag = pd.Series([0] * len(merged), index=merged.index)

        if "allowed_is_movie_metadata_consistency" in merged.columns:
            allow_flag = pd.to_numeric(merged["allowed_is_movie_metadata_consistency"], errors="coerce").fillna(0).astype(int)
        else:
            allow_flag = pd.Series([0] * len(merged), index=merged.index)

        if "allowed_is_strong_movie_consistency" in merged.columns:
            strong_allow_flag = pd.to_numeric(merged["allowed_is_strong_movie_consistency"], errors="coerce").fillna(0).astype(int)
        else:
            strong_allow_flag = pd.Series([0] * len(merged), index=merged.index)

        merged["candidate_and_allowed_both_movie_consistency"] = ((src_flag == 1) & (allow_flag == 1)).astype(int)
        merged["candidate_under_strong_movie_consistency_cell"] = strong_allow_flag

        # Movies typed whitelist flags
        merged["candidate_under_movie_time_whitelist_cell"] = numeric_col(merged, "allowed_is_movie_time_col", 0).astype(int)
        merged["candidate_under_movie_rating_whitelist_cell"] = numeric_col(merged, "allowed_is_movie_rating_col", 0).astype(int)
        merged["candidate_under_movie_people_whitelist_cell"] = numeric_col(merged, "allowed_is_movie_people_col", 0).astype(int)
        merged["candidate_under_movie_list_whitelist_cell"] = numeric_col(merged, "allowed_is_movie_list_col", 0).astype(int)
        merged["candidate_under_movie_text_whitelist_cell"] = numeric_col(merged, "allowed_is_movie_text_col", 0).astype(int)

    cand_unique_after = merged[["row_id", "column"]].drop_duplicates().shape[0] if not merged.empty else 0
    cand_rows_after = len(merged)

    # ------------------------------------------------------------
    # E. 保存主输出
    # ------------------------------------------------------------
    output_path = resolve_output_path(output_infer_features, candidate_features_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------
    # F. 保存 allowed 里有，但 candidate features 里没有的 cell
    # ------------------------------------------------------------
    merged_cells = merged[["row_id", "column"]].drop_duplicates() if not merged.empty else pd.DataFrame(columns=["row_id", "column"])
    missing_allowed = allow_unique.merge(merged_cells, on=["row_id", "column"], how="left", indicator=True)
    missing_allowed = missing_allowed[missing_allowed["_merge"] == "left_only"].drop(columns=["_merge"])

    missing_path = None
    if OUTPUT_MISSING_ALLOWED_CELLS and len(missing_allowed) > 0:
        missing_path = output_path.with_name(output_path.stem + "_missing_allowed_cells.csv")
        missing_allowed.to_csv(missing_path, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------
    # G. 保存被过滤掉的候选特征，方便排查
    # ------------------------------------------------------------
    filtered_out_path = None
    if OUTPUT_FILTERED_OUT_FEATURES:
        filtered_out = cand.merge(allow_unique.assign(__keep__=1), on=["row_id", "column"], how="left")
        filtered_out = filtered_out[filtered_out["__keep__"].isna()].drop(columns=["__keep__"])

        if len(filtered_out) > 0:
            filtered_out_path = output_path.with_name(output_path.stem + "_filtered_out.csv")
            filtered_out.to_csv(filtered_out_path, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------
    # H. 保存列级诊断统计
    # ------------------------------------------------------------
    column_summary_path = None
    if OUTPUT_COLUMN_SUMMARY:
        column_summary = build_column_summary(cand, merged, allow_unique)
        column_summary_path = output_path.with_name(output_path.stem + "_column_summary.csv")
        column_summary.to_csv(column_summary_path, index=False, encoding="utf-8-sig")
    else:
        column_summary = pd.DataFrame()

    # ------------------------------------------------------------
    # I. 保存 allowed meta 诊断统计
    # ------------------------------------------------------------
    allowed_meta_summary_path = None
    if OUTPUT_ALLOWED_META_SUMMARY:
        allowed_meta_summary = build_allowed_meta_summary(merged)
        if not allowed_meta_summary.empty:
            allowed_meta_summary_path = output_path.with_name(output_path.stem + "_allowed_meta_summary.csv")
            allowed_meta_summary.to_csv(allowed_meta_summary_path, index=False, encoding="utf-8-sig")
    else:
        allowed_meta_summary = pd.DataFrame()

    # ------------------------------------------------------------
    # J. 打印统计信息
    # ------------------------------------------------------------
    print("\n========== Build Movies infer candidate features from whitelist ==========")
    print(f"input_candidate_features: {candidate_features_file}")
    print(f"input_allowed_cells: {allowed_cells_file}")
    print(f"output_infer_features: {output_path}")
    print("")
    print("---- Cell-level statistics ----")
    print(f"candidate_unique_cells_before: {cand_unique_before}")
    print(f"allowed_unique_cells: {allow_unique_count}")
    print(f"candidate_unique_cells_after: {cand_unique_after}")
    print(f"missing_allowed_cells: {len(missing_allowed)}")
    print("")
    print("---- Row-level statistics ----")
    print(f"candidate_rows_before: {cand_rows_before}")
    print(f"candidate_rows_after: {cand_rows_after}")
    print(f"filtered_out_rows: {cand_rows_before - cand_rows_after}")
    print("")

    if missing_path is not None:
        print("[WARN] Some whitelist cells are not present in candidate features.")
        print(f"[WARN] missing allowed cells saved to: {missing_path}")

    if filtered_out_path is not None:
        print(f"[INFO] filtered-out candidate features saved to: {filtered_out_path}")

    if column_summary_path is not None:
        print(f"[INFO] column summary saved to: {column_summary_path}")

    if allowed_meta_summary_path is not None:
        print(f"[INFO] allowed meta summary saved to: {allowed_meta_summary_path}")

    # ------------------------------------------------------------
    # K. Movies consistency whitelist statistics
    # ------------------------------------------------------------
    print("")
    print("---- Movies whitelist diagnostic statistics ----")

    if "allowed_is_movie_metadata_consistency" in merged.columns:
        consistency_cells_kept = (
            merged[numeric_col(merged, "allowed_is_movie_metadata_consistency", 0).astype(int) == 1]
            [["row_id", "column"]]
            .drop_duplicates()
            .shape[0]
        )
        print(f"movie_metadata_consistency_allowed_cells_kept: {consistency_cells_kept}")

    if "allowed_is_strong_movie_consistency" in merged.columns:
        strong_consistency_cells_kept = (
            merged[numeric_col(merged, "allowed_is_strong_movie_consistency", 0).astype(int) == 1]
            [["row_id", "column"]]
            .drop_duplicates()
            .shape[0]
        )
        print(f"strong_movie_consistency_allowed_cells_kept: {strong_consistency_cells_kept}")

    if "candidate_equals_allowed_consensus_value" in merged.columns:
        matched_consensus_candidate_cells = (
            merged[numeric_col(merged, "candidate_equals_allowed_consensus_value", 0).astype(int) == 1]
            [["row_id", "column"]]
            .drop_duplicates()
            .shape[0]
        )
        print(f"candidate_equals_allowed_consensus_value_cells: {matched_consensus_candidate_cells}")

    if "candidate_equals_allowed_direct_value" in merged.columns:
        matched_direct_candidate_cells = (
            merged[numeric_col(merged, "candidate_equals_allowed_direct_value", 0).astype(int) == 1]
            [["row_id", "column"]]
            .drop_duplicates()
            .shape[0]
        )
        print(f"candidate_equals_allowed_direct_value_cells: {matched_direct_candidate_cells}")

    for direct_flag_col in [
        "candidate_under_movie_direct_metadata_cell",
        "candidate_under_movie_direct_rule_cell",
        "candidate_under_movie_metadata_candidate_cell",
    ]:
        if direct_flag_col in merged.columns:
            cnt = (
                merged[numeric_col(merged, direct_flag_col, 0).astype(int) == 1]
                [["row_id", "column"]]
                .drop_duplicates()
                .shape[0]
            )
            print(f"{direct_flag_col}_cells_kept: {cnt}")

    if "candidate_and_allowed_both_movie_consistency" in merged.columns:
        both_consistency_cells = (
            merged[numeric_col(merged, "candidate_and_allowed_both_movie_consistency", 0).astype(int) == 1]
            [["row_id", "column"]]
            .drop_duplicates()
            .shape[0]
        )
        print(f"candidate_and_allowed_both_movie_consistency_cells: {both_consistency_cells}")

    for flag_col in [
        "allowed_is_movie_direct_metadata_suspicious", "allowed_is_movie_direct_rule_suspicious",
        "allowed_is_movie_metadata_candidate_suspicious",
        "allowed_is_movie_time_col", "allowed_is_movie_rating_col",
        "allowed_is_movie_people_col", "allowed_is_movie_list_col", "allowed_is_movie_text_col",
        "allowed_fp_conflict_fd_hardcase", "allowed_sample_borderline_hardcase",
        "allowed_recall_recovery_hardcase", "allowed_persistent_fp_high_risk",
        "allowed_is_hard_case", "allowed_is_high_risk",
    ]:
        if flag_col in merged.columns:
            cnt = (
                merged[numeric_col(merged, flag_col, 0).astype(int) == 1]
                [["row_id", "column"]]
                .drop_duplicates()
                .shape[0]
            )
            print(f"{flag_col}_cells_kept: {cnt}")

    if not movie_consistency_summary_df.empty:
        print("")
        print("---- Movie metadata consistency summary source ----")
        print(f"movie_metadata_consistency_rows: {len(movie_consistency_summary_df)}")
        if "is_high_confidence" in movie_consistency_summary_df.columns:
            high_conf_count = int(
                pd.to_numeric(movie_consistency_summary_df["is_high_confidence"], errors="coerce")
                .fillna(0)
                .astype(int)
                .sum()
            )
            print(f"movie_metadata_consistency_high_confidence_rows: {high_conf_count}")

    print("")
    print("---- Top allowed columns ----")
    if not column_summary.empty:
        show_cols = [
            "column", "allowed_cell_count", "candidate_cell_count_before",
            "candidate_cell_count_after", "missing_allowed_cell_count", "candidate_row_count_after",
        ]
        show_cols = [c for c in show_cols if c in column_summary.columns]
        print(column_summary[show_cols].head(30).to_string(index=False))

    print(f"\n[OK] saved to: {output_path.resolve()}")

    return merged


# ============================================================
# 6. 主入口
# ============================================================

def main():
    build_whitelist_features(
        candidate_features_file=INPUT_CANDIDATE_FEATURES,
        allowed_cells_file=INPUT_ALLOWED_CELLS,
        output_infer_features=OUTPUT_INFER_FEATURES,
    )


if __name__ == "__main__":
    main()

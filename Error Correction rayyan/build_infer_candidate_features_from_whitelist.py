"""
Rayyan whitelist feature builder, logic-migrated from the Flights whitelist pipeline.

Migration principle:
- Keep the full Flights whitelist-building framework:
  candidate feature fallback, allowed-cell fallback, prediction-label filtering,
  whitelist merge, missing allowed-cell diagnostics, filtered-out feature export,
  allowed metadata merge, and consensus statistics.
- Convert Flights-specific diagnostics:
  flight/src/time consensus
      -> Rayyan metadata consensus / canonical source / language / ISSN / date diagnostics.
- Preserve downstream compatibility:
  input:  repair_candidate_features_v2.csv
  input:  predicted_error_positions_consensus_augmented.csv or predicted_error_positions.csv
  output: repair_candidate_features_infer_whitelist.csv
"""

import os
import pandas as pd
from pathlib import Path


# ============================================================
# 1. 配置区：直接在这里改路径
# ============================================================

# 上一步 Rayyan 特征构造脚本输出的文件
INPUT_CANDIDATE_FEATURES = "repair_candidate_features_v2.csv"

# 白名单来源，支持三种格式：
# A. allowed_cells_from_detection.csv：至少包含 row_id,column
# B. predicted_error_positions.csv：至少包含 row_id,column，可选 value/final_pred_label
# C. inference_all_predictions.csv：包含 final_pred_label / final_pred_label_name，会自动筛选最终预测为 error 的 cell
# D. predicted_error_positions_consensus_augmented.csv：Rayyan metadata consensus 增强白名单
INPUT_ALLOWED_CELLS = "predicted_error_positions_consensus_augmented.csv"

# 输出：如果是相对路径，会输出到当前运行目录
OUTPUT_INFER_FEATURES = "repair_candidate_features_infer_whitelist.csv"

# 是否额外输出缺失白名单 cell
OUTPUT_MISSING_ALLOWED_CELLS = True

# 是否额外输出被过滤掉的候选特征
OUTPUT_FILTERED_OUT_FEATURES = True

# 如果 INPUT_ALLOWED_CELLS 不存在，按顺序尝试这些 fallback。
# Rayyan 逻辑迁移版候选生成脚本会优先输出 predicted_error_positions_consensus_augmented.csv；
# 如果未启用 consensus，则回退 predicted_error_positions.csv；
# 如果前两者都不存在，则直接读取检测推理结果。
FALLBACK_ALLOWED_CELLS = [
    "predicted_error_positions.csv",
    "/mnt/mydata/dq/projects/Splittree/test/Error Detection rayyan/active_cleaning_loop_runs_rayyan_v4/iter_3/infer/inference_all_predictions.csv",
]

# 如果 INPUT_CANDIDATE_FEATURES 不存在，按顺序尝试这些 fallback。
FALLBACK_CANDIDATE_FEATURES = [
    "/mnt/mydata/dq/projects/Splittree/test/Error Correction rayyan/repair_candidate_features_v2.csv",
]

# Rayyan consensus suspicious cells，可选。
# 如果存在，会用于补充 allowed metadata 诊断字段，但不会单独扩大白名单；
# 扩大白名单应由 predicted_error_positions_consensus_augmented.csv 控制。
RAYYAN_CONSENSUS_SUSPICIOUS_CELLS = "rayyan_consensus_suspicious_cells.csv"


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


def filter_allowed_cells(allow_df):
    """
    自动兼容不同白名单输入：
    1. Rayyan consensus augmented whitelist：没有预测标签列，直接作为白名单；
    2. predicted_error_positions.csv：通常也没有预测标签列，直接作为白名单；
    3. inference_all_predictions.csv：有 final_pred_label / final_pred_label_name，自动筛 error；
    4. 兼容 pred_label / pred_label_name / detector_pred_label / raw_pred_label 等字段。
    """
    df = allow_df.copy()

    if "final_pred_label" in df.columns:
        mask = pd.to_numeric(df["final_pred_label"], errors="coerce").fillna(0).astype(int) == 1
        print("[INFO] allowed cells source: final_pred_label == 1")
        return df.loc[mask].copy()

    if "final_pred_label_name" in df.columns:
        mask = df["final_pred_label_name"].astype(str).str.strip().str.lower() == "error"
        print("[INFO] allowed cells source: final_pred_label_name == error")
        return df.loc[mask].copy()

    if "pred_label" in df.columns:
        mask = pd.to_numeric(df["pred_label"], errors="coerce").fillna(0).astype(int) == 1
        print("[WARN] no final_pred_label found; fallback to pred_label == 1")
        return df.loc[mask].copy()

    if "pred_label_name" in df.columns:
        mask = df["pred_label_name"].astype(str).str.strip().str.lower() == "error"
        print("[WARN] no final_pred_label_name found; fallback to pred_label_name == error")
        return df.loc[mask].copy()

    if "detector_pred_label" in df.columns:
        mask = pd.to_numeric(df["detector_pred_label"], errors="coerce").fillna(0).astype(int) == 1
        print("[WARN] no final_pred_label found; fallback to detector_pred_label == 1")
        return df.loc[mask].copy()

    if "detector_pred_label_name" in df.columns:
        mask = df["detector_pred_label_name"].astype(str).str.strip().str.lower() == "error"
        print("[WARN] no final_pred_label_name found; fallback to detector_pred_label_name == error")
        return df.loc[mask].copy()

    if "raw_pred_label" in df.columns:
        mask = pd.to_numeric(df["raw_pred_label"], errors="coerce").fillna(0).astype(int) == 1
        print("[WARN] no final/detector label found; fallback to raw_pred_label == 1")
        return df.loc[mask].copy()

    if "raw_pred_label_name" in df.columns:
        mask = df["raw_pred_label_name"].astype(str).str.strip().str.lower() == "error"
        print("[WARN] no final/detector label found; fallback to raw_pred_label_name == error")
        return df.loc[mask].copy()

    print("[INFO] no prediction label columns found; treat input file itself as whitelist")
    return df.copy()


def maybe_load_rayyan_consensus_suspicious(path):
    """
    可选读取 rayyan_consensus_suspicious_cells.csv。
    该文件由 Rayyan 候选生成逻辑迁移版生成，字段一般包括：
    row_id,column,value,consensus_name,lhs_key,consensus_value,
    consensus_confidence,consensus_margin,consensus_support_count,
    consensus_total_count,consensus_reason
    """
    if path is None or str(path).strip() == "":
        return pd.DataFrame()

    p = Path(str(path))
    if not p.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(p, encoding="utf-8-sig")
    except Exception as e:
        print(f"[WARN] failed to read Rayyan consensus suspicious cells: {p}, err={e}")
        return pd.DataFrame()

    if not {"row_id", "column"}.issubset(set(df.columns)):
        print(f"[WARN] Rayyan consensus suspicious file missing row_id/column, skipped: {p}")
        return pd.DataFrame()

    df = normalize_key_columns(df, "Rayyan consensus suspicious file")
    return df


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
        if c.startswith(prefix):
            continue
        rename[c] = f"{prefix}{c}"

    return meta_df.rename(columns=rename)


def safe_numeric_count(df, col):
    if col not in df.columns:
        return 0
    return int(pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int).sum())


# ============================================================
# 3. Rayyan metadata 字段配置
# ============================================================

RAYYAN_ALLOWED_META_COLS = [
    # key/display
    "row_id", "column", "value",

    # Rayyan consensus augmented whitelist / suspicious cells
    "consensus_name",
    "lhs_key",
    "consensus_value",
    "consensus_confidence",
    "consensus_margin",
    "consensus_support_count",
    "consensus_total_count",
    "consensus_reason",

    # Rayyan detection flat fields
    "semantic_type",
    "violation_count",
    "conflict_score",
    "value_frequency",
    "value_frequency_rank",
    "numeric_value",
    "date_value",
    "language_canonical",
    "issn_canonical",
    "neighbor_majority_value",
    "neighbor_majority_ratio",
    "prior_error_probability",
    "posterior_error_probability",
    "main_rule_type",
    "main_usage_role",
    "strong_rule_count",
    "candidate_generation_rule_count",
    "fd_like_count",
    "context_rule_count",
    "global_rule_count",
    "rare_value_count",
    "typo_rule_count",
    "pattern_rule_count",
    "schema_rule_count",
    "canonical_rule_count",

    # Rayyan buckets / risk
    "pattern_bucket",
    "rarity_bucket",
    "neighbor_bucket",
    "canonical_bucket",
    "date_value_bucket",
    "language_bucket",
    "bucket_id",
    "cluster_id",
    "dist_to_center",
    "rare_only_signal",
    "is_weak_only",
    "is_fp_risk_col",
    "rule_neighbor_conflict",
    "fp_conflict_fd_hardcase",
    "sample_borderline_hardcase",
    "recall_recovery_hardcase",
    "persistent_fp_high_risk",
    "is_hard_case",
    "is_high_risk",

    # detector/verifier/final outputs
    "raw_pred_error_prob",
    "raw_pred_label",
    "raw_pred_label_name",
    "raw_margin_to_threshold",
    "stage1_high_conf_error",
    "stage1_mid_zone",
    "detector_pred_error_prob_before_adjust",
    "detector_pred_error_prob",
    "detector_pred_label",
    "detector_pred_label_name",
    "detector_margin_to_threshold",
    "detector_threshold_used",
    "is_stage_disagree",
    "verifier_keep_error_prob",
    "verifier_decision",
    "verifier_threshold_used",
    "group_verifier_threshold",
    "verifier_bypass",
    "final_pred_label",
    "final_pred_label_name",
    "final_pred_error_prob_before_adjust",
    "final_pred_error_prob",
    "is_mid_prob",
    "is_near_threshold",
    "verifier_borderline",
    "verifier_reject_borderline",
]


def build_allow_meta(allow, consensus_suspicious_df):
    """
    构造 Rayyan allowed metadata。
    完整迁移 Flights 的 allowed meta merge 思路：
    - 保留 detection / verifier / bucket / risk 诊断字段；
    - 保留 Rayyan metadata consensus 诊断字段；
    - 如果 consensus suspicious 文件存在，用 row_id,column 补充 consensus 信息。
    """
    allow_meta_cols = [c for c in RAYYAN_ALLOWED_META_COLS if c in allow.columns]
    allow_meta = allow[allow_meta_cols].drop_duplicates(subset=["row_id", "column"], keep="first").copy()

    # 用 consensus_suspicious_df 补充 consensus metadata。
    if consensus_suspicious_df is not None and not consensus_suspicious_df.empty:
        suspicious_cols = [
            c for c in [
                "row_id", "column", "value",
                "consensus_name", "lhs_key", "consensus_value",
                "consensus_confidence", "consensus_margin",
                "consensus_support_count", "consensus_total_count",
                "consensus_reason",
            ]
            if c in consensus_suspicious_df.columns
        ]
        suspicious_meta = consensus_suspicious_df[suspicious_cols].drop_duplicates(subset=["row_id", "column"], keep="first").copy()

        if allow_meta.empty:
            allow_meta = suspicious_meta
        else:
            # 只补充 allow_meta 中没有的 consensus 字段，避免覆盖原白名单字段。
            allow_meta = allow_meta.merge(
                suspicious_meta,
                on=["row_id", "column"],
                how="left",
                suffixes=("", "_from_consensus_file")
            )

            for c in [
                "value",
                "consensus_name",
                "lhs_key",
                "consensus_value",
                "consensus_confidence",
                "consensus_margin",
                "consensus_support_count",
                "consensus_total_count",
                "consensus_reason",
            ]:
                cf = f"{c}_from_consensus_file"
                if cf in allow_meta.columns:
                    if c not in allow_meta.columns:
                        allow_meta[c] = allow_meta[cf]
                    else:
                        allow_meta[c] = allow_meta[c].where(allow_meta[c].notna(), allow_meta[cf])
                    allow_meta = allow_meta.drop(columns=[cf])

    if not allow_meta.empty:
        # Rayyan metadata consensus 诊断字段。
        if "consensus_reason" in allow_meta.columns:
            allow_meta["is_rayyan_metadata_consensus"] = (
                allow_meta["consensus_reason"]
                .astype(str)
                .str.contains("rayyan_metadata_consensus", case=False, na=False)
                .astype(int)
            )
        elif "consensus_name" in allow_meta.columns:
            allow_meta["is_rayyan_metadata_consensus"] = (
                allow_meta["consensus_name"].notna().astype(int)
            )
        else:
            allow_meta["is_rayyan_metadata_consensus"] = 0

        # 强 consensus：置信度、margin、support 同时满足。
        conf = pd.to_numeric(allow_meta.get("consensus_confidence", 0), errors="coerce").fillna(0)
        margin = pd.to_numeric(allow_meta.get("consensus_margin", 0), errors="coerce").fillna(0)
        support = pd.to_numeric(allow_meta.get("consensus_support_count", 0), errors="coerce").fillna(0)
        allow_meta["is_strong_rayyan_consensus"] = (
            (allow_meta["is_rayyan_metadata_consensus"].astype(int) == 1)
            & (conf >= 0.62)
            & (margin >= 0.20)
            & (support >= 2)
        ).astype(int)

        # Canonical / recovery hard-case diagnostics.
        for col in [
            "fp_conflict_fd_hardcase",
            "sample_borderline_hardcase",
            "recall_recovery_hardcase",
            "persistent_fp_high_risk",
            "is_hard_case",
            "is_high_risk",
        ]:
            if col not in allow_meta.columns:
                allow_meta[col] = 0

    return allow_meta


# ============================================================
# 4. 主逻辑
# ============================================================

def build_whitelist_features(
    candidate_features_file,
    allowed_cells_file,
    output_infer_features,
):
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

    allow_unique = allow[["row_id", "column"]].drop_duplicates().copy()

    # 可选读取 Rayyan consensus suspicious cells，用于补充 metadata。
    consensus_suspicious_df = maybe_load_rayyan_consensus_suspicious(RAYYAN_CONSENSUS_SUSPICIOUS_CELLS)
    if not consensus_suspicious_df.empty:
        print(f"[INFO] loaded Rayyan consensus suspicious cells: {RAYYAN_CONSENSUS_SUSPICIOUS_CELLS}, rows={len(consensus_suspicious_df)}")

    allow_meta = build_allow_meta(allow, consensus_suspicious_df)
    allow_meta = add_prefix_to_meta_columns(allow_meta, prefix="allowed_")

    # ------------------------------------------------------------
    # C. 统计过滤前信息
    # ------------------------------------------------------------
    cand_unique_before = cand[["row_id", "column"]].drop_duplicates().shape[0]
    cand_rows_before = len(cand)

    allow_column_counts = allow_unique["column"].value_counts().reset_index()
    allow_column_counts.columns = ["column", "allowed_cell_count"]

    cand_column_counts_before = cand[["row_id", "column"]].drop_duplicates()["column"].value_counts().reset_index()
    cand_column_counts_before.columns = ["column", "candidate_cell_count_before"]

    # ------------------------------------------------------------
    # D. 按 row_id + column 做白名单过滤
    # ------------------------------------------------------------
    merged = cand.merge(
        allow_unique.assign(__keep__=1),
        on=["row_id", "column"],
        how="inner",
    ).drop(columns=["__keep__"])

    if not allow_meta.empty:
        merged = merged.merge(
            allow_meta,
            on=["row_id", "column"],
            how="left",
        )

    # ------------------------------------------------------------
    # D2. Rayyan 白名单诊断特征补充
    # ------------------------------------------------------------
    if not merged.empty:
        # 候选是否与白名单 consensus value 相等。
        if "allowed_consensus_value" in merged.columns:
            merged["candidate_equals_allowed_consensus_value"] = (
                merged["candidate_value"].astype(str).str.strip().str.lower()
                == merged["allowed_consensus_value"].astype(str).str.strip().str.lower()
            ).astype(int)
        else:
            merged["candidate_equals_allowed_consensus_value"] = 0

        # 候选是否来自 Rayyan metadata consensus。
        if "contains_rayyan_consensus_source" in merged.columns:
            src_flag = pd.to_numeric(merged["contains_rayyan_consensus_source"], errors="coerce").fillna(0).astype(int)
        elif "is_from_rayyan_consensus" in merged.columns:
            src_flag = pd.to_numeric(merged["is_from_rayyan_consensus"], errors="coerce").fillna(0).astype(int)
        else:
            src_flag = pd.Series([0] * len(merged), index=merged.index)

        if "allowed_is_rayyan_metadata_consensus" in merged.columns:
            allow_flag = pd.to_numeric(merged["allowed_is_rayyan_metadata_consensus"], errors="coerce").fillna(0).astype(int)
        else:
            allow_flag = pd.Series([0] * len(merged), index=merged.index)

        merged["candidate_and_allowed_both_rayyan_consensus"] = ((src_flag == 1) & (allow_flag == 1)).astype(int)

        if "allowed_is_strong_rayyan_consensus" in merged.columns:
            strong_allow_flag = pd.to_numeric(merged["allowed_is_strong_rayyan_consensus"], errors="coerce").fillna(0).astype(int)
        else:
            strong_allow_flag = pd.Series([0] * len(merged), index=merged.index)

        merged["candidate_under_strong_rayyan_consensus_cell"] = strong_allow_flag

    cand_unique_after = merged[["row_id", "column"]].drop_duplicates().shape[0]
    cand_rows_after = len(merged)

    cand_column_counts_after = merged[["row_id", "column"]].drop_duplicates()["column"].value_counts().reset_index()
    cand_column_counts_after.columns = ["column", "candidate_cell_count_after"]

    # ------------------------------------------------------------
    # E. 保存主输出
    # ------------------------------------------------------------
    output_path = resolve_output_path(output_infer_features, candidate_features_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------
    # F. 保存 allowed 里有，但 candidate features 里没有的 cell
    # ------------------------------------------------------------
    missing_allowed = allow_unique.merge(
        merged[["row_id", "column"]].drop_duplicates(),
        on=["row_id", "column"],
        how="left",
        indicator=True,
    )
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
        filtered_out = cand.merge(
            allow_unique.assign(__keep__=1),
            on=["row_id", "column"],
            how="left",
        )
        filtered_out = filtered_out[filtered_out["__keep__"].isna()].drop(columns=["__keep__"])

        if len(filtered_out) > 0:
            filtered_out_path = output_path.with_name(output_path.stem + "_filtered_out.csv")
            filtered_out.to_csv(filtered_out_path, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------
    # H. 保存列级诊断统计
    # ------------------------------------------------------------
    column_summary = allow_column_counts.merge(cand_column_counts_before, on="column", how="outer")
    column_summary = column_summary.merge(cand_column_counts_after, on="column", how="outer")
    column_summary = column_summary.fillna(0)
    for c in ["allowed_cell_count", "candidate_cell_count_before", "candidate_cell_count_after"]:
        column_summary[c] = column_summary[c].astype(int)

    column_summary["missing_allowed_cell_count"] = (
        column_summary["allowed_cell_count"] - column_summary["candidate_cell_count_after"]
    ).clip(lower=0)

    column_summary_path = output_path.with_name(output_path.stem + "_column_summary.csv")
    column_summary.sort_values(["allowed_cell_count", "candidate_cell_count_after"], ascending=[False, False]).to_csv(
        column_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------
    # I. 打印统计信息
    # ------------------------------------------------------------
    print("\n========== Build Rayyan infer candidate features from whitelist ==========")
    print(f"input_candidate_features: {candidate_features_file}")
    print(f"input_allowed_cells: {allowed_cells_file}")
    print(f"output_infer_features: {output_path}")
    print("")
    print("---- Cell-level statistics ----")
    print(f"candidate_unique_cells_before: {cand_unique_before}")
    print(f"allowed_unique_cells: {len(allow_unique)}")
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

    print(f"[INFO] column summary saved to: {column_summary_path}")

    # ------------------------------------------------------------
    # J. Rayyan consensus whitelist statistics
    # ------------------------------------------------------------
    print("")
    print("---- Rayyan whitelist diagnostic statistics ----")

    if "allowed_is_rayyan_metadata_consensus" in merged.columns:
        consensus_cells_kept = (
            merged[pd.to_numeric(merged["allowed_is_rayyan_metadata_consensus"], errors="coerce").fillna(0).astype(int) == 1]
            [["row_id", "column"]]
            .drop_duplicates()
            .shape[0]
        )
        print(f"rayyan_metadata_consensus_allowed_cells_kept: {consensus_cells_kept}")

    if "allowed_is_strong_rayyan_consensus" in merged.columns:
        strong_consensus_cells_kept = (
            merged[pd.to_numeric(merged["allowed_is_strong_rayyan_consensus"], errors="coerce").fillna(0).astype(int) == 1]
            [["row_id", "column"]]
            .drop_duplicates()
            .shape[0]
        )
        print(f"strong_rayyan_consensus_allowed_cells_kept: {strong_consensus_cells_kept}")

    if "candidate_equals_allowed_consensus_value" in merged.columns:
        matched_consensus_candidate_cells = (
            merged[pd.to_numeric(merged["candidate_equals_allowed_consensus_value"], errors="coerce").fillna(0).astype(int) == 1]
            [["row_id", "column"]]
            .drop_duplicates()
            .shape[0]
        )
        print(f"candidate_equals_allowed_consensus_value_cells: {matched_consensus_candidate_cells}")

    if "candidate_and_allowed_both_rayyan_consensus" in merged.columns:
        both_consensus_cells = (
            merged[pd.to_numeric(merged["candidate_and_allowed_both_rayyan_consensus"], errors="coerce").fillna(0).astype(int) == 1]
            [["row_id", "column"]]
            .drop_duplicates()
            .shape[0]
        )
        print(f"candidate_and_allowed_both_rayyan_consensus_cells: {both_consensus_cells}")

    for flag_col in [
        "allowed_fp_conflict_fd_hardcase",
        "allowed_sample_borderline_hardcase",
        "allowed_recall_recovery_hardcase",
        "allowed_persistent_fp_high_risk",
        "allowed_is_hard_case",
        "allowed_is_high_risk",
    ]:
        if flag_col in merged.columns:
            cnt = (
                merged[pd.to_numeric(merged[flag_col], errors="coerce").fillna(0).astype(int) == 1]
                [["row_id", "column"]]
                .drop_duplicates()
                .shape[0]
            )
            print(f"{flag_col}_cells_kept: {cnt}")

    print("")
    print("---- Top allowed columns ----")
    if not column_summary.empty:
        print(column_summary.sort_values("allowed_cell_count", ascending=False).head(20).to_string(index=False))

    print(f"\n[OK] saved to: {output_path.resolve()}")

    return merged


# ============================================================
# 5. 主入口
# ============================================================

def main():
    build_whitelist_features(
        candidate_features_file=INPUT_CANDIDATE_FEATURES,
        allowed_cells_file=INPUT_ALLOWED_CELLS,
        output_infer_features=OUTPUT_INFER_FEATURES,
    )


if __name__ == "__main__":
    main()

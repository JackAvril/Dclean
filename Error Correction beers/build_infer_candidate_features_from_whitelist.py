import os
import pandas as pd
from pathlib import Path


# ============================================================
# 1. 配置区：直接在这里改路径
# ============================================================

# 上一步特征构造脚本输出的文件
INPUT_CANDIDATE_FEATURES = "repair_candidate_features_v2.csv"

# 白名单来源，支持三种格式：
# A. allowed_cells_from_detection.csv：至少包含 row_id,column
# B. predicted_error_positions.csv：至少包含 row_id,column，可选 value/final_pred_label
# C. inference_all_predictions.csv：包含 final_pred_label / final_pred_label_name，会自动筛选最终预测为 error 的 cell
# 默认使用 Beers 错误位置文件 predicted_error_positions.csv。
# 如果直接使用检测输出 inference_all_predictions.csv，也会自动按 final_pred_label 筛选。
INPUT_ALLOWED_CELLS = "predicted_error_positions.csv"

# 输出：如果是相对路径，会输出到当前运行目录
OUTPUT_INFER_FEATURES = "repair_candidate_features_infer_whitelist.csv"

# 是否额外输出缺失白名单 cell
OUTPUT_MISSING_ALLOWED_CELLS = True

# 是否额外输出被过滤掉的候选特征
OUTPUT_FILTERED_OUT_FEATURES = True

# 如果 INPUT_ALLOWED_CELLS 不存在，按顺序尝试这些 fallback。
# 这样你可以在不用 consensus 的时候仍然直接运行。
FALLBACK_ALLOWED_CELLS = [
    "/mnt/mydata/dq/projects/Splittree/test/Error Detection beers/active_cleaning_loop_runs_beers_v4/iter_3/infer/inference_all_predictions.csv",
]

# 如果 INPUT_CANDIDATE_FEATURES 不存在，按顺序尝试这些 fallback。
FALLBACK_CANDIDATE_FEATURES = [
    "/mnt/mydata/dq/projects/Splittree/test/Error Correction beers/repair_candidate_features_v2.csv",
]


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
    1. predicted_error_positions.csv / allowed_cells_from_detection.csv：没有预测标签列，直接作为白名单；
    2. inference_all_predictions.csv：有 final_pred_label / final_pred_label_name，自动筛 error；
    3. 兼容 pred_label / pred_label_name / detector_pred_label 等字段。
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

    print("[INFO] no prediction label columns found; treat input file itself as whitelist")
    return df.copy()


def build_whitelist_features(
    candidate_features_file,
    allowed_cells_file,
    output_infer_features,
):
    candidate_features_file = resolve_existing_file(
        candidate_features_file,
        FALLBACK_CANDIDATE_FEATURES,
        file_desc="candidate features file"
    )
    allowed_cells_file = resolve_existing_file(
        allowed_cells_file,
        FALLBACK_ALLOWED_CELLS,
        file_desc="allowed cells file"
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

    # 保留 Beers 检测/定位阶段的诊断字段，merge 到输出中便于后续排查。
    allow_meta_cols = [
        c for c in [
            "row_id", "column",
            "value",
            "semantic_type",
            "violation_count",
            "conflict_score",
            "numeric_value",
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
            "numeric_window_rule_count",
            "numeric_window_bucket",
            "numeric_value_bucket",
            "rare_only_signal",
            "is_weak_only",
            "is_fp_risk_col",
            "rule_neighbor_conflict",
            "numeric_borderline_hardcase",
            "semantic_recovery_hardcase",
            "persistent_fp_high_risk",
            "ounces_canonical_reject_hardcase",
            "is_hard_case",
            "is_high_risk",
            "final_pred_label",
            "final_pred_label_name",
            "final_pred_error_prob",
            "detector_pred_error_prob",
            "verifier_decision",
        ]
        if c in allow.columns
    ]

    allow_meta = allow[allow_meta_cols].drop_duplicates(subset=["row_id", "column"], keep="first").copy()

    # Beers 专属诊断标记：用于看哪些白名单 cell 来自数值边界/语义恢复/ounces 规范化风险。
    if not allow_meta.empty:
        for flag_col in [
            "numeric_borderline_hardcase",
            "semantic_recovery_hardcase",
            "persistent_fp_high_risk",
            "ounces_canonical_reject_hardcase",
            "is_hard_case",
            "is_high_risk",
        ]:
            if flag_col not in allow_meta.columns:
                allow_meta[flag_col] = 0

    # ------------------------------------------------------------
    # C. 统计过滤前信息
    # ------------------------------------------------------------
    cand_unique_before = cand[["row_id", "column"]].drop_duplicates().shape[0]
    cand_rows_before = len(cand)

    # ------------------------------------------------------------
    # D. 按 row_id + column 做白名单过滤
    # ------------------------------------------------------------
    merged = cand.merge(
        allow_unique.assign(__keep__=1),
        on=["row_id", "column"],
        how="inner"
    ).drop(columns=["__keep__"])

    if not allow_meta.empty:
        meta_cols_to_add = [c for c in allow_meta.columns if c not in {"row_id", "column"}]
        rename_map = {c: f"allowed_{c}" for c in meta_cols_to_add if c in merged.columns}
        allow_meta_renamed = allow_meta.rename(columns=rename_map)
        merged = merged.merge(
            allow_meta_renamed,
            on=["row_id", "column"],
            how="left"
        )

    cand_unique_after = merged[["row_id", "column"]].drop_duplicates().shape[0]
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
    missing_allowed = allow_unique.merge(
        merged[["row_id", "column"]].drop_duplicates(),
        on=["row_id", "column"],
        how="left",
        indicator=True
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
            how="left"
        )
        filtered_out = filtered_out[filtered_out["__keep__"].isna()].drop(columns=["__keep__"])

        if len(filtered_out) > 0:
            filtered_out_path = output_path.with_name(output_path.stem + "_filtered_out.csv")
            filtered_out.to_csv(filtered_out_path, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------
    # H. 打印统计信息
    # ------------------------------------------------------------
    print("\n========== Build Beers infer candidate features from whitelist ==========")
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
        print(f"[WARN] Some whitelist cells are not present in candidate features.")
        print(f"[WARN] missing allowed cells saved to: {missing_path}")

    if filtered_out_path is not None:
        print(f"[INFO] filtered-out candidate features saved to: {filtered_out_path}")

    beers_diag_cols = [
        "allowed_numeric_borderline_hardcase",
        "allowed_semantic_recovery_hardcase",
        "allowed_persistent_fp_high_risk",
        "allowed_ounces_canonical_reject_hardcase",
        "allowed_is_hard_case",
        "allowed_is_high_risk",
    ]
    existing_diag_cols = [c for c in beers_diag_cols if c in merged.columns]
    if existing_diag_cols:
        print("")
        print("---- Beers whitelist diagnostic statistics ----")
        for c in existing_diag_cols:
            cnt = (
                merged[pd.to_numeric(merged[c], errors="coerce").fillna(0).astype(int) == 1]
                [["row_id", "column"]]
                .drop_duplicates()
                .shape[0]
            )
            print(f"{c}_cells_kept: {cnt}")

    print(f"[OK] saved to: {output_path.resolve()}")

    return merged


# ============================================================
# 3. 主入口
# ============================================================

def main():
    build_whitelist_features(
        candidate_features_file=INPUT_CANDIDATE_FEATURES,
        allowed_cells_file=INPUT_ALLOWED_CELLS,
        output_infer_features=OUTPUT_INFER_FEATURES,
    )


if __name__ == "__main__":
    main()

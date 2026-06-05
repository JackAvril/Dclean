import os
import pandas as pd
from typing import Set, Tuple, Dict, Any, List


# ============================================================
# 1. 用户配置区
# ============================================================

CLEAN_CSV = "/mnt/mydata/dq/projects/Splittree/soccer/soccer_clean.csv"
DIRTY_CSV = "/mnt/mydata/dq/projects/Splittree/soccer/soccer_dirty.csv"

# 默认评估普通推理输出。
# 如果你跑的是主动学习主控脚本，可改成：
# "/mnt/mydata/dq/projects/Splittree/test/Error Detection Soccer/active_cleaning_loop_runs_soccer_v4/iter_3_infer/inference_all_predictions.csv"
# 或 best_model 对应的推理目录输出。
INFER_RESULT_CSV = (
    "/mnt/mydata/dq/projects/Splittree/test/Error Detection soccer/"
    "active_cleaning_loop_runs_soccer_v4/iter_3_infer/inference_all_predictions.csv"
)

OUTPUT_DIR = "eval_outputs_soccer_v4_with_verifier"

DROP_UNNAMED_COLUMNS = True

MISSING_TOKENS = {
    "", "empty", "null", "none", "nan", "na", "n/a", "missing", "unk",
    "unknown", "?", "not available", "contact airline"
}

# soccer 数据集列，主要用于后续按列统计和明细字段排序
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

# Soccer v2：与规则池、缩减候选、训练/推理脚本保持一致。
# Soccer v2：所有 soccer 列都作为候选目标列。
# name / surname 不再是 context-only。
PRIMARY_TARGET_COLUMNS = set(SOCCER_COLUMNS)

CONTEXT_ONLY_COLUMNS = set()

SOCCER_STRONG_RULE_TYPES = {
    "birthyear_numeric_range",
    "season_numeric_range",
    "position_domain",
    "birthyear_season_age_consistency",
}

SOCCER_FORMAT_RULE_TYPES = {
    "name_pattern",
    "surname_pattern",
    "manager_name_pattern",
    "team_name_pattern",
    "birthplace_location_pattern",
    "city_location_pattern",
    "stadium_name_pattern",
    "pattern",
    "generic_regex",
}

SOCCER_CONTEXT_RULE_TYPES = {
    "team_context_dominant",
    "team_season_manager_consistency_hint",
    "city_stadium_consistency_hint",
    "dominant_value_by_context",
    "dominant_value_by_context_pair",
}

# Soccer v2：soft-FD 只作为 evidence-only 诊断，不再计入强 FD-like signal。
SOCCER_FD_RULE_TYPES = {
    "functional_dependency",
}

SOCCER_EVIDENCE_ONLY_FD_RULE_TYPES = {
    "soft_functional_dependency",
}

# 是否额外输出按列统计、按规则统计
EXPORT_GROUP_SUMMARIES = True


# ============================================================
# 2. 基础函数
# ============================================================

def maybe_drop_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not DROP_UNNAMED_COLUMNS:
        return df.copy()
    keep_cols = [c for c in df.columns if not str(c).lower().startswith("unnamed:")]
    return df[keep_cols].copy()


def normalize_value(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.lower() in MISSING_TOKENS:
        return None
    return s


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        df[col] = df[col].map(normalize_value)
    return df


def safe_int(x, default=0):
    try:
        if pd.isna(x):
            return default
        return int(float(x))
    except Exception:
        return default


def safe_float(x, default=None):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def load_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df = maybe_drop_unnamed_columns(df)
    df = normalize_df(df)
    return df


def make_key(row_id: int, col: str) -> Tuple[int, str]:
    return int(row_id), str(col)


def bool_from_any(v, default=False) -> bool:
    if pd.isna(v):
        return default
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "error"}:
        return True
    if s in {"0", "false", "no", "n", "correct"}:
        return False
    try:
        return int(float(s)) == 1
    except Exception:
        return default


def as_int01(v, default=0) -> int:
    if pd.isna(v) or v is None:
        return default
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "error"}:
        return 1
    if s in {"0", "false", "no", "n", "correct"}:
        return 0
    try:
        return int(float(s))
    except Exception:
        return default


def safe_str(v, default=""):
    if pd.isna(v) or v is None:
        return default
    return str(v)


def safe_num_series(df: pd.DataFrame, col: str, default=0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series([default] * len(df), index=df.index)


def safe_str_series(df: pd.DataFrame, col: str, default="") -> pd.Series:
    if col in df.columns:
        return df[col].fillna(default).astype(str)
    return pd.Series([default] * len(df), index=df.index)


def add_eval_soccer_flags_to_infer_df(infer_df: pd.DataFrame) -> pd.DataFrame:
    """给评估阶段补齐 Soccer v2 分组字段，避免旧推理文件缺列时报错。"""
    df = infer_df.copy()

    numeric_flag_cols = [
        "target_is_primary_column",
        "target_is_context_only_column",
        "birthyear_value_invalid",
        "season_value_invalid",
        "position_value_invalid",
        "position_needs_canonicalization",
        "birthyear_season_age_conflict",
        "team_context_conflict",
        "format_rule_signal",
        "fd_like_signal",
        "evidence_only_fd_signal",
        "position_invalid_or_normalization",
        "has_birthyear_or_season_signal",
        "has_team_context_signal",
        "has_fd_like_signal", "evidence_only_fd_signal",
        "is_context_only_weak_signal",
        "soccer_primary_target_signal",
    ]

    for c in numeric_flag_cols:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    if "soccer_signal_strength" not in df.columns:
        df["soccer_signal_strength"] = 0.0
    df["soccer_signal_strength"] = pd.to_numeric(df["soccer_signal_strength"], errors="coerce").fillna(0.0)

    if "soccer_attribution_bucket" not in df.columns:
        df["soccer_attribution_bucket"] = "unknown_attribution"
    df["soccer_attribution_bucket"] = df["soccer_attribution_bucket"].fillna("unknown_attribution").astype(str)

    col = df["column"].astype(str)
    rule = df["main_rule_type"].astype(str) if "main_rule_type" in df.columns else pd.Series([""] * len(df), index=df.index)

    # v2 强制纠偏：所有 soccer 列都是 target；没有 context-only 列。
    df["target_is_primary_column"] = col.isin(PRIMARY_TARGET_COLUMNS).astype(int)
    df["target_is_context_only_column"] = col.isin(CONTEXT_ONLY_COLUMNS).astype(int)

    # soft-FD 降级：只作为 evidence-only，不作为 FD-like 强信号。
    existing_evidence_only_fd_signal = safe_num_series(df, "evidence_only_fd_signal", 0).astype(int)
    df["evidence_only_fd_signal"] = (
        (existing_evidence_only_fd_signal == 1)
        | rule.isin(SOCCER_EVIDENCE_ONLY_FD_RULE_TYPES)
    ).astype(int)

    if "evidence_only_fd_count" not in df.columns:
        df["evidence_only_fd_count"] = 0
    df["evidence_only_fd_count"] = pd.to_numeric(df["evidence_only_fd_count"], errors="coerce").fillna(0)
    df.loc[rule.isin(SOCCER_EVIDENCE_ONLY_FD_RULE_TYPES), "evidence_only_fd_count"] = (
        df.loc[rule.isin(SOCCER_EVIDENCE_ONLY_FD_RULE_TYPES), "evidence_only_fd_count"].clip(lower=1)
    )

    df["target_is_primary_column"] = ((df["target_is_primary_column"] == 1) | col.isin(PRIMARY_TARGET_COLUMNS)).astype(int)
    df["target_is_context_only_column"] = col.isin(CONTEXT_ONLY_COLUMNS).astype(int)

    df["birthyear_value_invalid"] = ((df["birthyear_value_invalid"] == 1) | rule.eq("birthyear_numeric_range")).astype(int)
    df["season_value_invalid"] = ((df["season_value_invalid"] == 1) | rule.eq("season_numeric_range")).astype(int)
    df["position_value_invalid"] = ((df["position_value_invalid"] == 1) | rule.eq("position_domain")).astype(int)
    df["birthyear_season_age_conflict"] = ((df["birthyear_season_age_conflict"] == 1) | rule.eq("birthyear_season_age_consistency")).astype(int)
    df["team_context_conflict"] = ((df["team_context_conflict"] == 1) | rule.isin(SOCCER_CONTEXT_RULE_TYPES)).astype(int)
    df["format_rule_signal"] = ((df["format_rule_signal"] == 1) | rule.isin(SOCCER_FORMAT_RULE_TYPES)).astype(int)
    df["fd_like_signal"] = ((df["fd_like_signal"] == 1) | rule.isin(SOCCER_FD_RULE_TYPES)).astype(int)

    df["position_invalid_or_normalization"] = (
        (df["position_invalid_or_normalization"] == 1)
        | (df["position_value_invalid"] == 1)
        | (df["position_needs_canonicalization"] == 1)
    ).astype(int)

    df["has_birthyear_or_season_signal"] = (
        (df["has_birthyear_or_season_signal"] == 1)
        | (df["birthyear_value_invalid"] == 1)
        | (df["season_value_invalid"] == 1)
        | (df["birthyear_season_age_conflict"] == 1)
    ).astype(int)

    df["has_team_context_signal"] = ((df["has_team_context_signal"] == 1) | (df["team_context_conflict"] == 1)).astype(int)

    fd_like_count = safe_num_series(df, "fd_like_count", 0)
    df["has_fd_like_signal"] = (
        (df["has_fd_like_signal"] == 1)
        | (df["fd_like_signal"] == 1)
        | ((fd_like_count > 0) & (~rule.isin(SOCCER_EVIDENCE_ONLY_FD_RULE_TYPES)))
    ).astype(int)

    # 如果 main_rule_type 是 soft-FD，则不允许仅因此变成 FD-like 强信号。
    df.loc[rule.isin(SOCCER_EVIDENCE_ONLY_FD_RULE_TYPES), ["fd_like_signal", "has_fd_like_signal"]] = 0

    soccer_domain = safe_num_series(df, "soccer_domain_rule_count", 0)
    soccer_format = safe_num_series(df, "soccer_format_rule_count", 0)
    soccer_numeric = safe_num_series(df, "soccer_numeric_rule_count", 0)
    soccer_cons = safe_num_series(df, "soccer_consistency_rule_count", 0)

    df["is_context_only_weak_signal"] = (
        (df["target_is_context_only_column"] == 1)
        & (soccer_domain <= 0)
        & (soccer_format <= 0)
        & (soccer_numeric <= 0)
        & (soccer_cons <= 0)
        & (~rule.isin(SOCCER_STRONG_RULE_TYPES | SOCCER_FORMAT_RULE_TYPES))
    ).astype(int)

    df["soccer_primary_target_signal"] = (
        (df["target_is_primary_column"] == 1)
        & (
            (df["has_birthyear_or_season_signal"] == 1)
            | (df["position_invalid_or_normalization"] == 1)
            | (df["format_rule_signal"] == 1)
            | (soccer_domain > 0)
            | (soccer_format > 0)
            | (soccer_numeric > 0)
            | (soccer_cons > 0)
        )
    ).astype(int)

    need_bucket = df["soccer_attribution_bucket"].isin(["", "missing", "unknown", "unknown_attribution", "nan", "None"])
    df.loc[need_bucket & (df["birthyear_value_invalid"] == 1), "soccer_attribution_bucket"] = "birthyear_invalid"
    df.loc[need_bucket & (df["season_value_invalid"] == 1), "soccer_attribution_bucket"] = "season_invalid"
    df.loc[need_bucket & (df["position_value_invalid"] == 1), "soccer_attribution_bucket"] = "position_invalid"
    df.loc[need_bucket & (df["position_needs_canonicalization"] == 1), "soccer_attribution_bucket"] = "position_canonicalization"
    df.loc[need_bucket & (df["birthyear_season_age_conflict"] == 1), "soccer_attribution_bucket"] = "birthyear_season_age_conflict"
    df.loc[need_bucket & (df["team_context_conflict"] == 1), "soccer_attribution_bucket"] = "team_context_conflict"
    df.loc[need_bucket & (df["format_rule_signal"] == 1), "soccer_attribution_bucket"] = "format_pattern_conflict"
    df.loc[need_bucket & (df["fd_like_signal"] == 1), "soccer_attribution_bucket"] = "fd_like_conflict"
    df.loc[need_bucket & col.isin(PRIMARY_TARGET_COLUMNS), "soccer_attribution_bucket"] = "primary_target_other"
    df.loc[need_bucket & col.isin(CONTEXT_ONLY_COLUMNS), "soccer_attribution_bucket"] = "context_only_other"

    return df


# ============================================================
# 3. 构造真实错误集合
# ============================================================

def build_true_error_cells(clean_df: pd.DataFrame, dirty_df: pd.DataFrame) -> Tuple[Set[Tuple[int, str]], Dict[Tuple[int, str], Dict[str, Any]]]:
    if clean_df.shape != dirty_df.shape:
        raise ValueError(f"干净表和脏表形状不一致: clean={clean_df.shape}, dirty={dirty_df.shape}")

    if list(clean_df.columns) != list(dirty_df.columns):
        raise ValueError(
            "干净表和脏表列名或列顺序不一致，请先对齐。\n"
            f"clean columns={list(clean_df.columns)}\n"
            f"dirty columns={list(dirty_df.columns)}"
        )

    true_errors = set()
    true_error_info = {}

    for row_id in range(len(clean_df)):
        for col in clean_df.columns:
            clean_val = clean_df.iat[row_id, clean_df.columns.get_loc(col)]
            dirty_val = dirty_df.iat[row_id, dirty_df.columns.get_loc(col)]
            if clean_val != dirty_val:
                key = (row_id, col)
                true_errors.add(key)
                true_error_info[key] = {"clean_value": clean_val, "dirty_value": dirty_val}

    return true_errors, true_error_info


# ============================================================
# 4. 解析推理结果
# ============================================================

def parse_label_from_value(v) -> bool:
    if pd.isna(v):
        return False
    val = str(v).strip().lower()

    if val in {"1", "true", "yes", "y"}:
        return True
    if val in {"0", "false", "no", "n"}:
        return False

    if val == "error":
        return True
    if val == "correct":
        return False

    try:
        return int(float(val)) == 1
    except Exception:
        return False


def parse_final_error_flag(row: pd.Series) -> bool:
    for col in ["final_pred_label", "pred_label", "raw_pred_label"]:
        if col in row.index and pd.notna(row[col]):
            return parse_label_from_value(row[col])
    for col in ["final_pred_label_name", "pred_label_name", "raw_pred_label_name"]:
        if col in row.index and pd.notna(row[col]):
            return parse_label_from_value(row[col])
    return False


def parse_detector_error_flag(row: pd.Series) -> bool:
    for col in ["detector_pred_label", "pred_label", "raw_pred_label"]:
        if col in row.index and pd.notna(row[col]):
            return parse_label_from_value(row[col])
    for col in ["detector_pred_label_name", "pred_label_name", "raw_pred_label_name"]:
        if col in row.index and pd.notna(row[col]):
            return parse_label_from_value(row[col])
    return False


def load_infer_result(path: str):
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df = maybe_drop_unnamed_columns(df)
    df = add_eval_soccer_flags_to_infer_df(df)

    required = ["row_id", "column"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"推理结果文件缺少必要列: {missing}")

    df["row_id"] = df["row_id"].apply(lambda x: safe_int(x, -1))
    df["column"] = df["column"].astype(str)
    df = df[df["row_id"] >= 0].copy()

    candidate_cells = set((int(r["row_id"]), str(r["column"])) for _, r in df.iterrows())

    predicted_error_cells = set()
    detector_predicted_error_cells = set()
    infer_row_map = {}

    for _, row in df.iterrows():
        key = (int(row["row_id"]), str(row["column"]))
        row_dict = row.to_dict()
        infer_row_map[key] = row_dict
        if parse_final_error_flag(row):
            predicted_error_cells.add(key)
        if parse_detector_error_flag(row):
            detector_predicted_error_cells.add(key)

    return df, candidate_cells, predicted_error_cells, detector_predicted_error_cells, infer_row_map


# ============================================================
# 5. 指标计算
# ============================================================

def calc_prf(pred_set: Set[Tuple[int, str]], true_set: Set[Tuple[int, str]]) -> Dict[str, Any]:
    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    precision = tp / len(pred_set) if len(pred_set) > 0 else 0.0
    recall = tp / len(true_set) if len(true_set) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def calc_column_metrics(pred_set: Set[Tuple[int, str]], true_set: Set[Tuple[int, str]], columns: List[str]) -> pd.DataFrame:
    rows = []
    for col in columns:
        true_col = set(k for k in true_set if k[1] == col)
        pred_col = set(k for k in pred_set if k[1] == col)
        prf = calc_prf(pred_col, true_col)
        rows.append({
            "column": col,
            "true_errors": len(true_col),
            "predicted_errors": len(pred_col),
            "tp": prf["tp"], "fp": prf["fp"], "fn": prf["fn"],
            "precision": prf["precision"], "recall": prf["recall"], "f1": prf["f1"],
        })
    return pd.DataFrame(rows)


def calc_candidate_column_metrics(candidate_set: Set[Tuple[int, str]], true_set: Set[Tuple[int, str]], columns: List[str]) -> pd.DataFrame:
    rows = []
    for col in columns:
        true_col = set(k for k in true_set if k[1] == col)
        cand_col = set(k for k in candidate_set if k[1] == col)
        covered = cand_col & true_col
        missed = true_col - cand_col
        coverage = len(covered) / len(true_col) if len(true_col) > 0 else 0.0
        cand_precision = len(covered) / len(cand_col) if len(cand_col) > 0 else 0.0
        rows.append({
            "column": col,
            "candidate_cells": len(cand_col),
            "true_errors": len(true_col),
            "covered_true_errors": len(covered),
            "missed_true_errors": len(missed),
            "candidate_coverage": coverage,
            "candidate_only_precision": cand_precision,
        })
    return pd.DataFrame(rows)


# ============================================================
# 6. 明细导出
# ============================================================

def get_threshold_value(infer_info: Dict[str, Any]):
    for c in [
        "final_decision_threshold", "verifier_decision_threshold", "detector_decision_threshold",
        "decision_threshold", "stage2_decision_threshold", "stage1_decision_threshold",
        "detector_threshold_used", "detector_threshold", "group_verifier_threshold", "verifier_threshold_used",
    ]:
        if c in infer_info:
            return infer_info.get(c)
    return None


def get_clean_dirty_value(row_id: int, col: str, clean_df: pd.DataFrame, dirty_df: pd.DataFrame):
    dirty_value = dirty_df.iloc[row_id][col] if row_id < len(dirty_df) and col in dirty_df.columns else None
    clean_value = clean_df.iloc[row_id][col] if row_id < len(clean_df) and col in clean_df.columns else None
    return clean_value, dirty_value


def build_detail_rows(keys: Set[Tuple[int, str]], clean_df: pd.DataFrame, dirty_df: pd.DataFrame,
                      infer_row_map: Dict[Tuple[int, str], Dict[str, Any]],
                      true_error_info: Dict[Tuple[int, str], Dict[str, Any]], detail_type: str) -> List[Dict[str, Any]]:
    rows = []
    useful_cols = [
        "value", "label_source", "semantic_type", "detected_type",
        "violation_count", "conflict_score", "value_frequency", "value_frequency_rank",
        "neighbor_majority_value", "neighbor_majority_ratio", "prior_error_probability", "posterior_error_probability",
        "main_rule_type", "main_usage_role", "strong_rule_count", "candidate_generation_rule_count",
        "fd_like_count", "evidence_only_fd_count", "evidence_only_fd_signal",
        "context_rule_count", "global_rule_count", "rare_value_count", "typo_rule_count",
        "pattern_rule_count", "schema_rule_count",
        "soccer_domain_rule_count", "soccer_format_rule_count", "soccer_numeric_rule_count",
        "soccer_consistency_rule_count", "soccer_profile_inconsistency_count", "soccer_consistency_score",
        "soccer_evidence_flags", "year_value", "canonical_position", "domain_valid", "soccer_value_bucket",
        "row_birthyear_int", "row_season_int", "row_age_at_season", "row_position_canonical",
        "target_is_primary_column", "target_is_context_only_column", "birthyear_value_invalid",
        "season_value_invalid", "position_value_invalid", "position_needs_canonicalization",
        "birthyear_season_age_conflict", "team_context_conflict", "format_rule_signal", "fd_like_signal",
        "position_invalid_or_normalization", "has_birthyear_or_season_signal", "has_team_context_signal",
        "has_fd_like_signal", "is_context_only_weak_signal", "soccer_signal_strength",
        "soccer_primary_target_signal", "soccer_attribution_bucket",
        "adult_domain_rule_count", "adult_format_rule_count", "adult_consistency_rule_count",
        "adult_v2_attribution_bucket", "time_window_rule_count", "time_value_minutes",
        "candidate_rule_ratio", "rule_total_count", "strong_rule_ratio", "context_rule_ratio",
        "fd_like_ratio", "global_rule_ratio", "soccer_rule_total_count", "soccer_domain_rule_ratio",
        "soccer_format_rule_ratio", "soccer_numeric_rule_ratio", "soccer_consistency_rule_ratio",
        "rule_strength_sum", "domain_invalid_flag", "soccer_consistency_flag", "high_soccer_consistency_score",
        "has_soccer_domain_signal", "has_soccer_format_signal", "has_soccer_numeric_signal",
        "has_soccer_consistency_signal", "birthyear_season_gap", "is_young_player_age", "is_old_player_age",
        "is_rare_value", "neighbor_disagree", "is_equal_to_neighbor_majority", "neighbor_agreement_score",
        "is_weak_only", "rare_only_signal", "is_fp_risk_col",
        "pattern_bucket", "rarity_bucket", "neighbor_bucket", "domain_bucket", "soccer_consistency_bucket",
        "soccer_value_bucket", "time_window_bucket", "time_value_bucket", "bucket_id", "cluster_id",
        "dist_to_center", "sample_role", "is_sampled", "signal_priority",
        "error_type", "reason_short", "reason_detailed", "suggested_correct_value", "needs_human_review", "evidence_used",
        "raw_pred_error_prob", "raw_pred_label", "raw_pred_label_name", "stage1_decision_threshold",
        "raw_margin_to_threshold", "stage1_high_conf_error", "stage1_mid_zone",
        "detector_pred_error_prob_before_adjust", "detector_pred_error_prob", "detector_pred_label",
        "detector_pred_label_name", "detector_decision_threshold", "detector_threshold_used", "detector_threshold",
        "detector_margin_to_threshold", "verifier_used", "verifier_rejected", "verifier_keep_error_prob",
        "verifier_decision", "verifier_decision_threshold", "verifier_threshold_used", "group_verifier_threshold",
        "final_pred_error_prob_before_adjust", "final_pred_error_prob", "final_pred_label",
        "final_pred_label_name", "final_decision_threshold", "high_risk", "is_high_risk", "is_hard_case",
        "hard_case_reason", "is_mid_prob", "is_near_threshold", "is_stage_disagree", "rule_neighbor_conflict",
        "soccer_signal_hardcase", "birthyear_season_hardcase", "position_hardcase", "context_fp_hardcase",
        "verifier_borderline", "verifier_reject_borderline", "fp_conflict_fd_hardcase", "sample_borderline_hardcase",
        "recall_recovery_hardcase", "persistent_fp_high_risk",
    ]

    for row_id, col in sorted(keys, key=lambda x: (x[0], x[1])):
        infer_info = infer_row_map.get((row_id, col), {})
        clean_value, dirty_value = get_clean_dirty_value(row_id, col, clean_df, dirty_df)
        row_series = pd.Series(infer_info) if len(infer_info) > 0 else pd.Series(dtype=object)
        out = {
            "detail_type": detail_type,
            "row_id": row_id,
            "column": col,
            "dirty_value": dirty_value,
            "clean_value": clean_value,
            "is_true_error": int((row_id, col) in true_error_info),
            "detector_predicted_error": int(parse_detector_error_flag(row_series)) if len(row_series) > 0 else 0,
            "final_predicted_error": int(parse_final_error_flag(row_series)) if len(row_series) > 0 else 0,
        }
        for c in useful_cols:
            out[c] = infer_info.get(c)
        out["decision_threshold"] = get_threshold_value(infer_info)
        rows.append(out)
    return rows


# ============================================================
# 7. group summary 导出
# ============================================================

def export_group_summaries(infer_df: pd.DataFrame, true_error_cells: Set[Tuple[int, str]],
                           predicted_error_cells: Set[Tuple[int, str]], detector_predicted_error_cells: Set[Tuple[int, str]],
                           candidate_cells: Set[Tuple[int, str]], output_dir: str):
    if not EXPORT_GROUP_SUMMARIES:
        return
    all_cols = sorted(infer_df["column"].dropna().astype(str).unique().tolist())
    calc_candidate_column_metrics(candidate_cells, true_error_cells, all_cols).to_csv(os.path.join(output_dir, "candidate_metrics_by_column.csv"), index=False, encoding="utf-8-sig")
    calc_column_metrics(detector_predicted_error_cells, true_error_cells, all_cols).to_csv(os.path.join(output_dir, "detector_metrics_by_column.csv"), index=False, encoding="utf-8-sig")
    calc_column_metrics(predicted_error_cells, true_error_cells, all_cols).to_csv(os.path.join(output_dir, "final_metrics_by_column.csv"), index=False, encoding="utf-8-sig")

    group_cols = [
        "column", "main_rule_type", "domain_bucket", "soccer_consistency_bucket", "soccer_attribution_bucket",
        "target_is_primary_column", "target_is_context_only_column", "has_birthyear_or_season_signal",
        "position_invalid_or_normalization", "has_team_context_signal", "has_fd_like_signal", "evidence_only_fd_signal",
        "is_context_only_weak_signal", "neighbor_bucket", "rarity_bucket", "pattern_bucket",
        "verifier_decision", "final_pred_label_name", "detector_pred_label_name",
    ]

    for gcol in group_cols:
        if gcol not in infer_df.columns:
            continue
        tmp = infer_df.copy()
        tmp["_key"] = list(zip(tmp["row_id"].astype(int), tmp["column"].astype(str)))
        tmp["_is_true_error"] = tmp["_key"].isin(true_error_cells).astype(int)
        tmp["_final_pred"] = tmp.apply(parse_final_error_flag, axis=1).astype(int)
        tmp["_detector_pred"] = tmp.apply(parse_detector_error_flag, axis=1).astype(int)
        summary = tmp.groupby(gcol, dropna=False).agg(
            rows=("_key", "count"),
            true_errors=("_is_true_error", "sum"),
            final_pred_errors=("_final_pred", "sum"),
            detector_pred_errors=("_detector_pred", "sum"),
        ).reset_index()
        summary["true_error_rate_in_group"] = summary["true_errors"] / summary["rows"].clip(lower=1)
        summary["final_pred_rate_in_group"] = summary["final_pred_errors"] / summary["rows"].clip(lower=1)
        summary["detector_pred_rate_in_group"] = summary["detector_pred_errors"] / summary["rows"].clip(lower=1)
        safe_name = gcol.replace("/", "_")
        summary.to_csv(os.path.join(output_dir, f"group_summary_by_{safe_name}.csv"), index=False, encoding="utf-8-sig")


def export_soccer_v2_diagnostics(infer_df: pd.DataFrame, true_error_cells: Set[Tuple[int, str]],
                                 predicted_error_cells: Set[Tuple[int, str]], detector_predicted_error_cells: Set[Tuple[int, str]],
                                 output_dir: str):
    tmp = infer_df.copy()
    tmp["_key"] = list(zip(tmp["row_id"].astype(int), tmp["column"].astype(str)))
    tmp["_is_true_error"] = tmp["_key"].isin(true_error_cells).astype(int)
    tmp["_final_pred"] = tmp.apply(parse_final_error_flag, axis=1).astype(int)
    tmp["_detector_pred"] = tmp.apply(parse_detector_error_flag, axis=1).astype(int)

    if "soccer_attribution_bucket" in tmp.columns:
        v2_bucket = tmp.groupby("soccer_attribution_bucket", dropna=False).agg(
            rows=("_key", "count"),
            true_errors=("_is_true_error", "sum"),
            detector_pred_errors=("_detector_pred", "sum"),
            final_pred_errors=("_final_pred", "sum"),
        ).reset_index()
        v2_bucket["true_error_rate_in_bucket"] = v2_bucket["true_errors"] / v2_bucket["rows"].clip(lower=1)
        v2_bucket["final_pred_rate_in_bucket"] = v2_bucket["final_pred_errors"] / v2_bucket["rows"].clip(lower=1)
        v2_bucket["precision_in_bucket"] = v2_bucket.apply(
            lambda r: r["true_errors"] / r["final_pred_errors"] if r["final_pred_errors"] > 0 else 0.0,
            axis=1,
        )
        v2_bucket.to_csv(os.path.join(output_dir, "soccer_v2_bucket_diagnostics.csv"), index=False, encoding="utf-8-sig")

        cross = tmp.groupby(["column", "soccer_attribution_bucket"], dropna=False).agg(
            rows=("_key", "count"),
            true_errors=("_is_true_error", "sum"),
            detector_pred_errors=("_detector_pred", "sum"),
            final_pred_errors=("_final_pred", "sum"),
        ).reset_index()
        cross["true_error_rate"] = cross["true_errors"] / cross["rows"].clip(lower=1)
        cross["final_pred_rate"] = cross["final_pred_errors"] / cross["rows"].clip(lower=1)
        cross.to_csv(os.path.join(output_dir, "soccer_v2_column_bucket_diagnostics.csv"), index=False, encoding="utf-8-sig")

    for gcol in [
        "target_is_primary_column", "target_is_context_only_column", "has_birthyear_or_season_signal",
        "position_invalid_or_normalization", "has_team_context_signal", "has_fd_like_signal", "evidence_only_fd_signal",
        "is_context_only_weak_signal", "soccer_primary_target_signal",
    ]:
        if gcol not in tmp.columns:
            continue
        summary = tmp.groupby(gcol, dropna=False).agg(
            rows=("_key", "count"),
            true_errors=("_is_true_error", "sum"),
            detector_pred_errors=("_detector_pred", "sum"),
            final_pred_errors=("_final_pred", "sum"),
        ).reset_index()
        summary["true_error_rate"] = summary["true_errors"] / summary["rows"].clip(lower=1)
        summary["final_pred_rate"] = summary["final_pred_errors"] / summary["rows"].clip(lower=1)
        summary.to_csv(os.path.join(output_dir, f"soccer_v2_summary_by_{gcol}.csv"), index=False, encoding="utf-8-sig")


# ============================================================
# 8. 主评估流程
# ============================================================

def evaluate_detection(clean_csv: str, dirty_csv: str, infer_csv: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    clean_df = load_table(clean_csv)
    dirty_df = load_table(dirty_csv)
    true_error_cells, true_error_info = build_true_error_cells(clean_df, dirty_df)
    infer_df, candidate_cells, predicted_error_cells, detector_predicted_error_cells, infer_row_map = load_infer_result(infer_csv)

    covered_true_errors = true_error_cells & candidate_cells
    candidate_missed_true_errors = true_error_cells - candidate_cells
    candidate_coverage = len(covered_true_errors) / len(true_error_cells) if len(true_error_cells) > 0 else 0.0
    candidate_only_precision = len(covered_true_errors) / len(candidate_cells) if len(candidate_cells) > 0 else 0.0

    detector_tp_keys = detector_predicted_error_cells & true_error_cells
    detector_fp_keys = detector_predicted_error_cells - true_error_cells
    detector_fn_keys = true_error_cells - detector_predicted_error_cells
    detector_prf = calc_prf(detector_predicted_error_cells, true_error_cells)

    tp_keys = predicted_error_cells & true_error_cells
    fp_keys = predicted_error_cells - true_error_cells
    fn_keys = true_error_cells - predicted_error_cells
    prf = calc_prf(predicted_error_cells, true_error_cells)

    n_verifier_used = 0
    n_verifier_rejected = 0
    if "verifier_used" in infer_df.columns:
        n_verifier_used = int(pd.to_numeric(infer_df["verifier_used"], errors="coerce").fillna(0).sum())
    elif "verifier_keep_error_prob" in infer_df.columns:
        n_verifier_used = int(infer_df["verifier_keep_error_prob"].notna().sum())
    if "verifier_rejected" in infer_df.columns:
        n_verifier_rejected = int(pd.to_numeric(infer_df["verifier_rejected"], errors="coerce").fillna(0).sum())
    elif "verifier_decision" in infer_df.columns:
        n_verifier_rejected = int((infer_df["verifier_decision"].astype(str) == "reject_error").sum())

    pd.DataFrame(build_detail_rows(tp_keys, clean_df, dirty_df, infer_row_map, true_error_info, "TP")).to_csv(os.path.join(output_dir, "tp_details.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(build_detail_rows(fp_keys, clean_df, dirty_df, infer_row_map, true_error_info, "FP")).to_csv(os.path.join(output_dir, "fp_details.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(build_detail_rows(fn_keys, clean_df, dirty_df, infer_row_map, true_error_info, "FN")).to_csv(os.path.join(output_dir, "fn_details.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(build_detail_rows(detector_tp_keys, clean_df, dirty_df, infer_row_map, true_error_info, "DETECTOR_TP")).to_csv(os.path.join(output_dir, "detector_tp_details.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(build_detail_rows(detector_fp_keys, clean_df, dirty_df, infer_row_map, true_error_info, "DETECTOR_FP")).to_csv(os.path.join(output_dir, "detector_fp_details.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(build_detail_rows(detector_fn_keys, clean_df, dirty_df, infer_row_map, true_error_info, "DETECTOR_FN")).to_csv(os.path.join(output_dir, "detector_fn_details.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(build_detail_rows(covered_true_errors, clean_df, dirty_df, infer_row_map, true_error_info, "CANDIDATE_COVERED_TRUE_ERROR")).to_csv(os.path.join(output_dir, "candidate_covered_true_errors.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(build_detail_rows(candidate_missed_true_errors, clean_df, dirty_df, infer_row_map, true_error_info, "CANDIDATE_MISSED_TRUE_ERROR")).to_csv(os.path.join(output_dir, "candidate_missed_true_errors.csv"), index=False, encoding="utf-8-sig")

    export_group_summaries(infer_df, true_error_cells, predicted_error_cells, detector_predicted_error_cells, candidate_cells, output_dir)
    export_soccer_v2_diagnostics(infer_df, true_error_cells, predicted_error_cells, detector_predicted_error_cells, output_dir)

    summary = {
        "dataset": "soccer", "clean_csv": clean_csv, "dirty_csv": dirty_csv, "infer_csv": infer_csv,
        "n_rows": len(clean_df), "n_cols": len(clean_df.columns), "n_cells": len(clean_df) * len(clean_df.columns),
        "n_true_errors": len(true_error_cells),
        "n_candidate_cells": len(candidate_cells),
        "n_true_errors_covered_by_candidate": len(covered_true_errors),
        "n_true_errors_missed_by_candidate": len(candidate_missed_true_errors),
        "candidate_coverage": candidate_coverage,
        "candidate_only_precision": candidate_only_precision,
        "n_detector_predicted_errors": len(detector_predicted_error_cells),
        "detector_tp": detector_prf["tp"], "detector_fp": detector_prf["fp"], "detector_fn": detector_prf["fn"],
        "detector_precision": detector_prf["precision"], "detector_recall": detector_prf["recall"], "detector_f1": detector_prf["f1"],
        "n_predicted_errors": len(predicted_error_cells),
        "tp": prf["tp"], "fp": prf["fp"], "fn": prf["fn"],
        "precision": prf["precision"], "recall": prf["recall"], "f1": prf["f1"],
        "n_verifier_used": n_verifier_used,
        "n_verifier_rejected": n_verifier_rejected,
        "n_primary_target_candidates": int(safe_num_series(infer_df, "target_is_primary_column", 0).sum()),
        "n_context_only_candidates": int(safe_num_series(infer_df, "target_is_context_only_column", 0).sum()),
        "n_birthyear_or_season_signal": int(safe_num_series(infer_df, "has_birthyear_or_season_signal", 0).sum()),
        "n_position_signal": int(safe_num_series(infer_df, "position_invalid_or_normalization", 0).sum()),
        "n_team_context_signal": int(safe_num_series(infer_df, "has_team_context_signal", 0).sum()),
        "n_fd_like_signal": int(safe_num_series(infer_df, "has_fd_like_signal", 0).sum()),
        "n_evidence_only_fd_signal": int(safe_num_series(infer_df, "evidence_only_fd_signal", 0).sum()),
        "n_context_only_weak_signal": int(safe_num_series(infer_df, "is_context_only_weak_signal", 0).sum()),
        "n_soccer_primary_target_signal": int(safe_num_series(infer_df, "soccer_primary_target_signal", 0).sum()),
    }

    if "high_risk" in infer_df.columns:
        summary["n_high_risk"] = int(pd.to_numeric(infer_df["high_risk"], errors="coerce").fillna(0).sum())
    elif "is_high_risk" in infer_df.columns:
        summary["n_high_risk"] = int(pd.to_numeric(infer_df["is_high_risk"], errors="coerce").fillna(0).sum())
    if "is_hard_case" in infer_df.columns:
        summary["n_hard_case"] = int(pd.to_numeric(infer_df["is_hard_case"], errors="coerce").fillna(0).sum())

    pd.DataFrame([summary]).to_csv(os.path.join(output_dir, "evaluation_summary.csv"), index=False, encoding="utf-8-sig")

    print("========== Soccer 评估结果 ==========")
    print(f"总行数: {summary['n_rows']}")
    print(f"总列数: {summary['n_cols']}")
    print(f"总单元格数: {summary['n_cells']}")
    print()
    print("---- 真实错误统计 ----")
    print(f"真实错误单元格数: {summary['n_true_errors']}")
    print()
    print("---- 候选范围统计 ----")
    print(f"候选范围单元格数: {summary['n_candidate_cells']}")
    print(f"进入候选范围的真实错误数: {summary['n_true_errors_covered_by_candidate']}")
    print(f"未进入候选范围的真实错误数: {summary['n_true_errors_missed_by_candidate']}")
    print(f"Candidate Coverage: {summary['candidate_coverage']:.6f}")
    print(f"Candidate-Only Precision: {summary['candidate_only_precision']:.6f}")
    print()
    print("---- Detector-Only 指标 ----")
    print(f"Detector 预测为错误的单元格数: {summary['n_detector_predicted_errors']}")
    print(f"Detector TP: {summary['detector_tp']}")
    print(f"Detector FP: {summary['detector_fp']}")
    print(f"Detector FN: {summary['detector_fn']}")
    print(f"Detector Precision: {summary['detector_precision']:.6f}")
    print(f"Detector Recall: {summary['detector_recall']:.6f}")
    print(f"Detector F1: {summary['detector_f1']:.6f}")
    print()
    print("---- 最终错误检测指标（Verifier后）----")
    print(f"最终预测为错误的单元格数: {summary['n_predicted_errors']}")
    print(f"TP: {summary['tp']}")
    print(f"FP: {summary['fp']}")
    print(f"FN: {summary['fn']}")
    print(f"Precision: {summary['precision']:.6f}")
    print(f"Recall: {summary['recall']:.6f}")
    print(f"F1: {summary['f1']:.6f}")
    print()
    print("---- Soccer v2 诊断统计 ----")
    print(f"Primary target 候选数: {summary.get('n_primary_target_candidates', 0)}")
    print(f"Context-only 候选数: {summary.get('n_context_only_candidates', 0)}")
    print(f"Birthyear/Season signal 数量: {summary.get('n_birthyear_or_season_signal', 0)}")
    print(f"Position signal 数量: {summary.get('n_position_signal', 0)}")
    print(f"Team-context signal 数量: {summary.get('n_team_context_signal', 0)}")
    print(f"FD-like signal 数量: {summary.get('n_fd_like_signal', 0)}")
    print(f"Evidence-only soft-FD signal 数量: {summary.get('n_evidence_only_fd_signal', 0)}")
    print(f"Soccer primary target signal 数量: {summary.get('n_soccer_primary_target_signal', 0)}")
    print(f"Context-only weak signal 数量: {summary.get('n_context_only_weak_signal', 0)}")
    print()
    print("---- Verifier 统计 ----")
    print(f"Verifier 参与样本数: {summary['n_verifier_used']}")
    print(f"Verifier 打回 correct 数: {summary['n_verifier_rejected']}")
    if "n_high_risk" in summary:
        print(f"High Risk 数量: {summary['n_high_risk']}")
    if "n_hard_case" in summary:
        print(f"Hard Case 数量: {summary['n_hard_case']}")
    print()
    print("---- 明细文件输出 ----")
    for name in [
        "evaluation_summary.csv", "tp_details.csv", "fp_details.csv", "fn_details.csv",
        "detector_tp_details.csv", "detector_fp_details.csv", "detector_fn_details.csv",
        "candidate_covered_true_errors.csv", "candidate_missed_true_errors.csv",
        "candidate_metrics_by_column.csv", "detector_metrics_by_column.csv", "final_metrics_by_column.csv",
        "soccer_v2_bucket_diagnostics.csv", "soccer_v2_column_bucket_diagnostics.csv",
    ]:
        print(os.path.join(output_dir, name))
    return summary


if __name__ == "__main__":
    evaluate_detection(CLEAN_CSV, DIRTY_CSV, INFER_RESULT_CSV, OUTPUT_DIR)

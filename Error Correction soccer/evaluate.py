#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Soccer full-table end-to-end repair evaluation.

This script is migrated from the Adult full-table evaluator and customized for Soccer.
It evaluates repair results over the whole table, not only over touched/predicted cells.

Default usage:
cd /mnt/mydata/dq/projects/Splittree/test/Error\ Correction\ soccer
python evaluate.py

Or explicitly:
python evaluate.py \
  --clean_file /mnt/mydata/dq/projects/Splittree/soccer/soccer_clean.csv \
  --dirty_file /mnt/mydata/dq/projects/Splittree/soccer/soccer_dirty.csv \
  --repair_result_file /mnt/mydata/dq/projects/Splittree/test/Error\ Correction\ soccer/repair_iterative_workdir_soccer/model_output/inference_top1_repairs.csv \
  --output_dir repair_eval_output_soccer
"""

import argparse
import os
import re
from typing import Any, Optional, Tuple

import pandas as pd


# ============================================================
# 1. Default paths
# ============================================================

DEFAULT_CLEAN_FILE = "/mnt/mydata/dq/projects/Splittree/soccer/soccer_clean.csv"
DEFAULT_DIRTY_FILE = "/mnt/mydata/dq/projects/Splittree/soccer/soccer_dirty.csv"

# 按你的要求，默认直接使用这个输入文件路径
DEFAULT_REPAIR_RESULT_FILE = "/mnt/mydata/dq/projects/Splittree/test/Error Correction soccer/repair_iterative_workdir_soccer/model_output/inference_top1_repairs.csv"

DEFAULT_OUTPUT_DIR = (
    "/mnt/mydata/dq/projects/Splittree/test/Error Correction soccer/repair_eval_output_soccer"
)

EMPTY_TOKENS = {
    "", "empty", "null", "none", "nan", "na", "n/a", "missing", "?", "{null}",
    "not_available", "not available", "unknown", "unk", "-", "--",
}

SOCCER_POSITION_VALUES = [
    "Goalkeeper", "Defender", "Midfield", "Forward",
]

# 常见同义 / 噪声写法归一到你的 Soccer 表中最可能使用的四类位置。
POSITION_ALIASES = {
    "goal keeper": "Goalkeeper",
    "goal-keeper": "Goalkeeper",
    "goal_keeper": "Goalkeeper",
    "goalkeeper": "Goalkeeper",
    "keeper": "Goalkeeper",
    "gk": "Goalkeeper",
    "defender": "Defender",
    "defence": "Defender",
    "defense": "Defender",
    "df": "Defender",
    "back": "Defender",
    "midfield": "Midfield",
    "midfielder": "Midfield",
    "middle": "Midfield",
    "mf": "Midfield",
    "forward": "Forward",
    "striker": "Forward",
    "attacker": "Forward",
    "winger": "Forward",
    "fw": "Forward",
    "cf": "Forward",
}

SOCCER_TEXT_COLUMNS = {
    "name", "surname", "birthplace", "team", "city", "stadium", "manager",
}

SOCCER_NUMERIC_COLUMNS = {
    "birthyear", "season",
}

SOCCER_DOMAIN_COLUMNS = {
    "position",
}

SOCCER_EXPECTED_COLUMNS = [
    "name", "surname", "birthyear", "birthplace", "position",
    "team", "city", "stadium", "season", "manager",
]


# ============================================================
# 2. Arguments and IO helpers
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Soccer full-table repair evaluation")
    p.add_argument("--clean_file", default=DEFAULT_CLEAN_FILE)
    p.add_argument("--dirty_file", default=DEFAULT_DIRTY_FILE)
    p.add_argument("--repair_result_file", default=DEFAULT_REPAIR_RESULT_FILE)
    p.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--case_insensitive_text", type=int, default=1)
    p.add_argument("--strict_mode", type=int, default=0)
    p.add_argument("--soccer_normalized_comparison", type=int, default=1)
    return p.parse_args()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def resolve_repair_result_file(path: str) -> str:
    """
    Use one explicit repair-result file path.
    No candidate list is used in this Soccer version.
    """
    path = str(path or "").strip()
    if not path:
        path = DEFAULT_REPAIR_RESULT_FILE

    if not os.path.exists(path):
        raise FileNotFoundError(f"修复结果文件不存在: {path}")

    if os.path.isdir(path):
        raise IsADirectoryError(
            f"repair_result_file 应该是具体 CSV 文件路径，不应该是目录: {path}"
        )

    return path


def read_table(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"表文件不存在: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def validate_same_shape(clean_df: pd.DataFrame, dirty_df: pd.DataFrame):
    if clean_df.shape != dirty_df.shape:
        raise ValueError(f"clean 与 dirty 形状不一致: clean={clean_df.shape}, dirty={dirty_df.shape}")

    if list(clean_df.columns) != list(dirty_df.columns):
        raise ValueError(
            "clean 与 dirty 列名不一致，请先对齐列顺序和列名。\n"
            f"clean columns={list(clean_df.columns)}\n"
            f"dirty columns={list(dirty_df.columns)}"
        )


def norm_raw(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()


def normalize_col_name(column: Any) -> str:
    c = norm_raw(column).lower().strip()
    aliases = {
        "birth_year": "birthyear",
        "birth year": "birthyear",
        "birth-year": "birthyear",
        "year_of_birth": "birthyear",
        "birthplace": "birthplace",
        "birth_place": "birthplace",
        "birth place": "birthplace",
        "pos": "position",
        "club": "team",
        "stadium_name": "stadium",
        "season_year": "season",
        "coach": "manager",
    }
    return aliases.get(c, c)


def normalize_empty(x: Any) -> str:
    s = norm_raw(x)
    if s.lower() in EMPTY_TOKENS:
        return ""
    return s


# ============================================================
# 3. Soccer-aware normalization
# ============================================================

def clean_text_token(s: str) -> str:
    """Normalize separators but keep enough signal for exact table comparison."""
    s = normalize_empty(s)
    if not s:
        return ""
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def text_key(s: Any, case_insensitive: bool = True) -> str:
    s = clean_text_token(norm_raw(s))
    if not s:
        return ""
    # Soccer 数据里很多名称使用下划线，比较时把连续空格和下划线统一。
    s = re.sub(r"[\s_]+", "_", s.strip())
    return s.lower() if case_insensitive else s


def fix_year_ocr_like(s: str) -> str:
    """Repair common OCR-like digits in year strings, e.g., 2O11 -> 2011, 199l -> 1991."""
    trans = str.maketrans({
        "O": "0", "o": "0", "Q": "0",
        "I": "1", "l": "1", "L": "1", "|": "1",
        "S": "5", "s": "5",
        "B": "8",
        "Z": "2", "z": "2",
    })
    return s.translate(trans)


def parse_year_like(x: Any) -> Optional[int]:
    s = normalize_empty(x)
    if not s:
        return None
    s = fix_year_ocr_like(s)
    # 允许 2012.0 这种 pandas 或模型输出格式。
    m_float = re.fullmatch(r"([12]\d{3})\.0+", s)
    if m_float:
        return int(m_float.group(1))
    m = re.search(r"(?<!\d)([12]\d{3})(?!\d)", s)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def canonical_year(column: Any, value: Any) -> str:
    col = normalize_col_name(column)
    y = parse_year_like(value)
    if y is None:
        return normalize_empty(value)
    if col == "birthyear":
        # Soccer 数据示例中 birthyear 大多在 1970-1995 附近，这里放宽避免误伤。
        if 1900 <= y <= 2025:
            return str(y)
    if col == "season":
        if 1900 <= y <= 2035:
            return str(y)
    return str(y)


def canonical_position(x: Any) -> str:
    s = normalize_empty(x)
    if not s:
        return ""
    key = re.sub(r"[\s_\-]+", " ", s.strip().lower())
    key_compact = key.replace(" ", "")
    if key in POSITION_ALIASES:
        return POSITION_ALIASES[key]
    if key_compact in POSITION_ALIASES:
        return POSITION_ALIASES[key_compact]
    for v in SOCCER_POSITION_VALUES:
        if v.lower() == key or v.lower() == key_compact:
            return v
    return s


def normalize_value_for_compare(
    column: Any,
    value: Any,
    case_insensitive_text: bool = True,
    soccer_normalized_comparison: bool = True,
) -> str:
    col = normalize_col_name(column)
    s = normalize_empty(value)
    if not s:
        return ""

    if not soccer_normalized_comparison:
        s = re.sub(r"\s+", " ", s.strip())
        return s.lower() if case_insensitive_text else s

    if col in {"birthyear", "season"}:
        return canonical_year(col, s)
    if col == "position":
        v = canonical_position(s)
        return v.lower() if case_insensitive_text else v
    if col in SOCCER_TEXT_COLUMNS:
        return text_key(s, case_insensitive=case_insensitive_text)

    s = re.sub(r"\s+", " ", s.strip())
    return s.lower() if case_insensitive_text else s


def stringify_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        out[col] = out[col].map(norm_raw).astype(object)
    return out


# ============================================================
# 4. Repair result loading and application
# ============================================================

def find_repair_value_column(df: pd.DataFrame) -> str:
    candidates = [
        "repaired_value",
        "predicted_repair_value",
        "predicted_repair",
        "repair_value",
        "predicted_value",
        "student_best_candidate",
        "student_top_candidate",
        "top_candidate_value",
        "candidate_value",
        "teacher_best_candidate",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        "修复结果文件中未找到修复值字段。支持字段包括："
        "repaired_value / predicted_repair_value / predicted_repair / repair_value / predicted_value / "
        "student_best_candidate / student_top_candidate / top_candidate_value / candidate_value / teacher_best_candidate"
    )


def load_repair_results(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"修复结果文件不存在: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")
    if "row_id" not in df.columns or "column" not in df.columns:
        raise ValueError("修复结果文件必须包含 row_id 和 column")

    df["row_id"] = pd.to_numeric(df["row_id"], errors="raise").astype(int)
    df["column"] = df["column"].astype(str).str.strip()
    repair_col = find_repair_value_column(df)

    keep_cols = ["row_id", "column", repair_col]
    optional_cols = [
        "dirty_value", "top_candidate_value", "student_top_candidate", "student_best_candidate",
        "student_score", "student_margin", "evidence_strength", "gate_pass", "gate_passed",
        "gate_reason", "student_reason_pred", "candidate_count", "top_candidate_sources",
        "candidate_rule_support_ratio", "soccer_gain", "soccer_profile_support",
        "soccer_consistency_gain", "candidate_domain_validity_gain", "candidate_in_soccer_domain",
        "position_consistency_gain", "birthyear_season_age_gain", "team_context_gain",
        "row_team", "row_city", "row_stadium", "row_manager",
    ]
    for c in optional_cols:
        if c in df.columns and c not in keep_cols:
            keep_cols.append(c)

    out = df[keep_cols].copy()
    out = out.rename(columns={repair_col: "predicted_repair"})
    out["predicted_repair"] = out["predicted_repair"].map(norm_raw)
    out = out.drop_duplicates(subset=["row_id", "column"], keep="first").reset_index(drop=True)
    return out


def apply_repairs_to_dirty(dirty_df: pd.DataFrame, repair_df: pd.DataFrame) -> pd.DataFrame:
    repaired_df = stringify_table(dirty_df)
    n_rows = len(repaired_df)
    valid_columns = set(repaired_df.columns)
    skipped_bad_row = 0
    skipped_bad_col = 0

    for _, r in repair_df.iterrows():
        row_id = int(r["row_id"])
        column = str(r["column"]).strip()
        pred_val = norm_raw(r["predicted_repair"])

        if row_id < 0 or row_id >= n_rows:
            skipped_bad_row += 1
            continue
        if column not in valid_columns:
            skipped_bad_col += 1
            continue
        repaired_df.at[row_id, column] = pred_val

    if skipped_bad_row > 0:
        print(f"[WARN] 跳过越界 row_id 修复记录数: {skipped_bad_row}")
    if skipped_bad_col > 0:
        print(f"[WARN] 跳过不存在 column 修复记录数: {skipped_bad_col}")

    return repaired_df


# ============================================================
# 5. Full-table evaluation
# ============================================================

def build_full_table_eval(
    clean_df: pd.DataFrame,
    dirty_df: pd.DataFrame,
    repaired_df: pd.DataFrame,
    repair_df: pd.DataFrame,
    case_insensitive_text: bool,
    strict_mode: bool,
    soccer_normalized_comparison: bool,
) -> pd.DataFrame:
    repair_pos_set = set((int(r["row_id"]), str(r["column"]).strip()) for _, r in repair_df.iterrows())
    repair_info_map = {}
    for _, r in repair_df.iterrows():
        repair_info_map[(int(r["row_id"]), str(r["column"]).strip())] = r.to_dict()

    rows = []
    for row_id in range(len(clean_df)):
        for column in clean_df.columns:
            column = str(column)
            clean_val = norm_raw(clean_df.at[row_id, column])
            dirty_val = norm_raw(dirty_df.at[row_id, column])
            repaired_val = norm_raw(repaired_df.at[row_id, column])

            dirty_norm = normalize_value_for_compare(column, dirty_val, case_insensitive_text, soccer_normalized_comparison)
            clean_norm = normalize_value_for_compare(column, clean_val, case_insensitive_text, soccer_normalized_comparison)
            repaired_norm = normalize_value_for_compare(column, repaired_val, case_insensitive_text, soccer_normalized_comparison)

            is_true_error = int(dirty_norm != clean_norm)
            is_touched_by_result = int((row_id, column) in repair_pos_set)
            is_changed = int(repaired_norm != dirty_norm)
            is_touched = int(is_touched_by_result or is_changed)
            is_final_correct = int(repaired_norm == clean_norm)

            if is_true_error == 1 and is_final_correct == 1:
                eval_type = "TP"
            elif is_true_error == 0 and is_final_correct == 0:
                eval_type = "FP"
            elif is_true_error == 1 and is_final_correct == 0:
                eval_type = "FN"
            else:
                eval_type = "TN"

            info = repair_info_map.get((row_id, column), {})
            row = {
                "row_id": row_id,
                "column": column,
                "dirty_value": dirty_val,
                "clean_value": clean_val,
                "repaired_value": repaired_val,
                "dirty_norm": dirty_norm,
                "clean_norm": clean_norm,
                "repaired_norm": repaired_norm,
                "is_true_error": is_true_error,
                "is_touched_by_result": is_touched_by_result,
                "is_changed": is_changed,
                "is_touched": is_touched,
                "is_final_correct": is_final_correct,
                "eval_type": eval_type,
            }

            if strict_mode:
                strict_dirty = normalize_empty(dirty_val)
                strict_clean = normalize_empty(clean_val)
                strict_repaired = normalize_empty(repaired_val)
                row["strict_is_true_error"] = int(strict_dirty != strict_clean)
                row["strict_is_final_correct"] = int(strict_repaired == strict_clean)
                row["strict_eval_type"] = (
                    "TP" if row["strict_is_true_error"] == 1 and row["strict_is_final_correct"] == 1
                    else "FP" if row["strict_is_true_error"] == 0 and strict_repaired != strict_clean
                    else "FN" if row["strict_is_true_error"] == 1 and row["strict_is_final_correct"] == 0
                    else "TN"
                )

            for extra_col in [
                "predicted_repair", "dirty_value", "top_candidate_value", "student_top_candidate",
                "student_best_candidate", "student_score", "student_margin", "evidence_strength",
                "gate_pass", "gate_passed", "gate_reason", "student_reason_pred", "candidate_count",
                "top_candidate_sources", "candidate_rule_support_ratio", "soccer_gain",
                "soccer_profile_support", "soccer_consistency_gain", "candidate_domain_validity_gain",
                "candidate_in_soccer_domain", "position_consistency_gain", "birthyear_season_age_gain",
                "team_context_gain", "row_team", "row_city", "row_stadium", "row_manager",
            ]:
                if extra_col in info:
                    out_col = f"repair_{extra_col}" if extra_col == "dirty_value" else extra_col
                    row[out_col] = info.get(extra_col)

            # Soccer 专属诊断字段
            col_norm = normalize_col_name(column)
            row["soccer_normalized_column"] = col_norm
            row["is_soccer_position_column"] = int(col_norm == "position")
            row["is_soccer_year_column"] = int(col_norm in {"birthyear", "season"})
            row["is_soccer_text_column"] = int(col_norm in SOCCER_TEXT_COLUMNS)
            row["case_or_separator_only_difference"] = int(
                normalize_empty(dirty_val) != normalize_empty(clean_val)
                and dirty_norm == clean_norm
            )
            rows.append(row)

    return pd.DataFrame(rows)


def compute_metrics(eval_df: pd.DataFrame) -> dict:
    tp = int((eval_df["eval_type"] == "TP").sum())
    fp = int((eval_df["eval_type"] == "FP").sum())
    fn = int((eval_df["eval_type"] == "FN").sum())
    tn = int((eval_df["eval_type"] == "TN").sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)

    return {
        "total_cells": len(eval_df),
        "true_error_cells": int(eval_df["is_true_error"].sum()),
        "touched_cells": int(eval_df["is_touched"].sum()),
        "touched_by_result_rows": int(eval_df["is_touched_by_result"].sum()),
        "changed_cells": int(eval_df["is_changed"].sum()),
        "remaining_wrong_cells_after_repair": int((eval_df["repaired_norm"] != eval_df["clean_norm"]).sum()),
        "unchanged_true_errors": int(((eval_df["is_true_error"] == 1) & (eval_df["repaired_norm"] == eval_df["dirty_norm"])).sum()),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Accuracy": accuracy,
    }


def build_column_summary(eval_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column, g in eval_df.groupby("column", sort=False):
        tp = int((g["eval_type"] == "TP").sum())
        fp = int((g["eval_type"] == "FP").sum())
        fn = int((g["eval_type"] == "FN").sum())
        tn = int((g["eval_type"] == "TN").sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)

        rows.append({
            "column": column,
            "normalized_column": normalize_col_name(column),
            "total_cells": len(g),
            "true_error_cells": int(g["is_true_error"].sum()),
            "touched_cells": int(g["is_touched"].sum()),
            "touched_by_result_rows": int(g["is_touched_by_result"].sum()),
            "changed_cells": int(g["is_changed"].sum()),
            "remaining_wrong_cells_after_repair": int((g["repaired_norm"] != g["clean_norm"]).sum()),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Accuracy": accuracy,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["true_error_cells", "F1"], ascending=[False, False]).reset_index(drop=True)
    return out


def build_gate_reason_summary(eval_df: pd.DataFrame) -> pd.DataFrame:
    if "gate_reason" not in eval_df.columns:
        return pd.DataFrame()

    touched = eval_df[eval_df["is_touched_by_result"] == 1].copy()
    if touched.empty:
        return pd.DataFrame()

    rows = []
    for (column, gate_reason), g in touched.groupby(["column", "gate_reason"], dropna=False):
        tp = int((g["eval_type"] == "TP").sum())
        fp = int((g["eval_type"] == "FP").sum())
        fn = int((g["eval_type"] == "FN").sum())
        tn = int((g["eval_type"] == "TN").sum())
        rows.append({
            "column": column,
            "gate_reason": gate_reason,
            "count": len(g),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "precision_within_reason": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        })
    return pd.DataFrame(rows).sort_values(["count", "TP"], ascending=[False, False]).reset_index(drop=True)


def build_soccer_column_type_summary(eval_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column, g in eval_df.groupby("column", sort=False):
        col_norm = normalize_col_name(column)
        tp = int((g["eval_type"] == "TP").sum())
        fp = int((g["eval_type"] == "FP").sum())
        fn = int((g["eval_type"] == "FN").sum())
        rows.append({
            "column": column,
            "normalized_column": col_norm,
            "is_position_column": int(col_norm == "position"),
            "is_year_column": int(col_norm in {"birthyear", "season"}),
            "is_text_column": int(col_norm in SOCCER_TEXT_COLUMNS),
            "true_error_cells": int(g["is_true_error"].sum()),
            "touched_cells": int(g["is_touched"].sum()),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "case_or_separator_only_differences": int(g["case_or_separator_only_difference"].sum()),
        })
    return pd.DataFrame(rows).sort_values(["true_error_cells", "TP"], ascending=[False, False]).reset_index(drop=True)


def build_soccer_age_consistency_summary(eval_df: pd.DataFrame) -> pd.DataFrame:
    """Extra diagnostic summary for birthyear/season consistency after repair."""
    needed = {"birthyear", "season"}
    cols_norm = {normalize_col_name(c): c for c in eval_df["column"].unique()}
    if not needed.issubset(set(cols_norm.keys())):
        return pd.DataFrame()

    by_col = cols_norm["birthyear"]
    season_col = cols_norm["season"]
    by = eval_df[eval_df["column"] == by_col][["row_id", "repaired_norm", "clean_norm", "dirty_norm"]].rename(
        columns={"repaired_norm": "birthyear_repaired", "clean_norm": "birthyear_clean", "dirty_norm": "birthyear_dirty"}
    )
    se = eval_df[eval_df["column"] == season_col][["row_id", "repaired_norm", "clean_norm", "dirty_norm"]].rename(
        columns={"repaired_norm": "season_repaired", "clean_norm": "season_clean", "dirty_norm": "season_dirty"}
    )
    merged = by.merge(se, on="row_id", how="inner")

    def to_int_or_none(x):
        try:
            if str(x).strip() == "":
                return None
            return int(float(str(x)))
        except Exception:
            return None

    rows = []
    for _, r in merged.iterrows():
        br = to_int_or_none(r["birthyear_repaired"])
        sr = to_int_or_none(r["season_repaired"])
        age = None if br is None or sr is None else sr - br
        rows.append({
            "row_id": int(r["row_id"]),
            "birthyear_dirty": r["birthyear_dirty"],
            "birthyear_clean": r["birthyear_clean"],
            "birthyear_repaired": r["birthyear_repaired"],
            "season_dirty": r["season_dirty"],
            "season_clean": r["season_clean"],
            "season_repaired": r["season_repaired"],
            "age_at_season_repaired": age,
            "age_valid_15_50": int(age is not None and 15 <= age <= 50),
        })
    return pd.DataFrame(rows)


# ============================================================
# 6. Main
# ============================================================

def main():
    args = parse_args()
    repair_result_file = resolve_repair_result_file(args.repair_result_file)
    output_dir = args.output_dir
    ensure_dir(output_dir)

    output_summary_csv = os.path.join(output_dir, "full_table_eval_summary.csv")
    output_cell_details_csv = os.path.join(output_dir, "full_table_eval_all_cells.csv")
    output_tp_csv = os.path.join(output_dir, "full_table_tp_details.csv")
    output_fp_csv = os.path.join(output_dir, "full_table_fp_details.csv")
    output_fn_csv = os.path.join(output_dir, "full_table_fn_details.csv")
    output_tn_csv = os.path.join(output_dir, "full_table_tn_details.csv")
    output_repaired_table_csv = os.path.join(output_dir, "repaired_table.csv")
    output_touched_csv = os.path.join(output_dir, "touched_cells_details.csv")
    output_column_summary_csv = os.path.join(output_dir, "column_level_eval_summary.csv")
    output_gate_summary_csv = os.path.join(output_dir, "gate_reason_eval_summary.csv")
    output_soccer_summary_csv = os.path.join(output_dir, "soccer_column_type_summary.csv")
    output_soccer_age_csv = os.path.join(output_dir, "soccer_age_consistency_summary.csv")

    print("[INFO] Loading clean / dirty tables...")
    clean_df_raw = read_table(args.clean_file)
    dirty_df_raw = read_table(args.dirty_file)
    validate_same_shape(clean_df_raw, dirty_df_raw)

    clean_df = stringify_table(clean_df_raw)
    dirty_df = stringify_table(dirty_df_raw)

    print("[INFO] Loading repair results...")
    repair_df = load_repair_results(repair_result_file)

    print("[INFO] Applying repairs...")
    repaired_df = apply_repairs_to_dirty(dirty_df, repair_df)
    repaired_df.to_csv(output_repaired_table_csv, index=False, encoding="utf-8-sig")

    print("[INFO] Building Soccer full-table evaluation...")
    eval_df = build_full_table_eval(
        clean_df=clean_df,
        dirty_df=dirty_df,
        repaired_df=repaired_df,
        repair_df=repair_df,
        case_insensitive_text=bool(args.case_insensitive_text),
        strict_mode=bool(args.strict_mode),
        soccer_normalized_comparison=bool(args.soccer_normalized_comparison),
    )

    metrics = compute_metrics(eval_df)
    metrics["use_soccer_normalized_comparison"] = int(bool(args.soccer_normalized_comparison))
    metrics["case_insensitive_text"] = int(bool(args.case_insensitive_text))
    metrics["repair_result_file"] = repair_result_file

    column_summary_df = build_column_summary(eval_df)
    gate_summary_df = build_gate_reason_summary(eval_df)
    soccer_summary_df = build_soccer_column_type_summary(eval_df)
    soccer_age_df = build_soccer_age_consistency_summary(eval_df)

    eval_df.to_csv(output_cell_details_csv, index=False, encoding="utf-8-sig")
    eval_df[eval_df["eval_type"] == "TP"].to_csv(output_tp_csv, index=False, encoding="utf-8-sig")
    eval_df[eval_df["eval_type"] == "FP"].to_csv(output_fp_csv, index=False, encoding="utf-8-sig")
    eval_df[eval_df["eval_type"] == "FN"].to_csv(output_fn_csv, index=False, encoding="utf-8-sig")
    eval_df[eval_df["eval_type"] == "TN"].to_csv(output_tn_csv, index=False, encoding="utf-8-sig")
    eval_df[eval_df["is_touched"] == 1].to_csv(output_touched_csv, index=False, encoding="utf-8-sig")

    pd.DataFrame([metrics]).to_csv(output_summary_csv, index=False, encoding="utf-8-sig")
    column_summary_df.to_csv(output_column_summary_csv, index=False, encoding="utf-8-sig")
    gate_summary_df.to_csv(output_gate_summary_csv, index=False, encoding="utf-8-sig")
    soccer_summary_df.to_csv(output_soccer_summary_csv, index=False, encoding="utf-8-sig")
    soccer_age_df.to_csv(output_soccer_age_csv, index=False, encoding="utf-8-sig")

    print("\n========== Soccer 全表端到端修复评估结果 ==========")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")

    print("\n---- 各列评估摘要 Top ----")
    if not column_summary_df.empty:
        show_cols = ["column", "true_error_cells", "touched_cells", "TP", "FP", "FN", "Precision", "Recall", "F1"]
        print(column_summary_df[show_cols].head(30).to_string(index=False))

    if not gate_summary_df.empty:
        print("\n---- Gate Reason 摘要 Top ----")
        show_cols = ["column", "gate_reason", "count", "TP", "FP", "FN", "TN", "precision_within_reason"]
        print(gate_summary_df[show_cols].head(30).to_string(index=False))

    if not soccer_summary_df.empty:
        print("\n---- Soccer Column Type 摘要 ----")
        show_cols = [
            "column", "normalized_column", "is_position_column", "is_year_column", "is_text_column",
            "true_error_cells", "touched_cells", "TP", "FP", "FN", "case_or_separator_only_differences",
        ]
        print(soccer_summary_df[show_cols].head(30).to_string(index=False))

    if not soccer_age_df.empty:
        invalid_age = int((soccer_age_df["age_valid_15_50"] == 0).sum())
        print(f"\n---- Soccer 年龄一致性摘要 ----")
        print(f"age_at_season 不在 [15, 50] 或无法解析的行数: {invalid_age}")
        print(f"年龄一致性明细已保存: {output_soccer_age_csv}")

    print(f"\n[OK] 评估摘要已保存: {output_summary_csv}")
    print(f"[OK] 列级评估摘要已保存: {output_column_summary_csv}")
    print(f"[OK] gate reason 摘要已保存: {output_gate_summary_csv}")
    print(f"[OK] Soccer 列类型摘要已保存: {output_soccer_summary_csv}")
    print(f"[OK] Soccer 年龄一致性摘要已保存: {output_soccer_age_csv}")
    print(f"[OK] 修复后的整表已保存: {output_repaired_table_csv}")
    print(f"[OK] 全量单元格评估明细已保存: {output_cell_details_csv}")
    print(f"[OK] touched 明细已保存: {output_touched_csv}")
    print(f"[OK] TP 明细已保存: {output_tp_csv}")
    print(f"[OK] FP 明细已保存: {output_fp_csv}")
    print(f"[OK] FN 明细已保存: {output_fn_csv}")
    print(f"[OK] TN 明细已保存: {output_tn_csv}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adult full-table end-to-end repair evaluation.

Adapted from the Flights full-table evaluator, but customized for Adult:
1. Default clean/dirty paths use Adult dataset.
2. Compatible with inference_top1_repairs.csv from infer_ltr_adult.py:
   repaired_value / predicted_repair_value / predicted_repair / repair_value / candidate_value...
3. Adult-aware normalized comparison:
   - empty/null/?/missing -> empty
   - income normalization: <=50K / >50K / LessThan50K / MoreThan50K
   - Adult categorical values compared case-insensitively with domain canonicalization
   - age/hoursperweek and other numeric-like columns compared as normalized numbers/ranges
4. Outputs full-table TP/FP/FN/TN details and Adult-specific gate/canonical summaries.

Usage
-----
cd /mnt/mydata/dq/projects/Splittree/test/Error\ Correction\ adult

python evaluate.py \
  --clean_file /mnt/mydata/dq/projects/Splittree/adult/adult_clean.csv \
  --dirty_file /mnt/mydata/dq/projects/Splittree/adult/adult_dirty.csv \
  --repair_result_file repair_iterative_workdir_adult/model_output/inference_top1_repairs.csv \
  --output_dir repair_eval_output_adult

If your inference_top1_repairs.csv is in the current folder:
python evaluate.py --repair_result_file inference_top1_repairs.csv
"""

import argparse
import os
import re
from typing import Any, Optional, Tuple

import pandas as pd


DEFAULT_CLEAN_FILE = "/mnt/mydata/dq/projects/Splittree/adult/adult_clean.csv"
DEFAULT_DIRTY_FILE = "/mnt/mydata/dq/projects/Splittree/adult/adult_dirty.csv"

DEFAULT_REPAIR_RESULT_CANDIDATES = [
    "repair_iterative_workdir_adult/model_output/inference_top1_repairs.csv",
    "pairwise_ltr_student_output_adult/inference_top1_repairs.csv",
    "model_output/inference_top1_repairs.csv",
    "inference_top1_repairs.csv",
]

DEFAULT_OUTPUT_DIR = "/mnt/mydata/dq/projects/Splittree/test/Error Correction adult/repair_eval_output_adult"

EMPTY_TOKENS = {"", "empty", "null", "none", "nan", "na", "n/a", "missing", "?", "{null}"}

ADULT_DOMAIN_VALUES = {
    "workclass": [
        "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
        "Local-gov", "State-gov", "Without-pay", "Never-worked",
    ],
    "education": [
        "Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th",
        "11th", "12th", "HS-grad", "Some-college", "Assoc-voc",
        "Assoc-acdm", "Bachelors", "Masters", "Prof-school", "Doctorate",
    ],
    "maritalstatus": [
        "Married-civ-spouse", "Divorced", "Never-married", "Separated",
        "Widowed", "Married-spouse-absent", "Married-AF-spouse",
    ],
    "occupation": [
        "Tech-support", "Craft-repair", "Other-service", "Sales",
        "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
        "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
        "Transport-moving", "Priv-house-serv", "Protective-serv",
        "Armed-Forces",
    ],
    "relationship": [
        "Wife", "Own-child", "Husband", "Not-in-family",
        "Other-relative", "Unmarried",
    ],
    "race": [
        "White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black",
    ],
    "sex": ["Female", "Male"],
    "income": ["LessThan50K", "MoreThan50K"],
}

NUMERIC_OR_RANGE_COLUMNS = {
    "age",
    "hoursperweek",
    "hours-per-week",
    "hours_per_week",
    "education_num",
    "education-num",
    "fnlwgt",
    "capital_gain",
    "capital-gain",
    "capital_loss",
    "capital-loss",
}


def parse_args():
    p = argparse.ArgumentParser(description="Adult full-table repair evaluation")
    p.add_argument("--clean_file", default=DEFAULT_CLEAN_FILE)
    p.add_argument("--dirty_file", default=DEFAULT_DIRTY_FILE)
    p.add_argument("--repair_result_file", default="")
    p.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--case_insensitive_text", type=int, default=1)
    p.add_argument("--strict_mode", type=int, default=0)
    return p.parse_args()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def resolve_repair_result_file(path: str) -> str:
    if path and str(path).strip():
        if not os.path.exists(path):
            raise FileNotFoundError(f"修复结果文件不存在: {path}")
        if os.path.isdir(path):
            direct = os.path.join(path, "inference_top1_repairs.csv")
            if os.path.exists(direct):
                return direct
            raise IsADirectoryError(f"repair_result_file 是目录且没有 inference_top1_repairs.csv: {path}")
        return path

    for p in DEFAULT_REPAIR_RESULT_CANDIDATES:
        if os.path.exists(p) and os.path.isfile(p):
            print(f"[INFO] auto detected repair result file: {p}")
            return p

    raise FileNotFoundError("未找到修复结果文件，请用 --repair_result_file 指定 inference_top1_repairs.csv。")


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
    c = norm_raw(column).lower()
    aliases = {
        "marital-status": "maritalstatus",
        "marital_status": "maritalstatus",
        "marital status": "maritalstatus",
        "hours-per-week": "hoursperweek",
        "hours_per_week": "hoursperweek",
        "hours per week": "hoursperweek",
        "native-country": "country",
        "native_country": "country",
        "capital-gain": "capital_gain",
        "capital_gain": "capital_gain",
        "capital-loss": "capital_loss",
        "capital_loss": "capital_loss",
        "education-num": "education_num",
        "education_num": "education_num",
    }
    return aliases.get(c, c)


def normalize_empty(x: Any) -> str:
    s = norm_raw(x)
    if s.lower() in EMPTY_TOKENS:
        return ""
    return s


def canonical_income(x: Any) -> str:
    s = normalize_empty(x)
    if not s:
        return ""
    t = s.replace(".", "").replace(" ", "")
    low = t.lower()
    if low in {"<=50k", "<=50", "less_than_50k", "lessthan50k", "less50k", "0", "false"}:
        return "LessThan50K"
    if low in {">50k", ">50", "more_than_50k", "morethan50k", "more50k", "1", "true"}:
        return "MoreThan50K"
    return s


def canonical_domain_value(column: Any, value: Any) -> str:
    col = normalize_col_name(column)
    s = normalize_empty(value)
    if not s:
        return ""
    if col == "income":
        return canonical_income(s)

    domain = ADULT_DOMAIN_VALUES.get(col)
    if domain:
        for v in domain:
            if v.lower() == s.lower():
                return v
        s2 = s.replace("_", "-")
        for v in domain:
            if v.lower() == s2.lower():
                return v
        s3 = s.replace("_", " ")
        for v in domain:
            if v.lower().replace("-", " ") == s3.lower().replace("-", " "):
                return v
    return s


def parse_numeric_or_range(x: Any) -> Tuple[Optional[float], Optional[float]]:
    s = normalize_empty(x)
    if not s:
        return None, None
    s = s.replace(",", "").strip()
    low = s.lower().replace(" ", "")

    if low.startswith("<"):
        try:
            v = float(low[1:])
            return 0.0, v - 1
        except Exception:
            return None, None

    if low.startswith(">"):
        try:
            v = float(low[1:])
            return v + 1, 120.0
        except Exception:
            return None, None

    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", s)
    if m:
        return float(m.group(1)), float(m.group(2))

    try:
        v = float(s)
        return v, v
    except Exception:
        return None, None


def canonical_numeric_or_range(x: Any) -> str:
    lo, hi = parse_numeric_or_range(x)
    if lo is None or hi is None:
        return normalize_empty(x)

    def fmt(v: float) -> str:
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        return f"{v:.6f}".rstrip("0").rstrip(".")

    if abs(lo - hi) < 1e-9:
        return fmt(lo)
    return f"{fmt(lo)}-{fmt(hi)}"


NUMERIC_NORMALIZED_COLUMNS = {normalize_col_name(c) for c in NUMERIC_OR_RANGE_COLUMNS}


def normalize_value_for_compare(column: Any, value: Any, case_insensitive_text: bool = True) -> str:
    col = normalize_col_name(column)
    s = normalize_empty(value)
    if not s:
        return ""

    if col == "income":
        return canonical_income(s)
    if col in ADULT_DOMAIN_VALUES:
        return canonical_domain_value(col, s)
    if col in NUMERIC_NORMALIZED_COLUMNS:
        return canonical_numeric_or_range(s)

    s = re.sub(r"\s+", " ", s.strip())
    return s.lower() if case_insensitive_text else s


def stringify_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        out[col] = out[col].map(norm_raw).astype(object)
    return out


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
        "candidate_rule_support_ratio", "adult_gain", "adult_profile_support",
        "relationship_consistency_gain", "education_age_conflict_reduction",
        "candidate_domain_validity_gain", "candidate_in_adult_domain",
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


def build_full_table_eval(clean_df, dirty_df, repaired_df, repair_df, case_insensitive_text: bool, strict_mode: bool) -> pd.DataFrame:
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

            dirty_norm = normalize_value_for_compare(column, dirty_val, case_insensitive_text)
            clean_norm = normalize_value_for_compare(column, clean_val, case_insensitive_text)
            repaired_norm = normalize_value_for_compare(column, repaired_val, case_insensitive_text)

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
                "top_candidate_sources", "candidate_rule_support_ratio", "adult_gain",
                "adult_profile_support", "relationship_consistency_gain",
                "education_age_conflict_reduction", "candidate_domain_validity_gain",
                "candidate_in_adult_domain",
            ]:
                if extra_col in info:
                    out_col = f"repair_{extra_col}" if extra_col == "dirty_value" else extra_col
                    row[out_col] = info.get(extra_col)

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


def build_adult_category_summary(eval_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column, g in eval_df.groupby("column", sort=False):
        col_norm = normalize_col_name(column)
        tp = int((g["eval_type"] == "TP").sum())
        fp = int((g["eval_type"] == "FP").sum())
        fn = int((g["eval_type"] == "FN").sum())
        rows.append({
            "column": column,
            "normalized_column": col_norm,
            "is_adult_domain_column": int(col_norm in ADULT_DOMAIN_VALUES),
            "is_numeric_or_range_column": int(col_norm in NUMERIC_NORMALIZED_COLUMNS),
            "true_error_cells": int(g["is_true_error"].sum()),
            "touched_cells": int(g["is_touched"].sum()),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "case_or_format_only_differences": int(
                ((g["dirty_value"].astype(str).str.strip() != g["clean_value"].astype(str).str.strip()) &
                 (g["dirty_norm"] == g["clean_norm"])).sum()
            ),
        })
    return pd.DataFrame(rows).sort_values(["true_error_cells", "TP"], ascending=[False, False]).reset_index(drop=True)


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
    output_adult_summary_csv = os.path.join(output_dir, "adult_column_type_summary.csv")

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

    print("[INFO] Building Adult full-table evaluation...")
    eval_df = build_full_table_eval(
        clean_df=clean_df,
        dirty_df=dirty_df,
        repaired_df=repaired_df,
        repair_df=repair_df,
        case_insensitive_text=bool(args.case_insensitive_text),
        strict_mode=bool(args.strict_mode),
    )

    metrics = compute_metrics(eval_df)
    metrics["use_adult_normalized_comparison"] = 1
    metrics["case_insensitive_text"] = int(bool(args.case_insensitive_text))
    metrics["repair_result_file"] = repair_result_file

    column_summary_df = build_column_summary(eval_df)
    gate_summary_df = build_gate_reason_summary(eval_df)
    adult_summary_df = build_adult_category_summary(eval_df)

    eval_df.to_csv(output_cell_details_csv, index=False, encoding="utf-8-sig")
    eval_df[eval_df["eval_type"] == "TP"].to_csv(output_tp_csv, index=False, encoding="utf-8-sig")
    eval_df[eval_df["eval_type"] == "FP"].to_csv(output_fp_csv, index=False, encoding="utf-8-sig")
    eval_df[eval_df["eval_type"] == "FN"].to_csv(output_fn_csv, index=False, encoding="utf-8-sig")
    eval_df[eval_df["eval_type"] == "TN"].to_csv(output_tn_csv, index=False, encoding="utf-8-sig")
    eval_df[eval_df["is_touched"] == 1].to_csv(output_touched_csv, index=False, encoding="utf-8-sig")

    pd.DataFrame([metrics]).to_csv(output_summary_csv, index=False, encoding="utf-8-sig")
    column_summary_df.to_csv(output_column_summary_csv, index=False, encoding="utf-8-sig")
    gate_summary_df.to_csv(output_gate_summary_csv, index=False, encoding="utf-8-sig")
    adult_summary_df.to_csv(output_adult_summary_csv, index=False, encoding="utf-8-sig")

    print("\n========== Adult 全表端到端修复评估结果 ==========")
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

    if not adult_summary_df.empty:
        print("\n---- Adult Column Type 摘要 ----")
        show_cols = [
            "column", "normalized_column", "is_adult_domain_column", "is_numeric_or_range_column",
            "true_error_cells", "touched_cells", "TP", "FP", "FN", "case_or_format_only_differences",
        ]
        print(adult_summary_df[show_cols].head(30).to_string(index=False))

    print(f"\n[OK] 评估摘要已保存: {output_summary_csv}")
    print(f"[OK] 列级评估摘要已保存: {output_column_summary_csv}")
    print(f"[OK] gate reason 摘要已保存: {output_gate_summary_csv}")
    print(f"[OK] Adult 列类型摘要已保存: {output_adult_summary_csv}")
    print(f"[OK] 修复后的整表已保存: {output_repaired_table_csv}")
    print(f"[OK] 全量单元格评估明细已保存: {output_cell_details_csv}")
    print(f"[OK] touched 明细已保存: {output_touched_csv}")
    print(f"[OK] TP 明细已保存: {output_tp_csv}")
    print(f"[OK] FP 明细已保存: {output_fp_csv}")
    print(f"[OK] FN 明细已保存: {output_fn_csv}")
    print(f"[OK] TN 明细已保存: {output_tn_csv}")


if __name__ == "__main__":
    main()

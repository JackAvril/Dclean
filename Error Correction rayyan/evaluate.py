import os
import re
import pandas as pd

# ============================================================
# 1. 用户配置区
# ============================================================

# Rayyan clean / dirty 表
CLEAN_FILE = "/mnt/mydata/dq/projects/Splittree/rayyan/rayyan_clean.csv"
DIRTY_FILE = "/mnt/mydata/dq/projects/Splittree/rayyan/rayyan_dirty.csv"

# Rayyan 推理结果
REPAIR_RESULT_FILE = "/mnt/mydata/dq/projects/Splittree/test/Error Correction rayyan/inference_top1_repairs.csv"

OUTPUT_DIR = (
    "/mnt/mydata/dq/projects/Splittree/test/Error Correction rayyan/"
    "repair_eval_output_rayyan"
)

OUTPUT_SUMMARY_CSV = os.path.join(OUTPUT_DIR, "full_table_eval_summary.csv")
OUTPUT_CELL_DETAILS_CSV = os.path.join(OUTPUT_DIR, "full_table_eval_all_cells.csv")
OUTPUT_TP_CSV = os.path.join(OUTPUT_DIR, "full_table_tp_details.csv")
OUTPUT_FP_CSV = os.path.join(OUTPUT_DIR, "full_table_fp_details.csv")
OUTPUT_FN_CSV = os.path.join(OUTPUT_DIR, "full_table_fn_details.csv")
OUTPUT_TN_CSV = os.path.join(OUTPUT_DIR, "full_table_tn_details.csv")
OUTPUT_REPAIRED_TABLE_CSV = os.path.join(OUTPUT_DIR, "repaired_table.csv")
OUTPUT_TOUCHED_CSV = os.path.join(OUTPUT_DIR, "touched_cells_details.csv")
OUTPUT_COLUMN_SUMMARY_CSV = os.path.join(OUTPUT_DIR, "column_level_eval_summary.csv")
OUTPUT_GATE_REASON_SUMMARY_CSV = os.path.join(OUTPUT_DIR, "gate_reason_eval_summary.csv")
OUTPUT_RAYYAN_CANONICAL_SUMMARY_CSV = os.path.join(OUTPUT_DIR, "rayyan_canonical_eval_summary.csv")
OUTPUT_SOURCE_SUMMARY_CSV = os.path.join(OUTPUT_DIR, "candidate_source_eval_summary.csv")

# 是否使用 Rayyan 专属规范化比较。
USE_RAYYAN_NORMALIZED_COMPARISON = True

# 普通文本列是否忽略大小写。默认 False，保持 strict textual evaluation。
TEXT_COLUMNS_CASE_INSENSITIVE = False
COMPRESS_TEXT_WHITESPACE = True


# ============================================================
# 2. 基础工具函数
# ============================================================

EMPTY_TOKENS = {"", "empty", "null", "none", "nan", "na", "n/a", "missing"}

LANGUAGE_CANONICAL_MAP = {
    "eng": "eng", "english": "eng", "en": "eng", "engl": "eng", "anglais": "eng",
    "fre": "fre", "french": "fre", "fr": "fre", "fra": "fre",
    "ger": "ger", "german": "ger", "de": "ger", "deu": "ger",
    "spa": "spa", "spanish": "spa", "es": "spa",
    "jpn": "jpn", "japanese": "jpn", "ja": "jpn",
    "chi": "chi", "chinese": "chi", "zh": "chi",
    "ita": "ita", "italian": "ita", "it": "ita",
    "dut": "dut", "dutch": "dut", "nl": "dut",
    "pol": "pol", "polish": "pol",
    "por": "por", "portuguese": "por",
    "rus": "rus", "russian": "rus",
}

RAYYAN_CANONICAL_COLUMNS = {
    "article_language",
    "journal_issn",
    "article_jcreated_at",
    "article_jvolumn",
    "article_jissue",
    "article_pagination",
}

RAYYAN_TEXT_METADATA_COLUMNS = {
    "journal_title",
    "jounral_abbreviation",
    "journal_abbreviation",
    "article_title",
    "author_list",
}


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def norm_text(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in EMPTY_TOKENS:
        return ""
    return s


def safe_float(x, default=0.0):
    try:
        if x is None or pd.isna(x):
            return default
        return float(str(x).strip().replace(",", ""))
    except Exception:
        return default


def safe_int(x, default=0):
    try:
        if x is None or pd.isna(x):
            return default
        return int(float(str(x).strip().replace(",", "")))
    except Exception:
        return default


def read_table(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"表文件不存在: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def validate_same_shape(clean_df: pd.DataFrame, dirty_df: pd.DataFrame):
    if clean_df.shape != dirty_df.shape:
        raise ValueError(
            f"clean 与 dirty 形状不一致: clean={clean_df.shape}, dirty={dirty_df.shape}"
        )

    if list(clean_df.columns) != list(dirty_df.columns):
        raise ValueError(
            "clean 与 dirty 列名不一致，请先对齐列顺序和列名。\n"
            f"clean columns={list(clean_df.columns)}\n"
            f"dirty columns={list(dirty_df.columns)}"
        )


def stringify_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        out[col] = out[col].map(norm_text).astype(object)
    return out


def compact_text(s: str) -> str:
    s = norm_text(s)
    if COMPRESS_TEXT_WHITESPACE:
        s = re.sub(r"\s+", " ", s).strip()
    return s


# ============================================================
# 3. Rayyan 专属规范化比较
# ============================================================

def normalize_language_candidate(x):
    s = norm_text(x)
    if s == "":
        return ""

    lower = s.lower()
    if "abstract language" in lower:
        parts = re.split(r"abstract language\s*:\s*", lower)
        if len(parts) >= 2:
            lower = parts[-1].strip(";: ,.")

    lower = lower.strip(";: ,.")
    if lower in LANGUAGE_CANONICAL_MAP:
        return LANGUAGE_CANONICAL_MAP[lower]

    token = re.sub(r"[^a-z]", "", lower)
    if token in LANGUAGE_CANONICAL_MAP:
        return LANGUAGE_CANONICAL_MAP[token]

    if len(token) == 3 and token.isalpha():
        return token

    return lower


def normalize_issn_candidate(x):
    s = norm_text(x)
    if s == "":
        return ""

    raw = s.upper().strip()
    raw = raw.replace("ISSN", "").replace(":", " ").strip()
    chars = re.sub(r"[^0-9X]", "", raw)

    if len(chars) != 8:
        return s

    return f"{chars[:4]}-{chars[4:]}"


def is_valid_issn(x):
    fixed = normalize_issn_candidate(x)
    if not fixed or not re.fullmatch(r"\d{4}-\d{3}[\dX]", fixed):
        return False

    digits = fixed.replace("-", "")
    total = 0
    try:
        for i, ch in enumerate(digits):
            val = 10 if ch == "X" else int(ch)
            total += val * (8 - i)
        return total % 11 == 0
    except Exception:
        return False


def normalize_date_candidate(x):
    s = norm_text(x)
    if s == "":
        return ""

    s = s.strip()

    m = re.fullmatch(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{mo}/{d}/{str(y)[-2:]}"

    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{2}|\d{4})", s)
    if m:
        mo, d, y = int(m.group(1)), int(m.group(2)), m.group(3)
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{mo}/{d}/{y[-2:]}"

    return s


def normalize_numeric_int_candidate(x, min_v=None, max_v=None):
    s = norm_text(x)
    if s == "":
        return ""

    m = re.search(r"-?\d+", s)
    if not m:
        return s

    v = int(m.group(0))
    if min_v is not None and v < min_v:
        return s
    if max_v is not None and v > max_v:
        return s
    return str(v)


def normalize_volume_candidate(x):
    return normalize_numeric_int_candidate(x, min_v=0, max_v=10000)


def normalize_issue_candidate(x):
    return normalize_numeric_int_candidate(x, min_v=0, max_v=1000)


def normalize_pagination_candidate(x):
    s = norm_text(x)
    if s == "":
        return ""

    s2 = s.strip()
    s2 = s2.replace("–", "-").replace("—", "-").replace("−", "-")
    s2 = re.sub(r"\s*-\s*", "-", s2)
    s2 = re.sub(r"\s+", " ", s2).strip()
    return s2


def normalize_for_compare(column: str, value):
    s = norm_text(value)

    if not USE_RAYYAN_NORMALIZED_COMPARISON:
        return s

    c = str(column).strip().lower()

    if c == "article_language":
        return normalize_language_candidate(s)
    if c == "journal_issn":
        return normalize_issn_candidate(s)
    if c == "article_jcreated_at":
        return normalize_date_candidate(s)
    if c == "article_jvolumn":
        return normalize_volume_candidate(s)
    if c == "article_jissue":
        return normalize_issue_candidate(s)
    if c == "article_pagination":
        return normalize_pagination_candidate(s)
    if c in {"index", "id"}:
        return normalize_numeric_int_candidate(s, min_v=0)

    text = compact_text(s)
    if TEXT_COLUMNS_CASE_INSENSITIVE and c in RAYYAN_TEXT_METADATA_COLUMNS:
        return text.lower()
    return text


def equal_for_eval(column: str, a, b) -> bool:
    return normalize_for_compare(column, a) == normalize_for_compare(column, b)


# ============================================================
# 4. 修复结果读取与写回
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
        "candidate_value",
        "teacher_best_candidate",
    ]

    for c in candidates:
        if c in df.columns:
            return c

    raise ValueError(
        "修复结果文件中未找到修复值字段。支持字段包括："
        "repaired_value / predicted_repair_value / predicted_repair / repair_value / predicted_value / "
        "student_best_candidate / student_top_candidate / candidate_value / teacher_best_candidate"
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
        "dirty_value",
        "student_top_candidate",
        "student_best_candidate",
        "student_score",
        "student_margin",
        "margin",
        "top2_score",
        "evidence_strength",
        "rule_support",
        "neighbor_majority_ratio",
        "rayyan_consensus_strength",
        "rayyan_consensus_confidence",
        "rayyan_consensus_support_count",
        "is_rayyan_consensus_candidate",
        "gate_passed",
        "pass_gate",
        "gate_reason",
        "student_reason_pred",
        "candidate_count",
        "candidate_source",
        "candidate_source_text",
        "contains_rayyan_metadata_source",
        "contains_journal_metadata_source",
        "contains_article_metadata_source",
        "contains_canonical_source",
        "contains_rayyan_consensus_source",
        "rayyan_consensus_candidate",
        "is_from_rayyan_consensus",
        "rayyan_consensus_table_match",
        "rayyan_consensus_table_max_confidence",
        "rayyan_consensus_table_max_margin",
        "rayyan_consensus_table_max_support_count",
        "rayyan_consensus_table_high_conf_match",
        "rayyan_candidate_is_language",
        "rayyan_candidate_is_issn",
        "rayyan_candidate_is_valid_issn",
        "rayyan_candidate_is_date",
        "rayyan_candidate_is_volume",
        "rayyan_candidate_is_issue",
        "rayyan_candidate_is_pagination",
        "rayyan_candidate_is_format_restoration",
        "candidate_equals_allowed_consensus_value",
        "candidate_and_allowed_both_rayyan_consensus",
        "candidate_under_strong_rayyan_consensus_cell",
        "allowed_consensus_name",
        "allowed_lhs_key",
        "allowed_consensus_value",
        "allowed_consensus_confidence",
        "allowed_consensus_margin",
        "allowed_consensus_support_count",
        "allowed_consensus_total_count",
        "allowed_consensus_reason",
        "allowed_is_rayyan_metadata_consensus",
        "allowed_is_strong_rayyan_consensus",
        "is_hard_case",
    ]

    for c in optional_cols:
        if c in df.columns and c not in keep_cols:
            keep_cols.append(c)

    out = df[keep_cols].copy()
    out = out.rename(columns={repair_col: "predicted_repair"})
    out["predicted_repair"] = out["predicted_repair"].map(norm_text)
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
        pred_val = norm_text(r["predicted_repair"])

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
# 5. 全表评估逻辑
# ============================================================

def build_full_table_eval(clean_df, dirty_df, repaired_df, repair_df):
    repair_pos_set = set(
        (int(r["row_id"]), str(r["column"]).strip())
        for _, r in repair_df.iterrows()
    )

    repair_info_map = {}
    for _, r in repair_df.iterrows():
        key = (int(r["row_id"]), str(r["column"]).strip())
        repair_info_map[key] = r.to_dict()

    rows = []

    for row_id in range(len(clean_df)):
        for column in clean_df.columns:
            column = str(column)

            clean_val = norm_text(clean_df.at[row_id, column])
            dirty_val = norm_text(dirty_df.at[row_id, column])
            repaired_val = norm_text(repaired_df.at[row_id, column])

            clean_cmp = normalize_for_compare(column, clean_val)
            dirty_cmp = normalize_for_compare(column, dirty_val)
            repaired_cmp = normalize_for_compare(column, repaired_val)

            is_true_error = int(dirty_cmp != clean_cmp)
            is_touched = int((row_id, column) in repair_pos_set or repaired_cmp != dirty_cmp)
            is_final_correct = int(repaired_cmp == clean_cmp)

            if is_true_error == 1 and is_final_correct == 1:
                eval_type = "TP"
            elif is_true_error == 0 and repaired_cmp != clean_cmp:
                eval_type = "FP"
            elif is_true_error == 1 and is_final_correct == 0:
                eval_type = "FN"
            else:
                eval_type = "TN"

            info = repair_info_map.get((row_id, column), {})
            c = column.lower()

            row = {
                "row_id": row_id,
                "column": column,
                "dirty_value": dirty_val,
                "clean_value": clean_val,
                "repaired_value": repaired_val,
                "dirty_value_cmp": dirty_cmp,
                "clean_value_cmp": clean_cmp,
                "repaired_value_cmp": repaired_cmp,
                "is_true_error": is_true_error,
                "is_touched": is_touched,
                "is_final_correct": is_final_correct,
                "eval_type": eval_type,
                "rayyan_column_is_canonical": int(c in RAYYAN_CANONICAL_COLUMNS),
                "rayyan_column_is_text_metadata": int(c in RAYYAN_TEXT_METADATA_COLUMNS),
                "dirty_issn_valid": int(c == "journal_issn" and is_valid_issn(dirty_val)),
                "clean_issn_valid": int(c == "journal_issn" and is_valid_issn(clean_val)),
                "repaired_issn_valid": int(c == "journal_issn" and is_valid_issn(repaired_val)),
            }

            for extra_col in [
                "predicted_repair",
                "student_top_candidate",
                "student_best_candidate",
                "student_score",
                "student_margin",
                "margin",
                "top2_score",
                "evidence_strength",
                "rule_support",
                "neighbor_majority_ratio",
                "rayyan_consensus_strength",
                "rayyan_consensus_confidence",
                "rayyan_consensus_support_count",
                "is_rayyan_consensus_candidate",
                "gate_passed",
                "pass_gate",
                "gate_reason",
                "student_reason_pred",
                "candidate_count",
                "candidate_source",
                "candidate_source_text",
                "contains_rayyan_metadata_source",
                "contains_journal_metadata_source",
                "contains_article_metadata_source",
                "contains_canonical_source",
                "contains_rayyan_consensus_source",
                "rayyan_consensus_candidate",
                "is_from_rayyan_consensus",
                "rayyan_consensus_table_match",
                "rayyan_consensus_table_max_confidence",
                "rayyan_consensus_table_max_margin",
                "rayyan_consensus_table_max_support_count",
                "rayyan_consensus_table_high_conf_match",
                "rayyan_candidate_is_language",
                "rayyan_candidate_is_issn",
                "rayyan_candidate_is_valid_issn",
                "rayyan_candidate_is_date",
                "rayyan_candidate_is_volume",
                "rayyan_candidate_is_issue",
                "rayyan_candidate_is_pagination",
                "rayyan_candidate_is_format_restoration",
                "candidate_equals_allowed_consensus_value",
                "candidate_and_allowed_both_rayyan_consensus",
                "candidate_under_strong_rayyan_consensus_cell",
                "allowed_consensus_name",
                "allowed_lhs_key",
                "allowed_consensus_value",
                "allowed_consensus_confidence",
                "allowed_consensus_margin",
                "allowed_consensus_support_count",
                "allowed_consensus_total_count",
                "allowed_consensus_reason",
                "allowed_is_rayyan_metadata_consensus",
                "allowed_is_strong_rayyan_consensus",
                "is_hard_case",
            ]:
                if extra_col in info:
                    row[extra_col] = info.get(extra_col)

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
        "changed_cells": int((eval_df["repaired_value_cmp"] != eval_df["dirty_value_cmp"]).sum()),
        "remaining_wrong_cells_after_repair": int((eval_df["repaired_value_cmp"] != eval_df["clean_value_cmp"]).sum()),
        "unchanged_true_errors": int(((eval_df["is_true_error"] == 1) & (eval_df["repaired_value_cmp"] == eval_df["dirty_value_cmp"])).sum()),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Accuracy": accuracy,
        "strict_changed_cells": int((eval_df["repaired_value"] != eval_df["dirty_value"]).sum()),
        "strict_remaining_wrong_cells_after_repair": int((eval_df["repaired_value"] != eval_df["clean_value"]).sum()),
        "use_rayyan_normalized_comparison": int(USE_RAYYAN_NORMALIZED_COMPARISON),
        "text_columns_case_insensitive": int(TEXT_COLUMNS_CASE_INSENSITIVE),
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
            "changed_cells": int((g["repaired_value_cmp"] != g["dirty_value_cmp"]).sum()),
            "remaining_wrong_cells_after_repair": int((g["repaired_value_cmp"] != g["clean_value_cmp"]).sum()),
            "strict_changed_cells": int((g["repaired_value"] != g["dirty_value"]).sum()),
            "strict_remaining_wrong_cells_after_repair": int((g["repaired_value"] != g["clean_value"]).sum()),
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
    touched = eval_df[eval_df["is_touched"] == 1].copy()
    if touched.empty:
        return pd.DataFrame()
    rows = []
    for (column, gate_reason), g in touched.groupby(["column", "gate_reason"], dropna=False):
        tp = int((g["eval_type"] == "TP").sum())
        fp = int((g["eval_type"] == "FP").sum())
        rows.append({
            "column": column,
            "gate_reason": gate_reason,
            "count": len(g),
            "TP": tp,
            "FP": fp,
            "FN": int((g["eval_type"] == "FN").sum()),
            "TN": int((g["eval_type"] == "TN").sum()),
            "precision_within_reason": tp / max(tp + fp, 1),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["FP", "FN", "count"], ascending=[False, False, False]).reset_index(drop=True)
    return out


def build_rayyan_canonical_summary(eval_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in sorted(set(eval_df["column"])):
        c = str(column).lower()
        if c not in RAYYAN_CANONICAL_COLUMNS and c not in RAYYAN_TEXT_METADATA_COLUMNS:
            continue
        g = eval_df[eval_df["column"] == column].copy()
        rows.append({
            "column": column,
            "is_canonical_column": int(c in RAYYAN_CANONICAL_COLUMNS),
            "is_text_metadata_column": int(c in RAYYAN_TEXT_METADATA_COLUMNS),
            "total_cells": len(g),
            "true_error_cells": int(g["is_true_error"].sum()),
            "touched_cells": int(g["is_touched"].sum()),
            "TP": int((g["eval_type"] == "TP").sum()),
            "FP": int((g["eval_type"] == "FP").sum()),
            "FN": int((g["eval_type"] == "FN").sum()),
            "TN": int((g["eval_type"] == "TN").sum()),
            "case_or_format_only_differences": int(((g["repaired_value"] != g["clean_value"]) & (g["repaired_value_cmp"] == g["clean_value_cmp"])).sum()),
            "dirty_cmp_unique": int(g["dirty_value_cmp"].nunique(dropna=False)),
            "clean_cmp_unique": int(g["clean_value_cmp"].nunique(dropna=False)),
            "repaired_cmp_unique": int(g["repaired_value_cmp"].nunique(dropna=False)),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["true_error_cells", "FP", "FN"], ascending=[False, False, False]).reset_index(drop=True)
    return out


def build_source_summary(eval_df: pd.DataFrame) -> pd.DataFrame:
    source_col = None
    for c in ["candidate_source", "candidate_source_text"]:
        if c in eval_df.columns:
            source_col = c
            break
    if source_col is None:
        return pd.DataFrame()
    touched = eval_df[eval_df["is_touched"] == 1].copy()
    if touched.empty:
        return pd.DataFrame()
    touched[source_col] = touched[source_col].fillna("").astype(str)
    rows = []
    for source, g in touched.groupby(source_col, dropna=False):
        rows.append({
            "candidate_source": source,
            "count": len(g),
            "TP": int((g["eval_type"] == "TP").sum()),
            "FP": int((g["eval_type"] == "FP").sum()),
            "FN": int((g["eval_type"] == "FN").sum()),
            "TN": int((g["eval_type"] == "TN").sum()),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["FP", "FN", "count"], ascending=[False, False, False]).reset_index(drop=True)
    return out


# ============================================================
# 6. main
# ============================================================

def main():
    ensure_dir(OUTPUT_DIR)

    print("[INFO] Loading clean / dirty tables...")
    clean_df_raw = read_table(CLEAN_FILE)
    dirty_df_raw = read_table(DIRTY_FILE)
    validate_same_shape(clean_df_raw, dirty_df_raw)

    clean_df = stringify_table(clean_df_raw)
    dirty_df = stringify_table(dirty_df_raw)

    print("[INFO] Loading repair results...")
    repair_df = load_repair_results(REPAIR_RESULT_FILE)

    print("[INFO] Applying repairs...")
    repaired_df = apply_repairs_to_dirty(dirty_df, repair_df)
    repaired_df.to_csv(OUTPUT_REPAIRED_TABLE_CSV, index=False, encoding="utf-8-sig")

    print("[INFO] Building full-table evaluation...")
    eval_df = build_full_table_eval(clean_df, dirty_df, repaired_df, repair_df)
    metrics = compute_metrics(eval_df)
    column_summary_df = build_column_summary(eval_df)
    gate_reason_summary_df = build_gate_reason_summary(eval_df)
    rayyan_canonical_summary_df = build_rayyan_canonical_summary(eval_df)
    source_summary_df = build_source_summary(eval_df)

    eval_df.to_csv(OUTPUT_CELL_DETAILS_CSV, index=False, encoding="utf-8-sig")
    eval_df[eval_df["eval_type"] == "TP"].to_csv(OUTPUT_TP_CSV, index=False, encoding="utf-8-sig")
    eval_df[eval_df["eval_type"] == "FP"].to_csv(OUTPUT_FP_CSV, index=False, encoding="utf-8-sig")
    eval_df[eval_df["eval_type"] == "FN"].to_csv(OUTPUT_FN_CSV, index=False, encoding="utf-8-sig")
    eval_df[eval_df["eval_type"] == "TN"].to_csv(OUTPUT_TN_CSV, index=False, encoding="utf-8-sig")
    eval_df[eval_df["is_touched"] == 1].to_csv(OUTPUT_TOUCHED_CSV, index=False, encoding="utf-8-sig")

    pd.DataFrame([metrics]).to_csv(OUTPUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    column_summary_df.to_csv(OUTPUT_COLUMN_SUMMARY_CSV, index=False, encoding="utf-8-sig")

    if not gate_reason_summary_df.empty:
        gate_reason_summary_df.to_csv(OUTPUT_GATE_REASON_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    if not rayyan_canonical_summary_df.empty:
        rayyan_canonical_summary_df.to_csv(OUTPUT_RAYYAN_CANONICAL_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    if not source_summary_df.empty:
        source_summary_df.to_csv(OUTPUT_SOURCE_SUMMARY_CSV, index=False, encoding="utf-8-sig")

    print("\n========== Rayyan 全表端到端修复评估结果 ==========")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")

    print("\n---- 各列评估摘要 Top ----")
    if not column_summary_df.empty:
        show_cols = ["column", "true_error_cells", "touched_cells", "TP", "FP", "FN", "Precision", "Recall", "F1"]
        print(column_summary_df[show_cols].head(20).to_string(index=False))

    if not gate_reason_summary_df.empty:
        print("\n---- Gate Reason 摘要 Top ----")
        print(gate_reason_summary_df.head(20).to_string(index=False))

    if not rayyan_canonical_summary_df.empty:
        print("\n---- Rayyan Canonical / Metadata 摘要 ----")
        show_cols = [
            "column", "is_canonical_column", "is_text_metadata_column", "true_error_cells",
            "touched_cells", "TP", "FP", "FN", "case_or_format_only_differences"
        ]
        print(rayyan_canonical_summary_df[show_cols].head(20).to_string(index=False))

    print(f"\n[OK] 评估摘要已保存: {OUTPUT_SUMMARY_CSV}")
    print(f"[OK] 列级评估摘要已保存: {OUTPUT_COLUMN_SUMMARY_CSV}")
    print(f"[OK] 修复后的整表已保存: {OUTPUT_REPAIRED_TABLE_CSV}")
    print(f"[OK] 全量单元格评估明细已保存: {OUTPUT_CELL_DETAILS_CSV}")
    print(f"[OK] touched 明细已保存: {OUTPUT_TOUCHED_CSV}")
    print(f"[OK] TP 明细已保存: {OUTPUT_TP_CSV}")
    print(f"[OK] FP 明细已保存: {OUTPUT_FP_CSV}")
    print(f"[OK] FN 明细已保存: {OUTPUT_FN_CSV}")
    print(f"[OK] TN 明细已保存: {OUTPUT_TN_CSV}")
    if not gate_reason_summary_df.empty:
        print(f"[OK] gate reason 摘要已保存: {OUTPUT_GATE_REASON_SUMMARY_CSV}")
    if not rayyan_canonical_summary_df.empty:
        print(f"[OK] Rayyan canonical 摘要已保存: {OUTPUT_RAYYAN_CANONICAL_SUMMARY_CSV}")
    if not source_summary_df.empty:
        print(f"[OK] candidate source 摘要已保存: {OUTPUT_SOURCE_SUMMARY_CSV}")


if __name__ == "__main__":
    main()

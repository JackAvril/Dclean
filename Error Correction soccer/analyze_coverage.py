#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze Soccer repair candidate coverage / oracle upper bound.

运行位置建议：
cd /mnt/mydata/dq/projects/Splittree/test/Error\ Correction\ soccer
python analyze_soccer_candidate_coverage.py

核心判断：
1. clean 不在候选列表里：候选生成问题
2. clean 在候选列表里但 rank 靠后：排序/特征/训练问题
3. clean 已经是 top1 但最终没修：推理 gate 或回填问题
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# ============================================================
# 1. 路径配置
# ============================================================

BASE_DIR = "/mnt/mydata/dq/projects/Splittree/test/Error Correction soccer"

CLEAN_FILE = "/mnt/mydata/dq/projects/Splittree/soccer/soccer_clean.csv"
DIRTY_FILE = "/mnt/mydata/dq/projects/Splittree/soccer/soccer_dirty.csv"

EVAL_DIR = os.path.join(BASE_DIR, "repair_eval_output_soccer")
FULL_TABLE_EVAL_FILE = os.path.join(EVAL_DIR, "full_table_eval_all_cells.csv")

RANKED_CANDIDATES_FILE = os.path.join(
    BASE_DIR,
    "repair_iterative_workdir_soccer",
    "model_output",
    "inference_all_candidate_ranked.csv",
)

OUTPUT_DIR = os.path.join(BASE_DIR, "candidate_coverage_analysis_soccer")

OUTPUT_SUMMARY_OVERALL = os.path.join(OUTPUT_DIR, "candidate_oracle_summary_overall.csv")
OUTPUT_SUMMARY_BY_COLUMN = os.path.join(OUTPUT_DIR, "candidate_oracle_summary_by_column.csv")
OUTPUT_TRUE_ERROR_DETAILS = os.path.join(OUTPUT_DIR, "candidate_oracle_true_error_details.csv")
OUTPUT_FN_DETAILS = os.path.join(OUTPUT_DIR, "candidate_oracle_fn_details.csv")
OUTPUT_CANDIDATE_CELL_SUMMARY = os.path.join(OUTPUT_DIR, "candidate_cell_summary.csv")
OUTPUT_MISSING_CANDIDATES = os.path.join(OUTPUT_DIR, "candidate_missing_true_errors.csv")
OUTPUT_RANKING_ERRORS = os.path.join(OUTPUT_DIR, "candidate_ranking_errors.csv")
OUTPUT_GATE_ERRORS = os.path.join(OUTPUT_DIR, "candidate_gate_errors.csv")


# ============================================================
# 2. 比较口径
# ============================================================

EMPTY_TOKENS = {"", "empty", "null", "none", "nan", "na", "n/a", "missing", "?", "nil"}

YEAR_COLUMNS = {"birthyear", "season"}
POSITION_COLUMNS = {"position"}
TEXT_COLUMNS_CASE_INSENSITIVE = {"name", "surname", "birthplace", "team", "city", "stadium", "manager"}

POSITION_MAP = {
    "goalkeeper": "Goalkeeper",
    "keeper": "Goalkeeper",
    "gk": "Goalkeeper",
    "defender": "Defender",
    "defence": "Defender",
    "defense": "Defender",
    "df": "Defender",
    "midfield": "Midfield",
    "midfielder": "Midfield",
    "mf": "Midfield",
    "forward": "Forward",
    "striker": "Forward",
    "winger": "Forward",
    "fw": "Forward",
}


def is_empty_value(x: Any) -> bool:
    if x is None:
        return True
    try:
        if pd.isna(x):
            return True
    except Exception:
        pass
    return str(x).strip().lower() in EMPTY_TOKENS


def display_value(x: Any) -> str:
    if is_empty_value(x):
        return ""
    return str(x).strip()


def normalize_year_like(x: Any) -> str:
    if is_empty_value(x):
        return ""
    s = str(x).strip()
    trans = str.maketrans({
        "O": "0", "o": "0",
        "I": "1", "l": "1", "|": "1",
        "S": "5", "s": "5",
        "B": "8",
    })
    s2 = s.translate(trans)

    m = re.search(r"(18\d{2}|19\d{2}|20\d{2})", s2)
    if m:
        return m.group(1)

    try:
        f = float(s2)
        if f.is_integer():
            i = int(f)
            if 1800 <= i <= 2100:
                return str(i)
    except Exception:
        pass

    return s


def normalize_position(x: Any) -> str:
    if is_empty_value(x):
        return ""
    s = str(x).strip()
    key = re.sub(r"[^a-zA-Z]", "", s).lower()
    if key in POSITION_MAP:
        return POSITION_MAP[key]
    if s.lower() in POSITION_MAP:
        return POSITION_MAP[s.lower()]
    return s


def norm_for_compare(x: Any, column: str) -> str:
    col = str(column).strip()
    if col in YEAR_COLUMNS:
        return normalize_year_like(x)
    if col in POSITION_COLUMNS:
        return normalize_position(x)
    if col in TEXT_COLUMNS_CASE_INSENSITIVE:
        return display_value(x).lower()
    return display_value(x)


# ============================================================
# 3. 通用工具
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_csv(path: str, desc: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{desc} 不存在: {path}")
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x: Any, default: int = 999999) -> int:
    try:
        if pd.isna(x):
            return default
        return int(float(x))
    except Exception:
        return default


def find_candidate_value_col(df: pd.DataFrame) -> str:
    candidates = [
        "candidate_value",
        "repair_candidate",
        "candidate",
        "value_candidate",
        "predicted_repair",
        "predicted_repair_value",
        "student_best_candidate",
        "student_top_candidate",
        "top_candidate_value",
        "teacher_best_candidate",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        "候选文件中找不到候选值字段。支持字段: "
        + ", ".join(candidates)
        + f"\n实际字段: {list(df.columns)}"
    )


def find_rank_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["student_rank", "candidate_rank", "rank", "ltr_rank", "final_rank"]:
        if c in df.columns:
            return c
    return None


def find_score_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["student_score", "score", "pred_score", "model_score", "ltr_score", "final_score", "candidate_score"]:
        if c in df.columns:
            return c
    return None


def find_source_cols(df: pd.DataFrame) -> List[str]:
    out = []
    for c in [
        "candidate_source",
        "candidate_sources",
        "source",
        "sources",
        "candidate_reason",
        "generation_reason",
        "candidate_generation_reason",
        "repair_source",
        "evidence_source",
    ]:
        if c in df.columns:
            out.append(c)
    return out


# ============================================================
# 4. 读取真实错误
# ============================================================

def build_true_error_df_from_clean_dirty() -> pd.DataFrame:
    clean_df = read_csv(CLEAN_FILE, "clean file")
    dirty_df = read_csv(DIRTY_FILE, "dirty file")

    if clean_df.shape != dirty_df.shape:
        raise ValueError(f"clean/dirty shape 不一致: clean={clean_df.shape}, dirty={dirty_df.shape}")
    if list(clean_df.columns) != list(dirty_df.columns):
        raise ValueError("clean/dirty 列不一致")

    rows = []
    for row_id in range(len(clean_df)):
        for col in clean_df.columns:
            dirty = dirty_df.at[row_id, col]
            clean = clean_df.at[row_id, col]
            if norm_for_compare(dirty, col) != norm_for_compare(clean, col):
                rows.append({
                    "row_id": row_id,
                    "column": str(col),
                    "dirty_value": display_value(dirty),
                    "clean_value": display_value(clean),
                    "repaired_value": "",
                    "eval_type": "",
                    "gate_reason": "",
                    "dirty_norm": norm_for_compare(dirty, col),
                    "clean_norm": norm_for_compare(clean, col),
                })
    return pd.DataFrame(rows)


def load_true_error_eval_df() -> pd.DataFrame:
    if os.path.exists(FULL_TABLE_EVAL_FILE):
        df = read_csv(FULL_TABLE_EVAL_FILE, "full_table_eval_all_cells")
        if "row_id" not in df.columns or "column" not in df.columns:
            raise ValueError("full_table_eval_all_cells.csv 必须包含 row_id,column")

        if "is_true_error" in df.columns:
            mask = pd.to_numeric(df["is_true_error"], errors="coerce").fillna(0).astype(int) == 1
            true_df = df.loc[mask].copy()
        elif "eval_type" in df.columns:
            true_df = df[df["eval_type"].isin(["TP", "FN"])].copy()
        else:
            true_df = df[
                df.apply(
                    lambda r: norm_for_compare(r.get("dirty_value", ""), r.get("column", ""))
                    != norm_for_compare(r.get("clean_value", ""), r.get("column", "")),
                    axis=1,
                )
            ].copy()

        for c in ["dirty_value", "clean_value", "repaired_value", "eval_type", "gate_reason"]:
            if c not in true_df.columns:
                true_df[c] = ""

        true_df["row_id"] = pd.to_numeric(true_df["row_id"], errors="raise").astype(int)
        true_df["column"] = true_df["column"].astype(str).str.strip()
        true_df["dirty_norm"] = true_df.apply(lambda r: norm_for_compare(r.get("dirty_value", ""), r["column"]), axis=1)
        true_df["clean_norm"] = true_df.apply(lambda r: norm_for_compare(r.get("clean_value", ""), r["column"]), axis=1)
        return true_df.reset_index(drop=True)

    print("[WARN] 没有找到 full_table_eval_all_cells.csv，改用 clean/dirty 重新构造真实错误。")
    return build_true_error_df_from_clean_dirty()


# ============================================================
# 5. 读取候选
# ============================================================

def load_ranked_candidates() -> pd.DataFrame:
    df = read_csv(RANKED_CANDIDATES_FILE, "inference_all_candidate_ranked.csv")

    if "row_id" not in df.columns or "column" not in df.columns:
        raise ValueError("候选排序文件必须包含 row_id,column")

    df["row_id"] = pd.to_numeric(df["row_id"], errors="raise").astype(int)
    df["column"] = df["column"].astype(str).str.strip()

    cand_col = find_candidate_value_col(df)
    rank_col = find_rank_col(df)
    score_col = find_score_col(df)
    source_cols = find_source_cols(df)

    df["_candidate_value_raw"] = df[cand_col].map(display_value)
    df["_candidate_norm"] = df.apply(lambda r: norm_for_compare(r["_candidate_value_raw"], r["column"]), axis=1)

    if rank_col:
        df["_rank_for_sort"] = df[rank_col].map(lambda x: safe_int(x, 999999))
    else:
        df["_rank_for_sort"] = 999999

    if score_col:
        df["_score_for_sort"] = df[score_col].map(lambda x: safe_float(x, 0.0))
    else:
        df["_score_for_sort"] = 0.0

    if rank_col is None and score_col is not None:
        df = df.sort_values(["row_id", "column", "_score_for_sort"], ascending=[True, True, False])
        df["_rank_for_sort"] = df.groupby(["row_id", "column"]).cumcount() + 1

    if source_cols:
        df["_source_text"] = df[source_cols].astype(str).agg(" | ".join, axis=1)
    else:
        df["_source_text"] = ""

    src_low = df["_source_text"].astype(str).str.lower()
    df["_has_neighbor_source"] = src_low.str.contains("neighbor|similar", regex=True).astype(int)
    df["_has_rule_source"] = src_low.str.contains("rule|expected|fd|profile|distribution|context", regex=True).astype(int)
    df["_has_profile_source"] = src_low.str.contains("profile|team|season|stadium|city|distribution", regex=True).astype(int)
    df["_is_weak_global_source"] = (
        src_low.str.contains("column_top|global_column|dictionary|domain", regex=True)
        & ~src_low.str.contains("neighbor|similar|rule|expected|profile|distribution|context", regex=True)
    ).astype(int)

    return df


def build_candidate_cell_summary(cand_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (row_id, col), g in cand_df.groupby(["row_id", "column"], sort=False):
        gg = g.sort_values(["_rank_for_sort", "_score_for_sort"], ascending=[True, False]).copy()

        unique_norms = []
        unique_values = []
        seen = set()

        for _, r in gg.iterrows():
            norm = str(r["_candidate_norm"])
            val = display_value(r["_candidate_value_raw"])
            if norm == "":
                continue
            if norm not in seen:
                unique_norms.append(norm)
                unique_values.append(val)
                seen.add(norm)

        top = gg.iloc[0] if len(gg) else {}

        rows.append({
            "row_id": int(row_id),
            "column": str(col),
            "candidate_count": int(len(gg)),
            "unique_candidate_count": int(len(unique_norms)),
            "top1_candidate_value": display_value(top.get("_candidate_value_raw", "")) if len(gg) else "",
            "top1_candidate_norm": str(top.get("_candidate_norm", "")) if len(gg) else "",
            "top1_source_text": str(top.get("_source_text", "")) if len(gg) else "",
            "top1_has_neighbor_source": int(top.get("_has_neighbor_source", 0)) if len(gg) else 0,
            "top1_has_rule_source": int(top.get("_has_rule_source", 0)) if len(gg) else 0,
            "top1_has_profile_source": int(top.get("_has_profile_source", 0)) if len(gg) else 0,
            "top1_is_weak_global_source": int(top.get("_is_weak_global_source", 0)) if len(gg) else 0,
            "candidate_norms_joined": " || ".join(unique_norms[:80]),
            "candidate_values_joined": " || ".join(unique_values[:80]),
        })

    return pd.DataFrame(rows)


# ============================================================
# 6. 覆盖率分析
# ============================================================

def compute_oracle_details(true_df: pd.DataFrame, cand_df: pd.DataFrame, cell_summary_df: pd.DataFrame) -> pd.DataFrame:
    cand_map: Dict[Tuple[int, str], pd.DataFrame] = {}
    for key, g in cand_df.groupby(["row_id", "column"], sort=False):
        cand_map[(int(key[0]), str(key[1]))] = g.sort_values(
            ["_rank_for_sort", "_score_for_sort"], ascending=[True, False]
        ).reset_index(drop=True)

    summary_map = {
        (int(r["row_id"]), str(r["column"])): r.to_dict()
        for _, r in cell_summary_df.iterrows()
    }

    rows = []

    for _, r in true_df.iterrows():
        row_id = int(r["row_id"])
        col = str(r["column"]).strip()
        key = (row_id, col)

        clean_val = display_value(r.get("clean_value", ""))
        clean_norm = norm_for_compare(clean_val, col)

        g = cand_map.get(key)
        s = summary_map.get(key, {})

        clean_in_candidates = 0
        clean_rank = ""
        clean_candidate_value = ""
        clean_candidate_source = ""
        clean_candidate_score = ""

        if g is not None and len(g) > 0:
            for idx, cand in g.iterrows():
                cand_norm = str(cand["_candidate_norm"])
                if clean_norm != "" and cand_norm == clean_norm:
                    clean_in_candidates = 1
                    clean_rank = int(idx + 1)
                    clean_candidate_value = display_value(cand["_candidate_value_raw"])
                    clean_candidate_source = str(cand.get("_source_text", ""))
                    clean_candidate_score = safe_float(cand.get("_score_for_sort", 0.0), 0.0)
                    break

        clean_in_top1 = int(clean_in_candidates == 1 and clean_rank != "" and int(clean_rank) <= 1)
        clean_in_top3 = int(clean_in_candidates == 1 and clean_rank != "" and int(clean_rank) <= 3)
        clean_in_top5 = int(clean_in_candidates == 1 and clean_rank != "" and int(clean_rank) <= 5)

        eval_type = str(r.get("eval_type", ""))
        if clean_in_candidates == 0:
            failure_type = "candidate_missing"
        elif clean_in_top1 == 0:
            failure_type = "ranking_error"
        elif eval_type == "FN":
            failure_type = "gate_or_apply_error"
        else:
            failure_type = "already_fixed_or_not_fn"

        rows.append({
            "row_id": row_id,
            "column": col,
            "dirty_value": display_value(r.get("dirty_value", "")),
            "clean_value": clean_val,
            "repaired_value": display_value(r.get("repaired_value", "")),
            "eval_type": eval_type,
            "gate_reason": str(r.get("gate_reason", "")),
            "dirty_norm": norm_for_compare(r.get("dirty_value", ""), col),
            "clean_norm": clean_norm,
            "repaired_norm": norm_for_compare(r.get("repaired_value", ""), col),
            "candidate_count": int(s.get("candidate_count", 0) or 0),
            "unique_candidate_count": int(s.get("unique_candidate_count", 0) or 0),
            "top1_candidate_value": s.get("top1_candidate_value", ""),
            "top1_candidate_norm": s.get("top1_candidate_norm", ""),
            "top1_source_text": s.get("top1_source_text", ""),
            "top1_has_neighbor_source": int(s.get("top1_has_neighbor_source", 0) or 0),
            "top1_has_rule_source": int(s.get("top1_has_rule_source", 0) or 0),
            "top1_has_profile_source": int(s.get("top1_has_profile_source", 0) or 0),
            "top1_is_weak_global_source": int(s.get("top1_is_weak_global_source", 0) or 0),
            "clean_in_candidates": clean_in_candidates,
            "clean_rank": clean_rank,
            "clean_in_top1": clean_in_top1,
            "clean_in_top3": clean_in_top3,
            "clean_in_top5": clean_in_top5,
            "clean_candidate_value": clean_candidate_value,
            "clean_candidate_source": clean_candidate_source,
            "clean_candidate_score": clean_candidate_score,
            "failure_type": failure_type,
            "candidate_norms_joined": s.get("candidate_norms_joined", ""),
            "candidate_values_joined": s.get("candidate_values_joined", ""),
        })

    return pd.DataFrame(rows)


def build_summary_by_column(detail_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for col, g in detail_df.groupby("column", sort=False):
        n = len(g)
        tp = int((g["eval_type"] == "TP").sum()) if "eval_type" in g.columns else 0
        fn = int((g["eval_type"] == "FN").sum()) if "eval_type" in g.columns else 0

        clean_in = int(g["clean_in_candidates"].sum())
        top1 = int(g["clean_in_top1"].sum())
        top3 = int(g["clean_in_top3"].sum())
        top5 = int(g["clean_in_top5"].sum())

        fn_g = g[g["eval_type"] == "FN"] if "eval_type" in g.columns else g.iloc[0:0]

        rows.append({
            "column": col,
            "true_error_cells": n,
            "TP_current": tp,
            "FN_current": fn,
            "actual_recall_current": tp / n if n else 0.0,
            "clean_in_candidates": clean_in,
            "oracle_recall_if_choose_clean": clean_in / n if n else 0.0,
            "clean_in_top1": top1,
            "top1_oracle_recall": top1 / n if n else 0.0,
            "clean_in_top3": top3,
            "top3_oracle_recall": top3 / n if n else 0.0,
            "clean_in_top5": top5,
            "top5_oracle_recall": top5 / n if n else 0.0,
            "fn_candidate_missing": int((fn_g["failure_type"] == "candidate_missing").sum()),
            "fn_ranking_error": int((fn_g["failure_type"] == "ranking_error").sum()),
            "fn_gate_or_apply_error": int((fn_g["failure_type"] == "gate_or_apply_error").sum()),
            "avg_candidate_count": float(g["candidate_count"].mean()) if len(g) else 0.0,
            "avg_unique_candidate_count": float(g["unique_candidate_count"].mean()) if len(g) else 0.0,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["true_error_cells", "oracle_recall_if_choose_clean"], ascending=[False, False])
    return out.reset_index(drop=True)


def build_overall_summary(detail_df: pd.DataFrame) -> pd.DataFrame:
    n = len(detail_df)
    tp = int((detail_df["eval_type"] == "TP").sum()) if "eval_type" in detail_df.columns else 0
    fn = int((detail_df["eval_type"] == "FN").sum()) if "eval_type" in detail_df.columns else 0

    clean_in = int(detail_df["clean_in_candidates"].sum())
    top1 = int(detail_df["clean_in_top1"].sum())
    top3 = int(detail_df["clean_in_top3"].sum())
    top5 = int(detail_df["clean_in_top5"].sum())

    fn_g = detail_df[detail_df["eval_type"] == "FN"] if "eval_type" in detail_df.columns else detail_df.iloc[0:0]

    return pd.DataFrame([{
        "true_error_cells": n,
        "TP_current": tp,
        "FN_current": fn,
        "actual_recall_current": tp / n if n else 0.0,
        "clean_in_candidates": clean_in,
        "oracle_recall_if_choose_clean": clean_in / n if n else 0.0,
        "clean_in_top1": top1,
        "top1_oracle_recall": top1 / n if n else 0.0,
        "clean_in_top3": top3,
        "top3_oracle_recall": top3 / n if n else 0.0,
        "clean_in_top5": top5,
        "top5_oracle_recall": top5 / n if n else 0.0,
        "fn_candidate_missing": int((fn_g["failure_type"] == "candidate_missing").sum()),
        "fn_ranking_error": int((fn_g["failure_type"] == "ranking_error").sum()),
        "fn_gate_or_apply_error": int((fn_g["failure_type"] == "gate_or_apply_error").sum()),
    }])


# ============================================================
# 7. main
# ============================================================

def main() -> None:
    ensure_dir(OUTPUT_DIR)

    print("========== Soccer Candidate Coverage Analysis ==========")
    print(f"FULL_TABLE_EVAL_FILE: {FULL_TABLE_EVAL_FILE}")
    print(f"RANKED_CANDIDATES_FILE: {RANKED_CANDIDATES_FILE}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")

    print("\n[1/5] Loading true error details...")
    true_df = load_true_error_eval_df()
    print(f"[INFO] true error cells: {len(true_df)}")

    print("\n[2/5] Loading ranked candidates...")
    cand_df = load_ranked_candidates()
    print(f"[INFO] candidate rows: {len(cand_df)}")
    print(f"[INFO] candidate unique cells: {cand_df[['row_id', 'column']].drop_duplicates().shape[0]}")

    print("\n[3/5] Building candidate cell summary...")
    cell_summary_df = build_candidate_cell_summary(cand_df)
    cell_summary_df.to_csv(OUTPUT_CANDIDATE_CELL_SUMMARY, index=False, encoding="utf-8-sig")
    print(f"[OK] saved: {OUTPUT_CANDIDATE_CELL_SUMMARY}")

    print("\n[4/5] Computing oracle coverage...")
    detail_df = compute_oracle_details(true_df, cand_df, cell_summary_df)
    detail_df.to_csv(OUTPUT_TRUE_ERROR_DETAILS, index=False, encoding="utf-8-sig")

    fn_df = detail_df[detail_df["eval_type"] == "FN"].copy() if "eval_type" in detail_df.columns else detail_df.copy()
    fn_df.to_csv(OUTPUT_FN_DETAILS, index=False, encoding="utf-8-sig")

    detail_df[detail_df["failure_type"] == "candidate_missing"].to_csv(
        OUTPUT_MISSING_CANDIDATES, index=False, encoding="utf-8-sig"
    )
    detail_df[detail_df["failure_type"] == "ranking_error"].to_csv(
        OUTPUT_RANKING_ERRORS, index=False, encoding="utf-8-sig"
    )
    detail_df[detail_df["failure_type"] == "gate_or_apply_error"].to_csv(
        OUTPUT_GATE_ERRORS, index=False, encoding="utf-8-sig"
    )

    print(f"[OK] saved: {OUTPUT_TRUE_ERROR_DETAILS}")
    print(f"[OK] saved: {OUTPUT_FN_DETAILS}")

    print("\n[5/5] Building summaries...")
    overall_df = build_overall_summary(detail_df)
    col_df = build_summary_by_column(detail_df)

    overall_df.to_csv(OUTPUT_SUMMARY_OVERALL, index=False, encoding="utf-8-sig")
    col_df.to_csv(OUTPUT_SUMMARY_BY_COLUMN, index=False, encoding="utf-8-sig")

    print("\n========== Overall Summary ==========")
    row = overall_df.iloc[0].to_dict()
    for k, v in row.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")

    print("\n========== Column Summary Top 20 ==========")
    show_cols = [
        "column",
        "true_error_cells",
        "TP_current",
        "FN_current",
        "actual_recall_current",
        "clean_in_candidates",
        "oracle_recall_if_choose_clean",
        "clean_in_top1",
        "top1_oracle_recall",
        "clean_in_top3",
        "top3_oracle_recall",
        "fn_candidate_missing",
        "fn_ranking_error",
        "fn_gate_or_apply_error",
        "avg_candidate_count",
    ]
    print(col_df[show_cols].head(20).to_string(index=False))

    oracle = float(row.get("oracle_recall_if_choose_clean", 0.0))
    top1 = float(row.get("top1_oracle_recall", 0.0))
    actual = float(row.get("actual_recall_current", 0.0))

    print("\n========== Diagnosis ==========")
    if oracle < 0.8:
        print("[结论] clean 值进入候选列表的覆盖率低于 0.8，主要瓶颈是候选生成 revise_candidate.py。")
    elif top1 < 0.8:
        print("[结论] clean 值多数进入候选，但 top1 不足，主要瓶颈是特征构建/排序模型。")
    elif actual < top1 - 0.05:
        print("[结论] clean 经常已经是 top1，但最终没修，主要瓶颈是 infer_ltr.py 的 gate 或回填逻辑。")
    else:
        print("[结论] 当前修复召回接近候选排序上限，要继续提升需要扩充候选和优化排序。")

    print(f"\n[OK] overall summary: {OUTPUT_SUMMARY_OVERALL}")
    print(f"[OK] column summary: {OUTPUT_SUMMARY_BY_COLUMN}")
    print(f"[OK] FN details: {OUTPUT_FN_DETAILS}")


if __name__ == "__main__":
    main()

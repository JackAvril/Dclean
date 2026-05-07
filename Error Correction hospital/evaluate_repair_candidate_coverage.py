import pandas as pd
from collections import defaultdict
from pathlib import Path


# ============================================================
# 1. 配置区：直接改这里
# ============================================================
DIRTY_CSV = "/mnt/mydata/dq/projects/Splittree/hospital/hospital_dirty_no_address23.csv"
CLEAN_CSV = "/mnt/mydata/dq/projects/Splittree/hospital/hospital_clean_no_address23.csv"

# revise_candidate.py 输出的展开候选文件
CANDIDATE_EXPANDED_CSV = "repair_candidates_expanded_v2.csv"

OUTPUT_DIR = "repair_candidate_coverage_output"


# ============================================================
# 2. 工具函数
# ============================================================
EMPTY_TOKENS = {"", "empty", "null", "none", "nan", "na", "n/a", "missing"}


def norm_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def canonical_text(x):
    s = norm_text(x).lower()
    return "empty" if s in EMPTY_TOKENS else norm_text(x)


def safe_mkdir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. 主逻辑
# ============================================================
def main():
    safe_mkdir(OUTPUT_DIR)

    dirty_df = pd.read_csv(DIRTY_CSV, encoding="utf-8-sig")
    clean_df = pd.read_csv(CLEAN_CSV, encoding="utf-8-sig")
    cand_df = pd.read_csv(CANDIDATE_EXPANDED_CSV, encoding="utf-8-sig")

    dirty_df = dirty_df.apply(lambda col: col.map(canonical_text))
    clean_df = clean_df.apply(lambda col: col.map(canonical_text))

    if "row_id" not in cand_df.columns or "column" not in cand_df.columns or "candidate_value" not in cand_df.columns:
        raise ValueError("CANDIDATE_EXPANDED_CSV 必须至少包含 row_id, column, candidate_value")

    cand_df["row_id"] = cand_df["row_id"].astype(int)
    cand_df["column"] = cand_df["column"].astype(str)
    cand_df["candidate_value"] = cand_df["candidate_value"].map(canonical_text)

    candidate_pool = defaultdict(set)
    candidate_count_per_col = defaultdict(int)

    for _, r in cand_df.iterrows():
        key = (int(r["row_id"]), str(r["column"]))
        candidate_pool[key].add(canonical_text(r["candidate_value"]))
        candidate_count_per_col[str(r["column"])] += 1

    total_cells = int(dirty_df.shape[0] * dirty_df.shape[1])

    total_true_errors = 0
    total_covered = 0

    per_col_stat = defaultdict(lambda: {"true_errors": 0, "covered": 0})
    detail_rows = []

    for row_id in range(len(dirty_df)):
        for col in dirty_df.columns:
            dirty_val = dirty_df.iloc[row_id][col]
            clean_val = clean_df.iloc[row_id][col]

            if dirty_val == clean_val:
                continue

            total_true_errors += 1
            pool = candidate_pool.get((row_id, str(col)), set())
            covered = int(clean_val in pool)
            total_covered += covered

            per_col_stat[str(col)]["true_errors"] += 1
            per_col_stat[str(col)]["covered"] += covered

            preview = list(sorted(pool))[:15]
            detail_rows.append({
                "row_id": row_id,
                "column": str(col),
                "dirty_value": dirty_val,
                "clean_value": clean_val,
                "candidate_count": len(pool),
                "is_covered": covered,
                "candidates_preview": " || ".join(preview),
            })

    summary = {
        "total_cells": total_cells,
        "true_error_cells": total_true_errors,
        "covered_error_cells": total_covered,
        "coverage": (total_covered / total_true_errors) if total_true_errors else 0.0,
        "candidate_rows": len(cand_df),
        "unique_candidate_cells": len(candidate_pool),
    }

    summary_df = pd.DataFrame([summary])

    per_col_rows = []
    for col, stat in sorted(per_col_stat.items(), key=lambda x: (-x[1]["true_errors"], x[0])):
        per_col_rows.append({
            "column": col,
            "true_errors": stat["true_errors"],
            "covered": stat["covered"],
            "coverage": stat["covered"] / stat["true_errors"] if stat["true_errors"] else 0.0,
            "candidate_rows": candidate_count_per_col.get(col, 0),
        })

    detail_df = pd.DataFrame(detail_rows)
    uncovered_df = detail_df[detail_df["is_covered"] == 0].copy()
    covered_df = detail_df[detail_df["is_covered"] == 1].copy()

    summary_path = Path(OUTPUT_DIR) / "repair_candidate_coverage_summary.csv"
    per_col_path = Path(OUTPUT_DIR) / "repair_candidate_coverage_by_column.csv"
    detail_path = Path(OUTPUT_DIR) / "repair_candidate_coverage_details.csv"
    uncovered_path = Path(OUTPUT_DIR) / "repair_candidate_uncovered_details.csv"
    covered_path = Path(OUTPUT_DIR) / "repair_candidate_covered_details.csv"

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(per_col_rows).to_csv(per_col_path, index=False, encoding="utf-8-sig")
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    uncovered_df.to_csv(uncovered_path, index=False, encoding="utf-8-sig")
    covered_df.to_csv(covered_path, index=False, encoding="utf-8-sig")

    print("========== Repair Candidate Coverage ==========")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"[OK] summary:   {summary_path}")
    print(f"[OK] by_column: {per_col_path}")
    print(f"[OK] details:   {detail_path}")
    print(f"[OK] uncovered: {uncovered_path}")


if __name__ == "__main__":
    main()

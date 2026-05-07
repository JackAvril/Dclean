
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


# ============================================================
# 1. 配置区：直接改这里，不用命令行传参
# ============================================================
CANDIDATE_FEATURES = "repair_candidate_features.csv"
CLEAN_CSV = "/mnt/mydata/dq/projects/Splittree/hospital/hospital_clean_no_address23.csv"
DIRTY_CSV = "/mnt/mydata/dq/projects/Splittree/hospital/hospital_dirty_no_address23.csv"
OUTPUT_DIR = "coverage_analysis_output"


EMPTY_TOKENS = {"", "empty", "null", "none", "nan", "na", "n/a", "missing"}


def parse_args():
    class Args:
        candidate_features = CANDIDATE_FEATURES
        clean_csv = CLEAN_CSV
        dirty_csv = DIRTY_CSV
        output_dir = OUTPUT_DIR
    return Args()


def norm_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def canonical_empty_text(x):
    s = norm_text(x).lower()
    return "empty" if s in EMPTY_TOKENS else norm_text(x)


def looks_like_code(x: str) -> int:
    s = norm_text(x)
    if s == "":
        return 0
    return int((bool(re.search(r"[_\-]", s)) or bool(re.search(r"\d", s))) and len(s) <= 32)


def is_digits_only_text(x: str) -> int:
    return int(bool(re.fullmatch(r"\d+", norm_text(x))))


def is_percent_like_text(x: str) -> int:
    return int(bool(re.fullmatch(r"\d{1,3}%", norm_text(x))))


def is_alpha_space_text(x: str) -> int:
    return int(bool(re.fullmatch(r"[A-Za-z\s\-\./']+", norm_text(x))))


def infer_column_profiles(candidate_df: pd.DataFrame):
    profiles = {}
    for col, g in candidate_df.groupby("column"):
        vals = pd.concat([g["dirty_value"], g["candidate_value"]], ignore_index=True).dropna().astype(str)
        vals = vals.map(canonical_empty_text)
        vals = vals[vals != "empty"]

        if len(vals) == 0:
            profiles[str(col)] = {
                "archetype": "mixed",
                "avg_len": 0.0,
                "digits_ratio": 0.0,
                "percent_ratio": 0.0,
                "code_ratio": 0.0,
                "alpha_ratio": 0.0,
                "unique_count": 0,
            }
            continue

        avg_len = float(vals.map(lambda x: len(norm_text(x))).mean())
        digits_ratio = float(vals.map(is_digits_only_text).mean())
        percent_ratio = float(vals.map(is_percent_like_text).mean())
        code_ratio = float(vals.map(looks_like_code).mean())
        alpha_ratio = float(vals.map(is_alpha_space_text).mean())
        unique_count = int(vals.nunique())

        archetype = "mixed"
        if percent_ratio >= 0.55:
            archetype = "percent_pattern"
        elif digits_ratio >= 0.75 and avg_len >= 6:
            archetype = "long_digit_id"
        elif digits_ratio >= 0.75:
            archetype = "short_digit_id"
        elif code_ratio >= 0.55:
            archetype = "structured_code"
        elif unique_count <= 8 and alpha_ratio >= 0.8:
            archetype = "small_enum"
        elif alpha_ratio >= 0.8 and avg_len >= 30:
            archetype = "long_text"
        elif alpha_ratio >= 0.8:
            archetype = "short_text"

        profiles[str(col)] = {
            "archetype": archetype,
            "avg_len": avg_len,
            "digits_ratio": digits_ratio,
            "percent_ratio": percent_ratio,
            "code_ratio": code_ratio,
            "alpha_ratio": alpha_ratio,
            "unique_count": unique_count,
        }
    return profiles


def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    cand = pd.read_csv(args.candidate_features, encoding="utf-8-sig")
    clean = pd.read_csv(args.clean_csv, encoding="utf-8-sig")
    dirty = pd.read_csv(args.dirty_csv, encoding="utf-8-sig")

    cand["row_id"] = cand["row_id"].astype(int)
    cand["column"] = cand["column"].astype(str)
    cand["dirty_value"] = cand["dirty_value"].map(canonical_empty_text)
    cand["candidate_value"] = cand["candidate_value"].map(canonical_empty_text)

    clean = pd.read_csv(args.clean_csv, encoding="utf-8-sig")
    dirty = pd.read_csv(args.dirty_csv, encoding="utf-8-sig")
    clean = clean.apply(lambda col: col.map(canonical_empty_text))
    dirty = dirty.apply(lambda col: col.map(canonical_empty_text))
    profiles = infer_column_profiles(cand)
    (outdir / "column_profiles.json").write_text(json.dumps(profiles, ensure_ascii=False, indent=2), encoding="utf-8")

    cand_pool = defaultdict(set)
    cand_count = defaultdict(int)
    for _, row in cand.iterrows():
        key = (int(row["row_id"]), str(row["column"]))
        cand_pool[key].add(canonical_empty_text(row["candidate_value"]))
        cand_count[str(row["column"])] += 1

    detail_rows = []
    per_col = defaultdict(lambda: {"errors": 0, "covered": 0})
    per_arch = defaultdict(lambda: {"errors": 0, "covered": 0})

    total_errors = 0
    total_covered = 0

    for row_id in range(len(clean)):
        for col in clean.columns:
            clean_val = canonical_empty_text(clean.iloc[row_id][col])
            dirty_val = canonical_empty_text(dirty.iloc[row_id][col])
            if clean_val == dirty_val:
                continue

            total_errors += 1
            arch = profiles.get(str(col), {}).get("archetype", "mixed")
            pool = cand_pool.get((row_id, str(col)), set())
            covered = int(clean_val in pool)
            total_covered += covered
            per_col[str(col)]["errors"] += 1
            per_col[str(col)]["covered"] += covered
            per_arch[str(arch)]["errors"] += 1
            per_arch[str(arch)]["covered"] += covered

            detail_rows.append({
                "row_id": row_id,
                "column": str(col),
                "archetype": arch,
                "dirty_value": dirty_val,
                "clean_value": clean_val,
                "candidate_count": len(pool),
                "is_covered": covered,
                "candidates_preview": " || ".join(list(sorted(pool))[:10]),
            })

    summary = {
        "true_error_cells": total_errors,
        "covered_error_cells": total_covered,
        "coverage": (total_covered / total_errors) if total_errors > 0 else 0.0,
        "columns_with_candidates": len(cand["column"].unique()),
        "total_candidate_rows": len(cand),
    }

    per_col_rows = []
    for col, stat in sorted(per_col.items(), key=lambda x: (-x[1]["errors"], x[0])):
        per_col_rows.append({
            "column": col,
            "archetype": profiles.get(col, {}).get("archetype", "mixed"),
            "errors": stat["errors"],
            "covered": stat["covered"],
            "coverage": stat["covered"] / stat["errors"] if stat["errors"] else 0.0,
            "candidate_rows": cand_count.get(col, 0),
        })

    per_arch_rows = []
    for arch, stat in sorted(per_arch.items(), key=lambda x: (-x[1]["errors"], x[0])):
        per_arch_rows.append({
            "archetype": arch,
            "errors": stat["errors"],
            "covered": stat["covered"],
            "coverage": stat["covered"] / stat["errors"] if stat["errors"] else 0.0,
        })

    pd.DataFrame([summary]).to_csv(outdir / "coverage_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(per_col_rows).to_csv(outdir / "coverage_by_column.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(per_arch_rows).to_csv(outdir / "coverage_by_archetype.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(detail_rows).to_csv(outdir / "coverage_details.csv", index=False, encoding="utf-8-sig")

    print("========== Candidate Coverage Analysis ==========")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"[OK] Output dir: {outdir}")


if __name__ == "__main__":
    main()

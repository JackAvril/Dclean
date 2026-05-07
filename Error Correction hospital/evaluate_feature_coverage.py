import pandas as pd
from collections import defaultdict
from pathlib import Path


# ============================================================
# 1. 配置区：直接改这里
# ============================================================
DIRTY_CSV = "/mnt/mydata/dq/projects/Splittree/hospital/hospital_dirty_no_address23.csv"
CLEAN_CSV = "/mnt/mydata/dq/projects/Splittree/hospital/hospital_clean_no_address23.csv"

# revise_candidate_v2.py 生成的候选展开文件
CANDIDATE_EXPANDED_CSV = "repair_candidates_expanded_v2.csv"

# build_features_v2.py 生成的特征文件
FEATURES_CSV = "repair_candidate_features_v2.csv"

OUTPUT_DIR = "feature_coverage_output"


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
    feat_df = pd.read_csv(FEATURES_CSV, encoding="utf-8-sig")

    dirty_df = dirty_df.apply(lambda col: col.map(canonical_text))
    clean_df = clean_df.apply(lambda col: col.map(canonical_text))

    required = {"row_id", "column", "candidate_value"}
    if not required.issubset(set(cand_df.columns)):
        raise ValueError(f"CANDIDATE_EXPANDED_CSV 必须至少包含 {required}")
    if not required.issubset(set(feat_df.columns)):
        raise ValueError(f"FEATURES_CSV 必须至少包含 {required}")

    cand_df["row_id"] = cand_df["row_id"].astype(int)
    cand_df["column"] = cand_df["column"].astype(str)
    cand_df["candidate_value"] = cand_df["candidate_value"].map(canonical_text)

    feat_df["row_id"] = feat_df["row_id"].astype(int)
    feat_df["column"] = feat_df["column"].astype(str)
    feat_df["candidate_value"] = feat_df["candidate_value"].map(canonical_text)

    cand_keys = set((int(r["row_id"]), str(r["column"]), canonical_text(r["candidate_value"])) for _, r in cand_df.iterrows())
    feat_keys = set((int(r["row_id"]), str(r["column"]), canonical_text(r["candidate_value"])) for _, r in feat_df.iterrows())

    missing_keys = cand_keys - feat_keys
    extra_keys = feat_keys - cand_keys

    cand_col_counter = defaultdict(int)
    feat_col_counter = defaultdict(int)
    miss_col_counter = defaultdict(int)

    for _, r in cand_df.iterrows():
        cand_col_counter[str(r["column"])] += 1
    for _, r in feat_df.iterrows():
        feat_col_counter[str(r["column"])] += 1
    for _, col, _ in missing_keys:
        miss_col_counter[str(col)] += 1

    truth_cand_keys = set()
    truth_feat_keys = set()
    truth_detail_rows = []
    per_col_truth = defaultdict(lambda: {"true_errors": 0, "truth_in_candidate": 0, "truth_in_feature": 0})

    total_true_errors = 0
    for row_id in range(len(dirty_df)):
        for col in dirty_df.columns:
            dirty_val = dirty_df.iloc[row_id][col]
            clean_val = clean_df.iloc[row_id][col]

            if dirty_val == clean_val:
                continue

            total_true_errors += 1
            per_col_truth[str(col)]["true_errors"] += 1

            truth_key = (row_id, str(col), canonical_text(clean_val))
            in_candidate = int(truth_key in cand_keys)
            in_feature = int(truth_key in feat_keys)

            if in_candidate:
                truth_cand_keys.add(truth_key)
                per_col_truth[str(col)]["truth_in_candidate"] += 1
            if in_feature:
                truth_feat_keys.add(truth_key)
                per_col_truth[str(col)]["truth_in_feature"] += 1

            truth_detail_rows.append({
                "row_id": row_id,
                "column": str(col),
                "dirty_value": dirty_val,
                "clean_value": clean_val,
                "truth_in_candidate_file": in_candidate,
                "truth_in_feature_file": in_feature,
                "lost_at_feature_stage": int(in_candidate == 1 and in_feature == 0),
            })

    summary = {
        "candidate_rows": len(cand_df),
        "feature_rows": len(feat_df),
        "candidate_unique_keys": len(cand_keys),
        "feature_unique_keys": len(feat_keys),
        "feature_retention_rate_by_unique_key": (len(feat_keys) / len(cand_keys)) if cand_keys else 0.0,
        "missing_feature_keys_from_candidate": len(missing_keys),
        "extra_feature_keys_not_in_candidate": len(extra_keys),
        "true_error_cells": total_true_errors,
        "truth_candidates_in_candidate_file": len(truth_cand_keys),
        "truth_candidates_in_feature_file": len(truth_feat_keys),
        "truth_feature_retention_rate": (len(truth_feat_keys) / len(truth_cand_keys)) if truth_cand_keys else 0.0,
        "truth_candidates_lost_at_feature_stage": len(truth_cand_keys - truth_feat_keys),
    }

    summary_df = pd.DataFrame([summary])

    per_col_rows = []
    all_cols = sorted(set(list(cand_col_counter.keys()) + list(feat_col_counter.keys()) + list(per_col_truth.keys())))
    for col in all_cols:
        cand_n = cand_col_counter.get(col, 0)
        feat_n = feat_col_counter.get(col, 0)
        miss_n = miss_col_counter.get(col, 0)
        stat = per_col_truth.get(col, {"true_errors": 0, "truth_in_candidate": 0, "truth_in_feature": 0})

        per_col_rows.append({
            "column": col,
            "candidate_rows": cand_n,
            "feature_rows": feat_n,
            "feature_retention_rate": (feat_n / cand_n) if cand_n else 0.0,
            "missing_feature_keys": miss_n,
            "true_errors": stat["true_errors"],
            "truth_in_candidate": stat["truth_in_candidate"],
            "truth_in_feature": stat["truth_in_feature"],
            "truth_feature_retention_rate": (
                stat["truth_in_feature"] / stat["truth_in_candidate"]
                if stat["truth_in_candidate"] else 0.0
            ),
        })

    missing_rows = []
    if missing_keys:
        cand_lookup = {}
        for _, r in cand_df.iterrows():
            key = (int(r["row_id"]), str(r["column"]), canonical_text(r["candidate_value"]))
            if key not in cand_lookup:
                cand_lookup[key] = r.to_dict()
        for key in sorted(missing_keys):
            r = cand_lookup.get(key, {})
            missing_rows.append({
                "row_id": key[0],
                "column": key[1],
                "candidate_value": key[2],
                "dirty_value": canonical_text(r.get("dirty_value", "")),
                "candidate_source": r.get("candidate_source", ""),
                "candidate_rank": r.get("candidate_rank", None),
                "source_score": r.get("source_score", None),
                "final_score": r.get("final_score", None),
            })

    truth_detail_df = pd.DataFrame(truth_detail_rows)
    truth_lost_df = truth_detail_df[truth_detail_df["lost_at_feature_stage"] == 1].copy()

    summary_path = Path(OUTPUT_DIR) / "feature_coverage_summary.csv"
    per_col_path = Path(OUTPUT_DIR) / "feature_coverage_by_column.csv"
    missing_path = Path(OUTPUT_DIR) / "missing_feature_keys_details.csv"
    truth_detail_path = Path(OUTPUT_DIR) / "truth_candidate_feature_retention_details.csv"
    truth_lost_path = Path(OUTPUT_DIR) / "truth_candidates_lost_at_feature_stage.csv"

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(per_col_rows).to_csv(per_col_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(missing_rows).to_csv(missing_path, index=False, encoding="utf-8-sig")
    truth_detail_df.to_csv(truth_detail_path, index=False, encoding="utf-8-sig")
    truth_lost_df.to_csv(truth_lost_path, index=False, encoding="utf-8-sig")

    print("========== Feature Coverage Check ==========")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"[OK] summary:    {summary_path}")
    print(f"[OK] by_column:  {per_col_path}")
    print(f"[OK] missing:    {missing_path}")
    print(f"[OK] truth_keep: {truth_detail_path}")
    print(f"[OK] truth_lost: {truth_lost_path}")


if __name__ == "__main__":
    main()

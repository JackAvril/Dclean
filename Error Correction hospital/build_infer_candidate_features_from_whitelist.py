import pandas as pd
from pathlib import Path

# ============================================================
# 1. 直接在这里改文件路径
# ============================================================
INPUT_CANDIDATE_FEATURES = "repair_candidate_features_v2.csv"
INPUT_ALLOWED_CELLS = "allowed_cells_from_detection.csv"
OUTPUT_INFER_FEATURES = "repair_candidate_features_infer_whitelist.csv"


# ============================================================
# 2. 主逻辑
# ============================================================
def main():
    cand = pd.read_csv(INPUT_CANDIDATE_FEATURES, encoding="utf-8-sig")
    allow = pd.read_csv(INPUT_ALLOWED_CELLS, encoding="utf-8-sig")

    required_cand = {"row_id", "column"}
    required_allow = {"row_id", "column"}

    if not required_cand.issubset(set(cand.columns)):
        raise ValueError(f"candidate features file must contain columns: {required_cand}")
    if not required_allow.issubset(set(allow.columns)):
        raise ValueError(f"allowed cells file must contain columns: {required_allow}")

    cand["row_id"] = pd.to_numeric(cand["row_id"], errors="raise").astype(int)
    cand["column"] = cand["column"].astype(str).str.strip()

    allow["row_id"] = pd.to_numeric(allow["row_id"], errors="raise").astype(int)
    allow["column"] = allow["column"].astype(str).str.strip()

    allow_unique = allow[["row_id", "column"]].drop_duplicates().copy()
    cand_unique_before = cand[["row_id", "column"]].drop_duplicates().shape[0]

    merged = cand.merge(
        allow_unique.assign(__keep__=1),
        on=["row_id", "column"],
        how="inner"
    ).drop(columns=["__keep__"])

    cand_unique_after = merged[["row_id", "column"]].drop_duplicates().shape[0]

    output_path = Path(OUTPUT_INFER_FEATURES)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")

    missing_allow = allow_unique.merge(
        merged[["row_id", "column"]].drop_duplicates(),
        on=["row_id", "column"],
        how="left",
        indicator=True
    )
    missing_allow = missing_allow[missing_allow["_merge"] == "left_only"].drop(columns=["_merge"])

    if len(missing_allow) > 0:
        missing_path = output_path.with_name(output_path.stem + "_missing_allowed_cells.csv")
        missing_allow.to_csv(missing_path, index=False, encoding="utf-8-sig")
        print(f"[WARN] Some whitelist cells are not present in candidate features. Saved to: {missing_path}")

    print("========== Build infer candidate features from whitelist ==========")
    print(f"input_candidate_features: {INPUT_CANDIDATE_FEATURES}")
    print(f"input_allowed_cells: {INPUT_ALLOWED_CELLS}")
    print(f"output_infer_features: {OUTPUT_INFER_FEATURES}")
    print(f"candidate_unique_cells_before: {cand_unique_before}")
    print(f"allowed_unique_cells: {len(allow_unique)}")
    print(f"candidate_unique_cells_after: {cand_unique_after}")
    print(f"output_rows: {len(merged)}")
    print(f"[OK] saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()

import os
import json
import re
import argparse
from typing import List, Dict, Optional

import pandas as pd
import torch
import torch.nn as nn


def parse_args():
    p = argparse.ArgumentParser(description="Inference + hard case mining")
    p.add_argument("--candidate_features", default="repair_candidate_features.csv")
    p.add_argument("--model_dir", default="pairwise_ltr_student_output_iter_f1_plus")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--global_min_margin", type=float, default=0.05)
    p.add_argument("--max_hard_cases", type=int, default=300)
    return p.parse_args()


ARGS = parse_args()
MODEL_DIR = ARGS.model_dir
CHECKPOINT_PATH = ARGS.checkpoint or os.path.join(MODEL_DIR, "best_model.pt")
FEATURE_COLUMNS_JSON = os.path.join(MODEL_DIR, "feature_columns.json")
REASON_LABELS_JSON = os.path.join(MODEL_DIR, "reason_label_mapping.json")

OUTPUT_ALL_SCORES_CSV = os.path.join(MODEL_DIR, "inference_all_candidate_scores.csv")
OUTPUT_ALL_RANKED_CSV = os.path.join(MODEL_DIR, "inference_all_candidate_ranked.csv")
OUTPUT_TOP1_CSV = os.path.join(MODEL_DIR, "inference_top1_repairs.csv")
OUTPUT_HARD_CASES_JSONL = os.path.join(MODEL_DIR, "hard_cases_for_llm.jsonl")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GLOBAL_MIN_MARGIN = ARGS.global_min_margin

# 注意：这里不再使用绝对 score 门控，只保留 margin 和结构/业务门控
COLUMN_GATE_CONFIG: Dict[str, Dict[str, float]] = {
    # 高风险列：继续保守，但只小幅放开
    "Score": {"min_margin": 0.035, "max_empty_to_nonempty": 0.0},
    "Sample": {"min_margin": 0.03, "max_empty_to_nonempty": 0.0},
    "Stateavg": {"min_margin": 0.055, "require_family_match": 1.0},
    "MeasureCode": {"min_margin": 0.055},
    # 长文本 / 小域 typo 列：适当再放一点
    "MeasureName": {"min_margin": 0.014},
    "Condition": {"min_margin": 0.012},
    "PhoneNumber": {"min_margin": 0.003},
    "ZipCode": {"min_margin": 0.003},
    "ProviderNumber": {"min_margin": 0.003},
    "State": {"min_margin": 0.003},
    "EmergencyService": {"min_margin": 0.003},
    "HospitalType": {"min_margin": 0.005},
    "HospitalOwner": {"min_margin": 0.005},
    "CountyName": {"min_margin": 0.008},
    "City": {"min_margin": 0.008},
    "Address1": {"min_margin": 0.008},
    "HospitalName": {"min_margin": 0.008},
}


def norm_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


EMPTY_TOKENS = {"", "empty", "null", "none", "nan", "na", "n/a", "missing"}


def canonical_empty_text(x):
    s = norm_text(x).lower()
    return "empty" if s in EMPTY_TOKENS else norm_text(x)


def load_json(path: str):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_jsonl_record(fp, obj):
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_checkpoint(path: str):
    return torch.load(path, map_location=DEVICE)


def is_empty_token(x: str) -> int:
    s = norm_text(x).lower()
    return int(s in EMPTY_TOKENS)


def looks_like_code(x: str) -> int:
    s = norm_text(x)
    if s == "":
        return 0
    cond1 = bool(re.search(r"[_\-]", s))
    cond2 = bool(re.search(r"\d", s))
    cond3 = len(s) <= 24
    return int((cond1 or cond2) and cond3)


def prefix_token(x: str) -> str:
    s = norm_text(x)
    if s == "":
        return ""
    parts = re.split(r"[_\-]", s)
    return parts[0] if parts else s


def family_token(x: str) -> str:
    s = norm_text(x)
    if s == "":
        return ""
    parts = re.split(r"[_\-]", s)
    if len(parts) >= 2:
        return parts[1]
    return parts[0]


US_STATE_CODES = {
    "al","ak","az","ar","ca","co","ct","de","fl","ga","hi","ia","id","il","in","ks","ky","la","ma","md","me",
    "mi","mn","mo","ms","mt","nc","nd","ne","nh","nj","nm","nv","ny","oh","ok","or","pa","ri","sc","sd","tn",
    "tx","ut","va","vt","wa","wi","wv","wy","dc"
}
YES_NO_DOMAIN = {"yes", "no"}
HOSPITAL_TYPE_DOMAIN = {
    "acute care hospitals",
    "critical access hospitals",
    "acute care hospital",
    "critical access hospital",
}
CONDITION_DOMAIN = {
    "heart attack",
    "heart failure",
    "pneumonia",
    "surgical infection prevention",
}


def is_digits_only(s: str) -> bool:
    return bool(re.fullmatch(r"\d+", norm_text(s)))


def is_percent_like(s: str) -> bool:
    return bool(re.fullmatch(r"\d{1,3}%", norm_text(s)))


def alpha_space_like(s: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z\s\-\./']+", norm_text(s)))


def tiny_edit_like(a: str, b: str, max_diff: int) -> bool:
    """
    很轻量的字符差异计数，不做完整编辑距离，只用于小范围放宽。
    """
    a = norm_text(a).lower()
    b = norm_text(b).lower()
    if abs(len(a) - len(b)) > max_diff:
        return False
    # 允许长度差，通过补齐比较
    la, lb = len(a), len(b)
    if la < lb:
        a = a + "#" * (lb - la)
    elif lb < la:
        b = b + "#" * (la - lb)
    diff = sum(ch1 != ch2 for ch1, ch2 in zip(a, b))
    return diff <= max_diff


def likely_measure_code_candidate(cand: str) -> bool:
    s = norm_text(cand).lower()
    return (
        s.startswith(("ami", "hf", "pn", "scip")) or
        ("ami" in s) or ("hf" in s) or ("pn" in s) or ("scip" in s)
    )


def likely_stateavg_candidate(cand: str) -> bool:
    s = norm_text(cand).lower()
    return "_" in s and likely_measure_code_candidate(s)


def pattern_override_pass(best_row: pd.Series, second_row: Optional[pd.Series]) -> tuple[bool, str]:
    col = str(best_row["column"])
    dirty = canonical_empty_text(best_row["dirty_value"])
    cand = canonical_empty_text(best_row["candidate_value"])
    second_score = float(second_row.get("student_score", 0.0)) if second_row is not None else 0.0
    margin = float(best_row.get("student_score", 0.0)) - second_score

    # 稳定 pattern / 小域列，允许更积极一点
    if col == "PhoneNumber" and is_digits_only(cand) and len(cand) >= 7 and cand != dirty and margin >= -0.005:
        return True, "pattern_phone_override"
    if col == "ZipCode" and is_digits_only(cand) and len(cand) in {5, 9} and cand != dirty and margin >= -0.005:
        return True, "pattern_zip_override"
    if col == "ProviderNumber" and is_digits_only(cand) and len(cand) >= 4 and cand != dirty and margin >= -0.005:
        return True, "pattern_provider_override"
    if col == "State" and cand.lower() in US_STATE_CODES and cand.lower() != dirty.lower() and margin >= -0.005:
        return True, "pattern_state_override"
    if col == "EmergencyService" and cand.lower() in YES_NO_DOMAIN and cand.lower() != dirty.lower() and margin >= -0.005:
        return True, "pattern_yesno_override"
    if col == "HospitalType" and cand.lower() in HOSPITAL_TYPE_DOMAIN and cand.lower() != dirty.lower() and margin >= -0.005:
        return True, "pattern_hospital_type_override"
    if col == "Condition" and cand.lower() in CONDITION_DOMAIN and cand.lower() != dirty.lower() and margin >= -0.005:
        return True, "pattern_condition_override"

    # 非空 Score / Sample typo：更积极一点
    if col == "Score" and dirty != "empty" and is_percent_like(cand) and cand != dirty and margin >= -0.002:
        return True, "pattern_score_nonempty_override"
    if col == "Sample" and dirty != "empty" and ("patients" in cand.lower()) and cand != dirty and margin >= -0.002:
        return True, "pattern_sample_nonempty_override"

    # 名称/地址类：只在 top1 候选看起来像正常文本，且与 dirty 很接近时放一点
    if col in {"CountyName", "City", "Address1", "HospitalName"}:
        if alpha_space_like(cand) and cand != dirty and tiny_edit_like(dirty, cand, 3) and margin >= -0.003:
            return True, f"pattern_{col.lower()}_override"

    # code-like：只做非常轻量的恢复，不彻底放开
    if col == "MeasureCode":
        if likely_measure_code_candidate(cand) and cand != dirty and tiny_edit_like(dirty, cand, 3) and margin >= -0.003:
            return True, "pattern_measurecode_override"

    if col == "Stateavg":
        if likely_stateavg_candidate(cand) and cand != dirty and tiny_edit_like(dirty, cand, 4) and margin >= -0.003:
            return True, "pattern_stateavg_override"

    return False, ""


class SharedScorerMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float, num_reason_classes: int):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.score_head = nn.Linear(prev, 1)
        self.reason_head = nn.Linear(prev, num_reason_classes)

    def forward_single(self, x):
        h = self.backbone(x)
        return self.score_head(h).squeeze(-1), self.reason_head(h), h


NON_FEATURE_COLUMNS = {
    "row_id", "column", "dirty_value", "candidate_value",
    "candidate_rank", "candidate_source", "extra_info",
    "neighbor_majority_value", "suggested_correct_value",
    "main_rule_type", "main_usage_role", "semantic_type"
}


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dirty_value"] = out["dirty_value"].map(canonical_empty_text)
    out["candidate_value"] = out["candidate_value"].map(canonical_empty_text)
    out["original_is_empty"] = out["dirty_value"].map(is_empty_token)
    out["candidate_is_empty"] = out["candidate_value"].map(is_empty_token)
    out["candidate_equals_dirty"] = (
        out["candidate_value"].map(norm_text) == out["dirty_value"].map(norm_text)
    ).astype(int)
    out["dirty_is_code_like"] = out["dirty_value"].map(looks_like_code)
    out["candidate_is_code_like"] = out["candidate_value"].map(looks_like_code)

    dirty_prefix = out["dirty_value"].map(prefix_token)
    cand_prefix = out["candidate_value"].map(prefix_token)
    dirty_family = out["dirty_value"].map(family_token)
    cand_family = out["candidate_value"].map(family_token)

    out["prefix_match"] = (dirty_prefix == cand_prefix).astype(int)
    out["family_match"] = (dirty_family == cand_family).astype(int)
    out["empty_to_nonempty"] = (
        (out["original_is_empty"] == 1) & (out["candidate_is_empty"] == 0)
    ).astype(int)
    out["candidate_is_digits_only"] = out["candidate_value"].map(lambda x: int(bool(re.fullmatch(r"\d+", norm_text(x)))))
    out["candidate_is_percent_like"] = out["candidate_value"].map(lambda x: int(bool(re.fullmatch(r"\d{1,3}%", norm_text(x)))))

    for col_name in ["Score", "Sample", "Stateavg", "MeasureCode", "MeasureName", "Condition"]:
        out[f"is_col_{col_name}"] = (out["column"] == col_name).astype(int)
    return out


def load_candidate_features_for_inference(path: str, trained_feature_columns: List[str]) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["row_id"] = df["row_id"].astype(int)
    df["column"] = df["column"].astype(str)
    df["candidate_value"] = df["candidate_value"].map(canonical_empty_text)
    df["dirty_value"] = df["dirty_value"].map(canonical_empty_text)
    df = add_derived_features(df)

    categorical_cols = [
        c for c in ["semantic_type", "main_rule_type", "main_usage_role", "candidate_source"]
        if c in df.columns
    ]
    df_cat = (
        pd.get_dummies(df[categorical_cols].fillna("unknown").astype(str), prefix=categorical_cols)
        if categorical_cols else pd.DataFrame(index=df.index)
    )

    numeric_feature_df = pd.DataFrame(index=df.index)
    for col in df.columns:
        if col in NON_FEATURE_COLUMNS or col in categorical_cols:
            continue
        numeric_feature_df[col] = pd.to_numeric(df[col], errors="coerce")
    numeric_feature_df = numeric_feature_df.fillna(0.0)

    feat_df = pd.concat([numeric_feature_df, df_cat], axis=1)
    for col in feat_df.columns:
        feat_df[col] = pd.to_numeric(feat_df[col], errors="coerce").fillna(0.0)

    for col in trained_feature_columns:
        if col not in feat_df.columns:
            feat_df[col] = 0.0

    feat_df = feat_df[trained_feature_columns].copy()
    for col in trained_feature_columns:
        df[col] = feat_df[col].values
    return df


@torch.no_grad()
def score_all_candidates(model, df: pd.DataFrame, feature_columns: List[str], reason_id_to_name: dict) -> pd.DataFrame:
    model.eval()
    feats = torch.tensor(df[feature_columns].astype(float).values, dtype=torch.float32, device=DEVICE)
    score, reason_logits, _ = model.forward_single(feats)
    reason_pred = reason_logits.argmax(dim=1).detach().cpu().numpy()

    out = df.copy()
    out["student_score"] = score.detach().cpu().numpy()
    out["student_reason_pred_id"] = reason_pred
    out["student_reason_pred"] = [reason_id_to_name.get(int(x), "unknown") for x in reason_pred]
    return out


def rank_candidates(scored_df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = ["row_id", "column", "student_score"]
    ascending = [True, True, False]
    if "candidate_rank" in scored_df.columns:
        sort_cols.append("candidate_rank")
        ascending.append(True)
    ranked = scored_df.sort_values(sort_cols, ascending=ascending).copy()
    ranked["student_rank"] = ranked.groupby(["row_id", "column"]).cumcount() + 1
    return ranked


def pass_column_gate(best_row: pd.Series, second_row: Optional[pd.Series]) -> tuple[bool, str]:
    col = str(best_row["column"])
    cfg = COLUMN_GATE_CONFIG.get(col, {})

    # 先做业务/结构硬约束
    if "max_empty_to_nonempty" in cfg and float(best_row.get("empty_to_nonempty", 0.0)) > float(cfg["max_empty_to_nonempty"]):
        return False, "empty_to_nonempty_block"
    if "require_prefix_match" in cfg and float(best_row.get("prefix_match", 0.0)) < float(cfg["require_prefix_match"]):
        return False, "prefix_mismatch"
    if "require_family_match" in cfg and float(best_row.get("family_match", 0.0)) < float(cfg["require_family_match"]):
        return False, "family_mismatch"

    second_score = float(second_row.get("student_score", 0.0)) if second_row is not None else 0.0
    margin = float(best_row.get("student_score", 0.0)) - second_score
    min_margin = float(cfg.get("min_margin", GLOBAL_MIN_MARGIN))
    if margin >= min_margin:
        return True, "pass"

    override_ok, override_reason = pattern_override_pass(best_row, second_row)
    if override_ok:
        return True, override_reason

    return False, f"margin<{min_margin}"


def build_top1_repairs_with_gate(ranked_df: pd.DataFrame) -> pd.DataFrame:
    out_rows = []
    for (row_id, column), g in ranked_df.groupby(["row_id", "column"], sort=False):
        g = g.sort_values(["student_rank"], ascending=[True]).reset_index(drop=True)
        best = g.iloc[0]
        second = g.iloc[1] if len(g) >= 2 else None
        passed, gate_reason = pass_column_gate(best, second)

        if passed:
            # Score / Sample：空值 case 继续保守，非空 typo case 放开
            if str(best["column"]) in {"Score", "Sample"} and canonical_empty_text(best["dirty_value"]) == "empty":
                final_value = "empty"
                selected_rank = -1
                selected_source = "gate_keep_original"
                final_reason = "keep_original_empty_case"
            else:
                final_value = canonical_empty_text(best["candidate_value"])
                selected_rank = int(best["student_rank"])
                selected_source = best.get("candidate_source", "")
                final_reason = best.get("student_reason_pred", "unknown")
        else:
            final_value = canonical_empty_text(best["dirty_value"])
            selected_rank = -1
            selected_source = "gate_keep_original"
            final_reason = "keep_original"

        out_rows.append({
            "row_id": int(row_id),
            "column": str(column),
            "dirty_value": canonical_empty_text(best["dirty_value"]),
            "student_best_candidate": final_value,
            "student_score": float(best["student_score"]),
            "student_reason_pred": final_reason,
            "gate_passed": int(passed),
            "gate_reason": gate_reason,
            "selected_student_rank": selected_rank,
            "selected_candidate_source": selected_source,
            "top1_raw_candidate": canonical_empty_text(best["candidate_value"]),
            "top1_raw_score": float(best["student_score"]),
            "top2_raw_candidate": canonical_empty_text(second["candidate_value"]) if second is not None else "",
            "top2_raw_score": float(second["student_score"]) if second is not None else None,
            "top_margin": float(best["student_score"]) - (float(second["student_score"]) if second is not None else 0.0),
        })
    return pd.DataFrame(out_rows)


def build_hard_cases_jsonl(ranked_df: pd.DataFrame, max_hard_cases: int):
    hard_rows = []
    for (row_id, column), g in ranked_df.groupby(["row_id", "column"], sort=False):
        g = g.sort_values(["student_rank"], ascending=[True]).reset_index(drop=True)
        if len(g) < 2:
            continue
        best = g.iloc[0]
        second = g.iloc[1]
        passed, gate_reason = pass_column_gate(best, second)
        margin = float(best["student_score"]) - float(second["student_score"])
        is_hard = (not passed) or (margin < GLOBAL_MIN_MARGIN + 0.02)
        if not is_hard:
            continue

        hard_rows.append({
            "row_id": int(row_id),
            "column": str(column),
            "dirty_value": canonical_empty_text(best["dirty_value"]),
            "candidate_a_value": canonical_empty_text(best["candidate_value"]),
            "candidate_b_value": canonical_empty_text(second["candidate_value"]),
            "candidate_a_score": float(best["student_score"]),
            "candidate_b_score": float(second["student_score"]),
            "top_margin": margin,
            "gate_passed": int(passed),
            "gate_reason": gate_reason,
            "candidate_a_features": {
                "candidate_rule_support_ratio": float(best.get("candidate_rule_support_ratio", 0.0)),
                "bayes_mean_cond_prob": float(best.get("bayes_mean_cond_prob", 0.0)),
                "edit_similarity_to_dirty": float(best.get("edit_similarity_to_dirty", 0.0)),
                "empty_to_nonempty": float(best.get("empty_to_nonempty", 0.0)),
                "prefix_match": float(best.get("prefix_match", 0.0)),
                "family_match": float(best.get("family_match", 0.0)),
            },
            "candidate_b_features": {
                "candidate_rule_support_ratio": float(second.get("candidate_rule_support_ratio", 0.0)),
                "bayes_mean_cond_prob": float(second.get("bayes_mean_cond_prob", 0.0)),
                "edit_similarity_to_dirty": float(second.get("edit_similarity_to_dirty", 0.0)),
                "empty_to_nonempty": float(second.get("empty_to_nonempty", 0.0)),
                "prefix_match": float(second.get("prefix_match", 0.0)),
                "family_match": float(second.get("family_match", 0.0)),
            },
        })

    hard_df = pd.DataFrame(hard_rows)
    if not hard_df.empty:
        hard_df = hard_df.sort_values(["top_margin", "column"], ascending=[True, True]).head(max_hard_cases).reset_index(drop=True)

    with open(OUTPUT_HARD_CASES_JSONL, "w", encoding="utf-8") as f:
        if not hard_df.empty:
            for _, row in hard_df.iterrows():
                write_jsonl_record(f, row.to_dict())
    return hard_df


def main():
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Missing checkpoint: {CHECKPOINT_PATH}")
    if not os.path.exists(FEATURE_COLUMNS_JSON):
        raise FileNotFoundError(f"Missing feature columns: {FEATURE_COLUMNS_JSON}")
    if not os.path.exists(REASON_LABELS_JSON):
        raise FileNotFoundError(f"Missing reason label mapping: {REASON_LABELS_JSON}")

    print(f"[INFO] DEVICE = {DEVICE}")

    trained_feature_columns = load_json(FEATURE_COLUMNS_JSON)
    reason_name_to_id = load_json(REASON_LABELS_JSON)
    reason_id_to_name = {int(v): k for k, v in reason_name_to_id.items()}

    ckpt = load_checkpoint(CHECKPOINT_PATH)
    hidden_dims = ckpt.get("hidden_dims", [128, 64])
    dropout = ckpt.get("dropout", 0.15)

    model = SharedScorerMLP(len(trained_feature_columns), hidden_dims, dropout, len(reason_name_to_id)).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[INFO] Loaded checkpoint: {CHECKPOINT_PATH}")
    print(f"[INFO] checkpoint epoch = {ckpt.get('epoch', 'unknown')}")

    cand_df = load_candidate_features_for_inference(ARGS.candidate_features, trained_feature_columns)
    print(f"[INFO] Candidate count: {len(cand_df)}")

    scored_df = score_all_candidates(model, cand_df, trained_feature_columns, reason_id_to_name)
    scored_df.to_csv(OUTPUT_ALL_SCORES_CSV, index=False, encoding="utf-8-sig")

    ranked_df = rank_candidates(scored_df)
    ranked_df.to_csv(OUTPUT_ALL_RANKED_CSV, index=False, encoding="utf-8-sig")

    top1_df = build_top1_repairs_with_gate(ranked_df)
    top1_df.to_csv(OUTPUT_TOP1_CSV, index=False, encoding="utf-8-sig")

    hard_df = build_hard_cases_jsonl(ranked_df, ARGS.max_hard_cases)
    print(f"[OK] All candidate scores saved: {OUTPUT_ALL_SCORES_CSV}")
    print(f"[OK] Ranked candidates saved: {OUTPUT_ALL_RANKED_CSV}")
    print(f"[OK] Top-1 repairs saved: {OUTPUT_TOP1_CSV}")
    print(f"[OK] Hard cases saved: {OUTPUT_HARD_CASES_JSONL}")
    print(f"[INFO] Hard case count: {len(hard_df)}")


if __name__ == "__main__":
    main()

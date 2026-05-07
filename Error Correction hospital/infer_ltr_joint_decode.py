
import argparse
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn


EMPTY_TOKENS = {"", "empty", "null", "none", "nan", "na", "n/a", "missing"}
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
MEASURE_ANCHOR_TOKENS = {
    "heart", "attack", "failure", "pneumonia", "surgery", "antibiotic", "infection",
    "beta", "blockers", "children", "asthma", "blood", "clot", "discharge", "arrival"
}


def parse_args():
    p = argparse.ArgumentParser(description="Joint decoding inference for correction")
    p.add_argument("--candidate_features", default="repair_candidate_features.csv")
    p.add_argument("--model_dir", default="pairwise_ltr_student_output_joint_decode")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--global_min_margin", type=float, default=0.05)
    p.add_argument("--joint_topk", type=int, default=3)
    p.add_argument("--max_hard_cases", type=int, default=300)
    return p.parse_args()


ARGS = parse_args()
MODEL_DIR = ARGS.model_dir
CHECKPOINT_PATH = (ARGS.checkpoint if str(ARGS.checkpoint).strip() != "" else os.path.join(MODEL_DIR, "best_model.pt"))
FEATURE_COLUMNS_JSON = os.path.join(MODEL_DIR, "feature_columns.json")
REASON_LABELS_JSON = os.path.join(MODEL_DIR, "reason_label_mapping.json")
COLUMN_PROFILES_JSON = os.path.join(MODEL_DIR, "column_profiles.json")

OUTPUT_ALL_SCORES_CSV = os.path.join(MODEL_DIR, "inference_all_candidate_scores.csv")
OUTPUT_ALL_RANKED_CSV = os.path.join(MODEL_DIR, "inference_all_candidate_ranked.csv")
OUTPUT_TOP1_CSV = os.path.join(MODEL_DIR, "inference_top1_repairs.csv")
OUTPUT_HARD_CASES_JSONL = os.path.join(MODEL_DIR, "hard_cases_for_llm.jsonl")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def norm_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def canonical_empty_text(x):
    s = norm_text(x).lower()
    return "empty" if s in EMPTY_TOKENS else norm_text(x)


def load_json(path: str):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_jsonl_record(fp, obj):
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")


def is_empty_token(x: str) -> int:
    return int(norm_text(x).lower() in EMPTY_TOKENS)


def looks_like_code(x: str) -> int:
    s = norm_text(x)
    if s == "":
        return 0
    return int((bool(re.search(r"[_\-]", s)) or bool(re.search(r"\d", s))) and len(s) <= 32)


def split_code_parts(x: str):
    s = norm_text(x)
    return [] if s == "" else re.split(r"[_\-]", s)


def prefix_token(x: str) -> str:
    parts = split_code_parts(x)
    return parts[0] if parts else ""


def family_token(x: str) -> str:
    parts = split_code_parts(x)
    if len(parts) >= 2:
        return parts[1]
    return parts[0] if parts else ""


def suffix_token(x: str) -> str:
    parts = split_code_parts(x)
    return parts[-1] if parts else ""


def is_digits_only(s: str) -> bool:
    return bool(re.fullmatch(r"\d+", norm_text(s)))


def is_percent_like(s: str) -> bool:
    return bool(re.fullmatch(r"\d{1,3}%", norm_text(s)))


def alpha_space_like(s: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z\s\-\./']+", norm_text(s)))


def patients_like(s: str) -> bool:
    return bool(re.fullmatch(r"\d+\s+patients", canonical_empty_text(s).lower()))


def tiny_edit_like(a: str, b: str, max_diff: int) -> bool:
    a = norm_text(a).lower()
    b = norm_text(b).lower()
    if abs(len(a) - len(b)) > max_diff:
        return False
    if len(a) < len(b):
        a = a + "#" * (len(b) - len(a))
    elif len(b) < len(a):
        b = b + "#" * (len(a) - len(b))
    diff = sum(ch1 != ch2 for ch1, ch2 in zip(a, b))
    return diff <= max_diff


def likely_measure_code_candidate(cand: str) -> bool:
    s = canonical_empty_text(cand).lower()
    return (
        s.startswith(("ami", "hf", "pn", "scip")) or
        ("ami" in s) or ("hf" in s) or ("pn" in s) or ("scip" in s)
    )


def likely_stateavg_candidate(cand: str) -> bool:
    s = canonical_empty_text(cand).lower()
    return "_" in s and likely_measure_code_candidate(s)


def extract_anchor_tokens(s: str):
    toks = set(re.findall(r"[a-z]+", canonical_empty_text(s).lower()))
    return toks & MEASURE_ANCHOR_TOKENS


def anchor_overlap_score(a: str, b: str) -> int:
    return len(extract_anchor_tokens(a) & extract_anchor_tokens(b))


def normalize_measurecode_like(s: str) -> str:
    s = canonical_empty_text(s).lower().replace("_", "-")
    if "_" in canonical_empty_text(s):
        s = canonical_empty_text(s).split("_", 1)[1].lower().replace("_", "-")
    return s


def measure_family_name(code: str) -> str:
    c = normalize_measurecode_like(code)
    if c.startswith("hf"):
        return "heart failure"
    if c.startswith("ami"):
        return "heart attack"
    if c.startswith("pn"):
        return "pneumonia"
    if c.startswith("scip"):
        return "surgical infection prevention"
    return ""


def pair_compat_score(measurecode: str, stateavg: str) -> int:
    mc = normalize_measurecode_like(measurecode)
    sa = normalize_measurecode_like(stateavg)
    score = 0
    if mc and sa and mc == sa:
        score += 4
    if family_token(measurecode) and family_token(stateavg) and family_token(measurecode) == family_token(stateavg):
        score += 2
    if suffix_token(measurecode) and suffix_token(stateavg) and suffix_token(measurecode) == suffix_token(stateavg):
        score += 2
    return score


def infer_archetype_from_profile(profile: dict) -> str:
    if not profile:
        return "mixed"
    return str(profile.get("archetype", "mixed"))


def get_column_archetype(column: str, column_profiles: Dict[str, dict]) -> str:
    prof = column_profiles.get(column, {})
    return infer_archetype_from_profile(prof)


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
    dirty_suffix = out["dirty_value"].map(suffix_token)
    cand_suffix = out["candidate_value"].map(suffix_token)

    out["prefix_match"] = (dirty_prefix == cand_prefix).astype(int)
    out["family_match"] = (dirty_family == cand_family).astype(int)
    out["suffix_match"] = (dirty_suffix == cand_suffix).astype(int)
    out["same_code_part_count"] = (
        out["dirty_value"].map(lambda x: len(split_code_parts(x))) ==
        out["candidate_value"].map(lambda x: len(split_code_parts(x)))
    ).astype(int)

    out["empty_to_nonempty"] = (
        (out["original_is_empty"] == 1) & (out["candidate_is_empty"] == 0)
    ).astype(int)
    out["candidate_is_digits_only"] = out["candidate_value"].map(lambda x: int(is_digits_only(x)))
    out["candidate_is_percent_like"] = out["candidate_value"].map(lambda x: int(is_percent_like(x)))
    out["candidate_contains_patients"] = out["candidate_value"].map(lambda x: int("patients" in norm_text(x).lower()))
    return out


def load_candidate_features_for_inference(path: str, trained_feature_columns: List[str]) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["row_id"] = df["row_id"].astype(int)
    df["column"] = df["column"].astype(str)
    df["candidate_value"] = df["candidate_value"].map(canonical_empty_text)
    df["dirty_value"] = df["dirty_value"].map(canonical_empty_text)
    df = add_derived_features(df)

    categorical_cols = [c for c in ["semantic_type", "main_rule_type", "main_usage_role", "candidate_source"] if c in df.columns]
    df_cat = pd.get_dummies(df[categorical_cols].fillna("unknown").astype(str), prefix=categorical_cols) if categorical_cols else pd.DataFrame(index=df.index)

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


def rule_support(row: pd.Series) -> float:
    try:
        return float(row.get("candidate_rule_support_ratio", 0.0))
    except Exception:
        return 0.0


def neighbor_ratio(row: pd.Series) -> float:
    try:
        return float(row.get("neighbor_majority_ratio", 0.0))
    except Exception:
        return 0.0


def candidate_pool(ranked_df: pd.DataFrame, row_id: int, column: str, topk: int) -> pd.DataFrame:
    g = ranked_df[(ranked_df["row_id"] == row_id) & (ranked_df["column"] == column)].copy()
    if g.empty:
        return g
    g = g.sort_values(["student_rank"], ascending=[True]).reset_index(drop=True)
    return g.head(min(topk, len(g))).copy()


# ---------------- safe columns deterministic selection ----------------
def choose_for_score(g: pd.DataFrame, dirty: str) -> Optional[pd.Series]:
    if dirty == "empty":
        return None
    # if already valid percent, keep step3 behavior: do not force replacement
    if is_percent_like(dirty):
        return None

    legal = []
    for _, row in g.iterrows():
        cand = canonical_empty_text(row["candidate_value"])
        if is_percent_like(cand):
            key = (
                int(is_percent_like(cand)),
                int(tiny_edit_like(dirty, cand, 4)),
                rule_support(row),
                neighbor_ratio(row),
                float(row["student_score"]),
                -int(row["student_rank"]),
            )
            legal.append((row, key))
    if not legal:
        return None
    legal.sort(key=lambda x: x[1], reverse=True)
    return legal[0][0]


def choose_for_sample(g: pd.DataFrame, dirty: str) -> Optional[pd.Series]:
    if dirty == "empty":
        return None

    legal = []
    for _, row in g.iterrows():
        cand = canonical_empty_text(row["candidate_value"])
        if patients_like(cand):
            key = (
                int(patients_like(cand)),
                int(tiny_edit_like(dirty, cand, 5)),
                rule_support(row),
                neighbor_ratio(row),
                float(row["student_score"]),
                -int(row["student_rank"]),
            )
            legal.append((row, key))
    if not legal:
        return None
    legal.sort(key=lambda x: x[1], reverse=True)
    return legal[0][0]


def choose_for_digits(g: pd.DataFrame, dirty: str) -> Optional[pd.Series]:
    legal = []
    for _, row in g.iterrows():
        cand = canonical_empty_text(row["candidate_value"])
        if is_digits_only(cand):
            key = (
                int(is_digits_only(cand)),
                int(tiny_edit_like(dirty, cand, 4)),
                rule_support(row),
                neighbor_ratio(row),
                float(row["student_score"]),
                -int(row["student_rank"]),
            )
            legal.append((row, key))
    if not legal:
        return None
    legal.sort(key=lambda x: x[1], reverse=True)
    return legal[0][0]


def choose_for_small_enum(g: pd.DataFrame, dirty: str) -> Optional[pd.Series]:
    legal = []
    for _, row in g.iterrows():
        cand = canonical_empty_text(row["candidate_value"]).lower()
        canonical = int(
            (cand in US_STATE_CODES)
            or (cand in YES_NO_DOMAIN)
            or (cand in HOSPITAL_TYPE_DOMAIN)
            or (cand in CONDITION_DOMAIN)
            or alpha_space_like(cand)
        )
        key = (
            canonical,
            int(tiny_edit_like(dirty, cand, 6)),
            rule_support(row),
            neighbor_ratio(row),
            float(row["student_score"]),
            -int(row["student_rank"]),
        )
        legal.append((row, key))
    if not legal:
        return None
    legal.sort(key=lambda x: x[1], reverse=True)
    return legal[0][0]


# ---------------- dangerous columns: filter + joint decoding ----------------
def filter_measurecode_pool(g: pd.DataFrame, dirty: str) -> pd.DataFrame:
    x = g.copy()
    x = x[x["candidate_value"].map(likely_measure_code_candidate)]
    if len(x) > 0:
        x = x[x["candidate_value"].map(lambda c: tiny_edit_like(dirty, c, 4))]
    return x if len(x) > 0 else g


def filter_stateavg_pool(g: pd.DataFrame, dirty: str) -> pd.DataFrame:
    x = g.copy()
    x = x[x["candidate_value"].map(likely_stateavg_candidate)]
    if "family_match" in x.columns and len(x) > 0:
        x = x[x["family_match"] >= 1.0]
    if "_" in dirty and "prefix_match" in x.columns and len(x) > 0:
        x = x[x["prefix_match"] >= 1.0]
    if "same_code_part_count" in x.columns and len(x) > 0:
        x = x[x["same_code_part_count"] >= 1.0]
    return x if len(x) > 0 else g


def best_row_by_base_score(g: pd.DataFrame) -> pd.Series:
    g = g.sort_values(["student_score", "student_rank"], ascending=[False, True]).reset_index(drop=True)
    return g.iloc[0]


def joint_decode_measure_bundle(
    mc_pool: pd.DataFrame,
    sa_pool: pd.DataFrame,
    cond_pool: pd.DataFrame,
    name_pool: pd.DataFrame,
) -> Tuple[Optional[pd.Series], Optional[pd.Series], Optional[pd.Series], Optional[pd.Series]]:
    """
    Joint decode 4 variables:
    MeasureCode, Stateavg, Condition, MeasureName
    """
    if mc_pool.empty:
        mc_pool = pd.DataFrame([{}])
    if sa_pool.empty:
        sa_pool = pd.DataFrame([{}])
    if cond_pool.empty:
        cond_pool = pd.DataFrame([{}])
    if name_pool.empty:
        name_pool = pd.DataFrame([{}])

    best_combo = None
    best_key = None

    for _, mc in mc_pool.iterrows():
        mc_val = canonical_empty_text(mc.get("candidate_value", ""))
        mc_family = measure_family_name(mc_val)

        for _, sa in sa_pool.iterrows():
            sa_val = canonical_empty_text(sa.get("candidate_value", ""))

            for _, cond in cond_pool.iterrows():
                cond_val = canonical_empty_text(cond.get("candidate_value", "")).lower()

                for _, name in name_pool.iterrows():
                    name_val = canonical_empty_text(name.get("candidate_value", "")).lower()

                    # individual scores
                    s_ind = 0.0
                    if mc_val:
                        s_ind += float(mc.get("student_score", 0.0)) + 0.8 * rule_support(mc)
                    if sa_val:
                        s_ind += float(sa.get("student_score", 0.0)) + 0.8 * rule_support(sa)
                    if cond_val:
                        s_ind += float(cond.get("student_score", 0.0)) + 0.6 * rule_support(cond)
                    if name_val:
                        s_ind += float(name.get("student_score", 0.0)) + 0.6 * rule_support(name)

                    # pair consistency
                    s_pair = 0.0
                    if mc_val and sa_val:
                        s_pair += 1.2 * pair_compat_score(mc_val, sa_val)

                    # condition consistency with code/name
                    if cond_val and mc_family:
                        if mc_family in cond_val:
                            s_pair += 2.0
                    if cond_val and name_val:
                        if any(tok in name_val for tok in cond_val.split()):
                            s_pair += 1.0

                    # measure name consistency with code family
                    if name_val and mc_family:
                        if mc_family == "heart failure" and ("heart" in name_val or "failure" in name_val):
                            s_pair += 2.0
                        elif mc_family == "heart attack" and ("heart" in name_val or "attack" in name_val):
                            s_pair += 2.0
                        elif mc_family == "pneumonia" and "pneumonia" in name_val:
                            s_pair += 2.0
                        elif mc_family == "surgical infection prevention" and (
                            "surgery" in name_val or "infection" in name_val or "antibiotic" in name_val
                        ):
                            s_pair += 2.0

                    key = (
                        s_ind + s_pair,
                        pair_compat_score(mc_val, sa_val) if mc_val and sa_val else -1,
                        float(mc.get("student_score", -999.0)),
                        float(sa.get("student_score", -999.0)),
                        float(cond.get("student_score", -999.0)),
                        float(name.get("student_score", -999.0)),
                    )

                    if best_key is None or key > best_key:
                        best_key = key
                        best_combo = (mc if mc_val else None, sa if sa_val else None, cond if cond_val else None, name if name_val else None)

    return best_combo if best_combo is not None else (None, None, None, None)


def build_top1_with_joint_decoding(ranked_df: pd.DataFrame, column_profiles: Dict[str, dict]) -> pd.DataFrame:
    """
    Main chain:
    1. safe columns deterministic selection
    2. joint decode MeasureCode/Stateavg/Condition/MeasureName
    3. fallback to top1 if nothing better
    """
    out_rows = []
    rows = sorted(set(int(x) for x in ranked_df["row_id"].unique()))
    cols = sorted(set(str(x) for x in ranked_df["column"].unique()))

    # first pass: safe columns + simple fallback
    provisional = {}
    for row_id in rows:
        for col in cols:
            pool = candidate_pool(ranked_df, row_id, col, ARGS.joint_topk)
            if pool.empty:
                continue

            dirty = canonical_empty_text(pool.iloc[0]["dirty_value"])
            chosen = None
            arch = get_column_archetype(col, column_profiles)

            if col == "Score" or arch == "percent_pattern":
                chosen = choose_for_score(pool, dirty)
            elif col == "Sample":
                chosen = choose_for_sample(pool, dirty)
            elif arch in {"long_digit_id", "short_digit_id"} or col in {"PhoneNumber", "ZipCode", "ProviderNumber"}:
                chosen = choose_for_digits(pool, dirty)
            elif arch == "small_enum" or col in {"State", "EmergencyService", "HospitalType", "HospitalOwner"}:
                chosen = choose_for_small_enum(pool, dirty)

            if chosen is None:
                if col == "MeasureCode":
                    chosen = best_row_by_base_score(filter_measurecode_pool(pool, dirty))
                elif col == "Stateavg":
                    chosen = best_row_by_base_score(filter_stateavg_pool(pool, dirty))
                else:
                    chosen = best_row_by_base_score(pool)

            provisional[(row_id, col)] = chosen

    # second pass: joint decode 4-column bundle
    for row_id in rows:
        mc_pool = filter_measurecode_pool(candidate_pool(ranked_df, row_id, "MeasureCode", ARGS.joint_topk), canonical_empty_text(provisional.get((row_id, "MeasureCode"), {}).get("dirty_value", "")) if (row_id, "MeasureCode") in provisional else "")
        sa_pool = filter_stateavg_pool(candidate_pool(ranked_df, row_id, "Stateavg", ARGS.joint_topk), canonical_empty_text(provisional.get((row_id, "Stateavg"), {}).get("dirty_value", "")) if (row_id, "Stateavg") in provisional else "")
        cond_pool = candidate_pool(ranked_df, row_id, "Condition", ARGS.joint_topk)
        name_pool = candidate_pool(ranked_df, row_id, "MeasureName", ARGS.joint_topk)

        mc_row, sa_row, cond_row, name_row = joint_decode_measure_bundle(mc_pool, sa_pool, cond_pool, name_pool)

        if mc_row is not None:
            provisional[(row_id, "MeasureCode")] = mc_row
        if sa_row is not None:
            provisional[(row_id, "Stateavg")] = sa_row
        if cond_row is not None:
            provisional[(row_id, "Condition")] = cond_row
        if name_row is not None:
            provisional[(row_id, "MeasureName")] = name_row

    # materialize final output
    for (row_id, col), chosen in sorted(provisional.items(), key=lambda x: (x[0][0], x[0][1])):
        dirty = canonical_empty_text(chosen.get("dirty_value", ""))
        final_value = canonical_empty_text(chosen.get("candidate_value", dirty))

        # conservative empty cases
        if col in {"Score", "Sample"} and dirty == "empty":
            final_value = dirty
            gate_passed = 0
            gate_reason = "keep_original_empty_case"
            selected_rank = -1
            selected_source = "gate_keep_original"
            reason = "keep_original_empty_case"
        else:
            gate_passed = 1
            gate_reason = "joint_or_deterministic"
            selected_rank = int(chosen.get("student_rank", -1))
            selected_source = str(chosen.get("candidate_source", "joint_decode"))
            reason = str(chosen.get("student_reason_pred", "joint_decode"))

        out_rows.append({
            "row_id": int(row_id),
            "column": str(col),
            "dirty_value": dirty,
            "student_best_candidate": final_value,
            "student_score": float(chosen.get("student_score", 0.0)),
            "student_reason_pred": reason,
            "gate_passed": int(gate_passed),
            "gate_reason": gate_reason,
            "selected_student_rank": selected_rank,
            "selected_candidate_source": selected_source,
            "top1_raw_candidate": canonical_empty_text(chosen.get("candidate_value", "")),
            "top1_raw_score": float(chosen.get("student_score", 0.0)),
            "top2_raw_candidate": "",
            "top2_raw_score": None,
            "top_margin": None,
        })

    return pd.DataFrame(out_rows)


def build_hard_cases_jsonl(ranked_df: pd.DataFrame, max_hard_cases: int):
    # simple hard-case export: top1-top2 close or structured code family ambiguity
    rows = []
    for (row_id, column), g in ranked_df.groupby(["row_id", "column"], sort=False):
        g = g.sort_values(["student_rank"], ascending=[True]).reset_index(drop=True)
        if len(g) < 2:
            continue
        best = g.iloc[0]
        second = g.iloc[1]
        margin = float(best["student_score"]) - float(second["student_score"])
        is_hard = False
        if margin < ARGS.global_min_margin + 0.02:
            is_hard = True
        if str(column) in {"MeasureCode", "Stateavg", "MeasureName", "Condition"}:
            is_hard = True if margin < 0.08 else is_hard
        if not is_hard:
            continue
        rows.append({
            "row_id": int(row_id),
            "column": str(column),
            "dirty_value": canonical_empty_text(best["dirty_value"]),
            "candidate_a_value": canonical_empty_text(best["candidate_value"]),
            "candidate_b_value": canonical_empty_text(second["candidate_value"]),
            "candidate_a_score": float(best["student_score"]),
            "candidate_b_score": float(second["student_score"]),
            "top_margin": margin,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["top_margin", "column"], ascending=[True, True]).head(max_hard_cases).reset_index(drop=True)
    with open(OUTPUT_HARD_CASES_JSONL, "w", encoding="utf-8") as f:
        if not df.empty:
            for _, row in df.iterrows():
                write_jsonl_record(f, row.to_dict())
    return df


def main():
    for p in [CHECKPOINT_PATH, FEATURE_COLUMNS_JSON, REASON_LABELS_JSON, COLUMN_PROFILES_JSON]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file: {p}")

    trained_feature_columns = load_json(FEATURE_COLUMNS_JSON)
    reason_name_to_id = load_json(REASON_LABELS_JSON)
    column_profiles = load_json(COLUMN_PROFILES_JSON)
    reason_id_to_name = {int(v): k for k, v in reason_name_to_id.items()}

    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    hidden_dims = ckpt.get("hidden_dims", [128, 64])
    dropout = ckpt.get("dropout", 0.15)

    model = SharedScorerMLP(len(trained_feature_columns), hidden_dims, dropout, len(reason_name_to_id)).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    cand_df = load_candidate_features_for_inference(ARGS.candidate_features, trained_feature_columns)
    print(f"[INFO] DEVICE = {DEVICE}")
    print(f"[INFO] Candidate count: {len(cand_df)}")

    scored_df = score_all_candidates(model, cand_df, trained_feature_columns, reason_id_to_name)
    scored_df.to_csv(OUTPUT_ALL_SCORES_CSV, index=False, encoding="utf-8-sig")

    ranked_df = rank_candidates(scored_df)
    ranked_df.to_csv(OUTPUT_ALL_RANKED_CSV, index=False, encoding="utf-8-sig")

    top1_df = build_top1_with_joint_decoding(ranked_df, column_profiles)
    top1_df.to_csv(OUTPUT_TOP1_CSV, index=False, encoding="utf-8-sig")

    hard_df = build_hard_cases_jsonl(ranked_df, ARGS.max_hard_cases)
    print(f"[OK] Top-1 repairs saved: {OUTPUT_TOP1_CSV}")
    print(f"[OK] Hard cases saved: {OUTPUT_HARD_CASES_JSONL}")
    print(f"[INFO] Hard case count: {len(hard_df)}")


if __name__ == "__main__":
    main()

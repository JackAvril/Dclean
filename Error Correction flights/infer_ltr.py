import argparse
import json
import os
import re
from typing import Dict, List, Optional

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
    "children s asthma care",
}
PROTECTED_NONEMPTY_COLUMNS = set()  # Flights 中不使用 Hospital 的受保护列
COLUMN_MARGIN_THRESHOLDS = {
    "sched_dep_time": 0.03,
    "act_dep_time": 0.03,
    "sched_arr_time": 0.03,
    "act_arr_time": 0.03,
}
DEFAULT_MARGIN_THRESHOLD = 0.05
TIME_COLUMNS = {"sched_dep_time", "act_dep_time", "sched_arr_time", "act_arr_time"}


def parse_args():
    p = argparse.ArgumentParser(description="Inference + hard case mining (expects prefiltered whitelist candidate_features)")
    p.add_argument("--candidate_features", default="/mnt/mydata/dq/projects/Splittree/test/Error Correction flights/repair_candidate_features_infer_whitelist.csv")
    p.add_argument("--allowed_cells_csv", default="/mnt/mydata/dq/projects/Splittree/test/Error Correction flights/predicted_error_positions.csv")
    p.add_argument("--model_dir", default="/mnt/mydata/dq/projects/Splittree/test/Error Correction flights/pairwise_ltr_student_output_flights")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--global_min_margin", type=float, default=0.05)
    p.add_argument("--joint_topk", type=int, default=4)
    p.add_argument("--max_hard_cases", type=int, default=300)
    return p.parse_args()


ARGS = parse_args()
MODEL_DIR = ARGS.model_dir
CHECKPOINT_PATH = ARGS.checkpoint or os.path.join(MODEL_DIR, "best_model.pt")
FEATURE_COLUMNS_JSON = os.path.join(MODEL_DIR, "feature_columns.json")
REASON_LABELS_JSON = os.path.join(MODEL_DIR, "reason_label_mapping.json")
COLUMN_PROFILES_JSON = os.path.join(MODEL_DIR, "column_profiles.json")
OUTPUT_ALL_SCORES_CSV = os.path.join(MODEL_DIR, "inference_all_candidate_scores.csv")
OUTPUT_ALL_RANKED_CSV = os.path.join(MODEL_DIR, "inference_all_candidate_ranked.csv")
OUTPUT_TOP1_CSV = os.path.join(MODEL_DIR, "inference_top1_repairs.csv")
OUTPUT_HARD_CASES_JSONL = os.path.join(MODEL_DIR, "hard_cases_for_llm.jsonl")
INPUT_WHITELIST_VIOLATIONS_CSV = os.path.join(MODEL_DIR, "input_whitelist_violations.csv")
OUTPUT_WHITELIST_VIOLATIONS_CSV = os.path.join(MODEL_DIR, "output_whitelist_violations.csv")
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


def is_time_column_name(column: str) -> int:
    c = norm_text(column).lower()
    return int(c in TIME_COLUMNS or c.endswith("_time"))


def parse_time_to_minutes_simple(x):
    s = canonical_empty_text(x).lower()
    if s in {"", "empty"}:
        return None
    s = s.replace(".", "")
    s = re.sub(r"\s+", " ", s).strip()
    ampm = None
    if "am" in s:
        ampm = "am"
    elif "pm" in s:
        ampm = "pm"
    s2 = s.replace("am", "").replace("pm", "").strip()
    hour = minute = None
    m = re.fullmatch(r"(\d{1,2})\s*:\s*(\d{2})", s2)
    if m:
        hour, minute = int(m.group(1)), int(m.group(2))
    else:
        digits = re.sub(r"\D", "", s2)
        if len(digits) in {3, 4}:
            hour, minute = int(digits[:-2]), int(digits[-2:])
    if hour is None or minute is None or minute < 0 or minute > 59:
        return None
    if ampm == "am":
        if hour == 12:
            hour = 0
        elif not (1 <= hour <= 11):
            return None
    elif ampm == "pm":
        if hour == 12:
            pass
        elif 1 <= hour <= 11:
            hour += 12
        else:
            return None
    else:
        if not (0 <= hour <= 23):
            return None
    return hour * 60 + minute


def is_valid_time_simple(x) -> int:
    return int(parse_time_to_minutes_simple(x) is not None)


def circular_minute_diff_simple(a, b):
    ma = parse_time_to_minutes_simple(a)
    mb = parse_time_to_minutes_simple(b)
    if ma is None or mb is None:
        return 0.0
    d = abs(ma - mb)
    return float(min(d, 1440 - d))


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
    return sum(ch1 != ch2 for ch1, ch2 in zip(a, b)) <= max_diff


def likely_measure_code_candidate(cand: str) -> bool:
    s = canonical_empty_text(cand).lower()
    return s != "empty" and (s.startswith(("ami", "hf", "pn", "scip")) or ("ami" in s) or ("hf" in s) or ("pn" in s) or ("scip" in s))


def likely_stateavg_candidate(cand: str) -> bool:
    s = canonical_empty_text(cand).lower()
    return s != "empty" and "_" in s and likely_measure_code_candidate(s)


def normalize_measurecode_like(s: str) -> str:
    raw = canonical_empty_text(s).lower()
    if "_" in raw:
        raw = raw.split("_", 1)[1]
    return raw.replace("_", "-")


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



def has_x_placeholder(s: str) -> int:
    return int("x" in norm_text(s).lower())


def extract_leading_number_token(s: str) -> str:
    m = re.match(r"\s*([^\s]+)", norm_text(s))
    return m.group(1) if m else ""


def score_template_match_score(dirty: str, cand: str) -> int:
    d = canonical_empty_text(dirty).lower()
    c = canonical_empty_text(cand).lower()
    if not is_percent_like(c):
        return -999

    score = 0
    if len(d) == len(c):
        score += 3
    if d.endswith("x") and c.endswith("%"):
        score += 3
    if "%" not in d and c.endswith("%"):
        score += 1

    n = min(len(d), len(c))
    for i in range(n):
        chd, chc = d[i], c[i]
        if chd == "x":
            score += 1
        elif chd == chc:
            score += 2
        else:
            score -= 4

    if len(c) > len(d):
        score -= (len(c) - len(d))
    elif len(d) > len(c):
        score -= 2 * (len(d) - len(c))
    return score


def sample_template_match_score(dirty: str, cand: str) -> int:
    d = canonical_empty_text(dirty).lower()
    c = canonical_empty_text(cand).lower()
    if not patients_like(c):
        return -999

    score = 0
    dirty_num = extract_leading_number_token(d)
    cand_num = extract_leading_number_token(c)
    if dirty_num and cand_num:
        if len(dirty_num) == len(cand_num):
            score += 2
        n = min(len(dirty_num), len(cand_num))
        for i in range(n):
            chd, chc = dirty_num[i], cand_num[i]
            if chd == "x":
                score += 1
            elif chd == chc:
                score += 2
            else:
                score -= 3

    # 强偏好把各种 patients 相关 typo 统一修成规范 "patients"
    if "patient" in d or "pat" in d:
        score += 2
    if c.endswith("patients"):
        score += 3

    # 文本整体轻度相似性
    score += int(tiny_edit_like(d, c, 12))
    return score


def infer_archetype_from_profile(profile: dict) -> str:
    if not profile:
        return "mixed"
    return str(profile.get("archetype", "mixed"))


def build_column_profiles_only(df: pd.DataFrame):
    profiles = {}
    for col, g in df.groupby("column"):
        vals = pd.concat([g["dirty_value"], g["candidate_value"]], ignore_index=True).dropna().astype(str)
        vals = vals.map(canonical_empty_text)
        vals = vals[vals != "empty"]
        if len(vals) == 0:
            profiles[str(col)] = {"archetype": "mixed", "avg_len": 0.0, "digits_ratio": 0.0, "percent_ratio": 0.0, "code_ratio": 0.0, "alpha_ratio": 0.0, "unique_count": 0, "sample_values": []}
            continue
        avg_len = float(vals.map(lambda x: len(norm_text(x))).mean())
        digits_ratio = float(vals.map(lambda x: int(bool(re.fullmatch(r"\d+", norm_text(x))))).mean())
        percent_ratio = float(vals.map(lambda x: int(bool(re.fullmatch(r"\d{1,3}%", norm_text(x))))).mean())
        code_ratio = float(vals.map(looks_like_code).mean())
        alpha_ratio = float(vals.map(lambda x: int(bool(re.fullmatch(r"[A-Za-z\s\-\./']+", norm_text(x))))).mean())
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
        profiles[str(col)] = {"archetype": archetype, "avg_len": avg_len, "digits_ratio": digits_ratio, "percent_ratio": percent_ratio, "code_ratio": code_ratio, "alpha_ratio": alpha_ratio, "unique_count": unique_count, "sample_values": list(vals.head(20))}
    return profiles


def ensure_column_profiles_json(candidate_features_path: str, output_json_path: str):
    if os.path.exists(output_json_path):
        return
    df = pd.read_csv(candidate_features_path, encoding="utf-8-sig")
    df["column"] = df["column"].astype(str).str.strip()
    df["dirty_value"] = df["dirty_value"].map(canonical_empty_text)
    df["candidate_value"] = df["candidate_value"].map(canonical_empty_text)
    profiles = build_column_profiles_only(df)
    os.makedirs(os.path.dirname(output_json_path) or ".", exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8-sig") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)


def get_column_archetype(column: str, column_profiles: Dict[str, dict]) -> str:
    return infer_archetype_from_profile(column_profiles.get(column, {}))


def load_allowed_cells(path: str):
    if path is None or str(path).strip() == "":
        return None
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"row_id", "column"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"allowed_cells_csv must contain columns: {required}")
    df["row_id"] = pd.to_numeric(df["row_id"], errors="raise").astype(int)
    df["column"] = df["column"].astype(str).str.strip()
    allowed = set((int(r["row_id"]), str(r["column"]).strip()) for _, r in df[["row_id", "column"]].drop_duplicates().iterrows())
    print(f"[INFO] allowed whitelist unique cells = {len(allowed)}")
    return allowed


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


NON_FEATURE_COLUMNS = {"row_id", "column", "dirty_value", "candidate_value", "candidate_rank", "candidate_source", "extra_info", "neighbor_majority_value", "suggested_correct_value", "main_rule_type", "main_usage_role", "semantic_type"}


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

    out["dirty_is_percent_like"] = out["dirty_value"].map(lambda x: int(is_percent_like(x)))
    out["dirty_is_patients_like"] = out["dirty_value"].map(lambda x: int(patients_like(x)))
    out["dirty_has_x_placeholder"] = out["dirty_value"].map(has_x_placeholder)
    out["candidate_is_digits_only"] = out["candidate_value"].map(lambda x: int(is_digits_only(x)))
    out["candidate_is_percent_like"] = out["candidate_value"].map(lambda x: int(is_percent_like(x)))
    out["candidate_contains_patients"] = out["candidate_value"].map(lambda x: int("patients" in norm_text(x).lower()))
    out["score_template_match_score"] = [
        score_template_match_score(d, c) for d, c in zip(out["dirty_value"], out["candidate_value"])
    ]
    out["sample_template_match_score"] = [
        sample_template_match_score(d, c) for d, c in zip(out["dirty_value"], out["candidate_value"])
    ]

    for col_name in ["Score", "Sample", "Stateavg", "MeasureCode", "MeasureName", "Condition"]:
        out[f"is_col_{col_name}"] = (out["column"] == col_name).astype(int)

    for col_name in ["sched_dep_time", "act_dep_time", "sched_arr_time", "act_arr_time"]:
        out[f"is_col_{col_name}"] = (out["column"] == col_name).astype(int)
    out["is_time_column_derived"] = out["column"].map(is_time_column_name)
    out["dirty_is_valid_time_derived"] = out["dirty_value"].map(is_valid_time_simple)
    out["candidate_is_valid_time_derived"] = out["candidate_value"].map(is_valid_time_simple)
    out["candidate_dirty_time_diff_derived"] = [
        circular_minute_diff_simple(d, c) for d, c in zip(out["dirty_value"], out["candidate_value"])
    ]
    return out



def validate_prefiltered_candidate_features(df: pd.DataFrame, allowed_cells):
    unique_cells = df[["row_id", "column"]].drop_duplicates().copy()
    if allowed_cells is None:
        print(f"[INFO] inference candidate unique cells (no whitelist validation) = {len(unique_cells)}")
        return
    bad_mask = unique_cells.apply(lambda r: (int(r["row_id"]), str(r["column"]).strip()) not in allowed_cells, axis=1)
    bad_count = int(bad_mask.sum())
    print(f"[INFO] prefiltered input whitelist violations = {bad_count}")
    if bad_count > 0:
        unique_cells.loc[bad_mask].to_csv(INPUT_WHITELIST_VIOLATIONS_CSV, index=False, encoding="utf-8-sig")
        raise ValueError(
            "candidate_features is not prefiltered to whitelist. "
            f"Violations saved to: {INPUT_WHITELIST_VIOLATIONS_CSV}"
        )


def load_candidate_features_for_inference(path: str, trained_feature_columns: List[str], allowed_cells=None) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["row_id"] = pd.to_numeric(df["row_id"], errors="raise").astype(int)
    df["column"] = df["column"].astype(str).str.strip()
    input_unique_cells = df[["row_id", "column"]].drop_duplicates().shape[0]
    print(f"[INFO] inference input unique candidate cells = {input_unique_cells}")
    validate_prefiltered_candidate_features(df, allowed_cells)

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


def evidence_strength(row: pd.Series) -> float:
    s = 1.2 * rule_support(row) + 0.8 * neighbor_ratio(row)
    for k, w in [("candidate_matches_similar_row_target", 0.8), ("is_rule_and_dictionary_supported", 0.6), ("candidate_in_column_top_values", 0.4)]:
        try:
            s += w * float(row.get(k, 0.0))
        except Exception:
            pass
    return s


def candidate_pool(ranked_df: pd.DataFrame, row_id: int, column: str, topk: int) -> pd.DataFrame:
    g = ranked_df[(ranked_df["row_id"] == row_id) & (ranked_df["column"] == column)].copy()
    if g.empty:
        return g
    g = g.sort_values(["student_rank"], ascending=[True]).reset_index(drop=True)
    return g.head(min(topk, len(g))).copy()


def choose_for_score(g: pd.DataFrame, dirty: str) -> Optional[pd.Series]:
    # 空值分数非常容易产生大规模误改，直接不修
    if dirty == "empty":
        return None
    # 本来就是合法百分比时，交给 gate 保守保留原值
    if is_percent_like(dirty):
        return None

    legal = []
    for _, row in g.iterrows():
        cand = canonical_empty_text(row["candidate_value"])
        if is_percent_like(cand):
            tpl = score_template_match_score(dirty, cand)
            key = (
                tpl,
                int(len(cand) == len(dirty)),
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
            tpl = sample_template_match_score(dirty, cand)
            key = (
                tpl,
                int(len(extract_leading_number_token(cand)) == len(extract_leading_number_token(dirty))),
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
            key = (int(tiny_edit_like(dirty, cand, 4)), rule_support(row), neighbor_ratio(row), float(row["student_score"]), -int(row["student_rank"]))
            legal.append((row, key))
    if not legal:
        return None
    legal.sort(key=lambda x: x[1], reverse=True)
    return legal[0][0]


def choose_for_small_enum(g: pd.DataFrame, dirty: str) -> Optional[pd.Series]:
    legal = []
    for _, row in g.iterrows():
        cand = canonical_empty_text(row["candidate_value"]).lower()
        canonical = int((cand in US_STATE_CODES) or (cand in YES_NO_DOMAIN) or (cand in HOSPITAL_TYPE_DOMAIN) or (cand in CONDITION_DOMAIN) or alpha_space_like(cand))
        key = (canonical, int(tiny_edit_like(dirty, cand, 6)), rule_support(row), neighbor_ratio(row), float(row["student_score"]), -int(row["student_rank"]))
        legal.append((row, key))
    if not legal:
        return None
    legal.sort(key=lambda x: x[1], reverse=True)
    return legal[0][0]


def filter_measurecode_pool(g: pd.DataFrame, dirty: str) -> pd.DataFrame:
    x = g.copy(); x = x[x["candidate_value"].map(likely_measure_code_candidate)]
    if len(x) > 0:
        x = x[x["candidate_value"].map(lambda c: tiny_edit_like(dirty, c, 4))]
    return x if len(x) > 0 else g


def filter_stateavg_pool(g: pd.DataFrame, dirty: str) -> pd.DataFrame:
    x = g.copy(); x = x[x["candidate_value"].map(likely_stateavg_candidate)]
    if "family_match" in x.columns and len(x) > 0:
        x = x[x["family_match"] >= 1.0]
    if "_" in dirty and "prefix_match" in x.columns and len(x) > 0:
        x = x[x["prefix_match"] >= 1.0]
    if "same_code_part_count" in x.columns and len(x) > 0:
        x = x[x["same_code_part_count"] >= 1.0]
    return x if len(x) > 0 else g


def best_row_by_base_score(g: pd.DataFrame) -> pd.Series:
    return g.sort_values(["student_score", "student_rank"], ascending=[False, True]).reset_index(drop=True).iloc[0]


def joint_decode_measure_bundle(mc_pool, sa_pool, cond_pool, name_pool):
    if mc_pool.empty: mc_pool = pd.DataFrame([{}])
    if sa_pool.empty: sa_pool = pd.DataFrame([{}])
    if cond_pool.empty: cond_pool = pd.DataFrame([{}])
    if name_pool.empty: name_pool = pd.DataFrame([{}])
    best_combo = None; best_key = None
    for _, mc in mc_pool.iterrows():
        mc_val = canonical_empty_text(mc.get("candidate_value", "")); mc_family = measure_family_name(mc_val)
        for _, sa in sa_pool.iterrows():
            sa_val = canonical_empty_text(sa.get("candidate_value", ""))
            for _, cond in cond_pool.iterrows():
                cond_val = canonical_empty_text(cond.get("candidate_value", "")).lower()
                for _, name in name_pool.iterrows():
                    name_val = canonical_empty_text(name.get("candidate_value", "")).lower()
                    s_ind = 0.0
                    if mc_val: s_ind += float(mc.get("student_score", 0.0)) + 0.8 * rule_support(mc)
                    if sa_val: s_ind += float(sa.get("student_score", 0.0)) + 0.8 * rule_support(sa)
                    if cond_val: s_ind += float(cond.get("student_score", 0.0)) + 0.6 * rule_support(cond)
                    if name_val: s_ind += float(name.get("student_score", 0.0)) + 0.6 * rule_support(name)
                    s_pair = 0.0
                    if mc_val and sa_val: s_pair += 1.2 * pair_compat_score(mc_val, sa_val)
                    if cond_val and mc_family and mc_family in cond_val: s_pair += 2.0
                    if cond_val and name_val and any(tok in name_val for tok in cond_val.split()): s_pair += 1.0
                    key = (s_ind + s_pair, pair_compat_score(mc_val, sa_val) if mc_val and sa_val else -1, float(mc.get("student_score", -999.0)), float(sa.get("student_score", -999.0)), float(cond.get("student_score", -999.0)), float(name.get("student_score", -999.0)))
                    if best_key is None or key > best_key:
                        best_key = key; best_combo = (mc if mc_val else None, sa if sa_val else None, cond if cond_val else None, name if name_val else None)
    return best_combo if best_combo is not None else (None, None, None, None)


def pass_column_gate(column: str, dirty: str, best: pd.Series, second: Optional[pd.Series], column_profiles: Dict[str, dict]):
    best_val = canonical_empty_text(best.get("candidate_value", dirty))
    second_score = float(second.get("student_score", 0.0)) if second is not None else 0.0
    margin = float(best.get("student_score", 0.0)) - second_score
    threshold = max(ARGS.global_min_margin, COLUMN_MARGIN_THRESHOLDS.get(column, DEFAULT_MARGIN_THRESHOLD))
    evidence = evidence_strength(best)
    arch = get_column_archetype(column, column_profiles)
    score_tpl = score_template_match_score(dirty, best_val)
    sample_tpl = sample_template_match_score(dirty, best_val)

    if str(column) in TIME_COLUMNS or str(column).endswith("_time"):
        # Flights 时间列 gate：优先保留规则/上下文强支持的合法时间，避免把合法原值轻易改掉。
        if best_val == dirty:
            return False, "time_keep_original_best_equals_dirty"
        if not is_valid_time_simple(best_val):
            return False, "time_candidate_invalid"
        rule_ev = rule_support(best)
        try:
            time_gain = float(best.get("candidate_time_context_gain", 0.0))
        except Exception:
            time_gain = 0.0
        if dirty == "empty":
            if rule_ev >= 0.2 or evidence >= 0.7 or margin >= threshold:
                return True, "time_empty_to_supported_value"
            return False, "time_empty_low_evidence"
        if is_valid_time_simple(dirty) and margin < max(threshold, 0.06) and evidence < 0.8:
            return False, "time_cleanlike_keep_original"
        if margin < threshold and evidence < 0.7 and time_gain <= 0:
            return False, "time_low_margin_low_evidence"
        return True, "time_passed_gate"

    if best_val == dirty:
        return False, "keep_original_best_equals_dirty"
    if column in PROTECTED_NONEMPTY_COLUMNS and dirty != "empty" and best_val == "empty":
        return False, "protected_nonempty_to_empty_block"

    if column == "Score":
        # 这是当前最主要的 FP 来源：clean empty Score 不能被自动补值
        if dirty == "empty":
            return False, "score_keep_if_original_empty"
        if is_percent_like(dirty):
            return False, "score_keep_if_already_valid"
        if not is_percent_like(best_val):
            return False, "score_candidate_not_valid_percent"
        # malformed percent（如 x00%, 95x）只有模板匹配足够强才放行
        if score_tpl < 4:
            return False, "score_template_mismatch"
        if margin < 0.01 and evidence < 0.6:
            return False, "score_low_margin_low_evidence"
    if column == "Sample":
        if dirty == "empty":
            return False, "sample_keep_if_original_empty"
        if not patients_like(best_val):
            return False, "sample_candidate_not_valid_pattern"
        if patients_like(dirty):
            # 原值已经合法时更保守
            if margin < max(threshold, 0.08) and evidence < 0.9:
                return False, "sample_cleanlike_keep_original"
        else:
            # 原值是 typo / malformed 时，只要模板强匹配就允许放行
            if sample_tpl < 4 and evidence < 0.5:
                return False, "sample_template_mismatch"
    if column == "Condition" and dirty.lower() in CONDITION_DOMAIN and (margin < threshold or evidence < 0.8):
        return False, "condition_cleanlike_keep_original"
    if column == "MeasureCode":
        if likely_measure_code_candidate(dirty) and (margin < threshold or evidence < 0.8):
            return False, "measurecode_cleanlike_keep_original"
        if not likely_measure_code_candidate(best_val):
            return False, "measurecode_candidate_not_code_like"
    if column == "Stateavg":
        if likely_stateavg_candidate(dirty) and (margin < threshold or evidence < 0.8):
            return False, "stateavg_cleanlike_keep_original"
        if not likely_stateavg_candidate(best_val):
            return False, "stateavg_candidate_not_stateavg_like"
    if column == "MeasureName":
        if dirty != "empty" and alpha_space_like(dirty) and len(dirty) >= 25 and (margin < threshold or evidence < 0.8):
            return False, "measurename_cleanlike_keep_original"
        if best_val == "empty":
            return False, "measurename_empty_block"
    if arch in {"small_enum", "structured_code", "long_text"} and margin < threshold and evidence < 0.8:
        return False, "generic_low_margin_keep_original"
    return True, "passed_gate"



def build_top1_with_gate_and_joint_decoding(ranked_df: pd.DataFrame, column_profiles: Dict[str, dict]) -> pd.DataFrame:
    out_rows = []
    rows = sorted(set(int(x) for x in ranked_df["row_id"].unique()))
    cols = sorted(set(str(x) for x in ranked_df["column"].unique()))
    provisional = {}
    raw_pools = {}
    for row_id in rows:
        for col in cols:
            pool = candidate_pool(ranked_df, row_id, col, ARGS.joint_topk)
            if pool.empty:
                continue
            raw_pools[(row_id, col)] = pool
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
    for row_id in rows:
        mc_pool = filter_measurecode_pool(candidate_pool(ranked_df, row_id, "MeasureCode", ARGS.joint_topk), canonical_empty_text(provisional.get((row_id, "MeasureCode"), {}).get("dirty_value", "")) if (row_id, "MeasureCode") in provisional else "")
        sa_pool = filter_stateavg_pool(candidate_pool(ranked_df, row_id, "Stateavg", ARGS.joint_topk), canonical_empty_text(provisional.get((row_id, "Stateavg"), {}).get("dirty_value", "")) if (row_id, "Stateavg") in provisional else "")
        cond_pool = candidate_pool(ranked_df, row_id, "Condition", ARGS.joint_topk)
        name_pool = candidate_pool(ranked_df, row_id, "MeasureName", ARGS.joint_topk)
        mc_row, sa_row, cond_row, name_row = joint_decode_measure_bundle(mc_pool, sa_pool, cond_pool, name_pool)
        if mc_row is not None: provisional[(row_id, "MeasureCode")] = mc_row
        if sa_row is not None: provisional[(row_id, "Stateavg")] = sa_row
        if cond_row is not None: provisional[(row_id, "Condition")] = cond_row
        if name_row is not None: provisional[(row_id, "MeasureName")] = name_row
    for (row_id, col), chosen in sorted(provisional.items(), key=lambda x: (x[0][0], x[0][1])):
        dirty = canonical_empty_text(chosen.get("dirty_value", ""))
        final_value = canonical_empty_text(chosen.get("candidate_value", dirty))
        pool = raw_pools.get((row_id, col))
        second = pool.sort_values(["student_rank"], ascending=[True]).reset_index(drop=True).iloc[1] if pool is not None and len(pool) >= 2 else None
        passed, gate_reason = pass_column_gate(col, dirty, chosen, second, column_profiles)
        if not passed:
            final_value = dirty
            gate_passed = 0
            selected_rank = -1
            selected_source = "gate_keep_original"
            reason = "keep_original"
        else:
            gate_passed = 1
            selected_rank = int(chosen.get("student_rank", -1))
            selected_source = str(chosen.get("candidate_source", "joint_decode"))
            reason = str(chosen.get("student_reason_pred", "joint_decode"))
        top2_val = canonical_empty_text(second.get("candidate_value", "")) if second is not None else ""
        top2_score = float(second.get("student_score", 0.0)) if second is not None else None
        top_margin = float(chosen.get("student_score", 0.0)) - (float(second.get("student_score", 0.0)) if second is not None else 0.0)
        out_rows.append({
            "row_id": int(row_id),
            "column": str(col).strip(),
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
            "top2_raw_candidate": top2_val,
            "top2_raw_score": top2_score,
            "top_margin": top_margin
        })
    return pd.DataFrame(out_rows)


def build_hard_cases_jsonl(ranked_df: pd.DataFrame, max_hard_cases: int):
    rows = []
    for (row_id, column), g in ranked_df.groupby(["row_id", "column"], sort=False):
        g = g.sort_values(["student_rank"], ascending=[True]).reset_index(drop=True)
        if len(g) < 2:
            continue
        best = g.iloc[0]
        second = g.iloc[1]
        margin = float(best["student_score"]) - float(second["student_score"])
        is_hard = margin < ARGS.global_min_margin + 0.02
        if str(column) in {"MeasureCode", "Stateavg", "MeasureName", "Condition"} and margin < 0.08:
            is_hard = True
        if (str(column) in TIME_COLUMNS or str(column).endswith("_time")) and margin < 0.08:
            is_hard = True
        if not is_hard:
            continue
        rows.append({
            "row_id": int(row_id),
            "column": str(column).strip(),
            "dirty_value": canonical_empty_text(best["dirty_value"]),
            "candidate_a_value": canonical_empty_text(best["candidate_value"]),
            "candidate_b_value": canonical_empty_text(second["candidate_value"]),
            "candidate_a_score": float(best["student_score"]),
            "candidate_b_score": float(second["student_score"]),
            "top_margin": margin
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
    ensure_column_profiles_json(ARGS.candidate_features, COLUMN_PROFILES_JSON)
    for p in [CHECKPOINT_PATH, FEATURE_COLUMNS_JSON, REASON_LABELS_JSON, COLUMN_PROFILES_JSON]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file: {p}")

    trained_feature_columns = load_json(FEATURE_COLUMNS_JSON)
    reason_name_to_id = load_json(REASON_LABELS_JSON)
    column_profiles = load_json(COLUMN_PROFILES_JSON)
    reason_id_to_name = {int(v): k for k, v in reason_name_to_id.items()}
    allowed_cells = load_allowed_cells(ARGS.allowed_cells_csv)

    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    hidden_dims = ckpt.get("hidden_dims", [128, 64])
    dropout = ckpt.get("dropout", 0.15)
    model = SharedScorerMLP(len(trained_feature_columns), hidden_dims, dropout, len(reason_name_to_id)).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    cand_df = load_candidate_features_for_inference(
        ARGS.candidate_features,
        trained_feature_columns,
        allowed_cells=allowed_cells
    )
    print(f"[INFO] DEVICE = {DEVICE}")
    print(f"[INFO] Candidate rows for inference: {len(cand_df)}")

    scored_df = score_all_candidates(model, cand_df, trained_feature_columns, reason_id_to_name)
    scored_df.to_csv(OUTPUT_ALL_SCORES_CSV, index=False, encoding="utf-8-sig")

    ranked_df = rank_candidates(scored_df)
    ranked_df.to_csv(OUTPUT_ALL_RANKED_CSV, index=False, encoding="utf-8-sig")

    top1_df = build_top1_with_gate_and_joint_decoding(ranked_df, column_profiles)
    top1_df["dirty_value"] = top1_df["dirty_value"].map(canonical_empty_text)
    top1_df["student_best_candidate"] = top1_df["student_best_candidate"].map(canonical_empty_text)
    top1_df["column"] = top1_df["column"].astype(str).str.strip()

    mask = (
        top1_df["column"].isin(PROTECTED_NONEMPTY_COLUMNS)
        & (top1_df["dirty_value"] != "empty")
        & (top1_df["student_best_candidate"] == "empty")
    )
    if mask.any():
        top1_df.loc[mask, "student_best_candidate"] = top1_df.loc[mask, "dirty_value"]
        top1_df.loc[mask, "gate_passed"] = 0
        top1_df.loc[mask, "gate_reason"] = "final_output_nonempty_protection"
        top1_df.loc[mask, "selected_student_rank"] = -1
        top1_df.loc[mask, "selected_candidate_source"] = "final_output_keep_original"
        top1_df.loc[mask, "student_reason_pred"] = "keep_original"
        print(f"[INFO] Final-output hard-fix reverted {int(mask.sum())} dangerous empty predictions.")

    unique_output_cells = top1_df[["row_id", "column"]].drop_duplicates().shape[0]
    print(f"[INFO] output unique repaired cells = {unique_output_cells}")

    input_keys = set(
        (int(r["row_id"]), str(r["column"]).strip())
        for _, r in cand_df[["row_id", "column"]].drop_duplicates().iterrows()
    )
    bad_mask = top1_df.apply(
        lambda r: (int(r["row_id"]), str(r["column"]).strip()) not in input_keys,
        axis=1
    )
    bad_count = int(bad_mask.sum())
    print(f"[INFO] output keys not in inference input = {bad_count}")

    if bad_count > 0:
        top1_df.loc[bad_mask].to_csv(OUTPUT_WHITELIST_VIOLATIONS_CSV, index=False, encoding="utf-8-sig")
        print(f"[WARN] Output contains keys outside inference input. Saved to: {OUTPUT_WHITELIST_VIOLATIONS_CSV}")
        top1_df = top1_df.loc[~bad_mask].copy()

    top1_df.to_csv(OUTPUT_TOP1_CSV, index=False, encoding="utf-8-sig")
    hard_df = build_hard_cases_jsonl(ranked_df, ARGS.max_hard_cases)
    print(f"[OK] Top-1 repairs saved: {OUTPUT_TOP1_CSV}")
    print(f"[OK] Hard cases saved: {OUTPUT_HARD_CASES_JSONL}")
    print(f"[INFO] Hard case count: {len(hard_df)}")


if __name__ == "__main__":
    main()

import os
import json
import math
import random
import re
import argparse
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def parse_args():
    p = argparse.ArgumentParser(description="Train / resume pairwise LTR student")
    p.add_argument("--candidate_features", default="repair_candidate_features_v2.csv")
    p.add_argument("--pairwise_responses", default="repair_pairwise_responses_v2.jsonl")
    p.add_argument("--output_dir", default="pairwise_ltr_student_output_iter")
    p.add_argument("--resume_checkpoint", default=None)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


ARGS = parse_args()
RANDOM_SEED = ARGS.seed
VAL_RATIO = ARGS.val_ratio
BATCH_SIZE = ARGS.batch_size
NUM_EPOCHS = ARGS.epochs
LEARNING_RATE = ARGS.lr
WEIGHT_DECAY = ARGS.weight_decay
GRAD_CLIP_NORM = 5.0
EARLY_STOPPING_PATIENCE = 6

HIDDEN_DIMS = [128, 64]
DROPOUT = 0.15

PAIRWISE_LOSS_WEIGHT = 1.0
REASON_LOSS_WEIGHT = 0.2

USE_RULE_REGULARIZATION = True
RULE_REG_STRENGTH = 0.05

COLUMN_WEIGHT_MAP = {
    "Score": 2.5,
    "Sample": 2.2,
    "Stateavg": 2.5,
    "MeasureCode": 2.5,
    "MeasureName": 2.0,
    "Condition": 1.5,
}
DEFAULT_SAMPLE_WEIGHT = 1.0

OUTPUT_DIR = ARGS.output_dir
OUTPUT_PAIRWISE_TRAIN_CSV = os.path.join(OUTPUT_DIR, "pairwise_train_materialized.csv")
OUTPUT_TRAIN_LOG_CSV = os.path.join(OUTPUT_DIR, "training_log.csv")
OUTPUT_FEATURE_COLUMNS_JSON = os.path.join(OUTPUT_DIR, "feature_columns.json")
OUTPUT_REASON_LABELS_JSON = os.path.join(OUTPUT_DIR, "reason_label_mapping.json")
OUTPUT_COLUMN_PROFILES_JSON = os.path.join(OUTPUT_DIR, "column_profiles.json")
OUTPUT_BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pt")
OUTPUT_LAST_MODEL_PATH = os.path.join(OUTPUT_DIR, "last_model.pt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMPTY_TOKENS = {"", "empty", "null", "none", "nan", "na", "n/a", "missing"}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_jsonl(path: str):
    rows = []
    decoder = json.JSONDecoder()
    with open(path, "r", encoding="utf-8-sig") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
                elif isinstance(obj, list):
                    rows.extend([x for x in obj if isinstance(x, dict)])
                continue
            except json.JSONDecodeError:
                pass

            idx = 0
            parsed_any = False
            while idx < len(line):
                while idx < len(line) and line[idx] not in "{[":
                    idx += 1
                if idx >= len(line):
                    break
                try:
                    obj, end = decoder.raw_decode(line, idx)
                    parsed_any = True
                    if isinstance(obj, dict):
                        rows.append(obj)
                    elif isinstance(obj, list):
                        rows.extend([x for x in obj if isinstance(x, dict)])
                    idx = end
                except json.JSONDecodeError:
                    idx += 1
            if not parsed_any:
                print(f"[WARN] Skip unparsable line: {path}, line={line_no}")
    return rows


def norm_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def canonical_empty_text(x):
    s = norm_text(x).lower()
    return "empty" if s in EMPTY_TOKENS else norm_text(x)


def safe_float(x, default=0.0):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return default
        return float(x)
    except Exception:
        return default


def is_empty_token(x: str) -> int:
    return int(norm_text(x).lower() in EMPTY_TOKENS)


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



def infer_column_profiles(df: pd.DataFrame):
    profiles = {}
    for col, g in df.groupby("column"):
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
                "sample_values": [],
            }
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

        profiles[str(col)] = {
            "archetype": archetype,
            "avg_len": avg_len,
            "digits_ratio": digits_ratio,
            "percent_ratio": percent_ratio,
            "code_ratio": code_ratio,
            "alpha_ratio": alpha_ratio,
            "unique_count": unique_count,
            "sample_values": list(vals.head(20)),
        }
    return profiles


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

    for col_name in ["Score", "Sample", "Stateavg", "MeasureCode", "MeasureName", "Condition"]:
        out[f"is_col_{col_name}"] = (out["column"] == col_name).astype(int)
    return out


def load_candidate_features(path: str):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["row_id"] = df["row_id"].astype(int)
    df["column"] = df["column"].astype(str)
    df["candidate_value"] = df["candidate_value"].map(norm_text)
    df["dirty_value"] = df["dirty_value"].map(norm_text)
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

    full_feat_df = pd.concat([numeric_feature_df, df_cat], axis=1)
    for col in full_feat_df.columns:
        full_feat_df[col] = pd.to_numeric(full_feat_df[col], errors="coerce").fillna(0.0)

    feature_columns = full_feat_df.columns.tolist()
    df_out = df.copy()
    for col in feature_columns:
        df_out[col] = full_feat_df[col].values
    return df_out, feature_columns


def build_candidate_index(df: pd.DataFrame, feature_columns: List[str]):
    idx = {}
    for _, row in df.iterrows():
        key = (int(row["row_id"]), str(row["column"]), norm_text(row["candidate_value"]))
        idx[key] = {
            "features": row[feature_columns].astype(float).values.astype(np.float32),
            "rule_support": safe_float(row.get("candidate_rule_support_ratio", 0.0), 0.0),
            "sample_weight": float(COLUMN_WEIGHT_MAP.get(str(row["column"]), DEFAULT_SAMPLE_WEIGHT)),
        }
    return idx


REASON_LABEL_DEFAULT = "uncertain"


def normalize_winner(x: str) -> str:
    x = str(x).strip()
    if x not in {"A", "B", "Unknown"}:
        return "Unknown"
    return x


def load_pairwise_responses(path: str):
    rows = load_jsonl(path)
    cleaned = []
    skipped = 0
    for x in rows:
        if not isinstance(x, dict) or "row_id" not in x or "column" not in x:
            skipped += 1
            continue
        try:
            cleaned.append({
                "row_id": int(x["row_id"]),
                "column": str(x["column"]),
                "dirty_value": norm_text(x.get("dirty_value", "")),
                "candidate_a_value": norm_text(x.get("candidate_a_value", "")),
                "candidate_b_value": norm_text(x.get("candidate_b_value", "")),
                "winner": normalize_winner(x.get("winner", "Unknown")),
                "reason_type": str(x.get("reason_type", REASON_LABEL_DEFAULT)).strip() or REASON_LABEL_DEFAULT,
            })
        except Exception:
            skipped += 1
    print(f"[INFO] Loaded pairwise responses: {len(cleaned)}")
    print(f"[INFO] Skipped bad responses: {skipped}")
    return cleaned


def build_reason_label_map(pairwise_rows):
    labels = sorted(set(x["reason_type"] for x in pairwise_rows) | {REASON_LABEL_DEFAULT})
    return {lab: i for i, lab in enumerate(labels)}


def materialize_pairwise_training_table(pairwise_rows, cand_index, reason_map):
    rows = []
    skipped = 0
    for x in pairwise_rows:
        if x["winner"] == "Unknown":
            continue
        key_a = (x["row_id"], x["column"], x["candidate_a_value"])
        key_b = (x["row_id"], x["column"], x["candidate_b_value"])
        if key_a not in cand_index or key_b not in cand_index:
            skipped += 1
            continue
        rows.append({
            "row_id": x["row_id"],
            "column": x["column"],
            "dirty_value": x["dirty_value"],
            "candidate_a_value": x["candidate_a_value"],
            "candidate_b_value": x["candidate_b_value"],
            "label_a_better": 1 if x["winner"] == "A" else 0,
            "reason_type": x["reason_type"],
            "reason_label": reason_map.get(x["reason_type"], reason_map[REASON_LABEL_DEFAULT]),
        })
    df = pd.DataFrame(rows)
    print(f"[INFO] Pairwise teacher samples: {len(df)}")
    print(f"[INFO] Missing-candidate skipped: {skipped}")
    return df


def split_by_cell(df: pd.DataFrame, val_ratio: float, seed: int):
    unique_cells = df[["row_id", "column"]].drop_duplicates().copy()
    unique_cells = unique_cells.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_val = max(1, int(len(unique_cells) * val_ratio)) if len(unique_cells) > 1 else 0
    val_cells = set((int(r["row_id"]), str(r["column"])) for _, r in unique_cells.iloc[:n_val].iterrows())
    is_val = df.apply(lambda r: (int(r["row_id"]), str(r["column"])) in val_cells, axis=1)
    return df[~is_val].copy().reset_index(drop=True), df[is_val].copy().reset_index(drop=True)


class PairwiseRankingDataset(Dataset):
    def __init__(self, pairwise_df: pd.DataFrame, cand_index):
        self.df = pairwise_df.reset_index(drop=True)
        self.cand_index = cand_index

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        key_a = (int(row["row_id"]), str(row["column"]), norm_text(row["candidate_a_value"]))
        key_b = (int(row["row_id"]), str(row["column"]), norm_text(row["candidate_b_value"]))
        sample_weight = max(
            self.cand_index[key_a]["sample_weight"],
            self.cand_index[key_b]["sample_weight"]
        )
        return {
            "feat_a": torch.tensor(self.cand_index[key_a]["features"], dtype=torch.float32),
            "feat_b": torch.tensor(self.cand_index[key_b]["features"], dtype=torch.float32),
            "label": torch.tensor(float(row["label_a_better"]), dtype=torch.float32),
            "reason_label": torch.tensor(int(row["reason_label"]), dtype=torch.long),
            "rule_a": torch.tensor(float(self.cand_index[key_a]["rule_support"]), dtype=torch.float32),
            "rule_b": torch.tensor(float(self.cand_index[key_b]["rule_support"]), dtype=torch.float32),
            "sample_weight": torch.tensor(float(sample_weight), dtype=torch.float32),
        }


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

    def forward(self, feat_a, feat_b):
        score_a, reason_a, _ = self.forward_single(feat_a)
        score_b, reason_b, _ = self.forward_single(feat_b)
        return {"score_a": score_a, "score_b": score_b, "reason_a": reason_a, "reason_b": reason_b}


@dataclass
class Metrics:
    loss: float
    pair_acc: float
    pair_count: int


def compute_loss(batch, outputs, ce_reason, bce_pair):
    label = batch["label"]
    sw = batch["sample_weight"]
    score_a = outputs["score_a"]
    score_b = outputs["score_b"]

    pair_logit = score_a - score_b
    pair_loss = (bce_pair(pair_logit, label) * sw).mean()

    reason_label = batch["reason_label"]
    use_a_mask = (label > 0.5)
    use_b_mask = ~use_a_mask
    reason_losses = []
    if use_a_mask.any():
        reason_losses.append(ce_reason(outputs["reason_a"][use_a_mask], reason_label[use_a_mask]))
    if use_b_mask.any():
        reason_losses.append(ce_reason(outputs["reason_b"][use_b_mask], reason_label[use_b_mask]))
    reason_loss = sum(reason_losses) / len(reason_losses) if reason_losses else torch.tensor(0.0, device=pair_logit.device)

    total_loss = PAIRWISE_LOSS_WEIGHT * pair_loss + REASON_LOSS_WEIGHT * reason_loss

    if USE_RULE_REGULARIZATION:
        desired = torch.sign(batch["rule_a"] - batch["rule_b"])
        reg = torch.relu(-desired * pair_logit)
        mask = (desired != 0).float()
        if mask.sum() > 0:
            reg_loss = (reg * mask * sw).sum() / (mask * sw).sum().clamp_min(1e-8)
            total_loss = total_loss + RULE_REG_STRENGTH * reg_loss
    return total_loss


def run_epoch(model, loader, optimizer=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    ce_reason = nn.CrossEntropyLoss()
    bce_pair = nn.BCEWithLogitsLoss(reduction="none")

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch in loader:
        feat_a = batch["feat_a"].to(DEVICE)
        feat_b = batch["feat_b"].to(DEVICE)
        label = batch["label"].to(DEVICE)
        batch_gpu = {
            "label": label,
            "reason_label": batch["reason_label"].to(DEVICE),
            "rule_a": batch["rule_a"].to(DEVICE),
            "rule_b": batch["rule_b"].to(DEVICE),
            "sample_weight": batch["sample_weight"].to(DEVICE),
        }

        with torch.set_grad_enabled(train_mode):
            outputs = model(feat_a, feat_b)
            loss = compute_loss(batch_gpu, outputs, ce_reason, bce_pair)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()

        pred = (torch.sigmoid(outputs["score_a"] - outputs["score_b"]) >= 0.5).float()
        total_correct += (pred == label).sum().item()
        total_count += label.numel()
        total_loss += loss.item() * label.numel()

    return Metrics(total_loss / max(total_count, 1), total_correct / max(total_count, 1), total_count)


def save_checkpoint(path: str, model, optimizer, epoch: int, best_val_loss: float, feature_columns, reason_map):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "feature_columns": feature_columns,
        "reason_map": reason_map,
        "hidden_dims": HIDDEN_DIMS,
        "dropout": DROPOUT,
    }, path)


def load_checkpoint(path: str, model, optimizer=None):
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


def main():
    set_seed(RANDOM_SEED)
    ensure_dir(OUTPUT_DIR)
    print(f"[INFO] DEVICE = {DEVICE}")

    cand_df, feature_columns = load_candidate_features(ARGS.candidate_features)
    train_unique_cells = cand_df[["row_id", "column"]].drop_duplicates().shape[0]
    print(f"[INFO] training candidate unique cells = {train_unique_cells}")

    column_profiles = infer_column_profiles(cand_df)
    cand_index = build_candidate_index(cand_df, feature_columns)

    pairwise_rows = load_pairwise_responses(ARGS.pairwise_responses)
    reason_map = build_reason_label_map(pairwise_rows)
    pairwise_df = materialize_pairwise_training_table(pairwise_rows, cand_index, reason_map)
    if len(pairwise_df) == 0:
        raise ValueError("No valid pairwise teacher samples.")

    pairwise_df.to_csv(OUTPUT_PAIRWISE_TRAIN_CSV, index=False, encoding="utf-8-sig")
    train_df, val_df = split_by_cell(pairwise_df, VAL_RATIO, RANDOM_SEED)
    print(f"[INFO] train pairs = {len(train_df)}, val pairs = {len(val_df)}")

    train_ds = PairwiseRankingDataset(train_df, cand_index)
    val_ds = PairwiseRankingDataset(val_df, cand_index)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = SharedScorerMLP(len(feature_columns), HIDDEN_DIMS, DROPOUT, len(reason_map)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    start_epoch = 1
    best_val_loss = float("inf")
    if ARGS.resume_checkpoint and os.path.exists(ARGS.resume_checkpoint):
        ckpt = load_checkpoint(ARGS.resume_checkpoint, model, optimizer)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        print(f"[INFO] Resume from {ARGS.resume_checkpoint}, start_epoch={start_epoch}, best_val_loss={best_val_loss:.6f}")

    with open(OUTPUT_FEATURE_COLUMNS_JSON, "w", encoding="utf-8-sig") as f:
        json.dump(feature_columns, f, ensure_ascii=False, indent=2)
    with open(OUTPUT_REASON_LABELS_JSON, "w", encoding="utf-8-sig") as f:
        json.dump(reason_map, f, ensure_ascii=False, indent=2)
    with open(OUTPUT_COLUMN_PROFILES_JSON, "w", encoding="utf-8-sig") as f:
        json.dump(column_profiles, f, ensure_ascii=False, indent=2)

    logs = []
    patience = 0
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        train_metrics = run_epoch(model, train_loader, optimizer=optimizer)
        val_metrics = run_epoch(model, val_loader, optimizer=None) if len(val_ds) > 0 else Metrics(0.0, 0.0, 0)
        logs.append({
            "epoch": epoch,
            "train_loss": train_metrics.loss,
            "train_pair_acc": train_metrics.pair_acc,
            "train_pair_count": train_metrics.pair_count,
            "val_loss": val_metrics.loss,
            "val_pair_acc": val_metrics.pair_acc,
            "val_pair_count": val_metrics.pair_count,
        })
        print(f"[EPOCH {epoch:03d}] train_loss={train_metrics.loss:.6f} train_acc={train_metrics.pair_acc:.4f} | val_loss={val_metrics.loss:.6f} val_acc={val_metrics.pair_acc:.4f}")

        save_checkpoint(OUTPUT_LAST_MODEL_PATH, model, optimizer, epoch, best_val_loss, feature_columns, reason_map)

        if val_metrics.loss < best_val_loss:
            best_val_loss = val_metrics.loss
            patience = 0
            save_checkpoint(OUTPUT_BEST_MODEL_PATH, model, optimizer, epoch, best_val_loss, feature_columns, reason_map)
            print(f"[INFO] New best model saved -> {OUTPUT_BEST_MODEL_PATH}")
        else:
            patience += 1
            if patience >= EARLY_STOPPING_PATIENCE:
                print(f"[INFO] Early stopping triggered at epoch {epoch}")
                break

    pd.DataFrame(logs).to_csv(OUTPUT_TRAIN_LOG_CSV, index=False, encoding="utf-8-sig")
    print(f"[OK] Training log saved: {OUTPUT_TRAIN_LOG_CSV}")
    print(f"[OK] Best checkpoint: {OUTPUT_BEST_MODEL_PATH}")
    print(f"[OK] Last checkpoint: {OUTPUT_LAST_MODEL_PATH}")


if __name__ == "__main__":
    main()

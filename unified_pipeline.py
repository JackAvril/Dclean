#!/usr/bin/env python3
"""Unified statistical data-cleaning pipeline for six target datasets.

This module is intentionally self-contained: the detection and correction stage
logic lives here, while each dataset only supplies configuration such as input
paths, output filenames, and a small number of thresholds.  The goal is to make
hospital, beers, flights, rayyan, adult, and soccer pass through the same code
path while keeping legacy-compatible output names whenever possible.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

pd = None


REPO_ROOT = Path(__file__).resolve().parent
TARGET_DATASETS: tuple[str, ...] = ("hospital", "beers", "flights", "rayyan", "adult", "soccer")
TASKS: tuple[str, ...] = ("detection", "correction")

MISSING_TOKENS = {
    "", "empty", "null", "none", "nan", "na", "n/a", "missing", "unk", "unknown", "not available"
}


@dataclass(frozen=True)
class Stage:
    name: str
    description: str
    llm: bool = False
    optional: bool = False


@dataclass(frozen=True)
class DatasetConfig:
    dataset: str
    dirty_csv: str
    clean_csv: str | None = None
    detection_dir: str | None = None
    correction_dir: str | None = None
    enum_unique_threshold: int = 50
    enum_ratio_threshold: float = 0.08
    fd_confidence_threshold: float = 0.98
    rare_value_max_count: int = 1
    rare_value_max_ratio: float = 0.002
    candidate_score_threshold: float = 0.75
    repair_min_score: float = 0.0
    detection_outputs: dict[str, str] = field(default_factory=dict)
    correction_outputs: dict[str, str] = field(default_factory=dict)
    disabled_stages: dict[str, set[str]] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def task_dir(self, task: str) -> Path:
        if task == "detection":
            return REPO_ROOT / (self.detection_dir or f"Error Detection {self.dataset}")
        if task == "correction":
            return REPO_ROOT / (self.correction_dir or f"Error Correction {self.dataset}")
        raise PipelineError(f"Unknown task: {task}")

    def output_name(self, task: str, key: str, default: str) -> str:
        outputs = self.detection_outputs if task == "detection" else self.correction_outputs
        return outputs.get(key, default)

    def disabled_for(self, task: str) -> set[str]:
        return self.disabled_stages.get(task, set())


@dataclass(frozen=True)
class PipelineConfig:
    task: str
    dataset: str
    workdir: Path
    stages: tuple[Stage, ...]
    dataset_config: DatasetConfig


DETECTION_FLOW: tuple[Stage, ...] = (
    Stage("build_rule_pool", "learn column profiles, pattern rules, rare-value rules, and FD-like rules"),
    Stage("shrink_search_space", "score suspicious cells from the unified rule pool"),
    Stage("summarize_full_contexts", "build rich JSONL and flat CSV context for suspicious cells"),
    Stage("cluster_and_sample", "assign buckets/clusters and choose representative cells"),
    Stage("annotate", "produce deterministic statistical labels for sampled cells", llm=True),
    Stage("label_propagation", "propagate sampled labels back to the candidate set"),
    Stage("build_final_training_set", "build a detector training table compatible with legacy filenames"),
    Stage("build_loading_set", "build detector inference input"),
    Stage("detector_loop", "run the unified statistical detector and write prediction outputs", llm=True),
    Stage("evaluate", "evaluate detection against clean/dirty tables when clean data exists", optional=True),
)

CORRECTION_FLOW: tuple[Stage, ...] = (
    Stage("wrong_position", "derive repair-allowed cells from detector predictions or clean/dirty diff", optional=True),
    Stage("revise_candidates", "generate repair candidates from column values and FD-like context"),
    Stage("build_features", "build unified repair candidate features"),
    Stage("build_infer_whitelist_features", "filter repair candidates to allowed cells"),
    Stage("llm_ranking", "produce deterministic teacher-style candidate rankings", llm=True),
    Stage("repair_loop", "rank candidates and emit top-1 repairs", llm=True),
    Stage("evaluate", "evaluate repairs against clean data when available", optional=True),
)

CANONICAL_FLOWS: dict[str, tuple[Stage, ...]] = {
    "detection": DETECTION_FLOW,
    "correction": CORRECTION_FLOW,
}


def legacy_outputs(dataset: str) -> tuple[dict[str, str], dict[str, str]]:
    detection = {
        "rule_pool": f"rule_pool_{dataset}.json",
        "candidate_jsonl": f"candidate_cells_{dataset}.jsonl",
        "candidate_csv": f"candidate_cells_{dataset}.csv",
        "contexts_jsonl": f"candidate_llm_contexts_{dataset}.jsonl",
        "contexts_csv": f"candidate_llm_contexts_flat_{dataset}.csv",
        "clustered_csv": f"candidate_clustered_{dataset}.csv",
        "sampled_csv": f"candidate_sampled_{dataset}.csv",
        "sampled_jsonl": f"candidate_sampled_with_context_{dataset}.jsonl",
        "labeled_jsonl": f"candidate_sampled_labeled_{dataset}.jsonl",
        "labeled_csv": f"candidate_sampled_labeled_{dataset}.csv",
        "propagated_jsonl": f"propagated_labels_{dataset}.jsonl",
        "propagated_csv": f"propagated_labels_{dataset}.csv",
        "final_train_csv": f"final_train_dataset_{dataset}.csv",
        "final_train_jsonl": f"final_train_dataset_{dataset}.jsonl",
        "final_summary_json": f"final_train_summary_{dataset}.json",
        "infer_ready_csv": f"candidate_infer_ready_{dataset}.csv",
        "detector_output_dir": f"unified_detector_output_{dataset}",
        "eval_output_dir": f"eval_outputs_{dataset}",
    }
    correction = {
        "predicted_positions": "predicted_error_positions.csv",
        "candidates_expanded": "repair_candidates_expanded.csv",
        "candidates_grouped": "repair_candidates_grouped.csv",
        "candidate_features": "repair_candidate_features_v2.csv",
        "infer_features": "repair_candidate_features_infer_whitelist.csv",
        "pairwise_requests": "repair_pairwise_requests_v2.jsonl",
        "pairwise_responses": "repair_pairwise_responses_v2.jsonl",
        "teacher_top1": "repair_teacher_top1_labels_v2.csv",
        "repair_output_dir": f"unified_repair_output_{dataset}",
        "eval_output_dir": f"repair_eval_outputs_{dataset}",
    }
    return detection, correction


def make_dataset_configs() -> dict[str, DatasetConfig]:
    configs: dict[str, DatasetConfig] = {}
    dirty_paths = {
        "hospital": "/mnt/mydata/dq/projects/Splittree/hospital/hospital_dirty_no_address23.csv",
        "beers": "/mnt/mydata/dq/projects/Splittree/beers/beers_dirty.csv",
        "flights": "/mnt/mydata/dq/projects/Splittree/flights/flights_dirty.csv",
        "rayyan": "/mnt/mydata/dq/projects/Splittree/rayyan/rayyan_dirty.csv",
        "adult": "/mnt/mydata/dq/projects/Splittree/adult/adult_dirty.csv",
        "soccer": "/mnt/mydata/dq/projects/Splittree/soccer/soccer_dirty.csv",
    }
    clean_paths = {
        "hospital": "/mnt/mydata/dq/projects/Splittree/hospital/hospital_clean_no_address23.csv",
        "beers": "/mnt/mydata/dq/projects/Splittree/beers/beers_clean.csv",
        "flights": "/mnt/mydata/dq/projects/Splittree/flights/flights_clean.csv",
        "rayyan": "/mnt/mydata/dq/projects/Splittree/rayyan/rayyan_clean.csv",
        "adult": "/mnt/mydata/dq/projects/Splittree/adult/adult_clean.csv",
        "soccer": "/mnt/mydata/dq/projects/Splittree/soccer/soccer_clean.csv",
    }
    dirs = {
        "hospital": ("Error Detection hospital", "Error Correction hospital"),
        "beers": ("Error Detection beers", "Error Correction beers"),
        "flights": ("Error Detection flights", "Error Correction flights"),
        "rayyan": ("Error Detection rayyan", "Error Correction rayyan"),
        "adult": ("Error Detection Adult", "Error Correction Adult"),
        "soccer": ("Error Detection soccer", "Error Correction soccer"),
    }
    for dataset in TARGET_DATASETS:
        det, corr = legacy_outputs(dataset)
        disabled: dict[str, set[str]] = {}
        if dataset == "rayyan":
            disabled["detection"] = {"evaluate"}
        if dataset == "hospital":
            det.update({
                "rule_pool": "rule_pool_high_recall_v2.json",
                "candidate_jsonl": "candidate_cells_v2.jsonl",
                "candidate_csv": "candidate_cells_v2.csv",
                "contexts_jsonl": "candidate_llm_contexts_v2.jsonl",
                "contexts_csv": "candidate_llm_contexts_flat_v2.csv",
                "clustered_csv": "candidate_clustered_v2.csv",
                "sampled_csv": "candidate_sampled_v2.csv",
                "sampled_jsonl": "candidate_sampled_with_context_v2.jsonl",
                "labeled_jsonl": "candidate_sampled_labeled_v2.jsonl",
                "labeled_csv": "candidate_sampled_labeled_v2.csv",
                "propagated_jsonl": "propagated_labels_v2.jsonl",
                "propagated_csv": "propagated_labels_v2.csv",
                "final_train_csv": "final_train_dataset_simple_v2.csv",
                "final_train_jsonl": "final_train_dataset_simple_v2.jsonl",
                "final_summary_json": "final_train_summary_simple_v2.json",
                "infer_ready_csv": "candidate_infer_ready_v2.csv",
                "detector_output_dir": "unified_detector_output_v2",
                "eval_output_dir": "eval_outputs_unified",
            })
            disabled["correction"] = {"build_infer_whitelist_features"}
        configs[dataset] = DatasetConfig(
            dataset=dataset,
            dirty_csv=dirty_paths[dataset],
            clean_csv=clean_paths[dataset],
            detection_dir=dirs[dataset][0],
            correction_dir=dirs[dataset][1],
            enum_unique_threshold={"flights": 30, "beers": 40, "adult": 80}.get(dataset, 50),
            enum_ratio_threshold={"flights": 0.05, "adult": 0.10}.get(dataset, 0.08),
            candidate_score_threshold={"hospital": 0.70, "adult": 0.72}.get(dataset, 0.75),
            detection_outputs=det,
            correction_outputs=corr,
            disabled_stages=disabled,
        )
    return configs


DATASETS = make_dataset_configs()


class PipelineError(RuntimeError):
    pass


def require_pandas():
    global pd
    if pd is None:
        try:
            import pandas as _pd
        except ModuleNotFoundError as exc:
            raise PipelineError("pandas is required to run data-cleaning stages; install pandas or use plan/list without running stages") from exc
        pd = _pd
    return pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_value(x: Any) -> str:
    if x is None:
        return ""
    if pd is not None and pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in MISSING_TOKENS:
        return ""
    return s


def norm_key(x: Any) -> str:
    return normalize_value(x).lower()


def is_missing(x: Any) -> bool:
    return normalize_value(x) == ""


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or (pd is not None and pd.isna(x)):
            return default
        return float(x)
    except Exception:
        return default


def detect_pattern(value: Any) -> str:
    s = normalize_value(value)
    if not s:
        return "MISSING"
    chars = []
    for ch in s:
        if ch.isdigit():
            chars.append("D")
        elif ch.isalpha():
            chars.append("A")
        elif ch.isspace():
            chars.append("S")
        else:
            chars.append(ch)
    compact = []
    for ch in chars:
        if not compact or compact[-1] != ch:
            compact.append(ch)
    return "".join(compact)[:80]


def read_table(path: str | Path) -> pd.DataFrame:
    require_pandas()
    path = Path(path)
    if not path.exists():
        raise PipelineError(f"Input table not found: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: Sequence[dict[str, Any]] | pd.DataFrame) -> None:
    require_pandas()
    ensure_dir(path.parent)
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    df.to_csv(path, index=False, encoding="utf-8-sig")


def out_path(cfg: DatasetConfig, task: str, key: str, default: str) -> Path:
    return cfg.task_dir(task) / cfg.output_name(task, key, default)


def detector_output_dir(cfg: DatasetConfig) -> Path:
    return out_path(cfg, "detection", "detector_output_dir", "unified_detector_output")


def repair_output_dir(cfg: DatasetConfig) -> Path:
    return out_path(cfg, "correction", "repair_output_dir", "unified_repair_output")


def column_profiles(df: pd.DataFrame, cfg: DatasetConfig) -> dict[str, dict[str, Any]]:
    profiles = {}
    n = max(len(df), 1)
    for col in df.columns:
        values = df[col].map(normalize_value)
        non_missing = values[values != ""]
        counts = Counter(non_missing)
        patterns = Counter(non_missing.map(detect_pattern))
        numeric = pd.to_numeric(non_missing, errors="coerce")
        numeric_ratio = float(numeric.notna().mean()) if len(non_missing) else 0.0
        unique_count = int(non_missing.nunique())
        unique_ratio = unique_count / max(len(non_missing), 1)
        dominant_pattern, dominant_pattern_count = (patterns.most_common(1)[0] if patterns else ("", 0))
        top_values = [{"value": k, "count": int(v)} for k, v in counts.most_common(20)]
        profiles[col] = {
            "column": col,
            "row_count": int(n),
            "missing_count": int((values == "").sum()),
            "non_missing_count": int(len(non_missing)),
            "missing_ratio": float((values == "").mean()),
            "unique_count": unique_count,
            "unique_ratio": float(unique_ratio),
            "numeric_ratio": numeric_ratio,
            "is_numeric": numeric_ratio >= 0.9,
            "is_enum": unique_count <= cfg.enum_unique_threshold or unique_ratio <= cfg.enum_ratio_threshold,
            "dominant_pattern": dominant_pattern,
            "dominant_pattern_ratio": dominant_pattern_count / max(len(non_missing), 1),
            "top_values": top_values,
            "value_counts": dict(counts),
        }
    return profiles


def build_fd_rules(df: pd.DataFrame, profiles: dict[str, dict[str, Any]], cfg: DatasetConfig) -> list[dict[str, Any]]:
    rules = []
    cols = list(df.columns)
    for lhs in cols:
        lhs_values = df[lhs].map(normalize_value)
        if lhs_values.nunique() > 5000:
            continue
        for rhs in cols:
            if lhs == rhs:
                continue
            groups: dict[str, Counter[str]] = defaultdict(Counter)
            support = 0
            for _, row in df[[lhs, rhs]].iterrows():
                lval = normalize_value(row[lhs])
                rval = normalize_value(row[rhs])
                if lval and rval:
                    groups[lval][rval] += 1
                    support += 1
            if support < 10:
                continue
            consistent = sum(counter.most_common(1)[0][1] for counter in groups.values())
            confidence = consistent / max(support, 1)
            if confidence >= cfg.fd_confidence_threshold:
                mapping = {k: v.most_common(1)[0][0] for k, v in groups.items() if v}
                rules.append({
                    "rule_id": f"fd::{lhs}->{rhs}",
                    "rule_type": "functional_dependency",
                    "lhs": [lhs],
                    "rhs": rhs,
                    "confidence": round(confidence, 6),
                    "support": int(support),
                    "mapping": mapping,
                    "priority": "high",
                })
    return rules


def build_rule_pool_stage(cfg: DatasetConfig) -> None:
    df = read_table(cfg.dirty_csv)
    profiles = column_profiles(df, cfg)
    schema_rules = []
    value_rules = []
    for col, profile in profiles.items():
        if profile["missing_ratio"] < 0.05 and profile["non_missing_count"] >= 10:
            schema_rules.append({
                "rule_id": f"not_null::{col}",
                "rule_type": "not_null",
                "column": col,
                "confidence": 1.0 - profile["missing_ratio"],
                "priority": "medium",
            })
        if profile["dominant_pattern_ratio"] >= 0.75 and profile["dominant_pattern"]:
            schema_rules.append({
                "rule_id": f"pattern::{col}",
                "rule_type": "pattern",
                "column": col,
                "pattern": profile["dominant_pattern"],
                "confidence": profile["dominant_pattern_ratio"],
                "priority": "medium",
            })
        total = max(profile["non_missing_count"], 1)
        rare_values = [
            value for value, count in profile["value_counts"].items()
            if count <= cfg.rare_value_max_count and count / total <= cfg.rare_value_max_ratio
        ][:100]
        if rare_values:
            value_rules.append({
                "rule_id": f"rare::{col}",
                "rule_type": "rare_value",
                "column": col,
                "rare_values": rare_values,
                "priority": "low",
            })
    rule_pool = {
        "dataset": cfg.dataset,
        "profiles": profiles,
        "schema_rules": schema_rules,
        "value_rules": value_rules,
        "fd_rules": build_fd_rules(df, profiles, cfg),
    }
    write_json(out_path(cfg, "detection", "rule_pool", "rule_pool.json"), rule_pool)


def priority_weight(priority: str) -> float:
    return {"high": 1.5, "medium": 1.0, "low": 0.5}.get(str(priority), 1.0)


def add_violation(cell_map: dict[tuple[int, str], dict[str, Any]], row_id: int, col: str, value: str, violation: dict[str, Any]) -> None:
    key = (int(row_id), col)
    item = cell_map.setdefault(key, {
        "row_id": int(row_id),
        "column": col,
        "value": value,
        "violated_rules": [],
        "conflict_score": 0.0,
    })
    item["violated_rules"].append(violation)
    item["conflict_score"] += priority_weight(violation.get("priority", "medium")) * safe_float(violation.get("confidence", 1.0), 1.0)


def shrink_search_space_stage(cfg: DatasetConfig) -> None:
    df = read_table(cfg.dirty_csv)
    pool = read_json(out_path(cfg, "detection", "rule_pool", "rule_pool.json"))
    cell_map: dict[tuple[int, str], dict[str, Any]] = {}
    schema_rules = pool.get("schema_rules", [])
    value_rules = pool.get("value_rules", [])
    fd_rules = pool.get("fd_rules", [])
    profiles = pool.get("profiles", {})

    for idx, row in df.iterrows():
        for rule in schema_rules:
            col = rule.get("column")
            if col not in df.columns:
                continue
            value = normalize_value(row[col])
            if rule["rule_type"] == "not_null" and not value:
                add_violation(cell_map, idx, col, value, {**rule, "reason": "missing value in mostly non-null column"})
            if rule["rule_type"] == "pattern" and value and detect_pattern(value) != rule.get("pattern"):
                add_violation(cell_map, idx, col, value, {**rule, "reason": "value pattern differs from dominant column pattern"})
        for rule in value_rules:
            col = rule.get("column")
            if col not in df.columns:
                continue
            value = normalize_value(row[col])
            if value and value in set(rule.get("rare_values", [])):
                add_violation(cell_map, idx, col, value, {**rule, "confidence": 0.5, "reason": "rare value"})
        for rule in fd_rules:
            lhs = rule.get("lhs", [])
            rhs = rule.get("rhs")
            if not lhs or rhs not in df.columns or lhs[0] not in df.columns:
                continue
            lval = normalize_value(row[lhs[0]])
            rval = normalize_value(row[rhs])
            expected = rule.get("mapping", {}).get(lval)
            if lval and rval and expected and rval != expected:
                add_violation(cell_map, idx, rhs, rval, {**rule, "expected": expected, "reason": f"expected {rhs}={expected} from {lhs[0]}={lval}"})

    candidates = list(cell_map.values())
    for item in candidates:
        item["violation_count"] = len(item["violated_rules"])
        item["conflict_score"] = round(float(item["conflict_score"]), 6)
        main_rule = item["violated_rules"][0] if item["violated_rules"] else {}
        item["main_rule_type"] = main_rule.get("rule_type", "")
        item["label_binary"] = int(item["conflict_score"] >= cfg.candidate_score_threshold)
    candidates.sort(key=lambda x: (x["conflict_score"], x["violation_count"]), reverse=True)
    write_jsonl(out_path(cfg, "detection", "candidate_jsonl", "candidate_cells.jsonl"), candidates)
    flat = []
    for item in candidates:
        flat.append({
            "row_id": item["row_id"],
            "column": item["column"],
            "value": item["value"],
            "violation_count": item["violation_count"],
            "conflict_score": item["conflict_score"],
            "main_rule_type": item["main_rule_type"],
            "rule_types": "|".join(v.get("rule_type", "") for v in item["violated_rules"]),
            "reasons": " || ".join(v.get("reason", "") for v in item["violated_rules"]),
        })
    write_csv(out_path(cfg, "detection", "candidate_csv", "candidate_cells.csv"), flat)


def row_similarity(row_a: pd.Series, row_b: pd.Series, target_col: str) -> float:
    cols = [c for c in row_a.index if c != target_col]
    if not cols:
        return 0.0
    same = sum(norm_key(row_a[c]) == norm_key(row_b[c]) and norm_key(row_a[c]) != "" for c in cols)
    return same / len(cols)


def summarize_full_contexts_stage(cfg: DatasetConfig) -> None:
    df = read_table(cfg.dirty_csv)
    pool = read_json(out_path(cfg, "detection", "rule_pool", "rule_pool.json"))
    candidates = read_jsonl(out_path(cfg, "detection", "candidate_jsonl", "candidate_cells.jsonl"))
    profiles = pool.get("profiles", {})
    detail_rows = []
    flat_rows = []
    for cand in candidates:
        row_id = int(cand["row_id"])
        col = cand["column"]
        value = cand.get("value", "")
        profile = profiles.get(col, {})
        counts = profile.get("value_counts", {})
        sorted_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        rank = next((i + 1 for i, (v, _) in enumerate(sorted_counts) if v == value), None)
        same_col_values = [v for v, _ in sorted_counts[:10]]
        similar_values = []
        if 0 <= row_id < len(df):
            row = df.iloc[row_id]
            sims = []
            for other_idx, other in df.iterrows():
                if int(other_idx) == row_id:
                    continue
                sim = row_similarity(row, other, col)
                if sim > 0:
                    sims.append((sim, normalize_value(other[col])))
            sims.sort(reverse=True)
            similar_values = [v for _, v in sims[:10] if v]
        neighbor_counter = Counter(similar_values)
        neighbor_majority, neighbor_count = (neighbor_counter.most_common(1)[0] if neighbor_counter else ("", 0))
        neighbor_ratio = neighbor_count / max(len(similar_values), 1)
        detail = {
            "row_id": row_id,
            "column": col,
            "value": value,
            "row_context": df.iloc[row_id].to_dict() if 0 <= row_id < len(df) else {},
            "column_profile": {k: v for k, v in profile.items() if k != "value_counts"},
            "top_column_values": same_col_values,
            "similar_row_values": similar_values,
            "neighbor_majority_value": neighbor_majority,
            "neighbor_majority_ratio": neighbor_ratio,
            "violated_rules": cand.get("violated_rules", []),
            "conflict_score": cand.get("conflict_score", 0.0),
        }
        detail_rows.append(detail)
        flat_rows.append({
            "row_id": row_id,
            "column": col,
            "value": value,
            "violation_count": cand.get("violation_count", 0),
            "conflict_score": cand.get("conflict_score", 0.0),
            "main_rule_type": cand.get("main_rule_type", ""),
            "value_frequency": counts.get(value, 0),
            "value_frequency_rank": rank,
            "neighbor_majority_value": neighbor_majority,
            "neighbor_majority_ratio": neighbor_ratio,
            "top_column_values": " || ".join(same_col_values),
        })
    write_jsonl(out_path(cfg, "detection", "contexts_jsonl", "candidate_llm_contexts.jsonl"), detail_rows)
    write_csv(out_path(cfg, "detection", "contexts_csv", "candidate_llm_contexts_flat.csv"), flat_rows)


def cluster_and_sample_stage(cfg: DatasetConfig) -> None:
    require_pandas()
    flat_path = out_path(cfg, "detection", "contexts_csv", "candidate_llm_contexts_flat.csv")
    details = { (r["row_id"], r["column"]): r for r in read_jsonl(out_path(cfg, "detection", "contexts_jsonl", "candidate_llm_contexts.jsonl")) }
    df = pd.read_csv(flat_path, dtype=str, keep_default_na=False, encoding="utf-8-sig") if flat_path.exists() else pd.DataFrame()
    if df.empty:
        write_csv(out_path(cfg, "detection", "clustered_csv", "candidate_clustered.csv"), df)
        write_csv(out_path(cfg, "detection", "sampled_csv", "candidate_sampled.csv"), df)
        write_jsonl(out_path(cfg, "detection", "sampled_jsonl", "candidate_sampled_with_context.jsonl"), [])
        return
    df["bucket_id"] = df["column"].astype(str) + "::" + df["main_rule_type"].astype(str)
    df["cluster_id"] = df.groupby("bucket_id").cumcount() // 25
    df["dist_to_center"] = 0.0
    df["sample_role"] = "all_candidates"
    sampled = df.groupby("bucket_id", group_keys=False).head(50).copy()
    write_csv(out_path(cfg, "detection", "clustered_csv", "candidate_clustered.csv"), df)
    write_csv(out_path(cfg, "detection", "sampled_csv", "candidate_sampled.csv"), sampled)
    json_rows = []
    for _, row in sampled.iterrows():
        key = (int(row["row_id"]), row["column"])
        json_rows.append({"summary": row.to_dict(), "detail_context": details.get(key, {})})
    write_jsonl(out_path(cfg, "detection", "sampled_jsonl", "candidate_sampled_with_context.jsonl"), json_rows)


def annotate_stage(cfg: DatasetConfig) -> None:
    require_pandas()
    sample_path = out_path(cfg, "detection", "sampled_csv", "candidate_sampled.csv")
    df = pd.read_csv(sample_path, dtype=str, keep_default_na=False, encoding="utf-8-sig") if sample_path.exists() else pd.DataFrame()
    rows = []
    for _, row in df.iterrows():
        score = safe_float(row.get("conflict_score"), 0.0)
        neighbor_ratio = safe_float(row.get("neighbor_majority_ratio"), 0.0)
        label = int(score >= cfg.candidate_score_threshold and neighbor_ratio < 0.95)
        rows.append({**row.to_dict(), "label_binary": label, "label": "error" if label else "correct", "confidence": min(0.99, max(0.5, score / 3.0)), "label_source": "unified_statistical"})
    write_jsonl(out_path(cfg, "detection", "labeled_jsonl", "candidate_sampled_labeled.jsonl"), rows)
    write_csv(out_path(cfg, "detection", "labeled_csv", "candidate_sampled_labeled.csv"), rows)


def label_propagation_stage(cfg: DatasetConfig) -> None:
    require_pandas()
    cand = pd.read_csv(out_path(cfg, "detection", "contexts_csv", "candidate_llm_contexts_flat.csv"), dtype=str, keep_default_na=False, encoding="utf-8-sig")
    labeled_path = out_path(cfg, "detection", "labeled_csv", "candidate_sampled_labeled.csv")
    labeled = pd.read_csv(labeled_path, dtype=str, keep_default_na=False, encoding="utf-8-sig") if labeled_path.exists() else pd.DataFrame()
    label_map = {(int(r.row_id), r.column): int(safe_float(r.label_binary, 0)) for r in labeled.itertuples()} if not labeled.empty else {}
    rows = []
    for _, row in cand.iterrows():
        key = (int(row["row_id"]), row["column"])
        label = label_map.get(key, int(safe_float(row.get("conflict_score"), 0.0) >= cfg.candidate_score_threshold))
        rows.append({**row.to_dict(), "label_binary": label, "label_source": "sampled" if key in label_map else "heuristic_propagation", "propagation_confidence": 0.95 if key in label_map else 0.75})
    write_jsonl(out_path(cfg, "detection", "propagated_jsonl", "propagated_labels.jsonl"), rows)
    write_csv(out_path(cfg, "detection", "propagated_csv", "propagated_labels.csv"), rows)


def build_final_training_set_stage(cfg: DatasetConfig) -> None:
    require_pandas()
    prop = pd.read_csv(out_path(cfg, "detection", "propagated_csv", "propagated_labels.csv"), dtype=str, keep_default_na=False, encoding="utf-8-sig")
    prop["sample_weight"] = prop.get("propagation_confidence", 1.0)
    write_csv(out_path(cfg, "detection", "final_train_csv", "final_train_dataset.csv"), prop)
    write_jsonl(out_path(cfg, "detection", "final_train_jsonl", "final_train_dataset.jsonl"), prop.to_dict("records"))
    summary = {
        "dataset": cfg.dataset,
        "rows": int(len(prop)),
        "errors": int(pd.to_numeric(prop.get("label_binary", 0), errors="coerce").fillna(0).sum()),
        "label_sources": prop.get("label_source", pd.Series(dtype=str)).value_counts().to_dict(),
    }
    write_json(out_path(cfg, "detection", "final_summary_json", "final_train_summary.json"), summary)


def build_loading_set_stage(cfg: DatasetConfig) -> None:
    require_pandas()
    contexts = pd.read_csv(out_path(cfg, "detection", "contexts_csv", "candidate_llm_contexts_flat.csv"), dtype=str, keep_default_na=False, encoding="utf-8-sig")
    clustered_path = out_path(cfg, "detection", "clustered_csv", "candidate_clustered.csv")
    clustered = pd.read_csv(clustered_path, dtype=str, keep_default_na=False, encoding="utf-8-sig") if clustered_path.exists() else pd.DataFrame()
    if not clustered.empty:
        contexts = contexts.merge(clustered[["row_id", "column", "bucket_id", "cluster_id", "sample_role"]], on=["row_id", "column"], how="left")
    contexts["label_source"] = "inference"
    write_csv(out_path(cfg, "detection", "infer_ready_csv", "candidate_infer_ready.csv"), contexts)


def detector_loop_stage(cfg: DatasetConfig) -> None:
    require_pandas()
    infer_path = out_path(cfg, "detection", "infer_ready_csv", "candidate_infer_ready.csv")
    df = pd.read_csv(infer_path, dtype=str, keep_default_na=False, encoding="utf-8-sig") if infer_path.exists() else pd.DataFrame()
    out_dir = detector_output_dir(cfg)
    ensure_dir(out_dir)
    rows = []
    for _, row in df.iterrows():
        score = safe_float(row.get("conflict_score"), 0.0)
        pred = int(score >= cfg.candidate_score_threshold)
        rows.append({**row.to_dict(), "error_probability": min(0.999, score / 3.0), "final_pred_label": pred, "final_pred_label_name": "error" if pred else "correct"})
    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(out_dir / "inference_all_predictions.csv", index=False, encoding="utf-8-sig")
    high_risk = pred_df[pred_df.get("final_pred_label", 0).astype(str) == "1"] if not pred_df.empty else pred_df
    high_risk.to_csv(out_dir / "inference_high_risk.csv", index=False, encoding="utf-8-sig")
    hard = pred_df[(pd.to_numeric(pred_df.get("error_probability", 0), errors="coerce").between(0.35, 0.75))] if not pred_df.empty else pred_df
    hard.to_csv(out_dir / "inference_hard_cases.csv", index=False, encoding="utf-8-sig")
    write_jsonl(out_dir / "hard_cases_for_llm.jsonl", hard.to_dict("records") if not hard.empty else [])


def evaluate_detection_stage(cfg: DatasetConfig) -> None:
    require_pandas()
    if not cfg.clean_csv or not Path(cfg.clean_csv).exists() or not Path(cfg.dirty_csv).exists():
        print(f"[SKIP] clean/dirty table unavailable for {cfg.dataset} detection evaluation")
        return
    clean = read_table(cfg.clean_csv)
    dirty = read_table(cfg.dirty_csv)
    preds_path = detector_output_dir(cfg) / "inference_all_predictions.csv"
    preds = pd.read_csv(preds_path, dtype=str, keep_default_na=False, encoding="utf-8-sig") if preds_path.exists() else pd.DataFrame()
    truth = {(i, c) for i in range(min(len(clean), len(dirty))) for c in clean.columns if c in dirty.columns and normalize_value(clean.at[i, c]) != normalize_value(dirty.at[i, c])}
    pred = {(int(r.row_id), r.column) for r in preds.itertuples() if str(getattr(r, "final_pred_label", "0")) == "1"} if not preds.empty else set()
    tp = len(truth & pred)
    precision = tp / max(len(pred), 1)
    recall = tp / max(len(truth), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    out_dir = out_path(cfg, "detection", "eval_output_dir", "eval_outputs")
    ensure_dir(out_dir)
    write_json(out_dir / "metrics.json", {"precision": precision, "recall": recall, "f1": f1, "truth": len(truth), "predicted": len(pred)})


def wrong_position_stage(cfg: DatasetConfig) -> None:
    require_pandas()
    pred_path = detector_output_dir(cfg) / "inference_all_predictions.csv"
    rows = []
    if pred_path.exists():
        preds = pd.read_csv(pred_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        for r in preds.itertuples():
            if str(getattr(r, "final_pred_label", "0")) == "1":
                rows.append({"row_id": int(r.row_id), "column": r.column, "value": getattr(r, "value", "")})
    elif cfg.clean_csv and Path(cfg.clean_csv).exists() and Path(cfg.dirty_csv).exists():
        clean, dirty = read_table(cfg.clean_csv), read_table(cfg.dirty_csv)
        for i in range(min(len(clean), len(dirty))):
            for c in clean.columns:
                if c in dirty.columns and normalize_value(clean.at[i, c]) != normalize_value(dirty.at[i, c]):
                    rows.append({"row_id": i, "column": c, "value": dirty.at[i, c]})
    write_csv(out_path(cfg, "correction", "predicted_positions", "predicted_error_positions.csv"), rows)


def revise_candidates_stage(cfg: DatasetConfig) -> None:
    require_pandas()
    dirty = read_table(cfg.dirty_csv)
    positions_path = out_path(cfg, "correction", "predicted_positions", "predicted_error_positions.csv")
    positions = pd.read_csv(positions_path, dtype=str, keep_default_na=False, encoding="utf-8-sig") if positions_path.exists() else pd.DataFrame()
    pool_path = out_path(cfg, "detection", "rule_pool", "rule_pool.json")
    pool = read_json(pool_path) if pool_path.exists() else {"profiles": {}, "fd_rules": []}
    expanded = []
    grouped = []
    for _, pos in positions.iterrows():
        row_id, col = int(pos["row_id"]), pos["column"]
        if col not in dirty.columns or row_id >= len(dirty):
            continue
        dirty_value = normalize_value(dirty.at[row_id, col])
        candidates: dict[str, tuple[float, str]] = {}
        for item in pool.get("profiles", {}).get(col, {}).get("top_values", [])[:20]:
            value = normalize_value(item.get("value"))
            if value and value != dirty_value:
                candidates[value] = max(candidates.get(value, (0.0, ""))[0], safe_float(item.get("count"), 1.0)), "column_top_value"
        for rule in pool.get("fd_rules", []):
            if rule.get("rhs") != col:
                continue
            lhs = rule.get("lhs", [None])[0]
            if lhs in dirty.columns:
                expected = rule.get("mapping", {}).get(normalize_value(dirty.at[row_id, lhs]))
                if expected and expected != dirty_value:
                    candidates[expected] = (9999.0, f"fd:{lhs}->{col}")
        ranked = sorted(candidates.items(), key=lambda kv: kv[1][0], reverse=True)[:25]
        for rank, (value, (score, source)) in enumerate(ranked, start=1):
            expanded.append({"row_id": row_id, "column": col, "dirty_value": dirty_value, "candidate_value": value, "candidate_rank": rank, "candidate_source": source, "generation_score": score})
        grouped.append({"row_id": row_id, "column": col, "dirty_value": dirty_value, "candidate_count": len(ranked), "candidates_joined": " || ".join(v for v, _ in ranked), "top_candidate": ranked[0][0] if ranked else ""})
    write_csv(out_path(cfg, "correction", "candidates_expanded", "repair_candidates_expanded.csv"), expanded)
    write_csv(out_path(cfg, "correction", "candidates_grouped", "repair_candidates_grouped.csv"), grouped)


def build_features_stage(cfg: DatasetConfig) -> None:
    require_pandas()
    cand_path = out_path(cfg, "correction", "candidates_expanded", "repair_candidates_expanded.csv")
    cands = pd.read_csv(cand_path, dtype=str, keep_default_na=False, encoding="utf-8-sig") if cand_path.exists() else pd.DataFrame()
    rows = []
    for _, row in cands.iterrows():
        dirty = normalize_value(row.get("dirty_value"))
        cand = normalize_value(row.get("candidate_value"))
        edit = SequenceMatcher(None, dirty.lower(), cand.lower()).ratio() if dirty or cand else 0.0
        gen = safe_float(row.get("generation_score"), 0.0)
        score = math.log1p(max(gen, 0.0)) + edit
        rows.append({**row.to_dict(), "edit_similarity_to_dirty": edit, "log_generation_score": math.log1p(max(gen, 0.0)), "unified_repair_score": score})
    write_csv(out_path(cfg, "correction", "candidate_features", "repair_candidate_features_v2.csv"), rows)


def build_infer_whitelist_features_stage(cfg: DatasetConfig) -> None:
    require_pandas()
    feats_path = out_path(cfg, "correction", "candidate_features", "repair_candidate_features_v2.csv")
    allowed_path = out_path(cfg, "correction", "predicted_positions", "predicted_error_positions.csv")
    feats = pd.read_csv(feats_path, dtype=str, keep_default_na=False, encoding="utf-8-sig") if feats_path.exists() else pd.DataFrame()
    allowed = pd.read_csv(allowed_path, dtype=str, keep_default_na=False, encoding="utf-8-sig") if allowed_path.exists() else pd.DataFrame()
    if not feats.empty and not allowed.empty:
        keys = set(zip(allowed["row_id"].astype(str), allowed["column"].astype(str)))
        feats = feats[feats.apply(lambda r: (str(r["row_id"]), str(r["column"])) in keys, axis=1)]
    write_csv(out_path(cfg, "correction", "infer_features", "repair_candidate_features_infer_whitelist.csv"), feats)


def llm_ranking_stage(cfg: DatasetConfig) -> None:
    require_pandas()
    feats_path = out_path(cfg, "correction", "infer_features", "repair_candidate_features_infer_whitelist.csv")
    if not feats_path.exists():
        feats_path = out_path(cfg, "correction", "candidate_features", "repair_candidate_features_v2.csv")
    feats = pd.read_csv(feats_path, dtype=str, keep_default_na=False, encoding="utf-8-sig") if feats_path.exists() else pd.DataFrame()
    requests = []
    responses = []
    top_rows = []
    for (row_id, col), group in feats.groupby(["row_id", "column"]) if not feats.empty else []:
        ranked = group.assign(_score=pd.to_numeric(group["unified_repair_score"], errors="coerce").fillna(0)).sort_values("_score", ascending=False)
        if ranked.empty:
            continue
        top = ranked.iloc[0]
        top_rows.append({"row_id": row_id, "column": col, "teacher_top1_value": top["candidate_value"], "teacher_score": top["_score"]})
        values = ranked.head(3).to_dict("records")
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                req = {"row_id": row_id, "column": col, "candidate_a_value": values[i]["candidate_value"], "candidate_b_value": values[j]["candidate_value"]}
                requests.append(req)
                responses.append({**req, "preferred": "a", "reason": "higher unified statistical repair score"})
    write_jsonl(out_path(cfg, "correction", "pairwise_requests", "repair_pairwise_requests_v2.jsonl"), requests)
    write_jsonl(out_path(cfg, "correction", "pairwise_responses", "repair_pairwise_responses_v2.jsonl"), responses)
    write_csv(out_path(cfg, "correction", "teacher_top1", "repair_teacher_top1_labels_v2.csv"), top_rows)


def repair_loop_stage(cfg: DatasetConfig) -> None:
    require_pandas()
    feats_path = out_path(cfg, "correction", "infer_features", "repair_candidate_features_infer_whitelist.csv")
    if not feats_path.exists():
        feats_path = out_path(cfg, "correction", "candidate_features", "repair_candidate_features_v2.csv")
    feats = pd.read_csv(feats_path, dtype=str, keep_default_na=False, encoding="utf-8-sig") if feats_path.exists() else pd.DataFrame()
    out_dir = repair_output_dir(cfg)
    ensure_dir(out_dir)
    if feats.empty:
        write_csv(out_dir / "inference_all_candidate_ranked.csv", feats)
        write_csv(out_dir / "inference_top1_repairs.csv", feats)
        return
    feats["repair_score"] = pd.to_numeric(feats["unified_repair_score"], errors="coerce").fillna(0)
    ranked = feats.sort_values(["row_id", "column", "repair_score"], ascending=[True, True, False]).copy()
    ranked["rank"] = ranked.groupby(["row_id", "column"]).cumcount() + 1
    top1 = ranked[ranked["rank"] == 1].copy()
    top1["repaired_value"] = top1["candidate_value"]
    ranked.to_csv(out_dir / "inference_all_candidate_ranked.csv", index=False, encoding="utf-8-sig")
    ranked.to_csv(out_dir / "inference_all_candidate_scores.csv", index=False, encoding="utf-8-sig")
    top1.to_csv(out_dir / "inference_top1_repairs.csv", index=False, encoding="utf-8-sig")
    hard = ranked.groupby(["row_id", "column"]).head(2)
    write_jsonl(out_dir / "hard_cases_for_llm.jsonl", hard.to_dict("records"))


def evaluate_correction_stage(cfg: DatasetConfig) -> None:
    require_pandas()
    if not cfg.clean_csv or not Path(cfg.clean_csv).exists():
        print(f"[SKIP] clean table unavailable for {cfg.dataset} correction evaluation")
        return
    clean = read_table(cfg.clean_csv)
    top_path = repair_output_dir(cfg) / "inference_top1_repairs.csv"
    repairs = pd.read_csv(top_path, dtype=str, keep_default_na=False, encoding="utf-8-sig") if top_path.exists() else pd.DataFrame()
    total = len(repairs)
    correct = 0
    for r in repairs.itertuples():
        row_id = int(r.row_id)
        col = r.column
        if row_id < len(clean) and col in clean.columns and normalize_value(getattr(r, "repaired_value", "")) == normalize_value(clean.at[row_id, col]):
            correct += 1
    out_dir = out_path(cfg, "correction", "eval_output_dir", "repair_eval_outputs")
    ensure_dir(out_dir)
    write_json(out_dir / "metrics.json", {"repair_accuracy": correct / max(total, 1), "total_repairs": total, "correct_repairs": correct})


DETECTION_STAGE_FUNCS: dict[str, Callable[[DatasetConfig], None]] = {
    "build_rule_pool": build_rule_pool_stage,
    "shrink_search_space": shrink_search_space_stage,
    "summarize_full_contexts": summarize_full_contexts_stage,
    "cluster_and_sample": cluster_and_sample_stage,
    "annotate": annotate_stage,
    "label_propagation": label_propagation_stage,
    "build_final_training_set": build_final_training_set_stage,
    "build_loading_set": build_loading_set_stage,
    "detector_loop": detector_loop_stage,
    "evaluate": evaluate_detection_stage,
}

CORRECTION_STAGE_FUNCS: dict[str, Callable[[DatasetConfig], None]] = {
    "wrong_position": wrong_position_stage,
    "revise_candidates": revise_candidates_stage,
    "build_features": build_features_stage,
    "build_infer_whitelist_features": build_infer_whitelist_features_stage,
    "llm_ranking": llm_ranking_stage,
    "repair_loop": repair_loop_stage,
    "evaluate": evaluate_correction_stage,
}

STAGE_FUNCS = {"detection": DETECTION_STAGE_FUNCS, "correction": CORRECTION_STAGE_FUNCS}


def build_pipeline(task: str, dataset: str) -> PipelineConfig:
    task = task.lower()
    dataset = dataset.lower()
    if task not in CANONICAL_FLOWS:
        raise PipelineError(f"Unknown task {task!r}. Available tasks: {', '.join(TASKS)}")
    if dataset not in DATASETS:
        raise PipelineError(f"Unknown dataset {dataset!r}. Available datasets: {', '.join(TARGET_DATASETS)}")
    cfg = DATASETS[dataset]
    disabled = cfg.disabled_for(task)
    stages = tuple(stage for stage in CANONICAL_FLOWS[task] if stage.name not in disabled)
    return PipelineConfig(task=task, dataset=dataset, workdir=cfg.task_dir(task), stages=stages, dataset_config=cfg)


def available_tasks() -> dict[str, list[str]]:
    return {task: list(TARGET_DATASETS) for task in TASKS}


def expand_datasets(task: str, dataset: str) -> list[str]:
    return list(TARGET_DATASETS) if dataset.lower() == "all" else [dataset.lower()]


def slice_stages(stages: Sequence[Stage], *, from_step: str | None, to_step: str | None, only_steps: Sequence[str], skip_steps: Sequence[str], include_llm: bool) -> tuple[Stage, ...]:
    selected = list(stages)
    names = [s.name for s in selected]
    if from_step:
        if from_step not in names:
            raise PipelineError(f"Unknown --from-step {from_step!r}. Valid steps: {', '.join(names)}")
        selected = selected[names.index(from_step):]
        names = [s.name for s in selected]
    if to_step:
        if to_step not in names:
            raise PipelineError(f"Unknown --to-step {to_step!r}. Valid steps after slicing: {', '.join(names)}")
        selected = selected[:names.index(to_step) + 1]
    if only_steps:
        only = set(only_steps)
        unknown = only.difference(s.name for s in stages)
        if unknown:
            raise PipelineError(f"Unknown --only-step value(s): {', '.join(sorted(unknown))}")
        selected = [s for s in selected if s.name in only]
    if skip_steps:
        skip = set(skip_steps)
        unknown = skip.difference(s.name for s in stages)
        if unknown:
            raise PipelineError(f"Unknown --skip-step value(s): {', '.join(sorted(unknown))}")
        selected = [s for s in selected if s.name not in skip]
    if not include_llm:
        selected = [s for s in selected if not s.llm]
    return tuple(selected)


def validate_pipeline(config: PipelineConfig, stages: Iterable[Stage] | None = None) -> list[str]:
    messages = []
    cfg = config.dataset_config
    if not config.workdir.exists():
        messages.append(f"missing workdir: {config.workdir}")
    if config.task == "detection" and not Path(cfg.dirty_csv).exists():
        messages.append(f"missing dirty_csv: {cfg.dirty_csv}")
    for stage in stages or config.stages:
        if stage.name not in STAGE_FUNCS[config.task]:
            messages.append(f"missing unified implementation for stage: {stage.name}")
    return messages


def print_plan(config: PipelineConfig, stages: Sequence[Stage]) -> None:
    print(f"Pipeline: {config.task}/{config.dataset}")
    print(f"Workdir : {config.workdir}")
    print(f"Dirty   : {config.dataset_config.dirty_csv}")
    if config.dataset_config.clean_csv:
        print(f"Clean   : {config.dataset_config.clean_csv}")
    print("Flow    : unified in-code statistical cleaning flow")
    print("Stages  :")
    for idx, stage in enumerate(stages, start=1):
        flags = []
        if stage.llm:
            flags.append("LLM-compatible")
        if stage.optional:
            flags.append("optional")
        suffix = f" [{' / '.join(flags)}]" if flags else ""
        print(f"  {idx:02d}. {stage.name}{suffix}")
        print(f"      # {stage.description}")


def run_pipeline(config: PipelineConfig, stages: Sequence[Stage], *, dry_run: bool) -> None:
    print_plan(config, stages)
    if dry_run:
        print("\n[DRY-RUN] No stages executed and no input files validated.")
        return
    ensure_dir(config.workdir)
    problems = validate_pipeline(config, stages)
    if problems:
        raise PipelineError("Pipeline validation failed:\n" + "\n".join(f"- {m}" for m in problems))
    for stage in stages:
        print(f"\n[RUN] {config.task}/{config.dataset}:{stage.name}", flush=True)
        STAGE_FUNCS[config.task][stage.name](config.dataset_config)


def add_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--dataset", required=True, help="Dataset name, or 'all'")
    parser.add_argument("--from-step", default=None)
    parser.add_argument("--to-step", default=None)
    parser.add_argument("--only-step", action="append", default=[])
    parser.add_argument("--skip-step", action="append", default=[])
    parser.add_argument("--skip-llm", action="store_true", help="Skip stages that normally use/replace LLM labeling loops")
    parser.add_argument("--dirty-csv", default=None, help="Override configured dirty CSV for one-off runs/tests")
    parser.add_argument("--clean-csv", default=None, help="Override configured clean CSV for one-off runs/tests")
    parser.add_argument("--workdir", default=None, help="Override task workdir for one-off runs/tests")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified statistical data-cleaning pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="List supported tasks and datasets")
    plan = sub.add_parser("plan", help="Print selected unified stages")
    add_selection_args(plan)
    plan.add_argument("--json", action="store_true")
    validate = sub.add_parser("validate", help="Validate selected unified stages and configured inputs")
    add_selection_args(validate)
    run = sub.add_parser("run", help="Run selected unified stages")
    add_selection_args(run)
    run.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def selected_pipelines(args: argparse.Namespace) -> list[tuple[PipelineConfig, tuple[Stage, ...]]]:
    out = []
    datasets = expand_datasets(args.task, args.dataset)
    if len(datasets) > 1 and (args.dirty_csv or args.clean_csv or args.workdir):
        raise PipelineError("--dirty-csv/--clean-csv/--workdir overrides are only supported for one dataset at a time")
    for dataset in datasets:
        config = build_pipeline(args.task, dataset)
        if args.dirty_csv or args.clean_csv or args.workdir:
            dataset_config = config.dataset_config
            if args.dirty_csv or args.clean_csv:
                dataset_config = replace(
                    dataset_config,
                    dirty_csv=args.dirty_csv or dataset_config.dirty_csv,
                    clean_csv=args.clean_csv if args.clean_csv is not None else dataset_config.clean_csv,
                )
            if args.workdir:
                if args.task == "detection":
                    dataset_config = replace(dataset_config, detection_dir=args.workdir)
                else:
                    dataset_config = replace(dataset_config, correction_dir=args.workdir)
            config = replace(config, dataset_config=dataset_config, workdir=dataset_config.task_dir(args.task))
        stages = slice_stages(config.stages, from_step=args.from_step, to_step=args.to_step, only_steps=args.only_step, skip_steps=args.skip_step, include_llm=not args.skip_llm)
        out.append((config, stages))
    return out


def pipeline_payload(config: PipelineConfig, stages: Sequence[Stage]) -> dict[str, Any]:
    return {
        "task": config.task,
        "dataset": config.dataset,
        "workdir": str(config.workdir),
        "dirty_csv": config.dataset_config.dirty_csv,
        "clean_csv": config.dataset_config.clean_csv,
        "stages": [{"name": s.name, "description": s.description, "llm": s.llm, "optional": s.optional} for s in stages],
        "outputs": config.dataset_config.detection_outputs if config.task == "detection" else config.dataset_config.correction_outputs,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "list":
            for task, datasets in available_tasks().items():
                print(f"{task}: {', '.join(datasets)}")
            return 0
        pipelines = selected_pipelines(args)
        if args.command == "plan":
            if args.json:
                print(json.dumps([pipeline_payload(c, s) for c, s in pipelines], ensure_ascii=False, indent=2))
            else:
                for i, (config, stages) in enumerate(pipelines):
                    if i:
                        print()
                    print_plan(config, stages)
            return 0
        if args.command == "validate":
            failed = False
            for config, stages in pipelines:
                messages = validate_pipeline(config, stages)
                if messages:
                    failed = True
                    print(f"{config.task}/{config.dataset}:")
                    for msg in messages:
                        print(f"  {msg}")
                else:
                    print(f"OK: {config.task}/{config.dataset} ({len(stages)} selected stages)")
            return 1 if failed else 0
        if args.command == "run":
            for i, (config, stages) in enumerate(pipelines):
                if i:
                    print()
                run_pipeline(config, stages, dry_run=args.dry_run)
            return 0
    except PipelineError as exc:
        print(f"ERROR: {exc}")
        return 1
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())

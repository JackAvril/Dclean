#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run correction ablation experiments for six datasets.

Datasets:
  hospital, flights, beers, rayyan, adult, soccer

Experiments implemented:
  1. w/o Candidate Constraints
     - Bypasses the candidate-constrained ranking pipeline.
     - Directly asks an LLM to produce a repair value for each detected error cell.
     - Output is still inference_top1_repairs.csv, so evaluate.py can be reused.

  2. w/o Pairwise Preference Distillation
     - Keeps the candidate generation / feature / whitelist stages.
     - Skips llm_ranking.py and train_ltr.py.
     - Directly ranks candidates by candidate-generation score:
         final_score_from_candidate_gen / final_score / source_score / candidate_score / candidate_rank
     - Output is still inference_top1_repairs.csv, so evaluate.py can be reused.

Assumed unified scripts are located in:
  /mnt/mydata/dq/projects/Splittree/test/Error Correction

with names:
  wrong_position.py
  revise_candidate.py
  build_features.py
  build_infer_candidate_features_from_whitelist.py
  evaluate.py

Optional:
  llm_ranking.py / train_ltr.py / infer_ltr.py / control_iteration.py are not required
  for these two ablations.

Recommended usage:
  cd "/mnt/mydata/dq/projects/Splittree/test/Error Correction"
  python run_correction_ablation_experiments.py --experiments no_pairwise_distillation

Direct LLM unconstrained ablation:
  export DEEPSEEK_API_KEY="your_key"
  python run_correction_ablation_experiments.py \
    --experiments no_candidate_constraints \
    --max_llm_cells_per_dataset 500 \
    --llm_max_workers 4

Run both:
  python run_correction_ablation_experiments.py \
    --experiments no_candidate_constraints,no_pairwise_distillation

Dry-run commands only:
  python run_correction_ablation_experiments.py --dry_run 1
"""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# ============================================================
# 1. Global paths
# ============================================================

PROJECT_ROOT = Path("/mnt/mydata/dq/projects/Splittree")
TEST_ROOT = PROJECT_ROOT / "test"
CORRECTION_CODE_DIR = TEST_ROOT / "Error Correction"
DEFAULT_OUTPUT_ROOT = CORRECTION_CODE_DIR / "ablation_runs"

DATASETS = ["hospital", "flights", "beers", "rayyan", "adult", "soccer"]

# Exact or likely detection output paths. If one path does not exist,
# the script will search the corresponding Error Detection folder.
DETECTION_PREDICTION_FILES = {
    "hospital": TEST_ROOT / "Error Detection hospital" / "active_cleaning_loop_runs_v4" / "iter_3" / "infer" / "inference_all_predictions.csv",
    "flights": TEST_ROOT / "Error Detection flights" / "active_cleaning_loop_runs_flights_v4" / "iter_3" / "infer" / "inference_all_predictions.csv",
    "beers": TEST_ROOT / "Error Detection beers" / "active_cleaning_loop_runs_beers_v4" / "iter_3" / "infer" / "inference_all_predictions.csv",
    "rayyan": TEST_ROOT / "Error Detection rayyan" / "active_cleaning_loop_runs_rayyan_v4" / "iter_3" / "infer" / "inference_all_predictions.csv",
    "adult": TEST_ROOT / "Error Detection Adult" / "active_cleaning_loop_runs_adult_v4" / "iter_3_infer" / "inference_all_predictions.csv",
    "soccer": TEST_ROOT / "Error Detection soccer" / "active_cleaning_loop_runs_soccer_v4" / "iter_3_infer" / "inference_all_predictions.csv",
}

DETECTION_DIRS = {
    "hospital": TEST_ROOT / "Error Detection hospital",
    "flights": TEST_ROOT / "Error Detection flights",
    "beers": TEST_ROOT / "Error Detection beers",
    "rayyan": TEST_ROOT / "Error Detection rayyan",
    "adult": TEST_ROOT / "Error Detection Adult",
    "soccer": TEST_ROOT / "Error Detection soccer",
}

CONTEXT_FILES = {
    "hospital": TEST_ROOT / "Error Detection hospital" / "candidate_llm_contexts_hospital.jsonl",
    "flights": TEST_ROOT / "Error Detection flights" / "candidate_llm_contexts_flights.jsonl",
    "beers": TEST_ROOT / "Error Detection beers" / "candidate_llm_contexts_beers.jsonl",
    "rayyan": TEST_ROOT / "Error Detection rayyan" / "candidate_llm_contexts_rayyan.jsonl",
    "adult": TEST_ROOT / "Error Detection Adult" / "candidate_llm_contexts_adult.jsonl",
    "soccer": TEST_ROOT / "Error Detection soccer" / "candidate_llm_contexts_soccer.jsonl",
}

FALLBACK_CONTEXT_FILES = {
    "hospital": TEST_ROOT / "Error Detection new" / "candidate_llm_contexts_v2.jsonl",
    "soccer": TEST_ROOT / "Error Detection soccer" / "candidate_llm_contexts_soccer.jsonl",
    "adult": TEST_ROOT / "Error Detection adult" / "candidate_llm_contexts_adult.jsonl",
}

SCRIPT_NAMES = {
    "wrong_position": "wrong_position.py",
    "revise_candidate": "revise_candidate.py",
    "build_features": "build_features.py",
    "build_whitelist": "build_infer_candidate_features_from_whitelist.py",
    "evaluate": "evaluate.py",
}


# ============================================================
# 2. Basic helpers
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Run correction ablation experiments for six datasets.")

    p.add_argument("--datasets", default=",".join(DATASETS),
                   help="Comma-separated datasets. Default: hospital,flights,beers,rayyan,adult,soccer")
    p.add_argument("--experiments", default="no_candidate_constraints,no_pairwise_distillation",
                   help="Comma-separated experiments: no_candidate_constraints,no_pairwise_distillation")
    p.add_argument("--code_dir", default=str(CORRECTION_CODE_DIR))
    p.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--dry_run", type=int, default=0)
    p.add_argument("--continue_on_error", type=int, default=1)

    # Direct LLM no-candidate-constraints settings.
    p.add_argument("--model_name", default=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))
    p.add_argument("--base_url", default=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--llm_max_workers", type=int, default=2)
    p.add_argument("--max_retries", type=int, default=3)
    p.add_argument("--retry_sleep_seconds", type=float, default=2.0)
    p.add_argument("--max_llm_cells_per_dataset", type=int, default=0,
                   help="0 means all detected cells. Use a smaller number for quick tests.")
    p.add_argument("--skip_existing_llm_responses", type=int, default=1)

    # Rule-score no-pairwise settings.
    p.add_argument("--include_dirty_as_candidate", type=int, default=1)
    p.add_argument("--rule_score_min_margin", type=float, default=-999.0,
                   help="For no_pairwise_distillation. Default accepts top1. Set e.g. 0.05 for a simple margin gate.")
    p.add_argument("--rule_score_skip_same_as_dirty", type=int, default=0)

    # Extra args forwarded to unified pipeline scripts.
    p.add_argument("--wrong_position_extra_args", default="")
    p.add_argument("--revise_extra_args", default="")
    p.add_argument("--build_features_extra_args", default="")
    p.add_argument("--whitelist_extra_args", default="")
    p.add_argument("--evaluate_extra_args", default="")

    return p.parse_args()


def split_csv_arg(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd: List[str], cwd: Path, dry_run: bool = False):
    print("\n[RUN]", " ".join(str(x) for x in cmd))
    print("[CWD]", str(cwd))
    if dry_run:
        return
    subprocess.run([str(x) for x in cmd], cwd=str(cwd), check=True)


def shell_split(s: str) -> List[str]:
    import shlex
    return shlex.split(s) if s else []


def write_json(path: Path, obj: Any):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(path: Path, obj: dict):
    ensure_dir(path.parent)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                print(f"[WARN] skip bad jsonl: {path}, line={line_no}, err={e}")
    return rows


def find_script(code_dir: Path, name_key: str) -> Path:
    p = code_dir / SCRIPT_NAMES[name_key]
    if not p.exists():
        raise FileNotFoundError(f"Script not found: {p}")
    return p


def find_detection_file(dataset: str) -> Path:
    p = DETECTION_PREDICTION_FILES[dataset]
    if p.exists():
        return p

    root = DETECTION_DIRS.get(dataset)
    if root and root.exists():
        matches = sorted(root.rglob("inference_all_predictions.csv"), key=lambda x: len(str(x)))
        if matches:
            print(f"[WARN] default detection file not found for {dataset}; use found file: {matches[0]}")
            return matches[0]

    raise FileNotFoundError(f"Detection prediction file not found for {dataset}. Tried: {p}")


def find_context_file(dataset: str) -> Optional[Path]:
    candidates = []
    if dataset in CONTEXT_FILES:
        candidates.append(CONTEXT_FILES[dataset])
    if dataset in FALLBACK_CONTEXT_FILES:
        candidates.append(FALLBACK_CONTEXT_FILES[dataset])
    root = DETECTION_DIRS.get(dataset)
    if root and root.exists():
        candidates.extend(sorted(root.rglob("candidate_llm_contexts*.jsonl")))

    for p in candidates:
        if p.exists():
            return p
    return None


def dataset_data_dir(dataset: str) -> Path:
    return PROJECT_ROOT / dataset


def find_clean_dirty_files(dataset: str) -> Tuple[Path, Path]:
    root = dataset_data_dir(dataset)
    if not root.exists():
        raise FileNotFoundError(f"Dataset data directory not found: {root}")

    # Hospital has a known special file name in your earlier evaluator.
    if dataset == "hospital":
        clean_special = root / "hospital_clean_no_address23.csv"
        dirty_special = root / "hospital_dirty_no_address23.csv"
        if clean_special.exists() and dirty_special.exists():
            return clean_special, dirty_special

    candidates = [
        (root / f"{dataset}_clean.csv", root / f"{dataset}_dirty.csv"),
        (root / f"{dataset}_clean_no_address23.csv", root / f"{dataset}_dirty_no_address23.csv"),
        (root / "clean.csv", root / "dirty.csv"),
        (root / f"{dataset}.clean.csv", root / f"{dataset}.dirty.csv"),
    ]

    for clean, dirty in candidates:
        if clean.exists() and dirty.exists():
            return clean, dirty

    clean_matches = sorted(root.glob("*clean*.csv"))
    dirty_matches = sorted(root.glob("*dirty*.csv"))
    if clean_matches and dirty_matches:
        print(f"[WARN] use detected clean/dirty files for {dataset}: {clean_matches[0]}, {dirty_matches[0]}")
        return clean_matches[0], dirty_matches[0]

    raise FileNotFoundError(f"Cannot find clean/dirty files in {root}")


def normalize_empty(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    s = str(x).strip()
    if s.lower() in {"", "nan", "none", "null", "na", "n/a", "missing", "empty", "?"}:
        return ""
    return s


# ============================================================
# 3. Step: wrong position
# ============================================================

def step_wrong_position(dataset: str, work_dir: Path, code_dir: Path, args) -> Path:
    input_file = find_detection_file(dataset)
    output_file = work_dir / "predicted_error_positions.csv"

    cmd = [
        "python", str(find_script(code_dir, "wrong_position")),
        "--dataset", dataset,
        "--input_file", str(input_file),
        "--output_file", str(output_file),
    ] + shell_split(args.wrong_position_extra_args)

    run_cmd(cmd, cwd=work_dir, dry_run=bool(args.dry_run))

    if not args.dry_run and not output_file.exists():
        raise FileNotFoundError(f"wrong_position did not create expected file: {output_file}")

    return output_file


# ============================================================
# 4. Experiment A: w/o Candidate Constraints
# ============================================================

def load_dirty_row_context(dirty_file: Path, row_id: int, max_cols: int = 30) -> Dict[str, Any]:
    # For huge files, nrows/skiprows would be more efficient, but six datasets here are manageable.
    df = pd.read_csv(dirty_file, dtype=str, keep_default_na=False, encoding="utf-8-sig", low_memory=False)
    if row_id < 0 or row_id >= len(df):
        return {}
    row = df.iloc[row_id].to_dict()
    # Keep columns in original order, but limit if extremely wide.
    return {str(k): normalize_empty(v) for k, v in list(row.items())[:max_cols]}


def build_direct_llm_prompt(dataset: str, row_id: int, column: str, dirty_value: str, row_context: Dict[str, Any]) -> str:
    dataset_hint = {
        "hospital": (
            "Hospital data. Pay attention to MeasureCode/Stateavg/Condition/MeasureName consistency, "
            "Score percent format, and Sample pattern."
        ),
        "flights": (
            "Flights data. Pay attention to scheduled/actual departure/arrival times and flight/source consistency."
        ),
        "beers": (
            "Beers data. Pay attention to state-city, brewery, beer style, ounces, abv, and ibu format."
        ),
        "rayyan": (
            "Rayyan biomedical literature metadata. Pay attention to language, ISSN, date, journal metadata, volume, issue, pagination."
        ),
        "adult": (
            "Adult census data. Pay attention to relationship/sex/maritalstatus/age/education/hours/income consistency."
        ),
        "soccer": (
            "Soccer player/team data. Pay attention to position domain, birthyear-season age consistency, team/city/stadium/manager context, and entity names."
        ),
    }.get(dataset, "Tabular data cleaning task.")

    return f"""
You are doing an unconstrained data repair ablation.

This ablation is w/o Candidate Constraints:
- You are NOT given a constrained candidate set.
- Directly propose the best repair value for the suspicious cell.
- If the original dirty value is already the best value or there is not enough evidence, output the original value.
- Do not invent unsupported values. Use row context and dataset knowledge.

Dataset hint:
{dataset_hint}

Suspicious cell:
row_id: {row_id}
column: {column}
dirty_value: {dirty_value}

Row context:
{json.dumps(row_context, ensure_ascii=False, indent=2)}

Return ONLY JSON:
{{
  "repair_value": "...",
  "confidence": 0.0_to_1.0,
  "reason": "short reason"
}}
""".strip()


def parse_llm_repair(text: str, fallback: str) -> Dict[str, Any]:
    if text is None:
        return {"repair_value": fallback, "confidence": 0.0, "reason": "empty response"}
    s = str(text).strip()
    try:
        obj = json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, flags=re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                obj = {}
        else:
            obj = {}

    val = normalize_empty(obj.get("repair_value", obj.get("predicted_repair", obj.get("value", ""))))
    if not val:
        val = fallback
    return {
        "repair_value": val,
        "confidence": obj.get("confidence", 0.0),
        "reason": obj.get("reason", ""),
    }


def call_direct_llm(client, args, prompt: str, fallback: str) -> Dict[str, Any]:
    last_err = None
    for attempt in range(1, args.max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=args.model_name,
                temperature=args.temperature,
                messages=[
                    {"role": "system", "content": "You are a careful data cleaning expert. Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
            )
            text = resp.choices[0].message.content
            parsed = parse_llm_repair(text, fallback=fallback)
            parsed["raw_response"] = text
            parsed["llm_error"] = ""
            return parsed
        except Exception as e:
            last_err = e
            time.sleep(args.retry_sleep_seconds * attempt)
    return {
        "repair_value": fallback,
        "confidence": 0.0,
        "reason": f"LLM failed: {last_err}",
        "raw_response": "",
        "llm_error": str(last_err),
    }


def run_no_candidate_constraints(dataset: str, work_dir: Path, code_dir: Path, args) -> Path:
    print(f"\n========== Ablation: w/o Candidate Constraints | {dataset} ==========")

    clean_file, dirty_file = find_clean_dirty_files(dataset)
    pred_file = step_wrong_position(dataset, work_dir, code_dir, args)

    model_output = work_dir / "model_output"
    ensure_dir(model_output)

    requests_jsonl = model_output / "direct_llm_repair_requests.jsonl"
    responses_jsonl = model_output / "direct_llm_repair_responses.jsonl"
    top1_file = model_output / "inference_top1_repairs.csv"

    pred_df = pd.read_csv(pred_file, encoding="utf-8-sig", low_memory=False)
    if "row_id" not in pred_df.columns or "column" not in pred_df.columns:
        raise ValueError(f"predicted error positions must contain row_id,column: {pred_file}")

    pred_df["row_id"] = pd.to_numeric(pred_df["row_id"], errors="raise").astype(int)
    pred_df["column"] = pred_df["column"].astype(str).str.strip()

    if args.max_llm_cells_per_dataset and args.max_llm_cells_per_dataset > 0:
        pred_df = pred_df.head(args.max_llm_cells_per_dataset).copy()

    # Build requests.
    if not args.dry_run:
        if requests_jsonl.exists():
            requests_jsonl.unlink()
        for _, r in pred_df.iterrows():
            row_id = int(r["row_id"])
            column = str(r["column"])
            dirty_value = normalize_empty(r.get("value", r.get("dirty_value", "")))
            if not dirty_value:
                # Load from table when not present in predicted positions.
                row_ctx = load_dirty_row_context(dirty_file, row_id)
                dirty_value = normalize_empty(row_ctx.get(column, ""))
            else:
                row_ctx = load_dirty_row_context(dirty_file, row_id)

            prompt = build_direct_llm_prompt(dataset, row_id, column, dirty_value, row_ctx)
            append_jsonl(requests_jsonl, {
                "request_id": f"{row_id}::{column}",
                "row_id": row_id,
                "column": column,
                "dirty_value": dirty_value,
                "row_context": row_ctx,
                "prompt": prompt,
            })

    print(f"[OK] direct LLM requests: {requests_jsonl}")

    if args.dry_run:
        return top1_file

    if OpenAI is None:
        raise RuntimeError("openai package is not installed. Install openai or only run no_pairwise_distillation.")
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is empty. Set it before running no_candidate_constraints, "
            "or run --experiments no_pairwise_distillation first."
        )

    client = OpenAI(api_key=api_key, base_url=args.base_url)
    requests = read_jsonl(requests_jsonl)

    existing = {}
    if args.skip_existing_llm_responses and responses_jsonl.exists():
        for obj in read_jsonl(responses_jsonl):
            existing[obj.get("request_id")] = obj

    pending = [x for x in requests if x.get("request_id") not in existing]
    print(f"[INFO] direct LLM total={len(requests)}, existing={len(existing)}, pending={len(pending)}")

    if pending:
        with ThreadPoolExecutor(max_workers=max(1, int(args.llm_max_workers))) as ex:
            futs = []
            for req in pending:
                futs.append(ex.submit(call_direct_llm, client, args, req["prompt"], req.get("dirty_value", "")))

            for req, fut in zip(pending, futs):
                out = fut.result()
                record = dict(req)
                record.update({
                    "predicted_repair": out["repair_value"],
                    "direct_llm_confidence": out.get("confidence", 0.0),
                    "direct_llm_reason": out.get("reason", ""),
                    "raw_response": out.get("raw_response", ""),
                    "llm_error": out.get("llm_error", ""),
                })
                append_jsonl(responses_jsonl, record)

    responses = read_jsonl(responses_jsonl)

    rows = []
    for obj in responses:
        rows.append({
            "row_id": int(obj["row_id"]),
            "column": str(obj["column"]),
            "dirty_value": obj.get("dirty_value", ""),
            "predicted_repair_value": obj.get("predicted_repair", obj.get("dirty_value", "")),
            "predicted_repair": obj.get("predicted_repair", obj.get("dirty_value", "")),
            "student_top_candidate": obj.get("predicted_repair", obj.get("dirty_value", "")),
            "student_score": obj.get("direct_llm_confidence", 0.0),
            "student_margin": obj.get("direct_llm_confidence", 0.0),
            "gate_passed": 1,
            "gate_reason": "no_candidate_constraints_direct_llm",
            "student_reason_pred": obj.get("direct_llm_reason", ""),
            "candidate_count": 0,
        })

    top1_df = pd.DataFrame(rows).drop_duplicates(subset=["row_id", "column"], keep="first")
    top1_df.to_csv(top1_file, index=False, encoding="utf-8-sig")
    print(f"[OK] no_candidate_constraints top1 repairs: {top1_file}, rows={len(top1_df)}")

    run_evaluate(dataset, work_dir, code_dir, top1_file, clean_file, dirty_file, args)
    return top1_file


# ============================================================
# 5. Experiment B: w/o Pairwise Preference Distillation
# ============================================================

def run_candidate_pipeline_until_whitelist(dataset: str, work_dir: Path, code_dir: Path, args) -> Tuple[Path, Path, Path]:
    clean_file, dirty_file = find_clean_dirty_files(dataset)
    pred_file = step_wrong_position(dataset, work_dir, code_dir, args)

    context_file = find_context_file(dataset)
    if context_file is None:
        context_file = Path("")

    expanded = work_dir / "repair_candidates_expanded.csv"
    grouped = work_dir / "repair_candidates_grouped.csv"
    features = work_dir / "repair_candidate_features_v2.csv"
    whitelist = work_dir / "repair_candidate_features_infer_whitelist.csv"

    # revise_candidate.py
    cmd = [
        "python", str(find_script(code_dir, "revise_candidate")),
        "--dataset", dataset,
        "--error_pred_file", str(pred_file),
        "--dirty_table_csv", str(dirty_file),
        "--output_expanded", str(expanded),
        "--output_grouped", str(grouped),
        "--include_dirty_value", str(int(args.include_dirty_as_candidate)),
    ]
    if str(context_file):
        cmd += ["--context_file", str(context_file)]
    cmd += shell_split(args.revise_extra_args)
    run_cmd(cmd, cwd=work_dir, dry_run=bool(args.dry_run))

    # build_features.py
    cmd = [
        "python", str(find_script(code_dir, "build_features")),
        "--dataset", dataset,
        "--candidates_file", str(expanded),
        "--raw_data_file", str(dirty_file),
        "--output_features_csv", str(features),
    ] + shell_split(args.build_features_extra_args)
    run_cmd(cmd, cwd=work_dir, dry_run=bool(args.dry_run))

    # build whitelist
    cmd = [
        "python", str(find_script(code_dir, "build_whitelist")),
        "--dataset", dataset,
        "--candidate_features", str(features),
        "--allowed_cells", str(pred_file),
        "--output", str(whitelist),
    ] + shell_split(args.whitelist_extra_args)
    run_cmd(cmd, cwd=work_dir, dry_run=bool(args.dry_run))

    if not args.dry_run:
        for p in [expanded, features, whitelist]:
            if not p.exists():
                raise FileNotFoundError(f"Expected pipeline output missing: {p}")

    return whitelist, clean_file, dirty_file


def choose_score_column(df: pd.DataFrame) -> Optional[str]:
    for c in ["final_score_from_candidate_gen", "final_score", "source_score", "candidate_score", "student_score"]:
        if c in df.columns:
            return c
    return None


def find_candidate_value_col(df: pd.DataFrame) -> str:
    for c in ["candidate_value", "repair_candidate", "candidate", "value_candidate", "predicted_repair", "student_top_candidate"]:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot find candidate value column in {list(df.columns)}")


def make_no_pairwise_top1(whitelist_features: Path, output_file: Path, args) -> pd.DataFrame:
    df = pd.read_csv(whitelist_features, encoding="utf-8-sig", low_memory=False)
    if "row_id" not in df.columns or "column" not in df.columns:
        raise ValueError("whitelist feature file must contain row_id,column")

    df["row_id"] = pd.to_numeric(df["row_id"], errors="raise").astype(int)
    df["column"] = df["column"].astype(str).str.strip()

    cand_col = find_candidate_value_col(df)
    score_col = choose_score_column(df)

    rows = []
    for (row_id, column), g0 in df.groupby(["row_id", "column"], sort=False):
        g = g0.copy()
        if score_col:
            g["_score"] = pd.to_numeric(g[score_col], errors="coerce").fillna(0.0)
        else:
            g["_score"] = 0.0

        if "candidate_rank" in g.columns:
            g["_rank"] = pd.to_numeric(g["candidate_rank"], errors="coerce").fillna(999999)
        else:
            g["_rank"] = range(1, len(g) + 1)

        g = g.sort_values(["_score", "_rank"], ascending=[False, True]).reset_index(drop=True)
        top = g.iloc[0]
        top2_score = float(g.iloc[1]["_score"]) if len(g) >= 2 else -999999.0
        margin = float(top["_score"]) - top2_score if len(g) >= 2 else float(top["_score"])

        dirty_value = normalize_empty(top.get("dirty_value", top.get("value", "")))
        pred = normalize_empty(top[cand_col])

        gate_passed = 1
        gate_reason = "no_pairwise_rule_score_top1"

        if args.rule_score_skip_same_as_dirty and pred == dirty_value:
            gate_passed = 0
            gate_reason = "skip_same_as_dirty"

        if margin < args.rule_score_min_margin:
            gate_passed = 0
            gate_reason = f"simple_margin_too_small:{margin:.6f}"

        # To preserve the repair-result output shape, we still output every top1 row.
        # If gate blocks, keep dirty value as repair.
        output_repair = pred if gate_passed else dirty_value

        row = {
            "row_id": int(row_id),
            "column": str(column),
            "dirty_value": dirty_value,
            "predicted_repair_value": output_repair,
            "predicted_repair": output_repair,
            "student_top_candidate": pred,
            "student_score": float(top["_score"]),
            "student_margin": margin,
            "gate_passed": int(gate_passed),
            "gate_reason": gate_reason,
            "student_reason_pred": "no_pairwise_preference_distillation",
            "candidate_count": len(g),
            "top_candidate_sources": top.get("candidate_sources", top.get("candidate_source", "")),
        }

        # Preserve useful diagnostics if present.
        for c in [
            "candidate_rule_support_ratio", "source_count", "is_from_rule", "is_from_neighbor",
            "is_from_similarity", "is_from_domain", "is_from_profile_key_majority",
            "is_from_entity_pair_majority", "is_from_team_season_position",
            "candidate_entity_safe_fill", "candidate_position_profile_strong",
            "candidate_has_high_conf_profile_majority",
        ]:
            if c in top.index:
                row[c] = top[c]

        rows.append(row)

    out = pd.DataFrame(rows).sort_values(["row_id", "column"]).reset_index(drop=True)
    ensure_dir(output_file.parent)
    out.to_csv(output_file, index=False, encoding="utf-8-sig")
    return out


def run_no_pairwise_distillation(dataset: str, work_dir: Path, code_dir: Path, args) -> Path:
    print(f"\n========== Ablation: w/o Pairwise Preference Distillation | {dataset} ==========")

    whitelist, clean_file, dirty_file = run_candidate_pipeline_until_whitelist(dataset, work_dir, code_dir, args)

    model_output = work_dir / "model_output"
    ensure_dir(model_output)

    top1_file = model_output / "inference_top1_repairs.csv"
    ranked_file = model_output / "inference_all_candidate_ranked.csv"
    scores_file = model_output / "inference_all_candidate_scores.csv"

    if not args.dry_run:
        # For compatibility with the normal inference output names.
        shutil.copy2(whitelist, scores_file)
        shutil.copy2(whitelist, ranked_file)
        out = make_no_pairwise_top1(whitelist, top1_file, args)
        print(f"[OK] no_pairwise_distillation top1 repairs: {top1_file}, rows={len(out)}")

    run_evaluate(dataset, work_dir, code_dir, top1_file, clean_file, dirty_file, args)
    return top1_file


# ============================================================
# 6. Evaluation
# ============================================================

def run_evaluate(dataset: str, work_dir: Path, code_dir: Path, repair_file: Path, clean_file: Path, dirty_file: Path, args):
    eval_dir = work_dir / "eval"

    cmd = [
        "python", str(find_script(code_dir, "evaluate")),
        "--dataset", dataset,
        "--clean_file", str(clean_file),
        "--dirty_file", str(dirty_file),
        "--repair_result_file", str(repair_file),
        "--output_dir", str(eval_dir),
    ] + shell_split(args.evaluate_extra_args)

    run_cmd(cmd, cwd=work_dir, dry_run=bool(args.dry_run))

    summary_file = eval_dir / "full_table_eval_summary.csv"
    if not args.dry_run and not summary_file.exists():
        raise FileNotFoundError(f"Evaluation summary missing: {summary_file}")


def collect_summaries(output_root: Path, experiments: List[str], datasets: List[str]) -> Path:
    rows = []
    for exp in experiments:
        for ds in datasets:
            summary = output_root / exp / ds / "eval" / "full_table_eval_summary.csv"
            if not summary.exists():
                continue
            try:
                df = pd.read_csv(summary, encoding="utf-8-sig")
                if not df.empty:
                    row = df.iloc[0].to_dict()
                    row["experiment"] = exp
                    row["dataset"] = ds
                    row["summary_file"] = str(summary)
                    rows.append(row)
            except Exception as e:
                print(f"[WARN] cannot read summary {summary}: {e}")

    out_file = output_root / "ablation_summary_all.csv"
    if rows:
        out = pd.DataFrame(rows)
        first_cols = ["experiment", "dataset", "Precision", "Recall", "F1", "Accuracy", "TP", "FP", "FN", "TN"]
        cols = [c for c in first_cols if c in out.columns] + [c for c in out.columns if c not in first_cols]
        out = out[cols]
        out.to_csv(out_file, index=False, encoding="utf-8-sig")
        print(f"[OK] summary saved: {out_file}")
        print(out[[c for c in first_cols if c in out.columns]].to_string(index=False))
    else:
        pd.DataFrame().to_csv(out_file, index=False, encoding="utf-8-sig")
        print(f"[WARN] no summaries collected. Empty file saved: {out_file}")
    return out_file


# ============================================================
# 7. Main
# ============================================================

def main():
    args = parse_args()

    datasets = split_csv_arg(args.datasets)
    experiments = split_csv_arg(args.experiments)

    unknown_ds = [d for d in datasets if d not in DATASETS]
    if unknown_ds:
        raise ValueError(f"Unknown datasets: {unknown_ds}")

    valid_exps = {"no_candidate_constraints", "no_pairwise_distillation"}
    unknown_exp = [e for e in experiments if e not in valid_exps]
    if unknown_exp:
        raise ValueError(f"Unknown experiments: {unknown_exp}. Valid={sorted(valid_exps)}")

    code_dir = Path(args.code_dir)
    output_root = Path(args.output_root)
    ensure_dir(output_root)

    print("========== Correction Ablation Runner ==========")
    print(f"code_dir: {code_dir}")
    print(f"output_root: {output_root}")
    print(f"datasets: {datasets}")
    print(f"experiments: {experiments}")
    print(f"dry_run: {args.dry_run}")

    write_json(output_root / "runner_config.json", {
        "code_dir": str(code_dir),
        "output_root": str(output_root),
        "datasets": datasets,
        "experiments": experiments,
        "max_llm_cells_per_dataset": args.max_llm_cells_per_dataset,
        "rule_score_min_margin": args.rule_score_min_margin,
        "rule_score_skip_same_as_dirty": args.rule_score_skip_same_as_dirty,
    })

    failures = []

    for exp in experiments:
        for ds in datasets:
            work_dir = output_root / exp / ds
            ensure_dir(work_dir)
            print(f"\n\n==================== {exp} | {ds} ====================")
            try:
                if exp == "no_candidate_constraints":
                    run_no_candidate_constraints(ds, work_dir, code_dir, args)
                elif exp == "no_pairwise_distillation":
                    run_no_pairwise_distillation(ds, work_dir, code_dir, args)
            except Exception as e:
                print(f"[ERROR] experiment failed: {exp} | {ds} | {e}")
                failures.append({"experiment": exp, "dataset": ds, "error": str(e)})
                if not args.continue_on_error:
                    raise

    if failures:
        fail_file = output_root / "ablation_failures.json"
        write_json(fail_file, failures)
        print(f"[WARN] failures saved: {fail_file}")

    collect_summaries(output_root, experiments, datasets)

    print("\n========== All requested ablations finished ==========")
    print(f"output_root: {output_root}")


if __name__ == "__main__":
    main()

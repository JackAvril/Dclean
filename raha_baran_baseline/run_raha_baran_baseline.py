#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run Raha + Baran as an end-to-end baseline.

Usage example:
python run_raha_baran_baseline.py \
  --name hospital \
  --dirty /path/to/dirty.csv \
  --clean /path/to/clean.csv \
  --out_dir ./baseline_outputs/hospital_raha_baran \
  --labeling_budget 20 \
  --baran_input raha

Notes:
1. This script follows the official benchmark style: clean.csv is used only to
   simulate tuple-level user labels for Raha/Baran, and then to evaluate results.
2. Do NOT use this as a real deployment script without clean.csv. For a paper
   baseline, this is a reasonable controlled setting if all semi-supervised
   methods receive the same labeling budget.
3. Baran can use either Raha-detected cells or oracle error locations:
   --baran_input raha   : end-to-end Raha + Baran baseline
   --baran_input oracle : oracle-detection Baran upper-bound repair baseline
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd


def patch_runtime_compatibility() -> None:
    """
    Compatibility patches for running old Raha/Baran code on newer Python/pandas/numpy.
    """
    # New pandas removed DataFrame.applymap().
    # Raha still calls applymap(), so we add it back as an alias of DataFrame.map().
    if not hasattr(pd.DataFrame, "applymap") and hasattr(pd.DataFrame, "map"):
        pd.DataFrame.applymap = pd.DataFrame.map

    # Older libraries may use removed numpy aliases.
    alias_map = {
        "int": int,
        "float": float,
        "bool": bool,
        "object": object,
        "str": str,
    }
    for name, typ in alias_map.items():
        if not hasattr(np, name):
            setattr(np, name, typ)


patch_runtime_compatibility()

try:
    import raha
except Exception as e:
    raise RuntimeError(
        "Cannot import package 'raha'. Install it first with: pip install raha\n"
        "If pip install fails on your server, use: git clone https://github.com/BigDaMa/raha.git && pip install -e raha"
    ) from e

Cell = Tuple[int, int]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_same_shape_and_columns(dirty_path: str, clean_path: str) -> None:
    dirty = pd.read_csv(dirty_path, dtype=str, keep_default_na=False, low_memory=False)
    clean = pd.read_csv(clean_path, dtype=str, keep_default_na=False, low_memory=False)
    if dirty.shape != clean.shape:
        raise ValueError(f"dirty and clean shape mismatch: dirty={dirty.shape}, clean={clean.shape}")
    if list(dirty.columns) != list(clean.columns):
        raise ValueError("dirty and clean columns mismatch. Please align column names and order first.")


def dictionary_to_csv(
    dictionary: Dict[Cell, Any],
    columns: list[str],
    output_path: Path,
    value_name: str,
) -> None:
    rows = []
    for (row_id, col_id), value in sorted(dictionary.items()):
        rows.append(
            {
                "row_id": int(row_id),
                "col_id": int(col_id),
                "column": columns[int(col_id)],
                value_name: value,
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8")


def evaluate_detection(d: "raha.Dataset", detected: Dict[Cell, Any]) -> dict:
    actual_errors = d.get_actual_errors_dictionary()
    detected_cells = set(detected.keys())
    actual_cells = set(actual_errors.keys())

    tp = len(detected_cells & actual_cells)
    fp = len(detected_cells - actual_cells)
    fn = len(actual_cells - detected_cells)
    tn = d.dataframe.shape[0] * d.dataframe.shape[1] - tp - fp - fn

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else 0.0

    return {
        "total_cells": int(d.dataframe.shape[0] * d.dataframe.shape[1]),
        "true_error_cells": int(len(actual_cells)),
        "predicted_error_cells": int(len(detected_cells)),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Accuracy": accuracy,
    }


def evaluate_repair_end_to_end(d: "raha.Dataset", corrections: Dict[Cell, Any]) -> dict:
    """
    End-to-end repair evaluation over the whole table.

    A true positive means: the cell was truly dirty and the final corrected value
    equals the clean value. A false positive means: the method changed a clean
    cell, or changed a dirty cell to a still-wrong value. A false negative means:
    a truly dirty cell is not correctly repaired after the method finishes.
    """
    dirty_df = d.dataframe.copy()
    clean_df = d.clean_dataframe.copy()
    repaired_df = dirty_df.copy()

    for cell, new_value in corrections.items():
        i, j = cell
        repaired_df.iloc[i, j] = raha.Dataset.value_normalizer(str(new_value))

    true_error_mask = dirty_df.values != clean_df.values
    repaired_correct_mask = repaired_df.values == clean_df.values
    changed_mask = dirty_df.values != repaired_df.values

    true_error_cells = int(true_error_mask.sum())
    changed_cells = int(changed_mask.sum())
    touched_cells = int(len(corrections))

    tp = int((true_error_mask & repaired_correct_mask).sum())
    # Changed cells that are not successful repairs are harmful changes.
    fp = int((changed_mask & ~((true_error_mask) & repaired_correct_mask)).sum())
    fn = int((true_error_mask & ~repaired_correct_mask).sum())
    total_cells = int(dirty_df.shape[0] * dirty_df.shape[1])
    tn = int(total_cells - tp - fp - fn)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / total_cells if total_cells else 0.0

    remaining_wrong = int((repaired_df.values != clean_df.values).sum())

    return {
        "total_cells": total_cells,
        "true_error_cells": true_error_cells,
        "touched_cells": touched_cells,
        "changed_cells": changed_cells,
        "remaining_wrong_cells_after_repair": remaining_wrong,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Accuracy": accuracy,
    }


def make_repaired_csv(d: "raha.Dataset", corrections: Dict[Cell, Any], output_path: Path) -> None:
    repaired_df = d.dataframe.copy()
    for (i, j), new_value in corrections.items():
        repaired_df.iloc[i, j] = raha.Dataset.value_normalizer(str(new_value))
    repaired_df.to_csv(output_path, index=False, encoding="utf-8")


def cleanup_previous_official_result_folder(dirty_path: str, dataset_name: str) -> None:
    """
    Raha stores cache under dirname(dirty)/raha-baran-results-{name}.
    If this folder exists, Raha may reuse previous strategy profiles.
    For repeated experiments with changed settings, cleaning it avoids stale cache.
    """
    result_folder = Path(dirty_path).resolve().parent / f"raha-baran-results-{dataset_name}"
    if result_folder.exists():
        shutil.rmtree(result_folder)


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dirty_path = str(Path(args.dirty).resolve())
    clean_path = str(Path(args.clean).resolve())
    ensure_same_shape_and_columns(dirty_path, clean_path)

    if args.clean_raha_cache:
        cleanup_previous_official_result_folder(dirty_path, args.name)

    dataset_dictionary = {
        "name": args.name,
        "path": dirty_path,
        "clean_path": clean_path,
    }

    # -------------------------
    # 1. Run Raha detection
    # -------------------------
    detection_app = raha.Detection()
    detection_app.LABELING_BUDGET = args.labeling_budget
    detection_app.USER_LABELING_ACCURACY = args.user_labeling_accuracy
    detection_app.VERBOSE = args.verbose
    detection_app.SAVE_RESULTS = True
    detection_app.CLASSIFICATION_MODEL = args.raha_classifier
    detection_app.LABEL_PROPAGATION_METHOD = args.label_propagation

    # Default official algorithms are [OD, PVD, RVD, KBVD].
    # KBVD may be slow or less useful if the built-in KATARA knowledge base is unrelated.
    if args.error_detectors:
        detection_app.ERROR_DETECTION_ALGORITHMS = [x.strip() for x in args.error_detectors.split(",") if x.strip()]

    detected_cells = detection_app.run(dataset_dictionary)

    d = raha.Dataset(dataset_dictionary)
    det_metrics = evaluate_detection(d, detected_cells)

    columns = d.dataframe.columns.tolist()
    dictionary_to_csv(detected_cells, columns, out_dir / "raha_detected_cells.csv", "dummy_value")
    with open(out_dir / "raha_detected_cells.pkl", "wb") as f:
        pickle.dump(detected_cells, f)
    with open(out_dir / "raha_detection_metrics.json", "w", encoding="utf-8") as f:
        json.dump(det_metrics, f, indent=2, ensure_ascii=False)

    # -------------------------
    # 2. Run Baran correction
    # -------------------------
    data_for_baran = raha.Dataset(dataset_dictionary)
    if args.baran_input == "raha":
        data_for_baran.detected_cells = dict(detected_cells)
    elif args.baran_input == "oracle":
        data_for_baran.detected_cells = dict(data_for_baran.get_actual_errors_dictionary())
    else:
        raise ValueError("--baran_input must be either 'raha' or 'oracle'")

    correction_app = raha.Correction()
    correction_app.LABELING_BUDGET = args.labeling_budget
    correction_app.VERBOSE = args.verbose
    correction_app.SAVE_RESULTS = True
    correction_app.CLASSIFICATION_MODEL = args.baran_classifier
    correction_app.NUM_WORKERS = args.num_workers
    correction_app.CHUNK_SIZE = args.chunk_size
    correction_app.MIN_CORRECTION_CANDIDATE_PROBABILITY = args.min_candidate_probability
    correction_app.MIN_CORRECTION_OCCURRENCE = args.min_correction_occurrence

    if args.pretrained_value_models:
        correction_app.PRETRAINED_VALUE_BASED_MODELS_PATH = str(Path(args.pretrained_value_models).resolve())

    corrected_cells = correction_app.run(data_for_baran)
    repair_metrics = evaluate_repair_end_to_end(data_for_baran, corrected_cells)

    dictionary_to_csv(corrected_cells, columns, out_dir / "baran_corrected_cells.csv", "corrected_value")
    with open(out_dir / "baran_corrected_cells.pkl", "wb") as f:
        pickle.dump(corrected_cells, f)
    with open(out_dir / "baran_repair_metrics_end_to_end.json", "w", encoding="utf-8") as f:
        json.dump(repair_metrics, f, indent=2, ensure_ascii=False)
    make_repaired_csv(data_for_baran, corrected_cells, out_dir / "baran_repaired.csv")

    summary = {
        "dataset": args.name,
        "dirty": dirty_path,
        "clean": clean_path,
        "labeling_budget": args.labeling_budget,
        "baran_input": args.baran_input,
        "raha_detection": det_metrics,
        "baran_repair_end_to_end": repair_metrics,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n========== Raha detection ==========")
    print(json.dumps(det_metrics, indent=2, ensure_ascii=False))
    print("\n========== Baran repair, end-to-end ==========")
    print(json.dumps(repair_metrics, indent=2, ensure_ascii=False))
    print(f"\nOutputs written to: {out_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Raha + Baran baseline")
    parser.add_argument("--name", required=True, help="Dataset name, e.g., hospital/flights/tax")
    parser.add_argument("--dirty", required=True, help="Path to dirty.csv")
    parser.add_argument("--clean", required=True, help="Path to clean.csv")
    parser.add_argument("--out_dir", required=True, help="Output directory")

    parser.add_argument("--labeling_budget", type=int, default=20, help="Tuple-level labeling budget used by Raha and Baran")
    parser.add_argument("--user_labeling_accuracy", type=float, default=1.0, help="Simulated user labeling accuracy for Raha")
    parser.add_argument("--baran_input", choices=["raha", "oracle"], default="raha", help="Use Raha detections or oracle error locations for Baran")

    parser.add_argument("--raha_classifier", default="GBC", choices=["ABC", "DTC", "GBC", "GNB", "KNC", "SGDC", "SVC"])
    parser.add_argument("--baran_classifier", default="ABC", choices=["ABC", "DTC", "GBC", "GNB", "KNC", "SGDC", "SVC"])
    parser.add_argument("--label_propagation", default="homogeneity", choices=["homogeneity", "majority"])
    parser.add_argument("--error_detectors", default="", help="Comma-separated detector list, e.g. OD,PVD,RVD,KBVD or PVD,RVD")

    parser.add_argument("--pretrained_value_models", default="", help="Optional Baran pretrained value model dictionary")
    parser.add_argument("--min_candidate_probability", type=float, default=0.0)
    parser.add_argument("--min_correction_occurrence", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=max(os.cpu_count() or 2, 2))
    parser.add_argument("--chunk_size", type=int, default=100)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean_raha_cache", action="store_true", help="Remove official Raha cache folder before running")
    parser.add_argument("--verbose", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())

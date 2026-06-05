#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HoloClean baseline runner for tabular data repair.

目标：
1) 读取 dirty.csv / clean.csv；
2) 可选：从 dirty.csv 自动挖掘高置信 FD，并转成 HoloClean denial constraints；
3) 调用 HoloClean 进行 error detection + repair；
4) 导出 repaired.csv；
5) 用 dirty / clean / repaired 做全表端到端修复评估。

注意：
- HoloClean 官方实现较老，推荐 Python 3.7 + PostgreSQL。
- clean.csv 只用于最终评估，不参与规则挖掘和修复。
"""
from __future__ import annotations
import argparse, json, os, re, sys, time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

def patch_runtime_compatibility() -> None:
    if not hasattr(time, "clock"):
        time.clock = time.perf_counter  # type: ignore[attr-defined]
    if not hasattr(pd.DataFrame, "applymap") and hasattr(pd.DataFrame, "map"):
        pd.DataFrame.applymap = pd.DataFrame.map  # type: ignore[attr-defined]
    for name, typ in {"int": int, "float": float, "bool": bool, "object": object, "str": str}.items():
        if not hasattr(np, name):
            setattr(np, name, typ)
patch_runtime_compatibility()

def read_csv_str(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)

def normalize_for_compare(df: pd.DataFrame, *, lower=False, strip=True) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = out[c].astype(str)
        if strip:
            out[c] = out[c].str.strip()
        if lower:
            out[c] = out[c].str.lower()
    return out

def ensure_same_schema(dirty: pd.DataFrame, clean: pd.DataFrame) -> None:
    if dirty.shape != clean.shape:
        raise ValueError(f"dirty and clean shape mismatch: dirty={dirty.shape}, clean={clean.shape}")
    if list(dirty.columns) != list(clean.columns):
        raise ValueError(f"dirty and clean columns mismatch.\ndirty={list(dirty.columns)}\nclean={list(clean.columns)}")

def safe_name(name: str) -> str:
    x = re.sub(r"[^A-Za-z0-9_]+", "_", name.strip()) or "dataset"
    if x[0].isdigit():
        x = "ds_" + x
    return x.lower()

def write_json(obj: dict, path) -> None:
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def make_holoclean_truth_file(dirty_path, clean_path, out_path, *, lower_compare=False, strip_compare=True) -> pd.DataFrame:
    dirty, clean = read_csv_str(dirty_path), read_csv_str(clean_path)
    ensure_same_schema(dirty, clean)
    d_cmp = normalize_for_compare(dirty, lower=lower_compare, strip=strip_compare)
    c_cmp = normalize_for_compare(clean, lower=lower_compare, strip=strip_compare)
    records = []
    for rid in range(len(dirty)):
        for col in dirty.columns:
            if d_cmp.at[rid, col] != c_cmp.at[rid, col]:
                records.append({"tid": rid, "attribute": col, "correct_val": clean.at[rid, col]})
    truth = pd.DataFrame(records, columns=["tid", "attribute", "correct_val"])
    truth.to_csv(out_path, index=False, encoding="utf-8")
    return truth

def parse_fd_spec(fd_text: str) -> Tuple[Tuple[str, ...], str]:
    if "->" not in fd_text:
        raise ValueError(f"Invalid FD spec: {fd_text}. Expected A->B or A,B->C")
    lhs, rhs = fd_text.split("->", 1)
    lhs_cols = tuple(x.strip() for x in re.split(r"[,+]", lhs) if x.strip())
    rhs_col = rhs.strip()
    if not lhs_cols or not rhs_col:
        raise ValueError(f"Invalid FD spec: {fd_text}")
    return lhs_cols, rhs_col

def fd_to_holoclean_dc(lhs: Sequence[str], rhs: str) -> str:
    eqs = "&".join([f"EQ(t1.{a},t2.{a})" for a in lhs])
    return f"t1&t2&{eqs}&IQ(t1.{rhs},t2.{rhs})"

def load_fd_file(fd_path) -> List[Tuple[Tuple[str, ...], str]]:
    fds = []
    for line in Path(fd_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fds.append(parse_fd_spec(line))
    return fds

def mine_single_lhs_fds(dirty: pd.DataFrame, *, max_lhs_cardinality=500, min_support=5, confidence_threshold=0.98, max_fds=200, exclude_cols=None):
    exclude = set(exclude_cols or [])
    cols = [c for c in dirty.columns if c not in exclude]
    df = dirty.copy()
    for c in cols:
        df[c] = df[c].astype(str).str.strip()
    fds = []
    for lhs in cols:
        lhs_card = df[lhs].nunique(dropna=False)
        if lhs_card <= 1 or lhs_card > max_lhs_cardinality:
            continue
        for rhs in cols:
            if rhs == lhs:
                continue
            sub = df[[lhs, rhs]].loc[df[lhs].astype(str).str.len() > 0]
            if len(sub) < min_support:
                continue
            group_sizes = sub.groupby(lhs).size()
            valid_groups = group_sizes[group_sizes >= min_support].index
            if len(valid_groups) == 0:
                continue
            sub = sub[sub[lhs].isin(valid_groups)]
            if len(sub) < min_support:
                continue
            majority_sum = sub.groupby(lhs)[rhs].agg(lambda s: s.value_counts(dropna=False).iloc[0]).sum()
            conf = float(majority_sum) / float(len(sub))
            support = int(len(sub))
            if conf >= confidence_threshold:
                fds.append(((lhs,), rhs, conf, support))
    fds.sort(key=lambda x: (x[2], x[3]), reverse=True)
    return fds[:max_fds]

def prepare_denial_constraints(*, dirty_path, out_path, dc_path=None, fd_path=None, auto_mine_fds=False, fd_confidence=0.98, fd_min_support=5, fd_max_lhs_cardinality=500, max_auto_fds=200):
    out_path = Path(out_path); constraints = []
    if dc_path:
        constraints = [x.strip() for x in Path(dc_path).read_text(encoding="utf-8").splitlines() if x.strip() and not x.strip().startswith("#")]
    elif fd_path:
        constraints = [fd_to_holoclean_dc(lhs, rhs) for lhs, rhs in load_fd_file(fd_path)]
    elif auto_mine_fds:
        dirty = read_csv_str(dirty_path)
        mined = mine_single_lhs_fds(dirty, confidence_threshold=fd_confidence, min_support=fd_min_support, max_lhs_cardinality=fd_max_lhs_cardinality, max_fds=max_auto_fds)
        constraints = [fd_to_holoclean_dc(lhs, rhs) for lhs, rhs, _, _ in mined]
        meta = [{"lhs": list(lhs), "rhs": rhs, "confidence": conf, "support": support, "dc": fd_to_holoclean_dc(lhs, rhs)} for lhs, rhs, conf, support in mined]
        write_json({"auto_mined_fds": meta}, out_path.with_suffix(".auto_fds.json"))
    out_path.write_text("\n".join(constraints) + ("\n" if constraints else ""), encoding="utf-8")
    return constraints

def import_holoclean_modules(holoclean_home: Optional[str] = None):
    if holoclean_home:
        home = str(Path(holoclean_home).resolve())
        if home not in sys.path:
            sys.path.insert(0, home)
    import holoclean  # type: ignore
    from detect import NullDetector, ViolationDetector  # type: ignore
    from repair.featurize import InitAttrFeaturizer, OccurAttrFeaturizer, FreqFeaturizer, ConstraintFeaturizer  # type: ignore
    return holoclean, NullDetector, ViolationDetector, InitAttrFeaturizer, OccurAttrFeaturizer, FreqFeaturizer, ConstraintFeaturizer

def load_external_detected_cells(detected_cells_path, dirty_path, hc_session) -> pd.DataFrame:
    """
    Load externally detected dirty cells and convert them into HoloClean's dk_cells format.

    Supported input columns:
      - row id: row_id / tuple_id / _tid_ / tid / row
      - attribute: column / attribute / col / attr

    Example input from your rule-pool shrinking stage:
      row_id,column,value,violation_count,conflict_score,...
    """
    ext = read_csv_str(detected_cells_path)
    dirty = read_csv_str(dirty_path)

    row_col_candidates = ["row_id", "tuple_id", "_tid_", "tid", "row"]
    attr_col_candidates = ["column", "attribute", "col", "attr"]

    row_col = next((c for c in row_col_candidates if c in ext.columns), None)
    attr_col = next((c for c in attr_col_candidates if c in ext.columns), None)

    if row_col is None or attr_col is None:
        raise ValueError(
            "detected_cells must contain one row-id column "
            f"{row_col_candidates} and one attribute column {attr_col_candidates}. "
            f"Got columns: {list(ext.columns)}"
        )

    errors_df = pd.DataFrame()
    errors_df["_tid_"] = pd.to_numeric(ext[row_col], errors="coerce")
    errors_df["attribute"] = ext[attr_col].astype(str).str.strip()

    # Drop malformed rows.
    errors_df = errors_df.dropna(subset=["_tid_", "attribute"]).copy()
    errors_df["_tid_"] = errors_df["_tid_"].astype(int)

    # HoloClean expects zero-based tuple ids. Your current files use zero-based row_id.
    # We still filter by valid row range to avoid HoloClean crashing on bad ids.
    valid_attrs = set(dirty.columns)
    n_rows = len(dirty)
    errors_df = errors_df[
        (errors_df["_tid_"] >= 0)
        & (errors_df["_tid_"] < n_rows)
        & (errors_df["attribute"].isin(valid_attrs))
    ].drop_duplicates().reset_index(drop=True)

    if len(errors_df) == 0:
        raise ValueError(
            "External detected_cells file was loaded, but no valid cells remain after filtering. "
            "Please check row_id base and column names."
        )

    errors_df["_cid_"] = errors_df.apply(
        lambda x: hc_session.ds.get_cell_id(int(x["_tid_"]), x["attribute"]),
        axis=1,
    )

    # Keep exact HoloClean columns and stable order.
    errors_df = errors_df[["_tid_", "attribute", "_cid_"]].drop_duplicates().reset_index(drop=True)
    return errors_df


def run_holoclean(*, dataset_name, dirty_path, dc_path, truth_path, holoclean_home, db_name, db_user, db_pwd, db_host, epochs, threads, batch_size, timeout, verbose, weak_label_thresh, domain_thresh_1, domain_thresh_2, max_domain, cor_strength, nb_cor_strength, detected_cells_path=None):
    holoclean, NullDetector, ViolationDetector, InitAttrFeaturizer, OccurAttrFeaturizer, FreqFeaturizer, ConstraintFeaturizer = import_holoclean_modules(holoclean_home)
    hc = holoclean.HoloClean(db_name=db_name, db_user=db_user, db_pwd=db_pwd, db_host=db_host, domain_thresh_1=domain_thresh_1, domain_thresh_2=domain_thresh_2, weak_label_thresh=weak_label_thresh, max_domain=max_domain, cor_strength=cor_strength, nb_cor_strength=nb_cor_strength, epochs=epochs, weight_decay=0.01, learning_rate=0.001, threads=threads, batch_size=batch_size, verbose=verbose, timeout=timeout, feature_norm=False, weight_norm=False, print_fw=False).session
    hc.load_data(safe_name(dataset_name), str(dirty_path))
    hc.load_dcs(str(dc_path))
    hc.ds.set_constraints(hc.get_dcs())

    if detected_cells_path:
        errors_df = load_external_detected_cells(detected_cells_path, dirty_path, hc)
        hc.detect_engine.store_detected_errors(errors_df)
        print(f"[INFO] Using external detected cells: {len(errors_df)} -> {detected_cells_path}")
    else:
        hc.detect_errors([NullDetector(), ViolationDetector()])

    hc.setup_domain()
    featurizers = [
        InitAttrFeaturizer(),
        OccurAttrFeaturizer(),
        FreqFeaturizer(),
    ]

    try:
        dc_text = Path(dc_path).read_text(encoding="utf-8").strip()
    except Exception:
        dc_text = ""

    if dc_text:
        featurizers.append(ConstraintFeaturizer())
    else:
        print("[WARN] No denial constraints found. Skipping ConstraintFeaturizer.")

    hc.repair_errors(featurizers)
    hc_report = None
    try:
        hc_report = hc.evaluate(fpath=str(truth_path), tid_col="tid", attr_col="attribute", val_col="correct_val")
    except Exception as e:
        print(f"[WARN] HoloClean built-in evaluate failed: {e}", file=sys.stderr)
    repaired = None
    if getattr(hc.ds, "repaired_data", None) is not None:
        repaired = hc.ds.repaired_data.df.copy()
    if repaired is None:
        raise RuntimeError("Cannot find repaired dataframe from hc.ds.repaired_data.df after hc.repair_errors().")
    return repaired, hc_report

def evaluate_full_table(dirty_path, clean_path, repaired_df: pd.DataFrame, *, lower_compare=False, strip_compare=True):
    dirty, clean = read_csv_str(dirty_path), read_csv_str(clean_path)
    ensure_same_schema(dirty, clean)
    repaired = repaired_df.copy()
    if "_tid_" in repaired.columns:
        repaired = repaired.sort_values("_tid_").drop(columns=["_tid_"]).reset_index(drop=True)
    missing = [c for c in dirty.columns if c not in repaired.columns]
    if missing:
        raise ValueError(f"repaired dataframe missing columns: {missing}")
    repaired = repaired[list(dirty.columns)].astype(str).reset_index(drop=True)
    if repaired.shape != dirty.shape:
        raise ValueError(f"repaired shape mismatch: repaired={repaired.shape}, dirty={dirty.shape}")
    d_cmp = normalize_for_compare(dirty, lower=lower_compare, strip=strip_compare)
    c_cmp = normalize_for_compare(clean, lower=lower_compare, strip=strip_compare)
    r_cmp = normalize_for_compare(repaired, lower=lower_compare, strip=strip_compare)
    true_error_mask = d_cmp.ne(c_cmp)
    changed_mask = r_cmp.ne(d_cmp)
    correct_repair_mask = true_error_mask & changed_mask & r_cmp.eq(c_cmp)
    tp = int(correct_repair_mask.values.sum())
    changed_cells = int(changed_mask.values.sum())
    true_error_cells = int(true_error_mask.values.sum())
    fp = int(changed_cells - tp)
    fn = int(true_error_cells - tp)
    total_cells = int(dirty.shape[0] * dirty.shape[1])
    tn = int(total_cells - tp - fp - fn)
    remaining_wrong = int(r_cmp.ne(c_cmp).values.sum())
    precision = tp / changed_cells if changed_cells else 0.0
    recall = tp / true_error_cells if true_error_cells else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / total_cells if total_cells else 0.0
    details = []
    for rid in range(len(dirty)):
        for col in dirty.columns:
            if bool(true_error_mask.at[rid, col]) or bool(changed_mask.at[rid, col]):
                details.append({"row_id": rid, "column": col, "dirty": dirty.at[rid, col], "clean": clean.at[rid, col], "repaired": repaired.at[rid, col], "is_true_error": bool(true_error_mask.at[rid, col]), "is_changed": bool(changed_mask.at[rid, col]), "is_correct_repair": bool(correct_repair_mask.at[rid, col])})
    metrics = {"total_cells": total_cells, "true_error_cells": true_error_cells, "changed_cells": changed_cells, "remaining_wrong_cells_after_repair": remaining_wrong, "TP": tp, "FP": fp, "FN": fn, "TN": tn, "Precision": precision, "Recall": recall, "F1": f1, "Accuracy": accuracy}
    return metrics, pd.DataFrame(details)

def build_arg_parser():
    p = argparse.ArgumentParser(description="Run HoloClean baseline on dirty/clean CSV.")
    p.add_argument("--name", required=True)
    p.add_argument("--dirty", required=True)
    p.add_argument("--clean", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--holoclean_home", default=None, help="Path to cloned HoloClean repo root")
    p.add_argument("--dc_path", default=None)
    p.add_argument("--fd_path", default=None)
    p.add_argument("--auto_mine_fds", action="store_true")
    p.add_argument(
        "--detected_cells",
        default=None,
        help=(
            "Optional CSV file of externally detected dirty cells. "
            "Expected columns include row_id/tuple_id/_tid_/tid/row and "
            "column/attribute/col/attr. If provided, HoloClean native detection is skipped."
        ),
    )
    p.add_argument("--fd_confidence", type=float, default=0.98)
    p.add_argument("--fd_min_support", type=int, default=5)
    p.add_argument("--fd_max_lhs_cardinality", type=int, default=500)
    p.add_argument("--max_auto_fds", type=int, default=200)
    p.add_argument("--db_name", default="holo")
    p.add_argument("--db_user", default="holocleanuser")
    p.add_argument("--db_pwd", default="abcd1234")
    p.add_argument("--db_host", default="localhost")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--timeout", type=int, default=180000)
    p.add_argument("--weak_label_thresh", type=float, default=0.99)
    p.add_argument("--domain_thresh_1", type=float, default=0.0)
    p.add_argument("--domain_thresh_2", type=float, default=0.0)
    p.add_argument("--max_domain", type=int, default=10000)
    p.add_argument("--cor_strength", type=float, default=0.6)
    p.add_argument("--nb_cor_strength", type=float, default=0.8)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--lower_compare", action="store_true")
    p.add_argument("--no_strip_compare", action="store_true")
    return p

def main():
    args = build_arg_parser().parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    dirty, clean = read_csv_str(args.dirty), read_csv_str(args.clean)
    ensure_same_schema(dirty, clean)
    truth_path = out_dir / "holoclean_truth_cells.csv"
    truth = make_holoclean_truth_file(args.dirty, args.clean, truth_path, lower_compare=args.lower_compare, strip_compare=not args.no_strip_compare)
    dc_out_path = out_dir / "holoclean_constraints.txt"
    constraints = prepare_denial_constraints(dirty_path=args.dirty, out_path=dc_out_path, dc_path=args.dc_path, fd_path=args.fd_path, auto_mine_fds=args.auto_mine_fds, fd_confidence=args.fd_confidence, fd_min_support=args.fd_min_support, fd_max_lhs_cardinality=args.fd_max_lhs_cardinality, max_auto_fds=args.max_auto_fds)
    print(f"[INFO] Dataset: {args.name}")
    print(f"[INFO] truth cells: {len(truth)} -> {truth_path}")
    print(f"[INFO] denial constraints: {len(constraints)} -> {dc_out_path}")
    try:
        repaired_df, hc_report = run_holoclean(dataset_name=args.name, dirty_path=args.dirty, dc_path=dc_out_path, truth_path=truth_path, holoclean_home=args.holoclean_home, db_name=args.db_name, db_user=args.db_user, db_pwd=args.db_pwd, db_host=args.db_host, epochs=args.epochs, threads=args.threads, batch_size=args.batch_size, timeout=args.timeout, verbose=args.verbose, weak_label_thresh=args.weak_label_thresh, domain_thresh_1=args.domain_thresh_1, domain_thresh_2=args.domain_thresh_2, max_domain=args.max_domain, cor_strength=args.cor_strength, nb_cor_strength=args.nb_cor_strength, detected_cells_path=args.detected_cells)
    except Exception as e:
        msg = str(e)
        if "No attribute contains erroneous cells" in msg:
            print("[WARN] HoloClean detected 0 erroneous cells. Falling back to no-op repair for end-to-end evaluation.")
            repaired_df = read_csv_str(args.dirty)
            hc_report = {
                "status": "noop_no_detected_errors",
                "reason": msg,
                "detected_cells": 0,
                "repaired_cells": 0,
            }
        else:
            raise
    repaired_to_save = repaired_df.copy()
    if "_tid_" in repaired_to_save.columns:
        repaired_to_save = repaired_to_save.sort_values("_tid_").drop(columns=["_tid_"]).reset_index(drop=True)
    repaired_to_save = repaired_to_save[list(dirty.columns)]
    repaired_path = out_dir / "holoclean_repaired.csv"
    repaired_to_save.to_csv(repaired_path, index=False, encoding="utf-8")
    metrics, details = evaluate_full_table(args.dirty, args.clean, repaired_df, lower_compare=args.lower_compare, strip_compare=not args.no_strip_compare)
    write_json(metrics, out_dir / "holoclean_repair_metrics_end_to_end.json")
    details.to_csv(out_dir / "holoclean_repair_details.csv", index=False, encoding="utf-8")
    hc_report_json = None
    if hc_report is not None:
        try:
            hc_report_json = hc_report._asdict()
        except Exception:
            hc_report_json = str(hc_report)
        write_json({"holoclean_builtin_report": hc_report_json}, out_dir / "holoclean_builtin_report.json")
    summary = {"dataset": args.name, "dirty": str(args.dirty), "clean": str(args.clean), "out_dir": str(out_dir), "truth_cells": int(len(truth)), "constraints_count": int(len(constraints)), "constraints_file": str(dc_out_path), "detected_cells_file": str(args.detected_cells) if args.detected_cells else None, "repaired_csv": str(repaired_path), "end_to_end_metrics": metrics, "holoclean_builtin_report": hc_report_json}
    write_json(summary, out_dir / "summary.json")
    print("\n========== HoloClean End-to-End Repair Metrics ==========")
    for k, v in metrics.items(): print(f"{k}: {v}")
    print(f"\n[OK] outputs saved to: {out_dir}")

if __name__ == "__main__":
    main()

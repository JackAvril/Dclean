#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_detection_ablations.py

批量运行 6 个数据集的错误检测消融实验。

执行链路：
python build_rule_pool.py
python shrink_search_space.py
python summarize_full_contexts.py
python cluster_and_sample_candidates.py
python annotate_with_deepseek.py  # 若不存在，自动尝试 annotate_with_llm.py
python label_propagation_for_candidates.py
python build_final_training_set.py
python build_loading_set.py
python control_iteration.py
python evaluate.py

消融：
full
no_evidence
no_label_propagation
no_hardcase_feedback
no_teacher

重要设计：
1. 数据集级并行：不同数据集可并行，同一数据集内部阶段仍严格串行。
2. 指标只从本次 run_dir/snapshots/evaluation_summary.csv 读取，避免读取旧结果。
3. 如果某个 variant status=failed，则默认不填 P/R/F1，避免旧指标误导。
4. no_teacher 会跳过 DeepSeek 标注，并自动用规则弱标签生成 candidate_sampled_labeled_<dataset>.csv/jsonl。
5. 该总控脚本通过环境变量和 ablation_flags.json 传递消融开关；
   各业务脚本需要读取这些开关，消融才会真正生效。
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_ROOT = "/mnt/mydata/dq/projects/Splittree/test"

DATASETS = ["hospital", "soccer", "adult", "rayyan", "beers", "flights"]

DATASET_DIR_CANDIDATES = {
    "hospital": ["Error Detection hospital", "Error Detection Hospital", "hospital"],
    "soccer": ["Error Detection soccer", "Error Detection Soccer", "soccer"],
    "adult": ["Error Detection Adult", "Error Detection adult", "adult"],
    "rayyan": ["Error Detection rayyan", "Error Detection Rayyan", "rayyan"],
    "beers": ["Error Detection beers", "Error Detection Beers", "beers"],
    "flights": ["Error Detection flights", "Error Detection Flights", "flights"],
}

PIPELINE = [
    "build_rule_pool.py",
    "shrink_search_space.py",
    "summarize_full_contexts.py",
    "cluster_and_sample_candidates.py",
    "annotate_with_deepseek.py",
    "label_propagation_for_candidates.py",
    "build_final_training_set.py",
    "build_loading_set.py",
    "control_iteration.py",
    "evaluate.py",
]

VALID_VARIANTS = [
    "full",
    "no_evidence",
    "no_label_propagation",
    "no_hardcase_feedback",
    "no_teacher",
]

VARIANT_DISPLAY_NAME = {
    "full": "Full / Ours",
    "no_evidence": "w/o Evidence",
    "no_label_propagation": "w/o Label Propagation",
    "no_hardcase_feedback": "w/o Hard-case Feedback",
    "no_teacher": "w/o Teacher",
}

SCRIPT_ALIASES = {
    "annotate_with_deepseek.py": [
        "annotate_with_deepseek.py",
        "annotate_with_llm.py",
        "annotate_with_llm_deepseek.py",
    ],
    "build_final_training_set.py": [
        "build_final_training_set.py",
        "build_final_train_dataset.py",
        "build_final_train_set.py",
        "build_training_set.py",
        "build_final_dataset.py",
    ],
    "build_loading_set.py": [
        "build_loading_set.py",
        "build_infer_candidate_features_from_whitelist.py",
        "build_inference_set.py",
        "build_infer_set.py",
    ],
}


@dataclass
class RunResult:
    dataset: str
    variant: str
    dataset_dir: str
    run_dir: str
    status: str
    failed_stage: str
    elapsed_sec: float
    summary_path: str
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    candidate_coverage: Optional[float] = None
    candidate_precision: Optional[float] = None
    final_predicted_errors: Optional[int] = None
    tp: Optional[int] = None
    fp: Optional[int] = None
    fn: Optional[int] = None
    error_message: str = ""


def now_str() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def safe_mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any):
    safe_mkdir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def find_dataset_dir(root: Path, dataset: str) -> Path:
    for name in DATASET_DIR_CANDIDATES.get(dataset, [dataset]):
        p = root / name
        if p.exists() and p.is_dir():
            return p

    lower_dataset = dataset.lower()
    for p in root.iterdir():
        if p.is_dir() and lower_dataset in p.name.lower() and "error detection" in p.name.lower():
            return p

    raise FileNotFoundError(f"找不到数据集目录: dataset={dataset}, root={root}")


def resolve_script(dataset_dir: Path, script_name: str) -> Optional[Path]:
    p = dataset_dir / script_name
    if p.exists():
        return p

    for cand in SCRIPT_ALIASES.get(script_name, []):
        p = dataset_dir / cand
        if p.exists():
            return p

    fuzzy_patterns = []
    if script_name == "build_final_training_set.py":
        fuzzy_patterns = ["build*final*train*.py", "build*training*set*.py", "*final*training*.py"]
    elif script_name == "build_loading_set.py":
        fuzzy_patterns = ["build*loading*set*.py", "build*infer*.py", "*inference*set*.py"]
    elif script_name == "annotate_with_deepseek.py":
        fuzzy_patterns = ["annotate*deepseek*.py", "annotate*llm*.py"]

    for pat in fuzzy_patterns:
        hits = sorted(dataset_dir.glob(pat))
        if hits:
            return hits[0]

    return None


def to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


def to_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "":
            return None
        return int(float(s))
    except Exception:
        return None


def read_csv_first_row(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            return row
    return {}


def extract_metrics_from_summary(summary_path: Optional[Path]) -> Dict[str, Any]:
    if summary_path is None or not summary_path.exists():
        return {}

    row = read_csv_first_row(summary_path)
    if not row:
        return {}

    # 宽表格式
    key_map = {
        "precision": ["precision", "Precision", "final_precision", "Final Precision"],
        "recall": ["recall", "Recall", "final_recall", "Final Recall"],
        "f1": ["f1", "F1", "final_f1", "Final F1"],
        "candidate_coverage": ["candidate_coverage", "Candidate Coverage", "candidate_recall"],
        "candidate_precision": ["candidate_precision", "candidate_only_precision", "Candidate-Only Precision"],
        "final_predicted_errors": ["final_predicted_errors", "final_pred_error_count", "predicted_error_count"],
        "tp": ["tp", "TP"],
        "fp": ["fp", "FP"],
        "fn": ["fn", "FN"],
    }

    out: Dict[str, Any] = {}
    for target, keys in key_map.items():
        val = None
        for k in keys:
            if k in row:
                val = row[k]
                break
        if target in {"final_predicted_errors", "tp", "fp", "fn"}:
            out[target] = to_int(val)
        else:
            out[target] = to_float(val)

    # 长表格式 metric,value
    if ("metric" in row and "value" in row) or ("name" in row and "value" in row):
        metric_col = "metric" if "metric" in row else "name"
        value_col = "value"
        table: Dict[str, Any] = {}
        with summary_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                table[str(r.get(metric_col, "")).strip()] = r.get(value_col)

        aliases = {
            "precision": ["precision", "final_precision", "Final Precision"],
            "recall": ["recall", "final_recall", "Final Recall"],
            "f1": ["f1", "final_f1", "Final F1"],
            "candidate_coverage": ["candidate_coverage", "Candidate Coverage"],
            "candidate_precision": ["candidate_precision", "candidate_only_precision", "Candidate-Only Precision"],
            "tp": ["tp", "TP"],
            "fp": ["fp", "FP"],
            "fn": ["fn", "FN"],
        }
        for target, keys in aliases.items():
            for k in keys:
                if k in table:
                    out[target] = to_int(table[k]) if target in {"tp", "fp", "fn"} else to_float(table[k])
                    break

    return out


def find_latest_file_since(dataset_dir: Path, filename_glob: str, since_ts: float) -> Optional[Path]:
    hits = list(dataset_dir.rglob(filename_glob))
    hits = [p for p in hits if p.is_file() and p.stat().st_mtime >= since_ts - 1.0]
    if not hits:
        return None
    hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0]


def find_latest_file_any(dataset_dir: Path, filename_glob: str) -> Optional[Path]:
    hits = list(dataset_dir.rglob(filename_glob))
    hits = [p for p in hits if p.is_file()]
    if not hits:
        return None
    hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0]


def copy_if_exists(src: Optional[Path], dst: Path):
    if src is not None and src.exists():
        safe_mkdir(dst.parent)
        shutil.copy2(src, dst)


def snapshot_outputs(dataset_dir: Path, run_dir: Path, since_ts: float):
    """只复制本次运行后新产生/更新的结果，避免把旧输出误认为本次结果。"""
    output_globs = [
        "evaluation_summary*.csv",
        "candidate_metrics_by_column*.csv",
        "detector_metrics_by_column*.csv",
        "final_metrics_by_column*.csv",
        "tp_details*.csv",
        "fp_details*.csv",
        "fn_details*.csv",
        "detector_tp_details*.csv",
        "detector_fp_details*.csv",
        "detector_fn_details*.csv",
        "inference_all_predictions.csv",
        "evidence_reliability*.csv",
        "evidence_reliability*.json",
    ]

    snap_dir = run_dir / "snapshots"
    safe_mkdir(snap_dir)

    copied = []
    for pat in output_globs:
        p = find_latest_file_since(dataset_dir, pat, since_ts)
        if p is not None:
            dst = snap_dir / p.name
            shutil.copy2(p, dst)
            copied.append(str(dst))

    return copied


def find_snapshot_summary(run_dir: Path) -> Optional[Path]:
    snap = run_dir / "snapshots"
    if not snap.exists():
        return None
    hits = list(snap.glob("evaluation_summary*.csv"))
    if not hits:
        return None
    hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0]


def run_command(cmd: List[str], cwd: Path, env: Dict[str, str], log_path: Path, dry_run: bool = False) -> int:
    safe_mkdir(log_path.parent)

    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"[CMD] {' '.join(cmd)}\n")
        log.write(f"[CWD] {cwd}\n")
        log.write(f"[TIME] {now_str()}\n\n")
        log.flush()

        if dry_run:
            log.write("[DRY-RUN] skipped execution.\n")
            return 0

        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)

        proc.wait()
        log.write(f"\n[EXIT_CODE] {proc.returncode}\n")
        return int(proc.returncode)


def build_ablation_flags(dataset: str, variant: str) -> Dict[str, Any]:
    flags = {
        "dataset": dataset,
        "variant": variant,
        "variant_display_name": VARIANT_DISPLAY_NAME.get(variant, variant),

        "use_structured_evidence": True,
        "use_multi_view_context": True,
        "use_neighbor_evidence": True,
        "use_bayesian_evidence": True,
        "use_evidence_reliability_gate": True,
        "use_precision_post_gate": True,

        "use_teacher": True,
        "use_deepseek_teacher": True,
        "use_llm_labels": True,
        "use_weak_rule_teacher": False,

        "use_label_propagation": True,
        "use_propagated_labels": True,

        "use_hardcase_feedback": True,
        "use_active_loop": True,
        "max_active_iterations": None,
        "sampling_strategy": "cluster",
    }

    if variant == "no_evidence":
        flags.update({
            "use_structured_evidence": False,
            "use_multi_view_context": False,
            "use_neighbor_evidence": False,
            "use_bayesian_evidence": False,
            "use_evidence_reliability_gate": False,
            "use_precision_post_gate": False,
        })

    elif variant == "no_label_propagation":
        flags.update({
            "use_label_propagation": False,
            "use_propagated_labels": False,
        })

    elif variant == "no_hardcase_feedback":
        flags.update({
            "use_hardcase_feedback": False,
            "use_active_loop": False,
            "max_active_iterations": 0,
        })

    elif variant == "no_teacher":
        flags.update({
            "use_teacher": False,
            "use_deepseek_teacher": False,
            "use_llm_labels": False,
            "use_weak_rule_teacher": True,
            "use_hardcase_feedback": False,
            "use_active_loop": False,
            "max_active_iterations": 0,
        })

    elif variant == "full":
        pass

    else:
        raise ValueError(f"Unknown variant: {variant}")

    return flags


def build_env(base_env: Dict[str, str], dataset: str, variant: str, flags: Dict[str, Any], run_dir: Path) -> Dict[str, str]:
    env = dict(base_env)
    env["ABLATION_MODE"] = variant
    env["DATASET_NAME"] = dataset
    env["ABLATION_RUN_DIR"] = str(run_dir)
    env["ABLATION_FLAGS_JSON"] = str(run_dir / "ablation_flags.json")

    def b(name: str, key: str):
        env[name] = "1" if bool(flags.get(key)) else "0"

    b("USE_STRUCTURED_EVIDENCE", "use_structured_evidence")
    b("USE_MULTI_VIEW_CONTEXT", "use_multi_view_context")
    b("USE_NEIGHBOR_EVIDENCE", "use_neighbor_evidence")
    b("USE_BAYESIAN_EVIDENCE", "use_bayesian_evidence")
    b("USE_EVIDENCE_RELIABILITY_GATE", "use_evidence_reliability_gate")
    b("USE_PRECISION_POST_GATE", "use_precision_post_gate")

    b("USE_TEACHER", "use_teacher")
    b("USE_DEEPSEEK_TEACHER", "use_deepseek_teacher")
    b("USE_LLM_LABELS", "use_llm_labels")
    b("USE_WEAK_RULE_TEACHER", "use_weak_rule_teacher")

    b("USE_LABEL_PROPAGATION", "use_label_propagation")
    b("USE_PROPAGATED_LABELS", "use_propagated_labels")

    b("USE_HARDCASE_FEEDBACK", "use_hardcase_feedback")
    b("USE_ACTIVE_LOOP", "use_active_loop")

    env["SAMPLING_STRATEGY"] = str(flags.get("sampling_strategy", "cluster"))
    if flags.get("max_active_iterations") is not None:
        env["MAX_ACTIVE_ITERATIONS"] = str(flags["max_active_iterations"])

    return env


def should_skip_script_for_variant(script_name: str, variant: str) -> bool:
    if variant == "no_teacher" and script_name == "annotate_with_deepseek.py":
        return True
    if variant == "no_label_propagation" and script_name == "label_propagation_for_candidates.py":
        return True
    return False


def find_candidate_sampled_csv(dataset_dir: Path, dataset: str) -> Optional[Path]:
    preferred = dataset_dir / f"candidate_sampled_{dataset}.csv"
    if preferred.exists():
        return preferred

    hits = list(dataset_dir.glob("candidate_sampled*.csv"))
    hits = [p for p in hits if "labeled" not in p.name.lower()]
    if not hits:
        return None
    hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0]


def weak_label_from_rule(row: Dict[str, str]) -> Tuple[str, int, float, str, str]:
    rule = str(row.get("main_rule_type", "") or "").strip()

    strong_error_rules = {
        "not_null",
        "high_risk_missing",
        "birthyear_numeric_range",
        "season_numeric_range",
        "birthyear_season_age_consistency",
        "position_domain",
        "name_pattern",
        "surname_pattern",
        "manager_name_pattern",
        "birthplace_location_pattern",
        "city_location_pattern",
        "stadium_name_pattern",
        "generic_regex",
        "pattern",
    }

    weak_or_noisy_rules = {
        "soft_functional_dependency",
        "functional_dependency",
        "rare_value",
        "rare_pattern",
        "global_dominant_value",
        "dominant_value_by_context",
        "dominant_value_by_context_pair",
        "team_context_dominant",
        "team_season_manager_consistency_hint",
        "city_stadium_consistency_hint",
        "team_name_pattern",
    }

    if rule in strong_error_rules:
        return "error", 1, 0.75, "weak_rule_error", f"weak teacher labels {rule} as error"

    for flag in [
        "birthyear_value_invalid",
        "season_value_invalid",
        "position_value_invalid",
        "position_needs_canonicalization",
        "birthyear_season_age_conflict",
    ]:
        v = row.get(flag)
        if str(v).strip() in {"1", "1.0", "true", "True"}:
            return "error", 1, 0.78, "weak_feature_error", f"weak teacher labels {flag}=1 as error"

    if rule in weak_or_noisy_rules:
        return "correct", 0, 0.60, "weak_rule_correct", f"weak teacher treats {rule} as insufficient evidence"

    return "correct", 0, 0.55, "weak_default_correct", "weak teacher default: insufficient strong evidence"


def generate_weak_teacher_labels(dataset_dir: Path, dataset: str, run_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    sampled_csv = find_candidate_sampled_csv(dataset_dir, dataset)
    if sampled_csv is None:
        print(f"[no_teacher] 找不到 candidate_sampled CSV，跳过弱标签生成: dataset={dataset}")
        return None, None

    out_csv = dataset_dir / f"candidate_sampled_labeled_{dataset}.csv"
    out_jsonl = dataset_dir / f"candidate_sampled_labeled_{dataset}.jsonl"

    rows = []
    with sampled_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label, label_binary, conf, error_type, reason = weak_label_from_rule(row)
            row["label"] = label
            row["label_binary"] = str(label_binary)
            row["is_error"] = "true" if label_binary == 1 else "false"
            row["confidence"] = f"{conf:.4f}"
            row["error_type"] = error_type
            row["reason_short"] = reason
            row["reason_detailed"] = reason
            row["evidence_used"] = "weak_rule"
            row["suggested_correct_value"] = ""
            row["needs_human_review"] = "true"
            row["label_source"] = "weak_rule_teacher"
            rows.append(row)

    if not rows:
        return None, None

    final_fields = list(rows[0].keys())
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=final_fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            obj = {
                "row_id": int(float(row.get("row_id", 0) or 0)),
                "column": row.get("column"),
                "value": row.get("value"),
                "label_result": {
                    "label": row["label"],
                    "is_error": bool(int(row["label_binary"])),
                    "confidence": float(row["confidence"]),
                    "error_type": row["error_type"],
                    "reason_short": row["reason_short"],
                    "reason_detailed": row["reason_detailed"],
                    "evidence_used": ["weak_rule"],
                    "suggested_correct_value": None,
                    "needs_human_review": True,
                },
                "summary_context": row,
                "label_source": "weak_rule_teacher",
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    copy_if_exists(out_csv, run_dir / "weak_teacher" / out_csv.name)
    copy_if_exists(out_jsonl, run_dir / "weak_teacher" / out_jsonl.name)

    print(f"[no_teacher] weak labeled CSV : {out_csv}")
    print(f"[no_teacher] weak labeled JSONL: {out_jsonl}")
    return out_csv, out_jsonl


def preflight_check_dataset(root: Path, dataset: str) -> Dict[str, Any]:
    dataset_dir = find_dataset_dir(root, dataset)
    stages = []
    missing = []
    for script_name in PIPELINE:
        p = resolve_script(dataset_dir, script_name)
        stages.append({
            "stage": script_name,
            "exists": p is not None,
            "resolved_script": str(p.name) if p is not None else "",
        })
        if p is None:
            missing.append(script_name)

    return {
        "dataset": dataset,
        "dataset_dir": str(dataset_dir),
        "missing": missing,
        "stages": stages,
    }


def write_preflight_report(root: Path, datasets: List[str], output_root: Path) -> Path:
    rows = []
    report = []
    for dataset in datasets:
        try:
            info = preflight_check_dataset(root, dataset)
            report.append(info)
            for st in info["stages"]:
                rows.append({
                    "dataset": dataset,
                    "dataset_dir": info["dataset_dir"],
                    "stage": st["stage"],
                    "exists": int(st["exists"]),
                    "resolved_script": st["resolved_script"],
                    "error": "",
                })
        except Exception as e:
            report.append({
                "dataset": dataset,
                "dataset_dir": "",
                "missing": PIPELINE,
                "error": str(e),
                "stages": [],
            })
            rows.append({
                "dataset": dataset,
                "dataset_dir": "",
                "stage": "__DATASET_DIR__",
                "exists": 0,
                "resolved_script": "",
                "error": str(e),
            })

    safe_mkdir(output_root)
    json_path = output_root / "preflight_report.json"
    csv_path = output_root / "preflight_report.csv"
    write_json(json_path, report)

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = ["dataset", "dataset_dir", "stage", "exists", "resolved_script", "error"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[OK] preflight json: {json_path}")
    print(f"[OK] preflight csv : {csv_path}")
    for info in report:
        if info.get("error"):
            print(f"[PREFLIGHT] {info.get('dataset')}: ERROR {info.get('error')}")
        elif info.get("missing"):
            print(f"[PREFLIGHT] {info.get('dataset')}: missing {info.get('missing')}")
        else:
            print(f"[PREFLIGHT] {info.get('dataset')}: all stages found")
    return csv_path


def run_one(
    root: Path,
    dataset: str,
    variant: str,
    output_root: Path,
    python_bin: str,
    dry_run: bool,
    resume: bool,
    start_from: Optional[str],
    stop_after: Optional[str],
    on_missing_stage: str = "skip",
) -> RunResult:
    t0 = time.time()
    dataset_dir = find_dataset_dir(root, dataset)
    run_dir = output_root / dataset / variant
    log_dir = run_dir / "logs"
    safe_mkdir(log_dir)

    flags = build_ablation_flags(dataset, variant)
    write_json(run_dir / "ablation_flags.json", flags)
    write_json(dataset_dir / "ablation_flags.json", flags)
    env = build_env(os.environ, dataset, variant, flags, run_dir)

    status = "success"
    failed_stage = ""
    error_message = ""

    print("\n" + "=" * 80)
    print(f"[RUN] dataset={dataset}, variant={variant}, dir={dataset_dir}")
    print("=" * 80)

    try:
        started = start_from is None

        for script_name in PIPELINE:
            if start_from is not None and script_name == start_from:
                started = True

            if not started:
                print(f"[SKIP before start_from] {script_name}")
                continue

            if should_skip_script_for_variant(script_name, variant):
                print(f"[SKIP for {variant}] {script_name}")
                if variant == "no_teacher" and script_name == "annotate_with_deepseek.py" and not dry_run:
                    generate_weak_teacher_labels(dataset_dir, dataset, run_dir)
                continue

            script_path = resolve_script(dataset_dir, script_name)
            if script_path is None:
                msg = f"Missing script: {script_name} in {dataset_dir}"
                if on_missing_stage == "fail":
                    failed_stage = script_name
                    raise FileNotFoundError(msg)
                print(f"[WARN] {msg}; skip this stage.")
                with (log_dir / f"{Path(script_name).stem}.missing.log").open("w", encoding="utf-8") as log:
                    log.write(f"[WARN] {msg}\n")
                continue

            if script_path.name != script_name:
                print(f"[ALIAS] stage {script_name} -> {script_path.name}")

            log_path = log_dir / f"{script_path.stem}.log"
            if resume and log_path.exists():
                txt = log_path.read_text(encoding="utf-8", errors="ignore")
                if "[EXIT_CODE] 0" in txt:
                    print(f"[RESUME skip] {script_path.name}")
                    if stop_after is not None and script_name == stop_after:
                        break
                    continue

            print(f"\n[STAGE] {script_path.name}")
            failed_stage = script_name
            rc = run_command([python_bin, script_path.name], dataset_dir, env, log_path, dry_run=dry_run)
            if rc != 0:
                raise RuntimeError(f"stage failed: {script_path.name}, exit={rc}")

            if stop_after is not None and script_name == stop_after:
                print(f"[STOP after] {script_name}")
                break

        if not dry_run:
            copied = snapshot_outputs(dataset_dir, run_dir, t0)
            print(f"[SNAPSHOT] copied {len(copied)} files to {run_dir / 'snapshots'}")

        failed_stage = ""

    except Exception as e:
        status = "failed"
        error_message = str(e)
        print(f"[FAILED] dataset={dataset}, variant={variant}, stage={failed_stage}: {e}")

    elapsed = time.time() - t0

    # 关键修复：只从本次 run_dir/snapshots 读指标，避免读取旧结果。
    summary_path = find_snapshot_summary(run_dir)
    metrics = extract_metrics_from_summary(summary_path) if (status == "success" and not dry_run) else {}

    result = RunResult(
        dataset=dataset,
        variant=variant,
        dataset_dir=str(dataset_dir),
        run_dir=str(run_dir),
        status=status,
        failed_stage=failed_stage,
        elapsed_sec=elapsed,
        summary_path=str(summary_path) if summary_path else "",
        precision=metrics.get("precision"),
        recall=metrics.get("recall"),
        f1=metrics.get("f1"),
        candidate_coverage=metrics.get("candidate_coverage"),
        candidate_precision=metrics.get("candidate_precision"),
        final_predicted_errors=metrics.get("final_predicted_errors"),
        tp=metrics.get("tp"),
        fp=metrics.get("fp"),
        fn=metrics.get("fn"),
        error_message=error_message,
    )

    write_json(run_dir / "run_result.json", result.__dict__)
    return result


def write_master_summary(results: List[RunResult], output_root: Path):
    safe_mkdir(output_root)
    out_csv = output_root / "ablation_master_summary.csv"

    fields = [
        "dataset",
        "variant",
        "variant_display_name",
        "status",
        "failed_stage",
        "precision",
        "recall",
        "f1",
        "candidate_coverage",
        "candidate_precision",
        "final_predicted_errors",
        "tp",
        "fp",
        "fn",
        "elapsed_sec",
        "dataset_dir",
        "run_dir",
        "summary_path",
        "error_message",
    ]

    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            row = dict(r.__dict__)
            row["variant_display_name"] = VARIANT_DISPLAY_NAME.get(r.variant, r.variant)
            writer.writerow(row)

    out_md = output_root / "ablation_master_summary.md"
    with out_md.open("w", encoding="utf-8") as f:
        f.write("| Dataset | Variant | Precision | Recall | F1 | Cand.Cov | Cand.Prec | Status | Failed Stage |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---|---|\n")

        for r in results:
            def fmt(x):
                return "" if x is None else f"{float(x):.4f}"

            f.write(
                f"| {r.dataset} | {VARIANT_DISPLAY_NAME.get(r.variant, r.variant)} "
                f"| {fmt(r.precision)} | {fmt(r.recall)} | {fmt(r.f1)} "
                f"| {fmt(r.candidate_coverage)} | {fmt(r.candidate_precision)} "
                f"| {r.status} | {r.failed_stage} |\n"
            )

    print(f"\n[OK] master summary saved: {out_csv}")
    print(f"[OK] markdown summary saved: {out_md}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run error-detection ablation experiments.")

    parser.add_argument("--root", type=str, default=DEFAULT_ROOT)
    parser.add_argument("--output-root", type=str, default="ablation_runs_detection")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--variants", nargs="+", default=VALID_VARIANTS, choices=VALID_VARIANTS)
    parser.add_argument("--python-bin", type=str, default=sys.executable)

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--start-from", type=str, default=None, choices=PIPELINE)
    parser.add_argument("--stop-after", type=str, default=None, choices=PIPELINE)

    parser.add_argument(
        "--on-missing-stage",
        type=str,
        default="skip",
        choices=["skip", "fail"],
        help="某阶段脚本缺失时如何处理：skip=跳过；fail=失败。",
    )
    parser.add_argument("--preflight-only", action="store_true")

    parser.add_argument(
        "--parallel-datasets",
        type=int,
        default=1,
        help="数据集级并行数。建议 2 或 3。",
    )
    parser.add_argument(
        "--parallel-unit",
        type=str,
        default="dataset",
        choices=["dataset", "dataset_variant"],
        help="dataset=数据集间并行，同一数据集内部 variants 串行；dataset_variant=所有 dataset×variant 并行。",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    root = Path(args.root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    safe_mkdir(output_root)

    print(f"[INFO] root        = {root}")
    print(f"[INFO] output_root = {output_root}")
    print(f"[INFO] datasets    = {args.datasets}")
    print(f"[INFO] variants    = {args.variants}")
    print(f"[INFO] python      = {args.python_bin}")
    print(f"[INFO] dry_run     = {args.dry_run}")
    print(f"[INFO] resume      = {args.resume}")
    print(f"[INFO] on_missing_stage = {args.on_missing_stage}")
    print(f"[INFO] parallel_datasets = {args.parallel_datasets}")
    print(f"[INFO] parallel_unit     = {args.parallel_unit}")

    write_preflight_report(root, args.datasets, output_root)
    if args.preflight_only:
        print("[INFO] preflight-only enabled; stop here.")
        return

    results: List[RunResult] = []
    max_workers = max(1, int(args.parallel_datasets))

    def run_single(dataset_name: str, variant_name: str) -> RunResult:
        return run_one(
            root=root,
            dataset=dataset_name,
            variant=variant_name,
            output_root=output_root,
            python_bin=args.python_bin,
            dry_run=args.dry_run,
            resume=args.resume,
            start_from=args.start_from,
            stop_after=args.stop_after,
            on_missing_stage=args.on_missing_stage,
        )

    def run_dataset_sequential(dataset_name: str) -> List[RunResult]:
        local_results: List[RunResult] = []
        for variant_name in args.variants:
            local_results.append(run_single(dataset_name, variant_name))
        return local_results

    if max_workers <= 1:
        for dataset in args.datasets:
            for variant in args.variants:
                result = run_single(dataset, variant)
                results.append(result)
                write_master_summary(results, output_root)

    elif args.parallel_unit == "dataset":
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            fut_map = {ex.submit(run_dataset_sequential, ds): ds for ds in args.datasets}
            for fut in as_completed(fut_map):
                ds = fut_map[fut]
                try:
                    ds_results = fut.result()
                    results.extend(ds_results)
                    write_master_summary(results, output_root)
                except Exception as e:
                    print(f"[FAILED DATASET] {ds}: {e}")

    else:
        tasks = [(ds, var) for ds in args.datasets for var in args.variants]
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            fut_map = {ex.submit(run_single, ds, var): (ds, var) for ds, var in tasks}
            for fut in as_completed(fut_map):
                ds, var = fut_map[fut]
                try:
                    result = fut.result()
                    results.append(result)
                    write_master_summary(results, output_root)
                except Exception as e:
                    print(f"[FAILED TASK] dataset={ds}, variant={var}: {e}")

    print("\n===== ALL DONE =====")
    write_master_summary(results, output_root)


if __name__ == "__main__":
    main()

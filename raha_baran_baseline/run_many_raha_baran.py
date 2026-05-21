#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch runner for multiple datasets.

Prepare a JSON config like:
[
  {"name": "hospital", "dirty": "/path/hospital_dirty.csv", "clean": "/path/hospital_clean.csv"},
  {"name": "flights", "dirty": "/path/flights_dirty.csv", "clean": "/path/flights_clean.csv"}
]

Then run:
python run_many_raha_baran.py --config datasets.json --base_out ./baseline_outputs --labeling_budget 20
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--base_out", required=True)
    parser.add_argument("--labeling_budget", type=int, default=20)
    parser.add_argument("--baran_input", choices=["raha", "oracle"], default="raha")
    parser.add_argument("--error_detectors", default="")
    parser.add_argument("--clean_raha_cache", action="store_true")
    args = parser.parse_args()

    datasets = json.load(open(args.config, "r", encoding="utf-8"))
    script = Path(__file__).resolve().parent / "run_raha_baran_baseline.py"
    base_out = Path(args.base_out).resolve()
    base_out.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    for ds in datasets:
        out_dir = base_out / f"{ds['name']}_raha_baran_{args.baran_input}"
        cmd = [
            sys.executable,
            str(script),
            "--name", ds["name"],
            "--dirty", ds["dirty"],
            "--clean", ds["clean"],
            "--out_dir", str(out_dir),
            "--labeling_budget", str(args.labeling_budget),
            "--baran_input", args.baran_input,
        ]
        if args.error_detectors:
            cmd += ["--error_detectors", args.error_detectors]
        if args.clean_raha_cache:
            cmd += ["--clean_raha_cache"]
        print("\nRunning:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        all_summaries.append(json.load(open(out_dir / "summary.json", "r", encoding="utf-8")))

    json.dump(all_summaries, open(base_out / "all_summaries.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nBatch summaries written to: {base_out / 'all_summaries.json'}")


if __name__ == "__main__":
    main()

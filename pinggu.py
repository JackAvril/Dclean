import json
import csv
from typing import Dict, Tuple, Any


# =========================
# 0) 配置
# =========================
REPAIR_JSONL = "hospital_repairs_top3_qwen_norule.jsonl"
GROUND_TRUTH_CSV = "hospital_clean.csv"
K = 3   # Top-k


# =========================
# 1) Ground Truth
# =========================
def load_ground_truth(path: str) -> Dict[Tuple[int, str], str]:
    gt = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (int(row["tid"]), row["attribute"])
            gt[key] = (row.get("correct_val") or "").strip()
    return gt


# =========================
# 2) Top-1 / Top-k 命中
# =========================
def hit_top1(obj: Dict[str, Any], correct_val: str) -> bool:
    return obj.get("chosen") == correct_val


def hit_topk(obj: Dict[str, Any], correct_val: str, k: int) -> bool:
    for cand in obj.get("topk", [])[:k]:
        if isinstance(cand, dict) and cand.get("value") == correct_val:
            return True
    return False


# =========================
# 3) P / R / F1 评估
# =========================
def evaluate_prf_from_current(
    repair_jsonl: str,
    gt_map: Dict[Tuple[int, str], str],
    k: int = 3,
):
    tp1 = fp1 = fn1 = 0
    tpk = fpk = fnk = 0

    n_eval = 0
    n_gt_error = 0
    n_gt_correct = 0
    n_over_repair = 0
    n_missing_gt = 0

    with open(repair_jsonl, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)

            key = (obj["row_id"], obj["col"])
            if key not in gt_map:
                n_missing_gt += 1
                continue

            correct_val = gt_map[key]
            current_val = (obj.get("current") or "").strip()
            should_change = bool(obj.get("should_change", False))

            gt_error = current_val != correct_val

            n_eval += 1
            if gt_error:
                n_gt_error += 1
            else:
                n_gt_correct += 1

            ok1 = hit_top1(obj, correct_val)
            okk = hit_topk(obj, correct_val, k)

            # ---------- @1 ----------
            if gt_error:
                if should_change:
                    if ok1:
                        tp1 += 1
                    else:
                        fp1 += 1
                else:
                    fn1 += 1
            else:
                if should_change:
                    fp1 += 1
                    n_over_repair += 1

            # ---------- @k ----------
            if gt_error:
                if should_change:
                    if okk:
                        tpk += 1
                    else:
                        fpk += 1
                else:
                    fnk += 1
            else:
                if should_change:
                    fpk += 1

    def safe_div(a, b):
        return a / b if b else 0.0

    p1 = safe_div(tp1, tp1 + fp1)
    r1 = safe_div(tp1, tp1 + fn1)
    f1 = safe_div(2 * p1 * r1, p1 + r1)

    pk = safe_div(tpk, tpk + fpk)
    rk = safe_div(tpk, tpk + fnk)
    fk = safe_div(2 * pk * rk, pk + rk)

    return {
        "evaluated_cells": n_eval,
        "gt_error_cells": n_gt_error,
        "gt_correct_cells": n_gt_correct,
        "missing_gt_cells": n_missing_gt,

        "P@1": p1, "R@1": r1, "F1@1": f1,
        f"P@{k}": pk, f"R@{k}": rk, f"F1@{k}": fk,

        "TP@1": tp1, "FP@1": fp1, "FN@1": fn1,
        f"TP@{k}": tpk, f"FP@{k}": fpk, f"FN@{k}": fnk,

        "over_repair_rate": safe_div(n_over_repair, n_gt_correct),
    }


# =========================
# 4) main
# =========================
if __name__ == "__main__":
    gt = load_ground_truth(GROUND_TRUTH_CSV)
    res = evaluate_prf_from_current(REPAIR_JSONL, gt, k=K)

    print("====== Repair Evaluation (from current) ======")
    print(f"Evaluated cells        : {res['evaluated_cells']}")
    print(f"GT error cells         : {res['gt_error_cells']}")
    print(f"GT correct cells       : {res['gt_correct_cells']}")
    print(f"Missing GT cells       : {res['missing_gt_cells']}")
    print("")
    print(f"P@1  {res['P@1']:.4f}  R@1  {res['R@1']:.4f}  F1@1  {res['F1@1']:.4f}")
    print(f"P@{K}  {res[f'P@{K}']:.4f}  R@{K}  {res[f'R@{K}']:.4f}  F1@{K}  {res[f'F1@{K}']:.4f}")
    print("")
    print(f"Over-repair rate       : {res['over_repair_rate']:.4f}")

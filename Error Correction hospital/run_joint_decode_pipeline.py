
import subprocess
from pathlib import Path


# ============================================================
# 1. 配置区：直接改这里，不用命令行传参
# ============================================================
CANDIDATE_FEATURES = "repair_candidate_features.csv"
PAIRWISE_RESPONSES = "repair_pairwise_responses.jsonl"
MODEL_DIR = "pairwise_ltr_student_output_joint_decode"

# 默认继续调用你当前较稳的训练脚本 + 新的联合解码推理脚本
TRAIN_SCRIPT = "train_ltr_iter_f1_step3.py"
INFER_SCRIPT = "infer_ltr_joint_decode_config.py"

EPOCHS = 20
BATCH_SIZE = 256
LR = 1e-3
GLOBAL_MIN_MARGIN = 0.05
JOINT_TOPK = 3
MAX_HARD_CASES = 300


def run_cmd(cmd):
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    script_dir = Path(__file__).resolve().parent
    train_script = script_dir / TRAIN_SCRIPT
    infer_script = script_dir / INFER_SCRIPT

    run_cmd([
        "python", str(train_script),
        "--candidate_features", CANDIDATE_FEATURES,
        "--pairwise_responses", PAIRWISE_RESPONSES,
        "--output_dir", MODEL_DIR,
        "--epochs", str(EPOCHS),
        "--batch_size", str(BATCH_SIZE),
        "--lr", str(LR),
    ])

    ckpt = Path(MODEL_DIR) / "best_model.pt"
    run_cmd([
        "python", str(infer_script),
        "--candidate_features", CANDIDATE_FEATURES,
        "--model_dir", MODEL_DIR,
        "--checkpoint", str(ckpt),
        "--global_min_margin", str(GLOBAL_MIN_MARGIN),
        "--joint_topk", str(JOINT_TOPK),
        "--max_hard_cases", str(MAX_HARD_CASES),
    ])

    print("[OK] Joint-decoding pipeline finished.")
    print("[OK] Final repairs:", Path(MODEL_DIR) / "inference_top1_repairs.csv")


if __name__ == "__main__":
    main()

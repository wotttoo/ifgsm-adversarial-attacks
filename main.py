"""
main.py
─────────────────────────────────────────────────────────────
Pipeline chính — chạy toàn bộ project I-FGSM theo thứ tự:
  1. Train model
  2. Exp1: Accuracy vs Epsilon
  3. Exp2: Accuracy vs Num Steps
  4. Exp3: Visualize adversarial examples

Cách dùng:
    python main.py                    # chạy toàn bộ
    python main.py --skip-train       # bỏ qua bước train
    python main.py --exp 1 2          # chỉ chạy exp 1 và 2
"""

import argparse
import os
import sys
import time
import torch
import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="I-FGSM Project — Full Pipeline"
    )
    parser.add_argument("--config",     type=str, default="configs/config.yaml")
    parser.add_argument("--skip-train", action="store_true",
                        help="Bỏ qua bước train (cần checkpoint có sẵn)")
    parser.add_argument("--exp",        nargs="*", type=int,
                        help="Chỉ chạy experiment cụ thể: --exp 1 3")
    return parser.parse_args()


def print_header(title: str) -> None:
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")


def main():
    args = parse_args()
    run_all = args.exp is None

    print_header("I-FGSM Adversarial Attack Project")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ds = cfg["dataset"]["name"]
    print(f"  Dataset  : {ds}")
    print(f"  Model    : {cfg['model']['name']}")
    print(f"  Epsilon  : {cfg['attack']['epsilon']}")
    print(f"  Steps    : {cfg['attack']['num_steps']}")

    # ── 0. Tạo thư mục output ─────────────────────────────────
    for d in ["results/figures", "results/logs", "results/checkpoints"]:
        os.makedirs(d, exist_ok=True)

    # ── 1. Train ──────────────────────────────────────────────
    if not args.skip_train:
        print_header("Bước 1: Huấn luyện Model")
        t0 = time.time()
        from train import main as train_main
        sys.argv = ["train.py", "--config", args.config]
        train_main()
        print(f"  ✓ Train xong ({time.time()-t0:.1f}s)")
    else:
        print("\n[Bỏ qua bước train]")

    # ── 2. Experiments ────────────────────────────────────────
    from experiments.exp1_epsilon   import run as run_exp1
    from experiments.exp2_steps     import run as run_exp2
    from experiments.exp3_visualize import run as run_exp3

    exp_map = {
        1: ("Exp1: Accuracy vs Epsilon",          run_exp1),
        2: ("Exp2: Accuracy vs Num Steps",         run_exp2),
        3: ("Exp3: Visualize Adversarial Examples",run_exp3),
    }

    exps_to_run = args.exp if args.exp else [1, 2, 3]

    for exp_id in exps_to_run:
        if exp_id not in exp_map:
            print(f"  [WARNING] Exp {exp_id} không tồn tại, bỏ qua.")
            continue
        title, run_fn = exp_map[exp_id]
        print_header(f"Bước {exp_id+1}: {title}")
        t0 = time.time()
        run_fn(config_path=args.config)
        print(f"  ✓ Xong ({time.time()-t0:.1f}s)")

    # ── 3. Tổng kết ───────────────────────────────────────────
    print_header("Tổng kết")
    print("  Kết quả đã lưu tại:")
    print("  ├── results/figures/   ← Biểu đồ & ảnh minh họa")
    print("  ├── results/logs/      ← Số liệu JSON")
    print("  └── results/checkpoints/ ← Model checkpoint")
    print("\n  File quan trọng:")
    for f in sorted(os.listdir("results/figures")):
        print(f"    📊 results/figures/{f}")
    print()


if __name__ == "__main__":
    main()

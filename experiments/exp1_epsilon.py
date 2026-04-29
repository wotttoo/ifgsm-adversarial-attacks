"""
experiments/exp1_epsilon.py
─────────────────────────────────────────────────────────────
Thí nghiệm 1: Khảo sát ảnh hưởng của Epsilon lên Accuracy

Kết quả kỳ vọng:
  - Accuracy giảm khi ε tăng
  - I-FGSM luôn mạnh hơn FGSM 1 bước
  - Với ε đủ lớn, accuracy về ~10% (random guess cho 10 classes)
"""

import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import yaml
import json

from models            import SimpleCNN
from utils.data_loader import get_dataloaders, get_in_channels, get_input_size
from utils.evaluator   import AdversarialEvaluator
from utils.visualization import plot_accuracy_vs_epsilon


def run(config_path: str = "../configs/config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        cfg["experiment"]["device"] if torch.cuda.is_available() else "cpu"
    )
    print(f"[Exp1] Device: {device}")

    # ── Load model ────────────────────────────────────────────
    ds_name    = cfg["dataset"]["name"]
    in_ch      = get_in_channels(ds_name)
    input_size = get_input_size(ds_name)

    model = SimpleCNN(in_channels=in_ch, num_classes=10, input_size=input_size)

    ckpt_path = os.path.join(
        cfg["train"]["save_dir"],
        f"{ds_name.lower()}_best.pth"
    )
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"  Loaded checkpoint: {ckpt_path}")
    else:
        print(f"  [WARNING] Không tìm thấy checkpoint: {ckpt_path}")
        print(f"  Chạy train.py trước!")
        return

    model = model.to(device)

    # ── Load test data ────────────────────────────────────────
    _, _, test_loader = get_dataloaders(
        ds_name,
        root       = cfg["dataset"]["root"],
        batch_size = cfg["dataset"]["batch_size"],
    )

    # ── Đánh giá ─────────────────────────────────────────────
    evaluator = AdversarialEvaluator(model, device=device)

    eps_list = cfg["experiment"]["epsilon_list"]
    results  = evaluator.evaluate_epsilon_range(
        loader       = test_loader,
        epsilon_list = eps_list,
        num_steps    = cfg["attack"]["num_steps"],
        max_batches  = 20,  # giới hạn để chạy nhanh
    )

    # ── Lưu kết quả ──────────────────────────────────────────
    os.makedirs(os.path.join(ROOT, "results", "logs"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "logs", f"exp1_epsilon_{ds_name.lower()}.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ── Vẽ biểu đồ ───────────────────────────────────────────
    plot_accuracy_vs_epsilon(
        results,
        save_path=os.path.join(ROOT, "results", "figures", f"exp1_acc_vs_epsilon_{ds_name.lower()}.png")
    )

    print("\n[Exp1] Hoàn tất!")


if __name__ == "__main__":
    run()

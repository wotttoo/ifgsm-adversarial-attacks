"""
experiments/exp2_steps.py
─────────────────────────────────────────────────────────────
Thí nghiệm 2: Khảo sát ảnh hưởng của số bước lặp T

Câu hỏi nghiên cứu:
  - Bao nhiêu bước thì I-FGSM "hội tụ"?
  - Thêm bước có luôn làm tăng attack strength?

Kết quả kỳ vọng:
  - Accuracy giảm nhanh từ T=1→10, chậm dần sau T=20
  - Sau một ngưỡng nào đó, thêm bước không có tác dụng nhiều
"""

import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import yaml
import json

from models              import SimpleCNN
from utils.data_loader   import get_dataloaders, get_in_channels, get_input_size
from utils.evaluator     import AdversarialEvaluator
from utils.visualization import plot_accuracy_vs_steps


def run(config_path: str = "../configs/config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        cfg["experiment"]["device"] if torch.cuda.is_available() else "cpu"
    )
    print(f"[Exp2] Device: {device}")

    ds_name    = cfg["dataset"]["name"]
    in_ch      = get_in_channels(ds_name)
    input_size = get_input_size(ds_name)

    model = SimpleCNN(in_channels=in_ch, num_classes=10, input_size=input_size)

    ckpt_path = os.path.join(
        cfg["train"]["save_dir"],
        f"{ds_name.lower()}_best.pth"
    )
    if not os.path.exists(ckpt_path):
        print(f"  [WARNING] Không tìm thấy checkpoint. Chạy train.py trước!"); return

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)

    _, _, test_loader = get_dataloaders(
        ds_name,
        root       = cfg["dataset"]["root"],
        batch_size = cfg["dataset"]["batch_size"],
    )

    evaluator = AdversarialEvaluator(model, device=device)
    epsilon   = cfg["attack"]["epsilon"]
    steps_list = cfg["experiment"]["steps_list"]

    results = evaluator.evaluate_steps(
        loader      = test_loader,
        epsilon     = epsilon,
        steps_list  = steps_list,
        max_batches = 20,
    )

    os.makedirs(os.path.join(ROOT, "results", "logs"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "logs", f"exp2_steps_{ds_name.lower()}.json"), "w") as f:
        json.dump(results, f, indent=2)

    plot_accuracy_vs_steps(
        results, epsilon=epsilon,
        save_path=os.path.join(ROOT, "results", "figures", f"exp2_acc_vs_steps_{ds_name.lower()}.png")
    )

    print("\n[Exp2] Hoàn tất!")


if __name__ == "__main__":
    run()

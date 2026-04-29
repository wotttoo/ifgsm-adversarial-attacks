"""
experiments/exp3_visualize.py
─────────────────────────────────────────────────────────────
Thí nghiệm 3: Trực quan hóa adversarial examples

Output:
  - Lưới ảnh: gốc | nhiễu×10 | đối kháng (cột = mẫu, hàng = loại)
  - Tiêu đề mỗi ảnh: nhãn thực / dự đoán (đỏ = sai, xanh = đúng)
  - Biểu đồ loss tăng dần qua từng bước I-FGSM
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import yaml

from models              import SimpleCNN
from attacks.ifgsm       import IFGSMAttack
from utils.data_loader   import get_dataloaders, get_in_channels, get_input_size
from utils.visualization import plot_adversarial_examples, plot_loss_evolution

MNIST_CLASSES  = [str(i) for i in range(10)]
CIFAR10_CLASSES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]


def run(config_path: str = "../configs/config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        cfg["experiment"]["device"] if torch.cuda.is_available() else "cpu"
    )
    print(f"[Exp3] Device: {device}")

    ds_name    = cfg["dataset"]["name"]
    in_ch      = get_in_channels(ds_name)
    input_size = get_input_size(ds_name)
    class_names = MNIST_CLASSES if ds_name.upper() == "MNIST" else CIFAR10_CLASSES

    model = SimpleCNN(in_channels=in_ch, num_classes=10, input_size=input_size)

    ckpt_path = os.path.join(cfg["train"]["save_dir"], f"{ds_name.lower()}_best.pth")
    if not os.path.exists(ckpt_path):
        print("  [WARNING] Không tìm thấy checkpoint. Chạy train.py trước!"); return

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    _, _, test_loader = get_dataloaders(
        ds_name, root=cfg["dataset"]["root"],
        batch_size=cfg["vis"]["num_examples"],   # lấy đúng số ảnh cần hiển thị
    )

    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)

    epsilon   = cfg["attack"]["epsilon"]
    num_steps = cfg["attack"]["num_steps"]

    attacker  = IFGSMAttack(model, epsilon=epsilon, num_steps=num_steps)
    adv_images = attacker(images, labels)

    # Lấy nhãn dự đoán
    with torch.no_grad():
        orig_preds = model(images).argmax(1).tolist()
        adv_preds  = model(adv_images).argmax(1).tolist()

    n_correct_clean = sum(p == l for p, l in zip(orig_preds, labels.tolist()))
    n_correct_adv   = sum(p == l for p, l in zip(adv_preds,  labels.tolist()))
    n              = len(labels)
    print(f"  Clean accuracy on batch: {100*n_correct_clean/n:.1f}%")
    print(f"  Adv   accuracy on batch: {100*n_correct_adv/n:.1f}%")

    # ── Vẽ lưới ảnh ───────────────────────────────────────────
    os.makedirs("../results/figures", exist_ok=True)
    plot_adversarial_examples(
        original    = images[:cfg["vis"]["num_examples"]].cpu(),
        adversarial = adv_images[:cfg["vis"]["num_examples"]].cpu(),
        orig_labels = orig_preds[:cfg["vis"]["num_examples"]],
        adv_labels  = adv_preds[:cfg["vis"]["num_examples"]],
        epsilon     = epsilon,
        num_steps   = num_steps,
        class_names = class_names,
        save_path   = f"../results/figures/exp3_examples_{ds_name.lower()}.png",
    )

    # ── Vẽ loss evolution ──────────────────────────────────────
    plot_loss_evolution(
        loss_history = attacker.last_stats["loss_history"],
        epsilon      = epsilon,
        save_path    = f"../results/figures/exp3_loss_evolution_{ds_name.lower()}.png",
    )

    print("\n[Exp3] Hoàn tất!")


if __name__ == "__main__":
    run()

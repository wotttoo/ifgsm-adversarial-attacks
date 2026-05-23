"""
experiments/exp_fgsm_epsilon_grid.py
─────────────────────────────────────────────────────────────
So sánh ảnh đối kháng FGSM khi thay đổi ε.

Grid: hàng = mẫu ảnh  |  cột = Original | ε=0.05 | ε=0.10 | ε=0.15 | ε=0.20 | ε=0.25 | ε=0.30
Mỗi ô hiển thị: ảnh đối kháng + nhãn dự đoán (✓/✗) + confidence %
                + nhiễu khuếch đại ×10 (hàng nhỏ bên dưới)

Output:
    results/figures/fgsm_epsilon_grid_mnist.png
    results/figures/fgsm_epsilon_grid_cifar10.png

Cách chạy:
    python experiments/exp_fgsm_epsilon_grid.py              # MNIST + CIFAR-10
    python experiments/exp_fgsm_epsilon_grid.py --dataset MNIST
    python experiments/exp_fgsm_epsilon_grid.py --dataset CIFAR10
"""

import sys, os, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import yaml

from models            import SimpleCNN
from utils.data_loader import get_dataloaders, get_in_channels, get_input_size
from utils.visualization import plot_fgsm_epsilon_grid

MNIST_CLASSES   = [str(i) for i in range(10)]
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def run_dataset(cfg: dict, ds_name: str, n_samples: int = 5) -> None:
    device = torch.device(
        cfg["experiment"]["device"] if torch.cuda.is_available() else "cpu"
    )

    in_ch      = get_in_channels(ds_name)
    input_size = get_input_size(ds_name)
    class_names = MNIST_CLASSES if ds_name.upper() == "MNIST" else CIFAR10_CLASSES

    print(f"\n[FGSM-εgrid] Dataset: {ds_name} | Device: {device}")

    # ── Load model ────────────────────────────────────────────
    ckpt_tag  = ds_name.lower()
    ckpt_path = os.path.join(ROOT, cfg["train"]["save_dir"], f"{ckpt_tag}_best.pth")
    if not os.path.exists(ckpt_path):
        print(f"  [ERROR] Không tìm thấy checkpoint: {ckpt_path}")
        print( "  Hãy train model trước: python train.py --dataset", ds_name)
        return

    net = SimpleCNN(in_channels=in_ch, num_classes=10, input_size=input_size)
    ckpt = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(ckpt["model_state"])
    net = net.to(device).eval()
    print(f"  Loaded: {ckpt_path}")

    # ── Lấy mẫu đúng từ test set ─────────────────────────────
    _, _, test_loader = get_dataloaders(
        ds_name,
        root       = cfg["dataset"]["root"],
        batch_size = n_samples * 8,
    )

    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        preds = net(images).argmax(1)
        mask  = preds == labels

    correct_images = images[mask]
    correct_labels = labels[mask].tolist()
    n_show = min(n_samples, len(correct_labels))
    print(f"  Lọc được {len(correct_labels)} mẫu đúng, hiển thị {n_show}")

    if n_show == 0:
        print("  [WARNING] Không có mẫu đúng, bỏ qua.")
        return

    # ── Tạo grid ──────────────────────────────────────────────
    epsilon_list = cfg["experiment"]["epsilon_list"]
    fig_dir      = os.path.join(ROOT, "results", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    save_path = os.path.join(fig_dir, f"fgsm_epsilon_grid_{ds_name.lower()}.png")

    print(f"  Đang tạo FGSM epsilon grid (ε = {epsilon_list}) ...")
    plot_fgsm_epsilon_grid(
        model        = net,
        images       = correct_images,
        labels       = correct_labels,
        epsilon_list = epsilon_list,
        dataset_name = ds_name,
        class_names  = class_names,
        n_samples    = n_show,
        save_path    = save_path,
    )
    print(f"  → {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="both",
        choices=["MNIST", "CIFAR10", "both"],
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--n_samples", type=int, default=5)
    args = parser.parse_args()

    config_path = os.path.join(ROOT, args.config)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    datasets = ["MNIST", "CIFAR10"] if args.dataset == "both" else [args.dataset]
    for ds in datasets:
        run_dataset(cfg, ds, n_samples=args.n_samples)

    print("\nHoàn tất!")


if __name__ == "__main__":
    main()

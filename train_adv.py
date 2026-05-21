"""
train_adv.py
─────────────────────────────────────────────────────────────
Script huấn luyện đối kháng (Adversarial Training với FGSM).

Checkpoint lưu tại: results/checkpoints/{dataset}_adv_best.pth

Cách dùng:
    python train_adv.py                          # MNIST, ε=0.3
    python train_adv.py --dataset CIFAR10        # CIFAR-10, ε=0.1
    python train_adv.py --dataset MNIST --epsilon 0.2 --adv-ratio 0.5
    python train_adv.py --dataset CIFAR10 --epochs 30
"""

import argparse
import os
import torch
import torch.optim as optim
import yaml

from models            import SimpleCNN
from utils.data_loader import get_dataloaders, get_in_channels, get_input_size
from utils.adv_trainer import AdvTrainer
from utils.visualization import plot_adv_training_history


# Epsilon mặc định hợp lý cho từng dataset
DEFAULT_EPSILON = {"MNIST": 0.3, "CIFAR10": 0.1}


def parse_args():
    p = argparse.ArgumentParser(description="Adversarial Training với FGSM")
    p.add_argument("--config",     type=str,   default="configs/config.yaml")
    p.add_argument("--dataset",    type=str,   default="MNIST",
                   choices=["MNIST", "CIFAR10"])
    p.add_argument("--epsilon",    type=float, default=None,
                   help="ε cho FGSM khi train (mặc định: 0.3 MNIST / 0.1 CIFAR10)")
    p.add_argument("--adv-ratio",  type=float, default=0.5,
                   help="Tỉ lệ loss adversarial (0–1, mặc định 0.5)")
    p.add_argument("--epochs",     type=int,   default=None)
    p.add_argument("--lr",         type=float, default=None)
    p.add_argument("--batch",      type=int,   default=None)
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ds_name   = args.dataset
    epsilon   = args.epsilon or DEFAULT_EPSILON[ds_name]
    adv_ratio = args.adv_ratio
    epochs    = args.epochs or cfg["train"]["epochs"]
    lr        = args.lr    or cfg["train"]["lr"]
    batch     = args.batch or cfg["dataset"]["batch_size"]

    # ── Device ────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    torch.manual_seed(cfg["experiment"]["seed"])
    print(f"[AdvTrain] Dataset={ds_name} | ε={epsilon} | adv_ratio={adv_ratio} | Device={device}")

    # ── Dataset ───────────────────────────────────────────────
    train_loader, val_loader, test_loader = get_dataloaders(
        ds_name,
        root        = cfg["dataset"]["root"],
        batch_size  = batch,
        val_split   = cfg["dataset"]["val_split"],
        num_workers = cfg["dataset"]["num_workers"],
        seed        = cfg["experiment"]["seed"],
    )

    # ── Model ─────────────────────────────────────────────────
    in_ch      = get_in_channels(ds_name)
    input_size = get_input_size(ds_name)
    model      = SimpleCNN(in_channels=in_ch, num_classes=10, input_size=input_size)
    save_tag   = f"{ds_name.lower()}_adv"

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: SimpleCNN | Params: {n_params:,} | Checkpoint: {save_tag}_best.pth")

    # ── Optimizer & Scheduler ─────────────────────────────────
    optimizer = optim.Adam(
        model.parameters(), lr=lr,
        weight_decay=cfg["train"]["weight_decay"]
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size = cfg["train"]["step_size"],
        gamma     = cfg["train"]["gamma"],
    )

    # ── Adversarial Training ──────────────────────────────────
    save_dir = cfg["train"]["save_dir"]
    trainer  = AdvTrainer(
        model         = model,
        optimizer     = optimizer,
        scheduler     = scheduler,
        device        = device,
        save_dir      = save_dir,
        epsilon_train = epsilon,
        adv_ratio     = adv_ratio,
        dataset_name  = ds_name,
    )

    history = trainer.fit(
        train_loader = train_loader,
        val_loader   = val_loader,
        epochs       = epochs,
        model_name   = save_tag,
    )

    # ── Test set evaluation ───────────────────────────────────
    print("\n── Đánh giá trên Test set ──")
    trainer.load_checkpoint(f"{save_tag}_best.pth")
    test_metrics = trainer.validate(test_loader)
    print(f"  Clean acc : {test_metrics['clean_acc']:.2f}%")
    print(f"  Robust acc: {test_metrics['rob_acc']:.2f}%  (ε={epsilon})")

    # ── Vẽ training history ───────────────────────────────────
    os.makedirs("results/figures", exist_ok=True)
    plot_adv_training_history(
        history,
        epsilon_train = epsilon,
        dataset_name  = ds_name,
        save_path     = f"results/figures/adv_training_history_{ds_name.lower()}.png",
    )
    print(f"\n✓ Checkpoint: {save_dir}/{save_tag}_best.pth")


if __name__ == "__main__":
    main()

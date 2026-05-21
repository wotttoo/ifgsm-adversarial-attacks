"""
utils/adv_trainer.py
─────────────────────────────────────────────────────────────
AdvTrainer — huấn luyện đối kháng với FGSM augmentation.

Chiến lược: mỗi batch tính loss hỗn hợp:
    loss = (1 - adv_ratio) * CE(model(x_clean), y)
         +      adv_ratio  * CE(model(x_adv),   y)

trong đó x_adv được sinh bằng FGSM với epsilon_train cố định.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List

from attacks.fgsm import fgsm_attack
from utils.data_loader import get_clip_values


class AdvTrainer:
    """
    Adversarial Trainer với FGSM augmentation.

    Args:
        model         : nn.Module
        optimizer     : Optimizer
        scheduler     : LR scheduler (tùy chọn)
        device        : torch.device
        save_dir      : thư mục lưu checkpoint
        epsilon_train : ε dùng cho FGSM khi train (L∞ budget)
        adv_ratio     : tỉ lệ loss adversarial trong tổng loss (0–1)
        dataset_name  : tên dataset để lấy clip bounds đúng
    """

    def __init__(
        self,
        model         : nn.Module,
        optimizer,
        scheduler     = None,
        device        : torch.device = torch.device("cpu"),
        save_dir      : str          = "./results/checkpoints",
        epsilon_train : float        = 0.3,
        adv_ratio     : float        = 0.5,
        dataset_name  : str          = "MNIST",
    ):
        self.model         = model.to(device)
        self.optimizer     = optimizer
        self.scheduler     = scheduler
        self.device        = device
        self.save_dir      = save_dir
        self.epsilon_train = epsilon_train
        self.adv_ratio     = adv_ratio
        self.criterion     = nn.CrossEntropyLoss()
        self.clip_min, self.clip_max = get_clip_values(dataset_name)

        os.makedirs(save_dir, exist_ok=True)

        self.history: Dict[str, List[float]] = {
            "train_loss"      : [],
            "train_clean_acc" : [],
            "train_rob_acc"   : [],
            "val_loss"        : [],
            "val_clean_acc"   : [],
            "val_rob_acc"     : [],
        }
        self.best_val_acc = 0.0

    # ── Train 1 epoch ─────────────────────────────────────────
    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        clean_correct = rob_correct = total = 0

        pbar = tqdm(loader, desc="  AdvTrain", leave=False, unit="batch")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            # ── Sinh ảnh đối kháng FGSM ───────────────────────
            adv_images = fgsm_attack(
                self.model, images, labels,
                epsilon  = self.epsilon_train,
                clip_min = self.clip_min,
                clip_max = self.clip_max,
            )

            # ── Loss hỗn hợp: clean + adversarial ────────────
            self.optimizer.zero_grad()
            self.model.train()

            out_clean = self.model(images)
            out_adv   = self.model(adv_images)

            loss = (
                (1 - self.adv_ratio) * self.criterion(out_clean, labels)
                +    self.adv_ratio  * self.criterion(out_adv,   labels)
            )
            loss.backward()
            self.optimizer.step()

            # ── Tracking ──────────────────────────────────────
            total_loss    += loss.item() * images.size(0)
            clean_correct += (out_clean.argmax(1) == labels).sum().item()
            rob_correct   += (out_adv.argmax(1)   == labels).sum().item()
            total         += labels.size(0)

            pbar.set_postfix(
                loss=f"{loss.item():.3f}",
                rob=f"{100*rob_correct/total:.1f}%",
            )

        return {
            "loss"      : total_loss    / total,
            "clean_acc" : 100.0 * clean_correct / total,
            "rob_acc"   : 100.0 * rob_correct   / total,
        }

    # ── Validate ──────────────────────────────────────────────
    def validate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        clean_correct = rob_correct = total = 0

        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                out_clean = self.model(images)
                loss      = self.criterion(out_clean, labels)
                total_loss    += loss.item() * images.size(0)
                clean_correct += (out_clean.argmax(1) == labels).sum().item()
                total         += labels.size(0)

        # Robust accuracy trên val — dùng FGSM
        self.model.eval()
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            adv = fgsm_attack(
                self.model, images, labels,
                epsilon  = self.epsilon_train,
                clip_min = self.clip_min,
                clip_max = self.clip_max,
            )
            with torch.no_grad():
                out_adv = self.model(adv)
            rob_correct += (out_adv.argmax(1) == labels).sum().item()

        return {
            "loss"      : total_loss    / total,
            "clean_acc" : 100.0 * clean_correct / total,
            "rob_acc"   : 100.0 * rob_correct   / total,
        }

    # ── Full training loop ────────────────────────────────────
    def fit(
        self,
        train_loader : DataLoader,
        val_loader   : DataLoader,
        epochs       : int,
        model_name   : str = "model_adv",
    ) -> Dict[str, List[float]]:
        print(f"\n{'='*60}")
        print(f" Adversarial Training (FGSM-AT)")
        print(f" ε_train={self.epsilon_train} | adv_ratio={self.adv_ratio} | {epochs} epochs")
        print(f" Device: {self.device}")
        print(f"{'='*60}")

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            tr = self.train_epoch(train_loader)
            vl = self.validate(val_loader)

            if self.scheduler:
                self.scheduler.step()

            self.history["train_loss"]     .append(tr["loss"])
            self.history["train_clean_acc"].append(tr["clean_acc"])
            self.history["train_rob_acc"]  .append(tr["rob_acc"])
            self.history["val_loss"]       .append(vl["loss"])
            self.history["val_clean_acc"]  .append(vl["clean_acc"])
            self.history["val_rob_acc"]    .append(vl["rob_acc"])

            elapsed = time.time() - t0
            print(
                f"Epoch [{epoch:3d}/{epochs}] "
                f"Train: loss={tr['loss']:.4f} clean={tr['clean_acc']:5.2f}% rob={tr['rob_acc']:5.2f}%  |  "
                f"Val: clean={vl['clean_acc']:5.2f}% rob={vl['rob_acc']:5.2f}%  "
                f"({elapsed:.1f}s)"
            )

            # Checkpoint theo val clean accuracy
            if vl["clean_acc"] > self.best_val_acc:
                self.best_val_acc = vl["clean_acc"]
                self.save_checkpoint(f"{model_name}_best.pth")
                print(f"  ✓ Saved best (val_clean={self.best_val_acc:.2f}%)")

        print(f"\nHoàn tất! Best val_clean_acc: {self.best_val_acc:.2f}%")
        return self.history

    # ── Checkpoint ────────────────────────────────────────────
    def save_checkpoint(self, filename: str) -> None:
        torch.save({
            "model_state"    : self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "history"        : self.history,
            "best_val_acc"   : self.best_val_acc,
            "epsilon_train"  : self.epsilon_train,
            "adv_ratio"      : self.adv_ratio,
        }, os.path.join(self.save_dir, filename))

    def load_checkpoint(self, filename: str) -> None:
        ckpt = torch.load(
            os.path.join(self.save_dir, filename), map_location=self.device
        )
        self.model.load_state_dict(ckpt["model_state"])
        self.history       = ckpt.get("history", self.history)
        self.best_val_acc  = ckpt.get("best_val_acc", 0.0)
        self.epsilon_train = ckpt.get("epsilon_train", self.epsilon_train)
        print(f"Loaded: {filename} (best_val_acc={self.best_val_acc:.2f}%)")

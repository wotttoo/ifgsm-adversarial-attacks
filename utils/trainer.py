"""
utils/trainer.py
─────────────────────────────────────────────────────────────
Trainer — huấn luyện và đánh giá model chuẩn.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Optional


class Trainer:
    """
    Trainer tiêu chuẩn hỗ trợ:
      - train / validate / evaluate
      - lưu & load checkpoint tốt nhất
      - ghi lịch sử loss / accuracy

    Args:
        model     : nn.Module
        optimizer : torch.optim.Optimizer
        scheduler : lr scheduler (tùy chọn)
        device    : torch.device
        save_dir  : thư mục lưu checkpoint
    """

    def __init__(
        self,
        model     : nn.Module,
        optimizer,
        scheduler = None,
        device    : torch.device = torch.device("cpu"),
        save_dir  : str          = "./results/checkpoints",
    ):
        self.model     = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device    = device
        self.save_dir  = save_dir
        self.criterion = nn.CrossEntropyLoss()

        os.makedirs(save_dir, exist_ok=True)

        # Lịch sử training
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "train_acc": [],
            "val_loss"  : [], "val_acc"  : [],
        }
        self.best_val_acc = 0.0

    # ── Train 1 epoch ─────────────────────────────────────────
    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(loader, desc="  Train", leave=False, unit="batch")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss    = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += labels.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return {
            "loss": total_loss / total,
            "acc" : 100.0 * correct / total,
        }

    # ── Validate 1 epoch ──────────────────────────────────────
    @torch.no_grad()
    def validate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            loss    = self.criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += labels.size(0)

        return {
            "loss": total_loss / total,
            "acc" : 100.0 * correct / total,
        }

    # ── Full training loop ────────────────────────────────────
    def fit(
        self,
        train_loader : DataLoader,
        val_loader   : DataLoader,
        epochs       : int,
        model_name   : str = "model",
    ) -> Dict[str, List[float]]:
        """
        Huấn luyện model qua nhiều epoch.

        Args:
            train_loader : DataLoader cho tập train
            val_loader   : DataLoader cho tập val
            epochs       : số epoch
            model_name   : tên file checkpoint

        Returns:
            history: dict chứa loss/acc qua từng epoch
        """
        print(f"\n{'='*55}")
        print(f" Bắt đầu training — {epochs} epochs | device: {self.device}")
        print(f"{'='*55}")

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            train_metrics = self.train_epoch(train_loader)
            val_metrics   = self.validate(val_loader)

            if self.scheduler is not None:
                self.scheduler.step()

            # Lưu lịch sử
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_acc"] .append(train_metrics["acc"])
            self.history["val_loss"]  .append(val_metrics["loss"])
            self.history["val_acc"]   .append(val_metrics["acc"])

            elapsed = time.time() - t0
            print(
                f"Epoch [{epoch:3d}/{epochs}] "
                f"Train: loss={train_metrics['loss']:.4f} acc={train_metrics['acc']:6.2f}%  |  "
                f"Val: loss={val_metrics['loss']:.4f} acc={val_metrics['acc']:6.2f}%  "
                f"({elapsed:.1f}s)"
            )

            # Lưu checkpoint tốt nhất
            if val_metrics["acc"] > self.best_val_acc:
                self.best_val_acc = val_metrics["acc"]
                self.save_checkpoint(f"{model_name}_best.pth")
                print(f"  ✓ Saved best checkpoint (val_acc={self.best_val_acc:.2f}%)")

        print(f"\nTraining hoàn tất! Best val_acc: {self.best_val_acc:.2f}%")
        return self.history

    # ── Evaluate trên test set ────────────────────────────────
    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """Đánh giá model trên test set, trả về loss + accuracy."""
        metrics = self.validate(loader)
        print(f"Test — loss: {metrics['loss']:.4f} | acc: {metrics['acc']:.2f}%")
        return metrics

    # ── Checkpoint ────────────────────────────────────────────
    def save_checkpoint(self, filename: str) -> None:
        path = os.path.join(self.save_dir, filename)
        torch.save({
            "model_state"    : self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "history"        : self.history,
            "best_val_acc"   : self.best_val_acc,
        }, path)

    def load_checkpoint(self, filename: str) -> None:
        path = os.path.join(self.save_dir, filename)
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.history      = ckpt.get("history", self.history)
        self.best_val_acc = ckpt.get("best_val_acc", 0.0)
        print(f"Loaded checkpoint: {path}")

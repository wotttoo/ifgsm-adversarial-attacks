"""
models/cnn.py
─────────────────────────────────────────────────────────────
SimpleCNN — mô hình CNN nhỏ gọn cho MNIST / CIFAR-10.

Architecture:
  MNIST  (1ch, 28x28):  Conv→Conv→Pool→Drop → FC→FC
  CIFAR  (3ch, 32x32):  Conv→Conv→Pool→Drop → Conv→Conv→Pool→Drop → FC→FC
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    CNN đơn giản, tự động điều chỉnh theo in_channels.

    Args:
        in_channels  : 1 (MNIST/grayscale) | 3 (CIFAR/RGB)
        num_classes  : số lớp đầu ra (mặc định 10)
        input_size   : kích thước ảnh đầu vào (mặc định 28 cho MNIST)
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10, input_size: int = 28):
        super().__init__()
        self.in_channels = in_channels
        self.input_size  = input_size

        # ── Block 1 ──────────────────────────────────────────
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # /2
            nn.Dropout2d(0.25),
        )

        # ── Block 2 (chỉ dùng cho CIFAR) ─────────────────────
        self.use_block2 = (in_channels == 3)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # /4
            nn.Dropout2d(0.25),
        ) if self.use_block2 else nn.Identity()

        # ── Tính kích thước flatten ────────────────────────────
        flat_size = self._get_flat_size()

        # ── Classifier ────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def _get_flat_size(self) -> int:
        """Tự động tính kích thước sau khi qua conv blocks."""
        with torch.no_grad():
            dummy = torch.zeros(1, self.in_channels, self.input_size, self.input_size)
            x = self.block1(dummy)
            if self.use_block2:
                x = self.block2(x)
            return x.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        if self.use_block2:
            x = self.block2(x)
        return self.classifier(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Trả về feature map trước lớp classifier (dùng cho visualize)."""
        x = self.block1(x)
        if self.use_block2:
            x = self.block2(x)
        return x


if __name__ == "__main__":
    # Kiểm tra nhanh
    model_mnist = SimpleCNN(in_channels=1, num_classes=10, input_size=28)
    model_cifar = SimpleCNN(in_channels=3, num_classes=10, input_size=32)

    x_mnist = torch.randn(4, 1, 28, 28)
    x_cifar = torch.randn(4, 3, 32, 32)

    print("MNIST output:", model_mnist(x_mnist).shape)   # [4, 10]
    print("CIFAR output:", model_cifar(x_cifar).shape)   # [4, 10]

    total_params = sum(p.numel() for p in model_mnist.parameters())
    print(f"MNIST params: {total_params:,}")

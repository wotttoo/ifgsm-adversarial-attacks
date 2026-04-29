"""
models/resnet.py
─────────────────────────────────────────────────────────────
Wrapper cho ResNet-18 từ torchvision, hỗ trợ MNIST & CIFAR-10.
"""

import torch.nn as nn
from torchvision import models


def get_resnet18(in_channels: int = 1, num_classes: int = 10) -> nn.Module:
    """
    Trả về ResNet-18 đã được điều chỉnh cho dataset nhỏ.

    - in_channels=1  : thay conv1 để nhận ảnh grayscale (MNIST)
    - in_channels=3  : giữ nguyên conv1 gốc (CIFAR-10 / ImageNet)
    - Thay fc cuối   : num_classes đầu ra

    Args:
        in_channels : số kênh đầu vào (1 hoặc 3)
        num_classes : số lớp phân loại

    Returns:
        model: nn.Module
    """
    model = models.resnet18(weights=None)

    # Thay conv đầu tiên nếu ảnh grayscale
    if in_channels != 3:
        model.conv1 = nn.Conv2d(
            in_channels, 64,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        # Bỏ maxpool đầu để không giảm kích thước quá nhiều với ảnh 28x28
        model.maxpool = nn.Identity()

    # Thay lớp fully-connected cuối
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model

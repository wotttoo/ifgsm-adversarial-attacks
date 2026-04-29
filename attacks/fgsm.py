"""
attacks/fgsm.py
─────────────────────────────────────────────────────────────
FGSM — Fast Gradient Sign Method (Goodfellow et al., 2015)
Dùng làm baseline để so sánh với I-FGSM.

Công thức:
    x_adv = x + ε · sign(∇ₓ J(θ, x, y))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def fgsm_attack(
    model     : nn.Module,
    images    : torch.Tensor,
    labels    : torch.Tensor,
    epsilon   : float = 0.3,
    clip_min  : float = 0.0,
    clip_max  : float = 1.0,
    targeted  : bool  = False,
) -> torch.Tensor:
    """
    Thực hiện FGSM attack (1 bước).

    Args:
        model    : mô hình phân lớp (ở eval mode)
        images   : ảnh gốc  [B, C, H, W], đã normalize về [0,1]
        labels   : nhãn thực hoặc nhãn mục tiêu (targeted)  [B]
        epsilon  : biên độ nhiễu tối đa (L∞ norm)
        clip_min : giá trị pixel nhỏ nhất sau attack
        clip_max : giá trị pixel lớn nhất sau attack
        targeted : True → giảm loss về nhãn mục tiêu (targeted attack)

    Returns:
        adv_images: ảnh đối kháng  [B, C, H, W]
    """
    images = images.clone().detach()
    images.requires_grad = True

    outputs = model(images)
    loss    = F.cross_entropy(outputs, labels)

    model.zero_grad()
    loss.backward()

    grad_sign = images.grad.data.sign()

    # Targeted: đi ngược chiều gradient (giảm loss)
    direction = -1 if targeted else 1
    adv_images = images + direction * epsilon * grad_sign
    adv_images = torch.clamp(adv_images, clip_min, clip_max)

    return adv_images.detach()

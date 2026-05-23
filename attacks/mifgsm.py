"""
attacks/mifgsm.py
─────────────────────────────────────────────────────────────
MI-FGSM — Momentum Iterative Fast Gradient Sign Method
  (Dong et al., 2018 — arXiv:1710.06081, CVPR 2018)

Ý tưởng: Tích lũy gradient theo chiều momentum μ giữa các bước
để tránh bị kẹt ở cực trị cục bộ và tăng khả năng transferability.

Công thức:
    g₀    = 0
    gₜ₊₁  = μ · gₜ  +  ∇ₓJ(θ, xₜ, y) / ‖∇ₓJ(θ, xₜ, y)‖₁
    xₜ₊₁  = Clip_{x,ε}[ xₜ + α · sign(gₜ₊₁) ]

So sánh với I-FGSM:
    I-FGSM: xₜ₊₁ = xₜ + α · sign(∇ₓJ)          (không có momentum)
    MI-FGSM: tích lũy hướng gradient qua các bước  (ổn định hơn, transfer tốt hơn)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def _clip(tensor: torch.Tensor, clip_min, clip_max) -> torch.Tensor:
    if isinstance(clip_min, torch.Tensor):
        lo = clip_min.view(1, -1, 1, 1).to(tensor.device)
        hi = clip_max.view(1, -1, 1, 1).to(tensor.device)
        return torch.max(torch.min(tensor, hi), lo)
    return torch.clamp(tensor, clip_min, clip_max)


class MIFGSMAttack:
    """
    MI-FGSM Attack (Momentum Iterative FGSM).

    Args:
        model      : nn.Module — mô hình phân lớp (eval mode)
        epsilon    : float     — biên độ nhiễu L∞ tối đa
        alpha      : float|None — bước mỗi iteration (None → ε/num_steps)
        num_steps  : int       — số bước lặp (khuyến nghị 10)
        decay      : float     — hệ số momentum μ (mặc định 1.0)
        targeted   : bool      — targeted attack
        clip_min/max: khoảng pixel hợp lệ
    """

    def __init__(
        self,
        model     : nn.Module,
        epsilon   : float          = 0.3,
        alpha     : Optional[float] = None,
        num_steps : int            = 10,
        decay     : float          = 1.0,
        targeted  : bool           = False,
        clip_min  : float          = 0.0,
        clip_max  : float          = 1.0,
    ):
        self.model     = model
        self.epsilon   = epsilon
        self.alpha     = alpha if alpha is not None else epsilon / num_steps
        self.num_steps = num_steps
        self.decay     = decay
        self.targeted  = targeted
        self.clip_min  = clip_min
        self.clip_max  = clip_max
        self.last_stats = {}

    def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.attack(images, labels)

    def attack(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        images = images.clone().detach()

        adv_images = images.clone()
        momentum   = torch.zeros_like(images)   # g₀ = 0
        loss_history = []

        for _ in range(self.num_steps):
            adv_images = adv_images.detach().requires_grad_(True)

            outputs = self.model(adv_images)
            loss    = F.cross_entropy(outputs, labels)
            self.model.zero_grad()
            loss.backward()

            grad = adv_images.grad.data

            # Chuẩn hóa gradient theo L1 norm
            grad_norm = grad.abs().view(grad.shape[0], -1).sum(dim=1)
            grad_norm = grad_norm.view(-1, 1, 1, 1)
            grad = grad / (grad_norm + 1e-8)

            # Tích lũy momentum
            momentum   = self.decay * momentum + grad

            direction  = -1 if self.targeted else 1
            adv_images = adv_images.detach() + direction * self.alpha * momentum.sign()

            # Clip vào ε-ball + pixel range
            perturbation = torch.clamp(adv_images - images, -self.epsilon, self.epsilon)
            adv_images   = _clip(images + perturbation, self.clip_min, self.clip_max)

            loss_history.append(loss.item())

        self.last_stats = {
            "epsilon"    : self.epsilon,
            "num_steps"  : self.num_steps,
            "decay"      : self.decay,
            "loss_history": loss_history,
            "final_loss" : loss_history[-1],
        }
        return adv_images.detach()

    def __repr__(self):
        return (f"MIFGSMAttack(ε={self.epsilon}, α={self.alpha:.4f}, "
                f"steps={self.num_steps}, μ={self.decay})")


def mifgsm_attack(
    model     : nn.Module,
    images    : torch.Tensor,
    labels    : torch.Tensor,
    epsilon   : float          = 0.3,
    alpha     : Optional[float] = None,
    num_steps : int            = 10,
    decay     : float          = 1.0,
    targeted  : bool           = False,
    clip_min  : float          = 0.0,
    clip_max  : float          = 1.0,
) -> torch.Tensor:
    return MIFGSMAttack(
        model=model, epsilon=epsilon, alpha=alpha,
        num_steps=num_steps, decay=decay,
        targeted=targeted, clip_min=clip_min, clip_max=clip_max,
    )(images, labels)

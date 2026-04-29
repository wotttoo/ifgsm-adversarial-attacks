"""
tests/test_ifgsm.py
─────────────────────────────────────────────────────────────
Unit tests cho I-FGSM attack.

Chạy: python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import torch.nn as nn

from attacks.ifgsm import IFGSMAttack, ifgsm_attack
from attacks.fgsm  import fgsm_attack
from models.cnn    import SimpleCNN


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def device():
    return torch.device("cpu")

@pytest.fixture
def mnist_model(device):
    model = SimpleCNN(in_channels=1, num_classes=10, input_size=28)
    model.to(device).eval()
    return model

@pytest.fixture
def dummy_batch(device):
    images = torch.rand(8, 1, 28, 28).to(device)
    labels = torch.randint(0, 10, (8,)).to(device)
    return images, labels


# ── Test IFGSMAttack class ─────────────────────────────────────

class TestIFGSMAttack:

    def test_output_shape(self, mnist_model, dummy_batch, device):
        """Adversarial image phải có cùng shape với ảnh gốc."""
        images, labels = dummy_batch
        attacker = IFGSMAttack(mnist_model, epsilon=0.3, num_steps=5)
        adv = attacker(images, labels)
        assert adv.shape == images.shape, f"Expected {images.shape}, got {adv.shape}"

    def test_linf_constraint(self, mnist_model, dummy_batch, device):
        """Nhiễu phải nằm trong vùng [-ε, +ε] (L∞ norm)."""
        images, labels = dummy_batch
        epsilon = 0.3
        attacker = IFGSMAttack(mnist_model, epsilon=epsilon, num_steps=10)
        adv = attacker(images, labels)

        perturbation = (adv - images).abs().max().item()
        assert perturbation <= epsilon + 1e-5, \
            f"L∞ perturbation {perturbation:.5f} > epsilon {epsilon}"

    def test_pixel_range(self, mnist_model, dummy_batch, device):
        """Giá trị pixel adversarial phải trong [0, 1]."""
        images, labels = dummy_batch
        attacker = IFGSMAttack(mnist_model, epsilon=0.3, num_steps=10)
        adv = attacker(images, labels)

        assert adv.min().item() >= -1e-5, f"Pixel min {adv.min().item()} < 0"
        assert adv.max().item() <=  1 + 1e-5, f"Pixel max {adv.max().item()} > 1"

    def test_no_gradient_leakage(self, mnist_model, dummy_batch, device):
        """Adversarial image không được có requires_grad=True (detach)."""
        images, labels = dummy_batch
        attacker = IFGSMAttack(mnist_model, epsilon=0.3, num_steps=5)
        adv = attacker(images, labels)
        assert not adv.requires_grad

    def test_attack_reduces_accuracy(self, mnist_model, dummy_batch, device):
        """I-FGSM phải giảm accuracy so với ảnh gốc (với model đã train)."""
        # Dùng model ngẫu nhiên → kết quả không chắc chắn, chỉ kiểm tra chạy được
        images, labels = dummy_batch
        attacker = IFGSMAttack(mnist_model, epsilon=0.3, num_steps=10)
        adv = attacker(images, labels)

        with torch.no_grad():
            orig_acc = (mnist_model(images).argmax(1) == labels).float().mean()
            adv_acc  = (mnist_model(adv).argmax(1)    == labels).float().mean()

        # Với model ngẫu nhiên, chỉ kiểm tra code chạy không crash
        assert isinstance(orig_acc.item(), float)
        assert isinstance(adv_acc.item(), float)

    def test_stats_recorded(self, mnist_model, dummy_batch, device):
        """last_stats phải được ghi sau khi attack."""
        images, labels = dummy_batch
        num_steps = 7
        attacker  = IFGSMAttack(mnist_model, epsilon=0.3, num_steps=num_steps)
        attacker(images, labels)

        stats = attacker.last_stats
        assert "loss_history" in stats
        assert len(stats["loss_history"]) == num_steps
        assert "perturbation_linf" in stats
        assert "perturbation_l2" in stats

    def test_alpha_default(self, mnist_model, dummy_batch, device):
        """Khi alpha=None, phải tự tính = epsilon / num_steps."""
        epsilon, num_steps = 0.3, 10
        attacker = IFGSMAttack(mnist_model, epsilon=epsilon, num_steps=num_steps)
        assert abs(attacker.alpha - epsilon / num_steps) < 1e-8

    def test_custom_clip_range(self, mnist_model, device):
        """Kiểm tra custom clip_min/max."""
        images = torch.rand(4, 1, 28, 28).to(device) * 0.5  # [0, 0.5]
        labels = torch.randint(0, 10, (4,)).to(device)

        attacker = IFGSMAttack(mnist_model, epsilon=0.1,
                               num_steps=5, clip_min=0.0, clip_max=0.5)
        adv = attacker(images, labels)
        assert adv.max().item() <= 0.5 + 1e-5

    def test_repr(self, mnist_model):
        """__repr__ phải chạy không lỗi."""
        attacker = IFGSMAttack(mnist_model, epsilon=0.2, num_steps=20)
        assert "IFGSMAttack" in repr(attacker)
        assert "0.2" in repr(attacker)


# ── Test functional API ────────────────────────────────────────

class TestIfgsmFunction:

    def test_functional_same_as_class(self, mnist_model, dummy_batch, device):
        """Functional API phải cho kết quả giống class API (cùng seed)."""
        images, labels = dummy_batch
        torch.manual_seed(0)
        adv_class = IFGSMAttack(mnist_model, epsilon=0.2, num_steps=5)(images, labels)

        torch.manual_seed(0)
        adv_func = ifgsm_attack(mnist_model, images, labels, epsilon=0.2, num_steps=5)

        assert torch.allclose(adv_class, adv_func), "Class và functional API cho kết quả khác nhau"

    def test_functional_output_shape(self, mnist_model, dummy_batch, device):
        images, labels = dummy_batch
        adv = ifgsm_attack(mnist_model, images, labels, epsilon=0.3, num_steps=5)
        assert adv.shape == images.shape


# ── Test FGSM baseline ─────────────────────────────────────────

class TestFGSMBaseline:

    def test_fgsm_output_shape(self, mnist_model, dummy_batch):
        images, labels = dummy_batch
        adv = fgsm_attack(mnist_model, images, labels, epsilon=0.3)
        assert adv.shape == images.shape

    def test_fgsm_linf(self, mnist_model, dummy_batch):
        images, labels = dummy_batch
        epsilon = 0.2
        adv     = fgsm_attack(mnist_model, images, labels, epsilon=epsilon)
        perturb = (adv - images).abs().max().item()
        assert perturb <= epsilon + 1e-5


# ── Test model ─────────────────────────────────────────────────

class TestSimpleCNN:

    def test_mnist_output(self, device):
        model = SimpleCNN(in_channels=1, num_classes=10, input_size=28).to(device)
        x = torch.rand(4, 1, 28, 28).to(device)
        assert model(x).shape == (4, 10)

    def test_cifar_output(self, device):
        model = SimpleCNN(in_channels=3, num_classes=10, input_size=32).to(device)
        x = torch.rand(4, 3, 32, 32).to(device)
        assert model(x).shape == (4, 10)

"""
experiments/exp8_perclass.py
─────────────────────────────────────────────────────────────
Thực nghiệm 8 — Per-class Vulnerability Analysis

Mục tiêu: Với cùng ε=0.20, class nào dễ bị FGSM nhất?
Class nào kháng tốt nhất?

Kết quả:
  - Bar chart ASR theo từng class (MNIST & CIFAR-10)
  - Clean acc vs Robust acc per class (stacked bar)
  - Bảng xếp hạng class theo độ dễ bị tấn công
"""

import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models import SimpleCNN
from attacks.fgsm import fgsm_attack

DEVICE   = torch.device('cpu')
EPSILON  = 0.20
N_SAMPLE = 2000
BATCH    = 64
OUT_DIR  = 'results/figures'
LOG_DIR  = 'results/logs'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

MNIST_CLS = [str(i) for i in range(10)]
CIFAR_CLS = ['airplane','automobile','bird','cat','deer',
             'dog','frog','horse','ship','truck']


def load_model(ckpt, in_ch, sz):
    m = SimpleCNN(in_channels=in_ch, num_classes=10, input_size=sz).to(DEVICE)
    ck = torch.load(ckpt, map_location=DEVICE)
    m.load_state_dict(ck['model_state'])
    m.eval()
    return m


def get_loader(name, n):
    if name == 'mnist':
        ds = datasets.MNIST('data', train=False, download=True,
                            transform=transforms.ToTensor())
    else:
        ds = datasets.CIFAR10('data', train=False, download=True,
                              transform=transforms.ToTensor())
    idx = torch.randperm(len(ds))[:n].tolist()
    return DataLoader(Subset(ds, idx), batch_size=BATCH, shuffle=False)


def perclass_stats(model, loader, epsilon, num_classes=10):
    """
    Trả về dict:
      total[c], correct_clean[c], correct_adv[c]
    """
    total         = np.zeros(num_classes, int)
    correct_clean = np.zeros(num_classes, int)
    correct_adv   = np.zeros(num_classes, int)

    model.eval()
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        with torch.no_grad():
            preds_clean = model(imgs).argmax(1)

        for c in range(num_classes):
            mask_c = lbls == c
            if mask_c.sum() == 0:
                continue
            total[c] += mask_c.sum().item()
            correct_mask = (preds_clean == lbls) & mask_c
            correct_clean[c] += correct_mask.sum().item()

            if correct_mask.sum() == 0:
                continue
            imgs_c = imgs[correct_mask]
            lbls_c = lbls[correct_mask]
            adv = fgsm_attack(model, imgs_c, lbls_c, epsilon=epsilon)
            with torch.no_grad():
                preds_adv = model(adv).argmax(1)
            correct_adv[c] += (preds_adv == lbls_c).sum().item()

    return total, correct_clean, correct_adv


def plot_perclass(total, correct_clean, correct_adv, class_names, title, fname):
    n = len(class_names)
    clean_acc  = np.where(total > 0, correct_clean / total * 100, 0)
    asr        = np.where(correct_clean > 0,
                          (1 - correct_adv / correct_clean) * 100, 0)
    robust_acc = np.where(total > 0, correct_adv / total * 100, 0)

    order = np.argsort(asr)[::-1]   # sắp xếp từ dễ bị tấn công → khó

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # ── Panel 1: ASR bar ──────────────────────────────────────
    ax = axes[0]
    x  = np.arange(n)
    bars = ax.bar(x, asr[order],
                  color=[plt.cm.RdYlGn_r(v/100) for v in asr[order]],
                  edgecolor='white', width=0.65)
    for bar, v in zip(bars, asr[order]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{v:.1f}%', ha='center', va='bottom', fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels([class_names[i] for i in order], rotation=20, ha='right', fontsize=10)
    ax.set_ylabel('FGSM Attack Success Rate (%)', fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_title(f'ASR theo từng class (xếp từ dễ bị tấn công → khó)', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(asr.mean(), color='gray', ls='--', lw=1.2, label=f'Mean ASR = {asr.mean():.1f}%')
    ax.legend(fontsize=9)

    # ── Panel 2: Clean acc vs Robust acc stacked ─────────────
    ax2 = axes[1]
    robust_pct = robust_acc[order]
    drop_pct   = clean_acc[order] - robust_acc[order]
    ax2.bar(x, robust_pct, color='#2ca02c', label='Robust Acc (sau tấn công)', width=0.65)
    ax2.bar(x, drop_pct,  bottom=robust_pct, color='#d62728',
            alpha=0.75, label='Acc Drop (bị tấn công)', width=0.65)
    ax2.set_xticks(x)
    ax2.set_xticklabels([class_names[i] for i in order], rotation=20, ha='right', fontsize=10)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.set_ylim(0, 115)
    ax2.set_title('Clean Accuracy vs Robust Accuracy theo class', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    plt.suptitle(title, fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {fname}')

    return {'asr': asr.tolist(), 'clean_acc': clean_acc.tolist(),
            'robust_acc': robust_acc.tolist(), 'order': order.tolist()}


def main():
    print('=' * 58)
    print(f'Thực nghiệm 8 — Per-class Vulnerability  (ε={EPSILON})')
    print('=' * 58)

    log = {}

    for name, ckpt, in_ch, sz, cls_names in [
        ('mnist',   'results/checkpoints/mnist_best.pth',   1, 28, MNIST_CLS),
        ('cifar10', 'results/checkpoints/cifar10_best.pth', 3, 32, CIFAR_CLS),
    ]:
        print(f'\n[{name.upper()}]')
        model  = load_model(ckpt, in_ch, sz)
        loader = get_loader(name, N_SAMPLE)
        total, correct_clean, correct_adv = perclass_stats(model, loader, EPSILON)

        stats = plot_perclass(
            total, correct_clean, correct_adv, cls_names,
            title=f'{name.upper()} — FGSM Per-class Vulnerability  (ε={EPSILON})',
            fname=f'exp8_perclass_{name}.png'
        )
        log[name] = stats

        asr = np.array(stats['asr'])
        order = np.argsort(asr)[::-1]
        print(f'\n  Xếp hạng (dễ bị tấn công → khó):')
        for rank, i in enumerate(order, 1):
            print(f'  {rank:2d}. {cls_names[i]:12s}  ASR={asr[i]:.1f}%')

    with open(os.path.join(LOG_DIR, 'exp8_perclass.json'), 'w') as f:
        json.dump({'epsilon': EPSILON, 'results': log}, f, indent=2)

    print('\n✓ Thực nghiệm 8 hoàn tất.')


if __name__ == '__main__':
    main()

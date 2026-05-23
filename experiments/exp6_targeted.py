"""
experiments/exp6_targeted.py
─────────────────────────────────────────────────────────────
Thực nghiệm 6 — Targeted FGSM

Mục tiêu: Đánh giá FGSM targeted attack — ép model dự đoán
sai sang đúng lớp target mà attacker muốn.

Kết quả:
  - Targeted Success Rate (TSR) theo từng lớp nguồn → lớp đích
  - Confusion matrix (source_class × target_class) của TSR
  - So sánh ASR: untargeted vs targeted tại cùng ε
"""

import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models import SimpleCNN
from attacks.fgsm import fgsm_attack

# ── Config ────────────────────────────────────────────────────
DEVICE      = torch.device('cpu')
EPSILON     = 0.20
N_SAMPLES   = 1000
BATCH       = 64
CKPT_MNIST  = 'results/checkpoints/mnist_best.pth'
CKPT_CIFAR  = 'results/checkpoints/cifar10_best.pth'
OUT_DIR     = 'results/figures'
LOG_DIR     = 'results/logs'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

MNIST_CLS  = [str(i) for i in range(10)]
CIFAR_CLS  = ['airplane','automobile','bird','cat','deer',
              'dog','frog','horse','ship','truck']


# ── Load model ────────────────────────────────────────────────
def load_model(ckpt, in_ch, sz):
    m = SimpleCNN(in_channels=in_ch, num_classes=10, input_size=sz).to(DEVICE)
    ck = torch.load(ckpt, map_location=DEVICE)
    m.load_state_dict(ck['model_state'])
    m.eval()
    return m


# ── Dataset ───────────────────────────────────────────────────
def get_loader(name, n):
    if name == 'mnist':
        ds = datasets.MNIST('data', train=False, download=True,
                            transform=transforms.ToTensor())
    else:
        ds = datasets.CIFAR10('data', train=False, download=True,
                              transform=transforms.ToTensor())
    idx = torch.randperm(len(ds))[:n].tolist()
    return DataLoader(Subset(ds, idx), batch_size=BATCH, shuffle=False)


# ── Targeted FGSM sweep ───────────────────────────────────────
def targeted_sweep(model, loader, num_classes, epsilon):
    """
    Trả về ma trận TSR [num_classes x num_classes].
    TSR[src][tgt] = tỉ lệ mẫu thuộc class src bị đánh thành class tgt.
    Đường chéo (src==tgt) = N/A (skip).
    """
    # Thu thập ảnh đúng theo từng class
    class_images  = {c: [] for c in range(num_classes)}
    class_labels  = {c: [] for c in range(num_classes)}

    model.eval()
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            preds = model(imgs).argmax(dim=1)
            for i in range(len(imgs)):
                if preds[i] == lbls[i]:
                    class_images[lbls[i].item()].append(imgs[i])
                    class_labels[lbls[i].item()].append(lbls[i].item())

    tsr_matrix = np.full((num_classes, num_classes), np.nan)

    for src in range(num_classes):
        if not class_images[src]:
            continue
        imgs_src = torch.stack(class_images[src])
        n_src    = len(imgs_src)

        for tgt in range(num_classes):
            if tgt == src:
                continue
            tgt_t = torch.full((n_src,), tgt, dtype=torch.long, device=DEVICE)
            adv   = fgsm_attack(model, imgs_src.to(DEVICE), tgt_t,
                                epsilon=epsilon, targeted=True)
            with torch.no_grad():
                preds_adv = model(adv).argmax(dim=1)
            tsr = (preds_adv == tgt_t).float().mean().item() * 100
            tsr_matrix[src][tgt] = tsr

    return tsr_matrix


# ── Untargeted vs Targeted ASR ────────────────────────────────
def compare_targeted_untargeted(model, loader, epsilon, num_classes):
    results = {'untargeted_asr': [], 'targeted_asr': []}
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        with torch.no_grad():
            preds = model(imgs).argmax(dim=1)
        correct_mask = preds == lbls
        if correct_mask.sum() == 0:
            continue
        imgs_c = imgs[correct_mask]
        lbls_c = lbls[correct_mask]

        # Untargeted
        adv_u = fgsm_attack(model, imgs_c, lbls_c, epsilon=epsilon, targeted=False)
        with torch.no_grad():
            preds_u = model(adv_u).argmax(dim=1)
        asr_u = (preds_u != lbls_c).float().mean().item() * 100
        results['untargeted_asr'].append(asr_u)

        # Targeted — random target != true label
        tgt_lbls = (lbls_c + torch.randint(1, num_classes, lbls_c.shape, device=DEVICE)) % num_classes
        adv_t = fgsm_attack(model, imgs_c, tgt_lbls, epsilon=epsilon, targeted=True)
        with torch.no_grad():
            preds_t = model(adv_t).argmax(dim=1)
        tsr = (preds_t == tgt_lbls).float().mean().item() * 100
        results['targeted_asr'].append(tsr)

    return {k: float(np.mean(v)) for k, v in results.items()}


# ── Plot: TSR heatmap ─────────────────────────────────────────
def plot_tsr_heatmap(tsr_matrix, class_names, title, fname):
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.isnan(tsr_matrix)
    sns.heatmap(tsr_matrix, mask=mask, annot=True, fmt='.1f',
                cmap='YlOrRd', vmin=0, vmax=100,
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, ax=ax, cbar_kws={'label': 'TSR (%)'})
    # Tô đường chéo (N/A)
    for i in range(len(class_names)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=True,
                                   color='#cccccc', lw=0))
        ax.text(i+0.5, i+0.5, 'N/A', ha='center', va='center',
                fontsize=8, color='#666666')
    ax.set_xlabel('Target Class (đích muốn đánh vào)', fontsize=11)
    ax.set_ylabel('Source Class (class thực của ảnh)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=14)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {fname}')


# ── Plot: Untargeted vs Targeted bar ─────────────────────────
def plot_comparison(results_dict, fname):
    datasets_  = list(results_dict.keys())
    x          = np.arange(len(datasets_))
    w          = 0.35
    u_vals     = [results_dict[d]['untargeted_asr'] for d in datasets_]
    t_vals     = [results_dict[d]['targeted_asr']   for d in datasets_]

    fig, ax = plt.subplots(figsize=(7, 5))
    b1 = ax.bar(x - w/2, u_vals, w, label='Untargeted ASR', color='#d62728', alpha=0.85)
    b2 = ax.bar(x + w/2, t_vals, w, label='Targeted TSR',   color='#1f77b4', alpha=0.85)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([d.upper() for d in datasets_], fontsize=12)
    ax.set_ylabel('Tỉ lệ tấn công thành công (%)', fontsize=11)
    ax.set_ylim(0, 110)
    ax.set_title(f'FGSM Untargeted ASR vs Targeted TSR  (ε={EPSILON})',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {fname}')


# ── Main ──────────────────────────────────────────────────────
def main():
    print('=' * 58)
    print('Thực nghiệm 6 — Targeted FGSM')
    print(f'ε={EPSILON}  |  N={N_SAMPLES}')
    print('=' * 58)

    comparison = {}

    for name, ckpt, in_ch, sz, cls_names in [
        ('mnist',   CKPT_MNIST, 1, 28, MNIST_CLS),
        ('cifar10', CKPT_CIFAR, 3, 32, CIFAR_CLS),
    ]:
        print(f'\n[{name.upper()}]')
        model  = load_model(ckpt, in_ch, sz)
        loader = get_loader(name, N_SAMPLES)

        # TSR matrix
        print('  Computing TSR matrix...')
        t0  = time.time()
        tsr = targeted_sweep(model, loader, num_classes=10, epsilon=EPSILON)
        print(f'  Done in {time.time()-t0:.1f}s')

        plot_tsr_heatmap(
            tsr, cls_names,
            title=f'{name.upper()} — Targeted FGSM TSR Matrix  (ε={EPSILON})\n'
                  f'Hàng = class nguồn  |  Cột = class đích  |  Giá trị = TSR (%)',
            fname=f'exp6_targeted_tsr_heatmap_{name}.png'
        )

        # Untargeted vs Targeted
        print('  Computing untargeted vs targeted comparison...')
        comp = compare_targeted_untargeted(model, loader, EPSILON, 10)
        comparison[name] = comp
        print(f'  Untargeted ASR: {comp["untargeted_asr"]:.1f}%')
        print(f'  Targeted  TSR:  {comp["targeted_asr"]:.1f}%')

        # Per-class avg TSR
        avg_tsr_as_src = np.nanmean(tsr, axis=1)   # trung bình theo hàng
        avg_tsr_as_tgt = np.nanmean(tsr, axis=0)   # trung bình theo cột
        print('\n  Avg TSR khi class là SOURCE (dễ bị tấn công khỏi):')
        for i, v in enumerate(avg_tsr_as_src):
            print(f'    {cls_names[i]:12s}: {v:.1f}%')
        print('\n  Avg TSR khi class là TARGET (dễ bị tấn công vào):')
        for i, v in enumerate(avg_tsr_as_tgt):
            print(f'    {cls_names[i]:12s}: {v:.1f}%')

    plot_comparison(comparison, 'exp6_targeted_vs_untargeted.png')

    # Save log
    log = {'epsilon': EPSILON, 'n_samples': N_SAMPLES, 'comparison': comparison}
    with open(os.path.join(LOG_DIR, 'exp6_targeted.json'), 'w') as f:
        json.dump(log, f, indent=2)

    print('\n✓ Thực nghiệm 6 hoàn tất.')


if __name__ == '__main__':
    main()

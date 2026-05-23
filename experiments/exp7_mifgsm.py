"""
experiments/exp7_mifgsm.py
─────────────────────────────────────────────────────────────
Thực nghiệm 7 — So sánh FGSM / I-FGSM / MI-FGSM

Mục tiêu:
  1. ASR theo ε: 3 attack trên MNIST và CIFAR-10
  2. Transferability: MI-FGSM vs FGSM (CIFAR-10, 3 model)
  3. Hiệu quả vs số bước T (I-FGSM vs MI-FGSM)
"""

import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models import SimpleCNN
from models.resnet   import get_resnet18
from models.mobilenet import get_mobilenetv2_cifar10
from attacks.fgsm   import fgsm_attack
from attacks.ifgsm  import ifgsm_attack
from attacks.mifgsm import mifgsm_attack

# ── Config ────────────────────────────────────────────────────
DEVICE    = torch.device('cpu')
EPSILONS  = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
STEPS_LIST = [1, 5, 10, 20, 40]
N_SAMPLES = 640
BATCH     = 64
OUT_DIR   = 'results/figures'
LOG_DIR   = 'results/logs'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

CKPTS = {
    'mnist':       ('results/checkpoints/mnist_best.pth',             SimpleCNN,               {'in_channels':1,'num_classes':10,'input_size':28}),
    'cifar10':     ('results/checkpoints/cifar10_best.pth',            SimpleCNN,               {'in_channels':3,'num_classes':10,'input_size':32}),
    'resnet18':    ('results/checkpoints/cifar10_resnet18_best.pth',   get_resnet18,            {'in_channels':3,'num_classes':10}),
    'mobilenetv2': ('results/checkpoints/cifar10_mobilenetv2_best.pth', get_mobilenetv2_cifar10, {}),
}


def load_model(key):
    path, fn, kwargs = CKPTS[key]
    if kwargs:
        m = fn(**kwargs).to(DEVICE)
    else:
        m = fn().to(DEVICE)
    ck = torch.load(path, map_location=DEVICE)
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


def eval_attack(model, loader, attack_fn, **kwargs):
    """Trả về (clean_acc, robust_acc, asr) theo %."""
    total = correct_clean = correct_adv = 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        with torch.no_grad():
            preds_clean = model(imgs).argmax(1)
        mask = preds_clean == lbls
        correct_clean += mask.sum().item()
        total += len(lbls)

        if mask.sum() == 0:
            continue
        imgs_c, lbls_c = imgs[mask], lbls[mask]
        adv = attack_fn(model, imgs_c, lbls_c, **kwargs)
        with torch.no_grad():
            preds_adv = model(adv).argmax(1)
        correct_adv += (preds_adv == lbls_c).sum().item()

    clean_acc  = correct_clean / total * 100
    n_correct  = correct_clean
    robust_acc = correct_adv / total * 100
    asr        = (1 - correct_adv / max(n_correct, 1)) * 100
    return clean_acc, robust_acc, asr


# ── Exp 7A: ASR vs epsilon ────────────────────────────────────
def exp7a_asr_vs_epsilon():
    print('\n[7A] ASR vs ε — MNIST & CIFAR-10')
    results = {}

    for ds_name, model_key in [('mnist','mnist'), ('cifar10','cifar10')]:
        model  = load_model(model_key)
        loader = get_loader(ds_name, N_SAMPLES)
        results[ds_name] = {
            'fgsm': [], 'ifgsm': [], 'mifgsm': []
        }
        for eps in EPSILONS:
            _, _, asr_f = eval_attack(model, loader, fgsm_attack, epsilon=eps)
            _, _, asr_i = eval_attack(model, loader, ifgsm_attack,
                                      epsilon=eps, num_steps=10)
            _, _, asr_m = eval_attack(model, loader, mifgsm_attack,
                                      epsilon=eps, num_steps=10, decay=1.0)
            results[ds_name]['fgsm'].append(asr_f)
            results[ds_name]['ifgsm'].append(asr_i)
            results[ds_name]['mifgsm'].append(asr_m)
            print(f'  {ds_name.upper()} ε={eps:.2f}: '
                  f'FGSM={asr_f:.1f}%  I-FGSM={asr_i:.1f}%  MI-FGSM={asr_m:.1f}%')

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {'fgsm': '#d62728', 'ifgsm': '#1f77b4', 'mifgsm': '#2ca02c'}
    labels = {'fgsm': 'FGSM (1 bước)', 'ifgsm': 'I-FGSM (10 bước)',
              'mifgsm': 'MI-FGSM (10 bước, μ=1)'}
    styles = {'fgsm': '--', 'ifgsm': '-', 'mifgsm': '-'}
    markers = {'fgsm': 'o', 'ifgsm': 's', 'mifgsm': '^'}

    for ax, ds in zip(axes, ['mnist', 'cifar10']):
        for att in ['fgsm', 'ifgsm', 'mifgsm']:
            ax.plot(EPSILONS, results[ds][att],
                    color=colors[att], linestyle=styles[att],
                    marker=markers[att], label=labels[att], lw=2, ms=6)
        ax.set_xlabel('Epsilon (ε)', fontsize=11)
        ax.set_ylabel('Attack Success Rate (%)', fontsize=11)
        ax.set_title(f'{ds.upper()} — FGSM vs I-FGSM vs MI-FGSM', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 105)

    plt.suptitle('So sánh ASR: FGSM / I-FGSM / MI-FGSM theo ε', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fname = 'exp7_attack_comparison_asr.png'
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {fname}')
    return results


# ── Exp 7B: ASR vs số bước T ─────────────────────────────────
def exp7b_asr_vs_steps():
    print('\n[7B] ASR vs số bước T — I-FGSM vs MI-FGSM (ε=0.20)')
    EPS = 0.20
    results = {}

    for ds_name, model_key in [('mnist','mnist'), ('cifar10','cifar10')]:
        model  = load_model(model_key)
        loader = get_loader(ds_name, N_SAMPLES)
        results[ds_name] = {'ifgsm': [], 'mifgsm': []}

        for T in STEPS_LIST:
            _, _, asr_i = eval_attack(model, loader, ifgsm_attack,
                                      epsilon=EPS, num_steps=T)
            _, _, asr_m = eval_attack(model, loader, mifgsm_attack,
                                      epsilon=EPS, num_steps=T, decay=1.0)
            results[ds_name]['ifgsm'].append(asr_i)
            results[ds_name]['mifgsm'].append(asr_m)
            print(f'  {ds_name.upper()} T={T:2d}: I-FGSM={asr_i:.1f}%  MI-FGSM={asr_m:.1f}%')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, ds in zip(axes, ['mnist', 'cifar10']):
        ax.plot(STEPS_LIST, results[ds]['ifgsm'],
                color='#1f77b4', marker='s', lw=2, ms=7, label='I-FGSM')
        ax.plot(STEPS_LIST, results[ds]['mifgsm'],
                color='#2ca02c', marker='^', lw=2, ms=7, label='MI-FGSM (μ=1)')
        ax.set_xlabel('Số bước T', fontsize=11)
        ax.set_ylabel('ASR (%)', fontsize=11)
        ax.set_title(f'{ds.upper()} — ASR theo số bước T  (ε=0.20)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 105)

    plt.suptitle('I-FGSM vs MI-FGSM: Hiệu quả theo số bước lặp T', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fname = 'exp7_steps_comparison.png'
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {fname}')
    return results


# ── Exp 7C: Transfer attack FGSM vs MI-FGSM ──────────────────
def exp7c_transfer():
    print('\n[7C] Transferability: FGSM vs MI-FGSM (CIFAR-10, ε=0.20)')
    EPS    = 0.20
    MODELS = ['cifar10', 'resnet18', 'mobilenetv2']
    LABELS = ['SimpleCNN', 'ResNet18', 'MobileNetV2']
    loader = get_loader('cifar10', N_SAMPLES)

    fgsm_matrix   = np.zeros((3, 3))
    mifgsm_matrix = np.zeros((3, 3))

    for i, src_key in enumerate(MODELS):
        src_model = load_model(src_key)
        for j, tgt_key in enumerate(MODELS):
            tgt_model = load_model(tgt_key)
            # Sinh adv từ src
            adv_list_f, adv_list_m, lbl_list = [], [], []
            for imgs, lbls in loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                with torch.no_grad():
                    preds = src_model(imgs).argmax(1)
                mask = preds == lbls
                if mask.sum() == 0:
                    continue
                imgs_c, lbls_c = imgs[mask], lbls[mask]
                adv_f = fgsm_attack(src_model, imgs_c, lbls_c, epsilon=EPS)
                adv_m = mifgsm_attack(src_model, imgs_c, lbls_c,
                                      epsilon=EPS, num_steps=10, decay=1.0)
                adv_list_f.append(adv_f)
                adv_list_m.append(adv_m)
                lbl_list.append(lbls_c)

            if not lbl_list:
                continue
            adv_f_all = torch.cat(adv_list_f)
            adv_m_all = torch.cat(adv_list_m)
            lbl_all   = torch.cat(lbl_list)

            with torch.no_grad():
                preds_f = tgt_model(adv_f_all).argmax(1)
                preds_m = tgt_model(adv_m_all).argmax(1)

            n = len(lbl_all)
            fgsm_matrix[i][j]   = (preds_f != lbl_all).float().mean().item() * 100
            mifgsm_matrix[i][j] = (preds_m != lbl_all).float().mean().item() * 100
            print(f'  {LABELS[i]:12s}→{LABELS[j]:12s}: '
                  f'FGSM={fgsm_matrix[i][j]:.1f}%  MI-FGSM={mifgsm_matrix[i][j]:.1f}%')

    # Plot side-by-side heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, mat, title in zip(axes,
                               [fgsm_matrix, mifgsm_matrix],
                               ['FGSM ASR (%)', 'MI-FGSM ASR (%)']):
        im = ax.imshow(mat, cmap='YlOrRd', vmin=0, vmax=100)
        plt.colorbar(im, ax=ax, label='ASR (%)')
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(LABELS, fontsize=9)
        ax.set_yticklabels(LABELS, fontsize=9)
        ax.set_xlabel('Target Model', fontsize=10)
        ax.set_ylabel('Source Model', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        for ii in range(3):
            for jj in range(3):
                val = mat[ii][jj]
                txt = f'{val:.1f}%' + (' (WB)' if ii == jj else '')
                ax.text(jj, ii, txt, ha='center', va='center',
                        fontsize=9, color='black' if val < 60 else 'white')

    plt.suptitle('Transferability CIFAR-10: FGSM vs MI-FGSM  (ε=0.20)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fname = 'exp7_transfer_fgsm_vs_mifgsm.png'
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {fname}')
    return fgsm_matrix.tolist(), mifgsm_matrix.tolist()


# ── Main ──────────────────────────────────────────────────────
def main():
    print('=' * 58)
    print('Thực nghiệm 7 — FGSM / I-FGSM / MI-FGSM Comparison')
    print('=' * 58)
    t0 = time.time()

    res_7a = exp7a_asr_vs_epsilon()
    res_7b = exp7b_asr_vs_steps()
    fgsm_t, mi_t = exp7c_transfer()

    log = {
        'exp7a_asr_vs_epsilon': res_7a,
        'exp7b_asr_vs_steps':   res_7b,
        'exp7c_transfer': {
            'fgsm_matrix':   fgsm_t,
            'mifgsm_matrix': mi_t,
        }
    }
    with open(os.path.join(LOG_DIR, 'exp7_mifgsm.json'), 'w') as f:
        json.dump(log, f, indent=2)

    print(f'\n✓ Thực nghiệm 7 hoàn tất trong {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
generate_arch_figs.py — VGG-style 3D CNN architecture diagrams.
Mỗi khối là một hộp 3D thực sự: front face gradient, top+right face shading,
nhiều conv layer = nhiều slab xếp chồng theo chiều sâu.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.colors as mc
import numpy as np
import os

os.makedirs('results/figures', exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Serif',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.3,
    'figure.facecolor': 'white',
})

# ── Perspective & layout ──────────────────────────────────────────────────────
PX  = 0.48    # x-shift per depth unit  (steeper → more 3-D)
PY  = 0.30    # y-shift per depth unit
FW  = 0.55    # front-face width of every slab
Y0  = 0.0     # common baseline

# ── Colour palette ────────────────────────────────────────────────────────────
C = dict(conv='#4472C4', pool='#C00000', fc='#548235',
         out='#375623', inp='#7F7F7F')

# ── Colour helpers ────────────────────────────────────────────────────────────
def _v(c):         return np.array(mc.to_rgb(c), float)
def _lt(c, f=1.45): return tuple(np.clip(_v(c)*f, 0, 1))
def _dk(c, f=0.52): return tuple(np.clip(_v(c)*f, 0, 1))

# ── Single slab with gradient front face ─────────────────────────────────────
def _slab(ax, x0, y0, w, h, d, color, zo=5):
    """
    Draw one 3-D slab.
      x0,y0 : bottom-left of front face
      w,h   : front-face width / height
      d     : depth into page  (∝ channel count)
    """
    dx, dy = d * PX, d * PY

    # — gradient front face (bottom darker → top brighter) ———————
    N = 20
    for i in range(N):
        t  = i / N
        yy = y0 + t * h
        hh = h / N
        fc = tuple(np.clip(_v(color) * (0.82 + 0.26 * t), 0, 1))
        ax.add_patch(Polygon(
            [(x0, yy), (x0+w, yy), (x0+w, yy+hh), (x0, yy+hh)],
            closed=True, fc=fc, ec='none', lw=0, zorder=zo))

    # — front face border ——————————————————————————————————————————
    ax.add_patch(Polygon(
        [(x0,y0),(x0+w,y0),(x0+w,y0+h),(x0,y0+h)],
        closed=True, fc='none', ec='white', lw=1.0, zorder=zo+1))

    # — top face (bright) ——————————————————————————————————————————
    ax.add_patch(Polygon(
        [(x0,y0+h),(x0+w,y0+h),(x0+w+dx,y0+h+dy),(x0+dx,y0+h+dy)],
        closed=True, fc=_lt(color), ec='white', lw=0.8, zorder=zo+1))

    # — right face (dark) ——————————————————————————————————————————
    ax.add_patch(Polygon(
        [(x0+w,y0),(x0+w+dx,y0+dy),(x0+w+dx,y0+h+dy),(x0+w,y0+h)],
        closed=True, fc=_dk(color), ec='white', lw=0.8, zorder=zo+1))


# ── Volume: n conv-layer slabs stacked in 3-D depth ──────────────────────────
def vol(ax, x0, y0, n, h, d, color, label=None, dim=None):
    """
    n   : number of conv layers (= number of slabs)
    h   : front-face height  (∝ spatial size)
    d   : depth per slab     (∝ channel count)
    Back slabs shifted by (i·d·PX, i·d·PY) so they peek behind the front one.
    """
    for i in range(n-1, -1, -1):           # draw back → front
        _slab(ax, x0 + i*d*PX, y0 + i*d*PY,
              FW, h, d, color, zo=20 + (n-i)*5)

    # label anchor: centre of the top perspective face (average over all slabs)
    cx = x0 + FW/2 + (n-1)*d*PX/2 + d*PX/2
    cy = y0 + h    + (n-1)*d*PY   + d*PY

    if label:
        ax.text(cx, cy + 0.16, label,
                ha='center', va='bottom', fontsize=9.5,
                fontweight='bold', color='#111111', zorder=90,
                bbox=dict(fc='white', ec='none', pad=2, alpha=0.75))

    if dim:
        ax.text(x0 + FW/2, y0 - 0.22, dim,
                ha='center', va='top', fontsize=7.8,
                color='#333333', zorder=90,
                bbox=dict(fc='white', ec='none', pad=2, alpha=0.85))

    return x0 + FW + n * d * PX     # rightmost x


def fc_vol(ax, x0, y0, h, d, color, fw=0.42, label=None, dim=None):
    """FC layer: single slab, slightly wider."""
    _slab(ax, x0, y0, fw, h, d, color, zo=20)
    cx = x0 + fw/2 + d*PX/2
    if label:
        ax.text(cx, y0+h+d*PY+0.16, label,
                ha='center', va='bottom', fontsize=9.5,
                fontweight='bold', color='#111111', zorder=90,
                bbox=dict(fc='white', ec='none', pad=2, alpha=0.75))
    if dim:
        ax.text(x0+fw/2, y0-0.22, dim,
                ha='center', va='top', fontsize=7.8,
                color='#333333', zorder=90,
                bbox=dict(fc='white', ec='none', pad=2, alpha=0.85))
    return x0 + fw + d * PX


def arr(ax, x1, x2, y):
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='->', color='#555555',
                                lw=1.4, mutation_scale=14), zorder=5)


def H(sp): return sp / 8.0           # height  ∝ spatial size
def D(ch): return max(0.40, ch/28.0) # depth   ∝ channel count


# ─────────────────────────────────────────────────────────────────────────────
#  Figure 1 — MNIST (1×28×28)
# ─────────────────────────────────────────────────────────────────────────────
def draw_mnist():
    fig, ax = plt.subplots(figsize=(13.5, 6.0))
    G = 0.75   # inter-group gap

    x = 0.5

    # Input
    re = vol(ax, x, Y0, 1, H(28), D(1), C['inp'],
             label='Input', dim='1×28×28')
    arr(ax, re, re+G, H(28)/2);  x = re + G

    # Block 1: Conv-1, Conv-2  (32 ch, 28×28)
    re = vol(ax, x, Y0, 2, H(28), D(32), C['conv'],
             label='Conv-1 / Conv-2')
    arr(ax, re, re+G, H(28)/2);  x = re + G

    # MaxPool → 32 ch, 14×14
    re = vol(ax, x, Y0, 1, H(14), D(32), C['pool'],
             label='MaxPool', dim='32 × 14 × 14')
    arr(ax, re, re+G, H(14)/2);  x = re + G

    # FC-1  (512)
    h1 = H(28) * 0.72
    re = fc_vol(ax, x, Y0, h1, 0.55, C['fc'], label='FC-1', dim='512')
    arr(ax, re, re+G*0.75, h1/2);  x = re + G*0.75

    # FC-2  (Output, 10)
    fc_vol(ax, x, Y0, H(28)*0.40, 0.55, C['out'],
           label='FC-2\n(Output)', dim='10')

    # Legend
    items = [('Conv + BN + ReLU', C['conv']), ('MaxPool 2×2', C['pool']),
             ('Fully Connected',  C['fc']),   ('Output (10 cls)', C['out'])]
    ax.legend([plt.Rectangle((0,0),1,1, fc=mc.to_rgb(c), ec='#888', lw=0.5)
               for _,c in items],
              [l for l,_ in items],
              loc='lower right', fontsize=9, framealpha=0.93,
              handlelength=1.6, handleheight=1.0, borderpad=0.7)

    ax.set_title('SimpleCNN Architecture — MNIST  (1 × 28 × 28)',
                 fontsize=13, fontweight='bold', pad=18)
    ax.set_xlim(-0.4, 12.5);  ax.set_ylim(-0.8, 6.2)
    ax.set_aspect('equal');   ax.axis('off')
    fig.savefig('results/figures/arch_mnist.png', dpi=300)
    plt.close(fig);  print('Saved: results/figures/arch_mnist.png')


# ─────────────────────────────────────────────────────────────────────────────
#  Figure 2 — CIFAR-10 (3×32×32)
# ─────────────────────────────────────────────────────────────────────────────
def draw_cifar():
    fig, ax = plt.subplots(figsize=(18.0, 6.0))
    G = 0.75

    x = 0.5

    # Input
    re = vol(ax, x, Y0, 1, H(32), D(3), C['inp'],
             label='Input', dim='3×32×32')
    arr(ax, re, re+G, H(32)/2);  x = re + G

    # Block 1: Conv-1, Conv-2  (32 ch, 32×32)
    re = vol(ax, x, Y0, 2, H(32), D(32), C['conv'],
             label='Conv-1 / Conv-2')
    arr(ax, re, re+G, H(32)/2);  x = re + G

    # MaxPool → 32 ch, 16×16
    re = vol(ax, x, Y0, 1, H(16), D(32), C['pool'],
             label='MaxPool', dim='32 × 16 × 16')
    arr(ax, re, re+G, H(16)/2);  x = re + G

    # Block 2: Conv-3, Conv-4  (64 ch, 16×16)
    re = vol(ax, x, Y0, 2, H(16), D(64), C['conv'],
             label='Conv-3 / Conv-4')
    arr(ax, re, re+G, H(16)/2);  x = re + G

    # MaxPool → 64 ch, 8×8
    re = vol(ax, x, Y0, 1, H(8), D(64), C['pool'],
             label='MaxPool', dim='64 × 8 × 8')
    arr(ax, re, re+G, H(8)/2);   x = re + G

    # FC-1  (512)
    h1 = H(32) * 0.72
    re = fc_vol(ax, x, Y0, h1, 0.55, C['fc'], label='FC-1', dim='512')
    arr(ax, re, re+G*0.75, h1/2);  x = re + G*0.75

    # FC-2  (Output, 10)
    fc_vol(ax, x, Y0, H(32)*0.38, 0.55, C['out'],
           label='FC-2\n(Output)', dim='10')

    # Legend
    items = [('Conv + BN + ReLU', C['conv']), ('MaxPool 2×2', C['pool']),
             ('Fully Connected',  C['fc']),   ('Output (10 cls)', C['out'])]
    ax.legend([plt.Rectangle((0,0),1,1, fc=mc.to_rgb(c), ec='#888', lw=0.5)
               for _,c in items],
              [l for l,_ in items],
              loc='lower right', fontsize=9, framealpha=0.93,
              handlelength=1.6, handleheight=1.0, borderpad=0.7)

    ax.set_title('SimpleCNN Architecture — CIFAR-10  (3 × 32 × 32)',
                 fontsize=13, fontweight='bold', pad=18)
    ax.set_xlim(-0.4, 17.5);  ax.set_ylim(-0.8, 6.8)
    ax.set_aspect('equal');   ax.axis('off')
    fig.savefig('results/figures/arch_cifar10.png', dpi=300)
    plt.close(fig);  print('Saved: results/figures/arch_cifar10.png')


if __name__ == '__main__':
    draw_mnist()
    draw_cifar()
    print('Done.')

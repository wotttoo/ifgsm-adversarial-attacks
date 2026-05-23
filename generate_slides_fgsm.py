"""
generate_slides_fgsm.py
Tạo file PowerPoint trình bày quá trình thực nghiệm FGSM.
Chạy:
    python generate_slides_fgsm.py
Output:
    results/TrinhChieu_FGSM.pptx
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt, Cm, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_THEME_COLOR
import copy

ROOT    = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(ROOT, "results", "figures")
OUT     = os.path.join(ROOT, "results", "TrinhChieu_FGSM.pptx")

# ── Màu sắc ─────────────────────────────────────────────────
C_NAVY   = RGBColor(0x1F, 0x38, 0x64)   # tiêu đề chính
C_BLUE   = RGBColor(0x2E, 0x75, 0xB6)   # tiêu đề phụ / accent
C_ORANGE = RGBColor(0xE3, 0x6B, 0x24)   # số liệu nổi bật
C_GREEN  = RGBColor(0x37, 0x86, 0x30)   # kết quả tốt
C_RED    = RGBColor(0xC0, 0x00, 0x00)   # cảnh báo
C_GRAY   = RGBColor(0x40, 0x40, 0x40)   # text thường
C_LGRAY  = RGBColor(0x88, 0x88, 0x88)   # text phụ
C_WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
C_BG     = RGBColor(0xF4, 0xF7, 0xFB)   # slide background

# ── Kích thước slide 16:9 ────────────────────────────────────
W = Inches(13.33)
H = Inches(7.5)


# ─────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────

def new_blank_slide(prs):
    layout = prs.slide_layouts[6]   # blank
    return prs.slides.add_slide(layout)


def bg_rect(slide, color=C_WHITE):
    """Tô nền toàn slide."""
    from pptx.util import Emu
    sp = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        0, 0, W, H
    )
    sp.fill.solid()
    sp.fill.fore_color.rgb = color
    sp.line.fill.background()
    sp.zorder = 0


def add_text_box(slide, text, left, top, width, height,
                 font_size=18, bold=False, italic=False,
                 color=C_GRAY, align=PP_ALIGN.LEFT,
                 word_wrap=True, font_name="Calibri"):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf    = txBox.text_frame
    tf.word_wrap = word_wrap
    p    = tf.paragraphs[0]
    p.alignment = align
    run  = p.add_run()
    run.text = text
    run.font.name  = font_name
    run.font.size  = Pt(font_size)
    run.font.bold  = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return txBox


def add_bullet_box(slide, items, left, top, width, height,
                   font_size=16, color=C_GRAY, bullet="•",
                   line_spacing=1.2, font_name="Calibri"):
    from pptx.util import Pt as PT
    from pptx.oxml.ns import qn
    from lxml import etree

    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf    = txBox.text_frame
    tf.word_wrap = True

    for i, item in enumerate(items):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.space_before = PT(4)
        p.space_after  = PT(4)
        run = p.add_run()
        run.text = f"{bullet}  {item}"
        run.font.name  = font_name
        run.font.size  = PT(font_size)
        run.font.color.rgb = color
    return txBox


def add_divider(slide, top, color=C_BLUE, thickness=Pt(1.5)):
    from pptx.util import Emu
    line = slide.shapes.add_shape(1, Inches(0.4), top, Inches(12.53), Emu(3000))
    line.fill.solid()
    line.fill.fore_color.rgb = color
    line.line.fill.background()


def add_image(slide, filename, left, top, width=None, height=None):
    path = os.path.join(FIG_DIR, filename)
    if not os.path.exists(path):
        add_text_box(slide, f"[{filename}]", left, top,
                     width or Inches(4), height or Inches(3),
                     font_size=10, color=C_RED)
        return None
    if width and height:
        return slide.shapes.add_picture(path, left, top, width, height)
    elif width:
        return slide.shapes.add_picture(path, left, top, width=width)
    elif height:
        return slide.shapes.add_picture(path, left, top, height=height)
    else:
        return slide.shapes.add_picture(path, left, top)


def header_bar(slide, title, subtitle=None):
    """Thanh tiêu đề gradient navy trên cùng."""
    bar = slide.shapes.add_shape(1, 0, 0, W, Inches(1.15))
    bar.fill.solid()
    bar.fill.fore_color.rgb = C_NAVY
    bar.line.fill.background()

    add_text_box(slide, title,
                 Inches(0.4), Inches(0.08), Inches(11), Inches(0.6),
                 font_size=26, bold=True, color=C_WHITE, font_name="Calibri")
    if subtitle:
        add_text_box(slide, subtitle,
                     Inches(0.4), Inches(0.65), Inches(11), Inches(0.42),
                     font_size=15, color=RGBColor(0xBF, 0xD7, 0xED), font_name="Calibri")


def slide_number(slide, num, total):
    add_text_box(slide, f"{num} / {total}",
                 Inches(12.3), Inches(7.1), Inches(0.9), Inches(0.35),
                 font_size=11, color=C_LGRAY, align=PP_ALIGN.RIGHT)


def accent_box(slide, text, left, top, width, height,
               bg=C_BLUE, fg=C_WHITE, font_size=20, bold=True):
    sp = slide.shapes.add_shape(1, left, top, width, height)
    sp.fill.solid()
    sp.fill.fore_color.rgb = bg
    sp.line.fill.background()
    tf = sp.text_frame
    tf.word_wrap = True
    p   = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = text
    run.font.name  = "Calibri"
    run.font.size  = Pt(font_size)
    run.font.bold  = bold
    run.font.color.rgb = fg


def kv_stat(slide, label, value, left, top, w=Inches(2.2), h=Inches(1.1),
            bg=C_NAVY, val_color=C_ORANGE):
    sp = slide.shapes.add_shape(1, left, top, w, h)
    sp.fill.solid()
    sp.fill.fore_color.rgb = bg
    from pptx.util import Pt as PT
    sp.line.fill.background()
    tf = sp.text_frame
    tf.word_wrap = False
    p1 = tf.paragraphs[0]
    p1.alignment = PP_ALIGN.CENTER
    r1 = p1.add_run()
    r1.text = value
    r1.font.name = "Calibri"; r1.font.size = PT(28)
    r1.font.bold = True; r1.font.color.rgb = val_color
    p2 = tf.add_paragraph()
    p2.alignment = PP_ALIGN.CENTER
    r2 = p2.add_run()
    r2.text = label
    r2.font.name = "Calibri"; r2.font.size = PT(11)
    r2.font.color.rgb = RGBColor(0xBF, 0xD7, 0xED)


def add_table(slide, headers, rows, left, top, width, height,
              header_bg=C_NAVY, alt_bg=RGBColor(0xE8, 0xF0, 0xF8),
              font_size=13):
    from pptx.util import Pt as PT
    cols = len(headers)
    table = slide.shapes.add_table(1 + len(rows), cols, left, top, width, height).table
    col_w = width // cols
    for i in range(cols):
        table.columns[i].width = col_w

    for i, h in enumerate(headers):
        cell = table.cell(0, i)
        cell.fill.solid()
        cell.fill.fore_color.rgb = header_bg
        p   = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = h
        run.font.name  = "Calibri"
        run.font.size  = PT(font_size)
        run.font.bold  = True
        run.font.color.rgb = C_WHITE

    for r, row in enumerate(rows):
        bg = alt_bg if r % 2 == 0 else C_WHITE
        for c, val in enumerate(row):
            cell = table.cell(r + 1, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg
            p   = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER
            run = p.add_run()
            run.text = str(val)
            run.font.name  = "Calibri"
            run.font.size  = PT(font_size)
            run.font.color.rgb = C_GRAY
    return table


# ─────────────────────────────────────────────────────────────
# Slides
# ─────────────────────────────────────────────────────────────

def slide_title(prs):
    slide = new_blank_slide(prs)
    bg_rect(slide, C_NAVY)

    # Hình nền trang trí
    deco = slide.shapes.add_shape(1, Inches(8.5), Inches(-0.5), Inches(6), Inches(8.5))
    deco.fill.solid()
    deco.fill.fore_color.rgb = RGBColor(0x2A, 0x47, 0x7A)
    deco.line.fill.background()

    add_text_box(slide, "BÁO CÁO THỰC NGHIỆM",
                 Inches(0.6), Inches(1.2), Inches(8), Inches(0.7),
                 font_size=20, bold=False, color=RGBColor(0xBF, 0xD7, 0xED),
                 font_name="Calibri")

    add_text_box(slide, "Tấn Công Đối Kháng FGSM",
                 Inches(0.6), Inches(1.9), Inches(8.5), Inches(1.1),
                 font_size=40, bold=True, color=C_WHITE, font_name="Calibri")

    add_text_box(slide, "Fast Gradient Sign Method",
                 Inches(0.6), Inches(3.0), Inches(8), Inches(0.6),
                 font_size=24, italic=True,
                 color=RGBColor(0x9D, 0xC3, 0xE6), font_name="Calibri")

    # Divider
    div = slide.shapes.add_shape(1, Inches(0.6), Inches(3.75), Inches(4), Inches(0.04))
    div.fill.solid(); div.fill.fore_color.rgb = C_ORANGE; div.line.fill.background()

    items = [
        "Dataset: MNIST & CIFAR-10",
        "Model: SimpleCNN, ResNet18, MobileNetV2",
        "Framework: PyTorch 2.0+ | Python 3.13",
        "Ngày: 14/05/2026",
    ]
    add_bullet_box(slide, items, Inches(0.6), Inches(3.95), Inches(7.5), Inches(2.2),
                   font_size=15, color=RGBColor(0xBF, 0xD7, 0xED), bullet="▸")


def slide_toc(prs, total):
    slide = new_blank_slide(prs)
    bg_rect(slide)
    header_bar(slide, "Nội dung trình bày")

    sections = [
        ("1", "Lý thuyết FGSM",               "Công thức, trực giác, ý nghĩa"),
        ("2", "Bộ dữ liệu & Mô hình",          "MNIST, CIFAR-10, SimpleCNN"),
        ("3", "Kết quả huấn luyện",             "Accuracy sau 20 epochs"),
        ("4", "Thực nghiệm 1 — ε vs ASR",       "FGSM sweep epsilon trên MNIST & CIFAR-10"),
        ("5", "Trực quan hóa ảnh đối kháng",    "FGSM Grid theo ε, phân phối xác suất"),
        ("6", "Transfer Attack",                "Cross-architecture black-box attack"),
        ("7", "Adversarial Training (FGSM-AT)", "Phòng thủ và đánh giá hiệu quả"),
        ("8", "Kết luận",                       "Tổng kết & hướng phát triển"),
    ]

    for i, (num, title, desc) in enumerate(sections):
        row = i // 2
        col = i % 2
        lx = Inches(0.5 + col * 6.4)
        ty = Inches(1.35 + row * 1.45)

        sp = slide.shapes.add_shape(1, lx, ty, Inches(5.9), Inches(1.25))
        sp.fill.solid()
        sp.fill.fore_color.rgb = RGBColor(0xF0, 0xF5, 0xFB)
        sp.line.color.rgb = C_BLUE

        nb = slide.shapes.add_shape(1, lx, ty, Inches(0.45), Inches(1.25))
        nb.fill.solid(); nb.fill.fore_color.rgb = C_BLUE; nb.line.fill.background()
        add_text_box(slide, num, lx + Inches(0.05), ty + Inches(0.3),
                     Inches(0.35), Inches(0.55), font_size=18, bold=True,
                     color=C_WHITE, align=PP_ALIGN.CENTER)

        add_text_box(slide, title, lx + Inches(0.55), ty + Inches(0.08),
                     Inches(5.1), Inches(0.45), font_size=15, bold=True, color=C_NAVY)
        add_text_box(slide, desc, lx + Inches(0.55), ty + Inches(0.52),
                     Inches(5.1), Inches(0.65), font_size=12, color=C_GRAY)

    slide_number(slide, 2, total)


def slide_theory(prs, total):
    slide = new_blank_slide(prs)
    bg_rect(slide)
    header_bar(slide, "Lý thuyết FGSM", "Fast Gradient Sign Method — Goodfellow et al., 2015")

    # Công thức
    accent_box(slide,
               "x_adv  =  x  +  ε · sign( ∇ₓ J(θ, x, y) )",
               Inches(1.0), Inches(1.3), Inches(11.3), Inches(0.85),
               bg=C_NAVY, font_size=24)

    # 4 thành phần
    parts = [
        ("x",      "Ảnh gốc đầu vào"),
        ("y",      "Nhãn thực sự"),
        ("J(θ,x,y)","Hàm mất mát (Cross-Entropy)"),
        ("∇ₓ J",   "Gradient theo pixel ảnh"),
        ("sign(·)", "Lấy dấu → +1 hoặc −1"),
        ("ε",       "Biên độ nhiễu tối đa (L∞)"),
        ("x_adv",   "Ảnh đối kháng sinh ra"),
    ]
    for i, (sym, meaning) in enumerate(parts):
        row = i // 4; col = i % 4
        lx = Inches(0.4 + col * 3.22)
        ty = Inches(2.35 + row * 0.95)
        sp = slide.shapes.add_shape(1, lx, ty, Inches(3.0), Inches(0.8))
        sp.fill.solid()
        sp.fill.fore_color.rgb = RGBColor(0xEA, 0xF2, 0xFB)
        sp.line.color.rgb = C_BLUE
        add_text_box(slide, sym, lx + Inches(0.08), ty + Inches(0.04),
                     Inches(0.8), Inches(0.38), font_size=17, bold=True,
                     color=C_NAVY, font_name="Courier New")
        add_text_box(slide, meaning, lx + Inches(0.08), ty + Inches(0.42),
                     Inches(2.8), Inches(0.32), font_size=11, color=C_GRAY)

    # Ý tưởng cốt lõi
    add_text_box(slide, "Ý tưởng cốt lõi",
                 Inches(0.4), Inches(4.3), Inches(3), Inches(0.4),
                 font_size=15, bold=True, color=C_NAVY)
    add_divider(slide, Inches(4.72), color=C_ORANGE, thickness=Pt(1))
    bullets = [
        "Tính gradient của loss theo ảnh đầu vào (không phải theo tham số mô hình)",
        "Lấy dấu gradient (sign) → mọi pixel đều bị dịch chuyển ±ε đồng đều",
        "Chỉ cần 1 lần forward + backward → cực kỳ nhanh (~3s/1000 ảnh trên CPU)",
        "Nhiễu vô hình với mắt người nhưng đủ để đánh lừa mô hình học sâu",
    ]
    add_bullet_box(slide, bullets, Inches(0.4), Inches(4.8), Inches(12.5), Inches(2.5),
                   font_size=14, color=C_GRAY)
    slide_number(slide, 3, total)


def slide_dataset(prs, total):
    slide = new_blank_slide(prs)
    bg_rect(slide)
    header_bar(slide, "Bộ dữ liệu & Mô hình")

    # MNIST box
    sp1 = slide.shapes.add_shape(1, Inches(0.4), Inches(1.3), Inches(5.9), Inches(3.8))
    sp1.fill.solid(); sp1.fill.fore_color.rgb = RGBColor(0xF0, 0xF5, 0xFB)
    sp1.line.color.rgb = C_BLUE

    add_text_box(slide, "MNIST", Inches(0.55), Inches(1.35),
                 Inches(2), Inches(0.45), font_size=20, bold=True, color=C_NAVY)
    add_text_box(slide, "Chữ số viết tay", Inches(0.55), Inches(1.77),
                 Inches(3), Inches(0.35), font_size=13, italic=True, color=C_BLUE)
    mnist_info = [
        "Kích thước: 28×28 px, grayscale (1 kênh)",
        "Train: 54,000 | Val: 6,000 | Test: 10,000",
        "10 lớp: chữ số 0–9",
        "Augment: RandomAffine ±10°, dịch ±10%",
        "Chuẩn hóa: [0.0, 1.0]",
        "Test Accuracy (SimpleCNN): 99.45%",
    ]
    add_bullet_box(slide, mnist_info, Inches(0.55), Inches(2.18),
                   Inches(5.6), Inches(2.7), font_size=13, bullet="▸")

    # CIFAR-10 box
    sp2 = slide.shapes.add_shape(1, Inches(6.8), Inches(1.3), Inches(5.9), Inches(3.8))
    sp2.fill.solid(); sp2.fill.fore_color.rgb = RGBColor(0xF0, 0xF5, 0xFB)
    sp2.line.color.rgb = C_BLUE

    add_text_box(slide, "CIFAR-10", Inches(6.95), Inches(1.35),
                 Inches(3), Inches(0.45), font_size=20, bold=True, color=C_NAVY)
    add_text_box(slide, "Ảnh vật thể thực tế", Inches(6.95), Inches(1.77),
                 Inches(3.5), Inches(0.35), font_size=13, italic=True, color=C_BLUE)
    cifar_info = [
        "Kích thước: 32×32 px, RGB (3 kênh)",
        "Train: 45,000 | Val: 5,000 | Test: 10,000",
        "10 lớp: airplane, car, bird, cat, deer...",
        "Augment: RandomCrop(32,pad=4) + HFlip",
        "Chuẩn hóa: mean=(0.491,0.482,0.447)",
        "Test Accuracy (SimpleCNN): 76.02%",
    ]
    add_bullet_box(slide, cifar_info, Inches(6.95), Inches(2.18),
                   Inches(5.6), Inches(2.7), font_size=13, bullet="▸")

    # SimpleCNN model info
    add_text_box(slide, "Mô hình: SimpleCNN",
                 Inches(0.4), Inches(5.25), Inches(4), Inches(0.4),
                 font_size=15, bold=True, color=C_NAVY)
    model_desc = (
        "Conv blocks → BatchNorm → ReLU → MaxPool → Dropout  |  "
        "Classifier: Flatten → Linear(512) → Dropout → Linear(10)"
    )
    add_text_box(slide, model_desc, Inches(0.4), Inches(5.65),
                 Inches(12.5), Inches(0.55), font_size=13, color=C_GRAY)

    # Stats
    for i, (lbl, val, bg) in enumerate([
        ("MNIST\nParams", "462K", C_NAVY),
        ("CIFAR-10\nParams", "2.2M", C_BLUE),
        ("Epochs", "20", RGBColor(0x37, 0x64, 0x91)),
        ("Optimizer", "Adam\n1e-3", RGBColor(0x2A, 0x56, 0x7E)),
    ]):
        kv_stat(slide, lbl, val,
                Inches(0.4 + i * 3.25), Inches(6.3),
                w=Inches(3.0), h=Inches(1.0), bg=bg, val_color=C_ORANGE)

    slide_number(slide, 4, total)


def slide_training(prs, total):
    slide = new_blank_slide(prs)
    bg_rect(slide)
    header_bar(slide, "Kết quả huấn luyện", "SimpleCNN — 20 epochs, Adam lr=0.001, StepLR(step=10, γ=0.1)")

    add_image(slide, "training_history_mnist.png",
              Inches(0.3), Inches(1.25), width=Inches(6.3))
    add_image(slide, "training_history_cifar10.png",
              Inches(6.8), Inches(1.25), width=Inches(6.3))

    add_text_box(slide, "MNIST", Inches(2.5), Inches(1.22),
                 Inches(2), Inches(0.3), font_size=13, bold=True, color=C_NAVY, align=PP_ALIGN.CENTER)
    add_text_box(slide, "CIFAR-10", Inches(9.2), Inches(1.22),
                 Inches(2), Inches(0.3), font_size=13, bold=True, color=C_NAVY, align=PP_ALIGN.CENTER)

    # Key stats
    stats = [
        ("MNIST\nVal Best", "98.98%", C_NAVY),
        ("MNIST\nTest Acc", "99.45%", C_GREEN),
        ("CIFAR-10\nVal Best", "74.36%", C_NAVY),
        ("CIFAR-10\nTest Acc", "76.02%", RGBColor(0x37, 0x64, 0x91)),
    ]
    for i, (lbl, val, bg) in enumerate(stats):
        kv_stat(slide, lbl, val,
                Inches(0.4 + i * 3.23), Inches(6.25),
                w=Inches(3.0), h=Inches(1.0), bg=bg, val_color=C_ORANGE)

    slide_number(slide, 5, total)


def slide_exp1_mnist(prs, total):
    slide = new_blank_slide(prs)
    bg_rect(slide)
    header_bar(slide, "Thực nghiệm 1 — ε vs ASR trên MNIST",
               "Sweep ε ∈ {0.05, 0.10, 0.15, 0.20, 0.25, 0.30} | T=40 bước | 1,273 mẫu đúng")

    add_image(slide, "exp1_acc_vs_epsilon_mnist.png",
              Inches(0.3), Inches(1.2), width=Inches(6.6))

    add_table(slide,
        headers=["ε", "Robust Acc", "ASR", "Thời gian"],
        rows=[
            ["0.05", "94.53%",  "4.95%",  "3.2s"],
            ["0.10", "70.63%",  "28.99%", "2.9s"],
            ["0.15", "42.73%",  "57.03%", "3.1s"],
            ["0.20", "25.16%",  "74.71%", "2.9s"],
            ["0.25", "15.94%",  "83.97%", "2.9s"],
            ["0.30", "11.17%",  "88.77%", "2.9s"],
        ],
        left=Inches(7.1), top=Inches(1.3), width=Inches(5.9), height=Inches(3.1),
        font_size=13,
    )

    # Key findings
    findings = [
        "FGSM bão hòa tại ~88.77% ASR dù ε tăng — trần 1 bước",
        "Clean accuracy: 99.45% → Robust acc: 11.17% ở ε=0.30",
        "Tốc độ: ~3s / 1,273 ảnh (chỉ 1 lần backward pass)",
    ]
    add_bullet_box(slide, findings, Inches(7.1), Inches(4.55), Inches(5.9), Inches(1.8),
                   font_size=13, color=C_GRAY)
    slide_number(slide, 6, total)


def slide_exp1_cifar(prs, total):
    slide = new_blank_slide(prs)
    bg_rect(slide)
    header_bar(slide, "Thực nghiệm 1 — ε vs ASR trên CIFAR-10",
               "Sweep ε ∈ {0.05 … 0.30} | T=40 bước | 973 mẫu đúng | Clean acc: 76.02%")

    add_image(slide, "exp1_acc_vs_epsilon_cifar10.png",
              Inches(0.3), Inches(1.2), width=Inches(6.6))

    add_table(slide,
        headers=["ε", "Robust Acc", "ASR", "Thời gian"],
        rows=[
            ["0.05", "21.02%", "72.35%", "2.5s"],
            ["0.10", "15.08%", "80.16%", "2.3s"],
            ["0.15", "12.66%", "83.35%", "2.3s"],
            ["0.20", "12.03%", "84.17%", "2.4s"],
            ["0.25", "11.64%", "84.69%", "2.2s"],
            ["0.30", "11.17%", "85.30%", "2.2s"],
        ],
        left=Inches(7.1), top=Inches(1.3), width=Inches(5.9), height=Inches(3.1),
        font_size=13,
    )

    findings = [
        "Ngay ε=0.05 đã đạt ASR 72.35% — CIFAR-10 nhạy hơn MNIST nhiều",
        "Bão hòa nhanh: ε=0.15→0.30 chỉ tăng thêm ~2pp ASR",
        "CIFAR-10 dễ bị tấn công hơn dù bài toán khó hơn (clean acc thấp)",
    ]
    add_bullet_box(slide, findings, Inches(7.1), Inches(4.55), Inches(5.9), Inches(1.8),
                   font_size=13, color=C_GRAY)
    slide_number(slide, 7, total)


def slide_comparison(prs, total):
    slide = new_blank_slide(prs)
    bg_rect(slide)
    header_bar(slide, "So sánh MNIST vs CIFAR-10", "Cùng mô hình SimpleCNN, cùng cấu hình FGSM")

    add_table(slide,
        headers=["Tiêu chí", "MNIST", "CIFAR-10"],
        rows=[
            ["Clean Accuracy",             "99.45%",   "76.02%"],
            ["FGSM ASR tại ε=0.05",        "4.95%",    "72.35%"],
            ["FGSM ASR tại ε=0.10",        "28.99%",   "80.16%"],
            ["FGSM ASR tại ε=0.20",        "74.71%",   "84.17%"],
            ["FGSM ASR ceiling (ε=0.30)",  "88.77%",   "85.30%"],
            ["ε để đạt ASR > 80%",         "~0.25",    "~0.10"],
            ["Thời gian trung bình",        "~3.0s",    "~2.3s"],
        ],
        left=Inches(0.5), top=Inches(1.3), width=Inches(12.3), height=Inches(3.6),
        font_size=15,
    )

    # Explanation
    explanation = [
        "CIFAR-10 nhạy hơn vì clean accuracy thấp → ranh giới quyết định gần với dữ liệu",
        "Không gian đầu vào cao hơn (3×32×32 = 3,072 vs 1×28×28 = 784) → gradient dễ tìm hướng tấn công",
        "FGSM tiệm cận trần năng lực giống nhau (~85–89%) — trần cố hữu của tấn công 1 bước",
    ]
    add_text_box(slide, "Tại sao CIFAR-10 nhạy hơn?",
                 Inches(0.5), Inches(5.1), Inches(4), Inches(0.4),
                 font_size=15, bold=True, color=C_NAVY)
    add_bullet_box(slide, explanation, Inches(0.5), Inches(5.55), Inches(12.3), Inches(1.7),
                   font_size=13, color=C_GRAY)
    slide_number(slide, 8, total)


def slide_visualize_grid(prs, total):
    slide = new_blank_slide(prs)
    bg_rect(slide)
    header_bar(slide, "Trực quan hóa — FGSM Grid theo ε",
               "Mỗi hàng: 1 mẫu ảnh. Cột 0: gốc. Cột 1–6: FGSM tại ε=0.05→0.30. Nhãn đỏ ✗ = bị đánh lừa.")

    add_image(slide, "fgsm_epsilon_grid_mnist.png",
              Inches(0.3), Inches(1.2), width=Inches(6.3))
    add_image(slide, "fgsm_epsilon_grid_cifar10.png",
              Inches(6.8), Inches(1.2), width=Inches(6.3))

    add_text_box(slide, "MNIST", Inches(2.4), Inches(1.19),
                 Inches(2), Inches(0.3), font_size=13, bold=True,
                 color=C_NAVY, align=PP_ALIGN.CENTER)
    add_text_box(slide, "CIFAR-10", Inches(9.2), Inches(1.19),
                 Inches(2), Inches(0.3), font_size=13, bold=True,
                 color=C_NAVY, align=PP_ALIGN.CENTER)

    obs = [
        "Ở ε nhỏ (0.05–0.10): ảnh gần như không đổi nhưng mô hình đã bị đánh lừa (CIFAR-10)",
        "Nhiễu ×10 bên dưới cho thấy cấu trúc gradient — không phải noise ngẫu nhiên",
        "Từ ε=0.15: nhãn đổi ✗ với confidence cao — mô hình 'tự tin sai'",
    ]
    add_bullet_box(slide, obs, Inches(0.3), Inches(6.5), Inches(12.7), Inches(0.9),
                   font_size=12, color=C_GRAY)
    slide_number(slide, 9, total)


def slide_visualize_probs(prs, total):
    slide = new_blank_slide(prs)
    bg_rect(slide)
    header_bar(slide, "Phân phối xác suất dự đoán trước và sau tấn công",
               "Xanh = trước tấn công | Đỏ = sau FGSM (ε=0.30) | Cột viền đen = nhãn đúng")

    add_image(slide, "exp3_pred_probs_mnist.png",
              Inches(0.3), Inches(1.2), width=Inches(6.3))
    add_image(slide, "exp3_pred_probs_cifar10.png",
              Inches(6.8), Inches(1.2), width=Inches(6.3))

    add_text_box(slide, "MNIST", Inches(2.4), Inches(1.19),
                 Inches(2), Inches(0.3), font_size=13, bold=True,
                 color=C_NAVY, align=PP_ALIGN.CENTER)
    add_text_box(slide, "CIFAR-10", Inches(9.2), Inches(1.19),
                 Inches(2), Inches(0.3), font_size=13, bold=True,
                 color=C_NAVY, align=PP_ALIGN.CENTER)

    obs = [
        "Trước tấn công: mô hình phân loại với độ tự tin ~99% (cột xanh cao ở nhãn đúng)",
        "Sau FGSM: xác suất dịch hoàn toàn sang lớp sai — cũng với độ tự tin ~99%",
        "Điều này xác nhận FGSM không chỉ làm mô hình sai mà còn làm nó 'cực kỳ tự tin sai'",
    ]
    add_bullet_box(slide, obs, Inches(0.3), Inches(6.5), Inches(12.7), Inches(0.9),
                   font_size=12, color=C_GRAY)
    slide_number(slide, 10, total)


def slide_transfer(prs, total):
    slide = new_blank_slide(prs)
    bg_rect(slide)
    header_bar(slide, "Transfer Attack — Cross-Architecture (FGSM)",
               "Ảnh đối kháng sinh từ model A có đánh lừa model B khác kiến trúc? | CIFAR-10 | ε=0.20")

    add_image(slide, "exp5_fgsm_only_heatmap_eps0.2.png",
              Inches(0.3), Inches(1.25), width=Inches(5.8))
    add_image(slide, "exp5_fgsm_only_transfer_asr_vs_epsilon.png",
              Inches(6.5), Inches(1.25), width=Inches(6.6))

    # Key insight boxes
    boxes = [
        ("White-box ASR\n(ε=0.20)", "81–89%", C_RED),
        ("Transfer ASR\n(black-box)", "64–76%", C_ORANGE),
        ("Cặp thấp nhất\nMNet→ResNet", "63.9%", RGBColor(0x37, 0x64, 0x91)),
        ("Cặp cao nhất\nSCNN→MNet", "73.8%", C_NAVY),
    ]
    for i, (lbl, val, bg) in enumerate(boxes):
        kv_stat(slide, lbl, val,
                Inches(0.4 + i * 3.23), Inches(6.25),
                w=Inches(3.0), h=Inches(1.0), bg=bg, val_color=C_ORANGE)

    slide_number(slide, 11, total)


def slide_transfer_insight(prs, total):
    slide = new_blank_slide(prs)
    bg_rect(slide)
    header_bar(slide, "Transfer Attack — Hàm ý an ninh thực tế")

    scenarios = [
        ("Black-box Attack\nVẫn hiệu quả",
         "Transfer rate 64–76% — không cần biết kiến trúc model đích\nvẫn có thể tấn công thành công",
         C_RED),
        ("Model mạnh hơn\nKhông an toàn hơn",
         "MobileNetV2 (clean 85%) vẫn bị SimpleCNN→MNet attack đạt 73.8% ASR",
         C_ORANGE),
        ("Khoảng cách kiến trúc\nCó ảnh hưởng nhưng...",
         "MNet↔ResNet transfer thấp nhất (64%) — nhưng vẫn nguy hiểm",
         RGBColor(0x37, 0x64, 0x91)),
        ("FGSM vs I-FGSM\nTransfer",
         "FGSM transfer đồng đều hơn I-FGSM — nhiễu 1 bước tổng quát hơn, ít overfit hơn",
         C_NAVY),
    ]

    for i, (title, body, bg) in enumerate(scenarios):
        row = i // 2; col = i % 2
        lx = Inches(0.4 + col * 6.5)
        ty = Inches(1.35 + row * 2.7)

        sp = slide.shapes.add_shape(1, lx, ty, Inches(6.1), Inches(2.4))
        sp.fill.solid(); sp.fill.fore_color.rgb = bg; sp.line.fill.background()

        add_text_box(slide, title, lx + Inches(0.15), ty + Inches(0.12),
                     Inches(5.8), Inches(0.75), font_size=16, bold=True, color=C_WHITE)
        add_text_box(slide, body, lx + Inches(0.15), ty + Inches(0.88),
                     Inches(5.8), Inches(1.4), font_size=13, color=RGBColor(0xDF, 0xEA, 0xF5))

    slide_number(slide, 12, total)


def slide_adv_training(prs, total):
    slide = new_blank_slide(prs)
    bg_rect(slide)
    header_bar(slide, "Adversarial Training — FGSM-AT",
               "Phòng thủ bằng cách huấn luyện model trên cả ảnh sạch lẫn ảnh đối kháng")

    # Công thức
    accent_box(slide,
               "loss  =  (1 − r) · CE(f(x), y)  +  r · CE(f(x_adv), y)     [r = 0.5]",
               Inches(0.5), Inches(1.25), Inches(12.3), Inches(0.75),
               bg=C_NAVY, font_size=19)

    add_image(slide, "adv_training_history_mnist.png",
              Inches(0.3), Inches(2.1), width=Inches(6.3))
    add_image(slide, "adv_training_history_cifar10.png",
              Inches(6.8), Inches(2.1), width=Inches(6.3))

    add_text_box(slide, "MNIST  (ε_train = 0.30)", Inches(1.5), Inches(2.08),
                 Inches(3.5), Inches(0.3), font_size=13, bold=True,
                 color=C_NAVY, align=PP_ALIGN.CENTER)
    add_text_box(slide, "CIFAR-10  (ε_train = 0.10)", Inches(7.5), Inches(2.08),
                 Inches(3.5), Inches(0.3), font_size=13, bold=True,
                 color=C_NAVY, align=PP_ALIGN.CENTER)

    stats = [
        ("MNIST Clean\nStd vs Adv", "99.3 → 99.0%", C_GREEN),
        ("MNIST FGSM ASR\nε=0.30", "89.2 → 7.0%", C_NAVY),
        ("CIFAR Clean\nStd vs Adv", "76.0 → 69.9%", RGBColor(0x37, 0x64, 0x91)),
        ("CIFAR FGSM ASR\nε=0.10", "82.2 → 68.8%", RGBColor(0xC0, 0x50, 0x00)),
    ]
    for i, (lbl, val, bg) in enumerate(stats):
        kv_stat(slide, lbl, val,
                Inches(0.4 + i * 3.23), Inches(6.3),
                w=Inches(3.0), h=Inches(1.0), bg=bg, val_color=C_ORANGE)

    slide_number(slide, 13, total)


def slide_adv_eval(prs, total):
    slide = new_blank_slide(prs)
    bg_rect(slide)
    header_bar(slide, "FGSM-AT: Standard vs Adversarial — So sánh ASR",
               "Standard model (đường đỏ) vs Adversarial model (đường xanh)")

    add_image(slide, "adv_robustness_comparison_mnist.png",
              Inches(0.3), Inches(1.2), width=Inches(6.3))
    add_image(slide, "adv_robustness_comparison_cifar10.png",
              Inches(6.8), Inches(1.2), width=Inches(6.3))

    add_text_box(slide, "MNIST", Inches(2.5), Inches(1.19),
                 Inches(2), Inches(0.3), font_size=13, bold=True,
                 color=C_NAVY, align=PP_ALIGN.CENTER)
    add_text_box(slide, "CIFAR-10", Inches(9.2), Inches(1.19),
                 Inches(2), Inches(0.3), font_size=13, bold=True,
                 color=C_NAVY, align=PP_ALIGN.CENTER)

    obs = [
        "MNIST: ASR giảm từ 89.2% → 7.0% (ε=0.30), clean accuracy chỉ giảm 0.3pp — gần như miễn phí",
        "CIFAR-10: ASR giảm ~13pp mỗi mức ε nhưng kèm trade-off clean accuracy −5.4pp",
        "Kết luận: Adversarial training hiệu quả hơn nhiều với bài toán đơn giản (MNIST)",
    ]
    add_bullet_box(slide, obs, Inches(0.3), Inches(6.5), Inches(12.7), Inches(0.9),
                   font_size=12, color=C_GRAY)
    slide_number(slide, 14, total)


def slide_conclusion(prs, total):
    slide = new_blank_slide(prs)
    bg_rect(slide, C_NAVY)

    add_text_box(slide, "Kết luận",
                 Inches(0.6), Inches(0.4), Inches(5), Inches(0.6),
                 font_size=28, bold=True, color=C_WHITE)
    div = slide.shapes.add_shape(1, Inches(0.6), Inches(1.05), Inches(3), Inches(0.04))
    div.fill.solid(); div.fill.fore_color.rgb = C_ORANGE; div.line.fill.background()

    conclusions = [
        ("1", "ε ảnh hưởng trực tiếp",
         "CIFAR-10: ASR 72% ngay ε=0.05 | MNIST: cần ε=0.25 để đạt ASR >80%\n"
         "FGSM bão hòa ~85–89% ASR — trần cố hữu của tấn công 1 bước"),
        ("2", "CIFAR-10 dễ bị tấn công hơn",
         "Dù bài toán khó hơn, clean accuracy thấp (76%) khiến ranh giới quyết định yếu\n"
         "→ Cải thiện clean accuracy chưa đủ để đảm bảo robustness"),
        ("3", "Transfer Attack vẫn nguy hiểm",
         "FGSM black-box ASR đạt 64–76% tại ε=0.20\n"
         "→ Attacker không cần biết kiến trúc đích vẫn tấn công hiệu quả"),
        ("4", "FGSM-AT hiệu quả trên MNIST",
         "ASR giảm 82pp (89→7%) với gần như không giảm clean accuracy\n"
         "CIFAR-10: moderate effect (−13pp ASR) với trade-off −5.4pp clean accuracy"),
    ]

    for i, (num, title, body) in enumerate(conclusions):
        row = i // 2; col = i % 2
        lx = Inches(0.4 + col * 6.5)
        ty = Inches(1.3 + row * 2.85)

        sp = slide.shapes.add_shape(1, lx, ty, Inches(6.1), Inches(2.6))
        sp.fill.solid()
        sp.fill.fore_color.rgb = RGBColor(0x2A, 0x47, 0x7A)
        sp.line.color.rgb = C_ORANGE

        add_text_box(slide, f"{num}.  {title}", lx + Inches(0.15), ty + Inches(0.12),
                     Inches(5.8), Inches(0.5), font_size=16, bold=True, color=C_ORANGE)
        add_text_box(slide, body, lx + Inches(0.15), ty + Inches(0.68),
                     Inches(5.8), Inches(1.75), font_size=12.5,
                     color=RGBColor(0xDF, 0xEA, 0xF5))

    slide_number(slide, total, total)


def slide_end(prs, total):
    slide = new_blank_slide(prs)
    bg_rect(slide, C_NAVY)

    deco = slide.shapes.add_shape(1, Inches(0), Inches(2.5), Inches(13.33), Inches(2.5))
    deco.fill.solid(); deco.fill.fore_color.rgb = RGBColor(0x2A, 0x47, 0x7A)
    deco.line.fill.background()

    add_text_box(slide, "Cảm ơn đã theo dõi!",
                 Inches(0), Inches(2.7), Inches(13.33), Inches(1.0),
                 font_size=40, bold=True, color=C_WHITE,
                 align=PP_ALIGN.CENTER)
    add_text_box(slide,
                 "Source code: github.com/wotttoo/ifgsm_project  |  Framework: PyTorch 2.0+",
                 Inches(0), Inches(4.0), Inches(13.33), Inches(0.45),
                 font_size=13, color=RGBColor(0xBF, 0xD7, 0xED),
                 align=PP_ALIGN.CENTER)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def build():
    prs = Presentation()
    prs.slide_width  = W
    prs.slide_height = H

    TOTAL = 16  # tổng số slide (không tính trang bìa và trang cuối)

    slide_title(prs)              # 1 — trang bìa
    slide_toc(prs, TOTAL)         # 2 — nội dung
    slide_theory(prs, TOTAL)      # 3 — lý thuyết
    slide_dataset(prs, TOTAL)     # 4 — dataset & model
    slide_training(prs, TOTAL)    # 5 — training results
    slide_exp1_mnist(prs, TOTAL)  # 6 — exp1 MNIST
    slide_exp1_cifar(prs, TOTAL)  # 7 — exp1 CIFAR-10
    slide_comparison(prs, TOTAL)  # 8 — so sánh
    slide_visualize_grid(prs, TOTAL)   # 9  — FGSM grid
    slide_visualize_probs(prs, TOTAL)  # 10 — pred probs
    slide_transfer(prs, TOTAL)         # 11 — transfer heatmap
    slide_transfer_insight(prs, TOTAL) # 12 — transfer insight
    slide_adv_training(prs, TOTAL)     # 13 — adv training
    slide_adv_eval(prs, TOTAL)         # 14 — adv eval
    slide_conclusion(prs, TOTAL)       # 15 — kết luận
    slide_end(prs, TOTAL)              # 16 — cảm ơn

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    prs.save(OUT)
    print(f"✓ Đã lưu: {OUT}")
    print(f"  Tổng số slide: {len(prs.slides)}")


if __name__ == "__main__":
    build()

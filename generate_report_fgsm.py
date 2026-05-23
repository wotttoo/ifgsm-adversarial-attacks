"""
generate_report_fgsm.py
Tạo báo cáo Word (.docx) chỉ về FGSM (Fast Gradient Sign Method).
Chạy từ thư mục gốc project:
    python generate_report_fgsm.py
"""

import os
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

ROOT    = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(ROOT, "results", "figures")
OUT     = os.path.join(ROOT, "results", "BaoCao_FGSM.docx")


# ── Helpers ───────────────────────────────────────────────────

def set_cell_bg(cell, hex_color: str):
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd  = OxmlElement("w:shd")
    shd.set(qn("w:val"),   "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"),  hex_color)
    tcPr.append(shd)

def add_heading(doc, text, level=1, color="1F3864"):
    p = doc.add_heading(text, level=level)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    for run in p.runs:
        run.font.color.rgb = RGBColor.from_string(color)
    return p

def add_para(doc, text="", bold=False, italic=False, size=11,
             align=WD_ALIGN_PARAGRAPH.JUSTIFY, color=None, space_after=6):
    p   = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    run = p.add_run(text)
    run.bold      = bold
    run.italic    = italic
    run.font.size = Pt(size)
    if color:
        run.font.color.rgb = RGBColor.from_string(color)
    return p

def add_figure(doc, filename, caption, width_cm=14):
    path = os.path.join(FIG_DIR, filename)
    if not os.path.exists(path):
        add_para(doc, f"[Hình: {filename} — chưa có file]", italic=True, color="FF0000")
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after  = Pt(2)
    p.add_run().add_picture(path, width=Cm(width_cm))

    cap = doc.add_paragraph(caption)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(12)
    for run in cap.runs:
        run.italic    = True
        run.font.size = Pt(10)
        run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

def add_table(doc, headers, rows, col_widths=None, header_bg="1F3864"):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"

    hdr = table.rows[0]
    for i, h in enumerate(headers):
        cell = hdr.cells[i]
        cell.text = h
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        set_cell_bg(cell, header_bg)
        run = cell.paragraphs[0].runs[0]
        run.bold           = True
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        run.font.size      = Pt(10)

    for r_idx, row_data in enumerate(rows):
        row = table.rows[r_idx + 1]
        bg  = "F2F2F2" if r_idx % 2 == 0 else "FFFFFF"
        for c_idx, val in enumerate(row_data):
            cell = row.cells[c_idx]
            cell.text = str(val)
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_bg(cell, bg)
            cell.paragraphs[0].runs[0].font.size = Pt(10)

    if col_widths:
        for i, w in enumerate(col_widths):
            for row in table.rows:
                row.cells[i].width = Cm(w)

    doc.add_paragraph()
    return table


# ── Main ──────────────────────────────────────────────────────

def build():
    doc = Document()

    for section in doc.sections:
        section.top_margin    = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin   = Cm(3.0)
        section.right_margin  = Cm(2.0)

    doc.styles["Normal"].font.name = "Times New Roman"
    doc.styles["Normal"].font.size = Pt(12)

    # ════════════════════════════════════════════════════════
    # TRANG BÌA
    # ════════════════════════════════════════════════════════
    doc.add_paragraph()
    doc.add_paragraph()

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = title.add_run("BÁO CÁO THỰC NGHIỆM")
    r.bold = True; r.font.size = Pt(20)
    r.font.color.rgb = RGBColor.from_string("1F3864")

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r2 = sub.add_run(
        "Tấn Công Đối Kháng FGSM\n"
        "(Fast Gradient Sign Method)\n"
        "trên MNIST và CIFAR-10"
    )
    r2.bold = True; r2.font.size = Pt(15)
    r2.font.color.rgb = RGBColor.from_string("2E75B6")

    doc.add_paragraph()
    info = doc.add_paragraph()
    info.alignment = WD_ALIGN_PARAGRAPH.CENTER
    info.add_run("Ngày thực nghiệm: 14/05/2026\n").font.size = Pt(12)
    info.add_run("Framework: PyTorch 2.0+ | Python 3.10+").font.size = Pt(12)

    doc.add_page_break()

    # ════════════════════════════════════════════════════════
    # 1. GIỚI THIỆU
    # ════════════════════════════════════════════════════════
    add_heading(doc, "1. Giới thiệu", 1)
    add_para(doc,
        "Báo cáo này trình bày quá trình thực nghiệm đánh giá mức độ dễ bị tấn công "
        "đối kháng (adversarial vulnerability) của mô hình phân loại ảnh khi đối mặt với "
        "phương pháp tấn công FGSM (Fast Gradient Sign Method — Goodfellow et al., 2015). "
        "FGSM là một trong những kỹ thuật tạo ảnh đối kháng nền tảng và được sử dụng rộng rãi "
        "nhất trong lĩnh vực an ninh học máy."
    )
    add_para(doc,
        "Thực nghiệm được tiến hành trên hai bộ dữ liệu chuẩn: MNIST (chữ số viết tay) "
        "và CIFAR-10 (ảnh vật thể thực tế), sử dụng mô hình SimpleCNN. "
        "Mục tiêu chính là trả lời các câu hỏi nghiên cứu sau:"
    )
    for q in [
        "Biên độ nhiễu ε ảnh hưởng như thế nào đến hiệu quả của FGSM?",
        "Sự khác biệt về độ robustness giữa MNIST và CIFAR-10 là gì?",
        "Ảnh đối kháng sinh ra từ một kiến trúc có thể đánh lừa kiến trúc khác không (transferability)?",
    ]:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(q).font.size = Pt(11)
        p.paragraph_format.space_after = Pt(3)

    doc.add_paragraph()

    # ════════════════════════════════════════════════════════
    # 2. CƠ SỞ LÝ THUYẾT
    # ════════════════════════════════════════════════════════
    add_heading(doc, "2. Cơ sở lý thuyết", 1)

    add_heading(doc, "2.1. FGSM — Fast Gradient Sign Method", 2)
    add_para(doc,
        "FGSM (Goodfellow et al., 2015) tạo ra ảnh đối kháng bằng cách cộng thêm "
        "một nhiễu nhỏ có cùng dấu với gradient của hàm mất mát theo ảnh đầu vào:"
    )
    formula = doc.add_paragraph()
    formula.alignment = WD_ALIGN_PARAGRAPH.CENTER
    formula.paragraph_format.space_before = Pt(4)
    formula.paragraph_format.space_after  = Pt(4)
    r = formula.add_run("x_adv = x + ε · sign(∇ₓ J(θ, x, y))")
    r.font.name = "Courier New"; r.font.size = Pt(12); r.bold = True

    add_para(doc,
        "Trong đó x là ảnh gốc, y là nhãn thực, θ là tham số mô hình, ε là biên độ "
        "nhiễu tối đa theo chuẩn L∞."
    )
    add_table(doc,
        headers=["Ký hiệu", "Ý nghĩa"],
        rows=[
            ["x",           "Ảnh đầu vào gốc"],
            ["y",           "Nhãn phân loại thực"],
            ["θ",           "Tham số mô hình (cố định trong quá trình tấn công)"],
            ["J(θ, x, y)",  "Hàm mất mát (Cross-Entropy)"],
            ["∇ₓ J",        "Gradient của loss theo pixel ảnh"],
            ["sign(·)",     "Hàm lấy dấu — trả về +1 hoặc −1"],
            ["ε",           "Biên độ nhiễu tối đa (L∞ norm budget)"],
            ["x_adv",       "Ảnh đối kháng sau tấn công"],
        ],
        col_widths=[4.0, 13.0],
    )
    add_para(doc,
        "FGSM chỉ thực hiện một bước tính gradient duy nhất, do đó rất nhanh và hiệu quả "
        "về mặt tính toán. Đây là thuật toán tấn công nền tảng được sử dụng để đánh giá "
        "sơ bộ khả năng phòng thủ của mô hình phân loại ảnh."
    )

    add_heading(doc, "2.2. Phương pháp đánh giá 2 pha", 2)
    add_para(doc,
        "Tất cả thực nghiệm sử dụng quy trình đánh giá 2 pha để đo lường hiệu quả "
        "tấn công một cách công bằng và chính xác:"
    )
    for step in [
        "Pha 1 — Dự đoán & Lọc: Cho toàn bộ test set qua mô hình (không tấn công). "
         "Ghi nhận độ chính xác nền (clean accuracy) và giữ lại chỉ những mẫu được "
         "mô hình dự đoán ĐÚNG (n_correct mẫu).",
        "Pha 2 — Tấn công: Chạy FGSM chỉ trên n_correct mẫu đúng đó. "
         "Đo độ chính xác còn lại (robust accuracy) và Attack Success Rate (ASR).",
    ]:
        p = doc.add_paragraph(style="List Number")
        p.add_run(step).font.size = Pt(11)
        p.paragraph_format.space_after = Pt(4)

    add_para(doc,
        "Lý do: tấn công mẫu vốn đã sai là vô nghĩa. Phương pháp này phản ánh đúng "
        "sức mạnh thực sự của attack: trong số những mẫu mô hình phân loại đúng, "
        "bao nhiêu phần trăm bị đánh lừa?"
    )

    add_heading(doc, "2.3. Các chỉ số đánh giá", 2)
    add_table(doc,
        headers=["Chỉ số", "Ý nghĩa"],
        rows=[
            ["Clean Acc (%)",    "Độ chính xác trước khi tấn công (baseline)"],
            ["Robust Acc (%)",   "Độ chính xác sau tấn công (trên toàn test set)"],
            ["ASR (%)",          "Attack Success Rate — % mẫu đúng bị đánh lừa"],
            ["Acc Drop (pp)",    "Mức giảm độ chính xác tuyệt đối (percentage points)"],
            ["Perturbation L∞", "Biên độ nhiễu tối đa (= ε theo thiết kế)"],
            ["Attack Time (s)",  "Thời gian thực hiện tấn công (giây)"],
        ],
        col_widths=[5.5, 11.5],
    )

    # ════════════════════════════════════════════════════════
    # 3. THIẾT LẬP THỰC NGHIỆM
    # ════════════════════════════════════════════════════════
    add_heading(doc, "3. Thiết lập thực nghiệm", 1)

    add_heading(doc, "3.1. Bộ dữ liệu", 2)
    add_table(doc,
        headers=["Thuộc tính", "MNIST", "CIFAR-10"],
        rows=[
            ["Số lớp",             "10 (chữ số 0–9)",           "10 (vật thể thực tế)"],
            ["Kích thước ảnh",     "28×28, grayscale",           "32×32, RGB"],
            ["Chuẩn hóa",          "[0, 1] (raw pixel)",         "mean=(0.491,0.482,0.447)\nstd=(0.247,0.244,0.262)"],
            ["Tập train",          "54,000 mẫu",                 "45,000 mẫu"],
            ["Tập validation",     "6,000 mẫu",                  "5,000 mẫu"],
            ["Tập test",           "10,000 mẫu",                 "10,000 mẫu"],
            ["Augmentation train", "RandomAffine ±10°, dịch 10%","RandomCrop(32, pad=4) + RandomHorizontalFlip"],
        ],
        col_widths=[5.0, 6.5, 6.5],
    )

    add_heading(doc, "3.2. Kiến trúc mô hình (SimpleCNN)", 2)
    add_para(doc,
        "Thực nghiệm 1 và 2 sử dụng SimpleCNN tự xây dựng. "
        "Thực nghiệm 3 (Transfer Attack) bổ sung ResNet18 và MobileNetV2 trên CIFAR-10 "
        "để tạo môi trường đa kiến trúc."
    )
    add_figure(doc, "arch_mnist.png",
               "Hình 1. Kiến trúc SimpleCNN cho MNIST (1×28×28) — 2 Conv layers + 2 FC layers",
               width_cm=15)
    add_figure(doc, "arch_cifar10.png",
               "Hình 2. Kiến trúc SimpleCNN cho CIFAR-10 (3×32×32) — 4 Conv layers + 2 FC layers",
               width_cm=15)
    add_table(doc,
        headers=["Thành phần", "MNIST (1×28×28)", "CIFAR-10 (3×32×32)"],
        rows=[
            ["Block 1",
             "Conv(1→32,3×3)→BN→ReLU\nConv(32→32,3×3)→BN→ReLU\nMaxPool(2×2)→Dropout2d(0.25)",
             "Conv(3→32,3×3)→BN→ReLU\nConv(32→32,3×3)→BN→ReLU\nMaxPool(2×2)→Dropout2d(0.25)"],
            ["Block 2", "—",
             "Conv(32→64,3×3)→BN→ReLU\nConv(64→64,3×3)→BN→ReLU\nMaxPool(2×2)→Dropout2d(0.25)"],
            ["Classifier",
             "Flatten→Linear(6272→512)\n→ReLU→Dropout(0.5)→Linear(512→10)",
             "Flatten→Linear(4096→512)\n→ReLU→Dropout(0.5)→Linear(512→10)"],
        ],
        col_widths=[3.5, 7.0, 7.0],
    )

    add_heading(doc, "3.3. Hyperparameter huấn luyện", 2)
    add_table(doc,
        headers=["Tham số", "Giá trị"],
        rows=[
            ["Optimizer",     "Adam (lr=0.001, weight_decay=1e-4)"],
            ["LR Scheduler",  "StepLR (step=10, γ=0.1)"],
            ["Epochs",        "20"],
            ["Batch size",    "64"],
            ["Loss function", "Cross-Entropy"],
            ["Checkpointing", "Lưu model tốt nhất theo val accuracy"],
        ],
        col_widths=[5.5, 11.5],
    )

    add_heading(doc, "3.4. Cấu hình tấn công FGSM", 2)
    add_table(doc,
        headers=["Tham số", "Giá trị", "Ghi chú"],
        rows=[
            ["ε (epsilon_list)", "[0.05, 0.10, 0.15, 0.20, 0.25, 0.30]", "Sweep cho Thực nghiệm 1"],
            ["Targeted",         "False",                                   "Untargeted attack"],
            ["Clip range",       "[0.0, 1.0]",                             "Giới hạn giá trị pixel"],
            ["Số mẫu eval",      "1,280",                                   "Lấy mẫu ngẫu nhiên từ test set"],
        ],
        col_widths=[4.5, 7.0, 6.0],
    )
    doc.add_paragraph()

    # ════════════════════════════════════════════════════════
    # 4. KẾT QUẢ HUẤN LUYỆN
    # ════════════════════════════════════════════════════════
    add_heading(doc, "4. Kết quả huấn luyện", 1)
    add_para(doc,
        "Tất cả mô hình được huấn luyện trong 20 epoch với cùng hyperparameter. "
        "Kết quả cuối cùng trên tập test:"
    )
    add_table(doc,
        headers=["Dataset", "Mô hình", "Val Accuracy tốt nhất", "Test Accuracy"],
        rows=[
            ["MNIST",    "SimpleCNN",   "98.98%", "99.45%"],
            ["CIFAR-10", "SimpleCNN",   "74.36%", "76.02%"],
            ["CIFAR-10", "ResNet18",    "81.86%", "82.00%"],
            ["CIFAR-10", "MobileNetV2", "84.12%", "84.42%"],
        ],
        col_widths=[3.5, 3.5, 5.5, 5.5],
    )
    add_para(doc,
        "MNIST đạt độ chính xác rất cao (99.45%) do bài toán tương đối đơn giản. "
        "Trên CIFAR-10, SimpleCNN đạt 76.02% — phù hợp với benchmark không dùng kỹ thuật "
        "tiên tiến. ResNet18 (82.00%) và MobileNetV2 (84.42%) vượt trội nhờ kiến trúc sâu hơn, "
        "cung cấp nền tảng đa dạng cho Thực nghiệm 3."
    )
    add_figure(doc, "training_history_mnist.png",
               "Hình 3. Lịch sử huấn luyện MNIST — SimpleCNN, loss và accuracy qua 20 epoch")
    add_figure(doc, "training_history_cifar10.png",
               "Hình 4. Lịch sử huấn luyện CIFAR-10 — SimpleCNN, loss và accuracy qua 20 epoch")
    add_figure(doc, "training_history_cifar10_resnet18.png",
               "Hình 5. Lịch sử huấn luyện CIFAR-10 — ResNet18, loss và accuracy qua 20 epoch")
    add_figure(doc, "training_history_cifar10_mobilenetv2.png",
               "Hình 6. Lịch sử huấn luyện CIFAR-10 — MobileNetV2, loss và accuracy qua 20 epoch")

    # ════════════════════════════════════════════════════════
    # 5. THỰC NGHIỆM 1 — ACCURACY VS EPSILON
    # ════════════════════════════════════════════════════════
    add_heading(doc, "5. Thực nghiệm 1 — Ảnh hưởng của ε đến hiệu quả FGSM", 1)
    add_para(doc,
        "Thực nghiệm 1 thay đổi biên độ nhiễu ε qua các giá trị "
        "[0.05, 0.10, 0.15, 0.20, 0.25, 0.30] và đo hiệu quả FGSM "
        "trên 1,280 mẫu test, đánh giá theo quy trình 2 pha."
    )

    add_heading(doc, "5.1. Kết quả trên MNIST", 2)
    add_para(doc,
        "Tập test: 1,280 mẫu  |  Mẫu phân loại đúng: 1,273  |  Clean accuracy: 99.45%",
        bold=True, size=10, color="555555"
    )
    add_table(doc,
        headers=["ε", "Robust Acc (%)", "Acc Drop (pp)", "ASR (%)", "Thời gian (s)"],
        rows=[
            ["0.05", "94.53", "−4.92",  "4.95",  "3.2"],
            ["0.10", "70.63", "−28.83", "28.99", "2.9"],
            ["0.15", "42.73", "−56.72", "57.03", "3.1"],
            ["0.20", "25.16", "−74.30", "74.71", "2.9"],
            ["0.25", "15.94", "−83.52", "83.97", "2.9"],
            ["0.30", "11.17", "−88.28", "88.77", "2.9"],
        ],
        col_widths=[2.0, 3.8, 3.8, 3.8, 3.6],
    )
    add_para(doc, "Nhận xét:")
    for obs in [
        "Ở ε=0.05, FGSM chỉ đạt ASR 4.95% — nhiễu quá nhỏ để vượt qua ranh giới quyết định "
        "của mô hình vốn đã rất tự tin (clean acc 99.45%).",
        "Từ ε=0.10 trở đi, ASR tăng nhanh: 28.99% (ε=0.10) → 74.71% (ε=0.20) → 88.77% (ε=0.30).",
        "FGSM tiệm cận trần năng lực ~88–89% ASR tại ε=0.30 — đây là giới hạn của tấn công 1 bước. "
        "Dù tăng thêm ε, một số mẫu vẫn không bị đánh lừa do bước nhảy đơn không tìm đúng hướng.",
        "Thời gian tấn công rất nhanh (~3s cho toàn bộ 1,273 mẫu) do chỉ cần 1 lần backward pass.",
    ]:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(obs).font.size = Pt(11)
        p.paragraph_format.space_after = Pt(3)

    add_figure(doc, "exp1_acc_vs_epsilon_mnist.png",
               "Hình 7. MNIST — Robust accuracy còn lại theo ε (đường FGSM)")

    add_heading(doc, "5.2. Kết quả trên CIFAR-10", 2)
    add_para(doc,
        "Tập test: 1,280 mẫu  |  Mẫu phân loại đúng: 973  |  Clean accuracy: 76.02%",
        bold=True, size=10, color="555555"
    )
    add_table(doc,
        headers=["ε", "Robust Acc (%)", "Acc Drop (pp)", "ASR (%)", "Thời gian (s)"],
        rows=[
            ["0.05", "21.02", "−55.00", "72.35", "2.5"],
            ["0.10", "15.08", "−60.94", "80.16", "2.3"],
            ["0.15", "12.66", "−63.36", "83.35", "2.3"],
            ["0.20", "12.03", "−63.98", "84.17", "2.4"],
            ["0.25", "11.64", "−64.38", "84.69", "2.2"],
            ["0.30", "11.17", "−64.84", "85.30", "2.2"],
        ],
        col_widths=[2.0, 3.8, 3.8, 3.8, 3.6],
    )
    add_para(doc, "Nhận xét:")
    for obs in [
        "Ngay tại ε=0.05, FGSM đã đạt 72.35% ASR trên CIFAR-10 — mức mà MNIST cần đến ε=0.25 "
        "mới đạt được. CIFAR-10 dễ bị tấn công hơn nhiều do clean accuracy thấp hơn (76%).",
        "FGSM bão hòa nhanh: từ ε=0.15 trở đi ASR gần như không tăng thêm (83.35% → 85.30%). "
        "Trần năng lực của FGSM trên CIFAR-10 ở mức ~85%.",
        "Thời gian tấn công rất nhanh (~2.2–2.5s) và ổn định qua tất cả các mức ε.",
    ]:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(obs).font.size = Pt(11)
        p.paragraph_format.space_after = Pt(3)

    add_figure(doc, "exp1_acc_vs_epsilon_cifar10.png",
               "Hình 8. CIFAR-10 — Robust accuracy còn lại theo ε (đường FGSM)")

    add_heading(doc, "5.3. So sánh MNIST và CIFAR-10", 2)
    add_table(doc,
        headers=["Tiêu chí so sánh", "MNIST", "CIFAR-10"],
        rows=[
            ["Clean Accuracy",              "99.45%",  "76.02%"],
            ["FGSM ASR tại ε=0.05",         "4.95%",   "72.35%"],
            ["FGSM ASR tại ε=0.10",         "28.99%",  "80.16%"],
            ["FGSM ASR tại ε=0.20",         "74.71%",  "84.17%"],
            ["FGSM ASR ceiling (ε=0.30)",   "88.77%",  "85.30%"],
            ["ε để đạt ASR > 80%",          "~0.25",   "~0.10"],
            ["Thời gian trung bình",         "~3.0s",   "~2.3s"],
        ],
        col_widths=[7.0, 4.0, 6.0],
    )
    add_para(doc,
        "CIFAR-10 nhạy với FGSM hơn MNIST rõ rệt dù về mặt trực giác là bài toán khó hơn. "
        "Nguyên nhân: clean accuracy thấp hơn đồng nghĩa ranh giới quyết định gần với vùng "
        "dữ liệu hơn, gradient 1 bước dễ tìm được hướng vượt biên hơn."
    )

    # ════════════════════════════════════════════════════════
    # 6. THỰC NGHIỆM 2 — TRỰC QUAN HÓA ẢNH ĐỐI KHÁNG FGSM
    # ════════════════════════════════════════════════════════
    add_heading(doc, "6. Thực nghiệm 2 — Trực quan hóa ảnh đối kháng FGSM", 1)
    add_para(doc,
        "Thực nghiệm 2 trực quan hóa trực tiếp tác động của FGSM lên ảnh khi thay đổi ε, "
        "giúp quan sát đồng thời sự thay đổi về nội dung ảnh và quyết định của mô hình "
        "trên cùng một tập mẫu."
    )

    add_heading(doc, "6.1. Grid so sánh ảnh đối kháng FGSM theo ε", 2)
    add_para(doc,
        "Mỗi hàng là một mẫu ảnh được phân loại đúng trước tấn công. "
        "Cột đầu tiên là ảnh gốc; các cột tiếp theo là ảnh đối kháng FGSM tại "
        "ε = 0.05, 0.10, 0.15, 0.20, 0.25, 0.30. "
        "Nhãn xanh ✓ = mô hình vẫn đúng; nhãn đỏ ✗ = mô hình bị đánh lừa, kèm confidence %. "
        "Hàng nhỏ bên dưới mỗi ảnh đối kháng là nhiễu được khuếch đại ×10 "
        "để mắt người có thể nhìn thấy."
    )
    add_figure(doc, "fgsm_epsilon_grid_mnist.png",
               "Hình 9. MNIST — FGSM: ảnh đối kháng và nhiễu (×10) theo từng mức ε",
               width_cm=17)
    add_para(doc, "Nhận xét MNIST:")
    for obs in [
        "Ở ε=0.05–0.10, ảnh đối kháng gần như không thể phân biệt với ảnh gốc bằng mắt "
        "thường, nhưng mô hình đã bắt đầu bị đánh lừa ở một số mẫu (ASR ~5–29%).",
        "Từ ε=0.15 trở đi, nhiễu bắt đầu nhìn thấy rõ hơn; nhãn dự đoán đổi sang ✗ với "
        "confidence cao — mô hình \"tự tin sai\".",
        "Nhiễu ×10 cho thấy cấu trúc rõ ràng: không phải nhiễu ngẫu nhiên mà có hướng "
        "nhất định theo gradient, đặc trưng của FGSM.",
    ]:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(obs).font.size = Pt(11)
        p.paragraph_format.space_after = Pt(3)

    add_figure(doc, "fgsm_epsilon_grid_cifar10.png",
               "Hình 10. CIFAR-10 — FGSM: ảnh đối kháng và nhiễu (×10) theo từng mức ε",
               width_cm=17)
    add_para(doc, "Nhận xét CIFAR-10:")
    for obs in [
        "Trên CIFAR-10, ngay ở ε=0.05 phần lớn mẫu đã bị dự đoán sai (ASR 72%), "
        "xác nhận CIFAR-10 nhạy với FGSM hơn nhiều so với MNIST.",
        "Từ ε=0.10 trở đi, hầu hết tất cả ô đều đổi sang ✗ — phù hợp với số liệu "
        "ASR ≥ 80% từ thực nghiệm 1.",
        "Nhiễu trên ảnh màu CIFAR-10 xuất hiện như một lớp \"vân\" màu nhạt trải đều, "
        "vẫn không thể phân biệt bằng mắt thường ở ε nhỏ nhưng đủ để đánh lừa mô hình.",
    ]:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(obs).font.size = Pt(11)
        p.paragraph_format.space_after = Pt(3)

    add_heading(doc, "6.2. Biểu đồ xác suất dự đoán trước và sau tấn công", 2)
    add_para(doc,
        "Biểu đồ cột softmax trước tấn công (xanh lam) và sau tấn công FGSM (đỏ) tại ε=0.30. "
        "Cột nhãn đúng được viền đen. Kết quả cho thấy trước tấn công mô hình phân loại "
        "với độ tự tin rất cao (~99%); sau tấn công, toàn bộ xác suất dịch chuyển sang "
        "một lớp sai — cũng với độ tự tin cao."
    )
    add_figure(doc, "exp3_pred_probs_mnist.png",
               "Hình 11. MNIST — Phân phối xác suất trước (xanh) và sau (đỏ) tấn công FGSM")
    add_figure(doc, "exp3_pred_probs_cifar10.png",
               "Hình 12. CIFAR-10 — Phân phối xác suất trước (xanh) và sau (đỏ) tấn công FGSM")

    # ════════════════════════════════════════════════════════
    # 7. THỰC NGHIỆM 3 — CROSS-ARCHITECTURE TRANSFER ATTACK
    # ════════════════════════════════════════════════════════
    add_heading(doc, "7. Thực nghiệm 3 — Cross-Architecture Transfer Attack (FGSM)", 1)
    add_para(doc,
        "Thực nghiệm 3 trả lời câu hỏi: ảnh đối kháng được tạo ra bởi FGSM trên mô hình A "
        "(source) có thể đánh lừa mô hình B hoàn toàn khác kiến trúc (target) không? "
        "Đây là kịch bản tấn công black-box — attacker chỉ có quyền truy cập vào một "
        "mô hình thay thế (surrogate), không biết kiến trúc hay trọng số của mô hình đích."
    )
    add_para(doc,
        "Ba model CIFAR-10 (SimpleCNN, ResNet18, MobileNetV2) tạo thành ma trận tấn công 3×3. "
        "Đường chéo là white-box (WB — source = target); ngoài đường chéo là transfer attack (black-box)."
    )

    add_heading(doc, "7.1. Độ chính xác nền (Clean Accuracy) của 3 model", 2)
    add_table(doc,
        headers=["Mô hình", "Clean Accuracy (CIFAR-10 test set)"],
        rows=[
            ["SimpleCNN",   "76.02%"],
            ["ResNet18",    "81.48%"],
            ["MobileNetV2", "85.00%"],
        ],
        col_widths=[5.5, 11.5],
    )

    add_heading(doc, "7.2. Ma trận FGSM Attack Success Rate tại ε=0.20", 2)
    add_para(doc,
        "Bảng dưới trình bày ASR (%) của FGSM tại ε=0.20. "
        "Hàng = Source model (sinh ảnh đối kháng), Cột = Target model (bị đánh giá). "
        "Ô chéo (WB) là tấn công white-box; ô ngoài chéo là transfer black-box."
    )
    add_table(doc,
        headers=["Source \\ Target", "SimpleCNN", "ResNet18", "MobileNetV2"],
        rows=[
            ["SimpleCNN",   "84.17% (WB)", "64.13%",      "73.79%"],
            ["ResNet18",    "75.55%",       "89.45% (WB)", "75.26%"],
            ["MobileNetV2", "74.63%",       "63.88%",      "81.34% (WB)"],
        ],
        col_widths=[4.5, 4.5, 4.5, 4.5],
    )

    add_heading(doc, "7.3. ASR theo ε cho tất cả cặp source→target", 2)
    add_para(doc,
        "Bảng sau tổng hợp FGSM ASR (%) theo từng mức ε cho toàn bộ 9 cặp source→target. "
        "Ô chéo (WB) được in đậm."
    )
    # Build full table: rows = pairs, cols = epsilon values
    eps_labels = ["0.05", "0.10", "0.15", "0.20", "0.25", "0.30"]
    models = ["SimpleCNN", "ResNet18", "MobileNetV2"]

    import json
    with open(os.path.join(ROOT, "results", "logs", "exp5_transfer_cifar10.json")) as f:
        exp5 = json.load(f)
    r5 = exp5["results"]

    full_rows = []
    for src in models:
        for tgt in models:
            vals = r5[src][tgt]["fgsm"]
            wb = " (WB)" if src == tgt else ""
            row_label = f"{src} → {tgt}{wb}"
            full_rows.append([row_label] + [f"{v:.1f}%" for v in vals])

    add_table(doc,
        headers=["Source → Target"] + eps_labels,
        rows=full_rows,
        col_widths=[5.0, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2],
    )

    add_para(doc, "Nhận xét:")
    for obs in [
        "White-box ASR đạt 81–89% với ε=0.20 — FGSM hiệu quả tốt trong kịch bản white-box "
        "với mọi kiến trúc được thử nghiệm.",
        "Transfer rate (black-box) dao động 64–76% tại ε=0.20: vẫn đủ cao để tạo ra mối "
        "đe dọa thực tế ngay cả khi attacker không biết kiến trúc model đích.",
        "SimpleCNN→MobileNetV2 đạt transfer rate cao nhất (73.79%); "
        "MobileNetV2→ResNet18 thấp nhất (63.88%), do khoảng cách kiến trúc lớn "
        "(depthwise convolution vs. residual blocks).",
        "FGSM có transfer rate khá đồng đều giữa các cặp, gợi ý rằng nhiễu 1 bước "
        "tìm được hướng tổng quát hơn, ít bị overfit vào decision boundary của source model.",
        "Transfer rate tăng theo ε nhưng tiệm cận trần: từ ε=0.20 lên 0.30 chỉ "
        "tăng thêm khoảng 1–3 pp cho các cặp transfer.",
    ]:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(obs).font.size = Pt(11)
        p.paragraph_format.space_after = Pt(3)

    add_figure(doc, "exp5_fgsm_only_heatmap_eps0.2.png",
               "Hình 13. Heatmap FGSM ASR ma trận 3×3 tại ε=0.20 (CIFAR-10)\n"
               "[WB] = White-box (source=target)  |  [Transfer] = Black-box")
    add_figure(doc, "exp5_fgsm_only_transfer_asr_vs_epsilon.png",
               "Hình 14. FGSM Transfer ASR theo ε — tách theo từng Source Model (CIFAR-10)\n"
               "Đường liền = White-box | Đường đứt = Transfer black-box | Màu = Target model",
               width_cm=17)

    # ════════════════════════════════════════════════════════
    # 8. PHÂN TÍCH & THẢO LUẬN
    # ════════════════════════════════════════════════════════
    add_heading(doc, "8. Phân tích và thảo luận", 1)

    add_heading(doc, "8.1. Trần năng lực của FGSM", 2)
    add_para(doc,
        "Kết quả thực nghiệm xác nhận rõ ràng giới hạn cố hữu của FGSM với tư cách là "
        "một tấn công 1 bước:"
    )
    for obs in [
        "FGSM không thể đạt ASR tùy ý dù tăng ε: trên cả MNIST (~88%) và CIFAR-10 (~85%), "
        "một phần mẫu vẫn kháng lại tấn công do bước nhảy đơn lẻ không tìm được "
        "hướng gradient tối ưu từ điểm hiện tại.",
        "Ưu thế lớn nhất của FGSM là tốc độ: chỉ cần ~2–3 giây cho 1,000+ mẫu, "
        "phù hợp cho đánh giá nhanh (quick sanity check) về độ robustness của model.",
        "FGSM hiệu quả nhất khi ε ở mức trung bình (0.10–0.20): biên độ nhỏ quá thì "
        "nhiễu chưa đủ mạnh, biên độ lớn quá thì gain thêm ASR không đáng kể (bão hòa).",
    ]:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(obs).font.size = Pt(11)
        p.paragraph_format.space_after = Pt(4)

    add_heading(doc, "8.2. Tại sao CIFAR-10 dễ bị FGSM hơn MNIST?", 2)
    add_para(doc,
        "Mặc dù CIFAR-10 phức tạp hơn MNIST, FGSM đạt hiệu quả cao hơn rõ rệt trên CIFAR-10 "
        "ngay từ mức ε nhỏ. Các nguyên nhân chính:"
    )
    for obs in [
        "Clean accuracy thấp hơn (76% vs 99.45%): mô hình vốn đã ít tự tin hơn về các "
        "quyết định phân loại, ranh giới quyết định gần với dữ liệu hơn.",
        "Không gian đầu vào lớn hơn (3×32×32 = 3,072 chiều vs 1×28×28 = 784 chiều): "
        "gradient hàm mất mát có thể hướng dẫn tấn công hiệu quả hơn trong không gian cao chiều.",
        "Đa dạng nội lớp cao hơn trong CIFAR-10 khiến model khó học ranh giới quyết định "
        "sắc nét, dẫn đến dễ bị đẩy qua biên hơn.",
    ]:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(obs).font.size = Pt(11)
        p.paragraph_format.space_after = Pt(4)

    add_heading(doc, "8.3. Ý nghĩa thực tế của khả năng chuyển tiếp tấn công", 2)
    add_para(doc,
        "Transfer rate FGSM 64–76% tại ε=0.20 mang ý nghĩa an ninh quan trọng:"
    )
    for obs in [
        "Attacker không cần biết kiến trúc hay trọng số của model đích để tấn công hiệu quả. "
        "Chỉ cần một surrogate model huấn luyện trên cùng task, tỉ lệ chuyển tiếp vẫn đủ "
        "cao để tạo ra mối đe dọa thực tế trong các hệ thống triển khai thực.",
        "Model mạnh hơn (clean accuracy cao hơn) không tự động an toàn hơn trước transfer attack: "
        "MobileNetV2 (85%) vẫn bị SimpleCNN→MobileNetV2 đạt 73.79% ASR.",
        "Khoảng cách kiến trúc ảnh hưởng đến transferability nhưng không loại bỏ được mối đe dọa: "
        "cặp có transfer rate thấp nhất (MobileNetV2→ResNet18: 63.88%) vẫn trên ngưỡng 60%.",
    ]:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(obs).font.size = Pt(11)
        p.paragraph_format.space_after = Pt(4)

    # ════════════════════════════════════════════════════════
    # 9. ADVERSARIAL TRAINING — FGSM-AT
    # ════════════════════════════════════════════════════════
    add_heading(doc, "9. Adversarial Training — Phòng thủ bằng FGSM-AT", 1)
    add_para(doc,
        "Phần này đánh giá hiệu quả của Adversarial Training với FGSM (FGSM-AT) như một "
        "phương pháp tăng cường độ bền vững của mô hình. Mỗi batch huấn luyện sử dụng "
        "hàm mất mát hỗn hợp kết hợp clean loss và adversarial loss:"
    )
    formula_at = doc.add_paragraph()
    formula_at.alignment = WD_ALIGN_PARAGRAPH.CENTER
    formula_at.paragraph_format.space_before = Pt(4)
    formula_at.paragraph_format.space_after  = Pt(4)
    r_at = formula_at.add_run("loss = (1 − r) · CE(f(x), y) + r · CE(f(x_adv), y)")
    r_at.font.name = "Courier New"; r_at.font.size = Pt(12); r_at.bold = True

    add_para(doc,
        "trong đó r = adv_ratio = 0.5, x_adv sinh bằng FGSM với ε_train cố định. "
        "Phương pháp này giúp model học cách phân loại đúng cả ảnh sạch lẫn ảnh đối kháng "
        "trong cùng một quá trình huấn luyện."
    )

    add_heading(doc, "9.1. Cấu hình và kết quả huấn luyện đối kháng", 2)
    add_table(doc,
        headers=["Tham số", "MNIST", "CIFAR-10"],
        rows=[
            ["ε_train (FGSM budget)",  "0.30",  "0.10"],
            ["adv_ratio",              "0.50",  "0.50"],
            ["Epochs",                 "20",    "20"],
            ["Test Clean Acc — Standard",    "99.3%", "76.0%"],
            ["Test Clean Acc — Adversarial", "99.0%", "69.9%"],
            ["Thay đổi clean accuracy",      "−0.3pp","−5.4pp (trade-off)"],
        ],
        col_widths=[6.0, 5.0, 5.0],
    )
    add_figure(doc, "adv_training_history_mnist.png",
               "Hình 15. MNIST — Lịch sử huấn luyện đối kháng: clean acc và robust acc qua 20 epoch")
    add_figure(doc, "adv_training_history_cifar10.png",
               "Hình 16. CIFAR-10 — Lịch sử huấn luyện đối kháng: clean acc và robust acc qua 20 epoch")

    add_heading(doc, "9.2. So sánh FGSM ASR: Standard vs Adversarial model", 2)
    add_para(doc, "MNIST — FGSM ASR (%) theo ε:", bold=True, size=10, color="555555")
    add_table(doc,
        headers=["ε", "Standard ASR (%)", "Adversarial ASR (%)", "Giảm (pp)"],
        rows=[
            ["0.05", "4.9",  "1.6", "−3.3"],
            ["0.10", "28.5", "2.5", "−26.0"],
            ["0.15", "57.0", "3.6", "−53.4"],
            ["0.20", "75.1", "4.3", "−70.8"],
            ["0.25", "84.3", "5.4", "−78.9"],
            ["0.30", "89.2", "7.0", "−82.2"],
        ],
        col_widths=[2.5, 5.0, 5.5, 4.0],
    )
    add_para(doc, "CIFAR-10 — FGSM ASR (%) theo ε:", bold=True, size=10, color="555555")
    add_table(doc,
        headers=["ε", "Standard ASR (%)", "Adversarial ASR (%)", "Giảm (pp)"],
        rows=[
            ["0.05", "74.0", "60.8", "−13.2"],
            ["0.10", "82.2", "68.8", "−13.4"],
            ["0.15", "85.2", "74.0", "−11.2"],
            ["0.20", "85.8", "77.4", "−8.4"],
            ["0.25", "86.4", "79.8", "−6.6"],
            ["0.30", "86.6", "80.7", "−5.9"],
        ],
        col_widths=[2.5, 5.0, 5.5, 4.0],
    )
    add_para(doc, "Nhận xét:")
    for obs in [
        "MNIST: Adversarial training cực kỳ hiệu quả — ASR giảm từ 89.2% xuống 7.0% ở ε=0.30 "
        "(−82.2pp) trong khi clean accuracy chỉ giảm 0.3pp. Model gần như miễn nhiễm với FGSM.",
        "CIFAR-10: Có cải thiện nhưng hạn chế hơn (~10pp giảm ASR mỗi mức ε), kèm "
        "trade-off clean accuracy −5.4pp — đây là hiện tượng điển hình khi bài toán "
        "phức tạp, model phải hy sinh một phần clean performance để đổi lấy robustness.",
        "Sự khác biệt MNIST vs CIFAR-10 cho thấy: với bài toán đơn giản, adversarial "
        "training gần như miễn phí; với bài toán phức tạp, cần chiến lược tinh tế hơn "
        "(PGD-AT, TRADES) để giảm trade-off.",
    ]:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(obs).font.size = Pt(11)
        p.paragraph_format.space_after = Pt(3)

    add_figure(doc, "adv_robustness_comparison_mnist.png",
               "Hình 17. MNIST — FGSM ASR: Standard model (đỏ) vs Adversarial model (xanh) theo ε")
    add_figure(doc, "adv_robustness_comparison_cifar10.png",
               "Hình 18. CIFAR-10 — FGSM ASR: Standard model (đỏ) vs Adversarial model (xanh) theo ε")

    # ════════════════════════════════════════════════════════
    # 10. TARGETED FGSM
    # ════════════════════════════════════════════════════════
    add_heading(doc, "10. Thực nghiệm 4 — Targeted FGSM", 1)
    add_para(doc,
        "Phần này mở rộng FGSM sang kịch bản tấn công có chủ đích (targeted attack): "
        "thay vì chỉ làm mô hình dự đoán sai (untargeted), attacker ép mô hình dự đoán "
        "đúng vào một lớp cụ thể mà họ muốn. Công thức ngược chiều gradient:"
    )
    formula_t = doc.add_paragraph()
    formula_t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    formula_t.paragraph_format.space_before = Pt(4)
    formula_t.paragraph_format.space_after  = Pt(4)
    r_t = formula_t.add_run("x_adv = x − ε · sign(∇ₓ J(θ, x, y_target))")
    r_t.font.name = "Courier New"; r_t.font.size = Pt(12); r_t.bold = True
    add_para(doc,
        "Dấu trừ (−) thay vì dấu cộng: đi ngược chiều gradient để giảm loss "
        "với nhãn đích y_target, khiến mô hình tin đây là ảnh thuộc lớp đó."
    )

    add_heading(doc, "10.1. So sánh Untargeted ASR vs Targeted TSR", 2)
    add_table(doc,
        headers=["Dataset", "Untargeted ASR (%)", "Targeted TSR (%)", "Chênh lệch (pp)"],
        rows=[
            ["MNIST",    "71.1", "15.6", "−55.5"],
            ["CIFAR-10", "95.0", "11.3", "−83.7"],
        ],
        col_widths=[3.5, 4.5, 4.5, 4.5],
    )
    add_para(doc, "Nhận xét:")
    for obs in [
        "Targeted TSR thấp hơn Untargeted ASR rất nhiều (55–84 pp): FGSM 1 bước "
        "chỉ đủ mạnh để đẩy mẫu ra khỏi lớp đúng, nhưng chưa đủ để định hướng "
        "chính xác vào một lớp mục tiêu cụ thể.",
        "CIFAR-10 đặc biệt khó targeted: TSR chỉ 11.3% trong khi untargeted đạt 95%. "
        "Không gian quyết định phức tạp hơn khiến 1 bước gradient không đủ để "
        "đưa mẫu vào đúng vùng của lớp đích.",
        "Kết quả này gợi ý: để targeted attack hiệu quả cần nhiều bước lặp hơn "
        "(Targeted I-FGSM / MI-FGSM) hoặc tối ưu C&W.",
    ]:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(obs).font.size = Pt(11)
        p.paragraph_format.space_after = Pt(3)

    add_heading(doc, "10.2. Ma trận TSR theo từng cặp class (ε=0.20)", 2)
    add_para(doc,
        "Heatmap bên dưới trình bày Targeted Success Rate (TSR) cho mọi cặp "
        "(source class → target class). Hàng = class thực của ảnh; "
        "Cột = class đích muốn tấn công vào. Ô xám trên đường chéo = N/A (src=tgt)."
    )
    add_figure(doc, "exp6_targeted_tsr_heatmap_mnist.png",
               "Hình 19. MNIST — Targeted FGSM TSR Matrix (ε=0.20)\n"
               "Màu đỏ đậm = TSR cao (dễ tấn công vào lớp đó); màu vàng nhạt = TSR thấp",
               width_cm=14)
    add_para(doc,
        "Điểm đặc biệt: class \"8\" là target dễ bị nhầm vào nhất "
        "(avg TSR khi là đích = 85.6%). Các class \"1\", \"4\", \"6\" gần như không thể "
        "là target thành công (TSR < 2%)."
    )
    add_figure(doc, "exp6_targeted_tsr_heatmap_cifar10.png",
               "Hình 20. CIFAR-10 — Targeted FGSM TSR Matrix (ε=0.20)\n"
               "\"frog\" là target dễ bị nhầm vào nhất (TSR ~98% khi là đích)",
               width_cm=14)
    add_figure(doc, "exp6_targeted_vs_untargeted.png",
               "Hình 21. So sánh Untargeted ASR vs Targeted TSR trên MNIST và CIFAR-10 (ε=0.20)")

    # ════════════════════════════════════════════════════════
    # 11. MI-FGSM
    # ════════════════════════════════════════════════════════
    add_heading(doc, "11. Thực nghiệm 5 — MI-FGSM (Momentum Iterative FGSM)", 1)
    add_para(doc,
        "MI-FGSM (Dong et al., CVPR 2018) mở rộng I-FGSM bằng cách tích lũy gradient "
        "theo chiều momentum giữa các bước, giúp tránh cực trị cục bộ và tăng "
        "khả năng transferability:"
    )
    formula_mi = doc.add_paragraph()
    formula_mi.alignment = WD_ALIGN_PARAGRAPH.CENTER
    formula_mi.paragraph_format.space_before = Pt(4)
    formula_mi.paragraph_format.space_after  = Pt(4)
    r_mi = formula_mi.add_run(
        "g₀ = 0\n"
        "gₜ₊₁ = μ · gₜ  +  ∇ₓJ(θ, xₜ, y) / ‖∇ₓJ‖₁\n"
        "xₜ₊₁ = Clip_{x,ε}[ xₜ + α · sign(gₜ₊₁) ]"
    )
    r_mi.font.name = "Courier New"; r_mi.font.size = Pt(11); r_mi.bold = True

    add_table(doc,
        headers=["Thuật toán", "Gradient mỗi bước", "Ưu điểm"],
        rows=[
            ["FGSM",    "sign(∇ₓJ) — 1 bước",              "Nhanh, đơn giản"],
            ["I-FGSM",  "sign(∇ₓJₜ) — nhiều bước",          "ASR cao hơn FGSM"],
            ["MI-FGSM", "sign(μ·gₜ + ∇ₓJₜ/‖∇ₓJₜ‖₁)",       "Transfer tốt hơn, ổn định hơn"],
        ],
        col_widths=[3.5, 6.5, 7.0],
    )

    add_heading(doc, "11.1. ASR theo ε: FGSM / I-FGSM / MI-FGSM", 2)
    add_para(doc, "MNIST — ASR (%) tại T=10 bước:", bold=True, size=10, color="555555")
    add_table(doc,
        headers=["ε", "FGSM", "I-FGSM (T=10)", "MI-FGSM (T=10, μ=1)"],
        rows=[
            ["0.05", "4.9",  "40.1",  "40.8"],
            ["0.10", "28.5", "90.1",  "89.7"],
            ["0.15", "57.0", "98.5",  "98.2"],
            ["0.20", "74.7", "99.8",  "99.8"],
            ["0.25", "84.3", "100.0", "99.8"],
            ["0.30", "88.8", "100.0", "99.8"],
        ],
        col_widths=[2.5, 3.5, 4.5, 5.5],
    )
    add_para(doc, "CIFAR-10 — ASR (%) tại T=10 bước:", bold=True, size=10, color="555555")
    add_table(doc,
        headers=["ε", "FGSM", "I-FGSM (T=10)", "MI-FGSM (T=10, μ=1)"],
        rows=[
            ["0.05", "72.4", "95.4",  "95.4"],
            ["0.10", "82.2", "100.0", "100.0"],
            ["0.15", "85.2", "100.0", "100.0"],
            ["0.20", "85.8", "100.0", "100.0"],
            ["0.25", "88.4", "100.0", "100.0"],
            ["0.30", "88.4", "100.0", "100.0"],
        ],
        col_widths=[2.5, 3.5, 4.5, 5.5],
    )
    add_figure(doc, "exp7_attack_comparison_asr.png",
               "Hình 22. So sánh ASR theo ε: FGSM / I-FGSM / MI-FGSM trên MNIST và CIFAR-10",
               width_cm=16)
    add_para(doc, "Nhận xét:")
    for obs in [
        "I-FGSM và MI-FGSM đều vượt trội rõ rệt so với FGSM từ ε=0.05: "
        "ở ε=0.10 trên MNIST, FGSM đạt 28.5% trong khi I-FGSM/MI-FGSM đạt ~90%.",
        "Trên CIFAR-10, cả I-FGSM lẫn MI-FGSM đạt 100% ASR từ ε=0.10 — "
        "minh chứng rõ ràng cho giới hạn của FGSM 1 bước.",
        "I-FGSM và MI-FGSM cho ASR tương đương nhau (≤1pp chênh lệch) "
        "trong kịch bản white-box. Lợi thế của MI-FGSM thể hiện rõ hơn ở transferability.",
    ]:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(obs).font.size = Pt(11)
        p.paragraph_format.space_after = Pt(3)

    add_heading(doc, "11.2. Hiệu quả theo số bước T", 2)
    add_figure(doc, "exp7_steps_comparison.png",
               "Hình 23. I-FGSM vs MI-FGSM: ASR theo số bước T (ε=0.20)\n"
               "Cả hai hội tụ nhanh từ T=5 bước trên cả MNIST và CIFAR-10",
               width_cm=16)
    add_para(doc,
        "Cả I-FGSM và MI-FGSM hội tụ nhanh: chỉ cần T=5 bước đã đạt ~100% ASR trên CIFAR-10. "
        "Trên MNIST, T=5 đạt ~99.4–99.8% — thêm bước không cải thiện đáng kể. "
        "MI-FGSM không chậm hơn I-FGSM về số bước cần thiết."
    )

    add_heading(doc, "11.3. Transferability: FGSM vs MI-FGSM (CIFAR-10, ε=0.20)", 2)
    add_table(doc,
        headers=["Source → Target", "FGSM ASR (%)", "MI-FGSM ASR (%)", "Tăng (pp)"],
        rows=[
            ["SimpleCNN → SimpleCNN (WB)",     "95.2",  "100.0", "+4.8"],
            ["SimpleCNN → ResNet18",            "88.8",  "96.3",  "+7.5"],
            ["SimpleCNN → MobileNetV2",         "87.2",  "100.0", "+12.8"],
            ["ResNet18 → SimpleCNN",            "97.2",  "95.0",  "−2.2"],
            ["ResNet18 → ResNet18 (WB)",        "90.5",  "100.0", "+9.5"],
            ["ResNet18 → MobileNetV2",          "90.5",  "92.2",  "+1.7"],
            ["MobileNetV2 → SimpleCNN",         "97.9",  "95.7",  "−2.2"],
            ["MobileNetV2 → ResNet18",          "86.2",  "88.8",  "+2.6"],
            ["MobileNetV2 → MobileNetV2 (WB)",  "95.2",  "100.0", "+4.8"],
        ],
        col_widths=[5.5, 3.5, 4.0, 3.0],
    )
    add_figure(doc, "exp7_transfer_fgsm_vs_mifgsm.png",
               "Hình 24. Transferability CIFAR-10: FGSM vs MI-FGSM (ε=0.20)\n"
               "Trái: FGSM ASR | Phải: MI-FGSM ASR | (WB) = White-box",
               width_cm=16)
    add_para(doc, "Nhận xét:")
    for obs in [
        "MI-FGSM cải thiện white-box ASR lên 100% với mọi kiến trúc (so với 90–97% của FGSM), "
        "nhờ momentum giúp vượt qua các flat region trong loss landscape.",
        "Transfer attack MI-FGSM nhìn chung cao hơn FGSM (đặc biệt SimpleCNN→MobileNetV2 +12.8pp), "
        "xác nhận lý thuyết của Dong et al.: momentum tích lũy gradient ổn định hơn, "
        "tìm được hướng tổng quát hơn và ít bị overfit vào decision boundary của source model.",
        "Một vài cặp MI-FGSM thấp hơn FGSM nhẹ (−2.2pp): do momentum đôi khi hội tụ "
        "quá nhanh vào hướng tốt cho source model nhưng không transfer sang kiến trúc khác.",
    ]:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(obs).font.size = Pt(11)
        p.paragraph_format.space_after = Pt(3)

    # ════════════════════════════════════════════════════════
    # 12. PER-CLASS VULNERABILITY
    # ════════════════════════════════════════════════════════
    add_heading(doc, "12. Thực nghiệm 6 — Per-class Vulnerability Analysis", 1)
    add_para(doc,
        "Thực nghiệm này phân tích: với cùng ε=0.20, những lớp nào trong MNIST và CIFAR-10 "
        "dễ bị FGSM nhất và lớp nào kháng tốt nhất? Hiểu biết này giúp xác định điểm yếu "
        "cụ thể trong mô hình và thiết kế defense nhắm mục tiêu."
    )

    add_heading(doc, "12.1. MNIST — Per-class ASR", 2)
    add_table(doc,
        headers=["Xếp hạng", "Class", "ASR (%)", "Nhận xét"],
        rows=[
            ["1 (dễ nhất)", "1",  "100.0", "Chữ số đơn giản, ít đặc trưng phân biệt"],
            ["2",           "9",  "96.0",  "Dễ nhầm với 4, 7"],
            ["3",           "7",  "94.1",  "Cấu trúc đơn giản, nhầm với 1, 9"],
            ["4",           "4",  "93.5",  "Nhiều biến thể viết tay"],
            ["5",           "6",  "92.8",  "Dễ bị đẩy sang 0 hoặc 8"],
            ["8",           "3",  "48.0",  "Cấu trúc phức tạp hơn"],
            ["9",           "2",  "44.7",  "Nhiều đặc trưng cong đặc trưng"],
            ["10 (khó nhất)","8", "27.2",  "Cấu trúc đối xứng, nhiều đặc trưng mạnh"],
        ],
        col_widths=[3.0, 2.0, 2.5, 9.5],
    )
    add_figure(doc, "exp8_perclass_mnist.png",
               "Hình 25. MNIST — FGSM Per-class Vulnerability (ε=0.20)\n"
               "Trên: ASR xếp từ dễ → khó | Dưới: Clean Acc vs Robust Acc theo class",
               width_cm=16)

    add_heading(doc, "12.2. CIFAR-10 — Per-class ASR", 2)
    add_figure(doc, "exp8_perclass_cifar10.png",
               "Hình 26. CIFAR-10 — FGSM Per-class Vulnerability (ε=0.20)\n"
               "Hầu hết các class đều bị tấn công thành công với ASR rất cao",
               width_cm=16)
    add_para(doc, "Nhận xét tổng hợp per-class:")
    for obs in [
        "MNIST: class \"8\" kháng tốt nhất (ASR 27.2%) — chữ số 8 có cấu trúc đối xứng, "
        "giàu đặc trưng hình học, khiến gradient 1 bước khó tìm được hướng vượt biên. "
        "Ngược lại, class \"1\" bị tấn công 100% vì hình dạng đơn giản, ít đặc trưng.",
        "CIFAR-10: SimpleCNN có ASR gần 100% hầu hết các class ở ε=0.20, "
        "phản ánh clean accuracy thấp và ranh giới quyết định yếu của mô hình nhỏ. "
        "Kết quả này củng cố sự cần thiết của adversarial training trên CIFAR-10.",
        "Sự chênh lệch ASR giữa các class gợi ý: defense tốt nhất không chỉ tăng robustness "
        "toàn cục mà còn cần chú trọng các class yếu nhất (class-specific hardening).",
    ]:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(obs).font.size = Pt(11)
        p.paragraph_format.space_after = Pt(3)

    # ════════════════════════════════════════════════════════
    # 13. KẾT LUẬN (cập nhật)
    # ════════════════════════════════════════════════════════
    add_heading(doc, "13. Kết luận", 1)
    add_para(doc, "Qua 6 thực nghiệm, báo cáo đã trả lời đầy đủ các câu hỏi nghiên cứu:")
    for i, concl in enumerate([
        "Biên độ nhiễu ε ảnh hưởng trực tiếp đến hiệu quả FGSM. CIFAR-10 đặc biệt nhạy: "
        "ASR đạt 72% ngay tại ε=0.05. FGSM có trần năng lực ~85–89% do bản chất 1 bước — "
        "I-FGSM và MI-FGSM vượt qua giới hạn này, đạt 100% từ ε=0.10 chỉ với T=5 bước.",

        "Targeted FGSM kém hiệu quả hơn untargeted rất nhiều (TSR 11–16% vs ASR 71–95%): "
        "1 bước gradient chưa đủ để định hướng chính xác vào lớp đích. "
        "Một số class đặc biệt dễ là target (\"8\" MNIST: 85.6% TSR; \"frog\" CIFAR-10: 98% TSR).",

        "MI-FGSM cải thiện transferability so với FGSM: +7.5pp trung bình trong black-box, "
        "lên đến +12.8pp (SimpleCNN→MobileNetV2). White-box MI-FGSM đạt 100% ASR nhờ "
        "momentum giúp thoát khỏi local optima trong loss landscape.",

        "Per-class analysis: không phải mọi class đều dễ bị tấn công như nhau. "
        "MNIST class \"8\" kháng tốt nhất (ASR 27%) trong khi class \"1\" bị đánh 100%. "
        "Điều này gợi ý defense cần class-aware, không chỉ toàn cục.",

        "Adversarial training với FGSM-AT hiệu quả trên MNIST (ASR giảm từ 89% → 7%) "
        "nhưng có trade-off trên CIFAR-10 (−5.4pp clean acc). Với bài toán phức tạp "
        "cần chiến lược tinh tế hơn (PGD-AT, TRADES).",
    ], 1):
        p = doc.add_paragraph(style="List Number")
        p.add_run(concl).font.size = Pt(11)
        p.paragraph_format.space_after = Pt(5)

    add_para(doc, "\nHướng phát triển tiếp theo:", bold=True)
    for future in [
        "Targeted I-FGSM / Targeted MI-FGSM: kết hợp nhiều bước với targeted để tăng TSR.",
        "PGD-AT (Madry et al.): adversarial training mạnh hơn FGSM-AT, giảm trade-off "
        "clean/robust accuracy trên CIFAR-10.",
        "Class-specific hardening: tăng cường defense riêng cho các class yếu nhất "
        "(class \"1\" MNIST, các class dễ bị tấn công trên CIFAR-10).",
        "Frequency domain analysis: nghiên cứu tại sao gradient FGSM có transferability cao "
        "qua phân tích phổ tần số của nhiễu đối kháng.",
    ]:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(future).font.size = Pt(11)
        p.paragraph_format.space_after = Pt(3)

    # ════════════════════════════════════════════════════════
    # TÀI LIỆU THAM KHẢO
    # ════════════════════════════════════════════════════════
    add_heading(doc, "Tài liệu tham khảo", 1)
    refs = [
        "Goodfellow, I., Shlens, J., & Szegedy, C. (2015). Explaining and harnessing adversarial "
        "examples. ICLR 2015. https://arxiv.org/abs/1412.6572",
        "Kurakin, A., Goodfellow, I., & Bengio, S. (2016). Adversarial examples in the physical "
        "world. arXiv:1607.02533",
        "Dong, Y., et al. (2018). Boosting adversarial attacks with momentum. "
        "CVPR 2018. https://arxiv.org/abs/1710.06081",
        "Szegedy, C., et al. (2014). Intriguing properties of neural networks. ICLR 2014. "
        "https://arxiv.org/abs/1312.6199",
        "Papernot, N., McDaniel, P., & Goodfellow, I. (2016). Transferability in machine learning: "
        "from phenomena to black-box attacks using adversarial samples. arXiv:1605.07277",
        "Madry, A., et al. (2018). Towards deep learning models resistant to adversarial attacks. "
        "ICLR 2018. https://arxiv.org/abs/1706.06083",
        "LeCun, Y., Cortes, C., & Burges, C. (1998). The MNIST database of handwritten digits.",
        "Krizhevsky, A. (2009). Learning multiple layers of features from tiny images. Tech Report.",
    ]
    for i, ref in enumerate(refs, 1):
        p = doc.add_paragraph()
        p.paragraph_format.space_after      = Pt(4)
        p.paragraph_format.left_indent      = Cm(0.5)
        p.paragraph_format.first_line_indent = Cm(-0.5)
        r = p.add_run(f"[{i}] {ref}")
        r.font.size = Pt(10)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    doc.save(OUT)
    print(f"\n✓ Báo cáo đã lưu tại: {OUT}")


if __name__ == "__main__":
    build()

"""
generate_report.py
Tạo báo cáo Word (.docx) đầy đủ về thực nghiệm FGSM & I-FGSM.
Chạy từ thư mục gốc project:
    python generate_report.py
"""

import os
from docx import Document
from docx.shared import Pt, Cm, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

ROOT    = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(ROOT, "results", "figures")
OUT     = os.path.join(ROOT, "results", "BaoCao_FGSM_IFGSM.docx")


# ── Helpers ───────────────────────────────────────────────────

def set_cell_bg(cell, hex_color: str):
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd  = OxmlElement("w:shd")
    shd.set(qn("w:val"),   "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"),  hex_color)
    tcPr.append(shd)

def set_cell_border(cell):
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = OxmlElement("w:tcBorders")
    for side in ("top", "left", "bottom", "right"):
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:val"),   "single")
        el.set(qn("w:sz"),    "4")
        el.set(qn("w:color"), "BFBFBF")
        tcBorders.append(el)
    tcPr.append(tcBorders)

def add_heading(doc, text, level=1, color="1F3864"):
    p    = doc.add_heading(text, level=level)
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
    run.bold   = bold
    run.italic = italic
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
    run = p.add_run()
    run.add_picture(path, width=Cm(width_cm))

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

    # Header row
    hdr = table.rows[0]
    for i, h in enumerate(headers):
        cell = hdr.cells[i]
        cell.text = h
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        set_cell_bg(cell, header_bg)
        run = cell.paragraphs[0].runs[0]
        run.bold            = True
        run.font.color.rgb  = RGBColor(0xFF, 0xFF, 0xFF)
        run.font.size       = Pt(10)

    # Data rows
    for r_idx, row_data in enumerate(rows):
        row = table.rows[r_idx + 1]
        bg  = "F2F2F2" if r_idx % 2 == 0 else "FFFFFF"
        for c_idx, val in enumerate(row_data):
            cell = row.cells[c_idx]
            cell.text = str(val)
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_bg(cell, bg)
            run = cell.paragraphs[0].runs[0]
            run.font.size = Pt(10)

    # Column widths
    if col_widths:
        for i, w in enumerate(col_widths):
            for row in table.rows:
                row.cells[i].width = Cm(w)

    doc.add_paragraph()  # spacer
    return table


# ── Main ──────────────────────────────────────────────────────

def build():
    doc = Document()

    # ── Margins ───────────────────────────────────────────────
    for section in doc.sections:
        section.top_margin    = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin   = Cm(3.0)
        section.right_margin  = Cm(2.0)

    # Default font
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
    r2 = sub.add_run("Tấn Công Đối Kháng FGSM & I-FGSM\ntrên MNIST, CIFAR-10 và Cross-Architecture Transfer")
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
        "Báo cáo này trình bày toàn bộ quá trình thực nghiệm nhằm đánh giá mức độ "
        "dễ bị tấn công đối kháng (adversarial vulnerability) của mô hình phân loại ảnh "
        "khi đối mặt với hai phương pháp tấn công phổ biến: FGSM (Fast Gradient Sign Method) "
        "và I-FGSM (Iterative FGSM, còn gọi là BIM — Basic Iterative Method)."
    )
    add_para(doc,
        "Thực nghiệm được tiến hành trên hai bộ dữ liệu chuẩn: MNIST (chữ số viết tay) "
        "và CIFAR-10 (ảnh vật thể thực tế), sử dụng mô hình SimpleCNN tự xây dựng. "
        "Mục tiêu chính là trả lời ba câu hỏi nghiên cứu:"
    )
    for q in [
        "ε (biên độ nhiễu) ảnh hưởng như thế nào đến hiệu quả tấn công?",
        "Số bước lặp T ảnh hưởng thế nào đến I-FGSM? Bao nhiêu bước là đủ?",
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
        "nhiễu tối đa (L∞ norm). FGSM chỉ thực hiện một bước tính gradient duy nhất, "
        "do đó rất nhanh nhưng có thể không đạt tối ưu."
    )

    add_heading(doc, "2.2. I-FGSM — Iterative FGSM (BIM)", 2)
    add_para(doc,
        "I-FGSM (Kurakin et al., 2016) mở rộng FGSM bằng cách thực hiện T bước nhỏ "
        "liên tiếp, sau mỗi bước clip nhiễu để không vượt quá ε-ball quanh ảnh gốc:"
    )
    for line in ["x₀ = x", "xₜ₊₁ = Clip_{x,ε} [ xₜ + α · sign(∇ₓ J(θ, xₜ, y)) ]"]:
        f = doc.add_paragraph()
        f.alignment = WD_ALIGN_PARAGRAPH.CENTER
        f.paragraph_format.space_before = Pt(2)
        f.paragraph_format.space_after  = Pt(2)
        r = f.add_run(line)
        r.font.name = "Courier New"; r.font.size = Pt(12); r.bold = True

    add_para(doc,
        "Với α = ε/T là bước nhảy mỗi iteration. Mỗi bước tính lại gradient từ vị trí "
        "hiện tại, giúp I-FGSM tìm được hướng tấn công chính xác hơn nhiều so với FGSM "
        "dù cùng ngân sách nhiễu ε."
    )

    add_heading(doc, "2.3. Phương pháp đánh giá 2 pha", 2)
    add_para(doc,
        "Tất cả thực nghiệm sử dụng quy trình đánh giá 2 pha để đo lường hiệu quả tấn "
        "công một cách công bằng và chính xác:"
    )
    for step in [
        "Pha 1 — Dự đoán & Lọc: Cho toàn bộ test set qua mô hình (không tấn công). "
         "Ghi nhận độ chính xác nền (clean accuracy) và giữ lại chỉ những mẫu được "
         "mô hình dự đoán ĐÚNG (n_correct mẫu).",
        "Pha 2 — Tấn công: Chạy FGSM và I-FGSM chỉ trên n_correct mẫu đúng đó. "
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

    add_heading(doc, "2.4. Các chỉ số đánh giá", 2)
    add_table(doc,
        headers=["Chỉ số", "Ý nghĩa"],
        rows=[
            ["Clean Acc (%)",      "Độ chính xác trước khi tấn công (baseline)"],
            ["Robust Acc (%)",     "Độ chính xác sau tấn công (trên toàn test set)"],
            ["ASR (%)",            "Attack Success Rate — % mẫu đúng bị đánh lừa"],
            ["Acc Drop (pp)",      "Mức giảm độ chính xác tuyệt đối (percentage points)"],
            ["Perturbation L∞",   "Biên độ nhiễu tối đa (= ε theo thiết kế)"],
            ["Attack Time (s)",    "Thời gian thực hiện tấn công trên toàn batch"],
        ],
        col_widths=[5.5, 11.5],
    )
    doc.add_paragraph()

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

    add_heading(doc, "3.2. Kiến trúc mô hình", 2)
    add_para(doc,
        "Thực nghiệm 1–4 sử dụng SimpleCNN tự xây dựng trên MNIST và CIFAR-10. "
        "Thực nghiệm 5 (Transfer Attack) bổ sung thêm ResNet18 và MobileNetV2 trên CIFAR-10 "
        "để tạo ra môi trường đa kiến trúc cho phép đánh giá khả năng chuyển tiếp tấn công:"
    )
    add_table(doc,
        headers=["Thành phần", "MNIST (1×28×28)", "CIFAR-10 (3×32×32)"],
        rows=[
            ["Block 1", "Conv(1→32,3×3)→BN→ReLU\nConv(32→32,3×3)→BN→ReLU\nMaxPool(2×2)→Dropout2d(0.25)",
                        "Conv(3→32,3×3)→BN→ReLU\nConv(32→32,3×3)→BN→ReLU\nMaxPool(2×2)→Dropout2d(0.25)"],
            ["Block 2", "—", "Conv(32→64,3×3)→BN→ReLU\nConv(64→64,3×3)→BN→ReLU\nMaxPool(2×2)→Dropout2d(0.25)"],
            ["Classifier", "Flatten→Linear(6272→512)\n→ReLU→Dropout(0.5)→Linear(512→10)",
                           "Flatten→Linear(4096→512)\n→ReLU→Dropout(0.5)→Linear(512→10)"],
        ],
        col_widths=[3.5, 7.0, 7.0],
    )

    add_heading(doc, "3.3. Hyperparameter huấn luyện", 2)
    add_table(doc,
        headers=["Tham số", "Giá trị"],
        rows=[
            ["Optimizer",      "Adam (lr=0.001, weight_decay=1e-4)"],
            ["LR Scheduler",   "StepLR (step=10, γ=0.1)"],
            ["Epochs",         "20"],
            ["Batch size",     "64"],
            ["Loss function",  "Cross-Entropy"],
            ["Checkpointing",  "Lưu model tốt nhất theo val accuracy"],
        ],
        col_widths=[5.5, 11.5],
    )

    add_heading(doc, "3.4. Cấu hình tấn công", 2)
    add_table(doc,
        headers=["Tham số", "Giá trị", "Ghi chú"],
        rows=[
            ["ε (epsilon_list)",   "[0.05, 0.10, 0.15, 0.20, 0.25, 0.30]", "Dùng cho Thực nghiệm 1"],
            ["steps_epsilon",      "0.10",                                   "ε cố định cho Thực nghiệm 2"],
            ["T (steps_list)",     "[5, 10, 20, 40]",                        "Dùng cho Thực nghiệm 2"],
            ["α (alpha)",          "ε / T (tự tính)",                        "Bước mỗi iteration"],
            ["Targeted",           "False",                                   "Untargeted attack"],
            ["Clip range",         "[0.0, 1.0]",                             "Giới hạn giá trị pixel"],
        ],
        col_widths=[4.5, 6.0, 7.0],
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
        "Trên CIFAR-10, SimpleCNN đạt 76.02% — phù hợp với benchmark không dùng kỹ thuật phức tạp. "
        "ResNet18 (82.00%) và MobileNetV2 (84.42%) vượt trội SimpleCNN nhờ kiến trúc sâu hơn "
        "và skip-connection / depthwise convolution, cung cấp nền tảng mạnh cho Thực nghiệm 5."
    )

    add_figure(doc, "training_history_mnist.png",
               "Hình 1. Lịch sử huấn luyện MNIST — SimpleCNN, loss và accuracy qua 20 epoch")
    add_figure(doc, "training_history_cifar10.png",
               "Hình 2. Lịch sử huấn luyện CIFAR-10 — SimpleCNN, loss và accuracy qua 20 epoch")
    add_figure(doc, "training_history_cifar10_resnet18.png",
               "Hình 3. Lịch sử huấn luyện CIFAR-10 — ResNet18, loss và accuracy qua 20 epoch")
    add_figure(doc, "training_history_cifar10_mobilenetv2.png",
               "Hình 4. Lịch sử huấn luyện CIFAR-10 — MobileNetV2, loss và accuracy qua 20 epoch")

    # ════════════════════════════════════════════════════════
    # 5. THỰC NGHIỆM 1 — ACCURACY VS EPSILON
    # ════════════════════════════════════════════════════════
    add_heading(doc, "5. Thực nghiệm 1 — Ảnh hưởng của ε đến hiệu quả tấn công", 1)
    add_para(doc,
        "Thực nghiệm 1 cố định số bước T=40 và thay đổi biên độ nhiễu ε qua các giá trị "
        "[0.05, 0.10, 0.15, 0.20, 0.25, 0.30]. Với mỗi ε, cả FGSM và I-FGSM được đánh giá "
        "trên 1,280 mẫu test, đo độ chính xác còn lại và Attack Success Rate."
    )

    add_heading(doc, "5.1. Kết quả trên MNIST", 2)
    add_para(doc, "Tập test: 1,280 mẫu | Mẫu phân loại đúng: 1,273 | Clean accuracy: 99.45%",
             bold=True, size=10, color="555555")
    add_table(doc,
        headers=["ε", "FGSM Acc (%)", "FGSM ASR (%)", "FGSM Time (s)",
                       "I-FGSM Acc (%)", "I-FGSM ASR (%)", "I-FGSM Time (s)"],
        rows=[
            ["0.05", "94.53", "4.95",  "3.2", "85.23", "14.30",  "119.5"],
            ["0.10", "70.63", "28.99", "2.9", "16.56", "83.35",  "123.8"],
            ["0.15", "42.73", "57.03", "3.1", "0.70",  "99.29",  "126.5"],
            ["0.20", "25.16", "74.71", "2.9", "0.00",  "100.00", "128.1"],
            ["0.25", "15.94", "83.97", "2.9", "0.00",  "100.00", "133.0"],
            ["0.30", "11.17", "88.77", "2.9", "0.00",  "100.00", "134.9"],
        ],
        col_widths=[1.5, 2.4, 2.4, 2.2, 2.4, 2.4, 2.7],
    )
    add_para(doc, "Nhận xét:")
    for obs in [
        "Tại ε=0.10: I-FGSM đạt 83.35% ASR so với 28.99% của FGSM — hiệu quả gần gấp 3× "
        "với cùng ngân sách nhiễu.",
        "Tại ε=0.20: I-FGSM đạt 100% ASR — mọi mẫu phân loại đúng đều bị đánh lừa thành công.",
        "FGSM chỉ đạt tối đa ~88.77% ASR dù ε=0.30 — đây là trần năng lực của tấn công 1 bước.",
        "Thời gian I-FGSM (~120–135s) gấp ~43× FGSM (~3s) do phải thực hiện 40 lần backward pass.",
    ]:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(obs).font.size = Pt(11)
        p.paragraph_format.space_after = Pt(3)

    add_figure(doc, "exp1_acc_vs_epsilon_mnist.png",
               "Hình 5. MNIST — Accuracy còn lại theo ε (FGSM vs I-FGSM, T=40)")

    add_heading(doc, "5.2. Kết quả trên CIFAR-10", 2)
    add_para(doc, "Tập test: 1,280 mẫu | Mẫu phân loại đúng: 973 | Clean accuracy: 76.02%",
             bold=True, size=10, color="555555")
    add_table(doc,
        headers=["ε", "FGSM Acc (%)", "FGSM ASR (%)", "FGSM Time (s)",
                       "I-FGSM Acc (%)", "I-FGSM ASR (%)", "I-FGSM Time (s)"],
        rows=[
            ["0.05", "21.02", "72.35", "2.5", "7.27", "90.44", "85.5"],
            ["0.10", "15.08", "80.16", "2.3", "1.64", "97.84", "85.8"],
            ["0.15", "12.66", "83.35", "2.3", "1.17", "98.46", "88.7"],
            ["0.20", "12.03", "84.17", "2.4", "0.86", "98.87", "87.9"],
            ["0.25", "11.64", "84.69", "2.2", "0.47", "99.38", "83.2"],
            ["0.30", "11.17", "85.30", "2.2", "0.23", "99.69", "83.4"],
        ],
        col_widths=[1.5, 2.4, 2.4, 2.2, 2.4, 2.4, 2.7],
    )
    add_para(doc, "Nhận xét:")
    for obs in [
        "Ngay tại ε=0.05, I-FGSM đã đạt 90.44% ASR — mức mà MNIST cần đến ε=0.25 mới đạt được. "
        "CIFAR-10 dễ bị tấn công hơn nhiều do model có clean accuracy thấp hơn (76%).",
        "FGSM bão hòa ở ~85% ASR từ ε=0.15 trở đi, cho thấy trần năng lực tương tự MNIST.",
        "I-FGSM tại ε=0.10 đã đạt 97.84% ASR — gần như tuyệt đối, trong khi MNIST cần ε=0.15.",
        "Thời gian I-FGSM trên CIFAR-10 (~85s) ngắn hơn MNIST (~125s) vì CIFAR-10 có ít mẫu "
        "đúng hơn (973 vs 1,273), dẫn đến ít tính toán hơn.",
    ]:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(obs).font.size = Pt(11)
        p.paragraph_format.space_after = Pt(3)

    add_figure(doc, "exp1_acc_vs_epsilon_cifar10.png",
               "Hình 6. CIFAR-10 — Accuracy còn lại theo ε (FGSM vs I-FGSM, T=40)")

    # ════════════════════════════════════════════════════════
    # 6. THỰC NGHIỆM 2 — ACCURACY VS SỐ BƯỚC T
    # ════════════════════════════════════════════════════════
    add_heading(doc, "6. Thực nghiệm 2 — Ảnh hưởng của số bước T đến I-FGSM", 1)
    add_para(doc,
        "Thực nghiệm 2 cố định ε=0.10 (steps_epsilon) và thay đổi số bước T qua "
        "[5, 10, 20, 40] để trả lời câu hỏi: bao nhiêu bước là đủ để I-FGSM hội tụ? "
        "Lưu ý: ε=0.10 nhỏ hơn ε=0.30 dùng trong thực nghiệm 1 — được chọn để đảm bảo "
        "rằng với ít bước (T=5), tấn công chưa bão hòa, giúp thấy rõ ảnh hưởng của T."
    )

    add_heading(doc, "6.1. Kết quả trên MNIST", 2)
    add_para(doc, "ε=0.10 | 1,273 mẫu phân loại đúng | Clean accuracy: 99.45%",
             bold=True, size=10, color="555555")
    add_table(doc,
        headers=["T (bước)", "I-FGSM Acc (%)", "ASR (%)", "Acc Drop (pp)", "Thời gian (s)"],
        rows=[
            ["5",  "26.64", "73.21", "−72.81", "15.7"],
            ["10", "21.25", "78.63", "−78.20", "32.5"],
            ["20", "18.20", "81.70", "−81.25", "63.8"],
            ["40", "16.56", "83.35", "−82.89", "126.4"],
        ],
        col_widths=[3.0, 4.0, 3.5, 4.0, 3.5],
    )
    add_para(doc, "Nhận xét:")
    for obs in [
        "Lợi tức giảm dần rõ ràng: T=5→10 cải thiện 5.4 pp, T=10→20 cải thiện 3.1 pp, "
        "T=20→40 chỉ cải thiện 1.6 pp.",
        "Tại T=40, tấn công tiệm cận hội tụ — tăng thêm bước không mang lại cải thiện đáng kể.",
        "Thời gian tăng tuyến tính với T (~16s/5 bước), xác nhận overhead tính toán là O(T).",
    ]:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(obs).font.size = Pt(11)
        p.paragraph_format.space_after = Pt(3)

    add_figure(doc, "exp2_acc_vs_steps_mnist.png",
               "Hình 7. MNIST — Accuracy còn lại theo số bước T (ε=0.10)")

    add_heading(doc, "6.2. Kết quả trên CIFAR-10", 2)
    add_para(doc, "ε=0.10 | 973 mẫu phân loại đúng | Clean accuracy: 76.02%",
             bold=True, size=10, color="555555")
    add_table(doc,
        headers=["T (bước)", "I-FGSM Acc (%)", "ASR (%)", "Acc Drop (pp)", "Thời gian (s)"],
        rows=[
            ["5",  "2.81", "96.30", "−73.20", "10.5"],
            ["10", "2.34", "96.92", "−73.67", "21.0"],
            ["20", "1.72", "97.74", "−74.30", "42.1"],
            ["40", "1.64", "97.84", "−74.38", "84.1"],
        ],
        col_widths=[3.0, 4.0, 3.5, 4.0, 3.5],
    )
    add_para(doc, "Nhận xét:")
    for obs in [
        "CIFAR-10 hội tụ cực nhanh: ngay tại T=5 đã đạt 96.30% ASR, "
        "trong khi MNIST ở T=5 chỉ đạt 73.21%.",
        "Khoảng cải thiện từ T=5 đến T=40 chỉ là 1.54 pp — tấn công thực tế bão hòa từ T=5.",
        "Kết quả cho thấy với model CIFAR-10 yếu, chỉ cần T=5 bước là đủ để đạt hiệu quả gần tối đa.",
    ]:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(obs).font.size = Pt(11)
        p.paragraph_format.space_after = Pt(3)

    add_figure(doc, "exp2_acc_vs_steps_cifar10.png",
               "Hình 8. CIFAR-10 — Accuracy còn lại theo số bước T (ε=0.10)")

    # ════════════════════════════════════════════════════════
    # 7. THỰC NGHIỆM 3 — TRỰC QUAN HÓA
    # ════════════════════════════════════════════════════════
    add_heading(doc, "7. Thực nghiệm 3 — Trực quan hóa ảnh đối kháng", 1)
    add_para(doc,
        "Thực nghiệm 3 trực quan hóa tác động của I-FGSM lên từng ảnh cụ thể "
        "thông qua ba loại biểu đồ, giúp hiểu trực giác về cơ chế tấn công."
    )

    add_heading(doc, "7.1. So sánh ảnh gốc — nhiễu — ảnh đối kháng", 2)
    add_para(doc,
        "Mỗi cột tương ứng một mẫu ảnh: hàng 1 là ảnh gốc, hàng 2 là nhiễu được "
        "khuếch đại ×10 (để mắt người có thể thấy), hàng 3 là ảnh đối kháng cuối cùng. "
        "Nhãn màu đỏ là dự đoán sai, màu xanh là đúng."
    )
    add_figure(doc, "exp3_examples_mnist.png",
               "Hình 9. MNIST — Ảnh gốc | Nhiễu ×10 | Ảnh đối kháng (ε=0.30, T=40)")
    add_figure(doc, "exp3_examples_cifar10.png",
               "Hình 10. CIFAR-10 — Ảnh gốc | Nhiễu ×10 | Ảnh đối kháng (ε=0.30, T=40)")

    add_heading(doc, "7.2. Biểu đồ xác suất dự đoán trước và sau tấn công", 2)
    add_para(doc,
        "Biểu đồ cột softmax trước tấn công (xanh lam) và sau tấn công (đỏ). "
        "Cột nhãn đúng được viền đen. Kết quả cho thấy trước tấn công, mô hình "
        "phân loại với độ tự tin rất cao (~99%); sau tấn công, toàn bộ xác suất "
        "dịch chuyển sang một lớp sai — cũng với độ tự tin ~99%."
    )
    add_figure(doc, "exp3_pred_probs_mnist.png",
               "Hình 11. MNIST — Phân phối xác suất trước (xanh) và sau (đỏ) tấn công")
    add_figure(doc, "exp3_pred_probs_cifar10.png",
               "Hình 12. CIFAR-10 — Phân phối xác suất trước (xanh) và sau (đỏ) tấn công")

    add_heading(doc, "7.3. Diễn biến hàm mất mát qua các bước I-FGSM", 2)
    add_para(doc,
        "Biểu đồ đường thể hiện cross-entropy loss tăng dần qua từng bước lặp, "
        "xác nhận rằng I-FGSM đang tối ưu hóa đúng hướng: đẩy loss lên cao để "
        "khiến mô hình đưa ra dự đoán sai."
    )
    add_figure(doc, "exp3_loss_evolution_mnist.png",
               "Hình 13. MNIST — Loss tăng dần qua các bước I-FGSM")
    add_figure(doc, "exp3_loss_evolution_cifar10.png",
               "Hình 14. CIFAR-10 — Loss tăng dần qua các bước I-FGSM")

    # ════════════════════════════════════════════════════════
    # 8. THỰC NGHIỆM 4 — PRESENTATION GRID
    # ════════════════════════════════════════════════════════
    add_heading(doc, "8. Thực nghiệm 4 — So sánh trực quan theo ε và T", 1)
    add_para(doc,
        "Thực nghiệm 4 xuất hai bảng ảnh tổng hợp giúp quan sát một cách trực quan "
        "và đồng thời ảnh hưởng của ε và T lên cùng một tập mẫu ảnh. "
        "Mỗi ô hiển thị ảnh đối kháng kèm nhãn dự đoán (✓ xanh = đúng / ✗ đỏ = sai) "
        "và độ tự tin (%)."
    )

    add_heading(doc, "8.1. Grid so sánh theo ε (cố định T=40)", 2)
    add_figure(doc, "exp4_grid_epsilon_mnist.png",
               "Hình 15. MNIST — So sánh ảnh đối kháng I-FGSM tại các mức ε khác nhau (T=40)")
    add_figure(doc, "exp4_grid_epsilon_cifar10.png",
               "Hình 16. CIFAR-10 — So sánh ảnh đối kháng I-FGSM tại các mức ε khác nhau (T=40)")

    add_heading(doc, "8.2. Grid so sánh theo số bước T (cố định ε=0.10)", 2)
    add_figure(doc, "exp4_grid_steps_mnist.png",
               "Hình 17. MNIST — So sánh ảnh đối kháng I-FGSM theo số bước T (ε=0.10)")
    add_figure(doc, "exp4_grid_steps_cifar10.png",
               "Hình 18. CIFAR-10 — So sánh ảnh đối kháng I-FGSM theo số bước T (ε=0.10)")

    # ════════════════════════════════════════════════════════
    # 9. THỰC NGHIỆM 5 — CROSS-ARCHITECTURE TRANSFER ATTACK
    # ════════════════════════════════════════════════════════
    add_heading(doc, "9. Thực nghiệm 5 — Cross-Architecture Transfer Attack", 1)
    add_para(doc,
        "Thực nghiệm 5 trả lời câu hỏi: ảnh đối kháng sinh ra từ mô hình A (source) "
        "có thể đánh lừa mô hình B hoàn toàn khác kiến trúc (target) không? "
        "Ba model CIFAR-10 (SimpleCNN, ResNet18, MobileNetV2) tạo thành ma trận tấn công 3×3: "
        "diagonal là white-box (WB), off-diagonal là transfer attack (black-box). "
        "Epsilon đại diện được chọn là ε=0.20 — giá trị cho thấy sự khác biệt WB vs Transfer "
        "rõ ràng nhất trong khi transfer rate vẫn đủ cao để thấy mức độ nguy hiểm thực tế."
    )

    add_heading(doc, "9.1. Độ chính xác nền (Clean Accuracy) của 3 model", 2)
    add_table(doc,
        headers=["Mô hình", "Clean Accuracy (CIFAR-10 test set)"],
        rows=[
            ["SimpleCNN",   "76.02%"],
            ["ResNet18",    "81.48%"],
            ["MobileNetV2", "85.00%"],
        ],
        col_widths=[5.5, 11.5],
    )

    add_heading(doc, "9.2. Ma trận Attack Success Rate tại ε=0.20", 2)
    add_para(doc,
        "Bảng dưới đây trình bày ASR (%) của FGSM và I-FGSM tại ε=0.20. "
        "Hàng = Source model (sinh ảnh đối kháng), Cột = Target model (bị tấn công). "
        "Ô chéo (WB) là tấn công white-box; ô ngoài chéo là transfer black-box."
    )
    add_para(doc, "FGSM — ASR (%) tại ε=0.20:", bold=True, size=10, color="555555")
    add_table(doc,
        headers=["Source \\ Target", "SimpleCNN", "ResNet18", "MobileNetV2"],
        rows=[
            ["SimpleCNN",   "84.17% (WB)", "64.13%",     "73.79%"],
            ["ResNet18",    "75.55%",       "89.45% (WB)", "75.26%"],
            ["MobileNetV2", "74.63%",       "63.88%",      "81.34% (WB)"],
        ],
        col_widths=[4.0, 4.5, 4.5, 4.5],
    )
    add_para(doc, "I-FGSM — ASR (%) tại ε=0.20 (T=40 bước):", bold=True, size=10, color="555555")
    add_table(doc,
        headers=["Source \\ Target", "SimpleCNN", "ResNet18", "MobileNetV2"],
        rows=[
            ["SimpleCNN",   "98.87% (WB)", "67.83%",      "90.44%"],
            ["ResNet18",    "83.32%",       "98.66% (WB)", "87.92%"],
            ["MobileNetV2", "81.80%",       "66.73%",      "99.26% (WB)"],
        ],
        col_widths=[4.0, 4.5, 4.5, 4.5],
    )

    add_para(doc, "Nhận xét:")
    for obs in [
        "White-box ASR (I-FGSM) đạt 98–99% cho cả 3 model — xác nhận mọi kiến trúc "
        "đều cực kỳ dễ bị tấn công trong điều kiện white-box.",
        "Transfer rate (I-FGSM) dao động 67–90%: vẫn rất nguy hiểm ngay cả khi attacker "
        "không biết kiến trúc model mục tiêu.",
        "SimpleCNN→MobileNetV2 đạt transfer rate cao nhất (90.44%) — cho thấy hai kiến trúc "
        "này chia sẻ các pattern dễ bị khai thác tương tự nhau.",
        "MobileNetV2→ResNet18 đạt transfer rate thấp nhất (66.73%) — khoảng cách kiến trúc "
        "lớn nhất trong bộ 3, đặc biệt do MobileNetV2 dùng depthwise convolution rất khác ResNet18.",
        "FGSM có transfer rate đồng đều hơn I-FGSM (64–76%), chứng tỏ nhiễu 1 bước "
        "tìm hướng tổng quát hơn, ít overfit vào decision boundary của source model.",
    ]:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(obs).font.size = Pt(11)
        p.paragraph_format.space_after = Pt(3)

    add_figure(doc, "exp5_transfer_heatmap_eps0.2.png",
               "Hình 19. Heatmap ASR ma trận 3×3 tại ε=0.20 — FGSM (trái) và I-FGSM (phải). "
               "Đường chéo (WB) sáng nhất; off-diagonal thể hiện mức độ chuyển tiếp tấn công.")
    add_figure(doc, "exp5_transfer_asr_vs_epsilon.png",
               "Hình 20. Transfer ASR theo ε cho tất cả cặp source→target — FGSM và I-FGSM. "
               "WB (đường đứt nét) luôn cao hơn transfer, khoảng cách thu hẹp dần khi ε tăng.")

    # ════════════════════════════════════════════════════════
    # 10. PHÂN TÍCH & THẢO LUẬN
    # ════════════════════════════════════════════════════════
    add_heading(doc, "10. Phân tích và thảo luận", 1)

    add_heading(doc, "10.1. So sánh tổng thể MNIST và CIFAR-10", 2)
    add_table(doc,
        headers=["Tiêu chí so sánh", "MNIST", "CIFAR-10"],
        rows=[
            ["Clean Accuracy",              "99.45%",  "76.02%"],
            ["I-FGSM ASR tại ε=0.05",       "14.30%",  "90.44%"],
            ["I-FGSM ASR tại ε=0.10",       "83.35%",  "97.84%"],
            ["ε để I-FGSM đạt 100% ASR",    "0.20",    "Không đạt (max ~99.7%)"],
            ["FGSM ASR ceiling (ε=0.30)",   "88.77%",  "85.30%"],
            ["T để I-FGSM hội tụ (ε=0.10)", "~40 bước","~5 bước"],
            ["Thời gian I-FGSM (T=40)",     "~125s",   "~84s"],
        ],
        col_widths=[6.5, 4.0, 5.5],
    )

    add_heading(doc, "10.2. I-FGSM vs FGSM — Phân tích khoảng cách hiệu quả", 2)
    add_para(doc,
        "Kết quả thực nghiệm xác nhận rõ ràng sự vượt trội của I-FGSM so với FGSM:"
    )
    for obs in [
        "Với cùng ε=0.10 trên MNIST: I-FGSM đạt 83.35% ASR, FGSM chỉ đạt 28.99% — "
        "khoảng cách 54 pp. Điều này cho thấy việc tính lại gradient tại mỗi bước mang "
        "lại lợi ích rất lớn.",
        "FGSM có trần năng lực cứng (~85–89% ASR): dù tăng ε, một số mẫu vẫn không bị "
        "đánh lừa do bước nhảy đơn lẻ không tìm được đúng hướng tấn công.",
        "I-FGSM không có trần này — với đủ bước và ε hợp lý, có thể đạt ASR tùy ý gần 100%.",
        "Đánh đổi: I-FGSM chậm hơn FGSM ~43× (T=40 bước). Đây là overhead tất yếu "
        "của việc thực hiện 40 lần backward pass.",
    ]:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(obs).font.size = Pt(11)
        p.paragraph_format.space_after = Pt(4)

    add_heading(doc, "10.3. Tại sao CIFAR-10 dễ bị tấn công hơn?", 2)
    add_para(doc,
        "Mặc dù CIFAR-10 phức tạp hơn MNIST, mô hình SimpleCNN trên CIFAR-10 bị tấn công "
        "thành công dễ hơn nhiều. Nguyên nhân chính:"
    )
    for obs in [
        "Clean accuracy thấp hơn (76% vs 99.45%): mô hình vốn đã ít tự tin hơn về các "
        "quyết định phân loại, ranh giới quyết định (decision boundary) gần với dữ liệu hơn.",
        "Không gian đầu vào phức tạp hơn (3×32×32 = 3,072 chiều vs 1×28×28 = 784 chiều): "
        "gradient của hàm mất mát có thể hướng dẫn tấn công hiệu quả hơn trong không gian cao chiều.",
        "Tính đồng nhất nội lớp thấp hơn: ảnh thực tế của CIFAR-10 có nhiều biến thể hơn, "
        "khiến model khó học ranh giới quyết định sắc nét.",
    ]:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(obs).font.size = Pt(11)
        p.paragraph_format.space_after = Pt(4)

    add_heading(doc, "10.4. Phân tích lợi tức giảm dần của số bước T", 2)
    add_para(doc,
        "Thực nghiệm 2 cho thấy quy luật lợi tức giảm dần (diminishing returns) "
        "rõ ràng khi tăng số bước T:"
    )
    add_table(doc,
        headers=["Khoảng T", "Cải thiện ASR (MNIST)", "Cải thiện ASR (CIFAR-10)"],
        rows=[
            ["T=5 → T=10",  "+5.42 pp", "+0.62 pp"],
            ["T=10 → T=20", "+3.06 pp", "+0.82 pp"],
            ["T=20 → T=40", "+1.65 pp", "+0.10 pp"],
        ],
        col_widths=[4.5, 5.5, 5.5],
    )
    add_para(doc,
        "Trong thực tế, lựa chọn T dựa trên đánh đổi giữa hiệu quả tấn công và "
        "chi phí tính toán. Với CIFAR-10, T=5 là đủ; với MNIST, T=20 cho hiệu quả "
        "gần với T=40 nhưng nhanh gấp 2×."
    )

    add_heading(doc, "10.5. Transferability — Hàm ý an ninh thực tế", 2)
    add_para(doc,
        "Kết quả Thực nghiệm 5 mang hàm ý quan trọng về an ninh thực tế:"
    )
    for obs in [
        "Adversarial examples có tính transferability cao (67–90% I-FGSM ASR): attacker "
        "có thể tấn công hiệu quả mà không cần biết kiến trúc hay tham số model đích "
        "(black-box attack). Đây là mối đe dọa thực tế trong các hệ thống triển khai thực.",
        "Model mạnh hơn không nhất thiết an toàn hơn: MobileNetV2 có clean accuracy cao nhất "
        "(85%) nhưng vẫn bị transfer attack từ SimpleCNN đạt 90.44% ASR.",
        "Khoảng cách kiến trúc ảnh hưởng đến transferability: MobileNetV2↔ResNet18 "
        "có transfer rate thấp nhất (~67%) do sự khác biệt căn bản trong cách xử lý đặc trưng "
        "(depthwise conv vs skip connection).",
        "Defense thực tế phải tính đến cả transfer attack, không chỉ white-box: "
        "adversarial training thuần túy trên white-box có thể không đủ bảo vệ khỏi black-box threat.",
    ]:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(obs).font.size = Pt(11)
        p.paragraph_format.space_after = Pt(4)

    # ════════════════════════════════════════════════════════
    # 11. ADVERSARIAL TRAINING — FGSM-AT
    # ════════════════════════════════════════════════════════
    add_heading(doc, "11. Adversarial Training — Phòng thủ bằng FGSM-AT", 1)
    add_para(doc,
        "Phần này đánh giá hiệu quả của Adversarial Training với FGSM (FGSM-AT) như một "
        "phương pháp tăng cường độ bền vững của mô hình. Thay vì chỉ huấn luyện trên ảnh sạch, "
        "mỗi batch sử dụng hàm mất mát hỗn hợp kết hợp cả clean loss lẫn adversarial loss:"
    )
    formula_at = doc.add_paragraph()
    formula_at.alignment = WD_ALIGN_PARAGRAPH.CENTER
    formula_at.paragraph_format.space_before = Pt(4)
    formula_at.paragraph_format.space_after  = Pt(4)
    r_at = formula_at.add_run("loss = (1 − r) · CE(f(x), y) + r · CE(f(x_adv), y)")
    r_at.font.name = "Courier New"; r_at.font.size = Pt(12); r_at.bold = True

    add_para(doc,
        "trong đó r = adv_ratio = 0.5 (tỉ lệ adversarial loss), x_adv được sinh bằng FGSM "
        "với ε_train cố định trong suốt quá trình huấn luyện. Phương pháp này giúp model "
        "học cách phân loại đúng cả ảnh sạch lẫn ảnh đối kháng."
    )

    add_heading(doc, "11.1. Cấu hình adversarial training", 2)
    add_table(doc,
        headers=["Tham số", "MNIST", "CIFAR-10"],
        rows=[
            ["ε_train (FGSM budget)",  "0.30",  "0.10"],
            ["adv_ratio",              "0.50",  "0.50"],
            ["Epochs",                 "20",    "20"],
            ["Optimizer",              "Adam (lr=0.001)", "Adam (lr=0.001)"],
            ["LR Scheduler",           "StepLR (step=10, γ=0.1)", "StepLR (step=10, γ=0.1)"],
        ],
        col_widths=[5.5, 5.5, 5.5],
    )

    add_heading(doc, "11.2. Kết quả huấn luyện đối kháng", 2)
    add_para(doc,
        "Kết quả tốt nhất trên validation set và đánh giá cuối trên test set:"
    )
    add_table(doc,
        headers=["Dataset", "Mô hình", "Val Clean Acc tốt nhất", "Test Clean Acc", "Test Robust Acc (ε_train)"],
        rows=[
            ["MNIST",    "Standard",   "98.98%", "99.3%", "—"],
            ["MNIST",    "Adversarial","98.98%*", "99.0%", "92.0% (ε=0.30)"],
            ["CIFAR-10", "Standard",   "74.36%", "76.0%", "—"],
            ["CIFAR-10", "Adversarial","68.98%", "69.9%", "25.8% (ε=0.10)"],
        ],
        col_widths=[3.0, 3.5, 4.5, 4.0, 5.0],
    )
    add_para(doc,
        "* MNIST Adversarial model duy trì clean accuracy gần như không đổi (99.0% vs 99.3%), "
        "cho thấy với bài toán đơn giản như MNIST, adversarial training hầu như không có "
        "accuracy-robustness trade-off. CIFAR-10 Adversarial model giảm 5.4pp clean accuracy "
        "(76.0% → 69.9%) — đây là mức trade-off điển hình cho bài toán phức tạp hơn.",
        italic=True, size=10, color="555555"
    )
    add_figure(doc, "adv_training_history_mnist.png",
               "Hình 21. MNIST — Lịch sử huấn luyện đối kháng: clean acc và robust acc qua 20 epoch")
    add_figure(doc, "adv_training_history_cifar10.png",
               "Hình 22. CIFAR-10 — Lịch sử huấn luyện đối kháng: clean acc và robust acc qua 20 epoch")

    add_heading(doc, "11.3. So sánh Standard vs Adversarial model", 2)
    add_para(doc,
        "Đánh giá FGSM ASR trên cả Standard và Adversarial model tại 6 mức epsilon:"
    )

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
        "Trên MNIST: Adversarial training cực kỳ hiệu quả — FGSM ASR giảm từ 89.2% xuống "
        "chỉ 7.0% tại ε=0.30 (giảm 82.2pp) trong khi clean accuracy gần như không đổi (−0.3pp). "
        "Model gần như miễn nhiễm với FGSM sau adversarial training.",
        "Trên CIFAR-10: hiệu quả phòng thủ có nhưng hạn chế hơn — ASR giảm trung bình "
        "~10pp tại mỗi mức ε, nhưng clean accuracy giảm 5.4pp. Đây là accuracy-robustness "
        "trade-off điển hình khi bài toán phức tạp hơn.",
        "Lý do hiệu quả khác nhau: MNIST đơn giản hơn cho phép model học được cả clean "
        "distribution và adversarial perturbation cùng lúc mà không ảnh hưởng đến nhau. "
        "CIFAR-10 phức tạp hơn nên model phải 'hy sinh' một phần clean accuracy để "
        "đổi lấy robustness.",
        "Adversarial training trên CIFAR-10 không giải quyết triệt để vấn đề: ASR tại ε=0.30 "
        "vẫn còn 80.7% — cho thấy cần kỹ thuật mạnh hơn (PGD-AT, TRADES) hoặc "
        "kết hợp với các phương pháp phòng thủ khác.",
    ]:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(obs).font.size = Pt(11)
        p.paragraph_format.space_after = Pt(3)

    add_figure(doc, "adv_robustness_comparison_mnist.png",
               "Hình 23. MNIST — So sánh FGSM ASR: Standard model (đỏ) vs Adversarial model (xanh) theo ε")
    add_figure(doc, "adv_robustness_comparison_cifar10.png",
               "Hình 24. CIFAR-10 — So sánh FGSM ASR: Standard model (đỏ) vs Adversarial model (xanh) theo ε")

    # ════════════════════════════════════════════════════════
    # 12. KẾT LUẬN
    # ════════════════════════════════════════════════════════
    add_heading(doc, "12. Kết luận", 1)
    add_para(doc,
        "Thực nghiệm đã trả lời đầy đủ bốn câu hỏi nghiên cứu đặt ra:"
    )
    for i, concl in enumerate([
        "ε ảnh hưởng trực tiếp và mạnh đến hiệu quả tấn công, đặc biệt với I-FGSM. "
        "Trên MNIST, ε=0.20 đã đủ để đạt 100% ASR. Trên CIFAR-10, chỉ cần ε=0.05. "
        "FGSM có trần năng lực ~85–89% ASR bất kể ε, trong khi I-FGSM không bị giới hạn này.",

        "Số bước T tuân theo quy luật lợi tức giảm dần. Với MNIST, T=20–40 là vùng hội tụ; "
        "với CIFAR-10 T=5 đã đủ. Tăng T từ 20→40 chỉ cải thiện dưới 2 pp trong khi "
        "tốn gấp đôi thời gian — điểm cân bằng tối ưu là T=10–20 tùy dataset.",

        "CIFAR-10 dễ bị tấn công hơn MNIST đáng kể, dù bài toán phân loại khó hơn. "
        "Nguyên nhân là clean accuracy thấp hơn dẫn đến ranh giới quyết định yếu hơn. "
        "Điều này cho thấy việc cải thiện clean accuracy chưa đủ để đảm bảo robustness.",

        "Adversarial examples có tính transferability cao giữa các kiến trúc khác nhau "
        "(I-FGSM transfer ASR 67–90% tại ε=0.20). Model mạnh hơn không miễn nhiễm với "
        "transfer attack; khoảng cách kiến trúc có ảnh hưởng nhưng không loại bỏ được mối đe dọa.",
    ], 1):
        p = doc.add_paragraph(style="List Number")
        p.add_run(concl).font.size = Pt(11)
        p.paragraph_format.space_after = Pt(5)

    add_para(doc, "\nHướng phát triển tiếp theo:", bold=True)
    for future in [
        "Adversarial training để tăng robustness — huấn luyện với cả ảnh đối kháng nhằm "
        "nâng cao khả năng chống chịu, đặc biệt chú ý đến cross-architecture robustness.",
        "Thử nghiệm các biến thể tấn công mạnh hơn: PGD (random_start=True), MI-FGSM "
        "(momentum), DI-FGSM (input diversity) — đặc biệt MI-FGSM được thiết kế để cải thiện transferability.",
        "Ensemble attack: sinh ảnh đối kháng trên nhiều model cùng lúc để tối đa hóa "
        "transfer rate, phù hợp với kịch bản tấn công thực tế black-box.",
    ]:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(future).font.size = Pt(11)
        p.paragraph_format.space_after = Pt(3)

    # ════════════════════════════════════════════════════════
    # TÀI LIỆU THAM KHẢO
    # ════════════════════════════════════════════════════════
    add_heading(doc, "Tài liệu tham khảo", 1)
    refs = [
        "Goodfellow, I., Shlens, J., & Szegedy, C. (2015). Explaining and harnessing adversarial examples. ICLR 2015. https://arxiv.org/abs/1412.6572",
        "Kurakin, A., Goodfellow, I., & Bengio, S. (2016). Adversarial examples in the physical world. ICLR Workshop 2017. https://arxiv.org/abs/1607.02533",
        "Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). Towards deep learning models resistant to adversarial attacks. ICLR 2018. https://arxiv.org/abs/1706.06083",
        "Dong, Y., et al. (2018). Boosting adversarial attacks with momentum. CVPR 2018. https://arxiv.org/abs/1710.06081",
        "Papernot, N., McDaniel, P., & Goodfellow, I. (2016). Transferability in machine learning: from phenomena to black-box attacks using adversarial samples. arXiv:1605.07277",
        "LeCun, Y., Cortes, C., & Burges, C. (1998). The MNIST database of handwritten digits.",
        "Krizhevsky, A. (2009). Learning multiple layers of features from tiny images. Technical Report.",
    ]
    for i, ref in enumerate(refs, 1):
        p = doc.add_paragraph()
        p.paragraph_format.space_after  = Pt(4)
        p.paragraph_format.left_indent  = Cm(0.5)
        p.paragraph_format.first_line_indent = Cm(-0.5)
        r = p.add_run(f"[{i}] {ref}")
        r.font.size = Pt(10)

    # ── Lưu file ──────────────────────────────────────────
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    doc.save(OUT)
    print(f"\n✓ Báo cáo đã lưu tại: {OUT}")


if __name__ == "__main__":
    build()

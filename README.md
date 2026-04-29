# 🎯 I-FGSM Adversarial Attack — Image Classifier

Mô phỏng tấn công đối kháng **I-FGSM (Iterative Fast Gradient Sign Method)** lên bộ phân lớp ảnh MNIST / CIFAR-10.

> 📄 Paper gốc: *Adversarial Examples in the Physical World* — Kurakin, Goodfellow & Bengio (2016)  
> 🔗 https://arxiv.org/abs/1607.02533

---

## 📁 Cấu trúc Project

```
ifgsm_project/
│
├── attacks/                    # ← Thuật toán tấn công
│   ├── fgsm.py                 #   FGSM baseline (1 bước)
│   └── ifgsm.py                #   I-FGSM chính (N bước lặp)
│
├── models/                     # ← Kiến trúc mô hình
│   ├── cnn.py                  #   SimpleCNN (MNIST & CIFAR)
│   └── resnet.py               #   ResNet-18 wrapper
│
├── utils/                      # ← Tiện ích
│   ├── data_loader.py          #   Load MNIST / CIFAR-10
│   ├── trainer.py              #   Training loop
│   ├── evaluator.py            #   Đánh giá dưới tấn công
│   └── visualization.py        #   Vẽ biểu đồ & ảnh
│
├── experiments/                # ← Các thí nghiệm
│   ├── exp1_epsilon.py         #   Khảo sát ε vs Accuracy
│   ├── exp2_steps.py           #   Khảo sát T vs Accuracy
│   └── exp3_visualize.py       #   Trực quan hóa adversarial
│
├── configs/
│   └── config.yaml             # ← Cấu hình toàn bộ project
│
├── results/                    # ← Output (tự tạo)
│   ├── figures/                #   Biểu đồ (.png)
│   ├── logs/                   #   Số liệu (.json)
│   └── checkpoints/            #   Model weights (.pth)
│
├── tests/
│   └── test_ifgsm.py           # ← Unit tests (pytest)
│
├── train.py                    # ← Script huấn luyện
├── main.py                     # ← Pipeline chính
└── requirements.txt
```

---

## ⚙️ Cài đặt

```bash
# 1. Clone / tải project
cd ifgsm_project

# 2. Tạo môi trường ảo (khuyến nghị)
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 3. Cài dependencies
pip install -r requirements.txt
```

---

## 🚀 Chạy nhanh

```bash
# Chạy toàn bộ pipeline (train + 3 experiments)
python main.py

# Chỉ train
python train.py

# Bỏ qua train (dùng checkpoint có sẵn)
python main.py --skip-train

# Chỉ chạy experiment cụ thể
python main.py --skip-train --exp 1 3

# Thay đổi dataset
python train.py --dataset CIFAR10
```

---

## 🔬 Công thức I-FGSM

```
x₀    = x
xₜ₊₁ = Clip_{x,ε} [ xₜ + α · sign(∇ₓ J(θ, xₜ, y)) ]
```

| Ký hiệu | Ý nghĩa |
|---|---|
| `ε` | Biên độ nhiễu tối đa (L∞ norm) |
| `α` | Bước mỗi iteration = ε / T |
| `T` | Số bước lặp |
| `Clip` | Giữ nhiễu trong [-ε, +ε] và pixel trong [0,1] |

---

## 📊 Thí nghiệm

| # | Thí nghiệm | Output |
|---|---|---|
| Exp1 | Accuracy vs Epsilon (ε) | `exp1_acc_vs_epsilon.png` |
| Exp2 | Accuracy vs Num Steps (T) | `exp2_acc_vs_steps.png` |
| Exp3 | Visualize gốc vs nhiễu vs đối kháng | `exp3_examples.png` |

---

## 🧪 Chạy tests

```bash
pip install pytest
python -m pytest tests/ -v
```

---

## 📝 Hyperparameters quan trọng

Chỉnh sửa trong `configs/config.yaml`:

```yaml
attack:
  epsilon  : 0.3    # biên độ nhiễu tối đa
  num_steps: 40     # số bước lặp T
  alpha    : null   # null → tự tính = epsilon/num_steps

experiment:
  epsilon_list: [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
  steps_list  : [5, 10, 20, 40]
```

---

## 📚 Tài liệu tham khảo

| Paper | Link |
|---|---|
| FGSM — Goodfellow et al. (2015) | https://arxiv.org/abs/1412.6572 |
| **I-FGSM** — Kurakin et al. (2016) | https://arxiv.org/abs/1607.02533 |
| MI-FGSM — Dong et al. (2018) | https://arxiv.org/abs/1710.06081 |
| DI-FGSM — Xie et al. (2019) | https://arxiv.org/abs/1803.06978 |

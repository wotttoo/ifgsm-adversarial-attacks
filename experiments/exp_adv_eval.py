"""
experiments/exp_adv_eval.py
─────────────────────────────────────────────────────────────
So sánh Standard Model vs Adversarially Trained Model (FGSM-AT).

Với mỗi dataset (MNIST / CIFAR-10):
  1. Load standard model  (mnist_best.pth / cifar10_best.pth)
  2. Load adversarial model (mnist_adv_best.pth / cifar10_adv_best.pth)
  3. Sweep epsilon → đo clean acc + FGSM robust acc cho cả 2 model
  4. Vẽ biểu đồ so sánh + lưu JSON

Output:
    results/logs/exp_adv_eval_{dataset}.json
    results/figures/adv_robustness_comparison_{dataset}.png

Cách chạy:
    python experiments/exp_adv_eval.py
    python experiments/exp_adv_eval.py --dataset CIFAR10
"""

import sys, os, argparse, json, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import yaml

from models              import SimpleCNN
from utils.data_loader   import get_dataloaders, get_in_channels, get_input_size, get_clip_values
from attacks.fgsm        import fgsm_attack
from utils.visualization import plot_robustness_comparison


def eval_model_fgsm(model, loader, epsilon_list, clip_min, clip_max, device, num_samples=1000):
    """
    Đánh giá model với FGSM tại nhiều mức epsilon.
    Trả về list dict chứa clean_acc, fgsm_acc, fgsm_asr cho từng epsilon.
    """
    model.eval()

    # Lấy đủ mẫu từ loader
    all_images, all_labels = [], []
    for imgs, lbls in loader:
        all_images.append(imgs)
        all_labels.append(lbls)
        if sum(x.size(0) for x in all_images) >= num_samples * 2:
            break
    all_images = torch.cat(all_images)[:num_samples * 2].to(device)
    all_labels = torch.cat(all_labels)[:num_samples * 2].to(device)

    # Phase 1: lọc mẫu đúng
    with torch.no_grad():
        preds = model(all_images).argmax(1)
        mask  = preds == all_labels

    correct_imgs  = all_images[mask]
    correct_lbls  = all_labels[mask]
    n_correct     = int(mask.sum().item())
    total         = all_images.size(0)
    clean_acc     = 100.0 * n_correct / total

    results = []
    for eps in epsilon_list:
        t0 = time.perf_counter()

        survived = 0
        bs = 64
        for start in range(0, n_correct, bs):
            imgs_b = correct_imgs[start:start+bs]
            lbls_b = correct_lbls[start:start+bs]
            adv = fgsm_attack(model, imgs_b, lbls_b,
                              epsilon=eps, clip_min=clip_min, clip_max=clip_max)
            with torch.no_grad():
                survived += (model(adv).argmax(1) == lbls_b).sum().item()

        fgsm_acc = 100.0 * survived / total
        fgsm_asr = 100.0 * (n_correct - survived) / n_correct
        elapsed  = time.perf_counter() - t0

        results.append({
            "epsilon"   : eps,
            "total"     : total,
            "n_correct" : n_correct,
            "clean_acc" : clean_acc,
            "fgsm_acc"  : fgsm_acc,
            "fgsm_asr"  : fgsm_asr,
            "time_s"    : round(elapsed, 3),
        })
        print(f"    ε={eps:.2f}: clean={clean_acc:.1f}% | fgsm_acc={fgsm_acc:.1f}% | ASR={fgsm_asr:.1f}%")

    return results


def run(config_path="configs/config.yaml", dataset="MNIST"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    ds_name     = dataset
    epsilon_list = cfg["experiment"]["epsilon_list"]
    num_samples  = cfg["experiment"]["num_samples"]
    save_dir     = cfg["train"]["save_dir"]

    device = torch.device(
        cfg["experiment"]["device"] if torch.cuda.is_available() else "cpu"
    )

    in_ch      = get_in_channels(ds_name)
    input_size = get_input_size(ds_name)
    clip_min, clip_max = get_clip_values(ds_name)

    _, _, test_loader = get_dataloaders(
        ds_name,
        root       = cfg["dataset"]["root"],
        batch_size = 128,
        seed       = cfg["experiment"]["seed"],
    )

    def load_model(ckpt_name):
        net  = SimpleCNN(in_channels=in_ch, num_classes=10, input_size=input_size)
        path = os.path.join(ROOT, save_dir, ckpt_name)
        if not os.path.exists(path):
            return None, path
        ckpt = torch.load(path, map_location=device)
        net.load_state_dict(ckpt["model_state"])
        net = net.to(device).eval()
        return net, path

    # ── Standard model ────────────────────────────────────────
    std_tag  = ds_name.lower()
    adv_tag  = f"{ds_name.lower()}_adv"

    print(f"\n[Exp-AdvEval] Dataset: {ds_name} | Device: {device}")

    std_model, std_path = load_model(f"{std_tag}_best.pth")
    if std_model is None:
        print(f"  [ERROR] Không tìm thấy: {std_path}")
        return

    adv_model, adv_path = load_model(f"{adv_tag}_best.pth")
    if adv_model is None:
        print(f"  [ERROR] Không tìm thấy: {adv_path}")
        print("  Hãy chạy trước: python train_adv.py --dataset", ds_name)
        return

    print(f"  Standard : {std_path}")
    print(f"  Adversarial: {adv_path}")

    # ── Đánh giá ──────────────────────────────────────────────
    print("\n  Đánh giá Standard model:")
    std_results = eval_model_fgsm(
        std_model, test_loader, epsilon_list, clip_min, clip_max, device, num_samples
    )

    print("\n  Đánh giá Adversarial model:")
    adv_results = eval_model_fgsm(
        adv_model, test_loader, epsilon_list, clip_min, clip_max, device, num_samples
    )

    # ── Lưu JSON ──────────────────────────────────────────────
    log_dir = os.path.join(ROOT, "results", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"exp_adv_eval_{ds_name.lower()}.json")
    with open(log_path, "w") as f:
        json.dump({"standard": std_results, "adversarial": adv_results}, f, indent=2)
    print(f"\n  Log: {log_path}")

    # ── Vẽ biểu đồ so sánh ───────────────────────────────────
    fig_dir = os.path.join(ROOT, "results", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    save_path = os.path.join(fig_dir, f"adv_robustness_comparison_{ds_name.lower()}.png")

    plot_robustness_comparison(
        std_results  = std_results,
        adv_results  = adv_results,
        dataset_name = ds_name,
        save_path    = save_path,
    )
    print(f"  Figure: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="both",
                        choices=["MNIST", "CIFAR10", "both"])
    parser.add_argument("--config",  type=str, default="configs/config.yaml")
    args = parser.parse_args()

    config_path = os.path.join(ROOT, args.config)
    datasets = ["MNIST", "CIFAR10"] if args.dataset == "both" else [args.dataset]
    for ds in datasets:
        run(config_path, ds)


if __name__ == "__main__":
    main()

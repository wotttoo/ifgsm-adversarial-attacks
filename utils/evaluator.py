"""
utils/evaluator.py
─────────────────────────────────────────────────────────────
Evaluator — đánh giá model trên clean và adversarial examples.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Optional

from attacks.ifgsm import IFGSMAttack
from attacks.fgsm  import fgsm_attack


class AdversarialEvaluator:
    """
    Đánh giá toàn diện model dưới tấn công I-FGSM (và FGSM baseline).

    Args:
        model     : nn.Module đã được train
        device    : torch.device
        clip_min  : giá trị pixel nhỏ nhất
        clip_max  : giá trị pixel lớn nhất
    """

    def __init__(
        self,
        model    : nn.Module,
        device   : torch.device = torch.device("cpu"),
        clip_min : float        = 0.0,
        clip_max : float        = 1.0,
    ):
        self.model    = model.to(device)
        self.device   = device
        self.clip_min = clip_min
        self.clip_max = clip_max

    # ── Đánh giá với 1 epsilon ────────────────────────────────
    def evaluate_epsilon(
        self,
        loader    : DataLoader,
        epsilon   : float,
        num_steps : int,
        alpha     : Optional[float] = None,
        max_batches : Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Đánh giá model với epsilon cố định.

        Returns:
            dict với clean_acc, fgsm_acc, ifgsm_acc và drop tương ứng
        """
        self.model.eval()
        attacker = IFGSMAttack(
            self.model, epsilon=epsilon,
            alpha=alpha, num_steps=num_steps,
            clip_min=self.clip_min, clip_max=self.clip_max,
        )

        stats = {"clean": 0, "fgsm": 0, "ifgsm": 0, "total": 0}

        for i, (images, labels) in enumerate(tqdm(loader, desc=f"  ε={epsilon:.3f}", leave=False)):
            if max_batches and i >= max_batches:
                break

            images, labels = images.to(self.device), labels.to(self.device)

            # Clean accuracy
            with torch.no_grad():
                preds = self.model(images).argmax(1)
                stats["clean"] += (preds == labels).sum().item()

            # FGSM accuracy
            adv_fgsm = fgsm_attack(self.model, images, labels, epsilon,
                                    self.clip_min, self.clip_max)
            with torch.no_grad():
                preds = self.model(adv_fgsm).argmax(1)
                stats["fgsm"] += (preds == labels).sum().item()

            # I-FGSM accuracy
            adv_ifgsm = attacker(images, labels)
            with torch.no_grad():
                preds = self.model(adv_ifgsm).argmax(1)
                stats["ifgsm"] += (preds == labels).sum().item()

            stats["total"] += labels.size(0)

        N = stats["total"]
        clean_acc = 100.0 * stats["clean"] / N
        fgsm_acc  = 100.0 * stats["fgsm"]  / N
        ifgsm_acc = 100.0 * stats["ifgsm"] / N

        return {
            "epsilon"   : epsilon,
            "clean_acc" : clean_acc,
            "fgsm_acc"  : fgsm_acc,
            "ifgsm_acc" : ifgsm_acc,
            "fgsm_drop" : clean_acc - fgsm_acc,
            "ifgsm_drop": clean_acc - ifgsm_acc,
        }

    # ── Đánh giá trên nhiều epsilon ───────────────────────────
    def evaluate_epsilon_range(
        self,
        loader       : DataLoader,
        epsilon_list : List[float],
        num_steps    : int,
        alpha        : Optional[float] = None,
        max_batches  : Optional[int]   = None,
    ) -> List[Dict[str, float]]:
        """
        Đánh giá model trên danh sách epsilon.

        Returns:
            list of dicts — mỗi dict chứa kết quả cho 1 epsilon
        """
        results = []

        print(f"\n{'─'*60}")
        print(f"  Đánh giá I-FGSM | steps={num_steps} | "
              f"epsilon: {epsilon_list}")
        print(f"{'─'*60}")

        for eps in epsilon_list:
            r = self.evaluate_epsilon(loader, eps, num_steps, alpha, max_batches)
            results.append(r)
            print(
                f"  ε={r['epsilon']:.3f} | "
                f"Clean={r['clean_acc']:6.2f}% | "
                f"FGSM={r['fgsm_acc']:6.2f}% (↓{r['fgsm_drop']:5.2f}%) | "
                f"I-FGSM={r['ifgsm_acc']:6.2f}% (↓{r['ifgsm_drop']:5.2f}%)"
            )

        return results

    # ── Đánh giá theo số bước ─────────────────────────────────
    def evaluate_steps(
        self,
        loader      : DataLoader,
        epsilon     : float,
        steps_list  : List[int],
        max_batches : Optional[int] = None,
    ) -> List[Dict[str, float]]:
        """
        Khảo sát ảnh hưởng của số bước lặp lên accuracy.
        """
        results = []

        print(f"\n{'─'*60}")
        print(f"  Khảo sát num_steps | ε={epsilon} | steps: {steps_list}")
        print(f"{'─'*60}")

        self.model.eval()
        for steps in steps_list:
            attacker = IFGSMAttack(
                self.model, epsilon=epsilon, num_steps=steps,
                clip_min=self.clip_min, clip_max=self.clip_max,
            )
            correct, total = 0, 0

            for i, (images, labels) in enumerate(
                tqdm(loader, desc=f"  steps={steps}", leave=False)
            ):
                if max_batches and i >= max_batches:
                    break
                images, labels = images.to(self.device), labels.to(self.device)
                adv = attacker(images, labels)
                with torch.no_grad():
                    preds = self.model(adv).argmax(1)
                    correct += (preds == labels).sum().item()
                    total   += labels.size(0)

            acc = 100.0 * correct / total
            results.append({"num_steps": steps, "adv_acc": acc})
            print(f"  steps={steps:3d} → adv_acc={acc:.2f}%")

        return results

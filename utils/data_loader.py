"""
utils/data_loader.py
─────────────────────────────────────────────────────────────
Load MNIST / CIFAR-10 với train / val / test split chuẩn.
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from typing import Tuple


def get_transforms(dataset_name: str) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Trả về (train_transform, test_transform) phù hợp với dataset.
    """
    if dataset_name.upper() == "MNIST":
        train_tf = transforms.Compose([
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
            transforms.ToTensor(),
        ])
        test_tf = transforms.Compose([
            transforms.ToTensor(),
        ])

    elif dataset_name.upper() == "CIFAR10":
        mean = (0.4914, 0.4822, 0.4465)
        std  = (0.2470, 0.2435, 0.2616)
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        raise ValueError(f"Dataset không được hỗ trợ: {dataset_name}")

    return train_tf, test_tf


def get_dataloaders(
    dataset_name : str  = "MNIST",
    root         : str  = "./data",
    batch_size   : int  = 64,
    val_split    : float = 0.1,
    num_workers  : int  = 2,
    seed         : int  = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Tạo train / val / test DataLoader.

    Args:
        dataset_name : "MNIST" | "CIFAR10"
        root         : thư mục lưu data
        batch_size   : batch size
        val_split    : tỷ lệ validation trong tập train
        num_workers  : số worker cho DataLoader
        seed         : random seed

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_tf, test_tf = get_transforms(dataset_name)

    # ── Load raw dataset ──────────────────────────────────────
    DS = datasets.MNIST if dataset_name.upper() == "MNIST" else datasets.CIFAR10

    full_train = DS(root=root, train=True,  download=True, transform=train_tf)
    test_set   = DS(root=root, train=False, download=True, transform=test_tf)

    # ── Train / Val split ─────────────────────────────────────
    n_val   = int(len(full_train) * val_split)
    n_train = len(full_train) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(
        full_train, [n_train, n_val], generator=generator
    )

    # ── DataLoaders ───────────────────────────────────────────
    train_loader = DataLoader(
        train_set, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )

    print(f"[DataLoader] {dataset_name}: "
          f"train={n_train:,} | val={n_val:,} | test={len(test_set):,}")

    return train_loader, val_loader, test_loader


def get_in_channels(dataset_name: str) -> int:
    """Trả về số kênh ảnh của dataset."""
    return 1 if dataset_name.upper() == "MNIST" else 3


def get_input_size(dataset_name: str) -> int:
    """Trả về kích thước ảnh H=W của dataset."""
    return 28 if dataset_name.upper() == "MNIST" else 32

# train_lora.py
# LoRA adaptation on ViT with label-ratio (Route B), supporting CIFAR10 / CIFAR100 / STL10.
#
# Key behavior:
# - Labels are used ONLY in this script (representation adaptation).
# - Set --label_ratio p to train LoRA on a class-balanced labeled subset (p% per class).
# - p=0 skips training and saves initial LoRA weights (Frozen-ViT baseline compatibility).
#
# Example:
#   python train_lora.py --dataset cifar10  --label_ratio 0.1 --epochs 20 --seed 0
#   python train_lora.py --dataset cifar100 --label_ratio 0.05 --epochs 10 --seed 0
#   python train_lora.py --dataset stl10    --label_ratio 0.1 --epochs 20 --seed 0

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from lora_vit import ViTWithLoRA
from utils import set_seed



# -----------------------------
# 1) Dataset helpers
# -----------------------------
def build_transform(img_size=224):
    # ViT pretraining typically uses ImageNet normalization; use it for stable transfer.
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_dataset(dataset_name: str, data_dir: str, train: bool, transform):
    """
    Returns: (dataset, num_classes)
    Supported:
      - cifar10:   50k train / 10k test, 10 classes
      - cifar100:  50k train / 10k test, 100 classes
      - stl10:     5k train (labeled) / 8k test, 10 classes
                  (unlabeled split exists but has no labels; not used for supervised adaptation)
    """
    name = dataset_name.lower()
    if name == "cifar10":
        ds = datasets.CIFAR10(root=data_dir, train=train, download=True, transform=transform)
        return ds, 10

    if name == "cifar100":
        ds = datasets.CIFAR100(root=data_dir, train=train, download=True, transform=transform)
        return ds, 100

    if name == "stl10":
        split = "train" if train else "test"
        ds = datasets.STL10(root=data_dir, split=split, download=True, transform=transform)
        return ds, 10

    raise ValueError(f"Unsupported dataset: {dataset_name}. Choose from cifar10, cifar100, stl10.")


def get_targets(ds):
    """
    Extract integer targets from torchvision datasets.
    - CIFAR10/CIFAR100: ds.targets (list[int])
    - STL10: ds.labels (np.ndarray[int])
    """
    if hasattr(ds, "targets"):
        return list(ds.targets)
    if hasattr(ds, "labels"):
        return list(ds.labels)
    raise AttributeError("Dataset does not expose targets/labels.")


# -----------------------------
# 2) Label-ratio sampling
# -----------------------------
def stratified_labeled_indices(targets, label_ratio, num_classes, seed=0, min_per_class=1):
    """
    Class-balanced sampling:
    - For each class c, sample round(p * n_c) examples, with lower bound min_per_class (if p>0).
    - If label_ratio <= 0, return empty list.
    """
    rng = np.random.RandomState(seed)
    targets = np.asarray(targets, dtype=int)

    labeled = []
    for c in range(num_classes):
        idx_c = np.where(targets == c)[0]
        rng.shuffle(idx_c)

        if label_ratio <= 0:
            n_c = 0
        else:
            n_c = int(round(label_ratio * len(idx_c)))
            n_c = max(n_c, min_per_class)
            n_c = min(n_c, len(idx_c))

        labeled.append(idx_c[:n_c])

    labeled = np.concatenate(labeled, axis=0) if len(labeled) else np.array([], dtype=int)
    rng.shuffle(labeled)
    return labeled.tolist()


def print_subset_stats(targets, labeled_idx, num_classes):
    targets = np.asarray(targets, dtype=int)
    labeled_targets = targets[np.asarray(labeled_idx, dtype=int)]
    counts = [(c, int((labeled_targets == c).sum())) for c in range(num_classes)]
    total = len(labeled_idx)
    msg = " | ".join([f"class{c}:{n}" for c, n in counts[:10]])
    if num_classes > 10:
        msg += f" | ... (total_classes={num_classes})"
    print(f"[LabelSubset] labeled_total={total} | {msg}")


# -----------------------------
# 3) Loader
# -----------------------------
def get_loader(
    dataset_name: str,
    data_dir: str,
    batch: int,
    train: bool,
    label_ratio: float,
    seed: int,
    min_per_class: int,
    num_workers: int,
    img_size: int = 224,
):
    tfm = build_transform(img_size=img_size)
    ds_full, num_classes = get_dataset(dataset_name, data_dir, train=train, transform=tfm)

    if train and label_ratio < 1.0:
        targets = get_targets(ds_full)
        labeled_idx = stratified_labeled_indices(
            targets=targets,
            label_ratio=label_ratio,
            num_classes=num_classes,
            seed=seed,
            min_per_class=min_per_class,
        )
        print_subset_stats(targets, labeled_idx, num_classes=num_classes)
        ds = Subset(ds_full, labeled_idx)
        print(f"[LabelRatio] p={label_ratio:.4f} | labeled={len(labeled_idx)} / total={len(ds_full)}")
    else:
        ds = ds_full
        if train:
            print(f"[LabelRatio] p={label_ratio:.4f} | labeled=ALL ({len(ds_full)})")

    loader = DataLoader(
        ds,
        batch_size=batch,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return loader, num_classes, len(ds_full)


# -----------------------------
# 4) Train (LoRA + linear head)
# -----------------------------
def train_supervised(
    dataset: str,
    data_dir: str,
    epochs: int,
    lr: float,
    r: int,
    alpha: int,
    dropout: float,
    model_name: str,
    label_ratio: float,
    seed: int,
    min_per_class: int,
    out: str,
    batch: int,
    num_workers: int,
    img_size: int,
):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build train loader first to know num_classes
    train_loader, num_classes, total_train = get_loader(
        dataset_name=dataset,
        data_dir=data_dir,
        batch=batch,
        train=True,
        label_ratio=label_ratio,
        seed=seed,
        min_per_class=min_per_class,
        num_workers=num_workers,
        img_size=img_size,
    )

    model = ViTWithLoRA(model_name=model_name, r=r, lora_alpha=alpha, lora_dropout=dropout).to(device)
    head = nn.Linear(model.backbone.num_features, num_classes).to(device)

    # Only train LoRA params + head (backbone should be frozen inside ViTWithLoRA)
    params = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    optim = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()

    # p=0 => skip adaptation entirely
    if label_ratio <= 0:
        torch.save(model.state_dict(), out)
        print(f"[LabelRatio] p=0 -> skip training. Saved initial LoRA weights to {out}")
        return

    model.train()
    head.train()

    for ep in range(epochs):
        running = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {ep+1}/{epochs}"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            feats = model(x)
            logits = head(feats)
            loss = criterion(logits, y)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            running += loss.item() * x.size(0)

        avg_loss = running / len(train_loader.dataset)
        print(f"Epoch {ep+1} | loss={avg_loss:.4f}")
    out = f"vit_lora_label{label_ratio}_{dataset}.pth"
    torch.save(model.state_dict(), out)
    print(f"Saved LoRA weights to vit_lora_label{label_ratio}_{dataset}.pth")


# -----------------------------
# 5) CLI
# -----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="cifar10",
                    choices=["cifar10", "cifar100", "stl10"],
                    help="dataset for LoRA adaptation")
    ap.add_argument("--data_dir", type=str, default="./data")

    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--model", dest="model_name", default="vit_base_patch16_224")

    # Route-B controls
    ap.add_argument("--label_ratio", type=float, default=1.0,
                    help="fraction of labeled data used in LoRA adaptation (0~1). 0 means skip adaptation.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_per_class", type=int, default=1)
    ap.add_argument("--out", type=str, default="vit_lora_label1_cifar10.pth")

    ap.add_argument("--batch", type=int, default=64)
    # Windows 下 num_workers>0 有时不稳定；如遇卡死/慢，改为 0。
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--img_size", type=int, default=224)

    args = ap.parse_args()
    train_supervised(**vars(args))


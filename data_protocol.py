import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def cifar10_dataset(data_dir, train, transform=None, download=False):
    return datasets.CIFAR10(
        root=data_dir,
        train=train,
        transform=transform,
        download=download,
    )


def stratified_indices(targets, label_ratio=0.01, seed=0):
    targets = np.asarray(targets, dtype=np.int64)
    classes = np.unique(targets)
    rng = np.random.default_rng(seed)
    selected = []

    for class_id in classes:
        class_indices = np.flatnonzero(targets == class_id)
        count = int(round(len(class_indices) * label_ratio))
        if label_ratio > 0:
            count = max(1, count)
        selected.extend(rng.choice(class_indices, size=count, replace=False).tolist())

    rng.shuffle(selected)
    return selected


def prepare_cifar10_split(
    data_dir,
    split_path,
    label_ratio=0.01,
    seed=0,
    download=False,
):
    split_path = Path(split_path)
    dataset = cifar10_dataset(data_dir, train=True, download=download)
    targets = np.asarray(dataset.targets, dtype=np.int64)

    if split_path.exists():
        payload = json.loads(split_path.read_text(encoding="utf-8"))
        expected = {
            "dataset": "cifar10",
            "train_size": len(dataset),
            "label_ratio": label_ratio,
            "seed": seed,
        }
        mismatches = {
            key: (payload.get(key), value)
            for key, value in expected.items()
            if payload.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Existing split metadata does not match: {mismatches}")
        labeled_indices = payload["labeled_indices"]
    else:
        labeled_indices = stratified_indices(targets, label_ratio=label_ratio, seed=seed)
        labeled_set = set(labeled_indices)
        unlabeled_indices = [i for i in range(len(dataset)) if i not in labeled_set]
        counts = Counter(targets[labeled_indices].tolist())
        payload = {
            "dataset": "cifar10",
            "train_size": len(dataset),
            "num_classes": 10,
            "label_ratio": label_ratio,
            "seed": seed,
            "labeled_count": len(labeled_indices),
            "labeled_per_class": {str(c): counts[c] for c in range(10)},
            "labeled_indices": labeled_indices,
            "unlabeled_indices": unlabeled_indices,
        }
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    counts = Counter(targets[labeled_indices].tolist())
    if len(labeled_indices) != 500 or any(counts[c] != 50 for c in range(10)):
        raise ValueError(
            "The CIFAR-10 1% protocol requires exactly 500 labels, 50 per class. "
            f"Got total={len(labeled_indices)}, counts={dict(counts)}"
        )
    return payload


def build_ssl_transform(img_size=224):
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                img_size,
                scale=(0.5, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))],
                p=0.5,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_weak_transform(img_size=224):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                img_size,
                scale=(0.8, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_strong_transform(img_size=224):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                img_size,
                scale=(0.5, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transform(img_size=224):
    return transforms.Compose(
        [
            transforms.Resize(
                img_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class MultiScaleEvalTransform:
    def __init__(self, img_size=224, resize_scales=(224, 240, 256)):
        self.transforms = [
            transforms.Compose(
                [
                    transforms.Resize(
                        scale,
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.CenterCrop(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ]
            )
            for scale in resize_scales
        ]

    def __call__(self, image):
        return torch.stack([transform(image) for transform in self.transforms], dim=0)


def build_multiscale_eval_transform(img_size=224, resize_scales=(224, 240, 256)):
    if not resize_scales:
        raise ValueError("At least one evaluation resize scale is required.")
    if any(scale < img_size for scale in resize_scales):
        raise ValueError(
            f"All resize scales must be >= img_size ({img_size}), got {resize_scales}."
        )
    return MultiScaleEvalTransform(
        img_size=img_size,
        resize_scales=tuple(resize_scales),
    )


def load_protocol_split(split_path):
    payload = json.loads(Path(split_path).read_text(encoding="utf-8"))
    required = {
        "dataset",
        "train_size",
        "num_classes",
        "label_ratio",
        "seed",
        "labeled_indices",
        "unlabeled_indices",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"Protocol split is missing fields: {missing}")
    if payload["dataset"] != "cifar10" or payload["label_ratio"] != 0.01:
        raise ValueError("Stage 2 requires the fixed CIFAR-10 1% protocol.")
    return payload


class TwoViewUnlabeledDataset(Dataset):
    """Expose two augmented views without exposing labels to the training loop."""

    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        image, _ = self.base_dataset[index]
        return self.transform(image), self.transform(image)


class LabeledTwoViewDataset(Dataset):
    def __init__(self, base_dataset, weak_transform, strong_transform):
        self.base_dataset = base_dataset
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        image, label = self.base_dataset[index]
        return self.weak_transform(image), self.strong_transform(image), label


class WeakStrongUnlabeledDataset(Dataset):
    def __init__(self, base_dataset, weak_transform, strong_transform):
        self.base_dataset = base_dataset
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        image, _ = self.base_dataset[index]
        return self.weak_transform(image), self.strong_transform(image)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)

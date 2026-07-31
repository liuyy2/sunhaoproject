import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def simsiam_aug(img_size=224):
    """Build the image augmentation pipeline used by SimSiam."""
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(img_size + 32),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


class TwoViewDataset(torch.utils.data.Dataset):
    """Wrap a dataset and return two augmented views of the same image."""

    def __init__(self, base_dataset, transform):
        self.base = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        x1 = self.transform(img)
        x2 = self.transform(img)
        return x1, x2, label


def simsiam_loss(p, z):
    """Negative cosine similarity loss used by SimSiam."""
    z = z.detach()
    p = F.normalize(p, dim=-1)
    z = F.normalize(z, dim=-1)
    return -(p * z).sum(dim=1).mean()


class SimSiamHead(nn.Module):
    """Projection head and prediction head for SimSiam."""

    def __init__(self, dim=768, proj_dim=2048, pred_dim=512):
        super().__init__()

        self.projector = nn.Sequential(
            nn.Linear(dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )

        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, pred_dim),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_dim, proj_dim),
        )

    def forward(self, feat):
        z = self.projector(feat)
        p = self.predictor(z)
        return p, z


def total_sim_loss(z1, p1, z2, p2):
    """Symmetric SimSiam loss for two views."""
    loss_1 = simsiam_loss(p1, z2)
    loss_2 = simsiam_loss(p2, z1)
    return (loss_1 + loss_2) / 2


def sep_reg(feat_batch):
    """Penalize similarity between different samples in a batch."""
    b_size = feat_batch.shape[0]
    if b_size < 2:
        return feat_batch.new_tensor(0.0)

    feat_norm = F.normalize(feat_batch, dim=-1)
    gram_matrix = torch.matmul(feat_norm, feat_norm.T)
    eye_mask = torch.eye(b_size, device=feat_batch.device, dtype=gram_matrix.dtype)
    cross_sim = gram_matrix * (1 - eye_mask)
    return cross_sim.sum() / (b_size * (b_size - 1))

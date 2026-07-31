# -*- coding: utf-8 -*-
"""
train_lora_unsup.py
支持两种训练模式：
1) 监督训练（默认）：CE + 冻结ViT骨干，仅训练LoRA + 线性分类头
2) 无监督训练（--unsup）：LoRA-ViT + 对称InfoNCE，对LoRA增量施加Frobenius正则
训练完成会保存：
- 监督模式：vit_lora_stl10.pth
- 无监督模式：vit_lora_unsup.pth
随后可直接用 extract_HAM.py 抽取 [CLS] 特征，再用 evaluate.py / anchor_spec.py 做聚类评测
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from tqdm import tqdm

# 你的LoRA-ViT封装
from lora_vit import ViTWithLoRA


# =========================
# 公用：数据变换 & loader
# =========================
def get_loader_supervised(batch=64, train=True, num_workers=4):
    tfm = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    ds = datasets.CIFAR10(root='./data', train=train, download=True, transform=tfm)
    return DataLoader(ds, batch_size=batch, shuffle=train, num_workers=num_workers, pin_memory=True)


class TwoCropsTransform:
    """返回同一图像的两种随机增强视图，用于对比学习"""
    def __init__(self, base_tfm):
        self.base_tfm = base_tfm
    def __call__(self, x):
        return self.base_tfm(x), self.base_tfm(x)


def get_loader_unsup(batch=128, num_workers=4):
    # 轻量但有效的SimCLR/DINO式增强
    tfm = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    two = TwoCropsTransform(tfm)
    ds = datasets.CIFAR10(root='./data', train=True, download=True, transform=two)
    return DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=True)


# =========================
# 公用：InfoNCE / 正则
# =========================
class ProjectionHead(nn.Module):
    """对比学习常用的2层MLP投影头"""
    def __init__(self, dim, hidden=None, out_dim=128):
        super().__init__()
        hidden = hidden or dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim, bias=True),
        )
    def forward(self, x):
        return self.net(x)


def info_nce_loss(z1, z2, tau=0.07):
    """对称InfoNCE"""
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = (z1 @ z2.t()) / tau  # [B,B]
    labels = torch.arange(z1.size(0), device=z1.device)
    loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)
    return loss * 0.5


def lora_fro_reg(model: nn.Module):
    """仅对LoRA参数施加 Frobenius 正则"""
    reg = 0.0
    for n, p in model.named_parameters():
        if p.requires_grad and ('lora_' in n):
            reg = reg + (p ** 2).sum()
    return reg


# =========================
# 监督训练（保持你原有流程）
# =========================
def train_supervised(epochs=5, lr=1e-3, r=8, alpha=16, dropout=0.0,
                     model_name='vit_base_patch16_224', weight_decay=1e-2,
                     batch_size=64, num_workers=4, num_classes=10):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = True

    # 冻结ViT骨干，仅训练LoRA
    model = ViTWithLoRA(model_name=model_name, r=r, lora_alpha=alpha, lora_dropout=dropout).to(device)

    # 线性分类头（监督模式专用）
    head = nn.Linear(model.backbone.num_features, num_classes).to(device)

    # 只训练LoRA参数 + 线性头
    params = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_loader = get_loader_supervised(batch=batch_size, train=True, num_workers=num_workers)
    test_loader = get_loader_supervised(batch=batch_size, train=False, num_workers=num_workers)

    best_acc = 0.0
    model.train(); head.train()
    for ep in range(epochs):
        running = 0.0
        correct, total = 0, 0
        for x, y in tqdm(train_loader, desc=f"[Sup] Epoch {ep+1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            feats = model(x)                 # [B, D] 由LoRA-ViT抽特征（冻结骨干，仅LoRA可训练）
            logits = head(feats)             # [B, C]
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        train_loss = running / total
        train_acc = correct / total
        print(f"Epoch {ep+1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}")

        # 简单val（用test loader）
        model.eval(); head.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = head(model(x))
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        test_acc = correct / total
        print(f"          test_acc={test_acc:.4f}")
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'vit_lora_stl10.pth')
            print("Saved supervised LoRA weights -> vit_lora_stl10.pth")
        model.train(); head.train()

    print(f"[Sup] Finished. Best test_acc={best_acc:.4f}")


# =========================
# 无监督训练（InfoNCE + LoRA正则）
# =========================
def train_unsupervised(epochs=30, lr=1e-4, r=8, alpha=16, dropout=0.0,
                       tau=0.07, weight_decay=0.05, reg_lambda=1e-4,
                       model_name='vit_base_patch16_224',
                       batch_size=128, num_workers=4):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = True

    model = ViTWithLoRA(model_name=model_name, r=r, lora_alpha=alpha, lora_dropout=dropout).to(device)
    proj = ProjectionHead(model.backbone.num_features, out_dim=128).to(device)

    # 只训练 LoRA + 投影头
    params = [p for p in model.parameters() if p.requires_grad] + list(proj.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    loader = get_loader_unsup(batch=batch_size, num_workers=num_workers)
    model.train(); proj.train()

    for ep in range(epochs):
        run_loss = 0.0
        n_samples = 0
        for (x1, x2), _ in tqdm(loader, desc=f"[Unsup] Epoch {ep+1}/{epochs}"):
            x1, x2 = x1.to(device), x2.to(device)

            f1 = model(x1)   # [B, D] [CLS]
            f2 = model(x2)

            z1 = proj(f1)    # 投影空间
            z2 = proj(f2)

            loss = info_nce_loss(z1, z2, tau=tau)
            loss = loss + reg_lambda * lora_fro_reg(model)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = x1.size(0)
            run_loss += loss.item() * bs
            n_samples += bs

        print(f"Epoch {ep+1}: unsup_loss={(run_loss / n_samples):.4f}")

    # 仅保存LoRA骨干（投影头仅参与训练，不影响后续抽特征）
    torch.save(model.state_dict(), 'vit_lora_unsup.pth')
    print("Saved unsupervised LoRA weights -> vit_lora_unsup.pth")


# =========================
# CLI
# =========================
def parse_args():
    ap = argparse.ArgumentParser()
    # 通用
    ap.add_argument('--model', dest='model_name', default='vit_base_patch16_224')
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--r', type=int, default=8)
    ap.add_argument('--alpha', type=int, default=16)
    ap.add_argument('--dropout', type=float, default=0.0)
    ap.add_argument('--num_workers', type=int, default=4)

    # 监督
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--weight_decay', type=float, default=1e-2)
    ap.add_argument('--num_classes', type=int, default=10)

    # 无监督
    ap.add_argument('--unsup', action='store_true', help='启用无监督对比学习训练')
    ap.add_argument('--tau', type=float, default=0.07)
    ap.add_argument('--wd', type=float, default=0.05)
    ap.add_argument('--reg_lambda', type=float, default=1e-4)
    ap.add_argument('--unsup_batch', type=int, default=128)

    return ap.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.unsup:
        train_unsupervised(
            epochs=args.epochs, lr=args.lr, r=args.r, alpha=args.alpha, dropout=args.dropout,
            tau=args.tau, weight_decay=args.wd, reg_lambda=args.reg_lambda,
            model_name=args.model_name, batch_size=args.unsup_batch, num_workers=args.num_workers
        )
    else:
        train_supervised(
            epochs=args.epochs, lr=args.lr, r=args.r, alpha=args.alpha, dropout=args.dropout,
            model_name=args.model_name, weight_decay=args.weight_decay,
            batch_size=args.batch, num_workers=args.num_workers, num_classes=args.num_classes
        )

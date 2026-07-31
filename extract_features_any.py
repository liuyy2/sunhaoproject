# extract_features_any.py  (fixed / clean for ablation)
import argparse, os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from lora_vit import ViTWithLoRA
from vit_frozen import FrozenViT

IMN_MEAN = (0.485, 0.456, 0.406)
IMN_STD  = (0.229, 0.224, 0.225)

def build_transform(pretrained: bool, grayscale_to_3ch: bool, is_train: bool):
    norm_mean, norm_std = (IMN_MEAN, IMN_STD) if pretrained else ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # 这里只做特征提取：用“测试/评估”风格即可（保持一致性）
    tfms = [transforms.Resize(224), transforms.CenterCrop(224)]
    if grayscale_to_3ch:
        tfms += [transforms.Grayscale(num_output_channels=3)]
    tfms += [transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)]
    return transforms.Compose(tfms)

def get_dataset(name: str, split: str, tfm):
    name = name.lower()
    if name == "cifar10":
        ds = datasets.CIFAR10("./data", train=(split=="train"), download=True, transform=tfm)
        num_classes = 10
    elif name == "cifar100":
        ds = datasets.CIFAR100("./data", train=(split=="train"), download=True, transform=tfm)
        num_classes = 100
    elif name == "stl10":
        ds = datasets.STL10("./data", split=split, download=True, transform=tfm)
        num_classes = 10
    elif name == "mnist":
        ds = datasets.MNIST("./data", train=(split=="train"), download=True, transform=tfm)
        num_classes = 10
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return ds, num_classes

def load_backbone(model, ckpt_path: str, strict_limit: int = 20):
    """
    Load ckpt once. If too many missing/unexpected keys -> raise (avoid silent failure).
    """
    if (ckpt_path is None) or (str(ckpt_path).lower() == "none") or (not os.path.exists(ckpt_path)):
        print("[CKPT] skipped (no ckpt). Using pretrained backbone only.")
        return None  # no ckpt obj

    print("[CKPT]", ckpt_path, "exists=True")
    obj = torch.load(ckpt_path, map_location="cpu")
    state = obj["backbone_state"] if (isinstance(obj, dict) and "backbone_state" in obj) else obj
    msg = model.load_state_dict(state, strict=False)
    missing = len(getattr(msg, "missing_keys", []))
    unexpected = len(getattr(msg, "unexpected_keys", []))
    print("[LOAD missing]", missing)
    print("[LOAD unexpected]", unexpected)

    # 强制约束：full baseline 特别怕“半加载”
    if missing > strict_limit or unexpected > strict_limit:
        raise RuntimeError(
            f"Checkpoint load looks wrong: missing={missing}, unexpected={unexpected}. "
            f"Please check model_name/timm version/ckpt format."
        )
    return obj

@torch.no_grad()
def main(dataset, split, mode, ckpt, model_name, r, alpha, dropout, pretrained, out_path, batch, use_head_logits, num_workers):
    grayscale_to_3ch = (dataset.lower() == "mnist")
    tfm = build_transform(pretrained=pretrained, grayscale_to_3ch=grayscale_to_3ch, is_train=False)
    ds, num_classes = get_dataset(dataset, split, tfm)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())

    device = "cuda" if torch.cuda.is_available() else "cpu"

    head = None
    if mode == "frozen":
        model = FrozenViT(model_name=model_name, pretrained=pretrained)
    elif mode == "lora":
        model = ViTWithLoRA(model_name=model_name, pretrained=pretrained, r=r, lora_alpha=alpha, lora_dropout=dropout)
    elif mode == "full":
        import timm
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
    elif mode == "head":
        model = FrozenViT(model_name=model_name, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # ---- load backbone once ----
    obj = load_backbone(model, ckpt_path=ckpt, strict_limit=20)

    # ---- optional head logits feature ----
    if use_head_logits:
        if obj is None or (not isinstance(obj, dict)) or ("head_state" not in obj):
            raise RuntimeError("use_head_logits=True but ckpt has no head_state.")
        if isinstance(model, FrozenViT):
            feat_dim = model.backbone.num_features
        elif isinstance(model, ViTWithLoRA):
            feat_dim = model.backbone.num_features
        else:
            feat_dim = model.num_features

        head = nn.Linear(feat_dim, num_classes)
        msg = head.load_state_dict(obj["head_state"], strict=False)
        print("[HEAD load missing]", len(getattr(msg, "missing_keys", [])))
        print("[HEAD load unexpected]", len(getattr(msg, "unexpected_keys", [])))

    model = model.to(device).eval()
    if head is not None:
        head = head.to(device).eval()

    feats, labels = [], []
    for x, y in tqdm(dl, desc=f"Extract {dataset}:{split} [{mode}]"):
        x = x.to(device, non_blocking=True)

        if use_head_logits:
            # logits feature
            if isinstance(model, FrozenViT):
                f = model.backbone(x)
            else:
                f = model(x)
            z = head(f)
            z = nn.functional.normalize(z, dim=-1)
            feats.append(z.detach().cpu().numpy())
        else:
            # backbone embedding feature
            if isinstance(model, FrozenViT):
                f = model.extract(x)
            elif isinstance(model, ViTWithLoRA):
                f = model.extract(x)
            else:
                f = nn.functional.normalize(model(x), dim=-1)
            feats.append(f.detach().cpu().numpy())

        labels.append(y.numpy())

    X = np.concatenate(feats, 0)
    y = np.concatenate(labels, 0)
    np.savez(out_path, X=X, y=y)
    print(f"[OK] Saved features to {out_path} | X={X.shape} | num_classes={num_classes}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100", "stl10", "mnist"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--mode", default="full", choices=["frozen", "lora", "full", "head"])
    ap.add_argument("--ckpt", default="none")
    ap.add_argument("--model_name", default="vit_base_patch16_224")
    ap.add_argument("--pretrained", type=lambda s: s.lower()=="true", default=True)

    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.0)

    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out_path", default="features_out.npz")
    ap.add_argument("--use_head_logits", type=lambda s: s.lower()=="true", default=False)
    ap.add_argument("--num_workers", type=int, default=0)  # Windows 建议 0

    args = ap.parse_args()
    main(**vars(args))

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from checkpointing import load_lora_state, save_stage1_checkpoint
from data_protocol import (
    TwoViewUnlabeledDataset,
    build_ssl_transform,
    cifar10_dataset,
    seed_worker,
)
from lora_vit import ViTWithLoRA
from utils import set_seed


class SimSiamHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=2048, output_dim=256, pred_dim=512):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim, bias=False),
            nn.BatchNorm1d(output_dim, affine=False),
        )
        self.predictor = nn.Sequential(
            nn.Linear(output_dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_dim, output_dim),
        )

    def forward(self, features):
        projection = self.projector(features)
        prediction = self.predictor(projection)
        return prediction, projection


def negative_cosine_similarity(prediction, target):
    prediction = F.normalize(prediction, dim=1)
    target = F.normalize(target.detach(), dim=1)
    return -(prediction * target).sum(dim=1).mean()


def simsiam_loss(p1, z1, p2, z2):
    return 0.5 * (
        negative_cosine_similarity(p1, z2)
        + negative_cosine_similarity(p2, z1)
    )


def variance_covariance_loss(z1, z2, variance_target=1.0):
    def variance_loss(z):
        std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-4)
        return F.relu(variance_target - std).mean()

    def covariance_loss(z):
        z = z - z.mean(dim=0)
        covariance = (z.T @ z) / max(1, z.size(0) - 1)
        covariance.fill_diagonal_(0)
        return covariance.square().sum() / z.size(1)

    return 0.5 * (variance_loss(z1) + variance_loss(z2)), 0.5 * (
        covariance_loss(z1) + covariance_loss(z2)
    )


@torch.no_grad()
def representation_diagnostics(feature_batches):
    features = torch.cat(feature_batches, dim=0).float()
    features = F.normalize(features, dim=1)
    feature_std = features.std(dim=0, unbiased=False).mean().item()

    centered = features - features.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(1, len(features) - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance.double()).clamp_min(0)
    probabilities = eigenvalues / eigenvalues.sum().clamp_min(1e-12)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    effective_rank = entropy.exp().item()
    return {"feature_std": feature_std, "effective_rank": effective_rank}


def learning_rate_factor(step, total_steps, warmup_steps):
    if step < warmup_steps:
        return float(step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def optimizer_to(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def validate_resume_config(saved_config, args):
    critical_keys = (
        "model",
        "r",
        "alpha",
        "last_n_blocks",
        "proj_hidden_dim",
        "proj_output_dim",
        "pred_dim",
        "batch_size",
        "grad_accum",
    )
    mismatches = {
        key: (saved_config.get(key), getattr(args, key))
        for key in critical_keys
        if saved_config.get(key) != getattr(args, key)
    }
    if mismatches:
        raise ValueError(f"Resume configuration mismatch: {mismatches}")


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1: CIFAR-10 unlabeled SimSiam adaptation.")
    parser.add_argument("--data-dir", default="./CIFAR-10/data")
    parser.add_argument("--output-dir", default="./weights/stage1_seed0")
    parser.add_argument("--model", default="vit_base_patch16_224")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--last-n-blocks", type=int, default=6)
    parser.add_argument("--proj-hidden-dim", type=int, default=2048)
    parser.add_argument("--proj-output-dim", type=int, default=256)
    parser.add_argument("--pred-dim", type=int, default=512)
    parser.add_argument("--variance-weight", type=float, default=0.05)
    parser.add_argument("--covariance-weight", type=float, default=0.005)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--grad-checkpointing", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size < 2:
        raise ValueError("SimSiam requires batch_size >= 2 because it uses BatchNorm.")
    if args.grad_accum < 1:
        raise ValueError("grad_accum must be >= 1.")

    set_seed(args.seed)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda" and not args.no_amp
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = build_ssl_transform(img_size=args.img_size)
    base_dataset = cifar10_dataset(
        args.data_dir,
        train=True,
        transform=None,
        download=args.download,
    )
    dataset = TwoViewUnlabeledDataset(base_dataset, transform)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        worker_init_fn=seed_worker,
        generator=generator,
    )

    model = ViTWithLoRA(
        model_name=args.model,
        pretrained=not args.no_pretrained,
        r=args.r,
        lora_alpha=args.alpha,
        last_n_blocks=args.last_n_blocks,
    ).to(device)
    if args.grad_checkpointing:
        model.backbone.set_grad_checkpointing(enable=True)

    ssl_head = SimSiamHead(
        input_dim=model.backbone.num_features,
        hidden_dim=args.proj_hidden_dim,
        output_dim=args.proj_output_dim,
        pred_dim=args.pred_dim,
    ).to(device)
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    trainable_parameters.extend(ssl_head.parameters())
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    updates_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_steps = updates_per_epoch * args.epochs
    warmup_steps = updates_per_epoch * args.warmup_epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: learning_rate_factor(step, total_steps, warmup_steps),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    start_epoch = 0
    global_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint.get("stage") != "stage1_ssl":
            raise ValueError(f"Not a stage1 checkpoint: {args.resume}")
        validate_resume_config(checkpoint["config"], args)
        load_lora_state(model, checkpoint["lora_state"])
        ssl_head.load_state_dict(checkpoint["ssl_head_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        optimizer_to(optimizer, device)
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint["global_step"])
        print(f"Resumed {args.resume} at epoch={start_epoch}, step={global_step}")

    config = vars(args).copy()
    config["pretrained"] = not args.no_pretrained
    config["lora_targets"] = model.lora_targets
    config["dataset_size"] = len(dataset)
    print(f"Device: {device} | AMP: {amp_enabled} | samples: {len(dataset)}")
    print(f"LoRA targets: {len(model.lora_targets)} layers")
    print(f"Updates/epoch: {updates_per_epoch} | effective batch: {args.batch_size * args.grad_accum}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        ssl_head.train()
        optimizer.zero_grad(set_to_none=True)
        totals = {"loss": 0.0, "sim": 0.0, "var": 0.0, "cov": 0.0}
        feature_batches = []
        progress = tqdm(loader, desc=f"Stage1 {epoch + 1}/{args.epochs}")

        for batch_index, (view1, view2) in enumerate(progress):
            view1 = view1.to(device, non_blocking=True)
            view2 = view2.to(device, non_blocking=True)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                features1 = model(view1)
                features2 = model(view2)
                prediction1, projection1 = ssl_head(features1)
                prediction2, projection2 = ssl_head(features2)
                loss_sim = simsiam_loss(
                    prediction1,
                    projection1,
                    prediction2,
                    projection2,
                )
                loss_var, loss_cov = variance_covariance_loss(projection1, projection2)
                loss = (
                    loss_sim
                    + args.variance_weight * loss_var
                    + args.covariance_weight * loss_cov
                )
                scaled_loss = loss / args.grad_accum

            scaler.scale(scaled_loss).backward()
            should_update = (
                (batch_index + 1) % args.grad_accum == 0
                or batch_index + 1 == len(loader)
            )
            if should_update:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            totals["loss"] += loss.detach().item()
            totals["sim"] += loss_sim.detach().item()
            totals["var"] += loss_var.detach().item()
            totals["cov"] += loss_cov.detach().item()
            if sum(batch.size(0) for batch in feature_batches) < 2048:
                feature_batches.append(features1.detach().float().cpu())
            progress.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        batches = len(loader)
        metrics = {key: value / batches for key, value in totals.items()}
        metrics.update(representation_diagnostics(feature_batches))
        metrics["lr"] = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch + 1}: loss={metrics['loss']:.4f} "
            f"std={metrics['feature_std']:.5f} "
            f"effective_rank={metrics['effective_rank']:.1f}"
        )

        latest_path = output_dir / "latest.pth"
        save_stage1_checkpoint(
            latest_path,
            model,
            ssl_head,
            optimizer,
            scheduler,
            scaler,
            epoch=epoch + 1,
            global_step=global_step,
            config=config,
            metrics=metrics,
        )
        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
            save_stage1_checkpoint(
                output_dir / f"epoch_{epoch + 1:03d}.pth",
                model,
                ssl_head,
                optimizer,
                scheduler,
                scaler,
                epoch=epoch + 1,
                global_step=global_step,
                config=config,
                metrics=metrics,
            )

    print(f"Stage 1 complete: {output_dir / 'latest.pth'}")


if __name__ == "__main__":
    main()

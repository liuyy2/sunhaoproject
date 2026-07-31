import argparse
import copy
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, Subset
from tqdm import tqdm

from checkpointing import load_lora_state, save_stage2_checkpoint
from data_protocol import (
    LabeledTwoViewDataset,
    WeakStrongUnlabeledDataset,
    build_strong_transform,
    build_weak_transform,
    cifar10_dataset,
    load_protocol_split,
    seed_worker,
)
from lora_vit import ViTWithLoRA
from utils import set_seed


class ContrastiveHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features):
        return F.normalize(self.net(features), dim=1)


def supervised_contrastive_loss(view1, view2, labels, temperature=0.1):
    features = F.normalize(torch.cat([view1, view2], dim=0).float(), dim=1)
    labels = torch.cat([labels, labels], dim=0)
    same_class = labels[:, None].eq(labels[None, :])
    nonself = ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positive_mask = same_class & nonself

    logits = features @ features.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * nonself
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    positives = positive_mask.sum(dim=1).clamp_min(1)
    return -((log_prob * positive_mask).sum(dim=1) / positives).mean()


@torch.no_grad()
def disable_lora_merging(model):
    for module in model.modules():
        if hasattr(module, "merge_weights"):
            module.merge_weights = False


@torch.no_grad()
def update_ema(ema_model, model, ema_classifier, classifier, decay):
    model_parameters = dict(model.named_parameters())
    for name, ema_parameter in ema_model.named_parameters():
        if "lora_" in name:
            ema_parameter.mul_(decay).add_(model_parameters[name], alpha=1.0 - decay)

    for ema_parameter, parameter in zip(
        ema_classifier.parameters(), classifier.parameters(), strict=True
    ):
        ema_parameter.mul_(decay).add_(parameter, alpha=1.0 - decay)


def lr_factor(step, total_steps, warmup_steps):
    if step < warmup_steps:
        return float(step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def optimizer_to(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 2: 1% labels with SupCon, FixMatch, and EMA."
    )
    parser.add_argument("--data-dir", default="./CIFAR-10/data")
    parser.add_argument("--split-path", default="./splits/cifar10_1pct_seed0.json")
    parser.add_argument("--stage1-checkpoint", default="./weights/stage1_seed0/latest.pth")
    parser.add_argument("--output-dir", default="./weights/stage2_seed0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--unlabeled-ramp-epochs", type=int, default=20)
    parser.add_argument("--labeled-batch-size", type=int, default=32)
    parser.add_argument("--unlabeled-batch-size", type=int, default=64)
    parser.add_argument("--lora-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--supcon-weight", type=float, default=0.2)
    parser.add_argument("--supcon-temperature", type=float, default=0.1)
    parser.add_argument("--unlabeled-weight", type=float, default=1.0)
    parser.add_argument("--pseudo-threshold", type=float, default=0.95)
    parser.add_argument("--distribution-momentum", type=float, default=0.999)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--contrast-hidden-dim", type=int, default=512)
    parser.add_argument("--contrast-output-dim", type=int, default=128)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def validate_stage1(stage1, split, args):
    if stage1.get("stage") != "stage1_ssl":
        raise ValueError("--stage1-checkpoint is not a stage 1 SSL checkpoint.")
    if stage1.get("epoch", 0) < stage1["config"]["epochs"]:
        raise ValueError("Stage 1 checkpoint has not completed its configured epochs.")
    if split["seed"] != args.seed:
        raise ValueError(
            f"Split seed {split['seed']} does not match training seed {args.seed}."
        )


def validate_resume(checkpoint, args):
    if checkpoint.get("stage") != "stage2_semi":
        raise ValueError("--resume is not a stage 2 checkpoint.")
    critical = (
        "stage1_checkpoint",
        "split_path",
        "labeled_batch_size",
        "unlabeled_batch_size",
        "contrast_hidden_dim",
        "contrast_output_dim",
        "seed",
    )
    mismatches = {
        key: (checkpoint["config"].get(key), getattr(args, key))
        for key in critical
        if checkpoint["config"].get(key) != getattr(args, key)
    }
    if mismatches:
        raise ValueError(f"Resume configuration mismatch: {mismatches}")


def main():
    args = parse_args()
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda" and not args.no_amp
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split = load_protocol_split(args.split_path)
    stage1 = torch.load(
        args.stage1_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    validate_stage1(stage1, split, args)
    stage1_config = stage1["config"]

    base_dataset = cifar10_dataset(args.data_dir, train=True, transform=None)
    if len(base_dataset) != split["train_size"]:
        raise ValueError("CIFAR-10 train size does not match the saved protocol.")
    weak_transform = build_weak_transform(args.img_size)
    strong_transform = build_strong_transform(args.img_size)
    labeled_dataset = LabeledTwoViewDataset(
        Subset(base_dataset, split["labeled_indices"]),
        weak_transform,
        strong_transform,
    )
    unlabeled_dataset = WeakStrongUnlabeledDataset(
        Subset(base_dataset, split["unlabeled_indices"]),
        weak_transform,
        strong_transform,
    )

    unlabeled_generator = torch.Generator().manual_seed(args.seed + 1)
    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        batch_size=args.unlabeled_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        worker_init_fn=seed_worker,
        generator=unlabeled_generator,
    )
    labeled_generator = torch.Generator().manual_seed(args.seed + 2)
    labeled_sampler = RandomSampler(
        labeled_dataset,
        replacement=True,
        num_samples=len(unlabeled_loader) * args.labeled_batch_size,
        generator=labeled_generator,
    )
    labeled_loader = DataLoader(
        labeled_dataset,
        batch_size=args.labeled_batch_size,
        sampler=labeled_sampler,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        worker_init_fn=seed_worker,
    )

    model = ViTWithLoRA(
        model_name=stage1_config["model"],
        pretrained=stage1_config["pretrained"],
        r=stage1_config["r"],
        lora_alpha=stage1_config["alpha"],
        last_n_blocks=stage1_config["last_n_blocks"],
    ).to(device)
    load_lora_state(model, stage1["lora_state"])
    feature_dim = model.backbone.num_features
    classifier = nn.Linear(feature_dim, split["num_classes"]).to(device)
    contrastive_head = ContrastiveHead(
        feature_dim,
        hidden_dim=args.contrast_hidden_dim,
        output_dim=args.contrast_output_dim,
    ).to(device)

    ema_model = copy.deepcopy(model).to(device)
    disable_lora_merging(ema_model)
    ema_classifier = copy.deepcopy(classifier).to(device)
    for parameter in ema_model.parameters():
        parameter.requires_grad = False
    for parameter in ema_classifier.parameters():
        parameter.requires_grad = False
    ema_model.eval()
    ema_classifier.eval()

    lora_parameters = [p for p in model.parameters() if p.requires_grad]
    head_parameters = list(classifier.parameters()) + list(contrastive_head.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_parameters, "lr": args.lora_lr},
            {"params": head_parameters, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = len(unlabeled_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_factor(step, total_steps, warmup_steps),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    model_probability = torch.full(
        (split["num_classes"],),
        1.0 / split["num_classes"],
        device=device,
    )

    start_epoch = 0
    global_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        validate_resume(checkpoint, args)
        load_lora_state(model, checkpoint["lora_state"])
        classifier.load_state_dict(checkpoint["classifier_state"])
        contrastive_head.load_state_dict(checkpoint["contrastive_head_state"])
        load_lora_state(ema_model, checkpoint["ema_lora_state"])
        ema_classifier.load_state_dict(checkpoint["ema_classifier_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        optimizer_to(optimizer, device)
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        model_probability.copy_(checkpoint["model_probability"].to(device))
        start_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint["global_step"])
        print(f"Resumed {args.resume} at epoch={start_epoch}, step={global_step}")

    config = vars(args).copy()
    config.update(
        {
            "model": stage1_config["model"],
            "pretrained": stage1_config["pretrained"],
            "r": stage1_config["r"],
            "alpha": stage1_config["alpha"],
            "last_n_blocks": stage1_config["last_n_blocks"],
            "labeled_count": len(labeled_dataset),
            "unlabeled_count": len(unlabeled_dataset),
        }
    )
    print(
        f"Device: {device} | labeled={len(labeled_dataset)} | "
        f"unlabeled={len(unlabeled_dataset)} | steps/epoch={steps_per_epoch}"
    )

    for epoch in range(start_epoch, args.epochs):
        model.train()
        classifier.train()
        contrastive_head.train()
        totals = {"loss": 0.0, "ce": 0.0, "supcon": 0.0, "fixmatch": 0.0}
        labeled_correct = 0
        labeled_total = 0
        accepted_total = 0
        unlabeled_total = 0
        pseudo_histogram = torch.zeros(split["num_classes"], device=device)
        labeled_iterator = iter(labeled_loader)
        ramp = min(1.0, float(epoch + 1) / max(1, args.unlabeled_ramp_epochs))
        current_unlabeled_weight = args.unlabeled_weight * ramp
        progress = tqdm(unlabeled_loader, desc=f"Stage2 {epoch + 1}/{args.epochs}")

        for weak_unlabeled, strong_unlabeled in progress:
            weak_labeled, strong_labeled, labels = next(labeled_iterator)
            weak_labeled = weak_labeled.to(device, non_blocking=True)
            strong_labeled = strong_labeled.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            weak_unlabeled = weak_unlabeled.to(device, non_blocking=True)
            strong_unlabeled = strong_unlabeled.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                teacher_features = ema_model(weak_unlabeled)
                teacher_probabilities = F.softmax(
                    ema_classifier(teacher_features).float(), dim=1
                )
                batch_probability = teacher_probabilities.mean(dim=0)
                model_probability.mul_(args.distribution_momentum).add_(
                    batch_probability,
                    alpha=1.0 - args.distribution_momentum,
                )
                aligned = teacher_probabilities / model_probability.clamp_min(1e-6)
                aligned = aligned / aligned.sum(dim=1, keepdim=True)
                confidence, pseudo_labels = aligned.max(dim=1)
                pseudo_mask = confidence.ge(args.pseudo_threshold)

            student_inputs = torch.cat(
                [weak_labeled, strong_labeled, strong_unlabeled], dim=0
            )
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                student_features = model(student_inputs)
                labeled_batch = labels.size(0)
                weak_features = student_features[:labeled_batch]
                strong_features = student_features[labeled_batch : 2 * labeled_batch]
                unlabeled_features = student_features[2 * labeled_batch :]

                weak_logits = classifier(weak_features)
                strong_unlabeled_logits = classifier(unlabeled_features)
                loss_ce = F.cross_entropy(weak_logits, labels)
                weak_projection = contrastive_head(weak_features)
                strong_projection = contrastive_head(strong_features)
                loss_supcon = supervised_contrastive_loss(
                    weak_projection,
                    strong_projection,
                    labels,
                    temperature=args.supcon_temperature,
                )
                per_sample_fixmatch = F.cross_entropy(
                    strong_unlabeled_logits,
                    pseudo_labels,
                    reduction="none",
                )
                loss_fixmatch = (
                    per_sample_fixmatch * pseudo_mask.to(per_sample_fixmatch.dtype)
                ).mean()
                loss = (
                    loss_ce
                    + args.supcon_weight * loss_supcon
                    + current_unlabeled_weight * loss_fixmatch
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            ema_decay = min(args.ema_decay, 1.0 - 1.0 / (global_step + 1.0))
            update_ema(ema_model, model, ema_classifier, classifier, ema_decay)

            totals["loss"] += loss.detach().item()
            totals["ce"] += loss_ce.detach().item()
            totals["supcon"] += loss_supcon.detach().item()
            totals["fixmatch"] += loss_fixmatch.detach().item()
            labeled_correct += weak_logits.detach().argmax(dim=1).eq(labels).sum().item()
            labeled_total += labels.size(0)
            accepted_total += pseudo_mask.sum().item()
            unlabeled_total += pseudo_mask.numel()
            if pseudo_mask.any():
                pseudo_histogram += torch.bincount(
                    pseudo_labels[pseudo_mask], minlength=split["num_classes"]
                )
            progress.set_postfix(
                loss=f"{loss.item():.3f}",
                accept=f"{accepted_total / unlabeled_total:.2%}",
            )

        metrics = {key: value / steps_per_epoch for key, value in totals.items()}
        metrics["labeled_accuracy"] = labeled_correct / labeled_total
        metrics["pseudo_acceptance"] = accepted_total / unlabeled_total
        metrics["pseudo_histogram"] = pseudo_histogram.long().cpu().tolist()
        metrics["lora_lr"] = scheduler.get_last_lr()[0]
        metrics["head_lr"] = scheduler.get_last_lr()[1]
        print(
            f"Epoch {epoch + 1}: loss={metrics['loss']:.4f} "
            f"labeled_acc={metrics['labeled_accuracy']:.4f} "
            f"pseudo_accept={metrics['pseudo_acceptance']:.2%}"
        )
        print(f"Pseudo histogram: {metrics['pseudo_histogram']}")

        save_args = dict(
            model=model,
            classifier=classifier,
            contrastive_head=contrastive_head,
            ema_model=ema_model,
            ema_classifier=ema_classifier,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            model_probability=model_probability,
            epoch=epoch + 1,
            global_step=global_step,
            config=config,
            metrics=metrics,
        )
        save_stage2_checkpoint(output_dir / "latest.pth", **save_args)
        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
            save_stage2_checkpoint(
                output_dir / f"epoch_{epoch + 1:03d}.pth", **save_args
            )

    print(f"Stage 2 complete: {output_dir / 'latest.pth'}")


if __name__ == "__main__":
    main()

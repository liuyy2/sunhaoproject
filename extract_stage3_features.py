import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from checkpointing import load_lora_state
from data_protocol import build_multiscale_eval_transform, cifar10_dataset
from lora_vit import ViTWithLoRA


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract deterministic EMA features from a stage 2 checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        default="./weights/stage2_seed0/latest.pth",
    )
    parser.add_argument("--data-dir", default="./CIFAR-10/data")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument(
        "--output",
        default="./features/stage2_seed0_test.npz",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional smoke-test limit; omit for final extraction.",
    )
    parser.add_argument(
        "--weights",
        choices=["ema", "student", "ensemble"],
        default="ema",
    )
    parser.add_argument(
        "--tta-scales",
        type=int,
        nargs="+",
        default=[224],
        help="Deterministic resize scales followed by a center crop.",
    )
    parser.add_argument(
        "--tta-flip",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def build_model_and_classifier(checkpoint, state_prefix, device):
    config = checkpoint["config"]
    model = ViTWithLoRA(
        model_name=config["model"],
        pretrained=config["pretrained"],
        r=config["r"],
        lora_alpha=config["alpha"],
        last_n_blocks=config["last_n_blocks"],
    )
    classifier = nn.Linear(model.backbone.num_features, 10)
    if state_prefix == "ema":
        load_lora_state(model, checkpoint["ema_lora_state"])
        classifier.load_state_dict(checkpoint["ema_classifier_state"])
    else:
        load_lora_state(model, checkpoint["lora_state"])
        classifier.load_state_dict(checkpoint["classifier_state"])
    return model.to(device).eval(), classifier.to(device).eval()


def load_stage2_models(checkpoint_path, weights, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("stage") != "stage2_semi":
        raise ValueError(f"Not a stage 2 checkpoint: {checkpoint_path}")

    state_prefixes = ["ema", "student"] if weights == "ensemble" else [weights]
    models = [
        build_model_and_classifier(checkpoint, state_prefix, device)
        for state_prefix in state_prefixes
    ]
    return models, checkpoint


def load_stage2_model(checkpoint_path, weights, device):
    """Backward-compatible single-model loader used by existing smoke checks."""
    if weights == "ensemble":
        raise ValueError("Use load_stage2_models for ensemble extraction.")
    models, checkpoint = load_stage2_models(checkpoint_path, weights, device)
    model, classifier = models[0]
    return model, classifier, checkpoint


@torch.inference_mode()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, checkpoint = load_stage2_models(
        args.checkpoint,
        args.weights,
        device,
    )

    img_size = checkpoint["config"].get("img_size", 224)
    transform = build_multiscale_eval_transform(
        img_size=img_size,
        resize_scales=args.tta_scales,
    )
    dataset = cifar10_dataset(
        args.data_dir,
        train=(args.split == "train"),
        transform=transform,
    )
    if args.max_samples is not None:
        if not 1 <= args.max_samples <= len(dataset):
            raise ValueError(
                f"max_samples must be in [1, {len(dataset)}], got {args.max_samples}."
            )
        dataset = Subset(dataset, range(args.max_samples))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    feature_chunks = []
    logit_chunks = []
    probability_chunks = []
    label_chunks = []
    for image_views, labels in tqdm(
        loader, desc=f"Extract {args.split} ({args.weights})"
    ):
        image_views = image_views.to(device, non_blocking=True)
        feature_sum = None
        logit_sum = None
        probability_sum = None
        prediction_count = 0

        for model, classifier in models:
            for view_index in range(image_views.size(1)):
                images = image_views[:, view_index]
                variants = [images]
                if args.tta_flip:
                    variants.append(torch.flip(images, dims=[3]))
                for variant in variants:
                    raw_features = model(variant).float()
                    normalized_features = F.normalize(raw_features, dim=1)
                    current_logits = classifier(raw_features)
                    current_probabilities = F.softmax(current_logits, dim=1)
                    feature_sum = (
                        normalized_features
                        if feature_sum is None
                        else feature_sum + normalized_features
                    )
                    logit_sum = (
                        current_logits if logit_sum is None else logit_sum + current_logits
                    )
                    probability_sum = (
                        current_probabilities
                        if probability_sum is None
                        else probability_sum + current_probabilities
                    )
                    prediction_count += 1

        features = F.normalize(feature_sum / prediction_count, dim=1)
        logits = logit_sum / prediction_count
        probabilities = probability_sum / prediction_count

        feature_chunks.append(features.cpu().numpy())
        logit_chunks.append(logits.cpu().numpy())
        probability_chunks.append(probabilities.cpu().numpy())
        label_chunks.append(labels.numpy())

    features = np.concatenate(feature_chunks).astype(np.float32, copy=False)
    logits = np.concatenate(logit_chunks).astype(np.float32, copy=False)
    probabilities = np.concatenate(probability_chunks).astype(np.float32, copy=False)
    labels = np.concatenate(label_chunks).astype(np.int64, copy=False)
    predictions = probabilities.argmax(axis=1).astype(np.int64, copy=False)
    metadata = {
        "checkpoint": str(Path(args.checkpoint)),
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_stage": checkpoint["stage"],
        "weights": args.weights,
        "split": args.split,
        "tta_flip": args.tta_flip,
        "tta_scales": args.tta_scales,
        "ensemble_members": len(models),
        "feature_normalized": True,
        "test_labels_used_during_extraction": False,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=features,
        logits=logits,
        probs=probabilities,
        pred=predictions,
        y=labels,
        metadata=np.asarray(json.dumps(metadata)),
    )
    print(
        f"Saved {output_path} | X={features.shape} | "
        f"logits={logits.shape} | weights={args.weights}"
    )


if __name__ == "__main__":
    main()

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from data_protocol import prepare_cifar10_split


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run reproducible stage 2 training and stage 3 evaluation."
    )
    parser.add_argument(
        "--mode",
        choices=["prepare", "train", "evaluate", "all"],
        required=True,
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--data-dir", default="./CIFAR-10/data")
    parser.add_argument(
        "--stage1-checkpoint",
        default="./weights/stage1_seed0/latest.pth",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--extract-batch-size", type=int, default=128)
    parser.add_argument("--force-evaluate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_command(command, dry_run):
    printable = subprocess.list2cmdline([str(part) for part in command])
    print(f"\n[COMMAND] {printable}", flush=True)
    if not dry_run:
        subprocess.run(
            [str(part) for part in command],
            cwd=PROJECT_ROOT,
            check=True,
        )


def load_checkpoint_metadata(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "stage": checkpoint.get("stage"),
        "epoch": int(checkpoint.get("epoch", 0)),
        "seed": checkpoint.get("config", {}).get("seed"),
    }


def prepare_seed_split(seed, data_dir, dry_run):
    split_path = PROJECT_ROOT / "splits" / f"cifar10_1pct_seed{seed}.json"
    if dry_run:
        print(f"[DRY RUN] prepare split: {split_path}")
        return split_path

    payload = prepare_cifar10_split(
        data_dir=PROJECT_ROOT / data_dir,
        split_path=split_path,
        label_ratio=0.01,
        seed=seed,
        download=False,
    )
    print(
        f"[SPLIT seed={seed}] labeled={payload['labeled_count']} "
        f"per_class={payload['labeled_per_class']}"
    )
    return split_path


def train_seed(args, seed):
    split_path = prepare_seed_split(seed, args.data_dir, args.dry_run)
    output_dir = PROJECT_ROOT / "weights" / f"stage2_seed{seed}"
    latest_path = output_dir / "latest.pth"

    resume = False
    if latest_path.exists() and not args.dry_run:
        metadata = load_checkpoint_metadata(latest_path)
        if metadata["stage"] != "stage2_semi" or metadata["seed"] != seed:
            raise ValueError(
                f"Incompatible checkpoint at {latest_path}: {metadata}"
            )
        if metadata["epoch"] >= args.epochs:
            print(
                f"[SKIP seed={seed}] stage 2 already complete "
                f"at epoch {metadata['epoch']}."
            )
            return
        resume = True

    command = [
        sys.executable,
        PROJECT_ROOT / "train_stage2_semi.py",
        "--data-dir",
        args.data_dir,
        "--split-path",
        split_path,
        "--stage1-checkpoint",
        args.stage1_checkpoint,
        "--output-dir",
        output_dir,
        "--epochs",
        args.epochs,
        "--warmup-epochs",
        5,
        "--unlabeled-ramp-epochs",
        20,
        "--labeled-batch-size",
        32,
        "--unlabeled-batch-size",
        64,
        "--num-workers",
        args.num_workers,
        "--seed",
        seed,
    ]
    if resume:
        command.extend(["--resume", latest_path])
    run_command(command, args.dry_run)


def require_completed_stage2(seed, epochs):
    checkpoint_path = PROJECT_ROOT / "weights" / f"stage2_seed{seed}" / "latest.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing stage 2 checkpoint for seed {seed}: {checkpoint_path}"
        )
    metadata = load_checkpoint_metadata(checkpoint_path)
    if (
        metadata["stage"] != "stage2_semi"
        or metadata["seed"] != seed
        or metadata["epoch"] < epochs
    ):
        raise ValueError(
            f"Stage 2 checkpoint for seed {seed} is incomplete: {metadata}"
        )
    return checkpoint_path


def evaluate_seed(args, seed):
    if args.dry_run:
        checkpoint_path = (
            PROJECT_ROOT / "weights" / f"stage2_seed{seed}" / "latest.pth"
        )
    else:
        checkpoint_path = require_completed_stage2(seed, args.epochs)

    feature_path = (
        PROJECT_ROOT / "features" / f"stage2_seed{seed}_ensemble_ms_test.npz"
    )
    result_path = (
        PROJECT_ROOT
        / "results"
        / f"stage2_seed{seed}_ensemble_ms_constrained_test.json"
    )

    if args.force_evaluate or not feature_path.exists():
        extract_command = [
            sys.executable,
            PROJECT_ROOT / "extract_stage3_features.py",
            "--checkpoint",
            checkpoint_path,
            "--data-dir",
            args.data_dir,
            "--split",
            "test",
            "--weights",
            "ensemble",
            "--tta-scales",
            224,
            240,
            256,
            "--batch-size",
            args.extract_batch_size,
            "--num-workers",
            args.num_workers,
            "--output",
            feature_path,
        ]
        run_command(extract_command, args.dry_run)
    else:
        print(f"[SKIP seed={seed}] features already exist: {feature_path}")

    if args.force_evaluate or not result_path.exists():
        evaluate_command = [
            sys.executable,
            PROJECT_ROOT / "evaluate_stage3.py",
            "--features",
            feature_path,
            "--clusters",
            10,
            "--seed",
            seed,
            "--skip-anchor",
            "--output-json",
            result_path,
        ]
        run_command(evaluate_command, args.dry_run)
    else:
        print(f"[SKIP seed={seed}] result already exists: {result_path}")


def write_manifest(args):
    manifest_path = PROJECT_ROOT / "results" / "multiseed_protocol.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seeds": args.seeds,
        "shared_stage1_checkpoint": args.stage1_checkpoint,
        "shared_stage1_seed": 0,
        "stage2_epochs": args.epochs,
        "labels_per_seed": 500,
        "labels_per_class": 50,
        "evaluation": {
            "weights": "ema_student_ensemble",
            "tta_scales": [224, 240, 256],
            "tta_flip": True,
            "primary_method": "BalancedKMeans-Logits",
            "known_test_class_prior": 1000,
        },
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[MANIFEST] {manifest_path}")


def main():
    args = parse_args()
    if not args.seeds or any(seed < 0 for seed in args.seeds):
        raise ValueError("--seeds must contain non-negative integers.")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must not contain duplicates.")

    if not args.dry_run:
        stage1_path = PROJECT_ROOT / args.stage1_checkpoint
        if not stage1_path.exists():
            raise FileNotFoundError(f"Missing stage 1 checkpoint: {stage1_path}")
        write_manifest(args)

    if args.mode == "prepare":
        for seed in args.seeds:
            prepare_seed_split(seed, args.data_dir, args.dry_run)
    if args.mode in {"train", "all"}:
        for seed in args.seeds:
            train_seed(args, seed)
    if args.mode in {"evaluate", "all"}:
        for seed in args.seeds:
            evaluate_seed(args, seed)

    print(f"\nCompleted mode={args.mode} for seeds={args.seeds}.")


if __name__ == "__main__":
    main()

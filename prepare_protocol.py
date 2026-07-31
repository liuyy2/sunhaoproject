import argparse

from data_protocol import prepare_cifar10_split


def main():
    parser = argparse.ArgumentParser(description="Create the fixed CIFAR-10 1% label split.")
    parser.add_argument("--data-dir", default="./CIFAR-10/data")
    parser.add_argument("--split-path", default="./splits/cifar10_1pct_seed0.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    payload = prepare_cifar10_split(
        data_dir=args.data_dir,
        split_path=args.split_path,
        label_ratio=0.01,
        seed=args.seed,
        download=args.download,
    )
    print(f"Split: {args.split_path}")
    print(f"Labeled: {payload['labeled_count']} / {payload['train_size']}")
    print(f"Per class: {payload['labeled_per_class']}")


if __name__ == "__main__":
    main()

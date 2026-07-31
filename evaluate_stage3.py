import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from anchor_spec import anchor_spectral_clustering


def clustering_accuracy(y_true, y_pred):
    true_labels = np.asarray(y_true, dtype=np.int64)
    predicted_labels = np.asarray(y_pred, dtype=np.int64)
    size = int(max(true_labels.max(), predicted_labels.max()) + 1)
    contingency = np.zeros((size, size), dtype=np.int64)
    np.add.at(contingency, (predicted_labels, true_labels), 1)
    rows, columns = linear_sum_assignment(contingency.max() - contingency)
    return contingency[rows, columns].sum() / len(true_labels)


def metrics(y_true, y_pred, aligned):
    accuracy = (
        np.mean(np.asarray(y_true) == np.asarray(y_pred))
        if aligned
        else clustering_accuracy(y_true, y_pred)
    )
    return {
        "ACC": float(accuracy),
        "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
        "ARI": float(adjusted_rand_score(y_true, y_pred)),
    }


def print_result(name, result):
    print(
        f"{name:24s} | ACC={result['ACC']:.4f} "
        f"NMI={result['NMI']:.4f} ARI={result['ARI']:.4f}"
    )


def kmeans_labels(features, clusters, seed):
    return KMeans(
        n_clusters=clusters,
        n_init=50,
        random_state=seed,
    ).fit_predict(features)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Leakage-controlled stage 3 clustering evaluation."
    )
    parser.add_argument(
        "--features",
        default="./features/stage2_seed0_test.npz",
    )
    parser.add_argument("--clusters", type=int, default=10)
    parser.add_argument("--anchors", type=int, default=1000)
    parser.add_argument("--anchor-neighbors", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-anchor", action="store_true")
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    data = np.load(args.features, allow_pickle=False)
    required = {"X", "logits", "probs", "pred", "y", "metadata"}
    missing = sorted(required - set(data.files))
    if missing:
        raise ValueError(f"Feature file is missing arrays: {missing}")

    features = data["X"].astype(np.float32, copy=False)
    logits = data["logits"].astype(np.float32, copy=False)
    probabilities = data["probs"].astype(np.float32, copy=False)
    predictions = data["pred"].astype(np.int64, copy=False)
    labels = data["y"].astype(np.int64, copy=False)
    metadata = json.loads(str(data["metadata"].item()))

    if metadata.get("split") != "test":
        raise ValueError("Final metrics must be computed on the CIFAR-10 test split.")
    if features.shape[0] != 10000 or len(labels) != 10000:
        raise ValueError(
            f"Expected 10,000 CIFAR-10 test examples, got {features.shape[0]}."
        )
    if args.clusters != 10:
        raise ValueError("The fixed CIFAR-10 protocol requires exactly 10 clusters.")

    # All predictions are produced before labels enter metric computation.
    predicted = {
        "EMAClassifier": (predictions, True),
        "KMeans-Backbone": (
            kmeans_labels(features, args.clusters, args.seed),
            False,
        ),
        "KMeans-Logits": (
            kmeans_labels(logits, args.clusters, args.seed),
            False,
        ),
        "KMeans-Probabilities": (
            kmeans_labels(probabilities, args.clusters, args.seed),
            False,
        ),
    }
    if not args.skip_anchor:
        predicted["Anchor-Backbone"] = (
            anchor_spectral_clustering(
                features,
                k=args.clusters,
                m=args.anchors,
                s=args.anchor_neighbors,
                seed=args.seed,
            ),
            False,
        )

    results = {
        name: metrics(labels, labels_pred, aligned)
        for name, (labels_pred, aligned) in predicted.items()
    }
    print(f"Feature metadata: {metadata}")
    for name, result in results.items():
        print_result(name, result)

    if args.output_json:
        payload = {
            "protocol": {
                "dataset": "cifar10",
                "test_size": len(labels),
                "labels_used_for_training": 500,
                "seed": args.seed,
                "anchors": args.anchors,
                "anchor_neighbors": args.anchor_neighbors,
            },
            "feature_metadata": metadata,
            "results": results,
        }
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"Saved metrics: {output_path}")


if __name__ == "__main__":
    main()

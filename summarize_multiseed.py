import argparse
import csv
import json
from pathlib import Path

import numpy as np


TARGETS = {"ACC": 0.981, "NMI": 0.954, "ARI": 0.959}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate seed-wise stage 3 evaluation JSON files."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument(
        "--method",
        default="BalancedKMeans-Logits",
    )
    parser.add_argument(
        "--output-json",
        default="./results/multiseed_summary.json",
    )
    parser.add_argument(
        "--output-csv",
        default="./results/multiseed_summary.csv",
    )
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def result_path(results_dir, seed):
    return (
        Path(results_dir)
        / f"stage2_seed{seed}_ensemble_ms_constrained_test.json"
    )


def main():
    args = parse_args()
    records = []
    missing = []
    all_method_records = []

    for seed in args.seeds:
        path = result_path(args.results_dir, seed)
        if not path.exists():
            missing.append(str(path))
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        results = payload.get("results", {})
        if args.method not in results:
            raise KeyError(f"{path} does not contain method {args.method!r}.")

        for method, values in results.items():
            all_method_records.append(
                {
                    "seed": seed,
                    "method": method,
                    "ACC": float(values["ACC"]),
                    "NMI": float(values["NMI"]),
                    "ARI": float(values["ARI"]),
                }
            )
        records.append(
            {
                "seed": seed,
                **{
                    metric: float(results[args.method][metric])
                    for metric in TARGETS
                },
            }
        )

    if missing and not args.allow_partial:
        raise FileNotFoundError(
            "Missing seed results:\n" + "\n".join(f"- {path}" for path in missing)
        )
    if not records:
        raise FileNotFoundError("No seed result files were found.")

    values = np.asarray(
        [[record[metric] for metric in TARGETS] for record in records],
        dtype=np.float64,
    )
    means = values.mean(axis=0)
    standard_deviations = (
        values.std(axis=0, ddof=1)
        if len(records) > 1
        else np.zeros(len(TARGETS), dtype=np.float64)
    )
    aggregate = {}
    for index, (metric, target) in enumerate(TARGETS.items()):
        aggregate[metric] = {
            "mean": float(means[index]),
            "std": float(standard_deviations[index]),
            "target": target,
            "mean_pass": bool(means[index] >= target),
        }

    aggregate_targets_pass = all(
        values["mean_pass"] for values in aggregate.values()
    )
    summary = {
        "method": args.method,
        "requested_seeds": args.seeds,
        "completed_seeds": [record["seed"] for record in records],
        "missing_files": missing,
        "complete": not missing,
        "per_seed": records,
        "aggregate": aggregate,
        "all_targets_pass": bool(not missing and aggregate_targets_pass),
        "protocol_note": (
            "Stage 1 seed 0 is shared. Stage 2 and the 1% labeled split vary by seed."
        ),
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["seed", "method", "ACC", "NMI", "ARI"],
        )
        writer.writeheader()
        writer.writerows(all_method_records)

    print(f"Primary method: {args.method}")
    for record in records:
        print(
            f"seed={record['seed']} | ACC={record['ACC']:.4f} "
            f"NMI={record['NMI']:.4f} ARI={record['ARI']:.4f}"
        )
    print("Aggregate:")
    for metric, values_for_metric in aggregate.items():
        print(
            f"{metric}: {values_for_metric['mean']:.4f} "
            f"+/- {values_for_metric['std']:.4f} "
            f"(target={values_for_metric['target']:.4f}, "
            f"pass={values_for_metric['mean_pass']})"
        )
    print(f"All targets pass: {summary['all_targets_pass']}")
    print(f"Saved: {output_json}")
    print(f"Saved: {output_csv}")


if __name__ == "__main__":
    main()

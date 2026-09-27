"""Hyperparameter sweep, tracked in MLflow.

Tracking is stored in `mlflow.db` (SQLite). Inspect it with
`mlflow ui --backend-store-uri sqlite:///mlflow.db` - without that flag MLflow looks
for an `mlruns/` directory instead and shows an empty UI.

A single training run tells you what one configuration scored. It does not tell you
whether that configuration mattered, which is the question MLflow exists to answer.
This sweep trains a small grid and reports sensitivity: how much the result actually
moves when a hyperparameter changes.

Sweep runs never overwrite `artifacts/metrics.json`. That file holds the canonical
result from the reference configuration, and a sweep is exploration, not a new headline.

Runtime is roughly (sum of epochs across the grid) x 90 seconds on an A100. The default
grid is about 40 minutes.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

ARTIFACTS = Path(__file__).parent / "artifacts"

# Learning rate is the hyperparameter most likely to matter for a fine-tune this short,
# so it gets the most points. Epochs are included to test whether the plateau observed
# in the reference run (validation F1 flat from epoch 6) holds across learning rates.
DEFAULT_GRID = {
    "lr": [1e-5, 3e-5, 5e-5, 1e-4],
    "epochs": [4.0, 8.0],
    "batch_size": [4],
}


def sensitivity(results: list[dict], key: str) -> dict:
    """How much does the score move across the values of one hyperparameter?"""
    groups: dict[float, list[float]] = {}
    for record in results:
        groups.setdefault(record[key], []).append(record["test_f1"])
    means = {value: statistics.mean(scores) for value, scores in groups.items()}
    values = {f"{k:g}": round(v, 4) for k, v in sorted(means.items())}
    if len(means) < 2:
        # Only one value was swept, so there is nothing to compare it against.
        return {"values": values, "spread": None, "note": "not swept"}
    return {
        "values": values,
        "spread": round(max(means.values()) - min(means.values()), 4),
        "best": f"{max(means, key=means.get):g}",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="tiny grid, for checking the plumbing")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="learning rate only, 4 epochs each (~25 min instead of ~70)",
    )
    args = parser.parse_args()

    from src.train_extractor import Config, run_training

    if args.quick:
        grid = {"lr": [3e-5, 5e-5], "epochs": [1.0], "batch_size": [4]}
    elif args.fast:
        # Learning rate carries most of the signal; dropping the epoch arm roughly
        # thirds the runtime at the cost of not testing the plateau claim.
        grid = {"lr": [1e-5, 3e-5, 5e-5, 1e-4], "epochs": [4.0], "batch_size": [4]}
    else:
        grid = DEFAULT_GRID
    combinations = [
        dict(zip(grid.keys(), values)) for values in itertools.product(*grid.values())
    ]

    total_epochs = sum(c["epochs"] for c in combinations)
    print(f"{len(combinations)} configurations, {total_epochs:.0f} epochs total")
    print(f"rough estimate: {total_epochs * 90 / 60:.0f} minutes on an A100\n")

    results = []
    for index, combo in enumerate(combinations, 1):
        name = f"lr{combo['lr']:g}-ep{combo['epochs']:g}-bs{combo['batch_size']}"
        print(f"[{index}/{len(combinations)}] {name}")
        started = time.time()

        metrics = run_training(
            Config(
                lr=combo["lr"],
                epochs=combo["epochs"],
                batch_size=combo["batch_size"],
                run_name=name,
                # Exploration must not clobber the canonical result or the saved model.
                write_artifacts=False,
                save_model=False,
                max_train=40 if args.quick else 0,
            )
        )

        results.append(
            {
                **combo,
                "run_name": name,
                "test_f1": metrics["test_f1"],
                "test_precision": metrics["test_precision"],
                "test_recall": metrics["test_recall"],
                "minutes": round((time.time() - started) / 60, 1),
            }
        )
        print(f"    F1 {metrics['test_f1']:.4f}  ({results[-1]['minutes']} min)\n")

    results.sort(key=lambda r: -r["test_f1"])
    best, worst = results[0], results[-1]

    summary = {
        "n_configurations": len(results),
        "results": results,
        "best": best,
        "spread_across_grid": round(best["test_f1"] - worst["test_f1"], 4),
        "sensitivity": {
            "learning_rate": sensitivity(results, "lr"),
            "epochs": sensitivity(results, "epochs"),
        },
        "note": (
            "One seed per configuration. Differences smaller than run-to-run seed "
            "variance should not be read as real."
        ),
    }

    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "sweep_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    rows = ["| lr | epochs | batch | F1 | precision | recall | min |",
            "|---:|---:|---:|---:|---:|---:|---:|"]
    rows += [
        f"| {r['lr']:g} | {r['epochs']:g} | {r['batch_size']} | **{r['test_f1']:.4f}** | "
        f"{r['test_precision']:.3f} | {r['test_recall']:.3f} | {r['minutes']} |"
        for r in results
    ]
    table = "\n".join(rows)
    (ARTIFACTS / "sweep_table.md").write_text(table, encoding="utf-8")

    print(table)
    print(f"\nbest: {best['run_name']} at F1 {best['test_f1']:.4f}")
    print(f"spread across the grid: {summary['spread_across_grid']:.4f}")
    print(f"learning-rate sensitivity: {summary['sensitivity']['learning_rate']}")
    print(f"epoch sensitivity:         {summary['sensitivity']['epochs']}")
    print(f"\nwritten to {ARTIFACTS / 'sweep_results.json'}")


if __name__ == "__main__":
    main()

"""Summarise the frozen B-score ablation artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENTS = (
    "antisymmetric_decoder",
    "antisymmetric_no_direct_bscore",
    "antisymmetric_no_node_bscore",
    "antisymmetric_no_bscore",
    "gbdt_tuned",
    "gbdt_no_bscore",
)


def prediction_metrics(path: Path) -> dict[str, float]:
    frame = pd.read_csv(path)
    y = frame["y_true"].to_numpy(dtype=float)
    probability = np.clip(
        frame["probability"].to_numpy(dtype=float), 1e-7, 1 - 1e-7
    )
    return {
        "n_matches": len(frame),
        "accuracy": np.mean((probability >= 0.5) == y),
        "log_loss": -np.mean(
            y * np.log(probability)
            + (1 - y) * np.log(1 - probability)
        ),
        "brier": np.mean((probability - y) ** 2),
    }


def collect(root: Path, scope: str, seeds: list[int]) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        directory = root / scope / f"seed_{seed}"
        for experiment in EXPERIMENTS:
            path = directory / f"{experiment}.csv"
            if path.is_file():
                rows.append(
                    {
                        "seed": seed,
                        "experiment": experiment,
                        **prediction_metrics(path),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scope",
        choices=("slams", "slams_masters"),
        default="slams_masters",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 2026]
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "results" / "frozen_predictions"
    results = collect(root, args.scope, args.seeds)
    if results.empty:
        raise SystemExit("No B-score ablation artifacts found.")
    print(results.to_string(index=False))
    print("\nMeans over available seeds")
    print(
        results.groupby("experiment")[
            ["accuracy", "log_loss", "brier"]
        ].mean().sort_values("log_loss").to_string()
    )


if __name__ == "__main__":
    main()

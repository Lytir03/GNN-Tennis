"""Aggregate frozen baseline/candidate artifacts across random seeds."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    from .experiment_tracking import (
        assert_compatible,
        load_prediction_artifact,
        probability_metrics,
    )
except ImportError:
    from experiment_tracking import (
        assert_compatible,
        load_prediction_artifact,
        probability_metrics,
    )


def multi_seed_comparison(
    results_root: str | Path,
    *,
    scope: str,
    seeds: Iterable[int],
    experiments: Iterable[str] = (
        "bscore_logit",
        "current_gnn",
        "antisymmetric_decoder",
    ),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-seed metrics and aggregate mean/std/win counts."""

    results_root = Path(results_root)
    experiments = tuple(experiments)
    rows = []
    for seed in seeds:
        directory = results_root / scope / f"seed_{seed}"
        loaded = {
            name: load_prediction_artifact(directory, name)
            for name in experiments
        }
        assert_compatible([manifest for _, manifest in loaded.values()])
        for name, (frame, manifest) in loaded.items():
            metrics = probability_metrics(frame[frame["phase"] == "test"])
            rows.append(
                {
                    "seed": seed,
                    "experiment": name,
                    **metrics,
                    "evaluation_hash": manifest["evaluation_hash"],
                }
            )

    per_seed = pd.DataFrame(rows)
    metric_columns = ["accuracy", "log_loss", "brier"]
    aggregate = (
        per_seed.groupby("experiment")[metric_columns]
        .agg(["mean", "std"])
    )
    aggregate.columns = [
        f"{metric}_{statistic}"
        for metric, statistic in aggregate.columns
    ]

    pivot = per_seed.pivot(
        index="seed", columns="experiment", values=metric_columns
    )
    candidate = "antisymmetric_decoder"
    baseline = "current_gnn"
    if candidate in experiments and baseline in experiments:
        aggregate["accuracy_wins_vs_current"] = pd.NA
        aggregate["log_loss_wins_vs_current"] = pd.NA
        aggregate["brier_wins_vs_current"] = pd.NA
        aggregate.loc[candidate, "accuracy_wins_vs_current"] = int(
            (pivot["accuracy"][candidate] > pivot["accuracy"][baseline]).sum()
        )
        aggregate.loc[candidate, "log_loss_wins_vs_current"] = int(
            (pivot["log_loss"][candidate] < pivot["log_loss"][baseline]).sum()
        )
        aggregate.loc[candidate, "brier_wins_vs_current"] = int(
            (pivot["brier"][candidate] < pivot["brier"][baseline]).sum()
        )

    return per_seed, aggregate

"""Compare frozen prediction artifacts across models and seeds.

Replaces `multi_seed_summary.py`, `bscore_ablation_summary.py` and the three
`experiments/0{3,4,5}_*.ipynb` comparison notebooks, which each re-implemented
a slightly different version of the same table.

Every comparison here is paired: the artifacts are checked to cover exactly the
same matches with the same labels before any metric is compared.  That check is
what makes a difference of a few thousandths of a log loss meaningful, and it
is the check that the temporal experiment silently failed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tennis_gnn.experiment_tracking import (  # noqa: E402
    assert_compatible,
    load_prediction_artifact,
    probability_metrics,
)


METRICS = ("accuracy", "log_loss", "brier")
LOWER_IS_BETTER = {"log_loss": True, "brier": True, "accuracy": False}


def artifact_root(scope: str) -> Path:
    return ROOT / "results" / "frozen_predictions" / scope


def available(scope: str, seed: int) -> list[str]:
    directory = artifact_root(scope) / f"seed_{seed}"
    return sorted(
        path.name.removesuffix(".manifest.json")
        for path in directory.glob("*.manifest.json")
    )


def per_seed_metrics(
    scope: str,
    seeds: list[int],
    experiments: list[str],
    *,
    phase: str = "test",
    strict: bool = True,
) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        directory = artifact_root(scope) / f"seed_{seed}"
        loaded = {}
        for name in experiments:
            if not (directory / f"{name}.manifest.json").is_file():
                continue
            loaded[name] = load_prediction_artifact(directory, name)
        if strict and len(loaded) > 1:
            # Refuses to average models that were not scored on identical
            # matches and labels.
            assert_compatible(
                [manifest for _, manifest in loaded.values()],
                require_same_seed=True,
            )
        for name, (frame, manifest) in loaded.items():
            subset = frame[frame["phase"] == phase]
            rows.append(
                {
                    "seed": seed,
                    "experiment": name,
                    **probability_metrics(subset),
                }
            )
    return pd.DataFrame(rows)


def aggregate(per_seed: pd.DataFrame) -> pd.DataFrame:
    summary = per_seed.groupby("experiment")[list(METRICS)].agg(
        ["mean", "std"]
    )
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary["n_seeds"] = per_seed.groupby("experiment").size()
    return summary.sort_values("log_loss_mean")


def paired_delta(
    per_seed: pd.DataFrame, candidate: str, baseline: str
) -> pd.DataFrame:
    """Seed-level paired comparison of two experiments."""

    pivot = per_seed.pivot(
        index="seed", columns="experiment", values=list(METRICS)
    )
    rows = []
    for metric in METRICS:
        if (
            candidate not in pivot[metric].columns
            or baseline not in pivot[metric].columns
        ):
            continue
        delta = pivot[metric][candidate] - pivot[metric][baseline]
        delta = delta.dropna()
        n = len(delta)
        wins = (
            (delta < 0).sum()
            if LOWER_IS_BETTER[metric]
            else (delta > 0).sum()
        )
        # Seed-level t interval; with five seeds this is the honest statement
        # of how repeatable the difference is.
        standard_error = (
            float(delta.std(ddof=1)) / np.sqrt(n) if n > 1 else float("nan")
        )
        half_width = 2.776 * standard_error if n == 5 else 1.96 * standard_error
        rows.append(
            {
                "metric": metric,
                "mean_delta": float(delta.mean()),
                "ci_low": float(delta.mean() - half_width),
                "ci_high": float(delta.mean() + half_width),
                "wins": f"{int(wins)}/{n}",
                "excludes_zero": bool(
                    (delta.mean() - half_width) * (delta.mean() + half_width)
                    > 0
                ),
            }
        )
    frame = pd.DataFrame(rows)
    frame.attrs["candidate"] = candidate
    frame.attrs["baseline"] = baseline
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", default="slams_masters")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help="defaults to every artifact present for the first seed",
    )
    parser.add_argument("--phase", default="test")
    parser.add_argument("--candidate", default=None)
    parser.add_argument("--baseline", default="gbdt_tuned")
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="allow artifacts scored on different evaluation sets",
    )
    args = parser.parse_args()

    experiments = args.experiments or available(args.scope, args.seeds[0])
    per_seed = per_seed_metrics(
        args.scope,
        args.seeds,
        experiments,
        phase=args.phase,
        strict=not args.no_strict,
    )
    if per_seed.empty:
        print("No artifacts found.")
        return

    print(f"=== {args.scope} | phase={args.phase} | seeds={args.seeds} ===\n")
    summary = aggregate(per_seed)
    print(
        summary[
            [
                "accuracy_mean",
                "log_loss_mean",
                "brier_mean",
                "log_loss_std",
                "n_seeds",
            ]
        ].to_string()
    )

    if args.candidate:
        print(f"\n=== {args.candidate} vs {args.baseline} (paired) ===")
        print(
            paired_delta(per_seed, args.candidate, args.baseline).to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()

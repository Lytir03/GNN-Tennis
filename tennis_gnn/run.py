"""Run named experiments and write frozen prediction artifacts.

This replaces the previous approach, in which a script rewrote individual cells
of a source notebook by string substitution and cell index to produce one
generated notebook per ablation.  That machinery broke whenever a cell moved,
and it made the actual experimental difference between two runs very hard to
see.  Here an experiment is a ``ModelConfig`` plus a ``TrainConfig``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tennis_gnn.experiment_tracking import (  # noqa: E402
    ArtifactManifest,
    save_prediction_artifact,
)
from tennis_gnn.config import (  # noqa: E402
    BASE_MODEL,
    TrainConfig,
    one_factor_ablations,
)
from tennis_gnn.data import TRAIN_END, VAL_END, load_dataset  # noqa: E402
from tennis_gnn.snapshots import load_or_build  # noqa: E402
from tennis_gnn.train import metrics, run_experiment  # noqa: E402


# Selected on validation only; see tennis_gnn/tune.py and
# results/tuning/gnn_validation_search.csv.
TUNED_TRAINING = TrainConfig(
    learning_rate=3e-4,
    steps_per_block=8,
    replay_batch_size=32,
    hidden_dim=32,
    dropout=0.3,
    passes=1,
    calibrate=True,
)


def artifact_directory(scope: str, seed: int) -> Path:
    return ROOT / "results" / "frozen_predictions" / scope / f"seed_{seed}"


def run_named(
    name: str,
    *,
    scope: str,
    seed: int,
    train_config: TrainConfig,
    verbose: bool = True,
) -> dict:
    ablations = one_factor_ablations()
    if name not in ablations:
        raise ValueError(
            f"Unknown experiment {name!r}; choose one of {tuple(ablations)}"
        )
    model_config = ablations[name]

    dataset = load_dataset(ROOT, scope=scope)
    snapshots = load_or_build(
        ROOT, dataset, model_config.edge_preset, seed=seed, verbose=verbose
    )

    start = time.time()
    result = run_experiment(
        snapshots,
        model_config,
        replace(train_config, seed=seed),
        verbose=verbose,
    )
    elapsed = time.time() - start

    predictions = result["predictions"]
    save_prediction_artifact(
        predictions,
        artifact_directory(scope, seed),
        ArtifactManifest(
            experiment=name,
            model_family=f"TennisGNN[{model_config.conv_type}]",
            tournament_scope=scope,
            seed=seed,
            train_end=TRAIN_END,
            validation_end=VAL_END,
            test_end=2020,
            update_phases=("train",),
            config={
                "model": asdict(model_config),
                "training": asdict(replace(train_config, seed=seed)),
                "temperature": result["temperature"],
                "optimiser_steps": result["steps"],
            },
        ),
        probability_column="probability",
    )

    test = predictions[predictions["phase"] == "test"]
    validation = predictions[predictions["phase"] == "val"]
    if verbose:
        print(
            f"\n{name} [{scope} seed {seed}] in {elapsed:.0f}s "
            f"(T={result['temperature']:.3f})"
        )
        print("  val :", metrics(validation))
        print("  test:", metrics(test))
    return {
        "experiment": name,
        "seed": seed,
        "temperature": result["temperature"],
        **{f"val_{k}": v for k, v in metrics(validation).items()},
        **{f"test_{k}": v for k, v in metrics(test).items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["base"],
        help="names from tennis_gnn.config.one_factor_ablations(), or 'all'",
    )
    parser.add_argument("--scope", default="slams_masters")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument(
        "--recipe",
        choices=("tuned", "legacy"),
        default="tuned",
        help="tuned = validation-selected schedule; legacy = original 2-step",
    )
    parser.add_argument("--summary", default=None)
    args = parser.parse_args()

    names = (
        list(one_factor_ablations())
        if args.experiments == ["all"]
        else args.experiments
    )
    training = TUNED_TRAINING if args.recipe == "tuned" else TrainConfig()

    rows = []
    for seed in args.seeds:
        for name in names:
            rows.append(
                run_named(
                    name, scope=args.scope, seed=seed, train_config=training
                )
            )

    frame = pd.DataFrame(rows)
    print("\n" + frame.to_string(index=False))
    if args.summary:
        output = ROOT / args.summary
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False)
        print(f"\nWrote {output}")


if __name__ == "__main__":
    main()

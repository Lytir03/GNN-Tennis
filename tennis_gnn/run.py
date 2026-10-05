# Runs named experiments and writes frozen prediction artifacts.
#
# This replaces the old approach, where a script rewrote individual cells
# of a source notebook by string substitution and cell index to produce
# one generated notebook per ablation. That machinery broke whenever a
# cell moved, and it made the actual experimental difference between two
# runs very hard to see. Here an experiment is just a ModelConfig plus a
# TrainConfig.

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
    ModelConfig,
    TrainConfig,
    one_factor_ablations,
)
from tennis_gnn.data import SCOPE_FILES, load_dataset  # noqa: E402
from tennis_gnn.snapshots import load_or_build  # noqa: E402
from tennis_gnn.train import (  # noqa: E402
    metrics,
    run_ensemble,
    run_experiment,
)


# Selected on validation log loss only, never on test.  See tennis_gnn/tune.py
# and results/tuning/.  Mini-batch replay rather than replaying the whole
# 200-graph buffer every step: four times the updates at under half the cost.
TUNED_TRAINING = TrainConfig(
    learning_rate=1e-4,
    steps_per_block=4,
    replay_batch_size=32,
    hidden_dim=32,
    dropout=0.3,
    passes=1,
    calibrate=True,
)

# Three passes over the training years.  RETAINED AS A NEGATIVE RESULT - do not
# use this as the default recipe.
#
# The reasoning that produced it: on seed 42 the model looked under-optimised
# rather than under-regularised (training log loss still falling, 0.5470 ->
# 0.5395, and no train/validation gap at all, 0.5470 vs 0.5496), and the 2016
# validation year cannot arbitrate between schedules - differences between every
# schedule tried are 0.002-0.004 log loss against a standard error of roughly
# 0.018 on 1075 matches.
#
# It did not replicate.  Across seeds 42/123/456/789/2026 this schedule wins
# exactly one - seed 42, the one the diagnostic was run on - and its mean test
# log loss (0.6034) is worse than TUNED_TRAINING's (0.6022).  The diagnostic
# correctly described one seed and generalised to none.
LONG_TRAINING = replace(TUNED_TRAINING, passes=3)

RECIPES = {
    "legacy": TrainConfig(),
    "tuned": TUNED_TRAINING,
    "long": LONG_TRAINING,
}


def artifact_directory(scope: str, seed: int) -> Path:
    return ROOT / "results" / "frozen_predictions" / scope / f"seed_{seed}"


def run_named(
    name: str,
    *,
    scope: str,
    seed: int,
    train_config: TrainConfig,
    verbose: bool = True,
    members: int = 1,
    artifact_name: str | None = None,
    model_config: ModelConfig | None = None,
    graph_window_days: int | None = None,
) -> dict:
    # Runs one experiment and freezes its predictions.
    #
    # model_config overrides the named ablation. It exists for
    # combinations that are deliberately not one-factor - the depth
    # ablation crossed with the history node features, for instance -
    # which shouldn't be added to one_factor_ablations() without breaking
    # its guarantee.
    if model_config is None:
        ablations = one_factor_ablations()
        if name not in ablations:
            raise ValueError(
                f"Unknown experiment {name!r}; choose one of {tuple(ablations)}"
            )
        model_config = ablations[name]

    _split = SCOPE_FILES[scope][2]
    dataset = load_dataset(ROOT, scope=scope)
    window_kwargs = (
        {} if graph_window_days is None
        else {"window_days": graph_window_days}
    )
    snapshots = load_or_build(
        ROOT,
        dataset,
        model_config.edge_preset,
        seed=seed,
        verbose=verbose,
        **window_kwargs,
    )

    start = time.time()
    if members > 1:
        result = run_ensemble(
            snapshots,
            model_config,
            replace(train_config, seed=seed),
            members=members,
            verbose=verbose,
        )
    else:
        result = run_experiment(
            snapshots,
            model_config,
            replace(train_config, seed=seed),
            verbose=verbose,
        )
    elapsed = time.time() - start

    predictions = result["predictions"]
    experiment_name = artifact_name or name
    save_prediction_artifact(
        predictions,
        artifact_directory(scope, seed),
        ArtifactManifest(
            experiment=experiment_name,
            model_family=f"TennisGNN[{model_config.conv_type}]",
            tournament_scope=scope,
            seed=seed,
            # Read from the scope's split rather than module constants: with
            # more than one split in play, hardcoding the original values makes
            # the manifest describe an experiment that did not happen, and
            # assert_compatible then refuses to compare artifacts that are in
            # fact comparable.
            train_end=_split.train_end,
            validation_end=_split.val_end,
            test_end=int(_split.rolling_end[:4]) - 1,
            update_phases=("train",),
            config={
                "model": asdict(model_config),
                "training": asdict(replace(train_config, seed=seed)),
                "temperature": result["temperature"],
                "optimiser_steps": result.get("steps"),
                "ensemble_members": members,
                "graph_window_days": graph_window_days,
            },
        ),
        probability_column="probability",
    )

    test = predictions[predictions["phase"] == "test"]
    validation = predictions[predictions["phase"] == "val"]
    if verbose:
        print(
            f"\n{experiment_name} [{scope} seed {seed}] in {elapsed:.0f}s "
            f"(T={result['temperature']:.3f})"
        )
        print("  val :", metrics(validation))
        print("  test:", metrics(test))
    return {
        "experiment": experiment_name,
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
        choices=tuple(RECIPES),
        default="tuned",
        help="legacy = original 2-step; tuned = validation-selected; "
        "long = three passes, chosen by the underfitting diagnostic",
    )
    parser.add_argument(
        "--members",
        type=int,
        default=1,
        help="ensemble size; >1 averages independently initialised models",
    )
    parser.add_argument(
        "--artifact-name",
        default=None,
        help="override the saved artifact name (e.g. base_ensemble)",
    )
    parser.add_argument("--summary", default=None)
    args = parser.parse_args()

    names = (
        list(one_factor_ablations())
        if args.experiments == ["all"]
        else args.experiments
    )
    training = RECIPES[args.recipe]

    rows = []
    for seed in args.seeds:
        for name in names:
            rows.append(
                run_named(
                    name,
                    scope=args.scope,
                    seed=seed,
                    train_config=training,
                    members=args.members,
                    artifact_name=args.artifact_name,
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

"""Validation-selected hyperparameter search for the GNN.

The GBDT baseline in this repository was given 32 tuned configurations, chosen
on the 2016 validation year.  The GNN was given none: it ran one fixed recipe
(2 optimiser steps per block at lr 1e-4) that nobody ever searched over.  Any
comparison between the two was therefore a comparison of a tuned model against
an untuned one.

This module gives the GNN the same treatment under the same rule: select on
validation log loss, never on test.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tennis_gnn.config import BASE_MODEL, TrainConfig  # noqa: E402
from tennis_gnn.data import load_dataset  # noqa: E402
from tennis_gnn.snapshots import load_or_build  # noqa: E402
from tennis_gnn.train import metrics, run_experiment  # noqa: E402


def search_space() -> list[TrainConfig]:
    """A staged grid over the optimisation recipe.

    The diagnosis driving these choices: the original recipe performed roughly
    670 optimiser steps in total, on a head initialised to output exactly zero.
    The model barely moved away from its B-score prior, so the search
    concentrates on how much optimisation happens and how it is regularised.
    """

    configs: list[TrainConfig] = []

    # Stage 1 - how much optimisation, and at what step size.
    for learning_rate in (1e-4, 3e-4, 1e-3):
        for steps, batch in (
            (2, None),  # the original recipe
            (4, 32),
            (8, 32),
            (16, 32),
        ):
            configs.append(
                TrainConfig(
                    learning_rate=learning_rate,
                    steps_per_block=steps,
                    replay_batch_size=batch,
                )
            )

    # Stage 2 - capacity and regularisation around a mid-strength schedule.
    for hidden in (32, 64):
        for dropout in (0.1, 0.3, 0.5):
            configs.append(
                TrainConfig(
                    learning_rate=3e-4,
                    steps_per_block=8,
                    replay_batch_size=32,
                    hidden_dim=hidden,
                    dropout=dropout,
                )
            )

    # Stage 3 - repeated passes over the training years.
    for passes in (2, 3):
        for learning_rate in (1e-4, 3e-4):
            configs.append(
                TrainConfig(
                    learning_rate=learning_rate,
                    steps_per_block=4,
                    replay_batch_size=32,
                    passes=passes,
                )
            )

    # Deduplicate while preserving order.
    seen = set()
    unique = []
    for config in configs:
        key = json.dumps(asdict(config), sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(config)
    return unique


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", default="slams_masters")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preset", default="full")
    parser.add_argument(
        "--output", default="results/tuning/gnn_validation_search.csv"
    )
    args = parser.parse_args()

    dataset = load_dataset(ROOT, scope=args.scope)
    snapshots = load_or_build(ROOT, dataset, args.preset, seed=args.seed)

    configs = search_space()
    print(f"Evaluating {len(configs)} configurations on validation only.\n")

    rows = []
    for index, base_config in enumerate(configs, start=1):
        config = replace(base_config, seed=args.seed)
        start = time.time()
        result = run_experiment(
            snapshots,
            BASE_MODEL,
            config,
            verbose=False,
            eval_phases=("val",),
        )
        predictions = result["predictions"]
        validation = predictions[predictions["phase"] == "val"]

        uncalibrated = metrics(validation)
        # Temperature is refitted from the same logits, so both variants come
        # from one training run rather than two.
        from tennis_gnn.train import fit_temperature
        import numpy as np

        temperature = fit_temperature(
            validation["logit"].to_numpy(), validation["y_true"].to_numpy()
        )
        calibrated_frame = validation.copy()
        calibrated_frame["probability"] = 1.0 / (
            1.0 + np.exp(-calibrated_frame["logit"].to_numpy() / temperature)
        )
        calibrated = metrics(calibrated_frame)

        rows.append(
            {
                **asdict(config),
                "steps": result["steps"],
                "temperature": temperature,
                "val_accuracy": uncalibrated["accuracy"],
                "val_log_loss": uncalibrated["log_loss"],
                "val_brier": uncalibrated["brier"],
                "val_log_loss_calibrated": calibrated["log_loss"],
                "val_brier_calibrated": calibrated["brier"],
                "seconds": time.time() - start,
            }
        )
        print(
            f"[{index:2d}/{len(configs)}] "
            f"lr={config.learning_rate:<7g} steps={config.steps_per_block:<3d}"
            f" batch={str(config.replay_batch_size):<5s}"
            f" h={config.hidden_dim:<3d} do={config.dropout:<4g}"
            f" passes={config.passes} -> "
            f"val_ll={uncalibrated['log_loss']:.4f} "
            f"(cal {calibrated['log_loss']:.4f}) "
            f"acc={uncalibrated['accuracy']:.4f} "
            f"[{time.time() - start:.0f}s]",
            flush=True,
        )

    frame = pd.DataFrame(rows).sort_values("val_log_loss_calibrated")
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"\nWrote {output}")
    print("\nTop 5 by calibrated validation log loss:")
    print(
        frame.head(5)[
            [
                "learning_rate",
                "steps_per_block",
                "replay_batch_size",
                "hidden_dim",
                "dropout",
                "passes",
                "val_log_loss",
                "val_log_loss_calibrated",
                "val_accuracy",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()

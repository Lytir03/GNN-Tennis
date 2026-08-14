"""Run frozen current-GNN and candidate artifacts for missing seeds."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import tempfile


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE / "experiments"
DEFAULT_SEEDS = (42, 123, 456, 789, 2026)


def artifact_exists(scope: str, seed: int, name: str) -> bool:
    directory = (
        HERE.parent
        / "results"
        / "frozen_predictions"
        / scope
        / f"seed_{seed}"
    )
    return (
        (directory / f"{name}.csv").is_file()
        and (directory / f"{name}.manifest.json").is_file()
    )


def execute(notebook: str, environment: dict[str, str], label: str) -> None:
    with tempfile.TemporaryDirectory(prefix="tennis-multiseed-") as directory:
        temporary = Path(directory)
        config = temporary / "jupyter-config"
        runtime = temporary / "jupyter-runtime"
        config.mkdir()
        runtime.mkdir()
        command = [
            "jupyter",
            "nbconvert",
            "--execute",
            "--to",
            "notebook",
            "--ExecutePreprocessor.timeout=-1",
            "--output",
            str(temporary / f"{label}.ipynb"),
            str(EXPERIMENTS / notebook),
        ]
        print(f"\n=== Running {label} ===", flush=True)
        subprocess.run(
            command,
            check=True,
            cwd=HERE.parent,
            env={
                **os.environ,
                "JUPYTER_CONFIG_DIR": str(config),
                "JUPYTER_RUNTIME_DIR": str(runtime),
                **environment,
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scope",
        choices=("slams", "slams_masters"),
        default="slams_masters",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for seed in args.seeds:
        common = {
            "TENNIS_RUN_SCOPE": args.scope,
            "TENNIS_SEED": str(seed),
        }
        baseline_ready = all(
            artifact_exists(args.scope, seed, name)
            for name in ("current_gnn", "bscore_logit")
        )
        if args.force or not baseline_ready:
            execute(
                "00_freeze_baselines.ipynb",
                common,
                f"{args.scope}_seed_{seed}_baselines",
            )
        else:
            print(f"Skipping existing baselines for seed {seed}", flush=True)

        if args.force or not artifact_exists(
            args.scope, seed, "antisymmetric_decoder"
        ):
            execute(
                "02_architecture_ablations.ipynb",
                {
                    **common,
                    "TENNIS_RUN_EXPERIMENT": "antisymmetric_decoder",
                },
                f"{args.scope}_seed_{seed}_antisymmetric",
            )
        else:
            print(f"Skipping existing candidate for seed {seed}", flush=True)


if __name__ == "__main__":
    main()

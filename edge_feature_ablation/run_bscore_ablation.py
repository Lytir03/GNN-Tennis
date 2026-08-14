"""Run only the frozen best-GNN B-score ablations."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
NOTEBOOK = HERE / "experiments" / "02_architecture_ablations.ipynb"
EXPERIMENTS = (
    "antisymmetric_no_direct_bscore",
    "antisymmetric_no_node_bscore",
    "antisymmetric_no_bscore",
)


def complete(scope: str, seed: int, experiment: str) -> bool:
    directory = (
        ROOT / "results" / "frozen_predictions" / scope / f"seed_{seed}"
    )
    return (
        (directory / f"{experiment}.csv").is_file()
        and (directory / f"{experiment}.manifest.json").is_file()
    )


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
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    jupyter = Path(sys.executable).with_name("jupyter")
    with tempfile.TemporaryDirectory(prefix="tennis-bscore-") as temporary:
        temporary_path = Path(temporary)
        config = temporary_path / "config"
        runtime = temporary_path / "runtime"
        config.mkdir()
        runtime.mkdir()
        for seed in args.seeds:
            for experiment in EXPERIMENTS:
                if complete(args.scope, seed, experiment) and not args.force:
                    print(f"Skipping {experiment}, seed {seed}", flush=True)
                    continue
                command = [
                    str(jupyter),
                    "nbconvert",
                    "--execute",
                    "--to",
                    "notebook",
                    "--ExecutePreprocessor.timeout=-1",
                    "--output",
                    str(temporary_path / f"{experiment}_{seed}.ipynb"),
                    str(NOTEBOOK),
                ]
                print(f"Running {experiment}, seed {seed}", flush=True)
                subprocess.run(
                    command,
                    cwd=ROOT,
                    check=True,
                    env={
                        **os.environ,
                        "TENNIS_RUN_SCOPE": args.scope,
                        "TENNIS_RUN_EXPERIMENT": experiment,
                        "TENNIS_SEED": str(seed),
                        "JUPYTER_CONFIG_DIR": str(config),
                        "JUPYTER_RUNTIME_DIR": str(runtime),
                    },
                )


if __name__ == "__main__":
    main()

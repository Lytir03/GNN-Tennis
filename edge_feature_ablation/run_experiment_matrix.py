"""Execute the remaining frozen ablations through Jupyter."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE / "experiments"

EDGE_RUNS = ("set_margin", "straight_sets", "match_status", "full")
ARCHITECTURE_RUNS = (
    "antisymmetric_decoder",
    "node_normalization",
    "mean_aggregation",
    "residual_connections",
    "tournament_context",
)
TARGETED_RUNS = (
    "antisymmetric_no_direct_bscore",
    "antisymmetric_no_node_bscore",
    "antisymmetric_no_bscore",
    "antisymmetric_straight_sets",
    "antisymmetric_mean",
    "antisymmetric_tournament_context",
    "antisymmetric_mean_tournament",
)


def execute(notebook: str, environment: dict[str, str], label: str) -> None:
    with tempfile.TemporaryDirectory(prefix="tennis-gnn-") as directory:
        temporary_directory = Path(directory)
        output = temporary_directory / f"{label}.ipynb"
        jupyter_config = temporary_directory / "jupyter-config"
        jupyter_runtime = temporary_directory / "jupyter-runtime"
        jupyter_config.mkdir()
        jupyter_runtime.mkdir()
        command = [
            "jupyter",
            "nbconvert",
            "--execute",
            "--to",
            "notebook",
            "--ExecutePreprocessor.timeout=-1",
            "--output",
            str(output),
            str(EXPERIMENTS / notebook),
        ]
        print(f"\n=== Running {label} ===", flush=True)
        subprocess.run(
            command,
            check=True,
            cwd=HERE.parent,
            env={
                **os.environ,
                "JUPYTER_CONFIG_DIR": str(jupyter_config),
                "JUPYTER_RUNTIME_DIR": str(jupyter_runtime),
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
        "--group",
        choices=("edge", "architecture", "targeted", "all"),
        default="all",
    )
    args = parser.parse_args()
    common = {"TENNIS_RUN_SCOPE": args.scope}

    if args.group in {"edge", "all"}:
        for ablation in EDGE_RUNS:
            execute(
                "01_edge_feature_ablations.ipynb",
                {**common, "TENNIS_RUN_EDGE_ABLATION": ablation},
                f"edge_{ablation}",
            )

    if args.group in {"architecture", "all"}:
        for experiment in ARCHITECTURE_RUNS:
            execute(
                "02_architecture_ablations.ipynb",
                {**common, "TENNIS_RUN_EXPERIMENT": experiment},
                experiment,
            )

    if args.group in {"targeted", "all"}:
        for experiment in TARGETED_RUNS:
            execute(
                "02_architecture_ablations.ipynb",
                {**common, "TENNIS_RUN_EXPERIMENT": experiment},
                experiment,
            )

    execute(
        "03_final_comparison.ipynb",
        common,
        "final_comparison",
    )


if __name__ == "__main__":
    main()

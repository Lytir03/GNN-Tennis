"""Run the two temporal variants for one or more seeds."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def complete(scope: str, seed: int, variant: str) -> bool:
    experiment = (
        "temporal_gnn_intransitivity"
        if variant == "intransitivity"
        else "temporal_gnn"
    )
    directory = ROOT / "results" / "frozen_predictions" / scope / f"seed_{seed}"
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
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for seed in args.seeds:
        for variant in ("base", "intransitivity"):
            if complete(args.scope, seed, variant) and not args.force:
                print(f"Skipping {variant}, seed {seed}")
                continue
            subprocess.run(
                [
                    sys.executable,
                    str(HERE / "train.py"),
                    "--scope",
                    args.scope,
                    "--seed",
                    str(seed),
                    "--variant",
                    variant,
                    "--epochs",
                    str(args.epochs),
                ],
                cwd=ROOT,
                check=True,
                env={
                    **os.environ,
                    # The current tennis-gnn conda environment combines a
                    # pip PyTorch wheel with conda's OpenBLAS/LLVM runtime.
                    # Keep execution single-threaded and allow those two
                    # OpenMP libraries to coexist on macOS.
                    "KMP_DUPLICATE_LIB_OK": "TRUE",
                    "OMP_NUM_THREADS": "1",
                },
            )


if __name__ == "__main__":
    main()

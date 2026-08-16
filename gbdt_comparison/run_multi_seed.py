"""Tune the GBDT for each requested seed, skipping completed artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def complete(scope: str, seed: int, feature_set: str) -> bool:
    directory = (
        ROOT / "results" / "frozen_predictions" / scope / f"seed_{seed}"
    )
    experiment = "gbdt_tuned" if feature_set == "full" else "gbdt_no_bscore"
    return (
        (directory / f"{experiment}.csv").is_file()
        and (directory / f"{experiment}.manifest.json").is_file()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scope",
        choices=("slams", "slams_masters", "full"),
        default="slams_masters",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=(42, 123, 456, 789, 2026),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--feature-set",
        choices=("full", "no_bscore"),
        default="full",
    )
    args = parser.parse_args()
    for seed in args.seeds:
        if complete(args.scope, seed, args.feature_set) and not args.force:
            print(f"Skipping completed GBDT seed {seed}", flush=True)
            continue
        command = [
            sys.executable,
            str(HERE / "train.py"),
            "--scope",
            args.scope,
            "--seed",
            str(seed),
            "--feature-set",
            args.feature_set,
        ]
        print(f"\n=== GBDT {args.scope} seed {seed} ===", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()

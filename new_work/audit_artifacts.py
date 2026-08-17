"""Sweep every frozen artifact for the anomalies that hide broken runs.

Written after a broken temperature fit sat undetected in 18 artifacts for weeks.
It was found by chasing one odd number by hand; this does that sweep for all of
them, so the next one is found by running a script rather than by luck.

Each check is a property that should hold of *any* honest run, so a violation is
always worth an explanation - not necessarily a bug, but never nothing:

  degenerate      the model emits (almost) one probability: it learnt nothing,
                  or something upstream is not reaching it
  extreme T       a temperature far from 1, or sitting on the +-4 log clamp,
                  which is the signature of a fit that failed rather than one
                  that found something
  worse than 0.5  test log loss above the coin flip: a model that is actively
                  anti-predictive
  overconfident   probabilities pinned at the extremes, where log loss explodes
  hash mismatch   the CSV no longer matches the manifest that describes it

Run: python new_work/audit_artifacts.py [--scope full]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tennis_gnn.experiment_tracking import evaluation_hash  # noqa: E402

CLAMP_LOW, CLAMP_HIGH = np.exp(-4.0), np.exp(4.0)
COIN_FLIP = float(np.log(2))


def audit_one(manifest_path: Path) -> list[str]:
    manifest = json.loads(manifest_path.read_text())
    csv_path = manifest_path.parent / (
        manifest_path.name.removesuffix(".manifest.json") + ".csv"
    )
    if not csv_path.is_file():
        return ["csv missing"]

    frame = pd.read_csv(csv_path)
    problems = []

    if evaluation_hash(frame) != manifest["evaluation_hash"]:
        problems.append("hash mismatch: CSV and manifest disagree")

    test = frame[frame["phase"] == "test"]
    if test.empty:
        return problems + ["no test rows"]

    p = test["probability"].to_numpy()
    y = test["y_true"].to_numpy()

    if not np.all(np.isfinite(p)):
        problems.append("non-finite probabilities")
    if p.min() < 0 or p.max() > 1:
        problems.append(f"probabilities outside [0,1]: [{p.min()}, {p.max()}]")
    if float(p.std()) < 1e-4:
        problems.append(f"degenerate: probability std {p.std():.2e}")

    clipped = np.clip(p, 1e-7, 1 - 1e-7)
    log_loss = float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)))
    if log_loss > COIN_FLIP + 1e-6:
        problems.append(
            f"worse than a coin flip: log loss {log_loss:.4f} > {COIN_FLIP:.4f}"
        )
    extreme = float(np.mean((p < 0.01) | (p > 0.99)))
    if extreme > 0.05:
        problems.append(f"overconfident: {extreme:.1%} of predictions beyond 1%")

    temperature = manifest.get("config", {}).get("temperature")
    if temperature is not None:
        if temperature <= CLAMP_LOW * 1.01 or temperature >= CLAMP_HIGH * 0.99:
            problems.append(f"temperature on the clamp: T={temperature:.4g}")
        elif not 0.2 <= temperature <= 5.0:
            problems.append(f"extreme temperature: T={temperature:.4g}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", default=None)
    args = parser.parse_args()

    pattern = (
        f"{args.scope}/seed_*/*.manifest.json"
        if args.scope
        else "*/seed_*/*.manifest.json"
    )
    paths = sorted((ROOT / "results" / "frozen_predictions").glob(pattern))

    flagged = 0
    for path in paths:
        problems = audit_one(path)
        if problems:
            flagged += 1
            label = f"{path.parent.parent.name}/{path.parent.name}/{path.stem}"
            print(f"{label.removesuffix('.manifest')}")
            for problem in problems:
                print(f"    - {problem}")

    print(f"\n{len(paths)} artifacts audited, {flagged} flagged")
    if flagged:
        print(
            "A flag is a question, not a verdict.  A model given no usable "
            "features\nshould be degenerate; that is the correct answer, and it "
            "is still worth\nseeing listed."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

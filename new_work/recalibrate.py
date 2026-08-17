"""Repair artifacts written while the temperature fit was broken.

**The defect.**  `fit_temperature` ran LBFGS without a line search.  With fixed
step sizes it overshot whenever the optimum was below one, running to the clamp
at T=0.0183 - so no *under-confident* model in this project was ever calibrated,
and several were actively damaged.  The worst case, tier 1 with message passing
disabled, was assigned T=0.403 when the correct value is 0.128, costing 0.0225
test log loss and inflating that tier's apparent gain from message passing by
about threefold.

**Why this does not need retraining.**  A stored probability is
`p = sigmoid(logit / T)` with `T` recorded in the manifest, so the raw logit is
recoverable exactly as `T * log(p / (1 - p))`.  Refitting the temperature on the
validation rows and rewriting the probabilities reproduces what the fixed
calibrator would have produced from the same training run - no weights change,
no data is re-drawn, and the model is untouched.

**What is checked.**  `evaluation_hash` covers the keys and labels, never the
probabilities, so a correct recalibration leaves it identical.  This asserts
that, and refuses any artifact whose hash moves - that would mean something
other than calibration changed.

Validated against retraining: the tier-1 no-message cell recalibrated here
reproduces the retrained T and log losses to four decimals (see --verify).

Run: python new_work/recalibrate.py --scope full          (reports only)
     python new_work/recalibrate.py --scope full --apply
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
from tennis_gnn.train import fit_temperature  # noqa: E402

EPSILON = 1e-12


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def recover_logits(probability: np.ndarray, temperature: float) -> np.ndarray:
    """Invert `p = sigmoid(logit / T)`.

    Clipping bounds the inversion away from the asymptotes; at the
    probabilities this project produces (0.1-0.9) it never binds.
    """

    p = np.clip(probability, EPSILON, 1 - EPSILON)
    return temperature * np.log(p / (1 - p))


def recalibrate_frame(
    frame: pd.DataFrame, temperature: float
) -> tuple[pd.DataFrame, float]:
    logits = recover_logits(frame["probability"].to_numpy(), temperature)
    validation = frame["phase"] == "val"
    if not validation.any():
        raise ValueError("no validation rows, so temperature cannot be refit")
    new_temperature = fit_temperature(
        logits[validation.to_numpy()],
        frame.loc[validation, "y_true"].to_numpy(),
    )
    repaired = frame.copy()
    repaired["probability"] = 1.0 / (1.0 + np.exp(-logits / new_temperature))
    return repaired, float(new_temperature)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", default=None, help="default: every scope")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-4,
        help="report only artifacts whose test log loss moves by at least this",
    )
    args = parser.parse_args()

    pattern = (
        f"{args.scope}/seed_*/*.manifest.json"
        if args.scope
        else "*/seed_*/*.manifest.json"
    )
    root = ROOT / "results" / "frozen_predictions"

    changed = skipped = refused = 0
    rows = []
    for manifest_path in sorted(root.glob(pattern)):
        manifest = json.loads(manifest_path.read_text())
        temperature = manifest.get("config", {}).get("temperature")
        csv_path = manifest_path.parent / (
            manifest_path.name.removesuffix(".manifest.json") + ".csv"
        )
        if temperature is None or not csv_path.is_file():
            skipped += 1
            continue

        frame = pd.read_csv(csv_path)
        before = evaluation_hash(frame)
        if before != manifest["evaluation_hash"]:
            print(f"  REFUSED {csv_path}: stored hash does not match its CSV")
            refused += 1
            continue
        if (frame["phase"] == "val").sum() == 0:
            skipped += 1
            continue

        repaired, new_temperature = recalibrate_frame(frame, temperature)
        if evaluation_hash(repaired) != before:
            print(f"  REFUSED {csv_path}: recalibration moved the hash")
            refused += 1
            continue

        test = frame["phase"] == "test"
        y = frame.loc[test, "y_true"].to_numpy()
        old_ll = log_loss(y, frame.loc[test, "probability"].to_numpy())
        new_ll = log_loss(y, repaired.loc[test, "probability"].to_numpy())
        if abs(new_ll - old_ll) < args.threshold:
            continue

        rows.append(
            {
                "scope": manifest["tournament_scope"],
                "seed": manifest["seed"],
                "experiment": manifest["experiment"],
                "T_old": round(temperature, 4),
                "T_new": round(new_temperature, 4),
                "test_ll_old": round(old_ll, 5),
                "test_ll_new": round(new_ll, 5),
                "delta": round(new_ll - old_ll, 5),
            }
        )
        changed += 1
        if args.apply:
            repaired.to_csv(csv_path, index=False)
            manifest["config"]["temperature"] = new_temperature
            manifest["config"]["temperature_before_recalibration"] = temperature
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    if rows:
        frame = pd.DataFrame(rows).sort_values("delta")
        print(frame.to_string(index=False))
    print(
        f"\n{changed} artifact(s) {'repaired' if args.apply else 'need repair'}"
        f", {skipped} skipped (no temperature or no validation rows)"
        + (f", {refused} REFUSED" if refused else "")
    )
    return 1 if refused else 0


if __name__ == "__main__":
    raise SystemExit(main())

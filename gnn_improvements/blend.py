# Combine frozen artifacts into one blended forecast.
#
# This is the thing the closest prior work found actually worked. The MagNet
# GNN paper (arXiv 2510.20454) reports its graph model tying Weighted Elo
# outright (Brier 0.214 vs 0.217, accuracy 65.7% vs 65.8%) and only extracts
# value by *combining* the two - 0.2113 vs 0.2130 - closing 10.7% of the gap
# to bookmaker odds on intransitive matchups. Different model families make
# different mistakes, and averaging them is usually worth more than making
# either one better.
#
# Nothing in this repo blends across experiments. run_ensemble averages seeds
# or inits of a single config, in process, and never touches saved artifacts.
#
# Weights are fitted on validation only, by a simplex grid over the members,
# and the result is written back as an ordinary frozen artifact so compare.py
# picks it up through available() with no special casing.
#
# Run:
#   python gnn_improvements/blend.py --scope slams_masters \
#       --members bscore_general_control elo_baseline gbdt_tuned

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tennis_gnn.compare import artifact_root  # noqa: E402
from tennis_gnn.data import SCOPE_FILES  # noqa: E402
from tennis_gnn.experiment_tracking import (  # noqa: E402
    KEY_COLUMNS,
    ArtifactManifest,
    evaluation_hash,
    load_prediction_artifact,
    save_prediction_artifact,
)

SEEDS = (42, 123, 456, 789, 2026)
EPS = 1e-7


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _simplex(count: int, step: float = 0.1):
    # Every weight vector on a `step` grid that sums to 1.
    ticks = int(round(1.0 / step))
    for raw in product(range(ticks + 1), repeat=count):
        if sum(raw) == ticks:
            yield np.array(raw, dtype=float) / ticks


def blend_seed(scope: str, seed: int, members: list[str], step: float) -> dict:
    directory = artifact_root(scope) / f"seed_{seed}"
    frames = {}
    for name in members:
        frame, _ = load_prediction_artifact(directory, name)
        frames[name] = frame

    # Every member must have been scored on the same matches and labels, or
    # the blend is averaging predictions about different things. Checked per
    # phase because artifacts legitimately differ in coverage.
    for phase in ("val", "test"):
        digests = {
            name: evaluation_hash(frame, phase=phase)
            for name, frame in frames.items()
        }
        if len(set(digests.values())) > 1:
            raise SystemExit(
                f"seed {seed}: members disagree on the {phase} set:\n  "
                + "\n  ".join(f"{d[:12]} {n}" for n, d in digests.items())
            )

    merged = None
    for name, frame in frames.items():
        part = frame[[*KEY_COLUMNS, "y_true", "probability"]].rename(
            columns={"probability": name}
        )
        merged = part if merged is None else merged.merge(
            part.drop(columns=["y_true"]), on=list(KEY_COLUMNS)
        )

    validation = merged[merged["phase"] == "val"]
    y_val = validation["y_true"].to_numpy()
    matrix_val = np.column_stack([validation[n].to_numpy() for n in members])

    best_weights, best_loss = None, np.inf
    for weights in _simplex(len(members), step):
        loss = _log_loss(y_val, matrix_val @ weights)
        if loss < best_loss:
            best_weights, best_loss = weights, loss

    matrix_all = np.column_stack([merged[n].to_numpy() for n in members])
    merged["probability"] = matrix_all @ best_weights

    split = SCOPE_FILES[scope][2]
    name = "blend_" + "_".join(m.replace("_", "")[:8] for m in members)
    save_prediction_artifact(
        merged,
        directory,
        ArtifactManifest(
            experiment=name,
            model_family="Blend",
            tournament_scope=scope,
            seed=seed,
            train_end=split.train_end,
            validation_end=split.val_end,
            test_end=int(split.rolling_end[:4]) - 1,
            update_phases=("train",),
            config={
                "members": members,
                "weights": {
                    m: float(w) for m, w in zip(members, best_weights)
                },
                "weight_grid_step": step,
                "selection_metric": "validation_log_loss",
            },
        ),
        probability_column="probability",
    )

    test = merged[merged["phase"] == "test"]
    y = test["y_true"].to_numpy()
    p = test["probability"].to_numpy()
    row = {
        "experiment": name,
        "seed": seed,
        "test_log_loss": _log_loss(y, p),
        "test_brier": float(np.mean((np.clip(p, EPS, 1 - EPS) - y) ** 2)),
        "test_accuracy": float(np.mean((p > 0.5) == (y > 0.5))),
    }
    for member, weight in zip(members, best_weights):
        row[f"w_{member}"] = float(weight)
        member_test = merged[merged["phase"] == "test"][member].to_numpy()
        row[f"ll_{member}"] = _log_loss(y, member_test)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", default="slams_masters")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--members", nargs="+", required=True)
    parser.add_argument("--step", type=float, default=0.1)
    args = parser.parse_args()

    rows = [
        blend_seed(args.scope, seed, args.members, args.step)
        for seed in args.seeds
    ]
    frame = pd.DataFrame(rows)
    print(frame.to_string(index=False))
    print(f"\nblend mean test_ll = {frame['test_log_loss'].mean():.4f}")
    for member in args.members:
        print(
            f"  member {member:28s} mean test_ll = "
            f"{frame['ll_' + member].mean():.4f}  "
            f"mean weight = {frame['w_' + member].mean():.2f}"
        )
    output = ROOT / "gnn_improvements" / "results"
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"blend_{args.scope}.csv"
    frame.to_csv(path, index=False)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()

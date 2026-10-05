# The two baselines the project has been compared against in the literature
# but never actually run inside its own pipeline.
#
# bscore_only is the important one. Spagnolo et al. report Brier 0.194 / log
# loss 0.573 for weighted Bonacich centrality alone on Masters+Slams, and this
# repo's GNN scores 0.1946 / 0.5703 on the same tiers - i.e. the whole model
# may be worth nothing over the rating it contains. That comparison is across
# papers, across data sources and across test years, so it can only suggest.
# This script settles it on our own matches: same snapshots, same orientation
# draw, same evaluation hash, so compare.py can test it head to head.
#
# The B-score model is literally sigmoid(scale * bscore_diff) with one scalar
# fitted on validation - no features, no training, no graph.
#
# elo_baseline exists because elo.ipynb produces a number in a notebook and
# nothing that can be compared. Same Elo, emitted as a frozen artifact.
#
# Both reuse load_or_build snapshots rather than re-deriving keys and labels.
# That is not a convenience: the GBDT baseline re-derived its own orientation
# draw with a mirrored rng, drifted out of sync, and produced an artifact whose
# labels disagreed with the GNN's on 50.3% of matches.
#
# Run: python gnn_improvements/baselines.py --scope slams_masters

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tennis_gnn.config import BASE_MODEL  # noqa: E402
from tennis_gnn.data import SCOPE_FILES, load_dataset  # noqa: E402
from tennis_gnn.experiment_tracking import (  # noqa: E402
    ArtifactManifest,
    save_prediction_artifact,
)
from tennis_gnn.snapshots import load_or_build  # noqa: E402
from tennis_gnn.train import _SURFACE_BSCORE_COLUMN  # noqa: E402

SEEDS = (42, 123, 456, 789, 2026)
INITIAL_ELO = 1500.0
K_FACTOR = 32.0


def _fit_scale(logits: np.ndarray, y: np.ndarray) -> float:
    # One scalar, chosen on validation only, by a coarse-then-fine sweep.
    # A sweep rather than LBFGS because this project already retracted one
    # result to a temperature fit that failed silently; a grid cannot.
    # The grid has to span both natural scales in play: a B-score difference
    # wants a multiplier near 10, an Elo difference wants ln(10)/400 ~ 0.0058.
    # A range that cannot reach the optimum returns its own boundary and looks
    # like a fitted value, which is how the first version of this produced an
    # Elo baseline with 66% accuracy and a log loss of 1.96.
    best, best_loss = 1.0, np.inf
    for coarse in np.logspace(-5, 3, 161):
        p = 1.0 / (1.0 + np.exp(-np.clip(coarse * logits, -30, 30)))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        if loss < best_loss:
            best, best_loss = coarse, loss
    for fine in np.linspace(best * 0.5, best * 1.5, 101):
        p = 1.0 / (1.0 + np.exp(-np.clip(fine * logits, -30, 30)))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        if loss < best_loss:
            best, best_loss = fine, loss
    lo, hi = np.logspace(-5, 3, 161)[[0, -1]]
    if best <= lo * 1.01 or best >= hi * 0.99:
        raise RuntimeError(
            f"scale fit hit the grid boundary at {best:g}; widen the range "
            "rather than trusting a value the search could not move off"
        )
    return float(best)


def _rows_from_snapshots(snapshots, raw_scores):
    # raw_scores(snapshot) -> per-match signed score for player A.
    rows = []
    for snapshot in snapshots:
        scores = raw_scores(snapshot)
        for position in range(len(snapshot.y)):
            rows.append(
                {
                    "phase": snapshot.phase,
                    "tourney_id": snapshot.tourney_id,
                    "round_order": snapshot.round_order,
                    "block_idx": snapshot.block_idx,
                    "row_in_block": position,
                    "y_true": float(snapshot.y[position]),
                    "score": float(scores[position]),
                }
            )
    return pd.DataFrame(rows)


def _bscore_scores(surface: bool, transform: str):
    # transform="log" is not a detail. A single multiplier in logit space is a
    # poor functional form for a quantity whose median is 3e-3 and whose max is
    # 0.57, and that is the same defect the GNN's skip connection has. Running
    # the standalone rating both ways tests the conditioning hypothesis without
    # any neural network in the way.
    def scores(snapshot):
        if not surface and transform == "raw":
            return snapshot.bscore_diff.numpy()
        column = _SURFACE_BSCORE_COLUMN.get(snapshot.surface, 0) if surface else 0
        values = snapshot.x[:, column].numpy()
        if transform == "log":
            values = np.log(np.clip(values, 1e-9, None))
        a = snapshot.player_a.numpy()
        b = snapshot.player_b.numpy()
        return values[a] - values[b]

    return scores


def build_elo(scope: str, seed: int) -> dict:
    # Walks the blocks in chronological order, exactly as the graph build
    # does: read ratings before a block, emit, then update with that block's
    # results. Nothing is ever scored from its own outcome.
    #
    # player_a/player_b in a snapshot are node indices *within that block's
    # graph*, so they cannot identify a player across blocks. The global
    # identity is recovered from the dataset instead: the GNN's _block_targets
    # walks block_matches in order, so match `position` is block_matches row
    # `position`, and y == 1 means player A was the winner.
    dataset = load_dataset(ROOT, scope=scope)
    snapshots = load_or_build(
        ROOT, dataset, BASE_MODEL.edge_preset, seed=seed, verbose=False
    )
    ratings: dict[str, float] = defaultdict(lambda: INITIAL_ELO)
    rows = []
    for snapshot in snapshots:
        block = dataset.block_matches(snapshot.tourney_id, snapshot.round_order)
        winners = block["winner_name"].tolist()
        losers = block["loser_name"].tolist()
        outcomes = []
        for position in range(len(snapshot.y)):
            winner, loser = winners[position], losers[position]
            a_is_winner = float(snapshot.y[position]) == 1.0
            player_a = winner if a_is_winner else loser
            player_b = loser if a_is_winner else winner
            rows.append(
                {
                    "phase": snapshot.phase,
                    "tourney_id": snapshot.tourney_id,
                    "round_order": snapshot.round_order,
                    "block_idx": snapshot.block_idx,
                    "row_in_block": position,
                    "y_true": float(snapshot.y[position]),
                    "score": ratings[player_a] - ratings[player_b],
                }
            )
            outcomes.append((winner, loser))
        for winner, loser in outcomes:
            expected = 1.0 / (
                1.0 + 10 ** ((ratings[loser] - ratings[winner]) / 400.0)
            )
            ratings[winner] += K_FACTOR * (1.0 - expected)
            ratings[loser] -= K_FACTOR * (1.0 - expected)

    frame = pd.DataFrame(rows)
    validation = frame[frame["phase"] == "val"]
    scale = _fit_scale(
        validation["score"].to_numpy(), validation["y_true"].to_numpy()
    )
    frame["probability"] = 1.0 / (
        1.0 + np.exp(-np.clip(scale * frame["score"].to_numpy(), -30, 30))
    )

    split = SCOPE_FILES[scope][2]
    save_prediction_artifact(
        frame,
        ROOT / "results" / "frozen_predictions" / scope / f"seed_{seed}",
        ArtifactManifest(
            experiment="elo_baseline",
            model_family="Elo",
            tournament_scope=scope,
            seed=seed,
            train_end=split.train_end,
            validation_end=split.val_end,
            test_end=int(split.rolling_end[:4]) - 1,
            update_phases=("train",),
            config={
                "method": "sigmoid(scale * elo_diff)",
                "k_factor": K_FACTOR,
                "initial_elo": INITIAL_ELO,
                "scale": scale,
                "selection_metric": "validation_log_loss",
            },
        ),
        probability_column="probability",
    )
    test = frame[frame["phase"] == "test"]
    p = np.clip(test["probability"].to_numpy(), 1e-7, 1 - 1e-7)
    y = test["y_true"].to_numpy()
    return {
        "experiment": "elo_baseline",
        "seed": seed,
        "scale": scale,
        "test_log_loss": float(
            -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        ),
        "test_brier": float(np.mean((p - y) ** 2)),
        "test_accuracy": float(np.mean((p > 0.5) == (y > 0.5))),
    }


def build_bscore(
    scope: str, seed: int, surface: bool, transform: str = "raw"
) -> dict:
    dataset = load_dataset(ROOT, scope=scope)
    snapshots = load_or_build(
        ROOT, dataset, BASE_MODEL.edge_preset, seed=seed, verbose=False
    )
    frame = _rows_from_snapshots(snapshots, _bscore_scores(surface, transform))

    validation = frame[frame["phase"] == "val"]
    scale = _fit_scale(
        validation["score"].to_numpy(), validation["y_true"].to_numpy()
    )
    frame["probability"] = 1.0 / (
        1.0 + np.exp(-np.clip(scale * frame["score"].to_numpy(), -30, 30))
    )

    split = SCOPE_FILES[scope][2]
    name = "bscore_surface_only" if surface else "bscore_only"
    if transform != "raw":
        name = f"{name}_{transform}"
    save_prediction_artifact(
        frame,
        ROOT / "results" / "frozen_predictions" / scope / f"seed_{seed}",
        ArtifactManifest(
            experiment=name,
            model_family="BScoreOnly",
            tournament_scope=scope,
            seed=seed,
            train_end=split.train_end,
            validation_end=split.val_end,
            test_end=int(split.rolling_end[:4]) - 1,
            update_phases=("train",),
            config={
                "method": "sigmoid(scale * bscore_diff)",
                "surface_matched": surface,
                "transform": transform,
                "scale": scale,
                "selection_metric": "validation_log_loss",
            },
        ),
        probability_column="probability",
    )
    test = frame[frame["phase"] == "test"]
    p = np.clip(test["probability"].to_numpy(), 1e-7, 1 - 1e-7)
    y = test["y_true"].to_numpy()
    return {
        "experiment": name,
        "seed": seed,
        "scale": scale,
        "test_log_loss": float(
            -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        ),
        "test_brier": float(np.mean((p - y) ** 2)),
        "test_accuracy": float(np.mean((p > 0.5) == (y > 0.5))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", default="slams_masters")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument(
        "--models", nargs="+",
        default=[
            "bscore", "bscore_log", "bscore_surface",
            "bscore_surface_log", "elo",
        ],
        help="any of bscore, bscore_log, bscore_surface, "
             "bscore_surface_log, elo",
    )
    args = parser.parse_args()

    builders = {
        "bscore": lambda s: build_bscore(args.scope, s, surface=False),
        "bscore_log": lambda s: build_bscore(
            args.scope, s, surface=False, transform="log"
        ),
        "bscore_surface": lambda s: build_bscore(args.scope, s, surface=True),
        "bscore_surface_log": lambda s: build_bscore(
            args.scope, s, surface=True, transform="log"
        ),
        "elo": lambda s: build_elo(args.scope, s),
    }
    unknown = set(args.models) - set(builders)
    if unknown:
        raise SystemExit(f"unknown models {sorted(unknown)}")

    rows = []
    for model in args.models:
        for seed in args.seeds:
            row = builders[model](seed)
            rows.append(row)
            print(
                f"{row['experiment']:20s} seed {seed:<5d} "
                f"scale={row['scale']:8.2f} "
                f"test_ll={row['test_log_loss']:.4f} "
                f"brier={row['test_brier']:.4f} acc={row['test_accuracy']:.4f}",
                flush=True,
            )
    frame = pd.DataFrame(rows)
    print("\nmean by model:")
    for name, group in frame.groupby("experiment"):
        print(
            f"  {name:20s} test_ll={group['test_log_loss'].mean():.4f}  "
            f"brier={group['test_brier'].mean():.4f}  "
            f"acc={group['test_accuracy'].mean():.4f}"
        )
    output = ROOT / "gnn_improvements" / "results"
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"baselines_{args.scope}.csv"
    frame.to_csv(path, index=False)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()

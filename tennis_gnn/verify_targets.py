"""Check that rebuilt targets reproduce the frozen evaluation set exactly.

The GBDT baseline and every GNN artifact share an ``evaluation_hash`` over
(phase, tourney_id, round_order, block_idx, row_in_block, y_true).  If this
package draws match orientations differently, its predictions are silently no
longer comparable with the published GBDT numbers.  This script fails loudly
instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tennis_gnn.experiment_tracking import (  # noqa: E402
    evaluation_hash,
)
from tennis_gnn.data import load_dataset  # noqa: E402
from tennis_gnn.snapshots import load_or_build  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", default="slams_masters")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preset", default="full")
    parser.add_argument("--reference", default="gbdt_tuned")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    dataset = load_dataset(ROOT, scope=args.scope)
    snapshots = load_or_build(
        ROOT, dataset, args.preset, seed=args.seed, rebuild=args.rebuild
    )

    rows = []
    for snapshot in snapshots:
        for position in range(len(snapshot.y)):
            rows.append(
                {
                    "phase": snapshot.phase,
                    "tourney_id": snapshot.tourney_id,
                    "round_order": snapshot.round_order,
                    "block_idx": snapshot.block_idx,
                    "row_in_block": position,
                    "y_true": float(snapshot.y[position]),
                }
            )
    rebuilt = pd.DataFrame(rows)
    rebuilt_test = rebuilt[rebuilt["phase"] == "test"].reset_index(drop=True)

    reference_path = (
        ROOT
        / "results"
        / "frozen_predictions"
        / args.scope
        / f"seed_{args.seed}"
        / f"{args.reference}.csv"
    )
    reference = pd.read_csv(reference_path)
    reference = reference[reference["phase"] == "test"]

    key = ["phase", "tourney_id", "round_order", "block_idx", "row_in_block"]
    rebuilt_sorted = rebuilt_test.sort_values(key).reset_index(drop=True)
    reference_sorted = (
        reference[key + ["y_true"]].sort_values(key).reset_index(drop=True)
    )

    print(f"rebuilt test rows : {len(rebuilt_sorted)}")
    print(f"reference rows    : {len(reference_sorted)}")
    print(f"rebuilt hash      : {evaluation_hash(rebuilt_sorted)}")
    print(f"reference hash    : {evaluation_hash(reference_sorted)}")

    if len(rebuilt_sorted) != len(reference_sorted):
        print("\nFAIL: different number of evaluated matches")
        raise SystemExit(1)

    merged = rebuilt_sorted.merge(
        reference_sorted, on=key, suffixes=("_new", "_ref"), how="outer"
    )
    key_misses = merged["y_true_new"].isna() | merged["y_true_ref"].isna()
    label_diff = (merged["y_true_new"] != merged["y_true_ref"]) & ~key_misses
    print(f"unmatched keys    : {int(key_misses.sum())}")
    print(f"label mismatches  : {int(label_diff.sum())}")

    if evaluation_hash(rebuilt_sorted) == evaluation_hash(reference_sorted):
        print("\nPASS: evaluation set reproduced exactly")
    else:
        print("\nFAIL: evaluation set differs from the frozen artifacts")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

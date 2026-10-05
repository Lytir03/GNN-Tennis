# Does message passing stop earning its place once the model is told
# everything the graph was reconstructing?
#
# The completed cells already point one way. At lr 1e-4 / 4 steps, going from
# no messages to two hops is worth +0.0880 log loss when the model has no
# ratings at all, +0.0221 once it has B-score, and -0.0036 once it also has
# the history statistics at the decoder. Message passing is not useless - it
# is the only thing carrying signal in the first row - but what it carries is
# largely redundant with features that can be supplied directly, and at the
# richest tier it is net harmful.
#
# new_work/feature_hop_grid.py measures the same thing but gives each tier its
# own validation-selected recipe. That is a defensible design (a recipe is
# selected for an architecture, and the hop contrast stays one-factor within a
# tier), but it leaves one hole for this particular question: tier 3 runs at
# lr 3e-4, and new_work/twohop_diagnostic.py established that two hops are
# unstable at that rate. The "message passing hurts at the richest tier" cell
# is therefore measured at exactly the learning rate known to punish two hops,
# which is the artifact this project already retracted once.
#
# So this reruns the whole grid at a single stable recipe. Absolute levels
# across tiers then mean less than they do with per-tier tuning - a tier is
# not on its own best recipe - but every hop contrast is clean AND measured at
# the same step size, which is what the claim needs.
#
# Run: python gnn_improvements/hop_grid.py

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tennis_gnn.config import BASE_MODEL  # noqa: E402
from tennis_gnn.run import TUNED_TRAINING, run_named  # noqa: E402

SEEDS = (42, 123, 456, 789, 2026)

TIERS = {
    "t0_nothing": replace(
        BASE_MODEL, direct_bscore_logit=False, node_bscore_features=False
    ),
    "t1_bscore": BASE_MODEL,
    "t2_history_nodes": replace(BASE_MODEL, node_history_features=True),
    "t3_history_decoder": replace(BASE_MODEL, history_to_decoder=True),
}
HOPS = ("none", 1, 2)


def config_for(model, hops):
    # "none" keeps every layer, parameter and LayerNorm and removes only the
    # messages. num_layers=0 would also delete the normalisation, so it would
    # change two things at once.
    if hops == "none":
        return replace(model, num_layers=1, disable_message_passing=True)
    return replace(model, num_layers=int(hops))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", default="slams_masters")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--tiers", nargs="+", default=list(TIERS))
    args = parser.parse_args()

    print(
        f"hop grid: scope {args.scope}, tiers {args.tiers}, hops {HOPS}, "
        f"one recipe (lr {TUNED_TRAINING.learning_rate}, "
        f"{TUNED_TRAINING.steps_per_block} steps)\n",
        flush=True,
    )

    rows = []
    for tier in args.tiers:
        for hops in HOPS:
            name = f"hopgrid_{tier}_{hops}hop"
            for seed in args.seeds:
                start = time.time()
                row = run_named(
                    "base",
                    scope=args.scope,
                    seed=seed,
                    train_config=TUNED_TRAINING,
                    verbose=False,
                    artifact_name=name,
                    model_config=config_for(TIERS[tier], hops),
                )
                row.update(tier=tier, hops=str(hops))
                rows.append(row)
                print(
                    f"{tier:20s} {str(hops):>4s} hop  seed {seed:<5d} "
                    f"val_ll={row['val_log_loss']:.4f} "
                    f"test_ll={row['test_log_loss']:.4f} "
                    f"[{time.time() - start:.0f}s]",
                    flush=True,
                )

    frame = pd.DataFrame(rows)
    output = ROOT / "gnn_improvements" / "results"
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"hop_grid_{args.scope}.csv"
    frame.to_csv(path, index=False)

    print("\nmean test log loss:")
    table = frame.pivot_table(
        index="tier", columns="hops", values="test_log_loss", aggfunc="mean"
    )
    print(table.to_string())
    if "none" in table.columns and "2" in table.columns:
        print("\nvalue of two hops over none (positive = message passing helps):")
        for tier, row in table.iterrows():
            print(f"  {tier:20s} {row['none'] - row['2']:+.4f}")
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()

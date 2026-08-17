"""Feature richness x receptive field: does the graph substitute for history?

The "none" column keeps every layer, parameter and LayerNorm and removes only
the messages.  An earlier version used num_layers=0, which also deletes the
normalisation, and a control has to remove one thing.

That reasoning stands; the evidence originally given for it does not.  The
`num_layers=0` cell was reported as diverging to 2.89 log loss, and that was
blamed on unnormalised height (~185) sitting beside B-scores (<1).  It was
mostly a broken temperature fit - see `new_work/recalibrate.py`.  Repaired, the
cell scores 0.652 against the honest control's 0.624: still worse, not
divergent.  The control is right for the reason stated in the first paragraph,
which is a design argument and needs no dramatic number behind it.

The hop count is the intervention and the feature tier is the moderator.  Read
the grid down a column: as the model is given more per-player information, the
value of message passing should fall to zero and then go negative, because the
graph was only ever reconstructing that information.

Recipes are per tier, each chosen by a validation-only search, because a recipe
is selected *for* an architecture.  The recipe is held fixed across the hop axis
within a tier, which is what keeps every hop contrast one-factor.
"""

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

from tennis_gnn.config import (  # noqa: E402
    BASE_MODEL,
    TrainConfig,
    one_factor_ablations,
)
from tennis_gnn.compare import per_seed_metrics  # noqa: E402
from tennis_gnn.run import run_named  # noqa: E402


SEEDS = (42, 123, 456, 789, 2026)
ABLATIONS = one_factor_ablations()

# Recipe per tier, from results/tuning/*.csv, selected on validation only.
BASE_RECIPE = TrainConfig(
    learning_rate=1e-4, steps_per_block=4, replay_batch_size=32,
    hidden_dim=32, dropout=0.3, passes=1, calibrate=True,
)
NODE_HISTORY_RECIPE = replace(BASE_RECIPE, steps_per_block=8)
DECODER_RECIPE = replace(BASE_RECIPE, learning_rate=3e-4)

# Ordered from least to most per-player information.
TIERS = {
    "0_no_bscore": {
        "model": ABLATIONS["no_bscore_at_all"],
        "recipe": BASE_RECIPE,
        # Flagged: this tier reuses tier 1's recipe rather than its own search.
        # The hop contrast within the tier is still clean; only its absolute
        # level should be read with that caveat.
        "own_recipe": False,
        "label": "no B-score, no history",
    },
    "1_bscore": {
        "model": BASE_MODEL,
        "recipe": BASE_RECIPE,
        "own_recipe": True,
        "label": "B-score + static",
    },
    "2_history_nodes": {
        "model": ABLATIONS["history_nodes"],
        "recipe": NODE_HISTORY_RECIPE,
        "own_recipe": True,
        "label": "+ history on nodes",
    },
    "3_history_decoder": {
        "model": ABLATIONS["history_decoder"],
        "recipe": DECODER_RECIPE,
        "own_recipe": True,
        "label": "+ history at decoder",
    },
}

HOPS = ("none", 1, 2)

# Cells already computed under exactly these settings; reused rather than rerun.
# Only valid for the scope they were run on - a wider scope shares no artifacts.
EXISTING_SLAMS_MASTERS = {
    ("1_bscore", 1): "gnn_one_hop",
    ("1_bscore", 2): "gnn_tuned",
    ("2_history_nodes", 1): "gnn_history_tuned_one_hop",
    ("2_history_nodes", 2): "gnn_history_tuned",
    ("3_history_decoder", 1): "gnn_decoder_one_hop",
    ("3_history_decoder", 2): "gnn_decoder",
}


def existing_for(scope: str) -> dict:
    return EXISTING_SLAMS_MASTERS if scope == "slams_masters" else {}


def artifact_name(tier: str, hops, scope: str) -> str:
    return existing_for(scope).get((tier, hops), f"grid_{tier}_{hops}hop")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", default="slams_masters")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument(
        "--tiers", nargs="+", default=list(TIERS),
        help="subset of tiers to compute; the summary table is only written "
             "when every tier is present",
    )
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    scope, seeds = args.scope, args.seeds
    wanted = [t for t in TIERS if t in set(args.tiers)]

    existing = existing_for(scope)
    todo = [] if args.summary_only else [
        (tier, hops)
        for tier in wanted
        for hops in HOPS
        if (tier, hops) not in existing
    ]
    print(
        f"scope={scope} seeds={seeds}: "
        f"{len(todo)} cells to compute, {len(existing)} reused\n",
        flush=True,
    )

    for tier, hops in todo:
        spec = TIERS[tier]
        config = (
            replace(spec["model"], num_layers=1, disable_message_passing=True)
            if hops == "none"
            else replace(spec["model"], num_layers=hops)
        )
        name = artifact_name(tier, hops, scope)
        for seed in seeds:
            start = time.time()
            row = run_named(
                "base",
                scope=scope,
                seed=seed,
                train_config=spec["recipe"],
                verbose=False,
                artifact_name=name,
                model_config=config,
            )
            print(
                f"{tier:<20s} {hops} hop  seed {seed:<5d} "
                f"test_ll={row['test_log_loss']:.4f} "
                f"[{time.time() - start:.0f}s]",
                flush=True,
            )

    names = [artifact_name(t, h, scope) for t in TIERS for h in HOPS]
    available = {
        n for n in names
        if (ROOT / "results" / "frozen_predictions" / scope
            / f"seed_{seeds[0]}" / f"{n}.manifest.json").is_file()
    }
    complete = [
        t for t in TIERS
        if all(artifact_name(t, h, scope) in available for h in HOPS)
    ]
    if not complete:
        print("\nNo tier is complete yet; skipping the summary table.")
        return
    per_seed = per_seed_metrics(scope, list(seeds), sorted(available))
    means = per_seed.groupby("experiment")[["accuracy", "log_loss"]].mean()

    rows = []
    for tier in complete:
        spec = TIERS[tier]
        row = {"tier": spec["label"], "own_recipe": spec["own_recipe"]}
        for hops in HOPS:
            row[f"{hops}_hop"] = means.loc[
                artifact_name(tier, hops, scope), "log_loss"
            ]
        # The quantity the thesis is about: what message passing is worth.
        row["gain_1_vs_none"] = row["1_hop"] - row["none_hop"]
        row["gain_2_vs_1"] = row["2_hop"] - row["1_hop"]
        rows.append(row)

    grid = pd.DataFrame(rows)
    output = ROOT / "new_work" / "results" / f"feature_hop_grid_{scope}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(output, index=False)

    print("\n" + "=" * 92)
    print("FEATURE RICHNESS x RECEPTIVE FIELD (test log loss, 5-seed means)")
    print("negative gain = more hops helped")
    print("=" * 92)
    print(grid.round(5).to_string(index=False))
    print(f"\nWrote {output}")
    gbdt = per_seed_metrics(scope, list(seeds), ["gbdt_tuned"])
    print(f"\nGBDT reference: {gbdt['log_loss'].mean():.5f}")


if __name__ == "__main__":
    main()

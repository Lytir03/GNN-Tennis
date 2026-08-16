"""Feature richness x receptive field: does the graph substitute for history?

The "none" column keeps every layer, parameter and LayerNorm and removes only
the messages.  An earlier version used num_layers=0, which also deletes the
normalisation - with height (~185) unnormalised beside B-scores (<1), that
made the B-score tier diverge to 2.89 log loss.  A control has to remove one
thing.

The hop count is the intervention and the feature tier is the moderator.  Read
the grid down a column: as the model is given more per-player information, the
value of message passing should fall to zero and then go negative, because the
graph was only ever reconstructing that information.

Recipes are per tier, each chosen by a validation-only search, because a recipe
is selected *for* an architecture.  The recipe is held fixed across the hop axis
within a tier, which is what keeps every hop contrast one-factor.
"""

from __future__ import annotations

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
EXISTING = {
    ("1_bscore", 1): "gnn_one_hop",
    ("1_bscore", 2): "gnn_tuned",
    ("2_history_nodes", 1): "gnn_history_tuned_one_hop",
    ("2_history_nodes", 2): "gnn_history_tuned",
    ("3_history_decoder", 1): "gnn_decoder_one_hop",
    ("3_history_decoder", 2): "gnn_decoder",
}


def artifact_name(tier: str, hops: int) -> str:
    return EXISTING.get((tier, hops), f"grid_{tier}_{hops}hop")


def main() -> None:
    todo = [
        (tier, hops)
        for tier in TIERS
        for hops in HOPS
        if (tier, hops) not in EXISTING
    ]
    print(f"{len(todo)} cells to compute, {len(EXISTING)} reused\n", flush=True)

    for tier, hops in todo:
        spec = TIERS[tier]
        config = (
            replace(spec["model"], num_layers=1, disable_message_passing=True)
            if hops == "none"
            else replace(spec["model"], num_layers=hops)
        )
        name = artifact_name(tier, hops)
        for seed in SEEDS:
            start = time.time()
            row = run_named(
                "base",
                scope="slams_masters",
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

    names = [artifact_name(t, h) for t in TIERS for h in HOPS]
    per_seed = per_seed_metrics("slams_masters", list(SEEDS), names)
    means = per_seed.groupby("experiment")[["accuracy", "log_loss"]].mean()

    rows = []
    for tier, spec in TIERS.items():
        row = {"tier": spec["label"], "own_recipe": spec["own_recipe"]}
        for hops in HOPS:
            row[f"{hops}_hop"] = means.loc[artifact_name(tier, hops), "log_loss"]
        # The quantity the thesis is about: what message passing is worth.
        row["gain_1_vs_none"] = row["1_hop"] - row["none_hop"]
        row["gain_2_vs_1"] = row["2_hop"] - row["1_hop"]
        rows.append(row)

    grid = pd.DataFrame(rows)
    output = ROOT / "new_work" / "results" / "feature_hop_grid.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(output, index=False)

    print("\n" + "=" * 92)
    print("FEATURE RICHNESS x RECEPTIVE FIELD (test log loss, 5-seed means)")
    print("negative gain = more hops helped")
    print("=" * 92)
    print(grid.round(5).to_string(index=False))
    print(f"\nWrote {output}")
    gbdt = per_seed_metrics("slams_masters", list(SEEDS), ["gbdt_tuned"])
    print(f"\nGBDT reference: {gbdt['log_loss'].mean():.5f}")


if __name__ == "__main__":
    main()

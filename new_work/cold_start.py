# Stage 5: the headline comparison and the cold-start / intransitivity strata.
#
# Two questions, in the order they have to be asked.
#
# 1. Does the model beat the GBDT at full scope? Paired across seeds on
#    identical matches and labels. The winning architecture at the
#    smaller scope used no message passing at all, so this is reported as
#    what it is: a tabular comparison between two models given the same
#    features.
#
# 2. Where, if anywhere, does the graph pay off? Within one architecture,
#    one hop against no messages - an intervention on the receptive
#    field, not a correlation with it. The pre-specified strata are:
#
#    - degree_stratum - cold start, how many prior opponents the
#      thinner-recorded player has. The hypothesis: the graph substitutes
#      for a history the model doesn't have, so it should pay off exactly
#      where that history is missing.
#    - two_hop_only - no prior meeting, but a shared opponent. The
#      intransitivity stratum: the only place a two-hop model has
#      information the one-hop tabular features can't hold.
#    - common_stratum, h2h_stratum - the finer cuts, reported for
#      completeness.
#
# Both get run at every feature tier that has artifacts, because the
# whole thesis claim is that the graph's value decays as the features
# improve. A cold-start effect that only exists at the poorest tier says
# something different from one that survives to the richest.
#
# Run: python new_work/cold_start.py --scope full

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tennis_gnn.compare import paired_delta, per_seed_metrics  # noqa: E402
from tennis_gnn.config import BASE_MODEL  # noqa: E402
from tennis_gnn.data import load_dataset  # noqa: E402
from tennis_gnn.snapshots import load_or_build  # noqa: E402
from tennis_gnn.stratify import (  # noqa: E402
    interaction_test,
    paired_frame,
    stratum_table,
)
from tennis_gnn.structure import add_strata, structure_table  # noqa: E402
from new_work.feature_hop_grid import TIERS, artifact_name  # noqa: E402


# The two interventions, and they answer different questions.  Confusing them
# is easy and was done once in this project's notes: an interaction from the
# "1 vs none" contrast was compared against one from "2 vs 1" as though the two
# numbers were the same quantity.
#
#   1 vs none  - is the graph worth anything at all?
#   2 vs 1     - is *relational* structure worth anything beyond each player's
#                own neighbourhood?  This is the intransitivity test: only a
#                two-layer model can route information along a path through a
#                shared opponent.  A one-layer model sees each player's own
#                opponents and nothing further.
CONTRASTS = {
    "1_vs_none": (1, "none"),
    "2_vs_1": (2, 1),
    # The same depth contrast with both arms retrained at lr 1e-4, because the
    # grid's tier-3 2-hop cells used a rate that is unstable at this scope.  See
    # new_work/depth_test.py.  Only tier 3 has these artifacts; other tiers are
    # skipped for want of a paired seed, which is the correct behaviour.
    "1_vs_none_stable": None,
    "2_vs_1_stable": None,
}

# Both arms retrained at lr 1e-4 so the contrast is one-factor.  Only tier 3 has
# these; other tiers are skipped for want of a paired seed, which is correct.
_STABLE = "depth_3_history_decoder_{}_lr1e4"
STABLE_DEPTH_NAMES = {
    "1_vs_none_stable": {
        "3_history_decoder": (_STABLE.format("1hop"), _STABLE.format("nonehop"))
    },
    "2_vs_1_stable": {
        "3_history_decoder": (_STABLE.format("2hop"), _STABLE.format("1hop"))
    },
}


def hop_pairs(scope: str, contrast: str) -> dict[str, tuple[str, str]]:
    # Tier -> (candidate, baseline), named as that scope stores them.
    #
    # Delegated to the grid rather than restated, because Slam+Masters
    # reuses the earlier gnn_decoder_one_hop-style names for cells that
    # already existed. Hardcoding the full-scope names here made this
    # analysis silently skip every tier at the smaller scope - and
    # skipping is exactly how a contradicting scope goes unnoticed.
    if CONTRASTS[contrast] is None:
        return dict(STABLE_DEPTH_NAMES[contrast])
    candidate_hops, baseline_hops = CONTRASTS[contrast]
    return {
        tier: (
            artifact_name(tier, candidate_hops, scope),
            artifact_name(tier, baseline_hops, scope),
        )
        for tier in TIERS
    }


# The headline: best GNN configuration against the tuned tabular baseline.
# Selected on validation, never on test: one hop at lr 1e-4 scores 0.61384 on
# validation against 0.61600 with messages disabled.  The grid's tier-3 cells are
# not used here because they carry the lr 3e-4 recipe that handicaps precisely
# the message-passing arms.
HEADLINE_GNN = "depth_3_history_decoder_1hop_lr1e4"
HEADLINE_BASELINE = "gbdt_tuned"

STRATA = ("degree_stratum", "two_hop_only", "common_stratum", "h2h_stratum")


def seeds_with(scope: str, *artifacts: str) -> list[int]:
    # Seeds for which every named artifact exists.
    #
    # A paired test over a seed that's missing one arm isn't paired, so
    # this takes the intersection rather than the union.
    root = ROOT / "results" / "frozen_predictions" / scope
    found = []
    for directory in sorted(root.glob("seed_*")):
        seed = int(directory.name.removeprefix("seed_"))
        if all(
            (directory / f"{name}.manifest.json").is_file() for name in artifacts
        ):
            found.append(seed)
    return sorted(found)


def build_structure(scope: str, seed: int) -> pd.DataFrame:
    # Structural labels for every predicted match.
    #
    # The descriptors are symmetric in (a, b) and depend only on the
    # graph, so a single seed's snapshots label every seed - see
    # tennis_gnn/structure.py.
    #
    # Cached to CSV because building it loads the whole snapshot file,
    # which is 3.9 GB at full scope. On a 24 GB machine that can't be
    # done alongside a training run without swapping, and swapping here
    # cost a sixfold slowdown once already. With the cache, every later
    # stratified analysis is cheap.

    # The cache holds the raw descriptors; the stratum cuts are re-applied on
    # read, both because they are cheap and because round-tripping an ordered
    # categorical through CSV loses its order and would scramble the tables.
    cache = ROOT / ".cache" / "structure" / f"{scope}__test.csv"
    if cache.is_file():
        print(f"structure: reusing {cache}")
        return add_strata(pd.read_csv(cache))

    dataset = load_dataset(ROOT, scope=scope)
    # Read the preset from the config rather than naming it here: a stale
    # literal would build a *second* snapshot cache and label the matches from
    # a graph no model in the comparison was trained on.
    snapshots = load_or_build(
        ROOT, dataset, BASE_MODEL.edge_preset, seed=seed, verbose=False
    )
    table = structure_table(snapshots)
    table = table[table["phase"] == "test"]
    cache.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(cache, index=False)
    print(f"structure: wrote {cache}")
    return add_strata(table)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", default="full")
    parser.add_argument("--structure-seed", type=int, default=42)
    parser.add_argument("--output-dir", default="new_work/results")
    args = parser.parse_args()
    scope = args.scope
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"1. HEADLINE  {HEADLINE_GNN} vs {HEADLINE_BASELINE}  [{scope}]")
    print("=" * 72)
    headline_seeds = seeds_with(scope, HEADLINE_GNN, HEADLINE_BASELINE)
    print(f"seeds: {headline_seeds}\n")
    per_seed = per_seed_metrics(
        scope, headline_seeds, [HEADLINE_GNN, HEADLINE_BASELINE]
    )
    print(per_seed.to_string(index=False))
    headline = paired_delta(per_seed, HEADLINE_GNN, HEADLINE_BASELINE)
    print("\npaired (negative log loss delta = GNN better):")
    print(headline.to_string(index=False))
    headline.to_csv(out / f"headline_{scope}.csv", index=False)

    print("\n" + "=" * 72)
    print("2. WHERE THE GRAPH PAYS  (per tier, per intervention)")
    print("=" * 72)
    structure = build_structure(scope, args.structure_seed)
    print(f"structure table: {len(structure)} test matches\n")

    stratum_rows, interaction_rows = [], []
    for contrast_name in CONTRASTS:
        print("\n" + "#" * 72)
        print(f"# INTERVENTION: {contrast_name}")
        print("#" * 72)
        for tier, (candidate, baseline) in hop_pairs(scope, contrast_name).items():
            seeds = seeds_with(scope, candidate, baseline)
            if len(seeds) < 2:
                print(f"-- {tier}: skipped, {len(seeds)} paired seed(s)\n")
                continue
            data = paired_frame(
                ROOT, scope, seeds, candidate, baseline, structure
            )
            print(
                f"-- {tier}  seeds {seeds}  "
                "(negative = the deeper model is better)"
            )
            for stratum in STRATA:
                table = stratum_table(data, stratum)
                table.insert(0, "intervention", contrast_name)
                table.insert(1, "tier", tier)
                table.insert(2, "by", stratum)
                stratum_rows.append(table)
                print(f"\n  by {stratum}:")
                print("   " + table.to_string(index=False).replace("\n", "\n   "))
                # An interaction is only defined for a two-level cut; the
                # ordered cuts are described by the table above.
                if table["stratum"].nunique() == 2:
                    interaction = interaction_test(data, stratum)
                    interaction.insert(0, "intervention", contrast_name)
                    interaction.insert(1, "tier", tier)
                    interaction.insert(2, "by", stratum)
                    interaction_rows.append(interaction)
                    print(
                        "   interaction: "
                        + interaction.to_string(index=False).replace(
                            "\n", "\n   "
                        )
                    )
            print()

    if stratum_rows:
        frame = pd.concat(stratum_rows, ignore_index=True)
        frame.to_csv(out / f"strata_{scope}.csv", index=False)
        print(f"Wrote {out / f'strata_{scope}.csv'}")
        comparisons = len(frame)
        print(
            f"\n{comparisons} stratum comparisons were computed.  No "
            "multiplicity correction is applied;\nread any single starred cell "
            "as a hypothesis, not a result, unless it was pre-specified."
        )
    if interaction_rows:
        frame = pd.concat(interaction_rows, ignore_index=True)
        frame.to_csv(out / f"interactions_{scope}.csv", index=False)
        print(f"Wrote {out / f'interactions_{scope}.csv'}")


if __name__ == "__main__":
    main()

"""A clean two-hop-versus-one-hop test at the richest feature tier.

**Why this run exists.**  The intransitivity hypothesis - that a graph model
earns its place on matches joined through a shared opponent - is a claim about
*depth*.  Only a two-layer model can route information along such a path; a
one-layer model sees each player's own opponents and stops.  So the test is
2 hops against 1 hop, holding everything else fixed.

At full scope that test came out inconclusive, and for an avoidable reason.  The
grid ran tier 3 on the recipe selected at the smaller scope (lr 3e-4), and the
diagnostic in `new_work/twohop_diagnostic.py` showed that rate is unstable at two
hops on five times the data: lowering it to 1e-4 recovered 0.0146 log loss and
stabilised runtimes.  The resulting 2-hop cells were noisy enough that the
depth interaction had a confidence interval nine times wider than tier 1's.

The fix is not to compare the repaired 2-hop cells against the grid's 1-hop
cells - those were trained at 3e-4, so the contrast would differ in two things
at once, which is the exact error this project has been correcting throughout.
**Both arms are retrained here at lr 1e-4**, so depth is the only difference.

Cost is about 2.2 hours; it is the price of one honest answer to the project's
central question.

Run: python new_work/depth_test.py
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tennis_gnn.run import run_named  # noqa: E402
from new_work.feature_hop_grid import BASE_RECIPE, TIERS  # noqa: E402

# BASE_RECIPE is lr 1e-4; the grid's DECODER_RECIPE is the same recipe at 3e-4.
# Using it for both arms is what makes this one-factor.
STABLE_RECIPE = BASE_RECIPE
SEEDS = (42, 123, 456, 789, 2026)
TIER = "3_history_decoder"


def artifact_for(hops: int) -> str:
    return f"depth_{TIER}_{hops}hop_lr1e4"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", default="full")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--hops", type=int, nargs="+", default=[1, 2])
    args = parser.parse_args()

    model = TIERS[TIER]["model"]
    print(
        f"depth test: tier {TIER}, hops {args.hops}, seeds {args.seeds}, "
        f"lr={STABLE_RECIPE.learning_rate}\n",
        flush=True,
    )
    for hops in args.hops:
        config = replace(model, num_layers=hops)
        name = artifact_for(hops)
        for seed in args.seeds:
            start = time.time()
            row = run_named(
                "base",
                scope=args.scope,
                seed=seed,
                train_config=STABLE_RECIPE,
                verbose=False,
                artifact_name=name,
                model_config=config,
            )
            print(
                f"{TIER} {hops} hop  seed {seed:<5d} "
                f"val_ll={row['val_log_loss']:.4f} "
                f"test_ll={row['test_log_loss']:.4f} "
                f"[{time.time() - start:.0f}s]",
                flush=True,
            )
    print("\nDEPTH TEST DONE")


if __name__ == "__main__":
    main()

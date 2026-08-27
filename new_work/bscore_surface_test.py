# Does the direct B-score logit skip connection help from a surface-matched
# value instead of the surface-agnostic one?
#
# The GNN's direct-to-logit skip connection (model.py, direct_bscore_logit)
# always read raw_bscore_general - the surface-agnostic B-score - even
# though node features already carry all four B-scores (general, hard,
# clay, grass) and the GBDT baseline gets a surface-matched B-score as an
# explicit feature. That's a real parity gap: the GBDT's tabular feature
# set already resolves "which B-score applies to this match" per row, while
# the GNN's skip connection never did.
#
# This is a one-factor test at today's best-known recipe, not a re-tuning
# run: BASE_MODEL and TUNED_TRAINING are both used unchanged, on the
# smaller slams_masters scope for speed. Two arms, everything else held
# fixed:
#   - control:  direct_bscore_surface=False (today's default)
#   - variant:  direct_bscore_surface=True  (surface-matched skip connection)
#
# Run: python new_work/bscore_surface_test.py

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tennis_gnn.config import BASE_MODEL  # noqa: E402
from tennis_gnn.run import TUNED_TRAINING, run_named  # noqa: E402

SCOPE = "slams_masters"
SEEDS = (42, 123, 456, 789, 2026)
ARMS = {
    "bscore_general_control": BASE_MODEL,
    "bscore_surface_variant": replace(BASE_MODEL, direct_bscore_surface=True),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", default=SCOPE)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = parser.parse_args()

    print(
        f"bscore surface test: scope {args.scope}, seeds {args.seeds}\n",
        flush=True,
    )

    rows = []
    for artifact_name, model_config in ARMS.items():
        for seed in args.seeds:
            start = time.time()
            row = run_named(
                "base",
                scope=args.scope,
                seed=seed,
                train_config=TUNED_TRAINING,
                verbose=False,
                artifact_name=artifact_name,
                model_config=model_config,
            )
            rows.append(row)
            print(
                f"{artifact_name:<24s} seed {seed:<5d} "
                f"val_ll={row['val_log_loss']:.4f} "
                f"test_ll={row['test_log_loss']:.4f} "
                f"[{time.time() - start:.0f}s]",
                flush=True,
            )

    print("\nMean test log loss by arm:")
    by_arm: dict[str, list[float]] = {}
    for row in rows:
        by_arm.setdefault(row["experiment"], []).append(row["test_log_loss"])
    for name, values in by_arm.items():
        mean = sum(values) / len(values)
        print(f"  {name:<24s} mean test_ll={mean:.4f}  (n={len(values)} seeds)")

    print("\nBSCORE SURFACE TEST DONE")


if __name__ == "__main__":
    main()

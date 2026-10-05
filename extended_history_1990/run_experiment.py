# Does a decade of warmup (1980-1989) instead of the usual five years, with
# online training starting 1990 instead of 2011, change anything?
#
# Uses today's best-known GNN recipe unchanged (BASE_MODEL, TUNED_TRAINING)
# on the new slams_masters_1990 scope - no re-tuning, this is a "try it and
# see" run, not a new search. Still fully online: weights only update on
# training-phase blocks, but features/history/graph keep updating
# block-by-block straight through val and test, same protocol as every
# other scope in this project.
#
# Data prerequisites (run once, before this script):
#   python extended_history_1990/build_matches.py
#   python extended_history_1990/build_bscore.py
#
# Run: python extended_history_1990/run_experiment.py [--seeds 42]

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tennis_gnn.config import BASE_MODEL  # noqa: E402
from tennis_gnn.run import TUNED_TRAINING, run_named  # noqa: E402

SCOPE = "slams_masters_1990"
SEEDS = (42, 123, 456, 789, 2026)
ARTIFACT_NAME = "extended_1990_warmup"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", default=SCOPE)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = parser.parse_args()

    print(f"extended warmup test: scope {args.scope}, seeds {args.seeds}\n", flush=True)

    rows = []
    for seed in args.seeds:
        start = time.time()
        row = run_named(
            "base",
            scope=args.scope,
            seed=seed,
            train_config=TUNED_TRAINING,
            verbose=False,
            artifact_name=ARTIFACT_NAME,
            model_config=BASE_MODEL,
        )
        rows.append(row)
        print(
            f"seed {seed:<5d} val_ll={row['val_log_loss']:.4f} "
            f"test_ll={row['test_log_loss']:.4f} "
            f"[{time.time() - start:.0f}s]",
            flush=True,
        )

    if rows:
        mean_val = sum(r["val_log_loss"] for r in rows) / len(rows)
        mean_test = sum(r["test_log_loss"] for r in rows) / len(rows)
        print(
            f"\nmean val_ll={mean_val:.4f}  mean test_ll={mean_test:.4f}  "
            f"(n={len(rows)} seeds)"
        )

    print("\nEXTENDED WARMUP TEST DONE")


if __name__ == "__main__":
    main()

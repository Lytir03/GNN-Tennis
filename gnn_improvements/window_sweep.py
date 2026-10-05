# Is message passing stopping helping because the graph is mostly stale?
#
# The hop grid found the value of two hops decaying from +0.0898 with no
# features, to +0.0221 with B-score, to -0.0036 once the history statistics
# reach the decoder. One explanation is substitution: the graph reconstructs
# what the history statistics state directly, so once they are supplied the
# graph is redundant. Another is staleness, and the graph is measurably stale.
#
# Measured on slams_masters test blocks, with the current 3-year window:
#
#   edges older than one year       65% / 67% / 81%   (2016 / 2018 / 2020)
#   nodes last playing >1 year ago  25% / 29% / 36%
#   nodes with degree 1             16% / 18% / 17%
#
# Stale nodes have median degree 1-3 against 19-24 for active players, and the
# fraction rises over time - so the graph the model is *tested* on is staler
# than the one it trained on. The decay weights on those old edges are only
# 0.28-0.41, but BASE_MODEL aggregates with sum, and old edges outnumber
# recent ones two to four times over. Volume beats weight: the recency decay
# is real and is being swamped by count.
#
# If that is the cause, shortening the window should push the two-hop
# contrast back towards positive at the richest tier. If the contrast is
# unchanged, staleness is not the mechanism and substitution stands.
#
# MAX_HISTORY_DAYS = 365 * 3 has never been varied - the same category of
# never-touched constant as replay_buffer_size, which did turn out to matter.
#
# Tier 1 is included as a positive control: two hops help there (+0.0221), so
# the window should not destroy that if the sweep is behaving sensibly.
#
# Each window needs its own snapshot build; the default (1095) reuses the
# existing cache.
#
# Run: python gnn_improvements/window_sweep.py

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

from dataclasses import replace as _replace_train  # noqa: E402
from tennis_gnn.config import BASE_MODEL  # noqa: E402
from tennis_gnn.run import TUNED_TRAINING, run_named  # noqa: E402

SEEDS = (42, 123, 456, 789, 2026)
WINDOWS = (365, 547, 730, 1095)
TIERS = {
    "t1_bscore": BASE_MODEL,
    "t3_history_decoder": replace(BASE_MODEL, history_to_decoder=True),
}
HOPS = ("none", 1, 2)


def config_for(model, hops, aggregation="sum"):
    if hops == "none":
        model = replace(model, num_layers=1, disable_message_passing=True)
    else:
        model = replace(model, num_layers=int(hops))
    return replace(model, aggregation=aggregation)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", default="slams_masters")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument(
        "--windows", type=int, nargs="+", default=list(WINDOWS)
    )
    parser.add_argument(
        "--hops", nargs="+", default=[str(h) for h in HOPS],
        help="any of none, 1, 2",
    )
    parser.add_argument("--tiers", nargs="+", default=list(TIERS))
    parser.add_argument(
        "--aggregation", nargs="+", default=["sum"],
        help="any of sum, mean - tests whether aggregation, not just the "
             "window, controls how much stale edges dilute recent ones",
    )
    parser.add_argument(
        "--buffer-size", type=int, default=None,
        help="override TrainConfig.replay_buffer_size (default 200)",
    )
    args = parser.parse_args()

    print(
        f"window sweep: scope {args.scope}, windows {args.windows} days, "
        f"tiers {args.tiers}, hops {args.hops}\n",
        flush=True,
    )

    train_config = TUNED_TRAINING
    if args.buffer_size is not None:
        train_config = _replace_train(
            train_config, replay_buffer_size=args.buffer_size
        )

    rows = []
    for window in args.windows:
        for tier in args.tiers:
            model = TIERS[tier]
            for hops in args.hops:
                for agg in args.aggregation:
                    name = f"win{window}_{tier}_{hops}hop_{agg}"
                    if args.buffer_size is not None:
                        name += f"_buf{args.buffer_size}"
                    for seed in args.seeds:
                        start = time.time()
                        row = run_named(
                            "base",
                            scope=args.scope,
                            seed=seed,
                            train_config=train_config,
                            verbose=False,
                            artifact_name=name,
                            model_config=config_for(model, hops, agg),
                            graph_window_days=window,
                        )
                        row.update(
                            window=window, tier=tier, hops=str(hops),
                            aggregation=agg,
                        )
                        rows.append(row)
                        print(
                            f"w{window:<5d} {tier:20s} {str(hops):>4s} hop "
                            f"{agg:4s} seed {seed:<5d} "
                            f"val_ll={row['val_log_loss']:.4f} "
                            f"test_ll={row['test_log_loss']:.4f} "
                            f"[{time.time() - start:.0f}s]",
                            flush=True,
                        )

    frame = pd.DataFrame(rows)
    output = ROOT / "gnn_improvements" / "results"
    output.mkdir(parents=True, exist_ok=True)
    # The windows and hops belong in the name: without them a later sweep
    # silently overwrites an earlier one's results, which already cost the
    # first four windows of this experiment (recovered from logs).
    tag = "w" + "-".join(str(w) for w in args.windows)
    tag += "_h" + "-".join(str(h) for h in args.hops)
    tag += "_a" + "-".join(args.aggregation)
    if args.buffer_size is not None:
        tag += f"_buf{args.buffer_size}"
    path = output / f"window_sweep_{args.scope}_{tag}.csv"
    frame.to_csv(path, index=False)

    print("\nmean test log loss:")
    print(
        frame.pivot_table(
            index=["tier", "window"], columns="hops", values="test_log_loss"
        ).to_string()
    )
    table = frame.pivot_table(
        index=["tier", "window"], columns="hops", values="test_log_loss"
    )
    if "none" in table.columns:
        for hop in ("1", "2"):
            if hop not in table.columns:
                continue
            print(f"\nvalue of {hop} hop(s) over none (positive = helps):")
            for (tier, window), row in table.iterrows():
                print(
                    f"  {tier:20s} window {window:5d}d  "
                    f"{row['none'] - row[hop]:+.4f}"
                )
    print(f"\nWrote {path}")
    print(
        "\nIf the tier-3 contrast climbs as the window shortens, staleness is "
        "the mechanism. If it is flat, substitution is, and shortening the "
        "window is not the fix."
    )


if __name__ == "__main__":
    main()

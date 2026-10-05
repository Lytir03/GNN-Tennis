# Coverage check and regression test for expanding beyond Slams + Masters.
#
# This deliberately does NOT expand anything yet. It reports how far
# B-score coverage falls short of the raw data and refuses to build a
# wider dataset, because the failure mode is silent: matches without a
# B-score snapshot fall back to the 25th-percentile default, and
# direct_bscore_logit feeds that default straight into the logit as the
# model's skill prior. A run like that would complete, produce plausible
# numbers, and be meaningless for most of its rows.
#
# The order of operations that keeps this honest:
#
# 1. Re-run preprocess/graph*.ipynb over the full tour with the logic
#    unchanged.
# 2. Run python new_work/expand_data.py --check-regression. It verifies
#    the regenerated snapshots reproduce the current values exactly on
#    the 244 tournaments already covered. If that fails, the
#    regeneration changed the definition and nothing downstream is
#    comparable to the published results.
# 3. Only then build the wider scope, and only then consider rewriting
#    the four surface notebooks into one parameterised script.
#
# Never change the definition and the scope in the same step: if the
# numbers move, you can't tell which one did it.

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw" / "sackmann"
START, END = "2011-01-01", "2021-01-01"


def raw_matches() -> pd.DataFrame:
    frames = [
        pd.read_csv(path, low_memory=False)
        for path in sorted(glob.glob(str(RAW / "atp_matches_20*.csv")))
    ]
    matches = pd.concat(frames, ignore_index=True)
    matches["tourney_date"] = pd.to_datetime(
        matches["tourney_date"], format="%Y%m%d", errors="coerce"
    )
    matches["tourney_id"] = matches["tourney_id"].astype(str)
    return matches[
        (matches["tourney_date"] >= START) & (matches["tourney_date"] < END)
    ]


def coverage() -> pd.DataFrame:
    matches = raw_matches()
    snapshots = pd.read_csv(PROCESSED / "bscore_snapshots.csv")
    covered = set(snapshots["tourney_id"].astype(str).unique())
    matches = matches.assign(covered=matches["tourney_id"].isin(covered))
    return (
        matches.groupby(["tourney_level", "covered"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={True: "b_score_present", False: "b_score_missing"})
    )


def check_regression() -> int:
    # Do regenerated snapshots still match on tournaments already covered?
    current = PROCESSED / "bscore_snapshots.csv"
    regenerated = PROCESSED / "bscore_snapshots_full.csv"
    if not regenerated.is_file():
        print(
            f"Nothing to check: {regenerated.name} does not exist yet.\n"
            "Re-run preprocess/graph*.ipynb over the full tour first, writing "
            "to that filename so the current file stays intact."
        )
        return 1

    key = ["tourney_id", "round_order", "player"]
    old = pd.read_csv(current)
    new = pd.read_csv(regenerated)
    for frame in (old, new):
        frame["tourney_id"] = frame["tourney_id"].astype(str)

    merged = old.merge(new, on=key, how="inner", suffixes=("_old", "_new"))
    if len(merged) != len(old):
        print(
            f"MISSING ROWS: {len(old) - len(merged)} of {len(old)} current "
            "snapshot rows are absent from the regenerated file. The "
            "regeneration dropped data it should have reproduced."
        )
        return 1

    difference = (merged["bscore_general_old"] - merged["bscore_general_new"]).abs()
    worst = float(difference.max())
    print(f"Compared {len(merged)} overlapping rows; max abs difference {worst:.3e}")
    if worst > 1e-9:
        print(
            "REGRESSION: the regenerated snapshots differ from the published "
            "ones. The definition changed, so no downstream result is "
            "comparable to the published results. Fix this before expanding."
        )
        return 1
    print("OK: regeneration reproduces the published snapshots exactly.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-regression", action="store_true")
    args = parser.parse_args()

    if args.check_regression:
        return check_regression()

    table = coverage()
    print("Raw ATP matches 2011-2020, by B-score coverage:\n")
    print(table.to_string())
    present = int(table.get("b_score_present", pd.Series(dtype=int)).sum())
    missing = int(table.get("b_score_missing", pd.Series(dtype=int)).sum())
    print(f"\ncovered {present}   missing {missing}   total {present + missing}")
    print(
        f"\nExpansion would multiply the usable data by "
        f"{(present + missing) / present:.1f}x - but only after B-score exists "
        "for those matches."
    )
    print(
        "\nNot building a wider scope: matches without a snapshot would take "
        "the 25th-percentile default as their skill prior, which fails "
        "silently. See new_work/STATUS.md, Task B."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

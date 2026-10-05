# Builds atp_matches_slams_masters_1990.csv: the same slams+masters filter
# as preprocess/data_prep_slams_only.ipynb, just starting 1980 instead of
# 2006, so there's a decade of match history (1980-1989) to warm up the
# B-score graph before online training starts in 1990.
#
# Reproduces the notebook's logic bit-for-bit (concat every raw yearly file,
# filter by date and tourney_level, sort) rather than improving it - the
# only change is the start date.
#
# Run: python extended_history_1990/build_matches.py

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "sackmann"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "atp_matches_slams_masters_1990.csv"

HISTORY_START = "1980-01-01"
DEFAULT_LEVELS = "G,M"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--levels", default=DEFAULT_LEVELS,
        help="comma-separated tourney_level codes to keep, e.g. G,M or G,M,A",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--history-start", default=HISTORY_START)
    args = parser.parse_args()
    tour_keep = args.levels.split(",")

    files = sorted(RAW.glob("atp_matches_*.csv"))
    matches = pd.concat((pd.read_csv(f, low_memory=False) for f in files), ignore_index=True)

    matches["tourney_date"] = pd.to_datetime(matches["tourney_date"], format="%Y%m%d")
    matches = matches[matches["tourney_date"] >= args.history_start]
    matches = matches.sort_values("tourney_date", ascending=True)

    matches = matches[matches["tourney_level"].isin(tour_keep)]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(args.output, index=False)
    print(
        f"{len(matches)} matches, {matches['tourney_date'].min().date()} to "
        f"{matches['tourney_date'].max().date()} -> {args.output}"
    )


if __name__ == "__main__":
    main()

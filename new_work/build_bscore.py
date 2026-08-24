# Rebuilds the B-score snapshots, parameterised by scope and surface.
#
# This replaces four near-identical notebooks (preprocess/graph.ipynb and
# its _clay1, _grass1, _hard1 copies) that only differed in a surface
# filter and their output filenames.
#
# The logic is reproduced exactly, not improved - that's the whole point.
# The verification step below runs this over the current tournament scope
# and requires it to reproduce the published snapshots to the last bit.
# Only once that passes is it safe to change the scope, because then any
# difference in the downstream results is attributable to the data and
# not to a redefinition.
#
# Method, following the original:
# - history is 2006-01-01 to 2010-12-31, the rolling period is 2011 onwards
# - for each tournament-round block, build a directed graph loser -> winner
#   over every match played before it, weighting each by 1 / (1 + age_days / 365)
# - B-score is the weighted eigenvector centrality of that graph, so
#   beating a strong player is worth more than beating a weak one
# - the block's players are scored from the graph before the block, then
#   the block gets appended to the history - that's what keeps it
#   leakage-free
#
# Usage:
#   python new_work/build_bscore.py --verify           # reproduce & compare
#   python new_work/build_bscore.py --scope full       # rebuild wider

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import networkx as nx
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw" / "sackmann"

HISTORY_START = "2006-01-01"
ROLLING_START = "2011-01-01"
ALPHA_DAYS = 365
ROUND_ORDER = {
    "R128": 1, "R64": 2, "R32": 3, "R16": 4, "QF": 5, "SF": 6, "F": 7,
}

# The whitelist from preprocess/data_prep.ipynb, kept verbatim so `--scope
# current` reproduces the published dataset.  Twelve of these match nothing in
# the raw data (e.g. 'Indian Wells Master' is missing its 's', 'Shangai
# Masters' is misspelt); harmless here only because the correctly spelt
# '... Masters' entries also appear.  This fragility is the reason to move to a
# level-based filter rather than a name-based one.
TOUR_KEEP = [
    "Wimbledon", "Roland Garros", "US Open", "Australian Open", "Madrid",
    "Montecarlo", "Rome", "Barcelona", "Munich", "Hamburg", "Rio de Janeiro",
    "Shanghai", "Paris", "Indian Wells", "Cincinnati", "Miami", "Toronto",
    "Montreal", "Halle", "Queen's Club", "Monte Carlo Masters", "Paris Masters",
    "Cincinnati Masters", "Indian Wells Master", "Miami Masters",
    "Rome Masters", "Madrid Masters", "Canada Masters", "Shangai Masters",
    "s Hertogenbosch", "Mallorca", "Eastbourne",
]

SURFACES = {
    "general": (None, "bscore_general", "bscore_snapshots.csv"),
    "clay": ("Clay", "bscore_clay", "bscore_snapshots_clay.csv"),
    "hard": ("Hard", "bscore_hard", "bscore_snapshots_hard.csv"),
    "grass": ("Grass", "bscore_grass", "bscore_snapshots_grass.csv"),
}


def load_matches(scope: str) -> pd.DataFrame:
    # Assembles the match table for a scope, straight from raw.
    frames = [
        pd.read_csv(path, low_memory=False)
        for path in sorted(RAW.glob("atp_matches_[0-9]*.csv"))
    ]
    matches = pd.concat(frames, ignore_index=True)
    matches["tourney_date"] = pd.to_datetime(
        matches["tourney_date"], format="%Y%m%d", errors="coerce"
    )
    matches = matches[matches["tourney_date"] >= HISTORY_START]
    matches = matches.sort_values("tourney_date", ascending=True)

    if scope == "current":
        matches = matches[matches["tourney_name"].isin(TOUR_KEEP)]
    elif scope == "full":
        # Every main-tour event: Grand Slams, Masters 1000 and ATP 250/500.
        # Davis Cup ("D") and the Finals ("F") are left out because their draw
        # formats are not comparable tournament rounds.
        matches = matches[matches["tourney_level"].isin(["G", "M", "A"])]
    else:
        raise ValueError("scope must be 'current' or 'full'")
    return matches.reset_index(drop=True)


def build_edge_frame(history: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    frame = history.copy()
    frame["age_days"] = (t - frame["tourney_date"]).dt.days
    frame["decay_weight"] = 1 / (1 + frame["age_days"] / ALPHA_DAYS)
    return (
        frame.groupby(["loser_name", "winner_name"], as_index=False)[
            "decay_weight"
        ]
        .sum()
        .rename(
            columns={
                "loser_name": "source",
                "winner_name": "target",
                "decay_weight": "weight",
            }
        )
    )


def build_graph(edges: pd.DataFrame, players=None) -> nx.DiGraph:
    graph = nx.from_pandas_edgelist(
        edges,
        source="source",
        target="target",
        edge_attr="weight",
        create_using=nx.DiGraph(),
    )
    if players is not None:
        graph.add_nodes_from(players)
    return graph


def compute_bscore(graph: nx.DiGraph) -> dict:
    return nx.eigenvector_centrality(
        graph, weight="weight", max_iter=1000, tol=1e-9
    )


def build_snapshots(
    matches: pd.DataFrame, surface: str | None, *, verbose: bool = True
) -> tuple[pd.DataFrame, float]:
    # Rolls the B-score forward one tournament-round at a time.
    if surface is not None:
        matches = matches[matches["surface"] == surface]

    matches = matches.copy()
    matches["round_order"] = matches["round"].map(ROUND_ORDER)

    history = matches[
        (matches["tourney_date"] >= HISTORY_START)
        & (matches["tourney_date"] < ROLLING_START)
    ].copy()
    future = matches[matches["tourney_date"] >= ROLLING_START].copy()

    # The fallback for a player absent from the graph, from the initial period.
    initial_players = pd.unique(
        pd.concat(
            [history["winner_name"], history["loser_name"]], ignore_index=True
        )
    )
    initial = compute_bscore(
        build_graph(
            build_edge_frame(history, pd.Timestamp("2010-12-31")),
            players=initial_players,
        )
    )
    default_bscore = float(pd.Series(list(initial.values())).quantile(0.25))

    blocks = (
        future[["tourney_date", "tourney_id", "round_order"]]
        .drop_duplicates()
        .sort_values(["tourney_date", "tourney_id", "round_order"])
        .reset_index(drop=True)
    )

    current = history
    rows: list[dict] = []
    start = time.time()
    for index, (_, tourney_id, round_order) in enumerate(
        blocks.itertuples(index=False, name=None)
    ):
        block = future[
            (future["tourney_id"] == tourney_id)
            & (future["round_order"] == round_order)
        ]
        if block.empty:
            continue
        t_block = block["tourney_date"].iloc[0]

        players = pd.unique(
            pd.concat(
                [current["winner_name"], current["loser_name"]],
                ignore_index=True,
            )
        )
        scores = compute_bscore(
            build_graph(build_edge_frame(current, t_block), players=players)
        )
        rows.extend(
            {
                "tourney_id": tourney_id,
                "round_order": round_order,
                "player": player,
                "score": score,
            }
            for player, score in scores.items()
        )
        current = pd.concat([current, block], ignore_index=True)

        if verbose and index % 100 == 0:
            print(
                f"    block {index:5d}/{len(blocks)} "
                f"players={len(scores):5d} [{time.time() - start:.0f}s]",
                flush=True,
            )

    return pd.DataFrame(rows), default_bscore


def verify() -> int:
    # Reproduces the published snapshots on the current scope, exactly.
    matches = load_matches("current")
    print(f"Rebuilding on the current scope ({len(matches)} matches)\n")
    ok = True
    for key, (surface, column, filename) in SURFACES.items():
        published_path = PROCESSED / filename
        if not published_path.is_file():
            print(f"  {key:8s} SKIP (no published file)")
            continue
        print(f"  {key} ...", flush=True)
        rebuilt, _ = build_snapshots(matches, surface, verbose=False)
        rebuilt = rebuilt.rename(columns={"score": column})
        published = pd.read_csv(published_path)
        for frame in (rebuilt, published):
            frame["tourney_id"] = frame["tourney_id"].astype(str)

        key_columns = ["tourney_id", "round_order", "player"]
        merged = published.merge(
            rebuilt, on=key_columns, how="outer",
            suffixes=("_published", "_rebuilt"), indicator=True,
        )
        only = merged["_merge"].value_counts()
        missing = int(only.get("left_only", 0)) + int(only.get("right_only", 0))
        both = merged[merged["_merge"] == "both"]
        worst = float(
            (both[f"{column}_published"] - both[f"{column}_rebuilt"])
            .abs()
            .max()
        )
        status = "OK" if missing == 0 and worst < 1e-9 else "MISMATCH"
        if status == "MISMATCH":
            ok = False
        print(
            f"  {key:8s} {status}  rows published={len(published)} "
            f"rebuilt={len(rebuilt)} unmatched={missing} "
            f"max_abs_diff={worst:.3e}"
        )

    print()
    if ok:
        print(
            "VERIFIED: this script reproduces the published B-score snapshots "
            "exactly.\nThe definition is unchanged, so a scope change is now "
            "attributable to the data alone."
        )
        return 0
    print(
        "FAILED: the rebuild differs from the published snapshots. Do NOT "
        "expand the scope until this reproduces exactly - otherwise a later "
        "change in results cannot be attributed to scope rather than method."
    )
    return 1


def build(scope: str, suffix: str) -> int:
    matches = load_matches(scope)
    print(
        f"scope={scope}: {len(matches)} matches, "
        f"{matches['tourney_id'].nunique()} tournaments, "
        f"{matches['tourney_date'].min().date()} to "
        f"{matches['tourney_date'].max().date()}\n"
    )
    for key, (surface, column, filename) in SURFACES.items():
        print(f"  building {key} ...", flush=True)
        snapshots, default_bscore = build_snapshots(matches, surface)
        snapshots = snapshots.rename(columns={"score": column})
        output = PROCESSED / filename.replace(".csv", f"_{suffix}.csv")
        snapshots.to_csv(output, index=False)
        print(
            f"  {key:8s} {len(snapshots)} rows, default={default_bscore:.6g} "
            f"-> {output.name}\n",
            flush=True,
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--scope", choices=("current", "full"))
    parser.add_argument("--suffix", default="full")
    args = parser.parse_args()
    if args.verify:
        return verify()
    if args.scope:
        return build(args.scope, args.suffix)
    parser.error("pass --verify or --scope")


if __name__ == "__main__":
    raise SystemExit(main())

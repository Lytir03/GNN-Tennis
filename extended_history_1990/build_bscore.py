# Rebuilds B-score snapshots for the extended-history experiment: warmup
# 1980-1989, rolling (online-trained) period starting 1990.
#
# Adapted from new_work/build_bscore.py - same method, reproduced rather
# than improved: for each tournament-round block, build a directed graph
# loser -> winner over every match played before it, weighting each by
# 1 / (1 + age_days / 365); B-score is the weighted eigenvector centrality
# of that graph, computed leakage-free (a block is scored from the graph
# before it, then appended to history).
#
# Kept as a separate script rather than adding flags to new_work/build_bscore.py
# so this experiment never touches a file the project's published results
# depend on. HISTORY_START/ROLLING_START are CLI-configurable here (they're
# hardcoded in new_work/build_bscore.py) since this experiment needs
# different values (1980-01-01 / 1990-01-01) than every existing scope
# (2006-01-01 / 2011-01-01).
#
# The match filter mirrors extended_history_1990/build_matches.py exactly
# (tourney_level in {G, M}) rather than new_work/build_bscore.py's
# name-based TOUR_KEEP whitelist, because the B-score graph must be built
# from the identical match set the pipeline will load - not just an
# equivalent one.
#
# Output goes to data/processed/ (suffixed) because that's the only place
# tennis_gnn.data._load_bscore_snapshots looks; the rest of this experiment
# stays inside extended_history_1990/.
#
# Usage:
#   python extended_history_1990/build_bscore.py \
#       --suffix 1990 --history-start 1980-01-01 --rolling-start 1990-01-01

from __future__ import annotations

import argparse
from pathlib import Path
import time

import networkx as nx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw" / "sackmann"

ALPHA_DAYS = 365
ROUND_ORDER = {
    "R128": 1, "R64": 2, "R32": 3, "R16": 4, "QF": 5, "SF": 6, "F": 7,
}
DEFAULT_LEVELS = "G,M"

SURFACES = {
    "general": (None, "bscore_general", "bscore_snapshots.csv"),
    "clay": ("Clay", "bscore_clay", "bscore_snapshots_clay.csv"),
    "hard": ("Hard", "bscore_hard", "bscore_snapshots_hard.csv"),
    "grass": ("Grass", "bscore_grass", "bscore_snapshots_grass.csv"),
}


def load_matches(history_start: str, levels: list[str]) -> pd.DataFrame:
    frames = [
        pd.read_csv(path, low_memory=False)
        for path in sorted(RAW.glob("atp_matches_[0-9]*.csv"))
    ]
    matches = pd.concat(frames, ignore_index=True)
    matches["tourney_date"] = pd.to_datetime(
        matches["tourney_date"], format="%Y%m%d", errors="coerce"
    )
    matches = matches[matches["tourney_date"] >= history_start]
    matches = matches[matches["tourney_level"].isin(levels)]
    return matches.sort_values("tourney_date", ascending=True).reset_index(drop=True)


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
    matches: pd.DataFrame,
    surface: str | None,
    *,
    history_start: str,
    rolling_start: str,
    verbose: bool = True,
) -> tuple[pd.DataFrame, float]:
    if surface is not None:
        matches = matches[matches["surface"] == surface]

    matches = matches.copy()
    matches["round_order"] = matches["round"].map(ROUND_ORDER)

    history = matches[
        (matches["tourney_date"] >= history_start)
        & (matches["tourney_date"] < rolling_start)
    ].copy()
    future = matches[matches["tourney_date"] >= rolling_start].copy()

    initial_players = pd.unique(
        pd.concat(
            [history["winner_name"], history["loser_name"]], ignore_index=True
        )
    )
    rolling_start_ts = pd.Timestamp(rolling_start) - pd.Timedelta(days=1)
    initial = compute_bscore(
        build_graph(
            build_edge_frame(history, rolling_start_ts),
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


def build(
    suffix: str, history_start: str, rolling_start: str, levels: list[str]
) -> int:
    matches = load_matches(history_start, levels)
    print(
        f"{len(matches)} matches, {matches['tourney_id'].nunique()} "
        f"tournaments, {matches['tourney_date'].min().date()} to "
        f"{matches['tourney_date'].max().date()} "
        f"(history_start={history_start}, rolling_start={rolling_start})\n"
    )
    for key, (surface, column, filename) in SURFACES.items():
        print(f"  building {key} ...", flush=True)
        snapshots, default_bscore = build_snapshots(
            matches,
            surface,
            history_start=history_start,
            rolling_start=rolling_start,
        )
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
    parser.add_argument("--suffix", default="1990")
    parser.add_argument("--history-start", default="1980-01-01")
    parser.add_argument("--rolling-start", default="1990-01-01")
    parser.add_argument(
        "--levels", default=DEFAULT_LEVELS,
        help="comma-separated tourney_level codes to keep, e.g. G,M or G,M,A",
    )
    args = parser.parse_args()
    return build(
        args.suffix, args.history_start, args.rolling_start,
        args.levels.split(","),
    )


if __name__ == "__main__":
    raise SystemExit(main())

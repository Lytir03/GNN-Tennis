"""Recency-weighted per-player match history, shared by both model families.

This code used to live inside ``gbdt_comparison/features.py``, where only the
tabular model could reach it.  That was the source of the information asymmetry
this branch has been closing: the GBDT received twelve recency-weighted history
statistics per player and the GNN received none, so a comparison between them
was partly a comparison of feature engineering effort.

It is shared rather than reimplemented on purpose.  Two implementations of "the
same" statistic drift - a different decay constant, a different treatment of
retirements - and the drift shows up as a model difference.  One definition,
imported by both, makes parity a property of the code instead of a claim in a
README.

The depth ablation is why these matter.  A one-hop GNN scores 0.6103 against the
GBDT's 0.6015: the graph's two-hop information is genuinely worth more than the
gap between the models, but the GNN's one-hop *representation* is worse than
these hand-built aggregates.  Giving the same aggregates to the nodes is meant
to remove that handicap while keeping the structural advantage.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tennis_gnn.edge_features import parse_match_score  # noqa: E402


HISTORY_STAT_NAMES = (
    "history_mass",
    "result_balance",
    "game_margin",
    "set_margin",
    "straight_balance",
    "completed_rate",
)

# Both history views the GBDT receives: all surfaces, and the match surface.
HISTORY_VIEWS = ("all", "surface")

# 6 statistics x 2 views.
HISTORY_FEATURE_DIM = len(HISTORY_STAT_NAMES) * len(HISTORY_VIEWS)


@dataclass(frozen=True)
class HistoricalPerformance:
    date: pd.Timestamp
    surface: str
    result: float
    game_margin: float
    set_margin: float
    straight_balance: float
    completed: float


def aggregate_history(
    records,
    current_date: pd.Timestamp,
    *,
    surface: str | None,
    alpha_days: float,
) -> dict[str, float]:
    """Recency-weighted summary of one player's recent matches.

    Weights decay as ``1 / (1 + age_days / alpha_days)``, so a match a year old
    counts half as much as one played today.  ``surface=None`` uses every
    surface; passing a surface restricts to it.  A player with no qualifying
    history gets zeros, which is the same neutral value an unseen player's node
    features take.
    """

    selected = [
        record
        for record in records
        if surface is None or record.surface == surface
    ]
    if not selected:
        return {name: 0.0 for name in HISTORY_STAT_NAMES}
    ages = np.asarray(
        [(current_date - record.date).days for record in selected],
        dtype=float,
    )
    weights = 1.0 / (1.0 + ages / alpha_days)
    mass = float(weights.sum())

    def weighted(attribute: str) -> float:
        values = np.asarray(
            [getattr(record, attribute) for record in selected],
            dtype=float,
        )
        return float(np.dot(weights, values) / mass)

    return {
        "history_mass": float(np.log1p(mass)),
        "result_balance": weighted("result"),
        "game_margin": weighted("game_margin"),
        "set_margin": weighted("set_margin"),
        "straight_balance": weighted("straight_balance"),
        "completed_rate": weighted("completed"),
    }


def match_records(match) -> tuple[HistoricalPerformance, HistoricalPerformance]:
    """The winner's and loser's records for one completed match."""

    parsed = parse_match_score(
        match.score, match.best_of, incomplete_margin_policy="zero"
    )
    date = match.tourney_date
    winner = HistoricalPerformance(
        date=date,
        surface=match.surface,
        result=1.0,
        game_margin=parsed.relative_game_diff,
        set_margin=parsed.set_margin_scaled,
        straight_balance=parsed.straight_sets_flag,
        completed=parsed.completed_match_flag,
    )
    loser = HistoricalPerformance(
        date=date,
        surface=match.surface,
        result=-1.0,
        game_margin=-parsed.relative_game_diff,
        set_margin=-parsed.set_margin_scaled,
        straight_balance=-parsed.straight_sets_flag,
        completed=parsed.completed_match_flag,
    )
    return winner, loser


class HistoryTracker:
    """Rolling per-player history with a fixed look-back window.

    Order of operations matters and is the caller's responsibility: **trim, then
    read, then append**.  Appending a block's matches before reading its
    features would leak the outcome of the very match being predicted.
    """

    def __init__(self, *, history_years: int = 3, alpha_days: float = 365.0):
        self.histories: dict[str, deque] = defaultdict(deque)
        self.cutoff_delta = pd.Timedelta(days=365 * history_years)
        self.alpha_days = alpha_days

    def trim(self, players, current_date: pd.Timestamp) -> None:
        cutoff = current_date - self.cutoff_delta
        for player in players:
            history = self.histories[player]
            while history and history[0].date < cutoff:
                history.popleft()

    def stats(
        self, player: str, current_date: pd.Timestamp, *, surface: str | None
    ) -> dict[str, float]:
        return aggregate_history(
            self.histories[player],
            current_date,
            surface=surface,
            alpha_days=self.alpha_days,
        )

    def feature_vector(
        self, player: str, current_date: pd.Timestamp, *, surface: str
    ) -> list[float]:
        """The twelve statistics, ordered ``all`` then ``surface``.

        This ordering defines the node-feature layout and must stay stable, so
        cached snapshots remain readable.
        """

        values: list[float] = []
        for view in HISTORY_VIEWS:
            stats = self.stats(
                player,
                current_date,
                surface=None if view == "all" else surface,
            )
            values.extend(float(stats[name]) for name in HISTORY_STAT_NAMES)
        return values

    def update(self, block_matches) -> None:
        """Append every match in a block, after its targets have been read."""

        for match in block_matches.itertuples(index=False):
            winner, loser = match_records(match)
            self.histories[match.winner_name].append(winner)
            self.histories[match.loser_name].append(loser)


def history_feature_names() -> list[str]:
    return [
        f"history_{view}_{name}"
        for view in HISTORY_VIEWS
        for name in HISTORY_STAT_NAMES
    ]

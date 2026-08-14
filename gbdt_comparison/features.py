"""Build leakage-safe tabular features from the same information as the GNN."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    from tennis_gnn.edge_features import parse_match_score
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tennis_gnn.edge_features import parse_match_score


ROUND_ORDER = {
    "R128": 1,
    "R64": 2,
    "R32": 3,
    "R16": 4,
    "QF": 5,
    "SF": 6,
    "F": 7,
}
SCORE_COLUMNS = (
    "bscore_general",
    "bscore_hard",
    "bscore_clay",
    "bscore_grass",
)
HISTORY_STAT_NAMES = (
    "history_mass",
    "result_balance",
    "game_margin",
    "set_margin",
    "straight_balance",
    "completed_rate",
)


@dataclass(frozen=True)
class HistoricalPerformance:
    date: pd.Timestamp
    surface: str
    result: float
    game_margin: float
    set_margin: float
    straight_balance: float
    completed: float


class BScoreSnapshots:
    def __init__(self, processed_directory: Path):
        general = pd.read_csv(processed_directory / "bscore_snapshots.csv")
        merged = general.copy()
        for surface in ("hard", "clay", "grass"):
            frame = pd.read_csv(
                processed_directory / f"bscore_snapshots_{surface}.csv"
            )
            merged = merged.merge(
                frame,
                on=["tourney_id", "round_order", "player"],
                how="left",
            )
        for column in SCORE_COLUMNS[1:]:
            merged[column] = merged[column].fillna(merged["bscore_general"])
        merged["tourney_id"] = merged["tourney_id"].astype(str)
        self.index: dict[tuple[str, int], dict[str, dict[str, float]]] = {}
        for (tourney_id, round_order), group in merged.groupby(
            ["tourney_id", "round_order"], sort=False
        ):
            self.index[(str(tourney_id), int(round_order))] = (
                group.set_index("player")[list(SCORE_COLUMNS)]
                .astype(float)
                .to_dict("index")
            )

    def get(
        self, tourney_id: Any, round_order: Any
    ) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
        snapshot = self.index.get(
            (str(tourney_id), int(round_order)), {}
        )
        defaults = {}
        for column in SCORE_COLUMNS:
            values = [
                float(scores[column])
                for scores in snapshot.values()
                if column in scores and np.isfinite(scores[column])
            ]
            defaults[column] = (
                float(np.quantile(values, 0.25)) if values else 0.0
            )
        return snapshot, defaults


def surface_bscore(scores: dict[str, float], surface: str) -> float:
    column = {
        "Hard": "bscore_hard",
        "Clay": "bscore_clay",
        "Grass": "bscore_grass",
    }.get(surface, "bscore_general")
    return float(scores.get(column, scores.get("bscore_general", 0.0)))


def surface_one_hot(surface: str) -> dict[str, float]:
    return {
        "surface_hard": float(surface == "Hard"),
        "surface_clay": float(surface == "Clay"),
        "surface_grass": float(surface == "Grass"),
    }


def build_static_player_table(matches: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    winner = matches[
        ["winner_name", "winner_ht", "winner_hand", "tourney_date"]
    ].rename(
        columns={
            "winner_name": "player",
            "winner_ht": "height",
            "winner_hand": "hand",
        }
    )
    loser = matches[
        ["loser_name", "loser_ht", "loser_hand", "tourney_date"]
    ].rename(
        columns={
            "loser_name": "player",
            "loser_ht": "height",
            "loser_hand": "hand",
        }
    )
    table = (
        pd.concat([winner, loser], ignore_index=True)
        .sort_values("tourney_date")
        .groupby("player")
        .last()
    )
    median_height = float(table["height"].dropna().median())
    table["height"] = table["height"].fillna(median_height)
    table["hand"] = table["hand"].fillna("R")
    return table, median_height


def aggregate_history(
    records: deque[HistoricalPerformance],
    current_date: pd.Timestamp,
    *,
    surface: str | None,
    alpha_days: float,
) -> dict[str, float]:
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


def add_pair_features(
    row: dict[str, Any],
    prefix: str,
    player_a: dict[str, float],
    player_b: dict[str, float],
) -> None:
    for name in player_a:
        a_value = float(player_a[name])
        b_value = float(player_b[name])
        row[f"{prefix}_{name}_a"] = a_value
        row[f"{prefix}_{name}_b"] = b_value
        row[f"{prefix}_{name}_diff"] = a_value - b_value
        row[f"{prefix}_{name}_abs_diff"] = abs(a_value - b_value)


def phase_for_year(year: int) -> str:
    if year <= 2015:
        return "train"
    if year <= 2016:
        return "val"
    return "test"


def build_feature_dataset(
    project_root: str | Path,
    *,
    tournament_scope: str = "slams_masters",
    seed: int = 42,
    start_date: str = "2011-01-01",
    end_date: str = "2021-01-01",
    history_years: int = 3,
    alpha_days: float = 365.0,
) -> pd.DataFrame:
    """Create one pre-match row per match with block-level leakage protection."""

    if tournament_scope not in {"slams", "slams_masters"}:
        raise ValueError("tournament_scope must be 'slams' or 'slams_masters'")
    root = Path(project_root)
    processed = root / "data" / "processed"
    filename = (
        "atp_matches_slams.csv"
        if tournament_scope == "slams"
        else "atp_matches_slams_masters.csv"
    )
    matches = pd.read_csv(processed / filename, low_memory=False)
    matches["tourney_id"] = matches["tourney_id"].astype(str)
    matches["tourney_date"] = pd.to_datetime(matches["tourney_date"])
    matches["round_order"] = matches["round"].map(ROUND_ORDER)
    matches = matches.dropna(
        subset=["winner_name", "loser_name", "round_order"]
    ).copy()
    matches["round_order"] = matches["round_order"].astype(int)
    matches = matches.sort_values(
        ["tourney_date", "tourney_id", "round_order"],
        kind="stable",
    ).reset_index(drop=True)

    snapshots = BScoreSnapshots(processed)
    static_players, median_height = build_static_player_table(matches)
    histories: dict[str, deque[HistoricalPerformance]] = defaultdict(deque)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    cutoff_delta = pd.Timedelta(days=365 * history_years)

    blocks = matches.groupby(
        ["tourney_date", "tourney_id", "round_order"],
        sort=False,
        dropna=False,
    )
    block_idx = -1
    for (date, tourney_id, round_order), block in blocks:
        date = pd.Timestamp(date)
        # History before 2011 is used for features but never emitted as target.
        emit = pd.Timestamp(start_date) <= date < pd.Timestamp(end_date)
        if emit:
            block_idx += 1
        history_cutoff = date - cutoff_delta
        players = pd.unique(
            pd.concat(
                [block["winner_name"], block["loser_name"]],
                ignore_index=True,
            )
        )
        for player in players:
            history = histories[player]
            while history and history[0].date < history_cutoff:
                history.popleft()

        snapshot, defaults = snapshots.get(tourney_id, round_order)
        if emit:
            for row_in_block, match in enumerate(block.itertuples(index=False)):
                winner = match.winner_name
                loser = match.loser_name
                a_is_winner = bool(rng.random() < 0.5)
                player_a = winner if a_is_winner else loser
                player_b = loser if a_is_winner else winner
                y = float(a_is_winner)
                surface = match.surface

                output: dict[str, Any] = {
                    "phase": phase_for_year(date.year),
                    "year": date.year,
                    "tourney_date": date,
                    "tourney_id": tourney_id,
                    "round_order": round_order,
                    "block_idx": block_idx,
                    "row_in_block": row_in_block,
                    "player_a": player_a,
                    "player_b": player_b,
                    "y_true": y,
                    "round_scaled": float(round_order) / 7.0,
                    "best_of_5_flag": float(match.best_of == 5),
                    "grand_slam_flag": float(match.tourney_level == "G"),
                }
                output.update(surface_one_hot(surface))

                a_scores = snapshot.get(player_a, defaults)
                b_scores = snapshot.get(player_b, defaults)
                a_static = (
                    static_players.loc[player_a]
                    if player_a in static_players.index
                    else {"height": median_height, "hand": "R"}
                )
                b_static = (
                    static_players.loc[player_b]
                    if player_b in static_players.index
                    else {"height": median_height, "hand": "R"}
                )
                node_a = {
                    "bscore_general": float(
                        a_scores.get("bscore_general", defaults["bscore_general"])
                    ),
                    "bscore_surface": surface_bscore(a_scores, surface),
                    "height": float(a_static["height"]),
                    "is_right": float(a_static["hand"] == "R"),
                }
                node_b = {
                    "bscore_general": float(
                        b_scores.get("bscore_general", defaults["bscore_general"])
                    ),
                    "bscore_surface": surface_bscore(b_scores, surface),
                    "height": float(b_static["height"]),
                    "is_right": float(b_static["hand"] == "R"),
                }
                add_pair_features(output, "node", node_a, node_b)

                for history_name, history_surface in (
                    ("all", None),
                    ("surface", surface),
                ):
                    history_a = aggregate_history(
                        histories[player_a],
                        date,
                        surface=history_surface,
                        alpha_days=alpha_days,
                    )
                    history_b = aggregate_history(
                        histories[player_b],
                        date,
                        surface=history_surface,
                        alpha_days=alpha_days,
                    )
                    add_pair_features(
                        output, f"history_{history_name}", history_a, history_b
                    )
                rows.append(output)

        # Update only after every target in the round has been generated.
        for match in block.itertuples(index=False):
            parsed = parse_match_score(
                match.score,
                match.best_of,
                incomplete_margin_policy="zero",
            )
            winner_record = HistoricalPerformance(
                date=date,
                surface=match.surface,
                result=1.0,
                game_margin=parsed.relative_game_diff,
                set_margin=parsed.set_margin_scaled,
                straight_balance=parsed.straight_sets_flag,
                completed=parsed.completed_match_flag,
            )
            loser_record = HistoricalPerformance(
                date=date,
                surface=match.surface,
                result=-1.0,
                game_margin=-parsed.relative_game_diff,
                set_margin=-parsed.set_margin_scaled,
                straight_balance=-parsed.straight_sets_flag,
                completed=parsed.completed_match_flag,
            )
            histories[match.winner_name].append(winner_record)
            histories[match.loser_name].append(loser_record)

    frame = pd.DataFrame(rows)
    numeric = frame.select_dtypes(include=[np.number]).columns
    frame[numeric] = frame[numeric].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return frame


NON_FEATURE_COLUMNS = {
    "phase",
    "year",
    "tourney_date",
    "tourney_id",
    "block_idx",
    "row_in_block",
    "player_a",
    "player_b",
    "y_true",
}


def model_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column not in NON_FEATURE_COLUMNS
        and pd.api.types.is_numeric_dtype(frame[column])
    ]

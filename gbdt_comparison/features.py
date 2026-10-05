# Builds leakage-safe tabular features from the same information the GNN gets.

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

# Imported rather than defined here so the GNN's node features and the GBDT's
# history features are the same computation, not two that happen to agree today.
from tennis_gnn.history import (  # noqa: E402
    HISTORY_STAT_NAMES,
    HistoricalPerformance,
    aggregate_history,
)
# The split lives in one place.  Redefining the phase boundaries here is exactly
# the drift that made the temporal experiment incomparable.
from tennis_gnn.data import SCOPE_FILES  # noqa: E402
from tennis_gnn.data import phase_for_year as _phase_for_year  # noqa: E402


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


class BScoreSnapshots:
    def __init__(self, processed_directory: Path, suffix: str = ""):
        # The suffix selects the scope's snapshots.  Without it a full-scope run
        # would silently read the Slam+Masters B-scores and score most players
        # from the percentile default - a failure that produces plausible
        # numbers and no error.
        general = pd.read_csv(
            processed_directory / f"bscore_snapshots{suffix}.csv"
        )
        merged = general.copy()
        for surface in ("hard", "clay", "grass"):
            frame = pd.read_csv(
                processed_directory
                / f"bscore_snapshots_{surface}{suffix}.csv"
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


def phase_for_year(year: int, split=None) -> str:
    # Phase boundaries, delegated to tennis_gnn.data.
    #
    # The GNN warm-up years never get emitted as GBDT rows, so "warmup"
    # collapses into "train" here - the same rows either model would call
    # trainable.
    split = split or SCOPE_FILES["slams_masters"][2]
    phase = _phase_for_year(year, split)
    return "train" if phase == "warmup" else phase


def build_feature_dataset(
    project_root: str | Path,
    *,
    tournament_scope: str = "slams_masters",
    seed: int = 42,
    start_date: str | None = None,
    end_date: str | None = None,
    history_years: int = 3,
    alpha_days: float = 365.0,
) -> pd.DataFrame:
    # Creates one pre-match row per match with block-level leakage protection.
    if tournament_scope not in SCOPE_FILES:
        raise ValueError(f"scope must be one of {tuple(SCOPE_FILES)}")
    scope_filename, scope_suffix, split = SCOPE_FILES[tournament_scope]
    if end_date is None:
        end_date = split.rolling_end
    if start_date is None:
        # Must follow the scope, exactly as end_date does.  This used to be
        # hardcoded to "2011-01-01", which on a scope whose rolling period
        # starts earlier silently dropped every training year before 2011 and
        # desynchronised the orientation draw from the GNN's - the two models
        # then had different labels for the same match and could not be
        # compared at all.
        start_date = split.rolling_start
    root = Path(project_root)
    processed = root / "data" / "processed"
    filename = (
        "atp_matches_slams.csv"
        if tournament_scope == "slams"
        else scope_filename
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

    snapshots = BScoreSnapshots(processed, scope_suffix)
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
                    "phase": phase_for_year(date.year, split),
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

"""Loading and temporal splitting of the match data.

This replaces the first dozen cells that were duplicated across every notebook
in ``NNs/``, ``NN_test/`` and ``Intransitivity_NNs/``.  The split boundaries
and the target-orientation draw are reproduced exactly, so predictions written
by this package remain key-compatible with the frozen GBDT artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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

START_DATE = "2006-01-01"
ROLLING_START = "2011-01-01"
ROLLING_END = "2021-01-01"
WARMUP_END = 2011
TRAIN_END = 2015
VAL_END = 2016


def phase_for_year(year: int) -> str:
    if year <= 2010:
        return "warmup"
    if year <= TRAIN_END:
        return "train"
    if year <= VAL_END:
        return "val"
    return "test"


def surface_one_hot(surface: Any) -> list[float]:
    return {
        "Hard": [1.0, 0.0, 0.0],
        "Clay": [0.0, 1.0, 0.0],
        "Grass": [0.0, 0.0, 1.0],
    }.get(surface, [0.0, 0.0, 0.0])


def surface_bscore(scores: dict, surface: Any) -> float:
    key = {
        "Hard": "bscore_hard",
        "Clay": "bscore_clay",
        "Grass": "bscore_grass",
    }.get(surface, "bscore_general")
    return float(scores.get(key, scores.get("bscore_general", 0.0)))


@dataclass
class Dataset:
    """Everything the trainer needs, loaded once."""

    matches: pd.DataFrame
    rolling_blocks: pd.DataFrame
    future_matches: pd.DataFrame
    bscore_index: dict
    player_static: pd.DataFrame
    median_height: float
    scope: str

    def block_matches(self, tourney_id: str, round_order: int) -> pd.DataFrame:
        return self.future_matches[
            (self.future_matches["tourney_id"] == tourney_id)
            & (self.future_matches["round_order"] == round_order)
        ]

    def bscore_snapshot(
        self, tourney_id: str, round_order: int
    ) -> tuple[dict, dict]:
        snapshot = self.bscore_index.get((tourney_id, round_order), {})
        defaults = {}
        for column in SCORE_COLUMNS:
            values = [
                float(entry[column])
                for entry in snapshot.values()
                if entry.get(column) is not None
                and not pd.isna(entry[column])
            ]
            defaults[column] = float(np.median(values)) if values else 0.0
        return snapshot, defaults


def _load_bscore_snapshots(processed: Path) -> dict:
    general = pd.read_csv(processed / "bscore_snapshots.csv")
    merged = general
    for surface in ("hard", "clay", "grass"):
        path = processed / f"bscore_snapshots_{surface}.csv"
        if path.is_file():
            merged = merged.merge(
                pd.read_csv(path),
                on=["tourney_id", "round_order", "player"],
                how="left",
            )
        else:
            merged[f"bscore_{surface}"] = np.nan
    for surface in ("hard", "clay", "grass"):
        column = f"bscore_{surface}"
        merged[column] = merged[column].fillna(merged["bscore_general"])

    index = {}
    for key, group in merged.groupby(["tourney_id", "round_order"]):
        index[key] = group.set_index("player")[
            list(SCORE_COLUMNS)
        ].to_dict("index")
    return index


def _build_player_static(matches: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    winners = matches[
        ["winner_name", "winner_ht", "winner_hand", "tourney_date"]
    ].rename(
        columns={
            "winner_name": "player",
            "winner_ht": "height",
            "winner_hand": "hand",
        }
    )
    losers = matches[
        ["loser_name", "loser_ht", "loser_hand", "tourney_date"]
    ].rename(
        columns={
            "loser_name": "player",
            "loser_ht": "height",
            "loser_hand": "hand",
        }
    )
    static = pd.concat([winners, losers], ignore_index=True)
    static = static.sort_values("tourney_date").groupby("player").last()
    median_height = float(static["height"].dropna().median())
    static["height"] = static["height"].fillna(median_height)
    static["hand"] = static["hand"].fillna("R")
    return static, median_height


def load_dataset(project_root: Path, *, scope: str = "slams_masters") -> Dataset:
    if scope not in {"slams", "slams_masters"}:
        raise ValueError("scope must be 'slams' or 'slams_masters'")

    processed = project_root / "data" / "processed"
    filename = (
        "atp_matches_slams.csv"
        if scope == "slams"
        else "atp_matches_slams_masters.csv"
    )
    matches = pd.read_csv(processed / filename)
    matches["tourney_date"] = pd.to_datetime(matches["tourney_date"])
    matches["round_order"] = matches["round"].map(ROUND_ORDER)
    matches = matches.dropna(
        subset=["winner_name", "loser_name", "round_order"]
    ).copy()
    matches["round_order"] = matches["round_order"].astype(int)
    matches = matches.sort_values(
        ["tourney_date", "tourney_id", "round_order"], kind="stable"
    ).reset_index(drop=True)

    intransitivity_path = (
        processed / "match_intransitivity_scores_with_levels.csv"
    )
    if intransitivity_path.is_file():
        intransitivity = pd.read_csv(intransitivity_path)
        intransitivity["tourney_date"] = pd.to_datetime(
            intransitivity["tourney_date"]
        )
        keys = [
            "tourney_id",
            "tourney_date",
            "round",
            "surface",
            "winner_name",
            "loser_name",
        ]
        matches = matches.merge(
            intransitivity[keys + ["intransitivity_level"]],
            on=keys,
            how="left",
        )
    else:
        matches["intransitivity_level"] = "missing"
    matches["intransitivity_level"] = matches["intransitivity_level"].fillna(
        "missing"
    )

    future = matches[
        (matches["tourney_date"] >= ROLLING_START)
        & (matches["tourney_date"] < ROLLING_END)
    ].copy()

    blocks = (
        future[["tourney_date", "tourney_id", "round_order"]]
        .drop_duplicates()
        .sort_values(
            ["tourney_date", "tourney_id", "round_order"], kind="stable"
        )
        .reset_index(drop=True)
    )
    blocks["year"] = blocks["tourney_date"].dt.year
    blocks["phase"] = blocks["year"].apply(phase_for_year)

    player_static, median_height = _build_player_static(matches)

    return Dataset(
        matches=matches,
        rolling_blocks=blocks,
        future_matches=future,
        bscore_index=_load_bscore_snapshots(processed),
        player_static=player_static,
        median_height=median_height,
        scope=scope,
    )

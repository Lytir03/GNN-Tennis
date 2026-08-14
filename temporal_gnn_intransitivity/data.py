"""Build a chronological, block-safe tennis event stream."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from edge_feature_ablation.edge_features import parse_match_score
from gbdt_comparison.features import ROUND_ORDER


INTRA_SOURCE_COLUMNS = (
    "norm_transitive",
    "norm_cyclic",
    "num_common_opponents",
    "num_local_edges",
)


def phase_for_date(date: pd.Timestamp) -> str:
    if date.year < 2011:
        return "warmup"
    if date.year <= 2015:
        return "train"
    if date.year == 2016:
        return "val"
    return "test"


def _merge_intransitivity(matches: pd.DataFrame, processed: Path) -> pd.DataFrame:
    intra = pd.read_csv(
        processed / "match_intransitivity_scores_with_levels.csv",
        low_memory=False,
    )
    intra["tourney_id"] = intra["tourney_id"].astype(str)
    intra["tourney_date"] = pd.to_datetime(intra["tourney_date"])
    keys = [
        "tourney_id",
        "tourney_date",
        "round",
        "winner_name",
        "loser_name",
    ]
    selected = intra[keys + list(INTRA_SOURCE_COLUMNS)].drop_duplicates(keys)
    merged = matches.merge(selected, on=keys, how="left", validate="many_to_one")
    merged[list(INTRA_SOURCE_COLUMNS)] = merged[
        list(INTRA_SOURCE_COLUMNS)
    ].fillna(0.0)
    return merged


def cyclic_energy_share(
    norm_cyclic: pd.Series | np.ndarray,
    norm_transitive: pd.Series | np.ndarray,
) -> np.ndarray:
    """Fraction of local Hodge energy unexplained by a transitive ranking."""

    cyclic_sq = np.square(np.asarray(norm_cyclic, dtype=float))
    transitive_sq = np.square(np.asarray(norm_transitive, dtype=float))
    total = cyclic_sq + transitive_sq
    return np.divide(
        cyclic_sq,
        total,
        out=np.zeros_like(total, dtype=float),
        where=total > 1e-12,
    )


def build_event_stream(
    project_root: str | Path,
    *,
    scope: str = "slams_masters",
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, int], dict[str, float]]:
    """Return one row per match, scored and updated only at round boundaries."""

    if scope not in {"slams", "slams_masters"}:
        raise ValueError("scope must be 'slams' or 'slams_masters'")
    root = Path(project_root)
    processed = root / "data" / "processed"
    filename = (
        "atp_matches_slams.csv"
        if scope == "slams"
        else "atp_matches_slams_masters.csv"
    )
    matches = pd.read_csv(processed / filename, low_memory=False)
    matches["tourney_id"] = matches["tourney_id"].astype(str)
    matches["tourney_date"] = pd.to_datetime(matches["tourney_date"])
    matches["round_order"] = matches["round"].map(ROUND_ORDER)
    matches = matches.dropna(
        subset=["winner_name", "loser_name", "round_order"]
    ).copy()
    matches = matches[
        (matches["tourney_date"] < pd.Timestamp("2021-01-01"))
    ].copy()
    matches["round_order"] = matches["round_order"].astype(int)
    matches = matches.sort_values(
        ["tourney_date", "tourney_id", "round_order"],
        kind="stable",
    ).reset_index(drop=True)
    matches = _merge_intransitivity(matches, processed)
    matches["cyclic_energy_share"] = cyclic_energy_share(
        matches["norm_cyclic"], matches["norm_transitive"]
    )
    matches["has_context_evidence"] = (
        (matches["num_common_opponents"] >= 2)
        & (matches["num_local_edges"] >= 3)
        & (
            np.square(matches["norm_cyclic"])
            + np.square(matches["norm_transitive"])
            > 1e-12
        )
    )

    players = sorted(
        set(matches["winner_name"]).union(matches["loser_name"])
    )
    player_to_idx = {player: idx for idx, player in enumerate(players)}

    # Levels are evaluation strata, with thresholds learned on training only.
    train_scores = matches.loc[
        matches["tourney_date"].dt.year.between(2011, 2015)
        & matches["has_context_evidence"],
        "cyclic_energy_share",
    ]
    low_threshold = float(train_scores.quantile(1 / 3))
    high_threshold = float(train_scores.quantile(2 / 3))

    train_mask = matches["tourney_date"].dt.year.between(2011, 2015)
    transformed = pd.DataFrame(
        {
            "cyclic_energy_share": matches["cyclic_energy_share"].astype(float),
            "num_common_opponents": np.log1p(
                matches["num_common_opponents"].astype(float)
            ),
            "num_local_edges": np.log1p(
                matches["num_local_edges"].astype(float)
            ),
        }
    )
    transformed_train = transformed.loc[train_mask]
    means = transformed_train.mean()
    stds = transformed_train.std(ddof=0).replace(0, 1.0)

    rng = np.random.default_rng(seed)
    rows = []
    last_date: dict[str, pd.Timestamp] = {}
    blocks = matches.groupby(
        ["tourney_date", "tourney_id", "round_order"],
        sort=False,
        dropna=False,
    )
    for block_idx, ((date, tourney_id, round_order), block) in enumerate(blocks):
        date = pd.Timestamp(date)
        pending_last_dates: dict[str, pd.Timestamp] = {}
        for row_in_block, match in enumerate(block.itertuples(index=False)):
            parsed = parse_match_score(
                match.score,
                match.best_of,
                incomplete_margin_policy="zero",
            )
            winner = match.winner_name
            loser = match.loser_name
            a_is_winner = bool(rng.random() < 0.5)
            player_a = winner if a_is_winner else loser
            player_b = loser if a_is_winner else winner

            score = float(match.cyclic_energy_share)
            if not bool(match.has_context_evidence):
                level = "no_context_evidence"
            elif score <= low_threshold:
                level = "low"
            elif score <= high_threshold:
                level = "medium"
            else:
                level = "high"

            raw_intra = {
                "cyclic_energy_share": score,
                "num_common_opponents": np.log1p(
                    float(match.num_common_opponents)
                ),
                "num_local_edges": np.log1p(float(match.num_local_edges)),
            }
            normalized_intra = {
                column: (raw_intra[column] - means[column]) / stds[column]
                for column in raw_intra
            }

            def recency(player: str) -> float:
                previous = last_date.get(player)
                if previous is None:
                    return 0.0
                days = max((date - previous).days, 0)
                return float(1.0 / (1.0 + days / 365.0))

            rows.append(
                {
                    "phase": phase_for_date(date),
                    "year": date.year,
                    "tourney_date": date,
                    "tourney_id": tourney_id,
                    "round": match.round,
                    "round_order": int(round_order),
                    "block_idx": block_idx,
                    "row_in_block": row_in_block,
                    "winner_name": winner,
                    "loser_name": loser,
                    "winner_idx": player_to_idx[winner],
                    "loser_idx": player_to_idx[loser],
                    "player_a": player_a,
                    "player_b": player_b,
                    "player_a_idx": player_to_idx[player_a],
                    "player_b_idx": player_to_idx[player_b],
                    "y_true": float(a_is_winner),
                    "surface_hard": float(match.surface == "Hard"),
                    "surface_clay": float(match.surface == "Clay"),
                    "surface_grass": float(match.surface == "Grass"),
                    "round_scaled": float(round_order) / 7.0,
                    "best_of_5_flag": float(match.best_of == 5),
                    "grand_slam_flag": float(match.tourney_level == "G"),
                    "relative_game_diff": parsed.relative_game_diff,
                    "set_margin_scaled": parsed.set_margin_scaled,
                    "straight_sets_flag": parsed.straight_sets_flag,
                    "completed_match_flag": parsed.completed_match_flag,
                    "winner_recency": recency(winner),
                    "loser_recency": recency(loser),
                    "intransitivity_score": score,
                    "intransitivity_level": level,
                    "intra_score_scaled": normalized_intra[
                        "cyclic_energy_share"
                    ],
                    "intra_common_scaled": normalized_intra[
                        "num_common_opponents"
                    ],
                    "intra_edges_scaled": normalized_intra["num_local_edges"],
                }
            )
            pending_last_dates[winner] = date
            pending_last_dates[loser] = date
        # No match in the same round can affect another match in that round.
        last_date.update(pending_last_dates)

    metadata = {
        "definition": (
            "norm_cyclic^2 / "
            "(norm_cyclic^2 + norm_transitive^2)"
        ),
        "minimum_common_opponents": 2,
        "minimum_local_edges": 3,
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
    }
    return pd.DataFrame(rows), player_to_idx, metadata


BASE_CONTEXT_COLUMNS = (
    "surface_hard",
    "surface_clay",
    "surface_grass",
    "round_scaled",
    "best_of_5_flag",
    "grand_slam_flag",
)
INTRA_CONTEXT_COLUMNS = (
    "intra_score_scaled",
    "intra_common_scaled",
    "intra_edges_scaled",
)
EDGE_COLUMNS = (
    "relative_game_diff",
    "set_margin_scaled",
    "straight_sets_flag",
    "completed_match_flag",
    "surface_hard",
    "surface_clay",
    "surface_grass",
    "round_scaled",
)

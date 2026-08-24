# Score-derived, directional edge features for tennis match graphs.
#
# The score in the Sackmann ATP data is written from the recorded winner's
# perspective. So a parsed game or set margin is positive on the
# loser -> winner edge and flipped for winner -> loser.

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import math
import re
from typing import Any, Iterable, Mapping, Sequence


_SET_RE = re.compile(r"^(?P<w>\d+)-(?P<l>\d+)(?:\([^)]*\))?$")
_RETIREMENT_RE = re.compile(r"(?:^|\s)(?:RET|RETD|RETIRED)(?:\.|\s|$)", re.I)
_WALKOVER_RE = re.compile(r"(?:^|\s)(?:W/O|WO|WALKOVER)(?:\.|\s|$)", re.I)
_OTHER_INCOMPLETE_RE = re.compile(
    r"(?:^|\s)(?:DEF|DEFAULT|ABD|ABN|ABANDONED)(?:\.|\s|$)", re.I
)


@dataclass(frozen=True)
class MatchScoreFeatures:
    # Features parsed once from a winner-perspective score.
    relative_game_diff: float
    set_margin_scaled: float
    straight_sets_flag: float
    completed_match_flag: float
    retirement_flag: float
    walkover_flag: float
    parsed_set_count: int


@dataclass(frozen=True)
class EdgeFeatureConfig:
    # Picks an edge-feature ablation without touching the rest of the run.
    signed_game_margin: bool = True
    include_set_margin: bool = False
    include_straight_sets: bool = False
    include_match_status: bool = False
    detailed_status_flags: bool = False
    direction_encoding: str = "signed"
    incomplete_margin_policy: str = "zero"
    exclude_unparseable_incomplete: bool = False
    include_tournament_context: bool = False

    def __post_init__(self) -> None:
        if self.direction_encoding not in {"signed", "legacy_01"}:
            raise ValueError("direction_encoding must be 'signed' or 'legacy_01'")
        if self.incomplete_margin_policy not in {"zero", "played"}:
            raise ValueError("incomplete_margin_policy must be 'zero' or 'played'")


# The first preset reproduces the pre-change representation described in the
# project: equal positive margins and a 0/1 reverse-edge indicator.  Every
# subsequent preset retains the same split, seed and training code.
ABLATION_PRESETS: Mapping[str, EdgeFeatureConfig] = {
    "current": EdgeFeatureConfig(
        signed_game_margin=False,
        direction_encoding="legacy_01",
        incomplete_margin_policy="played",
    ),
    "signed_game": EdgeFeatureConfig(incomplete_margin_policy="played"),
    "set_margin": EdgeFeatureConfig(
        include_set_margin=True,
        incomplete_margin_policy="played",
    ),
    "straight_sets": EdgeFeatureConfig(
        include_set_margin=True,
        include_straight_sets=True,
        incomplete_margin_policy="played",
    ),
    "match_status": EdgeFeatureConfig(
        include_set_margin=True,
        include_straight_sets=True,
        include_match_status=True,
        detailed_status_flags=True,
    ),
    "full": EdgeFeatureConfig(
        include_set_margin=True,
        include_straight_sets=True,
        include_match_status=True,
    ),
}


def _is_missing(value: Any) -> bool:
    return value is None or (
        isinstance(value, float) and math.isnan(value)
    )


def _is_completed_set(winner_games: int, loser_games: int) -> bool:
    # Recognises normal, advantage, and match-tiebreak set scores.
    high = max(winner_games, loser_games)
    low = min(winner_games, loser_games)
    if high == 7 and low in {5, 6}:
        return True
    if high >= 6 and high - low >= 2:
        return True
    return high >= 10 and high - low >= 2


def parse_match_score(
    score: Any,
    best_of: Any,
    *,
    incomplete_margin_policy: str = "zero",
) -> MatchScoreFeatures:
    # Parses granular match features without leaning on aggregate player
    # ratings.
    #
    # Incomplete matches are spotted from their score marker. With the
    # default "zero" policy their game/set margins get zeroed and
    # straight-sets is false. "played" keeps the margins from the games and
    # completed sets that actually happened before the interruption.
    if incomplete_margin_policy not in {"zero", "played"}:
        raise ValueError("incomplete_margin_policy must be 'zero' or 'played'")

    # Snapshot building re-reads three years of history for every tournament
    # round, so the same few thousand distinct score strings are parsed
    # millions of times per run.  Normalising to text makes the arguments
    # hashable and the result cacheable; the parse itself is unchanged.
    return _parse_match_score_cached(
        "" if _is_missing(score) else str(score).strip(),
        str(best_of),
        incomplete_margin_policy,
    )


@lru_cache(maxsize=None)
def _parse_match_score_cached(
    score_text: str,
    best_of: str,
    incomplete_margin_policy: str,
) -> MatchScoreFeatures:
    retirement = bool(_RETIREMENT_RE.search(score_text))
    walkover = bool(_WALKOVER_RE.search(score_text))
    other_incomplete = bool(_OTHER_INCOMPLETE_RE.search(score_text))
    completed = bool(score_text) and not (
        retirement or walkover or other_incomplete
    )

    parsed_sets: list[tuple[int, int]] = []
    for token in score_text.split():
        match = _SET_RE.fullmatch(token)
        if match:
            parsed_sets.append((int(match["w"]), int(match["l"])))

    winner_games = sum(w for w, _ in parsed_sets)
    loser_games = sum(l for _, l in parsed_sets)
    total_games = winner_games + loser_games
    relative_game_diff = (
        (winner_games - loser_games) / total_games if total_games else 0.0
    )

    completed_sets = [
        (w, l) for w, l in parsed_sets if _is_completed_set(w, l)
    ]
    winner_sets = sum(w > l for w, l in completed_sets)
    loser_sets = sum(l > w for w, l in completed_sets)

    try:
        max_sets = int(float(best_of))
    except (TypeError, ValueError):
        max_sets = 0
    if max_sets not in {3, 5}:
        max_sets = max(3, len(completed_sets))

    set_margin_scaled = (winner_sets - loser_sets) / max_sets
    straight_sets = bool(
        completed
        and completed_sets
        and loser_sets == 0
        and winner_sets == (max_sets // 2 + 1)
    )

    if not completed and incomplete_margin_policy == "zero":
        relative_game_diff = 0.0
        set_margin_scaled = 0.0
        straight_sets = False

    return MatchScoreFeatures(
        relative_game_diff=float(relative_game_diff),
        set_margin_scaled=float(set_margin_scaled),
        straight_sets_flag=float(straight_sets),
        completed_match_flag=float(completed),
        retirement_flag=float(retirement),
        walkover_flag=float(walkover),
        parsed_set_count=len(parsed_sets),
    )


def edge_feature_names(config: EdgeFeatureConfig) -> tuple[str, ...]:
    names = ["recency_weight", "signed_relative_game_diff"]
    if not config.signed_game_margin:
        names[1] = "relative_game_diff"
    if config.include_set_margin:
        names.append("signed_set_margin")
    if config.include_straight_sets:
        names.append("straight_sets_flag")
    if config.include_match_status:
        names.append("completed_match_flag")
        if config.detailed_status_flags:
            names.extend(["retirement_flag", "walkover_flag"])
    if config.include_tournament_context:
        names.extend(["best_of_5_flag", "grand_slam_flag"])
    names.extend(
        [
            "surface_hard",
            "surface_clay",
            "surface_grass",
            "round_scaled",
            "result_direction",
        ]
    )
    return tuple(names)


def _surface_one_hot(surface: Any) -> list[float]:
    return {
        "Hard": [1.0, 0.0, 0.0],
        "Clay": [0.0, 1.0, 0.0],
        "Grass": [0.0, 0.0, 1.0],
    }.get(surface, [0.0, 0.0, 0.0])


def _row_value(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, Mapping):
        return row.get(name, default)
    try:
        return row[name]
    except (KeyError, TypeError):
        return getattr(row, name, default)


def _direction_values(config: EdgeFeatureConfig) -> tuple[float, float]:
    if config.direction_encoding == "legacy_01":
        return 0.0, 1.0
    return 1.0, -1.0


def _features_for_direction(
    *,
    recency_weight: float,
    score_features: MatchScoreFeatures,
    surface: Any,
    round_scaled: float,
    sign: float,
    result_direction: float,
    config: EdgeFeatureConfig,
    best_of_five_flag: float,
    grand_slam_flag: float,
) -> list[float]:
    game_margin = score_features.relative_game_diff
    if config.signed_game_margin:
        game_margin *= sign

    features = [recency_weight, game_margin]
    if config.include_set_margin:
        features.append(sign * score_features.set_margin_scaled)
    if config.include_straight_sets:
        features.append(score_features.straight_sets_flag)
    if config.include_match_status:
        features.append(score_features.completed_match_flag)
        if config.detailed_status_flags:
            features.extend(
                [
                    score_features.retirement_flag,
                    score_features.walkover_flag,
                ]
            )
    if config.include_tournament_context:
        # These are properties of the historical match, available without
        # knowing its outcome.
        features.extend(
            [
                float(best_of_five_flag),
                float(grand_slam_flag),
            ]
        )
    features.extend(_surface_one_hot(surface))
    features.extend([round_scaled, result_direction])
    return [float(value) for value in features]


def build_bidirectional_edges(
    matches: Iterable[Any],
    player_to_idx: Mapping[Any, int],
    snapshot_date: Any,
    config: EdgeFeatureConfig,
    *,
    alpha_days: float = 365.0,
) -> tuple[list[list[int]], list[list[float]], dict[str, int]]:
    # Builds loser->winner and winner->loser edges from historical matches.
    edge_pairs: list[list[int]] = []
    edge_attributes: list[list[float]] = []
    stats = {
        "matches_seen": 0,
        "matches_used": 0,
        "matches_excluded": 0,
        "completed_matches": 0,
        "retirements": 0,
        "walkovers": 0,
    }
    forward_direction, reverse_direction = _direction_values(config)

    for row in matches:
        stats["matches_seen"] += 1
        loser = _row_value(row, "loser_name")
        winner = _row_value(row, "winner_name")
        if loser not in player_to_idx or winner not in player_to_idx:
            stats["matches_excluded"] += 1
            continue

        match_date = _row_value(row, "tourney_date")
        age_days = (snapshot_date - match_date).days
        if age_days < 0:
            stats["matches_excluded"] += 1
            continue

        parsed = parse_match_score(
            _row_value(row, "score"),
            _row_value(row, "best_of"),
            incomplete_margin_policy=config.incomplete_margin_policy,
        )
        stats["completed_matches"] += int(parsed.completed_match_flag)
        stats["retirements"] += int(parsed.retirement_flag)
        stats["walkovers"] += int(parsed.walkover_flag)

        if (
            config.exclude_unparseable_incomplete
            and not parsed.completed_match_flag
            and parsed.parsed_set_count == 0
        ):
            stats["matches_excluded"] += 1
            continue

        round_order = _row_value(row, "round_order")
        try:
            round_scaled = float(round_order) / 7.0
        except (TypeError, ValueError):
            round_scaled = 0.0
        if math.isnan(round_scaled):
            round_scaled = 0.0

        common = {
            "recency_weight": 1.0 / (1.0 + age_days / alpha_days),
            "score_features": parsed,
            "surface": _row_value(row, "surface"),
            "round_scaled": round_scaled,
            "config": config,
            "best_of_five_flag": float(
                str(_row_value(row, "best_of")) in {"5", "5.0"}
            ),
            "grand_slam_flag": float(
                _row_value(row, "tourney_level") == "G"
            ),
        }
        loser_idx = player_to_idx[loser]
        winner_idx = player_to_idx[winner]
        edge_pairs.extend(
            [[loser_idx, winner_idx], [winner_idx, loser_idx]]
        )
        edge_attributes.extend(
            [
                _features_for_direction(
                    **common, sign=1.0, result_direction=forward_direction
                ),
                _features_for_direction(
                    **common, sign=-1.0, result_direction=reverse_direction
                ),
            ]
        )
        stats["matches_used"] += 1

    return edge_pairs, edge_attributes, stats


def with_incomplete_policy(
    config: EdgeFeatureConfig,
    *,
    policy: str,
    exclude_unparseable: bool = False,
) -> EdgeFeatureConfig:
    # Convenience helper for sensitivity tests on incomplete matches.
    return replace(
        config,
        incomplete_margin_policy=policy,
        exclude_unparseable_incomplete=exclude_unparseable,
    )

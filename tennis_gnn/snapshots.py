# Builds the per-block graph snapshots once and caches them.
#
# A snapshot only depends on the data and the edge-feature preset - never
# on the model or the optimiser. The original notebook rebuilt every graph
# inside the training loop, so a hyperparameter search paid the graph
# construction cost once per configuration. Building them once and caching
# turns a search from hours into minutes - the single change that makes
# tuning the GNN affordable at all.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import pandas as pd
import torch

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tennis_gnn.edge_features import (  # noqa: E402
    ABLATION_PRESETS,
    EdgeFeatureConfig,
    build_bidirectional_edges,
    edge_feature_names,
)
from tennis_gnn.data import (  # noqa: E402
    ROLLING_START,
    Dataset,
    surface_one_hot,
)
from tennis_gnn.history import (  # noqa: E402
    HISTORY_FEATURE_DIM,
    HistoryTracker,
)


MAX_HISTORY_DAYS = 365 * 3
ALPHA_DAYS = 365.0

# B-score x4, height, handedness.  Node features up to this width are what the
# model saw before the history features were added; everything after is the
# twelve recency-weighted statistics the GBDT has always received.
LEGACY_NODE_DIM = 6


@dataclass
class BlockSnapshot:
    # One tournament-round: the graph before it, and the matches in it.
    block_idx: int
    tourney_id: str
    round_order: int
    phase: str
    year: int
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    player_a: torch.Tensor
    player_b: torch.Tensor
    context: torch.Tensor
    y: torch.Tensor
    bscore_diff: torch.Tensor
    intransitivity_level: list[str]
    surface: str


def _node_features(
    players,
    snapshot: dict,
    defaults: dict,
    static: dict,
    median_height: float,
    tracker: HistoryTracker,
    snapshot_date,
    surface: str,
) -> torch.Tensor:
    # Per-player node features, always including the history statistics.
    #
    # The history block is stored unconditionally so one cache serves both
    # the with-history and without-history models; ModelConfig.node_history_features
    # decides how much of it the network actually reads. Storing it is also
    # what makes the parity claim checkable: the same twelve numbers the
    # GBDT gets for the two players in a match are attached here to every
    # player in the graph.
    rows = []
    for player in players:
        scores = snapshot.get(player, defaults)
        height, hand = static.get(player, (median_height, "R"))
        if pd.isna(height):
            height = median_height
        rows.append(
            [
                float(scores.get("bscore_general", defaults["bscore_general"])),
                float(scores.get("bscore_hard", defaults["bscore_hard"])),
                float(scores.get("bscore_clay", defaults["bscore_clay"])),
                float(scores.get("bscore_grass", defaults["bscore_grass"])),
                float(height),
                1.0 if hand == "R" else 0.0,
            ]
            + tracker.feature_vector(player, snapshot_date, surface=surface)
        )
    tensor = torch.tensor(rows, dtype=torch.float)
    return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)


@dataclass
class BlockGraph:
    # One tournament-round's graph, before any orientation draw.
    #
    # Everything here depends only on the data and the edge preset, never
    # on the seed - that's what lets one build serve every seed.
    block_idx: int
    tourney_id: str
    round_order: int
    phase: str
    year: int
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    players: list[str]
    surface: str


def build_graphs(
    dataset: Dataset,
    edge_config: EdgeFeatureConfig,
    *,
    verbose: bool = True,
) -> list[BlockGraph]:
    # Builds every block's graph in chronological order, seed-independently.
    #
    # Kept separate from the targets because graph construction is by far
    # the expensive half and doesn't vary with the seed: building five
    # seeds the old way repeated this work five times over. Every block
    # gets emitted, even ones whose target set is empty, so that
    # attach_targets sees exactly the block sequence the original
    # single-pass build saw - the orientation draw is positional, so a
    # missing block would shift every later label.
    edge_dim = len(edge_feature_names(edge_config))

    # A dict lookup instead of a DataFrame .loc per player per block: the same
    # values, but ~240k pandas indexing calls removed from the build.
    static_lookup = {
        player: (row.height, row.hand)
        for player, row in zip(
            dataset.player_static.index,
            dataset.player_static.itertuples(index=False),
        )
    }
    median_height = dataset.median_height

    matches = dataset.matches
    # Strictly before the first rolling block: a block must never see its own
    # matches in the graph it is predicted from.
    history = matches[
        matches["tourney_date"] < pd.Timestamp(ROLLING_START)
    ].copy()

    # Seed the history tracker with the same pre-rolling matches, in date
    # order, so the deques are chronological and the trim is well defined.
    tracker = HistoryTracker(history_years=3, alpha_days=ALPHA_DAYS)
    tracker.update(history.sort_values("tourney_date", kind="stable"))

    graphs: list[BlockGraph] = []
    total = len(dataset.rolling_blocks)

    for block_idx, block in dataset.rolling_blocks.iterrows():
        tourney_id = block["tourney_id"]
        round_order = int(block["round_order"])
        t_block = block["tourney_date"]

        block_matches = dataset.block_matches(tourney_id, round_order)
        snapshot, defaults = dataset.bscore_snapshot(tourney_id, round_order)

        cutoff = t_block - pd.Timedelta(days=MAX_HISTORY_DAYS)
        trimmed = history[history["tourney_date"] >= cutoff]

        players = pd.Index(
            pd.unique(
                pd.concat(
                    [
                        trimmed["winner_name"],
                        trimmed["loser_name"],
                        block_matches["winner_name"],
                        block_matches["loser_name"],
                    ],
                    ignore_index=True,
                )
            )
        )
        player_to_idx = {player: i for i, player in enumerate(players)}

        # Trim before reading and append only after the targets are built:
        # appending first would leak the result of the very match predicted.
        tracker.trim(players, t_block)
        block_surface = str(block_matches["surface"].iloc[0])
        x = _node_features(
            players,
            snapshot,
            defaults,
            static_lookup,
            median_height,
            tracker,
            t_block,
            block_surface,
        )

        edge_pairs, edge_attributes, _ = build_bidirectional_edges(
            matches=trimmed.to_dict("records"),
            player_to_idx=player_to_idx,
            snapshot_date=t_block,
            config=edge_config,
            alpha_days=ALPHA_DAYS,
        )

        if edge_pairs:
            edge_index = (
                torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
            )
            edge_attr = torch.tensor(edge_attributes, dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, edge_dim), dtype=torch.float)
        edge_attr = torch.nan_to_num(
            edge_attr, nan=0.0, posinf=0.0, neginf=0.0
        )

        history = pd.concat([history, block_matches], ignore_index=True)
        tracker.update(block_matches)

        graphs.append(
            BlockGraph(
                block_idx=int(block_idx),
                tourney_id=tourney_id,
                round_order=round_order,
                phase=block["phase"],
                year=int(block["year"]),
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                players=list(players),
                surface=block_surface,
            )
        )

        if verbose and block_idx % 100 == 0:
            print(
                f"  graph {block_idx:5d}/{total} "
                f"nodes={x.shape[0]:4d} edges={edge_index.shape[1]:6d}",
                flush=True,
            )

    return graphs


def attach_targets(
    dataset: Dataset,
    graphs: Sequence[BlockGraph],
    *,
    seed: int = 42,
) -> list[BlockSnapshot]:
    # Draws the match orientations for one seed on top of prebuilt graphs.
    #
    # The draw consumes exactly one rng.random() per emitted match, in
    # block order, reproducing the sequence the GBDT feature builder uses.
    # That's what keeps the two model families comparable match by match,
    # so this has to iterate every graph in order - including blocks that
    # yield no targets.
    rng = np.random.default_rng(seed)
    snapshots: list[BlockSnapshot] = []

    for graph in graphs:
        block_matches = dataset.block_matches(
            graph.tourney_id, graph.round_order
        )
        snapshot, defaults = dataset.bscore_snapshot(
            graph.tourney_id, graph.round_order
        )
        player_to_idx = {player: i for i, player in enumerate(graph.players)}

        targets = _block_targets(
            block_matches, player_to_idx, rng, snapshot, defaults
        )
        if targets is None:
            continue

        snapshots.append(
            BlockSnapshot(
                block_idx=graph.block_idx,
                tourney_id=graph.tourney_id,
                round_order=graph.round_order,
                phase=graph.phase,
                year=graph.year,
                x=graph.x,
                edge_index=graph.edge_index,
                edge_attr=graph.edge_attr,
                surface=graph.surface,
                **targets,
            )
        )

    return snapshots


def _block_targets(block_matches, player_to_idx, rng, snapshot, defaults):
    player_a: list[int] = []
    player_b: list[int] = []
    contexts: list[list[float]] = []
    labels: list[float] = []
    bscore_diffs: list[float] = []
    levels: list[str] = []

    for match in block_matches.itertuples(index=False):
        winner = match.winner_name
        loser = match.loser_name
        if winner not in player_to_idx or loser not in player_to_idx:
            raise RuntimeError(
                "Block player missing from the graph; the orientation draw "
                "would desynchronise from the GBDT feature builder."
            )

        winner_scores = snapshot.get(winner, defaults)
        loser_scores = snapshot.get(loser, defaults)
        winner_bscore = float(
            winner_scores.get("bscore_general", defaults["bscore_general"])
        )
        loser_bscore = float(
            loser_scores.get("bscore_general", defaults["bscore_general"])
        )

        a_is_winner = bool(rng.random() < 0.5)
        if a_is_winner:
            player_a.append(player_to_idx[winner])
            player_b.append(player_to_idx[loser])
            labels.append(1.0)
            bscore_diffs.append(winner_bscore - loser_bscore)
        else:
            player_a.append(player_to_idx[loser])
            player_b.append(player_to_idx[winner])
            labels.append(0.0)
            bscore_diffs.append(loser_bscore - winner_bscore)

        # Full context is always stored; ModelConfig.rich_match_context
        # decides how much of it the model may see, so one cached snapshot
        # serves both variants.  The last two columns close an information
        # asymmetry: the GBDT baseline already received best-of-5 and
        # Grand-Slam flags for the match being predicted, and the GNN did not.
        contexts.append(
            surface_one_hot(match.surface)
            + [
                float(match.round_order) / 7.0,
                float(match.best_of == 5),
                float(match.tourney_level == "G"),
            ]
        )
        levels.append(getattr(match, "intransitivity_level", "missing"))

    if not labels:
        return None

    return {
        "player_a": torch.tensor(player_a, dtype=torch.long),
        "player_b": torch.tensor(player_b, dtype=torch.long),
        "context": torch.tensor(contexts, dtype=torch.float),
        "y": torch.tensor(labels, dtype=torch.float),
        "bscore_diff": torch.tensor(bscore_diffs, dtype=torch.float),
        "intransitivity_level": levels,
    }


def graph_cache_path(project_root: Path, scope: str, preset: str) -> Path:
    # Where the seed-independent graphs live.
    #
    # No seed in the key, because nothing here depends on one. The version
    # tag does belong: v2 graphs carry the twelve history statistics per
    # node, and a v1 file would load without error at the wrong feature
    # width. v3 adds the `surface` field to BlockGraph.
    return (
        project_root
        / ".cache"
        / "snapshots"
        / f"{scope}__{preset}__graphs__v3.pt"
    )


def cache_path(
    project_root: Path, scope: str, preset: str, seed: int
) -> Path:
    # The seed belongs in this key even though it does not belong in the graph
    # key: the target orientation draw depends on it, and omitting it would
    # silently serve one seed's labels to another. v3 adds the `surface`
    # field to BlockSnapshot.
    return (
        project_root
        / ".cache"
        / "snapshots"
        / f"{scope}__{preset}__seed{seed}__v3.pt"
    )


def load_or_build_graphs(
    project_root: Path,
    dataset: Dataset,
    preset: str,
    *,
    rebuild: bool = False,
    verbose: bool = True,
) -> list[BlockGraph]:
    path = graph_cache_path(project_root, dataset.scope, preset)
    if path.is_file() and not rebuild:
        if verbose:
            print(f"Loading cached graphs: {path.name}", flush=True)
        return torch.load(path, weights_only=False)

    if verbose:
        print(f"Building graphs for preset {preset!r}...", flush=True)
    graphs = build_graphs(
        dataset, ABLATION_PRESETS[preset], verbose=verbose
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graphs, path)
    if verbose:
        print(f"Cached {len(graphs)} graphs to {path}", flush=True)
    return graphs


def load_or_build(
    project_root: Path,
    dataset: Dataset,
    preset: str,
    *,
    seed: int = 42,
    rebuild: bool = False,
    verbose: bool = True,
) -> list[BlockSnapshot]:
    path = cache_path(project_root, dataset.scope, preset, seed)
    if path.is_file() and not rebuild:
        if verbose:
            print(f"Loading cached snapshots: {path.name}", flush=True)
        return torch.load(path, weights_only=False)

    graphs = load_or_build_graphs(
        project_root, dataset, preset, rebuild=rebuild, verbose=verbose
    )
    if verbose:
        print(f"Drawing targets for seed {seed}...", flush=True)
    snapshots = attach_targets(dataset, graphs, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(snapshots, path)
    if verbose:
        print(f"Cached {len(snapshots)} snapshots to {path}", flush=True)
    return snapshots

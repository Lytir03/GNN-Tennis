"""Cumulative, non-temporal model ablations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class ModelExperimentConfig:
    edge_preset: str
    direct_bscore_logit: bool = False
    node_bscore_features: bool = True
    antisymmetric_decoder: bool = False
    normalize_node_features: bool = False
    aggregation: str = "sum"
    residual_connections: bool = False
    tournament_context_edges: bool = False


MODEL_EXPERIMENTS: Mapping[str, ModelExperimentConfig] = {
    "current_gnn": ModelExperimentConfig(edge_preset="current"),
    "edge_full": ModelExperimentConfig(edge_preset="full"),
    "bscore_residual": ModelExperimentConfig(
        edge_preset="full",
        direct_bscore_logit=True,
    ),
    "antisymmetric_decoder": ModelExperimentConfig(
        edge_preset="full",
        direct_bscore_logit=True,
        antisymmetric_decoder=True,
    ),
    "antisymmetric_no_direct_bscore": ModelExperimentConfig(
        edge_preset="full",
        direct_bscore_logit=False,
        antisymmetric_decoder=True,
    ),
    "antisymmetric_no_node_bscore": ModelExperimentConfig(
        edge_preset="full",
        direct_bscore_logit=True,
        node_bscore_features=False,
        antisymmetric_decoder=True,
    ),
    "antisymmetric_no_bscore": ModelExperimentConfig(
        edge_preset="full",
        direct_bscore_logit=False,
        node_bscore_features=False,
        antisymmetric_decoder=True,
    ),
    "antisymmetric_straight_sets": ModelExperimentConfig(
        edge_preset="straight_sets",
        direct_bscore_logit=True,
        antisymmetric_decoder=True,
    ),
    "antisymmetric_mean": ModelExperimentConfig(
        edge_preset="full",
        direct_bscore_logit=True,
        antisymmetric_decoder=True,
        aggregation="mean",
    ),
    "antisymmetric_tournament_context": ModelExperimentConfig(
        edge_preset="full",
        direct_bscore_logit=True,
        antisymmetric_decoder=True,
        tournament_context_edges=True,
    ),
    "antisymmetric_mean_tournament": ModelExperimentConfig(
        edge_preset="full",
        direct_bscore_logit=True,
        antisymmetric_decoder=True,
        aggregation="mean",
        tournament_context_edges=True,
    ),
    "node_normalization": ModelExperimentConfig(
        edge_preset="full",
        direct_bscore_logit=True,
        antisymmetric_decoder=True,
        normalize_node_features=True,
    ),
    "mean_aggregation": ModelExperimentConfig(
        edge_preset="full",
        direct_bscore_logit=True,
        antisymmetric_decoder=True,
        normalize_node_features=True,
        aggregation="mean",
    ),
    "residual_connections": ModelExperimentConfig(
        edge_preset="full",
        direct_bscore_logit=True,
        antisymmetric_decoder=True,
        normalize_node_features=True,
        aggregation="mean",
        residual_connections=True,
    ),
    "tournament_context": ModelExperimentConfig(
        edge_preset="full",
        direct_bscore_logit=True,
        antisymmetric_decoder=True,
        normalize_node_features=True,
        aggregation="mean",
        residual_connections=True,
        tournament_context_edges=True,
    ),
    "full_non_temporal": ModelExperimentConfig(
        edge_preset="full",
        direct_bscore_logit=True,
        antisymmetric_decoder=True,
        normalize_node_features=True,
        aggregation="mean",
        residual_connections=True,
        tournament_context_edges=True,
    ),
}

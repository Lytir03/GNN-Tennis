"""Single source of truth for model and training configuration.

Two things are deliberately separated here:

``ModelConfig``
    *What the model is* - architecture and information flags.  These are the
    knobs an ablation turns.

``TrainConfig``
    *How the model is fitted* - optimiser, schedule and calibration.  These are
    the knobs the validation search turns.

Keeping them apart is what makes a clean ablation possible: an architecture
comparison must hold the training recipe fixed, and a tuning run must hold the
architecture fixed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping


@dataclass(frozen=True)
class ModelConfig:
    """Architecture and information content of the model."""

    edge_preset: str = "full"
    direct_bscore_logit: bool = True
    node_bscore_features: bool = True
    antisymmetric_decoder: bool = True
    normalize_node_features: bool = False
    aggregation: str = "sum"
    residual_connections: bool = False
    tournament_context_edges: bool = False
    # "gine" consumes edge features inside the message function; "gatv2" can
    # only use them to weight attention.  Kept as a flag so the layer choice
    # stays a reproducible one-line experiment.
    conv_type: str = "gine"
    attention_heads: int = 4

    def __post_init__(self) -> None:
        if self.aggregation not in {"sum", "mean"}:
            raise ValueError("aggregation must be 'sum' or 'mean'")
        if self.conv_type not in {"gine", "gatv2"}:
            raise ValueError("conv_type must be 'gine' or 'gatv2'")


@dataclass(frozen=True)
class TrainConfig:
    """Optimisation recipe.  Everything here is selectable on validation."""

    hidden_dim: int = 32
    dropout: float = 0.3
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    steps_per_block: int = 2
    replay_buffer_size: int = 200
    # None replays the whole buffer in one batch, which is what the original
    # notebook did.  An integer samples a mini-batch instead: cheaper per step
    # and noisier, so the same wall-clock budget buys far more updates.
    replay_batch_size: int | None = None
    passes: int = 1
    grad_clip: float | None = None
    # Temperature scaling fitted on validation only.  Preserves the ranking of
    # predictions and therefore accuracy; it only rescales confidence.
    calibrate: bool = False
    seed: int = 42


# The architecture that reproduces the historical `current_gnn` artifact.
# Retained so that any refactor can be checked against a frozen result.
LEGACY_MODEL = ModelConfig(
    edge_preset="current",
    direct_bscore_logit=False,
    antisymmetric_decoder=False,
)
LEGACY_TRAINING = TrainConfig()

# The reference architecture for every ablation below.  It is the best
# configuration found by the previous study: full edge features, B-score as an
# explicit residual, and an antisymmetric decoder.
BASE_MODEL = ModelConfig()


def one_factor_ablations() -> Mapping[str, ModelConfig]:
    """Return ablations that each differ from ``BASE_MODEL`` in one factor.

    The previous study defined its ablations *cumulatively*: every entry
    inherited all changes above it.  Because an early step in that chain
    (node feature normalisation) was strongly harmful, every later entry
    inherited the damage and none of them measured the factor named in its own
    title.  A one-factor-at-a-time design is the only way to attribute an
    effect to the thing being varied.
    """

    return {
        "base": BASE_MODEL,
        # Decoder / information content.
        "no_antisymmetric_decoder": replace(
            BASE_MODEL, antisymmetric_decoder=False
        ),
        "no_direct_bscore": replace(BASE_MODEL, direct_bscore_logit=False),
        "no_node_bscore": replace(BASE_MODEL, node_bscore_features=False),
        "no_bscore_at_all": replace(
            BASE_MODEL,
            direct_bscore_logit=False,
            node_bscore_features=False,
        ),
        # Message passing.
        "mean_aggregation": replace(BASE_MODEL, aggregation="mean"),
        "node_normalization": replace(
            BASE_MODEL, normalize_node_features=True
        ),
        "residual_connections": replace(
            BASE_MODEL, residual_connections=True
        ),
        # Convolution choice.
        "gatv2_instead_of_gine": replace(BASE_MODEL, conv_type="gatv2"),
        # Edge information.
        "tournament_context_edges": replace(
            BASE_MODEL, tournament_context_edges=True
        ),
        "edge_preset_current": replace(BASE_MODEL, edge_preset="current"),
        "edge_preset_signed_game": replace(
            BASE_MODEL, edge_preset="signed_game"
        ),
        "edge_preset_set_margin": replace(
            BASE_MODEL, edge_preset="set_margin"
        ),
        "edge_preset_straight_sets": replace(
            BASE_MODEL, edge_preset="straight_sets"
        ),
        "edge_preset_match_status": replace(
            BASE_MODEL, edge_preset="match_status"
        ),
    }

# Single source of truth for model and training config.
#
# Two things are kept apart on purpose:
# - ModelConfig: what the model IS - architecture and information flags.
#   These are the knobs an ablation turns.
# - TrainConfig: how the model is FITTED - optimiser, schedule, calibration.
#   These are the knobs a validation search turns.
#
# Keeping them separate is what makes a clean ablation possible: an
# architecture comparison has to hold the training recipe fixed, and a
# tuning run has to hold the architecture fixed.

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping


@dataclass(frozen=True)
class ModelConfig:
    # Architecture and what information the model gets to see.
    edge_preset: str = "full"
    direct_bscore_logit: bool = True
    # The direct-logit skip connection always read the surface-agnostic
    # B-score (node column 0), even though the GBDT baseline gets a
    # surface-matched B-score as an explicit feature. This flag switches the
    # skip connection to read the B-score for the match's actual surface
    # instead. Off by default: BASE_MODEL should stay the previously-found
    # best configuration until this is shown to help.
    direct_bscore_surface: bool = False
    node_bscore_features: bool = True
    # The twelve recency-weighted history statistics per player that the GBDT
    # has always received (result balance, game and set margins, straight-sets
    # balance, completion rate, on all surfaces and on the match surface).
    # Giving them to the nodes is the feature-parity fix: without it the GBDT
    # simply knows more about each player than the GNN does, and the depth
    # ablation shows that is exactly where the GNN loses ground.
    node_history_features: bool = False
    # The same twelve statistics, but delivered straight to the match decoder
    # instead of to the nodes.  Node features are smoothed by message passing,
    # and the depth reversal shows that smoothing is exactly what destroys them:
    # two hops beat one hop by 0.0081 on impoverished nodes and *lose* by 0.0059
    # once the nodes carry history.  Routing them past the encoder keeps each
    # player's own record undiluted while the graph still does the two-hop work
    # it is uniquely good at.
    history_to_decoder: bool = False
    antisymmetric_decoder: bool = True
    normalize_node_features: bool = False
    aggregation: str = "sum"
    residual_connections: bool = False
    tournament_context_edges: bool = False
    # The predicted match's own best-of-5 and Grand-Slam flags.  The GBDT
    # baseline always had these; the GNN did not, which made the comparison
    # one of information as much as of architecture.  Best-of-5 in particular
    # changes upset probability, and it is the main thing separating a Slam
    # from a Masters in the combined scope.
    rich_match_context: bool = True
    # "gine" consumes edge features inside the message function; "gatv2" can
    # only use them to weight attention.  Kept as a flag so the layer choice
    # stays a reproducible one-line experiment.
    conv_type: str = "gine"
    attention_heads: int = 4
    # Depth is the receptive field, and here that is the whole question.  With
    # one layer a player's embedding sees only their own opponents - which is
    # exactly the information the GBDT already gets as one-hop history
    # aggregates.  Only from two layers does a shared opponent between the two
    # players enter either embedding.  So `num_layers` is the direct,
    # interventional test of whether relational structure is worth anything:
    # if two hops beat one hop on two-hop-connected matches, the graph earns
    # its place; if not, no subgroup correlation can rescue it.
    num_layers: int = 2
    # The proper "no graph" control: same layers, same parameters, same
    # normalisation, no messages.  Preferred over num_layers=0, which removes
    # the LayerNorms as well and therefore does not isolate message passing.
    disable_message_passing: bool = False

    def __post_init__(self) -> None:
        if self.aggregation not in {"sum", "mean"}:
            raise ValueError("aggregation must be 'sum' or 'mean'")
        if self.conv_type not in {"gine", "gatv2"}:
            raise ValueError("conv_type must be 'gine' or 'gatv2'")
        if self.num_layers < 0:
            raise ValueError("num_layers must not be negative")


@dataclass(frozen=True)
class TrainConfig:
    # The optimisation recipe. Everything here can be picked on validation.
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
    # `seed` fixes the data: which player is 'A' in each match, and therefore
    # the labels themselves.  Two runs with different `seed` are two different
    # evaluation sets and cannot be averaged together.
    seed: int = 42
    # `init_seed` fixes only the weight initialisation and the replay sampling.
    # Varying it while holding `seed` fixed gives genuinely repeated runs of
    # the same problem, which is what an ensemble needs.  None means "follow
    # `seed`", reproducing the original coupled behaviour.
    init_seed: int | None = None

    @property
    def effective_init_seed(self) -> int:
        return self.seed if self.init_seed is None else self.init_seed


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
    # Each of these differs from BASE_MODEL in exactly one field.
    #
    # The old version of this study built its ablations cumulatively -
    # every entry inherited all the changes above it. One early step in that
    # chain (node feature normalisation) turned out to be badly harmful, so
    # every later row inherited the damage and none of them actually
    # measured the thing in its own name. One factor at a time is the only
    # way to know an effect came from what you think it came from.
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
        # Feature parity with the GBDT baseline, by two different routes.
        "history_nodes": replace(BASE_MODEL, node_history_features=True),
        "history_decoder": replace(BASE_MODEL, history_to_decoder=True),
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
        # Receptive field.  Zero hops removes message passing entirely and is
        # the control that says whether the graph contributes at all; one hop
        # restricts the model to each player's own opponents, which is the
        # information the GBDT already has; three hops checks that two is not
        # simply too shallow.
        # zero_hop deletes the LayerNorms too, so it is NOT a clean control;
        # kept only to document that.  Use no_message_passing instead.
        "zero_hop": replace(BASE_MODEL, num_layers=0),
        "no_message_passing": replace(
            BASE_MODEL, disable_message_passing=True
        ),
        "one_hop": replace(BASE_MODEL, num_layers=1),
        "three_hop": replace(BASE_MODEL, num_layers=3),
        # Match context available to the decoder.
        "legacy_match_context": replace(
            BASE_MODEL, rich_match_context=False
        ),
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

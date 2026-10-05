# The one tennis GNN.
#
# This class used to be copy-pasted, with small hand-edited differences,
# into eight notebooks. Now there's one definition, and the differences are
# expressed as ModelConfig flags.

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GINEConv

from tennis_gnn.config import ModelConfig

# surface one-hot (3) + round_scaled (1); the two extra flags come after.
LEGACY_CONTEXT_DIM = 4
# B-score x4, height, handedness; the history statistics come after.
LEGACY_NODE_DIM = 6
# Where the four Elo ratings start in a v4 node tensor: after the six legacy
# columns and the twelve history statistics.
ELO_COLUMN_OFFSET = 18
RATING_WIDTH = 4


def legacy_width(rating_source: str) -> int:
    # How wide the pre-history block is once the rating swap is applied.
    #
    # "both" carries two ratings, so its legacy block is four columns wider.
    # This has to be a function rather than the LEGACY_NODE_DIM constant: a
    # model with node_history_features=False narrows to the legacy block, and
    # narrowing to a fixed 6 would throw the second rating away - which is
    # exactly what happened the first time, silently turning "both" into the
    # control and producing five identical seeds.
    if rating_source == "both":
        return LEGACY_NODE_DIM + RATING_WIDTH
    return LEGACY_NODE_DIM


def height_column(rating_source: str) -> int:
    # Height sits immediately after the rating block, so its index moves when
    # "both" widens that block.
    return legacy_width(rating_source) - 2


def select_rating_columns(x: torch.Tensor, rating_source: str) -> torch.Tensor:
    # Puts the requested rating(s) at the front, ahead of the history block.
    #
    # "elo" returns a tensor of exactly the same width as "bscore", so the
    # swap changes which rating the model sees and nothing else - not the
    # parameter count, not the capacity. "both" is the only option that
    # widens the input, and its extra columns sit inside the legacy block so
    # they survive the node_history_features narrowing.
    if x.shape[1] <= ELO_COLUMN_OFFSET:
        # A pre-v4 tensor has no Elo columns; only "bscore" is meaningful.
        return x
    bscore = x[:, :RATING_WIDTH]
    elo = x[:, ELO_COLUMN_OFFSET:ELO_COLUMN_OFFSET + RATING_WIDTH]
    rest = x[:, RATING_WIDTH:ELO_COLUMN_OFFSET]
    if rating_source == "bscore":
        return torch.cat([bscore, rest], dim=1)
    if rating_source == "elo":
        return torch.cat([elo, rest], dim=1)
    return torch.cat([bscore, elo, rest], dim=1)


def _segment_standardise(
    values: torch.Tensor, batch: torch.Tensor | None
) -> torch.Tensor:
    # Column-wise z-score within each graph of a batch.
    #
    # batch is None for a single unbatched graph, which is the whole tensor.
    if batch is None:
        mean = values.mean(dim=0, keepdim=True)
        std = values.std(dim=0, keepdim=True, unbiased=False)
        return (values - mean) / torch.where(std < 1e-6, torch.ones_like(std), std)

    count = int(batch.max().item()) + 1
    sums = torch.zeros(count, values.shape[1], device=values.device)
    sums.index_add_(0, batch, values)
    sizes = torch.zeros(count, 1, device=values.device)
    sizes.index_add_(0, batch, torch.ones_like(values[:, :1]))
    mean = sums / sizes.clamp_min(1.0)

    centred = values - mean[batch]
    var = torch.zeros(count, values.shape[1], device=values.device)
    var.index_add_(0, batch, centred * centred)
    std = (var / sizes.clamp_min(1.0)).sqrt()
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return centred / std[batch]


class TennisGNN(nn.Module):
    # A GINE encoder (config.num_layers deep) with a pairwise match decoder.
    #
    # GINE is the right convolution here: the edges carry most of the
    # signal (margin, recency, surface, round), and GINE consumes edge
    # features inside the message function. Attention layers like GATv2
    # can only use edge features to weight neighbours, which is why the GAT
    # variant in this repo scored worse on every metric.

    def __init__(
        self,
        config: ModelConfig,
        *,
        node_in_dim: int,
        edge_in_dim: int,
        match_context_dim: int,
        hidden_dim: int = 32,
        dropout: float = 0.3,
        decoder_extra_dim: int = 0,
        feature_mean: torch.Tensor | None = None,
        feature_std: torch.Tensor | None = None,
    ):
        super().__init__()
        self.config = config
        self.node_encoder = nn.Linear(node_in_dim, hidden_dim)
        # Registered as buffers so they move with the model and are saved with
        # it: these are fitted constants, not parameters, and must never be
        # refit at prediction time.
        self.register_buffer(
            "feature_mean",
            torch.zeros(node_in_dim) if feature_mean is None else feature_mean,
        )
        self.register_buffer(
            "feature_std",
            torch.ones(node_in_dim) if feature_std is None else feature_std,
        )

        def message_mlp() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        def convolution() -> nn.Module:
            if config.conv_type == "gatv2":
                return GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // config.attention_heads,
                    edge_dim=edge_in_dim,
                    heads=config.attention_heads,
                    aggr=config.aggregation,
                )
            return GINEConv(
                message_mlp(), edge_dim=edge_in_dim, aggr=config.aggregation
            )

        self.convolutions = nn.ModuleList(
            convolution() for _ in range(config.num_layers)
        )
        self.norms = nn.ModuleList(
            nn.LayerNorm(hidden_dim) for _ in range(config.num_layers)
        )

        # Extras enter through the same four-way pairing as the embeddings, so
        # swapping (a, b) swaps them too - which is what keeps the
        # antisymmetrisation in `forward` exact rather than approximate.
        self.decoder_extra_dim = decoder_extra_dim
        decoder_in = (
            hidden_dim * 4 + match_context_dim + decoder_extra_dim * 4
        )
        self.match_head = nn.Sequential(
            nn.Linear(decoder_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        # Start as a pure B-score model: the head contributes nothing until it
        # has learnt something, which keeps early online updates stable.
        nn.init.zeros_(self.match_head[-1].weight)
        nn.init.zeros_(self.match_head[-1].bias)
        # Init is configurable because Adam moves this by at most about `lr`
        # per step: from 1.0 it cannot reach the ~15 the raw B-score scale
        # needs within a run's step budget.  See ModelConfig.bscore_scale_init.
        self.bscore_scale = nn.Parameter(
            torch.tensor(float(config.bscore_scale_init))
        )
        self.intercept = nn.Parameter(torch.zeros(1))

    @staticmethod
    def pair_features(h_a: torch.Tensor, h_b: torch.Tensor) -> torch.Tensor:
        return torch.cat([h_a, h_b, h_a - h_b, torch.abs(h_a - h_b)], dim=1)

    def pair_score(
        self,
        h_a: torch.Tensor,
        h_b: torch.Tensor,
        match_context: torch.Tensor,
        extra_a: torch.Tensor | None = None,
        extra_b: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = [self.pair_features(h_a, h_b)]
        if extra_a is not None:
            parts.append(self.pair_features(extra_a, extra_b))
        parts.append(match_context)
        return self.match_head(torch.cat(parts, dim=1)).squeeze(-1)

    def encode(self, data) -> torch.Tensor:
        x = select_rating_columns(data.x, self.config.rating_source)
        if not self.config.node_history_features:
            # Narrow rather than zero, so a model without the history features
            # is exactly the model that existed before they were stored - same
            # input width, same parameter count, same initialisation.
            x = x[:, :legacy_width(self.config.rating_source)]
        if not self.config.node_bscore_features:
            # Zero the four B-score channels while keeping input width and
            # parameter count fixed, so this is an information ablation only.
            x = x.clone()
            x[:, :4] = 0.0
        if self.config.node_feature_scaling == "fixed":
            # Constants fitted on the training blocks and frozen. The
            # narrowing above is a prefix slice, so a prefix of the stored
            # statistics is the right one to apply.
            width = x.shape[1]
            x = (x - self.feature_mean[:width]) / self.feature_std[:width]
        if not self.config.include_height:
            # After scaling, so the column is exactly zero rather than a
            # constant offset.
            x = x.clone()
            x[:, height_column(self.config.rating_source)] = 0.0
        if self.config.normalize_node_features and x.shape[0] > 1:
            x = x.clone()
            continuous = x[:, :5]
            mean = continuous.mean(dim=0, keepdim=True)
            std = continuous.std(dim=0, keepdim=True, unbiased=False)
            std = torch.where(std < 1e-6, torch.ones_like(std), std)
            x[:, :5] = (continuous - mean) / std
        if self.config.per_block_node_norm and x.shape[0] > 1:
            # Same standardisation, but segmented by graph so a batched
            # forward pass and a single-graph forward pass agree.  Without
            # the segmentation the statistics come from whatever happens to
            # share the batch, which is not a property of the block being
            # predicted.
            x = x.clone()
            x[:, :5] = _segment_standardise(
                x[:, :5], getattr(data, "batch", None)
            )

        edge_index = data.edge_index
        edge_attr = data.edge_attr
        if self.config.disable_message_passing:
            # Keep every layer, parameter and normalisation, and remove only the
            # messages.  This is the honest "no graph" control: setting
            # num_layers=0 would also delete the LayerNorms, so a difference
            # against it confounds message passing with normalisation - and with
            # unnormalised node features (height is ~185 against B-scores under
            # 1) that confound is large enough to make training diverge.
            edge_index = edge_index.new_empty((2, 0))
            edge_attr = edge_attr.new_empty((0, edge_attr.shape[1]))

        h = self.node_encoder(x)
        for convolution, norm in zip(self.convolutions, self.norms):
            message = norm(F.relu(convolution(h, edge_index, edge_attr)))
            h = h + message if self.config.residual_connections else message
        return h

    def forward(
        self,
        data,
        player_a_idx: torch.Tensor,
        player_b_idx: torch.Tensor,
        match_context: torch.Tensor,
    ) -> torch.Tensor:
        h = self.encode(data)
        h_a = h[player_a_idx]
        h_b = h[player_b_idx]

        if not self.config.rich_match_context:
            match_context = match_context[:, :LEGACY_CONTEXT_DIM]

        extra_a = extra_b = None
        if self.config.history_to_decoder:
            # Read straight from the stored node tensor, deliberately bypassing
            # `encode`: these are the features message passing must not touch.
            history = select_rating_columns(
                data.x, self.config.rating_source
            )[:, legacy_width(self.config.rating_source):]
            extra_a = history[player_a_idx]
            extra_b = history[player_b_idx]

        score_ab = self.pair_score(h_a, h_b, match_context, extra_a, extra_b)
        if self.config.antisymmetric_decoder:
            # Evaluating both orientations and antisymmetrising guarantees
            # P(A beats B) = 1 - P(B beats A) exactly, rather than leaving the
            # network to approximate a constraint we already know holds.
            # The extras swap with the embeddings, so the guarantee survives.
            score_ba = self.pair_score(
                h_b, h_a, match_context, extra_b, extra_a
            )
            correction = 0.5 * (score_ab - score_ba)
        else:
            correction = score_ab

        if self.config.direct_bscore_logit:
            bscore = (
                data.raw_bscore_surface
                if self.config.direct_bscore_surface
                else data.raw_bscore_general
            )
            skill_gap = bscore[player_a_idx] - bscore[player_b_idx]
            return self.bscore_scale * skill_gap + correction
        if self.config.antisymmetric_decoder:
            # An intercept would break logit(B, A) = -logit(A, B).
            return correction
        return self.intercept + correction

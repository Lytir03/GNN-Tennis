"""The single tennis GNN.

Previously this class was copy-pasted, with small hand-edited differences, into
eight notebooks.  There is now one definition, and the differences are
expressed as ``ModelConfig`` flags.
"""

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


class TennisGNN(nn.Module):
    """GINE encoder (``config.num_layers`` deep) with a pairwise match decoder.

    GINE is the right convolution for this graph: the edges carry most of the
    signal (margin, recency, surface, round) and GINE consumes edge features
    inside the message function.  Attention layers such as GATv2 can only use
    edge features to weight neighbours, which is why the GAT variant in this
    repository scored worse on every metric.
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        node_in_dim: int,
        edge_in_dim: int,
        match_context_dim: int,
        hidden_dim: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.config = config
        self.node_encoder = nn.Linear(node_in_dim, hidden_dim)

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

        self.match_head = nn.Sequential(
            nn.Linear(hidden_dim * 4 + match_context_dim, hidden_dim),
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
        self.bscore_scale = nn.Parameter(torch.tensor(1.0))
        self.intercept = nn.Parameter(torch.zeros(1))

    @staticmethod
    def pair_features(h_a: torch.Tensor, h_b: torch.Tensor) -> torch.Tensor:
        return torch.cat([h_a, h_b, h_a - h_b, torch.abs(h_a - h_b)], dim=1)

    def pair_score(
        self,
        h_a: torch.Tensor,
        h_b: torch.Tensor,
        match_context: torch.Tensor,
    ) -> torch.Tensor:
        return self.match_head(
            torch.cat([self.pair_features(h_a, h_b), match_context], dim=1)
        ).squeeze(-1)

    def encode(self, data) -> torch.Tensor:
        x = data.x
        if not self.config.node_history_features:
            # Narrow rather than zero, so a model without the history features
            # is exactly the model that existed before they were stored - same
            # input width, same parameter count, same initialisation.
            x = x[:, :LEGACY_NODE_DIM]
        if not self.config.node_bscore_features:
            # Zero the four B-score channels while keeping input width and
            # parameter count fixed, so this is an information ablation only.
            x = x.clone()
            x[:, :4] = 0.0
        if self.config.normalize_node_features and x.shape[0] > 1:
            x = x.clone()
            continuous = x[:, :5]
            mean = continuous.mean(dim=0, keepdim=True)
            std = continuous.std(dim=0, keepdim=True, unbiased=False)
            std = torch.where(std < 1e-6, torch.ones_like(std), std)
            x[:, :5] = (continuous - mean) / std

        h = self.node_encoder(x)
        for convolution, norm in zip(self.convolutions, self.norms):
            message = norm(
                F.relu(convolution(h, data.edge_index, data.edge_attr))
            )
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

        score_ab = self.pair_score(h_a, h_b, match_context)
        if self.config.antisymmetric_decoder:
            # Evaluating both orientations and antisymmetrising guarantees
            # P(A beats B) = 1 - P(B beats A) exactly, rather than leaving the
            # network to approximate a constraint we already know holds.
            score_ba = self.pair_score(h_b, h_a, match_context)
            correction = 0.5 * (score_ab - score_ba)
        else:
            correction = score_ab

        if self.config.direct_bscore_logit:
            bscore = data.raw_bscore_general
            skill_gap = bscore[player_a_idx] - bscore[player_b_idx]
            return self.bscore_scale * skill_gap + correction
        if self.config.antisymmetric_decoder:
            # An intercept would break logit(B, A) = -logit(A, B).
            return correction
        return self.intercept + correction

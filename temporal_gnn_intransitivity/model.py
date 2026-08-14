"""A compact recurrent interaction GNN for chronological tennis matches."""

from __future__ import annotations

import torch
from torch import nn


class TemporalTennisGNN(nn.Module):
    def __init__(
        self,
        num_players: int,
        *,
        hidden_dim: int = 32,
        edge_dim: int = 10,
        context_dim: int = 6,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.initial_state = nn.Embedding(num_players, hidden_dim)
        nn.init.normal_(self.initial_state.weight, std=0.02)
        self.message = nn.Sequential(
            nn.Linear(hidden_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.GRUCell(hidden_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 4 + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def new_state(self) -> torch.Tensor:
        return self.initial_state.weight

    @staticmethod
    def pair_features(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.cat([a, b, a - b, torch.abs(a - b)], dim=-1)

    def _pair_score(
        self, a: torch.Tensor, b: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        return self.head(
            torch.cat([self.pair_features(a, b), context], dim=-1)
        ).squeeze(-1)

    def predict(
        self,
        state: torch.Tensor,
        player_a: torch.Tensor,
        player_b: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        a = state[player_a]
        b = state[player_b]
        # Exact order antisymmetry prevents orientation artefacts.
        return 0.5 * (
            self._pair_score(a, b, context)
            - self._pair_score(b, a, context)
        )

    def update_block(
        self,
        state: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        messages = self.message(
            torch.cat([state[source], edge_features], dim=-1)
        )
        aggregated = state.new_zeros((state.shape[0], self.hidden_dim))
        counts = state.new_zeros((state.shape[0], 1))
        aggregated = aggregated.index_add(0, target, messages)
        counts = counts.index_add(
            0, target, torch.ones_like(target, dtype=state.dtype)[:, None]
        )
        active = counts.squeeze(-1) > 0
        mean_message = aggregated[active] / counts[active].clamp_min(1.0)
        new_values = self.update(mean_message, state[active])
        active_indices = active.nonzero(as_tuple=False).squeeze(-1)
        return state.index_copy(0, active_indices, new_values)

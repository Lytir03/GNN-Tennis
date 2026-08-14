"""Tests for the consolidated GNN package."""

from __future__ import annotations

from dataclasses import asdict, fields
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tennis_gnn.config import (  # noqa: E402
    BASE_MODEL,
    ModelConfig,
    one_factor_ablations,
)
from tennis_gnn.model import TennisGNN  # noqa: E402
from tennis_gnn.train import fit_temperature, metrics  # noqa: E402


def toy_graph(num_nodes: int = 6, edge_dim: int = 10) -> Data:
    generator = torch.Generator().manual_seed(0)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 1, 2], [1, 2, 3, 4, 5, 0, 0, 1]],
        dtype=torch.long,
    )
    x = torch.rand(num_nodes, 6, generator=generator)
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=torch.rand(edge_index.shape[1], edge_dim, generator=generator),
        raw_bscore_general=x[:, 0].clone(),
    )


def build(config: ModelConfig) -> TennisGNN:
    torch.manual_seed(0)
    model = TennisGNN(
        config,
        node_in_dim=6,
        edge_in_dim=10,
        match_context_dim=4,
        hidden_dim=16,
        dropout=0.0,
    )
    # The head is initialised to zero on purpose; give it real weights so the
    # decoder symmetry is actually exercised.
    torch.manual_seed(1)
    for parameter in model.match_head[-1].parameters():
        torch.nn.init.normal_(parameter, std=0.5)
    return model.eval()


class TestAntisymmetricDecoder(unittest.TestCase):
    def test_swapping_players_negates_the_logit(self) -> None:
        model = build(BASE_MODEL)
        data = toy_graph()
        a = torch.tensor([0, 2, 4])
        b = torch.tensor([1, 3, 5])
        context = torch.rand(3, 4, generator=torch.Generator().manual_seed(2))

        with torch.no_grad():
            forward = model(data, a, b, context)
            reverse = model(data, b, a, context)

        torch.testing.assert_close(forward, -reverse, atol=1e-6, rtol=0)

    def test_probabilities_are_complementary(self) -> None:
        model = build(BASE_MODEL)
        data = toy_graph()
        a = torch.tensor([0, 2])
        b = torch.tensor([1, 3])
        context = torch.rand(2, 4, generator=torch.Generator().manual_seed(3))

        with torch.no_grad():
            p_ab = torch.sigmoid(model(data, a, b, context))
            p_ba = torch.sigmoid(model(data, b, a, context))

        torch.testing.assert_close(
            p_ab + p_ba, torch.ones_like(p_ab), atol=1e-6, rtol=0
        )

    def test_plain_decoder_is_not_antisymmetric(self) -> None:
        """Guards the claim that the constraint is not free."""

        from dataclasses import replace

        model = build(replace(BASE_MODEL, antisymmetric_decoder=False))
        data = toy_graph()
        a = torch.tensor([0, 2])
        b = torch.tensor([1, 3])
        context = torch.rand(2, 4, generator=torch.Generator().manual_seed(4))

        with torch.no_grad():
            forward = model(data, a, b, context)
            reverse = model(data, b, a, context)

        self.assertFalse(torch.allclose(forward, -reverse, atol=1e-4))


class TestOneFactorAblations(unittest.TestCase):
    def test_each_ablation_changes_exactly_one_field(self) -> None:
        """The previous ablation set was cumulative.

        Every entry inherited all earlier changes, so an early harmful step
        contaminated every later result and no entry measured the factor named
        in its own title.  This test makes that mistake impossible to
        reintroduce silently.
        """

        base = asdict(BASE_MODEL)
        offenders = {}
        for name, config in one_factor_ablations().items():
            if name == "base":
                continue
            changed = [
                field.name
                for field in fields(ModelConfig)
                if asdict(config)[field.name] != base[field.name]
            ]
            # "no_bscore_at_all" is the deliberate two-factor combination of
            # the two single B-score ablations.
            allowed = 2 if name == "no_bscore_at_all" else 1
            if len(changed) > allowed:
                offenders[name] = changed
        self.assertEqual(offenders, {})

    def test_base_is_the_reference_configuration(self) -> None:
        self.assertEqual(one_factor_ablations()["base"], BASE_MODEL)


class TestNodeAblation(unittest.TestCase):
    def test_zeroing_node_bscore_leaves_the_direct_prior_intact(self) -> None:
        from dataclasses import replace

        data = toy_graph()
        original = data.raw_bscore_general.clone()
        model = build(replace(BASE_MODEL, node_bscore_features=False))
        with torch.no_grad():
            model(
                data,
                torch.tensor([0]),
                torch.tensor([1]),
                torch.rand(1, 4),
            )
        # The decoder's skill prior must survive the input ablation, and the
        # ablation must not mutate the caller's graph.
        torch.testing.assert_close(data.raw_bscore_general, original)
        torch.testing.assert_close(data.x[:, 0], original)


class TestTemperatureScaling(unittest.TestCase):
    def test_temperature_does_not_change_accuracy(self) -> None:
        rng = np.random.default_rng(0)
        logits = rng.normal(scale=2.0, size=500)
        y = (rng.random(500) < 1 / (1 + np.exp(-logits))).astype(float)

        temperature = fit_temperature(logits, y)
        self.assertGreater(temperature, 0.0)

        before = metrics(
            _frame(1.0 / (1.0 + np.exp(-logits)), y)
        )
        after = metrics(
            _frame(1.0 / (1.0 + np.exp(-logits / temperature)), y)
        )
        self.assertAlmostEqual(before["accuracy"], after["accuracy"], places=12)

    def test_temperature_improves_a_deliberately_overconfident_model(
        self,
    ) -> None:
        rng = np.random.default_rng(1)
        true_logits = rng.normal(scale=1.0, size=2000)
        y = (rng.random(2000) < 1 / (1 + np.exp(-true_logits))).astype(float)
        overconfident = true_logits * 3.0

        temperature = fit_temperature(overconfident, y)
        before = metrics(_frame(1 / (1 + np.exp(-overconfident)), y))
        after = metrics(
            _frame(1 / (1 + np.exp(-overconfident / temperature)), y)
        )
        self.assertLess(after["log_loss"], before["log_loss"])
        self.assertGreater(temperature, 1.5)


def _frame(probability, y):
    import pandas as pd

    return pd.DataFrame({"probability": probability, "y_true": y})


if __name__ == "__main__":
    unittest.main()

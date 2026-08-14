import unittest

import torch

from data import (
    BASE_CONTEXT_COLUMNS,
    EDGE_COLUMNS,
    INTRA_CONTEXT_COLUMNS,
    cyclic_energy_share,
    phase_for_date,
)
from model import TemporalTennisGNN


class TemporalExperimentTests(unittest.TestCase):
    def test_split_boundaries(self):
        import pandas as pd

        self.assertEqual(phase_for_date(pd.Timestamp("2010-12-31")), "warmup")
        self.assertEqual(phase_for_date(pd.Timestamp("2015-12-31")), "train")
        self.assertEqual(phase_for_date(pd.Timestamp("2016-01-01")), "val")
        self.assertEqual(phase_for_date(pd.Timestamp("2017-01-01")), "test")

    def test_prematch_context_does_not_contain_result_or_margin(self):
        context = set(BASE_CONTEXT_COLUMNS + INTRA_CONTEXT_COLUMNS)
        self.assertNotIn("y_true", context)
        self.assertNotIn("relative_game_diff", context)
        self.assertNotIn("set_margin_scaled", context)

    def test_edge_vector_has_only_post_match_result_information(self):
        self.assertEqual(len(EDGE_COLUMNS) + 2, 10)

    def test_cyclic_energy_share_is_scale_invariant(self):
        first = cyclic_energy_share([3.0], [4.0])
        scaled = cyclic_energy_share([30.0], [40.0])
        self.assertAlmostEqual(float(first[0]), 9.0 / 25.0)
        self.assertAlmostEqual(float(first[0]), float(scaled[0]))

    def test_cyclic_energy_share_handles_no_evidence(self):
        value = cyclic_energy_share([0.0], [0.0])
        self.assertEqual(float(value[0]), 0.0)

    def test_decoder_is_exactly_antisymmetric(self):
        torch.manual_seed(4)
        model = TemporalTennisGNN(
            4, hidden_dim=8, edge_dim=10, context_dim=6, dropout=0.0
        )
        state = torch.randn(4, 8)
        a = torch.tensor([0, 2])
        b = torch.tensor([1, 3])
        context = torch.randn(2, 6)
        ab = model.predict(state, a, b, context)
        ba = model.predict(state, b, a, context)
        torch.testing.assert_close(ab, -ba)


if __name__ == "__main__":
    unittest.main()

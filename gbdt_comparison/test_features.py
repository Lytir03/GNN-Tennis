from collections import deque
from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import HistoricalPerformance, aggregate_history, phase_for_year


class HistoricalFeatureTests(unittest.TestCase):
    def test_weighted_history_uses_player_perspective(self):
        records = deque(
            [
                HistoricalPerformance(
                    pd.Timestamp("2020-01-01"),
                    "Hard",
                    1.0,
                    0.2,
                    1 / 3,
                    1.0,
                    1.0,
                ),
                HistoricalPerformance(
                    pd.Timestamp("2020-01-02"),
                    "Clay",
                    -1.0,
                    -0.1,
                    -1 / 3,
                    0.0,
                    1.0,
                ),
            ]
        )
        result = aggregate_history(
            records,
            pd.Timestamp("2020-01-03"),
            surface="Hard",
            alpha_days=365,
        )
        self.assertAlmostEqual(result["result_balance"], 1.0)
        self.assertAlmostEqual(result["game_margin"], 0.2)

    def test_empty_history_is_zero(self):
        result = aggregate_history(
            deque(),
            pd.Timestamp("2020-01-03"),
            surface=None,
            alpha_days=365,
        )
        self.assertTrue(all(value == 0.0 for value in result.values()))

    def test_temporal_phases_match_gnn(self):
        self.assertEqual(phase_for_year(2015), "train")
        self.assertEqual(phase_for_year(2016), "val")
        self.assertEqual(phase_for_year(2017), "test")


if __name__ == "__main__":
    unittest.main()

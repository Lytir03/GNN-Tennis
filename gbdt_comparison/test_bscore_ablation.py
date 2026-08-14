import unittest

import pandas as pd

from train import select_feature_columns


class BScoreFeatureAblationTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(
            {
                "node_bscore_general_a": [1.0],
                "node_bscore_surface_diff": [0.2],
                "node_height_diff": [3.0],
                "history_all_result_diff": [0.4],
                "phase": ["train"],
                "y_true": [1],
            }
        )

    def test_full_keeps_bscore(self):
        columns = select_feature_columns(self.frame, "full")
        self.assertIn("node_bscore_general_a", columns)

    def test_no_bscore_removes_only_node_bscore_columns(self):
        columns = select_feature_columns(self.frame, "no_bscore")
        self.assertNotIn("node_bscore_general_a", columns)
        self.assertNotIn("node_bscore_surface_diff", columns)
        self.assertIn("node_height_diff", columns)
        self.assertIn("history_all_result_diff", columns)


if __name__ == "__main__":
    unittest.main()

import unittest

from experiment_config import MODEL_EXPERIMENTS


class TargetedExperimentConfigTests(unittest.TestCase):
    def test_targeted_experiments_do_not_inherit_rejected_changes(self):
        for name in (
            "antisymmetric_no_direct_bscore",
            "antisymmetric_no_node_bscore",
            "antisymmetric_no_bscore",
            "antisymmetric_straight_sets",
            "antisymmetric_mean",
            "antisymmetric_tournament_context",
            "antisymmetric_mean_tournament",
        ):
            config = MODEL_EXPERIMENTS[name]
            self.assertTrue(config.antisymmetric_decoder)
            self.assertFalse(config.normalize_node_features)
            self.assertFalse(config.residual_connections)

    def test_bscore_sources_are_ablated_independently(self):
        full = MODEL_EXPERIMENTS["antisymmetric_decoder"]
        no_direct = MODEL_EXPERIMENTS["antisymmetric_no_direct_bscore"]
        no_node = MODEL_EXPERIMENTS["antisymmetric_no_node_bscore"]
        none = MODEL_EXPERIMENTS["antisymmetric_no_bscore"]

        self.assertEqual(
            (full.direct_bscore_logit, full.node_bscore_features),
            (True, True),
        )
        self.assertEqual(
            (no_direct.direct_bscore_logit, no_direct.node_bscore_features),
            (False, True),
        )
        self.assertEqual(
            (no_node.direct_bscore_logit, no_node.node_bscore_features),
            (True, False),
        )
        self.assertEqual(
            (none.direct_bscore_logit, none.node_bscore_features),
            (False, False),
        )

    def test_mean_and_tournament_are_isolated(self):
        mean = MODEL_EXPERIMENTS["antisymmetric_mean"]
        tournament = MODEL_EXPERIMENTS[
            "antisymmetric_tournament_context"
        ]
        self.assertEqual(mean.aggregation, "mean")
        self.assertFalse(mean.tournament_context_edges)
        self.assertEqual(tournament.aggregation, "sum")
        self.assertTrue(tournament.tournament_context_edges)


if __name__ == "__main__":
    unittest.main()

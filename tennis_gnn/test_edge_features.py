from datetime import datetime
import unittest

from tennis_gnn.edge_features import (
    ABLATION_PRESETS,
    EdgeFeatureConfig,
    build_bidirectional_edges,
    edge_feature_names,
    parse_match_score,
)


class ScoreParsingTests(unittest.TestCase):
    def test_best_of_five_straight_sets(self):
        parsed = parse_match_score("6-3 7-6(4) 6-2", 5)
        self.assertAlmostEqual(parsed.relative_game_diff, 8 / 30)
        self.assertAlmostEqual(parsed.set_margin_scaled, 3 / 5)
        self.assertEqual(parsed.straight_sets_flag, 1.0)
        self.assertEqual(parsed.completed_match_flag, 1.0)

    def test_deciding_set_is_not_straight_sets(self):
        parsed = parse_match_score("6-4 3-6 7-5", 3)
        self.assertAlmostEqual(parsed.set_margin_scaled, 1 / 3)
        self.assertEqual(parsed.straight_sets_flag, 0.0)

    def test_retirement_is_flagged_and_default_margins_are_zero(self):
        parsed = parse_match_score("6-2 3-0 RET", 3)
        self.assertEqual(parsed.retirement_flag, 1.0)
        self.assertEqual(parsed.completed_match_flag, 0.0)
        self.assertEqual(parsed.relative_game_diff, 0.0)
        self.assertEqual(parsed.set_margin_scaled, 0.0)
        self.assertEqual(parsed.straight_sets_flag, 0.0)

    def test_walkover_is_incomplete_without_a_score(self):
        parsed = parse_match_score("W/O", 3)
        self.assertEqual(parsed.walkover_flag, 1.0)
        self.assertEqual(parsed.completed_match_flag, 0.0)
        self.assertEqual(parsed.parsed_set_count, 0)

    def test_played_policy_retains_retirement_partial_score(self):
        parsed = parse_match_score(
            "6-2 3-0 RET", 3, incomplete_margin_policy="played"
        )
        self.assertAlmostEqual(parsed.relative_game_diff, 7 / 11)
        self.assertAlmostEqual(parsed.set_margin_scaled, 1 / 3)


class DirectionTests(unittest.TestCase):
    def test_full_vector_has_requested_signs_and_order(self):
        row = {
            "winner_name": "Winner",
            "loser_name": "Loser",
            "tourney_date": datetime(2020, 1, 1),
            "score": "6-4 6-4",
            "best_of": 3,
            "surface": "Hard",
            "round_order": 7,
        }
        config = ABLATION_PRESETS["full"]
        pairs, attrs, stats = build_bidirectional_edges(
            [row],
            {"Loser": 0, "Winner": 1},
            datetime(2020, 1, 1),
            config,
        )
        self.assertEqual(pairs, [[0, 1], [1, 0]])
        self.assertEqual(
            edge_feature_names(config),
            (
                "recency_weight",
                "signed_relative_game_diff",
                "signed_set_margin",
                "straight_sets_flag",
                "completed_match_flag",
                "surface_hard",
                "surface_clay",
                "surface_grass",
                "round_scaled",
                "result_direction",
            ),
        )
        self.assertEqual(len(attrs[0]), 10)
        self.assertGreater(attrs[0][1], 0)
        self.assertEqual(attrs[1][1], -attrs[0][1])
        self.assertGreater(attrs[0][2], 0)
        self.assertEqual(attrs[1][2], -attrs[0][2])
        self.assertEqual(attrs[0][3:9], attrs[1][3:9])
        self.assertEqual(attrs[0][-1], 1.0)
        self.assertEqual(attrs[1][-1], -1.0)
        self.assertEqual(stats["matches_used"], 1)

    def test_current_preset_reproduces_unsigned_margin(self):
        row = {
            "winner_name": "W",
            "loser_name": "L",
            "tourney_date": datetime(2020, 1, 1),
            "score": "6-4 6-4",
            "best_of": 3,
            "surface": "Clay",
            "round_order": 1,
        }
        _, attrs, _ = build_bidirectional_edges(
            [row], {"L": 0, "W": 1}, datetime(2020, 1, 2),
            ABLATION_PRESETS["current"],
        )
        self.assertEqual(attrs[0][1], attrs[1][1])
        self.assertEqual((attrs[0][-1], attrs[1][-1]), (0.0, 1.0))

    def test_optional_tournament_context(self):
        row = {
            "winner_name": "W",
            "loser_name": "L",
            "tourney_date": datetime(2020, 1, 1),
            "score": "6-4 6-4 6-4",
            "best_of": 5,
            "tourney_level": "G",
            "surface": "Hard",
            "round_order": 1,
        }
        config = EdgeFeatureConfig(include_tournament_context=True)
        _, attrs, _ = build_bidirectional_edges(
            [row], {"L": 0, "W": 1}, datetime(2020, 1, 1), config
        )
        names = edge_feature_names(config)
        self.assertEqual(
            attrs[0][names.index("best_of_5_flag")], 1.0
        )
        self.assertEqual(
            attrs[0][names.index("grand_slam_flag")], 1.0
        )
        self.assertEqual(attrs[0][2:4], attrs[1][2:4])


if __name__ == "__main__":
    unittest.main()

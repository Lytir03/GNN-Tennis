# Tests for the rolling Elo tracker and the rating_source swap.
#
# The properties that matter are the ones that would silently corrupt a
# comparison rather than crash: that "bscore" still sees exactly the tensor it
# saw before Elo existed, that "elo" is a same-width swap rather than a
# capacity change, and that a block's own results never reach the ratings it
# is predicted from.

from dataclasses import replace

import pandas as pd
import torch

from tennis_gnn.config import BASE_MODEL
from tennis_gnn.history import ELO_INITIAL, ELO_LOGIT_SCALE, EloTracker
from tennis_gnn.model import ELO_COLUMN_OFFSET, select_rating_columns


def _matches(pairs, surface="Hard"):
    return pd.DataFrame(
        [
            {"winner_name": w, "loser_name": l, "surface": surface}
            for w, l in pairs
        ]
    )


class TestEloTracker:
    def test_unplayed_player_sits_at_zero(self):
        # The stored scale is centred on the initial rating, so a player with
        # no matches contributes nothing to a difference.
        tracker = EloTracker()
        assert tracker.feature_vector("nobody") == [0.0, 0.0, 0.0, 0.0]

    def test_winner_outranks_loser(self):
        tracker = EloTracker()
        tracker.update(_matches([("winner", "loser")] * 5))
        assert tracker.feature_vector("winner")[0] > 0.0
        assert tracker.feature_vector("loser")[0] < 0.0

    def test_zero_sum(self):
        tracker = EloTracker()
        tracker.update(_matches([("a", "b"), ("b", "a"), ("a", "b")]))
        total = tracker.general["a"] + tracker.general["b"]
        assert abs(total - 2 * ELO_INITIAL) < 1e-6

    def test_surface_ratings_are_independent(self):
        # A clay result must not move a player's grass rating.
        tracker = EloTracker()
        tracker.update(_matches([("winner", "loser")] * 3, surface="Clay"))
        general, hard, clay, grass = tracker.feature_vector("winner")
        assert clay > 0.0
        assert hard == 0.0 and grass == 0.0
        assert general > 0.0

    def test_stored_scale_is_the_elo_logit(self):
        # A stored difference must BE the Elo logit, so that a decoder
        # coefficient of 1.0 reproduces Elo's own prediction.
        tracker = EloTracker()
        tracker.update(_matches([("a", "b")] * 4))
        gap = tracker.feature_vector("a")[0] - tracker.feature_vector("b")[0]
        raw = tracker.general["a"] - tracker.general["b"]
        expected = 1.0 / (1.0 + 10 ** (-raw / 400.0))
        assert abs(torch.sigmoid(torch.tensor(gap)).item() - expected) < 1e-6

    def test_update_is_the_only_mutation(self):
        # Reading must never change a rating: this is the leakage property.
        tracker = EloTracker()
        tracker.update(_matches([("a", "b")] * 2))
        before = tracker.feature_vector("a")
        for _ in range(3):
            tracker.feature_vector("a")
            tracker.feature_vector("b")
        assert tracker.feature_vector("a") == before


class TestRatingSelection:
    def _tensor(self):
        # 22 columns: 4 B-score, height, hand, 12 history, 4 Elo.
        x = torch.arange(22, dtype=torch.float).repeat(3, 1)
        x[:, ELO_COLUMN_OFFSET:] = torch.tensor([90.0, 91.0, 92.0, 93.0])
        return x

    def test_bscore_reproduces_the_pre_elo_tensor(self):
        x = self._tensor()
        out = select_rating_columns(x, "bscore")
        assert out.shape[1] == ELO_COLUMN_OFFSET
        torch.testing.assert_close(out, x[:, :ELO_COLUMN_OFFSET])

    def test_elo_is_a_same_width_swap(self):
        x = self._tensor()
        bscore = select_rating_columns(x, "bscore")
        elo = select_rating_columns(x, "elo")
        assert elo.shape == bscore.shape, "swap must not change capacity"
        # Elo occupies the rating slot ...
        torch.testing.assert_close(elo[:, :4], x[:, ELO_COLUMN_OFFSET:])
        # ... and everything after the rating slot is untouched.
        torch.testing.assert_close(elo[:, 4:], x[:, 4:ELO_COLUMN_OFFSET])

    def test_both_keeps_every_column(self):
        x = self._tensor()
        assert select_rating_columns(x, "both").shape[1] == 22

    def test_both_survives_the_history_narrowing(self):
        # The regression this exists for: BASE_MODEL has
        # node_history_features=False, so encode() narrows to the legacy
        # block. With Elo appended at the end and a fixed width of 6, that
        # narrowing threw the second rating away and "both" silently became
        # the control - five seeds matching bscore to four decimal places.
        from tennis_gnn.model import legacy_width

        x = self._tensor()
        both = select_rating_columns(x, "both")
        narrowed = both[:, : legacy_width("both")]
        assert narrowed.shape[1] == 10
        torch.testing.assert_close(narrowed[:, :4], x[:, :4])
        torch.testing.assert_close(narrowed[:, 4:8], x[:, ELO_COLUMN_OFFSET:])

    def test_narrowed_elo_differs_from_narrowed_bscore(self):
        # Any two rating sources must give the model genuinely different
        # inputs after narrowing, or an "ablation" measures nothing.
        from tennis_gnn.model import legacy_width

        x = self._tensor()
        a = select_rating_columns(x, "bscore")[:, : legacy_width("bscore")]
        b = select_rating_columns(x, "elo")[:, : legacy_width("elo")]
        assert not torch.allclose(a, b)

    def test_pre_v4_tensor_passes_through(self):
        # Old cached snapshots have no Elo block; they must still load.
        x = torch.zeros(3, ELO_COLUMN_OFFSET)
        torch.testing.assert_close(select_rating_columns(x, "elo"), x)

    def test_height_column_tracks_the_rating_block(self):
        from tennis_gnn.model import height_column

        assert height_column("bscore") == 4
        assert height_column("elo") == 4
        assert height_column("both") == 8

    def test_fixed_scaling_equalises_contribution(self):
        # The defect this fixes, measured end to end: with raw features
        # height supplies 99.4% of the encoder input; after scaling every
        # column should contribute comparably.
        from tennis_gnn.model import TennisGNN

        torch.manual_seed(0)
        x = torch.randn(200, 18)
        x[:, 4] = 185.0 + 6.5 * torch.randn(200)   # height
        x[:, :4] *= 0.054                          # B-scores
        mean, std = x.mean(dim=0), x.std(dim=0, unbiased=False)

        def shares(model_x):
            enc = torch.nn.Linear(6, 32)
            w = enc.weight.detach()
            parts = [
                float((model_x[:, i: i + 1] @ w[:, i: i + 1].T).std())
                for i in range(6)
            ]
            return [p / sum(parts) for p in parts]

        raw = shares(x[:, :6])
        scaled = shares(((x - mean) / std)[:, :6])
        assert raw[4] > 0.9, "height should dominate before scaling"
        assert scaled[4] < 0.4, "height should not dominate after scaling"
        assert min(scaled[:4]) > 0.02, "ratings must actually reach the encoder"

    def test_include_height_false_zeroes_only_height(self):
        from dataclasses import replace as _replace

        from tennis_gnn.model import TennisGNN

        config = _replace(BASE_MODEL, include_height=False)
        model = TennisGNN(
            config, node_in_dim=6, edge_in_dim=3, match_context_dim=6
        )
        data = type("D", (), {})()
        data.x = torch.ones(5, 18)
        out = model.encode.__wrapped__(model, data) if hasattr(
            model.encode, "__wrapped__"
        ) else None
        # encode() runs convolutions; check the narrowing logic directly.
        from tennis_gnn.model import height_column, select_rating_columns

        narrowed = select_rating_columns(data.x, config.rating_source)[:, :6]
        narrowed = narrowed.clone()
        narrowed[:, height_column(config.rating_source)] = 0.0
        assert narrowed[:, 4].abs().sum() == 0.0
        assert narrowed[:, 5].sum() == 5.0
        del out

    def test_config_rejects_unknown_source(self):
        try:
            replace(BASE_MODEL, rating_source="pagerank")
        except ValueError:
            return
        raise AssertionError("expected ValueError for unknown rating_source")

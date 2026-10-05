# Tests for the B-score conditioning transforms and the per-block node
# normalisation added for the gnn_improvements/ experiments.
#
# The properties worth locking down are the ones the old
# normalize_node_features flag violates: a transform must not depend on what
# else happens to share the batch, and "raw" must be bit-identical to the
# behaviour every published artifact was produced under.

from dataclasses import replace

import torch
from torch_geometric.data import Batch, Data

from tennis_gnn.config import BASE_MODEL
from tennis_gnn.model import _segment_standardise
from tennis_gnn.train import to_pyg
from tennis_gnn.snapshots import BlockSnapshot


def _snapshot(bscores, surface="Hard", node_width=18):
    n = len(bscores)
    x = torch.zeros(n, node_width)
    x[:, 0] = torch.tensor(bscores, dtype=torch.float)
    x[:, 1] = torch.tensor(bscores, dtype=torch.float) * 2.0
    x[:, 4] = 185.0
    return BlockSnapshot(
        block_idx=0,
        tourney_id="t",
        round_order=1,
        phase="train",
        year=2011,
        x=x,
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, 3)),
        player_a=torch.tensor([0]),
        player_b=torch.tensor([1]),
        context=torch.zeros(1, 6),
        y=torch.tensor([1.0]),
        bscore_diff=torch.tensor([0.0]),
        intransitivity_level=["missing"],
        surface=surface,
    )


class TestBscoreTransform:
    def test_raw_is_unchanged(self):
        # Every published artifact was produced on the raw channel; this must
        # stay bit-identical or old results stop being reproducible.
        values = [5.6e-4, 1.2e-2, 3.1e-1, 1e-21]
        snapshot = _snapshot(values)
        data = to_pyg(snapshot, BASE_MODEL)
        torch.testing.assert_close(
            data.raw_bscore_general, snapshot.x[:, 0]
        )

    def test_log_is_floored(self):
        # An unfloored log of a 1e-21 B-score is -48, which would dominate the
        # skip connection with what really just means "unranked".
        snapshot = _snapshot([5.6e-4, 1e-21])
        data = to_pyg(
            snapshot, replace(BASE_MODEL, bscore_transform="log")
        )
        assert data.raw_bscore_general.min() >= torch.log(
            torch.tensor(1e-9)
        ) - 1e-5

    def test_logz_is_standardised(self):
        snapshot = _snapshot([5.6e-4, 1.2e-2, 3.1e-1, 2e-3, 7e-2])
        data = to_pyg(
            snapshot, replace(BASE_MODEL, bscore_transform="logz")
        )
        assert abs(float(data.raw_bscore_general.mean())) < 1e-5
        assert abs(float(data.raw_bscore_general.std(unbiased=False)) - 1.0) < 1e-5

    def test_zscore_is_scale_invariant(self):
        # The point of the transform: the same relative spread must produce
        # the same conditioned values whatever the graph size does to the
        # absolute magnitude of eigenvector centrality.
        small = _snapshot([1e-4, 2e-4, 4e-4])
        large = _snapshot([1e-2, 2e-2, 4e-2])
        cfg = replace(BASE_MODEL, bscore_transform="zscore")
        torch.testing.assert_close(
            to_pyg(small, cfg).raw_bscore_general,
            to_pyg(large, cfg).raw_bscore_general,
        )

    def test_surface_channel_is_transformed_too(self):
        snapshot = _snapshot([5.6e-4, 1.2e-2, 3.1e-1], surface="Hard")
        cfg = replace(BASE_MODEL, bscore_transform="logz")
        data = to_pyg(snapshot, cfg)
        assert abs(float(data.raw_bscore_surface.mean())) < 1e-5


class TestSegmentStandardise:
    def test_batched_matches_single_graph(self):
        # The property normalize_node_features gets wrong: a block's
        # conditioned features must not depend on what shares its batch.
        a = torch.tensor([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]])
        b = torch.tensor([[100.0, 7.0], [900.0, 9.0]])
        alone = _segment_standardise(a, None)
        batch = torch.tensor([0, 0, 0, 1, 1])
        together = _segment_standardise(torch.cat([a, b]), batch)
        torch.testing.assert_close(alone, together[:3])

    def test_constant_column_survives(self):
        values = torch.tensor([[2.0, 1.0], [2.0, 5.0]])
        out = _segment_standardise(values, None)
        assert torch.isfinite(out).all()


class TestScaleInit:
    def test_bscore_scale_init_is_respected(self):
        from tennis_gnn.model import TennisGNN

        model = TennisGNN(
            replace(BASE_MODEL, bscore_scale_init=15.0),
            node_in_dim=18,
            edge_in_dim=3,
            match_context_dim=6,
        )
        assert float(model.bscore_scale) == 15.0

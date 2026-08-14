from pathlib import Path
import tempfile
import unittest

import pandas as pd

from experiment_tracking import ArtifactManifest, save_prediction_artifact
from multi_seed_summary import multi_seed_comparison


class MultiSeedSummaryTests(unittest.TestCase):
    def test_aggregates_and_counts_candidate_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in (1, 2):
                output = root / "slams_masters" / f"seed_{seed}"
                base = pd.DataFrame(
                    {
                        "phase": ["test", "test"],
                        "tourney_id": ["A", "A"],
                        "round_order": [1, 1],
                        "block_idx": [1, 1],
                        "row_in_block": [0, 1],
                        "y_true": [1.0, 0.0],
                        "prob_before": [0.6, 0.4],
                    }
                )
                candidate = base.copy()
                candidate["prob_before"] = [0.8, 0.2]
                logit = base.copy()
                for name, frame in (
                    ("current_gnn", base),
                    ("antisymmetric_decoder", candidate),
                    ("bscore_logit", logit),
                ):
                    save_prediction_artifact(
                        frame,
                        output,
                        ArtifactManifest(
                            experiment=name,
                            model_family=name,
                            tournament_scope="slams_masters",
                            seed=seed,
                            train_end=2015,
                            validation_end=2016,
                            test_end=2020,
                            update_phases=("train",),
                            config={},
                        ),
                    )
            per_seed, aggregate = multi_seed_comparison(
                root, scope="slams_masters", seeds=(1, 2)
            )
            self.assertEqual(len(per_seed), 6)
            self.assertEqual(
                aggregate.loc[
                    "antisymmetric_decoder",
                    "log_loss_wins_vs_current",
                ],
                2,
            )


if __name__ == "__main__":
    unittest.main()

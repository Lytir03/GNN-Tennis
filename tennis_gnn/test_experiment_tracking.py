import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from tennis_gnn.experiment_tracking import (
    ArtifactManifest,
    assert_compatible,
    compare_artifacts,
    load_prediction_artifact,
    save_prediction_artifact,
)


def sample_predictions(probabilities):
    return pd.DataFrame(
        {
            "phase": ["test", "test"],
            "tourney_id": ["A", "A"],
            "round_order": [1, 1],
            "block_idx": [10, 10],
            "row_in_block": [0, 1],
            "y_true": [1.0, 0.0],
            "prob_before": probabilities,
        }
    )


def manifest(name, scope="slams_masters"):
    return ArtifactManifest(
        experiment=name,
        model_family="GINE",
        tournament_scope=scope,
        seed=42,
        train_end=2015,
        validation_end=2016,
        test_end=2020,
        update_phases=("train",),
        config={"name": name},
    )


class ArtifactTests(unittest.TestCase):
    def test_round_trip_and_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            save_prediction_artifact(
                sample_predictions([0.8, 0.2]),
                directory,
                manifest("current_gnn"),
            )
            save_prediction_artifact(
                sample_predictions([0.9, 0.1]),
                directory,
                manifest("candidate"),
            )
            frame, loaded_manifest = load_prediction_artifact(
                directory, "candidate"
            )
            self.assertEqual(len(frame), 2)
            self.assertEqual(loaded_manifest["seed"], 42)
            comparison = compare_artifacts(
                directory, ["current_gnn", "candidate"]
            )
            self.assertGreater(
                comparison.loc["candidate", "delta_vs_current_accuracy"],
                -1e-12,
            )
            self.assertLess(
                comparison.loc["candidate", "delta_vs_current_log_loss"],
                0.0,
            )

    def test_scope_mismatch_is_rejected(self):
        first = {
            **json.loads(json.dumps(manifest("a").__dict__)),
            "evaluation_hash": "same",
        }
        second = {
            **json.loads(
                json.dumps(manifest("b", scope="slams").__dict__)
            ),
            "evaluation_hash": "same",
        }
        with self.assertRaisesRegex(ValueError, "tournament_scope"):
            assert_compatible([first, second])

    def test_changed_labels_are_rejected_by_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            first = sample_predictions([0.8, 0.2])
            second = sample_predictions([0.8, 0.2])
            second.loc[0, "y_true"] = 0.0
            save_prediction_artifact(first, directory, manifest("a"))
            save_prediction_artifact(second, directory, manifest("b"))
            with self.assertRaisesRegex(ValueError, "evaluation_hash"):
                compare_artifacts(directory, ["a", "b"])


if __name__ == "__main__":
    unittest.main()

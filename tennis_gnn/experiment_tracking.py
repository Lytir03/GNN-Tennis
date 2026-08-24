# Saves and compares frozen, match-level experiment predictions.

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


KEY_COLUMNS = ("phase", "tourney_id", "round_order", "block_idx", "row_in_block")
PREDICTION_COLUMNS = (*KEY_COLUMNS, "y_true", "probability")


@dataclass(frozen=True)
class ArtifactManifest:
    experiment: str
    model_family: str
    tournament_scope: str
    seed: int
    train_end: int
    validation_end: int
    test_end: int
    update_phases: tuple[str, ...]
    config: Mapping[str, Any]
    row_count: int = 0
    evaluation_hash: str = ""


def _normalise_predictions(
    predictions: pd.DataFrame,
    *,
    probability_column: str,
) -> pd.DataFrame:
    frame = predictions.copy()
    if "row_in_block" not in frame:
        frame["row_in_block"] = frame.groupby("block_idx", sort=False).cumcount()
    frame = frame.rename(columns={probability_column: "probability"})
    missing = [column for column in PREDICTION_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"Prediction artifact is missing columns: {missing}")
    frame = frame[list(PREDICTION_COLUMNS)].copy()
    if frame.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("Prediction keys are not unique")
    if not frame["probability"].between(0.0, 1.0).all():
        raise ValueError("Probabilities must lie in [0, 1]")
    return frame.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)


def evaluation_hash(frame: pd.DataFrame, *, phase: str | None = None) -> str:
    # Hashes the matches and labels an artifact was scored on.
    #
    # phase restricts the hash to one phase. This matters because
    # artifacts legitimately differ in coverage: a model that records its
    # training-phase predictions for diagnostics has more rows than one
    # that only stores test, so their whole-artifact hashes differ even
    # when the test sets are identical. A comparison over a single phase
    # is valid exactly when the phase-restricted hashes agree, so that's
    # what the comparison code has to check.
    if phase is not None:
        frame = frame[frame["phase"] == phase]
    subset = frame[[*KEY_COLUMNS, "y_true"]].sort_values(list(KEY_COLUMNS))
    payload = subset.to_csv(index=False).encode()
    return hashlib.sha256(payload).hexdigest()


def save_prediction_artifact(
    predictions: pd.DataFrame,
    output_directory: str | Path,
    manifest: ArtifactManifest,
    *,
    probability_column: str = "prob_before",
) -> tuple[Path, Path]:
    # Saves predictions and a compatibility manifest.
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    frame = _normalise_predictions(
        predictions, probability_column=probability_column
    )
    final_manifest = {
        **asdict(manifest),
        "update_phases": list(manifest.update_phases),
        "row_count": len(frame),
        "evaluation_hash": evaluation_hash(frame),
    }
    csv_path = output / f"{manifest.experiment}.csv"
    manifest_path = output / f"{manifest.experiment}.manifest.json"
    frame.to_csv(csv_path, index=False)
    manifest_path.write_text(
        json.dumps(final_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return csv_path, manifest_path


def load_prediction_artifact(
    output_directory: str | Path,
    experiment: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    output = Path(output_directory)
    frame = pd.read_csv(output / f"{experiment}.csv")
    manifest = json.loads(
        (output / f"{experiment}.manifest.json").read_text(encoding="utf-8")
    )
    checked = _normalise_predictions(frame, probability_column="probability")
    if manifest["row_count"] != len(checked):
        raise ValueError(f"Row-count mismatch in artifact {experiment}")
    if manifest["evaluation_hash"] != evaluation_hash(checked):
        raise ValueError(f"Evaluation hash mismatch in artifact {experiment}")
    return checked, manifest


def assert_compatible(
    manifests: Iterable[Mapping[str, Any]],
    *,
    require_same_seed: bool = True,
    require_same_evaluation: bool = True,
) -> None:
    # Checks that artifacts describe the same experimental setup.
    #
    # Set require_same_evaluation=False when the caller has already
    # verified equality on the phase it actually compares (see
    # evaluation_hash's phase argument); the whole-artifact hash is too
    # strict in that case, since it also encodes which phases an artifact
    # happens to store.
    manifests = list(manifests)
    if len(manifests) < 2:
        return
    fields = [
        "tournament_scope",
        "train_end",
        "validation_end",
        "test_end",
        "update_phases",
    ]
    if require_same_evaluation:
        fields.append("evaluation_hash")
    if require_same_seed:
        fields.append("seed")
    reference = manifests[0]
    for manifest in manifests[1:]:
        differences = {
            field: (reference.get(field), manifest.get(field))
            for field in fields
            if reference.get(field) != manifest.get(field)
        }
        if differences:
            raise ValueError(
                "Incompatible experiment artifacts: "
                + json.dumps(differences, sort_keys=True)
            )


def probability_metrics(frame: pd.DataFrame) -> dict[str, float]:
    y = frame["y_true"].to_numpy(dtype=float)
    probability = np.clip(
        frame["probability"].to_numpy(dtype=float), 1e-7, 1 - 1e-7
    )
    prediction = (probability >= 0.5).astype(float)
    return {
        "n_matches": float(len(frame)),
        "accuracy": float(np.mean(prediction == y)),
        "log_loss": float(
            -np.mean(y * np.log(probability) + (1 - y) * np.log(1 - probability))
        ),
        "brier": float(np.mean((probability - y) ** 2)),
    }


def compare_artifacts(
    output_directory: str | Path,
    experiments: Iterable[str],
    *,
    phase: str = "test",
) -> pd.DataFrame:
    loaded = [
        load_prediction_artifact(output_directory, experiment)
        for experiment in experiments
    ]
    assert_compatible([manifest for _, manifest in loaded])
    rows = {}
    for frame, manifest in loaded:
        subset = frame[frame["phase"] == phase]
        rows[manifest["experiment"]] = probability_metrics(subset)
    result = pd.DataFrame.from_dict(rows, orient="index")
    if "current_gnn" in result.index:
        for metric in ("accuracy", "log_loss", "brier"):
            result[f"delta_vs_current_{metric}"] = (
                result[metric] - result.loc["current_gnn", metric]
            )
    return result

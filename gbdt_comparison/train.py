# Tunes and evaluates a HistGradientBoosting tennis baseline.

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from features import build_feature_dataset, model_feature_columns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tennis_gnn.data import SCOPE_FILES  # noqa: E402


def select_feature_columns(
    frame: pd.DataFrame, feature_set: str
) -> list[str]:
    columns = model_feature_columns(frame)
    if feature_set == "full":
        return columns
    if feature_set == "no_bscore":
        return [
            column
            for column in columns
            if not column.startswith("node_bscore_")
        ]
    raise ValueError(f"Unknown feature set: {feature_set}")


def metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    probability = np.clip(probability.astype(float), 1e-7, 1 - 1e-7)
    prediction = probability >= 0.5
    return {
        "n_matches": int(len(y)),
        "accuracy": float(np.mean(prediction == y)),
        "log_loss": float(
            -np.mean(
                y * np.log(probability)
                + (1 - y) * np.log(1 - probability)
            )
        ),
        "brier": float(np.mean((probability - y) ** 2)),
    }


def parameter_grid() -> list[dict[str, object]]:
    values = {
        "learning_rate": (0.03, 0.06),
        "max_iter": (150, 300),
        "max_leaf_nodes": (15, 31),
        "min_samples_leaf": (20, 50),
        "l2_regularization": (0.0, 1.0),
    }
    keys = tuple(values)
    return [
        dict(zip(keys, combination))
        for combination in itertools.product(*(values[key] for key in keys))
    ]


def run(
    project_root: Path,
    *,
    scope: str,
    seed: int,
    rebuild_features: bool,
    feature_set: str,
) -> None:
    _split = SCOPE_FILES[scope][2]
    output = project_root / "gbdt_comparison" / "results" / scope
    output.mkdir(parents=True, exist_ok=True)
    # Pickle preserves floating-point values exactly.  CSV round-trips can
    # perturb values around tree split thresholds and change fitted trees.
    cache = output / f"features_seed_{seed}.pkl"
    if rebuild_features or not cache.exists():
        frame = build_feature_dataset(
            project_root, tournament_scope=scope, seed=seed
        )
        frame.to_pickle(cache)
    else:
        frame = pd.read_pickle(cache)

    feature_columns = select_feature_columns(frame, feature_set)
    train = frame[frame["phase"] == "train"]
    validation = frame[frame["phase"] == "val"]
    test = frame[frame["phase"] == "test"]
    X_train, y_train = train[feature_columns], train["y_true"].to_numpy()
    X_val, y_val = validation[feature_columns], validation["y_true"].to_numpy()

    if feature_set == "full":
        tuning_rows = []
        best_params = None
        best_loss = np.inf
        for params in parameter_grid():
            model = HistGradientBoostingClassifier(
                **params,
                random_state=seed,
                early_stopping=False,
            )
            model.fit(X_train, y_train)
            probability = model.predict_proba(X_val)[:, 1]
            result = metrics(y_val, probability)
            tuning_rows.append({**params, **result})
            if result["log_loss"] < best_loss:
                best_loss = result["log_loss"]
                best_params = params

        assert best_params is not None
        tuning = pd.DataFrame(tuning_rows).sort_values("log_loss")
        tuning.to_csv(output / f"tuning_seed_{seed}.csv", index=False)
        validation_metrics = tuning.iloc[0][
            ["accuracy", "log_loss", "brier"]
        ].to_dict()
    else:
        # This is an information ablation, not a second tuning exercise.
        # Reuse the hyperparameters selected for the full GBDT at this seed.
        full_manifest_path = output / f"gbdt_manifest_seed_{seed}.json"
        if not full_manifest_path.is_file():
            raise FileNotFoundError(
                "Run the full GBDT first so its frozen hyperparameters can "
                f"be reused: {full_manifest_path}"
            )
        full_manifest = json.loads(full_manifest_path.read_text())
        best_params = full_manifest["best_parameters"]
        validation_model = HistGradientBoostingClassifier(
            **best_params,
            random_state=seed,
            early_stopping=False,
        )
        validation_model.fit(X_train, y_train)
        validation_probability = validation_model.predict_proba(X_val)[:, 1]
        validation_metrics = metrics(y_val, validation_probability)

    final_model = HistGradientBoostingClassifier(
        **best_params,
        random_state=seed,
        early_stopping=False,
    )
    # Validation is used only for hyperparameter selection.  Do not refit on
    # validation labels: the GNN comparison also updates on train only.
    final_model.fit(X_train, y_train)
    test_probability = final_model.predict_proba(
        test[feature_columns]
    )[:, 1]
    test_metrics = metrics(test["y_true"].to_numpy(), test_probability)
    predictions = test[
        [
            "phase",
            "tourney_id",
            "round_order",
            "block_idx",
            "row_in_block",
            "y_true",
        ]
    ].copy()
    predictions["probability"] = test_probability
    predictions.to_csv(
        output / f"gbdt_{feature_set}_predictions_seed_{seed}.csv",
        index=False,
    )

    manifest = {
        "model": "HistGradientBoostingClassifier",
        "scope": scope,
        "seed": seed,
        "feature_set": feature_set,
        "selection_metric": "validation_log_loss",
        "hyperparameters_reused_from": (
            None if feature_set == "full" else "gbdt_tuned"
        ),
        "best_parameters": best_params,
        "feature_columns": feature_columns,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "split": {
            "train_end": _split.train_end,
            "validation": _split.val_end,
            "test": f"{_split.val_end + 1}-{int(_split.rolling_end[:4]) - 1}",
        },
    }
    manifest_stem = (
        "gbdt_manifest"
        if feature_set == "full"
        else f"gbdt_{feature_set}_manifest"
    )
    (output / f"{manifest_stem}_seed_{seed}.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    # Also publish the prediction in the same frozen-artifact format used by
    # the GNN experiments, enabling strict match-level compatibility checks.
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from tennis_gnn.experiment_tracking import (
        ArtifactManifest,
        save_prediction_artifact,
    )

    save_prediction_artifact(
        predictions,
        project_root
        / "results"
        / "frozen_predictions"
        / scope
        / f"seed_{seed}",
        ArtifactManifest(
            experiment=(
                "gbdt_tuned"
                if feature_set == "full"
                else f"gbdt_{feature_set}"
            ),
            model_family="HistGradientBoostingClassifier",
            tournament_scope=scope,
            seed=seed,
            train_end=_split.train_end,
            validation_end=_split.val_end,
            test_end=int(_split.rolling_end[:4]) - 1,
            update_phases=("train",),
            config={
                "selection_metric": "validation_log_loss",
                "hyperparameters_reused_from": (
                    None if feature_set == "full" else "gbdt_tuned"
                ),
                "best_parameters": best_params,
                "feature_columns": feature_columns,
            },
        ),
        probability_column="probability",
    )
    print(json.dumps(manifest["test_metrics"], indent=2))
    print("Best parameters:", best_params)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scope",
        choices=(
            "slams", "slams_masters", "full",
            "slams_masters_1990", "full_1990",
        ),
        default="slams_masters",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument(
        "--feature-set",
        choices=("full", "no_bscore"),
        default="full",
    )
    args = parser.parse_args()
    run(
        Path(__file__).resolve().parents[1],
        scope=args.scope,
        seed=args.seed,
        rebuild_features=args.rebuild_features,
        feature_set=args.feature_set,
    )


if __name__ == "__main__":
    main()

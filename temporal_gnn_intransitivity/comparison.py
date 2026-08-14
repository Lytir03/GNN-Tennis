"""Compare temporal and frozen baselines by causal intransitivity stratum."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


MODELS = (
    "temporal_gnn",
    "temporal_gnn_intransitivity",
    "antisymmetric_no_bscore",
    "antisymmetric_no_node_bscore",
    "gbdt_no_bscore",
    "gbdt_tuned",
)
LEVEL_ORDER = ("no_context_evidence", "low", "medium", "high")
MERGE_KEYS = ("tourney_id", "round_order", "row_in_block")


def metric_values(frame: pd.DataFrame) -> dict[str, float]:
    y = frame["y_true"].to_numpy(dtype=float)
    probability = np.clip(
        frame["probability"].to_numpy(dtype=float), 1e-7, 1 - 1e-7
    )
    return {
        "n_matches": len(frame),
        "accuracy": np.mean((probability >= 0.5) == y),
        "log_loss": -np.mean(
            y * np.log(probability)
            + (1 - y) * np.log(1 - probability)
        ),
        "brier": np.mean((probability - y) ** 2),
    }


def load_comparison(
    project_root: str | Path,
    *,
    scope: str = "slams_masters",
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(project_root)
    payload = pd.read_pickle(
        root
        / "temporal_gnn_intransitivity"
        / "results"
        / scope
        / f"events_cyclic_share_seed_{seed}.pkl"
    )
    levels = payload["frame"]
    levels = levels[levels["phase"] == "test"][
        [*MERGE_KEYS, "intransitivity_score", "intransitivity_level"]
    ].copy()
    if levels.duplicated(list(MERGE_KEYS)).any():
        raise ValueError("Intransitivity merge keys are not unique")

    directory = root / "results" / "frozen_predictions" / scope / f"seed_{seed}"
    rows = []
    predictions = []
    for model in MODELS:
        path = directory / f"{model}.csv"
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        frame["tourney_id"] = frame["tourney_id"].astype(str)
        frame = frame.merge(
            levels,
            on=list(MERGE_KEYS),
            how="left",
            validate="one_to_one",
        )
        if frame["intransitivity_level"].isna().any():
            raise ValueError(f"Could not label every prediction for {model}")
        frame["model"] = model
        predictions.append(frame)
        rows.append({"model": model, "level": "all", **metric_values(frame)})
        for level in LEVEL_ORDER:
            group = frame[frame["intransitivity_level"] == level]
            rows.append(
                {"model": model, "level": level, **metric_values(group)}
            )
    return pd.DataFrame(rows), pd.concat(predictions, ignore_index=True)

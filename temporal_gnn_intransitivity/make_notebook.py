"""Generate the temporal/intransitivity comparison notebook."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "Temporal_GNN_intransitivity_comparison.ipynb"


def cell(kind: str, text: str) -> dict:
    result = {"cell_type": kind, "metadata": {}, "source": text.splitlines(True)}
    if kind == "code":
        result.update({"execution_count": None, "outputs": []})
    return result


cells = [
    cell(
        "markdown",
        """# Temporal GNN e intransitività

Scouting seed 42 su Slam+Masters. L'intransitività è la quota di energia
ciclica `cyclic² / (cyclic² + transitive²)`: non cresce con la quantità di
evidenza. Tutte le feature sono causali e le soglie dei livelli sono
calcolate esclusivamente sul training 2011–2015. La Temporal GNN non usa
B-score.
""",
    ),
    cell(
        "code",
        """from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

START = Path.cwd().resolve()
PROJECT_ROOT = next(
    candidate
    for candidate in (START, *START.parents)
    if (candidate / "temporal_gnn_intransitivity").is_dir()
    and (candidate / "results").is_dir()
)
sys.path.insert(0, str(PROJECT_ROOT))
from temporal_gnn_intransitivity.comparison import (
    LEVEL_ORDER, load_comparison
)

SCOPE = "slams_masters"
SEED = 42
summary, predictions = load_comparison(
    PROJECT_ROOT, scope=SCOPE, seed=SEED
)
""",
    ),
    cell("markdown", "## Risultati complessivi"),
    cell(
        "code",
        """overall = summary[summary["level"] == "all"].sort_values("log_loss")
display(overall.style.format({
    "accuracy": "{:.4f}", "log_loss": "{:.4f}", "brier": "{:.4f}"
}))
""",
    ),
    cell("markdown", "## Risultati per livello di intransitività"),
    cell(
        "code",
        """by_level = summary[summary["level"] != "all"].copy()
for metric in ["accuracy", "log_loss", "brier"]:
    print(metric)
    display(
        by_level.pivot(index="model", columns="level", values=metric)
        .reindex(columns=LEVEL_ORDER)
        .style.format("{:.4f}")
    )
""",
    ),
    cell(
        "markdown",
        """## Guadagno dato dal contesto di intransitività

Differenza `Temporal GNN + intransitività − Temporal GNN base`. Per accuracy
un valore positivo è migliore; per log loss e Brier un valore negativo è
migliore.
""",
    ),
    cell(
        "code",
        """pivot = summary.pivot(index="level", columns="model")
deltas = []
for level in ["all", *LEVEL_ORDER]:
    row = {"level": level}
    for metric in ["accuracy", "log_loss", "brier"]:
        row[f"delta_{metric}"] = (
            pivot.loc[level, (metric, "temporal_gnn_intransitivity")]
            - pivot.loc[level, (metric, "temporal_gnn")]
        )
    deltas.append(row)
display(pd.DataFrame(deltas).style.format({
    "delta_accuracy": "{:+.4f}",
    "delta_log_loss": "{:+.4f}",
    "delta_brier": "{:+.4f}",
}))
""",
    ),
    cell(
        "markdown",
        """## La GNN diventa relativamente migliore nei match high?

Confrontiamo il gap di log loss rispetto a GNN statica e GBDT senza B-score.
Un gap positivo significa che la Temporal GNN è peggiore.
""",
    ),
    cell(
        "code",
        """rows = []
for baseline in ["antisymmetric_no_bscore", "gbdt_no_bscore"]:
    for level in ["all", *LEVEL_ORDER]:
        rows.append({
            "baseline": baseline,
            "level": level,
            "logloss_gap": (
                pivot.loc[
                    level, ("log_loss", "temporal_gnn_intransitivity")
                ]
                - pivot.loc[level, ("log_loss", baseline)]
            ),
            "accuracy_gap": (
                pivot.loc[
                    level, ("accuracy", "temporal_gnn_intransitivity")
                ]
                - pivot.loc[level, ("accuracy", baseline)]
            ),
        })
gaps = pd.DataFrame(rows)
display(gaps.style.format({
    "logloss_gap": "{:+.4f}", "accuracy_gap": "{:+.4f}"
}))
""",
    ),
    cell("markdown", "## Visualizzazione per livello"),
    cell(
        "code",
        """selected = [
    "temporal_gnn",
    "temporal_gnn_intransitivity",
    "antisymmetric_no_bscore",
    "gbdt_no_bscore",
]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for axis, metric in zip(axes, ["accuracy", "log_loss"]):
    table = (
        by_level[by_level["model"].isin(selected)]
        .pivot(index="level", columns="model", values=metric)
        .reindex(LEVEL_ORDER)
    )
    table.plot(ax=axis, marker="o")
    axis.set_title(metric)
    axis.grid(alpha=0.25)
    axis.set_xlabel("intransitivity level")
fig.tight_layout()
plt.show()
""",
    ),
]
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n")
print("Wrote", OUTPUT)

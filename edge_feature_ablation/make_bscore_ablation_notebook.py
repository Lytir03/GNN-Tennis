"""Create the reproducible B-score ablation comparison notebook."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "experiments" / "05_bscore_ablation_comparison.ipynb"


def cell(kind: str, text: str) -> dict:
    value = {"cell_type": kind, "metadata": {}, "source": text.splitlines(True)}
    if kind == "code":
        value.update({"execution_count": None, "outputs": []})
    return value


cells = [
    cell(
        "markdown",
        """# B-score ablation: best GNN and GBDT

Confronto sui cinque seed con split temporale, training e test set invariati.
Le varianti GNN mantengono la stessa architettura; i quattro input B-score
vengono azzerati quando esclusi. Il GBDT senza B-score riusa gli
iperparametri del modello completo, senza un secondo tuning.
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
PROJECT_ROOT = START if (START / "results").is_dir() else START.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "edge_feature_ablation"))
from bscore_ablation_summary import collect

SCOPE = "slams_masters"
SEEDS = [42, 123, 456, 789, 2026]
RESULTS_ROOT = PROJECT_ROOT / "results" / "frozen_predictions"
""",
    ),
    cell(
        "markdown",
        """## Risultati per seed

Accuracy più alta è migliore; log loss e Brier più bassi sono migliori.
""",
    ),
    cell(
        "code",
        """results = collect(RESULTS_ROOT, SCOPE, SEEDS)
expected = 6 * len(SEEDS)
assert len(results) == expected, (
    f"Attesi {expected} artefatti, trovati {len(results)}"
)
display(results.sort_values(["seed", "experiment"]).reset_index(drop=True))
""",
    ),
    cell(
        "markdown",
        """## Media e deviazione standard sui seed""",
    ),
    cell(
        "code",
        """summary = (
    results.groupby("experiment")[["accuracy", "log_loss", "brier"]]
    .agg(["mean", "std"])
    .sort_values(("log_loss", "mean"))
)
display(summary.style.format("{:.6f}"))
""",
    ),
    cell(
        "markdown",
        """## Effetto appaiato della rimozione del B-score

Le differenze sono `modello ablated − modello completo`. Per log loss e
Brier un valore positivo indica un peggioramento; per accuracy un valore
negativo indica un peggioramento. Gli intervalli sono CI t al 95% sui cinque
seed e vanno interpretati con cautela dato il campione ridotto.
""",
    ),
    cell(
        "code",
        """from scipy.stats import t

comparisons = {
    "GNN: no direct vs full": (
        "antisymmetric_no_direct_bscore", "antisymmetric_decoder"
    ),
    "GNN: no node vs full": (
        "antisymmetric_no_node_bscore", "antisymmetric_decoder"
    ),
    "GNN: no B-score vs full": (
        "antisymmetric_no_bscore", "antisymmetric_decoder"
    ),
    "GBDT: no B-score vs full": ("gbdt_no_bscore", "gbdt_tuned"),
    "No B-score: GNN vs GBDT": (
        "antisymmetric_no_bscore", "gbdt_no_bscore"
    ),
}
pivot = results.pivot(index="seed", columns="experiment")
rows = []
for label, (candidate, baseline) in comparisons.items():
    for metric in ["accuracy", "log_loss", "brier"]:
        delta = pivot[metric][candidate] - pivot[metric][baseline]
        sem = delta.std(ddof=1) / np.sqrt(len(delta))
        radius = t.ppf(0.975, len(delta) - 1) * sem
        better = delta > 0 if metric == "accuracy" else delta < 0
        rows.append({
            "comparison": label,
            "metric": metric,
            "mean_delta": delta.mean(),
            "ci95_low": delta.mean() - radius,
            "ci95_high": delta.mean() + radius,
            "wins": int(better.sum()),
            "seeds": len(delta),
        })
paired = pd.DataFrame(rows)
display(paired.style.format({
    "mean_delta": "{:+.6f}",
    "ci95_low": "{:+.6f}",
    "ci95_high": "{:+.6f}",
}))
""",
    ),
    cell(
        "markdown",
        """## Visualizzazione

Ogni punto rappresenta un seed; la linea collega la media dei cinque seed.
""",
    ),
    cell(
        "code",
        """order = [
    "antisymmetric_decoder",
    "antisymmetric_no_direct_bscore",
    "antisymmetric_no_node_bscore",
    "antisymmetric_no_bscore",
    "gbdt_tuned",
    "gbdt_no_bscore",
]
labels = [
    "GNN full", "GNN no direct", "GNN no node", "GNN no B-score",
    "GBDT full", "GBDT no B-score",
]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for axis, metric in zip(axes, ["accuracy", "log_loss", "brier"]):
    for x, experiment in enumerate(order):
        values = results.loc[
            results["experiment"] == experiment, metric
        ].to_numpy()
        axis.scatter(np.full(len(values), x), values, alpha=0.7)
        axis.plot(x, values.mean(), marker="_", markersize=18, color="black")
    axis.set_title(metric)
    axis.set_xticks(range(len(order)), labels, rotation=35, ha="right")
    axis.grid(axis="y", alpha=0.25)
fig.tight_layout()
plt.show()
""",
    ),
    cell(
        "markdown",
        """## Lettura automatica essenziale""",
    ),
    cell(
        "code",
        """gnn_delta = paired[
    paired["comparison"].eq("GNN: no B-score vs full")
].set_index("metric")
gbdt_delta = paired[
    paired["comparison"].eq("GBDT: no B-score vs full")
].set_index("metric")
print(
    "GNN, rimozione completa B-score — delta medio log loss:",
    f"{gnn_delta.loc['log_loss', 'mean_delta']:+.6f}",
    "| vittorie:", int(gnn_delta.loc["log_loss", "wins"]), "/ 5",
)
print(
    "GBDT, rimozione completa B-score — delta medio log loss:",
    f"{gbdt_delta.loc['log_loss', 'mean_delta']:+.6f}",
    "| vittorie:", int(gbdt_delta.loc["log_loss", "wins"]), "/ 5",
)
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
OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print("Wrote", OUTPUT)

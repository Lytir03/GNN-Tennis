"""Generate a compact notebook for GBDT inspection and comparison."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def cell(kind: str, text: str) -> dict:
    result = {
        "cell_type": kind,
        "metadata": {},
        "source": text.strip("\n").splitlines(keepends=True),
    }
    if kind == "code":
        result.update({"execution_count": None, "outputs": []})
    return result


notebook = {
    "cells": [
        cell(
            "markdown",
            """# Tuned GBDT comparison

Confronto multi-seed tra B-score logit, GNN corrente, decoder antisimmetrico e
GBDT tuned. Gli artefatti devono condividere match, target, split e scope.""",
        ),
        cell(
            "code",
            """import os
from pathlib import Path
import sys

START = Path.cwd().resolve()
PROJECT_ROOT = START if (START / "edge_feature_ablation").is_dir() else START.parent
sys.path.insert(0, str(PROJECT_ROOT / "edge_feature_ablation"))
from multi_seed_summary import multi_seed_comparison""",
        ),
        cell(
            "code",
            """TOURNAMENT_SCOPE = os.environ.get(
    "TENNIS_RUN_SCOPE", "slams_masters"
)  # or "slams"
SEEDS = [42, 123, 456, 789, 2026]
RESULTS_ROOT = PROJECT_ROOT / "results" / "frozen_predictions" """,
        ),
        cell(
            "code",
            """per_seed, aggregate = multi_seed_comparison(
    RESULTS_ROOT,
    scope=TOURNAMENT_SCOPE,
    seeds=SEEDS,
    experiments=[
        "bscore_logit",
        "current_gnn",
        "antisymmetric_decoder",
        "gbdt_tuned",
    ],
)
aggregate""",
        ),
        cell("code", "per_seed"),
        cell(
            "code",
            """per_seed.pivot(
    index="seed",
    columns="experiment",
    values=["accuracy", "log_loss", "brier"],
)""",
        ),
        cell(
            "code",
            """pivot = per_seed.pivot(
    index="seed",
    columns="experiment",
    values=["accuracy", "log_loss", "brier"],
)
for baseline in ["bscore_logit", "current_gnn", "antisymmetric_decoder"]:
    print("\\nGBDT minus", baseline)
    display(pivot.xs("gbdt_tuned", level="experiment", axis=1)
            - pivot.xs(baseline, level="experiment", axis=1))""",
        ),
    ],
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
(HERE / "GBDT_comparison.ipynb").write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"
)
print("Wrote", HERE / "GBDT_comparison.ipynb")

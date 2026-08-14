"""Create the lightweight runner/comparison notebooks."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "experiments"
OUTPUT.mkdir(exist_ok=True)


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


def markdown(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip("\n").splitlines(keepends=True),
    }


def write(name: str, cells: list[dict]) -> None:
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
    path = OUTPUT / name
    path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print("Wrote", path)


setup = r'''
import os
from dataclasses import asdict
from pathlib import Path

START_DIRECTORY = Path.cwd().resolve()
if (START_DIRECTORY / "edge_feature_ablation").is_dir():
    PROJECT_ROOT = START_DIRECTORY
elif START_DIRECTORY.name == "experiments":
    PROJECT_ROOT = START_DIRECTORY.parent.parent
else:
    PROJECT_ROOT = START_DIRECTORY.parent

EDGE_DIRECTORY = PROJECT_ROOT / "edge_feature_ablation"
RESULTS_ROOT = PROJECT_ROOT / "results" / "frozen_predictions"
'''

runner = r'''
os.environ["TENNIS_MODEL_EXPERIMENT"] = EXPERIMENT_NAME
if EDGE_ABLATION is None:
    os.environ.pop("TENNIS_EDGE_ABLATION", None)
else:
    os.environ["TENNIS_EDGE_ABLATION"] = EDGE_ABLATION
os.environ["TENNIS_TOURNAMENT_SCOPE"] = TOURNAMENT_SCOPE

previous_directory = Path.cwd()
os.chdir(EDGE_DIRECTORY)
try:
    get_ipython().run_line_magic("run", "GNN_edge_feature_ablation.ipynb")
finally:
    os.chdir(previous_directory)
'''

save_imports = r'''
import sys
sys.path.insert(0, str(EDGE_DIRECTORY))
from experiment_tracking import ArtifactManifest, save_prediction_artifact

OUTPUT_DIRECTORY = (
    RESULTS_ROOT / TOURNAMENT_SCOPE / f"seed_{SEED}"
)
'''

manifest_expression = r'''
def make_manifest(name, family, config):
    return ArtifactManifest(
        experiment=name,
        model_family=family,
        tournament_scope=TOURNAMENT_SCOPE,
        seed=SEED,
        train_end=TRAIN_END,
        validation_end=VAL_END,
        test_end=int(future_matches["tourney_date"].dt.year.max()),
        update_phases=tuple(sorted(UPDATE_PHASES)),
        config=config,
    )
'''

write(
    "00_freeze_baselines.ipynb",
    [
        markdown(
            """# Freeze baselines

Esegue una volta la GNN originale e il B-score logit, quindi salva le
predizioni test match-per-match. Rieseguire solo se cambiano scope, split,
seed o protocollo di training."""
        ),
        code(setup),
        code(
            '''TOURNAMENT_SCOPE = os.environ.get(
    "TENNIS_RUN_SCOPE", "slams_masters"
)  # or "slams"
EXPERIMENT_NAME = "current_gnn"
EDGE_ABLATION = "current"'''
        ),
        code(runner),
        code(save_imports + manifest_expression),
        code(
            r'''
test_gnn = predictions_df[predictions_df["phase"] == "test"].copy()
gnn_paths = save_prediction_artifact(
    test_gnn,
    OUTPUT_DIRECTORY,
    make_manifest(
        "current_gnn",
        "GINE",
        {"model": asdict(MODEL_CONFIG), "edge": asdict(EDGE_CONFIG)},
    ),
)

test_logit = test_predictions.copy()
logit_paths = save_prediction_artifact(
    test_logit,
    OUTPUT_DIRECTORY,
    make_manifest(
        "bscore_logit",
        "LogisticRegression",
        {"features": ["x_bscore_diff_scaled"]},
    ),
    probability_column="bscore_prob",
)

print("Frozen GNN:", gnn_paths)
print("Frozen B-score logit:", logit_paths)
'''
        ),
    ],
)

write(
    "01_edge_feature_ablations.ipynb",
    [
        markdown(
            """# Edge-feature ablations

Cambiare `EDGE_ABLATION` ed eseguire il notebook. Ogni run viene confrontata
successivamente con le baseline congelate."""
        ),
        code(setup),
        code(
            '''TOURNAMENT_SCOPE = os.environ.get(
    "TENNIS_RUN_SCOPE", "slams_masters"
)  # must match the baseline
EDGE_ABLATION = os.environ.get(
    "TENNIS_RUN_EDGE_ABLATION", "signed_game"
)  # current, signed_game, set_margin, straight_sets, match_status, full
EXPERIMENT_NAME = "edge_full"'''
        ),
        code(runner),
        code(save_imports + manifest_expression),
        code(
            r'''
artifact_name = f"edge_{EDGE_ABLATION}"
test_run = predictions_df[predictions_df["phase"] == "test"].copy()
paths = save_prediction_artifact(
    test_run,
    OUTPUT_DIRECTORY,
    make_manifest(
        artifact_name,
        "GINE",
        {"model": asdict(MODEL_CONFIG), "edge": asdict(EDGE_CONFIG)},
    ),
)
print("Saved:", paths)
'''
        ),
    ],
)

write(
    "02_architecture_ablations.ipynb",
    [
        markdown(
            """# Non-temporal architecture ablations

Le configurazioni sono cumulative: termine B-score corretto, decoder
antisimmetrico, normalizzazione, mean aggregation, residual connection e
contesto del torneo. Nessuna rete temporale è inclusa."""
        ),
        code(setup),
        code(
            '''TOURNAMENT_SCOPE = os.environ.get(
    "TENNIS_RUN_SCOPE", "slams_masters"
)  # must match the baseline
EXPERIMENT_NAME = os.environ.get(
    "TENNIS_RUN_EXPERIMENT", "bscore_residual"
)  # see MODEL_EXPERIMENTS
EDGE_ABLATION = None  # use the edge preset declared by MODEL_EXPERIMENTS'''
        ),
        code(runner),
        code(save_imports + manifest_expression),
        code(
            r'''
test_run = predictions_df[predictions_df["phase"] == "test"].copy()
paths = save_prediction_artifact(
    test_run,
    OUTPUT_DIRECTORY,
    make_manifest(
        EXPERIMENT_NAME,
        "GINE",
        {"model": asdict(MODEL_CONFIG), "edge": asdict(EDGE_CONFIG)},
    ),
)
print("Saved:", paths)
'''
        ),
        code(
            '''print("Available experiments:")
for name, configuration in MODEL_EXPERIMENTS.items():
    print(name, asdict(configuration))'''
        ),
    ],
)

write(
    "03_final_comparison.ipynb",
    [
        markdown(
            """# Final frozen comparison

Carica solo artefatti già calcolati. I controlli bloccano confronti con match,
scope, split, seed o protocollo diversi."""
        ),
        code(setup),
        code(
            '''TOURNAMENT_SCOPE = os.environ.get(
    "TENNIS_RUN_SCOPE", "slams_masters"
)
SEED = int(os.environ.get("TENNIS_RUN_SEED", "42"))
OUTPUT_DIRECTORY = RESULTS_ROOT / TOURNAMENT_SCOPE / f"seed_{SEED}"'''
        ),
        code(
            r'''
import sys
sys.path.insert(0, str(EDGE_DIRECTORY))
from experiment_tracking import compare_artifacts

available = sorted(
    path.name.removesuffix(".manifest.json")
    for path in OUTPUT_DIRECTORY.glob("*.manifest.json")
)
print("Available artifacts:", available)
'''
        ),
        code(
            r'''
preferred_order = [
    "bscore_logit",
    "current_gnn",
    "edge_signed_game",
    "edge_set_margin",
    "edge_straight_sets",
    "edge_match_status",
    "edge_full",
    "bscore_residual",
    "antisymmetric_decoder",
    "antisymmetric_no_direct_bscore",
    "antisymmetric_straight_sets",
    "antisymmetric_mean",
    "antisymmetric_tournament_context",
    "antisymmetric_mean_tournament",
    "node_normalization",
    "mean_aggregation",
    "residual_connections",
    "tournament_context",
    "full_non_temporal",
]
experiments = [name for name in preferred_order if name in available]
comparison = compare_artifacts(
    OUTPUT_DIRECTORY,
    experiments,
    phase="test",
)
comparison
'''
        ),
        code(
            r'''
# Positive is better for accuracy; negative is better for the two losses.
comparison.sort_values("log_loss")
'''
        ),
    ],
)

write(
    "04_multi_seed_comparison.ipynb",
    [
        markdown(
            """# Multi-seed robustness

Confronta B-score logit, GNN corrente e decoder antisimmetrico sui cinque
seed congelati. Non esegue alcun training."""
        ),
        code(setup),
        code(
            '''TOURNAMENT_SCOPE = os.environ.get(
    "TENNIS_RUN_SCOPE", "slams_masters"
)
SEEDS = [42, 123, 456, 789, 2026]'''
        ),
        code(
            r'''
import sys
sys.path.insert(0, str(EDGE_DIRECTORY))
from multi_seed_summary import multi_seed_comparison

per_seed, aggregate = multi_seed_comparison(
    RESULTS_ROOT,
    scope=TOURNAMENT_SCOPE,
    seeds=SEEDS,
)
'''
        ),
        code("per_seed"),
        code("aggregate"),
        code(
            r'''
metric_table = per_seed.pivot(
    index="seed",
    columns="experiment",
    values=["accuracy", "log_loss", "brier"],
)
metric_table
'''
        ),
    ],
)

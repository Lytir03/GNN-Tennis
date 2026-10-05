# Repository structure

A rough map of the repository: what each folder is for and how data moves
between them. For per-file detail on the core code see
[`FILE_SUMMARIES.md`](FILE_SUMMARIES.md). For how to run things see
[`README.md`](README.md).

## Data flow

```
data/raw/sackmann/            raw yearly ATP match files (in git)
        │
        │  preprocess/*.ipynb, new_work/build_bscore.py,
        │  extended_history_1990/build_*.py
        ▼
data/processed/               cleaned matches + per-round B-score snapshots
        │                     (gitignored, regenerable)
        │  tennis_gnn/data.py, snapshots.py  (graph per tournament round)
        ▼
.cache/                       built graph snapshots / features (gitignored)
        │
        │  tennis_gnn/run.py, gbdt_comparison/train.py,
        │  gnn_improvements/*.py, extended_history_1990/run_experiment.py
        ▼
results/frozen_predictions/   per-match predictions + manifest, per scope/seed
        │
        │  tennis_gnn/compare.py, new_work/*.py, docs/thesis/make_figures.py
        ▼
result CSVs, figures, FINAL_RESULTS.md, docs/thesis/
```

## Top level

| Path | What it is |
|---|---|
| `README.md` | Setup and how to run everything |
| `STRUCTURE.md` | This file |
| `FINAL_RESULTS.md` | The settled results on one page |
| `FEATURES.md` | Reference for how features are built |
| `BSCORE.md` | Reference for the B-score rating |
| `FILE_SUMMARIES.md` | One paragraph per core source file |
| `environment.yml` | Conda environment (`tennis-gnn`) |
| `elo.ipynb` | Original Elo baseline notebook (`gnn_improvements/baselines.py` has the pipeline version) |

## Code

| Folder | What it is |
|---|---|
| `tennis_gnn/` | The core package: data loading and splits (`data.py`), per-round graph snapshots (`snapshots.py`, `edge_features.py`), per-player history (`history.py`), the model (`model.py`), online training (`train.py`), experiment configs (`config.py`), the runner (`run.py`), tuning (`tune.py`), artifact comparison (`compare.py`), strata analysis (`stratify.py`, `structure.py`), and tests (`test_*.py`) |
| `gbdt_comparison/` | Tuned `HistGradientBoostingClassifier` baseline with the same information as the GNN. `features.py` builds features; `train.py` tunes and fits; `run_multi_seed.py` loops over seeds. Results are in `results/<scope>/` and summarised in `RESULTS.md` |
| `gnn_improvements/` | Experiments that improved the GNN: non-learned baselines (`baselines.py`), B-score conditioning, blending artifacts (`blend.py`), hop grid, retuning, and the graph-window sweep (`window_sweep.py`, which produces the final model). Write-up in `FINDINGS.md`; CSVs in `results/` |
| `new_work/` | Full-tour expansion and the main analyses: B-score rebuild (`build_bscore.py`), feature × hop substitution grid, cold-start strata, depth tests, artifact audit. Write-up in `STATUS.md`; CSVs in `results/` |
| `extended_history_1990/` | A side experiment with a 1980–89 warm-up and training from 1990. Builds its own data and runs the final recipe unchanged |
| `preprocess/` | Notebooks that build `data/processed/` from the raw files |

## Data and results

| Folder | What it is |
|---|---|
| `data/raw/sackmann/` | Jeff Sackmann's ATP match files, 1968–2024 (`all_but_atp/` holds the non-tour-level files) |
| `data/processed/` | Derived datasets; only the `README.md` is in git |
| `results/frozen_predictions/<scope>/seed_<seed>/` | One `.csv` (per-match probabilities and labels) plus `.manifest.json` (config, evaluation hash, metrics) per model. Every comparison in the repo reads from here |
| `results/tuning/` | Validation-only hyperparameter search results |

Scopes: `slams` (Grand Slams), `slams_masters` (+ Masters 1000), `full`
(+ ATP 250/500), and the `_1990` variants with the longer warm-up.

## Docs

| Folder | What it is |
|---|---|
| `docs/thesis/` | One markdown file per thesis chapter, each claim pointing at the code or artifact behind it. `make_figures.py` regenerates every figure (PDF + PNG) from files on disk |

## Gitignored, local only

`data/processed/*` (except its README), `.cache/` (graph snapshots, can be
very large), GBDT feature caches (`features_seed_*.pkl/csv`), `.env` /
virtualenvs, `__pycache__/`, `.pytest_cache/`, `.DS_Store`.

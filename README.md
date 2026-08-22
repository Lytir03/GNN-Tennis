# GNN-Tennis

Predicting ATP match outcomes from a graph of past results, and asking whether
a graph neural network buys anything over a strong tabular baseline.

Players are nodes. Every past match becomes a pair of directed edges carrying
the margin, recency, surface and round. For each tournament round the graph is
rebuilt from matches played *before* that round, the round's matches are
predicted, and only then do those matches enter the graph.

## Layout

| Path | What it is |
|---|---|
| `tennis_gnn/` | The model, data pipeline, training, tuning and comparison code |
| `gbdt_comparison/` | Tuned `HistGradientBoostingClassifier` baseline on the same information |
| `new_work/` | Live/exploratory work-in-progress; see `new_work/STATUS.md` for what's settled vs. still moving |
| `inconclusive_and_superseded/` | Archived results that turned out not to hold up, kept with a README explaining why (includes `temporal_gnn_intransitivity/`, the exploratory recurrent-state variant) |
| `preprocess/` | Notebooks that build the processed datasets and the B-score snapshots |
| `results/frozen_predictions/` | Per-match predictions for every model and seed - the durable record |
| `results/tuning/` | Validation search results |
| `elo.ipynb` | Elo baseline |

## The split

Fixed and never varied: warm-up before 2011, training 2011-2015 (gradient
updates from 2012), validation 2016, test 2017-2020. Parameters are updated on
training blocks only. Validation is used to select hyperparameters and to fit
the calibration temperature; test is scored once.

## Running things

```bash
# One experiment, one seed
conda run -n tennis-gnn python tennis_gnn/run.py --experiments base --seeds 42

# Every one-factor ablation across seeds
conda run -n tennis-gnn python tennis_gnn/run.py \
  --experiments all --seeds 42 123 456 789 2026 \
  --summary results/ablations.csv

# The original untuned recipe, for comparison
conda run -n tennis-gnn python tennis_gnn/run.py \
  --experiments base --seeds 42 --recipe legacy

# Hyperparameter search (validation only)
conda run -n tennis-gnn python tennis_gnn/tune.py --scope slams_masters --seed 42

# Compare frozen artifacts
conda run -n tennis-gnn python tennis_gnn/compare.py \
  --scope slams_masters --seeds 42 123 456 789 2026 \
  --candidate base --baseline gbdt_tuned
```

`--scope slams` restricts everything to Grand Slams. The first run for a given
(scope, edge preset, seed) builds and caches the graph snapshots under
`.cache/`, which takes a few minutes; later runs load them.

Environment note: the runners set `OMP_NUM_THREADS=1` and
`KMP_DUPLICATE_LIB_OK=TRUE` because the pip PyTorch wheel and the conda NumPy
OpenMP runtime collide on macOS. That is an environment workaround, not an
experimental setting.

## Experiments are configuration, not code

An experiment is a `ModelConfig` (what the model is) plus a `TrainConfig` (how
it is fitted), both in `tennis_gnn/config.py`. Ablations vary **one factor at a
time** from `BASE_MODEL`; a test enforces this. An earlier version of this study
defined them cumulatively, which meant one harmful early step contaminated every
later result.

## Tests

```bash
conda run -n tennis-gnn python -m unittest discover -s tennis_gnn -t . -p "test_*.py"
conda run -n tennis-gnn python -m unittest discover -s gbdt_comparison -p "test_*.py"
conda run -n tennis-gnn python -m unittest discover -s inconclusive_and_superseded/temporal_gnn_intransitivity -p "test_*.py"
```

`tennis_gnn/verify_targets.py` checks that the rebuilt evaluation set still
matches the frozen artifacts hash for hash. Run it after touching the data
pipeline: if it fails, predictions are no longer comparable with the published
baselines.

## Open questions

- Does the GNN's advantage really concentrate on intransitive matches? The
  per-match `intransitivity_level` is carried through to the predictions, so
  this is a group-by on the test set rather than a separate training run.
- Is the temporal variant actually worse, or just smaller and untuned?
  See `inconclusive_and_superseded/temporal_gnn_intransitivity/README.md`.

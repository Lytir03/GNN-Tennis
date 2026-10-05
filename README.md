# GNN-Tennis

Predicting ATP match outcomes from a graph of past results, and testing whether
a graph neural network adds anything over a strong tabular baseline.

Players are nodes. Every past match becomes a pair of directed edges carrying
margin, recency, surface and round. For each tournament round the graph is
rebuilt from matches played *before* that round, the round is predicted, and
only then do those matches enter the graph.

- Results: [`FINAL_RESULTS.md`](FINAL_RESULTS.md)
- Repository layout: [`STRUCTURE.md`](STRUCTURE.md)
- Thesis chapters: [`docs/thesis/`](docs/thesis/README.md)

## 1. Set up the environment

```bash
conda env create -f environment.yml
conda activate tennis-gnn
```

PyTorch and PyTorch Geometric are installed through pip (see `environment.yml`).
On macOS the pip PyTorch wheel and conda's NumPy each ship an OpenMP runtime;
the runners set `OMP_NUM_THREADS=1` and `KMP_DUPLICATE_LIB_OK=TRUE` themselves.
If you import `torch` in your own scripts or notebooks, export the same two
variables first.

All commands below are run from the repository root.

## 2. Build the processed data

Only the raw Sackmann files (`data/raw/sackmann/`) are in git. Everything under
`data/processed/` is derived and gitignored, so a fresh clone has to rebuild it
(see [`data/processed/README.md`](data/processed/README.md) for which file
comes from where).

1. Run the notebooks in `preprocess/` in this order:
   `data_prep.ipynb` → `data_prep_slams_only.ipynb` → `graph.ipynb`
   (+ `graph_hard1.ipynb`, `graph_clay1.ipynb`, `graph_grass1.ipynb`) →
   `intransitivity_calc.ipynb`.
   This produces the `slams` and `slams_masters` scopes.
2. For the full-tour scope, rebuild B-score snapshots with:
   ```bash
   python new_work/build_bscore.py --verify
   ```
   ```bash
   python new_work/build_bscore.py --scope full
   ```
   `--verify` must reproduce the published snapshots exactly before the scope is
   widened. Note that no script in the repo currently writes
   `data/processed/atp_matches_full.csv` itself.
3. Optional, for the 1980–89 warm-up experiment:
   ```bash
   python extended_history_1990/build_matches.py
   ```
   ```bash
   python extended_history_1990/build_bscore.py
   ```

The first training run for a given (scope, edge preset, seed) builds graph
snapshots into `.cache/`. That takes a few minutes; later runs load them.

## 3. Run experiments

Scopes: `slams`, `slams_masters` (default), `full`, `slams_masters_1990`,
`full_1990`. Seeds used throughout: `42 123 456 789 2026`.

**The final model** (tier-3 features routed to the decoder, 365-day window,
one hop, sum aggregation):

```bash
python gnn_improvements/window_sweep.py --scope full --windows 365 --tiers t3_history_decoder --hops 1
```

**Base model and one-factor ablations** (`tennis_gnn/config.py`):

```bash
python tennis_gnn/run.py --experiments base --seeds 42
```
```bash
python tennis_gnn/run.py --experiments all --seeds 42 123 456 789 2026 --summary results/ablations.csv
```

**Tuned GBDT baseline:**

```bash
python gbdt_comparison/run_multi_seed.py --scope full --seeds 42 123 456 789 2026
```

**Non-learned baselines (B-score, Elo):**

```bash
python gnn_improvements/baselines.py --scope full
```

**Hyperparameter search** (validation only):

```bash
python tennis_gnn/tune.py --scope slams_masters --seed 42
```

Every run writes per-match predictions to
`results/frozen_predictions/<scope>/seed_<seed>/<name>.csv` plus a
`.manifest.json`. Those artifacts are the durable record; everything else is
computed from them.

## 4. Compare models

`compare.py` reads frozen artifacts and computes paired, per-seed contrasts. It
checks that both sides were evaluated on identical matches and labels first.

```bash
python tennis_gnn/compare.py --scope full --seeds 42 123 456 789 2026 --candidate win365_t3_history_decoder_1hop_sum --baseline gbdt_tuned
```

Analyses that read artifacts and retrain nothing:

```bash
python new_work/feature_hop_grid.py --scope full
```
```bash
python new_work/cold_start.py --scope full
```
```bash
python docs/thesis/make_figures.py
```

## 5. Tests

```bash
python -m unittest discover -s tennis_gnn -t . -p "test_*.py"
```
```bash
python -m unittest discover -s gbdt_comparison -p "test_*.py"
```

After touching the data pipeline, run `python tennis_gnn/verify_targets.py`.
It checks that the rebuilt evaluation set still matches the frozen artifacts
hash for hash. If it fails, new predictions are no longer comparable with the
published ones.

## The split

Fixed per scope and never varied. Weights update on training blocks only;
validation selects hyperparameters and fits the calibration temperature; test
is scored once.

| Scope | Warm-up | Train | Validation | Test |
|---|---|---|---|---|
| `slams_masters` | 2006–10 | 2011–15 | 2016 | 2017–20 |
| `full` | 2006–10 | 2011–16 | 2017–18 | 2019–24 |
| `slams_masters_1990` | 1980–89 | 1990–2011 | 2012–13 | 2014–20 |

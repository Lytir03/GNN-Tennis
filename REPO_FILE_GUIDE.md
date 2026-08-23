# Repo file guide

Folder by folder, what each file is for and why it exists as its own file.
Bigger/more complex files get more detail; small ones stay brief. This is a
navigation aid, not a substitute for `README.md` (orientation) or
`TUTOR_REPORT.md` (results/narrative).

---

## `tennis_gnn/` — core pipeline

The most complex directory; every module here is a load-bearing piece of the
experimental infrastructure, not a convenience script.

- **`__init__.py`** (19 lines) — re-exports the config surface (`ModelConfig`,
  `TrainConfig`, `BASE_MODEL`, `LEGACY_MODEL`, `one_factor_ablations`) so
  callers do `from tennis_gnn import X` instead of reaching into
  `tennis_gnn.config`. Standalone only as the package's public API boundary.

- **`config.py`** (205 lines) — the single source of truth for what an
  experiment *is*. Two frozen dataclasses: `ModelConfig` (architecture/
  information flags — the knobs an ablation turns) and `TrainConfig`
  (optimiser/schedule — the knobs a validation search turns). Deliberately
  separated: an architecture comparison must hold the training recipe fixed,
  and a tuning run must hold the architecture fixed, so mixing them into one
  dataclass would make that impossible to enforce. Defines `BASE_MODEL` (the
  reference config) and `LEGACY_MODEL` (reproduces the original pre-refactor
  artifact, kept for regression-checking). `one_factor_ablations()` returns
  ~19 named configs, each differing from `BASE_MODEL` in exactly one field —
  replacing an earlier *cumulative* ablation design where each entry
  inherited every change above it, silently contaminating every later
  measurement. Every flag carries an inline comment explaining the
  experimental reasoning (e.g. `disable_message_passing` vs `num_layers=0`,
  which also deletes LayerNorms and so isn't a clean control). Standalone
  because it's imported by nearly every other module and must have zero
  other dependencies to avoid import cycles.

- **`data.py`** (260 lines) — loads the match CSV and applies the fixed
  temporal split (`Split` class, `phase_for_year`). Replaces duplicated
  boilerplate that used to live at the top of every notebook. `Dataset`
  wraps the loaded frame with `block_matches`/`bscore_snapshot` accessors;
  `load_dataset()` is the main entry point. Standalone because it's the one
  place the split boundaries and the random orientation draw (which player
  is "A") are defined — every artifact's comparability depends on this logic
  being identical everywhere it's used.

- **`edge_features.py`** (400 lines) — parses raw match scores into
  directional, sign-corrected edge features (score is recorded winner-first
  in the source data; this flips it per direction). `EdgeFeatureConfig`
  selects a named preset (`full`, `signed_game`, `set_margin`, etc.);
  `build_bidirectional_edges` is the main builder. Substantial and
  standalone because score parsing is fiddly (retirements, incomplete sets,
  walkovers) and this is the only place it happens — both `snapshots.py` and
  the temporal model depend on `parse_match_score`.

- **`history.py`** (203 lines) — recency-weighted per-player match history
  (12 statistics: result/game/set margins, straight-sets rate, completion
  rate, per-surface and overall). Explicitly pulled out of
  `gbdt_comparison/features.py` where it used to live, because the GBDT had
  these features and the GNN didn't — an information asymmetry disguised as
  an architecture comparison. `HistoryTracker` maintains rolling per-player
  state; `aggregate_history` computes the decayed stats. Shared by both
  model families now so a definitional drift between two hand-written copies
  can't silently become a "model difference."

- **`model.py`** (196 lines) — the single `TennisGNN` class (GINE/GATv2 conv
  layers, antisymmetric pair decoder). Previously copy-pasted with small
  hand differences into eight notebooks; now one implementation driven
  entirely by `ModelConfig` flags. Standalone as the only place architecture
  is defined.

- **`snapshots.py`** (458 lines, largest non-test file) — builds and caches
  per-block graph snapshots (`BlockSnapshot`, `BlockGraph`, `build_graphs`,
  `attach_targets`, `load_or_build_graphs`). A snapshot depends only on data
  + edge preset, never on the model, so building it once and caching to
  `.cache/` turns a hyperparameter search from hours into minutes — the
  original notebook rebuilt graphs inside the training loop. Complex because
  it owns node-feature construction, target attachment (including the
  random A/B orientation draw), and the on-disk cache key/versioning.

- **`structure.py`** (123 lines) — computes graph-structural descriptors per
  match (degree, common-opponent count) used to test *where* a graph model
  could plausibly beat a tabular one, since the GBDT already gets one-hop
  history aggregates and has no head-to-head or common-opponent feature.
  `block_structure`, `structure_table`, `add_strata`.

- **`stratify.py`** (174 lines) — paired stratified comparison of two models
  across the structural buckets `structure.py` defines. Answers "is the
  graph buying anything, and where." Bakes in two explicit warnings against
  self-deception: multiple-comparisons inflation (prints the comparison
  count) and confounding (structural descriptors correlate with data
  volume). `paired_frame`, `stratum_table`, `interaction_test`.

- **`train.py`** (431 lines, largest module) — the training/eval loop:
  block-by-block graph construction → training-only gradient updates
  (2012–2015) → frozen-model prediction on validation/test. `run_experiment`
  is the main entry; `fit_temperature` does validation-only calibration;
  `run_ensemble` averages multiple seeds; `metrics`/`metrics_by_phase` score
  predictions. Large because it's the actual optimisation loop plus
  calibration plus evaluation, none of which factor cleanly apart.

- **`run.py`** (244 lines) — CLI that runs named experiments (from
  `one_factor_ablations()`) and writes frozen prediction artifacts. Replaces
  a previous approach that string-substituted cells into per-ablation
  generated notebooks — fragile and hard to diff. `run_named` does the
  actual dispatch; `main` is the CLI wrapper.

- **`tune.py`** (242 lines) — validation-only hyperparameter search for the
  GNN (`focused_space`, `search_space`, CLI in `main`). Exists because the
  GBDT baseline got 32 tuned configs and the GNN originally got none — the
  comparison had been baseline-tuned vs. untuned by omission.

- **`compare.py`** (240 lines) — compares frozen artifacts across
  models/seeds with paired significance testing (`per_seed_metrics`,
  `aggregate`, `paired_delta`, `t_critical`). Replaces three separate
  summary scripts/notebooks that each reimplemented a slightly different
  version of the same table. Every comparison is checked to cover the exact
  same matches/labels before any metric is computed — the check that the
  temporal-model comparison silently failed.

- **`verify_targets.py`** (101 lines) — single-purpose script: rebuild
  targets from the pipeline and assert they reproduce the frozen
  `evaluation_hash` exactly. Run after touching the data pipeline; fails
  loudly rather than letting predictions silently drift out of
  comparability.

- **`experiment_tracking.py`** (198 lines) — the artifact persistence layer:
  `ArtifactManifest`, `evaluation_hash` (hash over match keys + labels — the
  invariant everything else in the repo relies on),
  `save_prediction_artifact`/`load_prediction_artifact`,
  `assert_compatible`, `compare_artifacts`. Standalone as the lowest-level
  dependency of both `compare.py` and every audit/repair script in
  `new_work/`.

- **Tests** — `test_tennis_gnn.py` (571 lines, main pipeline),
  `test_edge_features.py` (135 lines, score parsing),
  `test_experiment_tracking.py` (101 lines, artifact hashing/compatibility).
  Kept separate per module under test rather than merged into one file,
  since each targets a different contract.

---

## `gbdt_comparison/` — tabular baseline

- **`__init__.py`** (2 lines) — package docstring only.
- **`features.py`** (378 lines) — builds leakage-safe tabular features from
  the *same* underlying information the GNN gets (`BScoreSnapshots`,
  `build_static_player_table`, `build_feature_dataset`,
  `model_feature_columns`). This is the feature-parity counterpart to
  `tennis_gnn/history.py`/`edge_features.py` — kept separate because it's a
  different model family's feature contract, even though some logic
  (surface one-hot, phase-for-year) is intentionally near-identical for
  comparability.
- **`train.py`** (261 lines) — tunes and evaluates
  `HistGradientBoostingClassifier` (`parameter_grid`, `run`, `main`).
  Standalone as the GBDT-specific training/eval loop, structurally parallel
  to `tennis_gnn/train.py`.
- **`run_multi_seed.py`** (65 lines) — thin CLI wrapper that loops
  `train.py` over seeds, skipping already-completed artifacts. Small,
  standalone only as a convenience script.
- **`test_bscore_ablation.py`** (34 lines) / **`test_features.py`**
  (62 lines) — small unit tests for feature-column selection and history
  aggregation, split by what they test.
- **`README.md`** / **`RESULTS.md`** — usage notes and the baseline's result
  numbers respectively.

---

## `new_work/` — live/exploratory scripts

Each script here corresponds to one specific investigative step or bug-fix
in the `STATUS.md` narrative; merging them would erase the "what was run,
in what order, to answer what question" structure the whole directory is
organized around.

- **`STATUS.md`** (519 lines) — the actual running lab notebook, not a
  README. Both original tasks (fill the feature×hop grid; expand to
  full-tour data) are marked done, and it documents two retractions found
  along the way (a broken temperature-fit bug, and a mistuned learning rate
  that had been hiding the graph's real contribution) plus the corrected
  headline result. Read this in full if you need the live state of the
  project, not this doc.
- **`build_bscore.py`** (314 lines) — replaces four near-identical
  preprocessing notebooks (`graph.ipynb` + 3 surface variants) with one
  parameterised script. Deliberately reproduces the old logic bit-for-bit
  rather than improving it, verified via `verify()` against the published
  snapshots before being trusted to run on the expanded scope — changing
  scope and logic in the same step would make any downstream difference
  unattributable.
- **`feature_hop_grid.py`** (237 lines) — runs the central experiment:
  feature-richness tier × hop-count grid, testing whether graph structure
  substitutes for per-player history. `main` is the driver.
- **`cold_start.py`** (264 lines) — Stage 5: the headline GBDT-vs-GNN
  comparison plus stratified cold-start/intransitivity analysis
  (`hop_pairs`, `build_structure`, `main`).
- **`depth_test.py`** (111 lines) — isolated 2-hop-vs-1-hop test at the
  richest feature tier, written to correct a recipe artifact that had
  inflated the apparent depth penalty.
- **`twohop_diagnostic.py`** (94 lines) — a smaller, two-seed diagnostic
  proving the 2-hop penalty was mostly a learning-rate artifact, not a depth
  effect — motivated `depth_test.py`'s corrected rerun.
- **`audit_artifacts.py`** (123 lines) — sweeps every frozen artifact for
  anomaly signatures (degenerate probabilities, extreme temperature) that
  would indicate a broken run; written after one such bug sat undetected for
  weeks.
- **`expand_data.py`** (137 lines) — coverage-check/regression-test script
  for the full-tour expansion; deliberately does *not* expand anything
  itself, only verifies it's safe to.
- **`results/`** — CSV outputs of the above scripts, with its own
  `README.md`.

`recalibrate.py` and `repair_manifests.py` (one-time migration scripts that
fixed 18 mis-calibrated artifacts and stale manifest metadata respectively)
were removed after their fix was fully applied — see
`new_work/results/README.md` for the note on what they did and why they're
gone.

---

## `inconclusive_and_superseded/`

- **`README.md`** — explains, per archived item, exactly why its result
  doesn't support its original claim.
- **`REPO_REVIEW.md`** — the original first-pass repo review, kept for
  history; superseded by `TUTOR_REPORT.md`.
- **`temporal_gnn_intransitivity/`** — the parked recurrent-state model
  variant:
  - `data.py` (281 lines) — builds a chronological, block-safe event stream
    (different shape from `tennis_gnn/data.py` because the temporal model
    consumes a sequence, not per-block graphs).
  - `model.py` (87 lines) — `TemporalTennisGNN`, a compact recurrent
    interaction model.
  - `train.py` (396 lines) — its own train/eval loop, structurally parallel
    to `tennis_gnn/train.py` but for a stateful recurrent model rather than
    block-rebuilt graphs.
  - `run_experiments.py` (73 lines) — CLI to run the two variants across
    seeds.
  - `test_temporal.py` (58 lines) — unit tests, including one asserting the
    decoder is exactly antisymmetric.
  - `README.md` — explains its own unresolved status.

---

## `preprocess/` — original data-prep notebooks

- **`data_prep.ipynb`** / **`data_prep_slams_only.ipynb`** (7 cells each) —
  filter raw Sackmann data to the study's tournament scope (Slams +
  Masters), producing `atp_matches.csv` and `atp_matches_slams_masters.csv`
  respectively. Near-identical, differing only in the filter.
- **`graph.ipynb`** / **`graph_clay1.ipynb`** / **`graph_grass1.ipynb`** /
  **`graph_hard1.ipynb`** (27–28 cells each) — build the B-score graph
  snapshots, one per surface plus a combined version. These four are the
  ones `new_work/build_bscore.py` replaces with one parameterised script —
  kept here as the original, unmodified source of truth the replacement was
  verified against.
- **`intransitivity_calc.ipynb`** (15 cells) — computes per-match
  intransitivity scores/levels and the intransitivity-split CSVs.

---

## Root files

- **`README.md`** — top-level orientation (layout table, split definition,
  how to run things).
- **`TUTOR_REPORT.md`** — the full narrative write-up/results report; the
  primary deliverable.
- **`elo.ipynb`** (4 cells) — standalone Elo-rating baseline, independent of
  the GNN/GBDT pipeline entirely (reads `atp_matches.csv` directly, no
  shared code).
- **`environment.yml`** — conda env spec (Python 3.11 + pandas/numpy/
  networkx/sklearn/etc.); the reproducibility entry point.
- **`.gitignore`** — excludes OS/Python/Jupyter junk plus specific large
  regenerable caches (feature pickles, full-scope B-score snapshots,
  `.cache/`).
- **`.gitattributes`** — one line, LF normalization for text files.

---

## `results/` and `data/`

- **`results/frozen_predictions/{full,slams,slams_masters}/seed_*/`** —
  per-scope, per-seed prediction CSVs + manifests for every named
  experiment; the durable comparison record `compare.py` reads.
- **`results/tuning/`** — validation search logs/CSVs.
- **`data/raw/sackmann/`** — untouched yearly source CSVs.
- **`data/processed/`** — derived datasets; indexed by
  `data/processed/README.md`.

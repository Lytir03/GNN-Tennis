# File summaries

One paragraph per file: what it does (3 sentences), then why it exists as its
own file (1-2 sentences). Covers the core code only — `tennis_gnn/`,
`gbdt_comparison/`, `new_work/`. For the folder-level map see `STRUCTURE.md`;
for results see `FINAL_RESULTS.md`.

---

## `tennis_gnn/`

### `__init__.py`
Re-exports the package's public config surface: `ModelConfig`, `TrainConfig`,
`BASE_MODEL`, `LEGACY_MODEL`, and `one_factor_ablations`. It does nothing
beyond forwarding names from `config.py`. It has no logic of its own — it's
just the entry point callers import from.
It exists so other code can write `from tennis_gnn import X` instead of
reaching into `tennis_gnn.config` directly.

### `config.py`
Defines the two frozen dataclasses that specify what an experiment *is*:
`ModelConfig` for architecture/information flags and `TrainConfig` for
optimiser/schedule settings. It defines `BASE_MODEL` (the reference config),
`LEGACY_MODEL` (reproduces the pre-refactor artifact), and
`one_factor_ablations()`, which returns ~19 configs each differing from
`BASE_MODEL` in exactly one field. It has zero dependencies on the rest of
the package.
It's kept separate and dependency-free because nearly every other module
imports it, and separating architecture from training knobs is what lets an
ablation and a tuning run each hold the other fixed.

### `data.py`
Loads the match CSV and applies the fixed temporal train/val/test split via
the `Split` class and `phase_for_year`. `Dataset` wraps the loaded frame with
`block_matches`/`bscore_snapshot` accessors, and `load_dataset()` is the main
entry point. It also owns the random draw for which player is labeled "A".
It exists as one file because the split boundaries and orientation logic
must be identical everywhere they're used, or artifacts stop being
comparable.

### `edge_features.py`
Parses raw match scores into directional, sign-corrected edge features,
flipping the winner-first source data per direction. `EdgeFeatureConfig`
selects a named preset (`full`, `signed_game`, `set_margin`, etc.), and
`build_bidirectional_edges` is the main builder. It's the only place score
parsing (retirements, incomplete sets, walkovers) happens.
It's standalone and sizeable because that parsing is fiddly and
`snapshots.py` depends on its `parse_match_score` function.

### `history.py`
Computes recency-weighted per-player match history: 12 statistics covering
result/game/set margins, straight-sets rate, completion rate, and per-surface
splits. `HistoryTracker` maintains rolling per-player state, and
`aggregate_history` computes the decayed stats. It's shared by both the GNN
and GBDT model families.
It was pulled out of `gbdt_comparison/features.py`, where it used to live
only for the GBDT, because that gave the GBDT information the GNN lacked and
disguised an information asymmetry as an architecture comparison.

### `model.py`
Defines the single `TennisGNN` class, using GINE/GATv2 convolution layers and
an antisymmetric pair decoder. Its behavior is driven entirely by
`ModelConfig` flags rather than hardcoded choices. It contains no training or
data logic, only the architecture.
It's the one place the architecture is defined, replacing a version of the
model that used to be copy-pasted with small hand differences into eight
separate notebooks.

### `snapshots.py`
Builds and caches per-block graph snapshots via `BlockSnapshot`,
`BlockGraph`, `build_graphs`, `attach_targets`, and `load_or_build_graphs`.
It owns node-feature construction, target attachment (including the random
A/B orientation draw), and the on-disk cache key/versioning. A snapshot
depends only on data and edge preset, never on the model.
It's the largest non-test module because caching graph-building separately
from training turns a hyperparameter search from hours into minutes — the
original notebook rebuilt graphs inside the training loop every time.

### `structure.py`
Computes graph-structural descriptors per match — degree and common-opponent
count — via `block_structure`, `structure_table`, and `add_strata`. These
descriptors bucket matches by how much graph-only information is available
beyond what a tabular model already sees. It has no training logic itself.
It exists to test *where* a graph model could plausibly beat a tabular one,
since the GBDT baseline already gets one-hop history but no head-to-head or
common-opponent feature.

### `stratify.py`
Runs paired stratified comparison of two models across the structural
buckets `structure.py` defines, via `paired_frame`, `stratum_table`, and
`interaction_test`. It answers whether the graph is buying anything, and in
which structural regime. It explicitly warns about multiple-comparisons
inflation (printing the comparison count) and about confounding, since
structural descriptors correlate with data volume.
It's kept separate from `structure.py` because computing descriptors and
statistically comparing models against them are different jobs with
different failure modes to guard against.

### `train.py`
Implements the training/eval loop: block-by-block graph construction,
training-only gradient updates over 2012-2015, then frozen-model prediction
on validation/test. `run_experiment` is the main entry, `fit_temperature`
does validation-only calibration, `run_ensemble` averages multiple seeds, and
`metrics`/`metrics_by_phase` score predictions. It's the largest module in
the package.
It's large because the optimisation loop, calibration, and evaluation don't
factor cleanly apart — they share state across the same temporal blocks.

### `run.py`
CLI that runs named experiments drawn from `one_factor_ablations()` and
writes frozen prediction artifacts to disk. `run_named` does the actual
dispatch and `main` is the CLI wrapper. It has no training logic of its own —
it drives `train.py`.
It replaces a previous approach that string-substituted cells into
per-ablation generated notebooks, which was fragile and hard to diff.

### `tune.py`
Runs a validation-only hyperparameter search for the GNN, defined via
`focused_space`/`search_space` and exposed through a CLI in `main`. It never
touches the test set. It's structurally separate from `train.py`'s
single-config training loop.
It exists because the GBDT baseline originally got 32 tuned configs while
the GNN got none — the comparison had been baseline-tuned vs. untuned by
omission until this was added.

### `compare.py`
Compares frozen prediction artifacts across models and seeds with paired
significance testing, via `per_seed_metrics`, `aggregate`, `paired_delta`,
and `t_critical`. Every comparison first checks that both artifacts cover
the exact same matches and labels before computing any metric. It's the
final step that turns saved predictions into a comparison table.
It replaces three separate summary scripts/notebooks that each reimplemented
a slightly different version of the same table, and the same-evaluation-set
check prevents silently comparing artifacts scored on different data.

### `verify_targets.py`
A single-purpose script that rebuilds targets from the pipeline and asserts
they reproduce the frozen `evaluation_hash` exactly. It does nothing else —
no training, no comparison. It's meant to be run after any change to the
data pipeline.
It fails loudly on mismatch rather than letting predictions silently drift
out of comparability with previously frozen artifacts.

### `experiment_tracking.py`
Implements the artifact persistence layer: `ArtifactManifest`,
`evaluation_hash` (a hash over match keys and labels), and
`save_prediction_artifact`/`load_prediction_artifact`, plus
`assert_compatible` and `compare_artifacts`. `evaluation_hash` is the
invariant every downstream comparison relies on. It has no model-specific
logic.
It's standalone as the lowest-level dependency of both `compare.py` and
every audit/repair script in `new_work/`.

### `test_tennis_gnn.py`, `test_edge_features.py`, `test_experiment_tracking.py`
Unit/integration tests for, respectively, the main training pipeline, the
score-parsing logic in `edge_features.py`, and artifact hashing/compatibility
in `experiment_tracking.py`. Each targets one module's contract rather than
testing the package as a whole. Together they're the regression net for
every other file in this list.
They're kept as separate files rather than one test suite because each one
targets a different module's contract and failure mode.

---

## `gbdt_comparison/`

### `__init__.py`
Contains only a package docstring, no logic or exports. It marks the
directory as an importable Python package. It has no other content.
It exists purely so `gbdt_comparison` can be imported as a package.

### `features.py`
Builds leakage-safe tabular features for the GBDT model from the same
underlying information the GNN gets, via `BScoreSnapshots`,
`build_static_player_table`, `build_feature_dataset`, and
`model_feature_columns`. Some logic, like surface one-hot encoding and
phase-for-year, is intentionally near-identical to the GNN side for
comparability. It has no training logic itself.
It's the feature-parity counterpart to `tennis_gnn/history.py` and
`edge_features.py`, kept separate because it defines a different model
family's feature contract even where the underlying logic overlaps.

### `train.py`
Tunes and evaluates a `HistGradientBoostingClassifier` via
`parameter_grid`, `run`, and `main`. It mirrors the shape of
`tennis_gnn/train.py` but for the tabular model. It writes frozen prediction
artifacts the same way the GNN pipeline does.
It's the GBDT-specific training/eval loop, kept structurally parallel to
`tennis_gnn/train.py` so the two model families stay directly comparable.

### `run_multi_seed.py`
A thin CLI wrapper that loops `train.py` over multiple seeds, skipping any
artifacts that already exist. It contains no modeling logic of its own. It's
the smallest file in the directory.
It's a standalone convenience script only, kept separate so `train.py` stays
focused on a single run.

### `test_bscore_ablation.py` / `test_features.py`
Small unit tests: the first checks feature-column selection under a
B-score ablation, the second checks history aggregation logic in
`features.py`. Both are short and target a narrow slice of behavior. Neither
tests the training loop itself.
They're split by what they test, following the same one-file-per-contract
pattern as the `tennis_gnn/` tests.

### `README.md` / `RESULTS.md`
`README.md` documents how to run the GBDT pipeline; `RESULTS.md` records its
result numbers. Neither contains code. They're the usage and results
reference for this directory specifically.
They're kept local to `gbdt_comparison/` rather than merged into the
top-level docs because they're specific to the tabular baseline.

---

## `new_work/`

Each script here corresponds to one specific investigative step or bug-fix
described in `STATUS.md`; they're kept as separate scripts rather than
merged because the directory's structure is meant to preserve "what was run,
in what order, to answer what question."

### `STATUS.md`
The running lab notebook for this directory, documenting both completed
original tasks (filling the feature×hop grid, expanding to full-tour data)
and two retractions found along the way (a broken temperature-fit bug and a
mistuned learning rate). It records the corrected headline result. It is not
a README and isn't meant to be a polished summary.
It's the authoritative live-state document for this project; read it in
full for current status rather than relying on this file.

### `build_bscore.py`
Replaces four near-identical preprocessing notebooks (`graph.ipynb` plus
three surface variants) with one parameterised script. It deliberately
reproduces the old notebook logic bit-for-bit rather than improving it, and
was verified via its own `verify()` function against the published
snapshots before being trusted on expanded data.
Keeping logic and scope changes separate (reproduce first, expand later) is
what makes any downstream difference attributable to one or the other.

### `feature_hop_grid.py`
Runs the central experiment: a feature-richness tier × hop-count grid
testing whether graph structure can substitute for per-player history.
`main` is the driver function. It writes results consumed by later analysis
scripts.
It's the script that produces the grid `STATUS.md`'s narrative is built
around.

### `cold_start.py`
Runs Stage 5: the headline GBDT-vs-GNN comparison plus a stratified
cold-start/intransitivity analysis, via `hop_pairs`, `build_structure`, and
`main`. It's the script that produces the paper's main comparison numbers.
It depends on artifacts already produced by `train.py` in both model
directories.
It's kept as its own script because it's a distinct analysis stage from
the feature/hop grid, run after that grid is complete.

### `depth_test.py`
An isolated 2-hop-vs-1-hop test run at the richest feature tier. It was
written specifically to correct a recipe artifact that had inflated the
apparent depth penalty in an earlier run. It's small and single-purpose.
It exists to produce a corrected, targeted rerun after
`twohop_diagnostic.py` identified the earlier result as an artifact, not a
real depth effect.

### `twohop_diagnostic.py`
A smaller, two-seed diagnostic script that shows the apparent 2-hop penalty
was mostly a learning-rate artifact rather than a genuine depth effect. It's
intentionally lightweight compared to a full experiment. It doesn't produce
a headline result itself.
It exists as the diagnostic that motivated `depth_test.py`'s corrected
rerun, and is kept separate as the record of that investigation.

### `audit_artifacts.py`
Sweeps every frozen prediction artifact for anomaly signatures, such as
degenerate probabilities or extreme calibration temperatures, that would
indicate a broken run. It reports findings rather than fixing anything. It
runs independently of any single experiment.
It was written after one such bug sat undetected in a frozen artifact for
weeks, so it exists to catch that failure mode going forward.

### `expand_data.py`
A coverage-check/regression-test script for the full-tour data expansion.
It deliberately does not expand any data itself — it only verifies that
doing so would be safe. It's meant to be run before, not during, an actual
expansion.
It's kept separate from the expansion logic itself so the safety check
can't be skipped by accident.

### `results/README.md`
Indexes the CSV outputs the other `new_work/` scripts produce, and notes
that `recalibrate.py` and `repair_manifests.py` (one-time migration scripts
that fixed mis-calibrated artifacts and stale manifest metadata) were
removed after their fixes were fully applied. It contains no code itself.
It exists so the removed scripts' purpose is documented even though the
scripts themselves are gone.

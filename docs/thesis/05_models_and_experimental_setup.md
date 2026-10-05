# Chapter 5 — Models and Experimental Setup

What is fitted, how it is fitted, and what makes two runs comparable. Every
subsection below is one bullet of the chapter outline.

Primary sources: [`tennis_gnn/model.py`](../../tennis_gnn/model.py),
[`tennis_gnn/config.py`](../../tennis_gnn/config.py),
[`tennis_gnn/train.py`](../../tennis_gnn/train.py),
[`tennis_gnn/tune.py`](../../tennis_gnn/tune.py),
[`tennis_gnn/compare.py`](../../tennis_gnn/compare.py),
[`gbdt_comparison/train.py`](../../gbdt_comparison/train.py).

---

## 5.0 The one rule that governs the chapter

An experiment in this repository is **not code**. It is a `ModelConfig` (what
the model is) plus a `TrainConfig` (how it is fitted), both declared in
`tennis_gnn/config.py`. Ablations are produced by `one_factor_ablations()`,
each entry differing from `BASE_MODEL` in **exactly one field**, and a unit test
enforces that.

This is a correction, not a style choice. An earlier version of the study
defined ablations *cumulatively*: every row inherited every change above it, one
early step (`node_normalization`) was badly harmful, and so no row measured the
factor named in its title. Those cumulative artifacts (`mean_aggregation`,
`node_normalization`, `residual_connections`, `tournament_context`,
`antisymmetric_*`) remain in `results/frozen_predictions/` but carry no evidence.

---

## 5.1 GBDT baseline

`gbdt_comparison/train.py`, a `HistGradientBoostingClassifier` on the same
information in flat tabular form.

**Features** (`gbdt_comparison/features.py`, documented in
[`FEATURES.md`](../../FEATURES.md) §7): per player, the general and
surface-matched B-score, height, handedness, and the twelve recency-weighted
history statistics (`history_all_*`, `history_surface_*`). `add_pair_features()`
then expands every one of them into four columns — `_a`, `_b`, `_diff`,
`_abs_diff` — so the tree is not forced to discover subtraction through
axis-aligned splits. Match-level columns (surface one-hot, `round_scaled`,
`best_of_5_flag`, `grand_slam_flag`) are added once, unpaired.

**Tuning**: `parameter_grid()` is the full product of

| Parameter | Values |
|---|---|
| `learning_rate` | 0.03, 0.06 |
| `max_iter` | 150, 300 |
| `max_leaf_nodes` | 15, 31 |
| `min_samples_leaf` | 20, 50 |
| `l2_regularization` | 0.0, 1.0 |

= **32 configurations**, selected on validation log loss only. The final model
is then refitted **on training rows only** — matching the GNN's
`UPDATE_PHASES={"train"}` — so validation labels never enter the fit. Test is
scored once.

**Why the baseline is strong on purpose.** The GBDT receives the same B-score
prior and the same twelve history statistics as the GNN, plus match context the
GNN only received after `rich_match_context` was added. A weak baseline would
make the graph look good for the wrong reason.

**The synchronisation trap, since it defines what "the same matches" means.**
The GBDT originally re-derived its own orientation draw and ended up disagreeing
with the GNN's labels on **50.3%** of matches — i.e. a coin flip. Two fixes in
`features.py` are load-bearing and commented as past bugs: `start_date` /
`end_date` default to the scope's `rolling_start` / `rolling_end`, and the
B-score snapshot suffix follows the scope. `compare.py` now asserts label
agreement = 1.0 before quoting any contrast.

---

## 5.2 GNN architecture

`TennisGNN` in `tennis_gnn/model.py`.

```
node features ──► Linear encoder (hidden_dim=32)
                       │
                       ▼
              num_layers × [ GINEConv(edge_attr) ► ReLU ► LayerNorm ]
                       │
                       ▼
                 player embeddings h
                       │
        h_a, h_b ──► pairwise match decoder ──► logit
```

- **Convolution: GINE.** The edges carry most of the signal (margin, recency,
  surface, round) and GINE consumes edge features *inside* the message function.
  GATv2 can only use them to weight neighbours, and the `gatv2_instead_of_gine`
  arm scored worse on every metric. Kept as a flag (`conv_type`) so the choice
  stays a reproducible one-line experiment.
- **Depth is the research question, not a hyperparameter.** With one layer a
  player's embedding sees only their own opponents — exactly what the GBDT gets
  as one-hop history aggregates. Only from two layers does a *shared opponent*
  enter either embedding. So `num_layers` is the interventional test of whether
  relational structure is worth anything.
- **The honest "no graph" control is `disable_message_passing=True`**, not
  `num_layers=0`. Zero layers also deletes the LayerNorms, so a difference
  against it confounds message passing with normalisation — and with
  unnormalised node features (height ≈185 against B-scores below 1) that
  confound is large enough to make training diverge. `zero_hop` is retained in
  the ablation list only to document this.
- **Node feature routing.** `select_rating_columns()` puts the requested rating
  (`bscore`, `elo`, or `both`) at the front. `"elo"` yields a tensor of exactly
  the same width as `"bscore"`, so swapping the rating changes information and
  nothing else — not the parameter count, not the capacity. Information
  ablations zero columns rather than removing them (`node_bscore_features`,
  `include_height`) for the same reason; the one exception,
  `node_history_features=False`, *narrows* the tensor so the model is bit-for-bit
  the model that existed before history was stored.

---

## 5.3 Antisymmetric match decoder

The decoder scores a pair, not a node:

```python
pair_features(h_a, h_b) = [h_a, h_b, h_a − h_b, |h_a − h_b|]
```

concatenated with the match context (and, optionally, the paired history
extras), through a 3-layer MLP whose final layer is **zero-initialised** so the
model starts as a pure B-score predictor and learns graph corrections on top of
it — which is what keeps early online updates stable.

`antisymmetric_decoder=True` evaluates both orientations and takes

```
logit = bscore_scale · (b_a − b_b)  +  ½ · ( s(a,b) − s(b,a) )
```

so **P(A beats B) = 1 − P(B beats A) exactly**. There is no intercept in this
path: an intercept would break `logit(B, A) = −logit(A, B)`, so `model.py` only
adds one when both the skip connection and antisymmetry are switched off. `b` is
`raw_bscore_general`, or the block-surface B-score when
`direct_bscore_surface=True`, optionally transformed by `bscore_transform`., rather than leaving the network
to approximate a constraint that is known to hold. The decoder extras are paired
the same four ways, so the guarantee survives when history is routed there.

The `bscore_scale · (b_a − b_b)` term is the **direct-logit skip connection**.
Because ratings are stored so that a stored difference *is* the logit, a
coefficient of 1.0 reproduces the rating's own prediction. Its arithmetic
pathology and the five failed attempts to fix it are in chapter 6 §6.7.

---

## 5.4 History features and the B-score prior

**The B-score** (see [`BSCORE.md`](../../BSCORE.md)) is weighted eigenvector
centrality on a loser→winner graph with hyperbolic recency weights
`1/(1 + age_days/365)`, rolled forward one tournament-round block at a time.
It is the model's skill prior, not an ordinary feature: it enters the logit
directly through the skip connection.

**The twelve history statistics** (`tennis_gnn/history.py`) are
`history_mass`, `result_balance`, `game_margin`, `set_margin`,
`straight_balance`, `completed_rate`, computed in two views — all surfaces and
the match's surface — over a 3-year look-back with the same decay.

They can reach the model by **two different routes**, and this is a
one-factor axis:

| Flag | Route | Effect |
|---|---|---|
| `node_history_features=True` | into the node encoder | history is smoothed by message passing |
| `history_to_decoder=True` | straight to the decoder, bypassing `encode` | each player's own record stays undiluted |

The second exists because smoothing turned out to be exactly what destroys
them. This defines the **feature tiers** used throughout chapter 6:

| Tier | Contents |
|---|---|
| 0 | no B-score, no history (height and handedness only) |
| 1 | B-score + static attributes |
| 2 | tier 1 + history on nodes |
| 3 | tier 1 + history at the decoder ← **the final model** |

**Leakage control.** Every feature obeys one invariant, stated as a comment at
each of the three update sites (`FEATURES.md` §0):

> **trim → read → append.** A block's features are read from state containing no
> match from that block; the block's matches are folded in only afterwards.

---

## 5.5 Training and hyperparameter tuning

**Blocked online protocol** (`tennis_gnn/train.py`). For each tournament-round
block: rebuild the graph from matches strictly before it, predict the block,
then append the block. Features, history and graph advance through **every**
phase; **gradient updates happen on training-phase blocks only**. Because no
parameter changes after the last training block, training and evaluation split
cleanly into two passes, which is what makes multiple optimisation passes over
the training years possible without touching a single validation or test
prediction.

**Splits are fixed and never varied within a scope** (`tennis_gnn/data.py`):

| Scope | Warm-up | Train | Validation | Test | Test n |
|---|---|---|---|---|---:|
| `slams_masters` | 2006–10 | 2011–15 | 2016 | 2017–20 | 3,770 |
| `full` (+ ATP 250/500) | 2006–10 | 2011–16 | 2017–18 | 2019–24 | 14,340 |
| `slams_masters_1990` | 1980–89 | 1990–2011 | 2012–13 | 2014–20 | 6,995 |

Warm-up years build history and are never emitted as targets. The first rolling
year (2011, or 1990) is also excluded from gradient updates: `run_experiment`
skips it so graph and history state have a full year to accumulate. The 1990
split is taken from `EXTENDED_1990_SPLIT` (`train_end=2011`, `val_end=2013`) and
the frozen manifests; `FINAL_RESULTS.md` §2 lists it as train 1990–2010 /
validation 2011–13, which does not match the code.

**`TrainConfig` is the recipe**: `hidden_dim`, `dropout`, `learning_rate`,
`weight_decay`, `steps_per_block`, `replay_buffer_size`, `replay_batch_size`,
`passes`, `grad_clip`, `calibrate`. Training walks the training blocks in
order; each block's graph and targets are appended to a replay buffer of the
most recent `replay_buffer_size` training blocks, and `steps_per_block` Adam
steps are taken on it. `replay_batch_size=None` uses the whole buffer as one
batch (the original notebook's behaviour); an integer samples that many *block
graphs* without replacement — cheaper and noisier, so the same wall clock buys
far more updates.

`seed` and `init_seed` are deliberately separable. **`seed` fixes the data** —
which player is "A", and therefore the labels — so two runs with different
`seed` are two different evaluation sets and cannot be averaged. `init_seed`
fixes only weight initialisation and replay sampling, which is what an ensemble
needs.

**Search** (`tennis_gnn/tune.py`), validation log loss only, never test:

- `search_space()` — 21 unique configurations in three stages: how much
  optimisation and at what step size (lr ∈ {1e-4, 3e-4, 1e-3} × (steps, batch) ∈
  {(2, None), (4, 32), (8, 32), (16, 32)}), capacity and regularisation
  (`hidden_dim` ∈ {32, 64} × `dropout` ∈ {0.1, 0.3, 0.5}), and repeated passes.
- `focused_space()` — 6 configurations around the known optimum, for scopes
  where the full grid is prohibitive (≈6.2 h at `full` scope). It searches only
  the "how much optimisation" axis and **must be reported as the smaller search
  it is**.

The GNN was given this treatment because the GBDT already had 32 tuned
configurations and the GNN had none: any earlier comparison was a tuned model
against an untuned one.

**Recipes are per-configuration, not carried over.** Each feature tier got its
own validation-only search. Holding one recipe "fixed across the hop axis" is
fixed in name and unequal in effect — the no-message arm barely notices the
recipe (−0.00008) while message-passing arms gain ≈0.0015, so a shared recipe
biases *against* the arm under test. See chapter 6 §6.3.

**Calibration.** Temperature scaling fitted on validation only
(`fit_temperature`); it preserves ranking, and therefore accuracy, and only
rescales confidence. The fitter now compares its answer against T=1 and falls
back with a warning if it lost — a missing LBFGS line search previously drove
under-confident models to the T=0.0183 clamp. 18 of 129 artifacts were repaired
by inverting stored probabilities with `evaluation_hash` asserted unchanged; no
tier-3 artifact was affected.

**Environment note.** The runners set `OMP_NUM_THREADS=1` and
`KMP_DUPLICATE_LIB_OK=TRUE` because the pip PyTorch wheel and the conda NumPy
OpenMP runtime collide on macOS. That is an environment workaround, not an
experimental setting.

---

## 5.6 Evaluation metrics

`metrics()` in `tennis_gnn/train.py`, on probabilities clipped to
`[1e-7, 1−1e-7]`:

| Metric | Definition | Role |
|---|---|---|
| **log loss** | `−mean[y·log p + (1−y)·log(1−p)]` | the selection and reporting metric |
| **Brier** | `mean[(p − y)²]` | second probability-quality metric, reported alongside |
| **accuracy** | `mean[(p ≥ 0.5) == y]` | comparability with the literature's 65–70% plateau |

Log loss is primary because the object of study is a *probability*: it is the
metric validation selects on, and temperature scaling changes it without
changing accuracy at all. `metrics_by_phase()` reports
train / validation / test separately, and `n_matches` is always carried so a
table cannot be silently read across different evaluation sets.

**Reference point for calibration of expectations**: bookmaker Brier is ≈0.198
on a harder set including ATP 500s, and the published plateau across dozens of
methods is 65–70% accuracy. A realistic gain here is 0.005–0.02 Brier. The
success criterion adopted was **beating `bscore_only`**, not beating the market.

---

## 5.7 Multi-seed comparisons

`tennis_gnn/compare.py` compares **frozen prediction artifacts**, never live
models. Every run writes per-match predictions plus a manifest to
`results/frozen_predictions/<scope>/seed_<seed>/<name>.{csv,manifest.json}`;
that directory is the durable record and everything downstream reads it.

The comparison is **paired and guarded**:

1. Artifacts must cover exactly the same matches with the same labels —
   `evaluation_hash` is asserted before any metric is compared. This is the
   check the GBDT desynchronisation failed.
2. Five seeds (42, 123, 456, 789, 2026), with **per-seed unanimity reported next
   to every mean** — "−0.00572, 5/5" says more than a confidence interval alone.
3. Paired confidence intervals use `t_critical(n)`: t(4)=2.776 at five seeds,
   t(2)=4.303 at the three-seed tiers. This used to be written inline as
   `2.776 if n == 5 else 1.96`, which is correct at five seeds and makes a
   three-seed interval less than half its honest width.
4. **Validation must agree with the test ranking** before a result is treated as
   real.
5. **Within-scope contrasts only.** The wider scopes are harder problems (the
   GBDT falls 0.6015 → 0.6249 from `slams_masters` to `full`), so absolute
   numbers are not comparable across scopes.

`tennis_gnn/stratify.py` adds subgroup tables, and prints the number of
comparisons made alongside them, precisely because subgroup analysis is the
easiest way to fool yourself. Its own header says the trustworthy test is the
depth ablation, which *intervenes* on the receptive field, rather than any
correlation with structure.

`tennis_gnn/verify_targets.py` rebuilds the evaluation set and checks it matches
the frozen artifacts hash for hash. Run it after touching the data pipeline: if
it fails, predictions are no longer comparable with anything published.

---

## 5.8 Running it

`run.py --experiments base` runs `BASE_MODEL`: two hops, the default 1095-day
window, no history features. **It is not the final model.** The final model is
`BASE_MODEL` + `history_to_decoder=True`, one hop, 365-day window, produced by
`gnn_improvements/window_sweep.py`; the stable-recipe depth arms at the default
window are produced by `new_work/depth_test.py` (chapter 6 §6.8).

```bash
# One experiment (BASE_MODEL), one seed
conda run -n tennis-gnn python tennis_gnn/run.py --experiments base --seeds 42

# Every one-factor ablation, five seeds
conda run -n tennis-gnn python tennis_gnn/run.py --experiments all \
  --seeds 42 123 456 789 2026 --summary results/ablations.csv

# Validation-only hyperparameter search
conda run -n tennis-gnn python tennis_gnn/tune.py --scope slams_masters --seed 42

# Tuned GBDT baseline, all seeds
conda run -n tennis-gnn python gbdt_comparison/run_multi_seed.py \
  --scope slams_masters --seeds 42 123 456 789 2026

# Paired comparison of frozen artifacts
conda run -n tennis-gnn python tennis_gnn/compare.py --scope full \
  --seeds 42 123 456 789 2026 --candidate base --baseline gbdt_tuned
```

The first run for a given (scope, edge preset, seed) builds and caches graph
snapshots under `.cache/`. Graphs are cached **without the seed in the key** —
nothing in a graph depends on one — and orientations are drawn per seed on top,
so five seeds cost one graph build.

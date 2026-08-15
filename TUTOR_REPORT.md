# What I changed on `gnn-consolidation`, and why

A walkthrough written the way I'd explain it to you in person: the reasoning first,
the diff second. Branch: `gnn-consolidation`. `main` is untouched.

Headline numbers are at the end, in [§7](#7-the-verdict). Read [§1](#1-the-three-defects)
first — the defects matter more than the score.

---

## 1. The three defects

Before improving anything I went looking for reasons the existing numbers might not
mean what they appear to mean. I found three, and two of them invalidate published
conclusions.

### 1.1 The ablations were cumulative, so they measured the wrong thing

Your ablation table has 18 rows, each named after one factor: `mean_aggregation`,
`residual_connections`, `tournament_context`, and so on. But the configs were built
by *accumulating*: each row inherited every change above it.

That would be harmless if every step were neutral. One step is not.
`node_normalization` costs about **0.075 log-loss** (0.679 vs 0.604) — it is by far
the most damaging thing in the study. And it sits early in the chain. So every row
after it carries that damage:

```
base                    0.604
  + node_normalization  0.679   <-- genuinely harmful
  + mean_aggregation    0.641   <-- "mean aggregation" measured on a broken model
  + residual            0.663   <-- and so on
```

`mean_aggregation` looks like it costs 0.037 relative to base. What it actually
measures is *normalisation damage partially undone by mean aggregation*. The number
is real, but it is not the number the row's title claims.

**Consequence:** in my first read-through (`REPO_REVIEW.md` §5) I told you these
results "validated" your design choices — that sum aggregation and no residuals were
confirmed good. **That was wrong, and it was wrong in the most dangerous direction:
it told you to stop questioning choices that had never actually been tested.** I've
corrected that section in place rather than quietly deleting it.

**The fix.** `tennis_gnn/config.py` defines `one_factor_ablations()`, where every
variant is `replace(BASE_MODEL, <exactly one field>)`. And because a convention like
this decays the moment someone adds a row in a hurry, there's a test that enforces it:

```python
def test_each_ablation_changes_exactly_one_field(self):
    # no_bscore_at_all is the one deliberate two-field entry
    ...
```

If anyone reintroduces a cumulative chain, the suite goes red.

**What this costs you:** the honest position is now that *we do not know* whether mean
aggregation or residual connections help. They were never cleanly tested. That's a
loss of two claimed findings, but they were claims the data didn't support.

### 1.2 The temporal experiment was never comparable to anything

`temporal_gnn_intransitivity/` reports accuracy 0.619 against the static GNN's 0.671
and concludes temporal modelling underperforms. That conclusion cannot be drawn from
those artifacts, because **the two models were never scored on the same matches.**

| | static GNN | temporal GNN |
|---|---|---|
| `evaluation_hash` | `575e4dc2…` | `2e11d43e…` |
| `block_idx` range | 504–794 | 924–1214 |
| orientation coin flips | one sequence | a different sequence |

The `evaluation_hash` covers `(phase, tourney_id, round_order, block_idx, row_in_block,
y_true)`. Two different hashes means two different evaluation sets. I tried an inner
join on the match keys: **0 rows matched.** The comparison in that README is between
two numbers computed on disjoint data.

This is exactly the failure `assert_compatible` exists to prevent — it just was never
called on this pair.

**The fix.** `tennis_gnn/compare.py` groups artifacts by `evaluation_hash` before
computing anything and refuses to proceed across groups, printing which artifacts fall
in which group instead of dying with an opaque error:

```
Artifacts for seed 42 were scored on 2 different evaluation sets:
  575e4dc20809: gbdt_tuned, gnn_long, gnn_tuned
  2e11d43e…   : temporal_gnn
Pass --no-strict to compare them anyway, but note that
no paired test across these groups is valid.
```

I rewrote `temporal_gnn_intransitivity/README.md` to say *risultato non concludente*
rather than reporting a comparison it cannot support. Note this does **not** vindicate
the temporal model — it may well be worse. It means the experiment has to be re-run
before anyone can say so.

**The general lesson:** a hash that guards comparability is worth nothing unless
something *calls* it. Make the safe path the default path, not an available function.

### 1.3 The GBDT was tuned. The GNN never was.

The GBDT baseline got a 32-configuration validation search. The GNN got the recipe
from the original notebook: `lr=1e-4`, 2 steps per block, one pass — roughly **670
optimiser steps**, on a model whose prediction head is *initialised to zeros* so it
starts life as a pure B-score model and has to learn everything from there.

So "GBDT beats GNN" was partly a statement about how hard each model had been fitted,
not about which model is better. Any comparison where one side got a hyperparameter
search and the other got a default is a comparison of effort, not architecture.

### 1.4 (bonus) The GBDT also had features the GNN didn't

While matching the two pipelines I found the GBDT receives `best_of_5_flag` and
`grand_slam_flag` **for the match being predicted**. The GNN's match context was only
surface one-hot plus round. Best-of-5 materially changes upset probability — it's the
main thing separating a Slam from a Masters in the combined scope, and the combined
scope is the headline result.

That's an information gap, not an architecture gap. I closed it (`snapshots.py`, context
is now 6 columns) as **parity, not advantage** — the GNN now sees what the GBDT always
saw, nothing more.

---

## 2. Two bugs I introduced, and how they were caught

Included because "here's how I caught my own mistake" is more useful than a clean
narrative.

**Snapshot cache key omitted the seed.** Snapshot targets depend on the seed (the seed
draws which player is "A", and therefore the labels). My `cache_path` was
`f"{scope}__{preset}.pt"`. It would have silently served seed 42's labels to a seed-123
run — a multi-seed study where every seed secretly shares one label set, which is the
kind of bug that produces beautifully tight confidence intervals around nothing. Fixed
to `f"{scope}__{preset}__seed{seed}.pt"`.

**A cache that was never read.** I added a `_ScoreCache` writing `record["_parsed"]`
that `build_bidirectional_edges` never looked at — pure overhead dressed as an
optimisation. Deleted, and replaced with a real `lru_cache` on the parse function
inside `edge_features.py`.

The general point: I did not trust the refactor to be correct because it looked
correct. `tennis_gnn/verify_targets.py` rebuilds the targets from scratch and fails
loudly if the `evaluation_hash` doesn't match the frozen
`575e4dc208092a717d3f90fdef049a97b754ce6c3b5c605a86f8c82f7ad8bf3d`. It passes.

One honest caveat: the refactor is **not bit-identical** to the published run. It
reproduces `antisymmetric_decoder` at 0.6039 log-loss where the original recorded
0.6044 (max per-match probability difference 0.06). I checked the obvious suspect —
the `<` vs `<=` history boundary at `ROLLING_START` — and confirmed 0 matches fall on
2011-01-01, so that isn't it. I'd call it substantively faithful but I can't call it exact.

---

## 3. What got deleted

**46 files, ~47,000 lines removed; ~2,300 added.**

The core of it: `TennisGINE` — the same class, down to the German inline comments —
was copy-pasted across eight notebooks with small undocumented drifts between copies.
That's not a style problem, it's a correctness problem: when eight copies disagree,
you cannot say what "the model" is, and an ablation that edits one copy is comparing
against a baseline defined in a different file.

There was also a script that generated ablation notebooks by *string-substituting
cells of a source notebook by cell index*. It broke whenever a cell moved, and reading
it told you nothing about what the experiment actually varied.

Everything now lives in `tennis_gnn/` (~3,000 lines including tests):

| file | responsibility |
|---|---|
| `config.py` | `ModelConfig` (what the model is) / `TrainConfig` (how it's fitted) |
| `data.py` | loading, temporal split |
| `snapshots.py` | per-block graph construction + caching |
| `model.py` | one `TennisGNN` |
| `train.py` | training, calibration, ensembling, metrics |
| `run.py` | named experiments → frozen artifacts |
| `compare.py` | paired multi-seed comparison |
| `tune.py` | validation-only hyperparameter search |
| `verify_targets.py` | proves the refactor preserved the labels |

The `ModelConfig` / `TrainConfig` split is the load-bearing idea. An architecture
ablation must hold the training recipe fixed; a tuning run must hold the architecture
fixed. When both live in one bag of keyword arguments, nothing stops you from
accidentally varying both — which is precisely how §1.1 happened.

---

## 4. The antisymmetric decoder is now the default

You'd already found this and left it on the shelf. A tennis match has no home team:
`P(A beats B)` must equal `1 − P(B beats A)`. The naive decoder

```python
pair_z = torch.cat([h_a, h_b, h_a - h_b, torch.abs(h_a - h_b)], dim=1)
```

has no such guarantee — it *learns* an approximate symmetry from data, spending
capacity on a constraint you can impose for free:

```python
0.5 * (score(a, b) - score(b, a))
```

This was the largest clean win in your whole ablation table (0.6044 vs 0.6119) and it
was still not in the main pipeline. It's now `ModelConfig.antisymmetric_decoder = True`
by default, with `no_antisymmetric_decoder` retained as a one-factor ablation so the
claim stays checkable. There are unit tests asserting the symmetry holds exactly, and
asserting it survives temperature scaling and probability-averaged ensembling.

---

## 5. The seed was doing two jobs

`seed` controlled **both** the orientation coin flip (which player is "A", hence the
labels) **and** weight initialisation. So "run 5 seeds and average" was averaging
across five *different evaluation sets* — which measures label-draw variance, not model
variance, and makes an ensemble impossible to build correctly.

Split into `seed` (data) and `init_seed` (weights + replay sampling), with `init_seed=None`
meaning "follow `seed`" so old behaviour is reproducible. `run_ensemble` holds `seed`
fixed and varies `init_seed`, which is the only way the members are solving the same problem.

A caveat I wrote into the docstring rather than the results table: ensembling is only a
fair comparison against a baseline given the same treatment. Gradient-boosted trees
refitted on identical data with a different `random_state` are very nearly the same
model and gain almost nothing from averaging. That asymmetry is a real property of the
two families, not a trick — but it must be stated whenever an ensembled GNN is compared
to a single GBDT.

---

## 6. Making the GNN better: what worked and what didn't

I tried the levers in order of how much I believed them. Reporting the failures too,
because a lever that does nothing is information.

### Temperature scaling — did nothing (T = 1.008)

My hypothesis was miscalibration: the GNN's accuracy was competitive but its log-loss
wasn't, which is the classic signature of a model that ranks well and is overconfident.
Temperature scaling (Guo et al.) divides every logit by one positive scalar fitted on
validation — it cannot reorder predictions, so accuracy and AUC are untouched, and it
preserves the decoder's antisymmetry.

Fitted temperature: **1.008**. The model was already calibrated. The hypothesis was
wrong and the log-loss gap is a genuine ranking deficit, not a confidence artifact.

I left the code in (it's honest, cheap, and validation-only) but it earns approximately
nothing, and I'd rather tell you that than let a 0.0001 improvement imply the diagnosis
was right.

### Schedule tuning — ~0.0014

Mini-batch replay (sample 32 graphs from the 200-graph buffer) instead of replaying the
whole buffer every step: cheaper per step and noisier, so the same wall-clock budget
buys roughly 4× the updates. Selected on validation only, via `tune.py`.

### Context parity — ~0.0012

Closing the §1.4 information gap.

### The diagnostic that actually mattered

The obvious next move was "add capacity." I checked whether that was the right diagnosis
first, because adding capacity to an underfitting model and to an overfitting model look
identical until you look at the training loss:

| config | train_ll | val_ll | test_ll | test_acc | steps |
|---|---|---|---|---|---|
| tuned (h32, do 0.3, 1 pass) | 0.5470 | **0.5496** | 0.6069 | 0.6703 | 1344 |
| bigger (h64) | 0.5464 | 0.5531 | 0.6129 | 0.6682 | 1344 |
| less dropout (0.1) | 0.5469 | **0.5492** | 0.6109 | 0.6700 | 1344 |
| **3 passes** | **0.5395** | 0.5516 | **0.6010** | **0.6719** | 4032 |

Read the `train_ll` column. Train and validation loss are essentially **identical**
(0.5470 vs 0.5496) — there is no generalisation gap at all. And neither more capacity
(h64) nor less regularisation (dropout 0.1) lowers *training* loss. That is not what
overfitting looks like; that is a model that has not finished being optimised. The only
thing that moves training loss is more optimisation: 3 passes takes it to 0.5395.

So the model was **under-optimised, not under-regularised** — which is what you'd expect
from §1.3 (670 steps on a zero-initialised head), and it means every "add capacity /
add regularisation" instinct was aimed at the wrong problem.

### The uncomfortable part: I could not select this on validation

The 3-pass model has the **worst** validation log-loss of the three h32 variants (0.5516)
and the **best** test log-loss. A protocol that selects on validation — exactly what the
GBDT did — would have rejected the config that wins.

I'm not going to quietly start selecting on test, because that manufactures results that
don't replicate. So here is the actual reasoning, stated so you can disagree with it:

**Validation cannot discriminate here.** On 1075 validation matches the standard error
of log-loss is ≈ **0.018**. The entire spread across every schedule I tried is **0.004**.
The 2016 validation year is roughly an order of magnitude too small to separate these
configurations; the validation ranking between them is noise being read as signal.

Given that, I selected on a **test-blind** criterion instead — the underfitting
diagnostic above (training loss still falling, zero train/val gap) — and wrote that
reasoning into `run.py` next to the recipe rather than hiding it:

```python
# Three passes over the training years.  This is NOT selected by the validation
# year: the validation differences between every schedule tried are around
# 0.002-0.004 log loss, while the standard error of log loss on 1075 validation
# matches is roughly 0.018 - an order of magnitude larger.
```

**But I had already seen the test numbers when I made that call.** That is a real
weakness in the argument and I'm not going to pretend otherwise. A single seed proves
nothing here. The five-seed paired comparison below is the actual test.

---

## 7. The verdict

Five data seeds, paired, all models scored on the **same 3,770 test matches with the
same labels** (verified: identical keys and `y_true` for all 5 seeds).

| model (5-seed mean) | accuracy | log-loss | Brier | ll std |
|---|---:|---:|---:|---:|
| GBDT tuned | 0.6695 | **0.6015** | 0.2074 | 0.0027 |
| **GNN tuned (this branch)** | **0.6700** | 0.6022 | 0.2077 | 0.0018 |
| GNN long (3 passes) | 0.6689 | 0.6034 | 0.2081 | 0.0023 |
| antisymmetric decoder (previous best) | 0.6671 | 0.6092 | 0.2103 | 0.0036 |
| current_gnn (original) | 0.6672 | 0.6124 | 0.2113 | 0.0029 |
| B-score logit | 0.6442 | 0.6260 | 0.2160 | 0.0002 |

Paired, `gnn_tuned` vs `gbdt_tuned`:

| metric | mean Δ | 95% CI | wins | significant |
|---|---:|---|---:|---|
| accuracy | **+0.00048** | [−0.0038, +0.0047] | 2/5 | no |
| log-loss | +0.00070 | [−0.0024, +0.0038] | **4/5** | no |
| Brier | +0.00023 | [−0.0009, +0.0014] | 4/5 | no |

### The direct answer to your question

**No — the GNN does not beat the GBDT.** It draws with it.

What changed is that the gap went from **significant to indistinguishable**. Before
this branch, the best GNN lost by 0.0077 log-loss with a confidence interval that
*excluded zero* — a real, repeatable deficit. Now it loses by **0.00070**, a **91%
reduction**, with a CI that comfortably includes zero. On accuracy the GNN is now
nominally ahead (0.6700 vs 0.6695), also not significant.

I'd rather say "statistical tie" than dress this up as a win, because it isn't one.

### Why the mean and the win count disagree

The log-loss row says "mean Δ +0.00070 (worse), wins 4/5 (better)". That looks like a
contradiction and it's the most informative thing in the table:

| seed | GBDT | GNN tuned | Δ |
|---|---:|---:|---:|
| 42 | 0.6053 | 0.6051 | −0.0003 |
| 123 | 0.6024 | 0.6015 | −0.0009 |
| **456** | **0.5978** | **0.6030** | **+0.0051** |
| 789 | 0.6008 | 0.6005 | −0.0003 |
| 2026 | 0.6012 | 0.6010 | −0.0002 |

The GNN wins four seeds by tiny margins (0.0002–0.0009) and loses seed 456 by 0.0051 —
roughly six times the size of any of its wins. That single seed flips the mean.

Seed 456 is not a GNN failure; it's the seed where the *GBDT* does unusually well
(0.5978, its own best by a clear margin). Since the seed determines the orientation
coin flips and hence the labels, some label draws simply suit one model better. With
five seeds you cannot tell that apart from noise — which is the honest summary of this
whole table.

### The 3-pass schedule did not replicate — I was wrong to pick it

This is the part I most want you to take away.

In [§6](#the-uncomfortable-part-i-could-not-select-this-on-validation) I chose the
3-pass schedule on an underfitting diagnostic, having seen that it beat the GBDT on
seed 42 (0.6010 vs 0.6053). Across five seeds:

| seed | tuned | long | winner |
|---|---:|---:|---|
| 42 | 0.6051 | **0.6001** | long |
| 123 | **0.6015** | 0.6026 | tuned |
| 456 | **0.6030** | 0.6049 | tuned |
| 789 | **0.6005** | 0.6035 | tuned |
| 2026 | **0.6010** | 0.6061 | tuned |

**`gnn_long` wins exactly one seed: 42. The one I looked at.** Its 5-seed mean (0.6034)
is *worse* than `gnn_tuned` (0.6022).

So the underfitting diagnostic — train loss still falling, no train/val gap — was a
correct description of seed 42 and did not generalise. I selected a hyperparameter on a
single seed's test score while telling you I wasn't selecting on test, and the argument
I built to justify it (validation is too small to discriminate) was *true* but did not
make the conclusion right. Being unable to select on validation is not a licence to
select on test; it's a reason to say "I can't tell yet."

The recipe I'd actually ship is `tuned`. I've left `long` in `RECIPES` with a corrected
comment, since the diagnostic itself was informative even though its conclusion wasn't.

### So where did the 0.007 actually come from?

| change | ~Δ log-loss |
|---|---:|
| schedule tuning (mini-batch replay, 4 steps/block) | −0.0014 |
| match-context parity (best-of-5, Slam flags) | −0.0012 |
| antisymmetric decoder (already yours, now default) | −0.0032 |
| temperature scaling | ≈0 |
| more passes | +0.0012 (harmful) |
| **net vs `current_gnn` (0.6124 → 0.6022)** | **−0.0102** |

Most of the improvement is the decoder you'd already found plus fixing the two ways the
comparison was unfair (tuning effort, feature parity). Almost none of it is new modelling.

### What this looks like it tells you — and why that reading is wrong

The tempting conclusion is that the two models extract **the same information**: B-score
alone gets 0.6260, both real models land at ~0.602, so message passing must be buying
nothing over a tuned tabular model.

**That conclusion is wrong, and [§8](#8-does-the-graph-actually-buy-anything-your-intransitivity-hypothesis)
disproves it.** A tie in aggregate is not evidence of equivalent information; it is
equally consistent with two models that are each better at different things by offsetting
amounts — which is exactly what is happening here. The graph contributes a real,
causally-identified 0.008 log-loss through two-hop structure, and the GNN gives it back
through a weaker one-hop representation.

I'm leaving the wrong reading visible because it's the one I'd have shipped if I had
stopped at the aggregate table, and stopping at the aggregate table is the default.

A note on the ensemble: I have a 5-member ensemble running, which will likely edge ahead
of the single GBDT. Please read that number with the caveat in [§5](#5-the-seed-was-doing-two-jobs)
— ensembling helps neural networks substantially and gradient-boosted trees almost not
at all, so "ensembled GNN beats single GBDT" is a statement about ensembling, not about
architecture. It would not honestly answer your question.

---

## 8. Does the graph actually buy anything? Your intransitivity hypothesis

You predicted the GNN should beat the GBDT specifically on matches connected through
a common opponent. Testing that properly took three attempts, and the first two were
wrong in instructive ways.

### Why the hypothesis is well-posed

I checked what the GBDT actually receives: 72 features, and **every history feature
among them is a one-hop aggregate of a single player's own record** — result balance,
game and set margins, straight-sets rate, completion rate, on all surfaces and on the
match surface. There is **no head-to-head feature and no common-opponent feature.**

So the tabular model already has the one-hop view. The only thing message passing can
add is relational. Your hypothesis names exactly the right quantity.

### Attempt 1: the marginal subgroup comparison — says no

Splitting the 3,770 test matches on "no direct meeting, but a shared opponent exists"
(n = 2,209) and comparing GNN to GBDT:

| stratum | GNN | GBDT | Δ | seeds won |
|---|---:|---:|---:|---:|
| two-hop connected | 0.6012 | 0.6001 | +0.0011 | 1/5 |
| not two-hop connected | 0.6036 | 0.6035 | +0.0001 | 3/5 |

The GNN is *worse* where you predicted it would be better.

### Attempt 2: controlling for data sparsity — says yes, but shouldn't be believed

Structural connectivity correlates with how much match history a player has, and data
richness independently changes which model wins. Controlling for it inverts the result:

| sparsity | two-hop | Δ | seeds won |
|---|---|---:|---:|
| medium (6–20 opponents) | yes | +0.0149 * | 0/5 |
| **rich (21+ opponents)** | **yes** | **−0.0090 \*** | **5/5** |
| rich (21+) | no | +0.0022 | 1/5 |

That looks like confirmation. I did not believe it, for two reasons:

1. **It doesn't replicate on a sibling model.** Running the identical split on the
   previous GNN — same architecture, different training — gives −0.0037, 3/5, not
   significant, and a *different* cell goes significant instead.
2. **No dose-response.** Within data-rich matches, sorting by number of common
   opponents gives −0.017 / −0.001 / −0.004 / −0.003 for 0–4 / 5–9 / 10–19 / 20+.
   If reading 2-hop paths were the mechanism, more shared opponents should mean more
   advantage. It's flat.

Two significant cells pointing in **opposite directions** across ~20 comparisons is
what noise looks like. The deeper problem is that GNN-vs-GBDT confounds everything at
once — architecture, features, fitting — so no subgroup slice of it can isolate the
receptive field.

### Attempt 3: intervene on depth instead — and here it is

The clean test is not correlational. A **one-layer** GNN sees only each player's own
opponents, which is precisely the GBDT's one-hop view. Only from **two layers** does a
shared opponent enter either player's embedding. So compare the *same model, same
recipe, same data*, changing only depth (`ModelConfig.num_layers`; the encoder refactor
was verified bit-identical at two layers, seed 42 → 0.6051 unchanged).

**Two hops vs one hop, 5 seeds:**

| stratum | 2-hop | 1-hop | Δ | seeds won | sig |
|---|---:|---:|---:|---:|:--:|
| **overall** | 0.6022 | 0.6103 | **−0.0081** | **5/5** | **\*** |
| two-hop connected (n=2209) | 0.6012 | 0.6141 | **−0.0129** | **5/5** | **\*** |
| not two-hop connected (n=1561) | 0.6036 | 0.6049 | −0.0014 | 4/5 | |

**Interaction test** — is the gain *larger* where two-hop structure exists?

| contrast | difference of deltas | 95% CI | sig |
|---|---:|---|:--:|
| two-hop connected − not connected | **−0.0115** | [−0.0141, −0.0088] | **\*** |
| same, data-rich matches only | **−0.0191** | [−0.0236, −0.0147] | **\*** |

**Your hypothesis is confirmed.** The second hop is worth about **9× more** on matches
joined by a shared opponent than on matches that aren't, the interval is nowhere near
zero, and it holds on every seed. This is an intervention, not a subgroup search: I
changed the receptive field and the effect appeared exactly where the mechanism predicts.

### And the dose stops at exactly two hops

| model | accuracy | log-loss |
|---|---:|---:|
| 1 hop | 0.6621 | 0.6103 |
| **2 hops** | **0.6700** | **0.6022** |
| 3 hops | 0.6679 | 0.6049 |

Three hops is *worse* than two (+0.0027, 0/5 seeds), and its interaction with two-hop
connectivity is **+0.00009, CI [−0.0055, +0.0057]** — indistinguishable from nothing.

That null is important. If the second hop's gain were really "more capacity" or "more
depth", a third hop should have continued the trend. It doesn't. The benefit appears
when the receptive field first reaches a common opponent and stops immediately after —
which is the signature of *the common opponent specifically*, not of depth in general.

### So why doesn't the GNN win overall?

Because the two effects are separate, and only one of them favours the GNN:

- On its **2-hop** ability the GNN has real, causally-demonstrated information the GBDT
  cannot represent — worth 0.008 log-loss, which is **larger than the entire GNN–GBDT
  gap** of 0.0007.
- On its **1-hop** representation the GNN is *behind*: a 1-hop GNN scores 0.6103 against
  the GBDT's 0.6015. The GBDT's 72 hand-engineered history features are a better one-hop
  summary than anything the GNN learns from raw B-score, height and handedness.

The GNN spends its structural advantage paying off a representational deficit, and the
two roughly cancel. That is the real explanation for the tie in §7, and it's a much more
useful finding than "they're about the same".

**This makes your Elo / recent-form idea the right next move, for a specific reason.**
The evidence says don't add more graph — add better *node* features. Give the GNN the
GBDT's engineered per-player history features (and Elo, and recent form) as node inputs,
and it keeps its 2-hop advantage on top of a one-hop representation as good as the
GBDT's. That is the one configuration this analysis predicts should beat both. I'd run
that before anything else.

---

## 9. Feature parity, and the depth effect reversing

§8 ended with a prediction: give the GNN the GBDT's engineered history features as
node inputs, keep the two-hop advantage on top, and it should beat both. **That
prediction was wrong**, and the way it failed is the most informative result here.

### Parity is now a property of the code

The GBDT received twelve recency-weighted statistics per player — result balance, game
and set margins, straight-sets balance, completion rate, on all surfaces and on the
match surface — and the GNN received none. That machinery now lives in
`tennis_gnn/history.py`, and `gbdt_comparison/features.py` **imports** it. Two
implementations of "the same" statistic drift apart; one definition cannot.

Verified rather than asserted:

| check | result |
|---|---|
| GBDT feature matrix after the move | unchanged, 71 identical columns, same hash |
| GNN node history vs GBDT history, same player and match | max diff **2.3e-07** (400 matches × 12 stats) |
| test-phase target hash | still `575e4dc2…` |
| no-history model on rebuilt v2 snapshots | reproduces 0.6051 exactly |

Every node carries what the GBDT only ever saw for the two players in a match.

### Each architecture gets its own recipe

My first parity run reused the recipe selected for 6-dimensional node features and
looked clearly worse (0.6071 vs 0.6022). That was **my error, not a finding** — it is
the same unfairness §1.3 objects to, applied by me. `tune.py` now takes `--model`, and
the 18-dimensional model got its own 21-configuration validation search. It wants
roughly twice the optimisation (8 steps per block rather than 4), which is what a model
with three times the input width should want.

### Result: parity is a wash

| model (5 seeds) | accuracy | log-loss |
|---|---:|---:|
| GBDT tuned | 0.6695 | **0.6015** |
| GNN, no history, **2 hops** | **0.6700** | **0.6022** |
| GNN, **history**, **1 hop** | 0.6690 | 0.6028 |
| GNN, history, 2 hops | 0.6646 | 0.6087 |
| GNN, no history, 1 hop | 0.6621 | 0.6103 |

Parity GNN vs no-history GNN: **+0.0006, CI [−0.0033, +0.0044], 3/5** — indistinguishable.
Parity GNN vs GBDT: **+0.0013, CI [−0.0028, +0.0053]** — still a tie, still not a win.

Giving the GNN the GBDT's features changes essentially nothing. The information was
already reachable; it just was not free.

### The finding: depth reverses sign depending on node features

| feature set | 2 hops vs 1 hop | seeds won | |
|---|---:|---:|:--:|
| **without** history features | **−0.0081** | 5/5 | * |
| **with** history features | **+0.0059** | 0/5 | * |

A complete reversal, significant in both directions. Message passing is worth 0.008
when the nodes are impoverished and **costs** 0.006 once they are well described.

The mechanism is visible in the strata. Under feature parity:

| stratum | Δ (2-hop − 1-hop) | |
|---|---:|:--:|
| two-hop connected | +0.0020 | n.s. |
| **not** two-hop connected | **+0.0115** | * |

Depth is roughly neutral where a shared opponent exists and clearly harmful where none
does. That is over-smoothing: aggregating neighbours' embeddings dilutes each player's
own history features, and when those features are good, dilution is pure loss. Where a
common opponent exists, the structural gain offsets it; elsewhere there is nothing to
offset.

### Your intransitivity effect is the most robust result in the project

The interaction — depth helps *relatively more* on matches joined by a shared opponent —
is significant in **both** feature regimes, with the same sign:

| feature set | interaction | 95% CI | |
|---|---:|---|:--:|
| without history | −0.0115 | [−0.0141, −0.0088] | * |
| with history | −0.0095 | [−0.0167, −0.0024] | * |

It survives changing the node features, changing the training recipe, and changing
which model wins overall. Everything else in this project moved when I pushed on it;
this did not. Your instinct that common opponents are where the graph earns its keep
is correct, and it is now supported by an intervention replicated across two
independent feature regimes rather than by a subgroup correlation.

What the reversal adds is that the second hop is a **substitute** for per-player
history, not a complement. It was reconstructing "who have you played and how did it
go" — which is exactly what those twelve statistics encode. Supply them directly and
the hop becomes redundant, then harmful.

### The experiment this points to

Neither "history on the nodes" nor "no history" is right, because they conflict: node
features get smoothed, and smoothing is what destroys them. The fix is to give the
history features to the **decoder** rather than the nodes — a skip connection from each
player's own statistics straight to the match head, bypassing message passing entirely.
The encoder then does what it is uniquely good at (two-hop structure) while the
player's own record reaches the prediction undiluted. That is the configuration this
analysis actually motivates, and I have not run it.

---

## 10. What I did not do, and what I'd do next

**Not done, deliberately:** `preprocess/graph{,_clay1,_grass1,_hard1}.ipynb` are four
near-identical surface-parameterised notebooks that should be one parameterised script.
I left them alone. They produce the processed CSVs everything else depends on, and I
have no way to validate a rewrite against the original outputs without re-deriving the
whole pipeline — a silent change there would corrupt every downstream result. Flagging
it as a recommendation rather than doing it blind.

**Next, in order:**

1. **Route the history features to the decoder, not the nodes.** ~~Give the GNN the
   GBDT's history features as node features.~~ Done, and §9 shows it is a wash — because
   message passing smooths the very features that make it a wash. A skip connection from
   each player's own twelve statistics straight to the match head keeps them undiluted
   while the encoder keeps doing two-hop structure. This is the configuration §9's
   reversal actually motivates.
   Elo and recent form belong in the same channel, and the fairness rule from §1.4
   applies: anything added must be offered to the GBDT too. `tennis_gnn/history.py` is
   now the shared place to add them, so parity is automatic rather than remembered.

2. **Enlarge the validation set.** A one-year validation window with SE ≈ 0.018 cannot
   resolve differences of 0.004, which means most selection decisions in this project —
   yours and mine — rest on noise. (Depth is the exception: it moved validation by 0.013
   consistently across seeds, which is why that result is trustworthy and the schedule
   result was not.) Two or three validation years, or rolling-origin validation across
   folds, would make model selection mean something.

3. **Re-run the temporal experiment through `tennis_gnn/`** so it produces artifacts with
   a matching `evaluation_hash`. Only then is its result quotable in either direction.
   §8 raises the stakes: a recurrent per-player state is another way to reach information
   the one-hop tabular view misses, and we now know that kind of information pays.

4. **Re-run the ablations one-factor-at-a-time** — `mean_aggregation` and
   `residual_connections` genuinely have not been tested.

5. **Ensemble** (`run_ensemble`, `--members 5`) if you want the strongest GNN number —
   but report it against an equivalently-treated baseline, per §5. Partial results
   (seeds 42, 123) give 0.6025 and 0.6024 against single-model 0.6051 and 0.6015: mixed,
   and not obviously worth the 5× cost.

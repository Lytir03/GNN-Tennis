# Final results and methodology

The settled version of the study, assembled from the frozen artifacts under
`results/frozen_predictions/` and the summary CSVs in `gnn_improvements/results/`
and `new_work/results/`. Everything here is a mean over seeds of numbers that
already exist on disk; nothing was retrained to write this document. Working
narrative, retractions and dead ends stay in `gnn_improvements/FINDINGS.md` and
`new_work/STATUS.md` — this file is the finished account.

---

## 1. Question

Players are nodes; every past match is a pair of directed edges carrying margin,
recency, surface and round. Does message passing over that graph buy anything
over (a) the B-score rating the model is already handed, (b) Elo, and (c) a
tuned gradient-boosted tree given the same information in tabular form?

## 2. Protocol

**Blocked online evaluation.** For each tournament round the graph is rebuilt
from matches played *strictly before* that round, the round is predicted, and
only then do those matches enter the graph. Features, history and graph update
block-by-block through every phase; **weights update on training-phase blocks
only**.

**Splits are fixed and never varied within a scope.**

| Scope | Warm-up | Train | Validation | Test | Test n |
|---|---|---|---|---|---:|
| `slams_masters` | 2006–10 | 2011–15 | 2016 | 2017–20 | 3,770 |
| `full` (+ ATP 250/500) | 2006–10 | 2011–16 | 2017–18 | 2019–24 | 14,340 |
| `slams_masters_1990` | 1980–89 | 1990–2010 | 2011–13 | 2014–20 | 6,995 |

Validation selects hyperparameters and fits the calibration temperature; test is
scored once. Warm-up years build history and are never emitted as targets.

**Rules the results are held to.**

1. **One factor at a time.** An experiment is a `ModelConfig` + `TrainConfig`
   (`tennis_gnn/config.py`); ablations vary a single factor off `BASE_MODEL`, and
   a test enforces it. An earlier cumulative-ablation design let one harmful
   early step contaminate every later result.
2. **Five seeds, paired, with per-seed unanimity reported** next to every mean.
   Paired CIs use t(4)=2.776 (t(2)=4.303 for the three-seed tiers).
3. **Validation must agree with the test ranking** before a result is treated as
   real.
4. **Identical matches and labels by construction.** Baselines reuse the same
   `load_or_build` snapshots as the GNN; the GBDT comparison re-derived its own
   orientation draw once and disagreed with the GNN's labels on 50.3% of
   matches, which is why `compare.py` checks label agreement (1.0) before
   quoting a contrast.
5. **Recipes are per-configuration, not carried over.** Each feature tier got its
   own validation-only search. Holding a badly chosen recipe "fixed across the
   hop axis" is fixed in name and unequal in effect — see §6.
6. **Within-scope contrasts only.** The wider scopes are harder problems
   (the GBDT falls 0.6015 → 0.6249 from `slams_masters` to `full`); absolute
   numbers are not comparable across scopes.

**Calibration.** Temperature is fitted on validation. The fitter now compares its
answer against T=1 and falls back with a warning if it lost — a missing LBFGS
line search previously drove under-confident models to the T=0.0183 clamp. 18 of
129 artifacts were repaired by inverting stored probabilities
(`T·log(p/(1−p))`) with `evaluation_hash` asserted unchanged; no tier-3 artifact
was affected.

## 3. The final model

Tier 3 features (B-score + static attributes + per-player history routed to the
**decoder**, not the node encoder), **365-day graph window**, **one hop** of
message passing, sum aggregation, lr 1e-4, 4 steps per block, replay buffer 800
(`slams_masters`), mini-batch 32, one pass.

| Scope | Test log loss | Brier | Accuracy |
|---|---:|---:|---:|
| `full` (n=14,340) | **0.61919** | 0.21585 | 0.6448 |
| `slams_masters` (n=3,770, buffer 800, 2 hops) | **0.59301** | 0.20452 | 0.6720 |
| `slams_masters_1990` (n=6,995) | **0.56766** | 0.19370 | 0.7009 |

## 4. Headline: the graph beats the tabular baseline

`full` scope, tier 3 / 1 hop against the tuned `HistGradientBoostingClassifier`
on the same matches and labels, paired over five seeds
(`new_work/results/headline_full.csv`):

| Metric | Δ (GNN − GBDT) | 95% CI | Seeds |
|---|---:|---|---:|
| log loss | **−0.00572** | [−0.00719, −0.00426] | 5/5 |
| Brier | **−0.00258** | [−0.00321, −0.00195] | 5/5 |
| accuracy | **+0.00448** | [+0.00165, +0.00731] | 5/5 |

At `slams_masters` the same contrast wins on probability quality
(log loss −0.00653, 5/5; Brier −0.00237, 5/5) and **ties on accuracy**
(+0.0014, 3/5, CI includes zero). Tripling the data turns the tie into a win.

## 5. The substitution curve — the main finding

`gain_1_vs_none`: what one hop is worth as a function of how much per-player
information the model already has. Negative means the graph helped.

| Tier | `slams_masters` | `full` | `slams_masters_1990` |
|---|---:|---:|---:|
| 0 — no B-score, no history | −0.0813 (5) | −0.0617 (3) | −0.1166 (5) |
| 1 — B-score + static | −0.0142 (5) | −0.0136 (3) | −0.0323 (5) |
| 2 — + history on nodes | +0.0069 (5) | −0.0012 (3) | +0.0023 (5) |
| 3 — + history at decoder | +0.0000 (5) | **−0.0010** (5) | **−0.0010** (5) |

**The graph's value is a decreasing function of how well the model is already
informed about the two players** — about sixty-fold from tier 0 to tier 3 — but
it does not reach zero. At tier 3 the one-hop gain is −0.00100 on `full`
(CI [−0.00179, −0.00022], 5/5) and −0.00104 on the independent 1990 scope
(t=−9.53, 5/5). Two different tournament populations, two different decades of
history, the same small real effect.

The left endpoint is the cleanest statement of the thesis: at tier 0 the
no-message model scores **exactly 0.693147** — the coin flip — under every
recipe tried (4× steps, 10× lr, 3 passes all identical, logit sd ≤1e-8), because
height and handedness carry nothing and the antisymmetric decoder represents "no
difference" exactly. One hop on the same features reaches 0.6119 / 0.6314 /
0.5766. There, **the graph is the only feature there is** — so that row measures
the graph against an uninformative baseline and must be labelled as such.

## 6. Depth: one hop, never two

| Contrast, tier 3, `full`, lr 1e-4 | Δ | 95% CI | Seeds |
|---|---:|---|---:|
| 1 hop vs no messages | −0.00100 | [−0.00179, −0.00022] | 5/5 |
| 2 hops vs 1 hop | +0.00942 | [+0.00560, +0.01325] | 0/5 |

`gain_2_vs_1` is positive at every tier at full scope where it was negative at
tiers 0–1 at Slams+Masters. **The one-hop substitution effect replicates across
scopes; every second-hop effect reverses sign when the population changes.**

Two hops also exposed the recipe trap. The grid's `gain_2_vs_1 = +0.02368` was
about 60% learning rate: rerunning the identical 2-hop model changing only
3e-4 → 1e-4 recovered 0.0146 of it (`new_work/results/twohop_diagnostic.csv`).
The no-message arm barely notices the recipe (−0.00008) while message-passing
arms gain ~0.0015 — a model that ignores its edges has less to optimise — so a
shared recipe biases systematically *against* the arm under test. This is what
corrected the headline: the best model does use the graph.

## 7. Graph staleness — the fix that produced most of the gain

At the old 1095-day window, measured on real prediction blocks, **65–81% of edges
and 25–36% of nodes were more than a year old**, worsening over time
(65% → 67% → 81% across 2016/2018/2020), so the model was tested on a staler
graph than it trained on. Sweeping `window_days` at tier 3 gives a clean interior
optimum at **365 days**, unanimous at both ends and **independently selected by
validation**:

| Window | 90d | 180d | 270d | **365d** | 547d | 730d | 1095d *(old default)* |
|---|---|---|---|---|---|---|---|
| 2hop − none | −0.0029 | −0.0022 | +0.0012 | **+0.0029** | +0.0020 | −0.0001 | −0.0043 |
| seeds | 0/5 | 2/5 | 3/5 | **5/5** | 5/5 | 3/5 | 0/5 |

It replicates on the 1990 scope: 365d beats 547d at one hop by −0.00213
(t=−12.22, 5/5). The proposed mechanism — that `sum` aggregation lets stale
edges' count overwhelm their recency decay — **was tested and refuted**: `mean`
aggregation recovered nothing at 1095d (t=−0.41) and was unanimously worse at
365d (+0.0024, t=5.42, 0/5). The window effect is solid; the explanation is open.

## 8. Baselines and blending

`slams_masters_1990`, five seeds. Standalone ratings have zero seed variance —
`sigmoid(scale·Δ)` is exactly antisymmetric.

| Model | Log loss | Brier | Accuracy |
|---|---:|---:|---:|
| **GNN (final)** | **0.56766** | 0.19370 | 0.7009 |
| Blend (GNN + Elo, weights fitted on validation) | 0.57124 | 0.19412 | 0.6972 |
| Elo (standalone) | 0.58050 | 0.19852 | 0.6867 |
| B-score (standalone) | 0.60729 | 0.20633 | 0.6885 |

GNN vs Elo −0.01284 (5/5); vs B-score −0.03964 (5/5). Two consequences:

- **Elo beats B-score as a standalone rating on our data**, the reverse of the
  published claim — but B-score is the better *input* to the GNN (0.5930 vs
  0.5972 at the tuned `slams_masters` config, t=9.61, 0/5 seeds for Elo).
  B-score is itself graph-derived (eigenvector centrality over the win/loss
  network), so it composes with message passing; Elo is a sequential update with
  no relational structure in it.
- **Blending stopped helping once the graph was fixed.** It was worth −0.0020
  (t=−8.70, 5/5) on the default model and −0.0003 (t=−1.67, 4/5) on the tuned
  one, with validation weight shifting 0.74 → 0.81 toward the GNN. Blending
  standalone B-score into anything always gets weight **0.00** — the GNN already
  contains it.

## 9. Where the graph helps, and where it does not

Tier 3, `full`, one hop vs none, by how many prior opponents the thinner-recorded
player has:

| Stratum | Matches | Δ | 95% CI | Seeds |
|---|---:|---:|---|---:|
| 0–5 (cold) | 1,404 | **+0.01426** | [+0.00357, +0.02496] | 0/5 |
| 6–20 | 2,449 | +0.00010 | [−0.00098, +0.00118] | 2/5 |
| 21+ | 10,487 | **−0.00129** | [−0.00241, −0.00016] | 5/5 |

(`new_work/results/strata_full.csv`, run at the carried-over lr 3e-4. Re-run at
the stable lr 1e-4 the two significant cells shrink to **+0.00413** (0/5) and
**−0.00229** (5/5) — the mistuned recipe exaggerated the size but not the
direction.)

**The cold-start hypothesis was tested and it failed.** At Slams+Masters the cold
stratum favoured message passing (−0.0048, 5/5); at tour scope the sign is
reversed and unanimous. A cold node has no neighbourhood to aggregate; one hop
over two or three edges cannot manufacture a skill estimate, it only dilutes the
B-score prior. So the graph **does not substitute for a missing per-player
history — it amplifies a present one.** The dose-response by shared opponents is
monotone in the same direction (+0.0173 at zero common opponents → −0.0013 at
15+), and the two cuts are strongly correlated, so this is probably one
phenomenon seen twice.

At tier 1 (B-score + static only), one hop helps in *every* stratum including
cold start, 3/3 seeds each. The substitution story belongs to the low-feature
regime, not to cold start.

**Intransitivity is a null.** Whether a second hop is worth more on matches with
no prior meeting but a shared opponent: at Slams+Masters this looked real across
three tiers (−0.0115 at tier 1, interval clear of zero); at full scope with both
arms on a stable recipe it is **−0.0001, CI [−0.00278, +0.00259]**. A tight null,
not a wide shrug. The full-scope interval excludes the entire Slams+Masters
interval — this is a refutation, not a failure to replicate.

**24 stratum comparisons were computed with no multiplicity correction.** The
degree cut and the `two_hop_only` contrast were pre-specified; the rest are
descriptive.

## 10. What did not work, and the pattern in it

- The direct-logit skip connection is arithmetically starved — `bscore_scale`
  starts at 1.0 and needs ~15, and Adam can move it 0.13 in 1,344 steps at
  lr 1e-4. **Every fix made it worse**: `logz_surface` +0.0034, `logz` +0.0055,
  `nodenorm` +0.0065, `scale15` +0.0080, `log` +0.0576; 24 of 25 arm-seeds lost,
  and control won on validation too. The arithmetic was right; the inference was
  wrong. The model does not want a strong fixed prior.
- Node features reach the encoder unnormalised, and **height accounts for 99.41%**
  of the encoder's input against 0.12% for four B-score channels. Standardising
  them on frozen training constants failed outright, every arm, every seed
  (`scaled` +0.0128, `scaled_history_nodes` +0.0171). B-score reaches the model
  through the skip connection, not the encoder, so a height-dominated encoder is
  harmless — the architecture routes around it, and standardising just makes
  useless columns shout equally.
- 3 hops over 2 (±0.0004, t<1), and `mean` aggregation as the staleness fix (§7).

Every fix reasoned from the architecture's internals failed — roughly a dozen
attempts, zero wins. What worked came from published evidence (blending) or a
hypothesis about the *data*: history routing, and the staleness fix, which came
from asking whether the graph was carrying old low-value matches around.
**The architecture was not broken in the ways that looked mechanically obvious;
what was fed into it was.**

## 11. Limitations

- **Absolute numbers are not comparable across scopes.** Only within-scope
  contrasts are interpretable.
- **Seeds are uneven by tier.** Tier 3 has five; tiers 0–2 have three at `full`,
  so their intervals use t=4.303. Every hop contrast is still within-tier and
  one-factor — less precise, not invalid. The grid CSVs carry `n_seeds`.
- **The `full`-scope tuning search was abandoned on cost** (161 ms/step against
  64 ms, ~12.5× compute; the 21-configuration grid costs 6.2 h). Tier 3's recipe
  was carried over from the smaller scope and confirmed by a single full-scope
  validation measurement. §6 shows how much that can matter.
- **The 1990 scope has no GBDT baseline.** `gbdt_comparison/results/full_1990/`
  holds one seed of a *different* scope (11,830 rows) and is not the comparison
  partner for the 6,995-match GNN artifacts. The 1990 scope is used here to
  replicate the hop and window effects, not to re-run the headline.
- **Tier 0 was reused from tier 1's recipe.** Its absolute level carries that
  caveat; the hop contrast within it is still clean.
- The B-score fallback: coverage is 99.1% in train, validation and test; the
  whole gap is warm-up years, which are never scored.

## 12. Reproducing

```bash
# The final model, one seed
conda run -n tennis-gnn python tennis_gnn/run.py --experiments base --seeds 42

# All one-factor ablations, five seeds
conda run -n tennis-gnn python tennis_gnn/run.py --experiments all \
  --seeds 42 123 456 789 2026 --summary results/ablations.csv

# The headline contrast against the GBDT, from frozen artifacts
conda run -n tennis-gnn python tennis_gnn/compare.py --scope full \
  --seeds 42 123 456 789 2026 --candidate base --baseline gbdt_tuned

# Substitution curve and strata (reads artifacts, retrains nothing)
python new_work/feature_hop_grid.py --scope full
python new_work/cold_start.py --scope full
```

`tennis_gnn/verify_targets.py` checks that a rebuilt evaluation set still matches
the frozen artifacts hash for hash. Run it after touching the data pipeline: if
it fails, predictions are no longer comparable with anything above.

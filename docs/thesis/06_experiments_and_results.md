# Chapter 6 — Experiments and Results

Six experiments, in the order the outline lists them. Every number below is a
mean over seeds of values that already exist on disk under
`results/frozen_predictions/`, `gnn_improvements/results/` or
`new_work/results/`; nothing was retrained to write this chapter. Negative
deltas mean the first-named model is better (log loss). "5/5" is per-seed
unanimity.

**Within-scope contrasts only.** The three scopes are different problems of
different difficulty; absolute levels are not comparable across them.

**Which artifacts, at `full` scope.** Two tier-3 families exist and they are not
interchangeable:

| Family | Window | Recipe | Produced by | Used for |
|---|---|---|---|---|
| `depth_3_history_decoder_{none,1,2}hop_lr1e4` | default 1095 d | lr 1e-4 | `new_work/depth_test.py` | the headline vs GBDT, the hop contrasts, strata re-runs |
| `win365_t3_history_decoder_{none,1hop_sum,2hop}` | 365 d | lr 1e-4 | `gnn_improvements/window_sweep.py` | the "final model" row (0.61919) |

`FINAL_RESULTS.md` presents the headline as the 365-day final model, but its
numbers come from the 1095-day family. Both are given below where it matters,
recomputed from the frozen prediction CSVs.

---

## 6.1 Baseline comparison: Elo, B-score, GBDT, GNN

`slams_masters_1990`, five seeds, from `gnn_improvements/results/baselines_slams_masters_1990.csv`
and `.../blend_slams_masters_1990.csv`. The GNN row is the 365-day, one-hop,
tier-3 artifact. Standalone ratings have exactly zero
seed variance — `sigmoid(scale · Δ)` is antisymmetric by construction.

| Model | Log loss | Brier | Accuracy |
|---|---:|---:|---:|
| **GNN (final)** | **0.56766** | 0.19370 | 0.7009 |
| Blend (GNN + Elo, weights fitted on validation) | 0.57124 | 0.19412 | 0.6972 |
| Elo (standalone) | 0.58050 | 0.19852 | 0.6867 |
| B-score (standalone) | 0.60729 | 0.20633 | 0.6885 |

GNN vs Elo **−0.01284 (5/5)**; GNN vs B-score **−0.03964 (5/5)**.

Two consequences worth stating separately:

- **Elo beats B-score as a standalone rating on our data** — the reverse of the
  published claim — **but B-score is the better *input* to the GNN**
  (0.5930 vs 0.5972 at the tuned `slams_masters` config, t=9.61, 0/5 seeds for
  Elo). B-score is itself graph-derived (eigenvector centrality over the
  win/loss network), so it composes with message passing; Elo is a sequential
  update with no relational structure in it.
- **Blending stopped helping once the graph was fixed.** At `slams_masters`
  (`gnn_improvements/FINDINGS.md` §6) a validation-weighted Elo blend was worth
  −0.0020 (t=−8.70, 5/5) on the default model and −0.0003 (t=−1.67, 4/5) on the
  tuned one (0.5930), with the validation weight shifting 0.74 → 0.81 toward the
  GNN. On `slams_masters_1990` the blend in the table is *worse* than the GNN
  alone. Blending
  standalone B-score into anything always gets weight **0.00** — the GNN already
  contains it.

The baseline artifacts are produced by `gnn_improvements/baselines.py`, which
deliberately reuses the GNN's own `load_or_build` snapshots so keys and labels
are identical *by construction* rather than by luck.

---

## 6.2 Effect of graph freshness

The largest single fix at `slams_masters`, and it came from a question about the
data rather than about the architecture.

Measured on `slams_masters` prediction blocks at the old 1095-day window, **65–81% of edges
and 25–36% of nodes were more than a year old**, and it worsened over time
(65% → 67% → 81% across 2016 / 2018 / 2020). The model was therefore *tested* on
a staler graph than it *trained* on.

Sweeping `window_days` at tier 3, **`slams_masters` scope, two hops against no
messages** (`gnn_improvements/window_sweep.py`,
`gnn_improvements/results/window_hop_all.csv`), gives a clean interior optimum at
**365 days**, unanimous at both ends and **independently selected by
validation**. The row is the log loss *saved* by two hops (no-messages minus
two-hop), so **positive means the graph helps**; "seeds" counts seeds where two
hops beat no messages:

| Window | 90d | 180d | 270d | **365d** | 547d | 730d | 1095d *(old default)* |
|---|---|---|---|---|---|---|---|
| none − 2hop | −0.0029 | −0.0022 | +0.0012 | **+0.0029** | +0.0020 | −0.0001 | −0.0043 |
| seeds | 0/5 | 2/5 | 3/5 | **5/5** | 5/5 | 3/5 | 0/5 |
| 2-hop test log loss | 0.6000 | 0.5992 | 0.5959 | **0.5942** | 0.5950 | 0.5971 | 0.6011 |

`FINAL_RESULTS.md` §7 and `FINDINGS.md` §2 label this row "2hop − none"; the CSV
shows the sign is the other way round (at 365 d: no messages 0.59714, two hops
0.59420). The no-message arm is flat at ≈0.5970 across every window, as it should
be, since without messages the window cannot reach the model.

It replicates on the 1990 scope: at one hop, 365 d beats 547 d by
−0.00212 (0.56766 vs 0.56978, 5/5) and beats the 1095-day hop-grid model
(0.56908).

**It does not show up at `full` at one hop.** The one-hop tier-3 model scores
0.61919 at 365 d and 0.61915 at 1095 d. The freshness effect is established for
two hops at `slams_masters` and for one hop at `slams_masters_1990`, not for the
`full` final model.

**The proposed mechanism was tested and refuted.** The hypothesis was that `sum`
aggregation lets the sheer count of stale edges overwhelm their recency decay.
`mean` aggregation recovered nothing at 1095d (t=−0.41) and was unanimously
worse at 365d (+0.0024, t=5.42, 0/5). **The window effect is solid; the
explanation is open.**

Note that `window_days` deliberately does *not* touch the history statistics —
`HistoryTracker` keeps its own 3-year window — so sweeping the window is a clean
one-factor test of graph *contents*.

---

## 6.3 No graph vs one hop vs two hops

Tier 3, `full` scope, lr 1e-4, five seeds, log loss. "Seeds" counts seeds where
the first-named arm is better.

| Contrast | Window | Δ | 95% CI | Seeds |
|---|---|---:|---|---:|
| 1 hop vs no messages | 1095 d (`depth_3_*`) | **−0.00100** | [−0.00179, −0.00022] | 5/5 |
| 1 hop vs no messages | 365 d (`win365_*`) | −0.00095 | [−0.00260, +0.00070] | 4/5 |
| 2 hops vs 1 hop | 1095 d | **+0.00942** | [+0.00560, +0.01324] | 0/5 |
| 2 hops vs 1 hop | 365 d | **+0.00550** | [+0.00269, +0.00830] | 0/5 |

The one-hop gain has the same size at both windows, but it is only significant
at 1095 d. At the 365-day final configuration its interval includes zero. The
second hop hurts unanimously at both.

`gain_2_vs_1` is positive at every tier at `full` scope, where it was negative
at tiers 0–1 at `slams_masters`. **The one-hop substitution effect replicates
across scopes; every second-hop effect reverses sign when the population
changes.** The final model therefore uses **one hop** at `full`. The tuned
`slams_masters` configuration keeps two, which is itself part of the evidence
that depth does not transfer.

**Two hops also exposed the recipe trap, and this is methodologically the most
important paragraph in the chapter.** The grid's `gain_2_vs_1 = +0.02368` was
about 60% learning rate: rerunning the identical 2-hop model changing only
3e-4 → 1e-4 recovered 0.0146 of it
(`new_work/results/twohop_diagnostic.csv`). The no-message arm barely notices
the recipe (−0.00008) while message-passing arms gain ≈0.0015 — a model that
ignores its edges has less to optimise — so a shared recipe biases
**systematically against the arm under test**. Correcting this is what corrected
the headline: the best model *does* use the graph.

Three hops over two, at `slams_masters`, is ±0.0004, t<1 — nothing.

---

## 6.4 Graph contribution at different amounts of engineered history

**The main finding.** `gain_1_vs_none` — what one hop is worth — as a function
of how much per-player information the model already has
(`new_work/feature_hop_grid.py`, results in
`new_work/results/feature_hop_grid_<scope>.csv`). Negative means the graph
helped; seed counts in parentheses.

| Tier | `slams_masters` | `full` | `slams_masters_1990` |
|---|---:|---:|---:|
| 0 — no B-score, no history | −0.0813 (5) | −0.0617 (3) | −0.1166 (5) |
| 1 — B-score + static | −0.0142 (5) | −0.0136 (3) | −0.0323 (5) |
| 2 — + history on nodes | +0.0069 (5) | −0.0012 (3) | +0.0023 (5) |
| 3 — + history at decoder | +0.0000 (5) | **−0.0010** (5)\* | **−0.0010** (5) |

Sources: `new_work/results/feature_hop_grid_{slams_masters,full}.csv` and
`gnn_improvements/results/hop_grid_slams_masters_1990.csv`, all at the default
1095-day window.

\* **The `full` tier-3 cell is not the grid's value.** The grid CSV records
+0.00047 for that cell, from the lr 3e-4 recipe that handicaps message-passing
arms (§6.3). The −0.0010 in the table is the stable-recipe re-run
(`depth_3_*_lr1e4`), substituted as in `FINAL_RESULTS.md` §5. So the tier-3 row
mixes recipes across scopes, and the table should say so wherever it appears.

**The graph's value is a decreasing function of how well the model is already
informed about the two players — about sixty-fold from tier 0 to tier 3 — and at
two of three scopes it does not reach zero.** At tier 3 the one-hop gain is
−0.00100 on `full` (CI [−0.00179, −0.00022], 5/5, at 1095 d) and −0.00104 on the
1990 scope (t=−9.53, 5/5, at 1095 d). At 365 d it is −0.00095 on
`full` (4/5, interval includes zero) and −0.00242 on the 1990 scope (0.56766 vs
0.57008, CI [−0.00279, −0.00205], 5/5). At `slams_masters` it is zero at both windows. Two different tournament populations, two different decades of
history, the same small real effect.

**The left endpoint, labelled honestly.** At tier 0 the no-message model scores
**exactly 0.693147** — the coin flip — under every recipe tried (4× steps, 10×
lr, 3 passes all identical, logit sd ≤1e-8), because height and handedness carry
nothing and the antisymmetric decoder represents "no difference" exactly. One
hop on the same features reaches 0.6119 / 0.6314 / 0.5766. There **the graph is
the only feature there is**, so that row measures the graph against an
uninformative baseline and must be read as such, not as a 0.08 improvement over
a real model.

Tier 2 vs tier 3 is the routing result: putting the same twelve statistics on
the *nodes* lets message passing smooth them away, and the graph's marginal
value turns positive (i.e. harmful) at two scopes out of three. Routing them to
the decoder keeps each player's own record undiluted while the graph still does
its own work.

---

## 6.5 Performance on sparse vs well-connected players

Tier 3, `full`, one hop vs none, stratified by how many prior opponents the
**thinner-recorded** player has (`new_work/cold_start.py` →
`new_work/results/strata_full.csv`):

| Stratum | Matches | Δ | 95% CI | Seeds |
|---|---:|---:|---|---:|
| 0–5 (cold) | 1,404 | **+0.01426** | [+0.00357, +0.02496] | 0/5 |
| 6–20 | 2,449 | +0.00010 | [−0.00098, +0.00118] | 2/5 |
| 21+ | 10,487 | **−0.00129** | [−0.00241, −0.00016] | 5/5 |

(Run at the carried-over lr 3e-4. Re-run at the stable lr 1e-4 the two
significant cells shrink to **+0.00413** (0/5) and **−0.00229** (5/5) — the
mistuned recipe exaggerated the size but not the direction.)

**The cold-start hypothesis was tested and it failed.** At `slams_masters` the
cold stratum favoured message passing (−0.0048, 5/5); at tour scope the sign is
reversed and unanimous. A cold node has no neighbourhood to aggregate: one hop
over two or three edges cannot manufacture a skill estimate, it only dilutes the
B-score prior. So **the graph does not substitute for a missing per-player
history — it amplifies a present one.**

The dose-response by shared opponents is monotone in the same direction
(+0.0173 at zero common opponents → −0.0013 at 15+), and the two cuts are
strongly correlated, so this is probably one phenomenon observed twice.

At tier 1 (B-score + static only), one hop helps in *every* stratum including
cold start, 3/3 seeds each. **The substitution story belongs to the low-feature
regime, not to cold start.**

**Intransitivity is a null.** Whether a second hop is worth more on matches with
no prior meeting but a shared opponent: at `slams_masters` this looked real
across three tiers (−0.0115 at tier 1, interval clear of zero); at `full` scope
with both arms on a stable recipe it is **−0.0001, CI [−0.00278, +0.00259]**. A
tight null, not a wide shrug — and the full-scope interval excludes the entire
`slams_masters` interval, so this is a refutation rather than a failure to
replicate.

**Multiplicity, stated rather than buried.** 24 stratum comparisons were
computed with no multiplicity correction. The degree cut and the `two_hop_only`
contrast were pre-specified; the rest are descriptive.

---

## 6.6 Final GNN vs tuned GBDT

**The final model.** Tier 3 features (B-score + static attributes + per-player
history routed to the **decoder**), **365-day graph window**, sum aggregation,
lr 1e-4, 4 steps per block, mini-batch 32 block graphs, one pass. One hop with a
replay buffer of 200 at `full` and `slams_masters_1990`; two hops with a buffer
of 800 at `slams_masters`.

| Scope | Test log loss | Brier | Accuracy |
|---|---:|---:|---:|
| `full` (n=14,340, 1 hop) | **0.61919** | 0.21585 | 0.6448 |
| `slams_masters` (n=3,770, 2 hops, buffer 800) | **0.59301** | 0.20452 | 0.6720 |
| `slams_masters_1990` (n=6,995, 1 hop) | **0.56766** | 0.19370 | 0.7009 |

**The headline.** `full` scope, tier 3 / 1 hop against the tuned
`HistGradientBoostingClassifier` (test log loss 0.62487) on the same matches and
the same labels, paired over five seeds. The GBDT's matches and labels are
guaranteed identical by the `evaluation_hash` check.

| Metric | Δ at 1095 d (`depth_3_…_1hop_lr1e4`, `headline_full.csv`) | Δ at 365 d (final model) | Seeds |
|---|---:|---:|---:|
| log loss | **−0.00572** [−0.00719, −0.00426] | **−0.00568** [−0.00649, −0.00486] | 5/5, 5/5 |
| Brier | **−0.00258** [−0.00321, −0.00195] | **−0.00249** [−0.00284, −0.00214] | 5/5, 5/5 |
| accuracy | **+0.00448** [+0.00165, +0.00731] | **+0.00343** [+0.00138, +0.00548] | 5/5, 5/5 |

The headline does not depend on the window: both one-hop tier-3 models beat the
GBDT on every metric in every seed.

At `slams_masters` the same contrast wins on probability quality (log loss
−0.00653, 5/5; Brier −0.00237, 5/5) and **ties on accuracy** (+0.0014, 3/5, CI
includes zero). **Tripling the data turns the tie into a win.**

For contrast, the earlier state of this comparison is preserved in
`gbdt_comparison/RESULTS.md`, where the GBDT beat the then-best GNN (the
antisymmetric decoder) by 0.0077 log loss at `slams_masters`. What changed
between the two is recorded in `gnn_improvements/FINDINGS.md` and
`new_work/STATUS.md`: per-configuration tuning, routing history to the decoder,
the graph window, the replay buffer, and at `full` one hop instead of two. The
repository has no single ablation that attributes the reversal among them.

---

## 6.7 What did not work, and the pattern in it

Reported because the pattern is itself a result.

- **The direct-logit skip connection is arithmetically starved.**
  `bscore_scale` starts at 1.0 and needs ≈15 (median |b_a − b_b| = 3.2e-03), and
  Adam can move it 0.13 in 1,344 steps at lr 1e-4. **Every fix made it worse**:
  `logz_surface` +0.0034, `logz` +0.0055, `nodenorm` +0.0065, `scale15` +0.0080,
  `log` +0.0576; 24 of 25 arm-seeds lost, and the control won on validation too.
  The arithmetic was right; the inference was wrong. **The model does not want a
  strong fixed prior.**
- **Node features reach the encoder unnormalised, and height accounts for
  99.41%** of the encoder's input against 0.12% for four B-score channels.
  Standardising on frozen training constants failed outright, every arm, every
  seed (`scaled` +0.0128, `scaled_history_nodes` +0.0171). B-score reaches the
  model through the skip connection, not the encoder, so a height-dominated
  encoder is harmless — the architecture routes around it, and standardising
  just makes useless columns shout equally.
- 3 hops over 2 at `slams_masters` (±0.0004, t<1), and `mean` aggregation as
  the staleness fix.
- **A flag that was never wired.** `tournament_context_edges` has an ablation
  entry, but nothing passes it to the edge builder (chapter 4 §4.3). Any result
  under that name is `base` run again.

**Every fix reasoned from the architecture's internals failed — roughly a dozen
attempts, zero wins. What worked came from published evidence (blending) or from
a hypothesis about the *data*: history routing, and the staleness fix. The
architecture was not broken in the ways that looked mechanically obvious; what
was fed into it was.**

---

## 6.8 Reproducing the tables

```bash
# §6.6 final model at full scope (365 d, tier 3, one hop) and its hop arms
python gnn_improvements/window_sweep.py --scope full --windows 365 \
  --tiers t3_history_decoder --hops none 1 2

# §6.3 stable-recipe depth arms at the default window (headline_full.csv source)
python new_work/depth_test.py --scope full --hops none 1 2

# §6.2 window sweep at slams_masters
python gnn_improvements/window_sweep.py --scope slams_masters \
  --tiers t3_history_decoder --windows 90 180 270 365 547 730 1095 --hops none 2

# §6.6 headline, from frozen artifacts
conda run -n tennis-gnn python tennis_gnn/compare.py --scope full \
  --seeds 42 123 456 789 2026 \
  --candidate depth_3_history_decoder_1hop_lr1e4 --baseline gbdt_tuned

# §6.4 substitution curve (trains any missing grid cells; --summary-only just
# reads artifacts) and §6.5 strata + headline (reads artifacts)
python new_work/feature_hop_grid.py --scope full --summary-only
python new_work/cold_start.py --scope full

# §6.1 baselines and blend
python gnn_improvements/baselines.py --scope slams_masters_1990
python gnn_improvements/blend.py --scope slams_masters_1990 --members <artifact> <artifact>
```

`run.py --experiments base` trains `BASE_MODEL` (two hops, 1095 d, no history),
which is not the final model.

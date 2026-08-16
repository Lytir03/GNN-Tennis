# Inconclusive and superseded

Nothing here is deleted and nothing here is wrong to have done. This folder holds
work whose **results do not support a conclusion**, either because the experiment
was not comparable, or because a later control showed it was measuring something
other than its title.

Kept because a thesis needs to show what was tried. Separated because leaving it
beside the live work would imply it still carries evidence.

---

## 1. `temporal_gnn_intransitivity/` — not comparable, queued for re-run

**Status: parked, scheduled for revival.** This is not a dead end; it is a result
that was never measurable.

The experiment reports accuracy 0.619 against the static GNN's 0.671 and concludes
temporal modelling underperforms. **That conclusion cannot be drawn from these
artifacts**, because the two models were never scored on the same matches:

| | static GNN | temporal GNN |
|---|---|---|
| `evaluation_hash` | `575e4dc2…` | `2e11d43e…` |
| `block_idx` range | 504–794 | 924–1214 |
| orientation draw | one sequence | a different sequence |

An inner join on the match keys returns **0 rows**. The published comparison is
between two numbers computed on disjoint data. `tennis_gnn/compare.py` now refuses
this automatically.

It is also confounded twice over: single-hop (less capacity than the 2-layer static
model) and untuned ("nessun tuning: 8 epoche" per its own README), while the GBDT got
32 tuned configurations.

**Why it is now more interesting, not less.** The main finding is that recency-weighted
per-player *history* is what carries the signal, and that message passing was only ever
a substitute for it. A recurrent per-player state is a different route to exactly that
quantity. This model may be right for a reason nobody has stated. Re-run it through
`tennis_gnn/` so it emits artifacts with a matching `evaluation_hash`, then it is
quotable in either direction.

Paths were changed from `parents[1]` to `parents[2]` when it moved down a level; its
6 tests pass here.

---

## 2. `REPO_REVIEW.md` — first-pass review, section 5 was wrong

The original walkthrough. Superseded by `TUTOR_REPORT.md`, kept because it records what
the repository looked like before consolidation.

Its §5 claimed the ablations "validated" the design choices. They did not: the ablations
were **cumulative**, so every row inherited `node_normalization`'s 0.075 log-loss damage
and none measured the factor in its own title. The correction is in the file.

---

## 3. Superseded frozen artifacts

These stay in `results/frozen_predictions/` because the comparison tooling reads that
directory, but they carry no current evidence:

| artifact | why superseded |
|---|---|
| `mean_aggregation`, `node_normalization`, `residual_connections`, `tournament_context`, `antisymmetric_mean*`, `antisymmetric_straight_sets` | built cumulatively; each inherits every change above it, so none is a one-factor measurement |
| `temporal_gnn`, `temporal_gnn_intransitivity` | different `evaluation_hash`; not comparable to anything |
| `current_gnn`, `bscore_residual` | pre-consolidation baselines, retained only to check the refactor reproduced them |

The live results are in `TUTOR_REPORT.md` §10.

---

## 4. Recommended for the appendix, not the argument

Not moved, because they are still valid one-factor measurements — but they are
**second-order** next to the feature-richness / hop-count axis, and spending thesis
space on them buys little:

- **GINE vs GATv2** — GINE wins (0.612 vs 0.615), but both are ~0.01 behind the
  decoder-routed model. The convolution choice is not where the variance lives.
- **Edge-feature presets** (`edge_full`, `edge_signed_game`, `edge_set_margin`,
  `edge_straight_sets`, `edge_match_status`) — `full` is best, but the zero-hop control
  shows message passing contributes nothing once history is supplied, which caps how
  much any edge-feature refinement can matter.

Report as an appendix table. The argument belongs to substitution and cold start.

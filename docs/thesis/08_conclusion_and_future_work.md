# Chapter 8 — Conclusion and Future Work

---

## 8.1 Summary of findings

1. **A GNN over the match graph beats a tuned gradient-boosted tree given the
   same information.** `full` scope, five paired seeds, same matches and same
   labels: log loss −0.00572 (5/5), Brier −0.00258 (5/5), accuracy +0.00448
   (5/5), and log loss −0.00568 (5/5) at the 365-day window. At `slams_masters`
   the same contrast wins on probability quality and ties on accuracy; tripling
   the data turns the tie into a win.

2. **The graph's marginal value is a decreasing function of how well the model
   is already informed about the two players — about sixty-fold from tier 0 to
   tier 3.** At tier 3, one hop is worth −0.00100 on `full` (5/5) and −0.00104
   on the 1990 scope (5/5) at the default window; at the final 365-day window it
   is −0.00095 on `full` (4/5, not significant) and −0.00242 on the 1990 scope
   (5/5);
   at `slams_masters` it is zero.

3. **Graph freshness matters where depth is used.** At the old 1095-day window,
   65–81% of edges were over a year old on `slams_masters` prediction blocks,
   worsening over time. For two hops at that scope a 365-day window is a clean
   interior optimum, unanimous at both ends and independently selected by
   validation; for one hop it replicates on the 1990 scope. At `full`, the
   one-hop model is unchanged by the window (0.61919 vs 0.61915).

4. **One hop, not two.** The one-hop effect replicates across scopes; every
   second-hop effect reverses sign when the population changes. Intransitivity —
   the natural mechanism for a second hop — is a tight null at `full`
   (−0.0001, CI [−0.00278, +0.00259]), and that interval excludes the entire
   `slams_masters` interval.

5. **Routing matters as much as the features.** The same twelve history
   statistics help at the decoder and hurt on the nodes, because message passing
   smooths away exactly the per-player sharpness they contain.

6. **The graph amplifies a present history, it does not substitute for a missing
   one.** It helps unanimously where a player has 21+ prior opponents and hurts
   unanimously where they have 0–5. The cold-start hypothesis was tested and
   failed.

7. **Elo beats B-score as a standalone rating on our data — and B-score is the
   better input to the GNN.** B-score is itself graph-derived, so it composes
   with message passing; Elo is a sequential update with no relational structure
   in it. Blending, worth −0.0020 on the untuned model, is worth −0.0003 once
   the graph is fixed.

8. **A methodological finding, reported as one.** Roughly a dozen fixes reasoned
   from the architecture's internals produced zero wins; both real gains came
   from hypotheses about the data. And a shared training recipe biases
   systematically against message-passing arms, which is what had previously
   made this project's own headline come out backwards.

---

## 8.2 The answer to the research question

> *Does graph structure add predictive information beyond standard historical
> features?*

**Yes, and the interesting part is how little, and where.**

A GNN over the match graph extracts information that a tuned tabular learner
cannot extract from the same features, unanimously across seeds. The part of that
attributable to message passing itself is measurable at `full` and on a second
tournament population, though at `full` only at one of the two windows tested. But
the contribution shrinks by roughly sixty-fold as hand-engineered per-player
history is added, which means **engineered history and one-hop message passing
are largely two encodings of the same object**. What survives the substitution is
small: about 0.16% of the log loss for one hop at the richest feature tier.

It is also **not** the thing the graph was expected to be good at. The residual
contribution appears where records are already thick, not where they are thin;
it comes from one hop, not from the relational depth that would justify the
representation on theoretical grounds; and the shared-opponent effect that would
be the clearest evidence of genuine relational reasoning is a tight null.

So the defensible conclusion is narrow and it is positive: **a GNN over the
tennis match graph extracts more from the same information than a tuned tabular
learner; the part attributable to message passing is small, one-hop, reproduced
on two of three scopes, and concentrated in well-connected players.**

---

## 8.3 Limitations

Restated in brief; full versions in chapter 7 §7.5 and `FINAL_RESULTS.md` §11.

- Effect sizes are small — a few thousandths of log loss — and the conclusions
  depend on the pairing, unanimity and hash-equality guards that make numbers of
  that size interpretable.
- Absolute performance is not comparable across scopes; only within-scope
  contrasts are.
- The `full`-scope hyperparameter search was abandoned on cost; tier 3's recipe
  was carried over and confirmed by one validation measurement. Chapter 7 §7.3 shows how
  much a recipe can matter.
- Tiers 0–2 run at three seeds at `full`, with intervals accordingly wider.
- The 1990 scope has no matching GBDT baseline and is used only to replicate the
  hop and window effects.
- 24 stratum comparisons, no multiplicity correction; only two were
  pre-specified.
- The staleness effect is established and its mechanism is not; the `sum`-versus-
  recency explanation was tested and refuted.
- At `full`, the headline and hop contrasts were computed on 1095-day artifacts
  and the "final model" is a 365-day artifact; the one-hop gain is significant at
  the former only.
- Implementation gaps: `tournament_context_edges` is not wired to the edge
  builder, and the newcomer B-score fallback differs between the GNN (median) and
  the GBDT (25th percentile).
- The whole study is one sport, one tour, one dataset lineage (Sackmann ATP),
  and one graph construction.

---

## 8.4 Future work

**Larger and different datasets.** Every scope widening in this project changed
a conclusion — the accuracy tie became a win, the cold-start result reversed,
the intransitivity effect vanished. WTA, Challengers, and a full 1990-onwards
tour build are the obvious next populations, and the value of running them is
precisely that they can overturn things. The tooling is already
scope-parameterised: `tennis_gnn/data.py` `SCOPE_FILES`,
`new_work/build_bscore.py --scope`, and `extended_history_1990/`.

**Temporal GNNs.** The current design rebuilds a static graph per block and
throws the previous one away; recency enters only as an edge weight and a window
cutoff. Architectures with explicit time — temporal message passing, continuous
time encodings, recurrent node states — target exactly the quantity the window
sweep showed to be decisive. Given that the staleness effect is large and its
mechanism is open, this is the most promising direction.

**Better graph construction.** The window sweep says *which* matches belong in
the graph matters more than anything tried on the architecture. Untested
variants: per-player adaptive windows, edges weighted by tournament level,
surface-specific subgraphs, dropping edges between players whose records have
diverged, and node-level rather than edge-level recency.

**Closing the artifact gap at `full`.** Re-run the stratified analysis and the
hop contrast at five seeds on the 365-day arms, so the final model, the headline
and the substitution curve all refer to the same configuration. It needs no new
code: `window_sweep.py` already produced the arms, and `cold_start.py` reads
artifacts (its artifact names would need pointing at the `win365_*` files).

**Finishing the tuning.** The `full`-scope 21-configuration grid was abandoned
at 6.2 h. Running it — and running a per-tier search at five seeds for tiers 0–2
— would remove the single largest caveat on the substitution curve.

**Mechanism for the freshness effect.** `mean` aggregation is refuted as the
explanation. Remaining candidates: a distribution shift in the graph itself over
the test years, an interaction between window length and the B-score's own
365-day decay, or edge-count heterogeneity across players. Each is measurable on
existing snapshots without retraining.

**A cold-start pathway, since the graph is not one.** Chapter 6 §6.5 shows one hop
actively hurts thin records. A model that routes around the graph when the
neighbourhood is small — or that learns when to trust it — would address the
only stratum where the current model is unanimously worse.

---

## 8.5 What the repository leaves behind

- `results/frozen_predictions/` — per-match predictions and manifests for every
  model, scope and seed. The durable record; every table in these chapters is
  read from it.
- `tennis_gnn/verify_targets.py` — rebuilds the evaluation set and checks it
  matches the frozen artifacts hash for hash. If it fails, nothing above is
  comparable any more.
- `FINAL_RESULTS.md` — the settled account.
- `gnn_improvements/FINDINGS.md`, `new_work/STATUS.md` — the working narrative,
  including the retractions, written as retractions rather than quiet
  amendments.

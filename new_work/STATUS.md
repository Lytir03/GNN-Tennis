# New work — where we actually are

Live work. Two tasks were requested here; **one is done, one is blocked** on a
dependency I did not want to rewrite without telling you.

---

## Task A — Fill the feature × hop grid ✅ running / done

**Purpose.** The central claim is that graph structure *substitutes* for per-player
history rather than adding to it. A single win/loss cannot show that. A grid can: hold
the hop count fixed and vary how much per-player information the model gets, and the
graph's marginal value should decay to zero — and past it — as the features improve.

The hop axis is the intervention (within one model family, everything else fixed). The
feature axis is the moderator. The cell-to-cell differences are the thesis figure.

Run: `python new_work/feature_hop_grid.py` → `new_work/results/feature_hop_grid.csv`

**Recipe provenance matters and is recorded per tier.** Each feature tier got its own
validation-only search (`tennis_gnn/tune.py --model ...`), because judging an
18-dimensional model on a recipe selected for a 6-dimensional one is the same
unfairness this project removed from the GBDT comparison. Tier 0 reuses tier 1's
recipe and is flagged as such in the output — read its absolute level with that
caveat, but the *hop* contrast within tier 0 is still clean, because the recipe is
held fixed across the hop axis.

---

## Task B — Expand the data ⛔ blocked on the B-score pipeline

**This is not a "point it at a bigger file" change, and I want to be exact about why.**

The prize is real. Raw Sackmann data, 2011–2020:

| | matches |
|---|---:|
| currently used (Slams + Masters) | 10,212 |
| **available in raw, all ATP** | **27,829** |

A 2.7× expansion. It would fix the noise floor that undermines every selection decision
in this project (validation SE ≈ 0.018 against effects of ~0.004) *and* multiply the
cold-start cases that carry the positive finding.

**The blocker: B-score snapshots do not exist for most of those matches.**

| tourney level | B-score covered | not covered |
|---|---:|---:|
| A (ATP 250/500) | 2,324 | **12,522** |
| M (Masters) | 3,909 | 1,350 |
| G (Slams) | 4,826 | 127 |
| D (Davis Cup) | 0 | 2,574 |
| F (Finals) | 0 | 197 |
| **total** | **11,059** | **16,770** |

B-score is a node feature *and* the model's skill prior — `direct_bscore_logit` adds
`bscore_scale * (b_a − b_b)` straight to the logit, and the head is initialised to zero
so the model *starts* as a pure B-score model. Running on matches without B-score means
falling back to the 25th-percentile default for most players, which would not be an
expansion so much as a dilution.

**What has to happen first.** `preprocess/` computes the B-score snapshots across four
near-identical surface-parameterised notebooks (`graph.ipynb`, `graph_clay1.ipynb`,
`graph_grass1.ipynb`, `graph_hard1.ipynb`). They must be re-run over the full tour.

I have deliberately not done this. Those notebooks generate the CSVs every downstream
result depends on, and I cannot validate a rewrite against the originals without
re-deriving the whole pipeline — a silent change there would corrupt every number in
`TUTOR_REPORT.md` without failing anything. That is your call to make, not mine.

**The two routes, and my recommendation.**

1. **Re-run the existing notebooks over the full tour, unchanged in logic.** Lowest risk.
   Verify by checking the regenerated snapshots reproduce the current values *exactly*
   on the 244 tournaments already covered — that is a real regression test, and it is
   cheap.
2. Rewrite them as one parameterised script. Cleaner, and the four-way duplication is a
   genuine defect, but it changes and expands at once, so a discrepancy afterwards is
   ambiguous.

**Do (1) first, expand, confirm the results still hold, and only then consider (2).**
Never change the definition and the scope in the same step.

A scaffold with the coverage check and the regression test is in
`new_work/expand_data.py`. It currently *reports* the gap and refuses to proceed rather
than silently filling B-score with defaults.

---

## Next, after these

3. **Cold start becomes the headline analysis.** The positive result is already
   measured: in the winning model, message passing helps significantly where the
   thinner-history player has 0–5 recorded opponents (−0.0048, 5/5 seeds, CI excludes
   zero) and hurts everywhere else. That stratum is 615 matches, and it was found after
   the main analysis — so it is a hypothesis for the expanded data to confirm, not a
   settled result. Task B is what gives it the sample size to be believed.

4. **Re-run the temporal model** through `tennis_gnn/` so it emits a matching
   `evaluation_hash`. See `inconclusive_and_superseded/README.md` §1 for why it is now
   more interesting than when it was parked.

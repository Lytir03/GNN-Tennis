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

---

## Full-scope run: cost, and what it forced

Measured, not estimated: **161 ms per optimiser step at full scope against 64 ms
at Slam+Masters**, from 5.2x the trainable blocks (1734 vs 336) and 2.4x the
edges per block (15255 vs 6418) - 12.5x compute. A 1-hop run takes 1058s, a
2-hop about 2100s.

That makes the planned 12 cells x 5 seeds roughly 19 hours, so two things gave
way. Both are limitations of this run, not findings:

1. **The full-scope tuning search was abandoned.** The 21-configuration grid
   costs 6.2h here; a 6-configuration focused search was substituted, and even
   that was stopped after one configuration when it collided with the grid.
   The recipe used for tier 3 (lr 3e-4, 4 steps, mini-batch 32, one pass) is
   therefore **carried over from the smaller scope**, confirmed by exactly one
   full-scope validation measurement (calibrated 0.6309). A schedule that suits
   5x the data better would not have been found.

2. **Seeds are uneven by tier.** Tier 3 gets 5 seeds because it carries the
   headline and the cold-start analysis; tiers 0, 1 and 2 get 3, which widens
   their paired intervals (t = 4.303 rather than 2.776). Every hop contrast is
   still within-tier and one-factor, so the comparisons remain valid - they are
   simply less precise. Report n per tier.

Tiers run in value order (3, then 1, 0, 2) so that an interruption still leaves
the headline and the substitution curve's endpoints intact.

### The 2-hop row is mostly the recipe, and here is the proof

Limitation (1) above was not left as a caveat. The tier-3 grid row reported
`gain_2_vs_1 = +0.02368` — apparently a large penalty for the second hop. Two
hops touch far more of the graph per update, so a learning rate selected at one
hop on a five-times-smaller dataset is the first thing that should be suspected.

`new_work/twohop_diagnostic.py` reruns the identical 2-hop model on the same
seeds and data, changing **only** the learning rate, 3e-4 → 1e-4:

| arm | seed 42 | seed 123 |
|---|---:|---:|
| 1 hop, lr 3e-4 (grid) | 0.62157 | 0.62011 |
| 2 hop, lr 3e-4 (grid) | 0.64515 | 0.63808 |
| **2 hop, lr 1e-4** | **0.62888** | **0.62523** |

Lowering the step size recovers **0.0146 of the 0.0237** — roughly 60% of the
apparent depth penalty was the recipe, not the depth. Per-run times also
stabilised (1039s/1047s against 1054–6839s under the old rate).

**How to read the grid row.** `gain_2_vs_1 = +0.02368` is an upper bound on the
2-hop penalty under a recipe held fixed across the hop axis. It is a valid
one-factor contrast — the recipe *is* held fixed, which is what the design
requires — but it is not a depth measurement, and must not be reported as one.
A residual **+0.0062** penalty survives the rate change on these two seeds, so
the second hop still looks unhelpful at this feature tier; two seeds is a
diagnostic, not an estimate, so no corrected effect size is quoted.

This is the same failure mode as the earlier `num_layers=0` control: an
intervention that quietly changed a second thing. It was caught here only
because the runtime blew up alongside the loss.

---

**Do not compare full-scope numbers with Slam+Masters ones.** The wider scope is
a harder problem - the GBDT falls from 0.6015 to 0.6249 - because ATP 250/500
draws bring weaker and less-recorded players. Only within-scope contrasts mean
anything.

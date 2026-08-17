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

Run: `python new_work/feature_hop_grid.py --scope {slams_masters,full}`
→ `new_work/results/feature_hop_grid_<scope>.csv`. The scope is in the filename
because the two are not comparable; an un-suffixed `feature_hop_grid.csv` from an
earlier run was deleted rather than left to be mistaken for either.

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

## Stage 5 — the headline holds, and the cold-start hypothesis does not

Run: `python new_work/cold_start.py --scope full`
→ `new_work/results/{headline,strata,interactions}_full.csv`

### The headline strengthens at full scope

`grid_3_history_decoder_nonehop` against `gbdt_tuned`, paired over five seeds on
identical matches and labels:

| metric | Δ (GNN − GBDT) | 95% CI | seeds won |
|---|---:|---|---:|
| log loss | **−0.00465** | [−0.00593, −0.00336] | 5/5 |
| Brier | **−0.00202** | [−0.00261, −0.00144] | 5/5 |
| accuracy | **+0.00508** | [+0.00295, +0.00721] | 5/5 |

At Slam+Masters this was a win on probability quality and a *tie* on accuracy.
At 2.7× the data it is a win on all three. The same caveat as before still
governs what it means: **the winning configuration has message passing
disabled**, so this is a tabular comparison between two models given the same
features, not a demonstration that the graph helps.

### The cold-start hypothesis was tested and it failed

This was the finding the expansion was built to confirm. It was flagged as "a
hypothesis for the expanded data to confirm, not a settled result", because it
was found after the main analysis. The expanded data refutes it, and does so
cleanly.

One hop against no messages, tier 3, five seeds, by how many prior opponents the
thinner-recorded player has:

| stratum | matches | Δ (1 hop − none) | 95% CI | seeds won |
|---|---:|---:|---|---:|
| **0–5 (cold)** | 1404 | **+0.01426** | [+0.00357, +0.02496] | **0/5** * |
| 6–20 | 2449 | +0.00010 | [−0.00098, +0.00118] | 2/5 |
| **21+** | 10487 | **−0.00129** | [−0.00241, −0.00016] | **5/5** * |

At Slam+Masters the cold stratum favoured message passing (−0.0048, 5/5). At
full scope the sign is **reversed** and unanimous the other way: 0/5 seeds, and
the graph does its only useful work where players are *well* recorded.

**Why this is the more believable direction.** A cold-start node has almost no
neighbourhood to aggregate. One hop over two or three edges cannot manufacture a
skill estimate; what it does is dilute the B-score prior, which at that point is
the only reliable signal the model has. Message passing needs structure to be
worth anything, and cold start is defined by not having it.

So the graph does not substitute for a missing per-player history. **It
amplifies a present one.**

### What survives: your intransitivity effect

The dose-response by shared opponents is monotone and points the same way:

| common opponents | matches | Δ (1 hop − none) | seeds won |
|---|---:|---:|---:|
| 0 | 920 | **+0.01731** * | 0/5 |
| 1–4 | 1814 | +0.00188 | 1/5 |
| 5–14 | 3037 | −0.00057 | 4/5 |
| 15+ | 8569 | −0.00126 | 5/5 |

And the pre-specified contrast — matches with no prior meeting but a shared
opponent, against everything else — is significant at full scope:

**interaction = −0.00336, CI [−0.00547, −0.00126], 5 seeds.**

At Slam+Masters this contrast had decayed to −0.0035 and was *not* significant at
the richest feature tier. Same effect size at full scope, now with the precision
to resolve it. This is the one relational claim in the project that has survived
every control and got stronger with more data.

Note the honest reading: `degree_stratum` and `common_stratum` are strongly
correlated — a player with few opponents has few shared ones — so "helps where
there are common opponents" and "helps where players are well recorded" are
probably one phenomenon seen twice, not two findings.

### At poor features, the graph helps everywhere

Tier 1, three seeds, same contrast:

| stratum | Δ (1 hop − none) | seeds won |
|---|---:|---:|
| 0–5 (cold) | −0.01670 * | 3/3 |
| 6–20 | −0.00371 * | 3/3 |
| 21+ | −0.01542 * | 3/3 |

Every stratum, every seed. When the model has only B-score and static
attributes, message passing is worth having *including* in cold start. The
substitution story is therefore alive — it just belongs to the low-feature
regime, not to cold start. Give the model the history features and the graph's
contribution collapses to zero overall and turns negative where data is thin.

**24 stratum comparisons were computed with no multiplicity correction.** The
degree cut and the `two_hop_only` contrast were pre-specified (carried over
unchanged from the smaller scope); the rest are descriptive.

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

It was caught only because the runtime blew up alongside the loss.

---

## The temperature fit was broken, and it changed one published number

**The defect.** `fit_temperature` ran LBFGS with no line search. Fixed-size
steps overshoot, so whenever the optimal temperature was *below* one the fit ran
past it to the clamp at T=0.0183. No under-confident model in this project was
ever calibrated, and several were made much worse than doing nothing.

Temperature is fitted on validation, where T=1 is always available — so a
correct fit can never lose to not calibrating. That guarantee is what should
have caught this years earlier, and it is now enforced: the fitter compares its
answer to T=1 and falls back with a warning if it lost.

**The repair needs no retraining.** A stored probability is `sigmoid(logit / T)`
with T in the manifest, so the raw logit comes back exactly as
`T * log(p / (1-p))`. `new_work/recalibrate.py` inverts, refits, rewrites, and
asserts `evaluation_hash` is unchanged (it covers keys and labels, never
probabilities). Checked against retraining the affected cell from scratch: same
temperature, same log loss, to four decimals.

**What moved.** 18 of 129 artifacts; 111 were already correct.

| cell | T before | T after | test log loss |
|---|---:|---:|---|
| `grid_1_bscore_0hop` (5 seeds) | 0.0183 | 0.25–0.34 | 2.89 → **0.652** |
| `grid_1_bscore_nonehop`, Slam+Masters (5 seeds) | 0.403 | 0.128 | 0.647 → **0.624** |
| `grid_1_bscore_nonehop`, full (2 seeds) | 0.513 | 0.188 | 0.662 → **0.645** |
| everything else | ≈1 | ≈1 | moves < 0.0005 |

**What it costs the argument.** One point on the substitution curve, and one
piece of supporting evidence.

Corrected `gain_1_vs_none` at Slam+Masters, five seeds:

| tier | before | **after** |
|---|---:|---:|
| 0 — no B-score, no history | −0.0813 | **−0.0813** |
| 1 — B-score + static | −0.0367 | **−0.0142** |
| 2 — + history on nodes | +0.0069 | **+0.0069** |
| 3 — + history at decoder | +0.0000 | **+0.0000** |

The graph's value at tier 1 was overstated threefold. The shape of the curve —
monotone decay, crossing zero between tiers 1 and 2 — is unchanged, and so is
every other tier. **The headline is untouched**: every tier-3 artifact fitted
T ≈ 1.0–1.2, the regime the bug never entered.

The second casualty is the justification for rejecting `num_layers=0` as a
control. It was rejected partly because it "diverged to 2.89", blamed on the
missing LayerNorms. Repaired, it scores 0.652 — worse than the honest control's
0.624, but not divergent. The design argument for `disable_message_passing`
stands on its own: a control must remove one thing. The dramatic number was
never the reason, and should not have been quoted as one.

---

## The two low-feature tiers are converged, not under-trained

A no-message model that predicts near-constantly invites the objection that it
was simply under-optimised — which would make every hop gain a mix of
information and optimisation. Tested directly at Slam+Masters, varying only the
recipe:

| tier 0, no messages | test log loss |
|---|---:|
| as run (lr 1e-4, 4 steps) | 0.6931 |
| 4× steps (16) | 0.6931 |
| 10× learning rate (1e-3) | 0.6931 |
| 3 passes | 0.6931 |

Exactly the coin flip under every recipe, with a logit standard deviation of
1e-8 to 0. This is not a failure to train: with the B-scores zeroed the model
sees only height and handedness, the antisymmetric decoder can represent "no
difference" exactly, and that is the correct answer. Give the same architecture
one hop on the same features and it reaches 0.6119. **At tier 0 the graph
supplies the entire signal** — which is the cleanest statement of the
substitution thesis in the project, and also why that row's −0.081 must be
labelled "against an uninformative baseline" rather than read as what message
passing is worth in general.

Tier 1 behaves the opposite way: more optimisation makes it sharply *worse*
(0.647 → 1.30 at 4× steps, → 1.65 at 10× the learning rate). Those runs are the
ones that drove the calibrator to its clamp, so the recipe held fixed across the
hop axis is also the only one that trains this tier stably.

---

**Do not compare full-scope numbers with Slam+Masters ones.** The wider scope is
a harder problem - the GBDT falls from 0.6015 to 0.6249 - because ATP 250/500
draws bring weaker and less-recorded players. Only within-scope contrasts mean
anything.

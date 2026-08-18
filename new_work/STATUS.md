# New work — where we actually are

Live work, newest sections last. Both original tasks are done — the grid is
filled and the data is expanded to the full ATP tour — and the analysis that
followed overturned two findings the smaller scope had supported. Where that
happened it is written up as a retraction, not quietly amended.

---

## Task A — Fill the feature × hop grid ✅ done

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

## Task B — Expand the data ✅ done

**Done.** The dataset is now the full ATP tour: **49,797 matches / 4,688 blocks**
against 10,212 before, split train 15,809 / validation 5,325 / test 14,340. The
validation standard error falls from ≈0.018 to ≈0.008, which is what makes
effects of a few thousandths resolvable at all.

**How the blocker was cleared.** B-score snapshots existed only for the 244
tournaments already in scope, and B-score is not an ordinary feature — it is the
model's skill prior, added straight to the logit with the head initialised to
zero. Running without it would have been dilution, not expansion.

`preprocess/` computed the snapshots across four near-identical
surface-parameterised notebooks. They are replaced by one parameterised script,
`new_work/build_bscore.py`, and the risk of changing the definition and the
scope in the same step was handled by not doing that: `--verify` reproduces the
published snapshots **exactly** on the old scope (1.54M rows, maximum absolute
difference 1e-16) before `--scope full` is allowed to widen anything.

**Coverage is not a problem, which had to be checked rather than assumed.**
Across the whole file only 71.5% of match-player slots carry a snapshot, which
looks alarming until it is split by phase:

| phase | B-score coverage | slots |
|---|---:|---:|
| warmup (2006–2010) | 0.0% | 27,420 |
| train | 99.10% | 31,618 |
| validation | 99.12% | 10,650 |
| **test** | **99.13%** | 28,680 |

The entire gap is the warm-up years, which are used to build history and are
never emitted as targets. Every year the models are actually scored on sits at
99%. The 25th-percentile fallback is doing nothing of consequence.

**One caution that is now the binding constraint.** The wider scope is a *harder*
problem — the GBDT falls from 0.6015 to 0.6249 — because ATP 250/500 draws bring
weaker and less-recorded players. Only within-scope contrasts mean anything, and
where the two scopes disagree (see Stage 5) it is the population that changed,
not just the sample size.

---

## Stage 5 — the headline holds; cold start does not

Run: `python new_work/cold_start.py --scope full`
→ `new_work/results/{headline,strata,interactions}_full.csv`

### The headline strengthens at full scope

`grid_3_history_decoder_nonehop` against `gbdt_tuned`, paired over five seeds on
identical matches and labels:

| metric | Δ (GNN − GBDT) | 95% CI | seeds won |
|---|---:|---|---:|
| log loss | **−0.00572** | [−0.00719, −0.00426] | 5/5 |
| Brier | **−0.00258** | [−0.00321, −0.00195] | 5/5 |
| accuracy | **+0.00448** | [+0.00165, +0.00731] | 5/5 |

At Slam+Masters this was a win on probability quality and a *tie* on accuracy.
At 2.7× the data it is a win on all three.

These numbers are for `depth_3_history_decoder_1hop_lr1e4`, chosen on validation
(0.61384 against 0.61600 for the no-message arm) and never on test. An earlier
version of this section reported −0.00465 for the grid's no-message cell, under
the belief that the winning configuration did not use the graph. It does — see
the headline correction below.

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

### The dose-response by shared opponents

| common opponents | matches | Δ (1 hop − none), tier 3 | seeds won |
|---|---:|---:|---:|
| 0 | 920 | **+0.01731** * | 0/5 |
| 1–4 | 1814 | +0.00188 | 1/5 |
| 5–14 | 3037 | −0.00057 | 4/5 |
| 15+ | 8569 | −0.00126 | 5/5 |

Monotone, and pointing the same way as the degree cut. Note these two are
strongly correlated — a player with few opponents has few shared ones — so
"helps where there are common opponents" and "helps where players are well
recorded" are probably one phenomenon seen twice, not two findings.

### The intransitivity test does not replicate — and I first read it wrong

**The error, stated plainly.** There are two interventions here and they answer
different questions:

* **1 hop vs none** — is the graph worth anything at all?
* **2 hops vs 1 hop** — is *relational* structure worth anything beyond each
  player's own neighbourhood? Only a two-layer model can route information along
  a path through a shared opponent. **This is the intransitivity test.**

I initially compared a `1_vs_none` interaction at full scope (−0.00336) against
the `2_vs_1` interaction from the earlier analysis (−0.0035) and reported that
the effect had "survived and sharpened". Those are different quantities. They
are not comparable, and at Slam+Masters they have *opposite signs* — which is
what it should have taken to notice. `new_work/cold_start.py` now runs both
interventions explicitly so the two can never be read as one number again.

**The intransitivity contrast, run correctly.** Matches with no prior meeting
but a shared opponent, against the rest; negative means the deeper model gains
more there:

| tier | Slam+Masters (5 seeds) | full scope |
|---|---|---|
| 0 — no B-score | −0.00987, [−0.01251, −0.00723] * | not yet computed |
| 1 — B-score + static | −0.01148, [−0.01412, −0.00884] * | **−0.00033, [−0.00328, +0.00262]** (3) |
| 2 — history on nodes | −0.00954, [−0.01670, −0.00238] * | not run |
| 3 — history at decoder | −0.00353, [−0.00794, +0.00088] | −0.00182, [−0.01286, +0.00921] (5) |

At tier 1 this is a **refutation, not a failure to replicate**: the full-scope
interval excludes the entire Slam+Masters interval. The effect that was
−0.0115 on 10,212 matches is −0.0003 on 49,797.

Tier 3's full-scope test is **inconclusive rather than negative** — its interval
is nine times wider than tier 1's, because the 2-hop cells were trained with the
carried-over lr 3e-4 that the diagnostic above showed to be unstable at this
scope. A clean tier-3 depth test needs those cells rerun at lr 1e-4, which has
not been done for five seeds.

**What this costs the thesis.** The intransitivity hypothesis was the most
attractive story in the project — a relational effect that a tabular model
structurally cannot hold. It looked real at Slam+Masters across three feature
tiers with intervals well clear of zero. It does not survive the tour-wide data.
The most likely reading is that Slam+Masters is a small, densely connected
population where shared opponents are informative about a narrow elite, and that
the pattern does not generalise once ATP 250/500 draws are included.

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

## The substitution curve, both scopes

This is the thesis figure, and it is the result that held up.

`gain_1_vs_none` — what one hop of message passing is worth, by how much
per-player information the model already has. Negative means the graph helped.

| tier | Slam+Masters (5 seeds) | full scope |
|---|---:|---:|
| 0 — no B-score, no history | −0.08125 | **−0.06171** (3) |
| 1 — B-score + static | −0.01418 | **−0.01355** (3) |
| 2 — + history on nodes | +0.00689 | **−0.00121** (3) |
| 3 — + history at decoder | +0.00002 | **+0.00047** (5) |

Tier 3's cell is the one run at lr 3e-4. Corrected to the stable recipe it is
**−0.00100**, so the full-scope curve read end to end is

**−0.0617 → −0.0136 → −0.0012 → −0.0010**

Tiers 0, 1 and 2 were already at lr 1e-4, so only the last point moves.

Monotone decay of the same shape on 2.7× the data and a different tournament
population. **The graph's value is a decreasing function of how well the model
is already informed about the two players** — it falls by a factor of about
sixty from tier 0 to tier 3.

But it does **not** reach zero, and this is the correction the recipe fix
forced. At Slam+Masters the curve crossed into positive territory at tier 2
(+0.0069) and the natural reading was "the graph is worth nothing, or less than
nothing, once you supply the features". At full scope, with every tier on a
stable recipe, it stays negative throughout and flattens at about −0.001, which
at tier 3 is significant on 5/5 seeds. The graph's contribution **asymptotes to
something small and real rather than vanishing.**

The left endpoint deserves its own sentence, because it is the cleanest thing in
the project. At tier 0 the no-message model scores **exactly 0.693147** — the
coin flip — under every recipe tried, because height and handedness carry
nothing and the antisymmetric decoder represents "no difference" exactly. One
hop on the same features scores 0.631. There, the graph is not helping the
features; **the graph is the only feature there is.**

**What did not replicate is the depth axis.** `gain_2_vs_1` is positive at every
tier at full scope (+0.0013 / +0.0023 / +0.0237) where it was negative at tiers 0
and 1 at Slam+Masters (−0.0086 / −0.0081). The second hop never helps on the
tour-wide data. Tier 3's +0.0237 is inflated by the unstable recipe and is being
re-measured properly by `depth_test.py`.

Read together with the intransitivity retraction above, the pattern is
consistent: **the one-hop substitution effect is robust across scopes; every
second-hop effect reverses sign when the population changes.**

---

## The recipe was hiding the graph — headline corrected

This is the most consequential thing in this file, and it reverses a claim the
project has carried since the beginning.

**What was claimed.** "The model that beats the GBDT does not use the graph."
The grid put tier 3 at 0.62022 with message passing disabled against 0.62070 at
one hop — `gain_1_vs_none = +0.00047`, message passing worth nothing.

**What was wrong with it.** Both cells used lr 3e-4, the recipe selected at the
smaller scope. The 2-hop diagnostic had already shown that rate is unstable at
full scope, and the obvious next question — *does it also handicap one hop?* —
went unasked. Retraining all three arms at lr 1e-4 (`new_work/depth_test.py`):

| arm, tier 3, lr 1e-4 | test log loss (5 seeds) |
|---|---:|
| no messages | 0.62015 |
| **one hop** | **0.61915** |
| two hops | 0.62857 |

| contrast | Δ log loss | 95% CI | seeds |
|---|---:|---|---:|
| **1 hop vs no messages** | **−0.00100** | [−0.00179, −0.00022] | **5/5** |
| 2 hops vs 1 hop | +0.00942 | [+0.00560, +0.01325] | 0/5 |
| **1 hop vs GBDT** | **−0.00572** | [−0.00719, −0.00426] | **5/5** |

**Message passing is worth a small but statistically reliable −0.0010 at the
richest feature tier.** The corrected headline is: *the best model does use the
graph, one hop of it, and it beats a tuned GBDT by 0.0057 log loss.*

**Why the error was systematic rather than bad luck.** The no-message arm barely
notices the recipe (−0.00008 between 3e-4 and 1e-4, not significant); the
message-passing arms gain about 0.0015. A model that ignores its edges has less
to optimise and is correspondingly insensitive. So holding a badly chosen recipe
"fixed across the hop axis" is *not* sufficient for a fair contrast — it is fixed
in name and unequal in effect, and it biases against exactly the arm under test.
Every hop contrast in this project that used a recipe tuned elsewhere is
suspect for this reason, and the tier 0 and 1 rows were run at 1e-4 already,
which is why their conclusions stand.

**What does not change.**

* The GNN still beats the GBDT, by more than reported (−0.0057, not −0.0047).
* Two hops still hurt, decisively (+0.0094, 0/5 seeds).
* The substitution curve keeps its shape — this moves tier 3's point from
  +0.0005 to −0.0010, still far above tier 0's −0.0617 and tier 1's −0.0136.
* **Cold start still reverses.** At the stable recipe, one hop hurts where the
  thinner-recorded player has 0–5 opponents (+0.00413, 0/5, significant) and
  helps at 21+ (−0.00229, 5/5, significant). The mistuned recipe exaggerated the
  size (+0.0143) but not the direction.

**And the intransitivity test is now conclusive rather than inconclusive.** With
both arms at a stable recipe, the `two_hop_only` interaction on the 2-vs-1
contrast is **−0.0001, CI [−0.00278, +0.00259]** — a tight null, not a wide
shrug. The hypothesis is answered: at the richest feature tier, on tour-wide
data, relational depth buys nothing.

---

## Run queue and what is deliberately not run

Strictly sequential — two concurrent jobs on this machine once caused 1.58M
swapouts and a sixfold slowdown, so nothing overlaps.

| order | run | cost | why |
|---|---|---:|---|
| done | tier 3, 5 seeds | — | headline + cold start |
| done | tier 1, 3 seeds | — | substitution curve middle |
| running | tier 0, 3 seeds | ~1.4h | curve's left endpoint |
| queued | `depth_test.py` | ~2.2h | one-factor 2-vs-1 at tier 3 |
| queued | tier 2, 3 seeds | ~1.4h | the crossover cell |

**Tier 2 was going to be dropped and is not.** It was listed as the least
informative tier, but it is where the graph's marginal value changes sign at the
smaller scope (−0.014 at tier 1, +0.007 at tier 2), so leaving it out would
leave the full-scope curve without the one cell the thesis figure turns on.

**Seeds are uneven by tier and that is a limitation, not a finding.** Tier 3 has
five; tiers 0, 1 and 2 have three, so their paired intervals use t = 4.303 rather
than 2.776. Every hop contrast is still within-tier and one-factor. Report n per
tier — the grid CSV now carries an `n_seeds` column for exactly this reason.

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

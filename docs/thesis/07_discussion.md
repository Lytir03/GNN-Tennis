# Chapter 7 — Discussion

What chapter 6's numbers mean, what they do not license, and where the
interpretation is weakest. The five sections are the outline's five bullets.

---

## 7.1 Graph structure does provide additional predictive information

The claim survives in its narrow form.

At tier 3 — the best configuration this project has, with a strong skill prior
and twelve engineered history statistics per player already in hand — one hop of
message passing is still worth **−0.00100 log loss on `full`**
(CI [−0.00179, −0.00022], 5/5 seeds) and **−0.00104 on `slams_masters_1990`**
(t=−9.53, 5/5), both at the default 1095-day window. Two tournament populations,
the same effect, both unanimous across seeds.

The claim has to carry two qualifications. At the 365-day window used by the
final model, the `full` one-hop gain is the same size (−0.00095) but only 4/5
seeds with an interval that includes zero, while on the 1990 scope it is larger
(−0.00242). And at `slams_masters` the one-hop gain is zero at both windows; there
the graph's value appears only at two hops (§7.4).

The same model beats a tuned `HistGradientBoostingClassifier` given the same
information as flat columns, on the same matches with the same labels, by
−0.00572 log loss (5/5) at `full` scope, or −0.00568 (5/5) at the 365-day window.
That is the strongest form of the claim available here, with one caveat: that
contrast differs from the GBDT in the whole model, not only the graph, so it
shows the GNN *as a system* extracts more than the tabular learner. The part
attributable to message passing alone is the hop contrast above, and it is about
a sixth of the GBDT gap.

**What this is not.** It is a small effect. The one-hop gain is about 0.16% of
the model's log loss (0.0010 of 0.619), and the gap to the GBDT about 0.9%,
against a bookmaker Brier of ≈0.198 and a published accuracy plateau of 65–70%.
The honest statement is *the graph contributes, measurably, and it contributes
little*. The thesis's contribution is the size and shape of that
contribution, not a performance claim.

**Why the claim is believable at this size.** Because the protocol was built to
make a few thousandths mean something: paired artifacts with `evaluation_hash`
asserted equal before any contrast, per-seed unanimity reported alongside every
mean, validation required to agree with the test ranking, and one factor varied
at a time with a test enforcing it. The project has already retracted results
that rested on noise (§7.5), which is the reason those guards exist.

---

## 7.2 Graph and engineered history capture partly overlapping information

This is what the substitution curve says, and it is the central finding.

| Tier | `slams_masters` | `full` | `slams_masters_1990` |
|---|---:|---:|---:|
| 0 — no B-score, no history | −0.0813 | −0.0617 | −0.1166 |
| 1 — B-score + static | −0.0142 | −0.0136 | −0.0323 |
| 2 — + history on nodes | +0.0069 | −0.0012 | +0.0023 |
| 3 — + history at decoder | +0.0000 | −0.0010\* | −0.0010 |

\* stable-recipe re-run, not the grid's own cell (chapter 6 §6.4).

The value of one hop falls by roughly **sixty-fold** from tier 0 to tier 3, and
the shape is the same at all three scopes. Engineered per-player history and
one-hop message passing are therefore **largely substitutes, not complements**:
they are two ways of summarising the same underlying object, a player's recent
record against recent opponents.

This is mechanically unsurprising once stated. One hop of message passing *is* a
learned aggregation over a player's own opponents, and `aggregate_history()` is
a hand-specified aggregation over the same set. What the grid establishes is
that the hand-specified version captures most, but not all, of what the learned
one finds.

**The routing result sharpens it.** Tier 2 and tier 3 contain the identical
twelve numbers; they differ only in whether those numbers pass through the
encoder (and so get smoothed by message passing) or go straight to the decoder.
Tier 2 makes the graph's marginal value *positive* — harmful — at two scopes out
of three. So the overlap is not benign: when the same information is present on
both pathways, the graph averages away the sharper of the two. This is the
argument for `history_to_decoder`, and it is a statement about information
routing rather than about capacity.

**Where the substitution does and does not happen** is §6.5's result, and it
runs opposite to the intuitive story. The graph helps most where records are
*thick* (21+ prior opponents: −0.00129, 5/5) and hurts where they are thin (0–5:
+0.01426, 0/5). A cold node has no neighbourhood to aggregate; one hop over two
or three edges cannot manufacture a skill estimate, it only dilutes the B-score
prior. **The graph amplifies a present history; it does not substitute for a
missing one.** The cold-start hypothesis, which motivated much of this work, was
tested and failed — at `slams_masters` it had looked true (−0.0048, 5/5), and
the sign reversed and became unanimous at tour scope.

---

## 7.3 The contribution shrinks with better features but does not disappear

Two distinct reasons to believe the tier-3 effect is real rather than a residue
of noise:

1. **It replicates on a second population.** `slams_masters_1990` is not fully
   independent of `full` — the test years overlap in 2019–20, and the Slams and
   Masters matches in those years are in both — but it has a different
   tournament mix (no ATP 250/500), a different training history (1990–2011
   against 2011–16) and a different test period (2014–20 against 2019–24). Both
   give −0.0010 at the default window, both 5/5.
2. **Known confounds were removed before the number was read.** The two-hop
   contrast had been inflated by a learning rate carried across arms, and the
   `full` tier-3 cell is quoted from the re-run where both arms share a stable
   recipe (§6.3, §6.4).

What the evidence does *not* show is a tier-3 one-hop effect at `slams_masters`:
it is zero there at both the 1095-day and the 365-day window. The window fix
recovered value for *two* hops at that scope, not one.

**The asymmetry that makes recipe hygiene non-optional.** The no-message arm
barely notices a change of learning rate (−0.00008) while message-passing arms
gain ≈0.0015 — a model that ignores its edges has less to optimise. A recipe
"held fixed across the hop axis" is therefore fixed in name and unequal in
effect, and it is biased *against* the arm under test. Any depth or hop result
in this literature reported under a single shared recipe should be read with
that in mind; it is what had previously made this project's own headline come
out the wrong way round.

**The left endpoint must be labelled, not quoted.** At tier 0 the no-message
model scores exactly 0.693147 — ln 2 — because height and handedness carry
nothing and the antisymmetric decoder represents "no difference" exactly. The
−0.08 there is the graph measured against an uninformative baseline. It is the
cleanest demonstration that the graph alone carries real signal (one hop reaches
0.6119 / 0.6314 / 0.5766 from nothing), and it is *not* evidence about the graph's
marginal value in a working model.

---

## 7.4 Optimal graph depth depends on the dataset and scope

**The one-hop substitution effect replicates across scopes; every second-hop
effect reverses sign when the population changes.** `gain_2_vs_1` is positive at
every tier at `full` where it was negative at tiers 0–1 at `slams_masters`; the
tuned `slams_masters` configuration keeps two hops while the `full` one takes
one; and the intransitivity result — the single most natural mechanism for a
second hop to help — is a **tight null** at `full` (−0.0001, CI [−0.00278,
+0.00259]) whose interval excludes the entire `slams_masters` interval.

The plausible reading is about graph density rather than about tennis. Adding
ATP 250/500 events roughly triples the matches but also adds weaker and
less-recorded players, so the second hop reaches a rapidly growing and
increasingly heterogeneous neighbourhood. Whatever a shared opponent is worth in a Slam-and-Masters graph, it is worth
something different when most of a player's two-hop neighbourhood consists of
players who barely appear.

This is a hypothesis, and this repository does not test it. What it does
establish is the methodological point: **depth is not a transferable
hyperparameter here.** A depth chosen on one tournament scope should not be
assumed on another, and any claim about "the right number of hops for tennis" is
under-specified without the population it was selected on.

---

## 7.5 Limitations and interpretation

Stated at full strength; the full list is `FINAL_RESULTS.md` §11.

**On which artifact a number comes from.**
- At `full`, the headline and hop contrasts in `FINAL_RESULTS.md` come from the
  1095-day `depth_3_*_lr1e4` artifacts, while the "final model" row is the
  365-day artifact. The headline survives at 365 d; the one-hop significance does
  not (chapter 6 §6.3).
- The substitution-curve tier-3 `full` cell is a stable-recipe re-run, not the
  grid's own lr 3e-4 cell (+0.00047).
- The window-sweep row in `FINAL_RESULTS.md` §7 and `FINDINGS.md` §2 is labelled
  "2hop − none" but is no-messages minus two hops.

**On implementation.**
- `ModelConfig.tournament_context_edges` is not connected to the edge builder, so
  that ablation is `base` re-run.
- A player absent from a B-score snapshot gets the snapshot **median** in the GNN
  and the **25th percentile** in the GBDT (chapter 3 §3.6). With 99.1% coverage
  in scored years this is small, but it is a difference in information between
  the models being compared.
- At `slams_masters` the B-score graph is built on a tournament-name whitelist
  that is wider than the `G, M` prediction scope (chapter 3 §3.1).

**On comparability.**
- Absolute numbers are **not comparable across scopes**. The GBDT alone falls
  0.6015 → 0.6249 from `slams_masters` to `full`. Only within-scope contrasts
  are interpretable.
- **Seeds are uneven by tier.** Tier 3 has five; tiers 0–2 have three at `full`,
  so those intervals use t(2)=4.303. Every hop contrast is still within-tier and
  one-factor — less precise, not invalid. `n_seeds` is carried in the grid CSVs.
- **Tier 0 reused tier 1's recipe.** Its absolute level carries that caveat;
  the hop contrast within it is still clean, because the recipe is held fixed
  across the hop axis *within* the tier.

**On the search.**
- **The `full`-scope tuning search was abandoned on cost** (161 ms/step against
  64 ms, ≈12.5× compute; the 21-configuration grid costs 6.2 h). Tier 3's recipe
  was carried over from the smaller scope and confirmed by a single full-scope
  validation measurement. §7.3 shows how much a recipe can matter, so this is
  the most load-bearing unfinished piece of the setup.
- `focused_space()` searches only the amount-of-optimisation axis. A good
  schedule far from that neighbourhood would not be found, and the results
  should be read as coming from a small search.

**On the baselines.**
- **The 1990 scope has no GBDT baseline.** `gbdt_comparison/results/full_1990/`
  holds one seed of a *different* scope (11,830 rows) and is not the comparison
  partner for the 6,995-match GNN artifacts. The 1990 scope is used to replicate
  the hop and window effects, not to re-run the headline.
- The comparison to published B-score results crosses papers, data sources and
  test years. It could raise the question; `baselines.py` answers it on our own
  matches, and that is the version that counts.

**On the subgroup analysis.**
- **24 stratum comparisons, no multiplicity correction.** The degree cut and the
  `two_hop_only` contrast were pre-specified; the rest are descriptive and
  should be read as descriptive.
- Structural descriptors correlate with how much data a player has, and data
  richness independently changes which model wins. `stratify.py` is written to
  make that confound visible — controlling for sparsity flips the sign — and its
  own header says the trustworthy test is the depth ablation, which intervenes
  on the receptive field instead of correlating with it.

**On the data.**
- B-score snapshot coverage is 99.1% of player slots in train, validation and
  test; the entire gap is warm-up years, which are never scored.
- Coverage across the whole file is 71.5% of match-player slots, which looks
  alarming until it is split by phase. It had to be checked rather than assumed,
  and is documented in `new_work/STATUS.md` Task B.

**On the explanations.**
- **The staleness effect is solid; its mechanism is open.** The `sum`-overwhelms-
  recency hypothesis was tested and refuted.
- **The direct-logit arithmetic was right and the inference from it was wrong.**
  Five principled fixes, all worse, control winning on validation too. A correct
  diagnosis of an internal quantity is not a prediction about behaviour.
- More generally: roughly a dozen fixes reasoned from the architecture's
  internals produced zero wins, while both real gains came from hypotheses about
  the data. That is a limitation of the method used, and it is reported as one.

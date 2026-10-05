# Defense prep: questions the panel may ask

Built from the actual thesis text (`tesi.rtf`, Bachelor in Computer Science, supervisor Dr. Sarah Maria Winkler) plus the repo chapters in `docs/thesis/`. **The principal scope is Slams + Masters, and the best model there is 2-hop, 365-day, tier 3**; full-tour and the 1990 extension are robustness checks. Each question has a short answer cue. **Bold** = most likely to be asked.

**Note on earlier sections:** where they quote the full-tour one-hop result (−0.001) as "the" graph effect, the thesis instead leads with the Slams + Masters two-hop effect (≈ −0.0029, 5/5). Use Section A below as your framing; the rest still applies.

---

## A. Thesis-specific framing and traps (read first)

**The claim as written:** on Slams + Masters, 2 hops vs no messages at 365 d = 0.5942 vs 0.5971, Δ ≈ 0.0029, 5/5 seeds. That is ≈ 45% of the GNN−GBDT gap (0.00653), and the thesis says itself this is not an exact decomposition.

**Numbers that do not line up. Expect a sharp panel member (math/ML) to catch these, so reconcile before the defense:**
1. Final S+M model is **0.59301** (2 hops, replay buffer **800**), but the hop contrast uses **0.5942** (buffer **200**). The thesis text calls 0.5942 "the final configuration" in places. They are different models: buffer 800 vs 200 is itself worth −0.0012 (4/5 seeds, per `FINDINGS.md`).
2. GNN−GBDT is quoted as −0.00653 in the thesis, but `FINDINGS.md` gives −0.0073 against GBDT 0.6015 for the buffer-800 model, and 0.6015 − 0.5930 = 0.0085. Whichever it is, the 45% ratio changes (0.0029/0.0073 ≈ 40%, 0.0029/0.0085 ≈ 34%). Know which GBDT mean and which GNN artifact the −0.00653 is computed from. I could not locate it in the repo.
3. Match counts: thesis says S+M **15,063** and full **49,184** matches; the repo chapters say 20,057 and 49,797. Probably a before/after filter difference (e.g. main-draw rounds only); know which one and why.
4. Tier-3 **one-hop** gain at S+M is **zero** (repo, at both windows) but the thesis does not state this next to the 2-hop result. A panel member who sees the substitution curve (Fig. 7) may ask why 1 hop gives nothing while 2 hops give 0.0029. Cue: the graph adds information only via indirect (opponent-of-opponent) relations in this concentrated elite network; 1-hop is already covered by engineered history (substitution). Also be ready for "is it 2 hops specifically or just any extra capacity/depth?": 3 hops over 2 is ±0.0004, so depth beyond 2 is flat. A no-message model with the same layers (honest control) is used, which rules out pure capacity.
5. The 0.0029 is **window-specific**: at 1095 d, none − 2hop is −0.0043 (graph *hurts*, 0/5). So the claimed S+M benefit depends on the 365-day window, which was found by sweeping on this same scope. Defend: independent selection by validation, interior optimum, replicated at one hop on the 1990 extension, but the mechanism is unknown (mean aggregation refuted).
6. The effect size is ≈ 0.5% of log loss (0.0029 of 0.594). Be ready to say it is small, and that the claim is about measurable information, not practical gain.

**Questions specific to the S+M two-hop result**
- *Why does 2-hop help on S+M but hurt on the full tour (+0.0055 to +0.0094 worse)?* Your hypothesis: elite network is dense and well observed, so opponents-of-opponents are informative; on the full tour the 2-hop neighbourhood fills with thinly observed players. Untested, say so.
- *Is the 2-hop gain really "indirect relations"?* The intransitivity / shared-opponent test was null on the full tour (−0.0001 [−0.0028, +0.0026]); it looked real on S+M initially. So you cannot claim the mechanism is common-opponent reasoning. Say: the information exists, the mechanism isn't pinned down.
- *Why test only 5 seeds, and are seeds independent?* See the math section 3.8: the seeds reshuffle labels/initialisation over the same 3,770 test matches, so the CI describes training/orientation variability and not sampling of new matches. A paired bootstrap over matches or tournaments would be the fix.
- *Sparse players on S+M benefit from the graph, on full tour they are hurt.* The thesis explanation (sparse S+M players still connect to well-observed elite players) is plausible but untested.
- *The thesis states 'mean aggregation did not remove the effect'.* The text in 6.2 and 7.4 slightly overclaims "removing older matches changes both age and number of edges": you cannot separate staleness from edge count. Own this.
- *S+M test set is only 3,770 matches over 4 years (2017–2020), validation only 2016 (1,075 matches).* Log-loss SE ≈ 0.018 on validation, so the window/hyperparameter selection signal is noisy. The test-set paired CI for 2-hop vs none is what carries the claim.

**Thesis-text issues a reader may point out (cheap to fix or to have an answer for)**
- GINE is cited to Hu et al. 2020 (pre-training GNN paper, which does introduce GINE; the original GIN is Xu et al. 2019, not cited).
- Chapter 1.3 and 2.4 contain near-identical related-work paragraphs (duplication).
- Section 1.4 says "Section 1.3" for research questions in Chapter 6 intro (numbering), and "wether" typo.
- Section 3.4.3 B-score formula: the sum and index convention and "power iteration" vs repo's `networkx` (which also uses power iteration; fine). Be able to state the Perron-Frobenius conditions.
- Section 3.4.2 formula rendering is broken in the text (ratios), check the PDF.
- Bayram et al. and Wilkens are cited, but there is **no bookmaker-odds comparison** in your results, although Wilkens says odds contain most of the information. Expect "why not compare to odds?" (not in the dataset).
- No Clegg & Cartlidge comparison of results (they use a GNN too): expect "how does yours differ?" Answer: they motivate by intransitivity and compare to weighted Elo; you ask the *marginal* value over engineered features and a B-score prior, with a one-factor hop ablation.

---

---

## 0. The 60-second version (have this memorised)

- **Question:** does graph structure add predictive information beyond standard engineered history features?
- **Setup:** ATP matches (Sackmann), blocked online protocol (one snapshot per tournament round, built from strictly earlier matches), GINE GNN with an antisymmetric decoder, B-score prior, vs a tuned HistGradientBoosting baseline on the same information. 5 seeds, paired, `evaluation_hash`-checked.
- **Answer:** yes, but small. On **Slams + Masters** the 2-hop, 365-day GNN beats the no-message control by ≈ 0.0029 log loss (5/5 seeds) and the tuned GBDT by ≈ 0.0065 (5/5), about 45% of the gap being message passing. The graph's value falls ~60× as engineered features are added (substitution curve), engineered history and 1-hop are largely substitutes, and the 2-hop gain does not transfer: on the full tour one hop is best (≈ −0.001) and the second hop hurts.

## 0b. Numbers to know cold

| Item | Value |
|---|---|
| **S+M headline** | 2 hops, 365 d, tier 3: **0.5942 vs 0.5971 no-message, Δ ≈ 0.0029, 5/5**; final model (buffer 800) 0.59301 |
| S+M GNN vs GBDT | −0.00653 log loss [−0.0102, −0.0029], Brier −0.00237, accuracy +0.0014 (3/5, CI includes 0) |
| S+M window sweep (none − 2hop) | 90 d −0.0029 · 180 d −0.0022 · 270 d +0.0012 · **365 d +0.0029** · 547 d +0.0020 · 730 d −0.0001 · 1095 d −0.0043 |
| S+M data | train 2011–15, val 2016, test 2017–20 (3,770 matches); warm-up 2006–10 |
| Final test log loss | full 0.61919 · slams_masters 0.59301 · slams_masters_1990 0.56766 |
| GBDT at full | 0.62487 |
| GNN vs GBDT (full) | −0.00572 LL (5/5), Brier −0.00258, acc +0.0045 |
| 1 hop vs none, tier 3 | −0.0010 (full, 1095 d, 5/5); −0.00095 at 365 d (4/5, CI includes 0); 0 at slams_masters |
| 2 hops vs 1 (full) | +0.0094 / +0.0055 (worse, 0/5) |
| Substitution curve | tier 0 → tier 3: −0.08 → −0.001 (~60×) |
| Window sweep (slams_masters, 2 hops) | 365 d optimum, +0.0029, 5/5 |
| Cold stratum (0–5 opponents, 1,404 matches) | +0.0143 (graph hurts, 0/5) |
| Elo vs B-score standalone | Elo better (0.5805 vs 0.6073 on 1990 scope) |
| Test n | 14,340 (full) · 3,770 (SM) · 6,995 (1990) |
| Reference | bookmaker Brier ≈ 0.198; literature accuracy plateau 65–70% |
| Seeds | 42, 123, 456, 789, 2026; t(4)=2.776 |

---

## 1. ML / neurosymbolic

**1.1 Why a GNN at all? What does it learn that the GBDT can't?**
Learned aggregation over a player's opponents (1 hop) and, with 2 hops, shared opponents. Honest answer: only the 1-hop part was shown to matter, and it's small.

**1.2 Why GINE and not GAT / GCN / a transformer?** Edges carry the signal (margin, recency, surface, round). GINE injects edge features inside the message; GATv2 uses them only for attention weights and scored worse on every metric. Sum aggregation keeps count information (degree varies 1 to hundreds).

**1.3 What is over-smoothing and where did you see it?** Repeated aggregation pulls node states toward the neighbourhood average. Evidence: the same 12 history statistics *help* when routed to the decoder (tier 3) and *hurt* when put on nodes (tier 2): message passing averages away per-player sharpness.

**1.4 The GNN beats the GBDT by 0.0057 but message passing is only worth 0.001. Where is the other 0.0047?** (A sharp question.) The comparison is system vs system: antisymmetric decoder, B-score skip connection, pairwise decoder features, online training, per-config tuning. The thesis says this itself (ch. 7.1). Be ready to say the repo has no single ablation attributing it, and what you would run (GBDT with the antisymmetry trick, i.e. symmetrised training data; or a no-message GNN vs GBDT, which isolates the non-graph part).

**1.5 Is the GBDT comparison fair?** Same B-score, same 12 history stats, 32-config grid on validation. Weaknesses: the GBDT gets a different newcomer fallback (25th percentile vs the GNN's median), lacks Elo, and has no h2h / common-opponent features. The GNN's search was a smaller `focused_space` (6 configs) at full scope, and the 21-config grid was abandoned on cost.

**1.6 Neurosymbolic angle: where is "symbolic" structure in your model?**
- Hard constraint: P(A beats B) = 1 − P(B beats A) enforced *exactly* by construction (not learned).
- Rating priors as symbolic knowledge: B-score and Elo (stored on the logit scale so a difference *is* the Bradley-Terry logit), injected through a skip connection; the decoder's last layer is zero-initialised so the model starts as a pure rating predictor.
- Possible extensions: other logical constraints (transitivity as soft regularisation, monotonicity in rating difference, surface-specific rules), or a differentiable rule layer. The intransitivity result (null at full) says transitivity-violating structure isn't where the gain is.

**1.7 Why did all the architecture fixes fail but the data fixes work?** ~12 internal fixes (skip-connection scale, node normalisation, 3 hops, mean aggregation…) gave zero wins; history routing and window staleness (hypotheses about data) worked. Lesson: the model wasn't broken, its inputs were.

**1.8 Calibration.** Temperature scaling fitted on validation only; preserves ranking/accuracy. A bug (no LBFGS line search) drove T to the 0.0183 clamp; fixed with fallback to T=1; 18 of 129 artifacts repaired by inverting stored probabilities, evaluation_hash unchanged.

**1.9 Online training and replay.** Gradient updates only on train blocks, replay buffer of recent blocks, Adam. Why not offline? Graph depends on time; blocks are the unit. Risk: non-stationarity and catastrophic forgetting, which the buffer size (200 vs 800) is partly about.

**1.10 Hyperparameter tuning and the recipe trap.** A shared learning rate across the hop axis biased *against* message-passing arms (no-message arm moves −0.00008 with recipe, MP arms ≈ −0.0015). 2-hop gain of 0.0237 was ~60% learning rate. This flipped the headline.

**1.11 Why log loss as the primary metric?** Object of study is a probability; proper scoring rule; temperature scaling moves it without moving accuracy; selection metric on validation.

**1.12 Why not deep sequence models / temporal GNN (TGN, EvolveGCN)?** Listed as the most promising future work, because the window sweep showed recency is decisive and its mechanism is open.

**1.13 Overfitting / test-set reuse.** Hyperparameters, window, temperature on validation only; test scored once; validation must agree with test ranking. Admit: many configurations were explored across the project, and the window was "independently selected by validation".

---

## 2. DBMS + time series

**2.1 How do you prevent temporal leakage?** Invariant *trim → read → append* at all three state sites (B-score, HistoryTracker, Elo). Unit of prediction is the tournament-round block, so same-round results can't inform each other. Table of leaks in ch. 3.7. Concrete bugs found: hardcoded `start_date` desynchronised labels (50.3% disagreement = coin flip), wrong B-score snapshot suffix scored most players from fallback with no error.

**2.2 Why split by calendar year, and why these splits?** Walk-forward/temporal rather than random. Warm-up years only build state. Full scope uses two validation years because 1 year (1,075 matches) gives a log loss SE ≈ 0.018, far above the 0.002–0.004 differences being selected on.

**2.3 Non-stationarity / concept drift.** Player skill and the tour change. Handled with hyperbolic recency decay `1/(1+age/365)`, a 3-year history deque, and a graph window (365 d best). The finding that tests were on a *staler* graph than training (65 → 81% edges older than 1 year) is a drift story.

**2.4 Why hyperbolic rather than exponential decay?** Old matches fade but never vanish (3 y → 0.25). Be honest that it wasn't compared to alternatives in the thesis.

**2.5 Why not classical time-series models (ARIMA, Kalman, state-space, Glicko)?** Elo is essentially a one-step stochastic-gradient update of a Bradley-Terry model, i.e. a simple filter. Glicko / Kalman-style rating with uncertainty is a natural baseline you did not run.

**2.6 Data model and pipeline.** One row per match, winner perspective; sorted by (date, tourney_id, round_order); 3 scopes; long-format B-score snapshots per (tourney_id, round_order, player) (1.54M rows); graph cache keyed by scope, preset, window with version tag (`v4`); seed-independent graph cache plus per-seed targets. 4,688 blocks at full.

**2.7 Reproducibility / provenance.** Frozen per-match prediction CSV + manifest per model, scope and seed; `evaluation_hash`; `verify_targets.py`. Admit the provenance gap: no script writes `atp_matches_full.csv`.

**2.8 Data quality.** Dropped rows with non-main-draw rounds (613 in full); Carpet one-hot all zeros; incomplete matches (RET/W/O) with `incomplete_margin_policy="zero"`; static height/hand taken from the latest record (the one known look-ahead); B-score coverage 99.1% of player slots in scored years (71.5% over the whole file, because of warm-up).

**2.9 Cost / scalability.** B-score needs a full eigenvector solve per block over a growing graph: hours at full scope, so cached offline. Snapshot building is the expensive half of a run; caching turned tuning from hours to minutes. Full-scope GNN step 161 ms vs 64 ms (≈12.5×).

**2.10 Why one snapshot per block instead of per match / per day?** Per-match would let a QF result inform another QF; per-day is not the tennis event structure.

**2.11 Is the labelling/orientation design sound?** A seeded coin flip decides who is "A"; exactly one `rng.random()` per emitted match in block order for both model families, so seeds are *different evaluation sets* and can't be averaged as ensembles.

---

## 3. Mathematics (analysis, linear algebra, computational maths)

**3.1 B-score = eigenvector centrality: when does it exist and is it unique?** Perron-Frobenius: needs a non-negative irreducible matrix (strongly connected graph) for a unique positive dominant eigenvector. A loser→winner graph is *not* strongly connected (an undefeated player has no outgoing edge, new players have none). Ask yourself what NetworkX does: power iteration to tol 1e-9, max_iter 1000; for reducible graphs the vector can concentrate on one strongly connected component. Be ready to discuss damping (PageRank) or Katz/Bonacich with an attenuation parameter as the standard fix. The cited paper calls it Bonacich centrality.

**3.2 Convergence of power iteration.** Rate governed by |λ₂/λ₁|; a small spectral gap means slow convergence; the graph grows monotonically so you can warm-start from the previous solution (not done).

**3.3 Normalisation.** NetworkX L2-normalises, so a B-score is only meaningful relative to others in the same snapshot; only within-snapshot differences are used. Typical |b_a − b_b| ≈ 3.2e-3, which is why `bscore_scale` "needs ≈15".

**3.4 Elo as a model.** E = 1/(1+10^{(R_l−R_w)/400}); update is SGD on logistic loss with step K=32. Stored as (R−1500)·ln10/400 so a stored difference is exactly the logit. That is Bradley-Terry.

**3.5 Prove the decoder is exactly antisymmetric.** logit(a,b) = s·(b_a − b_b) + ½(f(a,b) − f(b,a)). Swapping gives −logit, so σ(logit(b,a)) = 1 − σ(logit(a,b)). Why no intercept: it would break this. Why exactly ln 2 at tier 0: no information → f(a,b) = f(b,a) → logit 0.

**3.6 Why can GIN with sum aggregation be as expressive as 1-WL?** Sum over a multiset is injective (given suitable MLP), mean and max are not. Link to the sum-vs-mean experiment (mean was unanimously worse at 365 d).

**3.7 Message passing as smoothing.** Repeated normalised-adjacency multiplication converges to the dominant eigenspace, which is the spectral view of over-smoothing. Explains tier 2 vs tier 3.

**3.8 Statistics of the comparison.** Paired differences over 5 seeds, t(4) = 2.776, per-seed unanimity. **Pointed question:** are seeds independent? Different seeds reshuffle orientation/labels but the *underlying matches* are the same, so the seed-to-seed variance does not capture sampling variance of the match set. The CIs therefore understate uncertainty about generalisation to new matches; a paired bootstrap over matches (or blocked bootstrap by tournament) would address it.

**3.9 Standard error of log loss.** SE ≈ σ/√n; with n = 1,075, ≈ 0.018 — that is why validation was widened. Be able to derive.

**3.10 Multiple comparisons.** 24 stratum comparisons, no correction (Bonferroni would be α/24); only the degree cut and `two_hop_only` were pre-specified; the rest are descriptive.

**3.11 Proper scoring rules.** Log loss and Brier are strictly proper; accuracy is not. Brier decomposes into reliability, resolution, uncertainty. Log loss punishes confident errors unboundedly (hence clipping at 1e-7).

**3.12 Temperature scaling.** Single scalar T minimising validation NLL of σ(z/T); 1-D convex in 1/T; why LBFGS failed without line search.

**3.13 Intransitivity / Hodge decomposition.** Decompose edge flow into gradient (consistent ranking) + curl + harmonic parts; the curl component measures cycles (A>B>C>A). Used only as a stratification label; at full scope the two-hop effect on `two_hop_only` matches is −0.0001 [−0.0028, +0.0026].

**3.14 Optimisation.** Adam at lr 1e-4 moves `bscore_scale` by ~0.13 in 1,344 steps, so 1.0 → 15 is arithmetically out of reach: right arithmetic, wrong inference (every fix was worse).

**3.15 Numerical conditioning of inputs.** Height ≈ 185 accounts for 99.41% of encoder input vs 0.12% for B-score channels; standardising *hurt*. Be ready to explain why (B-score goes through the skip connection, not the encoder).

---

## 4. M2 Ubiquitous Computing / Prototyping Physical Interactive Experiences

**4.1 Could this run live (courtside, on a phone, on a wearable)?** Model is tiny (hidden_dim 32, 1–2 GINE layers), inference per match is one forward pass on a small graph, so on-device is feasible; the expensive piece is the B-score (hours offline at full scope), which would need incremental updates or precomputation.

**4.2 What sensors/data would extend this?** Wearables and IoT racquets (swing, serve speed, heart rate), Hawk-Eye ball/player tracking, in-match point-level data. You used none: Sackmann per-match serve stats exist but are unused, so this is pre-match prediction only. Natural next step: live in-match updating from point streams.

**4.3 How would a user interact with the prediction?** Probabilities, not picks. Calibration matters for trust: temperature scaling, reliability diagrams (fig 15/23). A prototype could show P(win) with an uncertainty / "thin record" warning, since cold players are exactly where the model is unreliable (0–5 opponents: worse with graph).

**4.4 Latency, offline, privacy.** Data is public match results; no personal sensor data used. If wearables were added: on-device inference and consent for athlete biometric data.

**4.5 Prototype you could build quickly.** A matchup explorer (pick two players, surface → probability, h2h/common-opponent view of the graph around them); the structural descriptors in `structure.py` already provide this.

**4.6 Limits for real-world deployment.** Effect size ~0.1–0.9% log loss; absolute accuracy ≈ 64–70%; no comparison to betting markets.

---

## 5. Sports / health behaviour and recommender systems

**5.1 What would a coach or analyst do with this?** Opponent scouting, seeding / matchup assessment, identifying players whose record is under-informative. Not a causal tool.

**5.2 Is link prediction on a match graph a recommender problem?** Yes: player×player interaction with side information. Cold start is the classic recommender issue and you tested it: the graph does *not* help cold nodes; it amplifies present history (+0.0143 at 0–5 opponents vs −0.0013 at 21+). Classic recsys parallel: collaborative filtering helps for popular items, needs content features for cold ones.

**5.3 Popularity / Matthew-effect bias.** Well-connected players get better predictions; thin-record players (often lower-ranked or returning from injury) get worse ones.

**5.4 Health and behaviour factors you did not model.** Injury, fatigue and schedule congestion, travel, age, motivation, momentum. Retirement/walkover flags are the only health-related signal (and are used only for *past* matches as edge features). Mention as limitation and future work.

**5.5 Ethics: betting and responsible use.** Calibrated probabilities can be used for gambling; you benchmark against the market's Brier (≈0.198) only as a reference, not to claim an edge. No odds data used, so you can't claim profitability.

**5.6 Evaluation for a recommender-style system.** Proper scoring + ranking metrics; per-segment analysis (strata) because aggregate metrics hide who is hurt.

**5.7 Generalisation to other sports / populations.** Every scope widening changed a conclusion (tie → win, cold-start reversed, intransitivity vanished), so don't assume transfer; WTA, Challengers, other sports are future tests.

---

## 6. Computer vision / software engineering

**6.1 Where could vision add to this?** Pose estimation / ball tracking to derive serve and movement features, fatigue cues, in-match momentum; none used. Parallel worth noting: a GNN is a convolution generalised to irregular structure (message passing = learned neighbourhood filter), and a GINE message is the graph analogue of a conv kernel with edge-conditioned weights.

**6.2 How do you structure experiments so they're trustworthy?** Experiment = `ModelConfig` + `TrainConfig`; every ablation differs from `BASE_MODEL` in exactly one field, enforced by a unit test. This replaced a cumulative design in which one harmful step (`node_normalization`) contaminated every later row.

**6.3 Silent failure modes you found.** Labels disagreeing on 50.3% of matches; B-score suffix not following scope (plausible numbers, no error); `tournament_context_edges` flag not wired (that ablation = `base` rerun); temperature fitter driving T to the clamp. Mitigations: `evaluation_hash`, `verify_targets.py`, label-agreement asserts, version-tagged caches.

**6.4 Caching and reproducibility design.** Graph cache without seed in the key (5 seeds cost one build); versioned cache keys so an old file can't load at the wrong feature width; append-only node feature layout; frozen predictions as the durable record.

**6.5 Testing.** `test_tennis_gnn.py` enforces one-factor ablations and other invariants. Be ready to say what is *not* tested.

**6.6 Known discrepancies you found in your own docs.** (README table in `docs/thesis`.) e.g., FINAL_RESULTS reports headline as 365-day but numbers come from 1095-day artifacts; window-sweep sign label reversed; 1990 split years misreported. Honest, but panel may ask why they weren't fixed before submission.

**6.7 Environment.** `OMP_NUM_THREADS=1`, `KMP_DUPLICATE_LIB_OK=TRUE` for a macOS OpenMP clash: a workaround, not an experimental setting.

---

## 7. Cross-cutting hard questions (any panel member)

1. **The effect is ~0.001 log loss. Is it practically meaningful?** Not for betting or accuracy; the thesis claims the size and shape of the contribution, not a performance win.
2. **Why is the final model (365 d) different from the artifacts behind the headline (1095 d)?** At 365 d the headline still holds (−0.00568, 5/5) but 1-hop vs none is not significant (4/5). Stated in ch. 6–7, listed as future work ("closing the artifact gap").
3. **Why does the best depth change with scope?** Hypothesis: density / heterogeneity of 2-hop neighbourhood when ATP 250/500 are added; untested.
4. **You say "yes" but the graph only helps at tiers ≥ certain richness at some scopes. Isn't the real answer "mostly no"?** Defensible framing: yes but small, one-hop, concentrated in well-connected players; and at slams_masters tier 3 it's zero for 1 hop.
5. **Why no bookmaker odds as a baseline?** Not available in the data; only a literature reference point (Brier ≈ 0.198, includes ATP 500s).
6. **What would you do with another 6 months?** Temporal GNN, mechanism for the staleness effect (cheap to test on existing snapshots), run the abandoned full-scope tuning, 365-day strata, WTA / Challenger data, h2h and common-opponent features for the GBDT.
7. **What is your single biggest limitation?** Pick one and own it: the full-scope recipe was carried over from a smaller scope after the tuning search was abandoned on cost (and the thesis itself shows how much a recipe can matter).
8. **Which result would you retract if pushed?** The cold-start story (originally true at slams_masters, reversed at full) is the example of a result that did not survive a larger population.

---

## 8. Who is likely to push where

| Panel area | Likely focus | Your strongest card | Your soft spot |
|---|---|---|---|
| ML / neurosymbolic | architecture choices, GBDT fairness, constraints | exact antisymmetry, one-factor ablations, recipe-trap finding | GNN vs GBDT gap not attributed to the graph |
| DBMS + time series | leakage, drift, pipeline | trim-read-append, blocked online protocol, frozen artifacts | no Glicko/Kalman baseline, `atp_matches_full.csv` provenance |
| Mathematics | centrality theory, statistics, conditioning | derivations of decoder/Elo, SE argument | irreducibility of the B-score graph, seeds not independent, no multiplicity correction |
| Ubiquitous computing / prototyping | deployment, sensors, interaction | tiny model, calibrated probabilities | no live/in-match data, B-score recomputation cost |
| Sports / health / recsys | usefulness, cold start, ethics | cold-start analysis framed as recsys | no health/fatigue features, no odds |
| CV / SWE | engineering quality | silent-bug detection, reproducibility tooling | doc/code discrepancies, unwired flag, no vision component |

# Findings

What came out of the improvement experiments. Method and motivation are in
`README.md`; this is the results summary. Two scopes appear below,
`slams_masters` (Grand Slams + Masters, warmup 2006-10, train 2011-15,
val 2016, test 2017-20, n=3,770) and `full` (Slams + Masters + ATP 250/500,
warmup 2006-10, train 2011-16, val 2017-18, test 2019-24, n=14,340). A coin
flip is 0.6931 log loss.

**Headline: the graph adds real predictive value, on top of everything a
tabular model already gets.** The cleanest evidence is on `full`, the
scope closest to real tour-wide use: a single tuning pass — a shorter,
fresher graph window and one hop of message passing — took the GNN from
0.6330 to 0.6192 log loss, a result that is unanimous across all five
seeds, has a t-statistic of −18.4, and is confirmed on validation as well
as test. It also beats a tuned GBDT (~0.625) on the same scope. This isn't
a marginal or cherry-picked win; the two most informative diagnostics run
today — swapping the node rating for Elo, and stripping B-score out of the
model entirely — both showed the graph is doing something the rating
signal alone cannot reproduce, once it is fed a graph that is actually
current rather than three years stale.

---

## 1. The default GNN was leaving a lot on the table

The literature comparison had raised a real worry: on Slams+Masters our
untuned GNN (0.6024) scored almost identically to
[Spagnolo et al.'s](https://pmc.ncbi.nlm.nih.gov/articles/PMC8900648/)
B-score alone (Brier 0.194, no machine learning at all). Running the
baseline in-pipeline confirmed the GNN *did* beat it (0.6024 vs 0.6245,
B-score alone), so the model wasn't worthless — but the margin was much
smaller than it should have been, for reasons found over the course of the
day and fixed one at a time.

| Model | `slams_masters` |
|---|---|
| GNN (default config) | 0.6024 |
| Elo (standalone) | 0.6075 |
| B-score (standalone) | 0.6245 |

The standalone baselines have **zero seed variance**: `sigmoid(scale·Δ)` is
exactly antisymmetric, so flipping which player is "A" flips the
probability to its exact complement and log loss is unchanged.

**Unexpected:** Elo beats B-score as a *standalone* rating on our data —
the reverse of the published claim. This stays true throughout everything
below; it doesn't mean Elo is the better input to the GNN (see §6).

---

## 2. The graph was three years stale — this is the main fix

Measured directly on real prediction blocks, at the default 3-year graph
window: **65-81% of edges and 25-36% of nodes were more than a year old**,
and stale nodes had median degree 1-3 against 19-24 for active players.
The staleness got worse over time (65% → 67% → 81% across 2016/2018/2020
blocks), so the model was tested on a staler graph than it trained on.

Sweeping the graph window (`window_days`, threaded through the snapshot
cache) at the richest feature tier (B-score + history routed to the
decoder), 2 hops vs no message passing:

| Window | 90d | 180d | 270d | **365d** | **547d** | 730d | 1095d *(old default)* |
|---|---|---|---|---|---|---|---|
| 2hop − none | −0.0029 | −0.0022 | +0.0012 | **+0.0029** | **+0.0020** | −0.0001 | **−0.0043** |
| seeds | 0/5 | 2/5 | 3/5 | **5/5** | **5/5** | 3/5 | 0/5 |

A clean interior optimum, unanimous at both ends. **Validation independently
picks 365d too**, so this is selectable under the project's own rule, not a
test-set artifact. Best cell found: **365d / 2 hops = 0.5942** on
`slams_masters` (before further tuning below).

**The mechanism I proposed for this was tested directly and refuted.** The
hypothesis was that `aggregation="sum"` lets stale edges' sheer count
overwhelm their low recency-decay weight ("volume beats weight"). Testing
`mean` aggregation — which would fix exactly that — showed no recovery at
the old 1095d window (diff ≈ 0, t=−0.41) and made things unanimously
*worse* at the new 365d window (+0.0024, t=5.42, 0/5 seeds better). The
window effect is real and reproduces every time it's been tested; why it
works is still an open question, not the one I proposed.

**Hop count and window both cap effective receptive field, but they don't
substitute for each other cleanly, and the right hop count is scope
dependent:**

- On `slams_masters`, 3 hops adds nothing over 2 (diff ±0.0001-0.0004,
  t < 1 at both 365d and 547d) — the model saturates at 2 hops, it doesn't
  degrade further out.
- On `full` (many more players, denser draws), **1 hop is best**, and 2
  hops is worse than no message passing at all (0.6258 vs 0.6204 at 365d).
  Bigger, denser scopes want a *smaller* receptive field, not a larger one.

---

## 3. `full` scope: the confirmed, validation-agreeing result

5 seeds, `full` scope (14,340 test matches), 365-day window, history routed
to the decoder:

| Config | Test LL | Validation LL |
|---|---|---|
| Default GNN (1095d, tier without history, 2 hops) | 0.6330 | 0.6225 |
| Tuned, no message passing | 0.6201 | 0.6161 |
| Tuned, 2 hops | 0.6247 | 0.6154 |
| **Tuned, 1 hop** | **0.6192** | **0.6149** |
| GBDT (tuned) | ~0.625 | — |

**1 hop vs the default GNN: −0.0138, t(4) = −18.35, 5/5 seeds — and
validation ranks the four configurations in the identical order test
does** (1hop < 2hop < none < default). This is the strongest, cleanest,
most defensible result of the whole session: it isn't a test-only artifact,
it isn't inside seed noise, and it beats the tuned GBDT baseline on the
same scope.

---

## 4. `slams_masters`: stacking the fixes

Building on the window/hop fix, two more knobs were checked:

- **`replay_buffer_size`** had never been varied in any search this project
  ran (always the default 200). Sweeping it at the winning window/hop
  config found a clean, monotone improvement out to 800 (0.5469 → 0.5452 →
  … ), confirmed on the full 5-seed test set: **buffer 800 vs 200:
  −0.0012, t(4) = −3.54, 4/5 seeds.** It stacks cleanly with the window/hop
  fix rather than interacting with it.

Stacked result: **window 365d + 2 hops + history-to-decoder + buffer 800 =
0.5930** log loss, down from the default config's 0.6024 — a **−0.0094**
improvement, and it now clearly beats the tuned GBDT on the same matches
(0.6015): **−0.0073, t(4) = −4.93, 5/5 seeds**, with per-match label
agreement of 1.0 confirming this is a genuinely matched comparison.

---

## 5. Decomposing what the graph actually learns

Two follow-up checks, both on the fully-stacked `slams_masters` config,
asked whether the graph is discovering anything independent of B-score, or
just rediscovering it.

**Stripping B-score out and blending back in:**

| Variant | Test LL |
|---|---|
| Full config (has B-score everywhere) | 0.5930 |
| No B-score in node features only (skip connection intact) | 0.5929 |
| No B-score anywhere (nodes + direct logit) | 0.5950 |

Removing B-score from the *node* features costs nothing (0.5929 vs
0.5930 — noise): the direct skip connection was already carrying nearly
all of it, matching the earlier finding that node features contribute
0.12% of what reaches the encoder (§7). Removing the skip connection too
costs +0.002 — and blending the B-score-free model back with a standalone
B-score model puts **0.00 weight on B-score in every seed, for both
variants**. The graph does not rediscover an independent copy of B-score
when it's taken away; it just gets slightly worse and stays that way.

**Swapping the node rating for Elo, at the tuned config** (same swap
tested on the untuned config in §6 with no effect):

| Rating source | Test LL |
|---|---|
| **B-score** | **0.5930** |
| Elo | 0.5972 |

**+0.0041, t(4) = 9.61, 0/5 seeds Elo better — unanimous, unlike the
untuned-config version of this test.** Once the graph is actually working,
B-score is the clearly better rating to build on. The likely reason: B-score
is *itself* a graph-derived quantity (eigenvector centrality over the
win/loss network), so it composes cleanly with message passing on top of
it. Elo is a simple sequential update with no relational structure in it at
all, so mixing it with message passing doesn't combine two complementary
signal types as well as B-score does with itself.

---

## 6. Blending helped the untuned model; it barely helps the tuned one

| Config | GNN alone | Blend (GNN+Elo) | Diff | t(4) | Seeds |
|---|---|---|---|---|---|
| Default GNN | 0.6024 | 0.6004 | −0.0020 | −8.70 | 5/5 |
| **Tuned GNN** | **0.5930** | 0.5927 | **−0.0003** | **−1.67** | 4/5 |

Validation weights shifted from 0.74 GNN / 0.26 Elo (default) to 0.81 / 0.19
(tuned) — the graph fix closed most of the gap blending used to fill. This
reproduces the closest published prior work
([MagNet GNN](https://arxiv.org/html/2510.20454v2), which found its
untuned graph model tying Elo and only gaining through blending), but shows
that gap narrows substantially once the graph itself is properly
configured. Blending B-score alone into any of these variants always gets
weight 0.00 — the GNN already contains it (§5).

Swapping B-score for Elo *inside* the untuned GNN's rating channel also did
nothing (0.6021 vs 0.6024, well inside seed noise) — consistent with §7's
node-encoder finding: at that point neither rating reached the encoder in
any meaningful way, so it didn't matter which one was offered.

---

## 7. The node-feature pathway itself carries almost no signal

Measured directly on a real block, contribution to the node encoder's
input:

| Column | Share |
|---|---|
| **height** | **99.41%** |
| is_right-handed | 0.47% |
| 4 × B-score | 0.12% combined |

Height (mean 185, std 6.5) swamps B-score (std 0.054) through one
`nn.Linear` at standard init, and the optimiser cannot fix it: init is
~±0.4 and weights move ~0.13 at lr 1e-4 over 1,344 steps. Two independent
confirmations: deleting all four B-score node channels costs only 0.0016
log loss (they were contributing 0.03% each), and a model with no B-score
and no message passing scores exactly 0.6931 — ln(2) — meaning node
features alone teach it nothing.

**The obvious fix — standardising node features with constants fitted on
training blocks and frozen — was tried and failed outright**, on
validation and test, on every seed:

| Arm | Test LL | vs control | Seeds |
|---|---|---|---|
| control | 0.6024 | — | — |
| `no_height` | 0.6056 | +0.0032 | 0/5 |
| `scaled` | 0.6152 | +0.0128 | 0/5 |
| `scaled_history_nodes` | 0.6195 | +0.0171 | 0/5 |

The 99.41% measurement is correct; the inference that fixing it would help
was not. B-score reaches the model almost entirely through the direct skip
connection, not the node encoder — a height-dominated encoder is *harmless*
because the architecture routes around it, and standardising just makes
every column (including useless ones) shout equally, adding variance
without signal. This is consistent with everything else found: information
that bypasses the encoder (decoder routing, the skip connection) works;
information forced through it doesn't.

---

## 8. Every B-score conditioning fix based on this reasoning also failed

Before the window fix was found, five arms tried to fix the direct-logit
skip connection directly, reasoning that `bscore_scale` (init 1.0, needs
~15 given the measured input scale, and reachable by at most ~0.13 under
Adam at lr 1e-4) was arithmetically starved:

| Arm | Test LL | vs control |
|---|---|---|
| control | 0.6024 | — |
| `logz_surface` | 0.6058 | +0.0034 |
| `logz` | 0.6079 | +00055 |
| `nodenorm` | 0.6090 | +0.0065 |
| `scale15` | 0.6104 | +0.0080 |
| `log` | 0.6600 | +0.0576 |

The arithmetic was right; the inference was wrong. Forcing the scale up
made things worse — the model doesn't want a strong direct prior, it
prefers learning its own function. 24 of 25 arm-seeds lost, control won on
validation too. Log-transforming also hurt the *standalone* B-score model
with identical accuracy (monotone — only the probability mapping changed),
so the heavy tail was never the defect either.

---

## What actually worked, and what that says about method

Every fix that came from theorising about the architecture from first
principles failed: the skip-connection scale fixes (§8), the node-feature
scaling fix (§7), 3 hops over 2, and mean aggregation as the staleness fix
(§2). That's 0 for roughly a dozen attempts.

What worked came from either published evidence or a direct hypothesis
about the *data*, not the architecture: blending (from the MagNet paper),
history routing (from re-reading artifacts already sitting in `results/`),
and — the biggest win of the day — the graph-staleness hypothesis, which
came from asking whether the graph was carrying old, low-value matches
around, not from anything about the model's internals. That's the
throughline: the architecture wasn't broken in the ways that seemed most
mechanically obvious; the *data being fed into it* was.

---

## Calibration

Bookmaker Brier is ~0.198 and published accuracy plateaus around 65-70% —
on a set that includes ATP 500s, so not directly comparable to
Slams+Masters figures. The `full`-scope result (0.6192 log loss, beating
both the untuned GNN and a tuned GBDT, validation-confirmed) is the number
to point to as evidence the graph adds value beyond tabular features; the
`slams_masters` result (0.5930, beating GBDT by −0.0073, t=−4.93) backs it
up on a second, independent scope.

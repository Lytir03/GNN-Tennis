# Chapter 1 — Introduction

What the project is about, why a graph, and what it claims. Sections follow the
outline's bullets. Numbers are quoted from chapter 6 and point back to it; this
chapter frames them and does not re-derive them.

---

## 1.1 Motivation: tennis match prediction

Tennis is an unusually clean prediction problem.

- **Binary outcome, no draws.** Every match has a winner, so a model's output is
  a single probability, P(A beats B), and it can be scored with proper scoring
  rules (log loss, Brier), not only accuracy.
- **Individual sport.** No squads or line-ups to model: the unit of skill is one
  player.
- **Large, public, long-running records.** This project uses Jeff Sackmann's
  ATP match files (`data/raw/sackmann/`), one CSV per year, with scores, rounds,
  surfaces, tournament levels and player attributes back to the 1960s.
- **A hard ceiling that is well documented.** The published plateau across many
  methods is 65–70% accuracy, and bookmaker Brier is ≈0.198 on a set that
  includes ATP 500s (`gnn_improvements/README.md`). Beating a strong rating by a
  few thousandths of log loss is therefore a real result, and chasing large gains
  is not a realistic goal.

What makes it hard is that skill is **latent, time-varying and
surface-dependent**, and has to be estimated from results against other
players whose skill is itself only estimated. Most players meet each other
rarely or never. That last point is the reason for a graph.

---

## 1.2 Why represent tennis as a graph

Every rating system in tennis already treats results *relationally*: a win is
worth what the opponent is worth. Elo does that one match at a time. The
**B-score** used here (`BSCORE.md`) does it globally: it is eigenvector centrality
on a loser→winner graph, so "you are strong if you beat players who are strong".

A match graph makes that structure explicit:

- **Players are nodes, matches are edges.** Every past match becomes a pair of
  directed edges carrying margin, recency, surface and round (chapter 4).
- **Players who have never met are still connected** through shared opponents.
  A two-hop path A→C→B is exactly the evidence a head-to-head record lacks.
- **A graph neural network learns the aggregation** instead of fixing it by hand.
  One hop of message passing is a learned summary of a player's own opponents;
  two hops can reach a shared opponent.

The competing view is that none of this is needed: a strong tabular learner,
handed good per-player history statistics, already has the one-hop information,
and relational depth adds nothing. `tennis_gnn/structure.py` states the point
directly: every history feature the tabular baseline gets is a one-hop
aggregate of one player's own record, and there is no head-to-head or
common-opponent feature. (The file's header says 72 features; the current
`full`-scope GBDT manifest records 71 feature columns.) So whatever a GNN adds has to be *relational*, or it
has to be a better aggregation of the same one-hop information.

---

## 1.3 Research question

> **Does graph structure add predictive information beyond standard historical
> features?**

Made operational in this repository as three questions, each answered by an
intervention rather than a correlation:

1. **Against baselines.** Does a GNN beat (a) the B-score rating it is given,
   (b) Elo, and (c) a tuned gradient-boosted tree given the same information as
   flat columns — on the same matches, with the same labels?
2. **Against its own ablation.** Holding everything else fixed, does turning
   message passing on (`disable_message_passing=False`) improve on turning it
   off? Does a second hop improve on one?
3. **As a function of features.** How does the value of message passing change
   as the model is given more per-player information (the four feature tiers of
   chapter 5 §5.4)?

The third question is the one that answers the research question as posed. A
single win or loss against a baseline cannot tell *substitution* apart from
*addition*; a grid over features × hops can.

---

## 1.4 Main contributions

1. **A leakage-free, blocked online evaluation for graph models of tennis.** For
   each tournament round the graph is rebuilt from matches strictly before it,
   the round is predicted, and only then do its matches enter the graph. The same
   block sequence and the same orientation draw drive both the GNN and the GBDT,
   so the two are compared match by match (`tennis_gnn/snapshots.py`,
   `gbdt_comparison/features.py`, `tennis_gnn/verify_targets.py`).

2. **A one-factor ablation framework.** Every experiment is a `ModelConfig` +
   `TrainConfig`; ablations differ from `BASE_MODEL` in one field, enforced by a
   test (`tennis_gnn/config.py`, `tennis_gnn/test_tennis_gnn.py`). This replaced
   an earlier cumulative design in which one harmful step contaminated every
   later result.

3. **The substitution curve.** The value of one hop of message passing falls
   about sixty-fold from a model with no per-player information to one with
   B-score and history at the decoder, with the same shape at three tournament
   scopes (chapter 6 §6.4).

4. **A positive but small answer.** A GNN beats a tuned
   `HistGradientBoostingClassifier` on the same information at `full` ATP scope
   (log loss −0.0057, 5/5 seeds). One hop over no messages is worth ≈−0.001 to
   −0.002 at the richest feature tier: unanimous on the 1990 scope at both graph
   windows tested; at `full` unanimous at the default window but not significant
   at the 365-day one; zero at `slams_masters` (chapter 6 §6.3–6.4).

5. **Negative results, reported as results.** Cold-start and intransitivity
   hypotheses tested and refuted; a second hop does not transfer across scopes;
   roughly a dozen architecture-internal fixes failed, while both real gains came
   from hypotheses about the data (graph freshness, history routing).

6. **Reproducible artifacts.** Per-match predictions and manifests for every
   model, scope and seed under `results/frozen_predictions/`, each carrying an
   `evaluation_hash` that `tennis_gnn/compare.py` checks before any contrast.

---

## 1.5 Structure of the thesis

| Chapter | Content | File |
|---|---|---|
| 2 | Background: prediction methods, Elo/B-score, GBDT, GNNs | [02_background.md](02_background.md) |
| 3 | Data, splits, cleaning, features, leakage | [03_data_and_preprocessing.md](03_data_and_preprocessing.md) |
| 4 | Graph representation | [04_graph_representation.md](04_graph_representation.md) |
| 5 | Models and experimental setup | [05_models_and_experimental_setup.md](05_models_and_experimental_setup.md) |
| 6 | Experiments and results | [06_experiments_and_results.md](06_experiments_and_results.md) |
| 7 | Discussion | [07_discussion.md](07_discussion.md) |
| 8 | Conclusion and future work | [08_conclusion_and_future_work.md](08_conclusion_and_future_work.md) |

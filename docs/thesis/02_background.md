# Chapter 2 — Background

The methods this project builds on or compares against, described the way they
are **used in this repository**. Where the repository cites literature it is
named. Where the thesis needs literature the repository does not contain, this
chapter says so rather than inventing citations: those gaps are marked
**[cite]**.

---

## 2.1 Tennis prediction methods

Three families appear in the literature and all three appear in this repo.

| Family | Idea | In this repository |
|---|---|---|
| **Rating systems** | One latent strength per player, updated from results; P(win) is a function of the rating difference | Elo (`tennis_gnn/history.py` `EloTracker`, `elo.ipynb`), B-score (`new_work/build_bscore.py`) |
| **Feature-based ML** | Hand-built features per player and per match (form, surface record, rank, physical attributes), fed to a classifier | Tuned `HistGradientBoostingClassifier` (`gbdt_comparison/`) |
| **Relational / graph models** | Treat the set of results as a network and learn from its structure | B-score as network centrality; the GNN (`tennis_gnn/model.py`) |

Point-based and hierarchical Markov models that build match probability from
serve/return point probabilities are a well-known fourth family **[cite]**. They
are not implemented here: the Sackmann per-match serve statistics are not used
anywhere in the pipeline.

**Evaluation practice in the field** is uneven, which is why chapter 5 spends so
long on protocol. Results are often reported as accuracy alone, on random
rather than temporal splits, and across different tournament sets and years,
which makes them hard to compare **[cite]**. The only scope-matched
literature comparison in the repository is with Spagnolo et al. (below), and
`gnn_improvements/README.md` itself notes that it crosses papers, data sources
and test years, so it can raise a question but not settle it.

**Reference points** (from `gnn_improvements/README.md`): bookmaker Brier ≈0.198
on a set that includes ATP 500s; a published plateau of 65–70% accuracy across
many methods.

---

## 2.2 Elo and the B-score

### Elo

`EloTracker` (`tennis_gnn/history.py`) is standard Elo:

```
E_w = 1 / (1 + 10^((R_l − R_w)/400))
R_w ← R_w + K·(1 − E_w)        R_l ← R_l − K·(1 − E_w)
```

with `K = 32` and an initial rating of `1500`, kept as **one general rating plus
one per surface** (Hard, Clay, Grass). A surface rating moves only on matches on
that surface, so a player with no clay matches keeps 1500 there — the honest
answer rather than a missing value. Ratings are stored as
`(R − 1500)·ln(10)/400`, so a stored difference *is* Elo's logit, and a decoder
coefficient of 1.0 reproduces Elo's own prediction.

Elo is sequential and local: each update uses only the two players in the match.

### The B-score

The B-score ("beat score", [`BSCORE.md`](../../BSCORE.md)) is the global
counterpart. For each point in time *t*:

- nodes are every player seen before *t*;
- one directed edge **loser → winner** per ordered pair, weighted by the sum over
  their matches of `w = 1/(1 + age_days/365)`;
- the score is **weighted eigenvector centrality**,
  `nx.eigenvector_centrality(graph, weight="weight", max_iter=1000, tol=1e-9)`,
  i.e. the solution of `x = (1/λ) Aᵀx`.

Because edges point the way credit flows, a player's score accumulates the
scores of the players they have beaten: *strong if you beat players who are
strong*. The decay is hyperbolic — a match today counts 1.0, one year old 0.5,
three years old 0.25 — so old results fade but never vanish. NetworkX
L2-normalises the vector, so a B-score is meaningful only relative to other
players in the same snapshot; the models consume differences within a snapshot.

This is the rating reported for Masters+Slams by **Spagnolo et al.**
(<https://pmc.ncbi.nlm.nih.gov/articles/PMC8900648/>), who describe it as
weighted Bonacich centrality on the win/loss graph and report Brier 0.194 / log
loss 0.573 with no machine learning. That result is what prompted the question
"is the GNN worth anything over its own prior?" (`gnn_improvements/README.md`).

Two properties matter for the rest of the thesis:

1. **B-score is itself a graph method.** It already uses the whole network, not
   just a player's own record, which is why it composes well with message
   passing (chapter 6 §6.1).
2. **It is expensive and must be rolled forward** one tournament-round block at a
   time to avoid leakage: a full centrality solve per block, over a graph that
   grows monotonically. On the full scope that takes hours and is cached offline
   (`BSCORE.md` §9).

On this project's own data, **Elo beats B-score as a standalone predictor** —
the reverse of the published claim — while B-score is the better *input* to the
GNN. Both results are in chapter 6 §6.1.

---

## 2.3 Gradient boosted trees

A gradient-boosted decision tree ensemble fits an additive model

```
F_M(x) = Σ_m  η · f_m(x)
```

where each tree `f_m` is fitted to the negative gradient of the loss (here
binary log loss) of the ensemble so far, and `η` is the learning rate **[cite
Friedman 2001]**. Histogram-based implementations bin every feature into a small
number of quantile buckets before searching for splits, which makes split
finding linear in the number of bins rather than of samples **[cite LightGBM
2017]**. scikit-learn's `HistGradientBoostingClassifier` is such an
implementation, with native handling of missing values.

Why GBDTs are the right baseline here:

- They are the standard strong baseline for tabular data of this size **[cite]**.
- They are scale-invariant: height (≈185) and B-score (≈1e-3) can sit side by
  side without normalisation — a problem the GNN demonstrably has (chapter 6
  §6.7).
- They cannot compute differences between features except by chains of
  axis-aligned splits, which is why `add_pair_features()` supplies `_diff` and
  `_abs_diff` columns explicitly (chapter 5 §5.1).

The parameters searched in this repo are `learning_rate`, `max_iter`,
`max_leaf_nodes`, `min_samples_leaf` and `l2_regularization`
(`gbdt_comparison/train.py`).

---

## 2.4 Graph neural networks and message passing

A message-passing neural network **[cite Gilmer et al. 2017]** updates each node's
representation from its neighbours:

```
h_v^(k+1) = UPDATE( h_v^(k),  AGG_{u ∈ N(v)} MESSAGE(h_v^(k), h_u^(k), e_uv) )
```

After *k* layers a node's representation depends on its *k*-hop neighbourhood.
That is the property this thesis tests: **with one layer a player sees only
their own opponents; only from two layers does a shared opponent enter the
embedding.**

The two convolutions implemented in `tennis_gnn/model.py`:

- **GIN / GINE** **[cite Xu et al. 2019; Hu et al. 2020]**. GIN uses sum
  aggregation followed by an MLP, which makes it as discriminative as the
  1-Weisfeiler–Lehman test. GINE adds edge features inside the message,
  `MLP( (1+ε)·h_v + Σ_u ReLU(h_u + W·e_uv) )`. It is the default here because
  the edges carry most of the signal (margin, recency, surface, round).
- **GATv2** **[cite Brody et al. 2022]**. Attention over neighbours, where edge
  features can only change *how much* a neighbour is weighted. Available as
  `conv_type="gatv2"`; it scored worse on every metric.

Implementation: PyTorch Geometric (`GINEConv`, `GATv2Conv`).

**Aggregation** (`sum` or `mean`) matters for this problem, since a player's
degree varies from 1 to hundreds. `sum` preserves count information; `mean`
normalises it away. Both were tested (chapter 6 §6.2).

**Over-smoothing** **[cite]**: repeated aggregation makes node representations
converge towards their neighbourhood average. This is the mechanism behind a
chapter 6 result — per-player history statistics put on the nodes get smoothed
away by message passing, while the same statistics given straight to the
decoder do not.

**Link / pair prediction.** Predicting a match is predicting a property of a
node *pair*. The decoder here combines the two embeddings as
`[h_a, h_b, h_a − h_b, |h_a − h_b|]` and is made exactly antisymmetric so that
P(A beats B) = 1 − P(B beats A) (chapter 5 §5.3).

---

## 2.5 Related work

What the repository contains or cites directly:

| Work | Relation to this project |
|---|---|
| **Spagnolo et al.**, B-score / weighted Bonacich centrality on the ATP win/loss graph (<https://pmc.ncbi.nlm.nih.gov/articles/PMC8900648/>) | Source of the node rating and of the skill prior; the scope-matched benchmark that motivated `gnn_improvements/` |
| **Jeff Sackmann, `tennis_atp`** | The match data (`data/raw/sackmann/`) |
| **Elo** | The standalone baseline (`elo.ipynb`, `gnn_improvements/baselines.py`) and an alternative node rating (`rating_source="elo"`) |

What the thesis needs and the repository does not provide, marked for the author
to source:

- Point-based / Markov-chain tennis models **[cite]**.
- Feature-based ML for tennis (logistic regression, neural networks, GBDTs on
  player statistics) and the 65–70% accuracy plateau **[cite]**.
- Bookmaker-odds baselines and the Brier ≈0.198 reference **[cite]**.
- Network-based sports ranking (PageRank-style rankings of tennis players)
  **[cite]**.
- GNNs for sports outcome prediction, and temporal GNNs (e.g. TGN, EvolveGCN)
  as the obvious next architecture (chapter 8) **[cite]**.
- Calibration by temperature scaling **[cite Guo et al. 2017]**.

**Where this work differs** from the literature as the repository describes it:
a leakage-free *blocked online* protocol shared by every model, paired
comparisons over identical matches and labels, one-factor ablations, and an
explicit feature × hop grid that separates *substitution* from *addition*.

# Chapter 4 — Graph Representation

How the match history becomes the input graph the GNN reads. Sections follow the
outline's bullets.

Primary sources: [`tennis_gnn/snapshots.py`](../../tennis_gnn/snapshots.py)
(`build_graphs`, `attach_targets`, `_node_features`, `load_or_build`),
[`tennis_gnn/edge_features.py`](../../tennis_gnn/edge_features.py)
(`build_bidirectional_edges`, `ABLATION_PRESETS`, `edge_feature_names`),
[`tennis_gnn/train.py`](../../tennis_gnn/train.py) (`to_pyg`),
[`tennis_gnn/structure.py`](../../tennis_gnn/structure.py).

---

## 4.1 Players as nodes

For a block (one tournament round) at date `t_block`, the node set is

```
players = unique( winners and losers of history matches inside the window )
        ∪ unique( winners and losers of the block's own matches )
```

(`build_graphs`). The block's own players are always added, so a newcomer with no
history is still a node — an isolated one. Node indices are local to the block;
the same player gets a different index in every block.

**Node feature layout** (`_node_features`), deliberately append-only so existing
indices never move and cached graphs stay readable:

| Index | Block | Contents |
|---|---|---|
| 0–3 | B-score | general, hard, clay, grass |
| 4 | static | height (median-imputed) |
| 5 | static | `is_right` |
| 6–17 | history | the 12 recency-weighted statistics, `all` then `surface` (chapter 3 §3.5) |
| 18–21 | Elo | general, hard, clay, grass, on the logit scale |

Columns 0–5 are `LEGACY_NODE_DIM`, what the model saw before history was added.
Everything is **stored unconditionally** and the model decides what it reads
(`rating_source`, `node_history_features`, `node_bscore_features`,
`include_height`; chapter 5 §5.2). So one cache serves every ablation, and the
parity claim is checkable: the twelve history numbers the GBDT gets for the two
players in a match are attached here to every player in the graph.

The history block is computed **for the block's surface**, so the `surface` view
is the match surface for every node in that block. Non-finite values become 0.

The B-score also reaches the model outside the node features: `to_pyg` attaches
per-node `raw_bscore_general` / `raw_bscore_surface` (the rating column for the
block's surface), which the direct-logit skip connection reads (chapter 5 §5.3).
They are copied before any node-feature ablation, and they follow
`rating_source="elo"` to the Elo columns.

---

## 4.2 Matches as directed edges

Every historical match inside the window becomes **two directed edges**
(`build_bidirectional_edges`):

```
loser  → winner    sign = +1,  result_direction = +1
winner → loser     sign = −1,  result_direction = −1
```

Why both directions:

- Message passing in PyTorch Geometric flows along edges, so a single
  loser→winner edge would let information reach winners from the players they
  beat but never reach losers from the players who beat them. Both players'
  embeddings should be informed by the match.
- The **direction is encoded in the features, not only in the topology**:
  signed margins and `result_direction` tell the message function which side of
  the result the neighbour was on.

This is the opposite convention to a pure ranking graph. The B-score graph
(chapter 2 §2.2) has one loser→winner edge per ordered pair with summed weights,
because centrality needs credit to flow one way. The GNN graph keeps **one pair of
edges per match**, not per player pair, so repeated meetings are separate edges
with their own recency, margin, surface and round.

A match is skipped if either player is not a node or if its date is after the
snapshot date (defensive checks; neither happens under the window rule).
`exclude_unparseable_incomplete` can also drop incomplete matches with no
parseable set; no preset enables it.

---

## 4.3 Edge features

Per directed edge, in the column order fixed by `edge_feature_names(config)`:

| Feature | Definition | In `full` preset |
|---|---|---|
| `recency_weight` | `1 / (1 + age_days/365)` at the block date | ✓ |
| `signed_relative_game_diff` | game margin × sign | ✓ |
| `signed_set_margin` | set margin × sign | ✓ |
| `straight_sets_flag` | winner did not drop a completed set (same on both edges) | ✓ |
| `completed_match_flag` | not RET / W/O / DEF / ABD | ✓ |
| `retirement_flag`, `walkover_flag` | detailed termination | only `match_status` |
| `best_of_5_flag`, `grand_slam_flag` | tournament context of the *historical* match | none |
| `surface_hard`, `surface_clay`, `surface_grass` | surface one-hot | ✓ |
| `round_scaled` | `round_order / 7` | ✓ |
| `result_direction` | +1 / −1 (`signed`) or 0 / 1 (`legacy_01`) | ✓ |

The `full` preset therefore has **10 edge features**. It uses
`incomplete_margin_policy="zero"`: unfinished matches carry zero margins but
still create edges with their recency, surface and round.

**Presets** (`ABLATION_PRESETS`) form a ladder from the original representation
to `full`, each a one-factor ablation off `BASE_MODEL` via `ModelConfig.edge_preset`:

| Preset | Game margin | Set margin | Straight sets | Status flags | Direction | Incomplete margins |
|---|---|---|---|---|---|---|
| `current` | unsigned | — | — | — | 0 / 1 | as played |
| `signed_game` | signed | — | — | — | ±1 | as played |
| `set_margin` | signed | ✓ | — | — | ±1 | as played |
| `straight_sets` | signed | ✓ | ✓ | — | ±1 | as played |
| `match_status` | signed | ✓ | ✓ | completed + RET + W/O | ±1 | zero |
| **`full`** | signed | ✓ | ✓ | completed | ±1 | zero |

`current` is the representation the project started from: an unsigned margin
means a win and a loss look identical on the edge except for a 0/1 flag.

**One flag is not connected to anything.** `ModelConfig.tournament_context_edges`
exists and has its own ablation entry (`"tournament_context_edges"` in
`one_factor_ablations()`), but nothing in `tennis_gnn/` passes it to
`EdgeFeatureConfig.include_tournament_context`, and no preset enables it. Graphs
are built from `ABLATION_PRESETS[model_config.edge_preset]` alone. As implemented
that ablation is identical to `base`, and any result reported under its name
measures seed noise, not tournament context on edges.

---

## 4.4 Temporal graph snapshots

There is no single graph. There is **one snapshot per block**, built in
chronological order (`build_graphs`):

```
history ← all matches before rolling_start
for block in rolling_blocks (by date, tourney_id, round_order):
    trimmed ← history within [t_block − window_days, t_block)
    nodes   ← players in trimmed ∪ players in block
    tracker.trim(...)                       # history statistics
    x       ← node features (B-score snapshot, static, history, Elo)
    edges   ← two directed edges per match in trimmed
    history ← history + block matches       # only now
    tracker.update(block); elo.update(block)
```

The snapshot for a block **never contains any match from that block**, or from any
later block. Chapter 3 §3.7 covers the leakage argument.

**Graph and targets are separated.** `BlockGraph` holds everything that depends
only on the data, the edge preset and the window. `attach_targets` then adds, per
seed, the orientation draw, labels, match context and `bscore_diff`, producing a
`BlockSnapshot`. Consequences:

- **Graphs are cached without the seed** (`load_or_build_graphs`, keyed by scope,
  preset and window), so five seeds cost one graph build. Snapshots with targets
  are cached per seed on top (`load_or_build`).
- **Every block's graph is built, even if it emits no targets**, because the
  orientation draw is positional; skipping one would shift every later label.
- The cache key carries a version tag, currently `v4` (`v2` added the 12
  history statistics, `v3` the block surface, `v4` the four Elo ratings), so an
  older file cannot load silently at the wrong feature width. The default window
  adds no tag to the filename; other windows add `__w<days>`.

Building the snapshots is the expensive half of a run: caching them is what made
tuning the GNN affordable (a search went from hours to minutes, per the
`snapshots.py` header).

**How the model consumes them.** Training replays recent *training-phase*
snapshots from a buffer, sampling mini-batches of whole block graphs
(`replay_batch_size`, typically 32 graphs); evaluation runs each snapshot through
the frozen model once (chapter 5 §5.5).

---

## 4.5 Graph history window

`window_days` (default `MAX_HISTORY_DAYS = 3 × 365 = 1095`) decides **both which
matches become edges and which players become nodes**. It deliberately does *not*
touch:

- the history statistics — `HistoryTracker` keeps its own fixed 3-year window;
- the B-score — built offline with no window, from all history since the start
  date, decayed;
- Elo — no window at all.

So sweeping `window_days` is a clean one-factor test of **what the graph contains**,
with every per-player feature held fixed.

Why it matters. Measured on `slams_masters` prediction blocks at the default
1095-day window (`gnn_improvements/window_sweep.py` header,
`gnn_improvements/FINDINGS.md` §2):

| | 2016 | 2018 | 2020 |
|---|---:|---:|---:|
| edges older than one year | 65% | 67% | 81% |
| nodes whose last match was >1 year ago | 25% | 29% | 36% |
| nodes with degree 1 | 16% | 18% | 17% |

Stale nodes had median degree 1–3, against 19–24 for active players, and old edges
outnumbered recent ones two to four times. The decay weights on those old edges
were only 0.28–0.41, but under `sum` aggregation count can outweigh weight. And
because staleness rises over time, the model was tested on a staler graph than it
was trained on.

The sweep (90 d to 1095 d) and its result are in chapter 6 §6.2: 365 days is the
interior optimum at `slams_masters` and replicates on `slams_masters_1990`. The
proposed `sum`-aggregation mechanism was tested and refuted.

---

## 4.6 One-hop vs two-hop neighbourhoods

With `num_layers = k`, a player's embedding depends on their `k`-hop neighbourhood
in the block's graph.

| Receptive field | What a player's embedding can see | Tabular analogue |
|---|---|---|
| **no messages** (`disable_message_passing=True`) | own node features only | per-player features |
| **1 hop** | own features + every opponent in the window, with the match edges | a *learned* version of the history statistics (a weighted summary of one's own opponents and results) |
| **2 hops** | + opponents' opponents — including **shared opponents** of A and B | none: the GBDT has no head-to-head or common-opponent feature |

This is why depth is the research question and not a tuning knob:

- **1 hop vs none** asks whether a learned aggregation over one's own record adds
  anything to the hand-built one.
- **2 hops vs 1** asks whether *relational* structure — a path A → C → B — adds
  anything. Only a model with at least two layers can route information through a
  shared opponent.

The honest "no graph" control keeps every layer, parameter and LayerNorm and
removes only the edges. `num_layers=0` also deletes the LayerNorms, so it
confounds message passing with normalisation (chapter 5 §5.2).

**Structural descriptors** (`tennis_gnn/structure.py`), computed per predicted
match from the block's edges, symmetric in A and B and independent of the seed:

| Descriptor | Meaning |
|---|---|
| `head_to_head` | prior meetings inside the window |
| `common_opponents` | distinct shared neighbours, excluding A and B |
| `degree_a`, `degree_b`, `degree_min` | distinct prior opponents; `degree_min` is the thinner-recorded player |
| `connected` | any prior meeting or shared opponent |

and the pre-specified strata built from them (`add_strata`), with cut points
chosen from the descriptor distributions, not from model performance:

| Stratum | Cuts |
|---|---|
| `degree_stratum` | 0–5 (cold), 6–20, 21+ |
| `common_stratum` | 0, 1–4, 5–14, 15+ |
| `h2h_stratum` | no prior meeting, 1, 2+ |
| `two_hop_only` | no prior meeting **and** ≥1 shared opponent — where a two-hop model has information a one-hop model lacks |

These are what chapter 6 §6.5 reports. The descriptors depend on the window: a
shared opponent from four years ago is not a neighbour at 365 days.

The literature-style intransitivity score (`preprocess/intransitivity_calc.ipynb`)
is a separate, earlier measure: a local Hodge decomposition over common-opponent
evidence, bucketed into `low` / `medium` / `high` / `no_context_evidence` and
carried on each snapshot as `intransitivity_level`. It is a stratification label,
never a feature.

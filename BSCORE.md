# The B-Score: definition and computation

The B-score ("beat score") is a time-decayed, leakage-free measure of a
player's standing, computed as the weighted eigenvector centrality of a
directed win graph. It is the main hand-engineered node feature fed to the
GNN and to the GBDT baseline.

Reference implementation: `new_work/build_bscore.py`
(the 1990-history variant, `extended_history_1990/build_bscore.py`, is the
same method with the start dates and tournament-level filter parameterised).

## 1. The idea

Elo treats every win as worth the same number of points against a given
opponent rating. The B-score instead asks a recursive question: *a player is
strong if they have beaten players who are themselves strong.* That is the
definition of eigenvector centrality, so the B-score is exactly that
centrality on a graph whose edges point from loser to winner.

## 2. The graph

For a given point in time `t`:

- **Nodes**: every player appearing in the match history before `t`.
- **Edges**: one directed edge `loser -> winner` per matchup pair, so the
  edge points in the direction that "credit" flows.
- **Edge weight**: each individual match contributes

  ```
  w = 1 / (1 + age_days / 365)          age_days = t - tourney_date
  ```

  and the weights of all matches between the same ordered pair are summed.
  A match played today contributes 1.0, one a year old 0.5, one three years
  old 0.25. The decay is hyperbolic, not exponential — old results fade but
  never vanish entirely.

Code: `build_edge_frame()` (weights, groupby) and `build_graph()`
(`nx.from_pandas_edgelist` into a `DiGraph`; isolated players are added
explicitly as nodes so nobody is silently dropped).

## 3. The score

```python
nx.eigenvector_centrality(graph, weight="weight", max_iter=1000, tol=1e-9)
```

This solves `x = (1/λ) A^T x` by power iteration, where `A` is the weighted
adjacency matrix. Because edges run loser → winner, a player's score
accumulates the scores of everyone they have beaten, each scaled by that
matchup's decayed weight. NetworkX returns the vector L2-normalised, so
B-scores are only meaningful **relative to other players in the same
snapshot** — never compare a raw B-score across snapshots or scopes. The
model consumes differences within a snapshot, which is well defined.

## 4. Rolling snapshots (why there is no leakage)

The score is not computed once. It is rolled forward one
**tournament-round block** at a time (`build_snapshots()`):

1. Seed the history with all matches from `HISTORY_START` (2006-01-01) up to
   `ROLLING_START` (2011-01-01). This period is burn-in only — no snapshots
   are emitted for it.
2. Enumerate every `(tourney_id, round_order)` block from the rolling start
   onwards, in chronological order (`R128=1 … F=7`).
3. For each block: build the graph from the history **as it stands before
   that block**, decayed to the block's own `tourney_date`, and record every
   player's score as that block's snapshot.
4. *Then* append the block's matches to the history and move on.

Because step 3 strictly precedes step 4, the snapshot used to predict a match
never contains that match — or any match from the same round of the same
tournament. That ordering is the whole leakage guarantee.

Output is one long-format row per `(tourney_id, round_order, player, score)`,
written to `data/processed/bscore_snapshots*.csv`.

## 5. Surface variants

Four snapshot files are produced by running the identical procedure on
filtered match sets (`SURFACES` in the script):

| Column | Matches used | File |
|---|---|---|
| `bscore_general` | all | `bscore_snapshots.csv` |
| `bscore_hard` | `surface == "Hard"` | `bscore_snapshots_hard.csv` |
| `bscore_clay` | `surface == "Clay"` | `bscore_snapshots_clay.csv` |
| `bscore_grass` | `surface == "Grass"` | `bscore_snapshots_grass.csv` |

Surface scores are normalised within their own subgraph, so a clay B-score is
comparable only to other clay B-scores from the same snapshot.

## 6. The default value

A player can be absent from a snapshot (first appearance, or no matches yet on
that surface). The fallback is the **25th percentile of the burn-in-period
score distribution**, computed once from the graph at 2010-12-31. A newcomer
is therefore treated as a below-median but not zero-strength player, which
avoids the discontinuity a literal 0 would introduce.

At load time (`tennis_gnn/data.py`) a missing surface score falls back to
`bscore_general` before the global default is used.

## 7. Downstream use

- `tennis_gnn/data.py` loads the four columns per player-snapshot and exposes
  `surface_bscore()`, which picks the column matching the match's surface.
- `tennis_gnn/snapshots.py` attaches the four values as node features and
  additionally emits a per-match `bscore_diff` (winner minus loser, sign
  flipped for the negated half of the symmetric dataset) as a decoder/GBDT
  feature.

## 8. Reproducibility

`python new_work/build_bscore.py --verify` rebuilds the snapshots on the
original tournament whitelist and asserts they match the published CSVs to
`1e-9`. This exists so that any change in downstream results after widening
the scope (`--scope full`, or the 1990 history build) is attributable to the
data and not to a silent redefinition of the metric.

## 9. Cost

Each block requires a full eigenvector-centrality solve over a graph that
grows monotonically. On the full scope this is tens of thousands of solves and
the build takes hours; it is run offline and the CSVs are cached (and
gitignored).

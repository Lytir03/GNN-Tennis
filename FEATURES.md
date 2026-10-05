# Feature construction

How every number either model sees is built, and why the two families see
the same information. Companion to `BSCORE.md`, which covers the B-score
itself.

Sources:
- `tennis_gnn/history.py` — recency-weighted history stats, Elo
- `tennis_gnn/edge_features.py` — score parsing, edge features
- `tennis_gnn/snapshots.py` — GNN node features, context, targets
- `gbdt_comparison/features.py` — the tabular baseline

## 0. The governing constraint

Everything is built **per tournament-round block**, in chronological order,
under one invariant:

> **trim → read → append.**
> A block's features are read from state that contains no match from that
> block. The block's matches are folded into state only afterwards.

This holds for the B-score snapshots, the history deques, and Elo alike. It is
stated in the code as a comment at each of the three update sites because
reversing the order produces a model that looks excellent and is worthless.

Both feature builders iterate the *same* block sequence and draw match
orientations from `np.random.default_rng(seed)` with exactly one
`rng.random()` per emitted match. That is what makes GNN and GBDT
predictions comparable match-by-match rather than merely on aggregate. Blocks
that emit no targets are still iterated, because the draw is positional — skip
one and every later label shifts.

## 1. The atomic unit: a parsed score

`parse_match_score(score, best_of, incomplete_margin_policy)` turns a
Sackmann score string (`"6-4 3-6 7-5"`, `"6-2 1-0 RET"`) into:

| Field | Meaning |
|---|---|
| `relative_game_diff` | `(winner_games − loser_games) / total_games` |
| `set_margin_scaled` | `(winner_sets − loser_sets) / best_of` |
| `straight_sets_flag` | won without dropping a completed set |
| `completed_match_flag` | not RET / W-O / DEF / ABD |
| `retirement_flag`, `walkover_flag` | the specific termination |
| `parsed_set_count` | sets that parsed as `N-M` |

Details worth knowing:
- Set completion accepts normal (`6-4`), advantage (`8-6`), tiebreak (`7-6`)
  and match-tiebreak (`10-8`) forms.
- `incomplete_margin_policy="zero"` (used for history stats) zeroes the
  margins of unfinished matches; `"played"` keeps what was actually played
  before the stoppage. Edge presets choose per ablation.
- The parse is `lru_cache`d on normalised text. Snapshot building re-reads
  three years of history per block, so the same few thousand score strings get
  parsed millions of times per run; the cache is a pure speed measure.

Scores are written **from the winner's perspective**, so a margin is positive
on the loser→winner edge and must be sign-flipped for the reverse.

## 2. Recency-weighted history (shared by both models)

`aggregate_history()` summarises one player's recent matches. Each match
carries a `HistoricalPerformance` record, signed `+1` for the winner and `−1`
for the loser (margins flipped likewise), so a single record type serves both
sides.

Weights use the same hyperbolic decay as the B-score:

```
w = 1 / (1 + age_days / alpha_days)      alpha_days = 365
```

Six statistics, all weight-normalised means except the first:

| Statistic | Definition |
|---|---|
| `history_mass` | `log1p(Σw)` — how much evidence exists at all |
| `result_balance` | weighted mean of ±1 results |
| `game_margin` | weighted mean signed `relative_game_diff` |
| `set_margin` | weighted mean signed `set_margin_scaled` |
| `straight_balance` | weighted mean signed straight-sets flag |
| `completed_rate` | weighted fraction of completed matches |

Two **views** are computed: `all` (every surface) and `surface` (restricted to
the current match's surface) → **12 numbers per player**. A player with no
qualifying history gets zeros, matching the neutral value an unseen player's
node features take.

`HistoryTracker` holds a per-player `deque` with a **3-year look-back**,
trimmed at each block. Its `feature_vector()` emits the twelve values ordered
`all` then `surface`; that ordering is load-bearing — cached snapshots are
unreadable if it moves.

## 3. Elo (GNN node features only)

`EloTracker`: standard Elo, `K=32`, initial `1500`, one general rating plus one
per surface (Hard/Clay/Grass). A surface rating moves only on that surface, so
a player with no clay matches keeps 1500 there — the honest answer, not a
missing value.

Ratings are stored **pre-multiplied by `ln(10)/400`** and offset by the
initial, so a stored difference *is* the Elo logit and a decoder coefficient of
1.0 exactly reproduces Elo's own prediction. Unlike the B-score this needs no
precomputed file — Elo is one cheap ordered pass, done inside the graph build.

## 4. GNN node features

`_node_features()` builds a row per player, in a deliberately append-only
layout:

| Index | Block | Contents |
|---|---|---|
| 0–3 | B-score | general, hard, clay, grass |
| 4–5 | static | height (median-imputed), `is_right` |
| 6–17 | history | the twelve statistics, `all` then `surface` |
| 18–21 | Elo | general, hard, clay, grass |

Indices 0–5 are `LEGACY_NODE_DIM` — what the model saw before history was
added. Everything is stored **unconditionally**; `ModelConfig.node_history_features`
and `ModelConfig.rating_source` decide what the network actually reads. One
cache therefore serves every ablation, and the parity claim is checkable: the
same twelve numbers the GBDT gets for two players are attached here to *every*
player in the graph. Non-finite values are `nan_to_num`'d to 0.

## 5. GNN edge features

`build_bidirectional_edges()` emits **two directed edges per historical
match** (loser→winner and winner→loser) over the matches inside the graph
window. Per edge:

- `recency_weight` — same `1/(1 + age/365)` decay
- `signed_relative_game_diff` — the game margin, sign-flipped on the reverse edge
- optional, by preset: `signed_set_margin`, `straight_sets_flag`,
  `completed_match_flag`, `retirement_flag`, `walkover_flag`,
  `best_of_5_flag`, `grand_slam_flag`
- always last: `surface_hard/clay/grass`, `round_scaled` (`round_order/7`),
  `result_direction`

`result_direction` encodes which way the edge runs: `+1 / −1` under the default
`signed` encoding, `0 / 1` under `legacy_01`. `ABLATION_PRESETS` ladders from
`current` (the original representation: unsigned margins, 0/1 direction) up to
`full`. `edge_feature_names(config)` is the single source of truth for the
resulting column order.

The graph **window** (`window_days`, default 3×365) decides both which matches
become edges and which players become nodes. It deliberately does *not* touch
the history statistics — `HistoryTracker` keeps its own 3-year window — so
sweeping the window is a clean one-factor test of graph contents.

## 6. Match context and targets

Per match, `_block_targets()` stores:

- `player_a`, `player_b` — node indices after a fair coin flip on orientation
- `y` — 1.0 if A is the true winner, else 0.0
- `context` — `surface_one_hot(3)`, `round_order/7`, `best_of_5_flag`,
  `grand_slam_flag`
- `bscore_diff` — general B-score, A minus B
- `intransitivity_level` — stratification label, not a feature

The last two context columns exist to close an information asymmetry: the GBDT
baseline always received best-of-5 and Grand-Slam flags for the match being
predicted and the GNN did not. Full context is always stored;
`ModelConfig.rich_match_context` gates what the model may read.

## 7. GBDT tabular features

`build_feature_dataset()` reproduces the same information as flat columns. Per
player it assembles:

- `bscore_general`, `bscore_surface` (the column matching the match surface)
- `height`, `is_right`
- the twelve history statistics, as `history_all_*` and `history_surface_*`

Then `add_pair_features()` expands **every** such statistic into four columns:

```
{prefix}_{name}_a, {prefix}_{name}_b, {prefix}_{name}_diff, {prefix}_{name}_abs_diff
```

Giving the tree both raw values and their difference/absolute difference saves
it from having to discover subtraction through axis-aligned splits. Match-level
columns (`round_scaled`, `best_of_5_flag`, `grand_slam_flag`, surface one-hot)
are added once, unpaired.

`NON_FEATURE_COLUMNS` fences off identifiers and the label;
`model_feature_columns()` returns everything numeric that survives. Infinities
and NaNs are replaced with 0.0 at the end.

Two synchronisation points are load-bearing and both are commented as
past bugs:
- `start_date` and `end_date` default to the **scope's** `rolling_start` /
  `rolling_end`. Hardcoding `"2011-01-01"` silently dropped training years on
  wider scopes and desynchronised the orientation draw from the GNN's — the
  two models then had different labels for the same match.
- The B-score snapshot **suffix follows the scope**. Without it a full-scope
  run reads Slam+Masters snapshots and scores most players from the percentile
  default: plausible numbers, no error.

`phase_for_year()` delegates to `tennis_gnn.data`, collapsing `warmup` into
`train` because GNN warm-up years never emit GBDT rows anyway.

## 8. Defaults and missing data, in one place

| Situation | Value |
|---|---|
| Player absent from a B-score snapshot | 25th percentile of that snapshot |
| Missing surface B-score | falls back to `bscore_general` |
| Missing height | median height across all players |
| Missing hand | `"R"` |
| No qualifying match history | zeros for all six statistics |
| No matches on a surface (Elo) | initial 1500 → feature value 0.0 |
| Any residual NaN/±inf | 0.0 |

## 9. Caching

Graphs are built once per `(scope, edge preset, window)` and cached
**without the seed in the key** — nothing in a graph depends on one.
`attach_targets()` then draws orientations per seed on top of the prebuilt
graphs, so five seeds cost one graph build instead of five. The cache key
carries a version tag: `v2` graphs include the twelve history statistics per
node, and a `v1` file would otherwise load without error at the wrong feature
width.

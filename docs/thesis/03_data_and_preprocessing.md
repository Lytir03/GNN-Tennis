# Chapter 3 — Data and Preprocessing

Where the matches come from, how they are filtered and split, and how every
number a model sees is built without looking at the future. Sections follow the
outline's bullets.

Primary sources: [`tennis_gnn/data.py`](../../tennis_gnn/data.py),
[`tennis_gnn/history.py`](../../tennis_gnn/history.py),
[`tennis_gnn/edge_features.py`](../../tennis_gnn/edge_features.py),
[`gbdt_comparison/features.py`](../../gbdt_comparison/features.py),
[`new_work/build_bscore.py`](../../new_work/build_bscore.py),
[`preprocess/`](../../preprocess/),
[`extended_history_1990/`](../../extended_history_1990/).
Detailed references: [`FEATURES.md`](../../FEATURES.md),
[`BSCORE.md`](../../BSCORE.md), [`data/processed/README.md`](../../data/processed/README.md).

Match counts below were computed directly from the processed CSVs with the
split rules in `tennis_gnn/data.py`.

---

## 3.1 ATP match dataset and tournament scope

**Source.** Jeff Sackmann's `tennis_atp` repository, one file per year
(`data/raw/sackmann/atp_matches_1968.csv` … `atp_matches_2024.csv`). One row per
match, written from the winner's perspective: `winner_name`, `loser_name`,
`score`, `best_of`, `round`, `surface`, `tourney_id`, `tourney_name`,
`tourney_level`, `tourney_date`, player height and hand.

**Scopes.** A scope is a match file plus a split, declared in
`tennis_gnn/data.py` `SCOPE_FILES`:

| Scope | Levels | File | Built by | Rows | Tournaments |
|---|---|---|---|---:|---:|
| `slams` | G | `atp_matches_slams.csv` | `preprocess/intransitivity_calc.ipynb` | — | — |
| `slams_masters` | G, M | `atp_matches_slams_masters.csv` | `preprocess/data_prep_slams_only.ipynb` | 20,057 | 238 |
| `full` | G, M, A | `atp_matches_full.csv` | no script in the repo writes it (see note) | 49,797 | 1,216 |
| `slams_masters_1990` | G, M | `atp_matches_slams_masters_1990.csv` | `extended_history_1990/build_matches.py` | 44,373 | 534 |
| `full_1990` | G, M, A | `atp_matches_full_1990.csv` | `extended_history_1990/build_matches.py` with `--levels G,M,A` | — | — |

`G` is Grand Slams, `M` Masters 1000, `A` other tour-level events (ATP 250/500).
Davis Cup (`D`) and the Tour Finals (`F`) are excluded because their draw formats
are not comparable tournament rounds (`new_work/build_bscore.py`).

The three scopes used for results are **`slams_masters`** (the original study),
**`full`** (the whole main tour: ≈3.5× the scored matches, and a harder problem
because ATP 250/500 draws bring weaker and less-recorded players), and
**`slams_masters_1990`** (the same tournaments with a decade of extra warm-up
history, used as an independent replication). `slams` appears only in the early
GBDT comparison (`gbdt_comparison/RESULTS.md`); `full_1990` has frozen
predictions but is not part of the reported results.

**A scope detail that affects the B-score.** The published `slams_masters`
B-score snapshots were built with `preprocess/data_prep.ipynb`'s *tournament-name
whitelist*, not the `G, M` level filter used for the match file. That whitelist
also contains several non-Masters events (e.g. Halle, Queen's Club, Barcelona,
Munich, Hamburg, Rio de Janeiro, 's-Hertogenbosch, Mallorca, Eastbourne), and
twelve of its names match nothing in the raw data because of misspellings
(`new_work/build_bscore.py` comments). So at `slams_masters` the B-score graph is
built on a slightly wider set of matches than the prediction graph. `--scope
current` reproduces this deliberately so the published results stay
reproducible; `--scope full` uses the level filter, and the 1990 builders
(`extended_history_1990/build_bscore.py`) use the same `G, M` level filter as
their match file on purpose, so the B-score graph and the prediction graph see
the same matches.

**Provenance gap.** No script or notebook in the repository writes
`atp_matches_full.csv`. The matching definition is the `G, M, A` level filter in
`new_work/build_bscore.py --scope full`, from 2006-01-01. If the thesis states
how the full-scope file was built, it should be regenerated from that filter and
checked against the existing file first.

---

## 3.2 Temporal train / validation / test split

Splits are **by calendar year**, fixed per scope, and never varied within a
scope. `phase_for_year()`:

```
year <  rolling_start  → warmup
year <= train_end      → train
year <= val_end        → val
otherwise              → test   (up to rolling_end, exclusive)
```

| Scope | Warm-up | Train | Validation | Test | Train n | Val n | Test n |
|---|---|---|---|---|---:|---:|---:|
| `slams_masters` | 2006–10 | 2011–15 | 2016 | 2017–20 | 5,367 | 1,075 | 3,770 |
| `full` | 2006–10 | 2011–16 | 2017–18 | 2019–24 | 15,809 | 5,325 | 14,340 |
| `slams_masters_1990` | 1980–89 | 1990–2011 | 2012–13 | 2014–20 | 23,338 | 2,142 | 6,995 |

- **Warm-up** years only build state (B-score graph, history deques, Elo) and are
  never emitted as targets.
- **Train** years produce gradient updates, **except the first rolling year**:
  `run_experiment` skips it as a soft buffer so state has a full year to
  accumulate (so gradients start in 2012 for the 2011-start scopes, 1991 for the
  1990 scope).
- **Validation** selects hyperparameters and fits the calibration temperature.
- **Test** is scored once.

Why the `full` split differs: the original one-year validation window has 1,075
matches, where the standard error of log loss is ≈0.018 — an order of magnitude
bigger than the differences being selected on (0.002–0.004). Two validation years
over the full tour give ≈5× more matches and a standard error of ≈0.008
(`tennis_gnn/data.py` `Split`, `new_work/STATUS.md`).

---

## 3.3 Cleaning and score processing

**Loading** (`load_dataset`):

- `round` is mapped to `round_order` with `R128=1, R64=2, R32=3, R16=4, QF=5,
  SF=6, F=7`. Rows with any other round (round robin, bronze medal, qualifying)
  or a missing player name are dropped — 613 rows in `full`, 84 in
  `slams_masters_1990`, none in `slams_masters`.
- Matches are sorted stably by `(tourney_date, tourney_id, round_order)`. That
  order *is* the timeline everything else iterates.
- **Blocks** are the unique `(tourney_id, round_order)` pairs in the rolling
  period: 795 at `slams_masters`, 4,688 at `full`, 2,545 at
  `slams_masters_1990`.
- Surfaces are Hard, Clay, Grass, plus a small number of Carpet matches (47 /
  326 / 1,244 rows), which one-hot encode as all zeros and get no surface Elo or
  surface B-score of their own.

**Score parsing** (`parse_match_score` in `tennis_gnn/edge_features.py`). A
Sackmann score string such as `"6-4 3-6 7-6(5)"` or `"6-2 1-0 RET"` becomes:

| Field | Meaning |
|---|---|
| `relative_game_diff` | (winner games − loser games) / total games |
| `set_margin_scaled` | (winner sets − loser sets) / best_of |
| `straight_sets_flag` | won without dropping a completed set |
| `completed_match_flag` | not RET / W/O / DEF / ABD |
| `retirement_flag`, `walkover_flag` | the specific termination |
| `parsed_set_count` | sets that parsed as `N-M` |

- A completed set is a normal (`6-4`), advantage (`8-6`), tiebreak (`7-6`) or
  match-tiebreak (`10-8`) score; tiebreak points in parentheses are ignored.
- Incomplete matches: `incomplete_margin_policy="zero"` zeroes their margins
  (used for the history statistics); `"played"` keeps what was played before the
  stoppage. Edge presets choose per ablation.
- Because scores are written from the winner's side, a margin is positive on the
  loser→winner edge and must be sign-flipped on the reverse edge.
- Parsing is `lru_cache`d: snapshot building re-reads three years of history per
  block, so the same few thousand strings are parsed millions of times.

**The score of the match being predicted is never a feature**, for either model.

---

## 3.4 Player and match features

**Static player attributes** (`_build_player_static`):

- `height` — the player's most recent recorded value; missing → median height
  over all players.
- `hand` → `is_right` (1.0 for `"R"`); missing hand → `"R"`.

These are taken from the latest record in the whole file, so in principle a
height recorded later is used earlier. Height and handedness do not change in any
way that matters here, and chapter 6 shows they carry no signal on their own (a
model with only these features scores exactly ln 2).

**Match context** (per predicted match, `_block_targets`):

| Column | Value |
|---|---|
| surface one-hot | Hard / Clay / Grass (Carpet → zeros) |
| `round_scaled` | `round_order / 7` |
| `best_of_5_flag` | `best_of == 5` |
| `grand_slam_flag` | `tourney_level == "G"` |

The last two were added to the GNN later (`ModelConfig.rich_match_context`) to
close an information gap: the GBDT always had them, and best-of-5 changes upset
probability.

**Orientation and label.** For each predicted match a fair coin decides which
player is "A": `y = 1` if A is the real winner. The draw uses
`np.random.default_rng(seed)`, **exactly one `rng.random()` per emitted match, in
block order**, in both the GNN and the GBDT feature builders. That is what makes
the two model families comparable match by match. Blocks that emit no targets
are still iterated, because the draw is positional: skip one and every later
label shifts. Different seeds are therefore different evaluation sets.

---

## 3.5 Historical features

`tennis_gnn/history.py`, shared by both model families.

Each past match is stored per player as a `HistoricalPerformance` record signed
`+1` for the winner and `−1` for the loser (margins flipped too). The recency
weight is

```
w = 1 / (1 + age_days / 365)
```

`aggregate_history()` computes six statistics:

| Statistic | Definition |
|---|---|
| `history_mass` | `log1p(Σw)` — how much evidence exists |
| `result_balance` | weighted mean of ±1 results |
| `game_margin` | weighted mean signed `relative_game_diff` |
| `set_margin` | weighted mean signed `set_margin_scaled` |
| `straight_balance` | weighted mean signed straight-sets flag |
| `completed_rate` | weighted fraction of completed matches |

in two views — **all surfaces** and **the match's surface** — so **12 numbers per
player**. `HistoryTracker` keeps a per-player deque with a **3-year look-back**,
trimmed at each block. A player with no qualifying history gets zeros.

The GBDT gets these twelve for the two players in a match. The GNN stores them
for **every** node in every graph, so the parity is checkable; whether the
network reads them, and where, is a model flag (chapter 5 §5.4).

---

## 3.6 B-score and Elo

Definitions are in chapter 2 §2.2. Here: how they are produced and attached.

**B-score** (`new_work/build_bscore.py`, originally `preprocess/graph*.ipynb`):

1. Seed history with matches from 2006-01-01 (1980-01-01 for the 1990 scopes) up
   to the rolling start.
2. For every block in chronological order, build the loser→winner graph from all
   history *before* the block, decayed to the block's date, solve eigenvector
   centrality, and record every player's score.
3. Then append the block's matches.

It is run four times — on all matches, and on Hard / Clay / Grass matches only —
giving `bscore_general`, `bscore_hard`, `bscore_clay`, `bscore_grass`, long-format
per `(tourney_id, round_order, player)`. Unlike the history statistics and the
graph, the B-score has **no look-back window**: all history since the start
date contributes, down-weighted by the hyperbolic decay.

`--verify` rebuilds the published snapshots on the original scope and requires
them to match exactly (1.54M rows, maximum absolute difference 1e-16) before a
wider scope may be built — so a change in results after widening is attributable
to the data, not to a redefinition.

**Missing values**, as implemented at load time:

| Situation | GNN (`tennis_gnn/data.py`) | GBDT (`gbdt_comparison/features.py`) |
|---|---|---|
| Missing surface B-score for a player in the snapshot | `bscore_general` | `bscore_general` |
| Player absent from the block's snapshot | **median** of that snapshot | **25th percentile** of that snapshot |

Every player already seen is in the snapshot, so "absent" means a true newcomer.
[`BSCORE.md`](../../BSCORE.md) §6 and [`FEATURES.md`](../../FEATURES.md) §8
describe the fallback as the 25th percentile for both models. In the code it is
the median for the GNN. Coverage is 99.1% of player slots in train, validation
and test at `full` (the entire gap is warm-up years), so the effect is small,
but it is a difference in information between the two model families.

**Elo** (`EloTracker`) is computed inside the graph build in one ordered pass,
general plus per-surface, `K=32`, initial 1500, stored on the logit scale. It is
a GNN node feature (`rating_source="elo"` or `"both"`) and a standalone baseline
(`gnn_improvements/baselines.py`), but not a GBDT feature.

---

## 3.7 Avoiding temporal leakage

One invariant, stated as a comment at each of the three state-update sites
(B-score snapshots, `HistoryTracker`, `EloTracker`):

> **trim → read → append.** A block's features are read from state that contains
> no match from that block. The block's matches are folded into state only
> afterwards.

What this rules out, concretely:

| Leak | How it is prevented |
|---|---|
| The predicted match's own result in its features | Features read before the block is appended |
| Results from the **same round** of the same tournament | The unit is the round block, not the match: a QF result cannot inform another QF |
| Later rounds informing earlier ones | Blocks iterate in `(date, tourney_id, round_order)` order |
| Future B-scores | One snapshot per block, computed from strictly earlier history |
| Tuning on test | Hyperparameters and temperature fitted on validation only; test scored once |
| Validation labels in the final fit | GNN updates weights on train blocks only; the GBDT is refitted on train rows only after selection |
| Two models silently scored on different labels | Shared positional orientation draw; `evaluation_hash` checked by `compare.py`; `verify_targets.py` rebuilds and re-hashes the evaluation set |

Two leak-shaped bugs were found and fixed, both commented in
`gbdt_comparison/features.py`:

- hardcoded `start_date="2011-01-01"` dropped training years on wider scopes and
  desynchronised the orientation draw — the GBDT's labels disagreed with the
  GNN's on 50.3% of matches;
- a B-score snapshot suffix that did not follow the scope made a `full`-scope run
  read Slam+Masters snapshots and score most players from the fallback —
  plausible numbers, no error.

The one known look-ahead is static player attributes (§3.4), which are taken
from the latest record in the file.

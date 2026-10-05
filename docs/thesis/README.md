# Thesis chapters, mapped onto the repository

One file per chapter of `Thesis Structure`. Each follows the outline's own
bullets as its sections, and every claim points at the file, config flag, CSV or
frozen artifact that produces it.

Nothing was retrained to write these. Numbers come from `FINAL_RESULTS.md`,
`gbdt_comparison/RESULTS.md`, `new_work/STATUS.md` and
`gnn_improvements/FINDINGS.md`, and where those disagree with the code or the
artifacts, the numbers were recomputed from `results/frozen_predictions/`, the
result CSVs, and the processed data. The differences are stated in the chapters
rather than silently fixed.

| Chapter | File | Covers |
|---|---|---|
| 1 | [01_introduction.md](01_introduction.md) | Motivation, why a graph, the research question, contributions |
| 2 | [02_background.md](02_background.md) | Prediction methods, Elo / B-score, GBDTs, GNNs and message passing, related work |
| 3 | [03_data_and_preprocessing.md](03_data_and_preprocessing.md) | Sackmann data and scopes, temporal splits, score parsing, player / match / history features, B-score and Elo, leakage |
| 4 | [04_graph_representation.md](04_graph_representation.md) | Nodes, directed edges, edge features and presets, per-block snapshots, the window, one vs two hops |
| 5 | [05_models_and_experimental_setup.md](05_models_and_experimental_setup.md) | GBDT baseline, GNN architecture, antisymmetric decoder, history routing and the B-score prior, training and tuning, metrics, multi-seed protocol |
| 6 | [06_experiments_and_results.md](06_experiments_and_results.md) | Baselines, graph freshness, hop depth, the feature × hop substitution grid, sparse vs well-connected strata, final GNN vs tuned GBDT |
| 7 | [07_discussion.md](07_discussion.md) | What the numbers mean, overlap between graph and history, why depth does not transfer, limitations |
| 8 | [08_conclusion_and_future_work.md](08_conclusion_and_future_work.md) | Summary, answer to the research question, limitations, next steps |

## Known discrepancies with the top-level documents

Found while checking these chapters against the code and artifacts. The
top-level documents have **not** been edited.

| Where | What it says | What the code / artifacts show | Chapter |
|---|---|---|---|
| `FINAL_RESULTS.md` §3–§6 | the headline and hop contrasts at `full` belong to the 365-day final model | they come from `depth_3_*_lr1e4` artifacts at the default 1095-day window; at 365 d the headline holds (−0.00568, 5/5) but one hop vs none is not significant (−0.00095, 4/5) | 6 §6.3, §6.6 |
| `FINAL_RESULTS.md` §5 | tier-3 `full` gain −0.0010 as a grid cell | the grid CSV has +0.00047 (lr 3e-4); −0.0010 is the stable-recipe re-run | 6 §6.4 |
| `FINAL_RESULTS.md` §7, `FINDINGS.md` §2 | window-sweep row "2hop − none"; scope not stated | the row is no-messages − two hops, at `slams_masters` | 6 §6.2 |
| `FINAL_RESULTS.md` §2 | `slams_masters_1990`: train 1990–2010, validation 2011–13 | `EXTENDED_1990_SPLIT` and manifests: train 1990–2011, validation 2012–13 | 3 §3.2 |
| `FINAL_RESULTS.md` §12 | `run.py --experiments base` runs "the final model" | it runs `BASE_MODEL` (2 hops, 1095 d, no history) | 5 §5.8, 6 §6.8 |
| `BSCORE.md` §6, `FEATURES.md` §8 | newcomer B-score = 25th percentile | GNN uses the snapshot median, GBDT the 25th percentile | 3 §3.6 |
| `config.py` `tournament_context_edges` | a one-factor ablation | not wired to the edge builder; identical to `base` | 4 §4.3 |
| `data/processed/` | — | no script writes `atp_matches_full.csv` | 3 §3.1 |

## Relation to the existing top-level documents

These chapters are organised by the thesis. The top-level documents are
organised by topic, and stay where they are because other files link to them:

| Document | Role |
|---|---|
| [`README.md`](../../README.md) | Repository entry point |
| [`FINAL_RESULTS.md`](../../FINAL_RESULTS.md) | The settled results, one page |
| [`FEATURES.md`](../../FEATURES.md) | Feature construction reference (chapters 3–4 in depth) |
| [`BSCORE.md`](../../BSCORE.md) | B-score reference (chapters 2–3 in depth) |
| [`gnn_improvements/FINDINGS.md`](../../gnn_improvements/FINDINGS.md), [`new_work/STATUS.md`](../../new_work/STATUS.md) | Working narrative, including retractions |

## Figures

Data figures are generated from files already on disk (processed matches,
frozen prediction artifacts, result CSVs); nothing is retrained. Each is saved
as a vector PDF for the thesis and a PNG preview.

```bash
~/miniconda3/envs/tennis-gnn/bin/python docs/thesis/make_figures.py
```

Numbering follows the figure plan; 1, 2, 3, 5 and 6 are diagrams to draw by hand.
Every "gain" axis is signed so that **positive means the graph / the GNN is better**.

| File | Place in thesis | Shows | Data |
|---|---|---|---|
| — Fig 1 (hand-drawn) | §4.2, replaces the placeholder | Rolling snapshot procedure | — |
| — Fig 2 (hand-drawn) | §4.1 | B-score graph vs GNN graph | — |
| — Fig 3 (hand-drawn) | §5.2–5.3 | Model architecture and antisymmetric decoder | — |
| `fig04_split_timeline` | §3.3 | Warm-up / train / validation / test per scope, match counts, first year without gradient updates | `tennis_gnn/data.py` splits, processed CSVs |
| — Fig 5 (hand-drawn) | §4 (one vs two hops) | Receptive field and shared opponent | — |
| — Fig 6 (hand-drawn, optional) | §5.4 | Online training vs frozen evaluation | — |
| `fig07_matches_per_year` | §3.1 | Main-draw matches per year by tournament level, full tour | `data/processed/atp_matches_full.csv` |
| `fig08_recency_weight` | §3.4.2 (optional) | Recency weight curve with the 365- and 1095-day marks | formula |
| `fig09_substitution_curve` | ch. 6, graph vs engineered history | Gain of one hop over no messages per feature tier and scope | frozen artifacts (`grid_*`, `gnn_*_one_hop`, `hopgrid_*`; full tier 3 = `depth_3_*_lr1e4`) |
| `fig10_window_sweep` | ch. 6, graph freshness | (a) Slams + Masters, two hops vs none by window; (b) 1990 extension, one hop | `gnn_improvements/results/window_hop_all.csv`, `hop_grid_slams_masters_1990.csv` |
| `fig11_strata_full` | ch. 6, sparse vs well-connected | One-hop gain by prior-opponent and common-opponent strata, tiers 1 and 3 | `new_work/results/strata_full.csv` |
| `fig12_hop_contrasts_full` | ch. 6, no graph vs one vs two hops | Paired contrasts at 1095 and 365 days with per-seed points | frozen `depth_3_*_lr1e4`, `win365_t3_*` |
| `fig13_gnn_vs_gbdt_full` | ch. 6, final GNN vs tuned GBDT | Paired improvement in log loss, Brier, accuracy at both windows | frozen artifacts vs `gbdt_tuned` |
| `fig14_baselines` | ch. 6, baseline comparison | Test log loss of Elo, B-score, GBDT, GNN per scope | frozen artifacts |
| `fig15_reliability_full` | ch. 6 final comparison, or §5.4 calibration | Reliability diagram, GNN vs GBDT, full tour | frozen artifacts |
| `fig16_recipe_sensitivity_full` | ch. 6/7, recipe sensitivity | Test log loss per arm at lr 3e-4 vs 1e-4 | frozen `grid_3_*`, `depth_3_*_lr1e4` |

### Slams + Masters figures (principal scope)

Final Slams + Masters model: tier 3, two hops, 365-day window, replay buffer 800
(`win365_t3_history_decoder_2hop_sum_buf800`). Depth and window arms exist only
with buffer 200, so hop contrasts are made inside that family
(`win{window}_t3_history_decoder_{k}hop`, lr 1e-4).

| File | Place in revised ch. 6 | Shows | Data |
|---|---|---|---|
| `fig17_sm_depth_by_window` | Fig. 6.2 (replaces `fig10`) | Gain of 1, 2 and 3 hops over no messages at every tested window | frozen `win*_t3_history_decoder_*` |
| `fig18_sm_hop_contrasts` | Fig. 6.3 (full-tour `fig12` as secondary) | 1 vs none, 2 vs none, 2 vs 1, 3 vs 2 at 365 d; 3 vs 2 at 547 d | frozen `win365_*`, `win547_*` |
| `fig19_sm_substitution_curve` | §6.4, next to `fig09` | One- and two-hop gain per feature tier (1095 d), with 365-day points for tiers 1 and 3 | frozen `hopgrid_*`, `win365_t1_*`, `win365_t3_*` |
| `fig20_sm_strata` (+ `.csv`) | Fig. 6.5, paired with full-tour `fig11` | Two- and one-hop gain by prior-opponent and common-opponent strata, 365-day graphs | frozen `win365_t3_*` + `.cache/snapshots/slams_masters__full__w365__seed*__v4.pt` via `tennis_gnn/structure.py` |
| `fig21_sm_gnn_vs_gbdt` | Fig. 6.6 (full-tour `fig13` as secondary) | Final GNN, 2-hop buffer-200 model and no-message model vs tuned GBDT | frozen artifacts vs `gbdt_tuned` |
| `fig22_sm_ablations` | §6.1 / §6.7 | One-factor variants of the final model: Elo prior, B-score removal, buffer, depth, mean aggregation, 1095-day window | frozen `win365_*`, `win1095_*` |
| `fig23_sm_reliability` | §6.6 or §5.4 | Reliability diagram, final GNN vs GBDT | frozen artifacts |

`fig20` needs the cached 365-day snapshots; the other figures read only CSVs.

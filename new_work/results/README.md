# What each result file is, and which claim rests on it

Every claim in `new_work/STATUS.md` should be traceable to a file here and, from
there, to the frozen prediction artifacts it was computed from. This is that
map. If a claim is not in this table, treat it as commentary rather than a
result.

## The files

| file | produced by | what it holds |
|---|---|---|
| `feature_hop_grid_slams_masters.csv` | `feature_hop_grid.py --scope slams_masters` | the substitution curve, 4 tiers × 3 hop counts, 5 seeds |
| `feature_hop_grid_full.csv` | `feature_hop_grid.py --scope full` | the same grid on the full tour; `n_seeds` differs by tier |
| `headline_full.csv` | `cold_start.py --scope full` | GNN vs GBDT, paired over 5 seeds |
| `strata_full.csv` | `cold_start.py --scope full` | per-stratum paired deltas, every tier × intervention |
| `interactions_full.csv` | `cold_start.py --scope full` | two-level stratum contrasts (the interaction tests) |
| `twohop_diagnostic.csv` | `twohop_diagnostic.py` | learning rate 3e-4 vs 1e-4 at two hops, 2 seeds |

`strata_*.csv` and `interactions_*.csv` carry an `intervention` column
(`1_vs_none`, `2_vs_1`, `2_vs_1_stable`). **Rows from different interventions
are different quantities and must never be compared.** Conflating two of them
produced a wrong claim in this project once already.

## The claims, and what supports each

| claim | file | how to read it |
|---|---|---|
| The GNN beats the tuned GBDT at full scope on log loss, Brier and accuracy | `headline_full.csv` | `excludes_zero` true on all three; 5/5 seeds |
| …but the winning configuration uses no message passing | grid `none_hop` column, tier 3 | 0.62022 with messages disabled against 0.62070 at one hop |
| Graph value decays as features improve (**the substitution curve**) | both `feature_hop_grid_*.csv` | `gain_1_vs_none` down the tiers, in both files |
| At tier 0 the graph supplies the entire signal | grid, tier 0 row | `none_hop` = 0.693147 exactly = the coin flip |
| Cold start does *not* favour message passing | `strata_full.csv`, `1_vs_none`, tier 3, `degree_stratum` | `0-5 (cold)` positive, 0/5 seeds |
| The intransitivity effect does not replicate | `interactions_full.csv`, `2_vs_1`, `two_hop_only` | compare against the same rows in the Slam+Masters run |
| The tier-3 2-hop penalty is mostly a recipe artifact | `twohop_diagnostic.csv` | lr 1e-4 recovers 0.0146 of 0.0237 |

## Reproducing

Artifacts live in `results/frozen_predictions/<scope>/seed_<n>/`, each a CSV of
per-match probabilities plus a manifest recording the model config, the training
recipe, the fitted temperature and an `evaluation_hash` over the keys and
labels. Every comparison checks that hash before pairing, so two models are
never compared unless they were scored on identical matches with identical
labels.

Analyses read the artifacts and never retrain, so all of these are seconds to
regenerate:

```
python new_work/feature_hop_grid.py --scope full --summary-only
python new_work/cold_start.py --scope full
python new_work/audit_artifacts.py
```

## Two health checks worth running after any new run

```
python new_work/recalibrate.py            # reports; --apply to repair
python new_work/audit_artifacts.py        # degenerate output, clamped T, hash drift
```

The first exists because a broken temperature fit damaged 18 artifacts before
anyone noticed; the second exists so the next such fault is found by running a
script rather than by chasing an odd number by hand.

## What is *not* here

- **Tier 2 at full scope** — queued; the crossover cell where the graph's
  marginal value changes sign.
- **A clean tier-3 depth test** — `depth_test.py` retrains both arms at lr 1e-4
  so the 2-vs-1 contrast is one-factor. Until it lands, the tier-3 `2_vs_1` row
  is inconclusive, not negative.
- **The temporal model** — still not re-run through `tennis_gnn/`, so it emits no
  comparable `evaluation_hash` and is excluded from every table above.

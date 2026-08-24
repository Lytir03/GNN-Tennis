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

The **`depth_*_lr1e4` artifacts** are the ones to trust at tier 3. The grid's
tier-3 row used lr 3e-4, carried over from the smaller scope, which handicaps
the message-passing arms and not the no-message arm — see the headline
correction in `STATUS.md`. Tiers 0, 1 and 2 were already at lr 1e-4.

`strata_*.csv` and `interactions_*.csv` carry an `intervention` column
(`1_vs_none`, `2_vs_1`, `1_vs_none_stable`, `2_vs_1_stable`). **Rows from different interventions
are different quantities and must never be compared.** Conflating two of them
produced a wrong claim in this project once already.

## The claims, and what supports each

| claim | file | how to read it |
|---|---|---|
| The GNN beats the tuned GBDT at full scope on log loss, Brier and accuracy | `headline_full.csv` | `excludes_zero` true on all three; 5/5 seeds |
| The best model uses **one hop** of message passing | `depth_3_history_decoder_*_lr1e4` artifacts | 0.61915 at one hop against 0.62015 with messages disabled, 5/5 seeds |
| Two hops hurt decisively | same artifacts | 0.62857 at two hops, 0/5 seeds |
| Graph value decays as features improve (**the substitution curve**) | both `feature_hop_grid_*.csv` | `gain_1_vs_none` down the tiers, in both files |
| At tier 0 the graph supplies the entire signal | grid, tier 0 row | `none_hop` = 0.693147 exactly = the coin flip |
| Cold start does *not* favour message passing | `strata_full.csv`, `1_vs_none`, tier 3, `degree_stratum` | `0-5 (cold)` positive, 0/5 seeds |
| The intransitivity effect does not replicate | `interactions_full.csv`, `2_vs_1_stable`, `two_hop_only` | −0.0001, CI [−0.0028, +0.0026]: a tight null, not a wide shrug |
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

## One health check worth running after any new run

```
python new_work/audit_artifacts.py        # degenerate output, clamped T, hash drift
```

This exists so a fault like the temperature-calibration bug (18 artifacts damaged
before anyone noticed, see `TUTOR_REPORT.md` §12.2) is found by running a script
rather than by chasing an odd number by hand. The one-time recalibration fix that
repaired those 18 artifacts has already been applied to every frozen artifact
here; the script that did it (`recalibrate.py`) has served its purpose and was
removed rather than kept as a migration script with nothing left to migrate.

## What is *not* here

- **Tiers 0, 1 and 2 at five seeds** — they have three, so their intervals use
  t = 4.303. Tier 2's strata in particular are all non-significant with wide
  intervals and should be read as contributing its grid row and nothing more.
- **A stable-recipe depth test at tiers 0, 1 and 2** — not needed, they already
  ran at lr 1e-4, but it means the `2_vs_1` rows for those tiers are directly
  comparable while tier 3's are not; use `2_vs_1_stable` for tier 3.

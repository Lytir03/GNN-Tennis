# gnn_improvements

Can the GNN beat the B-score rating it is built on, and can it be made
better? Screening runs on `slams_masters` (~90s a seed, so a 5-seed arm
costs about 8 minutes); confirmed wins are checked on `full`
(Slams+Masters+ATP 250/500) before they count.

## Why this exists

The scope-matched literature comparison was uncomfortable. On Slams+Masters
this repo's GNN scores Brier 0.1946 / log loss 0.5703, and
[Spagnolo et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC8900648/) report
Brier 0.194 / log loss 0.573 for the B-score alone — weighted Bonacich
centrality on the win/loss graph, with no machine learning anywhere in it.
That is the same rating this model already receives as an input, which raised
the possibility that the entire architecture is worth nothing over its own
prior.

That comparison crosses papers, data sources and test years, so it could only
raise the question. `baselines.py` answers it on our own matches.

Three mechanical defects were found while investigating, all untested:

1. **The direct-logit skip connection is arithmetically starved.**
   `model.py` computes `bscore_scale * (b_a - b_b)` with `bscore_scale`
   initialised to 1.0. The measured median `|b_a - b_b|` is 3.2e-03, so
   `bscore_scale` must reach roughly **15** before the term moves a logit by
   1. Adam moves a scalar by at most about `lr` per step, so in this scope's
   1,344 steps at lr 1e-4 it can travel **0.13**. The head's final layer is
   zero-initialised specifically so the model *starts* as pure scaled
   B-score — that intent is defeated by the scale mismatch.
2. **Node features reach the encoder unnormalised.** The LayerNorms sit after
   each convolution, not on the input. Node features mix height (~185) with
   B-scores (~5e-4).
3. **`normalize_node_features` is train/eval inconsistent.** It standardises
   over `dim=0` of whatever it is handed: a *batch* while training, a single
   graph while predicting. So it uses cross-block statistics for one and
   per-block statistics for the other. The project recorded node
   normalisation as "badly harmful", but that verdict came from the old
   cumulative-ablation chain and this bug, so it may never have had a fair
   test. `per_block_node_norm` is the corrected version; the old flag is left
   untouched so published ablations still reproduce.

## Scripts

| Script | What it does |
|---|---|
| `baselines.py` | Writes `bscore_only`, `bscore_surface_only`, their `_log` variants, and `elo_baseline` as frozen artifacts. Reuses `load_or_build` snapshots so keys and labels are identical to the GNN's **by construction** — the GBDT baseline re-derived its own orientation draw, drifted out of sync, and produced labels disagreeing with the GNN's on 50.3% of matches. |
| `bscore_conditioning.py` | The five arms below, one factor each off `bscore_general_control`. |
| `blend.py` | Weighted combination of any set of frozen artifacts, weights fitted on validation only, written back as an ordinary artifact so `compare.py` sees it. Nothing in the repo blended across experiments before (`run_ensemble` averages seeds of one config). |
| `retune.py` | Validation-only recipe search for whichever config wins. |

## Arms

| Arm | Change | Question |
|---|---|---|
| `scale15` | `bscore_scale_init=15` | Is the defect purely that the optimiser cannot reach a useful scale? |
| `log` | `bscore_transform="log"` | Does the heavy tail matter independently? |
| `logz` | per-block standardised log | Both at once, and invariant to the ~10x centrality drift between scopes |
| `logz_surface` | `logz` + surface-matched B-score | Compose with the surface fix that already won 5/5 seeds |
| `nodenorm` | `per_block_node_norm=True` | Does correct input normalisation help where the buggy version "hurt"? |

`scale15` against `log`/`logz` is the informative contrast: if raising the
init alone recovers the gain, the defect was scale and not distribution
shape.

## Reading the results honestly

- Several arms are typically screened at once. Report per-seed unanimity
  next to means, and check that validation agrees with the test-set
  ranking before treating anything as real — this project has already
  retracted results that rested on noise.
- Confirm on `full` before quoting a result as settled, not just
  `slams_masters` — see `FINDINGS.md` §3, where the best `slams_masters`
  hop count did not transfer directly.
- Calibrate expectations. Bookmaker Brier is ~0.198 *on a harder set that
  includes ATP 500s*, and the published plateau across dozens of methods is
  65–70% accuracy. A realistic gain here is 0.005–0.02 Brier. The success
  criterion is beating `bscore_only`, not beating the market.

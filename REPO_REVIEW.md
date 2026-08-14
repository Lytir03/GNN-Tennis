# GNN-Tennis — repo walkthrough (2026-08-14)

Read-only review of everything currently in the repo (committed and uncommitted). No code was changed.

> **Correction (same day).** Section 5 of this document was **wrong** and has been
> rewritten below. It claimed that the `mean_aggregation`, `node_normalization` and
> `residual_connections` results "validated" the current design. They did not: the
> original ablations were **cumulative**, so each row inherited every change above
> it and none of them measured the factor named in its own title. See
> `TUTOR_REPORT.md` for what was done about it. The rest of this document stands.

## 1. Snapshot

The repo has only 4 commits (`Initial commit` → `bscore and initial GNN` → `finished GNN+elo baseline+bscore baseline` → `small change`), but a lot of newer work sits uncommitted:

- **Untracked:** `Intransitivity_NNs/`, `NN_test/`, `edge_feature_ablation/`, `gbdt_comparison/`, `temporal_gnn_intransitivity/`, `preprocess/`, `results/`, several `data/processed/*.csv`, `tests.md`, 3 new notebooks in `NNs/`.
- **Deleted from root, moved into `preprocess/`:** `data_prep.ipynb`, `graph.ipynb`, `graph_clay1.ipynb`, `graph_grass1.ipynb`, `graph_hard1.ipynb` — content differs (edited), not identical copies. Nothing was lost, it's a move + edit.
- **No `.gitignore` anywhere.** `.DS_Store` and `NNs/.DS_Store` are tracked and keep showing up as "modified" for no real reason.

None of this is broken, but a good chunk of your actual experimental work (the ablation framework, the GBDT baseline, the temporal model) currently only exists in the working tree.

## 2. Architecture verdict: GINE vs GAT — **keep GINE**

You asked whether GINE is really the best layer. You already ran this experiment: `NN_test/GNN_GAT_else_same.ipynb` swaps `GINEConv` for `GATv2Conv` (4 heads, otherwise identical pipeline — same node/edge features, same training loop). Result, same test set:

| Layer | Accuracy | Log-loss | Brier |
|---|---:|---:|---:|
| GINE (`current_gnn`) | 0.6676 | 0.6119 | 0.2115 |
| GAT (`GNN_GAT_else_same`) | 0.6568 | 0.6149 | 0.2132 |

GINE wins on all three metrics. It also makes structural sense here: your graphs are edge-feature-heavy (10-dim match-outcome vectors carry most of the signal) and small (per tournament-round snapshots), which is exactly GINE's strength — it consumes edge features directly in the message function, where GAT only uses them to bias attention weights. **Verdict: don't switch to GAT**, the data you already generated backs up the current choice.

## 3. Temporal GNN verdict — **behind, but the comparison isn't fair yet**

`temporal_gnn_intransitivity/model.py` (`TemporalTennisGNN`) is a genuinely different architecture from the static `TennisGINE`, not just "GINE with a time axis": one `GRUCell`-updated hidden state per player, updated by a **single** incoming message per event (no multi-hop propagation the way the 2-layer static GINE has). Its own README states plainly: *"nessun tuning: 8 epoche, hidden size 32, AdamW"* — zero hyperparameter search.

Results (Slam+Masters, seed 42):

| Model | Accuracy | Log-loss |
|---|---:|---:|
| Temporal GNN | 0.6188 | 0.6589 |
| Temporal GNN + intransitivity features | 0.6382 | 0.6537 |
| Static GNN (GINE, no B-score) | 0.6714 | 0.6049 |
| GBDT (no B-score) | 0.6687 | 0.6054 |

That's a real gap (5–9 points of accuracy), and your instinct that something's off is reasonable. But it's confounded two ways: (1) it has strictly less representational capacity than the static model (single-hop vs. two-hop), and (2) it never went through the tuning pass the other two models got (GBDT: 32 tuned configs; static GNN: an 18-way architecture/edge ablation study). So this isn't "temporal modeling doesn't work for this problem," it's "this particular temporal model is small and untuned." **Recommendation: either give it a real hyperparameter sweep (hidden size, epochs, at minimum) and a second message-passing hop before judging it, or explicitly label it an exploratory dead-end and stop spending time on it** — right now it's in an ambiguous middle state.

## 4. Unused win: the antisymmetric decoder

Buried in `edge_feature_ablation/`'s 18-way comparison is your best-performing GNN variant, and it's not in your main pipeline. `antisymmetric_decoder` forces `P(A beats B) = 1 − P(B beats A)` by construction (`0.5 * (score(a,b) − score(b,a))`) — the same trick already used in `temporal_gnn_intransitivity/model.py`'s `predict()`. Applied to the static GNN:

| Variant | Accuracy | Log-loss |
|---|---:|---:|
| `current_gnn` (main pipeline, naive decoder) | 0.6676 | 0.6119 |
| `antisymmetric_decoder` | **0.6714** | **0.6044** |

This is the single largest, cleanest win in the whole ablation table, and per `gbdt_comparison/RESULTS.md` it holds up across 5 seeds with confidence intervals that exclude zero — it's not noise. But `NNs/GNN_experiment.ipynb` (your main notebook) still ends with the naive decoder:
```python
pair_z = torch.cat([h_a, h_b, h_a - h_b, torch.abs(h_a - h_b)], dim=1)
```
**Recommendation: merge the antisymmetric decoder into the main notebook.** This is low-risk (it's just enforcing a symmetry the model should have anyway) and already validated.

## 5. ~~Design choices already validated~~ — the ablations are confounded

**This section originally said the opposite of what follows. I got it wrong on the
first read and caught it later; the corrected version is below.**

Three other ablations in the same table make things look clearly *worse*:

| Ablation | Accuracy | Log-loss | vs. baseline |
|---|---:|---:|---:|
| `mean_aggregation` (sum → mean) | 0.6584 | 0.6414 | worse |
| `node_normalization` (extra norm) | 0.6366 | 0.6786 | much worse |
| `residual_connections` | 0.6485 | 0.6627 | worse |

I first read this as confirmation that the current design is near a local optimum.
It isn't, because **these ablations are cumulative, not one-factor-at-a-time**: each
row in the original study inherits every change in the rows above it. `node_normalization`
is genuinely and strongly harmful (0.679 vs 0.604), and because it sits early in the
chain, `mean_aggregation`, `residual_connections` and `tournament_context` all carry
that damage with them. Their numbers measure *inherited normalisation damage plus
their own effect*, confounded together — not the factor named in the title.

So the correct statement is: **we do not currently know whether mean aggregation or
residual connections help or hurt.** They were never cleanly tested. The only thing
this table establishes is that node-feature normalisation is bad.

This is fixed on the `gnn-consolidation` branch: `tennis_gnn/config.py`'s
`one_factor_ablations()` builds every variant by changing exactly one field of
`BASE_MODEL`, and `tennis_gnn/test_tennis_gnn.py::test_each_ablation_changes_exactly_one_field`
fails the test suite if anyone reintroduces a cumulative chain.

## 6. GBDT beats every GNN variant

`gbdt_comparison/` (tuned `HistGradientBoostingClassifier`, same features available to the GNN) outperforms **every** GNN variant, including the best `antisymmetric_decoder`, on accuracy/log-loss/Brier, on both Slam+Masters and Slam-only splits, and wins significantly in 4/5 seeds (`gbdt_comparison/RESULTS.md`):

| Model (Slam+Masters, 5-seed avg) | Accuracy | Log-loss |
|---|---:|---:|
| GBDT tuned | **0.6695** | **0.6015** |
| Antisymmetric decoder GNN | 0.6671 | 0.6092 |
| Current GNN | 0.6672 | 0.6124 |
| B-score logit | 0.6442 | 0.6260 |

This is worth stating plainly: right now, the GNN's added complexity (graphs, message passing, PyG) is not yet earning its keep against a much simpler tuned tabular model on the same underlying information. That doesn't mean drop the GNN — the intransitivity-subgroup analysis (`temporal_gnn_intransitivity/README.md`) suggests the GNN's relative edge grows on more intransitive matches, which is a real and interesting thread — but for the headline metric, GBDT is currently the stronger model. Consider either (a) an ensemble of GBDT + antisymmetric-decoder GNN, or (b) treating GBDT as the production baseline until a GNN variant clears it outright.

## 7. Housekeeping — trim / fix

- **Add a `.gitignore`**: at minimum `.DS_Store`, `__pycache__/`, `*.pyc`, `.ipynb_checkpoints/`. This alone stops the two spurious "modified" `.DS_Store` entries.
- **Commit the untracked folders.** `edge_feature_ablation/`, `gbdt_comparison/`, `temporal_gnn_intransitivity/`, `preprocess/`, and the intransitivity notebooks are real, substantial work (the ablation framework in particular is well built — dataclass-driven config, frozen-prediction artifacts with manifest hashes, multi-seed runners with DM significance tests). Right now it only exists on this machine.
- **Decide on `results/frozen_predictions/` (11 MB).** The whole design point of "frozen" predictions is reproducible comparison — if that's meant to be durable, it should be committed (or moved to something like Git LFS / DVC if you want to keep repo size down); right now it's untracked, so a `git clean` or a fresh clone would silently lose the baseline everything else compares against.
- **Consolidate the duplicated `TennisGINE` notebooks.** The same class (down to the German inline comments) is copy-pasted into `NNs/start.ipynb`, `GNN.ipynb`, `GNN_experiment.ipynb`, `GNN_Slams_only.ipynb`, `GNN_Slams_only_edge+.ipynb`, `GNN_Slams_only_modified.ipynb`, plus `NN_test/` and `Intransitivity_NNs/`. `edge_feature_ablation/` already shows the better pattern (shared `.py` modules imported by notebooks) — worth doing the same for the model class instead of hand-copying it each time a variant is needed. At minimum, `start.ipynb` and `GNN.ipynb` look like superseded early iterations of `GNN_experiment.ipynb`, and `GNN_Slams_only_modified.ipynb` looks superseded by `GNN_Slams_only_edge+.ipynb` — worth confirming before archiving/deleting them, since I didn't diff every cell.
- **`environment.yml` doesn't include `torch`/`torch_geometric`** — every README works around this with manual `conda run -n tennis-gnn` instructions and a note about an OpenMP conflict on macOS. Worth pinning the working install command (or the wheel version that avoids the conflict) directly in `environment.yml` or a top-level `SETUP.md` so it's not tribal knowledge split across three READMEs.
- **No root `README.md`.** All documentation lives in per-folder READMEs (in Italian) — fine as-is, but a 5-line root README pointing to `edge_feature_ablation/README.md`, `gbdt_comparison/README.md`, and `temporal_gnn_intransitivity/README.md` would help orient anyone (including future-you) opening the repo cold.

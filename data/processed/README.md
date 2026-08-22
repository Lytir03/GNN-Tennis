# data/processed

Derived from `data/raw/sackmann/` by the notebooks in `preprocess/`. Nothing
here should be hand-edited — regenerate by rerunning the owning notebook.

| File | Built by | What it is |
|---|---|---|
| `atp_matches.csv` | `preprocess/data_prep.ipynb` | Full ATP match history, cleaned and merged from the raw yearly Sackmann files |
| `atp_matches_slams_masters.csv` | `preprocess/data_prep_slams_only.ipynb` | `atp_matches.csv` filtered to Grand Slams and Masters events (the `slams_masters` scope used throughout the study) |
| `atp_matches_slams.csv` | `preprocess/intransitivity_calc.ipynb` | Grand-Slams-only subset used for intransitivity analysis |
| `bscore_0_general.csv`, `bscore_0_clay.csv`, `bscore_0_grass.csv`, `bscore_0_hard.csv` | `preprocess/graph*.ipynb` | Initial (round-0) B-score values per surface, before any snapshot accumulation |
| `atp_matches_with_bscore.csv` | `preprocess/graph.ipynb` | `atp_matches.csv` joined with B-score features (surface-agnostic) |
| `atp_matches_with_bscore_Hard.csv`, `atp_matches_with_bscore_clay.csv`, `atp_matches_with_bscore_Grass.csv` | `preprocess/graph_hard1.ipynb`, `graph_clay1.ipynb`, `graph_grass1.ipynb` | Same, restricted to one surface. Casing of the surface suffix is inconsistent (`Hard`/`clay`/`Grass`) — an artifact of each notebook being copy-edited separately, not a semantic difference |
| `bscore_snapshots.csv`, `bscore_snapshots_hard.csv`, `bscore_snapshots_clay.csv`, `bscore_snapshots_grass.csv` | `preprocess/graph.ipynb` and surface variants | Full per-round B-score snapshot history (largest files, up to 40MB) — the input the GNN pipeline reads to rebuild graphs |
| `match_intransitivity_scores_with_levels.csv` | `preprocess/intransitivity_calc.ipynb` | Per-match intransitivity score plus a discretized level (low/medium/high) |
| `atp_matches_with_bscore_and_intransitivity.csv` | `preprocess/intransitivity_calc.ipynb` | `atp_matches_with_bscore.csv` joined with intransitivity levels |
| `atp_matches_with_bscore_low_intransitivity.csv`, `..._medium_intransitivity.csv`, `..._high_intransitivity.csv` | `preprocess/intransitivity_calc.ipynb` | The above, split by intransitivity level for the group-by analysis in the open questions |

## Regenerating

Run the notebooks in `preprocess/` in this order: `data_prep.ipynb` →
`data_prep_slams_only.ipynb` → `graph.ipynb` (+ `graph_hard1.ipynb`,
`graph_clay1.ipynb`, `graph_grass1.ipynb`) → `intransitivity_calc.ipynb`.
Each writes only the files listed under it above.

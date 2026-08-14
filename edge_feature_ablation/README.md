# Edge-feature ablations

Questa cartella contiene una variante isolata dell'esperimento GINE esistente.
Non aggiunge rating o statistiche aggregate: le nuove feature derivano solo dal
singolo incontro.

## Vettore completo

L'ablation `full` usa, in quest'ordine:

```text
recency_weight
signed_relative_game_diff
signed_set_margin
straight_sets_flag
completed_match_flag
surface_hard
surface_clay
surface_grass
round_scaled
result_direction
```

Per `loser -> winner`, i due margini e `result_direction` sono positivi; per
`winner -> loser` sono negativi. Le feature che descrivono il match restano
uguali nei due versi.

`set_margin` conta soltanto set conclusi e viene diviso per `best_of` (3 o 5).
Un match è `straight_sets` soltanto se è completo, il perdente non ha vinto set
e il vincitore ha raggiunto il numero di set necessario.

## Match incompleti

Il parser distingue `RET`, `W/O` e anche i rari `DEF`/`ABD`. La configurazione
predefinita mantiene questi incontri, imposta `completed_match_flag = 0` e
azzera game margin, set margin e straight-sets. È la scelta più conservativa:
il punteggio parziale di un ritiro non viene scambiato per il margine finale.

Sono disponibili due sensitivity check:

- `incomplete_margin_policy="played"` conserva i game e i set effettivamente
  giocati prima dell'interruzione;
- `exclude_unparseable_incomplete=True` esclude walkover e altri match
  incompleti senza alcun set ricostruibile.

L'ablation `match_status` espone anche `retirement_flag` e `walkover_flag`
separati. `full` usa il vettore compatto richiesto con il solo
`completed_match_flag`.

## Preset

I preset sono cumulativi:

| Nome | Modifica |
|---|---|
| `current` | rappresentazione precedente: margine uguale nei due versi e flag 0/1 |
| `signed_game` | margine in game direzionale e direzione +1/-1 |
| `set_margin` | aggiunge il margine in set direzionale |
| `straight_sets` | aggiunge il flag straight-sets |
| `match_status` | aggiunge completed, retirement e walkover |
| `full` | combinazione compatta a 10 feature con completed |

`match_status` ha 12 feature perché serve a misurare se distinguere i tipi di
interruzione aiuta. `full` ne ha 10 e corrisponde al vettore richiesto.
Fino a `straight_sets` il punteggio parziale dei match incompleti è conservato,
come nella baseline; la policy conservativa che azzera quei margini entra
soltanto insieme ai flag di stato. Il passaggio 2 ricodifica inoltre il vecchio
flag reverse-edge 0/1 come direzione esplicita +1/-1: è una trasformazione
affine della stessa informazione, non una nuova feature.

## Esecuzione

Aprire ed eseguire `GNN_edge_feature_ablation.ipynb` con working directory
`edge_feature_ablation`, oppure selezionare il preset senza modificare il
notebook:

All'inizio del notebook è presente anche la flag manuale:

```python
TOURNAMENT_SCOPE = "slams_masters"
```

Usare `"slams"` per includere soltanto i Grand Slam, oppure
`"slams_masters"` per includere Grand Slam e Masters. La scelta viene applicata
sia alla GNN sia al B-score logit della comparison finale.

```bash
cd edge_feature_ablation
TENNIS_EDGE_ABLATION=signed_game jupyter nbconvert \
  --execute --to notebook GNN_edge_feature_ablation.ipynb \
  --output GNN_edge_feature_ablation_signed_game.ipynb
```

Ripetere con i sei nomi sopra. Il notebook legge solo
`TENNIS_EDGE_ABLATION`; seed (`42`), split temporali, dataset, modello,
ottimizzatore, replay buffer e procedura di training provengono invariati da
`NN_test/GNN_intransitivity_changes.ipynb`, cioè dalla variante bidirezionale
che usa `atp_matches_slams_masters.csv`.

Se cambia il notebook sorgente, rigenerare la variante:

```bash
python edge_feature_ablation/make_experiment_notebook.py
```

Eseguire i test del parser e della direzionalità con:

```bash
python -m unittest discover -s edge_feature_ablation -p "test_*.py"
```

## Esperimenti congelati

I notebook nella cartella `experiments/` evitano di ricalcolare le baseline:

1. `00_freeze_baselines.ipynb` esegue e salva una volta `current_gnn` e
   `bscore_logit`;
2. `01_edge_feature_ablations.ipynb` esegue una edge ablation selezionata;
3. `02_architecture_ablations.ipynb` esegue una modifica architetturale
   non temporale;
4. `03_final_comparison.ipynb` legge gli artefatti senza allenare modelli.

Le predizioni vengono salvate in:

```text
results/frozen_predictions/<scope>/seed_<seed>/
```

Ogni CSV contiene probabilità e target per singolo match. Il relativo manifest
registra scope, seed, split, fasi di aggiornamento e configurazione. Il notebook
finale interrompe il confronto se gli hash dei match o le configurazioni
sperimentali fondamentali non coincidono.

Le ablation architetturali disponibili sono cumulative:

```text
current_gnn
edge_full
bscore_residual
antisymmetric_decoder
node_normalization
mean_aggregation
residual_connections
tournament_context
full_non_temporal
```

Sono inoltre disponibili ablation mirate attorno al decoder antisimmetrico,
senza ereditare node normalization o residual connection:

```text
antisymmetric_no_direct_bscore
antisymmetric_no_node_bscore
antisymmetric_no_bscore
antisymmetric_straight_sets
antisymmetric_mean
antisymmetric_tournament_context
antisymmetric_mean_tournament
```

## Ablation del B-score

Le tre varianti usano la migliore GNN congelata e non modificano training,
split o iperparametri:

- `antisymmetric_no_direct_bscore`: B-score nei nodi, non nel decoder;
- `antisymmetric_no_node_bscore`: B-score diretto nel decoder, non nei nodi;
- `antisymmetric_no_bscore`: nessun B-score.

I quattro canali B-score dei nodi vengono azzerati, mantenendo invariata la
dimensione dell'input e il numero di parametri. Il GBDT `no_bscore` riusa gli
iperparametri selezionati dal GBDT completo, senza ulteriore tuning.

```bash
conda run -n tennis-gnn python edge_feature_ablation/run_bscore_ablation.py \
  --scope slams_masters --seeds 42
conda run -n gbdt-tennis python gbdt_comparison/run_multi_seed.py \
  --scope slams_masters --seeds 42 --feature-set no_bscore
python edge_feature_ablation/bscore_ablation_summary.py \
  --scope slams_masters --seeds 42
```

La matrice mirata può essere eseguita con:

```bash
conda run --no-capture-output -n tennis-gnn \
  python edge_feature_ablation/run_experiment_matrix.py \
  --scope slams_masters --group targeted
```

`tournament_context` aggiunge agli archi storici soltanto
`best_of_5_flag` e `grand_slam_flag`. Non è inclusa alcuna rete neurale
temporale.

## Verifica multi-seed

Il seed del notebook principale è configurabile tramite `TENNIS_SEED`.
Per congelare baseline e decoder antisimmetrico sui seed standard:

```bash
conda run --no-capture-output -n tennis-gnn \
  python edge_feature_ablation/run_multi_seed.py \
  --scope slams_masters --seeds 42 123 456 789 2026
```

Il runner salta gli artefatti già completi. Il riepilogo è disponibile in
`experiments/04_multi_seed_comparison.ipynb` e riporta metriche per seed,
media, deviazione standard e numero di vittorie rispetto a `current_gnn`.
Impostando `TENNIS_RUN_SCOPE=slams` lo stesso notebook legge gli artefatti
Grand Slam-only senza mescolarli con quelli Slam+Masters.

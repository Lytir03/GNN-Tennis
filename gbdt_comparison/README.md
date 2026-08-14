# GBDT comparison

Baseline tabulare basata su `HistGradientBoostingClassifier`, con tuning
esclusivamente sulla validation 2016 e test congelato 2017–2020.

## Feature

Il modello usa informazioni disponibili anche alla GNN:

- B-score generale e della superficie per entrambi i giocatori;
- altezza e mano;
- superficie, round, best-of-five e livello Slam;
- storico causale degli ultimi tre anni;
- recency `1 / (1 + age_days / 365)`;
- risultato, margine game, margine set, straight sets e completezza;
- aggregazioni generali e limitate alla superficie corrente.

Le statistiche storiche vengono costruite prima di aggiornare l'intero blocco
torneo/round, evitando leakage tra match dello stesso round. Score e margini del
match da prevedere non sono mai feature.

## Esecuzione

```bash
conda run --no-capture-output -n tennis-gnn \
  python gbdt_comparison/train.py \
  --scope slams_masters --seed 42 --rebuild-features
```

Il tuning esplora 32 configurazioni e seleziona la log-loss migliore sulla sola
validation. Dopo la selezione, il modello finale viene rifittato soltanto sul
training, come la GNN (`UPDATE_PHASES={"train"}`), e valutato una volta sul
test. Le label di validation non entrano nel fit finale.

Gli output vengono salvati in `gbdt_comparison/results/<scope>/`.
La cache delle feature usa pickle locale per conservare esattamente i float;
non caricare file pickle provenienti da fonti non fidate.
Una copia delle predizioni viene inoltre registrata nel formato congelato
comune sotto `results/frozen_predictions/<scope>/seed_<seed>/gbdt_tuned.*`.

Per eseguire tutti i seed:

```bash
conda run --no-capture-output -n tennis-gnn \
  python gbdt_comparison/run_multi_seed.py \
  --scope slams_masters --seeds 42 123 456 789 2026
```

`GBDT_comparison.ipynb` confronta poi GBDT, decoder antisimmetrico, GNN corrente
e B-score logit sugli stessi match.

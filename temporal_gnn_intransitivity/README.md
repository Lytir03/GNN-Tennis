# Temporal GNN e intransitività

Esperimento separato per verificare:

1. se uno stato dinamico per giocatore migliora rispetto alla GNN statica;
2. se il vantaggio della GNN è maggiore nei match localmente intransitivi;
3. se fornire esplicitamente l'intransitività causale migliora il modello.

## Protocollo

- warm-up storico: match precedenti al 2011;
- training: 2011–2015;
- validation report-only: 2016;
- test: 2017–2020;
- aggiornamento dello stato solo dopo avere predetto tutto il blocco
  torneo-round;
- nessun B-score;
- nessun tuning: 8 epoche, hidden size 32, AdamW;
- due varianti identiche:
  - `temporal_gnn`: non riceve l'intransitività;
  - `temporal_gnn_intransitivity`: riceve score, numero di avversari comuni
    e numero di archi locali.

Il punteggio Hodge locale disponibile nel progetto è già calcolato usando
solo la storia precedente. In questo esperimento l'intransitività è definita
come quota di energia ciclica:

```text
norm_cyclic² / (norm_cyclic² + norm_transitive²)
```

La quantità di evidenza non moltiplica più il punteggio. Serve invece come
filtro di affidabilità: occorrono almeno due avversari comuni, tre archi
locali ed energia non nulla. Le soglie low/medium/high vengono calcolate
usando esclusivamente il training 2011–2015.

La previsione avviene prima dell'aggiornamento. Dopo il match vengono inviati
due messaggi:

- loser → winner con margini positivi;
- winner → loser con margini negativi.

I messaggi includono margine in game e set, straight sets, completamento,
superficie, round, recency e direzione. La variante intransitiva aggiunge
anche il contesto Hodge causale.

## Esecuzione

```bash
conda run -n tennis-gnn python \
  temporal_gnn_intransitivity/run_experiments.py \
  --scope slams_masters --seeds 42
```

Il runner imposta `OMP_NUM_THREADS=1` e `KMP_DUPLICATE_LIB_OK=TRUE` perché
l'ambiente attuale combina il wheel pip di PyTorch con il runtime OpenMP
conda usato da NumPy. Questa è una compatibilità dell'ambiente macOS, non una
variante sperimentale.

Gli artefatti vengono salvati insieme alle baseline in:

```text
results/frozen_predictions/<scope>/seed_<seed>/
```

## Scouting seed 42

Su Slam+Masters:

| modello | accuracy | log loss | Brier |
|---|---:|---:|---:|
| Temporal GNN | 0.6188 | 0.6589 | 0.2326 |
| Temporal GNN + intransitività | 0.6382 | 0.6537 | 0.2301 |
| GNN statica senza B-score | 0.6714 | 0.6049 | 0.2087 |
| GBDT senza B-score | 0.6687 | 0.6054 | 0.2091 |

L'intransitività migliora la Temporal GNN soprattutto nel livello `high`
(accuracy `0.6278 → 0.6667`). In questo sottogruppo l'accuracy si avvicina
alla GNN statica senza B-score (`0.6752`) e al GBDT senza B-score (`0.6790`),
ma la log loss resta sensibilmente peggiore.

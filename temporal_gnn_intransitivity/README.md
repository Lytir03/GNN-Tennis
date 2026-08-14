# Temporal GNN e intransitività

Esperimento **esplorativo** su uno stato dinamico per giocatore. Verifica:

1. se uno stato ricorrente migliora rispetto alla GNN statica;
2. se il vantaggio della GNN è maggiore nei match localmente intransitivi;
3. se fornire esplicitamente l'intransitività causale migliora il modello.

## Stato attuale: risultato non concludente

I numeri qui sotto **non dimostrano che l'approccio temporale non funzioni**.
Due problemi rendono il confronto non interpretabile così com'è.

### 1. Gli artefatti non sono confrontabili match per match

Le predizioni salvate in `results/frozen_predictions/` per `temporal_gnn` e
`temporal_gnn_intransitivity` hanno un `evaluation_hash` diverso da quello di
tutti gli altri modelli del progetto:

| artefatti | evaluation_hash | block_idx (test) |
|---|---|---|
| GNN statica, GBDT, B-score logit | `575e4dc2…` | 504–794 |
| Temporal GNN | `2e11d43e…` | 924–1214 |

Le cause sono due: la numerazione dei blocchi parte da un punto diverso, e il
sorteggio dell'orientamento (quale giocatore è "A") consuma i numeri casuali in
un ordine diverso, quindi `y_true` differisce riga per riga. I match valutati
sono gli stessi 3770, e le metriche aggregate restano quindi indicative, ma
`assert_compatible` rifiuta correttamente questa coppia: **nessun test appaiato
- Diebold-Mariano, delta per match, intervalli di confidenza - è valido fra le
due famiglie.**

### 2. Il confronto è fra un modello non tarato e due tarati

Questo esperimento usa esplicitamente "nessun tuning: 8 epoche, hidden size 32,
AdamW". Il GBDT ha ricevuto 32 configurazioni selezionate sulla validation, e
la GNN statica è passata da uno studio di ablation. In più il modello temporale
è strutturalmente più piccolo: aggiorna lo stato con **un solo hop** di message
passing per evento, mentre la GNN statica ne ha due.

Quindi la differenza misurata confonde tre cose insieme: temporale contro
statico, tarato contro non tarato, e un hop contro due.

## Numeri (seed 42, Slam+Masters)

| modello | accuracy | log loss | Brier |
|---|---:|---:|---:|
| Temporal GNN | 0.6188 | 0.6589 | 0.2326 |
| Temporal GNN + intransitività | 0.6382 | 0.6537 | 0.2301 |
| GNN statica senza B-score | 0.6714 | 0.6049 | 0.2087 |
| GBDT senza B-score | 0.6687 | 0.6054 | 0.2091 |

L'intransitività migliora la Temporal GNN soprattutto nel livello `high`
(accuracy `0.6278 → 0.6667`). Questo segnale interno all'esperimento resta
valido, perché confronta due varianti costruite con la stessa pipeline e quindi
con lo stesso `evaluation_hash`.

## Cosa servirebbe per una conclusione

In ordine di importanza:

1. allineare la costruzione dei target a `tennis_gnn/snapshots.py`, così che gli
   artefatti condividano l'`evaluation_hash` e i test appaiati diventino leciti;
2. aggiungere un secondo hop di propagazione, per pareggiare la capacità;
3. dare allo schedule lo stesso budget di ricerca su validation usato dagli
   altri modelli (`tennis_gnn/tune.py` come riferimento).

Finché questi tre punti non sono fatti, il risultato corretto da riportare è
"non dimostrato", non "il temporale non aiuta".

## Protocollo

- warm-up storico: match precedenti al 2011;
- training: 2011–2015;
- validation report-only: 2016;
- test: 2017–2020;
- aggiornamento dello stato solo dopo avere predetto tutto il blocco
  torneo-round;
- nessun B-score;
- due varianti identiche: `temporal_gnn` non riceve l'intransitività,
  `temporal_gnn_intransitivity` riceve score, numero di avversari comuni e
  numero di archi locali.

L'intransitività è definita come quota di energia ciclica:

```text
norm_cyclic² / (norm_cyclic² + norm_transitive²)
```

La quantità di evidenza serve come filtro di affidabilità: almeno due avversari
comuni, tre archi locali ed energia non nulla. Le soglie low/medium/high sono
calcolate usando esclusivamente il training 2011–2015.

## Esecuzione

```bash
conda run -n tennis-gnn python \
  temporal_gnn_intransitivity/run_experiments.py \
  --scope slams_masters --seeds 42
```

Il runner imposta `OMP_NUM_THREADS=1` e `KMP_DUPLICATE_LIB_OK=TRUE` perché
l'ambiente combina il wheel pip di PyTorch con il runtime OpenMP conda usato da
NumPy. È una compatibilità dell'ambiente macOS, non una variante sperimentale.

Test:

```bash
conda run -n tennis-gnn python -m unittest discover \
  -s temporal_gnn_intransitivity -p "test_*.py"
```

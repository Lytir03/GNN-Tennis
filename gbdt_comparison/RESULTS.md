# Risultati multi-seed

Seed: `42, 123, 456, 789, 2026`.

## Slam + Masters

| Modello | Accuracy | Log-loss | Brier |
|---|---:|---:|---:|
| B-score logit | 0.644191 | 0.626019 | 0.216032 |
| GNN corrente | 0.667215 | 0.612380 | 0.211303 |
| Decoder antisimmetrico | 0.667109 | 0.609226 | 0.210325 |
| GBDT tuned | **0.669496** | **0.601504** | **0.207433** |

Rispetto al decoder antisimmetrico, il GBDT migliora mediamente log-loss di
`0.007721` e Brier di `0.002892`. Vince su entrambe in 4 seed su 5; gli
intervalli seed-level al 95% dei due delta escludono zero.

## Solo Slam

| Modello | Accuracy | Log-loss | Brier |
|---|---:|---:|---:|
| B-score logit | 0.707312 | 0.577825 | 0.193060 |
| GNN corrente | 0.669516 | 0.616166 | 0.213097 |
| Decoder antisimmetrico | 0.702362 | 0.583427 | 0.198568 |
| GBDT tuned | **0.709561** | **0.559467** | **0.188549** |

Il GBDT supera il decoder antisimmetrico in tutti i cinque seed su accuracy,
log-loss e Brier. Supera il B-score logit in 4/5 seed sull'accuracy e 5/5 su
log-loss e Brier.

Il tuning usa soltanto la validation 2016. Il fit finale usa il training fino
al 2015 e il test 2017–2020 viene valutato una volta.

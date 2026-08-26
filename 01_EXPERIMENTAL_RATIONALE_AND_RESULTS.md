# Experimental Rationale and Results

This document reports the project as a sequence of research questions: **why I ran each experiment, what I expected to learn, what the experiment showed, and how that changed the interpretation of the GNN.** Implementation details and debugging history are deliberately excluded unless they are necessary to understand a result.

## 1. Overall research question

The central question was not simply whether a GNN could predict tennis matches accurately, but **whether the graph representation contributes predictive information that a strong tabular model cannot recover as easily**.

The graph is a natural representation of tennis history: players are nodes and past matches create edges. This makes it possible for the model to use not only player-level summaries such as B-score or recent match statistics, but also the structure of the opponent network. The key experimental problem was therefore to separate three possible sources of performance:

1. better player-level information;
2. a better decoder or training procedure;
3. information that genuinely comes from message passing over the graph.

The final headline results are based on the **full ATP-tour dataset with 35,474 scored matches**, including 15,809 training, 5,325 validation and 14,340 test matches. Earlier experiments were run on the smaller Slam + Masters subset with 10,212 scored matches. Whenever conclusions changed after expanding the population, the full-tour result is treated as the stronger evidence.

---

## 2. First question: was the original GNN–GBDT gap actually architectural?

### Why I tested this

The original comparison suggested that the tuned GBDT was better than the GNN. Before interpreting that as evidence against the graph architecture, I wanted to determine whether the two models had received comparable optimisation effort and comparable information.

The comparison was not initially balanced. The GBDT had been selected from a 32-configuration validation search, whereas the GNN used essentially one inherited training recipe. The GBDT also received match-context variables such as `best_of_5_flag` and `grand_slam_flag` that the GNN did not receive.

The purpose of this experiment was therefore to ask:

> If the GNN is tuned properly and is given the same match-level information, does the apparent GBDT advantage remain?

### What I found

On the Slam + Masters data, after tuning the GNN and restoring context parity, the earlier statistically significant gap essentially disappeared.

| model, 5-seed mean | accuracy | log loss | Brier |
|---|---:|---:|---:|
| GBDT tuned | 0.6695 | **0.6015** | 0.2074 |
| GNN tuned | **0.6700** | 0.6022 | 0.2077 |

For GNN versus GBDT, the paired log-loss difference was **+0.00070**. The accuracy difference was also non-significant.

### What I took from it

At this stage the correct conclusion was a **statistical tie**, not a GNN win. More importantly, most of the previous deficit had not been evidence that graph modelling was intrinsically worse. It came from unequal tuning, unequal feature access, and an already-discovered decoder choice that had not yet been used in the main model.

This changed the direction of the work: rather than asking only how to increase aggregate accuracy, I moved to experiments designed to isolate **what the graph itself was contributing**.

---

## 3. Antisymmetric decoding: encoding a property of the prediction problem

### Why I tested it

A tennis match has no home-team ordering. If the probability that A beats B is `p`, then reversing the player order should give exactly `1-p`.

A generic pairwise neural decoder does not guarantee this. It must learn the symmetry from examples. I therefore tested an antisymmetric decoder of the form

`0.5 * (score(a,b) - score(b,a))`,

which enforces the relationship by construction.

### What I found

This was the largest clean architectural gain in the earlier ablation results: approximately **0.6044 versus 0.6119 log loss** for the non-antisymmetric alternative. In the later decomposition of the improved GNN, the decoder accounts for roughly **0.0032 log-loss improvement** relative to the original GNN configuration.

### What I took from it

The result suggests that a substantial part of the GNN improvement came not from deeper graph propagation but from **parameterising the prediction problem correctly**. The model no longer needs to spend capacity learning a symmetry that is known in advance.

---

## 4. Calibration experiment: was the log-loss gap caused by overconfidence?

### Why I tested it

The GNN's accuracy was competitive while its log loss was worse. A plausible explanation was that the ranking of matches was good but the predicted probabilities were too confident.

Temperature scaling is an appropriate diagnostic because it changes confidence without changing the ordering of predictions. If miscalibration were the main issue, a fitted temperature should noticeably improve log loss while leaving accuracy unchanged.

### What I found

The fitted temperature was **T = 1.008**, essentially the identity transformation. The improvement was negligible.

### What I took from it

The hypothesis was rejected. The remaining loss gap was not principally a calibration problem. The model needed to improve the underlying ordering or representation of matches rather than simply rescale its confidence.

---

## 5. Optimisation experiment: more capacity or more training?

### Why I tested it

Once the comparison was fairer, the next question was whether the GNN was limited by model capacity, regularisation, or incomplete optimisation. I compared a larger hidden dimension, lower dropout, and a longer training schedule while inspecting both training and validation loss.

### What I found

| configuration             | train log loss| validation log loss | test log loss | test accuracy |
|---                        |---       :|      ---:     |---:|---:|
| tuned, h32, dropout 0.3   | 0.5470    | **0.5496**    | 0.6069    | 0.6703 |
| h64                       | 0.5464    |      0.5531   | 0.6129    | 0.6682 |
| dropout 0.1               | 0.5469    | **0.5492**    | 0.6109    | 0.6700 |
| 3 passes                  | **0.5395**| 0.5516        | **0.6010** | **0.6719** |

The longer schedule reduced training loss substantially on the first inspected seed, while more capacity or less dropout did not.

However, the apparent test advantage of the 3-pass model **did not replicate across five seeds**. It won only on seed 42 and was worse on the other four. Its five-seed mean log loss was 0.6034, compared with 0.6022 for the tuned one-pass recipe.

### What I took from it

The useful result was not that three passes were better; they were not. The experiment showed that single-seed diagnostics can make a plausible optimisation story look much stronger than it is. The stable recipe remained the tuned one-pass configuration.

---

## 6. Intransitivity hypothesis: do common opponents make deeper graph propagation useful?

### Why I tested it

A major motivation for a GNN was the possibility that tennis contains useful relational structure that player-level aggregates do not fully capture. The specific hypothesis was that the GNN should gain on matches where two players are connected through common opponents.

This matters because the GBDT already receives many one-player history aggregates, but no explicit common-opponent feature. A clean way to test the graph hypothesis is therefore to ask whether expanding the GNN receptive field from one hop to two hops provides a larger gain when a shared opponent exists.

### Why the final experiment used depth rather than subgroup comparisons

A direct GNN-versus-GBDT subgroup comparison cannot isolate the graph effect because the models differ in architecture, features and fitting procedure at the same time. I therefore used **GNN depth as the intervention**:

- one hop exposes each player to their own direct opponents;
- two hops are required for information from a shared opponent to reach the representation.

The relevant experiment is therefore **two hops versus one hop**, not one hop versus no message passing.

### What I found

The result depended strongly on the tournament population.

#### Slam + Masters subset

The second-hop interaction appeared large and significant in several feature regimes. For example, with B-score plus static features, the interaction was **-0.01148**, with 95% CI **[-0.01412, -0.00884]**.

#### Full ATP tour

The effect disappeared. At the same feature tier, the interaction was **-0.00033**, with 95% CI **[-0.00328, +0.00262]**. With history fed directly to the decoder it was **-0.00009**, with 95% CI **[-0.00277, +0.00259]**.

More generally, on the full tour the second hop was harmful at every tested feature tier:

| feature tier | 2 hops vs 1 hop, full ATP tour |
|---|---:|
| no B-score, no history | +0.00127 |
| B-score + static | +0.00227 |
| + history on nodes | +0.01278 |
| + history at decoder | +0.00942 |

Negative values would mean that the second hop helped; these values are positive.

### What I took from it

The original intransitivity hypothesis was reasonable, but **relational depth beyond the direct neighbourhood did not generalise to tour-wide ATP data**.

The smaller Slam + Masters population is unusually dense and elite. Common opponents there are often highly recorded players and may carry stronger signal. Once ATP 250 and 500 tournaments are included, the same structural relationship becomes much less informative. The population expansion therefore changed the scientific conclusion, not merely the precision of the estimate.

The final conclusion is:

> **A second relational hop does not add useful predictive information on the full ATP tour.**

---

## 7. Feature-parity experiment: what is message passing substituting for?

### Why I tested it

Once deeper relational structure failed to explain the GNN's value, I wanted to determine whether message passing was mainly acting as an alternative way of reconstructing ordinary player-history features.

The GBDT had twelve recency-weighted history statistics per player, including result balance, game and set margins, straight-sets balance and completion rate, calculated both overall and on the current surface. I therefore gave the GNN the same history information and measured how the marginal value of message passing changed as player-level features became richer.

### What I found: the substitution curve

I held the architecture fixed, changed the information available to the model, and measured **one hop minus no message passing**. Negative values mean that the graph helped.

| tier | information available | Slam + Masters | full ATP tour |
|---|---|---:|---:|
| 0 | height, handedness                | -0.08125 | **-0.06171** |
| 1 | + B-score, general and surface    | -0.01418 | **-0.01355** |
| 2 | + 12 history statistics on nodes  | +0.00689 | **-0.00121** |
| 3 | + 12 history statistics at decoder | +0.00002 | **-0.00100** |

The shape is extremely clear. On the full tour, the graph contribution falls from roughly **0.062 log loss** when almost no useful player information is provided to about **0.001** when rich engineered history features are already available.

### The most informative endpoint

At tier 0, the no-message model produces log loss **0.693147**, exactly the coin-flip baseline. This remains true across different optimisation recipes. With only height and handedness there is essentially no predictive player information available to that model.

Adding one hop reduces log loss to about **0.631**. In this regime the graph is not merely refining existing player features; **the graph is the main source of usable information** because it exposes the model to whom each player has faced and how those matches ended.

### What I took from it

This became the central result of the project:

> **Graph structure and engineered player history are strongly substitutable sources of information.**

Message passing is highly valuable when the model has weak player features. As direct player-history features become richer, the incremental value of the graph falls sharply but does not disappear completely on the full tour.

---

## 8. Structural sparsity experiment: when does message passing help or hurt?

### Why I tested it

The substitution curve concerns feature richness. I also wanted to know whether the graph requires a sufficiently informative neighbourhood to be useful. A player with very few prior opponents gives the GNN little structure to aggregate.

I therefore stratified the one-hop versus no-message comparison by the number of prior opponents of the less-recorded player in the match.

### What I found

At the richest feature tier:

| prior opponents | matches | one hop vs no messages | seeds won by one hop |

| 0-5 | 1,404 | **+0.00413**    | 0/5 |
| 6-20 | 2,449 | +0.00155       | 1/5 |
| 21+ | 10,487 | **-0.00229**   | 5/5 |

Message passing is harmful in cold-start conditions and useful when players have richer graph histories.

A similar monotonic pattern appears when matches are grouped by the number of shared opponents: approximately +0.0035 at zero common opponents and -0.0024 at 15 or more.

### What I took from it

The graph cannot compensate for the absence of structure. When a player has only a few recorded edges, aggregating neighbours can dilute the strong B-score prior without adding enough evidence in return.

The result therefore complements the substitution curve:

- **poor player features + rich graph structure:** message passing can add a great deal;
- **rich player features + rich graph structure:** message passing adds a small but repeatable amount;
- **poor graph structure / cold start:** message passing can hurt.

---

## 9. Final architecture experiment: preserve player history and add only one graph hop

### Why I tested it

Feeding engineered history features into the node encoder did not improve the GNN much. A plausible explanation is that message passing smooths those player-specific statistics together with neighbourhood information.

I therefore tested a model that keeps the twelve history statistics **directly available to the match decoder**, while the graph encoder handles structural information separately. This allows the model to use undiluted player history and then ask whether a single graph hop still contributes anything beyond it.

### What I found

On the full ATP tour, with five paired seeds and identical matches and labels:

| model | accuracy | log loss |
|---|---:|---:|
| **GNN, history -> decoder, 1 hop** | **0.6458** | **0.61915** |
| GNN, history -> decoder, no messages | 0.6437 | 0.62015 |
| GBDT, tuned | 0.6413 | 0.62487 |
| GNN, history -> decoder, 2 hops | 0.6392 | 0.62857 |

Paired contrasts:

| contrast | difference | 95% CI | seed result |
|---|---:|---|---:|
| **GNN vs GBDT, log loss** | **-0.00572** | **[-0.00719, -0.00426]** | **5/5** |
| GNN vs GBDT, Brier | -0.00258 | [-0.00321, -0.00195] | 5/5 |
| GNN vs GBDT, accuracy | +0.00448 | [+0.00165, +0.00731] | 5/5 |
| **1 hop vs no messages** | **-0.00100** | **[-0.00179, -0.00022]** | **5/5** |
| 2 hops vs 1 hop | +0.00942 | [+0.00560, +0.01325] | 0/5 |

The one-hop model was also preferred on validation: **0.61384** versus **0.61600** for the no-message arm.

### What I took from it

This is the strongest final result. The GNN beats the tuned GBDT on all three metrics, on every seed, and the confidence intervals exclude zero.

However, the graph itself is only a **small part** of that advantage. Removing the single message-passing hop costs about **0.001 log loss**, whereas the full GNN advantage over the GBDT is about **0.0057**.

The final performance therefore comes from a combination of:

- the antisymmetric decoder;
- a B-score skill-gap residual;
- direct access to the twelve engineered history statistics at the decoder;
- one hop of message passing;
- a validation-selected training recipe.

The graph contribution is real and repeatable, but modest once strong player-history features are supplied.

---

## 10. Final interpretation

The experiments do **not** support a simple claim that deeper relational reasoning is the reason GNNs work for tennis. The evidence instead supports a more specific picture.

1. **The graph can reconstruct player history when explicit player features are weak.** With only height and handedness, one-hop message passing improves log loss by roughly 0.062 over the no-message model.
2. **The marginal value of graph structure falls steeply as direct player-history information improves.** At the richest feature tier, one hop is worth about 0.001 log loss.
3. **The graph needs enough structure to be useful.** It hurts cold-start players and helps well-connected players.
4. **More relational depth is not better.** Two hops are consistently worse than one on the full ATP tour, and there is no robust tour-wide common-opponent advantage.
5. **A carefully parameterised one-hop GNN can still outperform a tuned GBDT.** The final model improves log loss by 0.00572, Brier score by 0.00258 and accuracy by 0.00448, with the direction reproduced on all five seeds.

A concise thesis-style statement is therefore:

> On tour-wide ATP data, graph structure behaves primarily as an alternative representation of player match history rather than as evidence for useful deep relational propagation. Its predictive value is large when direct player information is weak, falls sharply as engineered history features are supplied, and remains small but significant when those features are already strong. Message passing requires a sufficiently rich neighbourhood, harms cold-start cases, and gains nothing from a second hop. A model combining an antisymmetric B-score-based decoder, direct player-history features and a single message-passing hop outperforms a tuned gradient-boosted baseline by 0.00572 log loss across all five seeds.

## 11. Remaining caveat on the headline

The final 0.00572 log-loss advantage is repeatable, but small. The learning rate used by the winning model was discovered while investigating an anomaly rather than through a complete tour-scale per-arm hyperparameter sweep. Because different ablation arms respond differently to training hyperparameters, a full per-arm search remains the most important robustness check for the final number.

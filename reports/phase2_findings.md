# Phase 2 findings

Model comparison and mobility deserts. Everything below is from
`python src/run_phase2.py --compare-universes --max-iter 20000`, fit on
2018–2021 and tested on 2022–2024.

## Which model?

Full universe (430 occupations, 170k pairs, 3.2% observed):

| Model | Poisson deviance ↓ | Spearman ↑ | recall@10 ↑ |
|---|---|---|---|
| `gravity_plus_history` | **948.4** | **0.417** | **0.408** |
| `flow_embedding` | 2069.7 | 0.416 | 0.408 |
| `boosted_poisson` | 2,262,692,680 | 0.225 | 0.306 |
| `ppml_gravity` | 822.1 | 0.315 | 0.261 |
| `ppml_raw_counts` | 838.9 | 0.304 | 0.260 |

**Past flow dominates occupational similarity.** Adding lagged pair flow to
the gravity model lifts recall@10 from 0.261 to 0.408 — a 56% improvement, and
far larger than anything the O*NET features contribute on their own.

`flow_embedding` — which never sees a single O*NET descriptor — matches it on
recall. That is the uncomfortable result this comparison existed to surface:
**most of the predictive signal is in the transition graph, not in what the
jobs involve.** The features earn their keep only where history is absent.

`gravity_plus_history` is the better of the two despite the tie on recall: its
deviance is less than half, meaning it gets the *levels* right and not just
the ordering. It also degrades gracefully on pairs with no history, which
`flow_embedding` cannot do at all.

**`boosted_poisson` is a negative result worth keeping.** Its deviance is nine
orders of magnitude off. It ranks acceptably (recall 0.306) but its predicted
levels are wild — Poisson-loss gradient boosting on a target this
overdispersed and this sparse is badly miscalibrated. Ranking metrics alone
would have hidden that completely.

### Structural vs forecasting

These are not competing on one axis:

- **Features-only** (`ppml_gravity`, recall **0.261**) answers *what about two
  occupations makes movement likely*. It transfers to pairs never observed.
- **With history** (`gravity_plus_history`, recall **0.408**) answers *what
  will flow next year*. Useless for a pair with no history.

**This is the number that matters for the deployment story.** A company
mapping its role catalogue onto this model has no internal transition history
on day one. It gets 0.261, not 0.408. The honest claim to make about
cold-start is the smaller one.

### Overdispersion

Person counts run a variance-to-mean ratio of **19.7**; the survey-weighted
target, about **39,000**. Poisson assumes 1.

PPML point estimates survive this — that robustness is why it's the standard
gravity estimator — and `ppml_raw_counts` confirms it: fitting honest person
counts instead of weighted ones moves recall by 0.001. The ranking is safe.
Any standard error from either is not.

## Does scoping to white collar help?

| Universe | Occupations | Best recall@10 | Spearman |
|---|---|---|---|
| all | 430 | 0.408 | 0.417 |
| white-collar | 257 | 0.485 | 0.470 |
| professional | 189 | **0.510** | **0.526** |

**Yes — narrowing to professional occupations improves recall@10 by 25%**
(0.408 → 0.510). Fewer occupations means a denser transition matrix and more
homogeneous mobility patterns, and the model gets measurably better.

### What it costs

Of the **94** low-paid, narrow-option occupations the full universe finds:

- **33 (35%)** survive a white-collar filter
- **11 (12%)** survive a professional filter

Where the deserts actually live:

| Count | SOC major group |
|---|---|
| 20 | Production |
| 17 | Office and Administrative Support |
| 10 | Personal Care and Service |
| 7 | Transportation and Material Moving |
| 6 | Healthcare Support |
| 5 | Sales and Related |

Those 94 occupations employ roughly **13.1 million people**.

So scoping to white collar buys a 25% accuracy gain and deletes **88%** of the
finding. These are two different products:

- **A career tool** — scope to white collar. The users have agency over career
  moves, the predictions are better, and the excluded occupations aren't the
  audience.
- **The research finding** — keep the full universe. Mobility deserts are
  overwhelmingly *not* white collar, and restricting to professional
  occupations means studying mobility among the already-mobile.

Both are supported. `--universe` makes it a flag rather than a fork.

## Mobility deserts, full universe

Narrowest options among low-paid occupations. `effective_destinations` is
`exp(entropy)` of the predicted destination mix — an occupation at 3.3 has
about three realistic ways out. `upward_share` is the fraction of predicted
outflow to occupations paying at least 10% more.

| Occupation | Effective destinations | Upward share | Median wage |
|---|---|---|---|
| Agricultural Inspectors | 3.31 | 0.03 | $49,940 |
| Paper Goods Machine Setters/Operators | 3.48 | 0.03 | $50,270 |
| Coin, Vending, and Amusement Machine Servicers | 3.09 | 0.08 | $47,450 |
| Refuse and Recyclable Material Collectors | 3.74 | 0.05 | $49,690 |
| Library Technicians | 3.25 | 0.09 | $44,580 |
| Meter Readers, Utilities | 4.67 | 0.02 | $48,150 |
| Medical Secretaries and Administrative Assistants | 5.14 | 0.04 | $45,930 |
| Credit Authorizers, Checkers, and Clerks | 5.87 | 0.01 | $50,080 |
| Manicurists and Pedicurists | 4.05 | 0.12 | $35,760 |
| Bus Drivers, School | 5.53 | 0.05 | $47,920 |

The `upward_share` column is the sharper signal. Several of these have a
handful of destinations *and* essentially none that pay more — Credit
Authorizers at 0.01 means about 1% of predicted moves lead to a raise.

## Caveats

- Deserts are computed from the best-ranking model's predictions, which
  include lagged flow. An occupation that was narrow historically will look
  narrow here. Distinguishing "narrow because of its skill profile" from
  "narrow because it has been" needs the features-only model, and the two
  disagree — worth a follow-up.
- `NARROW_THRESHOLD` (12 effective destinations) and the 40th-percentile wage
  cutoff are reporting choices, not discovered boundaries.
- Recall@10 of 0.51 at best still misses half the real destinations.

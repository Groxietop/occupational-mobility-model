# Phase 1 findings

First fit of the gravity model against real data. Everything below comes from
`python src/run_phase1.py --train-years 2018 2019 2020 2021 --test-years 2022 2023 2024 --max-iter 20000`.

## Data actually used

| | |
|---|---|
| CPS ASEC person-years | 1,124,817 |
| ...SOC-mapped and employed | 432,786 |
| ...that changed occupation | **48,887** |
| Occupations in the universe | 430 (of 530 Census codes) |
| Ordered pairs | 184,470 |
| Pairs with an observed move (train) | 5,441 — **3.2%** |
| OEWS wage coverage | 421/430 occupations (2025) |

The 3.2% density is the core difficulty: 97% of the matrix is zero, which is
why the estimator is PPML rather than anything that needs a log.

## Held-out scorecard (fit 2018–21, test 2022–24)

| Model | Poisson deviance ↓ | Spearman ↑ | recall@10 ↑ |
|---|---|---|---|
| `size_only` | 1142.5 | 0.183 | 0.087 |
| `equal_similarity` | 1030.6 | 0.218 | 0.118 |
| `onet_related` | 1163.2 | 0.217 | 0.162 |
| **`learned_gravity`** | **822.1** | **0.315** | **0.261** |

The learned model beats every baseline on every metric. Against the base-rate
baseline it roughly **triples** recall@10 (0.087 → 0.261); against the
equal-weighted similarity it currently replaces, it **doubles** it
(0.118 → 0.261).

Worth noting `onet_related` — O*NET's hand-curated related-occupation list —
beats equal-weighted similarity on recall while scoring *worse* on deviance.
Expert judgment picks plausible destinations but says nothing about volume.

## Learned domain weights

Standardised coefficients, so directly comparable to each other.

| Domain | Coefficient | Share of positive weight |
|---|---|---|
| interest | 0.413 | 23.7% |
| knowledge | 0.319 | 18.2% |
| skill | 0.299 | 17.1% |
| work_context | 0.248 | 14.2% |
| work_style | 0.247 | 14.1% |
| work_value | 0.133 | 7.6% |
| work_activity | 0.088 | 5.0% |
| ability | −0.136 | — |

**The headline: equal weighting was wrong.** The spread runs from 0.41 to
0.09 among the positive domains — nearly 5×. And the domain carrying the most
unique signal is `interest`, not `skill`, which is not what the equal-weighted
model assumed and not the intuitive answer.

## The `ability` coefficient is suppression, not a finding

`ability` comes out negative jointly. That does **not** mean similar abilities
discourage a move. Fit alone, every domain is strongly positive:

| Domain | Alone | In the joint model |
|---|---|---|
| skill | +1.059 | +0.299 |
| interest | +0.897 | +0.413 |
| ability | **+0.877** | **−0.136** |

`ability` correlates 0.74 with `skill` and 0.76 with `work_context`. Once
those are in the model it has no unique variance left, and the coefficient
flips. The correct reading is **redundant**, not repellent.

The design is otherwise fine to read one coefficient at a time: all eight VIFs
are below 4 (threshold 5) and the correlation matrix condition number is 29.7.
`src/diagnostics.py` computes both, and the tests assert the VIF bound holds.

## Caveats

- Coefficient **ordering and relative magnitude** is the result. Absolute
  values depend on feature standardisation and the ridge penalty.
- The solver needs ~7,250 iterations to converge on this panel. The default
  `--max-iter` of 1,000 stops short and the runner says so; don't read a
  truncated fit. (In this instance the truncated weights matched the converged
  ones to 3 decimal places, but that is luck, not a guarantee.)
- Recall@10 of 0.26 is a real improvement, not a solved problem. Three
  quarters of the top-10 destinations are still missed.

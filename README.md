# Occupational Mobility Model

Which job can you actually move to next, and how do we know?

This fits a **gravity model of occupational mobility**: O*NET occupational
features on one side, observed job-to-job moves from the CPS on the other, and
a learned weight on each O*NET domain in between. The output is a calibrated
answer to "given this occupation, where do people actually go" — validated on
held-out years, and benchmarked against the baselines it has to beat.

Every input comes from an API. There are no manual downloads and nothing to
re-do by hand when a new year of data lands.

> **Scope note.** This models *economy-wide* occupational mobility, not
> internal mobility within a single employer. There is no employer HR data
> here, and none is needed. See [Applying this inside a company](#applying-this-inside-a-company).

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env        # then fill in your two API keys

python src/fetch_data.py oews           # BLS employment + wages (~25s)
python src/fetch_data.py cps --submit   # queue the IPUMS extract
python src/fetch_data.py cps --status   # ...wait for it
python src/fetch_data.py cps --download

python src/run_phase1.py                # fit the model
python src/run_phase1.py --synthetic    # no keys needed; plumbing demo only
```

## Data sources

| Source | What it gives | Access |
|---|---|---|
| **O*NET 30.2** | 1,016 occupations × ~440 descriptors across 8 domains | Committed (CC-BY 4.0) |
| **BLS OEWS** | Employment and wages per occupation | API, free key |
| **IPUMS CPS ASEC** | Observed year-over-year occupation transitions, 2018–2024 | API, free key + CPS registration |

Get keys at [BLS](https://data.bls.gov/registrationEngine/) and
[IPUMS](https://account.ipums.org/api_keys). IPUMS additionally requires a
**per-collection registration for CPS** at
[uma.pop.umn.edu/cps/registration/new](https://uma.pop.umn.edu/cps/registration/new) —
the API key alone is not enough, and the error if you skip it is explicit.

### Two API traps worth knowing

**OEWS only carries the most recent year.** Passing `startyear`/`endyear` for
any earlier year returns `REQUEST_SUCCEEDED` with zero data points and a "No
Data Available" message — which reads exactly like a malformed series ID but
isn't. `src/sources/bls.py` never sends year parameters. Historical OEWS needs
the flat files at `download.bls.gov/pub/time.series/oe/`.

**OEWS series IDs are undocumented in the official API docs.** They're 25
characters of positional encoding; `src/sources/bls.py` documents the layout,
and the format is locked down by tests because a malformed ID fails silently.
Credit to [govex/bls-oews-api-tutorial](https://github.com/govex/bls-oews-api-tutorial)
for reverse-engineering it.

### A weight bug the API caught

ASEC samples must be weighted with `ASECWT`. The earlier notebook used
`WTFINL`, the basic monthly weight — and the IPUMS API rejects that variable
outright for ASEC samples: *"This variable is not available in any of the
samples currently selected."* Every weighted count produced from a hand-built
extract carrying `WTFINL` was weighted by a column that doesn't belong to the
supplement the transition signal lives in.

## The model

Occupational transitions have the same shape as trade or migration flows:
large destinations attract more movers, and "distance" suppresses flow. So the
specification is a gravity model,

```
E[moves i→j] = exp( origin_i
                    + b_size · log(employment_j)
                    + Σ_d  w_d · similarity_d(i, j)
                    + wage gap, job-zone gap, education gap,
                      same-industry, O*NET-related )
```

fit by **Poisson pseudo-maximum likelihood (PPML)**. Three structural pieces
do the real work:

| Piece | Why it's there |
|---|---|
| `origin_i` fixed effects | Absorb *how many* people leave occupation `i` at all, so the pair terms are identified purely off **where** leavers go. |
| `log(employment_j)` | The base-rate control. Most moves land in large occupations; without this the model rediscovers the size distribution of the labour market and reports it as insight. |
| PPML rather than log-OLS | ~430 reachable occupations give ~184k ordered pairs, and only a few thousand are ever observed. An estimator that can't take a zero is not usable here. |

`w_d` — the learned weight per O*NET domain — is the phase-1 finding. It
replaces the equal-weighted average in `similarity.py`, which assumed shared
*work values* predicts a career move as strongly as shared *skills*.

## Honest validation

Held out by **time**, never a random split of pairs — a random split leaks
`i→j` into training while testing on `j→i`. Default: fit 2018–2021, test
2022–2023.

Every run scores against three baselines, because a transition model can look
excellent and be worthless:

- **`size_only`** — destination employment and nothing else. The base-rate
  trap made explicit. It scores far better than intuition suggests, and any
  model that doesn't clearly beat it has discovered nothing.
- **`equal_similarity`** — the previous behaviour, eight domains at equal
  weight. The incumbent.
- **`onet_related`** — O*NET's own expert-curated related-occupation list. A
  human judgment of "could you move here", with no reference to whether
  anyone did.

Metrics: Poisson deviance (proper scoring rule for counts), Spearman on
observed flows, and **recall@10** — of the destinations people actually moved
to, how many appear in the model's top ten. That last one is what a career
tool is actually for.

## Phase 2: which model, and who's trapped

### Is the gravity model's structure earning its keep?

Five specifications, same held-out years, same metrics. The load-bearing one
is `flow_embedding`, which never sees an O*NET feature — if a model built
purely from who-moved-where predicts as well as one built from occupational
descriptors, the descriptors aren't doing the work.

| Model | What it is |
|---|---|
| `ppml_gravity` | Phase 1. Log-linear in features, Poisson, survey-weighted target. |
| `ppml_raw_counts` | Same spec on honest person counts (see the weighting note). |
| `boosted_poisson` | Same features, Poisson loss, no linearity assumption. |
| `flow_embedding` | SVD of the observed flow matrix. **No O*NET features at all.** |
| `gravity_plus_history` | Features *and* lagged flow. A forecast, not a structural model. |

```bash
python src/run_phase2.py --universe professional
python src/run_phase2.py --compare-universes
```

### The weighting problem

CPS gives each person a survey weight of ~1,000–3,000, so multiplying counts
by weights produces numbers that aren't counts: one surveyed mover becomes a
"count" of 2,000. The weighted target has a variance-to-mean ratio around
**39,000**; the underlying person counts, about **20**. Poisson assumes 1.

PPML point estimates survive this — that robustness is exactly why Santos
Silva & Tenreyro recommend it — which is why phase 1's rankings hold. Any
standard error from it does not. `ppml_raw_counts` exists to check whether
fitting honest counts changes the ranking; it doesn't materially.

### Structural vs. forecasting models

These answer different questions and shouldn't be read as competing on one
axis:

- **Features-only** (`ppml_gravity`) answers *what about two occupations makes
  movement between them likely*. It transfers to pairs never observed, which
  is precisely what the cold-start deployment story needs.
- **With history** (`gravity_plus_history`, `flow_embedding`) answers *what
  will flow next year*. Strictly better at forecasting, and useless for a pair
  with no history.

A company mapping its role catalogue onto this has no internal transition
history on day one. It gets the features-only number, not the forecasting one.

### Mobility deserts

An occupation is a desert when its predicted outflow concentrates on a
handful of destinations. Counting destinations doesn't capture that — the
model assigns *some* probability almost everywhere. Entropy does:

```
effective_destinations = exp(H(p))
```

where `p` is the predicted destination distribution. It reads directly: an
occupation with an effective count of 4 has four realistic ways out, however
many pairs are technically nonzero.

The quadrant that matters is **narrow options and low pay**. A well-paid
narrow occupation is a specialty; a poorly-paid narrow one is a trap. That
intersection is what del Rio-Chanona et al. found drives long-term
unemployment after an automation shock.

### Scoping the universe

`--universe` takes `all`, `white-collar` (SOC major groups 11–29 plus 41/43),
or `professional` (11–29 only). Definitions are in `src/soc.py` — "white
collar" has no official definition, so it's stated rather than assumed.

Scoping to white collar makes the matrix denser and the model more accurate.
It also **removes most of the mobility deserts**, which sit overwhelmingly in
the occupations it excludes. Use the narrow universe for a career tool; use
the full one for the research finding. `--compare-universes` reports both.

## Layout

```
src/
  sources/
    bls.py         OEWS employment + wages over the BLS API
    ipums.py       CPS ASEC extracts defined in code and submitted over the API
  soc.py           SOC plumbing: O*NET 10-digit ↔ SOC-6 ↔ Census OCC
  pairs.py         Pair features: per-domain similarity, signed gaps, wage gap
  transitions.py   IPUMS extract → observed weighted moves
  gravity.py       The PPML gravity model
  evaluate.py      Baselines and held-out scoring
  similarity.py    Original equal-weighted similarity (kept; now the baseline)
  models.py        Alternative specifications for the comparison
  deserts.py       Mobility-desert metrics
  diagnostics.py   Collinearity checks on the learned weights
  fetch_data.py    Pull every input from its API
  run_phase1.py    Fit and report
  run_phase2.py    Model comparison + mobility deserts
tests/
  synthetic.py     Flow generator with a known ground truth — tests only
```

The estimator is tested by generating flows from **known** domain weights and
checking it recovers their ordering, at the sparsity a real CPS extract
produces. Testing against real data could only ever confirm the code runs.

```bash
pytest -q
```

## Reading the weights

Coefficients are fit on standardised features, so they're directly comparable
to each other — that comparison is the entire point. Treat the **ordering and
relative magnitude** as the result, not the absolute values, which depend on
feature scaling and the ridge penalty.

If the solver stops at its iteration cap the runner says so loudly, and the
coefficients should not be read as a finding.

## Applying this inside a company

Every internal talent marketplace has a cold-start problem: a company that
hasn't tracked internal moves has no transition history to train on, so a
model has nothing to learn from on day one.

A model calibrated on national CPS data and transferred onto a company's role
catalogue is a legitimate answer — map internal job titles to SOC codes,
inherit the national transition priors, and refine them as internal moves
accumulate. That's the deployment path, and it's why the national calibration
is worth doing first.

## Limitations

- **CPS occupation coding is noisy.** Year-over-year occupation is
  self-reported and re-coded, so some "transitions" are coding artefacts. This
  is a known issue in the literature and sets a floor on achievable accuracy.
- **Coverage loss in the crosswalk.** ~430 of 530 Census occupation codes map
  cleanly to a SOC-6 present in O*NET; the rest are aggregate categories with
  no single counterpart and are dropped. OEWS covers 421 of those 430.
- **Aggregate, not individual.** This models flows between occupations, not
  any individual's probability of moving.
- **Observed ≠ advisable.** The model learns where people *did* go, which
  encodes existing labour market frictions and inequities. A well-travelled
  route is not automatically a good one.

## Prior art

The occupational-mobility-network framing follows
[del Rio-Chanona et al., *Occupational mobility and automation*](https://royalsocietypublishing.org/rsif/article/18/174/20200898/89844/Occupational-mobility-and-automation-a-data-driven)
([preprint](https://arxiv.org/abs/1906.04086)), which builds mobility networks
from CPS transitions and shows that network structure — not just automation
exposure — determines who gets stranded. PPML as the estimator for gravity
models with many zeros follows Santos Silva & Tenreyro (2006).

# Occupational Mobility Model

A gravity model of job-to-job moves. O*NET occupational features on one side,
observed CPS transitions on the other, and a learned weight per O*NET domain
in between. The output answers "given this occupation, where do people
actually go", validated on held-out years.

Every input is fetched by script, so a new release of any source is a one-line
refresh.

This models economy-wide occupational mobility, not internal mobility within a
single employer. There is no employer HR data here.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate   # Python 3.9+
pip install -r requirements.txt
cp .env.example .env        # then fill in three free API keys

python src/fetch_data.py onet           # version check
python src/fetch_data.py oews           # BLS employment + wages
python src/fetch_data.py cps --submit   # queue the IPUMS extract
python src/fetch_data.py cps --download # once it's ready

python src/run_phase1.py                # fit
python src/run_phase1.py --synthetic    # no keys needed, plumbing only
```

## Data

- **O*NET 31.0** — 1,016 occupations across 7 descriptor domains. The feature
  matrix comes from the bulk download; the API is used for version checks and
  single-occupation lookups.
- **BLS OEWS** — employment and wages per occupation (421 of 430 covered).
- **IPUMS CPS ASEC** — observed year-over-year occupation transitions,
  2018–2024.

Keys are free from [O*NET](https://services.onetcenter.org/developer/signup),
[BLS](https://data.bls.gov/registrationEngine/) and
[IPUMS](https://account.ipums.org/api_keys). IPUMS also needs a separate
[CPS collection registration](https://uma.pop.umn.edu/cps/registration/new) —
the API key alone isn't enough.

## The model

Occupational moves have the same shape as trade or migration flows: large
destinations attract more movers, and "distance" suppresses flow.

```
E[moves i→j] = exp( origin_i
                    + b · log(employment_j)
                    + Σ_d w_d · similarity_d(i, j)
                    + wage gap, job-zone gap, education gap, ... )
```

Fit by Poisson pseudo-maximum likelihood. Around 430 reachable occupations
gives ~184k ordered pairs of which only about 3% are ever observed, so an
estimator that handles zeros is required — log-OLS isn't usable here.

Origin fixed effects absorb how many people leave an occupation at all, which
means the pair terms are identified off *where* leavers go rather than how
many. `log(employment_j)` is the base-rate control; without it the model
rediscovers the size distribution of the labour market and reports it as a
finding.

## Results

Fit on 2018–21, tested on held-out 2022–24, against three baselines:
destination size alone, equal-weighted similarity (the previous approach), and
O*NET's own curated related-occupation list.

| Model | recall@10 | Spearman |
|---|---|---|
| `size_only` | 0.087 | 0.183 |
| `equal_similarity` | 0.121 | 0.219 |
| `onet_related` | 0.200 | 0.237 |
| **`learned_gravity`** | **0.262** | **0.323** |

recall@10 is the share of actual destinations that appear in the model's top
ten, out of 430. So roughly a quarter, against about 2% for a random guess.

The learned weights put skill first, then work context, interest and
knowledge, with work activity a distant last. Equal weighting was wrong: the
positive weights span nearly 4×. Ability comes out negative, which is
collinearity with skill and work context rather than a real effect.
`diagnostics.py` computes the VIFs that establish it.

Phase 2 compares five specifications, including a flow-embedding baseline that
sees no O*NET features at all. It predicts about as well, which means most of
the signal lives in the transition graph rather than in what the jobs involve.
The features earn their keep where history is absent, which is the cold-start
case. Numbers in [`reports/`](reports/).

## Caveats

- CPS occupation coding is self-reported and re-coded year to year, so some
  "transitions" are coding artefacts. This sets a floor on achievable accuracy.
- About 430 of 530 Census occupation codes map cleanly to a SOC-6 in O*NET;
  the rest are aggregate categories and get dropped.
- This models flows between occupations, not any individual's probability of
  moving.
- It learns where people *did* go, which encodes existing labour market
  frictions.

## Tests

```bash
pytest -q     # 99 tests
```

The estimator is tested against synthetic flows generated from known domain
weights, checking it recovers their ordering at realistic sparsity. Testing
against real data could only confirm the code runs.

## Prior art

The mobility-network framing follows
[del Rio-Chanona et al.](https://arxiv.org/abs/1906.04086), which builds
mobility networks from CPS transitions and shows network structure determines
who gets stranded. PPML for gravity models with many zeros follows Santos
Silva & Tenreyro (2006).

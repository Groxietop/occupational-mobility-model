# Occupational Mobility Model

Which job can you actually move to next, and how do we know?

This fits a **gravity model of occupational mobility**: O*NET occupational
features on one side, six years of observed job-to-job moves from the CPS on
the other, and a learned weight on each O*NET domain in between. The output is
a calibrated answer to "given this occupation, where do people actually go" —
validated on held-out years, and benchmarked against the baselines it has to
beat.

> **Scope note.** This models *economy-wide* occupational mobility, not
> internal mobility within a single employer. The distinction matters: there is
> no employer HR data here, and none is needed. See
> [Applying this inside a company](#applying-this-inside-a-company).

## What's new in phase 1

The original analysis computed cosine similarity between occupations by
averaging eight O*NET domains **with equal weight**, then plotted that
similarity against observed transitions and stopped.

Equal weighting is an assumption, and it's testable. There's no reason to
believe shared *work values* predicts a real career move as strongly as shared
*skills* — and six years of CPS data can say which actually does. Phase 1
replaces the assumption with a fit.

```bash
pip install -r requirements.txt

python src/run_phase1.py --synthetic   # no data download needed; plumbing demo
python src/run_phase1.py               # the real fit; needs the IPUMS extract
```

## The model

Occupational transitions have the same shape as trade or migration flows:
large destinations attract more movers, and "distance" suppresses flow. So the
specification is a gravity model,

```
E[moves i→j] = exp( origin_i
                    + b_size · log(employment_j)
                    + Σ_d  w_d · similarity_d(i, j)
                    + job-zone gap, education gap, same-industry, O*NET-related )
```

fit by **Poisson pseudo-maximum likelihood (PPML)**. Two structural pieces do
the real work:

| Piece | Why it's there |
|---|---|
| `origin_i` fixed effects | Absorb *how many* people leave occupation `i` at all, so the pair terms are identified purely off **where** leavers go. |
| `log(employment_j)` | The base-rate control. Most moves land in large occupations; without this the model rediscovers the size distribution of the labour market and reports it as insight. |
| PPML rather than log-OLS | With ~430 reachable occupations there are ~184k ordered pairs and only a few thousand are ever observed. An estimator that can't take a zero is not usable here. |

`w_d` — the learned weight per O*NET domain — is the phase-1 finding.

## Honest validation

Held out by **time**, never by a random split of pairs. Splitting pairs at
random leaks `i→j` into training while testing on `j→i`. Default is fit on
2018–2021, test on 2022–2023.

Every run scores the model against three baselines, because a transition model
can look excellent and be worthless:

- **`size_only`** — predict flow proportional to destination employment and
  nothing else. This is the base-rate trap made explicit. It scores far better
  than intuition suggests. Any model that doesn't clearly beat it has
  discovered nothing.
- **`equal_similarity`** — the previous behaviour: eight domains, equal weight.
  The incumbent this phase is trying to improve on.
- **`onet_related`** — O*NET's own expert-curated related-occupation list. A
  human judgment of "could you move here", with no reference to whether anyone
  did.

Metrics: Poisson deviance (proper scoring rule for counts), Spearman
correlation on observed flows, and **recall@10** — of the destinations people
actually moved to from an occupation, how many appear in the model's top ten.
Recall@10 is the one that maps to what a career tool is for.

## Getting the data

**O*NET** is already committed (`data/raw/onet_30_2/`, CC-BY 4.0).

**CPS transitions** require a free IPUMS extract — full download steps are at
the top of `notebooks/03_ipums_transitions.ipynb`. Place the result at
`data/raw/ipums_cps_asec.csv.gz`. Until then, `--synthetic` exercises the whole
pipeline against flows generated from a known process. That verifies the
plumbing and nothing else; it is not a result, and the runner says so.

Everything the model needs beyond O*NET comes out of that one extract —
including destination employment size, which is computed from the CPS weights
rather than requiring a separate BLS download.

## Layout

```
src/
  soc.py           SOC code plumbing: O*NET 10-digit ↔ SOC-6 ↔ Census OCC
  pairs.py         Pair-level features: per-domain similarity + signed gaps
  transitions.py   IPUMS extract → observed weighted moves (+ destination size)
  gravity.py       The PPML gravity model and its fitted-model object
  evaluate.py      Baselines and held-out scoring
  similarity.py    Original equal-weighted similarity (kept; now the baseline)
  run_phase1.py    End-to-end runner
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
the feature scaling and the ridge penalty.

If the solver stops at its iteration cap the runner says so loudly and the
coefficients should not be read as a finding.

## Applying this inside a company

Every internal talent marketplace has a cold-start problem: a company that
hasn't tracked internal moves has no transition history to train on, so a
model has nothing to learn from on day one.

A model calibrated on national CPS data and transferred onto a company's role
catalogue is a legitimate answer — map internal job titles to SOC codes,
inherit the national transition priors, and refine them as internal moves
accumulate. That's the deployment path, and it's the reason the national
calibration is worth doing first.

## Limitations

- **CPS occupation coding is noisy.** Year-over-year occupation is
  self-reported and re-coded, so some "transitions" are coding artefacts
  rather than real moves. This is a known issue in the literature and sets a
  floor on achievable accuracy.
- **Coverage loss in the crosswalk.** ~430 of 530 Census occupation codes map
  cleanly to a SOC-6 present in O*NET. The rest are aggregate Census
  categories with no single O*NET counterpart, and they're dropped.
- **Aggregate, not individual.** This models flows between occupations, not
  any individual's probability of moving. It says nothing about a specific
  person's prospects.
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

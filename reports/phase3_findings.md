# Phase 3 — geographic mobility deserts

Everything below is from `python src/run_phase3.py --skip-fetch`, fit on
2018–2021 and tested on 2022–2024, 430 occupations, 164,736 test pairs. Metro
data is OEWS May 2025 across 17 metropolitan areas.

**Headline: the concept survives, the current measurement does not.** Phase 3
was meant to turn the project's public question from "here is a gravity model"
into "where are the mobility deserts." The machinery now exists end to end. But
validating it surfaced two problems serious enough that no desert ranking —
national or geographic — should be published as a finding yet.

---

## 1. The desert score was not identified

Phase 2 computed deserts from `gravity_plus_history`, the best-predicting
model. The open concern was that this conflates two different things: an
occupation narrow because of its skill profile, and one narrow because it
always has been. Phase 3 computed deserts under both readings to separate them.

They do not agree. At all.

| | median effective destinations | Spearman vs. the other |
|---|---|---|
| `ppml_gravity` (features only) | 97.5 | — |
| `gravity_plus_history` | 7.8 | **−0.106** |

A rank correlation of −0.11 between two measures of the same quantity means
"mobility desert" was not a well-defined property of an occupation. It was a
property of the model chosen to measure it.

### Which one is right?

Observed CPS transitions settle it. Destination entropy computed directly from
the test-year flows is model-free, so it is the anchor. Restricted to the 285
occupations with at least five observed destination pairs:

| model | Spearman vs. observed destination entropy |
|---|---|
| `ppml_gravity` (features only) | **−0.11** |
| `gravity_plus_history` | **+0.69** |

**The features-only model cannot measure narrowness.** Its entropy is high for
occupations with generic O\*NET profiles and low for distinctive ones, which is
a different quantity entirely and is roughly orthogonal to how concentrated
real moves are.

So the original worry inverts. The fear was that the history-inclusive model
was contaminated; in fact it is the only one tracking reality, and the
"structural" alternative is not a more honest reading, it is a broken one.

### What this costs

The claim that mobility deserts are skill-structural is not supported. The
defensible statement is narrower and should be worded as such:

> Mobility deserts are occupations whose **observed** transitions are
> concentrated on few destinations, few of which pay more.

That is a fact about how the labour market has behaved, not about what a
worker's skills permit. Whether the cause is skill structure, credentialing,
information, or discrimination is not identified here, and the phrase
"structural" should not appear in public material until it is.

A second casualty: of the 21 occupations that rank as deserts under both
readings, **20 are well-paid licensed professions** — nurse anaesthetists,
pharmacists, occupational therapists, physician assistants. Those are
specialties, not traps. Only Refuse and Recyclable Material Collectors
($49,690) is both narrow and low-paid. Phase 2's finding that deserts are
overwhelmingly low-paid production and admin work comes from the history model
alone and does not survive being cross-checked.

---

## 2. The geographic layer is confounded by disclosure

The geographic question is the good one: if you are in occupation X in metro Y,
how many upward exits exist within reach? The implementation reweights the
national destination mix by local availability:

    availability(D, M) = employment(D, M) / employment(D, national)
    p_local(D | X, M)  ∝  p(D | X) · availability(D, M)

with raises judged on the metro's own wages. Mechanically it works, and it
produces exactly the artefact the reframe wanted — one occupation, every metro,
sorted worst to best.

It is also, on the raw panel, substantially a measure of BLS disclosure policy.

OEWS suppresses small cells, and small metros have more of them: this panel
publishes 246 occupations in Youngstown and 406 in New York. Because the
localised mix is renormalised over whatever is published, a metro with more
suppressed cells mechanically scores as having fewer exits.

| panel | corr(occupations published, mean viable upward exits) |
|---|---|
| raw, localising `ppml_gravity` | **+0.94** |
| raw, localising `gravity_plus_history` | **+0.36** |
| balanced (217 occupations in all 17 metros) | undefined — no variance by construction |

At +0.94 the metro ranking was not a finding about mobility.

### Balancing removes the signal too

Restricting to the 217 occupations every metro publishes eliminates the
confound. It also deletes the occupations that carry the geographic
information:

| | count | median national employment |
|---|---|---|
| kept (in all 17 metros) | 217 | 220,660 |
| dropped | 202 | 26,720 |

The dropped occupations are present in between 1 and 16 of the 17 metros —
which is precisely what geographic concentration looks like. **An occupation
absent from Youngstown's OEWS tables is both "suppressed" and "genuinely
unavailable", and OEWS cannot distinguish these.** Suppression and signal are
partly the same measurement.

On the balanced panel the metro index collapses to a 4.3–6.3 range with no
interpretable ordering — San Jose ranks worst, Atlanta best, and the manufacturing
metros sit mid-table. That is not a credible geography of opportunity; it is
what remains after the geography has been balanced away.

### Fragility

The same occupation, the same code, different national model:

| Machinists (51-4041), El Paso | viable upward exits |
|---|---|
| localising `ppml_gravity` | 8.80 |
| localising `gravity_plus_history` | 0.08 |

A hundredfold swing from a model choice means `viable_upward_exits =
effective_destinations × upward_share` is not yet a stable quantity. The
product compounds two model-sensitive terms.

---

## What is worth keeping

The machinery, and the negative results, which are real findings:

- `src/sources/oews_metro.py` — occupation × metro OEWS, incrementally cached.
  17 metros verified. Note CBSA 17460 (Cleveland-Elyria) returns no OEWS data;
  the live code is 17410, and a wrong area code returns success with zero rows
  rather than an error, so `verify_areas()` is not optional.
- `src/geography.py` — localisation, the balanced-panel correction, and the
  suppression diagnostic.
- The finding that features-only gravity cannot measure narrowness. That is
  worth stating publicly on its own; it is a genuine caution about a natural
  and appealing approach.

## What would make the geographic claim publishable

1. **Use a source without disclosure suppression.** Census QWI (LEHD) is an
   administrative near-universe at county and metro level, so an absent
   occupation is absent rather than undisclosed. Its limitation is that it is
   industry-coded, not occupation-coded, so this is not a drop-in replacement —
   ACS PUMS, which is occupation-coded microdata at PUMA level, is the more
   direct substitute despite being self-reported.
2. **Replace metro boundaries with commuting zones.** Metro areas are a crude
   proxy for reachable-without-moving, and the whole claim is about reach.
3. **Validate against something external.** A desert score that predicts
   nothing observable is a description, not a finding. Long-term unemployment
   duration or realised wage growth by occupation and place would test it.
4. **Report a single, stable quantity.** `upward_share` alone is better
   behaved than the product; the multiplication by effective destinations is
   where the fragility enters.

## Reproducing

```bash
python src/run_phase3.py                       # fetches metro data (~289 requests)
python src/run_phase3.py --skip-fetch          # uses the cache
python src/run_phase3.py --skip-fetch --occupation 51-4041
```

Outputs: `deserts_both_models.csv`, `geographic_deserts.csv`,
`metro_mobility_index.csv` in `data/processed/`.

Unrelated fix made along the way: `sources/bls.py` checked for an API key
before checking its cache, so cached re-runs failed without a key. Cache is now
checked first.

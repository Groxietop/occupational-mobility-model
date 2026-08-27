# Refreshing O*NET 30.2 → 31.0

The version check added alongside the O*NET API said the committed database
was two releases behind. This is what happened on refresh.

## It was not a version bump

O*NET 31.0 reorganised the content model. Fifteen files disappeared and
twenty appeared:

| 30.2 | 31.0 | Effect |
|---|---|---|
| `Skills.txt` | `Essential Skills.txt` + `Transferable Skills.txt` | Same **35 elements**, split by category — reconstructible exactly |
| `Interests.txt` | `Career Interest Types.txt` | Same **9 elements** |
| `Work Values.txt` | — | **Retired.** Zero rows in 31.0's Content Model Reference |
| `Technology Skills.txt` | `Software Skills.txt` | Category column renamed |
| `Education, Training, and Experience.txt` | `Education.txt` + `Training and Experience.txt` | Split |

Occupation set is unchanged: 1,016 in both, none added, none retired.

So the feature matrix goes from **8 domains to 7**. Work values is not
relocated or renamed — O*NET removed it.

## The builder came first

Nothing in the repo built `onet_master.parquet`. Every module read it; the
file was the only record of how it had been made. A refresh was impossible
without reverse-engineering the artifact.

`src/build_master.py` now does it, and the check that made it trustworthy is
that it **reproduces the existing 30.2 master exactly** — 1,016 × 445, same
occupation set, all 438 numeric columns matching to within 0.005. A builder
that can't reproduce the version it was derived from shouldn't be believed
about a new one.

## What changed in the learned weights

Three runs on identical CPS data (2018–21 train, 2022–24 test). The middle
column is a control: 30.2 with work values deleted, isolating "domain
removed" from "data re-rated".

| Domain | 30.2 (8 domains) | 30.2 control (7) | 31.0 (7) |
|---|---|---|---|
| interest | **0.413** (1st) | **0.402** (1st) | 0.345 (2nd) |
| skill | 0.299 (3rd) | 0.355 (2nd) | **0.386 (1st)** |
| knowledge | 0.319 (2nd) | 0.316 (3rd) | 0.310 (4th) |
| work_context | 0.248 (4th) | 0.243 (5th) | **0.339 (3rd)** |
| work_style | 0.247 (5th) | 0.285 (4th) | 0.228 (5th) |
| work_activity | 0.088 (7th) | 0.102 (6th) | 0.105 (6th) |
| ability | −0.136 | −0.134 | −0.192 |
| work_value | 0.133 (6th) | — | — |

The control does real work here. Two changes look similar and aren't:

**Skill overtaking interest is a genuine 31.0 effect.** Deleting work values
from 30.2 raises skill (0.299 → 0.355) as its weight redistributes, but
interest still ranks first. Only with 31.0's re-rated values does skill
actually take the top spot.

**Work context's jump is entirely 31.0.** The control barely moves it
(0.248 → 0.243); 31.0 lifts it to 0.339, from 4th to 3rd.

**Ability stays negative in all three runs.** That was the suppression
effect phase 1 flagged — negative jointly, +0.88 alone, because it
correlates 0.74 with skill. It surviving a data refresh and a domain removal
says it's structural, not an artifact of one release.

## The model got slightly better

Held out on 2022–24:

| | 30.2 | 31.0 |
|---|---|---|
| Poisson deviance ↓ | 822.1 | **805.3** |
| Spearman ↑ | 0.315 | **0.324** |
| recall@10 ↑ | 0.261 | **0.264** |

Small but consistent, on one fewer feature domain.

The `onet_related` baseline improved considerably more — recall@10 **0.162 →
0.200**. O*NET's own curated relatedness got materially better in 31.0, which
raises the bar the learned model has to clear.

## What is committed

The rebuilt master (1.2 MB parquet). Not the raw dump — ~99 MB unpacked, and
now reproducible:

```bash
python src/fetch_data.py onet --download 31.0
python src/build_master.py --raw data/raw/onet_31_0/db_31_0_text \
  --out data/processed/onet_master.parquet
```

## Caveats

- The 30.2 → 31.0 comparison is not a controlled experiment on the same
  data. O*NET re-rated occupations between releases, and the control isolates
  domain removal but not re-rating from anything else that changed.
- Two tests hardcoded "8 domains" and failed on a valid refresh. They now
  derive the expected set from the master. Worth watching for other places
  that assume a fixed feature space.

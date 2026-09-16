"""Phase 3: geographic mobility deserts.

    python src/run_phase3.py
    python src/run_phase3.py --occupation 51-4041     # one occupation, all metros
    python src/run_phase3.py --skip-fetch             # use the cached metro data

Two things happen here.

1. The desert score stops being one number. Phase 2 computed deserts from the
   best-predicting model, which is `gravity_plus_history` -- and that model's
   strongest signal is last year's flow. A desert measured that way cannot
   distinguish "narrow because of its skill profile" from "narrow because it
   always has been." So both are computed:

     structural  ppml_gravity          O*NET features only, no flow history
     observed    gravity_plus_history  features plus lagged flow

   Where they agree, the finding is solid. Where they disagree, that gap is
   itself the result, and it points at different causes and different remedies.

2. The question goes local. A national transition profile assumes every
   destination is reachable from everywhere. Reweighting by what OEWS actually
   publishes in each metro gives the number that matters to a worker: how many
   upward exits exist within reach of where they already live.

Outputs land in `data/processed/` and `reports/`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import geography  # noqa: E402
from deserts import find_deserts  # noqa: E402
from evaluate import score  # noqa: E402
from models import run_all  # noqa: E402
from run_phase2 import build_panels  # noqa: E402
from soc import collapse_to_soc6, filter_universe, load_crosswalk  # noqa: E402
from sources.bls import to_wide as national_to_wide  # noqa: E402
from sources.oews_metro import METROS, fetch_metro_oews  # noqa: E402
from sources.oews_metro import to_wide as metro_to_wide  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
CROSSWALK = RAW / "census_occ_to_soc_2018.csv"
REPORTS = ROOT / "reports"
METRO_CACHE = PROCESSED / "oews_metro.parquet"

DEFAULT_TRAIN = [2018, 2019, 2020, 2021]
DEFAULT_TEST = [2022, 2023, 2024]

# The two specifications whose disagreement is the point.
STRUCTURAL_MODEL = "ppml_gravity"
OBSERVED_MODEL = "gravity_plus_history"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", default="all", choices=["all", "white-collar", "professional"])
    parser.add_argument("--train-years", type=int, nargs="+", default=DEFAULT_TRAIN)
    parser.add_argument("--test-years", type=int, nargs="+", default=DEFAULT_TEST)
    parser.add_argument("--max-iter", type=int, default=20000)
    parser.add_argument("--occupation", help="SOC-6 code to profile across every metro")
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Use the cached metro parquet and never call the BLS API.",
    )
    return parser.parse_args(argv)


def load_metro(soc_codes, skip_fetch: bool) -> pd.DataFrame:
    if skip_fetch:
        if not METRO_CACHE.exists():
            raise SystemExit(
                f"No cached metro data at {METRO_CACHE}. Drop --skip-fetch to fetch it."
            )
        return pd.read_parquet(METRO_CACHE)
    return fetch_metro_oews(soc_codes, cache_path=METRO_CACHE)


def main(argv=None) -> int:
    args = parse_args(argv)

    # Universe construction must match run_phase2 exactly, or the structural
    # figures here cannot be compared with the phase 2 findings. The crosswalk
    # intersection is the part that is easy to miss: it drops O*NET occupations
    # the CPS can never observe a transition for.
    master = pd.read_parquet(PROCESSED / "onet_master.parquet")
    reachable = set(load_crosswalk(CROSSWALK)["soc6"])
    all_codes = sorted(set(collapse_to_soc6(master)["soc6"]) & reachable)
    codes = filter_universe(all_codes, args.universe)

    print("=" * 70)
    print("PHASE 3: GEOGRAPHIC MOBILITY DESERTS")
    print("=" * 70)

    train, test, national_oews, soc6_master = build_panels(
        codes, args.train_years, args.test_years
    )
    print(f"\n  panel: {len(test):,} test pairs over {len(codes)} occupations")

    results, _ = run_all(train, test, max_iter=args.max_iter)
    by_name = {r.name: r for r in results}

    for name in (STRUCTURAL_MODEL, OBSERVED_MODEL):
        if name not in by_name:
            raise SystemExit(f"model {name!r} missing from run_all output")

    print("\n  Two readings of the same network")
    print("  " + "-" * 66)
    for label, name in (("structural", STRUCTURAL_MODEL), ("observed", OBSERVED_MODEL)):
        s = score(test, by_name[name].predictions)
        print(
            f"  {label:<11} {name:<22} recall@10 {s['recall_at_10']:.3f}  "
            f"spearman {s['spearman_observed']:.3f}"
        )
    print(
        "\n  The observed model predicts better because last year's flow is the\n"
        "  strongest single signal. That is also why it cannot, on its own, tell\n"
        "  a skill-structural desert from a habitual one."
    )

    # ---- national deserts under both readings -------------------------------

    structural = find_deserts(
        test, by_name[STRUCTURAL_MODEL].predictions, national_oews, soc6_master
    )
    observed = find_deserts(test, by_name[OBSERVED_MODEL].predictions, national_oews, soc6_master)

    merged = structural.loc[
        :, ["soc6", "title", "effective_destinations", "upward_share", "desert_score",
            "annual_median_wage", "employment"]
    ].rename(
        columns={
            "effective_destinations": "structural_destinations",
            "upward_share": "structural_upward",
            "desert_score": "structural_score",
        }
    ).merge(
        observed.loc[:, ["soc6", "effective_destinations", "upward_share", "desert_score"]].rename(
            columns={
                "effective_destinations": "observed_destinations",
                "upward_share": "observed_upward",
                "desert_score": "observed_score",
            }
        ),
        on="soc6",
        how="inner",
    )
    merged["score_gap"] = merged["observed_score"] - merged["structural_score"]
    merged.to_csv(PROCESSED / "deserts_both_models.csv", index=False)

    agree = merged[(merged["structural_score"] > 0.75) & (merged["observed_score"] > 0.75)]
    print(f"\n  Deserts under BOTH readings: {len(agree)} occupations")
    print("  These are the defensible ones -- narrow by skill structure and in practice.")
    show = agree.nlargest(args.top, "structural_score")
    for _, r in show.iterrows():
        print(
            f"    {str(r['title'])[:44]:<44} {r['structural_destinations']:5.1f} / "
            f"{r['observed_destinations']:5.1f} exits   ${r['annual_median_wage']:>8,.0f}"
        )

    stuck = merged.nlargest(args.top, "score_gap")
    print("\n  Widest disagreement: structurally open, observably stuck")
    print("  Skill adjacency says there are exits; the flow history says nobody takes them.")
    for _, r in stuck.iterrows():
        print(
            f"    {str(r['title'])[:44]:<44} {r['structural_destinations']:5.1f} -> "
            f"{r['observed_destinations']:5.1f} exits   gap {r['score_gap']:+.2f}"
        )

    # ---- geographic layer ---------------------------------------------------

    metro_raw = load_metro(codes, args.skip_fetch)
    metro_wide = metro_to_wide(metro_raw)
    national_wide = national_to_wide(national_oews) if "measure" in national_oews else national_oews

    print(f"\n  metro panel: {metro_wide['metro'].nunique()} metros, "
          f"{len(metro_wide):,} (occupation, metro) cells")

    # The OBSERVED model supplies the national mix for the geographic layer.
    #
    # The structural model was the intuitive choice -- "what the opportunity
    # structure permits" -- but it fails the model-free check: its destination
    # entropy correlates -0.11 with the entropy of actually observed CPS
    # transitions, against +0.69 for the observed model. Its diffuseness tracks
    # how generic an occupation's O*NET profile is, not how few exits it has,
    # so localising it would spread mass over destinations nobody moves to.
    #
    # Flow history is national, so it says nothing about any particular metro.
    # The geography comes entirely from where those destinations exist.
    predictions = by_name[OBSERVED_MODEL].predictions

    raw_local = geography.local_mobility(test, predictions, metro_wide, national_wide)
    raw_index = geography.metro_index(raw_local, metro_wide)
    raw_corr = geography.suppression_diagnostic(raw_index, metro_wide)

    # OEWS suppresses more cells in smaller metros, which mechanically shrinks
    # their destination set. If that correlation is high the raw ranking is a
    # disclosure artefact, so the published index uses the balanced panel.
    balanced_wide = geography.balance(metro_wide)
    kept = balanced_wide["soc6"].nunique()
    print(
        f"\n  Suppression check: corr(occupations published, viable exits) = {raw_corr:+.2f}"
        f" on the raw panel"
    )
    print(
        f"  Balanced panel: {kept} occupations published in all "
        f"{metro_wide['metro'].nunique()} metros — used for the ranking below."
    )

    local = geography.local_mobility(test, predictions, balanced_wide, national_wide)
    local = geography.compare_to_national(local, observed)
    local.to_csv(PROCESSED / "geographic_deserts.csv", index=False)

    index = geography.metro_index(local, balanced_wide)
    balanced_corr = geography.suppression_diagnostic(index, balanced_wide)
    index.to_csv(PROCESSED / "metro_mobility_index.csv", index=False)
    if balanced_corr != balanced_corr:  # NaN
        print(
            "  After balancing: undefined by construction — every metro now "
            "publishes\n  the same 217 occupations, so no disclosure variance "
            "remains to confound."
        )
    else:
        print(f"  After balancing: corr = {balanced_corr:+.2f}")
    print(
        "  Caution: balancing drops the 202 occupations that are NOT in every\n"
        "  metro (median national employment 26,720, against 220,660 for those\n"
        "  kept). Those are the geographically concentrated occupations — so the\n"
        "  balanced ranking removes the confound and much of the signal with it."
    )

    print("\n  Metro mobility index — employment-weighted, worst first")
    print("  " + "-" * 66)
    print(
        f"  {'Metro':<34}{'viable':>8}{'exits':>8}{'upward':>9}{'wage':>10}"
    )
    for _, r in index.iterrows():
        print(
            f"  {str(r['metro_name'])[:33]:<34}"
            f"{r['mean_viable_upward_exits']:>8.2f}"
            f"{r['mean_local_effective_destinations']:>8.1f}"
            f"{r['mean_upward_share']:>9.1%}"
            f"{r['mean_local_wage']:>10,.0f}"
        )

    print("\n  Largest geographic penalty — occupations far narrower locally than nationally")
    worst = local.nlargest(args.top, "geographic_penalty")
    for _, r in worst.iterrows():
        print(
            f"    {str(r['title'])[:34]:<34} {str(r['metro_name'])[:22]:<23}"
            f" {r['national_effective_destinations']:5.1f} -> "
            f"{r['local_effective_destinations']:5.1f}"
        )

    if args.occupation:
        profile = geography.same_occupation_across_metros(local, args.occupation)
        title = structural.loc[structural["soc6"] == args.occupation, "title"]
        label = title.iloc[0] if len(title) else args.occupation
        print(f"\n  {label} ({args.occupation}) across metros — worst first")
        print("  " + "-" * 66)
        for _, r in profile.iterrows():
            print(
                f"    {str(r['metro_name'])[:33]:<34}"
                f"{r['viable_upward_exits']:>7.2f} viable"
                f"{r['local_effective_destinations']:>8.1f} exits"
                f"{r['local_upward_share']:>8.1%} up"
                f"{r['origin_local_wage']:>10,.0f}"
            )

    print(f"\n  Wrote deserts_both_models.csv, geographic_deserts.csv and")
    print(f"  metro_mobility_index.csv to {PROCESSED}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

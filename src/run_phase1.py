"""Phase 1 end to end: fit the gravity model and report the learned weights.

    python src/run_phase1.py                      # needs the IPUMS extract
    python src/run_phase1.py --synthetic          # no extract needed, demo only

The real run needs `data/raw/ipums_cps_asec.csv.gz` -- see the download
instructions at the top of `notebooks/03_ipums_transitions.ipynb`. The
`--synthetic` mode generates flows from a known process instead, which is
useful for checking the plumbing but tells you nothing about the labour
market.

Outputs land in `data/processed/`:

    domain_weights.csv        the learned weight per O*NET domain
    model_scorecard.csv       held-out comparison against the baselines
    gravity_coefficients.csv  every pair-level coefficient
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import compare  # noqa: E402
from gravity import build_panel, fit_gravity, predict  # noqa: E402
from pairs import add_wage_features, build_pair_features, related_pairs  # noqa: E402
from soc import collapse_to_soc6, load_crosswalk  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

# Located at run time: an API-downloaded extract (cps_00001.csv.gz) if one
# exists, else the legacy hand-built path.
LEGACY_IPUMS_FILE = RAW / "ipums_cps_asec.csv.gz"
CROSSWALK = RAW / "census_occ_to_soc_2018.csv"
MASTER = PROCESSED / "onet_master.parquet"

DEFAULT_TRAIN = [2018, 2019, 2020, 2021]
DEFAULT_TEST = [2022, 2023]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate flows from a known process instead of reading IPUMS.",
    )
    parser.add_argument(
        "--train-years", type=int, nargs="+", default=DEFAULT_TRAIN,
        help="Years to fit on.",
    )
    parser.add_argument(
        "--test-years", type=int, nargs="+", default=DEFAULT_TEST,
        help="Years to hold out. Must not overlap --train-years.",
    )
    parser.add_argument("--alpha", type=float, default=1e-4, help="L2 strength.")
    parser.add_argument("--top-k", type=int, default=10, help="k for recall@k.")
    parser.add_argument(
        "--max-iter", type=int, default=1000, help="Solver iteration cap."
    )
    parser.add_argument(
        "--no-wages",
        action="store_true",
        help="Skip the OEWS wage pull (avoids needing BLS_API_KEY).",
    )
    return parser.parse_args(argv)


def load_observed(train_years, test_years, synthetic, pair_features):
    """Return (transitions, sizes) either from IPUMS or from the test generator."""
    if synthetic:
        sys.path.insert(0, str(ROOT / "tests"))
        from synthetic import make_transitions

        print("  [synthetic] flows generated from a known process, NOT real data\n")
        return make_transitions(
            pair_features,
            {"skill": 2.0, "knowledge": 1.2, "work_activity": 0.6, "ability": 0.3},
            years=list(train_years) + list(test_years),
            seed=7,
        )

    from sources.ipums import latest_extract_file

    path = latest_extract_file(RAW) or (
        LEGACY_IPUMS_FILE if LEGACY_IPUMS_FILE.exists() else None
    )
    if path is None:
        raise SystemExit(
            f"No CPS extract found in {RAW}.\n"
            "Fetch one with:\n"
            "  python src/fetch_data.py cps --submit\n"
            "  python src/fetch_data.py cps --download\n"
            "or run with --synthetic to exercise the pipeline without it."
        )

    from transitions import aggregate_transitions, destination_size, extract_moves, load_ipums

    print(f"  reading {path.name}")
    raw = load_ipums(path)
    print(f"  loaded {len(raw):,} person-year records")
    moves = extract_moves(raw, CROSSWALK)
    print(f"  {len(moves):,} SOC-mapped employed person-years")
    print(f"  {int(moves['moved'].sum()):,} of them changed occupation")
    return aggregate_transitions(moves), destination_size(moves)


def main(argv=None) -> int:
    args = parse_args(argv)

    overlap = set(args.train_years) & set(args.test_years)
    if overlap:
        raise SystemExit(f"train and test years overlap: {sorted(overlap)}")

    print("Phase 1 — fitting the gravity model of occupational mobility\n")

    master = pd.read_parquet(MASTER)
    soc6_master = collapse_to_soc6(master)
    reachable = set(load_crosswalk(CROSSWALK)["soc6"])
    universe = sorted(set(soc6_master["soc6"]) & reachable)
    print(f"  {len(master):,} O*NET codes -> {len(soc6_master):,} SOC-6")
    print(f"  {len(universe):,} of those are reachable from CPS occupation codes")

    pair_features = build_pair_features(
        soc6_master, universe=universe, onet_related=related_pairs(master)
    )
    print(f"  {len(pair_features):,} ordered occupation pairs")

    if not args.no_wages:
        try:
            from sources.bls import fetch_oews, to_wide

            oews = to_wide(
                fetch_oews(universe, cache_path=PROCESSED / "oews.parquet")
            )
            pair_features = add_wage_features(pair_features, oews)
            covered = pair_features["d_log_wage"].notna().mean()
            print(
                f"  OEWS {int(oews['year'].max())}: wages for "
                f"{oews['soc6'].nunique()}/{len(universe)} occupations "
                f"({covered:.0%} of pairs get a wage gap)"
            )
        except Exception as exc:  # noqa: BLE001 -- wages are enrichment, not a hard dep
            print(f"  [warn] OEWS unavailable ({exc}); continuing without wage features")
    print()

    transitions, sizes = load_observed(
        args.train_years, args.test_years, args.synthetic, pair_features
    )

    train = build_panel(pair_features, transitions, sizes, args.train_years)
    test = build_panel(pair_features, transitions, sizes, args.test_years)
    observed_train = int((train["weighted_count"] > 0).sum())
    print(
        f"  train {args.train_years}: {len(train):,} pairs, "
        f"{observed_train:,} with an observed move "
        f"({observed_train / len(train):.1%})"
    )
    print(f"  test  {args.test_years}: {len(test):,} pairs\n")

    fit = fit_gravity(
        train,
        args.train_years,
        args.test_years,
        alpha=args.alpha,
        max_iter=args.max_iter,
    )
    weights = fit.domain_weights()

    if not fit.converged:
        print(
            f"  !! the solver stopped at the iteration cap ({fit.n_iter}). "
            "Re-run with a larger --max-iter before reading anything below "
            "as a result.\n"
        )
    else:
        print(f"  converged in {fit.n_iter} iterations\n")

    print("Learned O*NET domain weights (standardised, largest first)")
    print("-" * 58)
    for _, row in weights.iterrows():
        share = "" if pd.isna(row["share"]) else f"{row['share']:>7.1%}"
        print(f"  {row['domain']:<16} {row['coefficient']:>9.4f}  {share}")

    print("\nHeld-out scorecard")
    print("-" * 58)
    scorecard = compare(test, predict(fit, test), k=args.top_k)
    print(scorecard.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    PROCESSED.mkdir(parents=True, exist_ok=True)
    weights.to_csv(PROCESSED / "domain_weights.csv", index=False)
    scorecard.to_csv(PROCESSED / "model_scorecard.csv", index=False)
    fit.coefficients.to_csv(PROCESSED / "gravity_coefficients.csv", index=False)

    print("\nWrote:")
    for name in ("domain_weights.csv", "model_scorecard.csv", "gravity_coefficients.csv"):
        print(f"  data/processed/{name}")

    if args.synthetic:
        print("\n  Reminder: --synthetic output is not a finding. Get the IPUMS extract.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

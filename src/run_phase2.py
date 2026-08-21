"""Phase 2: model comparison and mobility deserts.

    python src/run_phase2.py                        # full universe
    python src/run_phase2.py --universe white-collar
    python src/run_phase2.py --compare-universes    # both, side by side

Answers two questions:

  1. Is the gravity model's structure earning its keep? Four specifications
     compete on the same held-out years, including one that never sees an
     O*NET feature.
  2. Which occupations have few realistic ways out, and which of those are
     also poorly paid?

Outputs land in `data/processed/` and `reports/`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gravity  # noqa: E402
from deserts import find_deserts, low_paid_and_trapped  # noqa: E402
from evaluate import score  # noqa: E402
from models import overdispersion, run_all  # noqa: E402
from pairs import add_wage_features, build_pair_features, related_pairs  # noqa: E402
from soc import collapse_to_soc6, filter_universe, load_crosswalk  # noqa: E402
from sources.bls import fetch_oews, to_wide  # noqa: E402
from sources.ipums import latest_extract_file  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
CROSSWALK = RAW / "census_occ_to_soc_2018.csv"

DEFAULT_TRAIN = [2018, 2019, 2020, 2021]
DEFAULT_TEST = [2022, 2023, 2024]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--universe", default="all", choices=["all", "white-collar", "professional"]
    )
    parser.add_argument(
        "--compare-universes",
        action="store_true",
        help="Run every universe and report how predictability differs.",
    )
    parser.add_argument("--train-years", type=int, nargs="+", default=DEFAULT_TRAIN)
    parser.add_argument("--test-years", type=int, nargs="+", default=DEFAULT_TEST)
    parser.add_argument("--max-iter", type=int, default=20000)
    parser.add_argument("--top", type=int, default=15)
    return parser.parse_args(argv)


def build_panels(universe, train_years, test_years):
    """Pair features + observed flows, split into train and test panels."""
    master = pd.read_parquet(PROCESSED / "onet_master.parquet")
    soc6_master = collapse_to_soc6(master)

    pair_features = build_pair_features(
        soc6_master, universe=universe, onet_related=related_pairs(master)
    )
    oews = to_wide(fetch_oews(universe, cache_path=PROCESSED / "oews.parquet"))
    pair_features = add_wage_features(pair_features, oews)

    from transitions import aggregate_transitions, destination_size, extract_moves, load_ipums

    path = latest_extract_file(RAW)
    if path is None:
        raise SystemExit(
            f"No CPS extract in {RAW}. Run: python src/fetch_data.py cps --submit"
        )
    moves = extract_moves(load_ipums(path), CROSSWALK)
    transitions = aggregate_transitions(moves)
    sizes = destination_size(moves)

    train = gravity.build_panel(pair_features, transitions, sizes, train_years)
    test = gravity.build_panel(pair_features, transitions, sizes, test_years)
    return train, test, oews, soc6_master


def run_universe(name, codes, args, verbose=True):
    """Fit every model on one universe, return the scorecard and desert table."""
    train, test, oews, soc6_master = build_panels(codes, args.train_years, args.test_years)

    if verbose:
        observed = int((train["weighted_count"] > 0).sum())
        print(f"\n{'=' * 66}")
        print(f"UNIVERSE: {name}  ({len(codes)} occupations)")
        print("=" * 66)
        print(f"  train pairs {len(train):,}, observed {observed:,} ({observed/len(train):.1%})")
        print(f"  overdispersion (person counts): {overdispersion(train['raw_count']):.1f}")
        print("  (Poisson assumes 1.0; PPML point estimates survive this, its SEs do not)")

    results, ppml_fit = run_all(train, test, max_iter=args.max_iter)

    rows = []
    for result in results:
        rows.append({"model": result.name, **score(test, result.predictions), "notes": result.notes})
    scorecard = pd.DataFrame(rows).sort_values("recall_at_10", ascending=False)

    if verbose:
        print(f"\n  Held-out model comparison (fit {args.train_years} -> test {args.test_years})")
        print("  " + "-" * 64)
        display = scorecard.drop(columns="notes")
        print("  " + display.to_string(index=False, float_format=lambda v: f"{v:.4f}").replace("\n", "\n  "))
        if not ppml_fit.converged:
            print(f"\n  !! PPML stopped at the iteration cap ({ppml_fit.n_iter}).")

    # Deserts are computed from the best-ranking model's predictions.
    best = max(results, key=lambda r: score(test, r.predictions)["recall_at_10"])
    deserts = find_deserts(test, best.predictions, oews, soc6_master)

    return scorecard, deserts, ppml_fit, best.name


def main(argv=None) -> int:
    args = parse_args(argv)

    master = pd.read_parquet(PROCESSED / "onet_master.parquet")
    reachable = set(load_crosswalk(CROSSWALK)["soc6"])
    all_codes = sorted(set(collapse_to_soc6(master)["soc6"]) & reachable)

    universes = (
        ["all", "white-collar", "professional"]
        if args.compare_universes
        else [args.universe]
    )

    summary = []
    for name in universes:
        codes = filter_universe(all_codes, name)
        scorecard, deserts, fit, best = run_universe(name, codes, args)

        suffix = "" if name == "all" else f"_{name.replace('-', '_')}"
        scorecard.to_csv(PROCESSED / f"model_comparison{suffix}.csv", index=False)
        deserts.to_csv(PROCESSED / f"mobility_deserts{suffix}.csv", index=False)

        top = scorecard.iloc[0]
        summary.append(
            {
                "universe": name,
                "occupations": len(codes),
                "best_model": top["model"],
                "recall_at_10": top["recall_at_10"],
                "spearman": top["spearman_observed"],
            }
        )

        if name == args.universe or not args.compare_universes:
            trapped = low_paid_and_trapped(deserts)
            print(f"\n  Narrowest options, low pay ({len(trapped)} occupations)")
            print("  " + "-" * 64)
            columns = ["title", "effective_destinations", "upward_share", "annual_median_wage"]
            print("  " + trapped[columns].head(args.top).to_string(
                index=False, float_format=lambda v: f"{v:.2f}"
            ).replace("\n", "\n  "))

    if args.compare_universes:
        print(f"\n{'=' * 66}")
        print("DOES SCOPING TO WHITE COLLAR IMPROVE PREDICTABILITY?")
        print("=" * 66)
        frame = pd.DataFrame(summary)
        print(frame.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        frame.to_csv(PROCESSED / "universe_comparison.csv", index=False)

    print("\nWrote model_comparison*.csv and mobility_deserts*.csv to data/processed/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

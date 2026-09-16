"""Worked example: credential -> job match, with the evidence attached.

    python src/run_credential_demo.py
    python src/run_credential_demo.py --cip 46.0302 --metro youngstown
    python src/run_credential_demo.py --list-trades

Takes one credential in one metro and shows the three things a crosswalk alone
cannot tell you: which of its matches exist locally, which of them pay, and
where the job leads afterwards.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import credentials  # noqa: E402
from sources.ipums import latest_extract_file  # noqa: E402
from sources.oews_metro import METRO_BY_SLUG  # noqa: E402
from sources.oews_metro import to_wide as metro_to_wide  # noqa: E402
from transitions import aggregate_transitions, extract_moves, load_ipums  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
CROSSWALK = RAW / "census_occ_to_soc_2018.csv"
TRANSITION_CACHE = PROCESSED / "observed_transitions.parquet"

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"


def money(value) -> str:
    return f"  ${value:>9,.0f}" if pd.notna(value) else "          —"


def count(value, width: int = 9) -> str:
    return f"{value:>{width},.0f}" if pd.notna(value) else " " * (width - 1) + "—"


def load_transitions() -> pd.DataFrame:
    if TRANSITION_CACHE.exists():
        return pd.read_parquet(TRANSITION_CACHE)
    path = latest_extract_file(RAW)
    if path is None:
        raise SystemExit(f"No CPS extract in {RAW}. Run: python src/fetch_data.py cps --submit")
    frame = aggregate_transitions(extract_moves(load_ipums(path), CROSSWALK))
    frame.to_parquet(TRANSITION_CACHE, index=False)
    return frame


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cip", default="48.0501", help="CIP 2020 program code")
    parser.add_argument("--metro", default="cleveland", help=f"one of: {', '.join(METRO_BY_SLUG)}")
    parser.add_argument("--anchor", help="SOC to trace progression from (default: largest locally)")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--list-trades", action="store_true", help="list trades CIP programs")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    crosswalk = credentials.load_cip_soc()

    if args.list_trades:
        trades = crosswalk[crosswalk["cip"].str.split(".").str[0].str.zfill(2).isin(
            ["46", "47", "48", "49"]
        )]
        counts = trades.groupby(["cip", "cip_title"]).size().sort_values(ascending=False)
        print(f"\n{BOLD}Trades programs (CIP 46–49){RESET}\n")
        for (cip, title), n in counts.items():
            print(f"  {cip:<10} {str(title)[:56]:<57} {n:>2} SOC matches")
        print()
        return 0

    if args.metro not in METRO_BY_SLUG:
        raise SystemExit(f"unknown metro {args.metro!r}; known: {', '.join(METRO_BY_SLUG)}")
    metro = METRO_BY_SLUG[args.metro]

    metro_wide = metro_to_wide(pd.read_parquet(PROCESSED / "oews_metro.parquet"))
    transitions = load_transitions()
    titles = credentials.soc_titles()

    placement = credentials.placement_table(crosswalk, args.cip, metro_wide, args.metro)
    program = placement["cip_title"].iloc[0]

    print()
    print("=" * 78)
    print(f"{BOLD}{program.rstrip('.')}{RESET}  {DIM}CIP {args.cip}{RESET}")
    print(f"{metro.name}")
    print("=" * 78)

    # ---- layer 1: what the crosswalk says, checked against the local market --
    print(f"\n{BOLD}1. The crosswalk's matches, checked against this metro{RESET}")
    print(f"{DIM}   NCES/BLS CIP-SOC 2020. Unranked, no geography, no wage — by design.{RESET}\n")
    print(f"   {'SOC':<9}{'occupation':<42}{'local jobs':>11}{'local pay':>12}   status")
    print("   " + "-" * 76)
    for _, row in placement.iterrows():
        title = str(row["soc_title"])[:40]
        mark = " [supervisory]" if row["supervisory"] else ""
        print(
            f"   {row['soc']:<9}{title:<42}{count(row['local_employment'], 11)}"
            f"{money(row['local_wage'])}  {row['local_status']}{mark}"
        )

    summary = credentials.coverage_summary(placement, None)
    print(
        f"\n   {summary['published_locally']} of {summary['candidates']} matches are "
        f"published in this metro."
    )
    print(
        f"   {summary['share_not_visible_locally']:.0%} of the national employment these "
        f"matches point to\n   ({summary['national_employment_not_visible_locally']:,.0f} of "
        f"{summary['national_employment_all_candidates']:,.0f} jobs) is invisible here."
    )
    if summary["largest_invisible"]:
        big = summary["largest_invisible"]
        print(
            f"\n   Largest invisible match: {big['soc']} {str(big['title'])[:38]}\n"
            f"   — {big['national_employment']:,.0f} jobs nationally, nothing published locally."
        )
        print(
            f"{DIM}   OEWS suppresses small cells, so this is near-certainly a disclosure gap\n"
            f"   rather than genuine absence. OEWS cannot distinguish the two, which is\n"
            f"   why a match engine must not read a missing cell as 'no jobs here'.{RESET}"
        )

    # ---- layer 2: progression ------------------------------------------------
    published = placement[placement["local_status"] == "published"]
    entry = published[~published["supervisory"]]
    if args.anchor:
        anchor = args.anchor
    elif len(entry):
        anchor = entry["soc"].iloc[0]
    elif len(published):
        anchor = published["soc"].iloc[0]
    else:
        anchor = placement["soc"].iloc[0]
    anchor_title = titles.get(anchor, anchor)

    supervisors = placement[placement["supervisory"] & placement["local_employment"].notna()]
    if len(supervisors):
        row = supervisors.iloc[0]
        print(
            f"\n   {len(supervisors)} match(es) are first-line supervisor roles — "
            f"{row['soc']} among them.\n   These require prior experience and are a "
            f"destination, not an entry point.\n   The crosswalk does not distinguish "
            f"them, so ranking on local employment alone\n   would place a new graduate "
            f"there."
        )

    crosswalk_socs = set(placement["soc"])
    progression = credentials.progression_table(
        transitions, anchor, metro_wide, args.metro, crosswalk_socs, top=args.top
    )

    print(f"\n{BOLD}2. Where that job actually leads{RESET}")
    print(f"{DIM}   Observed CPS moves out of {anchor} {anchor_title}, 2018–2025.{RESET}")
    print(f"{DIM}   Observed, not modelled — an auditor can check it against public microdata.{RESET}\n")

    if progression.empty:
        print("   No observed transitions for this occupation.")
    else:
        anchor_wage = placement.loc[placement["soc"] == anchor, "local_wage"]
        base = anchor_wage.iloc[0] if len(anchor_wage) and pd.notna(anchor_wage.iloc[0]) else None
        print(f"   {'share':>6}  {'SOC':<9}{'destination':<40}{'local pay':>12}  vs origin    n")
        print("   " + "-" * 76)
        for _, row in progression.iterrows():
            title = str(titles.get(row["soc_to"], "—"))[:38]
            delta = ""
            if base and pd.notna(row["dest_local_wage"]):
                change = row["dest_local_wage"] / base - 1
                delta = f"{change:+6.0%}"
            flag = "*" if row["aggregated_cps_code"] else " "
            print(
                f"   {row['share']:>6.1%}{flag} {row['soc_to']:<9}{title:<40}"
                f"{money(row['dest_local_wage'])}  {delta:>7}  {int(row['raw_count']):>3}"
            )
        if progression["aggregated_cps_code"].any():
            print(f"\n{DIM}   * CPS reports these as broad groups, not detailed SOC. No local\n"
                  f"     wage exists for a group, so those rows show no pay.{RESET}")
        priced = progression[
            progression["dest_local_wage"].notna() & (progression["raw_count"] >= 3)
        ]
        if base and len(priced):
            down = priced[priced["dest_local_wage"] < base]
            down_mass = down["share"].sum() / priced["share"].sum()
            short = anchor_title.split(",")[0].split(" and ")[0].strip().lower()
            print(
                f"\n   {len(down)} of {len(priced)} destinations that have a local wage and at\n"
                f"   least 3 observed movers pay LESS than {short} do here.\n"
                f"   {down_mass:.0%} of that flow moves down."
            )
        total_n = int(progression["raw_count"].sum())
        print(
            f"\n{DIM}   Sample: {total_n} observed moves across the destinations shown. CPS ASEC\n"
            f"   is thin at occupation level — these are directional, not precise.{RESET}"
        )

    # ---- layer 3: the scope gap ---------------------------------------------
    if not progression.empty and "in_crosswalk" in progression:
        inside = progression.loc[progression["in_crosswalk"], "share"].sum()
        print(f"\n{BOLD}3. The gap the crosswalk cannot close{RESET}\n")
        print(
            f"   Of where {anchor_title.lower()} actually go next, {inside:.0%} lands in an\n"
            f"   occupation the crosswalk lists for this program."
        )
        print(
            f"{DIM}\n   That is not an error in the crosswalk. CIP-SOC is a statement about\n"
            f"   ENTRY — which jobs a program prepares you for — and says nothing about\n"
            f"   progression. The gap is one of scope.\n\n"
            f"   It matters because WIOA grades states on employment in the 4th quarter\n"
            f"   after exit, not on placement. An engine built on the crosswalk alone\n"
            f"   optimises job one; the buyer's scorecard is graded on job two.{RESET}"
        )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

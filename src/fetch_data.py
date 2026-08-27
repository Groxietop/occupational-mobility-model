"""Pull every input this project needs from an API. No manual downloads.

    python src/fetch_data.py oews          # BLS employment + wages (seconds)
    python src/fetch_data.py cps --submit  # queue the IPUMS extract
    python src/fetch_data.py cps --status  # check on it
    python src/fetch_data.py cps --download

IPUMS extracts are queued server-side and take anywhere from a minute to an
hour, so submit and download are separate commands. The extract number is
cached in `data/raw/ipums_extract.json`, so `--status` and `--download` pick
up whatever was last submitted -- including from a previous session.

Keys come from the environment (`BLS_API_KEY`, `IPUMS_API_KEY`) or a local
`.env`, which is gitignored. Never commit them.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env loader so we don't take a dependency for four lines.

    Values already in the environment win, so an explicit export always
    overrides the file.
    """
    path = path or ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def fetch_oews(args) -> int:
    from soc import collapse_to_soc6, load_crosswalk
    from sources.bls import fetch_oews as pull
    from sources.bls import to_wide
    import pandas as pd

    master = pd.read_parquet(PROCESSED / "onet_master.parquet")
    soc6_master = collapse_to_soc6(master)
    reachable = set(load_crosswalk(RAW / "census_occ_to_soc_2018.csv")["soc6"])
    universe = sorted(set(soc6_master["soc6"]) & reachable)

    cache = PROCESSED / "oews.parquet"
    if args.refresh and cache.exists():
        cache.unlink()

    print(f"Fetching OEWS for {len(universe)} occupations...")
    tidy = pull(universe, cache_path=cache)
    wide = to_wide(tidy)
    print(f"  {wide['soc6'].nunique()}/{len(universe)} occupations covered")
    print(f"  data year: {int(wide['year'].max())}")
    print(f"  cached at {cache.relative_to(ROOT)}")
    return 0


def onet(args) -> int:
    """Check whether the committed O*NET bulk database has gone stale."""
    from sources.onet import ONetError, check_freshness

    freshness = check_freshness(RAW)
    print(f"O*NET local database:  {freshness.local_version or 'not found'}")
    print(f"O*NET published:       {freshness.published_version or 'unknown'}")
    if freshness.taxonomy:
        print(f"Taxonomy:              {freshness.taxonomy}")
    print()
    print(f"  {freshness.describe()}")

    if args.download:
        from sources.onet import download_bulk

        version = args.download
        print(f"\nDownloading O*NET {version} bulk database (~13 MB)...")
        where = download_bulk(version, RAW / f"onet_{version.replace('.', '_')}")
        print(f"  unpacked to {where}")
        print(f"  now run: python src/build_master.py --raw {where} "
              f"--out data/processed/onet_master.parquet")
        return 0

    if args.occupation:
        from sources.onet import domain_ratings, occupation as fetch_occupation

        try:
            detail = fetch_occupation(args.occupation)
            print(f"\n{detail['code']} — {detail['title']}")
            ratings = domain_ratings(args.occupation, args.domain)
            top = sorted(ratings.items(), key=lambda kv: -kv[1])[:8]
            print(f"  top {args.domain} ratings:")
            for name, score in top:
                print(f"    {score:>5.1f}  {name}")
        except ONetError as exc:
            print(f"  [warn] {exc}")

    # Stale is a finding, not a failure -- the pipeline still runs on the
    # committed database. Exit non-zero so CI can choose to notice.
    return 0 if freshness.is_current else 2


def _ipums_client():
    from sources.ipums import get_client

    return get_client()


def cps(args) -> int:
    from sources.ipums import ExtractHandle, build_extract, download, status, submit

    if args.submit:
        client = _ipums_client()
        extract = build_extract(years=args.years)
        print(
            f"Submitting CPS ASEC extract: {len(extract.samples)} samples, "
            f"{len(extract.variables)} variables"
        )
        handle = submit(client, extract, RAW)
        print(f"  extract #{handle.number} queued")
        print(f"  check with: python src/fetch_data.py cps --status")
        return 0

    handle = ExtractHandle.load(RAW)
    if handle is None:
        raise SystemExit(
            "No submitted extract on record. Run with --submit first."
        )

    client = _ipums_client()

    if args.status:
        print(f"extract #{handle.number}: {status(client, handle)}")
        return 0

    if args.download:
        state = status(client, handle)
        if state != "completed":
            raise SystemExit(
                f"extract #{handle.number} is '{state}', not ready to download yet."
            )
        path = download(client, handle, RAW)
        print(f"  downloaded {path.relative_to(ROOT)}")
        return 0

    print(f"extract #{handle.number}: {status(client, handle)}")
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="source", required=True)

    oews = sub.add_parser("oews", help="BLS employment and wages by occupation")
    oews.add_argument("--refresh", action="store_true", help="Ignore the cache.")
    oews.set_defaults(func=fetch_oews)

    cps_parser = sub.add_parser("cps", help="IPUMS CPS ASEC occupation transitions")
    cps_parser.add_argument("--submit", action="store_true", help="Queue a new extract.")
    cps_parser.add_argument("--status", action="store_true", help="Check the queue.")
    cps_parser.add_argument("--download", action="store_true", help="Fetch when ready.")
    cps_parser.add_argument(
        "--years", type=int, nargs="+", default=None, help="ASEC years to request."
    )
    cps_parser.set_defaults(func=cps)

    onet_parser = sub.add_parser(
        "onet", help="Check whether the committed O*NET database is current"
    )
    onet_parser.add_argument(
        "--occupation", default=None, help="Also spot-check one O*NET-SOC code"
    )
    onet_parser.add_argument(
        "--domain", default="skill", help="Domain to show for --occupation"
    )
    onet_parser.add_argument(
        "--download", default=None, metavar="VERSION",
        help="Fetch and unpack a bulk release, e.g. --download 31.0",
    )
    onet_parser.set_defaults(func=onet)

    return parser.parse_args(argv)


def main(argv=None) -> int:
    load_dotenv()
    args = parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

"""IPUMS CPS ASEC extracts over the API, instead of a manual download.

The manual path (log in, tick boxes, wait for an email, unzip into data/raw)
can't be automated or re-run, which makes the whole pipeline unreproducible
at its first step. This module defines the extract in code, submits it, and
caches the result, so `make data` is a real thing and adding a year is a one
line change.

Requires `IPUMS_API_KEY` in the environment (get one at
https://account.ipums.org/api_keys). Extracts are queued server-side and take
anywhere from a minute to an hour, so submission and download are separate
steps and the extract id is cached between them.

A note on weights
-----------------
ASEC samples must be weighted with `ASECWT`. `WTFINL` is the basic monthly
weight and is not merely the wrong choice here -- the IPUMS API rejects it
outright for ASEC samples ("This variable is not available in any of the
samples currently selected"). The original notebook read `WTFINL` from a
hand-built extract, which is the supplement's transition signal weighted by a
column that doesn't belong to it. We request ASECWT only.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_YEARS = list(range(2018, 2025))

# Core variables the transition model needs.
CORE_VARIABLES = [
    "YEAR",       # survey year
    "CPSIDP",     # person identifier
    "ASECWT",     # ASEC person weight -- the only valid one for this supplement
    "EMPSTAT",    # employment status
    "OCC",        # occupation, current
    "OCCLY",      # occupation, last year  <- the transition
    "IND",        # industry, current
    "INDLY",      # industry, last year
    "WKSWORK1",   # weeks worked last year
]

# Not needed for the gravity model, but they cost nothing to request now and
# unlock heterogeneity work later (do transition patterns differ by age,
# education, sex?) without a second extract and another wait.
EXTENDED_VARIABLES = [
    "AGE",
    "SEX",
    "EDUC",
    "INCWAGE",    # wage income -- lets wage change be measured per person
]

STATE_FILE = "ipums_extract.json"


def sample_id(year: int) -> str:
    """IPUMS sample identifier for the ASEC supplement of a given year."""
    return f"cps{year}_03s"


@dataclass
class ExtractHandle:
    """Enough state to reconnect to a submitted extract on a later run."""

    number: int
    collection: str = "cps"

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / STATE_FILE
        path.write_text(json.dumps({"number": self.number, "collection": self.collection}))
        return path

    @classmethod
    def load(cls, directory: Path) -> "ExtractHandle | None":
        path = Path(directory) / STATE_FILE
        if not path.exists():
            return None
        payload = json.loads(path.read_text())
        return cls(number=int(payload["number"]), collection=payload.get("collection", "cps"))


def get_client(api_key: str | None = None):
    """An authenticated IPUMS client. Never takes the key from a file in-repo."""
    from ipumspy import IpumsApiClient

    key = api_key or os.environ.get("IPUMS_API_KEY")
    if not key:
        raise RuntimeError(
            "IPUMS_API_KEY is not set. Get a key at "
            "https://account.ipums.org/api_keys and export it, or put it in a "
            "local .env file (which is gitignored)."
        )
    return IpumsApiClient(key)


def build_extract(years=None, extended: bool = True, description: str | None = None):
    """Define the CPS ASEC extract this project needs, in code."""
    from ipumspy import MicrodataExtract

    years = list(years or DEFAULT_YEARS)
    variables = CORE_VARIABLES + (EXTENDED_VARIABLES if extended else [])
    return MicrodataExtract(
        collection="cps",
        samples=[sample_id(y) for y in years],
        variables=variables,
        description=description or (
            f"Occupational mobility model: ASEC {min(years)}-{max(years)} "
            "occupation transitions"
        ),
        data_format="csv",
    )


def submit(client, extract, state_dir: Path) -> ExtractHandle:
    """Submit an extract and cache its number so we can pick it up later."""
    submitted = client.submit_extract(extract)
    handle = ExtractHandle(number=int(submitted.extract_id))
    handle.save(Path(state_dir))
    return handle


def status(client, handle: ExtractHandle) -> str:
    return client.extract_status(handle.number, collection=handle.collection)


def extract_filename(handle: ExtractHandle) -> str:
    """IPUMS names downloads `<collection>_<5-digit number>`, e.g. cps_00001."""
    return f"{handle.collection}_{handle.number:05d}"


def find_extract_file(handle: ExtractHandle, target_dir: Path) -> Path | None:
    """Locate a downloaded extract by its IPUMS name.

    Matching on the extract's own name rather than globbing for any CSV --
    data/raw also holds the Census crosswalk, and a bare `*.csv` glob happily
    returns that instead.
    """
    stem = extract_filename(handle)
    for suffix in (".csv.gz", ".csv", ".dat.gz"):
        candidate = Path(target_dir) / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def latest_extract_file(target_dir: Path) -> Path | None:
    """Most recent downloaded CPS extract, whatever its number."""
    matches = sorted(Path(target_dir).glob("cps_[0-9]*.csv.gz"))
    matches += sorted(Path(target_dir).glob("cps_[0-9]*.csv"))
    return matches[-1] if matches else None


def download(client, handle: ExtractHandle, target_dir: Path) -> Path:
    """Download a completed extract into `target_dir`, return the data file."""
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    client.download_extract(
        handle.number, collection=handle.collection, download_dir=str(target)
    )
    found = find_extract_file(handle, target)
    if found is None:
        raise FileNotFoundError(
            f"expected {extract_filename(handle)}.csv.gz in {target} after download"
        )
    return found

"""OEWS employment and wage data from the BLS public API.

Gives the model two things it can't get from O*NET: how many people actually
work in each occupation (the base-rate control the gravity model depends on)
and what each occupation pays (so a predicted move can be scored on whether
it's a raise).

Series ID construction
----------------------
OEWS series IDs are 25 characters and the official API docs don't document
them, which is why https://github.com/govex/bls-oews-api-tutorial exists:

    OE | U | areatype(1) | area(7) | industry(6) | occupation(6) | datatype(2)

    OEUN000000000000015125213
    ^^ ^ ^  ^^^^^^^ ^^^^^^ ^^^^^^ ^^
    |  | |  |       |      |      +-- 13 = annual median wage
    |  | |  |       |      +--------- 151252 = SOC 15-1252, Software Developers
    |  | |  |       +---------------- 000000 = all industries
    |  | |  +------------------------ 0000000 = national
    |  | +--------------------------- N = national area type
    |  +----------------------------- U = not seasonally adjusted
    +-------------------------------- OE = the OEWS survey

The one real trap: **the API only carries the most recent year.** Passing
`startyear`/`endyear` for any earlier year returns REQUEST_SUCCEEDED with
zero data points and a "No Data Available" message per year -- which reads
like a bad series ID but isn't. So we never send year parameters. Historical
OEWS needs the flat files at https://download.bls.gov/pub/time.series/oe/.

Limits with a registered v2 key: 500 requests/day, 50 series per request.
Covering ~430 occupations at 3 measures each is ~26 requests, so a full pull
costs about 5% of the daily budget.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path

import pandas as pd

API_V2 = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

NATIONAL_ALL_INDUSTRIES = ("N", "0000000", "000000")

# The three measures the model uses. See the tutorial's reference file for the
# full 01-17 list (percentiles, RSEs, location quotients).
MEASURES = {
    "employment": "01",
    "annual_mean_wage": "04",
    "annual_median_wage": "13",
}

MAX_SERIES_PER_REQUEST = 50
REQUEST_PAUSE_SECONDS = 0.5


def soc_to_occupation_code(soc6: str) -> str:
    """'15-1252' -> '151252'. OEWS strips the hyphen."""
    return str(soc6).replace("-", "").strip()


def series_id(soc6: str, measure: str) -> str:
    """Build the 25-character national, all-industries OEWS series ID."""
    if measure not in MEASURES:
        raise ValueError(f"unknown measure {measure!r}; expected one of {sorted(MEASURES)}")
    area_type, area, industry = NATIONAL_ALL_INDUSTRIES
    occupation = soc_to_occupation_code(soc6)
    if len(occupation) != 6:
        raise ValueError(f"expected a 6-digit SOC code, got {soc6!r}")
    sid = f"OE U {area_type} {area} {industry} {occupation} {MEASURES[measure]}".replace(" ", "")
    if len(sid) != 25:
        raise ValueError(f"built a malformed series id: {sid!r} ({len(sid)} chars)")
    return sid


def _post(series_ids: list[str], api_key: str, timeout: float = 60.0) -> dict:
    # Deliberately no startyear/endyear -- see the module docstring.
    body = json.dumps({"seriesid": series_ids, "registrationkey": api_key}).encode()
    request = urllib.request.Request(
        API_V2, data=body, headers={"Content-type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def fetch_oews(
    soc_codes,
    measures=None,
    api_key: str | None = None,
    cache_path: Path | None = None,
    pause: float = REQUEST_PAUSE_SECONDS,
) -> pd.DataFrame:
    """Fetch OEWS measures for a list of SOC-6 codes.

    Returns tidy rows: soc6, measure, value, year. Occupations OEWS doesn't
    publish (too small to disclose, or not a real SOC) simply don't appear --
    that's a real gap in the data, not an error, so it isn't filled in.
    """
    key = api_key or os.environ.get("BLS_API_KEY")
    if not key:
        raise RuntimeError(
            "BLS_API_KEY is not set. Register at "
            "https://data.bls.gov/registrationEngine/ and export it, or put it "
            "in a local .env file (which is gitignored)."
        )

    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            return pd.read_parquet(cache_path)

    measures = list(measures or MEASURES)
    wanted = [(code, measure) for code in soc_codes for measure in measures]
    lookup = {series_id(code, measure): (code, measure) for code, measure in wanted}
    ids = list(lookup)

    rows = []
    for start in range(0, len(ids), MAX_SERIES_PER_REQUEST):
        batch = ids[start : start + MAX_SERIES_PER_REQUEST]
        payload = _post(batch, key)
        if payload.get("status") != "REQUEST_SUCCEEDED":
            raise RuntimeError(
                f"BLS API returned {payload.get('status')}: {payload.get('message')}"
            )
        for series in payload["Results"]["series"]:
            code, measure = lookup[series["seriesID"]]
            for point in series.get("data", []):
                value = point["value"].replace(",", "").strip()
                if value in {"", "-", "*", "**", "#"}:
                    continue  # BLS suppression markers
                rows.append(
                    {
                        "soc6": code,
                        "measure": measure,
                        "value": float(value),
                        "year": int(point["year"]),
                    }
                )
        if pause and start + MAX_SERIES_PER_REQUEST < len(ids):
            time.sleep(pause)

    frame = pd.DataFrame(rows, columns=["soc6", "measure", "value", "year"])

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(cache_path, index=False)

    return frame


def to_wide(tidy: pd.DataFrame) -> pd.DataFrame:
    """Tidy OEWS rows -> one row per occupation, one column per measure."""
    if tidy.empty:
        return pd.DataFrame(columns=["soc6", "year", *MEASURES])
    wide = tidy.pivot_table(
        index=["soc6", "year"], columns="measure", values="value", aggfunc="first"
    ).reset_index()
    wide.columns.name = None
    return wide

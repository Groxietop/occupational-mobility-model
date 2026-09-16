"""OEWS employment and wages by metropolitan area.

`bls.py` fetches national OEWS, which is all the gravity model needs. The
geographic desert work needs the same measures cut by metro: which occupations
actually exist near a worker, in what numbers, and at what local wage.

Series IDs extend the national format documented in `bls.py`. The only change
is the area block:

    OE | U | areatype(1) | area(7) | industry(6) | occupation(6) | datatype(2)

    OEUM001446000000015125213   Boston, Software Developers, annual median
       ^ ^^^^^^^
       | +-------- 0014460 = CBSA 14460, LEFT-zero-padded to 7 digits
       +---------- M = metropolitan area (N = national, S = state)

Why the bulk flat files aren't used
-----------------------------------
https://www.bls.gov/oes/special-requests/oesm24ma.zip would give every
occupation x metro in one download, but BLS now returns 403 to non-browser
clients regardless of User-Agent. So this goes through the API instead, which
costs roughly 17 requests per metro (421 occupations x 2 measures, 50 series
per request) against a 500/day budget.

That budget is the reason for incremental caching: each metro is written to the
cache as soon as it lands, so an interrupted or rate-limited run resumes
instead of starting over.

The silent-failure trap
-----------------------
A wrong CBSA code does not error. It returns REQUEST_SUCCEEDED with zero data
points, which is indistinguishable from an occupation that is genuinely
suppressed for confidentiality. `verify_areas()` exists because of this: it
probes each metro with an occupation present in every local economy and reports
which ones came back empty.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

API_V2 = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

MAX_SERIES_PER_REQUEST = 50
REQUEST_PAUSE_SECONDS = 0.5
ALL_INDUSTRIES = "000000"

# Shared with bls.py; repeated here so this module reads on its own.
MEASURES = {
    "employment": "01",
    "annual_mean_wage": "04",
    "annual_median_wage": "13",
}

SUPPRESSION_MARKERS = {"", "-", "*", "**", "#"}

# An occupation present in essentially every local economy, used to prove a
# metro's area code is right. Retail Salespersons is the safest such probe:
# large, ubiquitous, and never suppressed in a metro of any size.
PROBE_SOC = "41-2031"


@dataclass(frozen=True)
class Metro:
    slug: str
    cbsa: str
    name: str
    kind: str  # shorthand for the local economy, used when reporting contrasts

    @property
    def area_code(self) -> str:
        return self.cbsa.zfill(7)


# A deliberately contrastive panel rather than the largest metros.
#
# The geographic desert claim is that a place can have jobs, low unemployment
# and decent wages and still offer its workers few upward exits. Testing that
# needs local economies that differ in *structure*, not just in size -- so this
# spans high-wage tech, diversified large metros, manufacturing-legacy regions,
# and agricultural valleys.
METROS: tuple[Metro, ...] = (
    # High-wage, professional-heavy
    Metro("san-jose", "41940", "San Jose-Sunnyvale-Santa Clara, CA", "tech"),
    Metro("boston", "14460", "Boston-Cambridge-Nashua, MA-NH", "tech"),
    Metro("seattle", "42660", "Seattle-Tacoma-Bellevue, WA", "tech"),
    # Large and diversified
    Metro("new-york", "35620", "New York-Newark-Jersey City, NY-NJ-PA", "diversified"),
    Metro("chicago", "16980", "Chicago-Naperville-Elgin, IL-IN-WI", "diversified"),
    Metro("los-angeles", "31080", "Los Angeles-Long Beach-Anaheim, CA", "diversified"),
    Metro("atlanta", "12060", "Atlanta-Sandy Springs-Alpharetta, GA", "diversified"),
    Metro("dallas", "19100", "Dallas-Fort Worth-Arlington, TX", "diversified"),
    # Manufacturing legacy
    Metro("detroit", "19820", "Detroit-Warren-Dearborn, MI", "manufacturing"),
    # CBSA 17460 (the old Cleveland-Elyria code) returns no OEWS data; the
    # revised code is 17410. Caught by verify_areas, not by an error.
    Metro("cleveland", "17410", "Cleveland, OH", "manufacturing"),
    Metro("buffalo", "15380", "Buffalo-Cheektowaga, NY", "manufacturing"),
    Metro("pittsburgh", "38300", "Pittsburgh, PA", "manufacturing"),
    Metro("youngstown", "49660", "Youngstown-Warren-Boardman, OH-PA", "manufacturing"),
    Metro("toledo", "45780", "Toledo, OH", "manufacturing"),
    # Agricultural / lower-wage
    Metro("bakersfield", "12540", "Bakersfield, CA", "agricultural"),
    Metro("fresno", "23420", "Fresno, CA", "agricultural"),
    Metro("el-paso", "21340", "El Paso, TX", "border"),
)

METRO_BY_SLUG = {m.slug: m for m in METROS}


class OEWSMetroError(RuntimeError):
    """The BLS API rejected a request, or no key is configured."""


def series_id(soc6: str, measure: str, area_code: str) -> str:
    """Build the 25-character metro OEWS series ID."""
    if measure not in MEASURES:
        raise ValueError(f"unknown measure {measure!r}; expected one of {sorted(MEASURES)}")
    occupation = str(soc6).replace("-", "").strip()
    if len(occupation) != 6:
        raise ValueError(f"expected a 6-digit SOC code, got {soc6!r}")
    if len(area_code) != 7:
        raise ValueError(f"area code must be 7 digits, got {area_code!r}")
    sid = f"OEUM{area_code}{ALL_INDUSTRIES}{occupation}{MEASURES[measure]}"
    if len(sid) != 25:
        raise ValueError(f"built a malformed series id: {sid!r} ({len(sid)} chars)")
    return sid


def _api_key(explicit: str | None = None) -> str:
    key = explicit or os.environ.get("BLS_API_KEY")
    if key:
        return key
    env = Path(__file__).resolve().parents[2] / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line.startswith("BLS_API_KEY=") and not line.startswith("#"):
                value = line.split("=", 1)[1].strip().strip("'\"")
                if value:
                    return value
    raise OEWSMetroError(
        "BLS_API_KEY is not set. Register at "
        "https://data.bls.gov/registrationEngine/ and export it, or put it in a "
        "local .env file (which is gitignored)."
    )


def _post(series_ids: list[str], api_key: str, timeout: float = 90.0) -> dict:
    # No startyear/endyear -- the API carries only the latest year for OEWS.
    body = json.dumps({"seriesid": series_ids, "registrationkey": api_key}).encode()
    request = urllib.request.Request(
        API_V2, data=body, headers={"Content-type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def _fetch_one_metro(
    metro: Metro,
    soc_codes: list[str],
    measures: list[str],
    api_key: str,
    pause: float,
) -> pd.DataFrame:
    lookup = {
        series_id(soc, measure, metro.area_code): (soc, measure)
        for soc in soc_codes
        for measure in measures
    }
    ids = list(lookup)

    rows: list[dict] = []
    for start in range(0, len(ids), MAX_SERIES_PER_REQUEST):
        batch = ids[start : start + MAX_SERIES_PER_REQUEST]
        payload = _post(batch, api_key)
        status = payload.get("status")
        if status != "REQUEST_SUCCEEDED":
            message = " ".join(payload.get("message", []) or [str(payload)])
            raise OEWSMetroError(f"BLS API returned {status}: {message}")
        for series in payload["Results"]["series"]:
            soc, measure = lookup[series["seriesID"]]
            for point in series.get("data", []):
                raw = str(point["value"]).replace(",", "").strip()
                if raw in SUPPRESSION_MARKERS:
                    continue
                try:
                    value = float(raw)
                except ValueError:
                    continue
                rows.append(
                    {
                        "metro": metro.slug,
                        "cbsa": metro.cbsa,
                        "metro_name": metro.name,
                        "kind": metro.kind,
                        "soc6": soc,
                        "measure": measure,
                        "value": value,
                        "year": int(point["year"]),
                    }
                )
        if pause and start + MAX_SERIES_PER_REQUEST < len(ids):
            time.sleep(pause)

    return pd.DataFrame(rows)


def fetch_metro_oews(
    soc_codes,
    metros=METROS,
    measures=("employment", "annual_median_wage"),
    api_key: str | None = None,
    cache_path: Path | None = None,
    pause: float = REQUEST_PAUSE_SECONDS,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fetch OEWS measures for every (occupation, metro) pair.

    Returns tidy rows: metro, cbsa, metro_name, kind, soc6, measure, value, year.

    Pairs OEWS does not publish simply do not appear. That absence is real
    information -- an occupation genuinely too small to disclose in a metro is
    an occupation a local worker cannot realistically move into -- so it is
    never filled in or interpolated.

    Caching is incremental: each metro is appended to `cache_path` as it lands,
    and metros already present are skipped. A run interrupted by the daily rate
    limit resumes where it stopped.
    """
    key = _api_key(api_key)
    soc_codes = list(soc_codes)
    measures = list(measures)

    cached = pd.DataFrame()
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            cached = pd.read_parquet(cache_path)

    done = set(cached["metro"].unique()) if not cached.empty else set()
    pending = [m for m in metros if m.slug not in done]

    if verbose and done:
        print(f"  cached: {len(done)} metros already fetched, skipping")
    if verbose and pending:
        cost = len(pending) * -(-len(soc_codes) * len(measures) // MAX_SERIES_PER_REQUEST)
        print(f"  fetching {len(pending)} metros (~{cost} API requests of 500/day)")

    frames = [cached] if not cached.empty else []
    for metro in pending:
        try:
            frame = _fetch_one_metro(metro, soc_codes, measures, key, pause)
        except OEWSMetroError as exc:
            if verbose:
                print(f"  {metro.slug}: STOPPED — {exc}")
                print("  partial results are cached; rerun to resume")
            break
        frames.append(frame)
        if verbose:
            covered = frame["soc6"].nunique()
            print(f"  {metro.slug:<14} {covered:>4}/{len(soc_codes)} occupations published")
        if cache_path is not None:
            pd.concat(frames, ignore_index=True).to_parquet(cache_path, index=False)

    if not frames:
        return pd.DataFrame(
            columns=["metro", "cbsa", "metro_name", "kind", "soc6", "measure", "value", "year"]
        )
    return pd.concat(frames, ignore_index=True)


def to_wide(tidy: pd.DataFrame) -> pd.DataFrame:
    """Tidy metro rows -> one row per (metro, occupation), one column per measure."""
    if tidy.empty:
        return pd.DataFrame(columns=["metro", "soc6", "year", *MEASURES])
    wide = tidy.pivot_table(
        index=["metro", "metro_name", "kind", "soc6", "year"],
        columns="measure",
        values="value",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    return wide


def verify_areas(metros=METROS, api_key: str | None = None) -> pd.DataFrame:
    """Confirm every metro's CBSA code returns data.

    A wrong code returns success with zero rows, so this cannot be checked by
    status. One probe occupation per metro costs a single API request in total.
    """
    key = _api_key(api_key)
    lookup = {series_id(PROBE_SOC, "employment", m.area_code): m for m in metros}
    payload = _post(list(lookup), key)
    if payload.get("status") != "REQUEST_SUCCEEDED":
        raise OEWSMetroError(f"BLS API returned {payload.get('status')}")

    rows = []
    for series in payload["Results"]["series"]:
        metro = lookup[series["seriesID"]]
        points = [
            p for p in series.get("data", []) if str(p["value"]).strip() not in SUPPRESSION_MARKERS
        ]
        rows.append(
            {
                "slug": metro.slug,
                "cbsa": metro.cbsa,
                "name": metro.name,
                "series_id": series["seriesID"],
                "ok": bool(points),
                "probe_employment": float(str(points[0]["value"]).replace(",", ""))
                if points
                else None,
            }
        )
    return pd.DataFrame(rows).sort_values("ok")


if __name__ == "__main__":
    frame = verify_areas()
    print(frame.to_string(index=False))
    bad = frame[~frame["ok"]]
    if len(bad):
        print(f"\n{len(bad)} metro(s) returned no data — check these CBSA codes")

"""Credential -> job matching, with the evidence for each match attached.

A credential-to-job engine answers: given this credential, in this region, which
jobs should this person be matched to, and is that a job that actually exists
and pays. Three separate questions, three separate sources, and the failure
modes are different for each.

The starting point is the NCES/BLS CIP-SOC crosswalk, which is what everyone
uses. Its own documentation is explicit about what it is:

    "The CIP SOC Crosswalk is not based on actual empirical data. Rather, the
    matches are based on the content of the CIP Code and SOC Code descriptions
    combined with expertise from statisticians at both federal agencies."
    -- https://nces.ed.gov/ipeds/cipcode/post3.aspx?y=56

That is a reasonable design for a candidate set and a bad one for a match. It
is unranked, has no geography, and no wage. This module adds the three layers a
defensible match needs on top of it.

What the crosswalk is NOT, and why that matters
-----------------------------------------------
CIP-SOC is a statement about ENTRY: which jobs a program prepares you for. It
is not a statement about PROGRESSION: where that job leads afterwards. Those are
different questions and the crosswalk only answers the first.

This distinction is easy to get wrong. Comparing "where do machinists move next"
against "what does machinist training prepare you for" and calling the gap an
error of the crosswalk would be comparing two different things. The gap is real
and decision-relevant, but it is a gap in SCOPE, not a mistake -- and it matters
because WIOA measures states on employment in the 4th quarter after exit, not
just placement. An engine built only on the crosswalk optimises job one; the
buyer's scorecard is graded on job two.
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

CROSSWALK_XLSX = RAW / "CIP2020_SOC2018_Crosswalk.xlsx"
NATIONAL_CACHE = PROCESSED / "oews_national_by_soc.parquet"

API_V2 = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
MEASURES = {"employment": "01", "annual_median_wage": "13"}
SUPPRESSION_MARKERS = {"", "-", "*", "**", "#"}

# Below this many national jobs, an occupation missing from a metro's tables is
# plausibly just small. Above it, absence is far more likely to be OEWS
# confidentiality suppression than genuine absence from a large metro -- and
# the two cannot be told apart from OEWS alone.
SUPPRESSION_SUSPICION_THRESHOLD = 40_000

# CPS collapses some detailed SOC codes into broad groups (trailing zero, e.g.
# 51-4120). Those are not true SOC-6 and will not join to the crosswalk.
def is_aggregated_soc(soc: str) -> bool:
    return str(soc).endswith("0") and not str(soc).endswith("00")


def is_first_line_supervisor(soc: str) -> bool:
    """True for the SOC first-line supervisor codes.

    Structural, not a guess: in SOC 2018 the minor group `10` inside major
    groups 35-53 is reserved for "First-Line Supervisors of ..." -- 47-1011
    (construction trades), 51-1011 (production), 49-1011 (mechanics), and so on.

    This matters because the CIP-SOC crosswalk lists supervisory and entry-level
    occupations side by side with nothing to tell them apart. For CIP 46.0302
    (Electrician) the largest match in Pittsburgh is 47-1011, First-Line
    Supervisors of Construction Trades -- a role that requires years of prior
    experience and is a destination, not an entry point. A matching engine that
    ranks purely on local employment will put a fresh credential holder there.

    The authoritative field is BLS Employment Projections' "work experience in a
    related occupation" assignment. It is not used here because BLS now returns
    403 to every non-browser client for the EP tables (the v2 timeseries API,
    which does not carry these assignments, is the only route still open). So
    this reads the SOC structure directly, which is stable and checkable.
    """
    text = str(soc)
    if len(text) < 5 or text[2] != "-":
        return False
    try:
        major = int(text[:2])
    except ValueError:
        return False
    return 35 <= major <= 53 and text[3:5] == "10"


def load_cip_soc(path: Path = CROSSWALK_XLSX) -> pd.DataFrame:
    """The CIP->SOC sheet as tidy rows: cip, cip_title, soc, soc_title."""
    frame = pd.read_excel(path, sheet_name="CIP-SOC", dtype=str)
    frame.columns = ["cip", "cip_title", "soc", "soc_title"]
    return frame.dropna(subset=["cip", "soc"])


def soc_titles(path: Path = CROSSWALK_XLSX) -> dict:
    """SOC code -> title, from the reverse sheet (broader coverage)."""
    frame = pd.read_excel(path, sheet_name="SOC-CIP", dtype=str)
    frame.columns = ["soc", "soc_title", "cip", "cip_title"]
    return frame.drop_duplicates("soc").set_index("soc")["soc_title"].to_dict()


def program_candidates(crosswalk: pd.DataFrame, cip: str) -> pd.DataFrame:
    """Every occupation the crosswalk says this program prepares a student for."""
    out = crosswalk[crosswalk["cip"] == cip]
    if out.empty:
        raise KeyError(f"CIP {cip!r} not found in the crosswalk")
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# National levels
#
# The model's own oews.parquet covers only the 421 occupations in the modelling
# universe. Crosswalk candidates routinely fall outside it -- 10 of the 13 for
# machining do -- and reading those absences as "BLS publishes nothing" would
# turn a gap in this pipeline into a false finding about the data. So national
# levels for arbitrary SOC codes are fetched directly and cached.
# ---------------------------------------------------------------------------


def _api_key() -> str:
    key = os.environ.get("BLS_API_KEY")
    if key:
        return key
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line.startswith("BLS_API_KEY=") and not line.startswith("#"):
                value = line.split("=", 1)[1].strip().strip("'\"")
                if value:
                    return value
    raise RuntimeError("BLS_API_KEY is not set; see .env.example")


def national_levels(socs, cache_path: Path = NATIONAL_CACHE) -> pd.DataFrame:
    """National employment and median wage for arbitrary SOC-6 codes."""
    socs = [s for s in dict.fromkeys(socs) if not is_aggregated_soc(s)]
    cached = pd.read_parquet(cache_path) if Path(cache_path).exists() else pd.DataFrame()
    have = set(cached["soc"]) if not cached.empty else set()
    missing = [s for s in socs if s not in have]

    if missing:
        ids = {}
        for soc in missing:
            for measure, code in MEASURES.items():
                ids[f"OEUN0000000000000{soc.replace('-', '')}{code}"] = (soc, measure)
        rows: dict = {}
        batch = list(ids)
        for start in range(0, len(batch), 50):
            body = json.dumps(
                {"seriesid": batch[start : start + 50], "registrationkey": _api_key()}
            ).encode()
            request = urllib.request.Request(
                API_V2, data=body, headers={"Content-type": "application/json"}
            )
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = json.loads(response.read().decode())
            if payload.get("status") != "REQUEST_SUCCEEDED":
                raise RuntimeError(f"BLS API returned {payload.get('status')}")
            for series in payload["Results"]["series"]:
                soc, measure = ids[series["seriesID"]]
                for point in series.get("data", []):
                    value = str(point["value"]).replace(",", "").strip()
                    if value in SUPPRESSION_MARKERS:
                        continue
                    rows.setdefault(soc, {"soc": soc})[measure] = float(value)
        fresh = pd.DataFrame(list(rows.values()))
        # Occupations BLS publishes nothing for still get a row, so a later run
        # does not re-request them.
        blanks = pd.DataFrame({"soc": [s for s in missing if s not in rows]})
        cached = pd.concat([cached, fresh, blanks], ignore_index=True)
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        cached.to_parquet(cache_path, index=False)

    return cached[cached["soc"].isin(socs)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# The three layers
# ---------------------------------------------------------------------------


def placement_table(
    crosswalk: pd.DataFrame,
    cip: str,
    metro_wide: pd.DataFrame,
    metro: str,
) -> pd.DataFrame:
    """Crosswalk candidates, ranked by what is actually available locally.

    `local_status` is the column that matters:

      published     OEWS reports this occupation in this metro
      suppressed?   large nationally but absent locally -- almost certainly
                    a confidentiality suppression, but OEWS cannot confirm it
      small/absent  small nationally too, so absence is unremarkable
    """
    candidates = program_candidates(crosswalk, cip)
    national = national_levels(candidates["soc"]).rename(
        columns={
            "employment": "national_employment",
            "annual_median_wage": "national_wage",
        }
    )
    local = (
        metro_wide[metro_wide["metro"] == metro]
        .loc[:, ["soc6", "employment", "annual_median_wage"]]
        .rename(
            columns={
                "soc6": "soc",
                "employment": "local_employment",
                "annual_median_wage": "local_wage",
            }
        )
    )

    out = candidates.merge(national, on="soc", how="left").merge(local, on="soc", how="left")

    def status(row):
        if pd.notna(row.get("local_employment")):
            return "published"
        if pd.notna(row.get("national_employment")) and (
            row["national_employment"] >= SUPPRESSION_SUSPICION_THRESHOLD
        ):
            return "suppressed?"
        return "small/absent"

    out["local_status"] = out.apply(status, axis=1)
    out["supervisory"] = out["soc"].map(is_first_line_supervisor)
    return out.sort_values(
        ["local_employment", "national_employment"], ascending=False, na_position="last"
    ).reset_index(drop=True)


def progression_table(
    transitions: pd.DataFrame,
    soc: str,
    metro_wide: pd.DataFrame,
    metro: str,
    crosswalk_socs: set | None = None,
    top: int = 12,
) -> pd.DataFrame:
    """Where people in this occupation actually go next, with local pay.

    Observed CPS moves, not model predictions: a state auditor can check this
    against the public microdata, which a fitted model's output does not allow.

    `raw_count` is reported because CPS ASEC transitions are thin at occupation
    level. A destination resting on three respondents is not a finding and the
    table should not hide that.
    """
    moves = transitions[
        (transitions["soc_from"] == soc) & (transitions["soc_to"] != soc)
    ]
    if moves.empty:
        return pd.DataFrame()

    grouped = (
        moves.groupby("soc_to")
        .agg(weighted=("weighted_count", "sum"), raw_count=("raw_count", "sum"))
        .sort_values("weighted", ascending=False)
    )
    grouped["share"] = grouped["weighted"] / grouped["weighted"].sum()

    local = (
        metro_wide[metro_wide["metro"] == metro]
        .loc[:, ["soc6", "annual_median_wage", "employment"]]
        .rename(
            columns={
                "soc6": "soc_to",
                "annual_median_wage": "dest_local_wage",
                "employment": "dest_local_employment",
            }
        )
    )
    out = grouped.reset_index().merge(local, on="soc_to", how="left")
    out["aggregated_cps_code"] = out["soc_to"].map(is_aggregated_soc)
    if crosswalk_socs is not None:
        out["in_crosswalk"] = out["soc_to"].isin(crosswalk_socs)
    return out.head(top)


def coverage_summary(
    placement: pd.DataFrame,
    progression: pd.DataFrame,
) -> dict:
    """The numbers worth quoting, with their denominators."""
    national_total = placement["national_employment"].sum()
    unpublished = placement.loc[
        placement["local_status"] != "published", "national_employment"
    ].sum()
    published = int((placement["local_status"] == "published").sum())

    summary = {
        "candidates": len(placement),
        "published_locally": published,
        "national_employment_all_candidates": float(national_total),
        "national_employment_not_visible_locally": float(unpublished),
        "share_not_visible_locally": float(unpublished / national_total)
        if national_total
        else float("nan"),
        "largest_invisible": None,
    }
    hidden = placement[placement["local_status"] == "suppressed?"]
    if len(hidden):
        row = hidden.nlargest(1, "national_employment").iloc[0]
        summary["largest_invisible"] = {
            "soc": row["soc"],
            "title": row["soc_title"],
            "national_employment": float(row["national_employment"]),
        }

    if progression is not None and len(progression):
        summary["observed_moves_sample"] = int(progression["raw_count"].sum())
        if "in_crosswalk" in progression:
            mass = progression["share"].sum()
            summary["progression_in_crosswalk_share"] = float(
                progression.loc[progression["in_crosswalk"], "share"].sum() / mass
            )
    return summary

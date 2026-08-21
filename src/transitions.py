"""Turn an IPUMS CPS ASEC extract into observed occupation-to-occupation moves.

This is the y side of the modelling problem, refactored out of
`notebooks/03_ipums_transitions.ipynb` so the model code and the notebook
can't drift apart.

Two things differ from the notebook version, both deliberate:

  1. `year` is carried through the aggregation instead of being collapsed
     away. Without it there is no way to hold out time, and holding out time
     is the only honest way to validate this model (see `gravity.py`).
  2. Destination employment size is computed from the same extract. It is the
     single most important control in the model -- most moves land in large
     occupations, and a model that isn't told how big each destination is will
     simply rediscover the size distribution of the labour market and call it
     a finding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from soc import load_crosswalk

# EMPSTAT: 10 = at work, 12 = has job, not at work last week.
EMPLOYED = (10, 12)
# OCC / OCCLY sentinels for "not applicable" and "unknown".
INVALID_OCC = {0, 9999}

IPUMS_COLUMNS = [
    "YEAR", "CPSIDP", "ASECWT", "WTFINL", "EMPSTAT",
    "OCC", "OCCLY", "IND", "INDLY", "WKSWORK1",
    # Requested by the API extract for later phases; harmless if absent.
    "AGE", "SEX", "EDUC", "INCWAGE",
]

# ASEC samples must be weighted with ASECWT. WTFINL is the basic *monthly*
# weight and is not the right weight for the March supplement -- which is
# where OCCLY (occupation last year) lives, i.e. the entire transition
# signal. Prefer ASECWT and fall back only if the extract predates it.
WEIGHT_PREFERENCE = ["asecwt", "wtfinl"]


def load_ipums(path, usecols=None) -> pd.DataFrame:
    """Read an IPUMS CPS extract, lowercase the columns, keep what we need."""
    usecols = set(usecols or IPUMS_COLUMNS)
    raw = pd.read_csv(path, usecols=lambda c: c.upper() in usecols, low_memory=False)
    raw.columns = raw.columns.str.lower()
    return raw


def resolve_weight(frame: pd.DataFrame) -> str:
    """Name of the person weight column to use, preferring the ASEC weight."""
    for candidate in WEIGHT_PREFERENCE:
        if candidate in frame.columns:
            return candidate
    raise KeyError(
        "no person weight column found; expected one of "
        f"{WEIGHT_PREFERENCE}, got {sorted(frame.columns)}"
    )


def _zero_pad(series: pd.Series) -> pd.Series:
    return series.astype("Int64").astype(str).str.zfill(4)


def extract_moves(raw: pd.DataFrame, crosswalk_path) -> pd.DataFrame:
    """Person-year records that record a genuine occupation change, SOC-mapped.

    Keeps one row per surveyed person-year, carrying the survey weight. The
    caller aggregates; keeping it long here means industry and year stay
    available for feature building.
    """
    df = raw[
        raw["empstat"].isin(EMPLOYED)
        & raw["occ"].notna()
        & raw["occly"].notna()
        & ~raw["occ"].isin(INVALID_OCC)
        & ~raw["occly"].isin(INVALID_OCC)
    ].copy()

    df["occ_code"] = _zero_pad(df["occ"])
    df["occly_code"] = _zero_pad(df["occly"])

    xwalk = load_crosswalk(crosswalk_path)
    code_to_soc = xwalk.set_index("census_occ_code")["soc6"].to_dict()

    df["soc_to"] = df["occ_code"].map(code_to_soc)
    df["soc_from"] = df["occly_code"].map(code_to_soc)
    df = df.dropna(subset=["soc_from", "soc_to"])

    # Same-industry moves are far more common than cross-industry ones, and
    # IND/INDLY are already in the standard extract -- this is free signal the
    # original notebook pulled and never used.
    if {"ind", "indly"}.issubset(df.columns):
        df["same_industry"] = (df["ind"] == df["indly"]).astype(float)
    else:
        df["same_industry"] = np.nan

    # A move at SOC-6 level. Census OCC codes are finer than SOC-6 in places,
    # so a changed OCC can still be the same SOC-6; that is not a transition.
    df["moved"] = (df["soc_from"] != df["soc_to"]).astype(int)
    # Normalise the weight column name so downstream code never has to care
    # which weight the extract carried.
    df["weight"] = df[resolve_weight(df)].astype(float)
    return df


def aggregate_transitions(moves: pd.DataFrame) -> pd.DataFrame:
    """Weighted (year, soc_from, soc_to) counts for genuine moves."""
    changed = moves[moves["moved"] == 1]
    agg = (
        changed.groupby(["year", "soc_from", "soc_to"], as_index=False)
        .agg(
            weighted_count=("weight", "sum"),
            raw_count=("weight", "size"),
            same_industry_share=("same_industry", "mean"),
        )
    )
    return agg


def destination_size(moves: pd.DataFrame) -> pd.DataFrame:
    """Weighted employment in each occupation, per year.

    Counted over every employed person in the extract, not just movers --
    this is the size of the destination pool, which is what a mover is
    choosing among.
    """
    return (
        moves.groupby(["year", "soc_to"], as_index=False)["weight"]
        .sum()
        .rename(columns={"soc_to": "soc6", "weight": "employment"})
    )


def origin_outflow(transitions: pd.DataFrame) -> pd.DataFrame:
    """Total weighted movers leaving each origin occupation, per year."""
    return (
        transitions.groupby(["year", "soc_from"], as_index=False)["weighted_count"]
        .sum()
        .rename(columns={"weighted_count": "outflow"})
    )

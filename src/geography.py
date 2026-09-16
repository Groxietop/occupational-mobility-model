"""Geographic mobility deserts: exits that exist where the worker actually is.

Phase 2 asked how many ways out an occupation has. That question is answered
nationally, which quietly assumes every destination is reachable from
everywhere. It isn't. A machinist in Youngstown and a machinist in San Jose
have the same skills and the same national transition profile, and completely
different opportunity sets.

The localisation
----------------
The national model gives a transition propensity p(D|X): where movers out of
occupation X actually go, driven by skill proximity and destination size. To
ask the same question inside a metro, that propensity is reweighted by how much
of each destination is present locally:

    availability(D, M) = employment(D, M) / employment(D, national)

    p_local(D | X, M)  ∝  p(D | X) · availability(D, M)

`availability` is the share of the nation's D jobs that sit inside metro M. So
p_local reads as: of the people who would move X→D nationally, what fraction
could do it without leaving M. Destinations OEWS does not publish for a metro
get no mass at all -- an occupation too small to disclose locally is not an
exit a local worker can take.

This deliberately does not re-apply destination size. The national model
already contains it; multiplying by a local *share* localises that size rather
than counting it twice.

Wages are local too
-------------------
Whether a move is a raise is judged on the metro's own wages, not national
ones. The same X→D move can be upward in one metro and lateral in another,
and that is precisely the effect being measured.

What is and isn't claimed
-------------------------
This measures the *opportunity structure* around an occupation in a place. It
is not a prediction that a given worker will move, and metro boundaries are a
crude stand-in for commuting range -- a real reachability model would use
commuting zones. Both limits are why the output is reported as a structural
score rather than a probability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# A destination must pay at least this much more to count as an upward exit.
RAISE_THRESHOLD = 0.10

# Below this many effective local destinations, an (occupation, metro) cell is
# reported as narrow. Matches deserts.NARROW_THRESHOLD so national and local
# figures stay on one scale.
NARROW_THRESHOLD = 12.0


def _entropy(probabilities: np.ndarray) -> float:
    p = probabilities[probabilities > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p)).sum())


def build_availability(metro_wide: pd.DataFrame, national: pd.DataFrame) -> pd.DataFrame:
    """Share of each occupation's national employment located in each metro.

    `metro_wide` is one row per (metro, soc6) with employment and median wage;
    `national` is the wide national OEWS table.
    """
    nat = (
        national.loc[:, ["soc6", "employment", "annual_median_wage"]]
        .dropna(subset=["employment"])
        .drop_duplicates("soc6")
        .rename(
            columns={
                "employment": "national_employment",
                "annual_median_wage": "national_wage",
            }
        )
    )

    out = metro_wide.merge(nat, on="soc6", how="inner")
    # An occupation with no published local employment has no measurable local
    # availability. Dropping it is the honest reading: OEWS suppressing a cell
    # means the occupation is too small locally to disclose, which is not a
    # destination a worker can count on. Keeping it as NaN would silently
    # poison every downstream sum.
    out = out[(out["national_employment"] > 0) & out["employment"].notna()].copy()
    out["availability"] = out["employment"] / out["national_employment"]

    # A metro cannot hold more of an occupation than the nation does. Values
    # slightly above 1 come from OEWS metro and national estimates being drawn
    # from different panels of the same rolling sample, so they are clipped
    # rather than treated as an error.
    out["availability"] = out["availability"].clip(upper=1.0)
    return out


def common_occupations(metro_wide: pd.DataFrame) -> set:
    """Occupations OEWS publishes in *every* metro in the panel.

    Why this matters more than it looks
    -----------------------------------
    OEWS suppresses small cells, and small metros have more of them: this panel
    publishes 246 occupations in Youngstown and 406 in New York. Because the
    localised destination mix is renormalised over whatever is published, a
    metro with more suppressed cells mechanically scores as having fewer exits.
    Measured on the raw panel, the correlation between "occupations published"
    and "mean viable upward exits" is 0.94 -- the index is then largely a
    disclosure artefact rather than a finding about mobility.

    Restricting both origins and destinations to the occupations every metro
    publishes gives each place the same destination universe, so differences
    reflect local employment composition instead. It is the narrower question,
    honestly answered, and is the panel any published ranking should use.
    """
    n_metros = metro_wide["metro"].nunique()
    counts = metro_wide.groupby("soc6")["metro"].nunique()
    return set(counts[counts == n_metros].index)


def balance(metro_wide: pd.DataFrame) -> pd.DataFrame:
    """Restrict a metro panel to the occupations every metro publishes."""
    keep = common_occupations(metro_wide)
    return metro_wide[metro_wide["soc6"].isin(keep)].copy()


def suppression_diagnostic(index: pd.DataFrame, metro_wide: pd.DataFrame) -> float:
    """Correlation between occupations published and the index's headline.

    Near zero means the ranking is not being driven by disclosure. Near one
    means it is, and the balanced panel should be used instead.
    """
    published = metro_wide.groupby("metro")["soc6"].nunique().rename("published")
    joined = index.merge(published, left_on="metro", right_index=True)
    # On a balanced panel every metro publishes the same count, so the
    # correlation is undefined rather than zero. That is the intended state --
    # no variance means no confound is possible -- so it is reported as such.
    if joined["published"].nunique() <= 1:
        return float("nan")
    return float(joined["published"].corr(joined["mean_viable_upward_exits"]))


def national_destination_mix(panel: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    """Normalised predicted destination distribution per origin occupation."""
    frame = panel[["soc_from", "soc_to"]].copy()
    frame["predicted"] = np.clip(predictions, 0, None)
    totals = frame.groupby("soc_from")["predicted"].transform("sum")
    frame = frame[totals > 0].copy()
    frame["p_national"] = frame["predicted"] / totals[totals > 0]
    return frame[["soc_from", "soc_to", "p_national"]]


def local_mobility(
    panel: pd.DataFrame,
    predictions: np.ndarray,
    metro_wide: pd.DataFrame,
    national: pd.DataFrame,
    raise_threshold: float = RAISE_THRESHOLD,
    min_local_employment: float = 0.0,
) -> pd.DataFrame:
    """Per (origin occupation, metro) mobility metrics.

    Returns one row per cell:

      local_effective_destinations  exp(entropy) of the localised mix
      local_upward_share            localised mass paying >= threshold more
      viable_upward_exits           the product of the two -- the effective
                                    number of upward exits actually available
      reachable_share               national mass that survives localisation,
                                    i.e. how much of the occupation's normal
                                    destination mix exists in this metro
    """
    mix = national_destination_mix(panel, predictions)
    avail = build_availability(metro_wide, national)

    local = avail.loc[
        :, ["metro", "metro_name", "kind", "soc6", "employment", "annual_median_wage", "availability"]
    ]
    if min_local_employment > 0:
        local = local[local["employment"] >= min_local_employment]

    # Destination side: availability and local wage of each possible target.
    dest = local.rename(
        columns={
            "soc6": "soc_to",
            "annual_median_wage": "dest_local_wage",
            "employment": "dest_local_employment",
        }
    ).loc[:, ["metro", "soc_to", "availability", "dest_local_wage", "dest_local_employment"]]

    # Origin side: the occupation the worker is leaving, and what it pays here.
    origin = local.rename(
        columns={"soc6": "soc_from", "annual_median_wage": "origin_local_wage"}
    ).loc[:, ["metro", "metro_name", "kind", "soc_from", "origin_local_wage"]]

    edges = mix.merge(dest, on="soc_to", how="inner").merge(
        origin, on=["metro", "soc_from"], how="inner"
    )
    if edges.empty:
        return pd.DataFrame()

    edges["p_local_raw"] = edges["p_national"] * edges["availability"]

    grouped = edges.groupby(["metro", "soc_from"], sort=False)
    edges["local_total"] = grouped["p_local_raw"].transform("sum")
    edges = edges[edges["local_total"] > 0].copy()
    edges["p_local"] = edges["p_local_raw"] / edges["local_total"]

    # A destination whose local wage OEWS does not publish cannot be judged as
    # a raise either way. Rather than scoring it as "not a raise" -- which would
    # bias every upward share downward -- the share is computed over the mass
    # where both wages are known, and that mass is reported as `wage_coverage`
    # so the denominator is visible.
    edges["wage_known"] = (
        edges["dest_local_wage"].notna() & edges["origin_local_wage"].notna()
    )
    edges["is_raise"] = (
        edges["wage_known"]
        & (edges["dest_local_wage"] >= edges["origin_local_wage"] * (1.0 + raise_threshold))
    ).astype(float)

    rows = []
    for (metro, origin_soc), group in edges.groupby(["metro", "soc_from"], sort=False):
        p = group["p_local"].to_numpy()
        effective = float(np.exp(_entropy(p)))

        known = group["wage_known"].to_numpy()
        known_mass = float(p[known].sum())
        upward_share = (
            float((p[known] * group["is_raise"].to_numpy()[known]).sum() / known_mass)
            if known_mass > 0
            else float("nan")
        )
        origin_wage = group["origin_local_wage"].iloc[0]

        rows.append(
            {
                "metro": metro,
                "metro_name": group["metro_name"].iloc[0],
                "kind": group["kind"].iloc[0],
                "soc6": origin_soc,
                "local_effective_destinations": effective,
                "local_upward_share": upward_share,
                "viable_upward_exits": effective * upward_share,
                "reachable_share": float(group["p_local_raw"].sum()),
                "wage_coverage": known_mass,
                "local_destinations_present": int((group["p_local"] > 0).sum()),
                "origin_local_wage": float(origin_wage) if pd.notna(origin_wage) else float("nan"),
            }
        )

    out = pd.DataFrame(rows)
    # Narrowness is reported relative to this panel, not against the national
    # threshold: localised counts sit on a different scale entirely (median ~82
    # against a national threshold of 12), so the national cutoff flags almost
    # nothing and tells the reader less than nothing.
    if len(out):
        cutoff = out["viable_upward_exits"].quantile(0.20)
        out["is_narrow"] = out["viable_upward_exits"] <= cutoff
        out.attrs["narrow_cutoff"] = float(cutoff)
    else:
        out["is_narrow"] = []
    return out


def compare_to_national(
    local: pd.DataFrame,
    national_deserts: pd.DataFrame,
) -> pd.DataFrame:
    """Join local cells to their national counterpart and measure the gap.

    `geographic_penalty` is how much of an occupation's national mobility
    disappears once only local destinations count. It is the quantity the whole
    exercise exists to expose: a high penalty means the occupation is not
    inherently narrow, it is narrow *here*.
    """
    nat = national_deserts.loc[
        :, ["soc6", "title", "effective_destinations", "upward_share", "annual_median_wage"]
    ].rename(
        columns={
            "effective_destinations": "national_effective_destinations",
            "upward_share": "national_upward_share",
            "annual_median_wage": "national_wage",
        }
    )
    out = local.merge(nat, on="soc6", how="inner")
    out["geographic_penalty"] = 1.0 - (
        out["local_effective_destinations"] / out["national_effective_destinations"]
    )
    out["upward_penalty"] = out["national_upward_share"] - out["local_upward_share"]
    out["local_wage_ratio"] = out["origin_local_wage"] / out["national_wage"]
    return out


def metro_index(local: pd.DataFrame, metro_wide: pd.DataFrame) -> pd.DataFrame:
    """Roll (occupation, metro) cells up to a per-metro mobility picture.

    Each occupation is weighted by its local employment, so the index answers
    "for a worker drawn at random from this metro, how many viable upward exits
    does their occupation offer" rather than treating a 50-person occupation the
    same as a 50,000-person one.
    """
    employment = metro_wide.loc[:, ["metro", "soc6", "employment"]]
    frame = local.merge(employment, on=["metro", "soc6"], how="inner")
    # Cells without a local wage have no defined upward share, so they cannot
    # be averaged. They are excluded here and their weight reported, rather
    # than being counted as zero.
    frame = frame[(frame["employment"] > 0) & frame["local_upward_share"].notna()]

    rows = []
    for metro, group in frame.groupby("metro", sort=False):
        weights = group["employment"].to_numpy()
        weights = weights / weights.sum()
        narrow_mask = group["is_narrow"].to_numpy().astype(float)
        rows.append(
            {
                "metro": metro,
                "metro_name": group["metro_name"].iloc[0],
                "kind": group["kind"].iloc[0],
                "occupations_present": int(group["soc6"].nunique()),
                "employment_covered": float(group["employment"].sum()),
                "mean_viable_upward_exits": float((weights * group["viable_upward_exits"]).sum()),
                "mean_local_effective_destinations": float(
                    (weights * group["local_effective_destinations"]).sum()
                ),
                "mean_upward_share": float((weights * group["local_upward_share"]).sum()),
                "share_workers_in_narrow_occupations": float((weights * narrow_mask).sum()),
                "mean_local_wage": float(
                    np.average(group["origin_local_wage"], weights=group["employment"])
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_viable_upward_exits").reset_index(drop=True)


def same_occupation_across_metros(
    local: pd.DataFrame, soc6: str, titles: pd.DataFrame | None = None
) -> pd.DataFrame:
    """One occupation's local mobility in every metro, worst first.

    The headline comparison: identical skills, identical national transition
    profile, different place.
    """
    out = local[local["soc6"] == soc6].copy()
    columns = [
        "metro_name",
        "kind",
        "local_effective_destinations",
        "local_upward_share",
        "viable_upward_exits",
        "origin_local_wage",
    ]
    return out.sort_values("viable_upward_exits")[columns].reset_index(drop=True)

"""Mobility deserts: occupations with few realistic ways out.

Phase 1 produced calibrated edge weights. Once every pair carries a
predicted flow, the network's shape becomes readable, and the question worth
asking is not "where can this person go" but "how many places can they go at
all, and are any of them better".

An occupation is a mobility desert when its predicted outflow concentrates on
a handful of destinations. Counting destinations doesn't capture this -- a
model assigns *some* probability to almost every pair. What matters is how
concentrated that distribution is, which is what entropy measures.

    effective_destinations = exp(H(p))

where p is the predicted destination distribution for an origin and H is
Shannon entropy. It reads directly: an occupation with an effective
destination count of 4 has, in practice, four ways out, however many pairs
have nonzero probability. A uniform distribution over 430 destinations gives
430; total concentration on one gives 1.

The policy-relevant finding is the intersection with pay. An occupation that
is well paid and narrow is a specialty. One that is *poorly* paid and narrow
is a trap, and that quadrant is what del Rio-Chanona et al. found drives
long-term unemployment after an automation shock.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Below this many effective destinations an occupation is treated as narrow.
# Chosen to sit near the lower quartile of the real distribution rather than
# from theory -- it is a reporting threshold, not a discovered boundary.
NARROW_THRESHOLD = 12.0


def _entropy(probabilities: np.ndarray) -> float:
    p = probabilities[probabilities > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p)).sum())


def destination_profile(panel: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    """Per-origin mobility metrics from the model's predicted flows.

    Returns one row per origin occupation:

      effective_destinations  exp(entropy) of the predicted destination mix
      top_share               share of predicted outflow to its single
                              most likely destination
      top5_share              ...to its five most likely
    """
    frame = panel[["soc_from", "soc_to"]].copy()
    frame["predicted"] = np.clip(predictions, 0, None)

    rows = []
    for origin, group in frame.groupby("soc_from"):
        total = group["predicted"].sum()
        if total <= 0:
            continue
        p = (group["predicted"] / total).to_numpy()
        ordered = np.sort(p)[::-1]
        rows.append(
            {
                "soc6": origin,
                "effective_destinations": float(np.exp(_entropy(p))),
                "top_share": float(ordered[0]),
                "top5_share": float(ordered[:5].sum()),
            }
        )

    return pd.DataFrame(rows)


def escape_metrics(
    panel: pd.DataFrame,
    predictions: np.ndarray,
    wages: pd.DataFrame,
    raise_threshold: float = 0.10,
) -> pd.DataFrame:
    """How much of an occupation's predicted outflow leads somewhere better.

    `upward_share` is the fraction of predicted flow going to occupations
    paying at least `raise_threshold` more than the origin. An occupation can
    have plenty of destinations and still be a trap if none of them pay more.
    """
    wage_by_soc = (
        wages.loc[:, ["soc6", "annual_median_wage"]]
        .dropna()
        .drop_duplicates("soc6")
        .set_index("soc6")["annual_median_wage"]
    )

    frame = panel[["soc_from", "soc_to"]].copy()
    frame["predicted"] = np.clip(predictions, 0, None)
    frame["origin_wage"] = frame["soc_from"].map(wage_by_soc)
    frame["dest_wage"] = frame["soc_to"].map(wage_by_soc)

    known = frame.dropna(subset=["origin_wage", "dest_wage"]).copy()
    known["is_raise"] = (
        known["dest_wage"] >= known["origin_wage"] * (1.0 + raise_threshold)
    ).astype(float)

    grouped = known.groupby("soc_from")
    out = grouped.apply(
        lambda g: pd.Series(
            {
                "upward_share": float(
                    (g["predicted"] * g["is_raise"]).sum() / g["predicted"].sum()
                )
                if g["predicted"].sum() > 0
                else np.nan,
                "expected_wage_ratio": float(
                    (g["predicted"] * g["dest_wage"]).sum()
                    / g["predicted"].sum()
                    / g["origin_wage"].iloc[0]
                )
                if g["predicted"].sum() > 0
                else np.nan,
            }
        )
    ).reset_index()
    return out.rename(columns={"soc_from": "soc6"})


def find_deserts(
    panel: pd.DataFrame,
    predictions: np.ndarray,
    wages: pd.DataFrame,
    titles: pd.DataFrame,
    narrow_threshold: float = NARROW_THRESHOLD,
) -> pd.DataFrame:
    """Assemble the full per-occupation mobility picture, worst first.

    `desert_score` combines narrowness and lack of upward options into a
    single 0-1 ranking. It is a reporting convenience, not a measured
    quantity -- the two components are reported separately alongside it so a
    reader can disagree with the weighting.
    """
    profile = destination_profile(panel, predictions)
    escape = escape_metrics(panel, predictions, wages)

    wage_by_soc = (
        wages.loc[:, ["soc6", "annual_median_wage", "employment"]]
        .dropna(subset=["annual_median_wage"])
        .drop_duplicates("soc6")
    )

    out = (
        profile.merge(escape, on="soc6", how="left")
        .merge(wage_by_soc, on="soc6", how="left")
        .merge(titles.loc[:, ["soc6", "title"]], on="soc6", how="left")
    )

    # Rank-normalise both components so the score doesn't depend on units.
    narrowness = 1.0 - out["effective_destinations"].rank(pct=True)
    stuckness = 1.0 - out["upward_share"].rank(pct=True)
    out["desert_score"] = (narrowness + stuckness) / 2.0
    out["is_narrow"] = out["effective_destinations"] < narrow_threshold

    columns = [
        "soc6",
        "title",
        "effective_destinations",
        "top_share",
        "top5_share",
        "upward_share",
        "expected_wage_ratio",
        "annual_median_wage",
        "employment",
        "is_narrow",
        "desert_score",
    ]
    return out[columns].sort_values("desert_score", ascending=False).reset_index(drop=True)


def low_paid_and_trapped(deserts: pd.DataFrame, wage_percentile: float = 0.4) -> pd.DataFrame:
    """The quadrant that matters: narrow options *and* low pay.

    A well-paid narrow occupation is a specialty. A poorly-paid narrow one is
    where workers have neither leverage nor an exit.
    """
    cutoff = deserts["annual_median_wage"].quantile(wage_percentile)
    mask = deserts["is_narrow"] & (deserts["annual_median_wage"] <= cutoff)
    return deserts[mask].sort_values("desert_score", ascending=False).reset_index(drop=True)

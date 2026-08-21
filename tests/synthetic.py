"""Synthetic transition data with a known ground truth, for tests only.

None of this is real labour market data and none of it is committed as such.
Its purpose is to let the estimator be checked against an answer we already
know: flows are generated from a gravity process with chosen domain weights,
so a correct implementation has to recover those weights' ordering. Testing
against real CPS data could only ever tell us the code runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_transitions(
    pair_features: pd.DataFrame,
    domain_weights: dict[str, float],
    years: list[int],
    size_coefficient: float = 0.8,
    seed: int = 0,
    scale: float = 0.01,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate (transitions, sizes) from a known gravity process.

    Returns frames shaped exactly like `transitions.aggregate_transitions`
    and `transitions.destination_size` produce from a real IPUMS extract.

    The default `scale` is tuned so that roughly 5-8% of ordered pairs see a
    move in a given year, which is the order of sparsity a real CPS ASEC
    extract produces over ~430 reachable occupations. Testing at a denser
    setting would flatter the estimator: the whole difficulty here is that
    the overwhelming majority of cells are zero.
    """
    rng = np.random.default_rng(seed)

    occupations = sorted(
        set(pair_features["soc_from"]) | set(pair_features["soc_to"])
    )
    # Employment is lognormal, which is roughly how occupation sizes behave.
    employment = pd.Series(
        np.exp(rng.normal(loc=10.0, scale=1.1, size=len(occupations))),
        index=occupations,
    )
    # Origins differ in how many leavers they produce, independent of size.
    origin_effect = pd.Series(
        rng.normal(loc=0.0, scale=0.5, size=len(occupations)), index=occupations
    )

    log_mu = (
        pair_features["soc_from"].map(origin_effect).to_numpy()
        + size_coefficient * np.log(pair_features["soc_to"].map(employment).to_numpy())
    )
    for domain, weight in domain_weights.items():
        column = f"sim__{domain}"
        if column in pair_features.columns:
            log_mu = log_mu + weight * pair_features[column].to_numpy()

    # Centre so the flow scale doesn't depend on the weight magnitudes chosen.
    log_mu = log_mu - log_mu.mean() + np.log(scale)
    mu = np.exp(np.clip(log_mu, -20, 12))

    transition_frames = []
    for year in years:
        counts = rng.poisson(mu)
        keep = counts > 0
        transition_frames.append(
            pd.DataFrame(
                {
                    "year": year,
                    "soc_from": pair_features["soc_from"].to_numpy()[keep],
                    "soc_to": pair_features["soc_to"].to_numpy()[keep],
                    "weighted_count": counts[keep].astype(float),
                    "raw_count": counts[keep].astype(float),
                    "same_industry_share": rng.uniform(0, 1, size=int(keep.sum())),
                }
            )
        )

    transitions = pd.concat(transition_frames, ignore_index=True)

    sizes = pd.concat(
        [
            pd.DataFrame(
                {"year": year, "soc6": occupations, "employment": employment.to_numpy()}
            )
            for year in years
        ],
        ignore_index=True,
    )

    return transitions, sizes

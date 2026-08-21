"""Build the pair-level feature table: one row per ordered occupation pair.

This is the X side of the modelling problem. For every ordered pair of
occupations (origin i, destination j) it produces:

  - one cosine similarity per O*NET domain (8 of them), computed the same way
    `similarity.py` does, but at SOC-6 level so the rows line up with CPS
  - job zone delta and education-rank delta, signed, origin -> destination
  - whether O*NET's own curated "Related Occupations" list contains the move

The signed deltas matter more than absolute distance: moving *down* an
education level is a very different act from moving *up* one, and a model
given only |delta| cannot tell those apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from similarity import DOMAINS
from soc import to_soc6

SIM_PREFIX = "sim__"


def _domain_similarity(frame: pd.DataFrame, prefix: str) -> np.ndarray | None:
    """Z-scored cosine similarity matrix for one O*NET domain."""
    cols = [c for c in frame.columns if c.startswith(prefix)]
    if not cols:
        return None
    X = frame[cols].astype(float)
    X = X.fillna(X.mean())
    # A domain where every occupation scores identically carries no signal and
    # would divide by zero in the scaler; drop those columns first.
    keep = X.std(axis=0) > 0
    X = X.loc[:, keep]
    if X.shape[1] == 0:
        return None
    return cosine_similarity(StandardScaler().fit_transform(X))


def related_pairs(master: pd.DataFrame) -> set[tuple[str, str]]:
    """O*NET's own curated related-occupation pairs, as SOC-6 tuples.

    O*NET publishes, per occupation, a handful of occupations judged to use
    similar enough skills that a worker could move with minimal extra
    preparation. It is an expert judgment rather than an observed outcome,
    which makes it both a useful feature and the natural baseline to beat.
    """
    out: set[tuple[str, str]] = set()
    if "primary_related_soc_codes" not in master.columns:
        return out
    for _, row in master.iterrows():
        raw = row.get("primary_related_soc_codes")
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            continue
        src = to_soc6(row["soc_code"])
        for token in str(raw).split(","):
            token = token.strip()
            if token:
                out.add((src, to_soc6(token)))
    return out


def build_pair_features(
    soc6_master: pd.DataFrame,
    universe: list[str] | None = None,
    onet_related: set[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """One row per ordered (soc_from, soc_to) pair, excluding self-pairs.

    `universe` restricts the occupation set -- pass the SOC-6 codes that
    actually appear in CPS so the table doesn't carry 400k pairs that can
    never be observed.
    """
    frame = soc6_master
    if universe is not None:
        frame = frame[frame["soc6"].isin(set(universe))]
    frame = frame.sort_values("soc6").reset_index(drop=True)

    codes = frame["soc6"].to_numpy()
    n = len(codes)
    if n < 2:
        raise ValueError(f"need at least 2 occupations to form pairs, got {n}")

    sims: dict[str, np.ndarray] = {}
    for domain, prefix in DOMAINS.items():
        mat = _domain_similarity(frame, prefix)
        if mat is not None:
            sims[domain] = mat

    if not sims:
        raise ValueError("no O*NET domain columns found on the master frame")

    # Every ordered pair, then drop the diagonal.
    i_idx, j_idx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    i_flat, j_flat = i_idx.ravel(), j_idx.ravel()
    off_diagonal = i_flat != j_flat
    i_flat, j_flat = i_flat[off_diagonal], j_flat[off_diagonal]

    data = {"soc_from": codes[i_flat], "soc_to": codes[j_flat]}
    for domain, mat in sims.items():
        data[f"{SIM_PREFIX}{domain}"] = mat[i_flat, j_flat]

    pairs = pd.DataFrame(data)

    # Signed deltas: positive means the destination sits higher than the origin.
    for col, name in (("job_zone", "d_job_zone"), ("education_rank", "d_education")):
        values = frame[col].to_numpy(dtype=float)
        pairs[name] = values[j_flat] - values[i_flat]

    if onet_related:
        keys = list(zip(pairs["soc_from"], pairs["soc_to"]))
        pairs["onet_related"] = np.fromiter(
            (1.0 if k in onet_related else 0.0 for k in keys),
            dtype=float,
            count=len(keys),
        )
    else:
        pairs["onet_related"] = 0.0

    return pairs


def similarity_columns(pairs: pd.DataFrame) -> list[str]:
    return sorted(c for c in pairs.columns if c.startswith(SIM_PREFIX))


def add_wage_features(pairs: pd.DataFrame, oews_wide: pd.DataFrame) -> pd.DataFrame:
    """Attach the OEWS wage gap between origin and destination.

    `d_log_wage` is log(median wage at destination) - log(median wage at
    origin): positive means the move is a raise. Logs rather than levels
    because a $10k gap means something very different at $30k than at $200k.

    OEWS suppresses estimates for occupations too small to publish, so some
    pairs have no wage gap. Those are left as NaN here and imputed with a
    missingness flag at model time, the same way the O*NET gaps are -- an
    unknown wage gap is its own state, not a zero one.
    """
    wages = (
        oews_wide.loc[:, ["soc6", "annual_median_wage"]]
        .dropna()
        .drop_duplicates("soc6")
        .set_index("soc6")["annual_median_wage"]
    )
    positive = wages[wages > 0]
    log_wage = np.log(positive)

    out = pairs.copy()
    origin = out["soc_from"].map(log_wage)
    destination = out["soc_to"].map(log_wage)
    out["d_log_wage"] = destination - origin
    return out

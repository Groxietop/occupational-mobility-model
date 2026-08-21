"""A gravity model of occupational mobility.

Occupational transitions have the same shape as trade or migration flows:
large destinations attract more movers, and "distance" suppresses flow. That
makes the gravity model the natural specification --

    E[moves i->j] = exp( origin_i + b_size * log(employment_j)
                         + sum_d  w_d * similarity_d(i, j)
                         + other pair terms )

and fitting it by Poisson pseudo-maximum likelihood (PPML) is the standard
estimator, because it handles the enormous number of zero-flow pairs natively
rather than requiring them to be dropped or log-shifted. With ~430 reachable
occupations there are ~184k ordered pairs and only a few thousand are ever
observed; an estimator that can't take a zero is not usable here.

Two structural pieces do the real work:

  origin fixed effects   absorb "how many people leave occupation i at all",
                         so the pair terms are identified purely off *where*
                         leavers go, not how many there are.

  log(employment_j)      is the base-rate control. Without it the model
                         rediscovers that big occupations absorb lots of
                         people and reports that as insight.

What comes out is `w_d` -- the learned weight on each O*NET domain. That
vector is the phase-1 finding: it replaces the equal-weighted average in
`similarity.py` with weights fit against six years of moves people actually
made.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler

from pairs import similarity_columns

# Pair-level terms that aren't per-domain similarities.
EXTRA_FEATURES = ["d_job_zone", "d_education", "same_industry", "onet_related"]

# Occupation employment is heavily right-skewed and shows up in the gravity
# specification in logs; this floor keeps log() finite for tiny cells.
MIN_EMPLOYMENT = 1.0


@dataclass
class GravityFit:
    """A fitted gravity model plus everything needed to interpret it."""

    model: PoissonRegressor
    scaler: StandardScaler
    feature_names: list[str]
    origins: list[str]
    train_years: list[int]
    test_years: list[int]
    coefficients: pd.DataFrame = field(default_factory=pd.DataFrame)
    converged: bool = True
    n_iter: int = 0

    def domain_weights(self) -> pd.DataFrame:
        """Learned weight per O*NET domain, largest first.

        Coefficients are on standardised features, so they are directly
        comparable to each other -- that comparison is the whole point.
        """
        weights = self.coefficients[
            self.coefficients["feature"].str.startswith("sim__")
        ].copy()
        weights["domain"] = weights["feature"].str.replace("sim__", "", regex=False)
        weights = weights.sort_values("coefficient", ascending=False)
        total = weights["coefficient"].clip(lower=0).sum()
        weights["share"] = (
            weights["coefficient"].clip(lower=0) / total if total > 0 else np.nan
        )
        return weights[["domain", "coefficient", "share"]].reset_index(drop=True)


def build_panel(
    pair_features: pd.DataFrame,
    transitions: pd.DataFrame,
    sizes: pd.DataFrame,
    years: list[int],
) -> pd.DataFrame:
    """Join pair features to observed flows for one span of years.

    Every ordered pair gets a row, whether or not a move was ever observed --
    the zeros carry as much information as the positives, and PPML can use
    them.
    """
    flows = (
        transitions[transitions["year"].isin(years)]
        .groupby(["soc_from", "soc_to"], as_index=False)
        .agg(
            weighted_count=("weighted_count", "sum"),
            raw_count=("raw_count", "sum"),
            same_industry=("same_industry_share", "mean"),
        )
    )

    employment = (
        sizes[sizes["year"].isin(years)]
        .groupby("soc6", as_index=False)["employment"]
        .mean()
    )

    panel = pair_features.merge(flows, on=["soc_from", "soc_to"], how="left")
    panel["weighted_count"] = panel["weighted_count"].fillna(0.0)
    panel["raw_count"] = panel["raw_count"].fillna(0.0)
    # An unobserved pair has no industry mix to average; 0 is the right prior
    # here -- we never saw anyone make this move within an industry.
    panel["same_industry"] = panel["same_industry"].fillna(0.0)

    panel = panel.merge(
        employment.rename(columns={"soc6": "soc_to", "employment": "dest_employment"}),
        on="soc_to",
        how="left",
    )
    panel["dest_employment"] = panel["dest_employment"].fillna(MIN_EMPLOYMENT)
    panel["log_dest_employment"] = np.log(
        panel["dest_employment"].clip(lower=MIN_EMPLOYMENT)
    )

    # Keep only origins that actually appear as a source of movers in this
    # span; an origin with no outflow contributes no identifying variation.
    live_origins = set(flows["soc_from"])
    panel = panel[panel["soc_from"].isin(live_origins)].reset_index(drop=True)
    return panel


def _design(
    panel: pd.DataFrame,
    origins: list[str],
    feature_names: list[str],
    scaler: StandardScaler | None,
) -> tuple[sparse.csr_matrix, StandardScaler]:
    """Sparse design matrix: standardised pair terms + origin fixed effects."""
    dense = panel[feature_names].to_numpy(dtype=float)

    if scaler is None:
        scaler = StandardScaler()
        dense = scaler.fit_transform(dense)
    else:
        dense = scaler.transform(dense)

    index = {code: i for i, code in enumerate(origins)}
    rows = np.arange(len(panel))
    cols = panel["soc_from"].map(index).to_numpy()
    known = ~pd.isna(cols)
    fe = sparse.csr_matrix(
        (np.ones(known.sum()), (rows[known], cols[known].astype(int))),
        shape=(len(panel), len(origins)),
    )

    return sparse.hstack([sparse.csr_matrix(dense), fe], format="csr"), scaler


def _prepare(panel: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Impute the pair terms that O*NET leaves blank, and flag that we did.

    `job_zone` and `education_rank` are missing for ~10% of occupations. A
    missingness flag lets the model treat "unknown education gap" as its own
    state instead of silently reading it as "no gap".
    """
    out = panel.copy()
    for col in ("d_job_zone", "d_education"):
        if col in out.columns:
            flag = f"{col}__missing"
            if flag not in out.columns:
                out[flag] = out[col].isna().astype(float)
            out[col] = out[col].fillna(0.0)
    return out


def feature_list(panel: pd.DataFrame) -> list[str]:
    names = similarity_columns(panel) + ["log_dest_employment"]
    names += [c for c in EXTRA_FEATURES if c in panel.columns]
    names += [c for c in panel.columns if c.endswith("__missing")]
    return names


def fit_gravity(
    train_panel: pd.DataFrame,
    train_years: list[int],
    test_years: list[int],
    alpha: float = 1e-4,
    max_iter: int = 400,
) -> GravityFit:
    """Fit the PPML gravity model on one panel.

    `alpha` is a light L2 ridge -- enough to keep the fixed effects for
    thinly-observed origins from running away, small enough that it does not
    meaningfully shrink the pair coefficients we want to read.
    """
    prepared = _prepare(train_panel, [])
    names = feature_list(prepared)
    origins = sorted(prepared["soc_from"].unique())

    X, scaler = _design(prepared, origins, names, scaler=None)
    y = prepared["weighted_count"].to_numpy(dtype=float)

    model = PoissonRegressor(alpha=alpha, max_iter=max_iter, fit_intercept=True)
    # sklearn signals non-convergence with a ConvergenceWarning, which is easy
    # to lose in output. Capture it and hand the caller an explicit flag --
    # coefficients from a run that stopped at the iteration cap should not be
    # read as a finding.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X, y)
    converged = not any(
        issubclass(w.category, ConvergenceWarning) for w in caught
    )

    coefficients = pd.DataFrame(
        {"feature": names, "coefficient": model.coef_[: len(names)]}
    ).sort_values("coefficient", ascending=False, key=abs)

    return GravityFit(
        model=model,
        scaler=scaler,
        feature_names=names,
        origins=origins,
        train_years=list(train_years),
        test_years=list(test_years),
        coefficients=coefficients.reset_index(drop=True),
        converged=converged,
        n_iter=int(np.ravel(model.n_iter_)[0]),
    )


def predict(fit: GravityFit, panel: pd.DataFrame) -> np.ndarray:
    prepared = _prepare(panel, [])
    for name in fit.feature_names:
        if name not in prepared.columns:
            prepared[name] = 0.0
    X, _ = _design(prepared, fit.origins, fit.feature_names, scaler=fit.scaler)
    return fit.model.predict(X)

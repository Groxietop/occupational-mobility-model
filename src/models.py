"""Alternative model specifications, so "the gravity model" is a choice.

Phase 1 fit one specification. This module puts three genuine alternatives
next to it on the same held-out years, because the interesting question isn't
whether the gravity model works -- it's whether its structure is earning its
keep against a flexible learner and against a model that ignores O*NET
entirely.

    ppml_gravity     the phase-1 model: log-linear in the features, Poisson
    ppml_raw_counts  same, but fit on person counts rather than survey-
                     weighted counts (see "the weighting problem" below)
    boosted_poisson  gradient boosting with a Poisson loss. Same features, no
                     linearity assumption, no interpretable weight vector.
    flow_embedding   matrix factorisation of the observed transition graph.
                     Uses NO O*NET features at all -- only who moved where.

That last one is the load-bearing comparison. If a model that has never seen
an O*NET descriptor predicts as well as one built entirely from them, the
O*NET features aren't doing the work and the whole premise needs revisiting.

The weighting problem
---------------------
CPS gives every person a survey weight of ~1,000-3,000. Multiplying counts by
those weights produces numbers that are not counts: a single surveyed mover
becomes a "count" of 2,000. Poisson's variance assumption is then wildly
violated -- the weighted target has a variance-to-mean ratio around 39,000,
against ~20 for the underlying person counts.

PPML point estimates survive this (that is exactly the robustness Santos
Silva & Tenreyro argue for), which is why phase 1's rankings are usable. But
any standard error from it is meaningless, and it's worth seeing whether
fitting honest counts changes the answer. Hence `ppml_raw_counts`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor

import gravity
from gravity import _design, _prepare, feature_list

# Overdispersion above this makes the Poisson variance assumption a fiction.
# Reported, not acted on -- see the module docstring.
OVERDISPERSION_WARN = 2.0


@dataclass
class ModelResult:
    name: str
    predictions: np.ndarray
    notes: str = ""


def overdispersion(counts: np.ndarray) -> float:
    """Variance-to-mean ratio. Poisson assumes 1.0."""
    counts = np.asarray(counts, dtype=float)
    mean = counts.mean()
    return float(counts.var() / mean) if mean > 0 else float("nan")


def fit_ppml(train: pd.DataFrame, test: pd.DataFrame, target="weighted_count", **kwargs):
    """Phase-1 PPML gravity, optionally against a different target column."""
    if target != "weighted_count":
        train = train.assign(weighted_count=train[target])
    fit = gravity.fit_gravity(train, [], [], **kwargs)
    return fit, gravity.predict(fit, test)


def fit_boosted_poisson(
    train: pd.DataFrame,
    test: pd.DataFrame,
    max_iter: int = 400,
    learning_rate: float = 0.06,
    max_leaf_nodes: int = 31,
    seed: int = 0,
) -> np.ndarray:
    """Gradient boosting with a Poisson loss on the same pair features.

    The origin enters as its log total outflow rather than as a categorical.
    That is what the linear model's fixed effects were actually capturing --
    how many leavers an occupation produces -- and unlike a categorical it
    doesn't run into sklearn's 255-category cap once the universe passes ~255
    occupations. Outflow is computed on the training years only and carried
    over to test, so no held-out information leaks in.
    """
    train_prepared = _prepare(train, [])
    test_prepared = _prepare(test, [])
    names = feature_list(train_prepared)

    outflow = train_prepared.groupby("soc_from")["weighted_count"].sum()
    log_outflow = np.log1p(outflow)
    fallback = float(log_outflow.median()) if len(log_outflow) else 0.0

    def design(frame):
        X = frame[names].to_numpy(dtype=float)
        origin = frame["soc_from"].map(log_outflow).fillna(fallback).to_numpy(dtype=float)
        return np.column_stack([X, origin])

    model = HistGradientBoostingRegressor(
        loss="poisson",
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        random_state=seed,
    )
    model.fit(design(train_prepared), train_prepared["weighted_count"].to_numpy())
    return np.clip(model.predict(design(test_prepared)), 1e-9, None)


def fit_flow_embedding(
    train: pd.DataFrame,
    test: pd.DataFrame,
    components: int = 24,
    seed: int = 0,
) -> np.ndarray:
    """Truncated SVD of the observed origin-by-destination flow matrix.

    Pure collaborative filtering: occupations get latent positions learned
    from who actually moved where, with no reference to what the jobs involve.
    The reconstruction fills in plausible flows for pairs never observed, the
    same way a recommender infers unrated items.

    Fit on log1p of the flows because the raw matrix is dominated by a handful
    of very large cells and SVD would otherwise spend all its components on
    them.
    """
    origins = sorted(set(train["soc_from"]) | set(test["soc_from"]))
    destinations = sorted(set(train["soc_to"]) | set(test["soc_to"]))
    row = {code: i for i, code in enumerate(origins)}
    col = {code: i for i, code in enumerate(destinations)}

    matrix = np.zeros((len(origins), len(destinations)), dtype=float)
    rows = train["soc_from"].map(row).to_numpy()
    cols = train["soc_to"].map(col).to_numpy()
    matrix[rows, cols] = np.log1p(train["weighted_count"].to_numpy(dtype=float))

    rank = min(components, min(matrix.shape) - 1)
    svd = TruncatedSVD(n_components=rank, random_state=seed)
    reduced = svd.fit_transform(matrix)
    reconstructed = reduced @ svd.components_

    predicted = reconstructed[
        test["soc_from"].map(row).to_numpy(), test["soc_to"].map(col).to_numpy()
    ]
    # Undo the log1p and floor at a small positive value so the deviance is
    # finite for pairs the factorisation drives negative.
    return np.clip(np.expm1(predicted), 1e-9, None)


def fit_gravity_plus_history(
    train: pd.DataFrame,
    test: pd.DataFrame,
    max_iter: int = 20000,
):
    """The gravity model, plus how much flow this pair carried in the past.

    `flow_embedding` beating the feature-based model is not a bug: past flow
    on a pair is strongly autocorrelated, and the embedding gets to use it
    while the gravity model is restricted to explaining flows from occupation
    characteristics alone. This specification gives the gravity model the same
    information.

    It changes the question being asked. Features-only answers "what about two
    occupations makes movement between them likely" -- a structural claim that
    transfers to pairs never observed, which is what the cold-start deployment
    story needs. Adding history answers "what will flow next year", which is a
    forecast. Both are legitimate; they are not the same model and should not
    be read as competing on one axis.
    """
    history = (
        train.groupby(["soc_from", "soc_to"], as_index=False)["weighted_count"]
        .sum()
        .rename(columns={"weighted_count": "prior_flow"})
    )

    def attach(frame):
        out = frame.merge(history, on=["soc_from", "soc_to"], how="left")
        out["prior_flow"] = out["prior_flow"].fillna(0.0)
        out["log_prior_flow"] = np.log1p(out["prior_flow"])
        return out.drop(columns="prior_flow")

    train_h, test_h = attach(train), attach(test)

    original = gravity.EXTRA_FEATURES
    try:
        gravity.EXTRA_FEATURES = original + ["log_prior_flow"]
        fit = gravity.fit_gravity(train_h, [], [], max_iter=max_iter)
        return fit, gravity.predict(fit, test_h)
    finally:
        gravity.EXTRA_FEATURES = original


def run_all(
    train: pd.DataFrame,
    test: pd.DataFrame,
    max_iter: int = 20000,
) -> tuple[list[ModelResult], object]:
    """Fit every specification on `train`, predict `test`. Returns the PPML fit too."""
    results: list[ModelResult] = []

    ppml_fit, ppml_pred = fit_ppml(train, test, max_iter=max_iter)
    results.append(
        ModelResult(
            "ppml_gravity",
            ppml_pred,
            "phase-1 spec; log-linear, survey-weighted target",
        )
    )

    _, raw_pred = fit_ppml(train, test, target="raw_count", max_iter=max_iter)
    # Rescale from person counts back onto the weighted scale so deviance is
    # comparable across rows of the scorecard.
    scale = train["weighted_count"].sum() / max(train["raw_count"].sum(), 1e-9)
    results.append(
        ModelResult(
            "ppml_raw_counts",
            raw_pred * scale,
            "same spec on honest person counts, rescaled",
        )
    )

    results.append(
        ModelResult(
            "boosted_poisson",
            fit_boosted_poisson(train, test),
            "same features, no linearity assumption",
        )
    )

    results.append(
        ModelResult(
            "flow_embedding",
            fit_flow_embedding(train, test),
            "no O*NET features at all; learns from flows only",
        )
    )

    _, hybrid_pred = fit_gravity_plus_history(train, test, max_iter=max_iter)
    results.append(
        ModelResult(
            "gravity_plus_history",
            hybrid_pred,
            "features AND lagged flow; a forecast, not a structural model",
        )
    )

    return results, ppml_fit

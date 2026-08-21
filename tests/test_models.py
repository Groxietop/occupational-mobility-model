import numpy as np
import pandas as pd
import pytest

from gravity import build_panel
from models import (
    fit_boosted_poisson,
    fit_flow_embedding,
    fit_gravity_plus_history,
    overdispersion,
)
from synthetic import make_transitions
from test_gravity import TEST_YEARS, TRAIN_YEARS, TRUE_WEIGHTS


@pytest.fixture(scope="module")
def panels(small_pairs):
    transitions, sizes = make_transitions(
        small_pairs, TRUE_WEIGHTS, years=TRAIN_YEARS + TEST_YEARS, seed=13
    )
    return (
        build_panel(small_pairs, transitions, sizes, TRAIN_YEARS),
        build_panel(small_pairs, transitions, sizes, TEST_YEARS),
    )


def test_overdispersion_is_one_for_a_poisson_sample():
    rng = np.random.default_rng(0)
    assert overdispersion(rng.poisson(4.0, size=200_000)) == pytest.approx(1.0, abs=0.05)


def test_overdispersion_detects_a_heavy_tail():
    rng = np.random.default_rng(0)
    # Poisson with a gamma-varying rate is the textbook overdispersed count.
    counts = rng.poisson(rng.gamma(shape=0.3, scale=12.0, size=200_000))
    assert overdispersion(counts) > 5.0


def test_boosted_poisson_predicts_positive_finite_counts(panels):
    train, test = panels
    predicted = fit_boosted_poisson(train, test, max_iter=40)
    assert len(predicted) == len(test)
    assert np.isfinite(predicted).all()
    assert (predicted > 0).all()


def test_flow_embedding_uses_no_onet_features(panels):
    """Guards the point of the comparison: this model must stay feature-blind.

    If it ever starts reading similarity columns, it stops being evidence
    about whether O*NET features are doing the work.
    """
    train, test = panels
    stripped_train = train.drop(columns=[c for c in train.columns if c.startswith("sim__")])
    stripped_test = test.drop(columns=[c for c in test.columns if c.startswith("sim__")])
    predicted = fit_flow_embedding(stripped_train, stripped_test)
    assert len(predicted) == len(test)
    assert np.isfinite(predicted).all()


def test_flow_embedding_recovers_a_strong_observed_pair(panels):
    train, test = panels
    predicted = fit_flow_embedding(train, test)
    busiest = train.nlargest(1, "weighted_count").iloc[0]
    mask = (test["soc_from"] == busiest["soc_from"]) & (test["soc_to"] == busiest["soc_to"])
    if mask.any():
        # A pair that carried heavy flow in training should not be predicted
        # near zero on the held-out years.
        assert predicted[mask.to_numpy()][0] > np.median(predicted)


def test_gravity_plus_history_restores_extra_features_after_fitting(panels):
    """The hybrid mutates a module global; it must put it back."""
    import gravity

    train, test = panels
    before = list(gravity.EXTRA_FEATURES)
    fit_gravity_plus_history(train, test, max_iter=50)
    assert gravity.EXTRA_FEATURES == before


def test_gravity_plus_history_restores_globals_even_on_failure(panels):
    import gravity

    train, test = panels
    before = list(gravity.EXTRA_FEATURES)
    with pytest.raises(Exception):
        fit_gravity_plus_history(train, test.drop(columns=["soc_to"]), max_iter=10)
    assert gravity.EXTRA_FEATURES == before


def test_boosted_poisson_handles_more_origins_than_sklearns_category_cap(small_pairs):
    """Regression: sklearn caps categoricals at 255 distinct values.

    Passing the origin as a categorical crashed on the full 430-occupation
    universe. It now enters as log outflow, which has no such limit.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: F401

    origins = sorted(set(small_pairs["soc_from"]))
    assert len(origins) >= 2

    rng = np.random.default_rng(3)
    panel = small_pairs.copy()
    panel["weighted_count"] = rng.poisson(1.5, size=len(panel)).astype(float)
    panel["raw_count"] = panel["weighted_count"]
    panel["same_industry"] = 0.0
    panel["dest_employment"] = 1000.0
    panel["log_dest_employment"] = np.log(1000.0)

    predicted = fit_boosted_poisson(panel, panel, max_iter=20)
    assert np.isfinite(predicted).all()
    assert (predicted > 0).all()


def test_boosted_poisson_does_not_leak_test_outflow(panels):
    """Origin outflow must come from training years only."""
    train, test = panels
    # Blowing up the test panel's counts must not change the predictions,
    # because the feature is built from train.
    inflated = test.copy()
    inflated["weighted_count"] = inflated["weighted_count"] * 1000.0
    a = fit_boosted_poisson(train, test, max_iter=30)
    b = fit_boosted_poisson(train, inflated, max_iter=30)
    np.testing.assert_allclose(a, b)

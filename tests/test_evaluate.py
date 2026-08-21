import numpy as np
import pandas as pd
import pytest

from evaluate import (
    baseline_predictions,
    compare,
    poisson_deviance,
    recall_at_k,
    spearman_on_observed,
)
from gravity import build_panel, fit_gravity, predict
from synthetic import make_transitions
from test_gravity import TEST_YEARS, TRAIN_YEARS, TRUE_WEIGHTS


def test_poisson_deviance_is_zero_for_a_perfect_fit():
    actual = np.array([0.0, 3.0, 7.0])
    assert poisson_deviance(actual, actual) == pytest.approx(0.0, abs=1e-9)


def test_poisson_deviance_penalises_a_worse_fit():
    actual = np.array([0.0, 3.0, 7.0])
    close = poisson_deviance(actual, np.array([0.1, 3.2, 6.5]))
    far = poisson_deviance(actual, np.array([5.0, 0.5, 1.0]))
    assert far > close > 0


def test_poisson_deviance_handles_zero_counts_without_nan():
    value = poisson_deviance(np.zeros(4), np.array([0.5, 1.0, 2.0, 0.1]))
    assert np.isfinite(value)


def test_recall_at_k_is_one_when_ranking_matches():
    panel = pd.DataFrame(
        {
            "soc_from": ["a", "a", "a", "b", "b", "b"],
            "weighted_count": [10.0, 5.0, 0.0, 8.0, 1.0, 0.0],
        }
    )
    perfect = np.array([10.0, 5.0, 0.1, 8.0, 1.0, 0.1])
    assert recall_at_k(panel, perfect, k=2) == pytest.approx(1.0)


def test_recall_at_k_is_zero_when_ranking_is_inverted():
    panel = pd.DataFrame(
        {"soc_from": ["a", "a"], "weighted_count": [10.0, 0.0]}
    )
    assert recall_at_k(panel, np.array([0.1, 99.0]), k=1) == pytest.approx(0.0)


def test_recall_ignores_origins_with_no_observed_moves():
    panel = pd.DataFrame(
        {
            "soc_from": ["a", "a", "quiet", "quiet"],
            "weighted_count": [4.0, 0.0, 0.0, 0.0],
        }
    )
    # 'quiet' contributes nothing rather than counting as a miss.
    assert recall_at_k(panel, np.array([4.0, 0.1, 1.0, 2.0]), k=1) == pytest.approx(1.0)


def test_spearman_is_nan_without_enough_observed_pairs():
    panel = pd.DataFrame({"soc_from": ["a", "a"], "weighted_count": [1.0, 0.0]})
    assert np.isnan(spearman_on_observed(panel, np.array([1.0, 2.0])))


def test_baselines_conserve_each_origins_outflow():
    """Baselines must predict the same total mass as was observed per origin.

    Otherwise a deviance comparison against the fitted model is measuring
    scale, not ranking quality.
    """
    panel = pd.DataFrame(
        {
            "soc_from": ["a", "a", "a"],
            "soc_to": ["x", "y", "z"],
            "weighted_count": [6.0, 3.0, 1.0],
            "dest_employment": [100.0, 50.0, 25.0],
            "onet_related": [1.0, 0.0, 0.0],
            "sim__skill": [0.5, 0.1, -0.2],
            "sim__ability": [0.4, 0.2, -0.1],
        }
    )
    for name, values in baseline_predictions(panel).items():
        assert values.sum() == pytest.approx(10.0), name


def test_scorecard_reports_every_baseline_and_the_model(small_pairs):
    transitions, sizes = make_transitions(
        small_pairs, TRUE_WEIGHTS, years=TRAIN_YEARS + TEST_YEARS, seed=3
    )
    train = build_panel(small_pairs, transitions, sizes, TRAIN_YEARS)
    test = build_panel(small_pairs, transitions, sizes, TEST_YEARS)

    fit = fit_gravity(train, TRAIN_YEARS, TEST_YEARS)
    table = compare(test, predict(fit, test))

    assert list(table["model"]) == [
        "size_only",
        "equal_similarity",
        "onet_related",
        "learned_gravity",
    ]
    assert table["poisson_deviance"].notna().all()


def test_learned_model_beats_the_base_rate_baseline(small_pairs):
    """The headline claim, checked on held-out years of a known process."""
    transitions, sizes = make_transitions(
        small_pairs, TRUE_WEIGHTS, years=TRAIN_YEARS + TEST_YEARS, seed=11
    )
    train = build_panel(small_pairs, transitions, sizes, TRAIN_YEARS)
    test = build_panel(small_pairs, transitions, sizes, TEST_YEARS)

    fit = fit_gravity(train, TRAIN_YEARS, TEST_YEARS)
    table = compare(test, predict(fit, test)).set_index("model")

    learned = table.loc["learned_gravity", "poisson_deviance"]
    size_only = table.loc["size_only", "poisson_deviance"]
    equal = table.loc["equal_similarity", "poisson_deviance"]

    assert learned < size_only
    assert learned < equal

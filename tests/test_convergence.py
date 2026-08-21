import pytest

from gravity import build_panel, fit_gravity
from synthetic import make_transitions
from test_gravity import TEST_YEARS, TRAIN_YEARS, TRUE_WEIGHTS


@pytest.fixture(scope="module")
def panel(small_pairs):
    transitions, sizes = make_transitions(
        small_pairs, TRUE_WEIGHTS, years=TRAIN_YEARS, seed=5
    )
    return build_panel(small_pairs, transitions, sizes, TRAIN_YEARS)


def test_reports_convergence_on_a_normal_fit(panel):
    fit = fit_gravity(panel, TRAIN_YEARS, TEST_YEARS, max_iter=1000)
    assert fit.converged
    assert 0 < fit.n_iter <= 1000


def test_flags_a_run_that_hit_the_iteration_cap(panel):
    """A truncated solve must be visible, not a warning nobody reads."""
    fit = fit_gravity(panel, TRAIN_YEARS, TEST_YEARS, max_iter=2)
    assert not fit.converged


def test_synthetic_sparsity_resembles_a_real_cps_extract(panel):
    density = (panel["weighted_count"] > 0).mean()
    # Real CPS ASEC over ~430 occupations leaves the overwhelming majority of
    # ordered pairs empty. If the fixture ever drifts dense, the estimator is
    # being tested on an easier problem than the one it has to solve.
    assert 0.02 < density < 0.25

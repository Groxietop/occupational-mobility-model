import numpy as np
import pytest

from gravity import build_panel, fit_gravity, predict
from synthetic import make_transitions

TRAIN_YEARS = [2018, 2019, 2020, 2021]
TEST_YEARS = [2022, 2023]

# The ground truth the synthetic flows are generated from: skill and knowledge
# drive moves, work values and interests do not.
TRUE_WEIGHTS = {
    "skill": 2.0,
    "knowledge": 1.2,
    "work_activity": 0.6,
    "ability": 0.3,
    "work_style": 0.0,
    "work_value": 0.0,
    "interest": 0.0,
    "work_context": 0.0,
}


@pytest.fixture(scope="module")
def synthetic_panels(small_pairs):
    transitions, sizes = make_transitions(
        small_pairs, TRUE_WEIGHTS, years=TRAIN_YEARS + TEST_YEARS, seed=7
    )
    train = build_panel(small_pairs, transitions, sizes, TRAIN_YEARS)
    test = build_panel(small_pairs, transitions, sizes, TEST_YEARS)
    return train, test


def test_panel_keeps_the_zero_flow_pairs(synthetic_panels, small_universe):
    train, _ = synthetic_panels
    # Every ordered pair with a live origin survives, zeros included -- PPML
    # needs them and dropping them is the classic way to break this model.
    assert (train["weighted_count"] == 0).sum() > 0
    assert train["weighted_count"].min() == 0.0
    assert len(train) > len(small_universe)


def test_panel_attaches_destination_size(synthetic_panels):
    train, _ = synthetic_panels
    assert train["dest_employment"].gt(0).all()
    assert np.isfinite(train["log_dest_employment"]).all()
    # Size is a property of the destination, so it must not vary by origin.
    per_dest = train.groupby("soc_to")["dest_employment"].nunique()
    assert (per_dest == 1).all()


def test_recovers_the_domain_weight_ordering(synthetic_panels):
    """The estimator has to find weights it was never told, from flows alone."""
    train, _ = synthetic_panels
    fit = fit_gravity(train, TRAIN_YEARS, TEST_YEARS)
    weights = fit.domain_weights().set_index("domain")["coefficient"]

    # The two domains that genuinely drive the synthetic flows must come out
    # on top of the four that carry no signal at all.
    signal = {"skill", "knowledge"}
    noise = {"work_style", "work_value", "interest", "work_context"}
    assert weights[list(signal)].min() > weights[list(noise)].max()
    # And the strongest true driver should rank first.
    assert weights.idxmax() == "skill"


def test_learns_a_positive_size_coefficient(synthetic_panels):
    train, _ = synthetic_panels
    fit = fit_gravity(train, TRAIN_YEARS, TEST_YEARS)
    size = fit.coefficients.set_index("feature")["coefficient"]["log_dest_employment"]
    assert size > 0


def test_predictions_are_positive_and_finite_on_held_out_years(synthetic_panels):
    train, test = synthetic_panels
    fit = fit_gravity(train, TRAIN_YEARS, TEST_YEARS)
    predicted = predict(fit, test)
    assert len(predicted) == len(test)
    assert np.isfinite(predicted).all()
    assert (predicted > 0).all()


def test_missing_deltas_are_imputed_and_flagged(small_pairs):
    transitions, sizes = make_transitions(
        small_pairs, TRUE_WEIGHTS, years=TRAIN_YEARS, seed=1
    )
    panel = build_panel(small_pairs, transitions, sizes, TRAIN_YEARS)
    assert panel["d_education"].isna().any()  # the real O*NET data has gaps

    fit = fit_gravity(panel, TRAIN_YEARS, TEST_YEARS)
    assert "d_education__missing" in fit.feature_names
    # Imputation must not leave NaNs anywhere in the fitted coefficients.
    assert np.isfinite(fit.coefficients["coefficient"]).all()


def test_domain_weight_shares_sum_to_one(synthetic_panels):
    train, _ = synthetic_panels
    fit = fit_gravity(train, TRAIN_YEARS, TEST_YEARS)
    shares = fit.domain_weights()["share"].dropna()
    assert shares.sum() == pytest.approx(1.0)

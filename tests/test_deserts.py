import numpy as np
import pandas as pd
import pytest

from deserts import (
    destination_profile,
    escape_metrics,
    find_deserts,
    low_paid_and_trapped,
)


def _panel(origins, destinations):
    rows = [(o, d) for o in origins for d in destinations if o != d]
    return pd.DataFrame(rows, columns=["soc_from", "soc_to"])


def test_effective_destinations_equals_n_for_a_uniform_spread():
    """A perfectly even spread over k destinations should read as k options."""
    panel = _panel(["a"], ["a", "x", "y", "z", "w"])
    predicted = np.ones(len(panel))
    profile = destination_profile(panel, predicted)
    assert profile["effective_destinations"].iloc[0] == pytest.approx(4.0)


def test_effective_destinations_collapses_to_one_when_concentrated():
    panel = _panel(["a"], ["a", "x", "y", "z"])
    predicted = np.array([1000.0, 1e-9, 1e-9])
    profile = destination_profile(panel, predicted)
    assert profile["effective_destinations"].iloc[0] < 1.05
    assert profile["top_share"].iloc[0] > 0.99


def test_effective_destinations_is_between_one_and_the_option_count():
    rng = np.random.default_rng(0)
    panel = _panel(["a"], ["a"] + [f"d{i}" for i in range(20)])
    profile = destination_profile(panel, rng.random(len(panel)))
    value = profile["effective_destinations"].iloc[0]
    assert 1.0 <= value <= 20.0


def test_origins_with_no_predicted_outflow_are_dropped():
    panel = _panel(["a", "quiet"], ["a", "quiet", "x"])
    predicted = np.where(panel["soc_from"] == "quiet", 0.0, 1.0)
    profile = destination_profile(panel, predicted)
    assert set(profile["soc6"]) == {"a"}


def test_upward_share_counts_only_meaningful_raises():
    panel = pd.DataFrame(
        {"soc_from": ["a", "a", "a"], "soc_to": ["hi", "same", "lo"]}
    )
    wages = pd.DataFrame(
        {
            "soc6": ["a", "hi", "same", "lo"],
            "annual_median_wage": [50_000.0, 80_000.0, 51_000.0, 30_000.0],
        }
    )
    # Equal flow to all three. Only 'hi' clears the +10% bar; 'same' is +2%.
    out = escape_metrics(panel, np.ones(3), wages, raise_threshold=0.10)
    assert out["upward_share"].iloc[0] == pytest.approx(1 / 3)


def test_expected_wage_ratio_is_relative_to_the_origin():
    panel = pd.DataFrame({"soc_from": ["a", "a"], "soc_to": ["x", "y"]})
    wages = pd.DataFrame(
        {"soc6": ["a", "x", "y"], "annual_median_wage": [50_000.0, 100_000.0, 50_000.0]}
    )
    out = escape_metrics(panel, np.ones(2), wages)
    # Even flow to a doubling and a lateral move -> 1.5x expected.
    assert out["expected_wage_ratio"].iloc[0] == pytest.approx(1.5)


def test_find_deserts_ranks_the_narrow_and_stuck_first():
    panel = _panel(["narrow", "broad"], ["narrow", "broad", "x", "y", "z"])
    # 'narrow' dumps everything into x; 'broad' spreads evenly.
    predicted = np.array(
        [1000.0 if (r.soc_from == "narrow" and r.soc_to == "x") else 1.0
         for r in panel.itertuples()]
    )
    wages = pd.DataFrame(
        {
            "soc6": ["narrow", "broad", "x", "y", "z"],
            "annual_median_wage": [40_000.0, 40_000.0, 39_000.0, 90_000.0, 90_000.0],
            "employment": [1000.0] * 5,
        }
    )
    titles = pd.DataFrame(
        {"soc6": ["narrow", "broad", "x", "y", "z"], "title": list("nbxyz")}
    )
    out = find_deserts(panel, predicted, wages, titles)
    assert out.iloc[0]["soc6"] == "narrow"
    assert out.set_index("soc6").loc["narrow", "effective_destinations"] < 2.0


def test_low_paid_and_trapped_excludes_well_paid_specialists():
    deserts = pd.DataFrame(
        {
            "soc6": ["poor", "rich"],
            "title": ["Poorly paid, few exits", "Well paid specialist"],
            "effective_destinations": [3.0, 3.0],
            "annual_median_wage": [30_000.0, 250_000.0],
            "is_narrow": [True, True],
            "desert_score": [0.9, 0.9],
        }
    )
    out = low_paid_and_trapped(deserts, wage_percentile=0.5)
    assert list(out["soc6"]) == ["poor"]

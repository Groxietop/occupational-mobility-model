import numpy as np
import pandas as pd
import pytest

from pairs import build_pair_features, related_pairs, similarity_columns


def _toy_master(n=5):
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "soc6": [f"11-10{i}1" for i in range(n)],
            "title": [f"Job {i}" for i in range(n)],
            "job_zone": np.arange(n, dtype=float),
            "education_rank": np.arange(n, dtype=float) + 2,
            "skill__a": rng.normal(size=n),
            "skill__b": rng.normal(size=n),
            "ability__a": rng.normal(size=n),
        }
    )


def test_pair_table_is_every_ordered_pair_without_self_pairs():
    pairs = build_pair_features(_toy_master(5))
    assert len(pairs) == 5 * 4
    assert (pairs["soc_from"] != pairs["soc_to"]).all()
    # Ordered, not unordered: both directions are present.
    keys = set(zip(pairs["soc_from"], pairs["soc_to"]))
    assert ("11-1001", "11-1011") in keys
    assert ("11-1011", "11-1001") in keys


def test_deltas_are_signed_and_directional():
    pairs = build_pair_features(_toy_master(3))
    forward = pairs[(pairs["soc_from"] == "11-1001") & (pairs["soc_to"] == "11-1021")]
    backward = pairs[(pairs["soc_from"] == "11-1021") & (pairs["soc_to"] == "11-1001")]
    assert forward["d_job_zone"].iloc[0] == pytest.approx(2.0)
    assert backward["d_job_zone"].iloc[0] == pytest.approx(-2.0)


def test_similarity_is_symmetric_across_direction():
    pairs = build_pair_features(_toy_master(4))
    sim_col = similarity_columns(pairs)[0]
    forward = pairs.set_index(["soc_from", "soc_to"])[sim_col]
    a, b = "11-1001", "11-1021"
    assert forward.loc[(a, b)] == pytest.approx(forward.loc[(b, a)])


def test_constant_domain_is_dropped_rather_than_dividing_by_zero():
    master = _toy_master(4)
    master["knowledge__flat"] = 1.0  # zero variance
    pairs = build_pair_features(master)
    assert "sim__knowledge" not in pairs.columns
    assert "sim__skill" in pairs.columns


def test_universe_restricts_the_occupation_set():
    pairs = build_pair_features(_toy_master(6), universe=["11-1001", "11-1011"])
    assert len(pairs) == 2
    assert set(pairs["soc_from"]) == {"11-1001", "11-1011"}


def test_needs_at_least_two_occupations():
    with pytest.raises(ValueError, match="at least 2"):
        build_pair_features(_toy_master(6), universe=["11-1001"])


def test_related_pairs_parses_the_onet_list():
    master = pd.DataFrame(
        {
            "soc_code": ["11-1011.00", "15-1252.00"],
            "primary_related_soc_codes": ["11-1021.00, 13-1111.00", None],
        }
    )
    assert related_pairs(master) == {("11-1011", "11-1021"), ("11-1011", "13-1111")}


def test_onet_related_flag_lands_on_the_right_pairs():
    pairs = build_pair_features(
        _toy_master(4), onet_related={("11-1001", "11-1021")}
    )
    flagged = pairs[pairs["onet_related"] == 1.0]
    assert list(zip(flagged["soc_from"], flagged["soc_to"])) == [("11-1001", "11-1021")]


def test_real_master_builds_a_full_pair_table(small_pairs, small_universe):
    n = len(small_universe)
    assert len(small_pairs) == n * (n - 1)
    assert len(similarity_columns(small_pairs)) == 8
    sims = small_pairs[similarity_columns(small_pairs)]
    assert sims.notna().all().all()
    assert sims.to_numpy().min() >= -1.0001
    assert sims.to_numpy().max() <= 1.0001

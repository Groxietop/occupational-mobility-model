import numpy as np
import pandas as pd
import pytest

from pairs import add_wage_features, build_pair_features
from sources import bls
from sources.ipums import CORE_VARIABLES, ExtractHandle, build_extract, sample_id
from test_pairs import _toy_master


# --- BLS series ID construction -------------------------------------------
# These are pure string assembly and worth locking down: a malformed OEWS id
# does not error, it returns REQUEST_SUCCEEDED with zero data points, so a
# silent typo looks exactly like "this occupation has no data".


def test_series_id_matches_the_documented_example():
    # Software Developers, national, all industries, annual median wage.
    assert bls.series_id("15-1252", "annual_median_wage") == "OEUN000000000000015125213"


def test_series_id_is_always_25_characters():
    for measure in bls.MEASURES:
        assert len(bls.series_id("29-1141", measure)) == 25


def test_series_id_encodes_the_measure_in_the_last_two_characters():
    assert bls.series_id("15-1252", "employment").endswith("01")
    assert bls.series_id("15-1252", "annual_mean_wage").endswith("04")
    assert bls.series_id("15-1252", "annual_median_wage").endswith("13")


def test_soc_code_loses_its_hyphen():
    assert bls.soc_to_occupation_code("15-1252") == "151252"


def test_rejects_a_malformed_soc_code():
    with pytest.raises(ValueError, match="6-digit SOC"):
        bls.series_id("15-12", "employment")


def test_rejects_an_unknown_measure():
    with pytest.raises(ValueError, match="unknown measure"):
        bls.series_id("15-1252", "hourly_tips")


def test_fetch_requires_a_key(monkeypatch):
    monkeypatch.delenv("BLS_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="BLS_API_KEY"):
        bls.fetch_oews(["15-1252"], api_key=None)


def test_to_wide_pivots_measures_into_columns():
    tidy = pd.DataFrame(
        {
            "soc6": ["15-1252", "15-1252", "29-1141"],
            "measure": ["employment", "annual_median_wage", "employment"],
            "value": [1687890.0, 135980.0, 3379720.0],
            "year": [2025, 2025, 2025],
        }
    )
    wide = bls.to_wide(tidy)
    row = wide[wide["soc6"] == "15-1252"].iloc[0]
    assert row["employment"] == 1687890.0
    assert row["annual_median_wage"] == 135980.0


def test_to_wide_on_empty_input_returns_the_expected_columns():
    wide = bls.to_wide(pd.DataFrame(columns=["soc6", "measure", "value", "year"]))
    assert "annual_median_wage" in wide.columns
    assert len(wide) == 0


# --- wage features ---------------------------------------------------------


def test_wage_gap_is_signed_and_logged():
    pairs = build_pair_features(_toy_master(3))
    oews = pd.DataFrame(
        {
            "soc6": ["11-1001", "11-1011", "11-1021"],
            "year": [2025, 2025, 2025],
            "annual_median_wage": [50_000.0, 100_000.0, 25_000.0],
        }
    )
    out = add_wage_features(pairs, oews)
    keyed = out.set_index(["soc_from", "soc_to"])["d_log_wage"]

    # Doubling the wage is +log(2); halving it is the exact negative.
    assert keyed.loc[("11-1001", "11-1011")] == pytest.approx(np.log(2))
    assert keyed.loc[("11-1011", "11-1001")] == pytest.approx(-np.log(2))


def test_missing_wages_stay_nan_rather_than_becoming_a_zero_gap():
    """OEWS suppresses small occupations; that is not a zero wage change."""
    pairs = build_pair_features(_toy_master(3))
    oews = pd.DataFrame(
        {"soc6": ["11-1001"], "year": [2025], "annual_median_wage": [50_000.0]}
    )
    out = add_wage_features(pairs, oews)
    assert out["d_log_wage"].isna().any()


def test_nonpositive_wages_are_dropped_before_taking_logs():
    pairs = build_pair_features(_toy_master(3))
    oews = pd.DataFrame(
        {
            "soc6": ["11-1001", "11-1011", "11-1021"],
            "year": [2025] * 3,
            "annual_median_wage": [50_000.0, 0.0, -1.0],
        }
    )
    out = add_wage_features(pairs, oews)
    assert np.isfinite(out["d_log_wage"].dropna()).all()


# --- IPUMS extract definition ---------------------------------------------


def test_sample_id_uses_the_asec_supplement_code():
    # Confirmed against the live API's own sample listing.
    assert sample_id(2024) == "cps2024_03s"


def test_extract_requests_the_asec_weight_not_just_the_monthly_one():
    """ASECWT is the correct weight for the March supplement.

    The original notebook weighted with WTFINL, the basic monthly weight,
    which is the wrong weight for ASEC-derived transitions.
    """
    assert "ASECWT" in CORE_VARIABLES
    extract = build_extract(years=[2023, 2024])
    names = {getattr(v, "name", v) for v in extract.variables}
    assert "ASECWT" in names
    assert "OCCLY" in names  # the transition itself


def test_extract_covers_the_requested_years():
    extract = build_extract(years=[2018, 2019, 2020])
    ids = {getattr(s, "id", s) for s in extract.samples}
    assert ids == {"cps2018_03s", "cps2019_03s", "cps2020_03s"}


def test_extract_handle_round_trips_through_disk(tmp_path):
    ExtractHandle(number=12345).save(tmp_path)
    restored = ExtractHandle.load(tmp_path)
    assert restored.number == 12345
    assert restored.collection == "cps"


def test_extract_handle_is_none_when_nothing_was_submitted(tmp_path):
    assert ExtractHandle.load(tmp_path) is None


# --- transitions weighting -------------------------------------------------


def test_prefers_the_asec_weight_when_both_are_present():
    from transitions import resolve_weight

    frame = pd.DataFrame({"asecwt": [1.0], "wtfinl": [2.0]})
    assert resolve_weight(frame) == "asecwt"


def test_falls_back_to_the_monthly_weight_for_older_extracts():
    from transitions import resolve_weight

    assert resolve_weight(pd.DataFrame({"wtfinl": [2.0]})) == "wtfinl"


def test_raises_when_no_weight_column_exists():
    from transitions import resolve_weight

    with pytest.raises(KeyError, match="no person weight"):
        resolve_weight(pd.DataFrame({"year": [2020]}))


def test_moves_carry_the_resolved_weight(tmp_path):
    """End-to-end: ASECWT should drive the aggregated counts, not WTFINL."""
    from transitions import aggregate_transitions, extract_moves

    crosswalk = tmp_path / "xwalk.csv"
    crosswalk.write_text(
        "census_occ_code,census_title,soc_code_6digit\n"
        "0010,Chief executives,11-1011\n"
        "0020,General managers,11-1021\n"
    )
    raw = pd.DataFrame(
        {
            "year": [2020, 2020],
            "empstat": [10, 10],
            "occ": [20, 20],
            "occly": [10, 10],
            "ind": [1, 1],
            "indly": [1, 2],
            "asecwt": [100.0, 200.0],
            "wtfinl": [1.0, 1.0],
        }
    )
    moves = extract_moves(raw, crosswalk)
    agg = aggregate_transitions(moves)

    assert len(agg) == 1
    # 100 + 200 from ASECWT, not 1 + 1 from WTFINL.
    assert agg["weighted_count"].iloc[0] == pytest.approx(300.0)
    assert agg["raw_count"].iloc[0] == 2
    assert agg["same_industry_share"].iloc[0] == pytest.approx(0.5)


# --- collinearity diagnostics ---------------------------------------------


def test_vif_is_one_for_orthogonal_features():
    from diagnostics import collinearity

    rng = np.random.default_rng(0)
    n = 4000
    pairs = pd.DataFrame(
        {"sim__a": rng.normal(size=n), "sim__b": rng.normal(size=n)}
    )
    frame = collinearity(pairs)
    assert frame["vif"].max() < 1.1
    assert frame["individually_readable"].all()


def test_vif_explodes_for_a_near_duplicate_feature():
    from diagnostics import collinearity

    rng = np.random.default_rng(0)
    base = rng.normal(size=4000)
    pairs = pd.DataFrame(
        {
            "sim__a": base,
            "sim__b": base + rng.normal(scale=0.01, size=4000),
            "sim__c": rng.normal(size=4000),
        }
    )
    frame = collinearity(pairs).set_index("domain")
    assert frame.loc["a", "vif"] > 50
    assert not frame.loc["a", "individually_readable"]
    assert frame.loc["c", "individually_readable"]


def test_real_similarity_domains_are_individually_readable(small_pairs):
    """Guards the headline claim: the per-domain weights can be read at all."""
    from diagnostics import collinearity, condition_number

    frame = collinearity(small_pairs)
    assert frame["individually_readable"].all(), frame.to_string()
    assert condition_number(small_pairs) < 100


def test_suppression_is_flagged_when_a_sign_flips():
    from diagnostics import suppressed_domains

    joint = pd.DataFrame(
        {"domain": ["ability", "skill"], "coefficient": [-0.14, 0.30], "share": [0.0, 0.5]}
    )
    out = suppressed_domains(joint, {"ability": 0.88, "skill": 1.06}).set_index("domain")
    assert bool(out.loc["ability", "sign_flipped"]) is True
    assert bool(out.loc["skill", "sign_flipped"]) is False

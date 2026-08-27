"""Tests for the O*NET client.

Network calls are not exercised here -- the version comparison and local
version parsing are the parts with actual logic, and they run offline.
"""

import pytest

from sources.onet import (
    DOMAIN_PATHS,
    Freshness,
    ONetError,
    domain_ratings,
    get_key,
    local_version,
)


# --- key handling ----------------------------------------------------------


def test_missing_key_explains_where_to_get_one(monkeypatch):
    monkeypatch.delenv("ONET_API_KEY", raising=False)
    with pytest.raises(ONetError, match="developer/signup"):
        get_key()


def test_explicit_key_beats_the_environment(monkeypatch):
    monkeypatch.setenv("ONET_API_KEY", "from-env")
    assert get_key("explicit") == "explicit"


# --- local version detection ----------------------------------------------


def test_reads_the_version_from_the_bulk_directory_name(tmp_path):
    (tmp_path / "onet_30_2").mkdir()
    assert local_version(tmp_path) == "30.2"


def test_handles_a_two_digit_minor_version(tmp_path):
    (tmp_path / "onet_31_10").mkdir()
    assert local_version(tmp_path) == "31.10"


def test_ignores_unrelated_directories(tmp_path):
    (tmp_path / "census_stuff").mkdir()
    (tmp_path / "onet_29_1").mkdir()
    assert local_version(tmp_path) == "29.1"


def test_returns_none_when_no_bulk_database_is_present(tmp_path):
    assert local_version(tmp_path) is None


def test_returns_none_for_a_missing_directory(tmp_path):
    assert local_version(tmp_path / "nope") is None


def test_real_repo_reports_the_version_the_master_was_built_from():
    assert local_version("data/raw", master_path="data/processed/onet_master.parquet") == "31.0"


# --- freshness comparison --------------------------------------------------


def test_matching_versions_are_current():
    freshness = Freshness(local_version="31.0", published_version="31.0")
    assert freshness.is_current
    assert "is current" in freshness.describe()


def test_behind_versions_say_what_to_do():
    freshness = Freshness(local_version="30.2", published_version="31.0")
    assert not freshness.is_current
    message = freshness.describe()
    assert "30.2" in message and "31.0" in message
    assert "database.html" in message  # points at the fix


def test_unknown_published_version_is_not_reported_as_stale():
    """A network failure must not masquerade as 'your data is old'."""
    freshness = Freshness(local_version="30.2", published_version=None)
    assert not freshness.known
    assert not freshness.is_current
    assert "could not compare" in freshness.describe()


def test_unknown_local_version_is_also_not_a_comparison():
    freshness = Freshness(local_version=None, published_version="31.0")
    assert not freshness.known
    assert "could not compare" in freshness.describe()


# --- domain coverage -------------------------------------------------------


def test_every_scoring_domain_has_an_api_path():
    """The eight domains the similarity model uses must all be reachable."""
    from similarity import DOMAINS

    assert set(DOMAIN_PATHS) == set(DOMAINS)


def test_unknown_domain_is_rejected_before_any_request():
    with pytest.raises(ValueError, match="unknown domain"):
        domain_ratings("15-1252.00", "vibes", api_key="unused")


# --- version provenance ----------------------------------------------------


def test_master_stamp_beats_a_stale_raw_directory(tmp_path):
    """Regression: the check reported a leftover folder, not the real source.

    Once the raw dump stopped being committed, data/raw/ still held an old
    onet_30_2/ while the master had been rebuilt from 31.0 -- and the
    freshness check confidently reported 30.2.
    """
    import pandas as pd

    from sources.onet import local_version

    raw = tmp_path / "raw"
    (raw / "onet_30_2").mkdir(parents=True)
    master = tmp_path / "master.parquet"
    pd.DataFrame({"soc_code": ["11-1011.00"], "onet_version": ["31.0"]}).to_parquet(master)

    assert local_version(raw) == "30.2"                       # folder alone
    assert local_version(raw, master_path=master) == "31.0"   # stamp wins


def test_falls_back_to_the_directory_for_an_unstamped_master(tmp_path):
    import pandas as pd

    from sources.onet import local_version

    raw = tmp_path / "raw"
    (raw / "onet_30_2").mkdir(parents=True)
    master = tmp_path / "master.parquet"
    pd.DataFrame({"soc_code": ["11-1011.00"]}).to_parquet(master)

    assert local_version(raw, master_path=master) == "30.2"


def test_missing_master_does_not_break_the_check(tmp_path):
    from sources.onet import local_version

    raw = tmp_path / "raw"
    (raw / "onet_31_0").mkdir(parents=True)
    assert local_version(raw, master_path=tmp_path / "nope.parquet") == "31.0"


def test_committed_master_reports_31_0():
    """The real artifact must know which release it came from."""
    from sources.onet import version_from_master

    assert version_from_master("data/processed/onet_master.parquet") == "31.0"

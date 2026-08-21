import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

MASTER = ROOT / "data" / "processed" / "onet_master.parquet"
CROSSWALK = ROOT / "data" / "raw" / "census_occ_to_soc_2018.csv"


@pytest.fixture(scope="session")
def onet_master() -> pd.DataFrame:
    if not MASTER.exists():
        pytest.skip(f"{MASTER} not present")
    return pd.read_parquet(MASTER)


@pytest.fixture(scope="session")
def crosswalk_path() -> Path:
    if not CROSSWALK.exists():
        pytest.skip(f"{CROSSWALK} not present")
    return CROSSWALK


@pytest.fixture(scope="session")
def soc6_master(onet_master):
    from soc import collapse_to_soc6

    return collapse_to_soc6(onet_master)


@pytest.fixture(scope="session")
def small_universe(soc6_master, crosswalk_path):
    """A 60-occupation slice, big enough to be realistic and fast to fit."""
    from soc import load_crosswalk

    reachable = set(load_crosswalk(crosswalk_path)["soc6"])
    codes = sorted(set(soc6_master["soc6"]) & reachable)
    return codes[:60]


@pytest.fixture(scope="session")
def small_pairs(soc6_master, small_universe, onet_master):
    from pairs import build_pair_features, related_pairs

    return build_pair_features(
        soc6_master, universe=small_universe, onet_related=related_pairs(onet_master)
    )

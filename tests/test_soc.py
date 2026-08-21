import pandas as pd

from soc import collapse_to_soc6, rank_education, to_soc6


def test_to_soc6_truncates_onet_detail_codes():
    assert to_soc6("15-1252.00") == "15-1252"
    assert to_soc6("15-1252.01") == "15-1252"
    assert to_soc6("15-1252") == "15-1252"


def test_rank_education_handles_long_form_labels():
    # O*NET spells several levels out with a trailing definition; the rank
    # has to survive that.
    short = rank_education("Bachelor's Degree")
    long_form = rank_education(
        "Post-Secondary Certificate - awarded for training completed after "
        "high school (for example, in agriculture or natural resources)"
    )
    assert short == 5.0
    assert long_form == 3.0
    assert rank_education(None) is None
    assert rank_education("Not a real level") is None


def test_education_ranks_are_ordered():
    assert rank_education("Less than a High School Diploma") < rank_education(
        "Bachelor's Degree"
    ) < rank_education("Doctoral Degree")


def test_collapse_averages_detail_codes_into_one_soc6_row():
    master = pd.DataFrame(
        {
            "soc_code": ["11-1011.00", "11-1011.03", "15-1252.00"],
            "title": ["Chief Executives", "Chief Sustainability Officers", "Devs"],
            "job_zone": [5.0, 5.0, 4.0],
            "required_education_level": ["Master's Degree", "Master's Degree", None],
            "skill__a": [1.0, 3.0, 5.0],
            "ability__b": [2.0, 4.0, 6.0],
        }
    )
    out = collapse_to_soc6(master)

    assert sorted(out["soc6"]) == ["11-1011", "15-1252"]
    ceo = out[out["soc6"] == "11-1011"].iloc[0]
    assert ceo["skill__a"] == 2.0  # mean of 1 and 3
    assert ceo["ability__b"] == 3.0
    assert ceo["education_rank"] == 6.0
    # The base '.00' title wins, not whichever row happened to sort first.
    assert ceo["title"] == "Chief Executives"


def test_collapse_on_real_master_covers_every_occupation(soc6_master, onet_master):
    assert len(soc6_master) < len(onet_master)  # detail codes were collapsed
    assert soc6_master["soc6"].is_unique
    assert soc6_master["soc6"].str.len().eq(7).all()
